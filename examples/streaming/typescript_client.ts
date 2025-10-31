/**
 * TypeScript WebSocket Client for Engram Streaming API
 *
 * Browser-compatible client demonstrating:
 * - WebSocket connection with auto-reconnect
 * - Stream initialization and session management
 * - Observation streaming with sequence tracking
 * - Flow control (pause/resume)
 * - Heartbeat handling
 * - Error recovery
 *
 * Usage:
 * ```typescript
 * const client = new EngramStreamingClient('ws://localhost:8080/v1/stream');
 * await client.connect('default');
 *
 * // Stream observations
 * const embedding = new Array(768).fill(0).map(() => Math.random());
 * await client.observe('episode_001', 'User clicked button', embedding, 0.85);
 *
 * // Flow control
 * await client.pause();
 * await client.resume();
 *
 * // Cleanup
 * await client.close();
 * ```
 */

/**
 * Stream message types matching the JSON protocol.
 */
type MessageType = 'init' | 'init_ack' | 'observation' | 'ack' | 'flow_control' | 'heartbeat' | 'close' | 'error';

interface InitRequest {
  type: 'init';
  memory_space_id: string;
  client_buffer_size?: number;
  enable_backpressure?: boolean;
}

interface InitAckResponse {
  type: 'init_ack';
  session_id: string;
  initial_sequence: number;
  capabilities: {
    max_observations_per_second: number;
    queue_capacity: number;
    supports_backpressure: boolean;
    supports_snapshot_isolation: boolean;
  };
}

interface ObservationRequest {
  type: 'observation';
  session_id: string;
  sequence_number: number;
  episode: {
    id: string;
    when?: string; // ISO 8601 timestamp
    what: string;
    embedding: number[]; // Must be 768 dimensions
    encoding_confidence?: number;
  };
}

interface AckResponse {
  type: 'ack';
  session_id: string;
  sequence_number: number;
  status: 'accepted' | 'indexed' | 'rejected';
  memory_id: string;
  committed_at: string;
}

interface FlowControlRequest {
  type: 'flow_control';
  session_id: string;
  action: 'pause' | 'resume';
}

interface HeartbeatMessage {
  type: 'heartbeat';
  session_id: string;
  timestamp: string;
}

interface ErrorResponse {
  type: 'error';
  message: string;
  session_id?: string;
}

type ServerMessage = InitAckResponse | AckResponse | HeartbeatMessage | ErrorResponse;

/**
 * Client configuration options.
 */
interface ClientConfig {
  /** WebSocket URL (default: ws://localhost:8080/v1/stream) */
  url: string;
  /** Client buffer size hint (default: 1000) */
  bufferSize?: number;
  /** Enable backpressure flow control (default: true) */
  enableBackpressure?: boolean;
  /** Auto-reconnect on disconnect (default: true) */
  autoReconnect?: boolean;
  /** Reconnect delay in ms (default: 1000) */
  reconnectDelay?: number;
  /** Max reconnect attempts (default: 5) */
  maxReconnectAttempts?: number;
}

/**
 * Engram WebSocket streaming client.
 *
 * Handles connection lifecycle, message serialization, and automatic reconnection.
 */
export class EngramStreamingClient {
  private ws: WebSocket | null = null;
  private config: Required<ClientConfig>;
  private sessionId: string | null = null;
  private sequence: number = 0;
  private memorySpaceId: string = 'default';
  private reconnectAttempts: number = 0;
  private messageHandlers: Map<MessageType, ((msg: any) => void)[]> = new Map();
  private pendingInit: Promise<void> | null = null;

  constructor(urlOrConfig: string | ClientConfig) {
    const config = typeof urlOrConfig === 'string' ? { url: urlOrConfig } : urlOrConfig;

    this.config = {
      url: config.url,
      bufferSize: config.bufferSize ?? 1000,
      enableBackpressure: config.enableBackpressure ?? true,
      autoReconnect: config.autoReconnect ?? true,
      reconnectDelay: config.reconnectDelay ?? 1000,
      maxReconnectAttempts: config.maxReconnectAttempts ?? 5,
    };
  }

  /**
   * Connect to the WebSocket server and initialize stream session.
   *
   * @param memorySpaceId - Memory space identifier (default: 'default')
   */
  async connect(memorySpaceId: string = 'default'): Promise<void> {
    this.memorySpaceId = memorySpaceId;

    // If already connecting, wait for that to complete
    if (this.pendingInit) {
      return this.pendingInit;
    }

    this.pendingInit = this.doConnect();
    try {
      await this.pendingInit;
    } finally {
      this.pendingInit = null;
    }
  }

  private async doConnect(): Promise<void> {
    return new Promise((resolve, reject) => {
      try {
        this.ws = new WebSocket(this.config.url);

        this.ws.onopen = () => {
          console.log('WebSocket connected');
          this.reconnectAttempts = 0;

          // Send initialization message
          const initMsg: InitRequest = {
            type: 'init',
            memory_space_id: this.memorySpaceId,
            client_buffer_size: this.config.bufferSize,
            enable_backpressure: this.config.enableBackpressure,
          };
          this.send(initMsg);
        };

        this.ws.onmessage = (event) => {
          try {
            const msg: ServerMessage = JSON.parse(event.data);
            this.handleMessage(msg);

            // Resolve connect promise when init_ack received
            if (msg.type === 'init_ack') {
              resolve();
            }
          } catch (err) {
            console.error('Failed to parse message:', err);
          }
        };

        this.ws.onerror = (error) => {
          console.error('WebSocket error:', error);
          reject(new Error('WebSocket connection failed'));
        };

        this.ws.onclose = (event) => {
          console.log('WebSocket closed:', event.code, event.reason);
          this.sessionId = null;

          // Auto-reconnect if enabled
          if (this.config.autoReconnect && this.reconnectAttempts < this.config.maxReconnectAttempts) {
            this.reconnectAttempts++;
            console.log(`Reconnecting (attempt ${this.reconnectAttempts}/${this.config.maxReconnectAttempts})...`);
            setTimeout(() => {
              this.connect(this.memorySpaceId).catch(console.error);
            }, this.config.reconnectDelay * this.reconnectAttempts);
          }
        };

        // Timeout after 10 seconds
        setTimeout(() => {
          if (this.sessionId === null) {
            reject(new Error('Connection timeout'));
          }
        }, 10000);
      } catch (err) {
        reject(err);
      }
    });
  }

  /**
   * Send an observation to the server.
   *
   * @param id - Episode identifier
   * @param what - Event description
   * @param embedding - 768-dimensional embedding vector
   * @param confidence - Encoding confidence (0.0 to 1.0)
   * @param when - Event timestamp (default: now)
   * @returns Promise that resolves when acknowledged
   */
  async observe(
    id: string,
    what: string,
    embedding: number[],
    confidence: number = 0.7,
    when?: Date
  ): Promise<AckResponse> {
    if (!this.sessionId) {
      throw new Error('Not connected. Call connect() first.');
    }

    if (embedding.length !== 768) {
      throw new Error(`Embedding must be exactly 768 dimensions, got ${embedding.length}`);
    }

    this.sequence++;

    const msg: ObservationRequest = {
      type: 'observation',
      session_id: this.sessionId,
      sequence_number: this.sequence,
      episode: {
        id,
        what,
        embedding,
        encoding_confidence: confidence,
        when: when?.toISOString(),
      },
    };

    return new Promise((resolve, reject) => {
      const handler = (ack: AckResponse) => {
        if (ack.sequence_number === this.sequence) {
          this.off('ack', handler);
          if (ack.status === 'rejected') {
            reject(new Error(`Observation rejected: ${ack.memory_id}`));
          } else {
            resolve(ack);
          }
        }
      };

      this.on('ack', handler);
      this.send(msg);

      // Timeout after 5 seconds
      setTimeout(() => {
        this.off('ack', handler);
        reject(new Error('Observation timeout'));
      }, 5000);
    });
  }

  /**
   * Pause the stream (stop sending observations).
   */
  async pause(): Promise<void> {
    if (!this.sessionId) {
      throw new Error('Not connected');
    }

    const msg: FlowControlRequest = {
      type: 'flow_control',
      session_id: this.sessionId,
      action: 'pause',
    };
    this.send(msg);
  }

  /**
   * Resume the stream (continue sending observations).
   */
  async resume(): Promise<void> {
    if (!this.sessionId) {
      throw new Error('Not connected');
    }

    const msg: FlowControlRequest = {
      type: 'flow_control',
      session_id: this.sessionId,
      action: 'resume',
    };
    this.send(msg);
  }

  /**
   * Close the WebSocket connection gracefully.
   */
  async close(): Promise<void> {
    if (!this.ws) {
      return;
    }

    // Send close message
    if (this.sessionId) {
      const msg = {
        type: 'close',
        session_id: this.sessionId,
        last_sequence: this.sequence,
      };
      this.send(msg);
    }

    // Close WebSocket
    this.ws.close();
    this.ws = null;
    this.sessionId = null;
    this.sequence = 0;
  }

  /**
   * Register a message handler for a specific message type.
   */
  on(type: MessageType, handler: (msg: any) => void): void {
    if (!this.messageHandlers.has(type)) {
      this.messageHandlers.set(type, []);
    }
    this.messageHandlers.get(type)!.push(handler);
  }

  /**
   * Unregister a message handler.
   */
  off(type: MessageType, handler: (msg: any) => void): void {
    const handlers = this.messageHandlers.get(type);
    if (handlers) {
      const index = handlers.indexOf(handler);
      if (index >= 0) {
        handlers.splice(index, 1);
      }
    }
  }

  /**
   * Get current session information.
   */
  getSession(): { sessionId: string | null; sequence: number; memorySpaceId: string } {
    return {
      sessionId: this.sessionId,
      sequence: this.sequence,
      memorySpaceId: this.memorySpaceId,
    };
  }

  private send(msg: any): void {
    if (!this.ws || this.ws.readyState !== WebSocket.OPEN) {
      throw new Error('WebSocket not connected');
    }
    this.ws.send(JSON.stringify(msg));
  }

  private handleMessage(msg: ServerMessage): void {
    console.log('Received:', msg.type, msg);

    // Handle init_ack specially
    if (msg.type === 'init_ack') {
      this.sessionId = msg.session_id;
      this.sequence = msg.initial_sequence;
      console.log('Session initialized:', this.sessionId);
    }

    // Dispatch to registered handlers
    const handlers = this.messageHandlers.get(msg.type);
    if (handlers) {
      handlers.forEach((handler) => handler(msg));
    }
  }
}

/**
 * Example usage demonstrating all client features.
 */
async function exampleUsage() {
  const client = new EngramStreamingClient('ws://localhost:8080/v1/stream');

  // Handle heartbeats
  client.on('heartbeat', (msg: HeartbeatMessage) => {
    console.log('Heartbeat received at', msg.timestamp);
  });

  // Handle errors
  client.on('error', (msg: ErrorResponse) => {
    console.error('Server error:', msg.message);
  });

  try {
    // Connect and initialize session
    console.log('Connecting...');
    await client.connect('default');
    console.log('Connected!', client.getSession());

    // Stream some observations
    for (let i = 0; i < 10; i++) {
      const embedding = new Array(768).fill(0).map(() => Math.random());
      const ack = await client.observe(
        `episode_${i}`,
        `User action ${i}`,
        embedding,
        0.85
      );
      console.log(`Observation ${i} acknowledged:`, ack.memory_id, ack.status);

      // Wait a bit between observations
      await new Promise(resolve => setTimeout(resolve, 100));
    }

    // Demonstrate flow control
    console.log('Pausing stream...');
    await client.pause();
    await new Promise(resolve => setTimeout(resolve, 2000));

    console.log('Resuming stream...');
    await client.resume();

    // Stream a few more
    for (let i = 10; i < 15; i++) {
      const embedding = new Array(768).fill(0).map(() => Math.random());
      const ack = await client.observe(
        `episode_${i}`,
        `User action ${i}`,
        embedding,
        0.85
      );
      console.log(`Observation ${i} acknowledged:`, ack.memory_id, ack.status);
    }

    // Close gracefully
    console.log('Closing connection...');
    await client.close();
    console.log('Done!');

  } catch (error) {
    console.error('Error:', error);
    await client.close();
  }
}

// Export for use in browser or Node.js
if (typeof module !== 'undefined' && module.exports) {
  module.exports = { EngramStreamingClient, exampleUsage };
}
