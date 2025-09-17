# Usage

## Quick Start

```bash
# Build and start Engram
cargo build
./target/debug/engram

# Verify it's running (in another terminal)
./target/debug/engram status

# Run tests to verify everything works
cargo test

# Stop the server
./target/debug/engram stop
```

## Interface Design

Engram exposes two interfaces optimized for different integration patterns:
- **gRPC**: Type-safe, high-performance streaming operations
- **HTTP**: Simple request-response for web integration

## gRPC Interface

Binary protocol with protobuf schemas for type safety and streaming capabilities.

### Service Definition
```protobuf
service MemoryService {
  rpc Store(Episode) returns (StoreResponse);
  rpc Recall(RecallRequest) returns (stream RecallResponse);
  rpc StreamRecall(stream RecallRequest) returns (stream RecallResponse);
  rpc Consolidate(ConsolidateRequest) returns (ConsolidateResponse);
  rpc Decay(DecayRequest) returns (DecayResponse);
}

message Episode {
  google.protobuf.Timestamp when = 1;
  repeated float embedding = 2;  // 768-dim
  float confidence = 3;
  map<string, Entity> entities = 4;
  Location location = 5;
  float emotional_valence = 6;
}

message RecallRequest {
  oneof cue {
    Embedding embedding = 1;
    ContextCue context = 2;
    TemporalCue temporal = 3;
  }
  float min_confidence = 4;
  int32 max_results = 5;
  int32 max_hops = 6;
  float decay_rate = 7;
}

message RecallResponse {
  Episode episode = 1;
  float confidence = 2;
  repeated string activation_path = 3;
}
```

### Client Usage
```python
import grpc
import engram_pb2_grpc as engram

channel = grpc.insecure_channel('localhost:50051')
memory = engram.MemoryServiceStub(channel)

# Store
episode = engram.Episode(
    embedding=text_to_embedding("met alice at conference"),
    emotional_valence=0.7
)
response = memory.Store(episode)

# Stream recall with activation
request = engram.RecallRequest(
    embedding=engram.Embedding(vector=query_vec),
    min_confidence=0.6
)
for recall in memory.Recall(request):
    print(f"{recall.episode.when}: {recall.confidence:.3f}")
```

### Performance Characteristics
- Latency: <1ms local, <5ms network
- Throughput: 100K episodes/second
- Streaming: Bidirectional with backpressure
- Connection: Persistent with multiplexing

## HTTP Interface

RESTful API with JSON payloads for web applications.

### Endpoints

#### Store Episode
```http
POST /api/v1/episodes
Content-Type: application/json

{
  "text": "met alice at conference",
  "location": {"lat": 37.7, "lon": -122.4},
  "entities": ["alice"],
  "valence": 0.7
}

Response: 201 Created
{
  "id": "ep_k7x9m2",
  "activation": 0.95,
  "stored_at": "2024-01-15T10:30:00Z"
}
```

#### Recall
```http
POST /api/v1/recall
Content-Type: application/json

{
  "embedding": [0.1, 0.2, ...],  // 768-dim
  "min_confidence": 0.6,
  "max_results": 10,
  "spreading": {
    "max_hops": 3,
    "decay_rate": 0.5
  }
}

Response: 200 OK
{
  "results": [
    {
      "episode": {...},
      "confidence": 0.87,
      "activation_path": ["ep_k7x9m2", "ep_h3n8p1"]
    }
  ],
  "total_activated": 47,
  "computation_time_ms": 12
}
```

#### Context-Based Recall
```http
GET /api/v1/recall?location=37.7,-122.4&time_range=2024-01-01,2024-01-15&confidence=0.5

Response: 200 OK
{
  "results": [...],
  "context_match_score": 0.73
}
```

#### Consolidation
```http
POST /api/v1/consolidate

{
  "duration_hours": 8,
  "type": "dream"  // or "compress", "extract_schema"
}

Response: 202 Accepted
{
  "job_id": "con_j8x2m9",
  "status_url": "/api/v1/jobs/con_j8x2m9"
}
```

#### Metrics
```http
GET /api/v1/metrics

Response: 200 OK
{
  "total_episodes": 45678,
  "active_episodes": 12334,
  "mean_activation": 0.42,
  "consolidation_rate_per_hour": 120,
  "memory_usage_bytes": 234567890,
  "last_consolidation": "2024-01-15T08:00:00Z"
}
```

### Server-Sent Events Streaming
```javascript
const eventSource = new EventSource('/api/v1/monitor/events?min_activation=0.7');

eventSource.addEventListener('activation', (event) => {
  const activation = JSON.parse(event.data);
  console.log(`Episode ${activation.id}: ${activation.level}`);
});

eventSource.addEventListener('consolidation', (event) => {
  const progress = JSON.parse(event.data);
  console.log(`Consolidation progress: ${progress.percentage}%`);
});
```

### Performance Characteristics
- Latency: 5-20ms depending on operation
- Throughput: 10K requests/second
- Payload size: ~2KB average
- Connection: HTTP/2 with keep-alive

## Selection Criteria

### Use gRPC when:
- Microsecond latency matters
- Type safety required across services
- Bidirectional streaming needed
- High-frequency operations (>1K/sec)

### Use HTTP when:
- Building web applications
- Browser-based clients
- Simple request-response patterns
- Developer exploration via curl/Postman

## Authentication

Both interfaces support:
- API key authentication via headers
- mTLS for service-to-service
- JWT tokens for user contexts

```http
# HTTP
Authorization: Bearer <token>
X-API-Key: <key>

# gRPC
metadata = [('authorization', 'Bearer <token>')]
```

## Error Handling

### gRPC Status Codes
- `NOT_FOUND`: No memories above confidence threshold
- `INVALID_ARGUMENT`: Malformed embedding or parameters
- `RESOURCE_EXHAUSTED`: Memory pressure, consolidation needed
- `DEADLINE_EXCEEDED`: Activation spreading timeout

### HTTP Status Codes
- `404`: No memories found matching criteria
- `400`: Invalid request parameters
- `507`: Storage capacity exceeded
- `504`: Query timeout during spreading

## Deployment

### Docker Compose
```yaml
services:
  engram:
    image: engram:latest
    ports:
      - "50051:50051"  # gRPC
      - "8080:8080"    # HTTP
    volumes:
      - ./data:/data
    environment:
      - ENGRAM_GRPC_PORT=50051
      - ENGRAM_HTTP_PORT=8080
      - ENGRAM_STORAGE_PATH=/data
```

### Kubernetes
```yaml
apiVersion: v1
kind: Service
metadata:
  name: engram
spec:
  ports:
  - name: grpc
    port: 50051
  - name: http
    port: 8080
  selector:
    app: engram
```

## Client Libraries

### Python
```bash
pip install engram-client
```

### Go
```bash
go get github.com/engram/client-go
```

### TypeScript
```bash
npm install @engram/client
```

All clients support both gRPC and HTTP transports with identical APIs.
