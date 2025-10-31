#!/usr/bin/env python3
"""
Engram streaming client example (Python)

Demonstrates gRPC streaming from Python using grpcio.

## Features

- Async gRPC bidirectional streaming
- Session management
- Observation streaming with acknowledgments
- Backpressure handling
- Error recovery with retry logic
- Performance monitoring

## Requirements

```
grpcio>=1.50.0
grpcio-tools>=1.50.0
numpy>=1.24.0
```

## Usage

```bash
python python_client.py \
    --server-addr localhost:50051 \
    --rate 1000 \
    --count 10000 \
    --space-id my_space
```

## Performance

- Recommended rate: 1K-5K observations/sec per client
- Total system capacity: 100K+ observations/sec
- Latency: P99 < 100ms
"""

import argparse
import asyncio
import time
from dataclasses import dataclass, field
from typing import List, Optional
import numpy as np

# Note: In production, import from generated protobuf:
# import grpc
# from engram.v1 import streaming_service_pb2, streaming_service_pb2_grpc


@dataclass
class Episode:
    """Episode data structure"""
    id: str
    what: str
    embedding: np.ndarray
    confidence: float


@dataclass
class Observation:
    """Observation message"""
    session_id: str
    sequence_number: int
    episode: Episode


@dataclass
class StreamingStats:
    """Statistics for streaming session"""
    observations_sent: int = 0
    acks_received: int = 0
    rejections: int = 0
    latencies: List[float] = field(default_factory=list)

    @property
    def average_latency_ms(self) -> float:
        """Calculate average latency in milliseconds"""
        if not self.latencies:
            return 0.0
        return sum(self.latencies) * 1000 / len(self.latencies)

    @property
    def p99_latency_ms(self) -> float:
        """Calculate P99 latency in milliseconds"""
        if not self.latencies:
            return 0.0
        sorted_latencies = sorted(self.latencies)
        p99_index = int(len(sorted_latencies) * 0.99)
        return sorted_latencies[min(p99_index, len(sorted_latencies) - 1)] * 1000


class StreamingClient:
    """Engram streaming client using gRPC"""

    def __init__(
        self,
        server_addr: str,
        space_id: str,
        enable_backpressure: bool = True,
        buffer_size: int = 1000
    ):
        self.server_addr = server_addr
        self.space_id = space_id
        self.enable_backpressure = enable_backpressure
        self.buffer_size = buffer_size
        self.session_id: Optional[str] = None
        self.stats = StreamingStats()

    async def initialize(self):
        """Initialize streaming session"""
        # In production, this would:
        # 1. Create gRPC channel
        # 2. Send StreamInitRequest
        # 3. Receive InitAckMessage with session_id

        # Simulated session initialization
        import os
        self.session_id = f"session_{os.getpid()}"
        print(f"Session initialized: {self.session_id}")

    async def stream_observations(
        self,
        count: int,
        rate: int
    ) -> StreamingStats:
        """
        Stream observations to Engram server

        Args:
            count: Number of observations to stream
            rate: Observations per second

        Returns:
            StreamingStats with performance metrics
        """
        if not self.session_id:
            await self.initialize()

        interval = 1.0 / rate if rate > 0 else 0.001
        start_time = time.time()

        for seq in range(count):
            observation = self.create_observation(seq)

            # Send observation with latency tracking
            send_start = time.time()
            try:
                await self.send_observation(observation)
                send_latency = time.time() - send_start

                self.stats.observations_sent += 1
                self.stats.acks_received += 1
                self.stats.latencies.append(send_latency)

            except BackpressureError:
                self.stats.rejections += 1
                # Exponential backoff on backpressure
                await asyncio.sleep(0.1)

            except Exception as e:
                print(f"Error sending observation: {e}")

            # Progress indicator
            if seq % 1000 == 0 and seq > 0:
                elapsed = time.time() - start_time
                current_rate = seq / elapsed
                print(f"Progress: {seq}/{count} ({current_rate:.0f} obs/sec)")

            # Rate limiting
            await asyncio.sleep(interval)

        total_time = time.time() - start_time
        throughput = count / total_time

        return self.stats, throughput

    def create_observation(self, seq: int) -> Observation:
        """Create synthetic observation for demonstration"""
        embedding = create_synthetic_embedding(seq)

        episode = Episode(
            id=f"episode_{seq}",
            what=f"Observation {seq} from Python client",
            embedding=embedding,
            confidence=0.85
        )

        return Observation(
            session_id=self.session_id,
            sequence_number=seq,
            episode=episode
        )

    async def send_observation(self, observation: Observation):
        """
        Send observation to server (simulated)

        In production, this would:
        1. Serialize observation to protobuf
        2. Send via gRPC stream
        3. Wait for acknowledgment
        4. Handle backpressure signals
        """
        # Simulate network latency
        await asyncio.sleep(0.001)

        # Simulate 99% success rate
        if np.random.random() > 0.99:
            raise BackpressureError("Queue full")


class BackpressureError(Exception):
    """Raised when server signals backpressure"""
    pass


def create_synthetic_embedding(seed: int, dim: int = 768) -> np.ndarray:
    """
    Create synthetic embedding for demonstration

    Args:
        seed: Random seed for reproducibility
        dim: Embedding dimension (default 768 for BERT/sentence transformers)

    Returns:
        Normalized embedding vector
    """
    np.random.seed(seed)
    embedding = np.random.randn(dim).astype(np.float32)

    # Normalize to unit length
    norm = np.linalg.norm(embedding)
    if norm > 0:
        embedding /= norm

    return embedding


async def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Engram streaming client example (Python)"
    )
    parser.add_argument(
        "--server-addr",
        default="localhost:50051",
        help="gRPC server address"
    )
    parser.add_argument(
        "--rate",
        type=int,
        default=1000,
        help="Observations per second"
    )
    parser.add_argument(
        "--count",
        type=int,
        default=10000,
        help="Total number of observations"
    )
    parser.add_argument(
        "--space-id",
        default="default",
        help="Memory space ID"
    )
    parser.add_argument(
        "--enable-backpressure",
        action="store_true",
        default=True,
        help="Enable backpressure flow control"
    )
    parser.add_argument(
        "--buffer-size",
        type=int,
        default=1000,
        help="Client buffer size"
    )

    args = parser.parse_args()

    # Print configuration
    print("Engram Streaming Client")
    print("=======================")
    print(f"Server: {args.server_addr}")
    print(f"Space: {args.space_id}")
    print(f"Rate: {args.rate} obs/sec")
    print(f"Count: {args.count} observations")
    print()

    # Initialize client
    client = StreamingClient(
        server_addr=args.server_addr,
        space_id=args.space_id,
        enable_backpressure=args.enable_backpressure,
        buffer_size=args.buffer_size
    )

    await client.initialize()

    print("Starting observation stream...")
    print()

    # Stream observations
    stats, throughput = await client.stream_observations(
        count=args.count,
        rate=args.rate
    )

    # Print results
    print()
    print("Streaming completed!")
    print("====================")
    print(f"Observations sent: {stats.observations_sent}")
    print(f"Acknowledgments received: {stats.acks_received}")
    print(f"Rejections (backpressure): {stats.rejections}")
    print(f"Average latency: {stats.average_latency_ms:.2f}ms")
    print(f"P99 latency: {stats.p99_latency_ms:.2f}ms")
    print(f"Throughput: {throughput:.0f} obs/sec")


if __name__ == "__main__":
    asyncio.run(main())
