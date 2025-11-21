# gRPC API Reference

Complete gRPC API reference for Engram's cognitive memory operations. This guide provides precise RPC method specifications with runnable examples in five languages: Rust, Python, TypeScript, Go, and Java.

## Table of Contents

- [Why gRPC?](#why-grpc)

- [Getting Started](#getting-started)

- [Authentication](#authentication)

- [Service Overview](#service-overview)

- [Core Memory Operations](#core-memory-operations)

- [Episodic Operations](#episodic-operations)

- [Consolidation Operations](#consolidation-operations)

- [Pattern Operations](#pattern-operations)

- [Monitoring Operations](#monitoring-operations)

- [Streaming Operations](#streaming-operations)

- [Connection Management](#connection-management)

- [Error Handling](#error-handling)

- [Performance Tuning](#performance-tuning)

## Why gRPC?

Use gRPC when you need:

- **High throughput**: 3-5x better performance than REST

- **Streaming operations**: Dream consolidation, real-time monitoring

- **Binary efficiency**: Smaller payloads for mobile/edge

- **Type safety**: Proto definitions prevent API misuse

Performance comparison (P99 latency):

| Operation | REST | gRPC | Improvement |
|-----------|------|------|-------------|
| Remember (single) | 45ms | 12ms | 3.7x faster |
| Recall (10 results) | 120ms | 35ms | 3.4x faster |
| Batch remember (100) | 890ms | 180ms | 4.9x faster |
| Dream (streaming) | N/A | 8ms/event | gRPC only |

For simple web frontends or occasional operations (<10/sec), use the [REST API](/reference/rest-api.md) instead.

## Getting Started

### Prerequisites

1. Start Engram server with gRPC enabled:

```bash
engram start --grpc-port 50051 --grpc-reflection

```

2. Generate client stubs from proto definitions:

```bash
# Proto definitions at proto/engram/v1/*.proto
git clone https://github.com/engram/engram
cd engram/proto

```

Language-specific generation covered in each section below.

### Server Address

Default: `localhost:50051`

Production: Use TLS with proper certificates (see [Security Guide](/operations/security.md))

## Authentication

gRPC requests require API key authentication when the server has authentication enabled. This uses the same security model as the HTTP API.

### Enabling Authentication

Server-side configuration in `engram.toml`:

```toml
[security]
auth_enabled = true
auth_mode = "api_key"

[security.api_keys]
storage = "sqlite"
storage_path = "/var/lib/engram/api_keys.db"
```

Start the server with gRPC listening:

```bash
engram start --grpc-port 50051
# Server will require valid API keys for all requests
```

### Adding Credentials to Requests

Include your API key in the gRPC metadata with the `authorization` header:

**Format:**

```
authorization: Bearer engram_key_{id}_{secret}
```

**Python:**

```python
import grpc

# Create channel
channel = grpc.aio.insecure_channel('localhost:50051')
client = engram_pb2_grpc.EngramServiceStub(channel)

# Add metadata with API key
metadata = [('authorization', 'Bearer engram_key_abc123_xyz789')]

# Make authenticated request
response = await client.Remember(request, metadata=metadata)
```

**Rust:**

```rust
use tonic::transport::Channel;
use tonic::Request;
use tonic::metadata::MetadataValue;

let channel = Channel::from_static("http://localhost:50051").connect().await?;
let mut client = EngramServiceClient::new(channel);

// Add authorization metadata
let mut request = Request::new(request);
request.metadata_mut().insert(
    "authorization",
    MetadataValue::from_str("Bearer engram_key_abc123_xyz789")?
);

let response = client.remember(request).await?;
```

**Go:**

```go
import "google.golang.org/grpc/metadata"

ctx := metadata.AppendToOutgoingContext(context.Background(),
    "authorization", "Bearer engram_key_abc123_xyz789")

response, err := client.Remember(ctx, request)
```

**TypeScript:**

```typescript
import { credentials, Metadata } from '@grpc/grpc-js';

const metadata = new Metadata();
metadata.add('authorization', 'Bearer engram_key_abc123_xyz789');

const call = client.Remember(request, { metadata });
call.on('data', (response) => {
    console.log('Stored:', response.memory_id);
});
```

**Java:**

```java
import io.grpc.Metadata;

Metadata metadata = new Metadata();
Metadata.Key<String> key = Metadata.Key.of("authorization", Metadata.ASCII_STRING_MARSHALLER);
metadata.put(key, "Bearer engram_key_abc123_xyz789");

// Use interceptor to attach metadata to all calls
channel = ManagedChannelBuilder
    .forAddress("localhost", 50051)
    .usePlaintext()
    .intercept(new ClientInterceptor() {
        public <ReqT, RespT> ClientCall<ReqT, RespT> interceptCall(
            MethodDescriptor<ReqT, RespT> method,
            CallOptions callOptions,
            Channel next) {
            return new SimpleForwardingClientCall<ReqT, RespT>(
                next.newCall(method, callOptions)) {
                public void start(Listener<RespT> responseListener, Metadata headers) {
                    headers.merge(metadata);
                    super.start(responseListener, headers);
                }
            };
        }
    })
    .build();
```

### Permission Model

Each API key grants specific permissions and space access:

**Memory Operations:**
- `memory:read` - Read-only operations (Recall, Recognize, Reminisce)
- `memory:write` - Write operations (Remember, Experience)

**Space Access:**
- Each key is restricted to specific memory spaces
- Requesting a space outside the key's allowed spaces returns `PERMISSION_DENIED`

**Example: Key with Limited Permissions**

```bash
# Create key with only read access
engram api-key generate \
  --name "analytics-reader" \
  --spaces "analytics_space" \
  --permissions "memory:read"

# This key can call:
# - Recall, Recognize, Reminisce

# This key cannot call:
# - Remember, Experience (returns PERMISSION_DENIED)
```

### Error Handling

Authentication failures return specific gRPC status codes:

| Status Code | Meaning | Resolution |
|-------------|---------|-----------|
| `UNAUTHENTICATED` (16) | Missing or invalid credentials | Add `authorization` metadata with valid API key |
| `UNAUTHENTICATED` (16) | API key format invalid | Verify key format: `engram_key_{id}_{secret}` |
| `UNAUTHENTICATED` (16) | API key not found | Check key ID exists, not revoked |
| `UNAUTHENTICATED` (16) | API key expired | Rotate key: `engram api-key rotate <key-id>` |
| `PERMISSION_DENIED` (7) | Insufficient permissions | Grant required permission to key |
| `PERMISSION_DENIED` (7) | Space access denied | Add space to key's allowed spaces |

**Python Error Handling:**

```python
import grpc

try:
    response = await client.Remember(request, metadata=metadata)
except grpc.aio.AioRpcError as e:
    if e.code() == grpc.StatusCode.UNAUTHENTICATED:
        print(f"Authentication failed: {e.details()}")
        # Verify API key format and validity
    elif e.code() == grpc.StatusCode.PERMISSION_DENIED:
        print(f"Permission denied: {e.details()}")
        # Check key permissions and space access
    else:
        raise
```

### When Authentication is Disabled

If the server has `auth_enabled = false`, the `authorization` header is optional. This is useful for:
- Development environments
- Internal services with network isolation
- Legacy client support

### Best Practices

1. **Store keys securely** - Use environment variables or secrets manager, never hardcode
2. **Use short-lived keys** - Set reasonable expiration dates
3. **Rotate regularly** - Replace old keys with rotated versions
4. **Monitor usage** - Check audit logs for unusual patterns
5. **Principle of least privilege** - Grant only required permissions and spaces
6. **Use TLS in production** - Always use `grpc.secure_channel()` with proper certificates

### Generating API Keys

Create keys for gRPC clients:

```bash
# Generate key for production gRPC client
engram api-key generate \
  --name "grpc-producer" \
  --spaces "production,analytics" \
  --permissions "memory:read,memory:write" \
  --expires-in "90d"

# Output
# API Key ID: engram_key_abc123def456
# Secret: xyz789uvw012qrs345
# Full Key: engram_key_abc123def456_xyz789uvw012qrs345
#
# Store this securely. It will not be shown again.
```

## Service Overview

The `EngramService` provides cognitive memory operations organized by progressive complexity:

### Level 1 (Essential) - 5 minutes

Core operations covering 80% of use cases:

- `Remember` - Store memories

- `Recall` - Retrieve memories

### Level 2 (Episodic) - 15 minutes

Rich contextual memories:

- `Experience` - Record episodic memories

- `Reminisce` - Query by context

### Level 3 (Advanced) - 45 minutes

Streaming and complex operations:

- `Dream` - Stream consolidation replay

- `Complete` - Pattern completion

- `Associate` - Create associations

- `MemoryFlow` - Bidirectional streaming

### Monitoring

System introspection:

- `Introspect` - System statistics

- `Stream` - Real-time activity

## Core Memory Operations

### Remember

Store a new memory with confidence scoring and automatic linking.

**Method Signature:**

```protobuf
rpc Remember(RememberRequest) returns (RememberResponse);

```

**Request Fields:**

- `memory_space_id` (string): Memory space for multi-tenant isolation

- `memory` (Memory): Memory object with embedding and confidence

- `auto_link` (bool): Automatically link to related memories

- `link_threshold` (float): Similarity threshold for auto-linking

**Response Fields:**

- `memory_id` (string): Stored memory identifier

- `storage_confidence` (Confidence): Storage success confidence

- `linked_memories` (repeated string): Auto-linked memory IDs

- `initial_state` (ConsolidationState): Initial consolidation state

#### Examples

<details>
<summary><strong>Rust</strong></summary>

```rust
use engram::v1::{
    engram_service_client::EngramServiceClient,
    RememberRequest, Memory, Confidence,
};
use tonic::Request;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Connect to Engram server
    let mut client = EngramServiceClient::connect("http://localhost:50051").await?;

    // Create memory with embedding
    let embedding = vec![0.12, -0.45, 0.78; 768]; // 768-dimensional vector

    let memory = Memory {
        content: "Rust provides memory safety without garbage collection".to_string(),
        embedding,
        confidence: Some(Confidence {
            value: 0.95,
            reasoning: "Verified from official documentation".to_string(),
            ..Default::default()
        }),
        ..Default::default()
    };

    // Store memory with auto-linking
    let request = Request::new(RememberRequest {
        memory_space_id: "default".to_string(),
        memory: Some(memory),
        auto_link: true,
        link_threshold: 0.7,
        ..Default::default()
    });

    let response = client.remember(request).await?;
    let result = response.into_inner();

    println!("Stored memory: {}", result.memory_id);
    println!("Storage confidence: {:.2}", result.storage_confidence.unwrap().value);
    println!("Linked to {} memories", result.linked_memories.len());

    Ok(())
}

```

**Cargo.toml:**

```toml
[dependencies]
engram = "0.1"
tonic = "0.12"
tokio = { version = "1", features = ["full"] }

```

</details>

<details>
<summary><strong>Python</strong></summary>

```python
import grpc
import asyncio
from engram.v1 import engram_pb2, engram_pb2_grpc

async def remember_example():
    # Connect to Engram server
    async with grpc.aio.insecure_channel('localhost:50051') as channel:
        client = engram_pb2_grpc.EngramServiceStub(channel)

        # Create memory with embedding
        embedding = [0.12, -0.45, 0.78] * 256  # 768 dimensions

        memory = engram_pb2.Memory(
            content="Python emphasizes code readability and simplicity",
            embedding=embedding,
            confidence=engram_pb2.Confidence(
                value=0.95,
                reasoning="Verified from authoritative source"
            )
        )

        # Store memory with auto-linking
        request = engram_pb2.RememberRequest(
            memory_space_id="default",
            memory=memory,
            auto_link=True,
            link_threshold=0.7
        )

        response = await client.Remember(request)

        print(f"Stored memory: {response.memory_id}")
        print(f"Storage confidence: {response.storage_confidence.value:.2f}")
        print(f"Linked to {len(response.linked_memories)} memories")

if __name__ == "__main__":
    asyncio.run(remember_example())

```

**Setup:**

```bash
pip install grpcio grpcio-tools
python -m grpc_tools.protoc -I../../proto \
    --python_out=. --grpc_python_out=. \
    ../../proto/engram/v1/*.proto

```

</details>

<details>
<summary><strong>TypeScript</strong></summary>

```typescript
import * as grpc from '@grpc/grpc-js';
import * as protoLoader from '@grpc/proto-loader';
import { ProtoGrpcType } from './generated/engram';

const PROTO_PATH = '../../proto/engram/v1/service.proto';

async function rememberExample() {
    // Load proto definition
    const packageDefinition = protoLoader.loadSync(PROTO_PATH, {
        keepCase: true,
        longs: String,
        enums: String,
        defaults: true,
        oneofs: true
    });

    const grpcObject = grpc.loadPackageDefinition(
        packageDefinition
    ) as unknown as ProtoGrpcType;

    // Connect to Engram server
    const client = new grpcObject.engram.v1.EngramService(
        'localhost:50051',
        grpc.credentials.createInsecure()
    );

    // Create memory with embedding
    const embedding = Array(768).fill(0).map(() => Math.random() * 2 - 1);

    const request = {
        memory_space_id: 'default',
        memory: {
            content: 'TypeScript adds static typing to JavaScript',
            embedding: embedding,
            confidence: {
                value: 0.95,
                reasoning: 'Verified from TypeScript documentation'
            }
        },
        auto_link: true,
        link_threshold: 0.7
    };

    // Store memory
    return new Promise((resolve, reject) => {
        client.Remember(request, (err, response) => {
            if (err) {
                reject(err);
                return;
            }

            console.log(`Stored memory: ${response.memory_id}`);
            console.log(`Storage confidence: ${response.storage_confidence?.value.toFixed(2)}`);
            console.log(`Linked to ${response.linked_memories.length} memories`);

            resolve(response);
        });
    });
}

rememberExample().catch(console.error);

```

**package.json:**

```json
{
  "dependencies": {
    "@grpc/grpc-js": "^1.9.0",
    "@grpc/proto-loader": "^0.7.0"
  }
}

```

</details>

<details>
<summary><strong>Go</strong></summary>

```go
package main

import (
    "context"
    "fmt"
    "log"

    pb "github.com/engram/engram/proto/engram/v1"
    "google.golang.org/grpc"
    "google.golang.org/grpc/credentials/insecure"
)

func main() {
    // Connect to Engram server
    conn, err := grpc.Dial(
        "localhost:50051",
        grpc.WithTransportCredentials(insecure.NewCredentials()),
    )
    if err != nil {
        log.Fatalf("Failed to connect: %v", err)
    }
    defer conn.Close()

    client := pb.NewEngramServiceClient(conn)

    // Create memory with embedding
    embedding := make([]float32, 768)
    for i := range embedding {
        embedding[i] = 0.5 // Simplified example
    }

    memory := &pb.Memory{
        Content: "Go provides efficient concurrency with goroutines",
        Embedding: embedding,
        Confidence: &pb.Confidence{
            Value: 0.95,
            Reasoning: "Verified from Go documentation",
        },
    }

    // Store memory with auto-linking
    request := &pb.RememberRequest{
        MemorySpaceId: "default",
        MemoryType: &pb.RememberRequest_Memory{
            Memory: memory,
        },
        AutoLink: true,
        LinkThreshold: 0.7,
    }

    ctx := context.Background()
    response, err := client.Remember(ctx, request)
    if err != nil {
        log.Fatalf("Remember failed: %v", err)
    }

    fmt.Printf("Stored memory: %s\n", response.MemoryId)
    fmt.Printf("Storage confidence: %.2f\n", response.StorageConfidence.Value)
    fmt.Printf("Linked to %d memories\n", len(response.LinkedMemories))
}

```

**go.mod:**

```go
module example

go 1.21

require (
    github.com/engram/engram v0.1.0
    google.golang.org/grpc v1.59.0
)

```

</details>

<details>
<summary><strong>Java</strong></summary>

```java
import io.grpc.ManagedChannel;
import io.grpc.ManagedChannelBuilder;
import io.engram.v1.EngramServiceGrpc;
import io.engram.v1.Memory;
import io.engram.v1.Confidence;
import io.engram.v1.RememberRequest;
import io.engram.v1.RememberResponse;

import java.util.ArrayList;
import java.util.List;

public class RememberExample {
    public static void main(String[] args) {
        // Connect to Engram server
        ManagedChannel channel = ManagedChannelBuilder
            .forAddress("localhost", 50051)
            .usePlaintext()
            .build();

        EngramServiceGrpc.EngramServiceBlockingStub client =
            EngramServiceGrpc.newBlockingStub(channel);

        // Create memory with embedding
        List<Float> embedding = new ArrayList<>(768);
        for (int i = 0; i < 768; i++) {
            embedding.add(0.5f); // Simplified example
        }

        Memory memory = Memory.newBuilder()
            .setContent("Java provides platform independence via JVM")
            .addAllEmbedding(embedding)
            .setConfidence(Confidence.newBuilder()
                .setValue(0.95f)
                .setReasoning("Verified from Java documentation")
                .build())
            .build();

        // Store memory with auto-linking
        RememberRequest request = RememberRequest.newBuilder()
            .setMemorySpaceId("default")
            .setMemory(memory)
            .setAutoLink(true)
            .setLinkThreshold(0.7f)
            .build();

        RememberResponse response = client.remember(request);

        System.out.printf("Stored memory: %s%n", response.getMemoryId());
        System.out.printf("Storage confidence: %.2f%n",
            response.getStorageConfidence().getValue());
        System.out.printf("Linked to %d memories%n",
            response.getLinkedMemoriesList().size());

        channel.shutdown();
    }
}

```

**pom.xml:**

```xml
<dependencies>
    <dependency>
        <groupId>io.grpc</groupId>
        <artifactId>grpc-netty-shaded</artifactId>
        <version>1.59.0</version>
    </dependency>
    <dependency>
        <groupId>io.grpc</groupId>
        <artifactId>grpc-protobuf</artifactId>
        <version>1.59.0</version>
    </dependency>
    <dependency>
        <groupId>io.grpc</groupId>
        <artifactId>grpc-stub</artifactId>
        <version>1.59.0</version>
    </dependency>
</dependencies>

```

</details>

### Recall

Retrieve memories using various cue types with spreading activation.

**Method Signature:**

```protobuf
rpc Recall(RecallRequest) returns (RecallResponse);

```

**Request Fields:**

- `memory_space_id` (string): Memory space identifier

- `cue` (Cue): Retrieval cue (embedding, semantic, context, temporal, or pattern)

- `max_results` (int32): Maximum memories to return

- `include_metadata` (bool): Include recall statistics

- `trace_activation` (bool): Include activation trace

**Response Fields:**

- `memories` (repeated Memory): Retrieved memories

- `recall_confidence` (Confidence): Overall recall confidence

- `metadata` (RecallMetadata): Recall statistics

- `traces` (repeated ActivationTrace): Spreading activation traces

#### Examples

<details>
<summary><strong>Python - Semantic Cue</strong></summary>

```python
import grpc
import asyncio
from engram.v1 import engram_pb2, engram_pb2_grpc

async def recall_semantic():
    async with grpc.aio.insecure_channel('localhost:50051') as channel:
        client = engram_pb2_grpc.EngramServiceStub(channel)

        # Semantic cue for natural language query
        cue = engram_pb2.Cue(
            semantic=engram_pb2.SemanticCue(
                query="programming language features",
                fuzzy_threshold=0.6,
                required_tags=["programming"],
                excluded_tags=["deprecated"]
            )
        )

        request = engram_pb2.RecallRequest(
            memory_space_id="default",
            cue=cue,
            max_results=10,
            include_metadata=True,
            trace_activation=True
        )

        response = await client.Recall(request)

        print(f"Recall confidence: {response.recall_confidence.value:.2f}")
        print(f"Found {len(response.memories)} memories:")

        for memory in response.memories:
            print(f"  - {memory.content[:50]}... (activation: {memory.activation:.2f})")

        if response.metadata:
            print(f"\nMetadata:")
            print(f"  Total activated: {response.metadata.total_activated}")
            print(f"  Above threshold: {response.metadata.above_threshold}")
            print(f"  Recall time: {response.metadata.recall_time_ms}ms")

asyncio.run(recall_semantic())

```

</details>

<details>
<summary><strong>Rust - Embedding Cue</strong></summary>

```rust
use engram::v1::{
    engram_service_client::EngramServiceClient,
    RecallRequest, Cue, EmbeddingCue,
};
use tonic::Request;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut client = EngramServiceClient::connect("http://localhost:50051").await?;

    // Embedding cue for similarity search
    let embedding = vec![0.15, -0.42, 0.81; 768];

    let cue = Cue {
        cue_type: Some(engram::v1::cue::CueType::Embedding(EmbeddingCue {
            vector: embedding,
            similarity_threshold: 0.75,
        })),
        ..Default::default()
    };

    let request = Request::new(RecallRequest {
        memory_space_id: "default".to_string(),
        cue: Some(cue),
        max_results: 10,
        include_metadata: true,
        trace_activation: true,
    });

    let response = client.recall(request).await?;
    let result = response.into_inner();

    println!("Recall confidence: {:.2}",
        result.recall_confidence.unwrap().value);
    println!("Found {} memories", result.memories.len());

    for memory in result.memories {
        println!("  - {} (activation: {:.2})",
            memory.content, memory.activation);
    }

    Ok(())
}

```

</details>

<details>
<summary><strong>Go - Context Cue</strong></summary>

```go
package main

import (
    "context"
    "fmt"
    "log"
    "time"

    pb "github.com/engram/engram/proto/engram/v1"
    "google.golang.org/grpc"
    "google.golang.org/grpc/credentials/insecure"
    "google.golang.org/protobuf/types/known/timestamppb"
)

func main() {
    conn, err := grpc.Dial("localhost:50051",
        grpc.WithTransportCredentials(insecure.NewCredentials()))
    if err != nil {
        log.Fatal(err)
    }
    defer conn.Close()

    client := pb.NewEngramServiceClient(conn)

    // Context cue for episodic retrieval
    timeStart := time.Date(2024, 10, 1, 0, 0, 0, 0, time.UTC)
    timeEnd := time.Date(2024, 10, 31, 23, 59, 59, 0, time.UTC)

    cue := &pb.Cue{
        CueType: &pb.Cue_Context{
            Context: &pb.ContextCue{
                TimeStart: timestamppb.New(timeStart),
                TimeEnd:   timestamppb.New(timeEnd),
                Location:  "office",
                Participants: []string{"Alice", "Bob"},
            },
        },
    }

    request := &pb.RecallRequest{
        MemorySpaceId:   "default",
        Cue:             cue,
        MaxResults:      10,
        IncludeMetadata: true,
        TraceActivation: true,
    }

    ctx := context.Background()
    response, err := client.Recall(ctx, request)
    if err != nil {
        log.Fatal(err)
    }

    fmt.Printf("Recall confidence: %.2f\n", response.RecallConfidence.Value)
    fmt.Printf("Found %d memories\n", len(response.Memories))

    for _, memory := range response.Memories {
        fmt.Printf("  - %s (activation: %.2f)\n",
            memory.Content, memory.Activation)
    }
}

```

</details>

### Forget

Remove or suppress memories with reversible forgetting modes.

**Method Signature:**

```protobuf
rpc Forget(ForgetRequest) returns (ForgetResponse);

```

**Forget Modes:**

- `FORGET_MODE_SUPPRESS`: Reduce activation (reversible)

- `FORGET_MODE_DELETE`: Permanent removal

- `FORGET_MODE_OVERWRITE`: Replace with new memory

#### Example (TypeScript)

```typescript
async function forgetExample() {
    const request = {
        memory_space_id: 'default',
        memory_id: 'mem_1698765432_a1b2c3d4',
        mode: 'FORGET_MODE_SUPPRESS'
    };

    return new Promise((resolve, reject) => {
        client.Forget(request, (err, response) => {
            if (err) {
                reject(err);
                return;
            }

            console.log(`Memories affected: ${response.memories_affected}`);
            console.log(`Forget confidence: ${response.forget_confidence?.value.toFixed(2)}`);
            console.log(`Reversible: ${response.reversible}`);

            resolve(response);
        });
    });
}

```

### Recognize

Check if a memory pattern feels familiar (recognition vs recall).

**Method Signature:**

```protobuf
rpc Recognize(RecognizeRequest) returns (RecognizeResponse);

```

#### Example (Java)

```java
public void recognizeExample() {
    RecognizeRequest request = RecognizeRequest.newBuilder()
        .setMemorySpaceId("default")
        .setContent("Python was created by Guido van Rossum")
        .setRecognitionThreshold(0.8f)
        .build();

    RecognizeResponse response = client.recognize(request);

    System.out.printf("Recognized: %b%n", response.getRecognized());
    System.out.printf("Recognition confidence: %.2f%n",
        response.getRecognitionConfidence().getValue());
    System.out.printf("Familiarity score: %.2f%n",
        response.getFamiliarityScore());

    for (Memory similar : response.getSimilarMemoriesList()) {
        System.out.printf("  Similar: %s%n", similar.getContent());
    }
}

```

## Episodic Operations

### Experience

Record episodic memory with what/when/where/who/why/how context.

**Method Signature:**

```protobuf
rpc Experience(ExperienceRequest) returns (ExperienceResponse);

```

#### Example (Python)

```python
from datetime import datetime
import grpc
from google.protobuf.timestamp_pb2 import Timestamp
from engram.v1 import engram_pb2, engram_pb2_grpc

async def experience_example():
    async with grpc.aio.insecure_channel('localhost:50051') as channel:
        client = engram_pb2_grpc.EngramServiceStub(channel)

        # Create rich episodic memory
        now = Timestamp()
        now.FromDatetime(datetime.now())

        episode = engram_pb2.Episode(
            what="Learned about gRPC streaming patterns",
            when=now,
            where_location="virtual meeting room",
            who=["team lead", "senior engineer"],
            why="improving API performance",
            how="comparing unary vs streaming approaches",
            emotional_valence=0.6,  # Positive experience
            importance=0.8,         # High importance
            encoding_confidence=engram_pb2.Confidence(
                value=0.9,
                reasoning="Clear memory, took detailed notes"
            )
        )

        request = engram_pb2.ExperienceRequest(
            memory_space_id="default",
            episode=episode,
            immediate_consolidation=False,
            context_links=["mem_grpc_docs", "mem_api_design"]
        )

        response = await client.Experience(request)

        print(f"Recorded episode: {response.episode_id}")
        print(f"Encoding quality: {response.encoding_quality.value:.2f}")
        print(f"Context links created: {response.context_links_created}")

```

### Reminisce

Query episodic memories by context.

**Method Signature:**

```protobuf
rpc Reminisce(ReminisceRequest) returns (ReminisceResponse);

```

#### Example (Rust)

```rust
use engram::v1::{
    engram_service_client::EngramServiceClient,
    ReminisceRequest,
};
use chrono::{Utc, Duration};
use prost_types::Timestamp;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut client = EngramServiceClient::connect("http://localhost:50051").await?;

    // Query memories from last week
    let now = Utc::now();
    let week_ago = now - Duration::days(7);

    let request = ReminisceRequest {
        memory_space_id: "default".to_string(),
        cue: Some(engram::v1::reminisce_request::Cue::AboutTopic(
            "neuroscience lectures".to_string()
        )),
        time_window_start: Some(Timestamp::from(week_ago)),
        time_window_end: Some(Timestamp::from(now)),
        include_emotional: true,
        ..Default::default()
    };

    let response = client.reminisce(request).await?;
    let result = response.into_inner();

    println!("Recall vividness: {:.2}", result.recall_vividness.unwrap().value);
    println!("Found {} episodes", result.episodes.len());

    for episode in result.episodes {
        println!("  - {}: {}", episode.when.unwrap(), episode.what);
    }

    if let Some(emotional) = result.emotional_summary.get("mean_valence") {
        println!("\nMean emotion: {}", emotional);
    }

    println!("\nThemes: {}", result.memory_themes.join(", "));

    Ok(())
}

```

## Consolidation Operations

### Consolidate

Trigger memory consolidation with selective criteria.

**Method Signature:**

```protobuf
rpc Consolidate(ConsolidateRequest) returns (ConsolidateResponse);

```

#### Example (Go)

```go
func consolidateExample() {
    // Consolidate memories older than 24 hours
    olderThan := timestamppb.New(time.Now().Add(-24 * time.Hour))

    request := &pb.ConsolidateRequest{
        MemorySpaceId: "default",
        Target: &pb.ConsolidateRequest_Criteria{
            Criteria: &pb.ConsolidationCriteria{
                OlderThan: olderThan,
                ImportanceThreshold: 0.5,
                Types: []pb.MemoryType{
                    pb.MemoryType_MEMORY_TYPE_EPISODIC,
                    pb.MemoryType_MEMORY_TYPE_SEMANTIC,
                },
            },
        },
        Mode: pb.ConsolidationMode_CONSOLIDATION_MODE_SYNAPTIC,
    }

    ctx := context.Background()
    response, err := client.Consolidate(ctx, request)
    if err != nil {
        log.Fatal(err)
    }

    fmt.Printf("Consolidated %d memories\n", response.MemoriesConsolidated)
    fmt.Printf("Created %d new associations\n", response.NewAssociations)
    fmt.Printf("Next consolidation: %s\n",
        response.NextConsolidation.AsTime().Format(time.RFC3339))
}

```

### Dream

Stream dream-like memory replay with insights (server streaming).

**Method Signature:**

```protobuf
rpc Dream(DreamRequest) returns (stream DreamResponse);

```

This is Engram's most cognitively interesting operation - watching memory consolidation happen in real-time.

#### Example (Python - Async Streaming)

```python
async def dream_example():
    async with grpc.aio.insecure_channel('localhost:50051') as channel:
        client = engram_pb2_grpc.EngramServiceStub(channel)

        request = engram_pb2.DreamRequest(
            memory_space_id="default",
            replay_cycles=20,
            creativity_factor=0.7,  # Balance exploration/exploitation
            generate_insights=True
        )

        print("Starting dream consolidation...")

        replay_count = 0
        insight_count = 0

        async for response in client.Dream(request):
            if response.HasField('replay'):
                replay_count += 1
                print(f"\nReplay {replay_count}:")
                print(f"  Memories: {', '.join(response.replay.memory_ids[:3])}...")
                print(f"  Novelty: {response.replay.sequence_novelty:.2f}")
                if response.replay.narrative:
                    print(f"  Narrative: {response.replay.narrative}")

            elif response.HasField('insight'):
                insight_count += 1
                print(f"\nInsight {insight_count}:")
                print(f"  {response.insight.description}")
                print(f"  Confidence: {response.insight.insight_confidence.value:.2f}")
                if response.insight.suggested_action:
                    print(f"  Suggested: {response.insight.suggested_action}")

            elif response.HasField('progress'):
                print(f"\nProgress:")
                print(f"  Replayed: {response.progress.memories_replayed}")
                print(f"  New connections: {response.progress.new_connections}")
                print(f"  Strength: {response.progress.consolidation_strength:.2f}")

        print(f"\nDream complete: {replay_count} replays, {insight_count} insights")

asyncio.run(dream_example())

```

#### Example (Rust - Stream Processing)

```rust
use engram::v1::{
    engram_service_client::EngramServiceClient,
    DreamRequest,
};
use tonic::Streaming;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut client = EngramServiceClient::connect("http://localhost:50051").await?;

    let request = DreamRequest {
        memory_space_id: "default".to_string(),
        replay_cycles: 20,
        creativity_factor: 0.7,
        generate_insights: true,
    };

    let mut stream = client.dream(request).await?.into_inner();

    println!("Starting dream consolidation...");

    let mut replay_count = 0;
    let mut insight_count = 0;

    while let Some(response) = stream.message().await? {
        match response.content {
            Some(engram::v1::dream_response::Content::Replay(replay)) => {
                replay_count += 1;
                println!("\nReplay {}:", replay_count);
                println!("  Memories: {:?}", &replay.memory_ids[..3.min(replay.memory_ids.len())]);
                println!("  Novelty: {:.2}", replay.sequence_novelty);
            }

            Some(engram::v1::dream_response::Content::Insight(insight)) => {
                insight_count += 1;
                println!("\nInsight {}:", insight_count);
                println!("  {}", insight.description);
                println!("  Confidence: {:.2}",
                    insight.insight_confidence.unwrap().value);
            }

            Some(engram::v1::dream_response::Content::Progress(progress)) => {
                println!("\nProgress:");
                println!("  Replayed: {}", progress.memories_replayed);
                println!("  New connections: {}", progress.new_connections);
                println!("  Strength: {:.2}", progress.consolidation_strength);
            }

            None => {}
        }
    }

    println!("\nDream complete: {} replays, {} insights", replay_count, insight_count);

    Ok(())
}

```

## Streaming Operations

### MemoryFlow (Bidirectional Streaming)

Interactive memory sessions with bidirectional communication.

**Method Signature:**

```protobuf
rpc MemoryFlow(stream MemoryFlowRequest) returns (stream MemoryFlowResponse);

```

#### Example (Go - Bidirectional Stream)

```go
func memoryFlowExample() {
    ctx := context.Background()
    stream, err := client.MemoryFlow(ctx)
    if err != nil {
        log.Fatal(err)
    }

    sessionID := "session_" + time.Now().Format("20060102_150405")
    seqNum := uint64(0)

    // Goroutine to receive responses
    go func() {
        for {
            response, err := stream.Recv()
            if err == io.EOF {
                return
            }
            if err != nil {
                log.Printf("Receive error: %v", err)
                return
            }

            switch result := response.Response.(type) {
            case *pb.MemoryFlowResponse_RememberResult:
                fmt.Printf("Stored: %s\n", result.RememberResult.MemoryId)

            case *pb.MemoryFlowResponse_RecallResult:
                fmt.Printf("Recalled %d memories\n",
                    len(result.RecallResult.Memories))

            case *pb.MemoryFlowResponse_Event:
                fmt.Printf("Event: %s\n", result.Event.Description)

            case *pb.MemoryFlowResponse_Status:
                fmt.Printf("Status: %s\n", result.Status.Message)
            }
        }
    }()

    // Send remember request
    seqNum++
    rememberReq := &pb.MemoryFlowRequest{
        MemorySpaceId: "default",
        Request: &pb.MemoryFlowRequest_Remember{
            Remember: &pb.RememberRequest{
                Memory: &pb.Memory{
                    Content: "Bidirectional streaming enables interactive sessions",
                    // ... other fields
                },
            },
        },
        SessionId: sessionID,
        SequenceNumber: seqNum,
    }

    if err := stream.Send(rememberReq); err != nil {
        log.Fatal(err)
    }

    // Send recall request
    seqNum++
    recallReq := &pb.MemoryFlowRequest{
        MemorySpaceId: "default",
        Request: &pb.MemoryFlowRequest_Recall{
            Recall: &pb.RecallRequest{
                Cue: &pb.Cue{
                    CueType: &pb.Cue_Semantic{
                        Semantic: &pb.SemanticCue{
                            Query: "streaming patterns",
                        },
                    },
                },
            },
        },
        SessionId: sessionID,
        SequenceNumber: seqNum,
    }

    if err := stream.Send(recallReq); err != nil {
        log.Fatal(err)
    }

    // Subscribe to events
    seqNum++
    subscribeReq := &pb.MemoryFlowRequest{
        MemorySpaceId: "default",
        Request: &pb.MemoryFlowRequest_Subscribe{
            Subscribe: &pb.StreamSubscription{
                EventTypes: []pb.StreamEventType{
                    pb.StreamEventType_STREAM_EVENT_TYPE_ACTIVATION,
                    pb.StreamEventType_STREAM_EVENT_TYPE_STORAGE,
                },
                MinImportance: 0.5,
                EnableBackpressure: true,
            },
        },
        SessionId: sessionID,
        SequenceNumber: seqNum,
    }

    if err := stream.Send(subscribeReq); err != nil {
        log.Fatal(err)
    }

    // Keep session alive
    time.Sleep(5 * time.Second)

    stream.CloseSend()
}

```

## Connection Management

### Connection Pooling

For production deployments, use connection pooling to amortize connection overhead.

#### Rust (Tonic)

```rust
use engram::v1::engram_service_client::EngramServiceClient;
use tonic::transport::{Channel, Endpoint};
use std::time::Duration;

async fn create_pooled_client() -> Result<EngramServiceClient<Channel>, Box<dyn std::error::Error>> {
    let endpoint = Endpoint::from_static("http://localhost:50051")
        .connect_timeout(Duration::from_secs(5))
        .timeout(Duration::from_secs(30))
        .tcp_keepalive(Some(Duration::from_secs(60)))
        .http2_keep_alive_interval(Duration::from_secs(30))
        .keep_alive_timeout(Duration::from_secs(10))
        .concurrency_limit(256);

    let channel = endpoint.connect().await?;
    Ok(EngramServiceClient::new(channel))
}

```

#### Python (grpcio)

```python
import grpc

# Connection options for production
options = [
    ('grpc.keepalive_time_ms', 30000),
    ('grpc.keepalive_timeout_ms', 10000),
    ('grpc.keepalive_permit_without_calls', 1),
    ('grpc.http2.max_pings_without_data', 0),
    ('grpc.http2.min_time_between_pings_ms', 10000),
    ('grpc.http2.min_ping_interval_without_data_ms', 30000),
]

channel = grpc.aio.insecure_channel('localhost:50051', options=options)
client = engram_pb2_grpc.EngramServiceStub(channel)

```

#### Go

```go
import (
    "google.golang.org/grpc"
    "google.golang.org/grpc/keepalive"
    "time"
)

func createClient() (*grpc.ClientConn, error) {
    kacp := keepalive.ClientParameters{
        Time:                30 * time.Second,
        Timeout:             10 * time.Second,
        PermitWithoutStream: true,
    }

    return grpc.Dial(
        "localhost:50051",
        grpc.WithInsecure(),
        grpc.WithKeepaliveParams(kacp),
        grpc.WithDefaultCallOptions(
            grpc.MaxCallRecvMsgSize(10 * 1024 * 1024), // 10MB
            grpc.MaxCallSendMsgSize(10 * 1024 * 1024),
        ),
    )
}

```

### Retry Logic

Implement exponential backoff for transient failures.

#### TypeScript

```typescript
import { Metadata, ServiceError, status } from '@grpc/grpc-js';

async function retryableCall<T>(
    fn: () => Promise<T>,
    maxRetries: number = 3
): Promise<T> {
    let lastError: ServiceError;

    for (let attempt = 0; attempt < maxRetries; attempt++) {
        try {
            return await fn();
        } catch (err) {
            lastError = err as ServiceError;

            // Only retry on transient errors
            if (
                lastError.code === status.UNAVAILABLE ||
                lastError.code === status.DEADLINE_EXCEEDED
            ) {
                const backoffMs = Math.pow(2, attempt) * 1000;
                console.log(`Retry attempt ${attempt + 1} after ${backoffMs}ms`);
                await sleep(backoffMs);
            } else {
                throw err; // Non-retriable error
            }
        }
    }

    throw lastError;
}

// Usage
const response = await retryableCall(() =>
    new Promise((resolve, reject) => {
        client.Remember(request, (err, response) => {
            if (err) reject(err);
            else resolve(response);
        });
    })
);

```

## Error Handling

gRPC errors map to standard status codes. See [Error Codes Reference](/reference/error-codes.md) for detailed catalog.

### Status Code Mapping

| gRPC Status | HTTP Equiv | Engram Error | Meaning |
|-------------|-----------|--------------|---------|
| INVALID_ARGUMENT | 400 | ERR-4001 | Invalid request field |
| NOT_FOUND | 404 | ERR-1003 | Memory space not found |
| ALREADY_EXISTS | 409 | ERR-1004 | Duplicate memory ID |
| DEADLINE_EXCEEDED | 408 | ERR-2003 | Activation timeout |
| RESOURCE_EXHAUSTED | 429 | ERR-5003 | Rate limit exceeded |
| UNAVAILABLE | 503 | ERR-5004 | Service unavailable |

### Error Handling Example (Java)

```java
import io.grpc.StatusRuntimeException;
import io.grpc.Status;

public void handleErrors() {
    try {
        RememberResponse response = client.remember(request);
    } catch (StatusRuntimeException e) {
        switch (e.getStatus().getCode()) {
            case INVALID_ARGUMENT:
                System.err.println("Invalid request: " + e.getMessage());
                // Fix request and retry
                break;

            case DEADLINE_EXCEEDED:
                System.err.println("Request timeout: " + e.getMessage());
                // Reduce query complexity and retry
                break;

            case UNAVAILABLE:
                System.err.println("Service unavailable: " + e.getMessage());
                // Implement exponential backoff
                break;

            default:
                System.err.println("Unexpected error: " + e.getMessage());
                throw e;
        }
    }
}

```

## Performance Tuning

### Batch Operations

Use streaming for batch operations to achieve 4-5x better throughput.

#### Python - Streaming Remember

```python
async def batch_remember(memories: list):
    async with grpc.aio.insecure_channel('localhost:50051') as channel:
        client = engram_pb2_grpc.EngramServiceStub(channel)

        async def request_generator():
            for memory in memories:
                yield engram_pb2.RememberRequest(
                    memory_space_id="default",
                    memory=memory,
                    auto_link=True
                )

        responses = []
        async for response in client.StreamingRemember(request_generator()):
            responses.append(response)
            print(f"Stored: {response.memory_id}")

        return responses

# Store 100 memories efficiently
memories = [create_memory(i) for i in range(100)]
results = await batch_remember(memories)
print(f"Stored {len(results)} memories via streaming")

```

### Message Size Limits

Adjust for large embeddings or batch operations:

```python
# Python
options = [
    ('grpc.max_send_message_length', 10 * 1024 * 1024),  # 10MB
    ('grpc.max_receive_message_length', 10 * 1024 * 1024),
]

```

```rust
// Rust
let endpoint = Endpoint::from_static("http://localhost:50051")
    .initial_connection_window_size(1024 * 1024)
    .initial_stream_window_size(1024 * 1024);

```

### Backpressure Handling

For streaming operations, implement backpressure to avoid overwhelming the server.

#### Go - Backpressure

```go
func streamWithBackpressure(client pb.EngramServiceClient) {
    ctx := context.Background()
    stream, err := client.MemoryFlow(ctx)
    if err != nil {
        log.Fatal(err)
    }

    // Monitor buffer usage
    bufferSize := 100
    bufferUsed := 0

    for {
        // Check buffer capacity
        if bufferUsed > bufferSize * 0.8 {
            // Send pause signal
            stream.Send(&pb.MemoryFlowRequest{
                Request: &pb.MemoryFlowRequest_Control{
                    Control: &pb.FlowControl{
                        Action: pb.FlowControl_ACTION_PAUSE,
                        BufferRemaining: int32(bufferSize - bufferUsed),
                        Reason: "Client buffer 80% full",
                    },
                },
            })

            // Drain buffer
            time.Sleep(100 * time.Millisecond)
            bufferUsed = 0

            // Resume
            stream.Send(&pb.MemoryFlowRequest{
                Request: &pb.MemoryFlowRequest_Control{
                    Control: &pb.FlowControl{
                        Action: pb.FlowControl_ACTION_RESUME,
                    },
                },
            })
        }

        // Process messages
        response, err := stream.Recv()
        if err == io.EOF {
            break
        }
        if err != nil {
            log.Fatal(err)
        }

        bufferUsed++
        // Process response...
    }
}

```

## Next Steps

- **Quick Start**: Try the [15-minute API Quickstart](/tutorials/api-quickstart.md)

- **REST Alternative**: See [REST API Reference](/reference/rest-api.md) for simpler integration

- **Error Handling**: Read [Error Codes Catalog](/reference/error-codes.md)

- **Examples**: Explore complete [Multi-Language Examples](/reference/api-examples/)

- **Production**: Check [Operations Guide](/operations/) for deployment

## Proto Definitions

All message definitions are available at:

- `proto/engram/v1/service.proto` - Service RPCs

- `proto/engram/v1/memory.proto` - Core types

Generate client stubs for your language using the proto compiler. See language-specific examples above for generation commands.
