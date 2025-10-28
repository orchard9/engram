# Example 01: Basic Remember/Recall

**Learning Goal**: Master the fundamental remember/recall cycle that covers 80% of Engram use cases.

**Difficulty**: Beginner
**Time**: 10 minutes
**Prerequisites**: Engram server running

## Cognitive Concept

This example demonstrates the core cognitive loop:

```
ENCODE → STORE → RETRIEVE → USE
   ↓        ↓        ↓
Remember  Storage  Recall
          (with     (with
         confidence) confidence)

```

Unlike traditional databases where writes either succeed or fail, Engram acknowledges that:

1. **Storage has confidence** - how certain are we the write succeeded?

2. **Memories have intrinsic confidence** - how certain are we this fact is true?

3. **Retrieval has confidence** - how certain are we we found the right memories?

## What You'll Learn

- Store a memory with `remember()` and confidence scoring

- Retrieve memories with `recall()` using semantic cues

- Understand activation vs confidence

- See auto-linking create connections

## Code Examples

### Python

**File**: `python.py`

```python
#!/usr/bin/env python3
"""Basic remember/recall example in Python."""

import grpc
from engram.v1 import engram_pb2, engram_pb2_grpc

async def main():
    # Connect to Engram
    async with grpc.aio.insecure_channel('localhost:50051') as channel:
        client = engram_pb2_grpc.EngramServiceStub(channel)

        # REMEMBER: Store a memory with confidence
        memory = engram_pb2.Memory(
            content="The Eiffel Tower is located in Paris, France",
            embedding=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],  # Simplified 8-dim
            confidence=engram_pb2.Confidence(
                value=0.95,
                reasoning="Verified geographical fact"
            )
        )

        response = await client.Remember(engram_pb2.RememberRequest(
            memory_space_id="default",
            memory=memory,
            auto_link=True,
            link_threshold=0.7
        ))

        print(f"Stored memory: {response.memory_id}")
        print(f"Storage confidence: {response.storage_confidence.value:.2f}")
        print(f"Linked memories: {len(response.linked_memories)}")

        # RECALL: Retrieve memory with semantic cue
        recall_response = await client.Recall(engram_pb2.RecallRequest(
            memory_space_id="default",
            cue=engram_pb2.Cue(
                semantic=engram_pb2.SemanticCue(query="Paris landmarks")
            ),
            max_results=5
        ))

        print(f"\nRecall confidence: {recall_response.recall_confidence.value:.2f}")
        print(f"Found {len(recall_response.memories)} memories:")

        for mem in recall_response.memories:
            print(f"  - {mem.content}")
            print(f"    Activation: {mem.activation:.2f}")
            print(f"    Confidence: {mem.confidence.value:.2f}")

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())

```

**Run**:

```bash
pip install grpcio grpcio-tools
python python.py

```

### Rust

**File**: `rust.rs`

```rust
use engram::v1::{
    engram_service_client::EngramServiceClient,
    RememberRequest, RecallRequest, Memory, Confidence, Cue, SemanticCue,
};
use tonic::Request;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Connect to Engram
    let mut client = EngramServiceClient::connect("http://localhost:50051").await?;

    // REMEMBER: Store a memory with confidence
    let memory = Memory {
        content: "The Eiffel Tower is located in Paris, France".to_string(),
        embedding: vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
        confidence: Some(Confidence {
            value: 0.95,
            reasoning: "Verified geographical fact".to_string(),
            ..Default::default()
        }),
        ..Default::default()
    };

    let response = client.remember(Request::new(RememberRequest {
        memory_space_id: "default".to_string(),
        memory: Some(memory),
        auto_link: true,
        link_threshold: 0.7,
        ..Default::default()
    })).await?;

    let result = response.into_inner();
    println!("Stored memory: {}", result.memory_id);
    println!("Storage confidence: {:.2}", result.storage_confidence.unwrap().value);
    println!("Linked memories: {}", result.linked_memories.len());

    // RECALL: Retrieve memory with semantic cue
    let cue = Cue {
        cue_type: Some(engram::v1::cue::CueType::Semantic(SemanticCue {
            query: "Paris landmarks".to_string(),
            ..Default::default()
        })),
        ..Default::default()
    };

    let recall_response = client.recall(Request::new(RecallRequest {
        memory_space_id: "default".to_string(),
        cue: Some(cue),
        max_results: 5,
        ..Default::default()
    })).await?;

    let recall_result = recall_response.into_inner();
    println!("\nRecall confidence: {:.2}", recall_result.recall_confidence.unwrap().value);
    println!("Found {} memories:", recall_result.memories.len());

    for mem in recall_result.memories {
        println!("  - {}", mem.content);
        println!("    Activation: {:.2}", mem.activation);
        println!("    Confidence: {:.2}", mem.confidence.unwrap().value);
    }

    Ok(())
}

```

**Run**:

```bash
cargo run --release

```

### Complete examples for TypeScript, Go, and Java

See individual language files for full implementations.

## Expected Output

```
Stored memory: mem_1698765432_a1b2c3d4
Storage confidence: 0.98
Linked memories: 0

Recall confidence: 0.89
Found 1 memories:
  - The Eiffel Tower is located in Paris, France
    Activation: 0.87
    Confidence: 0.95

```

## Key Concepts Explained

### Confidence vs Activation

- **Confidence (0.95)**: How certain the memory is correct (intrinsic property)

- **Activation (0.87)**: How accessible the memory is right now (dynamic, context-dependent)

- **Recall confidence (0.89)**: How certain the retrieval operation succeeded

A memory can be highly confident (0.95) but low activation (0.3) if it hasn't been accessed recently.

### Auto-Linking

`auto_link: true` tells Engram to automatically find related memories and create connections. This enables spreading activation during future recalls.

**Without auto-link**: Memories are isolated
**With auto-link**: Memories form a graph, improving future recall

### Semantic Cues

The query "Paris landmarks" doesn't contain the exact words "Eiffel Tower", but Engram finds it via semantic similarity. This is different from SQL `LIKE '%paris%'` which requires exact substring matching.

## Common Errors

### ERR-1001: Embedding Dimension Mismatch

**Error**:

```
Expected 768 dimensions, received 8 dimensions

```

**Fix**:

```python
# Production: Use proper 768-dim embeddings
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('all-mpnet-base-v2')
embedding = model.encode("The Eiffel Tower is in Paris")  # 768-dim

```

### ERR-1002: Invalid Confidence Value

**Error**:

```
Confidence must be in range [0.0, 1.0], received: 1.5

```

**Fix**:

```python
# Wrong
confidence = Confidence(value=1.5)  # Out of range

# Correct
confidence = Confidence(value=0.95)  # In [0.0, 1.0]

```

### Connection Refused

**Error**:

```
grpc._channel._InactiveRpcError: connect failed

```

**Fix**:

```bash
# Check if Engram is running
curl http://localhost:8080/api/v1/introspect

# If not, start it
engram start --grpc-port 50051

```

## Variations to Try

### 1. Store Multiple Memories

```python
facts = [
    "The Eiffel Tower was built in 1889",
    "The Eiffel Tower is 330 meters tall",
    "Gustave Eiffel designed the Eiffel Tower"
]

for fact in facts:
    await client.Remember(RememberRequest(
        memory=Memory(content=fact, embedding=get_embedding(fact), confidence=...)
    ))

# Now recall - see auto-linking in action
response = await client.Recall(RecallRequest(
    cue=Cue(semantic=SemanticCue(query="Eiffel Tower"))
))
print(f"Found {len(response.memories)} linked memories")

```

### 2. Vary Confidence Levels

```python
# High confidence (verified fact)
memory1 = Memory(
    content="Paris is the capital of France",
    confidence=Confidence(value=0.99, reasoning="Official geographic fact")
)

# Medium confidence (news report)
memory2 = Memory(
    content="Paris population is about 2.1 million",
    confidence=Confidence(value=0.75, reasoning="Recent census estimate")
)

# Low confidence (heard from friend)
memory3 = Memory(
    content="Paris has the best croissants",
    confidence=Confidence(value=0.4, reasoning="Subjective opinion")
)

# Recall with high threshold - only get verified facts
response = await client.Recall(RecallRequest(
    cue=...,
    confidence_threshold=0.8  # Only memory1 and memory2 returned
))

```

### 3. Trace Activation Spread

```python
response = await client.Recall(RecallRequest(
    cue=Cue(semantic=SemanticCue(query="Paris")),
    trace_activation=True  # Enable activation tracing
))

# See how activation spread through the memory graph
for trace in response.traces:
    print(f"Memory {trace.memory_id}")
    print(f"  Activation level: {trace.activation_level:.2f}")
    print(f"  Path: {' → '.join(trace.activation_path)}")

```

## Next Steps

**After mastering this example:**

1. Try [02-episodic-memory](../02-episodic-memory/) - Add rich contextual information

2. Explore [04-pattern-completion](../04-pattern-completion/) - Reconstruct partial memories

3. Study [05-error-handling](../05-error-handling/) - Build production resilience

**Related documentation:**

- [REST API Reference](/reference/rest-api.md#core-memory-operations)

- [gRPC API Reference](/reference/grpc-api.md#remember)

- [Confidence Scoring Guide](/explanation/confidence-scoring.md)

## Time to Complete

- **First run**: 10 minutes (reading + running)

- **With variations**: 20 minutes (try all 3 variations)

- **Deep understanding**: 30 minutes (read code + docs)

## Success Criteria

You've mastered this example when you can:

- [ ] Store a memory with confidence and explain what confidence means

- [ ] Retrieve memories using semantic queries

- [ ] Explain the difference between confidence and activation

- [ ] Understand when auto-linking is beneficial

- [ ] Vary confidence thresholds to filter results

**Congratulations!** You now understand 80% of Engram's core functionality. The remember/recall cycle is the foundation for all other operations.
