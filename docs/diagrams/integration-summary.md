# Diagram Integration Summary

This document tracks where architecture diagrams are referenced throughout Engram's documentation.

## Created Diagrams

### Milestone 11: Streaming Architecture

1. **ObservationQueue Flow** (`observation-queue-flow.md`)
   - Complete observation ingestion pipeline
   - Session management and state machines
   - Queue priority lanes and worker pool
   - Generation tracking and snapshot isolation

2. **Space-Partitioned HNSW** (`space-partitioned-hnsw.md`)
   - Multi-tenant zero-contention architecture
   - Worker assignment and work stealing
   - Performance scaling analysis
   - Memory layout and footprint

3. **Backpressure Mechanism** (`backpressure-mechanism.md`)
   - Adaptive admission control
   - State machine and thresholds
   - Client-side handling strategies
   - Performance characteristics

### Milestone 13: Cognitive Patterns

4. **Cognitive Patterns Flow** (`cognitive-patterns-flow.md`)
   - Semantic priming (Collins & Loftus 1975)
   - Proactive/retroactive interference
   - Fan effect (Anderson 1974)
   - Reconsolidation (Nader et al. 2000)

5. **Memory Consolidation Pipeline** (`memory-consolidation-pipeline.md`)
   - Encoding to long-term storage
   - State transitions and decay rates
   - Pattern extraction and schema formation
   - Biological analogies

## Documentation Integration

### Referenced In

#### `/docs/reference/streaming-performance-analysis.md`
**Updated**: Added links to streaming architecture diagrams
```markdown
## Architecture Diagrams

For visual understanding of the streaming architecture, see:
- [ObservationQueue Flow](../diagrams/observation-queue-flow.md)
- [Space-Partitioned HNSW](../diagrams/space-partitioned-hnsw.md)
- [Backpressure Mechanism](../diagrams/backpressure-mechanism.md)
```

### Recommended Additions

The following documentation files would benefit from diagram references:

#### Operations Documentation

**File**: `docs/operations/streaming.md` (to be created)
**Content**: Production deployment guide for streaming infrastructure
**Diagrams**:
- ObservationQueue Flow (understanding the pipeline)
- Space-Partitioned HNSW (capacity planning)
- Backpressure Mechanism (monitoring and alerting)

**Example Integration**:
```markdown
## Streaming Architecture

Engram's streaming infrastructure processes observations through a multi-stage
pipeline. For a visual overview, see the [ObservationQueue Flow diagram](../diagrams/observation-queue-flow.md).

### Key Components

1. **Session Management**: Each client maintains a session with monotonic
   sequence validation. See the session lifecycle in the flow diagram.

2. **Priority Queues**: Three priority lanes (High/Normal/Low) handle different
   observation types. Queue depth monitoring triggers backpressure as shown
   in the [Backpressure Mechanism diagram](../diagrams/backpressure-mechanism.md).
```

#### How-To Guides

**File**: `docs/howto/backpressure-handling.md` (to be created)
**Content**: Client implementation guide for handling backpressure
**Diagrams**:
- Backpressure Mechanism (state transitions)
- ObservationQueue Flow (ack/status messages)

**Example Integration**:
```markdown
## Understanding Backpressure States

Engram's backpressure system transitions through four states based on queue
depth. The [Backpressure Mechanism diagram](../diagrams/backpressure-mechanism.md)
shows the complete state machine and thresholds.

### Client Implementation

When queue depth exceeds 85%, the server sends a `StreamStatus` message
with `state=BACKPRESSURE`. Clients should pause sending observations:

\```python
async def handle_stream_status(status: StreamStatus):
    if status.state == BackpressureState.BACKPRESSURE:
        # Pause for retry_after duration
        await asyncio.sleep(status.retry_after_ms / 1000)
\```

See the backpressure diagram for complete flow sequence.
```

#### Explanation Documentation

**File**: `docs/explanation/cognitive-patterns.md` (to be created)
**Content**: Theoretical foundations of cognitive patterns
**Diagrams**:
- Cognitive Patterns Flow (all patterns overview)
- Memory Consolidation Pipeline (lifecycle)

**Example Integration**:
```markdown
## Semantic Priming

When "doctor" is recalled, related concepts like "nurse" and "hospital"
receive temporary activation boost. This implements Collins & Loftus (1975)
spreading activation theory.

See the [Cognitive Patterns Flow diagram](../diagrams/cognitive-patterns-flow.md)
for a visual explanation of the priming timeline and decay curve.

### Empirical Validation

Our implementation targets the following empirical benchmarks from Neely (1977):
- Automatic spreading activation: < 400ms timescale
- RT reduction: 10-20% for semantically related targets
- Decay half-life: ~300ms

The priming section of the cognitive patterns diagram shows the complete
activation spread and decay timeline with empirically-grounded parameters.
```

#### API Reference

**File**: `docs/reference/priming-api.md` (to be created)
**Content**: API documentation for priming engine
**Diagrams**:
- Cognitive Patterns Flow (priming section)

**Example Integration**:
```markdown
## SemanticPrimingEngine

\```rust
pub struct SemanticPrimingEngine {
    priming_strength: f32,      // Default: 0.15 (15% boost)
    decay_half_life: Duration,  // Default: 300ms
    // ...
}
\```

### Activation Spread

When a memory is recalled, the engine spreads activation to semantic neighbors
as shown in the [Cognitive Patterns Flow diagram](../diagrams/cognitive-patterns-flow.md#semantic-priming).

The spreading algorithm uses cosine similarity to weight activation:

\```rust
spread = base_activation × similarity × priming_strength
\```
```

**File**: `docs/reference/reconsolidation-api.md` (to be created)
**Content**: API documentation for reconsolidation engine
**Diagrams**:
- Cognitive Patterns Flow (reconsolidation section)
- Memory Consolidation Pipeline (plasticity window)

**Example Integration**:
```markdown
## ReconsolidationEngine

Memory reconsolidation allows modification of consolidated memories within
a temporal window post-recall. See the [Cognitive Patterns Flow diagram](../diagrams/cognitive-patterns-flow.md#reconsolidation)
for the complete timeline and boundary conditions.

### Plasticity Window

The reconsolidation window opens 1 hour post-recall and closes after 6 hours,
with peak plasticity at 3-4 hours. This matches the protein synthesis kinetics
from Nader & Einarsson (2010).

\```rust
pub struct ReconsolidationEngine {
    window_start: Duration,    // Default: 1 hour
    window_end: Duration,      // Default: 6 hours
    max_plasticity: f32,       // Default: 0.5
}
\```

The [Memory Consolidation Pipeline diagram](../diagrams/memory-consolidation-pipeline.md#reconsolidation)
shows the inverted-U plasticity curve and re-stabilization process.
```

#### Tutorial Documentation

**File**: `docs/tutorials/streaming-client.md` (to be created)
**Content**: Step-by-step guide to building a streaming client
**Diagrams**:
- ObservationQueue Flow (understanding the handshake)
- Backpressure Mechanism (handling errors)

**Example Integration**:
```markdown
## Step 3: Handling Backpressure

As you send observations, the server may respond with backpressure signals
when queue capacity is reached. The [Backpressure Mechanism diagram](../diagrams/backpressure-mechanism.md)
explains the four states and recovery flow.

### Implementation

\```python
while observations:
    try:
        response = await client.send(observation)
        if response.status == "BACKPRESSURE":
            # Server queue is full, pause sending
            await asyncio.sleep(response.retry_after / 1000)
            continue
    except ResourceExhausted:
        # Critical state, exponential backoff
        await exponential_backoff()
\```

See the backpressure diagram's "Client-Side Backpressure Handling" section
for complete pseudocode and retry strategies.
```

## Maintenance Checklist

When updating diagrams or creating new documentation:

- [ ] Update diagram file (Mermaid + ASCII)
- [ ] Update diagram README with new content
- [ ] Add references in relevant documentation
- [ ] Update this integration summary
- [ ] Test Mermaid rendering (GitHub, VitePress)
- [ ] Verify ASCII alignment in terminal
- [ ] Update "Last Updated" dates
- [ ] Commit with descriptive message

## Future Diagram Needs

Potential diagrams for future milestones:

### Milestone 12: GPU Acceleration
- CUDA kernel execution pipeline
- CPU/GPU memory transfer flow
- Batch embedding computation
- SIMD vectorization diagrams

### Milestone 14: Distribution
- Shard assignment algorithm
- Gossip protocol message flow
- Partition-tolerant operations
- Cross-shard query execution

### Milestone 15: Production Systems
- Monitoring architecture
- Query language parsing pipeline
- Debugging tool workflow
- Performance regression framework

## Diagram Quality Standards

All diagrams should include:

### Technical Accuracy
- [ ] Implementation matches diagram
- [ ] Performance numbers validated by benchmarks
- [ ] Empirical references cited (cognitive patterns)
- [ ] Boundary conditions explicitly stated

### Visual Clarity
- [ ] Consistent styling and colors
- [ ] Clear component labels
- [ ] Readable at standard screen sizes
- [ ] Legend for complex symbols

### Documentation Value
- [ ] Concrete examples with realistic values
- [ ] Step-by-step flow sequences
- [ ] Performance characteristics tables
- [ ] Integration with code examples

### Accessibility
- [ ] Both Mermaid and ASCII versions
- [ ] Alt text for complex diagrams
- [ ] High contrast colors
- [ ] No information conveyed by color alone

## Contact

For diagram updates or requests, see the project's contribution guidelines.
