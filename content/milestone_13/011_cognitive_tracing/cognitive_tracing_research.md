# Cognitive Tracing: Research and Technical Foundation

## Structured Event Tracing for Cognitive Operations

Traditional logging captures "what happened" at a code level. Cognitive tracing captures "what the memory system did" at a cognitive level. Instead of "spread_activation called with node 42," we want "semantic spreading from 'doctor' activated 'nurse' with strength 0.73 at +240ms."

This enables:
- Debugging cognitive patterns (why did this false memory form?)
- Validating temporal dynamics (did priming peak at expected SOA?)
- Visualizing memory operations (trace activation flow through graph)
- Performance analysis at cognitive granularity (which cognitive operations are slow?)

## Tracing Categories

**Category 1: Memory Operations**
- Encoding: what, when, strength, context
- Retrieval: cue, results, latency, confidence
- Consolidation: state transitions, timing
- Reconsolidation: window opening/closing, modifications

**Category 2: Cognitive Patterns**
- Priming: type (semantic/repetition/associative), magnitude, decay
- Interference: type (PI/RI/fan), competitors, reduction magnitude
- False memories: lure activation, source confusion
- Spacing: repetition intervals, benefit computation

**Category 3: Graph Operations**
- Spreading activation: paths, depths, activation values
- Pattern completion: partial cues, reconstructed patterns, confidence
- Similarity computation: dimensions, weights, final similarity

**Category 4: System Events**
- Consolidation pipeline: STM→LTM transfers
- Memory decay: strength adjustments over time
- Metrics collection: observation events (when metrics feature enabled)

## Event Structure

```rust
pub struct CognitiveEvent {
    pub timestamp: Instant,
    pub event_type: CognitiveEventType,
    pub metadata: HashMap<String, String>,
    pub span_id: Option<SpanId>,  // For hierarchical tracing
}

pub enum CognitiveEventType {
    Encoding { node_id: NodeId, strength: f32, context: Vec<NodeId> },
    Retrieval { cue: NodeId, results: Vec<(NodeId, f32)>, latency_us: u64 },

    PrimingActivation { prime: NodeId, target: NodeId, boost: f32, priming_type: PrimingType },
    InterferenceDetected { target: NodeId, competitors: Vec<NodeId>, reduction: f32, interference_type: InterferenceType },

    SpreadingStart { origin: NodeId, initial_activation: f32 },
    SpreadingHop { from: NodeId, to: NodeId, activation: f32, depth: usize },
    SpreadingComplete { total_activated: usize, max_depth: usize, duration_us: u64 },

    ConsolidationTransfer { node: NodeId, from_state: MemoryState, to_state: MemoryState },
    ReconsolidationWindow { node: NodeId, action: ReconWindowAction },

    FalseMemoryGenerated { lure: NodeId, activation_sources: Vec<NodeId>, confidence: f32 },
}

pub struct CognitiveSpan {
    pub span_id: SpanId,
    pub operation: String,
    pub start_time: Instant,
    pub end_time: Option<Instant>,
    pub events: Vec<CognitiveEvent>,
}
```

## Zero-Overhead Implementation

Like metrics (Task 001), tracing must be zero-overhead when disabled:

```rust
#[cfg(feature = "tracing")]
macro_rules! trace_cognitive {
    ($event:expr) => {
        COGNITIVE_TRACER.record($event);
    }
}

#[cfg(not(feature = "tracing"))]
macro_rules! trace_cognitive {
    ($event:expr) => {};
}

// Usage:
trace_cognitive!(CognitiveEvent {
    timestamp: Instant::now(),
    event_type: CognitiveEventType::PrimingActivation {
        prime: node_a,
        target: node_b,
        boost: 0.73,
        priming_type: PrimingType::Semantic,
    },
    metadata: HashMap::new(),
    span_id: None,
});
```

When feature disabled, macro expands to nothing, compiling away entirely.

## Lock-Free Event Recording

When enabled, use thread-local lock-free buffers:

```rust
thread_local! {
    static TRACE_BUFFER: RefCell<Vec<CognitiveEvent>> = RefCell::new(Vec::with_capacity(1000));
}

pub struct CognitiveTracer {
    global_buffer: Arc<Mutex<Vec<CognitiveEvent>>>,
    flush_interval: Duration,
}

impl CognitiveTracer {
    pub fn record(&self, event: CognitiveEvent) {
        TRACE_BUFFER.with(|buffer| {
            buffer.borrow_mut().push(event);

            // Flush to global if buffer full
            if buffer.borrow().len() >= 1000 {
                self.flush_thread_local();
            }
        });
    }

    fn flush_thread_local(&self) {
        TRACE_BUFFER.with(|buffer| {
            let mut global = self.global_buffer.lock().unwrap();
            global.append(&mut buffer.borrow_mut());
        });
    }
}
```

Thread-local buffers avoid contention. Periodic flushing batches lock acquisition.

## Export Formats

**Format 1: JSON Lines (for analysis)**
```json
{"timestamp": 1234567890, "event_type": "PrimingActivation", "prime": "doctor", "target": "nurse", "boost": 0.73}
{"timestamp": 1234568130, "event_type": "SpreadingHop", "from": "nurse", "to": "hospital", "activation": 0.58, "depth": 2}
```

**Format 2: OpenTelemetry (for distributed tracing)**
Compatible with Jaeger, Zipkin for visualizing cognitive operations in distributed systems.

**Format 3: Chrome Trace Format (for visualization)**
Can be loaded in chrome://tracing for interactive timeline visualization.

## Visualization Use Cases

**Use Case 1: DRM False Memory Formation**
Trace all spreading activation during study phase, showing how "bed," "rest," "tired" collectively activate "sleep" above retrieval threshold despite never being presented.

**Use Case 2: Interference Pattern**
Visualize competing activations during proactive interference, showing how List A items compete with List B items during retrieval.

**Use Case 3: Reconsolidation Window**
Timeline showing retrieval → lability window opening → potential modification → window closing → re-consolidation.

**Use Case 4: Priming Temporal Dynamics**
Plot priming boost magnitude over time from prime presentation, showing rise → plateau → decay matching Neely (1977) curves.

## Performance Budget

- Event creation: < 100ns (struct initialization)
- Thread-local push: < 50ns (vector append)
- Periodic flush: < 10μs (mutex + copy)
- Export to file: background thread, no impact on operations
- Total overhead when enabled: < 0.5%

## Integration with Metrics

Tracing and metrics are complementary:
- Metrics: aggregate statistics (counts, histograms, percentiles)
- Tracing: individual event streams (what happened, when, why)

Both use conditional compilation for zero overhead when disabled.

Combined, they provide complete observability: metrics for monitoring, tracing for debugging.
