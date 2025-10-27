# Task 011: Cognitive Tracing Infrastructure (CORRECTED)

**Status:** PENDING - REQUIRES COMPLETE RESPECIFICATION
**Priority:** P1
**Estimated Duration:** 5 days (increased from 2 days due to missing specification details)
**Dependencies:** Task 001 (Zero-Overhead Metrics) - MUST complete first
**Agent Review Required:** systems-architecture-optimizer (COMPLETED - see SYSTEMS_ARCHITECTURE_REVIEW.md)

---

## CRITICAL CORRECTIONS APPLIED

This is a complete rewrite of Task 011 based on systems architecture review findings.

**Original spec FAILED review due to:**
1. No bounded memory strategy (would leak 9 GB/hour)
2. DateTime overhead (200-500ns/event vs target <100ns)
3. Unbounded channel allocation
4. Prometheus misuse (it's for metrics, not events)
5. No specification of overhead when tracing ENABLED
6. Missing NodeId type definition
7. No discussion of allocation-free recording

**This corrected spec addresses all issues.**

---

## Overview

Implement structured event tracing for cognitive dynamics with bounded memory usage, minimal allocation overhead, and zero-cost when disabled. Events are stored in lock-free ring buffers with configurable sampling, then exported in standardized formats for debugging and production monitoring.

---

## Requirements

### Functional Requirements

**Event types to trace:**
- Priming events: type (semantic/associative/repetition), strength, source, target
- Interference events: type (proactive/retroactive/fan), magnitude, competing items
- Reconsolidation events: window position, modifications, plasticity factor
- False memory events: DRM critical lure generation, reconstruction details

**Export formats:**
- JSON: For external visualization tools and debugging
- OpenTelemetry (OTLP/gRPC): For distributed tracing integration
- Grafana Loki: For log aggregation and querying

**Configuration:**
- Enable/disable per event type
- Sampling rate (0.0 - 1.0)
- Ring buffer size (bounded memory)
- Export batch size and interval

---

### Performance Requirements (NEW - CRITICAL)

**When tracing DISABLED (feature flag off):**
- Overhead: 0ns (compiler eliminates all code)
- Binary size: No increase
- Memory: 0 bytes

**When tracing ENABLED but not configured (default):**
- Overhead: <10ns per potential event (one branch check)
- Memory: ~16 KB (ring buffer headers, no events)

**When tracing ENABLED and actively recording:**
- Overhead: <100ns per recorded event (P99)
- Memory: Bounded to ring_buffer_size × event_size
  - Default: 10,000 events × 64 bytes = 640 KB per thread
  - Max: Configurable, never exceeds specified limit
- Allocation: Zero allocations in hot path (pre-allocated ring buffers)

**Export overhead:**
- Batch export: <5ms for 10,000 events (async, background thread)
- No blocking of worker threads during export

---

## Architecture

### Ring Buffer Design

Use lock-free SPSC (single-producer, single-consumer) ring buffers per thread:

```
Thread 1: [RingBuffer 1] --\
Thread 2: [RingBuffer 2] ----> Collector Thread --> Export
Thread 3: [RingBuffer 3] --/
```

Benefits:
- Lock-free: No contention between threads
- Bounded: Fixed memory usage per thread
- Cache-friendly: Each thread owns its buffer (no false sharing)
- Drop policy: When full, drop oldest events (maintain recent history)

---

### Event Representation (CORRECTED)

**Fixed-size event struct to avoid allocations:**

```rust
use std::time::Instant;

/// Fixed-size node identifier (no heap allocation)
pub type NodeId = u64;

/// Cognitive event with minimal allocation overhead
///
/// Total size: 64 bytes (fits in 1 cache line)
#[repr(C, align(64))]
pub struct CognitiveEvent {
    /// Timestamp using monotonic clock (16 bytes)
    /// CORRECTED: Use Instant instead of DateTime<Utc> (40x faster)
    timestamp: Instant,

    /// Event type discriminant (1 byte)
    event_type: EventType,

    /// Event-specific data (47 bytes)
    data: EventData,
}

#[repr(u8)]
pub enum EventType {
    Priming = 0,
    Interference = 1,
    Reconsolidation = 2,
    FalseMemory = 3,
}

/// Union of event-specific data to maintain fixed size
///
/// All variants must fit in 47 bytes
#[repr(C)]
union EventData {
    priming: PrimingData,
    interference: InterferenceData,
    reconsolidation: ReconsolidationData,
    false_memory: FalseMemoryData,
}

#[repr(C)]
struct PrimingData {
    priming_type: PrimingType,     // 1 byte
    strength: f32,                  // 4 bytes
    source_node: NodeId,            // 8 bytes
    target_node: NodeId,            // 8 bytes
    _padding: [u8; 26],             // Pad to 47 bytes
}

// Similar for other event types...
```

**Key improvements:**
- Total size: 64 bytes (1 cache line)
- No heap allocations
- No String or Vec (all fixed-size)
- Instant timestamp: ~5ns overhead (vs 200-500ns for DateTime)

---

## Implementation Specifications

### File Structure

```
engram-core/src/tracing/
├── mod.rs                    # Public API and feature gates
├── event.rs                  # CognitiveEvent and type definitions
├── ring_buffer.rs            # Lock-free ring buffer implementation
├── collector.rs              # Background collection thread
├── config.rs                 # Configuration and sampling
└── exporters/
    ├── mod.rs
    ├── json.rs               # JSON export
    ├── otlp.rs               # OpenTelemetry OTLP/gRPC
    └── loki.rs               # Grafana Loki integration

engram-cli/src/
└── commands/tracing.rs       # CLI commands for tracing
```

---

### Core Tracing API

**File:** `/engram-core/src/tracing/mod.rs`

```rust
//! Cognitive event tracing with bounded memory and zero-cost abstraction
//!
//! When `tracing` feature is disabled, this module compiles to nothing.
//! When enabled, provides lock-free event recording with <100ns overhead.

#[cfg(feature = "tracing")]
mod event;
#[cfg(feature = "tracing")]
mod ring_buffer;
#[cfg(feature = "tracing")]
mod collector;
#[cfg(feature = "tracing")]
mod config;
#[cfg(feature = "tracing")]
pub mod exporters;

#[cfg(feature = "tracing")]
pub use event::{CognitiveEvent, EventType, PrimingType, InterferenceType};
#[cfg(feature = "tracing")]
pub use config::TracingConfig;

#[cfg(feature = "tracing")]
use ring_buffer::RingBuffer;
#[cfg(feature = "tracing")]
use std::sync::Arc;

/// Global tracing instance
#[cfg(feature = "tracing")]
pub struct CognitiveTracer {
    config: Arc<TracingConfig>,
    // Per-thread ring buffers
    buffers: dashmap::DashMap<std::thread::ThreadId, Arc<RingBuffer<CognitiveEvent>>>,
    // Background collector handle
    collector_handle: Option<std::thread::JoinHandle<()>>,
}

#[cfg(feature = "tracing")]
impl CognitiveTracer {
    /// Create new tracer with configuration
    pub fn new(config: TracingConfig) -> Self {
        let tracer = Self {
            config: Arc::new(config),
            buffers: dashmap::DashMap::new(),
            collector_handle: None,
        };

        // Start background collector thread
        let handle = collector::start_collector_thread(
            Arc::clone(&tracer.config),
            Arc::new(tracer.buffers.clone()),
        );

        Self {
            collector_handle: Some(handle),
            ..tracer
        }
    }

    /// Record priming event (zero-overhead when tracing disabled)
    #[inline(always)]
    pub fn trace_priming(
        &self,
        priming_type: PrimingType,
        strength: f32,
        source_node: NodeId,
        target_node: NodeId,
    ) {
        // Early return if this event type not enabled
        if !self.config.is_enabled(EventType::Priming) {
            return;
        }

        // Sampling check
        if !self.config.should_sample(EventType::Priming) {
            return;
        }

        // Get thread-local ring buffer
        let thread_id = std::thread::current().id();
        let buffer = self.buffers
            .entry(thread_id)
            .or_insert_with(|| Arc::new(RingBuffer::new(self.config.ring_buffer_size)));

        // Record event (lock-free push)
        let event = CognitiveEvent::new_priming(
            priming_type,
            strength,
            source_node,
            target_node,
        );

        buffer.push(event);
    }

    // Similar methods for other event types...
}

/// When tracing disabled, provide no-op implementations
#[cfg(not(feature = "tracing"))]
pub struct CognitiveTracer {
    _phantom: core::marker::PhantomData<()>,
}

#[cfg(not(feature = "tracing"))]
impl CognitiveTracer {
    #[inline(always)]
    pub fn trace_priming(
        &self,
        _priming_type: u8,
        _strength: f32,
        _source_node: u64,
        _target_node: u64,
    ) {
        // No-op when tracing disabled
    }
}
```

---

### Ring Buffer Implementation

**File:** `/engram-core/src/tracing/ring_buffer.rs`

```rust
//! Lock-free SPSC ring buffer for event storage
//!
//! Single producer (worker thread), single consumer (collector thread).
//! When full, drops oldest events to maintain bounded memory.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

pub struct RingBuffer<T> {
    /// Pre-allocated event storage
    buffer: Box<[Option<T>]>,
    /// Write position (producer)
    write_pos: AtomicUsize,
    /// Read position (consumer)
    read_pos: AtomicUsize,
    /// Buffer capacity (power of 2 for efficient modulo)
    capacity: usize,
}

impl<T: Copy> RingBuffer<T> {
    /// Create ring buffer with specified capacity
    ///
    /// Capacity is rounded up to next power of 2 for efficient modulo.
    pub fn new(capacity: usize) -> Self {
        let capacity = capacity.next_power_of_two();
        let mut buffer = Vec::with_capacity(capacity);
        buffer.resize_with(capacity, || None);

        Self {
            buffer: buffer.into_boxed_slice(),
            write_pos: AtomicUsize::new(0),
            read_pos: AtomicUsize::new(0),
            capacity,
        }
    }

    /// Push event (producer side)
    ///
    /// If buffer is full, overwrites oldest event (ring behavior).
    /// Returns true if event was recorded, false if dropped.
    #[inline]
    pub fn push(&self, event: T) -> bool {
        let write_pos = self.write_pos.load(Ordering::Relaxed);
        let read_pos = self.read_pos.load(Ordering::Acquire);

        let next_write = (write_pos + 1) % self.capacity;

        // Check if buffer is full
        if next_write == read_pos {
            // Buffer full - drop oldest event by advancing read position
            self.read_pos.store((read_pos + 1) % self.capacity, Ordering::Release);
        }

        // Write event
        // SAFETY: write_pos is exclusively owned by producer thread
        unsafe {
            let slot = &self.buffer[write_pos] as *const Option<T> as *mut Option<T>;
            *slot = Some(event);
        }

        // Advance write position
        self.write_pos.store(next_write, Ordering::Release);

        true
    }

    /// Pop event (consumer side)
    ///
    /// Returns None if buffer is empty.
    #[inline]
    pub fn pop(&self) -> Option<T> {
        let read_pos = self.read_pos.load(Ordering::Relaxed);
        let write_pos = self.write_pos.load(Ordering::Acquire);

        // Check if buffer is empty
        if read_pos == write_pos {
            return None;
        }

        // Read event
        // SAFETY: read_pos is exclusively owned by consumer thread
        let event = unsafe {
            let slot = &self.buffer[read_pos] as *const Option<T>;
            (*slot).clone()
        };

        // Advance read position
        self.read_pos.store((read_pos + 1) % self.capacity, Ordering::Release);

        event
    }

    /// Get number of events currently in buffer
    pub fn len(&self) -> usize {
        let write_pos = self.write_pos.load(Ordering::Acquire);
        let read_pos = self.read_pos.load(Ordering::Acquire);

        if write_pos >= read_pos {
            write_pos - read_pos
        } else {
            self.capacity - read_pos + write_pos
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

// SAFETY: RingBuffer is thread-safe for SPSC pattern
unsafe impl<T: Send> Send for RingBuffer<T> {}
unsafe impl<T: Send> Sync for RingBuffer<T> {}
```

---

### Configuration

**File:** `/engram-core/src/tracing/config.rs`

```rust
use crate::tracing::EventType;
use std::collections::HashSet;
use rand::{thread_rng, Rng};

pub struct TracingConfig {
    /// Which event types to trace
    pub enabled_events: HashSet<EventType>,

    /// Sampling rate per event type (0.0 - 1.0)
    pub sample_rates: std::collections::HashMap<EventType, f32>,

    /// Ring buffer size per thread
    pub ring_buffer_size: usize,

    /// Export batch size
    pub export_batch_size: usize,

    /// Export interval (milliseconds)
    pub export_interval_ms: u64,

    /// Export format
    pub export_format: ExportFormat,
}

#[derive(Debug, Clone, Copy)]
pub enum ExportFormat {
    Json,
    OtlpGrpc,
    Loki,
}

impl TracingConfig {
    /// Default configuration: disabled
    pub fn disabled() -> Self {
        Self {
            enabled_events: HashSet::new(),
            sample_rates: std::collections::HashMap::new(),
            ring_buffer_size: 0,
            export_batch_size: 0,
            export_interval_ms: 0,
            export_format: ExportFormat::Json,
        }
    }

    /// Development configuration: trace everything
    pub fn development() -> Self {
        let mut config = Self::disabled();
        config.enabled_events.insert(EventType::Priming);
        config.enabled_events.insert(EventType::Interference);
        config.enabled_events.insert(EventType::Reconsolidation);
        config.enabled_events.insert(EventType::FalseMemory);

        config.sample_rates.insert(EventType::Priming, 1.0);
        config.sample_rates.insert(EventType::Interference, 1.0);
        config.sample_rates.insert(EventType::Reconsolidation, 1.0);
        config.sample_rates.insert(EventType::FalseMemory, 1.0);

        config.ring_buffer_size = 10_000;
        config.export_batch_size = 1_000;
        config.export_interval_ms = 5_000;

        config
    }

    /// Production configuration: sampled tracing
    pub fn production() -> Self {
        let mut config = Self::disabled();
        config.enabled_events.insert(EventType::Priming);
        config.enabled_events.insert(EventType::Interference);

        // Sample 1% of events in production
        config.sample_rates.insert(EventType::Priming, 0.01);
        config.sample_rates.insert(EventType::Interference, 0.01);

        config.ring_buffer_size = 10_000;
        config.export_batch_size = 5_000;
        config.export_interval_ms = 30_000; // 30 seconds

        config
    }

    /// Check if event type is enabled
    #[inline]
    pub fn is_enabled(&self, event_type: EventType) -> bool {
        self.enabled_events.contains(&event_type)
    }

    /// Check if event should be sampled (stochastic sampling)
    #[inline]
    pub fn should_sample(&self, event_type: EventType) -> bool {
        let rate = self.sample_rates.get(&event_type).copied().unwrap_or(0.0);

        if rate >= 1.0 {
            return true;
        }

        if rate <= 0.0 {
            return false;
        }

        // Reservoir sampling
        thread_rng().gen::<f32>() < rate
    }
}
```

---

### JSON Exporter

**File:** `/engram-core/src/tracing/exporters/json.rs`

```rust
use crate::tracing::CognitiveEvent;
use serde_json::{json, Value};
use std::time::{SystemTime, UNIX_EPOCH};

pub struct JsonExporter {
    /// Process start time for converting Instant to absolute timestamp
    process_start: SystemTime,
}

impl JsonExporter {
    pub fn new() -> Self {
        Self {
            process_start: SystemTime::now(),
        }
    }

    /// Export batch of events to JSON
    pub fn export_batch(&self, events: &[CognitiveEvent]) -> Result<String, serde_json::Error> {
        let json_events: Vec<Value> = events
            .iter()
            .map(|event| self.event_to_json(event))
            .collect();

        serde_json::to_string(&json!({
            "events": json_events,
            "exported_at": SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_secs(),
        }))
    }

    fn event_to_json(&self, event: &CognitiveEvent) -> Value {
        // Convert Instant to absolute timestamp
        let elapsed = event.timestamp.duration_since(self.process_start);
        let timestamp = self.process_start + elapsed;

        // ... serialize event fields to JSON
        // (implementation details omitted for brevity)

        json!({
            "timestamp": timestamp.duration_since(UNIX_EPOCH).unwrap().as_secs(),
            "event_type": format!("{:?}", event.event_type),
            // ... event-specific fields
        })
    }
}
```

---

### CLI Commands

**File:** `/engram-cli/src/commands/tracing.rs`

```bash
# Export events to JSON
engram trace export --format json --output events.json

# Export last 1 hour of events
engram trace export --format json --time-range 1h --output recent.json

# Stream events in real-time
engram trace stream --format json

# Configure tracing
engram trace config --enable priming --sample-rate 0.1

# Show tracing statistics
engram trace stats
```

---

## Integration with Existing Tracing Crate (ALTERNATIVE)

**Justification for custom implementation vs Rust tracing crate:**

The Rust `tracing` crate is excellent for general-purpose logging but has limitations for our use case:

1. **Overhead:** Even with filtering, `tracing` has ~50-100ns overhead per span
2. **Memory:** Unbounded subscriber buffers can grow indefinitely
3. **Fixed schema:** Cognitive events have specific structure we need to preserve
4. **Export formats:** Need specialized OpenTelemetry semantic conventions

**However, we can INTEGRATE with tracing:**

```rust
// Use tracing for general logs, custom tracer for cognitive events
use tracing::{event, Level};

pub fn trace_priming_hybrid(/* ... */) {
    // Our custom high-performance tracer
    COGNITIVE_TRACER.trace_priming(/* ... */);

    // Also emit to standard tracing (for general debugging)
    event!(Level::DEBUG,
        priming_type = ?priming_type,
        strength = strength,
        "cognitive_priming_event"
    );
}
```

This gives us:
- Custom tracer for production (low overhead, bounded memory)
- Standard tracing for development (rich ecosystem, tooling)

---

## Acceptance Criteria

### Must Have

- [ ] Zero overhead when `tracing` feature disabled (verified via size_of tests)
- [ ] <100ns overhead when tracing enabled and recording (Criterion benchmark)
- [ ] Bounded memory usage (ring buffer size × event size × thread count)
- [ ] JSON export working with correct schema
- [ ] CLI export command functional
- [ ] Events include all required fields with fixed-size representation
- [ ] No allocations in hot path (verified via allocation profiler)

### Should Have

- [ ] OpenTelemetry OTLP/gRPC export
- [ ] Grafana Loki integration
- [ ] Configurable sampling rates per event type
- [ ] Background collector thread with graceful shutdown
- [ ] Event filtering by type/time range
- [ ] Memory overhead <1 MB per worker thread

### Nice to Have

- [ ] Real-time event streaming API
- [ ] Event correlation across types
- [ ] Compression for exported events
- [ ] Integration with standard `tracing` crate for dual logging

---

## Testing Strategy

```bash
# 1. Zero-overhead verification
cargo test --lib --no-default-features tracing::zero_overhead
# Expected: size_of::<CognitiveTracer>() == 0

# 2. Overhead benchmarks
cargo bench --bench tracing_overhead --features tracing
# Expected: <100ns per event record (P99)

# 3. Memory bounds test
cargo test --lib --features tracing tracing::memory_bounds
# Expected: Memory usage stable over 1M events

# 4. Export correctness
cargo test --lib --features tracing tracing::exporters
# Expected: All export formats produce valid output

# 5. Ring buffer stress test
cargo test --lib --features tracing ring_buffer::stress_test -- --ignored
# Expected: No data loss, correct event ordering
```

---

## Memory Overhead Calculation

**Per-thread overhead:**
- Ring buffer: 10,000 events × 64 bytes = 640 KB
- Metadata: ~16 KB (buffer indices, config)
- **Total: ~656 KB per thread**

**For 10 worker threads: ~6.5 MB**

**Export buffer (collector thread):**
- Batch size: 5,000 events × 64 bytes = 320 KB
- JSON serialization buffer: ~1 MB (temporary)
- **Total: ~1.3 MB**

**Global overhead: ~8 MB for typical deployment**

This is acceptable for production systems with GB-scale memory.

---

## Performance Validation

**Benchmark results (expected):**

```
tracing_overhead/record_priming
                        time:   [45.2 ns 47.8 ns 50.1 ns]
                        thrpt:  [19.9 M/s 20.9 M/s 22.1 M/s]

tracing_overhead/disabled_noop
                        time:   [0.12 ns 0.14 ns 0.16 ns]
                        (essentially zero - compiler optimization)

memory_bounds/1M_events
                        peak memory: 656 KB (expected)
                        final memory: 656 KB (no growth)
```

---

## Migration Path

**Phase 1:** Implement core infrastructure (ring buffers, event types)
**Phase 2:** Add JSON exporter and CLI commands
**Phase 3:** Integrate with cognitive modules (priming, interference, etc.)
**Phase 4:** Add OpenTelemetry and Loki exporters
**Phase 5:** Production validation and tuning

---

## References

1. Ring buffer design: https://www.snellman.net/blog/archive/2016-12-13-ring-buffers/
2. OpenTelemetry specification: https://opentelemetry.io/docs/specs/otel/
3. Grafana Loki LogQL: https://grafana.com/docs/loki/latest/logql/
4. Lock-free SPSC queues: Dmitry Vyukov, "Bounded MPMC queue"

---

**CRITICAL:** This specification must be reviewed and approved before implementation begins. Do NOT start coding until architecture is validated.
