# Task 006: Backpressure and Admission Control

**Status:** pending
**Estimated Effort:** 2 days
**Dependencies:** Task 005 (gRPC Streaming), Task 003 (Worker Pool for adaptive batching)
**Blocks:** Task 010 (Performance Benchmarking)

## Objective

Implement adaptive backpressure with flow control messages, server-side admission control, and adaptive batching under load to sustain 100K observations/sec without OOM.

## Background

**Current State:**
- `ObservationQueue::should_apply_backpressure()` detects pressure (> 80% capacity)
- `ObservationQueue::enqueue()` returns `QueueError::OverCapacity` when full
- `StreamStatus` message defined in protobuf

**Critical Gap:**
- No proactive emission of flow control to clients
- No adaptive batching (fixed batch size regardless of load)
- No retry-after calculation for admission control

**Performance Target:** Prevent queue growth beyond capacity while sustaining 100K obs/sec under optimal conditions.

## Implementation Specification

### Files to Create

1. **`engram-core/src/streaming/backpressure.rs`** (~300 lines)

Core components:
```rust
pub enum BackpressureState {
    Normal,        // < 50% capacity
    Warning,       // 50-80% capacity
    Critical,      // 80-95% capacity
    Overloaded,    // > 95% capacity
}

impl BackpressureState {
    pub fn from_pressure(pressure: f32) -> Self { /* ... */ }

    pub fn recommended_batch_size(&self) -> usize {
        match self {
            Self::Normal => 10,        // Low latency (10ms)
            Self::Warning => 100,      // Balanced (20ms)
            Self::Critical => 500,     // High throughput (50ms)
            Self::Overloaded => 1000,  // Maximum throughput (100ms)
        }
    }
}

pub struct BackpressureMonitor {
    observation_queue: Arc<ObservationQueue>,
    state_tx: broadcast::Sender<BackpressureState>,
    check_interval: Duration,
}

impl BackpressureMonitor {
    pub fn new(queue: Arc<ObservationQueue>, interval: Duration) -> Self;
    pub fn subscribe(&self) -> broadcast::Receiver<BackpressureState>;
    pub async fn run(&self); // Background monitoring loop
}

pub fn calculate_retry_after(queue_depth: usize, dequeue_rate: f32) -> Duration;
```

**Key Algorithm:**
```rust
pub async fn run(&self) {
    let mut current_state = BackpressureState::Normal;
    let mut interval = tokio::time::interval(self.check_interval);

    loop {
        interval.tick().await;

        let total_depth = self.observation_queue.total_depth();
        let total_capacity = self.observation_queue.total_capacity();
        let pressure = total_depth as f32 / total_capacity as f32;

        let new_state = BackpressureState::from_pressure(pressure);

        if new_state != current_state {
            // State changed - notify all subscribers
            let _ = self.state_tx.send(new_state);
            current_state = new_state;

            tracing::info!(
                "Backpressure state: {:?} (pressure: {:.1}%)",
                new_state,
                pressure * 100.0
            );
        }
    }
}
```

### Files to Modify

1. **`engram-cli/src/handlers/streaming.rs`**
   - Subscribe to backpressure monitor in `handle_observe_stream()`
   - Forward `StreamStatus::BACKPRESSURE` to active streams
   - Calculate retry-after on admission control rejection

2. **`engram-core/src/streaming/worker_pool.rs`** (Task 003 - future integration)
   - Use `BackpressureState::recommended_batch_size()` for adaptive batching
   - Adjust batch size based on current pressure

3. **`engram-core/src/streaming/mod.rs`**
   - Export backpressure module

## Detailed Implementation

### 1. Backpressure Monitor

**Purpose:** Periodically check queue depth and emit state changes to active streams

**Implementation:**
```rust
// In backpressure.rs

use tokio::sync::broadcast;
use std::time::Duration;

const DEFAULT_CHECK_INTERVAL: Duration = Duration::from_millis(100);

impl BackpressureMonitor {
    pub fn new(
        observation_queue: Arc<ObservationQueue>,
        check_interval: Duration,
    ) -> Self {
        let (state_tx, _) = broadcast::channel(32);
        Self {
            observation_queue,
            state_tx,
            check_interval,
        }
    }

    pub fn subscribe(&self) -> broadcast::Receiver<BackpressureState> {
        self.state_tx.subscribe()
    }

    // Main monitoring loop - spawn as background task
    pub async fn run(&self) {
        // Implementation shown above
    }
}
```

**Integration in streaming handler:**
```rust
// In streaming.rs

pub async fn handle_observe_stream(/* ... */) -> Result</* ... */> {
    // Create backpressure monitor
    let monitor = BackpressureMonitor::new(
        Arc::clone(&observation_queue),
        Duration::from_millis(100),
    );

    // Subscribe to state changes
    let mut backpressure_rx = monitor.subscribe();

    // Spawn monitor task
    tokio::spawn(async move { monitor.run().await });

    // Spawn task to forward backpressure state to client
    let tx_clone = tx.clone();
    let session_id_clone = session_id.clone();
    tokio::spawn(async move {
        while let Ok(state) = backpressure_rx.recv().await {
            let status = StreamStatus {
                state: match state {
                    BackpressureState::Normal => StreamState::Active,
                    BackpressureState::Warning => StreamState::Active,
                    BackpressureState::Critical => StreamState::Backpressure,
                    BackpressureState::Overloaded => StreamState::Overloaded,
                } as i32,
                message: format!("Queue pressure: {:?}", state),
                queue_depth: observation_queue.total_depth() as u32,
                queue_capacity: observation_queue.total_capacity() as u32,
                pressure: observation_queue.total_depth() as f32
                         / observation_queue.total_capacity() as f32,
            };

            let response = ObservationResponse {
                result: Some(observation_response::Result::Status(status)),
                session_id: session_id_clone.clone(),
                sequence_number: 0,
                server_timestamp: Some(prost_types::Timestamp::from(SystemTime::now())),
            };

            if tx_clone.send(Ok(response)).await.is_err() {
                break; // Client disconnected
            }
        }
    });

    // ... rest of stream handling ...
}
```

### 2. Admission Control with Retry-After

**Purpose:** Reject observations when queue full, provide accurate retry guidance

**Implementation:**
```rust
// In backpressure.rs

pub fn calculate_retry_after(
    queue_depth: usize,
    dequeue_rate: f32, // observations per second
) -> Duration {
    if dequeue_rate < 1.0 {
        return Duration::from_secs(60); // Pessimistic fallback
    }

    // Calculate time to drain to 50% capacity
    let target_depth = queue_depth / 2;
    let excess = queue_depth.saturating_sub(target_depth);
    let drain_seconds = excess as f32 / dequeue_rate;

    // Cap at 5 minutes to avoid unreasonable waits
    Duration::from_secs_f32(drain_seconds.min(300.0))
}
```

**Integration in observation enqueue:**
```rust
// In streaming.rs

match observation_queue.enqueue(
    memory_space_id,
    episode,
    sequence_number,
    ObservationPriority::Normal,
) {
    Ok(()) => {
        // Send ObservationAck with ACCEPTED status
        send_observation_ack(/* ... */).await;
    }
    Err(QueueError::OverCapacity { current, limit, priority }) => {
        // Calculate retry-after based on current drain rate
        let dequeue_rate = metrics.get_dequeue_rate_per_second();
        let retry_after = backpressure::calculate_retry_after(current, dequeue_rate);

        // Return RESOURCE_EXHAUSTED with retry guidance
        return Err(Status::resource_exhausted(format!(
            "Queue capacity exceeded for {:?} priority: {}/{} items. Retry after {}s",
            priority, current, limit, retry_after.as_secs()
        )));
    }
}
```

### 3. Adaptive Batching (Integration with Task 003)

**Purpose:** Adjust batch size based on queue pressure to trade latency for throughput

**Implementation in Worker Pool:**
```rust
// In worker_pool.rs (Task 003)

pub struct HnswWorker {
    observation_queue: Arc<ObservationQueue>,
    backpressure_monitor: Arc<BackpressureMonitor>,
    // ... other fields ...
}

impl HnswWorker {
    async fn process_batch(&self) {
        // Subscribe to backpressure state
        let mut pressure_rx = self.backpressure_monitor.subscribe();
        let mut current_batch_size = 10; // Default: low latency

        loop {
            // Update batch size based on latest pressure state
            if let Ok(state) = pressure_rx.try_recv() {
                current_batch_size = state.recommended_batch_size();
                tracing::debug!("Adjusted batch size to {}", current_batch_size);
            }

            // Dequeue batch with adaptive size
            let batch = self.observation_queue.dequeue_batch(current_batch_size);

            if batch.is_empty() {
                tokio::time::sleep(Duration::from_millis(10)).await;
                continue;
            }

            // Process batch (HNSW insertions)
            self.insert_batch(&batch).await;
        }
    }
}
```

## Testing Strategy

### Unit Tests

```rust
#[test]
fn test_backpressure_state_thresholds() {
    assert_eq!(BackpressureState::from_pressure(0.3), BackpressureState::Normal);
    assert_eq!(BackpressureState::from_pressure(0.6), BackpressureState::Warning);
    assert_eq!(BackpressureState::from_pressure(0.85), BackpressureState::Critical);
    assert_eq!(BackpressureState::from_pressure(0.98), BackpressureState::Overloaded);
}

#[test]
fn test_adaptive_batch_sizing() {
    assert_eq!(BackpressureState::Normal.recommended_batch_size(), 10);
    assert_eq!(BackpressureState::Warning.recommended_batch_size(), 100);
    assert_eq!(BackpressureState::Critical.recommended_batch_size(), 500);
    assert_eq!(BackpressureState::Overloaded.recommended_batch_size(), 1000);
}

#[test]
fn test_retry_after_calculation() {
    // Dequeue rate: 1000 obs/sec
    // Queue depth: 10000
    // Target: 5000
    // Excess: 5000
    // Time: 5 seconds
    let retry = calculate_retry_after(10_000, 1000.0);
    assert_eq!(retry, Duration::from_secs(5));
}

#[tokio::test]
async fn test_backpressure_monitor_state_changes() {
    let queue = Arc::new(ObservationQueue::new(QueueConfig {
        high_capacity: 100,
        normal_capacity: 100,
        low_capacity: 100,
    }));

    let monitor = BackpressureMonitor::new(
        Arc::clone(&queue),
        Duration::from_millis(10),
    );

    let mut rx = monitor.subscribe();

    tokio::spawn(async move { monitor.run().await });

    // Fill queue to 60% - should trigger Warning
    for i in 0..180 {
        queue.enqueue(
            MemorySpaceId::default(),
            test_episode(),
            i,
            ObservationPriority::Normal,
        ).unwrap();
    }

    // Verify state change within 100ms
    let state = tokio::time::timeout(Duration::from_millis(100), rx.recv())
        .await
        .unwrap()
        .unwrap();

    assert!(matches!(state, BackpressureState::Warning));
}
```

### Integration Tests

```rust
#[tokio::test]
async fn test_admission_control_rejects_when_full() {
    let (client, queue) = setup_test_streaming_service().await;

    // Fill queue to capacity
    for i in 0..queue.total_capacity() {
        client.observe(test_episode()).await.unwrap();
    }

    // Next observation should be rejected
    let result = client.observe(test_episode()).await;
    assert!(result.is_err());

    let status = result.unwrap_err();
    assert_eq!(status.code(), tonic::Code::ResourceExhausted);
    assert!(status.message().contains("Retry after"));
}

#[tokio::test]
async fn test_backpressure_notifies_active_streams() {
    let (client, queue) = setup_test_streaming_service().await;

    // Start streaming session
    let (tx, rx) = mpsc::channel(10);
    let mut response_stream = client.observe_stream(ReceiverStream::new(rx))
        .await
        .unwrap()
        .into_inner();

    // Fill queue to 85%
    for i in 0..(queue.total_capacity() * 85 / 100) {
        tx.send(observation_request(i)).await.unwrap();
    }

    // Client should receive StreamStatus::BACKPRESSURE
    let mut received_backpressure = false;
    while let Some(resp) = response_stream.next().await {
        let resp = resp.unwrap();
        if let Some(observation_response::Result::Status(status)) = resp.result {
            if status.state == StreamState::Backpressure as i32 {
                received_backpressure = true;
                break;
            }
        }
    }

    assert!(received_backpressure, "Client should receive backpressure notification");
}
```

### Performance Tests

```rust
#[tokio::test]
async fn test_monitor_overhead() {
    let queue = Arc::new(ObservationQueue::new(QueueConfig::default()));
    let monitor = BackpressureMonitor::new(queue, Duration::from_millis(1));

    // Measure CPU usage of monitor loop
    let start = Instant::now();
    tokio::spawn(async move { monitor.run().await });

    // Run for 10 seconds
    tokio::time::sleep(Duration::from_secs(10)).await;

    // Monitor overhead should be < 0.1% CPU
    // (Implementation note: measure via external profiler)
}

#[tokio::test]
async fn test_adaptive_batching_improves_throughput() {
    // Measure throughput at different pressure levels
    // Normal (batch=10): ~10K obs/sec
    // Warning (batch=100): ~50K obs/sec
    // Critical (batch=500): ~100K obs/sec
}
```

## Acceptance Criteria

### Functional
- [ ] Backpressure monitor detects state changes within 100ms
- [ ] Active streams receive `StreamStatus` when pressure > 80%
- [ ] Admission control rejects enqueue when queue at capacity
- [ ] Rejected observations receive retry-after estimate
- [ ] Retry-after calculation reflects actual drain rate (± 20% accuracy)

### Performance
- [ ] Monitor overhead < 0.1% CPU at all load levels
- [ ] State change notification latency < 10ms (monitor → client)
- [ ] Adaptive batching increases throughput under load:
  - Normal load (< 50%): 10K obs/sec, batch=10
  - High load (80-95%): 100K obs/sec, batch=500-1000
- [ ] No memory leaks under sustained backpressure (valgrind clean)

### Reliability
- [ ] Queue never exceeds configured capacity (hard guarantee)
- [ ] Backpressure doesn't cause session termination
- [ ] Recovery from overload state within 10s of load reduction
- [ ] Monitor continues functioning after worker crashes

## Definition of Done

- [ ] Code follows Rust Edition 2024 guidelines
- [ ] `make quality` passes (zero clippy warnings)
- [ ] All unit tests passing (coverage > 80%)
- [ ] All integration tests passing
- [ ] Performance benchmarks meeting targets
- [ ] Prometheus metrics added:
  - `engram_backpressure_state` (gauge: 0-3)
  - `engram_admission_control_rejects_total` (counter)
  - `engram_adaptive_batch_size` (histogram)
- [ ] Documentation updated in `docs/operations/streaming.md`
- [ ] Task renamed: `006_backpressure_admission_control_complete.md`
- [ ] Committed with message referencing milestone

## Notes

**Design Rationale:**

1. **Check interval 100ms:** Balances responsiveness vs overhead
2. **Thresholds (50/80/95%):** Based on queueing theory - avoid thrashing near capacity
3. **Adaptive batching:** Trades latency (10ms → 100ms) for 10x throughput
4. **Retry-after:** Client-side guidance, not enforced (HTTP 429 pattern)

**Future Enhancements:**

- Per-space backpressure (currently global)
- Client-specific rate limiting (token bucket)
- Predictive backpressure (queue growth rate)
- Load shedding (drop low-priority observations)
