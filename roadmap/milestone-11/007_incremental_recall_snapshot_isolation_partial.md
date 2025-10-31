# Task 007: Incremental Recall with Snapshot Isolation

**Status:** pending
**Estimated Effort:** 3 days
**Dependencies:** Task 005 (gRPC Streaming), Task 003 (Worker Pool for generation tracking)
**Blocks:** Task 010 (Performance Benchmarking)

## Objective

Implement snapshot-isolated recall that returns committed observations + probabilistically-available recent observations, with incremental result streaming (first result < 10ms, bounded staleness P99 < 100ms).

## Background

**Current State:**
- Recall exists in `engram-core/src/query/executor.rs`
- HNSW search in `engram-core/src/index/hnsw_search.rs`
- Returns `Vec<Memory>` (full result set, not streaming)

**Critical Gaps:**
- No snapshot isolation - sees all nodes regardless of commit status
- No generation tracking for visibility control
- No incremental streaming - must collect all results before returning

**Consistency Model:** Eventual with bounded staleness
- Observations **committed to HNSW** before snapshot T are visible
- Observations **in queue** may or may not be visible (probabilistic)
- Not linearizable - acceptable for cognitive memory

## Implementation Specification

### Files to Create

1. **`engram-core/src/streaming/recall.rs`** (~400 lines)

Core components:
```rust
pub struct SnapshotRecallConfig {
    pub snapshot_generation: u64,
    pub batch_size: usize,
    pub include_recent: bool,
}

pub struct IncrementalRecallStream {
    graph: Arc<HnswGraph>,
    cue: Cue,
    config: SnapshotRecallConfig,
    position: usize,
    results: Vec<(usize, f32)>, // (node_id, score)
}

impl IncrementalRecallStream {
    pub fn new(
        graph: Arc<HnswGraph>,
        cue: Cue,
        observation_queue: &ObservationQueue,
        include_recent: bool,
    ) -> Self;

    pub async fn search(&mut self) -> Result<(), RecallError>;
    pub fn next_batch(&mut self) -> Option<Vec<Memory>>;
    pub fn has_more(&self) -> bool;
}
```

### Files to Modify

1. **`engram-core/src/index/hnsw_node.rs`** - Add generation tracking
```rust
pub struct HnswNode<T> {
    pub id: String,
    pub data: T,
    pub neighbors: Vec<Vec<usize>>,

    // NEW: Generation tracking for snapshot isolation
    pub generation: u64,
    pub committed_at: Instant,
}
```

2. **`engram-core/src/index/hnsw_search.rs`** - Add filtered search
```rust
impl HnswGraph {
    pub fn search_with_filter<F>(
        &self,
        query: &[f32],
        k: usize,
        filter: F,
    ) -> Vec<(usize, f32)>
    where
        F: Fn(&HnswNode<Memory>) -> bool;
}
```

3. **`engram-core/src/streaming/observation_queue.rs`** - Track current generation
```rust
pub struct ObservationQueue {
    // ... existing fields ...
    current_generation: AtomicU64,
}

impl ObservationQueue {
    pub fn current_generation(&self) -> u64;
    pub fn mark_generation_committed(&self, generation: u64);
}
```

4. **`engram-cli/src/handlers/streaming.rs`** - Add recall stream handler
```rust
impl StreamingHandlers {
    pub async fn handle_recall_stream(
        &self,
        request: Request<StreamingRecallRequest>,
    ) -> Result<Response<impl Stream<Item = Result<StreamingRecallResponse, Status>>>, Status>;
}
```

## Detailed Implementation

### 1. Generation Tracking in HNSW Nodes

**Purpose:** Track when each node became visible for snapshot isolation

**Modify `hnsw_node.rs`:**
```rust
#[derive(Clone)]
pub struct HnswNode<T> {
    pub id: String,
    pub data: T,
    pub neighbors: Vec<Vec<usize>>,

    /// Generation (sequence number) when this node was inserted.
    /// Used for snapshot isolation - only nodes with generation <= snapshot
    /// are visible in a snapshot-isolated recall.
    pub generation: u64,

    /// Wall-clock timestamp when node was committed to index.
    /// Used for bounded staleness measurements.
    pub committed_at: Instant,
}

impl<T> HnswNode<T> {
    pub fn new(id: String, data: T, generation: u64) -> Self {
        Self {
            id,
            data,
            neighbors: Vec::new(),
            generation,
            committed_at: Instant::now(),
        }
    }
}
```

**Update HNSW construction to pass generation:**
```rust
// In hnsw_construction.rs
impl HnswGraph {
    pub fn insert_with_generation(
        &mut self,
        memory: Memory,
        generation: u64,
    ) -> Result<(), HnswError> {
        let node = HnswNode::new(
            memory.id.clone(),
            memory,
            generation,
        );

        let node_id = self.nodes.len();
        self.nodes.push(node);

        // ... existing insertion logic ...

        Ok(())
    }
}
```

### 2. Generation Tracking in ObservationQueue

**Purpose:** Track the highest generation committed to HNSW index

**Modify `observation_queue.rs`:**
```rust
pub struct ObservationQueue {
    // ... existing fields ...

    /// Current generation (highest committed observation).
    /// Updated atomically by worker pool when HNSW insertion completes.
    /// Used to capture snapshot for recall queries.
    current_generation: AtomicU64,
}

impl ObservationQueue {
    pub const fn new(config: QueueConfig) -> Self {
        Self {
            // ... existing fields ...
            current_generation: AtomicU64::new(0),
        }
    }

    /// Get current committed generation (snapshot point).
    #[must_use]
    pub fn current_generation(&self) -> u64 {
        self.current_generation.load(Ordering::SeqCst)
    }

    /// Mark a generation as committed (called by workers after HNSW insert).
    ///
    /// Uses fetch_max to handle out-of-order commits from parallel workers.
    pub fn mark_generation_committed(&self, generation: u64) {
        self.current_generation.fetch_max(generation, Ordering::SeqCst);
    }
}
```

**Integration in Worker Pool (Task 003):**
```rust
// In worker_pool.rs
impl HnswWorker {
    async fn process_observation(&self, obs: QueuedObservation) {
        // Insert into HNSW with generation
        self.graph.insert_with_generation(
            obs.episode.into(),
            obs.sequence_number,
        ).await?;

        // Mark generation as committed
        self.observation_queue.mark_generation_committed(obs.sequence_number);
    }
}
```

### 3. Filtered HNSW Search

**Purpose:** Search HNSW while filtering nodes by generation

**Modify `hnsw_search.rs`:**
```rust
impl HnswGraph {
    /// Standard HNSW search (unchanged).
    pub fn search(&self, query: &[f32], k: usize) -> Vec<(usize, f32)> {
        self.search_with_filter(query, k, |_| true) // No filter
    }

    /// HNSW search with custom node filter.
    ///
    /// The filter function is called for each candidate node during search.
    /// If filter returns false, the node is skipped (not added to results).
    ///
    /// Used for snapshot isolation: filter by `node.generation <= snapshot_gen`.
    pub fn search_with_filter<F>(
        &self,
        query: &[f32],
        k: usize,
        filter: F,
    ) -> Vec<(usize, f32)>
    where
        F: Fn(&HnswNode<Memory>) -> bool,
    {
        let mut candidates = Vec::with_capacity(k * 2);

        // Start at entry point (top layer)
        let mut entry_point = self.entry_point;

        // Navigate to layer 0 (ground layer)
        for layer in (1..=self.max_layer).rev() {
            entry_point = self.search_layer(query, entry_point, 1, layer)
                .into_iter()
                .next()
                .unwrap_or(entry_point);
        }

        // Search layer 0 with ef parameter
        let ef = k.max(self.ef_search);
        let mut results = self.search_layer_with_filter(
            query,
            entry_point,
            ef,
            0,
            &filter,
        );

        // Sort by distance and take top-k
        results.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        results.truncate(k);

        results
    }

    fn search_layer_with_filter<F>(
        &self,
        query: &[f32],
        entry_point: usize,
        ef: usize,
        layer: usize,
        filter: &F,
    ) -> Vec<(usize, f32)>
    where
        F: Fn(&HnswNode<Memory>) -> bool,
    {
        let mut visited = std::collections::HashSet::new();
        let mut candidates = std::collections::BinaryHeap::new();
        let mut results = Vec::new();

        let dist = self.distance(query, &self.nodes[entry_point].data.embedding);
        candidates.push((std::cmp::Reverse(dist), entry_point));
        visited.insert(entry_point);

        while let Some((std::cmp::Reverse(current_dist), current_id)) = candidates.pop() {
            let node = &self.nodes[current_id];

            // Apply filter - skip node if filter returns false
            if !filter(node) {
                continue;
            }

            results.push((current_id, current_dist));

            if results.len() >= ef {
                break;
            }

            // Explore neighbors at this layer
            if layer < node.neighbors.len() {
                for &neighbor_id in &node.neighbors[layer] {
                    if visited.insert(neighbor_id) {
                        let neighbor = &self.nodes[neighbor_id];
                        let neighbor_dist = self.distance(query, &neighbor.data.embedding);
                        candidates.push((std::cmp::Reverse(neighbor_dist), neighbor_id));
                    }
                }
            }
        }

        results
    }
}
```

### 4. Incremental Recall Stream

**Create `recall.rs`:**
```rust
//! Incremental recall with snapshot isolation for streaming queries.

use crate::index::{HnswGraph, HnswNode};
use crate::memory::Memory;
use crate::query::Cue;
use crate::streaming::ObservationQueue;
use std::sync::Arc;
use std::time::Instant;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum RecallError {
    #[error("Invalid cue: {0}")]
    InvalidCue(String),

    #[error("Search failed: {0}")]
    SearchFailed(String),
}

/// Configuration for snapshot-isolated recall.
pub struct SnapshotRecallConfig {
    /// Snapshot generation (observations with generation <= this are visible).
    pub snapshot_generation: u64,

    /// Batch size for incremental result streaming.
    pub batch_size: usize,

    /// Include in-flight observations (bounded staleness mode).
    /// If true, observations between snapshot_generation and current_generation
    /// may be visible (probabilistic).
    pub include_recent: bool,
}

/// Incremental recall stream that yields batches of results.
///
/// Executes HNSW search once, then streams results in batches for
/// low first-result latency.
pub struct IncrementalRecallStream {
    /// HNSW graph reference
    graph: Arc<HnswGraph>,

    /// Query cue
    cue: Cue,

    /// Snapshot configuration
    config: SnapshotRecallConfig,

    /// Current position in results (for batching)
    position: usize,

    /// Cached results from HNSW search (node_id, similarity_score)
    results: Vec<(usize, f32)>,

    /// Search execution timestamp (for latency measurement)
    search_started_at: Option<Instant>,
}

impl IncrementalRecallStream {
    /// Create a new incremental recall stream.
    ///
    /// Captures current generation from observation queue as snapshot point.
    pub fn new(
        graph: Arc<HnswGraph>,
        cue: Cue,
        observation_queue: &ObservationQueue,
        include_recent: bool,
    ) -> Self {
        let snapshot_generation = observation_queue.current_generation();

        Self {
            graph,
            cue,
            config: SnapshotRecallConfig {
                snapshot_generation,
                batch_size: 10, // Low latency default
                include_recent,
            },
            position: 0,
            results: Vec::new(),
            search_started_at: None,
        }
    }

    /// Execute HNSW search with snapshot isolation.
    ///
    /// This is a potentially expensive operation - run in blocking thread pool.
    pub async fn search(&mut self) -> Result<(), RecallError> {
        self.search_started_at = Some(Instant::now());

        // Convert cue to embedding vector
        let query_embedding = self.cue.to_embedding()
            .map_err(|e| RecallError::InvalidCue(format!("{e}")))?;

        // Execute HNSW search with visibility filter
        let graph = Arc::clone(&self.graph);
        let snapshot_gen = self.config.snapshot_generation;
        let include_recent = self.config.include_recent;

        self.results = tokio::task::spawn_blocking(move || {
            graph.search_with_filter(
                &query_embedding,
                100, // k neighbors
                |node: &HnswNode<Memory>| {
                    // Snapshot isolation visibility rule
                    if include_recent {
                        // Bounded staleness: show all nodes up to current time
                        true
                    } else {
                        // Strict snapshot: only committed before snapshot
                        node.generation <= snapshot_gen
                    }
                }
            )
        })
        .await
        .map_err(|e| RecallError::SearchFailed(format!("Join error: {e}")))?;

        Ok(())
    }

    /// Get next batch of results (for incremental streaming).
    ///
    /// Returns None when all results have been yielded.
    pub fn next_batch(&mut self) -> Option<Vec<Memory>> {
        if self.position >= self.results.len() {
            return None;
        }

        let end = (self.position + self.config.batch_size).min(self.results.len());

        let batch: Vec<Memory> = self.results[self.position..end]
            .iter()
            .map(|(node_id, _score)| {
                // TODO: Handle case where node_id is invalid
                self.graph.get_node(*node_id).unwrap().data.clone()
            })
            .collect();

        self.position = end;

        Some(batch)
    }

    /// Check if more results are available.
    #[must_use]
    pub const fn has_more(&self) -> bool {
        self.position < self.results.len()
    }

    /// Get total result count.
    #[must_use]
    pub fn total_count(&self) -> usize {
        self.results.len()
    }

    /// Get search latency (if search has been executed).
    #[must_use]
    pub fn search_latency(&self) -> Option<std::time::Duration> {
        self.search_started_at.map(|start| start.elapsed())
    }
}
```

### 5. gRPC Handler for Recall Stream

**Extend `streaming.rs`:**
```rust
impl StreamingHandlers {
    pub async fn handle_recall_stream(
        &self,
        request: Request<StreamingRecallRequest>,
    ) -> Result<Response<impl Stream<Item = Result<StreamingRecallResponse, Status>>>, Status> {
        let req = request.into_inner();

        // Resolve memory space
        let memory_space_id = self.resolve_memory_space(&req.memory_space_id)?;

        // Convert proto cue to core Cue
        let cue = CoreCue::try_from(req.cue.ok_or_else(||
            Status::invalid_argument("Missing cue")
        )?)
        .map_err(|e| Status::invalid_argument(format!("Invalid cue: {e}")))?;

        // Get graph for memory space
        let graph = self.store.get_graph(&memory_space_id)
            .map_err(|e| Status::internal(format!("Failed to get graph: {e}")))?;

        // Create incremental recall stream
        let mut recall_stream = IncrementalRecallStream::new(
            graph,
            cue,
            &self.observation_queue,
            req.snapshot_isolation,
        );

        let snapshot_gen = recall_stream.config.snapshot_generation;

        // Execute search (async, runs in blocking thread pool)
        recall_stream.search().await
            .map_err(|e| Status::internal(format!("Search failed: {e}")))?;

        // Create response stream
        let (tx, rx) = tokio::sync::mpsc::channel(10);

        // Spawn task to stream results incrementally
        tokio::spawn(async move {
            let mut batch_count = 0;

            while let Some(batch) = recall_stream.next_batch() {
                batch_count += 1;
                let has_more = recall_stream.has_more();

                // Convert core Memory to proto Memory
                let proto_memories: Vec<engram_proto::Memory> = batch
                    .into_iter()
                    .map(Into::into)
                    .collect();

                let response = StreamingRecallResponse {
                    results: proto_memories,
                    more_results: has_more,
                    metadata: Some(RecallMetadata {
                        total_activated: recall_stream.total_count() as i32,
                        above_threshold: recall_stream.total_count() as i32,
                        avg_activation: 0.8,
                        recall_time_ms: recall_stream.search_latency()
                            .unwrap_or_default()
                            .as_millis() as i64,
                        activation_path: vec![],
                    }),
                    snapshot_sequence: snapshot_gen,
                };

                if tx.send(Ok(response)).await.is_err() {
                    break; // Client disconnected
                }

                // Yield to allow other tasks to run
                tokio::task::yield_now().await;
            }

            tracing::debug!("Recall stream completed: {} batches streamed", batch_count);
        });

        Ok(Response::new(ReceiverStream::new(rx)))
    }
}
```

## Testing Strategy

### Unit Tests

```rust
#[test]
fn test_generation_tracking_basic() {
    let queue = ObservationQueue::new(QueueConfig::default());
    assert_eq!(queue.current_generation(), 0);

    queue.mark_generation_committed(5);
    assert_eq!(queue.current_generation(), 5);

    queue.mark_generation_committed(10);
    assert_eq!(queue.current_generation(), 10);
}

#[test]
fn test_generation_tracking_out_of_order() {
    let queue = ObservationQueue::new(QueueConfig::default());

    queue.mark_generation_committed(10);
    assert_eq!(queue.current_generation(), 10);

    // Out-of-order commit (earlier generation)
    queue.mark_generation_committed(5);
    assert_eq!(queue.current_generation(), 10); // Should not regress
}

#[tokio::test]
async fn test_incremental_recall_batching() {
    let graph = create_test_graph_with_100_nodes();
    let queue = ObservationQueue::new(QueueConfig::default());
    queue.mark_generation_committed(100);

    let mut stream = IncrementalRecallStream::new(
        graph,
        test_cue(),
        &queue,
        false,
    );

    stream.search().await.unwrap();

    // Verify incremental batches
    let batch1 = stream.next_batch().unwrap();
    assert_eq!(batch1.len(), 10);
    assert!(stream.has_more());

    let batch2 = stream.next_batch().unwrap();
    assert_eq!(batch2.len(), 10);
}

#[tokio::test]
async fn test_snapshot_isolation_filters_uncommitted() {
    let graph = create_test_graph();
    let queue = ObservationQueue::new(QueueConfig::default());

    // Mark only generations 1-50 as committed
    queue.mark_generation_committed(50);

    let mut stream = IncrementalRecallStream::new(
        graph,
        test_cue(),
        &queue,
        false, // Strict snapshot isolation
    );

    stream.search().await.unwrap();

    // Verify only generations 1-50 are visible
    let all_results = collect_all_batches(&mut stream);
    assert!(all_results.iter().all(|m| m.generation <= 50));
}
```

### Integration Tests

```rust
#[tokio::test]
async fn test_snapshot_isolation_consistency() {
    let store = setup_test_store_with_workers().await;

    // Store 100 observations
    for i in 1..=100 {
        store.observe(test_episode(i)).await.unwrap();
    }

    // Wait for first 50 to be committed
    wait_until_generation_committed(&store, 50).await;

    // Execute snapshot recall
    let recall = store.streaming_recall(test_cue(), true).await.unwrap();
    let results: Vec<_> = collect_all_batches(recall).await;

    // Should only see generations 1-50
    assert!(results.iter().all(|m| m.generation <= 50));
    assert!(results.len() >= 40 && results.len() <= 50); // Some filtering is OK
}

#[tokio::test]
async fn test_bounded_staleness_p99() {
    // Measure visibility latency: observation → visible in recall
    let store = setup_test_store_with_workers().await;
    let mut latencies = Vec::new();

    for i in 0..1000 {
        let start = Instant::now();
        let episode_id = format!("test_{i}");
        store.observe(test_episode_with_id(&episode_id)).await.unwrap();

        // Poll until visible in recall
        loop {
            let results = store.recall(test_cue()).await.unwrap();
            if results.iter().any(|m| m.id == episode_id) {
                let latency = start.elapsed();
                latencies.push(latency);
                break;
            }

            tokio::time::sleep(Duration::from_millis(1)).await;
        }
    }

    // Calculate P99
    latencies.sort();
    let p99_index = (latencies.len() * 99) / 100;
    let p99 = latencies[p99_index];

    assert!(p99 < Duration::from_millis(100), "P99 latency: {:?}", p99);
}

#[tokio::test]
async fn test_first_result_latency() {
    let store = setup_test_store().await;

    // Pre-populate with data
    for i in 0..1000 {
        store.observe(test_episode(i)).await.unwrap();
    }
    wait_until_all_committed(&store).await;

    // Measure first result latency
    let start = Instant::now();
    let mut recall_stream = store.streaming_recall(test_cue()).await.unwrap();

    let first_batch = recall_stream.next().await.unwrap().unwrap();
    let first_result_latency = start.elapsed();

    assert!(first_result_latency < Duration::from_millis(10),
        "First result latency: {:?}", first_result_latency);
}
```

## Acceptance Criteria

### Functional
- [ ] Recall sees all observations committed before snapshot
- [ ] Recall does NOT see uncommitted observations (snapshot_isolation=true)
- [ ] Recall MAY see recent observations (snapshot_isolation=false)
- [ ] Incremental streaming returns results in batches (batch_size=10)
- [ ] First result arrives within 10ms of request

### Performance
- [ ] Visibility latency P99 < 100ms (observation → recall)
- [ ] First result latency P99 < 10ms
- [ ] Full recall latency P99 < 100ms for 10K results
- [ ] Memory overhead: < 1MB per active recall stream

### Consistency
- [ ] No phantom reads (results don't change mid-stream)
- [ ] No lost observations (all committed eventually visible)
- [ ] Generation tracking monotonic (no regression)
- [ ] Confidence scores reflect staleness

## Definition of Done

- [ ] Code follows Rust Edition 2024 guidelines
- [ ] `make quality` passes
- [ ] All tests passing (unit + integration)
- [ ] Performance benchmarks meeting targets
- [ ] Documentation:
  - [ ] Rustdoc for all public types
  - [ ] Snapshot isolation semantics explained
  - [ ] Example in `examples/streaming/recall_client.rs`
- [ ] Metrics added:
  - [ ] `engram_recall_visibility_latency_ms` (histogram)
  - [ ] `engram_recall_first_result_latency_ms` (histogram)
- [ ] Task renamed: `007_incremental_recall_snapshot_isolation_complete.md`

## Notes

**Snapshot Isolation Semantics:**
- NOT linearizable (doesn't guarantee real-time order)
- Eventual consistency with bounded staleness (P99 < 100ms)
- Matches biological memory (you don't instantly recall new information)

**Generation vs Timestamp:**
- Generation: Logical clock (monotonic sequence)
- Timestamp: Wall-clock time
- Use generation for consistency, timestamp for metrics

**Performance Optimization:**
- Batch size tuning: smaller = lower latency, larger = better throughput
- Pre-sorted results: HNSW returns sorted by similarity
- Zero-copy: `Arc<Memory>` throughout pipeline
