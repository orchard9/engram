# Task 002: Graph Storage Adaptation

## Objective
Adapt the memory graph storage layer to efficiently handle dual memory types with separate DashMap configurations, cache-optimal layouts, and NUMA-aware placement for episodes vs concepts.

## Background
The current storage assumes homogeneous Memory nodes stored in a single DashMap. We need dual storage tiers optimized for the distinct access patterns of episodes (temporal locality, frequent updates) vs concepts (stable, semantic search).

## Requirements
1. Extend graph backend trait to support node type queries
2. Implement separate DashMap indices for episodes and concepts
3. Add type-aware iteration and filtering
4. Maintain backwards compatibility with existing UnifiedMemoryGraph
5. Optimize storage layout for cache efficiency with NUMA awareness
6. Add memory budget enforcement strategies
7. Include WAL integration for crash consistency
8. Define migration path from existing single-map storage

## Technical Specification

### Files to Modify
- `engram-core/src/memory_graph/traits.rs` - Add DualMemoryBackend trait
- `engram-core/src/memory_graph/backends/dashmap.rs` - Implement DualDashMapBackend
- `engram-core/src/memory_graph/graph.rs` - Add dual-backend constructor
- `engram-core/src/storage/mod.rs` - Memory budget coordinator

### Files to Create
- `engram-core/src/memory_graph/backends/dual_dashmap.rs` - New dual-map backend
- `engram-core/src/storage/dual_memory_budget.rs` - Budget enforcement

### New Traits
```rust
/// Extended trait for dual memory type operations
pub trait DualMemoryBackend: MemoryBackend {
    /// Add node with explicit type annotation
    fn add_node_typed(&self, node: DualMemoryNode) -> Result<NodeId>;

    /// Retrieve node with type information
    fn get_node_typed(&self, id: &NodeId) -> Result<Option<DualMemoryNode>>;

    /// Iterate only episode nodes (zero-allocation)
    fn iter_episodes(&self) -> Box<dyn Iterator<Item = DualMemoryNode>>;

    /// Iterate only concept nodes (zero-allocation)
    fn iter_concepts(&self) -> Box<dyn Iterator<Item = DualMemoryNode>>;

    /// Get counts by type: (episodes, concepts)
    fn count_by_type(&self) -> (usize, usize);

    /// Get memory usage by type in bytes
    fn memory_usage_by_type(&self) -> (usize, usize);
}
```

### Storage Layout - DashMap Configuration

Based on analysis of `engram-core/src/memory_graph/backends/dashmap.rs` and `engram-core/src/storage/hot_tier.rs`:

```rust
/// Dual-tier storage with separate DashMaps for cache efficiency
pub struct DualDashMapBackend {
    // EPISODE STORAGE (temporal locality, high churn)
    // Configuration: Shard count = 64 for fine-grained locking
    // Pre-allocated capacity: 1M episodes
    episodes: Arc<DashMap<NodeId, Arc<DualMemoryNode>>>,
    episode_edges: Arc<DashMap<NodeId, Vec<(NodeId, f32)>>>,
    episode_activation_cache: Arc<DashMap<NodeId, AtomicF32>>,

    // CONCEPT STORAGE (semantic locality, stable)
    // Configuration: Shard count = 16 (lower contention, larger shards)
    // Pre-allocated capacity: 100K concepts (10:1 episode:concept ratio)
    concepts: Arc<DashMap<NodeId, Arc<DualMemoryNode>>>,
    concept_edges: Arc<DashMap<NodeId, Vec<(NodeId, f32)>>>,
    concept_activation_cache: Arc<DashMap<NodeId, AtomicF32>>,

    // METADATA AND INDEXING
    // Fast type lookup without deserializing full node
    type_index: Arc<DashMap<NodeId, MemoryNodeType>>,

    // NUMA-aware placement (from storage/numa.rs)
    numa_topology: Arc<NumaTopology>,
    episode_socket: usize,  // Pin episodes to socket 0
    concept_socket: usize,  // Pin concepts to socket 1 (if multi-socket)

    // MEMORY BUDGET ENFORCEMENT
    budget: Arc<DualMemoryBudget>,

    // WAL for crash consistency (from storage/wal.rs)
    wal_writer: Arc<WalWriter>,

    // Metrics (from storage/mod.rs StorageMetrics)
    metrics: Arc<StorageMetrics>,
}

impl DualDashMapBackend {
    /// Create with NUMA-aware configuration
    pub fn new_numa_aware(
        episode_capacity: usize,
        concept_capacity: usize,
        episode_budget_mb: usize,
        concept_budget_mb: usize,
        wal_dir: PathBuf,
    ) -> Result<Self, MemoryError> {
        let topology = NumaTopology::detect()
            .map_err(|e| MemoryError::StorageError(format!("NUMA detection: {e}")))?;

        // Multi-socket strategy: separate episodes (temporal) and concepts (semantic)
        let episode_socket = 0;
        let concept_socket = if topology.socket_count > 1 { 1 } else { 0 };

        // DashMap with explicit shard count
        let episodes = Arc::new(DashMap::with_capacity_and_shard_amount(
            episode_capacity,
            64, // Fine-grained sharding for high concurrency
        ));

        let concepts = Arc::new(DashMap::with_capacity_and_shard_amount(
            concept_capacity,
            16, // Coarser sharding for stable access
        ));

        // Initialize WAL (from storage/wal.rs pattern)
        let metrics = Arc::new(StorageMetrics::new());
        let wal_writer = Arc::new(WalWriter::new(
            wal_dir,
            FsyncMode::PerBatch,
            Arc::clone(&metrics),
        )?);
        wal_writer.start()?;

        Ok(Self {
            episodes,
            episode_edges: Arc::new(DashMap::with_capacity_and_shard_amount(episode_capacity, 64)),
            episode_activation_cache: Arc::new(DashMap::with_capacity_and_shard_amount(episode_capacity, 64)),
            concepts,
            concept_edges: Arc::new(DashMap::with_capacity_and_shard_amount(concept_capacity, 16)),
            concept_activation_cache: Arc::new(DashMap::with_capacity_and_shard_amount(concept_capacity, 16)),
            type_index: Arc::new(DashMap::with_capacity(episode_capacity + concept_capacity)),
            numa_topology: Arc::new(topology),
            episode_socket,
            concept_socket,
            budget: Arc::new(DualMemoryBudget::new(episode_budget_mb, concept_budget_mb)),
            wal_writer,
            metrics,
        })
    }
}
```

### Cache Locality Optimizations

Following patterns from `engram-core/src/storage/hot_tier.rs` and `engram-core/src/storage/cache.rs`:

```rust
impl DualDashMapBackend {
    /// Insert with cache-line alignment and NUMA placement
    fn insert_aligned(&self, node: DualMemoryNode) -> Result<(), MemoryError> {
        let node_id = node.id;
        let node_arc = Arc::new(node);

        match &node_arc.node_type {
            MemoryNodeType::Episode { .. } => {
                // Check budget before insertion
                if !self.budget.can_allocate_episode() {
                    self.evict_lru_episode()?;
                }

                // Write to WAL for durability
                let entry = WalEntry::new_memory_update(&Memory::from(node_arc.as_ref()))?;
                self.wal_writer.write_async(entry);

                // Insert into episode map
                self.episodes.insert(node_id, Arc::clone(&node_arc));
                self.episode_activation_cache.insert(
                    node_id,
                    AtomicF32::new(node_arc.activation.load(Ordering::Relaxed))
                );

                self.budget.record_episode_allocation(std::mem::size_of::<DualMemoryNode>());
            }
            MemoryNodeType::Concept { .. } => {
                if !self.budget.can_allocate_concept() {
                    return Err(MemoryError::CapacityExceeded {
                        current: self.concepts.len(),
                        max: self.budget.concept_capacity(),
                    });
                }

                // Concepts are more stable - use sync write
                let entry = WalEntry::new_memory_update(&Memory::from(node_arc.as_ref()))?;
                self.wal_writer.write_sync(entry)?;

                self.concepts.insert(node_id, Arc::clone(&node_arc));
                self.concept_activation_cache.insert(
                    node_id,
                    AtomicF32::new(node_arc.activation.load(Ordering::Relaxed))
                );

                self.budget.record_concept_allocation(std::mem::size_of::<DualMemoryNode>());
            }
        }

        // Update type index for fast lookups
        self.type_index.insert(node_id, node_arc.node_type.clone());
        self.metrics.record_write(std::mem::size_of::<DualMemoryNode>() as u64);

        Ok(())
    }

    /// LRU eviction for episodes using access tracking (from storage/hot_tier.rs pattern)
    fn evict_lru_episode(&self) -> Result<(), MemoryError> {
        // Find least recently accessed episode
        let candidates: Vec<_> = self.episode_activation_cache
            .iter()
            .map(|entry| (*entry.key(), entry.value().load(Ordering::Relaxed)))
            .collect();

        if let Some((evict_id, _)) = candidates.iter()
            .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal)) {

            self.episodes.remove(evict_id);
            self.episode_edges.remove(evict_id);
            self.episode_activation_cache.remove(evict_id);
            self.type_index.remove(evict_id);

            self.budget.record_episode_deallocation(std::mem::size_of::<DualMemoryNode>());
        }

        Ok(())
    }
}
```

### NUMA-Aware Placement Strategies

Based on `engram-core/src/storage/numa.rs` implementation:

```rust
/// NUMA placement strategy for dual memory types
pub struct NumaPlacementStrategy {
    topology: Arc<NumaTopology>,
    episode_policy: NumaPolicy,
    concept_policy: NumaPolicy,
}

impl NumaPlacementStrategy {
    pub fn new(topology: Arc<NumaTopology>) -> Self {
        let (episode_policy, concept_policy) = if topology.socket_count > 1 {
            // Multi-socket: episodes on socket 0 (temporal locality)
            //              concepts on socket 1 (semantic locality)
            (NumaPolicy::Bind(0), NumaPolicy::Bind(1))
        } else {
            // Single-socket: interleave for balanced access
            (NumaPolicy::Interleaved, NumaPolicy::Interleaved)
        };

        Self {
            topology,
            episode_policy,
            concept_policy,
        }
    }

    /// Allocate episode storage with NUMA binding
    pub fn allocate_episode_storage(&self, size: usize) -> Result<NumaMemoryMap, StorageError> {
        let mapping = NumaMemoryMap::new(size, self.episode_policy, Arc::clone(&self.topology))?;

        // Prefetch and lock in memory for hot path
        mapping.prefetch(0, size);
        mapping.advise_random(); // Episodes accessed randomly
        mapping.lock_in_memory()?;

        Ok(mapping)
    }

    /// Allocate concept storage with NUMA binding
    pub fn allocate_concept_storage(&self, size: usize) -> Result<NumaMemoryMap, StorageError> {
        let mapping = NumaMemoryMap::new(size, self.concept_policy, Arc::clone(&self.topology))?;

        // Concepts accessed sequentially during consolidation
        mapping.prefetch(0, size);
        mapping.advise_sequential();
        mapping.lock_in_memory()?;

        Ok(mapping)
    }
}
```

### Memory Budget Enforcement

```rust
/// Dual memory budget coordinator
pub struct DualMemoryBudget {
    episode_budget_bytes: usize,
    concept_budget_bytes: usize,
    episode_allocated: AtomicUsize,
    concept_allocated: AtomicUsize,
    episode_capacity: usize,
    concept_capacity: usize,
}

impl DualMemoryBudget {
    pub fn new(episode_mb: usize, concept_mb: usize) -> Self {
        Self {
            episode_budget_bytes: episode_mb * 1024 * 1024,
            concept_budget_bytes: concept_mb * 1024 * 1024,
            episode_allocated: AtomicUsize::new(0),
            concept_allocated: AtomicUsize::new(0),
            episode_capacity: episode_mb * 1024 * 1024 / std::mem::size_of::<DualMemoryNode>(),
            concept_capacity: concept_mb * 1024 * 1024 / std::mem::size_of::<DualMemoryNode>(),
        }
    }

    pub fn can_allocate_episode(&self) -> bool {
        self.episode_allocated.load(Ordering::Relaxed) < self.episode_budget_bytes
    }

    pub fn can_allocate_concept(&self) -> bool {
        self.concept_allocated.load(Ordering::Relaxed) < self.concept_budget_bytes
    }

    pub fn record_episode_allocation(&self, bytes: usize) {
        self.episode_allocated.fetch_add(bytes, Ordering::Relaxed);
    }

    pub fn record_concept_allocation(&self, bytes: usize) {
        self.concept_allocated.fetch_add(bytes, Ordering::Relaxed);
    }

    pub fn record_episode_deallocation(&self, bytes: usize) {
        self.episode_allocated.fetch_sub(bytes, Ordering::Relaxed);
    }

    pub fn episode_capacity(&self) -> usize {
        self.episode_capacity
    }

    pub fn concept_capacity(&self) -> usize {
        self.concept_capacity
    }
}
```

### Migration Path from UnifiedMemoryGraph

```rust
impl DualDashMapBackend {
    /// Migrate from legacy single-map backend
    pub fn migrate_from_legacy(
        legacy: &DashMapBackend,
        classifier: impl Fn(&Memory) -> MemoryNodeType,
    ) -> Result<Self, MemoryError> {
        // Create new dual backend with capacities from legacy
        let legacy_count = legacy.count();
        let backend = Self::new_numa_aware(
            (legacy_count * 9) / 10, // 90% episodes estimate
            legacy_count / 10,        // 10% concepts estimate
            1024,                     // 1GB episode budget
            256,                      // 256MB concept budget
            PathBuf::from("./wal"),
        )?;

        // Migrate all memories with type classification
        for legacy_id in legacy.all_ids() {
            if let Some(memory) = legacy.retrieve(&legacy_id)? {
                let node_type = classifier(&memory);
                let dual_node = DualMemoryNode::from_memory(memory.as_ref(), node_type)?;
                backend.insert_aligned(dual_node)?;
            }
        }

        // Migrate edges
        for (from, to, weight) in legacy.all_edges()? {
            backend.add_edge(from, to, weight)?;
        }

        Ok(backend)
    }
}
```

## Performance Targets

Based on existing benchmarks in `engram-core/benches/`:

1. **Concurrent Insertion** (from `concurrent_hnsw_validation.rs`, `batch_hnsw_insert.rs`):
   - Episodes: >100K inserts/sec with 16 threads
   - Concepts: >10K inserts/sec with 4 threads
   - Target: <100μs P99 latency for episode insertion

2. **Type-Specific Iteration** (from `recall_performance.rs`, `spreading_benchmarks.rs`):
   - Episode iteration: >1M nodes/sec
   - Concept iteration: >500K nodes/sec
   - Zero-allocation iteration with DashMap iterators

3. **Memory Overhead** (from `comprehensive.rs`, `metrics_overhead.rs`):
   - <15% overhead vs single DashMap (type index cost)
   - NUMA placement reduces remote access by >60%
   - Cache hit rate >85% for episode operations

4. **Search Performance** (from `ann_validation.rs`, `vector_similarity_comparison.rs`):
   - Type-filtered search within 10% of homogeneous baseline
   - SIMD-optimized similarity maintains <5μs per comparison

5. **Spreading Activation** (from `spreading_benchmarks.rs`, `gpu_spreading.rs`):
   - Cross-type activation: <50ms for 1K episode → 100 concept links
   - Within-type activation: match baseline (<10ms for 1K nodes)

## Integration with Existing Systems

### HNSW Index Integration (from `engram-core/src/index/mod.rs`)

```rust
impl DualDashMapBackend {
    /// Build separate HNSW indices for episodes and concepts
    pub fn build_dual_indices(&self) -> Result<(HnswGraph, HnswGraph), MemoryError> {
        let episode_index = HnswGraph::new();
        let concept_index = HnswGraph::new();

        // Episodes: ef_construction=200, M=16 (high churn, need fast updates)
        for entry in self.episodes.iter() {
            let node = entry.value();
            episode_index.insert_node(
                HnswNode::from_memory(node.id, node.as_ref(), 0)?,
                &EPISODE_HNSW_PARAMS,
                &compute::create_vector_ops(),
            )?;
        }

        // Concepts: ef_construction=400, M=32 (stable, optimize search quality)
        for entry in self.concepts.iter() {
            let node = entry.value();
            concept_index.insert_node(
                HnswNode::from_memory(node.id, node.as_ref(), 0)?,
                &CONCEPT_HNSW_PARAMS,
                &compute::create_vector_ops(),
            )?;
        }

        Ok((episode_index, concept_index))
    }
}
```

## Implementation Notes

1. **DashMap Shard Configuration**: Episodes use 64 shards for fine-grained locking under high concurrency. Concepts use 16 shards since they have lower update frequency.

2. **NUMA Socket Assignment**: On multi-socket systems, bind episodes to socket 0 (temporal access pattern benefits from locality) and concepts to socket 1 (semantic search can use cross-socket bandwidth).

3. **WAL Integration**: Episodes use async WAL writes (batched), concepts use sync writes (critical semantic data).

4. **Memory Budget**: Hard limits prevent OOM. Episodes use LRU eviction, concepts trigger consolidation instead of eviction.

5. **Type Index**: Separate DashMap for O(1) type lookup avoids deserializing full nodes during type filtering.

6. **Cache Line Alignment**: DualMemoryNode should be ≤64 bytes or cache-line aligned to avoid false sharing in concurrent access.

## Testing Approach

1. **Unit Tests**:
   - CRUD operations on both types
   - Type-specific iteration correctness
   - Budget enforcement edge cases
   - NUMA placement validation

2. **Concurrent Access Tests** (from `concurrent_recall.rs`, `concurrent_hnsw_validation.rs`):
   - 16 threads inserting episodes + 4 threads inserting concepts
   - Validate no data races with ThreadSanitizer
   - Measure contention with perf counters

3. **Migration Tests**:
   - Migrate 1M legacy memories to dual storage
   - Validate all data preserved
   - Compare query results pre/post migration

4. **Performance Benchmarks** (using Criterion from existing benches):
   - Insertion throughput by type
   - Type-filtered iteration speed
   - Cross-type spreading activation latency
   - NUMA vs non-NUMA comparison

5. **Integration Tests**:
   - Full recall pipeline with dual types
   - WAL recovery with mixed episode/concept data
   - Budget exhaustion and recovery

## Acceptance Criteria

- [ ] Separate DashMap indices for episodes and concepts with documented shard counts
- [ ] Type-aware iteration with zero-allocation iterators
- [ ] NUMA-aware placement on multi-socket systems (verify with `numactl --hardware`)
- [ ] Memory budget enforcement with LRU eviction for episodes
- [ ] WAL integration for crash consistency
- [ ] Migration utility from legacy DashMapBackend
- [ ] Query performance within 10% of homogeneous baseline
- [ ] Concurrent access maintains data consistency under ThreadSanitizer
- [ ] <15% memory overhead vs single DashMap
- [ ] All existing `UnifiedMemoryGraph` tests pass with new backend

## Dependencies
- Task 001 (Dual Memory Types) - must complete first for type definitions

## Estimated Time
4 days (increased from 3 days due to NUMA placement and WAL integration complexity)

## References

Codebase patterns analyzed:
- `/Users/jordanwashburn/Workspace/orchard9/engram/engram-core/src/memory_graph/backends/dashmap.rs` - DashMap backend pattern
- `/Users/jordanwashburn/Workspace/orchard9/engram/engram-core/src/storage/hot_tier.rs` - Cache and eviction patterns
- `/Users/jordanwashburn/Workspace/orchard9/engram/engram-core/src/storage/numa.rs` - NUMA allocation strategies
- `/Users/jordanwashburn/Workspace/orchard9/engram/engram-core/src/storage/wal.rs` - Write-ahead log integration
- `/Users/jordanwashburn/Workspace/orchard9/engram/engram-core/src/index/mod.rs` - HNSW index construction
- Benchmark suite in `/Users/jordanwashburn/Workspace/orchard9/engram/engram-core/benches/`
