# Task 009: Distributed Query Execution - Comprehensive Analysis Report

**Date**: 2025-11-01
**Objective**: Understand implementation details for designing distributed query execution across memory spaces (Task 009 preparation)
**Thoroughness Level**: Very Thorough

---

## Executive Summary

The Engram codebase has sophisticated query infrastructure ready for distributed execution:

1. **Query Types**: Five core operations (RECALL, SPREAD, CONSOLIDATE, IMAGINE, PREDICT) with AST-based routing
2. **Result Handling**: Probabilistic results with confidence intervals, evidence chains, and uncertainty tracking
3. **Activation System**: Lock-free parallel spreading with tier-aware scheduling and confidence aggregation
4. **Execution Pipeline**: Async query executor with timeout enforcement, complexity limits, and multi-tenant isolation

**Key Finding**: The architecture supports scatter-gather patterns through `ActivationPath` structures and evidence chain merging, making distributed execution highly feasible.

---

## 1. Query Types and Implementations

### 1.1 RECALL Operation

**File**: `/Users/jordan/Workspace/orchard9/engram/engram-core/src/query/executor/recall.rs`

**Purpose**: Direct memory retrieval with pattern matching

**Input Structures**:
```rust
pub struct RecallQuery<'a> {
    pub pattern: Pattern<'a>,           // NodeId, Embedding, ContentMatch, Any
    pub constraints: Vec<Constraint>,   // Confidence, temporal, embedding similarity
    pub confidence_threshold: Option<ConfidenceThreshold>,
    pub base_rate: Option<f32>,
    pub limit: Option<usize>,
}

pub enum Pattern<'a> {
    NodeId(NodeIdentifier),
    Embedding { vector: Vec<f32>, threshold: f32 },
    ContentMatch(&'a str),
    Any,
}
```

**Execution Flow**:
1. Convert `Pattern` to `Cue` via `pattern_to_cue()`
2. Call `MemoryStore::recall(&cue)` for direct store operation
3. Apply constraints filtering (confidence, temporal, similarity)
4. Apply limit and confidence threshold
5. Transform to `ProbabilisticQueryResult` with evidence

**Output**: `ProbabilisticQueryResult` containing:
- Episodes: `Vec<(Episode, Confidence)>`
- Confidence interval with uncertainty
- Evidence chain tracking retrieval method
- Uncertainty sources from system state

**Biological Grounding**:
- NodeId lookup mimics CA3 pattern completion (hippocampal indexing)
- Embedding similarity reflects dentate gyrus pattern separation
- Semantic search corresponds to neocortical spreading

---

### 1.2 SPREAD Operation

**File**: `/Users/jordan/Workspace/orchard9/engram/engram-core/src/query/executor/spread.rs`

**Purpose**: Activation spreading from source memory through semantic associations

**Input Structure**:
```rust
pub struct SpreadQuery<'a> {
    pub source: NodeIdentifier,
    pub max_hops: Option<u16>,
    pub decay_rate: Option<f32>,
    pub activation_threshold: Option<f32>,
    pub refractory_period: Option<Duration>,
}

// Defaults:
// - max_hops: 3
// - decay_rate: 0.15 (15% per hop)
// - threshold: 0.02
```

**Execution Flow**:
1. Validate source node exists in memory graph
2. Get `ParallelSpreadingEngine` from memory store
3. Configure spreading parameters (decay, threshold, max_hops)
4. Execute `spread_activation()` with seed activation 1.0
5. Extract spreading results with activation paths
6. Transform to `ProbabilisticQueryResult`

**Decay Calculation**:
```rust
// decay_rate parameter to exponential conversion:
// decay_rate = 0.15 → exp_rate = -ln(1 - 0.15) ≈ 0.162
// A(depth) = exp(-exp_rate × depth)
// depth=1: 0.85, depth=2: 0.72, depth=3: 0.61
```

**Output**: Episodes activated + evidence from spreading paths

---

### 1.3 CONSOLIDATE Operation

**File**: `/Users/jordan/Workspace/orchard9/engram/engram-core/src/query/executor/consolidate.rs`

**Purpose**: Memory consolidation (episodic → semantic, sleep-based replay)

**Input Structure**:
```rust
pub struct ConsolidateQuery<'a> {
    pub episodes: EpisodeSelector,
    pub target: NodeIdentifier,
    pub scheduler_policy: Option<SchedulerPolicy>,
}

pub enum EpisodeSelector {
    All,
    Recent(Duration),
    WithTag(String),
}

pub enum SchedulerPolicy {
    Immediate,
    Interval(Duration),
    Threshold { activation: f32 },
}
```

**Execution Policies**:
- **Immediate**: Synchronous consolidation (blocks query)
- **Interval**: Background consolidation (every N seconds)
- **Threshold**: Consolidate when activation reaches level

**Output**: Consolidated semantic patterns + statistics

---

### 1.4 IMAGINE Operation

**File**: `/Users/jordan/Workspace/orchard9/engram/engram-core/src/query/executor/imagine.rs`

**Purpose**: Pattern completion (fill missing episode details)

**Input Structure**:
```rust
pub struct ImagineQuery<'a> {
    pub pattern: Pattern<'a>,           // Partial episode
    pub seeds: Vec<NodeIdentifier>,     // Context nodes
    pub novelty: Option<f32>,           // 0.0=recall only, 1.0=creative
    pub confidence_threshold: Option<f32>,
}
```

**Execution**:
1. Convert partial pattern to `PartialEpisode`
2. Use `PatternCompleter` (CA3 autoassociative memory)
3. Apply novelty parameter for creativity control
4. Track source attribution (recalled vs reconstructed)

**Output**: Completed episodes with source evidence

---

### 1.5 PREDICT Operation

**File**: `/Users/jordan/Workspace/orchard9/engram/engram-core/src/query/executor/predict.rs`

**Status**: Placeholder - Full implementation planned for Milestone 15

**Purpose**: Forecast future states based on temporal patterns

**Input Structure**:
```rust
pub struct PredictQuery<'a> {
    pub pattern: Pattern<'a>,
    pub context: Vec<NodeIdentifier>,
    pub horizon: Option<Duration>,
    pub confidence_constraint: Option<Confidence>,
}
```

**Current Implementation**:
- Returns `NoPredictionEngine` error
- Requires System 2 reasoning (not yet implemented)
- Reserved for causal inference + counterfactual reasoning

---

## 2. Result Structures

### 2.1 ProbabilisticQueryResult

**File**: `/Users/jordan/Workspace/orchard9/engram/engram-core/src/query/mod.rs`

**Definition**:
```rust
pub struct ProbabilisticQueryResult {
    /// Episodes with confidence scores
    pub episodes: Vec<(Episode, Confidence)>,
    
    /// Confidence interval around point estimate
    pub confidence_interval: ConfidenceInterval,
    
    /// Evidence chain with dependency tracking
    pub evidence_chain: Vec<Evidence>,
    
    /// Uncertainty sources from spreading/decay
    pub uncertainty_sources: Vec<UncertaintySource>,
}
```

**Key Methods**:
- `from_episodes(episodes)` - Create from raw results
- `and(&other)` - Intersection semantics (both must match)
- `or(&other)` - Union semantics (either matches)
- `not()` - Negation of confidence
- `is_successful()` - Non-empty + high confidence

---

### 2.2 ConfidenceInterval

**Definition**:
```rust
pub struct ConfidenceInterval {
    pub lower: Confidence,      // Lower bound
    pub upper: Confidence,      // Upper bound
    pub point: Confidence,      // Point estimate
    pub width: f32,             // Interval width
}
```

**Construction**:
```rust
// From point estimate with uncertainty
ConfidenceInterval::from_confidence_with_uncertainty(point, uncertainty)

// From point estimate only
ConfidenceInterval::from_confidence(confidence)
```

**Width Calculation**:
```rust
half_width = (uncertainty * point_value).min(point_value.min(1.0 - point_value))
width = half_width * 2.0
```

---

### 2.3 Evidence Structures

**Evidence Types**:
```rust
pub enum EvidenceSource {
    SpreadingActivation {
        source_episode: String,
        activation_level: Activation,
        path_length: u16,
    },
    
    TemporalDecay {
        original_confidence: Confidence,
        time_elapsed: Duration,
        decay_rate: f32,
    },
    
    DirectMatch {
        cue_id: String,
        similarity_score: f32,
        match_type: MatchType,
    },
    
    VectorSimilarity(Box<VectorSimilarityEvidence>),
}

pub enum MatchType {
    Embedding,
    Semantic,
    Temporal,
    Context,
}
```

**Evidence Chain**:
```rust
pub struct Evidence {
    pub source: EvidenceSource,
    pub strength: Confidence,
    pub timestamp: SystemTime,
    pub dependencies: Vec<EvidenceId>,  // For circular dependency detection
}
```

---

### 2.4 Uncertainty Sources

**Types**:
```rust
pub enum UncertaintySource {
    SystemPressure {
        pressure_level: f32,
        effect_on_confidence: f32,
    },
    
    SpreadingActivationNoise {
        activation_variance: f32,
        path_diversity: f32,
    },
    
    TemporalDecayUnknown {
        time_since_encoding: Duration,
        decay_model_uncertainty: f32,
    },
    
    MeasurementError {
        error_magnitude: f32,
        confidence_degradation: f32,
    },
}
```

---

## 3. Activation Spreading Algorithm

### 3.1 ParallelSpreadingEngine

**File**: `/Users/jordan/Workspace/orchard9/engram/engram-core/src/activation/parallel.rs`

**Architecture**:
```
                        ┌─────────────────────┐
                        │   Source Node (1.0) │
                        └──────────┬──────────┘
                                   │
                    ┌──────────────┼──────────────┐
                    ↓              ↓              ↓
              [Neighbor 1]   [Neighbor 2]   [Neighbor 3]
             decay=0.85      decay=0.85     decay=0.85
                    │              │              ↓
                    └──────────┬───┴──→ [Merge]   [N3.1]
                               ↓       (confidence
                            [Merge]     aggregation)
```

**Work-Stealing Thread Pool**:
- Worker count: Configurable (default: num_cpus)
- Work stealing ratio: Probability of stealing vs local work
- Batch size: Tasks processed per work-stealing cycle
- Phase barrier: Synchronization across worker phases

**Key Features**:

1. **Lock-Free Design**: Uses DashMap for concurrent activation tracking
2. **Cache Optimization**: Cache-line aligned `CacheOptimizedNode` for hot fields
3. **Prefetching**: CPU cache prefetch for neighbor traversal
4. **Tier-Aware Scheduling**: Different treatment for hot/warm/cold memories
5. **Latency Budget Management**: Ensures spreading completes within timeout
6. **Cycle Detection**: Prevents infinite loops in graph traversal

---

### 3.2 Spreading Results Structure

**Definition**:
```rust
pub struct SpreadingResults {
    /// Final storage-aware activations
    pub activations: Vec<StorageAwareActivation>,
    
    /// Aggregated statistics per storage tier
    pub tier_summaries: HashMap<StorageTier, TierSummary>,
    
    /// Detected cycle paths
    pub cycle_paths: Vec<Vec<NodeId>>,
    
    /// Deterministic trace of activation flow
    pub deterministic_trace: Vec<TraceEntry>,
}

pub struct StorageAwareActivation {
    pub memory_id: String,
    pub activation_level: AtomicF32,
    pub hop_count: AtomicU16,
    pub confidence: Confidence,
}
```

**Tier Summaries**:
```rust
pub struct TierSummary {
    pub tier: StorageTier,
    pub node_count: usize,
    pub avg_activation: f32,
    pub max_activation: f32,
    pub latency_ms: f32,
}
```

---

### 3.3 Confidence Aggregation

**File**: `/Users/jordan/Workspace/orchard9/engram/engram-core/src/activation/confidence_aggregation.rs`

**Algorithm**:
```rust
pub struct ConfidencePath {
    pub confidence: Confidence,
    pub hop_count: u16,
    pub source_tier: StorageTier,
    pub path_weight: f32,
}

pub fn aggregate_paths(&self, paths: &[ConfidencePath]) -> ConfidenceAggregationOutcome {
    // 1. Evaluate each path with decay
    let evaluated = paths.iter()
        .map(|p| apply_hop_decay(p.confidence, p.hop_count, decay_rate))
        .filter(|c| c.is_finite())
        .collect();
    
    // 2. Sort by probability (descending)
    sort_by_probability(&mut evaluated);
    
    // 3. Limit to max_paths
    let top_paths = take_top_n(&evaluated, max_paths);
    
    // 4. Combine via maximum-likelihood estimation
    // P(A ∨ B) = 1 - (1 - P(A)) × (1 - P(B))
    let log_one_minus_total = top_paths
        .iter()
        .map(|p| (1.0 - p.raw()).ln())
        .sum();
    let aggregate = 1.0 - log_one_minus_total.exp();
    
    ConfidenceAggregationOutcome {
        aggregate,
        contributing_paths: top_paths,
        skipped_paths: len - top_paths.len(),
    }
}
```

**Hop-Dependent Decay**:
- Each hop multiplies confidence by `exp(-decay_rate)`
- Default decay_rate: 0.35
- Path relevance decreases with distance
- Minimum confidence threshold: 0.001

---

## 4. Query Execution Pipeline

### 4.1 QueryExecutor (AST Routing)

**File**: `/Users/jordan/Workspace/orchard9/engram/engram-core/src/query/executor/query_executor.rs`

**Architecture**:
```
Query Text
    ↓
[Parser] → AST
    ↓
[QueryExecutor::execute()]
    ├─ Validate memory space exists
    ├─ Check query complexity (cost < limit)
    ├─ Route to handler (RECALL/SPREAD/etc.)
    └─ Enforce timeout
    ↓
[Handler] (RECALL/SPREAD/CONSOLIDATE/IMAGINE/PREDICT)
    ↓
[ProbabilisticQueryExecutor]
    ├─ Extract evidence from paths
    ├─ Aggregate confidence
    └─ Track uncertainty
    ↓
ProbabilisticQueryResult
```

**Configuration**:
```rust
pub struct AstQueryExecutorConfig {
    pub default_timeout: Duration,      // Default: 30s
    pub track_evidence: bool,           // Default: true
    pub max_query_cost: u64,            // Default: 100,000
}
```

**Timeout Enforcement**:
```rust
// Timeout-wrapped execution
let result = tokio::time::timeout(
    effective_timeout,
    self.execute_inner(query, context, space_handle)
).await;

// Returns Err if timeout exceeded
```

**Memory Space Validation**:
```rust
// Multi-tenant isolation
let space_handle = registry.get(&context.memory_space_id)?;
// Each space isolated, no cross-contamination
```

---

### 4.2 ProbabilisticQueryExecutor (Evidence Aggregation)

**File**: `/Users/jordan/Workspace/orchard9/engram/engram-core/src/query/executor/mod.rs`

**Configuration**:
```rust
pub struct QueryExecutorConfig {
    pub decay_rate: f32,                    // Default: 0.35
    pub min_confidence: Confidence,         // Default: 0.001
    pub max_paths: usize,                   // Default: 8
    pub track_evidence: bool,               // Default: true
    pub track_uncertainty: bool,            // Default: true
}
```

**Execute Method Signature**:
```rust
pub fn execute(
    &self,
    episodes: Vec<(Episode, Confidence)>,
    activation_paths: &[ActivationPath],
    uncertainty_sources: Vec<UncertaintySource>,
) -> ProbabilisticQueryResult
```

**Activation Path Structure**:
```rust
pub struct ActivationPath {
    pub source_episode_id: String,
    pub target_episode_id: String,
    pub activation: Activation,
    pub confidence: Confidence,
    pub hop_count: u16,
    pub source_tier: StorageTier,
    pub weight: f32,
}
```

**Processing Steps**:
1. Create base result from episodes
2. Extract evidence from activation paths
3. Convert paths to confidence paths for aggregation
4. Aggregate confidence with `ConfidenceAggregator`
5. Calculate uncertainty from path diversity
6. Return comprehensive result with evidence chain

---

## 5. Integration Points for Distributed Execution

### 5.1 Scatter-Gather Pattern

**Current Single-Node Pattern**:
```
Recall Request
    ↓
[Local MemoryStore::recall()]
    ↓
Results + Activation Paths
    ↓
[Merge Results]
    ↓
ProbabilisticQueryResult
```

**Distributed Pattern (M14)**:
```
Recall Request to Coordinator
    ├─→ [Node 1] recall() → Results₁
    ├─→ [Node 2] recall() → Results₂
    └─→ [Node 3] recall() → Results₃
    ↓
[Merge Results & Paths]
    ├─ Union episodes
    ├─ Merge evidence chains
    ├─ Combine uncertainties
    └─ Aggregate confidence
    ↓
ProbabilisticQueryResult
```

**Key for Scatter-Gather**:
- `episodes` can be merged via union
- `evidence_chain` accumulates from all nodes
- `confidence_interval` recomputed from all confidences
- `uncertainty_sources` combined

---

### 5.2 Evidence Chain Merging

**Current Implementation**:
```rust
fn merge_evidence_chains(
    &self,
    other_evidence: &[Evidence],
) -> Vec<Evidence> {
    let mut result = self.evidence_chain.clone();
    result.extend_from_slice(other_evidence);
    result
}
```

**For Distributed**:
- Add node origin tracking: `evidence.source.node_id`
- Detect duplicate paths (same source_episode → same target)
- Maintain dependency graph across nodes
- Enable cross-node validation

---

### 5.3 Confidence Aggregation with Path Information

**For Cross-Partition Spreading**:

1. **Same Node Paths**: Use existing aggregation
2. **Remote Node Paths**: 
   - Adjust for network latency uncertainty
   - Apply tier penalty for remote access
   - Track replication lag
   - Weight by node reliability

**Extended ActivationPath**:
```rust
pub struct ActivationPath {
    // Existing fields
    pub source_episode_id: String,
    pub target_episode_id: String,
    pub activation: Activation,
    pub confidence: Confidence,
    pub hop_count: u16,
    pub source_tier: StorageTier,
    pub weight: f32,
    
    // NEW for distributed:
    pub origin_node: NodeId,        // Which node originated path
    pub network_latency_ms: u32,    // RTT to that node
    pub replication_lag: Duration,  // Staleness of data
}
```

---

## 6. Memory Space Boundaries and Isolation

### 6.1 MemorySpaceRegistry

**File**: `/Users/jordan/Workspace/orchard9/engram/engram-core/src/registry/memory_space.rs`

**Multi-Tenant Structure**:
```rust
pub struct MemorySpaceRegistry {
    handles: DashMap<MemorySpaceId, Arc<SpaceHandle>>,
    init_lock: Mutex<()>,
    factory: Arc<StoreFactory>,
    persistence_root: PathBuf,
}

pub struct SpaceHandle {
    id: MemorySpaceId,
    store: Arc<MemoryStore>,
    directories: SpaceDirectories,
}

pub struct SpaceDirectories {
    root: PathBuf,      // /root/{space_id}/
    wal: PathBuf,       // /root/{space_id}/wal/
    hot: PathBuf,       // /root/{space_id}/hot/
    warm: PathBuf,      // /root/{space_id}/warm/
    cold: PathBuf,      // /root/{space_id}/cold/
}
```

**Isolation Mechanism**:
- Each space has independent directory hierarchy
- Each space has own `MemoryStore` instance
- No shared data between spaces
- Registry enforces access control via `MemorySpaceId`

---

### 6.2 QueryContext

**File**: `/Users/jordan/Workspace/orchard9/engram/engram-core/src/query/executor/context.rs`

**Definition**:
```rust
pub struct QueryContext {
    pub memory_space_id: MemorySpaceId,
    pub timeout: Option<Duration>,
    // Additional fields for distributed:
    // pub requesting_node: NodeId,
    // pub request_id: UUID,
    // pub trace_id: String,
}
```

**Implementation**:
```rust
impl QueryContext {
    pub fn with_timeout(space_id: MemorySpaceId, timeout: Duration) -> Self {
        Self {
            memory_space_id: space_id,
            timeout: Some(timeout),
        }
    }
    
    pub fn without_timeout(space_id: MemorySpaceId) -> Self {
        Self {
            memory_space_id: space_id,
            timeout: None,
        }
    }
}
```

---

### 6.3 Cross-Space Queries

**Current Status**: Not supported

**For M14 Distribution**:
1. **Same-Space Queries**: Execute via `registry.get(space_id)`
2. **Cross-Space Queries**:
   - Route to each space's coordinator
   - Gather results from multiple spaces
   - Merge via probabilistic combination
   - Track origin metadata

**Example Extension**:
```rust
pub struct CrossSpaceQuery {
    pub spaces: Vec<MemorySpaceId>,
    pub query: Box<Query>,
    pub merge_strategy: MergeStrategy,
}

pub enum MergeStrategy {
    Union,              // All results combined
    Intersection,       // Only common results
    WeightedAverage,    // By space reliability
}
```

---

## 7. Current Parallelization

### 7.1 Tokio Async Runtime

**Existing Use**:
```rust
// In QueryExecutor::execute()
match tokio::time::timeout(
    effective_timeout,
    self.execute_inner(query, context, space_handle)
).await {
    Ok(Ok(result)) => Ok(result),
    Ok(Err(e)) => Err(e),
    Err(_) => Err(QueryExecutionError::Timeout { ... }),
}
```

**Current Pattern**:
- Async query execution entry point
- Underlying handlers are synchronous (for now)
- Timeout enforcement via tokio::time::timeout

**For Distributed**:
1. **Fan-out Pattern**: Spawn tasks to multiple nodes
```rust
let handles: Vec<_> = nodes
    .iter()
    .map(|node| {
        tokio::spawn(async {
            node.execute_query(query).await
        })
    })
    .collect();

let results = futures::future::join_all(handles).await;
```

2. **Concurrent Recall**:
```rust
pub async fn parallel_recall(
    &self,
    spaces: Vec<MemorySpaceId>,
    cue: &Cue,
) -> Result<ProbabilisticQueryResult> {
    let futures = spaces
        .iter()
        .map(|space_id| {
            let cue = cue.clone();
            async move {
                self.recall_in_space(space_id, &cue).await
            }
        })
        .collect::<Vec<_>>();
    
    let results = join_all(futures).await;
    merge_results(results)
}
```

---

### 7.2 Work-Stealing in Spreading

**Lock-Free Parallelization**:
```rust
pub struct ParallelSpreadingEngine {
    scheduler: Arc<TierAwareSpreadingScheduler>,
    thread_handles: Vec<JoinHandle<()>>,
    // Work-stealing configuration
    work_stealing_ratio: f32,
    batch_size: usize,
}
```

**Current Parallelism**:
- Single-node work stealing (within-process)
- Cross-tier scheduling (hot/warm/cold memories get different priority)
- Cache-line optimization for false sharing prevention

**For Distributed**:
- Extend scheduler to span multiple nodes
- Remote work stealing protocol (gossip-based)
- Cross-partition graph traversal

---

## 8. Integration with Routing Layer (Task 006)

### 8.1 Current Routing

**QueryExecutor Routes To**:
```rust
match query {
    Query::Recall(q) => self.execute_recall(&q, context, &space_handle),
    Query::Spread(q) => Self::execute_spread(&q, context, &space_handle),
    Query::Predict(q) => Self::execute_predict(&q, context, &space_handle),
    Query::Imagine(q) => Self::execute_imagine(&q, context, &space_handle),
    Query::Consolidate(q) => Self::execute_consolidate(&q, context, &space_handle),
}
```

**Space Validation**:
```rust
let space_handle = self.registry.get(&context.memory_space_id)?;
```

---

### 8.2 For Distributed Routing (Task 006 Extension)

**Router Interface**:
```rust
pub trait DistributedRouter {
    async fn route_query(
        &self,
        query: &Query,
        context: &QueryContext,
    ) -> Result<RoutingDecision>;
}

pub enum RoutingDecision {
    Local(Arc<SpaceHandle>),
    Remote {
        nodes: Vec<NodeId>,
        merge_strategy: MergeStrategy,
    },
    Scatter {
        spaces: Vec<MemorySpaceId>,
        merge_strategy: MergeStrategy,
    },
}
```

**Routing Logic**:
1. **RECALL**: Direct to partition, scatter-gather if multi-space
2. **SPREAD**: Local engine (may cross partitions), gather results
3. **CONSOLIDATE**: Coordinator triggers on all replicas
4. **IMAGINE**: Requires full graph context, may need broadcast
5. **PREDICT**: Aggregates temporal patterns across partitions

---

## 9. Specific File Paths for Each Implementation

| Component | File | Key Structs |
|-----------|------|------------|
| **Query Types** | `engram-core/src/query/parser/ast.rs` | Query enum variants |
| **RECALL** | `engram-core/src/query/executor/recall.rs` | RecallExecutor, RecallExecutionError |
| **SPREAD** | `engram-core/src/query/executor/spread.rs` | SpreadExecutionError, transform_spreading_results |
| **CONSOLIDATE** | `engram-core/src/query/executor/consolidate.rs` | ConsolidateExecutionError |
| **IMAGINE** | `engram-core/src/query/executor/imagine.rs` | ImagineExecutionError |
| **PREDICT** | `engram-core/src/query/executor/predict.rs` | PredictExecutionError |
| **Main Executor** | `engram-core/src/query/executor/query_executor.rs` | QueryExecutor, AstQueryExecutorConfig, QueryExecutionError |
| **Prob. Executor** | `engram-core/src/query/executor/mod.rs` | ProbabilisticQueryExecutor, ActivationPath |
| **Result Types** | `engram-core/src/query/mod.rs` | ProbabilisticQueryResult, Evidence, ConfidenceInterval |
| **Spreading** | `engram-core/src/activation/parallel.rs` | ParallelSpreadingEngine, SpreadingResults |
| **Confidence Agg.** | `engram-core/src/activation/confidence_aggregation.rs` | ConfidenceAggregator, ConfidencePath |
| **Registry** | `engram-core/src/registry/memory_space.rs` | MemorySpaceRegistry, SpaceHandle |
| **Context** | `engram-core/src/query/executor/context.rs` | QueryContext |

---

## 10. Key Integration Points for Task 009

### 10.1 Query Execution Entry Point

**Location**: `engram-core/src/query/executor/query_executor.rs::QueryExecutor::execute()`

**Signature**:
```rust
pub async fn execute(
    &self,
    query: Query<'_>,
    context: QueryContext,
) -> Result<ProbabilisticQueryResult, QueryExecutionError>
```

**For Distribution**: Extend to handle remote node coordination

---

### 10.2 Result Merging Points

**Location 1**: `engram-core/src/query/mod.rs`

```rust
impl ProbabilisticQueryResult {
    pub fn and(&self, other: &Self) -> Self {  // Intersection
    pub fn or(&self, other: &Self) -> Self {   // Union
```

**Use Case**: Combine results from multiple nodes' recall operations

---

### 10.3 Evidence Chain Construction

**Location**: `engram-core/src/query/executor/mod.rs`

```rust
fn extract_evidence_from_paths(paths: &[ActivationPath]) -> Vec<Evidence>
```

**For Distribution**: Track which node/partition each evidence came from

---

### 10.4 Confidence Aggregation Integration

**Location**: `engram-core/src/activation/confidence_aggregation.rs`

```rust
pub fn aggregate_paths(&self, paths: &[ConfidencePath]) -> ConfidenceAggregationOutcome
```

**For Distribution**: Support cross-partition paths with adjusted weights

---

## 11. Summary: Ready for Task 009 Implementation

### Key Strengths

1. **Modular Architecture**: Each query type has clean separation of concerns
2. **Evidence Tracking**: Complete provenance available for distributed auditing
3. **Timeout Safety**: Built-in timeout enforcement prevents hanging
4. **Multi-Tenant Ready**: Registry enforces space isolation
5. **Confidence Composition**: AND/OR/NOT operations support result merging
6. **Spread Foundation**: Parallel engine ready to extend to distributed

### Implementation Readiness

| Aspect | Status | Notes |
|--------|--------|-------|
| Query routing | Ready | QueryExecutor handles all types |
| Result format | Ready | ProbabilisticQueryResult extensible |
| Async framework | Ready | tokio integration present |
| Timeout safety | Ready | QueryExecutor enforces limits |
| Evidence tracking | Ready | Complete chains maintained |
| Space isolation | Ready | Registry enforces boundaries |
| Confidence handling | Ready | Aggregation logic proven |
| Spreading algorithm | Ready | Can extend to cross-partition |

### Next Steps for Task 009

1. **Design scatter-gather protocol** for multi-space queries
2. **Extend ActivationPath** with origin tracking
3. **Implement result merge strategies** (union, intersection, weighted)
4. **Add cross-node evidence correlation** detection
5. **Design gossip convergence** for result consistency
6. **Implement timeout propagation** across network calls

