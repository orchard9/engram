# Task 006 (Distributed Routing Layer) - Comprehensive Implementation Analysis

**Date**: 2025-11-01  
**Thorougness Level**: Very Thorough  
**Repository**: /Users/jordan/Workspace/orchard9/engram  

---

## Executive Summary

This report provides detailed implementation guidance for Task 006 (Distributed Routing Layer) in Milestone 14. Based on a comprehensive analysis of the Engram codebase, the routing layer must intercept all query execution paths and direct operations to the appropriate cluster nodes based on memory space assignments. The analysis identifies exact integration points, current operation flows, API handler signatures, and existing patterns to follow.

---

## 1. Query Execution Entry Points

### 1.1 Store Operation Entry Point

**File**: `/Users/jordan/Workspace/orchard9/engram/engram-core/src/store.rs`

**Operation Signature** (Line 1151):
```rust
pub fn store(&self, episode: Episode) -> StoreResult
```

**Current Flow**:
1. Client → gRPC `remember()` handler (engram-cli/src/grpc.rs:134)
2. Handler extracts memory_space_id, resolves space via registry
3. Calls `store.store(episode)` (line 173 in grpc.rs)
4. Returns `StoreResult { activation, streaming_delivered }`

**Routing Integration Point**:
- Before: `store.store(episode)`
- After: `router.route_store(space_id).await` → determine target node
- Then: `router.execute_with_retry(&decision, |node| { execute_store_on_node(node, episode) })`

**StoreResult Structure** (line 88):
```rust
pub struct StoreResult {
    pub activation: Activation,
    pub streaming_delivered: bool,
}
```

### 1.2 Recall Operation Entry Point

**File**: `/Users/jordan/Workspace/orchard9/engram/engram-core/src/store.rs`

**Operation Signature** (Line 1786):
```rust
pub fn recall(&self, cue: &Cue) -> RecallResult
pub fn recall_with_mode(&self, cue: &Cue, mode: RecallMode) -> RecallResult
pub fn recall_probabilistic(&self, cue: &Cue) -> ProbabilisticQueryResult
```

**Current Flow**:
1. Client → gRPC `recall()` handler (engram-cli/src/grpc.rs:243)
2. Handler extracts cue_type (Semantic/Embedding/etc.)
3. Calls `store.recall(&cue)` (line 293 or similar in grpc.rs)
4. Returns `RecallResult { results, streaming_delivered }`

**RecallResult Structure** (line 120):
```rust
pub struct RecallResult {
    pub results: Vec<(Episode, Confidence)>,
    pub streaming_delivered: bool,
}
```

**Routing Integration Point**:
- Operation: `router.route_recall(space_id).await` → returns primary + replicas
- Retry on primary failure with replica fallback
- Response includes which node served the result (for debugging)

### 1.3 Consolidation Operation Entry Point

**File**: `/Users/jordan/Workspace/orchard9/engram/engram-core/src/consolidation/mod.rs`

**Current Implementation**:
- Consolidation runs locally on each node
- Results are synced via gossip protocol (Task 007)
- No remote consolidation calls needed

**Routing Strategy**:
- `route_consolidation()` returns LOCAL_ONLY strategy
- No remote execution
- Integration point: gossip delivery in Task 007

---

## 2. Current API Layer Analysis

### 2.1 gRPC Service Implementation

**File**: `/Users/jordan/Workspace/orchard9/engram/engram-cli/src/grpc.rs`

**Key Service Structure** (Line 41):
```rust
pub struct MemoryService {
    store: Arc<MemoryStore>,
    metrics: Arc<MetricsRegistry>,
    registry: Arc<MemorySpaceRegistry>,
    default_space: MemorySpaceId,
    streaming_handlers: Arc<StreamingHandlers>,
}
```

**Key Handler Method Pattern** (Line 134):
```rust
async fn remember(
    &self,
    request: Request<RememberRequest>,
) -> Result<Response<RememberResponse>, Status>
```

**Memory Space Resolution** (Line 100-125):
```rust
fn resolve_memory_space(
    &self,
    request_space_id: &str,
    _metadata: &tonic::metadata::MetadataMap,
) -> Result<MemorySpaceId, Status>
```

Current logic:
1. Check explicit memory_space_id in request (Priority 1)
2. TODO: Check X-Memory-Space header (Priority 2)
3. Fallback to default_space (Priority 3)

**Handler Entry Points** (implements EngramService):
- `remember()` (Line 134) - STORE operations
- `recall()` (Line 243) - READ operations
- `consolidate()` - CONSOLIDATION operations
- `dream()` - PATTERN-based operations
- `associate()` - LINKING operations

### 2.2 HTTP REST API Layer

**File**: `/Users/jordan/Workspace/orchard9/engram/engram-cli/src/api.rs`

**API State Structure** (Line 43):
```rust
pub struct ApiState {
    pub store: Arc<MemoryStore>,
    pub memory_service: Arc<MemoryService>,
    pub registry: Arc<MemorySpaceRegistry>,
    pub default_space: MemorySpaceId,
    pub metrics: Arc<MetricsRegistry>,
    pub auto_tuner: Arc<SpreadingAutoTuner>,
    pub shutdown_tx: Arc<tokio::sync::watch::Sender<bool>>,
    pub session_manager: Arc<SessionManager>,
    pub observation_queue: Arc<ObservationQueue>,
}
```

**Routes** (via `create_api_routes()`):
- POST `/remember` → store operation
- GET `/recall` → read operation  
- GET `/health` → health check
- WebSocket streaming endpoints

### 2.3 Memory Space Registry

**File**: `/Users/jordan/Workspace/orchard9/engram/engram-core/src/registry/memory_space.rs`

**Key Method** (Line 107):
```rust
pub async fn create_or_get(&self, space_id: &MemorySpaceId) -> Result<SpaceHandle, MemorySpaceError>
```

**SpaceHandle Interface**:
```rust
pub fn store(&self) -> Arc<MemoryStore>
pub fn handle(&self) -> Arc<SpaceHandle>
```

Current usage in gRPC handlers:
```rust
let handle = self.registry.create_or_get(&space_id).await?;
let store = handle.store();
let result = store.recall(&cue);
```

**ROUTING INTEGRATION**: Space resolution happens BEFORE store access
- Modify registry to return routing decision instead of store
- Or: Add routing layer between registry and store

---

## 3. Existing Client/Connection Patterns

### 3.1 gRPC Channel Management

**Framework**: `tonic` (v0.12)

**Current Usage** (Line 6 of grpc.rs):
```rust
use tonic::{Request, Response, Status, transport::Server};
```

**Server Startup** (grpc.rs Line 87-92):
```rust
Server::builder()
    .add_service(EngramServiceServer::new(self))
    .serve(addr)
    .await?
```

**Pattern Observed**:
- No connection pooling currently (single server instance)
- HTTP/2 multiplexing built-in to tonic
- Keep-alive configuration available via `transport::Endpoint`

### 3.2 Tokio Async Runtime

**Pattern**: All I/O is async via tokio
- `#[tokio::main]` in main.rs
- `tokio::sync` for synchronization
- Async/await throughout

**Existing Sync Primitives**:
- `Arc<RwLock<T>>` via parking_lot (store.rs)
- `DashMap<K, V>` for concurrent collections (store.rs)
- `Arc<tokio::sync::RwLock<T>>` for async locks (metrics)

### 3.3 Error Handling Pattern

**Crate**: `thiserror`

**Example from store.rs**:
```rust
#[derive(Debug, thiserror::Error)]
pub enum StoreError {
    #[error("Memory not found: {id}")]
    NotFound { id: String },
}
```

**HTTP Error Mapping** (api.rs):
```rust
Status::internal(format!("Failed to access memory space '{}': {}", space_id, e))
```

---

## 4. Memory Space Handle Usage

### 4.1 Current Space Resolution Pattern

**File**: engram-cli/src/grpc.rs (Line 142-151)

```rust
// Extract memory space from request
let space_id = self.resolve_memory_space(&req.memory_space_id, &metadata)?;

// Get space-specific store handle from registry
let handle = self.registry.create_or_get(&space_id).await.map_err(|e| {
    Status::internal(format!(
        "Failed to access memory space '{}': {}",
        space_id, e
    ))
})?;

// Extract store from handle
let store = handle.store();

// Perform operation
let store_result = store.store(episode);
```

### 4.2 Operations Receive Space Context

**Critical Finding**: All handler methods receive space_id as part of request, then:
1. Resolve space_id via `resolve_memory_space()`
2. Get space handle via `registry.create_or_get()`
3. Extract store from handle
4. Perform operation on local store

**Routing Integration Point**: Between steps 2 and 3
- After space_id resolved: determine which node hosts this space
- Route to correct node or execute locally
- Return result with metadata about which node served request

### 4.3 Space ID Structure

**Type**: `MemorySpaceId` (engram-core/src/types.rs)

**Validation**:
- 4-64 lowercase alphanumeric characters
- Used as key in registry
- Used in persistence (WAL paths, storage tiers)
- Immutable once created

---

## 5. Existing Retry and Error Handling

### 5.1 Current Error Propagation

**No retry logic currently exists** - Direct pass-through:

From store.rs (line 1169):
```rust
match self.graph.store_episode(wal_episode.clone()) {
    Ok(_) => 0.0,
    Err(error) => {
        tracing::warn!(?error, "failed to record episode in unified memory graph");
        0.1  // Penalty applied, not failure
    }
}
```

**Pattern**: Graceful degradation, not error propagation
- Operations return activation levels (0.0-1.0)
- No operation can fail at API level
- Errors are absorbed as reduced activation

### 5.2 gRPC Error Handling

**Pattern** (grpc.rs line 145-150):
```rust
.map_err(|e| {
    Status::internal(format!(
        "Failed to access memory space '{}': {}",
        space_id, e
    ))
})?
```

**Error Types Used**:
- `Status::invalid_argument()` - Bad request
- `Status::internal()` - Server error
- `Status::not_found()` - Resource missing

### 5.3 Storage Error Handling

**Pattern**: Penalty-based degradation
- Failures reduce activation/confidence
- No explicit retry loops
- System continues operation

**Routing Will Need**:
1. Connection failures → Retry with backoff
2. Timeout failures → Fallback to replicas
3. Circuit breaker → Fast-fail when node down
4. Partition detection → Local-only mode

---

## 6. Integration Points with Task 004 (Space Assignment)

### 6.1 Dependency Chain

**Task 004** provides `SpaceAssignmentManager`:
```rust
pub struct SpaceAssignmentManager {
    membership: Arc<SwimMembership>,
    assignments: DashMap<MemorySpaceId, SpaceAssignment>,
    placement: Arc<PlacementStrategy>,
    rebalancer: Arc<RebalancingCoordinator>,
    config: AssignmentConfig,
}
```

**Task 004 Provides**:
```rust
pub fn assign_space(&self, space_id: &MemorySpaceId) -> Result<SpaceAssignment, ClusterError>
pub fn get_assignment(&self, space_id: &MemorySpaceId) -> Option<SpaceAssignment>
```

**Returns**:
```rust
pub struct SpaceAssignment {
    pub space_id: MemorySpaceId,
    pub primary_node_id: String,
    pub replica_node_ids: Vec<String>,
    pub version: u64,
    pub assigned_at: chrono::DateTime<chrono::Utc>,
}
```

### 6.2 Router Usage of Assignments

**In Task 006 Router**:
```rust
pub async fn route_store(
    &self,
    space_id: &str,
) -> Result<RoutingDecision, RouterError> {
    // Get space assignment from Task 004
    let assignment = self.assignments.get_assignment(space_id).await?;
    
    // Return routing decision
    Ok(RoutingDecision {
        primary: assignment.primary,
        replicas: assignment.replicas,
        strategy: RoutingStrategy::Primary,
    })
}
```

### 6.3 Integration Flow

```
gRPC Handler (grpc.rs)
    ↓
resolve_memory_space(space_id) 
    ↓
[NEW] router.route_store(space_id)  ← Task 006
    ↓
SpaceAssignmentManager.get_assignment(space_id)  ← Task 004
    ↓
[NEW] router.execute_with_retry(decision, operation)
    ↓
Execute on target node (local or remote)
```

---

## 7. Exact File Paths and Line Numbers

### 7.1 Entry Points Summary

| Operation | File | Lines | Current Behavior |
|-----------|------|-------|------------------|
| Store | grpc.rs | 134-237 | Direct local store |
| Recall | grpc.rs | 243-450+ | Direct local recall |
| Consolidate | grpc.rs | 500+? | Direct local consolidation |
| Space Resolution | grpc.rs | 100-125 | Resolve ID, no routing |
| Registry Access | grpc.rs | 145-151 | Get space handle |

### 7.2 Supporting Type Definitions

| Type | File | Purpose |
|------|------|---------|
| `StoreResult` | store.rs:88 | Return type for store() |
| `RecallResult` | store.rs:120 | Return type for recall() |
| `MemoryService` | grpc.rs:41 | gRPC service impl |
| `ApiState` | api.rs:43 | HTTP state container |
| `SpaceHandle` | registry/memory_space.rs | Space-specific store access |
| `MemorySpaceId` | types.rs | Space identifier type |

### 7.3 Initialization Points

**Main Server Startup** (main.rs line 59):
```rust
Commands::Start { port, grpc_port } => {
    start_server(port, grpc_port, config_manager.config()).await
}
```

**gRPC Server** (line 82-92):
```rust
Server::builder()
    .add_service(EngramServiceServer::new(self))
    .serve(addr)
    .await?
```

**API Server** (creates ApiState):
```rust
let api_state = ApiState {
    store: Arc::clone(&store),
    memory_service: Arc::new(memory_service),
    registry: Arc::clone(&registry),
    ...
}
```

---

## 8. Struct Modifications Required

### 8.1 MemoryService Must Accept Router

**Current**:
```rust
pub struct MemoryService {
    store: Arc<MemoryStore>,
    metrics: Arc<MetricsRegistry>,
    registry: Arc<MemorySpaceRegistry>,
    default_space: MemorySpaceId,
    streaming_handlers: Arc<StreamingHandlers>,
}
```

**After Task 006**:
```rust
pub struct MemoryService {
    store: Arc<MemoryStore>,
    metrics: Arc<MetricsRegistry>,
    registry: Arc<MemorySpaceRegistry>,
    router: Arc<Router>,  // NEW - Task 006
    default_space: MemorySpaceId,
    streaming_handlers: Arc<StreamingHandlers>,
}
```

### 8.2 ApiState Must Accept Router

```rust
pub struct ApiState {
    pub store: Arc<MemoryStore>,
    pub memory_service: Arc<MemoryService>,
    pub registry: Arc<MemorySpaceRegistry>,
    pub router: Arc<Router>,  // NEW - Task 006
    pub default_space: MemorySpaceId,
    ...
}
```

### 8.3 Handler Signature Changes (Minimal)

No signature changes needed - routing happens internally:

```rust
// BEFORE
async fn remember(&self, request: Request<RememberRequest>) -> Result<Response<RememberResponse>, Status>

// AFTER - same signature, different implementation
async fn remember(&self, request: Request<RememberRequest>) -> Result<Response<RememberResponse>, Status> {
    let space_id = self.resolve_memory_space(&req.memory_space_id, &metadata)?;
    
    // NEW: Route operation
    let decision = self.router.route_store(&space_id).await
        .map_err(|e| Status::internal(e.to_string()))?;
    
    // NEW: Execute with retry
    let result = self.router.execute_with_retry(&decision, |node| {
        execute_store_on_remote_node(node, episode)
    }).await?;
    
    Ok(Response::new(result))
}
```

---

## 9. Existing Client Patterns to Follow

### 9.1 Async/Await Pattern

**Current Style** (found throughout):
```rust
pub async fn create_or_get(
    &self, 
    space_id: &MemorySpaceId
) -> Result<SpaceHandle, MemorySpaceError> {
    // Implementation
}
```

Router should follow same pattern:
```rust
pub async fn route_store(
    &self, 
    space_id: &str
) -> Result<RoutingDecision, RouterError> {
    // Implementation
}
```

### 9.2 Error Handling with thiserror

**Pattern**:
```rust
#[derive(Debug, thiserror::Error)]
pub enum StoreError {
    #[error("Memory not found: {id}")]
    NotFound { id: String },
}
```

Router errors should follow:
```rust
#[derive(Debug, thiserror::Error)]
pub enum RouterError {
    #[error("No assignment found for space: {space_id}")]
    NoAssignment { space_id: String },
    
    #[error("Circuit breaker open for node {node_id}")]
    CircuitBreakerOpen { node_id: String },
}
```

### 9.3 DashMap for Concurrent Collections

**Pattern** (store.rs):
```rust
pub hot_memories: DashMap<String, Arc<Memory>>
```

Router connection pool should use:
```rust
connections: Arc<DashMap<String, ChannelPool>>
```

### 9.4 Metrics Integration

**Current Pattern** (store.rs):
```rust
#[cfg(feature = "monitoring")]
let start = Instant::now();

#[cfg(feature = "monitoring")]
use crate::metrics::cognitive::CognitiveMetric;
```

Router should expose metrics:
```rust
pub fn record_routing_latency(&self, latency_ms: f64, node_id: &str)
pub fn record_retry_attempt(&self, attempt: usize)
pub fn record_circuit_breaker_state(&self, node_id: &str, open: bool)
```

---

## 10. Testing Patterns

### 10.1 Test Helpers Needed

Based on Task 006 spec, tests need:
```rust
fn test_node() -> NodeInfo
fn primary_node() -> NodeInfo  
fn replica1() -> NodeInfo
fn local_node() -> NodeInfo
fn test_router() -> Router
```

### 10.2 Integration Test Pattern

From existing tests (completion module):
```rust
#[tokio::test]
async fn test_router_with_real_cluster() {
    let cluster = TestCluster::new(3).await;
    let space_id = "test_space";
    cluster.assign_space(space_id, 0, vec![1, 2]).await;
    
    let router = cluster.router(0);
    let decision = router.route_store(space_id).await.unwrap();
    
    assert_eq!(decision.primary.id, cluster.node(0).id);
}
```

### 10.3 Benchmark Pattern

From benches/ directory:
```rust
use criterion::{black_box, criterion_group, criterion_main, Criterion};

fn bench_routing_decision(c: &mut Criterion) {
    c.bench_function("route_store", |b| {
        b.to_async(&runtime).iter(|| async {
            black_box(router.route_store("test_space").await)
        });
    });
}
```

---

## 11. Feature Flags and Modularity

### 11.1 Current Feature Flags

From Cargo.toml (in spec):
```toml
[features]
default = ["hnsw_index", "memory_mapped_persistence", "monitoring"]
hnsw_index = []
memory_mapped_persistence = []
monitoring = []
psychological_decay = []
pattern_completion = []
security = []
probabilistic_queries = []
```

### 11.2 Router Should Be

Optional feature: `distributed` or always-on since it's core to M14:
```toml
[features]
default = ["...", "distributed_routing"]
distributed_routing = []
```

### 11.3 Module Declaration

**engram-core/src/lib.rs** should add:
```rust
#[cfg(feature = "distributed_routing")]
pub mod cluster;
```

---

## 12. Integration Checklist for Implementation

### Phase 1: Infrastructure Setup
- [ ] Create `engram-core/src/cluster/` directory
- [ ] Create `engram-core/src/cluster/mod.rs`
- [ ] Define error types: `RouterError`, `CircuitBreakerError`, `PoolError`
- [ ] Export router module in lib.rs

### Phase 2: Core Components
- [ ] Implement `CircuitBreaker` (simplest, testable in isolation)
- [ ] Implement `ConnectionPool` (depends on tonic)
- [ ] Implement `Router` (orchestrates everything)

### Phase 3: Integration with gRPC
- [ ] Add `router: Arc<Router>` to `MemoryService`
- [ ] Modify `remember()` handler to use router
- [ ] Modify `recall()` handler to use router
- [ ] Modify `consolidate()` handler (local-only)

### Phase 4: Integration with HTTP API
- [ ] Add `router: Arc<Router>` to `ApiState`
- [ ] Update HTTP handlers to use router
- [ ] Ensure error responses map correctly

### Phase 5: Testing
- [ ] Unit tests for CircuitBreaker
- [ ] Unit tests for ConnectionPool
- [ ] Unit tests for Router routing logic
- [ ] Integration tests with mock cluster
- [ ] Benchmarks for latency targets

### Phase 6: Monitoring
- [ ] Add routing latency metrics
- [ ] Add retry count metrics
- [ ] Add circuit breaker state tracking
- [ ] Add pool utilization metrics

---

## 13. Key Implementation Decisions

### 13.1 Router Initialization Point

**Where**: `main.rs` start_server() function

**What**: After Task 001-004 complete, initialize router before creating gRPC service:

```rust
// 1. Initialize SWIM membership (Task 001)
let membership = Arc::new(SwimMembership::new(...));

// 2. Initialize node discovery (Task 002)
// 3. Initialize partition detector (Task 003)
// 4. Initialize space assignment (Task 004)

let assignments = Arc::new(SpaceAssignmentManager::new(...));

// 5. NEW - Initialize router (Task 006)
let connection_pool = Arc::new(ConnectionPool::new(PoolConfig::default()));
let partition_detector = Arc::new(PartitionDetector::new(...));
let router = Arc::new(Router::new(
    membership,
    assignments,
    connection_pool,
    partition_detector,
    RouterConfig::default(),
));

// 6. Pass router to MemoryService
let memory_service = Arc::new(MemoryService::new(
    ...,
    router,  // NEW
));
```

### 13.2 Remote Operation Execution

**How gRPC handlers execute remote operations**:

```rust
// In handler:
let decision = self.router.route_store(space_id).await?;

let response = self.router.execute_with_retry(&decision, |node| {
    // Clone necessary data for async move
    let episode_clone = episode.clone();
    let service_clone = Arc::clone(&self);
    
    async move {
        // Create connection to remote node
        let channel = connection_pool.get_channel(&node).await?;
        
        // Create gRPC client
        let mut client = EngramServiceClient::new(channel);
        
        // Execute RPC
        let request = RememberRequest { /* filled */ };
        let response = client.remember(Request::new(request)).await?;
        
        Ok(response)
    }
}).await?;
```

### 13.3 Local vs Remote Execution

Current behavior (all local):
```rust
let store = handle.store();
let result = store.recall(&cue);
```

After routing:
```rust
// Route determines which node
let decision = router.route_recall(space_id).await?;

// Execute with retry and fallback
let result = router.execute_with_retry(&decision, |node| {
    if node.is_local() {
        // Local execution - direct call
        let store = handle.store();
        async move { Ok(store.recall(&cue)) }
    } else {
        // Remote execution - gRPC call
        execute_recall_remote(node, cue)
    }
}).await?;
```

---

## 14. Performance Targets and Metrics

### 14.1 Routing Overhead Targets

| Operation | Target P99 | Measured |
|-----------|-----------|----------|
| Route decision computation | <50μs | TBD |
| Connection pool acquisition (cached) | <10μs | TBD |
| Connection pool acquisition (new) | <100ms | TBD |
| Circuit breaker state check | <1μs | TBD |
| Exponential backoff calculation | <100μs | TBD |
| **Total routing overhead** | **<1ms** | TBD |

### 14.2 Throughput Targets

- Routing should not impact throughput
- Connection pool should handle 100s of concurrent RPCs per connection
- Circuit breaker should have <1ns state change

### 14.3 Metrics to Collect

```rust
pub struct RoutingMetrics {
    routing_decision_latency: Histogram,  // microseconds
    connection_pool_hit_rate: Gauge,     // 0-1
    circuit_breaker_state: Gauge,        // 0=closed, 1=open, 2=half-open
    retry_count: Counter,                // total retries
    retry_latency_sum: Counter,          // total backoff delay
    remote_call_count: Counter,          // RPC invocations
    remote_call_latency: Histogram,      // RPC roundtrip time
}
```

---

## 15. Open Questions and Unknowns

### 15.1 Remote Operation Execution

**Question**: How to encode operation details for remote execution?

**Options**:
1. Serialize operation parameters, send to remote, deserialize and execute
2. Stream operation data via new RPC methods
3. Pre-position data on replica nodes

**Implication**: Task 006 must define RPC signatures for remote execution

### 15.2 Partition Detection Integration

**Question**: How does PartitionDetector (Task 003) signal to Router?

**Current**: Router calls `partition_detector.is_partitioned().await`

**Question**: What's the timeout? Should it be cached?

### 15.3 Rebalancing During Operations

**Question**: What happens to in-flight operations when space rebalances?

**Options**:
1. Fail the operation
2. Follow the rebalancing
3. Complete on old primary, let anti-entropy sync

---

## 16. Conclusion

The Distributed Routing Layer (Task 006) must be inserted between the gRPC/HTTP handlers and the MemoryStore operations. The integration point is minimal and non-breaking:

1. Add `router: Arc<Router>` to `MemoryService` and `ApiState`
2. In each handler, call `router.route_operation(space_id)` before executing
3. Use `router.execute_with_retry()` to handle failures and retries
4. For local execution, bypass remote call; for remote, use gRPC client

The router depends on Task 001 (SWIM), Task 002 (Discovery), Task 003 (Partition detection), and Task 004 (Space assignment) being complete. It enables Tasks 007-009 (Gossip, Conflict resolution, Distributed queries).

**Estimated Implementation Effort**: 3-4 days  
**Estimated Testing Effort**: 2-3 days  
**Critical Path Impact**: Blocks all distributed operations (Tasks 007-012)

