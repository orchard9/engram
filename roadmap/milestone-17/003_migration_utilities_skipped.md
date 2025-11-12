# Task 003: Migration Utilities

**STATUS**: ⏭️ SKIPPED - Deferred to Production Readiness Phase

## Why This Task is Skipped

**Decision Date**: 2025-11-09

**Rationale**:
1. **No Production Data to Migrate** - Engram is still in R&D phase with no production deployments requiring data migration
2. **Premature Optimization** - Building migration tooling before proving dual memory architecture works is premature
3. **Better Value Path** - Focus should be on implementing core dual memory capabilities (concept formation, consolidation) to validate the architecture
4. **Fresh Start Available** - Can test dual memory with fresh data, no migration needed
5. **Complexity vs Value** - Migration utilities are complex operational tooling that solves a problem we don't yet have

**When This Becomes Relevant**:
- Production deployment with existing single-type Memory data
- Users depending on system uptime (requiring online migration)
- Need to preserve existing semantic knowledge during upgrade
- Rollback capabilities become critical due to user risk

**Alternative Approach for Now**:
- Test dual memory with fresh synthetic data
- Create episodes directly as `DualMemoryNode::Episode` type
- Build concept formation and prove it works
- Return to migration when approaching production readiness

**Deferred To**: Milestone 17.5 (Production Readiness) or Milestone 18 (Deployment Tooling)

---

## Original Objective
Build production-grade utilities to migrate existing Memory nodes to DualMemoryNode format, including bulk migration tools, compatibility layers, and operational safety guarantees.

## Background
We need a smooth migration path from the current single-type system to dual memory without disrupting existing deployments. This requires careful checkpoint management, integrity verification, and support for both zero-downtime online migrations and fast offline bulk migrations.

## Requirements
1. Create migration tool to convert Memory to DualMemoryNode with checkpointing
2. Implement compatibility layer for gradual migration
3. Add migration progress tracking and resumption via persistent checkpoints
4. Provide rollback capabilities with cryptographic verification
5. Support both online (zero-downtime) and offline (bulk) migration modes
6. Implement rate limiting and backpressure for resource control
7. Add comprehensive dry-run mode with integrity validation
8. Create operational runbooks for migration monitoring

## Technical Specification

### Files to Create
- `engram-core/src/migration/dual_memory.rs` - Core migration engine
- `engram-core/src/migration/checkpoint.rs` - Checkpoint persistence and recovery
- `engram-core/src/migration/verification.rs` - Integrity validation and rollback
- `engram-cli/src/cli/migrate_dual.rs` - CLI migration command (extends existing migrate.rs)
- `engram-core/src/compat/dual_memory.rs` - Compatibility adapter layer
- `docs/operations/migration-dual-memory.md` - Migration runbook

### Files to Modify
- `engram-cli/src/cli/commands.rs` - Add migrate-dual subcommand
- `engram-cli/src/cli/mod.rs` - Wire up new migration module

### Checkpoint Format

Use `bincode` for high-performance serialization (already approved in chosen_libraries.md):

```rust
/// Checkpoint format stored in {space_id}/migration/checkpoint.bin
#[derive(Serialize, Deserialize)]
pub struct MigrationCheckpoint {
    /// Checkpoint format version for forward compatibility
    pub version: u32,
    /// Space identifier being migrated
    pub space_id: MemorySpaceId,
    /// Last successfully migrated NodeId
    pub last_migrated_node: NodeId,
    /// Total nodes to migrate (snapshot at start)
    pub total_nodes: u64,
    /// Nodes successfully migrated so far
    pub migrated_count: u64,
    /// Migration mode (online/offline)
    pub mode: MigrationMode,
    /// SHA256 hash of source data at checkpoint time
    pub source_hash: [u8; 32],
    /// Timestamp of last checkpoint write
    pub checkpoint_time: SystemTime,
    /// Migration configuration snapshot
    pub config: MigrationConfig,
}

impl MigrationCheckpoint {
    /// Persist checkpoint using bincode + fsync
    pub fn save(&self, path: &Path) -> Result<(), MigrationError>;

    /// Load checkpoint with integrity verification
    pub fn load(path: &Path) -> Result<Self, MigrationError>;

    /// Verify source data hasn't changed since checkpoint
    pub fn verify_source_integrity(&self, backend: &MemoryBackend) -> Result<bool, MigrationError>;
}
```

### Migration Engine Architecture

```rust
/// Production migration orchestrator with resource management
pub struct DualMemoryMigrator {
    space_id: MemorySpaceId,
    source: Arc<dyn MemoryBackend>,
    target: Arc<dyn DualMemoryBackend>,

    // Progress tracking
    progress: Arc<AtomicU64>,
    total_nodes: Arc<AtomicU64>,

    // Checkpointing
    checkpoint_dir: PathBuf,
    checkpoint_interval: usize, // checkpoint every N nodes

    // Rate limiting
    rate_limiter: RateLimiter,
    memory_budget_mb: usize,

    // Metrics
    metrics: Arc<MigrationMetrics>,
}

impl DualMemoryMigrator {
    /// Create new migrator with resource limits
    pub fn new(
        space_id: MemorySpaceId,
        source: Arc<dyn MemoryBackend>,
        target: Arc<dyn DualMemoryBackend>,
        config: MigrationConfig,
    ) -> Result<Self, MigrationError>;

    /// Zero-downtime online migration with read/write concurrency
    ///
    /// Strategy:
    /// 1. Maintain read availability from source during migration
    /// 2. Shadow writes to both source and target
    /// 3. Batch migrate in background with rate limiting
    /// 4. Atomic cutover when migration complete
    pub async fn migrate_online(&self, batch_size: usize) -> Result<MigrationStats, MigrationError>;

    /// Fast offline bulk migration (service downtime required)
    ///
    /// Strategy:
    /// 1. Stop all writes to source
    /// 2. Snapshot source tier checksums
    /// 3. Parallel batch migration with rayon
    /// 4. Verify target integrity
    /// 5. Atomic swap of storage backend
    pub async fn migrate_offline(&self) -> Result<MigrationStats, MigrationError>;

    /// Resume migration from last checkpoint
    pub async fn resume(&self) -> Result<MigrationStats, MigrationError>;

    /// Dry-run migration with validation only (no writes)
    pub async fn dry_run(&self) -> Result<ValidationReport, MigrationError>;

    /// Rollback to pre-migration state using checkpoints
    pub async fn rollback(&self, checkpoint: &MigrationCheckpoint) -> Result<RollbackStats, MigrationError>;
}

/// Rate limiter for memory budget enforcement
pub struct RateLimiter {
    max_batch_size: usize,
    memory_budget_bytes: usize,
    current_memory_usage: AtomicUsize,
    backpressure_threshold: f32, // 0.0-1.0
}

impl RateLimiter {
    /// Calculate next batch size based on memory pressure
    pub fn next_batch_size(&self) -> usize;

    /// Block until memory available for next batch
    pub async fn acquire_memory(&self, bytes: usize);

    /// Release memory after batch completion
    pub fn release_memory(&self, bytes: usize);
}
```

### Zero-Downtime Online Migration Strategy

```rust
/// Online migration coordinator with shadow writes
pub struct OnlineMigrationCoordinator {
    source: Arc<dyn MemoryBackend>,
    target: Arc<dyn DualMemoryBackend>,
    shadow_writer: Arc<ShadowWriter>,
    batch_migrator: Arc<BatchMigrator>,
}

impl OnlineMigrationCoordinator {
    /// Phase 1: Start shadow writing to target
    pub async fn start_shadow_writes(&self) -> Result<(), MigrationError>;

    /// Phase 2: Batch migrate historical data in background
    pub async fn migrate_historical_data(&self, rate_limit: usize) -> Result<MigrationStats, MigrationError>;

    /// Phase 3: Atomic cutover when caught up
    pub async fn atomic_cutover(&self) -> Result<(), MigrationError>;
}

/// Shadow writer that duplicates writes to both backends
pub struct ShadowWriter {
    source: Arc<dyn MemoryBackend>,
    target: Arc<dyn DualMemoryBackend>,
    conversion_fn: Arc<dyn Fn(Memory) -> DualMemoryNode + Send + Sync>,
}

impl ShadowWriter {
    /// Write to source, then asynchronously to target
    pub async fn write(&self, memory: Memory) -> Result<(), MigrationError>;
}
```

### Rollback Verification

```rust
/// Cryptographic verification for rollback safety
pub struct RollbackVerifier {
    checkpoint_dir: PathBuf,
}

impl RollbackVerifier {
    /// Compute rolling SHA256 hash of node stream
    pub fn compute_hash(&self, nodes: impl Iterator<Item = &Memory>) -> [u8; 32];

    /// Verify checkpoint hash matches current source state
    pub fn verify_checkpoint(&self, checkpoint: &MigrationCheckpoint, backend: &MemoryBackend) -> Result<bool, MigrationError>;

    /// Create rollback snapshot before migration
    pub async fn create_rollback_snapshot(&self, backend: &MemoryBackend) -> Result<RollbackSnapshot, MigrationError>;

    /// Restore from rollback snapshot
    pub async fn restore_snapshot(&self, snapshot: &RollbackSnapshot, backend: &mut MemoryBackend) -> Result<(), MigrationError>;
}

#[derive(Serialize, Deserialize)]
pub struct RollbackSnapshot {
    pub node_count: u64,
    pub hash: [u8; 32],
    pub tier_checksums: TierChecksums,
    pub wal_sequence: u64,
    pub snapshot_time: SystemTime,
}

#[derive(Serialize, Deserialize)]
pub struct TierChecksums {
    pub hot_tier_hash: [u8; 32],
    pub warm_tier_hash: [u8; 32],
    pub cold_tier_hash: [u8; 32],
}
```

### Memory Budget Management

For large-scale migrations (millions of nodes), enforce strict memory budgets:

```rust
pub struct MigrationConfig {
    /// Maximum memory for in-flight batches (default: 512MB)
    pub memory_budget_mb: usize,

    /// Batch size for migration (default: 1000 nodes)
    pub batch_size: usize,

    /// Checkpoint interval (default: every 10,000 nodes)
    pub checkpoint_interval: usize,

    /// Rate limit in nodes/second (default: unlimited)
    pub rate_limit: Option<usize>,

    /// Backpressure threshold (0.0-1.0, default: 0.8)
    pub backpressure_threshold: f32,

    /// Enable parallel batch processing
    pub parallel_batches: bool,

    /// Number of parallel workers (default: num_cpus / 2)
    pub num_workers: usize,
}

impl Default for MigrationConfig {
    fn default() -> Self {
        Self {
            memory_budget_mb: 512,
            batch_size: 1000,
            checkpoint_interval: 10_000,
            rate_limit: None,
            backpressure_threshold: 0.8,
            parallel_batches: true,
            num_workers: num_cpus::get() / 2,
        }
    }
}

/// Memory estimator for batch sizing
pub struct BatchMemoryEstimator;

impl BatchMemoryEstimator {
    /// Estimate memory footprint of migrating N nodes
    pub fn estimate_batch_memory(node_count: usize) -> usize {
        // DualMemoryNode: ~1KB base + 768 * 4 bytes (embedding) + overhead
        let node_size = 1024 + (768 * 4) + 256; // ~4KB per node
        node_count * node_size
    }

    /// Calculate max batch size for memory budget
    pub fn max_batch_size(memory_budget_bytes: usize) -> usize {
        memory_budget_bytes / Self::estimate_batch_memory(1)
    }
}
```

### Rate Limiting and Backpressure

```rust
/// Token bucket rate limiter for migration throughput control
pub struct MigrationRateLimiter {
    tokens: AtomicU64,
    capacity: u64,
    refill_rate: u64, // tokens per second
    last_refill: Mutex<Instant>,
}

impl MigrationRateLimiter {
    pub fn new(rate_limit: usize) -> Self;

    /// Acquire tokens for batch, blocking if unavailable
    pub async fn acquire(&self, count: usize);

    /// Refill token bucket based on elapsed time
    fn refill(&self);
}

/// Backpressure coordinator monitors system resources
pub struct BackpressureCoordinator {
    memory_threshold: f32,
    current_memory: Arc<AtomicUsize>,
    max_memory: usize,
}

impl BackpressureCoordinator {
    /// Check if system is under pressure
    pub fn is_backpressure(&self) -> bool;

    /// Wait for pressure to subside
    pub async fn wait_for_capacity(&self);
}
```

### CLI Interface

```rust
// In engram-cli/src/cli/migrate_dual.rs

/// Migrate memory space to dual memory format
pub async fn migrate_to_dual_memory(
    space_id: &str,
    mode: MigrationMode, // online or offline
    config: MigrationConfig,
    dry_run: bool,
) -> Result<(), anyhow::Error> {
    // 1. Load space persistence handle
    // 2. Create migrator with config
    // 3. Check for existing checkpoint
    // 4. Run migration with progress bar
    // 5. Verify completion
    // 6. Create rollback snapshot
}

pub enum MigrationMode {
    Online,  // Zero-downtime with shadow writes
    Offline, // Bulk migration with downtime
}
```

### Compatibility Layer

```rust
/// Transparently handle both Memory and DualMemoryNode formats
pub struct DualMemoryCompatibilityAdapter {
    backend: Arc<dyn DualMemoryBackend>,
    legacy_mode: AtomicBool,
    conversion_cache: DashMap<NodeId, DualMemoryNode>,
}

impl MemoryBackend for DualMemoryCompatibilityAdapter {
    fn add_memory(&self, memory: Memory) -> Result<(), BackendError> {
        // Convert to DualMemoryNode with Episode type default
        let dual_node = DualMemoryNode::from_memory(memory, MemoryNodeType::Episode);
        self.backend.add_node_typed(dual_node)
    }

    fn get_memory(&self, id: &NodeId) -> Result<Option<Memory>, BackendError> {
        // Retrieve DualMemoryNode and downconvert to Memory for compatibility
        self.backend.get_node(id)
            .map(|opt| opt.map(|dual| dual.to_memory()))
    }
}
```

## Operational Runbook

### Pre-Migration Checklist

1. Create full backup using `engram backup full --space {space_id}`
2. Verify backup integrity with `engram backup verify`
3. Estimate migration time: `engram migrate-dual estimate --space {space_id}`
4. Check available disk space (2x current space size recommended)
5. Verify memory budget settings match available RAM
6. Schedule maintenance window for offline migration (optional)

### Migration Execution

```bash
# Dry-run validation
engram migrate-dual --space production --dry-run

# Online migration (zero downtime)
engram migrate-dual --space production --mode online \
  --batch-size 1000 \
  --checkpoint-interval 10000 \
  --memory-budget-mb 512 \
  --rate-limit 5000

# Offline migration (faster, requires downtime)
engram migrate-dual --space production --mode offline \
  --parallel-workers 8

# Resume from checkpoint
engram migrate-dual --space production --resume
```

### Monitoring During Migration

Monitor these metrics via Prometheus:

- `engram_migration_progress_ratio` (0.0-1.0)
- `engram_migration_nodes_migrated_total`
- `engram_migration_batch_duration_seconds`
- `engram_migration_memory_usage_bytes`
- `engram_migration_checkpoint_writes_total`

CLI progress output:

```
[migration] Starting online migration for space: production
[migration] Total nodes: 1,247,832
[migration] Batch size: 1,000 | Memory budget: 512 MB | Rate limit: 5,000 nodes/sec
[migration]
[migration] Progress: [████████████████----] 80.3% (1,001,932 / 1,247,832)
[migration] Rate: 4,832 nodes/sec | ETA: 51 seconds
[migration] Memory: 387 MB / 512 MB (75.6%)
[migration] Last checkpoint: 2024-01-15T10:45:23Z (1,000,000 nodes)
```

### Post-Migration Validation

```bash
# Verify migration integrity
engram migrate-dual verify --space production

# Compare node counts
engram space stats --space production

# Run acceptance tests
engram validate --space production --test-suite migration
```

### Rollback Procedure

```bash
# Stop writes
engram space stop --space production

# Rollback to pre-migration state
engram migrate-dual rollback --space production --checkpoint latest

# Restore from backup if rollback fails
engram restore --space production --backup /backups/production_20240115.tar.zst

# Resume service
engram space start --space production
```

## Implementation Notes

- Default all migrated nodes to Episode type initially
- Track migration progress in `{space_dir}/migration/checkpoint.bin` using bincode serialization
- Use SHA256 rolling hashes for incremental verification (crc32c from chosen_libraries.md)
- Implement dry-run mode that validates without writing to target
- Online migration uses shadow writes to maintain consistency
- Offline migration uses rayon for parallel batch processing
- Memory budget enforced at batch granularity
- Checkpoint every 10,000 nodes by default (configurable)
- Rate limiting uses token bucket algorithm
- Backpressure monitoring checks memory_stats (from chosen_libraries.md)

## Testing Approach

1. Unit tests for individual node conversion (Memory -> DualMemoryNode)
2. Checkpoint serialization round-trip tests with bincode
3. Hash verification tests with known datasets
4. Integration tests with sample datasets (100, 1K, 10K, 100K nodes)
5. Rollback testing with integrity verification
6. Performance tests for large-scale migration (1M+ nodes)
7. Backpressure simulation under memory constraints
8. Checkpoint recovery tests with simulated crashes
9. Online migration with concurrent read/write workloads
10. Property tests for migration invariants (no data loss, count preservation)

## Acceptance Criteria

- [ ] CLI command `engram migrate-dual` with online/offline modes
- [ ] Progress tracking with live updates (nodes/sec, ETA, memory usage)
- [ ] Checkpoint persistence using bincode format
- [ ] Resume from checkpoint after interruption
- [ ] Rollback restores original state with hash verification
- [ ] Migration speed >10K nodes/second for offline mode
- [ ] Online migration maintains read availability (no downtime)
- [ ] Zero data loss verified via cryptographic hashes
- [ ] Memory budget enforcement prevents OOM
- [ ] Rate limiting respects configured limits
- [ ] Dry-run mode validates without side effects
- [ ] Operational runbook covers common scenarios
- [ ] Prometheus metrics exposed for monitoring
- [ ] Integration tests pass for 1M node dataset

## Dependencies

- Task 001 (Dual Memory Types) - REQUIRED
- Task 002 (Graph Storage Adaptation) - REQUIRED
- Existing backup/restore infrastructure in engram-cli/src/cli/backup.rs
- Existing WAL and checkpoint patterns in engram-core/src/storage/persistence.rs
- bincode for checkpoint serialization (approved in chosen_libraries.md)
- SHA256 from std or ring crate for hash verification

## Estimated Time

4 days (revised from 2 days due to production-grade requirements)