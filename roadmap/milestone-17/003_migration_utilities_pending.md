# Task 003: Migration Utilities

## Objective
Build utilities to migrate existing Memory nodes to DualMemoryNode format, including bulk migration tools and compatibility layers.

## Background
We need a smooth migration path from the current single-type system to dual memory without disrupting existing deployments.

## Requirements
1. Create migration tool to convert Memory to DualMemoryNode
2. Implement compatibility layer for gradual migration
3. Add migration progress tracking and resumption
4. Provide rollback capabilities
5. Support both online and offline migration modes

## Technical Specification

### Files to Create
- `engram-core/src/migration/dual_memory.rs` - Migration logic
- `engram-cli/src/commands/migrate.rs` - CLI migration command
- `engram-core/src/compat/dual_memory.rs` - Compatibility layer

### Migration Strategy
```rust
pub struct DualMemoryMigrator {
    source: Arc<dyn MemoryBackend>,
    target: Arc<dyn DualMemoryBackend>,
    progress: AtomicUsize,
    checkpoint: Option<NodeId>,
}

impl DualMemoryMigrator {
    pub async fn migrate_online(&self, batch_size: usize) -> Result<MigrationStats>;
    
    pub async fn migrate_offline(&self) -> Result<MigrationStats>;
    
    pub fn resume_from_checkpoint(&mut self, checkpoint: NodeId);
}
```

### Compatibility Layer
```rust
// Transparently handle both old and new formats
pub struct CompatibilityAdapter {
    backend: Arc<dyn DualMemoryBackend>,
    legacy_mode: AtomicBool,
}

impl MemoryBackend for CompatibilityAdapter {
    fn add_memory(&self, memory: Memory) -> Result<()> {
        let dual_node = self.convert_to_dual(memory);
        self.backend.add_node_typed(dual_node)
    }
}
```

## Implementation Notes
- Default all migrated nodes to Episode type initially
- Track migration progress in persistent storage
- Use checksums to verify migration correctness
- Implement dry-run mode for validation

## Testing Approach
1. Unit tests for individual node conversion
2. Integration tests with sample datasets
3. Rollback testing with data verification
4. Performance tests for large-scale migration

## Acceptance Criteria
- [ ] CLI command for migration with progress bar
- [ ] Online migration without service interruption
- [ ] Checkpoint/resume functionality works
- [ ] Rollback restores original state
- [ ] Migration speed >10K nodes/second
- [ ] Zero data loss during migration

## Dependencies
- Task 001 (Dual Memory Types)
- Task 002 (Graph Storage Adaptation)

## Estimated Time
2 days