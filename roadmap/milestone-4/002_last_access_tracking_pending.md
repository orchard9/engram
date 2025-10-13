# Task 002: Last Access Tracking

## Objective
Add `last_access` tracking to Memory and Episode types to enable temporal decay based on memory usage patterns.

## Priority
P0 (blocking - required for decay integration)

## Effort Estimate
1 day

## Dependencies
None

## Technical Approach

### Files to Modify
- `engram-core/src/memory.rs` - Add `last_access` field to Memory and Episode
- `engram-core/src/store.rs` - Update `last_access` on retrieval
- `engram-core/src/activation/recall.rs` - Update `last_access` during recall

### Design

**Memory/Episode Type Changes**:
```rust
// engram-core/src/memory.rs
pub struct Memory {
    // ... existing fields ...
    pub last_access: DateTime<Utc>,
    pub access_count: u64,  // Track retrieval frequency
}

pub struct Episode {
    // ... existing fields ...
    pub last_access: DateTime<Utc>,
    pub access_count: u64,
}
```

**Automatic Update During Recall**:
```rust
// engram-core/src/activation/recall.rs
impl CognitiveRecall {
    fn rank_results(...) -> Vec<RankedMemory> {
        // ... existing ranking code ...

        // Update last_access for retrieved episodes
        for result in &ranked {
            if let Some(episode) = store.get_episode_mut(&result.episode.id) {
                episode.last_access = Utc::now();
                episode.access_count += 1;
            }
        }

        ranked
    }
}
```

**Lazy Update Pattern**:
- Don't write `last_access` to storage immediately (would cause write amplification)
- Update in-memory only during recall
- Persist during next consolidation/checkpoint

## Acceptance Criteria

- [ ] Memory and Episode have `last_access: DateTime<Utc>` field
- [ ] Memory and Episode have `access_count: u64` field
- [ ] Default values set to creation time
- [ ] `last_access` updated automatically during recall operations
- [ ] `access_count` incremented on each retrieval
- [ ] No write amplification (lazy persistence)
- [ ] Backward compatibility preserved (migration for existing memories)
- [ ] Unit tests for access tracking

## Testing Approach

**Unit Tests**:
```rust
#[test]
fn test_last_access_initialized_to_creation_time() {
    let episode = Episode::new("content");
    assert_eq!(episode.last_access, episode.when);
}

#[test]
fn test_last_access_updated_on_recall() {
    let store = MemoryStore::new_temp();
    let recall = CognitiveRecall::new(...);

    let episode_id = store.insert_episode(Episode::new("test"));
    let initial_access = store.get_episode(&episode_id).unwrap().last_access;

    sleep(Duration::from_millis(100));

    let results = recall.recall(&cue, &store).unwrap();
    let updated_access = store.get_episode(&episode_id).unwrap().last_access;

    assert!(updated_access > initial_access);
}

#[test]
fn test_access_count_incremented() {
    // Similar test verifying access_count increases
}
```

## Risk Mitigation

**Risk**: Write amplification from frequent `last_access` updates
**Mitigation**: Lazy persistence - update in-memory, persist during consolidation. Batch updates every N minutes.

**Risk**: Backward compatibility with existing data
**Mitigation**: Migration code to set `last_access` to `when` (creation time) for old episodes. Default values in deserialization.

**Risk**: Thread safety with concurrent access
**Mitigation**: Use atomic operations for `access_count`. `last_access` updates don't need strong consistency (approximate is fine).

## Notes

This task establishes the foundation for temporal decay by tracking when memories were last accessed. The `access_count` field supports spaced repetition algorithms that adjust decay based on retrieval frequency.

**Design Principle**: Lazy updates to avoid write amplification. Approximate timestamps are acceptable - we don't need microsecond precision for decay calculation.
