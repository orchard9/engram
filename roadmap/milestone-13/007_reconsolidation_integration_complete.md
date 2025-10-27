# Task 007: Reconsolidation Integration with Consolidation Pipeline

**Status:** PENDING
**Priority:** P1
**Estimated Duration:** 2 days
**Dependencies:** Task 006 (Reconsolidation Core), M6 (Consolidation)
**Agent Review Required:** memory-systems-researcher

## Overview

Integrate reconsolidation engine (Task 006) with existing consolidation pipeline (M6). Reconsolidated memories must re-enter the consolidation process without creating conflicts or duplicates.

## Biology Foundation

**Source:** Nader et al. (2000), Lee (2009)

When a consolidated memory is recalled and modified during the reconsolidation window, it becomes labile (unstable) again and must undergo re-consolidation to re-stabilize. This re-uses the same molecular machinery as initial consolidation.

**Key Insight:** Reconsolidation is not a separate process - it's re-entry into the existing consolidation pipeline with modified content.

## Implementation Specifications

### File Structure
```
engram-core/src/cognitive/reconsolidation/
└── consolidation_integration.rs (new)

engram-core/tests/cognitive/
└── reconsolidation_consolidation_tests.rs (new)
```

### Integration Interface

**File:** `/engram-core/src/cognitive/reconsolidation/consolidation_integration.rs`

```rust
use crate::decay::consolidation::ConsolidationEngine;
use crate::cognitive::reconsolidation::ReconsolidationEngine;

/// Bridge between reconsolidation and consolidation systems
pub struct ReconsolidationConsolidationBridge {
    reconsolidation: Arc<ReconsolidationEngine>,
    consolidation: Arc<ConsolidationEngine>,
}

impl ReconsolidationConsolidationBridge {
    /// Process reconsolidated memory back into consolidation pipeline
    pub fn process_reconsolidated_memory(
        &self,
        modified_episode: Episode,
        reconsolidation_result: ReconsolidationResult
    ) -> Result<ConsolidationHandle> {
        // Tag episode as reconsolidated (not initial consolidation)
        let tagged = modified_episode.with_metadata(
            "reconsolidation_event",
            reconsolidation_result.plasticity_factor
        );

        // Re-enter consolidation pipeline
        // Uses same consolidation stages as initial learning
        self.consolidation.consolidate(tagged)
    }
}
```

## Integration Points

**M6 (Consolidation):** Re-use existing pipeline
- File: `engram-core/src/decay/consolidation.rs`
- Mechanism: Reconsolidated episodes marked with metadata, otherwise identical flow

**Task 006 (Reconsolidation):** Source of modified episodes
- File: `engram-core/src/cognitive/reconsolidation/mod.rs`
- Output: `ReconsolidationResult` with modified episode

**Metrics:** Distinguish initial vs reconsolidation events
- Separate counters: `consolidation_initial` vs `consolidation_reconsolidated`

## Testing Strategy

### Test 1: Reconsolidated Memories Re-consolidate
```rust
#[test]
fn test_reconsolidated_memory_re_enters_pipeline() {
    // 1. Store episode → consolidate
    // 2. Recall episode → modify during window
    // 3. Verify modified episode re-enters consolidation
    // 4. Check consolidation state after re-consolidation
}
```

### Test 2: No Conflicts Between Systems
```rust
#[test]
fn test_no_conflicts_between_consolidation_and_reconsolidation() {
    // Run consolidation and reconsolidation concurrently
    // Verify no deadlocks, race conditions, or duplicates
}
```

### Test 3: Metrics Distinguish Events
```rust
#[test]
fn test_metrics_distinguish_initial_vs_reconsolidation() {
    #[cfg(feature = "monitoring")]
    {
        let initial_count = metrics.consolidation_initial();
        let recon_count = metrics.consolidation_reconsolidated();
        assert_ne!(initial_count, recon_count);
    }
}
```

## Acceptance Criteria

### Must Have
- [ ] Reconsolidated memories re-enter M6 consolidation pipeline
- [ ] No conflicts between consolidation and reconsolidation
- [ ] Metrics distinguish initial consolidation from reconsolidation
- [ ] Episode metadata tracks reconsolidation events
- [ ] All tests pass
- [ ] `make quality` passes

### Should Have
- [ ] Reconsolidation performance overhead <10% vs initial consolidation
- [ ] Integration documented in API reference

### Nice to Have
- [ ] Visualization showing reconsolidation → consolidation flow
- [ ] Reconsolidation count tracked per episode

## Implementation Checklist

- [ ] Create `consolidation_integration.rs`
- [ ] Implement `ReconsolidationConsolidationBridge`
- [ ] Add episode metadata tagging for reconsolidation
- [ ] Update consolidation metrics to distinguish initial vs reconsolidation
- [ ] Write integration test for re-entry into pipeline
- [ ] Write concurrency test for no conflicts
- [ ] Write metrics test for event distinction
- [ ] Verify integration with M6 works correctly
- [ ] Run `make quality`

## Risks and Mitigations

**Risk:** Reconsolidation creates duplicate episodes
- **Mitigation:** Episode ID remains unchanged through reconsolidation
- **Mitigation:** Integration test validates no duplicates

**Risk:** Concurrency issues between consolidation and reconsolidation
- **Mitigation:** Both use same locking mechanisms from M6
- **Mitigation:** Stress test with concurrent operations

## References

1. Nader, K., et al. (2000). Fear memories require protein synthesis for reconsolidation. *Nature*.
2. Lee, J. L. (2009). Reconsolidation: maintaining memory relevance. *Trends in Neurosciences*.
