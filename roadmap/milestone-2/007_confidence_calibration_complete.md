# Task 007: Confidence Calibration for Storage Operations

## Status: Complete ✅
## Priority: P2 - Quality Enhancement
## Estimated Effort: 0.5 days
## Dependencies: Tasks 001-006 (storage system)

## Objective
Add confidence calibration for storage retrieval operations to ensure uncertainty is properly tracked across storage tiers.

## Current Implementation Status
- ✅ Tier-specific calibrator implemented with temporal decay support (`engram-core/src/storage/confidence.rs:1-145`).
- ✅ Hot, warm, and cold tier recall paths adjust confidence using the calibrator (`engram-core/src/storage/hot_tier.rs:100-121`, `engram-core/src/storage/warm_tier.rs:204-218`, `engram-core/src/storage/cold_tier.rs:660-671`).
- ✅ Helper tests cover calibrator defaults (`engram-core/src/storage/confidence.rs:196-211`).
- ⚠️ No metrics or logging track calibration adjustments; consider exposing stats in future.

## Remaining Work
None required for milestone completion. Optional follow-up: emit calibration stats via monitoring hooks if needed.

## Current State Analysis
- **Existing**: `Confidence` type throughout system
- **Existing**: Confidence tracking in episodes and memories
- **Missing**: Confidence adjustment for storage tier retrieval
- **Missing**: Calibration for storage-related uncertainty

## Implementation Plan

### Add Storage Confidence Adjustment (engram-core/src/storage/confidence.rs)
```rust
use crate::Confidence;
use std::time::Duration;

pub struct StorageConfidenceCalibrator {
    tier_confidence_factors: TierConfidenceFactors,
}

struct TierConfidenceFactors {
    hot_factor: f32,    // 1.0 - no degradation
    warm_factor: f32,   // 0.95 - slight degradation from compression
    cold_factor: f32,   // 0.9 - degradation from quantization
}

impl StorageConfidenceCalibrator {
    pub fn adjust_for_storage_tier(
        &self,
        original_confidence: Confidence,
        tier: StorageTier,
        time_in_storage: Duration,
    ) -> Confidence {
        let tier_factor = match tier {
            StorageTier::Hot => self.tier_confidence_factors.hot_factor,
            StorageTier::Warm => self.tier_confidence_factors.warm_factor,
            StorageTier::Cold => self.tier_confidence_factors.cold_factor,
        };
        
        // Apply temporal decay (very gradual)
        let days_stored = time_in_storage.as_secs() as f32 / 86400.0;
        let temporal_factor = (-days_stored / 3650.0).exp(); // 10-year half-life
        
        let adjusted = original_confidence.raw() * tier_factor * temporal_factor;
        Confidence::exact(adjusted.max(0.01)) // Minimum confidence
    }
}
```

### Integration with Tier Implementations
Update each tier's `recall` method to include confidence adjustment:

```rust
// In HotTier::recall
async fn recall(&self, cue: &Cue) -> Result<Vec<(Episode, Confidence)>, Self::Error> {
    let mut results = self.raw_recall(cue).await?;
    
    // Adjust confidence for hot tier characteristics
    for (episode, confidence) in &mut results {
        let storage_time = self.get_storage_duration(&episode.id);
        *confidence = self.calibrator.adjust_for_storage_tier(
            *confidence,
            StorageTier::Hot,
            storage_time,
        );
    }
    
    Ok(results)
}
```

## Acceptance Criteria
- [ ] All storage tiers return calibrated confidence scores
- [ ] Confidence decreases appropriately with storage tier
- [ ] Temporal decay is applied correctly
- [ ] No confidence values outside [0, 1] range

## Performance Targets
- Confidence calibration: <1μs per result
- No measurable impact on retrieval latency

## Risk Mitigation
- Conservative calibration factors
- Validate all confidence values are in [0, 1]
- Make calibration optional/configurable
