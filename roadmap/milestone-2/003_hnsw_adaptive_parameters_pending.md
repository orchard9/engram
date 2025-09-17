# Task 003: HNSW Adaptive Parameter Tuning

## Status: Pending
## Priority: P1 - Performance Critical  
## Estimated Effort: 1 day
## Dependencies: None (enhances existing HNSW)

## Objective
Add simple adaptive parameter tuning to existing `CognitiveHnswIndex` to automatically adjust parameters based on recall performance and query patterns.

## Current State Analysis
- **Existing**: `CognitiveHnswIndex` with `CognitiveHnswParams` in `index/mod.rs`
- **Existing**: Pressure adaptation in `PressureAdapter`
- **Missing**: Performance-based parameter adaptation
- **Missing**: Recall measurement and feedback

## Implementation Plan

### Modify Existing Files
- `engram-core/src/index/mod.rs:88-114` - Enhance `CognitiveHnswParams` with adaptation
- `engram-core/src/index/hnsw_search.rs:45-80` - Add recall measurement
- `engram-core/src/index/mod.rs:210-233` - Enhance search with adaptation

### Create New File
- `engram-core/src/index/adaptive_params.rs` - Simple adaptation logic

## Implementation Details

### Enhanced Parameters (engram-core/src/index/mod.rs:88-114)
```rust
pub struct CognitiveHnswParams {
    // ... existing fields ...
    
    // Adaptive tuning state
    pub adaptation_enabled: AtomicBool,
    pub target_recall: f32,           // Target recall@10 (e.g., 0.9)
    pub adaptation_rate: f32,         // Learning rate (e.g., 0.1)
    pub min_samples_for_adaptation: usize, // Min queries before adapting
}

impl CognitiveHnswParams {
    pub fn adapt_ef_search(&self, measured_recall: f32) {
        if !self.adaptation_enabled.load(Ordering::Relaxed) {
            return;
        }
        
        let current_ef = self.ef_search.load(Ordering::Relaxed);
        let error = self.target_recall - measured_recall;
        
        // Simple proportional controller
        let adjustment = (error * self.adaptation_rate * current_ef as f32) as i32;
        let new_ef = (current_ef as i32 + adjustment).clamp(16, 512) as usize;
        
        self.ef_search.store(new_ef, Ordering::Relaxed);
    }
}
```

### Adaptive Logic (engram-core/src/index/adaptive_params.rs)
```rust
use std::collections::VecDeque;
use std::sync::atomic::{AtomicU64, Ordering};

pub struct RecallTracker {
    recent_recalls: parking_lot::Mutex<VecDeque<f32>>,
    query_count: AtomicU64,
    window_size: usize,
}

impl RecallTracker {
    pub fn new(window_size: usize) -> Self {
        Self {
            recent_recalls: parking_lot::Mutex::new(VecDeque::with_capacity(window_size)),
            query_count: AtomicU64::new(0),
            window_size,
        }
    }
    
    pub fn record_recall(&self, recall: f32) {
        let mut recalls = self.recent_recalls.lock();
        
        if recalls.len() >= self.window_size {
            recalls.pop_front();
        }
        recalls.push_back(recall);
        
        self.query_count.fetch_add(1, Ordering::Relaxed);
    }
    
    pub fn average_recall(&self) -> Option<f32> {
        let recalls = self.recent_recalls.lock();
        
        if recalls.is_empty() {
            None
        } else {
            Some(recalls.iter().sum::<f32>() / recalls.len() as f32)
        }
    }
    
    pub fn should_adapt(&self, min_samples: usize) -> bool {
        self.query_count.load(Ordering::Relaxed) as usize >= min_samples
    }
}
```

### Enhanced Search with Recall Measurement (engram-core/src/index/mod.rs:210-233)
```rust
impl CognitiveHnswIndex {
    // Add field to struct around line 85:
    recall_tracker: RecallTracker,
    
    pub fn search_with_adaptation(
        &self,
        query: &[f32; 768],
        k: usize,
        threshold: Confidence,
        ground_truth: Option<&[String]>, // For recall measurement
    ) -> Vec<(String, Confidence)> {
        let results = self.search_with_confidence(query, k, threshold);
        
        // Measure recall if ground truth provided
        if let Some(truth) = ground_truth {
            let recall = self.compute_recall(&results, truth, k);
            self.recall_tracker.record_recall(recall);
            
            // Adapt parameters if enough samples
            if self.recall_tracker.should_adapt(self.params.min_samples_for_adaptation) {
                if let Some(avg_recall) = self.recall_tracker.average_recall() {
                    self.params.adapt_ef_search(avg_recall);
                }
            }
        }
        
        results
    }
    
    fn compute_recall(&self, results: &[(String, Confidence)], ground_truth: &[String], k: usize) -> f32 {
        let result_ids: std::collections::HashSet<_> = results.iter()
            .take(k)
            .map(|(id, _)| id)
            .collect();
            
        let truth_set: std::collections::HashSet<_> = ground_truth.iter().take(k).collect();
        
        let intersection_size = result_ids.intersection(&truth_set).count();
        intersection_size as f32 / k.min(ground_truth.len()) as f32
    }
}
```

### Enable Adaptation (engram-core/src/index/mod.rs:155-179)
```rust
impl CognitiveHnswIndex {
    pub fn new() -> Self {
        let params = Arc::new(CognitiveHnswParams {
            // ... existing fields ...
            adaptation_enabled: AtomicBool::new(false),
            target_recall: 0.9,
            adaptation_rate: 0.1,
            min_samples_for_adaptation: 20,
        });

        Self {
            // ... existing fields ...
            recall_tracker: RecallTracker::new(50), // 50-query window
            // ...
        }
    }
    
    pub fn enable_adaptation(&self, target_recall: f32) {
        self.params.target_recall = target_recall;
        self.params.adaptation_enabled.store(true, Ordering::Relaxed);
    }
}
```

## Acceptance Criteria
- [ ] Parameters adapt to achieve target recall within 50 queries
- [ ] Recall measurement accurate to within 5% of ground truth
- [ ] Adaptation does not degrade performance >10%
- [ ] Parameter changes are bounded and safe
- [ ] Integration with existing pressure adaptation

## Performance Targets
- Recall measurement overhead: <5% of query time
- Parameter adaptation: <1μs per query
- Achieve 90% recall@10 within 100 queries
- Maintain recall stability (±2%) after adaptation

## Risk Mitigation
- Conservative parameter bounds to prevent instability
- Adaptation can be disabled if performance degrades
- Exponential moving average for stable recall estimates
- Separate validation queries from adaptation feedback