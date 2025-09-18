# Task 003: HNSW Cognitive Dynamics Adaptation

## Status: Pending
## Priority: P1 - Performance Critical  
## Estimated Effort: 1.5 days
## Dependencies: None (enhances existing HNSW)

## Prerequisites

### Lock-Free Data Structure Requirements
- **ArrayQueue Integration**: Replace custom `CircularBuffer` with `crossbeam::queue::ArrayQueue<f32>` for activation history (proven lock-free safety)
- **Atomic Time Handling**: Convert `Duration` fields to `AtomicU64` nanoseconds to avoid shared mutable state in `CognitiveHnswParams`
- **ABI Compatibility**: Add new fields at end of existing structs to prevent breaking changes

### Missing Graph Operations
- **Neighbor-to-Episode Conversion**: Implement `find_connected_memories()` using existing `get_neighbors()` + memory lookup
- **Activation Merging**: Implement energy decay and confidence combination for spreading activation
- **Temporal Access Tracking**: Utilize existing `last_access_epoch` in `HnswNode` for locality detection

### Performance Safety Guards
- **Overhead Measurement**: Benchmark activation recording <0.5μs before enabling adaptation
- **Feature Gating**: Implement `dynamics_enabled` flag for graceful enable/disable
- **Circuit Breaker**: Automatic disable when overconfidence >50% to prevent graph degradation

## Objective
Enhance existing `CognitiveHnswIndex` with cognitive dynamics-based adaptation that responds to activation patterns, confidence distributions, and temporal access patterns - implementing biological principles rather than ML accuracy metrics.

## Current State Analysis
- **Existing**: `CognitiveHnswIndex` with pressure-based `PressureAdapter` in `index/mod.rs:131-141`
- **Existing**: Activation spreading with cognitive parameters (`activation_decay_rate`, `temporal_boost_factor`)
- **Existing**: Confidence-weighted operations and circuit breaker patterns in error recovery
- **Missing**: Activation pattern analysis for parameter adaptation
- **Missing**: Temporal locality tracking for dynamic tuning
- **Missing**: Confidence distribution monitoring for overconfidence prevention

## Implementation Plan

### Modify Existing Files
- `engram-core/src/index/mod.rs:101-114` - Enhance `CognitiveHnswParams` with cognitive adaptation state
- `engram-core/src/index/mod.rs:131-141` - Extend `PressureAdapter` with activation dynamics
- `engram-core/src/index/mod.rs:234-268` - Enhance spreading activation with adaptation feedback

### Create New File
- `engram-core/src/index/cognitive_dynamics.rs` - Activation pattern analysis and temporal adaptation

## Implementation Details

### Enhanced Cognitive Parameters (engram-core/src/index/mod.rs:101-114)
```rust
pub struct CognitiveHnswParams {
    // ... existing HNSW fields (m_max, ef_search, etc.) ...
    
    // NEW FIELDS: Add at end for ABI compatibility
    pub dynamics_enabled: AtomicBool,
    pub activation_sensitivity: f32,         // Non-atomic: set at initialization
    pub confidence_stability_target: f32,   // Non-atomic: set at initialization  
    pub temporal_locality_window_ns: AtomicU64, // Nanoseconds for thread safety
    pub overconfidence_threshold: f32,      // Non-atomic: set at initialization
    
    // Lock-free adaptation state
    pub adaptation_cycle: AtomicU64,
    pub last_adaptation_time: AtomicU64,
}

impl CognitiveHnswParams {
    /// Adapt parameters based on activation energy distribution
    pub fn adapt_to_activation_patterns(&self, dynamics: &ActivationDynamics) {
        if !self.dynamics_enabled.load(Ordering::Relaxed) {
            return;
        }
        
        let current_ef = self.ef_search.load(Ordering::Relaxed);
        let activation_density = dynamics.compute_activation_density();
        
        // Biological principle: Sparse activation = increase search width
        // Dense activation = interference, reduce search width
        let target_ef = if activation_density < 0.3 {
            // Too sparse - increase exploration
            (current_ef as f32 * 1.2).min(512.0) as usize
        } else if activation_density > 0.7 {
            // Too dense - reduce interference  
            (current_ef as f32 * 0.8).max(16.0) as usize
        } else {
            current_ef // Optimal range
        };
        
        self.ef_search.store(target_ef, Ordering::Relaxed);
    }
}
```

### Cognitive Dynamics Analysis (engram-core/src/index/cognitive_dynamics.rs)
```rust
use std::time::Instant;
use std::sync::atomic::{AtomicU64, Ordering};
use crossbeam::queue::ArrayQueue;
use parking_lot::Mutex;

/// Tracks activation patterns and confidence distributions for cognitive adaptation
pub struct ActivationDynamics {
    /// Lock-free sliding window of recent activation energies (64 samples)
    activation_history: ArrayQueue<f32>,
    
    /// Confidence variance tracking (requires occasional locking for variance calculation)
    confidence_variance: Mutex<RunningVariance>,
    
    /// Lock-free temporal access ring buffer (32 slots for temporal locality)
    access_timestamps: [AtomicU64; 32],
    access_index: AtomicU64,
    
    /// Lock-free overconfidence detection counters
    overconfident_connections: AtomicU64,
    total_connections: AtomicU64,
}

impl ActivationDynamics {
    pub fn new() -> Self {
        Self {
            activation_history: ArrayQueue::new(64), // Lock-free 64-sample window
            confidence_variance: Mutex::new(RunningVariance::new()),
            access_timestamps: std::array::from_fn(|_| AtomicU64::new(0)),
            access_index: AtomicU64::new(0),
            overconfident_connections: AtomicU64::new(0),
            total_connections: AtomicU64::new(0),
        }
    }
    
    /// Record activation energy from spreading activation process (CRITICAL PATH: <0.5μs)
    pub fn record_activation(&self, energy: f32, confidence: Confidence) {
        // Lock-free activation history update (drop oldest if full)
        if self.activation_history.push(energy).is_err() {
            // Queue full - pop and retry (maintains window size)
            let _ = self.activation_history.pop();
            let _ = self.activation_history.push(energy);
        }
        
        // Lock-free temporal access pattern (single atomic operation)
        let now = Instant::now().elapsed().as_nanos() as u64;
        let index = self.access_index.fetch_add(1, Ordering::Relaxed) % 32;
        self.access_timestamps[index as usize].store(now, Ordering::Relaxed);
        
        // Confidence variance update (less frequent, acceptable lock)
        if let Ok(mut variance) = self.confidence_variance.try_lock() {
            variance.update(confidence.raw());
        }
        // If lock contention, skip confidence update (non-critical)
    }
    
    /// Compute current activation density for parameter adaptation
    pub fn compute_activation_density(&self) -> f32 {
        // ArrayQueue doesn't allow iteration - sample current length and estimate
        let current_len = self.activation_history.len();
        if current_len == 0 {
            return 0.0;
        }
        
        // Estimate density based on recent activity (proxy: queue utilization)
        // Dense activation = queue stays full (len approaches capacity)
        // Sparse activation = queue rarely fills
        let capacity = self.activation_history.capacity();
        let utilization = current_len as f32 / capacity as f32;
        
        // Convert utilization to activation density (biological interpretation)
        utilization.clamp(0.0, 1.0)
    }
    
    /// Detect overconfidence patterns that could harm graph quality
    pub fn overconfidence_ratio(&self) -> f32 {
        let overconfident = self.overconfident_connections.load(Ordering::Relaxed);
        let total = self.total_connections.load(Ordering::Relaxed);
        if total == 0 { 0.0 } else { overconfident as f32 / total as f32 }
    }
    
    /// Measure temporal locality for cache-aware parameter tuning
    pub fn temporal_locality_factor(&self) -> f32 {
        let now = Instant::now().elapsed().as_nanos() as u64;
        let recent_threshold = Duration::from_millis(100).as_nanos() as u64;
        
        let recent_accesses = self.access_timestamps
            .iter()
            .map(|timestamp| timestamp.load(Ordering::Relaxed))
            .filter(|&timestamp| timestamp > 0 && (now - timestamp) < recent_threshold)
            .count();
            
        recent_accesses as f32 / 32.0 // Normalize to [0,1]
    }
}

/// Running variance calculation for confidence distribution (Welford's algorithm)
struct RunningVariance {
    count: u64,
    mean: f64,
    m2: f64, // Sum of squares of differences from mean
}

impl RunningVariance {
    pub fn new() -> Self {
        Self { count: 0, mean: 0.0, m2: 0.0 }
    }
    
    pub fn update(&mut self, value: f32) {
        self.count += 1;
        let delta = value as f64 - self.mean;
        self.mean += delta / self.count as f64;
        let delta2 = value as f64 - self.mean;
        self.m2 += delta * delta2;
    }
    
    pub fn variance(&self) -> f32 {
        if self.count < 2 { 0.0 } else { (self.m2 / (self.count - 1) as f64) as f32 }
    }
}
```

### Enhanced Spreading Activation with Cognitive Feedback (engram-core/src/index/mod.rs:234-268)
```rust
impl CognitiveHnswIndex {
    // Add field to struct around line 87:
    activation_dynamics: ActivationDynamics,
    
    /// Enhanced spreading activation with cognitive adaptation feedback
    pub fn apply_spreading_activation(
        &self,
        mut results: Vec<(crate::Episode, Confidence)>,
        cue: &crate::Cue,
        pressure: f32,
    ) -> Vec<(crate::Episode, Confidence)> {
        // Existing pressure adaptation
        self.pressure_adapter.adapt_params(pressure, &self.params);

        // Cognitive dynamics analysis
        let activation_energy = 1.0 - pressure;
        let temporal_locality = self.activation_dynamics.temporal_locality_factor();
        let overconfidence_ratio = self.activation_dynamics.overconfidence_ratio();
        
        // Record activation pattern for future adaptation
        for (_, confidence) in &results {
            self.activation_dynamics.record_activation(activation_energy, *confidence);
        }
        
        // Adapt parameters based on cognitive dynamics (not ML accuracy)
        if self.should_adapt_dynamics() {
            self.params.adapt_to_activation_patterns(&self.activation_dynamics);
        }
        
        // Apply cognitive principles to spreading activation
        let max_hops = self.compute_cognitive_hops(pressure, temporal_locality, overconfidence_ratio);
        
        // Enhanced activation spreading with confidence weighting
        for hop in 0..max_hops {
            let hop_energy = activation_energy * (0.8_f32).powi(hop as i32); // Decay per hop
            
            if hop_energy < 0.1 {
                break; // Below threshold
            }
            
            // Process each result for potential spreading
            let mut new_activations = Vec::new();
            for (episode, confidence) in &results {
                // Find connected memories via graph traversal
                let connected = self.find_connected_memories(&episode.id, hop_energy);
                new_activations.extend(connected);
            }
            
            // Merge new activations with existing results
            results = self.merge_activations(results, new_activations, hop_energy);
        }

        // Apply temporal boost for recent memories
        self.apply_temporal_boost(&mut results, temporal_locality);
        
        results
    }
    
    fn should_adapt_dynamics(&self) -> bool {
        // Adapt every 100 activation cycles, but not more than once per 10 seconds
        let cycle = self.params.adaptation_cycle.fetch_add(1, Ordering::Relaxed);
        let last_adaptation = self.params.last_adaptation_time.load(Ordering::Relaxed);
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        
        cycle % 100 == 0 && now.saturating_sub(last_adaptation) >= 10
    }
    
    fn compute_cognitive_hops(&self, pressure: f32, temporal_locality: f32, overconfidence: f32) -> usize {
        // Biological principle: High pressure = reduce exploration
        // High temporal locality = increase local search
        // High overconfidence = reduce to prevent bias amplification
        
        let base_hops = if pressure > 0.8 { 1 } else if pressure > 0.5 { 2 } else { 3 };
        
        let locality_adjustment = if temporal_locality > 0.7 { 1 } else { 0 };
        let confidence_adjustment = if overconfidence > 0.3 { -1 } else { 0 };
        
        (base_hops + locality_adjustment + confidence_adjustment).max(1).min(4)
    }
}
```

### Initialize Cognitive Dynamics (engram-core/src/index/mod.rs:154-179)
```rust
impl CognitiveHnswIndex {
    pub fn new() -> Self {
        let params = Arc::new(CognitiveHnswParams {
            // ... existing HNSW fields ...
            
            // NEW FIELDS: Cognitive dynamics configuration (added at end for ABI compatibility)
            dynamics_enabled: AtomicBool::new(false),
            activation_sensitivity: 0.15,                    // Non-atomic: set at init
            confidence_stability_target: 0.2,               // Non-atomic: set at init
            temporal_locality_window_ns: AtomicU64::new(500_000_000), // 500ms in nanoseconds
            overconfidence_threshold: 0.25,                 // Non-atomic: set at init
            
            // Lock-free adaptation control
            adaptation_cycle: AtomicU64::new(0),
            last_adaptation_time: AtomicU64::new(0),
        });

        Self {
            // ... existing fields ...
            activation_dynamics: ActivationDynamics::new(),
            // ...
        }
    }
    
    /// Enable cognitive dynamics adaptation with biological parameters
    /// NOTE: sensitivity and stability_target are set at initialization, not runtime
    pub fn enable_cognitive_adaptation(&self) {
        self.params.dynamics_enabled.store(true, Ordering::Relaxed);
    }
    
    /// Create index with custom cognitive parameters
    pub fn with_cognitive_params(
        activation_sensitivity: f32,
        stability_target: f32,
        overconfidence_threshold: f32,
    ) -> Self {
        let params = Arc::new(CognitiveHnswParams {
            // ... existing HNSW defaults ...
            
            // Custom cognitive configuration
            dynamics_enabled: AtomicBool::new(false), // Start disabled
            activation_sensitivity,
            confidence_stability_target: stability_target,
            temporal_locality_window_ns: AtomicU64::new(500_000_000),
            overconfidence_threshold,
            adaptation_cycle: AtomicU64::new(0),
            last_adaptation_time: AtomicU64::new(0),
        });

        Self {
            // ... existing construction ...
            activation_dynamics: ActivationDynamics::new(),
        }
    }
    
    /// Circuit breaker: Disable adaptation if system becomes unstable
    pub fn disable_adaptation_on_instability(&self) -> bool {
        let overconfidence = self.activation_dynamics.overconfidence_ratio();
        if overconfidence > 0.5 { // >50% overconfident connections
            self.params.dynamics_enabled.store(false, Ordering::Relaxed);
            true // Signal instability detected
        } else {
            false
        }
    }
}
```

## Acceptance Criteria
- [ ] Parameters adapt to activation density patterns within 100 spreading cycles
- [ ] Overconfidence detection prevents >30% biased connections
- [ ] Temporal locality tracking improves cache-aware parameter tuning
- [ ] Cognitive adaptation maintains ef_search bounds [16, 512]
- [ ] Circuit breaker disables adaptation when instability detected (>50% overconfidence)
- [ ] Integration with existing pressure adaptation preserves cognitive load response
- [ ] Lock-free implementation maintains <1μs adaptation overhead

## Performance Targets
- Activation pattern analysis: <0.5μs overhead per spreading activation
- Parameter adaptation: <1μs per 100 cycles (amortized)
- Cognitive dynamics tracking: <2% memory overhead
- Temporal locality cache hit improvement: >15% for repetitive access patterns
- Overconfidence prevention: Maintain graph quality under diverse confidence distributions

## Risk Mitigation
- **Circuit Breaker Pattern**: Automatic disable on instability (research-backed thresholds)
- **Conservative Bounds**: ef_search ∈ [16, 512], m_max ∈ [2, 64] to prevent pathological cases
- **Epoch-Based Reclamation**: Use crossbeam patterns for lock-free safety (Turon 2015)
- **Cache-Optimal Layout**: Separate hot adaptation data from cold embeddings (Frigo et al. 1999)
- **Biological Constraints**: Activation density thresholds based on neural interference research
- **Graceful Degradation**: Fall back to pressure-only adaptation if cognitive dynamics fail

## Research Integration
- **Confidence Weighting**: Implements Dredze et al. (2008) confidence-weighted principles
- **Cache Optimization**: Applies cache-oblivious algorithms for temporal locality
- **Circuit Breaker**: Follows Nygard (2007) patterns for production reliability
- **Lock-Free Safety**: Uses crossbeam epoch-based reclamation for ABA prevention