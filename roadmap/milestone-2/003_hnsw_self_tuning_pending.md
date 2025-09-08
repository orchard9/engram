# Task 003: HNSW Parameter Self-Tuning System

## Status: Pending  
## Priority: P0 - Critical Path
## Estimated Effort: 6 days
## Dependencies: Milestone-1/Task-002 (HNSW implementation)

## Objective
Implement lock-free adaptive HNSW parameter tuning that automatically adjusts M, ef_construction, and ef_search based on measured recall/precision tradeoffs and query patterns. Focus on zero-cost abstractions, cache-efficient graph traversal, and probabilistic confidence-aware search operations that leverage Engram's cognitive architecture principles.

## Current State Analysis
- **Existing**: Lock-free HNSW graph with crossbeam SkipMap layers (hnsw_graph.rs)
- **Existing**: SIMD-optimized vector operations with runtime dispatch (compute/dispatch.rs)
- **Existing**: Confidence-weighted search candidates and probabilistic operations
- **Existing**: Cognitive error handling with rich context (types.rs)
- **Missing**: Lock-free parameter adaptation during concurrent searches
- **Missing**: Cache-efficient edge storage with prefetching
- **Missing**: Reinforcement learning integration for parameter tuning
- **Missing**: Probabilistic edge pruning with confidence thresholds
- **Missing**: Zero-cost abstractions for parameter switching

## Technical Specification

### 1. Lock-Free Recall/Precision Measurement Framework

Implement cache-friendly measurement system using lock-free circular buffers and atomic operations for zero contention during concurrent searches:

```rust
// engram-core/src/index/hnsw_tuning.rs

use crate::{Confidence, compute::VectorOps};
use crossbeam_epoch::{self as epoch, Guard};
use dashmap::DashMap;
use std::sync::atomic::{AtomicU64, AtomicUsize, AtomicPtr, Ordering};
use std::sync::Arc;
use std::collections::HashMap;

/// Lock-free recall/precision monitor with cache-optimized layout
/// Aligns data structures on cache line boundaries to minimize false sharing
#[repr(align(64))] // Cache line alignment
pub struct LockFreeRecallMonitor {
    /// Lock-free ground truth cache with epoch-based reclamation
    ground_truth: DashMap<QueryFingerprint, Arc<GroundTruthSet>>,
    
    /// Lock-free circular buffer for recall measurements (cache-aligned)
    recall_buffer: LockFreeCircularBuffer<f32>,
    
    /// Lock-free circular buffer for precision measurements
    precision_buffer: LockFreeCircularBuffer<f32>,
    
    /// Hardware timestamp counters for sub-microsecond latency tracking
    latency_histogram: LockFreeHistogram,
    
    /// Confidence calibration for probabilistic quality assessment
    confidence_calibrator: ConfidenceCalibrator,
    
    /// Query pattern cache for fast similarity lookups
    pattern_cache: PatternCache,
}

/// Query fingerprint using SIMD-accelerated hash for cache efficiency
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct QueryFingerprint(u64);

impl QueryFingerprint {
    /// Generate fingerprint using SIMD-optimized hash of query vector
    pub fn from_query(query: &[f32; 768], vector_ops: &dyn VectorOps) -> Self {
        // Use first 16 dimensions for fast hash with SIMD
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        use std::hash::Hasher;
        
        // Hash normalized embedding chunks for translation invariance
        let norm = vector_ops.l2_norm_768(query);
        if norm > 0.0 {
            for chunk in query.chunks_exact(16) {
                let normalized: Vec<u32> = chunk.iter()
                    .map(|&x| ((x / norm) * 65536.0) as u32)
                    .collect();
                hasher.write_u32(normalized.iter().fold(0, |acc, &x| acc ^ x));
            }
        }
        
        Self(hasher.finish())
    }
}

/// Cache-aligned ground truth set with confidence intervals
#[repr(align(64))]
pub struct GroundTruthSet {
    /// Sorted results with confidence intervals
    results: Vec<(String, Confidence)>,
    
    /// SIMD-optimized distance precomputation
    distance_cache: Vec<f32>,
    
    /// Timestamp for cache invalidation
    created_at: std::time::Instant,
}

impl LockFreeRecallMonitor {
    /// Measure recall@k with confidence-aware scoring
    /// Uses lock-free algorithms throughout for zero contention
    pub fn measure_recall(
        &self,
        query: &[f32; 768],
        results: &[(String, Confidence)],
        k: usize,
        vector_ops: &dyn VectorOps,
    ) -> (f32, Confidence) {
        let fingerprint = QueryFingerprint::from_query(query, vector_ops);
        
        if let Some(ground_truth) = self.ground_truth.get(&fingerprint) {
            // Fast path: use cached ground truth
            self.compute_confidence_weighted_recall(&ground_truth, results, k)
        } else {
            // Slow path: estimate from consistency with probabilistic sampling
            self.estimate_recall_with_confidence(query, results, k, vector_ops)
        }
    }
    
    /// Estimate recall using confidence-weighted consistency sampling
    /// Uses SIMD-accelerated similarity computations for efficiency
    fn estimate_recall_with_confidence(
        &self,
        query: &[f32; 768],
        results: &[(String, Confidence)],
        k: usize,
        vector_ops: &dyn VectorOps,
    ) -> (f32, Confidence) {
        // Sample high-confidence reference set using elevated ef_search
        let reference_results = self.pattern_cache.get_reference_results(
            query, 
            k * 3, // Over-sample for statistical stability
            vector_ops
        );
        
        // Compute confidence-weighted overlap using SIMD operations
        let overlap = self.compute_simd_overlap(results, &reference_results, k);
        let confidence = self.confidence_calibrator.estimate_recall_confidence(
            overlap, 
            results.len(), 
            k
        );
        
        (overlap / k as f32, confidence)
    }
    
    /// Compute overlap using SIMD-accelerated string matching
    fn compute_simd_overlap(
        &self,
        results_a: &[(String, Confidence)],
        results_b: &[(String, Confidence)],
        k: usize,
    ) -> f32 {
        let mut weighted_overlap = 0.0_f32;
        
        // Use sorted order and two-pointer technique for O(n) overlap
        let mut i = 0;
        let mut j = 0;
        
        while i < results_a.len().min(k) && j < results_b.len().min(k) {
            match results_a[i].0.cmp(&results_b[j].0) {
                std::cmp::Ordering::Equal => {
                    // Weight overlap by confidence interval intersection
                    let conf_weight = results_a[i].1.raw().min(results_b[j].1.raw());
                    weighted_overlap += conf_weight;
                    i += 1;
                    j += 1;
                }
                std::cmp::Ordering::Less => i += 1,
                std::cmp::Ordering::Greater => j += 1,
            }
        }
        
        weighted_overlap
    }

    /// Compute confidence-weighted recall with statistical bounds
    fn compute_confidence_weighted_recall(
        &self,
        ground_truth: &GroundTruthSet,
        results: &[(String, Confidence)],
        k: usize,
    ) -> (f32, Confidence) {
        let overlap = self.compute_simd_overlap(results, &ground_truth.results, k);
        let recall = overlap / k as f32;
        
        // Compute confidence based on ground truth quality and sample size
        let confidence = self.confidence_calibrator.calibrate_recall(
            recall,
            ground_truth.results.len(),
            k,
        );
        
        (recall, confidence)
    }
}
```

### 2. Lock-Free Adaptive Parameter Tuning with Zero-Cost Abstractions

Implement parameter adaptation using lock-free data structures and compile-time optimized parameter switching:

```rust
// engram-core/src/index/adaptive_hnsw.rs

use super::{HnswGraph, CognitiveHnswParams};
use crate::{Confidence, compute::VectorOps};
use crossbeam_epoch::{self as epoch, Guard};
use std::sync::atomic::{AtomicUsize, AtomicU32, Ordering};
use std::sync::Arc;
use std::marker::PhantomData;

/// Lock-free adaptive HNSW index with zero-cost parameter abstractions
/// Uses const generics for compile-time optimization of common parameter sets
pub struct AdaptiveHnswIndex<const PARAM_SET: usize = 0> {
    /// Underlying lock-free HNSW graph
    graph: Arc<HnswGraph>,
    
    /// Lock-free parameter adaptation with epoch-based updates
    params: LockFreeParams<PARAM_SET>,
    
    /// Lock-free performance monitoring
    monitor: Arc<LockFreeRecallMonitor>,
    
    /// Reinforcement learning parameter tuner
    tuner: Arc<RLParameterTuner>,
    
    /// Cache-efficient query pattern analyzer
    pattern_analyzer: Arc<QueryPatternCache>,
    
    /// Vector operations dispatch
    vector_ops: Arc<dyn VectorOps>,
    
    /// Phantom data for const generic optimization
    _phantom: PhantomData<[(); PARAM_SET]>,
}

/// Lock-free parameter storage with cache-line alignment
/// Uses atomic operations for wait-free parameter updates during search
#[repr(align(64))] // Cache line alignment to prevent false sharing
struct LockFreeParams<const PARAM_SET: usize> {
    /// Current parameter epoch (for consistency)
    epoch: AtomicU32,
    
    /// Current parameter snapshot
    current: AtomicPtr<ParamSnapshot>,
    
    /// Pre-computed parameter configurations for zero-cost switching
    presets: [ParamSnapshot; 8],
    
    /// Lock-free parameter adaptation statistics
    adaptation_stats: AdaptationStats,
}

/// Immutable parameter snapshot for lock-free access
#[derive(Clone, Copy, Debug)]
struct ParamSnapshot {
    m: usize,              // Bidirectional links per layer 0 node
    m_l: usize,            // Links per node in layers > 0  
    ef_construction: usize, // Dynamic candidate list size during construction
    ef_search: usize,      // Search candidate list size
    ml: f32,               // Level generation factor
    
    // Cognitive-specific parameters for Engram integration
    confidence_threshold: f32,    // Minimum confidence for results
    decay_compensation: f32,      // Boost for decayed connections
    activation_boost: f32,        // Boost based on node activation
}

impl<const PARAM_SET: usize> AdaptiveHnswIndex<PARAM_SET> {
    /// Create adaptive index with compile-time parameter optimization
    pub fn new(
        graph: Arc<HnswGraph>,
        vector_ops: Arc<dyn VectorOps>,
    ) -> Self {
        let params = LockFreeParams::new_with_presets(PARAM_SET);
        let monitor = Arc::new(LockFreeRecallMonitor::new());
        let tuner = Arc::new(RLParameterTuner::new());
        let pattern_analyzer = Arc::new(QueryPatternCache::new());
        
        Self {
            graph,
            params,
            monitor,
            tuner,
            pattern_analyzer,
            vector_ops,
            _phantom: PhantomData,
        }
    }
    
    /// Lock-free adaptive search with zero-allocation fast path
    /// Optimizes for the common case of stable parameters
    pub fn adaptive_search(
        &self,
        query: &[f32; 768],
        k: usize,
        target_recall: f32,
    ) -> Vec<(String, Confidence)> {
        let guard = epoch::pin();
        
        // Fast path: try current parameters without adaptation overhead
        let current_params = self.params.load_current(&guard);
        
        // Probabilistic pattern analysis with caching
        let pattern = self.pattern_analyzer.analyze_cached(query, &*self.vector_ops);
        
        // Check if current parameters are likely optimal
        if self.tuner.is_likely_optimal(&current_params, &pattern, target_recall) {
            // Fast path: use current parameters with confidence boost
            return self.execute_search_with_confidence(
                query, k, &current_params, &pattern, &guard
            );
        }
        
        // Slow path: adapt parameters using reinforcement learning
        let optimal_params = self.tuner.adapt_parameters(
            &pattern,
            target_recall,
            &current_params,
        );
        
        // Atomic parameter update for concurrent searches
        self.params.try_update(optimal_params, &guard);
        
        self.execute_search_with_confidence(
            query, k, &optimal_params, &pattern, &guard
        )
    }
    
    /// Execute search with confidence calibration and performance monitoring
    fn execute_search_with_confidence(
        &self,
        query: &[f32; 768],
        k: usize,
        params: &ParamSnapshot,
        pattern: &QueryPattern,
        guard: &Guard,
    ) -> Vec<(String, Confidence)> {
        let start = std::time::Instant::now();
        
        // Execute search with cognitive parameters
        let mut results = self.graph.search(
            query,
            k,
            params.ef_search,
            Confidence::exact(params.confidence_threshold),
            &*self.vector_ops,
        );
        
        let latency = start.elapsed();
        
        // Measure recall with confidence intervals
        let (measured_recall, recall_confidence) = self.monitor.measure_recall(
            query, &results, k, &*self.vector_ops
        );
        
        // Update reinforcement learning model asynchronously
        self.tuner.update_model_async(
            pattern.clone(),
            *params,
            measured_recall,
            latency,
            recall_confidence,
        );
        
        // Apply confidence calibration to results
        for (_, confidence) in &mut results {
            *confidence = self.monitor.calibrate_result_confidence(
                *confidence,
                recall_confidence,
                measured_recall,
            );
        }
        
        results
    }
    
    /// Concurrent parameter adaptation without blocking searches
    /// Uses epoch-based reclamation for memory safety
    pub fn adapt_parameters_concurrent(
        &self,
        target_performance: PerformanceTarget,
    ) {
        let guard = epoch::pin();
        
        // Sample recent query patterns from cache
        let patterns = self.pattern_analyzer.sample_recent_patterns(100);
        
        // Run reinforcement learning optimization
        for pattern in patterns {
            let current = self.params.load_current(&guard);
            let optimal = self.tuner.optimize_for_pattern(&pattern, &current);
            
            // Try to update if significantly better
            if self.tuner.is_significantly_better(&optimal, &current) {
                self.params.try_update(optimal, &guard);
            }
        }
    }
}
```

### 3. Reinforcement Learning Parameter Tuning with Probabilistic Rewards

Implement RL-based parameter optimization using probabilistic rewards and confidence-weighted updates:

```rust
// engram-core/src/index/rl_tuner.rs

use crate::Confidence;
use crossbeam_epoch::{self as epoch, Guard};
use dashmap::DashMap;
use std::sync::atomic::{AtomicU32, AtomicUsize, Ordering};
use std::sync::Arc;
use std::collections::VecDeque;
use std::time::{Duration, Instant};

/// Reinforcement learning parameter tuner with probabilistic reward modeling
/// Uses multi-armed bandit with contextual information for parameter selection
pub struct RLParameterTuner {
    /// Lock-free bandit model with confidence intervals
    bandit: Arc<ContextualBandit>,
    
    /// Lock-free performance history with sliding window
    performance_buffer: LockFreeRingBuffer<PerformanceObservation>,
    
    /// Adaptive exploration rate with confidence-based annealing
    exploration_schedule: ExplorationSchedule,
    
    /// Performance constraints for safe exploration
    constraints: PerformanceConstraints,
    
    /// Pattern-specific parameter cache
    pattern_cache: DashMap<PatternFingerprint, CachedParams>,
}

/// Performance observation with confidence intervals
#[derive(Clone, Debug)]
struct PerformanceObservation {
    params: ParamSnapshot,
    pattern: QueryPattern,
    
    // Performance metrics with confidence
    recall: (f32, Confidence),           // (value, confidence)
    precision: (f32, Confidence),        // (value, confidence)  
    latency: Duration,                   // Measured latency
    throughput: f32,                     // Queries/second
    
    // Context information
    timestamp: Instant,
    query_load: f32,                     // Current system load
    cache_hit_rate: f32,                 // Pattern cache effectiveness
}

struct PerformancePoint {
    params: HnswParams,
    recall: f32,
    precision: f32,
    latency: Duration,
    query_pattern: QueryPattern,
}

/// Contextual multi-armed bandit for parameter selection
struct ContextualBandit {
    /// Feature extraction for query patterns
    feature_extractor: FeatureExtractor,
    
    /// Thompson sampling with confidence intervals
    arms: Vec<BanditArm>,
    
    /// Contextual linear bandit model
    linear_model: LinearBandit,
    
    /// Reward model with uncertainty quantification
    reward_model: ProbabilisticRewardModel,
}

/// Individual bandit arm representing parameter configuration
struct BanditArm {
    params: ParamSnapshot,
    
    // Performance statistics with confidence intervals
    recall_stats: BetaDistribution,      // Beta distribution for recall
    latency_stats: GammaDistribution,    // Gamma distribution for latency
    
    // Context-dependent performance
    context_model: ContextualRegressor,
    
    // Usage statistics
    selection_count: AtomicU32,
    last_selected: AtomicU64,            // Timestamp
}

impl RLParameterTuner {
    /// Get optimal parameters using contextual bandit with confidence bounds
    pub fn get_optimal_params(
        &self,
        pattern: &QueryPattern,
        target_recall: f32,
    ) -> ParamSnapshot {
        // Check cache for similar patterns first
        if let Some(cached) = self.check_pattern_cache(pattern) {
            if self.is_cache_valid(&cached, target_recall) {
                return cached.params;
            }
        }
        
        // Extract contextual features
        let context_features = self.bandit.feature_extractor.extract(pattern);
        
        // Decide exploration vs exploitation using confidence-based epsilon-greedy
        let exploration_prob = self.exploration_schedule.get_current_rate(
            &context_features,
            target_recall,
        );
        
        if rand::random::<f32>() < exploration_prob {
            self.explore_with_constraints(pattern, target_recall)
        } else {
            self.exploit_with_confidence_bounds(pattern, target_recall, &context_features)
        }
    }
    
    /// Exploit best parameters using upper confidence bounds
    fn exploit_with_confidence_bounds(
        &self,
        pattern: &QueryPattern,
        target_recall: f32,
        context_features: &[f32],
    ) -> ParamSnapshot {
        let mut best_arm = None;
        let mut best_ucb = f32::NEG_INFINITY;
        
        // Thompson sampling with contextual information
        for arm in &self.bandit.arms {
            // Predict performance using contextual model
            let predicted_recall = arm.context_model.predict_recall(context_features);
            let predicted_latency = arm.context_model.predict_latency(context_features);
            
            // Compute upper confidence bound with recall constraint
            let recall_confidence = arm.recall_stats.confidence_interval(0.95).1;
            let latency_confidence = arm.latency_stats.confidence_interval(0.95).0;
            
            // Multi-objective UCB balancing recall and latency
            let recall_score = if predicted_recall >= target_recall {
                recall_confidence
            } else {
                // Penalty for not meeting recall target
                predicted_recall - (target_recall - predicted_recall).powi(2)
            };
            
            let latency_score = 1.0 / (1.0 + latency_confidence.as_secs_f32());
            
            // Combined UCB score with confidence weighting
            let ucb_score = 0.7 * recall_score + 0.3 * latency_score;
            
            if ucb_score > best_ucb && self.satisfies_constraints(&arm.params) {
                best_ucb = ucb_score;
                best_arm = Some(arm.params);
            }
        }
        
        best_arm.unwrap_or_else(|| self.get_safe_default_params())
    }
    
    /// Explore parameters using guided random search with safety constraints
    fn explore_with_constraints(
        &self,
        pattern: &QueryPattern,
        target_recall: f32,
    ) -> ParamSnapshot {
        // Use current best as baseline
        let baseline = self.get_current_best_for_pattern(pattern, target_recall);
        
        // Gaussian exploration around current best with adaptive variance
        let exploration_variance = self.exploration_schedule.get_variance_for_pattern(pattern);
        
        let mut explored_params = baseline;
        
        // Explore ef_search with higher probability (most impactful parameter)
        if rand::random::<f32>() < 0.6 {
            let ef_delta = rand_normal(0.0, exploration_variance.ef_search);
            explored_params.ef_search = (
                (baseline.ef_search as f32 * (1.0 + ef_delta))
                    .clamp(16.0, 1024.0) as usize
            );
        }
        
        // Explore M with medium probability
        if rand::random::<f32>() < 0.3 {
            let m_delta = rand_normal(0.0, exploration_variance.m);
            explored_params.m = (
                (baseline.m as f32 * (1.0 + m_delta))
                    .clamp(4.0, 64.0) as usize
            );
        }
        
        // Explore cognitive parameters
        if rand::random::<f32>() < 0.2 {
            let conf_delta = rand_normal(0.0, exploration_variance.confidence);
            explored_params.confidence_threshold = (
                baseline.confidence_threshold + conf_delta
            ).clamp(0.1, 0.95);
        }
        
        // Ensure explored parameters satisfy performance constraints
        self.apply_safety_constraints(explored_params, &baseline)
    }
    
    /// Update bandit model with observed performance using confidence-weighted rewards
    pub fn update_model_async(
        &self,
        pattern: QueryPattern,
        params: ParamSnapshot,
        measured_recall: f32,
        latency: Duration,
        recall_confidence: Confidence,
    ) {
        let observation = PerformanceObservation {
            params,
            pattern: pattern.clone(),
            recall: (measured_recall, recall_confidence),
            precision: (0.0, Confidence::LOW), // TODO: implement precision measurement
            latency,
            throughput: self.estimate_current_throughput(),
            timestamp: Instant::now(),
            query_load: self.get_current_load(),
            cache_hit_rate: self.pattern_cache.hit_rate(),
        };
        
        // Add to performance buffer (lock-free)
        self.performance_buffer.push(observation.clone());
        
        // Update corresponding bandit arm with confidence-weighted reward
        if let Some(arm_id) = self.find_matching_arm(&params) {
            let arm = &self.bandit.arms[arm_id];
            
            // Update recall distribution with confidence weighting
            let weight = recall_confidence.raw();
            arm.recall_stats.update_weighted(measured_recall, weight);
            
            // Update latency distribution
            arm.latency_stats.update(latency.as_secs_f32());
            
            // Update contextual model
            let context_features = self.bandit.feature_extractor.extract(&pattern);
            arm.context_model.update(
                &context_features,
                measured_recall,
                latency.as_secs_f32(),
                weight,
            );
            
            arm.selection_count.fetch_add(1, Ordering::Relaxed);
        }
        
        // Update pattern cache
        self.update_pattern_cache(pattern, params, &observation);
        
        // Adaptive exploration rate based on convergence
        self.exploration_schedule.update_based_on_performance(&observation);
    }
    
    /// Calculate probabilistic reward with uncertainty quantification
    fn calculate_probabilistic_reward(
        &self,
        observation: &PerformanceObservation,
    ) -> (f32, f32) { // (reward, uncertainty)
        let (recall, recall_conf) = observation.recall;
        let latency_ms = observation.latency.as_secs_f32() * 1000.0;
        
        // Multi-objective reward balancing recall, latency, and confidence
        let recall_component = recall * recall_conf.raw();
        let latency_component = (-latency_ms / 100.0).exp(); // Exponential decay
        let throughput_component = (observation.throughput / 1000.0).min(1.0);
        
        // Weighted combination with adaptive weights based on current constraints
        let recall_weight = self.constraints.recall_weight();
        let latency_weight = self.constraints.latency_weight();
        let throughput_weight = 1.0 - recall_weight - latency_weight;
        
        let reward = recall_weight * recall_component
            + latency_weight * latency_component
            + throughput_weight * throughput_component;
        
        // Uncertainty based on confidence intervals and sample size
        let confidence_uncertainty = 1.0 - recall_conf.raw();
        let sample_uncertainty = 1.0 / (1.0 + observation.params.selection_count() as f32).sqrt();
        let total_uncertainty = confidence_uncertainty.max(sample_uncertainty);
        
        (reward, total_uncertainty)
    }
    
    /// Get safe default parameters that satisfy all constraints
    fn get_safe_default_params(&self) -> ParamSnapshot {
        ParamSnapshot {
            m: 16,
            m_l: 16,
            ef_construction: 200,
            ef_search: 100,
            ml: 1.0 / (2.0_f32.ln()),
            confidence_threshold: 0.5,
            decay_compensation: 0.1,
            activation_boost: 0.0,
        }
    }
    
    /// Check if parameters satisfy performance constraints
    fn satisfies_constraints(&self, params: &ParamSnapshot) -> bool {
        params.ef_search >= 16
            && params.ef_search <= 1024
            && params.m >= 4
            && params.m <= 64
            && params.confidence_threshold >= 0.0
            && params.confidence_threshold <= 1.0
    }
}
```

### 4. Cache-Efficient Query Pattern Analysis with SIMD Optimization

Implement pattern analysis using SIMD-accelerated feature extraction and cache-friendly data structures:

```rust
// engram-core/src/index/query_patterns.rs

#[derive(Debug, Clone)]
pub struct QueryPattern {
    /// Embedding statistics
    sparsity: f32,           // Percentage of near-zero values
    entropy: f32,            // Information entropy
    norm: f32,               // L2 norm
    
    /// Temporal patterns
    query_frequency: f32,    // Queries per second
    time_of_day: u8,        // Hour of day
    
    /// Result patterns
    typical_k: usize,        // Common k value
    selectivity: f32,        // How selective queries are
}

pub struct QueryPatternAnalyzer {
    /// Pattern clustering model
    clusterer: OnlineKMeans,
    
    /// Pattern statistics
    pattern_stats: DashMap<ClusterId, PatternStatistics>,
    
    /// Sliding window of recent queries
    recent_queries: CircularBuffer<QueryPattern>,
}

impl QueryPatternAnalyzer {
    pub fn analyze(&self, query: &[f32; 768]) -> QueryPattern {
        let sparsity = query.iter()
            .filter(|&&v| v.abs() < 1e-6)
            .count() as f32 / 768.0;
            
        let entropy = self.calculate_entropy(query);
        let norm = query.iter().map(|v| v * v).sum::<f32>().sqrt();
        
        QueryPattern {
            sparsity,
            entropy,
            norm,
            query_frequency: self.get_current_qps(),
            time_of_day: Local::now().hour() as u8,
            typical_k: self.get_typical_k(),
            selectivity: self.estimate_selectivity(query),
        }
    }
    
    fn estimate_selectivity(&self, query: &[f32; 768]) -> f32 {
        // Sample random vectors and check similarity distribution
        let sample_size = 100;
        let mut similarities = Vec::with_capacity(sample_size);
        
        for _ in 0..sample_size {
            let random_vec = self.get_random_indexed_vector();
            let sim = crate::compute::cosine_similarity_768(query, &random_vec);
            similarities.push(sim);
        }
        
        // High selectivity = few high similarity matches
        let high_sim_count = similarities.iter()
            .filter(|&&s| s > 0.8)
            .count();
            
        // Simple QPS estimation based on recent activity
        100.0 // Placeholder - would be implemented with sliding window
    }
}
```

## Integration Points

### Enhance HnswGraph (index/hnsw_graph.rs)
```rust
// Add to existing HnswGraph struct around line 14:
pub struct HnswGraph {
    // ... existing fields ...
    
    // Lock-free adaptive tuning integration
    adaptive_params: Option<Arc<LockFreeParams<0>>>,
    performance_monitor: Option<Arc<LockFreeRecallMonitor>>,
    
    // Cache-optimized edge storage
    edge_cache: Arc<EdgeCache>,
    
    // Probabilistic edge pruning
    pruning_strategy: Arc<dyn ProbabilisticPruning>,
}

// Enhanced search method with parameter adaptation:
impl HnswGraph {
    /// Search with lock-free parameter adaptation
    pub fn adaptive_search(
        &self,
        query: &[f32; 768],
        k: usize,
        target_confidence: Confidence,
        vector_ops: &dyn VectorOps,
    ) -> Vec<(String, Confidence)> {
        let guard = epoch::pin();
        
        // Get current parameters without blocking
        let params = self.adaptive_params
            .as_ref()
            .map(|p| p.load_current(&guard))
            .unwrap_or_default();
        
        // Execute search with cache-efficient traversal
        let mut results = self.search_with_cache_optimization(
            query,
            k * 2, // Over-fetch for better recall
            params.ef_search,
            target_confidence,
            vector_ops,
            &guard,
        );
        
        // Apply probabilistic filtering with confidence thresholds
        self.filter_by_confidence_and_diversity(
            &mut results,
            k,
            target_confidence,
            params.confidence_threshold,
        );
        
        results.truncate(k);
        results
    }
    
    /// Cache-optimized search with prefetching and SIMD batch operations
    fn search_with_cache_optimization(
        &self,
        query: &[f32; 768],
        k: usize,
        ef: usize,
        threshold: Confidence,
        vector_ops: &dyn VectorOps,
        guard: &Guard,
    ) -> Vec<(String, Confidence)> {
        // Pre-allocate with cache-aligned memory
        let mut candidates = Vec::with_capacity(ef);
        let mut visited = std::collections::HashSet::with_capacity(ef * 2);
        
        // Start from highest layer entry point
        let entry_point = self.find_best_entry_point(query, vector_ops, guard)?;
        
        // Multi-layer search with cache-aware traversal
        for layer in (0..16).rev() {
            if self.get_entry_point(layer) == u32::MAX {
                continue;
            }
            
            // Batch process neighbors with SIMD
            candidates = self.search_layer_with_batching(
                query,
                entry_point,
                if layer == 0 { ef } else { 1 },
                layer,
                vector_ops,
                &mut visited,
                guard,
            )?;
            
            // Prefetch next layer nodes
            if layer > 0 {
                self.prefetch_layer_nodes(&candidates, layer - 1, guard);
            }
        }
        
        // Convert to results with confidence
        self.candidates_to_results(candidates, threshold, guard)
    }
    
    /// Enable adaptive tuning with lock-free initialization
    pub fn enable_adaptive_tuning(
        &self,
        initial_params: ParamSnapshot,
    ) -> Result<(), super::HnswError> {
        // This method would be called during initialization
        // to set up the lock-free adaptive components
        Ok(())
    }
}
```

## Testing Strategy

### Unit Tests - Lock-Free Correctness and Performance
```rust
#[test]
fn test_lockfree_recall_measurement() {
    use crate::compute::dispatch::DispatchVectorOps;
    
    let monitor = LockFreeRecallMonitor::new();
    let vector_ops = DispatchVectorOps::new();
    
    // Test with confidence-weighted ground truth
    let ground_truth = vec![
        ("memory_1".to_string(), Confidence::exact(0.95)),
        ("memory_2".to_string(), Confidence::exact(0.90)),
        ("memory_3".to_string(), Confidence::interval(0.85, 0.80, 0.90)),
    ];
    
    let results = vec![
        ("memory_1".to_string(), Confidence::exact(0.94)),
        ("memory_3".to_string(), Confidence::interval(0.86, 0.82, 0.90)),
        ("memory_4".to_string(), Confidence::exact(0.80)),
    ];
    
    let query = [0.1f32; 768]; // Test query
    let (recall, confidence) = monitor.measure_recall(&query, &results, 3, &vector_ops);
    
    // Should handle confidence intervals correctly
    assert!((recall - 0.667).abs() < 0.05);
    assert!(confidence.raw() > 0.5); // Should have reasonable confidence
}

#[test]
fn test_lockfree_parameter_adaptation() {
    let tuner = RLParameterTuner::new();
    
    let pattern = QueryPattern {
        sparsity: 0.1,
        entropy: 0.8,
        norm: 1.0,
        embedding_stats: EmbeddingStats::default(),
        temporal_context: TemporalContext::peak_hours(),
        cognitive_context: CognitiveContext::high_precision(),
    };
    
    // Should adapt ef_search based on pattern complexity
    let params = tuner.get_optimal_params(&pattern, 0.95);
    assert!(params.ef_search >= 100); // High recall requirement
    assert!(params.confidence_threshold <= 0.6); // Should relax for recall
    
    // Test concurrent parameter updates
    let params2 = tuner.get_optimal_params(&pattern, 0.95);
    assert_eq!(params.ef_search, params2.ef_search); // Should be consistent
}

#[test]
fn test_simd_pattern_analysis() {
    use crate::compute::dispatch::DispatchVectorOps;
    
    let analyzer = QueryPatternCache::new();
    let vector_ops = DispatchVectorOps::new();
    
    // Test SIMD-accelerated feature extraction
    let query1 = [1.0f32; 768];
    let query2 = [0.5f32; 768];
    
    let pattern1 = analyzer.analyze_cached(&query1, &vector_ops);
    let pattern2 = analyzer.analyze_cached(&query2, &vector_ops);
    
    // Patterns should be different
    assert!((pattern1.norm - pattern2.norm).abs() > 0.1);
    assert!((pattern1.sparsity - pattern2.sparsity).abs() < 0.01); // Same sparsity
    
    // Test cache effectiveness
    let pattern1_cached = analyzer.analyze_cached(&query1, &vector_ops);
    assert_eq!(pattern1.fingerprint(), pattern1_cached.fingerprint());
}

#[test]
fn test_confidence_weighted_rewards() {
    let tuner = RLParameterTuner::new();
    
    let observation_high_conf = PerformanceObservation {
        params: ParamSnapshot::default(),
        pattern: QueryPattern::default(),
        recall: (0.9, Confidence::exact(0.95)),
        precision: (0.85, Confidence::exact(0.90)),
        latency: Duration::from_millis(50),
        throughput: 100.0,
        timestamp: Instant::now(),
        query_load: 0.5,
        cache_hit_rate: 0.8,
    };
    
    let observation_low_conf = PerformanceObservation {
        recall: (0.9, Confidence::interval(0.9, 0.7, 0.95)), // Same recall, lower confidence
        ..observation_high_conf.clone()
    };
    
    let (reward_high, uncertainty_high) = tuner.calculate_probabilistic_reward(&observation_high_conf);
    let (reward_low, uncertainty_low) = tuner.calculate_probabilistic_reward(&observation_low_conf);
    
    // High confidence observation should have higher reward and lower uncertainty
    assert!(reward_high > reward_low);
    assert!(uncertainty_high < uncertainty_low);
}
```

### Integration Tests - End-to-End Adaptive Behavior
```rust
#[test]
fn test_adaptive_search_improvement() {
    let mut index = create_test_index_with_data();
    index.enable_adaptive_tuning(0.9);
    
    let mut recalls = Vec::new();
    
    // Run queries and measure improvement
    for i in 0..100 {
        let query = generate_test_query(i);
        let results = index.adaptive_search(&query, 10, 0.9);
        
        let recall = measure_actual_recall(&results);
        recalls.push(recall);
    }
    
    // Recall should improve over time
    let early_avg = recalls[..20].iter().sum::<f32>() / 20.0;
    let late_avg = recalls[80..].iter().sum::<f32>() / 20.0;
    
    assert!(late_avg > early_avg * 1.1); // 10% improvement
}
```

## Acceptance Criteria
- [ ] Recall measurement accurate to within 2% of ground truth
- [ ] Parameters adapt to achieve target recall within 20 queries
- [ ] Query latency remains under SLA (1ms) while improving recall
- [ ] Pattern recognition identifies at least 5 distinct query types
- [ ] Exploration rate properly decays to pure exploitation
- [ ] Performance history used to predict optimal parameters

## Performance Targets
- Recall measurement overhead: <5% of query time
- Parameter adaptation: <10μs per query
- Achieve 90% recall@10 within 100 queries
- Maintain <1ms query latency at 90% recall
- Pattern analysis: <50μs per query

## Risk Mitigation
- Fallback to fixed parameters if tuning degrades performance
- Bounded exploration to prevent bad parameter choices
- Separate validation set for unbiased recall measurement
- Circuit breaker for unstable parameter oscillation

## Implementation Files

Create the following new files:
- `engram-core/src/index/hnsw_tuning.rs` - Lock-free recall/precision monitoring
- `engram-core/src/index/rl_tuner.rs` - Reinforcement learning parameter optimization  
- `engram-core/src/index/pattern_cache.rs` - SIMD-accelerated query pattern analysis
- `engram-core/src/index/adaptive_hnsw.rs` - Main adaptive HNSW interface
- `engram-core/src/index/edge_cache.rs` - Cache-efficient edge storage and prefetching
- `engram-core/src/index/probabilistic_pruning.rs` - Confidence-aware edge pruning

Modify existing files:
- `engram-core/src/index/hnsw_graph.rs` - Add adaptive search methods and cache optimization
- `engram-core/src/index/mod.rs` - Export new adaptive types
- `engram-core/src/lib.rs` - Add feature flags for adaptive tuning

Benchmarks:
- `engram-core/benches/adaptive_hnsw_benchmark.rs` - Performance comparison with fixed parameters
- `engram-core/benches/lockfree_benchmark.rs` - Concurrent access patterns

Integration tests:
- `engram-core/tests/adaptive_integration_tests.rs` - End-to-end adaptive behavior
- `engram-core/tests/confidence_calibration_tests.rs` - Probabilistic correctness

## Dependencies

Add to `Cargo.toml`:
```toml
[dependencies]
# Existing dependencies...

# For reinforcement learning
linfa = "0.7"
linfa-bayes = "0.7"

# For statistical distributions
statrs = "0.16"

# For lock-free data structures (already present)
crossbeam-epoch = "0.9"
crossbeam-utils = "0.8"

# For SIMD pattern analysis  
wide = "0.7"

[dev-dependencies]
# For statistical testing
approx = "0.5"
quickcheck = "1.0"
```

## Technical Architecture Summary

This enhanced HNSW self-tuning system leverages Jon Gjengset's expertise in lock-free concurrent data structures and high-performance Rust systems programming:

### Lock-Free Graph Traversal Optimizations
- **Epoch-Based Reclamation**: Uses crossbeam-epoch for safe concurrent parameter updates without blocking searches
- **Cache-Line Aligned Data**: Prevents false sharing between concurrent readers/writers  
- **Lock-Free Circular Buffers**: Zero-contention performance monitoring during high-throughput search operations

### Cache-Efficient Edge Storage
- **NUMA-Aware Allocation**: Data structures aligned to cache boundaries and NUMA topology
- **Prefetching Strategies**: Predictive prefetching of neighbor nodes during multi-layer traversal
- **SIMD Batch Operations**: Vectorized similarity computations for edge filtering and ranking

### Zero-Cost Abstractions
- **Const Generic Optimization**: Compile-time specialization for common parameter configurations
- **Monomorphization**: Runtime dispatch eliminated for hot paths through static trait bounds
- **Inline Assembly**: Critical similarity computations optimized with AVX2/AVX512 intrinsics

### Probabilistic Operations Integration
- **Confidence-Weighted Rewards**: RL model incorporates Engram's uncertainty quantification
- **Bayesian Parameter Updates**: Statistical confidence intervals guide exploration vs exploitation  
- **Cognitive Context Awareness**: Parameter adaptation considers memory consolidation states

This implementation maintains Engram's probabilistic semantics while achieving the performance characteristics demanded by production cognitive systems. The lock-free design ensures consistent low-latency operation under concurrent load, while the RL-based adaptation provides principled optimization of the recall/latency trade-off.