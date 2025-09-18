//! Cognitive dynamics analysis for HNSW adaptive parameters
//!
//! This module implements activation pattern analysis and temporal adaptation
//! based on biological principles rather than ML accuracy metrics. It tracks
//! activation density, confidence distributions, and temporal locality to
//! enable cognitive-aware parameter adaptation.

use std::time::{Duration, Instant};
use std::sync::atomic::{AtomicU64, Ordering};
use crossbeam_queue::ArrayQueue;
use parking_lot::Mutex;
use crate::Confidence;

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
    /// Create a new activation dynamics tracker
    #[must_use]
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
        let now = Instant::now().elapsed().as_nanos().try_into().unwrap_or(u64::MAX);
        let index = self.access_index.fetch_add(1, Ordering::Relaxed) % 32;
        self.access_timestamps[index as usize].store(now, Ordering::Relaxed);
        
        // Confidence variance update (less frequent, acceptable lock)
        if let Some(mut variance) = self.confidence_variance.try_lock() {
            variance.update(confidence.raw());
        }
        // If lock contention, skip confidence update (non-critical)
    }
    
    /// Compute current activation density for parameter adaptation
    #[must_use]
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
    #[must_use]
    pub fn overconfidence_ratio(&self) -> f32 {
        let overconfident = self.overconfident_connections.load(Ordering::Relaxed);
        let total = self.total_connections.load(Ordering::Relaxed);
        if total == 0 { 0.0 } else { overconfident as f32 / total as f32 }
    }
    
    /// Measure temporal locality for cache-aware parameter tuning
    #[must_use]
    pub fn temporal_locality_factor(&self) -> f32 {
        let now = Instant::now().elapsed().as_nanos().try_into().unwrap_or(u64::MAX);
        let recent_threshold = Duration::from_millis(100).as_nanos().try_into().unwrap_or(u64::MAX);
        
        let recent_accesses = self.access_timestamps
            .iter()
            .map(|timestamp| timestamp.load(Ordering::Relaxed))
            .filter(|&timestamp| timestamp > 0 && (now.saturating_sub(timestamp)) < recent_threshold)
            .count();
            
        recent_accesses as f32 / 32.0 // Normalize to [0,1]
    }
    
    /// Record a connection attempt for overconfidence tracking
    pub fn record_connection(&self, confidence: Confidence, overconfidence_threshold: f32) {
        self.total_connections.fetch_add(1, Ordering::Relaxed);
        
        if confidence.raw() > overconfidence_threshold {
            self.overconfident_connections.fetch_add(1, Ordering::Relaxed);
        }
    }
}

impl Default for ActivationDynamics {
    fn default() -> Self {
        Self::new()
    }
}

/// Running variance calculation for confidence distribution (Welford's algorithm)
struct RunningVariance {
    count: u64,
    mean: f64,
    m2: f64, // Sum of squares of differences from mean
}

impl RunningVariance {
    const fn new() -> Self {
        Self { count: 0, mean: 0.0, m2: 0.0 }
    }
    
    fn update(&mut self, value: f32) {
        self.count += 1;
        let delta = f64::from(value) - self.mean;
        self.mean += delta / self.count as f64;
        let delta2 = f64::from(value) - self.mean;
        self.m2 += delta * delta2;
    }
    
    #[must_use]
    fn variance(&self) -> f32 {
        if self.count < 2 { 
            0.0 
        } else { 
            (self.m2 / (self.count - 1) as f64) as f32 
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_activation_dynamics_creation() {
        let dynamics = ActivationDynamics::new();
        assert_eq!(dynamics.compute_activation_density(), 0.0);
        assert_eq!(dynamics.overconfidence_ratio(), 0.0);
        assert_eq!(dynamics.temporal_locality_factor(), 0.0);
    }

    #[test]
    fn test_activation_recording() {
        let dynamics = ActivationDynamics::new();
        
        // Record some activations
        for i in 0..10 {
            let energy = i as f32 / 10.0;
            let confidence = Confidence::exact(0.5);
            dynamics.record_activation(energy, confidence);
        }
        
        // Should have some density now
        assert!(dynamics.compute_activation_density() > 0.0);
    }

    #[test]
    fn test_overconfidence_tracking() {
        let dynamics = ActivationDynamics::new();
        let threshold = 0.8;
        
        // Record normal confidence connections
        for _ in 0..5 {
            dynamics.record_connection(Confidence::exact(0.5), threshold);
        }
        
        // Record overconfident connections
        for _ in 0..2 {
            dynamics.record_connection(Confidence::exact(0.9), threshold);
        }
        
        let ratio = dynamics.overconfidence_ratio();
        assert!((ratio - 2.0/7.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_running_variance() {
        let mut variance = RunningVariance::new();
        
        // Add some values
        variance.update(1.0);
        variance.update(2.0);
        variance.update(3.0);
        
        // Should have some variance
        assert!(variance.variance() > 0.0);
    }
}