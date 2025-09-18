//! Access tracking for tier migration decisions
//!
//! This module provides thread-safe tracking of memory access patterns
//! to inform intelligent tier migration decisions.

use dashmap::DashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, SystemTime};

/// Access pattern tracking for migration decisions
pub struct AccessTracker {
    /// Last access timestamp for each memory
    access_times: DashMap<String, SystemTime>,

    /// Total access count per memory
    access_counts: DashMap<String, u64>,

    /// Rolling activation history (last 10 values)
    activation_history: DashMap<String, Vec<f32>>,

    /// EWMA frequency estimates
    ewma_frequencies: DashMap<String, f32>,

    /// Adaptive EWMA alpha parameter (0.1-0.9)
    ewma_alpha: AtomicU64, // Stored as u64, divide by 1000 for f32

    /// Global access counter for statistics
    total_accesses: AtomicU64,
}

impl AccessTracker {
    /// Create a new access tracker
    pub fn new() -> Self {
        Self {
            access_times: DashMap::new(),
            access_counts: DashMap::new(),
            activation_history: DashMap::new(),
            ewma_frequencies: DashMap::new(),
            ewma_alpha: AtomicU64::new(300), // 0.3 as default (300/1000)
            total_accesses: AtomicU64::new(0),
        }
    }

    /// Create with custom EWMA alpha
    pub fn with_alpha(alpha: f32) -> Self {
        let mut tracker = Self::new();
        tracker.set_ewma_alpha(alpha);
        tracker
    }

    /// Set EWMA alpha parameter (0.1 to 0.9)
    pub fn set_ewma_alpha(&self, alpha: f32) {
        let alpha_int = (alpha.clamp(0.1, 0.9) * 1000.0) as u64;
        self.ewma_alpha.store(alpha_int, Ordering::Relaxed);
    }

    /// Get current EWMA alpha
    fn get_ewma_alpha(&self) -> f32 {
        self.ewma_alpha.load(Ordering::Relaxed) as f32 / 1000.0
    }

    /// Record a memory access with activation level
    pub fn record_access(&self, memory_id: &str, activation: f32) {
        let now = SystemTime::now();

        // Calculate time since last access for frequency update
        let time_delta = self.access_times
            .get(memory_id)
            .map(|entry| now.duration_since(*entry.value()).unwrap_or_default())
            .unwrap_or(Duration::from_secs(3600)); // Default to 1 hour if first access

        // Update access time
        self.access_times.insert(memory_id.to_string(), now);

        // Increment access count
        self.access_counts
            .entry(memory_id.to_string())
            .and_modify(|count| *count += 1)
            .or_insert(1);

        // Update EWMA frequency (accesses per hour)
        let instant_frequency = 3600.0 / time_delta.as_secs_f32().max(1.0);
        let alpha = self.get_ewma_alpha();

        self.ewma_frequencies
            .entry(memory_id.to_string())
            .and_modify(|freq| {
                *freq = alpha * instant_frequency + (1.0 - alpha) * (*freq);
            })
            .or_insert(instant_frequency);

        // Update activation history (keep last 10 values)
        self.activation_history
            .entry(memory_id.to_string())
            .and_modify(|history| {
                history.push(activation);
                if history.len() > 10 {
                    history.remove(0);
                }
            })
            .or_insert_with(|| vec![activation]);

        self.total_accesses.fetch_add(1, Ordering::Relaxed);
    }

    /// Get time since last access
    pub fn get_idle_time(&self, memory_id: &str) -> Duration {
        self.access_times
            .get(memory_id)
            .map(|entry| {
                SystemTime::now()
                    .duration_since(*entry.value())
                    .unwrap_or_default()
            })
            .unwrap_or(Duration::MAX)
    }

    /// Get EWMA-based access frequency (accesses per hour)
    pub fn get_access_frequency(&self, memory_id: &str) -> f32 {
        // Use EWMA frequency if available, fallback to simple calculation
        self.ewma_frequencies
            .get(memory_id)
            .map(|entry| *entry.value())
            .unwrap_or_else(|| {
                // Fallback to simple frequency calculation
                let count = self.access_counts
                    .get(memory_id)
                    .map(|entry| *entry.value())
                    .unwrap_or(0);

                let idle_hours = self.get_idle_time(memory_id).as_secs_f32() / 3600.0;

                if idle_hours < 0.001 {
                    count as f32
                } else {
                    count as f32 / idle_hours.max(1.0)
                }
            })
    }

    /// Adapt EWMA alpha based on prediction error
    pub fn adapt_alpha(&self, prediction_error: f32) {
        let current_alpha = self.get_ewma_alpha();
        let new_alpha = if prediction_error > 0.2 {
            // Large error, increase alpha for faster adaptation
            (current_alpha * 1.1).min(0.9)
        } else {
            // Small error, decrease alpha for stability
            (current_alpha * 0.95).max(0.1)
        };
        self.set_ewma_alpha(new_alpha);
    }

    /// Get average activation from recent history
    pub fn get_average_activation(&self, memory_id: &str) -> f32 {
        self.activation_history
            .get(memory_id)
            .map(|entry| {
                let history = entry.value();
                if history.is_empty() {
                    0.0
                } else {
                    history.iter().sum::<f32>() / history.len() as f32
                }
            })
            .unwrap_or(0.0)
    }

    /// Get trend in activation (positive = increasing, negative = decreasing)
    pub fn get_activation_trend(&self, memory_id: &str) -> f32 {
        self.activation_history
            .get(memory_id)
            .map(|entry| {
                let history = entry.value();
                if history.len() < 2 {
                    0.0
                } else {
                    // Simple linear trend: compare first half to second half
                    let mid = history.len() / 2;
                    let first_half_avg: f32 = history[..mid].iter().sum::<f32>() / mid as f32;
                    let second_half_avg: f32 = history[mid..].iter().sum::<f32>() / (history.len() - mid) as f32;
                    second_half_avg - first_half_avg
                }
            })
            .unwrap_or(0.0)
    }

    /// Check if memory has been accessed recently
    pub fn is_recently_accessed(&self, memory_id: &str, window: Duration) -> bool {
        self.get_idle_time(memory_id) < window
    }

    /// Get access statistics for a memory
    pub fn get_access_stats(&self, memory_id: &str) -> AccessStats {
        AccessStats {
            total_accesses: self.access_counts
                .get(memory_id)
                .map(|e| *e.value())
                .unwrap_or(0),
            idle_time: self.get_idle_time(memory_id),
            access_frequency: self.get_access_frequency(memory_id),
            average_activation: self.get_average_activation(memory_id),
            activation_trend: self.get_activation_trend(memory_id),
        }
    }

    /// Clear tracking data for a memory (e.g., after deletion)
    pub fn clear_memory(&self, memory_id: &str) {
        self.access_times.remove(memory_id);
        self.access_counts.remove(memory_id);
        self.activation_history.remove(memory_id);
    }

    /// Get total number of tracked memories
    pub fn tracked_count(&self) -> usize {
        self.access_times.len()
    }

    /// Get global access statistics
    pub fn global_stats(&self) -> GlobalAccessStats {
        GlobalAccessStats {
            total_accesses: self.total_accesses.load(Ordering::Relaxed),
            tracked_memories: self.tracked_count(),
            average_access_frequency: if self.access_counts.is_empty() {
                0.0
            } else {
                let total: u64 = self.access_counts.iter()
                    .map(|e| *e.value())
                    .sum();
                total as f32 / self.access_counts.len() as f32
            },
        }
    }
}

/// Access statistics for a single memory
#[derive(Debug, Clone)]
pub struct AccessStats {
    /// Total number of accesses
    pub total_accesses: u64,
    /// Time since last access
    pub idle_time: Duration,
    /// Access frequency (per hour)
    pub access_frequency: f32,
    /// Average activation level
    pub average_activation: f32,
    /// Trend in activation (-1.0 to 1.0)
    pub activation_trend: f32,
}

/// Global access statistics
#[derive(Debug, Clone)]
pub struct GlobalAccessStats {
    /// Total accesses across all memories
    pub total_accesses: u64,
    /// Number of tracked memories
    pub tracked_memories: usize,
    /// Average access frequency
    pub average_access_frequency: f32,
}

impl Default for AccessTracker {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread;

    #[test]
    fn test_access_tracking() {
        let tracker = AccessTracker::new();

        // Record multiple accesses
        tracker.record_access("memory1", 0.8);
        tracker.record_access("memory1", 0.9);
        tracker.record_access("memory2", 0.5);

        // Check access counts
        assert_eq!(tracker.get_access_stats("memory1").total_accesses, 2);
        assert_eq!(tracker.get_access_stats("memory2").total_accesses, 1);

        // Check that idle time is very small
        assert!(tracker.get_idle_time("memory1").as_secs() < 1);
    }

    #[test]
    fn test_activation_history() {
        let tracker = AccessTracker::new();

        // Record activation history
        for i in 0..15 {
            tracker.record_access("memory1", i as f32 / 10.0);
        }

        // Should keep only last 10 values (0.5 to 1.4)
        let avg = tracker.get_average_activation("memory1");
        assert!((avg - 0.95).abs() < 0.01); // Average of 0.5..1.4

        // Trend should be positive (increasing)
        let trend = tracker.get_activation_trend("memory1");
        assert!(trend > 0.0);
    }

    #[test]
    fn test_access_frequency() {
        let tracker = AccessTracker::new();

        // Record initial access
        tracker.record_access("memory1", 0.7);

        // Small sleep to create measurable idle time
        thread::sleep(Duration::from_millis(10));

        // Record more accesses
        tracker.record_access("memory1", 0.8);
        tracker.record_access("memory1", 0.9);

        let freq = tracker.get_access_frequency("memory1");
        assert!(freq > 0.0);

        let stats = tracker.get_access_stats("memory1");
        assert_eq!(stats.total_accesses, 3);
    }

    #[test]
    fn test_recent_access_check() {
        let tracker = AccessTracker::new();

        tracker.record_access("memory1", 0.5);

        // Should be recently accessed within 1 hour
        assert!(tracker.is_recently_accessed("memory1", Duration::from_secs(3600)));

        // Should not be recently accessed within 0 seconds
        thread::sleep(Duration::from_millis(10));
        assert!(!tracker.is_recently_accessed("memory1", Duration::from_millis(5)));
    }

    #[test]
    fn test_clear_memory() {
        let tracker = AccessTracker::new();

        tracker.record_access("memory1", 0.5);
        tracker.record_access("memory2", 0.7);

        assert_eq!(tracker.tracked_count(), 2);

        tracker.clear_memory("memory1");

        assert_eq!(tracker.tracked_count(), 1);
        assert_eq!(tracker.get_access_stats("memory1").total_accesses, 0);
        assert_eq!(tracker.get_access_stats("memory2").total_accesses, 1);
    }
}