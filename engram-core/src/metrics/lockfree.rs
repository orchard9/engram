//! Lock-free, wait-free metrics primitives with cache-line alignment

use crossbeam_utils::CachePadded;
use std::sync::atomic::{AtomicI64, AtomicU64, Ordering};

/// Lock-free counter with atomic operations
#[repr(align(64))] // Cache-line aligned
pub struct LockFreeCounter {
    value: CachePadded<AtomicU64>,
}

impl LockFreeCounter {
    #[must_use]
    pub const fn new() -> Self {
        Self {
            value: CachePadded::new(AtomicU64::new(0)),
        }
    }

    #[inline(always)]
    pub fn increment(&self, delta: u64) {
        self.value.fetch_add(delta, Ordering::Relaxed);
    }

    #[inline(always)]
    pub fn get(&self) -> u64 {
        self.value.load(Ordering::Acquire)
    }

    pub fn reset(&self) -> u64 {
        self.value.swap(0, Ordering::AcqRel)
    }
}

/// Lock-free gauge with atomic operations
#[repr(align(64))]
pub struct LockFreeGauge {
    value: CachePadded<AtomicI64>,
}

impl LockFreeGauge {
    #[must_use]
    pub const fn new() -> Self {
        Self {
            value: CachePadded::new(AtomicI64::new(0)),
        }
    }

    #[inline(always)]
    pub fn set(&self, value: i64) {
        self.value.store(value, Ordering::Release);
    }

    #[inline(always)]
    pub fn increment(&self, delta: i64) {
        self.value.fetch_add(delta, Ordering::Relaxed);
    }

    #[inline(always)]
    pub fn decrement(&self, delta: i64) {
        self.value.fetch_sub(delta, Ordering::Relaxed);
    }

    #[inline(always)]
    pub fn get(&self) -> i64 {
        self.value.load(Ordering::Acquire)
    }
}

/// Lock-free histogram with exponential buckets
pub struct LockFreeHistogram {
    /// Exponential bucket boundaries
    buckets: Vec<f64>,
    /// Atomic counters for each bucket
    counts: Vec<CachePadded<AtomicU64>>,
    /// Total count
    total_count: CachePadded<AtomicU64>,
    /// Sum of all values (for mean calculation)
    sum: CachePadded<AtomicU64>,
}

impl LockFreeHistogram {
    /// Create histogram with exponential buckets
    #[must_use]
    pub fn new() -> Self {
        Self::with_buckets(Self::default_exponential_buckets())
    }

    /// Create histogram with custom buckets
    #[must_use]
    pub fn with_buckets(buckets: Vec<f64>) -> Self {
        let counts = (0..=buckets.len())
            .map(|_| CachePadded::new(AtomicU64::new(0)))
            .collect();

        Self {
            buckets,
            counts,
            total_count: CachePadded::new(AtomicU64::new(0)),
            sum: CachePadded::new(AtomicU64::new(0)),
        }
    }

    /// Default exponential buckets for latency measurements
    fn default_exponential_buckets() -> Vec<f64> {
        let mut buckets = Vec::with_capacity(64);
        let mut boundary = 0.001; // Start at 1μs

        while boundary <= 10000.0 {
            // Up to 10s
            buckets.push(boundary);
            boundary *= 1.5; // Exponential growth
        }

        buckets
    }

    /// Record a value with <100ns overhead
    #[inline(always)]
    pub fn record(&self, value: f64) {
        // Find the appropriate bucket using binary search
        let bucket_idx = match self
            .buckets
            .binary_search_by(|b| b.partial_cmp(&value).unwrap())
        {
            Ok(idx) => idx + 1,
            Err(idx) => idx,
        };

        // Increment the bucket counter
        self.counts[bucket_idx].fetch_add(1, Ordering::Relaxed);
        self.total_count.fetch_add(1, Ordering::Relaxed);

        // Update sum for mean calculation
        let value_bits = value.to_bits();
        self.sum.fetch_add(value_bits, Ordering::Relaxed);
    }

    /// Calculate quantiles from the histogram
    pub fn quantiles(&self, quantiles: &[f64]) -> Vec<f64> {
        let total = self.total_count.load(Ordering::Acquire);
        if total == 0 {
            return vec![0.0; quantiles.len()];
        }

        let mut results = Vec::with_capacity(quantiles.len());
        let mut cumulative = 0u64;

        for quantile in quantiles {
            let target = (total as f64 * quantile) as u64;

            for (i, count) in self.counts.iter().enumerate() {
                cumulative += count.load(Ordering::Acquire);

                if cumulative >= target {
                    // Interpolate within the bucket
                    let bucket_value = if i == 0 {
                        0.0
                    } else if i > self.buckets.len() {
                        self.buckets[self.buckets.len() - 1]
                    } else {
                        self.buckets[i - 1]
                    };

                    results.push(bucket_value);
                    break;
                }
            }
        }

        results
    }

    /// Get the mean value
    pub fn mean(&self) -> f64 {
        let count = self.total_count.load(Ordering::Acquire);
        if count == 0 {
            return 0.0;
        }

        let sum_bits = self.sum.load(Ordering::Acquire);
        f64::from_bits(sum_bits) / count as f64
    }

    /// Get the total count
    pub fn count(&self) -> u64 {
        self.total_count.load(Ordering::Acquire)
    }

    /// Reset all counters
    pub fn reset(&self) {
        for count in &self.counts {
            count.store(0, Ordering::Release);
        }
        self.total_count.store(0, Ordering::Release);
        self.sum.store(0, Ordering::Release);
    }
}

impl Default for LockFreeCounter {
    fn default() -> Self {
        Self::new()
    }
}

impl Default for LockFreeGauge {
    fn default() -> Self {
        Self::new()
    }
}

impl Default for LockFreeHistogram {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use std::thread;

    #[test]
    fn test_lock_free_counter() {
        let counter = Arc::new(LockFreeCounter::new());
        let num_threads = 10;
        let increments_per_thread = 1000;

        let handles: Vec<_> = (0..num_threads)
            .map(|_| {
                let counter = Arc::clone(&counter);
                thread::spawn(move || {
                    for _ in 0..increments_per_thread {
                        counter.increment(1);
                    }
                })
            })
            .collect();

        for handle in handles {
            handle.join().unwrap();
        }

        assert_eq!(counter.get(), (num_threads * increments_per_thread) as u64);
    }

    #[test]
    fn test_lock_free_histogram() {
        let histogram = LockFreeHistogram::new();

        // Record some values
        histogram.record(0.001);
        histogram.record(0.01);
        histogram.record(0.1);
        histogram.record(1.0);
        histogram.record(10.0);

        assert_eq!(histogram.count(), 5);

        // Check quantiles
        let quantiles = histogram.quantiles(&[0.5, 0.9, 0.99]);
        assert_eq!(quantiles.len(), 3);

        // 50th percentile should be around 0.1
        assert!(quantiles[0] >= 0.01 && quantiles[0] <= 1.0);
    }

    #[test]
    fn test_lock_free_gauge() {
        let gauge = LockFreeGauge::new();

        gauge.set(100);
        assert_eq!(gauge.get(), 100);

        gauge.increment(50);
        assert_eq!(gauge.get(), 150);

        gauge.decrement(75);
        assert_eq!(gauge.get(), 75);
    }
}
