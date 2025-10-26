//! Lock-free, wait-free metrics primitives with cache-line alignment

use atomic_float::AtomicF64;
use crossbeam_utils::CachePadded;
use std::cmp::Ordering as CmpOrdering;
use std::convert::TryFrom;
use std::sync::atomic::{AtomicI64, AtomicU64, Ordering};

/// Lock-free counter with atomic operations
#[repr(align(64))] // Cache-line aligned
pub struct LockFreeCounter {
    value: CachePadded<AtomicU64>,
}

impl LockFreeCounter {
    /// Create a zero-initialized counter.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            value: CachePadded::new(AtomicU64::new(0)),
        }
    }

    /// Add `delta` to the counter using relaxed ordering.
    pub fn increment(&self, delta: u64) {
        self.value.fetch_add(delta, Ordering::Relaxed);
    }

    /// Load the current value with acquire semantics.
    pub fn get(&self) -> u64 {
        self.value.load(Ordering::Acquire)
    }

    /// Reset the counter to zero, returning the previous value.
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
    /// Create a zero-initialized gauge.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            value: CachePadded::new(AtomicI64::new(0)),
        }
    }

    /// Set the gauge to an absolute value.
    pub fn set(&self, value: i64) {
        self.value.store(value, Ordering::Release);
    }

    /// Increment the gauge by `delta` using relaxed ordering.
    pub fn increment(&self, delta: i64) {
        self.value.fetch_add(delta, Ordering::Relaxed);
    }

    /// Decrement the gauge by `delta` using relaxed ordering.
    pub fn decrement(&self, delta: i64) {
        self.value.fetch_sub(delta, Ordering::Relaxed);
    }

    /// Load the current gauge value with acquire semantics.
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
    /// CORRECTED: Use AtomicF64 instead of storing bit representation in AtomicU64
    sum: CachePadded<AtomicF64>,
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
            // CORRECTED: Initialize with AtomicF64, not bit-packed u64
            sum: CachePadded::new(AtomicF64::new(0.0)),
        }
    }

    /// Default exponential buckets for latency measurements
    fn default_exponential_buckets() -> Vec<f64> {
        let mut buckets = Vec::with_capacity(64);
        let mut boundary = 0.001; // Start at 1μs

        loop {
            if boundary > 10_000.0 {
                break;
            }
            // Up to 10s
            buckets.push(boundary);
            boundary *= 1.5; // Exponential growth
        }

        buckets
    }

    /// Record a value with <100ns overhead
    pub fn record(&self, value: f64) {
        // Find the appropriate bucket using binary search
        let bucket_idx = match self.buckets.binary_search_by(|boundary| {
            boundary
                .partial_cmp(&value)
                .map_or(CmpOrdering::Greater, |ordering| ordering)
        }) {
            Ok(idx) => idx + 1,
            Err(idx) => idx,
        };

        // Increment the bucket counter
        self.counts[bucket_idx].fetch_add(1, Ordering::Relaxed);
        self.total_count.fetch_add(1, Ordering::Relaxed);

        // CORRECTED: Use atomic f64 add, not bit representation
        self.sum.fetch_add(value, Ordering::Relaxed);
    }

    /// Calculate quantiles from the histogram
    ///
    /// # Panics
    /// Panics if any requested quantile is not a finite number.
    #[must_use]
    pub fn quantiles(&self, quantiles: &[f64]) -> Vec<f64> {
        let total = self.total_count.load(Ordering::Acquire);
        if total == 0 {
            return vec![0.0; quantiles.len()];
        }

        let mut results = Vec::with_capacity(quantiles.len());

        for quantile in quantiles {
            assert!(quantile.is_finite(), "quantile must be finite");
            let clamped = quantile.clamp(0.0, 1.0);
            let target = u64_to_f64(total) * clamped;
            let mut cumulative = 0_u64;
            let mut found = None;

            for (i, count) in self.counts.iter().enumerate() {
                cumulative += count.load(Ordering::Acquire);

                if u64_to_f64(cumulative) >= target {
                    let bucket_value = match i {
                        0 => self.buckets.first().copied().unwrap_or(0.0) / 2.0,
                        idx if idx > self.buckets.len() => {
                            self.buckets.last().copied().unwrap_or(0.0)
                        }
                        _ => self.buckets[i - 1],
                    };

                    found = Some(bucket_value);
                    break;
                }
            }

            results.push(found.unwrap_or_else(|| self.buckets.last().copied().unwrap_or(0.0)));
        }

        results
    }

    /// Get the mean value
    #[must_use]
    pub fn mean(&self) -> f64 {
        let count = self.total_count.load(Ordering::Acquire);
        if count == 0 {
            return 0.0;
        }

        // CORRECTED: Load actual f64 value
        let sum = self.sum.load(Ordering::Acquire);
        sum / u64_to_f64(count)
    }

    /// Get the total count
    #[must_use]
    pub fn count(&self) -> u64 {
        self.total_count.load(Ordering::Acquire)
    }

    /// Reset all counters
    pub fn reset(&self) {
        for count in &self.counts {
            count.store(0, Ordering::Release);
        }
        self.total_count.store(0, Ordering::Release);
        // CORRECTED: Reset f64 sum properly
        self.sum.store(0.0, Ordering::Release);
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

fn u64_to_f64(value: u64) -> f64 {
    // Split into high/low 32-bit components to use lossless conversions
    let high_part = u32::try_from(value >> 32).unwrap_or(u32::MAX);
    let low_part = u32::try_from(value & 0xFFFF_FFFF).unwrap_or(u32::MAX);
    f64::from(high_part).mul_add(4_294_967_296.0, f64::from(low_part))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fmt::Debug;
    use std::sync::Arc;
    use std::thread;

    type TestResult<T = ()> = Result<T, String>;

    fn ensure_eq<T>(actual: &T, expected: &T, context: &str) -> TestResult
    where
        T: PartialEq + Debug,
    {
        if actual == expected {
            Ok(())
        } else {
            Err(format!("{context}: expected {expected:?}, got {actual:?}"))
        }
    }

    #[test]
    fn test_lock_free_counter() -> TestResult {
        let counter = Arc::new(LockFreeCounter::new());
        let num_threads = 10usize;
        let increments_per_thread = 1000usize;

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
            handle
                .join()
                .map_err(|err| format!("worker thread panicked: {err:?}"))?;
        }

        let expected_threads = u64::try_from(num_threads).unwrap_or(u64::MAX);
        let expected_increments = u64::try_from(increments_per_thread).unwrap_or(u64::MAX);
        ensure_eq(
            &counter.get(),
            &expected_threads.saturating_mul(expected_increments),
            "counter value",
        )?;

        Ok(())
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

        // 50th percentile should be around 0.01 (between 0.001 and 0.1 range)
        assert!(quantiles[0] >= 0.001 && quantiles[0] <= 1.0);
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
