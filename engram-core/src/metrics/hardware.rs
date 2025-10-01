//! Hardware performance metrics collection

use crossbeam_utils::CachePadded;
use std::convert::TryFrom;
use std::sync::atomic::{AtomicU64, Ordering};

/// Hardware performance metrics collector
pub struct HardwareMetrics {
    /// SIMD instruction metrics
    simd_instructions: CachePadded<AtomicU64>,
    simd_cycles: CachePadded<AtomicU64>,

    /// Cache performance metrics
    l1_hits: CachePadded<AtomicU64>,
    l1_misses: CachePadded<AtomicU64>,
    l2_hits: CachePadded<AtomicU64>,
    l2_misses: CachePadded<AtomicU64>,
    l3_hits: CachePadded<AtomicU64>,
    l3_misses: CachePadded<AtomicU64>,

    /// Memory bandwidth metrics
    memory_reads: CachePadded<AtomicU64>,
    memory_writes: CachePadded<AtomicU64>,

    /// Branch prediction metrics
    branch_instructions: CachePadded<AtomicU64>,
    branch_misses: CachePadded<AtomicU64>,
}

impl HardwareMetrics {
    /// Create an empty hardware metrics collector with zeroed counters.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            simd_instructions: CachePadded::new(AtomicU64::new(0)),
            simd_cycles: CachePadded::new(AtomicU64::new(0)),
            l1_hits: CachePadded::new(AtomicU64::new(0)),
            l1_misses: CachePadded::new(AtomicU64::new(0)),
            l2_hits: CachePadded::new(AtomicU64::new(0)),
            l2_misses: CachePadded::new(AtomicU64::new(0)),
            l3_hits: CachePadded::new(AtomicU64::new(0)),
            l3_misses: CachePadded::new(AtomicU64::new(0)),
            memory_reads: CachePadded::new(AtomicU64::new(0)),
            memory_writes: CachePadded::new(AtomicU64::new(0)),
            branch_instructions: CachePadded::new(AtomicU64::new(0)),
            branch_misses: CachePadded::new(AtomicU64::new(0)),
        }
    }

    /// Record a hardware metric
    pub fn record(&self, metric: &HardwareMetric) {
        match metric {
            HardwareMetric::SimdOperation {
                instructions,
                cycles,
            } => {
                self.simd_instructions
                    .fetch_add(*instructions, Ordering::Relaxed);
                self.simd_cycles.fetch_add(*cycles, Ordering::Relaxed);
            }
            HardwareMetric::CacheAccess {
                level,
                hits,
                misses,
            } => match level {
                CacheLevel::L1 => {
                    self.l1_hits.fetch_add(*hits, Ordering::Relaxed);
                    self.l1_misses.fetch_add(*misses, Ordering::Relaxed);
                }
                CacheLevel::L2 => {
                    self.l2_hits.fetch_add(*hits, Ordering::Relaxed);
                    self.l2_misses.fetch_add(*misses, Ordering::Relaxed);
                }
                CacheLevel::L3 => {
                    self.l3_hits.fetch_add(*hits, Ordering::Relaxed);
                    self.l3_misses.fetch_add(*misses, Ordering::Relaxed);
                }
            },
            HardwareMetric::MemoryBandwidth { reads, writes } => {
                self.memory_reads.fetch_add(*reads, Ordering::Relaxed);
                self.memory_writes.fetch_add(*writes, Ordering::Relaxed);
            }
            HardwareMetric::BranchPrediction {
                instructions,
                misses,
            } => {
                self.branch_instructions
                    .fetch_add(*instructions, Ordering::Relaxed);
                self.branch_misses.fetch_add(*misses, Ordering::Relaxed);
            }
        }
    }

    /// Get SIMD utilization percentage
    #[must_use]
    pub fn simd_utilization(&self) -> f64 {
        let instructions = Self::to_f64(self.simd_instructions.load(Ordering::Acquire));
        let cycles = Self::to_f64(self.simd_cycles.load(Ordering::Acquire));

        if cycles > 0.0 {
            (instructions / cycles) * 100.0
        } else {
            0.0
        }
    }

    /// Get cache hit ratio for a specific level
    #[must_use]
    pub fn cache_hit_ratio(&self, level: CacheLevel) -> f64 {
        let (hits, misses) = match level {
            CacheLevel::L1 => (
                self.l1_hits.load(Ordering::Acquire),
                self.l1_misses.load(Ordering::Acquire),
            ),
            CacheLevel::L2 => (
                self.l2_hits.load(Ordering::Acquire),
                self.l2_misses.load(Ordering::Acquire),
            ),
            CacheLevel::L3 => (
                self.l3_hits.load(Ordering::Acquire),
                self.l3_misses.load(Ordering::Acquire),
            ),
        };

        let total = hits + misses;
        if total == 0 {
            return 0.0;
        }

        let hits_f64 = Self::to_f64(hits);
        let total_f64 = Self::to_f64(total);

        hits_f64 / total_f64
    }

    /// Get branch prediction accuracy
    #[must_use]
    pub fn branch_prediction_accuracy(&self) -> f64 {
        let instructions = self.branch_instructions.load(Ordering::Acquire);
        if instructions == 0 {
            return 0.0;
        }

        let misses = self.branch_misses.load(Ordering::Acquire);

        let miss_ratio = Self::to_f64(misses) / Self::to_f64(instructions);

        1.0 - miss_ratio
    }
}

impl HardwareMetrics {
    fn to_f64(value: u64) -> f64 {
        let high_part = u32::try_from(value >> 32).unwrap_or(u32::MAX);
        let low_part = u32::try_from(value & 0xFFFF_FFFF).unwrap_or(u32::MAX);
        f64::from(high_part).mul_add(4_294_967_296.0, f64::from(low_part))
    }
}

/// Hardware metric types
#[derive(Debug, Clone)]
pub enum HardwareMetric {
    /// Record SIMD execution stats for a single observation window.
    SimdOperation {
        /// Total vectorized instructions executed during the window.
        instructions: u64,
        /// CPU cycles consumed by the recorded SIMD work.
        cycles: u64,
    },
    /// Capture cache access balance for the specified level.
    CacheAccess {
        /// Cache tier the counters apply to.
        level: CacheLevel,
        /// Number of cache hits observed.
        hits: u64,
        /// Number of cache misses observed.
        misses: u64,
    },
    /// Report aggregate memory bandwidth usage.
    MemoryBandwidth {
        /// Bytes read from memory.
        reads: u64,
        /// Bytes written to memory.
        writes: u64,
    },
    /// Track branch prediction effectiveness.
    BranchPrediction {
        /// Branch instructions executed.
        instructions: u64,
        /// Branch mispredictions encountered.
        misses: u64,
    },
}

/// CPU cache levels
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CacheLevel {
    /// Level 1 data cache closest to the core.
    L1,
    /// Level 2 unified cache shared across a core cluster.
    L2,
    /// Level 3 cache shared by the socket.
    L3,
}

/// SIMD operation types
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SimdOperation {
    /// Vector cosine similarity workload.
    CosineSimilarity,
    /// Vector dot product calculation.
    DotProduct,
    /// Euclidean norm computation across a vector.
    L2Norm,
    /// Batched SIMD task covering multiple payloads.
    BatchOperation,
}

impl Default for HardwareMetrics {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_close(actual: f64, expected: f64) {
        let diff = (actual - expected).abs();
        assert!(
            diff <= f64::EPSILON,
            "expected {expected}, got {actual} (diff {diff})"
        );
    }

    #[test]
    fn test_simd_utilization() {
        let metrics = HardwareMetrics::new();

        metrics.record(&HardwareMetric::SimdOperation {
            instructions: 1000,
            cycles: 500,
        });

        let utilization = metrics.simd_utilization();
        assert_close(utilization, 200.0); // 1000/500 * 100
    }

    #[test]
    fn test_cache_hit_ratio() {
        let metrics = HardwareMetrics::new();

        metrics.record(&HardwareMetric::CacheAccess {
            level: CacheLevel::L1,
            hits: 900,
            misses: 100,
        });

        let hit_ratio = metrics.cache_hit_ratio(CacheLevel::L1);
        assert_close(hit_ratio, 0.9); // 900/1000
    }
}
