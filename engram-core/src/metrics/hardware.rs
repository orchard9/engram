//! Hardware performance metrics collection

use std::sync::atomic::{AtomicU64, Ordering};
use crossbeam_utils::CachePadded;
use atomic_float::AtomicF32;

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
    pub fn new() -> Self {
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
    #[inline(always)]
    pub fn record(&self, metric: HardwareMetric) {
        match metric {
            HardwareMetric::SimdOperation { instructions, cycles } => {
                self.simd_instructions.fetch_add(instructions, Ordering::Relaxed);
                self.simd_cycles.fetch_add(cycles, Ordering::Relaxed);
            }
            HardwareMetric::CacheAccess { level, hits, misses } => {
                match level {
                    CacheLevel::L1 => {
                        self.l1_hits.fetch_add(hits, Ordering::Relaxed);
                        self.l1_misses.fetch_add(misses, Ordering::Relaxed);
                    }
                    CacheLevel::L2 => {
                        self.l2_hits.fetch_add(hits, Ordering::Relaxed);
                        self.l2_misses.fetch_add(misses, Ordering::Relaxed);
                    }
                    CacheLevel::L3 => {
                        self.l3_hits.fetch_add(hits, Ordering::Relaxed);
                        self.l3_misses.fetch_add(misses, Ordering::Relaxed);
                    }
                }
            }
            HardwareMetric::MemoryBandwidth { reads, writes } => {
                self.memory_reads.fetch_add(reads, Ordering::Relaxed);
                self.memory_writes.fetch_add(writes, Ordering::Relaxed);
            }
            HardwareMetric::BranchPrediction { instructions, misses } => {
                self.branch_instructions.fetch_add(instructions, Ordering::Relaxed);
                self.branch_misses.fetch_add(misses, Ordering::Relaxed);
            }
        }
    }
    
    /// Get SIMD utilization percentage
    pub fn simd_utilization(&self) -> f32 {
        let instructions = self.simd_instructions.load(Ordering::Acquire) as f32;
        let cycles = self.simd_cycles.load(Ordering::Acquire) as f32;
        
        if cycles > 0.0 {
            (instructions / cycles) * 100.0
        } else {
            0.0
        }
    }
    
    /// Get cache hit ratio for a specific level
    pub fn cache_hit_ratio(&self, level: CacheLevel) -> f32 {
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
        if total > 0 {
            hits as f32 / total as f32
        } else {
            0.0
        }
    }
    
    /// Get branch prediction accuracy
    pub fn branch_prediction_accuracy(&self) -> f32 {
        let instructions = self.branch_instructions.load(Ordering::Acquire);
        let misses = self.branch_misses.load(Ordering::Acquire);
        
        if instructions > 0 {
            1.0 - (misses as f32 / instructions as f32)
        } else {
            0.0
        }
    }
}

/// Hardware metric types
#[derive(Debug, Clone)]
pub enum HardwareMetric {
    SimdOperation {
        instructions: u64,
        cycles: u64,
    },
    CacheAccess {
        level: CacheLevel,
        hits: u64,
        misses: u64,
    },
    MemoryBandwidth {
        reads: u64,
        writes: u64,
    },
    BranchPrediction {
        instructions: u64,
        misses: u64,
    },
}

/// CPU cache levels
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CacheLevel {
    L1,
    L2,
    L3,
}

/// SIMD operation types
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SimdOperation {
    CosineSimilarity,
    DotProduct,
    L2Norm,
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
    
    #[test]
    fn test_simd_utilization() {
        let metrics = HardwareMetrics::new();
        
        metrics.record(HardwareMetric::SimdOperation {
            instructions: 1000,
            cycles: 500,
        });
        
        let utilization = metrics.simd_utilization();
        assert_eq!(utilization, 200.0); // 1000/500 * 100
    }
    
    #[test]
    fn test_cache_hit_ratio() {
        let metrics = HardwareMetrics::new();
        
        metrics.record(HardwareMetric::CacheAccess {
            level: CacheLevel::L1,
            hits: 900,
            misses: 100,
        });
        
        let hit_ratio = metrics.cache_hit_ratio(CacheLevel::L1);
        assert_eq!(hit_ratio, 0.9); // 900/1000
    }
}