//! Cognitive pattern metrics with zero-overhead when monitoring disabled
//!
//! This module provides lock-free atomic metrics for tracking cognitive patterns
//! like priming, interference, reconsolidation, and false memory generation.
//!
//! # Zero-Overhead Abstraction
//!
//! When the `monitoring` feature is disabled, the `CognitivePatternMetrics` struct
//! is zero-sized (just `PhantomData`) and all methods compile to nothing. The compiler
//! completely eliminates the code, providing truly zero overhead.
//!
//! When `monitoring` is enabled, metrics use lock-free atomic operations with
//! cache-line padding to prevent false sharing, achieving <1% overhead.

use crate::metrics::lockfree::LockFreeHistogram;
use crossbeam_utils::CachePadded;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

/// Zero-overhead cognitive pattern metrics
///
/// When `monitoring` feature is disabled, this struct is zero-sized and all
/// methods are no-ops that compile away entirely.
///
/// When `monitoring` feature is enabled, provides lock-free atomic metrics
/// with <1% overhead.
#[cfg(feature = "monitoring")]
pub struct CognitivePatternMetrics {
    // Direct struct fields - NO Arc wrapper
    // This eliminates pointer indirection overhead (critical fix #1)

    // Priming metrics
    priming_events_total: CachePadded<AtomicU64>,
    priming_strength_histogram: LockFreeHistogram,
    priming_type_counters: [CachePadded<AtomicU64>; 3], // semantic, associative, repetition

    // Interference metrics
    interference_detections_total: CachePadded<AtomicU64>,
    proactive_interference_strength: LockFreeHistogram,
    retroactive_interference_strength: LockFreeHistogram,
    fan_effect_magnitude: LockFreeHistogram,

    // Reconsolidation metrics
    reconsolidation_events_total: CachePadded<AtomicU64>,
    reconsolidation_modifications: CachePadded<AtomicU64>,
    reconsolidation_window_hits: CachePadded<AtomicU64>,
    reconsolidation_window_misses: CachePadded<AtomicU64>,
    reconsolidation_rejections: [CachePadded<AtomicU64>; 4], // outside_window, too_soon, conflicting, other

    // False memory metrics
    false_memory_generations: CachePadded<AtomicU64>,
    drm_critical_lure_recalls: CachePadded<AtomicU64>,
    drm_list_item_recalls: CachePadded<AtomicU64>,

    // Spacing effect metrics
    massed_practice_events: CachePadded<AtomicU64>,
    distributed_practice_events: CachePadded<AtomicU64>,
    retention_improvement_histogram: LockFreeHistogram,
}

/// When monitoring disabled, use zero-sized type with PhantomData marker
#[cfg(not(feature = "monitoring"))]
pub struct CognitivePatternMetrics {
    _phantom: core::marker::PhantomData<()>,
}

impl CognitivePatternMetrics {
    /// Create new metrics instance
    #[cfg(feature = "monitoring")]
    #[must_use]
    pub fn new() -> Self {
        Self {
            priming_events_total: CachePadded::new(AtomicU64::new(0)),
            priming_strength_histogram: LockFreeHistogram::new(),
            priming_type_counters: [
                CachePadded::new(AtomicU64::new(0)),
                CachePadded::new(AtomicU64::new(0)),
                CachePadded::new(AtomicU64::new(0)),
            ],
            interference_detections_total: CachePadded::new(AtomicU64::new(0)),
            proactive_interference_strength: LockFreeHistogram::new(),
            retroactive_interference_strength: LockFreeHistogram::new(),
            fan_effect_magnitude: LockFreeHistogram::new(),
            reconsolidation_events_total: CachePadded::new(AtomicU64::new(0)),
            reconsolidation_modifications: CachePadded::new(AtomicU64::new(0)),
            reconsolidation_window_hits: CachePadded::new(AtomicU64::new(0)),
            reconsolidation_window_misses: CachePadded::new(AtomicU64::new(0)),
            reconsolidation_rejections: [
                CachePadded::new(AtomicU64::new(0)),
                CachePadded::new(AtomicU64::new(0)),
                CachePadded::new(AtomicU64::new(0)),
                CachePadded::new(AtomicU64::new(0)),
            ],
            false_memory_generations: CachePadded::new(AtomicU64::new(0)),
            drm_critical_lure_recalls: CachePadded::new(AtomicU64::new(0)),
            drm_list_item_recalls: CachePadded::new(AtomicU64::new(0)),
            massed_practice_events: CachePadded::new(AtomicU64::new(0)),
            distributed_practice_events: CachePadded::new(AtomicU64::new(0)),
            retention_improvement_histogram: LockFreeHistogram::new(),
        }
    }

    #[cfg(not(feature = "monitoring"))]
    #[must_use]
    pub const fn new() -> Self {
        Self {
            _phantom: core::marker::PhantomData,
        }
    }

    /// Record priming event with zero overhead when monitoring disabled
    ///
    /// # Performance
    /// - Monitoring disabled: 0ns (function is empty, completely optimized away)
    /// - Monitoring enabled: ~25ns (hot path, L1 cached)
    ///
    /// # Implementation note
    /// Using separate #[cfg] blocks instead of single block with unused variable
    /// suppression. This is more verbose but ensures compiler can optimize each
    /// variant independently.
    ///
    /// # Inlining
    /// `inline(always)` is intentional for zero-overhead abstraction. When monitoring
    /// is disabled, the compiler must completely eliminate these calls.
    #[inline(always)]
    #[allow(clippy::inline_always)]
    pub fn record_priming(&self, priming_type: PrimingType, strength: f32) {
        #[cfg(feature = "monitoring")]
        {
            self.priming_events_total.fetch_add(1, Ordering::Relaxed);
            self.priming_strength_histogram.record(f64::from(strength));

            let idx = priming_type as usize;
            self.priming_type_counters[idx].fetch_add(1, Ordering::Relaxed);
        }

        // When monitoring disabled, function body is empty - compiles to nothing
        #[cfg(not(feature = "monitoring"))]
        {
            let _ = (priming_type, strength); // Suppress unused warnings
        }
    }

    /// Record interference event
    ///
    /// # Performance
    /// - Monitoring disabled: 0ns
    /// - Monitoring enabled: ~80ns (includes histogram binary search)
    #[inline(always)]
    #[allow(clippy::inline_always)]
    pub fn record_interference(&self, interference_type: InterferenceType, magnitude: f32) {
        #[cfg(feature = "monitoring")]
        {
            self.interference_detections_total
                .fetch_add(1, Ordering::Relaxed);

            match interference_type {
                InterferenceType::Proactive => {
                    self.proactive_interference_strength
                        .record(f64::from(magnitude));
                }
                InterferenceType::Retroactive => {
                    self.retroactive_interference_strength
                        .record(f64::from(magnitude));
                }
                InterferenceType::Fan => {
                    self.fan_effect_magnitude.record(f64::from(magnitude));
                }
            }
        }

        #[cfg(not(feature = "monitoring"))]
        {
            let _ = (interference_type, magnitude);
        }
    }

    /// Record reconsolidation event
    ///
    /// # Parameters
    /// - `window_position`: 0.0-1.0 indicates within reconsolidation window,
    ///   values outside this range indicate miss
    ///
    /// # Performance
    /// - Monitoring disabled: 0ns
    /// - Monitoring enabled: ~25ns
    #[inline(always)]
    #[allow(clippy::inline_always)]
    pub fn record_reconsolidation(&self, window_position: f32) {
        #[cfg(feature = "monitoring")]
        {
            self.reconsolidation_events_total
                .fetch_add(1, Ordering::Relaxed);

            if (0.0..=1.0).contains(&window_position) {
                self.reconsolidation_window_hits
                    .fetch_add(1, Ordering::Relaxed);
            } else {
                self.reconsolidation_window_misses
                    .fetch_add(1, Ordering::Relaxed);
            }
        }

        #[cfg(not(feature = "monitoring"))]
        {
            let _ = window_position;
        }
    }

    /// Record memory modification during reconsolidation
    #[inline(always)]
    #[allow(clippy::inline_always)]
    pub fn record_reconsolidation_modification(&self) {
        #[cfg(feature = "monitoring")]
        {
            self.reconsolidation_modifications
                .fetch_add(1, Ordering::Relaxed);
        }
    }

    /// Record reconsolidation rejection with structured reason
    ///
    /// # Performance
    /// - Monitoring disabled: 0ns
    /// - Monitoring enabled: ~25ns
    ///
    /// # Parameters
    /// - `reason`: The specific reason for rejection, used for detailed tracking
    #[inline(always)]
    #[allow(clippy::inline_always)]
    pub fn record_reconsolidation_rejection(&self, reason: RejectionReason) {
        #[cfg(feature = "monitoring")]
        {
            // Count rejections as window misses
            self.reconsolidation_window_misses
                .fetch_add(1, Ordering::Relaxed);

            // Track structured rejection reasons for detailed analysis
            let idx = reason as usize;
            self.reconsolidation_rejections[idx].fetch_add(1, Ordering::Relaxed);
        }

        #[cfg(not(feature = "monitoring"))]
        {
            let _ = reason;
        }
    }

    /// Record false memory generation
    #[inline(always)]
    #[allow(clippy::inline_always)]
    pub fn record_false_memory(&self) {
        #[cfg(feature = "monitoring")]
        {
            self.false_memory_generations
                .fetch_add(1, Ordering::Relaxed);
        }
    }

    /// Record DRM critical lure recall (false memory paradigm)
    #[inline(always)]
    #[allow(clippy::inline_always)]
    pub fn record_drm_critical_lure(&self) {
        #[cfg(feature = "monitoring")]
        {
            self.drm_critical_lure_recalls
                .fetch_add(1, Ordering::Relaxed);
        }
    }

    /// Record DRM list item recall (true memory)
    #[inline(always)]
    #[allow(clippy::inline_always)]
    pub fn record_drm_list_item(&self) {
        #[cfg(feature = "monitoring")]
        {
            self.drm_list_item_recalls.fetch_add(1, Ordering::Relaxed);
        }
    }

    /// Record massed practice event (immediate repetition)
    #[inline(always)]
    #[allow(clippy::inline_always)]
    pub fn record_massed_practice(&self) {
        #[cfg(feature = "monitoring")]
        {
            self.massed_practice_events.fetch_add(1, Ordering::Relaxed);
        }
    }

    /// Record distributed practice event (spaced repetition)
    #[inline(always)]
    #[allow(clippy::inline_always)]
    pub fn record_distributed_practice(&self) {
        #[cfg(feature = "monitoring")]
        {
            self.distributed_practice_events
                .fetch_add(1, Ordering::Relaxed);
        }
    }

    /// Record retention improvement from spacing effect
    ///
    /// # Parameters
    /// - `improvement`: Ratio of retention with spacing vs. without (e.g., 1.5 = 50% improvement)
    #[inline(always)]
    #[allow(clippy::inline_always)]
    pub fn record_retention_improvement(&self, improvement: f32) {
        #[cfg(feature = "monitoring")]
        {
            self.retention_improvement_histogram
                .record(f64::from(improvement));
        }

        #[cfg(not(feature = "monitoring"))]
        {
            let _ = improvement;
        }
    }

    // ========== Query Methods ==========

    /// Get priming event total
    #[cfg(feature = "monitoring")]
    #[must_use]
    pub fn priming_events_total(&self) -> u64 {
        // CORRECTED: Use Relaxed ordering for counter reads
        // Acquire is unnecessarily strong - we just want latest visible value
        self.priming_events_total.load(Ordering::Relaxed)
    }

    #[cfg(not(feature = "monitoring"))]
    #[must_use]
    pub const fn priming_events_total(&self) -> u64 {
        0
    }

    /// Get priming type count for specific type
    #[cfg(feature = "monitoring")]
    #[must_use]
    pub fn priming_type_count(&self, priming_type: PrimingType) -> u64 {
        self.priming_type_counters[priming_type as usize].load(Ordering::Relaxed)
    }

    #[cfg(not(feature = "monitoring"))]
    #[must_use]
    pub const fn priming_type_count(&self, _priming_type: PrimingType) -> u64 {
        0
    }

    /// Get mean priming strength
    #[cfg(feature = "monitoring")]
    #[must_use]
    pub fn priming_mean_strength(&self) -> f64 {
        self.priming_strength_histogram.mean()
    }

    #[cfg(not(feature = "monitoring"))]
    #[must_use]
    pub const fn priming_mean_strength(&self) -> f64 {
        0.0
    }

    /// Get interference detection total
    #[cfg(feature = "monitoring")]
    #[must_use]
    pub fn interference_detections_total(&self) -> u64 {
        self.interference_detections_total.load(Ordering::Relaxed)
    }

    #[cfg(not(feature = "monitoring"))]
    #[must_use]
    pub const fn interference_detections_total(&self) -> u64 {
        0
    }

    /// Get reconsolidation events total
    #[cfg(feature = "monitoring")]
    #[must_use]
    pub fn reconsolidation_events_total(&self) -> u64 {
        self.reconsolidation_events_total.load(Ordering::Relaxed)
    }

    #[cfg(not(feature = "monitoring"))]
    #[must_use]
    pub const fn reconsolidation_events_total(&self) -> u64 {
        0
    }

    /// Get reconsolidation window hit rate
    #[cfg(feature = "monitoring")]
    #[must_use]
    pub fn reconsolidation_window_hit_rate(&self) -> f64 {
        let hits = self.reconsolidation_window_hits.load(Ordering::Relaxed);
        let misses = self.reconsolidation_window_misses.load(Ordering::Relaxed);
        let total = hits + misses;

        if total == 0 {
            0.0
        } else {
            f64::from(u32::try_from(hits).unwrap_or(u32::MAX))
                / f64::from(u32::try_from(total).unwrap_or(u32::MAX))
        }
    }

    #[cfg(not(feature = "monitoring"))]
    #[must_use]
    pub const fn reconsolidation_window_hit_rate(&self) -> f64 {
        0.0
    }

    /// Get reconsolidation rejection count for specific reason
    #[cfg(feature = "monitoring")]
    #[must_use]
    pub fn reconsolidation_rejection_count(&self, reason: RejectionReason) -> u64 {
        self.reconsolidation_rejections[reason as usize].load(Ordering::Relaxed)
    }

    #[cfg(not(feature = "monitoring"))]
    #[must_use]
    pub const fn reconsolidation_rejection_count(&self, _reason: RejectionReason) -> u64 {
        0
    }

    /// Get false memory generation count
    #[cfg(feature = "monitoring")]
    #[must_use]
    pub fn false_memory_generations(&self) -> u64 {
        self.false_memory_generations.load(Ordering::Relaxed)
    }

    #[cfg(not(feature = "monitoring"))]
    #[must_use]
    pub const fn false_memory_generations(&self) -> u64 {
        0
    }

    /// Get DRM false alarm rate (critical lures / total)
    #[cfg(feature = "monitoring")]
    #[must_use]
    pub fn drm_false_alarm_rate(&self) -> f64 {
        let lures = self.drm_critical_lure_recalls.load(Ordering::Relaxed);
        let items = self.drm_list_item_recalls.load(Ordering::Relaxed);
        let total = lures + items;

        if total == 0 {
            0.0
        } else {
            f64::from(u32::try_from(lures).unwrap_or(u32::MAX))
                / f64::from(u32::try_from(total).unwrap_or(u32::MAX))
        }
    }

    #[cfg(not(feature = "monitoring"))]
    #[must_use]
    pub const fn drm_false_alarm_rate(&self) -> f64 {
        0.0
    }

    /// Get spacing effect ratio (distributed / massed)
    #[cfg(feature = "monitoring")]
    #[must_use]
    pub fn spacing_ratio(&self) -> f64 {
        let distributed = self.distributed_practice_events.load(Ordering::Relaxed);
        let massed = self.massed_practice_events.load(Ordering::Relaxed);

        if massed == 0 {
            0.0
        } else {
            f64::from(u32::try_from(distributed).unwrap_or(u32::MAX))
                / f64::from(u32::try_from(massed).unwrap_or(u32::MAX))
        }
    }

    #[cfg(not(feature = "monitoring"))]
    #[must_use]
    pub const fn spacing_ratio(&self) -> f64 {
        0.0
    }

    /// Reset all metrics to zero
    #[cfg(feature = "monitoring")]
    pub fn reset(&self) {
        self.priming_events_total.store(0, Ordering::Release);
        self.priming_strength_histogram.reset();
        for counter in &self.priming_type_counters {
            counter.store(0, Ordering::Release);
        }
        self.interference_detections_total
            .store(0, Ordering::Release);
        self.proactive_interference_strength.reset();
        self.retroactive_interference_strength.reset();
        self.fan_effect_magnitude.reset();
        self.reconsolidation_events_total
            .store(0, Ordering::Release);
        self.reconsolidation_modifications
            .store(0, Ordering::Release);
        self.reconsolidation_window_hits.store(0, Ordering::Release);
        self.reconsolidation_window_misses
            .store(0, Ordering::Release);
        for counter in &self.reconsolidation_rejections {
            counter.store(0, Ordering::Release);
        }
        self.false_memory_generations.store(0, Ordering::Release);
        self.drm_critical_lure_recalls.store(0, Ordering::Release);
        self.drm_list_item_recalls.store(0, Ordering::Release);
        self.massed_practice_events.store(0, Ordering::Release);
        self.distributed_practice_events.store(0, Ordering::Release);
        self.retention_improvement_histogram.reset();
    }

    #[cfg(not(feature = "monitoring"))]
    pub fn reset(&self) {
        // No-op when monitoring disabled
    }
}

/// Type of priming effect
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum PrimingType {
    /// Semantic priming (related meaning)
    Semantic = 0,
    /// Associative priming (learned association)
    Associative = 1,
    /// Repetition priming (same stimulus)
    Repetition = 2,
}

/// Type of interference effect
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InterferenceType {
    /// Proactive interference (old interferes with new)
    Proactive,
    /// Retroactive interference (new interferes with old)
    Retroactive,
    /// Fan effect (multiple associations)
    Fan,
}

/// Reason for reconsolidation rejection
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum RejectionReason {
    /// Memory access attempted outside reconsolidation window
    OutsideWindow = 0,
    /// Too soon after previous reconsolidation attempt
    TooSoon = 1,
    /// Conflicting concurrent modification detected
    Conflicting = 2,
    /// Other unspecified reason
    Other = 3,
}

#[cfg(feature = "monitoring")]
impl Default for CognitivePatternMetrics {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(not(feature = "monitoring"))]
impl Default for CognitivePatternMetrics {
    fn default() -> Self {
        Self::new()
    }
}

/// Global cognitive patterns metrics instance (lazy-initialized)
static COGNITIVE_PATTERNS: std::sync::OnceLock<Arc<CognitivePatternMetrics>> =
    std::sync::OnceLock::new();

/// Get the global cognitive patterns metrics registry
#[must_use]
pub fn cognitive_patterns() -> Option<Arc<CognitivePatternMetrics>> {
    COGNITIVE_PATTERNS.get().cloned()
}

/// Initialize the global cognitive patterns metrics registry
pub fn init_cognitive_patterns() -> Arc<CognitivePatternMetrics> {
    Arc::clone(COGNITIVE_PATTERNS.get_or_init(|| Arc::new(CognitivePatternMetrics::new())))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[cfg(feature = "monitoring")]
    fn test_priming_metrics() {
        let metrics = CognitivePatternMetrics::new();

        metrics.record_priming(PrimingType::Semantic, 0.75);
        metrics.record_priming(PrimingType::Associative, 0.5);
        metrics.record_priming(PrimingType::Semantic, 0.9);

        assert_eq!(metrics.priming_events_total(), 3);
        assert_eq!(metrics.priming_type_count(PrimingType::Semantic), 2);
        assert_eq!(metrics.priming_type_count(PrimingType::Associative), 1);
        assert_eq!(metrics.priming_type_count(PrimingType::Repetition), 0);

        let mean = metrics.priming_mean_strength();
        assert!(mean > 0.6 && mean < 0.8, "mean should be ~0.716");
    }

    #[test]
    #[cfg(feature = "monitoring")]
    fn test_interference_metrics() {
        let metrics = CognitivePatternMetrics::new();

        metrics.record_interference(InterferenceType::Proactive, 0.6);
        metrics.record_interference(InterferenceType::Retroactive, 0.4);
        metrics.record_interference(InterferenceType::Fan, 0.8);

        assert_eq!(metrics.interference_detections_total(), 3);
    }

    #[test]
    #[cfg(feature = "monitoring")]
    fn test_reconsolidation_window() {
        let metrics = CognitivePatternMetrics::new();

        metrics.record_reconsolidation(0.5); // hit
        metrics.record_reconsolidation(0.8); // hit
        metrics.record_reconsolidation(1.2); // miss
        metrics.record_reconsolidation(0.1); // hit

        assert_eq!(metrics.reconsolidation_events_total(), 4);
        let hit_rate = metrics.reconsolidation_window_hit_rate();
        assert!((hit_rate - 0.75).abs() < 0.01, "hit rate should be 0.75");
    }

    #[test]
    #[cfg(feature = "monitoring")]
    fn test_drm_false_alarm_rate() {
        let metrics = CognitivePatternMetrics::new();

        metrics.record_drm_list_item(); // true memory
        metrics.record_drm_list_item(); // true memory
        metrics.record_drm_critical_lure(); // false memory
        metrics.record_drm_list_item(); // true memory

        let far = metrics.drm_false_alarm_rate();
        assert!((far - 0.25).abs() < 0.01, "FAR should be 0.25");
    }

    #[test]
    #[cfg(feature = "monitoring")]
    fn test_spacing_effect() {
        let metrics = CognitivePatternMetrics::new();

        metrics.record_massed_practice();
        metrics.record_distributed_practice();
        metrics.record_distributed_practice();
        metrics.record_distributed_practice();

        let ratio = metrics.spacing_ratio();
        assert!((ratio - 3.0).abs() < 0.01, "spacing ratio should be 3.0");
    }

    #[test]
    #[cfg(feature = "monitoring")]
    fn test_reset() {
        let metrics = CognitivePatternMetrics::new();

        metrics.record_priming(PrimingType::Semantic, 0.5);
        metrics.record_interference(InterferenceType::Proactive, 0.6);
        metrics.reset();

        assert_eq!(metrics.priming_events_total(), 0);
        assert_eq!(metrics.interference_detections_total(), 0);
    }

    #[test]
    #[cfg(not(feature = "monitoring"))]
    fn test_zero_size_when_disabled() {
        use std::mem::size_of;
        assert_eq!(
            size_of::<CognitivePatternMetrics>(),
            0,
            "CognitivePatternMetrics should be zero-sized when monitoring disabled"
        );
    }

    #[test]
    #[cfg(not(feature = "monitoring"))]
    fn test_methods_compile_when_disabled() {
        let metrics = CognitivePatternMetrics::new();
        metrics.record_priming(PrimingType::Semantic, 0.5);
        metrics.record_interference(InterferenceType::Proactive, 0.6);
        metrics.record_reconsolidation(0.5);

        // All query methods should return 0
        assert_eq!(metrics.priming_events_total(), 0);
        assert_eq!(metrics.interference_detections_total(), 0);
        assert_eq!(metrics.reconsolidation_events_total(), 0);
    }
}
