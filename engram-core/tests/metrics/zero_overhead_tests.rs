//! Verification that metrics have truly zero overhead when disabled
//!
//! These tests validate the zero-cost abstraction guarantees:
//! 1. When monitoring disabled, struct is zero-sized
//! 2. Methods compile but do nothing
//! 3. When monitoring enabled, overhead is <1%

#[cfg(not(feature = "monitoring"))]
#[test]
fn verify_zero_size_when_disabled() {
    use engram_core::metrics::cognitive_patterns::CognitivePatternMetrics;
    use std::mem::size_of;

    // When monitoring disabled, struct should be zero-sized (just PhantomData)
    assert_eq!(
        size_of::<CognitivePatternMetrics>(),
        0,
        "CognitivePatternMetrics should be zero-sized when monitoring disabled"
    );
}

#[cfg(not(feature = "monitoring"))]
#[test]
fn verify_methods_are_noops() {
    use engram_core::metrics::cognitive_patterns::{
        CognitivePatternMetrics, InterferenceType, PrimingType,
    };

    // Methods should compile and do nothing
    let metrics = CognitivePatternMetrics::new();
    metrics.record_priming(PrimingType::Semantic, 0.5);
    metrics.record_interference(InterferenceType::Proactive, 0.6);
    metrics.record_reconsolidation(0.5);
    metrics.record_false_memory();
    metrics.record_massed_practice();
    metrics.record_distributed_practice();

    // All query methods should return 0
    assert_eq!(metrics.priming_events_total(), 0);
    assert_eq!(metrics.interference_detections_total(), 0);
    assert_eq!(metrics.reconsolidation_events_total(), 0);
    assert_eq!(metrics.false_memory_generations(), 0);

    // If this compiles and runs, conditional compilation works
    // No assertions needed - successful compilation is the test
}

#[cfg(not(feature = "monitoring"))]
#[test]
fn verify_default_trait() {
    use engram_core::metrics::cognitive_patterns::CognitivePatternMetrics;

    // Default should work when monitoring disabled
    let metrics = CognitivePatternMetrics::default();
    assert_eq!(metrics.priming_events_total(), 0);
}

#[cfg(all(feature = "monitoring", test))]
#[test]
fn verify_overhead_under_one_percent() {
    use engram_core::metrics::cognitive_patterns::{CognitivePatternMetrics, PrimingType};
    use std::time::Instant;

    const ITERATIONS: usize = 1_000_000;

    // Baseline: operation without metrics
    fn baseline_op() -> f32 {
        let mut sum = 0.0f32;
        for i in 0..10 {
            sum += (i as f32).sin();
        }
        sum
    }

    // Baseline measurement
    let start = Instant::now();
    for _ in 0..ITERATIONS {
        std::hint::black_box(baseline_op());
    }
    let baseline_duration = start.elapsed();

    // With metrics
    let metrics = CognitivePatternMetrics::new();
    let start = Instant::now();
    for i in 0..ITERATIONS {
        std::hint::black_box(baseline_op());
        metrics.record_priming(PrimingType::Semantic, (i % 100) as f32 / 100.0);
    }
    let instrumented_duration = start.elapsed();

    let overhead = (instrumented_duration.as_nanos() as f64 - baseline_duration.as_nanos() as f64)
        / baseline_duration.as_nanos() as f64;

    println!("Baseline: {:?}", baseline_duration);
    println!("Instrumented: {:?}", instrumented_duration);
    println!("Overhead: {:.2}%", overhead * 100.0);

    assert!(
        overhead < 0.01,
        "Metrics overhead {:.2}% exceeds 1% threshold",
        overhead * 100.0
    );
}

#[cfg(all(feature = "monitoring", test))]
#[test]
fn verify_concurrent_correctness() {
    use engram_core::metrics::cognitive_patterns::{CognitivePatternMetrics, PrimingType};
    use std::sync::Arc;
    use std::thread;

    let metrics = Arc::new(CognitivePatternMetrics::new());
    let num_threads = 10;
    let increments_per_thread = 1000;

    let handles: Vec<_> = (0..num_threads)
        .map(|_| {
            let metrics = Arc::clone(&metrics);
            thread::spawn(move || {
                for _ in 0..increments_per_thread {
                    metrics.record_priming(PrimingType::Semantic, 0.5);
                }
            })
        })
        .collect();

    for handle in handles {
        handle.join().expect("thread panicked");
    }

    // After 10 threads each recording 1000 events, total should be 10,000
    assert_eq!(
        metrics.priming_events_total(),
        num_threads * increments_per_thread
    );
}

#[cfg(all(feature = "monitoring", test))]
#[test]
fn verify_histogram_mean_calculation() {
    use engram_core::metrics::cognitive_patterns::{CognitivePatternMetrics, PrimingType};

    let metrics = CognitivePatternMetrics::new();

    // Record known values
    metrics.record_priming(PrimingType::Semantic, 0.2);
    metrics.record_priming(PrimingType::Semantic, 0.4);
    metrics.record_priming(PrimingType::Semantic, 0.6);
    metrics.record_priming(PrimingType::Semantic, 0.8);

    let mean = metrics.priming_mean_strength();
    // Mean should be (0.2 + 0.4 + 0.6 + 0.8) / 4 = 0.5
    assert!(
        (mean - 0.5).abs() < 0.01,
        "mean should be ~0.5, got {}",
        mean
    );
}

#[cfg(all(feature = "monitoring", test))]
#[test]
fn verify_reset_clears_all_state() {
    use engram_core::metrics::cognitive_patterns::{
        CognitivePatternMetrics, InterferenceType, PrimingType,
    };

    let metrics = CognitivePatternMetrics::new();

    // Record various events
    metrics.record_priming(PrimingType::Semantic, 0.5);
    metrics.record_priming(PrimingType::Associative, 0.6);
    metrics.record_interference(InterferenceType::Proactive, 0.7);
    metrics.record_reconsolidation(0.5);
    metrics.record_false_memory();

    // Verify state is recorded
    assert!(metrics.priming_events_total() > 0);
    assert!(metrics.interference_detections_total() > 0);
    assert!(metrics.reconsolidation_events_total() > 0);
    assert!(metrics.false_memory_generations() > 0);

    // Reset
    metrics.reset();

    // Verify all state is cleared
    assert_eq!(metrics.priming_events_total(), 0);
    assert_eq!(metrics.interference_detections_total(), 0);
    assert_eq!(metrics.reconsolidation_events_total(), 0);
    assert_eq!(metrics.false_memory_generations(), 0);
    assert_eq!(metrics.priming_mean_strength(), 0.0);
}
