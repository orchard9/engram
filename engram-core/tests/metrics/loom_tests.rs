//! Loom-based verification of lock-free correctness
//!
//! These tests use the loom library to explore all possible thread interleavings
//! and verify that our lock-free data structures maintain correctness under
//! concurrent access.
//!
//! To run these tests:
//! ```bash
//! RUSTFLAGS="--cfg loom" cargo test --lib --release metrics::loom_tests
//! ```

#![cfg(all(test, loom, feature = "monitoring"))]

use loom::sync::Arc;
use loom::thread;

#[test]
fn loom_concurrent_counter_increments() {
    use engram_core::metrics::lockfree::LockFreeCounter;

    loom::model(|| {
        let counter = Arc::new(LockFreeCounter::new());

        let handles: Vec<_> = (0..2)
            .map(|_| {
                let counter = Arc::clone(&counter);
                thread::spawn(move || {
                    counter.increment(1);
                })
            })
            .collect();

        for h in handles {
            h.join().unwrap();
        }

        // After 2 threads each increment by 1, total must be 2
        assert_eq!(counter.get(), 2);
    });
}

#[test]
fn loom_concurrent_histogram_records() {
    use engram_core::metrics::lockfree::LockFreeHistogram;

    loom::model(|| {
        let histogram = Arc::new(LockFreeHistogram::new());

        let handles: Vec<_> = (0..2)
            .map(|i| {
                let histogram = Arc::clone(&histogram);
                thread::spawn(move || {
                    histogram.record(f64::from(i) + 1.0);
                })
            })
            .collect();

        for h in handles {
            h.join().unwrap();
        }

        // After 2 threads each record 1 value, count must be 2
        assert_eq!(histogram.count(), 2);

        // Mean should be (1.0 + 2.0) / 2 = 1.5
        let mean = histogram.mean();
        assert!(
            (mean - 1.5).abs() < 0.01,
            "mean = {}, expected ~1.5",
            mean
        );
    });
}

#[test]
fn loom_concurrent_priming_type_counters() {
    use engram_core::metrics::cognitive_patterns::{CognitivePatternMetrics, PrimingType};

    loom::model(|| {
        let metrics = Arc::new(CognitivePatternMetrics::new());

        let handles: Vec<_> = (0..2)
            .map(|_| {
                let metrics = Arc::clone(&metrics);
                thread::spawn(move || {
                    metrics.record_priming(PrimingType::Semantic, 0.5);
                })
            })
            .collect();

        for h in handles {
            h.join().unwrap();
        }

        // After 2 threads each record 1 event, total must be 2
        assert_eq!(metrics.priming_events_total(), 2);
    });
}

#[test]
fn loom_concurrent_different_priming_types() {
    use engram_core::metrics::cognitive_patterns::{CognitivePatternMetrics, PrimingType};

    loom::model(|| {
        let metrics = Arc::new(CognitivePatternMetrics::new());

        let h1 = {
            let metrics = Arc::clone(&metrics);
            thread::spawn(move || {
                metrics.record_priming(PrimingType::Semantic, 0.5);
            })
        };

        let h2 = {
            let metrics = Arc::clone(&metrics);
            thread::spawn(move || {
                metrics.record_priming(PrimingType::Associative, 0.6);
            })
        };

        h1.join().unwrap();
        h2.join().unwrap();

        // Total should be 2
        assert_eq!(metrics.priming_events_total(), 2);
        // Each type counter should be 1
        assert_eq!(metrics.priming_type_count(PrimingType::Semantic), 1);
        assert_eq!(metrics.priming_type_count(PrimingType::Associative), 1);
    });
}

#[test]
fn loom_concurrent_interference_recording() {
    use engram_core::metrics::cognitive_patterns::{CognitivePatternMetrics, InterferenceType};

    loom::model(|| {
        let metrics = Arc::new(CognitivePatternMetrics::new());

        let handles: Vec<_> = (0..2)
            .map(|_| {
                let metrics = Arc::clone(&metrics);
                thread::spawn(move || {
                    metrics.record_interference(InterferenceType::Proactive, 0.7);
                })
            })
            .collect();

        for h in handles {
            h.join().unwrap();
        }

        // After 2 threads each record 1 interference event, total must be 2
        assert_eq!(metrics.interference_detections_total(), 2);
    });
}

#[test]
fn loom_concurrent_reconsolidation() {
    use engram_core::metrics::cognitive_patterns::CognitivePatternMetrics;

    loom::model(|| {
        let metrics = Arc::new(CognitivePatternMetrics::new());

        let h1 = {
            let metrics = Arc::clone(&metrics);
            thread::spawn(move || {
                metrics.record_reconsolidation(0.5); // hit
            })
        };

        let h2 = {
            let metrics = Arc::clone(&metrics);
            thread::spawn(move || {
                metrics.record_reconsolidation(1.5); // miss
            })
        };

        h1.join().unwrap();
        h2.join().unwrap();

        // Total events should be 2
        assert_eq!(metrics.reconsolidation_events_total(), 2);
        // Hit rate should be 0.5 (1 hit, 1 miss)
        let hit_rate = metrics.reconsolidation_window_hit_rate();
        assert!(
            (hit_rate - 0.5).abs() < 0.01,
            "hit_rate = {}, expected 0.5",
            hit_rate
        );
    });
}

#[test]
fn loom_concurrent_gauge_operations() {
    use engram_core::metrics::lockfree::LockFreeGauge;

    loom::model(|| {
        let gauge = Arc::new(LockFreeGauge::new());

        let h1 = {
            let gauge = Arc::clone(&gauge);
            thread::spawn(move || {
                gauge.set(100);
            })
        };

        let h2 = {
            let gauge = Arc::clone(&gauge);
            thread::spawn(move || {
                gauge.increment(50);
            })
        };

        h1.join().unwrap();
        h2.join().unwrap();

        // Final value depends on interleaving, but should be valid
        let value = gauge.get();
        assert!(value == 100 || value == 150, "gauge value should be 100 or 150, got {}", value);
    });
}

#[test]
fn loom_histogram_reset_during_recording() {
    use engram_core::metrics::lockfree::LockFreeHistogram;

    loom::model(|| {
        let histogram = Arc::new(LockFreeHistogram::new());

        let h1 = {
            let histogram = Arc::clone(&histogram);
            thread::spawn(move || {
                histogram.record(1.0);
            })
        };

        let h2 = {
            let histogram = Arc::clone(&histogram);
            thread::spawn(move || {
                histogram.reset();
            })
        };

        h1.join().unwrap();
        h2.join().unwrap();

        // Count should be either 0 (reset after record) or 1 (reset before record)
        let count = histogram.count();
        assert!(count == 0 || count == 1, "count should be 0 or 1, got {}", count);
    });
}
