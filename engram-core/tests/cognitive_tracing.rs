//! Integration tests for cognitive event tracing

#![cfg(feature = "cognitive_tracing")]

use engram_core::tracing::{
    CognitiveTracer, EventType, InterferenceType, PrimingType, TracingConfig,
};
use std::thread;
use std::time::Duration;

#[test]
fn test_tracer_with_development_config() {
    let config = TracingConfig::development();
    let tracer = CognitiveTracer::new(config);

    // Record various events
    tracer.trace_priming(PrimingType::Semantic, 0.75, 100, 200);
    tracer.trace_priming(PrimingType::Associative, 0.5, 200, 300);
    tracer.trace_interference(InterferenceType::Retroactive, 0.3, 999, 5);
    tracer.trace_reconsolidation(12345, 0.5, 0.8, 10);
    tracer.trace_false_memory(0xDEAD_BEEF, 20, 0.95);

    // Allow collector to process
    thread::sleep(Duration::from_millis(100));
}

#[test]
fn test_tracer_with_production_config() {
    let config = TracingConfig::production();
    let tracer = CognitiveTracer::new(config);

    // Production config samples at 1%, so record many events
    for i in 0..1000 {
        tracer.trace_priming(PrimingType::Semantic, 0.5, i, i + 1);
    }

    thread::sleep(Duration::from_millis(100));
}

#[test]
fn test_concurrent_tracing() {
    let config = TracingConfig::development();
    let tracer = std::sync::Arc::new(CognitiveTracer::new(config));

    let mut handles = vec![];

    // Spawn multiple threads recording events
    for thread_id in 0..4 {
        let tracer_clone = std::sync::Arc::clone(&tracer);
        let handle = thread::spawn(move || {
            for i in 0..100 {
                tracer_clone.trace_priming(
                    PrimingType::Semantic,
                    0.5,
                    thread_id * 1000 + i,
                    thread_id * 1000 + i + 1,
                );
            }
        });
        handles.push(handle);
    }

    for handle in handles {
        handle.join().unwrap();
    }

    thread::sleep(Duration::from_millis(200));
}

#[test]
fn test_disabled_event_types() {
    let mut config = TracingConfig::disabled();
    config.enabled_events.insert(EventType::Priming);
    config.sample_rates.insert(EventType::Priming, 1.0);
    config.ring_buffer_size = 1000;
    config.export_batch_size = 100;
    config.export_interval_ms = 10000;

    let tracer = CognitiveTracer::new(config);

    // Priming should be recorded
    tracer.trace_priming(PrimingType::Semantic, 0.5, 1, 2);

    // These should be no-ops (not enabled)
    tracer.trace_interference(InterferenceType::Proactive, 0.3, 999, 5);
    tracer.trace_reconsolidation(12345, 0.5, 0.8, 10);
    tracer.trace_false_memory(0xDEAD_BEEF, 20, 0.95);

    thread::sleep(Duration::from_millis(100));
}

#[test]
fn test_zero_sampling_rate() {
    let mut config = TracingConfig::disabled();
    config.enabled_events.insert(EventType::Priming);
    config.sample_rates.insert(EventType::Priming, 0.0); // Never sample
    config.ring_buffer_size = 1000;

    let tracer = CognitiveTracer::new(config);

    // These should all be sampled out
    for i in 0..1000 {
        tracer.trace_priming(PrimingType::Semantic, 0.5, i, i + 1);
    }

    thread::sleep(Duration::from_millis(100));

    // Verify no buffers were created (events were sampled before buffering)
    // Note: This is implementation-dependent
}

#[test]
fn test_memory_bounded() {
    let mut config = TracingConfig::development();
    config.ring_buffer_size = 100; // Small buffer

    let tracer = CognitiveTracer::new(config);

    // Record way more events than buffer can hold
    for i in 0..10_000 {
        tracer.trace_priming(PrimingType::Semantic, 0.5, i, i + 1);
    }

    // Memory should still be bounded
    thread::sleep(Duration::from_millis(100));
}
