//! Benchmark cognitive tracing overhead

#![cfg(feature = "cognitive_tracing")]
#![allow(missing_docs)]

use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};
use engram_core::tracing::{CognitiveTracer, InterferenceType, PrimingType, TracingConfig};

fn benchmark_priming_event(c: &mut Criterion) {
    let config = TracingConfig::development();
    let tracer = CognitiveTracer::new(config);

    c.bench_function("trace_priming_enabled", |b| {
        b.iter(|| {
            tracer.trace_priming(
                black_box(PrimingType::Semantic),
                black_box(0.75),
                black_box(100),
                black_box(200),
            );
        });
    });
}

fn benchmark_interference_event(c: &mut Criterion) {
    let config = TracingConfig::development();
    let tracer = CognitiveTracer::new(config);

    c.bench_function("trace_interference_enabled", |b| {
        b.iter(|| {
            tracer.trace_interference(
                black_box(InterferenceType::Retroactive),
                black_box(0.5),
                black_box(999),
                black_box(5),
            );
        });
    });
}

fn benchmark_reconsolidation_event(c: &mut Criterion) {
    let config = TracingConfig::development();
    let tracer = CognitiveTracer::new(config);

    c.bench_function("trace_reconsolidation_enabled", |b| {
        b.iter(|| {
            tracer.trace_reconsolidation(
                black_box(12345),
                black_box(0.3),
                black_box(0.8),
                black_box(10),
            );
        });
    });
}

fn benchmark_false_memory_event(c: &mut Criterion) {
    let config = TracingConfig::development();
    let tracer = CognitiveTracer::new(config);

    c.bench_function("trace_false_memory_enabled", |b| {
        b.iter(|| {
            tracer.trace_false_memory(black_box(0xDEAD_BEEF), black_box(20), black_box(0.95));
        });
    });
}

fn benchmark_sampling_overhead(c: &mut Criterion) {
    let mut group = c.benchmark_group("sampling");

    for sample_rate in [0.0, 0.01, 0.1, 0.5, 1.0] {
        let mut config = TracingConfig::disabled();
        config
            .enabled_events
            .insert(engram_core::tracing::EventType::Priming);
        config
            .sample_rates
            .insert(engram_core::tracing::EventType::Priming, sample_rate);
        config.ring_buffer_size = 10_000;

        let tracer = CognitiveTracer::new(config);

        group.bench_with_input(
            BenchmarkId::from_parameter(format!("rate_{sample_rate:.2}")),
            &sample_rate,
            |b, _| {
                b.iter(|| {
                    tracer.trace_priming(
                        black_box(PrimingType::Semantic),
                        black_box(0.5),
                        black_box(1),
                        black_box(2),
                    );
                });
            },
        );
    }

    group.finish();
}

fn benchmark_concurrent_tracing(c: &mut Criterion) {
    use std::sync::Arc;
    use std::thread;

    let config = TracingConfig::development();
    let tracer = Arc::new(CognitiveTracer::new(config));

    c.bench_function("concurrent_4_threads", |b| {
        b.iter(|| {
            let mut handles = vec![];

            for _ in 0..4 {
                let tracer_clone = Arc::clone(&tracer);
                let handle = thread::spawn(move || {
                    for i in 0..100 {
                        tracer_clone.trace_priming(PrimingType::Semantic, 0.5, i, i + 1);
                    }
                });
                handles.push(handle);
            }

            for handle in handles {
                handle.join().unwrap();
            }
        });
    });
}

criterion_group!(
    benches,
    benchmark_priming_event,
    benchmark_interference_event,
    benchmark_reconsolidation_event,
    benchmark_false_memory_event,
    benchmark_sampling_overhead,
    benchmark_concurrent_tracing,
);

criterion_main!(benches);
