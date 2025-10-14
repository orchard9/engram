//! Benchmarks for GPU abstraction layer overhead

#![allow(missing_docs)]

use criterion::{Criterion, black_box, criterion_group, criterion_main};
use engram_core::activation::{
    AdaptiveConfig, AdaptiveSpreadingEngine, CpuFallback, GPUActivationBatch, GPUSpreadingInterface,
};

fn bench_cpu_fallback_direct(c: &mut Criterion) {
    let fallback = CpuFallback::new();
    let source = [1.0; 768];
    let mut batch = GPUActivationBatch::new(&source);

    // Add 100 targets
    for i in 0..100 {
        let mut target = [0.0; 768];
        target[i % 768] = 1.0;
        batch.add_target(&target, 0.5, 0.5);
    }

    c.bench_function("cpu_fallback_direct", |b| {
        b.iter(|| {
            let result = fallback.launch(&batch);
            futures::executor::block_on(result)
        });
    });
}

fn bench_adaptive_engine_cpu(c: &mut Criterion) {
    let config = AdaptiveConfig {
        gpu_threshold: 1000, // High threshold to force CPU
        enable_gpu: false,
    };
    let mut engine = AdaptiveSpreadingEngine::new(None, config, None);
    let source = [1.0; 768];
    let mut batch = GPUActivationBatch::new(&source);

    // Add 100 targets
    for i in 0..100 {
        let mut target = [0.0; 768];
        target[i % 768] = 1.0;
        batch.add_target(&target, 0.5, 0.5);
    }

    c.bench_function("adaptive_engine_cpu", |b| {
        b.iter(|| {
            let result = engine.spread(&batch);
            futures::executor::block_on(result)
        });
    });
}

fn bench_batch_construction(c: &mut Criterion) {
    c.bench_function("batch_construction_100", |b| {
        b.iter(|| {
            let source = [1.0; 768];
            let mut batch = GPUActivationBatch::new(&source);
            for i in 0..100 {
                let mut target = [0.0; 768];
                target[i % 768] = 1.0;
                batch.add_target(black_box(&target), 0.5, 0.5);
            }
            batch.ensure_contiguous();
            batch
        });
    });
}

fn bench_cosine_similarity(c: &mut Criterion) {
    let source = [1.0; 768];
    let target = [0.5; 768];

    c.bench_function("cosine_similarity_768d", |b| {
        b.iter(|| {
            let mut dot_product: f32 = 0.0;
            let mut norm_a: f32 = 0.0;
            let mut norm_b: f32 = 0.0;

            for i in 0..768 {
                dot_product += source[i] * target[i];
                norm_a += source[i] * source[i];
                norm_b += target[i] * target[i];
            }

            if norm_a > 0.0 && norm_b > 0.0 {
                dot_product / (norm_a.sqrt() * norm_b.sqrt())
            } else {
                0.0
            }
        });
    });
}

criterion_group!(
    benches,
    bench_cpu_fallback_direct,
    bench_adaptive_engine_cpu,
    bench_batch_construction,
    bench_cosine_similarity
);
criterion_main!(benches);
