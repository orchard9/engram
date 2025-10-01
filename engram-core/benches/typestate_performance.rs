//! Zero runtime cost validation benchmarks for typestate patterns
//!
//! These benchmarks verify that typestate patterns have identical runtime
//! performance to unsafe alternatives, demonstrating that compile-time safety
//! comes without performance compromise.

#![allow(missing_docs)]

use chrono::Utc;
use criterion::{Criterion, black_box, criterion_group, criterion_main};
use engram_core::Confidence;
use engram_core::memory::{Cue, CueBuilder, Episode, EpisodeBuilder, Memory, MemoryBuilder};

/// Benchmark Memory construction using typestate builder vs direct construction
/// This validates that the builder pattern has zero runtime overhead
fn benchmark_memory_construction(c: &mut Criterion) {
    let mut group = c.benchmark_group("memory_construction");

    group.bench_function("typestate_builder", |b| {
        b.iter(|| {
            let memory = MemoryBuilder::new()
                .id(black_box("benchmark_memory".to_string()))
                .embedding(black_box([0.5f32; 768]))
                .confidence(black_box(Confidence::HIGH))
                .content(black_box("Benchmark content".to_string()))
                .decay_rate(black_box(0.1))
                .build();
            black_box(memory);
        });
    });

    group.bench_function("direct_construction", |b| {
        b.iter(|| {
            let memory = Memory::new(
                black_box("benchmark_memory".to_string()),
                black_box([0.5f32; 768]),
                black_box(Confidence::HIGH),
            );
            // Note: Direct construction doesn't allow setting content/decay_rate
            // This demonstrates that builder provides more functionality at same cost
            black_box(memory);
        });
    });

    group.finish();
}

/// Benchmark Episode construction comparing typestate vs direct patterns
fn benchmark_episode_construction(c: &mut Criterion) {
    let mut group = c.benchmark_group("episode_construction");
    let now = Utc::now();

    group.bench_function("typestate_builder", |b| {
        b.iter(|| {
            let episode = EpisodeBuilder::new()
                .id(black_box("benchmark_episode".to_string()))
                .when(black_box(now))
                .what(black_box("Benchmark event".to_string()))
                .embedding(black_box([0.6f32; 768]))
                .confidence(black_box(Confidence::MEDIUM))
                .where_location(black_box("Benchmark Location".to_string()))
                .who(black_box(vec!["Alice".to_string(), "Bob".to_string()]))
                .decay_rate(black_box(0.05))
                .build();
            black_box(episode);
        });
    });

    group.bench_function("direct_construction", |b| {
        b.iter(|| {
            let episode = Episode::new(
                black_box("benchmark_episode".to_string()),
                black_box(now),
                black_box("Benchmark event".to_string()),
                black_box([0.6f32; 768]),
                black_box(Confidence::MEDIUM),
            );
            // Direct construction provides fewer customization options
            black_box(episode);
        });
    });

    group.finish();
}

/// Benchmark Cue construction with different search types
fn benchmark_cue_construction(c: &mut Criterion) {
    let mut group = c.benchmark_group("cue_construction");

    group.bench_function("typestate_embedding_cue", |b| {
        b.iter(|| {
            let cue = CueBuilder::new()
                .id(black_box("benchmark_cue".to_string()))
                .embedding_search(black_box([0.7f32; 768]), black_box(Confidence::HIGH))
                .cue_confidence(black_box(Confidence::MEDIUM))
                .result_threshold(black_box(Confidence::LOW))
                .max_results(black_box(50))
                .build();
            black_box(cue);
        });
    });

    group.bench_function("direct_embedding_cue", |b| {
        b.iter(|| {
            let cue = Cue::embedding(
                black_box("benchmark_cue".to_string()),
                black_box([0.7f32; 768]),
                black_box(Confidence::HIGH),
            );
            black_box(cue);
        });
    });

    group.bench_function("typestate_semantic_cue", |b| {
        b.iter(|| {
            let cue = CueBuilder::new()
                .id(black_box("semantic_cue".to_string()))
                .semantic_search(
                    black_box("benchmark query".to_string()),
                    black_box(Confidence::MEDIUM),
                )
                .max_results(black_box(25))
                .build();
            black_box(cue);
        });
    });

    group.bench_function("direct_semantic_cue", |b| {
        b.iter(|| {
            let cue = Cue::semantic(
                black_box("semantic_cue".to_string()),
                black_box("benchmark query".to_string()),
                black_box(Confidence::MEDIUM),
            );
            black_box(cue);
        });
    });

    group.finish();
}

/// Benchmark complex construction patterns to verify no overhead in realistic scenarios
fn benchmark_complex_patterns(c: &mut Criterion) {
    let mut group = c.benchmark_group("complex_patterns");

    group.bench_function("memory_episode_cue_creation", |b| {
        b.iter(|| {
            // Realistic pattern: create related memory, episode, and cue
            let memory = MemoryBuilder::new()
                .id(black_box("complex_memory".to_string()))
                .embedding(black_box([0.8f32; 768]))
                .confidence(black_box(Confidence::HIGH))
                .content(black_box("Complex benchmark content".to_string()))
                .build();

            let episode = EpisodeBuilder::new()
                .id(black_box("complex_episode".to_string()))
                .when(black_box(Utc::now()))
                .what(black_box("Complex benchmark event".to_string()))
                .embedding(black_box([0.8f32; 768]))
                .confidence(black_box(Confidence::HIGH))
                .build();

            let cue = CueBuilder::new()
                .id(black_box("complex_cue".to_string()))
                .semantic_search(
                    black_box("complex benchmark".to_string()),
                    black_box(Confidence::MEDIUM),
                )
                .build();

            black_box((memory, episode, cue));
        });
    });

    group.bench_function("batch_memory_creation", |b| {
        b.iter(|| {
            let memories: Vec<_> = (0..10)
                .map(|i| {
                    MemoryBuilder::new()
                        .id(black_box(format!("batch_memory_{i}")))
                        .embedding(black_box({
                            #[allow(clippy::cast_precision_loss)]
                            [i as f32 * 0.1; 768]
                        }))
                        .confidence(black_box(Confidence::MEDIUM))
                        .build()
                })
                .collect();
            black_box(memories);
        });
    });

    group.finish();
}

/// Benchmark memory operations to ensure typestate doesn't affect runtime usage
fn benchmark_memory_operations(c: &mut Criterion) {
    let memory = MemoryBuilder::new()
        .id("operation_test".to_string())
        .embedding([0.5f32; 768])
        .confidence(Confidence::HIGH)
        .build();

    let mut episode = EpisodeBuilder::new()
        .id("episode_ops".to_string())
        .when(Utc::now())
        .what("Test episode".to_string())
        .embedding([0.5f32; 768])
        .confidence(Confidence::MEDIUM)
        .build();

    let mut group = c.benchmark_group("memory_operations");

    group.bench_function("memory_activation_ops", |b| {
        b.iter(|| {
            memory.set_activation(black_box(0.7));
            let activation = memory.activation();
            memory.add_activation(black_box(0.1));
            black_box(activation);
        });
    });

    group.bench_function("memory_confidence_ops", |b| {
        b.iter(|| {
            let reliable = memory.seems_reliable();
            let accurate = memory.is_accurate();
            let trustworthy = memory.is_trustworthy();
            black_box((reliable, accurate, trustworthy));
        });
    });

    group.bench_function("episode_operations", |b| {
        b.iter(|| {
            let vivid = episode.is_vivid();
            let detailed = episode.is_detailed();
            let authentic = episode.feels_authentic();
            episode.record_recall();
            black_box((vivid, detailed, authentic));
        });
    });

    group.finish();
}

/// Benchmark compilation time impact measurement
/// This doesn't measure runtime but provides data on compile-time costs
fn benchmark_compilation_patterns(c: &mut Criterion) {
    let mut group = c.benchmark_group("compilation_patterns");

    // These benchmarks measure the runtime of creating builders
    // The actual compilation cost is measured by build time, not runtime
    group.bench_function("builder_state_transitions", |b| {
        b.iter(|| {
            // Measure the cost of state transitions (should be zero at runtime)
            let builder1 = MemoryBuilder::new();
            let builder2 = builder1.id(black_box("state_test".to_string()));
            let builder3 = builder2.embedding(black_box([0.9f32; 768]));
            let builder4 = builder3.confidence(black_box(Confidence::HIGH));
            let memory = builder4.build();
            black_box(memory);
        });
    });

    group.bench_function("phantom_type_overhead", |b| {
        b.iter(|| {
            // Phantom types should have zero runtime cost
            // This measures whether PhantomData affects performance
            let memory = MemoryBuilder::new()
                .id(black_box("phantom_test".to_string()))
                .embedding(black_box([0.1f32; 768]))
                .confidence(black_box(Confidence::LOW))
                .build();
            black_box(memory);
        });
    });

    group.finish();
}

criterion_group!(
    benches,
    benchmark_memory_construction,
    benchmark_episode_construction,
    benchmark_cue_construction,
    benchmark_complex_patterns,
    benchmark_memory_operations,
    benchmark_compilation_patterns
);
criterion_main!(benches);
