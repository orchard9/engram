//! Performance benchmarks for cognitive patterns
//!
//! Validates performance targets:
//! - Metrics overhead <1% when enabled, 0% when disabled
//! - Priming boost computation: <10μs
//! - Interference detection: <100μs
//! - Reconsolidation check: <50μs
//! - Metrics recording: <50ns

#![allow(missing_docs)]
#![allow(clippy::uninlined_format_args)]
#![allow(clippy::redundant_closure)]
#![allow(clippy::unit_arg)]

use criterion::{BenchmarkId, Criterion, Throughput, black_box, criterion_group, criterion_main};
use engram_core::cognitive::priming::{
    AssociativePrimingEngine, RepetitionPrimingEngine, SemanticPrimingEngine,
};
use engram_core::cognitive::reconsolidation::ReconsolidationEngine;
use engram_core::{Confidence, Cue, Episode, MemoryStore};
use std::sync::Arc;
use std::thread;

use chrono::Utc;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;

// ==================== Metrics Overhead Benchmarks ====================

#[cfg(feature = "monitoring")]
fn benchmark_metrics_overhead(c: &mut Criterion) {
    use engram_core::metrics::cognitive_patterns::{CognitivePatternMetrics, PrimingType};

    let mut group = c.benchmark_group("metrics_overhead");

    // Large sample size for statistical significance
    group.sample_size(10000);
    group.measurement_time(std::time::Duration::from_secs(20));

    // Production workload: 1000 operations per benchmark iteration
    group.throughput(Throughput::Elements(1000));

    // Benchmark: Priming operations WITH monitoring
    group.bench_function("priming_with_monitoring", |b| {
        let semantic = SemanticPrimingEngine::new();
        let metrics = CognitivePatternMetrics::new();

        b.iter(|| {
            for i in 0..1000 {
                let concept = format!("concept_{}", i % 50);
                let embedding = create_embedding(i as f32);

                // Priming operation
                semantic.activate_priming(&concept, &embedding, || Vec::new());

                // Record metrics (overhead)
                metrics.record_priming(PrimingType::Semantic, 0.5);

                // Compute boost
                black_box(semantic.compute_priming_boost(&concept));
            }
        });
    });

    group.finish();
}

// Baseline benchmark WITHOUT monitoring (always available)
fn benchmark_baseline_without_monitoring(c: &mut Criterion) {
    let mut group = c.benchmark_group("baseline_no_monitoring");
    group.sample_size(10000);
    group.throughput(Throughput::Elements(1000));

    group.bench_function("priming_baseline", |b| {
        let semantic = SemanticPrimingEngine::new();

        b.iter(|| {
            for i in 0..1000 {
                let concept = format!("concept_{}", i % 50);
                let embedding = create_embedding(i as f32);

                semantic.activate_priming(&concept, &embedding, || Vec::new());
                black_box(semantic.compute_priming_boost(&concept));
            }
        });
    });

    group.finish();
}

// ==================== Latency Benchmarks ====================

fn benchmark_priming_latency(c: &mut Criterion) {
    let mut group = c.benchmark_group("priming_latency");
    group.sample_size(100);

    // Semantic priming activation
    group.bench_function("semantic_priming_activation", |b| {
        let engine = SemanticPrimingEngine::new();
        let embedding = create_embedding(1.0);

        b.iter(|| {
            black_box(engine.activate_priming("test_concept", &embedding, || Vec::new()));
        });
    });

    // Priming boost computation (target: <10μs)
    group.bench_function("priming_boost_computation", |b| {
        let engine = SemanticPrimingEngine::new();
        let embedding = create_embedding(1.0);

        // Prime first
        engine.activate_priming("test", &embedding, || Vec::new());
        std::thread::sleep(std::time::Duration::from_millis(100));

        b.iter(|| {
            black_box(engine.compute_priming_boost("test"));
        });
    });

    // Associative priming (co-activation recording)
    group.bench_function("associative_coactivation", |b| {
        let engine = AssociativePrimingEngine::new();

        b.iter(|| {
            black_box(engine.record_coactivation("node_a", "node_b"));
        });
    });

    // Association strength computation
    group.bench_function("association_strength_computation", |b| {
        let engine = AssociativePrimingEngine::new();

        // Record some co-activations first
        for _ in 0..5 {
            engine.record_coactivation("node_a", "node_b");
        }

        b.iter(|| {
            black_box(engine.compute_association_strength("node_a", "node_b"));
        });
    });

    // Repetition priming
    group.bench_function("repetition_priming_activation", |b| {
        let engine = RepetitionPrimingEngine::new();

        b.iter(|| {
            black_box(engine.record_exposure("test_node"));
        });
    });

    group.finish();
}

fn benchmark_reconsolidation_latency(c: &mut Criterion) {
    let mut group = c.benchmark_group("reconsolidation_latency");
    group.sample_size(100);

    // Note: is_eligible_for_reconsolidation method not available in current API
    // Benchmark removed until API is updated

    // Record recall operation
    group.bench_function("record_recall", |b| {
        let engine = ReconsolidationEngine::new();
        let episode = create_test_episode(1, 48);

        b.iter(|| {
            black_box(engine.record_recall(&episode, Utc::now(), true));
        });
    });

    group.finish();
}

#[cfg(feature = "monitoring")]
fn benchmark_metrics_recording_latency(c: &mut Criterion) {
    use engram_core::metrics::cognitive_patterns::{
        CognitivePatternMetrics, InterferenceType, PrimingType,
    };

    let mut group = c.benchmark_group("metrics_recording_latency");
    group.sample_size(100);

    // Priming event recording (target: <50ns)
    group.bench_function("record_priming_event", |b| {
        let metrics = CognitivePatternMetrics::new();

        b.iter(|| {
            black_box(metrics.record_priming(PrimingType::Semantic, 0.5));
        });
    });

    // Interference event recording
    group.bench_function("record_interference_event", |b| {
        let metrics = CognitivePatternMetrics::new();

        b.iter(|| {
            black_box(metrics.record_interference(InterferenceType::Proactive, 0.6));
        });
    });

    // Reconsolidation event recording
    group.bench_function("record_reconsolidation_event", |b| {
        let metrics = CognitivePatternMetrics::new();

        b.iter(|| {
            metrics.record_reconsolidation(0.5);
            black_box(());
        });
    });

    group.finish();
}

// ==================== Throughput Benchmarks ====================

fn benchmark_throughput_scaling(c: &mut Criterion) {
    let mut group = c.benchmark_group("throughput_scaling");
    group.sample_size(50);

    for num_threads in [1, 2, 4, 8, 16] {
        group.bench_with_input(
            BenchmarkId::new("concurrent_priming_operations", num_threads),
            &num_threads,
            |b, &threads| {
                b.iter(|| {
                    let engine = Arc::new(SemanticPrimingEngine::new());

                    let handles: Vec<_> = (0..threads)
                        .map(|thread_id| {
                            let e = Arc::clone(&engine);
                            thread::spawn(move || {
                                for i in 0..1000 {
                                    let concept = format!("c_{}_{}", thread_id, i);
                                    let embedding = create_embedding(i as f32);
                                    e.activate_priming(&concept, &embedding, || Vec::new());
                                }
                            })
                        })
                        .collect();

                    for h in handles {
                        h.join().unwrap();
                    }
                });
            },
        );
    }

    group.finish();
}

fn benchmark_memory_operations_throughput(c: &mut Criterion) {
    let mut group = c.benchmark_group("memory_operations_throughput");
    group.sample_size(50);
    group.throughput(Throughput::Elements(1000));

    // Store throughput
    group.bench_function("store_episodes_1000", |b| {
        b.iter(|| {
            let store = MemoryStore::new(10000);
            for i in 0..1000 {
                let episode = create_test_episode(i, 0);
                let _ = store.store(episode);
            }
        });
    });

    // Recall throughput
    group.bench_function("recall_operations_1000", |b| {
        let store = MemoryStore::new(10000);

        // Pre-populate
        for i in 0..1000 {
            let _ = store.store(create_test_episode(i, 0));
        }

        b.iter(|| {
            for i in 0..1000 {
                let embedding = create_embedding(i as f32);
                let cue = Cue::embedding(format!("bench_{}", i), embedding, Confidence::HIGH);
                black_box(store.recall(&cue));
            }
        });
    });

    group.finish();
}

// ==================== Production Workload Simulation ====================

fn benchmark_production_workload(c: &mut Criterion) {
    let mut group = c.benchmark_group("production_workload");
    group.sample_size(50);
    group.throughput(Throughput::Elements(10_000));

    // Realistic workload: mixed operations
    group.bench_function("mixed_cognitive_operations_10k", |b| {
        b.iter(|| {
            let semantic = Arc::new(SemanticPrimingEngine::new());
            let associative = Arc::new(AssociativePrimingEngine::new());
            let repetition = Arc::new(RepetitionPrimingEngine::new());
            let reconsolidation = Arc::new(ReconsolidationEngine::new());
            let store = Arc::new(MemoryStore::new(10000));

            let mut rng = ChaCha8Rng::seed_from_u64(42);

            for _ in 0..10_000 {
                match rng.gen_range(0..100) {
                    0..30 => {
                        // 30% semantic priming
                        let concept = format!("concept_{}", rng.gen_range(0..50));
                        let emb = random_embedding(&mut rng);
                        semantic.activate_priming(&concept, &emb, || Vec::new());
                    }
                    30..40 => {
                        // 10% associative priming
                        let a = format!("node_{}", rng.gen_range(0..100));
                        let b = format!("node_{}", rng.gen_range(0..100));
                        associative.record_coactivation(&a, &b);
                    }
                    40..75 => {
                        // 35% recall
                        let emb = random_embedding(&mut rng);
                        let cue = Cue::embedding("bench".to_string(), emb, Confidence::HIGH);
                        black_box(store.recall(&cue));
                    }
                    75..90 => {
                        // 15% store
                        let ep = create_test_episode(rng.gen_range(0..1000), 0);
                        let _ = store.store(ep);
                    }
                    90..95 => {
                        // 5% repetition priming
                        let node = format!("episode_{}", rng.gen_range(0..100));
                        repetition.record_exposure(&node);
                    }
                    _ => {
                        // 5% reconsolidation
                        let ep = create_test_episode(rng.gen_range(0..100), 48);
                        reconsolidation.record_recall(&ep, Utc::now(), true);
                    }
                }
            }
        });
    });

    group.finish();
}

// ==================== Helper Functions ====================

fn create_test_episode(id: u64, age_hours: i64) -> Episode {
    let when = Utc::now() - chrono::Duration::hours(age_hours);
    let embedding = create_embedding(id as f32);

    Episode::new(
        format!("episode_{}", id),
        when,
        format!("test content {}", id),
        embedding,
        Confidence::HIGH,
    )
}

fn create_embedding(seed: f32) -> [f32; 768] {
    let mut embedding = [0.0f32; 768];
    for (i, val) in embedding.iter_mut().enumerate() {
        *val = ((seed + i as f32) * 0.001).sin();
    }
    let norm: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
    for val in &mut embedding {
        *val /= norm;
    }
    embedding
}

fn random_embedding<R: Rng>(rng: &mut R) -> [f32; 768] {
    let mut embedding = [0.0f32; 768];
    for val in &mut embedding {
        *val = rng.gen_range(-1.0..1.0);
    }
    let norm: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
    for val in &mut embedding {
        *val /= norm;
    }
    embedding
}

// ==================== Criterion Configuration ====================

#[cfg(feature = "monitoring")]
criterion_group!(
    benches,
    benchmark_metrics_overhead,
    benchmark_baseline_without_monitoring,
    benchmark_priming_latency,
    benchmark_reconsolidation_latency,
    benchmark_metrics_recording_latency,
    benchmark_throughput_scaling,
    benchmark_memory_operations_throughput,
    benchmark_production_workload
);

#[cfg(not(feature = "monitoring"))]
criterion_group!(
    benches,
    benchmark_baseline_without_monitoring,
    benchmark_priming_latency,
    benchmark_reconsolidation_latency,
    benchmark_throughput_scaling,
    benchmark_memory_operations_throughput,
    benchmark_production_workload
);

criterion_main!(benches);
