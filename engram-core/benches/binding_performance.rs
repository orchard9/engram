//! Performance benchmarks for concept bindings
//!
//! Benchmarks cover:
//! - Binding creation and atomic updates
//! - Index operations (add, retrieve, update)
//! - SIMD batch operations
//! - Garbage collection
//! - High fan-out scenarios (concept → 1000 episodes)

#![cfg(feature = "dual_memory_types")]
#![allow(missing_docs)]

use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};
use engram_core::memory::binding_ops::BindingBatchOps;
use engram_core::memory::bindings::ConceptBinding;
use engram_core::memory_graph::binding_index::BindingIndex;
use std::sync::Arc;
use uuid::Uuid;

fn bench_binding_creation(c: &mut Criterion) {
    c.bench_function("binding_new", |b| {
        b.iter(|| {
            black_box(ConceptBinding::new(
                Uuid::new_v4(),
                Uuid::new_v4(),
                0.8,
                0.6,
            ));
        });
    });
}

fn bench_atomic_strength_update(c: &mut Criterion) {
    let binding = Arc::new(ConceptBinding::new(
        Uuid::new_v4(),
        Uuid::new_v4(),
        0.5,
        0.6,
    ));

    c.bench_function("binding_add_activation_uncontended", |b| {
        b.iter(|| {
            binding.add_activation(black_box(0.1));
        });
    });

    c.bench_function("binding_apply_decay_uncontended", |b| {
        b.iter(|| {
            binding.apply_decay(black_box(0.95));
        });
    });
}

fn bench_index_operations(c: &mut Criterion) {
    let index = Arc::new(BindingIndex::new(0.1));
    let ep_id = Uuid::new_v4();
    let con_id = Uuid::new_v4();

    // Pre-add a binding
    let binding = ConceptBinding::new(ep_id, con_id, 0.8, 0.6);
    index.add_binding(binding);

    c.bench_function("index_add_binding", |b| {
        b.iter(|| {
            let binding = ConceptBinding::new(Uuid::new_v4(), Uuid::new_v4(), 0.8, 0.6);
            index.add_binding(binding);
            black_box(());
        });
    });

    c.bench_function("index_get_concepts_for_episode", |b| {
        b.iter(|| {
            black_box(index.get_concepts_for_episode(&ep_id));
        });
    });

    c.bench_function("index_get_episodes_for_concept", |b| {
        b.iter(|| {
            black_box(index.get_episodes_for_concept(&con_id));
        });
    });

    c.bench_function("index_update_binding_strength", |b| {
        b.iter(|| {
            black_box(index.update_binding_strength(&ep_id, &con_id, |s| s + 0.01));
        });
    });
}

fn bench_simd_batch_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("simd_batch_operations");

    for size in [100, 500, 1000] {
        let bindings: Vec<Arc<ConceptBinding>> = (0..size)
            .map(|_| {
                Arc::new(ConceptBinding::new(
                    Uuid::new_v4(),
                    Uuid::new_v4(),
                    0.5,
                    0.6,
                ))
            })
            .collect();

        group.bench_with_input(
            BenchmarkId::new("batch_add_activation", size),
            &bindings,
            |b, bindings| {
                b.iter(|| {
                    BindingBatchOps::batch_add_activation(black_box(bindings), black_box(0.1));
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("batch_apply_decay", size),
            &bindings,
            |b, bindings| {
                b.iter(|| {
                    BindingBatchOps::batch_apply_decay(black_box(bindings), black_box(0.95));
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("count_above_threshold", size),
            &bindings,
            |b, bindings| {
                b.iter(|| {
                    black_box(BindingBatchOps::count_above_threshold(
                        black_box(bindings),
                        black_box(0.5),
                    ));
                });
            },
        );
    }

    group.finish();
}

fn bench_high_fanout_scenario(c: &mut Criterion) {
    let index = BindingIndex::new(0.1);
    let con_id = Uuid::new_v4();

    // Create concept with 1000 episode bindings
    for _ in 0..1000 {
        let binding = ConceptBinding::new(Uuid::new_v4(), con_id, 0.8, 0.6);
        index.add_binding(binding);
    }

    c.bench_function("high_fanout_get_episodes (1000)", |b| {
        b.iter(|| {
            black_box(index.get_episodes_for_concept(&con_id));
        });
    });

    let episodes_arc = index.get_episodes_for_concept(&con_id);

    c.bench_function("high_fanout_simd_batch_activation (1000)", |b| {
        b.iter(|| {
            BindingBatchOps::batch_add_activation(black_box(&episodes_arc), black_box(0.01));
        });
    });
}

fn bench_garbage_collection(c: &mut Criterion) {
    let mut group = c.benchmark_group("garbage_collection");

    for weak_pct in [10, 30, 50] {
        let index = BindingIndex::new(0.3);
        let con_id = Uuid::new_v4();

        // Add mix of strong and weak bindings
        for i in 0..1000 {
            let strength = if i < weak_pct * 10 { 0.2 } else { 0.8 };
            let binding = ConceptBinding::new(Uuid::new_v4(), con_id, strength, 0.6);
            index.add_binding(binding);
        }

        group.bench_with_input(
            BenchmarkId::new("gc_1000_bindings", weak_pct),
            &index,
            |b, index| {
                b.iter(|| {
                    black_box(index.garbage_collect());
                });
            },
        );
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_binding_creation,
    bench_atomic_strength_update,
    bench_index_operations,
    bench_simd_batch_operations,
    bench_high_fanout_scenario,
    bench_garbage_collection
);
criterion_main!(benches);
