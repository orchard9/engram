//! Benchmark for DualMemoryBudget lock-free allocation tracking
//!
//! Measures overhead of budget operations under concurrent load.

#![allow(missing_docs)] // Benchmarks don't need documentation for all generated items

use criterion::{BenchmarkId, Criterion, Throughput, black_box, criterion_group, criterion_main};
use engram_core::storage::DualMemoryBudget;
use std::sync::Arc;
use std::thread;

const NODE_SIZE: usize = 3328; // DualMemoryNode size

fn bench_single_thread_allocation(c: &mut Criterion) {
    let mut group = c.benchmark_group("single_thread");

    let budget = DualMemoryBudget::new(512, 1024);

    group.bench_function("can_allocate_episode", |b| {
        b.iter(|| {
            black_box(budget.can_allocate_episode());
        });
    });

    group.bench_function("record_episode_allocation", |b| {
        b.iter(|| {
            budget.record_episode_allocation(black_box(NODE_SIZE));
        });
    });

    group.bench_function("record_episode_deallocation", |b| {
        b.iter(|| {
            budget.record_episode_deallocation(black_box(NODE_SIZE));
        });
    });

    group.bench_function("episode_utilization", |b| {
        b.iter(|| {
            black_box(budget.episode_utilization());
        });
    });

    group.finish();
}

fn bench_concurrent_allocation(c: &mut Criterion) {
    let mut group = c.benchmark_group("concurrent");
    group.throughput(Throughput::Elements(1));

    for num_threads in [2, 4, 8, 16] {
        let budget = Arc::new(DualMemoryBudget::new(1024, 2048));

        group.bench_with_input(
            BenchmarkId::new("allocation", num_threads),
            &num_threads,
            |b, &threads| {
                b.iter(|| {
                    let budget = Arc::clone(&budget);
                    let handles: Vec<_> = (0..threads)
                        .map(|_| {
                            let budget = Arc::clone(&budget);
                            thread::spawn(move || {
                                for _ in 0..100 {
                                    budget.record_episode_allocation(NODE_SIZE);
                                    budget.record_concept_allocation(NODE_SIZE);
                                }
                            })
                        })
                        .collect();

                    for handle in handles {
                        handle.join().unwrap();
                    }
                });
            },
        );
    }

    group.finish();
}

fn bench_mixed_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("mixed_operations");

    let budget = Arc::new(DualMemoryBudget::new(512, 1024));

    group.bench_function("alloc_check_dealloc", |b| {
        b.iter(|| {
            if budget.can_allocate_episode() {
                budget.record_episode_allocation(NODE_SIZE);
                let _ = black_box(budget.episode_utilization());
                budget.record_episode_deallocation(NODE_SIZE);
            }
        });
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_single_thread_allocation,
    bench_concurrent_allocation,
    bench_mixed_operations
);
criterion_main!(benches);
