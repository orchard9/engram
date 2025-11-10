//! Benchmarks for dual memory type system
//!
//! Validates that episode/concept type discrimination meets performance targets:
//! - Episode construction: <50ns (hot path, frequent)
//! - Concept construction: <100ns (cold path, infrequent)
//! - Atomic updates: <30ns (lock-free operations)
//! - Memory conversions: <200ns (includes UUID parsing)
//! - Memory overhead: <10% vs baseline Memory struct

#![allow(missing_docs)] // Benchmarks don't need documentation for all generated items

use criterion::{Criterion, black_box, criterion_group, criterion_main};
use engram_core::{Confidence, DualMemoryNode, Episode, Memory};
use uuid::Uuid;

fn benchmark_type_construction(c: &mut Criterion) {
    let mut group = c.benchmark_group("dual_memory_construction");

    // Target: <50ns for episode node creation (hot path)
    group.bench_function("create_episode_node", |b| {
        b.iter(|| {
            black_box(DualMemoryNode::new_episode(
                Uuid::new_v4(),
                "test".to_string(),
                [0.5f32; 768],
                Confidence::HIGH,
                0.7,
            ))
        });
    });

    // Target: <100ns for concept node creation (cold path, less critical)
    group.bench_function("create_concept_node", |b| {
        b.iter(|| {
            black_box(DualMemoryNode::new_concept(
                Uuid::new_v4(),
                [0.3f32; 768],
                0.85,
                10,
                Confidence::MEDIUM,
            ))
        });
    });

    group.finish();
}

fn benchmark_atomic_updates(c: &mut Criterion) {
    let episode = DualMemoryNode::new_episode(
        Uuid::new_v4(),
        "test".to_string(),
        [0.5f32; 768],
        Confidence::HIGH,
        0.7,
    );

    // Target: <25ns for atomic consolidation score update (matches metrics overhead)
    c.bench_function("atomic_consolidation_update", |b| {
        b.iter(|| {
            episode
                .node_type
                .update_consolidation_score(black_box(0.75))
        });
    });

    let concept =
        DualMemoryNode::new_concept(Uuid::new_v4(), [0.3f32; 768], 0.85, 10, Confidence::MEDIUM);

    // Target: <30ns for atomic instance count increment (AcqRel is slightly slower)
    c.bench_function("atomic_instance_increment", |b| {
        b.iter(|| concept.node_type.increment_instances());
    });

    // Target: <20ns for atomic activation update
    c.bench_function("atomic_activation_update", |b| {
        b.iter(|| episode.set_activation(black_box(0.6)));
    });
}

fn benchmark_conversions(c: &mut Criterion) {
    let memory = Memory::new("test".to_string(), [0.5f32; 768], Confidence::HIGH);

    // Target: <200ns for Memory -> DualMemoryNode (includes UUID parsing)
    c.bench_function("memory_to_dual_conversion", |b| {
        b.iter(|| black_box(DualMemoryNode::from(black_box(&memory))));
    });

    let dual = DualMemoryNode::new_episode(
        Uuid::new_v4(),
        "test".to_string(),
        [0.5f32; 768],
        Confidence::HIGH,
        0.7,
    );

    // Target: <150ns for DualMemoryNode -> Memory (copies embedding)
    c.bench_function("dual_to_memory_conversion", |b| {
        b.iter(|| black_box(dual.to_memory()));
    });

    // Episode to DualMemoryNode conversion
    c.bench_function("episode_to_dual_conversion", |b| {
        let episode = Episode::new(
            "test-episode".to_string(),
            chrono::Utc::now(),
            "Test content".to_string(),
            [0.4f32; 768],
            Confidence::MEDIUM,
        );
        b.iter(|| black_box(DualMemoryNode::from_episode(episode.clone(), 0.8)));
    });
}

fn benchmark_memory_overhead(c: &mut Criterion) {
    let mut group = c.benchmark_group("memory_footprint");

    // Measure allocation overhead for DualMemoryNode
    group.bench_function("allocate_1000_dual_nodes", |b| {
        b.iter(|| {
            let nodes: Vec<DualMemoryNode> = (0..1000)
                .map(|i| {
                    DualMemoryNode::new_episode(
                        Uuid::new_v4(),
                        format!("episode-{i}"),
                        [0.5f32; 768],
                        Confidence::HIGH,
                        0.7,
                    )
                })
                .collect();
            black_box(nodes)
        });
    });

    // Baseline: allocation overhead for Memory
    group.bench_function("allocate_1000_memory_nodes", |b| {
        b.iter(|| {
            let nodes: Vec<Memory> = (0..1000)
                .map(|i| Memory::new(format!("memory-{i}"), [0.5f32; 768], Confidence::HIGH))
                .collect();
            black_box(nodes)
        });
    });

    group.finish();
}

fn benchmark_type_checks(c: &mut Criterion) {
    let episode = DualMemoryNode::new_episode(
        Uuid::new_v4(),
        "test".to_string(),
        [0.5f32; 768],
        Confidence::HIGH,
        0.7,
    );

    let concept =
        DualMemoryNode::new_concept(Uuid::new_v4(), [0.3f32; 768], 0.85, 10, Confidence::MEDIUM);

    // Benchmark type discrimination (should be essentially free - just pattern matching)
    c.bench_function("is_episode_check", |b| {
        b.iter(|| black_box(episode.is_episode()));
    });

    c.bench_function("is_concept_check", |b| {
        b.iter(|| black_box(concept.is_concept()));
    });
}

fn benchmark_serialization_prep(c: &mut Criterion) {
    let dual = DualMemoryNode::new_episode(
        Uuid::new_v4(),
        "test".to_string(),
        [0.5f32; 768],
        Confidence::HIGH,
        0.7,
    );

    // Update atomic field
    dual.node_type.update_consolidation_score(0.75);

    // Benchmark serialization preparation
    c.bench_function("prepare_serialization", |b| {
        b.iter(|| {
            let mut node = dual.clone();
            node.node_type.prepare_serialization();
            black_box(node)
        });
    });

    // Benchmark atomic restoration
    c.bench_function("restore_atomics", |b| {
        b.iter(|| {
            let mut node = dual.clone();
            node.node_type.restore_atomics();
            black_box(node)
        });
    });
}

criterion_group!(
    benches,
    benchmark_type_construction,
    benchmark_atomic_updates,
    benchmark_conversions,
    benchmark_memory_overhead,
    benchmark_type_checks,
    benchmark_serialization_prep,
);
criterion_main!(benches);
