//! Integration tests for concept bindings and binding index
//!
//! Tests cover:
//! - Cache alignment verification
//! - Atomic strength updates (concurrent scenarios)
//! - Bidirectional consistency (episode ↔ concept)
//! - SIMD correctness (comparison vs scalar)
//! - Garbage collection
//! - Memory overhead measurement
//! - Spreading activation integration

#![cfg(feature = "dual_memory_types")]

use engram_core::memory::binding_ops::BindingBatchOps;
use engram_core::memory::bindings::{BindingRef, ConceptBinding};
use engram_core::memory_graph::binding_index::BindingIndex;
use std::sync::Arc;
use std::thread;
use uuid::Uuid;

#[test]
fn test_binding_cache_alignment() {
    // Verify 64-byte alignment for optimal cache performance
    assert_eq!(
        std::mem::align_of::<ConceptBinding>(),
        64,
        "ConceptBinding must be cache-line aligned"
    );

    // Verify size fits in one cache line
    let size = std::mem::size_of::<ConceptBinding>();
    assert!(
        size <= 64,
        "ConceptBinding size ({size} bytes) must fit in 64-byte cache line"
    );
}

#[test]
fn test_binding_creation_and_getters() {
    let ep_id = Uuid::new_v4();
    let con_id = Uuid::new_v4();

    let binding = ConceptBinding::new(ep_id, con_id, 0.8, 0.6);

    assert_eq!(binding.episode_id, ep_id);
    assert_eq!(binding.concept_id, con_id);
    assert!((binding.get_strength() - 0.8).abs() < 0.001);
    assert!((binding.contribution - 0.6).abs() < 0.001);
}

#[test]
fn test_binding_strength_clamping() {
    // Test upper bound
    let binding = ConceptBinding::new(Uuid::new_v4(), Uuid::new_v4(), 1.5, 0.5);
    assert!((binding.get_strength() - 1.0).abs() < 0.001);

    // Test lower bound
    let binding2 = ConceptBinding::new(Uuid::new_v4(), Uuid::new_v4(), -0.5, 0.5);
    assert!((binding2.get_strength() - 0.0).abs() < 0.001);
}

#[test]
fn test_atomic_strength_updates() {
    let binding = ConceptBinding::new(Uuid::new_v4(), Uuid::new_v4(), 0.5, 0.5);

    // Test addition
    binding.add_activation(0.2);
    assert!((binding.get_strength() - 0.7).abs() < 0.001);

    // Test saturation at 1.0
    binding.add_activation(0.5);
    assert!((binding.get_strength() - 1.0).abs() < 0.001);

    // Test decay
    binding.apply_decay(0.5);
    assert!((binding.get_strength() - 0.5).abs() < 0.001);
}

#[test]
fn test_concurrent_binding_updates() {
    let binding = Arc::new(ConceptBinding::new(
        Uuid::new_v4(),
        Uuid::new_v4(),
        0.0,
        0.5,
    ));

    // Spawn 100 threads, each adding 0.01 (should reach 1.0, clamped)
    let threads: Vec<_> = (0..100)
        .map(|_| {
            let b = Arc::clone(&binding);
            thread::spawn(move || {
                for _ in 0..10 {
                    b.add_activation(0.001);
                }
            })
        })
        .collect();

    for thread in threads {
        thread.join().expect("Thread panicked");
    }

    // Should have accumulated to 1.0 (clamped)
    assert!((binding.get_strength() - 1.0).abs() < 0.001);
}

#[test]
fn test_binding_index_bidirectional_consistency() {
    let index = BindingIndex::new(0.1);
    let ep_id = Uuid::new_v4();
    let con_id = Uuid::new_v4();

    // Add binding
    let binding = ConceptBinding::new(ep_id, con_id, 0.8, 0.6);
    index.add_binding(binding);

    // Verify episode → concepts
    let concepts = index.get_concepts_for_episode(&ep_id);
    assert_eq!(concepts.len(), 1);
    assert_eq!(concepts[0].concept_id, con_id);
    assert!((concepts[0].get_strength() - 0.8).abs() < 0.001);

    // Verify concept → episodes
    let episodes = index.get_episodes_for_concept(&con_id);
    assert_eq!(episodes.len(), 1);
    assert_eq!(episodes[0].episode_id, ep_id);
    assert!((episodes[0].get_strength() - 0.8).abs() < 0.001);

    // Verify count
    assert_eq!(index.count(), 1);
}

#[test]
fn test_binding_index_multiple_bindings() {
    let index = BindingIndex::new(0.1);
    let ep_id = Uuid::new_v4();
    let con_id1 = Uuid::new_v4();
    let con_id2 = Uuid::new_v4();
    let con_id3 = Uuid::new_v4();

    // Add 3 concepts for same episode
    index.add_binding(ConceptBinding::new(ep_id, con_id1, 0.8, 0.6));
    index.add_binding(ConceptBinding::new(ep_id, con_id2, 0.7, 0.5));
    index.add_binding(ConceptBinding::new(ep_id, con_id3, 0.6, 0.4));

    assert_eq!(index.count(), 3);

    // Verify all concepts retrieved
    let concepts = index.get_concepts_for_episode(&ep_id);
    assert_eq!(concepts.len(), 3);

    // Verify each concept has single episode
    for con_id in [con_id1, con_id2, con_id3] {
        let episodes = index.get_episodes_for_concept(&con_id);
        assert_eq!(episodes.len(), 1);
        assert_eq!(episodes[0].episode_id, ep_id);
    }
}

#[test]
fn test_binding_garbage_collection() {
    let index = BindingIndex::new(0.3); // GC threshold
    let ep_id1 = Uuid::new_v4();
    let ep_id2 = Uuid::new_v4();
    let con_id = Uuid::new_v4();

    // Add strong and weak bindings
    let strong = ConceptBinding::new(ep_id1, con_id, 0.8, 0.6);
    let weak = ConceptBinding::new(ep_id2, con_id, 0.2, 0.4);

    index.add_binding(strong);
    index.add_binding(weak);

    assert_eq!(index.count(), 2);

    // GC should remove weak binding
    let removed = index.garbage_collect();
    assert_eq!(removed, 1);
    assert_eq!(index.count(), 1);

    // Verify strong binding remains
    let concepts = index.get_concepts_for_episode(&ep_id1);
    assert_eq!(concepts.len(), 1);

    // Verify weak binding removed
    let concepts = index.get_concepts_for_episode(&ep_id2);
    assert!(concepts.is_empty());
}

#[test]
fn test_binding_removal() {
    let index = BindingIndex::new(0.1);
    let ep_id = Uuid::new_v4();
    let con_id1 = Uuid::new_v4();
    let con_id2 = Uuid::new_v4();

    index.add_binding(ConceptBinding::new(ep_id, con_id1, 0.8, 0.6));
    index.add_binding(ConceptBinding::new(ep_id, con_id2, 0.7, 0.5));

    assert_eq!(index.count(), 2);

    // Remove episode bindings
    let removed = index.remove_episode_bindings(&ep_id);
    assert_eq!(removed, 2);
    assert_eq!(index.count(), 0);

    // Verify removed from both indices
    assert!(index.get_concepts_for_episode(&ep_id).is_empty());
    assert!(index.get_episodes_for_concept(&con_id1).is_empty());
    assert!(index.get_episodes_for_concept(&con_id2).is_empty());
}

#[test]
fn test_simd_batch_add_activation() {
    let bindings: Vec<Arc<ConceptBinding>> = (0..100)
        .map(|_| {
            Arc::new(ConceptBinding::new(
                Uuid::new_v4(),
                Uuid::new_v4(),
                0.3,
                0.5,
            ))
        })
        .collect();

    BindingBatchOps::batch_add_activation(&bindings, 0.2);

    for binding in &bindings {
        assert!((binding.get_strength() - 0.5).abs() < 0.001);
    }
}

#[test]
fn test_simd_batch_apply_decay() {
    let bindings: Vec<Arc<ConceptBinding>> = (0..100)
        .map(|_| {
            Arc::new(ConceptBinding::new(
                Uuid::new_v4(),
                Uuid::new_v4(),
                0.8,
                0.5,
            ))
        })
        .collect();

    BindingBatchOps::batch_apply_decay(&bindings, 0.5);

    for binding in &bindings {
        assert!((binding.get_strength() - 0.4).abs() < 0.001);
    }
}

#[test]
fn test_simd_count_above_threshold() {
    let bindings: Vec<Arc<ConceptBinding>> = (0..100)
        .map(|i| {
            let strength = if i < 50 { 0.3 } else { 0.7 };
            Arc::new(ConceptBinding::new(
                Uuid::new_v4(),
                Uuid::new_v4(),
                strength,
                0.5,
            ))
        })
        .collect();

    let count = BindingBatchOps::count_above_threshold(&bindings, 0.5);
    assert_eq!(count, 50);
}

#[test]
fn test_simd_correctness_with_remainder() {
    // Test with non-multiple of 8 to ensure remainder handling
    let bindings: Vec<Arc<ConceptBinding>> = (0..13)
        .map(|_| {
            Arc::new(ConceptBinding::new(
                Uuid::new_v4(),
                Uuid::new_v4(),
                0.5,
                0.5,
            ))
        })
        .collect();

    BindingBatchOps::batch_add_activation(&bindings, 0.2);

    for binding in &bindings {
        assert!((binding.get_strength() - 0.7).abs() < 0.001);
    }
}

#[test]
fn test_binding_memory_overhead() {
    let index = BindingIndex::with_capacity(1000, 100, 0.1);

    // Add 100 bindings
    for _ in 0..100 {
        let binding = ConceptBinding::new(Uuid::new_v4(), Uuid::new_v4(), 0.8, 0.6);
        index.add_binding(binding);
    }

    let stats = index.memory_stats();
    assert_eq!(stats.total_bindings, 100);

    // Verify overhead is reasonable
    let binding_size = std::mem::size_of::<ConceptBinding>();
    assert_eq!(stats.binding_memory_bytes, 100 * binding_size);

    // Total overhead should be <20% of node storage
    let node_count = 1100; // 1000 episodes + 100 concepts
    let overhead_pct = stats.overhead_percentage(node_count);
    assert!(
        overhead_pct < 20.0,
        "Memory overhead ({overhead_pct:.2}%) exceeds 20% target"
    );
}

#[test]
fn test_binding_age_tracking() {
    use std::time::Duration;

    let binding = ConceptBinding::new(Uuid::new_v4(), Uuid::new_v4(), 0.5, 0.5);

    thread::sleep(Duration::from_millis(10));

    let age = binding.age_since_activation();
    assert!(age.num_milliseconds() >= 10);

    // Update should reset age
    binding.add_activation(0.1);
    thread::sleep(Duration::from_millis(5));

    let new_age = binding.age_since_activation();
    assert!(new_age.num_milliseconds() < age.num_milliseconds());
}

#[test]
fn test_binding_ref_creation() {
    let binding = ConceptBinding::new(Uuid::new_v4(), Uuid::new_v4(), 0.7, 0.6);
    let ref_binding = BindingRef::from_binding(&binding);

    assert_eq!(ref_binding.target_id, binding.concept_id);
    assert!((ref_binding.get_strength() - 0.7).abs() < 0.001);
    assert!((ref_binding.contribution - 0.6).abs() < 0.001);

    // Verify BindingRef is compact
    let ref_size = std::mem::size_of::<BindingRef>();
    assert!(
        ref_size <= 40,
        "BindingRef should be compact (got {ref_size} bytes)"
    );
}

#[test]
fn test_concurrent_index_operations() {
    let index = Arc::new(BindingIndex::new(0.1));
    let con_id = Uuid::new_v4();

    // Spawn 10 threads, each adding 10 bindings
    let threads: Vec<_> = (0..10)
        .map(|_| {
            let idx = Arc::clone(&index);
            let c_id = con_id;
            thread::spawn(move || {
                for _ in 0..10 {
                    let binding = ConceptBinding::new(Uuid::new_v4(), c_id, 0.8, 0.6);
                    idx.add_binding(binding);
                }
            })
        })
        .collect();

    for thread in threads {
        thread.join().expect("Thread panicked");
    }

    assert_eq!(index.count(), 100);

    // Verify all episodes bound to concept
    let episodes = index.get_episodes_for_concept(&con_id);
    assert_eq!(episodes.len(), 100);
}

#[test]
fn test_strength_update_through_index() {
    let index = BindingIndex::new(0.1);
    let ep_id = Uuid::new_v4();
    let con_id = Uuid::new_v4();

    let binding = ConceptBinding::new(ep_id, con_id, 0.5, 0.6);
    index.add_binding(binding);

    // Update through index
    let updated = index.update_binding_strength(&ep_id, &con_id, |s| s + 0.2);
    assert!(updated);

    // Verify new strength
    let strength = index.get_binding_strength(&ep_id, &con_id);
    assert!((strength.unwrap() - 0.7).abs() < 0.001);
}

#[test]
fn test_empty_index_operations() {
    let index = BindingIndex::new(0.1);

    assert_eq!(index.count(), 0);
    assert!(index.get_concepts_for_episode(&Uuid::new_v4()).is_empty());
    assert!(index.get_episodes_for_concept(&Uuid::new_v4()).is_empty());
    assert_eq!(index.garbage_collect(), 0);
}

#[test]
fn test_high_fanout_concept() {
    let index = BindingIndex::new(0.1);
    let con_id = Uuid::new_v4();

    // Create concept with 1000 episode bindings
    for _ in 0..1000 {
        let binding = ConceptBinding::new(Uuid::new_v4(), con_id, 0.8, 0.6);
        index.add_binding(binding);
    }

    assert_eq!(index.count(), 1000);

    // Verify retrieval is fast (Arc cloning, not deep copy)
    let start = std::time::Instant::now();
    let episodes = index.get_episodes_for_concept(&con_id);
    let elapsed = start.elapsed();

    assert_eq!(episodes.len(), 1000);
    assert!(
        elapsed.as_micros() < 1000,
        "High fan-out retrieval too slow: {elapsed:?}"
    );
}

#[test]
fn test_memory_stats_accuracy() {
    let index = BindingIndex::new(0.1);

    // Add known number of bindings
    for _ in 0..50 {
        let binding = ConceptBinding::new(Uuid::new_v4(), Uuid::new_v4(), 0.8, 0.6);
        index.add_binding(binding);
    }

    let stats = index.memory_stats();

    // Verify count matches
    assert_eq!(stats.total_bindings, 50);
    assert_eq!(index.count(), 50);

    // Verify memory calculations are consistent
    let expected_binding_memory = 50 * std::mem::size_of::<ConceptBinding>();
    assert_eq!(stats.binding_memory_bytes, expected_binding_memory);
}
