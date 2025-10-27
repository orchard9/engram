//! Concurrent cognitive operations safety tests
//!
//! Validates that all cognitive patterns work correctly under heavy concurrent load:
//! - No data races or lost updates
//! - No deadlocks or livelocks
//! - Consistent state maintained across all operations
//! - Metrics tracking remains accurate under concurrency

use engram_core::cognitive::priming::{
    AssociativePrimingEngine, RepetitionPrimingEngine, SemanticPrimingEngine,
};
use engram_core::cognitive::reconsolidation::{
    EpisodeModifications, ModificationType, ReconsolidationEngine,
};
use engram_core::{Confidence, Cue, Episode, MemoryStore};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::thread;

use chrono::Utc;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;

// Import shared test utilities (not needed - create_test_episode uses random_embedding)

// ==================== Heavy Concurrent Load Test ====================

#[test]
fn test_no_conflicts_between_concurrent_cognitive_systems() {
    // Spawn 16 threads (oversubscribed) to stress-test concurrency
    // Each thread performs random cognitive operations

    let semantic_engine = Arc::new(SemanticPrimingEngine::new());
    let associative_engine = Arc::new(AssociativePrimingEngine::new());
    let repetition_engine = Arc::new(RepetitionPrimingEngine::new());
    let reconsolidation_engine = Arc::new(ReconsolidationEngine::new());
    let store = Arc::new(MemoryStore::new(10000));

    // Pre-populate store with episodes
    for i in 0..100 {
        let episode = create_test_episode(i, 48);
        let _ = store.store(episode);
    }

    let handles: Vec<_> = (0..16)
        .map(|thread_id| {
            let semantic = Arc::clone(&semantic_engine);
            let associative = Arc::clone(&associative_engine);
            let repetition = Arc::clone(&repetition_engine);
            let reconsolidation = Arc::clone(&reconsolidation_engine);
            let store_ref = Arc::clone(&store);

            thread::spawn(move || {
                let mut rng = ChaCha8Rng::seed_from_u64(thread_id);

                for _ in 0..10_000 {
                    match rng.gen_range(0..6) {
                        0 => {
                            // Semantic priming
                            let concept = format!("concept_{}", rng.gen_range(0..50));
                            let embedding = random_embedding(&mut rng);
                            semantic.activate_priming(&concept, &embedding, Vec::new);
                        }
                        1 => {
                            // Associative priming
                            let node_a = format!("node_{}", rng.gen_range(0..100));
                            let node_b = format!("node_{}", rng.gen_range(0..100));
                            associative.record_coactivation(&node_a, &node_b);
                        }
                        2 => {
                            // Repetition priming
                            let node_id = format!("episode_{}", rng.gen_range(0..100));
                            repetition.record_exposure(&node_id);
                        }
                        3 => {
                            // Reconsolidation check
                            let episode_id = format!("episode_{}", rng.gen_range(0..100));
                            if let Some(episode) = store_ref.get_episode(&episode_id) {
                                let recall_time = Utc::now();
                                reconsolidation.record_recall(&episode, recall_time, true);
                            }
                        }
                        4 => {
                            // Recall operation
                            let embedding = random_embedding(&mut rng);
                            let cue =
                                Cue::embedding("test".to_string(), embedding, Confidence::HIGH);
                            let _ = store_ref.recall(&cue);
                        }
                        _ => {
                            // Store operation
                            let episode = create_test_episode(rng.gen_range(100..200), 0);
                            let _ = store_ref.store(episode);
                        }
                    }
                }
            })
        })
        .collect();

    // All threads must complete without panic
    for h in handles {
        h.join()
            .expect("Thread panicked during concurrent cognitive operations test");
    }

    // Verify: Store is still accessible (no deadlocks)
    let final_count = store.count();
    assert!(
        final_count >= 100,
        "Store should contain at least initial 100 episodes, got {final_count}"
    );
}

// ==================== Metrics Accuracy Under Concurrency ====================

#[cfg(feature = "monitoring")]
#[test]
fn test_metrics_track_all_events_no_lost_updates() {
    use engram_core::metrics::cognitive_patterns::{CognitivePatternMetrics, PrimingType};

    let metrics = Arc::new(CognitivePatternMetrics::new());
    let expected_priming = Arc::new(AtomicU64::new(0));
    let expected_interference = Arc::new(AtomicU64::new(0));
    let expected_reconsolidation = Arc::new(AtomicU64::new(0));

    let handles: Vec<_> = (0..8)
        .map(|thread_id| {
            let m = Arc::clone(&metrics);
            let exp_p = Arc::clone(&expected_priming);
            let exp_i = Arc::clone(&expected_interference);
            let exp_r = Arc::clone(&expected_reconsolidation);

            thread::spawn(move || {
                for i in 0..1000 {
                    match i % 3 {
                        0 => {
                            let priming_type = match thread_id % 3 {
                                0 => PrimingType::Semantic,
                                1 => PrimingType::Associative,
                                _ => PrimingType::Repetition,
                            };
                            m.record_priming(priming_type, 0.5);
                            exp_p.fetch_add(1, Ordering::Relaxed);
                        }
                        1 => {
                            use engram_core::metrics::cognitive_patterns::InterferenceType;
                            m.record_interference(InterferenceType::Proactive, 0.6);
                            exp_i.fetch_add(1, Ordering::Relaxed);
                        }
                        _ => {
                            m.record_reconsolidation(0.5);
                            exp_r.fetch_add(1, Ordering::Relaxed);
                        }
                    }
                }
            })
        })
        .collect();

    for h in handles {
        h.join().unwrap();
    }

    // Verify: All events recorded (no lost updates)
    let actual_priming = metrics.priming_events_total();
    let expected_priming_val = expected_priming.load(Ordering::Acquire);
    assert_eq!(
        actual_priming, expected_priming_val,
        "Lost priming updates: expected {expected_priming_val}, got {actual_priming}"
    );

    let actual_interference = metrics.interference_detections_total();
    let expected_interference_val = expected_interference.load(Ordering::Acquire);
    assert_eq!(
        actual_interference, expected_interference_val,
        "Lost interference updates: expected {expected_interference_val}, got {actual_interference}"
    );

    let actual_reconsolidation = metrics.reconsolidation_events_total();
    let expected_reconsolidation_val = expected_reconsolidation.load(Ordering::Acquire);
    assert_eq!(
        actual_reconsolidation, expected_reconsolidation_val,
        "Lost reconsolidation updates: expected {expected_reconsolidation_val}, got {actual_reconsolidation}"
    );
}

// ==================== Priming Concurrent Activation ====================

#[test]
fn test_concurrent_semantic_priming_no_race_conditions() {
    let semantic_engine = Arc::new(SemanticPrimingEngine::new());

    // Multiple threads activate priming for different concepts simultaneously
    let handles: Vec<_> = (0..8)
        .map(|thread_id| {
            let engine = Arc::clone(&semantic_engine);

            thread::spawn(move || {
                let mut rng = ChaCha8Rng::seed_from_u64(thread_id);

                for i in 0..1000 {
                    let concept = format!("concept_{thread_id}_{i}");
                    let embedding = random_embedding(&mut rng);

                    engine.activate_priming(&concept, &embedding, || {
                        vec![(
                            format!("related_{thread_id}_{i}"),
                            random_embedding(&mut rng),
                            1,
                        )]
                    });

                    // Also check priming boost concurrently
                    let _ = engine.compute_priming_boost(&concept);
                }
            })
        })
        .collect();

    for h in handles {
        h.join()
            .expect("Thread panicked during concurrent semantic priming");
    }

    // Verify: No panics occurred (implicit success)
}

// ==================== Reconsolidation Concurrent Window Tracking ====================

#[test]
fn test_concurrent_reconsolidation_window_tracking() {
    let engine = Arc::new(ReconsolidationEngine::new());

    // Multiple threads recording recalls and checking reconsolidation eligibility
    let handles: Vec<_> = (0..8)
        .map(|thread_id| {
            let reconsolidation = Arc::clone(&engine);

            thread::spawn(move || {
                for i in 0..1000 {
                    let episode_id = format!("episode_{}_{}", thread_id, i % 100);
                    let episode = create_test_episode(i as u64, 48);

                    // Record recall
                    let recall_time = Utc::now();
                    reconsolidation.record_recall(&episode, recall_time, true);

                    // Attempt reconsolidation to test eligibility
                    let modifications = EpisodeModifications {
                        what: Some("modified".to_string()),
                        where_location: None,
                        who: None,
                        modification_extent: 0.3,
                        modification_type: ModificationType::Update,
                    };
                    let attempt_time = recall_time + chrono::Duration::hours(3);
                    let _ = reconsolidation.attempt_reconsolidation(
                        &episode_id,
                        &modifications,
                        attempt_time,
                    );
                }
            })
        })
        .collect();

    for h in handles {
        h.join()
            .expect("Thread panicked during concurrent reconsolidation tracking");
    }

    // Verify: No panics or deadlocks
}

// ==================== Associative Priming Concurrent Co-activation ====================

#[test]
fn test_concurrent_associative_priming_coactivation() {
    let engine = Arc::new(AssociativePrimingEngine::new());

    // Multiple threads recording co-activations
    let handles: Vec<_> = (0..8)
        .map(|_thread_id| {
            let associative = Arc::clone(&engine);

            thread::spawn(move || {
                for i in 0..1000 {
                    let node_a = format!("node_{}", i % 50);
                    let node_b = format!("node_{}", (i + 1) % 50);

                    associative.record_coactivation(&node_a, &node_b);

                    // Concurrently compute association strength
                    let _ = associative.compute_association_strength(&node_a, &node_b);
                }
            })
        })
        .collect();

    for h in handles {
        h.join()
            .expect("Thread panicked during concurrent associative priming");
    }

    // Verify: Association strengths are non-negative and valid
    let strength = engine.compute_association_strength("node_0", "node_1");
    assert!(
        (0.0..=1.0).contains(&strength),
        "Association strength should be in [0, 1]: got {strength}"
    );
}

// ==================== Mixed Operation Stress Test ====================

#[test]
fn test_mixed_cognitive_operations_stress_test() {
    // Extremely high concurrency: 32 threads, 50K operations each
    // This test is designed to find any remaining race conditions

    let semantic = Arc::new(SemanticPrimingEngine::new());
    let associative = Arc::new(AssociativePrimingEngine::new());
    let repetition = Arc::new(RepetitionPrimingEngine::new());
    let reconsolidation = Arc::new(ReconsolidationEngine::new());
    let store = Arc::new(MemoryStore::new(10000));

    let panic_count = Arc::new(AtomicU64::new(0));

    let handles: Vec<_> = (0..32)
        .map(|thread_id| {
            let sem = Arc::clone(&semantic);
            let assoc = Arc::clone(&associative);
            let rep = Arc::clone(&repetition);
            let recon = Arc::clone(&reconsolidation);
            let st = Arc::clone(&store);
            let _pc = Arc::clone(&panic_count);

            thread::spawn(move || {
                let mut rng = ChaCha8Rng::seed_from_u64(thread_id);

                for _ in 0..50_000 {
                    // Randomly choose operation
                    match rng.gen_range(0..10) {
                        0 => {
                            let concept = format!("c_{}", rng.gen_range(0..100));
                            let emb = random_embedding(&mut rng);
                            sem.activate_priming(&concept, &emb, Vec::new);
                        }
                        1 => {
                            let _ =
                                sem.compute_priming_boost(&format!("c_{}", rng.gen_range(0..100)));
                        }
                        2 => {
                            let a = format!("n_{}", rng.gen_range(0..100));
                            let b = format!("n_{}", rng.gen_range(0..100));
                            assoc.record_coactivation(&a, &b);
                        }
                        3 => {
                            let a = format!("n_{}", rng.gen_range(0..100));
                            let b = format!("n_{}", rng.gen_range(0..100));
                            let _ = assoc.compute_association_strength(&a, &b);
                        }
                        4 => {
                            rep.record_exposure(&format!("e_{}", rng.gen_range(0..100)));
                        }
                        5 => {
                            let _ = rep
                                .compute_repetition_boost(&format!("e_{}", rng.gen_range(0..100)));
                        }
                        6 => {
                            let ep = create_test_episode(rng.gen_range(0..100), 48);
                            recon.record_recall(&ep, Utc::now(), true);
                        }
                        7 => {
                            let ep = create_test_episode(rng.gen_range(0..100), 48);
                            let recall_time = Utc::now();
                            recon.record_recall(&ep, recall_time, true);
                            let modifications = EpisodeModifications {
                                what: Some("modified".to_string()),
                                where_location: None,
                                who: None,
                                modification_extent: 0.3,
                                modification_type: ModificationType::Update,
                            };
                            let attempt_time = recall_time + chrono::Duration::hours(3);
                            let _ =
                                recon.attempt_reconsolidation(&ep.id, &modifications, attempt_time);
                        }
                        8 => {
                            let ep = create_test_episode(rng.gen_range(0..200), 0);
                            let _ = st.store(ep);
                        }
                        _ => {
                            let emb = random_embedding(&mut rng);
                            let cue = Cue::embedding("test".to_string(), emb, Confidence::HIGH);
                            let _ = st.recall(&cue);
                        }
                    }
                }
            })
        })
        .collect();

    // All threads must complete
    for h in handles {
        if h.join().is_err() {
            panic_count.fetch_add(1, Ordering::Relaxed);
        }
    }

    let panics = panic_count.load(Ordering::Acquire);
    assert_eq!(panics, 0, "Stress test had {panics} thread panics");
}

// ==================== Helper Functions ====================

fn create_test_episode(id: u64, age_hours: i64) -> Episode {
    let when = Utc::now() - chrono::Duration::hours(age_hours);
    // Use unique RNG seed for each episode to ensure diverse embeddings
    // This prevents deduplication from treating all episodes as duplicates
    let mut rng = ChaCha8Rng::seed_from_u64(id);
    let embedding = random_embedding(&mut rng);

    Episode::new(
        format!("episode_{id}"),
        when,
        format!("test content {id}"),
        embedding,
        Confidence::HIGH,
    )
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
