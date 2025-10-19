//! Skeleton for long-running consolidation regression coverage.
//! These tests are ignored by default to avoid CI flakiness until
//! the load harness is implemented.

#[cfg(feature = "pattern_completion")]
use chrono::{Duration, Utc};
#[cfg(feature = "pattern_completion")]
use engram_core::{Confidence, MemoryStore, completion::SchedulerConfig};
#[cfg(feature = "pattern_completion")]
use rand::{Rng, SeedableRng, rngs::StdRng};

/// TODO: Replace with an hour-long soak harness once the consolidation
/// module boundary lands (see docs/architecture/consolidation_boundary.md).
#[cfg(feature = "pattern_completion")]
#[test]
#[ignore = "load harness not implemented"]
fn consolidation_scheduler_soak_placeholder() {
    let store = MemoryStore::new(2048);
    let scheduler_config = SchedulerConfig {
        consolidation_interval_secs: 60,
        min_episodes_threshold: 64,
        max_episodes_per_run: 256,
        enabled: true,
    };

    seed_store_with_mock_episodes(&store, 512, 42);

    // TODO: spin up scheduler with mocked shutdown channel, await multiple runs,
    // and capture metric snapshots + belief logs over ≥1h. The scaffolding here
    // establishes the entrypoint without executing the heavy soak today.
    drop((store, scheduler_config));
}

#[cfg(not(feature = "pattern_completion"))]
#[test]
#[ignore]
fn consolidation_scheduler_soak_placeholder_noop() {}

#[cfg(feature = "pattern_completion")]
fn seed_store_with_mock_episodes(store: &MemoryStore, count: usize, seed: u64) {
    use engram_core::memory::EpisodeBuilder;

    let mut rng = StdRng::seed_from_u64(seed);
    for idx in 0..count {
        let embedding = create_embedding(&mut rng, idx as f32);
        let confidence = Confidence::exact(rng.gen_range(0.6..0.99));
        let episode = EpisodeBuilder::new()
            .id(format!("soak_ep_{idx}"))
            .when(Utc::now() - Duration::minutes(rng.gen_range(0..480)))
            .what(format!("seeded episode {idx}"))
            .embedding(embedding)
            .confidence(confidence)
            .build();

        store.store(episode);
    }

    // sanity check: ensure we actually stored the intended number of episodes
    assert!(store.count() >= count);
}

#[cfg(feature = "pattern_completion")]
fn create_embedding(rng: &mut StdRng, base: f32) -> [f32; 768] {
    let mut embedding = [0.0f32; 768];
    for (idx, slot) in embedding.iter_mut().enumerate() {
        let phase = base + idx as f32 * 0.013;
        let noise = rng.gen_range(-0.05..0.05);
        *slot = (phase.sin() + noise).clamp(-1.0, 1.0);
    }

    // normalize to unit vector to keep confidence calculations predictable
    let norm = embedding.iter().map(|v| v * v).sum::<f32>().sqrt();
    if norm > 0.0 {
        for value in &mut embedding {
            *value /= norm;
        }
    }
    embedding
}
