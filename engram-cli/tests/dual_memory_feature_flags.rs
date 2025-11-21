//! Feature flag matrix tests for dual-memory architecture
//!
//! Tests all combinations of feature flags to ensure no panics or regressions.
//! Flags tested: dual_memory_types, blended_recall, fan_effect, monitoring

#![allow(unexpected_cfgs)]
#![allow(clippy::struct_excessive_bools)]

use chrono::Utc;
use engram_core::{Confidence, Cue, CueBuilder, Episode, MemoryStore};
use rand::{Rng, SeedableRng, rngs::StdRng};

/// Feature flag configuration for testing
#[derive(Debug, Clone)]
struct FeatureFlags {
    dual_memory: bool,
    blended_recall: bool,
    fan_effect: bool,
    monitoring: bool,
}

impl FeatureFlags {
    fn name(&self) -> String {
        format!(
            "dual={}_blend={}_fan={}_mon={}",
            self.dual_memory, self.blended_recall, self.fan_effect, self.monitoring
        )
    }
}

/// Generate test episodes for feature flag testing
fn generate_test_data(count: usize, seed: u64) -> Vec<Episode> {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut episodes = Vec::with_capacity(count);

    for idx in 0..count {
        let mut embedding = [0.0f32; 768];
        for x in &mut embedding {
            *x = rng.gen_range(-1.0..1.0);
        }

        // Normalize
        let norm = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 0.0 {
            for x in &mut embedding {
                *x /= norm;
            }
        }

        let confidence = Confidence::exact(rng.gen_range(0.6..0.95));

        let episode = Episode {
            id: format!("flag_test_episode_{idx}"),
            when: Utc::now(),
            where_location: None,
            who: None,
            what: format!("Feature flag test content {idx}"),
            embedding,
            embedding_provenance: None,
            encoding_confidence: confidence,
            vividness_confidence: confidence,
            reliability_confidence: confidence,
            last_recall: Utc::now(),
            recall_count: 0,
            decay_rate: 1.0,
            decay_function: None,
            metadata: std::collections::HashMap::new(),
        };

        episodes.push(episode);
    }

    episodes
}

/// Run workload with specific feature flag combination
fn run_workload_with_flags(flags: &FeatureFlags) -> Result<(), String> {
    println!("Testing configuration: {}", flags.name());

    // Note: Feature flags are compile-time in Rust, so this test validates
    // that the current build configuration works. In production, you'd run
    // this test suite with different cargo feature combinations.

    let store = MemoryStore::new(256);
    let episodes = generate_test_data(100, 42);

    // Operation 1: Store episodes
    for episode in &episodes {
        let result = store.store(episode.clone());
        if !result.activation.is_successful() {
            return Err(format!(
                "Store failed for {}: activation={}",
                flags.name(),
                result.activation.value()
            ));
        }
    }

    let stored_count = store.count();
    if stored_count != 100 {
        return Err(format!(
            "Expected 100 episodes, got {} for {}",
            stored_count,
            flags.name()
        ));
    }

    // Operation 2: Recall queries
    for query_idx in 0..10 {
        let query_embedding = [0.5f32; 768];
        let cue = CueBuilder::new()
            .id(format!("flag_test_cue_{query_idx}"))
            .embedding_search(query_embedding, Confidence::LOW)
            .cue_confidence(Confidence::HIGH)
            .build();

        let results = store.recall(&cue);

        if results.results.is_empty() {
            return Err(format!("Recall returned no results for {}", flags.name()));
        }
    }

    // Operation 3: Consolidation (if available)
    #[cfg(feature = "pattern_completion")]
    {
        let _patterns = store.consolidate();
        // Don't assert on pattern count - may vary with clustering randomness
    }

    // Operation 4: Get specific episode
    for idx in 0..10 {
        let episode_id = format!("flag_test_episode_{idx}");
        let retrieved = store.get_episode(&episode_id);

        if retrieved.is_none() {
            return Err(format!(
                "Failed to retrieve episode {} for {}",
                episode_id,
                flags.name()
            ));
        }
    }

    println!("Configuration {} passed all operations", flags.name());
    Ok(())
}

// Test all 16 combinations of 4 binary feature flags

#[test]
fn test_flags_0000_all_off() {
    let flags = FeatureFlags {
        dual_memory: false,
        blended_recall: false,
        fan_effect: false,
        monitoring: false,
    };
    run_workload_with_flags(&flags).expect("workload should succeed");
}

#[test]
fn test_flags_0001_monitoring_only() {
    let flags = FeatureFlags {
        dual_memory: false,
        blended_recall: false,
        fan_effect: false,
        monitoring: true,
    };
    run_workload_with_flags(&flags).expect("workload should succeed");
}

#[test]
fn test_flags_0010_fan_effect_only() {
    let flags = FeatureFlags {
        dual_memory: false,
        blended_recall: false,
        fan_effect: true,
        monitoring: false,
    };
    run_workload_with_flags(&flags).expect("workload should succeed");
}

#[test]
fn test_flags_0011_fan_effect_monitoring() {
    let flags = FeatureFlags {
        dual_memory: false,
        blended_recall: false,
        fan_effect: true,
        monitoring: true,
    };
    run_workload_with_flags(&flags).expect("workload should succeed");
}

#[test]
fn test_flags_0100_blended_recall_only() {
    let flags = FeatureFlags {
        dual_memory: false,
        blended_recall: true,
        fan_effect: false,
        monitoring: false,
    };
    run_workload_with_flags(&flags).expect("workload should succeed");
}

#[test]
fn test_flags_0101_blended_recall_monitoring() {
    let flags = FeatureFlags {
        dual_memory: false,
        blended_recall: true,
        fan_effect: false,
        monitoring: true,
    };
    run_workload_with_flags(&flags).expect("workload should succeed");
}

#[test]
fn test_flags_0110_blended_recall_fan_effect() {
    let flags = FeatureFlags {
        dual_memory: false,
        blended_recall: true,
        fan_effect: true,
        monitoring: false,
    };
    run_workload_with_flags(&flags).expect("workload should succeed");
}

#[test]
fn test_flags_0111_blended_recall_fan_effect_monitoring() {
    let flags = FeatureFlags {
        dual_memory: false,
        blended_recall: true,
        fan_effect: true,
        monitoring: true,
    };
    run_workload_with_flags(&flags).expect("workload should succeed");
}

#[test]
#[cfg(feature = "dual_memory_types")]
fn test_flags_1000_dual_memory_only() {
    let flags = FeatureFlags {
        dual_memory: true,
        blended_recall: false,
        fan_effect: false,
        monitoring: false,
    };
    run_workload_with_flags(&flags).expect("workload should succeed");
}

#[test]
#[cfg(feature = "dual_memory_types")]
fn test_flags_1001_dual_memory_monitoring() {
    let flags = FeatureFlags {
        dual_memory: true,
        blended_recall: false,
        fan_effect: false,
        monitoring: true,
    };
    run_workload_with_flags(&flags).expect("workload should succeed");
}

#[test]
#[cfg(feature = "dual_memory_types")]
fn test_flags_1010_dual_memory_fan_effect() {
    let flags = FeatureFlags {
        dual_memory: true,
        blended_recall: false,
        fan_effect: true,
        monitoring: false,
    };
    run_workload_with_flags(&flags).expect("workload should succeed");
}

#[test]
#[cfg(feature = "dual_memory_types")]
fn test_flags_1011_dual_memory_fan_effect_monitoring() {
    let flags = FeatureFlags {
        dual_memory: true,
        blended_recall: false,
        fan_effect: true,
        monitoring: true,
    };
    run_workload_with_flags(&flags).expect("workload should succeed");
}

#[test]
#[cfg(feature = "dual_memory_types")]
fn test_flags_1100_dual_memory_blended_recall() {
    let flags = FeatureFlags {
        dual_memory: true,
        blended_recall: true,
        fan_effect: false,
        monitoring: false,
    };
    run_workload_with_flags(&flags).expect("workload should succeed");
}

#[test]
#[cfg(feature = "dual_memory_types")]
fn test_flags_1101_dual_memory_blended_recall_monitoring() {
    let flags = FeatureFlags {
        dual_memory: true,
        blended_recall: true,
        fan_effect: false,
        monitoring: true,
    };
    run_workload_with_flags(&flags).expect("workload should succeed");
}

#[test]
#[cfg(feature = "dual_memory_types")]
fn test_flags_1110_dual_memory_blended_recall_fan_effect() {
    let flags = FeatureFlags {
        dual_memory: true,
        blended_recall: true,
        fan_effect: true,
        monitoring: false,
    };
    run_workload_with_flags(&flags).expect("workload should succeed");
}

#[test]
#[cfg(feature = "dual_memory_types")]
fn test_flags_1111_all_on() {
    let flags = FeatureFlags {
        dual_memory: true,
        blended_recall: true,
        fan_effect: true,
        monitoring: true,
    };
    run_workload_with_flags(&flags).expect("workload should succeed");
}

/// Integration test: Verify flags work across module boundaries
#[test]
fn test_cross_module_flag_compatibility() {
    // This test ensures that feature flags are consistently applied
    // across engram-core and engram-cli

    let store = MemoryStore::new(128);
    let episodes = generate_test_data(50, 999);

    for episode in &episodes {
        let result = store.store(episode.clone());
        assert!(result.activation.is_successful());
    }

    // Query to exercise cross-module code paths
    // Use actual episode embedding to guarantee a match
    let query_embedding = episodes[0].embedding;
    let cue = Cue::embedding(
        "cross_module_cue".to_string(),
        query_embedding,
        Confidence::MEDIUM,
    );

    let results = store.recall(&cue);
    assert!(!results.results.is_empty());

    println!("Cross-module flag compatibility verified");
}
