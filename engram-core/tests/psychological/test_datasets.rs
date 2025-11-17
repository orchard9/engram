use chrono::{DateTime, Duration, Utc};
use engram_core::memory::types::Episode;
use engram_core::{Confidence, EMBEDDING_DIM};
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

/// Stimulus describing Anderson (1974) person-location materials
pub struct FanEffectStimulus {
    pub person: &'static str,
    pub fan: usize,
    pub empirical_rt_ms: f32,
    pub locations: &'static [&'static str],
}

/// Static list of stimulus sentences from Anderson (1974)
pub fn fan_effect_stimuli() -> &'static [FanEffectStimulus] {
    const DATA: &[FanEffectStimulus] = &[
        FanEffectStimulus {
            person: "doctor",
            fan: 1,
            empirical_rt_ms: 1159.0,
            locations: &["park"],
        },
        FanEffectStimulus {
            person: "fireman",
            fan: 2,
            empirical_rt_ms: 1236.0,
            locations: &["bank", "store"],
        },
        FanEffectStimulus {
            person: "lawyer",
            fan: 3,
            empirical_rt_ms: 1305.0,
            locations: &["church", "park", "store"],
        },
    ];
    DATA
}

/// Descriptor for Rosch (1975) bird exemplars
pub struct BirdExemplar {
    pub name: &'static str,
    pub features: &'static [&'static str],
    pub typicality: f32,
}

/// Prototypical and atypical bird exemplars with feature sets
pub fn bird_exemplars() -> Vec<BirdExemplar> {
    const TYPICAL: [&str; 5] = [
        "flies",
        "sings",
        "builds_nests",
        "small",
        "perches_in_trees",
    ];
    const ATYPICAL: [&str; 5] = ["swims", "large", "flightless", "lives_in_cold", "eats_fish"];

    vec![
        BirdExemplar {
            name: "robin",
            features: &TYPICAL,
            typicality: 0.95,
        },
        BirdExemplar {
            name: "sparrow",
            features: &TYPICAL,
            typicality: 0.93,
        },
        BirdExemplar {
            name: "bluebird",
            features: &TYPICAL,
            typicality: 0.92,
        },
        BirdExemplar {
            name: "penguin",
            features: &ATYPICAL,
            typicality: 0.45,
        },
        BirdExemplar {
            name: "ostrich",
            features: &ATYPICAL,
            typicality: 0.42,
        },
        BirdExemplar {
            name: "emu",
            features: &ATYPICAL,
            typicality: 0.40,
        },
    ]
}

/// Word pairs used for semantic priming validation
pub struct SemanticPrimingPair {
    pub prime: &'static str,
    pub target: &'static str,
    pub is_related: bool,
}

pub fn semantic_priming_pairs() -> Vec<SemanticPrimingPair> {
    vec![
        SemanticPrimingPair {
            prime: "doctor",
            target: "nurse",
            is_related: true,
        },
        SemanticPrimingPair {
            prime: "bread",
            target: "butter",
            is_related: true,
        },
        SemanticPrimingPair {
            prime: "lion",
            target: "tiger",
            is_related: true,
        },
        SemanticPrimingPair {
            prime: "chair",
            target: "table",
            is_related: true,
        },
        SemanticPrimingPair {
            prime: "doctor",
            target: "butter",
            is_related: false,
        },
        SemanticPrimingPair {
            prime: "bread",
            target: "tiger",
            is_related: false,
        },
        SemanticPrimingPair {
            prime: "lion",
            target: "table",
            is_related: false,
        },
    ]
}

/// Word list reused in context-dependent memory tests
pub fn context_word_list() -> Vec<&'static str> {
    vec![
        "pencil", "clock", "mountain", "river", "book", "window", "ocean", "forest", "cloud",
        "garden",
    ]
}

/// Create deterministic embedding from textual label
pub fn embedding_for_label(label: &str) -> [f32; EMBEDDING_DIM] {
    let mut seed = deterministic_seed(label);
    let mut embedding = [0.0_f32; EMBEDDING_DIM];

    for value in &mut embedding {
        // Xorshift64*
        seed ^= seed >> 12;
        seed ^= seed << 25;
        seed ^= seed >> 27;
        seed = seed.wrapping_mul(0x2545_F491_4F6C_DD1D);
        let normalized = ((seed >> 32) as u32) as f32 / u32::MAX as f32;
        *value = normalized * 2.0 - 1.0;
    }

    normalize(embedding)
}

/// Blend two embeddings with the provided weight
pub fn blend_embeddings(
    a: &[f32; EMBEDDING_DIM],
    b: &[f32; EMBEDDING_DIM],
    weight: f32,
) -> [f32; EMBEDDING_DIM] {
    let w = weight.clamp(0.0, 1.0);
    let a_w = 1.0 - w;
    let mut blended = [0.0_f32; EMBEDDING_DIM];
    for i in 0..EMBEDDING_DIM {
        blended[i] = a[i] * a_w + b[i] * w;
    }
    normalize(blended)
}

/// Create an episode using deterministic embeddings and feature-driven weighting
pub fn create_episode(
    name: &str,
    features: &[&str],
    typicality: f32,
    offset_hours: i64,
) -> Episode {
    let mut aggregate = [0.0_f32; EMBEDDING_DIM];
    for feature in features {
        let feature_embedding = embedding_for_label(feature);
        for (slot, value) in aggregate.iter_mut().zip(feature_embedding.iter()) {
            *slot += *value;
        }
    }
    let aggregate = normalize(aggregate);
    let schema_anchor = embedding_for_label("bird_schema_anchor");
    let atypical_anchor = embedding_for_label("outlier_bird_anchor");
    let blended_anchor = blend_embeddings(
        &schema_anchor,
        &atypical_anchor,
        (1.0 - typicality).clamp(0.0, 1.0),
    );
    let embedding = blend_embeddings(&aggregate, &blended_anchor, 0.5);

    let importance = typicality.clamp(0.3, 0.99);
    let timestamp = base_timestamp() + Duration::hours(offset_hours);
    Episode::new(
        format!("{name}_episode"),
        timestamp,
        format!("{name} observation"),
        embedding,
        Confidence::exact(importance),
    )
}

/// Generate embedding pair for semantic priming conditions
pub fn semantic_pair_embeddings(
    pair: &SemanticPrimingPair,
) -> ([f32; EMBEDDING_DIM], [f32; EMBEDDING_DIM]) {
    let prime = embedding_for_label(pair.prime);
    let target = embedding_for_label(pair.target);

    if pair.is_related {
        (prime, blend_embeddings(&target, &prime, 0.65))
    } else {
        (prime, blend_embeddings(&target, &prime, 0.05))
    }
}

/// Deterministic context embeddings for Godden & Baddeley (1975)
pub fn land_context_embedding() -> [f32; EMBEDDING_DIM] {
    embedding_for_label("land_environment_features")
}

pub fn water_context_embedding() -> [f32; EMBEDDING_DIM] {
    embedding_for_label("underwater_environment_features")
}

/// Deterministic timestamp reference
fn base_timestamp() -> DateTime<Utc> {
    DateTime::from_timestamp(1_700_000_000, 0).expect("valid reference timestamp")
}

fn deterministic_seed(label: &str) -> u64 {
    let mut hasher = DefaultHasher::new();
    label.hash(&mut hasher);
    hasher.finish()
}

fn normalize(mut embedding: [f32; EMBEDDING_DIM]) -> [f32; EMBEDDING_DIM] {
    let norm = embedding
        .iter()
        .map(|v| v * v)
        .sum::<f32>()
        .sqrt()
        .max(1e-6);
    for value in &mut embedding {
        *value /= norm;
    }
    embedding
}
