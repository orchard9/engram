//! Dual-memory confidence rules for episodic ↔ semantic propagation.
//!
//! The `DualMemoryConfidence` helper encapsulates the heuristics used across the
//! engine to keep probability mass bounded while activation flows between
//! hippocampal (episodic) and neocortical (semantic) representations.

use crate::{Confidence, Episode};

/// Parameters controlling how confidence propagates through dual-memory flows.
#[derive(Debug, Clone)]
pub struct DualMemoryConfidence {
    concept_formation_penalty: f32,
    binding_confidence_decay: f32,
    blend_confidence_bonus: f32,
}

impl DualMemoryConfidence {
    /// Create a new dual-memory confidence model with explicit parameters.
    #[must_use]
    pub const fn new(
        concept_formation_penalty: f32,
        binding_confidence_decay: f32,
        blend_confidence_bonus: f32,
    ) -> Self {
        Self {
            concept_formation_penalty,
            binding_confidence_decay,
            blend_confidence_bonus,
        }
    }

    /// Compute semantic concept confidence from clustered episodes.
    ///
    /// Returns [`Confidence::NONE`] if no episodes are provided.
    #[must_use]
    pub fn concept_confidence(&self, episodes: &[Episode], coherence: f32) -> Confidence {
        if episodes.is_empty() {
            return Confidence::NONE;
        }

        let coherence = coherence.clamp(0.0, 1.0);
        let aggregate = episodes
            .iter()
            .map(|episode| {
                // Average the three confidence channels to capture both vividness
                // and reliability signals before applying any semantic penalty.
                let encoding = episode.encoding_confidence.raw();
                let reliability = episode.reliability_confidence.raw();
                let vividness = episode.vividness_confidence.raw();
                (encoding + reliability + vividness) / 3.0
            })
            .sum::<f32>();

        #[allow(clippy::cast_precision_loss)]
        let avg = aggregate / episodes.len() as f32;
        let concept_conf = avg * self.concept_formation_penalty * coherence;
        Confidence::from_raw(concept_conf)
    }

    /// Propagate confidence through a binding, ensuring monotonic attenuation.
    #[must_use]
    pub fn propagate_through_binding(
        &self,
        source_confidence: Confidence,
        binding_strength: f32,
    ) -> Confidence {
        let clamped_strength = binding_strength.clamp(0.0, 1.0);
        let propagated = source_confidence.raw() * clamped_strength * self.binding_confidence_decay;
        Confidence::from_raw(propagated)
    }

    /// Blend episodic and semantic confidence with convergent evidence bonus.
    #[must_use]
    pub fn blend_confidence(
        &self,
        episodic_conf: Confidence,
        semantic_conf: Confidence,
        episodic_weight: f32,
        semantic_weight: f32,
    ) -> Confidence {
        let safe_ep_weight = episodic_weight.max(0.0);
        let safe_sem_weight = semantic_weight.max(0.0);
        let base = episodic_conf
            .combine_weighted(semantic_conf, safe_ep_weight, safe_sem_weight)
            .raw();

        let max_input = episodic_conf.raw().max(semantic_conf.raw());
        let convergent = episodic_conf.raw() > 0.5 && semantic_conf.raw() > 0.5;
        let blended = if convergent {
            (base * self.blend_confidence_bonus).min(max_input * self.blend_confidence_bonus)
        } else {
            base
        };
        Confidence::from_raw(blended)
    }

    /// Validate that repeated propagation decays exponentially.
    #[must_use]
    pub fn verify_multi_hop_decay(&self, hops: u32) -> bool {
        let initial = Confidence::exact(0.9);
        let mut current = initial;
        for _ in 0..hops {
            current = self.propagate_through_binding(current, 0.8);
        }

        let decay_factor = 0.8 * self.binding_confidence_decay;
        let expected_max = initial.raw() * decay_factor.powi(hops as i32);
        current.raw() <= expected_max + f32::EPSILON
    }

    /// Estimate overall confidence for an individual episode.
    #[must_use]
    pub fn calculate_episode_confidence(&self, episode: &Episode) -> Confidence {
        let encoding = episode.encoding_confidence.raw();
        let reliability = episode.reliability_confidence.raw();
        let vividness = episode.vividness_confidence.raw();

        // Favor encoding fidelity and reliability but keep vividness in the mix.
        let weighted = (encoding * 0.5) + (reliability * 0.3) + (vividness * 0.2);
        let decay_factor = (1.0 - episode.decay_rate).clamp(0.0, 1.0);
        Confidence::from_raw(weighted * self.concept_formation_penalty * decay_factor)
    }

    /// Accessor for the binding decay multiplier.
    #[must_use]
    pub const fn binding_decay(&self) -> f32 {
        self.binding_confidence_decay
    }

    /// Accessor for the blend bonus multiplier (1.0 = no bonus).
    #[must_use]
    pub const fn blend_bonus_multiplier(&self) -> f32 {
        self.blend_confidence_bonus
    }

    /// Accessor for the concept formation penalty.
    #[must_use]
    pub const fn concept_penalty(&self) -> f32 {
        self.concept_formation_penalty
    }
}

impl Default for DualMemoryConfidence {
    fn default() -> Self {
        Self::new(0.9, 0.95, 1.1)
    }
}
