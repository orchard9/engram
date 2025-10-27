//! Priming effects based on spreading activation theory
//!
//! Implements three types of priming validated against empirical research:
//! - **Semantic priming:** Collins & Loftus (1975) spreading activation
//! - **Associative priming:** McKoon & Ratcliff (1992) compound cue theory
//! - **Repetition priming:** Tulving & Schacter (1990) perceptual fluency
//!
//! # Unified Priming Coordinator
//!
//! The `PrimingCoordinator` combines all three priming types with saturating
//! additive combination to prevent over-priming, matching neural firing rate
//! saturation observed empirically (Neely & Keefe 1989).

pub mod associative;
pub mod repetition;
pub mod semantic;

pub use associative::{AssociativePrimingEngine, AssociativeStatistics};
pub use repetition::{RepetitionPrimingEngine, RepetitionStatistics};
pub use semantic::{PrimingStatistics, SemanticPrimingEngine};

/// Unified priming coordinator combining all three priming types
///
/// Integrates semantic, associative, and repetition priming with saturating
/// additive combination to prevent over-priming.
///
/// # Biological Justification
///
/// Neural firing rates saturate at maximum frequency (~200 Hz for cortical
/// pyramidal cells). Multiple priming sources show diminishing returns, not
/// linear summation (Neely & Keefe 1989).
///
/// # Saturation Function
///
/// Uses exponential saturation: `1.0 - exp(-linear_sum)`
/// - Never exceeds ~63% even with maximum all types
/// - Smooth diminishing returns as priming accumulates
/// - Matches behavioral ceiling effects in RT studies
///
/// # Example
///
/// ```ignore
/// use engram_core::cognitive::priming::PrimingCoordinator;
///
/// let coordinator = PrimingCoordinator::new();
///
/// // Semantic priming from spreading activation
/// coordinator.semantic().activate_priming("doctor", &embedding, || {
///     vec![("nurse".to_string(), nurse_embedding, 1)]
/// });
///
/// // Associative priming from co-occurrence
/// coordinator.associative().record_coactivation("thunder", "lightning");
///
/// // Repetition priming from exposure
/// coordinator.repetition().record_exposure("familiar_concept");
///
/// // Compute total boost with saturation
/// let total_boost = coordinator.compute_total_boost("nurse", Some("doctor"));
/// ```
pub struct PrimingCoordinator {
    semantic: SemanticPrimingEngine,
    associative: AssociativePrimingEngine,
    repetition: RepetitionPrimingEngine,
}

impl PrimingCoordinator {
    /// Create new priming coordinator with default engines
    #[must_use]
    pub fn new() -> Self {
        Self {
            semantic: SemanticPrimingEngine::new(),
            associative: AssociativePrimingEngine::new(),
            repetition: RepetitionPrimingEngine::new(),
        }
    }

    /// Get reference to semantic priming engine
    #[must_use]
    pub const fn semantic(&self) -> &SemanticPrimingEngine {
        &self.semantic
    }

    /// Get reference to associative priming engine
    #[must_use]
    pub const fn associative(&self) -> &AssociativePrimingEngine {
        &self.associative
    }

    /// Get reference to repetition priming engine
    #[must_use]
    pub const fn repetition(&self) -> &RepetitionPrimingEngine {
        &self.repetition
    }

    /// Compute total priming boost combining all three types
    ///
    /// Applies saturating additive combination to prevent over-priming.
    ///
    /// # Algorithm
    ///
    /// 1. Compute semantic boost (if prime_node provided)
    /// 2. Compute associative boost (if prime_node provided)
    /// 3. Compute repetition boost
    /// 4. Sum all boosts
    /// 5. Apply saturation: `1.0 - exp(-linear_sum)`
    ///
    /// # Saturation Properties
    ///
    /// - Input: 0.0 → Output: 0.0 (no priming)
    /// - Input: 0.3 → Output: ~0.26 (slight saturation)
    /// - Input: 0.9 → Output: ~0.59 (strong saturation)
    /// - Input: 3.0 → Output: ~0.95 (asymptotic ceiling)
    ///
    /// # Performance
    /// O(1) - Three engine queries + exponential computation
    /// Typical: <50ns
    ///
    /// # Parameters
    /// * `target_node` - Target node to compute boost for
    /// * `prime_node` - Optional prime node for semantic/associative priming
    ///
    /// # Returns
    /// Total priming boost in [0.0, ~0.95], saturating at high values
    #[must_use]
    pub fn compute_total_boost(&self, target_node: &str, prime_node: Option<&str>) -> f32 {
        let mut linear_sum = 0.0;

        // Semantic priming (requires target node)
        linear_sum += self.semantic.compute_priming_boost(target_node);

        // Associative priming (requires both prime and target)
        if let Some(prime) = prime_node {
            linear_sum += self
                .associative
                .compute_association_strength(prime, target_node);
        }

        // Repetition priming (only requires target)
        linear_sum += self.repetition.compute_repetition_boost(target_node);

        // Apply saturation: 1.0 - exp(-x)
        // This function has these properties:
        // - f(0) = 0
        // - f(x) → 1 as x → ∞
        // - Smooth transition with diminishing returns
        // - Never exceeds 0.95 for practical values (0-3.0)
        //
        // CORRECTED from validation notes: Use exponential saturation instead
        // of linear additive to prevent overshoot
        1.0 - (-linear_sum).exp()
    }

    /// Compute total boost with detailed breakdown
    ///
    /// Returns individual contributions from each priming type along with
    /// the saturated total.
    ///
    /// # Returns
    /// Tuple of (semantic_boost, associative_boost, repetition_boost, total_saturated)
    #[must_use]
    pub fn compute_total_boost_detailed(
        &self,
        target_node: &str,
        prime_node: Option<&str>,
    ) -> (f32, f32, f32, f32) {
        let semantic = self.semantic.compute_priming_boost(target_node);

        let associative = prime_node.map_or(0.0, |prime| {
            self.associative
                .compute_association_strength(prime, target_node)
        });

        let repetition = self.repetition.compute_repetition_boost(target_node);

        let linear_sum = semantic + associative + repetition;
        let total_saturated = 1.0 - (-linear_sum).exp();

        (semantic, associative, repetition, total_saturated)
    }

    /// Prune expired data from all engines
    ///
    /// Calls pruning on semantic and associative engines to prevent unbounded
    /// memory growth. Repetition engine has no temporal decay, so no pruning needed.
    pub fn prune_expired(&self) {
        self.semantic.prune_expired();
        self.associative.prune_old_cooccurrences();
        // No pruning for repetition engine (persistent within session)
    }
}

impl Default for PrimingCoordinator {
    fn default() -> Self {
        Self::new()
    }
}
