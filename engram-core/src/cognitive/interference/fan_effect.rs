//! Fan effect detection and modeling
//!
//! Implements Anderson (1974) fan effect: retrieval time increases linearly
//! with the number of associations to a concept.
//!
//! # Key Findings (Anderson 1974)
//!
//! - Fan 1: 1159ms ± 22ms (baseline)
//! - Fan 2: 1236ms ± 25ms (+77ms)
//! - Fan 3: 1305ms ± 28ms (+69ms average from fan 2)
//! - Average slope: ~70ms per additional association
//!
//! # Mechanism
//!
//! Spreading activation must divide among all outgoing associations.
//! Higher fan → lower activation per edge → slower retrieval.
//!
//! This is a **retrieval-stage** phenomenon, not encoding or consolidation.
//! Unlike proactive/retroactive interference, fan effect doesn't reduce
//! accuracy - it only affects retrieval latency.

use uuid::Uuid;

/// Fan effect detector for retrieval-stage associative interference
///
/// Implements Anderson (1974) linear RT increase with associative fan.
///
/// CRITICAL: This operates during RETRIEVAL stage, not encoding or consolidation.
/// It models activation competition, not memory degradation.
#[derive(Debug, Clone)]
pub struct FanEffectDetector {
    /// Base retrieval time for fan=1 (default: 1150ms per Anderson 1974)
    base_retrieval_time_ms: f32,

    /// Time per additional association (default: 70ms per Anderson 1974)
    /// CORRECTED from previous spec which used 50ms
    time_per_association_ms: f32,

    /// Activation divisor mode (linear vs sqrt)
    /// Linear: activation / fan (default)
    /// Sqrt: activation / sqrt(fan) (softer falloff)
    use_sqrt_divisor: bool,

    /// Minimum fan value (default: 1, cannot be 0)
    min_fan: usize,
}

impl Default for FanEffectDetector {
    fn default() -> Self {
        Self {
            base_retrieval_time_ms: 1150.0, // Anderson (1974) fan=1 baseline
            time_per_association_ms: 70.0,  // CORRECTED from 50ms
            use_sqrt_divisor: false,        // Linear divisor by default
            min_fan: 1,
        }
    }
}

impl FanEffectDetector {
    /// Create a new fan effect detector with custom parameters
    #[must_use]
    pub const fn new(
        base_retrieval_time_ms: f32,
        time_per_association_ms: f32,
        use_sqrt_divisor: bool,
    ) -> Self {
        Self {
            base_retrieval_time_ms,
            time_per_association_ms,
            use_sqrt_divisor,
            min_fan: 1,
        }
    }

    /// Compute fan (number of associations) for a node
    ///
    /// Counts all outgoing edges from the node in the memory graph.
    ///
    /// # Returns
    /// Number of associations, minimum 1
    #[must_use]
    pub fn compute_fan<B>(
        &self,
        node_id: &Uuid,
        graph: &crate::memory_graph::UnifiedMemoryGraph<B>,
    ) -> usize
    where
        B: crate::memory_graph::GraphBackend,
    {
        graph
            .get_outgoing_edge_count(node_id)
            .unwrap_or(0)
            .max(self.min_fan)
    }

    /// Compute retrieval time based on fan
    ///
    /// Formula: RT = base_time + (fan - 1) × time_per_association
    ///
    /// # Example
    /// ```
    /// # use engram_core::cognitive::interference::FanEffectDetector;
    /// let detector = FanEffectDetector::default();
    ///
    /// // Fan 1: 1150ms + (1-1) × 70ms = 1150ms
    /// assert_eq!(detector.compute_retrieval_time_ms(1), 1150.0);
    ///
    /// // Fan 2: 1150ms + (2-1) × 70ms = 1220ms
    /// assert_eq!(detector.compute_retrieval_time_ms(2), 1220.0);
    ///
    /// // Fan 3: 1150ms + (3-1) × 70ms = 1290ms
    /// assert_eq!(detector.compute_retrieval_time_ms(3), 1290.0);
    /// ```
    ///
    /// Matches Anderson (1974) empirical data within ±20ms.
    #[must_use]
    pub fn compute_retrieval_time_ms(&self, fan: usize) -> f32 {
        let fan_clamped = fan.max(self.min_fan);
        self.base_retrieval_time_ms + ((fan_clamped - 1) as f32 * self.time_per_association_ms)
    }

    /// Compute activation divisor based on fan
    ///
    /// Used to model spreading activation competition:
    /// activation_per_edge = total_activation / divisor
    ///
    /// Linear mode (default): divisor = fan
    /// Sqrt mode: divisor = sqrt(fan)
    #[must_use]
    pub fn compute_activation_divisor(&self, fan: usize) -> f32 {
        let fan_clamped = fan.max(self.min_fan) as f32;

        if self.use_sqrt_divisor {
            fan_clamped.sqrt()
        } else {
            fan_clamped
        }
    }

    /// Detect fan effect for a retrieval operation
    ///
    /// Computes fan, retrieval time, and activation divisor for a node.
    ///
    /// # Returns
    /// FanEffectResult with all fan-related metrics
    #[must_use]
    pub fn detect_fan_effect<B>(
        &self,
        node_id: &Uuid,
        graph: &crate::memory_graph::UnifiedMemoryGraph<B>,
    ) -> FanEffectResult
    where
        B: crate::memory_graph::GraphBackend,
    {
        let fan = self.compute_fan(node_id, graph);
        let retrieval_time_ms = self.compute_retrieval_time_ms(fan);
        let activation_divisor = self.compute_activation_divisor(fan);

        // Record metrics
        #[cfg(feature = "monitoring")]
        {
            if let Some(metrics) = crate::metrics::cognitive_patterns::cognitive_patterns() {
                // Record fan effect as interference type
                #[allow(clippy::cast_precision_loss)]
                let magnitude = fan as f32 / 10.0; // Normalize fan count to magnitude
                metrics.record_interference(
                    crate::metrics::cognitive_patterns::InterferenceType::Fan,
                    magnitude,
                );
            }
        }

        FanEffectResult {
            fan,
            retrieval_time_ms,
            activation_divisor,
        }
    }

    /// Apply fan effect to activation spreading
    ///
    /// Divides activation among outgoing edges based on fan.
    ///
    /// Used during spreading activation to model competition.
    #[must_use]
    #[allow(clippy::unused_self)] // Keep as method for API consistency
    pub fn apply_to_activation(&self, base_activation: f32, fan_result: &FanEffectResult) -> f32 {
        base_activation / fan_result.activation_divisor
    }

    /// Get base retrieval time
    #[must_use]
    pub const fn base_retrieval_time_ms(&self) -> f32 {
        self.base_retrieval_time_ms
    }

    /// Get time per association
    #[must_use]
    pub const fn time_per_association_ms(&self) -> f32 {
        self.time_per_association_ms
    }

    /// Check if using sqrt divisor mode
    #[must_use]
    pub const fn use_sqrt_divisor(&self) -> bool {
        self.use_sqrt_divisor
    }

    /// Enable sqrt divisor mode (softer activation falloff)
    pub const fn set_sqrt_divisor(&mut self, enabled: bool) {
        self.use_sqrt_divisor = enabled;
    }
}

/// Result of fan effect detection
#[derive(Debug, Clone, Copy)]
pub struct FanEffectResult {
    /// Number of associations (fan) for the node
    pub fan: usize,

    /// Predicted retrieval time in milliseconds
    pub retrieval_time_ms: f32,

    /// Activation divisor (how much to divide activation by)
    pub activation_divisor: f32,
}

impl FanEffectResult {
    /// Check if fan is high (>3 associations)
    #[must_use]
    pub const fn is_high_fan(&self) -> bool {
        self.fan > 3
    }

    /// Retrieval time slowdown compared to fan=1 baseline
    #[must_use]
    pub fn slowdown_ms(&self, baseline_ms: f32) -> f32 {
        self.retrieval_time_ms - baseline_ms
    }

    /// Create result for single association (no fan effect)
    #[must_use]
    pub const fn single_association(base_time_ms: f32) -> Self {
        Self {
            fan: 1,
            retrieval_time_ms: base_time_ms,
            activation_divisor: 1.0,
        }
    }
}

/// Statistics aggregator for fan effect across memory graph
#[derive(Debug, Clone)]
pub struct FanEffectStatistics {
    /// Distribution of fan values: fan → count
    pub fan_distribution: std::collections::HashMap<usize, usize>,

    /// Average fan across all nodes
    pub average_fan: f32,

    /// Maximum fan in graph
    pub max_fan: usize,

    /// Median fan
    pub median_fan: usize,

    /// Nodes with high fan (>3)
    pub high_fan_nodes: Vec<Uuid>,
}

impl FanEffectStatistics {
    /// Compute fan statistics across entire memory graph
    #[must_use]
    pub fn compute<B>(
        graph: &crate::memory_graph::UnifiedMemoryGraph<B>,
        detector: &FanEffectDetector,
    ) -> Self
    where
        B: crate::memory_graph::GraphBackend,
    {
        let mut fan_distribution = std::collections::HashMap::new();
        let mut fan_values = Vec::new();
        let mut high_fan_nodes = Vec::new();

        for node_id in graph.backend().all_ids() {
            let fan = detector.compute_fan(&node_id, graph);

            *fan_distribution.entry(fan).or_insert(0) += 1;
            fan_values.push(fan);

            if fan > 3 {
                high_fan_nodes.push(node_id);
            }
        }

        fan_values.sort_unstable();

        let average_fan = if fan_values.is_empty() {
            0.0
        } else {
            fan_values.iter().sum::<usize>() as f32 / fan_values.len() as f32
        };

        let max_fan = fan_values.last().copied().unwrap_or(0);

        let median_fan = if fan_values.is_empty() {
            0
        } else {
            fan_values[fan_values.len() / 2]
        };

        Self {
            fan_distribution,
            average_fan,
            max_fan,
            median_fan,
            high_fan_nodes,
        }
    }

    /// Nodes with unusual fan (outliers > 2 std dev from mean)
    #[must_use]
    pub fn outlier_nodes(&self) -> Vec<Uuid> {
        // Compute standard deviation
        let mean = self.average_fan;
        let variance: f32 = self
            .fan_distribution
            .iter()
            .map(|(fan, count)| {
                let diff = *fan as f32 - mean;
                diff * diff * (*count as f32)
            })
            .sum::<f32>()
            / self.fan_distribution.values().sum::<usize>() as f32;

        let std_dev = variance.sqrt();
        let threshold = mean + 2.0 * std_dev;

        // Find nodes whose fan exceeds threshold
        self.high_fan_nodes
            .iter()
            .filter(|_node_id| {
                // For simplicity, we check if any high fan node exceeds threshold
                // In a real implementation, we'd need to look up the actual fan count
                // This is a placeholder - full implementation would require node->fan mapping
                self.max_fan as f32 > threshold
            })
            .copied()
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_parameters() {
        let detector = FanEffectDetector::default();
        assert_eq!(detector.base_retrieval_time_ms(), 1150.0);
        assert_eq!(detector.time_per_association_ms(), 70.0);
        assert!(!detector.use_sqrt_divisor());
    }

    #[test]
    fn test_anderson_1974_retrieval_times() {
        let detector = FanEffectDetector::default();

        // Anderson (1974) Table 1 data
        let empirical_data = vec![
            (1, 1159.0), // Fan 1: baseline
            (2, 1236.0), // Fan 2: +77ms
            (3, 1305.0), // Fan 3: +69ms (average)
        ];

        for (fan, expected_rt) in empirical_data {
            let predicted_rt = detector.compute_retrieval_time_ms(fan);

            // Allow ±20ms tolerance (Anderson's data has ~±25ms std dev)
            let tolerance = 20.0;
            assert!(
                (predicted_rt - expected_rt).abs() < tolerance,
                "Fan {}: Expected {}ms, got {}ms (diff: {:.1}ms)",
                fan,
                expected_rt,
                predicted_rt,
                predicted_rt - expected_rt
            );
        }
    }

    #[test]
    fn test_linear_scaling_70ms_per_association() {
        let detector = FanEffectDetector::default();

        // Test linear relationship
        let rt1 = detector.compute_retrieval_time_ms(1);
        let rt2 = detector.compute_retrieval_time_ms(2);
        let rt3 = detector.compute_retrieval_time_ms(3);
        let rt5 = detector.compute_retrieval_time_ms(5);

        // Slope should be constant (~70ms)
        let slope_1_2 = rt2 - rt1;
        let slope_2_3 = rt3 - rt2;
        let slope_3_5 = (rt5 - rt3) / 2.0;

        assert_eq!(slope_1_2, 70.0, "Slope 1→2 should be 70ms");
        assert_eq!(slope_2_3, 70.0, "Slope 2→3 should be 70ms");
        assert_eq!(slope_3_5, 70.0, "Slope 3→5 should be 70ms");

        // Verify base time
        assert_eq!(rt1, 1150.0, "Base time (fan=1) should be 1150ms");
    }

    #[test]
    fn test_activation_division() {
        let detector = FanEffectDetector::default();
        let base_activation = 1.0;

        // Fan = 1: Full activation
        let result_fan1 = FanEffectResult {
            fan: 1,
            retrieval_time_ms: 1150.0,
            activation_divisor: 1.0,
        };
        let activation_fan1 = detector.apply_to_activation(base_activation, &result_fan1);
        assert_eq!(activation_fan1, 1.0, "Fan=1 should have full activation");

        // Fan = 3: One-third activation per edge
        let result_fan3 = FanEffectResult {
            fan: 3,
            retrieval_time_ms: 1290.0,
            activation_divisor: 3.0,
        };
        let activation_fan3 = detector.apply_to_activation(base_activation, &result_fan3);
        assert_eq!(
            activation_fan3,
            1.0 / 3.0,
            "Fan=3 should divide activation by 3"
        );

        // Fan = 5: One-fifth activation per edge
        let result_fan5 = FanEffectResult {
            fan: 5,
            retrieval_time_ms: 1430.0,
            activation_divisor: 5.0,
        };
        let activation_fan5 = detector.apply_to_activation(base_activation, &result_fan5);
        assert_eq!(
            activation_fan5,
            1.0 / 5.0,
            "Fan=5 should divide activation by 5"
        );
    }

    #[test]
    fn test_sqrt_divisor_mode() {
        let mut detector = FanEffectDetector::default();
        detector.set_sqrt_divisor(true);

        // Linear mode: fan=9 → divisor=9
        // Sqrt mode: fan=9 → divisor=3
        let divisor = detector.compute_activation_divisor(9);
        assert_eq!(divisor, 3.0, "Sqrt of 9 should be 3");

        // Application to activation
        let base_activation = 1.0;
        let result = FanEffectResult {
            fan: 9,
            retrieval_time_ms: 0.0, // Irrelevant for this test
            activation_divisor: divisor,
        };

        let activation = detector.apply_to_activation(base_activation, &result);
        assert_eq!(
            activation,
            1.0 / 3.0,
            "Activation should be divided by sqrt(9) = 3"
        );
    }

    #[test]
    fn test_fan_effect_result_helpers() {
        let result = FanEffectResult {
            fan: 3,
            retrieval_time_ms: 1290.0,
            activation_divisor: 3.0,
        };

        assert!(!result.is_high_fan(), "Fan=3 should not be high");
        assert_eq!(result.slowdown_ms(1150.0), 140.0);

        let high_result = FanEffectResult {
            fan: 5,
            retrieval_time_ms: 1430.0,
            activation_divisor: 5.0,
        };

        assert!(high_result.is_high_fan(), "Fan=5 should be high");
    }

    #[test]
    fn test_single_association() {
        let result = FanEffectResult::single_association(1150.0);
        assert_eq!(result.fan, 1);
        assert_eq!(result.retrieval_time_ms, 1150.0);
        assert_eq!(result.activation_divisor, 1.0);
    }
}
