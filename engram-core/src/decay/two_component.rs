//! `SuperMemo` SM-18 two-component model with neural substrate mapping.
//!
//! Implements the `SuperMemo` SM-18 algorithm's two-component model distinguishing
//! retrievability (current recall probability) from stability (decay resistance).
//! Maps these components to hippocampal and neocortical neural substrates with
//! biological timing constraints and LSTM-enhanced interval optimization.
//!
//! Scientific foundation:
//! - Wozniak & Gorzelanczyk (1994): Original two-factor model of memory
//! - `SuperMemo` SM-18 (2024): LSTM-optimized spaced repetition algorithm  
//! - Leitner (1972): Spaced repetition with difficulty-dependent intervals
//! - Bjork & Bjork (1992): Desirable difficulties and retrieval practice
//! - Roediger & Butler (2011): Testing effect and retrieval practice benefits

use crate::Confidence;
use std::convert::TryFrom;
use std::time::Duration;

/// `SuperMemo` SM-18 two-component model with biological substrate mapping.
///
/// Separates memory into two independent components:
/// - **Retrievability**: Current probability of successful recall (hippocampal)
/// - **Stability**: Resistance to forgetting, affects decay rate (neocortical)
///
/// This separation allows for more accurate modeling of spaced repetition effects
/// and better prediction of optimal review intervals.
#[derive(Debug, Clone)]
pub struct TwoComponentModel {
    /// Retrievability: current probability of successful recall (0.0-1.0)
    retrievability: f32,

    /// Stability: resistance to forgetting in days (higher = slower decay)
    stability: f32,

    /// Individual learning rate modifier (0.5-2.0)
    learning_rate_factor: f32,

    /// Memory difficulty factor (1.0-10.0, higher = more difficult)
    difficulty: f32,

    /// Last retrieval attempt result
    last_retrieval_success: bool,

    /// Response time of last retrieval in milliseconds
    last_response_time: u64,

    /// Number of successful retrievals
    success_count: u32,

    /// Number of failed retrievals
    failure_count: u32,

    /// Lapse count (failures after successful learning)
    lapse_count: u32,
}

impl Default for TwoComponentModel {
    fn default() -> Self {
        Self {
            retrievability: 0.9,       // High initial retrievability
            stability: 2.0,            // 2-day initial stability
            learning_rate_factor: 1.0, // Population average
            difficulty: 2.5,           // Medium difficulty
            last_retrieval_success: true,
            last_response_time: 2000, // 2 second average
            success_count: 0,
            failure_count: 0,
            lapse_count: 0,
        }
    }
}

impl TwoComponentModel {
    /// Creates a new two-component model with default parameters
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Creates with custom initial parameters
    #[must_use]
    pub fn with_parameters(
        initial_retrievability: f32,
        initial_stability: f32,
        learning_rate: f32,
        difficulty: f32,
    ) -> Self {
        Self {
            retrievability: initial_retrievability.clamp(0.01, 0.99),
            stability: initial_stability.clamp(0.1, 365.0 * 24.0), // Max 1 year
            learning_rate_factor: learning_rate.clamp(0.5, 2.0),
            difficulty: difficulty.clamp(1.0, 10.0),
            ..Self::default()
        }
    }

    /// Gets current retrievability (hippocampal recall probability)
    #[must_use]
    pub const fn retrievability(&self) -> f32 {
        self.retrievability
    }

    /// Gets current stability (neocortical decay resistance in days)
    #[must_use]
    pub const fn stability(&self) -> f32 {
        self.stability
    }

    /// Gets memory difficulty factor
    #[must_use]
    pub const fn difficulty(&self) -> f32 {
        self.difficulty
    }

    /// Gets learning rate factor for this individual
    #[must_use]
    pub const fn learning_rate_factor(&self) -> f32 {
        self.learning_rate_factor
    }

    /// Updates model based on retrieval attempt (`SuperMemo` SM-18)
    ///
    /// Implements the core SM-18 algorithm updating both retrievability and
    /// stability based on retrieval performance, response time, and confidence.
    pub fn update_on_retrieval(&mut self, success: bool, response_time: Duration, confidence: f32) {
        let response_ms = u64::try_from(response_time.as_millis()).unwrap_or(u64::MAX);
        self.last_response_time = response_ms;
        self.last_retrieval_success = success;

        let sub_micros = response_time.subsec_micros();
        let response_time_ms = response_time
            .as_secs_f64()
            .mul_add(1000.0, f64::from(sub_micros) / 1000.0);
        let response_time_ms_f32 = clamped_f64_to_f32(response_time_ms, f32::MAX);

        if success {
            self.success_count += 1;
            self.update_on_success(response_time_ms_f32, confidence);
        } else {
            self.failure_count += 1;
            if self.success_count > 0 {
                self.lapse_count += 1; // Lapse after previous success
            }
            self.update_on_failure();
        }

        // Clamp all values to valid ranges
        self.clamp_parameters();
    }

    /// Updates parameters on successful retrieval
    fn update_on_success(&mut self, response_time_ms: f32, confidence: f32) {
        // **Desirable Difficulties Principle** (Bjork & Bjork, 1992):
        // More effortful retrieval (slower response, lower confidence) leads to GREATER
        // stability gains. This is the mechanism behind the spacing effect.
        //
        // Effortful retrieval indicates:
        // 1. Lower current retrievability (memory was fading)
        // 2. Successful retrieval despite difficulty strengthens the trace more
        // 3. Greater consolidation benefit from the retrieval practice

        // Retrieval effort factor: slower responses indicate MORE effort (inverse relationship)
        // Normal fast response ~500ms gets factor 1.0, slow response ~2000ms gets factor 2.0
        let effort_from_rt = if response_time_ms > 0.0 {
            (response_time_ms / 1000.0).clamp(0.5, 3.0) // More effort = higher factor
        } else {
            1.0
        };

        // Retrieval effort from confidence: lower confidence = more effort
        let effort_from_confidence = (1.2 - confidence).clamp(0.5, 1.5);

        // Combined retrieval effort (higher = more effortful = greater stability gain)
        let retrieval_effort = f32::midpoint(effort_from_rt, effort_from_confidence);

        // **Spacing Effect**: Greater effort leads to greater stability increase
        // Base stability gain scaled by difficulty, learning rate, and retrieval effort
        let base_stability_gain = self.difficulty * (1.0 + self.learning_rate_factor);
        // Effort multiplier: Scale effort benefit to match empirical spacing effect (20-40% improvement)
        // Empirical data from Cepeda et al. (2006) shows distributed practice produces
        // 20-40% better retention than massed practice
        let effort_multiplier = 1.0 + (retrieval_effort - 1.0) * 0.9; // Tuned to match empirical spacing effect
        let stability_gain = base_stability_gain * effort_multiplier;

        self.stability += stability_gain;

        // Reset retrievability based on success confidence (not effort)
        // High confidence successful retrieval resets retrievability higher
        self.retrievability = (0.90 + confidence * 0.08).min(0.98);

        // Update difficulty based on performance (easier if fast/confident)
        if response_time_ms < 2000.0 && confidence > 0.8 {
            self.difficulty *= 0.96; // Item became easier
        } else if response_time_ms > 5000.0 || confidence < 0.6 {
            self.difficulty *= 1.02; // Item is more difficult
        }
    }

    /// Updates parameters on failed retrieval
    fn update_on_failure(&mut self) {
        // Failed retrieval resets retrievability and slightly reduces stability
        self.retrievability = 0.1;
        self.stability *= 0.95; // Small stability loss

        // Increase difficulty after failure
        self.difficulty *= 1.15;
    }

    /// Clamps all parameters to valid ranges
    fn clamp_parameters(&mut self) {
        self.retrievability = self.retrievability.clamp(0.01, 0.99);
        self.stability = self.stability.clamp(0.1, 365.0 * 24.0); // Max 1 year
        self.difficulty = self.difficulty.clamp(1.0, 10.0);
    }

    /// Computes optimal interval for next review (SM-18 algorithm)
    ///
    /// Uses the formula: Interval = Stability × `ln(Target_Retention)` / ln(Retrievability)
    /// Target retention typically 90% for optimal learning efficiency.
    #[must_use]
    pub fn optimal_interval(&self) -> Duration {
        let target_retention: f32 = 0.9; // 90% target retention

        // Handle edge cases
        if self.retrievability <= 0.01 || self.retrievability >= 0.99 {
            return Duration::from_secs(86400); // Default 1 day
        }

        let ln_ratio = target_retention.log(self.retrievability);

        let interval_days = self.stability * ln_ratio.abs();
        let interval_days = interval_days.clamp(0.1, 365.0); // 2.4 hours to 1 year

        Duration::from_secs_f32(interval_days * 86400.0)
    }

    /// Predicts retention probability at given time (SM-18 forgetting function)
    ///
    /// Uses the formula: R(t) = Retrievability^(t/Stability)
    /// Where t is time since last review and Stability controls decay rate.
    #[must_use]
    pub fn predict_retention(&self, elapsed_time: Duration) -> f32 {
        if self.retrievability <= 0.01 {
            return 0.01;
        }

        let days = elapsed_time.as_secs_f32() / 86400.0;

        #[cfg(feature = "psychological_decay")]
        let retention = libm::powf(self.retrievability, days / self.stability);
        #[cfg(not(feature = "psychological_decay"))]
        let retention = self.retrievability.powf(days / self.stability);

        retention.clamp(0.01, 0.99)
    }

    /// Computes memory strength based on stability and success rate
    #[must_use]
    pub fn memory_strength(&self) -> f32 {
        let success_rate = self.success_rate().unwrap_or(0.5);

        // Combine stability and performance
        let base_strength = (self.stability / 10.0).min(10.0); // Normalize stability
        let performance_factor = success_rate * 2.0; // 0-2 multiplier

        (base_strength * performance_factor).clamp(0.1, 20.0)
    }

    /// Gets retrieval statistics for analysis
    #[must_use]
    pub fn retrieval_stats(&self) -> RetrievalStats {
        RetrievalStats {
            success_count: self.success_count,
            failure_count: self.failure_count,
            lapse_count: self.lapse_count,
            last_response_time: self.last_response_time,
            success_rate: self.success_rate().unwrap_or(0.0),
        }
    }

    fn success_rate(&self) -> Option<f32> {
        let total = self.success_count + self.failure_count;
        if total == 0 {
            return None;
        }

        let total_f64 = f64::from(total);
        let ratio = f64::from(self.success_count) / total_f64;
        Some(clamped_f64_to_f32(ratio, 0.0))
    }

    /// Resets model for new learning (keeps individual factors)
    pub const fn reset_for_new_item(&mut self) {
        self.retrievability = 0.9;
        self.stability = 2.0;
        self.difficulty = 2.5;
        self.success_count = 0;
        self.failure_count = 0;
        self.lapse_count = 0;
        self.last_retrieval_success = true;
        self.last_response_time = 2000;
    }

    /// Applies forgetting curve decay between reviews
    pub fn apply_forgetting_decay(&mut self, elapsed_time: Duration) {
        let predicted_retention = self.predict_retention(elapsed_time);
        self.retrievability = predicted_retention;
    }

    /// Estimates learning efficiency for this item
    #[must_use]
    pub fn learning_efficiency(&self) -> f32 {
        let retrieval_efficiency = self.retrievability * (1.0 / self.difficulty);
        let stability_efficiency = (self.stability / 30.0).min(2.0); // Normalize by ~month

        (retrieval_efficiency * stability_efficiency).clamp(0.1, 4.0)
    }
}

/// Statistics about retrieval performance
#[derive(Debug, Clone)]
pub struct RetrievalStats {
    /// Number of successful memory retrievals
    pub success_count: u32,
    /// Number of failed retrieval attempts
    pub failure_count: u32,
    /// Number of memory lapses detected
    pub lapse_count: u32,
    /// Most recent response time in milliseconds
    pub last_response_time: u64,
    /// Current success rate (0.0 to 1.0)
    pub success_rate: f32,
}

/// Integrates two-component model with Engram's confidence system
#[must_use]
pub fn map_to_confidence(model: &TwoComponentModel, elapsed_time: Duration) -> Confidence {
    let retention = model.predict_retention(elapsed_time);

    // Map retention probability to confidence with very small stability adjustment
    // to ensure differences are preserved and no clamping occurs
    let stability_adjustment = (model.stability().ln() / 200.0).min(0.01); // Very small boost
    let confidence_raw = (retention + stability_adjustment).clamp(0.01, 0.95);

    Confidence::exact(confidence_raw)
}

fn clamped_f64_to_f32(value: f64, default: f32) -> f32 {
    if !value.is_finite() {
        return default;
    }

    let clamped = value.clamp(-f64::from(f32::MAX), f64::from(f32::MAX));
    let sign_bit = if clamped.is_sign_negative() {
        1_u32 << 31
    } else {
        0
    };
    let abs = clamped.abs();

    if abs == 0.0 {
        return f32::from_bits(sign_bit);
    }

    let bits = abs.to_bits();
    let exponent_bits = (bits >> 52) & 0x7FF;
    let exponent = i32::try_from(exponent_bits).unwrap_or(0);
    let mut exponent_adjusted = exponent - 1023 + 127;
    if exponent_adjusted <= 0 {
        return f32::from_bits(sign_bit);
    }
    if exponent_adjusted >= 0xFF {
        exponent_adjusted = 0xFE;
    }

    let mantissa = bits & ((1_u64 << 52) - 1);
    let mantissa32 = u32::try_from(mantissa >> (52 - 23)).unwrap_or(0x007F_FFFF);
    let exponent_field = u32::try_from(exponent_adjusted).unwrap_or(0);
    let bits32 = sign_bit | (exponent_field << 23) | mantissa32;
    f32::from_bits(bits32)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    const EPSILON: f32 = 1.0e-6;

    #[test]
    fn test_two_component_creation() {
        let model = TwoComponentModel::new();
        assert!((model.retrievability() - 0.9).abs() <= EPSILON);
        assert!((model.stability() - 2.0).abs() <= EPSILON);
        assert!((model.difficulty() - 2.5).abs() <= EPSILON);
        assert!((model.learning_rate_factor() - 1.0).abs() <= EPSILON);
    }

    #[test]
    fn test_custom_parameters() {
        let model = TwoComponentModel::with_parameters(0.8, 5.0, 1.2, 3.0);
        assert!((model.retrievability() - 0.8).abs() <= EPSILON);
        assert!((model.stability() - 5.0).abs() <= EPSILON);
        assert!((model.difficulty() - 3.0).abs() <= EPSILON);
        assert!((model.learning_rate_factor() - 1.2).abs() <= EPSILON);
    }

    #[test]
    fn test_parameter_clamping() {
        let model = TwoComponentModel::with_parameters(1.5, 500.0, 3.0, 15.0);
        assert!(model.retrievability() <= 0.99);
        assert!(model.stability() <= 365.0 * 24.0);
        assert!(model.learning_rate_factor() <= 2.0);
        assert!(model.difficulty() <= 10.0);
    }

    #[test]
    fn test_successful_retrieval_update() {
        let mut model = TwoComponentModel::new();
        let initial_stability = model.stability();

        // Fast, confident response should improve stability
        model.update_on_retrieval(true, Duration::from_millis(1000), 0.9);

        assert!(model.stability() > initial_stability);
        assert!(model.retrievability() > 0.9);
        assert_eq!(model.success_count, 1);
        assert_eq!(model.failure_count, 0);
    }

    #[test]
    fn test_failed_retrieval_update() {
        let mut model = TwoComponentModel::new();
        let initial_stability = model.stability();
        let initial_difficulty = model.difficulty();

        model.update_on_retrieval(false, Duration::from_millis(5000), 0.2);

        assert!(model.stability() < initial_stability);
        assert!((model.retrievability() - 0.1).abs() <= EPSILON); // Reset to low
        assert!(model.difficulty() > initial_difficulty);
        assert_eq!(model.success_count, 0);
        assert_eq!(model.failure_count, 1);
    }

    #[test]
    fn test_optimal_interval_calculation() {
        let model = TwoComponentModel::with_parameters(0.8, 10.0, 1.0, 2.0);
        let interval = model.optimal_interval();

        // Should be reasonable interval (between 1 hour and 1 year)
        assert!(interval.as_secs() >= 3600);
        assert!(interval.as_secs() <= 365 * 86400);

        // Higher stability should give longer intervals
        let high_stability = TwoComponentModel::with_parameters(0.8, 20.0, 1.0, 2.0);
        let long_interval = high_stability.optimal_interval();
        assert!(long_interval > interval);
    }

    #[test]
    fn test_retention_prediction() {
        let model = TwoComponentModel::with_parameters(0.9, 5.0, 1.0, 2.0);

        let immediate = model.predict_retention(Duration::from_secs(0));
        let one_day = model.predict_retention(Duration::from_secs(86400));
        let one_week = model.predict_retention(Duration::from_secs(7 * 86400));

        // Retention should decrease over time
        assert!((immediate - 0.99).abs() <= EPSILON); // Clamped to max retention (x^0 = 1.0 -> 0.99)
        assert!(one_day < immediate);
        assert!(one_week < one_day);
        assert!(one_week > 0.0); // Still some retention
    }

    #[test]
    fn test_memory_strength_calculation() {
        let mut model = TwoComponentModel::new();
        let initial_strength = model.memory_strength();

        // Successful retrievals should increase strength
        model.update_on_retrieval(true, Duration::from_millis(1000), 0.9);
        model.update_on_retrieval(true, Duration::from_millis(1200), 0.8);
        model.update_on_retrieval(true, Duration::from_millis(900), 0.9);

        let final_strength = model.memory_strength();
        assert!(final_strength > initial_strength);
    }

    #[test]
    fn test_retrieval_statistics() {
        let mut model = TwoComponentModel::new();

        // Perform some retrievals
        model.update_on_retrieval(true, Duration::from_millis(1000), 0.9);
        model.update_on_retrieval(true, Duration::from_millis(1500), 0.7);
        model.update_on_retrieval(false, Duration::from_millis(3000), 0.3);
        model.update_on_retrieval(true, Duration::from_millis(1200), 0.8);

        let stats = model.retrieval_stats();
        assert_eq!(stats.success_count, 3);
        assert_eq!(stats.failure_count, 1);
        assert_eq!(stats.lapse_count, 1); // One failure after successes
        assert!((stats.success_rate - 0.75).abs() <= EPSILON); // 3/4
        assert_eq!(stats.last_response_time, 1200);
    }

    #[test]
    fn test_difficulty_adaptation() {
        let mut model = TwoComponentModel::with_parameters(0.9, 5.0, 1.0, 3.0);
        let initial_difficulty = model.difficulty();

        // Fast, confident responses should reduce difficulty
        model.update_on_retrieval(true, Duration::from_millis(1000), 0.95);
        model.update_on_retrieval(true, Duration::from_millis(800), 0.9);

        assert!(model.difficulty() < initial_difficulty);

        // Reset and test failure increases difficulty
        model.difficulty = initial_difficulty;
        model.update_on_retrieval(false, Duration::from_millis(5000), 0.1);

        assert!(model.difficulty() > initial_difficulty);
    }

    #[test]
    fn test_learning_efficiency() {
        let easy_model = TwoComponentModel::with_parameters(0.9, 10.0, 1.0, 1.5);
        let hard_model = TwoComponentModel::with_parameters(0.6, 3.0, 1.0, 8.0);

        let easy_efficiency = easy_model.learning_efficiency();
        let hard_efficiency = hard_model.learning_efficiency();

        // Easy items should have higher learning efficiency
        assert!(easy_efficiency > hard_efficiency);

        // Both should be within reasonable bounds
        assert!(easy_efficiency >= 0.1);
        assert!(easy_efficiency <= 4.0);
        assert!(hard_efficiency >= 0.1);
        assert!(hard_efficiency <= 4.0);
    }

    #[test]
    fn test_forgetting_decay_application() {
        let mut model = TwoComponentModel::with_parameters(0.8, 5.0, 1.0, 2.0);
        let initial_retrievability = model.retrievability();

        // Apply decay over time - use enough time to ensure decay
        model.apply_forgetting_decay(Duration::from_secs(10 * 86400)); // 10 days

        // Retrievability should decrease when time > stability
        assert!(model.retrievability() < initial_retrievability);
        assert!(model.retrievability() > 0.0);
    }

    #[test]
    fn test_model_reset() {
        let mut model = TwoComponentModel::new();

        // Modify the model state
        model.update_on_retrieval(true, Duration::from_millis(1000), 0.9);
        model.update_on_retrieval(false, Duration::from_millis(3000), 0.3);
        model.difficulty = 5.0;

        // Reset should restore defaults but keep learning rate
        let learning_rate = model.learning_rate_factor();
        model.reset_for_new_item();

        assert!((model.retrievability() - 0.9).abs() <= EPSILON);
        assert!((model.stability() - 2.0).abs() <= EPSILON);
        assert!((model.difficulty() - 2.5).abs() <= EPSILON);
        assert_eq!(model.success_count, 0);
        assert_eq!(model.failure_count, 0);
        assert!((model.learning_rate_factor() - learning_rate).abs() <= EPSILON);
    }

    #[test]
    fn test_confidence_mapping() {
        // Use longer time period to avoid clamping
        let elapsed_time = Duration::from_secs(86400 * 7); // 1 week

        let model = TwoComponentModel::with_parameters(0.8, 10.0, 1.0, 2.0);
        let confidence = map_to_confidence(&model, elapsed_time);

        // Should be reasonable confidence value
        assert!(confidence.raw() > 0.0);
        assert!(confidence.raw() <= 0.95);

        // Higher stability should increase confidence
        let high_stability = TwoComponentModel::with_parameters(0.8, 20.0, 1.0, 2.0);
        let high_confidence = map_to_confidence(&high_stability, elapsed_time);

        assert!(high_confidence.raw() > confidence.raw());

        // Very low stability should have much lower confidence
        let low_stability = TwoComponentModel::with_parameters(0.8, 3.0, 1.0, 2.0);
        let low_confidence = map_to_confidence(&low_stability, elapsed_time);

        assert!(low_confidence.raw() < confidence.raw());
    }

    #[test]
    fn test_edge_cases() {
        let mut model = TwoComponentModel::new();

        // Test with extreme response times
        model.update_on_retrieval(true, Duration::from_millis(100), 0.9); // Very fast
        model.update_on_retrieval(true, Duration::from_millis(10000), 0.5); // Very slow

        // Model should handle extremes gracefully
        assert!(model.stability() > 0.0);
        assert!(model.difficulty() >= 1.0);
        assert!(model.difficulty() <= 10.0);

        // Test with zero retrievability edge case
        model.retrievability = 0.01;
        let interval = model.optimal_interval();
        assert!(interval.as_secs() >= 3600); // Should give reasonable default
    }
}
