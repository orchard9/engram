//! Individual differences in memory decay based on cognitive variation research.
//!
//! Models individual differences in memory systems based on empirical research
//! showing ±20% variation around population means for working memory capacity,
//! processing speed, attention control, and cognitive flexibility. These factors
//! modulate hippocampal and neocortical decay parameters following established
//! cognitive psychology findings.
//!
//! Scientific foundation:
//! - Engle et al. (1999): Working memory capacity individual differences
//! - Salthouse (1996): Processing speed and cognitive aging effects
//! - Miyake et al. (2000): Executive function and attention control
//! - Conway et al. (2003): Working memory capacity and long-term memory

use crate::Confidence;

#[cfg(all(feature = "psychological_decay", feature = "testing"))]
use rand_distr::{Distribution, Normal};

#[cfg(feature = "psychological_decay")]
/// Individual difference profile capturing cognitive variation in memory systems.
///
/// Models individual differences in four key cognitive dimensions that affect
/// memory decay parameters: working memory capacity, processing speed, attention
/// control, and cognitive flexibility. Based on population distributions from
/// cognitive psychology research.
#[derive(Debug, Clone)]
pub struct IndividualDifferenceProfile {
    /// Working memory capacity (7±2, affects chunking and encoding quality)
    pub wm_capacity: f32,

    /// Processing speed factor (affects encoding efficiency and retrieval time)
    pub processing_speed: f32,

    /// Attention control (affects interference resistance and focus)
    pub attention_control: f32,

    /// Cognitive flexibility (affects schema integration and updating)
    pub flexibility: f32,

    /// Hippocampal efficiency factor (individual variation in hippocampal function)
    pub hippocampal_efficiency: f32,

    /// Neocortical efficiency factor (individual variation in cortical function)
    pub neocortical_efficiency: f32,

    /// Sleep quality factor (affects consolidation during rest)
    pub sleep_quality: f32,

    /// Age factor (affects overall memory system efficiency)
    pub age_factor: f32,
}

impl Default for IndividualDifferenceProfile {
    fn default() -> Self {
        Self {
            wm_capacity: 1.0,
            processing_speed: 1.0,
            attention_control: 1.0,
            flexibility: 1.0,
            hippocampal_efficiency: 1.0,
            neocortical_efficiency: 1.0,
            sleep_quality: 1.0,
            age_factor: 1.0,
        }
    }
}

impl IndividualDifferenceProfile {
    /// Creates a new profile with population average values
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Creates profile with specific factor values
    #[must_use]
    pub fn with_factors(
        wm_capacity: f32,
        processing_speed: f32,
        attention_control: f32,
        flexibility: f32,
    ) -> Self {
        Self {
            wm_capacity: wm_capacity.clamp(0.5, 2.0),
            processing_speed: processing_speed.clamp(0.5, 2.0),
            attention_control: attention_control.clamp(0.5, 2.0),
            flexibility: flexibility.clamp(0.5, 2.0),
            ..Self::default()
        }
    }

    /// Samples profile from population distribution (μ=1.0, σ=0.2)
    ///
    /// Creates individual differences based on normal distribution with 20%
    /// standard deviation, matching empirical findings of cognitive variation.
    #[cfg(all(feature = "psychological_decay", feature = "testing"))]
    pub fn sample_from_population<R: rand::Rng>(rng: &mut R) -> Self {
        let normal = Normal::new(1.0, 0.2).unwrap();

        Self {
            wm_capacity: (normal.sample(rng) as f32).clamp(0.5, 2.0),
            processing_speed: (normal.sample(rng) as f32).clamp(0.5, 2.0),
            attention_control: (normal.sample(rng) as f32).clamp(0.5, 2.0),
            flexibility: (normal.sample(rng) as f32).clamp(0.5, 2.0),
            hippocampal_efficiency: (normal.sample(rng) as f32).clamp(0.5, 2.0),
            neocortical_efficiency: (normal.sample(rng) as f32).clamp(0.5, 2.0),
            sleep_quality: (normal.sample(rng) as f32).clamp(0.5, 2.0),
            age_factor: (normal.sample(rng) as f32).clamp(0.5, 1.5), // Age only reduces efficiency
        }
    }

    /// Samples from population with age-specific adjustments
    #[cfg(all(feature = "psychological_decay", feature = "testing"))]
    pub fn sample_with_age<R: rand::Rng>(rng: &mut R, age_years: f32) -> Self {
        let mut profile = Self::sample_from_population(rng);

        // Age-related decline in processing speed and working memory
        let age_decline = (age_years / 100.0).min(0.5); // Max 50% decline
        profile.processing_speed *= 1.0 - age_decline * 0.3;
        profile.wm_capacity *= 1.0 - age_decline * 0.2;
        profile.attention_control *= 1.0 - age_decline * 0.15;

        // Sleep quality may decline with age
        profile.sleep_quality *= 1.0 - age_decline * 0.25;

        // Overall age factor
        profile.age_factor = (1.0 - age_decline).clamp(0.5, 1.0);

        // Clamp all values
        profile.clamp_all_factors();
        profile
    }

    /// Clamps all factors to valid ranges
    const fn clamp_all_factors(&mut self) {
        self.wm_capacity = self.wm_capacity.clamp(0.5, 2.0);
        self.processing_speed = self.processing_speed.clamp(0.5, 2.0);
        self.attention_control = self.attention_control.clamp(0.5, 2.0);
        self.flexibility = self.flexibility.clamp(0.5, 2.0);
        self.hippocampal_efficiency = self.hippocampal_efficiency.clamp(0.5, 2.0);
        self.neocortical_efficiency = self.neocortical_efficiency.clamp(0.5, 2.0);
        self.sleep_quality = self.sleep_quality.clamp(0.5, 2.0);
        self.age_factor = self.age_factor.clamp(0.5, 1.5);
    }

    /// Applies individual differences to hippocampal decay tau parameter
    ///
    /// Working memory capacity and processing speed have largest effects on
    /// hippocampal encoding and maintenance according to research.
    #[must_use]
    pub fn modify_hippocampal_tau(&self, base_tau: f32) -> f32 {
        let combined_factor = self.hippocampal_efficiency.mul_add(
            0.30,
            self.attention_control.mul_add(
                0.20,
                self.wm_capacity.mul_add(0.25, self.processing_speed * 0.25),
            ),
        ); // Direct hippocampal efficiency

        base_tau * combined_factor * self.age_factor
    }

    /// Applies individual differences to neocortical decay tau parameter
    ///
    /// Cognitive flexibility and sleep quality have larger effects on neocortical
    /// consolidation and schema integration processes.
    #[must_use]
    pub fn modify_neocortical_tau(&self, base_tau: f32) -> f32 {
        let combined_factor = self.sleep_quality.mul_add(
            0.25,
            self.attention_control.mul_add(
                0.20,
                self.flexibility
                    .mul_add(0.30, self.neocortical_efficiency * 0.25),
            ),
        ); // Sleep-dependent consolidation

        base_tau * combined_factor * self.age_factor
    }

    /// Calculates schema integration efficiency based on cognitive profile
    ///
    /// Combines flexibility, working memory, and processing speed as these
    /// factors affect ability to integrate new information with existing schemas.
    #[must_use]
    pub fn schema_integration_efficiency(&self) -> f32 {
        let efficiency = self.neocortical_efficiency.mul_add(
            0.30,
            self.processing_speed.mul_add(
                0.10,
                self.flexibility.mul_add(0.40, self.wm_capacity * 0.20),
            ),
        ); // Neocortical function for schemas

        (efficiency * self.age_factor).clamp(0.2, 1.8)
    }

    /// Calculates pattern completion efficiency for hippocampal function
    ///
    /// Working memory capacity and hippocampal efficiency are key factors
    /// for successful pattern completion from partial cues.
    #[must_use]
    pub fn pattern_completion_efficiency(&self) -> f32 {
        let efficiency = self.attention_control.mul_add(
            0.25,
            self.wm_capacity
                .mul_add(0.35, self.hippocampal_efficiency * 0.40),
        ); // Focus needed for cue processing

        (efficiency * self.age_factor).clamp(0.3, 1.7)
    }

    /// Estimates retrieval speed modification factor
    ///
    /// Processing speed and attention control are primary factors affecting
    /// how quickly memories can be retrieved from storage.
    #[must_use]
    pub fn retrieval_speed_factor(&self) -> f32 {
        let speed_factor = self.hippocampal_efficiency.mul_add(
            0.25,
            self.processing_speed
                .mul_add(0.50, self.attention_control * 0.25),
        ); // Hippocampal efficiency for access

        (speed_factor * self.age_factor).clamp(0.4, 2.0)
    }

    /// Calculates consolidation effectiveness during sleep
    ///
    /// Sleep quality and neocortical efficiency are key for offline consolidation
    /// processes that strengthen memories during rest periods.
    #[must_use]
    pub fn consolidation_effectiveness(&self) -> f32 {
        let effectiveness = self.flexibility.mul_add(
            0.20,
            self.sleep_quality
                .mul_add(0.45, self.neocortical_efficiency * 0.35),
        ); // Flexibility helps memory reorganization

        (effectiveness * self.age_factor).clamp(0.3, 1.8)
    }

    /// Estimates interference resistance based on attention control
    ///
    /// Attention control is the primary factor in resisting interference from
    /// competing memories and distractors during encoding and retrieval.
    #[must_use]
    pub fn interference_resistance(&self) -> f32 {
        let resistance = self.processing_speed.mul_add(
            0.15,
            self.attention_control
                .mul_add(0.60, self.wm_capacity * 0.25),
        ); // Speed helps overcome interference

        (resistance * self.age_factor).clamp(0.3, 1.9)
    }

    /// Computes overall memory system efficiency
    ///
    /// Weighted combination of all factors providing general memory performance.
    #[must_use]
    pub fn overall_memory_efficiency(&self) -> f32 {
        let efficiency = self.sleep_quality.mul_add(0.10, self.neocortical_efficiency.mul_add(0.125, self.hippocampal_efficiency.mul_add(
            0.125,
            self.flexibility.mul_add(
                0.15,
                self.attention_control.mul_add(
                    0.20,
                    self.wm_capacity.mul_add(0.20, self.processing_speed * 0.20),
                ),
            ),
        )));

        (efficiency * self.age_factor).clamp(0.4, 1.8)
    }

    /// Applies individual differences to confidence calibration
    ///
    /// Higher cognitive abilities generally lead to better confidence calibration
    /// and reduced overconfidence effects.
    #[must_use]
    pub fn calibrate_confidence(&self, base_confidence: Confidence) -> Confidence {
        let calibration_factor = self.overall_memory_efficiency();

        // Better cognitive abilities reduce overconfidence
        // People with high ability are better calibrated (less adjustment needed)
        // People with low ability are more overconfident (more adjustment needed)
        let raw = base_confidence.raw();
        let calibrated = if raw > 0.7 {
            // High confidence: low ability needs more reduction (Dunning-Kruger)
            // calibration_factor ranges from ~0.4 to ~1.8
            // For low ability (0.875), reduce more: raw * ~0.85
            // For high ability (1.55), reduce less: raw * ~0.95
            let reduction_factor = 0.8 + 0.15 * calibration_factor.min(1.33);
            raw * reduction_factor
        } else {
            // Low confidence may get slight boost from good cognitive abilities
            raw * 0.05f32.mul_add(calibration_factor, 0.95)
        };

        Confidence::exact(calibrated)
    }

    /// Creates a profile representing cognitive decline (aging or impairment)
    #[must_use]
    pub fn with_decline(decline_factor: f32) -> Self {
        let clamped_decline = decline_factor.clamp(0.0, 0.8); // Max 80% decline

        Self {
            wm_capacity: clamped_decline.mul_add(-0.6, 1.0).clamp(0.5, 2.0),
            processing_speed: clamped_decline.mul_add(-0.7, 1.0).clamp(0.5, 2.0),
            attention_control: clamped_decline.mul_add(-0.5, 1.0).clamp(0.5, 2.0),
            flexibility: clamped_decline.mul_add(-0.4, 1.0).clamp(0.5, 2.0),
            hippocampal_efficiency: clamped_decline.mul_add(-0.8, 1.0).clamp(0.5, 2.0),
            neocortical_efficiency: clamped_decline.mul_add(-0.3, 1.0).clamp(0.5, 2.0),
            sleep_quality: clamped_decline.mul_add(-0.5, 1.0).clamp(0.5, 2.0),
            age_factor: (1.0 - clamped_decline).clamp(0.5, 1.0),
        }
    }

    /// Creates a profile representing enhanced cognitive abilities
    #[must_use]
    pub fn with_enhancement(enhancement_factor: f32) -> Self {
        let clamped_enhancement = enhancement_factor.clamp(0.0, 1.0);

        Self {
            wm_capacity: clamped_enhancement.mul_add(1.0, 1.0).clamp(0.5, 2.0),
            processing_speed: clamped_enhancement.mul_add(1.0, 1.0).clamp(0.5, 2.0),
            attention_control: clamped_enhancement.mul_add(1.0, 1.0).clamp(0.5, 2.0),
            flexibility: clamped_enhancement.mul_add(1.0, 1.0).clamp(0.5, 2.0),
            hippocampal_efficiency: clamped_enhancement.mul_add(1.0, 1.0).clamp(0.5, 2.0),
            neocortical_efficiency: clamped_enhancement.mul_add(1.0, 1.0).clamp(0.5, 2.0),
            sleep_quality: clamped_enhancement.mul_add(1.0, 1.0).clamp(0.5, 2.0),
            age_factor: 1.0, // Enhancement doesn't affect age
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_individual_profile_creation() {
        let profile = IndividualDifferenceProfile::new();
        assert_eq!(profile.wm_capacity, 1.0);
        assert_eq!(profile.processing_speed, 1.0);
        assert_eq!(profile.attention_control, 1.0);
        assert_eq!(profile.flexibility, 1.0);
    }

    #[test]
    fn test_profile_with_factors() {
        let profile = IndividualDifferenceProfile::with_factors(1.5, 0.8, 1.2, 0.9);
        assert_eq!(profile.wm_capacity, 1.5);
        assert_eq!(profile.processing_speed, 0.8);
        assert_eq!(profile.attention_control, 1.2);
        assert_eq!(profile.flexibility, 0.9);
    }

    #[test]
    fn test_factor_clamping() {
        let profile = IndividualDifferenceProfile::with_factors(3.0, 0.1, 2.5, -0.5);
        assert_eq!(profile.wm_capacity, 2.0); // Clamped to max
        assert_eq!(profile.processing_speed, 0.5); // Clamped to min
        assert_eq!(profile.attention_control, 2.0); // Clamped to max
        assert_eq!(profile.flexibility, 0.5); // Clamped to min
    }

    #[cfg(all(feature = "psychological_decay", feature = "testing"))]
    #[test]
    fn test_population_sampling() {
        let mut rng = rand::thread_rng();
        let profile = IndividualDifferenceProfile::sample_from_population(&mut rng);

        // All factors should be within valid ranges
        assert!(profile.wm_capacity >= 0.5 && profile.wm_capacity <= 2.0);
        assert!(profile.processing_speed >= 0.5 && profile.processing_speed <= 2.0);
        assert!(profile.attention_control >= 0.5 && profile.attention_control <= 2.0);
        assert!(profile.flexibility >= 0.5 && profile.flexibility <= 2.0);

        // Sample multiple profiles to check variation
        let mut profiles = Vec::new();
        for _ in 0..100 {
            profiles.push(IndividualDifferenceProfile::sample_from_population(
                &mut rng,
            ));
        }

        // Check that we get variation (not all identical)
        let first_wm = profiles[0].wm_capacity;
        let has_variation = profiles
            .iter()
            .any(|p| (p.wm_capacity - first_wm).abs() > 0.1);
        assert!(has_variation);
    }

    #[cfg(all(feature = "psychological_decay", feature = "testing"))]
    #[test]
    fn test_age_adjusted_sampling() {
        use rand::SeedableRng;
        use rand::rngs::StdRng;
        
        // Use seeded RNG for deterministic test results
        let mut rng = StdRng::seed_from_u64(12345);
        
        // Sample multiple times to get average behavior
        let mut young_processing_total = 0.0;
        let mut old_processing_total = 0.0;
        let mut young_wm_total = 0.0;
        let mut old_wm_total = 0.0;
        
        let samples = 50;
        for _ in 0..samples {
            let young_profile = IndividualDifferenceProfile::sample_with_age(&mut rng, 25.0);
            let old_profile = IndividualDifferenceProfile::sample_with_age(&mut rng, 75.0);
            
            young_processing_total += young_profile.processing_speed;
            old_processing_total += old_profile.processing_speed;
            young_wm_total += young_profile.wm_capacity;
            old_wm_total += old_profile.wm_capacity;
        }
        
        let young_processing_avg = young_processing_total / samples as f32;
        let old_processing_avg = old_processing_total / samples as f32;
        let young_wm_avg = young_wm_total / samples as f32;
        let old_wm_avg = old_wm_total / samples as f32;
        
        // Older profile should generally have lower efficiency on average
        assert!(old_processing_avg <= young_processing_avg * 1.05, 
               "Expected older processing speed ({:.3}) <= young processing speed ({:.3}) * 1.05", 
               old_processing_avg, young_processing_avg);
        assert!(old_wm_avg <= young_wm_avg * 1.05,
               "Expected older WM capacity ({:.3}) <= young WM capacity ({:.3}) * 1.05", 
               old_wm_avg, young_wm_avg);
    }

    #[test]
    fn test_hippocampal_tau_modification() {
        let high_ability = IndividualDifferenceProfile::with_factors(1.5, 1.5, 1.5, 1.0);
        let low_ability = IndividualDifferenceProfile::with_factors(0.7, 0.7, 0.7, 1.0);

        let base_tau = 1.0;
        let high_tau = high_ability.modify_hippocampal_tau(base_tau);
        let low_tau = low_ability.modify_hippocampal_tau(base_tau);

        // Higher abilities should increase tau (slower decay)
        assert!(high_tau > base_tau);
        assert!(low_tau < base_tau);
        assert!(high_tau > low_tau);
    }

    #[test]
    fn test_neocortical_tau_modification() {
        let flexible = IndividualDifferenceProfile::with_factors(1.0, 1.0, 1.0, 1.8);
        let inflexible = IndividualDifferenceProfile::with_factors(1.0, 1.0, 1.0, 0.6);

        let base_tau = 1.0;
        let flexible_tau = flexible.modify_neocortical_tau(base_tau);
        let inflexible_tau = inflexible.modify_neocortical_tau(base_tau);

        // Higher flexibility should improve neocortical consolidation
        assert!(flexible_tau > inflexible_tau);
    }

    #[test]
    fn test_schema_integration_efficiency() {
        let high_flexibility = IndividualDifferenceProfile::with_factors(1.0, 1.0, 1.0, 1.8);
        let low_flexibility = IndividualDifferenceProfile::with_factors(1.0, 1.0, 1.0, 0.6);

        let high_efficiency = high_flexibility.schema_integration_efficiency();
        let low_efficiency = low_flexibility.schema_integration_efficiency();

        // Higher flexibility should improve schema integration
        assert!(high_efficiency > low_efficiency);
        assert!(high_efficiency >= 0.2 && high_efficiency <= 1.8);
        assert!(low_efficiency >= 0.2 && low_efficiency <= 1.8);
    }

    #[test]
    fn test_pattern_completion_efficiency() {
        let high_wm = IndividualDifferenceProfile::with_factors(1.8, 1.0, 1.0, 1.0);
        let low_wm = IndividualDifferenceProfile::with_factors(0.6, 1.0, 1.0, 1.0);

        let high_completion = high_wm.pattern_completion_efficiency();
        let low_completion = low_wm.pattern_completion_efficiency();

        // Higher WM capacity should improve pattern completion
        assert!(high_completion > low_completion);
        assert!(high_completion >= 0.3 && high_completion <= 1.7);
    }

    #[test]
    fn test_retrieval_speed_factor() {
        let fast_processor = IndividualDifferenceProfile::with_factors(1.0, 1.8, 1.0, 1.0);
        let slow_processor = IndividualDifferenceProfile::with_factors(1.0, 0.6, 1.0, 1.0);

        let fast_speed = fast_processor.retrieval_speed_factor();
        let slow_speed = slow_processor.retrieval_speed_factor();

        // Higher processing speed should increase retrieval speed
        assert!(fast_speed > slow_speed);
        assert!(fast_speed >= 0.4 && fast_speed <= 2.0);
    }

    #[test]
    fn test_consolidation_effectiveness() {
        let mut good_sleeper = IndividualDifferenceProfile::default();
        good_sleeper.sleep_quality = 1.6;

        let mut poor_sleeper = IndividualDifferenceProfile::default();
        poor_sleeper.sleep_quality = 0.6;

        let good_consolidation = good_sleeper.consolidation_effectiveness();
        let poor_consolidation = poor_sleeper.consolidation_effectiveness();

        // Better sleep should improve consolidation
        assert!(good_consolidation > poor_consolidation);
    }

    #[test]
    fn test_interference_resistance() {
        let focused = IndividualDifferenceProfile::with_factors(1.0, 1.0, 1.8, 1.0);
        let distractible = IndividualDifferenceProfile::with_factors(1.0, 1.0, 0.6, 1.0);

        let focused_resistance = focused.interference_resistance();
        let distractible_resistance = distractible.interference_resistance();

        // Better attention control should increase interference resistance
        assert!(focused_resistance > distractible_resistance);
        assert!(focused_resistance >= 0.3 && focused_resistance <= 1.9);
    }

    #[test]
    fn test_overall_memory_efficiency() {
        let high_all = IndividualDifferenceProfile::with_factors(1.5, 1.5, 1.5, 1.5);
        let low_all = IndividualDifferenceProfile::with_factors(0.7, 0.7, 0.7, 0.7);

        let high_efficiency = high_all.overall_memory_efficiency();
        let low_efficiency = low_all.overall_memory_efficiency();

        // Higher abilities should give higher overall efficiency
        assert!(high_efficiency > low_efficiency);
        assert!(high_efficiency >= 0.4 && high_efficiency <= 1.8);
        assert!(low_efficiency >= 0.4 && low_efficiency <= 1.8);
    }

    #[test]
    fn test_confidence_calibration() {
        let high_ability = IndividualDifferenceProfile::with_factors(1.6, 1.6, 1.6, 1.6);
        let low_ability = IndividualDifferenceProfile::with_factors(0.7, 0.7, 0.7, 0.7);

        let overconfident = Confidence::exact(0.9);

        let high_calibrated = high_ability.calibrate_confidence(overconfident);
        let low_calibrated = low_ability.calibrate_confidence(overconfident);

        // Higher ability should lead to better calibration (less overconfidence)
        assert!(high_calibrated.raw() > low_calibrated.raw());
        assert!(high_calibrated.raw() <= overconfident.raw());
    }

    #[test]
    fn test_cognitive_decline_profile() {
        let decline_profile = IndividualDifferenceProfile::with_decline(0.5);

        // All factors should be reduced but within valid ranges
        assert!(decline_profile.wm_capacity < 1.0);
        assert!(decline_profile.processing_speed < 1.0);
        assert!(decline_profile.wm_capacity >= 0.5);
        assert!(decline_profile.processing_speed >= 0.5);
        assert!(decline_profile.age_factor == 0.5);

        // Processing speed should be most affected
        assert!(decline_profile.processing_speed <= decline_profile.wm_capacity);
    }

    #[test]
    fn test_cognitive_enhancement_profile() {
        let enhancement_profile = IndividualDifferenceProfile::with_enhancement(0.8);

        // All factors should be increased but within valid ranges
        assert!(enhancement_profile.wm_capacity > 1.0);
        assert!(enhancement_profile.processing_speed > 1.0);
        assert!(enhancement_profile.wm_capacity <= 2.0);
        assert!(enhancement_profile.processing_speed <= 2.0);
        assert!(enhancement_profile.age_factor == 1.0); // Age not affected by enhancement
    }

    #[test]
    fn test_extreme_decline_clamping() {
        let extreme_decline = IndividualDifferenceProfile::with_decline(1.0); // 100% decline

        // Should be clamped to minimum values
        assert!(extreme_decline.wm_capacity >= 0.5);
        assert!(extreme_decline.processing_speed >= 0.5);
        assert!(extreme_decline.age_factor >= 0.5);
    }

    #[test]
    fn test_extreme_enhancement_clamping() {
        let extreme_enhancement = IndividualDifferenceProfile::with_enhancement(2.0); // 200% enhancement

        // Should be clamped to maximum values
        assert!(extreme_enhancement.wm_capacity <= 2.0);
        assert!(extreme_enhancement.processing_speed <= 2.0);
    }
}
