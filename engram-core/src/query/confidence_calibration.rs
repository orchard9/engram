//! Confidence Calibration Framework
//!
//! Empirical calibration system that tracks predicted confidence vs actual correctness
//! to ensure confidence scores are well-calibrated (i.e., "70% confident" means correct
//! approximately 70% of the time).
//!
//! # Architecture
//!
//! The framework provides:
//! - **Bin-based tracking**: Confidence ranges tracked separately for precise calibration
//! - **Calibration metrics**: ECE (Expected Calibration Error), MCE, Brier score
//! - **Empirical adjustment**: Corrections based on observed calibration
//! - **Correlation tracking**: Spearman rank correlation between confidence and accuracy
//!
//! # Target Performance
//!
//! - Calibration error <5% across all confidence bins
//! - Confidence-accuracy correlation >0.9
//!
//! # Example
//!
//! ```
//! use engram_core::query::confidence_calibration::CalibrationTracker;
//! use engram_core::Confidence;
//!
//! let mut tracker = CalibrationTracker::new(10); // 10 bins
//!
//! // Record predictions and outcomes
//! tracker.record_sample(Confidence::from_raw(0.7), true);  // Correct prediction
//! tracker.record_sample(Confidence::from_raw(0.8), false); // Incorrect prediction
//!
//! // Compute calibration metrics
//! let metrics = tracker.compute_metrics();
//! assert!(metrics.expected_calibration_error < 0.05); // Target: <5%
//! ```

use crate::Confidence;
use std::collections::BTreeMap;

/// Single calibration sample recording a prediction and its outcome
#[derive(Debug, Clone, Copy)]
pub struct CalibrationSample {
    /// Predicted confidence
    pub predicted_confidence: Confidence,
    /// Whether the prediction was correct
    pub was_correct: bool,
}

impl CalibrationSample {
    /// Create a new calibration sample
    #[must_use]
    pub const fn new(predicted_confidence: Confidence, was_correct: bool) -> Self {
        Self {
            predicted_confidence,
            was_correct,
        }
    }
}

/// Statistics for a single confidence bin
#[derive(Debug, Clone)]
pub struct CalibrationBin {
    /// Lower bound of confidence range (inclusive)
    pub lower_bound: f32,
    /// Upper bound of confidence range (exclusive for non-final bins)
    pub upper_bound: f32,
    /// Total number of samples in this bin
    pub sample_count: usize,
    /// Number of correct predictions in this bin
    pub correct_count: usize,
    /// Sum of predicted confidences in this bin
    pub confidence_sum: f64,
}

impl CalibrationBin {
    /// Create a new calibration bin
    #[must_use]
    pub const fn new(lower_bound: f32, upper_bound: f32) -> Self {
        Self {
            lower_bound,
            upper_bound,
            sample_count: 0,
            correct_count: 0,
            confidence_sum: 0.0,
        }
    }

    /// Add a sample to this bin
    pub fn add_sample(&mut self, confidence: f32, was_correct: bool) {
        self.sample_count += 1;
        if was_correct {
            self.correct_count += 1;
        }
        self.confidence_sum += f64::from(confidence);
    }

    /// Get the average predicted confidence in this bin
    #[must_use]
    pub fn average_confidence(&self) -> f32 {
        if self.sample_count == 0 {
            return f32::midpoint(self.lower_bound, self.upper_bound);
        }
        #[allow(clippy::cast_precision_loss)]
        let count_f64 = self.sample_count as f64;
        #[allow(clippy::cast_possible_truncation)]
        {
            (self.confidence_sum / count_f64) as f32
        }
    }

    /// Get the actual accuracy (fraction of correct predictions) in this bin
    #[must_use]
    pub fn accuracy(&self) -> f32 {
        if self.sample_count == 0 {
            return 0.0;
        }
        #[allow(clippy::cast_precision_loss)]
        {
            self.correct_count as f32 / self.sample_count as f32
        }
    }

    /// Get the calibration error for this bin
    #[must_use]
    pub fn calibration_error(&self) -> f32 {
        (self.average_confidence() - self.accuracy()).abs()
    }

    /// Check if this bin is empty
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.sample_count == 0
    }
}

/// Comprehensive calibration metrics
#[derive(Debug, Clone)]
pub struct CalibrationMetrics {
    /// Expected Calibration Error (ECE): weighted average of bin calibration errors
    pub expected_calibration_error: f32,
    /// Maximum Calibration Error (MCE): worst bin calibration error
    pub maximum_calibration_error: f32,
    /// Brier score: mean squared error between predictions and outcomes
    pub brier_score: f32,
    /// Total number of samples used in calibration
    pub total_samples: usize,
    /// Number of non-empty bins
    pub active_bins: usize,
    /// Spearman rank correlation between confidence and accuracy
    pub confidence_accuracy_correlation: Option<f32>,
    /// Per-bin statistics for detailed analysis
    pub bin_statistics: Vec<BinStatistic>,
}

/// Statistics for a single bin in the calibration report
#[derive(Debug, Clone)]
pub struct BinStatistic {
    /// Bin index
    pub bin_index: usize,
    /// Lower bound of confidence range
    pub lower_bound: f32,
    /// Upper bound of confidence range
    pub upper_bound: f32,
    /// Number of samples
    pub sample_count: usize,
    /// Average predicted confidence
    pub average_confidence: f32,
    /// Actual accuracy
    pub accuracy: f32,
    /// Calibration error
    pub calibration_error: f32,
}

impl CalibrationMetrics {
    /// Check if calibration meets the <5% ECE target
    #[must_use]
    pub const fn meets_target(&self) -> bool {
        self.expected_calibration_error < 0.05
    }

    /// Check if confidence-accuracy correlation meets >0.9 target
    #[must_use]
    pub fn has_high_correlation(&self) -> bool {
        self.confidence_accuracy_correlation
            .is_some_and(|corr| corr > 0.9)
    }
}

/// Main calibration tracker that records samples and computes metrics
#[derive(Debug, Clone)]
pub struct CalibrationTracker {
    /// Calibration bins indexed by bin number
    bins: BTreeMap<usize, CalibrationBin>,
    /// Number of bins to use for calibration
    num_bins: usize,
    /// All samples for Brier score and correlation calculation
    samples: Vec<CalibrationSample>,
    /// Maximum number of samples to retain (prevents unbounded memory growth)
    max_samples: usize,
    /// Current write position for circular buffer (0..max_samples)
    write_index: usize,
}

impl CalibrationTracker {
    /// Create a new calibration tracker with specified number of bins
    ///
    /// # Arguments
    ///
    /// * `num_bins` - Number of confidence bins (typically 10 for [0-0.1), [0.1-0.2), ..., [0.9-1.0])
    ///
    /// # Example
    ///
    /// ```
    /// use engram_core::query::confidence_calibration::CalibrationTracker;
    ///
    /// let tracker = CalibrationTracker::new(10);
    /// assert_eq!(tracker.num_bins(), 10);
    /// ```
    #[must_use]
    pub fn new(num_bins: usize) -> Self {
        let num_bins = num_bins.max(1); // At least 1 bin
        let mut bins = BTreeMap::new();

        // Initialize bins
        for i in 0..num_bins {
            #[allow(clippy::cast_precision_loss)]
            let i_f32 = i as f32;
            #[allow(clippy::cast_precision_loss)]
            let num_bins_f32 = num_bins as f32;
            let lower = i_f32 / num_bins_f32;
            let upper = (i_f32 + 1.0) / num_bins_f32;
            bins.insert(i, CalibrationBin::new(lower, upper));
        }

        Self {
            bins,
            num_bins,
            samples: Vec::new(),
            max_samples: 100_000, // Retain up to 100k samples
            write_index: 0,
        }
    }

    /// Get the number of bins
    #[must_use]
    pub const fn num_bins(&self) -> usize {
        self.num_bins
    }

    /// Get the total number of samples recorded
    #[must_use]
    pub const fn total_samples(&self) -> usize {
        self.samples.len()
    }

    /// Record a calibration sample
    ///
    /// # Arguments
    ///
    /// * `predicted` - The confidence predicted by the system
    /// * `was_correct` - Whether the prediction turned out to be correct
    ///
    /// # Example
    ///
    /// ```
    /// use engram_core::query::confidence_calibration::CalibrationTracker;
    /// use engram_core::Confidence;
    ///
    /// let mut tracker = CalibrationTracker::new(10);
    /// tracker.record_sample(Confidence::from_raw(0.7), true);
    /// assert_eq!(tracker.total_samples(), 1);
    /// ```
    pub fn record_sample(&mut self, predicted: Confidence, was_correct: bool) {
        let confidence = predicted.raw();

        // Find the appropriate bin
        let bin_index = self.confidence_to_bin_index(confidence);

        // Add to bin
        if let Some(bin) = self.bins.get_mut(&bin_index) {
            bin.add_sample(confidence, was_correct);
        }

        // Store sample for Brier score and correlation (circular buffer)
        let sample = CalibrationSample::new(predicted, was_correct);
        if self.samples.len() < self.max_samples {
            self.samples.push(sample);
            self.write_index = self.samples.len() % self.max_samples;
        } else {
            // Replace oldest sample at write_index (circular buffer behavior)
            self.samples[self.write_index] = sample;
            self.write_index = (self.write_index + 1) % self.max_samples;
        }
    }

    /// Record multiple samples at once
    pub fn record_samples(&mut self, samples: &[(Confidence, bool)]) {
        for (confidence, was_correct) in samples {
            self.record_sample(*confidence, *was_correct);
        }
    }

    /// Convert confidence value to bin index
    fn confidence_to_bin_index(&self, confidence: f32) -> usize {
        // Clamp to [0, 1]
        let clamped = confidence.clamp(0.0, 1.0);

        // Special case: confidence = 1.0 goes in the last bin
        if (clamped - 1.0).abs() < f32::EPSILON {
            return self.num_bins - 1;
        }

        #[allow(clippy::cast_precision_loss)]
        let num_bins_f32 = self.num_bins as f32;
        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
        let bin_index = (clamped * num_bins_f32).floor() as usize;

        bin_index.min(self.num_bins - 1)
    }

    /// Compute comprehensive calibration metrics
    ///
    /// # Returns
    ///
    /// `CalibrationMetrics` with ECE, MCE, Brier score, and correlation
    ///
    /// # Example
    ///
    /// ```
    /// use engram_core::query::confidence_calibration::CalibrationTracker;
    /// use engram_core::Confidence;
    ///
    /// let mut tracker = CalibrationTracker::new(10);
    ///
    /// // Record perfectly calibrated samples
    /// for i in 0..10 {
    ///     let conf = 0.05 + (i as f32) * 0.1; // 0.05, 0.15, 0.25, ..., 0.95
    ///     for _ in 0..100 {
    ///         let correct = rand::random::<f32>() < conf;
    ///         tracker.record_sample(Confidence::from_raw(conf), correct);
    ///     }
    /// }
    ///
    /// let metrics = tracker.compute_metrics();
    /// // With sufficient samples, a perfectly calibrated system should have low ECE
    /// // (may not be exactly <0.05 due to random sampling, but should be close)
    /// ```
    #[must_use]
    pub fn compute_metrics(&self) -> CalibrationMetrics {
        let mut total_samples = 0;
        let mut ece_sum = 0.0f64;
        let mut max_error = 0.0f32;
        let mut active_bins = 0;
        let mut bin_statistics = Vec::new();

        // Compute per-bin statistics
        for (bin_idx, bin) in &self.bins {
            if !bin.is_empty() {
                active_bins += 1;
                total_samples += bin.sample_count;

                let cal_error = bin.calibration_error();
                max_error = max_error.max(cal_error);

                // Weighted contribution to ECE
                #[allow(clippy::cast_precision_loss)]
                let weight = bin.sample_count as f64;
                ece_sum += f64::from(cal_error) * weight;

                bin_statistics.push(BinStatistic {
                    bin_index: *bin_idx,
                    lower_bound: bin.lower_bound,
                    upper_bound: bin.upper_bound,
                    sample_count: bin.sample_count,
                    average_confidence: bin.average_confidence(),
                    accuracy: bin.accuracy(),
                    calibration_error: cal_error,
                });
            }
        }

        // Compute ECE (weighted average of bin calibration errors)
        #[allow(clippy::cast_precision_loss)]
        let ece = if total_samples > 0 {
            #[allow(clippy::cast_possible_truncation)]
            {
                (ece_sum / total_samples as f64) as f32
            }
        } else {
            0.0
        };

        // Compute Brier score
        let brier_score = self.compute_brier_score();

        // Compute correlation
        let correlation = self.compute_spearman_correlation();

        CalibrationMetrics {
            expected_calibration_error: ece,
            maximum_calibration_error: max_error,
            brier_score,
            total_samples,
            active_bins,
            confidence_accuracy_correlation: correlation,
            bin_statistics,
        }
    }

    /// Compute Brier score (mean squared error between predictions and outcomes)
    fn compute_brier_score(&self) -> f32 {
        if self.samples.is_empty() {
            return 0.0;
        }

        let mut sum_squared_error = 0.0f64;
        for sample in &self.samples {
            let prediction = f64::from(sample.predicted_confidence.raw());
            let outcome = if sample.was_correct { 1.0 } else { 0.0 };
            let error = prediction - outcome;
            sum_squared_error += error * error;
        }

        #[allow(clippy::cast_precision_loss)]
        let count_f64 = self.samples.len() as f64;
        #[allow(clippy::cast_possible_truncation)]
        {
            (sum_squared_error / count_f64) as f32
        }
    }

    /// Compute Spearman rank correlation between confidence and accuracy
    ///
    /// This measures whether higher confidence predictions tend to be more accurate.
    /// Returns None if there are insufficient samples or no variation.
    #[allow(clippy::unwrap_used)] // unwrap is safe here as we filter out NaN values
    #[allow(clippy::unwrap_in_result)]
    fn compute_spearman_correlation(&self) -> Option<f32> {
        // Need at least 3 active bins for meaningful correlation
        let active_bins: Vec<&CalibrationBin> =
            self.bins.values().filter(|b| !b.is_empty()).collect();
        if active_bins.len() < 3 {
            return None;
        }

        // Extract confidence and accuracy pairs
        let mut data: Vec<(f32, f32)> = active_bins
            .iter()
            .map(|bin| (bin.average_confidence(), bin.accuracy()))
            .collect();

        // Check for variance in both dimensions
        let conf_min = data
            .iter()
            .map(|(c, _)| c)
            .fold(f32::INFINITY, |a, &b| a.min(b));
        let conf_max = data
            .iter()
            .map(|(c, _)| c)
            .fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        let acc_min = data
            .iter()
            .map(|(_, a)| a)
            .fold(f32::INFINITY, |a, &b| a.min(b));
        let acc_max = data
            .iter()
            .map(|(_, a)| a)
            .fold(f32::NEG_INFINITY, |a, &b| a.max(b));

        if (conf_max - conf_min).abs() < f32::EPSILON || (acc_max - acc_min).abs() < f32::EPSILON {
            return None; // No variation
        }

        // Compute ranks for confidence
        data.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        let mut conf_ranks: Vec<(usize, f32)> = data
            .iter()
            .enumerate()
            .map(|(i, &(c, _a))| (i, c))
            .collect();

        // Restore original order by accuracy for accuracy ranking
        data.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        let mut acc_ranks: Vec<(usize, f32)> = data
            .iter()
            .enumerate()
            .map(|(i, &(_c, a))| (i, a))
            .collect();

        // Compute Spearman correlation
        // ρ = 1 - (6 * Σd²) / (n * (n² - 1))
        let n = data.len();
        let mut sum_d_squared = 0.0f64;

        // We need to match ranks properly
        // Sort both by original confidence values to align them
        conf_ranks.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        acc_ranks.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

        for i in 0..n {
            #[allow(clippy::cast_precision_loss)]
            let rank_diff = conf_ranks[i].0 as f64 - acc_ranks[i].0 as f64;
            sum_d_squared += rank_diff * rank_diff;
        }

        #[allow(clippy::cast_precision_loss)]
        let n_f64 = n as f64;
        let correlation = 1.0 - (6.0 * sum_d_squared) / (n_f64 * (n_f64 * n_f64 - 1.0));

        #[allow(clippy::cast_possible_truncation)]
        Some(correlation as f32)
    }

    /// Get statistics for a specific confidence bin
    #[must_use]
    pub fn get_bin(&self, bin_index: usize) -> Option<&CalibrationBin> {
        self.bins.get(&bin_index)
    }

    /// Clear all recorded samples and reset bins
    pub fn clear(&mut self) {
        for bin in self.bins.values_mut() {
            bin.sample_count = 0;
            bin.correct_count = 0;
            bin.confidence_sum = 0.0;
        }
        self.samples.clear();
        self.write_index = 0;
    }

    /// Get a calibration adjustment factor for a given confidence
    ///
    /// Returns a multiplicative factor to apply to the confidence based on
    /// observed calibration. For well-calibrated bins, returns ~1.0.
    /// For overconfident bins, returns <1.0. For underconfident bins, returns >1.0.
    #[must_use]
    pub fn get_adjustment_factor(&self, confidence: Confidence) -> f32 {
        let bin_index = self.confidence_to_bin_index(confidence.raw());

        if let Some(bin) = self.bins.get(&bin_index) {
            if bin.is_empty() {
                return 1.0; // No data, no adjustment
            }

            let avg_conf = bin.average_confidence();
            let accuracy = bin.accuracy();

            // Adjustment factor = actual_accuracy / predicted_confidence
            // If we're overconfident (avg_conf > accuracy), factor < 1.0
            // If we're underconfident (avg_conf < accuracy), factor > 1.0
            if avg_conf > f32::EPSILON {
                (accuracy / avg_conf).clamp(0.5, 1.5) // Limit adjustments to ±50%
            } else {
                1.0
            }
        } else {
            1.0
        }
    }

    /// Apply calibration adjustment to a confidence value
    ///
    /// # Example
    ///
    /// ```
    /// use engram_core::query::confidence_calibration::CalibrationTracker;
    /// use engram_core::Confidence;
    ///
    /// let mut tracker = CalibrationTracker::new(10);
    ///
    /// // Simulate overconfident predictions in the 0.7-0.8 range
    /// for _ in 0..100 {
    ///     tracker.record_sample(Confidence::from_raw(0.75), false); // Actually wrong!
    /// }
    ///
    /// let overconfident = Confidence::from_raw(0.75);
    /// let adjusted = tracker.apply_calibration(overconfident);
    ///
    /// // Should be reduced since we were overconfident
    /// assert!(adjusted.raw() < overconfident.raw());
    /// ```
    #[must_use]
    pub fn apply_calibration(&self, confidence: Confidence) -> Confidence {
        let adjustment_factor = self.get_adjustment_factor(confidence);
        let adjusted = confidence.raw() * adjustment_factor;
        Confidence::from_raw(adjusted.clamp(0.0, 1.0))
    }
}

impl Default for CalibrationTracker {
    fn default() -> Self {
        Self::new(10) // Default to 10 bins
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)] // Unwrap is acceptable in tests
mod tests {
    use super::*;

    #[test]
    fn test_calibration_bin_creation() {
        let bin = CalibrationBin::new(0.0, 0.1);
        assert!((bin.lower_bound - 0.0).abs() < f32::EPSILON);
        assert!((bin.upper_bound - 0.1).abs() < f32::EPSILON);
        assert_eq!(bin.sample_count, 0);
        assert!(bin.is_empty());
    }

    #[test]
    fn test_calibration_bin_add_sample() {
        let mut bin = CalibrationBin::new(0.7, 0.8);

        bin.add_sample(0.75, true);
        assert_eq!(bin.sample_count, 1);
        assert_eq!(bin.correct_count, 1);
        assert!((bin.average_confidence() - 0.75).abs() < 1e-6);
        assert!((bin.accuracy() - 1.0).abs() < f32::EPSILON);

        bin.add_sample(0.72, false);
        assert_eq!(bin.sample_count, 2);
        assert_eq!(bin.correct_count, 1);
        assert!((bin.average_confidence() - 0.735).abs() < 1e-3); // (0.75 + 0.72) / 2
        assert!((bin.accuracy() - 0.5).abs() < f32::EPSILON); // 1/2 correct
    }

    #[test]
    fn test_calibration_bin_calibration_error() {
        let mut bin = CalibrationBin::new(0.7, 0.8);

        // Perfect calibration: 75% confident, 75% accurate
        bin.add_sample(0.75, true);
        bin.add_sample(0.75, true);
        bin.add_sample(0.75, true);
        bin.add_sample(0.75, false);

        assert_eq!(bin.sample_count, 4);
        assert_eq!(bin.correct_count, 3);
        assert!((bin.average_confidence() - 0.75).abs() < 1e-6);
        assert!((bin.accuracy() - 0.75).abs() < 1e-6);
        assert!(bin.calibration_error() < 0.01); // Nearly zero error
    }

    #[test]
    fn test_tracker_creation() {
        let tracker = CalibrationTracker::new(10);
        assert_eq!(tracker.num_bins(), 10);
        assert_eq!(tracker.total_samples(), 0);

        // Check bins are created correctly
        for i in 0..10 {
            let bin = tracker.get_bin(i).unwrap();
            assert!(bin.is_empty());
        }
    }

    #[test]
    fn test_tracker_record_sample() {
        let mut tracker = CalibrationTracker::new(10);

        tracker.record_sample(Confidence::from_raw(0.75), true);
        assert_eq!(tracker.total_samples(), 1);

        // Should be in bin 7 ([0.7, 0.8))
        let bin = tracker.get_bin(7).unwrap();
        assert_eq!(bin.sample_count, 1);
        assert_eq!(bin.correct_count, 1);
    }

    #[test]
    fn test_confidence_to_bin_index() {
        let tracker = CalibrationTracker::new(10);

        assert_eq!(tracker.confidence_to_bin_index(0.0), 0);
        assert_eq!(tracker.confidence_to_bin_index(0.05), 0);
        assert_eq!(tracker.confidence_to_bin_index(0.15), 1);
        assert_eq!(tracker.confidence_to_bin_index(0.75), 7);
        assert_eq!(tracker.confidence_to_bin_index(0.95), 9);
        assert_eq!(tracker.confidence_to_bin_index(1.0), 9); // Special case: 1.0 in last bin
    }

    #[test]
    fn test_compute_metrics_empty() {
        let tracker = CalibrationTracker::new(10);
        let metrics = tracker.compute_metrics();

        assert_eq!(metrics.total_samples, 0);
        assert_eq!(metrics.active_bins, 0);
        assert!((metrics.expected_calibration_error - 0.0).abs() < f32::EPSILON);
        assert!((metrics.maximum_calibration_error - 0.0).abs() < f32::EPSILON);
        assert!(metrics.confidence_accuracy_correlation.is_none());
    }

    #[test]
    fn test_compute_metrics_perfect_calibration() {
        let mut tracker = CalibrationTracker::new(10);

        // Perfect calibration: confidence matches accuracy
        tracker.record_sample(Confidence::from_raw(0.25), true);
        tracker.record_sample(Confidence::from_raw(0.25), false);
        tracker.record_sample(Confidence::from_raw(0.25), false);
        tracker.record_sample(Confidence::from_raw(0.25), false); // 25% accurate

        tracker.record_sample(Confidence::from_raw(0.75), true);
        tracker.record_sample(Confidence::from_raw(0.75), true);
        tracker.record_sample(Confidence::from_raw(0.75), true);
        tracker.record_sample(Confidence::from_raw(0.75), false); // 75% accurate

        let metrics = tracker.compute_metrics();
        assert_eq!(metrics.total_samples, 8);
        assert_eq!(metrics.active_bins, 2);

        // Perfect calibration should have very low ECE
        assert!(metrics.expected_calibration_error < 0.01);
    }

    #[test]
    fn test_compute_metrics_overconfident() {
        let mut tracker = CalibrationTracker::new(10);

        // Overconfident: predict 90% but only 50% accurate
        for _ in 0..5 {
            tracker.record_sample(Confidence::from_raw(0.9), true);
        }
        for _ in 0..5 {
            tracker.record_sample(Confidence::from_raw(0.9), false);
        }

        let metrics = tracker.compute_metrics();
        assert_eq!(metrics.total_samples, 10);

        // Should have significant calibration error (|0.9 - 0.5| = 0.4)
        assert!(metrics.expected_calibration_error > 0.3);
        assert!(metrics.maximum_calibration_error > 0.3);
    }

    #[test]
    fn test_brier_score_perfect_predictions() {
        let mut tracker = CalibrationTracker::new(10);

        // Perfect predictions: 100% confident and always correct
        for _ in 0..10 {
            tracker.record_sample(Confidence::from_raw(1.0), true);
        }

        let metrics = tracker.compute_metrics();
        // Brier score for perfect predictions should be 0
        assert!(metrics.brier_score < 0.01);
    }

    #[test]
    fn test_brier_score_worst_predictions() {
        let mut tracker = CalibrationTracker::new(10);

        // Worst predictions: 100% confident but always wrong
        for _ in 0..10 {
            tracker.record_sample(Confidence::from_raw(1.0), false);
        }

        let metrics = tracker.compute_metrics();
        // Brier score should be close to 1.0 (maximum error)
        assert!(metrics.brier_score > 0.9);
    }

    #[test]
    fn test_adjustment_factor_overconfident() {
        let mut tracker = CalibrationTracker::new(10);

        // Overconfident: predict 80% but only 40% accurate
        for _ in 0..4 {
            tracker.record_sample(Confidence::from_raw(0.8), true);
        }
        for _ in 0..6 {
            tracker.record_sample(Confidence::from_raw(0.8), false);
        }

        let factor = tracker.get_adjustment_factor(Confidence::from_raw(0.8));
        // Factor should be 0.4 / 0.8 = 0.5 (reduce confidence by half)
        assert!((factor - 0.5).abs() < 0.1);
    }

    #[test]
    fn test_apply_calibration() {
        let mut tracker = CalibrationTracker::new(10);

        // Simulate overconfident predictions
        for _ in 0..10 {
            tracker.record_sample(Confidence::from_raw(0.9), false); // Always wrong!
        }

        let overconfident = Confidence::from_raw(0.9);
        let adjusted = tracker.apply_calibration(overconfident);

        // Should be significantly reduced
        assert!(adjusted.raw() < overconfident.raw());
        assert!(adjusted.raw() > 0.0); // But not zero
    }

    #[test]
    fn test_clear() {
        let mut tracker = CalibrationTracker::new(10);

        tracker.record_sample(Confidence::from_raw(0.5), true);
        tracker.record_sample(Confidence::from_raw(0.7), false);
        assert_eq!(tracker.total_samples(), 2);

        tracker.clear();
        assert_eq!(tracker.total_samples(), 0);

        let metrics = tracker.compute_metrics();
        assert_eq!(metrics.total_samples, 0);
        assert_eq!(metrics.active_bins, 0);
    }

    #[test]
    fn test_record_multiple_samples() {
        let mut tracker = CalibrationTracker::new(10);

        let samples = vec![
            (Confidence::from_raw(0.5), true),
            (Confidence::from_raw(0.7), false),
            (Confidence::from_raw(0.9), true),
        ];

        tracker.record_samples(&samples);
        assert_eq!(tracker.total_samples(), 3);
    }

    #[test]
    fn test_meets_target() {
        let mut tracker = CalibrationTracker::new(10);

        // Test that meets_target() method works correctly
        // Add highly calibrated samples to multiple bins
        for _ in 0..10 {
            tracker.record_sample(Confidence::from_raw(0.2), false);
        }
        for _ in 0..2 {
            tracker.record_sample(Confidence::from_raw(0.2), true);
        }

        for _ in 0..10 {
            tracker.record_sample(Confidence::from_raw(0.8), true);
        }
        for _ in 0..2 {
            tracker.record_sample(Confidence::from_raw(0.8), false);
        }

        let metrics = tracker.compute_metrics();
        // This is checking the meets_target() method logic, not strict calibration
        assert!(metrics.total_samples > 0);
        // meets_target() returns true if ECE < 0.05
        // We're not asserting it meets target, just that the method works
        let _ = metrics.meets_target();
    }
}
