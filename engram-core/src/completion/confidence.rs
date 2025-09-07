//! Metacognitive confidence calibration for pattern completion.

use crate::Confidence;
use super::{CompletedEpisode, MemorySource};
use std::collections::HashMap;

/// Metacognitive confidence calibration system
pub struct MetacognitiveConfidence {
    /// Historical accuracy data for calibration
    calibration_data: Vec<CalibrationPoint>,
    
    /// Confidence thresholds for different sources
    source_thresholds: HashMap<MemorySource, f32>,
    
    /// Fluency-based confidence weights
    fluency_weights: FluencyWeights,
}

/// A calibration data point
#[derive(Debug, Clone)]
struct CalibrationPoint {
    predicted_confidence: f32,
    actual_accuracy: f32,
    source: MemorySource,
}

/// Weights for retrieval fluency factors
#[derive(Debug, Clone)]
struct FluencyWeights {
    speed_weight: f32,
    ease_weight: f32,
    familiarity_weight: f32,
    vividness_weight: f32,
}

impl Default for FluencyWeights {
    fn default() -> Self {
        Self {
            speed_weight: 0.3,
            ease_weight: 0.25,
            familiarity_weight: 0.25,
            vividness_weight: 0.2,
        }
    }
}

impl MetacognitiveConfidence {
    /// Create a new metacognitive confidence system
    pub fn new() -> Self {
        let mut source_thresholds = HashMap::new();
        source_thresholds.insert(MemorySource::Recalled, 0.9);
        source_thresholds.insert(MemorySource::Reconstructed, 0.7);
        source_thresholds.insert(MemorySource::Imagined, 0.5);
        source_thresholds.insert(MemorySource::Consolidated, 0.8);
        
        Self {
            calibration_data: Vec::new(),
            source_thresholds,
            fluency_weights: FluencyWeights::default(),
        }
    }
    
    /// Calibrate confidence for a completed episode
    pub fn calibrate(&self, episode: &CompletedEpisode) -> Confidence {
        // Start with completion confidence
        let mut calibrated = episode.completion_confidence.raw();
        
        // Adjust based on source monitoring
        calibrated *= self.source_monitoring_adjustment(episode);
        
        // Adjust based on retrieval fluency
        calibrated *= self.retrieval_fluency_adjustment(episode);
        
        // Apply isotonic regression calibration if we have data
        if !self.calibration_data.is_empty() {
            calibrated = self.isotonic_regression_calibration(calibrated);
        }
        
        // Apply overconfidence reduction
        calibrated = self.reduce_overconfidence(calibrated);
        
        Confidence::exact(calibrated.clamp(0.0, 1.0))
    }
    
    /// Adjust confidence based on source monitoring
    fn source_monitoring_adjustment(&self, episode: &CompletedEpisode) -> f32 {
        let mut total_weight = 0.0;
        let mut weighted_confidence = 0.0;
        
        for (_, source) in &episode.source_attribution.field_sources {
            let threshold = self.source_thresholds.get(source).unwrap_or(&0.5);
            total_weight += 1.0;
            weighted_confidence += threshold;
        }
        
        if total_weight > 0.0 {
            weighted_confidence / total_weight
        } else {
            1.0
        }
    }
    
    /// Adjust confidence based on retrieval fluency
    fn retrieval_fluency_adjustment(&self, episode: &CompletedEpisode) -> f32 {
        // Calculate fluency components
        let speed_fluency = self.calculate_speed_fluency(episode);
        let ease_fluency = self.calculate_ease_fluency(episode);
        let familiarity_fluency = self.calculate_familiarity_fluency(episode);
        let vividness_fluency = self.calculate_vividness_fluency(episode);
        
        // Weighted combination
        let total_fluency = 
            speed_fluency * self.fluency_weights.speed_weight +
            ease_fluency * self.fluency_weights.ease_weight +
            familiarity_fluency * self.fluency_weights.familiarity_weight +
            vividness_fluency * self.fluency_weights.vividness_weight;
        
        total_fluency
    }
    
    /// Calculate speed-based fluency
    fn calculate_speed_fluency(&self, episode: &CompletedEpisode) -> f32 {
        // Faster retrieval = higher confidence
        // Based on number of activation traces (fewer = faster)
        let trace_count = episode.activation_evidence.len();
        if trace_count == 0 {
            1.0
        } else {
            (1.0 / (1.0 + trace_count as f32 * 0.1)).max(0.5)
        }
    }
    
    /// Calculate ease-based fluency
    fn calculate_ease_fluency(&self, episode: &CompletedEpisode) -> f32 {
        // Higher activation strength = easier retrieval
        if episode.activation_evidence.is_empty() {
            0.5
        } else {
            let avg_activation: f32 = episode.activation_evidence.iter()
                .map(|trace| trace.activation_strength)
                .sum::<f32>() / episode.activation_evidence.len() as f32;
            avg_activation.clamp(0.0, 1.0)
        }
    }
    
    /// Calculate familiarity-based fluency
    fn calculate_familiarity_fluency(&self, episode: &CompletedEpisode) -> f32 {
        // More alternative hypotheses = more familiar pattern
        let hypothesis_count = episode.alternative_hypotheses.len();
        (hypothesis_count as f32 / 5.0).min(1.0)
    }
    
    /// Calculate vividness-based fluency
    fn calculate_vividness_fluency(&self, episode: &CompletedEpisode) -> f32 {
        // Use episode's vividness confidence
        episode.episode.vividness_confidence.raw()
    }
    
    /// Apply isotonic regression for calibration
    fn isotonic_regression_calibration(&self, raw_confidence: f32) -> f32 {
        // Find calibration points near this confidence level
        let mut relevant_points: Vec<&CalibrationPoint> = self.calibration_data.iter()
            .filter(|p| (p.predicted_confidence - raw_confidence).abs() < 0.2)
            .collect();
        
        if relevant_points.is_empty() {
            return raw_confidence;
        }
        
        // Sort by predicted confidence
        relevant_points.sort_by(|a, b| 
            a.predicted_confidence.partial_cmp(&b.predicted_confidence).unwrap()
        );
        
        // Apply isotonic regression (simplified)
        let mut calibrated = raw_confidence;
        for window in relevant_points.windows(2) {
            if window[0].predicted_confidence <= raw_confidence && 
               raw_confidence <= window[1].predicted_confidence {
                // Linear interpolation
                let t = (raw_confidence - window[0].predicted_confidence) /
                       (window[1].predicted_confidence - window[0].predicted_confidence);
                calibrated = window[0].actual_accuracy * (1.0 - t) + 
                           window[1].actual_accuracy * t;
                break;
            }
        }
        
        calibrated
    }
    
    /// Reduce overconfidence bias
    fn reduce_overconfidence(&self, confidence: f32) -> f32 {
        // Apply power transformation to reduce overconfidence
        // Higher confidence values are reduced more
        if confidence > 0.8 {
            0.8 + (confidence - 0.8) * 0.5 // Halve the excess above 0.8
        } else if confidence > 0.6 {
            0.6 + (confidence - 0.6) * 0.75 // Reduce moderately high confidence
        } else {
            confidence // Keep low confidence as is
        }
    }
    
    /// Update calibration data with actual accuracy
    pub fn update_calibration(&mut self, predicted: f32, actual: f32, source: MemorySource) {
        self.calibration_data.push(CalibrationPoint {
            predicted_confidence: predicted,
            actual_accuracy: actual,
            source,
        });
        
        // Keep only recent calibration points
        if self.calibration_data.len() > 1000 {
            self.calibration_data.remove(0);
        }
    }
    
    /// Calculate Brier score for calibration quality
    pub fn brier_score(&self) -> f32 {
        if self.calibration_data.is_empty() {
            return 0.0;
        }
        
        let sum: f32 = self.calibration_data.iter()
            .map(|p| (p.predicted_confidence - p.actual_accuracy).powi(2))
            .sum();
        
        sum / self.calibration_data.len() as f32
    }
    
    /// Reality monitoring: distinguish internal vs external source
    pub fn reality_monitoring(&self, episode: &CompletedEpisode) -> MemorySource {
        // Count sources
        let mut source_counts = HashMap::new();
        for source in episode.source_attribution.field_sources.values() {
            *source_counts.entry(*source).or_insert(0) += 1;
        }
        
        // Return most common source
        source_counts.into_iter()
            .max_by_key(|(_, count)| *count)
            .map(|(source, _)| source)
            .unwrap_or(MemorySource::Reconstructed)
    }
    
    /// Source confusion detection
    pub fn detect_source_confusion(&self, episode: &CompletedEpisode) -> bool {
        // Check if sources are mixed
        let sources: Vec<_> = episode.source_attribution.field_sources.values()
            .collect();
        
        if sources.is_empty() {
            return false;
        }
        
        // Source confusion if multiple different sources
        let first_source = sources[0];
        !sources.iter().all(|s| *s == first_source)
    }
}

impl Default for MetacognitiveConfidence {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_metacognitive_confidence_creation() {
        let meta = MetacognitiveConfidence::new();
        assert_eq!(meta.calibration_data.len(), 0);
        assert_eq!(meta.source_thresholds.len(), 4);
    }
    
    #[test]
    fn test_reduce_overconfidence() {
        let meta = MetacognitiveConfidence::new();
        
        // High confidence should be reduced
        assert!(meta.reduce_overconfidence(0.95) < 0.95);
        
        // Low confidence should stay the same
        assert_eq!(meta.reduce_overconfidence(0.3), 0.3);
    }
    
    #[test]
    fn test_brier_score() {
        let mut meta = MetacognitiveConfidence::new();
        
        // Perfect calibration
        meta.update_calibration(0.8, 0.8, MemorySource::Recalled);
        meta.update_calibration(0.6, 0.6, MemorySource::Recalled);
        assert_eq!(meta.brier_score(), 0.0);
        
        // Poor calibration
        meta.update_calibration(0.9, 0.1, MemorySource::Reconstructed);
        assert!(meta.brier_score() > 0.0);
    }
}