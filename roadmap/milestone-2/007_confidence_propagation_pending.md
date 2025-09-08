# Task 007: Confidence Propagation Through Vector Operations

## Status: Pending
## Priority: P1 - Correctness Critical  
## Estimated Effort: 2 days
## Dependencies: Milestone-1/Task-006 (probabilistic query engine)

## Objective
Implement proper confidence propagation through all vector storage and retrieval operations, ensuring uncertainty is tracked and propagated correctly according to probability theory.

## Current State Analysis
- **Existing**: Basic Confidence type and intervals from milestone-1/task-006
- **Existing**: Confidence in Episode and Memory types
- **Missing**: Confidence propagation through storage tiers
- **Missing**: Uncertainty from vector operations
- **Missing**: Calibrated confidence scores

## Technical Specification

### 1. Confidence in Vector Operations

```rust
// engram-core/src/storage/confidence_tracking.rs

use std::f32::consts::PI;

/// Extended confidence for vector operations
#[derive(Debug, Clone)]
pub struct VectorConfidence {
    /// Base confidence from storage
    storage_confidence: Confidence,
    
    /// Confidence from similarity computation
    similarity_confidence: Confidence,
    
    /// Confidence from retrieval path
    retrieval_confidence: Confidence,
    
    /// Combined confidence with proper propagation
    combined: ConfidenceInterval,
}

impl VectorConfidence {
    /// Create from vector similarity score
    pub fn from_similarity(similarity: f32, method: SimilarityMethod) -> Self {
        let similarity_confidence = match method {
            SimilarityMethod::Cosine => {
                // Cosine similarity confidence based on angle
                let angle = (similarity.clamp(-1.0, 1.0).acos() / PI) as f32;
                Confidence::exact(1.0 - angle)
            }
            SimilarityMethod::Euclidean(max_dist) => {
                // Normalize distance to confidence
                Confidence::exact((max_dist - similarity) / max_dist)
            }
            SimilarityMethod::DotProduct => {
                // Normalized dot product
                Confidence::exact(similarity.clamp(0.0, 1.0))
            }
        };
        
        Self {
            storage_confidence: Confidence::HIGH,
            similarity_confidence,
            retrieval_confidence: Confidence::HIGH,
            combined: ConfidenceInterval::from_confidence(similarity_confidence),
        }
    }
    
    /// Propagate confidence through retrieval operation
    pub fn propagate_retrieval(
        &mut self,
        storage_tier: StorageTier,
        retrieval_method: RetrievalMethod,
    ) {
        // Adjust confidence based on storage tier
        self.storage_confidence = match storage_tier {
            StorageTier::Hot => Confidence::CERTAIN,
            StorageTier::Warm => Confidence::HIGH,
            StorageTier::Cold => Confidence::MEDIUM,
        };
        
        // Adjust for retrieval method
        self.retrieval_confidence = match retrieval_method {
            RetrievalMethod::Direct => Confidence::CERTAIN,
            RetrievalMethod::Index => Confidence::HIGH,
            RetrievalMethod::Approximate => Confidence::MEDIUM,
            RetrievalMethod::Reconstructed => Confidence::LOW,
        };
        
        // Combine using probability theory
        self.combined = self.combine_confidences();
    }
    
    fn combine_confidences(&self) -> ConfidenceInterval {
        // Use multiplication for independent confidence sources
        let point = self.storage_confidence
            .and(self.similarity_confidence)
            .and(self.retrieval_confidence);
            
        // Calculate uncertainty from individual sources
        let storage_uncertainty = 1.0 - self.storage_confidence.raw();
        let similarity_uncertainty = 1.0 - self.similarity_confidence.raw();
        let retrieval_uncertainty = 1.0 - self.retrieval_confidence.raw();
        
        // Propagate uncertainty
        let total_uncertainty = 1.0 - (1.0 - storage_uncertainty) * 
                                     (1.0 - similarity_uncertainty) * 
                                     (1.0 - retrieval_uncertainty);
        
        ConfidenceInterval::from_confidence_with_uncertainty(point, total_uncertainty)
    }
}
```

### 2. Calibrated Confidence Scores

```rust
// engram-core/src/storage/confidence_calibration.rs

use statrs::distribution::{Beta, Continuous};

/// Confidence calibration using isotonic regression
pub struct ConfidenceCalibrator {
    /// Calibration curve learned from data
    calibration_curve: IsotonicRegression,
    
    /// Beta distribution parameters for prior
    beta_prior: Beta,
    
    /// Calibration statistics
    stats: CalibrationStats,
}

#[derive(Debug, Clone)]
struct CalibrationStats {
    /// Expected calibration error
    ece: f32,
    
    /// Maximum calibration error
    mce: f32,
    
    /// Brier score
    brier_score: f32,
    
    /// Number of samples
    n_samples: usize,
}

impl ConfidenceCalibrator {
    /// Calibrate raw confidence score
    pub fn calibrate(&self, raw_confidence: f32) -> Confidence {
        // Apply isotonic regression calibration
        let calibrated = self.calibration_curve.predict(raw_confidence);
        
        // Apply Bayesian adjustment with prior
        let posterior = self.apply_beta_prior(calibrated);
        
        Confidence::exact(posterior)
    }
    
    fn apply_beta_prior(&self, score: f32) -> f32 {
        // Bayesian update with Beta prior
        let alpha = self.beta_prior.shape_a();
        let beta = self.beta_prior.shape_b();
        
        // Posterior mean with pseudocounts
        (score * self.stats.n_samples as f32 + alpha) / 
        (self.stats.n_samples as f32 + alpha + beta)
    }
    
    /// Update calibration with observed outcomes
    pub fn update(&mut self, predictions: &[(f32, bool)]) {
        // Group predictions into bins
        const N_BINS: usize = 10;
        let mut bins = vec![Vec::new(); N_BINS];
        
        for &(confidence, outcome) in predictions {
            let bin_idx = ((confidence * N_BINS as f32) as usize).min(N_BINS - 1);
            bins[bin_idx].push((confidence, outcome));
        }
        
        // Calculate calibration error
        let mut ece = 0.0;
        let mut mce = 0.0;
        
        for bin in &bins {
            if bin.is_empty() { continue; }
            
            let avg_confidence = bin.iter().map(|(c, _)| c).sum::<f32>() / bin.len() as f32;
            let accuracy = bin.iter().filter(|(_, o)| *o).count() as f32 / bin.len() as f32;
            
            let error = (avg_confidence - accuracy).abs();
            ece += error * bin.len() as f32 / predictions.len() as f32;
            mce = mce.max(error);
        }
        
        self.stats.ece = ece;
        self.stats.mce = mce;
        self.stats.n_samples += predictions.len();
        
        // Update isotonic regression
        self.calibration_curve.fit(predictions);
    }
}
```

### 3. Uncertainty from Storage Operations

```rust
// engram-core/src/storage/storage_uncertainty.rs

/// Track uncertainty introduced by storage operations
pub struct StorageUncertainty {
    /// Quantization error from compression
    quantization_error: f32,
    
    /// Precision loss from format conversion
    precision_loss: f32,
    
    /// Uncertainty from lossy operations
    compression_uncertainty: f32,
    
    /// Time-based decay uncertainty
    temporal_uncertainty: f32,
}

impl StorageUncertainty {
    /// Calculate uncertainty for vector in storage
    pub fn calculate(
        vector: &[f32; 768],
        storage_tier: StorageTier,
        time_in_storage: Duration,
    ) -> Self {
        let quantization_error = match storage_tier {
            StorageTier::Hot => 0.0, // Full precision
            StorageTier::Warm => Self::calculate_f16_quantization_error(vector),
            StorageTier::Cold => Self::calculate_int8_quantization_error(vector),
        };
        
        let precision_loss = match storage_tier {
            StorageTier::Hot => 0.0,
            StorageTier::Warm => 1e-4, // f32 to f16
            StorageTier::Cold => 1e-2, // f32 to int8
        };
        
        let compression_uncertainty = match storage_tier {
            StorageTier::Hot => 0.0,
            StorageTier::Warm => 0.01, // Lossless compression
            StorageTier::Cold => 0.05, // Lossy compression
        };
        
        // Exponential decay of confidence over time
        let days_stored = time_in_storage.as_secs() as f32 / 86400.0;
        let temporal_uncertainty = 1.0 - (-days_stored / 365.0).exp();
        
        Self {
            quantization_error,
            precision_loss,
            compression_uncertainty,
            temporal_uncertainty,
        }
    }
    
    fn calculate_f16_quantization_error(vector: &[f32; 768]) -> f32 {
        vector.iter()
            .map(|&v| {
                let f16_val = half::f16::from_f32(v);
                let reconstructed = f16_val.to_f32();
                (v - reconstructed).abs()
            })
            .sum::<f32>() / 768.0
    }
    
    fn calculate_int8_quantization_error(vector: &[f32; 768]) -> f32 {
        let min = vector.iter().fold(f32::INFINITY, |a, &b| a.min(b));
        let max = vector.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        let scale = (max - min) / 255.0;
        
        vector.iter()
            .map(|&v| {
                let quantized = ((v - min) / scale).round() as u8;
                let reconstructed = quantized as f32 * scale + min;
                (v - reconstructed).abs()
            })
            .sum::<f32>() / 768.0
    }
    
    /// Combine uncertainties into confidence adjustment
    pub fn to_confidence_adjustment(&self) -> f32 {
        // Combine uncertainties (assuming independence)
        let total_uncertainty = 1.0 - 
            (1.0 - self.quantization_error) *
            (1.0 - self.precision_loss) *
            (1.0 - self.compression_uncertainty) *
            (1.0 - self.temporal_uncertainty);
            
        1.0 - total_uncertainty.min(1.0)
    }
}
```

### 4. Confidence-Aware Retrieval

```rust
// engram-core/src/storage/confidence_retrieval.rs

impl TieredStorage {
    /// Retrieve with confidence tracking
    pub fn retrieve_with_confidence(
        &self,
        id: &str,
    ) -> Result<(Vec<f32>, VectorConfidence)> {
        // Determine which tier contains the vector
        let (tier, vector) = self.locate_and_retrieve(id)?;
        
        // Calculate storage uncertainty
        let storage_time = self.get_storage_duration(id);
        let storage_uncertainty = StorageUncertainty::calculate(
            &vector,
            tier,
            storage_time,
        );
        
        // Create confidence with proper propagation
        let mut confidence = VectorConfidence {
            storage_confidence: Confidence::exact(
                storage_uncertainty.to_confidence_adjustment()
            ),
            similarity_confidence: Confidence::HIGH,
            retrieval_confidence: self.get_retrieval_confidence(tier),
            combined: ConfidenceInterval::from_confidence(Confidence::HIGH),
        };
        
        confidence.propagate_retrieval(tier, RetrievalMethod::Direct);
        
        Ok((vector, confidence))
    }
    
    /// Batch retrieval with confidence
    pub fn retrieve_batch_with_confidence(
        &self,
        ids: &[String],
    ) -> Result<Vec<(Vec<f32>, VectorConfidence)>> {
        ids.par_iter()
            .map(|id| self.retrieve_with_confidence(id))
            .collect()
    }
    
    fn get_retrieval_confidence(&self, tier: StorageTier) -> Confidence {
        match tier {
            StorageTier::Hot => Confidence::CERTAIN,
            StorageTier::Warm => Confidence::HIGH,
            StorageTier::Cold => Confidence::MEDIUM,
        }
    }
}
```

## Integration Points

### Modify MemoryStore recall (store.rs)
```rust
// Update around line 250:
pub fn recall(&self, cue: Cue) -> Vec<(Episode, Confidence)> {
    let mut results = Vec::new();
    
    match cue.cue_type {
        CueType::Embedding { vector, threshold } => {
            // Get vectors with confidence
            let candidates = self.vector_storage
                .similarity_search_with_confidence(&vector, 100);
                
            for (id, similarity, vector_confidence) in candidates {
                if let Some(episode) = self.episodes.read().get(&id) {
                    // Combine episode confidence with vector confidence
                    let combined_confidence = episode.encoding_confidence
                        .and(vector_confidence.combined.as_confidence());
                        
                    if combined_confidence.raw() >= threshold {
                        results.push((episode.clone(), combined_confidence));
                    }
                }
            }
        }
        // ... other cue types
    }
    
    results
}
```

## Testing Strategy

### Unit Tests
```rust
#[test]
fn test_confidence_propagation() {
    let mut confidence = VectorConfidence::from_similarity(0.9, SimilarityMethod::Cosine);
    
    confidence.propagate_retrieval(StorageTier::Warm, RetrievalMethod::Index);
    
    // Should reduce confidence for warm tier and index retrieval
    assert!(confidence.combined.point.raw() < 0.9);
    assert!(confidence.combined.width > 0.0); // Has uncertainty
}

#[test]
fn test_confidence_calibration() {
    let mut calibrator = ConfidenceCalibrator::new();
    
    // Train with known outcomes
    let predictions = vec![
        (0.9, true),
        (0.9, true),
        (0.9, false), // Overconfident
        (0.1, false),
        (0.1, false),
        (0.1, true), // Underconfident
    ];
    
    calibrator.update(&predictions);
    
    // Should adjust overconfident predictions down
    let calibrated = calibrator.calibrate(0.9);
    assert!(calibrated.raw() < 0.9);
}
```

### Property Tests
```rust
proptest! {
    #[test]
    fn confidence_monotonic(
        similarity in 0.0f32..1.0,
        tier in 0usize..3,
    ) {
        let tier = match tier {
            0 => StorageTier::Hot,
            1 => StorageTier::Warm,
            _ => StorageTier::Cold,
        };
        
        let conf1 = VectorConfidence::from_similarity(similarity, SimilarityMethod::Cosine);
        let conf2 = VectorConfidence::from_similarity(similarity * 0.9, SimilarityMethod::Cosine);
        
        // Higher similarity should give higher confidence
        prop_assert!(conf1.combined.point >= conf2.combined.point);
    }
}
```

## Acceptance Criteria
- [ ] All vector operations return calibrated confidence
- [ ] Confidence propagation follows probability theory
- [ ] Storage tier affects confidence appropriately
- [ ] Calibration reduces expected calibration error <0.1
- [ ] Uncertainty intervals contain true values 95% of time
- [ ] No confidence values outside [0, 1] range

## Performance Targets
- Confidence calculation: <1μs per operation
- Calibration: <10μs per score
- Uncertainty propagation: <5μs per operation
- Batch confidence: <1ms for 1000 vectors
- Calibration update: <100ms for 10K samples

## Risk Mitigation
- Default to conservative confidence estimates
- Validate all confidence values in [0, 1]
- Regular recalibration with new data
- Monitoring of calibration metrics (ECE, MCE)