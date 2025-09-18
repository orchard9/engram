# Confidence Calibration for Storage Operations Research

## Research Topics for Milestone 2 Task 007: Uncertainty Quantification in Tiered Storage

### 1. Confidence Calibration in Machine Learning
- Platt scaling and isotonic regression for probability calibration
- Temperature scaling in neural network outputs
- Expected Calibration Error (ECE) and reliability diagrams
- Confidence intervals vs prediction intervals
- Bayesian approaches to uncertainty quantification

### 2. Uncertainty in Information Retrieval Systems
- Probabilistic ranking models (BM25, language models)
- Query-document similarity confidence estimation
- Relevance feedback and confidence adjustment
- Uncertainty propagation in retrieval pipelines
- Meta-search confidence aggregation strategies

### 3. Cognitive Confidence and Metacognition
- Feeling of knowing (FOK) and tip-of-tongue phenomena
- Confidence-accuracy correlation in human memory
- Metacognitive monitoring and control processes
- Signal detection theory in recognition memory
- Confidence as a continuous decision variable

### 4. Storage System Reliability and Error Models
- Bit error rates in different storage media
- Silent data corruption probabilities
- RAID reliability calculations
- Mean time to data loss (MTTDL) models
- Storage tier-specific failure characteristics

### 5. Temporal Degradation Models
- Data decay in magnetic storage
- Flash memory retention characteristics
- Cloud storage durability guarantees
- Archival storage longevity studies
- Information-theoretic entropy increase over time

### 6. Compression and Quantization Impact on Confidence
- Lossy compression error bounds
- Product quantization reconstruction error
- Rate-distortion theory applications
- Confidence preservation through encoding
- Perceptual quality metrics for compressed data

## Research Findings

### Confidence Calibration in Machine Learning

**Calibration Definition:**
A model is well-calibrated if P(Y=y|f(x)=p) = p, meaning when the model predicts probability p, the true probability of correctness is also p.

**Modern Calibration Techniques:**

1. **Temperature Scaling**: Post-processing technique that adjusts logits by temperature T:
```
calibrated_confidence = softmax(logits / T)
```
Optimal T found via validation set optimization. Typically T ∈ [0.5, 5.0].

2. **Platt Scaling**: Fits sigmoid to uncalibrated scores:
```
calibrated = 1 / (1 + exp(A * uncalibrated + B))
```
Parameters A, B learned from held-out data.

3. **Isotonic Regression**: Non-parametric method that enforces monotonicity while minimizing squared error. Produces piecewise constant calibration function.

**Calibration Metrics:**

- **Expected Calibration Error (ECE)**: Average difference between confidence and accuracy across bins
```
ECE = Σ(|Bm|/n) * |acc(Bm) - conf(Bm)|
```

- **Maximum Calibration Error (MCE)**: Worst-case calibration error
- **Brier Score**: Mean squared difference between predicted probability and actual outcome
- **Negative Log-Likelihood**: Penalizes confident wrong predictions heavily

Research shows neural networks tend to be overconfident (ECE ≈ 0.15-0.20), requiring calibration.

### Uncertainty in Information Retrieval Systems

**Sources of Uncertainty in IR:**

1. **Query Uncertainty**: Ambiguous or underspecified queries
2. **Document Uncertainty**: Relevance estimation errors
3. **Collection Uncertainty**: Incomplete or biased corpus
4. **Model Uncertainty**: Algorithmic approximations

**Probabilistic Ranking Principles:**
The Probability Ranking Principle (Robertson, 1977) states documents should be ranked by P(relevant|document, query).

**BM25 Confidence Estimation:**
```
confidence = 1 / (1 + exp(-score/σ))
```
Where σ is learned from relevance judgments. Typical values: σ ∈ [1.0, 3.0].

**Language Model Confidence:**
Query likelihood models provide natural probability interpretations:
```
P(Q|D) = Π P(qi|D)
confidence = exp(log P(Q|D) / |Q|)
```

**Fusion Confidence Aggregation:**
When combining multiple retrieval systems:
- **CombSUM**: Simple addition (assumes independence)
- **CombMNZ**: Multiply by number of non-zero scores
- **Weighted**: Learn optimal weights from validation
- **Probabilistic**: Model joint probability distribution

### Cognitive Confidence and Metacognition

**Human Confidence Characteristics:**

1. **Hard-Easy Effect**: Overconfidence on hard tasks, underconfidence on easy tasks
2. **Dunning-Kruger Effect**: Low-ability individuals overestimate confidence
3. **Hindsight Bias**: Confidence increases after knowing the answer
4. **Confidence-Accuracy Correlation**: Typically r = 0.4-0.6 in recognition tasks

**Signal Detection Theory Framework:**
- **d'** (sensitivity): Ability to discriminate signal from noise
- **β** (bias): Decision criterion placement
- **Confidence**: Distance from decision boundary

Type 1 Signal Detection:
```
confidence = |evidence - criterion|
```

Type 2 Signal Detection (metacognitive):
```
metacognitive_confidence = P(correct | evidence, decision)
```

**Feeling of Knowing (FOK) Research:**
FOK accuracy correlates with:
- Partial information activation (r = 0.35)
- Cue familiarity (r = 0.42)
- Previous retrieval success (r = 0.51)

**Metacognitive Control:**
Confidence drives strategic behavior:
- High confidence → Fast response, less checking
- Low confidence → Slower response, more verification
- Uncertainty → Information seeking behavior

### Storage System Reliability Models

**Storage Media Error Characteristics:**

1. **HDD (Hot Tier Candidate)**:
- Bit error rate: 10^-14 to 10^-15
- Annual failure rate: 2-4%
- Seek errors increase with age
- No data degradation when powered

2. **SSD (Warm Tier)**:
- Bit error rate: 10^-16 to 10^-17
- Annual failure rate: 0.5-1.5%
- Write endurance limits (3K-10K cycles)
- Retention decreases with wear

3. **Cloud/Tape (Cold Tier)**:
- Cloud durability: 99.999999999% (11 nines)
- Tape bit error rate: 10^-17 to 10^-19
- 30-year archival life
- Requires periodic refresh

**MTTDL Calculations:**
Mean Time To Data Loss for replicated storage:
```
MTTDL = MTBF^n / (n * MTTR^(n-1))
```
Where n = replication factor, MTBF = mean time between failures, MTTR = mean time to repair

**Silent Data Corruption:**
Undetected bit flips occur at rates:
- Memory: 10^-10 per bit per hour
- Network: 10^-10 per bit transferred
- Storage: 10^-14 per bit per year

Mitigation requires checksums with confidence adjustment based on verification.

### Temporal Degradation Models

**Magnetic Media Degradation:**
Magnetization decay follows Arrhenius equation:
```
τ = τ0 * exp(KuV/kBT)
```
Where Ku = anisotropy constant, V = grain volume, T = temperature

**Flash Memory Retention:**
Charge loss from floating gate:
```
Vth(t) = Vth(0) - α * log(1 + t/τ)
```
Retention time strongly temperature-dependent: halves every 10°C increase.

**Information Entropy Increase:**
Shannon entropy of stored data increases over time:
```
H(t) = H(0) + k * log(t)
```
Confidence should decrease proportionally: confidence(t) = confidence(0) * exp(-k * log(t))

**Archival Storage Best Practices:**
- Refresh cycles: Every 5-10 years for tape
- Environmental control: 15-23°C, 40-50% humidity
- Format migration: Before obsolescence
- Integrity verification: Annual checksums

### Compression Impact on Confidence

**Lossy Compression Error Bounds:**

1. **Product Quantization Error:**
Mean squared error for k-means quantization:
```
MSE = σ² * (1 - 1/k)^(2/d)
```
Where σ² = variance, k = codebook size, d = dimensions

2. **Vector Quantization Confidence:**
Reconstruction confidence based on distance to centroid:
```
confidence = exp(-||x - centroid||² / 2σ²)
```

3. **Rate-Distortion Optimal Confidence:**
Shannon's rate-distortion function gives minimum achievable distortion:
```
D(R) = σ² * 2^(-2R)
```
Confidence should reflect this theoretical limit.

**Perceptual Quality Metrics:**
- **PSNR**: Peak signal-to-noise ratio (dB)
- **SSIM**: Structural similarity index
- **Perceptual Loss**: Deep feature matching

For embeddings, cosine similarity preservation:
```
confidence = cosine_sim(original, compressed)^2
```

## Implementation Strategy for Engram

### 1. Tier-Specific Confidence Factors

**Base Confidence Model:**
```rust
pub struct TierConfidence {
    // Storage reliability factors
    hot_reliability: f32,   // 0.999 - RAM with ECC
    warm_reliability: f32,  // 0.995 - SSD with wear
    cold_reliability: f32,  // 0.990 - Compressed/quantized

    // Compression impact factors
    warm_compression: f32,  // 0.98 - Lossless or light
    cold_compression: f32,  // 0.92 - Product quantization

    // Access latency confidence
    hot_latency_factor: f32,   // 1.00 - Immediate
    warm_latency_factor: f32,  // 0.98 - May need loading
    cold_latency_factor: f32,  // 0.95 - Reconstruction needed
}
```

### 2. Temporal Decay Model

**Biologically-Inspired Decay:**
```rust
pub fn temporal_confidence_decay(
    initial_confidence: f32,
    time_elapsed: Duration,
    memory_strength: f32,
) -> f32 {
    // Ebbinghaus forgetting curve with strength modulation
    let days = time_elapsed.as_secs() as f32 / 86400.0;
    let retention = (-(days / memory_strength)).exp();

    // Minimum confidence floor (never completely forgotten)
    let min_confidence = 0.1;

    initial_confidence * retention.max(min_confidence)
}
```

### 3. Calibration Framework

**Multi-Level Calibration:**
```rust
pub struct ConfidenceCalibrator {
    // Tier-specific calibrators
    tier_calibrators: HashMap<StorageTier, TierCalibrator>,

    // Global calibration
    temperature: f32,  // For temperature scaling
    platt_params: (f32, f32),  // For Platt scaling

    // Metacognitive adjustment
    metacognitive_model: MetacognitiveConfidence,
}

impl ConfidenceCalibrator {
    pub fn calibrate(
        &self,
        raw_confidence: f32,
        tier: StorageTier,
        context: &RetrievalContext,
    ) -> f32 {
        // 1. Tier-specific adjustment
        let tier_adjusted = self.tier_calibrators[&tier]
            .adjust(raw_confidence, context);

        // 2. Temperature scaling
        let temp_scaled = self.temperature_scale(tier_adjusted);

        // 3. Platt scaling for final calibration
        let platt_scaled = self.platt_scale(temp_scaled);

        // 4. Metacognitive bounds
        self.metacognitive_model.bound(platt_scaled, context)
    }

    fn temperature_scale(&self, confidence: f32) -> f32 {
        // Apply learned temperature
        let logit = (confidence / (1.0 - confidence)).ln();
        let scaled_logit = logit / self.temperature;
        1.0 / (1.0 + (-scaled_logit).exp())
    }

    fn platt_scale(&self, confidence: f32) -> f32 {
        let (a, b) = self.platt_params;
        1.0 / (1.0 + (a * confidence + b).exp())
    }
}
```

### 4. Compression-Aware Confidence

**Product Quantization Impact:**
```rust
pub struct CompressionConfidence {
    codebook_sizes: HashMap<StorageTier, usize>,
    reconstruction_errors: HashMap<StorageTier, f32>,
}

impl CompressionConfidence {
    pub fn quantization_confidence(
        &self,
        original_norm: f32,
        reconstructed_norm: f32,
        codebook_size: usize,
    ) -> f32 {
        // Theoretical reconstruction quality
        let theoretical_quality = 1.0 - (1.0 / codebook_size as f32).powf(2.0/768.0);

        // Actual reconstruction quality
        let actual_quality = (reconstructed_norm / original_norm).min(1.0);

        // Combined confidence
        theoretical_quality * actual_quality
    }
}
```

### 5. Validation and Monitoring

**Calibration Validation:**
```rust
pub struct CalibrationValidator {
    bins: Vec<CalibrationBin>,
    metrics: CalibrationMetrics,
}

struct CalibrationBin {
    confidence_range: (f32, f32),
    predictions: Vec<f32>,
    outcomes: Vec<bool>,
}

impl CalibrationValidator {
    pub fn expected_calibration_error(&self) -> f32 {
        let mut total_error = 0.0;
        let total_samples = self.total_samples();

        for bin in &self.bins {
            let bin_accuracy = bin.accuracy();
            let bin_confidence = bin.mean_confidence();
            let bin_weight = bin.size() as f32 / total_samples as f32;

            total_error += bin_weight * (bin_accuracy - bin_confidence).abs();
        }

        total_error
    }

    pub fn reliability_diagram(&self) -> Vec<(f32, f32)> {
        self.bins.iter()
            .map(|bin| (bin.mean_confidence(), bin.accuracy()))
            .collect()
    }
}
```

### 6. Runtime Adaptation

**Online Calibration Updates:**
```rust
pub struct AdaptiveCalibrator {
    alpha: f32,  // Learning rate
    history: CircularBuffer<CalibrationSample>,
}

impl AdaptiveCalibrator {
    pub fn update(&mut self, predicted: f32, actual: bool) {
        let sample = CalibrationSample {
            confidence: predicted,
            correct: actual,
            timestamp: SystemTime::now(),
        };

        self.history.push(sample);

        // Update calibration parameters
        if self.history.len() >= MIN_SAMPLES {
            self.recalibrate();
        }
    }

    fn recalibrate(&mut self) {
        // Gradient descent on calibration error
        let grad = self.calibration_gradient();
        self.temperature *= 1.0 - self.alpha * grad.temperature;
        self.platt_params.0 -= self.alpha * grad.platt_a;
        self.platt_params.1 -= self.alpha * grad.platt_b;
    }
}
```

## Key Implementation Insights

1. **Storage tier directly impacts confidence** - Hot tier maintains highest confidence, cold tier requires adjustment for compression
2. **Temporal decay should be gradual** - Use long half-lives (years) to maintain useful confidence over time
3. **Calibration must be validated** - Use ECE and reliability diagrams to ensure well-calibrated probabilities
4. **Compression introduces quantifiable uncertainty** - Product quantization error can be computed theoretically
5. **Human metacognition provides useful bounds** - FOK research suggests confidence ranges for different retrieval scenarios
6. **Online adaptation improves calibration** - Learn from prediction-outcome pairs to refine calibration
7. **Different tiers need different strategies** - Hot tier focuses on reliability, cold tier on reconstruction quality
8. **Confidence should drive behavior** - Low confidence triggers verification or alternative retrieval paths
9. **Minimum confidence floor prevents zero values** - Always maintain some possibility of correctness
10. **Calibration overhead must be minimal** - Sub-microsecond adjustment to avoid impacting retrieval performance

This research provides a comprehensive framework for implementing storage-aware confidence calibration that accounts for tier characteristics, temporal degradation, and compression impacts while maintaining computational efficiency.