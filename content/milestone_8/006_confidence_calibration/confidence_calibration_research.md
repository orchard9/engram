# Confidence Calibration: Research Foundations

## Calibration in Probability Judgments

### Brier Score (1950)
Measures accuracy of probabilistic predictions. Penalizes both under-confidence and over-confidence.

**Formula:** BS = (1/N) Σ (forecast - outcome)²

Perfect calibration: Brier score = 0. Events predicted at 70% occur 70% of the time.

### Calibration Curves (Murphy & Winkler 1977)
Plot predicted probability vs. observed frequency in bins.

Perfect calibration: Diagonal line (predicted = observed).
Over-confident: Above diagonal (70% predictions → 50% accuracy).
Under-confident: Below diagonal (50% predictions → 70% accuracy).

**Engram Application:** Bin completion confidences [0-0.1, 0.1-0.2, ..., 0.9-1.0]. Measure reconstruction accuracy per bin. Calibration error = deviation from diagonal.

## Multi-Factor Confidence Computation

### Combining Multiple Signals
Confidence should aggregate multiple factors, not single score.

**Factors for Pattern Completion:**
1. Convergence speed (CA3 iterations): Faster → higher confidence
2. Energy reduction (attractor depth): Deeper basin → higher confidence
3. Field consensus (neighbor agreement): Higher agreement → higher confidence
4. Plausibility score (semantic coherence): More plausible → higher confidence

**Combination:** Weighted average with empirically-tuned weights.

### Koriat's Cue-Utilization Framework (1997)
People use multiple cues to assess confidence:
- Intrinsic cues: Properties of item itself (difficulty, familiarity)
- Extrinsic cues: Learning conditions (study time, repetition)
- Mnemonic cues: Retrieval fluency (speed, ease)

**Analog for Completion:**
- Intrinsic: Pattern strength, cue overlap
- Extrinsic: Number of source episodes, pattern age
- Mnemonic: Convergence speed, consensus

## Metacognitive Confidence

### Fleming & Dolan (2012): Metacognition in Brain
Metacognitive confidence = confidence in your confidence.

"I remember this, but I'm not sure if my memory is accurate."

**Neural Substrate:** Prefrontal cortex monitors primary memory systems, assesses reliability.

**Engram Implementation:** Compare alternative hypotheses for internal consistency. High consistency → high metacognitive confidence. Low consistency → uncertainty about reconstruction quality.

## Calibration Frameworks

### Platt Scaling (1999)
Post-hoc calibration: Fit logistic regression mapping raw scores to calibrated probabilities.

**Train:** On validation set with ground truth.
**Apply:** Transform completion confidence to calibrated probability.

### Isotonic Regression (Zadrozny & Elkan 2002)
Non-parametric calibration. Learns monotonic mapping from raw to calibrated confidence.

More flexible than Platt scaling. Handles non-linear calibration curves.

**Engram Choice:** Isotonic regression (fewer assumptions, better fits empirical calibration curves).

## Confidence-Accuracy Correlation

### Spearman Rank Correlation
Measure correlation between confidence ranking and accuracy ranking.

Perfect correlation: ρ = 1.0. Items ranked high confidence are actually more accurate.
No correlation: ρ = 0.0. Confidence uninformative.

**Target:** ρ > 0.80 for well-calibrated system.

### Resolution vs. Calibration Trade-off
Resolution: How much confidence scores vary.
Calibration: How well confidence matches accuracy.

High resolution, poor calibration: Over-confident on easy, under-confident on hard.
Low resolution, good calibration: All confidences near average (uninformative).

**Optimal:** High resolution AND good calibration.

## References

1. Brier, G. W. (1950). Verification of forecasts expressed in terms of probability. Monthly Weather Review, 78(1), 1-3.
2. Murphy, A. H., & Winkler, R. L. (1977). Reliability of subjective probability forecasts of precipitation and temperature. Applied Statistics, 41-47.
3. Koriat, A. (1997). Monitoring one's own knowledge during study. Journal of Experimental Psychology: General, 126(4), 349.
4. Fleming, S. M., & Dolan, R. J. (2012). The neural basis of metacognitive ability. Philosophical Transactions of the Royal Society B, 367(1594), 1338-1349.
5. Platt, J. (1999). Probabilistic outputs for support vector machines and comparisons to regularized likelihood methods. Advances in Large Margin Classifiers, 10(3), 61-74.
6. Zadrozny, B., & Elkan, C. (2002). Transforming classifier scores into accurate multiclass probability estimates. KDD.
