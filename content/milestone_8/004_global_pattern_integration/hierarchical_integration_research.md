# Hierarchical Evidence Integration: Research Foundations

## Bayesian Evidence Combination

### Pearl (1988): Probabilistic Reasoning
Bayesian networks combine evidence from multiple sources using conditional probabilities. Independent evidence multiplies likelihoods.

**Combination Rule:**
```
P(H | E1, E2) = P(E1, E2 | H) × P(H) / P(E1, E2)
```

For independent evidence:
```
P(H | E1, E2) ∝ P(E1 | H) × P(E2 | H) × P(H)
```

**Engram Application:** Local context and global patterns are conditionally independent given the true episode. Combine their evidence multiplicatively.

### Jaynes (1957): Maximum Entropy Principle
When multiple probability distributions are consistent with constraints, choose the one with maximum entropy (minimum assumptions).

**Application:** When local and global evidence conflict, weight by strength rather than forcing agreement.

## Hierarchical Integration Models

### Hemmer & Steyvers (2009): Bayesian Model of Memory
Memory reconstructs episodes using hierarchical Bayesian inference:
- Prior: Semantic knowledge (global patterns)
- Likelihood: Episodic evidence (local context)
- Posterior: Reconstructed memory (combination)

**Key Finding:** Reconstructions blend semantic regularities with episodic details. Strength of blending depends on evidence quality.

**Implications:** Engram's adaptive weighting implements this: weak episodic evidence → rely on semantic priors.

### Griffiths et al. (2008): Hierarchical Bayesian Models
Cognitive processes implement hierarchical inference:
- Lower level: Specific instances (episodes)
- Higher level: Abstract patterns (semantic knowledge)
- Integration: Bidirectional influence

**Application:** Local reconstruction (low level) and global patterns (high level) influence each other through evidence aggregation.

## Agreement Boosting and Disagreement Handling

### Condorcet's Jury Theorem (1785)
If multiple independent sources have >50% accuracy, their majority vote is more accurate than any individual.

**Modern Application:** When local and global sources agree, boost confidence. Disagreement signals uncertainty.

**Engram Implementation:**
- Agreement: confidence_combined = confidence_local × confidence_global / P(agreement | random)
- Disagreement: confidence = max(confidence_local, confidence_global) with penalty

### Ensemble Methods (Breiman 1996)
Combining predictions from multiple models improves accuracy if models are diverse and better than random.

**Key Requirements:**
- Diversity: Models make different errors
- Accuracy: Each model better than chance

Local (temporal) and global (semantic) are diverse by construction. Both exceed random baseline.

## Adaptive Weighting Strategies

### Varying Uncertainty (Kahneman & Tversky 1974)
Humans adaptively weight evidence based on perceived reliability. High-certainty evidence weighted more than low-certainty evidence.

**Adaptive Formula:**
```
weight_local = confidence_local / (confidence_local + confidence_global)
weight_global = confidence_global / (confidence_local + confidence_global)
```

Normalizes to sum=1.0 while respecting relative strengths.

### Temporal vs Statistical Evidence (Shiffrin & Steyvers 1997)
Temporal proximity strong cue for related episodes. Statistical patterns strong for typical features.

**Heuristic:** Recent specific instance > distant pattern, but strong pattern > weak instance.

## References

1. Pearl, J. (1988). Probabilistic reasoning in intelligent systems. Morgan Kaufmann.
2. Jaynes, E. T. (1957). Information theory and statistical mechanics. Physical Review, 106(4), 620.
3. Hemmer, P., & Steyvers, M. (2009). A Bayesian account of reconstructive memory. Topics in Cognitive Science, 1(1), 189-202.
4. Griffiths, T. L., Kemp, C., & Tenenbaum, J. B. (2008). Bayesian models of cognition. Cambridge Handbook of Computational Psychology.
5. Breiman, L. (1996). Bagging predictors. Machine Learning, 24(2), 123-140.
6. Kahneman, D., & Tversky, A. (1974). Judgment under uncertainty: Heuristics and biases. Science, 185(4157), 1124-1131.
