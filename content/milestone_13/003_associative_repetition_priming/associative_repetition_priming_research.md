# Associative and Repetition Priming: Research and Technical Foundation

## The Phenomena

While semantic priming reflects meaning-based relationships, associative and repetition priming reveal different memory mechanisms. Tulving & Schacter (1990) distinguished between these priming types, showing they operate independently and engage different neural substrates.

**Repetition Priming:** Encountering a stimulus makes it easier to process that exact stimulus again. Jacoby & Dallas (1981) found that previously seen words are identified 20-40ms faster in perceptual identification tasks, even days later. This effect is perceptual and specific - seeing "TABLE" in uppercase doesn't fully prime "table" in lowercase.

**Associative Priming:** Word pairs that frequently co-occur prime each other regardless of semantic relationship. McKoon & Ratcliff (1992) showed that "thunder-lightning" produces 40-60ms facilitation not because of semantic similarity (they're different phenomena) but because of learned associations from co-occurrence.

## Temporal and Behavioral Characteristics

The temporal dynamics differ markedly from semantic priming:

**Repetition Priming Duration:**
Tulving et al. (1982) found repetition priming lasting hours to days, far longer than semantic priming's seconds. The effect shows:
- Immediate facilitation: 30-50ms at first re-exposure
- Persistent effects: 15-25ms facilitation after 24 hours
- Gradual decay: logarithmic rather than exponential
- Specificity: modality and format-dependent

**Associative Priming Time Course:**
McKoon & Ratcliff (1992) demonstrated:
- Fast onset: effects visible at 100ms SOA
- Peak magnitude: 40-60ms at 250-400ms SOA
- Longer persistence: measurable effects at 1500ms SOA
- Independence from semantic relatedness

## Computational Requirements

Implementing these priming types requires distinct mechanisms:

**For Repetition Priming:**

1. **Instance Storage:** Each presentation creates a memory trace with perceptual details. Subsequent presentations match against these traces, producing facilitation proportional to match quality.

2. **Trace Strength:** Repeated exposures strengthen traces multiplicatively, not additively. Second presentation creates larger boost than third, following power law.

3. **Specificity Encoding:** Store modality (visual/auditory), format (uppercase/lowercase), context. Partial matches produce partial priming.

4. **Decay Model:** Logarithmic decay `strength = initial * (1 - log(t)/log(max_t))` matches empirical data better than exponential for long-duration effects.

**For Associative Priming:**

1. **Co-occurrence Statistics:** Track word pair frequencies from corpus data or user interactions. Use pointwise mutual information (PMI) to identify strong associations.

2. **Bidirectional Associations:** "Thunder" primes "lightning" and vice versa, with potentially asymmetric strengths based on conditional probabilities.

3. **Temporal Contiguity:** Weight recent co-occurrences more heavily than distant ones, matching how human associations form from recent experience.

4. **Independence from Semantics:** Associative links should activate even for semantically unrelated pairs (e.g., "salt-pepper", "bread-butter").

## Implementation Architecture

**Repetition Priming System:**

```rust
pub struct RepetitionTrace {
    node_id: NodeId,
    strength: f32,
    modality: Modality,
    format: Format,
    context: Vec<NodeId>,
    timestamp: Instant,
}

pub struct RepetitionPrimingEngine {
    traces: Vec<RepetitionTrace>,
    strength_boost: f32,
    decay_log_base: f32,
}

impl RepetitionPrimingEngine {
    pub fn record_presentation(&mut self, node: NodeId, details: PresentationDetails) {
        // Find matching traces
        let matching_traces = self.traces.iter_mut()
            .filter(|t| t.node_id == node && self.matches(t, &details));

        if let Some(trace) = matching_traces.next() {
            // Strengthen existing trace (power law)
            trace.strength *= self.strength_boost;
        } else {
            // Create new trace
            self.traces.push(RepetitionTrace {
                node_id: node,
                strength: 1.0,
                modality: details.modality,
                format: details.format,
                context: details.context,
                timestamp: Instant::now(),
            });
        }
    }

    pub fn compute_boost(&self, node: NodeId, query_details: &PresentationDetails) -> f32 {
        self.traces.iter()
            .filter(|t| t.node_id == node)
            .map(|trace| {
                let match_quality = self.compute_match(trace, query_details);
                let elapsed_secs = trace.timestamp.elapsed().as_secs_f32();
                let decay = self.logarithmic_decay(elapsed_secs);
                trace.strength * match_quality * decay
            })
            .sum()
    }

    fn logarithmic_decay(&self, elapsed_secs: f32) -> f32 {
        // Decay over 24 hours: 1.0 at t=0, 0.5 at t=12h, 0.2 at t=24h
        let max_time = 86400.0;  // 24 hours
        1.0 - (elapsed_secs.ln() / max_time.ln()).max(0.0)
    }
}
```

**Associative Priming System:**

```rust
pub struct AssociativePrimingEngine {
    cooccurrence_matrix: HashMap<(NodeId, NodeId), f32>,
    temporal_weights: VecDeque<f32>,
}

impl AssociativePrimingEngine {
    pub fn learn_association(&mut self, node_a: NodeId, node_b: NodeId, weight: f32) {
        // Bidirectional with temporal weighting
        let temporal_factor = self.temporal_weights.back().unwrap_or(&1.0);

        *self.cooccurrence_matrix.entry((node_a, node_b)).or_insert(0.0) +=
            weight * temporal_factor;
        *self.cooccurrence_matrix.entry((node_b, node_a)).or_insert(0.0) +=
            weight * temporal_factor;
    }

    pub fn compute_boost(&self, prime: NodeId, target: NodeId) -> f32 {
        self.cooccurrence_matrix
            .get(&(prime, target))
            .copied()
            .unwrap_or(0.0)
    }

    pub fn update_temporal_weights(&mut self) {
        // Decay old weights, add new weight for current time window
        self.temporal_weights.iter_mut().for_each(|w| *w *= 0.95);
        self.temporal_weights.push_back(1.0);

        // Keep only recent history (last 100 time windows)
        if self.temporal_weights.len() > 100 {
            self.temporal_weights.pop_front();
        }
    }
}
```

## Validation Against Empirical Data

**Repetition Priming Validation:**

Target: Jacoby & Dallas (1981) findings
- Immediate re-presentation: 30-50ms facilitation
- 1-hour delay: 25-35ms facilitation
- 24-hour delay: 15-25ms facilitation
- Cross-format penalty: 30-40% reduction

**Associative Priming Validation:**

Target: McKoon & Ratcliff (1992) findings
- Strong pairs (PMI > 5): 40-60ms facilitation
- Moderate pairs (PMI 2-5): 20-40ms facilitation
- Weak pairs (PMI < 2): 0-15ms facilitation
- Peak effect: 250-400ms SOA

## Integration Strategy

Both priming types integrate with the existing priming boost mechanism:

```rust
pub fn compute_total_boost(&self, target: NodeId, context: &RetrievalContext) -> f32 {
    let semantic_boost = self.semantic_priming.compute_boost(target);
    let repetition_boost = self.repetition_priming.compute_boost(target, &context.details);
    let associative_boost = self.associative_priming.compute_boost(
        context.recent_prime,
        target
    );

    // Boosts combine additively (independent mechanisms)
    semantic_boost + repetition_boost + associative_boost
}
```

## Performance Budgets

**Repetition Priming:**
- Trace storage: 48 bytes per trace
- Max active traces: 10,000 (480KB)
- Lookup time: O(log n) with index
- Boost computation: < 2μs

**Associative Priming:**
- Co-occurrence matrix: sparse representation
- Entries: ~1M pairs at 12 bytes each (12MB)
- Lookup time: O(1) hash table
- Boost computation: < 100ns

Combined overhead for both systems: less than 3μs per retrieval, maintaining our performance budget.

## Statistical Power Requirements

Repetition priming effects are larger (d = 0.8-1.2) than semantic priming, requiring smaller samples for validation. Associative priming has medium effects (d = 0.6-0.8). Our validation strategy:

- Repetition: N = 500 trials per condition
- Associative: N = 800 trials per condition
- Both: paired t-tests, power = 0.80, alpha = 0.05

This ensures robust detection of effects matching published magnitudes.
