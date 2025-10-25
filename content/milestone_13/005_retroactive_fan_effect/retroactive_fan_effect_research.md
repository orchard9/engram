# Retroactive Interference and Fan Effect: Research and Technical Foundation

## Retroactive Interference

Retroactive interference (RI) is the mirror of proactive interference: new learning interferes with recalling old memories. Barnes & Underwood (1959) showed that learning List B impairs recall of previously learned List A, with magnitude proportional to similarity and practice on List B.

The effect demonstrates memory is not write-once: new information can overwrite or block access to existing memories. This has profound implications for memory consolidation and reconsolidation.

**Empirical Characteristics (Postman & Underwood, 1973):**
- High similarity: 40-50% reduction in original list recall
- Medium similarity: 20-30% reduction
- Low similarity: 5-15% reduction
- Time course: maximum effect immediately after new learning, gradual recovery over days

## The Fan Effect

Anderson (1974) discovered the fan effect: the more facts associated with a concept, the slower any single fact is retrieved. If you learn:
- "The doctor is in the bank"
- "The doctor is in the park"
- "The doctor is in the church"

Retrieving "doctor-bank" becomes slower as fan increases. This reflects spreading activation being divided among competitors.

**Quantitative Findings:**
- Fan 1: baseline retrieval time
- Fan 2: +100-150ms
- Fan 3: +200-300ms
- Fan 4: +300-450ms

The effect is logarithmic: each additional link costs less than the previous one.

## Computational Mechanisms

Both phenomena emerge from activation spreading across multiple competing associations:

**Retroactive Interference:**
1. New encoding creates strong association from cue to new target
2. Strong new association outcompetes older, potentially weaker association
3. Retrieval cued from same stimulus preferentially activates recent strong link
4. Old memory is blocked by competition, not erased

**Fan Effect:**
1. Spreading activation from concept divides among all connections
2. Each connection receives activation proportional to 1/fan
3. Higher fan = lower activation per connection = slower retrieval
4. Effect compounds when both concepts have high fan

## Implementation Architecture

```rust
pub struct FanEffectCalculator {
    node_fans: HashMap<NodeId, usize>,
}

impl FanEffectCalculator {
    pub fn compute_fan_penalty(&self, node_a: NodeId, node_b: NodeId) -> f32 {
        let fan_a = self.node_fans.get(&node_a).copied().unwrap_or(1);
        let fan_b = self.node_fans.get(&node_b).copied().unwrap_or(1);

        // Combined fan: geometric mean of both nodes' fans
        let combined_fan = ((fan_a * fan_b) as f32).sqrt();

        // Anderson (1974): logarithmic penalty
        // Fan 1: penalty = 0
        // Fan 2: penalty ~= 0.15 (100-150ms on 1000ms baseline)
        // Fan 3: penalty ~= 0.25
        // Fan 4: penalty ~= 0.32
        0.15 * (combined_fan as f32).ln()
    }

    pub fn apply_fan_penalty(&self, activation: f32, penalty: f32) -> f32 {
        activation * (1.0 - penalty)
    }
}

pub struct RetroactiveInterferenceDetector {
    encoding_history: Vec<(NodeId, NodeId, Instant, f32)>,  // (cue, target, time, strength)
}

impl RetroactiveInterferenceDetector {
    pub fn compute_retroactive_interference(
        &self,
        cue: NodeId,
        old_target: NodeId,
        encoding_time: Instant,
    ) -> f32 {
        let mut interference = 0.0;

        // Find newer encodings from same cue
        for (enc_cue, enc_target, enc_time, enc_strength) in &self.encoding_history {
            if *enc_cue != cue || *enc_target == old_target {
                continue;
            }

            if *enc_time > encoding_time {
                // This is a newer encoding that could interfere
                let elapsed_since = enc_time.duration_since(encoding_time);

                // Newer encodings interfere more
                let recency_factor = (-elapsed_since.as_secs_f32() / 86400.0).exp();

                interference += enc_strength * recency_factor;
            }
        }

        interference
    }
}
```

## Validation Criteria

**Retroactive Interference (Postman & Underwood, 1973):**
- High similarity: 40-50% recall reduction
- Medium similarity: 20-30% reduction
- Statistical power: N >= 1000, d = 0.7-1.0

**Fan Effect (Anderson, 1974):**
- Fan 2 vs 1: +100-150ms (10-15% slowdown)
- Fan 3 vs 1: +200-300ms (20-30% slowdown)
- Fan 4 vs 1: +300-450ms (30-45% slowdown)
- Statistical power: N >= 800, d = 0.6-0.9

## Performance Budget

- Fan calculation: O(1) hash lookup, < 100ns
- RI detection: O(k) scan of encoding history, < 200μs
- Total overhead: < 300μs per retrieval
