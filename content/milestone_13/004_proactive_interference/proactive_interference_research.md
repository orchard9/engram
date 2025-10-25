# Proactive Interference: Research and Technical Foundation

## The Phenomenon

Proactive interference (PI) occurs when old memories interfere with learning or recalling new ones. Underwood (1957) demonstrated this in classic experiments where participants learned multiple lists of words. Learning List A made it harder to recall List B, with interference proportional to similarity and recency of List A.

The effect is remarkably robust. Anderson (1974) found 20-40% reduction in recall accuracy when prior learning was similar to target material. McGeoch (1942) showed interference increases with:
- Similarity between old and new material
- Strength of old memories (more practice = more interference)
- Recency of old learning (recent memories interfere more)
- Number of competing alternatives

## Computational Mechanisms

PI emerges from competition during retrieval. When cued with "fruit," both "apple" from List A and "orange" from List B are activated. If "apple" has higher activation (due to stronger encoding or more practice), it wins the competition and blocks "orange" retrieval.

**Key Components:**

1. **Response Competition:** Multiple candidates compete for retrieval. Winner takes all based on activation strength.

2. **Similarity-Based Activation:** Cues activate similar items proportional to semantic/associative overlap.

3. **Recency Weighting:** Recent memories have residual activation boost, but very recent strong memories create strong competition.

4. **Inhibition Mechanism:** Successfully retrieving target may require inhibiting competitors, creating lasting suppression.

## Implementation Architecture

```rust
pub struct ProactiveInterferenceDetector {
    // Track recent encodings for interference detection
    recent_encodings: VecDeque<(NodeId, Instant, f32)>,  // (node, time, strength)
    similarity_threshold: f32,
    interference_window: Duration,
}

impl ProactiveInterferenceDetector {
    pub fn compute_interference(
        &self,
        target: NodeId,
        cue: NodeId,
        graph: &MemoryGraph,
    ) -> f32 {
        let mut interference = 0.0;

        // Find recent similar encodings
        for (competitor, timestamp, strength) in &self.recent_encodings {
            if *competitor == target {
                continue;  // Not interfering with itself
            }

            // Check similarity to current cue
            let similarity = graph.semantic_similarity(cue, *competitor);

            if similarity > self.similarity_threshold {
                // Compute recency weight
                let elapsed = timestamp.elapsed();
                let recency = if elapsed < self.interference_window {
                    1.0 - (elapsed.as_secs_f32() / self.interference_window.as_secs_f32())
                } else {
                    0.0
                };

                // Interference proportional to competitor strength, similarity, and recency
                interference += strength * similarity * recency;
            }
        }

        interference
    }

    pub fn adjust_activation_for_interference(
        &self,
        target_activation: f32,
        interference: f32,
    ) -> f32 {
        // Anderson (1974): ~20-40% reduction at high interference
        let interference_ratio = interference / (1.0 + interference);
        target_activation * (1.0 - 0.4 * interference_ratio)
    }
}
```

## Validation Criteria

**Target: Anderson (1974) findings**
- High similarity (>0.8): 30-40% recall reduction
- Medium similarity (0.5-0.8): 15-25% reduction
- Low similarity (<0.5): 0-10% reduction
- Effect duration: strongest in first hour, fades over 24 hours

**Statistical Requirements:**
- N >= 800 trials per condition
- Paired comparisons of recall accuracy
- Effect size: Cohen's d = 0.6-0.9
- Significance: p < 0.001

## Performance Budget

- Interference computation: < 100μs (scan recent encodings)
- Recent encoding storage: 48 bytes per entry
- Window size: 1000 recent items (48KB)
- Activation adjustment: < 1μs (simple math)
