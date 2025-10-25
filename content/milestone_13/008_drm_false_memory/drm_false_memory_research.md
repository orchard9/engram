# DRM False Memory Paradigm: Research and Technical Foundation

## The Groundbreaking Experiment

Roediger & McDermott (1995) revived Deese's (1959) paradigm, creating one of the most reliable false memory phenomena in psychology. Participants study word lists like:

**List: "sleep"**
bed, rest, awake, tired, dream, wake, snooze, blanket, doze, slumber, snore, nap, peace, yawn, drowsy

Note: "sleep" is NOT presented. Yet 55-65% of participants falsely recall or recognize it as having been studied. The effect is remarkably robust across hundreds of replications.

## Theoretical Mechanisms

**Spreading Activation Theory (Collins & Loftus, 1975):**
Each studied word ("bed", "rest", "awake") activates the critical lure ("sleep") through semantic associations. Cumulative activation from 15 related words makes "sleep" feel familiar, creating false memory.

**Fuzzy Trace Theory (Brainerd & Reyna, 2002):**
People encode both verbatim traces (exact words) and gist traces (general theme). The gist trace captures "sleep-related words," which matches the critical lure perfectly, creating false recognition based on meaning rather than perceptual memory.

Both theories predict the same outcome through different mechanisms. Our implementation uses spreading activation, which maps directly to our graph architecture.

## Quantitative Characteristics

Roediger & McDermott (1995) found:
- False recall: 55-65% for critical lures
- Veridical recall: 60-70% for studied words
- False recognition: 75-85% (higher than recall)
- Confidence: false memories often rated as confident as true memories

Key modulating factors:
- List length: 15 words optimal, shorter lists reduce effect
- Semantic strength: high BAS (backward associative strength) essential
- Study time: 1-2 seconds per word optimal
- Retention interval: effect persists for at least 24 hours

## Implementation Requirements

To replicate the DRM paradigm, we need:

**1. List Generation:**
```rust
pub fn generate_drm_list(critical_lure: NodeId, graph: &MemoryGraph) -> Vec<NodeId> {
    // Find 15 words with highest backward associative strength to critical lure
    let mut candidates: Vec<(NodeId, f32)> = graph.nodes()
        .map(|node| {
            let bas = graph.associative_strength(node, critical_lure);
            (node, bas)
        })
        .filter(|(_, bas)| *bas > 0.3)  // Threshold for inclusion
        .collect();

    candidates.sort_by_key(|(_, bas)| OrderedFloat(-bas));
    candidates.into_iter()
        .take(15)
        .map(|(node, _)| node)
        .collect()
}
```

**2. Study Phase:**
```rust
pub fn study_drm_list(
    &mut self,
    word_list: Vec<NodeId>,
    study_time_ms: u64,
) {
    for word in word_list {
        self.encode(word, EncodeParams {
            duration: Duration::from_millis(study_time_ms),
            attention: 1.0,
        });

        // Inter-stimulus interval
        thread::sleep(Duration::from_millis(500));
    }
}
```

**3. False Memory Detection:**
```rust
pub fn detect_false_memory(
    &self,
    retrieved: &[NodeId],
    studied: &[NodeId],
    critical_lures: &[NodeId],
) -> FalseMemoryMetrics {
    let mut false_recalls = 0;
    let mut true_recalls = 0;

    for node in retrieved {
        if critical_lures.contains(node) {
            false_recalls += 1;
        } else if studied.contains(node) {
            true_recalls += 1;
        }
    }

    FalseMemoryMetrics {
        false_recall_rate: false_recalls as f32 / critical_lures.len() as f32,
        true_recall_rate: true_recalls as f32 / studied.len() as f32,
        false_recall_count: false_recalls,
        true_recall_count: true_recalls,
    }
}
```

## Validation Criteria

**Primary Criterion (Must Match):**
- False recall rate: 55-65% ± 10%
- If system produces 45% or 75%, it's out of bounds
- Statistical significance: p < 0.001 vs chance (0%)

**Secondary Criteria (Should Match):**
- True recall rate: 60-70%
- False recognition: 75-85%
- Confidence ratings: false ~= true

**Statistical Requirements:**
- N >= 1000 trials (study-test cycles)
- Multiple critical lures (>=20 different lists)
- Counterbalancing of list presentation order
- Power = 0.95 for detecting d = 0.8 effect

## Critical Implementation Details

**Detail 1: Backward Associative Strength**

BAS from list word to critical lure must be high. "Bed" strongly associates to "sleep," not vice versa. This is captured in directed edge weights in our graph.

**Detail 2: Cumulative Activation**

Each list word contributes activation to the critical lure. With 15 words, cumulative activation can exceed studied word activation, creating false memory. Our spreading activation must accumulate rather than replace.

**Detail 3: Retrieval Cues**

Roediger & McDermott used free recall (no cues). We must test retrieval without providing list words as cues, relying on context-based spreading activation.

**Detail 4: Time Course**

Encoding must allow sufficient spreading time (1-2 sec per word). Retrieval must happen after cumulative activation has decayed somewhat (immediate test vs delayed test comparison).

## Performance Implications

DRM validation requires:
- Generate 20+ lists: one-time cost, < 1 second total
- Study phase: 15 words × 2 sec = 30 sec per trial
- Retrieval: standard spreading activation, < 100ms
- 1000 trials: ~8-10 hours of simulated time
- Can be parallelized and sped up by reducing inter-stimulus intervals in simulation

## Integration with Existing Systems

DRM validation tests the entire cognitive stack:
- M3 (Spreading Activation): cumulative activation computation
- M4 (Temporal Dynamics): decay over retention interval
- M6 (Consolidation): STM to LTM transfer of false memories
- M8 (Pattern Completion): retrieving critical lure from degraded cues
- M13 Tasks 002-003 (Priming): semantic priming contributes to false activation

This makes DRM validation a comprehensive acceptance test for cognitive plausibility.
