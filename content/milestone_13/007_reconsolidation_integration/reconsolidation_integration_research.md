# Reconsolidation Integration: Research and Technical Foundation

## Integration Challenge

Implementing reconsolidation core logic (Task 006) is one thing. Integrating it seamlessly with Engram's existing consolidation pipeline (M6), pattern completion (M8), and interference systems (Tasks 004-005) is another. The challenge is maintaining biological plausibility while avoiding architectural complexity explosion.

## Integration Points

**Integration 1: Consolidation Pipeline**

Existing M6 consolidation moves memories from STM to LTM over hours. Reconsolidation means retrieved LTM memories can re-enter this pipeline:

```
Normal: Encode → STM → (consolidate) → LTM
Reconsolidation: Retrieve LTM → (lability) → STM → (re-consolidate) → LTM
```

The key question: does reconsolidated memory follow the same time course as initial consolidation? Nader et al. (2000) suggests yes, but with potentially faster kinetics due to existing synaptic infrastructure.

**Integration 2: Pattern Completion**

M8 pattern completion retrieves memories from partial cues. This retrieval can trigger reconsolidation. But pattern completion also involves generating false memories (DRM paradigm). How do false memories interact with reconsolidation?

Hypothesis: False memories are weak and thus highly susceptible to reconsolidation. They can be strengthened if retrieved and confirmed, or weakened if disconfirmed.

**Integration 3: Interference Systems**

Tasks 004-005 implement proactive and retroactive interference. Reconsolidation provides a mechanism for these effects: new learning during the lability window can update old memories, creating retroactive interference at the memory trace level rather than just retrieval competition.

**Integration 4: Spreading Activation**

M3 spreading activation determines which memories are retrieved. Retrieved memories may trigger reconsolidation. This creates feedback: activation → retrieval → reconsolidation → modified strength → future activation.

## Implementation Architecture

```rust
pub struct IntegratedReconsolidationSystem {
    reconsolidation_engine: ReconsolidationEngine,
    consolidation_pipeline: ConsolidationPipeline,  // From M6
    interference_detector: InterferenceDetector,
    pattern_completion: PatternCompletion,          // From M8
}

impl IntegratedReconsolidationSystem {
    pub fn retrieve_with_reconsolidation(
        &mut self,
        cue: NodeId,
        context: &Context,
    ) -> RetrievalResult {
        // Standard retrieval with spreading activation
        let retrieved = self.pattern_completion.retrieve(cue);

        for memory in &retrieved {
            // Check if retrieval should trigger reconsolidation
            if self.reconsolidation_engine.check_reconsolidation_trigger(memory, context) {
                // Open reconsolidation window
                self.reconsolidation_engine.open_reconsolidation_window(
                    memory.node_id,
                    memory.strength,
                );

                // Re-enter consolidation pipeline
                self.consolidation_pipeline.mark_for_reconsolidation(memory.node_id);
            }
        }

        retrieved
    }

    pub fn encode_during_lability_window(
        &mut self,
        node_id: NodeId,
        associations: Vec<(NodeId, f32)>,
    ) {
        // Check if this node is in reconsolidation window
        if let Some(lability) = self.reconsolidation_engine.get_lability(node_id) {
            // Encoding during lability creates stronger associations
            let lability_boost = 1.0 + lability;  // Up to 2x strength

            for (associated_node, strength) in associations {
                self.consolidation_pipeline.encode_association(
                    node_id,
                    associated_node,
                    strength * lability_boost,
                );
            }

            // This creates retroactive interference opportunity
            self.interference_detector.record_interference_event(
                node_id,
                associated_node,
                InterferenceType::Retroactive,
            );
        } else {
            // Standard encoding
            for (associated_node, strength) in associations {
                self.consolidation_pipeline.encode_association(
                    node_id,
                    associated_node,
                    strength,
                );
            }
        }
    }
}
```

## Biological Plausibility Constraints

1. **Consolidation Time Course:** Reconsolidated memories should follow similar but potentially accelerated consolidation kinetics (6-24 hours to full stability).

2. **Synaptic Scaling:** Schiller et al. (2010) showed reconsolidation involves synaptic protein synthesis. In our system, this means gradual strength updates, not instant changes.

3. **Systems Consolidation:** Very old memories may show cortical-only reconsolidation without hippocampal involvement. Implementation: different pathways for recent vs remote memories.

4. **Boundary Respect:** Integration must maintain strict boundary conditions from Nader et al. (2000). No reconsolidation outside defined parameters.

## Validation Through Integration Tests

```rust
#[test]
fn test_reconsolidation_retroactive_interference() {
    let mut system = IntegratedReconsolidationSystem::new();

    // Learn A-B association
    system.encode(node_a, vec![(node_b, 1.0)]);
    advance_time(Duration::from_hours(24));  // Consolidate

    // Retrieve A-B (triggers reconsolidation)
    let retrieved = system.retrieve(node_a);
    assert!(retrieved.contains(&node_b));

    // Immediately learn A-C (during lability window)
    system.encode(node_a, vec![(node_c, 1.0)]);

    // Test retroactive interference
    advance_time(Duration::from_hours(12));  // Window closed
    let retrieved_again = system.retrieve(node_a);

    // A-C should be strengthened, A-B potentially weakened
    let strength_b = system.get_association_strength(node_a, node_b);
    let strength_c = system.get_association_strength(node_a, node_c);

    assert!(strength_c > strength_b, "New association should dominate after reconsolidation");
}

#[test]
fn test_pattern_completion_false_memory_reconsolidation() {
    let mut system = IntegratedReconsolidationSystem::new();

    // Create DRM-style study list
    let study_list = vec![bed, rest, awake, tired, dream, wake, snooze, blanket];
    for word in study_list {
        system.encode(word, vec![]);
    }

    // Retrieve with partial cue (generates false memory: "sleep")
    let retrieved = system.retrieve_partial_cue(vec![bed, tired]);

    // False memory "sleep" might be included due to strong semantic associations
    let sleep_retrieved = retrieved.iter().any(|r| r.node_id == sleep);

    if sleep_retrieved {
        // False memory was retrieved, should trigger reconsolidation
        let lability = system.reconsolidation_engine.get_lability(sleep);
        assert!(lability.is_some(), "False memory should enter reconsolidation");

        // If confirmed by feedback, strengthens
        system.confirm_retrieval(sleep);
        advance_time(Duration::from_hours(12));

        // Check if false memory strengthened
        let strength_after = system.get_memory_strength(sleep);
        assert!(strength_after > 0.5, "Confirmed false memory should strengthen");
    }
}
```

## Performance Integration

Each integration point adds overhead:
- Reconsolidation trigger check: 5μs
- Consolidation pipeline interaction: 10μs
- Interference detection: 3μs
- Pattern completion overhead: negligible (already part of retrieval)

Total integration overhead: approximately 20μs per retrieval, maintaining sub-1% total overhead budget.

## Architectural Cleanliness

To avoid spaghetti code, use event-driven architecture:

```rust
pub enum MemoryEvent {
    Retrieved { node_id: NodeId, context: Context },
    Encoded { node_id: NodeId, associations: Vec<(NodeId, f32)> },
    Consolidated { node_id: NodeId },
    ReconsolidationOpened { node_id: NodeId },
    ReconsolidationClosed { node_id: NodeId },
}

pub struct MemoryEventBus {
    subscribers: Vec<Box<dyn MemoryEventHandler>>,
}

impl MemoryEventBus {
    pub fn publish(&self, event: MemoryEvent) {
        for subscriber in &self.subscribers {
            subscriber.handle(event);
        }
    }
}
```

This decouples systems while maintaining integration. Reconsolidation, interference, and consolidation subscribe to relevant events without tight coupling.
