# Source Attribution System: Architectural Perspectives

## Cognitive Architecture: Reality Monitoring

Johnson's Source Monitoring Framework: The brain tracks whether details came from perception (external) or imagination (internal).

Task 005 implements this for memory systems:
- Recalled = external (came from user's query)
- Reconstructed = internal (generated from patterns)
- Imagined = weak internal (speculation)
- Consolidated = learned prior (semantic knowledge)

Four source types match cognitive taxonomy. Transparent attribution prevents confabulation.

## Memory Systems: Metacognitive Monitoring

Koriat & Goldsmith: People regulate memory output based on metacognitive monitoring. When confidence low, withhold output or provide alternatives.

Task 005 implements regulation through:
- CA1 gating threshold (withhold if confidence <0.7)
- Alternative hypotheses (show uncertainty explicitly)
- Source confidence (independent of field confidence)

This matches human metacognition: I remember something, but I'm not sure if I experienced it or imagined it.

## Systems Architecture: Audit Trail

Production systems need observability. When completion goes wrong, you need to debug why.

Source attribution provides audit trail:
```json
{
  "field": "beverage",
  "value": "coffee",
  "source": "Reconstructed",
  "evidence": [
    {"episode": "123", "similarity": 0.82, "weight": 0.7},
    {"episode": "127", "similarity": 0.76, "weight": 0.6}
  ],
  "source_confidence": 0.72
}
```

For each field, you can trace:
- Why this value was chosen
- Which episodes contributed
- What confidence to assign
- Whether attribution is reliable

Enables debugging, user transparency, and compliance (explainable AI).

## Rust Implementation: Zero-Cost Enum Tags

Source attribution adds no runtime cost:
```rust
pub enum MemorySource {
    Recalled,
    Reconstructed,
    Imagined,
    Consolidated,
}
```

Enum tag = single byte. No allocations. Embedded in IntegratedField struct. Zero overhead.

Source classification = pattern matching on evidence contributions. Compiles to efficient branching.

Result: Full source provenance with negligible performance impact.
