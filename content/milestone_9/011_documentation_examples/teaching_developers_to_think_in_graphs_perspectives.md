# Perspectives: Teaching Developers to Think in Graphs

## Technical Communication Lead Perspective

Documentation is a cognitive interface. Bad docs = cognitive friction.

**Key insight**: Developers don't read docs sequentially. They:
1. Search for specific problem
2. Skim for relevant section
3. Copy/paste example
4. Modify for their use case

Optimize for this pattern, not linear reading.

**Structure**:
- Tutorials: "Your first query" (copy/paste ready)
- How-To: "Solve X problem" (specific solutions)
- Explanation: "Why it works this way" (mental models)
- Reference: "All syntax options" (lookup table)

Diátaxis framework. Works for APIs.

## Cognitive Architecture Perspective

Mental model mismatch = learning friction.

SQL developers think: Entities → Tables, Relationships → Joins
Graph developers think: Nodes → Entities, Edges → Relationships

The shift: Relationships become first-class.

**Teaching strategy**: Bridge from familiar to unfamiliar.

Example:
"In SQL, finding related records requires JOINs. In Engram, traversing relationships is SPREAD - a first-class operation, not a JOIN."

Anchor new concepts to existing knowledge.

## Memory Systems Perspective

Progressive disclosure matches human learning.

Don't dump entire syntax reference on newcomers. Build complexity gradually:

**Stage 1**: Simple queries (success builds confidence)
**Stage 2**: Add constraints (small complexity increase)
**Stage 3**: Composition (combine primitives)
**Stage 4**: Optimization (performance tuning)

Each stage: Explain → Example → Exercise.

Cognitive load management. Too much too fast = overwhelm.
