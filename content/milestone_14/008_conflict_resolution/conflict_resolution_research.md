# Research: Conflict Resolution in Distributed Cognitive Systems

## The Conflict Problem

When nodes consolidate memories independently, they create conflicts. Same episodes might consolidate into different patterns. Different episodes might consolidate into semantically similar patterns. These conflicts must resolve without losing information.

## Conflict Types in Engram

**Type 1: Divergent consolidation**
Episodes {E1, E2, E3} consolidate to different patterns on different nodes due to timing or local context differences.

**Type 2: Concurrent creation**
Both nodes independently create "morning routine" pattern with different supporting episodes and confidence scores.

**Type 3: Update conflicts**
One node updates a pattern's confidence to 0.9, another updates to 0.7, changes happen concurrently.

## Resolution Strategies

**Last-Write-Wins (LWW)**: Use timestamps, keep most recent. Simple but loses information.

**Multi-Value**: Keep all versions, let application resolve. Correct but complex for clients.

**Semantic Merging**: Use domain knowledge (confidence scores, episode overlap) to merge intelligently. Best for cognitive systems.

Engram uses semantic merging with vector clock causality:

```rust
fn resolve_conflict(local: Pattern, remote: Pattern) -> Pattern {
    match local.vector_clock.compare(&remote.vector_clock) {
        Ordering::Greater => local,  // Local is newer
        Ordering::Less => remote,    // Remote is newer
        Ordering::Concurrent => {
            // Merge with confidence-weighted averaging
            merge_semantic_patterns(local, remote)
        }
    }
}

fn merge_semantic_patterns(p1: Pattern, p2: Pattern) -> Pattern {
    let total_conf = p1.confidence + p2.confidence;
    Pattern {
        pattern_id: p1.pattern_id,
        episodes: p1.episodes.union(p2.episodes),
        embedding: (p1.embedding * p1.confidence + p2.embedding * p2.confidence) / total_conf,
        confidence: total_conf / 2.0,  // Average confidence
        vector_clock: p1.vector_clock.merge(p2.vector_clock),
    }
}
```

## Academic Foundation

- **CRDTs**: Shapiro et al. (2011) - conflict-free replicated data types
- **Bayesian Conflict Resolution**: Fink (2011) - probabilistic merging
- **Operational Transformation**: Ellis & Gibbs (1989) - concurrent editing conflicts
- **Vector Clocks**: Fidge (1988) - causality-based resolution

Engram's approach combines vector clocks for causality with confidence-weighted merging for cognitive realism.
