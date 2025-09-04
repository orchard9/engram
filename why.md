# Why Engram

## Problem Statement

Existing graph databases optimize for ACID compliance and deterministic query results. Cognitive systems require probabilistic retrieval, temporal decay, and reconstructive recall. The impedance mismatch between traditional database guarantees and memory dynamics forces researchers to implement memory mechanisms as application logic rather than database primitives.

## Technical Gap

Current solutions fail to provide:
- Native uncertainty quantification on graph elements
- Activation spreading as a first-class query mechanism  
- Temporal dynamics without explicit application management
- Hebbian weight adjustment at the storage layer
- Content-addressable retrieval with partial pattern matching

## Core Hypothesis

Memory-oriented computation requires fundamentally different primitives than transaction-oriented storage. By implementing cognitive principles directly in the storage engine, we eliminate the abstraction penalty of simulating memory atop traditional databases.

## Specific Deficiencies Addressed

1. **Reconstruction vs Reproduction**: Memories change during retrieval. Current databases treat this as corruption rather than feature.

2. **Forgetting Curves**: Exponential decay is computationally expensive when implemented above the storage layer. Native implementation enables efficient bulk decay operations.

3. **Spreading Activation**: Graph traversal treats edges as binary connections. Activation spreading requires continuous propagation with decay.

4. **Pattern Completion**: Partial cues should retrieve complete patterns probabilistically. Hash indices and B-trees cannot support this natively.

5. **Consolidation**: Episodic to semantic transformation requires background processes that current databases treat as expensive migrations rather than continuous operations.

## Target Applications

- Cognitive architectures requiring biologically plausible memory
- Reinforcement learning agents with episodic memory
- Personal AI systems with autobiographical memory
- Temporal knowledge graphs with uncertainty
- Predictive systems based on episodic sequences

## Non-Goals

We explicitly do not target:
- Financial transactions requiring ACID guarantees
- Systems requiring deterministic query results
- Applications needing perfect durability
- Use cases requiring SQL compatibility
- Workloads optimizing for write throughput over read pattern diversity
