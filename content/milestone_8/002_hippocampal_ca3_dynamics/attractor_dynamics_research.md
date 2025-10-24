# Hippocampal CA3 Attractor Dynamics: Research Foundations

## Hopfield Networks and Energy Minimization

### Hopfield (1982): Neural Networks and Physical Systems
John Hopfield's seminal paper demonstrated that neural networks with symmetric weights implement content-addressable memory through energy minimization.

**Energy Function:**
```
E = -0.5 * Σᵢⱼ wᵢⱼ sᵢ sⱼ
```

**Key Properties:**
- Symmetric weight matrix (wᵢⱼ = wⱼᵢ) ensures energy decreases monotonically
- Local minima correspond to stored patterns (attractor basins)
- Network converges to nearest stored pattern from partial cues
- Storage capacity: ~0.15N patterns for N neurons

**Implications for Engram:**
CA3 must maintain weight symmetry. Energy should decrease during convergence. Non-convergence signals conflicting patterns or insufficient cue strength.

## Hippocampal CA3 as Autoassociative Memory

### Marr (1971): Simple Memory Theory
David Marr proposed archicortex (hippocampus) as rapid storage system using:
- Sparse coding (few active neurons)
- Recurrent collaterals (CA3 → CA3 connections)
- Heteroassociative links (CA3 → CA1 output)

**Storage Mechanism:**
CA3 recurrent collaterals implement Hebbian learning: neurons that fire together wire together. Stored patterns create attractor basins.

**Retrieval Mechanism:**
Partial cue activates subset of CA3 neurons. Recurrent connections amplify activation toward stored pattern. CA1 output gates high-confidence completions.

### McNaughton & Morris (1987): Autoassociative Memory Requirements
Mathematical analysis of hippocampal autoassociation:
- Minimum cue overlap: 10-30% for reliable completion
- Sparsity critical: Dense patterns cause catastrophic interference
- Recurrent weight strength determines attractor basin depth
- CA1 acts as comparator/novelty detector

## Pattern Separation vs Pattern Completion

### Treves & Rolls (1994): Capacity Analysis
Computational analysis separating storage (DG/CA3) from retrieval (CA3/CA1):

**Dentate Gyrus:**
- Pattern separation: Make similar inputs distinct
- Orthogonalization through sparse coding
- Prevents catastrophic interference

**CA3:**
- Pattern completion: Reconstruct whole from part
- Autoassociative retrieval via recurrent collaterals
- Storage capacity: C = (k²/a) ln(1/a) where k=sparsity, a=activity

**CA1:**
- Output gating and comparison
- Novelty detection (mismatch between CA3 and EC input)
- Temporal sequencing

**Optimal Parameters:**
- CA3 sparsity: 2-5% active neurons
- Cue overlap requirement: 20-40% for reliable completion
- Recurrent weight strength: Balanced between stability and capacity

### Leutgeb et al. (2007): Pattern Separation Empirical Data
In vivo recordings showing:
- DG remaps completely between environments (pattern separation)
- CA3 partial remapping (pattern completion bias)
- CA1 intermediate remapping (balanced)

Pattern separation index: Measure of neural code distinctness between similar inputs.

**Implications:**
Engram needs both separation (at encoding) and completion (at retrieval). Task 002 implements completion. Future: orthogonalization at storage.

## Theta Rhythm Constraints

### Buzsáki (2002): Theta Oscillations and Memory
Hippocampal theta rhythm (4-8 Hz) organizes memory encoding/retrieval:
- Theta cycles: ~125-250ms period (140ms typical)
- Each cycle contains 5-7 gamma cycles (40-100 Hz)
- Encoding phase: EC input → CA3/CA1
- Retrieval phase: CA3 → CA1 output
- Completion must occur within single theta cycle

**Biological Constraint:**
Pattern completion limited to 7 iterations (gamma cycles per theta cycle). Exceeding this violates biological plausibility.

**Engram Implementation:**
```rust
max_iterations: 7  // Theta rhythm constraint
```

Convergence within 7 iterations >95% of cases, or return explicit failure.

### Skaggs et al. (1996): Theta Phase Precession
Place cells fire at progressively earlier theta phases as animal traverses place field. Enables temporal sequencing within theta cycles.

**Application to Completion:**
Iterations should correspond to gamma cycles. Early iterations activate broad population, later iterations refine to specific pattern.

## CA1 Output Gating

### Lisman & Otmakhova (2001): CA1 as Comparator
CA1 receives two inputs:
- CA3 completion output
- Direct EC input (what actually happened)

Compares inputs to detect novelty/mismatch. High mismatch → novelty signal → enhanced encoding.

**Engram CA1 Gate:**
Compares CA3 completion confidence to threshold. Low confidence → reject output, return "insufficient evidence." Prevents hallucination.

### Hasselmo et al. (2002): Acetylcholine Modulation
ACh controls encoding/retrieval balance:
- High ACh: Suppress CA3 recurrent connections, favor encoding
- Low ACh: Enhance recurrent connections, favor retrieval

**Computational Analog:**
Configurable CA3 weight strength. High weights = strong completion (retrieval mode). Low weights = sparse completion (encoding mode).

## Convergence Dynamics

### Amit (1989): Attractor Neural Networks
Analysis of convergence in Hopfield-like networks:
- Basin of attraction: Region in state space converging to same pattern
- Spurious attractors: Local minima not corresponding to stored patterns
- Capacity limits: Too many patterns create spurious attractors

**Basin Depth Metric:**
Energy difference between random state and attractor. Deeper basins → more reliable completion, fewer iterations.

**Engram Monitoring:**
Track final energy, energy delta, iterations to convergence. Correlation with reconstruction accuracy validates attractor dynamics.

### Hertz, Krogh & Palmer (1991): Statistical Mechanics of Learning
Thermodynamic analysis of neural network dynamics:
- Temperature parameter controls stochasticity
- Zero temperature: Deterministic updates (Engram uses this)
- Simulated annealing: Gradually decrease temperature to escape local minima

**Convergence Guarantee:**
Symmetric weights + monotonic energy decrease → convergence guaranteed (may be spurious attractor).

## Biological Validation Criteria

### Computational Neuroscience Standards
Pattern completion implementation should match:

1. **Convergence Speed:** 5-7 iterations typical (theta rhythm constraint)
2. **Energy Landscape:** Monotonic decrease >99% of cases
3. **Sparsity:** 2-5% active neurons (k-winner-take-all)
4. **Cue Overlap:** 20-40% minimum for reliable completion
5. **Capacity Scaling:** C ~ k² ln(1/k) for k=sparsity
6. **Graceful Degradation:** Accuracy decreases smoothly with cue quality

### Empirical Benchmarks from Literature
- Treves & Rolls: Completion accuracy >90% with 40% cue, >70% with 25% cue
- Marr capacity bound: ~15% of neuron count for uncorrelated patterns
- McNaughton: CA1 gating threshold ~0.6-0.8 for behavioral relevance

**Engram Targets:**
- Convergence rate >95% within 7 iterations
- Energy decrease >99% of cases
- Accuracy >85% with 30% cue overlap
- CA1 threshold 0.7 (default, configurable)

## Implementation Considerations

### Weight Matrix Structure
Hebbian learning: W = Σᵢ (pᵢ pᵢᵀ - I)

For 768-dim embeddings: 768×768 weight matrix (2.4MB float32).

**Sparse Alternatives:**
Most weights small. Can threshold to sparse matrix if memory becomes issue.

### Sparsity Enforcement
k-winner-take-all: Keep top k% activations, zero rest.

Biological: Lateral inhibition. Computational: Sort and threshold.

**Fast Implementation:**
Partial quicksort (nth_element) to find k-th largest. O(N) average case. Faster than full sort.

### Numerical Stability
Sigmoid activation can saturate (gradients near zero). Use bounded activations:
```rust
activation[i] = activation[i].max(-10.0).min(10.0);  // Prevent overflow
```

Energy computation: Watch for numerical overflow with large weight matrices.

## References

1. Hopfield, J. J. (1982). Neural networks and physical systems with emergent collective computational abilities. PNAS, 79(8), 2554-2558.
2. Marr, D. (1971). Simple memory: A theory for archicortex. Philosophical Transactions of the Royal Society B, 262(841), 23-81.
3. McNaughton, B. L., & Morris, R. G. M. (1987). Hippocampal synaptic enhancement and information storage within a distributed memory system. Trends in Neurosciences, 10(10), 408-415.
4. Treves, A., & Rolls, E. T. (1994). Computational analysis of the role of the hippocampus in memory. Hippocampus, 4(3), 374-391.
5. Buzsáki, G. (2002). Theta oscillations in the hippocampus. Neuron, 33(3), 325-340.
6. Leutgeb, S., et al. (2007). Pattern separation in the dentate gyrus and CA3 of the hippocampus. Science, 315(5814), 961-966.
7. Lisman, J. E., & Otmakhova, N. A. (2001). Storage, recall, and novelty detection of sequences by the hippocampus. Hippocampus, 11(3), 256-264.
8. Amit, D. J. (1989). Modeling brain function: The world of attractor neural networks. Cambridge University Press.
9. Hertz, J., Krogh, A., & Palmer, R. G. (1991). Introduction to the theory of neural computation. Addison-Wesley.
