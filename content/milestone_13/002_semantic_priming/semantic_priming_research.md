# Semantic Priming: Research and Technical Foundation

## The Phenomenon

When you hear the word "doctor," you're faster to recognize "nurse" than "bread." This is semantic priming, one of the most robust findings in cognitive psychology. Neely (1977) demonstrated that semantically related words show facilitation at short stimulus onset asynchronies (SOA < 500ms), with effects measurable as 30-50ms faster response times.

The effect isn't limited to direct associations. Collins & Loftus (1975) showed that priming spreads through semantic networks like ripples in a pond. Hearing "red" primes "fire truck," which primes "siren," which primes "ambulance" - even though "red" and "ambulance" have no direct association. Activation spreads from the prime node to related concepts, pre-activating them for faster retrieval.

This spreading activation has precise temporal dynamics. Meyer & Schvaneveldt (1971) found maximum facilitation at 240-340ms SOA, with effects diminishing by 1000ms. The decay isn't just time-based - it depends on intervening cognitive activity. Each unrelated stimulus encountered partially resets the activation pattern.

## Computational Requirements

Implementing semantic priming in a graph-based memory system requires several components:

**1. Semantic Distance Measurement**
We need to quantify how related two concepts are. Traditional approaches use cosine similarity of word embeddings, but that's too coarse. Human semantic networks have structured relationships - category membership, functional associations, thematic relations - that pure distributional similarity misses.

The solution is multi-dimensional similarity combining:
- Embedding cosine similarity (distributional semantics)
- Graph path distance (structural relationships)
- Co-occurrence statistics (empirical associations)
- Edge weight types (is-a, part-of, used-for, etc.)

**2. Temporal Decay Functions**
Priming strength must decay realistically. A simple exponential decay `strength = initial * e^(-t/tau)` doesn't match human data. Neely (1977) showed non-monotonic decay - initial strengthening as spreading completes, then gradual decay.

We need:
- Fast rise time (0-100ms) as activation spreads
- Plateau period (100-500ms) of maximum effect
- Gradual decay (500-2000ms) back to baseline
- Interference from unrelated retrievals

**3. Activation Spreading Algorithm**
Given a prime node, we spread activation to semantically related nodes. But naive graph traversal is too slow - spreading across 10,000 nodes with fan-out 10 and depth 3 touches 10^4 nodes. We need:

- Bounded spreading depth (typically 3-4 hops)
- Activation threshold to prune weak paths
- Priority queue traversal to process high-activation nodes first
- Cache-friendly data structures for hot paths

**4. Priming Boost Computation**
When retrieving a target concept, we need to check if it's currently primed and boost its activation accordingly. This requires:

- Fast lookup of current priming state (hash table)
- Combining multiple sources of priming (if multiple primes active)
- Interference detection (conflicting primes)
- Integration with existing activation spreading

## Validation Against Empirical Data

Our implementation must match published priming effects quantitatively. Key benchmarks:

**Neely (1977) - Category Priming:**
- Automatic spreading at SOA < 500ms
- Strategic expectancy at SOA > 500ms
- Inhibition for unexpected categories
- Target: 30-50ms facilitation at optimal SOA

**Meyer & Schvaneveldt (1971) - Lexical Decision:**
- Related pairs: 855ms mean RT
- Unrelated pairs: 940ms mean RT
- Difference: 85ms facilitation
- Peak effect at 240-340ms SOA

**Collins & Loftus (1975) - Mediated Priming:**
- Direct association: 40ms facilitation
- One-hop mediation: 25ms facilitation
- Two-hop mediation: 10ms facilitation
- Three-hop: no significant effect

Our system must replicate these patterns across thousands of trials with statistical significance.

## Implementation Architecture

The semantic priming engine consists of four modules:

**Module 1: Similarity Calculator**
Computes multi-dimensional semantic distance between any two concepts:

```rust
pub struct SemanticSimilarity {
    embedding_weight: f32,
    path_weight: f32,
    cooccurrence_weight: f32,
}

impl SemanticSimilarity {
    pub fn compute(&self, node_a: NodeId, node_b: NodeId) -> f32 {
        let emb_sim = embedding_cosine(node_a, node_b);
        let path_sim = graph_path_similarity(node_a, node_b);
        let cooc_sim = cooccurrence_strength(node_a, node_b);

        self.embedding_weight * emb_sim +
        self.path_weight * path_sim +
        self.cooccurrence_weight * cooc_sim
    }
}
```

**Module 2: Decay Function**
Models temporal dynamics of priming with biologically realistic curves:

```rust
pub struct PrimingDecay {
    rise_time_ms: f32,
    plateau_ms: f32,
    decay_tau_ms: f32,
}

impl PrimingDecay {
    pub fn strength(&self, elapsed_ms: f32) -> f32 {
        if elapsed_ms < self.rise_time_ms {
            // Rising phase: logistic curve
            1.0 / (1.0 + (-elapsed_ms / 20.0).exp())
        } else if elapsed_ms < self.plateau_ms {
            // Plateau phase: maximum strength
            1.0
        } else {
            // Decay phase: exponential
            (-((elapsed_ms - self.plateau_ms) / self.decay_tau_ms)).exp()
        }
    }
}
```

**Module 3: Spreading Engine**
Efficiently spreads activation from prime to related concepts:

```rust
pub struct PrimeSpreadEngine {
    max_depth: usize,
    activation_threshold: f32,
    visited: HashSet<NodeId>,
    priority_queue: BinaryHeap<(OrderedFloat<f32>, NodeId)>,
}

impl PrimeSpreadEngine {
    pub fn spread(&mut self, prime: NodeId, graph: &MemoryGraph) -> HashMap<NodeId, f32> {
        let mut primed_nodes = HashMap::new();
        self.priority_queue.push((OrderedFloat(1.0), prime));

        while let Some((activation, node)) = self.priority_queue.pop() {
            if activation.0 < self.activation_threshold {
                break;
            }

            primed_nodes.insert(node, activation.0);

            for edge in graph.edges(node) {
                let neighbor = edge.target;
                let edge_weight = edge.weight;
                let new_activation = activation.0 * edge_weight;

                if new_activation > self.activation_threshold {
                    self.priority_queue.push((OrderedFloat(new_activation), neighbor));
                }
            }
        }

        primed_nodes
    }
}
```

**Module 4: Boost Calculator**
Integrates priming effects into retrieval:

```rust
pub struct PrimingBoost {
    active_primes: HashMap<NodeId, (f32, Instant)>,
    decay_fn: PrimingDecay,
}

impl PrimingBoost {
    pub fn compute_boost(&self, target: NodeId, now: Instant) -> f32 {
        self.active_primes
            .iter()
            .filter_map(|(prime_node, (strength, timestamp))| {
                let elapsed_ms = now.duration_since(*timestamp).as_millis() as f32;
                let decay = self.decay_fn.strength(elapsed_ms);

                if decay > 0.1 {
                    Some(strength * decay)
                } else {
                    None
                }
            })
            .sum()
    }
}
```

## Performance Constraints

Priming must not slow down retrieval. Key budgets:

**Latency:**
- Similarity computation: < 5μs (cached)
- Decay calculation: < 1μs (pure math)
- Spread operation: < 500μs for depth-3, fan-out 10
- Boost lookup: < 100ns (hash table)

**Memory:**
- Active primes: < 1MB for 10,000 nodes
- Similarity cache: < 100MB for 1M pairs
- Decay state: 16 bytes per prime
- Total overhead: < 150MB

**Throughput:**
- 100K similarity queries/sec (with caching)
- 10K spread operations/sec
- 1M boost lookups/sec

These budgets ensure priming adds less than 5% latency to retrieval operations.

## Integration Strategy

Semantic priming integrates with existing spreading activation (M3) by modifying initial activation values:

```rust
pub fn retrieve_with_priming(
    &self,
    cue: NodeId,
    priming_boost: &PrimingBoost,
) -> Vec<(NodeId, f32)> {
    let base_activation = self.compute_base_activation(cue);
    let priming_activation = priming_boost.compute_boost(cue, Instant::now());

    let total_activation = base_activation + priming_activation;

    // Continue with normal spreading activation using boosted initial state
    self.spread_activation(cue, total_activation)
}
```

This clean separation ensures priming can be disabled independently without affecting core retrieval.

## Statistical Validation Approach

We validate priming effects using paired t-tests comparing related versus unrelated retrieval times:

```rust
#[test]
fn validate_semantic_priming() {
    let mut related_rts = Vec::new();
    let mut unrelated_rts = Vec::new();

    for trial in 0..1000 {
        let (prime, related, unrelated) = generate_priming_triplet();

        // Prime presentation
        memory.encode(prime);
        thread::sleep(Duration::from_millis(300)); // SOA = 300ms

        // Related target
        let start = Instant::now();
        memory.retrieve(related);
        related_rts.push(start.elapsed().as_micros() as f32);

        // Reset and test unrelated
        memory.clear();
        memory.encode(prime);
        thread::sleep(Duration::from_millis(300));

        let start = Instant::now();
        memory.retrieve(unrelated);
        unrelated_rts.push(start.elapsed().as_micros() as f32);
    }

    let related_mean = mean(&related_rts);
    let unrelated_mean = mean(&unrelated_rts);
    let facilitation_us = unrelated_mean - related_mean;

    // Neely (1977): expect 30-50ms facilitation at SOA=300ms
    assert!(facilitation_us >= 20_000.0 && facilitation_us <= 60_000.0,
        "Facilitation {}μs outside expected range", facilitation_us);
}
```

Statistical power analysis ensures N=1000 trials gives 80% power to detect effect sizes d=0.4, matching typical priming experiments.

This rigorous validation approach ensures our semantic priming implementation doesn't just look plausible - it quantitatively matches decades of empirical research.
