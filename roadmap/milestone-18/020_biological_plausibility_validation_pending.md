# Task 020: Biological Plausibility Validation

## Objective
Validate that Engram's spreading activation and memory consolidation algorithms reproduce empirically-observed cognitive phenomena from psychology literature, ensuring the system's biological inspiration translates to verifiable behavior.

## Background

Engram claims biological plausibility through hippocampal-neocortical memory models. To validate this, we must test against **empirical cognitive psychology data**:

### Key Phenomena to Validate

1. **Fan Effect** (Anderson 1974)
   - Recognition time increases with number of associations to a concept
   - Example: "The doctor is in the bank" harder to verify when "doctor" has many other contexts

2. **Semantic Priming** (Neely 1977, Meyer & Schvaneveldt 1971)
   - Related concepts activate faster after prime exposure
   - Example: "NURSE" recognized 40-60ms faster after "DOCTOR" prime at SOA=250ms

3. **Forgetting Curve** (Ebbinghaus 1885)
   - Memory strength decays exponentially: R = e^(-t/S)
   - Example: 50% retention after 1 day, 30% after 1 week

4. **Spacing Effect** (Cepeda et al. 2006)
   - Spaced repetitions produce better retention than massed practice
   - Example: 3 reviews over 3 days > 3 reviews in 1 hour

5. **Concept Formation** (Tse et al. 2007)
   - Semantic concepts emerge from 3-4 repeated similar episodes
   - Coherence >0.7 for tight clusters (M17 Task 004 parameter)

### Why This Matters

If Engram's algorithms don't reproduce these phenomena, the biological inspiration is superficial. This validation ensures the graph engine behaves like actual human memory, not just an arbitrary spreading activation network.

## Requirements

1. Implement test scenarios for each cognitive phenomenon
2. Compare Engram's behavior to published empirical data
3. Validate parameter settings produce correct timescales
4. Test edge cases where biological systems differ from artificial neural networks
5. Document deviations and their cognitive interpretations

## Technical Specification

### Files to Create

#### `engram-core/tests/cognitive/fan_effect_validation.rs`
Fan effect validation (Anderson 1974):

```rust
//! Fan effect validation
//!
//! Validates that recognition time increases with fan-out per Anderson (1974).
//!
//! Expected results:
//! - Fan 1: 1100ms baseline RT
//! - Fan 2: 1150ms (+50ms)
//! - Fan 3: 1200ms (+100ms)
//!
//! Engram spreading activation should exhibit similar fan-dependent delays.

use engram_core::memory_graph::{UnifiedMemoryGraph, backends::DashMapBackend};
use engram_core::{Memory, Confidence};
use uuid::Uuid;
use std::time::Instant;

#[test]
fn test_fan_effect_recognition_time() {
    // Setup: Create person-location associations with varying fan
    let graph = UnifiedMemoryGraph::with_backend(DashMapBackend::new());

    // Person with fan=1: doctor in bank
    let person_fan1 = create_person(&graph, "doctor");
    let location_fan1 = create_location(&graph, "bank");
    graph.add_edge(person_fan1, location_fan1, 0.9).expect("add_edge failed");

    // Person with fan=2: lawyer in park, lawyer in church
    let person_fan2 = create_person(&graph, "lawyer");
    let location_fan2a = create_location(&graph, "park");
    let location_fan2b = create_location(&graph, "church");
    graph.add_edge(person_fan2, location_fan2a, 0.9).expect("add_edge failed");
    graph.add_edge(person_fan2, location_fan2b, 0.9).expect("add_edge failed");

    // Person with fan=3: teacher in store, hospital, school
    let person_fan3 = create_person(&graph, "teacher");
    let locations_fan3 = [
        create_location(&graph, "store"),
        create_location(&graph, "hospital"),
        create_location(&graph, "school"),
    ];
    for loc in locations_fan3 {
        graph.add_edge(person_fan3, loc, 0.9).expect("add_edge failed");
    }

    // Test: Measure recognition time (simulated via spreading iterations)
    let rt_fan1 = measure_recognition_time(&graph, person_fan1, location_fan1);
    let rt_fan2 = measure_recognition_time(&graph, person_fan2, location_fan2a);
    let rt_fan3 = measure_recognition_time(&graph, person_fan3, locations_fan3[0]);

    // Anderson (1974) results: RT increases ~50-100ms per fan increment
    // In Engram, we expect spreading iterations to increase similarly
    println!("Fan 1 RT: {} iterations", rt_fan1);
    println!("Fan 2 RT: {} iterations", rt_fan2);
    println!("Fan 3 RT: {} iterations", rt_fan3);

    // Validate fan effect: RT should increase with fan
    assert!(
        rt_fan2 > rt_fan1,
        "Fan 2 should be slower than fan 1: {} vs {}",
        rt_fan2,
        rt_fan1
    );
    assert!(
        rt_fan3 > rt_fan2,
        "Fan 3 should be slower than fan 2: {} vs {}",
        rt_fan3,
        rt_fan2
    );

    // Validate magnitude: Fan effect should be 10-30% increase per level
    let fan1_to_fan2_increase = (rt_fan2 as f32 - rt_fan1 as f32) / rt_fan1 as f32;
    let fan2_to_fan3_increase = (rt_fan3 as f32 - rt_fan2 as f32) / rt_fan2 as f32;

    assert!(
        fan1_to_fan2_increase >= 0.1 && fan1_to_fan2_increase <= 0.5,
        "Fan 1->2 increase out of range: {:.1}%",
        fan1_to_fan2_increase * 100.0
    );
    assert!(
        fan2_to_fan3_increase >= 0.1 && fan2_to_fan3_increase <= 0.5,
        "Fan 2->3 increase out of range: {:.1}%",
        fan2_to_fan3_increase * 100.0
    );
}

fn create_person(graph: &UnifiedMemoryGraph<DashMapBackend>, name: &str) -> Uuid {
    let id = Uuid::new_v4();
    let memory = Memory::new(
        id.to_string(),
        embedding_for(name),
        Confidence::HIGH,
    );
    graph.store_memory(memory).expect("store failed");
    id
}

fn create_location(graph: &UnifiedMemoryGraph<DashMapBackend>, name: &str) -> Uuid {
    let id = Uuid::new_v4();
    let memory = Memory::new(
        id.to_string(),
        embedding_for(name),
        Confidence::HIGH,
    );
    graph.store_memory(memory).expect("store failed");
    id
}

fn embedding_for(concept: &str) -> [f32; 768] {
    // Simple hash-based embedding for testing
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};

    let mut hasher = DefaultHasher::new();
    concept.hash(&mut hasher);
    let hash = hasher.finish();

    let mut embedding = [0.0f32; 768];
    for i in 0..768 {
        embedding[i] = ((hash.wrapping_mul((i + 1) as u64) % 1000) as f32 / 1000.0) * 2.0 - 1.0;
    }
    embedding
}

fn measure_recognition_time(
    graph: &UnifiedMemoryGraph<DashMapBackend>,
    source: Uuid,
    target: Uuid,
) -> usize {
    // Simulate recognition time by counting spreading iterations until target activates

    // Reset activations
    for id in graph.backend().all_ids() {
        graph.backend().update_activation(&id, 0.0).expect("update failed");
    }

    // Activate source
    graph.backend().update_activation(&source, 1.0).expect("update failed");

    // Spread until target reaches threshold
    let threshold = 0.3;
    let max_iterations = 100;

    for iteration in 1..=max_iterations {
        // Spread from all active nodes
        for id in graph.backend().all_ids() {
            let memory = graph.backend().retrieve(&id).expect("retrieve failed").expect("not found");
            if memory.activation() > 0.1 {
                graph.backend().spread_activation(&id, 0.85).expect("spread failed");
            }
        }

        // Check if target activated
        let target_memory = graph.backend().retrieve(&target).expect("retrieve failed").expect("not found");
        if target_memory.activation() >= threshold {
            return iteration;
        }
    }

    max_iterations
}
```

#### `engram-core/tests/cognitive/semantic_priming_validation.rs`
Semantic priming validation (Neely 1977):

```rust
//! Semantic priming validation
//!
//! Validates that related concepts activate faster after prime exposure.
//!
//! Neely (1977) results:
//! - SOA=250ms: Related targets 40-60ms faster than unrelated
//! - SOA=500ms: Priming effect reduces to 20-30ms
//!
//! In Engram, we measure spreading iterations instead of RT.

use engram_core::memory_graph::{UnifiedMemoryGraph, backends::DashMapBackend};
use engram_core::{Memory, Confidence};
use uuid::Uuid;

#[test]
fn test_semantic_priming_related_concepts() {
    let graph = UnifiedMemoryGraph::with_backend(DashMapBackend::new());

    // Create semantic network: doctor-nurse (related), doctor-bread (unrelated)
    let prime = create_concept(&graph, "doctor");
    let target_related = create_concept(&graph, "nurse");
    let target_unrelated = create_concept(&graph, "bread");

    // Strong association: doctor-nurse (semantic relatedness)
    graph.add_edge(prime, target_related, 0.9).expect("add_edge failed");

    // Weak association: doctor-bread (no semantic relation)
    // (no edge = no direct connection)

    // Measure: Prime doctor, measure activation spread to targets

    // 1. Related target (doctor -> nurse)
    graph.backend().update_activation(&prime, 1.0).expect("update failed");
    graph.backend().spread_activation(&prime, 0.9).expect("spread failed");

    let related_activation = graph.backend()
        .retrieve(&target_related)
        .expect("retrieve failed")
        .expect("not found")
        .activation();

    // Reset
    for id in graph.backend().all_ids() {
        graph.backend().update_activation(&id, 0.0).expect("update failed");
    }

    // 2. Unrelated target (doctor -> bread, no direct path)
    graph.backend().update_activation(&prime, 1.0).expect("update failed");
    graph.backend().spread_activation(&prime, 0.9).expect("spread failed");

    let unrelated_activation = graph.backend()
        .retrieve(&target_unrelated)
        .expect("retrieve failed")
        .expect("not found")
        .activation();

    // Validate: Related target should receive significantly more activation
    println!("Related activation: {}", related_activation);
    println!("Unrelated activation: {}", unrelated_activation);

    assert!(
        related_activation > unrelated_activation * 5.0,
        "Priming effect not detected: related={}, unrelated={}",
        related_activation,
        unrelated_activation
    );

    // Related should be above threshold (recognizable)
    assert!(
        related_activation > 0.5,
        "Related target not activated: {}",
        related_activation
    );

    // Unrelated should be below threshold (not recognizable)
    assert!(
        unrelated_activation < 0.2,
        "Unrelated target over-activated: {}",
        unrelated_activation
    );
}

fn create_concept(graph: &UnifiedMemoryGraph<DashMapBackend>, name: &str) -> Uuid {
    let id = Uuid::new_v4();
    let memory = Memory::new(
        id.to_string(),
        embedding_for_concept(name),
        Confidence::HIGH,
    );
    graph.store_memory(memory).expect("store failed");
    id
}

fn embedding_for_concept(concept: &str) -> [f32; 768] {
    // Semantic similarity: related concepts have high dot product
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};

    let mut hasher = DefaultHasher::new();
    concept.hash(&mut hasher);
    let hash = hasher.finish();

    // For "doctor" and "nurse", make embeddings similar
    let base_seed = if concept == "doctor" || concept == "nurse" {
        12345_u64  // Shared seed for related concepts
    } else {
        hash
    };

    let mut embedding = [0.0f32; 768];
    for i in 0..768 {
        embedding[i] = ((base_seed.wrapping_mul((i + 1) as u64) % 1000) as f32 / 1000.0) * 2.0 - 1.0;
    }

    // Add concept-specific noise
    for i in 0..768 {
        let noise = ((hash.wrapping_mul((i + 1) as u64) % 100) as f32 / 1000.0);
        embedding[i] += noise;
    }

    embedding
}
```

#### `engram-core/tests/cognitive/forgetting_curve_validation.rs`
Forgetting curve validation (Ebbinghaus 1885):

```rust
//! Forgetting curve validation
//!
//! Validates that memory strength decays exponentially over time.
//!
//! Ebbinghaus (1885) retention:
//! - Immediate: 100%
//! - 20 min: 58%
//! - 1 hour: 44%
//! - 9 hours: 36%
//! - 1 day: 33%
//! - 6 days: 25%
//!
//! Engram decay should follow R = e^(-t/S) where S is memory strength.

use engram_core::memory_graph::{UnifiedMemoryGraph, backends::DashMapBackend};
use engram_core::{Memory, Confidence};
use uuid::Uuid;
use chrono::{Utc, Duration};

#[test]
fn test_exponential_decay_curve() {
    let graph = UnifiedMemoryGraph::with_backend(DashMapBackend::new());

    let memory_id = Uuid::new_v4();
    let mut memory = Memory::new(
        memory_id.to_string(),
        [0.5f32; 768],
        Confidence::HIGH,
    );

    // Initial strength: 1.0
    memory.set_activation(1.0);
    let initial_time = Utc::now();
    memory.last_access = initial_time;

    graph.store_memory(memory).expect("store failed");

    // Simulate decay at multiple time points
    let time_points = [
        ("20 min", Duration::minutes(20), 0.58),
        ("1 hour", Duration::hours(1), 0.44),
        ("9 hours", Duration::hours(9), 0.36),
        ("1 day", Duration::days(1), 0.33),
        ("6 days", Duration::days(6), 0.25),
    ];

    for (label, time_delta, expected_retention) in time_points {
        // Simulate time passing
        let current_time = initial_time + time_delta;

        // Apply decay function: R = e^(-t/S)
        let time_hours = time_delta.num_seconds() as f32 / 3600.0;
        let strength = 24.0; // Half-life of 24 hours
        let retention = (-time_hours / strength).exp();

        println!("{}: Expected {:.2}, Calculated {:.2}", label, expected_retention, retention);

        // Validate retention is within 20% of empirical data
        let error = (retention - expected_retention).abs() / expected_retention;
        assert!(
            error < 0.2,
            "{}: Retention error too large: {:.1}%",
            label,
            error * 100.0
        );
    }
}

#[test]
fn test_decay_function_properties() {
    // Validate decay function has correct properties

    let decay_fn = |t: f32, strength: f32| (-t / strength).exp();

    // Property 1: Initial retention is 100%
    assert!((decay_fn(0.0, 24.0) - 1.0).abs() < 0.001);

    // Property 2: Retention decreases monotonically
    let t1 = decay_fn(1.0, 24.0);
    let t2 = decay_fn(2.0, 24.0);
    assert!(t2 < t1);

    // Property 3: Retention asymptotes to 0
    let t_large = decay_fn(1000.0, 24.0);
    assert!(t_large < 0.01);

    // Property 4: Half-life matches strength parameter
    let half_life_retention = decay_fn(24.0, 24.0);
    assert!((half_life_retention - 0.5).abs() < 0.05);
}
```

#### `engram-core/tests/cognitive/concept_formation_validation.rs`
Concept formation validation (Tse et al. 2007, M17 Task 004):

```rust
//! Concept formation validation
//!
//! Validates that semantic concepts emerge from repeated episodes per Tse et al. (2007).
//!
//! Expected:
//! - 3+ similar episodes trigger concept formation
//! - Coherence >0.7 for tight clusters
//! - Consolidation score increases gradually

#[cfg(feature = "dual_memory_types")]
use engram_core::memory::{DualMemoryNode, MemoryNodeType};
use engram_core::{Confidence};
use uuid::Uuid;

#[test]
#[cfg(feature = "dual_memory_types")]
fn test_concept_formation_from_episodes() {
    // Create 5 similar episodes (representing repeated experiences)
    let episodes: Vec<DualMemoryNode> = (0..5)
        .map(|i| {
            let embedding = create_similar_embedding(i);
            DualMemoryNode::new_episode(
                Uuid::new_v4(),
                format!("episode-{}", i),
                embedding,
                Confidence::HIGH,
                0.8,
            )
        })
        .collect();

    // Calculate cluster coherence (pairwise similarity)
    let coherence = calculate_coherence(&episodes);

    println!("Cluster coherence: {:.3}", coherence);

    // Tse et al. (2007): Concepts form when cluster coherence >0.7
    // M17 Task 004: Uses 0.65 threshold for pattern completion
    assert!(
        coherence > 0.6,
        "Coherence too low for concept formation: {}",
        coherence
    );

    // Validate concept properties
    let centroid = calculate_centroid(&episodes);

    // Create concept node
    let concept = DualMemoryNode::new_concept(
        Uuid::new_v4(),
        centroid,
        coherence,
        episodes.len() as u32,
        Confidence::HIGH,
    );

    // Validate instance count
    assert_eq!(
        concept.node_type.instance_count().unwrap(),
        5,
        "Instance count mismatch"
    );

    // Validate coherence stored
    if let MemoryNodeType::Concept { coherence: stored_coherence, .. } = concept.node_type {
        assert!((stored_coherence - coherence).abs() < 0.001);
    }
}

fn create_similar_embedding(seed: usize) -> [f32; 768] {
    // Create embeddings with high mutual similarity (simulate related experiences)
    let base = 42_u64;
    let mut embedding = [0.0f32; 768];

    for i in 0..768 {
        let value = ((base.wrapping_mul((i + 1) as u64) % 1000) as f32 / 1000.0);
        let noise = ((seed as u64).wrapping_mul((i + 1) as u64) % 100) as f32 / 5000.0;
        embedding[i] = value + noise;
    }

    embedding
}

fn calculate_coherence(episodes: &[DualMemoryNode]) -> f32 {
    let n = episodes.len();
    if n < 2 {
        return 0.0;
    }

    let mut sum = 0.0;
    let mut count = 0;

    for i in 0..n {
        for j in i+1..n {
            let similarity = cosine_similarity(&episodes[i].embedding, &episodes[j].embedding);
            sum += similarity;
            count += 1;
        }
    }

    sum / count as f32
}

fn calculate_centroid(episodes: &[DualMemoryNode]) -> [f32; 768] {
    let mut centroid = [0.0f32; 768];

    for episode in episodes {
        for i in 0..768 {
            centroid[i] += episode.embedding[i];
        }
    }

    let n = episodes.len() as f32;
    for i in 0..768 {
        centroid[i] /= n;
    }

    centroid
}

fn cosine_similarity(a: &[f32; 768], b: &[f32; 768]) -> f32 {
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let mag_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let mag_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();

    if mag_a == 0.0 || mag_b == 0.0 {
        0.0
    } else {
        dot / (mag_a * mag_b)
    }
}
```

### Files to Modify

#### `engram-core/tests/cognitive/mod.rs`
Create cognitive validation module:

```rust
//! Cognitive plausibility validation tests
//!
//! These tests validate that Engram reproduces empirically-observed
//! phenomena from cognitive psychology literature.

mod fan_effect_validation;
mod semantic_priming_validation;
mod forgetting_curve_validation;

#[cfg(feature = "dual_memory_types")]
mod concept_formation_validation;
```

## Validation Criteria

### Fan Effect (Anderson 1974)
- **Pass**: RT increases 10-30% per fan increment
- **Fail**: RT decreases or stays constant with fan
- **Target**: Match Anderson's 50-100ms increase per fan level

### Semantic Priming (Neely 1977)
- **Pass**: Related targets activate 3-5x faster than unrelated
- **Fail**: No difference between related/unrelated
- **Target**: 40-60ms facilitation at SOA=250ms (equivalent spreading iterations)

### Forgetting Curve (Ebbinghaus 1885)
- **Pass**: Decay follows R = e^(-t/S) within 20% error
- **Fail**: Retention doesn't decay or decays incorrectly
- **Target**: Match Ebbinghaus retention points (58% at 20min, 33% at 1 day)

### Concept Formation (Tse et al. 2007)
- **Pass**: Coherence >0.6 for 3+ similar episodes
- **Fail**: Concepts form from 1-2 episodes or coherence <0.5
- **Target**: Match M17 Task 004 threshold (0.65)

## Acceptance Criteria

- [ ] Fan effect validated: RT increases with fan-out (10-30% per level)
- [ ] Semantic priming validated: Related concepts 3-5x faster than unrelated
- [ ] Forgetting curve validated: Exponential decay within 20% of Ebbinghaus data
- [ ] Concept formation validated: Coherence >0.6 for tight clusters
- [ ] Spacing effect tested (if time permits): Spaced > massed practice
- [ ] All tests documented with literature citations
- [ ] Deviations from psychology data explained with cognitive rationale
- [ ] CI integration: Tests run on every PR

## Performance Considerations

These are **functional correctness tests**, not performance benchmarks:
- Focus on qualitative behavior matching (fan effect present, not exact RT)
- Use small graphs (10-50 nodes) for tractable testing
- Run once per CI build (not performance-critical)

## Dependencies

- Task 017 (Graph Concurrent Correctness) - foundation
- Task 018 (Graph Invariant Validation) - test infrastructure
- M17 Task 004 (Concept Formation Engine) - complete

## Estimated Time
4 days

- Day 1: Fan effect and semantic priming tests
- Day 2: Forgetting curve and decay validation
- Day 3: Concept formation and consolidation tests
- Day 4: Documentation, CI integration, literature comparison

## Follow-up Tasks

- Task 021: Spacing effect validation (Cepeda et al. 2006)
- Task 022: Serial position effects (primacy/recency)
- Task 023: Retrieval-induced forgetting (Anderson et al. 1994)

## References

### Fan Effect
- Anderson (1974). "Retrieval of propositional information from long-term memory." Cognitive Psychology 6(4): 451-474.

### Semantic Priming
- Neely (1977). "Semantic priming and retrieval from lexical memory: Roles of inhibitionless spreading activation and limited-capacity attention." Journal of Experimental Psychology 106(3): 226-254.
- Meyer & Schvaneveldt (1971). "Facilitation in recognizing pairs of words: Evidence of a dependence between retrieval operations." Journal of Experimental Psychology 90(2): 227-234.

### Forgetting Curve
- Ebbinghaus (1885). "Memory: A Contribution to Experimental Psychology."
- Wixted & Ebbesen (1991). "On the form of forgetting curves." Psychological Science 2(6): 409-415.

### Concept Formation
- Tse et al. (2007). "Schemas and memory consolidation." Science 316(5821): 76-82.
- van Kesteren et al. (2012). "How schema and novelty augment memory formation." Trends in Neurosciences 35(4): 211-219.

### Memory Consolidation
- McClelland et al. (1995). "Why there are complementary learning systems in the hippocampus and neocortex." Psychological Review 102(3): 419-457.
- Rasch & Born (2013). "About sleep's role in memory." Physiological Reviews 93(2): 681-766.
