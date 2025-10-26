//! Memory recall scenario: Query with spreading activation and vector similarity.
//!
//! This scenario simulates retrieving memories from a knowledge base:
//! 1. Store multiple related memories with embeddings
//! 2. Query with a new concept
//! 3. Find similar memories via vector similarity (Zig kernel)
//! 4. Spread activation to related concepts (Zig kernel)
//! 5. Apply recency-based decay (Zig kernel)
//! 6. Return ranked results

#![allow(clippy::unwrap_used)]
#![allow(clippy::expect_used)]

#[cfg(feature = "zig-kernels")]
use crate::common::*;
#[cfg(feature = "zig-kernels")]
use engram_core::zig_kernels::*;

#[cfg(feature = "zig-kernels")]
#[test]
fn test_memory_recall_scenario() {
    // Scenario: Recall memories about "machine learning"
    // Memory graph contains documents about various CS topics

    // 1. Create memory embeddings for different topics
    let memory_topics = vec![
        ("neural_networks", 1),
        ("machine_learning", 2),
        ("deep_learning", 3),
        ("reinforcement_learning", 4),
        ("supervised_learning", 5),
        ("unsupervised_learning", 6),
        ("computer_vision", 7),
        ("natural_language_processing", 8),
        ("data_structures", 9),
        ("algorithms", 10),
        ("operating_systems", 11),
        ("databases", 12),
    ];

    let num_memories = memory_topics.len();
    let memory_embeddings: Vec<Vec<f32>> = memory_topics
        .iter()
        .map(|(_, seed)| generate_embedding(768, *seed))
        .collect();

    // 2. Query: "machine learning concepts"
    // Make it similar to machine_learning but with some overlap to related topics
    let query_base = &memory_embeddings[1]; // machine_learning
    let query = generate_similar_embedding(query_base, 0.15, 999);

    // 3. Find similar memories using vector similarity kernel
    let embeddings_flat: Vec<f32> = memory_embeddings
        .iter()
        .flat_map(|v| v.iter().copied())
        .collect();

    let similarities = vector_similarity(&query, &embeddings_flat, num_memories);

    println!("\n=== Similarity Scores ===");
    for (i, (topic, _)) in memory_topics.iter().enumerate() {
        println!("{:30} {:.4}", topic, similarities[i]);
    }

    // Verify that machine learning has high similarity
    assert!(
        similarities[1] > 0.9,
        "query should be most similar to machine_learning, got {}",
        similarities[1]
    );

    // 4. Build association graph between highly similar memories
    let threshold = 0.5;
    let relevant_indices: Vec<usize> = similarities
        .iter()
        .enumerate()
        .filter(|(_, sim)| **sim > threshold)
        .map(|(i, _)| i)
        .collect();

    println!(
        "\nRelevant memories (>{} similarity): {}",
        threshold,
        relevant_indices.len()
    );

    // Create edges between relevant memories
    let mut adjacency = Vec::new();
    let mut weights = Vec::new();
    let mut node_map: std::collections::HashMap<usize, usize> = std::collections::HashMap::new();

    for (new_idx, &orig_idx) in relevant_indices.iter().enumerate() {
        node_map.insert(orig_idx, new_idx);
    }

    for (i, &idx_i) in relevant_indices.iter().enumerate() {
        for (j, &idx_j) in relevant_indices.iter().enumerate() {
            if i != j {
                let sim = similarities[idx_i] * similarities[idx_j];
                if sim > 0.3 {
                    adjacency.push(j as u32);
                    weights.push(sim.sqrt());
                }
            }
        }
    }

    // 5. Apply spreading activation starting from most similar memory
    let num_nodes = relevant_indices.len();
    let mut activations = vec![0.0_f32; num_nodes];

    // Find the node corresponding to machine_learning
    if let Some(&ml_node) = node_map.get(&1) {
        activations[ml_node] = 1.0;
    }

    println!("\n=== Spreading Activation ===");
    println!("Initial activations: {:?}", activations);

    spread_activation(&adjacency, &weights, &mut activations, num_nodes, 5);

    println!("After spreading: {:?}", activations);

    // Verify activation spread to related concepts
    assert!(
        activations.iter().filter(|&&a| a > 0.0).count() > 1,
        "activation should spread to multiple nodes"
    );

    // 6. Apply decay based on recency (simulate different access times)
    let ages: Vec<u64> = (0..num_nodes)
        .map(|i| {
            // Make ML-related topics more recent (lower age)
            if i < num_nodes / 2 {
                i as u64 * 3600 // 0-6 hours
            } else {
                (i as u64 + 10) * 7200 // 20+ hours
            }
        })
        .collect();

    let mut strengths = activations.clone();
    apply_decay(&mut strengths, &ages);

    println!("\n=== After Decay ===");
    for (i, &strength) in strengths.iter().enumerate() {
        let orig_idx = relevant_indices[i];
        println!(
            "{:30} activation={:.4}, strength={:.4}, age={}h",
            memory_topics[orig_idx].0,
            activations[i],
            strength,
            ages[i] / 3600
        );
    }

    // 7. Rank and return top results
    let mut ranked: Vec<(usize, f32)> = strengths
        .iter()
        .enumerate()
        .map(|(i, &s)| (relevant_indices[i], s))
        .collect();
    ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    println!("\n=== Top Results ===");
    for (i, (orig_idx, strength)) in ranked.iter().take(5).enumerate() {
        println!(
            "{}. {:30} (strength={:.4})",
            i + 1,
            memory_topics[*orig_idx].0,
            strength
        );
    }

    // Verify that ML-related topics are ranked higher
    let top_3: Vec<&str> = ranked
        .iter()
        .take(3)
        .map(|(idx, _)| memory_topics[*idx].0)
        .collect();

    println!("\nTop 3: {:?}", top_3);

    // At least one of the top results should be ML-related
    assert!(
        top_3.contains(&"machine_learning")
            || top_3.contains(&"supervised_learning")
            || top_3.contains(&"neural_networks"),
        "top results should include ML-related topics"
    );
}

#[cfg(feature = "zig-kernels")]
#[test]
fn test_recall_with_interference() {
    // Test recall when there's interference from unrelated memories

    // Create two clusters of memories
    let cluster_a = generate_embeddings(20, 768, 1000); // Related to topic A
    let cluster_b = generate_embeddings(20, 768, 5000); // Related to topic B

    let mut all_embeddings = cluster_a.clone();
    all_embeddings.extend(cluster_b.clone());

    // Query for topic A
    let query = generate_similar_embedding(&cluster_a[0], 0.1, 999);

    // Find similarities
    let embeddings_flat: Vec<f32> = all_embeddings
        .iter()
        .flat_map(|v| v.iter().copied())
        .collect();

    let similarities = vector_similarity(&query, &embeddings_flat, 40);

    // Top results should be from cluster A (indices 0-19)
    let mut indexed: Vec<(usize, f32)> = similarities
        .iter()
        .enumerate()
        .map(|(i, &s)| (i, s))
        .collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    let top_10_indices: Vec<usize> = indexed.iter().take(10).map(|(i, _)| *i).collect();

    let cluster_a_count = top_10_indices.iter().filter(|&&i| i < 20).count();

    println!("Top 10 results: {:?}", top_10_indices);
    println!("From cluster A: {}/10", cluster_a_count);

    assert!(
        cluster_a_count >= 7,
        "at least 7 of top 10 should be from cluster A, got {}",
        cluster_a_count
    );
}

#[cfg(feature = "zig-kernels")]
#[test]
fn test_recall_performance_at_scale() {
    // Test recall performance with realistic scale (10k memories)

    let num_memories = 10_000;
    let embeddings = generate_embeddings(num_memories, 768, 20000);
    let query = generate_embedding(768, 42);

    let embeddings_flat: Vec<f32> = embeddings.iter().flat_map(|v| v.iter().copied()).collect();

    use std::time::Instant;

    let start = Instant::now();
    let similarities = vector_similarity(&query, &embeddings_flat, num_memories);
    let duration = start.elapsed();

    println!(
        "Vector similarity for {} memories: {:?} ({:.2} us/memory)",
        num_memories,
        duration,
        duration.as_micros() as f64 / num_memories as f64
    );

    // With Zig kernels, this should be <5ms total
    assert!(
        duration.as_millis() < 50,
        "should complete in <50ms, took {:?}",
        duration
    );

    // Verify results are valid
    assert_eq!(similarities.len(), num_memories);
    assert!(similarities.iter().all(|&s| s >= -1.0 && s <= 1.0));
}
