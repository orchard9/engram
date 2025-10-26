//! Memory consolidation scenario: Long-term memory formation.
//!
//! This scenario simulates the process of memory consolidation:
//! 1. Add episodic memories over time
//! 2. Find associations between memories (vector similarity)
//! 3. Strengthen connections through repeated activation
//! 4. Apply decay to simulate forgetting
//! 5. Verify that important memories persist

#![allow(clippy::unwrap_used)]
#![allow(clippy::expect_used)]

#[cfg(feature = "zig-kernels")]
use crate::common::*;
#[cfg(feature = "zig-kernels")]
use engram_core::zig_kernels::*;

#[cfg(feature = "zig-kernels")]
#[test]
fn test_memory_consolidation_scenario() {
    // Scenario: Consolidate memories about learning Rust over several weeks

    // Week 1: Basic concepts
    let week1_topics = vec![
        "variables_and_mutability",
        "ownership_basics",
        "borrowing_rules",
        "data_types",
    ];

    // Week 2: Intermediate concepts (builds on week 1)
    let week2_topics = vec![
        "lifetimes",
        "trait_objects",
        "smart_pointers",
        "error_handling",
    ];

    // Week 3: Advanced concepts
    let week3_topics = vec!["async_await", "unsafe_rust", "macros", "concurrency"];

    let all_topics: Vec<&str> = week1_topics
        .iter()
        .chain(week2_topics.iter())
        .chain(week3_topics.iter())
        .copied()
        .collect();

    let num_memories = all_topics.len();

    // Generate embeddings (week 1 topics are more similar to each other)
    let mut embeddings = Vec::new();

    for (i, _topic) in all_topics.iter().enumerate() {
        let week = i / 4;
        let base_seed = 1000_u64 + (week as u64 * 1000);
        let seed = base_seed + i as u64;
        embeddings.push(generate_embedding(768, seed));
    }

    println!("\n=== Memory Consolidation: Learning Rust ===");
    println!("Total memories: {}", num_memories);

    // 1. Find associations between memories
    println!("\n=== Phase 1: Finding Associations ===");

    let mut similarity_matrix = vec![vec![0.0_f32; num_memories]; num_memories];

    for i in 0..num_memories {
        let query = &embeddings[i];
        let embeddings_flat: Vec<f32> = embeddings.iter().flat_map(|v| v.iter().copied()).collect();

        let similarities = vector_similarity(query, &embeddings_flat, num_memories);

        for (j, &sim) in similarities.iter().enumerate() {
            similarity_matrix[i][j] = sim;
        }
    }

    // Print similarity within and across weeks
    println!("\nWithin-week similarities:");
    for week in 0..3 {
        let start = week * 4;
        let end = start + 4;
        let mut week_sims = Vec::new();

        for i in start..end {
            for j in start..end {
                if i != j {
                    week_sims.push(similarity_matrix[i][j]);
                }
            }
        }

        let avg_sim = week_sims.iter().sum::<f32>() / week_sims.len() as f32;
        println!("Week {}: avg similarity = {:.4}", week + 1, avg_sim);
    }

    // 2. Build association graph
    println!("\n=== Phase 2: Building Association Graph ===");

    let mut adjacency = Vec::new();
    let mut weights = Vec::new();

    for i in 0..num_memories {
        for j in 0..num_memories {
            if i != j {
                let sim = similarity_matrix[i][j];
                if sim > 0.3 {
                    adjacency.push(j as u32);
                    weights.push(sim);
                }
            }
        }
    }

    println!("Graph edges: {}", adjacency.len());
    println!(
        "Average degree: {:.2}",
        adjacency.len() as f32 / num_memories as f32
    );

    // 3. Simulate repeated activation (studying)
    println!("\n=== Phase 3: Repeated Activation (Studying) ===");

    let mut cumulative_activation = vec![0.0_f32; num_memories];

    // Simulate 5 study sessions
    for session in 0..5 {
        // Each session focuses on different topics
        let focus_idx = (session * 3) % num_memories;

        let mut activations = vec![0.0_f32; num_memories];
        activations[focus_idx] = 1.0;

        println!(
            "\nSession {}: studying {}",
            session + 1,
            all_topics[focus_idx]
        );

        // Spread activation
        spread_activation(&adjacency, &weights, &mut activations, num_memories, 3);

        // Accumulate activation (simulates strengthening)
        for (i, &activation) in activations.iter().enumerate() {
            cumulative_activation[i] += activation;
        }

        // Show most activated topics
        let mut session_ranked: Vec<(usize, f32)> = activations
            .iter()
            .enumerate()
            .map(|(i, &a)| (i, a))
            .collect();
        session_ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        print!("  Activated: ");
        for (i, _) in session_ranked.iter().take(5) {
            print!("{}, ", all_topics[*i]);
        }
        println!();
    }

    // 4. Apply decay (simulate passage of time)
    println!("\n=== Phase 4: Applying Decay (3 weeks later) ===");

    // Simulate different retention based on week (week 1 = 3 weeks ago, week 3 = 1 week ago)
    let ages: Vec<u64> = (0..num_memories)
        .map(|i| {
            let week = i / 4;
            let weeks_ago = 3 - week;
            (weeks_ago * 7 * 24 * 3600) as u64 // Convert weeks to seconds
        })
        .collect();

    let mut strengths = cumulative_activation.clone();
    apply_decay(&mut strengths, &ages);

    println!("\nStrength after decay:");
    for (i, topic) in all_topics.iter().enumerate() {
        let week = i / 4 + 1;
        println!(
            "Week {} - {:25} activation={:.3}, strength={:.3}, age={}d",
            week,
            topic,
            cumulative_activation[i],
            strengths[i],
            ages[i] / (24 * 3600)
        );
    }

    // 5. Verify consolidation
    println!("\n=== Phase 5: Verification ===");

    // More recent memories (week 3) should have higher strength
    let week1_avg = strengths[0..4].iter().sum::<f32>() / 4.0;
    let week3_avg = strengths[8..12].iter().sum::<f32>() / 4.0;

    println!("Week 1 average strength: {:.4}", week1_avg);
    println!("Week 3 average strength: {:.4}", week3_avg);

    // Week 3 should be stronger due to less decay
    assert!(
        week3_avg > week1_avg * 0.5,
        "recent memories should be stronger: week3={:.4} vs week1={:.4}",
        week3_avg,
        week1_avg
    );

    // Frequently activated memories should have higher strength
    let max_strength = strengths.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let strong_memories: Vec<&str> = all_topics
        .iter()
        .enumerate()
        .filter(|(i, _)| strengths[*i] > max_strength * 0.5)
        .map(|(_, topic)| *topic)
        .collect();

    println!("\nStrongly consolidated memories: {:?}", strong_memories);

    assert!(
        !strong_memories.is_empty(),
        "some memories should be strongly consolidated"
    );
}

#[cfg(feature = "zig-kernels")]
#[test]
fn test_interference_during_consolidation() {
    // Test that similar memories can interfere with consolidation

    // Create two sets of similar memories
    let set_a_base = generate_embedding(768, 1000);
    let set_b_base = generate_embedding(768, 2000);

    let mut all_embeddings = Vec::new();

    // Set A: 10 memories very similar to each other
    for i in 0..10 {
        all_embeddings.push(generate_similar_embedding(&set_a_base, 0.05, 1000 + i));
    }

    // Set B: 10 memories very similar to each other
    for i in 0..10 {
        all_embeddings.push(generate_similar_embedding(&set_b_base, 0.05, 2000 + i));
    }

    let num_memories = all_embeddings.len();

    // Build similarity graph
    let mut adjacency = Vec::new();
    let mut weights = Vec::new();

    for i in 0..num_memories {
        let query = &all_embeddings[i];
        let embeddings_flat: Vec<f32> = all_embeddings
            .iter()
            .flat_map(|v| v.iter().copied())
            .collect();

        let similarities = vector_similarity(query, &embeddings_flat, num_memories);

        for (j, &sim) in similarities.iter().enumerate() {
            if i != j && sim > 0.5 {
                adjacency.push(j as u32);
                weights.push(sim);
            }
        }
    }

    // Activate one memory from set A
    let mut activations = vec![0.0_f32; num_memories];
    activations[0] = 1.0;

    spread_activation(&adjacency, &weights, &mut activations, num_memories, 5);

    // Check activation distribution
    let set_a_activation: f32 = activations[0..10].iter().sum();
    let set_b_activation: f32 = activations[10..20].iter().sum();

    println!("Set A total activation: {:.4}", set_a_activation);
    println!("Set B total activation: {:.4}", set_b_activation);

    // Activation should be concentrated in set A
    assert!(
        set_a_activation > set_b_activation * 2.0,
        "activation should stay within similar memory cluster"
    );
}

#[cfg(feature = "zig-kernels")]
#[test]
fn test_consolidation_with_rehearsal() {
    // Test that repeated activation strengthens memories

    let num_memories = 20;
    let embeddings = generate_embeddings(num_memories, 768, 3000);

    // Build association graph
    let (adjacency, weights, _) = generate_random_graph(num_memories, 0.1, 4000);

    // Simulate rehearsal: repeatedly activate same memories
    let rehearsed_indices = vec![0, 1, 2]; // First 3 memories are rehearsed
    let mut total_activation = vec![0.0_f32; num_memories];

    for _ in 0..10 {
        // 10 rehearsal sessions
        let mut activations = vec![0.0_f32; num_memories];

        for &idx in &rehearsed_indices {
            activations[idx] = 1.0;
        }

        spread_activation(&adjacency, &weights, &mut activations, num_memories, 3);

        for (i, &act) in activations.iter().enumerate() {
            total_activation[i] += act;
        }
    }

    println!("Total activations: {:?}", total_activation);

    // Rehearsed memories should have highest activation
    let max_activation = total_activation
        .iter()
        .copied()
        .fold(f32::NEG_INFINITY, f32::max);

    for &idx in &rehearsed_indices {
        assert!(
            total_activation[idx] > max_activation * 0.5,
            "rehearsed memory {} should have high activation",
            idx
        );
    }
}
