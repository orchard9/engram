//! Pattern completion scenario: Associative recall from partial cues.
//!
//! This scenario demonstrates pattern completion through spreading activation:
//! 1. Create a structured pattern (sequence, hierarchy, or network)
//! 2. Present a partial cue (subset of the pattern)
//! 3. Use spreading activation to complete the pattern
//! 4. Verify that missing elements are recalled
//!
//! Based on hippocampal pattern completion research (McClelland et al., 1995)

#![allow(clippy::unwrap_used)]
#![allow(clippy::expect_used)]

#[cfg(feature = "zig-kernels")]
use crate::common::*;
#[cfg(feature = "zig-kernels")]
use engram_core::zig_kernels::*;

#[cfg(feature = "zig-kernels")]
#[test]
fn test_sequence_completion() {
    // Scenario: Complete a learned sequence from partial cue
    // Pattern: A -> B -> C -> D -> E
    // Cue: A
    // Expected: Activation spreads to B, C, D, E

    let sequence = vec!["A", "B", "C", "D", "E"];
    let num_nodes = sequence.len();

    // Create embeddings where sequential items are similar
    let mut embeddings = Vec::new();
    let base = generate_embedding(768, 5000);

    for i in 0..num_nodes {
        // Each item is similar to its neighbors
        let noise = 0.1 + (i as f32 * 0.05);
        embeddings.push(generate_similar_embedding(&base, noise, 5000 + i as u64));
    }

    // Build chain graph: A -> B -> C -> D -> E
    let mut adjacency = Vec::new();
    let mut weights = Vec::new();

    for i in 0..(num_nodes - 1) {
        adjacency.push((i + 1) as u32);
        weights.push(0.9); // Strong connections in sequence
    }

    println!("\n=== Sequence Pattern Completion ===");
    println!("Learned sequence: {:?}", sequence);
    println!("Cue: A");

    // Present cue: activate A
    let mut activations = vec![0.0_f32; num_nodes];
    activations[0] = 1.0;

    println!("\nInitial activation: {:?}", activations);

    // Spread activation to complete pattern
    spread_activation(&adjacency, &weights, &mut activations, num_nodes, 10);

    println!("After spreading: {:?}", activations);

    // Print results
    println!("\nCompleted pattern:");
    for (i, item) in sequence.iter().enumerate() {
        println!("  {} -> activation = {:.4}", item, activations[i]);
    }

    // Verify pattern completion
    assert!(activations[0] > 0.0, "cue A should remain active");

    for i in 1..num_nodes {
        assert!(
            activations[i] > 0.0,
            "item {} should be activated by spreading",
            sequence[i]
        );
    }

    // Activation should generally decrease along the chain
    assert!(
        activations[0] >= activations[1],
        "activation should be highest at cue"
    );
}

#[cfg(feature = "zig-kernels")]
#[test]
fn test_partial_pattern_completion() {
    // Scenario: Complete a pattern from multiple partial cues
    // Pattern: Hierarchical tree
    //          Root
    //         /    \
    //        A      B
    //       / \    / \
    //      A1 A2  B1 B2
    //
    // Cue: A1 and B1 (leaf nodes)
    // Expected: Activation spreads up to Root and across to siblings

    let nodes = vec!["Root", "A", "B", "A1", "A2", "B1", "B2"];
    let num_nodes = nodes.len();

    // Build tree edges (bidirectional for completion)
    let edges = vec![
        (0, 1, 0.8), // Root <-> A
        (1, 0, 0.8),
        (0, 2, 0.8), // Root <-> B
        (2, 0, 0.8),
        (1, 3, 0.9), // A <-> A1
        (3, 1, 0.9),
        (1, 4, 0.9), // A <-> A2
        (4, 1, 0.9),
        (2, 5, 0.9), // B <-> B1
        (5, 2, 0.9),
        (2, 6, 0.9), // B <-> B2
        (6, 2, 0.9),
    ];

    let mut adjacency = Vec::new();
    let mut weights = Vec::new();

    for (from, to, weight) in edges {
        adjacency.push(to);
        weights.push(weight);
    }

    println!("\n=== Hierarchical Pattern Completion ===");
    println!("Tree structure:");
    println!("          Root");
    println!("         /    \\");
    println!("        A      B");
    println!("       / \\    / \\");
    println!("      A1 A2  B1 B2");
    println!("\nPartial cues: A1, B1");

    // Present partial cues
    let mut activations = vec![0.0_f32; num_nodes];
    activations[3] = 1.0; // A1
    activations[5] = 1.0; // B1

    println!("\nInitial activation: {:?}", activations);

    // Spread activation
    spread_activation(&adjacency, &weights, &mut activations, num_nodes, 8);

    println!("After spreading: {:?}", activations);

    // Print completed pattern
    println!("\nCompleted pattern:");
    for (i, node) in nodes.iter().enumerate() {
        println!("  {:4} -> activation = {:.4}", node, activations[i]);
    }

    // Verify completion
    assert!(activations[0] > 0.0, "Root should be activated");
    assert!(activations[1] > 0.0, "A should be activated");
    assert!(activations[2] > 0.0, "B should be activated");
    assert!(
        activations[4] > 0.0,
        "A2 should be activated (sibling of A1)"
    );
    assert!(
        activations[6] > 0.0,
        "B2 should be activated (sibling of B1)"
    );

    println!("\nPattern completion successful!");
}

#[cfg(feature = "zig-kernels")]
#[test]
fn test_associative_recall_with_decay() {
    // Scenario: Recall associated memories with decay
    // Pattern: Word associations (king-queen, man-woman, etc.)
    // Simulate that some associations are stronger (more recent/frequent)

    let associations = vec![
        ("king", "queen", 0.95, 0),          // Recent, strong
        ("man", "woman", 0.90, 3600),        // 1 hour ago
        ("boy", "girl", 0.85, 7200),         // 2 hours ago
        ("prince", "princess", 0.80, 86400), // 1 day ago
    ];

    let num_pairs = associations.len();
    let num_nodes = num_pairs * 2;

    // Map each word to a node index
    let mut word_to_idx = std::collections::HashMap::new();
    let mut idx_to_word = Vec::new();

    for (i, (w1, w2, _, _)) in associations.iter().enumerate() {
        if !word_to_idx.contains_key(w1) {
            let idx = idx_to_word.len();
            word_to_idx.insert(*w1, idx);
            idx_to_word.push(*w1);
        }
        if !word_to_idx.contains_key(w2) {
            let idx = idx_to_word.len();
            word_to_idx.insert(*w2, idx);
            idx_to_word.push(*w2);
        }
    }

    let num_nodes = idx_to_word.len();

    // Build association graph
    let mut adjacency = Vec::new();
    let mut weights = Vec::new();

    for (w1, w2, strength, _) in &associations {
        let idx1 = word_to_idx[w1];
        let idx2 = word_to_idx[w2];

        // Bidirectional association
        adjacency.push(idx2 as u32);
        weights.push(*strength);
        adjacency.push(idx1 as u32);
        weights.push(*strength);
    }

    println!("\n=== Associative Recall with Decay ===");
    println!("Associations:");
    for (w1, w2, strength, age) in &associations {
        println!(
            "  {} <-> {} (strength={:.2}, age={}s)",
            w1, w2, strength, age
        );
    }

    // Test 1: Cue with "king", expect "queen"
    println!("\n--- Test 1: Cue = 'king' ---");

    let mut activations = vec![0.0_f32; num_nodes];
    activations[word_to_idx["king"]] = 1.0;

    spread_activation(&adjacency, &weights, &mut activations, num_nodes, 3);

    println!("Activations before decay:");
    for (i, word) in idx_to_word.iter().enumerate() {
        if activations[i] > 0.01 {
            println!("  {} = {:.4}", word, activations[i]);
        }
    }

    // Apply decay based on association age
    let mut ages = vec![86400_u64; num_nodes]; // Default: old
    for (w1, w2, _, age) in &associations {
        ages[word_to_idx[w1]] = *age;
        ages[word_to_idx[w2]] = *age;
    }

    let mut strengths = activations.clone();
    apply_decay(&mut strengths, &ages);

    println!("\nStrengths after decay:");
    for (i, word) in idx_to_word.iter().enumerate() {
        if strengths[i] > 0.01 {
            println!("  {} = {:.4}", word, strengths[i]);
        }
    }

    // "queen" should be strongly activated (recent association)
    assert!(
        strengths[word_to_idx["queen"]] > 0.5,
        "queen should be strongly recalled"
    );

    // Test 2: Partial cue pattern completion
    println!("\n--- Test 2: Multiple cues ---");

    let mut activations2 = vec![0.0_f32; num_nodes];
    activations2[word_to_idx["man"]] = 1.0;
    activations2[word_to_idx["boy"]] = 1.0;

    spread_activation(&adjacency, &weights, &mut activations2, num_nodes, 3);

    let mut strengths2 = activations2.clone();
    apply_decay(&mut strengths2, &ages);

    println!("Completed associations:");
    for (i, word) in idx_to_word.iter().enumerate() {
        if strengths2[i] > 0.01 {
            println!("  {} = {:.4}", word, strengths2[i]);
        }
    }

    // Associated words should be activated
    assert!(
        strengths2[word_to_idx["woman"]] > 0.0,
        "woman should be recalled"
    );
    assert!(
        strengths2[word_to_idx["girl"]] > 0.0,
        "girl should be recalled"
    );
}

#[cfg(feature = "zig-kernels")]
#[test]
fn test_noisy_cue_completion() {
    // Scenario: Complete pattern from noisy/degraded cue
    // Simulate partial or corrupted memory retrieval

    let num_memories = 50;
    let embeddings = generate_embeddings(num_memories, 768, 6000);

    // Create a strongly connected cluster (0-9) representing a coherent memory
    let mut adjacency = Vec::new();
    let mut weights = Vec::new();

    // Strong connections within cluster
    for i in 0..10 {
        for j in 0..10 {
            if i != j {
                adjacency.push(j);
                weights.push(0.85);
            }
        }
    }

    // Weak connections to rest of memories
    for i in 0..10 {
        for j in 10..num_memories {
            if i % 3 == 0 {
                // Sparse connections
                adjacency.push(j as u32);
                weights.push(0.3);
            }
        }
    }

    println!("\n=== Noisy Cue Pattern Completion ===");

    // Present noisy cue: weak activation of cluster memories
    let mut activations = vec![0.0_f32; num_memories];

    // Simulate noisy cue: only partial activation of cluster
    activations[0] = 0.8; // Strong
    activations[1] = 0.6; // Medium
    activations[2] = 0.4; // Weak
    // Others in cluster start at 0

    println!("Noisy cue: {:?}", &activations[0..10]);

    spread_activation(&adjacency, &weights, &mut activations, num_memories, 10);

    println!("After completion: {:?}", &activations[0..10]);

    // Verify pattern completion within cluster
    let cluster_activation: f32 = activations[0..10].iter().sum();
    let other_activation: f32 = activations[10..].iter().sum();

    println!("\nCluster activation: {:.4}", cluster_activation);
    println!("Other activation: {:.4}", other_activation);

    assert!(
        cluster_activation > other_activation * 2.0,
        "cluster should be preferentially completed"
    );

    // All cluster members should have some activation
    for i in 0..10 {
        assert!(
            activations[i] > 0.0,
            "cluster member {} should be activated",
            i
        );
    }
}

#[cfg(feature = "zig-kernels")]
#[test]
fn test_context_dependent_completion() {
    // Scenario: Pattern completion depends on context
    // Word "bank" can mean financial institution or river edge
    // Context determines which pattern is completed

    let financial_words = vec!["bank", "money", "loan", "account", "deposit"];
    let river_words = vec!["bank", "river", "water", "shore", "stream"];

    // Create two separate contexts with "bank" as shared node
    let num_nodes = 9; // bank + 4 financial + 4 river words
    let mut idx_to_word = vec!["bank"]; // Node 0
    idx_to_word.extend(&financial_words[1..]);
    idx_to_word.extend(&river_words[1..]);

    let mut adjacency = Vec::new();
    let mut weights = Vec::new();

    // Financial context: bank (0) connected to financial words (1-4)
    for i in 1..5 {
        adjacency.push(i);
        weights.push(0.9);
        adjacency.push(0); // Bidirectional
        weights.push(0.9);
    }

    // River context: bank (0) connected to river words (5-8)
    for i in 5..9 {
        adjacency.push(i);
        weights.push(0.9);
        adjacency.push(0); // Bidirectional
        weights.push(0.9);
    }

    println!("\n=== Context-Dependent Pattern Completion ===");

    // Test 1: Financial context cue
    println!("\n--- Context: Financial ---");
    println!("Cue: money + bank");

    let mut activations = vec![0.0_f32; num_nodes];
    activations[0] = 0.5; // bank (ambiguous)
    activations[1] = 1.0; // money (context)

    spread_activation(&adjacency, &weights, &mut activations, num_nodes, 5);

    println!("Completed pattern:");
    for (i, word) in idx_to_word.iter().enumerate() {
        if activations[i] > 0.01 {
            println!("  {} = {:.4}", word, activations[i]);
        }
    }

    let financial_activation: f32 = activations[1..5].iter().sum();
    let river_activation: f32 = activations[5..9].iter().sum();

    assert!(
        financial_activation > river_activation,
        "financial context should be preferred"
    );

    // Test 2: River context cue
    println!("\n--- Context: River ---");
    println!("Cue: water + bank");

    let mut activations2 = vec![0.0_f32; num_nodes];
    activations2[0] = 0.5; // bank (ambiguous)
    activations2[6] = 1.0; // water (context)

    spread_activation(&adjacency, &weights, &mut activations2, num_nodes, 5);

    println!("Completed pattern:");
    for (i, word) in idx_to_word.iter().enumerate() {
        if activations2[i] > 0.01 {
            println!("  {} = {:.4}", word, activations2[i]);
        }
    }

    let financial_activation2: f32 = activations2[1..5].iter().sum();
    let river_activation2: f32 = activations2[5..9].iter().sum();

    assert!(
        river_activation2 > financial_activation2,
        "river context should be preferred"
    );

    println!("\nContext-dependent completion successful!");
}
