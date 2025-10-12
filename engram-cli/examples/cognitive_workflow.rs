//! Cognitive Workflow Scenario
//!
//! Demonstrates basic memory formation and recall operations.
//!
//! Run with: cargo run --example cognitive_workflow

use chrono::Utc;
use engram_core::{Confidence, Cue, EpisodeBuilder, MemoryStore};
use std::time::Instant;

fn main() {
    println!("=== Cognitive Workflow Scenario ===\n");

    let store = MemoryStore::new(10_000);

    // Phase 1: Memory Formation
    println!("Phase 1: Memory Formation");
    let formation_start = Instant::now();

    let learning_session = vec![
        (
            "python_basics",
            "Python is a high-level programming language",
        ),
        ("rust_basics", "Rust is a systems programming language"),
        (
            "golang_basics",
            "Go is a statically typed compiled language",
        ),
    ];

    for (id, content) in &learning_session {
        let episode = EpisodeBuilder::new()
            .id((*id).to_string())
            .when(Utc::now())
            .what((*content).to_string())
            .embedding([0.5; 768])
            .confidence(Confidence::from_raw(0.85))
            .build();

        store.store(episode);
    }

    let formation_elapsed = formation_start.elapsed();
    println!(
        "  Stored {} memories in {:?}",
        learning_session.len(),
        formation_elapsed
    );

    // Phase 2: Cue-Based Recall
    println!("\nPhase 2: Cue-Based Recall");

    let cue = Cue::embedding(
        "programming_cue".to_string(),
        [0.5; 768],
        Confidence::from_raw(0.9),
    );

    let recall_start = Instant::now();
    let results = store.recall(&cue);
    let recall_elapsed = recall_start.elapsed();

    println!(
        "  Found {} memories in {:?}",
        results.results.len(),
        recall_elapsed
    );

    for (episode, confidence) in results.results.iter().take(5) {
        println!("    - {} (confidence: {:.3})", episode.id, confidence.raw());
    }

    // Summary
    println!("\n=== Summary ===");
    println!(
        "Formation:  {:?} ({} memories)",
        formation_elapsed,
        learning_session.len()
    );
    println!(
        "Recall:     {:?} ({} results)",
        recall_elapsed,
        results.results.len()
    );
    println!("\n✓ Cognitive workflow completed!");
}
