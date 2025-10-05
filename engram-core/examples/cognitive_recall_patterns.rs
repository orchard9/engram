//! Cognitive Recall Patterns Example
//!
//! Demonstrates the three recall modes and how to interpret recall results.
//! This example shows the API design and expected usage patterns.

#[cfg(feature = "hnsw_index")]
use engram_core::activation::{RecallConfig, RecallMetrics, RecallMode};
#[cfg(feature = "hnsw_index")]
use std::time::Duration;

#[cfg(not(feature = "hnsw_index"))]
fn main() {
    println!("This example requires the 'hnsw_index' feature to be enabled.");
    println!("Run with: cargo run --example cognitive_recall_patterns --features hnsw_index");
}

#[cfg(feature = "hnsw_index")]
fn main() {
    println!("=== Cognitive Recall Patterns ===\n");

    // Example 1: Configuration for different recall modes
    demonstrate_recall_modes();

    // Example 2: Metrics monitoring
    demonstrate_metrics();

    // Example 3: Result interpretation
    demonstrate_results();
}

#[cfg(feature = "hnsw_index")]
fn demonstrate_recall_modes() {
    println!("1. RECALL MODES\n");

    // Similarity mode: Fast, vector-based retrieval
    let similarity_config = RecallConfig {
        recall_mode: RecallMode::Similarity,
        time_budget: Duration::from_millis(5),
        min_confidence: 0.1,
        max_results: 10,
        enable_recency_boost: false,
        recency_boost_factor: 1.0,
        recency_window: Duration::from_secs(0),
    };
    println!("  Similarity Mode:");
    println!("    - Uses: HNSW vector similarity only");
    println!("    - Speed: Fastest (<5ms)");
    println!("    - Quality: Good for known queries");
    println!("    - Config: {:?}\n", similarity_config.recall_mode);

    // Spreading mode: Context-aware traversal
    let spreading_config = RecallConfig {
        recall_mode: RecallMode::Spreading,
        time_budget: Duration::from_millis(10),
        min_confidence: 0.2,
        max_results: 20,
        enable_recency_boost: true,
        recency_boost_factor: 1.3,
        recency_window: Duration::from_secs(3600),
    };
    println!("  Spreading Mode:");
    println!("    - Uses: Graph traversal + activation spreading");
    println!("    - Speed: Moderate (<10ms)");
    println!("    - Quality: Best for exploratory queries");
    println!("    - Config: {:?}\n", spreading_config.recall_mode);

    // Hybrid mode: Best of both with fallback
    let hybrid_config = RecallConfig {
        recall_mode: RecallMode::Hybrid,
        time_budget: Duration::from_millis(10),
        min_confidence: 0.15,
        max_results: 15,
        enable_recency_boost: true,
        recency_boost_factor: 1.2,
        recency_window: Duration::from_secs(1800),
    };
    println!("  Hybrid Mode (RECOMMENDED for production):");
    println!("    - Uses: Spreading with similarity fallback");
    println!("    - Speed: Adaptive (<10ms with timeout)");
    println!("    - Quality: Balanced performance/quality");
    println!("    - Config: {:?}\n", hybrid_config.recall_mode);
}

#[cfg(feature = "hnsw_index")]
fn demonstrate_metrics() {
    println!("\n2. METRICS MONITORING\n");

    let metrics = RecallMetrics::new();

    println!("  Available metrics for production monitoring:");
    println!("    - total_recalls: Total recall operations");
    println!("    - similarity_mode_count: Similarity-only recalls");
    println!("    - spreading_mode_count: Spreading-only recalls");
    println!("    - hybrid_mode_count: Hybrid mode recalls");
    println!("    - fallbacks_total: Number of fallbacks to similarity");
    println!("    - time_budget_violations: Recalls exceeding budget");
    println!("    - recall_activation_mass: Total activation across results");
    println!("    - seeding_failures: Vector seeding errors");
    println!("    - spreading_failures: Graph traversal errors\n");

    println!("  Example: Check fallback rate");
    println!("    fallback_rate = fallbacks_total / total_recalls");
    println!("    Current: {:.1}%\n", metrics.fallback_rate() * 100.0);
}

#[cfg(feature = "hnsw_index")]
fn demonstrate_results() {
    println!("\n3. RESULT INTERPRETATION\n");

    println!("  RankedMemory fields:");
    println!("    - episode: The recalled memory");
    println!("    - activation: Spreading activation level (0.0-1.0)");
    println!("    - confidence: Aggregate confidence score");
    println!("    - similarity: Vector similarity (if from seeding)");
    println!("    - recency_boost: Temporal boost applied");
    println!("    - rank_score: Final ranking score\n");

    println!("  Ranking formula:");
    println!("    rank_score = (activation * 0.4) +");
    println!("                 (confidence * 0.3) +");
    println!("                 (similarity * 0.2) +");
    println!("                 (recency_boost * 0.1)\n");

    println!("  Example interpretation:");
    println!("    Result 1:");
    println!("      activation: 0.85  (strong graph activation)");
    println!("      confidence: 0.92  (high path confidence)");
    println!("      similarity: 0.78  (good vector match)");
    println!("      recency_boost: 0.15  (recent memory)");
    println!("      rank_score: 0.847  → Top result\n");

    println!("    Result 2:");
    println!("      activation: 0.45  (weak graph activation)");
    println!("      confidence: 0.65  (medium path confidence)");
    println!("      similarity: 0.95  (excellent vector match)");
    println!("      recency_boost: 0.0  (old memory)");
    println!("      rank_score: 0.585  → Lower rank despite similarity\n");
}
