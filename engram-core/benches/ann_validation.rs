//! Validation tests for ANN implementations
//!
//! Verifies that Engram achieves ≥90% recall@10 compared to ground truth.
//! Only runs when the `ann_benchmarks` feature is enabled.

#![cfg(feature = "ann_benchmarks")]

mod support;

use support::ann_common::AnnIndex;
use support::datasets::DatasetLoader;
use support::engram_ann::EngramOptimizedAnnIndex;
use support::faiss_ann::FaissAnnIndex;
use support::annoy_ann::AnnoyAnnIndex;

/// Compute recall@k: fraction of ground truth neighbors found in results
fn compute_recall(results: &[usize], ground_truth: &[usize]) -> f32 {
    if ground_truth.is_empty() {
        return 0.0;
    }

    let found = results
        .iter()
        .filter(|&idx| ground_truth.contains(idx))
        .count();

    found as f32 / ground_truth.len() as f32
}

#[test]
fn validate_engram_recall_90_percent() {
    // Load test dataset (10K vectors for validation)
    let dataset = DatasetLoader::load_sift1m_mock();

    println!("Dataset loaded: {} vectors, {} queries",
        dataset.vectors.len(), dataset.queries.len());

    // Build ground truth using FAISS Flat (exact search)
    println!("Building ground truth with FAISS Flat...");
    let mut ground_truth_index = FaissAnnIndex::new_flat(768)
        .expect("Failed to create FAISS Flat index");
    ground_truth_index.build(&dataset.vectors)
        .expect("Failed to build ground truth index");

    let ground_truth: Vec<Vec<usize>> = dataset
        .queries
        .iter()
        .map(|query| {
            ground_truth_index
                .search(query, 10)
                .into_iter()
                .map(|(idx, _)| idx)
                .collect()
        })
        .collect();

    // Build Engram index
    println!("Building Engram HNSW index...");
    let mut engram = EngramOptimizedAnnIndex::new();
    engram.build(&dataset.vectors)
        .expect("Failed to build Engram index");

    // Compute recall for Engram
    println!("Computing recall for Engram...");
    let mut total_recall = 0.0;
    for (query, truth) in dataset.queries.iter().zip(ground_truth.iter()) {
        let results = engram.search(query, 10);
        let indices: Vec<usize> = results.into_iter().map(|(idx, _)| idx).collect();
        let recall = compute_recall(&indices, truth);
        total_recall += recall;
    }

    let avg_recall = total_recall / dataset.queries.len() as f32;

    println!("Engram Recall@10: {:.2}%", avg_recall * 100.0);

    // Assert that Engram achieves ≥90% recall
    assert!(
        avg_recall >= 0.90,
        "Engram recall ({:.2}%) below 90% target",
        avg_recall * 100.0
    );
}

#[test]
fn compare_all_implementations() {
    // Use a smaller dataset for comparison test
    let dataset = DatasetLoader::generate_synthetic(1000, 100);

    println!("Synthetic dataset: {} vectors, {} queries",
        dataset.vectors.len(), dataset.queries.len());

    // Build all indices
    println!("Building Engram index...");
    let mut engram = EngramOptimizedAnnIndex::new();
    engram.build(&dataset.vectors)
        .expect("Failed to build Engram index");

    println!("Building FAISS-HNSW index...");
    let mut faiss = FaissAnnIndex::new_hnsw(768, 16)
        .expect("Failed to create FAISS index");
    faiss.build(&dataset.vectors)
        .expect("Failed to build FAISS index");

    println!("Building Annoy index...");
    let mut annoy = AnnoyAnnIndex::new(768, 10)
        .expect("Failed to create Annoy index");
    annoy.build(&dataset.vectors)
        .expect("Failed to build Annoy index");

    // Compare results for first query
    let query = &dataset.queries[0];

    let engram_results = engram.search(query, 10);
    let faiss_results = faiss.search(query, 10);
    let annoy_results = annoy.search(query, 10);

    println!("\nResults for first query:");
    println!("Engram: {:?}", engram_results.iter().map(|(i, _)| i).collect::<Vec<_>>());
    println!("FAISS:  {:?}", faiss_results.iter().map(|(i, _)| i).collect::<Vec<_>>());
    println!("Annoy:  {:?}", annoy_results.iter().map(|(i, _)| i).collect::<Vec<_>>());

    // All should return requested number of results
    assert_eq!(engram_results.len(), 10, "Engram should return 10 results");
    assert_eq!(faiss_results.len(), 10, "FAISS should return 10 results");
    assert_eq!(annoy_results.len(), 10, "Annoy should return 10 results");

    // Results should be sorted by similarity (highest first)
    for results in [&engram_results, &faiss_results, &annoy_results] {
        for i in 0..results.len()-1 {
            assert!(
                results[i].1 >= results[i+1].1,
                "Results should be sorted by similarity"
            );
        }
    }

    println!("\nMemory usage:");
    println!("Engram: {} bytes", engram.memory_usage());
    println!("FAISS:  {} bytes", faiss.memory_usage());
    println!("Annoy:  {} bytes", annoy.memory_usage());
}

#[test]
fn validate_recall_computation() {
    let results = vec![0, 1, 3, 5, 7];
    let ground_truth = vec![0, 1, 2, 3, 4];

    let recall = compute_recall(&results, &ground_truth);

    // 3 out of 5 correct (0, 1, 3)
    assert!((recall - 0.6).abs() < 0.01);
}

#[test]
fn validate_perfect_recall() {
    let results = vec![0, 1, 2, 3, 4];
    let ground_truth = vec![0, 1, 2, 3, 4];

    let recall = compute_recall(&results, &ground_truth);

    assert!((recall - 1.0).abs() < 0.01);
}

#[test]
fn validate_zero_recall() {
    let results = vec![5, 6, 7, 8, 9];
    let ground_truth = vec![0, 1, 2, 3, 4];

    let recall = compute_recall(&results, &ground_truth);

    assert!(recall.abs() < 0.01);
}
