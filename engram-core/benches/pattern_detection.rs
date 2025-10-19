//! Benchmarks for pattern detection performance.
//!
//! Performance targets:
//! - 100 episodes: <100ms
//! - 1000 episodes: <1s

#![allow(missing_docs)]

use chrono::{Duration, Utc};
use criterion::{BenchmarkId, Criterion, Throughput, black_box, criterion_group, criterion_main};
use engram_core::consolidation::{PatternDetectionConfig, PatternDetector, StatisticalFilter};
use engram_core::{Confidence, Episode};

/// Helper to create test episodes with random-like embeddings
fn create_test_episode(id: usize, cluster_id: usize, variation: f32) -> Episode {
    let base_value = (cluster_id as f32) * 0.2;
    let mut embedding = [base_value; 768];

    // Add variation to make episodes similar but not identical
    for (i, item) in embedding.iter_mut().enumerate() {
        *item += ((i + id) as f32 * variation) % 0.1;
    }

    Episode::new(
        format!("episode_{id}"),
        Utc::now() + Duration::seconds(id as i64 * 60),
        format!("test content {id}"),
        embedding,
        Confidence::exact(0.9),
    )
}

/// Create a collection of episodes with specified clusters
fn create_episode_collection(num_episodes: usize, num_clusters: usize) -> Vec<Episode> {
    let episodes_per_cluster = num_episodes / num_clusters;
    let mut episodes = Vec::with_capacity(num_episodes);

    for cluster_id in 0..num_clusters {
        for i in 0..episodes_per_cluster {
            let episode = create_test_episode(
                cluster_id * episodes_per_cluster + i,
                cluster_id,
                0.01, // Small variation within cluster
            );
            episodes.push(episode);
        }
    }

    // Add remaining episodes if num_episodes not evenly divisible
    for i in (num_clusters * episodes_per_cluster)..num_episodes {
        let cluster_id = i % num_clusters;
        episodes.push(create_test_episode(i, cluster_id, 0.01));
    }

    episodes
}

fn bench_pattern_detection(c: &mut Criterion) {
    let mut group = c.benchmark_group("pattern_detection");

    // Test different episode counts
    for num_episodes in &[10, 50, 100, 250, 500, 1000] {
        let config = PatternDetectionConfig::default();
        let detector = PatternDetector::new(config);

        // Create episodes with 5 clusters
        let episodes = create_episode_collection(*num_episodes, 5);

        group.throughput(Throughput::Elements(*num_episodes as u64));
        group.bench_with_input(
            BenchmarkId::new("detect_patterns", num_episodes),
            &episodes,
            |b, eps| {
                b.iter(|| {
                    let patterns = detector.detect_patterns(black_box(eps));
                    black_box(patterns);
                });
            },
        );
    }

    group.finish();
}

fn bench_clustering_algorithms(c: &mut Criterion) {
    let mut group = c.benchmark_group("clustering");

    let config = PatternDetectionConfig {
        min_cluster_size: 3,
        similarity_threshold: 0.8,
        max_patterns: 100,
    };
    let detector = PatternDetector::new(config);

    // Test with varying numbers of clusters
    for num_clusters in &[2, 5, 10, 20] {
        let episodes = create_episode_collection(100, *num_clusters);

        group.bench_with_input(
            BenchmarkId::new("clusters", num_clusters),
            &episodes,
            |b, eps| {
                b.iter(|| {
                    let patterns = detector.detect_patterns(black_box(eps));
                    black_box(patterns);
                });
            },
        );
    }

    group.finish();
}

fn bench_statistical_filtering(c: &mut Criterion) {
    let mut group = c.benchmark_group("statistical_filtering");

    let config = PatternDetectionConfig::default();
    let detector = PatternDetector::new(config);
    let filter = StatisticalFilter::default();

    // Pre-generate episodes and patterns
    let episodes = create_episode_collection(100, 5);
    let patterns = detector.detect_patterns(&episodes);

    group.bench_function("filter_patterns", |b| {
        b.iter(|| {
            let filtered = filter.filter_patterns(black_box(&patterns));
            black_box(filtered);
        });
    });

    group.finish();
}

fn bench_similarity_thresholds(c: &mut Criterion) {
    let mut group = c.benchmark_group("similarity_threshold");

    let episodes = create_episode_collection(100, 5);

    for threshold in &[0.5, 0.6, 0.7, 0.8, 0.9] {
        let config = PatternDetectionConfig {
            min_cluster_size: 3,
            similarity_threshold: *threshold,
            max_patterns: 100,
        };
        let detector = PatternDetector::new(config);

        group.bench_with_input(
            BenchmarkId::new("threshold", format!("{threshold:.1}")),
            &episodes,
            |b, eps| {
                b.iter(|| {
                    let patterns = detector.detect_patterns(black_box(eps));
                    black_box(patterns);
                });
            },
        );
    }

    group.finish();
}

fn bench_pattern_merging(c: &mut Criterion) {
    let mut group = c.benchmark_group("pattern_merging");

    // Create many small similar clusters that will merge
    let config = PatternDetectionConfig {
        min_cluster_size: 3,
        similarity_threshold: 0.75,
        max_patterns: 100,
    };
    let detector = PatternDetector::new(config);

    let episodes = create_episode_collection(200, 10);

    group.bench_function("merge_similar_patterns", |b| {
        b.iter(|| {
            let patterns = detector.detect_patterns(black_box(&episodes));
            black_box(patterns);
        });
    });

    group.finish();
}

fn bench_end_to_end_pipeline(c: &mut Criterion) {
    let mut group = c.benchmark_group("end_to_end");

    let config = PatternDetectionConfig::default();
    let detector = PatternDetector::new(config);
    let filter = StatisticalFilter::default();

    for num_episodes in &[50, 100, 500] {
        let episodes = create_episode_collection(*num_episodes, 5);

        group.throughput(Throughput::Elements(*num_episodes as u64));
        group.bench_with_input(
            BenchmarkId::new("detect_and_filter", num_episodes),
            &episodes,
            |b, eps| {
                b.iter(|| {
                    let patterns = detector.detect_patterns(black_box(eps));
                    let filtered = filter.filter_patterns(black_box(&patterns));
                    black_box(filtered);
                });
            },
        );
    }

    group.finish();
}

fn bench_large_scale(c: &mut Criterion) {
    let mut group = c.benchmark_group("large_scale");
    // Reduce sample size for large benchmarks
    group.sample_size(10);

    let config = PatternDetectionConfig::default();
    let detector = PatternDetector::new(config);

    // Test performance target: <1s for 1000 episodes
    let episodes_1k = create_episode_collection(1000, 10);

    group.throughput(Throughput::Elements(1000));
    group.bench_function("1000_episodes", |b| {
        b.iter(|| {
            let patterns = detector.detect_patterns(black_box(&episodes_1k));
            black_box(patterns);
        });
    });

    // Test 5000 episodes for stress testing
    let episodes_5k = create_episode_collection(5000, 20);

    group.throughput(Throughput::Elements(5000));
    group.bench_function("5000_episodes", |b| {
        b.iter(|| {
            let patterns = detector.detect_patterns(black_box(&episodes_5k));
            black_box(patterns);
        });
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_pattern_detection,
    bench_clustering_algorithms,
    bench_statistical_filtering,
    bench_similarity_thresholds,
    bench_pattern_merging,
    bench_end_to_end_pipeline,
    bench_large_scale
);
criterion_main!(benches);
