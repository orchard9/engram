//! Benchmark for cold tier columnar storage performance
use criterion::{Criterion, black_box, criterion_group, criterion_main};
use engram_core::storage::{ColdTier, ColdTierConfig, StorageError, StorageTierBackend};
use engram_core::{Confidence, Memory};
use std::sync::Arc;
use tokio::runtime::Runtime;

const DATASET_SIZE: usize = 2048;

fn build_memory(id: usize, embedding: &[f32; 768]) -> Arc<Memory> {
    use engram_core::EpisodeBuilder;
    let episode = EpisodeBuilder::new()
        .id(format!("bench_{id}"))
        .when(chrono::Utc::now())
        .what("bench".to_string())
        .embedding(*embedding)
        .confidence(Confidence::HIGH)
        .build();
    Arc::new(Memory::from_episode(episode, 0.5))
}

fn generate_dataset() -> Vec<[f32; 768]> {
    let mut dataset = Vec::with_capacity(DATASET_SIZE);
    for idx in 0..DATASET_SIZE {
        let mut embedding = [0.0f32; 768];
        for (dim, value) in embedding.iter_mut().enumerate() {
            let seed = (idx * 37 + dim) as f32;
            *value = (seed.sin() + seed.cos()).mul_add(0.5, 0.0);
        }
        dataset.push(embedding);
    }
    dataset
}

fn populate_tier(tier: &ColdTier, dataset: &[[f32; 768]]) -> Result<(), StorageError> {
    let rt = Runtime::new().expect("tokio runtime");
    for (idx, embedding) in dataset.iter().enumerate() {
        let memory = build_memory(idx, embedding);
        rt.block_on(tier.store(memory))?;
    }
    Ok(())
}

fn cold_tier_benchmarks(c: &mut Criterion) {
    let dataset = generate_dataset();

    let plain_config = ColdTierConfig {
        capacity: DATASET_SIZE,
        enable_compression: false,
        ..Default::default()
    };
    let plain_tier = ColdTier::with_config(&plain_config);
    populate_tier(&plain_tier, &dataset).expect("populate plain tier");

    let mut compressed_config = plain_config;
    compressed_config.enable_compression = true;
    let compressed_tier = ColdTier::with_config(&compressed_config);
    populate_tier(&compressed_tier, &dataset).expect("populate compressed tier");

    let query = dataset[dataset.len() / 2];

    c.bench_function("cold_tier_similarity_plain", |b| {
        b.iter(|| {
            let _ = plain_tier.similarity_search(black_box(&query), 0.75);
        });
    });

    c.bench_function("cold_tier_similarity_compressed", |b| {
        b.iter(|| {
            let _ = compressed_tier.similarity_search(black_box(&query), 0.75);
        });
    });
}

#[allow(missing_docs)]
mod benches {
    use super::{cold_tier_benchmarks, criterion_group};
    criterion_group!(cold_tier_columnar, cold_tier_benchmarks);
}
criterion_main!(benches::cold_tier_columnar);
