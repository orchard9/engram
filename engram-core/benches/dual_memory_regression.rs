#![allow(missing_docs)]

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use engram_core::optimization::DualMemoryCache;
use rand::{Rng, SeedableRng, rngs::SmallRng};
use uuid::Uuid;

const SAMPLE_SIZES: [usize; 3] = [1_000, 10_000, 50_000];

fn bench_centroid_lookup(c: &mut Criterion) {
    let mut group = c.benchmark_group("dual_memory_centroid_lookup");

    for &size in &SAMPLE_SIZES {
        group.throughput(Throughput::Elements(size as u64));
        group.bench_with_input(BenchmarkId::new("hit", size), &size, |b, &n| {
            let (cache, ids) = build_cache_with_centroids(n);
            b.iter(|| {
                for id in &ids {
                    criterion::black_box(cache.get_centroid(id));
                }
            });
        });
    }

    group.finish();
}

fn bench_binding_updates(c: &mut Criterion) {
    let mut group = c.benchmark_group("dual_memory_binding_updates");

    for &size in &SAMPLE_SIZES {
        group.throughput(Throughput::Elements(size as u64));
        group.bench_with_input(BenchmarkId::new("sequential", size), &size, |b, &n| {
            let cache = DualMemoryCache::new();
            let bindings = random_bindings(n);
            b.iter(|| {
                for (concept_id, episode_id, strength) in &bindings {
                    cache.update_binding(*concept_id, *episode_id, *strength);
                }
            });
        });
    }

    group.finish();
}

fn build_cache_with_centroids(count: usize) -> (DualMemoryCache, Vec<Uuid>) {
    let cache = DualMemoryCache::new();
    let mut rng = SmallRng::seed_from_u64(42);
    let mut ids = Vec::with_capacity(count);
    for _ in 0..count {
        let id = Uuid::new_v4();
        let centroid = random_embedding(&mut rng);
        cache.upsert_centroid(id, &centroid);
        ids.push(id);
    }
    (cache, ids)
}

fn random_bindings(count: usize) -> Vec<(Uuid, Uuid, f32)> {
    let mut rng = SmallRng::seed_from_u64(7);
    (0..count)
        .map(|_| (Uuid::new_v4(), Uuid::new_v4(), rng.r#gen()))
        .collect()
}

fn random_embedding(rng: &mut SmallRng) -> [f32; engram_core::EMBEDDING_DIM] {
    let mut embedding = [0.0f32; engram_core::EMBEDDING_DIM];
    for value in &mut embedding {
        *value = rng.r#gen();
    }
    embedding
}

criterion_group!(
    dual_memory_benches,
    bench_centroid_lookup,
    bench_binding_updates
);
criterion_main!(dual_memory_benches);
