//! Criterion benchmarks for cosine similarity and related vector operations.

#![allow(missing_docs)]

use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};
use engram_core::activation::simd_optimization::SimdActivationMapper;
use engram_core::compute::{
    VectorOps, cosine_similarity_768, cosine_similarity_batch_768, scalar::ScalarVectorOps,
};

fn generate_test_vector() -> [f32; 768] {
    let mut result = [0.0f32; 768];
    for (i, item) in result.iter_mut().enumerate() {
        let index = f32::from(u16::try_from(i).expect("vector index within u16"));
        *item = index.sin().mul_add(0.5, 0.5); // Generate deterministic test data
    }
    result
}

fn generate_test_batch(size: usize) -> Vec<[f32; 768]> {
    (0..size)
        .map(|i| {
            let mut vec = generate_test_vector();
            // Add variation based on index
            for (j, item) in vec.iter_mut().enumerate().take(i % 100) {
                let offset = f32::from(u16::try_from(j).expect("offset within u16"));
                *item *= 0.9 + (offset / 1000.0);
            }
            vec
        })
        .collect()
}

fn bench_cosine_similarity(c: &mut Criterion) {
    let vector_a = generate_test_vector();
    let mut vector_b = generate_test_vector();
    // Make vector_b slightly different
    for item in vector_b.iter_mut().take(100) {
        *item *= 0.9;
    }

    let scalar_ops = ScalarVectorOps::new();

    c.bench_function("cosine_similarity_scalar", |b| {
        b.iter(|| {
            black_box(scalar_ops.cosine_similarity_768(black_box(&vector_a), black_box(&vector_b)))
        });
    });

    c.bench_function("cosine_similarity_simd", |b| {
        b.iter(|| {
            black_box(cosine_similarity_768(
                black_box(&vector_a),
                black_box(&vector_b),
            ))
        });
    });
}

fn bench_cosine_similarity_batch(c: &mut Criterion) {
    let query = generate_test_vector();
    let mut vectors = Vec::with_capacity(100);

    for i in 0..100 {
        let mut vec = generate_test_vector();
        // Add some variation
        for item in vec.iter_mut().take(i % 50) {
            let index = f32::from(u8::try_from(i).expect("batch index within u8"));
            *item *= index.mul_add(0.002, 0.8);
        }
        vectors.push(vec);
    }

    let scalar_ops = ScalarVectorOps::new();

    c.bench_function("cosine_similarity_batch_scalar", |b| {
        b.iter(|| {
            black_box(
                scalar_ops.cosine_similarity_batch_768(black_box(&query), black_box(&vectors)),
            )
        });
    });

    c.bench_function("cosine_similarity_batch_simd", |b| {
        b.iter(|| {
            black_box(cosine_similarity_batch_768(
                black_box(&query),
                black_box(&vectors),
            ))
        });
    });
}

fn bench_dot_product(c: &mut Criterion) {
    let vector_a = generate_test_vector();
    let vector_b = generate_test_vector();

    let scalar_ops = ScalarVectorOps::new();

    c.bench_function("dot_product_scalar", |b| {
        b.iter(|| {
            black_box(scalar_ops.dot_product_768(black_box(&vector_a), black_box(&vector_b)))
        });
    });

    // Note: dot_product_768 is not exposed publicly, so we'll skip SIMD benchmark for dot product
    // This shows that cosine_similarity is the primary optimization target
}

fn bench_batch_spreading(c: &mut Criterion) {
    let mut group = c.benchmark_group("batch_spreading");

    // Benchmark different batch sizes to find optimal SIMD batch size
    for batch_size in [8, 16, 32, 64, 128, 256] {
        let query = generate_test_vector();
        let vectors = generate_test_batch(batch_size);

        group.bench_with_input(
            BenchmarkId::new("batch_cosine_similarity", batch_size),
            &batch_size,
            |b, _| {
                b.iter(|| {
                    black_box(cosine_similarity_batch_768(
                        black_box(&query),
                        black_box(&vectors),
                    ))
                });
            },
        );
    }

    group.finish();
}

fn bench_sigmoid_activation(c: &mut Criterion) {
    let mapper = SimdActivationMapper::new();

    // Generate similarity scores
    let mut similarities = Vec::with_capacity(1000);
    for i in 0..1000 {
        let val = f32::from(u16::try_from(i).expect("index within u16"));
        similarities.push((val / 1000.0) - 0.5); // Range from -0.5 to 0.5
    }

    c.bench_function("batch_sigmoid_activation", |b| {
        b.iter(|| {
            black_box(mapper.batch_sigmoid_activation(
                black_box(&similarities),
                black_box(0.5),
                black_box(0.1),
            ))
        });
    });
}

fn bench_fma_confidence_aggregate(c: &mut Criterion) {
    let mapper = SimdActivationMapper::new();

    let mut activations = vec![0.5f32; 1000];
    let confidence_weights = vec![0.8f32; 1000];
    let path_confidence = 0.9f32;

    c.bench_function("fma_confidence_aggregate", |b| {
        b.iter(|| {
            mapper.fma_confidence_aggregate(
                black_box(&mut activations),
                black_box(&confidence_weights),
                black_box(path_confidence),
            );
        });
    });
}

fn bench_integrated_spreading_pipeline(c: &mut Criterion) {
    let mut group = c.benchmark_group("integrated_spreading_pipeline");

    for batch_size in [32, 64, 128] {
        let query = generate_test_vector();
        let vectors = generate_test_batch(batch_size);
        let mapper = SimdActivationMapper::new();

        group.bench_with_input(
            BenchmarkId::new("full_pipeline", batch_size),
            &batch_size,
            |b, _| {
                b.iter(|| {
                    // Step 1: Compute batch similarities (SIMD optimized)
                    let similarities =
                        cosine_similarity_batch_768(black_box(&query), black_box(&vectors));

                    // Step 2: Convert to activations (SIMD optimized)
                    let activations = mapper.batch_sigmoid_activation(
                        black_box(&similarities),
                        black_box(0.5),
                        black_box(0.1),
                    );

                    black_box(activations);
                });
            },
        );
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_cosine_similarity,
    bench_cosine_similarity_batch,
    bench_dot_product,
    bench_batch_spreading,
    bench_sigmoid_activation,
    bench_fma_confidence_aggregate,
    bench_integrated_spreading_pipeline,
);
criterion_main!(benches);
