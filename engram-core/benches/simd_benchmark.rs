use criterion::{black_box, criterion_group, criterion_main, Criterion};
use engram_core::compute::{scalar::ScalarVectorOps, VectorOps, cosine_similarity_768, cosine_similarity_batch_768};

fn generate_test_vector() -> [f32; 768] {
    let mut result = [0.0f32; 768];
    for (i, item) in result.iter_mut().enumerate() {
        *item = (i as f32).sin() * 0.5 + 0.5; // Generate deterministic test data
    }
    result
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
            black_box(scalar_ops.cosine_similarity_768(
                black_box(&vector_a),
                black_box(&vector_b),
            ))
        })
    });

    c.bench_function("cosine_similarity_simd", |b| {
        b.iter(|| {
            black_box(cosine_similarity_768(
                black_box(&vector_a),
                black_box(&vector_b),
            ))
        })
    });
}

fn bench_cosine_similarity_batch(c: &mut Criterion) {
    let query = generate_test_vector();
    let mut vectors = Vec::with_capacity(100);
    
    for i in 0..100 {
        let mut vec = generate_test_vector();
        // Add some variation
        for item in vec.iter_mut().take(i % 50) {
            *item *= 0.8 + (i as f32) * 0.002;
        }
        vectors.push(vec);
    }

    let scalar_ops = ScalarVectorOps::new();

    c.bench_function("cosine_similarity_batch_scalar", |b| {
        b.iter(|| {
            black_box(scalar_ops.cosine_similarity_batch_768(
                black_box(&query),
                black_box(&vectors),
            ))
        })
    });

    c.bench_function("cosine_similarity_batch_simd", |b| {
        b.iter(|| {
            black_box(cosine_similarity_batch_768(
                black_box(&query),
                black_box(&vectors),
            ))
        })
    });
}

fn bench_dot_product(c: &mut Criterion) {
    let vector_a = generate_test_vector();
    let vector_b = generate_test_vector();

    let scalar_ops = ScalarVectorOps::new();

    c.bench_function("dot_product_scalar", |b| {
        b.iter(|| {
            black_box(scalar_ops.dot_product_768(
                black_box(&vector_a),
                black_box(&vector_b),
            ))
        })
    });

    // Note: dot_product_768 is not exposed publicly, so we'll skip SIMD benchmark for dot product
    // This shows that cosine_similarity is the primary optimization target
}

criterion_group!(
    benches,
    bench_cosine_similarity,
    bench_cosine_similarity_batch,
    bench_dot_product
);
criterion_main!(benches);