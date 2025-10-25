//! Benchmark suite for tokenizer performance validation.
//!
//! Validates that tokenization meets sub-microsecond latency for short queries
//! and sub-10μs latency for complex queries, with zero allocations on hot path.
#![allow(missing_docs)] // Benchmarks don't require documentation

use criterion::{BenchmarkId, Criterion, Throughput, black_box, criterion_group, criterion_main};
use engram_core::query::parser::{Token, Tokenizer};

fn bench_tokenize_simple(c: &mut Criterion) {
    let query = "RECALL episode WHERE confidence > 0.7";

    c.bench_function("tokenize_simple", |b| {
        b.iter(|| {
            let mut tokenizer = Tokenizer::new(black_box(query));
            let mut count = 0;
            while let Ok(token) = tokenizer.next_token() {
                black_box(&token); // Prevent optimization
                count += 1;
                if token.value == Token::Eof {
                    break;
                }
            }
            count
        });
    });
}

fn bench_tokenize_complex(c: &mut Criterion) {
    let query = "SPREAD FROM node_123 MAX_HOPS 5 DECAY 0.15 THRESHOLD 0.1";

    c.bench_function("tokenize_complex", |b| {
        b.iter(|| {
            let mut tokenizer = Tokenizer::new(black_box(query));
            let mut count = 0;
            while let Ok(token) = tokenizer.next_token() {
                black_box(&token);
                count += 1;
                if token.value == Token::Eof {
                    break;
                }
            }
            count
        });
    });
}

fn bench_tokenize_varying_length(c: &mut Criterion) {
    let mut group = c.benchmark_group("tokenize_length");

    for size in &[10, 50, 100, 500, 1000] {
        let query = format!(
            "RECALL {} WHERE confidence > 0.7",
            "node_".repeat(*size / 10)
        );
        group.throughput(Throughput::Bytes(query.len() as u64));

        group.bench_with_input(BenchmarkId::from_parameter(size), &query, |b, q| {
            b.iter(|| {
                let mut tokenizer = Tokenizer::new(black_box(q));
                let mut count = 0;
                while let Ok(token) = tokenizer.next_token() {
                    black_box(&token);
                    count += 1;
                    if token.value == Token::Eof {
                        break;
                    }
                }
                count
            });
        });
    }

    group.finish();
}

fn bench_keyword_recognition(c: &mut Criterion) {
    // Test keyword lookup performance (should be <5ns per lookup)
    let queries = [
        "RECALL",
        "recall", // Case-insensitive
        "Recall",
        "PREDICT",
        "IMAGINE",
        "CONSOLIDATE",
        "SPREAD",
        "WHERE",
        "GIVEN",
        "BASED",
        "MAX_HOPS",
        "THRESHOLD",
    ];

    c.bench_function("keyword_recognition", |b| {
        b.iter(|| {
            for query in &queries {
                let mut tokenizer = Tokenizer::new(black_box(query));
                let token = tokenizer.next_token().unwrap();
                black_box(&token);
            }
        });
    });
}

fn bench_number_parsing(c: &mut Criterion) {
    let mut group = c.benchmark_group("number_parsing");

    group.bench_function("integer", |b| {
        let query = "123456";
        b.iter(|| {
            let mut tokenizer = Tokenizer::new(black_box(query));
            black_box(tokenizer.next_token().unwrap());
        });
    });

    group.bench_function("float", |b| {
        let query = "0.12345";
        b.iter(|| {
            let mut tokenizer = Tokenizer::new(black_box(query));
            black_box(tokenizer.next_token().unwrap());
        });
    });

    group.finish();
}

fn bench_zero_copy_verification(c: &mut Criterion) {
    // Verify identifiers use zero-copy slices (pointer comparison)
    let query = "episode_memory_node_identifier";

    c.bench_function("zero_copy_identifier", |b| {
        b.iter(|| {
            let query_ptr = black_box(query).as_ptr();
            let mut tokenizer = Tokenizer::new(black_box(query));
            if let Ok(token) = tokenizer.next_token()
                && let Token::Identifier(ident) = token.value
            {
                // Verify it's a slice, not a copy
                assert_eq!(ident.as_ptr(), query_ptr);
                black_box(ident);
            }
        });
    });
}

criterion_group!(
    benches,
    bench_tokenize_simple,
    bench_tokenize_complex,
    bench_tokenize_varying_length,
    bench_keyword_recognition,
    bench_number_parsing,
    bench_zero_copy_verification
);
criterion_main!(benches);
