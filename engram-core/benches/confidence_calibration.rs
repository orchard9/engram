use criterion::{black_box, criterion_group, criterion_main, Criterion};
use engram_core::{Confidence, storage::confidence::{StorageConfidenceCalibrator, ConfidenceTier}};
use std::time::Duration;

fn bench_confidence_calibration(c: &mut Criterion) {
    let calibrator = StorageConfidenceCalibrator::new();
    let original_confidence = Confidence::from_raw(0.8);
    let storage_duration = Duration::from_secs(3600); // 1 hour

    c.bench_function("single_confidence_adjustment", |b| {
        b.iter(|| {
            black_box(calibrator.adjust_for_storage_tier(
                black_box(original_confidence),
                black_box(ConfidenceTier::Warm),
                black_box(storage_duration),
            ))
        })
    });

    // Batch calibration benchmark
    let mut batch_data = vec![
        (original_confidence, ConfidenceTier::Hot, Duration::from_secs(0)),
        (original_confidence, ConfidenceTier::Warm, Duration::from_secs(3600)),
        (original_confidence, ConfidenceTier::Cold, Duration::from_secs(86400)),
    ];

    c.bench_function("batch_confidence_adjustment", |b| {
        b.iter(|| {
            calibrator.adjust_batch(black_box(&mut batch_data));
        })
    });

    // Tier-only calibration (no temporal decay)
    c.bench_function("tier_only_adjustment", |b| {
        b.iter(|| {
            black_box(calibrator.adjust_for_tier_only(
                black_box(original_confidence),
                black_box(ConfidenceTier::Warm),
            ))
        })
    });
}

criterion_group!(benches, bench_confidence_calibration);
criterion_main!(benches);