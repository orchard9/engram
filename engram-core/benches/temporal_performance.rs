//! Performance benchmarks for temporal decay computations
//!
//! Validates that decay computations meet the <1ms P95 performance target
//! required for Milestone 4 acceptance criteria.

#![allow(missing_docs)]
#![allow(clippy::semicolon_if_nothing_returned)]
#![allow(unused_must_use)]

use chrono::Utc;
use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};
use engram_core::{
    Confidence,
    decay::{BiologicalDecaySystem, DecayConfigBuilder, DecayFunction},
};
use std::sync::Arc;
use std::time::Duration;

fn bench_exponential_decay(c: &mut Criterion) {
    let config = DecayConfigBuilder::new()
        .exponential(1.96) // Standard Ebbinghaus tau
        .build();
    let decay_system = Arc::new(BiologicalDecaySystem::with_config(config));

    c.bench_function("exponential_decay_single", |b| {
        b.iter(|| {
            decay_system.compute_decayed_confidence(
                black_box(Confidence::exact(0.8)),
                black_box(Duration::from_secs(3600 * 2)), // 2 hours
                black_box(1),
                black_box(Utc::now()),
                black_box(None),
            )
        })
    });
}

fn bench_power_law_decay(c: &mut Criterion) {
    let config = DecayConfigBuilder::new()
        .power_law(0.18) // Bahrick permastore exponent
        .build();
    let decay_system = Arc::new(BiologicalDecaySystem::with_config(config));

    c.bench_function("power_law_decay_single", |b| {
        b.iter(|| {
            decay_system.compute_decayed_confidence(
                black_box(Confidence::exact(0.8)),
                black_box(Duration::from_secs(3600 * 24)), // 24 hours
                black_box(1),
                black_box(Utc::now()),
                black_box(None),
            )
        })
    });
}

fn bench_two_component_decay_hippocampal(c: &mut Criterion) {
    let config = DecayConfigBuilder::new().two_component(3).build();
    let decay_system = Arc::new(BiologicalDecaySystem::with_config(config));

    c.bench_function("two_component_hippocampal", |b| {
        b.iter(|| {
            decay_system.compute_decayed_confidence(
                black_box(Confidence::exact(0.8)),
                black_box(Duration::from_secs(3600 * 6)),
                black_box(1), // Below consolidation threshold
                black_box(Utc::now()),
                black_box(None),
            )
        })
    });
}

fn bench_two_component_decay_neocortical(c: &mut Criterion) {
    let config = DecayConfigBuilder::new().two_component(3).build();
    let decay_system = Arc::new(BiologicalDecaySystem::with_config(config));

    c.bench_function("two_component_neocortical", |b| {
        b.iter(|| {
            decay_system.compute_decayed_confidence(
                black_box(Confidence::exact(0.8)),
                black_box(Duration::from_secs(3600 * 6)),
                black_box(5), // Above consolidation threshold
                black_box(Utc::now()),
                black_box(None),
            )
        })
    });
}

fn bench_hybrid_decay_short_term(c: &mut Criterion) {
    let config = DecayConfigBuilder::new()
        .hybrid(0.8, 0.25, 86400) // Standard hybrid parameters
        .build();
    let decay_system = Arc::new(BiologicalDecaySystem::with_config(config));

    c.bench_function("hybrid_decay_short_term", |b| {
        b.iter(|| {
            decay_system.compute_decayed_confidence(
                black_box(Confidence::exact(0.8)),
                black_box(Duration::from_secs(3600 * 12)), // 12 hours (< 24h)
                black_box(1),
                black_box(Utc::now()),
                black_box(None),
            )
        })
    });
}

fn bench_hybrid_decay_long_term(c: &mut Criterion) {
    let config = DecayConfigBuilder::new().hybrid(0.8, 0.25, 86400).build();
    let decay_system = Arc::new(BiologicalDecaySystem::with_config(config));

    c.bench_function("hybrid_decay_long_term", |b| {
        b.iter(|| {
            decay_system.compute_decayed_confidence(
                black_box(Confidence::exact(0.8)),
                black_box(Duration::from_secs(3600 * 48)), // 48 hours (> 24h)
                black_box(1),
                black_box(Utc::now()),
                black_box(None),
            )
        })
    });
}

fn bench_batch_decay_100_memories(c: &mut Criterion) {
    let decay_system = Arc::new(BiologicalDecaySystem::new());

    // Prepare batch data
    let confidences: Vec<Confidence> = (0..100)
        .map(|i| Confidence::exact(0.5 + (i as f32) * 0.005))
        .collect();

    let durations: Vec<Duration> = (0..100)
        .map(|i| Duration::from_secs(3600 * (i % 24 + 1)))
        .collect();

    c.bench_function("batch_decay_100_memories", |b| {
        b.iter(|| {
            for (i, conf) in confidences.iter().enumerate() {
                black_box(decay_system.compute_decayed_confidence(
                    *conf,
                    durations[i],
                    (i % 10) as u64,
                    Utc::now(),
                    None,
                ));
            }
        })
    });
}

fn bench_varying_elapsed_times(c: &mut Criterion) {
    let decay_system = Arc::new(BiologicalDecaySystem::new());
    let mut group = c.benchmark_group("decay_by_elapsed_time");

    for hours in &[1, 6, 24, 168, 720] {
        group.bench_with_input(
            BenchmarkId::from_parameter(format!("{hours}h")),
            hours,
            |b, &hours| {
                b.iter(|| {
                    decay_system.compute_decayed_confidence(
                        black_box(Confidence::exact(0.8)),
                        black_box(Duration::from_secs(3600 * hours)),
                        black_box(1),
                        black_box(Utc::now()),
                        black_box(None),
                    )
                })
            },
        );
    }

    group.finish();
}

fn bench_varying_access_counts(c: &mut Criterion) {
    let config = DecayConfigBuilder::new().two_component(3).build();
    let decay_system = Arc::new(BiologicalDecaySystem::with_config(config));
    let mut group = c.benchmark_group("decay_by_access_count");

    for access_count in &[1, 3, 5, 10, 100] {
        group.bench_with_input(
            BenchmarkId::from_parameter(format!("{access_count}")),
            access_count,
            |b, &access_count| {
                b.iter(|| {
                    decay_system.compute_decayed_confidence(
                        black_box(Confidence::exact(0.8)),
                        black_box(Duration::from_secs(3600 * 6)),
                        black_box(access_count),
                        black_box(Utc::now()),
                        black_box(None),
                    )
                })
            },
        );
    }

    group.finish();
}

fn bench_all_decay_functions_comparison(c: &mut Criterion) {
    let mut group = c.benchmark_group("decay_function_comparison");

    let functions = vec![
        (
            "exponential",
            DecayFunction::Exponential { tau_hours: 1.96 },
        ),
        ("power_law", DecayFunction::PowerLaw { beta: 0.18 }),
        (
            "two_component",
            DecayFunction::TwoComponent {
                consolidation_threshold: 3,
            },
        ),
        (
            "hybrid",
            DecayFunction::Hybrid {
                short_term_tau: 0.8,
                long_term_beta: 0.25,
                transition_point: 86400,
            },
        ),
    ];

    for (name, decay_fn) in functions {
        let config = DecayConfigBuilder::new().enabled(true).build();
        let mut decay_system = BiologicalDecaySystem::with_config(config);
        decay_system.config.default_function = decay_fn;
        let decay_system = Arc::new(decay_system);

        group.bench_with_input(
            BenchmarkId::from_parameter(name),
            &decay_system,
            |b, system| {
                b.iter(|| {
                    system.compute_decayed_confidence(
                        black_box(Confidence::exact(0.8)),
                        black_box(Duration::from_secs(3600 * 6)),
                        black_box(1),
                        black_box(Utc::now()),
                        black_box(None),
                    )
                })
            },
        );
    }

    group.finish();
}

fn bench_decay_disabled_overhead(c: &mut Criterion) {
    let enabled_config = DecayConfigBuilder::new()
        .exponential(1.96)
        .enabled(true)
        .build();
    let disabled_config = DecayConfigBuilder::new()
        .exponential(1.96)
        .enabled(false)
        .build();

    let enabled_system = Arc::new(BiologicalDecaySystem::with_config(enabled_config));
    let disabled_system = Arc::new(BiologicalDecaySystem::with_config(disabled_config));

    c.bench_function("decay_enabled", |b| {
        b.iter(|| {
            enabled_system.compute_decayed_confidence(
                black_box(Confidence::exact(0.8)),
                black_box(Duration::from_secs(3600)),
                black_box(1),
                black_box(Utc::now()),
                black_box(None),
            )
        })
    });

    c.bench_function("decay_disabled", |b| {
        b.iter(|| {
            disabled_system.compute_decayed_confidence(
                black_box(Confidence::exact(0.8)),
                black_box(Duration::from_secs(3600)),
                black_box(1),
                black_box(Utc::now()),
                black_box(None),
            )
        })
    });
}

fn bench_per_memory_override_overhead(c: &mut Criterion) {
    let decay_system = Arc::new(BiologicalDecaySystem::new());

    c.bench_function("no_override", |b| {
        b.iter(|| {
            decay_system.compute_decayed_confidence(
                black_box(Confidence::exact(0.8)),
                black_box(Duration::from_secs(3600)),
                black_box(1),
                black_box(Utc::now()),
                black_box(None),
            )
        })
    });

    c.bench_function("with_override", |b| {
        b.iter(|| {
            decay_system.compute_decayed_confidence(
                black_box(Confidence::exact(0.8)),
                black_box(Duration::from_secs(3600)),
                black_box(1),
                black_box(Utc::now()),
                black_box(Some(DecayFunction::PowerLaw { beta: 0.2 })),
            )
        })
    });
}

criterion_group!(
    benches,
    bench_exponential_decay,
    bench_power_law_decay,
    bench_two_component_decay_hippocampal,
    bench_two_component_decay_neocortical,
    bench_hybrid_decay_short_term,
    bench_hybrid_decay_long_term,
    bench_batch_decay_100_memories,
    bench_varying_elapsed_times,
    bench_varying_access_counts,
    bench_all_decay_functions_comparison,
    bench_decay_disabled_overhead,
    bench_per_memory_override_overhead,
);

criterion_main!(benches);
