#![allow(missing_docs)]

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use std::time::Duration;

use engram_core::activation::ParallelSpreadingConfig;
use engram_core::activation::test_support::{deterministic_config, run_spreading};
use support::graph_builders::{GraphFixture, barabasi_albert, chain, fan};

#[path = "../tests/support/mod.rs"]
mod support;

struct BenchmarkScenario {
    name: &'static str,
    fixture: GraphFixture,
    config: ParallelSpreadingConfig,
}

fn build_scenarios() -> Vec<BenchmarkScenario> {
    let mut small = deterministic_config(0x5BAD_F00D);
    small.num_threads = 1;
    small.max_depth = 4;

    let mut medium = deterministic_config(0x51_2E);
    medium.num_threads = 2;
    medium.max_depth = 5;

    let mut large = deterministic_config(0xA1_1E);
    large.num_threads = 4;
    large.max_depth = 6;

    let mut fan_cfg = deterministic_config(0xFA_05);
    fan_cfg.num_threads = 1;
    fan_cfg.max_depth = 3;

    vec![
        BenchmarkScenario {
            name: "chain_small",
            fixture: chain(64),
            config: small,
        },
        BenchmarkScenario {
            name: "scale_free_medium",
            fixture: barabasi_albert(2_000, 3, 0xBADC_AB1E),
            config: medium,
        },
        BenchmarkScenario {
            name: "scale_free_large",
            fixture: barabasi_albert(10_000, 4, 0xDEAD_BEEF),
            config: large,
        },
        BenchmarkScenario {
            name: "fan_effect",
            fixture: fan("fan_anchor", 32),
            config: fan_cfg,
        },
    ]
}

fn spreading_benchmarks(c: &mut Criterion) {
    let mut group = c.benchmark_group("spreading_activation");
    group.sample_size(30);
    group.warm_up_time(Duration::from_secs(1));
    group.measurement_time(Duration::from_secs(5));

    for scenario in build_scenarios() {
        group.bench_function(BenchmarkId::new("scenario", scenario.name), |b| {
            let fixture = scenario.fixture.clone();
            let base_config = scenario.config.clone();
            b.iter(|| {
                let mut config = base_config.clone();
                fixture.apply_config_adjustments(&mut config);
                let run = run_spreading(fixture.graph(), &fixture.seeds, config)
                    .expect("benchmark spread");
                criterion::black_box(run.metrics.total_activations);
            });
        });
    }

    group.finish();
}

fn configure_criterion() -> Criterion {
    Criterion::default()
        .output_directory(std::path::Path::new("docs/assets/benchmarks/spreading"))
        .confidence_level(0.95)
        .noise_threshold(0.02)
}

criterion_group! {
    name = spreading_group;
    config = configure_criterion();
    targets = spreading_benchmarks
}

criterion_main!(spreading_group);
