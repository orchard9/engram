//! Benchmark for HNSW activation dynamics
use criterion::{Criterion, black_box, criterion_group, criterion_main};
use engram_core::Confidence;
use engram_core::index::ActivationDynamics;

fn benchmark_record_activation(c: &mut Criterion) {
    c.bench_function("activation_dynamics_record", |b| {
        let dynamics = ActivationDynamics::new();
        b.iter(|| {
            dynamics.record_activation(black_box(0.42), Confidence::HIGH);
        });
    });
}

#[allow(missing_docs)]
mod benches {
    use super::{benchmark_record_activation, criterion_group};
    criterion_group!(hnsw_dynamics, benchmark_record_activation);
}

criterion_main!(benches::hnsw_dynamics);
