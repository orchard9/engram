//! Comprehensive benchmarking entry point for Engram milestone 1.

#![allow(missing_docs)]

use criterion::{criterion_group, criterion_main};

mod milestone_1;

use milestone_1::benchmark_milestone_1;

criterion_group!(benches, benchmark_milestone_1);

criterion_main!(benches);
