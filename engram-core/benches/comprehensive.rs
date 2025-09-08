use criterion::{Criterion, criterion_group, criterion_main};

mod milestone_1;

use milestone_1::benchmark_milestone_1;

criterion_group!(benches, benchmark_milestone_1);

criterion_main!(benches);
