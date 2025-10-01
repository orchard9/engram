# 07 Activation Decay And Metrics – Style & Safety Cleanups

## Scope
- `engram-core/src/metrics/numa_aware.rs`
- `engram-core/src/metrics/health.rs`
- `engram-core/src/activation/accumulator.rs`
- `engram-core/src/activation/traversal.rs`
- `engram-core/src/metrics/lockfree.rs`
- `engram-core/src/metrics/prometheus.rs`
- `engram-core/src/metrics/streaming.rs`
- `engram-core/src/activation/simd_optimization.rs`

## Current hotspots
| File | Issues | Relevant lints |
| --- | ---: | --- |
| `engram-core/src/metrics/numa_aware.rs` | 7 | unused self (3), field `topology` is never read (1), fields `policy` and `collectors` are never read (1) |
| `engram-core/src/metrics/health.rs` | 6 | unused self (6) |
| `engram-core/src/activation/accumulator.rs` | 2 | field `decay rates` is never read (1), unused self (1) |
| `engram-core/src/activation/traversal.rs` | 2 | fields `num workers` and `chunk size` are never read (1), unused self (1) |
| `engram-core/src/metrics/lockfree.rs` | 2 | unwrap used (2) |
| `engram-core/src/metrics/prometheus.rs` | 2 | function `format duration seconds` is never used (1), function `format bytes` is never used (1) |
| `engram-core/src/metrics/streaming.rs` | 2 | unwrap used (2) |
| `engram-core/src/activation/simd_optimization.rs` | 1 | unused self (1) |

## Recommended approach
- Replace `unwrap`/`expect` with error propagation or tailored assertions.
- Remove, rename, or gate unused parameters and fields to keep APIs tight.
- Adopt idiomatic combinators (`map_or`, `ok_or`, `matches!`) in place of manual branching.
- Trim unused struct members or wire them into tests so the compiler sees their value.
