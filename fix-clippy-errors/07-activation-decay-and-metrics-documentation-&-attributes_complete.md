# 07 Activation Decay And Metrics – Documentation & Attributes

## Scope
- `engram-core/src/activation/storage_aware.rs`
- `engram-core/src/activation/latency_budget.rs`
- `engram-core/src/metrics/lockfree.rs`
- `engram-core/src/activation/mod.rs`
- `engram-core/src/activation/scheduler.rs`
- `engram-core/src/metrics/numa_aware.rs`

## Current hotspots
| File | Issues | Relevant lints |
| --- | ---: | --- |
| `engram-core/src/activation/storage_aware.rs` | 3 | missing const for fn (2), must use candidate (1) |
| `engram-core/src/activation/latency_budget.rs` | 2 | missing const for fn (2) |
| `engram-core/src/metrics/lockfree.rs` | 2 | missing panics doc (2) |
| `engram-core/src/activation/mod.rs` | 1 | missing const for fn (1) |
| `engram-core/src/activation/scheduler.rs` | 1 | missing const for fn (1) |
| `engram-core/src/metrics/numa_aware.rs` | 1 | missing errors doc (1) |

## Recommended approach
- Add the missing `# Errors` / `# Panics` sections so linted APIs describe failure modes.
- Apply `#[must_use]` or adjust return signatures where results should not be ignored.
- Promote simple helpers to `const fn` (or justify why they cannot be const).
