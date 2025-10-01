# 05 Query And Cue Pipeline – Style & Safety Cleanups

## Scope
- `engram-core/src/index/mod.rs`
- `engram-core/src/query/integration.rs`
- `engram-core/src/cue/handlers.rs`
- `engram-core/src/index/cognitive_dynamics.rs`
- `engram-core/src/query/evidence.rs`
- `engram-core/src/query/mod.rs`

## Current hotspots
| File | Issues | Relevant lints |
| --- | ---: | --- |
| `engram-core/src/index/mod.rs` | 5 | unused self (2), field `generation` is never read (1), fields `last pressure check` and `pressure sensitivity` are never read (1) |
| `engram-core/src/query/integration.rs` | 5 | unused self (4), manual let else (1) |
| `engram-core/src/cue/handlers.rs` | 2 | map unwrap or (2) |
| `engram-core/src/index/cognitive_dynamics.rs` | 1 | method `variance` is never used (1) |
| `engram-core/src/query/evidence.rs` | 1 | match like matches macro (1) |
| `engram-core/src/query/mod.rs` | 1 | large enum variant (1) |

## Recommended approach
- Replace `unwrap`/`expect` with error propagation or tailored assertions.
- Remove, rename, or gate unused parameters and fields to keep APIs tight.
- Adopt idiomatic combinators (`map_or`, `ok_or`, `matches!`) in place of manual branching.
- Trim unused struct members or wire them into tests so the compiler sees their value.
