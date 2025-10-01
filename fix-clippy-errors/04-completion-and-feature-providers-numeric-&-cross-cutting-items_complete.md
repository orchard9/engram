# 04 Completion And Feature Providers – Numeric & Cross-Cutting Items

## Scope
- `engram-core/src/completion/hippocampal.rs`
- `engram-core/src/completion/reconstruction.rs`
- `engram-core/src/completion/consolidation.rs`
- `engram-core/src/completion/context.rs`
- `engram-core/src/completion/hypothesis.rs`
- `engram-core/src/features/decay.rs`

## Current hotspots
| File | Issues | Relevant lints |
| --- | ---: | --- |
| `engram-core/src/completion/hippocampal.rs` | 17 | cast precision loss (10), cast possible truncation (2), cast sign loss (2) |
| `engram-core/src/completion/reconstruction.rs` | 6 | cast precision loss (6) |
| `engram-core/src/completion/consolidation.rs` | 5 | cast precision loss (5) |
| `engram-core/src/completion/context.rs` | 4 | cast precision loss (4) |
| `engram-core/src/completion/hypothesis.rs` | 2 | assigning clones (1), cast precision loss (1) |
| `engram-core/src/features/decay.rs` | 1 | suboptimal flops (1) |

## Recommended approach
- Replace lossy `as` casts with checked conversions or widen data types as appropriate.
- Exploit `mul_add` / fused ops or precompute invariants to silence floating-point efficiency lints.
- Pass heavy data (embeddings, counters) by reference to avoid copies and appease Clippy.
