# 01 Memory Graph And Store – Documentation & Attributes

## Scope
- `engram-core/src/memory_graph/graph.rs`
- `engram-core/src/memory_graph/traits.rs`
- `engram-core/src/memory.rs`
- `engram-core/src/memory_graph/backends/dashmap.rs`
- `engram-core/src/memory_graph/backends/hashmap.rs`
- `engram-core/src/memory_graph/backends/infallible.rs`

## Current hotspots
| File | Issues | Relevant lints |
| --- | ---: | --- |
| `engram-core/src/memory_graph/graph.rs` | 22 | missing errors doc (13), must use candidate (7), missing const for fn (2) |
| `engram-core/src/memory_graph/traits.rs` | 14 | missing errors doc (14) |
| `engram-core/src/memory.rs` | 6 | missing const for fn (3), missing panics doc (3) |
| `engram-core/src/memory_graph/backends/dashmap.rs` | 2 | doc markdown (2) |
| `engram-core/src/memory_graph/backends/hashmap.rs` | 2 | doc markdown (2) |
| `engram-core/src/memory_graph/backends/infallible.rs` | 2 | doc markdown (1), must use candidate (1) |

## Recommended approach
- Add the missing `# Errors` / `# Panics` sections so linted APIs describe failure modes.
- Apply `#[must_use]` or adjust return signatures where results should not be ignored.
- Promote simple helpers to `const fn` (or justify why they cannot be const).
- Fix intra-doc markdown (backticks around identifiers, valid headings).
