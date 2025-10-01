# 01 Memory Graph And Store – Style & Safety Cleanups

## Scope
- `engram-core/src/store.rs`
- `engram-core/src/memory.rs`
- `engram-core/src/memory_graph/backends/dashmap.rs`
- `engram-core/src/memory_graph/backends/infallible.rs`
- `engram-core/src/memory_graph/backends/hashmap.rs`
- `engram-core/src/memory_graph/graph.rs`

## Current hotspots
| File | Issues | Relevant lints |
| --- | ---: | --- |
| `engram-core/src/store.rs` | 16 | unused self (6), items after statements (2), ref option (2) |
| `engram-core/src/memory.rs` | 12 | unwrap used (10), struct field names (2) |
| `engram-core/src/memory_graph/backends/dashmap.rs` | 10 | option if let else (3), explicit iter loop (2), unnecessary filter map (1) |
| `engram-core/src/memory_graph/backends/infallible.rs` | 7 | fields `activation`, `degraded`, and `pressure` are never read (1), non canonical partial ord impl (1), clone on copy (1) |
| `engram-core/src/memory_graph/backends/hashmap.rs` | 5 | option if let else (3), unwrap or default (1), set contains or insert (1) |
| `engram-core/src/memory_graph/graph.rs` | 3 | unnecessary map or (3) |

## Recommended approach
- Replace `unwrap`/`expect` with error propagation or tailored assertions.
- Remove, rename, or gate unused parameters and fields to keep APIs tight.
- Adopt idiomatic combinators (`map_or`, `ok_or`, `matches!`) in place of manual branching.
- Trim unused struct members or wire them into tests so the compiler sees their value.
- Switch to iterator adapters (e.g., `for item in collection`) for clarity and perf hints.
