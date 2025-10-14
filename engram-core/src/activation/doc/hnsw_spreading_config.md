# HnswSpreadingConfig Cheat Sheet

| Field | Default | Recommended Range | Notes |
| --- | --- | --- | --- |
| `similarity_threshold` | `0.5` | 0.45 – 0.7 | Minimum cosine similarity. Lowering increases branching factor. |
| `distance_decay` | `0.8` | 0.6 – 0.9 | Exponential decay per hop. Lower values preserve activation deeper in the hierarchy. |
| `max_hops` | `3` | 2 – 4 | HNSW layer traversal depth. Higher values explore more neighbours but raise latency. |
| `use_hierarchical` | `true` | `true` | Enables top-down navigation through HNSW layers for better recall diversity. |
| `confidence_threshold` | `Confidence::LOW` | `Confidence::from_raw(0.45)` – `Confidence::from_raw(0.8)` | Prevents low-confidence edges from dominating spreads. |

Only available when the `hnsw_index` feature is enabled.
