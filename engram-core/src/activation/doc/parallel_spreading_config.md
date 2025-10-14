# ParallelSpreadingConfig Cheat Sheet

| Field | Default | Recommended Range | Cognitive / Operational Notes |
| --- | --- | --- | --- |
| `num_threads` | `num_cpus::get()` | 2 – physical cores | Controls parallel workers. Lower the value for tenant isolation. |
| `work_stealing_ratio` | `0.65` | 0.3 – 0.9 | Higher values spread activation more evenly but add scheduling overhead. |
| `batch_size` | `64` | 32 – 96 | Larger batches improve throughput but may exceed hop latency budgets. |
| `pool_initial_size` | `4096` | 1024 – 8192 | Preallocated activation records. Align with expected frontier size. |
| `numa_aware` | `false` | `true` when NUMA detected | Pin memory pools per NUMA node to reduce cross-node traffic. |
| `max_depth` | `3` | 2 – 5 | Cognitive distance (hops). Higher depth improves recall but increases fan effect. |
| `decay_function` | `DecayFunction::Exponential { lambda: 0.65 }` | 0.4 – 0.8 | Models forgetting curves; lower lambda slows decay. |
| `threshold` | `0.1` | 0.05 – 0.3 | Activation floor. Raising reduces noise, lowering increases breadth. |
| `cycle_detection` | `true` | `true` | Prevents runaway activation in cyclic graphs. |
| `cycle_penalty_factor` | `0.5` | 0.2 – 0.8 | Dampens activation when a cycle is hit. Lower to aggressively suppress loops. |
| `tier_timeouts` | `[4ms, 7ms, 12ms]` | tune per workload | Deadline per storage tier. Keep hot tier <5 ms for responsiveness. |
| `priority_hot_tier` | `true` | `true` | Prioritise short-hop traversals before warm/cold expansion. |
| `deterministic` | `false` | `true` in debugging | Ensures reproducible spreads and enables trace generation. |
| `seed` | `None` | Any `u64` | Used when `deterministic` is `true` to seed RNG. |
| `phase_sync_interval` | `10ms` | 5 – 25 ms | Synchronises workers in deterministic mode; longer intervals increase variance. |
| `enable_gpu` | `false` | `true` when GPU available | Activates GPU batch kernels. Guard with `gpu_threshold` to avoid tiny batches. |
| `gpu_threshold` | `64` | 16 – 256 | Minimum batch size before GPU offload. |
| `enable_memory_pool` | `true` | `true` | Reuse activation records; set to `false` only for diagnostics. |
| `adaptive_batcher_config` | `None` | Use Task 010 defaults | Enables EWMA-based batch tuning; ensures throughput parity with time budgets. |

> Defaults drawn from `ParallelSpreadingConfig::default()` in `engram-core/src/activation/mod.rs` (Task 010).
