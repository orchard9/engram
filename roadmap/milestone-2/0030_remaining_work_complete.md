# Task 0030: Milestone 2 Remaining Work Tracker

**Status:** Pending
**Owner:** TBD
**Purpose:** Capture the concrete follow-ups required to declare Milestone 2 truly complete.

## Outstanding Items

1. **Task 002 – Content-Addressable Retrieval**
   - Replace the current raw-float semantic hash with the specified BLAKE3 + quantization workflow.
   - Persist reproducible LSH projections (random seeds, projection vectors) instead of recomputing on the fly.
   - Add a direct `MemoryStore::get_by_content_address` path so deduplication and recalls can resolve matches without scanning hot tier maps.
   - Emit deduplication metrics/logs (`content_dedup_{skip,replace,merge}_total`) so operations can monitor churn.

2. **Task 003 – HNSW Cognitive Dynamics Adaptation**
   - Benchmark `ActivationDynamics::record_activation` and prove the <0.5 µs budget (criterion bench or traced timing in tests).
   - Expose adaptation/circuit-breaker stats via Prometheus or structured logs (`hnsw_dynamics_disabled_total`, activation density gauges, etc.).
   - Extend `vector_storage_integration` to assert that enabling dynamics improves recall/latency relative to a disabled control run.

3. **Task 004 – Columnar Cold Storage**
   - Introduce 64-byte aligned column buffers (AlignedColumn) instead of relying on `Vec<f32>` allocations.
   - Implement optional product-quantization/compression to hit the documented compression ratios.
   - Add criterion benchmarks to demonstrate the 10–100× analytical speedup target and reference them in docs.

4. **Task 005 – FAISS/Annoy Benchmarks**
   - Automate CSV/JSON export from the criterion harness so Engram vs FAISS vs Annoy results are versioned.
   - Fail CI when recall@10 < 0.90 or latency exceeds 1 ms by adding assertions around the exported metrics.
   - Publish the benchmark report (docs + dashboards) referenced in the original spec.

5. **Task 008 – Integration & Validation**
   - Tighten performance assertions to <1 ms and 90% recall once benchmarks confirm feasibility.
   - Add explicit confidence-calibration checks per tier (ECE thresholds) rather than logging-only validation.
   - (Optional) Fold FAISS/Annoy comparisons into the integration harness after Task 005 lands.

## Exit Criteria

Milestone 2 can be called complete when all five buckets above have code + tests merged, benchmarks recorded under `engram-core/benches/`, and CI enforces the documented SLOs.
