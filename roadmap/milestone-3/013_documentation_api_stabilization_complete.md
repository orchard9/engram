# Task 013: Documentation and API Stabilization

## Objective
Document the spreading activation API, provide runnable examples, and ship visualization/debug tooling so teams can adopt cognitive recall safely.

## Priority
P2 (Quality Enhancement)

## Effort Estimate
0.5 days

## Dependencies
- Task 012: Production Integration and Monitoring

## Technical Approach

### Documentation Structure (Diátaxis)
- Add Diátaxis directories under `docs/` (`tutorials/`, `howto/`, `explanation/`, `reference/`) and wire them into the VitePress sidebar/nav in `docs/.vitepress/config.js` so the spreading docs sit beside the existing `getting-started` and `api` sections.
- `docs/tutorials/spreading_getting_started.md`: walkthrough that starts from `engram-cli` defaults, shows enabling `RecallMode::Hybrid` via `RecallConfig` (`engram-core/src/activation/recall.rs`) and, when Task 012 lands, how to toggle monitoring/visualization flags in the CLI; embed copy/pastable snippets for `cargo run -p engram-cli -- start` + example recall requests.
- `docs/howto/spreading_performance.md`: consolidate Task 010 metrics (latency presets, `ParallelSpreadingConfig::num_threads`, `decay_function`) with Task 012 telemetry (`engram_spreading_latency_seconds`, `engram_spreading_pool_utilization`) pulled from `engram-core/src/activation/mod.rs` and the Prometheus exporter; include decision tables that point to production thresholds.
- `docs/howto/spreading_monitoring.md`: document operator runbooks that reference `engram-cli/src/api.rs` SSE feeds and `SpreadingMetrics::record_pool_snapshot`; describe alert hooks that Task 012 contributes and how to validate via `curl /metrics`.
- `docs/explanation/cognitive_spreading.md`: describe semantic priming, fan effect, decay curves with diagrams rendered from the new visualizer output; link directly to `TraceEntry` (`engram-core/src/activation/mod.rs`) and Task 011 validation artifacts in `docs/assets`.
- `docs/reference/spreading_api.md`: tabulate field-by-field docs for `SpreadingConfig`, `ParallelSpreadingConfig`, and `GPUSpreadingInterface`, linking back to the Rustdoc tables added below; include feature gating call-outs (`#[doc(cfg(feature = "hnsw_index"))]`) so users know when APIs are available.
- Capture static assets (`docs/assets/spreading_example.png`, `docs/assets/spreading_legend.svg`) and reference them from tutorials/how-tos; add alt text that satisfies WCAG guidance noted in the research packet.

### API Comments & Examples
- Create `engram-core/src/activation/doc/parallel_spreading_config.md`, `.../hnsw_spreading_config.md`, and `.../gpu_interface.md` containing Markdown tables (columns: field, default, recommended range, cognitive rationale) and include them via `#[doc = include_str!("doc/parallel_spreading_config.md")]` on the respective structs/traits (`ParallelSpreadingConfig` in `activation/mod.rs`, `SpreadingConfig` in `activation/hnsw_integration.rs`, `GPUSpreadingInterface` in `activation/gpu_interface.rs`).
- Add `#[doc(cfg(feature = "hnsw_index"))]` on the HNSW-specific config so docs.rs shows the gating, and expand field docs to explain how decay / hop limits align with semantic distance from Task 011 experiments.
- Rebuild `engram-core/examples/cognitive_recall_patterns.rs` to construct a minimal `MemoryStore`, populate a graph with cues (`doctor`, `nurse`, episodic fragments) via `ActivationGraphExt::add_edge`, and run three distinct spreads: semantic priming, episodic reconstruction (partial cues filling in via recall), and confidence-guided exploration (sorting by `RankedMemory::confidence`).
- Keep the example deterministic by using `ParallelSpreadingConfig::deterministic(seed)` and enabling `trace_activation_flow`; print both the ranked results and a hint about generating a DOT file with the new visualizer.
- Update workspace automation so CI runs `cargo run -p engram-core --example cognitive_recall_patterns --features hnsw_index` (nightly or as part of `make quality`) and fails fast if the example output changes.
- Add a short doctest example to `RecallMode`/`RecallConfig` illustrating how the tutorial’s configuration lines map to Rust API calls.

### Visualization Tooling
- Add a reusable helper (`engram-core/src/activation/visualization.rs`) that transforms `SpreadingResults` + `TraceEntry` into a GraphViz DOT string (node color by activation heat, edge width by confidence, legend metadata baked into comments) so both the docs site and CLI tooling share identical output.
- Create `engram-cli/examples/spreading_visualizer.rs` (binary example) that: runs a seeded recall via `CognitiveRecallBuilder` with `trace_activation_flow = true`, writes the DOT file to disk, optionally shells out to `dot -Tpng` when GraphViz is installed, and reports the output path. Accept `--input-trace` for piping stored JSON traces later.
- Provide fixtures under `docs/assets/spreading/trace_samples/` (serialized `SpreadingResults` with deterministic seeds) so the visualizer can be demonstrated without rebuilding the entire workload.
- Document the workflow in `docs/howto/spreading_debugging.md`: enable tracing (`ParallelSpreadingConfig::trace_activation_flow`), run the visualizer, view rendered PNG/SVG; include CLI/session recordings for reference.
- Ensure Task 005 cycle metadata and Task 006 deterministic traces are surfaced by importing `cycle_paths`/`deterministic_trace` from `SpreadingResults`; add tests under `engram-core/tests/spreading_visualization.rs` using `insta` to guard the generated DOT (whitespace-stripped) against regressions.

### Performance Tuning Guide
- Base the tuning guide on real metrics captured in `task_010_alignment_report.md` and Task 012 monitoring: highlight presets for low-latency (`max_depth=2`, aggressive `threshold`), balanced, and high-recall (`max_depth=4`, slower decay) by referencing concrete `ParallelSpreadingConfig` snippets.
- Include dashboards (Grafana JSON from Task 012) and explain how to interpret `SpreadingMetrics::pool_hit_rate`, `adaptive_*` counters, and Prometheus histograms; embed screenshots saved under `docs/assets/spreading_performance/`.
- Add a troubleshooting section mapping warning signs (e.g., `latency_budget_violations` spikes) to remediation steps (tighten `batch_size`, enable circuit breaker) and cross-link to monitoring/how-to docs.
- Provide comparison tables that show before/after metric deltas when toggling `spreading_api_beta`, making it obvious how operators should phase the rollout.

### API Stability & Versioning
- Introduce a concrete CLI/world configuration: create `engram-cli/config/default.toml` with a `[feature_flags] spreading_api_beta = true` stanza, and load/merge it in a new `engram-cli/src/config.rs` helper that `handle_config_command` uses (persist overrides in `~/.config/engram/config.toml`).
- Wire the feature flag through `engram-cli/src/main.rs` so starting the server logs whether spreading is in beta; expose a `ConfigAction::Set` path that flips the flag and writes back to disk.
- Update `README.md` (Feature matrix + Troubleshooting) and create `docs/changelog.md` entry announcing the beta status, supported scenarios, and upgrade checklist for Milestone 4 stabilization.
- Add compatibility notes to `engram-core/src/lib.rs` docs and `docs/reference/spreading_api.md` clarifying that GPU backends remain experimental until the flag graduates.
- Extend `FEATURES.md` / `coding_guidelines.md` with guidance on documenting future API toggles, ensuring release notes and docs stay synchronized.

## Acceptance Criteria
- [ ] `npm --prefix docs run build` and the new `npm --prefix docs run lint` pass locally and in CI, proving the Diátaxis pages render with the updated navigation.
- [ ] `cargo doc --package engram-core --features hnsw_index --no-deps` shows injected Markdown tables for `ParallelSpreadingConfig`, `SpreadingConfig`, and `GPUSpreadingInterface` (verified via `rg` or docs.rs screenshot) without new Clippy warnings.
- [ ] `cargo run -p engram-core --example cognitive_recall_patterns --features hnsw_index` executes inside CI (hooked into `make quality` or a dedicated workflow) and prints the three documented scenarios.
- [ ] `cargo run -p engram-cli --example spreading_visualizer --features hnsw_index -- --output ./target/spread.dot` succeeds and the generated DOT matches the checked-in `insta` snapshot (ensuring deterministic visualization output).
- [ ] `lychee --config docs/.lychee.toml docs/**/*.md README.md` (or equivalent script) reports zero broken links, and snapshot tests for the visualizer pass (`cargo test -p engram-core visualization -- --ignored` if needed).
- [ ] `docs/changelog.md` + `README.md` clearly mark the spreading API as beta with instructions for flipping `spreading_api_beta`, and the CLI config commands can read/write the flag.

## Testing Approach
- Add a `cargo test -p engram-core --doc` gate plus a dedicated `cargo test -p engram-core visualization -- --ignored` that exercises the DOT snapshot.
- Extend `Makefile` (or introduce `.github/workflows/docs.yml`) to run: `npm --prefix docs ci`, `cargo fmt`, `cargo clippy --workspace --all-targets --all-features`, `cargo run -p engram-core --example cognitive_recall_patterns --features hnsw_index`, and the new link checker.
- For the CLI config loader, add unit tests under `engram-cli/tests/config_tests.rs` to verify reading defaults, persisting overrides, and toggling `spreading_api_beta`.
- Use `insta` snapshots for both DOT and rendered PNG metadata (`target/spread.png.meta`) so regressions surface clearly.
- Smoke-test the tutorial flow by running `cargo run -p engram-cli -- start` + example recall requests with the feature flag disabled/enabled, capturing outputs for docs.

## Risk Mitigation
- **Docs drift** → pin doc builds + linting in `make quality`, require reviewers to check `lychee` + `insta` updates, and source Markdown tables from the same files embedded into Rustdoc to avoid duplication.
- **Visualization confusion** → ship a legend + accessibility callouts directly in the DOT footer; add a docs section on interpreting activation mass vs. confidence so operators know what they are reviewing.
- **API churn** → enforce that `spreading_api_beta` toggles are recorded in `docs/changelog.md` and `FEATURES.md`, and fail CI if the config helper is missing entries for new spread-related knobs.
- **Tooling instability** → run visualizer snapshot tests with deterministic seeds and guard GraphViz shell-out behind a feature detection (skip gracefully in CI environments without GraphViz).

## Notes
- Relevant artifacts: `engram-core/src/activation/mod.rs`, `engram-core/src/activation/hnsw_integration.rs`, `engram-core/examples/cognitive_recall_patterns.rs`
- GPU + visualization references: `engram-core/src/activation/gpu_interface.rs`, planned `engram-core/src/activation/visualization.rs`
- CLI + config touchpoints: `engram-cli/src/main.rs`, `engram-cli/src/config.rs` (new), `engram-cli/src/cli/commands.rs`
- Documentation plumbing: `docs/.vitepress/config.js`, `docs/assets/`, Diátaxis Markdown files to add under `docs/tutorials|howto|explanation|reference`
- Performance data sources: `task_010_alignment_report.md`, Task 012 monitoring outputs, Grafana/Prometheus artifacts committed with Task 012
