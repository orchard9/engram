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
- **Tutorials** (`docs/tutorials/spreading_getting_started.md`): guide enabling spreading via `SpreadingConfig` / `RecallMode`.
- **How-to guides** (`docs/howto/spreading_performance.md`, `docs/howto/spreading_monitoring.md`) referencing metrics from Task 012.
- **Explanation** (`docs/explanation/cognitive_spreading.md`) covering semantic priming, fan effect, decay with links to Task 011 validation.
- **Reference** (`docs/reference/spreading_api.md`) documenting `SpreadingConfig`, `ParallelSpreadingConfig`, `GPUSpreadingInterface`.

### API Comments & Examples
- Annotate `SpreadingConfig` and `ParallelSpreadingConfig` in `activation/mod.rs` with cognitive context and performance implications. Use `#[doc = include_str!(...)]` to include Markdown tables.
- Add `examples/cognitive_recall_patterns.rs` demonstrating:
  1. Semantic priming (doctor → nurse)
  2. Episodic reconstruction from partial cues
  3. Confidence-guided exploration
  Run via `cargo run --example cognitive_recall_patterns` and ensure in CI.

### Visualization Tooling
- Implement `tools/spreading_visualizer.rs` (or `scripts/visualize_spreading.rs`) generating GraphViz DOT files from `ActivationResult`. Integrate with Task 005 cycle metadata and deterministic traces (Task 006).
- Document usage in `docs/howto/spreading_debugging.md` with sample images (`assets/spreading_example.png`).

### Performance Tuning Guide
- Publish `docs/howto/spreading_performance.md` summarizing Task 010’s tuning presets (low latency vs high recall) and mapping metrics (`engram_spreading_latency_seconds`, `engram_spreading_pool_utilization`).

### API Stability & Versioning
- Update `README.md` + `docs/changelog.md` to mark spreading API as **beta** until Milestone 4.
- Introduce feature flag `spreading_api_beta` in configuration files and note compatibility expectations.

## Acceptance Criteria
- [ ] Docs generated with `mdbook` or existing static site; linted via CI (markdownlint/Vale)
- [ ] `SpreadingConfig` Rustdoc includes parameter tables and cognitive context
- [ ] Example compile + run under CI (`cargo test --doc`, `cargo run --example cognitive_recall_patterns`)
- [ ] Visualization tool produces DOT/PNG illustrating activation heatmap and confidence edges
- [ ] Performance tuning guide references benchmark metrics from Task 010 and monitoring outputs from Task 012
- [ ] Changelog documents API stability level and migration notes

## Testing Approach
- Run doc tests (`cargo test --doc`) and example execution in CI
- Snapshot test for visualization output using `insta` (store DOT or rendered PNG hash)
- Link checker for docs (`lychee` or similar) to prevent rot

## Risk Mitigation
- **Docs drift** → treat docs as code: require PR review, run linting, enforce examples in CI
- **Visualization confusion** → provide legend and color-blind-friendly palette (WCAG 2.1 compliant)
- **API churn** → track changes in changelog and version configuration schema accordingly

## Notes
Relevant artifacts:
- `SpreadingConfig` (`engram-core/src/activation/mod.rs`)
- Metrics snapshots (Task 012) for performance guide
- Validation outputs (Task 011) for example narratives
