# Documenting Cognitive Recall: Turning Spreading Into a Usable API

*Perspective: Technical Communication*

Spreading activation is powerful, but only if developers understand how to configure, monitor, and reason about it. Task 013 focuses on packaging the new capabilities behind clear documentation, runnable examples, and visual tooling so teams can adopt cognitive recall confidently.

## Organizing the Story with Diátaxis
We structure docs into four pillars:
- **Tutorials** walk through enabling spreading in an existing project.
- **How-to guides** cover performance tuning, monitoring setup, and debugging activation flows.
- **Explanations** describe cognitive principles like semantic priming and the fan effect, linking them to configuration parameters.
- **Reference** lists API signatures and exhaustive configuration tables.

This separation keeps readers oriented—newcomers start with tutorials, experts jump straight to reference.

## Annotating the API with Cognitive Context
Rustdoc comments now pair each configuration field with both technical and cognitive explanations. Example: `max_hop_count` notes the computational complexity (branching_factor^hops) and the psychological rationale (2–4 hops mimic human associative distance). We use `#[doc = include_str!(...)]` to embed Markdown tables summarizing default values, ranges, and production recommendations.

## Runnable Examples
Three new examples demonstrate core recall patterns:
1. **Semantic priming** – show `NURSE` ranking higher after cue `DOCTOR`.
2. **Episodic reconstruction** – retrieve full memories from partial cues.
3. **Confidence-guided exploration** – interpret confidence scores when exploring deeper hops.

Examples live in `examples/` and run via `cargo run --example cognitive_recall_patterns`. CI executes them nightly to guarantee they compile and behave as described.

## Performance Tuning Guide
The tuning guide translates Task 010 metrics into concrete advice: parameter presets for low latency, high recall, and balanced modes; target metrics to watch; auto-tuning instructions. Each recommendation references benchmark data from the validation suite so readers know the advice is empirical.

## Visualization Tool
`SpreadingVisualizer` outputs GraphViz DOT files that color nodes by activation level and weight edges by confidence. Developers can render them with `dot -Tpng`. The docs include before/after screenshots showing how cycle protection changes activation flow, making abstract concepts tangible.

## Keeping Docs Fresh
Docs-as-code practices ensure longevity: markdown linting, doctests, broken-link checks, and versioned releases. Every API change updates a changelog entry, and feature flags are documented with stability levels. This discipline keeps documentation in lockstep with the codebase as Milestone 4 and beyond evolve spreading further.

## References
- Kostecki, D. "The Diátaxis documentation framework." (2020).
- Newell, A. *Unified Theories of Cognition.* (1990).
