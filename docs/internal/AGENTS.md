# Repository Guidelines
## Project Structure & Module Organization
Workspace crates: `engram-core` (engine), `engram-storage` (persistence), `engram-cli` (CLI and orchestration), and `engram-proto` (gRPC/OpenAPI bindings under `proto/`). Benchmarks live in `engram-core/benches`, property seeds in `engram-core/proptest-regressions`, shared integration tests in each crate's `tests/`, and long-form guidance in `docs/` plus `content/`.
## Build, Test, and Development Commands
- `cargo build --workspace --features full` — feature-complete build.
- `cargo run -p engram-cli --bin engram` — launch the CLI/server locally.
- `cargo test --workspace` — full suite; add `-p engram-core -- --ignored` for slow cases.
- `cargo clippy --workspace --all-targets --all-features -D warnings` — lint respecting `clippy.toml`.
- `cargo fmt --all` — format before review.
- `./benchmark-startup.sh` — record cold-start metrics and archive generated HTML with perf notes.
## Coding Style & Naming Conventions
Target Rust 2024: avoid `unwrap`/`expect` outside tests, document every `unsafe`, and prefer `Arc<T>` for shared state. Follow Rust casing defaults (`PascalCase` types, `snake_case` functions, `SCREAMING_SNAKE_CASE` constants) and model domain IDs as newtypes. Keep modules single-purpose with minimal public APIs and doctested examples.
## Testing Guidelines
Co-locate unit tests with implementations; use crate-level `tests/` for integration and CLI smoke coverage. Exercise probabilistic logic with `proptest` or fuzzers from `engram-core/fuzz`, refreshing seeds in `engram-core/proptest-regressions` when behavior changes. Pair performance claims with `criterion` runs or startup benchmarks.
## Task Execution Workflow
Follow the "How to do a task" process in `CLAUDE.md` when working through roadmap items:
1. Read the task file carefully to confirm requirements and collect context from `vision.md` and `milestones.md`.
2. Rename the task file suffix from `_pending` to `_in_progress` before you start coding.
3. Plan tests up front using the Pareto principle—target the highest-impact cases first.
4. Implement the feature until the relevant tests pass, then review your changes against the architecture.
5. Iterate on any fixes found during review and add integration coverage as needed, again leaning on the Pareto principle.
6. Run `make quality` to ensure fmt, clippy, docs, and lint gates stay green.
7. Verify your implementation against the task requirements; if gaps remain, open a follow-up task in the same milestone with the original task name as a prefix.
8. Rename the task file from `_in_progress` to `_complete` once everything aligns.
9. Stage only the intended changes with `git status`, then commit.

If you hit a blocker at any point, rename the task file to `_blocked` and document the reason inside the task file before pausing.
## Commit & Pull Request Guidelines
Adopt Conventional Commit prefixes (`feat:`, `fix:`, `chore:`) with imperative subjects under 50 chars and bodies capturing intent. Link issues, call out feature flags touched, and attach relevant test or benchmark output. Confirm `cargo fmt`, `cargo clippy`, and `cargo test --workspace` succeed before requesting review; capture screenshots only when CLI UX changes.
## Reference Documents
- [README.md](README.md)
- [coding_guidelines.md](coding_guidelines.md)
- [CLAUDE.md](CLAUDE.md)
- [.claude/agents/cognitive-architecture-designer.md](.claude/agents/cognitive-architecture-designer.md)
- [.claude/agents/documentation-validator.md](.claude/agents/documentation-validator.md)
- [.claude/agents/gpu-acceleration-architect.md](.claude/agents/gpu-acceleration-architect.md)
- [.claude/agents/graph-systems-acceptance-tester.md](.claude/agents/graph-systems-acceptance-tester.md)
- [.claude/agents/memory-systems-researcher.md](.claude/agents/memory-systems-researcher.md)
- [.claude/agents/rust-graph-engine-architect.md](.claude/agents/rust-graph-engine-architect.md)
- [.claude/agents/systems-architecture-optimizer.md](.claude/agents/systems-architecture-optimizer.md)
- [.claude/agents/systems-product-planner.md](.claude/agents/systems-product-planner.md)
- [.claude/agents/technical-communication-lead.md](.claude/agents/technical-communication-lead.md)
- [.claude/agents/utoipa-documentation-expert.md](.claude/agents/utoipa-documentation-expert.md)
- [.claude/agents/verification-testing-lead.md](.claude/agents/verification-testing-lead.md)
## Agent Selection
Pick the agent whose specialization matches your task, then open the detailed brief in `/Users/jordanwashburn/Workspace/orchard9/engram/.claude/agents` before you begin to ensure the guidance is fresh.
## Agents
cognitive-architecture-designer: Use for designing memory consolidation algorithms, System 2 reasoning, or biologically-inspired neural systems
systems-architecture-optimizer: Use for low-level tiered storage design, lock-free data structures, cache optimization, or NUMA performance
gpu-acceleration-architect: Use for CUDA kernel implementation, GPU memory patterns, or parallel graph algorithm optimization
rust-graph-engine-architect: Use for high-performance graph engine design, concurrent data structures, cache-optimal algorithms, or probabilistic operations
verification-testing-lead: Use for differential testing between Rust/Zig implementations, fuzzing harnesses, formal verification, or validating algorithmic correctness
systems-product-planner: Use for defining technical roadmaps, creating implementation specs, prioritizing features, or making architectural decisions
technical-communication-lead: Use for explaining complex Engram concepts to external audiences, creating developer documentation, or writing blog posts about the project
memory-systems-researcher: Use for validating memory consolidation algorithms, implementing hippocampal-neocortical interactions, or ensuring biological plausibility
graph-systems-acceptance-tester: Use for validating graph database functionality, testing spreading activation algorithms, verifying memory consolidation behaviors, or ensuring API compatibility
## Security & Configuration Tips
macOS contributors enabling SMT verification must export the Z3 environment variables in `README.md` and run `brew install z3`. Document new secrets or ports in `engram-cli` configs and prefer async or memory-mapped paths via `engram-storage` for hot operations.
