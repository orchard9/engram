.PHONY: fmt lint test docs-lint example-cognitive quality
.PHONY: consolidation-soak

fmt:
	cargo fmt --all

lint:
	cargo clippy --workspace --all-targets --features "default,cognitive_tracing,pattern_completion" -- -D warnings

test:
	cargo test --workspace -- --test-threads=1

docs-lint:
	npm --prefix docs ci --no-progress
	npm --prefix docs run lint

example-cognitive:
	cargo run -p engram-core --example cognitive_recall_patterns --features hnsw_index --quiet

quality: fmt lint test docs-lint example-cognitive

consolidation-soak:
	cargo run --bin consolidation-soak -- --duration-secs 3600 --scheduler-interval-secs 60 --sample-interval-secs 60 --output-dir ./docs/assets/consolidation/baseline
