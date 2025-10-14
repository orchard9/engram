.PHONY: fmt lint test docs-lint example-cognitive quality

fmt:
	cargo fmt --all

lint:
	cargo clippy --workspace --all-targets --all-features -- -D warnings

test:
	cargo test --workspace -- --test-threads=1

docs-lint:
	npm --prefix docs ci --no-progress
	npm --prefix docs run lint

example-cognitive:
	cargo run -p engram-core --example cognitive_recall_patterns --features hnsw_index --quiet

quality: fmt lint test docs-lint example-cognitive
