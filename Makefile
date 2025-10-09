.PHONY: fmt lint test quality

fmt:
	cargo fmt --all

lint:
	cargo clippy --workspace --all-targets --all-features -- -D warnings

test:
	cargo test --workspace -- --test-threads=1

quality: fmt lint test
