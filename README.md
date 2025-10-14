# Engram

A high-performance cognitive graph database with biologically-inspired memory consolidation and probabilistic query processing.

## Overview

Engram combines cognitive science principles with modern systems engineering to create a graph database that mimics human memory patterns. It features:

- **Biologically-Inspired Architecture**: Memory consolidation, forgetting curves, and spreading activation
- **Probabilistic Operations**: Confidence-based queries with uncertainty propagation
- **High Performance**: Lock-free concurrent data structures and SIMD-optimized operations
- **Graceful Degradation**: Robust error handling with automatic recovery strategies

## Quick Start

### Prerequisites

- Rust 1.75+ with Edition 2021
- For SMT verification features: Z3 SMT Solver

### macOS Setup (Required for Z3 Features)

On macOS, you need to set up environment variables for the Z3 SMT solver dependency. Add these to your shell profile (`.zshrc`, `.bash_profile`, etc.):

```bash
# Z3 SMT Solver environment variables for Rust z3-sys crate
export Z3_SYS_Z3_HEADER="/opt/homebrew/include/z3.h"
export PKG_CONFIG_PATH="/opt/homebrew/lib/pkgconfig:$PKG_CONFIG_PATH"
export LIBRARY_PATH="/opt/homebrew/lib:$LIBRARY_PATH"
export DYLD_LIBRARY_PATH="/opt/homebrew/lib:$DYLD_LIBRARY_PATH"
export BINDGEN_EXTRA_CLANG_ARGS="-I/opt/homebrew/include"
```

Then install Z3:

```bash
brew install z3
```

### Building

```bash
# Clone the repository
git clone <repository-url>
cd engram

# Build with default features
cargo build --release

# Build with all features
cargo build --release --features full

# Build without optional features
cargo build --release --no-default-features
```

### Running Tests

```bash
# Run all tests
cargo test

# Run specific test suite
cargo test --test error_recovery_integration

# Run with specific features
cargo test --features "hnsw_index,psychological_decay"
```

## Architecture

### Core Components

- **`engram-core`**: Core graph engine and memory operations
- **`engram-cli`**: Command-line interface and server
- **Memory Systems**: Episodic and semantic memory with consolidation
- **Probabilistic Queries**: Uncertainty-aware search and retrieval
- **Error Recovery**: Production-ready error handling with graceful degradation

### Features

- **`hnsw_index`**: High-performance approximate nearest neighbor search
- **`memory_mapped_persistence`**: Memory-mapped storage with NUMA awareness
- **`psychological_decay`**: Biologically-inspired forgetting curves
- **`pattern_completion`**: Neural-inspired pattern completion
- **`probabilistic_queries`**: Confidence-based query processing
- **`monitoring`**: Built-in streaming metrics and structured observability logs
- **`smt_verification`**: SMT-based correctness verification (requires Z3)
- **Spreading Activation (beta)**: Enabled via the `spreading_api_beta` feature flag. Toggle with `engram config set feature_flags.spreading_api_beta <true|false>`.

## Development

### Error Handling

Engram uses a comprehensive error handling system with recovery strategies:

```rust
use engram_core::error::{EngramError, RecoveryStrategy, ErrorRecovery};

// Automatic retry with exponential backoff
let result = ErrorRecovery::with_retry(
    || async { 
        // Your operation here
        store.write_episode(&episode)
    },
    RecoveryStrategy::Retry {
        max_attempts: 3,
        backoff_ms: 100,
    },
).await?;

// Graceful fallback
let result = ErrorRecovery::with_fallback(
    || hnsw_search(&query),      // Try HNSW first
    || linear_search(&query),    // Fallback to linear search
)?;
```

### Safety Guarantees

The codebase enforces safety through Clippy lints:

```rust
#![warn(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
#![deny(clippy::unwrap_in_result, clippy::panic_in_result_fn)]
```

Use the provided migration tools to find and fix unsafe patterns:

```bash
python scripts/fix_unwraps.py src/
```

### Testing

Comprehensive test coverage includes:
- Unit tests for individual components
- Integration tests for feature combinations  
- Property-based testing for probabilistic operations
- Error recovery scenario testing
- Performance benchmarks

### Documentation

- [Vision and Architecture](vision.md)
- [Milestone Planning](roadmap/)
- [API Documentation](https://docs.rs/engram-core)
- [Examples](examples/)
- [Changelog](docs/changelog.md)

## Contributing

1. Read the [coding guidelines](coding_guidelines.md)
2. Check the [current milestones](roadmap/)
3. Run tests and ensure they pass
4. Follow the error handling patterns
5. Submit pull requests with clear descriptions

## License

[License information]

## Troubleshooting

### Build Issues on macOS

If you encounter Z3-related build errors on macOS:

1. Ensure Z3 is installed: `brew install z3`
2. Set the environment variables listed above
3. Restart your terminal
4. Try building again

For other build issues, check:
- Rust version: `rustc --version` (should be 1.75+)
- Available features: `cargo build --help`
- Clean build: `cargo clean && cargo build`

### Feature Compatibility

Some features may not be compatible:
- Run `cargo check --features <feature>` to test individual features
- See [feature compatibility matrix](roadmap/milestone-1/) for details

### Performance

For optimal performance:
- Use `--release` builds in production
- Enable appropriate features for your use case
- Consider NUMA topology for large deployments
