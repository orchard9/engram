# Contributing to Engram

Thank you for contributing to Engram! This document outlines the development workflow and code quality standards.

## Development Setup

### Quick Start

```bash
# Clone the repository
git clone https://github.com/orchard9/engram.git
cd engram

# Install git hooks for code quality enforcement
./scripts/setup-dev-hooks.sh

# Build the project
cargo build

# Run tests
cargo test
```

### Git Hooks

We use git pre-commit hooks to enforce code quality standards. The hook runs `make quality` before each commit, which includes:

- **Code formatting** (`cargo fmt`)
- **Linting** (`cargo clippy -- -D warnings`)
- **Tests** (`cargo test`)
- **Documentation** (builds and validates docs)

**Installing hooks:**
```bash
./scripts/setup-dev-hooks.sh
```

**Bypassing hooks (not recommended):**
```bash
git commit --no-verify
```

## Code Quality Standards

### Before Every Commit

Run `make quality` to ensure all checks pass:

```bash
make quality
```

This is **mandatory** - commits that don't pass quality checks will be rejected by CI.

### Formatting

We use `rustfmt` with default settings:

```bash
cargo fmt --all
```

### Linting

We enforce zero clippy warnings:

```bash
cargo clippy --workspace --all-targets --all-features -- -D warnings
```

**Clippy policy:**
- Fix all warnings before committing
- Use `#[allow(...)]` only when necessary with clear comments explaining why
- Test code can use `#[allow(clippy::unwrap_used)]` and `#[allow(clippy::float_cmp)]`

### Testing

All tests must pass:

```bash
cargo test --workspace
```

**Testing guidelines:**
- Write tests for all new functionality
- Aim for 80% code coverage
- Use property-based testing (proptest) for mathematical functions
- Integration tests for complete workflows

## Pull Request Process

1. Create a feature branch from `main`
2. Make your changes
3. Run `make quality` to verify all checks pass
4. Commit with clear, descriptive messages
5. Push and create a Pull Request
6. Address review feedback
7. Once approved, we'll merge your PR

## Commit Message Format

Use clear, concise commit messages:

```
type: Short description (50 chars or less)

Longer description if needed. Explain what changed and why,
not how (the diff shows how).

- Bullet points are fine
- Reference issues: Fixes #123
```

**Types:**
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style/formatting
- `refactor`: Code refactoring
- `test`: Adding/updating tests
- `chore`: Maintenance tasks

## Code Style

### Rust

- Follow Rust API guidelines: https://rust-lang.github.io/api-guidelines/
- Prefer explicit types over `impl Trait` in public APIs
- Document all public items with `///` doc comments
- Use `#[must_use]` for functions that return computed values

### Error Handling

- Use `thiserror` for error types
- Provide context with error messages
- Never use `.unwrap()` or `.expect()` in production code (tests OK)

### Performance

- Profile before optimizing
- Use benchmarks (`cargo bench`) to validate improvements
- Document performance characteristics in doc comments

## Architecture

See `vision.md` for system architecture and design principles.

Key principles:
- Biologically-inspired memory systems
- Zero-copy where possible
- Lock-free data structures for hot paths
- Graceful degradation under pressure

## Getting Help

- Read the documentation: `cargo doc --open`
- Check existing issues: https://github.com/orchard9/engram/issues
- Ask questions in discussions

## License

By contributing, you agree that your contributions will be licensed under the same license as Engram (MIT OR Apache-2.0).
