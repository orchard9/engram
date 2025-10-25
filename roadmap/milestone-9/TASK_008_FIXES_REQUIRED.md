# Task 008: Query Validation Suite - Required Fixes Summary

**Review Date**: 2025-10-25
**Reviewer**: Professor John Regehr
**Status**: REQUIRES IMMEDIATE ACTION

This document provides specific fixes needed to complete Task 008 properly.

---

## Critical Finding: Silent Clamping in Confidence::exact()

**Root Cause**: `/Users/jordanwashburn/Workspace/orchard9/engram/engram-core/src/lib.rs:101-106`

The `Confidence::exact()` function **silently clamps** values to [0,1]:

```rust
/// Creates exact confidence value with automatic clamping to 0..=1
#[must_use]
pub const fn exact(value: f32) -> Self {
    // Const-compatible clamping
    // ... implementation clamps to [0,1]
}
```

**Impact**: The parser uses `Confidence::exact()` everywhere, causing all out-of-range values to be silently accepted and clamped instead of rejected with validation errors.

**Solution**: Parser must validate BEFORE calling `Confidence::exact()`. Add validation layer in parser.

---

##Fix 1: Add Parser Validation Functions

Create a new file: `/Users/jordanwashburn/Workspace/orchard9/engram/engram-core/src/query/parser/validation.rs`

```rust
//! Value validation for query parser.
//!
//! These functions perform range validation BEFORE constructing AST types,
//! ensuring that invalid inputs are rejected with clear error messages rather
//! than silently clamped.

use super::error::{ParseError, ParserContext};
use super::token::Position;
use crate::Confidence;

/// Validate confidence value is in range [0.0, 1.0].
///
/// Returns ParseError if out of range, otherwise returns valid Confidence.
pub fn validate_confidence(
    value: f32,
    position: Position,
) -> Result<Confidence, ParseError> {
    if value < 0.0 {
        return Err(ParseError::validation_error(
            format!("Confidence value {value} is negative"),
            position,
            "Confidence must be between 0.0 and 1.0",
            "WHERE confidence > 0.7",
        ));
    }
    if value > 1.0 {
        return Err(ParseError::validation_error(
            format!("Confidence value {value} exceeds 1.0"),
            position,
            "Confidence must be between 0.0 and 1.0",
            "WHERE confidence > 0.7",
        ));
    }
    Ok(Confidence::exact(value))
}

/// Validate MAX_HOPS is in range [1, 100].
pub fn validate_max_hops(
    value: u16,
    position: Position,
) -> Result<u16, ParseError> {
    if value == 0 {
        return Err(ParseError::validation_error(
            "MAX_HOPS cannot be zero",
            position,
            "MAX_HOPS must be at least 1 (use value between 1 and 100)",
            "SPREAD FROM node MAX_HOPS 5",
        ));
    }
    if value > 100 {
        return Err(ParseError::validation_error(
            format!("MAX_HOPS value {value} exceeds maximum of 100"),
            position,
            "MAX_HOPS must be between 1 and 100 for performance reasons",
            "SPREAD FROM node MAX_HOPS 50",
        ));
    }
    Ok(value)
}

/// Validate decay rate is in range [0.0, 1.0].
pub fn validate_decay_rate(
    value: f32,
    position: Position,
) -> Result<f32, ParseError> {
    if value < 0.0 {
        return Err(ParseError::validation_error(
            format!("DECAY rate {value} is negative"),
            position,
            "Decay rate must be between 0.0 and 1.0",
            "SPREAD FROM node DECAY 0.15",
        ));
    }
    if value > 1.0 {
        return Err(ParseError::validation_error(
            format!("DECAY rate {value} exceeds 1.0"),
            position,
            "Decay rate must be between 0.0 and 1.0",
            "SPREAD FROM node DECAY 0.15",
        ));
    }
    Ok(value)
}

/// Validate threshold is in range [0.0, 1.0].
pub fn validate_threshold(
    value: f32,
    position: Position,
) -> Result<f32, ParseError> {
    if value < 0.0 {
        return Err(ParseError::validation_error(
            format!("THRESHOLD value {value} is negative"),
            position,
            "Threshold must be between 0.0 and 1.0",
            "SPREAD FROM node THRESHOLD 0.1",
        ));
    }
    if value > 1.0 {
        return Err(ParseError::validation_error(
            format!("THRESHOLD value {value} exceeds 1.0"),
            position,
            "Threshold must be between 0.0 and 1.0",
            "SPREAD FROM node THRESHOLD 0.1",
        ));
    }
    Ok(value)
}

/// Validate novelty is in range [0.0, 1.0].
pub fn validate_novelty(
    value: f32,
    position: Position,
) -> Result<f32, ParseError> {
    if value < 0.0 {
        return Err(ParseError::validation_error(
            format!("NOVELTY value {value} is negative"),
            position,
            "Novelty must be between 0.0 and 1.0",
            "IMAGINE episode BASED ON seed NOVELTY 0.3",
        ));
    }
    if value > 1.0 {
        return Err(ParseError::validation_error(
            format!("NOVELTY value {value} exceeds 1.0"),
            position,
            "Novelty must be between 0.0 and 1.0",
            "IMAGINE episode BASED ON seed NOVELTY 0.3",
        ));
    }
    Ok(value)
}

/// Validate base rate is in range [0.0, 1.0].
pub fn validate_base_rate(
    value: f32,
    position: Position,
) -> Result<f32, ParseError> {
    if value < 0.0 {
        return Err(ParseError::validation_error(
            format!("BASE_RATE value {value} is negative"),
            position,
            "Base rate must be between 0.0 and 1.0",
            "RECALL episode WITH BASE_RATE 0.1",
        ));
    }
    if value > 1.0 {
        return Err(ParseError::validation_error(
            format!("BASE_RATE value {value} exceeds 1.0"),
            position,
            "Base rate must be between 0.0 and 1.0",
            "RECALL episode WITH BASE_RATE 0.1",
        ));
    }
    Ok(value)
}

/// Validate limit is positive.
pub fn validate_limit(
    value: i64,
    position: Position,
) -> Result<usize, ParseError> {
    if value <= 0 {
        return Err(ParseError::validation_error(
            format!("LIMIT value {value} is not positive"),
            position,
            "Limit must be at least 1",
            "RECALL episode LIMIT 10",
        ));
    }
    Ok(value as usize)
}

/// Validate horizon is non-negative.
pub fn validate_horizon(
    value: i64,
    position: Position,
) -> Result<u64, ParseError> {
    if value < 0 {
        return Err(ParseError::validation_error(
            format!("HORIZON value {value} is negative"),
            position,
            "Horizon must be zero or positive (seconds)",
            "PREDICT episode GIVEN context HORIZON 3600",
        ));
    }
    Ok(value as u64)
}

/// Validate scheduler interval is positive.
pub fn validate_scheduler_interval(
    value: i64,
    position: Position,
) -> Result<u64, ParseError> {
    if value <= 0 {
        return Err(ParseError::validation_error(
            format!("Scheduler interval {value} is not positive"),
            position,
            "Scheduler interval must be at least 1 second",
            "CONSOLIDATE episodes INTO semantic SCHEDULER interval 3600",
        ));
    }
    Ok(value as u64)
}

/// Validate scheduler threshold is in range [0.0, 1.0].
pub fn validate_scheduler_threshold(
    value: f32,
    position: Position,
) -> Result<f32, ParseError> {
    if value < 0.0 {
        return Err(ParseError::validation_error(
            format!("Scheduler threshold {value} is negative"),
            position,
            "Scheduler threshold must be between 0.0 and 1.0",
            "CONSOLIDATE episodes INTO semantic SCHEDULER threshold 0.5",
        ));
    }
    if value > 1.0 {
        return Err(ParseError::validation_error(
            format!("Scheduler threshold {value} exceeds 1.0"),
            position,
            "Scheduler threshold must be between 0.0 and 1.0",
            "CONSOLIDATE episodes INTO semantic SCHEDULER threshold 0.5",
        ));
    }
    Ok(value)
}

/// Validate identifier is not just underscores.
pub fn validate_identifier(
    name: &str,
    position: Position,
) -> Result<&str, ParseError> {
    if name.chars().all(|c| c == '_') {
        return Err(ParseError::invalid_syntax(
            "Identifier cannot consist only of underscores",
            position,
            "Use alphanumeric characters in identifier",
            "RECALL episode_123",
        ));
    }
    if name.len() > 1000 {
        return Err(ParseError::validation_error(
            format!("Identifier too long: {} characters", name.len()),
            position,
            "Identifier must be shorter than 1000 characters",
            "RECALL episode_with_reasonable_name",
        ));
    }
    Ok(name)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_validate_confidence_valid() {
        let pos = Position::new(0, 1, 1);
        assert!(validate_confidence(0.0, pos).is_ok());
        assert!(validate_confidence(0.5, pos).is_ok());
        assert!(validate_confidence(1.0, pos).is_ok());
    }

    #[test]
    fn test_validate_confidence_negative() {
        let pos = Position::new(0, 1, 1);
        let err = validate_confidence(-0.5, pos).unwrap_err();
        assert!(err.to_string().contains("negative"));
    }

    #[test]
    fn test_validate_confidence_too_high() {
        let pos = Position::new(0, 1, 1);
        let err = validate_confidence(1.5, pos).unwrap_err();
        assert!(err.to_string().contains("1.0"));
    }

    #[test]
    fn test_validate_max_hops_valid() {
        let pos = Position::new(0, 1, 1);
        assert!(validate_max_hops(1, pos).is_ok());
        assert!(validate_max_hops(50, pos).is_ok());
        assert!(validate_max_hops(100, pos).is_ok());
    }

    #[test]
    fn test_validate_max_hops_zero() {
        let pos = Position::new(0, 1, 1);
        let err = validate_max_hops(0, pos).unwrap_err();
        assert!(err.to_string().contains("zero"));
    }

    #[test]
    fn test_validate_max_hops_too_large() {
        let pos = Position::new(0, 1, 1);
        let err = validate_max_hops(1000, pos).unwrap_err();
        assert!(err.to_string().contains("100"));
    }

    #[test]
    fn test_validate_identifier_underscores_only() {
        let pos = Position::new(0, 1, 1);
        let err = validate_identifier("____", pos).unwrap_err();
        assert!(err.to_string().contains("underscores"));
    }

    #[test]
    fn test_validate_identifier_too_long() {
        let pos = Position::new(0, 1, 1);
        let long_name = "a".repeat(1001);
        let err = validate_identifier(&long_name, pos).unwrap_err();
        assert!(err.to_string().contains("long"));
    }
}
```

---

## Fix 2: Update parser.rs to Use Validation

In `/Users/jordanwashburn/Workspace/orchard9/engram/engram-core/src/query/parser/parser.rs`:

1. Add at top:
```rust
mod validation;
use validation::*;
```

2. Replace all direct `Confidence::exact()` calls with `validate_confidence()`
3. Replace all direct parameter assignments with validation calls

Example changes:

```rust
// BEFORE:
let confidence = Confidence::exact(self.parse_float()?);

// AFTER:
let value = self.parse_float()?;
let position = self.position();
let confidence = validate_confidence(value, position)?;
```

```rust
// BEFORE:
max_hops = Some(u16::try_from(hops).map_err(...)?);

// AFTER:
let hops_u16 = u16::try_from(hops).map_err(...)?;
let position = self.position();
max_hops = Some(validate_max_hops(hops_u16, position)?);
```

---

## Fix 3: Add Constraint Validation

In `/Users/jordanwashburn/Workspace/orchard9/engram/engram-core/src/query/parser/validation.rs`, add:

```rust
use std::collections::HashMap;

/// Validate constraints don't have duplicates or contradictions.
pub fn validate_constraints(
    constraints: &[Constraint],
    position: Position,
) -> Result<(), ParseError> {
    // Check for duplicates
    let mut field_counts = HashMap::new();
    for constraint in constraints {
        let field_name = match constraint {
            Constraint::ConfidenceAbove(_) | Constraint::ConfidenceBelow(_) | Constraint::ConfidenceEquals(_) => "confidence",
            Constraint::CreatedAfter(_) | Constraint::CreatedBefore(_) => "created",
            Constraint::MemorySpace(_) => "memory_space",
            Constraint::ContentMatch(_) => "content",
            Constraint::SimilarTo { .. } => "content_similarity",
        };

        *field_counts.entry(field_name).or_insert(0) += 1;
    }

    // Report duplicates
    for (field, count) in &field_counts {
        if *count > 1 {
            return Err(ParseError::validation_error(
                format!("Duplicate constraint on field '{field}' ({count} times)"),
                position,
                format!("Use a single {field} constraint instead of {count}"),
                "WHERE confidence > 0.7 AND created > \"2024-01-01\"",
            ));
        }
    }

    // Check for contradictions (confidence > X AND confidence < Y where X > Y)
    let confidence_above = constraints.iter().find_map(|c| {
        if let Constraint::ConfidenceAbove(conf) = c {
            Some(conf.raw())
        } else {
            None
        }
    });

    let confidence_below = constraints.iter().find_map(|c| {
        if let Constraint::ConfidenceBelow(conf) = c {
            Some(conf.raw())
        } else {
            None
        }
    });

    if let (Some(above), Some(below)) = (confidence_above, confidence_below) {
        if above >= below {
            return Err(ParseError::validation_error(
                format!(
                    "Contradictory confidence constraints: > {above} AND < {below}"
                ),
                position,
                "Ensure lower bound is less than upper bound",
                "WHERE confidence > 0.3 AND confidence < 0.9",
            ));
        }
    }

    // Limit total constraint count
    if constraints.len() > 10 {
        return Err(ParseError::validation_error(
            format!("Too many constraints: {} (maximum 10)", constraints.len()),
            position,
            "Simplify query by reducing number of constraints",
            "WHERE confidence > 0.7 AND created > \"2024-01-01\"",
        ));
    }

    Ok(())
}
```

---

## Fix 4: Update mod.rs

In `/Users/jordanwashburn/Workspace/orchard9/engram/engram-core/src/query/parser/mod.rs`:

```rust
pub mod validation;
pub use validation::*;
```

---

## Fix 5: Clippy Fixes for recall.rs

In `/Users/jordanwashburn/Workspace/orchard9/engram/engram-core/src/query/executor/recall.rs`:

Replace all `.unwrap()` and `.expect()` with proper error propagation:

```rust
// BEFORE (line 501):
let filtered = executor
    .apply_single_constraint(episodes, &constraint)
    .unwrap();

// AFTER:
let filtered = executor
    .apply_single_constraint(episodes, &constraint)
    .map_err(|e| {
        QueryExecutionError::ConstraintError {
            constraint: format!("{:?}", constraint),
            source: Box::new(e),
        }
    })?;
```

**Note**: This requires adding a new error variant to `QueryExecutionError` if it doesn't exist:

```rust
#[derive(Debug, thiserror::Error)]
pub enum QueryExecutionError {
    // ... existing variants

    #[error("Failed to apply constraint {constraint}")]
    ConstraintError {
        constraint: String,
        #[source]
        source: Box<dyn std::error::Error + Send + Sync>,
    },
}
```

---

## Fix 6: Clippy Fixes for spread.rs

In `/Users/jordanwashburn/Workspace/orchard9/engram/engram-core/src/query/executor/spread.rs`:

Add to `Cargo.toml` dependencies:
```toml
approx = "0.5"
```

Replace float comparisons:

```rust
// BEFORE (line 336):
assert_eq!(
    query.effective_decay_rate(),
    SpreadQuery::DEFAULT_DECAY_RATE
);

// AFTER:
use approx::assert_relative_eq;
const EPSILON: f32 = 1e-6;
assert_relative_eq!(
    query.effective_decay_rate(),
    SpreadQuery::DEFAULT_DECAY_RATE,
    epsilon = EPSILON
);
```

Or without external crate:

```rust
const EPSILON: f32 = 1e-6;
assert!(
    (query.effective_decay_rate() - SpreadQuery::DEFAULT_DECAY_RATE).abs() < EPSILON,
    "Expected {}, got {}",
    SpreadQuery::DEFAULT_DECAY_RATE,
    query.effective_decay_rate()
);
```

---

## Fix 7: Remove Unused Imports

In `/Users/jordanwashburn/Workspace/orchard9/engram/engram-core/tests/error_message_validation.rs`:

```rust
// Remove line 19:
use engram_core::query::parser::{ParseError, Parser};
// Keep only:
use engram_core::query::parser::Parser;

// Remove line 22:
use query_language_corpus::{ErrorType, QueryCorpus};
// Keep only:
use query_language_corpus::QueryCorpus;
```

In `/Users/jordanwashburn/Workspace/orchard9/engram/engram-core/tests/query_language_corpus.rs`:

Add `#[allow(dead_code)]` to unused test structs that are used by other test files.

---

## Fix 8: Implement Missing Infrastructure

### 8a. Fuzzing Setup

Create `/Users/jordanwashburn/Workspace/orchard9/engram/engram-core/fuzz/Cargo.toml`:

```toml
[package]
name = "engram-core-fuzz"
version = "0.0.0"
publish = false
edition = "2024"

[package.metadata]
cargo-fuzz = true

[dependencies]
libfuzzer-sys = "0.4"
engram-core = { path = ".." }
arbitrary = { version = "1.3", features = ["derive"] }

[[bin]]
name = "query_parser"
path = "fuzz_targets/query_parser.rs"
test = false
doc = false

[[bin]]
name = "query_parser_structured"
path = "fuzz_targets/query_parser_structured.rs"
test = false
doc = false
```

Create `/Users/jordanwashburn/Workspace/orchard9/engram/engram-core/fuzz/fuzz_targets/query_parser.rs`:

```rust
#![no_main]
use libfuzzer_sys::fuzz_target;
use engram_core::query::parser::Parser;

fuzz_target!(|data: &[u8]| {
    if let Ok(s) = std::str::from_utf8(data) {
        // Parser should never panic on any input
        let _ = Parser::parse(s);
    }
});
```

Create `/Users/jordanwashburn/Workspace/orchard9/engram/engram-core/fuzz/fuzz_targets/query_parser_structured.rs`:

```rust
#![no_main]
use libfuzzer_sys::fuzz_target;
use arbitrary::Arbitrary;
use engram_core::query::parser::Parser;

#[derive(Arbitrary, Debug)]
struct FuzzQuery {
    operation: FuzzOperation,
    pattern: String,
    constraints: Vec<FuzzConstraint>,
}

#[derive(Arbitrary, Debug)]
enum FuzzOperation {
    Recall,
    Spread,
    Predict,
    Imagine,
    Consolidate,
}

#[derive(Arbitrary, Debug)]
struct FuzzConstraint {
    field: String,
    operator: FuzzOperator,
    value: FuzzValue,
}

#[derive(Arbitrary, Debug)]
enum FuzzOperator {
    GreaterThan,
    LessThan,
    Equal,
}

#[derive(Arbitrary, Debug)]
enum FuzzValue {
    Float(f32),
    String(String),
}

impl FuzzQuery {
    fn to_query_string(&self) -> String {
        let operation_str = match self.operation {
            FuzzOperation::Recall => "RECALL",
            FuzzOperation::Spread => "SPREAD FROM",
            FuzzOperation::Predict => "PREDICT",
            FuzzOperation::Imagine => "IMAGINE",
            FuzzOperation::Consolidate => "CONSOLIDATE",
        };

        let mut query = format!("{} {}", operation_str, self.pattern);

        if !self.constraints.is_empty() {
            query.push_str(" WHERE ");
            let constraint_strs: Vec<String> = self.constraints.iter().map(|c| {
                let op = match c.operator {
                    FuzzOperator::GreaterThan => ">",
                    FuzzOperator::LessThan => "<",
                    FuzzOperator::Equal => "=",
                };
                let value = match &c.value {
                    FuzzValue::Float(f) => f.to_string(),
                    FuzzValue::String(s) => format!("\"{}\"", s),
                };
                format!("{} {} {}", c.field, op, value)
            }).collect();
            query.push_str(&constraint_strs.join(" AND "));
        }

        query
    }
}

fuzz_target!(|fuzz_query: FuzzQuery| {
    let query_string = fuzz_query.to_query_string();
    let _ = Parser::parse(&query_string);
});
```

Add to `/Users/jordanwashburn/Workspace/orchard9/engram/engram-core/Cargo.toml`:

```toml
[dev-dependencies]
# ... existing dev-dependencies
arbitrary = { version = "1.3", features = ["derive"] }
```

### 8b. Performance Benchmarks

Create `/Users/jordanwashburn/Workspace/orchard9/engram/engram-core/benches/query_parser_performance.rs`:

```rust
use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use engram_core::query::parser::Parser;
use std::time::Duration;

fn benchmark_parse_times(c: &mut Criterion) {
    let mut group = c.benchmark_group("parser_performance");

    // Set strict performance targets
    group.significance_level(0.01)
        .sample_size(1000)
        .measurement_time(Duration::from_secs(10));

    let queries = vec![
        ("simple_recall", "RECALL episode"),
        ("recall_with_constraints", "RECALL episode WHERE confidence > 0.7"),
        ("complex_spread", "SPREAD FROM node MAX_HOPS 5 DECAY 0.15 THRESHOLD 0.1"),
        ("multiline", "RECALL episode\n  WHERE confidence > 0.7\n  AND created > \"2024-01-01\""),
    ];

    for (name, query) in queries {
        group.bench_with_input(BenchmarkId::new("parse", name), &query, |b, q| {
            b.iter(|| Parser::parse(black_box(q)))
        });
    }

    group.finish();
}

fn benchmark_parse_regression(c: &mut Criterion) {
    // Regression test: fail if parse time exceeds baseline by >10%
    let baseline = Duration::from_micros(100);
    let query = "RECALL episode WHERE confidence > 0.7";

    c.bench_function("parse_regression_guard", |b| {
        b.iter(|| {
            let start = std::time::Instant::now();
            let _ = Parser::parse(black_box(query));
            let elapsed = start.elapsed();

            // Note: In actual CI, this would fail the build
            if elapsed > baseline * 110 / 100 {
                eprintln!(
                    "WARNING: Parse time regression: {:?} exceeds baseline {:?} by >10%",
                    elapsed, baseline
                );
            }
        });
    });
}

criterion_group!(benches, benchmark_parse_times, benchmark_parse_regression);
criterion_main!(benches);
```

Add to `/Users/jordanwashburn/Workspace/orchard9/engram/engram-core/Cargo.toml`:

```toml
[[bench]]
name = "query_parser_performance"
harness = false

[dev-dependencies]
criterion = { version = "0.5", features = ["html_reports"] }
```

---

## Fix 9: Implement AST Display for Round-Trip Testing

Add to `/Users/jordanwashburn/Workspace/orchard9/engram/engram-core/src/query/parser/ast.rs`:

```rust
impl std::fmt::Display for Query {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            Query::Recall(r) => write!(f, "{}", r),
            Query::Spread(s) => write!(f, "{}", s),
            Query::Predict(p) => write!(f, "{}", p),
            Query::Imagine(i) => write!(f, "{}", i),
            Query::Consolidate(c) => write!(f, "{}", c),
        }
    }
}

impl std::fmt::Display for RecallQuery {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "RECALL {}", self.pattern)?;
        if !self.constraints.is_empty() {
            write!(f, " WHERE ")?;
            for (i, constraint) in self.constraints.iter().enumerate() {
                if i > 0 {
                    write!(f, " AND ")?;
                }
                write!(f, "{}", constraint)?;
            }
        }
        if let Some(limit) = self.limit {
            write!(f, " LIMIT {}", limit)?;
        }
        Ok(())
    }
}

impl std::fmt::Display for SpreadQuery {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "SPREAD FROM {}", self.source)?;
        if let Some(hops) = self.max_hops {
            write!(f, " MAX_HOPS {}", hops)?;
        }
        if let Some(decay) = self.decay_rate {
            write!(f, " DECAY {:.2}", decay)?;
        }
        if let Some(threshold) = self.activation_threshold {
            write!(f, " THRESHOLD {:.2}", threshold)?;
        }
        Ok(())
    }
}

// Similar Display implementations for PredictQuery, ImagineQuery, ConsolidateQuery, etc.
```

Then enable the round-trip property test in `/Users/jordanwashburn/Workspace/orchard9/engram/engram-core/tests/query_parser_property_tests.rs`:

```rust
#[test]
fn prop_parse_unparse_roundtrip() {
    proptest!(ProptestConfig::with_cases(1000), |(query in valid_query_generator())| {
        let ast1 = Parser::parse(&query);
        if ast1.is_err() {
            return Ok(());
        }

        let ast1 = ast1.unwrap();
        let unparsed = ast1.to_string();
        let ast2 = Parser::parse(&unparsed).expect("round-trip should parse");

        prop_assert_eq!(
            format!("{:?}", ast1),
            format!("{:?}", ast2),
            "Round-trip failed:\nOriginal: {}\nUnparsed: {}\n",
            query, unparsed
        );

        Ok(())
    });
}
```

---

## Execution Checklist

- [ ] 1. Create `validation.rs` with all validation functions
- [ ] 2. Update `parser.rs` to call validation functions
- [ ] 3. Add constraint validation and call it after parsing WHERE clause
- [ ] 4. Fix all `.unwrap()` and `.expect()` in `recall.rs`
- [ ] 5. Fix float comparisons in `spread.rs`
- [ ] 6. Remove unused imports in test files
- [ ] 7. Create fuzzing infrastructure (Cargo.toml + 2 fuzz targets)
- [ ] 8. Create performance benchmarks
- [ ] 9. Implement Display for AST types
- [ ] 10. Enable round-trip property test
- [ ] 11. Run `make quality` and fix any remaining issues
- [ ] 12. Run all tests and verify 100% pass
- [ ] 13. Run fuzzer for 1M iterations: `cargo fuzz run query_parser -- -runs=1000000`
- [ ] 14. Run benchmarks and establish baseline: `cargo bench --bench query_parser_performance`
- [ ] 15. Update task file status from `_complete` to `_in_progress`
- [ ] 16. After all fixes, update task file back to `_complete`

---

## Estimated Time

- Validation functions: 2 hours
- Parser updates: 2-3 hours
- Clippy fixes: 1-2 hours
- Fuzzing setup: 1 hour
- Benchmarks: 1 hour
- AST Display: 1-2 hours
- Testing and verification: 2-3 hours

**Total**: ~10-15 hours

---

## After Fixes Complete

Verify:
1. `cargo test --test query_language_corpus` - all tests pass
2. `cargo test --test error_message_validation` - all tests pass
3. `cargo test --test query_parser_property_tests` - all 7 properties pass
4. `make quality` - zero warnings
5. `cargo fuzz run query_parser -- -runs=1000000` - no crashes
6. `cargo bench --bench query_parser_performance` - <100μs P90

Then update task status to truly complete.
