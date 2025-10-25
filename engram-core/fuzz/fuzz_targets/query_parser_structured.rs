//! Structured fuzzer for query parser - generates grammar-aware queries.
//!
//! This fuzzer uses the `arbitrary` crate to generate structured queries
//! that are more likely to exercise deep parser logic and find subtle bugs.
//!
//! ## Advantages over Basic Fuzzing
//!
//! - Generates syntactically plausible queries
//! - Exercises more parser code paths
//! - Finds semantic validation bugs
//! - Better code coverage per iteration
//!
//! ## Usage
//!
//! ```bash
//! # Run structured fuzzer
//! cargo fuzz run query_parser_structured -- -runs=1000000
//!
//! # With dictionary for better mutations
//! cargo fuzz run query_parser_structured -- -dict=fuzz/query.dict -runs=1000000
//! ```

#![no_main]

use libfuzzer_sys::fuzz_target;
use arbitrary::Arbitrary;
use engram_core::query::parser::Parser;

/// Fuzzing wrapper for query operations
#[derive(Arbitrary, Debug, Clone)]
enum FuzzOperation {
    Recall,
    Spread,
    Predict,
    Imagine,
    Consolidate,
}

/// Fuzzing wrapper for query constraints
#[derive(Arbitrary, Debug, Clone)]
struct FuzzConstraint {
    field: String,
    operator: FuzzOperator,
    value: FuzzValue,
}

/// Fuzzing wrapper for comparison operators
#[derive(Arbitrary, Debug, Clone)]
enum FuzzOperator {
    GreaterThan,
    LessThan,
    Equal,
}

impl FuzzOperator {
    fn to_str(&self) -> &'static str {
        match self {
            Self::GreaterThan => ">",
            Self::LessThan => "<",
            Self::Equal => "=",
        }
    }
}

/// Fuzzing wrapper for constraint values
#[derive(Arbitrary, Debug, Clone)]
enum FuzzValue {
    Float(f32),
    String(String),
    Integer(u32),
}

impl FuzzValue {
    fn to_query_string(&self) -> String {
        match self {
            Self::Float(f) => format!("{}", f),
            Self::String(s) => format!("\"{}\"", s.replace('"', "\\\"")),
            Self::Integer(i) => format!("{}", i),
        }
    }
}

/// Structured fuzz query
#[derive(Arbitrary, Debug)]
struct FuzzQuery {
    operation: FuzzOperation,
    pattern: String,
    constraints: Vec<FuzzConstraint>,
    max_hops: Option<u16>,
    decay: Option<f32>,
    threshold: Option<f32>,
    novelty: Option<f32>,
    limit: Option<u32>,
}

impl FuzzQuery {
    /// Convert fuzz query to actual query string
    fn to_query_string(&self) -> String {
        let mut query = String::new();

        // Add operation keyword
        match self.operation {
            FuzzOperation::Recall => {
                query.push_str("RECALL ");
                query.push_str(&self.pattern);
            }
            FuzzOperation::Spread => {
                query.push_str("SPREAD FROM ");
                query.push_str(&self.pattern);

                if let Some(hops) = self.max_hops {
                    query.push_str(&format!(" MAX_HOPS {}", hops));
                }
                if let Some(decay) = self.decay {
                    query.push_str(&format!(" DECAY {}", decay));
                }
                if let Some(threshold) = self.threshold {
                    query.push_str(&format!(" THRESHOLD {}", threshold));
                }
            }
            FuzzOperation::Predict => {
                query.push_str("PREDICT ");
                query.push_str(&self.pattern);
                query.push_str(" GIVEN context");
            }
            FuzzOperation::Imagine => {
                query.push_str("IMAGINE ");
                query.push_str(&self.pattern);
                query.push_str(" BASED ON seed");

                if let Some(novelty) = self.novelty {
                    query.push_str(&format!(" NOVELTY {}", novelty));
                }
            }
            FuzzOperation::Consolidate => {
                query.push_str("CONSOLIDATE ");
                query.push_str(&self.pattern);
                query.push_str(" INTO target");
            }
        }

        // Add constraints
        if !self.constraints.is_empty() {
            query.push_str(" WHERE ");
            for (i, constraint) in self.constraints.iter().enumerate() {
                if i > 0 {
                    query.push_str(" AND ");
                }
                query.push_str(&format!(
                    "{} {} {}",
                    constraint.field,
                    constraint.operator.to_str(),
                    constraint.value.to_query_string()
                ));
            }
        }

        // Add limit for RECALL
        if matches!(self.operation, FuzzOperation::Recall) {
            if let Some(limit) = self.limit {
                query.push_str(&format!(" LIMIT {}", limit));
            }
        }

        query
    }
}

fuzz_target!(|fuzz_query: FuzzQuery| {
    // Convert fuzz query to string
    let query_string = fuzz_query.to_query_string();

    // Parser should never panic, even on malformed structured queries
    let _ = Parser::parse(&query_string);
});
