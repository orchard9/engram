//! Basic fuzzer for query parser - tests parser robustness with arbitrary input.
//!
//! This fuzzer generates completely random byte sequences and ensures the
//! parser never panics, crashes, or exhibits undefined behavior.
//!
//! ## Usage
//!
//! ```bash
//! # Install cargo-fuzz if not already installed
//! cargo install cargo-fuzz
//!
//! # Run fuzzer for 1 million iterations
//! cargo fuzz run query_parser -- -runs=1000000
//!
//! # Run fuzzer for 8 hours (overnight)
//! cargo fuzz run query_parser -- -max_total_time=28800
//!
//! # Generate coverage report
//! cargo fuzz coverage query_parser
//! ```

#![no_main]

use libfuzzer_sys::fuzz_target;
use engram_core::query::parser::Parser;

fuzz_target!(|data: &[u8]| {
    // Convert bytes to UTF-8 string (ignore invalid UTF-8)
    if let Ok(s) = std::str::from_utf8(data) {
        // Parser should never panic on any input
        // It must always return Ok or Err gracefully
        let _ = Parser::parse(s);
    }
});
