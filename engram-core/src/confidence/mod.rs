//! Dual-memory confidence modeling utilities.
//!
//! This module captures the cognitive rules that govern how uncertainty travels
//! between episodic and semantic representations.  We expose both the runtime
//! propagation helpers and the accompanying SMT-backed verification entry
//! points (enabled with the `smt_verification` feature).

pub mod dual_memory;

#[cfg(feature = "smt_verification")]
pub mod verification;

pub use dual_memory::DualMemoryConfidence;
