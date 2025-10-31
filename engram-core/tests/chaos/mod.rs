//! Chaos testing framework for streaming memory operations.
//!
//! This module provides tools for chaos engineering validation of the streaming
//! memory system. It includes fault injectors, validators, and comprehensive
//! chaos tests that verify correctness under adverse conditions.
//!
//! ## Testing Philosophy
//!
//! Chaos engineering is about discovering system weaknesses before they cause
//! problems in production. We inject controlled failures to validate that:
//!
//! 1. **Eventual consistency is maintained** - All acknowledged observations
//!    eventually become visible in recalls
//! 2. **No data loss occurs** - Even under packet loss, delays, and crashes
//! 3. **Sequence ordering is preserved** - Monotonic sequence numbers maintained
//! 4. **Graph integrity is maintained** - HNSW structure remains valid
//! 5. **System recovers gracefully** - Returns to normal after chaos stops
//!
//! ## Fault Types
//!
//! - **Network delays**: Variable latency (0-100ms) to test temporal ordering
//! - **Packet loss**: Random drops (1%) to test retry logic
//! - **Queue overflow**: Burst loads to test admission control
//! - **Clock skew**: Time drift (±5s) to test timestamp handling
//!
//! ## Validators
//!
//! - `EventualConsistencyValidator`: Tracks acked observations and validates recall
//! - `SequenceValidator`: Ensures monotonic sequence numbers
//! - `GraphIntegrityValidator`: Checks HNSW structural invariants
//!
//! ## Running Tests
//!
//! ```bash
//! # Run all chaos tests
//! cargo test --test '*' chaos
//!
//! # Run 10-minute sustained chaos test
//! cargo test chaos_10min_sustained --release -- --ignored --nocapture
//!
//! # Run specific chaos scenario
//! cargo test chaos_network_delays --nocapture
//! ```
//!
//! ## Research Foundation
//!
//! Based on Netflix's chaos engineering principles:
//! - Principles of Chaos Engineering (2011)
//! - Chaos Monkey and the Simian Army
//! - Testing in Production: The Netflix Way
//!
//! And distributed systems consistency research:
//! - Bailis, P. et al. (2013). "Quantifying eventual consistency with PBS"
//! - Jepsen test suite methodology
//! - Kyle Kingsbury's distributed systems testing work

pub mod fault_injector;
pub mod streaming_chaos;
pub mod validators;

pub use fault_injector::{
    BurstLoadGenerator, ChaosScenario, ChaosScenarioBuilder, ClockSkewSimulator, DelayInjector,
    PacketLossSimulator, PacketLossStats,
};
pub use validators::{
    ChaosTestStats, EventualConsistencyValidator, GraphIntegrityValidator, SequenceValidator,
    ValidationError,
};
