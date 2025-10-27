//! Integration test suite runner for cognitive patterns
//!
//! This file serves as the entry point for all cognitive pattern integration tests.

// Shared test utilities
mod helpers;

mod integration {
    pub mod cognitive_patterns_integration;
    pub mod concurrent_cognitive_operations;
    pub mod cross_phenomenon_interactions;
    pub mod soak_test_memory_leaks;
}
