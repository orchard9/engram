//! Memory types and conversions for dual memory architecture

// Core memory types module
pub mod types;

// Re-export everything from types for backwards compatibility
pub use types::*;

// Export dual memory types when feature is enabled
#[cfg(feature = "dual_memory_types")]
pub mod binding_ops;
#[cfg(feature = "dual_memory_types")]
pub mod bindings;
#[cfg(feature = "dual_memory_types")]
pub mod conversions;
#[cfg(feature = "dual_memory_types")]
pub mod dual_types;

#[cfg(feature = "dual_memory_types")]
pub use bindings::{BindingRef, ConceptBinding};
#[cfg(feature = "dual_memory_types")]
pub use dual_types::{DualMemoryNode, EpisodeId, MemoryNodeType};
