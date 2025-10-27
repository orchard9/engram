//! Memory interference patterns
//!
//! Implements interference effects from cognitive psychology including
//! proactive, retroactive, and fan effects based on empirical research.

pub mod fan_effect;
pub mod proactive;
pub mod retroactive;

pub use fan_effect::{FanEffectDetector, FanEffectResult, FanEffectStatistics};
pub use proactive::{ProactiveInterferenceDetector, ProactiveInterferenceResult};
pub use retroactive::{RetroactiveInterferenceDetector, RetroactiveInterferenceResult};

/// Type of interference effect
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InterferenceType {
    /// Proactive interference (old memories interfere with new learning)
    Proactive,
    /// Retroactive interference (new learning interferes with old memories)
    Retroactive,
    /// Fan effect (multiple associations slow retrieval)
    Fan,
}
