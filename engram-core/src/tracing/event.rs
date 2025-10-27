//! Fixed-size cognitive event representation for zero-allocation tracing
//!
//! All events are 64 bytes (1 cache line) to avoid heap allocations and
//! maximize cache efficiency.

use std::time::Instant;

/// Fixed-size node identifier (no heap allocation)
pub type NodeId = u64;

/// Cognitive event with minimal allocation overhead
///
/// Total size: 64 bytes (fits in 1 cache line)
#[repr(C, align(64))]
#[derive(Debug, Clone, Copy)]
pub struct CognitiveEvent {
    /// Timestamp using monotonic clock (16 bytes)
    pub timestamp: Instant,

    /// Event type discriminant (1 byte)
    pub event_type: EventType,

    /// Padding for alignment (7 bytes)
    _padding: [u8; 7],

    /// Event-specific data (40 bytes)
    pub data: EventData,
}

impl CognitiveEvent {
    /// Create a new priming event
    #[inline]
    #[must_use]
    pub fn new_priming(
        priming_type: PrimingType,
        strength: f32,
        source_node: NodeId,
        target_node: NodeId,
    ) -> Self {
        Self {
            timestamp: Instant::now(),
            event_type: EventType::Priming,
            _padding: [0; 7],
            data: EventData {
                priming: PrimingData {
                    priming_type,
                    strength,
                    source_node,
                    target_node,
                    _padding: [0; 19],
                },
            },
        }
    }

    /// Create a new interference event
    #[inline]
    #[must_use]
    pub fn new_interference(
        interference_type: InterferenceType,
        magnitude: f32,
        target_episode_id: u64,
        competing_episode_count: u32,
    ) -> Self {
        Self {
            timestamp: Instant::now(),
            event_type: EventType::Interference,
            _padding: [0; 7],
            data: EventData {
                interference: InterferenceData {
                    interference_type,
                    magnitude,
                    target_episode_id,
                    competing_episode_count,
                    _padding: [0; 19],
                },
            },
        }
    }

    /// Create a new reconsolidation event
    #[inline]
    #[must_use]
    pub fn new_reconsolidation(
        episode_id: u64,
        window_position: f32,
        plasticity_factor: f32,
        modification_count: u32,
    ) -> Self {
        Self {
            timestamp: Instant::now(),
            event_type: EventType::Reconsolidation,
            _padding: [0; 7],
            data: EventData {
                reconsolidation: ReconsolidationData {
                    episode_id,
                    window_position,
                    plasticity_factor,
                    modification_count,
                    _padding: [0; 20],
                },
            },
        }
    }

    /// Create a new false memory event
    #[inline]
    #[must_use]
    pub fn new_false_memory(
        critical_lure_hash: u64,
        source_list_size: u32,
        reconstruction_confidence: f32,
    ) -> Self {
        Self {
            timestamp: Instant::now(),
            event_type: EventType::FalseMemory,
            _padding: [0; 7],
            data: EventData {
                false_memory: FalseMemoryData {
                    critical_lure_hash,
                    source_list_size,
                    reconstruction_confidence,
                    _padding: [0; 24],
                },
            },
        }
    }
}

/// Event type discriminant
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum EventType {
    /// Priming event (semantic, associative, or repetition)
    Priming = 0,
    /// Interference event (proactive, retroactive, or fan effect)
    Interference = 1,
    /// Reconsolidation event (memory modification during retrieval)
    Reconsolidation = 2,
    /// False memory event (DRM paradigm critical lure)
    FalseMemory = 3,
}

/// Union of event-specific data to maintain fixed size
///
/// All variants must fit in 40 bytes
#[repr(C)]
#[derive(Clone, Copy)]
pub union EventData {
    pub priming: PrimingData,
    pub interference: InterferenceData,
    pub reconsolidation: ReconsolidationData,
    pub false_memory: FalseMemoryData,
}

impl std::fmt::Debug for EventData {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("EventData").finish_non_exhaustive()
    }
}

/// Priming event data (40 bytes)
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PrimingData {
    pub priming_type: PrimingType, // 1 byte
    pub strength: f32,             // 4 bytes
    pub source_node: NodeId,       // 8 bytes
    pub target_node: NodeId,       // 8 bytes
    _padding: [u8; 19],            // 19 bytes padding = 40 total
}

/// Interference event data (40 bytes)
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct InterferenceData {
    pub interference_type: InterferenceType, // 1 byte
    pub magnitude: f32,                      // 4 bytes
    pub target_episode_id: u64,              // 8 bytes
    pub competing_episode_count: u32,        // 4 bytes
    _padding: [u8; 19],                      // 19 bytes padding = 36 -> round to 40
}

/// Reconsolidation event data (40 bytes)
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct ReconsolidationData {
    pub episode_id: u64,         // 8 bytes
    pub window_position: f32,    // 4 bytes
    pub plasticity_factor: f32,  // 4 bytes
    pub modification_count: u32, // 4 bytes
    _padding: [u8; 20],          // 20 bytes padding = 40 total
}

/// False memory event data (40 bytes)
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct FalseMemoryData {
    pub critical_lure_hash: u64,        // 8 bytes
    pub source_list_size: u32,          // 4 bytes
    pub reconstruction_confidence: f32, // 4 bytes
    _padding: [u8; 24],                 // 24 bytes padding = 40 total
}

/// Type of priming effect
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PrimingType {
    /// Semantic priming (meaning-based)
    Semantic = 0,
    /// Associative priming (relationship-based)
    Associative = 1,
    /// Repetition priming (exposure-based)
    Repetition = 2,
}

/// Type of interference effect
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InterferenceType {
    /// Proactive interference (old memories interfere with new)
    Proactive = 0,
    /// Retroactive interference (new memories interfere with old)
    Retroactive = 1,
    /// Fan effect (multiple associations reduce retrieval strength)
    Fan = 2,
}

#[cfg(test)]
#[allow(unsafe_code)] // Tests need to access union fields
mod tests {
    use super::*;

    #[test]
    fn test_event_size() {
        // Verify that CognitiveEvent is exactly 64 bytes (1 cache line)
        assert_eq!(
            std::mem::size_of::<CognitiveEvent>(),
            64,
            "CognitiveEvent must be exactly 64 bytes"
        );
    }

    #[test]
    fn test_event_alignment() {
        // Verify cache-line alignment
        assert_eq!(
            std::mem::align_of::<CognitiveEvent>(),
            64,
            "CognitiveEvent must be 64-byte aligned"
        );
    }

    #[test]
    #[allow(clippy::float_cmp)] // Exact comparison intended for test validation
    fn test_priming_event_creation() {
        let event = CognitiveEvent::new_priming(PrimingType::Semantic, 0.75, 1234, 5678);

        assert_eq!(event.event_type, EventType::Priming);

        unsafe {
            assert_eq!(event.data.priming.priming_type, PrimingType::Semantic);
            assert_eq!(event.data.priming.strength, 0.75);
            assert_eq!(event.data.priming.source_node, 1234);
            assert_eq!(event.data.priming.target_node, 5678);
        }
    }

    #[test]
    #[allow(clippy::float_cmp)] // Exact comparison intended for test validation
    fn test_interference_event_creation() {
        let event = CognitiveEvent::new_interference(InterferenceType::Retroactive, 0.5, 9999, 3);

        assert_eq!(event.event_type, EventType::Interference);

        unsafe {
            assert_eq!(
                event.data.interference.interference_type,
                InterferenceType::Retroactive
            );
            assert_eq!(event.data.interference.magnitude, 0.5);
            assert_eq!(event.data.interference.target_episode_id, 9999);
            assert_eq!(event.data.interference.competing_episode_count, 3);
        }
    }

    #[test]
    #[allow(clippy::float_cmp)] // Exact comparison intended for test validation
    fn test_reconsolidation_event_creation() {
        let event = CognitiveEvent::new_reconsolidation(12345, 0.3, 0.8, 5);

        assert_eq!(event.event_type, EventType::Reconsolidation);

        unsafe {
            assert_eq!(event.data.reconsolidation.episode_id, 12345);
            assert_eq!(event.data.reconsolidation.window_position, 0.3);
            assert_eq!(event.data.reconsolidation.plasticity_factor, 0.8);
            assert_eq!(event.data.reconsolidation.modification_count, 5);
        }
    }

    #[test]
    #[allow(clippy::float_cmp)] // Exact comparison intended for test validation
    fn test_false_memory_event_creation() {
        let event = CognitiveEvent::new_false_memory(0xDEAD_BEEF, 10, 0.95);

        assert_eq!(event.event_type, EventType::FalseMemory);

        unsafe {
            assert_eq!(event.data.false_memory.critical_lure_hash, 0xDEAD_BEEF);
            assert_eq!(event.data.false_memory.source_list_size, 10);
            assert_eq!(event.data.false_memory.reconstruction_confidence, 0.95);
        }
    }
}
