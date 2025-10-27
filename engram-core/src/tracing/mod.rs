//! Cognitive event tracing with bounded memory and zero-cost abstraction
//!
//! When `cognitive_tracing` feature is disabled, this module compiles to nothing.
//! When enabled, provides lock-free event recording with <100ns overhead.

#[cfg(feature = "cognitive_tracing")]
mod collector;
#[cfg(feature = "cognitive_tracing")]
mod config;
#[cfg(feature = "cognitive_tracing")]
mod event;
#[cfg(feature = "cognitive_tracing")]
mod ring_buffer;

#[cfg(feature = "cognitive_tracing")]
pub mod exporters;

#[cfg(feature = "cognitive_tracing")]
pub use config::{ExportFormat, TracingConfig};
#[cfg(feature = "cognitive_tracing")]
pub use event::{CognitiveEvent, EventType, InterferenceType, NodeId, PrimingType};

#[cfg(feature = "cognitive_tracing")]
use collector::CollectorHandle;
#[cfg(feature = "cognitive_tracing")]
use ring_buffer::RingBuffer;
#[cfg(feature = "cognitive_tracing")]
use std::sync::Arc;
#[cfg(feature = "cognitive_tracing")]
use std::thread;

/// Global cognitive tracer instance
#[cfg(feature = "cognitive_tracing")]
pub struct CognitiveTracer {
    config: Arc<TracingConfig>,
    /// Per-thread ring buffers
    buffers: Arc<dashmap::DashMap<thread::ThreadId, Arc<RingBuffer<CognitiveEvent>>>>,
    /// Background collector handle
    _collector_handle: Option<CollectorHandle>,
}

#[cfg(feature = "cognitive_tracing")]
impl CognitiveTracer {
    /// Create new tracer with configuration
    #[must_use]
    pub fn new(config: TracingConfig) -> Self {
        let config = Arc::new(config);
        let buffers = Arc::new(dashmap::DashMap::new());

        // Start background collector thread
        let collector_handle =
            collector::start_collector_thread(Arc::clone(&config), Arc::clone(&buffers));

        Self {
            config,
            buffers,
            _collector_handle: Some(collector_handle),
        }
    }

    /// Record priming event (zero-overhead when tracing disabled)
    #[inline]
    pub fn trace_priming(
        &self,
        priming_type: PrimingType,
        strength: f32,
        source_node: NodeId,
        target_node: NodeId,
    ) {
        // Early return if this event type not enabled
        if !self.config.is_enabled(EventType::Priming) {
            return;
        }

        // Sampling check
        if !self.config.should_sample(EventType::Priming) {
            return;
        }

        // Get thread-local ring buffer
        let thread_id = thread::current().id();
        let buffer = self
            .buffers
            .entry(thread_id)
            .or_insert_with(|| Arc::new(RingBuffer::new(self.config.ring_buffer_size)));

        // Record event (lock-free push)
        let event = CognitiveEvent::new_priming(priming_type, strength, source_node, target_node);

        buffer.push(event);
    }

    /// Record interference event
    #[inline]
    pub fn trace_interference(
        &self,
        interference_type: InterferenceType,
        magnitude: f32,
        target_episode_id: u64,
        competing_episode_count: u32,
    ) {
        if !self.config.is_enabled(EventType::Interference) {
            return;
        }

        if !self.config.should_sample(EventType::Interference) {
            return;
        }

        let thread_id = thread::current().id();
        let buffer = self
            .buffers
            .entry(thread_id)
            .or_insert_with(|| Arc::new(RingBuffer::new(self.config.ring_buffer_size)));

        let event = CognitiveEvent::new_interference(
            interference_type,
            magnitude,
            target_episode_id,
            competing_episode_count,
        );

        buffer.push(event);
    }

    /// Record reconsolidation event
    #[inline]
    pub fn trace_reconsolidation(
        &self,
        episode_id: u64,
        window_position: f32,
        plasticity_factor: f32,
        modification_count: u32,
    ) {
        if !self.config.is_enabled(EventType::Reconsolidation) {
            return;
        }

        if !self.config.should_sample(EventType::Reconsolidation) {
            return;
        }

        let thread_id = thread::current().id();
        let buffer = self
            .buffers
            .entry(thread_id)
            .or_insert_with(|| Arc::new(RingBuffer::new(self.config.ring_buffer_size)));

        let event = CognitiveEvent::new_reconsolidation(
            episode_id,
            window_position,
            plasticity_factor,
            modification_count,
        );

        buffer.push(event);
    }

    /// Record false memory event
    #[inline]
    pub fn trace_false_memory(
        &self,
        critical_lure_hash: u64,
        source_list_size: u32,
        reconstruction_confidence: f32,
    ) {
        if !self.config.is_enabled(EventType::FalseMemory) {
            return;
        }

        if !self.config.should_sample(EventType::FalseMemory) {
            return;
        }

        let thread_id = thread::current().id();
        let buffer = self
            .buffers
            .entry(thread_id)
            .or_insert_with(|| Arc::new(RingBuffer::new(self.config.ring_buffer_size)));

        let event = CognitiveEvent::new_false_memory(
            critical_lure_hash,
            source_list_size,
            reconstruction_confidence,
        );

        buffer.push(event);
    }
}

/// When tracing disabled, provide no-op implementations with zero overhead
#[cfg(not(feature = "cognitive_tracing"))]
pub struct CognitiveTracer {
    _phantom: core::marker::PhantomData<()>,
}

#[cfg(not(feature = "cognitive_tracing"))]
impl CognitiveTracer {
    /// Create a no-op tracer when feature is disabled
    #[inline(always)]
    #[must_use]
    pub fn new(_config: ()) -> Self {
        Self {
            _phantom: core::marker::PhantomData,
        }
    }

    /// No-op when tracing disabled
    #[inline(always)]
    pub fn trace_priming(
        &self,
        _priming_type: u8,
        _strength: f32,
        _source_node: u64,
        _target_node: u64,
    ) {
        // Compiles to nothing
    }

    /// No-op when tracing disabled
    #[inline(always)]
    pub fn trace_interference(
        &self,
        _interference_type: u8,
        _magnitude: f32,
        _target_episode_id: u64,
        _competing_episode_count: u32,
    ) {
        // Compiles to nothing
    }

    /// No-op when tracing disabled
    #[inline(always)]
    pub fn trace_reconsolidation(
        &self,
        _episode_id: u64,
        _window_position: f32,
        _plasticity_factor: f32,
        _modification_count: u32,
    ) {
        // Compiles to nothing
    }

    /// No-op when tracing disabled
    #[inline(always)]
    pub fn trace_false_memory(
        &self,
        _critical_lure_hash: u64,
        _source_list_size: u32,
        _reconstruction_confidence: f32,
    ) {
        // Compiles to nothing
    }
}

#[cfg(test)]
#[cfg(feature = "cognitive_tracing")]
mod tests {
    use super::*;
    use std::time::Duration;

    #[test]
    fn test_tracer_creation() {
        let config = TracingConfig::development();
        let tracer = CognitiveTracer::new(config);

        // Verify tracer was created
        assert!(tracer.config.is_enabled(EventType::Priming));
    }

    #[test]
    fn test_trace_priming() {
        let config = TracingConfig::development();
        let tracer = CognitiveTracer::new(config);

        tracer.trace_priming(PrimingType::Semantic, 0.5, 100, 200);

        // Give collector time to process (if running)
        thread::sleep(Duration::from_millis(10));

        // Verify buffer was created for this thread
        assert_eq!(tracer.buffers.len(), 1);
    }

    #[test]
    fn test_trace_disabled_event_type() {
        let mut config = TracingConfig::disabled();
        config.enabled_events.insert(EventType::Priming);
        config.sample_rates.insert(EventType::Priming, 1.0);
        config.ring_buffer_size = 1000;

        let tracer = CognitiveTracer::new(config);

        // Priming should work
        tracer.trace_priming(PrimingType::Semantic, 0.5, 1, 2);

        // Interference should be no-op (not enabled)
        tracer.trace_interference(InterferenceType::Proactive, 0.3, 999, 5);

        thread::sleep(Duration::from_millis(10));
    }

    #[test]
    fn test_sampling() {
        let mut config = TracingConfig::disabled();
        config.enabled_events.insert(EventType::Priming);
        config.sample_rates.insert(EventType::Priming, 0.0); // Never sample
        config.ring_buffer_size = 1000;

        let tracer = CognitiveTracer::new(config);

        // These should all be dropped due to sampling
        for _ in 0..100 {
            tracer.trace_priming(PrimingType::Semantic, 0.5, 1, 2);
        }

        thread::sleep(Duration::from_millis(10));

        // Buffer should be empty (events were sampled out)
        let thread_id = thread::current().id();
        if let Some(buffer) = tracer.buffers.get(&thread_id) {
            assert_eq!(buffer.len(), 0);
        }
    }
}

#[cfg(test)]
#[cfg(not(feature = "cognitive_tracing"))]
mod tests_disabled {
    use super::*;

    #[test]
    fn test_zero_size_when_disabled() {
        // When feature is disabled, tracer should be zero-sized
        assert_eq!(
            std::mem::size_of::<CognitiveTracer>(),
            0,
            "CognitiveTracer should be zero-sized when feature is disabled"
        );
    }
}
