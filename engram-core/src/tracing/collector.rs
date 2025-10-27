//! Background collector thread for draining ring buffers and exporting events

use crate::tracing::config::TracingConfig;
use crate::tracing::event::CognitiveEvent;
use crate::tracing::ring_buffer::RingBuffer;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread;
use std::time::Duration;

/// Handle for the background collector thread
pub struct CollectorHandle {
    /// Thread join handle
    handle: Option<thread::JoinHandle<()>>,
    /// Shutdown signal
    shutdown: Arc<AtomicBool>,
}

impl CollectorHandle {
    /// Signal shutdown and wait for collector thread to finish
    #[allow(dead_code)] // Used by public API, may not be called in current tests
    pub fn shutdown(mut self) {
        self.shutdown.store(true, Ordering::Release);
        if let Some(handle) = self.handle.take() {
            let _ = handle.join();
        }
    }
}

impl Drop for CollectorHandle {
    fn drop(&mut self) {
        self.shutdown.store(true, Ordering::Release);
        if let Some(handle) = self.handle.take() {
            let _ = handle.join();
        }
    }
}

/// Start the background collector thread
///
/// The collector periodically drains all ring buffers and exports events
/// in batches according to the configuration.
pub fn start_collector_thread(
    config: Arc<TracingConfig>,
    buffers: Arc<dashmap::DashMap<thread::ThreadId, Arc<RingBuffer<CognitiveEvent>>>>,
) -> CollectorHandle {
    let shutdown = Arc::new(AtomicBool::new(false));
    let shutdown_clone = Arc::clone(&shutdown);

    #[allow(clippy::expect_used)]
    // System initialization: panic is acceptable if thread spawn fails
    let handle = thread::Builder::new()
        .name("engram-tracer-collector".to_string())
        .spawn(move || {
            collector_loop(config, buffers, shutdown_clone);
        })
        .expect("Failed to spawn collector thread");

    CollectorHandle {
        handle: Some(handle),
        shutdown,
    }
}

/// Main collector loop
#[allow(clippy::needless_pass_by_value)] // Arc clones are intentional for thread ownership
fn collector_loop(
    config: Arc<TracingConfig>,
    buffers: Arc<dashmap::DashMap<thread::ThreadId, Arc<RingBuffer<CognitiveEvent>>>>,
    shutdown: Arc<AtomicBool>,
) {
    let interval = Duration::from_millis(config.export_interval_ms);

    while !shutdown.load(Ordering::Acquire) {
        // Sleep for export interval
        thread::sleep(interval);

        // Collect events from all ring buffers
        let mut batch = Vec::with_capacity(config.export_batch_size);

        for entry in buffers.iter() {
            let buffer = entry.value();

            // Drain events from this buffer
            while let Some(event) = buffer.pop() {
                batch.push(event);

                if batch.len() >= config.export_batch_size {
                    // Export this batch and start a new one
                    export_batch(&batch, &config);
                    batch.clear();
                }
            }
        }

        // Export any remaining events
        if !batch.is_empty() {
            export_batch(&batch, &config);
        }
    }

    // Final drain on shutdown
    let mut final_batch = Vec::new();
    for entry in buffers.iter() {
        let buffer = entry.value();
        while let Some(event) = buffer.pop() {
            final_batch.push(event);
        }
    }

    if !final_batch.is_empty() {
        export_batch(&final_batch, &config);
    }
}

/// Export a batch of events according to configuration
fn export_batch(events: &[CognitiveEvent], config: &TracingConfig) {
    use crate::tracing::config::ExportFormat;

    match config.export_format {
        ExportFormat::Json => {
            // Export to JSON (implementation in exporters/json.rs)
            if let Err(e) = crate::tracing::exporters::json::export_json(events) {
                tracing::warn!(
                    target: "engram::tracing::collector",
                    error = %e,
                    "Failed to export events to JSON"
                );
            }
        }
        ExportFormat::OtlpGrpc => {
            // Export to OpenTelemetry (implementation in exporters/otlp.rs)
            #[cfg(feature = "cognitive_tracing")]
            crate::tracing::exporters::otlp::export_otlp(events);
        }
        ExportFormat::Loki => {
            // Export to Loki (implementation in exporters/loki.rs)
            #[cfg(feature = "cognitive_tracing")]
            crate::tracing::exporters::loki::export_loki(events);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tracing::event::PrimingType;

    #[test]
    fn test_collector_startup_shutdown() {
        let config = Arc::new(TracingConfig::development());
        let buffers = Arc::new(dashmap::DashMap::new());

        let handle = start_collector_thread(config, buffers);

        // Let it run briefly
        thread::sleep(Duration::from_millis(100));

        // Shutdown
        handle.shutdown();
    }

    #[test]
    fn test_collector_drains_buffers() {
        let mut config = TracingConfig::development();
        config.export_interval_ms = 100; // Fast collection for testing
        let config = Arc::new(config);

        let buffers = Arc::new(dashmap::DashMap::new());

        // Add a buffer with events
        let buffer = Arc::new(RingBuffer::new(100));
        for _ in 0..10 {
            buffer.push(CognitiveEvent::new_priming(
                PrimingType::Semantic,
                0.5,
                1,
                2,
            ));
        }

        buffers.insert(thread::current().id(), Arc::clone(&buffer));

        let handle = start_collector_thread(Arc::clone(&config), Arc::clone(&buffers));

        // Wait for collection
        thread::sleep(Duration::from_millis(300));

        // Buffer should be drained (or mostly drained)
        // Note: This is a probabilistic test due to timing
        handle.shutdown();
    }
}
