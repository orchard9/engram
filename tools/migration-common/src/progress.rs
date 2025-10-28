//! Progress tracking and reporting for migrations

use parking_lot::Mutex;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

/// Tracks migration progress with ETA calculation
pub struct ProgressTracker {
    total_records: Option<u64>,
    processed_records: Arc<AtomicU64>,
    start_time: Instant,
    last_report_time: Arc<Mutex<Instant>>,
    report_interval: Duration,
}

impl ProgressTracker {
    /// Create a new progress tracker
    #[must_use]
    pub fn new(total_records: Option<u64>, report_interval: Duration) -> Self {
        Self {
            total_records,
            processed_records: Arc::new(AtomicU64::new(0)),
            start_time: Instant::now(),
            last_report_time: Arc::new(Mutex::new(Instant::now())),
            report_interval,
        }
    }

    /// Increment processed count
    pub fn increment(&self, count: u64) {
        self.processed_records.fetch_add(count, Ordering::Relaxed);
    }

    /// Get current processed count
    #[must_use]
    pub fn processed(&self) -> u64 {
        self.processed_records.load(Ordering::Relaxed)
    }

    /// Check if it's time to report progress
    #[must_use]
    pub fn should_report(&self) -> bool {
        let last_report = *self.last_report_time.lock();
        last_report.elapsed() >= self.report_interval
    }

    /// Report current progress
    pub fn report(&self) {
        let processed = self.processed_records.load(Ordering::Relaxed);
        let elapsed = self.start_time.elapsed();
        let rate = if elapsed.as_secs() > 0 {
            #[allow(clippy::cast_precision_loss)]
            let rate_f64 = processed as f64 / elapsed.as_secs_f64();
            rate_f64
        } else {
            0.0
        };

        if let Some(total) = self.total_records {
            let remaining = total.saturating_sub(processed);
            #[allow(clippy::cast_possible_truncation)]
            let eta = if rate > 0.0 {
                #[allow(clippy::cast_precision_loss, clippy::cast_sign_loss)]
                Duration::from_secs((remaining as f64 / rate) as u64)
            } else {
                Duration::from_secs(0)
            };

            #[allow(clippy::cast_precision_loss)]
            let percentage = (processed as f64 / total as f64) * 100.0;

            tracing::info!(
                processed = processed,
                total = total,
                percentage = format!("{:.1}%", percentage),
                rate = format!("{:.0} records/sec", rate),
                eta = format!("{:?}", eta),
                "Migration progress"
            );
        } else {
            tracing::info!(
                processed = processed,
                rate = format!("{:.0} records/sec", rate),
                elapsed = format!("{:?}", elapsed),
                "Migration progress"
            );
        }

        *self.last_report_time.lock() = Instant::now();
    }

    /// Force a progress report regardless of interval
    pub fn force_report(&self) {
        self.report();
    }

    /// Get final statistics
    #[must_use]
    pub fn final_stats(&self) -> ProgressStats {
        let processed = self.processed_records.load(Ordering::Relaxed);
        let elapsed = self.start_time.elapsed();
        #[allow(clippy::cast_precision_loss)]
        let avg_rate = if elapsed.as_secs() > 0 {
            processed as f64 / elapsed.as_secs_f64()
        } else {
            0.0
        };

        ProgressStats {
            total_processed: processed,
            total_time: elapsed,
            avg_rate,
        }
    }
}

/// Final progress statistics
#[derive(Debug, Clone)]
pub struct ProgressStats {
    /// Total records processed
    pub total_processed: u64,
    /// Total time elapsed
    pub total_time: Duration,
    /// Average processing rate (records/sec)
    pub avg_rate: f64,
}
