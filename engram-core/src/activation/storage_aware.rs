use crate::activation::ActivationRecord;
use atomic_float::AtomicF32;
use std::convert::TryFrom;
use std::fmt;
use std::sync::atomic::{AtomicU8, AtomicU16, AtomicU64, Ordering};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

/// Storage tier classification for activation spreading
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum StorageTier {
    /// In-memory hot tier with minimal latency.
    Hot = 0,
    /// SSD-backed warm tier balancing capacity and latency.
    Warm = 1,
    /// Columnar cold tier optimized for capacity over latency.
    Cold = 2,
}

impl StorageTier {
    /// Activation threshold that must be exceeded for the tier.
    #[must_use]
    pub const fn activation_threshold(self) -> f32 {
        match self {
            Self::Hot => 0.01,
            Self::Warm => 0.05,
            Self::Cold => 0.1,
        }
    }

    /// Latency budget enforced for the tier.
    #[must_use]
    pub const fn latency_budget(self) -> Duration {
        match self {
            Self::Hot => Duration::from_micros(100),
            Self::Warm => Duration::from_millis(1),
            Self::Cold => Duration::from_millis(10),
        }
    }

    /// Confidence adjustment factor per tier.
    #[must_use]
    pub const fn confidence_factor(self) -> f32 {
        match self {
            Self::Hot => 1.0,
            Self::Warm => 0.95,
            Self::Cold => 0.9,
        }
    }

    /// Derive a tier heuristic based on activation depth.
    #[must_use]
    pub const fn from_depth(depth: u16) -> Self {
        if depth == 0 {
            Self::Hot
        } else if depth <= 2 {
            Self::Warm
        } else {
            Self::Cold
        }
    }
}

/// Atomic bitflag container for activation metadata.
#[derive(Debug, Default)]
pub struct ActivationFlags {
    bits: AtomicU8,
}

impl ActivationFlags {
    /// Flag indicating migration is in progress.
    pub const MIGRATION_PENDING: u8 = 0b0000_0001;
    /// Flag indicating the activation exceeded latency budget.
    pub const LATENCY_OVER_BUDGET: u8 = 0b0000_0010;
    /// Flag indicating the activation attained high confidence.
    pub const HIGH_CONFIDENCE: u8 = 0b0000_0100;
    /// Flag indicating activation was throttled due to tier limits.
    pub const THROTTLED: u8 = 0b0000_1000;

    /// Create a new empty flag container.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            bits: AtomicU8::new(0),
        }
    }

    /// Insert the provided flag bit.
    pub fn insert(&self, flag: u8) {
        let mut current = self.bits.load(Ordering::Relaxed);
        loop {
            let updated = current | flag;
            match self.bits.compare_exchange_weak(
                current,
                updated,
                Ordering::Relaxed,
                Ordering::Relaxed,
            ) {
                Ok(_) => return,
                Err(actual) => current = actual,
            }
        }
    }

    /// Remove the provided flag bit.
    pub fn remove(&self, flag: u8) {
        let mut current = self.bits.load(Ordering::Relaxed);
        let mask = !flag;
        loop {
            let updated = current & mask;
            match self.bits.compare_exchange_weak(
                current,
                updated,
                Ordering::Relaxed,
                Ordering::Relaxed,
            ) {
                Ok(_) => return,
                Err(actual) => current = actual,
            }
        }
    }

    /// Check if the container currently has the flag set.
    #[must_use]
    pub fn contains(&self, flag: u8) -> bool {
        self.bits.load(Ordering::Relaxed) & flag == flag
    }

    /// Clear all stored flags.
    pub fn clear(&self) {
        self.bits.store(0, Ordering::Relaxed);
    }

    /// Return the underlying bit representation.
    #[must_use]
    pub fn bits(&self) -> u8 {
        self.bits.load(Ordering::Relaxed)
    }
}

impl Clone for ActivationFlags {
    fn clone(&self) -> Self {
        Self {
            bits: AtomicU8::new(self.bits.load(Ordering::Relaxed)),
        }
    }
}

impl fmt::Display for ActivationFlags {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:#010b}", self.bits())
    }
}

/// Storage-aware activation record aligned to a cache line.
#[repr(C, align(64))]
#[derive(Debug)]
pub struct StorageAwareActivation {
    /// Identifier for the memory node.
    pub memory_id: String,
    /// Activation level observed for this node.
    pub activation_level: AtomicF32,
    /// Confidence associated with the activation.
    pub confidence: AtomicF32,
    /// Hop count tracking traversal depth.
    pub hop_count: AtomicU16,
    /// Storage tier classification for the memory.
    pub storage_tier: StorageTier,
    /// Metadata flags for migration and throttling status.
    pub flags: ActivationFlags,
    /// Creation instant used for latency calculations.
    pub creation_time: Instant,
    /// Last update timestamp represented as nanoseconds since UNIX epoch.
    pub last_update: AtomicU64,
    /// Observed access latency for the activation.
    pub access_latency: Duration,
    /// Cached confidence multiplier for the tier.
    pub tier_confidence_factor: f32,
}

impl StorageAwareActivation {
    /// Construct a new storage-aware activation for the provided tier.
    #[must_use]
    pub fn new(memory_id: String, storage_tier: StorageTier) -> Self {
        let now = Instant::now();
        let timestamp = current_time_nanos();
        Self {
            memory_id,
            activation_level: AtomicF32::new(0.0),
            confidence: AtomicF32::new(0.0),
            hop_count: AtomicU16::new(0),
            storage_tier,
            flags: ActivationFlags::new(),
            creation_time: now,
            last_update: AtomicU64::new(timestamp),
            access_latency: Duration::default(),
            tier_confidence_factor: storage_tier.confidence_factor(),
        }
    }

    /// Construct activation metadata from a core activation record.
    #[must_use]
    pub fn from_activation_record(record: &ActivationRecord, tier: StorageTier) -> Self {
        let activation = Self::new(record.node_id.clone(), tier);
        let activation_value = record.activation_atomic().load(Ordering::Relaxed);
        activation
            .activation_level
            .store(activation_value, Ordering::Relaxed);

        let confidence_value = record.get_confidence();
        let adjusted_confidence = activation.adjust_confidence_for_tier(confidence_value);
        activation
            .confidence
            .store(adjusted_confidence, Ordering::Relaxed);

        let visits = record.visits_atomic().load(Ordering::Relaxed);
        let hop_value = u16::try_from(visits).unwrap_or(u16::MAX);
        activation.hop_count.store(hop_value, Ordering::Relaxed);

        let timestamp = record.timestamp.load(Ordering::Relaxed);
        if timestamp != 0 {
            activation.last_update.store(timestamp, Ordering::Relaxed);
        }

        activation
    }

    /// Activation threshold derived from the tier configuration.
    #[must_use]
    pub const fn tier_threshold(&self) -> f32 {
        self.storage_tier.activation_threshold()
    }

    /// Adjust the provided confidence score according to tier factor.
    #[must_use]
    pub const fn adjust_confidence_for_tier(&self, confidence: f32) -> f32 {
        (confidence * self.tier_confidence_factor).clamp(0.0, 1.0)
    }

    /// Update activation atomically and refresh metadata timestamps.
    pub fn accumulate_activation(&self, delta: f32) -> f32 {
        let mut current = self.activation_level.load(Ordering::Relaxed);
        loop {
            let updated = (current + delta).clamp(0.0, 1.0);
            if let Err(next) = self.activation_level.compare_exchange_weak(
                current,
                updated,
                Ordering::Relaxed,
                Ordering::Relaxed,
            ) {
                current = next;
                continue;
            }
            self.touch();
            return updated;
        }
    }

    /// Record the observed latency for the activation.
    pub const fn record_latency(&mut self, latency: Duration) {
        self.access_latency = latency;
    }

    /// Increment hop count and return the updated value.
    pub fn increment_hop(&self) -> u16 {
        self.hop_count.fetch_add(1, Ordering::Relaxed) + 1
    }

    /// Determine if the activation fits within the provided latency budget.
    #[must_use]
    pub fn can_access_within_budget(&self, latency_budget: Duration) -> bool {
        self.access_latency <= latency_budget
    }

    /// Update last update timestamp with the current time.
    pub fn touch(&self) {
        self.last_update
            .store(current_time_nanos(), Ordering::Relaxed);
    }

    /// Retrieve the last update timestamp in nanoseconds.
    #[must_use]
    pub fn last_update_nanos(&self) -> u64 {
        self.last_update.load(Ordering::Relaxed)
    }
}

fn current_time_nanos() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .ok()
        .and_then(|duration| u64::try_from(duration.as_nanos()).ok())
        .unwrap_or(u64::MAX)
}

#[cfg(test)]
mod tests {
    use super::{ActivationFlags, StorageAwareActivation, StorageTier};

    #[test]
    fn test_storage_aware_activation_creation() {
        let activation = StorageAwareActivation::new("mem-1".to_string(), StorageTier::Hot);
        assert_eq!(activation.memory_id, "mem-1");
        assert_eq!(activation.storage_tier, StorageTier::Hot);
        assert!(
            activation
                .activation_level
                .load(std::sync::atomic::Ordering::Relaxed)
                .abs()
                < f32::EPSILON
        );
        assert!(
            activation
                .confidence
                .load(std::sync::atomic::Ordering::Relaxed)
                .abs()
                < f32::EPSILON
        );
        assert_eq!(
            activation
                .hop_count
                .load(std::sync::atomic::Ordering::Relaxed),
            0
        );
        assert!((activation.tier_confidence_factor - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_tier_threshold_values() {
        assert!((StorageTier::Hot.activation_threshold() - 0.01).abs() < f32::EPSILON);
        assert!((StorageTier::Warm.activation_threshold() - 0.05).abs() < f32::EPSILON);
        assert!((StorageTier::Cold.activation_threshold() - 0.1).abs() < f32::EPSILON);
    }

    #[test]
    fn test_confidence_adjustment_per_tier() {
        let hot = StorageAwareActivation::new("h".into(), StorageTier::Hot);
        let warm = StorageAwareActivation::new("w".into(), StorageTier::Warm);
        let cold = StorageAwareActivation::new("c".into(), StorageTier::Cold);

        assert!((hot.adjust_confidence_for_tier(0.8) - 0.8).abs() < f32::EPSILON);
        assert!((warm.adjust_confidence_for_tier(0.8) - 0.76).abs() < 1e-6);
        assert!((cold.adjust_confidence_for_tier(0.8) - 0.72).abs() < 1e-6);
    }

    #[test]
    fn test_atomic_activation_updates() {
        use std::sync::Arc;
        use std::thread;

        let activation = Arc::new(StorageAwareActivation::new("mem".into(), StorageTier::Hot));
        let mut handles = Vec::new();

        for _ in 0..8 {
            let cloned = Arc::clone(&activation);
            handles.push(thread::spawn(move || {
                for _ in 0..1_000 {
                    cloned.accumulate_activation(0.0005);
                }
            }));
        }

        for handle in handles {
            if let Err(panic) = handle.join() {
                std::panic::resume_unwind(panic);
            }
        }

        let value = activation
            .activation_level
            .load(std::sync::atomic::Ordering::Relaxed);
        assert!(value <= 1.0);
        assert!(value > 0.0);
    }

    #[test]
    fn test_activation_flags() {
        let flags = ActivationFlags::new();
        assert_eq!(flags.bits(), 0);
        flags.insert(ActivationFlags::MIGRATION_PENDING);
        assert!(flags.contains(ActivationFlags::MIGRATION_PENDING));
        flags.remove(ActivationFlags::MIGRATION_PENDING);
        assert!(!flags.contains(ActivationFlags::MIGRATION_PENDING));
        flags.insert(ActivationFlags::LATENCY_OVER_BUDGET | ActivationFlags::THROTTLED);
        assert!(flags.contains(ActivationFlags::LATENCY_OVER_BUDGET));
        assert!(flags.contains(ActivationFlags::THROTTLED));
        flags.clear();
        assert_eq!(flags.bits(), 0);
    }

    #[test]
    fn test_tier_from_depth() {
        assert_eq!(StorageTier::from_depth(0), StorageTier::Hot);
        assert_eq!(StorageTier::from_depth(1), StorageTier::Warm);
        assert_eq!(StorageTier::from_depth(2), StorageTier::Warm);
        assert_eq!(StorageTier::from_depth(3), StorageTier::Cold);
        assert_eq!(StorageTier::from_depth(10), StorageTier::Cold);
    }
}
