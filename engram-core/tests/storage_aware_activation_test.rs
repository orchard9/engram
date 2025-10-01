//! Tests for storage-aware activation spreading.

#![allow(missing_docs)]

use std::sync::atomic::Ordering;
use std::time::Duration;

use engram_core::activation::{
    ActivationRecord, ActivationTask,
    latency_budget::LatencyBudgetManager,
    queue::PriorityActivationQueue,
    storage_aware::{ActivationFlags, StorageAwareActivation, StorageTier},
};

#[test]
fn storage_aware_activation_conversion() {
    let mut record = ActivationRecord::new("node-1".to_string(), 0.1);
    record.set_storage_tier(StorageTier::Warm);
    record.visits.fetch_add(3, Ordering::Relaxed);
    assert!(!record.accumulate_activation(0.03));
    assert!(record.accumulate_activation(0.07));

    let activation = record.to_storage_aware();
    assert_eq!(activation.storage_tier, StorageTier::Warm);
    assert!((activation.tier_threshold() - 0.05).abs() < f32::EPSILON);
    assert!((activation.activation_level.load(Ordering::Relaxed) - 0.07).abs() < 1e-6);
    assert!(activation.hop_count.load(Ordering::Relaxed) >= 3);
}

#[test]
fn priority_queue_respects_tier_priorities() {
    let queue = PriorityActivationQueue::new();

    let hot_task = ActivationTask::new("hot".to_string(), 1.0, 0.6, 1.0, 0, 4)
        .with_storage_tier(StorageTier::Hot);
    let warm_task = ActivationTask::new("warm".to_string(), 0.5, 0.5, 0.8, 1, 4)
        .with_storage_tier(StorageTier::Warm);
    let cold_task = ActivationTask::new("cold".to_string(), 0.5, 0.1, 0.1, 2, 4)
        .with_storage_tier(StorageTier::Cold);

    queue.push(cold_task);
    queue.push(warm_task);
    queue.push(hot_task);

    let first = queue.pop().expect("expected hot task");
    assert_eq!(first.storage_tier, Some(StorageTier::Hot));

    let second = queue.pop().expect("expected warm task");
    assert_eq!(second.storage_tier, Some(StorageTier::Warm));

    let third = queue.pop().expect("expected cold task");
    assert_eq!(third.storage_tier, Some(StorageTier::Cold));
    assert!(queue.pop().is_none());
}

#[test]
fn latency_budget_manager_enforces_budget() {
    let manager = LatencyBudgetManager::new();
    let mut activation = StorageAwareActivation::new("latency".into(), StorageTier::Warm);
    activation.record_latency(Duration::from_millis(2));
    let budget = manager.budget_for(StorageTier::Warm);
    assert!(!activation.can_access_within_budget(budget));
    assert!(!manager.within_budget(StorageTier::Warm, Duration::from_millis(2)));

    activation
        .flags
        .insert(ActivationFlags::LATENCY_OVER_BUDGET);
    assert!(
        activation
            .flags
            .contains(ActivationFlags::LATENCY_OVER_BUDGET)
    );
}
