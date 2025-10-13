//! Integration tests for feature abstraction layer
//!
//! Tests the feature provider pattern and graceful degradation

use chrono::Utc;
use engram_core::features::{FeatureProvider, FeatureRegistry};
use engram_core::{Confidence, Episode};
use std::path::PathBuf;
use std::time::Duration;

#[test]
fn test_feature_registry_initialization() {
    let registry = FeatureRegistry::new();

    // Should always have null implementations
    assert!(registry.get("index_null").is_some());
    assert!(registry.get("storage_null").is_some());
    assert!(registry.get("decay_null").is_some());
    assert!(registry.get("monitoring_null").is_some());
    assert!(registry.get("completion_null").is_some());
}

#[test]
fn test_graceful_fallback() {
    let registry = FeatureRegistry::new();

    // Get best provider should return either real or null implementation
    let index_provider = registry.get_best("index");
    assert!(index_provider.name().contains("index"));

    let storage_provider = registry.get_best("storage");
    assert!(storage_provider.name().contains("storage"));
}

#[test]
fn test_null_index_provider() {
    use engram_core::features::index::IndexProvider;
    use engram_core::features::null_impls::NullIndexProvider;

    let provider = NullIndexProvider::new();
    assert!(!provider.is_enabled());
    assert_eq!(provider.name(), "index_null");

    let _index = provider.create_index();

    // Create test episodes
    let episodes = vec![
        Episode {
            id: "test1".to_string(),
            when: Utc::now(),
            where_location: None,
            who: None,
            what: "Test content 1".to_string(),
            embedding: [0.1; 768],
            embedding_provenance: None, // Test episode doesn't need provenance
            encoding_confidence: Confidence::HIGH,
            vividness_confidence: Confidence::MEDIUM,
            reliability_confidence: Confidence::HIGH,
            last_recall: Utc::now(),
            recall_count: 1,
            decay_rate: 1.0,
        },
        Episode {
            id: "test2".to_string(),
            when: Utc::now(),
            where_location: None,
            who: None,
            what: "Test content 2".to_string(),
            embedding: [0.2; 768],
            embedding_provenance: None, // Test episode doesn't need provenance
            encoding_confidence: Confidence::MEDIUM,
            vividness_confidence: Confidence::LOW,
            reliability_confidence: Confidence::MEDIUM,
            last_recall: Utc::now(),
            recall_count: 2,
            decay_rate: 0.9,
        },
    ];

    // Test build
    let mut index = provider.create_index();
    assert!(index.build(&episodes).is_ok());
    assert_eq!(index.size(), 2);

    // Test search
    let query = [0.15; 768];
    let results = index.search(&query, 1).unwrap();
    assert_eq!(results.len(), 1);
}

#[test]
fn test_null_storage_provider() {
    use engram_core::features::null_impls::NullStorageProvider;
    use engram_core::features::storage::StorageProvider;

    let provider = NullStorageProvider::new();
    assert!(!provider.is_enabled());
    assert_eq!(provider.name(), "storage_null");

    let path = PathBuf::from("/tmp/test_storage");
    let mut storage = provider.create_storage(&path);

    // Create test episode
    let episode = Episode {
        id: "test".to_string(),
        when: Utc::now(),
        where_location: None,
        who: None,
        what: "Test content".to_string(),
        embedding: [0.1; 768],
        embedding_provenance: None, // Test episode doesn't need provenance
        encoding_confidence: Confidence::HIGH,
        vividness_confidence: Confidence::MEDIUM,
        reliability_confidence: Confidence::HIGH,
        last_recall: Utc::now(),
        recall_count: 1,
        decay_rate: 1.0,
    };

    // Test store and retrieve
    assert!(storage.store(&episode).is_ok());
    let retrieved = storage.retrieve("test").unwrap();
    assert!(retrieved.is_some());
    assert_eq!(retrieved.unwrap().id, "test");

    // Test stats
    let stats = storage.stats();
    assert_eq!(stats.total_items, 1);
}

#[test]
fn test_null_decay_provider() {
    use engram_core::features::decay::DecayProvider;
    use engram_core::features::null_impls::NullDecayProvider;

    let provider = NullDecayProvider::new();
    assert!(!provider.is_enabled());
    assert_eq!(provider.name(), "decay_null");

    let decay = provider.create_decay();

    // Test decay calculation
    let elapsed = Duration::from_secs(3600); // 1 hour
    let factor = decay.calculate_decay(elapsed);
    assert!(factor > 0.0 && factor <= 1.0);

    // Test applying decay
    let mut episode = Episode {
        id: "test".to_string(),
        when: Utc::now(),
        where_location: None,
        who: None,
        what: "Test content".to_string(),
        embedding: [0.1; 768],
        embedding_provenance: None, // Test episode doesn't need provenance
        encoding_confidence: Confidence::HIGH,
        vividness_confidence: Confidence::MEDIUM,
        reliability_confidence: Confidence::HIGH,
        last_recall: Utc::now(),
        recall_count: 1,
        decay_rate: 1.0,
    };

    decay.apply_decay(&mut episode, elapsed);
    assert!(episode.decay_rate > 0.0 && episode.decay_rate <= 1.0);
}

#[test]
fn test_null_monitoring_provider() {
    use engram_core::features::monitoring::MonitoringProvider;
    use engram_core::features::null_impls::NullMonitoringProvider;

    let provider = NullMonitoringProvider::new();
    assert!(!provider.is_enabled());
    assert_eq!(provider.name(), "monitoring_null");

    let monitoring = provider.create_monitoring();

    // Test that null monitoring doesn't crash
    monitoring.record_counter("test_counter", 1, &[]);
    monitoring.record_gauge("test_gauge", 0.5, &[]);
    monitoring.record_histogram("test_histogram", 100.0, &[]);

    let timer = monitoring.start_timer("test_timer");
    timer.stop();

    let metric = monitoring.get_metric("test").unwrap();
    match metric {
        engram_core::features::monitoring::MetricValue::Counter(v) => assert_eq!(v, 0),
        _ => panic!("Expected counter metric"),
    }
}

#[test]
fn test_null_completion_provider() {
    use engram_core::Memory;
    use engram_core::features::completion::CompletionProvider;
    use engram_core::features::null_impls::NullCompletionProvider;
    use std::sync::Arc;

    let provider = NullCompletionProvider::new();
    assert!(!provider.is_enabled());
    assert_eq!(provider.name(), "completion_null");

    let completion = provider.create_completion();

    // Create test memories
    let partial = Memory::new("partial".to_string(), [0.5; 768], Confidence::MEDIUM);

    let candidate = Arc::new(Memory::new(
        "candidate".to_string(),
        [0.6; 768],
        Confidence::HIGH,
    ));

    // Test completion
    let matches = completion
        .complete(&partial, &[candidate.clone()], 0.0)
        .unwrap();
    assert!(!matches.is_empty());

    // Test that sequence prediction fails gracefully
    let prediction = completion.predict_next(&[partial.clone()], &[candidate]);
    assert!(prediction.is_err());

    // Test gap filling
    let filled = completion
        .fill_gaps(&partial, &[true, false, true])
        .unwrap();
    assert_eq!(filled.id, partial.id);
}

#[test]
fn test_feature_compatibility() {
    let registry = FeatureRegistry::new();

    // Test compatible features
    assert!(registry.check_compatibility(&["index", "storage"]).is_ok());
    assert!(
        registry
            .check_compatibility(&["decay", "monitoring"])
            .is_ok()
    );

    // All features should be compatible by default
    assert!(
        registry
            .check_compatibility(&["index", "storage", "decay", "monitoring", "completion"])
            .is_ok()
    );
}

#[test]
fn test_feature_status_summary() {
    let registry = FeatureRegistry::new();
    let summary = registry.status_summary();

    // Should contain status for all feature types
    assert!(summary.contains("index:"));
    assert!(summary.contains("storage:"));
    assert!(summary.contains("decay:"));
    assert!(summary.contains("monitoring:"));
    assert!(summary.contains("completion:"));
}
