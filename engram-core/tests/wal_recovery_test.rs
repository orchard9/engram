#![cfg(feature = "memory_mapped_persistence")]
#![allow(missing_docs)]

use chrono::Utc;
use engram_core::{Confidence, EpisodeBuilder, MemoryStore};
use tempfile::tempdir;

#[test]
fn wal_recovery_round_trip() {
    let temp_dir = tempdir().expect("temporary directory");

    {
        let mut store = MemoryStore::new(16)
            .with_persistence(temp_dir.path())
            .expect("enable persistence");
        store.initialize_persistence().expect("start persistence");

        let episode = EpisodeBuilder::new()
            .id("wal_recovery_episode".to_string())
            .when(Utc::now())
            .what("wal recovery payload".to_string())
            .embedding([0.1_f32; 768])
            .confidence(Confidence::HIGH)
            .build();

        let store_result = store.store(episode);
        assert!(
            store_result.activation.is_successful(),
            "expected successful store"
        );
        store.shutdown().expect("shutdown persistence");
    }

    {
        let mut store = MemoryStore::new(16)
            .with_persistence(temp_dir.path())
            .expect("enable persistence");
        let recovered = store.recover_from_wal().expect("replay wal entries");
        assert_eq!(recovered, 1, "should recover previously stored episode");
        assert_eq!(
            store.count(),
            1,
            "hot tier should contain recovered episode"
        );

        let episode = store
            .get_episode("wal_recovery_episode")
            .expect("episode available after recovery");
        assert_eq!(episode.what, "wal recovery payload");

        store.initialize_persistence().expect("start persistence");
        store.shutdown().expect("shutdown persistence");
    }
}
