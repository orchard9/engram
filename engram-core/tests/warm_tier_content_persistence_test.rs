//! Test warm tier content persistence fix
//!
//! Validates that content is correctly stored and retrieved from warm tier,
//! not replaced with placeholder strings.

#[cfg(feature = "memory_mapped_persistence")]
mod warm_tier_content_tests {
    use engram_core::{
        Confidence, CueBuilder, EpisodeBuilder, Memory,
        storage::{StorageMetrics, StorageTierBackend, WarmTier},
    };
    use std::sync::Arc;
    use tempfile::TempDir;

    fn create_test_memory_with_content(id: &str, content: &str, activation: f32) -> Arc<Memory> {
        let episode = EpisodeBuilder::new()
            .id(id.to_string())
            .when(chrono::Utc::now())
            .what(content.to_string())
            .embedding([0.5f32; 768])
            .confidence(Confidence::HIGH)
            .build();

        Arc::new(Memory::from_episode(episode, activation))
    }

    #[tokio::test]
    async fn test_warm_tier_content_round_trip() {
        let temp_dir = TempDir::new().expect("failed to create temp directory");
        let metrics = Arc::new(StorageMetrics::new());

        let warm_tier = WarmTier::new(temp_dir.path().join("warm_tier.dat"), 1000, metrics)
            .expect("failed to create warm tier");

        // Test data
        let test_cases = vec![
            ("short", "Short content"),
            (
                "long",
                "This is a much longer piece of content that spans multiple cache lines and tests the variable-length storage mechanism with realistic user data. It should be preserved exactly as written, including all punctuation, spacing, and formatting.",
            ),
            ("unicode", "Unicode test: 你好世界 🚀 Привет мир"),
            ("empty", ""),
            (
                "special_chars",
                "Line1\nLine2\tTabbed\r\nWindows line ending",
            ),
        ];

        for (id, original_content) in test_cases {
            // Create memory with specific content
            let memory = create_test_memory_with_content(id, original_content, 0.8);

            // Store to warm tier
            warm_tier
                .store(memory)
                .await
                .expect("failed to store memory");

            // Retrieve via iteration
            let memories: Vec<_> = warm_tier.iter_memories().collect();
            let found = memories
                .iter()
                .find(|(found_id, _)| found_id == id)
                .unwrap_or_else(|| panic!("memory {id} not found in iteration"));

            // Verify content preserved exactly
            assert_eq!(
                found.1.what, original_content,
                "Content mismatch for '{}': expected '{}', got '{}'",
                id, original_content, found.1.what
            );
        }
    }

    #[tokio::test]
    async fn test_warm_tier_multiple_memories_content_isolation() {
        let temp_dir = TempDir::new().expect("failed to create temp directory");
        let metrics = Arc::new(StorageMetrics::new());

        let warm_tier = WarmTier::new(temp_dir.path().join("warm_tier.dat"), 1000, metrics)
            .expect("failed to create warm tier");

        // Store multiple memories with different content
        let memory1 = create_test_memory_with_content("mem1", "First memory content", 0.7);
        let memory2 = create_test_memory_with_content("mem2", "Second memory content", 0.6);
        let memory3 = create_test_memory_with_content("mem3", "Third memory content", 0.5);

        warm_tier
            .store(memory1)
            .await
            .expect("failed to store memory1");
        warm_tier
            .store(memory2)
            .await
            .expect("failed to store memory2");
        warm_tier
            .store(memory3)
            .await
            .expect("failed to store memory3");

        // Verify each memory has its own content
        let memories: Vec<_> = warm_tier.iter_memories().collect();
        assert_eq!(memories.len(), 3, "expected 3 memories");

        for (id, episode) in memories {
            match id.as_str() {
                "mem1" => assert_eq!(episode.what, "First memory content"),
                "mem2" => assert_eq!(episode.what, "Second memory content"),
                "mem3" => assert_eq!(episode.what, "Third memory content"),
                _ => panic!("unexpected memory id: {id}"),
            }
        }
    }

    #[tokio::test]
    async fn test_warm_tier_recall_with_content() {
        let temp_dir = TempDir::new().expect("failed to create temp directory");
        let metrics = Arc::new(StorageMetrics::new());

        let warm_tier = WarmTier::new(temp_dir.path().join("warm_tier.dat"), 1000, metrics)
            .expect("failed to create warm tier");

        let test_content = "Test content for recall";
        let memory = create_test_memory_with_content("recall_test", test_content, 0.8);

        warm_tier
            .store(memory)
            .await
            .expect("failed to store memory");

        // Recall via cue
        let cue = CueBuilder::new()
            .id("test_cue".to_string())
            .embedding_search([0.5f32; 768], Confidence::LOW)
            .max_results(10)
            .build();

        let results = warm_tier
            .recall(&cue)
            .await
            .expect("failed to recall memories");

        assert!(!results.is_empty(), "recall should return results");

        // Find our test memory
        let found = results
            .iter()
            .find(|(episode, _)| episode.id == "recall_test")
            .expect("test memory not found in recall results");

        assert_eq!(
            found.0.what, test_content,
            "recalled content should match original"
        );
    }

    #[tokio::test]
    async fn test_warm_tier_large_content() {
        let temp_dir = TempDir::new().expect("failed to create temp directory");
        let metrics = Arc::new(StorageMetrics::new());

        let warm_tier = WarmTier::new(temp_dir.path().join("warm_tier.dat"), 1000, metrics)
            .expect("failed to create warm tier");

        // Create a large content string (10KB)
        let large_content = "A".repeat(10_000);
        let memory = create_test_memory_with_content("large", &large_content, 0.7);

        warm_tier
            .store(memory)
            .await
            .expect("failed to store large memory");

        // Retrieve and verify
        let memories: Vec<_> = warm_tier.iter_memories().collect();
        let found = memories
            .iter()
            .find(|(id, _)| id == "large")
            .expect("large memory not found");

        assert_eq!(
            found.1.what.len(),
            10_000,
            "large content should be preserved"
        );
        assert_eq!(found.1.what, large_content, "large content should match");
    }

    #[tokio::test]
    async fn test_warm_tier_none_content() {
        let temp_dir = TempDir::new().expect("failed to create temp directory");
        let metrics = Arc::new(StorageMetrics::new());

        let warm_tier = WarmTier::new(temp_dir.path().join("warm_tier.dat"), 1000, metrics)
            .expect("failed to create warm tier");

        // Create memory without content
        let mut memory = Memory::new("no_content".to_string(), [0.5f32; 768], Confidence::HIGH);
        memory.content = None;

        warm_tier
            .store(Arc::new(memory))
            .await
            .expect("failed to store memory");

        // Retrieve and verify
        let memories: Vec<_> = warm_tier.iter_memories().collect();
        let found = memories
            .iter()
            .find(|(id, _)| id == "no_content")
            .expect("memory not found");

        // Should have placeholder for None content
        assert_eq!(found.1.what, "Memory no_content");
    }

    #[tokio::test]
    async fn test_warm_tier_content_stress() {
        let temp_dir = TempDir::new().expect("failed to create temp directory");
        let metrics = Arc::new(StorageMetrics::new());

        let warm_tier = WarmTier::new(temp_dir.path().join("warm_tier.dat"), 1000, metrics)
            .expect("failed to create warm tier");

        // Store 100 memories with varying content sizes
        let expected_contents: Vec<(String, String)> = (0..100)
            .map(|i| {
                let id = format!("mem_{i}");
                let content = format!("Content for memory {} - {}", i, "padding ".repeat(i % 10));
                (id, content)
            })
            .collect();

        for (id, content) in &expected_contents {
            let memory = create_test_memory_with_content(id, content, 0.5);
            warm_tier
                .store(memory)
                .await
                .expect("failed to store memory");
        }

        // Verify all memories have correct content
        let memories: Vec<_> = warm_tier.iter_memories().collect();
        assert_eq!(memories.len(), 100, "expected 100 memories");

        for (id, expected_content) in expected_contents {
            let found = memories
                .iter()
                .find(|(found_id, _)| found_id == &id)
                .unwrap_or_else(|| panic!("memory {id} not found"));

            assert_eq!(found.1.what, expected_content, "content mismatch for {id}");
        }
    }

    #[tokio::test]
    async fn test_warm_tier_utf8_edge_cases() {
        let temp_dir = TempDir::new().expect("failed to create temp directory");
        let metrics = Arc::new(StorageMetrics::new());

        let warm_tier = WarmTier::new(temp_dir.path().join("warm_tier.dat"), 1000, metrics)
            .expect("failed to create warm tier");

        let test_cases = vec![
            ("emoji", "🚀🌟💻🔥⚡"),
            ("chinese", "我能吞下玻璃而不伤身体"),
            ("arabic", "أنا قادر على أكل الزجاج دون أن يؤلمني"),
            ("hebrew", "אני יכול לאכול זכוכית וזה לא מזיק לי"),
            ("mixed", "Hello 你好 мир 🌍"),
        ];

        for (id, content) in test_cases {
            let memory = create_test_memory_with_content(id, content, 0.5);
            warm_tier
                .store(memory)
                .await
                .expect("failed to store memory");

            let memories: Vec<_> = warm_tier.iter_memories().collect();
            let found = memories
                .iter()
                .find(|(found_id, _)| found_id == id)
                .unwrap_or_else(|| panic!("memory {id} not found"));

            assert_eq!(found.1.what, content, "utf8 content mismatch for {id}");
        }
    }
}
