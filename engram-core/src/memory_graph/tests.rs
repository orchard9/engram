//! Tests for memory graph consolidation and migration
//!
//! This module ensures that the unified memory graph architecture
//! maintains backward compatibility and provides equivalent functionality
//! across all backend implementations.

#[cfg(test)]
mod migration_tests {
    use crate::memory::{Memory, MemoryBuilder};
    use crate::memory_graph::{
        DashMapBackend, GraphConfig, HashMapBackend, InfallibleBackend, UnifiedMemoryGraph,
    };
    use crate::numeric::{saturating_f32_from_f64, u64_to_f64};
    use crate::{Confidence, Cue, CueType};
    use std::convert::TryFrom;
    use std::fmt::Debug;
    use std::sync::Arc;
    use std::thread;
    use std::time::SystemTime;

    type TestResult<T = ()> = Result<T, String>;

    fn ensure(condition: bool, message: impl Into<String>) -> TestResult {
        if condition {
            Ok(())
        } else {
            Err(message.into())
        }
    }

    fn ensure_eq<T>(actual: &T, expected: &T, context: &str) -> TestResult
    where
        T: PartialEq + Debug,
    {
        if actual == expected {
            Ok(())
        } else {
            Err(format!("{context}: expected {expected:?}, got {actual:?}"))
        }
    }

    trait IntoTestResult<T> {
        fn into_test_result(self, context: &str) -> TestResult<T>;
    }

    impl<T, E: std::fmt::Debug> IntoTestResult<T> for Result<T, E> {
        fn into_test_result(self, context: &str) -> TestResult<T> {
            self.map_err(|err| format!("{context}: {err:?}"))
        }
    }

    impl<T> IntoTestResult<T> for Option<T> {
        fn into_test_result(self, context: &str) -> TestResult<T> {
            self.ok_or_else(|| context.to_string())
        }
    }

    fn usize_to_f32(value: usize) -> f32 {
        debug_assert!(value < (1 << 24));
        let value_u64 = u64::try_from(value).unwrap_or(u64::MAX);
        saturating_f32_from_f64(u64_to_f64(value_u64))
    }

    /// Helper function to create a memory with a specific embedding pattern
    fn create_memory_with_embedding(id: &str, embedding: &[f32], confidence: Confidence) -> Memory {
        let mut arr = [0.0; 768];
        for (i, &v) in embedding.iter().enumerate().take(768) {
            arr[i] = v;
        }
        Memory::new(id.to_string(), arr, confidence)
    }

    /// Test that the deprecated `MemoryGraph` still works during migration
    #[test]
    #[allow(deprecated)]
    fn test_backward_compatibility() -> TestResult {
        use crate::MemoryNode;
        use crate::graph::MemoryGraph;

        let mut old_graph = MemoryGraph::new();

        let node = MemoryNode {
            id: "test_node".to_string(),
            content: b"test content".to_vec(),
            embedding: Some(vec![0.1; 768]),
            activation: 0.8,
            created_at: SystemTime::now(),
            last_accessed: SystemTime::now(),
            confidence: Confidence::HIGH,
            _state: std::marker::PhantomData,
        };

        old_graph.store(node);
        ensure_eq(&old_graph.len(), &1_usize, "legacy graph len")?;

        let retrieved = old_graph
            .get("test_node")
            .into_test_result("expected stored node")?;
        ensure_eq(
            &retrieved.content,
            &b"test content".to_vec(),
            "legacy node content",
        )?;

        Ok(())
    }

    /// Test that all backends produce equivalent results for basic operations
    #[test]
    fn test_backend_equivalence() -> TestResult {
        let mut embedding = [0.0; 768];
        embedding.iter_mut().for_each(|v| *v = 0.1);
        let memory = Memory::new("test_memory".to_string(), embedding, Confidence::HIGH);

        // Create graphs with different backends
        let hashmap_graph =
            UnifiedMemoryGraph::new(HashMapBackend::default(), GraphConfig::default());
        let dashmap_graph =
            UnifiedMemoryGraph::new(DashMapBackend::default(), GraphConfig::default());
        let infallible_graph =
            UnifiedMemoryGraph::new(InfallibleBackend::default(), GraphConfig::default());

        // Store the same memory in all backends
        let id1 = hashmap_graph
            .store_memory(memory.clone())
            .into_test_result("store memory in HashMap backend")?;
        let id2 = dashmap_graph
            .store_memory(memory.clone())
            .into_test_result("store memory in DashMap backend")?;
        let id3 = infallible_graph
            .store_memory(memory)
            .into_test_result("store memory in Infallible backend")?;

        // All should successfully retrieve
        ensure(
            hashmap_graph
                .retrieve(&id1)
                .into_test_result("retrieve from HashMap backend")?
                .is_some(),
            "hashmap backend should retrieve stored memory",
        )?;
        ensure(
            dashmap_graph
                .retrieve(&id2)
                .into_test_result("retrieve from DashMap backend")?
                .is_some(),
            "dashmap backend should retrieve stored memory",
        )?;
        ensure(
            infallible_graph
                .retrieve(&id3)
                .into_test_result("retrieve from Infallible backend")?
                .is_some(),
            "infallible backend should retrieve stored memory",
        )?;

        // All should have count of 1
        ensure_eq(&hashmap_graph.count(), &1_usize, "hashmap count")?;
        ensure_eq(&dashmap_graph.count(), &1_usize, "dashmap count")?;
        ensure_eq(&infallible_graph.count(), &1_usize, "infallible count")?;

        Ok(())
    }

    /// Test concurrent access with `DashMapBackend`
    #[test]
    fn test_concurrent_access() -> TestResult {
        let graph = Arc::new(UnifiedMemoryGraph::new(
            DashMapBackend::default(),
            GraphConfig::default(),
        ));

        let mut handles = vec![];

        // Spawn multiple threads to store memories concurrently
        for i in 0..10 {
            let graph_clone = graph.clone();
            let handle = thread::spawn(move || {
                let value = usize_to_f32(i);
                let embedding = vec![value; 768];
                let memory = create_memory_with_embedding(
                    &format!("memory_{i}"),
                    &embedding,
                    Confidence::MEDIUM,
                );
                graph_clone.store_memory(memory)
            });
            handles.push(handle);
        }

        // Collect all IDs
        let mut ids = Vec::with_capacity(10);
        for handle in handles {
            let id = handle
                .join()
                .into_test_result("store thread panicked")?
                .into_test_result("store memory succeeded")?;
            ids.push(id);
        }

        // Verify all memories were stored
        ensure_eq(&graph.count(), &10_usize, "concurrent store count")?;

        // Verify each memory can be retrieved
        for id in ids {
            ensure(
                graph
                    .retrieve(&id)
                    .into_test_result("retrieve stored memory")?
                    .is_some(),
                "retrieved memory should exist",
            )?;
        }

        Ok(())
    }

    /// Test graph operations with `GraphBackend` implementations
    #[test]
    fn test_graph_operations() -> TestResult {
        let graph = UnifiedMemoryGraph::new(DashMapBackend::default(), GraphConfig::default());

        // Store two memories
        let mut embedding1 = [0.0; 768];
        embedding1.iter_mut().for_each(|v| *v = 0.1);
        let memory1 = Memory::new("memory1".to_string(), embedding1, Confidence::HIGH);

        let mut embedding2 = [0.0; 768];
        embedding2.iter_mut().for_each(|v| *v = 0.2);
        let memory2 = Memory::new("memory2".to_string(), embedding2, Confidence::HIGH);

        let id1 = graph
            .store_memory(memory1)
            .into_test_result("store first memory")?;
        let id2 = graph
            .store_memory(memory2)
            .into_test_result("store second memory")?;

        // Add edge between them
        graph
            .add_edge(id1, id2, 0.8)
            .into_test_result("add edge between memories")?;

        // Check neighbors
        let neighbors = graph
            .get_neighbors(&id1)
            .into_test_result("neighbors should be retrievable")?;
        ensure_eq(&neighbors.len(), &1_usize, "neighbor count")?;
        ensure_eq(&neighbors[0].0, &id2, "neighbor id")?;
        ensure(
            (neighbors[0].1 - 0.8).abs() < 1e-6,
            "neighbor weight should match",
        )?;

        // Test BFS traversal
        let traversal = graph
            .traverse_bfs(&id1, 2)
            .into_test_result("BFS traversal should succeed")?;
        ensure(traversal.contains(&id1), "BFS should include start node")?;
        ensure(traversal.contains(&id2), "BFS should include neighbor")?;

        Ok(())
    }

    /// Test similarity search across backends
    #[test]
    fn test_similarity_search() -> TestResult {
        let embedding1 = vec![1.0, 0.0, 0.0];
        let embedding2 = vec![0.9, 0.1, 0.0];
        let embedding3 = vec![0.0, 0.0, 1.0];

        let memory1 = create_memory_with_embedding("similar1", &embedding1, Confidence::HIGH);
        let memory2 = create_memory_with_embedding("similar2", &embedding2, Confidence::HIGH);
        let memory3 = create_memory_with_embedding("different", &embedding3, Confidence::HIGH);

        // Test with HashMap backend
        let graph = UnifiedMemoryGraph::new(HashMapBackend::default(), GraphConfig::default());

        graph
            .store_memory(memory1)
            .into_test_result("store first similar memory")?;
        graph
            .store_memory(memory2)
            .into_test_result("store second similar memory")?;
        graph
            .store_memory(memory3)
            .into_test_result("store different memory")?;

        // Search for similar memories
        // Need to create a 768-dimensional array for the query
        let mut query_embedding = [0.0; 768];
        for (i, &v) in embedding1.iter().enumerate().take(768) {
            query_embedding[i] = v;
        }
        let results = graph
            .similarity_search(&query_embedding, 2, Confidence::LOW)
            .into_test_result("similarity search should succeed")?;

        // Should find the two similar memories
        ensure_eq(&results.len(), &2_usize, "similarity result length")?;

        // First result should be the exact match or very similar
        ensure(results[0].1 > 0.8, "top similarity should exceed threshold")?;

        Ok(())
    }

    /// Test infallible backend graceful degradation
    #[test]
    fn test_infallible_degradation() {
        let backend = InfallibleBackend::new(5); // Small capacity for testing
        let graph = UnifiedMemoryGraph::new(backend, GraphConfig::default());

        // Store more memories than capacity
        for i in 0..10 {
            let embedding = vec![usize_to_f32(i); 768];
            let memory = create_memory_with_embedding(
                &format!("memory_{i}"),
                &embedding,
                Confidence::MEDIUM,
            );
            // Should not error even when over capacity
            assert!(graph.store_memory(memory).is_ok());
        }

        // Count should be limited by eviction
        assert!(graph.count() <= 5);
    }

    /// Test cue-based recall
    #[test]
    fn test_recall_with_cue() -> TestResult {
        let graph = UnifiedMemoryGraph::new(HashMapBackend::default(), GraphConfig::default());

        // Store memories with different embeddings
        for i in 0..5 {
            let mut embedding = vec![0.0; 768];
            embedding[i] = 1.0;
            let id = format!("memory_{i}");
            let memory = MemoryBuilder::new()
                .id(id.clone())
                .embedding({
                    let mut arr = [0.0; 768];
                    for (idx, &v) in embedding.iter().enumerate().take(768) {
                        arr[idx] = v;
                    }
                    arr
                })
                .confidence(Confidence::HIGH)
                .content(id.clone()) // Set content to match the ID
                .build();
            graph
                .store_memory(memory)
                .into_test_result("store memory for cue recall test")?;
        }

        // Create cue for recall
        let mut target_embedding = [0.0; 768];
        target_embedding[2] = 1.0;

        let cue = Cue {
            id: "test_cue".to_string(),
            cue_type: CueType::Embedding {
                vector: target_embedding,
                threshold: Confidence::LOW,
            },
            cue_confidence: Confidence::HIGH,
            result_threshold: Confidence::LOW,
            max_results: 3,
            embedding_provenance: None, // Test cue doesn't need provenance
        };

        let recalled = graph
            .recall(&cue)
            .into_test_result("recall with cue should succeed")?;

        // Should recall at least one memory
        ensure(!recalled.is_empty(), "cue recall should return results")?;

        // The best match should be memory_2
        ensure(
            recalled[0]
                .content
                .as_ref()
                .is_some_and(|c| c.contains("memory_2")),
            "expected memory_2 to be recalled",
        )?;

        Ok(())
    }

    /// Test that the new architecture doesn't break existing functionality
    #[test]
    fn test_no_regression() -> TestResult {
        use crate::graph::{create_concurrent_graph, create_simple_graph};

        // Test factory functions
        let simple = create_simple_graph();
        let concurrent = create_concurrent_graph();

        // Both should work correctly
        let embedding = vec![0.5f32; 768];
        let id1 = simple
            .store_memory(create_memory_with_embedding(
                "test",
                &embedding,
                Confidence::MEDIUM,
            ))
            .into_test_result("store memory in simple graph")?;
        let id2 = concurrent
            .store_memory(create_memory_with_embedding(
                "test",
                &embedding,
                Confidence::MEDIUM,
            ))
            .into_test_result("store memory in concurrent graph")?;

        ensure(
            simple
                .retrieve(&id1)
                .into_test_result("retrieve from simple graph")?
                .is_some(),
            "simple graph retrieval",
        )?;
        ensure(
            concurrent
                .retrieve(&id2)
                .into_test_result("retrieve from concurrent graph")?
                .is_some(),
            "concurrent graph retrieval",
        )?;

        Ok(())
    }

    /// Performance benchmark comparison (not a pass/fail test)
    #[test]
    #[ignore] // Run with --ignored flag for benchmarking
    fn benchmark_backends() -> TestResult {
        use std::time::Instant;

        let memory_count: usize = 1000;
        let total_f32 = usize_to_f32(memory_count);
        let memories: Vec<Memory> = (0..memory_count)
            .map(|i| {
                let ratio = usize_to_f32(i) / total_f32;
                let embedding = vec![ratio; 768];
                create_memory_with_embedding(&format!("memory_{i}"), &embedding, Confidence::MEDIUM)
            })
            .collect();

        // Benchmark HashMap backend
        let hashmap_graph =
            UnifiedMemoryGraph::new(HashMapBackend::default(), GraphConfig::default());
        let start = Instant::now();
        for memory in &memories {
            hashmap_graph
                .store_memory(memory.clone())
                .into_test_result("store memory in HashMap backend")?;
        }
        let hashmap_time = start.elapsed();
        println!("HashMap backend: {hashmap_time:?} for {memory_count} stores");

        // Benchmark DashMap backend
        let dashmap_graph =
            UnifiedMemoryGraph::new(DashMapBackend::default(), GraphConfig::default());
        let start = Instant::now();
        for memory in &memories {
            dashmap_graph
                .store_memory(memory.clone())
                .into_test_result("store memory in DashMap backend")?;
        }
        let dashmap_time = start.elapsed();
        println!("DashMap backend: {dashmap_time:?} for {memory_count} stores");

        // Benchmark Infallible backend
        let infallible_graph = UnifiedMemoryGraph::new(
            InfallibleBackend::new(memory_count * 2),
            GraphConfig::default(),
        );
        let start = Instant::now();
        for memory in memories {
            infallible_graph
                .store_memory(memory)
                .into_test_result("store memory in Infallible backend")?;
        }
        let infallible_time = start.elapsed();
        println!("Infallible backend: {infallible_time:?} for {memory_count} stores");

        // No regression assertion - just informational
        println!("\nPerformance comparison complete. All backends functional.");

        Ok(())
    }
}
