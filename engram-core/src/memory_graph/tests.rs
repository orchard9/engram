//! Tests for memory graph consolidation and migration
//!
//! This module ensures that the unified memory graph architecture
//! maintains backward compatibility and provides equivalent functionality
//! across all backend implementations.

#[cfg(test)]
mod migration_tests {
    use crate::memory_graph::{
        UnifiedMemoryGraph, HashMapBackend, DashMapBackend, InfallibleBackend,
        GraphConfig,
    };
    use crate::memory::{Memory, MemoryBuilder};
    use crate::{Confidence, Cue, CueType};
    use uuid::Uuid;
    use std::sync::Arc;
    use std::thread;
    
    /// Helper function to create a memory with a uniform embedding
    fn create_memory(id: &str, value: f32, confidence: Confidence) -> Memory {
        let mut embedding = [0.0; 768];
        embedding.iter_mut().for_each(|v| *v = value);
        Memory::new(id.to_string(), embedding, confidence)
    }
    
    /// Helper function to create a memory with a specific embedding pattern
    fn create_memory_with_embedding(id: &str, embedding: Vec<f32>, confidence: Confidence) -> Memory {
        let mut arr = [0.0; 768];
        for (i, &v) in embedding.iter().enumerate().take(768) {
            arr[i] = v;
        }
        Memory::new(id.to_string(), arr, confidence)
    }

    /// Test that the deprecated MemoryGraph still works during migration
    #[test]
    #[allow(deprecated)]
    fn test_backward_compatibility() {
        use crate::graph::MemoryGraph;
        use crate::MemoryNode;
        
        let mut old_graph = MemoryGraph::new();
        
        use std::time::SystemTime;
        let node = MemoryNode {
            id: "test_node".to_string(),
            content: "test content".as_bytes().to_vec(),
            embedding: Some(vec![0.1; 768]),
            activation: 0.8,
            created_at: SystemTime::now(),
            last_accessed: SystemTime::now(),
            confidence: Confidence::HIGH,
            _state: std::marker::PhantomData,
        };
        
        old_graph.store(node.clone());
        assert_eq!(old_graph.len(), 1);
        
        let retrieved = old_graph.get("test_node");
        assert!(retrieved.is_some());
        assert_eq!(retrieved.unwrap().content, "test content".as_bytes().to_vec());
    }
    
    /// Test that all backends produce equivalent results for basic operations
    #[test]
    fn test_backend_equivalence() {
        let mut embedding = [0.0; 768];
        embedding.iter_mut().for_each(|v| *v = 0.1);
        let memory = Memory::new("test_memory".to_string(), embedding, Confidence::HIGH);
        
        // Create graphs with different backends
        let hashmap_graph = UnifiedMemoryGraph::new(
            HashMapBackend::default(),
            GraphConfig::default()
        );
        let dashmap_graph = UnifiedMemoryGraph::new(
            DashMapBackend::default(),
            GraphConfig::default()
        );
        let infallible_graph = UnifiedMemoryGraph::new(
            InfallibleBackend::default(),
            GraphConfig::default()
        );
        
        // Store the same memory in all backends
        let id1 = hashmap_graph.store_memory(memory.clone()).unwrap();
        let id2 = dashmap_graph.store_memory(memory.clone()).unwrap();
        let id3 = infallible_graph.store_memory(memory.clone()).unwrap();
        
        // All should successfully retrieve
        assert!(hashmap_graph.retrieve(&id1).unwrap().is_some());
        assert!(dashmap_graph.retrieve(&id2).unwrap().is_some());
        assert!(infallible_graph.retrieve(&id3).unwrap().is_some());
        
        // All should have count of 1
        assert_eq!(hashmap_graph.count(), 1);
        assert_eq!(dashmap_graph.count(), 1);
        assert_eq!(infallible_graph.count(), 1);
    }
    
    /// Test concurrent access with DashMapBackend
    #[test]
    fn test_concurrent_access() {
        let graph = Arc::new(UnifiedMemoryGraph::new(
            DashMapBackend::default(),
            GraphConfig::default()
        ));
        
        let mut handles = vec![];
        
        // Spawn multiple threads to store memories concurrently
        for i in 0..10 {
            let graph_clone = graph.clone();
            let handle = thread::spawn(move || {
                let memory = create_memory_with_embedding(
                    &format!("memory_{}", i),
                    vec![i as f32; 768],
                    Confidence::MEDIUM
                );
                graph_clone.store_memory(memory).unwrap()
            });
            handles.push(handle);
        }
        
        // Collect all IDs
        let ids: Vec<Uuid> = handles.into_iter()
            .map(|h| h.join().unwrap())
            .collect();
        
        // Verify all memories were stored
        assert_eq!(graph.count(), 10);
        
        // Verify each memory can be retrieved
        for id in ids {
            assert!(graph.retrieve(&id).unwrap().is_some());
        }
    }
    
    /// Test graph operations with GraphBackend implementations
    #[test]
    fn test_graph_operations() {
        let graph = UnifiedMemoryGraph::new(
            DashMapBackend::default(),
            GraphConfig::default()
        );
        
        // Store two memories
        let mut embedding1 = [0.0; 768];
        embedding1.iter_mut().for_each(|v| *v = 0.1);
        let memory1 = Memory::new("memory1".to_string(), embedding1, Confidence::HIGH);
        
        let mut embedding2 = [0.0; 768];
        embedding2.iter_mut().for_each(|v| *v = 0.2);
        let memory2 = Memory::new("memory2".to_string(), embedding2, Confidence::HIGH);
        
        let id1 = graph.store_memory(memory1).unwrap();
        let id2 = graph.store_memory(memory2).unwrap();
        
        // Add edge between them
        graph.add_edge(id1, id2, 0.8).unwrap();
        
        // Check neighbors
        let neighbors = graph.get_neighbors(&id1).unwrap();
        assert_eq!(neighbors.len(), 1);
        assert_eq!(neighbors[0].0, id2);
        assert!((neighbors[0].1 - 0.8).abs() < 1e-6);
        
        // Test BFS traversal
        let traversal = graph.traverse_bfs(&id1, 2).unwrap();
        assert!(traversal.contains(&id1));
        assert!(traversal.contains(&id2));
    }
    
    /// Test similarity search across backends
    #[test]
    fn test_similarity_search() {
        let embedding1 = vec![1.0, 0.0, 0.0];
        let embedding2 = vec![0.9, 0.1, 0.0];
        let embedding3 = vec![0.0, 0.0, 1.0];
        
        let memory1 = create_memory_with_embedding("similar1", embedding1.clone(), Confidence::HIGH);
        let memory2 = create_memory_with_embedding("similar2", embedding2.clone(), Confidence::HIGH);
        let memory3 = create_memory_with_embedding("different", embedding3.clone(), Confidence::HIGH);
        
        // Test with HashMap backend
        let graph = UnifiedMemoryGraph::new(
            HashMapBackend::default(),
            GraphConfig::default()
        );
        
        graph.store_memory(memory1.clone()).unwrap();
        graph.store_memory(memory2.clone()).unwrap();
        graph.store_memory(memory3.clone()).unwrap();
        
        // Search for similar memories
        // Need to create a 768-dimensional array for the query
        let mut query_embedding = [0.0; 768];
        for (i, &v) in embedding1.iter().enumerate().take(768) {
            query_embedding[i] = v;
        }
        let results = graph.similarity_search(&query_embedding, 2, Confidence::LOW).unwrap();
        
        // Should find the two similar memories
        assert_eq!(results.len(), 2);
        
        // First result should be the exact match or very similar
        assert!(results[0].1 > 0.8);
    }
    
    /// Test infallible backend graceful degradation
    #[test]
    fn test_infallible_degradation() {
        let backend = InfallibleBackend::new(5); // Small capacity for testing
        let graph = UnifiedMemoryGraph::new(backend, GraphConfig::default());
        
        // Store more memories than capacity
        for i in 0..10 {
            let memory = create_memory_with_embedding(
                &format!("memory_{}", i),
                vec![i as f32; 768],
                Confidence::MEDIUM
            );
            // Should not error even when over capacity
            assert!(graph.store_memory(memory).is_ok());
        }
        
        // Count should be limited by eviction
        assert!(graph.count() <= 5);
    }
    
    /// Test cue-based recall
    #[test]
    fn test_recall_with_cue() {
        let graph = UnifiedMemoryGraph::new(
            HashMapBackend::default(),
            GraphConfig::default()
        );
        
        // Store memories with different embeddings
        for i in 0..5 {
            let mut embedding = vec![0.0; 768];
            embedding[i] = 1.0;
            let id = format!("memory_{}", i);
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
                .content(id.clone())  // Set content to match the ID
                .build();
            graph.store_memory(memory).unwrap();
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
        };
        
        let recalled = graph.recall(&cue).unwrap();
        
        // Should recall at least one memory
        assert!(!recalled.is_empty());
        
        // The best match should be memory_2
        assert!(recalled[0].content.as_ref().map_or(false, |c| c.contains("memory_2")));
    }
    
    /// Test that the new architecture doesn't break existing functionality
    #[test]
    fn test_no_regression() {
        use crate::graph::{create_simple_graph, create_concurrent_graph};
        
        // Test factory functions
        let simple = create_simple_graph();
        let concurrent = create_concurrent_graph();
        
        // Both should work correctly
        let memory = create_memory_with_embedding("test", vec![0.5; 768], Confidence::MEDIUM);
        
        let id1 = simple.store_memory(memory.clone()).unwrap();
        let id2 = concurrent.store_memory(memory).unwrap();
        
        assert!(simple.retrieve(&id1).unwrap().is_some());
        assert!(concurrent.retrieve(&id2).unwrap().is_some());
    }
    
    /// Performance benchmark comparison (not a pass/fail test)
    #[test]
    #[ignore] // Run with --ignored flag for benchmarking
    fn benchmark_backends() {
        use std::time::Instant;
        
        let memory_count = 1000;
        let memories: Vec<Memory> = (0..memory_count)
            .map(|i| create_memory_with_embedding(
                &format!("memory_{}", i),
                vec![i as f32 / memory_count as f32; 768],
                Confidence::MEDIUM
            ))
            .collect();
        
        // Benchmark HashMap backend
        let hashmap_graph = UnifiedMemoryGraph::new(
            HashMapBackend::default(),
            GraphConfig::default()
        );
        let start = Instant::now();
        for memory in &memories {
            hashmap_graph.store_memory(memory.clone()).unwrap();
        }
        let hashmap_time = start.elapsed();
        println!("HashMap backend: {:?} for {} stores", hashmap_time, memory_count);
        
        // Benchmark DashMap backend
        let dashmap_graph = UnifiedMemoryGraph::new(
            DashMapBackend::default(),
            GraphConfig::default()
        );
        let start = Instant::now();
        for memory in &memories {
            dashmap_graph.store_memory(memory.clone()).unwrap();
        }
        let dashmap_time = start.elapsed();
        println!("DashMap backend: {:?} for {} stores", dashmap_time, memory_count);
        
        // Benchmark Infallible backend
        let infallible_graph = UnifiedMemoryGraph::new(
            InfallibleBackend::new(memory_count * 2),
            GraphConfig::default()
        );
        let start = Instant::now();
        for memory in &memories {
            infallible_graph.store_memory(memory.clone()).unwrap();
        }
        let infallible_time = start.elapsed();
        println!("Infallible backend: {:?} for {} stores", infallible_time, memory_count);
        
        // No regression assertion - just informational
        println!("\nPerformance comparison complete. All backends functional.");
    }
}