use engram_core::{
    MemoryBuilder, MemoryStore, Confidence, EpisodeBuilder,
    storage::{ContentAddress, ContentIndex, SemanticDeduplicator, MergeStrategy},
};
use std::sync::Arc;
use std::time::Instant;
use chrono::Utc;

#[test]
fn test_content_address_generation_performance() {
    // Test that content address generation meets <10μs target
    let mut embedding = [0.0f32; 768];
    for i in 0..768 {
        embedding[i] = (i as f32 / 768.0).sin(); // Create varied embedding
    }
    
    // Warm up
    for _ in 0..100 {
        let _ = ContentAddress::from_embedding(&embedding);
    }
    
    // Measure
    let iterations = 10000;
    let start = Instant::now();
    for _ in 0..iterations {
        let _ = ContentAddress::from_embedding(&embedding);
    }
    let elapsed = start.elapsed();
    
    let per_op = elapsed / iterations;
    println!("Content address generation: {:?} per operation", per_op);
    assert!(per_op.as_micros() < 10, "Content address generation should be <10μs, got {:?}", per_op);
}

#[test]
fn test_deduplication_check_performance() {
    // Test that deduplication check meets <100μs target
    let mut dedup = SemanticDeduplicator::new(0.95, MergeStrategy::KeepHighestConfidence);
    
    // Create test memories
    let mut memories = Vec::new();
    for i in 0..100 {
        let mut embedding = [0.0f32; 768];
        embedding[0] = i as f32 / 100.0;
        
        let memory = MemoryBuilder::new()
            .id(format!("mem_{}", i))
            .embedding(embedding)
            .confidence(Confidence::exact(0.8))
            .content(format!("Test memory {}", i))
            .build();
        memories.push(Arc::new(memory));
    }
    
    // Create new memory to check
    let mut new_embedding = [0.5f32; 768];
    new_embedding[0] = 0.45; // Similar but not identical
    let new_memory = MemoryBuilder::new()
        .id("new_memory".to_string())
        .embedding(new_embedding)
        .confidence(Confidence::exact(0.9))
        .content("New test memory".to_string())
        .build();
    
    // Warm up
    for _ in 0..100 {
        let _ = dedup.check_duplicate(&new_memory, &memories);
    }
    
    // Measure
    let iterations = 1000;
    let start = Instant::now();
    for _ in 0..iterations {
        let _ = dedup.check_duplicate(&new_memory, &memories);
    }
    let elapsed = start.elapsed();
    
    let per_op = elapsed / iterations;
    println!("Deduplication check: {:?} per operation", per_op);
    assert!(per_op.as_micros() < 100, "Deduplication check should be <100μs, got {:?}", per_op);
}

#[test]
fn test_lsh_bucket_lookup_performance() {
    // Test that LSH bucket lookup meets <50μs target
    let index = ContentIndex::new();
    
    // Populate index with test data
    for i in 0..1000 {
        let mut embedding = [0.0f32; 768];
        embedding[0] = (i as f32 / 1000.0).sin();
        embedding[1] = (i as f32 / 1000.0).cos();
        
        let address = ContentAddress::from_embedding(&embedding);
        index.insert(address, format!("memory_{}", i));
    }
    
    // Create lookup address
    let mut lookup_embedding = [0.0f32; 768];
    lookup_embedding[0] = 0.5;
    let lookup_address = ContentAddress::from_embedding(&lookup_embedding);
    
    // Warm up
    for _ in 0..100 {
        let _ = index.find_similar(&lookup_address);
    }
    
    // Measure
    let iterations = 10000;
    let start = Instant::now();
    for _ in 0..iterations {
        let _ = index.find_similar(&lookup_address);
    }
    let elapsed = start.elapsed();
    
    let per_op = elapsed / iterations;
    println!("LSH bucket lookup: {:?} per operation", per_op);
    assert!(per_op.as_micros() < 50, "LSH bucket lookup should be <50μs, got {:?}", per_op);
}

#[test]
fn test_stable_content_hashing() {
    // Test that same embedding always produces same hash
    let mut embedding = [0.0f32; 768];
    for i in 0..768 {
        embedding[i] = (i as f32 * 0.01).sin();
    }
    
    let addr1 = ContentAddress::from_embedding(&embedding);
    let addr2 = ContentAddress::from_embedding(&embedding);
    
    assert_eq!(addr1, addr2, "Same embedding should produce identical content addresses");
}

#[test]
fn test_lsh_similarity_grouping() {
    // Test that vectors >0.9 similarity end up in same LSH bucket frequently
    let index = ContentIndex::new();
    let mut same_bucket_count = 0;
    let trials = 100;
    
    for trial in 0..trials {
        // Create base embedding
        let mut base_embedding = [0.0f32; 768];
        for i in 0..768 {
            base_embedding[i] = ((trial + i) as f32 / 768.0).sin();
        }
        
        // Create similar embedding (>0.9 cosine similarity)
        let mut similar_embedding = base_embedding.clone();
        for i in 0..768 {
            // Add small noise to maintain high similarity
            similar_embedding[i] += (i as f32 * 0.001).sin() * 0.05;
        }
        
        // Normalize to maintain unit length
        let norm: f32 = similar_embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
        for val in &mut similar_embedding {
            *val /= norm;
        }
        
        let addr1 = ContentAddress::from_embedding(&base_embedding);
        let addr2 = ContentAddress::from_embedding(&similar_embedding);
        
        if addr1.same_bucket(&addr2) {
            same_bucket_count += 1;
        }
    }
    
    let same_bucket_rate = same_bucket_count as f32 / trials as f32;
    println!("Same LSH bucket rate for similar vectors: {:.2}%", same_bucket_rate * 100.0);
    
    // Expect at least 60% of similar vectors to end up in same bucket
    assert!(same_bucket_rate > 0.6, 
            "Similar vectors should frequently share LSH buckets, got {:.2}%", 
            same_bucket_rate * 100.0);
}

#[test]
fn test_exact_duplicate_detection() {
    // Test zero false negatives for exact matches
    let index = ContentIndex::new();
    
    // Insert 1000 unique memories
    for i in 0..1000 {
        let mut embedding = [0.0f32; 768];
        embedding[0] = i as f32 / 1000.0;
        embedding[1] = (i as f32).sin();
        
        let address = ContentAddress::from_embedding(&embedding);
        assert!(index.insert(address.clone(), format!("memory_{}", i)),
                "Should insert unique address");
        
        // Try to insert duplicate - should fail
        assert!(!index.insert(address.clone(), format!("duplicate_{}", i)),
                 "Should reject duplicate address");
        
        // Verify we can retrieve it
        assert_eq!(index.get(&address), Some(format!("memory_{}", i)),
                   "Should retrieve correct memory ID");
    }
}

#[test]
fn test_memory_overhead() {
    // Test that memory overhead is <10%
    let base_memory_size = std::mem::size_of::<[f32; 768]>() + 
                           std::mem::size_of::<String>() * 2 + // ID + content
                           std::mem::size_of::<f32>() * 2 + // confidence + activation
                           std::mem::size_of::<chrono::DateTime<chrono::Utc>>(); // timestamp
    
    let content_address_size = std::mem::size_of::<ContentAddress>();
    let index_entry_size = content_address_size + std::mem::size_of::<String>(); // address + memory_id
    
    let overhead_ratio = index_entry_size as f32 / base_memory_size as f32;
    println!("Memory overhead: {:.2}%", overhead_ratio * 100.0);
    
    assert!(overhead_ratio < 0.10, 
            "Memory overhead should be <10%, got {:.2}%", 
            overhead_ratio * 100.0);
}

#[test]
fn test_content_based_retrieval() {
    // Test end-to-end content-based retrieval
    let store = MemoryStore::new(10000);
    
    // Store some memories
    let mut memories = Vec::new();
    for i in 0..10 {
        let mut embedding = [0.0f32; 768];
        embedding[0] = (i as f32 / 10.0).sin();
        embedding[1] = (i as f32 / 10.0).cos();
        
        let memory = MemoryBuilder::new()
            .id(format!("mem_{}", i))
            .embedding(embedding)
            .confidence(Confidence::exact(0.8 + i as f32 * 0.01))
            .content(format!("Test memory {}", i))
            .build();
        
        // Convert Memory to Episode for store
        let episode = EpisodeBuilder::new()
            .id(memory.id.clone())
            .when(Utc::now())
            .what(memory.content.clone().unwrap_or_default())
            .embedding(embedding)
            .confidence(memory.confidence)
            .build();
        
        store.store(episode.clone());
        memories.push(memory);
    }
    
    // Test recall by content
    let target_embedding = memories[5].embedding.clone();
    let results = store.recall_by_content(&target_embedding);
    
    if let Some(result_episode) = results {
        assert_eq!(result_episode.id, "mem_5", "Should find exact match");
    } else {
        panic!("Should find memory by content");
    }
    
    // Test find similar content
    let mut similar_embedding = target_embedding.clone();
    similar_embedding[0] += 0.01; // Slight modification
    
    let similar_results = store.find_similar_content(&similar_embedding, 5);
    assert!(similar_results.len() <= 5, "Should respect limit");
    assert!(similar_results.iter().any(|(ep, _)| ep.id == "mem_5"),
            "Should find similar memory");
}

#[test]
fn test_deduplication_strategies() {
    let store = MemoryStore::new(10000);
    
    // Test KeepHighestConfidence strategy (default)
    let mut embedding1 = [0.5f32; 768];
    let memory1 = MemoryBuilder::new()
        .id("mem_low_conf".to_string())
        .embedding(embedding1)
        .confidence(Confidence::exact(0.6))
        .content("Low confidence memory".to_string())
        .build();
    
    let episode1 = EpisodeBuilder::new()
        .id(memory1.id.clone())
        .when(Utc::now())
        .what(memory1.content.clone().unwrap_or_default())
        .embedding(embedding1)
        .confidence(memory1.confidence)
        .build();
    store.store(episode1);
    
    // Store similar memory with higher confidence
    embedding1[0] = 0.501; // Very similar
    let memory2 = MemoryBuilder::new()
        .id("mem_high_conf".to_string())
        .embedding(embedding1)
        .confidence(Confidence::exact(0.9))
        .content("High confidence memory".to_string())
        .build();
    
    let episode2 = EpisodeBuilder::new()
        .id(memory2.id.clone())
        .when(Utc::now())
        .what(memory2.content.clone().unwrap_or_default())
        .embedding(embedding1)
        .confidence(memory2.confidence)
        .build();
    store.store(episode2);
    
    // Check deduplication stats
    let stats = store.deduplication_stats();
    assert!(stats.duplicates_found.load(std::sync::atomic::Ordering::Relaxed) > 0,
            "Should have found duplicates");
}

#[test]
fn test_hamming_distance_calculation() {
    // Test Hamming distance between LSH buckets
    let mut embedding1 = [0.0f32; 768];
    let mut embedding2 = [0.0f32; 768];
    
    // Create orthogonal embeddings
    embedding1[0] = 1.0;
    embedding2[1] = 1.0;
    
    let addr1 = ContentAddress::from_embedding(&embedding1);
    let addr2 = ContentAddress::from_embedding(&embedding2);
    
    let distance = addr1.hamming_distance(&addr2);
    println!("Hamming distance between orthogonal vectors: {}", distance);
    
    // Orthogonal vectors should have significant Hamming distance
    assert!(distance > 8, "Orthogonal vectors should have large Hamming distance");
    
    // Same vector should have zero distance
    assert_eq!(addr1.hamming_distance(&addr1), 0, "Same address should have zero distance");
}

#[test]
fn test_content_index_nearby_search() {
    let index = ContentIndex::new();
    
    // Create cluster of similar memories
    let base_embedding = [0.5f32; 768];
    for i in 0..100 {
        let mut embedding = base_embedding.clone();
        embedding[i % 768] += (i as f32 * 0.001).sin();
        
        let address = ContentAddress::from_embedding(&embedding);
        index.insert(address, format!("memory_{}", i));
    }
    
    // Search for nearby content
    let search_address = ContentAddress::from_embedding(&base_embedding);
    let nearby = index.find_nearby(&search_address, 5); // Within Hamming distance 5
    
    assert!(!nearby.is_empty(), "Should find nearby content addresses");
    println!("Found {} nearby addresses within Hamming distance 5", nearby.len());
}