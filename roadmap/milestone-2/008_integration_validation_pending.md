# Task 008: Integration and End-to-End Validation

## Status: Pending
## Priority: P0 - Critical Path
## Estimated Effort: 1 day
## Dependencies: Tasks 001-007 (all milestone-2 tasks)

## Objective
Integrate all vector-native storage components and validate the complete system meets performance targets (90% recall@10, <1ms latency) with proper confidence propagation.

## Current State Analysis
- **To Complete**: Three-tier storage architecture
- **To Complete**: Content-addressable retrieval
- **To Complete**: HNSW self-tuning
- **To Complete**: Migration policies
- **To Validate**: End-to-end performance
- **To Validate**: Confidence propagation correctness

## Technical Specification

### 1. Integration Test Framework

```rust
// engram-core/tests/vector_storage_integration.rs

use engram_core::*;
use proptest::prelude::*;

pub struct IntegrationTestHarness {
    /// Complete storage system
    storage: TieredStorage,
    
    /// Memory store with vector integration
    memory_store: MemoryStore,
    
    /// Test data generator
    data_generator: TestDataGenerator,
    
    /// Performance validator
    validator: PerformanceValidator,
}

impl IntegrationTestHarness {
    /// Run complete integration test suite
    pub async fn run_full_validation(&mut self) -> ValidationResults {
        let mut results = ValidationResults::default();
        
        // Phase 1: Storage tier functionality
        results.storage_tier = self.validate_storage_tiers().await?;
        
        // Phase 2: Content addressing
        results.content_addressing = self.validate_content_addressing().await?;
        
        // Phase 3: HNSW self-tuning
        results.hnsw_tuning = self.validate_hnsw_adaptation().await?;
        
        // Phase 4: Migration policies
        results.migration = self.validate_migrations().await?;
        
        // Phase 5: Performance targets
        results.performance = self.validate_performance_targets().await?;
        
        // Phase 6: Confidence propagation
        results.confidence = self.validate_confidence_propagation().await?;
        
        results
    }
}
```

### 2. Storage Tier Validation

```rust
// engram-core/tests/storage_tier_validation.rs

impl IntegrationTestHarness {
    async fn validate_storage_tiers(&mut self) -> Result<TierValidation> {
        let mut validation = TierValidation::default();
        
        // Test 1: Hot tier performance
        let hot_vectors = self.data_generator.generate_hot_vectors(10_000);
        let start = Instant::now();
        
        for (id, vector) in hot_vectors.iter() {
            self.storage.hot.store(id, vector, Default::default())?;
        }
        
        validation.hot_write_time = start.elapsed();
        
        // Verify retrieval latency
        let mut retrieval_times = Vec::new();
        for (id, _) in hot_vectors.iter().take(1000) {
            let start = Instant::now();
            let _ = self.storage.hot.retrieve(id)?;
            retrieval_times.push(start.elapsed());
        }
        
        validation.hot_p95_latency = percentile(&retrieval_times, 95);
        assert!(validation.hot_p95_latency < Duration::from_nanos(100));
        
        // Test 2: Warm tier compression
        let warm_vectors = self.data_generator.generate_vectors(50_000);
        let uncompressed_size = warm_vectors.len() * 768 * 4;
        
        for (id, vector) in warm_vectors.iter() {
            self.storage.warm.store(id, vector, Default::default())?;
        }
        
        let compressed_size = self.storage.warm.memory_usage();
        validation.warm_compression_ratio = uncompressed_size as f32 / compressed_size as f32;
        assert!(validation.warm_compression_ratio > 1.5);
        
        // Test 3: Cold tier SIMD operations
        let cold_vectors = self.data_generator.generate_vectors(100_000);
        
        for batch in cold_vectors.chunks(1000) {
            self.storage.cold.append_batch(batch)?;
        }
        
        let query = [0.5f32; 768];
        let start = Instant::now();
        let _ = unsafe { 
            self.storage.cold.similarity_search_simd(&query, 100) 
        };
        validation.cold_search_time = start.elapsed();
        assert!(validation.cold_search_time < Duration::from_millis(1));
        
        Ok(validation)
    }
}
```

### 3. Content Addressing Validation

```rust
impl IntegrationTestHarness {
    async fn validate_content_addressing(&mut self) -> Result<ContentValidation> {
        let mut validation = ContentValidation::default();
        
        // Test deduplication
        let vector = [0.7f32; 768];
        let mut similar = vector;
        similar[0] = 0.701; // Very similar
        
        let addr1 = self.storage.content_store.store_by_content(vector, Default::default())?;
        let addr2 = self.storage.content_store.store_by_content(similar, Default::default())?;
        
        // Should deduplicate
        validation.deduplication_works = addr1 == addr2;
        assert!(validation.deduplication_works);
        
        // Test content retrieval
        let retrieved = self.storage.content_store.retrieve_by_content(&addr1)?;
        assert!(retrieved.is_some());
        
        // Test LSH bucketing
        let similar_vectors: Vec<_> = (0..100)
            .map(|i| {
                let mut v = vector;
                v[i] = v[i] + 0.01;
                v
            })
            .collect();
            
        let mut same_bucket_count = 0;
        for v in &similar_vectors {
            let addr = ContentAddress::from_embedding(v);
            if addr.lsh_bucket == addr1.lsh_bucket {
                same_bucket_count += 1;
            }
        }
        
        validation.lsh_effectiveness = same_bucket_count as f32 / 100.0;
        assert!(validation.lsh_effectiveness > 0.8); // Most similar vectors in same bucket
        
        Ok(validation)
    }
}
```

### 4. HNSW Adaptation Validation

```rust
impl IntegrationTestHarness {
    async fn validate_hnsw_adaptation(&mut self) -> Result<HnswValidation> {
        let mut validation = HnswValidation::default();
        
        // Build index with known data
        let vectors = self.data_generator.generate_vectors(10_000);
        let queries = self.data_generator.generate_queries(100);
        
        // Enable adaptive tuning
        self.storage.hnsw.enable_adaptive_tuning(0.9);
        
        // Measure initial recall
        let mut initial_recalls = Vec::new();
        for query in queries.iter().take(10) {
            let results = self.storage.hnsw.search(query, 10);
            let recall = self.calculate_recall(&results, query);
            initial_recalls.push(recall);
        }
        
        validation.initial_recall = initial_recalls.iter().sum::<f32>() / 10.0;
        
        // Run adaptive searches
        for query in queries.iter() {
            let _ = self.storage.hnsw.adaptive_search(query, 10, 0.9);
        }
        
        // Measure improved recall
        let mut final_recalls = Vec::new();
        for query in queries.iter().take(10) {
            let results = self.storage.hnsw.search(query, 10);
            let recall = self.calculate_recall(&results, query);
            final_recalls.push(recall);
        }
        
        validation.final_recall = final_recalls.iter().sum::<f32>() / 10.0;
        validation.improvement = validation.final_recall / validation.initial_recall;
        
        assert!(validation.final_recall >= 0.9);
        assert!(validation.improvement >= 1.1); // At least 10% improvement
        
        Ok(validation)
    }
}
```

### 5. Migration Policy Validation

```rust
impl IntegrationTestHarness {
    async fn validate_migrations(&mut self) -> Result<MigrationValidation> {
        let mut validation = MigrationValidation::default();
        
        // Setup initial data distribution
        let hot_vectors = self.setup_hot_tier_data(1000);
        let warm_vectors = self.setup_warm_tier_data(5000);
        let cold_vectors = self.setup_cold_tier_data(10000);
        
        // Simulate access patterns
        self.simulate_access_patterns().await;
        
        // Run migration cycle
        let start = Instant::now();
        let results = self.storage.run_migration_cycle().await?;
        validation.migration_time = start.elapsed();
        
        // Verify correct migrations
        validation.successful_migrations = results.successful;
        validation.failed_migrations = results.failed;
        
        assert_eq!(validation.failed_migrations, 0);
        
        // Test memory pressure response
        self.simulate_memory_pressure();
        let pressure_results = self.storage.run_migration_cycle().await?;
        
        validation.pressure_response_time = pressure_results.duration;
        assert!(validation.pressure_response_time < Duration::from_millis(100));
        
        Ok(validation)
    }
}
```

### 6. Performance Target Validation

```rust
impl IntegrationTestHarness {
    async fn validate_performance_targets(&mut self) -> Result<PerformanceValidation> {
        let mut validation = PerformanceValidation::default();
        
        // Load standard dataset
        let dataset = DatasetLoader::load_sift1m()?;
        
        // Build complete system
        for (idx, vector) in dataset.vectors.iter().enumerate() {
            let episode = Episode {
                id: format!("vec_{}", idx),
                embedding: *vector,
                when: Utc::now(),
                encoding_confidence: Confidence::HIGH,
                // ... other fields
            };
            self.memory_store.store(episode)?;
        }
        
        // Measure recall@10
        let mut recalls = Vec::new();
        let mut latencies = Vec::new();
        
        for (query_idx, query) in dataset.queries.iter().enumerate() {
            let start = Instant::now();
            
            let cue = Cue::from_embedding(query.to_vec(), 0.0);
            let results = self.memory_store.recall(cue);
            
            let latency = start.elapsed();
            latencies.push(latency);
            
            let result_ids: Vec<_> = results.iter()
                .take(10)
                .map(|(ep, _)| ep.id.clone())
                .collect();
                
            let recall = self.calculate_recall_at_k(
                &result_ids,
                &dataset.ground_truth[query_idx],
                10,
            );
            recalls.push(recall);
        }
        
        validation.avg_recall_at_10 = recalls.iter().sum::<f32>() / recalls.len() as f32;
        validation.p95_latency = percentile(&latencies, 95);
        validation.p99_latency = percentile(&latencies, 99);
        
        // Verify targets
        assert!(validation.avg_recall_at_10 >= 0.9, 
                "Recall@10 {} below target 0.9", validation.avg_recall_at_10);
        assert!(validation.p95_latency < Duration::from_millis(1),
                "P95 latency {:?} above 1ms", validation.p95_latency);
        
        Ok(validation)
    }
}
```

### 7. Confidence Propagation Validation

```rust
impl IntegrationTestHarness {
    async fn validate_confidence_propagation(&mut self) -> Result<ConfidenceValidation> {
        let mut validation = ConfidenceValidation::default();
        
        // Test confidence through storage tiers
        let vector = [0.5f32; 768];
        let id = "test_confidence";
        
        // Store in hot tier
        self.storage.hot.store(id, &vector, Default::default())?;
        let (_, hot_conf) = self.storage.retrieve_with_confidence(id)?;
        
        // Migrate to warm
        self.storage.migrate_to_tier(id, StorageTier::Warm).await?;
        let (_, warm_conf) = self.storage.retrieve_with_confidence(id)?;
        
        // Migrate to cold
        self.storage.migrate_to_tier(id, StorageTier::Cold).await?;
        let (_, cold_conf) = self.storage.retrieve_with_confidence(id)?;
        
        // Verify confidence decreases with tier
        assert!(hot_conf.combined.point > warm_conf.combined.point);
        assert!(warm_conf.combined.point > cold_conf.combined.point);
        
        // Test calibration
        let mut predictions = Vec::new();
        for _ in 0..1000 {
            let query = self.data_generator.generate_query();
            let results = self.memory_store.recall(Cue::from_embedding(query, 0.0));
            
            for (episode, confidence) in results.iter().take(10) {
                let is_relevant = self.is_relevant(&episode, &query);
                predictions.push((confidence.raw(), is_relevant));
            }
        }
        
        validation.ece = calculate_expected_calibration_error(&predictions);
        assert!(validation.ece < 0.1);
        
        Ok(validation)
    }
}
```

## Integration Points

This task integrates all components from Tasks 001-007:
- Three-tier storage (Task 001)
- Content-addressable retrieval (Task 002)  
- HNSW self-tuning (Task 003)
- Columnar storage (Task 004)
- FAISS/Annoy benchmarks (Task 005)
- Migration policies (Task 006)
- Confidence propagation (Task 007)

## Testing Strategy

### End-to-End Tests
```rust
#[tokio::test]
async fn test_complete_integration() {
    let mut harness = IntegrationTestHarness::new();
    let results = harness.run_full_validation().await.unwrap();
    
    assert!(results.all_passed());
    println!("Integration validation results: {:?}", results);
}

#[test]
fn test_concurrent_operations() {
    let storage = Arc::new(TieredStorage::new());
    
    let handles: Vec<_> = (0..10).map(|i| {
        let storage = storage.clone();
        thread::spawn(move || {
            for j in 0..1000 {
                let vector = [i as f32 * 0.1; 768];
                storage.store(&format!("{}_{}", i, j), vector).unwrap();
            }
        })
    }).collect();
    
    for h in handles { h.join().unwrap(); }
    
    assert_eq!(storage.total_vectors(), 10_000);
}
```

## Acceptance Criteria
- [ ] All storage tiers function correctly
- [ ] Content deduplication works as expected
- [ ] HNSW achieves 90% recall@10 after tuning
- [ ] Migrations complete without data loss
- [ ] System achieves <1ms P95 query latency
- [ ] Confidence propagation is mathematically correct
- [ ] Integration tests pass on CI/CD

## Performance Targets
- Full system recall@10: ≥90%
- Query P95 latency: <1ms
- Query P99 latency: <5ms
- Storage overhead: <20% vs raw vectors
- Migration cycle: <5s for 10K vectors
- System supports 1M+ vectors

## Risk Mitigation
- Gradual rollout with feature flags
- Extensive integration testing
- Performance regression detection
- Rollback plan for each component
- Monitoring and alerting setup