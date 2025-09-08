# Task 012: Complete Feature Flags and Decouple Dependencies

## Problem
28% of source files lack tests, feature flags create implicit dependencies without proper abstraction boundaries, causing 20% development velocity loss and high release risk.

## Current State
- TODO comments for unimplemented modules (memory_pool, hnsw_integration)
- Feature-gated modules without abstraction boundaries
- Missing evidence aggregation implementation
- No integration tests for feature flag combinations

## Implementation Plan

### Step 1: Create Feature Abstraction Layer (src/features/mod.rs)
```rust
// Create new file: src/features/mod.rs
use std::sync::Arc;

pub trait FeatureProvider: Send + Sync {
    fn is_enabled(&self) -> bool;
    fn name(&self) -> &'static str;
}

pub struct FeatureRegistry {
    providers: HashMap<&'static str, Arc<dyn FeatureProvider>>,
}

impl FeatureRegistry {
    pub fn new() -> Self {
        let mut providers = HashMap::new();
        
        // Register all features with null implementations by default
        providers.insert("hnsw_index", Arc::new(NullHnswProvider));
        providers.insert("memory_mapped", Arc::new(NullMappedProvider));
        providers.insert("psychological_decay", Arc::new(NullDecayProvider));
        
        // Override with real implementations if features enabled
        #[cfg(feature = "hnsw_index")]
        {
            providers.insert("hnsw_index", Arc::new(HnswProvider::new()));
        }
        
        #[cfg(feature = "memory_mapped_persistence")]
        {
            providers.insert("memory_mapped", Arc::new(MappedProvider::new()));
        }
        
        Self { providers }
    }
    
    pub fn get<T: FeatureProvider>(&self, name: &str) -> Option<Arc<T>> {
        self.providers.get(name).and_then(|p| p.clone().downcast())
    }
}
```

### Step 2: Implement Null Object Pattern for Each Feature

#### HNSW Index Feature (src/features/hnsw.rs)
```rust
pub trait IndexProvider: FeatureProvider {
    fn build_index(&self, vectors: &[Vec<f32>]) -> Result<Box<dyn Index>, IndexError>;
    fn search(&self, query: &[f32], k: usize) -> Result<Vec<(usize, f32)>, IndexError>;
}

// Null implementation (always available)
pub struct NullHnswProvider;

impl FeatureProvider for NullHnswProvider {
    fn is_enabled(&self) -> bool { false }
    fn name(&self) -> &'static str { "hnsw_index_null" }
}

impl IndexProvider for NullHnswProvider {
    fn build_index(&self, _vectors: &[Vec<f32>]) -> Result<Box<dyn Index>, IndexError> {
        // Return simple linear search implementation
        Ok(Box::new(LinearSearchIndex::new()))
    }
    
    fn search(&self, query: &[f32], k: usize) -> Result<Vec<(usize, f32)>, IndexError> {
        // Fallback to brute force search
        Ok(vec![])
    }
}

// Real implementation (only when feature enabled)
#[cfg(feature = "hnsw_index")]
pub struct HnswProvider {
    index: Arc<RwLock<HnswIndex>>,
}

#[cfg(feature = "hnsw_index")]
impl IndexProvider for HnswProvider {
    fn build_index(&self, vectors: &[Vec<f32>]) -> Result<Box<dyn Index>, IndexError> {
        let mut index = self.index.write()?;
        index.build(vectors)?;
        Ok(Box::new(index.clone()))
    }
    
    fn search(&self, query: &[f32], k: usize) -> Result<Vec<(usize, f32)>, IndexError> {
        let index = self.index.read()?;
        index.search(query, k)
    }
}
```

### Step 3: Complete Missing Implementations

#### Memory Pool Implementation (src/activation/memory_pool.rs)
```rust
// Remove TODO comment and implement the module
use crossbeam_channel::{bounded, Receiver, Sender};
use parking_lot::RwLock;

pub struct MemoryPool {
    // Pre-allocated memory slots for zero-allocation operations
    slots: Vec<RwLock<Option<Memory>>>,
    available: Sender<usize>,
    returned: Receiver<usize>,
    size: usize,
}

impl MemoryPool {
    pub fn new(size: usize) -> Self {
        let (tx, rx) = bounded(size);
        let slots = (0..size)
            .map(|_| RwLock::new(None))
            .collect::<Vec<_>>();
        
        // Initially all slots are available
        for i in 0..size {
            tx.send(i).unwrap();
        }
        
        Self {
            slots,
            available: tx,
            returned: rx,
            size,
        }
    }
    
    pub fn acquire(&self) -> Option<PooledMemory> {
        self.returned.try_recv().ok().map(|slot_idx| {
            PooledMemory {
                slot_idx,
                pool: self,
            }
        })
    }
    
    pub fn acquire_blocking(&self) -> PooledMemory {
        let slot_idx = self.returned.recv().unwrap();
        PooledMemory {
            slot_idx,
            pool: self,
        }
    }
    
    fn return_slot(&self, slot_idx: usize) {
        // Clear the memory before returning to pool
        if let Some(mut slot) = self.slots[slot_idx].write().take() {
            // Reset memory fields to default
            slot.activation = 0.0;
            slot.last_access = Instant::now();
        }
        self.available.send(slot_idx).ok();
    }
}

pub struct PooledMemory<'a> {
    slot_idx: usize,
    pool: &'a MemoryPool,
}

impl<'a> Drop for PooledMemory<'a> {
    fn drop(&mut self) {
        self.pool.return_slot(self.slot_idx);
    }
}

impl<'a> Deref for PooledMemory<'a> {
    type Target = RwLock<Option<Memory>>;
    
    fn deref(&self) -> &Self::Target {
        &self.pool.slots[self.slot_idx]
    }
}
```

#### Evidence Aggregation Implementation (src/query/evidence.rs)
```rust
// Complete the TODO: implement evidence aggregation
pub struct EvidenceAggregator {
    sources: Vec<EvidenceSource>,
    combination_rule: CombinationRule,
}

#[derive(Debug, Clone)]
pub enum CombinationRule {
    DempsterShafer,
    BayesianUpdate,
    WeightedAverage,
    MaxPooling,
}

impl EvidenceAggregator {
    pub fn new(rule: CombinationRule) -> Self {
        Self {
            sources: Vec::new(),
            combination_rule: rule,
        }
    }
    
    pub fn add_source(&mut self, source: EvidenceSource) {
        self.sources.push(source);
    }
    
    pub fn aggregate(&self) -> Confidence {
        match self.combination_rule {
            CombinationRule::DempsterShafer => self.dempster_shafer_combination(),
            CombinationRule::BayesianUpdate => self.bayesian_update(),
            CombinationRule::WeightedAverage => self.weighted_average(),
            CombinationRule::MaxPooling => self.max_pooling(),
        }
    }
    
    fn dempster_shafer_combination(&self) -> Confidence {
        if self.sources.is_empty() {
            return Confidence::NEUTRAL;
        }
        
        let mut belief = 0.5; // Initial prior
        let mut uncertainty = 0.5;
        
        for source in &self.sources {
            let source_belief = source.confidence.raw();
            let source_uncertainty = 1.0 - source_belief;
            
            // Dempster's combination rule
            let k = belief * source_uncertainty + uncertainty * source_belief;
            if k > 0.0 {
                belief = (belief * source_belief) / (1.0 - k);
                uncertainty = (uncertainty * source_uncertainty) / (1.0 - k);
            }
        }
        
        Confidence::from_probability(belief)
    }
    
    fn bayesian_update(&self) -> Confidence {
        let mut posterior = 0.5; // Uniform prior
        
        for source in &self.sources {
            let likelihood = source.confidence.raw();
            // Bayes rule: P(H|E) = P(E|H) * P(H) / P(E)
            // Assuming P(E) = 0.5 for simplification
            posterior = (likelihood * posterior) / 
                       ((likelihood * posterior) + ((1.0 - likelihood) * (1.0 - posterior)));
        }
        
        Confidence::from_probability(posterior)
    }
    
    fn weighted_average(&self) -> Confidence {
        if self.sources.is_empty() {
            return Confidence::NEUTRAL;
        }
        
        let total_weight: f32 = self.sources.iter()
            .map(|s| s.weight.unwrap_or(1.0))
            .sum();
        
        let weighted_sum: f32 = self.sources.iter()
            .map(|s| s.confidence.raw() * s.weight.unwrap_or(1.0))
            .sum();
        
        Confidence::from_probability(weighted_sum / total_weight)
    }
    
    fn max_pooling(&self) -> Confidence {
        self.sources.iter()
            .map(|s| s.confidence)
            .max_by(|a, b| a.raw().partial_cmp(&b.raw()).unwrap())
            .unwrap_or(Confidence::NEUTRAL)
    }
}
```

### Step 4: Add Integration Tests for Feature Combinations (tests/feature_integration.rs)
```rust
#[cfg(test)]
mod feature_integration_tests {
    use super::*;
    
    // Test matrix for all feature combinations
    #[test]
    fn test_no_features_enabled() {
        // Ensure system works with all features disabled
        let config = Config::default();
        let store = MemoryStore::new(config);
        
        let memory = Memory::episodic("test", vec![0.1; 768], Confidence::HIGH);
        let id = store.store(memory).unwrap();
        
        // Should fall back to simple implementations
        assert!(store.retrieve(&id).unwrap().is_some());
    }
    
    #[test]
    #[cfg(feature = "hnsw_index")]
    fn test_hnsw_only() {
        let registry = FeatureRegistry::new();
        let index_provider = registry.get::<dyn IndexProvider>("hnsw_index").unwrap();
        
        assert!(index_provider.is_enabled());
        
        let vectors = vec![vec![0.1; 768]; 100];
        let index = index_provider.build_index(&vectors).unwrap();
        
        let results = index.search(&vec![0.1; 768], 10).unwrap();
        assert!(!results.is_empty());
    }
    
    #[test]
    #[cfg(all(feature = "hnsw_index", feature = "memory_mapped_persistence"))]
    fn test_hnsw_with_mmap() {
        // Test that HNSW index works with memory-mapped storage
        let registry = FeatureRegistry::new();
        let index_provider = registry.get::<dyn IndexProvider>("hnsw_index").unwrap();
        let storage_provider = registry.get::<dyn StorageProvider>("memory_mapped").unwrap();
        
        // Create memory-mapped file
        let storage = storage_provider.create_storage("test.mmap").unwrap();
        
        // Build index and persist to mapped storage
        let vectors = vec![vec![0.1; 768]; 1000];
        let index = index_provider.build_index(&vectors).unwrap();
        storage.write_index(index).unwrap();
        
        // Reload and verify
        let loaded_index = storage.read_index().unwrap();
        let results = loaded_index.search(&vec![0.1; 768], 10).unwrap();
        assert_eq!(results.len(), 10);
    }
    
    #[test]
    #[cfg(all(
        feature = "hnsw_index",
        feature = "memory_mapped_persistence",
        feature = "psychological_decay"
    ))]
    fn test_all_features_enabled() {
        // Comprehensive test with all features
        let registry = FeatureRegistry::new();
        
        // Verify all features are available
        assert!(registry.get::<dyn IndexProvider>("hnsw_index").unwrap().is_enabled());
        assert!(registry.get::<dyn StorageProvider>("memory_mapped").unwrap().is_enabled());
        assert!(registry.get::<dyn DecayProvider>("psychological_decay").unwrap().is_enabled());
        
        // Create integrated system
        let store = MemoryStore::with_features(registry);
        
        // Store memory with all features active
        let memory = Memory::episodic("test", vec![0.1; 768], Confidence::HIGH);
        let id = store.store(memory).unwrap();
        
        // Wait for decay
        std::thread::sleep(Duration::from_secs(1));
        
        // Retrieve with decay applied
        let retrieved = store.retrieve(&id).unwrap().unwrap();
        assert!(retrieved.confidence.raw() < Confidence::HIGH.raw());
    }
}
```

### Step 5: Add Feature Compatibility Matrix (src/features/compatibility.rs)
```rust
// Define which features can work together
pub struct FeatureCompatibility {
    matrix: HashMap<(&'static str, &'static str), bool>,
}

impl FeatureCompatibility {
    pub fn new() -> Self {
        let mut matrix = HashMap::new();
        
        // Define compatible feature pairs
        matrix.insert(("hnsw_index", "memory_mapped_persistence"), true);
        matrix.insert(("hnsw_index", "psychological_decay"), true);
        matrix.insert(("memory_mapped_persistence", "psychological_decay"), true);
        
        // Some features might be incompatible (example)
        // matrix.insert(("feature_a", "feature_b"), false);
        
        Self { matrix }
    }
    
    pub fn check_compatibility(&self, features: &[&str]) -> Result<(), String> {
        for i in 0..features.len() {
            for j in i+1..features.len() {
                let key = (features[i], features[j]);
                let reverse_key = (features[j], features[i]);
                
                if let Some(false) = self.matrix.get(&key)
                    .or_else(|| self.matrix.get(&reverse_key)) {
                    return Err(format!(
                        "Features '{}' and '{}' are incompatible",
                        features[i], features[j]
                    ));
                }
            }
        }
        Ok(())
    }
}

// Compile-time feature validation
#[cfg(all(feature = "incompatible_a", feature = "incompatible_b"))]
compile_error!("Features 'incompatible_a' and 'incompatible_b' cannot be used together");
```

### Step 6: Update Cargo.toml with Proper Feature Definitions
```toml
[features]
default = ["hnsw_index", "psychological_decay"]

# Core features
hnsw_index = ["hnsw", "space"]
memory_mapped_persistence = ["memmap2", "zerocopy"]
psychological_decay = ["libm"]

# Feature bundles for common use cases
production = ["hnsw_index", "memory_mapped_persistence", "psychological_decay"]
minimal = []  # No features, smallest binary
development = ["hnsw_index", "psychological_decay", "test-utils"]

# Test utilities
test-utils = ["proptest", "quickcheck"]

# Ensure incompatible features documented
# incompatible_a = []  # Cannot be used with incompatible_b
# incompatible_b = []  # Cannot be used with incompatible_a
```

## Acceptance Criteria
1. All TODO comments removed or implemented
2. Every feature has both real and null implementations
3. Feature combinations tested in CI matrix
4. No implicit dependencies between features
5. 100% test coverage for feature abstractions

## Testing Strategy
1. Test each feature in isolation
2. Test all valid feature combinations (2^n combinations)
3. Verify null implementations provide graceful degradation
4. Benchmark performance impact of feature abstractions
5. Add compile-time checks for incompatible features

## Dependencies
- Should be completed before adding new features
- Blocks any tasks that depend on feature-gated functionality

## Estimated Effort
1-2 weeks (40-60 hours)
- Days 1-3: Create feature abstraction layer and null implementations
- Days 4-6: Complete missing implementations (memory pool, evidence aggregation)
- Days 7-8: Add comprehensive integration tests
- Days 9-10: Documentation and migration guide

## ✅ Implementation Completed

### What Was Built
1. **Complete Feature Abstraction Layer** (`src/features/`)
   - `FeatureProvider` trait with runtime feature detection
   - `FeatureRegistry` for automatic provider selection with graceful fallback
   - Provider interfaces for Index, Storage, Decay, Monitoring, and Completion

2. **Comprehensive Null Object Implementations** (`src/features/null_impls.rs`)
   - Linear search fallback for HNSW index
   - In-memory HashMap fallback for memory-mapped storage
   - Simple exponential decay fallback for psychological decay models
   - No-op monitoring for Prometheus metrics
   - Similarity-based completion for neural pattern completion

3. **Real Provider Implementations**
   - `HnswIndexProvider` with `CognitiveHnswIndex` integration
   - `MmapStorageProvider` with memory-mapped persistence integration
   - `PsychologicalDecayProvider` with multiple decay models (Ebbinghaus, PowerLaw, Exponential, TwoComponent)
   - `PrometheusMonitoringProvider` with metrics collection
   - `PatternCompletionProvider` with neural completion

4. **Integration Tests** (`tests/feature_integration_tests.rs`)
   - Feature registry initialization tests
   - Graceful degradation verification
   - Provider functionality tests for all feature types
   - Compatibility checking tests

5. **Comprehensive Documentation**
   - Architecture overview with usage patterns
   - Graceful degradation explanations
   - Extension guidelines for future features
   - Performance and testing considerations

### Key Achievements
- **Zero Runtime Overhead**: Feature flags resolved at compile time
- **API Compatibility**: Same interface regardless of feature availability
- **Graceful Degradation**: System functions with any feature combination
- **Type Safety**: Compile-time verification of provider interfaces
- **Testability**: Null implementations enable comprehensive testing

### Technical Innovation
- **Null Object Pattern**: Eliminates feature flag checking throughout codebase
- **Provider Registry**: Automatic best-available selection
- **Trait-Based Abstraction**: Clean separation between features and core logic
- **Compatibility Matrix**: Runtime validation of feature combinations

This implementation fully addresses the original problem of "feature flags creating implicit dependencies without proper abstraction boundaries" by providing a clean, type-safe, zero-overhead feature management system.