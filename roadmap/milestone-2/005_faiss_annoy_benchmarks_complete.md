# Task 005: FAISS and Annoy Benchmark Framework

## Status: Partial ✅
## Priority: P1 - Validation Critical
## Estimated Effort: 1 day
## Dependencies: Tasks 001-004 (complete storage system)

## Objective
Create comprehensive benchmark framework comparing Engram's vector storage against FAISS and Annoy on standard ANN datasets, validating 90% recall@10 with <1ms query time.

## Current Implementation Status
- ✅ Benchmark harness with pluggable `AnnIndex` trait (`engram-core/benches/ann_comparison.rs:1-197`).
- ✅ Synthetic and mock dataset loaders available (`engram-core/benches/datasets.rs:1-178`).
- ✅ Placeholder FAISS/Annoy implementations provide API compatibility for framework smoke tests (`engram-core/benches/mock_faiss.rs:1-126`, `engram-core/benches/mock_annoy.rs:1-133`).
- ⚠️ Real FAISS/Annoy bindings not integrated; mocks do exact search with perturbation and cannot validate true performance.
- ⚠️ No benchmark outputs captured (CSV/JSON) or compared against Engram’s native index; the CLI/CI still lacks instructions for running the suite.
- ⚠️ Recall target verification (<1 ms, 90% recall@10) not enforced; current harness only prints metrics.

## Remaining Work for Completion
1. Replace mock implementations with actual FAISS and Annoy bindings (or wrap official C++ libraries) and wire build instructions into repository docs.
2. Automate benchmark run (e.g., criterion group or standalone binary) that exports CSV summaries for Engram vs FAISS/Annoy.
3. Add assertion layer comparing recall/latency to spec thresholds and fail CI when regressions occur.

## Current State Analysis
- **Existing**: Basic benchmarking from milestone-1/task-009
- **Existing**: HNSW index implementation
- **Missing**: FAISS integration and comparison
- **Missing**: Annoy integration and comparison
- **Missing**: Standard ANN dataset loading

## Technical Specification

### 1. Benchmark Framework Architecture

```rust
// engram-core/benches/ann_comparison.rs

use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};

pub trait AnnIndex: Send + Sync {
    /// Build index from vectors
    fn build(&mut self, vectors: &[[f32; 768]]) -> Result<()>;
    
    /// Search for k nearest neighbors
    fn search(&self, query: &[f32; 768], k: usize) -> Vec<(usize, f32)>;
    
    /// Get index size in bytes
    fn memory_usage(&self) -> usize;
    
    /// Name for reporting
    fn name(&self) -> &str;
}

pub struct BenchmarkFramework {
    /// Implementations to compare
    implementations: Vec<Box<dyn AnnIndex>>,
    
    /// Test datasets
    datasets: Vec<AnnDataset>,
    
    /// Ground truth for recall calculation
    ground_truth: HashMap<String, Vec<Vec<(usize, f32)>>>,
    
    /// Results collector
    results: BenchmarkResults,
}

#[derive(Debug, Clone)]
pub struct AnnDataset {
    name: String,
    vectors: Vec<[f32; 768]>,
    queries: Vec<[f32; 768]>,
    ground_truth: Vec<Vec<usize>>, // True k-NN for each query
}

impl BenchmarkFramework {
    pub fn run_comparison(&mut self) -> BenchmarkResults {
        for dataset in &self.datasets {
            for implementation in &mut self.implementations {
                self.benchmark_implementation(implementation.as_mut(), dataset);
            }
        }
        
        self.results.clone()
    }
    
    fn benchmark_implementation(
        &mut self,
        index: &mut dyn AnnIndex,
        dataset: &AnnDataset,
    ) {
        // Build index
        let build_start = Instant::now();
        index.build(&dataset.vectors).unwrap();
        let build_time = build_start.elapsed();
        
        // Measure recall and latency
        let mut recalls = Vec::new();
        let mut latencies = Vec::new();
        
        for (query_idx, query) in dataset.queries.iter().enumerate() {
            let search_start = Instant::now();
            let results = index.search(query, 10);
            let latency = search_start.elapsed();
            
            let recall = self.calculate_recall(
                &results,
                &dataset.ground_truth[query_idx],
                10,
            );
            
            recalls.push(recall);
            latencies.push(latency);
        }
        
        // Record results
        self.results.record(
            index.name(),
            dataset.name.clone(),
            BenchmarkMetrics {
                build_time,
                avg_recall: recalls.iter().sum::<f32>() / recalls.len() as f32,
                avg_latency: latencies.iter().sum::<Duration>() / latencies.len() as u32,
                p95_latency: percentile(&latencies, 95),
                p99_latency: percentile(&latencies, 99),
                memory_usage: index.memory_usage(),
            },
        );
    }
}
```

### 2. Engram Implementation Wrapper

```rust
// engram-core/benches/engram_ann.rs

pub struct EngramAnnIndex {
    store: MemoryStore,
    storage: TieredStorage,
    hnsw: AdaptiveHnswIndex,
}

impl AnnIndex for EngramAnnIndex {
    fn build(&mut self, vectors: &[[f32; 768]]) -> Result<()> {
        // Configure for benchmark
        self.hnsw.set_parameters(HnswParams {
            m: 16,
            ef_construction: 200,
            ef_search: 100,
            max_m: 32,
        });
        
        // Build index
        for (idx, vector) in vectors.iter().enumerate() {
            let episode = Episode {
                id: format!("vec_{}", idx),
                embedding: *vector,
                when: Utc::now(),
                what: String::new(),
                encoding_confidence: Confidence::HIGH,
                // ... other fields
            };
            
            self.store.store(episode)?;
        }
        
        Ok(())
    }
    
    fn search(&self, query: &[f32; 768], k: usize) -> Vec<(usize, f32)> {
        let cue = Cue::from_embedding(query.to_vec(), 0.0);
        let results = self.store.recall(cue);
        
        results.into_iter()
            .take(k)
            .map(|(episode, conf)| {
                let idx = episode.id[4..].parse().unwrap(); // Extract index
                (idx, conf.raw())
            })
            .collect()
    }
    
    fn memory_usage(&self) -> usize {
        self.storage.memory_usage() + self.hnsw.memory_usage()
    }
    
    fn name(&self) -> &str {
        "Engram"
    }
}
```

### 3. FAISS Integration

```rust
// engram-core/benches/faiss_ann.rs

use faiss::{Index, IndexFlatL2, IndexIVFFlat, index_factory};

pub struct FaissAnnIndex {
    index: Box<dyn Index>,
    dimension: usize,
}

impl FaissAnnIndex {
    pub fn new_ivf_flat(dimension: usize, nlist: usize) -> Result<Self> {
        let description = format!("IVF{},Flat", nlist);
        let index = index_factory(dimension, &description, faiss::MetricType::L2)?;
        
        Ok(Self {
            index,
            dimension,
        })
    }
    
    pub fn new_hnsw(dimension: usize, m: usize) -> Result<Self> {
        let description = format!("HNSW{}", m);
        let index = index_factory(dimension, &description, faiss::MetricType::L2)?;
        
        Ok(Self {
            index,
            dimension,
        })
    }
}

impl AnnIndex for FaissAnnIndex {
    fn build(&mut self, vectors: &[[f32; 768]]) -> Result<()> {
        // Flatten vectors for FAISS
        let flat_vectors: Vec<f32> = vectors.iter()
            .flat_map(|v| v.iter().copied())
            .collect();
            
        // Train if needed (for IVF)
        if self.index.is_trained() == false {
            self.index.train(&flat_vectors)?;
        }
        
        // Add vectors
        self.index.add(&flat_vectors)?;
        
        Ok(())
    }
    
    fn search(&self, query: &[f32; 768], k: usize) -> Vec<(usize, f32)> {
        let mut distances = vec![0.0f32; k];
        let mut indices = vec![0i64; k];
        
        self.index.search(query, k, &mut distances, &mut indices).unwrap();
        
        indices.iter()
            .zip(distances.iter())
            .map(|(&idx, &dist)| (idx as usize, 1.0 / (1.0 + dist))) // Convert to similarity
            .collect()
    }
    
    fn memory_usage(&self) -> usize {
        // Estimate based on index type and size
        self.index.ntotal() * self.dimension * 4 + 1024 * 1024 // Overhead
    }
    
    fn name(&self) -> &str {
        "FAISS"
    }
}
```

### 4. Annoy Integration

```rust
// engram-core/benches/annoy_ann.rs

use annoy::{AnnoyIndex, IndexType};

pub struct AnnoyAnnIndex {
    index: AnnoyIndex,
    dimension: usize,
    n_trees: usize,
}

impl AnnoyAnnIndex {
    pub fn new(dimension: usize, n_trees: usize) -> Result<Self> {
        let index = AnnoyIndex::new(dimension, IndexType::Angular)?;
        
        Ok(Self {
            index,
            dimension,
            n_trees,
        })
    }
}

impl AnnIndex for AnnoyAnnIndex {
    fn build(&mut self, vectors: &[[f32; 768]]) -> Result<()> {
        for (idx, vector) in vectors.iter().enumerate() {
            self.index.add_item(idx, vector)?;
        }
        
        self.index.build(self.n_trees)?;
        
        Ok(())
    }
    
    fn search(&self, query: &[f32; 768], k: usize) -> Vec<(usize, f32)> {
        let (indices, distances) = self.index.get_nns_by_vector(query, k, -1);
        
        indices.into_iter()
            .zip(distances.into_iter())
            .map(|(idx, dist)| (idx, 1.0 - dist)) // Angular distance to similarity
            .collect()
    }
    
    fn memory_usage(&self) -> usize {
        self.index.memory_usage()
    }
    
    fn name(&self) -> &str {
        "Annoy"
    }
}
```

### 5. Standard Dataset Loaders

```rust
// engram-core/benches/datasets.rs

pub struct DatasetLoader;

impl DatasetLoader {
    /// Load SIFT1M dataset
    pub fn load_sift1m() -> Result<AnnDataset> {
        let base_vectors = Self::read_fvecs("sift/sift_base.fvecs")?;
        let query_vectors = Self::read_fvecs("sift/sift_query.fvecs")?;
        let ground_truth = Self::read_ivecs("sift/sift_groundtruth.ivecs")?;
        
        // Convert to 768-dim by padding
        let base_768 = Self::pad_to_768(base_vectors);
        let query_768 = Self::pad_to_768(query_vectors);
        
        Ok(AnnDataset {
            name: "SIFT1M".to_string(),
            vectors: base_768,
            queries: query_768,
            ground_truth,
        })
    }
    
    /// Load GloVe dataset
    pub fn load_glove() -> Result<AnnDataset> {
        // Load pre-computed 768-dim GloVe embeddings
        let vectors = Self::read_glove_embeddings("glove/glove.840B.300d.txt")?;
        
        // Generate queries and ground truth
        let queries = Self::sample_queries(&vectors, 1000);
        let ground_truth = Self::compute_ground_truth(&vectors, &queries);
        
        Ok(AnnDataset {
            name: "GloVe".to_string(),
            vectors,
            queries,
            ground_truth,
        })
    }
    
    fn pad_to_768(vectors: Vec<Vec<f32>>) -> Vec<[f32; 768]> {
        vectors.into_iter()
            .map(|v| {
                let mut arr = [0.0f32; 768];
                arr[..v.len()].copy_from_slice(&v);
                arr
            })
            .collect()
    }
}
```

## Integration Points

### Create benchmark binary
```rust
// benches/vector_comparison.rs

use criterion::{criterion_group, criterion_main, Criterion};

fn benchmark_recall(c: &mut Criterion) {
    let dataset = DatasetLoader::load_sift1m().unwrap();
    
    let mut group = c.benchmark_group("recall@10");
    
    // Engram
    let engram = EngramAnnIndex::new();
    group.bench_function("engram", |b| {
        b.iter(|| {
            engram.search(&dataset.queries[0], 10)
        });
    });
    
    // FAISS
    let faiss = FaissAnnIndex::new_hnsw(768, 16).unwrap();
    group.bench_function("faiss", |b| {
        b.iter(|| {
            faiss.search(&dataset.queries[0], 10)
        });
    });
    
    // Annoy
    let annoy = AnnoyAnnIndex::new(768, 10).unwrap();
    group.bench_function("annoy", |b| {
        b.iter(|| {
            annoy.search(&dataset.queries[0], 10)
        });
    });
    
    group.finish();
}

criterion_group!(benches, benchmark_recall);
criterion_main!(benches);
```

## Testing Strategy

### Integration Tests
```rust
#[test]
fn test_recall_measurement() {
    let framework = BenchmarkFramework::new();
    
    let results = vec![(0, 0.9), (1, 0.8), (3, 0.7)];
    let ground_truth = vec![0, 1, 2];
    
    let recall = framework.calculate_recall(&results, &ground_truth, 3);
    assert!((recall - 0.667).abs() < 0.01); // 2 out of 3
}

#[test]
fn test_all_implementations_achieve_target() {
    let mut framework = BenchmarkFramework::new();
    framework.add_implementation(Box::new(EngramAnnIndex::new()));
    framework.add_implementation(Box::new(FaissAnnIndex::new_hnsw(768, 16).unwrap()));
    framework.add_implementation(Box::new(AnnoyAnnIndex::new(768, 10).unwrap()));
    
    let results = framework.run_comparison();
    
    for (impl_name, metrics) in results.iter() {
        assert!(
            metrics.avg_recall >= 0.9,
            "{} recall {} below target",
            impl_name,
            metrics.avg_recall
        );
        
        assert!(
            metrics.avg_latency < Duration::from_millis(1),
            "{} latency {:?} above 1ms",
            impl_name,
            metrics.avg_latency
        );
    }
}
```

## Acceptance Criteria
- [ ] Benchmark framework runs all three implementations
- [ ] Standard datasets (SIFT1M, GloVe) load correctly
- [ ] Recall@10 measurement accurate and consistent
- [ ] Engram achieves ≥90% recall@10 on all datasets
- [ ] Query latency <1ms for 95th percentile
- [ ] Results exported in standard format (CSV, JSON)

## Performance Targets
- Engram recall@10: ≥90%
- Engram P95 latency: <1ms
- Engram memory usage: <2x FAISS for same recall
- Build time: <60s for 1M vectors
- Benchmark runtime: <30 minutes for full comparison

## Risk Mitigation
- Use official FAISS/Annoy bindings for fair comparison
- Multiple runs to account for variance
- Warm-up iterations before measurement
- Ground truth validation against reference implementations
