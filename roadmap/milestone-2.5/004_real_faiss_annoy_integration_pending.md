# Task 004: Real FAISS/Annoy Library Integration

## Status: Pending
## Priority: P1 - Important (Validation Critical)
## Estimated Effort: 2 days
## Dependencies: None

## Objective

Replace mock FAISS/Annoy implementations with real library bindings to validate Engram's 90% recall@10 performance claim against industry-standard ANN systems.

## Current State

**Framework EXISTS but Uses MOCKS:**
- ✅ `engram-core/benches/ann_comparison.rs:1-197` - Full benchmark framework
- ✅ `engram-core/benches/datasets.rs:1-178` - Dataset loaders
- ❌ `engram-core/benches/mock_faiss.rs:1-126` - **Fake FAISS** (exact search + noise)
- ❌ `engram-core/benches/mock_annoy.rs:1-133` - **Fake Annoy** (exact search + noise)
- ❌ Roadmap claims "complete" but mocks can't validate real performance

**Current Mock Implementation:**
```rust
// engram-core/benches/mock_faiss.rs:80-95 (CURRENT)
fn search(&self, query: &[f32; 768], k: usize) -> Vec<(usize, f32)> {
    // FAKE: Exact search with random perturbation
    let mut results = exact_search(query, &self.data);
    results.iter_mut().for_each(|(_, dist)| {
        *dist += rand::random::<f32>() * 0.1; // Fake ANN error
    });
    results
}
```

## Implementation Steps

### Step 1: Add Real Library Dependencies (30 min)

**File**: `engram-core/Cargo.toml`

**Line 45** - Add FAISS/Annoy dependencies:
```toml
[dev-dependencies]
# ... existing dev deps ...

# ANN library bindings for benchmarks
faiss = { version = "0.11", optional = true }
annoy = { version = "0.2", optional = true }

[features]
# ... existing features ...

# Benchmark features
ann_benchmarks = ["faiss", "annoy"]
```

### Step 2: Implement Real FAISS Wrapper (3 hours)

**File**: `engram-core/benches/faiss_ann.rs` (create new)

```rust
//! Real FAISS implementation for ANN benchmarks

use crate::ann_comparison::{AnnIndex, Result};
use faiss::{Index, IndexFlatL2, IndexHNSW, MetricType, index_factory};

pub struct FaissAnnIndex {
    index: Box<dyn Index>,
    dimension: usize,
    index_type: FaissIndexType,
}

#[derive(Debug, Clone, Copy)]
pub enum FaissIndexType {
    FlatL2,
    HNSW { m: usize },
    IVFFlat { nlist: usize },
}

impl FaissAnnIndex {
    /// Create FAISS Flat (exact) index
    pub fn new_flat(dimension: usize) -> Result<Self> {
        let index = IndexFlatL2::new(dimension)?;
        Ok(Self {
            index: Box::new(index),
            dimension,
            index_type: FaissIndexType::FlatL2,
        })
    }

    /// Create FAISS HNSW index (comparable to Engram)
    pub fn new_hnsw(dimension: usize, m: usize) -> Result<Self> {
        let description = format!("HNSW{}", m);
        let index = index_factory(dimension, &description, MetricType::L2)?;

        Ok(Self {
            index,
            dimension,
            index_type: FaissIndexType::HNSW { m },
        })
    }

    /// Create FAISS IVF index
    pub fn new_ivf_flat(dimension: usize, nlist: usize) -> Result<Self> {
        let description = format!("IVF{},Flat", nlist);
        let index = index_factory(dimension, &description, MetricType::L2)?;

        Ok(Self {
            index,
            dimension,
            index_type: FaissIndexType::IVFFlat { nlist },
        })
    }
}

impl AnnIndex for FaissAnnIndex {
    fn build(&mut self, vectors: &[[f32; 768]]) -> Result<()> {
        // Flatten to FAISS format: [x1,y1,z1, x2,y2,z2, ...]
        let flat_vectors: Vec<f32> = vectors
            .iter()
            .flat_map(|v| v.iter().copied())
            .collect();

        let n = vectors.len();

        // Train index if needed (IVF requires training)
        if !self.index.is_trained() {
            self.index.train(n, &flat_vectors)?;
        }

        // Add vectors to index
        self.index.add(n, &flat_vectors)?;

        Ok(())
    }

    fn search(&self, query: &[f32; 768], k: usize) -> Vec<(usize, f32)> {
        let mut distances = vec![0.0f32; k];
        let mut labels = vec![0i64; k];

        // FAISS search returns (distances, labels)
        match self.index.search(1, query, k, &mut distances, &mut labels) {
            Ok(_) => {
                labels
                    .iter()
                    .zip(distances.iter())
                    .filter(|(&label, _)| label >= 0) // Filter invalid results
                    .map(|(&label, &dist)| {
                        // Convert L2 distance to similarity score
                        let similarity = 1.0 / (1.0 + dist);
                        (label as usize, similarity)
                    })
                    .collect()
            }
            Err(e) => {
                eprintln!("FAISS search failed: {:?}", e);
                Vec::new()
            }
        }
    }

    fn memory_usage(&self) -> usize {
        // Estimate: vectors + index overhead
        let n = self.index.ntotal() as usize;
        let vector_bytes = n * self.dimension * std::mem::size_of::<f32>();

        let overhead = match self.index_type {
            FaissIndexType::FlatL2 => 0,
            FaissIndexType::HNSW { m } => n * m * 8, // ~8 bytes per link
            FaissIndexType::IVFFlat { nlist } => nlist * self.dimension * 4,
        };

        vector_bytes + overhead
    }

    fn name(&self) -> &str {
        match self.index_type {
            FaissIndexType::FlatL2 => "FAISS-Flat",
            FaissIndexType::HNSW { m } => "FAISS-HNSW",
            FaissIndexType::IVFFlat { nlist } => "FAISS-IVF",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_faiss_hnsw() {
        let mut index = FaissAnnIndex::new_hnsw(128, 16).unwrap();

        // Create test data
        let vectors: Vec<[f32; 128]> = (0..100)
            .map(|i| {
                let mut v = [0.0; 128];
                v[0] = i as f32;
                v
            })
            .collect();

        // Convert to 768-dim by padding
        let vectors_768: Vec<[f32; 768]> = vectors
            .iter()
            .map(|v| {
                let mut padded = [0.0; 768];
                padded[..128].copy_from_slice(v);
                padded
            })
            .collect();

        index.build(&vectors_768).unwrap();

        // Search
        let mut query = [0.0; 768];
        query[0] = 5.0; // Should match vector 5

        let results = index.search(&query, 5);
        assert!(!results.is_empty());
    }
}
```

### Step 3: Implement Real Annoy Wrapper (2 hours)

**File**: `engram-core/benches/annoy_ann.rs` (create new)

```rust
//! Real Annoy implementation for ANN benchmarks

use crate::ann_comparison::{AnnIndex, Result};
use annoy::{Annoy, IndexBuilder, Search};

pub struct AnnoyAnnIndex {
    index: Option<Annoy>,
    dimension: usize,
    n_trees: usize,
    builder: IndexBuilder,
}

impl AnnoyAnnIndex {
    pub fn new(dimension: usize, n_trees: usize) -> Result<Self> {
        Ok(Self {
            index: None,
            dimension,
            n_trees,
            builder: IndexBuilder::new(dimension),
        })
    }
}

impl AnnIndex for AnnoyAnnIndex {
    fn build(&mut self, vectors: &[[f32; 768]]) -> Result<()> {
        // Add vectors to builder
        for (idx, vector) in vectors.iter().enumerate() {
            self.builder.add_item(idx, vector)?;
        }

        // Build index with specified number of trees
        let index = self.builder.build(self.n_trees)?;
        self.index = Some(index);

        Ok(())
    }

    fn search(&self, query: &[f32; 768], k: usize) -> Vec<(usize, f32)> {
        let index = match &self.index {
            Some(idx) => idx,
            None => return Vec::new(),
        };

        // Annoy search returns indices and distances
        match index.search(query, k, Search::MaxNodes(1000)) {
            Ok(results) => {
                results
                    .into_iter()
                    .map(|(idx, dist)| {
                        // Convert angular distance to similarity
                        // Angular distance ∈ [0, 2], similarity ∈ [0, 1]
                        let similarity = 1.0 - (dist / 2.0);
                        (idx, similarity)
                    })
                    .collect()
            }
            Err(e) => {
                eprintln!("Annoy search failed: {:?}", e);
                Vec::new()
            }
        }
    }

    fn memory_usage(&self) -> usize {
        let index = match &self.index {
            Some(idx) => idx,
            None => return 0,
        };

        // Estimate: n_items * n_trees * node_size
        let n_items = index.len();
        let tree_overhead = 32; // bytes per node
        n_items * self.n_trees * tree_overhead
    }

    fn name(&self) -> &str {
        "Annoy"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_annoy_search() {
        let mut index = AnnoyAnnIndex::new(768, 10).unwrap();

        // Create test vectors
        let vectors: Vec<[f32; 768]> = (0..100)
            .map(|i| {
                let mut v = [0.0; 768];
                v[0] = i as f32 / 100.0;
                v
            })
            .collect();

        index.build(&vectors).unwrap();

        // Search
        let mut query = [0.0; 768];
        query[0] = 0.05;

        let results = index.search(&query, 5);
        assert!(!results.is_empty());
    }
}
```

### Step 4: Update Benchmark to Use Real Libraries (1 hour)

**File**: `engram-core/benches/ann_comparison.rs`

**Line 10-20** - Update imports:
```rust
// Remove mock imports
// mod mock_faiss;
// mod mock_annoy;

// Add real implementations
#[cfg(feature = "ann_benchmarks")]
mod faiss_ann;
#[cfg(feature = "ann_benchmarks")]
mod annoy_ann;

// Conditional compilation for benchmarks
#[cfg(not(feature = "ann_benchmarks"))]
compile_error!("ANN benchmarks require 'ann_benchmarks' feature. Run with: cargo bench --features ann_benchmarks");
```

**Line 150-180** - Update benchmark main:
```rust
fn run_ann_comparison(c: &mut Criterion) {
    let dataset = load_sift1m_sample(1000); // 1k vectors for quick test

    let mut group = c.benchmark_group("ann_comparison");

    // Engram HNSW
    group.bench_function("engram", |b| {
        let mut engram = EngramAnnIndex::new();
        engram.build(&dataset.vectors).unwrap();

        b.iter(|| {
            let results = engram.search(&dataset.queries[0], 10);
            criterion::black_box(results);
        });
    });

    // FAISS HNSW
    #[cfg(feature = "ann_benchmarks")]
    group.bench_function("faiss_hnsw", |b| {
        let mut faiss = faiss_ann::FaissAnnIndex::new_hnsw(768, 16).unwrap();
        faiss.build(&dataset.vectors).unwrap();

        b.iter(|| {
            let results = faiss.search(&dataset.queries[0], 10);
            criterion::black_box(results);
        });
    });

    // Annoy
    #[cfg(feature = "ann_benchmarks")]
    group.bench_function("annoy", |b| {
        let mut annoy = annoy_ann::AnnoyAnnIndex::new(768, 10).unwrap();
        annoy.build(&dataset.vectors).unwrap();

        b.iter(|| {
            let results = annoy.search(&dataset.queries[0], 10);
            criterion::black_box(results);
        });
    });

    group.finish();
}
```

### Step 5: Add Recall Validation Test (2 hours)

**File**: `engram-core/benches/ann_validation.rs` (create new)

```rust
//! Validate Engram achieves ≥90% recall@10 compared to ground truth

use engram_core::benches::{
    ann_comparison::{AnnIndex, compute_recall},
    datasets::load_sift1m_sample,
    faiss_ann::FaissAnnIndex,
    annoy_ann::AnnoyAnnIndex,
};

#[test]
#[cfg(feature = "ann_benchmarks")]
fn validate_engram_recall() {
    let dataset = load_sift1m_sample(10_000);

    // Compute ground truth using FAISS Flat (exact search)
    let mut ground_truth_index = FaissAnnIndex::new_flat(768).unwrap();
    ground_truth_index.build(&dataset.vectors).unwrap();

    let ground_truth: Vec<Vec<usize>> = dataset
        .queries
        .iter()
        .map(|query| {
            ground_truth_index
                .search(query, 10)
                .into_iter()
                .map(|(idx, _)| idx)
                .collect()
        })
        .collect();

    // Test Engram
    let mut engram = EngramAnnIndex::new();
    engram.build(&dataset.vectors).unwrap();

    let mut total_recall = 0.0;
    for (query, truth) in dataset.queries.iter().zip(ground_truth.iter()) {
        let results = engram.search(query, 10);
        let indices: Vec<usize> = results.into_iter().map(|(idx, _)| idx).collect();
        let recall = compute_recall(&indices, truth);
        total_recall += recall;
    }

    let avg_recall = total_recall / dataset.queries.len() as f32;

    println!("Engram Recall@10: {:.2}%", avg_recall * 100.0);
    assert!(
        avg_recall >= 0.90,
        "Engram recall ({:.2}%) below 90% target",
        avg_recall * 100.0
    );
}

fn compute_recall(results: &[usize], ground_truth: &[usize]) -> f32 {
    let found = results
        .iter()
        .filter(|&idx| ground_truth.contains(idx))
        .count();

    found as f32 / ground_truth.len() as f32
}

#[test]
#[cfg(feature = "ann_benchmarks")]
fn compare_all_implementations() {
    let dataset = load_sift1m_sample(1000);

    // Build all indices
    let mut engram = EngramAnnIndex::new();
    engram.build(&dataset.vectors).unwrap();

    let mut faiss = FaissAnnIndex::new_hnsw(768, 16).unwrap();
    faiss.build(&dataset.vectors).unwrap();

    let mut annoy = AnnoyAnnIndex::new(768, 10).unwrap();
    annoy.build(&dataset.vectors).unwrap();

    // Compare results
    let query = &dataset.queries[0];

    let engram_results = engram.search(query, 10);
    let faiss_results = faiss.search(query, 10);
    let annoy_results = annoy.search(query, 10);

    println!("Engram: {:?}", engram_results.iter().map(|(i, _)| i).collect::<Vec<_>>());
    println!("FAISS:  {:?}", faiss_results.iter().map(|(i, _)| i).collect::<Vec<_>>());
    println!("Annoy:  {:?}", annoy_results.iter().map(|(i, _)| i).collect::<Vec<_>>());

    // All should return 10 results
    assert_eq!(engram_results.len(), 10);
    assert_eq!(faiss_results.len(), 10);
    assert_eq!(annoy_results.len(), 10);
}
```

### Step 6: Update Documentation (30 min)

**File**: `roadmap/milestone-2/005_faiss_annoy_benchmarks_complete.md`

**Line 3-7** - Update status:
```markdown
# Task 005: FAISS and Annoy Benchmark Framework

## Status: **Complete** ✅
## Completion: 100% (Real FAISS/Annoy bindings integrated)
## Priority: P1 - Validation Critical
```

**Line 15-20** - Update implementation status:
```markdown
## Current Implementation Status
- ✅ Benchmark harness with pluggable `AnnIndex` trait
- ✅ Real FAISS bindings via faiss-rs (HNSW, IVF, Flat indices)
- ✅ Real Annoy bindings via annoy-rs
- ✅ SIFT1M dataset loader and ground truth validation
- ✅ Recall@10 validation test (Engram achieves >90%)
- ✅ Performance comparison benchmarks (Criterion)
```

## Testing Strategy

### Build and Run

```bash
# Install FAISS system dependencies (macOS)
brew install faiss

# Install FAISS system dependencies (Ubuntu)
sudo apt-get install libfaiss-dev

# Run benchmarks with real libraries
cargo bench --features ann_benchmarks

# Run recall validation
cargo test --features ann_benchmarks --test ann_validation

# Quick smoke test (1k vectors)
cargo bench --features ann_benchmarks -- --quick
```

### Expected Results

**Recall@10 (Target: ≥90%)**
- Engram HNSW: 92-95%
- FAISS HNSW: 93-96%
- Annoy: 88-92%

**Query Latency P95 (Target: <1ms)**
- Engram: 0.8-1.2ms
- FAISS: 0.6-1.0ms
- Annoy: 1.0-1.5ms

**Memory Usage (1M vectors)**
- Engram: ~3.5GB
- FAISS HNSW: ~4.0GB
- Annoy: ~2.8GB

## Acceptance Criteria

- [ ] FAISS bindings compile and link correctly
- [ ] Annoy bindings compile and link correctly
- [ ] Benchmark runs without crashes
- [ ] Engram recall@10 ≥90% on SIFT1M sample
- [ ] Query latency P95 <1ms
- [ ] Results exported to `target/criterion/` directory
- [ ] Documentation updated to remove "mock" references

## Performance Targets

- Recall@10: ≥90% (Engram vs ground truth)
- Query latency: <1ms P95 (10k dataset)
- Memory usage: <2x FAISS for equivalent recall
- Build time: <60s for 1M vectors

## Files to Create

1. `engram-core/benches/faiss_ann.rs` - Real FAISS wrapper
2. `engram-core/benches/annoy_ann.rs` - Real Annoy wrapper
3. `engram-core/benches/ann_validation.rs` - Recall validation tests

## Files to Modify

1. `engram-core/Cargo.toml` - Add faiss/annoy dependencies
2. `engram-core/benches/ann_comparison.rs` - Use real implementations
3. `roadmap/milestone-2/005_faiss_annoy_benchmarks_complete.md` - Update status

## Files to Delete

1. `engram-core/benches/mock_faiss.rs` - Remove mock
2. `engram-core/benches/mock_annoy.rs` - Remove mock

## System Dependencies

**macOS:**
```bash
brew install faiss
```

**Ubuntu/Debian:**
```bash
sudo apt-get install libfaiss-dev
```

**Build from source (if packages unavailable):**
```bash
git clone https://github.com/facebookresearch/faiss.git
cd faiss
cmake -B build -DFAISS_ENABLE_GPU=OFF .
make -C build -j8
sudo make -C build install
```

## Notes

- FAISS requires C++11 compiler and BLAS library
- Annoy is header-only, easier to integrate
- Consider using pre-built Docker image for CI/CD
- Results will vary based on dataset and parameters
