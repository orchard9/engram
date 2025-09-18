# Task 004: Columnar Cold Storage with SIMD

## Status: Pending
## Priority: P2 - Optimization
## Estimated Effort: 3-4 days
## Dependencies: Task 001 (cold tier implementation)

## Objective
Optimize cold tier storage with columnar layout for SIMD batch operations on large vector datasets, achieving 10-100x performance improvement for analytical workloads through cache-optimal memory access patterns and vectorized operations.

## Research-Based Design Principles

### Key Insights from Vector Database Analysis
- **Apache Arrow format**: Zero-copy data interchange with 64-byte alignment for AVX-512
- **Faiss IVF architecture**: Columnar storage achieves >1M QPS on billion-scale datasets
- **Memory bandwidth optimization**: Sequential access improves utilization 3-5x
- **SIMD instruction utilization**: Increases from 20% to 85% with proper layout
- **Cache miss reduction**: 60-80% improvement through column-oriented operations

### Performance Characteristics
- Analytical queries: 10-100x speedup over row storage
- Memory bandwidth: 3-5x improvement through sequential access
- SIMD parallelism: 8x with AVX2, 16x with AVX-512
- Compression ratios: 2-10x improvement due to column homogeneity
- Cache utilization: Improves from 25% to 85%

## Current Codebase Analysis

### Existing Implementation
The cold tier already has a basic columnar storage implementation in `engram-core/src/storage/cold_tier.rs`:
- **ColumnarData struct** (lines 32-196): Already stores embeddings in column-major format
- **Batch similarity search** (lines 125-153): Basic SIMD-ready batch processing
- **Compaction support** (lines 156-196): Memory compaction with gap removal
- **ColdTier struct** (lines 199-353): Tier management with DashMap index

### SIMD Infrastructure
The compute module (`engram-core/src/compute/`) provides:
- **AVX2 implementations** (`avx2.rs`): FMA-optimized operations for x86_64
- **AVX512 implementations** (`avx512.rs`): 512-bit vector operations
- **Runtime dispatch** (`dispatch.rs`, `mod.rs`): CPU feature detection and automatic selection
- **VectorOps trait** (lines 39-61): Common interface for all SIMD implementations

## Files to Modify

### 1. Enhance `engram-core/src/storage/cold_tier.rs`
**Current Issues:**
- Line 81: Stores embeddings in row-major format (extends entire embedding at once)
- Lines 108-123: `get_embedding()` retrieves by row, not optimized for column access
- Lines 125-153: `batch_similarity_search()` processes row-by-row instead of column-wise
- No memory alignment for SIMD operations (needs 64-byte alignment)
- No compression or lazy loading

**Required Changes:**
- Replace `embeddings: Vec<f32>` with proper columnar structure
- Add 64-byte aligned allocation for AVX-512
- Implement true column-major storage with chunking
- Add SIMD gather/scatter operations for transpose
- Implement compression with product quantization

### 2. Create New `engram-core/src/storage/columnar.rs`
**New Module for Advanced Columnar Operations:**
```rust
// New file structure:
pub mod aligned;      // 64-byte aligned memory allocation
pub mod chunks;       // 1024-vector chunk management
pub mod compression;  // Product quantization and dictionary encoding
pub mod simd_ops;    // SIMD batch operations
pub mod query_plan;  // Query optimization and planning
```

### 3. Extend `engram-core/src/compute/mod.rs`
**Add New Methods to VectorOps Trait:**
- `fma_accumulate()`: Fused multiply-add for column operations
- `gather_f32()`: SIMD gather for non-contiguous access
- `horizontal_sum()`: Reduction operations for similarity scores
- `batch_dot_product_columnar()`: Optimized for columnar layout

### 4. Update `engram-core/src/compute/avx2.rs` and `avx512.rs`
**Add Columnar-Specific SIMD Operations:**
- Implement gather/scatter for efficient transpose
- Add FMA operations for dot product accumulation
- Implement horizontal reduction for similarity scores
- Add prefetching for streaming column access

## Implementation Plan

### Phase 1: Refactor Existing Columnar Layout (Day 1)

#### Step 1.1: Modify `engram-core/src/storage/cold_tier.rs`

**Replace lines 32-68 (ColumnarData struct):**
```rust
use std::alloc::{alloc_zeroed, Layout};
use std::ptr;

#[repr(align(64))]  // AVX-512 alignment
pub struct AlignedColumn {
    // Each column stores one dimension across all vectors
    data: *mut f32,           // 64-byte aligned raw pointer
    capacity: usize,          // Allocated capacity
    len: usize,              // Current number of values
    layout: Layout,          // Memory layout for deallocation
}

impl AlignedColumn {
    fn new(capacity: usize) -> Self {
        let layout = Layout::from_size_align(
            capacity * std::mem::size_of::<f32>(),
            64  // 64-byte alignment for AVX-512
        ).unwrap();

        let data = unsafe {
            let ptr = alloc_zeroed(layout);
            if ptr.is_null() {
                std::alloc::handle_alloc_error(layout);
            }
            ptr as *mut f32
        };

        Self { data, capacity, len: 0, layout }
    }

    unsafe fn push(&mut self, value: f32) {
        if self.len < self.capacity {
            ptr::write(self.data.add(self.len), value);
            self.len += 1;
        }
    }

    unsafe fn get_slice(&self) -> &[f32] {
        std::slice::from_raw_parts(self.data, self.len)
    }
}

struct ColumnarData {
    // 768 columns, one for each embedding dimension
    embedding_columns: Vec<AlignedColumn>,
    // Metadata columns (not aligned, regular storage)
    confidences: Vec<f32>,
    activations: Vec<f32>,
    creation_times: Vec<u64>,
    access_times: Vec<u64>,
    contents: Vec<String>,
    memory_ids: Vec<String>,
    count: usize,
    capacity: usize,
}

```

**Replace lines 69-104 (insert method):**
```rust
impl ColumnarData {
    fn new(capacity: usize) -> Self {
        let mut embedding_columns = Vec::with_capacity(768);
        for _ in 0..768 {
            embedding_columns.push(AlignedColumn::new(capacity));
        }

        Self {
            embedding_columns,
            confidences: Vec::with_capacity(capacity),
            activations: Vec::with_capacity(capacity),
            creation_times: Vec::with_capacity(capacity),
            access_times: Vec::with_capacity(capacity),
            contents: Vec::with_capacity(capacity),
            memory_ids: Vec::with_capacity(capacity),
            count: 0,
            capacity,
        }
    }

    fn insert(&mut self, memory: &Memory) -> Result<usize, StorageError> {
        if self.count >= self.capacity {
            return Err(StorageError::AllocationFailed("Capacity exceeded".to_string()));
        }

        let index = self.count;

        // Store embedding in true column-major format
        unsafe {
            for (dim, &value) in memory.embedding.iter().enumerate() {
                self.embedding_columns[dim].push(value);
            }
        }

        // Store metadata
        self.confidences.push(memory.confidence.raw());
        self.activations.push(memory.activation());
        self.creation_times.push(/* timestamp */);
        self.access_times.push(/* timestamp */);
        self.contents.push(memory.content.clone().unwrap_or_default());
        self.memory_ids.push(memory.id.clone());

        self.count += 1;
        Ok(index)
    }
}
```

#### Step 1.2: Update batch_similarity_search (lines 125-153)

**Replace with SIMD-optimized columnar version:**
```rust
fn batch_similarity_search(
    &self,
    query: &[f32; 768],
    threshold: f32,
) -> Vec<(usize, f32)> {
    use crate::compute;

    let mut similarities = vec![0.0f32; self.count];

    // Process columns in SIMD-friendly chunks
    const SIMD_WIDTH: usize = 8;  // AVX2 processes 8 floats

    for dim in 0..768 {
        let query_val = query[dim];
        let column_data = unsafe {
            self.embedding_columns[dim].get_slice()
        };

        // SIMD FMA: similarities += column * query_val
        compute::get_vector_ops().fma_accumulate(
            column_data,
            query_val,
            &mut similarities
        );
    }

    // Normalize similarities and filter by threshold
    let mut results: Vec<(usize, f32)> = similarities
        .iter()
        .enumerate()
        .filter_map(|(idx, &sim)| {
            if sim >= threshold {
                Some((idx, sim))
            } else {
                None
            }
        })
        .collect();

    results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    results
}
```

### Phase 2: Add SIMD Operations to Compute Module (Day 2)

#### Step 2.1: Extend `engram-core/src/compute/mod.rs` (line 61, add to VectorOps trait)

```rust
pub trait VectorOps: Send + Sync {
    // ... existing methods ...

    /// Fused multiply-add for columnar operations
    /// Computes: accumulator[i] += column[i] * scalar
    fn fma_accumulate(&self, column: &[f32], scalar: f32, accumulator: &mut [f32]);

    /// SIMD gather for non-contiguous memory access
    fn gather_f32(&self, base: &[f32], indices: &[usize]) -> Vec<f32>;

    /// Horizontal sum reduction across SIMD lanes
    fn horizontal_sum(&self, values: &[f32]) -> f32;

    /// Batch dot product optimized for columnar layout
    fn batch_dot_product_columnar(
        &self,
        query: &[f32; 768],
        columns: &[&[f32]],
        results: &mut [f32],
    );
}
```

#### Step 2.2: Implement in `engram-core/src/compute/avx2.rs` (add after line 94)

```rust
impl Avx2VectorOps {
    // Add these methods to the existing implementation

    fn fma_accumulate(&self, column: &[f32], scalar: f32, accumulator: &mut [f32]) {
        unsafe { fma_accumulate_avx2(column, scalar, accumulator) };
    }

    fn gather_f32(&self, base: &[f32], indices: &[usize]) -> Vec<f32> {
        unsafe { gather_f32_avx2(base, indices) }
    }

    fn horizontal_sum(&self, values: &[f32]) -> f32 {
        unsafe { horizontal_sum_avx2(values) }
    }
}

// Add these functions after existing AVX2 implementations
#[target_feature(enable = "avx2,fma")]
unsafe fn fma_accumulate_avx2(column: &[f32], scalar: f32, accumulator: &mut [f32]) {
    use std::arch::x86_64::*;

    let scalar_vec = _mm256_set1_ps(scalar);
    let chunks = column.len() / 8;

    for i in 0..chunks {
        let offset = i * 8;
        let col_vec = _mm256_loadu_ps(column.as_ptr().add(offset));
        let acc_vec = _mm256_loadu_ps(accumulator.as_ptr().add(offset));
        let result = _mm256_fmadd_ps(col_vec, scalar_vec, acc_vec);
        _mm256_storeu_ps(accumulator.as_mut_ptr().add(offset), result);
    }

    // Handle remainder with scalar operations
    for i in (chunks * 8)..column.len() {
        accumulator[i] += column[i] * scalar;
    }
}

#[target_feature(enable = "avx2")]
unsafe fn gather_f32_avx2(base: &[f32], indices: &[usize]) -> Vec<f32> {
    use std::arch::x86_64::*;

    let mut result = Vec::with_capacity(indices.len());

    // Process 8 indices at a time using gather
    let chunks = indices.len() / 8;

    for i in 0..chunks {
        let offset = i * 8;
        let idx_vec = _mm256_loadu_si256(indices.as_ptr().add(offset) as *const __m256i);
        let gathered = _mm256_i32gather_ps(base.as_ptr(), idx_vec, 4);

        let mut temp = [0f32; 8];
        _mm256_storeu_ps(temp.as_mut_ptr(), gathered);
        result.extend_from_slice(&temp);
    }

    // Handle remainder
    for &idx in &indices[chunks * 8..] {
        result.push(base[idx]);
    }

    result
}
```

### Phase 3: Create Columnar Module (Day 3)

#### Step 3.1: Create new file `engram-core/src/storage/columnar.rs`

```rust
//! Advanced columnar storage with SIMD optimization and compression

use super::{StorageError, AlignedColumn};
use crate::compute::VectorOps;
use std::sync::Arc;

pub mod compression;
pub mod query_plan;

/// Product quantization for vector compression
pub struct ProductQuantizer {
    codebooks: Vec<Vec<f32>>,  // Centroids for each subspace
    subspace_dims: usize,      // Dimensions per subspace (typically 8)
    num_centroids: usize,      // Number of centroids per subspace (256 for u8)
}

impl ProductQuantizer {
    pub fn new(subspace_dims: usize) -> Self {
        Self {
            codebooks: Vec::new(),
            subspace_dims,
            num_centroids: 256,
        }
    }

    pub fn train(&mut self, vectors: &[[f32; 768]], num_iterations: usize) {
        // K-means clustering for each subspace
        let num_subspaces = 768 / self.subspace_dims;

        for subspace in 0..num_subspaces {
            let start_dim = subspace * self.subspace_dims;
            let end_dim = start_dim + self.subspace_dims;

            // Extract subspace data
            let subspace_data: Vec<Vec<f32>> = vectors
                .iter()
                .map(|v| v[start_dim..end_dim].to_vec())
                .collect();

            // Run k-means to find centroids
            let centroids = self.kmeans(&subspace_data, self.num_centroids, num_iterations);
            self.codebooks.push(centroids);
        }
    }

    pub fn encode(&self, vector: &[f32; 768]) -> Vec<u8> {
        let mut codes = Vec::with_capacity(768 / self.subspace_dims);

        for (subspace, codebook) in self.codebooks.iter().enumerate() {
            let start_dim = subspace * self.subspace_dims;
            let end_dim = start_dim + self.subspace_dims;
            let subvector = &vector[start_dim..end_dim];

            // Find nearest centroid
            let mut min_dist = f32::MAX;
            let mut best_code = 0u8;

            for (code, centroid) in codebook.iter().enumerate() {
                let dist = self.euclidean_distance(subvector, centroid);
                if dist < min_dist {
                    min_dist = dist;
                    best_code = code as u8;
                }
            }

            codes.push(best_code);
        }

        codes
    }
}

```

#### Step 3.2: Add to `engram-core/src/storage/mod.rs` (after line 23)

```rust
pub mod columnar;  // Add this line
```

### Phase 4: Integration and Testing (Day 4)

#### Step 4.1: Update tests in `engram-core/src/storage/cold_tier.rs` (add after line 724)

```rust
#[cfg(test)]
mod columnar_tests {
    use super::*;

    #[test]
    fn test_aligned_column_allocation() {
        let column = AlignedColumn::new(1000);

        // Verify 64-byte alignment
        assert_eq!(column.data as usize % 64, 0);
        assert_eq!(column.capacity, 1000);
        assert_eq!(column.len, 0);
    }

    #[test]
    fn test_columnar_transpose() {
        let mut data = ColumnarData::new(10);

        // Create test memories with distinct embeddings
        for i in 0..5 {
            let mut embedding = [0.0f32; 768];
            embedding[0] = i as f32;

            let memory = create_test_memory_with_embedding(
                &format!("mem_{}", i),
                0.5,
                embedding
            );

            data.insert(&memory).unwrap();
        }

        // Verify columnar storage
        assert_eq!(data.count, 5);

        // Check first dimension column contains [0.0, 1.0, 2.0, 3.0, 4.0]
        unsafe {
            let first_col = data.embedding_columns[0].get_slice();
            for i in 0..5 {
                assert_eq!(first_col[i], i as f32);
            }
        }
    }

    #[tokio::test]
    async fn test_simd_similarity_search() {
        let cold_tier = ColdTier::new(1000);

        // Store test vectors
        for i in 0..100 {
            let mut embedding = [0.1f32; 768];
            embedding[i % 768] = 1.0;  // Make each vector unique

            let memory = create_test_memory_with_embedding(
                &format!("vec_{}", i),
                0.3,
                embedding
            );

            cold_tier.store(memory).await.unwrap();
        }

        // Search with query vector
        let mut query = [0.1f32; 768];
        query[5] = 1.0;  // Should match vec_5 closely

        let cue = CueBuilder::new()
            .embedding_search(query, Confidence::LOW)
            .max_results(10)
            .build();

        let results = cold_tier.recall(&cue).await.unwrap();

        assert!(!results.is_empty());
        // vec_5 should be among top results
        let ids: Vec<String> = results.iter().map(|(e, _)| e.id.clone()).collect();
        assert!(ids.contains(&"vec_5".to_string()));
    }

    #[test]
    fn test_fma_accumulate() {
        use crate::compute;

        let column = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let mut accumulator = vec![0.0; 8];
        let scalar = 2.0;

        compute::get_vector_ops().fma_accumulate(&column, scalar, &mut accumulator);

        // Verify FMA results: accumulator[i] = column[i] * 2.0
        assert_eq!(accumulator, vec![2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0]);
    }
}
```

#### Step 4.2: Create benchmark tests in `engram-core/benches/columnar_bench.rs`

```rust
use criterion::{black_box, criterion_group, criterion_main, Criterion};
use engram_core::storage::cold_tier::ColdTier;

fn bench_columnar_similarity(c: &mut Criterion) {
    let rt = tokio::runtime::Runtime::new().unwrap();

    c.bench_function("columnar_similarity_100k", |b| {
        let cold_tier = setup_cold_tier_with_vectors(100_000);
        let query = [0.5f32; 768];

        b.iter(|| {
            rt.block_on(async {
                cold_tier.simd_similarity_search(&query, 100, 0.5)
            })
        });
    });
}

criterion_group!(benches, bench_columnar_similarity);
criterion_main!(benches);
```

## Enhanced Acceptance Criteria

### Performance Requirements
- [ ] **Memory Bandwidth**: >40% reduction through sequential access patterns
- [ ] **SIMD Speedup**: >8x improvement over scalar operations (AVX2 target)
- [ ] **Query Throughput**: >100K similarity searches/second on modern hardware
- [ ] **Memory Overhead**: <15% vs optimized row storage
- [ ] **Cache Efficiency**: >70% L3 cache hit rate during batch operations
- [ ] **Compression Ratio**: >3x size reduction with <5% accuracy loss

### Quality Requirements
- [ ] **NUMA Awareness**: Thread-local allocation with <10% cross-socket traffic
- [ ] **Error Handling**: Graceful degradation to row storage on SIMD failures
- [ ] **Backward Compatibility**: Seamless integration with existing StorageTier API
- [ ] **Monitoring Integration**: Real-time performance metrics and alerting
- [ ] **Memory Safety**: Zero unsafe operations in public API surface

## Detailed Performance Targets

### Latency Targets
- **Similarity search**: <5ms for 100K vectors (p99)
- **Batch append**: <0.5ms for 1000 vectors
- **Column loading**: <50ms for 1M vector column from cold storage
- **Compression**: <100ms background compression per 10K vectors

### Throughput Targets
- **Query processing**: >200K QPS sustained load
- **Batch ingestion**: >50K vectors/second with real-time indexing
- **Memory bandwidth**: >80% of theoretical peak during SIMD operations
- **Storage I/O**: >500MB/s for compressed column access

### Resource Utilization
- **CPU utilization**: >90% during batch similarity computation
- **Memory efficiency**: <8GB RAM for 1M 768-dimensional vectors
- **Storage compression**: <50% of uncompressed size with lossless encoding
- **Network bandwidth**: <100MB/s for distributed query coordination

## Risk Mitigation Strategies

### Technical Risks
1. **SIMD compatibility**: Runtime CPU feature detection with scalar fallback
2. **Memory alignment**: Automatic alignment enforcement with compile-time checks
3. **Numerical stability**: Validated against reference implementations
4. **Compression accuracy**: Configurable quality vs size trade-offs
5. **NUMA scalability**: Automatic work distribution based on topology

### Operational Risks
1. **Migration complexity**: Gradual rollout with A/B testing
2. **Memory pressure**: Dynamic compression based on available RAM
3. **Query pattern changes**: Adaptive access pattern learning
4. **Hardware diversity**: Multi-architecture support (x86-64, ARM64)
5. **Production monitoring**: Comprehensive observability and alerting

This enhanced implementation plan incorporates research findings to deliver production-ready columnar storage with measurable performance improvements and robust operational characteristics.