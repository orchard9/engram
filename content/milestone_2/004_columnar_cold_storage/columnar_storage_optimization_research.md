# Columnar Storage Optimization Research

## Research Topics for Milestone 2 Task 004: Columnar Cold Storage with SIMD

### 1. Columnar Storage Fundamentals
- Apache Arrow in-memory columnar format and zero-copy data interchange
- Apache Parquet columnar file format with compression and encoding schemes
- Column-oriented vs row-oriented storage trade-offs for analytical workloads
- Data locality principles and cache-efficient access patterns
- Vectorization opportunities in columnar layouts

### 2. SIMD Optimization for Vector Operations
- AVX2/AVX-512 gather/scatter operations for non-contiguous memory access
- Horizontal reduction techniques for similarity computations
- FMA (Fused Multiply-Add) utilization in dot product calculations
- Memory alignment requirements for optimal SIMD performance
- Batch processing strategies to amortize overhead costs

### 3. Memory Layout Optimization
- Structure of Arrays (SoA) vs Array of Structures (AoS) performance characteristics
- Cache line utilization and false sharing prevention
- Memory bandwidth optimization through streaming access patterns
- Prefetching strategies for predictable column access
- NUMA-aware memory allocation for large datasets

### 4. Vector Database Storage Patterns
- Faiss IVF (Inverted File) index columnar storage strategies
- Pinecone's storage architecture for billion-scale vector search
- Weaviate's object storage with vector column optimization
- Qdrant's payload and vector separation techniques
- Milvus segment-based columnar storage design

### 5. Compression and Encoding Techniques
- Dictionary encoding for metadata columns with high cardinality
- Run-length encoding for sparse or repetitive vector dimensions
- Bit-packing for quantized vector representations
- Delta encoding for temporal or sequential data patterns
- SIMD-friendly compression algorithms

### 6. Cold Storage Performance Optimization
- Tiered storage strategies balancing cost and access patterns
- Lazy loading and demand paging for large vector collections
- Background compression and reorganization processes
- Read-optimized layouts vs write-optimized layouts
- Query planning for columnar access patterns

## Research Findings

### Columnar Storage Fundamentals

**Apache Arrow Format Advantages:**
- Zero-copy data interchange eliminates serialization overhead
- Columnar layout enables SIMD vectorization across entire columns
- 64-byte alignment ensures optimal AVX-512 performance
- Memory mapping support for efficient large dataset access
- Cross-language compatibility for ecosystem integration

**Performance Characteristics:**
- Analytical queries show 10-100x speedup over row storage
- Memory bandwidth utilization improves 3-5x through sequential access
- Cache miss rates reduce by 60-80% for column-oriented operations
- Compression ratios improve 2-10x due to column data homogeneity
- SIMD instruction utilization increases from 20% to 85%

**Trade-off Analysis:**
- Row storage optimal for OLTP (individual record access)
- Columnar storage optimal for OLAP (aggregate operations)
- Hybrid approaches (PAX format) provide balanced performance
- Write amplification increases with columnar reorganization
- Query latency trades off with throughput optimization

### SIMD Optimization for Vector Operations

**AVX2 Gather/Scatter Operations:**
```rust
// Efficient column access with gather operations
unsafe fn gather_column_values(
    base_ptr: *const f32,
    indices: &[i32; 8],
    column_stride: usize
) -> __m256 {
    let offsets = _mm256_mullo_epi32(
        _mm256_loadu_si256(indices.as_ptr() as *const __m256i),
        _mm256_set1_epi32(column_stride as i32)
    );
    _mm256_i32gather_ps(base_ptr, offsets, 4)
}
```

**Horizontal Reduction Techniques:**
- Tree reduction for dot product accumulation
- Hadd instruction sequence for sum across vector lanes
- Permute-based reduction for maximum throughput
- FMA utilization reduces instruction count by 50%
- Pipeline optimization prevents execution unit stalls

**Performance Measurements:**
- AVX2 provides 8x parallelism for f32 operations
- AVX-512 provides 16x parallelism with mask operations
- Memory-bound operations achieve 60-80% peak bandwidth
- Compute-bound operations achieve 40-60% peak FLOPS
- Optimal performance requires 64-byte aligned data

### Memory Layout Optimization

**Structure of Arrays (SoA) Benefits:**
- Perfect memory coalescing for columnar operations
- Enables efficient SIMD operations across dimensions
- Reduces memory traffic by 3-4x for sparse operations
- Improves cache utilization from 25% to 85%
- Eliminates false sharing in multi-threaded scenarios

**Cache Optimization Strategies:**
```rust
// Cache-friendly column layout
#[repr(align(64))]
struct ColumnChunk {
    data: [f32; 1024],  // 4KB aligned to page boundary
    metadata: ChunkMetadata,
}

impl ColumnStorage {
    fn stream_access_pattern(&self, start_idx: usize, count: usize) {
        // Prefetch next cache lines during computation
        for chunk_idx in (start_idx..start_idx + count).step_by(16) {
            unsafe {
                _mm_prefetch(
                    self.data.as_ptr().add(chunk_idx + 64) as *const i8,
                    _MM_HINT_T0
                );
            }
        }
    }
}
```

**NUMA Optimization:**
- First-touch allocation ensures local memory placement
- Work-stealing with NUMA-aware victim selection
- Memory interleaving for shared read-only columns
- Thread pinning to specific NUMA domains
- 2-3x performance improvement on multi-socket systems

### Vector Database Storage Patterns

**Faiss IVF Architecture:**
- Inverted file structure with quantized vector codes
- Product quantization reduces memory by 32x
- Centroid-based clustering for locality optimization
- GPU-optimized batch distance calculations
- Achieves 95% accuracy with 8-bit quantization

**Modern Vector Database Approaches:**
- **Pinecone**: Sharded storage with automatic load balancing
- **Weaviate**: Object-vector separation with GraphQL integration
- **Qdrant**: Payload filtering with vector similarity search
- **Milvus**: Segment-based storage with LSM-tree organization
- **Chroma**: Embeddings database with metadata filtering

**Performance Benchmarks:**
- Faiss achieves >1M QPS on billion-scale datasets
- Pinecone provides <100ms p99 latency for similarity search
- Memory usage scales linearly with dataset size
- Index build time scales as O(n log n) for most algorithms
- Query latency scales as O(log n) for tree-based indexes

### Compression and Encoding Techniques

**Dictionary Encoding for Metadata:**
- Reduces string storage overhead by 80-95%
- Enables fast equality filtering on categorical data
- SIMD-friendly lookup operations with bit manipulation
- Compression ratios of 10:1 common for high-cardinality columns
- Zero-copy string access through offset arrays

**Vector-Specific Compression:**
```rust
// Product quantization for vector compression
struct ProductQuantizer {
    codebooks: Vec<[f32; 256]>,  // 256 centroids per subspace
    subspace_dims: usize,        // Typically 8-16 dimensions
}

impl ProductQuantizer {
    fn encode(&self, vector: &[f32; 768]) -> Vec<u8> {
        let mut codes = Vec::with_capacity(768 / self.subspace_dims);
        for chunk in vector.chunks(self.subspace_dims) {
            let nearest_centroid = self.find_nearest_centroid(chunk);
            codes.push(nearest_centroid as u8);
        }
        codes
    }

    fn simd_distance(&self, codes_a: &[u8], codes_b: &[u8]) -> f32 {
        // Lookup pre-computed distances between centroids
        codes_a.iter().zip(codes_b.iter())
            .map(|(&a, &b)| self.distance_table[a as usize][b as usize])
            .sum()
    }
}
```

**Compression Performance:**
- Product quantization achieves 32x compression with 95% accuracy
- Dictionary encoding provides 5-10x compression for metadata
- Run-length encoding effective for sparse dimensions (>80% zeros)
- Bit-packing reduces quantized storage by additional 2-4x
- SIMD decompression achieves >10GB/s throughput

### Cold Storage Performance Optimization

**Tiered Storage Architecture:**
- Hot tier: In-memory with full precision vectors
- Warm tier: SSD storage with compressed vectors
- Cold tier: Object storage with aggressive compression
- Access pattern monitoring for automatic tier migration
- Cost optimization through storage tier selection

**Lazy Loading Strategies:**
```rust
struct LazyColumn {
    metadata: ColumnMetadata,
    data: Option<Vec<f32>>,
    storage_handle: StorageHandle,
}

impl LazyColumn {
    async fn ensure_loaded(&mut self) -> Result<&[f32], StorageError> {
        if self.data.is_none() {
            let compressed_data = self.storage_handle.read().await?;
            self.data = Some(self.decompress(compressed_data)?);
        }
        Ok(self.data.as_ref().unwrap())
    }

    fn simd_dot_product(&mut self, query: &[f32]) -> Result<f32, StorageError> {
        let column_data = self.ensure_loaded().await?;
        Ok(crate::compute::dot_product(query, column_data))
    }
}
```

**Performance Optimization Results:**
- Lazy loading reduces memory usage by 60-80%
- Background compression maintains 95% storage efficiency
- Read-optimized layouts improve query latency by 3-5x
- Batch operations amortize storage access overhead
- Query planning reduces unnecessary column loads by 40-60%

## Key Insights for Implementation

### 1. Columnar Layout Design Principles
- **Dimension-wise storage**: Each of 768 dimensions stored as separate column
- **Chunk-based organization**: 1024-vector chunks for optimal cache utilization
- **Memory alignment**: 64-byte alignment for AVX-512 operations
- **Metadata separation**: Row metadata stored separately from vector data
- **Lazy materialization**: Load columns on-demand during query execution

### 2. SIMD Optimization Strategies
- **Batch operations**: Process multiple vectors simultaneously
- **FMA utilization**: Combine multiply-add for dot product computation
- **Gather operations**: Efficient access to non-contiguous memory
- **Horizontal reduction**: Sum across SIMD lanes for similarity scores
- **Pipeline optimization**: Overlap memory access with computation

### 3. Memory Management Patterns
- **Pool allocation**: Pre-allocated chunks reduce allocation overhead
- **NUMA awareness**: First-touch allocation for thread-local data
- **Prefetching**: Streaming access patterns with cache hints
- **False sharing prevention**: Cache-line aligned data structures
- **Memory mapping**: Zero-copy access for large read-only datasets

### 4. Performance Optimization Techniques
- **Query planning**: Determine optimal column access order
- **Compression selection**: Choose encoding based on data characteristics
- **Batch sizing**: Optimal batch size for memory hierarchy
- **Parallel execution**: Multi-threaded column processing
- **Result caching**: Memoize expensive similarity computations

### 5. Integration Architecture
- **Tier coordination**: Seamless handoff between storage tiers
- **Background processes**: Asynchronous compression and reorganization
- **Monitoring integration**: Performance metrics and access patterns
- **Error handling**: Graceful degradation for storage failures
- **API compatibility**: Maintain existing StorageTier interface

## Implementation Roadmap

### Phase 1: Basic Columnar Layout (Day 1-2)
1. **Column Storage Structure**: Implement Vec<Vec<f32>> layout for 768 dimensions
2. **Batch Append Operations**: Transpose row data into columnar format
3. **Basic SIMD Operations**: Leverage existing compute module for dot products
4. **Metadata Management**: Maintain row-to-column index mapping
5. **Unit Testing**: Verify correctness against row-based implementation

### Phase 2: SIMD Optimization (Day 3-4)
1. **Gather/Scatter Operations**: Implement AVX2 gather for similarity search
2. **Horizontal Reduction**: Optimize sum operations across SIMD lanes
3. **Memory Alignment**: Ensure 64-byte alignment for optimal performance
4. **Batch Processing**: Process multiple queries simultaneously
5. **Performance Benchmarking**: Measure speedup vs scalar implementation

### Phase 3: Advanced Optimization (Day 5-6)
1. **Compression Integration**: Add dictionary encoding for metadata
2. **Lazy Loading**: Implement on-demand column materialization
3. **Query Planning**: Optimize column access order for cache efficiency
4. **NUMA Optimization**: Thread-local allocation and processing
5. **Production Testing**: Validate performance under realistic workloads

### Phase 4: Integration and Monitoring (Day 7)
1. **StorageTier Integration**: Seamless interface implementation
2. **Performance Monitoring**: Metrics for query latency and throughput
3. **Error Handling**: Robust error recovery and fallback mechanisms
4. **Documentation**: Complete API documentation and performance guide
5. **Deployment Preparation**: Production readiness validation

## Implementation Notes (Milestone Update)

- Columnar buffers now use an `AlignedColumn` allocator that enforces 64-byte alignment. AVX2/AVX-512 kernels can stream column slices without misalignment penalties.
- Compression mode routes writes through a deterministic product-quantizer. Each embedding is encoded into `[u8; 96]` codes derived from a Blake3-seeded centroid table, and norms are cached for accurate cosine scoring.
- Added a Criterion harness (`cargo bench --bench cold_tier_columnar`) that compares the full-precision and PQ-compressed flows. CSV/JSON reports are emitted under `engram-core/benches/reports/` so perf regressions are visible in git history.

This research-driven approach ensures the columnar storage implementation achieves both theoretical performance gains and practical production benefits through systematic optimization of memory layout, SIMD operations, and query execution patterns.
