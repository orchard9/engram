# Task 004: Columnar Vector Storage with SIMD Optimization

## Status: Pending
## Priority: P1 - Performance Critical
## Estimated Effort: 3 days
## Dependencies: Task 001 (three-tier storage), Milestone-1/Task-001 (SIMD operations)

## Objective
Implement columnar storage layout for vector embeddings in the cold tier, optimizing for SIMD operations and batch processing with memory-mapped files.

## Current State Analysis
- **Existing**: SIMD vector operations from milestone-1/task-001
- **Existing**: Basic memory-mapped storage from milestone-1/task-003
- **Missing**: Columnar layout for vectors
- **Missing**: SIMD-optimized batch operations on columns
- **Missing**: Efficient transposition between row and column formats

## Technical Specification

### 1. Columnar Vector Layout

```rust
// engram-core/src/storage/columnar.rs

use memmap2::{MmapMut, MmapOptions};
use std::fs::File;

/// Columnar storage for 768-dimensional vectors
pub struct ColumnarVectorStorage {
    /// Each dimension stored as a separate column
    columns: Vec<ColumnStore>,
    
    /// Row count tracking
    num_vectors: AtomicUsize,
    
    /// Metadata per row
    metadata: Vec<VectorMetadata>,
    
    /// Column statistics for optimization
    column_stats: Vec<ColumnStatistics>,
}

/// Single column storage with SIMD alignment
#[repr(align(64))]
pub struct ColumnStore {
    /// Memory-mapped data file
    mmap: MmapMut,
    
    /// Current size in elements
    size: usize,
    
    /// Capacity in elements
    capacity: usize,
    
    /// Statistics for this column
    stats: ColumnStatistics,
}

#[derive(Debug, Clone)]
struct ColumnStatistics {
    min: f32,
    max: f32,
    mean: f32,
    variance: f32,
    sparsity: f32, // Percentage of zeros
}

impl ColumnarVectorStorage {
    /// Create new columnar storage
    pub fn create(path: &Path, initial_capacity: usize) -> Result<Self> {
        let mut columns = Vec::with_capacity(768);
        
        for dim in 0..768 {
            let column_path = path.join(format!("dim_{:03}.col", dim));
            let column = ColumnStore::create(&column_path, initial_capacity)?;
            columns.push(column);
        }
        
        Ok(Self {
            columns,
            num_vectors: AtomicUsize::new(0),
            metadata: Vec::with_capacity(initial_capacity),
            column_stats: vec![ColumnStatistics::default(); 768],
        })
    }
    
    /// Append vector in columnar format
    pub fn append_vector(&mut self, vector: &[f32; 768]) -> Result<usize> {
        let row_id = self.num_vectors.fetch_add(1, Ordering::SeqCst);
        
        // Write each dimension to its column
        for (dim, &value) in vector.iter().enumerate() {
            self.columns[dim].append(value)?;
            self.column_stats[dim].update(value);
        }
        
        Ok(row_id)
    }
    
    /// Batch append with transposition
    pub fn append_batch(&mut self, vectors: &[[f32; 768]]) -> Result<Vec<usize>> {
        let start_row = self.num_vectors.load(Ordering::SeqCst);
        let mut row_ids = Vec::with_capacity(vectors.len());
        
        // Transpose for cache-efficient column writes
        for dim in 0..768 {
            let column_data: Vec<f32> = vectors.iter()
                .map(|v| v[dim])
                .collect();
            
            self.columns[dim].append_batch(&column_data)?;
        }
        
        for i in 0..vectors.len() {
            row_ids.push(start_row + i);
        }
        
        self.num_vectors.fetch_add(vectors.len(), Ordering::SeqCst);
        Ok(row_ids)
    }
}
```

### 2. SIMD-Optimized Column Operations

```rust
// engram-core/src/storage/columnar_ops.rs

use std::arch::x86_64::*;

impl ColumnarVectorStorage {
    /// SIMD-optimized similarity search across columns
    #[target_feature(enable = "avx2,fma")]
    pub unsafe fn similarity_search_simd(
        &self,
        query: &[f32; 768],
        k: usize,
    ) -> Vec<(usize, f32)> {
        let num_vectors = self.num_vectors.load(Ordering::SeqCst);
        let mut scores = vec![0.0f32; num_vectors];
        let mut norms_a = vec![0.0f32; num_vectors];
        let mut norms_b = vec![0.0f32; num_vectors];
        
        // Process columns in chunks for cache efficiency
        const COLUMN_CHUNK: usize = 8;
        
        for col_chunk in (0..768).step_by(COLUMN_CHUNK) {
            for col in col_chunk..col_chunk.min(col_chunk + COLUMN_CHUNK).min(768) {
                let query_val = query[col];
                let column_data = self.columns[col].as_slice();
                
                // Skip sparse columns for efficiency
                if self.column_stats[col].sparsity > 0.9 && query_val.abs() < 1e-6 {
                    continue;
                }
                
                // SIMD processing of column
                self.process_column_simd(
                    column_data,
                    query_val,
                    &mut scores,
                    &mut norms_a,
                    &mut norms_b,
                );
            }
        }
        
        // Compute final cosine similarities
        let mut results: Vec<(usize, f32)> = (0..num_vectors)
            .map(|i| {
                let similarity = scores[i] / (norms_a[i].sqrt() * norms_b[i].sqrt() + 1e-8);
                (i, similarity)
            })
            .collect();
            
        // Partial sort for top-k
        results.select_nth_unstable_by(k.min(results.len()), |a, b| {
            b.1.partial_cmp(&a.1).unwrap()
        });
        
        results.truncate(k);
        results
    }
    
    #[target_feature(enable = "avx2,fma")]
    unsafe fn process_column_simd(
        &self,
        column: &[f32],
        query_val: f32,
        scores: &mut [f32],
        norms_a: &mut [f32],
        norms_b: &mut [f32],
    ) {
        let query_vec = _mm256_set1_ps(query_val);
        
        // Process 8 elements at a time
        let chunks = column.chunks_exact(8);
        let remainder = chunks.remainder();
        
        for (i, chunk) in chunks.enumerate() {
            let base_idx = i * 8;
            
            // Load column values
            let col_vec = _mm256_loadu_ps(chunk.as_ptr());
            
            // Dot product accumulation
            let dot = _mm256_fmadd_ps(col_vec, query_vec, _mm256_loadu_ps(scores.as_ptr().add(base_idx)));
            _mm256_storeu_ps(scores.as_mut_ptr().add(base_idx), dot);
            
            // Norm accumulation
            let norm_a = _mm256_fmadd_ps(col_vec, col_vec, _mm256_loadu_ps(norms_a.as_ptr().add(base_idx)));
            _mm256_storeu_ps(norms_a.as_mut_ptr().add(base_idx), norm_a);
            
            let norm_b = _mm256_fmadd_ps(query_vec, query_vec, _mm256_loadu_ps(norms_b.as_ptr().add(base_idx)));
            _mm256_storeu_ps(norms_b.as_mut_ptr().add(base_idx), norm_b);
        }
        
        // Handle remainder with scalar code
        let base_idx = (column.len() / 8) * 8;
        for (j, &val) in remainder.iter().enumerate() {
            let idx = base_idx + j;
            scores[idx] += val * query_val;
            norms_a[idx] += val * val;
            norms_b[idx] += query_val * query_val;
        }
    }
}
```

### 3. Row-Column Transposition

```rust
// engram-core/src/storage/transposition.rs

pub struct VectorTransposer {
    /// Temporary buffer for transposition
    buffer: Vec<f32>,
    
    /// Block size for cache-optimal transposition
    block_size: usize,
}

impl VectorTransposer {
    /// Transpose from row-major to column-major
    pub fn rows_to_columns(
        &mut self,
        rows: &[[f32; 768]],
    ) -> Vec<Vec<f32>> {
        let num_rows = rows.len();
        let mut columns = vec![Vec::with_capacity(num_rows); 768];
        
        // Cache-blocked transposition
        const BLOCK: usize = 32;
        
        for row_block in (0..num_rows).step_by(BLOCK) {
            for col_block in (0..768).step_by(BLOCK) {
                // Transpose block
                for row in row_block..row_block.min(row_block + BLOCK).min(num_rows) {
                    for col in col_block..col_block.min(col_block + BLOCK).min(768) {
                        columns[col].push(rows[row][col]);
                    }
                }
            }
        }
        
        columns
    }
    
    /// Transpose from column-major to row-major
    pub fn columns_to_rows(
        &mut self,
        columns: &[Vec<f32>],
        num_rows: usize,
    ) -> Vec<[f32; 768]> {
        let mut rows = vec![[0.0f32; 768]; num_rows];
        
        // Parallel transposition for large datasets
        rows.par_iter_mut()
            .enumerate()
            .for_each(|(row_idx, row)| {
                for (col_idx, column) in columns.iter().enumerate() {
                    row[col_idx] = column[row_idx];
                }
            });
            
        rows
    }
}
```

### 4. Compression for Cold Storage

```rust
// engram-core/src/storage/compression.rs

use zstd::stream::encode_all;

pub struct ColumnCompressor {
    /// Compression level (1-22)
    level: i32,
    
    /// Dictionary for better compression
    dictionary: Option<Vec<u8>>,
}

impl ColumnCompressor {
    /// Compress column with optimal settings for f32 data
    pub fn compress_column(&self, column: &[f32]) -> Result<Vec<u8>> {
        // Quantize to 16-bit for better compression
        let quantized = self.quantize_f32_to_f16(column);
        
        // Apply compression
        let compressed = if let Some(dict) = &self.dictionary {
            zstd::encode_all_with_dictionary(&quantized, self.level, dict)?
        } else {
            encode_all(&quantized[..], self.level)?
        };
        
        Ok(compressed)
    }
    
    fn quantize_f32_to_f16(&self, values: &[f32]) -> Vec<u8> {
        let mut output = Vec::with_capacity(values.len() * 2);
        
        for &val in values {
            let f16_val = half::f16::from_f32(val);
            output.extend_from_slice(&f16_val.to_bits().to_le_bytes());
        }
        
        output
    }
}
```

## Integration Points

### Modify ColdTier (storage/cold_tier.rs from Task 001)
```rust
// Update around line 20:
pub struct ColdTier {
    /// Columnar storage for SIMD operations
    columnar: ColumnarVectorStorage,
    
    /// Transposition utility
    transposer: VectorTransposer,
    
    /// Compression for inactive data
    compressor: ColumnCompressor,
    
    /// Index for ID to row mapping
    id_index: HashMap<String, usize>,
}

// Add methods around line 50:
impl StorageTier for ColdTier {
    fn store(&self, id: &str, vector: &[f32; 768], metadata: StorageMetadata) -> Result<()> {
        let row_id = self.columnar.append_vector(vector)?;
        self.id_index.insert(id.to_string(), row_id);
        Ok(())
    }
    
    fn retrieve_batch(&self, ids: &[String]) -> Result<Vec<(Vec<f32>, Confidence)>> {
        let row_ids: Vec<usize> = ids.iter()
            .filter_map(|id| self.id_index.get(id))
            .copied()
            .collect();
            
        self.columnar.retrieve_rows(&row_ids)
    }
}
```

## Testing Strategy

### Unit Tests
```rust
#[test]
fn test_columnar_append_retrieve() {
    let storage = ColumnarVectorStorage::create_temp().unwrap();
    
    let vector = [0.5f32; 768];
    let row_id = storage.append_vector(&vector).unwrap();
    
    let retrieved = storage.retrieve_row(row_id).unwrap();
    assert_eq!(retrieved, vector);
}

#[test]
fn test_simd_column_search() {
    let mut storage = ColumnarVectorStorage::create_temp().unwrap();
    
    // Add test vectors
    for i in 0..1000 {
        let mut vec = [0.0f32; 768];
        vec[i % 768] = 1.0;
        storage.append_vector(&vec).unwrap();
    }
    
    let query = [0.5f32; 768];
    let results = unsafe { storage.similarity_search_simd(&query, 10) };
    
    assert_eq!(results.len(), 10);
}
```

### Performance Benchmarks
```rust
#[bench]
fn bench_columnar_batch_append(b: &mut Bencher) {
    let mut storage = ColumnarVectorStorage::create_temp().unwrap();
    let vectors: Vec<[f32; 768]> = (0..1000)
        .map(|i| [i as f32 / 1000.0; 768])
        .collect();
    
    b.iter(|| {
        storage.append_batch(&vectors).unwrap();
    });
}

#[bench]
fn bench_simd_similarity_search(b: &mut Bencher) {
    let storage = create_test_columnar_storage(10_000);
    let query = [0.7f32; 768];
    
    b.iter(|| {
        unsafe { storage.similarity_search_simd(&query, 100) }
    });
}
```

## Acceptance Criteria
- [ ] Columnar storage reduces memory bandwidth by >40%
- [ ] SIMD column operations achieve >8x speedup over scalar
- [ ] Transposition overhead <10% of total operation time
- [ ] Compression achieves >50% size reduction for cold data
- [ ] Batch operations process >100K vectors/second
- [ ] Memory-mapped columns support datasets >100GB

## Performance Targets
- Column append: <10μs per vector
- Batch append: <1ms for 1000 vectors
- SIMD similarity search: <100μs for 10K vectors
- Transposition: <5ms for 10K vectors
- Compression ratio: >2:1 for typical embeddings

## Risk Mitigation
- Fallback to row storage if columnar overhead too high
- Adaptive compression based on data characteristics
- Memory-mapped file size limits handled gracefully
- SIMD validation against scalar implementation