#include "../common.cuh"
#include <stdio.h>
#include <math.h>

// High-performance batch cosine similarity kernel for Engram memory retrieval
//
// ARCHITECTURE:
// - Warp-level parallelism: Each warp (32 threads) processes one target vector
// - Each thread computes 24 dimensions (768 / 32 = 24)
// - Warp shuffle reduction (register-to-register, no shared memory)
// - Query vector cached in constant memory (broadcast to all threads)
// - Target vectors loaded with coalesced memory access
//
// PERFORMANCE TARGETS:
// - CPU AVX-512 baseline: ~2.1 µs/vector
// - GPU target: ~0.3 µs/vector (7x speedup)
// - Break-even: 64 vectors (accounting for kernel launch overhead)
//
// NUMERICAL STABILITY:
// - IEEE 754 compliant (no fast-math)
// - Kahan summation for dot product accumulation
// - Handles zero vectors gracefully (returns 0.0)
// - Matches CPU scalar implementation within 1e-6 tolerance

// Constants for query vector in constant memory (broadcasted to all threads)
// Maximum 768-dimensional vector = 3KB (well under 64KB constant memory limit)
__constant__ float d_query[EMBEDDING_DIM];

// Device function: Compute dot product and norm using warp-level reduction
// Each thread computes partial sum over its assigned dimensions
// Returns dot(query, target) and norm(target)^2 for lane 0
__device__ inline void warp_cosine_components(
    const float* __restrict__ target,
    float query_norm_sq,
    float& dot_product,
    float& target_norm_sq
) {
    const int lane_id = threadIdx.x % WARP_SIZE;
    const int dims_per_thread = EMBEDDING_DIM / WARP_SIZE; // 768 / 32 = 24
    const int start_dim = lane_id * dims_per_thread;

    // Kahan summation for numerical stability
    // Reduces floating-point error accumulation in dot product
    float dot_sum = 0.0f;
    float dot_compensation = 0.0f;
    float norm_sum = 0.0f;
    float norm_compensation = 0.0f;

    // Each thread computes 24 multiply-adds
    // Unroll by 4 for ILP (instruction-level parallelism)
    #pragma unroll 4
    for (int i = 0; i < dims_per_thread; i++) {
        const int dim = start_dim + i;
        const float q_val = d_query[dim];
        const float t_val = target[dim];

        // Kahan summation for dot product
        float dot_y = (q_val * t_val) - dot_compensation;
        float dot_t = dot_sum + dot_y;
        dot_compensation = (dot_t - dot_sum) - dot_y;
        dot_sum = dot_t;

        // Kahan summation for target norm
        float norm_y = (t_val * t_val) - norm_compensation;
        float norm_t = norm_sum + norm_y;
        norm_compensation = (norm_t - norm_sum) - norm_y;
        norm_sum = norm_t;
    }

    // Warp shuffle reduction: combine partial sums across warp
    // Uses register-to-register transfers (faster than shared memory)
    // Pattern: threads offset by 16, 8, 4, 2, 1 exchange and sum
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        dot_sum += __shfl_down_sync(0xffffffff, dot_sum, offset);
        norm_sum += __shfl_down_sync(0xffffffff, norm_sum, offset);
    }

    // Only lane 0 has the final reduced values
    dot_product = dot_sum;
    target_norm_sq = norm_sum;
}

// Batch cosine similarity kernel
// Grid: (batch_size + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK blocks
// Block: 256 threads = 8 warps (tuned for memory coalescing)
// Each warp computes similarity for one target vector
//
// Arguments:
//   targets: Device array of target vectors [batch_size][768] (row-major)
//   query_norm_sq: Precomputed ||query||^2 (passed from host)
//   results: Device array for output similarities [batch_size]
//   batch_size: Number of target vectors to compare
__global__ void batch_cosine_similarity_kernel(
    const float* __restrict__ targets,
    float query_norm_sq,
    float* __restrict__ results,
    int batch_size
) {
    // Determine which target vector this warp processes
    const int warp_id = threadIdx.x / WARP_SIZE;
    const int warps_per_block = blockDim.x / WARP_SIZE;
    const int global_warp_id = blockIdx.x * warps_per_block + warp_id;
    const int target_idx = global_warp_id;

    // Bounds check
    if (target_idx >= batch_size) {
        return;
    }

    // Pointer to this warp's target vector
    const float* target = targets + target_idx * EMBEDDING_DIM;

    // Compute dot(query, target) and ||target||^2 using warp reduction
    float dot_product;
    float target_norm_sq;
    warp_cosine_components(target, query_norm_sq, dot_product, target_norm_sq);

    // Only lane 0 of each warp writes the result
    const int lane_id = threadIdx.x % WARP_SIZE;
    if (lane_id == 0) {
        // Handle zero vectors: return 0.0 similarity
        if (query_norm_sq == 0.0f || target_norm_sq == 0.0f) {
            results[target_idx] = 0.0f;
        } else {
            // Cosine similarity = dot(q, t) / (||q|| * ||t||)
            float similarity = dot_product / (sqrtf(query_norm_sq) * sqrtf(target_norm_sq));

            // Clamp to [-1.0, 1.0] for numerical stability
            // (floating-point error can cause slight overflow)
            similarity = fmaxf(-1.0f, fminf(1.0f, similarity));

            results[target_idx] = similarity;
        }
    }
}

// External C interface for Rust FFI
extern "C" {

// Compute batch cosine similarity between query and multiple targets
//
// This function expects:
// 1. Query vector already copied to constant memory via cuda_cosine_set_query
// 2. Targets in device memory (already allocated and copied)
// 3. Results buffer in device memory (already allocated)
//
// Thread configuration:
// - Block size: 256 threads (8 warps)
// - Grid size: (batch_size + 7) / 8 blocks
//
// Arguments:
//   d_targets: Device pointer to target vectors [batch_size][768]
//   query_norm_sq: Precomputed ||query||^2
//   d_results: Device pointer to output buffer [batch_size]
//   batch_size: Number of vectors in batch
//
// Returns:
//   0: Success
//  -1: Invalid batch size (must be > 0)
//  -2: Kernel launch failed
int cuda_cosine_similarity_batch(
    const float* d_targets,
    float query_norm_sq,
    float* d_results,
    int batch_size
) {
    if (batch_size <= 0) {
        fprintf(stderr, "Invalid batch size: %d\n", batch_size);
        return -1;
    }

    // Launch configuration
    const int threads_per_block = 256; // 8 warps
    const int warps_per_block = threads_per_block / WARP_SIZE;
    const int blocks = (batch_size + warps_per_block - 1) / warps_per_block;

    // Launch kernel
    batch_cosine_similarity_kernel<<<blocks, threads_per_block>>>(
        d_targets,
        query_norm_sq,
        d_results,
        batch_size
    );

    // Check for launch errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Kernel launch failed: %s\n", cudaGetErrorString(err));
        return -2;
    }

    return 0;
}

// Copy query vector to constant memory
//
// This must be called before cuda_cosine_similarity_batch.
// Constant memory is cached and broadcasted efficiently to all threads.
//
// Arguments:
//   h_query: Host pointer to query vector [768]
//
// Returns:
//   0: Success
//  -1: cudaMemcpyToSymbol failed
int cuda_cosine_set_query(const float* h_query) {
    cudaError_t err = cudaMemcpyToSymbol(
        d_query,
        h_query,
        EMBEDDING_DIM * sizeof(float),
        0,
        cudaMemcpyHostToDevice
    );

    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMemcpyToSymbol failed: %s\n", cudaGetErrorString(err));
        return -1;
    }

    return 0;
}

// Convenience wrapper: compute batch cosine similarity end-to-end
//
// This function handles all memory management:
// 1. Allocates device memory for targets and results
// 2. Copies targets to device
// 3. Copies query to constant memory
// 4. Launches kernel
// 5. Copies results back to host
// 6. Frees device memory
//
// Use this for simple cases. For multiple batches with same query,
// use the lower-level functions to avoid redundant memory transfers.
//
// Arguments:
//   h_query: Host query vector [768]
//   h_targets: Host target vectors [batch_size][768]
//   query_norm_sq: Precomputed ||query||^2
//   h_results: Host output buffer [batch_size]
//   batch_size: Number of target vectors
//
// Returns:
//   0: Success
//  -1: cudaMalloc failed
//  -2: cudaMemcpy failed
//  -3: Kernel launch failed
//  -4: cudaDeviceSynchronize failed
int cuda_cosine_similarity_batch_managed(
    const float* h_query,
    const float* h_targets,
    float query_norm_sq,
    float* h_results,
    int batch_size
) {
    if (batch_size <= 0) {
        fprintf(stderr, "Invalid batch size: %d\n", batch_size);
        return -1;
    }

    float *d_targets = nullptr;
    float *d_results = nullptr;
    cudaError_t err;

    // Allocate device memory
    size_t targets_size = batch_size * EMBEDDING_DIM * sizeof(float);
    size_t results_size = batch_size * sizeof(float);

    err = cudaMalloc((void**)&d_targets, targets_size);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMalloc targets failed: %s\n", cudaGetErrorString(err));
        return -1;
    }

    err = cudaMalloc((void**)&d_results, results_size);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMalloc results failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_targets);
        return -1;
    }

    // Copy targets to device
    err = cudaMemcpy(d_targets, h_targets, targets_size, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy targets failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_targets);
        cudaFree(d_results);
        return -2;
    }

    // Copy query to constant memory
    int status = cuda_cosine_set_query(h_query);
    if (status != 0) {
        cudaFree(d_targets);
        cudaFree(d_results);
        return status;
    }

    // Launch kernel
    status = cuda_cosine_similarity_batch(d_targets, query_norm_sq, d_results, batch_size);
    if (status != 0) {
        cudaFree(d_targets);
        cudaFree(d_results);
        return -3;
    }

    // Wait for kernel to complete
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_targets);
        cudaFree(d_results);
        return -4;
    }

    // Copy results back to host
    err = cudaMemcpy(h_results, d_results, results_size, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy results failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_targets);
        cudaFree(d_results);
        return -2;
    }

    // Cleanup
    cudaFree(d_targets);
    cudaFree(d_results);

    return 0;
}

} // extern "C"
