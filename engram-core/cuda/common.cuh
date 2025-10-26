#ifndef ENGRAM_CUDA_COMMON_H
#define ENGRAM_CUDA_COMMON_H

#include <cuda_runtime.h>
#include <stdint.h>

// Common CUDA kernel definitions and utilities for Engram GPU acceleration

// Error checking macro for CUDA API calls
#define CUDA_CHECK(call)                                                      \
    do {                                                                      \
        cudaError_t err = call;                                              \
        if (err != cudaSuccess) {                                            \
            fprintf(stderr, "CUDA error at %s:%d: %s\n",                     \
                    __FILE__, __LINE__, cudaGetErrorString(err));            \
            return -1;                                                        \
        }                                                                     \
    } while (0)

// Common constants
#define EMBEDDING_DIM 768
#define WARP_SIZE 32

// Common kernel launch parameters
#define DEFAULT_BLOCK_SIZE 256
#define MAX_BLOCKS_PER_GRID 65535

// Compute grid dimensions for a given problem size
inline void compute_grid_dimensions(int problem_size, int block_size,
                                    int& grid_size) {
    grid_size = (problem_size + block_size - 1) / block_size;
    if (grid_size > MAX_BLOCKS_PER_GRID) {
        grid_size = MAX_BLOCKS_PER_GRID;
    }
}

// Warp-level reduction primitives
__device__ inline float warp_reduce_sum(float val) {
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

// Block-level reduction using shared memory
template <typename T>
__device__ inline T block_reduce_sum(T val, T* shared) {
    int lane = threadIdx.x % WARP_SIZE;
    int wid = threadIdx.x / WARP_SIZE;

    // Warp-level reduction
    val = warp_reduce_sum(val);

    // Write warp results to shared memory
    if (lane == 0) {
        shared[wid] = val;
    }
    __syncthreads();

    // Final reduction of warp results
    if (wid == 0) {
        val = (threadIdx.x < blockDim.x / WARP_SIZE) ? shared[lane] : 0;
        val = warp_reduce_sum(val);
    }

    return val;
}

// Cache-optimized vector operations
namespace engram {
namespace cuda {

// Fast dot product for 768-dimensional vectors
__device__ inline float dot_product_768(const float* __restrict__ a,
                                       const float* __restrict__ b) {
    float sum = 0.0f;
    #pragma unroll 8
    for (int i = 0; i < EMBEDDING_DIM; i++) {
        sum += a[i] * b[i];
    }
    return sum;
}

// Fast L2 norm for 768-dimensional vectors
__device__ inline float l2_norm_768(const float* __restrict__ vec) {
    float sum = 0.0f;
    #pragma unroll 8
    for (int i = 0; i < EMBEDDING_DIM; i++) {
        float val = vec[i];
        sum += val * val;
    }
    return sqrtf(sum);
}

// Cosine similarity between two 768-dimensional vectors
__device__ inline float cosine_similarity_768(const float* __restrict__ a,
                                              const float* __restrict__ b) {
    float dot = dot_product_768(a, b);
    float norm_a = l2_norm_768(a);
    float norm_b = l2_norm_768(b);
    return dot / (norm_a * norm_b + 1e-8f);
}

} // namespace cuda
} // namespace engram

#endif // ENGRAM_CUDA_COMMON_H
