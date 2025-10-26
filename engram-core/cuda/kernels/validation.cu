#include "../common.cuh"
#include <stdio.h>

// Minimal validation kernel to verify CUDA toolchain integration
// This kernel performs a trivial computation to prove that:
// 1. CUDA code compiles correctly
// 2. Kernels can be launched from Rust
// 3. Data can be transferred between host and device
// 4. Results are computed correctly on GPU

__global__ void hello_kernel(int* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        // Simple computation: output[i] = i * 2
        output[idx] = idx * 2;
    }
}

// Test kernel with 768-dimensional vector operations
__global__ void vector_test_kernel(const float* input, float* output, int count) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < count) {
        // Simple vector operation: accumulate first 8 elements
        float sum = 0.0f;
        const float* vec = input + idx * EMBEDDING_DIM;

        #pragma unroll
        for (int i = 0; i < 8; i++) {
            sum += vec[i];
        }

        output[idx] = sum;
    }
}

// External C interface for Rust FFI
extern "C" {

// Validate that CUDA environment is functional
// Returns:
//   0: Success - GPU is available and functional
//  -1: cudaGetDeviceCount failed
//  -2: No CUDA devices found
//  -3: cudaMalloc failed (GPU memory allocation error)
int cuda_validate_environment() {
    int device_count;
    cudaError_t err = cudaGetDeviceCount(&device_count);

    if (err != cudaSuccess) {
        fprintf(stderr, "cudaGetDeviceCount failed: %s\n",
                cudaGetErrorString(err));
        return -1;
    }

    if (device_count == 0) {
        fprintf(stderr, "No CUDA devices found\n");
        return -2;
    }

    // Try allocating and freeing a small amount of memory
    void* test_ptr;
    err = cudaMalloc(&test_ptr, 1024);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMalloc test failed: %s\n",
                cudaGetErrorString(err));
        return -3;
    }

    cudaFree(test_ptr);
    return 0;
}

// Test kernel launch with trivial computation
// Arguments:
//   h_output: Host array to receive results (must be allocated by caller)
//   size: Number of elements in output array
// Returns:
//   0: Success - kernel launched and completed correctly
//  -1: cudaMalloc failed
//  -2: cudaMemcpy failed
int cuda_test_kernel_launch(int* h_output, int size) {
    int* d_output;

    // Allocate device memory
    cudaError_t err = cudaMalloc((void**)&d_output, size * sizeof(int));
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed: %s\n", cudaGetErrorString(err));
        return -1;
    }

    // Launch kernel with appropriate grid dimensions
    int block_size = DEFAULT_BLOCK_SIZE;
    int grid_size;
    compute_grid_dimensions(size, block_size, grid_size);

    hello_kernel<<<grid_size, block_size>>>(d_output, size);

    // Wait for kernel to complete
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize failed: %s\n",
                cudaGetErrorString(err));
        cudaFree(d_output);
        return -2;
    }

    // Copy results back to host
    err = cudaMemcpy(h_output, d_output, size * sizeof(int),
                     cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_output);
        return -2;
    }

    cudaFree(d_output);
    return 0;
}

// Test vector operations kernel
// Arguments:
//   h_input: Host array of 768-dimensional vectors (count * 768 floats)
//   h_output: Host array to receive results (count floats)
//   count: Number of vectors
// Returns:
//   0: Success
//  -1: cudaMalloc failed
//  -2: cudaMemcpy failed
int cuda_test_vector_kernel(const float* h_input, float* h_output, int count) {
    float *d_input, *d_output;
    size_t input_size = count * EMBEDDING_DIM * sizeof(float);
    size_t output_size = count * sizeof(float);

    // Allocate device memory
    cudaError_t err = cudaMalloc((void**)&d_input, input_size);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMalloc input failed: %s\n",
                cudaGetErrorString(err));
        return -1;
    }

    err = cudaMalloc((void**)&d_output, output_size);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMalloc output failed: %s\n",
                cudaGetErrorString(err));
        cudaFree(d_input);
        return -1;
    }

    // Copy input to device
    err = cudaMemcpy(d_input, h_input, input_size, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy input failed: %s\n",
                cudaGetErrorString(err));
        cudaFree(d_input);
        cudaFree(d_output);
        return -2;
    }

    // Launch kernel
    int block_size = DEFAULT_BLOCK_SIZE;
    int grid_size;
    compute_grid_dimensions(count, block_size, grid_size);

    vector_test_kernel<<<grid_size, block_size>>>(d_input, d_output, count);

    // Wait for kernel to complete
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize failed: %s\n",
                cudaGetErrorString(err));
        cudaFree(d_input);
        cudaFree(d_output);
        return -2;
    }

    // Copy results back
    err = cudaMemcpy(h_output, d_output, output_size, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy output failed: %s\n",
                cudaGetErrorString(err));
        cudaFree(d_input);
        cudaFree(d_output);
        return -2;
    }

    cudaFree(d_input);
    cudaFree(d_output);
    return 0;
}

} // extern "C"
