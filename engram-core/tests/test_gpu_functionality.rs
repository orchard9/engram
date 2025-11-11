//! GPU functionality tests

#![cfg(feature = "gpu")]

use engram_core::compute::cuda;

#[test]
fn test_gpu_basic_functionality() {
    println!("\n=== GPU Functionality Test ===");

    // Check if GPU is available
    if !cuda::is_available() {
        println!("GPU not available, skipping test");
        return;
    }

    println!("✓ GPU is available");

    // Get device info
    match cuda::ffi::get_device_count() {
        Ok(count) => {
            println!("✓ Found {} CUDA device(s)", count);
            assert!(count > 0, "Expected at least one CUDA device");

            // Query first device
            if let Ok(props) = cuda::ffi::get_device_properties(0) {
                println!("✓ Device 0: {}", props.device_name());
                println!("  Compute capability: {}.{}", props.major, props.minor);
                println!(
                    "  Memory: {} GB",
                    props.total_global_mem / (1024 * 1024 * 1024)
                );
            }
        }
        Err(e) => panic!("Failed to get device count: {}", e),
    }

    // Test memory allocation
    match cuda::ffi::malloc_managed(1024 * 1024) {
        Ok(ptr) => {
            println!("✓ Successfully allocated 1MB of unified memory");
            let _ = cuda::ffi::free(ptr);
            println!("✓ Successfully freed memory");
        }
        Err(e) => panic!("Failed to allocate memory: {}", e),
    }

    // Test kernel launch
    let mut output = vec![0i32; 1024];
    match cuda::ffi::test_kernel_launch(&mut output) {
        Ok(()) => {
            println!("✓ Successfully launched test kernel");
            // Verify some results - kernel computes output[i] = i * 2
            assert_eq!(output[0], 0, "Expected output[0] = 0");
            assert_eq!(output[512], 1024, "Expected output[512] = 1024");
            println!("✓ Kernel computation verified");
        }
        Err(e) => panic!("Failed to launch kernel: {}", e),
    }

    // Test cosine similarity (only when CUDA is actually available)
    #[cfg(cuda_available)]
    {
        use engram_core::compute::VectorOps;
        use engram_core::compute::cuda::cosine_similarity::GpuCosineSimilarity;

        let gpu_ops = GpuCosineSimilarity::new();
        let query = [1.0f32; 768];
        let targets = vec![[1.0f32; 768]; 128];

        let results = gpu_ops.cosine_similarity_batch_768(&query, &targets);
        println!("✓ Successfully computed batch cosine similarity");
        assert_eq!(results.len(), 128);

        // Check accuracy
        for (i, &sim) in results.iter().enumerate() {
            assert!(
                (sim - 1.0).abs() < 1e-6,
                "Expected similarity ~1.0, got {} at index {}",
                sim,
                i
            );
        }
        println!("✓ GPU computation accuracy verified");

        // Check that GPU was actually used for large batch
        assert!(
            gpu_ops.gpu_call_count() > 0,
            "Expected GPU to be used for batch of 128 vectors"
        );
        println!("✓ GPU call count: {}", gpu_ops.gpu_call_count());
    }

    #[cfg(not(cuda_available))]
    {
        println!("! Cosine similarity test skipped (CUDA not available at compile time)");
    }

    println!("\n=== All GPU tests passed ===\n");
}
