//! CUDA GPU acceleration for Engram memory operations
//!
//! This module provides GPU-accelerated implementations of core memory operations:
//! - Vector similarity computation (cosine similarity, dot product)
//! - Spreading activation matrix operations
//! - HNSW neighbor scoring and distance calculations
//!
//! The module gracefully handles systems without CUDA:
//! - If CUDA toolkit is unavailable at build time, stub functions are generated
//! - Runtime checks detect GPU availability and fall back to CPU implementations
//! - All GPU functions return Result types for proper error handling
//!
//! # Architecture
//!
//! - `ffi`: Low-level CUDA FFI bindings with safe wrappers
//! - `cosine_similarity`: GPU-accelerated batch cosine similarity computation
//! - Future modules: `spreading`, `hnsw` for domain-specific kernels
//!
//! # Usage
//!
//! ```rust,ignore
//! use engram_core::compute::cuda;
//!
//! // Check if GPU is available
//! if cuda::is_available() {
//!     println!("GPU acceleration enabled");
//! } else {
//!     println!("Falling back to CPU");
//! }
//! ```

pub mod ffi;

#[cfg(cuda_available)]
pub mod cosine_similarity;

#[cfg(cuda_available)]
pub mod unified_memory;

#[cfg(cuda_available)]
pub mod spreading;

#[cfg(cuda_available)]
pub mod hnsw;

#[cfg(cuda_available)]
pub mod memory_pressure;

// Performance tracker is always available (tracks CPU performance too)
pub mod performance_tracker;

// Hybrid executor is always available (provides CPU-only fallback)
pub mod hybrid;

use std::sync::OnceLock;

/// GPU availability status
static GPU_AVAILABLE: OnceLock<bool> = OnceLock::new();

/// Check if CUDA GPU is available and functional
///
/// This function performs a one-time check on first call and caches the result.
/// It verifies:
/// - CUDA toolkit was available at build time
/// - GPU device is present at runtime
/// - Basic memory allocation works
///
/// Returns true if GPU can be used, false otherwise.
#[must_use]
pub fn is_available() -> bool {
    *GPU_AVAILABLE.get_or_init(|| {
        #[cfg(cuda_available)]
        {
            // CUDA was available at build time, check runtime availability
            match ffi::validate_environment() {
                Ok(()) => {
                    tracing::info!("CUDA GPU acceleration enabled");
                    true
                }
                Err(e) => {
                    tracing::warn!("CUDA GPU not available at runtime: {}", e);
                    false
                }
            }
        }
        #[cfg(not(cuda_available))]
        {
            // CUDA was not available at build time
            tracing::debug!("CUDA support not compiled in (toolkit was not found during build)");
            false
        }
    })
}

/// Get information about available GPU devices
///
/// Returns a vector of device descriptions, or an empty vector if no GPUs available.
#[must_use]
#[allow(clippy::missing_const_for_fn)]
pub fn get_device_info() -> Vec<DeviceInfo> {
    #[cfg(cuda_available)]
    {
        match ffi::get_device_count() {
            Ok(count) => (0..count)
                .filter_map(|i| {
                    ffi::get_device_properties(i)
                        .ok()
                        .map(|props| DeviceInfo::from_properties(i, props))
                })
                .collect(),
            Err(_) => Vec::new(),
        }
    }
    #[cfg(not(cuda_available))]
    {
        Vec::new()
    }
}

/// Information about a CUDA device
#[derive(Debug, Clone)]
pub struct DeviceInfo {
    /// CUDA device index
    pub device_id: i32,
    /// Device name
    pub name: String,
    /// Compute capability as (major, minor) version tuple
    pub compute_capability: (i32, i32),
    /// Total global memory in gigabytes
    pub total_memory_gb: f64,
    /// Number of streaming multiprocessors
    pub multiprocessor_count: i32,
}

impl DeviceInfo {
    #[cfg(cuda_available)]
    fn from_properties(device_id: i32, props: ffi::CudaDeviceProperties) -> Self {
        Self {
            device_id,
            name: props.device_name(),
            compute_capability: props.compute_capability(),
            total_memory_gb: props.total_global_mem as f64 / (1024.0 * 1024.0 * 1024.0),
            multiprocessor_count: props.multi_processor_count,
        }
    }
}

impl std::fmt::Display for DeviceInfo {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "GPU {}: {} (SM {}.{}, {:.1} GB, {} SMs)",
            self.device_id,
            self.name,
            self.compute_capability.0,
            self.compute_capability.1,
            self.total_memory_gb,
            self.multiprocessor_count
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gpu_availability() {
        let available = is_available();
        println!("GPU available: {available}");

        if available {
            let devices = get_device_info();
            println!("Found {} GPU device(s)", devices.len());
            for device in devices {
                println!("  {device}");
            }
        }
    }

    #[test]
    #[cfg(cuda_available)]
    fn test_validation_kernel() {
        if !is_available() {
            println!("GPU not available, skipping kernel test");
            return;
        }

        let size = 1024;
        let mut output = vec![0i32; size];

        ffi::test_kernel_launch(&mut output).expect("Kernel launch failed");

        // Verify computation: output[i] should be i * 2
        for (i, &val) in output.iter().enumerate() {
            assert_eq!(val, (i * 2) as i32, "Kernel computation error at index {i}");
        }
    }

    #[test]
    #[cfg(cuda_available)]
    fn test_vector_kernel() {
        if !is_available() {
            println!("GPU not available, skipping vector kernel test");
            return;
        }

        let count = 128;
        let input: Vec<[f32; 768]> = vec![[1.0; 768]; count];
        let mut output = vec![0.0f32; count];

        ffi::test_vector_kernel(&input, &mut output).expect("Vector kernel failed");

        // Kernel sums first 8 elements, so expected result is 8.0
        for (i, &val) in output.iter().enumerate() {
            assert!(
                (val - 8.0).abs() < 1e-5,
                "Vector kernel error at index {i}: expected 8.0, got {val}"
            );
        }
    }
}
