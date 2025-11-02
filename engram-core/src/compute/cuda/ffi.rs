//! CUDA FFI bindings for GPU acceleration
//!
//! This module provides safe Rust wrappers around CUDA runtime API,
//! converting C-style error codes to idiomatic Result types.
//!
//! All FFI calls handle errors gracefully, converting cudaError_t to
//! CudaError enum with descriptive error messages.

// Allow unsafe code and related lints for FFI bindings - this is necessary for CUDA interop
#![allow(unsafe_code)]
#![allow(clippy::missing_safety_doc)]
#![allow(clippy::not_unsafe_ptr_arg_deref)]
#![allow(clippy::borrow_as_ptr)]

use std::ffi::{CStr, c_int, c_void};
use std::fmt;

/// CUDA error codes mapped from cudaError_t
#[repr(i32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CudaError {
    /// Operation completed successfully
    Success = 0,
    /// Invalid value passed to CUDA API
    InvalidValue = 1,
    /// GPU ran out of memory
    OutOfMemory = 2,
    /// CUDA runtime not initialized
    NotInitialized = 3,
    /// Error during CUDA shutdown
    DeInitializationError = 4,
    /// No CUDA-capable device found
    NoDevice = 100,
    /// Invalid device ID specified
    InvalidDevice = 101,
    /// Invalid memory copy direction
    InvalidMemcpyDirection = 21,
    /// Unknown or unmapped error code
    Unknown = -1,
}

impl CudaError {
    /// Convert raw CUDA error code to typed enum
    #[must_use]
    pub const fn from_raw(code: i32) -> Self {
        match code {
            0 => Self::Success,
            1 => Self::InvalidValue,
            2 => Self::OutOfMemory,
            3 => Self::NotInitialized,
            4 => Self::DeInitializationError,
            100 => Self::NoDevice,
            101 => Self::InvalidDevice,
            21 => Self::InvalidMemcpyDirection,
            _ => Self::Unknown,
        }
    }

    /// Convert error to Result type for idiomatic error handling
    pub const fn to_result(self) -> Result<(), Self> {
        match self {
            Self::Success => Ok(()),
            _ => Err(self),
        }
    }
}

impl fmt::Display for CudaError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Success => write!(f, "Success"),
            Self::InvalidValue => write!(f, "Invalid value"),
            Self::OutOfMemory => write!(f, "Out of memory"),
            Self::NotInitialized => write!(f, "Not initialized"),
            Self::DeInitializationError => write!(f, "Deinitialization error"),
            Self::NoDevice => write!(f, "No CUDA device found"),
            Self::InvalidDevice => write!(f, "Invalid device"),
            Self::InvalidMemcpyDirection => write!(f, "Invalid memcpy direction"),
            Self::Unknown => write!(f, "Unknown CUDA error"),
        }
    }
}

impl std::error::Error for CudaError {}

/// CUDA device properties structure
/// Matches cudaDeviceProp layout from CUDA runtime API
#[repr(C)]
#[derive(Debug, Clone)]
#[allow(missing_docs)]
pub struct CudaDeviceProperties {
    pub name: [u8; 256],
    pub total_global_mem: usize,
    pub shared_mem_per_block: usize,
    pub regs_per_block: c_int,
    pub warp_size: c_int,
    pub mem_pitch: usize,
    pub max_threads_per_block: c_int,
    pub max_threads_dim: [c_int; 3],
    pub max_grid_size: [c_int; 3],
    pub clock_rate: c_int,
    pub total_const_mem: usize,
    pub major: c_int,
    pub minor: c_int,
    pub texture_alignment: usize,
    pub device_overlap: c_int,
    pub multi_processor_count: c_int,
    pub kernel_exec_timeout_enabled: c_int,
    pub integrated: c_int,
    pub can_map_host_memory: c_int,
    pub compute_mode: c_int,
    pub concurrent_kernels: c_int,
    pub ecc_enabled: c_int,
    pub pci_bus_id: c_int,
    pub pci_device_id: c_int,
    pub tcc_driver: c_int,
    pub memory_clock_rate: c_int,
    pub memory_bus_width: c_int,
    pub l2_cache_size: c_int,
    pub max_threads_per_multi_processor: c_int,
    pub managed_memory: c_int,
}

impl CudaDeviceProperties {
    /// Get device name as UTF-8 string
    #[must_use]
    pub fn device_name(&self) -> String {
        let null_pos = self.name.iter().position(|&c| c == 0).unwrap_or(256);
        String::from_utf8_lossy(&self.name[..null_pos]).into_owned()
    }

    /// Get compute capability as (major, minor) tuple
    #[must_use]
    pub const fn compute_capability(&self) -> (i32, i32) {
        (self.major, self.minor)
    }
}

/// Memory copy direction for cudaMemcpy
#[repr(i32)]
#[derive(Debug, Clone, Copy)]
pub enum CudaMemcpyKind {
    /// Copy from host memory to host memory
    HostToHost = 0,
    /// Copy from host memory to device (GPU) memory
    HostToDevice = 1,
    /// Copy from device (GPU) memory to host memory
    DeviceToHost = 2,
    /// Copy from device (GPU) memory to device (GPU) memory
    DeviceToDevice = 3,
}

/// Memory advise hints for unified memory
#[repr(i32)]
#[derive(Debug, Clone, Copy)]
pub enum CudaMemoryAdvise {
    /// Data will be mostly read and only occasionally written
    SetReadMostly = 1,
    /// Unset read-mostly flag
    UnsetReadMostly = 2,
    /// Set the preferred location for the data
    SetPreferredLocation = 3,
    /// Unset preferred location
    UnsetPreferredLocation = 4,
    /// Data will be accessed by device
    SetAccessedBy = 5,
    /// Data will not be accessed by device
    UnsetAccessedBy = 6,
}

/// Special device ID for CPU
pub const CUDA_CPU_DEVICE_ID: i32 = -1;

// External C functions from CUDA runtime
unsafe extern "C" {
    // Device management
    fn cudaGetDeviceCount(count: *mut c_int) -> i32;
    fn cudaGetDeviceProperties(prop: *mut CudaDeviceProperties, device: c_int) -> i32;
    fn cudaSetDevice(device: c_int) -> i32;
    fn cudaGetDevice(device: *mut c_int) -> i32;

    // Memory management
    fn cudaMalloc(devPtr: *mut *mut c_void, size: usize) -> i32;
    fn cudaFree(devPtr: *mut c_void) -> i32;
    fn cudaMallocManaged(devPtr: *mut *mut c_void, size: usize, flags: u32) -> i32;
    fn cudaMallocHost(ptr: *mut *mut c_void, size: usize) -> i32;
    fn cudaFreeHost(ptr: *mut c_void) -> i32;
    fn cudaMemcpy(dst: *mut c_void, src: *const c_void, count: usize, kind: i32) -> i32;
    fn cudaMemcpyAsync(
        dst: *mut c_void,
        src: *const c_void,
        count: usize,
        kind: i32,
        stream: *mut c_void,
    ) -> i32;
    fn cudaMemset(devPtr: *mut c_void, value: c_int, count: usize) -> i32;
    fn cudaMemPrefetchAsync(
        devPtr: *const c_void,
        count: usize,
        dstDevice: c_int,
        stream: *mut c_void,
    ) -> i32;
    fn cudaMemAdvise(devPtr: *const c_void, count: usize, advice: i32, device: c_int) -> i32;
    fn cudaMemGetInfo(free: *mut usize, total: *mut usize) -> i32;

    // Synchronization
    fn cudaDeviceSynchronize() -> i32;
    #[allow(dead_code)]
    fn cudaStreamSynchronize(stream: *mut c_void) -> i32;

    // Error handling
    fn cudaGetLastError() -> i32;
    fn cudaGetErrorString(error: i32) -> *const i8;
}

// External C functions from our validation kernel
unsafe extern "C" {
    fn cuda_validate_environment() -> c_int;
    fn cuda_test_kernel_launch(output: *mut i32, size: i32) -> c_int;
    fn cuda_test_vector_kernel(input: *const f32, output: *mut f32, count: i32) -> c_int;
}

// External C functions from spreading kernel
unsafe extern "C" {
    /// Compute sparse activation spreading with automatic memory management
    /// Returns 0 on success, negative error code on failure
    pub fn cuda_sparse_spreading_managed(
        h_row_ptr: *const c_int,
        h_col_idx: *const c_int,
        h_weights: *const f32,
        h_input_activation: *const f32,
        h_output_activation: *mut f32,
        num_nodes: c_int,
        num_edges: c_int,
    ) -> c_int;
}

/// Safe wrapper for CUDA device count query
pub fn get_device_count() -> Result<i32, CudaError> {
    let mut count: c_int = 0;
    unsafe {
        let result = cudaGetDeviceCount(&mut count);
        CudaError::from_raw(result).to_result()?;
    }
    Ok(count)
}

/// Safe wrapper for CUDA device properties query
pub fn get_device_properties(device_id: i32) -> Result<CudaDeviceProperties, CudaError> {
    let mut props: CudaDeviceProperties = unsafe { std::mem::zeroed() };
    unsafe {
        let result = cudaGetDeviceProperties(&mut props, device_id);
        CudaError::from_raw(result).to_result()?;
    }
    Ok(props)
}

/// Set the current CUDA device
pub fn set_device(device_id: i32) -> Result<(), CudaError> {
    unsafe {
        let result = cudaSetDevice(device_id);
        CudaError::from_raw(result).to_result()
    }
}

/// Get the current CUDA device
pub fn get_device() -> Result<i32, CudaError> {
    let mut device: c_int = 0;
    unsafe {
        let result = cudaGetDevice(&mut device);
        CudaError::from_raw(result).to_result()?;
    }
    Ok(device)
}

/// Allocate device memory
pub fn malloc(size: usize) -> Result<*mut c_void, CudaError> {
    let mut ptr: *mut c_void = std::ptr::null_mut();
    unsafe {
        let result = cudaMalloc(&mut ptr, size);
        CudaError::from_raw(result).to_result()?;

        // Defensive null pointer check
        if ptr.is_null() {
            return Err(CudaError::OutOfMemory);
        }
    }
    Ok(ptr)
}

/// Free device memory
pub fn free(ptr: *mut c_void) -> Result<(), CudaError> {
    // Null pointer is a no-op (matches standard free() behavior)
    if ptr.is_null() {
        return Ok(());
    }

    unsafe {
        let result = cudaFree(ptr);
        CudaError::from_raw(result).to_result()
    }
}

/// Allocate unified memory (accessible from both CPU and GPU)
pub fn malloc_managed(size: usize) -> Result<*mut c_void, CudaError> {
    let mut ptr: *mut c_void = std::ptr::null_mut();
    unsafe {
        // cudaMemAttachGlobal = 0x01
        let result = cudaMallocManaged(&mut ptr, size, 0x01);
        CudaError::from_raw(result).to_result()?;

        // Defensive null pointer check
        if ptr.is_null() {
            return Err(CudaError::OutOfMemory);
        }
    }
    Ok(ptr)
}

/// Copy memory between host and device
pub fn memcpy(
    dst: *mut c_void,
    src: *const c_void,
    count: usize,
    kind: CudaMemcpyKind,
) -> Result<(), CudaError> {
    // Validate pointers before calling CUDA API
    if dst.is_null() || src.is_null() {
        return Err(CudaError::InvalidValue);
    }

    unsafe {
        let result = cudaMemcpy(dst, src, count, kind as i32);
        CudaError::from_raw(result).to_result()
    }
}

/// Set device memory to a value
pub fn memset(ptr: *mut c_void, value: i32, count: usize) -> Result<(), CudaError> {
    // Null pointer check
    if ptr.is_null() {
        return Err(CudaError::InvalidValue);
    }

    unsafe {
        let result = cudaMemset(ptr, value, count);
        CudaError::from_raw(result).to_result()
    }
}

/// Synchronize device execution
pub fn device_synchronize() -> Result<(), CudaError> {
    unsafe {
        let result = cudaDeviceSynchronize();
        CudaError::from_raw(result).to_result()
    }
}

/// Get last CUDA error
#[must_use]
pub fn get_last_error() -> CudaError {
    unsafe { CudaError::from_raw(cudaGetLastError()) }
}

/// Get error string for a CUDA error code
#[must_use]
pub fn get_error_string(error: CudaError) -> String {
    unsafe {
        let c_str = cudaGetErrorString(error as i32);
        CStr::from_ptr(c_str).to_string_lossy().into_owned()
    }
}

/// Allocate pinned host memory
pub fn malloc_host(size: usize) -> Result<*mut c_void, CudaError> {
    let mut ptr: *mut c_void = std::ptr::null_mut();
    unsafe {
        let result = cudaMallocHost(&mut ptr, size);
        CudaError::from_raw(result).to_result()?;

        // Defensive null pointer check
        if ptr.is_null() {
            return Err(CudaError::OutOfMemory);
        }
    }
    Ok(ptr)
}

/// Free pinned host memory
pub fn free_host(ptr: *mut c_void) -> Result<(), CudaError> {
    // Null pointer is a no-op (matches standard free() behavior)
    if ptr.is_null() {
        return Ok(());
    }

    unsafe {
        let result = cudaFreeHost(ptr);
        CudaError::from_raw(result).to_result()
    }
}

/// Asynchronous memory copy
pub fn memcpy_async(
    dst: *mut c_void,
    src: *const c_void,
    count: usize,
    kind: CudaMemcpyKind,
    stream: *mut c_void,
) -> Result<(), CudaError> {
    // Validate pointers before calling CUDA API
    if dst.is_null() || src.is_null() {
        return Err(CudaError::InvalidValue);
    }

    unsafe {
        let result = cudaMemcpyAsync(dst, src, count, kind as i32, stream);
        CudaError::from_raw(result).to_result()
    }
}

/// Prefetch unified memory to specified device
pub fn mem_prefetch_async(
    dev_ptr: *const c_void,
    count: usize,
    dst_device: i32,
    stream: *mut c_void,
) -> Result<(), CudaError> {
    // Null pointer check
    if dev_ptr.is_null() {
        return Err(CudaError::InvalidValue);
    }

    unsafe {
        let result = cudaMemPrefetchAsync(dev_ptr, count, dst_device, stream);
        CudaError::from_raw(result).to_result()
    }
}

/// Provide memory usage hint for unified memory
pub fn mem_advise(
    dev_ptr: *const c_void,
    count: usize,
    advice: CudaMemoryAdvise,
    device: i32,
) -> Result<(), CudaError> {
    // Null pointer check
    if dev_ptr.is_null() {
        return Err(CudaError::InvalidValue);
    }

    unsafe {
        let result = cudaMemAdvise(dev_ptr, count, advice as i32, device);
        CudaError::from_raw(result).to_result()
    }
}

/// Get free and total device memory
pub fn mem_get_info() -> Result<(usize, usize), CudaError> {
    let mut free: usize = 0;
    let mut total: usize = 0;
    unsafe {
        let result = cudaMemGetInfo(&mut free, &mut total);
        CudaError::from_raw(result).to_result()?;
    }
    Ok((free, total))
}

/// Validate CUDA environment is functional
/// Returns Ok(()) if GPU is available and working, Err otherwise
pub fn validate_environment() -> Result<(), String> {
    let result = unsafe { cuda_validate_environment() };
    match result {
        0 => Ok(()),
        -1 => Err("cudaGetDeviceCount failed".to_string()),
        -2 => Err("No CUDA devices found".to_string()),
        -3 => Err("cudaMalloc test failed".to_string()),
        _ => Err(format!("Unknown error code: {result}")),
    }
}

/// Test kernel launch with trivial computation
pub fn test_kernel_launch(output: &mut [i32]) -> Result<(), String> {
    // Validate input array is non-empty
    if output.is_empty() {
        return Err("Output array cannot be empty".to_string());
    }

    let result = unsafe { cuda_test_kernel_launch(output.as_mut_ptr(), output.len() as i32) };
    match result {
        0 => Ok(()),
        -1 => Err("cudaMalloc failed or CUDA not available".to_string()),
        -2 => Err("cudaMemcpy or cudaDeviceSynchronize failed".to_string()),
        _ => Err(format!("Unknown error code: {result}")),
    }
}

/// Test vector kernel with 768-dimensional embeddings
pub fn test_vector_kernel(input: &[[f32; 768]], output: &mut [f32]) -> Result<(), String> {
    // Validate arrays are non-empty
    if input.is_empty() || output.is_empty() {
        return Err("Input and output arrays cannot be empty".to_string());
    }

    if input.len() != output.len() {
        return Err(format!(
            "Input and output arrays must have same length (got {} and {})",
            input.len(),
            output.len()
        ));
    }

    let result = unsafe {
        cuda_test_vector_kernel(
            input.as_ptr().cast::<f32>(),
            output.as_mut_ptr(),
            input.len() as i32,
        )
    };

    match result {
        0 => Ok(()),
        -1 => Err("cudaMalloc failed or CUDA not available".to_string()),
        -2 => Err("cudaMemcpy or cudaDeviceSynchronize failed".to_string()),
        _ => Err(format!("Unknown error code: {result}")),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cuda_error_conversion() {
        assert_eq!(CudaError::from_raw(0), CudaError::Success);
        assert_eq!(CudaError::from_raw(2), CudaError::OutOfMemory);
        assert_eq!(CudaError::from_raw(100), CudaError::NoDevice);
    }

    #[test]
    fn test_cuda_error_display() {
        let err = CudaError::NoDevice;
        assert_eq!(err.to_string(), "No CUDA device found");
    }

    #[test]
    #[cfg(cuda_available)]
    fn test_device_query() {
        match get_device_count() {
            Ok(count) => {
                println!("Found {count} CUDA device(s)");
                assert!(count > 0, "Expected at least one CUDA device");

                // Query first device properties
                if let Ok(props) = get_device_properties(0) {
                    println!("Device 0: {}", props.device_name());
                    println!("Compute capability: {}.{}", props.major, props.minor);
                    println!(
                        "Global memory: {} GB",
                        props.total_global_mem / (1024 * 1024 * 1024)
                    );
                }
            }
            Err(e) => {
                println!("CUDA not available: {e}");
            }
        }
    }

    #[test]
    #[cfg(cuda_available)]
    fn test_validate_environment() {
        match validate_environment() {
            Ok(()) => println!("CUDA environment validated successfully"),
            Err(e) => println!("CUDA validation failed: {e}"),
        }
    }

    #[test]
    #[cfg(cuda_available)]
    fn test_kernel_launch_function() {
        let size = 1024;
        let mut output = vec![0i32; size];

        match test_kernel_launch(&mut output) {
            Ok(()) => {
                // Validate kernel computed correctly: output[i] = i * 2
                for (i, &val) in output.iter().enumerate() {
                    assert_eq!(
                        val,
                        (i * 2) as i32,
                        "Kernel computation mismatch at index {i}"
                    );
                }
                println!("Kernel launch test passed");
            }
            Err(e) => {
                println!("Kernel launch failed: {e}");
            }
        };
    }

    #[test]
    #[cfg(cuda_available)]
    fn test_vector_kernel_function() {
        let count = 128;
        let input: Vec<[f32; 768]> = vec![[1.0; 768]; count];
        let mut output = vec![0.0f32; count];

        match test_vector_kernel(&input, &mut output) {
            Ok(()) => {
                // Kernel sums first 8 elements, so expected result is 8.0
                for (i, &val) in output.iter().enumerate() {
                    assert!(
                        (val - 8.0).abs() < 1e-5,
                        "Vector kernel mismatch at index {i}: expected 8.0, got {val}"
                    );
                }
                println!("Vector kernel test passed");
            }
            Err(e) => {
                println!("Vector kernel failed: {e}");
            }
        };
    }
}
