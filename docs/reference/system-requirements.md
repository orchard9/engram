# System Requirements

## Software Requirements

### Rust Toolchain
- **Minimum Version**: Rust 1.82.0 (required for Edition 2024 support)
- **Recommended**: Latest stable release (1.91.0 or newer)
- **Installation**:
  ```bash
  curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
  rustup update stable
  ```

### System Libraries
- **hwloc**: Hardware locality library for NUMA awareness
  - Ubuntu/Debian: `sudo apt-get install libhwloc-dev`
  - macOS: `brew install hwloc`
  - RHEL/CentOS: `sudo yum install hwloc-devel`

- **libudev** (Linux only): Device management library
  - Ubuntu/Debian: `sudo apt-get install libudev-dev`
  - RHEL/CentOS: `sudo yum install systemd-devel`

- **pkg-config**: Build configuration tool
  - Ubuntu/Debian: `sudo apt-get install pkg-config`
  - macOS: `brew install pkg-config`

### Optional GPU Support
- **CUDA Toolkit**: 11.0 or newer (for GPU acceleration)
  - Compute Capability: 6.0+ (Pascal architecture or newer)
  - Driver Version: 450.80.02 or newer
  - Installation varies by platform - see [NVIDIA CUDA Downloads](https://developer.nvidia.com/cuda-downloads)

## Hardware Requirements

### Minimum Requirements
- **CPU**: x86_64 or ARM64 processor with SSE4.2 or NEON support
- **RAM**: 8GB
- **Storage**: 10GB free space
- **OS**: Linux (Ubuntu 20.04+), macOS 12+, or Windows 10+ with WSL2

### Recommended Production Requirements
- **CPU**: 16+ cores with AVX2 support
- **RAM**: 64GB+ for large memory graphs
- **Storage**: NVMe SSD with 100GB+ free space
- **Network**: 10Gbps for distributed deployments
- **GPU** (optional): NVIDIA GPU with 16GB+ VRAM

### NUMA Considerations
For optimal performance on NUMA systems:
- Enable NUMA awareness in BIOS
- Use `numactl` to bind Engram to specific NUMA nodes
- Configure memory allocation policies in Engram config

## Operating System Support

### Linux
- **Recommended**: Ubuntu 20.04 LTS or newer
- **Supported**:
  - Debian 10+
  - RHEL/CentOS 8+
  - Fedora 34+
  - Arch Linux (latest)

### macOS
- **Minimum**: macOS 12 (Monterey)
- **Recommended**: macOS 14 (Sonoma) or newer
- Note: GPU acceleration not available on macOS

### Windows
- **Minimum**: Windows 10 version 2004
- **Recommended**: Windows 11 with WSL2
- Note: Native Windows support is experimental; WSL2 recommended

## Build Dependencies

### Core Dependencies
All handled automatically by Cargo:
- `tokio`: Async runtime
- `serde`: Serialization
- `dashmap`: Concurrent data structures
- `rayon`: Data parallelism
- `tracing`: Observability

### Feature Flags
- `default`: Core functionality
- `gpu`: CUDA acceleration (requires CUDA toolkit)
- `zig-kernels`: Zig performance kernels (requires Zig 0.13.0+)
- `cognitive_tracing`: Extended cognitive metrics
- `pattern_completion`: Pattern recognition features

## Verification

Check your system meets requirements:

```bash
# Check Rust version
rustc --version  # Should show 1.82.0 or newer

# Check system libraries
pkg-config --exists hwloc && echo "hwloc: OK" || echo "hwloc: MISSING"

# Check GPU support (optional)
nvcc --version 2>/dev/null && echo "CUDA: OK" || echo "CUDA: Not available"

# Build Engram
cargo build --release
```

## Troubleshooting

### Common Issues

1. **"error: failed to parse the `edition` key"**
   - Update Rust: `rustup update stable`
   - Verify version: `rustc --version` (must be 1.82.0+)

2. **"Could not find library hwloc"**
   - Install hwloc development package for your OS
   - Set PKG_CONFIG_PATH if installed in non-standard location

3. **"unable to find library -ludev"** (Linux)
   - Install libudev-dev (Ubuntu/Debian) or systemd-devel (RHEL)

4. **GPU not detected**
   - Verify CUDA installation: `nvidia-smi`
   - Check CUDA_PATH environment variable
   - Ensure nvcc is in PATH

## Performance Tuning

For optimal performance:
1. Build with `--release` flag
2. Enable LTO: Set `CARGO_PROFILE_RELEASE_LTO=true`
3. Use native CPU features: `RUSTFLAGS="-C target-cpu=native"`
4. Configure huge pages for large memory workloads
5. Disable CPU frequency scaling for consistent latency
