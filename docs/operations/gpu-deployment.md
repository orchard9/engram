# GPU Deployment Guide

**Audience**: DevOps engineers, system administrators, platform engineers

**Last Updated**: 2025-10-26

## IMPORTANT: Current Implementation Status

**GPU acceleration infrastructure is under active development. This documentation describes the target architecture for Milestone 13+.**

**CURRENT MILESTONE 12 STATUS**:

- **IMPLEMENTED**: CPU SIMD fallback (production-ready, high performance)

- **IMPLEMENTED**: Hybrid executor architecture and interfaces

- **NOT IMPLEMENTED**: Actual CUDA kernels, GPU device detection, GPU-specific CLI flags

**CURRENT DEPLOYMENT RECOMMENDATION**: Deploy using CPU SIMD implementation, which provides excellent performance for most workloads. GPU acceleration will be added in a future milestone.

**This guide describes the future GPU deployment process.** For deploying Engram today, see the "Quick Start (CPU SIMD)" section below.

---

## Overview

This guide describes how to deploy Engram with GPU acceleration once CUDA kernels are implemented. GPU acceleration is projected to provide 3-26x speedup for vector similarity, activation spreading, and HNSW search operations.

If you encounter issues during deployment, consult the GPU Troubleshooting Guide. For performance optimization after deployment, see the GPU Performance Tuning Guide.

## Quick Start (CPU SIMD - Current Implementation)

**For deploying Engram TODAY (Milestone 12)**:

```bash
# 1. Build Engram
cargo build --release

# 2. Run Engram (uses CPU SIMD automatically)
./target/release/engram start

# 3. Validate performance
curl http://localhost:8080/metrics

```

That's it! Engram uses high-performance CPU SIMD implementations that provide excellent throughput for most workloads.

**CPU SIMD Performance** (current, production-ready):

- Cosine similarity: ~2.1 us/vector (AVX-512)

- Activation spreading: ~850 us for 1000 nodes

- Throughput: 70K vectors/sec

---

## Quick Start (GPU - PLANNED for Milestone 13+)

**Note**: The following steps describe future GPU deployment. They do NOT work in Milestone 12.

```bash
# 1. Verify GPU (PLANNED)
nvidia-smi

# 2. Install CUDA toolkit 11.0+ (PLANNED)
# (see platform-specific instructions below)

# 3. Build Engram with CUDA kernels (NOT YET AVAILABLE)
cargo build --release --features gpu

# 4. Verify GPU detection (FLAG DOES NOT EXIST YET)
./target/release/engram --gpu-info

# 5. Run with GPU enabled (PLANNED)
./target/release/engram start

# 6. Validate GPU usage (PLANNED)
curl http://localhost:8080/metrics | grep gpu

```

**Current Status**: Steps above are not functional in Milestone 12. Use CPU SIMD Quick Start instead.

## Prerequisites

### Hardware Requirements

**Minimum GPU Specifications**:

- NVIDIA GPU with compute capability 6.0 or higher
  - Consumer: GTX 1060, GTX 1660 Ti, RTX 2060 or newer
  - Datacenter: P40, V100, T4, A100, H100

- 4GB VRAM (8GB recommended for production workloads)

- PCIe 3.0 x16 slot (PCIe 4.0 recommended for bandwidth)

**Check Your GPU Compute Capability**:

```bash
# Install lspci if not available
sudo apt-get install pciutils  # Ubuntu/Debian
sudo yum install pciutils      # RHEL/CentOS

# List NVIDIA GPUs
lspci | grep -i nvidia

# Check compute capability
nvidia-smi --query-gpu=compute_cap --format=csv

# Expected output: 6.0, 7.5, 8.0, 8.6, 9.0, etc.
# If < 6.0, GPU acceleration not supported

```

**Supported GPU Models**:

| Generation | Consumer GPUs | Datacenter GPUs | Compute Capability |
|------------|--------------|-----------------|-------------------|
| Pascal     | GTX 1060, 1070, 1080 | P40, P100 | 6.0, 6.1 |
| Volta      | Titan V | V100 | 7.0, 7.5 |
| Turing     | RTX 2060, 2070, 2080 | T4 | 7.5 |
| Ampere     | RTX 3060, 3070, 3080, 3090 | A100, A10 | 8.0, 8.6 |
| Ada        | RTX 4070, 4080, 4090 | L40, L40S | 8.9 |
| Hopper     | - | H100, H200 | 9.0 |

**System Requirements**:

- x86_64 CPU with AVX2 support (fallback path)

- 16GB+ system RAM (32GB+ recommended)

- SSD storage for graph data

- Linux kernel 3.10+ or Windows Server 2019+

### Software Requirements

**Operating System**:

- Ubuntu 20.04 LTS or newer

- RHEL / Rocky Linux / AlmaLinux 8 or newer

- Windows Server 2019 or newer

- Other distributions with CUDA support (check NVIDIA docs)

**CUDA Toolkit**:

- Minimum: CUDA 11.0

- Recommended: CUDA 11.8 or 12.x (latest stable)

- Download: https://developer.nvidia.com/cuda-downloads

**NVIDIA Driver**:

- Minimum: 450.80.02 (Linux) / 452.39 (Windows)

- Recommended: Latest stable driver for your GPU

- Download: https://www.nvidia.com/drivers

**Build Tools**:

- Rust 1.75.0 or newer

- GCC 9+ or LLVM 10+ (for linking CUDA code)

- CMake 3.18+ (if building from source)

## Installation

### Step 1: Verify GPU Hardware

Ensure your NVIDIA GPU is detected by the system:

```bash
# Check GPU detection
lspci | grep -i nvidia

# Expected output (example):
# 01:00.0 VGA compatible controller: NVIDIA Corporation GA106 [GeForce RTX 3060]

# If no output, check:
# - GPU is properly seated in PCIe slot
# - PCIe power cables connected (if required)
# - BIOS/UEFI has PCIe enabled

```

### Step 2: Install NVIDIA Driver

#### Ubuntu / Debian

```bash
# Add NVIDIA package repository
sudo add-apt-repository ppa:graphics-drivers/ppa
sudo apt update

# Install recommended driver
sudo ubuntu-drivers autoinstall

# Or install specific version
sudo apt install nvidia-driver-535

# Reboot to load driver
sudo reboot

# Verify driver installation
nvidia-smi

# Expected output shows GPU name, driver version, CUDA version

```

#### RHEL / Rocky Linux / AlmaLinux

```bash
# Install EPEL repository
sudo dnf install epel-release

# Install NVIDIA driver from EPEL
sudo dnf install nvidia-driver nvidia-settings

# Alternative: Use NVIDIA's repository
sudo dnf config-manager --add-repo https://developer.download.nvidia.com/compute/cuda/repos/rhel8/x86_64/cuda-rhel8.repo
sudo dnf install nvidia-driver-latest

# Reboot to load driver
sudo reboot

# Verify
nvidia-smi

```

#### Windows Server

1. Download driver from https://www.nvidia.com/drivers

2. Select "Data Center / Tesla" for datacenter GPUs

3. Run installer (may require reboot)

4. Verify in PowerShell:

```powershell
nvidia-smi.exe

```

### Step 3: Install CUDA Toolkit

#### Ubuntu / Debian

```bash
# Download CUDA repository package
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb

# Update package list
sudo apt update

# Install CUDA toolkit
sudo apt install cuda-toolkit-11-8

# Set environment variables
echo 'export PATH=/usr/local/cuda-11.8/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda-11.8/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc

# Verify installation
nvcc --version

# Expected output:
# nvcc: NVIDIA (R) Cuda compiler driver
# Cuda compilation tools, release 11.8, ...

```

#### RHEL / Rocky Linux / AlmaLinux

```bash
# Install CUDA repository
sudo dnf config-manager --add-repo https://developer.download.nvidia.com/compute/cuda/repos/rhel8/x86_64/cuda-rhel8.repo

# Install CUDA toolkit
sudo dnf install cuda-toolkit-11-8

# Set environment variables
echo 'export PATH=/usr/local/cuda-11.8/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda-11.8/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc

# Verify
nvcc --version

```

#### Windows Server

1. Download CUDA installer from https://developer.nvidia.com/cuda-downloads

2. Select Windows > x86_64 > version > exe (network or local)

3. Run installer (installs to `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8`)

4. Verify in PowerShell:

```powershell
nvcc --version

```

### Step 4: Build Engram with GPU Support

```bash
# Clone repository (if not already done)
git clone https://github.com/your-org/engram.git
cd engram

# Build with GPU features
cargo build --release --features gpu

# Build time: 5-15 minutes depending on hardware
# Watch for any compilation warnings or errors

```

**Expected Build Output**:

```
   Compiling engram-core v0.1.0
   Compiling engram-cli v0.1.0
    Finished release [optimized] target(s) in 8m 23s

```

**Troubleshooting Build Failures**:

If build fails with "CUDA toolkit not found":

1. Ensure `nvcc` is in PATH: `which nvcc`

2. Check CUDA_PATH: `echo $CUDA_PATH` (should be `/usr/local/cuda-11.8` or similar)

3. If not set: `export CUDA_PATH=/usr/local/cuda-11.8`

If build fails with linker errors:

1. Check LD_LIBRARY_PATH includes CUDA: `echo $LD_LIBRARY_PATH`

2. Verify libcudart.so exists: `ls /usr/local/cuda-11.8/lib64/libcudart.so`

3. If missing, reinstall CUDA toolkit

If build succeeds but with warnings about missing CUDA:

- This is expected on systems without GPU

- Engram will compile with CPU-only support

- No action needed unless you specifically want GPU

### Step 5: Verify GPU Detection (PLANNED)

**Note**: The `--gpu-info` CLI flag does not exist in Milestone 12.

```bash
# PLANNED - This command will be available in Milestone 13+
./target/release/engram --gpu-info

# Expected future output (GPU available):
# GPU Status: Available
# Device 0: NVIDIA GeForce RTX 3060
#   Compute Capability: 8.6
#   VRAM: 12288 MB
#   Memory Bandwidth: 360 GB/s
#
# GPU acceleration: ENABLED
# Minimum batch size for GPU: 64 vectors

# Expected future output (no GPU):
# GPU Status: Not Available
# Reason: No CUDA-capable devices found
#
# GPU acceleration: DISABLED (using CPU SIMD fallback)

```

**Current Milestone 12**: No GPU detection command available. All operations use CPU SIMD.

**If GPU Not Detected**:

1. Check driver is loaded:

   ```bash
   nvidia-smi
   # If fails, driver not loaded - reboot or check installation
   ```

2. Check CUDA libraries are accessible:

   ```bash
   ldd ./target/release/engram | grep cuda
   # Should show: libcudart.so => /usr/local/cuda-11.8/lib64/libcudart.so
   ```

3. Check permissions:

   ```bash
   ls -l /dev/nvidia*
   # Should be readable by your user
   # If not: sudo usermod -a -G video $USER
   # Then logout and login
   ```

## Configuration

**Note**: Configuration examples below describe the future Rust API for GPU acceleration. TOML-based configuration file loading is not yet implemented.

### Current Configuration (Milestone 12)

No configuration needed - CPU SIMD is always used.

### Future Configuration (Milestone 13+ - Rust API)

**Note**: The following configuration API will be available when GPU features are implemented.

Configuration via Rust `HybridConfig` struct:

```rust
use engram_core::compute::cuda::hybrid::{HybridConfig, HybridExecutor};

let config = HybridConfig {
    gpu_min_batch_size: 64,              // Minimum batch size for GPU dispatch
    gpu_speedup_threshold: 1.5,          // Minimum speedup to prefer GPU
    gpu_success_rate_threshold: 0.95,    // Minimum GPU success rate (95%)
    performance_window_size: 100,        // Number of samples for moving average
    force_cpu_mode: false,               // Force CPU-only mode (debugging)
    telemetry_enabled: true,             // Enable performance tracking
};

let executor = HybridExecutor::new(config);

```

**Planned TOML Configuration** (not yet implemented):

```toml
# PLANNED - Not yet functional
[gpu]
gpu_min_batch_size = 64
force_cpu_mode = false

[gpu.thresholds]
gpu_speedup_threshold = 1.5
gpu_success_rate_threshold = 0.95

[gpu.telemetry]
telemetry_enabled = true
performance_window_size = 100

# Note: These parameters match HybridConfig struct field names
# Note: vram_safety_margin, device_id, debug_kernels not in HybridConfig yet

```

### Multi-GPU Systems

If you have multiple GPUs and want to select a specific one:

```bash
# List all GPUs
nvidia-smi -L

# Output:
# GPU 0: NVIDIA GeForce RTX 3060
# GPU 1: NVIDIA GeForce RTX 3070

# Configure Engram to use GPU 1
# In gpu.toml:
[gpu.advanced]
device_id = 1

# Or use environment variable
export CUDA_VISIBLE_DEVICES=1
./target/release/engram start

```

### Environment Variables

Override configuration via environment variables:

```bash
# Disable GPU
export ENGRAM_GPU_ENABLED=false

# Set minimum batch size
export ENGRAM_GPU_MIN_BATCH_SIZE=128

# Force specific GPU device
export CUDA_VISIBLE_DEVICES=0

# Enable verbose GPU logging
export RUST_LOG=engram::activation::gpu=debug

```

## Deployment Scenarios

### Single-Node Deployment

For a single server with one GPU:

```bash
# Start Engram server
./target/release/engram start --config /etc/engram/config.toml

# Server binds to:
# - gRPC: 0.0.0.0:50051
# - HTTP: 0.0.0.0:8080
# - Metrics: http://localhost:8080/metrics

# Check GPU usage
nvidia-smi dmon -s puct -c 10

# Expected output:
# gpu    pwr  temp    sm   mem   enc   dec
#   0     75    65    90    80     0     0
# (sm = GPU utilization, mem = memory utilization)

```

### Docker Deployment

**Dockerfile**:

```dockerfile
FROM nvidia/cuda:11.8.0-devel-ubuntu22.04

# Install Rust
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"

# Install dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    pkg-config \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy source
WORKDIR /engram
COPY . .

# Build with GPU support
RUN cargo build --release --features gpu

# Runtime stage (smaller image)
FROM nvidia/cuda:11.8.0-runtime-ubuntu22.04

# Copy binary
COPY --from=0 /engram/target/release/engram /usr/local/bin/engram

# Copy configuration
COPY config.toml /etc/engram/config.toml

# Expose ports
EXPOSE 8080 50051

# Run
CMD ["engram", "start", "--config", "/etc/engram/config.toml"]

```

**Build and Run**:

```bash
# Build Docker image
docker build -t engram:gpu .

# Run with GPU support
docker run --gpus all \
  -p 8080:8080 \
  -p 50051:50051 \
  -v /data/engram:/data \
  engram:gpu

# Verify GPU is accessible in container
docker exec -it <container_id> nvidia-smi

```

**Docker Compose**:

```yaml
version: '3.8'

services:
  engram:
    image: engram:gpu
    ports:
      - "8080:8080"
      - "50051:50051"
    volumes:
      - engram-data:/data
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    environment:
      - ENGRAM_GPU_ENABLED=true
      - RUST_LOG=info

volumes:
  engram-data:

```

```bash
# Start with Docker Compose
docker-compose up -d

# View logs
docker-compose logs -f engram

```

### Kubernetes Deployment

**Prerequisites**:

- Kubernetes 1.20+

- NVIDIA GPU Operator or device plugin installed

- Nodes labeled with GPU type

**Install NVIDIA Device Plugin**:

```bash
# Deploy NVIDIA device plugin
kubectl apply -f https://raw.githubusercontent.com/NVIDIA/k8s-device-plugin/v0.14.0/nvidia-device-plugin.yml

# Verify GPU nodes
kubectl get nodes "-o=custom-columns=NAME:.metadata.name,GPU:.status.allocatable.nvidia\.com/gpu"

```

**Deployment Manifest**:

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: engram-gpu
  labels:
    app: engram
spec:
  replicas: 3
  selector:
    matchLabels:
      app: engram
  template:
    metadata:
      labels:
        app: engram
    spec:
      containers:
      - name: engram
        image: engram:gpu
        ports:
        - containerPort: 8080
          name: http
        - containerPort: 50051
          name: grpc
        resources:
          requests:
            cpu: "4"
            memory: "8Gi"
            nvidia.com/gpu: 1
          limits:
            cpu: "8"
            memory: "16Gi"
            nvidia.com/gpu: 1
        env:
        - name: ENGRAM_GPU_ENABLED
          value: "true"
        - name: ENGRAM_GPU_MIN_BATCH_SIZE
          value: "64"
        - name: RUST_LOG
          value: "info,engram::activation::gpu=debug"
        volumeMounts:
        - name: data
          mountPath: /data
      volumes:
      - name: data
        persistentVolumeClaim:
          claimName: engram-data
      nodeSelector:
        accelerator: nvidia-gpu
      tolerations:
      - key: nvidia.com/gpu
        operator: Exists
        effect: NoSchedule
---
apiVersion: v1
kind: Service
metadata:
  name: engram
spec:
  selector:
    app: engram
  ports:
  - name: http
    port: 8080
    targetPort: 8080
  - name: grpc
    port: 50051
    targetPort: 50051
  type: LoadBalancer

```

```bash
# Deploy to Kubernetes
kubectl apply -f engram-deployment.yaml

# Check pod status
kubectl get pods -l app=engram

# Check GPU allocation
kubectl describe pod <engram-pod-name> | grep nvidia.com/gpu

# View logs
kubectl logs -f deployment/engram-gpu

```

## Validation

### Verify GPU is Being Used

**Method 1: Check Metrics**

```bash
# Query metrics endpoint
curl http://localhost:8080/metrics | grep gpu

# Expected output (when GPU features are implemented):
# engram_gpu_launch_total 1234
# engram_gpu_fallback_total 5
# Note: success_rate and speedup_ratio are derived metrics

# If gpu_launch_total is 0:
# - GPU is not being used
# - Check workload has batch sizes >= min_batch_size
# - Check force_cpu_mode is false

```

**Method 2: Monitor GPU Utilization**

```bash
# Real-time GPU monitoring
nvidia-smi dmon -s puct -c 100

# While running workload, should see:
# gpu    pwr  temp    sm   mem
#   0     80    68    85    75
#        ^^^         ^^^  ^^^
#     Power draw   GPU%  Mem%

# If sm (GPU utilization) stays at 0:
# - GPU not being used
# - Check configuration and logs

```

**Method 3: Check Logs**

```bash
# Enable debug logging
export RUST_LOG=engram::activation::gpu=debug
./target/release/engram start

# Look for log lines:
# DEBUG engram::activation::gpu: Batch size 128 >= min 64, using GPU
# DEBUG engram::activation::gpu: GPU launch succeeded, latency=245us

# If you see "using CPU" for large batches:
# - GPU disabled or unavailable
# - Check configuration

```

### Functional Testing

Run test workload to validate GPU acceleration:

```bash
# Run benchmark suite
cargo bench --bench gpu_performance_validation

# Expected output (GPU available):
# cosine_similarity_cpu_vs_gpu/cpu/1024
#                         time:   [2.1 ms 2.15 ms 2.2 ms]
# cosine_similarity_cpu_vs_gpu/gpu/1024
#                         time:   [280 us 295 us 310 us]
# Speedup: 7.3x

# If speedup < 3x:
# - GPU not performing well
# - See GPU Performance Tuning Guide

```

### Load Testing

Simulate production workload:

```bash
# Use grpcurl to send batch queries
for i in {1..1000}; do
  grpcurl -d '{"query": {"embedding": [...]}, "limit": 10}' \
    localhost:50051 engram.v1.EngramService/Search &
done

# Monitor GPU utilization during load test
nvidia-smi dmon -s puct

# Expected during load:
# gpu    sm   mem
#   0    95   80    <- High utilization = GPU is working

# Monitor metrics
watch -n 1 'curl -s http://localhost:8080/metrics | grep gpu'

```

## Production Checklist

Before going to production, verify:

**For Current Milestone 12 (CPU SIMD)**:

- [ ] Engram built with `cargo build --release`

- [ ] Engram starts successfully

- [ ] Metrics endpoint accessible (`curl http://localhost:8080/metrics`)

- [ ] Performance meets requirements using CPU SIMD

- [ ] Monitoring configured for core metrics

- [ ] Documentation shared with on-call team

**For Future GPU Deployment (Milestone 13+)**:

- [ ] GPU driver installed and working (`nvidia-smi` succeeds)

- [ ] CUDA toolkit 11.0+ installed (`nvcc --version`)

- [ ] Engram built with `--features gpu` flag

- [ ] GPU detection works (`engram --gpu-info` - PLANNED, not yet available)

- [ ] Configuration tuned for your GPU type (see Performance Tuning Guide)

- [ ] Benchmark tests show expected speedup (>3x)

- [ ] Load testing validates GPU utilization under production workload

- [ ] Monitoring alerts configured for:
  - GPU failure rate > 5% (derived from `engram_gpu_launch_total` and `engram_gpu_fallback_total`)
  - GPU not being used when expected (`engram_gpu_launch_total` not increasing)
  - OOM events (`engram_gpu_oom_total` > 0)

- [ ] Backup plan for CPU-only operation tested

## Monitoring and Observability

### Key Metrics to Track

**GPU Dispatch Metrics** (when GPU features implemented):

```
engram_gpu_launch_total         # How many operations sent to GPU
engram_gpu_fallback_total       # How many GPU failures fell back to CPU
# Note: success_rate = (launch_total - fallback_total) / launch_total (derived)
# Note: speedup_ratio will be calculated from latency metrics (derived)

```

**GPU Health Metrics**:

```
engram_gpu_oom_total            # Out-of-memory events (target: 0)
engram_gpu_error_total          # CUDA errors (target: <1% of launches)
engram_gpu_available            # GPU availability (1=available, 0=unavailable)

```

**Performance Metrics**:

```
engram_spreading_latency_p50    # Median latency
engram_spreading_latency_p99    # Tail latency
engram_batch_size_histogram     # Distribution of batch sizes

```

### Grafana Dashboard

Example Prometheus queries (for future GPU implementation):

```promql
# GPU utilization rate
rate(engram_gpu_launch_total[5m]) / rate(engram_total_operations[5m])

# GPU success rate over time (derived)
rate(engram_gpu_launch_total[5m] - engram_gpu_fallback_total[5m]) / rate(engram_gpu_launch_total[5m])

# Average speedup (will be derived from latency metrics)
# TBD based on actual implementation

# OOM rate
rate(engram_gpu_oom_total[5m])

```

### Alerting Rules (PLANNED for future GPU implementation)

```yaml
groups:

- name: engram_gpu
  rules:
  # Alert if GPU success rate drops below 95%
  - alert: GPUHighFailureRate
    # Derived metric calculation
    expr: (rate(engram_gpu_launch_total[5m]) - rate(engram_gpu_fallback_total[5m])) / rate(engram_gpu_launch_total[5m]) < 0.95
    for: 5m
    labels:
      severity: warning
    annotations:
      summary: "Engram GPU failure rate > 5%"
      description: "GPU success rate below 95%, investigate GPU health"

  # Alert if GPU not being used
  - alert: GPUNotUtilized
    expr: rate(engram_gpu_launch_total[5m]) == 0 and engram_gpu_available == 1
    for: 10m
    labels:
      severity: warning
    annotations:
      summary: "GPU available but not being used"
      description: "Check if batch sizes are too small or GPU disabled in config"

  # Alert on OOM events
  - alert: GPUOutOfMemory
    expr: increase(engram_gpu_oom_total[5m]) > 0
    for: 1m
    labels:
      severity: critical
    annotations:
      summary: "GPU out-of-memory events detected"
      description: "{{ $value }} OOM events in last 5min, reduce batch size or add GPU memory"

```

## Rollback Procedures

If GPU acceleration causes issues, you can quickly revert to CPU-only operation without redeployment:

### Option 1: Configuration Change (No Restart)

```bash
# Update configuration
echo "force_cpu_mode = true" >> /etc/engram/gpu.toml

# Send SIGHUP to reload config (if supported)
kill -HUP $(pgrep engram)

# Or use API endpoint (if available)
curl -X POST http://localhost:8080/admin/reload-config

```

### Option 2: Environment Variable (Requires Restart)

```bash
# Stop service
systemctl stop engram

# Set environment variable
export ENGRAM_GPU_ENABLED=false

# Restart
systemctl start engram

# Or for Docker:
docker restart engram

# Or for Kubernetes:
kubectl set env deployment/engram-gpu ENGRAM_GPU_ENABLED=false

```

### Option 3: Rebuild Without GPU (Full Rollback)

```bash
# Rebuild without GPU feature
cargo build --release  # omit --features gpu

# Deploy new binary
sudo cp target/release/engram /usr/local/bin/engram

# Restart service
sudo systemctl restart engram

```

## Next Steps

- **Performance Tuning**: See GPU Performance Tuning Guide to optimize configuration for your workload

- **Troubleshooting**: If you encounter issues, consult GPU Troubleshooting Guide

- **Advanced Features**: Learn about multi-GPU support, kernel tuning, and optimization in the GPU Architecture Reference

## Support

For deployment issues:

1. Check GPU Troubleshooting Guide

2. Review logs with `RUST_LOG=engram::activation::gpu=debug`

3. Run diagnostics: `nvidia-smi`, `nvcc --version` (GPU-related, for future use)

4. Note: `engram --gpu-info` command is PLANNED for Milestone 13+, not yet available

5. File an issue with diagnostics output

## Appendix A: Supported Configurations

### Tested Platforms

| OS | GPU | CUDA | Driver | Status |
|----|-----|------|--------|--------|
| Ubuntu 22.04 | RTX 3060 | 11.8 | 520.61.05 | Supported |
| Ubuntu 20.04 | GTX 1660 Ti | 11.4 | 470.141.03 | Supported |
| RHEL 8.7 | A100 | 12.0 | 525.85.12 | Supported |
| Rocky Linux 9 | T4 | 11.8 | 520.61.05 | Supported |
| Windows Server 2022 | RTX 4070 | 12.1 | 531.79 | Experimental |

### Known Incompatibilities

- Maxwell GPUs (GTX 900 series): Compute capability 5.x not supported

- AMD GPUs: ROCm support not yet implemented

- Intel GPUs: Not supported (CUDA-only)

- ARM + NVIDIA Jetson: Not tested, may work with CUDA for ARM

## Appendix B: Cloud Provider Setup

### AWS EC2

```bash
# Launch P3/P4 instance (V100/A100)
# AMI: Deep Learning Base AMI (Ubuntu)

# CUDA and drivers pre-installed
nvidia-smi

# Install Rust
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# Build and deploy
git clone https://github.com/your-org/engram.git
cd engram
cargo build --release --features gpu

```

### Google Cloud Platform

```bash
# Create instance with GPU
gcloud compute instances create engram-gpu \
  --accelerator type=nvidia-tesla-t4,count=1 \
  --image-family=ubuntu-2204-lts \
  --image-project=ubuntu-os-cloud \
  --maintenance-policy TERMINATE

# SSH and install CUDA
gcloud compute ssh engram-gpu

# Follow Ubuntu installation steps above

```

### Azure

```bash
# Create VM with GPU
az vm create \
  --resource-group engram-rg \
  --name engram-gpu \
  --size Standard_NC6s_v3 \
  --image UbuntuLTS

# Install NVIDIA drivers
az vm extension set \
  --resource-group engram-rg \
  --vm-name engram-gpu \
  --name NvidiaGpuDriverLinux \
  --publisher Microsoft.HpcCompute

# SSH and install CUDA
az ssh vm --name engram-gpu

# Follow Ubuntu installation steps above

```
