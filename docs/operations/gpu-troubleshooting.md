# GPU Troubleshooting Guide

**Audience**: DevOps engineers, SREs, system administrators

**Last Updated**: 2025-10-26

## IMPORTANT: Implementation Status

**This troubleshooting guide describes issues for FUTURE GPU implementation (Milestone 13+).**

**CURRENT MILESTONE 12 STATUS**:
- **IMPLEMENTED**: CPU SIMD fallback (production-ready)
- **NOT IMPLEMENTED**: CUDA kernels, GPU detection, GPU-specific errors

**CURRENT BEHAVIOR**: All operations use CPU SIMD. The troubleshooting steps below will be relevant when CUDA kernels are implemented in Milestone 13+.

**For current troubleshooting**: Use standard Rust error messages and CPU performance profiling. GPU-specific troubleshooting does not apply yet.

---

## Overview

This guide will help diagnose and resolve common issues with GPU-accelerated Engram deployments once CUDA kernels are implemented. It describes CUDA errors, performance problems, and operational issues that may occur with GPU acceleration.

For deployment instructions, see the GPU Deployment Guide. For performance optimization, see the GPU Performance Tuning Guide.

## Quick Diagnostic Checklist

Run through this checklist first when encountering GPU issues:

```bash
# 1. Is the GPU detected by the system?
nvidia-smi
# If fails: Driver issue (see Section: GPU Not Detected)

# 2. Is CUDA toolkit installed?
nvcc --version
# If fails: CUDA installation issue (see Section: CUDA Toolkit Issues)

# 3. Does Engram detect the GPU?
./target/release/engram --gpu-info
# If "Not Available": See Section: Engram Cannot Detect GPU

# 4. Is GPU being used during operation?
curl http://localhost:8080/metrics | grep gpu_launches_total
# If value is 0 or not increasing: See Section: GPU Not Being Used

# 5. Are there GPU errors in logs?
journalctl -u engram | grep -i cuda
# If CUDA errors: See Section: CUDA Error Codes

# 6. What is the GPU utilization?
nvidia-smi dmon -s puct -c 10
# If sm% is 0: GPU not active (see Section: Low GPU Utilization)
# If mem% is 100%: OOM issue (see Section: Out of Memory)
```

## Common Issues

### GPU Not Detected by System

**Symptom**: `nvidia-smi` command fails or shows no GPUs

**Diagnostic Output**:
```
NVIDIA-SMI has failed because it couldn't communicate with the NVIDIA driver.
Make sure that the latest NVIDIA driver is installed and running.
```

**Causes and Solutions**:

#### Cause 1: Driver Not Installed

```bash
# Check if driver is installed
dpkg -l | grep nvidia-driver  # Ubuntu/Debian
rpm -qa | grep nvidia-driver   # RHEL/CentOS

# If no output, install driver
sudo ubuntu-drivers autoinstall  # Ubuntu
sudo dnf install nvidia-driver   # RHEL

sudo reboot
```

#### Cause 2: Driver Not Loaded

```bash
# Check if driver module is loaded
lsmod | grep nvidia

# If no output, manually load driver
sudo modprobe nvidia

# If modprobe fails, check dmesg for errors
dmesg | grep -i nvidia

# Common error: "version magic mismatch"
# Solution: Reinstall driver matching your kernel
uname -r  # Check kernel version
sudo apt install linux-headers-$(uname -r)  # Install matching headers
sudo apt install --reinstall nvidia-driver-535
```

#### Cause 3: Secure Boot Enabled

```bash
# Check if Secure Boot is enabled
mokutil --sb-state

# If "SecureBoot enabled":
# Option 1: Disable Secure Boot in BIOS/UEFI (recommended)
# Option 2: Sign NVIDIA driver modules (complex, see NVIDIA docs)
```

#### Cause 4: GPU Not Detected by Hardware

```bash
# Check if GPU appears in PCI devices
lspci | grep -i nvidia

# If no output:
# 1. Check GPU is properly seated in PCIe slot
# 2. Check PCIe power cables connected (8-pin/6-pin)
# 3. Check BIOS/UEFI settings:
#    - PCIe enabled
#    - "Above 4G Decoding" enabled (if available)
#    - Resize BAR enabled (for newer GPUs)
# 4. Try GPU in different PCIe slot
# 5. Test GPU in different machine (hardware failure?)
```

### CUDA Toolkit Issues

**Symptom**: `nvcc --version` fails or wrong version

**Diagnostic Output**:
```
bash: nvcc: command not found
```

**Solutions**:

#### Solution 1: CUDA Not Installed

```bash
# Ubuntu/Debian
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt update
sudo apt install cuda-toolkit-11-8

# RHEL/Rocky/AlmaLinux
sudo dnf config-manager --add-repo https://developer.download.nvidia.com/compute/cuda/repos/rhel8/x86_64/cuda-rhel8.repo
sudo dnf install cuda-toolkit-11-8
```

#### Solution 2: CUDA Not in PATH

```bash
# Check where CUDA is installed
ls -la /usr/local/cuda*

# Typical locations:
# /usr/local/cuda-11.8
# /usr/local/cuda (symlink to versioned directory)

# Add to PATH
echo 'export PATH=/usr/local/cuda-11.8/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda-11.8/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc

# Verify
nvcc --version
which nvcc
```

#### Solution 3: Wrong CUDA Version

```bash
# Check installed CUDA versions
ls -la /usr/local/ | grep cuda

# Output might show multiple versions:
# cuda-11.4/
# cuda-11.8/
# cuda -> cuda-11.4  (symlink)

# Change symlink to desired version
sudo rm /usr/local/cuda
sudo ln -s /usr/local/cuda-11.8 /usr/local/cuda

# Update PATH to use generic /usr/local/cuda
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

# Verify
nvcc --version
```

### Engram Cannot Detect GPU

**Symptom**: `engram --gpu-info` shows "GPU Status: Not Available"

**Diagnostic Output**:
```
GPU Status: Not Available
Reason: CUDA initialization failed
```

**Diagnostic Steps**:

#### Step 1: Check Driver/CUDA Compatibility

```bash
# Check NVIDIA driver version
nvidia-smi | grep "Driver Version"

# Check CUDA runtime version
nvidia-smi | grep "CUDA Version"

# Driver version must support CUDA toolkit version
# CUDA 11.x requires driver >= 450.80.02
# CUDA 12.x requires driver >= 525.60.13

# If driver too old, update:
sudo apt install nvidia-driver-535  # Ubuntu
sudo dnf install nvidia-driver-latest  # RHEL
```

#### Step 2: Check Library Linking

```bash
# Check if Engram binary links to CUDA libraries
ldd ./target/release/engram | grep cuda

# Expected output:
# libcudart.so.11.0 => /usr/local/cuda-11.8/lib64/libcudart.so.11.0

# If "not found":
export LD_LIBRARY_PATH=/usr/local/cuda-11.8/lib64:$LD_LIBRARY_PATH
ldconfig  # Update library cache (may need sudo)

# Try again
./target/release/engram --gpu-info
```

#### Step 3: Check Permissions

```bash
# Check if current user can access GPU
ls -l /dev/nvidia*

# Should show:
# crw-rw-rw- 1 root root 195, 0 /dev/nvidia0
#           ^^^
#           World-readable

# If not:
# Option 1: Add user to video group
sudo usermod -a -G video $USER
# Logout and login for changes to take effect

# Option 2: Fix udev rules
echo 'KERNEL=="nvidia*", MODE="0666"' | sudo tee /etc/udev/rules.d/70-nvidia.rules
sudo udevadm control --reload-rules
sudo udevadm trigger

# Reboot
sudo reboot
```

#### Step 4: Check for Conflicting Processes

```bash
# Check if another process is using GPU exclusively
nvidia-smi

# Look at "Processes" section at bottom
# If you see process with mode "E" (exclusive):
# That process has exclusive GPU access

# Solution: Stop exclusive process or configure GPU for shared mode
# Set compute mode to default (shared)
sudo nvidia-smi -c 0
```

#### Step 5: Verify CUDA Initialization

```bash
# Enable verbose CUDA logging
export CUDA_LAUNCH_BLOCKING=1
export CUDA_VISIBLE_DEVICES=0

# Run with debug logging
RUST_LOG=engram::activation::gpu=debug ./target/release/engram --gpu-info

# Look for specific error messages in output
```

### GPU Not Being Used

**Symptom**: Metrics show `engram_gpu_launches_total` is 0 or not increasing

**Diagnostic Steps**:

#### Step 1: Check Configuration

```bash
# Verify GPU is enabled in configuration
cat /etc/engram/gpu.toml | grep enabled

# Should show:
# enabled = true

# Check if force_cpu_mode is enabled
cat /etc/engram/gpu.toml | grep force_cpu_mode

# Should show:
# force_cpu_mode = false

# If force_cpu_mode = true, GPU is intentionally disabled
```

#### Step 2: Check Batch Sizes

```bash
# Check minimum batch size threshold
cat /etc/engram/gpu.toml | grep min_batch_size

# If min_batch_size = 128, batches smaller than 128 use CPU

# Check actual batch sizes in workload
curl http://localhost:8080/metrics | grep batch_size_histogram

# If all batches are smaller than min_batch_size, GPU won't be used
# Solution: Lower min_batch_size or increase batch sizes in application
```

#### Step 3: Check GPU Success Rate

```bash
# Check if GPU is failing and falling back to CPU
curl http://localhost:8080/metrics | grep gpu_success_rate

# If success_rate < 0.95, Engram disables GPU automatically

# Check why GPU is failing
journalctl -u engram | grep "GPU launch failed"

# Common causes:
# - OOM errors (see Section: Out of Memory)
# - CUDA errors (see Section: CUDA Error Codes)
```

#### Step 4: Enable Debug Logging

```bash
# Enable GPU debug logging
export RUST_LOG=engram::activation::gpu=debug
systemctl restart engram

# Watch logs
journalctl -u engram -f

# Look for lines like:
# "Batch size 64 < min 128, using CPU" <- Batch too small
# "GPU success rate 0.85 < threshold 0.95, using CPU" <- Too many failures
# "GPU not available, using CPU" <- GPU disabled or not detected
```

### Out of Memory (OOM)

**Symptom**: `cudaErrorMemoryAllocation` errors in logs

**Diagnostic Output**:
```
ERROR engram::activation::gpu: GPU launch failed: OutOfMemory
WARN  engram::activation::gpu: GPU OOM at batch size 4096, falling back to CPU
```

**Diagnostic Steps**:

#### Step 1: Check Available VRAM

```bash
# Check GPU memory usage
nvidia-smi --query-gpu=memory.used,memory.free,memory.total --format=csv

# Output:
# memory.used [MiB], memory.free [MiB], memory.total [MiB]
# 8192 MiB, 4096 MiB, 12288 MiB
#         ^^^^
#         Only 4GB free

# If memory.free is low, other processes are using VRAM
nvidia-smi

# Check "Processes" section for other GPU users
# If other processes exist:
# - Stop them if possible
# - Or reduce Engram batch sizes
```

#### Step 2: Check Batch Size

```bash
# Check what batch size triggered OOM
journalctl -u engram | grep "OOM at batch size"

# Output:
# GPU OOM at batch size 4096

# Calculate memory needed:
# 4096 vectors * 768 dimensions * 4 bytes (f32) = 12MB per batch
# Plus intermediate results and overhead ~= 50MB total

# If this exceeds available VRAM, reduce batch size
# Edit /etc/engram/gpu.toml:
[gpu]
min_batch_size = 2048  # Reduce from 4096

systemctl restart engram
```

#### Step 3: Adjust VRAM Safety Margin

```bash
# Increase VRAM safety margin
# Edit /etc/engram/gpu.toml:
[gpu]
vram_safety_margin = 0.7  # Was 0.8, now only use 70% of VRAM

# This reserves more VRAM for system and reduces OOM risk
systemctl restart engram
```

#### Step 4: Check for Memory Leaks

```bash
# Monitor GPU memory over time
watch -n 1 nvidia-smi

# If memory.used keeps increasing and never decreases:
# Potential memory leak in CUDA code

# Restart Engram to clear leaked memory
systemctl restart engram

# If memory immediately climbs again, file a bug report
```

### CUDA Error Codes (FOR FUTURE GPU IMPLEMENTATION)

**Important**: The CUDA errors described below CANNOT occur in Milestone 12 since no CUDA code is executed. This section is for future reference when CUDA kernels are implemented in Milestone 13+.

**Future Symptom**: CUDA errors in logs with numeric codes

**Common CUDA Errors** (will be relevant when GPU features are implemented):

#### cudaErrorInvalidValue (Error 1)

```
ERROR: CUDA error 1: invalid argument
```

**Causes**:
- Null pointer passed to kernel
- Invalid dimension (batch size 0, negative values)
- Misaligned memory

**Solution**:
```bash
# Enable CUDA error checking
export CUDA_LAUNCH_BLOCKING=1

# Run with debug logging
RUST_LOG=debug ./target/release/engram start

# Look for exact line where error occurs
# File bug report with backtrace
```

#### cudaErrorMemoryAllocation (Error 2)

See "Out of Memory" section above.

#### cudaErrorInitializationError (Error 3)

```
ERROR: CUDA error 3: initialization error
```

**Causes**:
- Driver/CUDA version mismatch
- GPU in bad state
- Insufficient permissions

**Solution**:
```bash
# Reset GPU
sudo nvidia-smi --gpu-reset

# If fails, reboot
sudo reboot

# Check driver/CUDA compatibility
nvidia-smi | grep "CUDA Version"
nvcc --version

# Ensure driver supports CUDA version
```

#### cudaErrorInsufficientDriver (Error 35)

```
ERROR: CUDA error 35: insufficient driver
```

**Cause**: NVIDIA driver is too old for CUDA version

**Solution**:
```bash
# Check current driver version
nvidia-smi | grep "Driver Version"

# Check required driver version
# CUDA 11.8 requires driver >= 520.61.05
# CUDA 12.0 requires driver >= 525.60.13

# Update driver
sudo apt install nvidia-driver-535  # Ubuntu
sudo dnf install nvidia-driver-latest  # RHEL

sudo reboot
```

#### cudaErrorNoDevice (Error 100)

```
ERROR: CUDA error 100: no CUDA-capable device detected
```

**Causes**:
- GPU not detected by system
- Wrong CUDA_VISIBLE_DEVICES setting

**Solution**:
```bash
# Check if GPU is visible
nvidia-smi

# Check CUDA_VISIBLE_DEVICES
echo $CUDA_VISIBLE_DEVICES

# If set to empty or invalid value:
export CUDA_VISIBLE_DEVICES=0

# Or unset it to use all GPUs
unset CUDA_VISIBLE_DEVICES
```

### Low GPU Utilization

**Symptom**: `nvidia-smi` shows GPU utilization (sm%) is low (<30%) during workload

**Diagnostic Steps**:

#### Step 1: Check Batch Sizes

```bash
# Low GPU utilization often means batches are too small
curl http://localhost:8080/metrics | grep batch_size_histogram

# If most batches are < 256 vectors:
# GPU is underutilized because there's not enough parallel work

# Solution: Increase batch sizes in application
# Or lower min_batch_size to use CPU for small batches
```

#### Step 2: Check Launch Frequency

```bash
# Check how often GPU is launched
curl http://localhost:8080/metrics | grep gpu_launches_total

# Note the value, wait 10 seconds, check again
# If launches per second is < 10:
# GPU is idle most of the time

# This is normal for low-QPS workloads
# If QPS is high but GPU launches are low:
# - Batches may be too small (below min_batch_size)
# - Check logs for why GPU isn't being used
```

#### Step 3: Check for CPU Bottleneck

```bash
# Monitor CPU usage during workload
top

# If CPU is at 100% but GPU is at 30%:
# CPU is bottleneck, GPU is waiting for data

# Solutions:
# - Increase CPU cores
# - Optimize data preparation code
# - Use batch pipelining (future feature)
```

#### Step 4: Profile Kernel Execution

```bash
# Use NVIDIA profiler to see detailed GPU activity
nsys profile -o timeline.qdrep ./target/release/engram start

# Analyze with nsys-ui (requires GUI):
nsys-ui timeline.qdrep

# Or generate report:
nsys stats --report cuda_gpu_kern_sum timeline.qdrep

# Look for:
# - Kernel execution time vs idle time
# - Memory transfer overhead
# - Launch frequency
```

### Performance Regression

**Symptom**: GPU speedup is lower than expected (< 3x vs CPU)

**Diagnostic Steps**:

#### Step 1: Run Benchmark Suite

```bash
# Run official benchmarks
cargo bench --bench gpu_performance_validation

# Compare results to baseline (see performance_report.md)
# If speedup < 3x, investigate:
```

#### Step 2: Check GPU Clock Speed

```bash
# Check current GPU clocks
nvidia-smi --query-gpu=clocks.sm,clocks.mem --format=csv

# Compare to maximum clocks
nvidia-smi --query-gpu=clocks.max.sm,clocks.max.mem --format=csv

# If current << max, GPU may be throttling
# Check temperature and power limit
nvidia-smi --query-gpu=temperature.gpu,power.draw,power.limit --format=csv

# If temperature > 80C or power.draw ~= power.limit:
# GPU is thermal or power throttling

# Solutions:
# - Improve cooling (clean fans, better case airflow)
# - Increase power limit (if supported):
sudo nvidia-smi -pl 250  # Set power limit to 250W (check max for your GPU)
```

#### Step 3: Check for PCIe Bottleneck

```bash
# Check PCIe generation and width
nvidia-smi --query-gpu=pcie.link.gen.current,pcie.link.width.current --format=csv

# Expected:
# 4, 16  (PCIe Gen 4 x16)
# 3, 16  (PCIe Gen 3 x16)

# If you see:
# 3, 8   (PCIe Gen 3 x8) <- Half bandwidth
# 1, 16  (PCIe Gen 1 x16) <- Very slow

# Possible causes:
# - GPU in wrong PCIe slot (use x16 slot closest to CPU)
# - BIOS PCIe setting wrong
# - Riser cable limiting speed

# Check PCIe in BIOS:
# - Enable "Gen 4" or "Gen 3" (not "Auto" or "Gen 1")
# - Enable "x16" mode
```

#### Step 4: Check Driver/CUDA Optimization

```bash
# Ensure using CUDA runtime (not debug build)
ldd ./target/release/engram | grep cuda

# Should link to libcudart.so, not libcudart_static.a

# Ensure CUDA kernel caching enabled
export CUDA_CACHE_DISABLE=0  # Should be 0 (enabled)

# Check for debug flags accidentally left on
echo $CUDA_LAUNCH_BLOCKING  # Should be unset or 0

# If set to 1, kernels run synchronously (slow)
unset CUDA_LAUNCH_BLOCKING
```

## Diagnostic Commands Reference

### Quick Health Check

```bash
# One-liner to check entire CUDA stack
nvidia-smi && nvcc --version && ./target/release/engram --gpu-info && curl -s http://localhost:8080/metrics | grep -E "gpu_(launches|fallbacks|success_rate)"
```

### Detailed GPU Information

```bash
# Full GPU details
nvidia-smi -q

# Specific fields
nvidia-smi --query-gpu=name,compute_cap,memory.total,memory.free,temperature.gpu,power.draw --format=csv

# Monitor GPU in real-time
nvidia-smi dmon -s puct -c 100

# Export to file for analysis
nvidia-smi --query-gpu=timestamp,utilization.gpu,utilization.memory,memory.used,temperature.gpu --format=csv -l 1 > gpu_metrics.csv
```

### CUDA Profiling

```bash
# System-wide timeline (what's running when)
nsys profile -o timeline.qdrep ./target/release/engram start

# Kernel-level details (why kernels are slow)
ncu --set full -o kernel_profile.ncu-rep ./target/release/engram start

# Memory bandwidth analysis
ncu --metrics dram__throughput.avg.pct_of_peak_sustained_elapsed ./target/release/engram start

# Warp efficiency
ncu --metrics smsp__average_warps_issue_stalled_per_issue_active.pct ./target/release/engram start
```

### Engram Diagnostics (PLANNED for Milestone 13+)

**Note**: The CLI commands below do NOT exist in Milestone 12. They describe planned future functionality.

```bash
# Check GPU detection (PLANNED - not yet implemented)
./target/release/engram --gpu-info

# Dump configuration (PLANNED - not yet implemented)
./target/release/engram config show

# Test GPU with minimal workload (PLANNED - not yet implemented)
./target/release/engram benchmark --gpu --batch-size 128

# Check metrics (currently available, but GPU metrics will be 0 or absent)
curl http://localhost:8080/metrics | grep -E "gpu|spreading"
```

**Current Milestone 12**: No GPU-specific CLI commands available.

## Decision Trees (FOR FUTURE GPU IMPLEMENTATION)

**Note**: These decision trees will be useful when GPU features are implemented in Milestone 13+. They are not applicable to Milestone 12.

### GPU Not Working - Where to Start? (Future)

```
nvidia-smi works?
├─ YES: CUDA installed and drivers OK
│   └─ engram --gpu-info shows GPU?
│       ├─ YES: GPU detected by Engram
│       │   └─ gpu_launches_total > 0?
│       │       ├─ YES: GPU is being used - check performance
│       │       └─ NO: See "GPU Not Being Used" section
│       └─ NO: See "Engram Cannot Detect GPU" section
└─ NO: Driver or hardware issue
    └─ lspci | grep nvidia shows GPU?
        ├─ YES: GPU detected, driver issue
        │   └─ See "GPU Not Detected by System" section
        └─ NO: Hardware issue
            └─ Check:
                - GPU properly seated
                - Power cables connected
                - BIOS settings
                - Try different PCIe slot
```

### Performance Issues - Diagnosis Flow

```
Expected speedup achieved?
├─ NO: Performance problem
│   └─ GPU utilization high (>80%)?
│       ├─ YES: GPU is working hard but slow
│       │   └─ Check:
│       │       - GPU temperature/throttling
│       │       - PCIe bandwidth
│       │       - Batch sizes (too large?)
│       │       - Run ncu profiling
│       └─ NO: GPU is idle
│           └─ Check:
│               - Batch sizes too small?
│               - CPU bottleneck?
│               - Launch frequency low?
│               - Run nsys profiling
└─ YES: Working as expected
    └─ Monitor for regressions
```

## When to File a Bug Report

File a bug report if:

1. **GPU detection fails after following all troubleshooting steps**
   - `nvidia-smi` works
   - CUDA toolkit installed correctly
   - Still `engram --gpu-info` shows "Not Available"

2. **Frequent OOM despite reasonable batch sizes**
   - Batch size < 1024 vectors
   - VRAM available > 4GB
   - Still getting OOM errors

3. **Speedup significantly lower than baseline**
   - Measured < 2x speedup
   - Baseline predicts > 5x speedup
   - All optimizations applied

4. **CUDA errors that persist after driver/CUDA reinstall**
   - Consistent error codes
   - Reproducible steps
   - Happens on multiple machines

**What to Include in Bug Report**:

```bash
# System information
uname -a
nvidia-smi
nvcc --version
./target/release/engram --version

# Configuration
cat /etc/engram/gpu.toml

# Logs with debug enabled
RUST_LOG=engram::activation::gpu=debug ./target/release/engram start 2>&1 | head -100

# Metrics snapshot
curl http://localhost:8080/metrics | grep -E "gpu|spreading"

# Benchmark results
cargo bench --bench gpu_performance_validation 2>&1 | grep Speedup
```

## Additional Resources

- **NVIDIA Documentation**: https://docs.nvidia.com/cuda/
- **CUDA Troubleshooting**: https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#troubleshooting
- **GPU Architecture Reference**: See GPU Architecture Reference doc
- **Performance Tuning**: See GPU Performance Tuning Guide
- **Community Support**: GitHub Discussions (link TBD)
