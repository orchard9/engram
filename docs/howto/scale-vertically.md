# How to Scale Vertically

Step-by-step guide for vertical scaling of Engram deployments across different platforms. This guide provides detailed procedures, validation steps, and troubleshooting for scaling CPU, memory, and storage resources.

## Overview

Vertical scaling increases resources (CPU, memory, storage) for a single Engram instance. This guide covers:

- Kubernetes deployments (production recommended)
- Docker deployments (development/small production)
- Bare metal deployments (high-performance dedicated servers)
- Storage expansion (all platforms)

**Time Required:** 5-30 minutes depending on platform
**Downtime:** 0-5 minutes (varies by platform and configuration)

## Before You Scale

### Pre-Scaling Checklist

Complete these steps before any scaling operation:

- [ ] **Backup completed**: Verify backup within last 24 hours
- [ ] **Metrics captured**: Save baseline performance metrics
- [ ] **Maintenance window**: Schedule during low-traffic period if possible
- [ ] **Stakeholders notified**: Inform team of planned scaling
- [ ] **Rollback plan**: Document steps to revert if issues occur
- [ ] **Monitoring ready**: Open dashboards to watch scaling progress

### Capture Baseline Metrics

```bash
#!/bin/bash
# Save pre-scaling metrics for comparison

echo "Capturing baseline metrics at $(date)"

# CPU utilization
curl -s http://localhost:7432/metrics | jq '.cpu' > /tmp/pre-scaling-cpu.json

# Memory usage
curl -s http://localhost:7432/metrics | jq '.memory' > /tmp/pre-scaling-memory.json

# Performance
curl -s http://localhost:7432/metrics | jq '.performance' > /tmp/pre-scaling-performance.json

# Tier distribution
curl -s http://localhost:7432/api/v1/admin/storage/stats | jq '.' > /tmp/pre-scaling-storage.json

echo "Baseline saved to /tmp/pre-scaling-*.json"
```

## Kubernetes Vertical Scaling

### Step 1: Check Current Resources

```bash
# View current deployment configuration
kubectl get deployment engram -o yaml | grep -A10 resources

# Example output:
# resources:
#   limits:
#     cpu: "4"
#     memory: 8Gi
#   requests:
#     cpu: "2"
#     memory: 4Gi

# Check current pod resource usage
kubectl top pods -l app=engram
```

### Step 2: Verify Pod Health

```bash
# Ensure pods are healthy before scaling
kubectl get pods -l app=engram

# Check for any restart loops or errors
kubectl describe pods -l app=engram | grep -A5 "State:"

# Verify application is responding
kubectl exec -it $(kubectl get pod -l app=engram -o jsonpath='{.items[0].metadata.name}') \
  -- curl -f http://localhost:7432/health
```

### Step 3: Update Resource Limits

Choose your target resources based on capacity planning.

**Option A: Using kubectl set resources (recommended)**

```bash
# Scale to 8 cores, 16GB RAM
kubectl set resources deployment engram \
  --limits=cpu=8,memory=16Gi \
  --requests=cpu=4,memory=8Gi

# Verify change was applied
kubectl get deployment engram -o yaml | grep -A10 resources
```

**Option B: Using kubectl edit**

```bash
# Edit deployment interactively
kubectl edit deployment engram

# Find the resources section and update:
# resources:
#   limits:
#     cpu: "8"
#     memory: 16Gi
#   requests:
#     cpu: "4"
#     memory: 8Gi

# Save and exit (:wq in vim)
```

**Option C: Using kubectl apply with YAML**

```bash
# Create deployment patch file
cat > /tmp/engram-resources-patch.yaml <<EOF
apiVersion: apps/v1
kind: Deployment
metadata:
  name: engram
spec:
  template:
    spec:
      containers:
      - name: engram
        resources:
          limits:
            cpu: "8"
            memory: 16Gi
          requests:
            cpu: "4"
            memory: 8Gi
EOF

# Apply patch
kubectl apply -f /tmp/engram-resources-patch.yaml
```

### Step 4: Trigger Rolling Update

```bash
# Restart deployment with new resources
kubectl rollout restart deployment engram

# Monitor rollout progress (wait for completion)
kubectl rollout status deployment engram --timeout=10m

# Watch pods being recreated
watch -n 2 kubectl get pods -l app=engram
```

**Expected behavior:**
- New pod(s) created with increased resources
- Old pod(s) terminated after new pods are ready
- Rolling update ensures zero downtime (if replicas > 1)

### Step 5: Validate New Resources

```bash
# Verify pods are running with new resources
kubectl get pods -l app=engram

# Check actual resource allocation
kubectl top pods -l app=engram

# Describe pod to see resource limits
kubectl describe pod -l app=engram | grep -A5 "Limits:"

# Example output:
# Limits:
#   cpu:     8
#   memory:  16Gi
# Requests:
#   cpu:     4
#   memory:  8Gi
```

### Step 6: Verify Application Health

```bash
# Check application health endpoint
kubectl exec -it $(kubectl get pod -l app=engram -o jsonpath='{.items[0].metadata.name}') \
  -- curl -f http://localhost:7432/health

# Expected: {"status":"healthy","version":"..."}

# Check logs for errors
kubectl logs -l app=engram --tail=50 | grep -i error

# Run smoke tests
kubectl exec -it $(kubectl get pod -l app=engram -o jsonpath='{.items[0].metadata.name}') \
  -- /app/scripts/smoke_test.sh
```

### Step 7: Monitor for 30 Minutes

```bash
# Watch metrics in real-time
watch -n 10 'kubectl exec -it $(kubectl get pod -l app=engram -o jsonpath="{.items[0].metadata.name}") -- curl -s http://localhost:7432/metrics | jq "{cpu: .cpu.utilization_percent, memory: .memory.usage_percent, latency_p99: .performance.p99_latency_ms}"'

# Check every minute for 10 minutes
for i in {1..10}; do
  echo "Minute $i: Checking metrics..."
  kubectl exec -it $(kubectl get pod -l app=engram -o jsonpath='{.items[0].metadata.name}') \
    -- curl -s http://localhost:7432/metrics | grep -E "cpu_utilization|memory_usage|p99_latency"
  sleep 60
done
```

### Rollback Procedure (Kubernetes)

If scaling causes issues:

```bash
# Rollback to previous deployment
kubectl rollout undo deployment engram

# Monitor rollback
kubectl rollout status deployment engram

# Verify rollback succeeded
kubectl get pods -l app=engram
kubectl top pods -l app=engram
```

**Expected Duration:** 3-5 minutes
**Downtime:** 0 seconds (with replicas > 1), <30 seconds (single replica)

## Docker Vertical Scaling

### Step 1: Check Current Container

```bash
# View current container stats
docker stats engram --no-stream

# Example output:
# CONTAINER ID   NAME     CPU %     MEM USAGE / LIMIT   MEM %     NET I/O
# abc123...      engram   45.2%     4.2GiB / 8GiB       52.5%     ...

# Inspect container configuration
docker inspect engram | jq '.[0].HostConfig | {CpuQuota, Memory}'
```

### Step 2: Enable Maintenance Mode

```bash
# Signal application to stop accepting new requests
docker exec engram curl -X POST http://localhost:7432/api/v1/admin/maintenance/enable

# Verify maintenance mode enabled
docker exec engram curl -s http://localhost:7432/health | jq '.maintenance_mode'

# Expected: true
```

### Step 3: Wait for In-Flight Operations

```bash
# Wait 30 seconds for operations to complete
echo "Waiting for in-flight operations to complete..."
sleep 30

# Verify queue is empty
docker exec engram curl -s http://localhost:7432/metrics | jq '.spreading.queue_depth'

# Expected: 0 or very low number
```

### Step 4: Stop Container Gracefully

```bash
# Stop container with 30 second grace period
echo "Stopping container gracefully..."
docker stop --time 30 engram

# Verify container stopped
docker ps -a | grep engram

# Expected status: Exited
```

### Step 5: Remove Old Container

```bash
# Remove container (keep volumes!)
docker rm engram

# Verify removal
docker ps -a | grep engram

# Expected: no output (container removed)
```

### Step 6: Start with New Resources

```bash
# Start container with increased resources (8 cores, 16GB RAM)
docker run -d \
  --name engram \
  --cpus="8" \
  --memory="16g" \
  --memory-reservation="8g" \
  --memory-swap="18g" \
  -v engram-data:/data \
  -v engram-wal:/wal \
  -v engram-config:/etc/engram \
  -p 7432:7432 \
  -p 9090:9090 \
  --restart unless-stopped \
  --log-driver json-file \
  --log-opt max-size=100m \
  --log-opt max-file=3 \
  engram:latest

# Verify container started
docker ps | grep engram
```

**Resource Limits Explanation:**
- `--cpus="8"`: Limit to 8 CPU cores
- `--memory="16g"`: Hard memory limit (OOM if exceeded)
- `--memory-reservation="8g"`: Soft limit (preferred allocation)
- `--memory-swap="18g"`: Total memory + swap (16GB + 2GB swap)

### Step 7: Wait for Startup

```bash
# Wait for application to be ready (up to 2 minutes)
echo "Waiting for Engram to start..."
timeout 120 bash -c 'until docker exec engram curl -f http://localhost:7432/health 2>/dev/null; do sleep 5; echo -n "."; done'

echo ""
echo "Engram is ready!"
```

### Step 8: Verify New Resources

```bash
# Check container resource allocation
docker stats engram --no-stream

# Inspect resource limits
docker inspect engram | jq '.[0].HostConfig | {
  CpuQuota,
  Memory,
  MemoryReservation,
  MemorySwap
}'

# Expected:
# {
#   "CpuQuota": 800000,  # 8 cores * 100000
#   "Memory": 17179869184,  # 16 GB
#   "MemoryReservation": 8589934592,  # 8 GB
#   "MemorySwap": 19327352832  # 18 GB
# }
```

### Step 9: Disable Maintenance Mode

```bash
# Resume normal operations
docker exec engram curl -X POST http://localhost:7432/api/v1/admin/maintenance/disable

# Verify maintenance mode disabled
docker exec engram curl -s http://localhost:7432/health | jq '.maintenance_mode'

# Expected: false
```

### Step 10: Run Smoke Tests

```bash
# Execute smoke tests inside container
docker exec engram /app/scripts/smoke_test.sh

# Check logs for errors
docker logs engram --tail=100 | grep -i error

# Monitor metrics
docker exec engram curl -s http://localhost:7432/metrics | jq '.cpu, .memory, .performance'
```

### Rollback Procedure (Docker)

If scaling causes issues:

```bash
# Stop new container
docker stop engram
docker rm engram

# Start with original resources (4 cores, 8GB)
docker run -d \
  --name engram \
  --cpus="4" \
  --memory="8g" \
  --memory-reservation="4g" \
  -v engram-data:/data \
  -v engram-wal:/wal \
  -p 7432:7432 \
  -p 9090:9090 \
  --restart unless-stopped \
  engram:latest

# Verify rollback
docker stats engram --no-stream
docker exec engram curl -f http://localhost:7432/health
```

**Expected Duration:** 2-4 minutes
**Downtime:** 1-3 minutes (cold start time)

## Bare Metal Vertical Scaling

### Step 1: Check Current Resource Usage

```bash
# Check systemd service status
systemctl status engram

# View current resource limits
systemctl show engram | grep -E "CPUQuota|MemoryLimit|IOWeight"

# Check actual process usage
ps -p $(pgrep engram) -o %cpu,%mem,rss,vsz,cmd
```

### Step 2: Update Systemd Service

```bash
# Edit service file
sudo systemctl edit --full engram.service

# Update resource limits (example for 8 cores, 16GB):
# [Service]
# CPUQuota=800%           # 8 cores (100% per core)
# MemoryLimit=16G         # 16 GB hard limit
# MemoryHigh=14G          # Start throttling at 14GB
# IOWeight=500            # I/O priority (default 100, max 10000)
# CPUWeight=500           # CPU priority (default 100, max 10000)
# Nice=-10                # Process priority (-20 to 19, lower is higher priority)

# Save and exit
```

**Example Full Service File:**
```ini
[Unit]
Description=Engram Memory Service
After=network.target

[Service]
Type=simple
User=engram
Group=engram
WorkingDirectory=/opt/engram
ExecStart=/opt/engram/bin/engram-server --config /etc/engram/config.yaml
Restart=always
RestartSec=10

# Resource Limits
CPUQuota=800%
MemoryLimit=16G
MemoryHigh=14G
IOWeight=500
CPUWeight=500
Nice=-10

# NUMA Optimization (if applicable)
CPUAffinity=0 1 2 3 4 5 6 7

# Security
NoNewPrivileges=true
PrivateTmp=true
ProtectSystem=strict
ProtectHome=true
ReadWritePaths=/opt/engram/data /opt/engram/wal

[Install]
WantedBy=multi-user.target
```

### Step 3: Reload Systemd Configuration

```bash
# Reload systemd daemon to pick up changes
sudo systemctl daemon-reload

# Verify configuration loaded
systemctl show engram | grep -E "CPUQuota|MemoryLimit"
```

### Step 4: Restart Service

```bash
# Restart Engram with new resource limits
sudo systemctl restart engram

# Monitor restart
sudo systemctl status engram

# Expected: active (running)
```

### Step 5: Verify Resource Allocation

```bash
# Check service is running
sudo systemctl status engram

# Verify resource limits applied
systemctl show engram | grep -E "CPUQuota|MemoryLimit|CPUAffinity"

# Check actual process resources
ps -p $(pgrep engram) -o %cpu,%mem,rss,vsz,cmd

# Verify CPU affinity
taskset -cp $(pgrep engram)

# Expected: current affinity list: 0-7
```

### Step 6: Monitor Logs

```bash
# Watch logs for startup errors
sudo journalctl -u engram -f --since "1 minute ago"

# Check for memory allocation issues
sudo journalctl -u engram | grep -i "memory\|oom"

# Check for CPU throttling
sudo journalctl -u engram | grep -i "cpu\|throttl"
```

### Rollback Procedure (Bare Metal)

If scaling causes issues:

```bash
# Edit service file to restore original limits
sudo systemctl edit --full engram.service

# Restore original values:
# CPUQuota=400%
# MemoryLimit=8G

# Reload and restart
sudo systemctl daemon-reload
sudo systemctl restart engram

# Verify rollback
systemctl show engram | grep -E "CPUQuota|MemoryLimit"
```

**Expected Duration:** 1-3 minutes
**Downtime:** 10-60 seconds (service restart)

## Storage Expansion

### Kubernetes Storage Expansion

**Step 1: Identify PersistentVolumeClaim**

```bash
# List PVCs used by Engram
kubectl get pvc -l app=engram

# Describe PVC to see current size
kubectl describe pvc engram-data
```

**Step 2: Expand PVC**

```bash
# Edit PVC to increase size (example: 50GB → 100GB)
kubectl edit pvc engram-data

# Update spec.resources.requests.storage:
# spec:
#   resources:
#     requests:
#       storage: 100Gi

# Save and exit
```

**Step 3: Verify Expansion**

```bash
# Check PVC status
kubectl get pvc engram-data

# Wait for CAPACITY to update
watch kubectl get pvc engram-data

# Verify filesystem expansion
kubectl exec -it $(kubectl get pod -l app=engram -o jsonpath='{.items[0].metadata.name}') \
  -- df -h /data
```

**Note:** Requires StorageClass with `allowVolumeExpansion: true`

### Docker Volume Expansion

Docker volumes automatically expand with the underlying filesystem.

**Step 1: Identify Volume**

```bash
# List volumes used by container
docker inspect engram | jq '.[0].Mounts'

# Check volume size (depends on host filesystem)
docker system df -v | grep engram
```

**Step 2: Expand Host Filesystem**

```bash
# If using cloud provider (AWS EBS example):
# 1. Expand EBS volume in AWS console or CLI
# 2. Extend partition and filesystem:

sudo growpart /dev/nvme1n1 1
sudo resize2fs /dev/nvme1n1p1

# Verify new size
df -h /var/lib/docker
```

**Step 3: Verify Available Space**

```bash
# Check available space in container
docker exec engram df -h /data

# Restart container if needed
docker restart engram
```

### Bare Metal Storage Expansion

**Option A: Expand Existing Filesystem**

```bash
# Extend logical volume (LVM example)
sudo lvextend -L +50G /dev/vg0/engram-data
sudo resize2fs /dev/vg0/engram-data

# Verify new size
df -h /opt/engram/data
```

**Option B: Add New Volume**

```bash
# Create new mount point
sudo mkdir /opt/engram/data-extended

# Format new disk
sudo mkfs.ext4 /dev/sdb1

# Mount new volume
sudo mount /dev/sdb1 /opt/engram/data-extended

# Update fstab for persistence
echo "/dev/sdb1 /opt/engram/data-extended ext4 defaults 0 2" | sudo tee -a /etc/fstab

# Configure Engram to use multiple data paths
# Update /etc/engram/config.yaml:
# storage:
#   data_paths:
#     - /opt/engram/data
#     - /opt/engram/data-extended

# Restart Engram
sudo systemctl restart engram
```

**Expected Duration:** 5-15 minutes
**Downtime:** 0 seconds (live expansion) or 1-2 minutes (restart required)

## Post-Scaling Validation

### Validation Checklist

After any scaling operation, validate:

- [ ] **Service running**: Application health endpoint responds
- [ ] **Resources allocated**: Correct CPU/memory/storage limits
- [ ] **Performance improved**: Latency and throughput metrics better
- [ ] **No errors**: No errors in logs or metrics
- [ ] **Tier distribution**: Hot/warm/cold tiers balanced
- [ ] **Smoke tests pass**: Basic operations work correctly

### Automated Validation Script

```bash
#!/bin/bash
# Post-scaling validation script

set -e

echo "=========================================="
echo "Post-Scaling Validation"
echo "=========================================="
echo ""

# 1. Health check
echo "1. Checking application health..."
HEALTH=$(curl -sf http://localhost:7432/health | jq -r '.status')
if [ "$HEALTH" = "healthy" ]; then
    echo "   OK: Application is healthy"
else
    echo "   ERROR: Application is not healthy: $HEALTH"
    exit 1
fi

# 2. CPU utilization
echo "2. Checking CPU utilization..."
CPU=$(curl -s http://localhost:7432/metrics | jq '.cpu.utilization_percent')
echo "   CPU: ${CPU}% (target: <70%)"
if (( $(echo "$CPU < 70" | bc -l) )); then
    echo "   OK: CPU utilization normal"
else
    echo "   WARNING: CPU utilization still high after scaling"
fi

# 3. Memory usage
echo "3. Checking memory usage..."
MEM=$(curl -s http://localhost:7432/metrics | jq '.memory.usage_percent')
echo "   Memory: ${MEM}% (target: <80%)"
if (( $(echo "$MEM < 80" | bc -l) )); then
    echo "   OK: Memory usage normal"
else
    echo "   WARNING: Memory usage still high after scaling"
fi

# 4. Latency
echo "4. Checking latency..."
P99=$(curl -s http://localhost:7432/metrics | jq '.performance.p99_latency_ms')
echo "   P99 Latency: ${P99}ms (target: <10ms)"
if (( $(echo "$P99 < 10" | bc -l) )); then
    echo "   OK: Latency acceptable"
else
    echo "   WARNING: Latency still high after scaling"
fi

# 5. Error rate
echo "5. Checking error rate..."
ERRORS=$(curl -s http://localhost:7432/metrics | jq '.errors.rate_per_sec')
echo "   Error rate: ${ERRORS}/sec (target: <1)"
if (( $(echo "$ERRORS < 1" | bc -l) )); then
    echo "   OK: Error rate normal"
else
    echo "   WARNING: Elevated error rate"
fi

# 6. Tier distribution
echo "6. Checking tier distribution..."
HOT_RATIO=$(curl -s http://localhost:7432/api/v1/admin/storage/stats | \
    jq '.tier_distribution.hot / .tier_distribution.total')
echo "   Hot tier ratio: ${HOT_RATIO} (target: <0.10)"
if (( $(echo "$HOT_RATIO < 0.10" | bc -l) )); then
    echo "   OK: Tier distribution healthy"
else
    echo "   WARNING: Hot tier ratio high"
fi

echo ""
echo "=========================================="
echo "Validation Complete"
echo "=========================================="
```

### Compare Pre/Post Metrics

```bash
#!/bin/bash
# Compare pre-scaling and post-scaling metrics

echo "Performance Comparison:"
echo ""

# CPU
PRE_CPU=$(jq '.utilization_percent' /tmp/pre-scaling-cpu.json)
POST_CPU=$(curl -s http://localhost:7432/metrics | jq '.cpu.utilization_percent')
CPU_IMPROVEMENT=$(echo "scale=1; ($PRE_CPU - $POST_CPU) / $PRE_CPU * 100" | bc)

echo "CPU Utilization:"
echo "  Before: ${PRE_CPU}%"
echo "  After:  ${POST_CPU}%"
echo "  Change: ${CPU_IMPROVEMENT}% reduction"
echo ""

# Latency
PRE_P99=$(jq '.p99_latency_ms' /tmp/pre-scaling-performance.json)
POST_P99=$(curl -s http://localhost:7432/metrics | jq '.performance.p99_latency_ms')
LATENCY_IMPROVEMENT=$(echo "scale=1; ($PRE_P99 - $POST_P99) / $PRE_P99 * 100" | bc)

echo "P99 Latency:"
echo "  Before: ${PRE_P99}ms"
echo "  After:  ${POST_P99}ms"
echo "  Change: ${LATENCY_IMPROVEMENT}% improvement"
echo ""

# Validation
if (( $(echo "$CPU_IMPROVEMENT < 10" | bc -l) )) && (( $(echo "$LATENCY_IMPROVEMENT < 20" | bc -l) )); then
    echo "WARNING: Scaling provided minimal improvement"
    echo "Investigate bottleneck - may not be resource-constrained"
else
    echo "SUCCESS: Scaling improved performance significantly"
fi
```

## Troubleshooting

### Issue: Pods/Containers Won't Start After Scaling

**Symptoms:**
- Pods stuck in Pending or CrashLoopBackOff
- Container exits immediately after start
- Out of memory errors in logs

**Diagnosis:**
```bash
# Kubernetes
kubectl describe pod -l app=engram | grep -A10 "Events:"
kubectl logs -l app=engram --tail=100

# Docker
docker logs engram --tail=100
docker inspect engram | jq '.[0].State'
```

**Solutions:**
1. Reduce resource requests if node can't accommodate
2. Check for OOM kills: `dmesg | grep -i oom`
3. Verify volumes are accessible
4. Check configuration file compatibility

### Issue: Performance Did Not Improve

**Symptoms:**
- CPU/memory increased but latency unchanged
- Throughput did not increase
- Queue depth still growing

**Diagnosis:**
```bash
# Identify bottleneck
./scripts/diagnose_bottleneck.sh

# Check for lock contention
curl -s http://localhost:7432/metrics | jq '.concurrency.lock_wait_ms'

# Check for disk I/O bottleneck
iostat -x 5 3
```

**Solutions:**
1. **Lock contention**: Not CPU-bound, optimize concurrency
2. **Disk I/O**: Upgrade to NVMe or increase IOPS
3. **Network**: Check bandwidth saturation
4. **NUMA**: Enable NUMA-aware allocation

### Issue: Memory Usage Still High

**Symptoms:**
- Memory usage >80% after scaling
- Frequent evictions
- Allocation failures

**Diagnosis:**
```bash
# Check tier distribution
curl -s http://localhost:7432/api/v1/admin/storage/stats | jq '.tier_distribution'

# Check for memory leaks
curl -s http://localhost:7432/metrics | jq '.memory.leak_detector'
```

**Solutions:**
1. Lower hot tier eviction threshold
2. Increase tier rebalancing frequency
3. Check for memory leaks in application
4. Verify decay functions are active

## Summary

- Choose platform-specific procedure (Kubernetes, Docker, bare metal)
- Capture baseline metrics before scaling
- Follow step-by-step procedure carefully
- Validate resources allocated correctly
- Monitor for 30 minutes post-scaling
- Compare pre/post metrics to confirm improvement
- Keep rollback plan ready

## Next Steps

1. Review capacity planning to determine target resources
2. Schedule maintenance window for scaling
3. Execute appropriate scaling procedure
4. Validate and monitor
5. Document actual vs expected improvements
6. Adjust scaling triggers based on results

## Related Documentation

- [Scaling Guide](/operations/scaling.md)
- [Capacity Planning](/operations/capacity-planning.md)
- [Resource Requirements](/reference/resource-requirements.md)
- [Performance Tuning](/operations/performance-tuning.md)
- [Troubleshooting](/operations/troubleshooting.md)
