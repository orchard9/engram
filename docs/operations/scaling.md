# Scaling Engram

Comprehensive guide to vertical and horizontal scaling strategies for Engram deployments. This document provides decision matrices, procedures, and operational guidance for scaling Engram to meet performance and capacity requirements.

## Overview

Engram's tiered memory architecture enables efficient scaling across multiple dimensions:

- **Vertical Scaling**: Add CPU, memory, or storage to a single instance
- **Horizontal Scaling**: Distribute workload across multiple instances (future)
- **Tier Optimization**: Balance data across hot/warm/cold tiers for cost efficiency

This guide focuses on current vertical scaling capabilities and provides a foundation for future horizontal scaling.

## Scaling Decision Matrices

Use these matrices to determine when and how to scale based on observed metrics.

### CPU Scaling Decision Matrix

| Current Utilization | P99 Latency | Action | Implementation |
|---------------------|-------------|---------|----------------|
| <50% | <10ms | No action | Monitor for trends, review quarterly |
| 50-70% | <10ms | Plan scaling | Schedule maintenance window within 2 weeks |
| 70-85% | 10-50ms | Scale immediately | Add 2x cores (up to 8 cores) within 48 hours |
| 70-85% | >50ms | Scale urgently | Add 4x cores (up to 16 cores) within 24 hours |
| >85% | Any | Emergency scale | Maximum available cores + investigate bottleneck |

**Key Metrics to Monitor:**
```
engram_cpu_utilization_percent{core}
engram_spreading_queue_depth
engram_consolidation_lag_seconds
engram_p99_latency_ms{operation="spreading"}
```

**Scaling Actions:**
1. Identify bottleneck: spreading, consolidation, or API overhead
2. Add cores in NUMA-friendly increments (2, 4, 8, 16, 32)
3. Monitor for 30 minutes post-scaling
4. Validate latency returns to <10ms

### Memory Scaling Decision Matrix

| Memory Pressure | Tier Distribution | Action | Implementation |
|-----------------|-------------------|---------|----------------|
| <60% | Balanced | No action | Monitor growth rate weekly |
| 60-75% | Hot-heavy (>50%) | Optimize | Increase eviction rate to warm tier |
| 75-85% | Balanced | Scale RAM | Add 50% more RAM within 1 week |
| 75-85% | Cold-heavy | Migrate | Move cold data to disk, reduce mmap windows |
| >85% | Any | Emergency | Double RAM + immediate tier rebalancing |

**Key Metrics to Monitor:**
```
engram_memory_usage_bytes{tier}
engram_eviction_rate{from_tier,to_tier}
engram_allocation_failures_total
engram_hot_tier_ratio
```

**Scaling Actions:**
1. Check tier distribution (hot should be <10% of total nodes)
2. Adjust eviction thresholds before adding RAM
3. Add memory in 25-50% increments
4. Verify allocation failures stop

### Storage Scaling Triggers

| Metric | Threshold | Action | Procedure |
|--------|-----------|--------|-----------|
| Disk utilization | >70% | Expand volume | Add 2x current size or next tier |
| WAL size | >10GB | Increase compaction | Reduce retention, increase frequency |
| IOPS utilization | >80% | Upgrade disk type | Move to NVMe or increase IOPS allocation |
| Snapshot count | >10 | Prune snapshots | Keep only last 5 + weekly archives |
| Compaction backlog | >5 segments | Dedicate CPU | Allocate 1-2 cores to compaction |

**Key Metrics to Monitor:**
```
engram_storage_usage_bytes{tier}
engram_wal_size_bytes
engram_compaction_pending_segments
engram_disk_iops_utilized_percent
```

**Scaling Actions:**
1. Verify available disk space >30%
2. Expand volume or add new disk tier
3. Run compaction to reclaim space
4. Archive old snapshots to object storage

## Vertical Scaling Procedures

### Kubernetes Vertical Scaling

**Pre-requisites:**
- kubectl access to cluster
- Backup completed within last 24 hours
- Maintenance window scheduled

**Procedure:**

```bash
#!/bin/bash
# Vertical scaling for Kubernetes deployment

# 1. Check current resource allocation
echo "Current resources:"
kubectl get deployment engram -o yaml | grep -A10 resources

# 2. Verify current pod status
kubectl get pods -l app=engram

# 3. Update resource limits (example: scale to 8 CPU, 16GB RAM)
kubectl set resources deployment engram \
  --limits=cpu=8,memory=16Gi \
  --requests=cpu=4,memory=8Gi

# 4. Trigger rolling update
kubectl rollout restart deployment engram

# 5. Monitor rollout status
kubectl rollout status deployment engram --timeout=5m

# 6. Verify new pods are running
kubectl get pods -l app=engram

# 7. Check resource allocation
kubectl top pods -l app=engram

# 8. Verify application health
kubectl exec -it $(kubectl get pod -l app=engram -o jsonpath='{.items[0].metadata.name}') \
  -- curl -f http://localhost:7432/health || echo "Health check failed"

# 9. Monitor metrics for 10 minutes
for i in {1..10}; do
  echo "Minute $i: Checking metrics..."
  kubectl exec -it $(kubectl get pod -l app=engram -o jsonpath='{.items[0].metadata.name}') \
    -- curl -s http://localhost:7432/metrics | grep -E "cpu_utilization|memory_usage"
  sleep 60
done

echo "Scaling complete. Monitor dashboards for 30 minutes."
```

**Expected Duration:** 3-5 minutes

**Rollback Procedure:**
```bash
# If scaling causes issues, rollback to previous version
kubectl rollout undo deployment engram
kubectl rollout status deployment engram
```

### Docker Vertical Scaling

**Pre-requisites:**
- Docker access
- Backup completed
- Data volume persisted

**Procedure:**

```bash
#!/bin/bash
# Vertical scaling for Docker container

# 1. Check current container resources
echo "Current container stats:"
docker stats engram --no-stream

# 2. Enable maintenance mode (if supported)
docker exec engram curl -X POST http://localhost:7432/api/v1/admin/maintenance/enable

# 3. Wait for in-flight operations to complete (30 seconds)
sleep 30

# 4. Stop current container gracefully (30 second timeout)
docker stop --time 30 engram

# 5. Remove old container (keep data volume)
docker rm engram

# 6. Start new container with increased resources
docker run -d \
  --name engram \
  --cpus="8" \
  --memory="16g" \
  --memory-reservation="8g" \
  --memory-swap="18g" \
  -v engram-data:/data \
  -v engram-wal:/wal \
  -p 7432:7432 \
  -p 9090:9090 \
  --restart unless-stopped \
  engram:latest

# 7. Wait for startup (up to 2 minutes)
timeout 120 bash -c 'until docker exec engram curl -f http://localhost:7432/health; do sleep 5; done'

# 8. Verify resources
docker stats engram --no-stream

# 9. Disable maintenance mode
docker exec engram curl -X POST http://localhost:7432/api/v1/admin/maintenance/disable

# 10. Run smoke tests
docker exec engram /app/scripts/smoke_test.sh

echo "Scaling complete. Monitor for 30 minutes."
```

**Expected Duration:** 2-4 minutes (including cold start)

**Rollback Procedure:**
```bash
# Stop new container and restart with original resources
docker stop engram
docker rm engram
docker run -d --name engram --cpus="4" --memory="8g" -v engram-data:/data engram:latest
```

### Bare Metal Vertical Scaling

**Pre-requisites:**
- systemd service configuration
- Root access
- Hardware resources available

**Procedure:**

```bash
#!/bin/bash
# Vertical scaling for bare metal deployment

# 1. Check current resource usage
systemctl status engram
ps aux | grep engram | grep -v grep

# 2. Update systemd service configuration
sudo nano /etc/systemd/system/engram.service
# Update CPUQuota, MemoryLimit, etc.

# Example changes:
# CPUQuota=800%     # 8 cores
# MemoryLimit=16G

# 3. Reload systemd configuration
sudo systemctl daemon-reload

# 4. Restart service with new limits
sudo systemctl restart engram

# 5. Verify service started successfully
sudo systemctl status engram

# 6. Check resource allocation
ps -p $(pgrep engram) -o %cpu,%mem,rss,vsz,cmd

# 7. Monitor logs for errors
sudo journalctl -u engram -f --since "1 minute ago"
```

**Expected Duration:** 1-3 minutes

### Storage Expansion

**NVMe/SSD Volume Expansion (AWS EBS Example):**

```bash
#!/bin/bash
# Expand EBS volume for Engram storage

# 1. Identify current volume
VOLUME_ID=$(aws ec2 describe-volumes \
  --filters "Name=attachment.instance-id,Values=$INSTANCE_ID" \
  --query "Volumes[0].VolumeId" --output text)

echo "Current volume: $VOLUME_ID"

# 2. Modify volume size (example: expand to 500GB)
aws ec2 modify-volume --volume-id $VOLUME_ID --size 500

# 3. Wait for modification to complete
aws ec2 wait volume-available --volume-ids $VOLUME_ID

# 4. Extend filesystem (on instance)
sudo growpart /dev/nvme1n1 1
sudo resize2fs /dev/nvme1n1p1

# 5. Verify new size
df -h /data

echo "Storage expansion complete."
```

## Tier Rebalancing

When memory pressure is high, rebalance data across tiers instead of immediately adding RAM.

### Manual Tier Rebalancing

```bash
#!/bin/bash
# Rebalance storage tiers to reduce memory pressure

# 1. Check current tier distribution
curl -s http://localhost:7432/api/v1/admin/storage/stats | jq '.tier_distribution'

# Example output:
# {
#   "hot": 150000,      # 15% of nodes (too high)
#   "warm": 300000,     # 30%
#   "cold": 550000      # 55%
# }

# 2. Adjust eviction thresholds to be more aggressive
curl -X POST http://localhost:7432/api/v1/admin/storage/eviction \
  -H "Content-Type: application/json" \
  -d '{
    "hot_threshold": 0.2,    # Evict to warm if activation < 0.2 (was 0.3)
    "warm_threshold": 0.05   # Evict to cold if activation < 0.05 (was 0.1)
  }'

# 3. Trigger immediate eviction sweep
curl -X POST http://localhost:7432/api/v1/admin/storage/evict-now

# 4. Monitor tier migration progress (watch for 5 minutes)
watch -n 5 'curl -s http://localhost:7432/api/v1/admin/storage/stats | jq .tier_distribution'

# 5. Verify memory usage decreased
curl -s http://localhost:7432/metrics | jq '.memory.hot_tier_bytes'

# Expected outcome: Hot tier reduced to <10% of total nodes
```

### Automated Tier Rebalancing Policy

Configure automatic tier rebalancing based on memory pressure:

```yaml
# engram.yaml configuration
storage:
  tier_policy:
    enabled: true

    # Hot tier eviction policy
    hot_tier:
      max_ratio: 0.10              # Never exceed 10% of total nodes
      activation_threshold: 0.3     # Evict if activation < 0.3
      eviction_interval: 60s        # Check every minute

    # Warm tier eviction policy
    warm_tier:
      max_ratio: 0.30              # Never exceed 30% of total nodes
      activation_threshold: 0.1    # Evict if activation < 0.1
      eviction_interval: 300s      # Check every 5 minutes

    # Memory pressure triggers
    memory_pressure:
      high_threshold: 0.80         # Start aggressive eviction at 80%
      critical_threshold: 0.90     # Emergency eviction at 90%
```

## NUMA-Aware Scaling

For systems with >32GB RAM, configure NUMA-aware memory allocation to avoid cross-node memory access penalties.

### NUMA Configuration

**Check NUMA topology:**
```bash
numactl --hardware

# Example output:
# available: 2 nodes (0-1)
# node 0 cpus: 0 1 2 3 4 5 6 7
# node 0 size: 65536 MB
# node 0 free: 32000 MB
# node distances:
# node   0   1
#   0:  10  20
#   1:  20  10
```

**Pin Engram to NUMA node 0:**
```bash
# Start with memory bound to node 0 for hot tier
numactl --membind=0 --cpunodebind=0 engram-server \
  --hot-tier-size=16GB \
  --numa-aware=true

# Verify NUMA allocation
numastat engram-server

# Expected: Most memory on node 0
```

**Monitor NUMA statistics:**
```bash
# Check for excessive remote memory access
numastat -p $(pgrep engram-server)

# Look for low "numa_foreign" and "local_node" > 95%
```

## CPU Affinity for Performance

Pin critical threads to dedicated cores to reduce context switching and improve cache locality.

### CPU Pinning

```bash
# Pin Engram to cores 0-7 (first NUMA node)
taskset -c 0-7 engram-server

# Or use systemd service configuration
# Add to /etc/systemd/system/engram.service:
# CPUAffinity=0 1 2 3 4 5 6 7

# Verify CPU affinity
taskset -cp $(pgrep engram-server)
# Expected: current affinity list: 0-7
```

### Thread Pinning (Advanced)

For maximum performance, pin different workload types to dedicated cores:

```bash
# Configuration in engram.yaml
threading:
  spreading_cores: [0, 1, 2, 3]      # Dedicate cores 0-3 to spreading
  consolidation_cores: [4, 5]         # Dedicate cores 4-5 to consolidation
  api_cores: [6, 7]                   # Dedicate cores 6-7 to API handling

  # Enable thread pinning
  pin_threads: true

  # Prevent thread migration
  cpu_isolation: true
```

**Expected Performance Improvement:** 15-25% throughput increase

## Monitoring Scaling Effectiveness

After scaling, monitor these metrics to validate effectiveness:

### Post-Scaling Validation Checklist

```bash
#!/bin/bash
# Validate scaling effectiveness

echo "=== Post-Scaling Validation ==="
echo ""

# 1. CPU utilization normalized
echo "CPU Utilization (should be <70%):"
curl -s http://localhost:7432/metrics | jq '.cpu.utilization_percent'

# 2. Memory pressure reduced
echo "Memory Usage (should be <80%):"
curl -s http://localhost:7432/metrics | jq '.memory.usage_percent'

# 3. Latency improved
echo "P99 Latency (should be <10ms):"
curl -s http://localhost:7432/metrics | jq '.latency.p99_ms'

# 4. Queue depth stable
echo "Spreading Queue Depth (should be <50):"
curl -s http://localhost:7432/metrics | jq '.spreading.queue_depth'

# 5. Error rate normal
echo "Error Rate (should be <0.1%):"
curl -s http://localhost:7432/metrics | jq '.errors.rate_per_sec'

# 6. Tier distribution healthy
echo "Hot Tier Ratio (should be <10%):"
curl -s http://localhost:7432/api/v1/admin/storage/stats | jq '.tier_distribution.hot / .tier_distribution.total'

echo ""
echo "=== Validation Complete ==="
```

### Performance Regression Detection

Compare pre-scaling and post-scaling performance:

```bash
#!/bin/bash
# Capture baseline before scaling
./scripts/benchmark_deployment.sh > pre-scaling-benchmark.json

# Perform scaling operation
# ...

# Capture performance after scaling
./scripts/benchmark_deployment.sh > post-scaling-benchmark.json

# Compare results
python3 <<EOF
import json

with open('pre-scaling-benchmark.json') as f:
    pre = json.load(f)

with open('post-scaling-benchmark.json') as f:
    post = json.load(f)

# Calculate improvements
cpu_improvement = (pre['cpu_usage'] - post['cpu_usage']) / pre['cpu_usage'] * 100
latency_improvement = (pre['p99_latency'] - post['p99_latency']) / pre['p99_latency'] * 100
throughput_improvement = (post['ops_per_sec'] - pre['ops_per_sec']) / pre['ops_per_sec'] * 100

print(f"CPU Usage: {cpu_improvement:.1f}% reduction")
print(f"P99 Latency: {latency_improvement:.1f}% improvement")
print(f"Throughput: {throughput_improvement:.1f}% increase")

# Validate scaling was effective
if cpu_improvement < 10 and throughput_improvement < 20:
    print("WARNING: Scaling did not significantly improve performance")
    print("Investigate bottleneck: may not be CPU/memory limited")
EOF
```

## Scaling Alert Rules

Configure Prometheus alerts to automate scaling decisions:

```yaml
# prometheus-alerts.yaml
groups:
  - name: engram-scaling
    interval: 30s
    rules:
      # CPU scaling alert
      - alert: ScaleUpCPURequired
        expr: |
          (avg_over_time(engram_cpu_utilization_percent[5m]) > 70)
          and
          (rate(engram_spreading_queue_depth[5m]) > 0)
        for: 10m
        labels:
          severity: warning
          component: scaling
        annotations:
          summary: "CPU utilization high - scaling recommended"
          description: "CPU at {{ $value }}% for 10+ minutes with growing queue"
          action: "Add 2x CPU cores or investigate bottleneck"
          runbook: "https://docs.engram.dev/operations/scaling#cpu-scaling"

      # Memory scaling alert
      - alert: ScaleUpMemoryRequired
        expr: |
          (engram_memory_usage_bytes{tier="hot"} / engram_memory_limit_bytes) > 0.80
        for: 5m
        labels:
          severity: warning
          component: scaling
        annotations:
          summary: "Memory pressure high - scaling recommended"
          description: "Hot tier at {{ $value | humanizePercentage }} capacity"
          action: "Add 50% more RAM or increase eviction rate"
          runbook: "https://docs.engram.dev/operations/scaling#memory-scaling"

      # Storage scaling alert
      - alert: ScaleUpStorageRequired
        expr: |
          predict_linear(engram_storage_usage_bytes[6h], 86400)
          > engram_storage_limit_bytes * 0.90
        for: 30m
        labels:
          severity: warning
          component: scaling
        annotations:
          summary: "Storage capacity will be exhausted within 24 hours"
          description: "Predicted to reach {{ $value | humanize1024 }} in 24h"
          action: "Expand volume by 2x or archive old data"
          runbook: "https://docs.engram.dev/operations/scaling#storage-scaling"

      # Scaling effectiveness validation
      - alert: ScalingIneffective
        expr: |
          (engram_cpu_utilization_percent > 80)
          and
          (changes(engram_cpu_limit_cores[30m]) > 0)
        for: 15m
        labels:
          severity: critical
          component: scaling
        annotations:
          summary: "Recent scaling did not reduce CPU utilization"
          description: "CPU still at {{ $value }}% after scaling event"
          action: "Investigate bottleneck - may not be CPU-bound"
          runbook: "https://docs.engram.dev/operations/troubleshooting#scaling-ineffective"
```

## Cost Optimization Through Right-Sizing

Avoid over-provisioning by following a phased scaling approach:

### Phase 1: Initial Deployment (Day 0-7)

```bash
# Start with 2x calculated baseline from capacity planner
./scripts/estimate_capacity.sh 1000000 0.1 50 5000 30 300

# Example output:
# Recommended: 4 cores, 8GB RAM, 100GB storage

# Deploy with 2x headroom
# Actual deployment: 8 cores, 16GB RAM, 200GB storage
```

**Goals:**
- Ensure adequate headroom for unexpected spikes
- Capture actual utilization patterns
- Identify workload characteristics

### Phase 2: Optimization (Day 7-30)

```bash
# Analyze 7-day metrics
./scripts/analyze_metrics.sh --period 7d --output capacity-analysis.json

# Example findings:
# - Peak CPU: 45% (of 8 cores)
# - Peak Memory: 60% (of 16GB)
# - Peak Storage: 40% (of 200GB)

# Right-size to 1.3x peak
# New deployment: 4 cores (1.3 * 45% * 8), 10GB RAM, 100GB storage
```

**Expected Cost Savings:** 30-40% reduction

### Phase 3: Steady State (Day 30+)

```bash
# Target utilization ranges
# CPU: 60-70% average, 85% peak
# Memory: 65-75% average, 85% peak
# Storage: 50-65% average, 75% peak

# Adjust monthly based on growth trends
# Set up auto-scaling for CPU/memory (when available)
```

**Expected Cost Savings:** 40-50% vs initial over-provisioning

## Disaster Recovery Scaling Considerations

Scale disaster recovery resources based on Recovery Time Objective (RTO):

### Cold Standby (RTO: 1-4 hours)

```yaml
# Minimal resources for cold DR
resources:
  cpu: 2 cores          # Minimum to run
  memory: 4GB           # Enough for WAL replay
  storage: 1.5x production  # Full data + WAL

cost: 10-20% of production
```

### Warm Standby (RTO: 5-30 minutes)

```yaml
# Moderate resources for warm DR
resources:
  cpu: 50% of production
  memory: 75% of production  # Most of hot tier
  storage: 1.5x production

cost: 40-60% of production
```

### Hot Standby (RTO: <5 minutes)

```yaml
# Full resources for hot DR
resources:
  cpu: 100% of production
  memory: 100% of production
  storage: 1.5x production

cost: 100-120% of production
```

## Horizontal Scaling (Future)

Horizontal scaling will be supported in Milestone 17. Planned capabilities:

### Shard-Based Scaling

- Consistent hashing for data distribution
- Automatic rebalancing on node addition/removal
- Query routing layer for transparent access

### Read Replicas

- Asynchronous replication from primary
- Eventually consistent reads
- Automatic failover for high availability

### Federated Memory Spaces

- Geographic distribution for latency optimization
- Cross-region replication for disaster recovery
- Hierarchical memory consolidation

## Troubleshooting Scaling Issues

### Scaling Did Not Improve Performance

**Symptom:** Added CPU/memory but latency still high

**Diagnosis:**
```bash
# Identify actual bottleneck
./scripts/diagnose_bottleneck.sh

# Check for:
# 1. Lock contention (high CPU but low throughput)
# 2. Network saturation (high bandwidth usage)
# 3. Disk I/O bottleneck (high iowait)
# 4. Memory bandwidth (NUMA cross-node access)
```

**Resolution:**
- Lock contention: Reduce concurrent operations
- Network: Upgrade link or batch requests
- Disk I/O: Move to NVMe or add IOPS
- NUMA: Enable NUMA pinning

### Memory Usage Still High After Tier Rebalancing

**Symptom:** Hot tier not evicting to warm tier

**Diagnosis:**
```bash
# Check eviction metrics
curl -s http://localhost:7432/metrics | jq '.eviction'

# Verify activation distribution
curl -s http://localhost:7432/api/v1/admin/storage/stats | jq '.activation_histogram'
```

**Resolution:**
- Lower eviction threshold more aggressively
- Increase eviction sweep frequency
- Verify nodes are decaying properly (check decay function)
- Force manual eviction of specific nodes

### Storage Expansion Not Reflected

**Symptom:** Filesystem not using new capacity

**Diagnosis:**
```bash
# Check partition and filesystem
lsblk
df -h /data
```

**Resolution:**
```bash
# Extend partition if needed
sudo growpart /dev/nvme1n1 1

# Resize filesystem
sudo resize2fs /dev/nvme1n1p1  # ext4
# or
sudo xfs_growfs /data          # xfs
```

## Summary

- Use decision matrices to determine when to scale
- Follow documented procedures for consistent results
- Monitor post-scaling metrics to validate effectiveness
- Right-size deployments to optimize costs
- Configure alerts for proactive scaling
- Plan DR resources based on RTO requirements

## Next Steps

1. Run capacity planner: `./scripts/estimate_capacity.sh`
2. Review capacity planning guide: `/docs/operations/capacity-planning.md`
3. Set up monitoring alerts: `/docs/operations/alerting.md`
4. Practice scaling in staging environment
5. Document your scaling procedures and thresholds

## Related Documentation

- [Capacity Planning](/operations/capacity-planning.md)
- [How-to: Scale Vertically](/howto/scale-vertically.md)
- [Resource Requirements](/reference/resource-requirements.md)
- [Performance Tuning](/operations/performance-tuning.md)
- [Monitoring](/operations/monitoring.md)
