# Streaming Infrastructure Monitoring

**Production Operations Guide for Engram Streaming**

This document provides comprehensive monitoring guidance for Engram's streaming infrastructure, including metric definitions, expected operational ranges, troubleshooting procedures, and capacity planning.

## Table of Contents

1. [Metric Catalog](#metric-catalog)
2. [Baseline Expectations](#baseline-expectations)
3. [Alert Runbook](#alert-runbook)
4. [Dashboard Guide](#dashboard-guide)
5. [Troubleshooting Workflows](#troubleshooting-workflows)
6. [Capacity Planning](#capacity-planning)

---

## Metric Catalog

### Counter Metrics

#### `engram_streaming_observations_total`

**Type**: Counter
**Labels**: `memory_space`, `priority`
**Description**: Total observations successfully processed through streaming interface
**Expected Range**: Depends on workload (typically 100-100K/sec)
**Alert Threshold**: < 100/sec for >10min indicates low throughput

**Usage**:
```promql
# Observations per second by space
rate(engram_streaming_observations_total[1m])

# Total throughput across all spaces
sum(rate(engram_streaming_observations_total[1m]))
```

#### `engram_streaming_observations_rejected_total`

**Type**: Counter
**Labels**: `memory_space`, `reason`
**Description**: Observations rejected by admission control
**Expected Range**: 0 under normal operation
**Alert Threshold**: > 1% of total observations

**Rejection Reasons**:
- `over_capacity`: Queue full, backpressure active
- `invalid_sequence`: Sequence number gap or duplicate
- `duplicate`: Observation already processed

**Usage**:
```promql
# Rejection rate by reason
sum(rate(engram_streaming_observations_rejected_total[5m])) by (reason)

# Rejection percentage
sum(rate(engram_streaming_observations_rejected_total[5m])) /
sum(rate(engram_streaming_observations_total[5m]))
```

#### `engram_streaming_backpressure_activations_total`

**Type**: Counter
**Labels**: `memory_space`, `level`
**Description**: Total backpressure state transitions
**Expected Range**: Occasional transitions OK, frequent (>100/sec) indicates capacity issues
**Alert Threshold**: > 100 activations/sec for >5min

**Levels**: `normal`, `warning`, `critical`, `overloaded`

#### `engram_streaming_batches_processed_total`

**Type**: Counter
**Labels**: `worker_id`
**Description**: Total HNSW batch insertions completed by workers
**Usage**: Track worker activity, identify idle or crashed workers

#### `engram_streaming_batch_failures_total`

**Type**: Counter
**Labels**: `worker_id`, `error_type`
**Description**: Total batch insertion failures (CRITICAL metric)
**Expected Range**: 0
**Alert Threshold**: Any non-zero value is critical

#### `engram_streaming_work_stolen_total`

**Type**: Counter
**Labels**: `worker_id`
**Description**: Total work-stealing operations
**Expected Range**: 10-30% of total batches under load imbalance
**Alert Threshold**: > 30% indicates suboptimal space distribution

### Gauge Metrics

#### `engram_streaming_queue_depth`

**Type**: Gauge
**Labels**: `priority`
**Description**: Current observation queue depth by priority
**Expected Range**:
- High: 0-1000 (rarely over 10% capacity)
- Normal: 100-50,000 (varies with load)
- Low: 0-10,000 (background processing)

**Alert Thresholds**:
- Warning: Total > 80,000 OR high > 8,000 OR normal > 80,000
- Critical: Total > 90,000 OR high > 9,000 OR normal > 90,000

**Usage**:
```promql
# Total queue depth
sum(engram_streaming_queue_depth)

# Queue utilization percentage
sum(engram_streaming_queue_depth) / 160000 * 100
```

#### `engram_streaming_worker_utilization`

**Type**: Gauge
**Labels**: `worker_id`
**Description**: Worker thread utilization percentage (0-100)
**Expected Range**: 40-80%
**Alert Thresholds**:
- < 40%: Over-provisioned (reduce workers to save resources)
- > 80%: Approaching saturation (scale workers)
- Delta > 40%: Load imbalance

**Usage**:
```promql
# Average worker utilization
avg(engram_streaming_worker_utilization)

# Load imbalance
max(engram_streaming_worker_utilization) - min(engram_streaming_worker_utilization)
```

#### `engram_streaming_active_sessions_total`

**Type**: Gauge
**Description**: Total active streaming sessions (concurrent clients)
**Expected Range**: Depends on deployment (1-1000s)
**Alert Threshold**: Sudden drops indicate client connectivity issues

#### `engram_streaming_backpressure_state`

**Type**: Gauge
**Labels**: `memory_space`
**Description**: Current backpressure state (0=Normal, 1=Warning, 2=Critical, 3=Overloaded)
**Expected Range**: Mostly 0, occasional spikes to 1
**Alert Threshold**: > 1 for >10min

#### `engram_streaming_queue_memory_bytes`

**Type**: Gauge
**Description**: Total memory used by observation queues
**Expected Range**: ~100 bytes per queued observation
**Usage**: Capacity planning for memory allocation

### Histogram Metrics

#### `engram_streaming_observation_latency_seconds`

**Type**: Histogram
**Labels**: `memory_space`
**Description**: Time from observation enqueue to HNSW insertion completion
**Expected P99**: < 100ms
**Alert Threshold**: P99 > 100ms for >5min

**Buckets**: 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0 seconds

**Usage**:
```promql
# P99 latency
histogram_quantile(0.99,
  rate(engram_streaming_observation_latency_seconds_bucket[5m]))

# P50 latency
histogram_quantile(0.50,
  rate(engram_streaming_observation_latency_seconds_bucket[5m]))
```

#### `engram_streaming_recall_latency_seconds`

**Type**: Histogram
**Labels**: `memory_space`
**Description**: Time to execute incremental recall query
**Expected P99**: < 50ms
**Alert Threshold**: P99 > 50ms for >5min

**Buckets**: 0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.5 seconds

#### `engram_streaming_batch_size`

**Type**: Histogram
**Labels**: `worker_id`
**Description**: Worker batch size distribution
**Expected Range**: Adapts from 10 (low load) to 1000 (high load)

**Buckets**: 1, 10, 50, 100, 250, 500, 1000

#### `engram_streaming_queue_wait_time_seconds`

**Type**: Histogram
**Labels**: `priority`
**Description**: Queue wait time before dequeue (scheduling latency)
**Expected Range**: < 10ms under normal operation
**Alert Threshold**: P99 > 100ms indicates processing lag

---

## Baseline Expectations

### Normal Operation

| Metric | Expected Value | Unit |
|--------|----------------|------|
| Observation Rate | 1K-100K | obs/sec |
| Queue Depth | 1K-50K | observations |
| Worker Utilization | 40-80 | % |
| P99 Observation Latency | 10-50 | ms |
| P99 Recall Latency | 5-20 | ms |
| Backpressure Activations | 0-10 | /min |
| Batch Failures | 0 | /min |
| Rejection Rate | 0 | % |

### Warning Thresholds

| Condition | Threshold | Action |
|-----------|-----------|--------|
| Queue Depth | > 80K (80%) | Monitor closely, prepare to scale |
| Worker Utilization | > 80% | Scale workers horizontally |
| P99 Latency | > 100ms | Investigate bottlenecks |
| Backpressure Activations | > 100/sec | Scale immediately |
| Load Imbalance | Delta > 40% | Review space distribution |

### Critical Thresholds

| Condition | Threshold | Action |
|-----------|-----------|--------|
| Queue Depth | > 90K (90%) | IMMEDIATE scaling required |
| Worker Down | Any worker | Page on-call, investigate |
| Batch Failures | Any failures | CRITICAL: Data loss risk |
| Rejection Rate | > 10% | Emergency capacity addition |

---

## Alert Runbook

### High Queue Depth (Warning)

**Condition**: Queue depth > 80% capacity for >2min

**Symptoms**:
- Increasing latencies
- Occasional backpressure warnings
- Queue depth steadily rising

**Investigation**:
1. Check worker utilization: Are workers saturated?
2. Check for stuck workers: Are any workers idle with non-empty queues?
3. Review observation rate: Is load higher than expected?
4. Check HNSW performance: Are insertions slowing down?

**Resolution**:
```bash
# Scale workers horizontally (Kubernetes)
kubectl scale deployment engram-workers --replicas=16

# Check worker status
kubectl get pods -l app=engram-worker

# Review worker logs
kubectl logs -l app=engram-worker --tail=100 --timestamps
```

**Prevention**:
- Set up autoscaling based on queue depth
- Monitor baseline load patterns
- Pre-scale before known high-load periods

---

### Critical Queue Depth

**Condition**: Queue depth > 90% capacity for >1min

**Symptoms**:
- High latencies (>500ms)
- Frequent admission control rejections
- Client errors increasing
- Backpressure state = Critical/Overloaded

**Investigation**:
1. **Immediate**: Scale workers NOW
2. Check for cascading failures (worker crashes)
3. Verify storage capacity (disk full?)
4. Check memory pressure (OOM kills?)

**Resolution**:
```bash
# IMMEDIATE: Scale workers aggressively
kubectl scale deployment engram-workers --replicas=32

# Check system resources
kubectl top nodes
kubectl top pods

# Check for OOM events
kubectl get events --sort-by='.lastTimestamp' | grep OOM

# Temporary: Increase queue capacity (if memory allows)
kubectl set env deployment/engram-workers QUEUE_CAPACITY=200000
```

**Escalation**: If scaling doesn't help within 5min, engage senior oncall

---

### Worker Down

**Condition**: Worker not responding to health checks

**Symptoms**:
- `up{job="engram-streaming"} == 0`
- Reduced total throughput
- Increased load on remaining workers

**Investigation**:
```bash
# Check worker status
kubectl describe pod <worker-pod-name>

# Get recent logs (before crash)
kubectl logs <worker-pod-name> --previous --tail=200

# Check resource constraints
kubectl get pod <worker-pod-name> -o jsonpath='{.status.containerStatuses[0].state}'

# Check node health
kubectl describe node <node-name>
```

**Common Causes**:
1. OOM kill (check `dmesg | grep -i kill`)
2. Panic/unhandled exception (check logs)
3. Deadlock (rare with lock-free design)
4. Node failure (check node status)

**Resolution**:
```bash
# Restart worker pod
kubectl delete pod <worker-pod-name>  # Will auto-recreate

# If persistent, rollback deployment
kubectl rollout undo deployment/engram-workers

# Scale temporarily to compensate
kubectl scale deployment engram-workers --replicas=12
```

---

### High Observation Latency

**Condition**: P99 latency > 100ms for >5min

**Symptoms**:
- Slow client responses
- Increasing queue depth
- User complaints about lag

**Investigation**:
1. Check worker utilization (saturated?)
2. Check queue depth (backpressure?)
3. Profile HNSW insertion times
4. Check CPU throttling
5. Review HNSW index sizes

**Resolution**:
```bash
# If worker saturation:
kubectl scale deployment engram-workers --replicas=16

# If HNSW performance:
# Review index sizes
kubectl exec -it <worker-pod> -- curl localhost:9090/metrics | grep hnsw_index_size

# If CPU throttling:
kubectl set resources deployment engram-workers --requests=cpu=4000m --limits=cpu=8000m
```

**Optimization**:
- Tune HNSW parameters (M, ef_construction)
- Enable SIMD acceleration
- Consider space-based index sharding

---

### Frequent Backpressure

**Condition**: > 100 activations/sec for >5min

**Symptoms**:
- Client throttling
- Bursty latencies
- Unstable performance

**Investigation**:
1. Is this sustained load or burst?
2. Are workers scaled appropriately for load?
3. Is work stealing helping or hurting?

**Resolution**:
```bash
# Scale workers
kubectl scale deployment engram-workers --replicas=20

# Tune queue capacity
kubectl set env deployment/engram-workers \
  NORMAL_QUEUE_CAPACITY=150000

# Adjust work stealing threshold
kubectl set env deployment/engram-workers \
  WORK_STEAL_THRESHOLD=2000
```

**Prevention**:
- Set up predictive autoscaling
- Pre-scale before daily peak hours
- Implement client-side rate limiting

---

### Batch Insertion Failures

**Condition**: Any non-zero batch failures

**Severity**: CRITICAL (data loss risk)

**Symptoms**:
- `engram_streaming_batch_failures_total` > 0
- Missing observations in recall
- Data inconsistencies

**Investigation**:
1. Check worker error logs immediately
2. Verify HNSW index integrity
3. Check file system health
4. Review memory availability

**Resolution**:
```bash
# Get error details from logs
kubectl logs -l app=engram-worker --tail=500 | grep -i "error\|fail"

# Check disk space
kubectl exec <worker-pod> -- df -h

# Verify index files
kubectl exec <worker-pod> -- ls -lh /data/hnsw/

# If corruption suspected, rebuild index
# (Follow disaster recovery procedures)
```

**Escalation**: IMMEDIATE escalation to senior oncall and engineering lead

---

## Dashboard Guide

### Panel Descriptions

#### 1. System Overview
**What**: High-level KPIs (obs/sec, queue depth, active sessions)
**Normal**: Steady observation rate, queue depth < 50K, sessions stable
**Warning**: Queue depth > 80K, sudden session drop
**Action**: Drill into specific panels for details

#### 2. Observation Rate by Space
**What**: Time series of observations/sec per memory space
**Normal**: Smooth curves, predictable patterns
**Warning**: Sudden spikes or drops, one space dominating
**Action**: Identify hot spaces, consider rebalancing

#### 3. Queue Depth by Priority
**What**: Gauge showing high/normal/low queue depths
**Normal**: Green (< 8K for all priorities)
**Warning**: Yellow (> 8K), approach red zone carefully
**Action**: Scale workers if approaching limits

#### 4. Worker Utilization Heatmap
**What**: Heat visualization of worker CPU usage
**Normal**: Even distribution, 40-80% range
**Warning**: Red zones (>80%), or large imbalance
**Action**: Scale workers or investigate load distribution

#### 5. Observation Latency Distribution
**What**: P50/P99/P99.9 latency time series
**Normal**: P99 < 100ms, low variance
**Warning**: P99 > 100ms, increasing trend
**Action**: Investigate bottlenecks, consider scaling

#### 6-7. Backpressure Panels
**What**: Activation rate and state timeline
**Normal**: Minimal activations, mostly "Normal" state
**Warning**: Frequent activations, elevated states
**Action**: Scale workers, review client behavior

#### 8. Recall Latency
**What**: P99 recall query latency
**Normal**: < 50ms
**Warning**: > 50ms, increasing trend
**Action**: Review HNSW index sizes, tune ef_search

#### 9. Batch Processing Stats
**What**: Batches/sec per worker, work stealing rate
**Normal**: Even distribution, low stealing (< 20%)
**Warning**: Uneven distribution, high stealing (> 30%)
**Action**: Review space-to-worker hashing

#### 10. Batch Size Distribution
**What**: Histogram of actual batch sizes used
**Normal**: Adaptive sizing (small batches at low load, large at high)
**Warning**: Always small (not scaling) or always large (always overloaded)
**Action**: Verify adaptive batching is working

#### 11. Error Rate
**What**: Batch failures and rejections per minute
**Normal**: Zero
**Warning**: Any non-zero value
**Action**: Immediate investigation

---

## Troubleshooting Workflows

### Workflow 1: Performance Degradation

**Symptom**: Users report slow streaming responses

**Steps**:
1. Open Grafana dashboard
2. Check System Overview panel
   - Is queue depth high? → Go to [High Queue Depth](#high-queue-depth-warning)
   - Is latency high? → Continue to step 3
3. Check Observation Latency panel
   - P99 > 100ms? → Investigate worker saturation
   - P50 also high? → Systemic bottleneck
4. Check Worker Utilization
   - All workers > 80%? → Scale workers
   - One worker stuck? → Restart that worker
   - Uneven distribution? → Review space assignment
5. Check Backpressure State
   - Elevated? → System overloaded, scale NOW
   - Normal? → Issue is elsewhere (HNSW tuning, CPU limits)

### Workflow 2: Client Connection Issues

**Symptom**: Clients report connection failures or rejections

**Steps**:
1. Check Active Sessions gauge
   - Dropping? → Network or server issue
   - Stable? → Continue to step 2
2. Check Error Rate panel
   - High rejections? → Queue overload (admission control active)
   - Zero? → Issue is client-side
3. Check rejection reasons
   ```promql
   sum(rate(engram_streaming_observations_rejected_total[5m])) by (reason)
   ```
4. Resolution based on reason:
   - `over_capacity`: Scale workers, increase queue size
   - `invalid_sequence`: Client bug (sequence gaps/duplicates)
   - `duplicate`: Retry logic issue on client

### Workflow 3: Data Loss Investigation

**Symptom**: Observations missing from recall

**Steps**:
1. Check batch failure rate
   ```promql
   sum(rate(engram_streaming_batch_failures_total[5m]))
   ```
2. If > 0: CRITICAL PATH
   - Get error logs from all workers
   - Check HNSW index integrity
   - Engage engineering team
3. If = 0: Check admission control
   - Were observations rejected?
   - Check rejection counter by reason
4. Verify observation was actually sent
   - Check client logs
   - Verify sequence number ranges

---

## Capacity Planning

### Determining Worker Count

**Formula**:
```
workers = ceil(target_throughput_ops_per_sec / 12500)
```

Where 12,500 ops/sec is the tested single-worker capacity.

**Examples**:
- 100K obs/sec → 8 workers
- 250K obs/sec → 20 workers
- 1M obs/sec → 80 workers

**Headroom**: Add 20-30% overhead for burst capacity:
```
workers = ceil(target_throughput * 1.25 / 12500)
```

### Queue Capacity Sizing

**Formula**:
```
queue_capacity = worker_count * 12500 * desired_buffer_seconds
```

**Recommendations**:
- Low latency focus: 10-second buffer
- High throughput focus: 30-second buffer
- Burst tolerance: 60-second buffer

**Example** (8 workers, 30-second buffer):
```
queue_capacity = 8 * 12500 * 30 = 3,000,000 observations
```

**Memory requirements**: ~100 bytes per observation
```
queue_memory = 3,000,000 * 100 = 300 MB
```

### Scaling Triggers

**Horizontal Scaling (Add Workers)**:
- Worker utilization > 80% for >5min
- P99 latency > 100ms for >10min
- Queue depth > 70% for >10min
- Sustained backpressure (>1min in Warning/Critical)

**Vertical Scaling (More CPU/Memory)**:
- CPU throttling detected (check `container_cpu_cfs_throttled_seconds_total`)
- Memory pressure (approaching pod limits)
- HNSW index size growing (larger indices need more RAM)

### Cost Optimization

**When to Scale Down**:
- Worker utilization < 30% for >1 hour
- Queue consistently empty
- Off-peak hours (if workload is predictable)

**Autoscaling Configuration** (Kubernetes HPA):
```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: engram-workers-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: engram-workers
  minReplicas: 4
  maxReplicas: 32
  metrics:
  - type: Pods
    pods:
      metric:
        name: engram_streaming_worker_utilization
      target:
        type: AverageValue
        averageValue: "70"
  - type: Pods
    pods:
      metric:
        name: engram_streaming_queue_depth
      target:
        type: AverageValue
        averageValue: "50000"
  behavior:
    scaleDown:
      stabilizationWindowSeconds: 300
      policies:
      - type: Percent
        value: 50
        periodSeconds: 60
    scaleUp:
      stabilizationWindowSeconds: 0
      policies:
      - type: Percent
        value: 100
        periodSeconds: 30
      - type: Pods
        value: 4
        periodSeconds: 30
      selectPolicy: Max
```

---

## Related Documentation

- [Streaming Architecture Overview](../architecture/streaming.md)
- [Performance Tuning Guide](./streaming-tuning.md)
- [Disaster Recovery Procedures](./disaster-recovery.md)
- [Client Integration Guide](../guides/streaming-client.md)

---

**Document Version**: 1.0
**Last Updated**: 2025-10-31
**Maintained By**: Engram Platform Team
**Feedback**: [Submit Issue](https://github.com/engram/engram/issues/new?labels=docs)
