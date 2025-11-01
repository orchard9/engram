# Task 012: Operational Runbook and Production Validation

**Status**: Pending
**Estimated Duration**: 2 days
**Dependencies**: Tasks 001-011 complete
**Owner**: TBD

## Objective

Document comprehensive operational procedures for distributed Engram clusters and validate production-readiness through rigorous load testing, partition recovery, and operational runbook execution by external operators. This task ensures operators can deploy, manage, troubleshoot, and scale distributed clusters without requiring deep knowledge of Engram's internals.

## Research Foundation

Production database operations require documented procedures that anticipate failure modes, not just happy paths. CockroachDB's operational runbooks distinguish between routine maintenance (rolling upgrades, node additions) and emergency procedures (partition recovery, split-brain resolution). Cassandra's production guides emphasize capacity planning with specific thresholds (20% imbalance triggers rebalancing). Modern SLO-based monitoring (Google SRE) shifts from infrastructure metrics to user-impacting service levels - latency percentiles, error budgets, multi-window burn rates.

**Operational maturity levels:**
1. **Level 1 (Basic):** Manual procedures, ad-hoc troubleshooting, reactive monitoring
2. **Level 2 (Managed):** Documented runbooks, proactive alerts, basic automation
3. **Level 3 (Production-Grade):** Automated remediation, SLO-driven alerts, capacity forecasting
4. **Level 4 (Self-Healing):** Autonomous rebalancing, predictive failure detection, zero-touch recovery

Engram targets Level 3 for M14: comprehensive runbooks with automation hooks, SLO-based alerting, validated recovery procedures.

**Key operational challenges in distributed databases:**

1. **Cluster Deployment Complexity:** Single-node to 3-node to N-node expansion requires careful sequencing. CockroachDB's approach: seed nodes form initial quorum, new nodes join via seed list, automatic rebalancing triggers when imbalance exceeds threshold (default 20%). Engram adaptation: SWIM membership handles join/leave, consistent hashing ensures even space distribution, Merkle tree sync detects divergence.

2. **Network Partition Recovery:** Partitions are inevitable (median time between network failures in AWS: 2.8 hours, Google Cloud: 1.1 hours). Recovery procedures must prevent split-brain, ensure deterministic merge, validate convergence. Yugabyte's fast failover: detect partition via gossip timeout (3s), promote replica in majority partition (<5s), reject writes in minority partition, merge state on heal using timestamp-based conflict resolution.

3. **Backup and Restore in Distributed Mode:** Consistent snapshots across nodes require coordination. Oracle's distributed backup strategy: create global SCN (System Change Number) across all nodes, take per-node snapshots at that SCN, store mapping for restoration. Point-in-time recovery: combine base snapshot with WAL replay up to target timestamp, verify consistency via checksum trees.

4. **Rolling Upgrades:** Zero-downtime upgrades require version compatibility. CockroachDB's approach: N and N+1 versions must interoperate, drain node before upgrade (reject new writes, complete in-flight), upgrade one node at a time, verify health before proceeding. Estimated time: 4-5 minutes per node.

5. **Capacity Planning and Rebalancing:** Fixed partitioning (10N partitions for N nodes) allows flexible node addition. ClickHouse's rebalancing: track per-node load (CPU, disk, memory), move partitions when imbalance exceeds 20%, rate-limit data movement to <30% network bandwidth, verify data integrity post-move.

**Proven SLO frameworks:**

Google's multi-window, multi-burn-rate alerting eliminates 90% of false positives:
- **Fast burn (2% budget in 1 hour):** SEV1 critical, page immediately, 5-minute response SLA
- **Medium burn (5% budget in 6 hours):** SEV2 high, ticket + notification, 30-minute response SLA
- **Slow burn (10% budget in 3 days):** SEV3 medium, ticket only, next business day response SLA

Metrics that matter (Google SRE Golden Signals):
1. **Latency:** Request duration distribution (P50, P90, P99)
2. **Traffic:** Requests per second
3. **Errors:** Error rate as percentage of requests
4. **Saturation:** Resource utilization (CPU, memory, disk I/O)

**Load testing methodology (YCSB best practices):**

Workload characteristics:
- **Uniform distribution:** Every key equally likely (unrealistic but reproducible)
- **Zipfian distribution:** 80/20 rule, mirrors real-world access patterns
- **Latest distribution:** Recently inserted keys accessed more often

Test phases:
1. **Load phase:** Populate database with initial dataset
2. **Run phase:** Execute mixed workload (read/write ratio)
3. **Ramp phase:** Gradually increase load to find breaking point

Success criteria:
- **Sustained throughput:** Maintain target ops/sec for 1 hour without degradation
- **Latency SLOs:** P99 <100ms at target load
- **Error rate:** <0.1% under normal conditions
- **Recovery time:** <60s after node failure

**Engram-specific considerations:**

Unlike traditional databases, Engram's cognitive dynamics introduce unique operational requirements:
- **Consolidation convergence:** Gossip protocol must converge within 60 seconds after partition heal
- **Confidence degradation:** Partial failures reduce confidence, don't block operations
- **Activation spreading:** Distributed queries use scatter-gather with timeout (5s default)
- **Memory space isolation:** Multi-tenant workloads require per-space SLOs

## Technical Specification

### Runbook Structure

Each operational procedure follows a standardized format to minimize operator error and enable measurement of Mean Time to Recovery (MTTR):

```markdown
## Procedure: [Name]

**When to Use:** [Triggering conditions or scenarios]
**Prerequisites:** [Required state, tools, access levels]
**Estimated Duration:** [Expected completion time]
**Risk Level:** [Low/Medium/High - impact of failure]
**Rollback Plan:** [How to undo if procedure fails]

### Steps

1. **[Action Verb] [Object]**
   ```bash
   # Specific command with inline comments
   command --flag value
   ```
   Expected output: [What success looks like]
   Failure mode: [What error indicates, how to recover]

2. **Verify [Outcome]**
   ```bash
   # Validation command
   check-status | grep "expected-state"
   ```
   Success criteria: [Specific threshold or value]

### Troubleshooting

| Symptom | Cause | Resolution |
|---------|-------|------------|
| ... | ... | ... |

### Success Metrics

- Time to complete: <X minutes
- Zero errors in logs
- All health checks pass
```

### Core Operational Procedures

#### 1. Cluster Deployment

**1.1 Single-Node to Distributed Migration**

Moving from single-node to distributed cluster without downtime:

```bash
# Step 1: Verify single-node health
./scripts/engram_diagnostics.sh --validate-baseline
# Expected: All metrics within normal ranges, consolidation stable

# Step 2: Enable cluster mode on existing node (becomes seed node)
# Edit config/engram.toml:
[cluster]
enabled = true
node_id = "node-1-seed"

[cluster.swim]
bind_addr = "10.0.1.10:7946"

[cluster.seed_nodes]
addrs = []  # Empty for first node

# Step 3: Restart with cluster mode
systemctl restart engram
# Wait 30s for SWIM to initialize

# Step 4: Verify cluster membership
curl http://localhost:7432/api/v1/cluster/members | jq .
# Expected: {"members": [{"id": "node-1-seed", "state": "alive"}]}

# Step 5: Add second node
# On node-2, edit config/engram.toml:
[cluster]
enabled = true
node_id = "node-2"

[cluster.swim]
bind_addr = "10.0.1.11:7946"

[cluster.seed_nodes]
addrs = ["10.0.1.10:7946"]

# Start node-2
systemctl start engram

# Step 6: Wait for gossip convergence (5-10 seconds)
# On node-1:
watch -n1 'curl -s http://localhost:7432/api/v1/cluster/members | jq ".members | length"'
# Expected: 2 after convergence

# Step 7: Trigger space assignment rebalancing
curl -X POST http://localhost:7432/api/v1/cluster/rebalance
# Expected: {"status": "rebalancing", "estimated_duration_seconds": 300}

# Step 8: Monitor rebalancing progress
watch -n5 'curl -s http://localhost:7432/api/v1/cluster/balance | jq .'
# Expected: imbalance <0.20 (20%) when complete

# Step 9: Verify no data loss
./scripts/validate_cluster_consistency.sh
# Expected: Zero divergence across nodes

# Step 10: Add third node (repeat steps 5-9)
```

**1.2 Fresh 3-Node Cluster Deployment**

Bootstrap a new cluster from scratch:

```bash
# Step 1: Deploy infrastructure (Kubernetes example)
kubectl apply -f deployments/kubernetes/distributed/

# Step 2: Wait for all pods ready
kubectl wait --for=condition=ready pod -l app=engram --timeout=300s

# Step 3: Verify SWIM membership
kubectl exec engram-0 -- curl http://localhost:7432/api/v1/cluster/members
# Expected: 3 nodes, all state=alive

# Step 4: Initialize space assignments
kubectl exec engram-0 -- curl -X POST http://localhost:7432/api/v1/cluster/initialize
# Expected: {"status": "initialized", "spaces_assigned": 0, "nodes": 3}

# Step 5: Validate cluster health
./scripts/cluster_health_check.sh
# Expected: All checks pass, no warnings
```

**1.3 Scaling to N Nodes**

Add capacity by expanding to N nodes:

```bash
# Step 1: Determine target node count based on capacity plan
# Rule: Add node when any metric exceeds 70% sustained for 1 hour
# - CPU >70%
# - Memory >70%
# - Disk I/O wait >30%
# - Average space imbalance >15%

# Step 2: Add node N to cluster
# Deploy new node with seed_nodes pointing to existing cluster

# Step 3: Wait for membership convergence
timeout 30 bash -c 'until curl -sf http://new-node:7432/api/v1/cluster/members | jq -e ".members | length == N"; do sleep 1; done'

# Step 4: Trigger automatic rebalancing
# Rebalancing starts automatically when new node joins
# Monitor progress:
curl http://localhost:7432/api/v1/cluster/rebalance/status
# Expected: {"state": "in_progress", "partitions_moved": X, "partitions_remaining": Y}

# Step 5: Verify balanced distribution
curl http://localhost:7432/api/v1/cluster/balance | jq .
# Expected: max_imbalance <0.20
```

#### 2. Node Management

**2.1 Adding a Node (Capacity Expansion)**

```bash
# Prerequisites:
# - Cluster is healthy (all nodes alive)
# - Target node meets minimum specs (4 CPU, 8GB RAM, 100GB disk)
# - Network connectivity verified (ping, SWIM port 7946 open)

# Step 1: Capacity planning validation
./scripts/capacity_planner.sh --current-load --target-nodes $((CURRENT + 1))
# Expected: Projected imbalance <20%, sufficient headroom

# Step 2: Deploy new node
# See section 1.3 for detailed steps

# Step 3: Monitor rebalancing impact on latency
# Acceptable: P99 latency increase <50% during rebalancing
watch -n10 'curl -s http://localhost:7432/metrics | grep engram_memory_operation_duration_seconds | grep quantile=\"0.99\"'

# Step 4: Verify rebalancing completes within SLA
# SLA: Rebalancing completes within 10 minutes per 100GB data
# Monitor progress:
START=$(date +%s)
until curl -s http://localhost:7432/api/v1/cluster/rebalance/status | jq -e '.state == "complete"'; do
  ELAPSED=$(($(date +%s) - START))
  if [ $ELAPSED -gt 600 ]; then
    echo "ERROR: Rebalancing exceeded 10-minute SLA"
    exit 1
  fi
  sleep 10
done

# Step 5: Validate even distribution
./scripts/validate_cluster_balance.sh --threshold 0.20
# Expected: PASS (imbalance <20%)
```

**2.2 Removing a Node (Graceful Decommission)**

```bash
# Prerequisites:
# - Cluster has N>3 nodes (don't reduce below minimum quorum)
# - Target node is not seed node (or migrate seed first)
# - Sufficient capacity on remaining nodes

# Step 1: Pre-decommission validation
REMAINING=$((NODES - 1))
./scripts/capacity_planner.sh --current-load --target-nodes $REMAINING
# Expected: Remaining nodes can handle load

# Step 2: Mark node for decommission (prevents new assignments)
curl -X POST http://localhost:7432/api/v1/cluster/nodes/$NODE_ID/decommission
# Expected: {"status": "draining"}

# Step 3: Wait for data migration (automatic)
# Primary space assignments move to replicas
# Replica assignments move to other nodes
watch -n5 'curl -s http://localhost:7432/api/v1/cluster/nodes/$NODE_ID | jq .primary_spaces'
# Expected: [] (empty array when drained)

# Step 4: Verify replicas promoted
# For each space previously on decommissioned node, verify new primary assigned
./scripts/validate_replica_promotion.sh --node $NODE_ID
# Expected: All spaces have new primary

# Step 5: Graceful shutdown
curl -X POST http://localhost:7432/api/v1/cluster/nodes/$NODE_ID/shutdown
# Node performs final gossip update (state=Left), flushes WAL, exits
# Wait for clean exit (timeout 30s)

# Step 6: Remove from infrastructure
# Kubernetes: kubectl delete pod engram-N
# systemd: systemctl stop engram@node-N
# Docker: docker stop engram-node-N

# Step 7: Verify cluster health
./scripts/cluster_health_check.sh
# Expected: Remaining nodes healthy, no orphaned spaces
```

**2.3 Handling Unplanned Node Failure**

```bash
# Scenario: Node crashes or becomes unreachable

# Step 1: Detection (automatic via SWIM)
# SWIM marks node as Suspect after probe timeout (500ms + indirect probes)
# After suspect_timeout (5s), node marked Dead

# Step 2: Monitor automatic failover
# System automatically promotes replicas to primary for affected spaces
# Timeline: <5 seconds from Dead status to replica promotion

# Step 3: Verify failover completed
curl http://localhost:7432/api/v1/cluster/spaces | jq '.spaces[] | select(.primary_node == "failed-node-id")'
# Expected: [] (no spaces should have failed node as primary)

# Step 4: Check for under-replicated spaces
curl http://localhost:7432/api/v1/cluster/replication/status | jq '.under_replicated'
# Expected: List of spaces with <N replicas

# Step 5: Trigger re-replication if needed
curl -X POST http://localhost:7432/api/v1/cluster/replication/repair
# Expected: {"status": "repairing", "spaces_affected": X}

# Step 6: Decide: repair vs replace
# Repair if: Transient failure (network blip, kernel panic)
# Replace if: Hardware failure, permanent network partition

# Step 7a: If repairing - wait for node recovery
# SWIM will detect node alive again, mark as Alive
# Automatic reconciliation via gossip protocol

# Step 7b: If replacing - add new node
# See section 2.1 (Adding a Node)
```

#### 3. Network Partition Handling

**3.1 Partition Detection**

```bash
# Automatic detection via SWIM protocol:
# - Nodes in minority partition marked Dead by majority
# - Nodes in majority partition continue operating
# - Minority partition enters read-only mode (automatic)

# Manual verification:
# On suspected minority node:
curl http://localhost:7432/api/v1/cluster/partition/status
# Expected: {"partitioned": true, "is_majority": false, "mode": "read_only"}

# On majority node:
curl http://localhost:7432/api/v1/cluster/partition/status
# Expected: {"partitioned": true, "is_majority": true, "mode": "read_write"}
```

**3.2 Partition Recovery (Automatic)**

```bash
# Step 1: Network heals (automatic in most cases)
# Nodes regain connectivity

# Step 2: SWIM detects healing
# Nodes receive Alive gossip messages from previously Dead nodes
# Both sides detect partition resolved

# Step 3: Gossip-based state merge
# Merkle tree comparison detects divergent consolidation state
# Delta synchronization exchanges only different data
# Conflict resolution via vector clocks + confidence voting

# Step 4: Monitor convergence
watch -n2 'curl -s http://localhost:7432/api/v1/cluster/convergence | jq .'
# Expected: divergence_count decreases to 0 within 60 seconds

# Step 5: Validate consistency
./scripts/validate_cluster_consistency.sh
# Expected: Zero inconsistencies after convergence
```

**3.3 Partition Recovery (Manual Intervention)**

```bash
# Scenario: Automatic merge conflicts require operator decision

# Step 1: Identify conflicting consolidations
curl http://localhost:7432/api/v1/cluster/conflicts | jq .
# Example: [
#   {
#     "space_id": "tenant-a",
#     "type": "consolidation_divergence",
#     "side_a": {"node": "node-1", "pattern_count": 15, "confidence": 0.85},
#     "side_b": {"node": "node-2", "pattern_count": 12, "confidence": 0.78}
#   }
# ]

# Step 2: Choose resolution strategy
# Options:
# - "higher_confidence": Keep side with higher confidence (default)
# - "merge": Combine both patterns with reduced confidence
# - "manual": Inspect and resolve case-by-case

# Step 3: Apply resolution
curl -X POST http://localhost:7432/api/v1/cluster/conflicts/resolve \
  -H "Content-Type: application/json" \
  -d '{"strategy": "higher_confidence"}'
# Expected: {"resolved": 5, "remaining": 0}

# Step 4: Force re-convergence
curl -X POST http://localhost:7432/api/v1/cluster/gossip/full-sync
# Expected: {"status": "syncing"}

# Step 5: Wait for convergence
timeout 120 bash -c 'until curl -s http://localhost:7432/api/v1/cluster/convergence | jq -e ".converged == true"; do sleep 2; done'

# Step 6: Verify no data loss
./scripts/validate_partition_recovery.sh --before $PARTITION_START --after $(date +%s)
# Expected: All writes acknowledged before partition are present
```

#### 4. Backup and Restore in Distributed Mode

**4.1 Consistent Snapshot Backup**

```bash
# Step 1: Initiate distributed snapshot
# Coordinator node creates global snapshot ID and timestamp
curl -X POST http://localhost:7432/api/v1/cluster/backup/snapshot \
  -H "Content-Type: application/json" \
  -d '{"description": "pre-upgrade-backup"}'
# Expected: {
#   "snapshot_id": "snap_2025_11_01_12_34_56",
#   "timestamp": 1730467296.123,
#   "nodes": 5
# }

# Step 2: Wait for all nodes to complete local snapshots
# Each node takes snapshot of its data at the global timestamp
watch -n2 'curl -s http://localhost:7432/api/v1/cluster/backup/snapshot/$SNAPSHOT_ID/status | jq .'
# Expected: {"status": "complete", "nodes_completed": 5, "nodes_total": 5}

# Step 3: Verify snapshot consistency
./scripts/validate_backup_consistency.sh --snapshot $SNAPSHOT_ID
# Expected: Merkle tree checksums match across all nodes for shared spaces

# Step 4: Export to external storage
for NODE in node-1 node-2 node-3; do
  ssh $NODE "engram backup export --snapshot $SNAPSHOT_ID --destination s3://backups/engram/$SNAPSHOT_ID/$NODE/"
done

# Step 5: Validate backup integrity
aws s3 ls s3://backups/engram/$SNAPSHOT_ID/ --recursive | wc -l
# Expected: >0 files, total size matches expectation
```

**4.2 Point-in-Time Recovery**

```bash
# Scenario: Restore cluster to specific timestamp (e.g., before data corruption)

# Step 1: Identify target timestamp
TARGET_TIME="2025-11-01T12:00:00Z"
UNIX_TIME=$(date -d "$TARGET_TIME" +%s)

# Step 2: Find nearest snapshot before target time
curl http://localhost:7432/api/v1/cluster/backup/snapshots | jq ".[] | select(.timestamp < $UNIX_TIME) | .snapshot_id" | head -1
# Example: "snap_2025_11_01_11_30_00"

# Step 3: Stop cluster (to prevent concurrent writes during restore)
for NODE in node-1 node-2 node-3; do
  curl -X POST http://$NODE:7432/api/v1/shutdown
done

# Step 4: Restore base snapshot on each node
for NODE in node-1 node-2 node-3; do
  ssh $NODE "engram restore --snapshot $SNAPSHOT_ID --source s3://backups/engram/$SNAPSHOT_ID/$NODE/ --skip-wal"
done

# Step 5: Replay WAL entries from snapshot time to target time
for NODE in node-1 node-2 node-3; do
  ssh $NODE "engram wal-replay --from $SNAPSHOT_TIME --to $UNIX_TIME"
done

# Step 6: Restart cluster
for NODE in node-1 node-2 node-3; do
  ssh $NODE "systemctl start engram"
done

# Step 7: Wait for cluster convergence
sleep 30
./scripts/cluster_health_check.sh
# Expected: All nodes alive, cluster converged

# Step 8: Verify data restored to target time
./scripts/validate_point_in_time.sh --target-time $UNIX_TIME
# Expected: Data matches expected state at target time
```

**4.3 Automated Backup Schedule**

```bash
# Configure automated backups via cron or Kubernetes CronJob

# Cron example (run on coordinator node):
# Daily full snapshot at 2 AM
0 2 * * * /usr/local/bin/engram-backup.sh full

# Hourly incremental WAL backup
0 * * * * /usr/local/bin/engram-backup.sh incremental

# Backup script (/usr/local/bin/engram-backup.sh):
#!/bin/bash
set -euo pipefail

MODE=$1  # full or incremental

if [ "$MODE" == "full" ]; then
  SNAPSHOT_ID=$(curl -s -X POST http://localhost:7432/api/v1/cluster/backup/snapshot | jq -r .snapshot_id)
  echo "Created snapshot: $SNAPSHOT_ID"

  # Wait for completion
  until curl -s http://localhost:7432/api/v1/cluster/backup/snapshot/$SNAPSHOT_ID/status | jq -e '.status == "complete"'; do
    sleep 5
  done

  # Export to S3
  for NODE in $(curl -s http://localhost:7432/api/v1/cluster/members | jq -r '.members[].id'); do
    engram backup export --snapshot $SNAPSHOT_ID --destination s3://backups/engram/$SNAPSHOT_ID/$NODE/
  done

elif [ "$MODE" == "incremental" ]; then
  # Backup WAL files written in last hour
  engram wal-backup --since "1 hour ago" --destination s3://backups/engram/wal/
fi

# Prune old backups (keep 30 days)
aws s3 rm s3://backups/engram/ --recursive --exclude "*" --include "snap_*" --older-than 30d
```

#### 5. Rolling Upgrades (Zero-Downtime)

```bash
# Prerequisites:
# - Version N+1 is compatible with version N (check release notes)
# - Backup completed before upgrade
# - Cluster is healthy (all nodes alive)

# Step 1: Verify version compatibility
./scripts/check_version_compatibility.sh --from $CURRENT_VERSION --to $TARGET_VERSION
# Expected: COMPATIBLE

# Step 2: Upgrade one node at a time (start with non-seed node)
for NODE in node-3 node-2 node-1; do
  echo "Upgrading $NODE..."

  # 2a. Drain node (reject new primary assignments)
  curl -X POST http://$NODE:7432/api/v1/cluster/nodes/$NODE/drain

  # 2b. Wait for in-flight operations to complete (timeout 60s)
  timeout 60 bash -c "until curl -s http://$NODE:7432/api/v1/cluster/nodes/$NODE/drain/status | jq -e '.in_flight_operations == 0'; do sleep 1; done"

  # 2c. Graceful shutdown
  curl -X POST http://$NODE:7432/api/v1/shutdown

  # 2d. Wait for process exit
  ssh $NODE "timeout 30 bash -c 'until ! pgrep engram >/dev/null; do sleep 1; done'"

  # 2e. Upgrade binary
  ssh $NODE "sudo systemctl stop engram && sudo cp /tmp/engram-$TARGET_VERSION /usr/local/bin/engram"

  # 2f. Restart with new version
  ssh $NODE "sudo systemctl start engram"

  # 2g. Wait for node to rejoin cluster
  timeout 60 bash -c "until curl -s http://localhost:7432/api/v1/cluster/members | jq -e '.members[] | select(.id == \"$NODE\") | .state == \"alive\"'; do sleep 2; done"

  # 2h. Verify node health
  curl http://$NODE:7432/health | jq -e '.status == "healthy"'

  # 2i. Re-enable primary assignments (undrain)
  curl -X POST http://$NODE:7432/api/v1/cluster/nodes/$NODE/undrain

  echo "$NODE upgraded successfully"

  # Wait 60s before upgrading next node (stability buffer)
  sleep 60
done

# Step 3: Verify cluster health after upgrade
./scripts/cluster_health_check.sh
# Expected: All nodes alive, version $TARGET_VERSION, no errors

# Step 4: Finalize upgrade (enable new features if applicable)
curl -X POST http://localhost:7432/api/v1/cluster/finalize-upgrade
# Expected: {"status": "finalized", "version": "$TARGET_VERSION"}

# Estimated total time: (4-5 minutes per node) * N nodes + buffer
# Example: 5-node cluster = ~25 minutes
```

### Monitoring and Alerting Strategy

#### SLO Definitions

**Service Level Indicators (SLIs):**

1. **Availability SLI**
   - Measurement: Successful health check responses / Total health checks
   - Target: 99.9% (43 minutes downtime per month)
   - Implementation: `engram_up{job="engram"}` from Prometheus blackbox exporter

2. **Latency SLI (Memory Operations)**
   - Measurement: P99 latency for store/recall/delete operations
   - Target: <100ms (cognitive plausibility threshold)
   - Implementation: `histogram_quantile(0.99, engram_memory_operation_duration_seconds)`

3. **Error Rate SLI**
   - Measurement: Failed operations / Total operations
   - Target: <0.1% (999 successes per 1000 requests)
   - Implementation: `rate(engram_operation_errors_total[5m]) / rate(engram_operations_total[5m])`

4. **Consolidation Freshness SLI**
   - Measurement: Time since last successful consolidation
   - Target: <450s (1.5x scheduler interval)
   - Implementation: `time() - engram_consolidation_last_success_timestamp_seconds`

5. **Replication Lag SLI (Distributed Mode)**
   - Measurement: Time delay between primary write and replica acknowledgment
   - Target: <1s under normal load
   - Implementation: `engram_replication_lag_seconds{quantile="0.99"}`

**Service Level Objectives (SLOs):**

| SLO | SLI | Target | Time Window | Error Budget |
|-----|-----|--------|-------------|--------------|
| Availability | Health checks | 99.9% | 30 days | 43m/month |
| Read Latency | P99 recall duration | <100ms | 7 days | 1% requests |
| Write Latency | P99 store duration | <50ms | 7 days | 1% requests |
| Error Rate | Operation failures | <0.1% | 30 days | 43k errors/30M ops |
| Consolidation | Freshness | <450s | 24 hours | 5 stale periods |
| Replication Lag | P99 lag | <1s | 7 days | 1% slow replicas |

#### Alert Thresholds and Severity Levels

**Multi-Window Multi-Burn-Rate Alerts:**

```yaml
# prometheus/alerts.yml

groups:
  - name: engram_availability_slo
    interval: 30s
    rules:
      # Fast burn: 2% budget in 1 hour = SEV1
      - alert: EngramAvailabilityFastBurn
        expr: |
          (
            sum(rate(probe_success{job="engram"}[1h])) / sum(rate(probe_success{job="engram"}[1h]) + rate(probe_failure{job="engram"}[1h]))
          ) < 0.98  # 99.9% - 2% = 97.9%, round to 98%
        for: 5m
        labels:
          severity: critical
          slo: availability
          burn_rate: fast
        annotations:
          summary: "Engram availability SLO fast burn"
          description: "Consuming 2% error budget in 1 hour. Availability: {{ $value | humanizePercentage }}"
          runbook_url: "https://docs.engram.example/operations/runbooks/availability-slo-breach"
          response_time: "5 minutes"

      # Medium burn: 5% budget in 6 hours = SEV2
      - alert: EngramAvailabilityMediumBurn
        expr: |
          (
            sum(rate(probe_success{job="engram"}[6h])) / sum(rate(probe_success{job="engram"}[6h]) + rate(probe_failure{job="engram"}[6h]))
          ) < 0.95  # 99.9% - 5% = 94.9%, round to 95%
        for: 15m
        labels:
          severity: warning
          slo: availability
          burn_rate: medium
        annotations:
          summary: "Engram availability SLO medium burn"
          description: "Consuming 5% error budget in 6 hours. Availability: {{ $value | humanizePercentage }}"
          runbook_url: "https://docs.engram.example/operations/runbooks/availability-slo-breach"
          response_time: "30 minutes"

      # Slow burn: 10% budget in 3 days = SEV3
      - alert: EngramAvailabilitySlow Burn
        expr: |
          (
            sum(rate(probe_success{job="engram"}[3d])) / sum(rate(probe_success{job="engram"}[3d]) + rate(probe_failure{job="engram"}[3d]))
          ) < 0.90  # 99.9% - 10% = 89.9%, round to 90%
        for: 1h
        labels:
          severity: info
          slo: availability
          burn_rate: slow
        annotations:
          summary: "Engram availability SLO slow burn"
          description: "Consuming 10% error budget in 3 days. Availability: {{ $value | humanizePercentage }}"
          runbook_url: "https://docs.engram.example/operations/runbooks/availability-slo-breach"
          response_time: "next business day"

  - name: engram_latency_slo
    interval: 30s
    rules:
      - alert: EngramLatencySLOBreach
        expr: |
          histogram_quantile(0.99, rate(engram_memory_operation_duration_seconds_bucket[5m])) > 0.100
        for: 5m
        labels:
          severity: warning
          slo: latency
        annotations:
          summary: "Engram P99 latency exceeds 100ms"
          description: "P99 latency: {{ $value }}s (target: <100ms)"
          runbook_url: "https://docs.engram.example/operations/runbooks/high-latency"

  - name: engram_distributed_health
    interval: 30s
    rules:
      - alert: EngramClusterPartitioned
        expr: |
          engram_cluster_partitioned > 0
        for: 1m
        labels:
          severity: critical
        annotations:
          summary: "Engram cluster network partition detected"
          description: "Cluster is partitioned. Minority partition in read-only mode."
          runbook_url: "https://docs.engram.example/operations/runbooks/partition-recovery"

      - alert: EngramReplicationLagHigh
        expr: |
          histogram_quantile(0.99, rate(engram_replication_lag_seconds_bucket[5m])) > 1.0
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "Engram replication lag exceeds 1 second"
          description: "P99 replication lag: {{ $value }}s (target: <1s)"
          runbook_url: "https://docs.engram.example/operations/runbooks/replication-lag"

      - alert: EngramNodeDown
        expr: |
          up{job="engram"} == 0
        for: 1m
        labels:
          severity: critical
        annotations:
          summary: "Engram node is down"
          description: "Node {{ $labels.instance }} is unreachable"
          runbook_url: "https://docs.engram.example/operations/runbooks/node-failure"

      - alert: EngramGossipNotConverged
        expr: |
          engram_cluster_divergence_count > 0
        for: 2m
        labels:
          severity: warning
        annotations:
          summary: "Engram gossip protocol not converged"
          description: "{{ $value }} divergent states detected after partition heal"
          runbook_url: "https://docs.engram.example/operations/runbooks/gossip-convergence"
```

#### Grafana Dashboard Layouts

**Dashboard 1: Distributed Cluster Overview**

Panels (4x3 grid):

Row 1: Cluster Health
- Panel 1: Cluster Membership Status (stat panel)
  - Metric: `count(up{job="engram"} == 1)` vs `count(up{job="engram"})`
  - Display: "5 / 5 nodes alive" with green/red background
- Panel 2: Partition Status (stat panel)
  - Metric: `max(engram_cluster_partitioned)`
  - Display: "No partition" (green) or "PARTITIONED" (red)
- Panel 3: Rebalancing Status (gauge)
  - Metric: `engram_cluster_imbalance_ratio`
  - Thresholds: <0.15 (green), 0.15-0.20 (yellow), >0.20 (red)
  - Display: "Imbalance: 8%"
- Panel 4: SLO Compliance (stat panel)
  - Metric: `(1 - sum(rate(engram_operation_errors_total[30d])) / sum(rate(engram_operations_total[30d]))) * 100`
  - Display: "99.95% (target: 99.9%)"

Row 2: Performance Metrics
- Panel 5: Request Rate (graph)
  - Metric: `sum(rate(engram_operations_total[5m])) by (operation)`
  - Stacked area chart: store, recall, delete, query operations
- Panel 6: P99 Latency by Operation (graph)
  - Metric: `histogram_quantile(0.99, sum(rate(engram_memory_operation_duration_seconds_bucket[5m])) by (operation, le))`
  - Line chart with threshold annotation at 100ms
- Panel 7: Error Rate (graph)
  - Metric: `sum(rate(engram_operation_errors_total[5m])) by (error_type)`
  - Stacked area chart
- Panel 8: Replication Lag P99 (graph)
  - Metric: `histogram_quantile(0.99, rate(engram_replication_lag_seconds_bucket[5m]))`
  - Line chart with threshold at 1s

Row 3: Resource Utilization
- Panel 9: CPU Usage per Node (graph)
  - Metric: `100 - (avg by (instance) (rate(node_cpu_seconds_total{mode="idle"}[5m])) * 100)`
  - Stacked area chart by node
- Panel 10: Memory Usage per Node (graph)
  - Metric: `node_memory_Active_bytes / node_memory_MemTotal_bytes * 100`
  - Line chart by node
- Panel 11: Disk I/O Wait (graph)
  - Metric: `rate(node_disk_io_time_seconds_total[5m])`
  - Line chart by node
- Panel 12: Network Throughput (graph)
  - Metric: `rate(node_network_receive_bytes_total[5m]) + rate(node_network_transmit_bytes_total[5m])`
  - Stacked area: receive (blue), transmit (green)

**Dashboard 2: Operational Runbook Dashboard**

Purpose: Quick reference for on-call operators

Panels (2 columns):

Left Column: Key Metrics
- Panel 1: Health Status (stat panel, large font)
  - Green: "ALL SYSTEMS OPERATIONAL"
  - Red: "INCIDENT IN PROGRESS"
- Panel 2: Active Alerts (table)
  - Query: Prometheus alerts with severity, summary, runbook link
- Panel 3: Recent Operations (logs panel)
  - LogQL: `{job="engram"} | json | operation != "" | line_format "{{.timestamp}} [{{.operation}}] {{.message}}"`
  - Last 20 operations

Right Column: Runbook Quick Links
- Panel 4: Common Procedures (text panel)
  - Markdown with links to runbook sections:
    - [Add Node](#adding-a-node)
    - [Remove Node](#removing-a-node)
    - [Partition Recovery](#partition-recovery)
    - [Rolling Upgrade](#rolling-upgrades)
    - [Backup/Restore](#backup-and-restore)
- Panel 5: Troubleshooting Decision Tree (iframe)
  - Embedded flowchart from docs
- Panel 6: Emergency Contacts (text panel)
  - On-call rotation, escalation paths

### Troubleshooting Guide

#### Common Issues and Resolution Steps

**Issue 1: Cluster Won't Form (SWIM Membership Fails)**

Context: Nodes don't discover each other, membership remains size 1

Action:
```bash
# Check SWIM port connectivity
nc -zv <seed-node> 7946
# If fails: verify firewall rules, security groups

# Check seed node configuration
curl http://<seed-node>:7432/api/v1/cluster/config | jq .seed_nodes
# Verify seed list is correct

# Check logs for SWIM errors
journalctl -u engram -n 100 | grep -i swim
# Look for: "Failed to bind", "Connection refused", "Timeout"

# Verify node IDs are unique
for NODE in node-1 node-2 node-3; do
  curl http://$NODE:7432/api/v1/cluster/config | jq .node_id
done
# All should be different
```

Verification:
```bash
curl http://localhost:7432/api/v1/cluster/members | jq '.members | length'
# Expected: N (number of deployed nodes)
```

**Issue 2: Rebalancing Stuck or Slow**

Context: Imbalance ratio remains >20% despite rebalancing triggered

Action:
```bash
# Check rebalancing status
curl http://localhost:7432/api/v1/cluster/rebalance/status | jq .
# Look for: "state": "stuck", errors in recent_failures

# Identify stuck partitions
curl http://localhost:7432/api/v1/cluster/partitions | jq '.[] | select(.rebalancing == true) | {id, source_node, target_node, progress_pct}'

# Check network throughput between nodes
# Rebalancing requires significant data transfer
iftop -i eth0  # On nodes involved in rebalancing

# Check if target nodes have available capacity
curl http://localhost:7432/api/v1/cluster/nodes | jq '.[] | {id, disk_used_pct, memory_used_pct}'
# If any >90%, scale up resources

# Force retry stuck partitions
curl -X POST http://localhost:7432/api/v1/cluster/rebalance/retry
```

Verification:
```bash
curl http://localhost:7432/api/v1/cluster/balance | jq .max_imbalance
# Expected: <0.20
```

**Issue 3: Gossip Not Converging After Partition Heal**

Context: Divergence count remains >0 for >2 minutes after partition resolves

Action:
```bash
# Check gossip health
curl http://localhost:7432/api/v1/cluster/gossip/health | jq .
# Look for: high message_drop_rate, merkle_mismatches

# Identify divergent spaces
curl http://localhost:7432/api/v1/cluster/divergence | jq '.[] | {space_id, node_a, node_b, divergence_type}'

# Force full gossip sync (expensive, use sparingly)
curl -X POST http://localhost:7432/api/v1/cluster/gossip/full-sync

# Monitor convergence
watch -n2 'curl -s http://localhost:7432/api/v1/cluster/convergence | jq .'
```

Verification:
```bash
curl http://localhost:7432/api/v1/cluster/convergence | jq .converged
# Expected: true within 60s
```

**Issue 4: Split-Brain Detected**

Context: Two partitions both claim majority status (rare but critical)

Action:
```bash
# CRITICAL: This indicates a configuration error or SWIM bug
# DO NOT attempt automatic resolution

# Step 1: Identify partition sides
for NODE in $(cat nodes.txt); do
  echo "$NODE:"
  curl -s http://$NODE:7432/api/v1/cluster/partition/status | jq '{is_majority, members}'
done

# Step 2: Determine true majority
# Count nodes in each partition, verify against actual cluster size

# Step 3: Manually designate minority partition
# On each node in minority partition:
curl -X POST http://$MINORITY_NODE:7432/api/v1/cluster/force-minority-mode
# This forces read-only mode

# Step 4: Wait for network healing
# Step 5: Validate automatic merge occurs
# Step 6: File bug report - split-brain should never occur
```

**Issue 5: High Replication Lag**

Context: P99 replication lag >1s sustained for >5 minutes

Action:
```bash
# Identify slow replicas
curl http://localhost:7432/api/v1/cluster/replication/lag | jq '.[] | select(.lag_seconds > 1) | {space_id, replica_node, lag_seconds}'

# Check replica node health
curl http://<slow-replica>:7432/health | jq .
# Look for high CPU, disk I/O wait, memory pressure

# Check network latency between primary and replica
ping -c 10 <replica-node>
# Look for packet loss, high latency variance

# Adjust replication batch size (if lag is due to throughput)
curl -X POST http://localhost:7432/api/v1/cluster/replication/config \
  -H "Content-Type: application/json" \
  -d '{"batch_size": 50}'  # Default: 100, reduce for lower latency

# If replica persistently slow, promote different replica
curl -X POST http://localhost:7432/api/v1/cluster/spaces/<space-id>/promote-replica \
  -H "Content-Type: application/json" \
  -d '{"new_replica_node": "<faster-node>"}'
```

Verification:
```bash
curl http://localhost:7432/metrics | grep engram_replication_lag_seconds | grep quantile=\"0.99\"
# Expected: <1.0
```

### Production Validation Checklist

Execute this checklist before declaring distributed cluster production-ready:

#### Pre-Deployment Validation

- [ ] **Capacity Planning**
  - [ ] Resource requirements documented (CPU, memory, disk, network)
  - [ ] Scaling triggers defined (thresholds for adding nodes)
  - [ ] Cost model created ($/month for target workload)

- [ ] **Configuration Review**
  - [ ] All nodes use consistent SWIM parameters
  - [ ] Seed nodes identified and highly available
  - [ ] Firewall rules allow SWIM (7946), HTTP (7432), gRPC (50051)
  - [ ] TLS certificates valid for 90+ days (if TLS enabled)

- [ ] **Monitoring Setup**
  - [ ] Prometheus scraping all nodes
  - [ ] Grafana dashboards imported
  - [ ] Alert rules configured in Alertmanager
  - [ ] Runbook URLs in alert annotations
  - [ ] PagerDuty/Slack integration tested

- [ ] **Backup Configuration**
  - [ ] Automated snapshot schedule configured
  - [ ] Backup destination accessible from all nodes
  - [ ] Retention policy configured (e.g., 30 days)
  - [ ] Restore procedure tested successfully

#### Load Testing (100K ops/sec validation)

```bash
# Test 1: Baseline Single-Node Performance
./scripts/benchmark.sh --mode single-node --duration 3600 --target-ops 20000
# Expected: Sustain 20K ops/sec for 1 hour, P99 <50ms

# Test 2: Distributed Cluster Performance (5 nodes)
./scripts/benchmark.sh --mode distributed --nodes 5 --duration 3600 --target-ops 100000
# Expected: Sustain 100K ops/sec for 1 hour, P99 <100ms
# Target: 20K ops/sec per node in distributed mode

# Test 3: Scaling Efficiency
./scripts/benchmark.sh --mode scaling --min-nodes 3 --max-nodes 8 --step 1 --duration 600
# Expected: Linear scaling (2x nodes ≈ 2x throughput)

# Test 4: Mixed Workload
./scripts/benchmark.sh --mode mixed --read-pct 70 --write-pct 30 --duration 3600 --target-ops 100000
# Expected: Maintain target ops/sec, verify read-heavy workload efficiency

# Test 5: Latency Under Load
./scripts/latency_test.sh --target-ops 100000 --duration 600
# Expected: P50 <20ms, P90 <50ms, P99 <100ms, P99.9 <200ms
```

Load test success criteria:
- **Sustained throughput:** 100K ops/sec for 1 hour without degradation
- **Latency SLO:** P99 <100ms throughout test
- **Error rate:** <0.1% (100 errors per 100K operations)
- **Resource utilization:** CPU <80%, Memory <80%, no OOM kills
- **Replication lag:** P99 <1s throughout test

#### Partition Testing

```bash
# Test 1: Clean Network Split (2|3 partition)
./scripts/partition_test.sh --scenario clean-split --partition-size 2 --duration 300
# Expected: Minority partition enters read-only, majority continues, merge on heal within 60s

# Test 2: Asymmetric Partition (node A→B works, B→A fails)
./scripts/partition_test.sh --scenario asymmetric --source node-1 --target node-2 --duration 300
# Expected: SWIM detects via indirect probes, failover completes

# Test 3: Cascading Node Failures
./scripts/partition_test.sh --scenario cascading --failure-rate "1 node per 30s" --duration 300
# Expected: Cluster remains available until >50% nodes lost

# Test 4: Partition During High Load
./scripts/partition_test.sh --scenario split-under-load --ops-per-sec 50000 --duration 600
# Expected: Writes continue on majority side, reads degrade gracefully, no data loss

# Test 5: Partition Healing with Conflicting Writes
./scripts/partition_test.sh --scenario conflicting-writes --partition-duration 120
# Expected: Conflicts resolved deterministically, convergence <60s, zero data loss
```

Partition test success criteria:
- **Failover time:** Primary failure detected and replica promoted in <5s
- **Availability during partition:** Majority partition maintains >99% availability
- **Convergence after heal:** Gossip converges in <60s
- **Data integrity:** Zero acknowledged writes lost
- **Conflict resolution:** All conflicts resolved deterministically

#### Operational Runbook Validation

Execute each runbook procedure with external operator (someone unfamiliar with Engram internals):

- [ ] **Cluster Deployment**
  - [ ] External operator deploys 3-node cluster from docs successfully
  - [ ] Time to deploy: <30 minutes
  - [ ] Zero escalations required

- [ ] **Node Addition**
  - [ ] External operator adds 4th node following runbook
  - [ ] Rebalancing completes within SLA (10 min per 100GB)
  - [ ] Final imbalance <20%

- [ ] **Node Removal**
  - [ ] External operator decommissions node following runbook
  - [ ] Data migrated without loss
  - [ ] Cluster remains healthy

- [ ] **Rolling Upgrade**
  - [ ] External operator upgrades cluster from version N to N+1
  - [ ] Zero downtime (availability SLO maintained)
  - [ ] Upgrade completes in estimated time (5 min/node)

- [ ] **Partition Recovery**
  - [ ] External operator follows partition recovery runbook
  - [ ] Manual intervention steps clear and unambiguous
  - [ ] Convergence verified using provided scripts

- [ ] **Backup and Restore**
  - [ ] External operator creates snapshot backup
  - [ ] External operator performs point-in-time restore
  - [ ] Restored data matches expected state

Runbook validation success criteria:
- **Clarity:** External operator completes without asking clarifying questions
- **Completeness:** All required commands and verification steps present
- **Accuracy:** Actual outcomes match documented expectations
- **Safety:** Rollback procedures prevent data loss if procedure fails

#### Monitoring and Alerting Validation

- [ ] **Alert Firing**
  - [ ] Trigger each critical alert via chaos injection
  - [ ] Verify alert reaches PagerDuty/Slack within 1 minute
  - [ ] Verify runbook URL links to correct section

- [ ] **Dashboard Functionality**
  - [ ] All dashboard panels load without errors
  - [ ] Data appears for all metrics (no "No data" panels)
  - [ ] Time ranges adjustable without breaking queries

- [ ] **Log Aggregation**
  - [ ] Logs from all nodes appear in Loki
  - [ ] LogQL queries return expected results
  - [ ] Log retention policy enforced (30 days)

- [ ] **SLO Tracking**
  - [ ] Error budget calculation accurate
  - [ ] Burn rate alerts fire at correct thresholds
  - [ ] SLO compliance dashboard updates in real-time

## Files to Create

1. `docs/operations/distributed-deployment.md` - Complete cluster deployment guide
2. `docs/operations/cluster-management.md` - Node addition/removal procedures
3. `docs/operations/partition-handling.md` - Network partition detection and recovery
4. `docs/operations/distributed-backup-restore.md` - Backup strategies for distributed mode
5. `docs/operations/rolling-upgrades.md` - Zero-downtime upgrade procedures
6. `docs/operations/distributed-troubleshooting.md` - Troubleshooting guide with decision trees
7. `docs/operations/capacity-planning-distributed.md` - Capacity planning formulas and tooling
8. `deployments/prometheus/distributed-alerts.yml` - SLO-based alert rules
9. `deployments/grafana/dashboards/distributed-cluster-overview.json` - Cluster health dashboard
10. `deployments/grafana/dashboards/operational-runbook.json` - On-call operator dashboard
11. `scripts/cluster_health_check.sh` - Automated health validation
12. `scripts/validate_cluster_consistency.sh` - Data consistency validation
13. `scripts/validate_cluster_balance.sh` - Rebalancing verification
14. `scripts/validate_partition_recovery.sh` - Partition recovery verification
15. `scripts/capacity_planner.sh` - Capacity planning calculator
16. `scripts/load_test_distributed.sh` - 100K ops/sec load test harness

## Files to Modify

1. `docs/operations/monitoring.md` - Add distributed cluster metrics
2. `docs/operations/troubleshooting.md` - Add distributed-specific issues
3. `docs/operations/alerting.md` - Add SLO-based alert documentation
4. `deployments/prometheus/alerts.yml` - Extend with distributed alerts
5. `scripts/engram_diagnostics.sh` - Add cluster health checks
6. `README.md` - Add distributed deployment quick start

## Testing Strategy

### Unit Tests

Test individual operational scripts in isolation:

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_cluster_health_check_script() {
        // Mock 3-node cluster
        let cluster = MockCluster::new(3).await;

        // Run health check script
        let output = Command::new("./scripts/cluster_health_check.sh")
            .env("CLUSTER_NODES", cluster.node_addresses())
            .output()
            .await
            .unwrap();

        assert!(output.status.success());
        assert!(String::from_utf8_lossy(&output.stdout).contains("All checks passed"));
    }

    #[tokio::test]
    async fn test_capacity_planner_accurate() {
        let current_load = LoadMetrics {
            ops_per_sec: 50000,
            cpu_pct: 60,
            memory_pct: 55,
            disk_io_pct: 40,
        };

        let planner = CapacityPlanner::new(current_load, 3);
        let recommendation = planner.nodes_required_for_load(100000);

        // Should recommend scaling from 3 to 6 nodes (2x load)
        assert_eq!(recommendation, 6);
    }
}
```

### Integration Tests

Test runbook procedures end-to-end:

```rust
#[tokio::test]
#[ignore] // Long-running test
async fn test_runbook_add_node() {
    // Start 3-node cluster
    let cluster = IntegrationCluster::deploy(3).await;

    // Execute add-node runbook
    cluster.run_runbook("add_node.sh", &["node-4"]).await.unwrap();

    // Verify cluster has 4 nodes
    let members = cluster.get_members().await;
    assert_eq!(members.len(), 4);

    // Verify rebalancing completed
    let balance = cluster.get_balance().await;
    assert!(balance.max_imbalance < 0.20);

    // Verify no data loss
    let consistency = cluster.validate_consistency().await;
    assert_eq!(consistency.divergence_count, 0);
}

#[tokio::test]
#[ignore]
async fn test_runbook_rolling_upgrade() {
    let cluster = IntegrationCluster::deploy(3).await;

    // Insert test data
    cluster.insert_test_data(10000).await;

    // Start continuous query workload
    let workload = cluster.start_continuous_queries(1000).await; // 1K QPS

    // Execute rolling upgrade runbook
    cluster.run_runbook("rolling_upgrade.sh", &["v0.2.0"]).await.unwrap();

    // Stop workload
    workload.stop().await;

    // Verify zero downtime (availability SLO maintained)
    assert!(workload.availability() > 0.999);

    // Verify all nodes upgraded
    for node in cluster.nodes() {
        assert_eq!(node.version().await, "v0.2.0");
    }

    // Verify test data intact
    assert_eq!(cluster.count_memories().await, 10000);
}
```

### Load Testing

```bash
# scripts/load_test_distributed.sh

#!/bin/bash
set -euo pipefail

TARGET_OPS=${1:-100000}
DURATION=${2:-3600}  # 1 hour default
NODES=${3:-5}

echo "Starting distributed load test:"
echo "  Target: $TARGET_OPS ops/sec"
echo "  Duration: $DURATION seconds"
echo "  Nodes: $NODES"

# Deploy cluster
./deploy_test_cluster.sh --nodes $NODES

# Wait for cluster ready
until ./scripts/cluster_health_check.sh; do
  echo "Waiting for cluster ready..."
  sleep 5
done

# Configure load test
cat > /tmp/ycsb_workload <<EOF
recordcount=10000000
operationcount=$((TARGET_OPS * DURATION))
workload=site.ycsb.workloads.CoreWorkload
readproportion=0.70
updateproportion=0.25
insertproportion=0.05
requestdistribution=zipfian
EOF

# Run load test with YCSB
ycsb run engram \
  -P /tmp/ycsb_workload \
  -threads $((TARGET_OPS / 1000)) \
  -target $TARGET_OPS \
  -s | tee /tmp/load_test_results.txt

# Analyze results
echo ""
echo "=== Load Test Results ==="
grep "OVERALL" /tmp/load_test_results.txt
grep "READ-P99" /tmp/load_test_results.txt
grep "UPDATE-P99" /tmp/load_test_results.txt

# Verify SLOs
THROUGHPUT=$(grep "OVERALL.*Throughput" /tmp/load_test_results.txt | awk '{print $3}')
P99_LATENCY=$(grep "READ-P99" /tmp/load_test_results.txt | awk '{print $3}')

if (( $(echo "$THROUGHPUT < $TARGET_OPS * 0.95" | bc -l) )); then
  echo "FAIL: Throughput $THROUGHPUT < target $TARGET_OPS"
  exit 1
fi

if (( $(echo "$P99_LATENCY > 100" | bc -l) )); then
  echo "FAIL: P99 latency ${P99_LATENCY}ms > 100ms"
  exit 1
fi

echo "PASS: All SLOs met"
```

## Acceptance Criteria

1. **Complete Runbooks**
   - [ ] All 6 core procedures documented (deploy, add node, remove node, partition recovery, backup/restore, rolling upgrade)
   - [ ] Each procedure includes prerequisites, steps, verification, troubleshooting
   - [ ] External operator successfully executes all procedures without escalation
   - [ ] Average procedure completion time within documented estimates

2. **Monitoring and Alerting**
   - [ ] SLO definitions documented for availability, latency, error rate, consolidation, replication lag
   - [ ] Multi-window multi-burn-rate alerts configured (fast/medium/slow burn)
   - [ ] All critical alerts have runbook URL annotations
   - [ ] Grafana dashboards load without errors, all panels have data

3. **Troubleshooting Guide**
   - [ ] Decision trees for 4 major categories (service unavailability, performance, data integrity, resource exhaustion)
   - [ ] Top 5 distributed-specific issues documented with resolution steps
   - [ ] Integration with existing troubleshooting.md (common-issues.md)
   - [ ] Escalation paths defined with clear criteria

4. **Production Validation Tests**
   - [ ] Load test: 100K ops/sec sustained for 1 hour, P99 <100ms
   - [ ] Partition test: Clean split with automatic recovery in <60s, zero data loss
   - [ ] Failover test: Primary failure → replica promotion in <5s
   - [ ] Rebalancing test: Add node → even distribution in <10min per 100GB
   - [ ] Operational test: External operator deploys and manages cluster successfully

5. **Capacity Planning**
   - [ ] Formulas documented for resource requirements (CPU, memory, disk, network)
   - [ ] Scaling triggers defined (thresholds for adding/removing nodes)
   - [ ] Capacity planning script automates calculations
   - [ ] Cost model template provided

## Success Metrics

1. **Operational Maturity**
   - Mean Time to Detect (MTTD): <1 minute for critical issues (via monitoring)
   - Mean Time to Resolve (MTTR): <30 minutes for P1 issues (via runbooks)
   - Runbook coverage: 100% of operational procedures documented
   - Escalation rate: <10% (external operators resolve 90% without escalation)

2. **Performance**
   - Sustained throughput: 100K ops/sec for 1 hour without degradation
   - Latency SLO compliance: P99 <100ms for 99.9% of 1-hour windows
   - Error rate: <0.1% under normal load, <1% during failures
   - Replication lag: P99 <1s under normal load

3. **Reliability**
   - Availability: >99.9% during partition testing (minority partition read-only acceptable)
   - Failover time: <5 seconds from failure detection to replica promotion
   - Convergence time: <60 seconds after partition heal
   - Data integrity: Zero acknowledged writes lost during partition tests

4. **Usability**
   - External operator success rate: >90% complete procedures without help
   - Documentation clarity score: >4.5/5 (post-validation survey)
   - Average time to find relevant runbook section: <2 minutes
   - False positive alert rate: <5% (via multi-burn-rate filtering)

## Dependencies

This task depends on completion of:
- Task 001: SWIM membership (needed for cluster formation validation)
- Task 002: Discovery service (needed for node join procedures)
- Task 005: Replication protocol (needed for failover testing)
- Task 007: Gossip protocol (needed for partition recovery validation)
- Task 011: Jepsen testing (provides consistency validation framework)

## Next Steps

After completing this task:
- Milestone 14 is complete - distributed cluster is production-ready
- Begin real-world pilot deployments with select customers
- Collect production metrics to refine capacity planning models
- Iterate on runbooks based on operator feedback
- Plan for future enhancements (multi-region, auto-scaling, predictive failure detection)
