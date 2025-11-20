# Troubleshooting Guide

Comprehensive troubleshooting procedures for Engram production issues. This guide enables operators to resolve 80% of issues without escalation.

## Quick Start

When experiencing issues:

1. **Run diagnostic script first**: `./scripts/diagnose_health.sh`

2. **Identify issue category** using the decision trees below

3. **Follow specific resolution** from Common Issues section

4. **Verify fix** using provided verification steps

For SEV1 incidents, immediately collect debug information:

```bash
./scripts/collect_debug_info.sh

```

## Decision Trees

### Decision Tree 1: Service Unavailability

```
Service Unavailable (HTTP 502/503/timeout)
│
├─ Process not running?
│  ├─ YES → Check logs for startup errors
│  │        ├─ "Failed to bind" → Port conflict (Issue #1)
│  │        ├─ "Permission denied" → Fix data directory permissions (Issue #1)
│  │        └─ "Cannot deserialize WAL" → Data corruption (Issue #5)
│  └─ NO → Process running but not responding
│           ├─ Check CPU usage → If >90% → Performance issue (Issue #2)
│           ├─ Check memory usage → If >90% → Memory leak (Issue #4)
│           └─ Check open FDs → If >80% limit → FD leak (Issue #4)
│
├─ Port reachable but HTTP returns error?
│  ├─ 404 → Incorrect endpoint or space ID (Issue #6)
│  ├─ 500 → Internal server error → Check logs
│  └─ 503 → Service overloaded → Scale or reduce load (Issue #2)
│
└─ Network connectivity issue?
   ├─ Firewall blocking port? → Open ports
   ├─ DNS resolution failing? → Check /etc/hosts
   └─ TLS certificate expired? → Renew certificate (Issue #10)

```

### Decision Tree 2: Performance Degradation

```
Slow Queries / High Latency (P99 >100ms)
│
├─ Check metrics: engram_memory_operation_duration_seconds
│  ├─ ALL operations slow → System-wide issue
│  │  ├─ CPU saturated? → Scale vertically or reduce concurrency
│  │  ├─ Disk I/O wait? → Move to SSD, check WAL lag (Issue #3)
│  │  └─ Memory pressure? → Reduce hot tier size (Issue #4)
│  │
│  └─ SPECIFIC operations slow → Query/index issue
│     ├─ Query operations slow? → Index corruption (Issue #9)
│     ├─ Store operations slow? → WAL lag (Issue #3)
│     └─ Activation slow? → Hot tier thrashing → Increase cache
│
├─ Check consolidation status
│  ├─ Consolidation stuck? → Issue #8
│  └─ WAL lag high? → Issue #3
│
└─ Check for specific error patterns in logs
   ├─ "Index corrupted" → Rebuild index (Issue #9)
   ├─ "Pattern matching timeout" → Adjust thresholds
   └─ "Activation below threshold" → Lower activation threshold

```

### Decision Tree 3: Data Integrity Issues

```
Data Corruption / Inconsistent Results
│
├─ Error contains "deserialize" or "checksum"?
│  ├─ YES → WAL corruption (Issue #5)
│  │        ├─ Single file? → Move aside, continue
│  │        └─ Multiple files? → Restore from backup
│  │
│  └─ NO → Logical corruption
│
├─ Error contains "NaN" or "Infinity"?
│  ├─ YES → Numerical corruption (Issue #7)
│  │        ├─ In confidence? → Sanitize confidence values
│  │        └─ In embeddings? → Identify source, sanitize
│  │
│  └─ NO → Other integrity issue
│
├─ Multi-space isolation violation?
│  ├─ YES → Issue #6 (space isolation)
│  │        ├─ Space not created? → Create space
│  │        └─ HTTP routing gap? → Upgrade or workaround
│  │
│  └─ NO → Data loss scenario
│
└─ Memories missing or incorrect?
   ├─ Check WAL replay status → Incomplete replay?
   ├─ Check backups → Recent backup available?
   └─ Last resort → Restore from backup

```

### Decision Tree 4: Resource Exhaustion

```
Resource Exhaustion (OOM, Disk Full, FD Leak)
│
├─ Out of memory (OOM)?
│  ├─ Check RSS → Growing unbounded? → Memory leak (Issue #4)
│  ├─ Check hot tier size → Too large? → Reduce cache
│  └─ Check query concurrency → Too many? → Add backpressure
│
├─ Disk full?
│  ├─ Check WAL directory → Large? → Issue #3 (WAL lag)
│  ├─ Check tier directories → Unbalanced? → Rebalance tiers
│  └─ Check backup directory → Old backups? → Prune backups
│
├─ Too many open files (EMFILE)?
│  ├─ Check ulimit → Too low? → Increase: ulimit -n 65536
│  ├─ Check fd leaks → Growing? → Identify leak, restart
│  └─ Check concurrent connections → Too many? → Connection pooling
│
└─ CPU saturation?
   ├─ Activation spreading? → Reduce concurrency, add GPU
   ├─ Consolidation? → Reduce batch size, increase interval
   └─ Query processing? → Add caching, optimize queries

```

## Error Pattern Catalog

### Pattern 1: Node/Memory Access Errors

**Log Example:**

```
ERROR Memory node 'session_abc' not found

```

**Meaning**: Attempted to access a memory that doesn't exist in the current graph.

**Common Causes**:

- Memory ID typo

- Space isolation violation

- WAL replay incomplete

- Memory was deleted

**Recovery Strategy**: Retry, PartialResult (return similar nodes with low confidence)

**Diagnostic**:

```bash
# List available nodes in space
curl http://localhost:7432/api/v1/memories | jq '.[].id'

# Check if space is correct
curl -H "X-Memory-Space: your-space" http://localhost:7432/api/v1/memories

```

### Pattern 2: Activation/Confidence Violations

**Log Example:**

```
ERROR Invalid activation level: 1.5 (must be in range [0.0, 1.0])
ERROR Invalid confidence interval: mean=0.8 not in range [0.9, 1.0]

```

**Meaning**: Numerical values outside valid ranges due to computational issues.

**Common Causes**:

- Numerical instability

- NaN propagation

- Floating-point overflow

- Division by zero

**Recovery Strategy**: Fallback (clamp to valid range), RequiresIntervention (corruption)

**Resolution**: See Issue #7 (NaN/Infinity errors)

### Pattern 3: Storage/Persistence Failures

**Log Example:**

```
ERROR WAL operation failed: write
ERROR Failed to prepare persistence directory '/data/engram/space_a'
ERROR Serialization failed: NaN values in embeddings

```

**Meaning**: Cannot write data to disk or read persisted data.

**Common Causes**:

- Disk full

- Permission denied

- Filesystem corruption

- NaN/Inf in data

**Recovery Strategy**: Retry (transient I/O), Fallback (read-only mode), RequiresIntervention (restore backup)

**Resolution**: See Issue #3 (WAL lag), Issue #5 (Data corruption)

### Pattern 4: Multi-Space Isolation Violations

**Log Example:**

```
ERROR Memory space 'tenant_x' not found
ERROR Failed to initialise memory store for space 'tenant_y'
ERROR Missing memory space header
ERROR Invalid space ID: must match ^[a-z0-9_-]+$

```

**Meaning**: Multi-tenant space isolation is broken or space routing failed.

**Common Causes**:

- Space not created (automatic creation may be disabled)

- Invalid space ID format (uppercase letters, special characters)

- Registry corruption or initialization failure

- Directory permission mismatch

- HTTP routing not wired (known issue from Milestone 7)

- Missing `X-Memory-Space` header in request

**Recovery Strategy**: ContinueWithoutFeature (fall back to default space), RequiresIntervention (re-create space)

**Resolution**: See Issue #6 (Multi-space isolation)

**Detailed Troubleshooting**:

```bash
# List all available spaces
curl http://localhost:7432/api/v1/spaces
# Or via CLI
./target/release/engram space list

# Create missing space explicitly
./target/release/engram space create production

# Verify space isolation
curl -H "X-Memory-Space: tenant-a" \
  http://localhost:7432/api/v1/memories/recall?query=test

# Check space directory structure
ls -la ~/.local/share/engram/
# Should show: default/, production/, staging/, etc.

# Verify WAL recovery logs on startup
./target/release/engram start 2>&1 | grep "Recovered"
# Expected: "Recovered 'space_name': N entries, 0 corrupted"

# Check space-specific metrics
curl http://localhost:7432/api/v1/system/health | jq '.spaces[] | {space, memories, pressure}'
```

**Space ID Validation Rules**:
- Only lowercase letters, digits, underscores, hyphens
- Length: 1-64 characters
- Pattern: `^[a-z0-9_-]+$`
- Examples:
  - Valid: `production`, `tenant-123`, `team_alpha`
  - Invalid: `Production` (uppercase), `tenant@123` (@ not allowed), `a` (too short if policy enforces minimum)

### Pattern 5: Index/Query Failures

**Log Example:**

```
ERROR Index corrupted or unavailable
ERROR Query failed: activation level below threshold
ERROR Pattern matching failed: insufficient evidence

```

**Meaning**: Cannot use indices for fast lookups, falling back to linear search.

**Common Causes**:

- Index file corruption

- Low activation energy

- Threshold misconfiguration

**Recovery Strategy**: Fallback (linear search), PartialResult (return low-confidence matches), Retry (with adjusted threshold)

**Resolution**: See Issue #9 (Index corruption)

### Pattern 6: Consolidation/WAL Lag

**Log Example:**

```
WARN WAL lag 15.3s exceeds threshold (10s)
WARN Consolidation cycle failed: pattern detection timeout
ERROR Failed to deserialize WAL entry at offset 1234567

```

**Meaning**: Write-Ahead Log is not being consolidated to tiers fast enough.

**Common Causes**:

- High write rate

- Consolidation disabled/stuck

- Corrupted WAL entry

- Insufficient resources

**Recovery Strategy**: Retry (increase consolidation workers), RequiresIntervention (restore from backup, skip corrupt entry)

**Resolution**: See Issue #3 (WAL lag), Issue #8 (Consolidation stuck)

### Pattern 7: Resource Exhaustion

**Log Example:**

```
ERROR Memory allocation failed: out of memory
ERROR Failed to open file: too many open files (EMFILE)
ERROR Disk write failed: no space left on device (ENOSPC)

```

**Meaning**: System resources exhausted.

**Common Causes**:

- Memory leak

- FD leak

- Disk full (WAL accumulation)

- Too many concurrent connections

**Recovery Strategy**: Restart (clear hot tier), Fallback (reduce cache size), RequiresIntervention (expand disk, fix leak)

**Resolution**: See Issue #4 (Memory leak)

## Common Issues

See [common-issues.md](./common-issues.md) for detailed Context→Action→Verification for the top 10 issues:

1. Engram Won't Start (Category 1: Service Failure)

2. High Latency / Slow Queries (Category 3: Performance)

3. WAL Lag Increasing (Category 2: Resource Exhaustion)

4. Memory Leak / High Memory Usage (Category 2: Resource Exhaustion)

5. Data Corruption (Category 4: Data Integrity)

6. Multi-Space Isolation Violation (Category 4: Data Integrity)

7. NaN/Infinity in Confidence Scores (Category 4: Data Integrity)

8. Consolidation Stuck/Not Running (Category 3: Performance)

9. Index Corruption (Category 4: Data Integrity)

10. gRPC Connection Failures (Category 5: Configuration)

## Diagnostic Tools

### Health Check

```bash
./scripts/diagnose_health.sh

```

Performs 10 health checks in <30 seconds:

1. Process status

2. HTTP health endpoint

3. gRPC endpoint

4. Storage tiers

5. WAL status

6. Memory usage

7. Open file descriptors

8. Recent errors in logs

9. Network connectivity

10. Metrics availability

### Debug Bundle Collection

```bash
./scripts/collect_debug_info.sh

```

Creates a comprehensive debug bundle in <1 minute containing:

- System and process information

- Recent logs (last 1000 lines)

- Configuration files

- Health and metrics snapshots

- Diagnostic report

Send the resulting tarball to support for analysis.

### Log Analysis

```bash
./scripts/analyze_logs.sh "1 hour ago"

```

Analyzes logs and categorizes errors into 7 families:

1. Node/Memory Access Errors

2. Activation/Confidence Errors

3. Storage/Persistence Errors

4. Multi-Space Errors

5. Index/Query Errors

6. NaN/Numerical Errors

7. Resource Exhaustion

Provides actionable recommendations based on patterns detected.

### Emergency Recovery

```bash
# Always use --dry-run first
./scripts/emergency_recovery.sh --sanitize-nan --dry-run
./scripts/emergency_recovery.sh --fix-wal-corruption --space tenant_a --backup-first
./scripts/emergency_recovery.sh --rebuild-indices --dry-run
./scripts/emergency_recovery.sh --reset-space tenant_b --backup-first
./scripts/emergency_recovery.sh --restore-latest
./scripts/emergency_recovery.sh --readonly-mode

```

Six recovery modes for critical failures:

1. `--sanitize-nan`: Remove NaN/Infinity values

2. `--fix-wal-corruption`: Move corrupted WAL entries aside

3. `--rebuild-indices`: Rebuild all indices from scratch

4. `--reset-space`: Reset specific memory space to empty

5. `--restore-latest`: Restore from most recent backup

6. `--readonly-mode`: Start in read-only mode

Always use `--dry-run` first and `--backup-first` for safety.

## Step-by-Step Diagnosis

When troubleshooting an unknown issue:

### Step 1: Collect Information (5 minutes)

```bash
# Run health diagnostic
./scripts/diagnose_health.sh > /tmp/health_$(date +%s).txt

# Analyze recent logs
./scripts/analyze_logs.sh "30 minutes ago" > /tmp/logs_$(date +%s).txt

# Check metrics
curl http://localhost:7432/api/v1/system/health | jq . > /tmp/api_health_$(date +%s).json
curl http://localhost:7432/metrics > /tmp/metrics_$(date +%s).txt

```

### Step 2: Identify Category (2 minutes)

Review the diagnostic output and identify which category:

- **Category 1: Service Failure** - Process not running, immediate crash

- **Category 2: Resource Exhaustion** - Disk full, OOM, FD leaks

- **Category 3: Performance** - High latency, throughput collapse

- **Category 4: Data Integrity** - WAL corruption, NaN values

- **Category 5: Configuration** - Invalid config, version mismatch

Use the appropriate decision tree for your category.

### Step 3: Apply Resolution (5-60 minutes)

Follow the specific issue resolution from the common-issues.md guide.

Each resolution follows this format:

- **Context**: What's happening and why

- **Action**: Specific commands to run

- **Verification**: How to confirm the fix worked

### Step 4: Verify Fix (5 minutes)

```bash
# Re-run health check
./scripts/diagnose_health.sh

# Verify specific metrics improved
curl http://localhost:7432/api/v1/system/health | jq .

# Check logs show no new errors
./scripts/analyze_logs.sh "5 minutes ago"

```

### Step 5: Document (5 minutes)

If this is a new issue not in the troubleshooting guide:

1. Document what caused it

2. Document the resolution steps

3. Add to common-issues.md

4. Update monitoring to detect early

## When to Escalate

### Level 1 → Level 2 Escalation (to Senior Engineer)

Escalate after 30 minutes if:

- Unknown error patterns not in this guide

- Resolution attempts failed 2+ times

- Data corruption detected

- Multi-space isolation broken

- Security concerns

### Level 2 → Level 3 Escalation (to Core Development Team)

Escalate if:

- Software bug confirmed

- Design flaw identified

- Architectural issue

- Security vulnerability

- Multi-hour outage with no resolution

### Emergency Escalation (Immediate)

Escalate immediately for:

- SEV1 incidents (complete service outage)

- Active data loss

- Security breach

- Silent data corruption propagating

See [incident-response.md](./incident-response.md) for detailed escalation procedures.

## Monitoring Integration

Each Prometheus alert links to a specific troubleshooting section:

- `EngramDown` → Issue #1 (Service won't start)

- `HighMemoryOperationLatency` → Issue #2 (High latency)

- `WALLagHigh` → Issue #3 (WAL lag)

- `HighMemoryUsage` → Issue #4 (Memory leak)

- `WALLagCritical` → Issue #5 (Data corruption risk)

- `IndexFallbackActive` → Issue #9 (Index corruption)

Alert definitions include `runbook_url` annotations pointing to this guide.

## Recovery Time Objectives

Expected resolution times by category:

- **Category 1 (Service Failure)**: <5 minutes with diagnostic script

- **Category 2 (Resource Exhaustion)**: 5-15 minutes with automated remediation

- **Category 3 (Performance)**: 15-30 minutes with profiling tools

- **Category 4 (Data Integrity)**: 30-60 minutes with backup restoration

- **Category 5 (Configuration)**: 10-20 minutes with validation tools

## Additional Resources

- [Common Issues FAQ](./common-issues.md) - Top 10 issues with detailed resolutions

- [Incident Response Guide](./incident-response.md) - SEV levels, escalation, communication

- [Log Analysis Guide](./log-analysis.md) - Error pattern catalog and log interpretation

- [Monitoring Guide](./monitoring.md) - Metrics, alerts, and dashboards

- [Backup and Restore](./backup-restore.md) - Disaster recovery procedures

## Getting Help

- **Documentation**: Check this troubleshooting guide first

- **Community**: Engram community forum

- **Support**: support@engram.example.com

- **Emergency**: For SEV1 incidents, page on-call engineer immediately

When requesting help, always include:

1. Output from `./scripts/diagnose_health.sh`

2. Output from `./scripts/analyze_logs.sh "1 hour ago"`

3. Debug bundle from `./scripts/collect_debug_info.sh`

4. Description of what you've tried so far
