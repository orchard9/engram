# Log Analysis Guide

This guide explains how to analyze Engram logs, recognize error patterns, and map them to recovery strategies.

## Quick Start

```bash
# Analyze recent logs with automatic recommendations
./scripts/analyze_logs.sh "1 hour ago"

# Save analysis to file
./scripts/analyze_logs.sh "1 hour ago" /tmp/log-analysis-$(date +%s).txt

# Analyze different time ranges
./scripts/analyze_logs.sh "30 minutes ago"
./scripts/analyze_logs.sh "1 day ago"
./scripts/analyze_logs.sh "2024-10-27 14:00:00"
```

The analysis script categorizes errors into 7 families and provides specific recommendations for each pattern.

## Log Levels

Engram uses standard log levels with specific meanings:

### ERROR
**When used**: Unexpected failures that prevent an operation from completing

**Requires action**: Yes - investigate and resolve

**Examples**:
- Failed to deserialize WAL entry
- Memory node not found
- Cannot bind to port
- Out of memory

**Recovery**: Typically automatic via retry or fallback, but root cause should be investigated

### WARN
**When used**: Unexpected situations that don't prevent operation but indicate potential problems

**Requires action**: Monitor and investigate if recurring

**Examples**:
- WAL lag exceeds threshold
- High memory usage
- Slow query detected
- Index fallback active

**Recovery**: System continues operating but performance may degrade

### INFO
**When used**: Normal operational messages and significant state changes

**Requires action**: No, informational only

**Examples**:
- Server started successfully
- Consolidation cycle completed
- Configuration loaded
- Space created

**Recovery**: N/A - normal operation

### DEBUG
**When used**: Detailed information for troubleshooting (disabled by default)

**Requires action**: No, diagnostic only

**Enable with**:
```bash
RUST_LOG=engram=debug engram start
# Or for specific modules:
RUST_LOG=engram_core::activation=debug,engram_core::query=debug engram start
```

## Error Pattern Catalog

Engram's error types (CoreError, MemorySpaceError, EngramError) map to 7 error pattern families, each with specific recovery strategies.

### Pattern 1: Node/Memory Access Errors

**Log Examples**:
```
ERROR Memory node 'session_abc' not found
ERROR Invalid node ID: 'malformed-id'
ERROR Memory 'user_123_context' not found in space 'tenant_a'
```

**Meaning**: Attempted to access a memory that doesn't exist in the current graph.

**Expected Error Type**: `CoreError::NotFound` or `MemorySpaceError::NotFound`

**Common Causes**:
1. Memory ID typo in client code
2. Space isolation violation (querying wrong space)
3. WAL replay incomplete (memory not yet loaded)
4. Memory was deleted
5. Space doesn't exist

**Recovery Strategy**:
- `Retry` - Transient issue, try again
- `PartialResult` - Return similar nodes with low confidence
- `ContinueWithoutFeature` - Skip this memory and continue

**Diagnostic Commands**:
```bash
# List available nodes in space
curl http://localhost:7432/api/v1/memories | jq '.[].id'

# Check if space is correct
curl -H "X-Memory-Space: your-space" http://localhost:7432/api/v1/memories

# Verify space exists
curl http://localhost:7432/api/v1/system/health | jq '.spaces'
```

**Resolution**: See [Issue #1](./common-issues.md#issue-1-engram-wont-start) or [Issue #6](./common-issues.md#issue-6-multi-space-isolation-violation)

---

### Pattern 2: Activation/Confidence Violations

**Log Examples**:
```
ERROR Invalid activation level: 1.5 (must be in range [0.0, 1.0])
ERROR Invalid confidence interval: mean=0.8 not in range [0.9, 1.0]
ERROR Confidence value out of bounds: -0.1
ERROR Activation decay resulted in NaN
```

**Meaning**: Numerical values outside valid ranges due to computational issues.

**Expected Error Type**: `CoreError::InvalidInput` or validation errors

**Common Causes**:
1. Numerical instability in activation calculation
2. NaN propagation from previous computation
3. Floating-point overflow
4. Division by zero
5. Invalid input from client

**Recovery Strategy**:
- `Fallback` - Clamp to valid range [0.0, 1.0]
- `RequiresIntervention` - If corruption is widespread

**Diagnostic Commands**:
```bash
# Check for invalid confidence values
curl http://localhost:7432/api/v1/memories | \
  jq '.[] | select(.confidence < 0 or .confidence > 1)'

# Check metrics for NaN
curl http://localhost:7432/metrics | grep -E "NaN|Inf"

# Review recent errors
journalctl -u engram | grep -E "activation|confidence" | tail -20
```

**Resolution**: See [Issue #7](./common-issues.md#issue-7-naninfinity-in-confidence-scores)

---

### Pattern 3: Storage/Persistence Failures

**Log Examples**:
```
ERROR WAL operation failed: write
ERROR Failed to prepare persistence directory '/data/engram/space_a'
ERROR Serialization failed: NaN values in embeddings
ERROR Disk write failed: No space left on device (ENOSPC)
ERROR Failed to deserialize WAL entry at offset 1234567
ERROR Checksum mismatch in file: hot-tier-0001.dat
```

**Meaning**: Cannot write data to disk or read persisted data.

**Expected Error Type**: `CoreError::Persistence` or `io::Error`

**Common Causes**:
1. Disk full (check `/data/engram` and WAL directories)
2. Permission denied (check ownership and modes)
3. Filesystem corruption
4. NaN/Inf in data being serialized
5. Unclean shutdown left corrupt files

**Recovery Strategy**:
- `Retry` - Transient I/O error, try again
- `Fallback` - Enable read-only mode to prevent further corruption
- `RequiresIntervention` - Restore from backup

**Diagnostic Commands**:
```bash
# Check disk space
df -h ${ENGRAM_DATA_DIR:-./data}

# Check permissions
ls -la ${ENGRAM_DATA_DIR:-./data}

# Check for corruption
journalctl -u engram | grep -E "deserialize|checksum|corruption"

# Count WAL files (indicator of lag)
find ${ENGRAM_DATA_DIR:-./data}/wal -name "*.log" | wc -l
```

**Resolution**: See [Issue #3](./common-issues.md#issue-3-wal-lag-increasing) or [Issue #5](./common-issues.md#issue-5-data-corruption)

---

### Pattern 4: Multi-Space Isolation Violations

**Log Examples**:
```
ERROR Memory space 'tenant_x' not found
ERROR Failed to initialise memory store for space 'tenant_y'
ERROR Space 'tenant_z' not in registry
ERROR Cannot access memory from space 'space_a' in context of 'space_b'
```

**Meaning**: Multi-tenant space isolation is broken or space doesn't exist.

**Expected Error Type**: `MemorySpaceError::NotFound` or `MemorySpaceError::IsolationViolation`

**Common Causes**:
1. Space not created in registry
2. Registry corruption
3. Directory permission mismatch
4. HTTP routing not wired (known issue from Milestone 7)
5. X-Memory-Space header not being passed correctly

**Recovery Strategy**:
- `ContinueWithoutFeature` - Fall back to default space
- `RequiresIntervention` - Re-create space or restore from backup

**Diagnostic Commands**:
```bash
# Check space registry
curl http://localhost:7432/api/v1/system/health | jq '.spaces'

# Verify space directories exist
ls -la ${ENGRAM_DATA_DIR:-./data}/

# Test space isolation
curl -H "X-Memory-Space: space_a" \
  http://localhost:7432/api/v1/memories/test_mem
curl -H "X-Memory-Space: space_b" \
  http://localhost:7432/api/v1/memories/test_mem
# Should return different results
```

**Resolution**: See [Issue #6](./common-issues.md#issue-6-multi-space-isolation-violation)

---

### Pattern 5: Index/Query Failures

**Log Examples**:
```
ERROR Index corrupted or unavailable
ERROR Query failed: activation level below threshold
ERROR Pattern matching failed: insufficient evidence
ERROR Index lookup failed, falling back to linear search
ERROR Cannot rebuild index: insufficient memory
```

**Meaning**: Cannot use indices for fast lookups, performance degrades.

**Expected Error Type**: `CoreError::IndexCorruption` or query errors

**Common Causes**:
1. Index file corruption (unclean shutdown, disk failure)
2. Low activation energy (not actually an error, just low confidence)
3. Threshold misconfiguration (too strict)
4. Index not yet built (first startup)

**Recovery Strategy**:
- `Fallback` - Linear search (slow but functional)
- `PartialResult` - Return low-confidence matches
- `Retry` - With adjusted threshold

**Diagnostic Commands**:
```bash
# Check index status
curl http://localhost:7432/api/v1/system/health | jq '.indices'

# Look for index errors
journalctl -u engram | grep -i index | grep -E "ERROR|corrupt|fallback"

# Check index files
ls -lh ${ENGRAM_DATA_DIR:-./data}/*/indices/

# Monitor fallback rate
curl http://localhost:7432/metrics | grep index_fallback
```

**Resolution**: See [Issue #9](./common-issues.md#issue-9-index-corruption)

---

### Pattern 6: Consolidation/WAL Lag

**Log Examples**:
```
WARN WAL lag 15.3s exceeds threshold (10s)
WARN Consolidation cycle failed: pattern detection timeout
ERROR Failed to deserialize WAL entry at offset 1234567
WARN Consolidation not running: check interval not met
ERROR Pattern detection timeout after 60 seconds
```

**Meaning**: Write-Ahead Log is not being consolidated to tiers fast enough.

**Expected Error Type**: Warnings mostly, errors if WAL is corrupted

**Common Causes**:
1. High write rate exceeding consolidation capacity
2. Consolidation disabled or stuck
3. Corrupted WAL entry blocking consolidation
4. Insufficient resources (CPU, memory)
5. Pattern detection timeout too short

**Recovery Strategy**:
- `Retry` - Increase consolidation workers or batch size
- `RequiresIntervention` - Restore from backup if WAL is corrupted, skip corrupt entry

**Diagnostic Commands**:
```bash
# Check WAL lag
curl -G http://localhost:9090/api/v1/query \
  --data-urlencode 'query=engram_wal_lag_seconds'

# Check consolidation metrics
curl -G http://localhost:9090/api/v1/query \
  --data-urlencode 'query=rate(engram_consolidation_cycles_total[5m])'

# Count WAL files
find ${ENGRAM_DATA_DIR:-./data}/wal -name "*.log" | wc -l

# Check for consolidation errors
journalctl -u engram | grep -i consolidation | tail -50
```

**Resolution**: See [Issue #3](./common-issues.md#issue-3-wal-lag-increasing) or [Issue #8](./common-issues.md#issue-8-consolidation-stucknot-running)

---

### Pattern 7: Resource Exhaustion

**Log Examples**:
```
ERROR Memory allocation failed: out of memory
ERROR Failed to open file: too many open files (EMFILE)
ERROR Disk write failed: no space left on device (ENOSPC)
ERROR Cannot allocate memory for hot tier
ERROR Connection limit reached: 1024 concurrent connections
```

**Meaning**: System resources (memory, disk, file descriptors, connections) exhausted.

**Expected Error Type**: `std::alloc::AllocError`, `io::Error`, system errors

**Common Causes**:
1. Memory leak (unbounded RSS growth)
2. File descriptor leak (not closing files)
3. Disk full (WAL accumulation, logs, backups)
4. Too many concurrent connections
5. Hot tier configured too large for available memory

**Recovery Strategy**:
- `Restart` - Clear hot tier and leaked resources
- `Fallback` - Reduce cache size, reject new connections
- `RequiresIntervention` - Expand disk, fix leak, increase limits

**Diagnostic Commands**:
```bash
# Check memory usage
ps aux | grep engram
cat /proc/$(pgrep engram)/status | grep Vm

# Check disk usage
df -h ${ENGRAM_DATA_DIR:-./data}
du -sh ${ENGRAM_DATA_DIR:-./data}/*

# Check file descriptors
ls /proc/$(pgrep engram)/fd | wc -l
ulimit -n

# Check connections
netstat -an | grep :7432 | grep ESTABLISHED | wc -l
```

**Resolution**: See [Issue #4](./common-issues.md#issue-4-memory-leak--high-memory-usage)

---

## Recovery Strategies

Engram uses a structured approach to error recovery, matching error types to recovery strategies:

### Retry
**When**: Transient failures likely to succeed on retry

**Implementation**: Automatic retry with exponential backoff

**Examples**:
- Network timeouts
- Temporary I/O errors
- Lock contention
- Rate limiting

**Monitoring**: Count retry attempts, alert if >5 retries

---

### Fallback
**When**: Primary method fails but alternative exists

**Implementation**: Gracefully degrade to alternative approach

**Examples**:
- Index corrupted → linear search
- Hot tier full → read from warm tier
- Activation below threshold → lower threshold
- Pattern matching timeout → skip pattern detection

**Monitoring**: Track fallback rate, alert if >5%

---

### PartialResult
**When**: Complete result unavailable but partial result acceptable

**Implementation**: Return what's available with low confidence flag

**Examples**:
- Some memories not accessible → return subset
- Pattern matching incomplete → return partial patterns
- Activation spreading timeout → return partial activation graph

**Monitoring**: Track partial result rate, alert if >10%

---

### ContinueWithoutFeature
**When**: Feature unavailable but system can operate without it

**Implementation**: Disable feature and continue core operations

**Examples**:
- Consolidation stuck → continue accepting writes
- Index unavailable → continue with linear search
- Multi-space unavailable → fall back to default space

**Monitoring**: Track disabled features, alert on any

---

### RequiresIntervention
**When**: Automatic recovery impossible, operator must intervene

**Implementation**: Log error, trigger alert, wait for operator action

**Examples**:
- Data corruption detected
- Disk full
- Security violation
- Configuration invalid

**Monitoring**: Page on-call immediately

---

## Log Analysis Workflow

### Step 1: Collect Logs

```bash
# Using journalctl (systemd)
journalctl -u engram --since "1 hour ago" --no-pager > /tmp/logs.txt

# Using log file
tail -10000 /var/log/engram.log > /tmp/logs.txt

# Or use analysis script (automatic)
./scripts/analyze_logs.sh "1 hour ago" > /tmp/analysis.txt
```

### Step 2: Identify Error Patterns

```bash
# Run automated analysis
./scripts/analyze_logs.sh "1 hour ago"

# Look at error categories section:
# 1. Node/Memory Access Errors: X
# 2. Activation/Confidence Errors: Y
# 3. Storage/Persistence Errors: Z
# etc.
```

### Step 3: Review Top Error Patterns

The analysis script shows top error patterns sorted by frequency:

```
=== TOP ERROR PATTERNS ===

    42  Failed to deserialize WAL entry
    15  Memory node not found
     8  Index corrupted or unavailable
     3  Invalid activation level
```

### Step 4: Check Recommendations

The script provides actionable recommendations based on patterns:

```
=== ACTIONABLE RECOMMENDATIONS ===

HIGH PRIORITY: Storage/Persistence Errors
  Issue: 42 storage errors detected
  Action: Check disk health and WAL lag
  Reference: Issue 3, 5 in troubleshooting guide
  Command: ./scripts/diagnose_health.sh
```

### Step 5: Execute Recommended Actions

Follow the recommendations from the analysis script or refer to the linked issues in the [Common Issues Guide](./common-issues.md).

## Advanced Log Analysis

### Correlating Errors with Metrics

```bash
# Get error timestamp
ERROR_TIME=$(journalctl -u engram | grep "ERROR" | tail -1 | awk '{print $1, $2, $3}')

# Query metrics at that time
curl -G http://localhost:9090/api/v1/query \
  --data-urlencode "query=engram_memory_operation_duration_seconds" \
  --data-urlencode "time=$ERROR_TIME"
```

### Finding Error Context

```bash
# Show 10 lines before and after each error
journalctl -u engram --since "1 hour ago" | grep -B 10 -A 10 "ERROR"

# Or use analyze script context mode
./scripts/analyze_logs.sh "1 hour ago" | grep -B 5 -A 5 "ERROR"
```

### Tracking Error Rates Over Time

```bash
# Errors per 5-minute window
journalctl -u engram --since "1 day ago" | \
  grep "ERROR" | \
  awk '{print $1" "$2}' | \
  cut -c1-16 | \
  uniq -c

# Output shows errors per 5-min bucket:
#   5 Oct 27 14:00
#  12 Oct 27 14:05
#   3 Oct 27 14:10
```

### Identifying Error Cascades

```bash
# Look for multiple related errors in short time
journalctl -u engram --since "1 hour ago" | \
  grep "ERROR" | \
  awk '{print $3}' | \
  uniq -c | \
  sort -rn | \
  head -10

# Shows which seconds had multiple errors (cascade indicator)
```

## Log-Based Alerts

Configure Prometheus Loki to create alerts from log patterns:

```yaml
# Example Loki alert rules
groups:
  - name: engram_log_alerts
    interval: 1m
    rules:
      - alert: HighErrorRate
        expr: |
          sum(rate({job="engram"} |= "ERROR" [5m])) > 10
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High error rate in Engram logs"
          description: ">10 errors/sec for 5 minutes"

      - alert: DataCorruptionDetected
        expr: |
          count_over_time({job="engram"} |~ "corruption|checksum|deserialize.*failed" [5m]) > 0
        for: 1m
        labels:
          severity: critical
        annotations:
          summary: "Data corruption detected in logs"
          description: "Immediate investigation required"

      - alert: WALLagIncreasing
        expr: |
          count_over_time({job="engram"} |~ "WAL lag.*exceeds" [10m]) > 20
        for: 10m
        labels:
          severity: warning
        annotations:
          summary: "Persistent WAL lag warnings"
          description: "Check consolidation status"
```

## Debug Logging

For troubleshooting specific issues, enable debug logging:

```bash
# Enable debug logging for all engram modules
RUST_LOG=engram=debug systemctl restart engram

# Enable for specific modules only
RUST_LOG=engram_core::activation=debug,engram_core::query=debug systemctl restart engram

# Enable trace level (very verbose)
RUST_LOG=engram_core::spreading=trace systemctl restart engram

# Disable after troubleshooting (debug logs are verbose)
unset RUST_LOG
systemctl restart engram
```

**Warning**: Debug and trace logging significantly increases log volume. Only enable when actively troubleshooting.

## Integration with Monitoring

See [Monitoring Guide](./monitoring.md) for:
- Setting up Loki for log aggregation
- Creating log-based alerts
- Correlating logs with metrics
- Dashboards showing error rates

## Log Retention

Recommended log retention:

- **System logs** (journald): 7-14 days
- **Application logs** (file): 30 days
- **Aggregated logs** (Loki): 90 days
- **Critical error logs**: Archive indefinitely

Configure in `/etc/systemd/journald.conf`:

```ini
[Journal]
SystemMaxUse=2G
MaxRetentionSec=14d
```

## Troubleshooting the Analysis Script

If `analyze_logs.sh` fails:

```bash
# Check if journalctl is available
command -v journalctl

# Check if log files exist
ls -la /var/log/engram.log
ls -la ${ENGRAM_DATA_DIR:-./data}/engram.log

# Run with verbose output
bash -x ./scripts/analyze_logs.sh "1 hour ago"

# Check permissions
ls -la ./scripts/analyze_logs.sh
chmod +x ./scripts/analyze_logs.sh
```

## Additional Resources

- [Troubleshooting Guide](./troubleshooting.md) - Decision trees and diagnostic procedures
- [Common Issues](./common-issues.md) - Top 10 issues with resolutions
- [Incident Response](./incident-response.md) - Escalation and communication
- [Monitoring Guide](./monitoring.md) - Metrics and alerting

## Getting Help

When reporting issues, always include:

1. Output from `./scripts/analyze_logs.sh "1 hour ago"`
2. Recent error timeline from logs
3. Error category breakdown
4. Any recommendations from the analysis script

This gives support engineers the context needed to diagnose issues quickly.
