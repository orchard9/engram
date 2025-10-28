# Common Issues FAQ

This guide provides detailed troubleshooting procedures for the 10 most common Engram production issues. Each issue follows a Context→Action→Verification format for systematic resolution.

Use the [decision trees](./troubleshooting.md#decision-trees) to quickly identify which issue you're experiencing.

## Quick Index

1. [Engram Won't Start](#issue-1-engram-wont-start) - Service Failure
2. [High Latency / Slow Queries](#issue-2-high-latency--slow-queries) - Performance
3. [WAL Lag Increasing](#issue-3-wal-lag-increasing) - Resource Exhaustion
4. [Memory Leak / High Memory Usage](#issue-4-memory-leak--high-memory-usage) - Resource Exhaustion
5. [Data Corruption](#issue-5-data-corruption) - Data Integrity
6. [Multi-Space Isolation Violation](#issue-6-multi-space-isolation-violation) - Data Integrity
7. [NaN/Infinity in Confidence Scores](#issue-7-naninfinity-in-confidence-scores) - Data Integrity
8. [Consolidation Stuck/Not Running](#issue-8-consolidation-stucknot-running) - Performance
9. [Index Corruption](#issue-9-index-corruption) - Data Integrity
10. [gRPC Connection Failures](#issue-10-grpc-connection-failures) - Configuration

## Issue 1: Engram Won't Start

**Category**: Service Failure
**Expected Resolution Time**: <5 minutes
**Severity**: SEV1 (if production), SEV3 (if development)

### Context

The Engram process exits immediately after start, or the health endpoint is not accessible after startup. This is typically caused by port conflicts, permission issues, or configuration errors.

**Symptoms**:
- Process exits immediately after `engram start` or `systemctl start engram`
- Error in logs: "Failed to bind to address"
- Health endpoint returns connection refused
- Port already in use

**Common Causes**:
- Port 7432 (HTTP) or 50051 (gRPC) already in use
- Data directory permissions incorrect
- Invalid configuration file
- Missing dependencies
- Corrupted binary

### Action

**Step 1: Check if process is running**

```bash
pgrep -x engram
ps aux | grep engram
```

If process is not running, check logs for startup errors:

```bash
# Using journald
journalctl -u engram -n 50 --no-pager

# Using log file
tail -50 /var/log/engram.log
```

**Step 2: Identify the root cause**

**Port conflict** (error contains "Failed to bind" or "Address already in use"):

```bash
# Check what's using the ports
sudo lsof -i :7432
sudo lsof -i :50051

# Kill conflicting process or change Engram's port
sudo kill <PID>

# Or update config to use different port
# Edit config.toml:
# [http]
# port = 7433
```

**Permission denied** (error contains "Permission denied" or "Cannot create directory"):

```bash
# Check data directory ownership
ls -la ${ENGRAM_DATA_DIR:-./data}

# Fix ownership
sudo chown -R engram:engram ${ENGRAM_DATA_DIR:-./data}
sudo chmod -R 755 ${ENGRAM_DATA_DIR:-./data}
```

**Configuration error** (error contains "Invalid configuration" or "Failed to parse"):

```bash
# Validate configuration
engram config validate

# Check for syntax errors
cat ${ENGRAM_CONFIG:-~/.config/engram/config.toml}

# Use default config if corrupted
mv ${ENGRAM_CONFIG:-~/.config/engram/config.toml} ${ENGRAM_CONFIG}.backup
engram config init
```

**Missing dependencies**:

```bash
# Check shared library dependencies
ldd $(which engram)

# On macOS
otool -L $(which engram)
```

### Verification

```bash
# Start Engram
systemctl start engram
# Or
engram start

# Wait for startup (typically <5 seconds)
sleep 5

# Verify process is running
pgrep -x engram

# Check health endpoint
curl http://localhost:7432/api/v1/system/health

# Should return HTTP 200 with JSON status
```

**Success criteria**:
- Process stays running (PID exists after 30 seconds)
- Health endpoint returns 200 OK
- No errors in recent logs

**If still failing**: Collect debug information and escalate:

```bash
./scripts/collect_debug_info.sh
```

## Issue 2: High Latency / Slow Queries

**Category**: Performance
**Expected Resolution Time**: 15-30 minutes
**Severity**: SEV2 (if affecting users), SEV3 (if sporadic)

### Context

Query operations are taking longer than expected, typically P99 latency >100ms. This degrades user experience and can lead to timeout errors.

**Symptoms**:
- Alert: "HighMemoryOperationLatency"
- P99 latency >100ms in metrics
- Client applications timing out
- Slow API responses

**Common Causes**:
- Missing or corrupted indices (falling back to linear search)
- Hot tier too small (cache thrashing)
- Disk I/O bottleneck (slow storage)
- CPU saturation (too many concurrent operations)
- Large result sets without pagination

### Action

**Step 1: Check current performance metrics**

```bash
# Check P99 latency
curl -G http://localhost:9090/api/v1/query \
  --data-urlencode 'query=histogram_quantile(0.99, engram_memory_operation_duration_seconds_bucket)'

# Check operation breakdown
curl http://localhost:7432/metrics | grep engram_memory_operation_duration
```

**Step 2: Profile performance**

```bash
# Run performance profiling
./scripts/profile_performance.sh 60

# Analyze slow queries
./scripts/analyze_slow_queries.sh 100 1h
```

**Step 3: Apply appropriate fix**

**If index is corrupted** (metrics show high fallback rate):

```bash
# Rebuild indices
./scripts/emergency_recovery.sh --rebuild-indices --dry-run
./scripts/emergency_recovery.sh --rebuild-indices

# Monitor rebuild progress
watch 'curl -s http://localhost:7432/api/v1/system/health | jq .indices'
```

**If hot tier is too small** (metrics show high cache miss rate):

```toml
# Edit config.toml
[storage.hot_tier]
max_nodes = 100000  # Increase from default 10000
max_memory_mb = 2048  # Increase from default 512
```

**If disk I/O is bottleneck** (high iowait in top):

```bash
# Check disk performance
iostat -x 1 5

# Consider:
# 1. Move data to faster storage (SSD)
# 2. Enable compression to reduce I/O
# 3. Increase hot tier to reduce disk access
```

**If CPU is saturated** (>90% CPU usage):

```bash
# Reduce concurrent operations
# Edit config.toml:
[query]
max_concurrent = 8  # Reduce from default 16

# Or scale vertically (add more CPU cores)
```

### Verification

```bash
# Run benchmark after changes
./scripts/benchmark_deployment.sh 60 10

# Check P99 latency in results
grep "P99 Latency" /tmp/benchmark_report.txt

# Should show latency <100ms
# Monitor for 10 minutes to ensure sustained improvement
watch -n 10 'curl -s http://localhost:7432/metrics | grep operation_duration | tail -5'
```

**Success criteria**:
- P99 latency <100ms
- No timeout errors in client applications
- Index fallback rate <1%

## Issue 3: WAL Lag Increasing

**Category**: Resource Exhaustion
**Expected Resolution Time**: 5-15 minutes
**Severity**: SEV2 (if growing rapidly), SEV3 (if stable)

### Context

The Write-Ahead Log (WAL) is accumulating faster than consolidation can process it. This leads to disk space exhaustion and slower startup times.

**Symptoms**:
- Alert: "WALLagHigh" or "WALLagCritical"
- Metric `engram_wal_lag_seconds` > 10
- Disk usage growing rapidly
- Many .log files in WAL directory

**Common Causes**:
- Consolidation not running or stuck
- Write rate exceeds consolidation capacity
- Disk full (preventing WAL flush)
- Pattern detection timeout

### Action

**Step 1: Check WAL status**

```bash
# Check WAL lag metric
curl -G http://localhost:9090/api/v1/query \
  --data-urlencode 'query=engram_wal_lag_seconds'

# Count WAL files
find ${ENGRAM_DATA_DIR:-./data}/wal -name "*.log" | wc -l

# Check WAL directory size
du -sh ${ENGRAM_DATA_DIR:-./data}/wal
```

**Step 2: Identify consolidation status**

```bash
# Check consolidation metrics
curl -G http://localhost:9090/api/v1/query \
  --data-urlencode 'query=rate(engram_consolidation_cycles_total[5m])'

# Check for consolidation errors
journalctl -u engram | grep -i consolidation | tail -50

# Check if consolidation is enabled
grep consolidation ${ENGRAM_CONFIG:-~/.config/engram/config.toml}
```

**Step 3: Fix the issue**

**If consolidation is disabled**:

```toml
# Edit config.toml
[consolidation]
enabled = true
check_interval_secs = 60
idle_threshold_secs = 300
```

**If consolidation is stuck** (no cycles completing):

```bash
# Check for stuck consolidation threads
pstack $(pgrep engram) | grep -A 10 consolidation

# Restart Engram to unstick
systemctl restart engram
```

**If write rate is too high**:

```toml
# Increase consolidation workers
[consolidation]
max_workers = 4  # Increase from default 2
batch_size = 1000  # Increase from default 500

# Or reduce write throughput from clients
```

**If disk is full**:

```bash
# Free up space
./scripts/prune_backups.sh 30  # Keep only last 30 days

# Compact WAL (if tool available)
./scripts/wal_compact.sh

# Or expand disk volume
```

### Verification

```bash
# Monitor WAL lag - should decrease
watch -n 5 "curl -s -G http://localhost:9090/api/v1/query \
  --data-urlencode 'query=engram_wal_lag_seconds' | jq -r '.data.result[0].value[1]'"

# WAL lag should decrease below 10 seconds within 10 minutes
# WAL file count should stabilize or decrease

# Check consolidation is running
curl http://localhost:7432/api/v1/system/health | jq '.consolidation_rate'
# Should be > 0.0
```

**Success criteria**:
- WAL lag < 10 seconds
- Consolidation cycle completing regularly
- WAL file count stable or decreasing
- Disk usage not growing

## Issue 4: Memory Leak / High Memory Usage

**Category**: Resource Exhaustion
**Expected Resolution Time**: 10-20 minutes (immediate) or escalation (root cause)
**Severity**: SEV2 (if growing), SEV3 (if stable)

### Context

The Engram process is consuming excessive memory (>4GB RSS) or memory usage is growing unbounded, leading to OOM conditions.

**Symptoms**:
- Alert: "HighMemoryUsage"
- RSS >4GB and growing
- OOM killer terminates process
- System swap usage high

**Common Causes**:
- Hot tier configured too large
- Memory leak in code
- Too many memory spaces loaded
- Large embeddings or query results not freed

### Action

**Step 1: Check current memory usage**

```bash
# Check process memory
ps aux | grep engram
# Or more detailed
cat /proc/$(pgrep engram)/status | grep Vm

# Check memory over time
./scripts/profile_performance.sh 120
cat ./profile-*/memory_usage.txt
```

**Step 2: Identify memory consumers**

```bash
# Check hot tier size
curl http://localhost:7432/api/v1/system/health | jq '.hot_tier_nodes'

# Check number of spaces
curl http://localhost:7432/api/v1/system/health | jq '.spaces | length'

# Check for memory-heavy operations
curl http://localhost:7432/metrics | grep engram_memory_bytes
```

**Step 3: Apply fix**

**If hot tier is too large**:

```toml
# Edit config.toml
[storage.hot_tier]
max_nodes = 10000  # Reduce from current value
max_memory_mb = 1024  # Reduce from current value
```

**If too many spaces are loaded**:

```bash
# Archive unused spaces
engram space archive <space_id>

# Or delete unused spaces
engram space delete <space_id> --confirm
```

**If memory leak is suspected**:

```bash
# Immediate mitigation: restart process
systemctl restart engram

# Collect memory profile for developers
./scripts/profile_performance.sh 300
./scripts/collect_debug_info.sh

# Monitor to see if memory grows again
watch -n 60 "ps -o rss= -p $(pgrep engram)"
```

### Verification

```bash
# Memory should stabilize after restart
watch -n 5 "ps -p $(pgrep engram) -o rss="

# Monitor for 30 minutes to ensure no growth
# RSS should remain stable or grow slowly with normal load

# Check metrics
curl http://localhost:7432/metrics | grep engram_memory_bytes
```

**Success criteria**:
- RSS < 4GB (or within configured limits)
- Memory not growing unbounded
- No OOM events

**If memory leak persists**: Escalate to development team with memory profile.

## Issue 5: Data Corruption

**Category**: Data Integrity
**Expected Resolution Time**: 30-60 minutes
**Severity**: SEV1 (if preventing startup or causing data loss)

### Context

Data files are corrupted, preventing Engram from starting or causing incorrect query results. This is a critical issue requiring immediate attention.

**Symptoms**:
- Error: "Failed to deserialize WAL"
- Error: "Checksum mismatch"
- Engram crashes on startup
- Inconsistent query results
- Data files unreadable

**Common Causes**:
- Disk failure or filesystem corruption
- Unclean shutdown (kill -9, power loss)
- Software bug in serialization
- Bit rot (rare on modern systems)

### Action

**Step 1: Identify extent of corruption**

```bash
# Check logs for corruption errors
journalctl -u engram | grep -A 5 "corruption\|deserialize\|checksum"

# Try to start Engram and capture error
systemctl start engram
sleep 5
journalctl -u engram -n 100 --no-pager

# Check which files are corrupted
dmesg | grep -i "I/O error\|corruption"
```

**Step 2: Attempt recovery**

**For corrupted WAL files**:

```bash
# Use emergency recovery to quarantine corrupt files
./scripts/emergency_recovery.sh --fix-wal-corruption --dry-run
./scripts/emergency_recovery.sh --fix-wal-corruption --backup-first

# This moves corrupt files to quarantine directory
# Engram can then start with partial data
```

**For widespread corruption**:

```bash
# Stop Engram
systemctl stop engram

# Verify backup integrity
./scripts/verify_backup.sh backups/latest-*.tar.zst

# Restore from most recent good backup
./scripts/emergency_recovery.sh --restore-latest --backup-first

# Or manual restore
./scripts/restore.sh backups/latest-full-*.tar.zst
```

**Step 3: Prevent recurrence**

```bash
# Check disk health
sudo smartctl -a /dev/sda

# Enable fsync for durability (trade-off: slower writes)
# Edit config.toml:
[storage]
fsync_on_write = true
verify_checksums = true
```

### Verification

```bash
# Start Engram
systemctl start engram

# Run health check
./scripts/diagnose_health.sh

# Verify data integrity
curl http://localhost:7432/api/v1/memories | jq 'length'
# Should return expected number of memories

# Run test queries
curl "http://localhost:7432/api/v1/query?cue=test&limit=10"
# Should return results without errors
```

**Success criteria**:
- Engram starts successfully
- No deserialization errors in logs
- Query results are consistent
- Health check passes

**Post-recovery**:
1. Document the corruption event
2. Review disk health and filesystem
3. Consider enabling additional data integrity checks

## Issue 6: Multi-Space Isolation Violation

**Category**: Data Integrity
**Expected Resolution Time**: 15-30 minutes
**Severity**: SEV2 (data isolation is critical for multi-tenant)

### Context

Memories from one space are appearing in another, or the X-Memory-Space header is not being respected. This violates tenant isolation guarantees.

**Symptoms**:
- Memories from space A appearing in space B queries
- 404 errors for valid space IDs
- Error: "Memory space 'tenant_x' not found"
- X-Memory-Space header ignored

**Common Causes**:
- Space not created in registry
- HTTP routing not wired to space isolation (known gap from Milestone 7)
- Registry corruption
- Directory permission mismatch

### Action

**Step 1: Verify space exists**

```bash
# Check space registry
curl http://localhost:7432/api/v1/system/health | jq '.spaces'

# Verify space directories exist
ls -la ${ENGRAM_DATA_DIR:-./data}/

# Check for the specific space
ls -la ${ENGRAM_DATA_DIR:-./data}/tenant_a
```

**Step 2: Test isolation**

```bash
# Create test memory in space A
curl -X POST -H "X-Memory-Space: space_a" \
  http://localhost:7432/api/v1/memories \
  -H "Content-Type: application/json" \
  -d '{"id":"test_isolation_a","embedding":[0.1,0.2],"confidence":0.9}'

# Try to access from space B (should fail)
curl -H "X-Memory-Space: space_b" \
  http://localhost:7432/api/v1/memories/test_isolation_a

# Should return 404 or empty, NOT the memory
```

**Step 3: Fix the issue**

**If space doesn't exist**:

```bash
# Create space via API
curl -X POST http://localhost:7432/api/v1/spaces/create \
  -H "Content-Type: application/json" \
  -d '{"space_id": "tenant_a"}'

# Verify creation
curl http://localhost:7432/api/v1/system/health | jq '.spaces'
```

**If directory permissions are wrong**:

```bash
# Fix ownership
sudo chown -R engram:engram ${ENGRAM_DATA_DIR:-./data}/tenant_a
sudo chmod -R 755 ${ENGRAM_DATA_DIR:-./data}/tenant_a
```

**If HTTP routing gap (known issue from Milestone 7)**:

```bash
# This is a known architectural gap where X-Memory-Space header
# is extracted but not fully wired through all operations

# Workaround 1: Upgrade to version with fix
# Workaround 2: Use gRPC API which has proper space isolation
# Workaround 3: Run separate Engram instances per tenant
```

**If registry is corrupted**:

```bash
# Stop Engram
systemctl stop engram

# Remove corrupted registry
rm -rf ${ENGRAM_DATA_DIR:-./data}/tenant_a/.registry

# Restore from backup
./scripts/restore.sh backups/tenant_a-*.tar.zst

# Start Engram
systemctl start engram
```

### Verification

```bash
# Create memory in space A
MEM_A_ID=$(curl -X POST -H "X-Memory-Space: space_a" \
  http://localhost:7432/api/v1/memories \
  -H "Content-Type: application/json" \
  -d '{"id":"verify_a","embedding":[0.1],"confidence":0.9}' | jq -r .id)

# Verify accessible from space A
curl -H "X-Memory-Space: space_a" \
  http://localhost:7432/api/v1/memories/$MEM_A_ID
# Should return the memory

# Verify NOT accessible from space B
curl -H "X-Memory-Space: space_b" \
  http://localhost:7432/api/v1/memories/$MEM_A_ID
# Should return 404 or error

# Create memory in space B
MEM_B_ID=$(curl -X POST -H "X-Memory-Space: space_b" \
  http://localhost:7432/api/v1/memories \
  -H "Content-Type: application/json" \
  -d '{"id":"verify_b","embedding":[0.2],"confidence":0.8}' | jq -r .id)

# Query space A should not return B's memory
curl -H "X-Memory-Space: space_a" \
  "http://localhost:7432/api/v1/query?cue=verify&limit=10" | \
  jq '.[] | select(.id == "'$MEM_B_ID'")'
# Should return empty
```

**Success criteria**:
- Memories in space A not visible from space B
- Each space has independent query results
- Space directories isolated on disk

## Issue 7: NaN/Infinity in Confidence Scores

**Category**: Data Integrity
**Expected Resolution Time**: 15-30 minutes
**Severity**: SEV2 (data corruption), SEV1 (if propagating)

### Context

Confidence scores, activation levels, or embeddings contain NaN (Not a Number) or Infinity values, causing serialization failures and incorrect probabilistic operations.

**Symptoms**:
- Error: "Serialization failed: NaN values in embeddings"
- Error: "Invalid confidence interval: mean=NaN"
- JSON serialization failures
- Activation spreading returns infinite values
- Queries return invalid confidence scores

**Common Causes**:
- Division by zero in activation calculation
- log(0) or sqrt(negative) in confidence computation
- Floating-point overflow in embedding operations
- Corrupted data imported from external source

### Action

**Step 1: Identify NaN values**

```bash
# Check for NaN in recent memories via API
curl http://localhost:7432/api/v1/memories | \
  jq '.[] | select(.confidence | (isnan or isinfinite))'

# Check activation metrics
curl http://localhost:7432/metrics | grep engram_activation | grep -E "NaN|Inf"

# Review logs for numerical issues
journalctl -u engram | grep -E "NaN|Infinity|confidence.*invalid" | tail -20
```

**Step 2: Prevent further NaN propagation**

```bash
# Enable validation in config
# Edit config.toml:
cat >> ${ENGRAM_CONFIG:-~/.config/engram/config.toml} <<EOF

[validation]
check_finite_embeddings = true
check_finite_confidence = true
clamp_invalid_values = true
replace_nan_with_zero = true
EOF

# Restart Engram to apply
systemctl restart engram
```

**Step 3: Sanitize existing data**

```bash
# Use emergency recovery tool
./scripts/emergency_recovery.sh --sanitize-nan --dry-run

# Review what will be changed, then execute
./scripts/emergency_recovery.sh --sanitize-nan --backup-first
```

**Step 4: Identify root cause**

```bash
# Enable debug logging for numerical operations
export RUST_LOG=engram_core::activation=debug,engram_core::query=debug
systemctl restart engram

# Watch for division by zero warnings
journalctl -u engram -f | grep -E "division|sqrt|log|overflow"

# Check specific operations that might cause NaN:
# - Confidence interval calculation with empty data
# - Activation decay with invalid time delta
# - Pattern completion with zero-norm embeddings
```

### Verification

```bash
# All confidence values should be in [0, 1]
curl http://localhost:7432/api/v1/memories | \
  jq '.[] | select(.confidence < 0 or .confidence > 1 or (.confidence | isnan))'

# Should return empty array

# Test activation spreading (should not produce NaN)
curl -X POST http://localhost:7432/api/v1/activate \
  -H "Content-Type: application/json" \
  -d '{"cue":"test","max_depth":3}' | \
  jq '.activations[] | select(.level | (isnan or isinfinite))'

# Should return empty array

# Check metrics for NaN
curl http://localhost:7432/metrics | grep -E "NaN|Inf"
# Should return nothing
```

**Success criteria**:
- No NaN or Infinity in confidence scores
- Activation spreading completes without errors
- Serialization succeeds for all memories
- Validation catches future NaN at source

## Issue 8: Consolidation Stuck/Not Running

**Category**: Performance
**Expected Resolution Time**: 10-20 minutes
**Severity**: SEV3 (leads to WAL lag if prolonged)

### Context

Background consolidation process is not running or stuck, causing WAL to accumulate and memory patterns not to be detected.

**Symptoms**:
- WAL size growing indefinitely
- Disk usage increasing despite no new memories
- Metric `engram_consolidation_cycles_total` not increasing
- No "Consolidation cycle complete" in logs

**Common Causes**:
- Consolidation disabled in config
- Consolidation thread deadlocked
- Pattern detection timeout
- Insufficient memory for pattern detection

### Action

**Step 1: Check consolidation status**

```bash
# Check consolidation metrics
curl -G http://localhost:9090/api/v1/query \
  --data-urlencode 'query=rate(engram_consolidation_cycles_total[5m])'

# Should be > 0 if consolidation is running

# Check consolidation config
grep consolidation ${ENGRAM_CONFIG:-~/.config/engram/config.toml}

# Look for consolidation activity in logs
journalctl -u engram | grep -i consolidation | tail -50
```

**Step 2: Check for stuck threads**

```bash
# Check for stuck consolidation threads (Linux)
pstack $(pgrep engram) | grep -A 10 consolidation

# Check system resources
top -p $(pgrep engram)
# Look for high CPU or memory usage
```

**Step 3: Fix the issue**

**If consolidation is disabled**:

```toml
# Edit config.toml
[consolidation]
enabled = true
check_interval_secs = 60
idle_threshold_secs = 300
```

**If consolidation is stuck**:

```bash
# Restart Engram to unstick threads
systemctl restart engram

# Monitor to see if consolidation resumes
journalctl -u engram -f | grep consolidation
```

**If pattern detection times out**:

```toml
# Increase timeout and reduce requirements
[consolidation.pattern_detection]
timeout_secs = 300  # Increase from default 60
min_pattern_support = 3  # Reduce from default 5
min_confidence = 0.6  # Reduce from default 0.8
```

**If insufficient memory**:

```toml
# Reduce batch size
[consolidation]
max_batch_size = 500  # Reduce from default 1000
```

### Verification

```bash
# Watch consolidation progress
watch -n 10 'curl -s http://localhost:7432/api/v1/system/health | \
  jq "{consolidation_rate, wal_lag_ms, last_cycle}"'

# Consolidation rate should be > 0.0
# WAL lag should stabilize or decrease
# last_cycle timestamp should update every check_interval_secs

# Check for completion messages in logs
journalctl -u engram -f | grep "Consolidation cycle complete"
# Should appear every few minutes
```

**Success criteria**:
- Consolidation cycles completing regularly
- WAL lag decreasing or stable
- Pattern detection succeeding
- No timeout errors in logs

## Issue 9: Index Corruption

**Category**: Data Integrity
**Expected Resolution Time**: 20-40 minutes
**Severity**: SEV3 (performance degradation, not data loss)

### Context

Index files are corrupted, causing queries to fall back to slow linear search. System remains functional but performance degrades significantly.

**Symptoms**:
- Error: "Index corrupted or unavailable"
- Queries much slower than normal
- Alert: "IndexFallbackActive"
- Inconsistent query results
- "Falling back to linear search" in logs

**Common Causes**:
- Unclean shutdown during index write
- Disk corruption
- Software bug in index serialization
- Concurrent access without proper locking

### Action

**Step 1: Check index status**

```bash
# Check index health
curl http://localhost:7432/api/v1/system/health | jq '.indices'

# Look for status != "healthy"

# Look for index errors in logs
journalctl -u engram | grep -i index | grep -E "ERROR|corrupt|fallback"

# Verify index files exist
ls -lh ${ENGRAM_DATA_DIR:-./data}/*/indices/

# Check for index rebuild in progress
ps aux | grep engram | grep index_rebuild
```

**Step 2: Rebuild indices**

```bash
# Trigger index rebuild via API
./scripts/emergency_recovery.sh --rebuild-indices --dry-run

# Review plan, then execute
./scripts/emergency_recovery.sh --rebuild-indices

# Or via API directly
curl -X POST http://localhost:7432/api/v1/system/rebuild-indices

# For specific space
curl -X POST "http://localhost:7432/api/v1/system/rebuild-indices?space=tenant_a"

# Or via CLI
engram rebuild-indices --space all --background
```

**Step 3: Monitor rebuild**

```bash
# Watch rebuild progress
watch 'curl -s http://localhost:7432/api/v1/system/health | jq .indices'

# Check logs for completion
journalctl -u engram -f | grep "Index rebuild"
```

**Step 4: Prevent future corruption**

```toml
# Enable index checksums
# Edit config.toml:
[storage.indices]
enable_checksums = true
verify_on_load = true
auto_rebuild_on_corruption = true
```

**Fix permissions if needed**:

```bash
# Ensure index files are writable
sudo chown -R engram:engram ${ENGRAM_DATA_DIR:-./data}/*/indices/
sudo chmod -R 644 ${ENGRAM_DATA_DIR:-./data}/*/indices/*.idx
```

### Verification

```bash
# Query performance should improve after rebuild
time curl "http://localhost:7432/api/v1/query?cue=test&limit=100"
# Should complete in <1 second

# Index status should show "healthy"
curl http://localhost:7432/api/v1/system/health | \
  jq '.indices[] | select(.status != "healthy")'
# Should return empty

# No fallback warnings in recent logs
journalctl -u engram --since "5 minutes ago" | grep -i "fallback"
# Should return nothing

# Run performance benchmark
./scripts/benchmark_deployment.sh 60 10
grep "Index Hit Rate" /tmp/benchmark_report.txt
# Should be >95%
```

**Success criteria**:
- All indices show status "healthy"
- Query performance restored to normal
- No fallback to linear search
- Index hit rate >95%

## Issue 10: gRPC Connection Failures

**Category**: Configuration
**Expected Resolution Time**: 10-20 minutes
**Severity**: SEV3 (if HTTP works), SEV2 (if only gRPC available)

### Context

gRPC clients cannot connect to Engram, while HTTP endpoint may work fine. This affects clients using the gRPC API.

**Symptoms**:
- gRPC clients cannot connect
- Error: "failed to connect to all addresses"
- Error: "connection refused" on port 50051
- HTTP works but gRPC doesn't
- Timeout on gRPC method calls

**Common Causes**:
- gRPC not enabled in config
- Port 50051 blocked by firewall
- TLS misconfiguration
- gRPC client using wrong protocol (TLS vs plaintext)

### Action

**Step 1: Check if gRPC is enabled and listening**

```bash
# Check if gRPC port is open
sudo lsof -i :50051
netstat -tulpn | grep 50051

# Check gRPC config
grep grpc ${ENGRAM_CONFIG:-~/.config/engram/config.toml}

# Test gRPC endpoint
grpcurl -plaintext localhost:50051 list
```

**Step 2: Fix configuration**

**If gRPC is not enabled**:

```toml
# Edit config.toml
[grpc]
enabled = true
address = "0.0.0.0:50051"
```

**If port conflict**:

```bash
# Find what's using the port
sudo lsof -i :50051

# Kill conflicting process
sudo kill <PID>

# Or change Engram's gRPC port
# Edit config.toml:
# [grpc]
# address = "0.0.0.0:50052"
```

**If TLS misconfiguration**:

```toml
# For plaintext (development only)
[grpc]
enabled = true
address = "0.0.0.0:50051"
# No TLS config

# For TLS (production)
[grpc]
enabled = true
address = "0.0.0.0:50051"

[grpc.tls]
enabled = true
cert_path = "/etc/engram/tls/server.crt"
key_path = "/etc/engram/tls/server.key"
# Optional: client_ca_path = "/etc/engram/tls/ca.crt"
```

**If firewall blocking**:

```bash
# Check firewall rules
sudo iptables -L -n | grep 50051

# Open port (iptables)
sudo iptables -A INPUT -p tcp --dport 50051 -j ACCEPT

# Open port (firewalld)
sudo firewall-cmd --permanent --add-port=50051/tcp
sudo firewall-cmd --reload

# Open port (ufw)
sudo ufw allow 50051/tcp
```

**Step 3: Restart Engram**

```bash
systemctl restart engram

# Wait for startup
sleep 5
```

### Verification

```bash
# gRPC should respond to list command
grpcurl -plaintext localhost:50051 list

# Should show available services:
# engram.v1.EngramService

# List service methods
grpcurl -plaintext localhost:50051 list engram.v1.EngramService

# Should show methods like:
# engram.v1.EngramService.CreateMemory
# engram.v1.EngramService.Query
# etc.

# Test a simple gRPC call
grpcurl -plaintext -d '{"cue":"test","limit":10}' \
  localhost:50051 engram.v1.EngramService/Query

# Should return JSON response without errors

# If using TLS, test with TLS
grpcurl -cacert /etc/engram/tls/ca.crt \
  localhost:50051 list
```

**Success criteria**:
- `grpcurl list` succeeds
- gRPC methods are callable
- Client applications can connect
- No connection refused errors

## Navigation

- [Back to Troubleshooting Guide](./troubleshooting.md)
- [Incident Response Procedures](./incident-response.md)
- [Log Analysis Guide](./log-analysis.md)
- [Monitoring and Alerts](./monitoring.md)

## Getting Help

If you cannot resolve an issue using this guide:

1. Run diagnostic script: `./scripts/diagnose_health.sh`
2. Collect debug bundle: `./scripts/collect_debug_info.sh`
3. Review [Incident Response Guide](./incident-response.md) for escalation procedures
4. Contact: support@engram.example.com (include debug bundle)

For SEV1 incidents, page on-call engineer immediately and start incident response flow.
