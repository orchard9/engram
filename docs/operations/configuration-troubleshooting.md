# Configuration Troubleshooting

Real-world configuration mistakes with concrete symptoms, root causes, fixes, and validation steps. Each example comes from actual deployments.

## Hot Tier Thrashing

### Symptom

- High latency (P99 >100ms)
- `engram_storage_migrations_total{from="hot",to="warm"}` increasing rapidly
- `engram_storage_hot_tier_hit_rate` <0.90
- CPU usage moderate but latency high

### Root Cause

Hot tier capacity too small for working set size. When hot tier fills up, least-recently-used memories evict to warm tier. If these memories get accessed again soon, they migrate back to hot tier, creating thrashing.

### Fix

Increase hot_capacity to 1.5x-2x current working set:

```toml
[persistence]
hot_capacity = 250_000  # Was 100,000

```

### Validation

Monitor for 24 hours:

```promql
# Hit rate should improve to >0.95
engram_storage_hot_tier_hit_rate{space="default"}

# Migration rate should decrease
rate(engram_storage_migrations_total{from="hot",to="warm"}[5m])

# P99 latency should drop
histogram_quantile(0.99, engram_spreading_latency_seconds_bucket)

```

### Cost

RAM increase: 150,000 * 12KB = 1.8GB additional RAM

---

## Aggressive Decay Causing Memory Loss

### Symptom

- Memories disappearing unexpectedly
- Low recall accuracy on known-good queries
- `engram_memory_avg_strength` declining over time
- Users reporting "forgotten" information

### Root Cause

`base_decay_rate` too aggressive combined with infrequent access. Memories decay faster than they're being reinforced.

**Example bad config:**

```rust
DecayConfig {
    base_decay_rate: 0.01,  // 1% daily decay - TOO AGGRESSIVE
    activation_boost: 0.05,  // Weak reinforcement
    ..Default::default()
}

```

After 1 week without access: `strength * (1 - 0.01)^7 = strength * 0.93` (7% loss)

After 1 month: `strength * (1 - 0.01)^30 = strength * 0.74` (26% loss)

### Fix

Reduce base_decay_rate to match access patterns:

```rust
DecayConfig {
    base_decay_rate: 0.001,  // 0.1% daily decay
    activation_boost: 0.1,   // Standard reinforcement
    ..Default::default()
}

```

After 1 month: `strength * (1 - 0.001)^30 = strength * 0.97` (3% loss)

### Validation

```promql
# Average strength should stabilize or increase
engram_memory_avg_strength

# Forgotten memories rate should decrease
rate(engram_memory_forgotten_total[1h])

```

---

## Consolidation Disabled, Disk Growing Unbounded

### Symptom

- Disk usage growing linearly with time
- WAL segments accumulating in `data_root/*/wal/`
- `engram_storage_disk_usage_bytes` increasing without bound
- No `engram_storage_compaction_total` metrics

### Root Cause

Consolidation disabled or interval too long. Episodes accumulate without being compacted into semantic patterns.

**Example bad config:**

```rust
ConsolidationConfig {
    dream: DreamConfig {
        dream_duration: Duration::from_secs(86400),  // 24 hours - TOO LONG
        ..Default::default()
    },
    ..Default::default()
}

```

### Fix

Enable consolidation with reasonable interval:

```rust
ConsolidationConfig {
    dream: DreamConfig {
        dream_duration: Duration::from_secs(600),  // 10 minutes
        min_episode_age: Duration::from_secs(86400),  // 1 day
        ..Default::default()
    },
    compaction: CompactionConfig {
        min_age: Duration::from_secs(86400),
        target_compression_ratio: 0.5,
    },
}

```

### Validation

```promql
# Compaction should occur regularly
rate(engram_storage_compaction_total[1h])

# Disk usage should stabilize or decrease
engram_storage_disk_usage_bytes{tier="warm"}

# Episodes should decrease after consolidation
engram_storage_episode_count

```

---

## Spreading Threshold Too High, Empty Results

### Symptom

- Recall returns empty or very few results (<5)
- `engram_spreading_results_empty_count` high
- Users reporting "system can't find anything"
- `engram_spreading_results_count` P50 <10

### Root Cause

`threshold` set too high, filtering out valid results. Activation values rarely exceed threshold even for good matches.

**Example bad config:**

```rust
ParallelSpreadingConfig {
    decay_rate: 0.75,  # Low decay compounds the problem
    threshold: 0.5,    # TOO HIGH
    max_hops: 3,
    ..Default::default()
}

```

After 3 hops: `activation = 1.0 * 0.75^3 = 0.42` (below threshold!)

### Fix

Lower threshold to balanced setting:

```rust
ParallelSpreadingConfig {
    decay_rate: 0.85,  # Moderate decay
    threshold: 0.15,   # BALANCED
    max_hops: 5,
    ..Default::default()
}

```

After 5 hops: `activation = 1.0 * 0.85^5 = 0.44` (above threshold)

### Validation

```promql
# Empty results should decrease
rate(engram_spreading_results_empty_count[5m])

# Result counts should increase
histogram_quantile(0.50, engram_spreading_results_count_bucket)

```

---

## Max Workers Exceeds CPU Cores

### Symptom

- High context switching (`vmstat` shows high `cs` column)
- CPU saturation without proportional throughput
- `engram_spreading_parallel_efficiency` <0.70
- System feels sluggish despite available CPU

### Root Cause

`max_workers` set too high for available cores. Thread contention and context switching overhead dominate actual work.

**Example bad config on 8-core machine:**

```rust
ParallelSpreadingConfig {
    max_workers: 64,  # TOO MANY for 8 cores
    ..Default::default()
}

```

### Fix

Match `max_workers` to CPU cores:

```rust
ParallelSpreadingConfig {
    max_workers: 8,   # Match physical cores
    ..Default::default()
}

```

For I/O-bound workloads, 2x cores acceptable:

```rust
ParallelSpreadingConfig {
    max_workers: 16,  # 2x cores for I/O overlap
    ..Default::default()
}

```

### Validation

```promql
# Parallel efficiency should improve
engram_spreading_parallel_efficiency  # Target: >0.80

# Worker utilization should be balanced
engram_spreading_worker_utilization

```

System metrics should show reduced context switching:

```bash
vmstat 1 5  # Watch 'cs' column - should decrease

```

---

## Tier Capacity Inversion

### Symptom

- Warnings in logs: "warm tier full, cannot evict from hot tier"
- `engram_storage_hot_tier_size` stuck at capacity
- Memories can't migrate properly between tiers

### Root Cause

Tier capacities configured in wrong order: hot > warm or warm > cold.

**Example bad config:**

```toml
[persistence]
hot_capacity = 500_000   # Largest tier!
warm_capacity = 100_000  # Smaller than hot - WRONG
cold_capacity = 50_000   # Smaller than warm - WRONG

```

### Fix

Ensure proper tier ordering: hot < warm < cold

```toml
[persistence]
hot_capacity = 100_000    # Smallest (most frequently accessed)
warm_capacity = 1_000_000  # Medium (recently accessed)
cold_capacity = 10_000_000 # Largest (archive)

```

### Validation

Configuration should pass validation:

```bash
./scripts/validate_config.sh config.toml

```

Should see:

```
[PASS] hot_capacity <= warm_capacity <= cold_capacity

```

---

## Multi-Space RAM Exhaustion

### Symptom

- Out-of-memory (OOM) errors
- System swapping heavily
- `engram_storage_memory_bytes` near system limits
- Kernel OOM killer terminating Engram process

### Root Cause

Didn't account for hot tier RAM multiplied by number of spaces.

**Example bad config with 10 spaces:**

```toml
[memory_spaces]
bootstrap_spaces = ["space1", "space2", ..., "space10"]

[persistence]
hot_capacity = 500_000  # 6GB RAM per space
# Total: 6GB * 10 = 60GB RAM (exceeds 32GB system!)

```

### Fix

Calculate total RAM requirement first:

```
total_ram = hot_capacity * 12KB * num_spaces

```

Ensure <80% of system RAM:

```toml
[memory_spaces]
bootstrap_spaces = ["space1", ..., "space10"]

[persistence]
hot_capacity = 200_000  # 2.4GB RAM per space
# Total: 2.4GB * 10 = 24GB RAM (fits in 32GB system)

```

### Validation

Check actual memory usage:

```bash
ps aux | grep engram  # Check RSS column

```

Monitor memory metrics:

```promql
# Total memory usage across all spaces
sum(engram_storage_memory_bytes{tier="hot"})

```

---

## Data Root Permissions

### Symptom

- Engram fails to start
- Error: "Permission denied" in logs
- Cannot create directories in `data_root`

### Root Cause

`data_root` directory not writable by Engram process user.

**Example bad config:**

```toml
[persistence]
data_root = "/var/lib/engram"  # Owned by root, Engram runs as 'engram' user

```

### Fix

Create directory with correct ownership:

```bash
sudo mkdir -p /var/lib/engram
sudo chown engram:engram /var/lib/engram
sudo chmod 755 /var/lib/engram

```

Or use user-writable path:

```toml
[persistence]
data_root = "/home/engram/data"

```

### Validation

Test write permissions:

```bash
sudo -u engram touch /var/lib/engram/.test
sudo -u engram rm /var/lib/engram/.test

```

Should succeed without errors.

---

## Memory Space ID Invalid Characters

### Symptom

- Configuration validation fails
- Error: "Invalid memory space ID"
- Cannot create memory spaces

### Root Cause

Memory space IDs contain invalid characters or wrong length.

**Example bad config:**

```toml
[memory_spaces]
default_space = "tenant@acme"  # '@' is invalid
bootstrap_spaces = ["ab", "tenant_1", "this-is-a-very-long-space-name-that-exceeds-64-characters-limit"]

```

### Fix

Use only alphanumeric, hyphens, underscores, 3-64 characters:

```toml
[memory_spaces]
default_space = "tenant-acme"  # Valid
bootstrap_spaces = ["dev", "tenant-1", "production-workspace"]

```

### Validation

Run config validation:

```bash
engram validate config --config config.toml

```

Should pass with:

```
[INFO] memory_spaces.default_space is valid: tenant-acme
[INFO] memory_spaces.bootstrap_spaces validated: 3 space(s)

```

---

## Production Using Home Directory

### Symptom

- Data lost when user account removed
- Backup scripts can't find data
- Permission issues on multi-user systems

### Root Cause

Production deployment using `~/.local/share/engram` instead of system directory.

**Example bad config:**

```toml
[persistence]
data_root = "~/.local/share/engram"  # User directory

```

### Fix

Use FHS-compliant system directory:

```toml
[persistence]
data_root = "/var/lib/engram"  # System directory

```

### Validation

Check directory location:

```bash
readlink -f ~/.local/share/engram  # Should NOT be used in production
ls -ld /var/lib/engram              # Should exist with proper ownership

```

---

## Latency Budget Too Strict

### Symptom

- Many partial results
- `engram_spreading_latency_budget_violations_total` increasing
- `engram_spreading_partial_results_total` high
- Users getting incomplete answers

### Root Cause

`latency_budget_ms` too low for actual query complexity.

**Example bad config:**

```rust
ParallelSpreadingConfig {
    max_hops: 8,                      # Deep traversal
    latency_budget_ms: Some(50),      # 50ms budget - TOO STRICT
    ..Default::default()
}

```

### Fix

Increase budget or reduce max_hops:

**Option 1: Increase budget**

```rust
ParallelSpreadingConfig {
    max_hops: 8,
    latency_budget_ms: Some(500),     # 500ms budget
    ..Default::default()
}

```

**Option 2: Reduce depth**

```rust
ParallelSpreadingConfig {
    max_hops: 5,                      # Shallower traversal
    latency_budget_ms: Some(100),     # 100ms budget
    ..Default::default()
}

```

### Validation

```promql
# Budget violations should decrease
rate(engram_spreading_latency_budget_violations_total[5m])

# Partial results should decrease
rate(engram_spreading_partial_results_total[5m])

# P99 latency should be under budget
histogram_quantile(0.99, engram_spreading_latency_seconds_bucket) < (latency_budget_ms / 1000)

```

---

## Further Reading

- [Configuration Reference](/docs/reference/configuration.md) - Parameter details
- [Configuration Management](/docs/operations/configuration-management.md) - Best practices
- [Performance Tuning](/docs/operations/performance-tuning.md) - Advanced optimization
- [Monitoring](/docs/operations/monitoring.md) - Metrics and alerts
