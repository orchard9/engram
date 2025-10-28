# Configure Engram for Production

Complete configuration walkthroughs for common production scenarios. Each scenario includes requirements, hardware specs, full configuration, and expected performance.

## Overview

This guide provides copy-paste configurations for three representative production workloads:

1. **High-Throughput Agent System** - 100K ops/sec, low latency
2. **Research Knowledge Graph** - 100M nodes, exploratory queries
3. **Personal Assistant (Edge Device)** - Resource-constrained, offline-capable

## Scenario 1: High-Throughput Agent System

### Requirements

- **Workload:** Multi-agent AI system with 1000 concurrent agents
- **Operations:** 100,000 ops/sec (reads + writes)
- **Corpus Size:** 10M memories
- **Latency Target:** P99 <10ms
- **Access Pattern:** High temporal locality (working set ~100K memories)

### Hardware Specification

```
CPU: 32 cores (64 vCPUs with hyperthreading)
RAM: 64GB
Disk: 1TB NVMe SSD (warm tier) + 4TB HDD (cold tier)
Network: 10 Gbps
OS: Linux (Ubuntu 22.04 LTS)

```

### Complete Configuration

**File:** `/etc/engram/config.toml`

```toml
#
# High-Throughput Agent System Configuration
# Target: 100K ops/sec, P99 <10ms
#

[feature_flags]
spreading_api_beta = true

[memory_spaces]
# Single shared space for all agents (soft isolation via memory IDs)
default_space = "agents"
bootstrap_spaces = ["agents"]

[persistence]
# Production FHS-compliant data root
data_root = "/var/lib/engram"

# Hot tier: Large working set in RAM
# 1M memories * 12KB = 12GB RAM
hot_capacity = 1_000_000

# Warm tier: Recently accessed, SSD-backed
# 5M memories * 10KB = 50GB SSD
warm_capacity = 5_000_000

# Cold tier: Long-term archive on HDD
# 10M memories * 8KB = 80GB HDD
cold_capacity = 10_000_000

```

### Spreading Configuration (in code)

```rust
let spreading_config = ParallelSpreadingConfig {
    decay_rate: 0.85,                // Balanced spreading
    threshold: 0.15,                 // Moderate filtering
    max_hops: 3,                     // Fast, localized spreading
    max_workers: 32,                 // Full CPU utilization
    latency_budget_ms: Some(10),     // Strict 10ms budget
};

```

### Consolidation Configuration (in code)

```rust
let consolidation_config = ConsolidationConfig {
    pattern_detection: PatternDetectionConfig {
        min_cluster_size: 3,
        similarity_threshold: 0.80,
        max_patterns: 100,
    },
    dream: DreamConfig {
        dream_duration: Duration::from_secs(600),     // 10min cycles
        min_episode_age: Duration::from_secs(86400),  // 1 day
        replay_speed: 15.0,
        ..Default::default()
    },
    compaction: CompactionConfig {
        min_age: Duration::from_secs(86400),
        target_compression_ratio: 0.5,
    },
};

```

### System Configuration

**Systemd service:** `/etc/systemd/system/engram.service`

```ini
[Unit]
Description=Engram Cognitive Graph Database
After=network.target

[Service]
Type=simple
User=engram
Group=engram
WorkingDirectory=/var/lib/engram
ExecStart=/usr/local/bin/engram server --config /etc/engram/config.toml

# Resource limits
MemoryMax=64G
CPUQuota=3200%  # 32 cores
IOWeight=500

# Restart policy
Restart=on-failure
RestartSec=5s

# Logging
StandardOutput=journal
StandardError=journal
SyslogIdentifier=engram

[Install]
WantedBy=multi-user.target

```

### Kernel Tuning

**File:** `/etc/sysctl.d/99-engram.conf`

```ini
# Network tuning
net.core.somaxconn = 4096
net.ipv4.tcp_max_syn_backlog = 8192
net.ipv4.ip_local_port_range = 1024 65535

# File descriptor limits
fs.file-max = 1000000

# Transparent huge pages (THP)
vm.nr_hugepages = 512

```

Apply with: `sudo sysctl -p /etc/sysctl.d/99-engram.conf`

### Expected Performance

```
Throughput: 100,000-150,000 ops/sec
P50 latency: <2ms
P99 latency: <10ms
P99.9 latency: <25ms

Hot tier hit rate: >95%
Warm tier hit rate: >90%
Cold tier access rate: <5%

Memory usage: ~12GB RAM (hot tier)
Disk usage: ~130GB total
CPU utilization: 60-80% (leaves headroom for spikes)

```

### Monitoring Queries

```promql
# Throughput
rate(engram_operations_total[1m])

# Latency percentiles
histogram_quantile(0.50, engram_operation_latency_seconds_bucket)
histogram_quantile(0.99, engram_operation_latency_seconds_bucket)
histogram_quantile(0.999, engram_operation_latency_seconds_bucket)

# Hit rates
engram_storage_hot_tier_hit_rate{space="agents"}
engram_storage_warm_tier_hit_rate{space="agents"}

# Resource utilization
engram_storage_memory_bytes{tier="hot"}
engram_storage_disk_usage_bytes{tier="warm"}
engram_storage_disk_usage_bytes{tier="cold"}

```

---

## Scenario 2: Research Knowledge Graph

### Requirements

- **Workload:** Academic research system with exploratory queries
- **Corpus Size:** 100M nodes (papers, authors, concepts, citations)
- **Access Pattern:** Sparse, exploratory (cold-data heavy)
- **Query Type:** Deep graph traversal, finding distant connections
- **Latency Target:** P99 <100ms (exploratory queries tolerate higher latency)

### Hardware Specification

```
CPU: 16 cores
RAM: 32GB
Disk: 2TB HDD (sufficient for cold-data workload)
Network: 1 Gbps
OS: Linux (Ubuntu 22.04 LTS)

```

### Complete Configuration

**File:** `/etc/engram/config.toml`

```toml
#
# Research Knowledge Graph Configuration
# Target: 100M corpus, exploratory queries
#

[feature_flags]
spreading_api_beta = true

[memory_spaces]
default_space = "research"
bootstrap_spaces = ["research"]

[persistence]
data_root = "/var/lib/engram-research"

# Hot tier: Small working set (most queries touch cold data)
# 100K memories * 12KB = 1.2GB RAM
hot_capacity = 100_000

# Warm tier: Moderate recently-accessed cache
# 10M memories * 10KB = 100GB disk
warm_capacity = 10_000_000

# Cold tier: Massive archive for full corpus
# 100M memories * 8KB = 800GB disk
cold_capacity = 100_000_000

```

### Spreading Configuration (optimized for exploration)

```rust
let spreading_config = ParallelSpreadingConfig {
    decay_rate: 0.90,                // Slow decay for distant connections
    threshold: 0.05,                 // Low threshold for broad recall
    max_hops: 10,                    // Deep traversal across concept domains
    max_workers: 16,                 // Match CPU cores
    latency_budget_ms: None,         // No budget, prioritize completeness
};

```

### Consolidation Configuration

```rust
let consolidation_config = ConsolidationConfig {
    pattern_detection: PatternDetectionConfig {
        min_cluster_size: 5,         // Strong patterns (research themes)
        similarity_threshold: 0.75,   // Broader thematic grouping
        max_patterns: 500,           // Extract many research patterns
    },
    dream: DreamConfig {
        dream_duration: Duration::from_secs(3600),    // 1-hour cycles
        min_episode_age: Duration::from_secs(604800),  // 1 week (let patterns stabilize)
        replay_speed: 10.0,
        ..Default::default()
    },
    compaction: CompactionConfig {
        min_age: Duration::from_secs(604800),  # 1 week
        target_compression_ratio: 0.3,         # Aggressive compaction (storage-constrained)
    },
};

```

### Expected Performance

```
Throughput: 100-1,000 ops/sec (query-heavy, not high-throughput)
P50 latency: <20ms
P99 latency: <100ms
P99.9 latency: <500ms

Hot tier hit rate: ~70% (acceptable for cold-data workload)
Warm tier hit rate: ~85%
Cold tier access rate: ~15-30% (expected for exploratory queries)

Memory usage: ~1.2GB RAM
Disk usage: ~900GB total
CPU utilization: 30-50% (query processing, not saturation)

```

### Optimizations

**Use HDD efficiently:**

```bash
# Enable read-ahead for sequential cold tier scans
sudo blockdev --setra 8192 /dev/sda

# Disable write barriers (HDD-specific)
# Add 'nobarrier' to fstab mount options

```

**Memory-mapped file tuning:**

```bash
# Increase vm.max_map_count for warm tier mmap
sudo sysctl -w vm.max_map_count=262144

```

---

## Scenario 3: Personal Assistant (Edge Device)

### Requirements

- **Workload:** Personal knowledge assistant on laptop/edge device
- **Corpus Size:** 500K memories (personal notes, emails, documents)
- **Resource Constraints:** Limited RAM, battery-powered
- **Access Pattern:** Recent memories (temporal locality)
- **Offline Capability:** Must work without network

### Hardware Specification

```
CPU: 4 cores (Intel Core i5 or Apple M-series)
RAM: 8GB (shared with OS and other apps)
Disk: 256GB SSD
Platform: macOS or Linux
Power: Battery-powered (optimize for efficiency)

```

### Complete Configuration

**File:** `~/.config/engram/config.toml`

```toml
#
# Personal Assistant Edge Configuration
# Target: Low memory footprint, fast startup, offline-capable
#

[feature_flags]
spreading_api_beta = true

[memory_spaces]
default_space = "personal"
bootstrap_spaces = ["personal"]

[persistence]
# User directory (suitable for single-user personal device)
data_root = "~/.local/share/engram"

# Hot tier: Minimal RAM footprint
# 10K memories * 12KB = 120MB RAM
hot_capacity = 10_000

# Warm tier: Recently accessed personal memories
# 50K memories * 10KB = 500MB disk
warm_capacity = 50_000

# Cold tier: Full personal knowledge archive
# 500K memories * 8KB = 4GB disk
cold_capacity = 500_000

```

### Spreading Configuration (optimized for efficiency)

```rust
let spreading_config = ParallelSpreadingConfig {
    decay_rate: 0.85,                // Standard spreading
    threshold: 0.10,                 // Moderate filtering
    max_hops: 4,                     // Limited depth for speed
    max_workers: 2,                  // Conservative (save cores for other apps)
    latency_budget_ms: Some(50),     // Snappy interactive latency
};

```

### Consolidation Configuration (aggressive for space)

```rust
let consolidation_config = ConsolidationConfig {
    pattern_detection: PatternDetectionConfig {
        min_cluster_size: 2,         // Detect patterns early
        similarity_threshold: 0.80,
        max_patterns: 50,
    },
    dream: DreamConfig {
        dream_duration: Duration::from_secs(300),     // 5min cycles
        min_episode_age: Duration::from_secs(3600),   // 1 hour (rapid consolidation)
        replay_speed: 20.0,                           // Fast replay
        ..Default::default()
    },
    compaction: CompactionConfig {
        min_age: Duration::from_secs(3600),  // 1 hour
        target_compression_ratio: 0.3,       // Aggressive (space-constrained)
    },
};

```

### LaunchAgent Configuration (macOS)

**File:** `~/Library/LaunchAgents/com.engram.personal.plist`

```xml
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.engram.personal</string>
    <key>ProgramArguments</key>
    <array>
        <string>/usr/local/bin/engram</string>
        <string>server</string>
        <string>--config</string>
        <string>/Users/YOUR_USERNAME/.config/engram/config.toml</string>
    </array>
    <key>RunAtLoad</key>
    <true/>
    <key>KeepAlive</key>
    <true/>
    <key>StandardOutPath</key>
    <string>/Users/YOUR_USERNAME/Library/Logs/engram.log</string>
    <key>StandardErrorPath</key>
    <string>/Users/YOUR_USERNAME/Library/Logs/engram-error.log</string>
</dict>
</plist>

```

Load with: `launchctl load ~/Library/LaunchAgents/com.engram.personal.plist`

### Expected Performance

```
Throughput: 10-100 ops/sec (personal use, not high-volume)
P50 latency: <10ms
P99 latency: <50ms

Hot tier hit rate: >90% (recent memories accessed frequently)
Warm tier hit rate: >85%

Memory usage: ~120MB RAM (minimal footprint)
Disk usage: ~4.5GB total
CPU utilization: <10% (efficient, battery-friendly)

Startup time: <1 second (fast for interactive use)

```

### Battery Optimization

**Reduce consolidation frequency when on battery:**

```rust
// Detect battery status and adjust consolidation
let consolidation_config = if on_battery_power() {
    ConsolidationConfig {
        dream: DreamConfig {
            dream_duration: Duration::from_secs(1800),  // 30min (less frequent)
            ..Default::default()
        },
        ..Default::default()
    }
} else {
    ConsolidationConfig::default()  // Normal when plugged in
};

```

---

## Validation Checklist

For any production deployment:

### 1. Validate Configuration

```bash
# Shell validation
./scripts/validate_config.sh /etc/engram/config.toml --deployment production

# CLI validation
engram validate config --config /etc/engram/config.toml --deployment production

```

### 2. Verify Resource Availability

```bash
# Check RAM
free -h  # Ensure sufficient free memory

# Check disk space
df -h /var/lib/engram

# Check CPU
lscpu  # Verify core count matches config

```

### 3. Test Configuration

```bash
# Start Engram with test config
engram server --config /etc/engram/config.toml --dry-run

# Should output:
# "Configuration valid, would start server with:"
# ... (config summary)

```

### 4. Run Deployment Validation

```bash
engram validate deployment --environment production

```

### 5. Monitor Initial Startup

```bash
# Start server
systemctl start engram

# Watch logs for errors
journalctl -u engram -f

# Check metrics endpoint
curl http://localhost:9090/metrics

```

### 6. Load Test

```bash
# Run load test against staging first
./tools/loadtest/run.sh --config staging-config.toml --duration 300s

# Validate performance meets targets
# Check P99 latency, hit rates, throughput

```

---

## Further Reading

- [Configuration Reference](/docs/reference/configuration.md) - Complete parameter details
- [Configuration Management](/docs/operations/configuration-management.md) - Best practices
- [Configuration Troubleshooting](/docs/operations/configuration-troubleshooting.md) - Common mistakes
- [Performance Tuning](/docs/operations/performance-tuning.md) - Advanced optimization
- [Production Deployment](/docs/operations/production-deployment.md) - Deployment procedures
