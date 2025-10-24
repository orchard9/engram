# Task 010: Configuration Reference & Best Practices — pending

**Priority:** P2
**Estimated Effort:** 2 days
**Dependencies:** Task 004 (Performance Tuning)

## Objective

Create comprehensive configuration documentation that transforms an operator from "what parameters exist?" to "I know exactly how to tune this for my workload." Think of configuration as a conversation, not a dictionary - every parameter needs a story about when to change it and what breaks if you get it wrong.

## Context

Engram's configuration spans multiple domains: tiered storage, spreading activation, consolidation, memory spaces, and persistence. Right now, parameters live scattered across different config structs with defaults hidden in code. Operators need a single source of truth with environment-specific templates and validation tooling.

The real challenge: cognitive graph databases have unfamiliar tuning knobs. "Hot tier capacity" isn't like "connection pool size" - operators don't have intuition yet. Documentation must bridge from familiar concepts (caching, memory limits, batch sizes) to cognitive primitives (activation thresholds, decay rates, consolidation frequencies).

## Key Deliverables

### 1. Complete Configuration Parameter Reference

**File:** `/docs/reference/configuration.md`

Comprehensive reference organized by configuration domain:

**CLI Configuration** (`~/.config/engram/config.toml`):
- `feature_flags.spreading_api_beta` (bool, default: true)
  - Purpose: Enable experimental spreading activation API endpoints
  - When to disable: If stability over features is critical for production
  - Impact: Disables `/api/v1/spread` and `/api/v1/activate` endpoints
  - Validation: Must be boolean (true/false)

- `memory_spaces.default_space` (string, default: "default")
  - Purpose: Default memory space ID for operations without explicit space header
  - Valid values: Alphanumeric, hyphens, underscores (3-64 characters)
  - When to change: Multi-tenant deployments should use tenant-specific defaults
  - Example: "tenant-acme-prod", "agent-assistant-v2"

- `memory_spaces.bootstrap_spaces` (array of strings, default: ["default"])
  - Purpose: Memory spaces to create automatically on startup
  - Valid values: Array of valid memory space IDs
  - When to change: Pre-provision spaces for known tenants
  - Example: `["tenant-a", "tenant-b", "shared"]`

**Persistence Configuration** (`persistence.*`):
- `persistence.data_root` (string, default: "~/.local/share/engram")
  - Purpose: Root directory for all memory space storage
  - Valid values: Absolute path or tilde-expanded home directory
  - When to change: Production deployments use `/var/lib/engram` or mounted volumes
  - Disk requirements: 2x expected memory corpus size for WAL + tiers
  - Example production: "/mnt/engram-data"

- `persistence.hot_capacity` (usize, default: 100000)
  - Purpose: Maximum memories in hot tier (in-memory DashMap) per space
  - Valid range: 1000 - 10000000
  - Memory impact: ~12KB per memory (embedding + metadata)
  - Tuning guide:
    - Development: 10000-50000 (120MB-600MB RAM)
    - Staging: 50000-100000 (600MB-1.2GB RAM)
    - Production: 100000-1000000 (1.2GB-12GB RAM)
  - When to increase: High working set, frequent access to same memories
  - When to decrease: Memory pressure, cold-data workloads
  - Monitoring: Watch `engram_storage_hot_tier_size` metric

- `persistence.warm_capacity` (usize, default: 1000000)
  - Purpose: Maximum memories in warm tier (memory-mapped files) per space
  - Valid range: 10000 - 100000000
  - Disk impact: ~10KB per memory in compressed format
  - Tuning guide:
    - Development: 100000-500000 (1GB-5GB disk)
    - Staging: 500000-2000000 (5GB-20GB disk)
    - Production: 1000000-10000000 (10GB-100GB disk)
  - When to increase: Large corpus, moderate access frequency
  - Performance note: Warm tier reads are 5-10x slower than hot tier
  - Monitoring: Watch `engram_storage_warm_tier_size` metric

- `persistence.cold_capacity` (usize, default: 10000000)
  - Purpose: Maximum memories in cold tier (columnar storage) per space
  - Valid range: 100000 - 1000000000
  - Disk impact: ~8KB per memory in columnar format
  - Tuning guide:
    - Development: 1000000 (8GB disk)
    - Staging: 5000000-10000000 (40GB-80GB disk)
    - Production: 10000000-100000000 (80GB-800GB disk)
  - When to increase: Long-term archival, large-scale knowledge graphs
  - Performance note: Cold tier reads are 50-100x slower than hot tier
  - Monitoring: Watch `engram_storage_cold_tier_size` metric

**Spreading Activation Configuration** (`ParallelSpreadingConfig`):
- `decay_rate` (f32, default: 0.85)
  - Purpose: Activation decay per hop during spreading
  - Valid range: 0.5-0.99
  - Effect: Lower values = more localized spreading, higher = broader search
  - Tuning guide:
    - Narrow context (0.70-0.80): Precise, local recalls
    - Balanced (0.80-0.90): General-purpose spreading
    - Broad context (0.90-0.95): Exploratory, distant associations
  - Example: Setting 0.75 means activation drops to 75% after each hop

- `threshold` (f32, default: 0.1)
  - Purpose: Minimum activation level to include in results
  - Valid range: 0.01-0.5
  - Effect: Lower values = more results (with lower confidence)
  - Tuning guide:
    - High precision (0.2-0.3): Few, confident results
    - Balanced (0.1-0.2): Standard production setting
    - High recall (0.05-0.1): Exploratory, cast wide net
  - Performance impact: Lower threshold = more memory operations

- `max_hops` (u16, default: 5)
  - Purpose: Maximum spreading distance from seed nodes
  - Valid range: 1-20
  - Effect: Higher values explore more distant associations
  - Tuning guide:
    - Direct connections (1-2): Immediate neighbors only
    - Local context (3-5): Standard spreading depth
    - Extended context (6-10): Deep relationship exploration
  - Performance impact: Latency grows exponentially with hops

- `max_workers` (usize, default: num_cpus::get())
  - Purpose: Thread pool size for parallel spreading
  - Valid range: 1 - 2x CPU cores
  - Tuning guide:
    - CPU-bound workloads: num_cpus
    - I/O-bound workloads: 2x num_cpus
    - Low latency priority: num_cpus / 2
  - Monitoring: Watch `engram_spreading_parallel_efficiency` metric

**Consolidation Configuration** (`ConsolidationConfig`):
- Pattern Detection:
  - `pattern_detection.min_cluster_size` (usize, default: 3)
    - Purpose: Minimum episodes to form a pattern cluster
    - Valid range: 2-20
    - Tuning: Higher = fewer, stronger patterns

  - `pattern_detection.similarity_threshold` (f32, default: 0.8)
    - Purpose: Cosine similarity threshold for clustering
    - Valid range: 0.5-0.95
    - Effect: Higher = more specific patterns, lower = broader themes

  - `pattern_detection.max_patterns` (usize, default: 100)
    - Purpose: Maximum patterns extracted per consolidation cycle
    - Valid range: 10-1000
    - Performance impact: Higher values increase memory usage

- Dream Configuration:
  - `dream.dream_duration` (Duration, default: 600s)
    - Purpose: Length of each offline consolidation cycle
    - Valid range: 60s-3600s
    - Production setting: 300-600s (5-10 minutes)

  - `dream.replay_speed` (f32, default: 15.0)
    - Purpose: Replay speed multiplier (faster than real-time)
    - Valid range: 10.0-20.0
    - Biological basis: Hippocampal replay is 10-20x faster

  - `dream.min_episode_age` (Duration, default: 86400s)
    - Purpose: Minimum age before episode eligible for consolidation
    - Valid range: 3600s-604800s (1 hour - 1 week)
    - Tuning: Shorter for rapid consolidation, longer for stability

- Compaction Configuration:
  - `compaction.min_age` (Duration, default: 86400s)
    - Purpose: Minimum episode age before compaction eligible
    - Valid range: 3600s-604800s

  - `compaction.target_compression_ratio` (f32, default: 0.5)
    - Purpose: Target size reduction after compaction
    - Valid range: 0.3-0.7
    - Effect: 0.5 means 50% size reduction goal

**Decay Configuration** (`DecayConfig`):
- `base_decay_rate` (f32, default: 0.001)
  - Purpose: Daily decay rate for unused memories
  - Valid range: 0.0001-0.01
  - Effect: Higher = faster forgetting
  - Biological basis: Matches Ebbinghaus forgetting curves

- `activation_boost` (f32, default: 0.1)
  - Purpose: Strength increase per memory access
  - Valid range: 0.01-0.5
  - Spaced repetition: Repeated access strengthens memory

### 2. Environment-Specific Configuration Templates

**File:** `/config/production.toml`
```toml
# Production Configuration Template
# High-capacity, multi-tenant deployment with full observability

[feature_flags]
spreading_api_beta = true

[memory_spaces]
default_space = "default"
bootstrap_spaces = ["default", "tenant-a", "tenant-b"]

[persistence]
data_root = "/var/lib/engram"
hot_capacity = 500_000    # 6GB RAM per space
warm_capacity = 5_000_000  # 50GB disk per space
cold_capacity = 50_000_000 # 400GB disk per space

[spreading]
decay_rate = 0.85
threshold = 0.15
max_hops = 5
max_workers = 16

[consolidation]
enabled = true
interval_seconds = 300  # 5 minutes
min_episode_age_seconds = 86400  # 1 day

[monitoring]
prometheus_port = 9090
log_level = "info"
tracing_enabled = true
```

**File:** `/config/staging.toml`
```toml
# Staging Configuration Template
# Medium capacity for pre-production testing

[feature_flags]
spreading_api_beta = true

[memory_spaces]
default_space = "staging"
bootstrap_spaces = ["staging"]

[persistence]
data_root = "/opt/engram-staging"
hot_capacity = 100_000    # 1.2GB RAM
warm_capacity = 1_000_000  # 10GB disk
cold_capacity = 10_000_000 # 80GB disk

[spreading]
decay_rate = 0.85
threshold = 0.1
max_hops = 5
max_workers = 8

[consolidation]
enabled = true
interval_seconds = 600  # 10 minutes
min_episode_age_seconds = 3600  # 1 hour for faster testing

[monitoring]
prometheus_port = 9091
log_level = "debug"
tracing_enabled = true
```

**File:** `/config/development.toml`
```toml
# Development Configuration Template
# Minimal resources for local development

[feature_flags]
spreading_api_beta = true

[memory_spaces]
default_space = "dev"
bootstrap_spaces = ["dev", "test"]

[persistence]
data_root = "~/.local/share/engram-dev"
hot_capacity = 10_000    # 120MB RAM
warm_capacity = 100_000  # 1GB disk
cold_capacity = 1_000_000 # 8GB disk

[spreading]
decay_rate = 0.85
threshold = 0.1
max_hops = 3  # Faster for interactive testing
max_workers = 4

[consolidation]
enabled = true
interval_seconds = 60  # 1 minute for rapid testing
min_episode_age_seconds = 1  # Immediate consolidation

[monitoring]
prometheus_port = 9092
log_level = "debug"
tracing_enabled = false  # Reduce overhead
```

### 3. Configuration Validation Tooling

**File:** `/scripts/validate_config.sh`

Shell script that validates configuration files before deployment:

```bash
#!/usr/bin/env bash
# Configuration validation for Engram deployments
#
# Usage: ./validate_config.sh [config_file]
# Exit codes: 0 = valid, 1 = invalid, 2 = file not found

set -euo pipefail

CONFIG_FILE="${1:-config.toml}"

# Validation rules
validate_config() {
    echo "Validating configuration: $CONFIG_FILE"

    # Check file exists
    if [[ ! -f "$CONFIG_FILE" ]]; then
        echo "ERROR: Config file not found: $CONFIG_FILE"
        exit 2
    fi

    # Parse TOML (requires toml-cli: cargo install toml-cli)
    if ! toml get "$CONFIG_FILE" &>/dev/null; then
        echo "ERROR: Invalid TOML syntax"
        exit 1
    fi

    # Validate persistence.hot_capacity
    hot_cap=$(toml get "$CONFIG_FILE" persistence.hot_capacity 2>/dev/null || echo "100000")
    if [[ "$hot_cap" -lt 1000 || "$hot_cap" -gt 10000000 ]]; then
        echo "ERROR: hot_capacity must be between 1000-10000000 (got: $hot_cap)"
        exit 1
    fi

    # Validate persistence.data_root exists
    data_root=$(toml get "$CONFIG_FILE" persistence.data_root 2>/dev/null || echo "")
    if [[ -z "$data_root" ]]; then
        echo "ERROR: persistence.data_root is required"
        exit 1
    fi

    # Validate spreading.decay_rate
    decay_rate=$(toml get "$CONFIG_FILE" spreading.decay_rate 2>/dev/null || echo "0.85")
    if ! awk "BEGIN {exit !($decay_rate >= 0.5 && $decay_rate <= 0.99)}"; then
        echo "ERROR: spreading.decay_rate must be between 0.5-0.99 (got: $decay_rate)"
        exit 1
    fi

    # All validations passed
    echo "✓ Configuration valid"
    return 0
}

validate_config
```

Add Rust-based validation in CLI:

**File:** `/engram-cli/src/cli/validate.rs`

```rust
use anyhow::{Result, Context, bail};
use crate::config::CliConfig;

pub fn validate_config(config: &CliConfig) -> Result<Vec<ValidationWarning>> {
    let mut warnings = Vec::new();

    // Validate hot capacity
    if config.persistence.hot_capacity < 1000 {
        bail!("hot_capacity too low: {} (minimum 1000)", config.persistence.hot_capacity);
    }
    if config.persistence.hot_capacity > 10_000_000 {
        bail!("hot_capacity too high: {} (maximum 10000000)", config.persistence.hot_capacity);
    }

    // Warn if hot > warm > cold ordering violated
    if config.persistence.hot_capacity > config.persistence.warm_capacity {
        warnings.push(ValidationWarning {
            level: WarnLevel::Warning,
            message: format!(
                "hot_capacity ({}) exceeds warm_capacity ({}). Consider increasing warm tier.",
                config.persistence.hot_capacity, config.persistence.warm_capacity
            ),
        });
    }

    // Validate data root is absolute path in production
    if !config.persistence.data_root.starts_with('/') &&
       !config.persistence.data_root.starts_with('~') {
        warnings.push(ValidationWarning {
            level: WarnLevel::Error,
            message: "data_root must be absolute path or tilde-expanded".to_string(),
        });
    }

    // Validate memory space IDs
    for space_id in &config.memory_spaces.bootstrap_spaces {
        if space_id.len() < 3 || space_id.len() > 64 {
            bail!("Memory space ID '{}' must be 3-64 characters", space_id);
        }
        if !space_id.chars().all(|c| c.is_alphanumeric() || c == '-' || c == '_') {
            bail!("Memory space ID '{}' contains invalid characters", space_id);
        }
    }

    Ok(warnings)
}

#[derive(Debug)]
pub struct ValidationWarning {
    pub level: WarnLevel,
    pub message: String,
}

#[derive(Debug)]
pub enum WarnLevel {
    Info,
    Warning,
    Error,
}
```

### 4. Best Practices Guide with Rationale

**File:** `/docs/operations/configuration-management.md`

Organized by deployment scenario with technical justification:

**Section: Capacity Planning**
- Calculate memory requirements: `hot_capacity * 12KB = hot tier RAM`
- Calculate disk requirements: `warm_capacity * 10KB + cold_capacity * 8KB = total disk`
- Rule of thumb: 80% utilization triggers tier migration (prevent thrashing)
- Example calculation for 1M memory corpus:
  - Hot tier: 100K memories = 1.2GB RAM
  - Warm tier: 500K memories = 5GB disk
  - Cold tier: 400K memories = 3.2GB disk
  - Total: 1.2GB RAM + 8.2GB disk

**Section: Spreading Tuning by Workload**
- Precision-focused workloads (legal, medical):
  - `decay_rate = 0.75`, `threshold = 0.25`, `max_hops = 3`
  - Rationale: Minimize false positives, prefer local context
- Exploratory workloads (research, discovery):
  - `decay_rate = 0.90`, `threshold = 0.05`, `max_hops = 8`
  - Rationale: Cast wide net, surface distant connections
- Balanced workloads (general assistant):
  - `decay_rate = 0.85`, `threshold = 0.15`, `max_hops = 5`
  - Rationale: Balance precision and recall

**Section: Consolidation Strategy**
- Aggressive consolidation (storage-constrained):
  - `interval_seconds = 300`, `min_episode_age = 3600`
  - Rationale: Rapid compaction, sacrifice some recent precision
- Conservative consolidation (accuracy-critical):
  - `interval_seconds = 3600`, `min_episode_age = 604800`
  - Rationale: Preserve episodic fidelity, let patterns stabilize
- Balanced (production default):
  - `interval_seconds = 600`, `min_episode_age = 86400`
  - Rationale: Daily consolidation, week to stabilize patterns

**Section: Multi-Tenant Isolation**
- One memory space per tenant (strong isolation)
- Shared bootstrap_spaces for common knowledge
- Per-space capacity limits prevent noisy neighbor issues
- Configuration example:
  ```toml
  [memory_spaces]
  default_space = "shared"
  bootstrap_spaces = ["shared", "tenant-a", "tenant-b", "tenant-c"]
  ```

### 5. Common Configuration Mistakes and Fixes

**File:** `/docs/operations/configuration-troubleshooting.md`

Real-world mistakes with concrete fixes:

**Mistake 1: Hot tier too small, constant thrashing**
- Symptom: High latency, `engram_storage_migrations_total` increasing rapidly
- Root cause: `hot_capacity = 1000` with 100K active working set
- Fix: Increase hot_capacity to 1.5x working set size
- Validation: `engram_storage_hot_tier_hit_rate` should be >0.95

**Mistake 2: Decay rate too aggressive, forgetting too fast**
- Symptom: Memories disappearing unexpectedly, low recall accuracy
- Root cause: `decay_rate = 0.95` with `base_decay_rate = 0.01`
- Fix: Reduce base_decay_rate to 0.001 or increase activation frequency
- Validation: Monitor `engram_memory_avg_strength` over time

**Mistake 3: Consolidation disabled, disk usage growing**
- Symptom: Disk usage grows unbounded, WAL segments accumulate
- Root cause: `consolidation.enabled = false` or very long intervals
- Fix: Enable consolidation with `interval_seconds = 600`
- Validation: `engram_storage_compaction_total` should increment regularly

**Mistake 4: Threshold too high, empty result sets**
- Symptom: Recall returns empty or very few results
- Root cause: `spreading.threshold = 0.5` filters out too many results
- Fix: Lower threshold to 0.1-0.15 for balanced recall
- Validation: `engram_spreading_results_empty_count` should be low

**Mistake 5: Max workers exceeds CPU cores**
- Symptom: High context switching, CPU saturation without throughput
- Root cause: `max_workers = 64` on 8-core machine
- Fix: Set max_workers = num_cpus (8 in this case)
- Validation: `engram_spreading_parallel_efficiency` should be >0.8

### 6. Configuration Examples for Different Use Cases

**File:** `/docs/howto/configure-for-production.md`

Scenario-based configuration with full explanations:

**Scenario 1: High-Throughput Agent System**
- Requirements: 100K ops/sec, 10M memory corpus, low latency
- Configuration:
  ```toml
  [persistence]
  hot_capacity = 1_000_000  # Large working set in RAM
  warm_capacity = 5_000_000
  cold_capacity = 50_000_000

  [spreading]
  decay_rate = 0.85
  threshold = 0.15
  max_hops = 3  # Limit depth for latency
  max_workers = 32  # Aggressive parallelism

  [consolidation]
  interval_seconds = 600  # Frequent but not blocking
  ```
- Hardware: 64GB RAM, 32 cores, NVMe SSD
- Expected performance: P99 < 10ms, 95% hit rate

**Scenario 2: Research Knowledge Graph**
- Requirements: 100M nodes, exploratory queries, deep connections
- Configuration:
  ```toml
  [persistence]
  hot_capacity = 100_000  # Small hot tier, mostly cold storage
  warm_capacity = 10_000_000
  cold_capacity = 100_000_000  # Massive cold tier

  [spreading]
  decay_rate = 0.90  # Broader spreading
  threshold = 0.05  # Lower threshold for exploration
  max_hops = 10  # Deep traversal
  max_workers = 16

  [consolidation]
  interval_seconds = 3600  # Hourly consolidation
  min_episode_age = 604800  # Week to stabilize patterns
  ```
- Hardware: 32GB RAM, 16 cores, 2TB HDD
- Expected performance: P99 < 100ms, 70% hit rate

**Scenario 3: Personal Assistant (Edge Device)**
- Requirements: Low memory, offline-capable, fast startup
- Configuration:
  ```toml
  [persistence]
  hot_capacity = 10_000  # Minimal RAM footprint
  warm_capacity = 50_000
  cold_capacity = 500_000

  [spreading]
  decay_rate = 0.85
  threshold = 0.1
  max_hops = 4
  max_workers = 2  # Limited CPU cores

  [consolidation]
  interval_seconds = 300  # Aggressive compaction
  min_episode_age = 3600  # Quick consolidation
  ```
- Hardware: 2GB RAM, 2 cores, 16GB flash storage
- Expected performance: P99 < 50ms, 90% hit rate

## Implementation Specifications

### Files to Create

1. **Configuration Reference**
   - `/docs/reference/configuration.md` (3000 lines)
   - Comprehensive parameter catalog with defaults, ranges, rationale

2. **Environment Templates**
   - `/config/production.toml` (100 lines)
   - `/config/staging.toml` (100 lines)
   - `/config/development.toml` (100 lines)
   - Copy-paste ready configurations

3. **Validation Tooling**
   - `/scripts/validate_config.sh` (200 lines)
   - `/engram-cli/src/cli/validate.rs` (300 lines)
   - Automated validation before deployment

4. **Best Practices**
   - `/docs/operations/configuration-management.md` (1500 lines)
   - Scenario-based tuning guides

5. **Troubleshooting**
   - `/docs/operations/configuration-troubleshooting.md` (800 lines)
   - Real-world mistakes and fixes

6. **How-To Guide**
   - `/docs/howto/configure-for-production.md` (1200 lines)
   - Complete configuration walkthroughs by use case

### Files to Modify

1. **CLI Config Module**
   - `/engram-cli/src/config.rs`
   - Add validation functions
   - Add display formatting for diagnostics

2. **CLI Commands**
   - `/engram-cli/src/cli/commands.rs`
   - Add `engram validate config` subcommand
   - Add `engram config check` for live config inspection

3. **Default Config**
   - `/engram-cli/config/default.toml`
   - Add inline comments explaining each parameter
   - Add validation ranges as comments

### Integration Points

1. **With Task 004 (Performance Tuning)**
   - Cross-reference tuning recommendations
   - Link configuration parameters to performance baselines
   - Include benchmark results for different config profiles

2. **With Task 003 (Monitoring)**
   - Map configuration parameters to monitoring metrics
   - Document which metrics validate configuration choices
   - Alert thresholds based on configuration

3. **With Task 001 (Deployment)**
   - Environment templates used in Docker/K8s deployments
   - ConfigMaps generated from validation scripts
   - Deployment-specific configuration overrides

## Acceptance Criteria

1. **Completeness**
   - [ ] All configuration structs documented (CliConfig, PersistenceConfig, MemorySpacesConfig, ParallelSpreadingConfig, ConsolidationConfig, PatternDetectionConfig, DreamConfig, DecayConfig)
   - [ ] Every parameter has: type, default, valid range, purpose, tuning guide, monitoring metric
   - [ ] Environment templates for dev/staging/production validated

2. **Validation Tooling**
   - [ ] Shell validation script catches all invalid configurations
   - [ ] Rust validation integrated into CLI (`engram validate config`)
   - [ ] Validation runs in <1 second for typical config
   - [ ] Error messages include parameter name, invalid value, valid range, suggestion

3. **Best Practices**
   - [ ] At least 5 complete scenario-based configurations
   - [ ] Each scenario includes: requirements, config, hardware, expected performance
   - [ ] Capacity planning formulas validated against real deployments
   - [ ] Multi-tenant isolation pattern documented

4. **Troubleshooting**
   - [ ] At least 10 common configuration mistakes documented
   - [ ] Each mistake includes: symptom, root cause, fix, validation
   - [ ] Fixes tested on real deployments
   - [ ] Monitoring queries provided for each symptom

5. **Developer Experience**
   - [ ] Configuration reference navigable in <3 clicks
   - [ ] Copy-paste templates work without modification on target environment
   - [ ] Validation catches 100% of invalid configs before deployment
   - [ ] Error messages actionable without consulting documentation

6. **External Review**
   - [ ] Configuration reference reviewed by external operator
   - [ ] At least one scenario configuration deployed successfully by external operator
   - [ ] Validation tooling catches real misconfigurations in testing

## Testing Strategy

### Unit Tests

Test configuration validation logic:

```rust
#[test]
fn test_hot_capacity_validation() {
    let mut config = CliConfig::default();

    // Valid capacity
    config.persistence.hot_capacity = 50000;
    assert!(validate_config(&config).is_ok());

    // Too low
    config.persistence.hot_capacity = 500;
    assert!(validate_config(&config).is_err());

    // Too high
    config.persistence.hot_capacity = 20_000_000;
    assert!(validate_config(&config).is_err());
}

#[test]
fn test_memory_space_id_validation() {
    let mut config = CliConfig::default();

    // Valid ID
    config.memory_spaces.default_space = "tenant-acme-prod".into();
    assert!(validate_config(&config).is_ok());

    // Invalid characters
    config.memory_spaces.default_space = "tenant@acme".into();
    assert!(validate_config(&config).is_err());

    // Too short
    config.memory_spaces.default_space = "ab".into();
    assert!(validate_config(&config).is_err());
}
```

### Integration Tests

Test configuration templates work in real deployments:

```bash
#!/bin/bash
# Test environment templates deploy successfully

set -e

# Test production template
cp config/production.toml /tmp/test-prod.toml
./scripts/validate_config.sh /tmp/test-prod.toml
engram --config /tmp/test-prod.toml validate

# Test staging template
cp config/staging.toml /tmp/test-staging.toml
./scripts/validate_config.sh /tmp/test-staging.toml
engram --config /tmp/test-staging.toml validate

# Test development template
cp config/development.toml /tmp/test-dev.toml
./scripts/validate_config.sh /tmp/test-dev.toml
engram --config /tmp/test-dev.toml validate

echo "All templates validated successfully"
```

### User Acceptance Tests

External operator test:
1. Operator receives only public documentation (no internal guidance)
2. Operator configures Engram for "high-throughput agent" scenario
3. Configuration validates successfully
4. Deployment achieves documented performance targets
5. Time to complete: Target <30 minutes

## Follow-Up Tasks

1. **Dynamic Reconfiguration** (Milestone 17)
   - Hot-reload configuration without restart
   - Per-memory-space configuration overrides
   - Runtime tuning based on metrics

2. **Configuration UI** (Milestone 18)
   - Web-based configuration editor
   - Visual capacity planning calculator
   - Real-time validation feedback

3. **Configuration Recommendations** (Milestone 19)
   - ML-based auto-tuning from workload patterns
   - Predictive capacity planning
   - Anomaly detection for misconfiguration

4. **Configuration Migration** (Milestone 16+)
   - Version migration tooling (v1 config → v2 config)
   - Deprecation warnings for old parameters
   - Automatic upgrade on startup

## Technical Notes

### Configuration Architecture

Engram uses layered configuration with this precedence (highest to lowest):
1. Command-line flags (`--hot-capacity 50000`)
2. Environment variables (`ENGRAM_HOT_CAPACITY=50000`)
3. User config file (`~/.config/engram/config.toml`)
4. System config file (`/etc/engram/config.toml`)
5. Default config (embedded in binary)

### Validation Philosophy

Configuration validation follows "fail fast, explain clearly":
- Invalid configs rejected at startup (no runtime surprises)
- Error messages include current value, expected range, example fix
- Warnings for suboptimal but valid configurations
- Validation completes in <1s (no blocking startup)

### Documentation Strategy

Configuration docs follow Julia Evans' "zines approach":
- Start with WHY parameter exists before HOW to tune it
- Use concrete examples with specific numbers (not "adjust as needed")
- Visual aids for capacity planning (ASCII tables, formulas)
- Real-world scenarios over abstract principles
- Every parameter maps to observable behavior (metrics, logs)

### Performance Considerations

Configuration validation overhead:
- TOML parsing: ~500µs for typical config
- Rust validation: ~100µs for all checks
- Shell validation: ~50ms (subprocess overhead)
- Total startup impact: <1ms

### Migration from Existing Configs

For operators migrating from Milestone 15:
- Old `config.toml` format remains compatible
- New parameters have sensible defaults
- Deprecation warnings for 2 releases before removal
- Migration script: `engram config migrate old.toml new.toml`

## Success Metrics

1. **Configuration Accuracy**
   - 100% of documented parameters exist in code
   - 100% of parameters have defaults, ranges, validation
   - 0 production incidents from misconfiguration

2. **Operator Velocity**
   - External operator configures production deployment in <30 minutes
   - 90% of operators use environment templates without modification
   - <5% of configurations fail validation

3. **Documentation Quality**
   - Configuration reference ranks in top 3 most-visited docs pages
   - Average time on page >5 minutes (indicates thorough reading)
   - <10% bounce rate from configuration docs (indicates completeness)

4. **Validation Effectiveness**
   - Validation catches 100% of invalid configurations in testing
   - 0 false positives (valid configs rejected)
   - <1% false negatives (invalid configs accepted)
