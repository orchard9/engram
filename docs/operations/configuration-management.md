# Configuration Management Best Practices

This guide provides scenario-based configuration strategies for Engram deployments. Unlike the parameter reference which explains what each knob does, this shows when and why to turn them.

## Overview

Engram's configuration spans multiple domains with unfamiliar tuning parameters. This guide bridges from familiar deployment patterns to cognitive graph primitives through worked examples and capacity planning formulas.

## Capacity Planning

### Memory Requirements Formula

Calculate hot tier RAM per memory space:

```
hot_tier_ram = hot_capacity * 12KB

```

**Example calculations:**

```
100,000 memories  = 1.2GB RAM
500,000 memories  = 6GB RAM
1,000,000 memories = 12GB RAM

```

**Multi-space deployments:**

```
total_ram = hot_capacity * 12KB * num_spaces

```

Example: 3 spaces with 500K hot capacity each:

```
total_ram = 500,000 * 12KB * 3 = 18GB

```

**Rule of thumb:** Keep hot tier allocation under 80% of total RAM to leave headroom for:

- OS and system processes (20-30% of RAM)
- Spreading activation working sets (10-20% of RAM)
- Consolidation buffers (5-10% of RAM)

### Disk Requirements Formula

**Warm tier disk:**

```
warm_tier_disk = warm_capacity * 10KB * num_spaces

```

**Cold tier disk:**

```
cold_tier_disk = cold_capacity * 8KB * num_spaces

```

**Total disk with overhead:**

```
total_disk = (warm_tier_disk + cold_tier_disk) * 1.2

```

The 1.2 multiplier accounts for:

- WAL segments (10%)
- Indexes and metadata (5%)
- Compaction working space (5%)

**Example: 1M corpus, single space:**

```
Hot: 100K * 12KB = 1.2GB RAM
Warm: 500K * 10KB = 5GB disk
Cold: 400K * 8KB = 3.2GB disk
Total: 1.2GB RAM + 9.8GB disk (12GB with overhead)

```

## Spreading Activation Tuning by Workload

### Precision-Focused Workloads

**Use cases:** Legal research, medical diagnosis, financial analysis

**Goal:** Minimize false positives, prefer local context

**Configuration:**

```rust
ParallelSpreadingConfig {
    decay_rate: 0.75,    // Tight decay, activation dies quickly
    threshold: 0.25,     // High threshold, only strong matches
    max_hops: 3,         // Limit to immediate neighborhood
    max_workers: 16,     // Standard parallelism
    latency_budget_ms: Some(100),  // 100ms budget for interactive use
}

```

**Why these settings:**

- `decay_rate = 0.75`: After 3 hops, activation drops to 0.75^3 = 0.42, ensuring distant nodes rarely qualify
- `threshold = 0.25`: Filters out weak associations, returning only high-confidence matches
- `max_hops = 3`: Prevents traversing into unrelated concept clusters

**Expected behavior:**

- Result sets: 10-50 memories
- P99 latency: <50ms
- Precision: >0.95 (few false positives)
- Recall: ~0.70 (misses some relevant but distant associations)

### Exploratory Workloads

**Use cases:** Research, discovery, brainstorming, literature review

**Goal:** Cast wide net, surface distant connections

**Configuration:**

```rust
ParallelSpreadingConfig {
    decay_rate: 0.90,    // Slow decay, activation reaches far
    threshold: 0.05,     // Low threshold, include weak signals
    max_hops: 8,         // Deep traversal
    max_workers: 32,     // High parallelism for large frontier
    latency_budget_ms: None,  // No budget, prioritize completeness
}

```

**Why these settings:**

- `decay_rate = 0.90`: After 8 hops, activation is 0.90^8 = 0.43, still above threshold
- `threshold = 0.05`: Includes weak but potentially interesting associations
- `max_hops = 8`: Reaches across concept boundaries

**Expected behavior:**

- Result sets: 100-1000 memories
- P99 latency: <1s (exploratory queries tolerate higher latency)
- Precision: ~0.60 (more false positives acceptable)
- Recall: >0.90 (catches nearly all relevant memories)

### Balanced General-Purpose

**Use cases:** Personal assistants, general knowledge retrieval, Q&A systems

**Goal:** Balance precision and recall for everyday queries

**Configuration:**

```rust
ParallelSpreadingConfig {
    decay_rate: 0.85,    // Moderate decay
    threshold: 0.15,     // Balanced threshold
    max_hops: 5,         // Standard depth
    max_workers: 16,     // CPU-matched parallelism
    latency_budget_ms: Some(200),  // 200ms budget for good UX
}

```

**Why these settings:**

- `decay_rate = 0.85`: Default setting, well-tested across workloads
- `threshold = 0.15`: Filters noise while preserving useful associations
- `max_hops = 5`: Reaches typical semantic distance in knowledge graphs

**Expected behavior:**

- Result sets: 20-100 memories
- P99 latency: <100ms
- Precision: ~0.80
- Recall: ~0.80

## Consolidation Strategy Selection

### Aggressive Consolidation

**Use cases:** Storage-constrained deployments, cost-optimized cloud

**Goal:** Rapid compaction, sacrifice some episodic precision

**Configuration:**

```rust
ConsolidationConfig {
    pattern_detection: PatternDetectionConfig {
        min_cluster_size: 2,        // Detect patterns early
        similarity_threshold: 0.75,  // Broader clustering
        max_patterns: 200,          // Extract many patterns
    },
    dream: DreamConfig {
        dream_duration: Duration::from_secs(300),   // 5-minute cycles
        min_episode_age: Duration::from_secs(3600), // 1 hour
        replay_speed: 20.0,                         // Fast replay
        ..Default::default()
    },
    compaction: CompactionConfig {
        min_age: Duration::from_secs(3600),         // Compact after 1 hour
        target_compression_ratio: 0.3,              // 70% size reduction
    },
}

```

**Trade-offs:**

- Pros: Rapid space reclamation, lower storage costs
- Cons: Some recent episodic details lost, patterns may be premature

**Monitoring:**

- Watch `engram_storage_disk_usage_bytes` - should decrease after consolidation
- Monitor `engram_consolidation_patterns_detected_total` - should be high
- Check `engram_compaction_ratio` - target >0.7 (70% reduction)

### Conservative Consolidation

**Use cases:** Accuracy-critical systems, episodic fidelity required

**Goal:** Preserve episodic details, let patterns fully stabilize

**Configuration:**

```rust
ConsolidationConfig {
    pattern_detection: PatternDetectionConfig {
        min_cluster_size: 5,        // Strong evidence required
        similarity_threshold: 0.85,  // High similarity needed
        max_patterns: 50,           // Fewer, stronger patterns
    },
    dream: DreamConfig {
        dream_duration: Duration::from_secs(1800),   // 30-minute cycles
        min_episode_age: Duration::from_secs(604800), // 1 week
        replay_speed: 10.0,                          // Slow, careful replay
        ..Default::default()
    },
    compaction: CompactionConfig {
        min_age: Duration::from_secs(604800),        // Compact after 1 week
        target_compression_ratio: 0.7,               // 30% size reduction
    },
}

```

**Trade-offs:**

- Pros: High episodic fidelity, well-validated patterns
- Cons: Higher storage costs, delayed pattern availability

**Monitoring:**

- Watch `engram_consolidation_patterns_detected_total` - should be lower
- Monitor `engram_consolidation_cluster_coherence` - should be >0.85
- Check episode retention with `engram_storage_episode_count`

### Balanced Production Default

**Use cases:** General production deployments, multi-purpose systems

**Goal:** Daily consolidation, weekly pattern stabilization

**Configuration:**

```rust
ConsolidationConfig {
    pattern_detection: PatternDetectionConfig {
        min_cluster_size: 3,        // Standard threshold
        similarity_threshold: 0.80,  // Balanced similarity
        max_patterns: 100,          // Moderate pattern extraction
    },
    dream: DreamConfig {
        dream_duration: Duration::from_secs(600),    // 10-minute cycles
        min_episode_age: Duration::from_secs(86400),  // 1 day
        replay_speed: 15.0,                          // Biological default
        ..Default::default()
    },
    compaction: CompactionConfig {
        min_age: Duration::from_secs(86400),         // Compact after 1 day
        target_compression_ratio: 0.5,               // 50% size reduction
    },
}

```

## Multi-Tenant Isolation Patterns

### Strong Isolation (Separate Spaces)

**Use case:** SaaS with strict tenant data separation

**Pattern:** One memory space per tenant

**Configuration:**

```toml
[memory_spaces]
default_space = "shared"  # Shared knowledge space
bootstrap_spaces = ["shared", "tenant-a", "tenant-b", "tenant-c"]

[persistence]
# Capacity per space
hot_capacity = 100_000    # 1.2GB RAM per tenant
warm_capacity = 1_000_000  # 10GB disk per tenant
cold_capacity = 10_000_000 # 80GB disk per tenant

```

**Capacity planning:**

For N tenants:

```
Total RAM: hot_capacity * 12KB * N + (hot_capacity * 12KB)  # +1 for shared
Total Disk: (warm + cold) * (10KB + 8KB) * N + shared_space_disk

```

Example with 10 tenants:

```
RAM:  100K * 12KB * 11 = 13.2GB
Disk: (1M * 10KB + 10M * 8KB) * 11 = 990GB

```

**Benefits:**

- Complete data isolation
- Independent capacity scaling
- Per-tenant performance tuning

**Trade-offs:**

- Higher RAM usage (no sharing of hot tier)
- More complex capacity management

### Soft Isolation (Shared Space with Namespacing)

**Use case:** Internal teams, less strict separation requirements

**Pattern:** Single shared space with namespace prefixes

**Configuration:**

```toml
[memory_spaces]
default_space = "shared"
bootstrap_spaces = ["shared"]

[persistence]
# Shared capacity across all tenants
hot_capacity = 1_000_000  # 12GB RAM total
warm_capacity = 10_000_000 # 100GB disk total
cold_capacity = 100_000_000 # 800GB disk total

```

**Memory ID convention:** `{tenant}:{memory_id}`

**Benefits:**

- Lower RAM usage (shared hot tier)
- Simpler capacity management
- Natural knowledge sharing between teams

**Trade-offs:**

- No hard isolation (all data in same space)
- Noisy neighbor risk (one tenant can fill hot tier)

## Monitoring-Driven Tuning

### Hot Tier Hit Rate Tuning

**Goal:** Maintain >95% hit rate for optimal latency

**Check:**

```promql
engram_storage_hot_tier_hit_rate{space="default"}

```

**If hit rate <90%:**

1. Check working set size: `engram_storage_hot_tier_size`
2. Compare to capacity: `engram_storage_hot_capacity`
3. If size consistently near capacity, increase `hot_capacity`

**Calculation:**

```
new_hot_capacity = current_hot_capacity * 1.5  # 50% increase

```

**Validate:**

- Deploy new config
- Monitor hit rate for 24 hours
- Should stabilize >95%

### Spreading Latency Optimization

**Goal:** P99 latency <100ms for interactive workloads

**Check:**

```promql
histogram_quantile(0.99, engram_spreading_latency_seconds_bucket)

```

**If P99 >100ms:**

**Option 1: Reduce depth**

```rust
max_hops: 5 -> 3  # Halves traversal depth

```

**Option 2: Increase threshold**

```rust
threshold: 0.10 -> 0.15  # Filters more aggressively

```

**Option 3: Add more workers (if CPU underutilized)**

```rust
max_workers: 16 -> 32  # Doubles parallelism

```

**Validate each change:**

```promql
# Check P99 latency after change
histogram_quantile(0.99, engram_spreading_latency_seconds_bucket)

# Ensure parallel efficiency stays >0.80
engram_spreading_parallel_efficiency

```

### Consolidation Cycle Tuning

**Goal:** Complete cycles within dream_duration

**Check:**

```promql
engram_consolidation_dream_duration_seconds

```

**If duration > dream_duration:**

**Option 1: Reduce max_patterns**

```rust
max_patterns: 100 -> 50  # Halves pattern extraction work

```

**Option 2: Increase min_cluster_size**

```rust
min_cluster_size: 3 -> 5  # Fewer patterns pass threshold

```

**Option 3: Increase min_episode_age**

```rust
min_episode_age: Duration::from_secs(86400) -> Duration::from_secs(172800)  # 2 days

```

## Common Configuration Mistakes

See `/docs/operations/configuration-troubleshooting.md` for detailed troubleshooting of common misconfigurations.

## Further Reading

- [Configuration Reference](/docs/reference/configuration.md) - Complete parameter documentation
- [Configuration Troubleshooting](/docs/operations/configuration-troubleshooting.md) - Common mistakes and fixes
- [Configure for Production](/docs/howto/configure-for-production.md) - Complete deployment scenarios
- [Performance Tuning](/docs/operations/performance-tuning.md) - Advanced optimization strategies
