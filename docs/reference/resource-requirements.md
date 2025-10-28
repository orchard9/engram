# Resource Requirements

Comprehensive resource specifications and sizing tables for Engram deployments. Use these tables to select appropriate instance types and configurations based on your workload characteristics.

## Quick Reference Tables

### Small Deployments (<1M Nodes)

| Workload Type | Nodes | Ops/sec | CPU Cores | Memory | Storage | Instance Type (AWS) | Monthly Cost |
|---------------|-------|---------|-----------|---------|---------|---------------------|--------------|
| Development   | 100K  | 100     | 2         | 4 GB    | 20 GB   | t3.medium           | ~$30         |
| Small Prod    | 500K  | 1,000   | 4         | 8 GB    | 50 GB   | m6i.xlarge          | ~$150        |
| Growing       | 1M    | 5,000   | 8         | 16 GB   | 100 GB  | m6i.2xlarge         | ~$300        |

### Medium Deployments (1M-10M Nodes)

| Workload Type | Nodes | Ops/sec | CPU Cores | Memory | Storage | Instance Type (AWS) | Monthly Cost |
|---------------|-------|---------|-----------|---------|---------|---------------------|--------------|
| Standard      | 5M    | 10K     | 16        | 32 GB   | 200 GB  | m6i.4xlarge         | ~$600        |
| Read-Heavy    | 5M    | 20K     | 32        | 64 GB   | 150 GB  | m6i.8xlarge         | ~$1,200      |
| Write-Heavy   | 5M    | 15K     | 16        | 48 GB   | 500 GB  | r6i.4xlarge         | ~$900        |
| Analytical    | 10M   | 5K      | 32        | 128 GB  | 300 GB  | r6i.8xlarge         | ~$1,800      |

### Large Deployments (10M-100M Nodes)

| Workload Type | Nodes | Ops/sec | CPU Cores | Memory  | Storage | Instance Type (AWS) | Monthly Cost |
|---------------|-------|---------|-----------|---------|---------|---------------------|--------------|
| Enterprise    | 50M   | 50K     | 64        | 256 GB  | 1 TB    | m6i.16xlarge        | ~$2,400      |
| High-Perf     | 100M  | 100K    | 96        | 384 GB  | 2 TB    | r6i.24xlarge        | ~$5,400      |

Note: Costs are approximate AWS on-demand prices as of 2024. Use spot instances for 60-70% savings on non-critical workloads.

## Detailed Sizing by Use Case

### Use Case 1: LLM Memory Augmentation

**Workload Characteristics:**
- Long-term memory for AI agents
- Read-heavy (80% reads, 20% writes)
- Fast retrieval required (<10ms P99)
- Large embedding size (768-1536 dimensions)

**Resource Requirements per 1M Memories:**

| Resource | Requirement | Calculation |
|----------|-------------|-------------|
| CPU | 4 cores | 2,500 retrievals/sec/core × 1.3 read multiplier |
| Memory | 6 GB | 100K hot (10%) × 3KB × 1.5 overhead + 500MB runtime |
| Storage | 25 GB | 1M × 3KB × 0.4 compression + WAL + snapshots |
| IOPS | 2,000 | 5,000 ops/sec × 0.3 cache miss × 1.5 overhead |
| Network | 200 Mbps | 5,000 req/sec × (512B + 4KB) × 8 bits |

**Recommended Configuration:**
```yaml
# For 10M memories, 50K retrievals/sec
instance: m6i.8xlarge (32 vCPU, 128 GB RAM)
storage: 500 GB NVMe SSD (gp3, 6000 IOPS)
cost: ~$1,200/month + $40/month storage
```

### Use Case 2: Knowledge Graph Queries

**Workload Characteristics:**
- Entity and relationship storage
- Graph traversals (1-5 hops average)
- Mixed read/write (60% reads, 40% writes)
- Complex pattern matching

**Resource Requirements per 1M Entities:**

| Resource | Requirement | Calculation |
|----------|-------------|-------------|
| CPU | 6 cores | Traversal overhead + spreading + consolidation |
| Memory | 8 GB | Graph structure + indices + traversal workspace |
| Storage | 40 GB | Entities + relationships + indices |
| IOPS | 5,000 | Random access for traversals |
| Network | 100 Mbps | Smaller payloads, more requests |

**Recommended Configuration:**
```yaml
# For 100M entities, 1B relationships
instance: m6i.16xlarge (64 vCPU, 256 GB RAM)
storage: 2 TB NVMe SSD (io2, 20000 IOPS)
cost: ~$2,400/month + $500/month storage
```

### Use Case 3: Time-Series Pattern Memory

**Workload Characteristics:**
- High-frequency event ingestion
- Write-heavy (90% writes, 10% reads)
- Time-based access patterns
- Pattern detection and anomaly analysis

**Resource Requirements per 1M Events/Day:**

| Resource | Requirement | Calculation |
|----------|-------------|-------------|
| CPU | 3 cores | Ingestion + consolidation overhead |
| Memory | 4 GB | Small hot window (last hour) |
| Storage | 100 GB/week | 1M/day × 7 days × 3KB × 0.4 compression |
| IOPS | 10,000 | High write throughput |
| Network | 50 Mbps | Ingestion traffic |

**Recommended Configuration:**
```yaml
# For 10M events/day (7-day retention)
instance: c6i.8xlarge (32 vCPU, 64 GB RAM)
storage: 5 TB NVMe SSD (gp3, 16000 IOPS)
cost: ~$1,100/month + $400/month storage
```

### Use Case 4: Semantic Search & RAG

**Workload Characteristics:**
- Document embeddings and retrieval
- Read-heavy with batch ingestion
- Similarity search (vector operations)
- GPU acceleration beneficial

**Resource Requirements per 1M Documents:**

| Resource | Requirement | Calculation |
|----------|-------------|-------------|
| CPU | 8 cores | Vector operations (or 4 cores + GPU) |
| Memory | 12 GB | Embeddings + similarity indices |
| Storage | 30 GB | Compressed document embeddings |
| GPU | Optional | 1× T4 for 10x spreading speedup |
| Network | 500 Mbps | Large embedding transfers |

**Recommended Configuration:**
```yaml
# For 50M documents, 10K searches/sec
instance: m6i.8xlarge (32 vCPU, 128 GB RAM)
gpu: g4dn.2xlarge (8 vCPU, 32 GB, 1× T4)  # Optional
storage: 1 TB NVMe SSD (gp3, 6000 IOPS)
cost: ~$1,200/month (CPU) or ~$700/month (GPU instance)
```

### Use Case 5: Real-Time Recommendation Engine

**Workload Characteristics:**
- User preferences and item embeddings
- Extremely low latency required (<5ms P99)
- High concurrency (100K+ req/sec)
- Frequent updates

**Resource Requirements per 1M Users:**

| Resource | Requirement | Calculation |
|----------|-------------|-------------|
| CPU | 64 cores | High concurrency + low latency target |
| Memory | 64 GB | Large hot tier (50% in memory) |
| Storage | 100 GB | User profiles + item embeddings |
| IOPS | 30,000 | Ultra-low latency reads |
| Network | 2 Gbps | High request volume |

**Recommended Configuration:**
```yaml
# For 10M users, 100K recommendations/sec
instance: m6i.24xlarge (96 vCPU, 384 GB RAM)
storage: 500 GB NVMe SSD (io2, 50000 IOPS)
cost: ~$3,600/month + $1,500/month storage
network: 10 Gbps link
```

## Resource Multipliers by Workload Pattern

### CPU Multipliers

| Pattern | Multiplier | Reason |
|---------|------------|--------|
| Baseline (mixed) | 1.0× | 60% read, 40% write |
| Read-heavy | 1.3× | More spreading activation |
| Write-heavy | 0.9× | More consolidation, less spreading |
| Analytical | 1.8× | Deep traversals, pattern completion |
| Real-time | 2.0× | Low latency requires overprovisioning |
| Batch | 0.7× | Latency tolerant, high utilization OK |

### Memory Multipliers

| Pattern | Multiplier | Reason |
|---------|------------|--------|
| Baseline (mixed) | 1.0× | Balanced tier distribution |
| Read-heavy | 1.0× | Stable hot tier |
| Write-heavy | 1.2× | Growing hot tier, buffering |
| Analytical | 1.5× | Large working sets |
| Real-time | 1.4× | Larger hot tier for speed |
| Batch | 0.8× | Can tolerate cold tier access |

### Storage Multipliers

| Pattern | Multiplier | Reason |
|---------|------------|--------|
| Baseline (mixed) | 1.0× | Standard WAL + snapshots |
| Read-heavy | 0.5× | Minimal WAL growth |
| Write-heavy | 1.5× | Large WAL, frequent snapshots |
| Analytical | 0.3× | Read-only or minimal writes |
| Real-time | 1.2× | Continuous updates |
| Batch | 1.0× | Periodic bulk loads |

## Cloud Provider Instance Recommendations

### AWS EC2 Instance Types

#### General Purpose (M-Series)

Best for: Balanced workloads, mixed read/write

| Instance Type | vCPU | Memory | Network | Cost/Month | Best For |
|---------------|------|--------|---------|------------|----------|
| m6i.xlarge | 4 | 16 GB | Up to 12.5 Gbps | ~$150 | Small production |
| m6i.2xlarge | 8 | 32 GB | Up to 12.5 Gbps | ~$300 | Medium production |
| m6i.4xlarge | 16 | 64 GB | Up to 12.5 Gbps | ~$600 | Large production |
| m6i.8xlarge | 32 | 128 GB | 12.5 Gbps | ~$1,200 | Enterprise |
| m6i.16xlarge | 64 | 256 GB | 25 Gbps | ~$2,400 | Very large |

#### Memory Optimized (R-Series)

Best for: Read-heavy, large hot tier, analytical

| Instance Type | vCPU | Memory | Network | Cost/Month | Best For |
|---------------|------|--------|---------|------------|----------|
| r6i.xlarge | 4 | 32 GB | Up to 12.5 Gbps | ~$250 | Memory-intensive small |
| r6i.2xlarge | 8 | 64 GB | Up to 12.5 Gbps | ~$500 | Memory-intensive medium |
| r6i.4xlarge | 16 | 128 GB | Up to 12.5 Gbps | ~$900 | Analytical workloads |
| r6i.8xlarge | 32 | 256 GB | 12.5 Gbps | ~$1,800 | Large analytical |
| r6i.16xlarge | 64 | 512 GB | 25 Gbps | ~$3,600 | Very large analytical |

#### Compute Optimized (C-Series)

Best for: Write-heavy, high throughput, low latency

| Instance Type | vCPU | Memory | Network | Cost/Month | Best For |
|---------------|------|--------|---------|------------|----------|
| c6i.2xlarge | 8 | 16 GB | Up to 12.5 Gbps | ~$270 | Ingestion pipelines |
| c6i.4xlarge | 16 | 32 GB | Up to 12.5 Gbps | ~$540 | High throughput |
| c6i.8xlarge | 32 | 64 GB | 12.5 Gbps | ~$1,100 | Very high throughput |

#### GPU Instances (G-Series)

Best for: Large-scale spreading, vector operations

| Instance Type | vCPU | GPU | Memory | Cost/Month | Best For |
|---------------|------|-----|---------|------------|----------|
| g4dn.xlarge | 4 | 1× T4 | 16 GB | ~$400 | GPU-accelerated small |
| g4dn.2xlarge | 8 | 1× T4 | 32 GB | ~$600 | GPU-accelerated medium |
| g4dn.12xlarge | 48 | 4× T4 | 192 GB | ~$3,900 | Large-scale GPU |

### GCP Compute Engine

#### Standard Instances (N-Series)

| Instance Type | vCPU | Memory | Best For | Cost/Month |
|---------------|------|--------|----------|------------|
| n2-standard-4 | 4 | 16 GB | Small production | ~$150 |
| n2-standard-8 | 8 | 32 GB | Medium production | ~$300 |
| n2-standard-16 | 16 | 64 GB | Large production | ~$600 |
| n2-standard-32 | 32 | 128 GB | Enterprise | ~$1,200 |

#### Memory Optimized

| Instance Type | vCPU | Memory | Best For | Cost/Month |
|---------------|------|--------|----------|------------|
| n2-highmem-8 | 8 | 64 GB | Memory-intensive | ~$500 |
| n2-highmem-16 | 16 | 128 GB | Large memory | ~$1,000 |
| n2-highmem-32 | 32 | 256 GB | Very large memory | ~$2,000 |

#### Compute Optimized

| Instance Type | vCPU | Memory | Best For | Cost/Month |
|---------------|------|--------|----------|------------|
| c2-standard-8 | 8 | 32 GB | High throughput | ~$350 |
| c2-standard-16 | 16 | 64 GB | Very high throughput | ~$700 |

### Azure Virtual Machines

#### General Purpose (D-Series)

| Instance Type | vCPU | Memory | Best For | Cost/Month |
|---------------|------|--------|----------|------------|
| Standard_D4s_v5 | 4 | 16 GB | Small production | ~$160 |
| Standard_D8s_v5 | 8 | 32 GB | Medium production | ~$320 |
| Standard_D16s_v5 | 16 | 64 GB | Large production | ~$640 |
| Standard_D32s_v5 | 32 | 128 GB | Enterprise | ~$1,280 |

#### Memory Optimized (E-Series)

| Instance Type | vCPU | Memory | Best For | Cost/Month |
|---------------|------|--------|----------|------------|
| Standard_E8s_v5 | 8 | 64 GB | Memory-intensive | ~$520 |
| Standard_E16s_v5 | 16 | 128 GB | Large memory | ~$1,040 |
| Standard_E32s_v5 | 32 | 256 GB | Very large memory | ~$2,080 |

## Storage Recommendations

### Storage Types by Use Case

| Use Case | Type | IOPS | Throughput | Cost/GB/Month | Best For |
|----------|------|------|------------|---------------|----------|
| Development | HDD/Standard | 500 | 60 MB/s | $0.04 | Non-critical, testing |
| Small Production | SSD/gp3 | 3,000 | 125 MB/s | $0.08 | Cost-effective |
| Medium Production | SSD/gp3 | 6,000 | 250 MB/s | $0.10 | Balanced performance |
| Large Production | SSD/gp3 | 16,000 | 500 MB/s | $0.15 | High throughput |
| Ultra-Low Latency | NVMe/io2 | 50,000+ | 1,000 MB/s | $0.30+ | Sub-ms latency |

### AWS EBS Volume Recommendations

| Workload | Volume Type | Size | IOPS | Throughput | Cost/Month |
|----------|-------------|------|------|------------|------------|
| Development | gp2 | 100 GB | 300 | 128 MB/s | ~$10 |
| Small Prod | gp3 | 200 GB | 3,000 | 125 MB/s | ~$16 |
| Medium Prod | gp3 | 500 GB | 6,000 | 250 MB/s | ~$50 |
| Large Prod | gp3 | 2 TB | 16,000 | 500 MB/s | ~$300 |
| Ultra Perf | io2 | 2 TB | 50,000 | 1,000 MB/s | ~$1,500 |

### GCP Persistent Disk Recommendations

| Workload | Disk Type | Size | IOPS | Cost/Month |
|----------|-----------|------|------|------------|
| Development | Standard | 100 GB | 300 | ~$4 |
| Small Prod | SSD | 200 GB | 6,000 | ~$35 |
| Medium Prod | SSD | 500 GB | 15,000 | ~$85 |
| Large Prod | SSD | 2 TB | 60,000 | ~$340 |

### Azure Managed Disk Recommendations

| Workload | Disk Type | Size | IOPS | Cost/Month |
|----------|-----------|------|------|------------|
| Development | Standard HDD | 128 GB | 500 | ~$5 |
| Small Prod | Premium SSD | 256 GB | 1,100 | ~$40 |
| Medium Prod | Premium SSD | 512 GB | 2,300 | ~$75 |
| Large Prod | Premium SSD | 2 TB | 7,500 | ~$300 |
| Ultra Perf | Ultra SSD | 2 TB | 50,000 | ~$1,000+ |

## Network Requirements

### Bandwidth by Workload

| Workload | Request Rate | Avg Request | Avg Response | Bandwidth | Recommended |
|----------|--------------|-------------|--------------|-----------|-------------|
| Small | 100 req/sec | 512 B | 4 KB | 4 Mbps | 100 Mbps |
| Medium | 1,000 req/sec | 512 B | 4 KB | 36 Mbps | 1 Gbps |
| Large | 10,000 req/sec | 512 B | 4 KB | 360 Mbps | 1 Gbps |
| Very Large | 100,000 req/sec | 512 B | 4 KB | 3.6 Gbps | 10 Gbps |

### Network Features by Cloud Provider

**AWS:**
- Up to 12.5 Gbps: Most instance types
- Up to 25 Gbps: Large instances (*.8xlarge, *.16xlarge)
- Enhanced Networking: ENA driver for low latency
- Network Cost: $0.01/GB outbound (inter-region)

**GCP:**
- 10 Gbps: Standard for most instances
- 32 Gbps: Large instances (32+ vCPU)
- Network Cost: $0.01/GB outbound (inter-region)

**Azure:**
- Accelerated Networking: Available on most sizes
- 25 Gbps: Large instances (16+ vCPU)
- Network Cost: $0.01/GB outbound (inter-region)

## Minimum Requirements

### Development Environment

**Absolute Minimum:**
```yaml
CPU: 2 cores
Memory: 4 GB
Storage: 20 GB SSD
Network: 100 Mbps
Cost: ~$30/month (t3.medium or equivalent)

Suitable for:
- Feature development
- Integration testing
- Small demos (<100K nodes)
```

### Production Minimum

**Small Production:**
```yaml
CPU: 4 cores
Memory: 8 GB
Storage: 50 GB NVMe SSD
Network: 1 Gbps
Cost: ~$150/month (m6i.xlarge or equivalent)

Suitable for:
- Small deployments (<1M nodes)
- Low traffic (<1K req/sec)
- Non-critical applications
```

## Scaling Thresholds

### When to Scale Up

| Resource | Current | Action | Target |
|----------|---------|--------|--------|
| CPU | >70% for 10+ min | Add 2x cores | 40-60% utilization |
| Memory | >80% for 5+ min | Add 50% RAM | 60-75% utilization |
| Storage | >70% used | Double capacity | 40-60% utilization |
| IOPS | >80% utilized | Upgrade disk type | 50-70% utilization |
| Network | >70% bandwidth | Upgrade instance | 40-60% utilization |

### Resource Headroom Recommendations

| Phase | CPU | Memory | Storage | Rationale |
|-------|-----|--------|---------|-----------|
| Initial (Day 0-7) | 2.0× | 2.0× | 2.0× | Unknown workload, safety margin |
| Optimization (Day 7-30) | 1.5× | 1.5× | 1.5× | Patterns identified, right-sizing |
| Steady State (Day 30+) | 1.3× | 1.3× | 1.5× | Predictable, minimal waste |

## Summary

- Use quick reference tables for initial sizing
- Choose instance type based on workload pattern
- Plan for growth with appropriate headroom
- Monitor actual utilization and right-size
- Consider spot instances for 60-70% cost savings
- Review and adjust quarterly

## Related Documentation

- [Capacity Planning](/operations/capacity-planning.md)
- [Scaling Guide](/operations/scaling.md)
- [How-to: Scale Vertically](/howto/scale-vertically.md)
- [Performance Tuning](/operations/performance-tuning.md)
- [Cost Optimization](/operations/cost-optimization.md)
