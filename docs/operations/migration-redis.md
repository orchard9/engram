# Redis to Engram Migration Guide

## Overview

Migrate Redis cache and data store contents to Engram's cognitive memory graph, mapping TTL values to decay rates and preserving Redis data structure semantics.

## Mapping Strategy

### Keys to Memories

- **Key Name** → Memory ID (prefixed with `redis_`)

- **Value** → Memory content (JSON-encoded with type metadata)

- **TTL** → Memory decay rate

- **Key Type** → Metadata in content

### TTL to Decay Rate Conversion

```
decay_rate = 1.0 / (ttl_seconds / 3600.0)

Examples:
  TTL = 3600s (1 hour)  → decay_rate = 1.0
  TTL = 86400s (1 day)  → decay_rate = 0.042
  No TTL                → decay_rate = 0.01 (slow background decay)

```

### Value Type Handling

#### String

- Direct string value as content

- Embedding from string content

- High confidence

#### Hash

- Serialized as JSON object

- Embedding from concatenated field values

- Option to split into separate memories per field

#### List

- Serialized as JSON array

- Embedding from concatenated elements

- Order preserved in metadata

#### Set

- Serialized as JSON array

- Embedding from concatenated members

- Marked as unordered

#### Sorted Set (ZSet)

- Member scores → Initial activation levels

- Creates edges between consecutive members

- Preserves ranking relationships

## Prerequisites

- Redis 5.0+ running and accessible

- Engram instance running

- Access to RDB file (for offline migration) or live Redis connection

- Disk space for checkpoints

## Quick Start

### Live Migration (SCAN)

```bash
migrate-redis \
  --source redis://localhost:6379 \
  --source-db 0 \
  --target http://localhost:8080 \
  --memory-space "redis_cache" \
  --ttl-as-decay \
  --batch-size 1000

```

### RDB File Migration (Faster)

```bash
migrate-redis \
  --source redis://localhost:6379 \
  --target http://localhost:8080 \
  --use-rdb /var/lib/redis/dump.rdb \
  --ttl-as-decay \
  --batch-size 10000

```

### Multiple Databases

```bash
# Migrate database 0
migrate-redis \
  --source redis://localhost:6379 \
  --source-db 0 \
  --target http://localhost:8080 \
  --memory-space "redis_db0"

# Migrate database 1
migrate-redis \
  --source redis://localhost:6379 \
  --source-db 1 \
  --target http://localhost:8080 \
  --memory-space "redis_db1"

```

## Configuration

### Connection Options

- `--source` - Redis connection URI

- `--source-db` - Redis database number (default: 0)

- `--target` - Engram HTTP API endpoint

### Migration Strategy

- `--memory-space` - Memory space for Redis keys (default: `redis_cache`)

- `--use-rdb` - Path to RDB file for offline migration

- `--ttl-as-decay` - Map TTL values to decay rates

### Performance Options

- `--batch-size` - Keys per batch
  - SCAN mode: 1000-5000
  - RDB mode: 5000-10000

## Performance Tuning

### Migration Modes

#### SCAN Mode (Live Migration)

- **Throughput**: 1k-5k keys/sec

- **Memory**: <500MB

- **Advantage**: No downtime

- **Disadvantage**: Slower

#### RDB Mode (Offline Migration)

- **Throughput**: 10k-100k keys/sec

- **Memory**: <500MB

- **Advantage**: Much faster

- **Disadvantage**: Requires RDB file access

### Batch Size Recommendations

- **Small keys (<1KB)**: 5000-10000

- **Medium keys (1KB-10KB)**: 1000-5000

- **Large keys (>10KB)**: 100-1000

## Data Type Specifics

### Sorted Sets (ZSets)

Sorted sets receive special handling:

1. **Scores normalized** to [0, 1] range for activation levels

2. **Rank-based edges** created between adjacent members

3. **Useful for**: Leaderboards, ranked lists, time-series data

Example:

```
ZADD leaderboard 100 "user1" 200 "user2" 300 "user3"

Becomes:

- Memory: redis_leaderboard

- Edges: user1 → user2 → user3 (rank order)

- Activations: user1=0.33, user2=0.67, user3=1.0

```

### Expiring Keys

Keys with TTL are mapped to fast-decaying memories:

```
SET session:abc123 "user_data" EX 3600

Becomes:

- Memory: redis_session:abc123

- Decay rate: 1.0 (will decay in ~1 hour)

- Content: "user_data"

```

## Validation

### Automatic Validation

Run with `--validate` to check:

1. **Key count** matches source

2. **Sample validation** of 1000 random keys

3. **TTL mapping** accuracy (<5% error)

4. **Embedding quality**

### Manual Validation

```bash
./scripts/validate_migration.sh redis \
  redis://localhost:6379 \
  http://localhost:8080 \
  redis_cache

```

## Troubleshooting

### Connection Timeout

**Error**: `Connection timeout to Redis`

**Solution**: Check Redis `timeout` configuration. For migrations, set to 0 (no timeout):

```
CONFIG SET timeout 0

```

### Memory Pressure on Redis

**Issue**: Redis memory usage spikes during SCAN

**Solution**: Use RDB migration mode or reduce batch size.

### Large Keys

**Warning**: `Key user:data is 50MB, migration may be slow`

**Solution**: Consider excluding very large keys or migrating them separately with smaller batch size.

## Production Checklist

Before production migration:

- [ ] Back up Redis with SAVE or BGSAVE

- [ ] Test migration on replica if available

- [ ] Verify Engram has sufficient storage

- [ ] Choose migration mode (SCAN vs RDB)

- [ ] Schedule during low-traffic window (for SCAN mode)

- [ ] Monitor Redis memory and CPU during migration

- [ ] Plan for validation time

- [ ] Document TTL decay rate mapping

## Advanced Usage

### Dry Run

Test without writing to Engram:

```bash
migrate-redis \
  --source redis://localhost:6379 \
  --target http://localhost:8080 \
  --dry-run

```

### Checkpoint and Resume

For very large Redis instances:

```bash
migrate-redis \
  --source redis://localhost:6379 \
  --target http://localhost:8080 \
  --checkpoint-file /tmp/redis_migration.json \
  --batch-size 5000

```

If interrupted, resume with the same command.

## See Also

- [Neo4j Migration Guide](migration-neo4j.md)

- [PostgreSQL Migration Guide](migration-postgresql.md)

- [Decay Rate Configuration](../reference/decay-rates.md)
