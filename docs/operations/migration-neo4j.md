# Neo4j to Engram Migration Guide

## Overview

Migrate existing Neo4j graph databases to Engram's cognitive memory graph. This tool transforms Neo4j nodes into Engram memories while preserving relationships and semantic content through automated embedding generation.

## Mapping Strategy

### Nodes to Memories

- **Node ID** → Memory ID (prefixed with `neo4j_node_`)

- **Node Labels** → Memory space selection (primary label determines space)

- **Node Properties** → JSON content, embedding from text fields

- **Creation Timestamp** → `Memory.created_at` (uses Neo4j audit properties if available)

- **Default Activation** → 0.5 (configurable)

- **Default Confidence** → MEDIUM (configurable)

### Relationships to Edges

- **Relationship Type** → Edge metadata

- **Relationship Properties** → Edge weight calculation

- **Default Edge Weight** → 0.7

- **Bidirectional** → Creates two edges (A→B and B→A)

## Prerequisites

- Neo4j 4.0+ running and accessible

- Engram instance running

- Network connectivity between migration tool and both databases

- Sufficient disk space for checkpoints (approximately 1MB per 100k records)

- Memory: 2GB+ recommended for large migrations

## Quick Start

### Basic Migration

```bash
migrate-neo4j \
  --source bolt://localhost:7687 \
  --source-user neo4j \
  --source-password yourpassword \
  --target http://localhost:8080 \
  --memory-space-prefix "neo4j" \
  --batch-size 1000

```

### With Custom Label Mapping

```bash
migrate-neo4j \
  --source bolt://localhost:7687 \
  --source-user neo4j \
  --source-password yourpassword \
  --target http://localhost:8080 \
  --label-to-space "Person:people,Company:companies,Product:products" \
  --batch-size 10000

```

### Resumable Migration with Checkpoints

```bash
migrate-neo4j \
  --source bolt://localhost:7687 \
  --source-user neo4j \
  --source-password yourpassword \
  --target http://localhost:8080 \
  --checkpoint-file /tmp/neo4j_migration.json \
  --validate

```

## Configuration

### Connection Options

- `--source` - Neo4j Bolt URI (e.g., `bolt://localhost:7687`)

- `--source-user` - Neo4j username

- `--source-password` - Neo4j password

- `--target` - Engram HTTP API endpoint

### Memory Space Options

- `--memory-space-prefix` - Prefix for auto-generated memory spaces (default: `neo4j`)

- `--label-to-space` - Explicit label to space mapping (format: `Label1:space1,Label2:space2`)

### Performance Options

- `--batch-size` - Records per batch (default: 1000, recommended: 5000-10000)

- `--checkpoint-file` - Path to checkpoint file for resumable migrations

### Migration Options

- `--dry-run` - Validate without writing to Engram

- `--validate` - Run validation checks after migration

- `--skip-edges` - Migrate only nodes, skip relationships

## Performance Tuning

### Batch Size

- **Small datasets (<10k nodes)**: 1000

- **Medium datasets (10k-1M nodes)**: 5000-10000

- **Large datasets (>1M nodes)**: 10000-50000

Higher batch sizes improve throughput but increase memory usage.

### Expected Throughput

- Simple nodes (few properties): 30k-50k nodes/sec

- Complex nodes (many properties): 10k-20k nodes/sec

- With relationship migration: 5k-10k nodes/sec

### Memory Usage

- Base overhead: ~500MB

- Per 10k nodes buffered: ~50MB

- Embedding cache: Variable (depends on content diversity)

## Validation

### Automatic Validation

Run with `--validate` flag to automatically validate:

1. **Count Validation**: Verify all nodes were migrated

2. **Sample Validation**: Check 1000 random nodes for correctness

3. **Edge Integrity**: Ensure no orphaned relationships

4. **Embedding Quality**: Check for zero embeddings

### Manual Validation

```bash
# Run validation script
./scripts/validate_migration.sh neo4j \
  bolt://localhost:7687 \
  http://localhost:8080 \
  neo4j_default

```

## Troubleshooting

### Connection Errors

**Error**: `Failed to connect to Neo4j`

**Solution**: Verify Neo4j is running and accessible. Check firewall rules and network connectivity.

### Out of Memory

**Error**: `OutOfMemoryError` during migration

**Solution**: Reduce `--batch-size` or increase JVM heap size for Neo4j.

### Slow Performance

**Issue**: Migration is slower than expected

**Solutions**:

- Increase `--batch-size`

- Ensure Neo4j has appropriate indexes on node IDs

- Check network latency between migration tool and databases

### Checkpoint Recovery

**Issue**: Migration failed mid-way

**Solution**: Re-run with the same `--checkpoint-file` to resume from the last checkpoint.

## Production Checklist

Before running a production migration:

- [ ] Back up source Neo4j database

- [ ] Test migration on a sample dataset

- [ ] Verify Engram instance has sufficient storage

- [ ] Configure checkpoint file path

- [ ] Set appropriate batch size based on testing

- [ ] Schedule migration during low-traffic window

- [ ] Monitor disk space and memory usage

- [ ] Plan for validation time (approximately 10% of migration time)

- [ ] Have rollback plan ready

## Advanced Usage

### Dry Run

Test the migration without writing to Engram:

```bash
migrate-neo4j \
  --source bolt://localhost:7687 \
  --source-user neo4j \
  --source-password yourpassword \
  --target http://localhost:8080 \
  --dry-run

```

### Skip Relationships

Migrate only nodes (faster for initial testing):

```bash
migrate-neo4j \
  --source bolt://localhost:7687 \
  --source-user neo4j \
  --source-password yourpassword \
  --target http://localhost:8080 \
  --skip-edges

```

## See Also

- [PostgreSQL Migration Guide](migration-postgresql.md)

- [Redis Migration Guide](migration-redis.md)

- [Migration Tutorial](../tutorials/migrate-from-neo4j.md)
