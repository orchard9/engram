# PostgreSQL to Engram Migration Guide

## Overview

Migrate relational PostgreSQL databases to Engram's cognitive memory graph, preserving foreign key relationships as semantic edges and converting rows to memories with context-aware embeddings.

## Mapping Strategy

### Tables to Memory Spaces

- **One memory space per table** (default)
- **Custom mapping** via `--table-to-space` configuration
- **Example**: `users` table → `user_memories` space

### Rows to Memories

- **Primary Key** → Memory ID (e.g., `users_123`)
- **Text Columns** → Concatenated for embedding generation
- **Timestamp Columns** → `Memory.created_at`
- **JSON Columns** → Parsed and merged into content
- **All Columns** → JSON-serialized in `Memory.content`

### Foreign Keys to Edges

- **FK Relationship** → Directed edge (child → parent)
- **Edge Weight** → 0.8 (high confidence for referential integrity)
- **Multi-column FKs** → Single edge with composite metadata

## Prerequisites

- PostgreSQL 12+ running and accessible
- Engram instance running
- Read permissions on all tables to migrate
- Disk space for checkpoints
- Memory: 1GB+ recommended

## Quick Start

### Basic Migration

```bash
migrate-postgresql \
  --source "postgresql://user:password@localhost/mydb" \
  --target http://localhost:8080 \
  --batch-size 1000
```

### With Custom Table Mapping

```bash
migrate-postgresql \
  --source "postgresql://user:password@localhost/mydb" \
  --target http://localhost:8080 \
  --table-to-space "users:user_space,orders:order_space,products:product_space" \
  --text-columns "users:name,bio,notes;orders:description,comments" \
  --batch-size 5000
```

### Parallel Migration

```bash
migrate-postgresql \
  --source "postgresql://user:password@localhost/mydb" \
  --target http://localhost:8080 \
  --parallel-workers 8 \
  --batch-size 5000 \
  --checkpoint-file /tmp/pg_migration.json
```

## Configuration

### Connection Options

- `--source` - PostgreSQL connection string
- `--target` - Engram HTTP API endpoint

### Schema Mapping

- `--table-to-space` - Explicit table to memory space mapping
- `--text-columns` - Columns to use for embedding generation
- `--timestamp-column` - Column name for creation timestamp (default: `created_at`)

### Performance Options

- `--batch-size` - Rows per batch (default: 1000, recommended: 5000)
- `--parallel-workers` - Number of parallel table extractors (default: 4)
- `--checkpoint-file` - Path to checkpoint file

### Migration Options

- `--dry-run` - Validate without writing
- `--validate` - Run validation after migration

## Performance Tuning

### Batch Size

- **Small tables (<100k rows)**: 1000
- **Medium tables (100k-10M rows)**: 5000-10000
- **Large tables (>10M rows)**: 10000-20000

### Parallel Workers

- **Few large tables**: 1-2 workers per table
- **Many small tables**: 4-8 workers
- **Mixed workload**: 4-6 workers

### Expected Throughput

- Narrow tables (few columns): 15k-20k rows/sec
- Wide tables (many columns): 5k-10k rows/sec
- With heavy text content: 2k-5k rows/sec

## Referential Integrity

The migration tool automatically:

1. **Analyzes schema** to build FK dependency graph
2. **Topologically sorts tables** (parents before children)
3. **Migrates in dependency order** to preserve relationships
4. **Validates all FK edges** exist in Engram
5. **Reports orphaned FKs** for manual resolution

## Validation

### Automatic Validation

Run with `--validate` flag to check:

1. **Row counts** match source database
2. **Sample validation** of 1000 random rows
3. **FK integrity** - all foreign keys have valid targets
4. **Embedding quality** - no zero vectors

### Manual Validation

```bash
./scripts/validate_migration.sh postgresql \
  "postgresql://user:password@localhost/mydb" \
  http://localhost:8080 \
  my_memory_space
```

## Troubleshooting

### Permission Denied

**Error**: `permission denied for table users`

**Solution**: Grant read access to migration user:

```sql
GRANT SELECT ON ALL TABLES IN SCHEMA public TO migration_user;
```

### Orphaned Foreign Keys

**Warning**: `Found 42 orphaned foreign keys`

**Cause**: Child rows reference non-existent parent rows (data integrity issue in source)

**Solution**: Clean up source database or manually create placeholder memories in Engram.

### Slow Text Column Extraction

**Issue**: Migration is slow due to large text columns

**Solution**: Limit text columns used for embeddings via `--text-columns` flag.

## Production Checklist

Before production migration:

- [ ] Back up PostgreSQL database
- [ ] Test on sample subset of tables
- [ ] Verify sufficient Engram storage
- [ ] Configure checkpoint file
- [ ] Test FK mapping with small dataset
- [ ] Schedule during maintenance window
- [ ] Monitor query performance during migration
- [ ] Plan for validation (10-15% of migration time)

## Advanced Usage

### Incremental Migration

For ongoing databases, migrate in phases:

1. **Phase 1**: Static reference tables
2. **Phase 2**: Transactional tables up to date X
3. **Phase 3**: Delta migration for new records

### Custom Embedding Strategy

Specify which columns to use for semantic embeddings:

```bash
migrate-postgresql \
  --source "postgresql://user:pass@localhost/mydb" \
  --target http://localhost:8080 \
  --text-columns "users:name,bio;products:title,description,tags"
```

## See Also

- [Neo4j Migration Guide](migration-neo4j.md)
- [Redis Migration Guide](migration-redis.md)
- [Schema Mapping Best Practices](../explanation/schema-mapping.md)
