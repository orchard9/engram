# Database Migration Guide

Comprehensive guide for migrating from traditional databases to Engram.

## Overview

This guide covers migration strategies from:
- PostgreSQL and relational databases
- Neo4j and graph databases
- Redis and key-value stores
- MongoDB and document databases
- Elasticsearch and search engines

## Quick Links

For detailed code examples and working implementations, see:
- [Migration Examples](/reference/api-examples/10-migration-examples/) - Complete migration code

## Migration Strategies

### Snapshot + Backfill
Best for: Static or slowly-changing datasets

1. Take snapshot of source database
2. Bulk import to Engram
3. Backfill incremental changes
4. Cutover when synchronized

### Dual-Write
Best for: Critical systems requiring zero downtime

1. Write to both databases during transition
2. Read from Engram with fallback to source
3. Gradually migrate read traffic
4. Decommission source when validated

### Staged Migration
Best for: Large datasets with mixed criticality

1. Migrate non-critical data first
2. Validate accuracy and performance
3. Migrate critical data in maintenance window
4. Monitor and rollback if needed

## Key Considerations

### Embedding Generation
Every memory in Engram requires a semantic embedding. Choose appropriate models:

- **General text**: `all-mpnet-base-v2` (768-dim)
- **Code**: `microsoft/codebert-base` (768-dim)
- **Scientific**: `allenai/scibert` (768-dim)
- **Multilingual**: `paraphrase-multilingual-mpnet-base-v2` (768-dim)

### Confidence Scores
Map source database certainty to Engram confidence:

- **Primary keys/verified data**: 0.9-0.95
- **Indexed columns**: 0.8-0.9
- **Unverified user content**: 0.6-0.8
- **Derived/computed values**: 0.5-0.7

### Relationship Mapping
Transform relationships to Engram associations:

| Source | Engram Equivalent |
|--------|------------------|
| Foreign keys | Associations with type "references" |
| Graph edges | Typed associations with strength |
| Document references | Associations with confidence |
| Cache dependencies | Episodic links with decay |

## Performance Optimization

- Use batch operations (100-500 records per batch)
- Pre-compute embeddings before import
- Use gRPC streaming for large datasets
- Monitor hot tier pressure during migration
- Consider warm-starting to cold tier

## Validation Checklist

- [ ] Record count matches source
- [ ] Sample queries return expected results
- [ ] Relationships preserved correctly
- [ ] Metadata transferred complete
- [ ] Performance meets requirements
- [ ] Rollback plan tested

## See Also

- [Batch Operations Example](/reference/api-examples/09-batch-operations/) - Bulk import patterns
- [Performance Tuning](/operations/performance-tuning.md) - Optimize throughput
- [Monitoring](/operations/monitoring.md) - Track migration progress
