# Example 10: Database Migration

**Learning Goal**: Migrate data from traditional databases (PostgreSQL, Neo4j, Redis) to Engram.

**Difficulty**: Advanced
**Time**: 30 minutes
**Prerequisites**: Completed Examples 01, 09

## Cognitive Concept

Migrating to Engram requires translating:
- **SQL rows** → Semantic memories with embeddings
- **Graph edges** → Association types with confidence
- **Cache entries** → Episodic memories with decay
- **Deterministic queries** → Probabilistic recalls

Key insight: Don't just copy data structure, transform to cognitive model.

## What You'll Learn

- Extract data from source databases
- Generate embeddings for text content
- Map relationships to association types
- Set appropriate confidence scores
- Validate migration completeness

## Migration Strategies

### Strategy 1: Snapshot + Backfill
1. Take snapshot of source database
2. Bulk import to Engram
3. Backfill incremental changes
4. Cutover when synchronized

### Strategy 2: Dual-Write
1. Write to both databases during transition
2. Read from Engram with fallback to source
3. Gradually migrate read traffic
4. Decommission source when validated

### Strategy 3: Staged Migration
1. Migrate non-critical data first
2. Validate accuracy and performance
3. Migrate critical data in maintenance window
4. Monitor and rollback if needed

## Code Examples

See language-specific implementations in this directory:

- `postgres_migration.py` - PostgreSQL to Engram
- `neo4j_migration.py` - Neo4j graph to Engram
- `redis_migration.py` - Redis cache to Engram
- `mongodb_migration.py` - MongoDB documents to Engram
- `elasticsearch_migration.py` - Elasticsearch to Engram

## PostgreSQL Migration Example

```python
# Extract from PostgreSQL
conn = psycopg2.connect("postgresql://localhost/mydb")
cursor = conn.cursor()
cursor.execute("SELECT id, content, created_at, metadata FROM documents")

# Transform to Engram memories
for row in cursor:
    id, content, created_at, metadata = row

    # Generate embedding
    embedding = embedding_model.encode(content)

    # Create memory
    memory = Memory(
        content=content,
        embedding=embedding,
        confidence=Confidence(
            value=0.95,  # High confidence for verified DB data
            reasoning="Migrated from authoritative PostgreSQL database"
        ),
        metadata={
            "source": "postgres",
            "source_id": id,
            "migrated_at": datetime.now().isoformat(),
            **metadata
        }
    )

    engram_client.remember(memory)
```

## Neo4j Migration Example

```python
# Extract graph from Neo4j
with driver.session() as session:
    result = session.run("""
        MATCH (n)-[r]->(m)
        RETURN n.id, n.content, type(r), r.weight, m.id, m.content
    """)

    # Transform to Engram associations
    for record in result:
        # Remember both nodes
        node1 = engram_client.remember(Memory(content=record['n.content'], ...))
        node2 = engram_client.remember(Memory(content=record['m.content'], ...))

        # Create association
        engram_client.associate(
            from_memory_id=node1.memory_id,
            to_memory_id=node2.memory_id,
            association_type=record['type(r)'].lower(),
            strength=record['r.weight'] or 0.5
        )
```

## Embedding Generation

Choose embedding model based on content type:

| Content Type | Recommended Model | Dimensions |
|--------------|------------------|------------|
| General text | `all-mpnet-base-v2` | 768 |
| Code | `microsoft/codebert-base` | 768 |
| Scientific | `allenai/scibert` | 768 |
| Multilingual | `paraphrase-multilingual-mpnet-base-v2` | 768 |

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-mpnet-base-v2')
embeddings = model.encode(texts, batch_size=32)
```

## Validation Checklist

- [ ] Row/node count matches (source vs Engram)
- [ ] Sample queries return expected results
- [ ] Performance meets requirements
- [ ] Relationships preserved (for graph migrations)
- [ ] Metadata transferred correctly
- [ ] Confidence scores appropriate
- [ ] Embeddings semantically meaningful

## Common Pitfalls

1. **Forgetting embeddings**: Every memory needs an embedding vector
2. **Wrong embedding dimension**: Must match server config (default: 768)
3. **Too-high confidence**: Migrated data isn't always certain (use 0.7-0.9)
4. **Losing relationships**: Track source relationships and recreate
5. **Skipping validation**: Always validate sample queries after migration

## Performance Tips

- Use batch operations (100-500 per batch)
- Pre-compute embeddings before import
- Use streaming for large datasets (>100K records)
- Monitor hot tier pressure during import
- Consider warm-starting to cold tier for large corpora

## Rollback Plan

1. Keep source database running during validation
2. Tag all migrated memories with `migration_id`
3. Can delete by tag if migration fails
4. Maintain source-to-engram ID mapping

## Next Steps

- [09-batch-operations](../09-batch-operations/) - Optimize bulk import
- [Operations Guide](/operations/) - Production deployment
- [Monitoring](/operations/monitoring.md) - Track migration progress
