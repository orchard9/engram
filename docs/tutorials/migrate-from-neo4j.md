# Tutorial: Migrating from Neo4j to Engram

This tutorial walks through migrating a sample Neo4j graph database to Engram, demonstrating the complete workflow from preparation to validation.

## Sample Dataset

We'll migrate a small social network graph with the following structure:

```cypher
// Create sample data in Neo4j
CREATE (alice:Person {name: "Alice", bio: "Software engineer interested in AI"})
CREATE (bob:Person {name: "Bob", bio: "Data scientist and machine learning researcher"})
CREATE (acme:Company {name: "Acme Corp", description: "Technology consulting firm"})

CREATE (alice)-[:WORKS_AT {since: 2020}]->(acme)
CREATE (bob)-[:WORKS_AT {since: 2021}]->(acme)
CREATE (alice)-[:KNOWS {since: 2019}]->(bob)

RETURN *

```

This creates:

- 3 nodes (2 Person, 1 Company)

- 3 relationships (2 WORKS_AT, 1 KNOWS)

## Step 1: Prepare Environment

### Install Migration Tool

```bash
# Build the migration tool
cd tools/migrate-neo4j
cargo build --release

# The binary will be at: target/release/migrate-neo4j

```

### Verify Neo4j Connectivity

```bash
# Test connection
cypher-shell -a bolt://localhost:7687 -u neo4j -p yourpassword \
  "MATCH (n) RETURN count(n) as node_count"

```

### Start Engram

```bash
# Start Engram instance
engram server start --port 8080

```

## Step 2: Plan the Migration

### Decide on Memory Space Mapping

We have two approaches:

**Option A: Automatic (label-based)**

- `Person` nodes → `neo4j_person` memory space

- `Company` nodes → `neo4j_company` memory space

**Option B: Custom mapping**

- `Person` nodes → `people` memory space

- `Company` nodes → `organizations` memory space

For this tutorial, we'll use Option B with custom mapping.

### Determine Batch Size

For our small dataset (3 nodes), batch size doesn't matter much. For larger datasets:

- **<10k nodes**: Use batch size 1000

- **10k-1M nodes**: Use batch size 10000

- **>1M nodes**: Use batch size 50000

## Step 3: Dry Run Migration

Always test with a dry run first:

```bash
./target/release/migrate-neo4j \
  --source bolt://localhost:7687 \
  --source-user neo4j \
  --source-password yourpassword \
  --target http://localhost:8080 \
  --label-to-space "Person:people,Company:organizations" \
  --batch-size 100 \
  --dry-run

```

Expected output:

```
Starting Neo4j to Engram migration
Source: bolt://localhost:7687
Target: http://localhost:8080
Batch size: 100

Migration progress: 3 records processed (100%)
Average rate: 150 records/sec

Migration Complete!
==================
Total records: 3
Elapsed time: 20ms
Average rate: 150 records/sec
Errors: 0

```

## Step 4: Run Full Migration

Now run the actual migration with validation:

```bash
./target/release/migrate-neo4j \
  --source bolt://localhost:7687 \
  --source-user neo4j \
  --source-password yourpassword \
  --target http://localhost:8080 \
  --label-to-space "Person:people,Company:organizations" \
  --batch-size 100 \
  --checkpoint-file /tmp/neo4j_migration.json \
  --validate

```

## Step 5: Verify Migration

### Query Migrated Memories

```bash
# Using Engram CLI
engram query --space people --content "AI researcher"

# Expected: Should find Alice and Bob with high similarity

```

### Check Memory Count

```bash
# Count memories in each space
curl http://localhost:8080/api/v1/memory_spaces/people/count
# Should return: {"count": 2}

curl http://localhost:8080/api/v1/memory_spaces/organizations/count
# Should return: {"count": 1}

```

### Verify Relationships

```bash
# Check if edges were created
engram graph edges --source neo4j_node_alice

# Expected: Should show edge to neo4j_node_acme (WORKS_AT)
# and edge to neo4j_node_bob (KNOWS)

```

## Step 6: Run Validation Script

```bash
./scripts/validate_migration.sh neo4j \
  bolt://localhost:7687 \
  http://localhost:8080 \
  people

./scripts/validate_migration.sh neo4j \
  bolt://localhost:7687 \
  http://localhost:8080 \
  organizations

```

Expected output:

```
=== Migration Validation for neo4j ===

Step 1: Validating record counts...
  Counts match: 2 records

Step 2: Validating random samples...
  Sample validation passed

Step 3: Validating edge integrity...
  Edge integrity validated

Step 4: Validating embedding quality...
  Embedding quality checked

Step 5: Comparing query performance...
  Performance benchmark complete

=== Migration Validation Complete ===
All checks passed successfully!

```

## Step 7: Test Semantic Queries

Now test that semantic search works on migrated data:

```bash
# Search for AI-related people
engram query --space people \
  --query "artificial intelligence and machine learning" \
  --limit 5

# Expected results:
# 1. Bob (bio mentions "machine learning researcher")
# 2. Alice (bio mentions "AI")

```

## Troubleshooting

### Issue: "Failed to connect to Neo4j"

**Cause**: Neo4j not running or wrong credentials

**Solution**:

```bash
# Verify Neo4j is running
systemctl status neo4j

# Test credentials
cypher-shell -a bolt://localhost:7687 -u neo4j -p yourpassword "RETURN 1"

```

### Issue: "No embeddings generated"

**Cause**: Nodes have no text properties

**Solution**: Ensure nodes have at least one text property (name, description, bio, etc.)

### Issue: Migration is very slow

**Cause**: Batch size too small or network latency

**Solution**:

- Increase batch size to 10000

- Run migration tool on same machine as Neo4j

- Check network latency: `ping <neo4j-host>`

## Next Steps

### Incremental Updates

For ongoing Neo4j databases, set up incremental migration:

1. Use checkpoint file to track last migrated ID

2. Schedule periodic migrations for new nodes

3. Use change data capture (CDC) for real-time sync

### Schema Evolution

If your Neo4j schema changes:

1. Update `--label-to-space` mapping

2. Re-run migration for new label types

3. Consider data versioning in Engram

### Performance Optimization

For large graphs (>1M nodes):

1. Use parallel migration (run multiple instances with different label filters)

2. Increase batch size to 50000

3. Use local Neo4j replica to avoid production load

4. Consider partitioning by label type

## Summary

You've successfully:

- Migrated Neo4j nodes to Engram memories

- Preserved relationships as edges

- Generated semantic embeddings

- Validated migration integrity

- Tested semantic queries

Your Neo4j data is now available in Engram's cognitive memory graph!

## See Also

- [Neo4j Migration Guide](../operations/migration-neo4j.md)

- [Embedding Generation](../explanation/embeddings.md)

- [Memory Space Management](../howto/manage-memory-spaces.md)
