# Task 007: Database Migration Tooling & Guides — architecture complete, integration pending

**Phase 1 Status**: COMPLETE (Architecture & Documentation)
**Phase 2 Status**: PENDING (Storage Layer Integration)
**Priority:** P1 (High)
**Estimated Effort:** Phase 1: 5-6 days (DONE) | Phase 2: 3-4 days (PENDING)
**Dependencies:** None

## Objective

Build production-grade migration tools for Neo4j, PostgreSQL, and Redis to Engram's cognitive memory graph. Enable users to migrate existing databases with <1% data loss, validated integrity, and efficient streaming for large datasets (>1M nodes). Migration tools must map traditional database concepts to Engram's probabilistic memory model while preserving semantic relationships and generating high-quality embeddings.

## Key Deliverables

### CLI Migration Tools
- `/tools/migrate-neo4j/src/main.rs` - Neo4j migration CLI with streaming
- `/tools/migrate-neo4j/src/graph_mapper.rs` - Graph-to-memory transformation logic
- `/tools/migrate-neo4j/src/cypher_exporter.rs` - Cypher query builder for batched export
- `/tools/migrate-postgresql/src/main.rs` - PostgreSQL migration CLI
- `/tools/migrate-postgresql/src/schema_analyzer.rs` - Schema introspection and FK graph
- `/tools/migrate-postgresql/src/relational_mapper.rs` - Row-to-memory transformation
- `/tools/migrate-redis/src/main.rs` - Redis migration CLI
- `/tools/migrate-redis/src/rdb_parser.rs` - RDB/AOF streaming parser
- `/tools/migrate-redis/src/ttl_mapper.rs` - TTL to decay rate conversion

### Validation Infrastructure
- `/tools/migration-common/src/validator.rs` - Shared validation primitives
- `/tools/migration-common/src/streaming.rs` - Batching and backpressure handling
- `/tools/migration-common/src/embedding_generator.rs` - Embedding generation pipeline
- `/scripts/validate_migration.sh` - Automated validation orchestration

### Documentation
- `/docs/operations/migration-neo4j.md` - Neo4j migration guide with performance tuning
- `/docs/operations/migration-postgresql.md` - PostgreSQL migration guide with schema mapping
- `/docs/operations/migration-redis.md` - Redis migration guide with TTL mapping
- `/docs/tutorials/migrate-from-neo4j.md` - Step-by-step tutorial with example graph

## Technical Specifications

### 1. Neo4j Migration Architecture

**Data Model Mapping:**
```rust
// Neo4j Node → Engram Memory
struct NodeMapper {
    // Map node labels to memory space IDs
    label_to_space: HashMap<String, MemorySpaceId>,
    // Embedding generator for node properties
    embedding_generator: Arc<EmbeddingGenerator>,
}

// Node → Memory transformation:
// - Node ID → Memory.id (prefixed with "neo4j_node_")
// - Node labels → Memory space selection (primary label determines space)
// - Node properties → JSON content, generate embedding from text fields
// - Creation timestamp → Memory.created_at (use Neo4j audit props if available)
// - Default activation = 0.5, confidence = MEDIUM (user-configurable)
```

**Relationship Mapping:**
```rust
// Neo4j Relationship → Engram Edge (Memory → Memory)
// - Relationship type → Edge metadata (stored in edge properties)
// - Relationship properties → Edge weight calculation (numeric props → normalized weight)
// - Default edge weight = 0.7, decays based on relationship age
// - Bidirectional relationships create two edges (A→B and B→A)
```

**Batching Strategy:**
```cypher
// Cypher query for streaming node export (batch size 10k):
CALL apoc.periodic.iterate(
  "MATCH (n) RETURN n",
  "WITH n CALL custom.export.node(n) YIELD value RETURN value",
  {batchSize: 10000, parallel: false}
)

// For relationships:
CALL apoc.periodic.iterate(
  "MATCH ()-[r]->() RETURN r",
  "WITH r CALL custom.export.relationship(r) YIELD value RETURN value",
  {batchSize: 10000, parallel: false}
)
```

**Embedding Generation:**
- Extract text properties (description, name, content) and concatenate
- Use `EmbeddingProvider` to generate 768-dim embedding with provenance tracking
- Cache embeddings by content hash to avoid duplicate generation
- Batch embedding requests (100 items) to amortize model initialization cost
- Fall back to zero vector with LOW confidence if no text content available

**Performance Characteristics:**
- Throughput: 10k-50k nodes/sec (depending on property complexity)
- Memory usage: <2GB for streaming pipeline (excludes embedding model)
- Parallelism: Single-threaded extraction, parallel embedding generation
- Progress tracking: Log progress every 10k nodes with ETA calculation

### 2. PostgreSQL Migration Architecture

**Schema Analysis:**
```rust
// Introspect PostgreSQL schema to build FK graph
struct SchemaAnalyzer {
    tables: Vec<TableMetadata>,
    foreign_keys: Vec<ForeignKey>,
    // Topologically sorted tables for dependency-respecting migration
    migration_order: Vec<String>,
}

// TableMetadata captures columns, types, indexes
// Used to determine:
// - Which columns contain text (for embedding generation)
// - Which columns represent timestamps (for Memory.created_at)
// - Which columns are primary keys (for Memory.id construction)
```

**Data Model Mapping:**
```rust
// Table → Memory Space
// - One memory space per table (space_id = table name)
// - Alternative: User can provide mapping config (e.g., customers+orders → "crm")

// Row → Memory
// - Primary key columns → Memory.id (e.g., "users_123")
// - Text columns (VARCHAR, TEXT) → concatenated for embedding
// - Timestamp columns → Memory.created_at (use created_at/updated_at if present)
// - JSON columns → parsed and merged into content
// - All columns → JSON-serialized Memory.content for preservation

// Foreign Key → Edge
// - FK relationship → directed edge (child → parent)
// - Edge weight = 0.8 (high confidence for referential integrity)
// - Multi-column FKs create single edge with composite metadata
```

**Extraction Strategy:**
```sql
-- Stream rows with cursor to avoid loading entire table
DECLARE row_cursor CURSOR FOR
  SELECT * FROM table_name ORDER BY primary_key;

-- Fetch in batches of 5000 rows
FETCH FORWARD 5000 FROM row_cursor;

-- For large tables (>10M rows), use parallel export:
-- Partition by primary key ranges and spawn multiple workers
```

**Embedding Generation:**
- Concatenate all text columns (VARCHAR, TEXT, CHAR) with field names as prefixes
- Example: "name: John Doe | bio: Software engineer | notes: Interested in AI"
- Generate embedding using `EmbeddingProvider` with provenance (PostgreSQL source)
- Handle NULL values gracefully (skip field in concatenation)
- For tables without text columns, generate embedding from JSON-serialized row

**Referential Integrity Preservation:**
- Migrate tables in topological order (parents before children)
- Build FK edge graph after all memories inserted
- Validate all FK relationships exist in Engram before committing edges
- Report orphaned FKs (child rows with no parent) for manual resolution

**Performance Characteristics:**
- Throughput: 5k-20k rows/sec (depends on table width and text volume)
- Memory usage: <1GB for streaming pipeline
- Parallelism: Parallel table extraction (one worker per table)
- Batching: Insert memories in batches of 1000 via `BatchEngine`

### 3. Redis Migration Architecture

**Data Model Mapping:**
```rust
// Redis Key → Memory
// - Key name → Memory.id (prefixed with "redis_")
// - Value → Memory.content (JSON-encoded with type metadata)
// - TTL → Memory.decay_rate (convert TTL seconds to decay rate)
// - Key type (string/hash/list/set/zset) → metadata in content

// TTL to Decay Rate Conversion:
// decay_rate = 1.0 / (ttl_seconds / 3600.0)
// Example: TTL=3600s (1 hour) → decay_rate=1.0 (decay after 1 hour)
// No TTL → decay_rate=0.01 (slow background decay)
```

**Value Type Handling:**
```rust
enum RedisValue {
    String(String) => {
        // Direct string value
        // Embedding from string content
        // Confidence: HIGH (simple type)
    },
    Hash(HashMap<String, String>) => {
        // Serialize as JSON object
        // Embedding from concatenated field values
        // Each field could be separate memory (configurable)
    },
    List(Vec<String>) => {
        // Serialize as JSON array
        // Embedding from concatenated elements
        // Preserve order in metadata
    },
    Set(HashSet<String>) => {
        // Serialize as JSON array
        // Embedding from concatenated members
        // Mark as unordered in metadata
    },
    ZSet(Vec<(String, f64)>) => {
        // Sorted set members with scores
        // Scores → initial activation levels
        // Create edges between consecutive members
        // Embedding from member content
    },
}
```

**Extraction Strategy:**
```bash
# Use redis-cli with --rdb or --pipe for efficient export
redis-cli --rdb dump.rdb

# Parse RDB file with streaming parser (avoid loading entire dump)
# Alternative: Use SCAN command for live migration (slower but no downtime)
SCAN 0 COUNT 1000
```

**Sorted Set Optimization:**
```rust
// ZSet scores represent importance/ranking
// Map to activation levels: normalize scores to [0, 1]
// Create edges between adjacent members (rank-based relationships)
// Example: leaderboard entries have edges to neighbors
```

**Performance Characteristics:**
- Throughput: 10k-100k keys/sec (RDB parsing) or 1k-5k keys/sec (SCAN)
- Memory usage: <500MB for streaming RDB parser
- Parallelism: Single-threaded RDB parsing, parallel embedding generation
- Batching: Insert in batches of 5000 keys

### 4. Shared Migration Infrastructure

**Streaming and Batching:**
```rust
// Unified streaming pipeline for all migration sources
struct MigrationPipeline {
    source: Box<dyn DataSource>,           // Neo4j/PostgreSQL/Redis
    transformer: Box<dyn MemoryTransformer>, // Source-specific mapper
    embedder: Arc<EmbeddingGenerator>,     // Shared embedding generation
    validator: Arc<Validator>,             // Integrity checks
    batch_size: usize,                     // Configurable (default 1000)
}

// DataSource trait for source-agnostic extraction
trait DataSource {
    fn next_batch(&mut self) -> Result<Vec<SourceRecord>, MigrationError>;
    fn total_records(&self) -> Option<u64>; // For progress tracking
    fn checkpoint(&self) -> Result<Checkpoint, MigrationError>; // Resume support
}

// Backpressure handling
impl MigrationPipeline {
    async fn run(&mut self) -> Result<MigrationReport, MigrationError> {
        let semaphore = Arc::new(Semaphore::new(10)); // Max 10 in-flight batches

        loop {
            let batch = self.source.next_batch()?;
            if batch.is_empty() { break; }

            let permit = semaphore.acquire().await?;
            tokio::spawn(async move {
                // Process batch: transform → embed → store
                let _permit = permit; // Hold permit until complete
            });
        }
    }
}
```

**Embedding Generation Pipeline:**
```rust
// High-throughput embedding generation with caching
struct EmbeddingGenerator {
    provider: Arc<dyn EmbeddingProvider>,
    cache: Arc<DashMap<ContentHash, [f32; 768]>>,
    batch_size: usize, // Batch requests to provider (default 100)
}

impl EmbeddingGenerator {
    async fn generate_batch(&self, texts: Vec<String>)
        -> Result<Vec<[f32; 768]>, EmbeddingError>
    {
        // Check cache first
        let mut uncached = Vec::new();
        let mut results = vec![None; texts.len()];

        for (i, text) in texts.iter().enumerate() {
            let hash = ContentHash::from(text);
            if let Some(embedding) = self.cache.get(&hash) {
                results[i] = Some(*embedding);
            } else {
                uncached.push((i, text.clone()));
            }
        }

        // Generate embeddings for uncached texts
        if !uncached.is_empty() {
            let fresh_embeddings = self.provider
                .embed_batch(uncached.iter().map(|(_, t)| t.as_str()).collect())
                .await?;

            for ((i, text), emb) in uncached.iter().zip(fresh_embeddings) {
                self.cache.insert(ContentHash::from(text), emb);
                results[*i] = Some(emb);
            }
        }

        Ok(results.into_iter().map(|r| r.unwrap()).collect())
    }
}
```

**Validation Procedures:**
```rust
struct MigrationValidator {
    source_stats: SourceStatistics,
    target_handle: Arc<SpaceHandle>,
}

impl MigrationValidator {
    // 1. Count validation
    async fn validate_counts(&self) -> Result<CountReport, ValidationError> {
        let source_count = self.source_stats.total_records;
        let target_count = self.target_handle.store().memory_count().await?;

        let loss_rate = (source_count - target_count) as f64 / source_count as f64;
        ensure!(loss_rate < 0.01, "Data loss {:.2}% exceeds 1% threshold", loss_rate * 100.0);

        Ok(CountReport { source_count, target_count, loss_rate })
    }

    // 2. Sample validation (random 1000 records)
    async fn validate_samples(&self, sample_size: usize)
        -> Result<SampleReport, ValidationError>
    {
        let mut rng = rand::thread_rng();
        let samples: Vec<_> = self.source_stats.record_ids
            .choose_multiple(&mut rng, sample_size)
            .collect();

        let mut matches = 0;
        for source_id in samples {
            let source_record = self.fetch_source_record(source_id)?;
            let target_memory = self.target_handle.store()
                .get_memory(&format!("migrated_{}", source_id))
                .await?;

            if self.records_match(&source_record, &target_memory) {
                matches += 1;
            }
        }

        let match_rate = matches as f64 / sample_size as f64;
        ensure!(match_rate > 0.99, "Sample match rate {:.2}% below 99%", match_rate * 100.0);

        Ok(SampleReport { sample_size, matches, match_rate })
    }

    // 3. Relationship integrity
    async fn validate_edges(&self) -> Result<EdgeReport, ValidationError> {
        // Verify all edges have valid source and target memories
        let edges = self.target_handle.store().all_edges().await?;

        let mut orphaned = Vec::new();
        for edge in edges {
            if !self.target_handle.store().exists(&edge.source).await? ||
               !self.target_handle.store().exists(&edge.target).await? {
                orphaned.push(edge);
            }
        }

        ensure!(orphaned.is_empty(), "Found {} orphaned edges", orphaned.len());

        Ok(EdgeReport { total_edges: edges.len(), orphaned_count: orphaned.len() })
    }

    // 4. Semantic similarity validation (spot check embeddings)
    async fn validate_embeddings(&self, sample_size: usize)
        -> Result<EmbeddingReport, ValidationError>
    {
        // Verify embedding quality by checking semantic coherence
        let samples = self.sample_memories(sample_size).await?;

        let mut quality_scores = Vec::new();
        for memory in samples {
            // Check embedding is not zero vector
            let is_zero = memory.embedding.iter().all(|&x| x.abs() < 1e-6);
            ensure!(!is_zero, "Found zero embedding for memory {}", memory.id);

            // Check embedding is normalized (unit length)
            let norm: f32 = memory.embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
            let quality = (norm - 1.0).abs(); // Should be close to 1.0
            quality_scores.push(quality);
        }

        let avg_quality = quality_scores.iter().sum::<f32>() / sample_size as f32;
        ensure!(avg_quality < 0.1, "Average embedding quality {:.4} exceeds threshold", avg_quality);

        Ok(EmbeddingReport { sample_size, avg_quality })
    }
}
```

**Checkpointing for Resume:**
```rust
// Support resumable migration for long-running transfers
struct MigrationCheckpoint {
    last_processed_id: String,
    records_migrated: u64,
    timestamp: DateTime<Utc>,
    checkpoint_file: PathBuf,
}

impl MigrationCheckpoint {
    fn save(&self) -> Result<(), std::io::Error> {
        let json = serde_json::to_string_pretty(self)?;
        std::fs::write(&self.checkpoint_file, json)?;
        Ok(())
    }

    fn load(path: &Path) -> Result<Self, std::io::Error> {
        let json = std::fs::read_to_string(path)?;
        Ok(serde_json::from_str(&json)?)
    }
}
```

**Progress Reporting:**
```rust
// Real-time progress with ETA
struct ProgressTracker {
    total_records: u64,
    processed_records: AtomicU64,
    start_time: Instant,
}

impl ProgressTracker {
    fn report(&self) {
        let processed = self.processed_records.load(Ordering::Relaxed);
        let elapsed = self.start_time.elapsed();
        let rate = processed as f64 / elapsed.as_secs_f64();
        let remaining = self.total_records - processed;
        let eta = Duration::from_secs((remaining as f64 / rate) as u64);

        tracing::info!(
            processed = processed,
            total = self.total_records,
            rate = format!("{:.0} records/sec", rate),
            eta = format!("{:?}", eta),
            "Migration progress"
        );
    }
}
```

### 5. CLI Interface Design

**Common Arguments:**
```bash
# All migration CLIs share consistent flags
--source <CONNECTION_STRING>    # Source database connection
--target <ENGRAM_URL>           # Target Engram instance
--memory-space <SPACE_ID>       # Target memory space (or auto-create)
--batch-size <N>                # Records per batch (default: 1000)
--parallel-workers <N>          # Parallel extraction workers (default: 4)
--checkpoint-file <PATH>        # Resume from checkpoint
--dry-run                       # Validate without writing
--embedding-provider <PROVIDER> # OpenAI/HuggingFace/Local (default: auto-detect)
--confidence <0.0-1.0>          # Initial confidence for migrated memories
--validate                      # Run validation after migration
--skip-edges                    # Migrate nodes only, skip relationships
```

**Neo4j Example:**
```bash
migrate-neo4j \
  --source bolt://localhost:7687 \
  --source-user neo4j \
  --source-password secret \
  --target http://localhost:8080 \
  --memory-space-prefix "neo4j" \
  --label-to-space "Person:people,Company:companies" \
  --batch-size 10000 \
  --checkpoint-file /tmp/neo4j_migration.json \
  --validate
```

**PostgreSQL Example:**
```bash
migrate-postgresql \
  --source "postgresql://user:pass@localhost/mydb" \
  --target http://localhost:8080 \
  --table-to-space "users:user_space,orders:order_space" \
  --text-columns "users:name,bio;orders:notes" \
  --timestamp-column created_at \
  --batch-size 5000 \
  --parallel-workers 8 \
  --validate
```

**Redis Example:**
```bash
migrate-redis \
  --source redis://localhost:6379 \
  --source-db 0 \
  --target http://localhost:8080 \
  --memory-space "redis_cache" \
  --use-rdb /var/lib/redis/dump.rdb \
  --ttl-as-decay \
  --batch-size 10000 \
  --validate
```

### 6. Validation Script

**`/scripts/validate_migration.sh`:**
```bash
#!/bin/bash
# Automated migration validation orchestration

set -euo pipefail

SOURCE_TYPE="$1"  # neo4j|postgresql|redis
SOURCE_CONN="$2"
ENGRAM_URL="$3"
MEMORY_SPACE="$4"

echo "=== Migration Validation for $SOURCE_TYPE ==="

# 1. Count validation
echo "Step 1: Validating record counts..."
SOURCE_COUNT=$(fetch_source_count "$SOURCE_TYPE" "$SOURCE_CONN")
TARGET_COUNT=$(curl -s "$ENGRAM_URL/api/v1/memory_spaces/$MEMORY_SPACE/count" | jq -r '.count')

if [ "$SOURCE_COUNT" -ne "$TARGET_COUNT" ]; then
    echo "ERROR: Count mismatch (source: $SOURCE_COUNT, target: $TARGET_COUNT)"
    exit 1
fi
echo "✓ Counts match: $SOURCE_COUNT records"

# 2. Sample validation
echo "Step 2: Validating random samples..."
SAMPLE_SIZE=1000
./tools/validate_samples.sh "$SOURCE_TYPE" "$SOURCE_CONN" "$ENGRAM_URL" "$MEMORY_SPACE" "$SAMPLE_SIZE"
echo "✓ Sample validation passed"

# 3. Edge integrity
echo "Step 3: Validating edge integrity..."
ORPHANED=$(curl -s "$ENGRAM_URL/api/v1/memory_spaces/$MEMORY_SPACE/validate_edges" | jq -r '.orphaned_count')
if [ "$ORPHANED" -gt 0 ]; then
    echo "ERROR: Found $ORPHANED orphaned edges"
    exit 1
fi
echo "✓ Edge integrity validated"

# 4. Embedding quality
echo "Step 4: Validating embedding quality..."
ZERO_EMBEDDINGS=$(curl -s "$ENGRAM_URL/api/v1/memory_spaces/$MEMORY_SPACE/validate_embeddings" | jq -r '.zero_count')
if [ "$ZERO_EMBEDDINGS" -gt 0 ]; then
    echo "WARNING: Found $ZERO_EMBEDDINGS zero embeddings"
fi
echo "✓ Embedding quality checked"

# 5. Performance comparison
echo "Step 5: Comparing query performance..."
./tools/benchmark_queries.sh "$SOURCE_TYPE" "$SOURCE_CONN" "$ENGRAM_URL" "$MEMORY_SPACE"
echo "✓ Performance benchmark complete"

echo "=== Migration Validation Complete ==="
```

## Acceptance Criteria

- [ ] Neo4j migration handles graphs with >1M nodes in <30 minutes (single machine)
- [ ] PostgreSQL migration preserves all foreign key relationships (100% integrity)
- [ ] Redis migration maintains TTL semantics via decay rates (<5% error)
- [ ] All migrations complete with <1% data loss (validated via count + sample checks)
- [ ] Migration validation script catches 100% of data integrity issues in test suite
- [ ] Embedding generation uses content-based caching (>80% cache hit rate on duplicate content)
- [ ] Resumable migration from checkpoint after failure (<10 seconds overhead)
- [ ] Migration guides tested by external user unfamiliar with Engram (success without support)
- [ ] Performance benchmarks show comparable query latency (within 2x of source database)
- [ ] Memory usage stays below 2GB for streaming pipelines (tested with 10M record dataset)

## Testing Strategy

1. **Unit Tests**: Test individual mappers (NodeMapper, TableMapper, KeyMapper) with fixtures
2. **Integration Tests**: End-to-end migration with small test databases (1k-10k records)
3. **Load Tests**: Large-scale migration with 1M+ records, measure throughput and memory
4. **Chaos Tests**: Simulate failures (network interruption, Engram restart) and verify resume
5. **Validation Tests**: Deliberately introduce errors (orphaned edges, zero embeddings) and verify detection

## Documentation Strategy

Each migration guide follows this structure:
1. **Overview**: High-level mapping explanation (source concepts → Engram concepts)
2. **Prerequisites**: Required source database version, Engram version, system resources
3. **Quick Start**: 5-minute example with sample database
4. **Configuration**: Detailed flag explanations with performance implications
5. **Schema Mapping**: How to customize table/label/key → memory space mapping
6. **Embedding Generation**: Options for embedding providers, caching strategies
7. **Performance Tuning**: Batch size, parallel workers, memory limits
8. **Validation**: How to verify migration success
9. **Troubleshooting**: Common errors and resolutions
10. **Production Checklist**: Pre-flight checks for large migrations

## Integration Points

- **Storage Layer**: Uses `MemorySpaceRegistry` and `SpaceHandle` for multi-tenant isolation
- **Embedding System**: Leverages `EmbeddingProvider` with provenance tracking
- **Batch Operations**: Uses `BatchEngine` for high-throughput inserts
- **Monitoring**: Emits migration metrics via `engram-metrics` package
- **Error Handling**: Follows Engram error conventions (context + suggestion + example)

## Follow-Up Tasks

- Future: Incremental migration support (continuous sync during migration via CDC)
- Future: Automated rollback on migration failure (snapshot-based restore)
- Future: Conflict resolution for duplicate keys (merge strategies)
- Future: Bidirectional sync (Engram → Neo4j/PostgreSQL/Redis)
- Future: Schema evolution detection (alert on source schema changes post-migration)
