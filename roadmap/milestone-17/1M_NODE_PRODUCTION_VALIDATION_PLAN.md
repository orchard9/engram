# 1M Node Production Validation Plan

**Objective**: Validate Engram performance at production scale (1M nodes) using hybrid real + synthetic data

**Strategy**: Option 4 - Hybrid approach balancing realism and practicality
- 50K real Wikipedia articles (semantic foundation)
- 950K synthetic memories (clustered around Wikipedia centroids)
- Total: 1M nodes with realistic semantic structure

## Current Status

### Phase 1: Real Data Foundation (IN PROGRESS)
**Status**: Wikipedia ingestion running (PID 39107, 56% CPU)
**Target**: 50,000 Wikipedia articles
**Duration**: ~12 hours at 1.1 articles/sec
**Started**: 2025-11-09 00:16 AM

```bash
# Monitor progress:
ps aux | grep "python ingest.py"

# Check memory count:
curl http://localhost:7432/api/v1/memories | python3 -c "import sys, json; print(len(json.load(sys.stdin).get('memories', [])))"
```

### Phase 2: Centroid Extraction (READY)
**Status**: Tools built and tested
**Duration**: ~2 minutes
**Dependencies**: Phase 1 completion (50K articles)

```bash
cd tools/synthetic_data_generator
source venv/bin/activate

python extract_centroids.py \
  --endpoint http://localhost:7432 \
  --num-centroids 100 \
  --output data/centroids.npy
```

**Output**:
- `data/centroids.npy` - 100 cluster centroids (100, 768) array
- `data/centroids.json` - Metadata (num centroids, source count, etc.)

### Phase 3: Synthetic Generation (READY)
**Status**: Tools built and tested
**Duration**: ~5 minutes
**Dependencies**: Phase 2 completion (centroids extracted)

```bash
cd tools/synthetic_data_generator
source venv/bin/activate

python generate_synthetic.py \
  --centroids data/centroids.npy \
  --count 950000 \
  --std-dev 0.1 \
  --output data/synthetic_memories.jsonl
```

**Output**:
- `data/synthetic_memories.jsonl` - 950K synthetic memories (~3-4 GB)
- Each memory has 768-dim embedding clustered around Wikipedia centroids

### Phase 4: Bulk Ingestion (READY)
**Status**: Tools built and tested
**Duration**: ~30-60 minutes at 300-500 mem/sec
**Dependencies**: Phase 3 completion (synthetic data generated)

```bash
cd tools/synthetic_data_generator
source venv/bin/activate

python bulk_ingest.py \
  --input data/synthetic_memories.jsonl \
  --endpoint http://localhost:7432 \
  --concurrent 20 \
  --batch-size 1000
```

**Expected throughput**: 300-500 memories/sec
**Progress tracking**: Real-time throughput, error count, ETA

### Phase 5: Performance Testing (READY)
**Status**: M17 baseline infrastructure exists
**Duration**: 60 seconds per test
**Dependencies**: Phase 4 completion (1M nodes ingested)

```bash
# Update scenario to use 1M nodes
# Edit scenarios/m17_baseline.toml: num_nodes = 1000000

# Run baseline test
./scripts/m17_performance_check.sh 001 1m_baseline

# Compare vs 1K baseline
./scripts/compare_m17_performance.sh 001
```

**Metrics to track**:
- P99 latency (baseline: 0.501ms)
- Throughput (baseline: 999.9 ops/sec)
- Error rate (baseline: 0.0%)

## Timeline

| Phase | Duration | Starts After | Completion |
|-------|----------|--------------|------------|
| 1. Wikipedia ingestion | ~12 hours | Immediate | +12 hours |
| 2. Centroid extraction | ~2 min | Phase 1 | +12h 2m |
| 3. Synthetic generation | ~5 min | Phase 2 | +12h 7m |
| 4. Bulk ingestion | ~30-60 min | Phase 3 | +13h 7m |
| 5. Performance testing | ~5 min | Phase 4 | +13h 12m |
| **Total** | **~13 hours** | - | - |

## Expected Performance Changes

### At 1M Nodes (vs 1K baseline)

**Hypothesis**: Performance will degrade as data exceeds cache size

**1K Baseline (cache-resident)**:
- Dataset size: ~20MB (fits in L3 cache)
- P99 latency: 0.501ms
- Throughput: 999.9 ops/sec

**1M Baseline (cache + DRAM)**:
- Dataset size: ~20GB (HNSW index + embeddings)
- P99 latency: **Expected 5-50ms** (10-100x slower)
- Throughput: **Expected 100-500 ops/sec** (2-10x slower)
- Error rate: **Expected 0%** (same)

**Why the degradation?**
1. **Cache misses**: Data doesn't fit in L3 cache (32MB typical)
2. **DRAM latency**: ~100ns vs L3 ~10ns = 10x slower
3. **Index depth**: HNSW traversal deeper at 1M nodes
4. **Memory bandwidth**: Saturation at high concurrency

**Success criteria** (vs vector DB benchmarks):
- P99 latency < 50ms ✓ (Qdrant: 38ms, Milvus: <2ms)
- Throughput > 100 ops/sec ✓ (Redis: 1,238 QPS)
- Error rate = 0% ✓

## Comparison to Industry Benchmarks

| Database | Dataset | P99 Latency | Throughput | Notes |
|----------|---------|-------------|------------|-------|
| **Engram (1K)** | 1K nodes | 0.501ms | 999 ops/s | Cache-resident |
| **Engram (1M)** | 1M nodes | 5-50ms (est) | 100-500 ops/s (est) | Hybrid data |
| Qdrant | 1M vectors | 38.71ms | ~1,238 QPS | Vector DB |
| Pinecone | 1M vectors | <2ms | High | Vector DB |
| Neo4j | Complex | 13-135ms | 112-531 QPS | Graph DB |
| Redis | 1B vectors | 200ms median | 66K inserts/s | In-memory |

**Engram's position**:
- Faster than graph databases (Neo4j)
- Competitive with vector databases (Qdrant, Pinecone)
- Slower than pure in-memory (Redis at 1B scale)

## Tools Created

### 1. Centroid Extraction Tool
**File**: `tools/synthetic_data_generator/extract_centroids.py`
- Fetches all embeddings from Engram
- Runs MiniBatch k-means clustering
- Saves centroids for synthetic generation

### 2. Synthetic Data Generator
**File**: `tools/synthetic_data_generator/generate_synthetic.py`
- Generates embeddings clustered around centroids
- Creates realistic content strings
- Outputs JSONL for bulk ingestion

### 3. Bulk Ingestor
**File**: `tools/synthetic_data_generator/bulk_ingest.py`
- Async parallel HTTP requests
- Progress tracking with throughput
- Error handling and retry logic

### 4. Documentation
**File**: `tools/synthetic_data_generator/README.md`
- Complete workflow guide
- Performance estimates
- Troubleshooting tips

## Next Steps (After Wikipedia Ingestion Completes)

1. **Extract centroids** from 50K Wikipedia embeddings
2. **Generate 950K synthetic memories** clustered around centroids
3. **Bulk ingest** synthetic data (~30-60 min)
4. **Run M17 baseline test** with 1M nodes
5. **Compare performance** vs 1K baseline
6. **Document results** in `PERFORMANCE_LOG.md`

## Monitoring Commands

```bash
# Check Wikipedia ingestion progress
ps aux | grep "python ingest.py"

# Check Engram memory count (after ingestion)
curl http://localhost:7432/api/v1/memories | python3 -c "import sys, json; print(len(json.load(sys.stdin).get('memories', [])))"

# Check server health
curl http://localhost:7432/health

# Monitor disk space (20GB needed)
df -h .

# Monitor memory usage
top -l 1 | grep Engram
```

## Troubleshooting

### Wikipedia ingestion failed
- Check server logs
- Verify virtual environment activated
- Restart from checkpoint if supported

### Centroid extraction fails
- Verify 50K+ memories exist in database
- Check API endpoint responds
- Ensure sufficient RAM (4GB+)

### Bulk ingestion slow
- Increase `--concurrent` to 50-100
- Check CPU/disk I/O
- Restart Engram server if degraded

### Performance test shows high error rate
- Verify all 1M nodes ingested successfully
- Check server is healthy
- Ensure loadtest scenario correctly configured

## Success Metrics

**Production validation successful if:**
1. ✅ 1M nodes ingested with 0% error rate
2. ✅ P99 latency < 50ms (competitive with vector DBs)
3. ✅ Throughput > 100 ops/sec (usable for production)
4. ✅ Error rate remains 0% under load

**Failure indicates need for**:
- HNSW parameter tuning (M, ef_construction)
- Index optimization (cache-friendly layout)
- Query optimization (batch operations)
- Hardware upgrade (more RAM, faster SSD)

## References

- M17 Baseline: `roadmap/milestone-17/PERFORMANCE_BASELINE.md`
- Comparison Script: `scripts/compare_m17_performance.sh`
- Wikipedia Ingestor: `tools/wikipedia_ingest/ingest.py`
- Synthetic Generator: `tools/synthetic_data_generator/`

---

**Status**: Phase 1 in progress (Wikipedia ingestion)
**Next action**: Wait for 50K articles, then extract centroids
**ETA to 1M nodes**: ~13 hours from now
