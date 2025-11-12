# Synthetic Data Generator for Engram

Production-quality tool for generating large-scale synthetic datasets (100K-1M nodes) for Engram performance testing. Generates realistic data by clustering synthetic embeddings around real Wikipedia centroids.

## Workflow

### Phase 1: Ingest Real Wikipedia Data (Foundation)
```bash
# Ingest 50K real Wikipedia articles (~12 hours)
cd tools/wikipedia_ingest
source venv/bin/activate
python ingest.py --endpoint http://localhost:7432 --limit 50000
```

### Phase 2: Extract Centroids from Real Data
```bash
# Extract 100 centroids from 50K Wikipedia embeddings
cd tools/synthetic_data_generator
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

python extract_centroids.py \
  --endpoint http://localhost:7432 \
  --num-centroids 100 \
  --output centroids.npy
```

This will create:
- `centroids.npy` - 100 cluster centroids (100, 768) array
- `centroids.json` - Metadata (num centroids, source count, etc.)

### Phase 3: Generate Synthetic Data
```bash
# Generate 950K synthetic memories clustered around centroids
python generate_synthetic.py \
  --centroids centroids.npy \
  --count 950000 \
  --std-dev 0.1 \
  --output synthetic_memories.jsonl

# Expected output size: ~3-4 GB JSONL file
```

Parameters:
- `--count`: Number of synthetic memories (default: 950,000)
- `--std-dev`: Gaussian noise std dev (lower = tighter clusters)
- `--confidence`: Base confidence score (default: 0.7)
- `--seed`: Random seed for reproducibility

### Phase 4: Bulk Ingest Synthetic Data
```bash
# Ingest 950K synthetic memories (~30-60 min at 300-500 mem/s)
python bulk_ingest.py \
  --input synthetic_memories.jsonl \
  --endpoint http://localhost:7432 \
  --concurrent 20 \
  --batch-size 1000

# Monitor progress with:
# - Real-time throughput (mem/s)
# - Error count
# - ETA
```

Parameters:
- `--concurrent`: Parallel requests (default: 20)
- `--batch-size`: Memories per batch (default: 1000)
- `--limit`: Limit ingestion for testing

## Result

**Total Dataset:**
- 50K real Wikipedia articles (semantic foundation)
- 950K synthetic memories (realistic clusters)
- 1M total nodes for production-scale testing

**Performance Characteristics:**
- Realistic semantic structure (not random)
- Enables pattern completion testing (clustered data)
- Representative of production workloads
- Reproducible (deterministic seeds)

## Performance Estimates

| Phase | Time | Throughput |
|-------|------|------------|
| 50K Wikipedia ingestion | ~12 hours | 1.1 art/s |
| Centroid extraction | ~2 min | N/A |
| Synthetic generation | ~5 min | N/A |
| Bulk ingestion | ~30-60 min | 300-500 mem/s |
| **Total** | **~13 hours** | |

## Testing with Synthetic Data

After ingestion, run M17 baseline test with 1M nodes:

```bash
# Update scenario to use 1M nodes
# Edit scenarios/m17_baseline.toml: num_nodes = 1000000

# Run performance test
./scripts/m17_performance_check.sh 001 before

# Compare vs 1K baseline
./scripts/compare_m17_performance.sh 001
```

Expected results:
- P99 latency will increase (data doesn't fit in cache)
- Throughput may decrease (index traversal depth)
- Error rate should remain 0%

This validates Engram's performance at production scale.

## Troubleshooting

### Centroid extraction fails with "No memories found"
- Verify Wikipedia ingestion completed successfully
- Check Engram server is running: `./target/release/engram status`
- Test API: `curl http://localhost:7432/api/v1/memories`

### Bulk ingestion slow (<100 mem/s)
- Increase `--concurrent` (try 50-100)
- Check CPU/disk I/O utilization
- Verify database isn't degraded (restart server)

### Memory errors during generation
- Reduce `--count` and generate in multiple batches
- Synthetic generation uses ~4GB RAM for 950K memories

## File Sizes

| File | Size | Description |
|------|------|-------------|
| centroids.npy | ~300 KB | 100 centroids × 768 dim × 4 bytes |
| synthetic_memories.jsonl | ~3-4 GB | 950K memories with embeddings |

## Design Rationale

**Why hybrid (real + synthetic)?**
- Real data provides semantic foundation
- Synthetic data provides scale
- Combined dataset balances realism and practicality

**Why cluster around centroids?**
- Preserves semantic structure from Wikipedia
- Enables pattern completion testing
- Realistic similarity distributions

**Why not all synthetic?**
- Random embeddings don't test realistic queries
- No pattern completion (uncorrelated data)
- Doesn't validate cognitive operations
