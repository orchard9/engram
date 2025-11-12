# Wikipedia Ingestion Tool

Production-quality tool for ingesting Wikipedia articles into Engram for realistic performance testing.

## Architecture

### Data Source
- **Simple English Wikipedia** (~200K articles, ~600MB uncompressed)
- Downloaded from Wikimedia dumps: https://dumps.wikimedia.org/simplewiki/latest/
- Format: MediaWiki XML dump

### Pipeline Stages

1. **Download** - Fetch latest Simple English Wikipedia dump
2. **Parse** - Stream parse XML to extract article title + text
3. **Embed** - Generate 768-dimensional embeddings using sentence-transformers/all-mpnet-base-v2
4. **Ingest** - Batch POST to Engram HTTP API with proper error handling
5. **Verify** - Confirm articles stored correctly

### Performance Targets

- **Throughput**: 100-500 articles/second (batch processing)
- **Memory**: <1GB peak (streaming XML parsing)
- **Accuracy**: 100% article storage (retry on transient failures)

## Components

### ingest.py
Main ingestion script with:
- Streaming XML parser (ElementTree iterparse)
- Sentence-transformers embedding generation (batched)
- Parallel HTTP requests to Engram (asyncio)
- Progress tracking (tqdm)
- Checkpoint/resume support
- Error handling and retry logic

### requirements.txt
Python dependencies:
- sentence-transformers (embeddings)
- httpx (async HTTP client)
- tqdm (progress bars)
- numpy (vector operations)

## Usage

### Initial Setup

```bash
cd tools/wikipedia_ingest

# Install dependencies
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Download Wikipedia dump (automatic on first run)
python ingest.py --download-only
```

### Ingest Wikipedia

```bash
# Full ingestion (all articles)
python ingest.py --endpoint http://localhost:7432

# Limit to N articles (for testing)
python ingest.py --endpoint http://localhost:7432 --limit 1000

# Resume from checkpoint
python ingest.py --endpoint http://localhost:7432 --resume checkpoints/latest.json

# Batch size tuning (default: 32)
python ingest.py --endpoint http://localhost:7432 --batch-size 64
```

### Verify Ingestion

```bash
# Check article count
curl http://localhost:7432/api/v1/memories/count

# Sample random memories
curl http://localhost:7432/api/v1/memories?limit=10
```

## Implementation Details

### Embedding Model
- **Model**: sentence-transformers/all-mpnet-base-v2
- **Dimensions**: 768 (matches Engram EMBEDDING_DIM)
- **Quality**: SOTA for semantic similarity (SBERT benchmark)
- **Speed**: ~1000 sentences/sec on CPU, ~10K/sec on GPU

### Wikipedia Parsing
- **Strategy**: Stream parsing (memory efficient)
- **Filters**: Skip redirects, disambiguation pages, stubs
- **Text extraction**: Clean wikitext → plain text
- **Metadata**: Preserve title, timestamp for episodes

### Batching Strategy
- **Embedding batching**: 32 articles per batch (GPU memory efficient)
- **HTTP batching**: Pipeline 10 concurrent requests (network efficient)
- **Checkpoint**: Every 1000 articles (resume support)

### Error Handling
- **Transient failures**: Exponential backoff retry (3 attempts)
- **Permanent failures**: Log and continue (don't block ingestion)
- **Checkpoint**: Save progress every 1000 articles
- **Resume**: Skip already-ingested articles on restart

## Performance Characteristics

### Resource Usage
- **CPU**: 1-2 cores (embedding generation)
- **GPU**: Optional (10x speedup for embeddings)
- **Memory**: ~800MB peak (model + batches)
- **Network**: ~1-5 MB/s (API requests)
- **Disk**: ~600MB (Wikipedia dump)

### Throughput
- **CPU-only**: ~100-200 articles/sec
- **With GPU**: ~500-1000 articles/sec
- **Total time**: ~15-30 minutes for 200K articles (CPU)

## Data Quality

### Article Filtering
- Minimum length: 100 characters (skip stubs)
- Exclude: Redirects, disambiguation, templates
- Content: Extract main article text (no infoboxes)

### Embedding Quality
- Semantic coherence: Validated on SBERT benchmarks
- Dimensional consistency: All vectors exactly 768 dims
- Confidence scores: Derived from article length and quality

## Monitoring

### Progress Tracking
- Real-time progress bar (tqdm)
- ETA calculation
- Throughput metrics (articles/sec)
- Error rate tracking

### Logs
- `logs/ingest.log` - Detailed ingestion log
- `logs/errors.log` - Failed articles for retry
- `checkpoints/*.json` - Resume checkpoints

## Testing

### Unit Tests
```bash
pytest tests/test_parser.py
pytest tests/test_embeddings.py
pytest tests/test_api_client.py
```

### Integration Test
```bash
# Ingest 100 articles to test Engram instance
python ingest.py --endpoint http://localhost:7432 --limit 100 --test
```

## Production Checklist

- [ ] Engram server running on target endpoint
- [ ] Sufficient disk space (~600MB for dump)
- [ ] Python 3.8+ with venv
- [ ] Dependencies installed (requirements.txt)
- [ ] GPU available (optional, for speedup)
- [ ] Network connectivity to Wikimedia and Engram

## Troubleshooting

### "Model download failed"
- Ensure internet connectivity for Hugging Face Hub
- Model downloads automatically on first run (~420MB)
- Cached in ~/.cache/huggingface/

### "Connection refused to Engram"
- Verify Engram server running: `engram status`
- Check endpoint URL matches: `--endpoint http://localhost:7432`
- Test connectivity: `curl http://localhost:7432/health`

### "Out of memory"
- Reduce batch size: `--batch-size 16`
- Close other applications
- Use GPU if available (frees CPU memory)

### "Ingestion stalled"
- Check Engram server health: `engram status`
- Review logs: `tail -f logs/ingest.log`
- Resume from checkpoint: `--resume checkpoints/latest.json`
