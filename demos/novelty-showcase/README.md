# Engram Novelty Showcase Demo

An 8-minute interactive demonstration of Engram's three most novel features.

## What This Demo Showcases

Based on comprehensive novelty analysis (scored 1-10 on Technical Novelty, Implementation Quality, Practical Impact, Differentiation, and Value Delivery):

### 1. Psychological Decay (Score: 8.2/10)
**Why Novel**: No other production database implements Ebbinghaus forgetting curves as a storage primitive.

**Demo**:
- Store 3 memories at different ages (1 day, 1 week, 1 month old)
- Show automatic decay over time with confidence intervals
- Demonstrate spaced repetition effect (accessing old memory refreshes it)

**Impact**: Eliminates manual TTL management, enables spaced repetition systems, provides human-like memory behavior

### 2. Spreading Activation (Score: 7.3/10)
**Why Novel**: Uses cognitive science activation dynamics instead of traditional graph traversal

**Demo**:
- Build knowledge graph about cellular biology
- Query for "mitochondria"
- Show activation spreading to related concepts (ATP, respiration, DNA)
- Contrast with SQL WHERE clauses (only exact matches)

**Impact**: Discover connections without schema knowledge, more human-like association, no explicit relationship traversal needed

### 3. Memory Consolidation (Score: 7.2/10)
**Why Novel**: Automatic pattern extraction from episodes with no manual ETL

**Demo**:
- Store 20 episodes about "learning Rust" over several days
- Show automatic scheduler running every 60s
- Display discovered semantic patterns
- Demonstrate storage compression (20:1 ratio)

**Impact**: Database learns from your data automatically, significant storage savings, maintains citation trails

## Prerequisites

### Running Server
```bash
# Start Engram
cd /path/to/engram
./target/release/engram start

# Verify server is running
curl http://localhost:7432/health
```

### Required Tools
- `curl` (for API calls)
- `jq` (for JSON formatting)
- Bash 4.0+ (for script execution)

Install jq:
```bash
# macOS
brew install jq

# Ubuntu/Debian
apt-get install jq
```

## Running the Demo

### Step 0: Pre-flight Check (Recommended)
Verify all prerequisites before starting:
```bash
cd demos/novelty-showcase
./preflight-check.sh
```

This checks for:
- Required tools (curl, jq)
- Engram server status
- API endpoint accessibility
- Demo script permissions

### Option 1: Interactive Mode (Recommended)
```bash
cd demos/novelty-showcase
chmod +x demo.sh
./demo.sh
```

Press ENTER after each section to proceed at your own pace.

### Option 2: Automated Mode
```bash
# Run without pauses (for recording/presentations)
cd demos/novelty-showcase
bash demo.sh < /dev/null
```

## Demo Structure

### Act 1: The Problem (30 seconds)
Explains why traditional databases fail at human-like memory:
- Binary TTL vs gradual decay
- Exact matches vs associative recall
- No pattern learning

### Act 2: Psychological Decay (2 minutes)
Demonstrates Ebbinghaus forgetting curves:
1. Store memories at different ages
2. Show confidence decay over time
3. Access old memory to refresh it
4. Re-query to show spaced repetition effect

**Key Metric**: Decay accuracy within 5% of Ebbinghaus curve

### Act 3: Spreading Activation (2 minutes)
Shows associative memory retrieval:
1. Build knowledge graph (5-6 related concepts)
2. Query for single concept
3. Display spreading activation results
4. Contrast with SQL WHERE clause

**Key Metric**: <10ms activation spreading for single-hop

### Act 4: Memory Consolidation (2 minutes)
Demonstrates automatic pattern learning:
1. Store 20 related episodes (backdated)
2. Check consolidation scheduler status
3. Show discovered semantic patterns
4. Display storage compression ratio

**Key Metric**: 20:1 compression ratio (95% reduction)

### Act 5: The Synthesis (1 minute)
Complex query showcasing all features working together:
- Query: "programming" with low threshold
- Shows: decay, spreading, and consolidated patterns
- Demonstrates multiple retrieval pathways

### Finale: Key Metrics (1 minute)
Summary table comparing Engram to traditional databases

## Expected Output

### Successful Run Indicators
- ✓ All curl commands return HTTP 200
- ✓ Confidence scores show decay pattern (older = lower confidence)
- ✓ Spreading activation finds related concepts
- ✓ Consolidation scheduler shows active status

### Common Issues

#### "Connection refused" Error
```bash
# Server not running - start it first
./target/release/engram start
```

#### "Command not found: jq"
```bash
# Install jq for JSON formatting
brew install jq  # macOS
apt-get install jq  # Linux
```

#### Episodes not consolidating
**Expected behavior**: Episodes must be >1 day old before consolidation (biological design)

The demo uses backdated timestamps to work around this:
```bash
ONE_WEEK_AGO=$(date -u -v-7d +"%Y-%m-%dT%H:%M:%SZ")
```

#### Slow spreading activation
Check server logs for performance issues:
```bash
./target/release/engram logs
```

## Demo Timing

| Section | Duration | Purpose |
|---------|----------|---------|
| Act 1 | 30s | Establish the problem |
| Act 2 | 2min | Psychological decay |
| Act 3 | 2min | Spreading activation |
| Act 4 | 2min | Memory consolidation |
| Act 5 | 1min | Synthesis |
| Finale | 1min | Metrics & positioning |
| **Total** | **~8min** | Complete demo |

## Customization

### Adjust Timing
Edit pause durations in demo.sh:
```bash
PAUSE_SHORT=2    # Default: 2 seconds
PAUSE_MEDIUM=3   # Default: 3 seconds
PAUSE_LONG=5     # Default: 5 seconds
```

### Change Server URL
```bash
BASE_URL="http://localhost:8080"  # Default: 7432
```

### Add Custom Data
Modify the data in each act:
- Act 2: Lines 85-110 (decay demo data)
- Act 3: Lines 150-175 (spreading activation data)
- Act 4: Lines 220-250 (consolidation episodes)

## Cleanup After Demo

Remove demo data:
```bash
# Option 1: Restart server (clears in-memory data)
./target/release/engram stop
./target/release/engram start

# Option 2: Use API to delete specific memories
curl -X DELETE http://localhost:7432/api/v1/memories/{memory_id}
```

## Using This Demo

### For Presentations
1. Run in "automated mode" for smooth flow
2. Record terminal with asciinema for sharing
3. Highlight key metrics in finale section

### For Sales/Marketing
1. Focus on Act 5 (synthesis) showing all features together
2. Emphasize practical impact statements
3. Reference competitive positioning table

### For Technical Evaluation
1. Examine API responses in detail
2. Compare confidence intervals across queries
3. Verify consolidation compression ratios
4. Test spreading activation with custom queries

## Related Documentation

- Novelty Analysis: `/tmp/engram_novelty_analysis.md`
- Full API Documentation: http://localhost:7432/docs
- Vision Document: `vision.md`
- Quickstart Guide: `quickstart.md`
- UAT Results: `uat-results/SUMMARY.md`

## Troubleshooting

### Demo Script Won't Execute
```bash
chmod +x demo.sh
```

### Date Command Errors
**macOS vs Linux date syntax differs**

The script handles both:
```bash
# macOS: date -v-7d
# Linux: date -d '7 days ago'
```

### JSON Parsing Errors
Check jq version:
```bash
jq --version  # Should be 1.6+
```

### Memory Not Found
Ensure sufficient time between store and recall:
```bash
sleep 1  # Allow indexing to complete
```

## Success Criteria

A successful demo shows:

1. **Decay**: Older memories have lower confidence scores
2. **Spreading**: Related concepts discovered without explicit traversal
3. **Consolidation**: Scheduler active, patterns discovered (or noted as pending)
4. **Performance**: All queries complete in <100ms
5. **Reliability**: No HTTP errors, all API calls succeed

## Contributing

To improve this demo:

1. Add more realistic use cases
2. Include visualization of spreading activation
3. Create video walkthrough
4. Add performance benchmarking
5. Include error scenario demonstrations

## Questions?

- GitHub Issues: https://github.com/orchard9/engram/issues
- Documentation: http://localhost:7432/docs
- Vision Document: `vision.md`
