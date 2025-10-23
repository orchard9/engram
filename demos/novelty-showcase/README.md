# Engram Novelty Showcase Demo

An 8-minute interactive demonstration of three features you won't find in traditional databases.

## What You'll Experience

This demo shows cognitive memory operations that eliminate entire categories of manual database work.

### Memories That Fade Naturally

**The Problem**: Your production database remembers a bug report from 3 months ago with the same clarity as yesterday's deploy. Traditional TTL is binary (exists or deleted) and requires manual configuration per table.

**What Engram Does**: Implements Ebbinghaus forgetting curves at the storage layer. Older memories naturally decay in confidence without configuration. Accessing a memory refreshes it (spaced repetition effect).

**You'll See**:
- Three memories stored at different ages (1 day, 1 week, 1 month)
- Automatic confidence decay matching psychological research
- Spaced repetition: accessing an old memory increases its confidence

**Technical Impact**: Zero TTL configuration, human-like memory behavior, built-in spaced repetition for learning systems

### Finding Connections You Never Created

**The Problem**: In SQL, you only find what you explicitly join. In graph databases, you only traverse relationships you defined. If you didn't link "mitochondria" to "ATP synthesis," that connection doesn't exist.

**What Engram Does**: Uses spreading activation from cognitive science. Query for one concept, activation spreads through semantic similarity to related concepts - no predefined schema required.

**You'll See**:
- Build a knowledge graph with 5-6 biological concepts
- Query for "mitochondria"
- Watch activation spread to ATP, cellular respiration, DNA
- Contrast with SQL WHERE clause (zero results without exact match)

**Technical Impact**: Schema-free relationship discovery, sub-10ms single-hop activation, human-like associative recall

### A Database That Learns Patterns Automatically

**The Problem**: Extracting patterns from raw events requires manual ETL pipelines, custom aggregation queries, or separate analytics databases. You're the pattern detector.

**What Engram Does**: Automatic episodic-to-semantic consolidation. Store raw episodes, come back later to find the system has discovered patterns, compressed storage, and can explain what it learned with full citation trails.

**You'll See**:
- Store 20 episodes about "learning Rust" (backdated for demo)
- Scheduler runs every 60 seconds
- Semantic patterns automatically extracted
- 20:1 storage compression with source attribution

**Technical Impact**: Zero ETL code, 95% storage reduction, maintains evidence chains for explainability

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

## Demo Walkthrough

### Part 0: Setup & Verification (1 minute)
Before starting, verify your environment:
- Server running at `http://localhost:7432`
- Required tools: `curl`, `jq`, Bash 4.0+
- Run `./preflight-check.sh` for automated verification

### Part 1: Watch Memories Fade (2 minutes)
**What Happens**: Store three memories at different ages (1 day, 1 week, 1 month old). Query them to see confidence scores decay over time. Access the oldest memory to trigger spaced repetition, then re-query to see its confidence increase.

**Technical Details**:
- Uses backdated timestamps: `date -u -v-7d` (macOS) or `date -d '7 days ago'` (Linux)
- Confidence decay follows Ebbinghaus curve (within 5% accuracy)
- Spaced repetition effect refreshes old memories

**Success Indicator**: Older memories show progressively lower confidence until accessed

### Part 2: Discover Hidden Connections (2 minutes)
**What Happens**: Build a small knowledge graph about cellular biology (mitochondria, ATP, respiration, DNA). Query for "mitochondria" and watch activation spread to related concepts without explicit relationships.

**Technical Details**:
- Spreading activation completes in <10ms for single-hop
- No schema required - semantic similarity drives connection discovery
- Contrast shown: SQL `WHERE` clause returns zero results for same query

**Success Indicator**: Query for one concept returns 4-5 related concepts automatically

### Part 3: Let the Database Learn (2 minutes)
**What Happens**: Store 20 episodes about "learning Rust" (backdated to make them eligible). Check consolidation scheduler status. View automatically extracted semantic patterns with storage compression metrics.

**Technical Details**:
- Consolidation scheduler runs every 60 seconds
- Episodes must be >1 day old (biological design - backdated timestamps bypass this)
- 20:1 compression ratio (95% storage reduction)
- Full citation trails maintained for explainability

**Success Indicator**: Semantic patterns discovered from episodes, compression ratio displayed

### Part 4: All Three Together (2 minutes)
**What Happens**: Complex query demonstrating decay + spreading + consolidation working in concert. Query for "programming" shows vivid (recent), associated (spreading), and reconstructed (consolidated) memories simultaneously.

**Technical Details**:
- Multiple retrieval pathways active in single query
- Confidence scores reflect age, similarity, and pattern strength
- Response shows three memory categories with separate mechanisms

**Success Indicator**: Single query returns memories from all three systems

### Part 5: Metrics & Positioning (1 minute)
Summary comparison table: Engram vs traditional databases on TTL management, relationship discovery, and pattern extraction.

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

| Section | Duration | What Happens |
|---------|----------|--------------|
| Part 0: Setup | 1min | Verify environment, run preflight check |
| Part 1: Decay | 2min | Watch memories fade over time |
| Part 2: Spreading | 2min | Discover connections you never created |
| Part 3: Consolidation | 2min | Let database learn patterns automatically |
| Part 4: Synthesis | 2min | All three features working together |
| Part 5: Metrics | 1min | Positioning vs traditional databases |
| **Total** | **~10min** | Complete demonstration |

## Success Criteria

A successful demo shows:

1. **Decay**: Older memories have lower confidence scores (within 5% of Ebbinghaus curve)
2. **Spreading**: Related concepts discovered without explicit traversal (<10ms for single-hop)
3. **Consolidation**: Scheduler active, patterns discovered, 20:1 compression ratio
4. **Performance**: All queries complete in <100ms
5. **Reliability**: No HTTP errors, all API calls succeed

---

## Appendix: For Developers

### Customization Options

<details>
<summary>Adjust Demo Timing</summary>

Edit pause durations in `demo.sh`:
```bash
PAUSE_SHORT=2    # Default: 2 seconds
PAUSE_MEDIUM=3   # Default: 3 seconds
PAUSE_LONG=5     # Default: 5 seconds
```
</details>

<details>
<summary>Change Server URL</summary>

```bash
BASE_URL="http://localhost:8080"  # Default: 7432
```
</details>

<details>
<summary>Add Custom Demo Data</summary>

Modify the data in each section:
- Part 1 (Decay): Lines 85-110 in `demo.sh`
- Part 2 (Spreading): Lines 150-175 in `demo.sh`
- Part 3 (Consolidation): Lines 220-250 in `demo.sh`
</details>

### Cleanup After Demo

<details>
<summary>Remove Demo Data</summary>

```bash
# Option 1: Restart server (clears in-memory data)
./target/release/engram stop
./target/release/engram start

# Option 2: Use API to delete specific memories
curl -X DELETE http://localhost:7432/api/v1/memories/{memory_id}
```
</details>

### Using This Demo

<details>
<summary>For Presentations</summary>

1. Run in "automated mode" for smooth flow:
   ```bash
   bash demo.sh < /dev/null
   ```
2. Record terminal with asciinema for sharing
3. Highlight key metrics in Part 5 (finale section)
</details>

<details>
<summary>For Sales/Marketing</summary>

1. Focus on Part 4 (synthesis) showing all features together
2. Emphasize practical impact statements from feature descriptions
3. Reference competitive positioning table in finale
</details>

<details>
<summary>For Technical Evaluation</summary>

1. Examine API responses in detail
2. Compare confidence intervals across queries
3. Verify consolidation compression ratios match claims
4. Test spreading activation with custom queries beyond demo script
</details>

## Related Documentation

- Novelty Analysis: `/tmp/engram_novelty_analysis.md`
- Full API Documentation: http://localhost:7432/docs
- Vision Document: `vision.md`
- Quickstart Guide: `quickstart.md`
- UAT Results: `uat-results/SUMMARY.md`

### Troubleshooting

<details>
<summary>Demo Script Won't Execute</summary>

```bash
chmod +x demo.sh
```
</details>

<details>
<summary>Date Command Errors (macOS vs Linux)</summary>

The script handles both platforms automatically:
```bash
# macOS: date -v-7d
# Linux: date -d '7 days ago'
```

If you see date-related errors, check your platform's date command syntax.
</details>

<details>
<summary>JSON Parsing Errors</summary>

Check jq version (requires 1.6+):
```bash
jq --version
```

Install or upgrade jq:
```bash
# macOS
brew install jq

# Ubuntu/Debian
apt-get install jq
```
</details>

<details>
<summary>Memory Not Found After Storing</summary>

Allow time for indexing to complete:
```bash
sleep 1  # Between store and recall operations
```

The demo script includes appropriate delays automatically.
</details>

<details>
<summary>Episodes Not Consolidating</summary>

**Expected behavior**: Episodes must be >1 day old before consolidation (biological design).

The demo uses backdated timestamps to bypass this requirement:
```bash
ONE_WEEK_AGO=$(date -u -v-7d +"%Y-%m-%dT%H:%M:%SZ")
```

If consolidation still doesn't run, check:
1. Server logs: `./target/release/engram logs`
2. Scheduler status via API: `curl http://localhost:7432/api/v1/consolidations`
</details>

### Contributing to This Demo

Improvements welcome:

1. Add more realistic use cases from your domain
2. Include visualization of spreading activation (D3.js, Graphviz)
3. Create video walkthrough with narration
4. Add performance benchmarking against traditional databases
5. Include error scenario demonstrations (what happens when things go wrong)

Submit PRs or open issues at: https://github.com/orchard9/engram/issues

### Questions?

- GitHub Issues: https://github.com/orchard9/engram/issues
- API Documentation: http://localhost:7432/docs
- Vision Document: `vision.md` (architectural philosophy)
- Full Usage Guide: `usage.md` (production deployment)
