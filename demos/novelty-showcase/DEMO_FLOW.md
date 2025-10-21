# Novelty Showcase Demo Flow

Visual guide to the 8-minute demonstration structure.

## Timeline Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    ENGRAM NOVELTY SHOWCASE                   │
│                         (~8 minutes)                         │
└─────────────────────────────────────────────────────────────┘

00:00 ━━━━━━━━━━━━┓ ACT 1: The Problem
00:30              ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
                                                 │
02:30 ━━━━━━━━━━━━┓ ACT 2: Psychological Decay   │
04:30              ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┫
                                                 │
                   ACT 3: Spreading Activation   │  Main Demo
06:30 ━━━━━━━━━━━━┛                             │
                                                 │
                   ACT 4: Memory Consolidation   │
07:30 ━━━━━━━━━━━━┛                             │
                                                 │
                   ACT 5: The Synthesis          │
08:00 ━━━━━━━━━━━━┛                             │
                                                 ┘
                   Finale: Key Metrics
08:30 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
```

## Detailed Act Breakdown

### Act 1: The Problem (30 seconds)
**Objective**: Establish why cognitive memory is needed

```
Problem Statement
├─ Traditional DB: Binary TTL (remember all or forget all)
├─ Human Memory: Gradual decay with reinforcement
├─ Traditional DB: Exact matches only
├─ Human Memory: Associative recall
└─ Traditional DB: No pattern learning
   Human Memory: Consolidation during sleep

→ Engram bridges this gap
```

**Key Message**: "Databases should think like you do"

---

### Act 2: Psychological Decay (2 minutes)
**Novelty Score**: 8.2/10
**Objective**: Demonstrate Ebbinghaus forgetting curves

```
Step 1: Store 3 memories at different ages
┌─────────────────────────────────────┐
│ Memory 1: 1 day old   (Paris)       │ Confidence: 0.92
│ Memory 2: 1 week old  (Tokyo)       │ Confidence: 0.78
│ Memory 3: 1 month old (Berlin)      │ Confidence: 0.45
└─────────────────────────────────────┘

Step 2: Recall all three → Show decay pattern

Step 3: Access Berlin memory (spaced repetition)
        ↓
Step 4: Re-query → Berlin confidence boosted to 0.68
```

**Key Metrics**:
- Decay accuracy: ±5% of Ebbinghaus curve
- Automatic confidence adjustment
- No manual TTL management

**Impact Statement**:
> "The database automatically manages memory strength, just like your brain"

---

### Act 3: Spreading Activation (2 minutes)
**Novelty Score**: 7.3/10
**Objective**: Show associative memory retrieval

```
Build Knowledge Graph:
    Mitochondria ──┬── ATP
                   ├── Cellular respiration
                   ├── Mitochondrial DNA
                   └── Endosymbiotic theory

Query: "mitochondria"

Activation Spreading:
    [Query] → Mitochondria (1.0)
           └─→ ATP (0.85)               ← semantic similarity
           └─→ Respiration (0.72)       ← conceptual link
           └─→ Endosymbiotic (0.45)     ← weak indirect link

Compare to SQL:
    WHERE content LIKE '%mitochondria%'
    Result: Only exact "mitochondria" matches (4 results)

    Engram spreading activation:
    Result: All related concepts (5+ results)
```

**Key Metrics**:
- Spreading speed: <10ms single-hop
- No explicit relationship traversal needed
- Schema-free association discovery

**Impact Statement**:
> "Engram thinks associatively, not just through exact matches"

---

### Act 4: Memory Consolidation (2 minutes)
**Novelty Score**: 7.2/10
**Objective**: Automatic pattern learning from experiences

```
Phase 1: Store 20 Episodes (backdated)
┌───────────────────────────────────────────┐
│ Day 1:  "Read Rust book on ownership"     │
│ Day 2:  "Practiced borrowing in Rust"     │
│ Day 3:  "Debugged lifetime errors"        │
│ ...                                       │
│ Day 20: "Implemented linked list in Rust" │
└───────────────────────────────────────────┘
         ↓
Phase 2: Consolidation Scheduler (60s cadence)
         ↓
Phase 3: Pattern Detection (hierarchical clustering)
         ↓
Phase 4: Semantic Extraction
┌────────────────────────────────────────────┐
│ Pattern: "Active learning of Rust"        │
│ Strength: 0.82                            │
│ Sources: 20 episodes                      │
│ Features: Temporal sequence (daily)       │
└────────────────────────────────────────────┘

Storage Impact:
    Before: 20 episodes × 3KB = 60KB
    After:  3 patterns × 1KB = 3KB
    Compression: 20:1 ratio (95% reduction)
```

**Key Metrics**:
- Compression ratio: 20:1 typical
- Automatic discovery (no manual ETL)
- Citation trail preserved

**Impact Statement**:
> "The database learns patterns from your data automatically"

---

### Act 5: The Synthesis (1 minute)
**Objective**: Show all features working together

```
Complex Query: "programming" (threshold: 0.2)

Retrieval Pathways:
┌──────────────────────────────────────────────────┐
│ 1. Direct Matches                                │
│    • "Rust programming language" (conf: 0.9)     │
│                                                  │
│ 2. Spreading Activation                          │
│    • "Software engineering" (activated: 0.7)     │
│    • "Code debugging" (activated: 0.5)           │
│                                                  │
│ 3. Consolidated Patterns                         │
│    • "Active learning behavior" (pattern: 0.82)  │
│                                                  │
│ 4. Decay-Adjusted Results                        │
│    • Recent memories: Higher confidence          │
│    • Older memories: Lower but still accessible  │
└──────────────────────────────────────────────────┘

All 3 novel features working in concert:
  ✓ Decay affects confidence scores
  ✓ Spreading finds related concepts
  ✓ Consolidation reveals meta-patterns
```

**Impact Statement**:
> "Rich, multi-pathway recall - just like human memory"

---

### Finale: Key Metrics (1 minute)
**Objective**: Competitive positioning and technical summary

#### Comparison Table

| Feature              | Engram                    | Redis     | Neo4j          | Pinecone      |
|----------------------|---------------------------|-----------|----------------|---------------|
| Forgetting           | Ebbinghaus curve          | Binary    | None           | None          |
| Association          | Spreading activation      | None      | Traversal      | Similarity    |
| Pattern learning     | Automatic consolidation   | None      | None           | None          |
| Confidence intervals | Yes (probabilistic)       | No        | No             | Scores only   |
| Storage efficiency   | 20:1 compression          | None      | None           | Index only    |

#### Target Use Cases

```
Primary:
├─ AI agents with human-like memory
├─ Spaced repetition learning systems
├─ Knowledge graphs with associative retrieval
└─ Cognitive architecture research

Secondary:
├─ Applications requiring uncertainty tracking
├─ Systems needing automatic pattern discovery
└─ Memory-aware AI assistants
```

#### Positioning Statement

```
┌─────────────────────────────────────────────┐
│  Engram is not a "better graph database"    │
│  It's a "cognitive memory system"           │
│                                             │
│  The only database that thinks like you do  │
└─────────────────────────────────────────────┘
```

---

## Demo Success Indicators

### ✅ Working Correctly When:

1. **Decay Demo**:
   - Older memories have measurably lower confidence
   - Accessing memory boosts its confidence
   - Confidence changes follow expected curve

2. **Spreading Demo**:
   - Related concepts discovered without explicit links
   - Activation levels correlate with semantic distance
   - Results include concepts not in query

3. **Consolidation Demo**:
   - Scheduler shows active status
   - Patterns discovered (or noted as pending if too recent)
   - Storage compression ratio calculated

4. **Performance**:
   - All API calls complete in <100ms
   - No HTTP errors
   - Server remains stable throughout

### ❌ Issues to Watch For:

- Connection refused → Server not running
- Empty results → Data not indexed yet (wait 1-2s)
- No consolidation → Episodes too recent (expected, noted in demo)
- Slow queries → Check server logs for performance issues

---

## Customization Options

### Adjust Demo Timing
```bash
# Edit demo.sh
PAUSE_SHORT=2    # Wait after minor steps
PAUSE_MEDIUM=3   # Wait after API calls
PAUSE_LONG=5     # Wait for user to read results
```

### Change Data Volume
```bash
# Act 4: Consolidation episodes
for i in $(seq 1 50); do  # Change from 20 to 50
    # Store more episodes
done
```

### Add Custom Use Cases
Each act is modular - add new demonstrations:
```bash
act6_custom_use_case() {
    print_header "ACT 6: Your Custom Demo"
    # Your custom demonstration
}
```

---

## Presentation Tips

### For Technical Audiences
- Emphasize implementation details (Ebbinghaus curve accuracy, HNSW indexing)
- Show API responses in detail
- Discuss performance metrics
- Compare to academic cognitive architectures

### For Business Audiences
- Focus on use cases (AI assistants, learning systems)
- Highlight automation (no manual ETL, no TTL management)
- Emphasize cost savings (20:1 compression)
- Show competitive differentiation table

### For Sales Demos
- Start with Act 5 (synthesis) to show full capability
- Then zoom into Act 2-4 for technical depth
- End with competitive positioning
- Keep under 5 minutes for initial pitch

### For Recorded Demos
- Run in automated mode (no pauses)
- Record with asciinema: `asciinema rec demo.cast`
- Add voiceover explaining each section
- Share recording at: https://asciinema.org

---

## Related Resources

- **Novelty Analysis**: `/tmp/engram_novelty_analysis.md`
- **Full README**: [demos/novelty-showcase/README.md](README.md)
- **API Documentation**: http://localhost:7432/docs
- **Vision Document**: [vision.md](../../vision.md)
- **UAT Results**: [uat-results/SUMMARY.md](../../uat-results/SUMMARY.md)

---

## Questions & Feedback

Improve this demo:
- Add visualization of activation spreading
- Include performance benchmarks
- Create video walkthrough
- Add error scenario demonstrations

Submit issues: https://github.com/orchard9/engram/issues
