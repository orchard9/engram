# Query Language Reference

Engram's query language is designed around cognitive operations rather than traditional database queries. Think of it like asking your brain to recall something, not like querying a SQL database.

## Philosophy

Traditional databases force you to think in tables and joins. Engram lets you think in memories and associations. Every query returns confidence intervals because, like human memory, nothing is absolutely certain.

## Query Operations

### RECALL - Retrieve Memories

Retrieves memories matching a pattern. Like asking your brain "what do I remember about X?"

**Basic Syntax:**

```
RECALL <pattern> [WHERE <constraints>] [CONFIDENCE <threshold>] [BASE_RATE <value>]
```

**Patterns:**

1. **By Node ID** - Exact memory identifier

   ```
   RECALL episode_123
   ```

2. **By Embedding** - Semantic similarity search

   ```
   RECALL [0.1, 0.2, 0.3, ...] THRESHOLD 0.8
   ```

   The vector must be exactly 768 dimensions. Threshold is optional (defaults to 0.8) and represents minimum cosine similarity [0.0-1.0].

3. **By Content** - Substring match

   ```
   RECALL "neural networks"
   ```

**Constraints:**

Add `WHERE` clause to filter results:

- **Content matching:**

  ```
  WHERE content CONTAINS "machine learning"
  ```

- **Temporal filtering:**

  ```
  WHERE created BEFORE "2024-01-01"
  WHERE created AFTER "1704067200"
  ```

  Accepts ISO 8601 dates or Unix timestamps (seconds since epoch).

- **Confidence filtering:**

  ```
  WHERE confidence > 0.7
  WHERE confidence < 0.9
  ```

- **Memory space filtering:**

  ```
  WHERE IN SPACE space_id
  ```

Multiple constraints are implicitly AND-ed:

```
RECALL episode WHERE confidence > 0.7 content CONTAINS "AI"
```

**Confidence Thresholds:**

Filter results by confidence:

```
RECALL episode CONFIDENCE > 0.8
RECALL episode CONFIDENCE < 0.5
```

**Base Rate Priors:**

Specify prior probability for Bayesian updating:

```
RECALL episode BASE_RATE 0.3
```

**Complete Example:**

```
RECALL [0.1, 0.2, 0.3] THRESHOLD 0.85
  WHERE confidence > 0.7
  content CONTAINS "deep learning"
  created AFTER "2024-01-01"
  CONFIDENCE > 0.8
```

**Performance Characteristics:**

- Node ID lookup: O(1) average, <1ms
- Embedding similarity: O(n log k) with HNSW index, typically 5-50ms for millions of memories
- Content match: O(n) scan, avoid on large datasets
- Constraints evaluated after pattern matching

---

### SPREAD - Activation Spreading

Spreads activation through the memory graph from a source node. Like how thinking about "coffee" naturally leads to "morning" then "alarm clock".

**Syntax:**

```
SPREAD FROM <node_id> [MAX_HOPS <n>] [DECAY <rate>] [THRESHOLD <activation>]
```

**Parameters:**

- **FROM** (required): Source node identifier
- **MAX_HOPS** (optional): Maximum spreading distance (default: 3)
  - Range: 0-65535
  - Typical values: 2-5
- **DECAY** (optional): Activation decay per hop (default: 0.1)
  - Range: 0.0-1.0
  - 0.0 = no decay, 1.0 = instant decay
  - Typical values: 0.1-0.3
- **THRESHOLD** (optional): Minimum activation to continue (default: 0.01)
  - Range: 0.0-1.0
  - Lower = wider spread, higher = focused activation

**Examples:**

Basic spreading:

```
SPREAD FROM episode_123
```

Controlled spreading:

```
SPREAD FROM concept_ai MAX_HOPS 5 DECAY 0.15 THRESHOLD 0.05
```

Wide exploration:

```
SPREAD FROM node_start MAX_HOPS 10 DECAY 0.05 THRESHOLD 0.001
```

Focused activation:

```
SPREAD FROM memory_core MAX_HOPS 2 DECAY 0.3 THRESHOLD 0.1
```

**Performance Characteristics:**

- Exponential in hop count: O(2^hops) for typical graphs
- 3 hops: ~1-10ms
- 5 hops: ~10-100ms
- 10 hops: potentially seconds (use with care)
- Bounded by threshold cutoff in practice

**Use Cases:**

- Finding related concepts
- Context gathering for predictions
- Exploratory memory traversal
- Association discovery

---

### PREDICT - Future State Prediction

Predicts likely future states based on current context. Like your brain predicting "the next word will probably be..."

**Syntax:**

```
PREDICT <pattern> GIVEN <context> [HORIZON <seconds>] [CONFIDENCE <interval>]
```

**Parameters:**

- **pattern**: What to predict (same format as RECALL patterns)
- **GIVEN**: Context nodes (comma-separated list)
- **HORIZON** (optional): Time horizon in seconds
- **CONFIDENCE** (optional): Constrain prediction confidence

**Examples:**

Basic prediction:

```
PREDICT next_event GIVEN current_episode
```

Multiple context nodes:

```
PREDICT outcome GIVEN context1, context2, context3
```

With time horizon:

```
PREDICT state GIVEN current HORIZON 3600
```

(Predict state 1 hour in the future)

With confidence constraint:

```
PREDICT event GIVEN ctx1, ctx2 CONFIDENCE [0.5, 0.9]
```

**Performance Characteristics:**

- Base cost: 2000 units + 500 per context node
- Typical latency: 10-50ms
- Depends on context size and graph density

**Algorithm:**

- Spreads activation from context nodes
- Identifies high-activation patterns
- Weights by temporal correlation
- Returns confidence intervals based on evidence strength

---

### IMAGINE - Pattern Completion

Generates novel combinations through creative pattern completion. Like asking your brain "what would X combined with Y look like?"

**Syntax:**

```
IMAGINE <pattern> [BASED ON <seeds>] [NOVELTY <level>] [CONFIDENCE <threshold>]
```

**Parameters:**

- **pattern**: Pattern to complete
- **BASED ON** (optional): Seed nodes for generation (comma-separated)
- **NOVELTY** (optional): Creativity level 0.0-1.0 (default: 0.5)
  - 0.0 = conservative (stick close to known patterns)
  - 0.5 = moderate creativity
  - 1.0 = highly creative (more distant associations)
- **CONFIDENCE** (optional): Minimum confidence threshold

**Examples:**

Basic imagination:

```
IMAGINE new_concept
```

Seeded generation:

```
IMAGINE hybrid BASED ON concept1, concept2
```

Controlled creativity:

```
IMAGINE novel_idea BASED ON seed1, seed2 NOVELTY 0.3
```

Highly creative:

```
IMAGINE wild_idea NOVELTY 0.9 CONFIDENCE > 0.4
```

**Performance Characteristics:**

- Base cost: 5000 units + 1000 per seed
- Most expensive operation (requires pattern completion)
- Typical latency: 50-200ms
- May return multiple possibilities with confidence scores

**Use Cases:**

- Creative problem solving
- Analogy generation
- Concept blending
- Hypothetical reasoning

---

### CONSOLIDATE - Memory Consolidation

Merges and strengthens episodic memories into semantic knowledge. Like how repeated experiences become learned concepts.

**Syntax:**

```
CONSOLIDATE <episodes> INTO <target> [SCHEDULER <policy>]
```

**Episode Selectors:**

1. **Specific pattern:**

   ```
   CONSOLIDATE episode_pattern INTO semantic_concept
   ```

2. **With constraints:**

   ```
   CONSOLIDATE WHERE confidence > 0.7 INTO knowledge_node
   ```

**Scheduler Policies** (optional):

- **Immediate**: Consolidate right away
- **Interval**: Periodic consolidation
- **Threshold**: Activation-based triggering

**Examples:**

Basic consolidation:

```
CONSOLIDATE episode_123 INTO concept_ml
```

Batch consolidation:

```
CONSOLIDATE WHERE created AFTER "2024-01-01" INTO monthly_summary
```

Pattern-based:

```
CONSOLIDATE "learning experience" INTO expertise_node
```

**Performance Characteristics:**

- Base cost: 10,000 units (most expensive)
- Typical latency: 100-500ms
- Writes to graph (not read-only)
- May trigger background compaction

**Effects:**

- Strengthens connections between related episodes
- Increases confidence of consolidated memories
- May prune redundant episodic details
- Updates semantic layer

---

## Data Types

### Confidence

Represents probability or certainty [0.0, 1.0]:

- `0.0` = no confidence (essentially false)
- `0.5` = uncertain (equal likelihood)
- `1.0` = certain (as sure as possible)

Always represented as interval in results: `[lower, upper]`

### Node Identifier

String identifying a memory node:

- Maximum length: 256 bytes
- Valid characters: letters, digits, underscore, hyphen
- Examples: `episode_123`, `concept_ai`, `user-memory-42`

### Embedding Vector

Dense vector representation (768 dimensions):

- Must be exactly 768 floats
- Typically normalized (unit length)
- Generated by embedding models (e.g., text-embedding-ada-002)

### Time

Continuous time representation:

- ISO 8601 strings: `"2024-01-01T12:00:00Z"`
- Unix timestamps: `"1704067200"` (seconds since epoch)
- Duration in seconds: `3600` (1 hour)

---

## Operators

### Comparison Operators

Used in WHERE clauses and confidence filters:

- `>` Greater than
- `<` Less than
- `>=` Greater than or equal (currently supported in tokenizer)
- `<=` Less than or equal (currently supported in tokenizer)
- `=` Equal (currently supported in tokenizer)

**Note:** Currently only `>` and `<` are fully implemented in confidence constraints.

---

## Error Handling

All queries return confidence intervals, not exceptions. A query with no matches returns empty results with low confidence, not an error.

Syntax errors return detailed messages with:

1. Exact position (line, column)
2. What was found vs. expected
3. Actionable suggestion
4. Example of correct syntax

Example error:

```
Parse error at line 1, column 8:
  Found: ","
  Expected: identifier or embedding [...] or string literal

Suggestion: RECALL requires a pattern: node ID, embedding vector, or content match
Example: RECALL episode_123
```

For complete error reference, see [error-catalog.md](./error-catalog.md).

---

## Performance Guidelines

### Query Cost Hierarchy

From cheapest to most expensive:

1. **Node ID recall**: ~1ms
2. **Spread (3 hops)**: ~5ms
3. **Embedding similarity**: ~10ms
4. **Predict**: ~20ms
5. **Imagine**: ~100ms
6. **Consolidate**: ~200ms

### Optimization Tips

**Fast Queries:**

- Use node IDs when you know them
- Limit spread hop count (2-3 typical)
- Use higher thresholds for narrower results

**Avoid When Possible:**

- Content substring matching (O(n) scan)
- Very large embeddings (stick to 768 dims)
- Deep spreading (>5 hops)
- High novelty IMAGINE queries

**Batch Operations:**
Multiple independent queries can run concurrently. The engine uses lock-free data structures optimized for parallel access.

---

## Grammar Summary

```
query ::= recall_query
        | spread_query
        | predict_query
        | imagine_query
        | consolidate_query

recall_query ::= RECALL pattern
                 [WHERE constraints]
                 [CONFIDENCE threshold]
                 [BASE_RATE value]

spread_query ::= SPREAD FROM node_id
                 [MAX_HOPS integer]
                 [DECAY float]
                 [THRESHOLD float]

predict_query ::= PREDICT pattern
                  GIVEN context_list
                  [HORIZON duration]
                  [CONFIDENCE interval]

imagine_query ::= IMAGINE pattern
                  [BASED ON seed_list]
                  [NOVELTY float]
                  [CONFIDENCE threshold]

consolidate_query ::= CONSOLIDATE episodes
                      INTO node_id
                      [SCHEDULER policy]

pattern ::= node_id
          | embedding
          | string_literal

embedding ::= '[' float_list ']' [THRESHOLD float]

constraints ::= constraint+

constraint ::= content CONTAINS string
             | created BEFORE timestamp
             | created AFTER timestamp
             | confidence > float
             | confidence < float
             | IN SPACE id

context_list ::= node_id (',' node_id)*
seed_list ::= node_id (',' node_id)*
```

---

## Next Steps

- **Examples**: See [engram-core/examples/query_examples.rs](../../engram-core/examples/query_examples.rs) for runnable code
- **Error Reference**: See [error-catalog.md](./error-catalog.md) for all error messages
- **API**: See [cli.md](./cli.md) for command-line usage
- **Spreading**: See [spreading_api.md](./spreading_api.md) for activation spreading details
