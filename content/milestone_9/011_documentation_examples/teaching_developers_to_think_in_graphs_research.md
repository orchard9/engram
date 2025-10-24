# Research: Teaching Developers to Think in Graphs

## Key Findings

### 1. Mental Model Shift

Developers come from SQL/relational backgrounds. Graph thinking is different:

**Relational**: Entities in tables, relationships via foreign keys
**Graph**: Nodes ARE entities, edges ARE relationships (first-class)

**Key cognitive shift**: From "join tables to find relationships" to "traverse edges to find relationships"

### 2. Documentation Structure (Diátaxis Framework)

**Tutorials**: Learning-oriented (newcomers)
- "Your First RECALL Query"
- "Understanding SPREAD Activation"
- "Building Cognitive Memory Systems"

**How-To Guides**: Problem-oriented (practitioners)
- "How to Query Episodic Memory"
- "How to Optimize SPREAD Performance"
- "How to Debug Low-Confidence Results"

**Explanation**: Understanding-oriented (learners)
- "Why RECALL Is Not SELECT"
- "How Spreading Activation Works"
- "Confidence Calibration Explained"

**Reference**: Information-oriented (experts)
- "Query Language Syntax Reference"
- "API Documentation"
- "Error Code Catalog"

### 3. Progressive Disclosure

Start simple, add complexity gradually:

**Level 1**: Basic RECALL
```
RECALL episode
```

**Level 2**: Add constraints
```
RECALL episode WHERE confidence > 0.7
```

**Level 3**: Semantic similarity
```
RECALL episode WHERE content SIMILAR TO embedding
```

**Level 4**: Composition with SPREAD
```
RECALL episode THEN SPREAD MAX_HOPS 2
```

Each level builds on previous. Don't overwhelm initially.

### 4. Concrete Examples

Abstract explanations confuse. Concrete examples clarify:

Bad: "SPREAD implements spreading activation with exponential decay"
Good: "Think 'coffee' → 'morning' activates strongly (0.86), 'breakfast' weakly (0.43)"

Use everyday analogies, specific numbers, relatable scenarios.

### 5. Interactive Playgrounds

Static docs insufficient. Developers need hands-on:
- Online query playground (in-browser)
- Example datasets (pre-loaded memories)
- Instant feedback (parse errors, results)

Learning by doing > learning by reading.

### 6. Error Message Documentation

Document every error with:
- What it means
- Why it occurs
- How to fix it
- Example of correct usage

Users search error messages. Make them findable.

## References

1. "Diátaxis Framework", Daniele Procida (documentation structure)
2. "Stripe API Documentation" (best practices)
3. "Learning by Doing", Kolb's experiential learning theory
4. "Mental Models", Nielsen Norman Group (UX research)
