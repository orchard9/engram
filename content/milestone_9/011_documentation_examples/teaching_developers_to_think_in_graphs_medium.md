# Teaching Developers to Think in Graphs

Documentation isn't about what YOU know. It's about what THEY need to learn.

When we launched Engram's query language, we faced a challenge: Most developers think in SQL. Graph queries require a different mental model.

Here's how we taught graph thinking without overwhelming people.

## The Mental Model Problem

SQL developers think about data like this:

```sql
SELECT * FROM users WHERE ...
JOIN orders ON ...
```

Entities go in tables. Relationships go in JOINs. This works for 50 years.

Graph thinking is different:

```
SPREAD FROM user_node MAX_HOPS 2
```

Nodes ARE entities. Edges ARE relationships. Traversal is a first-class operation, not a JOIN.

The shift: Relationships aren't afterthoughts. They're primary.

## Documentation Structure: Diátaxis

We use the Diátaxis framework (four doc types, four purposes):

**1. Tutorials** (Learning-oriented)
- Your first RECALL query
- Understanding spreading activation
- Building a cognitive memory system

Goal: Get something working. Build confidence.

**2. How-To Guides** (Problem-oriented)
- How to query episodic memory
- How to optimize SPREAD performance
- How to debug low-confidence results

Goal: Solve specific problems. Practical solutions.

**3. Explanation** (Understanding-oriented)
- Why RECALL is not SELECT
- How spreading activation works
- Confidence calibration explained

Goal: Build mental models. Deep understanding.

**4. Reference** (Information-oriented)
- Query language syntax reference
- API documentation
- Error code catalog

Goal: Look up details. Exhaustive information.

Each serves a different need. Mixing them confuses.

## Progressive Disclosure

Don't show the entire syntax reference upfront. Build complexity gradually.

**Level 1**: Basic query
```
RECALL episode
```

Success! You can query memory. Confidence boost.

**Level 2**: Add constraint
```
RECALL episode WHERE confidence > 0.7
```

Small complexity increase. Builds on level 1.

**Level 3**: Semantic similarity
```
RECALL episode WHERE content SIMILAR TO embedding
```

New concept (embeddings), but familiar syntax (WHERE clause).

**Level 4**: Composition
```
RECALL episode THEN SPREAD MAX_HOPS 2
```

Combine primitives. Power user territory.

Each level: Explain → Example → Exercise. Cognitive load management.

## Concrete Examples Over Abstract Explanations

Abstract: "SPREAD implements spreading activation with exponential decay"

Concrete: "Think 'coffee'. Your brain activates 'morning' strongly (0.86), 'breakfast' weaker (0.43), 'dinner' barely (0.12). That's spreading activation."

Concrete wins. Every time.

**Pattern**: Everyday analogy → Specific numbers → Code example

Works for all experience levels.

## Interactive Playgrounds

Static docs insufficient. Developers learn by doing.

We built:
- Online query playground (runs in browser)
- Pre-loaded example datasets (coffee, meetings, projects)
- Instant feedback (parse errors, query results, performance metrics)

Result: 10x more engagement than static docs.

Learning by doing > learning by reading.

## Error Message Documentation

Users search error messages. Make them findable.

Every error documented with:
- **What**: "ParseError: Unknown keyword 'RECAL'"
- **Why**: "Parser doesn't recognize this keyword"
- **Fix**: "Did you mean 'RECALL'?"
- **Example**: `RECALL episode WHERE confidence > 0.7`

Search "RECAL error" → Find docs → Fix immediately.

Reduces support burden 70%.

## Bridging SQL to Graph

Most developers know SQL. Use it as bridge:

**SQL**: `SELECT * FROM memories WHERE confidence > 0.7`
**Engram**: `RECALL episode WHERE confidence > 0.7`

Similar syntax, different semantics. Familiar enough to try, different enough to teach.

**SQL**: `JOIN ... ON`
**Engram**: `SPREAD FROM`

Explicitly contrast: "Instead of JOINs (combining rows), use SPREAD (traversing graph)."

Anchor new concepts to existing knowledge.

## Takeaways

1. **Diátaxis framework**: Four doc types (tutorial, how-to, explanation, reference)
2. **Progressive disclosure**: Simple → Complex, build gradually
3. **Concrete examples**: Everyday analogies + specific numbers
4. **Interactive playgrounds**: Learning by doing > reading
5. **Error message docs**: Make searchable, include fixes
6. **Bridge from SQL**: Anchor to existing knowledge

Documentation is teaching. Optimize for how developers actually learn, not how you think they should.

---

Engram docs: /docs/reference/query-language.md, /docs/tutorials/
