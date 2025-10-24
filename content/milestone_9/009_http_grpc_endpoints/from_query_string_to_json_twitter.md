# Twitter Thread: From Query String to JSON

**Tweet 1/5**

API design question: Should queries be decomposed JSON or text strings?

Bad: `{"operation": "RECALL", "filters": [...]}`
Good: `{"query": "RECALL episode WHERE confidence > 0.7"}`

Here's why query-as-string wins for cognitive APIs.

---

**Tweet 2/5**

Decomposed JSON couples API to grammar.

Add new operation? Breaking API change.
Add new constraint? Breaking API change.

Query-as-string decouples them.

Parser is the compatibility layer, not API schema.

Same pattern SQL databases use. Works for 50 years.

---

**Tweet 3/5**

Multi-tenant routing: Memory space in header, not query.

```
X-Memory-Space: user_123
{"query": "RECALL episode"}
```

Why?
- Caching: By query text alone
- Security: Validate access at API layer
- Clarity: Query = intent, header = context

Clean separation.

---

**Tweet 4/5**

gRPC 2x faster than HTTP/JSON.

Benchmark (1000 queries):
- HTTP/JSON: 12ms P99
- gRPC/Protobuf: 6ms P99

Binary encoding, fewer steps, zero-copy.

Support both: HTTP for dev, gRPC for prod.

---

**Tweet 5/5**

Evidence as optional field:

With: +2ms latency, full explainability
Without: Faster, less debuggable

User choice. Prod skips evidence. Dev includes it.

Performance vs observability, configurable.

Code: https://github.com/engram-memory/engram

---

**Hashtags**: #APIDesign #gRPC #REST #SystemsDesign #Rust
