# Multi-Language API Examples

Complete runnable examples for Engram's cognitive memory operations, organized by use case rather than language. Each example demonstrates a specific pattern or operation with implementations in Rust, Python, TypeScript, Go, and Java.

## Philosophy: Learn by Operation, Not by Language

Instead of showing "here's how to use Engram in Python," we show "here's how to handle errors" with examples in all five languages. This helps you understand the cognitive pattern first, then see it implemented idiomatically in your language.

## Quick Start

1. **Choose an operation** that matches your use case

2. **Read the README** in that operation's directory

3. **Pick your language** and run the example

4. **Understand the pattern**, then adapt to your needs

## Example Categories

### [01-basic-remember-recall](./01-basic-remember-recall/)

**Learning Goal**: Master the fundamental remember/recall cycle

The "Hello World" of Engram. Store a memory and retrieve it.

**Key Concepts**:

- Confidence scoring on storage

- Semantic retrieval cues

- Understanding activation vs confidence

**When to use**:

- First time using Engram

- Building CRUD operations

- Testing connectivity

**Languages**: Rust | Python | TypeScript | Go | Java
**Difficulty**: Beginner
**Time**: 10 minutes

---

### [02-episodic-memory](./02-episodic-memory/)

**Learning Goal**: Record rich contextual memories with what/when/where/who/why/how

Store experiences with full episodic context, then query by temporal or spatial cues.

**Key Concepts**:

- Episodic encoding structure

- Context-based retrieval

- Emotional valence tracking

**When to use**:

- Event logging systems

- Personal memory assistants

- Timeline reconstruction

**Languages**: Rust | Python | TypeScript | Go | Java
**Difficulty**: Intermediate
**Time**: 20 minutes

---

### [03-streaming-operations](./03-streaming-operations/)

**Learning Goal**: Use gRPC streaming for high-throughput and real-time operations

Demonstrates server streaming (dream consolidation), client streaming (batch uploads), and bidirectional streaming (interactive sessions).

**Key Concepts**:

- Server-sent events vs gRPC streaming

- Backpressure management

- Flow control

**When to use**:

- Batch imports (>100 memories)

- Real-time consolidation monitoring

- Interactive memory sessions

**Languages**: Rust | Python | TypeScript | Go | Java
**Difficulty**: Advanced
**Time**: 45 minutes

---

### [04-pattern-completion](./04-pattern-completion/)

**Learning Goal**: Reconstruct full memories from partial cues

Like tip-of-the-tongue experiences - you know part of a memory and want to complete it.

**Key Concepts**:

- Pattern cues vs semantic cues

- Creativity vs accuracy tradeoff

- Field-level confidence

**When to use**:

- Autocomplete features

- Memory reconstruction

- Fuzzy matching

**Languages**: Rust | Python | TypeScript | Go | Java
**Difficulty**: Intermediate
**Time**: 25 minutes

---

### [05-error-handling](./05-error-handling/)

**Learning Goal**: Handle errors gracefully with educational error messages

Demonstrates retry logic, graceful degradation, and interpreting Engram's educational error format.

**Key Concepts**:

- Retriable vs non-retriable errors

- Progressive confidence thresholds

- Educational error messages

**When to use**:

- Production resilience

- Debugging failed operations

- Understanding error semantics

**Languages**: Rust | Python | TypeScript | Go | Java
**Difficulty**: Intermediate
**Time**: 30 minutes

---

### [06-authentication](./06-authentication/)

**Learning Goal**: Secure API access with JWT and API keys

**Key Concepts**:

- API key generation

- JWT token authentication

- Multi-tenant memory space isolation

**When to use**:

- Production deployments

- Multi-tenant applications

- Secure API access

**Languages**: Rust | Python | TypeScript | Go | Java
**Difficulty**: Intermediate
**Time**: 20 minutes

---

### [07-performance-optimization](./07-performance-optimization/)

**Learning Goal**: Achieve maximum throughput and minimum latency

**Key Concepts**:

- Connection pooling

- Batch operations

- Streaming vs unary calls

- Caching strategies

**When to use**:

- High-throughput requirements (>100 req/sec)

- Latency-sensitive applications

- Optimizing production deployments

**Languages**: Rust | Python | TypeScript | Go | Java
**Difficulty**: Advanced
**Time**: 45 minutes

---

### [08-multi-tenant](./08-multi-tenant/)

**Learning Goal**: Isolate memories across tenants with memory spaces

**Key Concepts**:

- Memory space creation

- Tenant-scoped authentication

- Cross-space queries (admin only)

**When to use**:

- SaaS applications

- Multi-customer deployments

- Isolated data requirements

**Languages**: Rust | Python | TypeScript | Go | Java
**Difficulty**: Intermediate
**Time**: 25 minutes

---

### [09-batch-operations](./09-batch-operations/)

**Learning Goal**: Efficiently process large volumes of memories

**Key Concepts**:

- Streaming remember for bulk uploads

- Pagination for large recalls

- Rate limiting strategies

**When to use**:

- Data imports

- Migrations

- Bulk processing

**Languages**: Rust | Python | TypeScript | Go | Java
**Difficulty**: Intermediate
**Time**: 30 minutes

---

### [10-migration-examples](./10-migration-examples/)

**Learning Goal**: Migrate from traditional databases to Engram

**Key Concepts**:

- Neo4j graph migration

- PostgreSQL table migration

- Generating embeddings from existing data

**When to use**:

- Moving from Neo4j

- Upgrading from SQL databases

- Hybrid deployments

**Languages**: Rust | Python | TypeScript | Go | Java
**Difficulty**: Advanced
**Time**: 60 minutes

---

## Running Examples

### Prerequisites

Each language has specific requirements. See individual example READMEs for setup.

**Common requirements**:

- Engram server running on `localhost:50051` (gRPC) or `localhost:8080` (REST)

- Embedding model for production examples (test examples use simplified embeddings)

**Quick server start**:

```bash
engram start --grpc-port 50051 --http-port 8080

```

### Example Structure

Each example directory contains:

```
01-basic-remember-recall/
├── README.md              # Operation overview and learning goals
├── rust.rs                # Rust implementation
├── python.py              # Python implementation
├── typescript.ts          # TypeScript implementation
├── go.go                  # Go implementation
├── java.java              # Java implementation
├── expected_output.txt    # What successful run looks like
└── common_errors.md       # Troubleshooting guide

```

### Quick Run Commands

**Python**:

```bash
cd 01-basic-remember-recall
pip install -r requirements.txt
python python.py

```

**Rust**:

```bash
cd 01-basic-remember-recall
cargo run --release

```

**TypeScript**:

```bash
cd 01-basic-remember-recall
npm install
npx ts-node typescript.ts

```

**Go**:

```bash
cd 01-basic-remember-recall
go run go.go

```

**Java**:

```bash
cd 01-basic-remember-recall
mvn compile exec:java

```

## Learning Path

### For Beginners

1. Start with [01-basic-remember-recall](./01-basic-remember-recall/) - understand core concepts

2. Try [02-episodic-memory](./02-episodic-memory/) - learn contextual encoding

3. Explore [05-error-handling](./05-error-handling/) - build resilience

### For Production Deployments

1. Review [06-authentication](./06-authentication/) - secure your API

2. Study [07-performance-optimization](./07-performance-optimization/) - maximize throughput

3. Implement [08-multi-tenant](./08-multi-tenant/) - isolate customer data

### For Advanced Users

1. Master [03-streaming-operations](./03-streaming-operations/) - bidirectional communication

2. Deep-dive [04-pattern-completion](./04-pattern-completion/) - cognitive pattern matching

3. Execute [10-migration-examples](./10-migration-examples/) - migrate existing systems

## Cognitive Patterns Reference

### When to Use Each Operation

| Use Case | Operation | Example Directory |
|----------|-----------|-------------------|
| Store new facts | Remember | 01-basic-remember-recall |
| Semantic search | Recall with semantic cue | 01-basic-remember-recall |
| Event logging | Experience | 02-episodic-memory |
| Timeline queries | Reminisce | 02-episodic-memory |
| Autocomplete | Pattern completion | 04-pattern-completion |
| Insight generation | Dream consolidation | 03-streaming-operations |
| Bulk import | Streaming remember | 09-batch-operations |
| Real-time monitoring | Stream events | 03-streaming-operations |

### Performance Characteristics

| Operation | Latency (P99) | Throughput | Best Transport |
|-----------|---------------|------------|----------------|
| Remember (single) | 45ms | 80/sec | REST |
| Remember (batch) | 180ms for 100 | 390/sec | gRPC streaming |
| Recall (semantic) | 120ms | 45/sec | REST |
| Dream (stream) | 8ms/event | N/A | gRPC streaming |
| Consolidate | 4.2s for 1000 | N/A | REST |

## Common Gotchas

### Embedding Dimensions

All examples use 8-dimensional embeddings for simplicity. Production requires 768-dimensional vectors:

```python
# Test/tutorial (simplified)
embedding = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]  # 8-dim

# Production (realistic)
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('all-mpnet-base-v2')
embedding = model.encode("text here")  # 768-dim

```

### Confidence is Always Required

Unlike traditional databases, Engram requires confidence on every memory:

```python
# WRONG - will fail
memory = Memory(content="fact", embedding=[...])

# CORRECT - confidence required
memory = Memory(
    content="fact",
    embedding=[...],
    confidence=Confidence(value=0.8, reasoning="Source: Wikipedia")
)

```

### REST vs gRPC Performance

For <10 requests/sec, either works. For >100 requests/sec, use gRPC:

```
Single requests: REST is fine (simpler)
Batch operations: gRPC 4-5x faster
Streaming: gRPC only option

```

## IDE Integration

### VS Code

Each example directory includes `.vscode/launch.json` for one-click debugging:

```bash
# Open example in VS Code
code 01-basic-remember-recall

# Press F5 to run with debugging

```

### IntelliJ IDEA

Java examples include `.idea/` configuration for direct execution.

### Cursor / Claude Code

Examples are optimized for AI-assisted learning with clear comments and cognitive explanations.

## Contributing Examples

Want to add an example? Follow this template:

1. **Choose an operation** not yet covered

2. **Write README** explaining cognitive pattern

3. **Implement in 5 languages** (Rust, Python, TypeScript, Go, Java)

4. **Include expected output** and common errors

5. **Test with fresh environment** (15-minute rule)

6. **Submit PR** with runnable examples

See [Contributing Guide](../../../CONTRIBUTING.md) for details.

## Support

- **Documentation**: [REST API](/reference/rest-api.md) | [gRPC API](/reference/grpc-api.md)

- **Errors**: [Error Codes Catalog](/reference/error-codes.md)

- **Tutorial**: [15-Minute Quickstart](/tutorials/api-quickstart.md)

- **Operations**: [Production Guide](/operations/)

## Example Quality Standards

Every example in this directory:

- ✅ Runs successfully on clean Engram installation

- ✅ Completes within stated time estimate

- ✅ Includes educational comments explaining cognitive concepts

- ✅ Handles errors gracefully with remediation hints

- ✅ Uses idiomatic patterns for each language

- ✅ Demonstrates one clear concept (not kitchen sink)

- ✅ Includes expected output for verification

## Next Steps

1. **Start learning**: Pick [01-basic-remember-recall](./01-basic-remember-recall/)

2. **Explore patterns**: Browse by use case, not language

3. **Build production**: Study [07-performance-optimization](./07-performance-optimization/)

4. **Get help**: Check [Error Codes](/reference/error-codes.md) if stuck

**Time investment**: 2-4 hours to try all examples
**Reward**: Deep understanding of cognitive memory operations
**Outcome**: Production-ready Engram integration
