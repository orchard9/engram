# API Quickstart: Your First 15 Minutes with Engram

Get from zero to first successful memory operation in 15 minutes. Research shows developers who succeed quickly are 3x more likely to adopt a technology in production.

This tutorial builds three "aha!" moments:

1. **"It just works"** - Store and retrieve your first memory

2. **"This is different"** - See how confidence scores work

3. **"This is powerful"** - Watch dream consolidation create insights

## Prerequisites

- Engram server running (see [Installation Guide](/guide/installation.md))

- curl or Python 3.8+ installed

- 15 minutes of focused time

## Part 1: Your First Memory (5 minutes)

### Goal

Store a memory and retrieve it. Experience the core remember/recall cycle.

### Step 1: Verify Engram is Running

```bash
# Check server health
curl http://localhost:8080/api/v1/introspect

# Should return JSON with "healthy": true

```

If this fails, start Engram:

```bash
engram start --grpc-port 50051 --http-port 8080

```

### Step 2: Store Your First Memory

Let's store a simple fact. Notice we include a **confidence score** - this is key to how Engram differs from traditional databases.

```bash
curl -X POST http://localhost:8080/api/v1/memories/remember \
  -H "Content-Type: application/json" \
  -d '{
    "memory": {
      "content": "The mitochondria is the powerhouse of the cell",
      "embedding": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
      "confidence": {
        "value": 0.95,
        "reasoning": "Verified biological fact from textbook"
      }
    },
    "auto_link": true
  }'

```

**Note**: In production, use proper 768-dimensional embeddings from models like `sentence-transformers`. This shortened example is for learning.

### Expected Response

```json
{
  "memory_id": "mem_1698765432_a1b2c3d4",
  "storage_confidence": {
    "value": 0.98,
    "category": "CONFIDENCE_CATEGORY_CERTAIN",
    "reasoning": "Successfully stored in hot tier"
  },
  "linked_memories": [],
  "initial_state": "CONSOLIDATION_STATE_RECENT"
}

```

**What just happened?**

- Engram stored your memory with ID `mem_1698765432_a1b2c3d4`

- Storage confidence (0.98) is even higher than your input confidence (0.95) - Engram is certain the write succeeded

- The memory starts in "recent" state - it'll be consolidated later

- No linked memories yet (your memory space is empty)

### Step 3: Retrieve Your Memory

Now let's recall what we just stored:

```bash
curl -X POST http://localhost:8080/api/v1/memories/recall \
  -H "Content-Type: application/json" \
  -d '{
    "cue": {
      "semantic": {
        "query": "cell biology"
      }
    },
    "max_results": 5
  }'

```

### Expected Response

```json
{
  "memories": [
    {
      "id": "mem_1698765432_a1b2c3d4",
      "content": "The mitochondria is the powerhouse of the cell",
      "activation": 0.87,
      "confidence": {
        "value": 0.95,
        "category": "CONFIDENCE_CATEGORY_CERTAIN"
      }
    }
  ],
  "recall_confidence": {
    "value": 0.89,
    "category": "CONFIDENCE_CATEGORY_HIGH",
    "reasoning": "Strong semantic match with query"
  },
  "metadata": {
    "total_activated": 1,
    "above_threshold": 1,
    "recall_time_ms": 12
  }
}

```

**Aha! Moment #1: "It just works"**

You stored a memory and retrieved it with a semantic query. The system understood "cell biology" relates to your mitochondria memory - even though the exact words don't match.

Notice two confidence scores:

- **Memory confidence** (0.95): How sure we are the fact is correct

- **Recall confidence** (0.89): How sure we are we found the right memories

This cognitive approach means you always know how certain your results are.

## Part 2: Understanding Confidence (5 minutes)

### Goal

See how confidence scores enable graceful degradation instead of hard failures.

### Step 4: Store Memories with Different Confidence Levels

Let's add three more memories with varying certainty:

```bash
# High confidence (verified fact)
curl -X POST http://localhost:8080/api/v1/memories/remember \
  -H "Content-Type: application/json" \
  -d '{
    "memory": {
      "content": "DNA contains genetic information",
      "embedding": [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
      "confidence": {"value": 0.95, "reasoning": "Fundamental biological fact"}
    }
  }'

# Medium confidence (heard from colleague)
curl -X POST http://localhost:8080/api/v1/memories/remember \
  -H "Content-Type: application/json" \
  -d '{
    "memory": {
      "content": "CRISPR might cure genetic diseases",
      "embedding": [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.1],
      "confidence": {"value": 0.6, "reasoning": "Promising research, not yet proven"}
    }
  }'

# Low confidence (vague memory)
curl -X POST http://localhost:8080/api/v1/memories/remember \
  -H "Content-Type: application/json" \
  -d '{
    "memory": {
      "content": "I think ribosomes do something with proteins",
      "embedding": [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.1, 0.2],
      "confidence": {"value": 0.4, "reasoning": "Uncertain memory from years ago"}
    }
  }'

```

### Step 5: Recall with Confidence Threshold

Now retrieve memories, but only high-confidence ones:

```bash
curl -X POST http://localhost:8080/api/v1/memories/recall \
  -H "Content-Type: application/json" \
  -d '{
    "cue": {
      "semantic": {
        "query": "biology facts"
      }
    },
    "max_results": 10,
    "confidence_threshold": 0.8
  }'

```

**Response**: You'll get only the high-confidence memories (mitochondria, DNA), not the uncertain ones.

Now try with a lower threshold:

```bash
curl -X POST http://localhost:8080/api/v1/memories/recall \
  -H "Content-Type: application/json" \
  -d '{
    "cue": {
      "semantic": {
        "query": "biology facts"
      }
    },
    "max_results": 10,
    "confidence_threshold": 0.3
  }'

```

**Response**: Now you'll get all memories, including the uncertain "I think ribosomes..." one.

**Aha! Moment #2: "This is different from SQL"**

Traditional databases give you everything or nothing. Engram gives you *gradated results* - choose how certain you need to be.

This is powerful for:

- **Research**: Lower threshold to explore possibilities

- **Production**: High threshold for reliable facts only

- **Learning**: See what you're uncertain about

## Part 3: Dream Consolidation (5 minutes)

### Goal

Watch Engram consolidate memories and discover connections, like sleep consolidates your memories.

### Step 6: Add More Memories

Let's add a few more memories so consolidation has material to work with:

```bash
# Add 5 more memories (abbreviated for brevity)
for i in {1..5}; do
  curl -X POST http://localhost:8080/api/v1/memories/remember \
    -H "Content-Type: application/json" \
    -d "{
      \"memory\": {
        \"content\": \"Biology fact $i about cells\",
        \"embedding\": [0.$i, 0.$((i+1)), 0.$((i+2)), 0.$((i+3)), 0.$((i+4)), 0.$((i+5)), 0.$((i+6)), 0.$((i+7))],
        \"confidence\": {\"value\": 0.8}
      }
    }"
done

```

### Step 7: Trigger Dream Consolidation

Now for the magic - trigger dream consolidation and watch memories get consolidated:

```bash
# Using Server-Sent Events (SSE) to stream consolidation
curl -X POST http://localhost:8080/api/v1/consolidation/dream \
  -H "Content-Type: application/json" \
  -H "Accept: text/event-stream" \
  -d '{
    "replay_cycles": 5,
    "creativity_factor": 0.7,
    "generate_insights": true
  }'

```

### Expected Stream

```
event: replay
data: {"memory_ids": ["mem_1698765432_a1b2c3d4", "mem_1698765001_x1y2z3"], "sequence_novelty": 0.45, "narrative": "Connection between mitochondria and cellular energy"}

event: progress
data: {"memories_replayed": 4, "new_connections": 2, "consolidation_strength": 0.67}

event: insight
data: {"description": "All biology facts relate to cellular function", "confidence": {"value": 0.78}, "suggested_action": "Create explicit 'cellular_biology' association"}

event: replay
data: {"memory_ids": ["mem_1698765001_x1y2z3", "mem_1698764890_p4q5r6"], "sequence_novelty": 0.82, "narrative": "Novel connection between DNA and protein synthesis"}

event: done
data: {"total_replayed": 8, "total_insights": 2, "total_connections": 5}

```

**Aha! Moment #3: "This is like sleep!"**

You just watched Engram:

1. Replay memories in sequences

2. Find connections between related concepts

3. Generate insights you didn't explicitly program

4. Strengthen associations (like sleep strengthens learning)

This consolidation:

- Makes future recalls faster (spreading activation uses these connections)

- Discovers patterns you didn't explicitly encode

- Mirrors how biological memory actually works

## What You Just Learned

In 15 minutes, you:

1. **Stored and retrieved** memories with semantic queries

2. **Used confidence scores** for graceful degradation

3. **Watched consolidation** create emergent connections

### Key Differences from Traditional Databases

| Traditional Database | Engram |
|---------------------|---------|
| Exact match or nothing | Gradated confidence scores |
| No notion of uncertainty | Confidence on every operation |
| Static relationships | Dynamic spreading activation |
| No emergent behavior | Consolidation discovers patterns |

## Next Steps

### Dive Deeper

- **REST API**: [Complete REST API Reference](/reference/rest-api.md)

- **gRPC API**: [High-Performance gRPC Guide](/reference/grpc-api.md)

- **Error Handling**: [Error Codes Catalog](/reference/error-codes.md)

### Common Use Cases

- **Semantic Search**: [Embedding-Based Retrieval Tutorial](/tutorials/semantic-search.md)

- **Episodic Memory**: [Recording Events with Context](/tutorials/episodic-memory.md)

- **Knowledge Graphs**: [Building Interconnected Knowledge](/tutorials/knowledge-graphs.md)

### Production Deployment

- **Authentication**: [API Keys and JWT](/operations/authentication.md)

- **Performance**: [Tuning for High Throughput](/operations/performance-tuning.md)

- **Monitoring**: [Health Checks and Metrics](/operations/monitoring.md)

## Troubleshooting

### "I can't connect to the server"

```bash
# Check if Engram is running
curl http://localhost:8080/api/v1/introspect

# If not, start it
engram start --grpc-port 50051 --http-port 8080

# Check logs
journalctl -u engram -f

```

### "My recall returns nothing"

- Lower your confidence threshold: `"confidence_threshold": 0.3`

- Check if memories were actually stored: `curl http://localhost:8080/api/v1/introspect | jq '.statistics.total_memories'`

- Verify your embedding similarity: Use same embedding model for store and recall

### "Embeddings are confusing"

For this tutorial, we used short 8-dimensional vectors for simplicity. In production:

```python
# Install sentence-transformers
pip install sentence-transformers

# Generate proper 768-dimensional embeddings
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-mpnet-base-v2')
embedding = model.encode("The mitochondria is the powerhouse of the cell")
# embedding.shape: (768,) - full-dimensional vector

```

### "Where do I get API keys?"

```bash
# Generate API key
engram auth create-key --name "my-app"

# Output: ek_live_1234567890abcdef

# Use in requests
curl -H "Authorization: Bearer ek_live_1234567890abcdef" \
  http://localhost:8080/api/v1/memories/recall

```

## Real-World Example: Personal Knowledge Assistant

Here's a complete example building a personal knowledge assistant:

```python
#!/usr/bin/env python3
"""Personal Knowledge Assistant - Store and recall information with confidence."""

import requests
from sentence_transformers import SentenceTransformer

# Initialize embedding model
model = SentenceTransformer('all-mpnet-base-v2')

class KnowledgeAssistant:
    def __init__(self, base_url="http://localhost:8080/api/v1"):
        self.base_url = base_url

    def learn(self, fact, confidence=0.8, source="manual_entry"):
        """Store a new fact."""
        embedding = model.encode(fact).tolist()

        response = requests.post(f"{self.base_url}/memories/remember", json={
            "memory": {
                "content": fact,
                "embedding": embedding,
                "confidence": {
                    "value": confidence,
                    "reasoning": f"From {source}"
                }
            },
            "auto_link": True,
            "link_threshold": 0.7
        })

        result = response.json()
        print(f"Learned: {fact}")
        print(f"  Confidence: {result['storage_confidence']['value']:.2f}")
        print(f"  Linked to {len(result.get('linked_memories', []))} existing memories")

        return result['memory_id']

    def recall(self, question, min_confidence=0.6):
        """Recall relevant facts."""
        response = requests.post(f"{self.base_url}/memories/recall", json={
            "cue": {
                "semantic": {
                    "query": question,
                    "fuzzy_threshold": 0.5
                }
            },
            "max_results": 5,
            "confidence_threshold": min_confidence
        })

        result = response.json()
        memories = result['memories']

        print(f"Question: {question}")
        print(f"Recall confidence: {result['recall_confidence']['value']:.2f}")
        print(f"\nRelevant facts ({len(memories)}):")

        for mem in memories:
            print(f"  [{mem['confidence']['value']:.2f}] {mem['content']}")

        return memories

# Use the assistant
assistant = KnowledgeAssistant()

# Learn some facts
assistant.learn("Python was created by Guido van Rossum in 1991", confidence=0.95, source="Wikipedia")
assistant.learn("Python emphasizes code readability", confidence=0.90, source="Official docs")
assistant.learn("Django is a Python web framework", confidence=0.85, source="Tutorial")

# Ask questions
assistant.recall("Who created Python?")
assistant.recall("What makes Python special?")
assistant.recall("Python web development")

```

Run this and you'll see:

```
Learned: Python was created by Guido van Rossum in 1991
  Confidence: 0.98
  Linked to 0 existing memories

Learned: Python emphasizes code readability
  Confidence: 0.97
  Linked to 1 existing memories

Learned: Django is a Python web framework
  Confidence: 0.96
  Linked to 2 existing memories

Question: Who created Python?
Recall confidence: 0.92

Relevant facts (1):
  [0.95] Python was created by Guido van Rossum in 1991

Question: What makes Python special?
Recall confidence: 0.88

Relevant facts (1):
  [0.90] Python emphasizes code readability

Question: Python web development
Recall confidence: 0.85

Relevant facts (2):
  [0.90] Python emphasizes code readability
  [0.85] Django is a Python web framework

```

## Congratulations

You've completed the API quickstart. You now understand:

- How to store memories with confidence

- How recall works with semantic queries

- How consolidation creates emergent connections

- How this differs from traditional databases

**Time spent**: ~15 minutes
**Skills gained**: Core Engram operations
**Next milestone**: Build a production application

**Share your experience**: What "aha!" moment resonated most? Let us know in [GitHub Discussions](https://github.com/engram/engram/discussions).
