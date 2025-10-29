# Example 02: Episodic Memory

**Learning Goal**: Store and retrieve rich contextual memories with temporal, spatial, and emotional dimensions.

**Difficulty**: Intermediate
**Time**: 15 minutes
**Prerequisites**: Completed Example 01

## Cognitive Concept

Episodic memory captures the "what, when, where, who, and how it felt" of experiences:

```
EXPERIENCE = {
  what: content/action
  when: timestamp/duration
  where: location/context
  who: agents involved
  emotion: valence/arousal
}
```

Unlike semantic memory (timeless facts), episodic memories are anchored in specific moments and decay faster without rehearsal.

## What You'll Learn

- Store episodic memories with `experience()`
- Query by temporal context with `reminisce()`
- Understand temporal decay patterns
- See emotional context in recall

## Example Use Cases

- Personal assistant: "What did I work on last Tuesday?"
- Event timeline: "Show me interactions with this person"
- Context retrieval: "What was I thinking when I made this decision?"

## Code Examples

See language-specific implementations in this directory:

- `rust.rs` - Rust implementation using engram-client
- `python.py` - Python with asyncio
- `typescript.ts` - TypeScript with Promises
- `go.go` - Go with goroutines
- `java.java` - Java with CompletableFuture

## Next Steps

After mastering episodic memory, explore:
- [03-streaming-operations](../03-streaming-operations/) - Real-time memory flows
- [04-pattern-completion](../04-pattern-completion/) - Predictive memory completion
