# HTTP API Cognitive Ergonomics Perspectives for Engram

## Cognitive-Architecture Perspective

The fundamental tension between REST's resource-oriented model and cognitive memory operations requires careful architectural consideration. Human memory doesn't operate through CRUD—it operates through encoding, consolidation, retrieval, and reconsolidation cycles. Our HTTP API must bridge this conceptual gap without sacrificing the cognitive fidelity that makes Engram unique.

Consider how REST's stateless constraint conflicts with memory's inherently stateful nature. When we retrieve a memory, we strengthen it through reconsolidation. A pure REST GET operation implies no state change, yet cognitive retrieval inherently modifies memory traces. We should embrace POST for retrieval operations: `POST /memories/recall` with retrieval cues in the request body. This violates REST orthodoxy but preserves cognitive accuracy.

The hierarchical nature of URL paths maps poorly to the associative structure of memory networks. Rather than forcing memories into `/categories/subcategories/items`, we should expose the graph structure directly: `/memories/{id}/associations?depth=2&activation_threshold=0.3`. This allows spreading activation patterns to emerge naturally through the API rather than being constrained by hierarchical thinking.

Memory consolidation presents unique challenges for synchronous HTTP. Rather than blocking on long-running consolidation, we should implement a two-phase commit pattern: immediate acknowledgment with a consolidation ticket, followed by asynchronous processing. The response includes both immediate storage confirmation and a consolidation timeline: `{"stored": true, "consolidation_id": "abc123", "phases": {"fast": "2s", "slow": "30m", "systems": "6h"}}`. Clients can poll or subscribe to consolidation events through Server-Sent Events.

The API should expose cognitive load indicators in response headers. Just as the brain has limited working memory capacity, our API should communicate processing constraints: `X-Cognitive-Load: 0.7`, `X-Working-Memory-Used: 5/7`, `X-Attention-Available: 0.3`. This helps developers understand when they're overwhelming the system's cognitive resources.

Pattern completion and imagination require endpoints that generate rather than just retrieve. A `POST /memories/imagine` endpoint accepts partial patterns and returns completed memories with confidence scores. Similarly, `/memories/dream` could trigger offline consolidation and reorganization processes, returning a stream of memory transformations.

Error states should reflect cognitive phenomena. Rather than generic 404s, return cognitive-specific errors: "Cannot retrieve: interference from similar memories", "Retrieval blocked: recent encoding not yet consolidated", "Pattern too ambiguous: multiple completions possible". These errors educate developers about memory dynamics while providing actionable feedback.

The API must handle the distinction between remembering (successful retrieval) and knowing (familiarity without details). Responses should include both recognition confidence and recall fidelity: `{"recognition": 0.95, "recall_fidelity": 0.4, "memory": {...}}`. This dual-process model helps developers understand when they have partial matches versus full retrievals.

## Memory-Systems Perspective

The distinction between episodic and semantic memory fundamentally shapes our API design. Episodic memories—rich, contextual, time-bound experiences—require different access patterns than semantic memories—abstract, consolidated knowledge. Our endpoints should reflect this: `/memories/episodic/replay` for re-experiencing specific events versus `/memories/semantic/query` for knowledge retrieval.

Spreading activation, the core mechanism of memory retrieval, demands a graph-native API approach. Traditional REST encourages thinking about individual resources, but memory operates through activation cascades. Our API should support activation queries: `POST /activation/spread` with parameters for initial activation strength, decay rate, and termination threshold. The response streams activated nodes as they cross threshold, mimicking the temporal dynamics of neural activation.

Memory confidence isn't binary—it exists on a continuum from vague familiarity to vivid recollection. Every response must include confidence metadata at multiple levels: item confidence, source confidence, and temporal confidence. This manifests as: `{"memory": {...}, "confidence": {"content": 0.9, "source": 0.6, "timing": 0.3}}`. Developers can then implement appropriate fallback strategies for low-confidence retrievals.

The API should model memory retrieval as a two-stage process: fast familiarity followed by slower recollection. Initial responses return high-confidence, immediately available memories with a continuation token for deeper retrieval: `{"immediate": [...], "continuation": "token_xyz", "estimated_additional": 15}`. Clients can then choose whether to wait for exhaustive retrieval or proceed with partial results.

Forgetting isn't deletion—it's accessibility reduction. The `/memories/forget` endpoint shouldn't remove data but should reduce activation weights and increase retrieval thresholds. The API should support both active suppression and passive decay: `POST /memories/suppress` for intentional forgetting and `POST /memories/decay` for time-based degradation. Responses indicate accessibility changes rather than deletion confirmation.

Context-dependent retrieval requires rich cue specification. Rather than simple keyword search, our API accepts multidimensional retrieval cues: `{"cue": {"content": "meeting", "context": {"mood": "anxious", "location": "office"}, "timeframe": "last_week"}}`. The response includes not just matching memories but also the specific cue-memory associations that triggered retrieval.

Memory consolidation states must be queryable. Developers need to know whether memories are in fast-changing hippocampal storage or stable neocortical storage. Include consolidation metadata: `{"consolidation_state": "systems", "stability": 0.8, "last_retrieval": "2024-01-15", "retrieval_count": 7}`. This helps predict memory permanence and modification resistance.

The API should support memory reconsolidation during retrieval. When memories are retrieved, they become temporarily labile. Provide an optional update window: `{"memory": {...}, "reconsolidation_window": "300s", "update_token": "xyz"}`. During this window, the memory can be modified with the token, mimicking biological reconsolidation.

## Rust-Graph-Engine Perspective

HTTP introduces substantial overhead for high-frequency graph operations. Every request involves serialization, network transmission, deserialization, and response marshaling. For Engram's spreading activation algorithms, which might touch thousands of nodes, we must minimize this overhead through careful protocol design.

Implement batch operations as first-class citizens. Rather than individual node queries, support graph traversal specifications: `POST /graph/traverse` with a traversal program expressed as a compact DSL. This reduces round-trips from O(n) to O(1) for complex graph operations. The DSL compiles to efficient Rust traversal code, maintaining type safety across the HTTP boundary.

Serialization costs demand careful format selection. While JSON provides excellent debugging ergonomics, consider Protocol Buffers or MessagePack for production workloads. Support content negotiation: `Accept: application/x-msgpack` for efficient binary encoding. For large graph responses, implement streaming with zero-copy serialization directly from graph memory to network buffers.

Type safety erodes at HTTP boundaries. Generate TypeScript and Python types from Rust structures using `serde` attributes. But go further: implement a graph schema endpoint `/schema/types` that returns runtime type information. This enables dynamic clients to validate operations before execution, catching type errors early.

Connection pooling and keep-alive become critical for graph workloads. A single user session might generate hundreds of graph operations. HTTP/2 multiplexing reduces connection overhead, but consider WebSocket upgrades for truly interactive sessions: `GET /graph/session` with `Upgrade: websocket`. This provides a bidirectional channel for streaming graph operations without HTTP overhead.

Memory efficiency requires careful response pagination. Graph queries can return massive result sets. Implement cursor-based pagination with size limits based on actual memory usage, not item count: `{"nodes": [...], "cursor": "...", "memory_used": "12MB", "memory_limit": "50MB"}`. This prevents response buffer exhaustion while maintaining predictable memory usage.

Zero-copy optimizations should extend through the stack. Use `bytes::Bytes` for request/response bodies, enabling reference-counted sharing without copies. For large graph exports, implement `sendfile` syscall optimization, directly streaming from mapped graph memory to the network socket, bypassing userspace entirely.

Cache graph computations aggressively. Spreading activation and pattern completion are expensive operations. Implement ETag support with graph version vectors: `ETag: "graph-v7-activation-h3x9"`. Include activation parameters in the cache key. For frequently accessed patterns, precompute and cache entire activation cascades.

Profile serialization hotspots relentlessly. What looks like a graph algorithm bottleneck might actually be JSON encoding. Use `cargo flamegraph` to identify serialization overhead. Consider custom serializers for hot paths—hand-rolled number formatting can be 10x faster than generic serialization for numeric-heavy graph data.

## Systems-Architecture Perspective

Rate limiting for cognitive operations requires nuanced strategies beyond simple request counting. Memory consolidation, spreading activation, and pattern completion have vastly different computational costs. Implement cost-based rate limiting where each operation consumes "cognitive tokens" proportional to its complexity: spreading activation might cost 10 tokens per depth level, while simple storage costs 1 token.

Caching strategies must respect memory dynamics. Unlike traditional APIs where GET results are indefinitely cacheable, memory retrieval changes memory state. Implement cache headers that reflect consolidation state: `Cache-Control: private, max-age=300, must-revalidate-after-retrieval`. After retrieval, the memory's accessibility has changed, invalidating prior cache entries. Use `Vary: X-Retrieval-Context` to ensure context-dependent retrieval doesn't serve inappropriate cached results.

Load balancing across Engram instances requires session affinity for memory coherence. A user's recent memories might exist only in fast storage on a specific instance. Implement consistent hashing based on user ID: requests for the same memory space route to the same instance. Include instance hints in responses: `X-Engram-Instance: engram-7.region-us-west.local`, allowing clients to prefer the same instance for related operations.

Long-running consolidation processes challenge HTTP's request-response model. Rather than WebSockets or long polling, implement a job queue pattern. Consolidation requests return immediately with a job ID. Status checks are cheap: `GET /jobs/{id}/status`. When complete, results are available at `/jobs/{id}/result` with a TTL. This pattern scales horizontally and survives connection drops.

Circuit breakers must distinguish between system overload and cognitive processing time. A spreading activation query might take 30 seconds not because the system is failing, but because activation is propagating through a dense graph region. Implement semantic circuit breakers: track operation types separately, with different timeout and failure thresholds for each cognitive operation class.

Memory pressure requires careful response streaming. Large graph retrievals can exhaust server memory if fully materialized. Implement response streaming with backpressure: write response chunks as graph traversal proceeds. If the client can't keep up, pause traversal rather than buffering indefinitely. Include progress indicators: `X-Progress: nodes=1500/5000, depth=3/5`.

Horizontal scaling demands distributed graph consistency. When memories span multiple instances, implement two-phase commit for multi-partition operations. Expose partition topology to clients: `X-Partition-Map: {"/memories/episodic": "shard-1", "/memories/semantic": "shard-2"}`. This allows sophisticated clients to optimize query patterns for partition locality.

Monitoring must track cognitive metrics beyond traditional RED (Rate, Errors, Duration) signals. Monitor spreading activation patterns, consolidation queue depths, memory interference rates, and pattern completion confidence distributions. Expose these through a `/metrics/cognitive` endpoint with Prometheus-compatible format. Alert on cognitive anomalies: sudden drops in retrieval confidence might indicate corruption before traditional errors appear.