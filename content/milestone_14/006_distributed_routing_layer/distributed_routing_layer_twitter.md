# Twitter Thread: Distributed Routing in Engram

## Tweet 1
Three routing approaches: client-side (clients know topology), proxy-based (centralized forwarder), peer-to-peer (any node can route). Engram uses P2P - no central coordinator, distributed knowledge, robust to failures. Like brain regions routing information without a dispatcher.

## Tweet 2
Routing table maps spaces to primaries and replicas. Propagates via gossip in O(log N) rounds. Eventually consistent means temporary staleness possible. Solution: self-correcting forwarding. Wrong node responds "try Node C instead, version 42". One extra hop, then fixed.

## Tweet 3
Writes must go to primary (only primary mutates). Reads can hit any replica (load distribution). Prefer local execution when possible - if current node owns space, zero network hops. With balanced load, ~10% of queries execute locally.

## Tweet 4
Connection pooling eliminates TCP handshake overhead. Maintain one gRPC channel per node, reuse across requests. HTTP/2 multiplexing means one connection handles hundreds of concurrent requests. Steady-state routing overhead = serialization plus network latency only.

## Tweet 5
Retry logic with exponential backoff prevents thundering herds. Retriable errors (timeout, connection refused) trigger backoff retries (10ms, 20ms, 40ms). Non-retriable errors (bad request) fail fast. Self-healing through retries.

## Tweet 6
Benchmarks on 10-node cluster: 35ns routing table lookup (DashMap), 1.3ms remote routing overhead (dominated by network RTT), 99.8% connection pool hit rate. Local execution 0.5ms p50, remote 1.8ms p50, stale routing 3.2ms p50 (self-corrects).

## Tweet 7
Stale routing measured at <0.1% of requests in steady state. Primarily occurs during rebalancing or failover windows. Gossip propagation keeps tables fresh. Self-correction ensures eventual success regardless of staleness.

## Tweet 8
Biological parallel: brain regions route information without central dispatch. Visual cortex processes locally when possible, forwards to motor cortex when needed. Each region knows neighbors and targets. Engram mirrors this distributed routing architecture.
