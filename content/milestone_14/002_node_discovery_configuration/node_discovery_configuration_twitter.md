# Twitter Thread: Node Discovery in Distributed Systems

## Tweet 1 (Hook)
The chicken-and-egg problem of distributed systems: nodes need to coordinate before they can coordinate. How does a new Engram instance find existing cluster members when it doesn't know who they are?

## Tweet 2 (Problem Space)
Traditional approaches: hardcode IPs (brittle), broadcast on local network (doesn't scale), use external coordinator (adds complexity). Each works in specific environments but fails in others. We need something universal.

## Tweet 3 (Three Approaches)
Three discovery strategies, each with tradeoffs: Static seeds (reliable, manual), DNS SRV records (cloud-native, can be stale), Cloud provider APIs (fully dynamic, needs auth). Pick one? No - support all three.

## Tweet 4 (Hybrid Solution)
Engram tries discovery methods in priority order: static seeds first (instant), DNS second (fast), cloud API third (comprehensive). First success wins. Fallback provides robustness across environments.

## Tweet 5 (Configuration Philosophy)
Single-node by default, distributed opt-in. Minimal config runs standalone. Add cluster section to enable distributed mode. Environment variables override for Kubernetes. Same binary works everywhere.

## Tweet 6 (Join Protocol)
Discovery finds candidates. Join protocol creates membership: contact seed, receive cluster snapshot, announce via gossip. Within O(log N) rounds, all nodes know about the new member. Bootstrapping complete.

## Tweet 7 (Observability)
Discovery failures are painful to debug. HTTP endpoint shows exactly what happened: which methods tried, which succeeded, how many peers found, error details. Metrics track success rate over time.

## Tweet 8 (Testing Strategy)
Testing requires mocking external systems. DNS test server, Kubernetes API mock, docker-compose integration tests. Each layer tested independently, then combined. Discovery becomes reliable infrastructure.
