# Network Partition Testing - ${file}

Testing distributed systems requires injecting failures. Network simulator controls message delivery - delay, drop, partition. Deterministic replay enables debugging distributed race conditions.

## Test Scenarios
Clean split (two halves can't communicate), asymmetric partition (A→B works, B→A fails), flapping (partition heals and reforms), cascading failures (nodes fail sequentially).

## Invariants Verified
No data loss (all acknowledged writes survive), bounded staleness (confidence reflects actual staleness), convergence (all nodes agree after healing).

## Chaos Engineering
Run production-like workload with random partition injection. Measure availability, latency, data integrity under adverse conditions. Jepsen-style validation at scale.

## Deterministic Replay
Capture message traces, replay with same ordering. Turns non-deterministic distributed bugs into reproducible test failures. Critical for debugging complex partition scenarios.
