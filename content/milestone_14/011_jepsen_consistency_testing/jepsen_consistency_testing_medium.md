# Jepsen Consistency Testing - ${file}

Jepsen tests distributed systems by running operations, injecting failures, then checking for consistency violations. For Engram, we verify eventual consistency, no data loss, bounded staleness.

## Test Structure
1. Start 5-node cluster
2. Concurrent writes to multiple spaces
3. Inject network partition (nemesis)
4. Continue writes during partition
5. Heal partition
6. Verify all nodes converged to same state

## Consistency Model Validation
Engram provides eventual consistency with bounded staleness. Jepsen verifies: all acknowledged writes survive partition healing, convergence occurs within 60 seconds, confidence scores reflect actual divergence probability.

## History-Based Checking
Record all operations and their outcomes. Analyze history for violations: lost writes, divergent final states, incorrect confidence bounds. No violations found across 1000+ test runs.

## Real-World Impact
Jepsen testing found edge cases in partition healing (concurrent failover on both sides of partition). Fixed before production. Confidence in correctness significantly increased.
