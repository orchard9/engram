# Operational Runbook - ${file}

Production deployment requires operational playbooks. Engram's distributed architecture introduces complexity - this runbook covers deployment, scaling, troubleshooting, recovery.

## Deployment Patterns
Single-node (development), multi-node with replication (production), multi-zone (high availability). Start with single-node, migrate to distributed by adding cluster config and seeds.

## Adding Nodes
1. Configure new node with cluster seeds
2. Node discovers peers via DNS/k8s API
3. Joins cluster via SWIM
4. Receives rebalancing assignments
5. Transfers assigned spaces
6. Begins serving traffic

Takes 5-15 minutes for production-size spaces.

## Handling Failures
Node failure detected by SWIM in <2s. Replicas automatically promoted for affected spaces. Rebalancing redistributes load. No manual intervention needed for single-node failures.

## Monitoring
Key metrics: cluster size, partition events, replication lag, query latency p50/p99, space distribution balance. Alert on: sustained partition, replication lag >5s, imbalance >20%.

## Troubleshooting
Split-brain: check vector clocks for concurrent updates. Slow queries: check space distribution and rebalancing status. High replication lag: check network saturation and disk I/O.
