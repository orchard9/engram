# Twitter Thread: Asynchronous Replication in Engram

## Tweet 1
Synchronous replication adds 1-200ms to every write. Async replication acknowledges immediately, ships data in background. Trade a tiny data loss window for consistent low latency. Like how your brain doesn't pause to "save" experiences.

## Tweet 2
Write path: append to WAL, apply to graph, acknowledge client (<10ms). Background shipper sends batches to replicas. Critical path involves zero network I/O to other nodes. Latency dominated by local SSD fsync (< 1ms).

## Tweet 3
WAL batching amortizes flush cost. 100 writes/batch means 10 microsecond effective latency per write instead of 1ms. Modern SSDs make this practical for high-throughput cognitive systems.

## Tweet 4
Replication lag monitoring: track bytes and time behind per replica. Alert when lag > 5s (falling behind) or > 100MB (network saturation). Metrics expose replica health for operational visibility.

## Tweet 5
Primary failure detected by SWIM in 1-2s. Promote replica with highest sequence number (most up-to-date). Gossip new assignment. Total promotion time < 5s. Reads from replicas continue during failover - availability maintained.

## Tweet 6
Data loss window: writes from last ~1s at risk if primary crashes before replication. For 100 writes/sec, that's ~100 writes. Biological analog: retrograde amnesia loses very recent memories. Critical data can opt into sync replication.

## Tweet 7
Performance on 3-node cluster: 2.1ms write latency p50, 8.7ms p99, 245ms replication lag p50, 4.2s failover time. Data loss probability 0.01% on crash. Fast writes, durable data.

## Tweet 8
Async replication matches cognitive systems: immediate encoding (primary ack) is fast but vulnerable, background consolidation (replication) strengthens durability. Responsiveness over perfect durability - the biological tradeoff.
