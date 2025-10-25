# Container Orchestration and Deployment - Twitter Thread

**Tweet 1/8**

Deploying a sub-5ms graph database in containers without killing performance.

Default Docker config adds 200us network overhead + 50us I/O overhead + unpredictable CPU throttling.

Here's how we kept Engram's P50 latency at 3.2ms in production containers.

Thread:

---

**Tweet 2/8**

The network tax: Docker bridge mode routes every request through virtual ethernet + NAT.

Cost: +200 microseconds per request.

Solution: hostNetwork=true in Kubernetes. Skip the bridge entirely.

Result: +10us overhead instead. 20x improvement.

Tradeoff: Document port allocation clearly.

---

**Tweet 3/8**

The storage tax: Overlay filesystems do copy-on-write for every modification.

Cost: +50 microseconds per I/O operation. At 10K IOPS, that's 500ms of pure overhead per second.

Solution: hostPath volumes or local-path-provisioner.

Mount host directories directly. No overlay.

---

**Tweet 4/8**

The CPU throttling trap: Setting CPU limit != request causes unpredictable latency spikes.

Kubernetes CFS enforces 100ms periods. Use 300ms CPU in one period? Get throttled for 200ms.

Solution: Set limit = request, or omit limits entirely.

Let processes burst when idle cores available.

---

**Tweet 5/8**

Memory hierarchy matters in containers too.

Fast tier (active working set): tmpfs volume, RAM-backed, serves 80% of requests
Warm tier (frequent access): SSD persistent volume, 19% of requests
Cold tier (archival): S3-compatible object storage, 1% of requests

Right data in right place.

---

**Tweet 6/8**

The warmup problem: When containers start, graphs are cold. No cached patterns. No primed indexes.

Naive approach: Serve traffic immediately with 50-100ms latency until warm.

Better: Use readiness probes to delay traffic until cache hit rate >70%.

Users never see cold starts.

---

**Tweet 7/8**

StatefulSet vs Deployment for graph databases:

Deployment: Stateless, any pod can die
StatefulSet: Stable identity, ordered shutdown, persistent volumes

Graph state requires stable identity. Pod engram-0 always mounts same PVC.

Graceful shutdown needs 30s to flush memories. SIGTERM handling matters.

---

**Tweet 8/8**

Results after optimization:

P50: 3.2ms (bare metal: 3.0ms)
Container overhead: 6%
Image size: 15MB (multi-stage build)
Deployment time: 45 minutes first-time

Containers don't have to destroy performance.

But defaults will. Optimize the critical path.
