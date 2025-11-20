# Engram Cluster Verification Cookbook

Practical scenarios for verifying Engram's distributed cluster functionality using Docker Compose and Kubernetes (Kind).

## Prerequisites

- Docker 24.0+ with BuildKit enabled
- docker-compose v2.20+
- kind v0.20+ (for Kubernetes testing)
- kubectl v1.28+

## Scenario 1: Docker Compose 3-Node Cluster

### Build and Start

```bash
cd deployments/docker/cluster

# Build the Engram image (takes ~3 minutes first time)
docker compose build

# Start the 3-node cluster
docker compose up -d

# Wait for cluster convergence (15-30 seconds)
sleep 30

# Check all containers are running
docker compose ps
```

Expected output:
```
NAME            IMAGE                  STATUS         PORTS
engram-node1    engram/local:latest    Up (healthy)   0.0.0.0:7432->7432/tcp, 0.0.0.0:50051->50051/tcp
engram-node2    engram/local:latest    Up (healthy)   0.0.0.0:7433->7432/tcp, 0.0.0.0:50052->50051/tcp
engram-node3    engram/local:latest    Up (healthy)   0.0.0.0:7434->7432/tcp, 0.0.0.0:50053->50051/tcp
```

### Verify SWIM Membership Convergence

Each node should see 2 remote peers (the other nodes in the cluster):

```bash
# Check node1 membership
docker exec engram-node1 /usr/local/bin/engram status

# Check node2 membership
docker exec engram-node2 /usr/local/bin/engram status

# Check node3 membership
docker exec engram-node3 /usr/local/bin/engram status
```

Expected: Each node shows cluster health with:
- `alive: 2` (two other nodes)
- `suspect: 0`
- `dead: 0`

### Test Memory Replication

Create a memory on node1 and verify it replicates:

```bash
# Create memory on node1
curl -X POST http://localhost:7432/api/v1/memory \
  -H "Content-Type: application/json" \
  -d '{
    "content": "Test memory for cluster replication",
    "memory_space": "default",
    "tags": ["cluster-test"]
  }'

# Query from node2 (should see the memory)
curl http://localhost:7433/api/v1/memory?space=default

# Query from node3 (should see the memory)
curl http://localhost:7434/api/v1/memory?space=default
```

Expected: All three nodes return the same memory.

### Test Node Failure and Recovery

```bash
# Stop node2 to simulate failure
docker compose stop engram-node2

# Wait for failure detection (2-3 seconds)
sleep 3

# Check node1 membership - should mark node2 as dead
docker exec engram-node1 /usr/local/bin/engram status

# Restart node2
docker compose start engram-node2

# Wait for rejoin (5-10 seconds)
sleep 10

# Verify node2 rejoined - should be alive again
docker exec engram-node1 /usr/local/bin/engram status
```

Expected: Node2 marked as `dead: 1` after stop, then returns to `alive: 2` after restart.

### Test Space Assignment and Routing

```bash
# Create multiple memory spaces to test consistent hashing
for i in {1..10}; do
  curl -X POST http://localhost:7432/api/v1/memory \
    -H "Content-Type: application/json" \
    -d "{
      \"content\": \"Memory in space $i\",
      \"memory_space\": \"space-$i\",
      \"tags\": [\"routing-test\"]
    }"
done

# Check which node owns each space via logs
docker compose logs engram-node1 | grep "space assignment"
docker compose logs engram-node2 | grep "space assignment"
docker compose logs engram-node3 | grep "space assignment"
```

Expected: Spaces distributed across nodes using jump-consistent hashing.

### Monitor with Prometheus and Grafana

```bash
# Start monitoring stack
docker compose --profile monitoring up -d

# Wait for Prometheus and Grafana to start
sleep 15

# Access Grafana
open http://localhost:3000
# Login: admin/admin (or GRAFANA_PASSWORD from .env)

# Access Prometheus
open http://localhost:9090
```

Check metrics:
- `engram_cluster_members_alive` should be 2 for each node
- `engram_cluster_members_dead` should be 0
- `engram_replication_lag_seconds` should be <1s

### Cleanup

```bash
# Stop all services
docker compose --profile monitoring down

# Remove volumes (warning: deletes all data)
docker compose down -v
```

## Scenario 2: Kubernetes with Kind

### Create Kind Cluster

```bash
cd deployments/kubernetes

# Create 3-node Kind cluster
kind create cluster --config kind-cluster.yaml

# Verify cluster nodes
kubectl get nodes
```

Expected output:
```
NAME                        STATUS   ROLES           AGE   VERSION
engram-test-control-plane   Ready    control-plane   1m    v1.28.0
engram-test-worker          Ready    <none>          1m    v1.28.0
engram-test-worker2         Ready    <none>          1m    v1.28.0
```

### Build and Load Engram Image

```bash
# Build the Engram image
cd ../../
docker build -f deployments/docker/Dockerfile -t engram/engram:latest .

# Load image into Kind
kind load docker-image engram/engram:latest --name engram-test
```

### Deploy Engram Cluster

```bash
cd deployments/kubernetes

# Apply the cluster manifest
kubectl apply -f engram-cluster.yaml

# Watch pod startup
kubectl get pods -n engram-cluster -w
```

Expected: 3 pods named `engram-0`, `engram-1`, `engram-2` reach Running status.

### Verify DNS-Based Discovery

```bash
# Check DNS resolution of headless service
kubectl run -n engram-cluster test-dns --image=busybox:1.36 --rm -it --restart=Never -- \
  nslookup engram-headless.engram-cluster.svc.cluster.local
```

Expected: Shows 3 A records (one per pod).

### Verify SWIM Membership via Pod Logs

```bash
# Check engram-0 logs
kubectl logs -n engram-cluster engram-0 | grep -i "swim\|membership\|peer"

# Check engram-1 logs
kubectl logs -n engram-cluster engram-1 | grep -i "swim\|membership\|peer"

# Check engram-2 logs
kubectl logs -n engram-cluster engram-2 | grep -i "swim\|membership\|peer"
```

Expected: Logs show peer discovery and SWIM gossip messages.

### Test via NodePort Service

```bash
# Get NodePort
kubectl get svc -n engram-cluster engram-http

# Access via Kind host
curl http://localhost:7432/api/v1/system/health

# Create test memory
curl -X POST http://localhost:7432/api/v1/memory \
  -H "Content-Type: application/json" \
  -d '{
    "content": "Kubernetes cluster test",
    "memory_space": "default",
    "tags": ["k8s-test"]
  }'

# Verify health shows cluster info
curl http://localhost:7432/api/v1/system/health | jq '.cluster'
```

Expected: Health endpoint shows cluster node count and membership.

### Test Pod Deletion (Self-Healing)

```bash
# Delete one pod
kubectl delete pod -n engram-cluster engram-1

# StatefulSet automatically recreates it
kubectl get pods -n engram-cluster -w

# After new pod is Running, verify it rejoined cluster
kubectl logs -n engram-cluster engram-1 | grep "joined cluster"
```

Expected: New `engram-1` pod rejoins cluster automatically via DNS discovery.

### Scale the Cluster

```bash
# Scale to 5 replicas
kubectl scale statefulset -n engram-cluster engram --replicas=5

# Watch new pods start
kubectl get pods -n engram-cluster -w

# Verify all 5 nodes see each other
for i in {0..4}; do
  kubectl exec -n engram-cluster engram-$i -- /usr/local/bin/engram status
done
```

Expected: Each node shows `alive: 4` (four other peers).

### Cleanup

```bash
# Delete namespace
kubectl delete namespace engram-cluster

# Delete Kind cluster
kind delete cluster --name engram-test
```

## Scenario 3: Load Testing Cluster Behavior

### Docker Compose Load Test

```bash
# Start cluster
cd deployments/docker/cluster
docker compose up -d
sleep 30

# Run load test with multiple concurrent writers
for i in {1..100}; do
  curl -X POST http://localhost:7432/api/v1/memory \
    -H "Content-Type: application/json" \
    -d "{\"content\": \"Load test memory $i\", \"memory_space\": \"loadtest\"}" &
done
wait

# Verify all memories replicated to all nodes
for port in 7432 7433 7434; do
  echo "Node on port $port:"
  curl "http://localhost:$port/api/v1/memory?space=loadtest" | jq 'length'
done
```

Expected: All three nodes report same memory count.

### Kubernetes Load Test

```bash
# Deploy cluster
kubectl apply -f deployments/kubernetes/engram-cluster.yaml
kubectl wait --for=condition=ready pod -n engram-cluster -l app=engram --timeout=120s

# Port-forward to access cluster
kubectl port-forward -n engram-cluster svc/engram-http 7432:7432 &
PF_PID=$!

# Run load test
for i in {1..100}; do
  curl -X POST http://localhost:7432/api/v1/memory \
    -H "Content-Type: application/json" \
    -d "{\"content\": \"K8s load test memory $i\", \"memory_space\": \"k8s-loadtest\"}" &
done
wait

# Verify replication
curl "http://localhost:7432/api/v1/memory?space=k8s-loadtest" | jq 'length'

# Stop port-forward
kill $PF_PID
```

Expected: All 100 memories present.

## Troubleshooting

### Docker Compose Issues

**Problem**: Containers fail health checks
```bash
# Check logs for errors
docker compose logs engram-node1

# Verify network connectivity
docker exec engram-node1 ping -c 3 engram-node2

# Check SWIM port binding
docker exec engram-node1 netstat -uln | grep 7946
```

**Problem**: Nodes don't discover each other
```bash
# Verify seed nodes in config
docker exec engram-node1 cat /config/engram/config.toml | grep seed_nodes

# Check SWIM gossip traffic
docker exec engram-node1 tcpdump -i eth0 udp port 7946 -c 10
```

### Kubernetes Issues

**Problem**: Pods stuck in Pending
```bash
# Check events
kubectl describe pod -n engram-cluster engram-0

# Check PVC binding
kubectl get pvc -n engram-cluster
```

**Problem**: DNS resolution fails
```bash
# Test from within cluster
kubectl run -n engram-cluster test-dns --image=busybox:1.36 --rm -it --restart=Never -- \
  nslookup engram-headless
```

**Problem**: Pods can't reach each other on SWIM port
```bash
# Check network policies
kubectl get networkpolicies -n engram-cluster

# Test UDP connectivity
kubectl exec -n engram-cluster engram-0 -- nc -zu engram-1.engram-headless 7946
```

## Performance Validation

### Measure Cluster Overhead

```bash
# Start single node
docker run -d --name engram-single \
  -p 7435:7432 \
  engram/local:latest start --http-port 7432

# Start 3-node cluster
cd deployments/docker/cluster
docker compose up -d
sleep 30

# Benchmark single node
time for i in {1..1000}; do
  curl -s -X POST http://localhost:7435/api/v1/memory \
    -H "Content-Type: application/json" \
    -d "{\"content\": \"bench $i\"}" > /dev/null
done

# Benchmark cluster
time for i in {1..1000}; do
  curl -s -X POST http://localhost:7432/api/v1/memory \
    -H "Content-Type: application/json" \
    -d "{\"content\": \"bench $i\"}" > /dev/null
done
```

Expected: Cluster overhead <15% for write operations due to replication.

### Verify Convergence Time

```bash
# Create memory on node1
time curl -X POST http://localhost:7432/api/v1/memory \
  -H "Content-Type: application/json" \
  -d '{"content": "convergence test", "memory_space": "convergence"}'

# Immediately check node3
curl "http://localhost:7434/api/v1/memory?space=convergence"
```

Expected: Memory visible on all nodes within <100ms.

## Summary

This cookbook verifies:
- ✅ SWIM membership protocol works correctly
- ✅ Cluster convergence happens within 30 seconds
- ✅ DNS-based discovery functions in Kubernetes
- ✅ Static seed discovery works in Docker Compose
- ✅ Node failures detected within 2-3 seconds
- ✅ Space assignment distributes load via consistent hashing
- ✅ Replication factor=2 creates replicas on two nodes
- ✅ Self-healing works in Kubernetes StatefulSets
- ✅ Cluster scales horizontally (tested up to 5 nodes)

For production deployment guidance, see `docs/operations/clustering.md`.
