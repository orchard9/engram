# Production Deployment

Complete guide for deploying Engram in production environments using Docker, Kubernetes, or Helm.

## Prerequisites

### System Requirements

- 2+ CPU cores (4+ recommended for production)

- 4GB RAM minimum (8GB+ recommended)

- 20GB disk space for data storage

- Network access for container registry (if using pre-built images)

### Software Requirements

**For Docker deployment:**

- Docker 24.0+ with BuildKit support

- docker-compose 2.0+ (optional, for stack deployment)

**For Kubernetes deployment:**

- Kubernetes 1.28+

- kubectl configured with cluster access

- StorageClass with ReadWriteOnce support

- Minimum 20Gi persistent volume capacity

**For Helm deployment:**

- Helm 3.0+

- Kubernetes 1.28+

- kubectl configured with cluster access

## Docker Deployment

### Build from Source

```bash
# Clone repository
git clone https://github.com/orchard9/engram.git
cd engram

# Build image with BuildKit
DOCKER_BUILDKIT=1 docker build \
  -t engram:latest \
  -f deployments/docker/Dockerfile \
  .

# Verify image size (<60MB target)
docker images engram:latest

```

### Run Container

```bash
# Create data directory
mkdir -p ./engram-data

# Run with basic configuration
docker run -d \
  --name engram \
  -p 7432:7432 \
  -p 50051:50051 \
  -v $(pwd)/engram-data:/data \
  --read-only \
  --tmpfs /tmp:noexec,nosuid,size=100M \
  --security-opt no-new-privileges:true \
  --cap-drop ALL \
  --cap-add NET_BIND_SERVICE \
  engram:latest

# Verify health
docker ps
curl http://localhost:7432/health/alive
curl http://localhost:7432/api/v1/system/health | jq .

```

### Production Configuration

```bash
# Run with production settings
docker run -d \
  --name engram \
  --hostname engram-primary \
  -p 7432:7432 \
  -p 50051:50051 \
  -v $(pwd)/engram-data:/data \
  -v $(pwd)/config:/config:ro \
  --read-only \
  --tmpfs /tmp:noexec,nosuid,size=100M \
  --security-opt no-new-privileges:true \
  --security-opt apparmor:docker-default \
  --cap-drop ALL \
  --cap-add NET_BIND_SERVICE \
  --memory=4g \
  --memory-reservation=2g \
  --cpus=2.0 \
  --restart=unless-stopped \
  -e RUST_LOG=info \
  -e ENGRAM_CACHE_SIZE_MB=2048 \
  -e MIMALLOC_LARGE_OS_PAGES=1 \
  engram:latest

# Monitor logs
docker logs -f engram

# Check resource usage
docker stats engram

```

#### Gossip advertise address

Distributed deployments must ensure each node advertises a routable address for SWIM membership gossip. When `[cluster.network].swim_bind` uses `0.0.0.0`, set `[cluster.network].advertise_addr = "<host-ip>:7946"` in `engram.toml` or export `ENGRAM_CLUSTER_ADVERTISE_ADDR` before starting the server. Containerized clusters that rely on the static seed list can keep using the auto-detection fallback, but any environment tweaks (different networks, custom seeds, Kubernetes host networking) should explicitly set the advertise address so peers never learn `0.0.0.0`.

#### Cluster config template

The repository ships with `engram-cli/config/cluster.toml`, a ready-to-use template for multi-node deployments. Copy it to your config directory (typically `~/.config/engram/config.toml` or `$XDG_CONFIG_HOME/engram/config.toml`) and adjust the seed list plus `advertise_addr` for your environment. The template keeps single-node defaults in `config/default.toml`, so you can layer the cluster file on top without rewriting every setting.

#### Placement and rebalance controls

The `[cluster.replication]` section now exposes jump-hash and diversity knobs to keep assignments stable when nodes join or leave. `jump_buckets` controls how many virtual buckets the planner hashes through (higher values further smooth reassignments), while `rack_penalty` and `zone_penalty` weight the rendezvous scores so that replicas prefer distinct racks/zones before falling back to the next-best candidate.

During operations you can inspect and steer the rebalance coordinator without leaving the HTTP API:

- `GET /cluster/rebalance` returns cached assignment counts per node plus the most recent migration plans.
- `POST /cluster/rebalance` rescans all cached spaces and queues migrations for any that need to move to a new primary.

#### Replication monitoring

- `/cluster/health` now includes a `replication` block summarizing each `(space, replica)` pair. The API exposes the local and replicated sequence plus the current lag so you can spot slow followers.
- `engram status --json` surfaces the same data for CLI workflows; look for replicas whose lag exceeds the configured `cluster.replication.lag_threshold`.
- The gRPC service exposes `ApplyReplicationBatch` and `GetReplicationStatus` for automation. Operators normally won't call them directly—the runtime streams batches automatically—but they are useful for integration tests and observability tooling.
- Tune `[cluster.replication]` in `config.toml` (batch size, compression, lag threshold) to balance WAN bandwidth and recovery speed. Changes take effect on restart.
- `POST /cluster/migrate` with `{ "space": "<id>" }` forces a single memory space to migrate immediately, which is handy when soaking a new node before bringing the rest of the cluster online.

These hooks mirror the gRPC RPCs (`RebalanceSpaces`, `MigrateSpace`) so automation can consume the same workflow Engram’s CLI uses.

## docker-compose Deployment

### Basic Stack

```bash
cd deployments/docker

# Create data directory
mkdir -p ./data

# Start stack
docker-compose up -d

# Verify all services
docker-compose ps
docker-compose logs -f engram

# Health check
curl http://localhost:7432/health | jq .

```

### With Monitoring

```bash
# Start stack with Prometheus and Grafana
docker-compose --profile monitoring up -d

# Access services
# Engram:     http://localhost:7432
# Prometheus: http://localhost:9090
# Grafana:    http://localhost:3000 (admin/admin)

# Verify monitoring
curl http://localhost:9090/-/healthy
curl http://localhost:3000/api/health

```

### Configuration Override

```bash
# Create .env file
cat > .env <<EOF
RUST_LOG=debug
CACHE_SIZE_MB=4096
ENGRAM_DATA_PATH=/var/lib/engram
GRAFANA_PASSWORD=secure-password
EOF

# Start with custom config
docker-compose up -d

# Test graceful shutdown
docker-compose stop -t 30
docker-compose logs engram | grep "shutdown"

```

## Kubernetes Deployment

### Apply Manifests

```bash
cd deployments/kubernetes

# Review manifests
kubectl apply -f . --dry-run=client

# Create namespace (optional)
kubectl create namespace engram

# Apply all manifests
kubectl apply -f . -n engram

# Monitor deployment
kubectl get pods -n engram -w
kubectl get statefulset -n engram
kubectl get svc -n engram
kubectl get pvc -n engram

```

### Verify Deployment

```bash
# Check pod status
kubectl get pods -n engram
kubectl describe pod engram-0 -n engram

# Check logs
kubectl logs -f engram-0 -n engram

# Port forward for testing
kubectl port-forward pod/engram-0 7432:7432 50051:50051 -n engram &

# Health check
curl http://localhost:7432/api/v1/system/health | jq .

# Test persistence
curl -X POST http://localhost:7432/api/v1/memories/remember \
  -H "Content-Type: application/json" \
  -d '{"content": "test memory", "confidence": 0.95}'

# Restart pod and verify data persists
kubectl delete pod engram-0 -n engram
kubectl wait --for=condition=ready pod/engram-0 -n engram --timeout=60s
curl http://localhost:7432/api/v1/memories/search?q=test | jq .

```

### Access Service

```bash
# Via port-forward (development)
kubectl port-forward service/engram 7432:7432 -n engram

# Via LoadBalancer (production)
kubectl get svc engram -n engram
# Wait for EXTERNAL-IP to be assigned
export ENGRAM_URL=$(kubectl get svc engram -n engram -o jsonpath='{.status.loadBalancer.ingress[0].ip}')
curl http://$ENGRAM_URL:7432/health

```

### Update Configuration

```bash
# Edit ConfigMap
kubectl edit configmap engram-config -n engram

# Restart StatefulSet to apply changes
kubectl rollout restart statefulset engram -n engram

# Watch rollout status
kubectl rollout status statefulset engram -n engram

```

## Helm Deployment

### Install Chart

```bash
cd deployments/helm

# Lint chart
helm lint engram

# Install with default values
helm install engram engram -n engram --create-namespace

# Install with custom values
cat > values-production.yaml <<EOF
statefulset:
  replicaCount: 1

resources:
  requests:
    cpu: 2000m
    memory: 4Gi
  limits:
    cpu: 4000m
    memory: 8Gi

persistence:
  size: 50Gi
  storageClass: fast-ssd

service:
  type: LoadBalancer

monitoring:
  prometheus:
    enabled: true
    serviceMonitor:
      enabled: true
EOF

helm install engram engram \
  -n engram \
  --create-namespace \
  -f values-production.yaml \
  --wait --timeout 5m

```

### Verify Installation

```bash
# Check Helm release
helm list -n engram
helm status engram -n engram

# Check resources
kubectl get all -n engram -l app.kubernetes.io/name=engram

# View logs
kubectl logs -f -n engram -l app.kubernetes.io/name=engram

# Test health
kubectl port-forward -n engram service/engram 7432:7432 &
curl http://localhost:7432/api/v1/system/health | jq .

```

### Upgrade Deployment

```bash
# Update values
cat > values-updated.yaml <<EOF
image:
  tag: "0.1.1"

resources:
  limits:
    memory: 16Gi
EOF

# Perform upgrade
helm upgrade engram engram \
  -n engram \
  -f values-updated.yaml \
  --wait

# Verify upgrade
helm history engram -n engram
kubectl get pods -n engram

```

### Rollback

```bash
# Rollback to previous version
helm rollback engram 1 -n engram

# Verify rollback
helm history engram -n engram
kubectl get pods -n engram

```

### Uninstall

```bash
# Uninstall release (preserves PVCs)
helm uninstall engram -n engram --wait

# Delete PVCs if needed
kubectl delete pvc -n engram -l app.kubernetes.io/name=engram

```

## Bare-Metal Deployment

### Build Binary

```bash
# Install Rust
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# Build release binary
cargo build --release -p engram-cli

# Binary location
ls -lh target/release/engram

```

### Install as systemd Service

```bash
# Copy binary
sudo cp target/release/engram /usr/local/bin/
sudo chmod +x /usr/local/bin/engram

# Create user and data directory
sudo useradd -r -s /bin/false -d /var/lib/engram engram
sudo mkdir -p /var/lib/engram
sudo chown engram:engram /var/lib/engram

# Copy the production-ready service file
sudo cp deployments/systemd/engram.service /etc/systemd/system/

# Enable and start service
sudo systemctl daemon-reload
sudo systemctl enable engram
sudo systemctl start engram

# Check status
sudo systemctl status engram
sudo journalctl -u engram -f

# The service file includes:
# - RUST_BACKTRACE=full for panic capture
# - Comprehensive security hardening
# - Proper resource limits
# - Automatic restart on failure
# See deployments/systemd/engram.service for full configuration

```

## Configuration

### Data Directory

```bash
# Docker
-v /path/to/data:/data

# Kubernetes
# Configured via PVC in StatefulSet

# Bare-metal
ENGRAM_DATA_DIR=/var/lib/engram

```

### Port Configuration

```bash
# HTTP API (default: 7432)
--http-port 7432
# or
ENGRAM_HTTP_PORT=7432

# gRPC (default: 50051)
--grpc-port 50051
# or
ENGRAM_GRPC_PORT=50051

```

### Resource Limits

**Docker:**

```bash
--memory=4g
--memory-reservation=2g
--cpus=2.0

```

**Kubernetes (via Helm):**

```yaml
resources:
  requests:
    cpu: 1000m
    memory: 2Gi
  limits:
    cpu: 2000m
    memory: 4Gi

```

### Logging Configuration

```bash
# Log level
RUST_LOG=info  # error, warn, info, debug, trace

# Enable backtraces
RUST_BACKTRACE=1

# Docker logging
--log-driver json-file
--log-opt max-size=10m
--log-opt max-file=3

```

## Verification

### Health Check Endpoints

```bash
# Lightweight liveness probe
curl http://localhost:7432/health/alive
# Expected: HTTP 200

# Simple health check
curl http://localhost:7432/health
# Expected: {"status": "healthy", ...}

# Comprehensive health with metrics
curl http://localhost:7432/api/v1/system/health | jq .
# Expected: {"status": "healthy", "checks": [...], "spaces": [...]}

```

### Store Test Memory

```bash
# Remember episode
curl -X POST http://localhost:7432/api/v1/memories/remember \
  -H "Content-Type: application/json" \
  -d '{
    "content": "deployment test memory",
    "confidence": 0.95
  }' | jq .

# Expected: {"id": "...", "confidence": 0.95, ...}

```

### Recall Test Memory

```bash
# Search memories
curl http://localhost:7432/api/v1/memories/search?q=deployment | jq .

# Expected: {"memories": [...], "count": 1}

```

### Monitor Logs

**Docker:**

```bash
docker logs -f engram

```

**Kubernetes:**

```bash
kubectl logs -f engram-0 -n engram

```

**Bare-metal:**

```bash
sudo journalctl -u engram -f

```

## Troubleshooting

### Container Won't Start

```bash
# Check Docker logs
docker logs engram

# Common issues:
# - Port already in use (change with --http-port)
# - Insufficient memory (increase --memory limit)
# - Permission denied on /data (check volume permissions)

```

### Health Check Failing

```bash
# Check if process is running
docker exec engram ps aux

# Test health endpoint directly
docker exec engram curl http://localhost:7432/health/alive

# Review startup logs
docker logs engram | grep -E "started|error|failed"

```

### Port Conflicts

```bash
# Find process using port 7432
lsof -i :7432

# Use alternative ports
docker run ... -p 8432:8432 \
  -e ENGRAM_HTTP_PORT=8432 \
  engram:latest

```

### Permission Errors

```bash
# Fix data directory permissions
sudo chown -R 65534:65534 ./engram-data

# Or use Docker volume
docker volume create engram-data
docker run ... -v engram-data:/data ...

```

### Kubernetes Pod Not Ready

```bash
# Check pod events
kubectl describe pod engram-0 -n engram

# Check probe status
kubectl get pod engram-0 -n engram -o yaml | grep -A 10 "conditions:"

# Common issues:
# - PVC not bound (check storageClass)
# - Init container failed (check permissions)
# - Startup timeout (increase failureThreshold)

```

### StatefulSet Update Stuck

```bash
# Check update strategy (should be OnDelete)
kubectl get statefulset engram -n engram -o yaml | grep updateStrategy

# Manual pod deletion for updates
kubectl delete pod engram-0 -n engram

# Wait for pod to recreate
kubectl wait --for=condition=ready pod/engram-0 -n engram

```

## Security Best Practices

### Container Security

- Use distroless base image (no shell access)

- Run as non-root user (uid 65534)

- Read-only root filesystem

- Drop all Linux capabilities except NET_BIND_SERVICE

- Enable AppArmor/SELinux profiles

- Scan images for vulnerabilities (Trivy, Snyk)

### Network Security

- Use LoadBalancer with source IP restrictions

- Configure NetworkPolicies in Kubernetes

- Enable TLS for production traffic (Task 008)

- Implement firewall rules for bare-metal

### Data Security

- Encrypt data at rest (storage-level encryption)

- Secure volume mounts with proper permissions

- Use Kubernetes Secrets for sensitive data

- Regular backup procedures (Task 002)

## Performance Tuning

### Memory Allocator

```bash
# Enable huge pages
MIMALLOC_LARGE_OS_PAGES=1
MIMALLOC_RESERVE_HUGE_OS_PAGES=4

# NUMA awareness
MIMALLOC_USE_NUMA_NODES=4

```

### Cache Configuration

```bash
ENGRAM_CACHE_SIZE_MB=2048

```

### Resource Reservation

Ensure minimum resources for consistent performance:

```yaml
resources:
  requests:
    cpu: 1000m
    memory: 2Gi

```

## Panic and Crash Capture

Engram uses Rust's panic system for catastrophic failures. To ensure all panics are captured in production, proper configuration is critical.

### Why This Matters

Without panic capture:
- Server crashes are silent - no diagnostic information
- Debugging production issues becomes impossible
- Root cause analysis requires reproducing the issue

With proper panic capture:
- Full stack traces captured automatically
- Structured logging integrates with monitoring systems
- Incidents can be diagnosed from crash logs alone

### Custom Panic Hook

Engram includes a custom panic hook (added in `engram-cli/src/main.rs`) that:

- Logs panics through the tracing infrastructure (structured, forwarded to Sentry/CloudWatch/etc.)
- Includes full backtraces when `RUST_BACKTRACE` is set
- Writes to both structured logs AND stderr (fallback if tracing fails)
- Never loses panic messages even if the process crashes immediately

### Environment Variables

**CRITICAL**: Always set these in production:

```bash
# Full stack traces (recommended for production)
export RUST_BACKTRACE=full

# Abbreviated traces (use for lower log volume)
export RUST_BACKTRACE=1

# Logging level
export RUST_LOG=info  # or "debug" for verbose logging
```

### Docker Configuration

Already configured in `docker-compose.yml`:

```yaml
environment:
  RUST_BACKTRACE: ${RUST_BACKTRACE:-1}  # Default to 1, override via .env
  RUST_LOG: ${RUST_LOG:-info}
```

View panic logs:

```bash
# Follow logs in real-time
docker logs -f engram-db

# Search for panics
docker logs engram-db 2>&1 | grep -A 50 "PANIC"

# Export logs for analysis
docker logs engram-db > engram-crash-$(date +%Y%m%d-%H%M%S).log
```

### Systemd Configuration

Use the provided service file at `deployments/systemd/engram.service`:

```bash
# Install service file
sudo cp deployments/systemd/engram.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable engram
sudo systemctl start engram
```

The service file includes:
- `Environment="RUST_BACKTRACE=full"` for complete traces
- `StandardOutput=journal` and `StandardError=journal` for systemd log capture
- Proper restart policies for crash recovery

View panic logs:

```bash
# Real-time monitoring
sudo journalctl -u engram -f

# Search for panics
sudo journalctl -u engram | grep -A 50 "PANIC"

# Last 100 lines
sudo journalctl -u engram -n 100

# Export for analysis
sudo journalctl -u engram --since "1 hour ago" > engram-crash-$(date +%Y%m%d-%H%M%S).log
```

### Kubernetes Configuration

Add to deployment spec:

```yaml
apiVersion: apps/v1
kind: StatefulSet
metadata:
  name: engram
spec:
  template:
    spec:
      containers:
      - name: engram
        env:
        - name: RUST_BACKTRACE
          value: "full"
        - name: RUST_LOG
          value: "info"
```

View panic logs:

```bash
# Follow logs
kubectl logs -f engram-0 -n engram

# Search for panics in all pods
kubectl logs -l app=engram -n engram | grep -A 50 "PANIC"

# Get logs from crashed pod
kubectl logs engram-0 -n engram --previous
```

### Panic Log Format

With Engram's custom panic hook, panics are logged with:

```
═══════════════════════════════════════════════════════════
PANIC at src/store.rs:245:13: attempt to divide by zero
═══════════════════════════════════════════════════════════
Backtrace:
   0: std::backtrace::Backtrace::force_capture
             at /rustc/.../library/std/src/backtrace.rs:101:13
   1: engram_cli::setup_panic_hook::{{closure}}
             at ./engram-cli/src/main.rs:49:22
   2: std::panicking::rust_panic_with_hook
             at /rustc/.../library/std/src/panicking.rs:785:13
   ... (full trace continues)
═══════════════════════════════════════════════════════════
```

Additionally, the panic is logged through tracing as a structured log:

```json
{
  "timestamp": "2025-11-10T17:29:58.234Z",
  "level": "ERROR",
  "message": "PANIC: Thread panicked - this is a critical error that should be investigated",
  "fields": {
    "message": "attempt to divide by zero",
    "location": "src/store.rs:245:13",
    "backtrace": "... (full trace) ..."
  }
}
```

### Monitoring and Alerting

Set up alerts for panics:

**Prometheus/Grafana**:
```promql
# Alert if any panics detected in logs
rate(log_messages{level="error", message=~".*PANIC.*"}[5m]) > 0
```

**Kubernetes**:
```bash
# CrashLoopBackOff indicates repeated crashes
kubectl get pods -n engram -w
```

**Docker**:
```bash
# Check restart count
docker inspect engram-db --format='{{.RestartCount}}'
```

### Crash Recovery

Engram's restart policies ensure automatic recovery:

- **Docker**: `restart: unless-stopped` (docker-compose.yml)
- **Systemd**: `Restart=always` with 5-second delay
- **Kubernetes**: StatefulSet automatically recreates pods

After a crash:
1. Verify data integrity (WAL recovery runs automatically)
2. Check monitoring dashboards for anomalies
3. Review panic logs for root cause
4. File bug report with full trace

### Production Checklist

- [ ] `RUST_BACKTRACE=full` set in all environments
- [ ] Logs captured to persistent storage (journald, Docker volumes, or log aggregator)
- [ ] Monitoring alerts configured for panics
- [ ] Restart policies tested (verify server recovers from kill -9)
- [ ] Log rotation configured (prevent disk exhaustion)
- [ ] Team knows how to access panic logs

## Next Steps

- [Setup Monitoring](monitoring.md) - Configure Prometheus and Grafana (Task 003)

- [Configure Backups](backup-restore.md) - Setup automated backups (Task 002)

- [Performance Tuning](performance-tuning.md) - Optimize for production workloads (Task 004)

- [Scaling Guide](scaling.md) - Horizontal and vertical scaling strategies (Milestone 14)
