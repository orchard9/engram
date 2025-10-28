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
sudo useradd -r -s /bin/false engram
sudo mkdir -p /var/lib/engram
sudo chown engram:engram /var/lib/engram

# Create systemd service
sudo tee /etc/systemd/system/engram.service <<EOF
[Unit]
Description=Engram Cognitive Graph Database
After=network.target

[Service]
Type=simple
User=engram
Group=engram
WorkingDirectory=/var/lib/engram
Environment="ENGRAM_DATA_DIR=/var/lib/engram"
Environment="RUST_LOG=info"
ExecStart=/usr/local/bin/engram start --http-port 7432 --grpc-port 50051
Restart=always
RestartSec=10

# Security hardening
NoNewPrivileges=true
PrivateTmp=true
ProtectSystem=strict
ProtectHome=true
ReadWritePaths=/var/lib/engram

[Install]
WantedBy=multi-user.target
EOF

# Enable and start service
sudo systemctl daemon-reload
sudo systemctl enable engram
sudo systemctl start engram

# Check status
sudo systemctl status engram
sudo journalctl -u engram -f

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

## Next Steps

- [Setup Monitoring](monitoring.md) - Configure Prometheus and Grafana (Task 003)

- [Configure Backups](backup-restore.md) - Setup automated backups (Task 002)

- [Performance Tuning](performance-tuning.md) - Optimize for production workloads (Task 004)

- [Scaling Guide](scaling.md) - Horizontal and vertical scaling strategies (Milestone 14)
