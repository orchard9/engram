# Task 001: Container & Orchestration Deployment — pending

**Priority:** P0 (Critical Path)
**Estimated Effort:** 3 days
**Dependencies:** None

## Objective

Deploy Engram via Docker and Kubernetes with production-grade configurations. Enable operators to deploy in <30 minutes using containerized infrastructure. Support local development (docker-compose), small deployments (Docker), and production (Kubernetes/Helm).

## Integration Points

**Uses:**
- `/engram-cli/src/main.rs` - Server entry point with CLI commands:
  - `engram start --http-port 7432 --grpc-port 50051` - Main server startup
  - `engram status --json` - Health check for container probes
  - `engram benchmark` - Performance testing capability
- `/engram-cli/src/config.rs` - Configuration system supporting:
  - TOML configuration files mounted at `/config/engram.toml`
  - Environment variable overrides (ENGRAM_* prefix)
  - Feature flags and persistence settings
- `/engram-cli/src/api.rs` - HTTP endpoints for health checks:
  - `/health/alive` - Lightweight liveness probe (startup/liveness)
  - `/health` - Simple health check (liveness probe)
  - `/api/v1/system/health` - Comprehensive health with metrics (readiness)
- `/Cargo.toml` - Build configuration with:
  - Release profile optimizations (LTO, codegen-units=1, strip)
  - Workspace dependencies for consistent versions
  - Target Rust 2024 edition
- `/scripts/check_engram_health.sh` - Existing health check script (reference implementation)

**Creates:**
- `/deployments/docker/Dockerfile` - Multi-stage container build
- `/deployments/docker/docker-compose.yml` - Local development stack
- `/deployments/docker/.dockerignore` - Build optimization
- `/deployments/kubernetes/deployment.yaml` - K8s deployment manifest
- `/deployments/kubernetes/service.yaml` - K8s service manifest
- `/deployments/kubernetes/configmap.yaml` - Configuration template
- `/deployments/kubernetes/secret.yaml.example` - Secrets template
- `/deployments/kubernetes/pvc.yaml` - Persistent volume claim
- `/deployments/helm/engram/Chart.yaml` - Helm chart metadata
- `/deployments/helm/engram/values.yaml` - Configurable values
- `/deployments/helm/engram/templates/deployment.yaml` - Templated deployment
- `/deployments/helm/engram/templates/service.yaml` - Templated service
- `/deployments/helm/engram/templates/configmap.yaml` - Templated config
- `/deployments/helm/engram/templates/_helpers.tpl` - Helm helpers

**Updates:**
- `/docs/operations/production-deployment.md` - Complete deployment guide

## Technical Specifications

### Dockerfile Requirements

**Multi-stage build with optimal layering:**

```dockerfile
# Stage 1: Dependency caching layer
FROM rust:1.83-slim AS deps
WORKDIR /build
# Copy only manifests first for maximum cache reuse
COPY Cargo.toml Cargo.lock ./
COPY engram-core/Cargo.toml engram-core/
COPY engram-cli/Cargo.toml engram-cli/
COPY engram-proto/Cargo.toml engram-proto/
COPY engram-storage/Cargo.toml engram-storage/
COPY xtask/Cargo.toml xtask/
# Create dummy source files to build dependencies only
RUN mkdir -p engram-core/src engram-cli/src engram-proto/src engram-storage/src xtask/src && \
    echo "fn main() {}" > engram-cli/src/main.rs && \
    touch engram-core/src/lib.rs engram-proto/src/lib.rs engram-storage/src/lib.rs xtask/src/main.rs && \
    cargo build --release -p engram-cli && \
    rm -rf engram-*/src xtask/src

# Stage 2: Build the actual binary
FROM deps AS builder
COPY . .
# Touch main.rs to ensure rebuild with actual source
RUN touch engram-cli/src/main.rs && \
    cargo build --release -p engram-cli && \
    # Extract binary size for validation
    ls -lh target/release/engram

# Stage 3: Minimal runtime with security hardening
FROM gcr.io/distroless/cc-debian12:nonroot
# Use distroless for minimal attack surface (6MB base)
# Contains only glibc and essential libraries

# Copy binary with proper permissions
COPY --from=builder --chown=nonroot:nonroot /build/target/release/engram /usr/local/bin/engram

# Create data directory with correct ownership
USER nonroot:nonroot
WORKDIR /data

# Expose ports (HTTP API and gRPC)
EXPOSE 7432 50051

# Health check using built-in CLI status command
HEALTHCHECK --interval=30s --timeout=3s --start-period=10s --retries=3 \
  CMD ["/usr/local/bin/engram", "status", "--json"] || exit 1

# Signal handling for graceful shutdown
STOPSIGNAL SIGTERM

# Set memory allocator for better container performance
ENV MIMALLOC_SHOW_STATS=0
ENV RUST_LOG=info

ENTRYPOINT ["/usr/local/bin/engram"]
CMD ["start", "--http-port", "7432", "--grpc-port", "50051"]
```

**Container hardening:**
- Distroless base image (no shell, package manager, or utilities)
- Read-only root filesystem capability
- Drop all Linux capabilities except NET_BIND_SERVICE
- Security scanning with Trivy/Snyk in CI
- No SUID/SGID binaries
- Minimal attack surface (~6MB base + ~50MB binary)

**Build optimization:**
- Dependency caching layer saves 80% rebuild time
- Profile-guided optimization (PGO) for 15-20% performance gain
- Link-time optimization (LTO) enabled in release profile
- Strip debug symbols (already configured in Cargo.toml)
- Target CPU features: `-C target-cpu=x86-64-v2` for broad compatibility

**NUMA and CPU affinity:**
```dockerfile
# For NUMA-aware deployments (optional enhanced Dockerfile)
ENV ENGRAM_NUMA_NODE=0
ENV ENGRAM_CPU_AFFINITY="0-3"
```

### docker-compose.yml Requirements

**Production-grade compose configuration with performance tuning:**

```yaml
version: '3.8'

services:
  engram:
    build:
      context: ../..
      dockerfile: deployments/docker/Dockerfile
      cache_from:
        - engram/engram:deps-cache
      args:
        BUILDKIT_INLINE_CACHE: 1
    image: engram/engram:latest
    container_name: engram-db
    hostname: engram-primary
    ports:
      - "7432:7432"    # HTTP API
      - "50051:50051"  # gRPC
    volumes:
      - type: volume
        source: engram-data
        target: /data
        volume:
          nocopy: true
      - type: bind
        source: ./config
        target: /config
        read_only: true
    environment:
      # Core configuration
      ENGRAM_DATA_DIR: /data
      ENGRAM_HTTP_PORT: 7432
      ENGRAM_GRPC_PORT: 50051
      RUST_LOG: ${RUST_LOG:-info}
      RUST_BACKTRACE: ${RUST_BACKTRACE:-1}

      # Performance tuning
      ENGRAM_CACHE_SIZE_MB: ${CACHE_SIZE_MB:-2048}
      ENGRAM_CONSOLIDATION_INTERVAL_SEC: 60
      ENGRAM_FLUSH_INTERVAL_SEC: 30

      # Memory allocator tuning for containers
      MIMALLOC_LARGE_OS_PAGES: 1
      MIMALLOC_RESERVE_HUGE_OS_PAGES: 4

    # Resource limits and reservations
    deploy:
      resources:
        limits:
          cpus: '2.0'
          memory: 4G
        reservations:
          cpus: '1.0'
          memory: 2G

    # Security options
    security_opt:
      - no-new-privileges:true
      - apparmor:docker-default
    cap_drop:
      - ALL
    cap_add:
      - NET_BIND_SERVICE  # Allow binding to privileged ports if needed
    read_only: true        # Read-only root filesystem
    tmpfs:
      - /tmp:noexec,nosuid,size=100M

    # Health monitoring
    healthcheck:
      test: ["CMD", "/usr/local/bin/engram", "status", "--json"]
      interval: 30s
      timeout: 3s
      start_period: 10s
      retries: 3

    # Restart policy
    restart: unless-stopped

    # Logging configuration
    logging:
      driver: "json-file"
      options:
        max-size: "10m"
        max-file: "3"
        compress: "true"

    # Network configuration
    networks:
      engram-net:
        aliases:
          - engram-db

    # Signal handling
    stop_signal: SIGTERM
    stop_grace_period: 30s

  # Optional: Prometheus for metrics collection
  prometheus:
    image: prom/prometheus:v2.45.0
    container_name: engram-prometheus
    profiles: ["monitoring"]  # Only start with --profile monitoring
    ports:
      - "9090:9090"
    volumes:
      - prometheus-data:/prometheus
      - ./monitoring/prometheus.yml:/etc/prometheus/prometheus.yml:ro
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.path=/prometheus'
      - '--web.console.libraries=/usr/share/prometheus/console_libraries'
      - '--web.console.templates=/usr/share/prometheus/consoles'
      - '--storage.tsdb.retention.time=7d'
    networks:
      - engram-net
    restart: unless-stopped
    depends_on:
      - engram

  # Optional: Grafana for visualization
  grafana:
    image: grafana/grafana:10.0.0
    container_name: engram-grafana
    profiles: ["monitoring"]  # Only start with --profile monitoring
    ports:
      - "3000:3000"
    volumes:
      - grafana-data:/var/lib/grafana
      - ./monitoring/grafana/dashboards:/var/lib/grafana/dashboards:ro
      - ./monitoring/grafana/provisioning:/etc/grafana/provisioning:ro
    environment:
      GF_SECURITY_ADMIN_PASSWORD: ${GRAFANA_PASSWORD:-admin}
      GF_INSTALL_PLUGINS: ${GRAFANA_PLUGINS:-}
    networks:
      - engram-net
    restart: unless-stopped
    depends_on:
      - prometheus

volumes:
  engram-data:
    driver: local
    driver_opts:
      type: none
      o: bind
      device: ${ENGRAM_DATA_PATH:-./data}
  prometheus-data:
    driver: local
  grafana-data:
    driver: local

networks:
  engram-net:
    driver: bridge
    ipam:
      driver: default
      config:
        - subnet: 172.28.0.0/16
```

**Performance optimizations:**
- BuildKit caching for faster rebuilds
- Memory allocator tuning (MIMALLOC with huge pages)
- Resource reservations ensure minimum performance
- Optimized volume mounts with `nocopy` flag
- Network aliases for service discovery

### Kubernetes Manifests Requirements

**StatefulSet (not Deployment - for stable network identity and ordered operations):**

```yaml
apiVersion: apps/v1
kind: StatefulSet
metadata:
  name: engram
  labels:
    app: engram
    component: database
    version: "0.1.0"
spec:
  serviceName: engram-headless
  replicas: 1  # Single node (distributed in Milestone 14)
  selector:
    matchLabels:
      app: engram
  updateStrategy:
    type: OnDelete  # Manual control for stateful upgrades
  template:
    metadata:
      labels:
        app: engram
        component: database
      annotations:
        prometheus.io/scrape: "true"
        prometheus.io/port: "7432"
        prometheus.io/path: "/metrics"
    spec:
      # Pod scheduling and affinity
      affinity:
        podAntiAffinity:
          requiredDuringSchedulingIgnoredDuringExecution:
          - labelSelector:
              matchExpressions:
              - key: app
                operator: In
                values: [engram]
            topologyKey: kubernetes.io/hostname
        nodeAffinity:
          preferredDuringSchedulingIgnoredDuringExecution:
          - weight: 100
            preference:
              matchExpressions:
              - key: node.kubernetes.io/instance-type
                operator: In
                values: ["memory-optimized", "compute-optimized"]

      # Security context for pod
      securityContext:
        runAsNonRoot: true
        runAsUser: 65534  # nonroot user
        runAsGroup: 65534
        fsGroup: 65534
        fsGroupChangePolicy: "OnRootMismatch"

      # Init container for data directory permissions
      initContainers:
      - name: init-permissions
        image: busybox:1.36
        command: ['sh', '-c', 'chown -R 65534:65534 /data']
        volumeMounts:
        - name: data
          mountPath: /data
        securityContext:
          runAsUser: 0  # Need root for chown

      containers:
      - name: engram
        image: engram/engram:0.1.0
        imagePullPolicy: IfNotPresent

        # Command and args
        command: ["/usr/local/bin/engram"]
        args:
        - "start"
        - "--http-port=7432"
        - "--grpc-port=50051"

        # Ports
        ports:
        - containerPort: 7432
          name: http
          protocol: TCP
        - containerPort: 50051
          name: grpc
          protocol: TCP

        # Environment variables
        env:
        - name: ENGRAM_DATA_DIR
          value: "/data"
        - name: ENGRAM_NODE_ID
          valueFrom:
            fieldRef:
              fieldPath: metadata.name
        - name: ENGRAM_POD_IP
          valueFrom:
            fieldRef:
              fieldPath: status.podIP
        - name: RUST_LOG
          value: "info"
        - name: MIMALLOC_LARGE_OS_PAGES
          value: "1"

        # Probes with proper timing
        startupProbe:
          httpGet:
            path: /health/alive
            port: http
          initialDelaySeconds: 5
          periodSeconds: 5
          timeoutSeconds: 2
          successThreshold: 1
          failureThreshold: 12  # 60 seconds total

        livenessProbe:
          httpGet:
            path: /health
            port: http
          initialDelaySeconds: 0  # After startup probe
          periodSeconds: 30
          timeoutSeconds: 3
          successThreshold: 1
          failureThreshold: 3

        readinessProbe:
          httpGet:
            path: /api/v1/system/health
            port: http
            httpHeaders:
            - name: Accept
              value: application/json
          initialDelaySeconds: 0
          periodSeconds: 10
          timeoutSeconds: 3
          successThreshold: 1
          failureThreshold: 3

        # Resources
        resources:
          requests:
            cpu: "1000m"
            memory: "2Gi"
            ephemeral-storage: "1Gi"
          limits:
            cpu: "2000m"
            memory: "4Gi"
            ephemeral-storage: "2Gi"

        # Volume mounts
        volumeMounts:
        - name: data
          mountPath: /data
        - name: config
          mountPath: /config
          readOnly: true
        - name: tmp
          mountPath: /tmp

        # Container security
        securityContext:
          allowPrivilegeEscalation: false
          capabilities:
            drop: ["ALL"]
            add: ["NET_BIND_SERVICE"]
          readOnlyRootFilesystem: true
          seccompProfile:
            type: RuntimeDefault

      # Volumes (PVC created by StatefulSet)
      volumes:
      - name: config
        configMap:
          name: engram-config
          defaultMode: 0444
      - name: tmp
        emptyDir:
          sizeLimit: 100Mi

      # Service account
      serviceAccountName: engram

      # Termination grace period for clean shutdown
      terminationGracePeriodSeconds: 30

      # DNS configuration for better resolution
      dnsPolicy: ClusterFirst
      dnsConfig:
        options:
        - name: ndots
          value: "2"

  # Volume claim template for StatefulSet
  volumeClaimTemplates:
  - metadata:
      name: data
      labels:
        app: engram
    spec:
      accessModes: ["ReadWriteOnce"]
      storageClassName: fast-ssd  # Use appropriate storage class
      resources:
        requests:
          storage: 20Gi
```

**Service configurations:**

```yaml
# Headless service for StatefulSet
apiVersion: v1
kind: Service
metadata:
  name: engram-headless
  labels:
    app: engram
spec:
  clusterIP: None
  selector:
    app: engram
  ports:
  - name: http
    port: 7432
  - name: grpc
    port: 50051

---
# Regular service for client access
apiVersion: v1
kind: Service
metadata:
  name: engram
  labels:
    app: engram
  annotations:
    service.beta.kubernetes.io/aws-load-balancer-type: "nlb"  # For AWS
spec:
  type: LoadBalancer
  selector:
    app: engram
  ports:
  - name: http
    port: 7432
    targetPort: http
    protocol: TCP
  - name: grpc
    port: 50051
    targetPort: grpc
    protocol: TCP
  sessionAffinity: ClientIP
  sessionAffinityConfig:
    clientIP:
      timeoutSeconds: 10800  # 3 hours
```

**ConfigMap with production tuning:**

```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: engram-config
data:
  engram.toml: |
    [feature_flags]
    spreading_api_beta = true

    [memory_spaces]
    default_space = "default"
    bootstrap_spaces = ["default"]

    [persistence]
    data_root = "/data"
    hot_capacity = 10000
    warm_capacity = 100000
    cold_capacity = 1000000
    flush_interval_sec = 30
    compression_enabled = true

    [consolidation]
    enabled = true
    interval_sec = 60
    batch_size = 1000
    parallel_workers = 4

    [cache]
    size_mb = 2048
    ttl_sec = 3600

    [performance]
    max_concurrent_queries = 100
    query_timeout_sec = 30
```

### Helm Chart Requirements

**Production-grade Helm chart with comprehensive configurability:**

**Chart.yaml:**
```yaml
apiVersion: v2
name: engram
description: Cognitive graph database for episodic and semantic memory
version: 0.1.0
appVersion: "0.1.0"
type: application
keywords:
  - database
  - graph
  - memory
  - cognitive
  - probabilistic
maintainers:
  - name: Engram Contributors
    email: maintainers@engram.io
dependencies:
  - name: prometheus
    version: "15.x.x"
    repository: https://prometheus-community.github.io/helm-charts
    condition: monitoring.prometheus.enabled
  - name: grafana
    version: "6.x.x"
    repository: https://grafana.github.io/helm-charts
    condition: monitoring.grafana.enabled
```

**values.yaml with production defaults:**
```yaml
# Global settings
global:
  # Image pull secrets for private registries
  imagePullSecrets: []
  # Storage class for all PVCs
  storageClass: "fast-ssd"

# Deployment configuration
statefulset:
  # Number of replicas (1 for now, multi-node in M14)
  replicaCount: 1
  # Update strategy
  updateStrategy:
    type: OnDelete  # Manual control for stateful services

# Image configuration
image:
  repository: engram/engram
  pullPolicy: IfNotPresent
  tag: ""  # Defaults to chart appVersion

# Service account configuration
serviceAccount:
  create: true
  annotations: {}
  name: ""  # Generated if not set

# Security context
securityContext:
  runAsNonRoot: true
  runAsUser: 65534
  runAsGroup: 65534
  fsGroup: 65534
  fsGroupChangePolicy: "OnRootMismatch"

# Container security
containerSecurityContext:
  allowPrivilegeEscalation: false
  capabilities:
    drop: ["ALL"]
    add: ["NET_BIND_SERVICE"]
  readOnlyRootFilesystem: true
  seccompProfile:
    type: RuntimeDefault

# Service configuration
service:
  type: ClusterIP  # Options: ClusterIP, NodePort, LoadBalancer
  httpPort: 7432
  grpcPort: 50051

  # LoadBalancer specific
  loadBalancerIP: ""
  loadBalancerSourceRanges: []

  # Annotations (e.g., for cloud providers)
  annotations: {}
    # service.beta.kubernetes.io/aws-load-balancer-type: "nlb"
    # service.beta.kubernetes.io/azure-load-balancer-internal: "true"

  # Session affinity
  sessionAffinity: ClientIP
  sessionAffinityTimeoutSeconds: 10800

# Ingress configuration
ingress:
  enabled: false
  className: "nginx"
  annotations: {}
    # nginx.ingress.kubernetes.io/backend-protocol: "GRPC"
    # cert-manager.io/cluster-issuer: "letsencrypt"
  hosts:
    - host: engram.example.com
      paths:
        - path: /
          pathType: Prefix
  tls: []
    # - secretName: engram-tls
    #   hosts:
    #     - engram.example.com

# Resource configuration with vertical pod autoscaling support
resources:
  limits:
    cpu: 2000m
    memory: 4Gi
    ephemeral-storage: 2Gi
  requests:
    cpu: 1000m
    memory: 2Gi
    ephemeral-storage: 1Gi

# Autoscaling configuration
autoscaling:
  enabled: false
  minReplicas: 1
  maxReplicas: 10
  targetCPUUtilizationPercentage: 80
  targetMemoryUtilizationPercentage: 80

# Persistence configuration
persistence:
  enabled: true
  # Storage class (uses global if not set)
  storageClass: ""
  # Access mode
  accessMode: ReadWriteOnce
  # Size
  size: 20Gi
  # Annotations for the PVC
  annotations: {}
  # Existing claim to use
  existingClaim: ""
  # Volume snapshot to restore from
  dataSource: {}

# Probe configuration
probes:
  startup:
    enabled: true
    initialDelaySeconds: 5
    periodSeconds: 5
    timeoutSeconds: 2
    failureThreshold: 12
    successThreshold: 1

  liveness:
    enabled: true
    initialDelaySeconds: 0
    periodSeconds: 30
    timeoutSeconds: 3
    failureThreshold: 3
    successThreshold: 1

  readiness:
    enabled: true
    initialDelaySeconds: 0
    periodSeconds: 10
    timeoutSeconds: 3
    failureThreshold: 3
    successThreshold: 1

# Node selection
nodeSelector: {}
  # node.kubernetes.io/instance-type: memory-optimized

# Tolerations for node taints
tolerations: []
  # - key: "dedicated"
  #   operator: "Equal"
  #   value: "database"
  #   effect: "NoSchedule"

# Affinity rules
affinity:
  # Pod anti-affinity to spread across nodes
  podAntiAffinity:
    enabled: true
    topologyKey: kubernetes.io/hostname

  # Node affinity for specific hardware
  nodeAffinity:
    enabled: false
    matchExpressions: []
      # - key: node.kubernetes.io/instance-type
      #   operator: In
      #   values: ["memory-optimized", "compute-optimized"]

# Priority class
priorityClassName: ""

# Topology spread constraints for even distribution
topologySpreadConstraints: []
  # - maxSkew: 1
  #   topologyKey: topology.kubernetes.io/zone
  #   whenUnsatisfiable: DoNotSchedule

# Application configuration
config:
  # Core settings
  dataDir: /data
  logLevel: info

  # Feature flags
  featureFlags:
    spreadingApiBeta: true

  # Memory spaces
  memorySpaces:
    defaultSpace: "default"
    bootstrapSpaces: ["default"]

  # Persistence settings
  persistence:
    hotCapacity: 10000
    warmCapacity: 100000
    coldCapacity: 1000000
    flushIntervalSec: 30
    compressionEnabled: true

  # Consolidation settings
  consolidation:
    enabled: true
    intervalSec: 60
    batchSize: 1000
    parallelWorkers: 4

  # Cache settings
  cache:
    sizeMb: 2048
    ttlSec: 3600

  # Performance settings
  performance:
    maxConcurrentQueries: 100
    queryTimeoutSec: 30

  # Memory allocator tuning
  mimalloc:
    largeOsPages: true
    reserveHugeOsPages: 4

# Environment variables
env: []
  # - name: CUSTOM_ENV
  #   value: "custom_value"

# Extra volumes
extraVolumes: []
  # - name: custom-config
  #   configMap:
  #     name: custom-config

# Extra volume mounts
extraVolumeMounts: []
  # - name: custom-config
  #   mountPath: /custom
  #   readOnly: true

# Monitoring configuration
monitoring:
  # Prometheus metrics
  prometheus:
    enabled: false
    # ServiceMonitor for Prometheus Operator
    serviceMonitor:
      enabled: false
      interval: 30s
      scrapeTimeout: 10s
      labels: {}

  # Grafana dashboards
  grafana:
    enabled: false
    dashboards:
      enabled: false
      labels:
        grafana_dashboard: "1"

# Backup configuration (for Task 002)
backup:
  enabled: false
  schedule: "0 2 * * *"  # Daily at 2 AM
  retention: 7  # Days
  storageClass: "slow-disk"
  size: 50Gi

# Network policies
networkPolicy:
  enabled: false
  policyTypes:
    - Ingress
    - Egress
  ingress: []
  egress: []

# Pod disruption budget
podDisruptionBudget:
  enabled: false
  minAvailable: 1
  # maxUnavailable: 1

# Extra manifests to deploy
extraManifests: []
```

**Template structure with production patterns:**

```yaml
# templates/_helpers.tpl
{{- define "engram.name" -}}
{{- default .Chart.Name .Values.nameOverride | trunc 63 | trimSuffix "-" }}
{{- end }}

{{- define "engram.fullname" -}}
{{- if .Values.fullnameOverride }}
{{- .Values.fullnameOverride | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- $name := default .Chart.Name .Values.nameOverride }}
{{- printf "%s-%s" .Release.Name $name | trunc 63 | trimSuffix "-" }}
{{- end }}
{{- end }}

{{- define "engram.labels" -}}
helm.sh/chart: {{ include "engram.chart" . }}
{{ include "engram.selectorLabels" . }}
{{- if .Chart.AppVersion }}
app.kubernetes.io/version: {{ .Chart.AppVersion | quote }}
{{- end }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
{{- end }}

{{- define "engram.podAnnotations" -}}
checksum/config: {{ include (print $.Template.BasePath "/configmap.yaml") . | sha256sum }}
{{- with .Values.podAnnotations }}
{{- toYaml . | nindent 0 }}
{{- end }}
{{- end }}
```

## Testing Requirements

### Container Security Scanning
```bash
# Build and scan with Trivy
docker build -t engram:test -f deployments/docker/Dockerfile .
trivy image --severity HIGH,CRITICAL engram:test

# Verify distroless base has minimal attack surface
docker run --rm -it --entrypoint sh engram:test 2>&1 | grep "No such file"  # Should fail

# Check container size (target: <60MB)
docker images engram:test --format "table {{.Repository}}\t{{.Tag}}\t{{.Size}}"

# Verify non-root user
docker run --rm engram:test id | grep "uid=65534"

# Test read-only filesystem
docker run --rm --read-only \
  --tmpfs /tmp:noexec,nosuid,size=100M \
  -v engram-test-data:/data \
  engram:test status
```

### Docker Performance Testing
```bash
# Build with BuildKit for cache analysis
DOCKER_BUILDKIT=1 docker build \
  --progress=plain \
  --target deps \
  -t engram:deps-cache \
  -f deployments/docker/Dockerfile .

# Measure startup time
docker run -d --name engram-perf \
  -p 7432:7432 \
  --cpus="2" \
  --memory="4g" \
  engram:test

# Wait for readiness and measure time
start=$(date +%s)
until curl -s http://localhost:7432/api/v1/system/health | grep -q healthy; do
  sleep 0.1
done
end=$(date +%s)
echo "Startup time: $((end - start)) seconds"  # Target: <10s

# Load test with concurrent connections
for i in {1..100}; do
  curl -X POST http://localhost:7432/api/v1/memories/remember \
    -H "Content-Type: application/json" \
    -d "{\"content\": \"test-$i\", \"confidence\": 0.9}" &
done
wait

# Check memory usage
docker stats --no-stream engram-perf

# Cleanup
docker stop engram-perf && docker rm engram-perf
```

### docker-compose Production Testing
```bash
cd deployments/docker

# Test with production settings
RUST_LOG=debug \
CACHE_SIZE_MB=4096 \
ENGRAM_DATA_PATH=/var/lib/engram \
docker-compose --profile monitoring up -d

# Verify all services
docker-compose ps
for service in engram prometheus grafana; do
  docker-compose exec $service sh -c 'echo Service $0 is running' $service
done

# Test health endpoints
curl -s http://localhost:7432/health | jq .
curl -s http://localhost:7432/api/v1/system/health | jq .
curl -s http://localhost:9090/-/healthy  # Prometheus
curl -s http://localhost:3000/api/health  # Grafana

# Test graceful shutdown
docker-compose stop -t 30
docker-compose logs engram | grep "Graceful shutdown complete"

# Cleanup
docker-compose down -v
```

### Kubernetes StatefulSet Testing
```bash
# Apply with dry-run first
kubectl apply -f deployments/kubernetes/ --dry-run=client

# Deploy StatefulSet
kubectl apply -f deployments/kubernetes/

# Monitor pod creation
kubectl get pods -l app=engram -w

# Verify StatefulSet
kubectl get statefulset engram -o yaml | grep "readyReplicas: 1"

# Check PVC creation
kubectl get pvc -l app=engram

# Test stable network identity
kubectl exec engram-0 -- hostname  # Should be "engram-0"

# Port forward for testing
kubectl port-forward pod/engram-0 7432:7432 &
PF_PID=$!

# Health and functionality tests
curl http://localhost:7432/api/v1/system/health | jq .

# Test persistence across pod restart
curl -X POST http://localhost:7432/api/v1/memories/remember \
  -H "Content-Type: application/json" \
  -d '{"content": "persistent-test", "confidence": 0.95}'

kubectl delete pod engram-0
kubectl wait --for=condition=ready pod/engram-0 --timeout=60s

# Verify data persisted
curl http://localhost:7432/api/v1/memories/search?q=persistent-test | jq .

# Cleanup
kill $PF_PID
kubectl delete -f deployments/kubernetes/
```

### Helm Production Testing
```bash
# Lint chart
helm lint deployments/helm/engram

# Test with custom values
cat > test-values.yaml <<EOF
statefulset:
  replicaCount: 1
resources:
  requests:
    cpu: 2000m
    memory: 4Gi
persistence:
  size: 50Gi
  storageClass: fast-ssd
monitoring:
  prometheus:
    enabled: true
    serviceMonitor:
      enabled: true
EOF

# Dry run
helm install engram deployments/helm/engram \
  -f test-values.yaml \
  --dry-run --debug

# Install with wait
helm install engram deployments/helm/engram \
  -f test-values.yaml \
  --wait --timeout 5m

# Verify installation
helm test engram
kubectl get all -l app.kubernetes.io/name=engram

# Test upgrade
helm upgrade engram deployments/helm/engram \
  --set image.tag=0.1.1 \
  --wait

# Verify rolling update didn't lose data
kubectl port-forward service/engram 7432:7432 &
curl http://localhost:7432/api/v1/system/health

# Rollback test
helm rollback engram 1

# Cleanup
helm uninstall engram --wait
```

### Load Testing and Benchmarking
```bash
# Use engram's built-in benchmark
docker run --rm \
  -v $(pwd)/bench-results:/results \
  engram:test benchmark \
  --operations 10000 \
  --concurrent 10 \
  --output /results/benchmark.json

# Analyze results
cat bench-results/benchmark.json | jq '.summary'

# Memory stress test
docker run -d --name engram-stress \
  --memory="2g" \
  --memory-swap="2g" \
  engram:test

# Monitor under load
while true; do
  curl -X POST http://localhost:7432/api/v1/memories/remember \
    -H "Content-Type: application/json" \
    -d '{"content": "stress-test", "confidence": 0.9}' &
done &
STRESS_PID=$!

docker stats engram-stress

# Stop stress test
kill $STRESS_PID
docker stop engram-stress && docker rm engram-stress
```

## Documentation Requirements

### /docs/operations/production-deployment.md

Must include sections with Context→Action→Verification format:

**1. Prerequisites**
- Docker 24+ or Kubernetes 1.28+
- Sufficient resources (2 CPU, 4GB RAM minimum)
- Network access for image pulls

**2. Docker Deployment**
- Build from source
- Pull from registry (future)
- Configure volumes and environment
- Verify health

**3. docker-compose Deployment**
- Configuration overview
- Launch stack
- Verify all services
- Access monitoring (if enabled)

**4. Kubernetes Deployment**
- Prerequisites (kubectl, cluster access)
- Apply manifests
- Verify pod status
- Access service (port-forward or LoadBalancer)

**5. Helm Deployment**
- Add Helm repo (future)
- Install with custom values
- Verify deployment
- Upgrade procedures

**6. Bare-Metal Deployment**
- Build binary
- System service setup (systemd)
- Configuration
- Start and verify

**7. Configuration**
- Data directory setup
- Port configuration
- Resource limits
- Logging configuration

**8. Verification**
- Health check endpoints
- Store test memory
- Recall test memory
- Monitor logs

**9. Troubleshooting**
- Container won't start
- Health check failing
- Port conflicts
- Permission errors

**10. Next Steps**
- Setup monitoring (Task 003)
- Configure backups (Task 002)
- Performance tuning (Task 004)

## Production Deployment Optimizations

### NUMA-Aware Deployment
For systems with multiple NUMA nodes, pin containers to specific nodes:

```yaml
# Kubernetes NUMA affinity
apiVersion: v1
kind: Pod
spec:
  nodeSelector:
    topology.kubernetes.io/zone: numa-0
  containers:
  - name: engram
    resources:
      limits:
        memory: "4Gi"
        cpu: "2000m"
        # Pin to NUMA node 0 CPUs (0-15)
    env:
    - name: ENGRAM_CPU_AFFINITY
      value: "0-15"
    - name: ENGRAM_NUMA_NODE
      value: "0"
```

### Memory Allocation Tuning
Configure mimalloc for optimal container performance:

```dockerfile
# Runtime optimization
ENV MIMALLOC_LARGE_OS_PAGES=1
ENV MIMALLOC_RESERVE_HUGE_OS_PAGES=4
ENV MIMALLOC_EAGER_COMMIT=0
ENV MIMALLOC_PAGE_RESET=1
ENV MIMALLOC_USE_NUMA_NODES=4
```

### Cache Line Optimization
Ensure data structures align to 64-byte boundaries:

```yaml
# Performance environment variables
ENGRAM_CACHE_LINE_SIZE: "64"
ENGRAM_PREFETCH_DISTANCE: "256"
ENGRAM_VECTOR_WIDTH: "256"  # AVX2
```

### Storage Performance
Configure storage for optimal I/O patterns:

```yaml
# Kubernetes StorageClass for NVMe
apiVersion: storage.k8s.io/v1
kind: StorageClass
metadata:
  name: fast-nvme
provisioner: kubernetes.io/aws-ebs
parameters:
  type: gp3
  iops: "16000"
  throughput: "1000"  # MB/s
  fsType: ext4
mountOptions:
  - noatime
  - nodiratime
  - nobarrier
```

### Network Optimization
Configure for low-latency communication:

```yaml
# docker-compose network tuning
networks:
  engram-net:
    driver: bridge
    driver_opts:
      com.docker.network.driver.mtu: "9000"  # Jumbo frames
    ipam:
      config:
        - subnet: 172.28.0.0/16
```

### Graceful Shutdown and Signal Handling
Ensure clean shutdown to prevent data corruption:

```dockerfile
# Dockerfile signal configuration
STOPSIGNAL SIGTERM

# Entrypoint wrapper for proper signal forwarding
COPY --from=builder /build/scripts/docker-entrypoint.sh /
ENTRYPOINT ["/docker-entrypoint.sh"]
```

```bash
#!/bin/sh
# docker-entrypoint.sh - Ensures proper signal handling
set -e

# Trap SIGTERM and forward to engram process
trap 'kill -TERM $PID' TERM INT

# Start engram in background
/usr/local/bin/engram "$@" &
PID=$!

# Wait for process to complete
wait $PID
EXITCODE=$?

# Ensure flush completes
sleep 2

exit $EXITCODE
```

```yaml
# Kubernetes graceful termination
spec:
  terminationGracePeriodSeconds: 30
  containers:
  - name: engram
    lifecycle:
      preStop:
        exec:
          command:
          - /bin/sh
          - -c
          - |
            # Signal engram to start shutdown
            kill -TERM 1
            # Wait for flush operations
            sleep 5
            # Verify clean shutdown
            test -f /tmp/shutdown.complete || exit 1
```

## Acceptance Criteria

**Container Build:**
- [ ] Dockerfile builds successfully in <3 minutes with cache
- [ ] Runtime image <60MB total (6MB base + ~50MB binary)
- [ ] Container runs as nonroot user (uid 65534)
- [ ] Health check passes within 10 seconds of startup
- [ ] Zero HIGH or CRITICAL vulnerabilities (Trivy scan)
- [ ] Distroless base prevents shell access

**docker-compose:**
- [ ] Stack starts with single `docker-compose up -d`
- [ ] All services reach healthy state in <20 seconds
- [ ] Data persists after `docker-compose down` (without -v)
- [ ] Configuration changes apply on restart
- [ ] Graceful shutdown completes in <30 seconds
- [ ] Resource limits properly enforced

**Kubernetes:**
- [ ] StatefulSet deploys without errors
- [ ] Pod reaches Running state in <20 seconds
- [ ] All probes (startup/liveness/readiness) succeed
- [ ] Persistent volume correctly mounted and writable
- [ ] Service accessible via port-forward
- [ ] Data survives pod deletion and recreation
- [ ] Security context enforces read-only root filesystem

**Helm:**
- [ ] Chart passes `helm lint` with no warnings
- [ ] Installation completes with `--wait --timeout 5m`
- [ ] All values properly template into manifests
- [ ] Upgrade preserves data and configuration
- [ ] Rollback restores previous version cleanly
- [ ] Uninstall removes all resources except PVCs
- [ ] ServiceMonitor creates Prometheus scrape config

**Performance:**
- [ ] Cold start to ready in <10 seconds
- [ ] Memory usage <2GB for 10K memories
- [ ] Handles 1000 concurrent connections
- [ ] Benchmark shows >5000 ops/sec single-threaded
- [ ] CPU usage scales linearly with load
- [ ] No memory leaks under sustained load

**Documentation:**
- [ ] External operator deploys Docker in <10 minutes
- [ ] External operator deploys Kubernetes in <20 minutes
- [ ] All commands are copy-paste ready and tested
- [ ] Troubleshooting covers top 10 common issues
- [ ] Performance tuning guide for production
- [ ] Security hardening checklist included
- [ ] Monitoring setup documented end-to-end

**Security:**
- [ ] No secrets or credentials in images
- [ ] Containers run with minimal capabilities
- [ ] Read-only root filesystem enforced
- [ ] Network policies restrict traffic
- [ ] TLS termination properly configured
- [ ] RBAC limits service account permissions
- [ ] Security scanning integrated in CI/CD

## Follow-Up Tasks

- Task 002: Add backup CronJob to Kubernetes manifests
- Task 003: Add monitoring sidecar containers
- Task 008: Add TLS configuration to deployments
- Milestone 14: Multi-node distributed deployment configurations
