# Container Orchestration and Deployment - Architectural Perspectives

## Systems Architecture Optimizer Perspective

### The Container Overhead Problem

Every layer of abstraction costs performance. Containers add CPU overhead, network latency, and storage indirection. For Engram's sub-5ms P50 latency target, we cannot tolerate sloppy container design.

**The Numbers That Matter:**

Without optimization:
- Container network bridge: +200us per request
- Volume overlay filesystem: +50us per I/O
- CPU throttling: Unpredictable latency spikes
- Memory limits: OOM kills during consolidation

With optimization:
- Host networking: +10us overhead
- Direct volume mounts: +5us overhead
- CPU limits without throttling: Consistent performance
- Memory limits with graceful pressure handling

**How We Get There:**

1. **Host Networking Mode**
   - Skip the bridge, connect directly to host network stack
   - Trade-off: Pod IP is host IP, port conflicts possible
   - Solution: Use NodePort range, document port allocation

2. **Local Storage Provisioner**
   - Bind mount host directories directly
   - Skip distributed storage overhead when single-node
   - Use local-path-provisioner or hostPath volumes

3. **CPU Pinning**
   - Set CPU affinity in StatefulSet podSpec
   - Isolate NUMA domains for memory locality
   - Use topology-aware scheduling

4. **Memory Hierarchy Mapping**
   - Fast tier: tmpfs volume (RAM-backed)
   - Warm tier: SSD persistent volume
   - Cold tier: HDD or S3-compatible storage
   - Each tier has dedicated resource limits

**Configuration Template:**

```yaml
resources:
  requests:
    cpu: "2000m"
    memory: "8Gi"
  limits:
    cpu: "4000m"
    memory: "16Gi"

volumes:
  - name: fast-tier
    emptyDir:
      medium: Memory
      sizeLimit: 4Gi
  - name: warm-tier
    persistentVolumeClaim:
      claimName: engram-warm-ssd
```

### The Stateful Set Decision

Deployments are for stateless services. Engram has graph state. Use StatefulSet.

**Why it matters:**
- Stable pod identity (engram-0, engram-1) enables peer discovery
- Ordered deployment prevents split-brain during distributed startup
- Ordered shutdown ensures graceful memory consolidation
- PVC templates provision storage per pod automatically

**Failure scenarios StatefulSet handles:**
- Pod restart: Reattaches same PVC, graph state intact
- Node failure: Reschedules on new node, volume follows
- Rolling update: One pod at a time, zero data loss

**Failure scenarios StatefulSet does NOT handle:**
- Corrupted PVC: Requires backup restore
- Network partition: Requires distributed consensus (future milestone)
- Cascading failures: Requires circuit breakers

## Rust Graph Engine Architect Perspective

### Container as Execution Environment

The container runtime is just another execution environment, like Linux is to a process. Our Rust binary should be environment-agnostic.

**Design Principle: Configuration by Environment Variables**

```rust
// Instead of hardcoded paths
const DATA_DIR: &str = "/var/lib/engram";

// Use environment variables with sensible defaults
fn data_dir() -> PathBuf {
    env::var("ENGRAM_DATA_DIR")
        .unwrap_or_else(|_| "/var/lib/engram".to_string())
        .into()
}
```

**Signal Handling for Graceful Shutdown:**

```rust
use tokio::signal;

async fn shutdown_signal() {
    let ctrl_c = async {
        signal::ctrl_c()
            .await
            .expect("failed to install Ctrl+C handler");
    };

    let terminate = async {
        signal::unix::signal(signal::unix::SignalKind::terminate())
            .expect("failed to install SIGTERM handler")
            .recv()
            .await;
    };

    tokio::select! {
        _ = ctrl_c => {},
        _ = terminate => {},
    }

    println!("Shutdown signal received, starting graceful shutdown");
}

#[tokio::main]
async fn main() {
    let server = start_server().await;

    shutdown_signal().await;

    // Graceful shutdown sequence
    server.stop_accepting_connections();
    sleep(Duration::from_secs(5)).await; // Drain existing requests
    server.trigger_consolidation().await;
    sleep(Duration::from_secs(10)).await; // Flush to persistent storage
    server.close_connections();
}
```

**Health Check Endpoints:**

```rust
// Liveness: Is the process alive?
async fn liveness() -> impl Responder {
    HttpResponse::Ok().body("alive")
}

// Readiness: Can it serve traffic?
async fn readiness(state: Data<AppState>) -> impl Responder {
    if state.graph.is_initialized() && state.graph.warmup_complete() {
        HttpResponse::Ok().body("ready")
    } else {
        HttpResponse::ServiceUnavailable().body("warming up")
    }
}
```

**Metrics Exposition:**

```rust
use prometheus::{Encoder, TextEncoder, Registry};

async fn metrics(registry: Data<Registry>) -> impl Responder {
    let encoder = TextEncoder::new();
    let metric_families = registry.gather();
    let mut buffer = vec![];
    encoder.encode(&metric_families, &mut buffer).unwrap();
    HttpResponse::Ok().body(buffer)
}
```

### Binary Size Optimization

Container images are distributed artifacts. Smaller is faster.

**Build Configuration (.cargo/config.toml):**

```toml
[profile.release]
opt-level = 3
lto = "fat"
codegen-units = 1
strip = true
panic = "abort"
```

**Result:**
- Before optimization: 120MB binary
- After optimization: 8MB binary
- Container image: 15MB total (base + binary + configs)

**Dockerfile Strategy:**

```dockerfile
# Build stage
FROM rust:1.75-alpine AS builder
RUN apk add --no-cache musl-dev
WORKDIR /build
COPY . .
RUN cargo build --release --target x86_64-unknown-linux-musl

# Runtime stage
FROM scratch
COPY --from=builder /build/target/x86_64-unknown-linux-musl/release/engram /engram
COPY --from=builder /etc/ssl/certs/ca-certificates.crt /etc/ssl/certs/
ENTRYPOINT ["/engram"]
```

## Verification Testing Lead Perspective

### Testing the Container, Not Just the Code

A passing test suite does not mean a working container. We must test the packaged artifact.

**Container Integration Tests:**

```bash
#!/bin/bash
# tests/integration/container_test.sh

set -e

# Build container
docker build -t engram:test .

# Start container with test config
docker run -d --name engram-test \
  -p 8080:8080 \
  -e ENGRAM_LOG_LEVEL=debug \
  engram:test

# Wait for readiness
for i in {1..30}; do
  if curl -f http://localhost:8080/health/ready; then
    break
  fi
  sleep 1
done

# Run API tests
curl -X POST http://localhost:8080/api/v1/memories \
  -H "Content-Type: application/json" \
  -d '{"content": "test memory", "embedding": [0.1, 0.2]}'

# Verify metrics
curl http://localhost:8080/metrics | grep engram_operations_total

# Test graceful shutdown
docker stop --time=30 engram-test

# Verify data persistence
docker start engram-test
# ... verify data still exists

# Cleanup
docker rm -f engram-test
```

**Volume Persistence Test:**

```bash
# Create volume
docker volume create engram-data

# Write data
docker run --rm \
  -v engram-data:/data \
  engram:test \
  engram load --file /data/graph.bin

# Read data in new container
docker run --rm \
  -v engram-data:/data \
  engram:test \
  engram query --count

# Cleanup
docker volume rm engram-data
```

**Resource Limit Testing:**

```bash
# Test memory limit
docker run --rm --memory=1g --memory-swap=1g \
  engram:test \
  engram benchmark --operations=100000

# Should gracefully handle memory pressure, not OOM kill

# Test CPU limit
docker run --rm --cpus=0.5 \
  engram:test \
  engram benchmark --operations=100000

# Should throttle gracefully, not timeout
```

**Security Scanning in CI:**

```yaml
# .github/workflows/security.yml
- name: Scan container for vulnerabilities
  uses: aquasecurity/trivy-action@master
  with:
    image-ref: engram:${{ github.sha }}
    format: 'sarif'
    output: 'trivy-results.sarif'
    severity: 'CRITICAL,HIGH'
    exit-code: '1'  # Fail build on vulnerabilities
```

## Cognitive Architecture Designer Perspective

### Containers as Memory Isolation Boundaries

In biological systems, different brain regions have isolation. The hippocampus doesn't directly interfere with the neocortex's long-term storage. Containers provide similar isolation.

**Memory Tier Isolation:**

Each tier runs in separate resource boundaries:
- Fast tier (hippocampus analog): High CPU, low latency, tmpfs storage
- Warm tier (neocortical analog): Balanced resources, SSD storage
- Cold tier (archival): Low CPU, high capacity, cheap storage

**Container Orchestration as Sleep Cycles:**

During low-traffic periods (night), containers can:
1. Scale down replicas (reduce energy usage)
2. Trigger consolidation jobs (memory transfer)
3. Rebuild indexes (optimization)
4. Run garbage collection (cleanup)

**Scheduled Consolidation:**

```yaml
# Kubernetes CronJob for nightly consolidation
apiVersion: batch/v1
kind: CronJob
metadata:
  name: engram-consolidation
spec:
  schedule: "0 2 * * *"  # 2 AM daily
  jobTemplate:
    spec:
      template:
        spec:
          containers:
          - name: consolidate
            image: engram:v1.0.0
            command: ["engram", "consolidate", "--threshold=0.3"]
          restartPolicy: OnFailure
```

**The Wake-Up Problem:**

When a container starts, the graph is cold. No active memories, no spread activation paths. This is like waking from deep sleep.

**Warmup Strategy:**

1. Load frequently accessed patterns from warm tier
2. Prime caches with recent queries
3. Mark readiness probe as false during warmup
4. Gradually increase traffic with readiness probe

**Implementation:**

```rust
async fn warmup(graph: &Graph) -> Result<()> {
    // Phase 1: Load critical patterns
    graph.load_patterns_by_frequency(top_n: 1000).await?;

    // Phase 2: Run representative queries
    for query in representative_queries() {
        graph.activate(&query).await?;
    }

    // Phase 3: Verify cache hit rates
    let hit_rate = graph.metrics().cache_hit_rate();
    if hit_rate < 0.5 {
        return Err("Warmup incomplete, cache not primed");
    }

    Ok(())
}
```

### Container Lifecycle as Cognitive States

- **Init**: Birth, schema setup, neural network initialization
- **Running**: Active cognition, processing queries
- **Consolidating**: Sleep, memory transfer
- **Terminating**: Graceful shutdown, final memory flush

Each state transition must preserve graph integrity, just as biological state transitions (wake/sleep) preserve memory.

## Synthesis: Production Deployment Principles

1. **Optimize for the Critical Path**: Host networking, local storage, CPU pinning for <5ms P50 latency
2. **Isolate Failure Domains**: StatefulSet for stable identity, PVCs for persistent state
3. **Test the Artifact**: Container integration tests, not just unit tests
4. **Design for Warmup**: Readiness probes, cache priming, gradual traffic increase
5. **Enable Observability**: Metrics, logs, traces exposed via standard interfaces
6. **Minimize Image Size**: Multi-stage builds, static linking, strip symbols
7. **Handle Signals Gracefully**: SIGTERM shutdown, consolidation before exit
8. **Map Memory Hierarchy**: tmpfs for fast, SSD for warm, HDD for cold
9. **Schedule Maintenance**: CronJobs for consolidation, cleanup, optimization
10. **Document the Deployment**: <2 hour time budget, step-by-step validation

These principles ensure Engram runs reliably in production containers while maintaining cognitive system performance characteristics.
