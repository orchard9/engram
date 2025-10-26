# Zig Kernels - Rollback Procedures

## Overview

This document provides step-by-step procedures for rolling back Zig performance kernels to Rust-only implementations. Rollback strategies range from emergency (immediate failover) to gradual (controlled degradation).

Use this guide when:

- Zig kernels cause production incidents (crashes, correctness issues)
- Performance degrades below acceptable thresholds
- Arena overflows cannot be resolved through tuning
- Planned maintenance requires disabling Zig kernels

## Decision Matrix

Choose rollback strategy based on issue severity:

| Severity | Symptoms | Rollback Strategy | Timeline |
|----------|----------|-------------------|----------|
| **Critical** | Crashes, data corruption, incorrect results | Emergency Rollback | Immediate (5-10 min) |
| **High** | Performance regression >20%, high error rate | Gradual Rollback | Within 1 hour |
| **Medium** | Arena overflows <5%, latency spikes | Configuration Tuning | Within 1 day |
| **Low** | Warning logs, minor performance variance | Monitor and Investigate | Within 1 week |

## Emergency Rollback

Use this procedure for critical issues requiring immediate failover to Rust baseline.

### Prerequisites

- Access to production deployment infrastructure
- Ability to rebuild and deploy binaries
- Rollback window approval (if required)

### Step 1: Stop the Service

```bash
# systemd (Linux)
sudo systemctl stop engram

# macOS (launchd)
sudo launchctl stop com.orchard9.engram

# Docker
docker stop engram-container

# Kubernetes
kubectl scale deployment engram --replicas=0
```

**Recovery Time Objective (RTO)**: Service offline for 5-10 minutes during rebuild and restart.

### Step 2: Rebuild Without Zig Feature

```bash
# Navigate to project directory
cd /opt/engram  # Or your deployment path

# Clean previous build artifacts
cargo clean

# Rebuild Rust-only binary (NO --features zig-kernels)
cargo build --release

# Verify Zig symbols are absent
nm target/release/engram-cli | grep engram_vector_similarity
# Should output: (no results)

# Alternatively, use objdump to verify
objdump -T target/release/engram-cli | grep zig
# Should output: (no results)
```

### Step 3: Deploy Rolled-Back Binary

```bash
# Backup current binary
sudo cp /usr/local/bin/engram-cli /usr/local/bin/engram-cli.zig-backup

# Deploy new binary
sudo cp target/release/engram-cli /usr/local/bin/engram-cli

# Verify binary is updated
/usr/local/bin/engram-cli --version
```

### Step 4: Restart Service

```bash
# systemd (Linux)
sudo systemctl start engram
sudo systemctl status engram  # Verify running

# macOS (launchd)
sudo launchctl start com.orchard9.engram

# Docker
docker start engram-container

# Kubernetes
kubectl scale deployment engram --replicas=3  # Or original replica count
```

### Step 5: Verify Rollback

```bash
# Check logs for Zig kernel messages (should be absent)
journalctl -u engram --since "5 minutes ago" | grep -i "zig"
# Should output: (no results)

# Run health check
curl http://localhost:7432/health
# Expected: {"status": "healthy"}

# Verify Rust baseline is active
curl http://localhost:7432/internal/zig/status
# Expected: {"zig_kernels_enabled": false}
```

### Step 6: Monitor Baseline Performance

```bash
# Monitor performance returns to Rust baseline
cargo bench --bench regression

# Check key metrics
curl http://localhost:7432/metrics | grep kernel_duration
```

**Expected**: Performance returns to pre-Zig baseline (slower but stable).

### Step 7: Incident Response

1. **Document the issue**:
   - Symptoms observed (crashes, errors, performance degradation)
   - Logs captured during incident
   - Metrics snapshots before/during/after rollback
   - Timeline of events

2. **File incident report**:
   - Create GitHub issue with `incident` and `zig-kernels` labels
   - Include reproduction steps if possible
   - Attach logs, metrics, and configuration

3. **Notify team**:
   - Post in incident response channel
   - Tag on-call engineer and Engram team lead
   - Schedule post-mortem within 48 hours

4. **Post-mortem actions**:
   - Root cause analysis
   - Fix validation in staging
   - Re-deployment plan with rollback contingency

## Gradual Rollback

Use this procedure for non-critical issues where controlled degradation is acceptable.

### Strategy 1: Canary Rollback

Gradually reduce Zig kernel usage across instance fleet:

#### Phase 1: Roll Back Canary (5% of traffic)

```bash
# On canary instances
./scripts/deploy_rust_only.sh canary

# Monitor for 1 hour
# Check error rates, latency, throughput
```

#### Phase 2: Roll Back 25% of Fleet

```bash
# Expand to 25% of production instances
for instance in prod-1 prod-2 prod-3; do
  ssh $instance "./scripts/deploy_rust_only.sh"
done

# Monitor for 30 minutes
```

#### Phase 3: Roll Back Remaining 75%

```bash
# Complete rollback across fleet
./scripts/deploy_rust_only_all.sh

# Verify all instances are Rust-only
```

**Total Rollback Time**: 2-3 hours with monitoring windows

### Strategy 2: Traffic Shifting (Load Balancer)

If your deployment uses load balancers with traffic control:

```bash
# Route 10% traffic to Rust-only instances
lb-control set-weight zig-instances:90 rust-instances:10

# Monitor for 30 minutes, then increase
lb-control set-weight zig-instances:50 rust-instances:50

# Continue gradual shift
lb-control set-weight zig-instances:0 rust-instances:100

# Decommission Zig instances
```

### Strategy 3: Feature Flag Toggle

If runtime feature flags are implemented:

```rust
// Disable Zig kernels via configuration
curl -X POST http://localhost:7432/config/set \
  -H "Content-Type: application/json" \
  -d '{"zig_kernels_enabled": false}'

// No restart required - takes effect on next kernel invocation
```

**Advantage**: Instant rollback without rebuild or restart.
**Requirement**: Feature flag support must be implemented.

## Configuration-Based Mitigation

Before full rollback, attempt tuning to resolve issues:

### Arena Overflow Resolution

If rollback is due to arena overflows:

```bash
# Increase arena size temporarily
export ENGRAM_ARENA_SIZE=8388608  # 8MB

# Restart service
sudo systemctl restart engram

# Monitor overflow rate
watch -n 5 'curl -s http://localhost:7432/internal/zig/arena_stats | jq .total_overflows'
```

If overflows persist, proceed with rollback.

### Performance Regression Mitigation

If performance degrades:

```bash
# Reduce thread count to decrease contention
export RAYON_NUM_THREADS=4

# Restart and benchmark
sudo systemctl restart engram
cargo bench --bench regression

# If performance improves, keep reduced threads
# Otherwise, proceed with rollback
```

## Verification After Rollback

### Functional Verification

Run integration tests against rolled-back system:

```bash
# Run full integration test suite
cargo test --test integration --release

# Run HTTP API smoke tests
./scripts/smoke_test.sh

# Expected: All tests pass
```

### Performance Verification

```bash
# Benchmark Rust baseline performance
cargo bench --bench baseline_performance

# Verify performance matches historical baseline
# (should be within 5% of pre-Zig introduction)
```

### Error Rate Verification

```bash
# Check error logs from past 10 minutes
journalctl -u engram --since "10 minutes ago" | grep -i error
# Should see NO Zig-related errors

# Monitor error rate metrics
curl http://localhost:7432/metrics | grep error_rate
# Should return to baseline levels
```

## Common Rollback Scenarios

### Scenario 1: Arena Overflows Causing Errors

**Symptoms**:
- High error rate in logs (OutOfMemory errors)
- `total_overflows` metric increasing rapidly
- Client-visible failures

**Root Cause**: Arena capacity insufficient for workload

**Rollback Decision**:
1. **First**: Attempt increasing arena size (`ENGRAM_ARENA_SIZE=8388608`)
2. **If errors persist**: Perform **Gradual Rollback** within 1 hour
3. **If critical**: Perform **Emergency Rollback** immediately

**Follow-up**:
- Analyze peak memory usage during failed workload
- Calculate required arena size: `peak_usage * 1.5`
- Re-enable Zig kernels with larger arenas in staging
- Validate before re-deploying to production

### Scenario 2: Performance Regression

**Symptoms**:
- Increased latency (p50, p99 above thresholds)
- Reduced throughput
- Regression benchmarks fail

**Root Cause**: Possible causes:
- SIMD not enabled (CPU lacks AVX2/NEON)
- Thread contention or lock thrashing
- Arena allocations slower than expected

**Rollback Decision**:
1. **First**: Verify CPU features and thread count
2. **If no improvement**: Perform **Gradual Rollback** within 4 hours
3. **If impact >20%**: Perform **Emergency Rollback** within 1 hour

**Follow-up**:
- Profile with perf/Instruments to identify bottleneck
- Test on identical hardware in staging
- Validate performance improvements before re-deployment

### Scenario 3: Numerical Divergence

**Symptoms**:
- Differential tests fail in production
- Incorrect results compared to Rust baseline
- Data integrity concerns

**Root Cause**: Floating-point precision issue or FFI bug

**Rollback Decision**:
- **Immediate Emergency Rollback** (correctness issue)
- Treat as critical incident

**Follow-up**:
- Capture failing test cases and inputs
- File critical bug report with reproduction
- Investigate numerical accuracy in Zig kernels
- Re-enable only after fix validated in staging

### Scenario 4: Build Failures After Upgrade

**Symptoms**:
- `cargo build --features zig-kernels` fails
- Zig compiler errors during build
- ABI compatibility issues

**Root Cause**: Zig version incompatibility or build system changes

**Rollback Decision**:
- **Build Rust-only version** immediately
- Deploy to avoid production outage

**Follow-up**:
- Pin Zig version to 0.13.0
- Update build scripts to verify Zig version
- Test builds in CI before merging

## Rollback Testing

Regularly test rollback procedures to ensure they work when needed:

### Monthly Rollback Drill

```bash
# 1. Deploy with Zig kernels to staging
./scripts/deploy_with_zig.sh staging

# 2. Let it run for 5 minutes (simulate production)
sleep 300

# 3. Perform emergency rollback
./scripts/rollback_to_rust.sh staging

# 4. Verify system health
./scripts/health_check.sh staging

# Expected: Staging returns to healthy state within 10 minutes
```

### Rollback Checklist

Use this checklist during drills:

- [ ] Stop service completes without errors
- [ ] Rebuild completes in <5 minutes
- [ ] Binary deployment succeeds
- [ ] Service restart succeeds
- [ ] Health checks pass
- [ ] Zig symbols absent from binary (verified)
- [ ] Logs show no Zig-related messages
- [ ] Performance returns to baseline
- [ ] Integration tests pass
- [ ] Total rollback time <10 minutes

## Post-Rollback Actions

After completing rollback:

### 1. Root Cause Analysis

- Identify why rollback was necessary
- Document specific failure mode
- Determine if issue is reproducible

### 2. Fix and Validate

- Address root cause in development environment
- Write regression test to prevent recurrence
- Validate fix in staging with production-like workload
- Run extended soak test (24+ hours)

### 3. Re-Deployment Plan

When re-enabling Zig kernels:

- Start with canary deployment (5% traffic)
- Monitor for 48 hours before expanding
- Have rollback plan ready (this document)
- Communicate plan to operations team

### 4. Documentation Update

- Update this document with lessons learned
- Add new troubleshooting guidance
- Document configuration changes
- Update deployment runbooks

## Emergency Contacts

For rollback assistance:

- **On-call Engineer**: Check on-call rotation
- **Engram Team Lead**: Contact via team channel
- **Emergency Escalation**: Page engineering manager for critical incidents

## Rollback Scripts

### /scripts/rollback_to_rust.sh

```bash
#!/bin/bash
# Emergency rollback to Rust-only implementation
set -euo pipefail

ENVIRONMENT=${1:-production}

echo "Rolling back to Rust-only implementation for $ENVIRONMENT..."

# Stop service
systemctl stop engram

# Rebuild without Zig
cd /opt/engram
cargo clean
cargo build --release

# Deploy
cp target/release/engram-cli /usr/local/bin/engram-cli

# Restart
systemctl start engram

# Verify
sleep 5
systemctl status engram

echo "Rollback complete. Verify health:"
echo "  curl http://localhost:7432/health"
```

### /scripts/deploy_rust_only.sh

```bash
#!/bin/bash
# Deploy Rust-only build (for gradual rollback)
set -euo pipefail

INSTANCE=${1:-$(hostname)}

echo "Deploying Rust-only build to $INSTANCE..."

# Build on CI/build server (not on production instance)
ssh build-server "cd /builds/engram && cargo build --release"

# Copy to instance
scp build-server:/builds/engram/target/release/engram-cli $INSTANCE:/tmp/

# Deploy on instance
ssh $INSTANCE "sudo systemctl stop engram && \
  sudo cp /tmp/engram-cli /usr/local/bin/engram-cli && \
  sudo systemctl start engram"

echo "Deployment complete."
```

## See Also

- [Operations Guide](./zig_performance_kernels.md) - Deployment and configuration
- [Architecture Documentation](../internal/zig_architecture.md) - Internal design
- [Performance Regression Guide](../internal/performance_regression_guide.md) - Benchmarking
