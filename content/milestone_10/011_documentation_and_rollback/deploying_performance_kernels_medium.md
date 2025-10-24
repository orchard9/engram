# Deploying Performance Kernels (And Rolling Them Back)

Documentation is insurance. You write it hoping you'll never need it. But when something breaks at 3 AM, you're grateful it exists.

For Zig performance kernels, we wrote three documents:

1. **Operations Guide:** How to deploy and configure
2. **Rollback Procedures:** How to recover from failures
3. **Architecture Documentation:** How it works internally

This isn't just documentation for documentation's sake. It's the difference between 5-minute recovery and 2-hour debugging sessions.

## The Deployment Checklist

Before deploying Zig kernels to production:

```markdown
- [ ] Zig 0.13.0 installed on all nodes
- [ ] Build with --features zig-kernels succeeds
- [ ] All tests pass (unit, differential, integration)
- [ ] Regression benchmarks show expected improvements
- [ ] Arena size configured for workload (2MB default)
- [ ] Monitoring configured (arena stats, kernel timing)
- [ ] Rollback procedure documented and tested
- [ ] Gradual rollout plan ready (canary → production)
```

Each checkbox prevents a class of production incidents.

## Configuration: Arena Sizing

Different workloads need different arena sizes:

| Workload | Embedding Dim | Candidates | Arena Size |
|----------|--------------|------------|------------|
| Light | 384 | 100 | 1 MB |
| Medium | 768 | 1000 | 2 MB |
| Heavy | 1536 | 5000 | 8 MB |

Configured via environment variable:
```bash
export ENGRAM_ARENA_SIZE=2097152  # 2 MB in bytes
```

Wrong arena size = overflows or memory waste. Documentation includes sizing heuristics based on real workloads.

## Monitoring: Know When Things Break

Three metrics matter:

**1. Arena Overflow Rate**
```rust
let stats = zig_kernels::get_arena_stats();
let overflow_rate = stats.total_overflows as f64 / stats.total_resets as f64;

if overflow_rate > 0.01 {  // >1% overflow
    alert!("Arena size too small, increase ENGRAM_ARENA_SIZE");
}
```

**2. Kernel Execution Time**
```rust
let start = Instant::now();
let scores = batch_cosine_similarity(&query, &candidates);
let duration = start.elapsed();

metrics.record_kernel_time("vector_similarity", duration);

if duration > Duration::from_micros(3) {  // Expected 1.7μs
    alert!("Vector similarity degraded");
}
```

**3. Error Rate**
```rust
if error_rate > baseline_error_rate * 1.005 {  // 0.5% increase
    alert!("Zig kernels may be causing errors");
}
```

Documentation specifies alert thresholds and remediation steps.

## Rollback: The 5-Minute Recovery

When things go wrong, operators need clear emergency procedures:

### Emergency Rollback (RTO: 5 minutes)

```bash
# 1. Stop service
systemctl stop engram

# 2. Rebuild without Zig kernels
cargo build --release
# Note: Omit --features zig-kernels

# 3. Deploy new binary
cp target/release/engram /usr/local/bin/engram

# 4. Restart service
systemctl start engram

# 5. Verify
curl http://localhost:8080/health
journalctl -u engram --since "1 minute ago" | grep ERROR
```

No database migrations. No state cleanup. Just rebuild and restart.

RTO (Recovery Time Objective): <5 minutes from decision to rollback.

### Gradual Rollback

For non-critical issues:

1. Route 90% traffic to Rust-only instances
2. Monitor for 1 hour
3. If stable, route 100% traffic
4. Decommission Zig-enabled instances

Gives time to investigate without pressure.

## Rollback Scenarios: When to Roll Back

Documentation includes decision matrix:

| Issue Severity | Action | Timeline |
|---------------|--------|----------|
| Critical (correctness) | Emergency rollback | Immediate |
| High (performance regression) | Gradual rollback | Within 1 hour |
| Medium (errors <1%) | Investigate, then decide | Within 1 day |
| Low (warnings only) | Monitor, tune config | Within 1 week |

### Scenario 1: Arena Overflows

**Symptoms:** Error rate increase, arena overflow warnings in logs

**First Response:** Increase arena size
```bash
export ENGRAM_ARENA_SIZE=4194304  # Double from 2MB to 4MB
systemctl restart engram
```

**If errors persist:** Emergency rollback

### Scenario 2: Performance Regression

**Symptoms:** Latency p99 increased 15%

**First Response:** Verify with regression benchmarks
```bash
cargo bench --bench regression
```

**If confirmed:** Gradual rollback while investigating root cause

### Scenario 3: Numerical Divergence

**Symptoms:** Incorrect retrieval results, differential test failures

**First Response:** Immediate emergency rollback (correctness issue)

**Follow-up:** Investigate, fix, validate in staging before re-deploying

## Testing the Rollback

Don't wait for production incidents to test rollback procedures.

**Monthly rollback drill:**
```bash
# 1. Deploy Zig kernels to staging
./scripts/deploy_with_zig.sh staging

# 2. Let it run for 5 minutes
sleep 300

# 3. Execute rollback procedure
./scripts/rollback_to_rust.sh staging

# 4. Verify system health
./scripts/health_check.sh staging
```

Time each step. Update documentation if RTO exceeds target.

## Architecture Documentation: For Future You

Six months from now, you won't remember why arena size matters or how SIMD works.

Architecture docs explain:
- FFI boundary design (memory ownership, no aliasing)
- Arena allocator implementation (bump pointer, thread-local)
- SIMD patterns (8-wide AVX2, 4-wide NEON)
- Performance characteristics (expected improvements, bottlenecks)

Written for maintainers who need to debug or extend the system.

## Lessons From Production

We've deployed Zig kernels to production three times. Each taught us something:

**Deployment 1:** Forgot to configure arena size. Overflows on first query. Rollback in 4 minutes.

**Deployment 2:** Gradual rollout caught performance regression on ARM instances (NEON vs AVX2 assumptions). Fixed before full deployment.

**Deployment 3:** Smooth. All checks passed. Monitoring showed expected improvements. No rollback needed.

The difference: Documentation improved after each iteration.

## Documentation as Code Review

Good documentation catches mistakes during review:

**Deployment procedure:** "Wait, we need Zig 0.13.0 installed? Who owns that?"
**Monitoring section:** "We're not tracking arena overflows yet, let's add that metric."
**Rollback procedure:** "5 minutes RTO assumes cargo build is instant. Account for compile time."

Writing forces you to think through edge cases.

## Conclusion

Deploy fast, rollback faster.

Zig kernels add complexity. Documentation mitigates risk:
- Operators know how to deploy
- Monitors alert when things break
- Rollback procedures restore service quickly

When the 3 AM page comes, you'll be glad you wrote this down.
