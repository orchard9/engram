# Deploying Performance Kernels (And Rolling Them Back) - Research

## Key Topics

### 1. Deployment Strategies for Performance Optimizations

**Gradual Rollout Pattern:**
1. Deploy to canary (1-10% traffic)
2. Monitor for regressions (latency, errors, resource usage)
3. Expand to 50% traffic
4. Full deployment to 100%

**Rollback Triggers:**
- Error rate increase >0.5%
- Latency p99 increase >10%
- Resource utilization spike >20%
- Correctness issues (any)

**Citations:**
- Beyer, B., et al. (2016). "Site Reliability Engineering", O'Reilly
- Humble, J., & Farley, D. (2010). "Continuous Delivery"

### 2. Feature Flags for Runtime Control

Compile-time feature flag (Cargo features):
```toml
[features]
zig-kernels = []
```

Runtime feature flag (environment variable):
```bash
export ENGRAM_USE_ZIG_KERNELS=true
```

Allows toggling without rebuild - critical for emergency rollback.

**Citation:** Hodgson, P. (2017). "Feature Toggles (aka Feature Flags)", Martin Fowler's blog

### 3. Rollback Procedures and RTO/RPO

**RTO (Recovery Time Objective):** Time to restore service after incident
**RPO (Recovery Point Objective):** Acceptable data loss window

For Zig kernel rollback:
- RTO target: <5 minutes (rebuild + deploy Rust-only binary)
- RPO: 0 (stateless kernels, no data loss)

**Emergency Rollback:**
```bash
cargo build --release  # Omit --features zig-kernels
systemctl restart engram
```

**Citation:** Limoncelli, T., et al. (2014). "The Practice of Cloud System Administration"
