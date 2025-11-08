# Engram Performance Baseline Repository

This directory contains performance baseline measurements from various machines and environments to track Engram's resource usage and identify optimization opportunities.

## Directory Structure

```
reviewed-baselines/
├── README.md                 # This file
├── {MACHINE}-{YYYYMMDD}/    # Machine-specific baseline results
│   ├── system-info.txt      # Hardware and OS specifications
│   ├── 1min-baseline.txt    # 1-minute baseline raw data
│   ├── 15min-baseline.txt   # 15-minute baseline raw data
│   ├── 24hr-baseline.txt    # 24-hour baseline raw data (if available)
│   ├── metrics-summary.json # Key metrics in JSON format
│   └── analysis.md          # Detailed analysis and recommendations
└── baseline-template.md     # Template for new baseline reports
```

## Baseline Process

1. **Build**: `cargo build --release`
2. **Start**: `./target/release/engram start`
3. **1-minute test**: Wait 2 minutes, capture metrics
4. **15-minute test**: Wait 16 minutes, capture metrics  
5. **24-hour test**: Let run for 24+ hours, capture metrics
6. **Analysis**: Document findings and recommendations

## Key Metrics to Track

- **CPU Usage** (idle, average, peak)
- **Memory RSS** (startup, stable, growth rate)
- **Memory VSZ** (virtual memory size)
- **Thread Count**
- **File Descriptors**
- **Network Connections**
- **Disk I/O**
- **Response Times** (health check, API calls)

## Baseline Standards

### 🟢 Good (Production Ready)
- CPU idle: < 5%
- Memory RSS: < 2GB (idle)
- Memory growth: < 100MB/hour
- Health check: < 10ms

### 🟡 Warning (Needs Optimization)
- CPU idle: 5-15%
- Memory RSS: 2-5GB (idle)
- Memory growth: 100-500MB/hour
- Health check: 10-50ms

### 🔴 Critical (Blocker)
- CPU idle: > 15%
- Memory RSS: > 5GB (idle)
- Memory growth: > 500MB/hour
- Health check: > 50ms

## Recent Baselines

| Date | Machine | CPU (idle) | Memory (RSS) | Status | Link |
|------|---------|------------|--------------|--------|------|
| 2025-11-08 | JML | 20-22% | 8.7GB | 🔴 Critical | [Analysis](JML-20251108/analysis.md) |

## Common Issues Found

1. **High Idle CPU**: Background loops, excessive polling
2. **High Memory Usage**: Pre-allocated pools, retained data
3. **Memory Growth**: Leaks, unbounded caches
4. **Slow Response**: Lock contention, synchronous I/O

## Optimization Tracking

Track improvements between baseline runs to measure optimization effectiveness.