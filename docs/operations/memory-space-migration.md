# Memory Space Migration Guide

Guide for migrating existing Engram deployments to multi-tenant memory spaces.

## Overview

Engram's memory spaces provide isolated multi-tenant environments with separate:
- Memory storage and persistence layers
- Spreading activation graphs
- Health metrics and diagnostics
- WAL (Write-Ahead Log) and tier storage

Each space operates independently while sharing the same Engram instance.

## Prerequisites

- Engram build with Milestone 7 features (multi-tenancy support)
- Existing single-tenant deployment (optional)
- Understanding of current data layout and access patterns

## Backward Compatibility

Memory spaces maintain 100% backward compatibility:

- Existing deployments work unchanged without configuration
- All operations default to the `default` space when no space is specified
- No breaking changes to existing API contracts
- Gradual migration path allows phased client updates

## Migration Steps

### 1. Upgrade Engram Binary

Stop existing instance:
```bash
./target/debug/engram stop
```

Build or deploy new version with multi-tenancy support:
```bash
git pull origin main
cargo build --release
```

### 2. Update Configuration

Add persistence settings to your configuration file (or use defaults):

```toml
[persistence]
data_root = "~/.local/share/engram"  # Tilde expansion supported
hot_capacity = 100000
warm_capacity = 1000000
cold_capacity = 10000000
```

Configuration file locations:
- Linux: `~/.config/engram/config.toml`
- macOS: `~/Library/Application Support/engram/config.toml`
- Custom: Specify with `--config` flag

### 3. Start Server with Auto-Discovery

Start the upgraded Engram instance:
```bash
./target/release/engram start
```

The registry automatically:
1. Scans `data_root` for existing space directories
2. Discovers any previously created memory spaces
3. Runs WAL recovery for each space
4. Makes all spaces available for requests

Check startup logs for recovery status:
```
INFO  engram_core::registry: Discovered 3 memory spaces: [default, production, staging]
INFO  engram_core::registry: Recovered 'default': 1200 entries, 0 corrupted, took 45ms
INFO  engram_core::registry: Recovered 'production': 3500 entries, 0 corrupted, took 120ms
```

### 4. Verify Existing Data

Check that your existing data is accessible in the `default` space:

```bash
# Via CLI
./target/release/engram status

# Via HTTP
curl http://localhost:7432/api/v1/system/health
```

Expected response:
```json
{
  "spaces": [
    {
      "space": "default",
      "memories": 1200,
      "pressure": 0.0,
      "wal_lag_ms": 0.0,
      "consolidation_rate": 0.0
    }
  ]
}
```

### 5. Create Additional Spaces

Create new memory spaces for tenants:

```bash
# Via CLI
./target/release/engram space create production
./target/release/engram space create staging

# Via HTTP (spaces are created automatically on first use)
curl -X POST http://localhost:7432/api/v1/memories/remember \
  -H "Content-Type: application/json" \
  -H "X-Memory-Space: production" \
  -d '{"content": "First production memory", "confidence": 0.95}'
```

List all spaces:
```bash
./target/release/engram space list

# Or via HTTP
curl http://localhost:7432/api/v1/spaces
```

### 6. Migrate Clients Gradually

Update clients to specify memory spaces using one of three methods:

#### Option 1: HTTP Header (Recommended)

```bash
curl -H "X-Memory-Space: production" \
  http://localhost:7432/api/v1/memories/recall?query=data
```

Advantages:
- Cleanest separation of routing from business logic
- Works with all HTTP methods (GET, POST, etc.)
- Easy to add via middleware or API gateway

#### Option 2: Query Parameter

```bash
curl "http://localhost:7432/api/v1/memories/recall?space=production&query=data"
```

Advantages:
- No header manipulation needed
- Works with simple HTTP clients
- Easy to test in browser

#### Option 3: Request Body

```bash
curl -X POST http://localhost:7432/api/v1/memories/remember \
  -H "Content-Type: application/json" \
  -d '{
    "content": "Data",
    "confidence": 0.9,
    "memory_space": "production"
  }'
```

Advantages:
- All request data in one place
- No URL modification needed
- Works for complex POST requests

#### Routing Precedence

When multiple sources specify a space:
1. `X-Memory-Space` header (highest priority)
2. `?space=<space_id>` query parameter
3. `"memory_space"` field in JSON body

This allows header-based routing to override application-level defaults.

### 7. gRPC Client Migration

For gRPC clients, update protobuf message construction:

```python
import grpc
import engram_pb2_grpc as engram

# Old way (uses 'default' space)
episode = engram.Episode(
    embedding=vec,
    confidence=0.95
)

# New way (specify space)
episode = engram.Episode(
    embedding=vec,
    confidence=0.95,
    memory_space_id="production"
)
```

Apply to all RPC methods:
- `Store()` - memory_space_id in Episode
- `Recall()` - memory_space_id in RecallRequest
- `streaming_remember()` - memory_space_id in each RememberRequest
- `streaming_recall()` - memory_space_id in each RecallRequest
- `stream()` - memory_space_id in StreamRequest

## Monitoring Multi-Tenant Deployments

### Per-Space Health Metrics

Query per-space health via HTTP:
```bash
curl http://localhost:7432/api/v1/system/health | jq
```

Response shows metrics for all spaces:
```json
{
  "spaces": [
    {
      "space": "default",
      "memories": 1200,
      "pressure": 0.0,
      "wal_lag_ms": 0.0,
      "consolidation_rate": 0.0
    },
    {
      "space": "production",
      "memories": 3500,
      "pressure": 0.15,
      "wal_lag_ms": 2.5,
      "consolidation_rate": 12.3
    }
  ]
}
```

### CLI Status with Space Filter

View all spaces:
```bash
./target/release/engram status
```

Filter to specific space:
```bash
./target/release/engram status --space production
```

Output includes formatted table:
```
Per-Space Metrics:
┌────────────────────┬───────────┬──────────┬─────────────┬─────────────────┐
│ Space              │ Memories  │ Pressure │ WAL Lag (ms)│ Consolidation   │
├────────────────────┼───────────┼──────────┼─────────────┼─────────────────┤
│ production         │      3500 │    15.0% │        2.50 │        12.30/s │
└────────────────────┴───────────┴──────────┴─────────────┴─────────────────┘
```

### Directory Structure Inspection

Spaces create isolated directory hierarchies:

```
~/.local/share/engram/
├── default/
│   ├── wal/
│   ├── hot/
│   ├── warm/
│   └── cold/
├── production/
│   ├── wal/
│   ├── hot/
│   ├── warm/
│   └── cold/
└── staging/
    ├── wal/
    ├── hot/
    ├── warm/
    └── cold/
```

Check directory sizes:
```bash
du -sh ~/.local/share/engram/*/
```

## Advanced Configuration

### Custom Data Root

Override default data location:

```toml
[persistence]
data_root = "/mnt/engram-data"
```

Or via environment variable:
```bash
export ENGRAM_DATA_ROOT="/mnt/engram-data"
./target/release/engram start
```

### Per-Space Tier Capacities

Tier capacities are currently global across all spaces. Future enhancement will support per-space capacity configuration.

## Troubleshooting

### Space Not Found Errors

**Symptom**: API returns 404 or "space not found" errors

**Causes**:
1. Space name typo in client code
2. Space not created yet (automatic creation may be disabled)
3. Registry startup issues

**Solutions**:
```bash
# List existing spaces
./target/release/engram space list

# Create space explicitly
./target/release/engram space create <space_name>

# Check server logs
tail -f /var/log/engram/server.log
```

### WAL Recovery Failures

**Symptom**: Server startup reports WAL corruption

**Causes**:
1. Unclean shutdown (power loss)
2. Disk corruption
3. Concurrent writes (improper shutdown)

**Solutions**:
```bash
# Check recovery logs on startup
./target/release/engram start 2>&1 | grep "Recovered"

# If corruption is limited, server continues with partial recovery
# Full WAL rebuild (DESTRUCTIVE - only if backup exists):
rm -rf ~/.local/share/engram/<space>/wal/*
# Server will start with empty space
```

### Cross-Space Data Leakage

**Symptom**: Queries return data from wrong tenant

**Causes**:
1. Client not setting space correctly
2. Default space fallback behavior
3. HTTP routing gap (Task 004 follow-up incomplete)

**Verification**:
```bash
# Store in space A
curl -X POST http://localhost:7432/api/v1/memories/remember \
  -H "X-Memory-Space: tenant-a" \
  -d '{"content": "Tenant A data", "confidence": 0.9}'

# Query from space B (should return empty)
curl -H "X-Memory-Space: tenant-b" \
  "http://localhost:7432/api/v1/memories/recall?query=Tenant%20A"

# Expected: No results (empty array)
# If you see results: HTTP routing gap, see Known Issues below
```

### Performance Degradation

**Symptom**: Slower response times after enabling multi-tenancy

**Causes**:
1. Registry contention under high load
2. Increased directory I/O
3. Per-space consolidation overhead

**Diagnostics**:
```bash
# Check per-space pressure
curl http://localhost:7432/api/v1/system/health | jq '.spaces[] | {space, pressure}'

# Monitor registry lock contention (if available)
curl http://localhost:7432/metrics | grep registry_lock_wait
```

**Mitigations**:
- Reduce number of active spaces
- Increase tier capacities to reduce consolidation frequency
- Use space-specific rate limiting at API gateway

## Known Issues

### HTTP Routing Gap (Task 004 Follow-Up)

**Status**: Documented gap, fix pending

**Issue**: X-Memory-Space header is extracted but not wired to memory operations in HTTP handlers

**Impact**: HTTP requests with X-Memory-Space header may return 404

**Workaround**: Use gRPC interface for production multi-tenant deployments until HTTP routing is complete

**Tracking**: See `roadmap/milestone-7/004_COMPLETION_REVIEW.md`

### Health Endpoint Format (Task 006b Follow-Up)

**Status**: Documented gap, fix pending

**Issue**: Response format mismatch in spaces array parsing

**Impact**: Some monitoring tools may fail to parse per-space metrics

**Workaround**: Use CLI status command for reliable per-space metrics

**Tracking**: See `roadmap/milestone-7/007_COMPLETION_REVIEW.md`

### Streaming API Isolation (Task 005c Follow-Up)

**Status**: Partial implementation, full isolation deferred

**Issue**: Space extraction added but full per-space event streaming incomplete

**Impact**: Streaming endpoints (SSE, gRPC streams) may leak data across spaces

**Workaround**: Avoid streaming APIs for multi-tenant production use until isolation is complete

**Tracking**: See `roadmap/milestone-7/005_COMPLETION_REVIEW.md`

## Rollback Procedure

If migration causes issues, rollback to single-tenant mode:

### 1. Stop Multi-Tenant Server

```bash
./target/release/engram stop
```

### 2. Restore Previous Binary

```bash
# If you have a backup
cp /backup/engram ./target/release/engram

# Or rebuild from previous commit
git checkout <previous-commit>
cargo build --release
```

### 3. Consolidate Spaces (Optional)

If you created multiple spaces and want to merge them:

```bash
# This is manual - copy WAL entries from all spaces into 'default'
cat ~/.local/share/engram/production/wal/* \
    ~/.local/share/engram/staging/wal/* \
    >> ~/.local/share/engram/default/wal/merged.wal

# Remove other spaces
rm -rf ~/.local/share/engram/{production,staging}
```

Note: This is a simplified merge. Production rollback should use proper WAL replay.

### 4. Start Single-Tenant Server

```bash
./target/release/engram start
```

All data will be in the `default` space, accessible without specifying space.

## Best Practices

1. **Start Small**: Create 2-3 spaces initially to validate isolation
2. **Monitor Early**: Set up per-space metrics monitoring before production traffic
3. **Test Fallback**: Verify default space behavior works as expected
4. **Document Spaces**: Maintain mapping of space IDs to tenant/environment names
5. **Backup Per-Space**: Implement backup strategy that preserves space isolation
6. **Load Test**: Validate performance with realistic multi-tenant load patterns

## Further Reading

- [README - Memory Spaces Section](../../README.md#memory-spaces-multi-tenancy)
- [API Reference - Multi-Tenancy](../api/index.md#multi-tenancy-memory-spaces)
- [Usage Guide - Multi-Tenancy](../../usage.md#multi-tenancy-memory-spaces)
- [Milestone 7 Completion Summary](../../roadmap/milestone-7/MILESTONE_7_COMPLETION_SUMMARY.md)

## Support

For issues or questions:
- GitHub Issues: https://github.com/orchard9/engram/issues
- Documentation: https://docs.engram.dev
- Task tracking: `roadmap/milestone-7/`
