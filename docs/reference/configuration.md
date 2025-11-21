# Configuration Reference

Complete reference for all Engram configuration parameters. This guide documents every configuration option with defaults, valid ranges, tuning guidance, and monitoring metrics.

## Overview

Engram uses layered configuration with the following precedence (highest to lowest):

1. **Command-line flags** - `--hot-capacity 50000`

2. **Environment variables** - `ENGRAM_HOT_CAPACITY=50000`

3. **User config file** - `~/.config/engram/config.toml`

4. **System config file** - `/etc/engram/config.toml`

5. **Default config** - Embedded in binary

Configuration spans multiple domains:

- **Feature Flags** - Enable/disable experimental features

- **Memory Spaces** - Multi-tenancy and isolation

- **Persistence** - Tiered storage capacity limits

- **Security** - Authentication, authorization, and CORS

- **Cluster** - Distributed deployment and replication

- **Spreading Activation** - Graph traversal parameters

- **Consolidation** - Memory compression and pattern extraction

- **Decay** - Forgetting curves and memory strength

## Configuration File Format

Engram uses TOML format for configuration files. Here's a minimal example:

```toml
[feature_flags]
spreading_api_beta = true

[memory_spaces]
default_space = "default"
bootstrap_spaces = ["default"]

[persistence]
data_root = "~/.local/share/engram"
hot_capacity = 100_000
warm_capacity = 1_000_000
cold_capacity = 10_000_000

```

## Feature Flags Configuration

Control experimental and beta features.

### `feature_flags.spreading_api_beta`

**Type:** `bool`
**Default:** `true`
**Valid values:** `true`, `false`

**Purpose:** Enable experimental spreading activation API endpoints.

**When to disable:** If stability over features is critical for production deployments. Disabling removes `/api/v1/spread` and `/api/v1/activate` endpoints.

**Impact:**

- Enabled: Full spreading activation API available

- Disabled: Only basic memory storage/retrieval APIs available

**Validation:** Must be boolean value.

**Example:**

```toml
[feature_flags]
spreading_api_beta = true  # Enable spreading APIs

```

---

## Memory Spaces Configuration

Memory spaces provide multi-tenant isolation. Each space is an independent cognitive graph with separate storage and activation state.

### `memory_spaces.default_space`

**Type:** `string`
**Default:** `"default"`
**Valid values:** Alphanumeric, hyphens, underscores (3-64 characters)

**Purpose:** Default memory space ID for operations without explicit space header.

**When to change:** Multi-tenant deployments should use tenant-specific defaults. Single-tenant systems can use descriptive names matching their domain.

**Validation rules:**

- Length: 3-64 characters

- Characters: `a-z`, `A-Z`, `0-9`, `-`, `_`

- Must not start with hyphen or underscore

**Examples:**

```toml
# Multi-tenant SaaS
default_space = "tenant-acme-prod"

# Single-tenant agent
default_space = "assistant-v2"

# Development environment
default_space = "dev"

```

**Related metrics:**

- `engram_memory_space_count` - Track number of active spaces

- `engram_memory_space_memory_count{space="..."}` - Per-space memory count

### `memory_spaces.bootstrap_spaces`

**Type:** `array of strings`
**Default:** `["default"]`
**Valid values:** Array of valid memory space IDs

**Purpose:** Memory spaces to create automatically on startup. Pre-provisioning spaces avoids cold-start latency on first access.

**When to change:**

- Multi-tenant: Pre-provision known tenant spaces

- Shared knowledge: Create common knowledge space

- Testing: Create test-specific spaces

**Validation rules:**

- Each entry must be valid memory space ID

- Array can be empty (no automatic provisioning)

- No duplicate IDs allowed

**Examples:**

```toml
# Multi-tenant with shared space
bootstrap_spaces = ["tenant-a", "tenant-b", "shared"]

# Single tenant only
bootstrap_spaces = ["production"]

# Development with test spaces
bootstrap_spaces = ["dev", "test", "integration"]

```

**Performance impact:**

- Each space allocates hot tier capacity on startup

- 100K hot capacity = ~1.2GB RAM per space

- Consider memory budget when pre-provisioning many spaces

**Monitoring:**

- `engram_memory_space_bootstrap_duration_seconds` - Space creation latency

- `engram_memory_space_bootstrap_failures_total` - Failed creations

---

## Persistence Configuration

Controls tiered storage capacity and data root location.

### Architecture Overview

Engram uses a three-tier storage architecture:

```
┌─────────────────────────────────────────────────────┐
│ HOT TIER (In-Memory DashMap)                        │
│ - Fastest access (~100ns)                           │
│ - Working set of frequently accessed memories       │
│ - ~12KB per memory (embedding + metadata)           │
└─────────────────────────────────────────────────────┘
                    ↓ Evict cold memories
┌─────────────────────────────────────────────────────┐
│ WARM TIER (Memory-Mapped Files)                     │
│ - Fast access (~1-5μs)                              │
│ - Recently accessed but evicted from hot tier       │
│ - ~10KB per memory (compressed format)              │
└─────────────────────────────────────────────────────┘
                    ↓ Age-based migration
┌─────────────────────────────────────────────────────┐
│ COLD TIER (Columnar Storage)                        │
│ - Archive access (~10-50μs)                         │
│ - Long-term storage for infrequently accessed       │
│ - ~8KB per memory (columnar compression)            │
└─────────────────────────────────────────────────────┘

```

**Migration policy:**

- Hot → Warm: When hot tier reaches 80% capacity, least recently used (LRU) evicted

- Warm → Cold: After 7 days without access

- Cold → Warm: On access, if confidence > 0.3

- Warm → Hot: On frequent access (3+ times in 24 hours)

### `persistence.data_root`

**Type:** `string`
**Default:** `"~/.local/share/engram"`
**Valid values:** Absolute path or tilde-expanded home directory

**Purpose:** Root directory for all memory space storage. Each memory space creates a subdirectory under this root.

**When to change:**

- **Production:** Use `/var/lib/engram` or mounted volumes

- **Docker:** Use `/data/engram` mounted from host

- **Kubernetes:** Use persistent volume claim mount point

- **Multi-instance:** Use unique paths per instance

**Disk requirements:**

- Calculate total disk: `(warm_capacity * 10KB) + (cold_capacity * 8KB)` per space

- Add 20% overhead for WAL and indexes

- Recommend SSD for warm tier, HDD acceptable for cold tier

**Directory structure:**

```
data_root/
├── memory-space-1/
│   ├── hot/           # Hot tier metadata
│   ├── warm/          # Memory-mapped files
│   ├── cold/          # Columnar storage
│   └── wal/           # Write-ahead log
├── memory-space-2/
│   └── ...
└── system/
    └── metrics/       # Persistent metrics

```

**Validation rules:**

- Must be absolute path or start with `~`

- Parent directory must be writable

- Recommend dedicated filesystem for production

**Examples:**

```toml
# Development (local user directory)
data_root = "~/.local/share/engram"

# Production (system directory)
data_root = "/var/lib/engram"

# Docker (mounted volume)
data_root = "/data/engram"

# Multi-instance (unique paths)
data_root = "/var/lib/engram-instance-1"

```

**Monitoring:**

- `engram_storage_disk_usage_bytes{tier="warm"}` - Disk usage per tier

- `engram_storage_disk_usage_bytes{tier="cold"}`

- `engram_storage_wal_size_bytes` - WAL disk usage

### `persistence.hot_capacity`

**Type:** `usize`
**Default:** `100_000`
**Valid range:** `1,000` - `10,000,000`

**Purpose:** Maximum memories in hot tier (in-memory DashMap) per space. This is your working set size - memories accessed frequently enough to stay in RAM.

**Memory impact:**

- Approximately **12KB per memory** (768-dim f32 embedding + metadata)

- Formula: `hot_capacity * 12KB = hot tier RAM per space`

**Tuning guide by environment:**

| Environment | Hot Capacity | RAM Usage | Use Case |
|------------|--------------|-----------|----------|
| Development | 10,000 - 50,000 | 120MB - 600MB | Local testing, small datasets |
| Staging | 50,000 - 100,000 | 600MB - 1.2GB | Pre-production validation |
| Production (Small) | 100,000 - 250,000 | 1.2GB - 3GB | Single agent, focused domain |
| Production (Medium) | 250,000 - 500,000 | 3GB - 6GB | Multi-agent, broad domain |
| Production (Large) | 500,000 - 1,000,000 | 6GB - 12GB | High-scale, multiple tenants |
| Production (XL) | 1,000,000 - 10,000,000 | 12GB - 120GB | Research corpora, knowledge graphs |

**When to increase:**

- High working set: Same memories accessed repeatedly

- Low hit rate: `engram_storage_hot_tier_hit_rate < 0.90`

- High migration rate: `engram_storage_migrations_total` increasing rapidly

- Available RAM: System has unused memory capacity

**When to decrease:**

- Memory pressure: Out-of-memory errors or swapping

- Cold-data workload: Few repeated accesses

- Cost optimization: Reducing cloud instance size

- Multiple spaces: Allocating capacity across tenants

**Performance characteristics:**

- Hot tier access: ~100ns latency (in-memory hashmap)

- Cache miss to warm tier: ~1-5μs (5-50x slower)

- Thrashing threshold: >80% capacity triggers aggressive eviction

**Validation rules:**

- Minimum: 1,000 (prevents config errors)

- Maximum: 10,000,000 (prevents excessive RAM allocation)

- Recommend: hot_capacity ≤ warm_capacity ≤ cold_capacity

**Examples:**

```toml
# Development: Minimal RAM footprint
hot_capacity = 10_000  # 120MB RAM

# Production: Balanced working set
hot_capacity = 500_000  # 6GB RAM

# Large-scale: High-throughput system
hot_capacity = 2_000_000  # 24GB RAM

```

**Monitoring metrics:**

```promql
# Current hot tier size
engram_storage_hot_tier_size{space="default"}

# Hot tier hit rate (target: >0.95)
engram_storage_hot_tier_hit_rate{space="default"}

# Migration rate (indicator of thrashing)
rate(engram_storage_migrations_total{from="hot",to="warm"}[5m])

# Memory usage
engram_storage_memory_bytes{tier="hot",space="default"}

```

**Troubleshooting:**

**Problem:** Hit rate < 0.90, high latency

- **Root cause:** Hot tier too small for working set

- **Fix:** Increase hot_capacity to 1.5x current working set

- **Validation:** Hit rate should stabilize >0.95

**Problem:** Out-of-memory errors

- **Root cause:** Hot tier allocation exceeds available RAM

- **Fix:** Decrease hot_capacity or add RAM

- **Formula:** `(hot_capacity * 12KB * num_spaces) < 80% of total RAM`

**Problem:** Frequent migrations, stable hit rate

- **Root cause:** Access pattern has natural churn

- **Fix:** This is normal behavior, no action needed

- **Optimization:** Consider warm tier tuning instead

### `persistence.warm_capacity`

**Type:** `usize`
**Default:** `1_000_000`
**Valid range:** `10,000` - `100,000,000`

**Purpose:** Maximum memories in warm tier (memory-mapped files) per space. This is your "recently accessed but evicted" cache layer.

**Disk impact:**

- Approximately **10KB per memory** (compressed format)

- Formula: `warm_capacity * 10KB = warm tier disk per space`

**Tuning guide by environment:**

| Environment | Warm Capacity | Disk Usage | Access Latency |
|------------|---------------|------------|----------------|
| Development | 100,000 - 500,000 | 1GB - 5GB | 1-5μs |
| Staging | 500,000 - 2,000,000 | 5GB - 20GB | 1-5μs |
| Production (Small) | 1,000,000 - 5,000,000 | 10GB - 50GB | 1-5μs |
| Production (Medium) | 5,000,000 - 10,000,000 | 50GB - 100GB | 2-8μs |
| Production (Large) | 10,000,000 - 50,000,000 | 100GB - 500GB | 5-10μs |
| Production (XL) | 50,000,000 - 100,000,000 | 500GB - 1TB | 10-20μs |

**When to increase:**

- Large corpus: Total memories > current warm capacity

- Moderate access frequency: Memories accessed weekly

- High warm→cold migration: Warm tier fills up quickly

- Available disk: System has unused SSD capacity

**When to decrease:**

- Disk constraints: Limited SSD storage

- Cold-data workload: Most data rarely accessed

- Cost optimization: Reducing storage costs

- Cold tier sufficient: Most queries satisfied by cold tier

**Performance characteristics:**

- Warm tier access: ~1-5μs latency (memory-mapped files)

- Cache miss to cold tier: ~10-50μs (10-50x slower)

- Memory-mapped overhead: Virtual memory, actual RAM usage varies

**Storage format:**

- File structure: One file per memory (~10KB)

- Compression: LZ4 (fast decompression, ~60% compression ratio)

- Indexing: B-tree index for fast lookup

- Memory mapping: Zero-copy access via mmap

**Validation rules:**

- Minimum: 10,000 (prevents config errors)

- Maximum: 100,000,000 (prevents excessive disk usage)

- Recommend: warm_capacity ≥ hot_capacity

- Recommend: warm_capacity ≤ cold_capacity

**Examples:**

```toml
# Development: Moderate working set
warm_capacity = 100_000  # 1GB disk

# Production: Large actively-used corpus
warm_capacity = 5_000_000  # 50GB disk

# Large-scale: Massive recently-accessed data
warm_capacity = 20_000_000  # 200GB disk

```

**Monitoring metrics:**

```promql
# Current warm tier size
engram_storage_warm_tier_size{space="default"}

# Warm tier hit rate (target: >0.80)
engram_storage_warm_tier_hit_rate{space="default"}

# Migration rate (warm to cold)
rate(engram_storage_migrations_total{from="warm",to="cold"}[1h])

# Disk usage
engram_storage_disk_usage_bytes{tier="warm",space="default"}

```

**Filesystem recommendations:**

- **Best:** SSD (NVMe > SATA)

- **Good:** High-RPM HDD (10K+ RPM) with large cache

- **Acceptable:** Standard HDD for cost-optimized deployments

**Troubleshooting:**

**Problem:** High warm→cold migration, queries slow

- **Root cause:** Warm tier too small for access pattern

- **Fix:** Increase warm_capacity to reduce evictions

- **Validation:** Migration rate should decrease

**Problem:** Disk space warnings

- **Root cause:** Warm tier consuming too much disk

- **Fix:** Decrease warm_capacity or add storage

- **Migration:** Consider tiering policy adjustment

### `persistence.cold_capacity`

**Type:** `usize`
**Default:** `10_000,000`
**Valid range:** `100,000` - `1,000,000,000`

**Purpose:** Maximum memories in cold tier (columnar storage) per space. This is your long-term archival storage for infrequently accessed memories.

**Disk impact:**

- Approximately **8KB per memory** (columnar compression)

- Formula: `cold_capacity * 8KB = cold tier disk per space`

**Tuning guide by environment:**

| Environment | Cold Capacity | Disk Usage | Access Latency |
|------------|---------------|------------|----------------|
| Development | 1,000,000 | 8GB | 10-50μs |
| Staging | 5,000,000 - 10,000,000 | 40GB - 80GB | 10-50μs |
| Production (Small) | 10,000,000 - 50,000,000 | 80GB - 400GB | 10-50μs |
| Production (Medium) | 50,000,000 - 100,000,000 | 400GB - 800GB | 20-100μs |
| Production (Large) | 100,000,000 - 500,000,000 | 800GB - 4TB | 50-200μs |
| Research Archive | 500,000,000 - 1,000,000,000 | 4TB - 8TB | 100-500μs |

**When to increase:**

- Long-term archival: Preserving historical knowledge

- Large-scale corpus: Millions to billions of memories

- Compliance requirements: Data retention policies

- Available disk: Cheap HDD storage available

**When to decrease:**

- Disk constraints: Limited archive storage

- Data lifecycle: Aggressive pruning of old data

- Cost optimization: Reducing storage costs

**Performance characteristics:**

- Cold tier access: ~10-50μs latency (columnar read + decompression)

- Large scan performance: Optimized for analytical queries

- Compression ratio: ~65-70% (better than warm tier)

**Storage format:**

- File structure: Columnar format (Apache Arrow-compatible)

- Compression: ZSTD (high compression ratio)

- Indexing: Zone maps for range queries

- Batch access: Optimized for bulk reads

**Validation rules:**

- Minimum: 100,000 (prevents config errors)

- Maximum: 1,000,000,000 (1 billion memories)

- Recommend: cold_capacity ≥ warm_capacity

**Examples:**

```toml
# Development: Minimal archive
cold_capacity = 1_000_000  # 8GB disk

# Production: Standard deployment
cold_capacity = 50_000_000  # 400GB disk

# Large-scale: Research knowledge graph
cold_capacity = 500_000_000  # 4TB disk

```

**Monitoring metrics:**

```promql
# Current cold tier size
engram_storage_cold_tier_size{space="default"}

# Cold tier access rate
rate(engram_storage_cold_tier_accesses_total{space="default"}[1h])

# Disk usage
engram_storage_disk_usage_bytes{tier="cold",space="default"}

# Compaction effectiveness
engram_storage_compaction_ratio{tier="cold",space="default"}

```

**Filesystem recommendations:**

- **Best:** HDD with large sequential throughput

- **Good:** Object storage (S3, GCS, Azure Blob) for archival

- **Acceptable:** Any block storage with adequate capacity

**Troubleshooting:**

**Problem:** Slow analytical queries over cold data

- **Root cause:** Cold tier not optimized for query pattern

- **Fix:** Ensure columnar format enabled, check indexes

- **Optimization:** Consider partitioning by time or domain

**Problem:** Disk space exhausted

- **Root cause:** Cold tier reached capacity

- **Fix:** Increase cold_capacity or implement pruning

- **Lifecycle:** Define data retention policy

---

---

## Security Configuration

Controls authentication, authorization, rate limiting, and CORS settings for API access.

### Configuration Structure

Security is configured via the `[security]` section in your configuration file:

```toml
[security]
auth_mode = "none"
rate_limiting = false

[security.api_keys]
backend = "file"
storage_path = "./data/api_keys.db"
rotation_days = 90
warn_before_expiry_days = 14

[security.cors]
allowed_origins = ["*"]
allowed_methods = ["GET", "POST", "DELETE", "OPTIONS"]
max_age_seconds = 3600
```

### Security Configuration Fields

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `auth_mode` | string enum | `"none"` | Authentication mode: `"none"` or `"api_key"` |
| `rate_limiting` | boolean | `false` | Enable rate limiting for API requests |
| `api_keys.backend` | string enum | `"file"` | Storage backend type (currently only `"file"` supported) |
| `api_keys.storage_path` | path | `"./data/api_keys.db"` | Path to API key storage database |
| `api_keys.rotation_days` | integer | `90` | API key rotation period in days |
| `api_keys.warn_before_expiry_days` | integer | `14` | Days before expiry to warn |
| `cors.allowed_origins` | array of strings | `["*"]` | Allowed CORS origins |
| `cors.allowed_methods` | array of strings | `["GET", "POST", "DELETE", "OPTIONS"]` | Allowed HTTP methods |
| `cors.max_age_seconds` | integer | `3600` | CORS preflight cache duration in seconds |

### `security.auth_mode`

**Type:** `string` (enum)
**Default:** `"none"`
**Valid values:** `"none"`, `"api_key"`

**Purpose:** Controls authentication mode for API access.

**Authentication Modes:**

**1. None (Default)**
```toml
[security]
auth_mode = "none"
```

- No authentication required
- Suitable for development environments
- Default for backward compatibility
- All API requests accepted without credentials

**2. API Key**
```toml
[security]
auth_mode = "api_key"
```

- Require API key for all requests
- Keys stored in configured backend
- Suitable for production deployments
- Keys passed via `Authorization: Bearer <key>` header

**When to use:**

- **Development:** Use `"none"` for local testing and rapid iteration
- **Staging:** Use `"api_key"` to validate authentication flow
- **Production:** Always use `"api_key"` for deployed systems
- **Multi-tenant:** Use `"api_key"` with per-tenant keys

**Environment Variable Override:**

Set `ENGRAM_AUTH_MODE` to override configuration file:

```bash
export ENGRAM_AUTH_MODE=api_key
engram server
```

**Migration Guide:**

To enable authentication on an existing deployment:

1. Update configuration file to `auth_mode = "api_key"`
2. Generate initial API keys (see M15 Task 009 tutorials)
3. Distribute keys to authorized clients
4. Restart Engram server
5. Update client applications to include `Authorization` header

**Examples:**

```toml
# Development: No authentication
[security]
auth_mode = "none"

# Production: API key authentication
[security]
auth_mode = "api_key"
```

### `security.rate_limiting`

**Type:** `boolean`
**Default:** `false`

**Purpose:** Enable rate limiting to prevent API abuse and ensure fair resource allocation.

**When to enable:**

- Production deployments to prevent abuse
- Multi-tenant systems for fair resource allocation
- Public-facing APIs to control load
- Cost management for metered infrastructure

**When to disable:**

- Development environments for rapid iteration
- Trusted internal deployments
- Single-tenant dedicated systems
- Load testing scenarios

**Environment Variable Override:**

```bash
export ENGRAM_RATE_LIMIT=true
engram server
```

**Examples:**

```toml
# Development: Disabled
[security]
rate_limiting = false

# Production: Enabled
[security]
rate_limiting = true
```

### `security.api_keys.backend`

**Type:** `string` (enum)
**Default:** `"file"`
**Valid values:** `"file"`

**Purpose:** Storage backend for API keys and metadata.

**Currently Supported:**

- `"file"` - File-based storage using local database

**Future Support (Planned):**

- `"postgres"` - PostgreSQL database
- `"redis"` - Redis for distributed deployments
- `"vault"` - HashiCorp Vault integration

**Examples:**

```toml
[security.api_keys]
backend = "file"  # Only option currently
```

### `security.api_keys.storage_path`

**Type:** `string` (path)
**Default:** `"./data/api_keys.db"`
**Valid values:** Absolute path or relative to working directory

**Purpose:** File path for API key database.

**Path Guidelines:**

**Development:**
```toml
storage_path = "./data/api_keys.db"  # Relative to working directory
```

**Production:**
```toml
storage_path = "/var/lib/engram/api_keys.db"  # Absolute system path
```

**Docker:**
```toml
storage_path = "/data/engram/api_keys.db"  # Mounted volume
```

**Security Considerations:**

- Ensure parent directory has restrictive permissions (700)
- File should be readable only by Engram process user
- Back up regularly as part of disaster recovery
- Encrypt at rest for compliance requirements
- Do not store in world-readable locations

**Environment Variable Override:**

```bash
export ENGRAM_API_KEY_PATH=/secure/path/keys.db
engram server
```

**Examples:**

```toml
# Development
[security.api_keys]
storage_path = "./data/api_keys.db"

# Production
[security.api_keys]
storage_path = "/var/lib/engram/api_keys.db"

# Docker
[security.api_keys]
storage_path = "/data/engram/api_keys.db"
```

### `security.api_keys.rotation_days`

**Type:** `integer`
**Default:** `90`
**Valid range:** `1` - `3650` (10 years)

**Purpose:** Automatic API key rotation period in days. Keys older than this threshold are flagged for rotation.

**Rotation Guidelines by Environment:**

| Environment | Rotation Days | Rationale |
|-------------|---------------|-----------|
| Development | 365+ | Convenience, minimal risk |
| Staging | 90 - 180 | Match production policy |
| Production (High Security) | 30 - 60 | Frequent rotation, minimize exposure |
| Production (Standard) | 90 | Balance security and operational overhead |
| Production (Low Risk) | 180 - 365 | Reduce rotation frequency |

**Compliance Recommendations:**

- **PCI DSS:** 90 days maximum
- **HIPAA:** 90 days recommended
- **SOC 2:** 90 days typical
- **ISO 27001:** 90 days recommended
- **Custom Policies:** Align with organization standards

**Validation:**

- Must be greater than 0
- Must be greater than `warn_before_expiry_days`

**Examples:**

```toml
# High security production
[security.api_keys]
rotation_days = 30

# Standard production
[security.api_keys]
rotation_days = 90

# Development
[security.api_keys]
rotation_days = 365
```

### `security.api_keys.warn_before_expiry_days`

**Type:** `integer`
**Default:** `14`
**Valid range:** `1` - `rotation_days - 1`

**Purpose:** Number of days before expiry to emit warnings. Allows proactive key rotation before expiration.

**Warning Guidelines:**

| Rotation Period | Recommended Warning | Rationale |
|-----------------|---------------------|-----------|
| 30 days | 7 days | One week notice |
| 60 days | 14 days | Two weeks notice |
| 90 days | 14 - 21 days | Two to three weeks notice |
| 180+ days | 30 days | One month notice |

**Warning Mechanisms:**

- Server logs warning messages
- Monitoring metrics exposed
- API responses include warning headers
- Automated alerts (if configured)

**Validation:**

- Must be greater than 0
- Must be less than `rotation_days`

**Examples:**

```toml
# 30-day rotation with 1-week warning
[security.api_keys]
rotation_days = 30
warn_before_expiry_days = 7

# 90-day rotation with 2-week warning
[security.api_keys]
rotation_days = 90
warn_before_expiry_days = 14
```

### `security.cors.allowed_origins`

**Type:** `array of strings`
**Default:** `["*"]`
**Valid values:** Array of origin URLs or `"*"` wildcard

**Purpose:** Configure allowed origins for Cross-Origin Resource Sharing (CORS).

**Origin Formats:**

- Full URL: `"https://app.example.com"`
- Wildcard: `"*"` (allow all origins)
- Subdomain wildcard: `"https://*.example.com"` (not currently supported)
- Multiple specific origins: `["https://app1.com", "https://app2.com"]`

**Security Guidelines:**

**Development:**
```toml
allowed_origins = ["*"]  # Allow all for convenience
```

**Production:**
```toml
allowed_origins = ["https://app.example.com", "https://admin.example.com"]  # Specific domains only
```

**Validation:**

- Array cannot be empty
- Each origin must be valid URL or `"*"`
- Use specific origins in production, never `"*"`

**Security Considerations:**

- Never use `"*"` in production
- Specify exact domains, including protocol and port
- Update when adding new frontend applications
- Review regularly as part of security audits

**Examples:**

```toml
# Development: Allow all
[security.cors]
allowed_origins = ["*"]

# Production: Specific domains
[security.cors]
allowed_origins = ["https://app.example.com", "https://admin.example.com"]

# Production: Single domain
[security.cors]
allowed_origins = ["https://app.example.com"]
```

### `security.cors.allowed_methods`

**Type:** `array of strings`
**Default:** `["GET", "POST", "DELETE", "OPTIONS"]`
**Valid values:** HTTP method names

**Purpose:** Configure allowed HTTP methods for CORS requests.

**Common Methods:**

- `GET` - Read operations
- `POST` - Create operations
- `PUT` - Full update operations
- `PATCH` - Partial update operations
- `DELETE` - Delete operations
- `OPTIONS` - CORS preflight requests (always required)
- `HEAD` - Metadata requests

**Default Rationale:**

- `GET` - Read memories and query operations
- `POST` - Create memories and execute operations
- `DELETE` - Remove memories
- `OPTIONS` - Required for CORS preflight

**Validation:**

- Array cannot be empty
- Must include `"OPTIONS"` for CORS to function
- Methods are case-sensitive (use uppercase)

**Examples:**

```toml
# Default: Read, create, delete
[security.cors]
allowed_methods = ["GET", "POST", "DELETE", "OPTIONS"]

# Read-only API
[security.cors]
allowed_methods = ["GET", "HEAD", "OPTIONS"]

# Full CRUD API
[security.cors]
allowed_methods = ["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"]
```

### `security.cors.max_age_seconds`

**Type:** `integer`
**Default:** `3600`
**Valid range:** `1` - `86400` (24 hours)

**Purpose:** Duration (in seconds) that browsers cache CORS preflight responses.

**Tuning Guidelines:**

| Value | Use Case | Rationale |
|-------|----------|-----------|
| 300 - 600 | Frequently changing CORS policy | Short cache for flexibility |
| 3600 | Standard production | 1-hour cache balances performance and flexibility |
| 7200 - 86400 | Stable CORS policy | Longer cache reduces preflight overhead |

**Performance Impact:**

- Higher values reduce preflight requests
- Lower values allow faster CORS policy updates
- Browsers may enforce lower maximums
- Consider client request patterns

**Validation:**

- Must be greater than 0
- Recommend 3600 (1 hour) for most deployments
- Maximum 86400 (24 hours) per spec

**Examples:**

```toml
# Aggressive caching (stable policy)
[security.cors]
max_age_seconds = 86400  # 24 hours

# Standard (balanced)
[security.cors]
max_age_seconds = 3600  # 1 hour

# Conservative (frequently changing)
[security.cors]
max_age_seconds = 600  # 10 minutes
```

## Environment Variable Precedence

Security configuration supports environment variable overrides for operational flexibility:

| Environment Variable | Configuration Key | Type | Example |
|---------------------|-------------------|------|---------|
| `ENGRAM_AUTH_MODE` | `security.auth_mode` | string | `export ENGRAM_AUTH_MODE=api_key` |
| `ENGRAM_API_KEY_PATH` | `security.api_keys.storage_path` | path | `export ENGRAM_API_KEY_PATH=/secure/keys.db` |
| `ENGRAM_RATE_LIMIT` | `security.rate_limiting` | boolean | `export ENGRAM_RATE_LIMIT=true` |

**Precedence Order (highest to lowest):**

1. Environment variables
2. User configuration file (`~/.config/engram/config.toml`)
3. System configuration file (`/etc/engram/config.toml`)
4. Default configuration (embedded in binary)

## Configuration Examples

### Development Configuration

```toml
[security]
# No authentication for local development
auth_mode = "none"
rate_limiting = false

[security.api_keys]
backend = "file"
storage_path = "./data/api_keys.db"
rotation_days = 365
warn_before_expiry_days = 30

[security.cors]
# Allow all origins for development
allowed_origins = ["*"]
allowed_methods = ["GET", "POST", "DELETE", "OPTIONS"]
max_age_seconds = 3600
```

### Production Configuration

```toml
[security]
# API key authentication required
auth_mode = "api_key"
rate_limiting = true

[security.api_keys]
backend = "file"
storage_path = "/var/lib/engram/api_keys.db"
rotation_days = 30
warn_before_expiry_days = 7

[security.cors]
# Specific domains only
allowed_origins = ["https://app.example.com"]
allowed_methods = ["GET", "POST", "DELETE", "OPTIONS"]
max_age_seconds = 3600
```

### Docker Configuration

```toml
[security]
auth_mode = "api_key"
rate_limiting = true

[security.api_keys]
backend = "file"
# Mounted volume path
storage_path = "/data/engram/api_keys.db"
rotation_days = 90
warn_before_expiry_days = 14

[security.cors]
allowed_origins = ["https://app.example.com"]
allowed_methods = ["GET", "POST", "DELETE", "OPTIONS"]
max_age_seconds = 3600
```

## Validation

Security configuration is validated on server startup. Common validation errors:

**Empty API key storage path:**
```
Error: api_keys.storage_path cannot be empty
```

**Invalid rotation period:**
```
Error: api_keys.rotation_days must be greater than 0
```

**Invalid warning period:**
```
Error: api_keys.warn_before_expiry_days (30) must be less than rotation_days (30)
```

**Empty CORS origins:**
```
Error: cors.allowed_origins cannot be empty
```

**Empty CORS methods:**
```
Error: cors.allowed_methods cannot be empty
```

**Invalid max age:**
```
Error: cors.max_age_seconds must be greater than 0
```

## Security Best Practices

1. **Never use `auth_mode = "none"` in production** - Always enable authentication for deployed systems

2. **Use specific CORS origins** - Never use `"*"` wildcard in production environments

3. **Secure API key storage** - Set restrictive file permissions (600) on `storage_path`

4. **Enable rate limiting** - Protect against abuse in production deployments

5. **Regular key rotation** - Follow compliance requirements (typically 30-90 days)

6. **Monitor expiry warnings** - Rotate keys proactively before expiration

7. **Back up API key database** - Include `storage_path` in backup procedures

8. **Use environment variables** - Override sensitive paths via environment in containers

## Migration from No Authentication

To enable authentication on an existing Engram deployment:

1. **Update configuration:**
   ```toml
   [security]
   auth_mode = "api_key"
   ```

2. **Generate initial API keys** (see security tutorials in M15 Task 009)

3. **Distribute keys to clients** - Share securely via encrypted channels

4. **Update client applications** - Add `Authorization: Bearer <key>` header

5. **Test authentication** - Verify all clients can authenticate

6. **Deploy configuration** - Restart Engram with new configuration

7. **Monitor logs** - Watch for authentication failures

**Backward Compatibility:**

- Default configuration remains `auth_mode = "none"`
- No breaking changes for existing deployments
- Explicit opt-in required for authentication

## Related Documentation

- **[Security Tutorials](../tutorials/)** (M15 Task 009) - Step-by-step guides for enabling authentication
- **[Operations: Security Hardening](../operations/)** - Production security practices
- **[How-To: Configure for Production](../howto/configure-for-production.md)** - Complete production setup
- **[Reference: CLI](../reference/cli.md)** - Command-line options for security

---

## Cluster Configuration

Enable and configure distributed deployment with SWIM membership.

### `cluster.enabled`

**Type:** `bool`
**Default:** `false`
**Valid values:** `true`, `false`

**Purpose:** Enable distributed cluster mode with SWIM membership protocol.

**When to enable:** When deploying 2+ nodes for high availability or horizontal scaling.

**Impact:**
- Enabled: Node joins cluster, participates in replication and space assignment
- Disabled: Single-node mode with no gossip overhead

**Example:**
```toml
[cluster]
enabled = true
node_id = "engram-node-1"
```

### `cluster.node_id`

**Type:** `string`
**Default:** Auto-generated UUID
**Valid values:** Any unique identifier

**Purpose:** Stable node identifier for cluster membership.

**Best practices:**
- Use hostname or static identifier in production
- Ensure uniqueness across cluster
- Do not change after initial deployment

### `cluster.discovery`

**Type:** `object`
**Default:** Static with empty seed list

**Supported Types:**

**1. Static Discovery (Docker Compose)**
```toml
[cluster.discovery]
type = "static"
seed_nodes = [
    "engram-node1:7946",
    "engram-node2:7946",
    "engram-node3:7946"
]
```

**2. DNS Discovery (Kubernetes)**
```toml
[cluster.discovery]
type = "dns"
service = "engram-headless.engram-cluster.svc.cluster.local"
port = 7946
refresh_interval = "30s"
```

**3. Consul Discovery (Service Registry)**
```toml
[cluster.discovery]
type = "consul"
addr = "http://consul.service.consul:8500"
service_name = "engram"
tag = "production"
```

### `cluster.swim`

SWIM protocol tuning parameters.

```toml
[cluster.swim]
ping_interval = "1s"         # Interval between direct probes
ack_timeout = "600ms"        # Probe timeout
suspicion_timeout = "2s"     # Time before suspect → dead
indirect_probes = 2          # Number of indirect probe requests
gossip_batch = 6             # Max rumors per message
```

**Tuning Guidance:**
- **Low latency networks** (<1ms): Reduce timeouts to 300ms/1s
- **High latency networks** (>10ms): Increase to 1s/5s
- **Large clusters** (>10 nodes): Increase `indirect_probes` to 3-4

### `cluster.replication`

Memory space replication configuration.

```toml
[cluster.replication]
factor = 2                    # Number of replicas per space
timeout = "1s"                # Write acknowledgement timeout
placement = "random"          # random | rack-aware | zone-aware
lag_threshold = "5s"          # Max acceptable replication lag
```

**Placement Strategies:**
- `random`: Uniform distribution (default)
- `rack-aware`: Avoid same rack
- `zone-aware`: Avoid same availability zone

### `cluster.network`

Network transport configuration.

```toml
[cluster.network]
swim_bind = "0.0.0.0:7946"          # SWIM gossip UDP port
api_bind = "0.0.0.0:50051"          # gRPC API TCP port
advertise_addr = "10.0.1.5:7946"    # Optional: externally reachable address
```

**Advertise Address:**
- Leave unset for auto-detection (connects to seed to learn local IP)
- Set explicitly when behind NAT or with multiple network interfaces
- Can use `ENGRAM_CLUSTER_ADVERTISE_ADDR` environment variable

### `cluster.partition`

Partition detection parameters.

```toml
[cluster.partition]
majority_threshold = 0.6      # Minimum reachable percentage
detection_window = "10s"      # Healthy window before clearing partition
check_interval = "2s"         # Partition status computation cadence
```

**Monitoring:**
- Check `engram_cluster_partition_state` metric
- Watch logs for "Partition detected" warnings
- Verify `engram_cluster_members_alive` count

For deployment guides, see:
- [Cluster Verification Cookbook](../operations/cluster_verification_cookbook.md)
- [Docker Compose Setup](../../deployments/docker/cluster/README.md)
- [Kubernetes Manifests](../../deployments/kubernetes/engram-cluster.yaml)

---

## Spreading Activation Configuration

Spreading activation is how Engram traverses the cognitive graph, propagating activation from seed memories to related concepts. Think of it like neural activation spreading through a brain network.

### Configuration Structure

Spreading activation is configured in code via `ParallelSpreadingConfig` struct. Future versions will expose these in TOML configuration files.

```rust
pub struct ParallelSpreadingConfig {
    pub decay_rate: f32,
    pub threshold: f32,
    pub max_hops: u16,
    pub max_workers: usize,
    pub latency_budget_ms: Option<u64>,
}

```

### `spreading.decay_rate`

**Type:** `f32`
**Default:** `0.85`
**Valid range:** `0.5` - `0.99`

**Purpose:** Activation decay per hop during spreading. Controls how quickly activation weakens as it propagates through the graph.

**How it works:**

- Activation at hop N: `activation_0 * (decay_rate ^ N)`

- Example with decay_rate=0.85:
  - Hop 0 (seed): 1.0
  - Hop 1: 0.85 (85%)
  - Hop 2: 0.72 (72%)
  - Hop 3: 0.61 (61%)
  - Hop 4: 0.52 (52%)
  - Hop 5: 0.44 (44%)

**Effect on spreading:**

- **Lower values (0.50-0.80):** More localized spreading, focused on immediate neighbors

- **Balanced (0.80-0.90):** Standard spreading with moderate falloff

- **Higher values (0.90-0.99):** Broader spreading, reaches distant associations

**Tuning guide by workload:**

| Workload Type | Decay Rate | Rationale |
|--------------|------------|-----------|
| Precision-focused (legal, medical) | 0.70 - 0.80 | Minimize false positives, stay local |
| Balanced (general assistant) | 0.80 - 0.90 | Balance precision and recall |
| Exploratory (research, discovery) | 0.90 - 0.95 | Surface distant connections |
| Semantic search | 0.85 - 0.88 | Moderate spreading from query terms |
| Analogical reasoning | 0.88 - 0.92 | Bridge conceptual gaps |

**Interaction with other parameters:**

- Works with `max_hops`: Lower decay allows fewer hops

- Works with `threshold`: Higher decay needs lower threshold

**Examples:**

```rust
// Narrow context: Precise, local recalls
let config = ParallelSpreadingConfig {
    decay_rate: 0.75,  // Drops to 0.42 by hop 5
    threshold: 0.25,   // Higher threshold for precision
    max_hops: 3,       // Limit depth
    ..Default::default()
};

// Balanced: General-purpose spreading
let config = ParallelSpreadingConfig {
    decay_rate: 0.85,  // Standard decay
    threshold: 0.15,   // Balanced threshold
    max_hops: 5,       // Standard depth
    ..Default::default()
};

// Broad context: Exploratory, distant associations
let config = ParallelSpreadingConfig {
    decay_rate: 0.92,  // Slow decay (0.66 by hop 5)
    threshold: 0.08,   // Lower threshold for recall
    max_hops: 8,       // Deep traversal
    ..Default::default()
};

```

**Monitoring metrics:**

```promql
# Average activation at different hops
engram_spreading_activation_by_hop{hop="1"}
engram_spreading_activation_by_hop{hop="3"}
engram_spreading_activation_by_hop{hop="5"}

# Result distribution
engram_spreading_results_count_bucket{le="10"}
engram_spreading_results_count_bucket{le="100"}

```

**Troubleshooting:**

**Problem:** Empty or very few results

- **Root cause:** Decay too aggressive, activation drops below threshold

- **Fix:** Increase decay_rate (e.g., 0.75 → 0.85) or lower threshold

- **Validation:** `engram_spreading_results_empty_count` should decrease

**Problem:** Too many low-quality results

- **Root cause:** Decay too slow, activating distant unrelated nodes

- **Fix:** Decrease decay_rate (e.g., 0.95 → 0.85) or raise threshold

- **Validation:** Precision metrics should improve

### `spreading.threshold`

**Type:** `f32`
**Default:** `0.1`
**Valid range:** `0.01` - `0.5`

**Purpose:** Minimum activation level to include in results. Acts as a quality filter - only memories with activation above this threshold are returned.

**Effect on results:**

- **Lower values (0.01-0.10):** More results, higher recall, lower precision

- **Balanced (0.10-0.20):** Standard production setting

- **Higher values (0.20-0.50):** Fewer results, lower recall, higher precision

**Tuning guide by workload:**

| Workload Type | Threshold | Expected Behavior |
|--------------|-----------|-------------------|
| High precision (legal, medical) | 0.20 - 0.30 | Few, highly confident results |
| Balanced (general assistant) | 0.10 - 0.20 | Moderate result set |
| High recall (search, discovery) | 0.05 - 0.10 | Many results, cast wide net |
| Exploratory (research) | 0.03 - 0.08 | Maximum associations |

**Performance impact:**

- Lower threshold = More nodes to process = Higher latency

- Lower threshold = Larger result sets = More memory usage

- Recommend: Start with 0.15, tune based on result quality

**Interaction with decay_rate:**

- High decay (0.70) + High threshold (0.25) = Very narrow spreading

- High decay (0.70) + Low threshold (0.10) = Narrow but inclusive

- Low decay (0.92) + High threshold (0.25) = Broad but selective

- Low decay (0.92) + Low threshold (0.10) = Very broad spreading

**Examples:**

```rust
// High precision: Legal research
let config = ParallelSpreadingConfig {
    decay_rate: 0.75,
    threshold: 0.25,  // Only strong activations
    max_hops: 3,
    ..Default::default()
};

// Balanced: General assistant
let config = ParallelSpreadingConfig {
    decay_rate: 0.85,
    threshold: 0.15,  // Balanced filtering
    max_hops: 5,
    ..Default::default()
};

// High recall: Research exploration
let config = ParallelSpreadingConfig {
    decay_rate: 0.90,
    threshold: 0.05,  // Include weak activations
    max_hops: 8,
    ..Default::default()
};

```

**Monitoring metrics:**

```promql
# Results count distribution
histogram_quantile(0.50, engram_spreading_results_count_bucket)
histogram_quantile(0.95, engram_spreading_results_count_bucket)

# Empty result rate (indicator threshold too high)
rate(engram_spreading_results_empty_count[5m])

# Average minimum activation in results
engram_spreading_min_activation

```

**Troubleshooting:**

**Problem:** Empty result sets

- **Root cause:** Threshold too high for activation levels

- **Fix:** Lower threshold incrementally (e.g., 0.25 → 0.15 → 0.10)

- **Validation:** `engram_spreading_results_empty_count` should drop

**Problem:** Low-quality results (false positives)

- **Root cause:** Threshold too low, including weak activations

- **Fix:** Raise threshold incrementally (e.g., 0.05 → 0.10 → 0.15)

- **Validation:** Manually review result quality, check precision metrics

### `spreading.max_hops`

**Type:** `u16`
**Default:** `5`
**Valid range:** `1` - `20`

**Purpose:** Maximum spreading distance from seed nodes. Controls how far activation can propagate through the graph.

**Effect on spreading:**

- **Lower values (1-2):** Direct connections only, very fast

- **Balanced (3-5):** Standard spreading depth for most use cases

- **Higher values (6-10):** Extended context, explore distant associations

- **Very high (11-20):** Research/experimental, significant latency

**Relationship with graph theory:**

- Most real-world graphs have small-world property

- Average path length in social networks: 4-6 hops

- Cognitive associations: typically within 3-5 hops

- Diminishing returns: Activation often too weak beyond 7-8 hops

**Tuning guide by workload:**

| Workload Type | Max Hops | Rationale |
|--------------|----------|-----------|
| Direct lookup | 1 - 2 | Immediate neighbors only |
| Local context | 3 - 5 | Standard cognitive distance |
| Extended context | 6 - 8 | Deep relationship exploration |
| Research queries | 9 - 12 | Distant conceptual bridges |
| Experimental | 13 - 20 | Full graph exploration |

**Performance impact:**

- Latency grows **exponentially** with hops (due to branching factor)

- Example with average branching factor of 10:
  - Hop 1: 10 nodes
  - Hop 2: 100 nodes
  - Hop 3: 1,000 nodes
  - Hop 4: 10,000 nodes
  - Hop 5: 100,000 nodes (may hit capacity limits)

**Interaction with other parameters:**

- High max_hops + High decay_rate: Activation survives to distant nodes

- High max_hops + Low decay_rate: Activation dies before reaching max hops

- High max_hops + Low threshold: Massive result sets

**Examples:**

```rust
// Fast direct connections
let config = ParallelSpreadingConfig {
    decay_rate: 0.85,
    threshold: 0.15,
    max_hops: 2,  // Immediate neighbors only
    max_workers: 8,
    ..Default::default()
};

// Standard cognitive spreading
let config = ParallelSpreadingConfig {
    decay_rate: 0.85,
    threshold: 0.15,
    max_hops: 5,  // Balance depth and performance
    max_workers: 16,
    ..Default::default()
};

// Deep exploratory search
let config = ParallelSpreadingConfig {
    decay_rate: 0.92,  // Higher to survive deep hops
    threshold: 0.08,   // Lower to catch weak distant signals
    max_hops: 10,      // Deep traversal
    max_workers: 32,   // More parallelism for large search
    ..Default::default()
};

```

**Monitoring metrics:**

```promql
# Activation depth histogram
engram_spreading_depth_histogram_bucket{le="3"}
engram_spreading_depth_histogram_bucket{le="5"}
engram_spreading_depth_histogram_bucket{le="8"}

# Nodes visited per hop
engram_spreading_nodes_by_hop{hop="1"}
engram_spreading_nodes_by_hop{hop="5"}

# Latency by max_hops configuration
engram_spreading_latency_seconds{max_hops="5"}

```

**Troubleshooting:**

**Problem:** Queries timing out

- **Root cause:** Max hops too high, exponential explosion

- **Fix:** Reduce max_hops (e.g., 10 → 5) or increase max_workers

- **Validation:** P99 latency should drop below budget

**Problem:** Missing relevant results

- **Root cause:** Max hops too low, not reaching related concepts

- **Fix:** Increase max_hops incrementally (e.g., 3 → 5 → 7)

- **Validation:** Recall metrics should improve

### `spreading.max_workers`

**Type:** `usize`
**Default:** `num_cpus::get()` (number of CPU cores)
**Valid range:** `1` - `2 * num_cpus`

**Purpose:** Thread pool size for parallel spreading. Controls how many worker threads participate in graph traversal.

**How parallelism works:**

- Work-stealing queue: Workers dynamically balance load

- Lock-free data structures: Minimize contention

- Cache-aware scheduling: Optimize for NUMA systems

**Tuning guide by workload:**

| Workload Type | Worker Count | Rationale |
|--------------|--------------|-----------|
| CPU-bound (in-memory graph) | num_cpus | Maximize CPU utilization |
| I/O-bound (disk-backed graph) | 2 * num_cpus | Overlap I/O with compute |
| Low latency priority | num_cpus / 2 | Reduce scheduling overhead |
| High throughput | num_cpus | Standard parallelism |
| Batch processing | 2 * num_cpus | Maximize throughput |

**Hardware considerations:**

- **Single-socket:** Use num_cpus

- **Multi-socket (NUMA):** Use num_cpus per socket

- **Hyperthreading:** Physical cores only for best performance

- **Shared hosting:** Limit to allocated cores

**Performance characteristics:**

- **Scaling efficiency:** 80-90% parallel efficiency up to 32 workers

- **Diminishing returns:** Beyond 32 workers, overhead increases

- **Memory pressure:** Each worker allocates ~1MB for working set

**Examples:**

```rust
// Low latency (minimize overhead)
let config = ParallelSpreadingConfig {
    max_workers: 4,  // Half of 8-core system
    ..Default::default()
};

// Balanced (standard setting)
let config = ParallelSpreadingConfig {
    max_workers: num_cpus::get(),  // Match CPU count
    ..Default::default()
};

// High throughput (I/O bound)
let config = ParallelSpreadingConfig {
    max_workers: num_cpus::get() * 2,  // Oversubscribe for I/O overlap
    ..Default::default()
};

// Manual tuning for 16-core NUMA system
let config = ParallelSpreadingConfig {
    max_workers: 16,  // Explicit count
    ..Default::default()
};

```

**Monitoring metrics:**

```promql
# Parallel efficiency (target: >0.80)
engram_spreading_parallel_efficiency

# Worker utilization
engram_spreading_worker_utilization

# Work stealing events (load balancing)
rate(engram_spreading_work_stealing_total[5m])

# Queue depth by worker
engram_spreading_queue_depth{worker="0"}

```

**Troubleshooting:**

**Problem:** Low parallel efficiency (<0.70)

- **Root cause:** Too many workers, scheduling overhead dominates

- **Fix:** Reduce max_workers (e.g., 64 → 32 → 16)

- **Validation:** Efficiency should increase to >0.80

**Problem:** CPU underutilization

- **Root cause:** Too few workers for available cores

- **Fix:** Increase max_workers to match num_cpus

- **Validation:** CPU utilization should increase

**Problem:** High context switching

- **Root cause:** Workers exceeding physical cores

- **Fix:** Set max_workers = physical CPU cores (not logical)

- **Validation:** Context switch rate should decrease

### `spreading.latency_budget_ms`

**Type:** `Option<u64>`
**Default:** `None` (no budget enforcement)
**Valid range:** `10` - `10000` (milliseconds)

**Purpose:** Maximum allowed latency for spreading activation. If spreading exceeds this budget, it terminates early with partial results.

**When to set:**

- **Interactive systems:** User-facing queries need <100ms response

- **Real-time agents:** Decision loops need <50ms latency

- **Batch processing:** No budget needed, prioritize completeness

**Budget enforcement:**

- Checked at each hop boundary

- Graceful degradation: Returns best results so far

- Metrics logged: `engram_spreading_latency_budget_violations_total`

**Examples:**

```rust
// Interactive assistant (strict latency)
let config = ParallelSpreadingConfig {
    latency_budget_ms: Some(50),  // 50ms budget
    max_hops: 3,                  // Limit depth to meet budget
    ..Default::default()
};

// Web API (moderate latency)
let config = ParallelSpreadingConfig {
    latency_budget_ms: Some(200),  // 200ms budget
    max_hops: 5,
    ..Default::default()
};

// Batch analytics (no budget)
let config = ParallelSpreadingConfig {
    latency_budget_ms: None,  // Prioritize completeness
    max_hops: 10,
    ..Default::default()
};

```

**Monitoring metrics:**

```promql
# Budget violation rate
rate(engram_spreading_latency_budget_violations_total[5m])

# Actual latency vs budget
histogram_quantile(0.99, engram_spreading_latency_seconds_bucket)

# Partial result rate (budget hit before completion)
rate(engram_spreading_partial_results_total[5m])

```

---

## Consolidation Configuration

Memory consolidation transforms episodic memories into semantic knowledge through pattern detection and compaction. This mimics how the brain transfers memories from hippocampus to neocortex during sleep.

### Configuration Structure

Consolidation is configured via multiple structs:

```rust
pub struct ConsolidationConfig {
    pub pattern_detection: PatternDetectionConfig,
    pub dream: DreamConfig,
    pub compaction: CompactionConfig,
}

```

### Pattern Detection Configuration

#### `pattern_detection.min_cluster_size`

**Type:** `usize`
**Default:** `3`
**Valid range:** `2` - `20`

**Purpose:** Minimum episodes required to form a pattern cluster. Higher values produce fewer, stronger patterns.

**Effect:**

- **Lower (2-3):** More patterns, some weak

- **Balanced (3-5):** Standard pattern strength

- **Higher (6-10):** Fewer, very strong patterns

**Tuning guide:**

- **Exploratory systems:** 2-3 (detect emerging patterns early)

- **Production systems:** 3-5 (balance sensitivity and strength)

- **High-precision systems:** 5-10 (only strong, recurring patterns)

**Examples:**

```rust
// Early pattern detection
let config = PatternDetectionConfig {
    min_cluster_size: 2,  // Detect as soon as pattern repeats
    similarity_threshold: 0.85,  // Higher threshold for quality
    max_patterns: 100,
};

// Balanced production
let config = PatternDetectionConfig {
    min_cluster_size: 3,  // Standard threshold
    similarity_threshold: 0.80,
    max_patterns: 100,
};

// High-confidence patterns only
let config = PatternDetectionConfig {
    min_cluster_size: 8,  // Strong recurrence required
    similarity_threshold: 0.85,
    max_patterns: 50,
};

```

#### `pattern_detection.similarity_threshold`

**Type:** `f32`
**Default:** `0.8`
**Valid range:** `0.5` - `0.95`

**Purpose:** Cosine similarity threshold for clustering episodes into patterns. Higher values require more similar episodes.

**Effect:**

- **Lower (0.50-0.70):** Broader patterns, thematic grouping

- **Balanced (0.75-0.85):** Standard semantic similarity

- **Higher (0.85-0.95):** Specific patterns, near-duplicates only

**Interpretation:**

- 0.50: Loosely related (different topics, common domain)

- 0.70: Moderately related (same domain, different specifics)

- 0.80: Closely related (same topic, different perspectives)

- 0.90: Very similar (same topic, minor variations)

- 0.95: Near-identical (almost duplicates)

**Tuning guide:**

```rust
// Broad thematic patterns (exploratory)
let config = PatternDetectionConfig {
    min_cluster_size: 3,
    similarity_threshold: 0.65,  // Loose grouping
    max_patterns: 200,
};

// Standard semantic clustering
let config = PatternDetectionConfig {
    min_cluster_size: 3,
    similarity_threshold: 0.80,  // Balanced
    max_patterns: 100,
};

// Specific recurring patterns
let config = PatternDetectionConfig {
    min_cluster_size: 5,
    similarity_threshold: 0.90,  // Very similar required
    max_patterns: 50,
};

```

#### `pattern_detection.max_patterns`

**Type:** `usize`
**Default:** `100`
**Valid range:** `10` - `1000`

**Purpose:** Maximum patterns extracted per consolidation cycle. Limits memory usage and processing time.

**Performance impact:**

- Each pattern: ~800KB (embedding + metadata + source episodes)

- 100 patterns = ~80MB RAM during consolidation

- 1000 patterns = ~800MB RAM during consolidation

**Tuning guide:**

- **Memory-constrained:** 10-50 patterns

- **Balanced:** 100-200 patterns

- **High-capacity:** 500-1000 patterns

### Dream Configuration

#### `dream.dream_duration`

**Type:** `Duration`
**Default:** `600s` (10 minutes)
**Valid range:** `60s` - `3600s` (1 minute - 1 hour)

**Purpose:** Length of each offline consolidation cycle. Mimics sleep cycles in biological brains.

**Biological basis:**

- Human sleep cycles: 90 minutes

- REM sleep (consolidation): ~20 minutes per cycle

- Scaled for computational efficiency: 5-10 minutes

**Tuning guide:**

```rust
// Rapid consolidation (development/testing)
let config = DreamConfig {
    dream_duration: Duration::from_secs(60),  // 1 minute
    ..Default::default()
};

// Balanced production
let config = DreamConfig {
    dream_duration: Duration::from_secs(300),  // 5 minutes
    ..Default::default()
};

// Extended consolidation (research)
let config = DreamConfig {
    dream_duration: Duration::from_secs(1800),  // 30 minutes
    ..Default::default()
};

```

#### `dream.replay_speed`

**Type:** `f32`
**Default:** `15.0`
**Valid range:** `10.0` - `20.0`

**Purpose:** Replay speed multiplier (faster than real-time). Based on hippocampal replay research showing 10-20x speedup.

**Biological basis:**

- Hippocampal replay during sleep: 10-20x real-time

- Sharp-wave ripples: 150-250 Hz (vs 5-10 Hz during encoding)

- Enables rapid consolidation of day's experiences

**Effect:**

- Higher replay speed = More episodes processed per dream cycle

- Lower replay speed = More careful consolidation

#### `dream.min_episode_age`

**Type:** `Duration`
**Default:** `86400s` (1 day)
**Valid range:** `3600s` - `604800s` (1 hour - 1 week)

**Purpose:** Minimum age before episode eligible for consolidation. Allows initial stabilization before pattern extraction.

**Tuning guide by use case:**

| Use Case | Min Episode Age | Rationale |
|----------|----------------|-----------|
| Rapid consolidation | 1 hour | Immediate pattern extraction |
| Development/testing | 1 hour | Fast iteration |
| Balanced production | 1 day | Daily consolidation |
| Stable patterns | 3 days | Let patterns stabilize |
| Conservative | 1 week | Maximum stability |

**Examples:**

```rust
// Rapid consolidation (testing)
let config = DreamConfig {
    min_episode_age: Duration::from_secs(3600),  // 1 hour
    dream_duration: Duration::from_secs(300),
    ..Default::default()
};

// Production (daily consolidation)
let config = DreamConfig {
    min_episode_age: Duration::from_secs(86400),  // 1 day
    dream_duration: Duration::from_secs(600),
    ..Default::default()
};

// Conservative (weekly consolidation)
let config = DreamConfig {
    min_episode_age: Duration::from_secs(604800),  // 1 week
    dream_duration: Duration::from_secs(1800),
    ..Default::default()
};

```

### Compaction Configuration

#### `compaction.min_age`

**Type:** `Duration`
**Default:** `86400s` (1 day)
**Valid range:** `3600s` - `604800s`

**Purpose:** Minimum episode age before compaction eligible. Ensures episodes have stabilized before compression.

#### `compaction.target_compression_ratio`

**Type:** `f32`
**Default:** `0.5`
**Valid range:** `0.3` - `0.7`

**Purpose:** Target size reduction after compaction (0.5 = 50% reduction).

**Effect:**

- 0.3: Aggressive compaction (70% reduction)

- 0.5: Balanced compaction (50% reduction)

- 0.7: Conservative compaction (30% reduction)

**Trade-offs:**

- Aggressive: More disk savings, some information loss

- Conservative: Less disk savings, preserves more details

---

## Decay Configuration

Controls forgetting curves and memory strength over time. Based on cognitive psychology research (Ebbinghaus, power law decay).

### Configuration Structure

```rust
pub struct DecayConfig {
    pub base_decay_rate: f32,
    pub activation_boost: f32,
    pub consolidation_benefit: f32,
}

```

### `decay.base_decay_rate`

**Type:** `f32`
**Default:** `0.001`
**Valid range:** `0.0001` - `0.01`

**Purpose:** Daily decay rate for unused memories. Controls how quickly memories fade without rehearsal.

**Biological basis:**

- Ebbinghaus forgetting curve: Exponential decay

- Typical retention: 50% after 1 day, 30% after 1 week

- Formula: `strength_t = strength_0 * e^(-base_decay_rate * t)`

**Effect:**

- Lower (0.0001): Slow forgetting, long retention

- Balanced (0.001): Matches empirical data

- Higher (0.01): Rapid forgetting, aggressive pruning

**Tuning guide:**

```rust
// Long-term knowledge base (encyclopedic)
let config = DecayConfig {
    base_decay_rate: 0.0001,  // Very slow decay
    ..Default::default()
};

// Balanced (matches cognitive research)
let config = DecayConfig {
    base_decay_rate: 0.001,  // Standard decay
    ..Default::default()
};

// Short-term working memory
let config = DecayConfig {
    base_decay_rate: 0.01,  // Rapid decay
    ..Default::default()
};

```

### `decay.activation_boost`

**Type:** `f32`
**Default:** `0.1`
**Valid range:** `0.01` - `0.5`

**Purpose:** Strength increase per memory access. Implements spaced repetition effect.

**Effect:**

- Each recall: `strength += activation_boost`

- Repeated recalls strengthen memory over time

- Spaced repetition: More effective than massed practice

**Tuning guide:**

- Learning systems: 0.2-0.5 (strong reinforcement)

- Production systems: 0.1-0.2 (balanced strengthening)

- Read-heavy systems: 0.05-0.1 (minimal strengthening)

---

## Environment-Specific Configuration Examples

### Development Configuration

```toml
# Development: Minimal resources, fast iteration
[feature_flags]
spreading_api_beta = true

[memory_spaces]
default_space = "dev"
bootstrap_spaces = ["dev", "test"]

[persistence]
data_root = "~/.local/share/engram-dev"
hot_capacity = 10_000        # 120MB RAM
warm_capacity = 100_000      # 1GB disk
cold_capacity = 1_000_000    # 8GB disk

```

### Staging Configuration

```toml
# Staging: Medium capacity for pre-production testing
[feature_flags]
spreading_api_beta = true

[memory_spaces]
default_space = "staging"
bootstrap_spaces = ["staging"]

[persistence]
data_root = "/opt/engram-staging"
hot_capacity = 100_000       # 1.2GB RAM
warm_capacity = 1_000_000    # 10GB disk
cold_capacity = 10_000_000   # 80GB disk

```

### Production Configuration

```toml
# Production: High-capacity, multi-tenant deployment
[feature_flags]
spreading_api_beta = true

[memory_spaces]
default_space = "default"
bootstrap_spaces = ["default", "tenant-a", "tenant-b"]

[persistence]
data_root = "/var/lib/engram"
hot_capacity = 500_000       # 6GB RAM per space
warm_capacity = 5_000_000    # 50GB disk per space
cold_capacity = 50_000_000   # 400GB disk per space

```

---

## Capacity Planning

### Memory Requirements

**Hot tier RAM calculation:**

```
hot_tier_ram = hot_capacity * 12KB * num_spaces

```

**Examples:**

- 100K hot, 1 space: 100,000 * 12KB = 1.2GB

- 500K hot, 3 spaces: 500,000 \* 12KB \* 3 = 18GB

- 1M hot, 10 spaces: 1,000,000 \* 12KB \* 10 = 120GB

**Disk Requirements**

**Warm tier disk:**

```
warm_tier_disk = warm_capacity * 10KB * num_spaces

```

**Cold tier disk:**

```
cold_tier_disk = cold_capacity * 8KB * num_spaces

```

**Total disk:**

```
total_disk = (warm_tier_disk + cold_tier_disk) * 1.2  # 20% overhead

```

**Example calculations:**

**Small deployment (1M corpus, 1 space):**

- Hot: 100K * 12KB = 1.2GB RAM

- Warm: 500K * 10KB = 5GB disk

- Cold: 400K * 8KB = 3.2GB disk

- Total: 1.2GB RAM + 9.8GB disk (12GB with overhead)

**Medium deployment (10M corpus, 3 spaces):**

- Hot: 500K \* 12KB \* 3 = 18GB RAM

- Warm: 5M \* 10KB \* 3 = 150GB disk

- Cold: 4.5M \* 8KB \* 3 = 108GB disk

- Total: 18GB RAM + 310GB disk (372GB with overhead)

**Large deployment (100M corpus, 10 spaces):**

- Hot: 1M \* 12KB \* 10 = 120GB RAM

- Warm: 10M \* 10KB \* 10 = 1TB disk

- Cold: 89M \* 8KB \* 10 = 7.1TB disk

- Total: 120GB RAM + 9.7TB disk (11.6TB with overhead)

---

## Monitoring Metrics Reference

### Storage Metrics

```promql
# Hot tier utilization
engram_storage_hot_tier_size{space="default"} / engram_storage_hot_capacity

# Hot tier hit rate (target: >0.95)
engram_storage_hot_tier_hit_rate{space="default"}

# Warm tier size
engram_storage_warm_tier_size{space="default"}

# Cold tier size
engram_storage_cold_tier_size{space="default"}

# Migration rate (hot to warm)
rate(engram_storage_migrations_total{from="hot",to="warm"}[5m])

# Disk usage
engram_storage_disk_usage_bytes{tier="warm",space="default"}
engram_storage_disk_usage_bytes{tier="cold",space="default"}

```

### Spreading Metrics

```promql
# Spreading latency (P99)
histogram_quantile(0.99, engram_spreading_latency_seconds_bucket)

# Parallel efficiency (target: >0.80)
engram_spreading_parallel_efficiency

# Results count distribution
histogram_quantile(0.50, engram_spreading_results_count_bucket)

# Budget violations
rate(engram_spreading_latency_budget_violations_total[5m])

# Worker utilization
engram_spreading_worker_utilization

```

### Consolidation Metrics

```promql
# Pattern detection rate
rate(engram_consolidation_patterns_detected_total[1h])

# Compaction effectiveness
engram_storage_compaction_ratio{tier="cold"}

# Dream cycle duration
engram_consolidation_dream_duration_seconds

```

### Decay Metrics

```promql
# Average memory strength
engram_memory_avg_strength

# Decay applications
rate(engram_decay_applications_total[5m])

# Forgotten memories
rate(engram_memory_forgotten_total[1h])

```

---

## Configuration Validation

### Validation Rules

**Memory space IDs:**

- Length: 3-64 characters

- Characters: `a-z`, `A-Z`, `0-9`, `-`, `_`

- Must not start with `-` or `_`

**Capacity constraints:**

- `hot_capacity ≥ 1,000`

- `warm_capacity ≥ 10,000`

- `cold_capacity ≥ 100,000`

- Recommend: `hot_capacity ≤ warm_capacity ≤ cold_capacity`

**Spreading constraints:**

- `0.5 ≤ decay_rate ≤ 0.99`

- `0.01 ≤ threshold ≤ 0.5`

- `1 ≤ max_hops ≤ 20`

- `1 ≤ max_workers ≤ 256`

**Decay constraints:**

- `0.0001 ≤ base_decay_rate ≤ 0.01`

- `0.01 ≤ activation_boost ≤ 0.5`

### Validation Commands

**Shell validation:**

```bash
./scripts/validate_config.sh config.toml

```

**Rust validation:**

```bash
engram validate config --config config.toml

```

**Docker validation:**

```bash
docker run engram:latest engram validate config --config /etc/engram/config.toml

```

---

## Configuration Migration

### Upgrading from Previous Versions

**Version compatibility:**

- v0.1.x → v0.2.x: No breaking changes

- v0.2.x → v0.3.x: Add `bootstrap_spaces` field

- Future versions: Migration tool provided

**Migration command:**

```bash
engram config migrate old.toml new.toml

```

---

## Further Reading

- [Operations: Configuration Management](/docs/operations/configuration-management.md) - Scenario-based tuning

- [Operations: Troubleshooting](/docs/operations/configuration-troubleshooting.md) - Common mistakes

- [How-To: Configure for Production](/docs/howto/configure-for-production.md) - Complete walkthroughs

- [Reference: CLI](/docs/reference/cli.md) - Command-line options

- [Operations: Performance Tuning](/docs/operations/performance-tuning.md) - Performance optimization

## Cluster Replication Configuration

Distributed deployments rely on the `[cluster.replication]` table for placement and streaming behavior.

### `cluster.replication.factor`

**Type:** `integer` — **Default:** `2`

Number of replicas per memory space (excluding the primary). Increase for higher durability at the cost of additional storage and replication bandwidth.

### `cluster.replication.timeout`

**Type:** `duration string` — **Default:** `"1s"`

Upper bound for write acknowledgements. Writes are accepted optimistically, but this timeout bounds how long the primary waits for replica confirmations before logging a warning.

### `cluster.replication.placement`

**Type:** `string` (`random`, `rackaware`, `zoneaware`)

Replica diversity strategy. Default `random` is topology agnostic; rack/zone aware penalties spread replicas across failure domains recorded on `NodeInfo`.

### `cluster.replication.jump_buckets`

**Type:** `integer`

Number of virtual buckets used by jump-consistent hashing. Higher values smooth redistribution when nodes join/leave (16,384 is typically sufficient for thousands of spaces).

### `cluster.replication.rack_penalty` / `zone_penalty`

**Type:** `float` in `[0.0, 1.0]`

Penalty multipliers applied when a candidate replica shares the same rack/zone as a previously-selected node. Lower values strongly discourage co-locating replicas.

### `cluster.replication.lag_threshold`

**Type:** `duration string` — **Default:** `"5s"`

Maximum acceptable replication lag before Engram emits warnings and surfaces red indicators in `/cluster/health`. When the local → replica sequence gap exceeds this threshold, operators should investigate network partitions or slow disks.

### `cluster.replication.catch_up_batch_bytes`

**Type:** `integer` (bytes) — **Default:** `2_097_152`

Target payload size for each replication batch. Larger values improve throughput when recovering a replica, while smaller values minimize steady-state latency.

### `cluster.replication.compression`

**Type:** `string` (`none`, `lz4`, `zstd`) — **Default:** `none`

Compression algorithm applied to replication batches. Leave `none` for low-latency LANs. Use `lz4` or `zstd` when WAN bandwidth is constrained.

### `cluster.replication.io_uring_enabled`

**Type:** `bool` — **Default:** `false`

Enables io_uring fast paths when running on Linux kernels that support it. Set to `false` on macOS or kernels older than 5.10.
