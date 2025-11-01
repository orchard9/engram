# Documentation Changelog

## 2025-11-01 - M13 Cognitive Patterns Complete + M14 Prerequisites Met

### Added

- **Cognitive Patterns Engine** (Milestone 13): Production-ready cognitive psychology validation
  - Semantic priming (Neely 1977 validation)
  - Proactive and retroactive interference detection (Anderson 1974, McGeoch 1942)
  - Fan effect implementation with empirical validation
  - Reconsolidation core with memory restabilization (Nader 2000)
  - DRM false memory paradigm validation (Roediger & McDermott 1995)
  - Spacing effect validation (Bjork & Bjork 1992)
  - Zero-overhead metrics (<1% performance impact, zero-cost when disabled)
  - Comprehensive Grafana dashboards for cognitive pattern monitoring

- **M14 Prerequisites Completed**:
  - Consolidation determinism fixes (5 core improvements with property-based validation)
  - 100% test health achieved (1,035/1,035 passing, zero clippy warnings)
  - Critical bug fix: Two-component decay model spacing effect (now follows desirable difficulties principle)

### Changed

- Pattern completion status: Beta → Production ready (M8 validated)
- Test coverage: Added 300+ cognitive pattern validation tests
- Consolidation: Now deterministic and ready for distributed architecture

## 2025-10-26 - Production Operations & GPU Infrastructure

### Added

- **GPU Acceleration Infrastructure** (Milestone 12): CUDA kernel framework
  - CUDA build system with graceful CPU fallback
  - Hybrid CPU/GPU executor with 5-rule decision tree
  - Unified memory allocator with zero-copy transfers
  - OOM prevention with automatic batch splitting
  - Cross-platform support (Windows/Linux/macOS)
  - GPU operations guide with 3,458 lines of documentation

- **Production Operations Documentation** (Milestone 16):
  - Container orchestration and deployment guides
  - Backup/disaster recovery procedures
  - Performance tuning and capacity planning
  - Security hardening guides (authentication, authorization, audit logging)
  - Database migration tooling (Neo4j, PostgreSQL, Redis)
  - Comprehensive troubleshooting runbooks
  - Operations CLI enhancements

### Note

- GPU acceleration: Infrastructure complete, CUDA kernels awaiting hardware validation
- Current production: CPU SIMD optimizations active (Zig kernels 15-35% faster)

## 2025-10-25 - Zig Performance Kernels

### Added

- **Zig Performance Kernels** (Milestone 10):
  - Vector similarity kernel (25% improvement)
  - Activation spreading kernel (35% improvement)
  - Memory decay kernel (27% improvement)
  - Zero-copy FFI with thread-safe arena allocators
  - 30,000+ differential tests with 1e-6 epsilon tolerance
  - 1,815 lines of operations documentation
  - Performance regression framework

## 2025-10-23 - Multi-Interface Layer & Query Language

### Added

- **Query Language** (Milestone 9): SQL-like syntax for memory operations
  - RECALL operation with confidence filters
  - SPREAD operation for multi-hop activation spreading
  - CONSOLIDATE operation for pattern detection
  - COMPLETE operation for pattern completion
  - IMAGINE operation for confidence-based reconstruction
  - Recursive descent parser with helpful error messages
  - HTTP/gRPC query execution endpoints

- **Multi-Interface Layer** (Milestone 15):
  - HTTP REST API with OpenAPI/Swagger documentation
  - gRPC service with streaming support
  - Server-Sent Events (SSE) for real-time monitoring
  - CORS support for web clients
  - Multi-tenant routing via X-Memory-Space header
  - Contract tests validating API equivalence

## 2025-10-21 - Memory Consolidation & Multi-Tenancy

### Added

- **Multi-Tenant Memory Spaces** (Milestone 7): Complete isolation infrastructure for multi-tenant deployments
  - `MemorySpaceRegistry` for thread-safe space management using DashMap
  - Per-space persistence with isolated WAL/tier storage directories
  - `X-Memory-Space` HTTP header for space routing (header → query → body precedence)
  - CLI commands: `space list`, `space create`, `status --space <name>`
  - Per-space health metrics endpoint `/api/v1/system/health` with spaces array
  - Automatic space discovery and WAL recovery on startup
  - 100% backward compatibility via `default` space fallback

- Comprehensive migration guide at `docs/operations/memory-space-migration.md`

- Memory Spaces section added to README with usage examples

- Multi-tenancy documentation in API reference and usage guide

### Changed

- Health endpoint response structure: changed from `{"memory": {...}}` to `{"spaces": [{...}]}`

- Status CLI command now displays per-space metrics in formatted table

- gRPC message schema: added `memory_space_id` field to request messages

### Known Issues

- HTTP routing gap: X-Memory-Space header extracted but not fully wired to operations (Task 004 follow-up pending)

- Streaming API: Space extraction added but full isolation incomplete (Task 005c follow-up pending)

- Health endpoint: Response format mismatch in validation tests (Task 006b follow-up pending)

### Previous Features

- Marked the spreading activation API as **beta**. The CLI now manages a `spreading_api_beta` feature flag stored in `~/.config/engram/config.toml` with defaults from `engram-cli/config/default.toml`.

- Added Diátaxis-aligned documentation for spreading activation (tutorial, how-to guides, explanation, and reference), along with deterministic visualization tooling.
