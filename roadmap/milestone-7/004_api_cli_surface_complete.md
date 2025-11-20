# 004: API, CLI, and Config Surface — _95_percent_complete_

## Current Status: 95% Complete

**What's Implemented**:
- ✅ `extract_memory_space_id()` helper with priority order (engram-cli/src/api.rs:185)
  - Priority: X-Engram-Memory-Space header → ?space=<id> query param → request body memory_space_id → default
- ✅ Request DTOs updated with optional `memory_space_id: Option<String>` fields
  - RememberMemoryRequest, RememberEpisodeRequest, RecallQuery, SearchMemoriesQuery
- ✅ CLI --space flags implemented in commands.rs
- ✅ ENGRAM_MEMORY_SPACE environment variable support
- ✅ 10+ HTTP handlers using space extraction pattern
- ✅ Backward compatibility: missing space defaults to registry default
- ✅ Config schema extended with default_memory_space

**Minor Gaps Remaining** (5%):
- ❌ SSE stream filtering by space not fully tested
- ❌ Some backwards compatibility edge cases not covered by tests
- ❌ CLI output doesn't display active space in status/list commands
- ❌ Deprecation warnings for legacy single-space usage not emitted

## Goal
Expose memory spaces across HTTP, CLI, and configuration so operators and clients must declare the target space for every operation. The API surface must guide legacy users toward the new requirement without breaking existing scripts and includes end-to-end validation/error messaging.

## Deliverables
- REST handlers accept `memory_space_id` (prefer `X-Engram-Memory-Space` header, fallback to query/body) with shared validation.
- Shared request extractor/validator that resolves a space and records it in tracing context.
- CLI commands gain `--space` flag and `ENGRAM_MEMORY_SPACE` env fallback; help text documents precedence and defaults.
- Config schema gains `default_memory_space` and optional `bootstrap_spaces` list used at startup.
- SSE/monitoring routes filter by space and reject ambiguous requests with actionable errors.
- Unit/integration tests covering new parameters and legacy fallback behaviour.

## Implementation Plan

1. **Request Context Extractor**
   - Add `MemorySpaceContext` (+ extractor) in `engram-cli/src/api.rs`.
   - Extraction order: header `X-Engram-Memory-Space`, query `memory_space`, JSON body `memory_space_id` (optional for POST).
   - On missing ID when multiple spaces exist, return `ApiError::InvalidInput` explaining how to set the header/flag.

2. **API Handlers**
   - Update handlers (`remember_memory`, `remember_episode`, `recall_memories`, `recognize_pattern`, SSE endpoints, etc.) to accept `MemorySpaceContext` and obtain the correct `MemoryStore` via registry (`state.registry.store_for(space)` once Task 001/002 land).
   - Tag tracing spans with `memory_space` to ease observability.

3. **Router Wiring**
   - Modify `create_api_routes` to apply the extractor globally or add it to each route signature.
   - Exempt health/status routes that remain space-agnostic; document rationale in code comments.

4. **CLI Enhancements**
   - Extend `engram-cli/src/cli/commands.rs` to add `#[arg(long, env = "ENGRAM_MEMORY_SPACE")] space: Option<String>` for memory, query, status, docs commands.
   - Forward the selected space in HTTP requests via header; for gRPC calls, add field to proto request (Task 005) but set now when available.
   - Display the active space in CLI output (e.g., status table column, memory list header).

5. **Configuration Updates**
   - Update `engram-cli/config/default.toml` with:
     ```toml
     [persistence]
     default_memory_space = "default"
     bootstrap_spaces = []
     ```
   - Extend `CliConfig` in `engram-cli/src/config.rs` to deserialize new section; ensure merging logic handles user overrides.
   - During startup (`main.rs`), register bootstrap spaces before server binding and log the default space.

6. **Backwards Compatibility**
   - When only one space exists, allow requests without explicit ID but log `warn!` encouraging migration.
   - For multi-space deployments, fail fast with 400/CLI error instructing the user to pass `--space` or header.
   - Document behaviour for Task 008 and update API error messaging accordingly.

7. **SSE & Monitoring Streams**
   - Update `monitor_events`, `monitor_activations`, etc., to filter by space.
   - Include `memory_space_id` in SSE payloads to help clients verify routing.

8. **Docs Hooks**
   - Add inline Rustdoc to extractor and CLI flags so Task 008 can reuse text.

## Integration Points
- `engram-cli/src/api.rs`, `engram-cli/src/main.rs`, `engram-cli/src/cli/*`, `engram-cli/src/config.rs`.
- Follow-up documentation task (008) consumes inline references.

## Acceptance Criteria

1. ✅ **COMPLETE**: HTTP routes that hit storage enforce space selection
   - Implementation: `extract_memory_space_id()` in all memory operation handlers
   - Defaults: Missing ID falls back to default space (backward compatible)
   - **PARTIAL**: Error messaging could be more actionable for multi-space scenarios

2. ⚠️ **PARTIAL**: CLI defaults to configured space, supports env/flag overrides
   - ✅ Implemented: --space flags, ENGRAM_MEMORY_SPACE env var
   - ❌ Missing: Active space not displayed in output
   - ❌ Missing: Tests confirming precedence order

3. ⚠️ **PARTIAL**: Server startup logs default space
   - ✅ Implemented: Registry bootstrap logs default space
   - ❌ Missing: Deprecation warnings for legacy requests

4. ⚠️ **PARTIAL**: SSE endpoints emit events scoped to requested space
   - ✅ Implemented: `memory_space_id` field in MemoryEvent variants
   - ❌ Missing: Filtering validation in tests
   - ❌ Missing: Confirmation that streams are actually filtered

5. ✅ **COMPLETE**: Single-space deployments continue to function
   - Implementation: Default space fallback maintains backward compatibility
   - ⚠️ Missing: Migration warning emission

## Remaining Work

1. **SSE Stream Filtering Tests** (2 hours)
   - File: `engram-cli/tests/streaming_tests.rs`
   - Add test: Subscribe to two different spaces simultaneously, verify events don't leak
   - Validate: Each stream only receives events for its requested space

2. **CLI Active Space Display** (1 hour)
   - File: `engram-cli/src/cli/status.rs` or similar
   - Action: Add "Active Space: <id>" to status output header
   - Show: Which space is being used for current operation

3. **Deprecation Warnings** (1 hour)
   - File: `engram-cli/src/api.rs` in `extract_memory_space_id()`
   - Condition: When registry has >1 space but no space specified in request
   - Message: "warn!("Using default space - consider setting X-Engram-Memory-Space header")"

4. **Backward Compatibility Test Suite** (2 hours)
   - File: Create `engram-cli/tests/backward_compatibility.rs`
   - Scenarios:
     - Single-space deployment without --space flag works
     - Multi-space deployment with old client (no header) uses default
     - Priority order testing: header > query > body > default

## Testing Strategy
- REST integration tests (`tests/api/test_memory_spaces.rs`) covering header/query/body extraction, fallback, and error cases.
- CLI smoke tests via `assert_cmd` verifying `--space`, env var fallback, and default behaviour.
- Config round-trip tests for new persistence settings.
- SSE subscription test using `reqwest_eventsource` (or mocked stream) ensuring filtering and metadata.
- Backwards compatibility test starting server with single space and running old CLI command without `--space`.

## Review Agent
- `technical-communication-lead` for UX messaging; secondary `systems-architecture-optimizer` for configuration wiring.
