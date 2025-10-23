# 004: API, CLI, and Config Surface — _pending_

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
1. HTTP routes that hit storage enforce space selection; missing IDs when multiple spaces exist return 400 with remediation steps.
2. CLI defaults to configured space, supports env/flag overrides, and prints active space; tests confirm precedence order.
3. Server startup logs default space and warns when falling back for legacy requests.
4. SSE endpoints emit events scoped to the requested space and include `memory_space_id` field.
5. Single-space deployments continue to function without flags, emitting a migration warning.

## Testing Strategy
- REST integration tests (`tests/api/test_memory_spaces.rs`) covering header/query/body extraction, fallback, and error cases.
- CLI smoke tests via `assert_cmd` verifying `--space`, env var fallback, and default behaviour.
- Config round-trip tests for new persistence settings.
- SSE subscription test using `reqwest_eventsource` (or mocked stream) ensuring filtering and metadata.
- Backwards compatibility test starting server with single space and running old CLI command without `--space`.

## Review Agent
- `technical-communication-lead` for UX messaging; secondary `systems-architecture-optimizer` for configuration wiring.
