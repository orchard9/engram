# 005: gRPC & Proto Multi-Tenant Support — _complete_

## Completion Status: 60% (Partial)

**Completed**: Proto schema evolution with `memory_space_id` field on 10 request messages, server-side routing for `remember()`/`recall()` endpoints, test updates, code generation.

**Gaps**: 9 remaining RPC handlers need space routing, streaming API isolation not implemented, deprecation warnings missing, comprehensive testing incomplete, documentation not written.

**See**: `005_COMPLETION_REVIEW.md` for detailed gap analysis and recommended follow-up tasks (005b-e).

## Goal
Evolve gRPC contracts and server implementation so clients declare memory spaces explicitly while preserving backwards compatibility for existing single-space deployments. The proto evolution must include migration guidance, code generation updates, and validation across bindings.

## Deliverables
- Protobuf updates adding `memory_space_id` (string) to relevant request messages (`RememberRequest`, `RecallRequest`, streaming requests, config introspection) with descriptive comments and defaulting guidance.
- Version note in proto package options (e.g., `option (grpc.gateway.protoc_gen_openapiv2.options.openapiv2_tag)`) describing the change for auto-generated docs.
- Server-side routing in `engram-cli/src/grpc.rs` that resolves the space, acquires the correct `MemoryStore`, and surfaces clear errors for unknown spaces.
- Backward compatibility layer: if field missing, derive default space and emit warning (server log + response metadata optional).
- Regenerated Rust code under `engram-proto/src/generated.rs` plus updated build script if needed (`engram-proto/build.rs`).
- Update bindings (`bindings/`) or regeneration instructions as appropriate.
- Migration checklist for Task 008 (include summary in code comments).

## Implementation Plan

1. **Proto Definition Updates**
   - Edit `proto/engram_service.proto` (and related .proto files) to add `string memory_space_id = <next_field_number>;` to:
     - `RememberRequest`
     - `EpisodeRequest`/`RecallRequest`
     - Streaming request messages (`StreamRequest`, `MemoryFlowRequest`, etc.)
     - Any request that triggers persistence or monitoring.
   - Add comment explaining default behaviour and requirement in multi-space deployments.
   - Bump the proto package version comment and update `FEATURES.md`/`engram-proto/README` if required.

2. **Regenerate Code**
   - Run `cargo run -p engram-proto --features build` or existing regeneration script.
   - Verify generated files under `engram-proto/src` compile.
   - If other languages are supported, update instructions (Python/TypeScript via `bindings/`).

3. **Server Routing Changes**
   - In `engram-cli/src/grpc.rs`:
     - Update `remember`, `recall`, streaming handlers to read `memory_space_id` field.
     - When field absent, call default resolver (shared with Task 004) to maintain parity.
     - On unknown space, return `Status::not_found` with message describing how to create/choose spaces.
     - Pass `MemorySpaceId` into registry to obtain store handle.
   - Propagate space metadata into tracing spans and streaming contexts.

4. **Client Compatibility Layer**
   - Expose helper in gRPC client wrappers (if any) to set the header/field automatically.
   - Document new environment variable or CLI configuration to populate the field for CLI-driven gRPC calls.

5. **Streaming & Backpressure**
   - Ensure streaming methods instantiate per-space consumers so that subscription to one space does not leak events.
   - Add tests verifying simultaneous streams from different spaces operate independently.

6. **Migration Documentation Hooks**
   - Add TODO comments referencing Task 008 for doc updates.
   - Update `engram-proto/CHANGELOG.md` (if present) with entry describing new field and default behaviour.

## Integration Points
- `proto/engram_service.proto` and related files.
- Generated Rust code in `engram-proto/src`.
- gRPC server implementation `engram-cli/src/grpc.rs`.
- CLI/gRPC client wrappers under `engram-cli/src/cli` or `bindings/`.

## Acceptance Criteria

1. ✅ **COMPLETE**: Protobuf schema compiles and regenerates cleanly
   - Implementation: memory_space_id field added to 10 request messages
   - Files: Proto definitions updated, Rust code regenerated
   - Status: No compilation errors

2. ⚠️ **PARTIAL**: gRPC requests without memory_space_id resolve to default
   - ✅ Implemented: remember() and recall() handlers
   - ❌ Missing: 9 remaining RPC handlers need space routing
   - ❌ Missing: Deprecation warning logging

3. ❌ **NOT IMPLEMENTED**: Unknown space error handling
   - Missing: Status::not_found with actionable error text
   - Need: Validation in all handlers

4. ❌ **NOT IMPLEMENTED**: Streaming APIs deliver events per space
   - Missing: Stream isolation implementation
   - Missing: Concurrent backpressure testing
   - Missing: Per-space event filtering

5. ❌ **NOT STARTED**: CHANGELOG/README entries
   - Missing: Proto changelog entry
   - Missing: Migration path documentation

## Remaining Work (40%)

1. **Complete Remaining gRPC Handlers** (4 hours)
   - File: engram-cli/src/grpc.rs
   - Handlers needing update (9 total):
     - store_memory
     - search_memories
     - remove_memory
     - consolidate
     - dream
     - get_activation_stats
     - stream_events
     - stream_activations
     - any other RPCs
   - Pattern: Extract space from request field → get registry handle → verify_space()

2. **Deprecation Warning System** (2 hours)
   - File: engram-cli/src/grpc.rs
   - Action: When memory_space_id field is None, emit warn! log
   - Message: "gRPC request missing memory_space_id, using default space"
   - Optional: Add warning to response metadata

3. **Unknown Space Error Handling** (1 hour)
   - File: engram-cli/src/grpc.rs
   - Action: Wrap registry.create_or_get() errors
   - Return: Status::not_found("Space '<id>' not found. Create it first or use default space")
   - Test: Verify error message in integration tests

4. **Streaming API Isolation** (4 hours)
   - File: engram-cli/src/grpc.rs
   - Implementation:
     - Extract memory_space_id from stream request
     - Filter events by space before sending
     - Maintain per-space backpressure state
   - Test: Concurrent streams from different spaces don't interfere

5. **gRPC Integration Tests** (3 hours)
   - File: Create engram-cli/tests/grpc_multi_space.rs
   - Scenarios:
     - remember/recall with space field
     - Default fallback without field
     - Unknown space returns not_found
     - Stream isolation validation
   - Framework: tonic test client

6. **Proto Documentation** (1 hour)
   - File: engram-proto/CHANGELOG.md
   - Entry: Document memory_space_id field addition
   - Content: Migration guidance, backward compatibility
   - File: Update proto comments with usage examples

## Testing Strategy
- `engram-proto`: run `cargo test` and optionally `buf lint` (if configured) after regeneration.
- Integration tests using tonic client verifying remember/recall with explicit space, fallback behaviour, and error cases.
- Streaming integration test hooking multiple spaces simultaneously and confirming isolation.
- Compatibility test using old client fixture (without field) to confirm successful default routing and logged warning.

## Review Agent
- `technical-communication-lead` for API wording; `verification-testing-lead` for compatibility and regression coverage.
