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
1. Protobuf schema compiles; regeneration yields no manual diff aside from expected field additions.
2. gRPC requests without `memory_space_id` resolve to default space with logged deprecation warning; tests assert warning emitted.
3. Requests specifying unknown space return `Status::not_found` with actionable error text.
4. Streaming APIs deliver events only for the requested space and independent backpressure works under concurrent load.
5. CHANGELOG/README entries describe the change and migration path.

## Testing Strategy
- `engram-proto`: run `cargo test` and optionally `buf lint` (if configured) after regeneration.
- Integration tests using tonic client verifying remember/recall with explicit space, fallback behaviour, and error cases.
- Streaming integration test hooking multiple spaces simultaneously and confirming isolation.
- Compatibility test using old client fixture (without field) to confirm successful default routing and logged warning.

## Review Agent
- `technical-communication-lead` for API wording; `verification-testing-lead` for compatibility and regression coverage.
