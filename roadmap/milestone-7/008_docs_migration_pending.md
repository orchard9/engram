# 008: Documentation & Migration Guide — _pending_

## Goal
Update public and internal documentation to explain memory spaces, migration steps, and operational workflows for multi-tenant deployments. Deliverables must equip operators, developers, and client teams with clear upgrade guidance and troubleshooting.

## Deliverables
- README + usage updates introducing memory spaces, default behaviour, CLI flags, and example workflows.
- API reference updates (`docs/api/*`, `engram-cli/src/docs.rs`, auto-generated utoipa docs) covering headers, query params, SSE metadata, and gRPC field additions.
- Comprehensive migration guide (`docs/operations/memory_space_migration.md` or similar) detailing server upgrades, data migration (if any), client rollout sequencing, monitoring changes, and rollback steps.
- Release notes / changelog entries summarizing milestone impact (tie into `docs/changelog.md` and `engram-proto` notes).
- Troubleshooting appendix covering common errors (missing header, unknown space, recovery failure) with remediation.
- Update to diagnostics documentation referencing per-space metrics (Task 006).

## Implementation Plan

1. **Inventory Updates**
   - Read outputs from Tasks 003–007 to ensure docs align with implementation.
   - Compile list of files needing updates: `README.md`, `usage.md`, `docs/getting-started.md`, `docs/api/*.md`, `engram-cli/src/docs.rs`, `docs/operations/operations.md`, `docs/metrics-schema-changelog.md`, `docs/changelog.md`.

2. **README & Usage**
   - Add overview section “Memory Spaces” with explanation, default behaviour, CLI examples (`engram memory create --space alpha`).
   - Update quickstart commands to include optional `--space` flag and call out default fallback.

3. **API Reference**
   - Document new header `X-Engram-Memory-Space`, query parameter, SSE payload fields, and gRPC request fields.
   - Update OpenAPI (utoipa) docs by editing `engram-cli/src/api.rs` doc comments and ensure `/docs` UI reflects changes.
   - Provide sample curl commands showing header usage.

4. **Migration Guide**
   - Create new doc `docs/operations/memory_space_migration.md` (or update existing operations guide) covering:
     - Pre-requisites (upgrade to Milestone 7 build, backup data).
     - Config changes (`default_memory_space`, `bootstrap_spaces`).
     - Step-by-step server upgrade (stop old instance, run upgrade, verify recovery per space).
     - Client rollout steps (set header/env/flags, gRPC field update).
     - Monitoring updates (per-space metrics, diagnostics script) referencing Task 006.
     - Rollback plan (revert to single space) and validation checklist.

5. **Release Notes / Changelog**
   - Add entry to `docs/changelog.md` summarizing memory space release, referencing milestone tasks.
   - Update `engram-proto/CHANGELOG.md` describing gRPC field addition and compatibility notes.

6. **Troubleshooting & Diagnostics**
   - Update `engram-cli/src/docs.rs` troubleshooting section with new errors (e.g., “Missing memory space header” guidance).
   - Document use of diagnostics script to inspect per-space metrics, linking to Task 006 updates.

7. **Cross-Linking**
   - Ensure docs cross-reference: migration guide links to API reference, metrics doc, CLI docs; README links to migration guide.

8. **Validation**
   - Run markdown lint/link checker (existing script or `mdbook` validator).
   - Submit docs to `technical-communication-lead` + `documentation-validator` for review.

## Integration Points
- `README.md`, `usage.md`, `docs/` tree, `engram-cli/src/docs.rs`, `docs/changelog.md`, `docs/metrics-schema-changelog.md`, `docs/operations/`.
- `engram-proto/CHANGELOG.md` for gRPC note.

## Acceptance Criteria
1. Documentation reflects new APIs, CLI options, and defaults; code samples compile/run.
2. Migration guide covers upgrade flow, client changes, monitoring, rollback, and verification checklist.
3. Troubleshooting section lists common multi-space issues with remediation steps.
4. Diagnostics and metrics docs reference per-space enhancements without inconsistencies.
5. Link checker/markdown lint passes; reviewer sign-off recorded.

## Testing Strategy
- Run documentation validator agent.
- Execute link checker/markdown lint (`npm run lint-docs` or existing script).
- Spot-test sample commands against local dev environment to confirm accuracy.

## Review Agent
- `technical-communication-lead` (primary) with `documentation-validator` follow-up.
