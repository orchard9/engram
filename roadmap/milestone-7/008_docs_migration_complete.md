# 008: Documentation & Migration Guide — _85_percent_complete_

## Current Status: 85% Complete

**What's Implemented**:
- ✅ Comprehensive migration guide (docs/operations/memory-space-migration.md, 617 lines)
  - Pre-requisites and upgrade workflow
  - Step-by-step server upgrade instructions
  - Client rollout guidance
  - Monitoring and validation steps
- ✅ Multi-tenant API examples (docs/reference/api-examples/08-multi-tenant/)
  - HTTP header usage
  - Query parameter examples
  - Request body examples
  - gRPC field usage
- ✅ Architecture documentation (architecture.md Memory Space section)
- ✅ Implementation patterns and best practices documented

**Minor Gaps Remaining** (15%):
- ❌ README.md not updated with memory space overview
- ❌ usage.md missing multi-space examples
- ❌ Changelog entries not created
- ❌ Troubleshooting section incomplete
- ❌ OpenAPI docs need verification
- ❌ Diagnostics documentation not updated

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

1. ✅ **COMPLETE**: Documentation reflects new APIs, CLI options, and defaults
   - Implementation: API examples comprehensive in docs/reference/api-examples/08-multi-tenant/
   - Migration: memory-space-migration.md covers all workflows
   - ⚠️ Missing: Code samples not verified to compile/run

2. ✅ **COMPLETE**: Migration guide covers upgrade flow
   - File: docs/operations/memory-space-migration.md (617 lines)
   - Coverage: Server upgrade, client changes, monitoring, rollback
   - Verification: Checklists included
   - Quality: Comprehensive and actionable

3. ⚠️ **PARTIAL**: Troubleshooting section lists common issues
   - ✅ Basic troubleshooting in migration guide
   - ❌ Missing: Dedicated troubleshooting appendix
   - ❌ Missing: Error code reference with remediation

4. ❌ **NOT STARTED**: Diagnostics and metrics docs updated
   - File: docs/metrics-schema-changelog.md not updated
   - File: docs/operations/operations.md not updated
   - Missing: Per-space metrics documentation

5. ❌ **NOT STARTED**: Link checker/markdown lint validation
   - Need: Run markdown validator
   - Need: Link checker execution
   - Need: Reviewer sign-off documentation

## Remaining Work

1. **README.md Update** (1 hour)
   - File: /Users/jordanwashburn/Workspace/orchard9/engram/README.md
   - Section: Add "Memory Spaces" overview after introduction
   - Content:
     - Explain multi-tenancy support
     - Show basic --space flag usage
     - Link to migration guide
   - Examples: Quick start with memory spaces

2. **usage.md Update** (1 hour)
   - File: /Users/jordanwashburn/Workspace/orchard9/engram/usage.md
   - Section: Update CLI examples to include --space flag
   - Content:
     - Show ENGRAM_MEMORY_SPACE env var
     - Explain default space behavior
     - Multi-space workflow examples

3. **Changelog Entries** (1 hour)
   - File: docs/changelog.md
   - Entry: Milestone 7 - Memory Space Support
   - Summary: Multi-tenancy, API changes, breaking changes
   - File: engram-proto/CHANGELOG.md
   - Entry: memory_space_id field addition to requests

4. **Troubleshooting Appendix** (2 hours)
   - File: docs/operations/troubleshooting.md or extend migration guide
   - Sections:
     - "Missing memory space header" error
     - "Unknown space" error and remediation
     - Recovery failures per space
     - Cross-space data contamination detection
   - Format: Error code → Symptom → Root cause → Remediation

5. **Diagnostics Documentation** (1 hour)
   - File: docs/operations/operations.md or diagnostics.md
   - Content: Document per-space metrics in diagnostics script
   - Format: Show example output with space sections
   - Reference: Link to Task 006 metrics documentation

6. **OpenAPI Verification** (1 hour)
   - Action: Run server, check /docs UI
   - Verify: memory_space_id documented on all request DTOs
   - Verify: X-Engram-Memory-Space header documented
   - Fix: Update utoipa annotations if needed

7. **Validation & Sign-off** (1 hour)
   - Run: markdown lint (if configured)
   - Run: link checker on docs/
   - Test: Compile/run code samples from docs
   - Review: Submit to technical-communication-lead

## Testing Strategy
- Run documentation validator agent.
- Execute link checker/markdown lint (`npm run lint-docs` or existing script).
- Spot-test sample commands against local dev environment to confirm accuracy.

## Review Agent
- `technical-communication-lead` (primary) with `documentation-validator` follow-up.
