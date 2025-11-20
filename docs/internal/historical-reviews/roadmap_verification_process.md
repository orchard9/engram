# Roadmap Verification & Update Process

This document captures the workflow we’re following to reconcile the roadmap with the actual codebase. The goal is to ensure every milestone/task file reflects reality, identify remaining work, and keep the project’s planning artifacts trustworthy.

## 1. Choose Scope
1. Start at the highest-level roadmap index (`milestones.md`) to note which milestones are claimed “complete” vs. “in progress”.
2. Iterate milestones sequentially (0 → 1 → 2 …) unless a user requests a specific target.

## 2. Inspect Task Files
For each milestone directory:
1. List all `NNN_*` task files.
2. Read each markdown file for a `Status:` line and acceptance criteria.
3. Look for inconsistencies like “Status: Pending” or unchecked acceptance boxes even when the filename ends in `_complete`.

## 3. Verify Against Code
Whenever a task claims completion, validate it in the repository:
- Identify referenced files/paths from the task doc.
- Open those files to confirm the implementation exists and matches the description (e.g., router metrics, memory backends, completion modules).
- Search for related tests (`cargo test -p <crate> task_specific_test`).

If the code matches the spec, update the task file to “COMPLETE ✅” and capture references/notes. If the code is missing (or only partially implemented), leave the status as pending/partial and jot down the missing work.

## 4. Record Remaining Work
For milestones with multiple outstanding items, create or update a `0030_*` (or similar) tracking doc summarizing what’s left. Include:
- Open tasks + gaps (e.g., missing benchmarks, configs, tests).
- Concrete files/paths that need updates.
- Exit criteria to call the milestone done.

## 5. Update Artifacts
1. Edit task docs with accurate statuses, implementation notes, and references to the actual code/tests.
2. Where we’ve confirmed completion, mention the specific files/sections (e.g., `engram-core/src/completion/mod.rs:1-120`).
3. Document new capabilities (e.g., router health metrics) in the corresponding milestone file so future readers know what shipped.

## 6. Run Targeted Tests
For code touched during validation (e.g., router), run the relevant unit/integration tests and note the commands in the task doc. Typical pattern:
```
cargo test -p engram-cli router::tests::… -- --exact
```
Only run the tests needed to verify the change; there’s no need for a full workspace build unless we modify sweeping infrastructure.

## 7. Communicate Next Steps
After each milestone review:
- Summarize what’s now confirmed complete vs. pending.
- Suggest the next logical milestone or follow-up tasks.
- Highlight any new docs/files we added (e.g., `roadmap/milestone-2/0030_remaining_work_pending.md`).

Follow this loop for subsequent milestones until the roadmap accurately mirrors the codebase.
