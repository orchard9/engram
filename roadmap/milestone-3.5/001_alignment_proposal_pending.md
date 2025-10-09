# Milestone 3.5: Runtime & Roadmap Alignment Proposal

## Status: PENDING

## Objective
Bring the shipped runtime surface back in sync with the roadmap by aligning client/Server APIs, activating the gRPC layer, grounding observability streams in real data, and reconciling roadmap bookkeeping. This milestone focuses on corrective integration, not net-new features.

## Alignment Workstreams

### 1. Unify CLI and HTTP API contracts
- **Symptom**: `engram memory ...` commands call `/api/v1/memories`, `/api/v1/memories/search`, and `/api/v1/memories/{id}` (`engram-cli/src/cli/memory.rs:67`, `:103`, `:140`), but the Axum router only exposes `/api/v1/memories/remember|recall|recognize` (`engram-cli/src/api.rs:1836`-`1838`).
- **Resolution options** (pick one, document decision):
  1. Update the CLI to target the existing verbs (`remember`, `recall`) and adjust payloads to match `remember_memory` expectations (`engram-cli/src/api.rs:421`-`500`).
  2. Or, add pragmatic REST endpoints (`GET /api/v1/memories`, `POST /api/v1/memories`) that mirror current CLI semantics and forward into the existing handlers.
- **Implementation specifics** (if choosing option 1):
  - Change the POST URL in `create_memory` to `/api/v1/memories/remember` and send the request body expected by `RememberMemoryRequest` (`engram-cli/src/cli/memory.rs:68`-`94`, `engram-cli/src/api.rs:205`-`268`).
  - Update `get_memory`, `search_memories`, `list_memories`, and `delete_memory` to either call the new API verbs or remove/unblock corresponding server routes; add handlers under `create_api_routes` if we keep those commands (`engram-cli/src/api.rs:1834`-`1848`).
  - Add regression coverage in `engram-cli/tests/` that exercises the happy path via `cargo test -p engram-cli` once the server is running.

### 2. Start and integrate the gRPC service
- **Symptom**: `MemoryService::serve` is fully implemented (`engram-cli/src/grpc.rs:35`-`58`), but `start_server` only prints the intended port and never spawns tonic (`engram-cli/src/main.rs:102`-`207`). Recall currently returns placeholder data (`engram-cli/src/grpc.rs:164`-`182`).
- **Implementation specifics**:
  - Spawn `MemoryService::new(Arc::clone(&memory_store))` inside `start_server` and run `serve` on the `actual_grpc_port` in a `tokio::spawn` (or `tokio::select!`) alongside the Axum server (`engram-cli/src/main.rs:156`-`207`). Ensure the task is awaited on shutdown.
  - Replace the recall stub with a call into `MemoryStore::recall`, mapping results into `RecallResponse` (`engram-cli/src/grpc.rs:164`-`182`); reuse the conversion logic already present in the HTTP handler (`engram-cli/src/api.rs:703`-`745`).
  - Extend smoke tests or add a `tonic` client test under `engram-cli/tests/` validating both Remember/Recall round-trips.

### 3. Ground streaming/monitoring endpoints in real events
- **Symptom**: `/api/v1/stream/activities` and `/api/v1/stream/memories` emit random data from background tasks (`engram-cli/src/api.rs:1234`-`1288`, `:1304`-`1344`), so consumers cannot observe actual activation or store events.
- **Implementation specifics**:
  - Introduce an event broadcaster in `MemoryStore` (e.g., `tokio::sync::broadcast::Sender`) and publish from `store` / `recall` (`engram-core/src/store.rs:820`-`920`, `:1102`-`1194`).
  - Replace the random loops with receivers that forward real events to SSE clients, carrying confidence/activation data that matches the roadmap semantics (`engram-cli/src/api.rs:1234`-`1345`).
  - Document backpressure behaviour and test with integration coverage to ensure clients disconnect cleanly.

### 4. Reconcile roadmap status with implementation
- **Symptom**: Multiple roadmap files marked `_complete` still declare `## Status: PENDING`, e.g. the gRPC service record (`roadmap/milestone-0/016_grpc_service_complete.md:1`-`29`).
- **Implementation specifics**:
  - Review each `_complete` file, adjust status blocks, and ensure acceptance criteria accurately reflect the shipping functionality once Workstreams 1–3 land.
  - If deliverables remain incomplete, rename files back to `_pending` per `CLAUDE.md` process and capture blockers inline.

### 5. Decide on `engram-storage` usage
- **Symptom**: The active server path never touches `engram-storage`; only `main_backup.rs` wires it in (`engram-cli/src/main_backup.rs:14`-`38`).
- **Implementation specifics**:
  - Either integrate the tiered storage abstractions into `start_server` (behind feature flags) or document that `engram-storage` is deprecated and update the roadmap/milestones to avoid confusion.
  - If integrating, audit `MemoryStore::with_persistence` (`engram-core/src/store.rs:381`-`414`) to ensure the crates remain coherent.

## Validation Checklist
- `cargo fmt --all`, `cargo clippy --workspace --all-targets --all-features`, and `cargo test --workspace` succeed.
- CLI smoke tests confirm parity between commands and HTTP routes.
- gRPC Remember/Recall integration test passes against a running server.
- SSE clients receive deterministic data tied to real memory operations.
- Roadmap statuses match reality and document any remaining gaps.

## Exit Criteria
Milestone 3.5 is complete when the CLI, HTTP API, and gRPC stack operate over the same memory pipeline, observability reflects live state, and the roadmap accurately describes what ships today. Any scope left for future work must spawn follow-up tasks under Milestone 4 or later.

