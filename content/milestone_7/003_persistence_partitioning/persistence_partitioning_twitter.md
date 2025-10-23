1/ Engram’s memory spaces only work if persistence is isolated. Right now every WAL, tier, and compaction worker is global—two agents share the same log (`engram-core/src/store.rs`).

2/ Milestone 7 fixes that: each `MemorySpaceId` gets its own directory tree `<data_root>/<space>/{wal,hot,warm,cold}` with sanitized names (`engram-core/src/storage/wal.rs`).

3/ The registry provisions handles lazily. Creating a space spins up WAL writer, tier queues, and metrics labels; idle spaces don’t burn threads (`roadmap/milestone-7/003_persistence_partitioning_pending.md`).

4/ WAL recovery enumerates spaces independently. If one tenant corrupted its log, others keep serving traffic while we rebuild the failed space (`engram-core/src/storage/wal.rs`).

5/ Metrics go per-space: WAL lag, compaction backlog, eviction pressure. Operators finally see which agent is noisy before throttling (`engram-core/src/metrics`).

6/ CLI tooling lands too—`engram space list/create` plus diagnostics that read per-space health straight from disk (`engram-cli/src/docs.rs`).

7/ With persistence partitioned, API/gRPC layers can safely demand `memory_space_id`, and validation suites can prove zero cross-space leaks.

8/ Multi-tenant Engram becomes real storage infrastructure, not just a doc promise. Memory spaces stay autonomous from disk to activation.
