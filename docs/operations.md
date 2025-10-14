# Engram Operations Runbook

This runbook provides step-by-step guidance for operating an Engram deployment. It follows the Context → Action → Verification format so on-call engineers can respond quickly under pressure.

## 1. Prerequisites
- Engram binary (`cargo build -p engram-cli --release`) installed on each node.
- Shared configuration directory (defaults to `$HOME/.config/engram`).
- Data directory writable by the Engram process (defaults to `$ENGRAM_DATA_DIR` or `./data`).
- `curl`, `tar`, and `python3` available for health checks and backups.
- `scripts/backup_engram.sh` and `scripts/check_engram_health.sh` in `$PATH`.

## 2. Start Engram
- **Context**: Bring a node online.
- **Action**: `engram start --single-node --port 0` (port `0` lets the OS choose a free port).
- **Verification**: Run `scripts/check_engram_health.sh http://127.0.0.1:<port>`; expect `status=healthy`.

### Progress Output Expectations
The CLI should emit staged messages:
1. `Starting Engram…`
2. `Binding interfaces…`
3. `Loading storage tiers…`
4. `Ready! health=http://127.0.0.1:<port>/api/v1/system/health`

If any stage stalls for >30s, capture logs and proceed to the troubleshooting section.

## 3. Stop Engram
- **Context**: Gracefully stop a node for maintenance.
- **Action**: `engram stop --pid-file <pid_path>`.
- **Verification**: `pgrep engram` should return no processes; health check returns connection refused.

To force terminate (only if graceful stop fails): `engram stop --pid-file <pid_path> --force`.

## 4. Status Inspection
- **Context**: Verify cluster state without changing it.
- **Action**: `engram status --format json`.
- **Verification**: Ensure `state` is either `Running` or `Degraded`; investigate if `state != Running`.

## 5. Backup & Restore
### Backup
- **Context**: Capture consistent snapshot for disaster recovery.
- **Action**: `scripts/backup_engram.sh <data_dir> <backup_root>`.
- **Verification**: Script prints `Backup stored at …`; verify tarball contents via `tar -tf`.

### Restore
1. Stop Engram (`engram stop …`).
2. Extract latest backup: `tar -xf <backup>.tar.gz -C <data_dir_parent>`.
3. Start Engram; confirm health.
4. Run an application-level smoke test (e.g., store & recall a memory).

## 6. Monitoring & Alerting
- Use `scripts/check_engram_health.sh` to poll `/api/v1/system/health`.
- Consume metrics via the internal streaming endpoint or structured logs when the `monitoring` feature is enabled.
- Establish alerts for:
  - Health status != `healthy` for >1m.
  - WAL lag > 1s.
  - Tier migration failures (`StorageError::MigrationFailed`).

## 7. Capacity & Tier Management
- Monitor hot tier usage via `engram status --format json` (look at `hot_tier.utilization`).
- When utilization >80%, schedule migrations or add capacity.
- For manual migration kick-off call the admin API (future work) or scale out nodes.

## 8. Incident Response
| Scenario | Symptoms | Immediate Actions | Verification |
|----------|----------|-------------------|-------------|
| Health endpoint degraded | `/api/v1/system/health` returns `degraded` | Check logs for storage or WAL errors; run `scripts/check_engram_health.sh` repeatedly | Status recovers to `healthy` |
| WAL replay failure on restart | Startup logs show `Failed to deserialize WAL` | Move corrupt WAL files aside (`mv data/wal/*.log data/wal/corrupt/`), restart, initiate restore | Health endpoint healthy, data recovered from backup |
| Tier migration backlog | Metrics show `migration_queue_depth` increasing | Temporarily raise warm tier capacity or trigger manual drain | Queue returns to baseline |

Escalate to the architecture working group if more than one node experiences the same failure within 30 minutes.

## 9. Troubleshooting Quick Reference
- `engram status --verbose` – Detailed node state.
- `RUST_LOG=debug engram start …` – Enables detailed logging when reproducing an issue.
- `journalctl -u engram` – If Engram runs under systemd.
- `lsof -i :<port>` – Diagnose port conflicts if startup fails.

## 10. Change Management Checklist
1. Announce change in #engram-ops with start/end time.
2. Verify backups taken within last 24h.
3. Execute change (deployment, configuration update, etc.).
4. Run post-change smoke test.
5. Update status in #engram-ops and log in change tracker.

## Appendix A – Command Reference
- `engram start [--single-node] [--port <port>]` – Start node.
- `engram stop --pid-file <path> [--force]` – Stop node.
- `engram status [--format json]` – Inspect state.
- `engram config list` – View configuration values.
- `engram config set <key> <value>` – Update configuration.

Keep this document alongside the roadmap. Updates should be reviewed whenever CLI or storage behavior changes.
