# Backup and Disaster Recovery - Twitter Thread

**Tweet 1/8**

Most teams discover their backups don't work when they desperately need them to work.

The script ran for 6 months. Monitoring says green. Then disaster strikes. Backup is corrupted.

Here's how to build bulletproof backup for graph databases with RTO <30min, RPO <5min.

Thread:

---

**Tweet 2/8**

The multi-tier consistency problem:

Fast tier backup at T=0
Warm tier backup at T=5min
Cold tier backup at T=10min

Result: Inconsistent state. Memory in fast but not warm tier = broken references on restore.

Solution: Coordinated snapshot at same timestamp across all tiers.

---

**Tweet 3/8**

Full vs incremental backup strategy:

Full: Nightly at 2 AM, 2-3 min quiesce, 100GB compressed
Incremental: Every 5 minutes, lock-free append, 10GB/day

Why both? Recovery scenarios:
- Last night: 10 min
- 2 hours ago: 15 min
- 30 min ago: 20 min
- 2 min ago: 25 min

All under 30 min RTO.

---

**Tweet 4/8**

Point-in-time recovery is your time machine.

Operator deletes critical node at 14:32. Detected at 14:45.

Restore to 14:30 (before deletion) using base backup + operation log replay.

Every mutation logged. Every state recoverable. No data loss.

This saved us more times than I'll admit.

---

**Tweet 5/8**

The 3-2-1 backup rule:

3 copies: Production + local backup + cloud
2 media: SSD + S3
1 offsite: Cross-region replication

Retention:
- 24 hourly
- 7 daily
- 4 weekly
- 12 monthly

Protects against disk failure, datacenter loss, and ransomware.

---

**Tweet 6/8**

Trust but verify: Automated validation after EVERY backup.

1. Checksum verification
2. Test restore to tmp directory
3. Graph integrity check
4. Performance benchmark
5. Alert if any step fails

Untested backups are Schrodinger's backups. Both working and broken until you need them.

---

**Tweet 7/8**

Disaster recovery runbook with actual commands:

Complete data loss scenario:
1. Download backup from S3: 2 min
2. Restore full backup: 10 min
3. Apply incrementals: 5 min
4. Start server: 30 sec
5. Warmup: 2 min
6. Smoke test: 1 min

Total: 26 minutes. Measured, not guessed.

---

**Tweet 8/8**

Monthly chaos drills:

Week 1: Deliberate corruption
Week 2: Disk full scenario
Week 3: Network partition
Week 4: Full datacenter loss

Document failures. Fix failures. Update runbook with actual times.

The worst time to learn your backup doesn't work is when you need it to work.
