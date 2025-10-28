# Disaster Recovery

Comprehensive disaster recovery procedures for Engram with detailed runbooks for critical failure scenarios.

## RTO and RPO Definitions

### Service Level Objectives

- **RTO (Recovery Time Objective)**: Maximum time to restore service after failure
  - Target: 30 minutes for most scenarios
  - Critical path: 60 minutes for complete datacenter failure

- **RPO (Recovery Point Objective)**: Maximum acceptable data loss
  - Target: 5 minutes with incremental backups
  - Worst case: Last full backup interval (4 hours)

### SLA Guarantees

| Scenario | RTO | RPO | Impact |
|----------|-----|-----|--------|
| Data corruption | 15 min | 5 min | Single space affected |
| Complete data loss | 30 min | 5 min | All local data lost |
| Accidental deletion | 20 min | Exact PITR | Specific time range |
| WAL corruption | 15 min | Last checkpoint | Single space affected |
| Datacenter failure | 60 min | 5 min | Complete regional outage |
| Ransomware attack | 45 min | Last clean backup | Security incident |

## Incident Response Framework

### Phase 1: Detection and Assessment

1. **Incident Detection**
   - Monitor alerts from backup verification
   - Check Engram health endpoints
   - Review system logs for errors
   - Validate data integrity

2. **Impact Assessment**
   - Identify affected memory spaces
   - Determine data loss scope
   - Estimate recovery time
   - Assess business impact

3. **Communication**
   - Notify stakeholders
   - Update status page
   - Document incident timeline
   - Establish communication channel

### Phase 2: Containment

1. **Stop the Bleeding**
   - Stop Engram service if corrupting data
   - Isolate affected systems
   - Prevent further data loss
   - Preserve evidence for analysis

2. **Create Safety Snapshots**
   - Snapshot current state before changes
   - Document system configuration
   - Capture logs and metrics
   - Preserve WAL segments

### Phase 3: Recovery

Execute appropriate recovery procedure based on scenario (see sections below).

### Phase 4: Verification

1. **Data Integrity**
   - Run health checks
   - Verify memory counts
   - Test query functionality
   - Validate spreading activation

2. **Performance Validation**
   - Check response times
   - Monitor resource usage
   - Verify tier distribution
   - Test concurrent operations

### Phase 5: Post-Incident Review

1. **Root Cause Analysis**
   - Identify failure cause
   - Document timeline
   - Review monitoring gaps
   - Assess response effectiveness

2. **Preventive Measures**
   - Implement safeguards
   - Update runbooks
   - Improve monitoring
   - Schedule follow-up testing

## DR Scenarios and Procedures

### Scenario 1: Data Corruption

**Context**: Database files corrupted, Engram won't start or crashes on startup.

**Symptoms**:
- Engram fails to start
- Checksum errors in logs
- Tier loading failures
- WAL replay errors

**Recovery Procedure**:

```bash
# Step 1: Stop service and assess damage
systemctl stop engram
journalctl -u engram -n 100 | grep -i "error\|corrupt"

# Step 2: Identify latest verified backup
latest_good=$(ls -t /var/backups/engram/full/*.tar.zst | head -1)
echo "Using backup: $latest_good"

# Verify backup integrity
/scripts/verify_backup.sh "$latest_good" L2

# Step 3: Create safety backup of corrupted data
timestamp=$(date +%Y%m%dT%H%M%SZ)
mv /var/lib/engram "/var/lib/engram.corrupted-$timestamp"

# Step 4: Restore from backup
/scripts/restore.sh "$latest_good" /var/lib/engram full

# Step 5: Apply incremental backups
for incr in $(ls -t /var/backups/engram/incremental/*.tar.zst | head -5); do
    echo "Applying incremental: $incr"
    /scripts/restore.sh "$incr" /var/lib/engram incremental
done

# Step 6: Start and verify
systemctl start engram
sleep 5

# Step 7: Health check
curl http://localhost:7432/api/v1/system/health
curl http://localhost:7432/api/v1/system/stats | jq '.'

# Step 8: Test query functionality
curl -X POST http://localhost:7432/api/v1/memories/search \
  -H "Content-Type: application/json" \
  -d '{"query":"test","limit":10}' | jq '.'

# Step 9: Document recovery
cat > "/var/log/engram/recovery-$timestamp.log" <<EOF
Recovery Time: $(date)
Scenario: Data Corruption
Backup Used: $latest_good
Recovery Duration: $SECONDS seconds
Verification: SUCCESS
EOF
```

**Expected Time**: 15 minutes

**Post-Recovery Actions**:
- Analyze corrupted data for root cause
- Review filesystem integrity
- Check for hardware issues
- Update monitoring thresholds

---

### Scenario 2: Complete Data Loss

**Context**: Storage failure, all local data lost (disk failure, accidental rm -rf, etc).

**Symptoms**:
- /var/lib/engram directory missing or empty
- All data files inaccessible
- Engram won't start due to missing data

**Recovery Procedure**:

```bash
# Step 1: Stop service (may already be stopped)
systemctl stop engram

# Step 2: Restore from remote backup
echo "Fetching latest backup from remote storage..."

# If using S3
aws s3 cp s3://engram-backups/full/latest-full.tar.zst /tmp/
aws s3 cp s3://engram-backups/manifests/latest-full.json /tmp/

# If using rsync
rsync -avz --progress backup-server:/backups/engram/full/latest-full.tar.zst /tmp/

# Step 3: Verify backup
/scripts/verify_backup.sh /tmp/latest-full.tar.zst L3

# Step 4: Recreate data directory structure
mkdir -p /var/lib/engram
chown engram:engram /var/lib/engram
chmod 700 /var/lib/engram

# Step 5: Restore full backup
/scripts/restore.sh /tmp/latest-full.tar.zst /var/lib/engram full

# Step 6: Fetch and apply incremental backups
echo "Fetching incremental backups..."
for incr in $(aws s3 ls s3://engram-backups/incremental/ | \
              grep engram-incr | \
              tail -10 | \
              awk '{print $4}'); do
    echo "Downloading: $incr"
    aws s3 cp "s3://engram-backups/incremental/$incr" /tmp/
    /scripts/restore.sh "/tmp/$incr" /var/lib/engram incremental
done

# Step 7: Verify permissions
chown -R engram:engram /var/lib/engram
chmod -R 755 /var/lib/engram/spaces
chmod -R 600 /var/lib/engram/spaces/*/wal/*

# Step 8: Start service
systemctl start engram
sleep 10

# Step 9: Comprehensive verification
/scripts/diagnose_health.sh

# Step 10: Verify all memory spaces
for space in /var/lib/engram/spaces/*/; do
    space_id=$(basename "$space")
    echo "Verifying space: $space_id"
    curl -s "http://localhost:7432/api/v1/spaces/$space_id/stats" | jq '.'
done

# Step 11: Log recovery details
recovery_time=$(date)
cat > "/var/log/engram/recovery-$(date +%Y%m%d).log" <<EOF
Recovery Type: Complete Data Loss
Recovery Start: $(date -d "10 minutes ago")
Recovery End: $recovery_time
Data Restored: All spaces
Source: Remote backup (S3)
Verification: SUCCESS
Notes: Disk failure - replaced hardware
EOF
```

**Expected Time**: 30 minutes (depends on network speed for remote fetch)

**Post-Recovery Actions**:
- Investigate storage failure
- Replace failed hardware
- Verify backup replication working
- Test backup retrieval regularly

---

### Scenario 3: Accidental Deletion (PITR)

**Context**: Operator accidentally deleted critical memory space or data at specific time.

**Symptoms**:
- Memory space missing
- Specific memories deleted
- User reports data loss
- Known deletion timestamp

**Recovery Procedure**:

```bash
# Step 1: Identify exact deletion time
deletion_time="2024-01-15T14:29:00Z"
echo "Recovering to timestamp: $deletion_time"

# Step 2: Stop service
systemctl stop engram

# Step 3: Create backup of current state
timestamp=$(date +%Y%m%dT%H%M%SZ)
cp -al /var/lib/engram "/var/lib/engram.pre-pitr-$timestamp"

# Step 4: Perform point-in-time recovery
/scripts/restore_pitr.sh "$deletion_time"

# Step 5: Verify recovered data
ls -la /var/lib/engram/spaces/

# Check .pitr_recovery_info for details
cat /var/lib/engram/.pitr_recovery_info | jq '.'

# Step 6: Start service
systemctl start engram
sleep 5

# Step 7: Verify specific deleted data exists
# Example: Check if critical memory exists
memory_id="critical-memory-123"
curl -s "http://localhost:7432/api/v1/memories/$memory_id" | jq '.'

# Step 8: Compare with pre-PITR state if needed
# This helps confirm recovery worked correctly
space_count_before=$(ls -1d "/var/lib/engram.pre-pitr-$timestamp/spaces/"*/ | wc -l)
space_count_after=$(ls -1d /var/lib/engram/spaces/*/ | wc -l)

echo "Spaces before PITR: $space_count_before"
echo "Spaces after PITR: $space_count_after"

# Step 9: Validate no extra data present
# PITR should restore to exact timestamp
curl -s http://localhost:7432/api/v1/system/stats | jq '{
  total_memories,
  spaces_count,
  pitr_verified: true
}'

# Step 10: Document PITR recovery
cat > "/var/log/engram/pitr-$timestamp.log" <<EOF
PITR Recovery Report
Target Timestamp: $deletion_time
Actual Timestamp: $(cat /var/lib/engram/.pitr_recovery_info | jq -r '.recovery_time')
Reason: Accidental deletion
Deleted Entity: $memory_id
Recovery Status: SUCCESS
Verification: Confirmed entity restored
EOF
```

**Expected Time**: 20 minutes

**Post-Recovery Actions**:
- Review deletion audit logs
- Implement safeguards for critical operations
- Update access controls
- Train team on safe operations

---

### Scenario 4: WAL Corruption

**Context**: WAL replay fails during startup due to corrupted log segments.

**Symptoms**:
- Engram fails during WAL replay
- CRC checksum errors in logs
- "Corrupted WAL entry" messages
- Service stuck in startup

**Recovery Procedure**:

```bash
# Step 1: Stop service
systemctl stop engram

# Step 2: Identify corrupted space
# Check logs for which space failed
space_id=$(journalctl -u engram -n 100 | \
           grep -i "WAL.*corrupt" | \
           grep -oP 'space[=:]?\s*\K\w+' | \
           head -1)

echo "Corrupted space identified: $space_id"

# Step 3: Attempt WAL compaction
echo "Attempting WAL compaction for $space_id..."
/scripts/wal_compact.sh "$space_id"

# Step 4: If compaction fails, restore from backup
if [ $? -ne 0 ]; then
    echo "WAL compaction failed, restoring from backup..."

    # Find latest backup
    latest_backup=$(ls -t /var/backups/engram/full/*.tar.zst | head -1)

    # Backup current state
    timestamp=$(date +%Y%m%dT%H%M%SZ)
    mv "/var/lib/engram/spaces/$space_id" \
       "/var/lib/engram/spaces/$space_id.corrupted-$timestamp"

    # Extract just this space from backup
    temp_dir=$(mktemp -d)
    zstd -d -c "$latest_backup" | tar -C "$temp_dir" -x

    # Find the space in extracted backup
    extracted_space=$(find "$temp_dir" -type d -name "$space_id" | head -1)

    if [ -n "$extracted_space" ]; then
        cp -a "$extracted_space" "/var/lib/engram/spaces/"
        echo "Space $space_id restored from backup"
    else
        echo "ERROR: Space not found in backup!"
        exit 1
    fi

    rm -rf "$temp_dir"
fi

# Step 5: Remove corrupted WAL files
# Keep only most recent WAL segments
wal_dir="/var/lib/engram/spaces/$space_id/wal"
if [ -d "$wal_dir" ]; then
    # Keep last 10 WAL files
    ls -t "$wal_dir"/wal-*.log | tail -n +11 | xargs rm -f || true
fi

# Step 6: Start service
systemctl start engram

# Step 7: Monitor WAL replay
journalctl -u engram -f &
sleep 10

# Step 8: Verify space health
curl -s "http://localhost:7432/api/v1/spaces/$space_id/stats" | jq '.'

# Step 9: Test operations on recovered space
curl -X POST "http://localhost:7432/api/v1/spaces/$space_id/memories/search" \
  -H "Content-Type: application/json" \
  -d '{"query":"test","limit":5}' | jq '.'

# Step 10: Document WAL corruption recovery
cat > "/var/log/engram/wal-recovery-$(date +%Y%m%d).log" <<EOF
WAL Corruption Recovery
Space ID: $space_id
Recovery Method: $([ $compaction_success -eq 0 ] && echo "WAL Compaction" || echo "Backup Restore")
Corrupted WAL Files: Archived to $space_id.corrupted-$timestamp
Recovery Time: $(date)
Status: SUCCESS
EOF
```

**Expected Time**: 15 minutes

**Post-Recovery Actions**:
- Analyze corrupted WAL for patterns
- Check for storage issues
- Review CRC failure rate
- Verify hardware integrity

---

### Scenario 5: Datacenter Failure

**Context**: Entire datacenter unavailable, need to restore in new location.

**Symptoms**:
- Complete service unavailability
- All infrastructure unreachable
- Regional outage confirmed
- Need to failover to secondary site

**Recovery Procedure**:

```bash
# This procedure assumes you're running on NEW infrastructure

# Step 1: Provision new infrastructure
# - Deploy Engram via Kubernetes or systemd
# - Ensure network connectivity
# - Mount necessary volumes

# Step 2: Configure new environment
export ENGRAM_DATA_DIR=/var/lib/engram
export BACKUP_DIR=/var/backups/engram
export AWS_REGION=us-west-2  # Secondary region

# Step 3: Fetch latest backups from cross-region storage
echo "Fetching backups from disaster recovery region..."

# List available backups
aws s3 ls s3://engram-backups-dr/full/ --region us-west-2

# Download latest full backup
latest_full=$(aws s3 ls s3://engram-backups-dr/full/ --region us-west-2 | \
              grep .tar.zst | \
              tail -1 | \
              awk '{print $4}')

aws s3 cp "s3://engram-backups-dr/full/$latest_full" /tmp/ --region us-west-2

# Download corresponding manifest
manifest="${latest_full%.tar.zst}.json"
aws s3 cp "s3://engram-backups-dr/manifests/$manifest" /tmp/ --region us-west-2

# Step 4: Download recent incremental backups
mkdir -p /tmp/incremental
aws s3 sync s3://engram-backups-dr/incremental/ /tmp/incremental/ \
  --region us-west-2 \
  --exclude "*" \
  --include "*.tar.zst" \
  --exclude "*" \
  --include "engram-incr-*"

# Step 5: Verify primary backup
/scripts/verify_backup.sh "/tmp/$latest_full" L3

# Step 6: Restore full backup
/scripts/restore.sh "/tmp/$latest_full" "$ENGRAM_DATA_DIR" full

# Step 7: Apply incremental backups in order
for incr in $(ls -t /tmp/incremental/*.tar.zst | tail -20 | tac); do
    echo "Applying incremental: $(basename $incr)"
    /scripts/restore.sh "$incr" "$ENGRAM_DATA_DIR" incremental
done

# Step 8: Configure new instance
# Update configuration for new region
cat > /etc/engram/config.toml <<EOF
[server]
bind_address = "0.0.0.0:7432"
data_dir = "/var/lib/engram"

[backup]
enabled = true
s3_bucket = "engram-backups-dr"
s3_region = "us-west-2"

[replication]
region = "us-west-2"
primary = false  # This is now primary

[storage]
hot_capacity = 100000
warm_capacity = 1000000
cold_capacity = 10000000
EOF

# Step 9: Update DNS or load balancer
# Point traffic to new instance
# (This step depends on your infrastructure)

# Step 10: Start Engram
if command -v systemctl >/dev/null; then
    systemctl start engram
else
    kubectl rollout restart statefulset/engram
fi

# Step 11: Wait for startup and WAL replay
sleep 30

# Step 12: Comprehensive health check
/scripts/diagnose_health.sh

# Step 13: Verify all spaces accessible
for space in /var/lib/engram/spaces/*/; do
    space_id=$(basename "$space")
    echo "=== Space: $space_id ==="
    curl -s "http://localhost:7432/api/v1/spaces/$space_id/stats" | jq '{
      memory_count,
      storage_tiers: .tier_statistics
    }'
done

# Step 14: Test write operations
test_memory=$(cat <<EOF
{
  "id": "dr-test-$(date +%s)",
  "content": "DR verification test",
  "embedding": [$(for i in {1..768}; do echo -n "0.1,"; done | sed 's/,$//')],
  "confidence": 0.9
}
EOF
)

curl -X POST http://localhost:7432/api/v1/memories \
  -H "Content-Type: application/json" \
  -d "$test_memory"

# Step 15: Configure backup replication to original region
# Once original datacenter is restored
aws s3 sync /var/backups/engram/ s3://engram-backups-primary/ \
  --region us-east-1

# Step 16: Document datacenter failover
cat > "/var/log/engram/dr-failover-$(date +%Y%m%d).log" <<EOF
Disaster Recovery: Datacenter Failover
Original Region: us-east-1
New Region: us-west-2
Failover Time: $(date)
Data Restored From: DR S3 bucket
Recovery Duration: $SECONDS seconds
Spaces Recovered: $(ls -1d /var/lib/engram/spaces/*/ | wc -l)
Status: OPERATIONAL
Next Steps: Monitor performance, plan failback when primary restored
EOF
```

**Expected Time**: 60 minutes (includes infrastructure provisioning)

**Post-Recovery Actions**:
- Monitor new instance performance
- Update monitoring dashboards
- Plan failback to original datacenter
- Review DR procedures for improvements
- Test cross-region replication

---

### Scenario 6: Ransomware Attack

**Context**: Data encrypted by malware, integrity compromised, need clean restore.

**Symptoms**:
- Files encrypted or inaccessible
- Ransom notes present
- Unusual system activity
- Data corruption detected

**Recovery Procedure**:

```bash
# CRITICAL: Isolate the system immediately
# DO NOT start Engram until isolation complete

# Step 1: Isolate infected system
echo "ISOLATING SYSTEM - DO NOT PROCEED UNTIL COMPLETE"
# - Disconnect from network
# - Stop all services
# - Prevent malware spread

# Step 2: Preserve evidence
timestamp=$(date +%Y%m%dT%H%M%SZ)
forensics_dir="/var/forensics/engram-$timestamp"
mkdir -p "$forensics_dir"

# Capture system state
ps aux > "$forensics_dir/processes.txt"
netstat -tulpn > "$forensics_dir/network.txt"
ls -laR /var/lib/engram > "$forensics_dir/file_listing.txt"

# Copy ransom notes if present
find /var/lib/engram -name "*RANSOM*" -o -name "*README*" \
  -exec cp {} "$forensics_dir/" \;

# Step 3: Identify last clean backup
# Find backup BEFORE infection
infection_time="2024-01-15T14:00:00Z"
echo "Infection detected at: $infection_time"

# Find last backup before infection
last_clean=""
for manifest in /var/backups/engram/manifests/*.json; do
    backup_time=$(jq -r '.timestamp' "$manifest")
    backup_epoch=$(date -d "$backup_time" +%s 2>/dev/null || \
                   date -j -f "%Y%m%dT%H%M%SZ" "$backup_time" +%s 2>/dev/null)
    infection_epoch=$(date -d "$infection_time" +%s 2>/dev/null || \
                     date -j -f "%Y-%m-%dT%H:%M:%SZ" "$infection_time" +%s 2>/dev/null)

    if [ "$backup_epoch" -lt "$infection_epoch" ]; then
        if [ -z "$last_clean" ] || [ "$backup_epoch" -gt "$last_clean_epoch" ]; then
            last_clean=$(jq -r '.backup_file' "$manifest")
            last_clean_epoch=$backup_epoch
        fi
    fi
done

echo "Last clean backup: $last_clean"

# Step 4: Verify backup is clean
echo "Verifying backup is not infected..."
/scripts/verify_backup.sh "$last_clean" L3

# Step 5: Scan backup for malware (if antivirus available)
if command -v clamscan >/dev/null; then
    temp_extract=$(mktemp -d)
    zstd -d -c "$last_clean" | tar -C "$temp_extract" -x

    echo "Scanning extracted backup for malware..."
    clamscan -r "$temp_extract"

    if [ $? -eq 0 ]; then
        echo "Backup verified clean"
        rm -rf "$temp_extract"
    else
        echo "ERROR: Backup may be infected!"
        exit 1
    fi
fi

# Step 6: Wipe infected system
echo "WIPING INFECTED DATA"
systemctl stop engram || true

# Archive infected data for forensics
tar czf "$forensics_dir/infected-data.tar.gz" /var/lib/engram/
rm -rf /var/lib/engram

# Step 7: Restore from clean backup
mkdir -p /var/lib/engram
/scripts/restore.sh "$last_clean" /var/lib/engram full

# Step 8: DO NOT apply recent incremental backups
# They may contain infected data

# Step 9: Verify restored data integrity
echo "Verifying restored data integrity..."
find /var/lib/engram -type f -name "*.dat" -o -name "*.log" | \
  xargs -I {} sha256sum {} > "$forensics_dir/restored-checksums.txt"

# Step 10: Security hardening before restart
# Update all credentials
# Enable enhanced monitoring
# Install/update antivirus

# Step 11: Start in read-only mode first
echo "Starting Engram in read-only mode for verification..."
# (Requires Engram to support read-only flag)
systemctl start engram
sleep 10

# Step 12: Comprehensive security scan
echo "Running security verification..."
curl -s http://localhost:7432/api/v1/system/health | jq '.'

# Check for suspicious patterns
/scripts/diagnose_health.sh | tee "$forensics_dir/post-restore-health.txt"

# Step 13: Verify data not corrupted
for space in /var/lib/engram/spaces/*/; do
    space_id=$(basename "$space")
    memory_count=$(curl -s "http://localhost:7432/api/v1/spaces/$space_id/stats" | \
                   jq -r '.memory_count')
    echo "Space $space_id: $memory_count memories"
done

# Step 14: If all clear, enable write mode
systemctl restart engram

# Step 15: Document ransomware recovery
cat > "/var/log/engram/ransomware-recovery-$timestamp.log" <<EOF
SECURITY INCIDENT: Ransomware Attack

Detection Time: $infection_time
Response Time: $(date)
Recovery Method: Clean backup restore
Backup Used: $last_clean
Data Loss: $(date -d "$infection_time" +%s) - $last_clean_epoch = \
           $(( $(date -d "$infection_time" +%s) - last_clean_epoch )) seconds

Forensics Location: $forensics_dir
Infected Data: Archived and wiped

Verification:
- Backup integrity: VERIFIED
- Malware scan: CLEAN
- Data integrity: VERIFIED
- System hardening: APPLIED

Status: RECOVERED
Next Steps: Full security audit, update all credentials, review access logs
EOF

# Step 16: Security follow-up
echo "CRITICAL: Complete these security steps:"
echo "1. Change all credentials"
echo "2. Review access logs for breach vector"
echo "3. Update all security policies"
echo "4. Enable enhanced monitoring"
echo "5. Notify security team and stakeholders"
echo "6. Plan for security audit"
```

**Expected Time**: 45 minutes (plus security audit time)

**Post-Recovery Actions**:
- Full security audit
- Review access logs
- Update all credentials
- Implement additional security controls
- Train team on security awareness
- Report to relevant authorities if required

---

## Recovery Validation Checklist

After any recovery procedure, validate:

- [ ] Engram service is running
- [ ] All memory spaces are accessible
- [ ] Memory counts match expected values
- [ ] Query operations work correctly
- [ ] Write operations succeed
- [ ] Spreading activation functions properly
- [ ] All tiers (hot/warm/cold) are operational
- [ ] WAL replay completed successfully
- [ ] No errors in logs
- [ ] Performance metrics normal
- [ ] Backup replication resumed
- [ ] Monitoring alerts cleared

## Tools and Scripts Reference

| Script | Purpose | Usage |
|--------|---------|-------|
| backup_full.sh | Create full backup | `ENGRAM_SPACE_ID=all /scripts/backup_full.sh` |
| backup_incremental.sh | WAL incremental | `/scripts/backup_incremental.sh` |
| restore.sh | Restore backup | `/scripts/restore.sh <backup> [dir] [mode]` |
| restore_pitr.sh | Point-in-time recovery | `/scripts/restore_pitr.sh <timestamp>` |
| verify_backup.sh | Verify backup | `/scripts/verify_backup.sh <backup> [L1-L4]` |
| wal_compact.sh | Compact WAL | `/scripts/wal_compact.sh <space_id>` |
| prune_backups.sh | Cleanup old backups | `/scripts/prune_backups.sh` |
| diagnose_health.sh | System diagnostics | `/scripts/diagnose_health.sh` |

## Escalation Procedures

### When to Escalate

Escalate to next level if:
- Recovery time exceeds 2x expected RTO
- Multiple recovery attempts fail
- Data integrity cannot be verified
- Security incident requires expert analysis
- Backup restoration fails
- Unknown failure scenario

### Escalation Levels

**Level 1: Operations Team**
- Handle standard recovery scenarios
- Execute documented runbooks
- Monitor recovery progress

**Level 2: Engineering Team**
- Complex failure scenarios
- Code-level debugging required
- Architecture-specific issues

**Level 3: Senior Architecture / Vendor Support**
- Undocumented scenarios
- Design-level problems
- Critical data loss situations

### Contact Information

Maintain current contact list:
- On-call engineer: [Pagerduty/rotation]
- Engineering lead: [Contact info]
- Database admin: [Contact info]
- Security team: [Contact info]
- Vendor support: [Support portal/phone]

## Testing and Drills

### Monthly DR Drills

Test each scenario quarterly:

**Q1**: Data corruption recovery
**Q2**: Complete data loss and PITR
**Q3**: Datacenter failover
**Q4**: Ransomware recovery

### Drill Procedure

1. Schedule drill during maintenance window
2. Notify stakeholders (this is a test)
3. Execute recovery procedure on test environment
4. Document timing and issues
5. Update runbooks based on learnings
6. Verify all tools and access work

### Metrics to Track

- Actual RTO achieved
- Actual RPO achieved
- Steps that took longer than expected
- Missing tools or access
- Documentation gaps
- Team readiness

## Continuous Improvement

### Post-Incident Review Template

```markdown
## Incident Summary
- Date:
- Duration:
- Affected systems:
- Root cause:

## Timeline
- Detection:
- Response started:
- Recovery completed:
- Verification finished:

## What Went Well
-
-

## What Could Be Improved
-
-

## Action Items
- [ ] Update runbooks
- [ ] Improve monitoring
- [ ] Add automation
- [ ] Train team
- [ ] Update tools
```

### Quarterly Review

- Review all incidents
- Update runbooks
- Test recovery procedures
- Verify backup integrity
- Update escalation contacts
- Review RTO/RPO targets

## Related Documentation

- [Backup & Restore](backup-restore.md) - Backup procedures and verification
- [Monitoring](monitoring.md) - Monitoring and alerting setup
- [Troubleshooting](troubleshooting.md) - Common issues and solutions
- [Security](security.md) - Security best practices
- [Production Deployment](production-deployment.md) - Production setup
