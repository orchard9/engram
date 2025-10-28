# Incident Response Guide

This guide defines incident response procedures for Engram production failures, including severity levels, escalation paths, and communication templates.

## Quick Reference

| Severity | Response Time | RTO | Impact | Escalation |
|----------|--------------|-----|--------|------------|
| SEV1 | Immediate | 30 min | Service down, data loss | Page on-call → War room |
| SEV2 | <15 min | 2 hours | Degraded performance | Notify on-call → Senior engineer |
| SEV3 | <2 hours | 8 hours | Single user affected | Create ticket → Next window |
| SEV4 | Next day | N/A | Questions, requests | Support ticket |

## Severity Levels

### SEV1 - Critical

**Definition**: Complete service outage, active data loss, or security breach affecting all users.

**Response Time**: Immediate (page on-call engineer)
**RTO**: 30 minutes
**Communication**: Updates every 30 minutes until resolved

**Examples**:
- Engram process down and cannot restart
- Data loss occurring (WAL corruption, filesystem failure)
- Security breach (unauthorized access, data exfiltration)
- Silent data corruption propagating (NaN values spreading)
- All API endpoints returning errors
- Database completely inaccessible

**Required Actions**:
1. Page on-call engineer immediately
2. Start incident timer and create war room (Zoom/Slack)
3. Run diagnostic script: `./scripts/diagnose_health.sh > /tmp/sev1-$(date +%s).txt`
4. Collect debug bundle: `./scripts/collect_debug_info.sh`
5. Post initial status update within 15 minutes
6. Execute emergency recovery if needed: `./scripts/emergency_recovery.sh`
7. Update stakeholders every 30 minutes
8. Document all actions taken

**Decision Criteria**:
- Can ANY user access the service? → NO = SEV1
- Is data being lost RIGHT NOW? → YES = SEV1
- Is this a security incident? → YES = SEV1
- Can we restore service within 30 minutes? → NO = Consider escalation

### SEV2 - High

**Definition**: Degraded performance affecting multiple users or single-space complete outage in multi-tenant deployment.

**Response Time**: <15 minutes (notify on-call)
**RTO**: 2 hours
**Communication**: Initial update within 30 minutes, then hourly

**Examples**:
- High error rates (>5% of requests failing)
- Severe performance degradation (P99 >500ms)
- Critical alerts firing (WALLagCritical, MemoryPressureHigh)
- Single memory space completely down (multi-tenant)
- Consolidation completely stuck
- Index corruption affecting all queries

**Required Actions**:
1. Notify on-call engineer within 15 minutes
2. Run diagnostic script and identify root cause category
3. Apply immediate mitigation (restart, reduce load, failover)
4. Monitor metrics for improvement
5. Post status update within 30 minutes
6. Schedule root cause analysis within 24 hours
7. Update stakeholders hourly until resolved

**Decision Criteria**:
- Are >50% of users affected? → YES = SEV2
- Are error rates >5%? → YES = SEV2
- Is P99 latency >500ms? → YES = SEV2
- Can we restore within 2 hours? → NO = Consider SEV1

### SEV3 - Medium

**Definition**: Single user/space affected with workaround available, or non-critical feature unavailable.

**Response Time**: <2 hours
**RTO**: 8 hours (or next maintenance window)
**Communication**: Status update when resolved

**Examples**:
- Single user experiencing errors (others working)
- Warning alerts firing (WALLagWarning, DiskUsageHigh)
- Non-critical feature down (consolidation stuck, index fallback active)
- Performance degradation only under high load
- Specific query pattern failing
- Minor configuration issue

**Required Actions**:
1. Create incident ticket with diagnosis results
2. Apply fix during next maintenance window if needed
3. Document issue in troubleshooting guide
4. Add monitoring for early detection
5. Notify affected user of resolution

**Decision Criteria**:
- Is only one user affected? → YES = SEV3
- Is there a workaround? → YES = SEV3
- Can we defer fix to maintenance window? → YES = SEV3
- Is service still functional overall? → YES = SEV3

### SEV4 - Low

**Definition**: Questions about usage, feature requests, or minor configuration issues with no impact.

**Response Time**: Next business day
**RTO**: N/A
**Communication**: Response via support ticket

**Examples**:
- Questions about configuration or usage
- Feature requests or enhancements
- Documentation gaps or typos
- Minor performance tuning requests
- Informational alerts

**Required Actions**:
1. Respond via support ticket or email
2. Update documentation if needed
3. Consider for future roadmap
4. Track in backlog for prioritization

## Incident Response Flow

### Phase 1: Detection (0-5 minutes)

**Goal**: Identify and classify the incident

**Actions**:
1. Alert fires or user reports issue
2. Acknowledge alert/ticket
3. Classify severity level (SEV1-4)
4. Start incident timer
5. Open incident tracking ticket

**Tools**:
- Prometheus alerts
- Health check failures
- User reports
- Monitoring dashboards

**Outputs**:
- Incident ticket created
- Severity classification
- Initial responder assigned

### Phase 2: Triage (5-15 minutes)

**Goal**: Understand the scope and impact

**Actions**:
1. Run diagnostic script
   ```bash
   ./scripts/diagnose_health.sh > /tmp/incident-$(date +%s).txt
   ```

2. Collect key metrics
   ```bash
   curl http://localhost:7432/api/v1/system/health > /tmp/health.json
   curl http://localhost:7432/metrics > /tmp/metrics.txt
   ```

3. Analyze recent logs
   ```bash
   ./scripts/analyze_logs.sh "30 minutes ago" > /tmp/log-analysis.txt
   ```

4. Identify affected scope
   - How many users affected?
   - Which memory spaces affected?
   - Which operations failing?
   - Is this isolated or widespread?

5. Classify issue category using [decision trees](./troubleshooting.md#decision-trees):
   - Category 1: Service Failure
   - Category 2: Resource Exhaustion
   - Category 3: Performance Degradation
   - Category 4: Data Integrity
   - Category 5: Configuration/Deployment

**Tools**:
- `diagnose_health.sh`
- `analyze_logs.sh`
- Grafana dashboards
- Decision trees from troubleshooting guide

**Outputs**:
- Root cause hypothesis
- Affected scope quantified
- Issue category identified
- Mitigation plan drafted

### Phase 3: Mitigation (15-60 minutes)

**Goal**: Restore service to functional state

**Actions**:

**For known issues**: Apply resolution from [common-issues.md](./common-issues.md)

**For SEV1 service down**:
1. Check if process is running: `pgrep engram`
2. Attempt restart: `systemctl restart engram`
3. If restart fails, check logs for startup errors
4. Apply emergency recovery if needed
5. If data corruption suspected, restore from backup

**For SEV2 performance degradation**:
1. Identify bottleneck (CPU/memory/disk/network)
2. Apply immediate mitigation:
   - Scale resources if capacity issue
   - Restart if resource leak
   - Reduce load if overloaded
   - Enable degraded mode if available
3. Monitor for improvement

**For data integrity issues**:
1. Assess extent of corruption
2. Stop writes if corruption is spreading (read-only mode)
3. Restore from backup if needed
4. Quarantine corrupted data

**Emergency Recovery**:
```bash
# Always dry-run first
./scripts/emergency_recovery.sh <mode> --dry-run

# Then execute with backup
./scripts/emergency_recovery.sh <mode> --backup-first
```

**Tools**:
- `emergency_recovery.sh`
- `restore.sh`
- Service restart procedures
- Load balancer reconfiguration

**Outputs**:
- Service restored to functional state
- Temporary mitigation in place
- Monitoring confirms improvement

### Phase 4: Resolution (60 minutes - ongoing)

**Goal**: Fully resolve the underlying issue

**Actions**:
1. Apply permanent fix (not just mitigation)
2. Verify fix doesn't introduce new issues
3. Run comprehensive testing
4. Monitor for recurrence (24-48 hours)
5. Remove temporary mitigations
6. Restore normal operations

**Verification Steps**:
```bash
# Health check
./scripts/diagnose_health.sh

# Performance benchmark
./scripts/benchmark_deployment.sh 60 10

# Log analysis (should show no new errors)
./scripts/analyze_logs.sh "1 hour ago"

# Monitor metrics
watch 'curl -s http://localhost:7432/api/v1/system/health | jq .'
```

**Outputs**:
- Permanent fix applied
- All tests passing
- Metrics normal
- Incident can be closed

### Phase 5: Post-Incident (24-72 hours)

**Goal**: Learn from the incident and prevent recurrence

**Actions**:
1. Write incident report (use template below)
2. Conduct post-incident review meeting
3. Identify root cause and contributing factors
4. Update runbooks and documentation
5. Implement prevention measures
6. Update monitoring and alerts
7. Share learnings with team

**Outputs**:
- Incident report published
- Action items assigned with owners
- Documentation updated
- Prevention measures implemented

## Escalation Paths

### Level 1: On-Call Operator (First Responder)

**Capabilities**:
- Run diagnostic scripts
- Apply known fixes from runbooks
- Restart services
- Execute emergency recovery procedures
- Collect debug information

**Authority**:
- Non-destructive operations
- Service restarts
- Apply configuration changes
- Execute documented recovery procedures

**Escalation Triggers**:
- Unknown error patterns not in troubleshooting guide
- Data corruption detected
- Security concerns
- Resolution attempts failed after 30 minutes
- Permanent fix requires code changes

**Response Time**: Immediate (for SEV1/2)

**Contact**: Pager/on-call rotation

### Level 2: Senior Engineer (Subject Matter Expert)

**Capabilities**:
- Code analysis and debugging
- Debug builds and profiling
- Direct database inspection
- Custom recovery scripts
- Customer communication

**Authority**:
- Destructive operations (with backup)
- Emergency configuration changes
- Direct customer communication
- Approve emergency patches

**Escalation Triggers**:
- Multi-hour outage
- Complex data corruption
- Architectural issues
- Need for code changes
- Customer escalation

**Response Time**: <2 hours (for SEV1), <8 hours (for SEV2)

**Contact**: Direct phone/Slack to senior engineer on-call

### Level 3: Core Development Team

**Capabilities**:
- Source code fixes
- Emergency patches and releases
- Architectural decisions
- Design changes

**Authority**:
- All operations
- Release emergency patch
- Customer notification for widespread issues
- Post-mortem decisions

**Escalation Triggers**:
- Software bugs requiring code fix
- Design flaws requiring architectural change
- Security vulnerabilities
- Need for emergency release

**Response Time**: <4 hours (for SEV1), next business day (for SEV2)

**Contact**: Engineering team lead or CTO

## Escalation Decision Matrix

| Situation | Level 1 (Operator) | Level 2 (Engineer) | Level 3 (Dev Team) |
|-----------|-------------------|-------------------|-------------------|
| Service won't start (known issue) | Fix | - | - |
| Service won't start (unknown) | Escalate → | Debug | - |
| High latency (config) | Tune | - | - |
| High latency (code) | Escalate → | Profile | Fix |
| WAL corruption (single file) | Quarantine | - | - |
| WAL corruption (widespread) | Escalate → | Restore | Investigate |
| NaN values appearing | Sanitize | - | - |
| NaN values reappearing | Escalate → | Debug | Fix |
| Multi-space isolation broken | Escalate → | Investigate | Patch |
| Security breach | Escalate → Escalate → | All-hands |
| Performance tuning needed | Attempt | Optimize | - |
| Index corruption | Rebuild | - | - |
| gRPC not working | Fix config | - | - |
| Feature not working as designed | - | - | Triage |

## Communication Templates

### SEV1 Incident Notification (Internal)

**Subject**: [SEV1] Engram Production Outage - [Brief Description]

```
SEVERITY: SEV1 (Critical)
START TIME: YYYY-MM-DD HH:MM UTC
STATUS: Investigating / Mitigating / Resolved
IMPACT: [e.g., "All users unable to access service" / "Active data loss in progress"]

SYMPTOMS:
- [Primary symptom: e.g., "HTTP 503 errors on all requests"]
- [Secondary symptoms: e.g., "Process consuming 100% CPU and unresponsive"]

AFFECTED SCOPE:
- Users: [All / Specific tenants / Percentage]
- Operations: [Which operations are failing]
- Duration: [How long has this been happening]

DIAGNOSIS:
- [Initial findings from diagnostic script]
- [Error patterns identified in logs]
- [Metrics showing the issue]

IMMEDIATE ACTIONS TAKEN:
- [Actions taken so far with timestamps]
- [Current mitigation in progress]

NEXT STEPS:
- [Planned actions with expected timelines]
- [ETA for next update]

WAR ROOM: [Zoom/Slack link]
INCIDENT COMMANDER: [Name]
ON-CALL ENGINEER: [Name]

Next update: [Timestamp - 30 minutes from now]
```

### SEV1 Progress Update (Internal)

**Subject**: [SEV1] Update [#N] - [Brief Description]

```
SEVERITY: SEV1 (Critical)
STATUS: [Investigating / Mitigating / Resolved]
UPDATE TIME: YYYY-MM-DD HH:MM UTC
ELAPSED TIME: [Duration since start]

PROGRESS SINCE LAST UPDATE:
- [What was attempted]
- [Results of attempts]
- [New findings]

CURRENT STATUS:
- [What's happening now]
- [What's working / not working]

NEXT STEPS:
- [Immediate next actions]
- [Expected timeline]

ETA TO RESOLUTION: [Best estimate or "Unknown - continuing investigation"]

Next update: [Timestamp]
```

### SEV1 Customer Communication (External)

**Subject**: Service Disruption - [YYYY-MM-DD HH:MM UTC]

```
We are currently experiencing a service disruption affecting memory operations.

IMPACT: [Describe user-visible impact in non-technical terms]
- [e.g., "Users cannot create or query memories"]
- [e.g., "Read operations work but writes are failing"]

START TIME: [When issue began]
CURRENT STATUS: [Investigating / Working on fix / Testing fix / Resolved]

Our team is actively working to restore service. We will provide updates
every 30 minutes until the issue is resolved.

For urgent inquiries: support@engram.example.com
Status page: https://status.engram.example.com

Next update: [Timestamp]

We apologize for the inconvenience and appreciate your patience.
```

### SEV1 Resolution Notification (External)

**Subject**: Service Restored - [YYYY-MM-DD HH:MM UTC]

```
The service disruption that began at [START TIME] has been resolved.

RESOLUTION TIME: YYYY-MM-DD HH:MM UTC
DURATION: [X hours Y minutes]

WHAT HAPPENED:
[Brief, non-technical explanation of the issue]

WHAT WE DID:
[Brief explanation of the fix]

IMPACT:
- Users affected: [Number or scope]
- Operations affected: [What wasn't working]
- Data loss: [None / Scope if any]

PREVENTION:
We are implementing the following measures to prevent recurrence:
- [Specific preventive action #1]
- [Specific preventive action #2]

A detailed post-incident report will be published within 72 hours at:
https://status.engram.example.com/incident-reports

Thank you for your patience during this incident. If you have questions
or concerns, please contact support@engram.example.com.
```

### Post-Incident Report Template

**File**: `incident-reports/INC-YYYY-MM-DD-NNN.md`

```markdown
# Incident Report: [Short Descriptive Title]

**Incident ID**: INC-YYYY-MM-DD-NNN
**Severity**: SEV1 / SEV2 / SEV3 / SEV4
**Date**: YYYY-MM-DD
**Duration**: X hours Y minutes
**Impact**: [Users affected, operations impacted, data loss if any]
**Status**: Resolved

## Executive Summary

[2-3 sentence summary of what happened, why, and how it was resolved]

## Timeline (All times in UTC)

| Time | Event | Actor |
|------|-------|-------|
| HH:MM | Incident begins (first alert or symptom) | System |
| HH:MM | Alert fires | Monitoring |
| HH:MM | On-call engineer paged | System |
| HH:MM | Engineer acknowledges | [Name] |
| HH:MM | War room created | [Name] |
| HH:MM | Root cause identified | [Name] |
| HH:MM | Mitigation applied | [Name] |
| HH:MM | Service partially restored | [Name] |
| HH:MM | Full service restored | [Name] |
| HH:MM | Incident closed | [Name] |

## Root Cause

### Technical Explanation

[Detailed technical explanation of what failed and why. Be specific about:
- Which component failed
- What triggered the failure
- Why existing safeguards didn't prevent it
- What conditions were necessary for this to occur]

### Contributing Factors

1. [Factor #1: e.g., "Insufficient monitoring of consolidation lag"]
2. [Factor #2: e.g., "No automatic recovery for this failure mode"]
3. [Factor #3: e.g., "Gap in testing coverage for this scenario"]

## Impact Analysis

### Users Affected
- Total users: [Number or percentage]
- Affected tenants: [List if multi-tenant]
- Geographic distribution: [If relevant]

### Operations Impacted
- Operations failed: [Number and percentage of total]
- Operations degraded: [Number with latency >threshold]
- Read vs Write: [Breakdown if relevant]

### Data Integrity
- Data loss: [Yes/No, scope if yes]
- Data corruption: [Yes/No, scope if yes]
- Recovery: [How data was recovered if applicable]

### Business Impact
- Duration: [Time to detect + time to resolve]
- Revenue impact: [If calculable]
- SLA impact: [If SLA exists]
- Customer escalations: [Number]

## Detection

### How We Found Out
[How the incident was detected: alert, user report, monitoring, etc.]

### Time to Detect
[How long between incident start and detection]

### Why Detection Took This Long
[If detection was delayed, explain why]

## Response

### What Went Well
- [Positive aspect #1: e.g., "Diagnostic scripts quickly identified the issue"]
- [Positive aspect #2: e.g., "Team responded within 5 minutes"]
- [Positive aspect #3: e.g., "Communication was clear and frequent"]

### What Didn't Go Well
- [Problem #1: e.g., "Lacked runbook for this specific scenario"]
- [Problem #2: e.g., "Emergency recovery script had a bug"]
- [Problem #3: e.g., "Escalation path was unclear"]

### Resolution Steps

1. [Step 1 with timestamp and result]
2. [Step 2 with timestamp and result]
3. [Step 3 with timestamp and result]
...

## Prevention Measures

### Immediate Actions (Completed)

| Action | Owner | Completion Date | Status |
|--------|-------|-----------------|--------|
| [Immediate fix applied] | [Name] | YYYY-MM-DD | Complete |
| [Monitoring added] | [Name] | YYYY-MM-DD | Complete |

### Short-Term Actions (1-4 weeks)

| Action | Owner | Target Date | Status |
|--------|-------|-------------|--------|
| [Improve monitoring for X] | [Name] | YYYY-MM-DD | In Progress |
| [Update runbook for Y] | [Name] | YYYY-MM-DD | Pending |
| [Add test coverage for Z] | [Name] | YYYY-MM-DD | Pending |

### Long-Term Actions (1-3 months)

| Action | Owner | Target Date | Status |
|--------|-------|-------------|--------|
| [Architectural improvement] | [Name] | YYYY-MM-DD | Planned |
| [Automated recovery for scenario] | [Name] | YYYY-MM-DD | Planned |

## Lessons Learned

### Technical Lessons
- [Lesson #1: e.g., "System needs automatic recovery for consolidation deadlocks"]
- [Lesson #2: e.g., "WAL corruption can propagate if not detected early"]

### Process Lessons
- [Lesson #1: e.g., "Need clearer escalation criteria in runbooks"]
- [Lesson #2: e.g., "War room should be created within 5 minutes for SEV1"]

### Documentation Updates
- [Update #1: Added Issue #11 to common-issues.md]
- [Update #2: Updated decision tree for this scenario]

## Appendix

### Links
- Incident ticket: [URL]
- War room chat log: [URL]
- Grafana dashboard: [URL]
- Debug bundle: [URL]

### Diagnostic Output
```
[Relevant diagnostic output]
```

### Error Logs
```
[Relevant error logs]
```

### Metrics
[Screenshots or data showing the incident in metrics]

---

**Report Author**: [Name]
**Review Date**: YYYY-MM-DD
**Approved By**: [Name, Title]
```

## Incident Management Best Practices

### During an Incident

**DO**:
- Stay calm and methodical
- Document all actions with timestamps
- Communicate frequently (every 30 minutes for SEV1)
- Focus on restoration first, investigation second
- Use diagnostic tools before manual debugging
- Create backups before destructive operations
- Follow runbooks when available

**DON'T**:
- Panic or rush without thinking
- Make changes without documenting them
- Go silent (lack of update is worse than "no progress" update)
- Skip backups to save time
- Try multiple fixes simultaneously
- Restart services without understanding why they failed
- Forget to monitor after applying fix

### Communication Guidelines

**Internal Communication**:
- Use dedicated war room (Zoom/Slack)
- Assign incident commander role
- Use structured updates (status, actions, next steps)
- Keep stakeholders informed
- Document everything in incident ticket

**External Communication**:
- Be honest but not alarmist
- Use simple, non-technical language
- State what we know, what we're doing, when next update
- Never speculate about causes until confirmed
- Apologize for inconvenience
- Follow up with post-incident report

### Handoff Procedures

If incident spans multiple shifts:

1. Schedule handoff meeting 15 minutes before shift change
2. Outgoing engineer provides:
   - Current status summary
   - Actions taken so far
   - Current hypothesis
   - Next steps planned
   - Outstanding questions
3. Incoming engineer confirms understanding
4. Update war room with handoff complete
5. Outgoing engineer remains available for 30 minutes

## Tools and Resources

### Diagnostic Tools
- `./scripts/diagnose_health.sh` - 10 health checks in <30s
- `./scripts/collect_debug_info.sh` - Debug bundle in <1 min
- `./scripts/analyze_logs.sh` - Log analysis with recommendations
- `./scripts/emergency_recovery.sh` - Emergency recovery modes

### Documentation
- [Troubleshooting Guide](./troubleshooting.md) - Decision trees and error patterns
- [Common Issues](./common-issues.md) - Top 10 issues with resolutions
- [Monitoring Guide](./monitoring.md) - Metrics and alerts
- [Backup and Restore](./backup-restore.md) - Recovery procedures

### Dashboards
- Engram Health Dashboard: [URL]
- Performance Metrics: [URL]
- Alert Status: [URL]

### Contact Information
- On-call rotation: [Pager link]
- Engineering escalation: [Slack channel]
- Customer support: support@engram.example.com
- Security incidents: security@engram.example.com

## Training and Drills

### Incident Response Training

All on-call engineers must complete:
1. Incident response procedure review
2. Troubleshooting guide walkthrough
3. Tool familiarity (all diagnostic scripts)
4. Simulated SEV1 incident (tabletop exercise)
5. Communication template practice

### Tabletop Exercises

Quarterly tabletop exercises covering:
- SEV1 complete outage
- SEV2 performance degradation
- Data corruption scenario
- Security incident
- Multi-failure cascade

### Success Criteria
- Response within defined time for severity
- Correct use of diagnostic tools
- Proper escalation decisions
- Clear communication
- Incident successfully resolved
- Post-incident report completed

## Metrics and Continuous Improvement

### Incident Metrics to Track
- Time to detect (alert to acknowledgment)
- Time to mitigate (acknowledgment to service restored)
- Time to resolve (acknowledgment to root cause fixed)
- Number of incidents by severity per month
- Repeat incidents (same root cause)
- Escalation rate (% requiring Level 2 or 3)

### Goals
- SEV1 MTTD (Mean Time To Detect) < 5 minutes
- SEV1 MTTR (Mean Time To Resolve) < 30 minutes
- SEV2 MTTR < 2 hours
- <10% repeat incidents
- >80% SEV3 resolved by Level 1

### Review Cadence
- Weekly: Review all incidents from past week
- Monthly: Incident trends and metrics review
- Quarterly: Process improvements and training updates

---

For active incidents, see: [Incident Tracking Dashboard](./incident-tracking)

For historical incidents, see: [Incident Reports Archive](./incident-reports)
