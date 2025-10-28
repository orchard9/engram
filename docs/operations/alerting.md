# Alerting and Incident Response Guide

Comprehensive guide to Engram alert rules, response procedures, and escalation criteria.

## Alert Philosophy

Engram alerts are designed following these principles:

1. **Signal over Noise** - Only alert on actionable issues requiring human intervention
2. **Cognitive SLOs** - Thresholds derived from biological plausibility (e.g., <100ms spreading)
3. **Tiered Severity** - Critical (page on-call), Warning (investigate), Info (awareness)
4. **Empirically Validated** - All thresholds tested against baseline distributions
5. **Context-Rich** - Alerts include runbooks, rationale, and validation methods

## Alert Severity Levels

### Critical (severity: critical)
**Response Time:** Immediate (page on-call)
**Impact:** Service degraded or unavailable
**Examples:** EngramDown, ActivationPoolExhaustion, ConsolidationFailureStreak

### Warning (severity: warning)
**Response Time:** Within 30 minutes (business hours)
**Impact:** Performance degraded, SLO at risk
**Examples:** SpreadingLatencySLOBreach, ConsolidationStaleness

### Info (severity: info)
**Response Time:** Next business day
**Impact:** Informational, trend awareness
**Examples:** ConsolidationNoveltyStagnation, AdaptiveBatchingNotConverging

## Alert Reference

### Service Availability Alerts

#### EngramDown
**Severity:** Critical
**Trigger:** `up{job="engram"} == 0 for 1m`
**Meaning:** Engram instance unreachable for 1 minute

**Response Procedure:**
1. Check pod/container status:
   ```bash
   kubectl get pods -l app=engram
   kubectl describe pod <pod-name>
   ```

2. Review container logs:
   ```bash
   kubectl logs <pod-name> --tail=100
   ```

3. Check liveness probe:
   ```bash
   kubectl get pod <pod-name> -o yaml | grep -A 10 livenessProbe
   ```

4. Common Causes:
   - OOM kill (check `kubectl describe pod` for OOMKilled)
   - Crash loop (check logs for panic/segfault)
   - Resource limits (CPU throttling, disk full)
   - Network partition

5. Immediate Actions:
   - If OOM: Increase memory limits, investigate leak
   - If crash: Review logs, rollback recent changes
   - If resource limits: Scale resources, investigate spike

**Escalation:** If pod won't start after 3 restart attempts, escalate to on-call engineer

#### HealthProbeFailure
**Severity:** Warning
**Trigger:** `engram_health_status{probe="spreading"} == 2 for 2m`
**Meaning:** Spreading activation health probe reporting critical state

**Response Procedure:**
1. Check activation pool metrics:
   ```promql
   activation_pool_available_records
   activation_pool_hit_rate
   ```

2. Review spreading activation logs:
   ```logql
   {job="engram", target="engram_core::activation"} | json | level="ERROR"
   ```

3. Common Causes:
   - Pool exhaustion (low available records)
   - High failure rate (breaker open)
   - Resource contention (high latency)

4. Immediate Actions:
   - If pool exhaustion: Scale pool size, reduce concurrency
   - If breaker open: Investigate root cause of failures
   - If high latency: Check GPU utilization, network latency

**Auto-Resolution:** Health probe has 2-minute hysteresis, may auto-clear

---

### Cognitive Performance SLO Alerts

#### SpreadingLatencySLOBreach
**Severity:** Warning
**Trigger:** Hot tier P90 latency >100ms for 5 minutes
**Meaning:** Spreading activation exceeding cognitive plausibility threshold

**Response Procedure:**
1. Check current latency distribution:
   ```promql
   histogram_quantile(0.9, rate(engram_spreading_latency_hot_seconds_bucket[5m]))
   ```

2. Identify slow operations:
   ```logql
   {job="engram"} | json | operation="spreading" | duration_ms > 100
   ```

3. Common Causes:
   - Large graph traversals (high fan-out nodes)
   - GPU contention (concurrent spreading operations)
   - Memory pressure (swapping, GC pauses)
   - Tier migration lag (hot tier not warm)

4. Tuning Actions:
   - Adjust adaptive batch sizes:
     ```yaml
     spreading_config:
       adaptive_batching:
         target_latency_ms: 80  # Reduce from 100ms
     ```
   - Increase GPU resources
   - Optimize graph topology (reduce max fan-out)
   - Pre-warm frequently accessed subgraphs

5. Verify improvement:
   ```promql
   rate(engram_spreading_latency_budget_violations_total[5m])
   ```

**Escalation:** If latency remains >150ms after tuning, escalate for architectural review

#### ConsolidationStaleness
**Severity:** Warning
**Trigger:** `engram_consolidation_freshness_seconds > 900 for 5m`
**Meaning:** Consolidation snapshot over 15 minutes old (2x health contract)

**Response Procedure:**
1. Check consolidation scheduler status:
   ```logql
   {job="engram", target="engram_core::consolidation::service"} | json
   ```

2. Verify background worker health:
   ```promql
   rate(engram_consolidation_runs_total[10m])
   ```

3. Common Causes:
   - Scheduler paused/stopped
   - Storage tier unavailable (can't write snapshot)
   - Consolidation deadlock (rare, check logs for timeout)
   - High CPU usage preventing scheduler execution

4. Immediate Actions:
   - Restart consolidation scheduler (if safe):
     ```bash
     # Send HUP signal to reload config
     kubectl exec <pod> -- kill -HUP 1
     ```
   - Check storage tier health:
     ```promql
     engram_storage_tier_utilization_ratio{tier="hot"}
     ```
   - Review consolidation failure logs:
     ```logql
     {job="engram"} | json | level="ERROR" | message =~ "consolidation"
     ```

5. Verify recovery:
   ```promql
   engram_consolidation_freshness_seconds < 450
   ```

**Auto-Resolution:** Alert clears when consolidation runs successfully and freshness <900s

#### ConsolidationFailureStreak
**Severity:** Critical
**Trigger:** 3+ consolidation failures in 15 minutes
**Meaning:** Systematic consolidation issue, not transient error

**Response Procedure:**
1. Identify failure cause from logs:
   ```logql
   {job="engram", target="engram_core::consolidation"} | json | level="ERROR"
   ```

2. Common failure modes:
   - Storage write failures (disk full, permissions)
   - Validation failures (corrupted data)
   - Deadlock/timeout (rare)

3. Immediate Actions:
   - If storage full: Clear space, increase limits
   - If validation failures: Review recent writes, rollback if needed
   - If timeout: Increase consolidation timeout, reduce data size

4. Emergency Mitigation:
   - Disable consolidation temporarily:
     ```yaml
     consolidation:
       enabled: false
     ```
   - System will use last good snapshot until resolved

5. Verify recovery:
   ```promql
   rate(engram_consolidation_runs_total[5m]) > 0
   rate(engram_consolidation_failures_total[5m]) == 0
   ```

**Escalation:** If failures persist after storage/config fixes, escalate immediately

---

### Storage and Capacity Alerts

#### WALLagHigh
**Severity:** Warning
**Trigger:** `engram_wal_lag_seconds > 10 for 5m`
**Meaning:** WAL replay lag exceeds durability SLO

**Response Procedure:**
1. Check WAL write rate:
   ```promql
   rate(engram_wal_writes_total[5m])
   ```

2. Identify replay bottleneck:
   ```logql
   {job="engram", target="engram_core::storage::wal"} | json | duration_ms > 1000
   ```

3. Common Causes:
   - High write volume (replay can't keep up)
   - Disk I/O saturation
   - Compaction blocking replay

4. Immediate Actions:
   - Reduce write rate (rate limiting)
   - Increase disk IOPS
   - Pause non-critical writes

5. Verify improvement:
   ```promql
   engram_wal_lag_seconds < 5
   ```

**Data Loss Risk:** 10s lag = potential loss of last 10s of writes on crash

---

### Activation Pool Alerts

#### ActivationPoolExhaustion
**Severity:** Critical
**Trigger:** `activation_pool_available_records < 10 for 2m`
**Meaning:** Pool nearly exhausted, spreading operations may block/fail

**Response Procedure:**
1. Check current pool state:
   ```promql
   activation_pool_available_records
   activation_pool_in_flight_records
   activation_pool_high_water_mark
   ```

2. Identify resource leak:
   ```logql
   {job="engram"} | json | message =~ "pool.*release.*failed"
   ```

3. Common Causes:
   - Concurrent spreading spike (legitimate load)
   - Pool record leak (not released after use)
   - Pool size too small for workload

4. Immediate Actions:
   - Increase pool size (hot config reload):
     ```yaml
     spreading:
       activation_pool_size: 2048  # Double from 1024
     ```
   - Restart service to clear leaked records (last resort)

5. Verify recovery:
   ```promql
   activation_pool_available_records > 50
   ```

**Impact:** When exhausted, spreading operations block, causing cascading latency

#### ActivationPoolLowHitRate
**Severity:** Warning
**Trigger:** `activation_pool_hit_rate < 0.50 for 15m`
**Meaning:** Pool inefficient, most operations allocate new records

**Response Procedure:**
1. Analyze hit rate trend:
   ```promql
   activation_pool_hit_rate
   activation_pool_total_reused / (activation_pool_total_created + activation_pool_total_reused)
   ```

2. Common Causes:
   - Workload changed (different query patterns)
   - Pool size too small (all slots in use)
   - Record expiration too aggressive

3. Tuning Actions:
   - Increase pool size
   - Adjust record TTL
   - Pre-warm pool for common patterns

4. Target: >80% hit rate under steady load

**Performance Impact:** Low hit rate increases allocation overhead, minor latency impact

---

### Circuit Breaker Alerts

#### SpreadingCircuitBreakerOpen
**Severity:** Warning
**Trigger:** `engram_spreading_breaker_state == 1 for 5m`
**Meaning:** Circuit breaker open, spreading failing fast

**Response Procedure:**
1. Identify failure rate:
   ```promql
   rate(engram_spreading_failures_total[5m]) / rate(engram_spreading_activations_total[5m])
   ```

2. Review failure causes:
   ```logql
   {job="engram"} | json | operation="spreading" | level="ERROR"
   ```

3. Common Causes:
   - Downstream dependency unavailable (GPU, storage)
   - Input validation failures (malformed queries)
   - Resource exhaustion (pool, memory)

4. Immediate Actions:
   - Fix root cause (restore dependency, fix input validation)
   - Breaker will auto-transition to half-open after timeout
   - Manual reset (use with caution):
     ```bash
     curl -X POST http://localhost:7432/api/v1/system/spreading/breaker/reset
     ```

5. Monitor recovery:
   ```promql
   engram_spreading_breaker_state  # Should return to 0 (closed)
   ```

**Auto-Recovery:** Breaker transitions closed → open → half-open → closed automatically

#### SpreadingCircuitBreakerFlapping
**Severity:** Warning
**Trigger:** >3 state transitions in 10 minutes
**Meaning:** Unstable spreading layer, breaker oscillating

**Response Procedure:**
1. Analyze transition pattern:
   ```promql
   rate(engram_spreading_breaker_transitions_total[10m])
   ```

2. Root Cause: Threshold at edge of stability

3. Tuning Actions:
   - Adjust breaker thresholds:
     ```yaml
     spreading:
       circuit_breaker:
         failure_threshold: 10  # Increase from 5
         timeout_seconds: 60    # Increase from 30
     ```

4. Test stability:
   - Monitor for 30 minutes after tuning
   - Should see <1 transition per hour under normal load

**Long-term Fix:** Review spreading reliability, reduce failure rate at source

---

### Adaptive Batching Alerts

#### AdaptiveBatchingNotConverging
**Severity:** Info
**Trigger:** Hot tier confidence <30% for 30 minutes
**Meaning:** Adaptive controller unable to find stable batch size

**Response Procedure:**
1. Check convergence confidence:
   ```promql
   adaptive_batch_hot_confidence
   adaptive_batch_warm_confidence
   adaptive_batch_cold_confidence
   ```

2. Common Causes:
   - Highly variable workload (latency spikes)
   - Guardrails too restrictive
   - Controller parameters need tuning

3. Tuning Actions:
   - Widen guardrails:
     ```yaml
     adaptive_batching:
       min_batch_size: 8   # Reduce from 16
       max_batch_size: 256 # Increase from 128
     ```
   - Adjust learning rate:
     ```yaml
     adaptive_batching:
       alpha: 0.2  # Increase from 0.1 for faster convergence
     ```

4. Monitor improvement:
   ```promql
   avg_over_time(adaptive_batch_hot_confidence[10m])  # Target >50%
   ```

**Impact:** Low confidence doesn't directly impact performance, but limits optimization

#### AdaptiveGuardrailHitRateHigh
**Severity:** Info
**Trigger:** >0.1 guardrail hits/sec for 15 minutes
**Meaning:** Controller frequently hitting configuration limits

**Response Procedure:**
1. Identify which guardrails:
   ```logql
   {job="engram"} | json | message =~ "guardrail"
   ```

2. Common Limits:
   - min_batch_size (controller wants smaller batches)
   - max_batch_size (controller wants larger batches)
   - latency_budget (controller out of headroom)

3. Capacity Planning:
   - If hitting max: Workload outgrowing current capacity
   - If hitting min: Overprovisioned, reduce resources
   - If hitting latency: Need faster infrastructure (GPU, storage)

**Action:** Review capacity plan, schedule scaling if needed

---

## Alert Silencing

### When to Silence

Silence alerts during:
- Planned maintenance windows
- Known degraded states (dependency outage)
- Chaos engineering tests
- Load testing

### How to Silence

**Via Alertmanager UI:**
1. Open http://localhost:9093
2. Click alert → Silence
3. Set duration and reason
4. Include oncall engineer name

**Via CLI:**
```bash
amtool silence add \
  alertname=SpreadingLatencySLOBreach \
  --duration=1h \
  --comment="Load test in progress - oncall: alice"
```

### Best Practices

- Always include reason and oncall name
- Use shortest viable duration
- Document in incident log
- Set reminder to un-silence

---

## Alert Integration

### PagerDuty

Configure in `alertmanager.yml`:
```yaml
receivers:
  - name: pagerduty-critical
    pagerduty_configs:
      - service_key: <integration-key>
        severity: critical
        description: '{{ .CommonAnnotations.summary }}'
        details:
          runbook: '{{ .CommonAnnotations.runbook }}'
          firing: '{{ .Alerts.Firing | len }}'

route:
  group_by: ['alertname', 'component']
  receiver: pagerduty-critical
  routes:
    - match:
        severity: critical
      receiver: pagerduty-critical
      continue: true
```

### Slack

```yaml
receivers:
  - name: slack-warnings
    slack_configs:
      - api_url: <webhook-url>
        channel: '#engram-alerts'
        title: '{{ .CommonAnnotations.summary }}'
        text: '{{ .CommonAnnotations.description }}'
        actions:
          - type: button
            text: 'Runbook'
            url: '{{ .CommonAnnotations.runbook }}'
          - type: button
            text: 'Silence'
            url: '{{ .ExternalURL }}/#/silences/new'
```

---

## Testing Alerts

Validate alerts before production:

```bash
# Test alert query syntax
promtool check rules deployments/prometheus/alerts.yml

# Inject failure to trigger alert
kubectl exec engram-pod -- kill -STOP 1

# Verify alert fires in Prometheus
curl http://localhost:9090/api/v1/alerts | jq '.data.alerts[] | select(.labels.alertname=="EngramDown")'

# Check Alertmanager routing
curl http://localhost:9093/api/v2/alerts
```

---

## Escalation Matrix

| Alert | Severity | Response Time | Escalation After |
|-------|----------|---------------|------------------|
| EngramDown | Critical | Immediate | 15 minutes |
| ActivationPoolExhaustion | Critical | Immediate | 15 minutes |
| ConsolidationFailureStreak | Critical | Immediate | 15 minutes |
| SpreadingLatencySLOBreach | Warning | 30 minutes | 2 hours |
| ConsolidationStaleness | Warning | 30 minutes | 2 hours |
| ActivationPoolLowHitRate | Warning | 1 hour | Next business day |
| ConsolidationNoveltyStagnation | Info | Next day | N/A |
| AdaptiveBatchingNotConverging | Info | Next day | N/A |

---

## Incident Response Checklist

When alert fires:

- [ ] Acknowledge alert in PagerDuty/Slack
- [ ] Check Grafana dashboard for context
- [ ] Review logs via Loki for errors
- [ ] Follow runbook procedure
- [ ] Document actions in incident log
- [ ] Verify alert clears after fix
- [ ] Post-incident: Update runbook if needed
- [ ] Schedule postmortem for critical incidents

---

## Metrics for Alert Health

Monitor your monitoring:

```promql
# Alerts firing by severity
count by (severity) (ALERTS{alertstate="firing"})

# Alert flapping (firing → resolved → firing)
changes(ALERTS{alertname="EngramDown"}[1h]) > 4

# Time to first page
histogram_quantile(0.9, rate(alertmanager_notification_latency_seconds_bucket[1h]))
```

Target SLOs:
- <1% false positive rate
- <5% false negative rate
- <60s time to notification
- <2 flaps per week per alert

---

## Next Steps

- [Monitoring Guide](monitoring.md) - Metrics reference and dashboards
- [Performance Tuning](performance-tuning.md) - Optimize based on alert patterns
- [Troubleshooting](troubleshooting.md) - Detailed debugging procedures
