# Engram Operational Guide

*Production-ready procedures designed for cognitive ergonomics under stress*

## 🚨 Emergency Procedures (Cognitive Load: LOW)

### System Unresponsive
**SYMPTOMS**: Server not responding to health checks, connection refused
**IMMEDIATE ACTION** (2 minutes):
```bash
# 1. Check if process exists
engram status

# 2. If running but unresponsive, force restart
engram stop --force
engram start

# 3. Verify recovery
engram status
curl http://localhost:7432/health/alive
```

### Memory Leak Detected  
**SYMPTOMS**: RAM usage >90%, system slowdown
**IMMEDIATE ACTION** (3 minutes):
```bash
# 1. Check memory usage
ps aux | grep engram
top -p $(pgrep engram)

# 2. Restart service to clear memory
engram stop
engram start

# 3. Monitor for recurrence
watch -n 10 'ps aux | grep engram'
```

### Port Conflict
**SYMPTOMS**: "Address already in use" error on startup  
**IMMEDIATE ACTION** (1 minute):
```bash
# 1. Check what's using the port
lsof -i :7432

# 2. Start on different port
engram start --port 7433

# 3. Update client connections to new port
```

---

## 📋 Common Operations (Cognitive Load: LOW-MEDIUM)

<details>
<summary>Start/Stop/Restart Procedures</summary>

### Starting Engram Server

**CONTEXT**: When to start the server
- System is fully stopped (verify with `engram status`)
- Sufficient resources available (minimum 2GB RAM, 1GB disk)
- No port conflicts on default ports 7432/50051

**ACTION**: Step-by-step startup
```bash
# 1. Verify system is stopped
engram status
# Expected output: "No server running"

# 2. Start server with monitoring
engram start --port 7432 --grpc-port 50051
# Expected: Server starts, shows HTTP and gRPC endpoints

# 3. Wait for initialization (typical: 5-15 seconds)
# - Server binding: Address shows in output
# - Process startup: PID file created
# - API availability: Health endpoint responsive
```

**VERIFICATION**: Confirming successful start
```bash
# 1. Status check passes
engram status
# Expected: "Server online" with PID and port info

# 2. Health endpoint responds
curl http://localhost:7432/api/v1/system/health
# Expected: HTTP 200 with JSON health status

# 3. API endpoints accessible  
curl http://localhost:7432/api/v1/system/introspect
# Expected: System introspection information

# 4. gRPC service available (when implemented)
# grpcurl localhost:50051 list
```

**TROUBLESHOOTING**: If verification fails
- If status shows offline: Check logs for startup errors
- If health check fails: Port may be occupied, try different port
- If connection refused: Firewall or network configuration issue

### Stopping Engram Server

**CONTEXT**: When to stop the server
- Maintenance window scheduled
- System upgrade required
- Resource cleanup needed

**ACTION**: Graceful shutdown
```bash
# 1. Initiate graceful shutdown
engram stop
# Expected: Server begins shutdown process

# 2. Wait for completion (typical: 5-10 seconds)
# - Existing connections complete
# - Resources cleaned up
# - PID file removed
```

**VERIFICATION**: Confirming successful stop
```bash
# 1. Status check shows offline
engram status
# Expected: "No server running"

# 2. Process no longer exists
ps aux | grep engram
# Expected: No engram processes listed

# 3. Port is released
lsof -i :7432
# Expected: No processes using port
```

**FORCE SHUTDOWN**: If graceful stop fails
```bash
# Only use if graceful shutdown hangs >30 seconds
engram stop --force

# Verify forced shutdown worked
engram status
```

### Server Health Monitoring

**CONTEXT**: Ongoing health assessment
- Regular health checks every 5-10 minutes
- Before making configuration changes
- After any system modifications

**ACTION**: Health check procedure
```bash
# 1. Basic status check
engram status
# Shows: online/offline, PID, ports

# 2. Detailed health with JSON output
engram status --json
# Shows: detailed status, health endpoints, response times

# 3. Continuous monitoring
engram status --watch
# Updates every 5 seconds (Ctrl+C to exit)
```

**VERIFICATION**: Healthy system indicators
- Status: "online"  
- Health: "responsive"
- Response time: <100ms
- Memory usage: <80% of available
- No error messages in status output

</details>

<details>
<summary>Memory Operations</summary>

### Creating Memories

**CONTEXT**: Adding new information to the system
- Server is running and healthy
- Content is ready for storage
- Confidence level determined (if applicable)

**ACTION**: Memory creation
```bash
# 1. Basic memory creation
engram memory create "This is a test memory"
# Expected: Memory ID returned

# 2. Memory with confidence level
engram memory create "Important fact" --confidence 0.9
# Expected: Memory created with specified confidence

# 3. Verify creation
engram memory list --limit 1
# Expected: New memory appears in list
```

### Searching Memories

**CONTEXT**: Retrieving stored information
- Query terms identified
- Search scope determined
- Result limits set appropriately

**ACTION**: Memory search
```bash
# 1. Basic text search
engram memory search "test query"
# Expected: Relevant memories returned

# 2. Limited search results
engram memory search "pattern" --limit 5
# Expected: Maximum 5 results returned

# 3. Get specific memory by ID
engram memory get "memory-id-here"
# Expected: Full memory details displayed
```

### Memory Maintenance

**CONTEXT**: Routine memory management
- System cleanup required
- Storage optimization needed
- Data consistency verification

**ACTION**: Maintenance operations
```bash
# 1. List all memories with pagination
engram memory list --limit 10 --offset 0
# Expected: First 10 memories displayed

# 2. Delete specific memory (use with caution)
engram memory delete "memory-id-to-remove"
# Expected: Memory removed from system

# 3. Verify deletion
engram memory get "memory-id-to-remove"
# Expected: "Memory not found" error
```

</details>

<details>
<summary>Configuration Management</summary>

### Viewing Configuration

**CONTEXT**: Understanding current system settings
- Troubleshooting configuration issues
- Preparing for changes
- Documentation requirements

**ACTION**: Configuration inspection
```bash
# 1. Show all configuration
engram config list
# Expected: All settings with current values

# 2. Show specific section
engram config list --section network
# Expected: Network-related settings only

# 3. Get specific setting
engram config get network.port
# Expected: Current port value
```

### Modifying Configuration

**CONTEXT**: Changing system behavior
- Performance tuning required
- Port changes needed
- Resource limits adjustment

**ACTION**: Configuration changes
```bash
# 1. Set configuration value
engram config set memory.cache_size "200MB"
# Expected: Setting updated

# 2. Verify change took effect
engram config get memory.cache_size
# Expected: New value displayed

# 3. Restart server to apply changes
engram stop
engram start
```

**VERIFICATION**: Configuration change success
- New value returned by get command
- Server starts successfully with new configuration
- System behavior reflects changes
- No errors in startup logs

</details>

---

## 🔧 Maintenance Tasks (Cognitive Load: MEDIUM)

<details>
<summary>Backup and Recovery</summary>

### Creating System Backup

**CONTEXT**: Data protection and recovery preparation
- Before major system changes
- Regular backup schedule (daily/weekly)
- Before software updates

**ACTION**: Backup procedure
```bash
#!/bin/bash
# backup-engram.sh - Executable backup procedure

set -euo pipefail

echo "=== Engram Backup Procedure ==="
echo "Context: Creating verified backup of memory system"

# 1. Verify system health
echo "Checking system health..."
if ! engram status | grep -q "online"; then
    echo "ERROR: System offline. Cannot backup stopped system."
    exit 1
fi

echo "✓ System online and ready for backup"

# 2. Create backup directory with timestamp
BACKUP_DIR="backup-$(date +%Y%m%d_%H%M%S)"
mkdir -p "$BACKUP_DIR"

echo "Creating backup in: $BACKUP_DIR"

# 3. Export system data (when backup functionality is implemented)
echo "⚠️  Full backup functionality not yet implemented"
echo "Manual backup steps:"
echo "1. Stop server: engram stop"
echo "2. Copy data directory if it exists"
echo "3. Copy configuration files"
echo "4. Restart server: engram start"

# 4. Validate backup (when implemented)
echo "✓ Backup procedure template ready"
echo "SUCCESS: Backup framework created at $BACKUP_DIR"
```

### Disaster Recovery

**CONTEXT**: System failure requiring data restoration
- Data corruption detected
- Hardware failure recovery
- Complete system rebuild

**ACTION**: Recovery procedure
```bash
#!/bin/bash  
# restore-engram.sh - Disaster recovery procedure

set -euo pipefail

BACKUP_DIR="$1"

echo "=== Engram Recovery Procedure ==="
echo "Context: Restoring system from backup: $BACKUP_DIR"

# 1. Verify backup exists
if [ ! -d "$BACKUP_DIR" ]; then
    echo "ERROR: Backup directory $BACKUP_DIR not found"
    exit 1
fi

echo "✓ Backup directory verified"

# 2. Stop any running instances
echo "Stopping any running instances..."
engram stop || true

# 3. Restore from backup (when implemented)
echo "⚠️  Restoration functionality not yet implemented"
echo "Manual recovery steps:"
echo "1. Restore data directory from backup"
echo "2. Restore configuration files"  
echo "3. Start server: engram start"
echo "4. Verify system health: engram status"

echo "SUCCESS: Recovery procedure template ready"
```

</details>

<details>
<summary>Performance Monitoring</summary>

### System Performance Assessment

**CONTEXT**: Evaluating system performance
- Regular performance reviews
- Before capacity planning
- Investigating slowdowns

**ACTION**: Performance monitoring
```bash
# 1. Resource utilization
top -p $(pgrep engram)
# Monitor: CPU usage, memory consumption

# 2. System status with timing
time engram status
# Expected: Status response <1 second

# 3. Memory operation performance
time engram memory create "performance test"
# Expected: Creation response <100ms

# 4. Search performance
time engram memory search "test"
# Expected: Search response <500ms
```

**VERIFICATION**: Performance within bounds
- CPU usage: <50% during normal operations
- Memory usage: <80% of available RAM
- Response times: Status <1s, Operations <500ms
- No timeout errors in operations

</details>

---

## 🎯 Advanced Operations (Cognitive Load: HIGH)

<details>
<summary>Troubleshooting Guide</summary>

### Decision Tree: Server Won't Start

```
SYMPTOM: "engram start" fails or hangs

Q1: Is port 7432 available?
└─ NO → Check what's using port: lsof -i :7432
    └─ Kill conflicting process or use different port
└─ YES → Continue to Q2

Q2: Are system resources sufficient?
└─ NO → Free memory/disk space or allocate more resources  
└─ YES → Continue to Q3

Q3: Do permissions allow server startup?
└─ NO → Check file/directory permissions, run as appropriate user
└─ YES → Check logs for specific error messages

ESCALATION: If all checks pass but startup fails:
- Check system logs: journalctl -u engram
- Review configuration files
- Contact system administrator
```

### Decision Tree: Slow Response Times  

```
SYMPTOM: Operations taking >1 second to complete

Q1: Is CPU usage >80%?
└─ YES → System overloaded
    └─ Reduce concurrent operations
    └─ Check for competing processes
└─ NO → Continue to Q2

Q2: Is memory usage >90%?  
└─ YES → Memory pressure
    └─ Restart server to clear memory
    └─ Consider memory configuration tuning
└─ NO → Continue to Q3

Q3: Are there many memories stored?
└─ YES → Performance may degrade with scale
    └─ Consider optimization strategies
    └─ Monitor memory count growth
└─ NO → Investigate network connectivity

ESCALATION: If performance doesn't improve:
- Enable detailed logging
- Run benchmark tests
- Review system configuration
```

### Decision Tree: Memory Operations Failing

```  
SYMPTOM: Memory create/search/get operations return errors

Q1: Is server running and responsive?
└─ NO → Start server: engram start
    └─ Verify health: engram status
└─ YES → Continue to Q2

Q2: Are API endpoints accessible?
└─ NO → Check firewall, network configuration
    └─ Verify HTTP API on port 7432
└─ YES → Continue to Q3

Q3: Is the operation syntax correct?
└─ NO → Check command help: engram memory --help  
    └─ Verify required parameters provided
└─ YES → Check server logs for specific errors

ESCALATION: If operations still fail:
- Check server resource usage
- Review error logs in detail
- Test with minimal examples
```

</details>

<details>
<summary>Advanced Configuration</summary>

### Performance Tuning

**CONTEXT**: Optimizing system performance  
- Response times need improvement
- Memory usage optimization required
- Capacity planning implementation

**ACTION**: Systematic tuning approach
```bash
# 1. Baseline measurement
echo "Recording baseline performance..."
time engram memory create "baseline test"
time engram memory search "baseline"  
ps aux | grep engram # Note memory usage

# 2. Configuration assessment  
echo "Current configuration:"
engram config list

# 3. Performance configuration (when implemented)
echo "⚠️  Performance tuning not yet fully implemented"
echo "Future tuning parameters:"
echo "- memory.cache_size: Memory allocated for caching"  
echo "- memory.gc_threshold: Garbage collection trigger"
echo "- network.timeout: Request timeout settings"

# 4. Post-tuning verification
echo "After configuration changes, verify:"
echo "1. Server restarts successfully"
echo "2. Performance metrics improved"  
echo "3. No errors in operation"
```

### System Optimization

**CONTEXT**: Long-term system health
- Regular maintenance schedule
- Performance degradation prevention  
- Resource utilization optimization

**ACTION**: Optimization routine
```bash
#!/bin/bash
# optimize-engram.sh - System optimization routine

echo "=== Engram System Optimization ==="

# 1. Health assessment
echo "1. Checking system health..."
engram status

# 2. Resource utilization review
echo "2. Reviewing resource usage..."
ps aux | grep engram
df -h # Check disk space

# 3. Performance testing
echo "3. Testing performance..."
time engram memory create "optimization test $(date)"
time engram memory search "optimization"

# 4. Cleanup operations (when implemented)
echo "4. Future optimization features:"
echo "- Memory garbage collection"
echo "- Index optimization"
echo "- Connection pool tuning"

echo "✓ Optimization review completed"
```

</details>

---

## 📊 Monitoring and Alerting

### Continuous Health Monitoring

**Setup monitoring for production deployment**:

```bash
#!/bin/bash
# monitor-engram.sh - Continuous monitoring script

while true; do
    # Check server status
    if ! engram status --json | jq -r '.status' | grep -q "online"; then
        echo "ALERT: Engram server offline at $(date)"
        # Add alerting mechanism here (email, Slack, etc.)
    fi
    
    # Check health endpoint response time
    start_time=$(date +%s%N)
    curl -s http://localhost:7432/api/v1/system/health >/dev/null 2>&1
    end_time=$(date +%s%N)
    response_time=$(( (end_time - start_time) / 1000000 )) # Convert to ms
    
    if [ $response_time -gt 1000 ]; then
        echo "ALERT: Slow response time ${response_time}ms at $(date)"
    fi
    
    # Check memory usage
    memory_percent=$(ps aux | grep engram | awk '{print $4}' | head -1)
    if [ "$(echo "$memory_percent > 80" | bc -l)" == "1" ]; then
        echo "ALERT: High memory usage ${memory_percent}% at $(date)"
    fi
    
    sleep 300 # Check every 5 minutes
done
```

### Production Metrics

**Key metrics to monitor**:
- **Server Status**: online/offline state  
- **Response Time**: API endpoint response latency
- **Memory Usage**: Process memory consumption
- **Error Rate**: Failed operations percentage
- **Memory Count**: Number of stored memories

**Alert Thresholds**:
- Response time >1000ms
- Memory usage >80%  
- Server offline >30 seconds
- Error rate >5%

---

## 🔥 Incident Response Playbooks

### Playbook: Complete System Failure

**RECOGNITION PATTERNS**:
- Server status shows offline
- Health checks fail completely
- No response from any endpoints

**IMMEDIATE ACTIONS** (5 minutes):
```bash
# 1. Confirm system state
engram status
ps aux | grep engram

# 2. Check system resources
free -h
df -h  

# 3. Attempt restart
engram stop --force
engram start

# 4. Verify recovery
engram status
curl http://localhost:7432/health/alive
```

**SUCCESS CRITERIA**:
- Server status returns to online
- Health endpoints respond
- Basic operations work
- No error messages in logs

**ESCALATION**: If recovery fails after 10 minutes:
- Investigate system logs
- Check for hardware issues  
- Consider disaster recovery procedures
- Contact technical lead

### Playbook: Performance Degradation

**RECOGNITION PATTERNS**:
- Response times >1 second consistently
- User complaints about slowness
- Operations timing out

**IMMEDIATE ACTIONS** (10 minutes):
```bash
# 1. Resource assessment
top -p $(pgrep engram)
free -h

# 2. Quick performance test
time engram memory create "perf test $(date)"
time engram memory search "test"

# 3. If memory usage high, restart
if [ $(ps aux | grep engram | awk '{print $4}' | head -1 | cut -d. -f1) -gt 80 ]; then
    engram stop
    engram start
fi

# 4. Monitor improvement
watch -n 10 'time engram status'
```

**SUCCESS CRITERIA**:
- Response times <500ms
- Memory usage <80%
- Operations complete without timeout
- User satisfaction restored

---

## 📚 Reference Information

### Command Reference

| Command | Purpose | Usage |
|---------|---------|-------|
| `engram start` | Start server | `engram start [--port PORT]` |
| `engram stop` | Stop server | `engram stop [--force]` |  
| `engram status` | Check status | `engram status [--json] [--watch]` |
| `engram memory create` | Create memory | `engram memory create "content"` |
| `engram memory search` | Search memories | `engram memory search "query"` |
| `engram memory get` | Get by ID | `engram memory get ID` |
| `engram memory list` | List memories | `engram memory list [--limit N]` |
| `engram memory delete` | Delete memory | `engram memory delete ID` |
| `engram config list` | Show config | `engram config list [--section NAME]` |
| `engram config get` | Get setting | `engram config get KEY` |
| `engram config set` | Set setting | `engram config set KEY VALUE` |

### Default Configuration

| Setting | Default Value | Description |
|---------|---------------|-------------|
| Network Port | 7432 | HTTP API port |
| gRPC Port | 50051 | gRPC service port |  
| Log Level | info | Logging verbosity |
| Memory Cache | 100MB | Memory allocation for caching |
| GC Threshold | 0.7 | Garbage collection trigger |

### File Locations

| File | Location | Purpose |
|------|----------|---------|
| PID File | `~/.engram/engram.pid` | Process tracking |
| State File | `~/.engram/state.json` | System state |
| Config File | `~/.engram/config.toml` | Configuration |
| Log Files | `~/.engram/logs/` | Application logs |

### Exit Codes

| Code | Meaning |
|------|---------|
| 0 | Success |
| 1 | General error |
| 2 | Configuration error |  
| 3 | Connection failed |
| 4 | Operation timeout |

---

*This operational guide follows cognitive ergonomics principles for stress-resistant documentation. Each procedure uses Context-Action-Verification (CAV) format and progressive disclosure to minimize cognitive load during incidents.*

*Version: 1.0 | Last Updated: $(date +%Y-%m-%d)*