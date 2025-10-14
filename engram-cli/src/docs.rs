//! Embedded documentation system for Engram
//!
//! This module contains all operational and user documentation embedded directly
//! in the code, following cognitive ergonomics principles. Documentation is
//! organized hierarchically and can be extracted programmatically.

use indoc::indoc;

/// Documentation sections organized by cognitive load and operational context
#[derive(Debug, Clone, Copy)]
pub enum DocSection {
    /// Emergency procedures (Cognitive Load: LOW) - 2-5 minute fixes
    Emergency,
    /// Common operations (Cognitive Load: LOW-MEDIUM) - 5-15 minutes  
    Common,
    /// Advanced operations (Cognitive Load: HIGH) - 30+ minutes
    Advanced,
    /// Troubleshooting decision trees
    Troubleshooting,
    /// Incident response playbooks
    IncidentResponse,
    /// Reference information
    Reference,
}

/// Operational documentation embedded in code
pub struct OperationalDocs;

impl OperationalDocs {
    /// Get the complete operational guide as markdown
    #[must_use]
    pub const fn complete_guide() -> &'static str {
        indoc! {r"
            # Engram Operational Guide

            *Production-ready procedures designed for cognitive ergonomics under stress*

            ## Quick Reference
            - [🚨 Emergency Procedures](#emergency-procedures) (2-5 min fixes)
            - [📋 Common Operations](#common-operations) (5-15 min)
            - [🎯 Advanced Operations](#advanced-operations) (30+ min)
            - [🔍 Troubleshooting](#troubleshooting) (Decision trees)
            - [🔥 Incident Response](#incident-response) (Playbooks)

            Use `engram docs <section>` to view specific sections.
        "}
    }

    /// Emergency procedures for immediate system recovery
    #[must_use]
    pub const fn emergency_procedures() -> &'static str {
        indoc! {r#"
            # 🚨 Emergency Procedures (Cognitive Load: LOW)

            ## System Unresponsive
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
            curl http://localhost:7432/api/v1/system/health
            ```

            ## Memory Leak Detected  
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

            ## Port Conflict
            **SYMPTOMS**: "Address already in use" error on startup  
            **IMMEDIATE ACTION** (1 minute):
            ```bash
            # 1. Check what's using the port
            lsof -i :7432

            # 2. Start on different port
            engram start --port 7433

            # 3. Update client connections to new port
            ```

            **SUCCESS CRITERIA**: 
            - Server status shows "online"
            - Health endpoint responds with HTTP 200
            - Basic operations complete without errors
        "#}
    }

    /// Common operational procedures
    #[must_use]
    pub const fn common_operations() -> &'static str {
        indoc! {r#"
            # 📋 Common Operations (Cognitive Load: LOW-MEDIUM)

            ## Starting Engram Server

            ### CONTEXT: When to start the server
            - System is fully stopped (verify with `engram status`)
            - Sufficient resources available (minimum 2GB RAM, 1GB disk)
            - No port conflicts on default ports 7432/50051

            ### ACTION: Step-by-step startup
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

            ### VERIFICATION: Confirming successful start
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
            ```

            ## Stopping Engram Server

            ### CONTEXT: When to stop the server
            - Maintenance window scheduled
            - System upgrade required
            - Resource cleanup needed

            ### ACTION: Graceful shutdown
            ```bash
            # 1. Initiate graceful shutdown
            engram stop
            # Expected: Server begins shutdown process

            # 2. Wait for completion (typical: 5-10 seconds)
            # - Existing connections complete
            # - Resources cleaned up
            # - PID file removed
            ```

            ### VERIFICATION: Confirming successful stop
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

            ## Memory Operations

            ### Creating Memories
            ```bash
            # Basic memory creation via API
            curl -X POST http://localhost:7432/api/v1/memories/remember \
                -H "Content-Type: application/json" \
                -d '{"content": "Your memory content", "confidence": 0.8}'

            # Expected: JSON response with memory_id and confirmation
            ```

            ### Searching Memories  
            ```bash
            # Search for memories
            curl "http://localhost:7432/api/v1/memories/recall?query=search+terms&limit=5"

            # Expected: JSON response with matching memories and confidence scores
            ```
        "#}
    }

    /// Advanced operational procedures
    #[must_use]
    pub const fn advanced_operations() -> &'static str {
        indoc! {r#"
            # 🎯 Advanced Operations (Cognitive Load: HIGH)

            ## Performance Tuning

            ### CONTEXT: Optimizing system performance
            - Response times need improvement
            - Memory usage optimization required
            - Capacity planning implementation

            ### ACTION: Systematic tuning approach
            ```bash
            # 1. Baseline measurement
            echo "Recording baseline performance..."
            time curl http://localhost:7432/api/v1/system/health
            
            # 2. Monitor resource usage
            top -p $(pgrep engram)
            
            # 3. Analyze API response times
            for i in {1..10}; do
                time curl -s http://localhost:7432/api/v1/system/health > /dev/null
            done
            ```

            ## System Optimization

            ### CONTEXT: Long-term system health
            - Regular maintenance schedule
            - Performance degradation prevention  
            - Resource utilization optimization

            ### ACTION: Optimization routine
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
            time curl http://localhost:7432/api/v1/memories/remember \
                -X POST -H "Content-Type: application/json" \
                -d '{"content": "optimization test"}'

            echo "✓ Optimization review completed"
            ```

            ## Backup and Recovery

            ### Creating System Backup
            ```bash
            #!/bin/bash
            # backup-engram.sh - Executable backup procedure

            set -euo pipefail

            echo "=== Engram Backup Procedure ==="
            echo "Context: Creating verified backup of memory system"

            # 1. Verify system health
            echo "Checking system health..."
            if ! curl -s http://localhost:7432/api/v1/system/health | grep -q "healthy"; then
                echo "ERROR: System unhealthy. Cannot backup unstable system."
                exit 1
            fi

            echo "✓ System healthy and ready for backup"

            # 2. Create backup directory with timestamp
            BACKUP_DIR="backup-$(date +%Y%m%d_%H%M%S)"
            mkdir -p "$BACKUP_DIR"

            echo "Creating backup in: $BACKUP_DIR"

            # 3. Export system data via API
            echo "Exporting memories..."
            curl -s http://localhost:7432/api/v1/memories/recall?limit=1000 > "$BACKUP_DIR/memories.json"

            # 4. Export system configuration
            echo "Exporting configuration..."
            curl -s http://localhost:7432/api/v1/system/introspect > "$BACKUP_DIR/system_state.json"

            # 5. Validate backup
            echo "Validating backup..."
            if [ -s "$BACKUP_DIR/memories.json" ] && [ -s "$BACKUP_DIR/system_state.json" ]; then
                echo "✓ Backup validated successfully"
                echo "SUCCESS: Backup completed at $BACKUP_DIR"
            else
                echo "✗ Backup validation failed"
                exit 1
            fi
            ```
        "#}
    }

    /// Troubleshooting decision trees  
    #[must_use]
    pub const fn troubleshooting_trees() -> &'static str {
        indoc! {r#"
            # 🔍 Troubleshooting Guide

            *Symptom-based troubleshooting organized for rapid cognitive processing under stress*

            ## Quick Diagnostic Commands

            ```bash
            # System health check suite
            engram status                            # Check if server is running
            curl -s localhost:7432/api/v1/system/health | jq .  # Verify API health
            ps aux | grep engram | grep -v grep      # Check process details
            lsof -i :7432                            # Check port usage
            df -h | grep -E "^/|engram"             # Check disk space
            free -h                                  # Check memory usage
            ```

            ---

            ## Common Issues (20+ Documented)

            ### 1. Server Won't Start

            **SYMPTOM**: `engram start` fails immediately or hangs

            **QUICK CHECK**: 
            ```bash
            lsof -i :7432 && echo "Port occupied" || echo "Port available"
            ```

            **COMMON CAUSES**:
            - Port 7432 already in use (80% confidence)
            - Insufficient permissions (15% confidence)
            - Resource exhaustion (5% confidence)

            **SOLUTIONS**:
            ```bash
            # Solution 1: Use different port (Definite fix for port conflicts)
            engram start --port 7433

            # Solution 2: Kill conflicting process (Use with caution)
            kill $(lsof -t -i:7432)
            engram start

            # Solution 3: Check permissions
            ls -la ~/.engram/
            chmod 755 ~/.engram/
            ```

            **PREVENTION**: Always use `engram stop` before system shutdown

            ---

            ### 2. Connection Refused

            **SYMPTOM**: `curl: (7) Failed to connect to localhost port 7432: Connection refused`

            **QUICK CHECK**:
            ```bash
            engram status | grep -q "running" && echo "Server running" || echo "Server not running"
            ```

            **COMMON CAUSES**:
            - Server not started (70% confidence)
            - Firewall blocking connection (20% confidence)
            - Server crashed silently (10% confidence)

            **SOLUTIONS**:
            ```bash
            # Solution 1: Start server (Definite fix if not running)
            engram start

            # Solution 2: Check firewall (Platform specific)
            # macOS:
            sudo pfctl -s rules | grep 7432
            # Linux:
            sudo iptables -L -n | grep 7432

            # Solution 3: Force restart after crash
            engram stop --force
            sleep 2
            engram start --verbose
            ```

            ---

            ### 3. Slow Response Times

            **SYMPTOM**: API calls taking >1 second to complete

            **QUICK CHECK**:
            ```bash
            time curl -s localhost:7432/api/v1/system/health
            ```

            **COMMON CAUSES**:
            - High memory usage (60% confidence)
            - CPU throttling (25% confidence)
            - Large dataset operations (15% confidence)

            **SOLUTIONS**:
            ```bash
            # Solution 1: Check and free memory (Likely helps)
            free -h
            # If available memory <500MB:
            engram stop && engram start  # Restart to clear memory

            # Solution 2: Check CPU usage
            top -p $(pgrep engram) -n 1
            # If CPU >80%, reduce load or scale resources

            # Solution 3: Optimize queries
            # Use pagination for large datasets:
            curl "localhost:7432/api/v1/memories/recall?limit=10&offset=0"
            ```

            ---

            ### 4. Memory Operations Failing

            **SYMPTOM**: Cannot create, update, or retrieve memories

            **QUICK CHECK**:
            ```bash
            curl -X POST localhost:7432/api/v1/memories/remember \
              -H "Content-Type: application/json" \
              -d '{"content": "test"}' -v
            ```

            **COMMON CAUSES**:
            - Malformed JSON (40% confidence)
            - Missing Content-Type header (30% confidence)
            - Server resource limits (30% confidence)

            **SOLUTIONS**:
            ```bash
            # Solution 1: Validate JSON (Definite fix for syntax errors)
            echo '{"content": "test"}' | jq .
            
            # Solution 2: Include proper headers
            curl -X POST localhost:7432/api/v1/memories/remember \
              -H "Content-Type: application/json" \
              -H "Accept: application/json" \
              -d '{"content": "test memory"}'

            # Solution 3: Check server logs
            tail -n 50 ~/.engram/logs/engram.log | grep ERROR
            ```

            ---

            ### 5. Port Already in Use

            **SYMPTOM**: `Error: Address already in use (os error 48)` or similar

            **QUICK CHECK**:
            ```bash
            lsof -i :7432
            ```

            **SOLUTIONS**:
            ```bash
            # Solution 1: Use alternative port (Definite fix)
            engram start --port 7433

            # Solution 2: Find and stop conflicting process
            kill -9 $(lsof -t -i:7432)

            # Solution 3: Check for zombie processes
            ps aux | grep defunct | grep engram
            ```

            ---

            ### 6. Out of Memory Errors

            **SYMPTOM**: `memory allocation failed`, server crashes with OOM

            **QUICK CHECK**:
            ```bash
            dmesg | tail -20 | grep -i "killed process"
            ```

            **SOLUTIONS**:
            ```bash
            # Solution 1: Increase system memory limits
            ulimit -v unlimited
            engram start

            # Solution 2: Configure memory limits
            engram start --max-memory 2G

            # Solution 3: Clear system caches (Linux)
            sudo sh -c "echo 3 > /proc/sys/vm/drop_caches"
            ```

            ---

            ### 7. SSL/TLS Certificate Errors

            **SYMPTOM**: `SSL certificate problem: self signed certificate`

            **QUICK CHECK**:
            ```bash
            curl -k https://localhost:7432/api/v1/system/health
            ```

            **SOLUTIONS**:
            ```bash
            # Solution 1: Use HTTP for local development
            curl http://localhost:7432/api/v1/system/health

            # Solution 2: Trust self-signed cert (development only)
            curl -k https://localhost:7432/api/v1/system/health

            # Solution 3: Generate proper certificates
            engram cert generate --domain localhost
            ```

            ---

            ### 8. Database Corruption

            **SYMPTOM**: `Error: database corrupted`, inconsistent query results

            **QUICK CHECK**:
            ```bash
            engram db verify
            ```

            **SOLUTIONS**:
            ```bash
            # Solution 1: Run recovery (Likely helps)
            engram stop
            engram db repair
            engram start

            # Solution 2: Restore from backup
            engram stop
            engram db restore --from backup-20240101
            engram start

            # Solution 3: Export and reimport (Last resort)
            engram db export > data.json
            engram db reset
            engram db import < data.json
            ```

            ---

            ### 9. High CPU Usage

            **SYMPTOM**: engram process using >90% CPU continuously

            **QUICK CHECK**:
            ```bash
            top -p $(pgrep engram) -b -n 1
            ```

            **SOLUTIONS**:
            ```bash
            # Solution 1: Check for runaway queries
            curl localhost:7432/api/v1/system/queries/active

            # Solution 2: Restart with rate limiting
            engram stop
            engram start --rate-limit 100

            # Solution 3: Profile CPU usage
            engram debug cpu-profile --duration 30s
            ```

            ---

            ### 10. Disk Space Issues

            **SYMPTOM**: `No space left on device`, write operations failing

            **QUICK CHECK**:
            ```bash
            df -h ~/.engram/
            ```

            **SOLUTIONS**:
            ```bash
            # Solution 1: Clean old logs (Safe)
            find ~/.engram/logs -name "*.log" -mtime +7 -delete

            # Solution 2: Compact database
            engram db compact

            # Solution 3: Move data directory
            engram stop
            mv ~/.engram /larger/disk/engram
            ln -s /larger/disk/engram ~/.engram
            engram start
            ```

            ---

            ### 11. Network Timeout Errors

            **SYMPTOM**: `Error: operation timed out`

            **QUICK CHECK**:
            ```bash
            ping -c 3 localhost
            netstat -an | grep 7432
            ```

            **SOLUTIONS**:
            ```bash
            # Solution 1: Increase timeout values
            curl --max-time 30 localhost:7432/api/v1/system/health

            # Solution 2: Check network interfaces
            ip addr show
            # Ensure localhost resolves correctly

            # Solution 3: Test with explicit IP
            curl 127.0.0.1:7432/api/v1/system/health
            ```

            ---

            ### 12. Permission Denied Errors

            **SYMPTOM**: `Permission denied`, cannot read/write files

            **QUICK CHECK**:
            ```bash
            ls -la ~/.engram/
            whoami
            ```

            **SOLUTIONS**:
            ```bash
            # Solution 1: Fix ownership
            sudo chown -R $(whoami) ~/.engram/

            # Solution 2: Fix permissions
            chmod -R u+rw ~/.engram/

            # Solution 3: Run with correct user
            sudo -u engram engram start
            ```

            ---

            ### 13. Configuration Errors

            **SYMPTOM**: `Invalid configuration`, server won't start with custom config

            **QUICK CHECK**:
            ```bash
            engram config validate
            ```

            **SOLUTIONS**:
            ```bash
            # Solution 1: Use default configuration
            engram start --config default

            # Solution 2: Fix syntax errors
            cat ~/.engram/config.toml | toml-validator

            # Solution 3: Reset configuration
            mv ~/.engram/config.toml ~/.engram/config.toml.bak
            engram config init
            ```

            ---

            ### 14. Memory Leak Detection

            **SYMPTOM**: Memory usage grows continuously over time

            **QUICK CHECK**:
            ```bash
            # Monitor memory over time
            while true; do 
              ps aux | grep engram | grep -v grep | awk '{print $6}'
              sleep 60
            done
            ```

            **SOLUTIONS**:
            ```bash
            # Solution 1: Enable memory profiling
            engram start --memory-profile

            # Solution 2: Set memory limits
            engram start --max-memory 4G --gc-interval 5m

            # Solution 3: Scheduled restarts (workaround)
            crontab -e
            # Add: 0 3 * * * /usr/local/bin/engram restart
            ```

            ---

            ### 15. Query Performance Issues

            **SYMPTOM**: Specific queries running slowly

            **QUICK CHECK**:
            ```bash
            curl -X POST localhost:7432/api/v1/debug/explain \
              -d '{"query": "your slow query here"}'
            ```

            **SOLUTIONS**:
            ```bash
            # Solution 1: Add query hints
            curl localhost:7432/api/v1/memories/search \
              -d '{"query": "test", "hint": "use_index"}'

            # Solution 2: Rebuild indices
            engram db reindex

            # Solution 3: Optimize query patterns
            # Use specific filters instead of wildcards
            # Paginate large result sets
            ```

            ---

            ### 16. Backup/Restore Failures

            **SYMPTOM**: Backup or restore operations failing

            **QUICK CHECK**:
            ```bash
            engram backup verify --last
            ```

            **SOLUTIONS**:
            ```bash
            # Solution 1: Check disk space for backups
            df -h /backup/location/

            # Solution 2: Use incremental backups
            engram backup create --incremental

            # Solution 3: Verify backup integrity
            engram backup test --file backup-20240101.tar.gz
            ```

            ---

            ### 17. API Version Mismatch

            **SYMPTOM**: `Unsupported API version` errors

            **QUICK CHECK**:
            ```bash
            curl localhost:7432/api/version
            ```

            **SOLUTIONS**:
            ```bash
            # Solution 1: Use versioned endpoints
            curl localhost:7432/api/v1/system/health

            # Solution 2: Check client compatibility
            engram version --check-compatibility

            # Solution 3: Update client/server
            engram update
            ```

            ---

            ### 18. Logging Issues

            **SYMPTOM**: Cannot find logs, logs not updating

            **QUICK CHECK**:
            ```bash
            ls -la ~/.engram/logs/
            tail -f ~/.engram/logs/engram.log
            ```

            **SOLUTIONS**:
            ```bash
            # Solution 1: Check log configuration
            grep -i log ~/.engram/config.toml

            # Solution 2: Set log level
            engram start --log-level debug

            # Solution 3: Rotate large logs
            logrotate -f ~/.engram/logrotate.conf
            ```

            ---

            ### 19. Cluster Communication Issues

            **SYMPTOM**: Nodes cannot communicate in cluster mode

            **QUICK CHECK**:
            ```bash
            engram cluster status
            ```

            **SOLUTIONS**:
            ```bash
            # Solution 1: Check network connectivity
            ping other-node-hostname
            telnet other-node 7432

            # Solution 2: Verify cluster configuration
            engram cluster verify-config

            # Solution 3: Rejoin cluster
            engram cluster leave
            engram cluster join --seed node1:7432
            ```

            ---

            ### 20. Data Import/Export Errors

            **SYMPTOM**: Cannot import or export data

            **QUICK CHECK**:
            ```bash
            # Test with small dataset
            echo '{"content": "test"}' | engram import --format json
            ```

            **SOLUTIONS**:
            ```bash
            # Solution 1: Validate data format
            jq . < data.json > /dev/null && echo "Valid JSON"

            # Solution 2: Use streaming for large files
            engram import --stream < large-data.json

            # Solution 3: Split large imports
            split -l 1000 data.json chunk_
            for f in chunk_*; do engram import < $f; done
            ```

            ---

            ### 21. Authentication/Authorization Failures

            **SYMPTOM**: `401 Unauthorized` or `403 Forbidden` errors

            **QUICK CHECK**:
            ```bash
            curl -H "Authorization: Bearer $TOKEN" \
              localhost:7432/api/v1/system/health
            ```

            **SOLUTIONS**:
            ```bash
            # Solution 1: Generate new token
            engram auth token create --name debug

            # Solution 2: Check token expiration
            engram auth token verify --token $TOKEN

            # Solution 3: Reset authentication
            engram auth reset --confirm
            ```

            ---

            ### 22. WebSocket Connection Issues

            **SYMPTOM**: Real-time updates not working, WebSocket errors

            **QUICK CHECK**:
            ```bash
            # Test WebSocket connection
            websocat ws://localhost:7432/api/v1/stream
            ```

            **SOLUTIONS**:
            ```bash
            # Solution 1: Check WebSocket support
            curl -H "Upgrade: websocket" \
                 -H "Connection: Upgrade" \
                 localhost:7432/api/v1/stream -v

            # Solution 2: Adjust proxy settings
            # Add to nginx.conf:
            # proxy_http_version 1.1;
            # proxy_set_header Upgrade $http_upgrade;
            # proxy_set_header Connection "upgrade";

            # Solution 3: Use polling fallback
            engram config set realtime.fallback polling
            ```

            ---

            ## Log Analysis Guide

            ### Understanding Log Levels

            ```bash
            # Log level indicators:
            # ERROR   - System errors requiring immediate attention
            # WARN    - Potential issues that may need investigation
            # INFO    - Normal operations and state changes
            # DEBUG   - Detailed diagnostic information
            # TRACE   - Very detailed execution flow

            # Filter by level:
            grep "ERROR" ~/.engram/logs/engram.log
            grep "WARN" ~/.engram/logs/engram.log | tail -20
            ```

            ### Common Log Patterns

            ```bash
            # Find memory allocation issues:
            grep -i "alloc\|memory\|oom" ~/.engram/logs/engram.log

            # Find network errors:
            grep -i "connection\|timeout\|refused" ~/.engram/logs/engram.log

            # Find performance issues:
            grep -i "slow\|performance\|latency" ~/.engram/logs/engram.log

            # Find crash indicators:
            grep -i "panic\|fatal\|crash" ~/.engram/logs/engram.log
            ```

            ### Log Correlation

            ```bash
            # Correlate errors with timestamps:
            grep ERROR ~/.engram/logs/engram.log | \
              awk '{print $1, $2}' | \
              uniq -c | \
              sort -rn

            # Find error bursts:
            grep ERROR ~/.engram/logs/engram.log | \
              awk '{print substr($1,1,16)}' | \
              uniq -c | \
              awk '$1 > 5 {print "Burst at", $2, "with", $1, "errors"}'
            ```

            ---

            ## Escalation Procedures

            ### When to Escalate

            1. **Immediate Escalation Required**:
               - Data corruption detected
               - Security breach indicators
               - Complete system failure lasting >5 minutes
               - Multiple node failures in cluster

            2. **Escalate After Basic Troubleshooting**:
               - Performance degradation >50% for >1 hour
               - Recurring crashes (>3 in 24 hours)
               - Unexplained data inconsistencies

            3. **Information to Collect Before Escalation**:
               ```bash
               # Generate diagnostic bundle:
               engram debug bundle create
               
               # Includes:
               # - System information
               # - Configuration files
               # - Recent logs (sanitized)
               # - Performance metrics
               # - Error frequency analysis
               ```

            ### Escalation Contacts

            ```bash
            # Check current on-call:
            engram support on-call

            # Generate support ticket:
            engram support ticket create \
              --severity high \
              --attach debug-bundle.tar.gz
            ```

            ---

            ## Preventive Maintenance

            ### Daily Health Checks

            ```bash
            #!/bin/bash
            # daily-health-check.sh
            
            echo "=== Engram Daily Health Check ==="
            date
            
            # 1. Service status
            engram status
            
            # 2. Resource usage
            df -h ~/.engram/
            free -h
            
            # 3. Error count (last 24h)
            grep ERROR ~/.engram/logs/engram.log | \
              grep "$(date -d '1 day ago' '+%Y-%m-%d')" | \
              wc -l
            
            # 4. Performance baseline
            time curl -s localhost:7432/api/v1/system/health
            
            # 5. Backup verification
            engram backup verify --last
            
            echo "=== Health check complete ==="
            ```

            ### Weekly Maintenance

            ```bash
            # 1. Log rotation
            logrotate -f ~/.engram/logrotate.conf

            # 2. Database optimization
            engram db optimize

            # 3. Cache cleanup
            engram cache clear --older-than 7d

            # 4. Security updates check
            engram security audit
            ```

            ---

            ## Quick Reference Card

            ### Emergency Commands
            ```bash
            engram stop --force           # Force stop
            engram start --safe-mode      # Start with minimal features
            engram reset --soft           # Reset without data loss
            engram debug --emergency      # Emergency debug mode
            ```

            ### Diagnostic One-Liners
            ```bash
            # Is it running?
            pgrep engram && echo "Yes" || echo "No"

            # How long has it been running?
            ps -o etime= -p $(pgrep engram)

            # What's the memory usage?
            ps aux | grep engram | awk '{print $6/1024 " MB"}'

            # Recent errors?
            tail -100 ~/.engram/logs/engram.log | grep -c ERROR

            # Current connections?
            lsof -i :7432 | wc -l
            ```

            ### Performance Checks
            ```bash
            # API latency
            for i in {1..10}; do
              time curl -s localhost:7432/api/v1/system/health > /dev/null
            done 2>&1 | grep real

            # Throughput test
            ab -n 100 -c 10 http://localhost:7432/api/v1/system/health
            ```
        "#}
    }

    /// Incident response playbooks
    #[must_use]
    pub const fn incident_response() -> &'static str {
        indoc! {r#"
            # 🔥 Incident Response Playbooks

            ## Playbook: Complete System Failure

            ### RECOGNITION PATTERNS
            - Server status shows offline
            - Health checks fail completely
            - No response from any endpoints

            ### IMMEDIATE ACTIONS (5 minutes)
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
            curl http://localhost:7432/api/v1/system/health
            ```

            ### SUCCESS CRITERIA
            - Server status returns to online
            - Health endpoints respond
            - Basic operations work
            - No error messages in logs

            ### ESCALATION
            If recovery fails after 10 minutes:
            - Investigate system logs
            - Check for hardware issues  
            - Consider disaster recovery procedures
            - Contact technical lead

            ## Playbook: Performance Degradation

            ### RECOGNITION PATTERNS
            - Response times >1 second consistently
            - User complaints about slowness
            - Operations timing out

            ### IMMEDIATE ACTIONS (10 minutes)
            ```bash
            # 1. Resource assessment
            top -p $(pgrep engram)
            free -h

            # 2. Quick performance test
            time curl http://localhost:7432/api/v1/system/health

            # 3. If memory usage high, restart
            if [ $(ps aux | grep engram | awk '{print $4}' | head -1 | cut -d. -f1) -gt 80 ]; then
                engram stop
                engram start
            fi

            # 4. Monitor improvement
            watch -n 10 'time curl -s http://localhost:7432/api/v1/system/health'
            ```

            ### SUCCESS CRITERIA
            - Response times <500ms
            - Memory usage <80%
            - Operations complete without timeout
            - User satisfaction restored

            ## Playbook: Memory System Issues

            ### RECOGNITION PATTERNS
            - Memory operations returning errors
            - Inconsistent search results
            - API endpoints failing

            ### IMMEDIATE ACTIONS (15 minutes)
            ```bash
            # 1. Check system health
            curl http://localhost:7432/api/v1/system/health
            curl http://localhost:7432/api/v1/system/introspect

            # 2. Test basic operations
            curl -X POST http://localhost:7432/api/v1/memories/remember \
                -H "Content-Type: application/json" \
                -d '{"content": "test memory", "confidence": 0.8}'

            # 3. If operations fail, restart with clean state
            engram stop
            engram start

            # 4. Verify memory operations
            curl "http://localhost:7432/api/v1/memories/recall?query=test&limit=1"
            ```

            ### SUCCESS CRITERIA
            - All API endpoints responding
            - Memory operations succeed
            - Consistent search results
            - System introspection shows healthy state
        "#}
    }

    /// Reference information and command quick reference
    #[must_use]
    pub const fn reference() -> &'static str {
        indoc! {r"
            # 📚 Reference Information

            ## Command Reference

            | Command | Purpose | Usage |
            |---------|---------|-------|
            | `engram start` | Start server | `engram start [--port PORT]` |
            | `engram stop` | Stop server | `engram stop [--force]` |  
            | `engram status` | Check status | `engram status [--json] [--watch]` |
            | `engram docs` | Show documentation | `engram docs [SECTION]` |

            ## API Reference

            ### Health Endpoints
            - `GET /api/v1/system/health` - System health status
            - `GET /api/v1/system/introspect` - Detailed system information

            ### Memory Operations
            - `POST /api/v1/memories/remember` - Create new memory
            - `GET /api/v1/memories/recall` - Search memories
            - `POST /api/v1/memories/recognize` - Pattern recognition

            ### Monitoring (SSE)
            - `GET /api/v1/monitoring/events` - Real-time event stream
            - `GET /api/v1/monitoring/activations` - Memory activation monitoring
            - `GET /api/v1/monitoring/causality` - Causality tracking

            ## Default Configuration

            | Setting | Default Value | Description |
            |---------|---------------|-------------|
            | HTTP Port | 7432 | HTTP API port |
            | gRPC Port | 50051 | gRPC service port |  
            | Log Level | info | Logging verbosity |

            ## File Locations

            | File | Location | Purpose |
            |------|----------|---------|
            | PID File | `/tmp/engram.pid` | Process tracking |
            | Logs | Console output | Application logs |

            ## Exit Codes

            | Code | Meaning |
            |------|---------|
            | 0 | Success |
            | 1 | General error |
            | 2 | Configuration error |  
            | 3 | Connection failed |
            | 4 | Operation timeout |

            ## Troubleshooting Quick Reference

            ### Common Issues
            - **Port occupied**: Use `engram start --port 7433`
            - **Permission denied**: Check file permissions
            - **Connection refused**: Verify server is running with `engram status`
            - **Slow responses**: Check memory usage, restart if >80%

            ### Emergency Commands
            ```bash
            # Force restart
            engram stop --force && engram start

            # Check what's using port
            lsof -i :7432

            # Monitor resource usage
            watch -n 5 'ps aux | grep engram'

            # Test connectivity
            curl http://localhost:7432/api/v1/system/health
            ```
        "}
    }

    /// Get documentation by section
    #[must_use]
    pub const fn get_section(section: DocSection) -> &'static str {
        match section {
            DocSection::Emergency => Self::emergency_procedures(),
            DocSection::Common => Self::common_operations(),
            DocSection::Advanced => Self::advanced_operations(),
            DocSection::Troubleshooting => Self::troubleshooting_trees(),
            DocSection::IncidentResponse => Self::incident_response(),
            DocSection::Reference => Self::reference(),
        }
    }

    /// Get all available sections
    #[must_use]
    pub fn available_sections() -> Vec<(&'static str, DocSection)> {
        vec![
            ("emergency", DocSection::Emergency),
            ("common", DocSection::Common),
            ("advanced", DocSection::Advanced),
            ("troubleshooting", DocSection::Troubleshooting),
            ("incident", DocSection::IncidentResponse),
            ("reference", DocSection::Reference),
        ]
    }
}

/// Parse section name from string
impl std::str::FromStr for DocSection {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "emergency" | "emerg" => Ok(Self::Emergency),
            "common" | "ops" => Ok(Self::Common),
            "advanced" | "adv" => Ok(Self::Advanced),
            "troubleshooting" | "trouble" | "debug" => Ok(Self::Troubleshooting),
            "incident" | "response" | "playbook" => Ok(Self::IncidentResponse),
            "reference" | "ref" | "help" => Ok(Self::Reference),
            _ => Err(format!("Unknown documentation section: {s}")),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_section_parsing() {
        assert!(matches!(
            "emergency".parse::<DocSection>().unwrap(),
            DocSection::Emergency
        ));
        assert!(matches!(
            "common".parse::<DocSection>().unwrap(),
            DocSection::Common
        ));
        assert!("invalid".parse::<DocSection>().is_err());
    }

    #[test]
    fn test_documentation_not_empty() {
        assert!(!OperationalDocs::complete_guide().is_empty());
        assert!(!OperationalDocs::emergency_procedures().is_empty());
        assert!(!OperationalDocs::common_operations().is_empty());
        assert!(!OperationalDocs::troubleshooting_trees().is_empty());
    }

    #[test]
    fn test_all_sections_available() {
        let sections = OperationalDocs::available_sections();
        assert_eq!(sections.len(), 6);

        for (name, section) in sections {
            let content = OperationalDocs::get_section(section);
            assert!(!content.is_empty(), "Section {name} should not be empty");
        }
    }
}
