# Task 003: Competitive Benchmark Suite Runner

**Status**: Pending
**Complexity**: Moderate
**Dependencies**: Task 001 (requires scenarios), Task 002 (requires baseline doc)
**Estimated Effort**: 5 hours

## Objective

Create a production-hardened bash script that orchestrates running all competitive scenarios with comprehensive diagnostics, robust error handling, and system resource management. This script must be reliable enough for quarterly automated execution.

## Specifications

Create `scripts/competitive_benchmark_suite.sh` with the following behavior:

### 1. Pre-flight System Checks

Execute comprehensive pre-flight validation before any benchmark execution:

**Binary Verification**:
- Verify `loadtest` binary exists at `target/release/loadtest`
- Verify binary is built in release mode (check with `file` command for optimization flags)
- Verify binary version matches expected (extract from `cargo metadata`)
- Fail fast if binary missing or debug build detected

**Scenario Validation**:
- Verify `scenarios/competitive/` directory exists
- Verify all expected `*.toml` files are present (fail if count != 4)
- Validate each TOML can be parsed (basic syntax check)
- Verify scenario names are unique

**Output Directory**:
- Create `tmp/competitive_benchmarks/` if missing
- Verify write permissions to output directory
- Check disk space available (warn if <10GB free, fail if <5GB)
- Clean up partial results from interrupted previous runs

**System Resource Checks**:
- **RAM**: Check available memory (warn if <16GB total, fail if <8GB available)
- **CPU**: Detect CPU frequency scaling (warn if performance governor not active)
- **Thermal**: Check CPU temperature if sensors available (warn if >80C)
- **Background Load**: Check system load average (warn if >2.0 on 1-min avg)
- **Process Isolation**: Verify no other Engram processes running (`ps aux | grep engram`)
- **Network**: Verify localhost connectivity (ping test)

**Tooling Dependencies**:
- Verify `jq` is installed (required for JSON parsing)
- Verify `bc` is installed (required for arithmetic comparisons)
- Warn if optional tools missing: `perf`, `flamegraph`, `htop`

**Pre-flight Exit Codes**:
- 0: All checks passed, proceed with benchmarks
- 1: Critical check failed (missing binary, insufficient resources)
- 2: Soft failures (warnings issued but can proceed)

### 2. Benchmark Execution with Isolation

For each scenario in `scenarios/competitive/*.toml`:

**Pre-scenario Validation**:
- Verify system is stable (check load average, no CPU throttling)
- Verify no leftover Engram processes from previous scenario
- Sleep 30s cooldown from previous scenario (allow thermal stabilization)
- Check available memory (fail if <4GB available for scenario)

**Execution**:
- Start Engram server on isolated port (7432)
- Wait for server readiness (health check, not just sleep)
- Run loadtest: `loadtest run --scenario <file> --duration 60s --endpoint http://localhost:7432`
- Capture stdout to `tmp/competitive_benchmarks/<timestamp>_<scenario_name>.txt`
- Capture stderr to `tmp/competitive_benchmarks/<timestamp>_<scenario_name>_stderr.txt`
- Monitor server process during test (CPU, memory, threads)

**Diagnostics Collection**:
- Run `engram_diagnostics.sh` and save to `<timestamp>_<scenario_name>_diag.txt`
- Capture system metrics:
  - CPU usage (per-core if possible)
  - Memory stats (RSS, VSZ, swap usage)
  - I/O stats (reads/writes, latency)
  - Network stats (packets, errors)
- Save to `<timestamp>_<scenario_name>_sys.txt`
- Optional: Generate flame graph if `perf` available (save as `<timestamp>_<scenario_name>_flamegraph.svg`)

**Cleanup**:
- Stop Engram server gracefully (SIGTERM, wait 5s, SIGKILL if needed)
- Verify server stopped (check process, no zombie)
- Verify no leaked file descriptors
- Clear system page cache (if root, otherwise warn)

**Error Handling**:
- Scenario timeout: Kill after 120s (60s test + 60s buffer)
- Server crash: Log crash info, mark scenario as failed, continue
- Loadtest failure: Log error, preserve partial results, continue
- Partial results handling: Mark output files with `.partial` suffix if incomplete

### 3. Metadata and Context Capture

Create comprehensive metadata file `tmp/competitive_benchmarks/<timestamp>_metadata.txt`:

**Git Context**:
- Commit hash: `git rev-parse HEAD`
- Branch name: `git rev-parse --abbrev-ref HEAD`
- Dirty status: `git status --porcelain` (warn if uncommitted changes)
- Last commit message: `git log -1 --pretty=%B`

**System Information**:
- OS: `uname -a`
- CPU model: `sysctl -n machdep.cpu.brand_string` (macOS) or `/proc/cpuinfo` (Linux)
- CPU cores: Physical and logical core counts
- RAM total: In GB, from `sysctl hw.memsize` or `/proc/meminfo`
- Disk type: SSD vs HDD (detect from `/sys/block` or `diskutil`)

**Software Versions**:
- Engram core: `cargo metadata --format-version 1 | jq '.packages[] | select(.name == "engram-core") | .version'`
- Rust version: `rustc --version`
- Loadtest version: `cargo metadata --format-version 1 | jq '.packages[] | select(.name == "loadtest") | .version'`

**Benchmark Configuration**:
- Total scenarios executed: Count
- Duration per scenario: 60s
- Cooldown period: 30s
- Timestamp format: ISO 8601 (`YYYY-MM-DDTHH:MM:SSZ`)

### 4. Output Format and Naming

**Timestamp Format**: `YYYY-MM-DD_HH-MM-SS` (e.g., `2025-11-08_14-30-00`)

**File Naming Convention**:
```
<timestamp>_<scenario_name>.txt           # Loadtest stdout results
<timestamp>_<scenario_name>_stderr.txt    # Loadtest stderr errors
<timestamp>_<scenario_name>_diag.txt      # Engram diagnostics
<timestamp>_<scenario_name>_sys.txt       # System metrics
<timestamp>_<scenario_name>_flamegraph.svg # Optional flame graph
<timestamp>_metadata.txt                   # Metadata and context
<timestamp>_summary.txt                    # Execution summary
```

**Scenario Name Extraction**: TOML filename without extension
- `scenarios/competitive/qdrant_ann_1m_768d.toml` → `qdrant_ann_1m_768d`

**Atomic Writes**: Use temp files + `mv` for atomic writes (prevent partial file corruption)

### 5. Robust Error Handling

**Scenario Isolation**:
- Each scenario runs in isolated execution context
- One scenario failure does not cascade to others
- Failed scenarios logged but execution continues

**Retry Logic**:
- Transient failures (server startup timeout): Retry once with 10s delay
- Persistent failures (binary missing): Fail immediately
- Network failures (localhost unreachable): Retry with exponential backoff (1s, 2s, 4s)

**Failure Tracking**:
- Maintain array of failed scenarios: `FAILED_SCENARIOS=()`
- Log failure reason for each: `FAILURE_REASONS=()` (parallel array)
- Include in summary output

**Exit Codes**:
- 0: All scenarios passed
- 1: One or more scenarios failed
- 2: Pre-flight checks failed (no scenarios executed)
- 130: Interrupted by user (SIGINT/SIGTERM)

**Error Logging**:
- Log errors to stderr with timestamps
- Include context: scenario name, phase (pre-flight, execution, cleanup)
- Preserve all output even on failure (no truncation)

### 6. Signal Handling and Cleanup

**Signal Handlers**:
- SIGINT (Ctrl+C): Graceful shutdown, cleanup current scenario, exit 130
- SIGTERM: Same as SIGINT
- EXIT trap: Ensure cleanup always runs (kill server, remove temp files)

**Cleanup Actions**:
- Kill any orphaned Engram server processes
- Remove temporary files (e.g., `*.tmp`)
- Flush buffered output to log files
- Print partial results summary if interrupted

**Idempotency**:
- Safe to re-run after interruption
- Detect and clean up previous partial runs
- Use unique timestamps to avoid overwriting results

### 7. Progress Reporting

**Real-time Progress**:
- Print scenario name before execution
- Print live progress during 60s test (e.g., "30s elapsed...")
- Print success/failure immediately after each scenario
- Print ETA for full suite completion

**Summary Output**:
Create `<timestamp>_summary.txt` with:
- Total scenarios: 4
- Passed: count
- Failed: count (with names)
- Total duration: wall-clock time
- Output location: `tmp/competitive_benchmarks/<timestamp>/`

**Final Message**:
```
Competitive benchmark suite complete.
Results in tmp/competitive_benchmarks/<timestamp>/
Summary: 3/4 scenarios passed, 1 failed (qdrant_ann_1m_768d)
```

### 8. Performance Monitoring During Execution

**Per-scenario Monitoring**:
- Sample system metrics every 5s during 60s test (12 samples)
- Track peak CPU, peak memory, average I/O wait
- Detect anomalies: CPU throttling, OOM events, swap thrashing

**Threshold Detection**:
- Warn if CPU usage <50% (potential bottleneck elsewhere)
- Warn if memory usage >90% (risk of OOM)
- Warn if I/O wait >20% (disk bottleneck)

**Output to sys.txt**:
```
Timestamp | CPU% | Mem(MB) | Threads | IO_Wait% | Net(KB/s)
00:00:05  | 87.3 | 2048    | 32      | 1.2      | 450
00:00:10  | 91.2 | 2156    | 32      | 0.8      | 520
...
PEAK:     | 94.5 | 2301    | 34      | 2.1      | 680
AVG:      | 89.1 | 2187    | 32.3    | 1.1      | 512
```

## File Paths

```
scripts/competitive_benchmark_suite.sh
tmp/competitive_benchmarks/.gitkeep
```

## Acceptance Criteria

1. **Functionality**:
   - Script runs all 4 competitive scenarios without manual intervention
   - Each scenario produces 4-5 output files (results, stderr, diagnostics, sys metrics, optional flamegraph)
   - Metadata file contains commit hash, system info, and version info
   - Summary file provides clear pass/fail status

2. **Performance**:
   - Script completes in <10 minutes on 1M node scenarios
   - System resource checks complete in <10s
   - Cooldown periods respected (30s between scenarios)

3. **Reliability**:
   - Exit code correctly indicates success/failure
   - One scenario failure doesn't prevent others from running
   - Interrupted execution can be resumed (idempotent)
   - No orphaned processes after completion or interruption

4. **Quality**:
   - Script passes `shellcheck` linting with zero warnings
   - Script passes `bash -n` syntax validation
   - All error paths tested (missing binary, insufficient memory, etc.)
   - Signal handling tested (SIGINT during execution)

5. **Observability**:
   - Clear progress output to stdout
   - All errors logged to stderr with context
   - Summary provides actionable information
   - Diagnostic files parseable by automated tools

## Testing Approach

```bash
# 1. Validate shell syntax and style
shellcheck scripts/competitive_benchmark_suite.sh
bash -n scripts/competitive_benchmark_suite.sh

# 2. Dry-run mode (mock execution, no actual benchmarks)
./scripts/competitive_benchmark_suite.sh --dry-run
# Expected: Print execution plan, verify all checks, exit 0

# 3. Pre-flight check only mode
./scripts/competitive_benchmark_suite.sh --preflight-only
# Expected: Run all pre-flight checks, exit with status code

# 4. Single scenario test (for debugging)
./scripts/competitive_benchmark_suite.sh --scenario qdrant_ann_1m_768d
# Expected: Run only specified scenario, full diagnostics

# 5. Full integration test
time ./scripts/competitive_benchmark_suite.sh
# Expected: <10 minutes, all scenarios pass

# 6. Verify output structure
LATEST=$(ls -td tmp/competitive_benchmarks/*/ | head -1)
ls -lh "$LATEST"
# Expected: 4 scenarios × 4 files + 1 metadata + 1 summary = 18 files minimum

# 7. Verify metadata completeness
cat "$LATEST"/*_metadata.txt
# Expected: All required fields present (commit, CPU, RAM, versions)

# 8. Test error handling (missing binary)
mv target/release/loadtest target/release/loadtest.bak
./scripts/competitive_benchmark_suite.sh
# Expected: Pre-flight failure, exit code 2, clear error message
mv target/release/loadtest.bak target/release/loadtest

# 9. Test interruption handling (SIGINT)
./scripts/competitive_benchmark_suite.sh &
sleep 10
kill -INT %1
# Expected: Graceful shutdown, partial results preserved, exit 130

# 10. Verify idempotency (safe to re-run)
./scripts/competitive_benchmark_suite.sh
./scripts/competitive_benchmark_suite.sh
# Expected: No conflicts, new timestamp, both runs succeed
```

## Integration Points

- **Invoked by**: Task 005 quarterly review automation
- **Output consumed by**: Task 004 report generator
- **Dependencies**: `loadtest` binary from existing M17 infrastructure
- **Utilities**: `engram_diagnostics.sh` for process health checks

## Implementation Guidance

### Script Structure

```bash
#!/bin/bash
# Competitive Benchmark Suite Runner
# Production-grade orchestration script for quarterly baseline comparisons

set -euo pipefail  # Exit on error, undefined var, pipe failure
IFS=$'\n\t'        # Safe field splitting

# Configuration
readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
readonly OUTPUT_DIR="$PROJECT_ROOT/tmp/competitive_benchmarks"
readonly SCENARIOS_DIR="$PROJECT_ROOT/scenarios/competitive"
readonly LOADTEST_BIN="$PROJECT_ROOT/target/release/loadtest"
readonly ENGRAM_BIN="$PROJECT_ROOT/target/release/engram"

# Runtime state
TIMESTAMP=""
FAILED_SCENARIOS=()
FAILURE_REASONS=()

# Cleanup trap (always runs on exit)
cleanup() {
    local exit_code=$?
    # Kill any leftover processes
    pkill -P $$ engram 2>/dev/null || true
    # Flush logs
    sync
    exit $exit_code
}
trap cleanup EXIT
trap 'exit 130' INT TERM

# Functions: preflight_checks(), run_scenario(), collect_diagnostics(), ...
```

### Key Implementation Details

1. **Health Check for Server Readiness**:
   ```bash
   wait_for_server() {
       local max_attempts=30
       for i in $(seq 1 $max_attempts); do
           if curl -s http://localhost:7432/health >/dev/null 2>&1; then
               return 0
           fi
           sleep 1
       done
       return 1
   }
   ```

2. **Atomic File Writes**:
   ```bash
   atomic_write() {
       local content=$1
       local output_file=$2
       local temp_file="${output_file}.tmp.$$"

       echo "$content" > "$temp_file"
       mv "$temp_file" "$output_file"
   }
   ```

3. **Resource Checks with bc**:
   ```bash
   check_memory() {
       local total_gb=$(sysctl -n hw.memsize | awk '{print $1/1024/1024/1024}')
       if (( $(echo "$total_gb < 16" | bc -l) )); then
           warn "Low memory: ${total_gb}GB (recommended: 16GB+)"
       fi
   }
   ```

4. **Scenario Isolation**:
   ```bash
   run_scenario() {
       local scenario_file=$1
       (
           # Subshell provides isolation
           set -e
           local scenario_name=$(basename "$scenario_file" .toml)

           # Pre-checks
           verify_system_stable || return 1

           # Execute
           "$LOADTEST_BIN" run --scenario "$scenario_file" --duration 60
       ) || {
           FAILED_SCENARIOS+=("$scenario_name")
           FAILURE_REASONS+=("$?")
           return 1
       }
   }
   ```

## Performance Considerations

- Use `jq -c` for compact JSON parsing (faster)
- Avoid spawning subprocesses in loops (use builtins)
- Use process substitution for parallel monitoring
- Buffer log writes to reduce I/O overhead
- Use `sleep` strategically to avoid busy-waiting

## Security Considerations

- Never execute untrusted TOML files (validate schema)
- Use `mktemp` for temporary files (avoid /tmp races)
- Validate all user input (if adding CLI flags)
- Use `timeout` command to prevent infinite hangs
- Sanitize file paths (prevent directory traversal)

## Future Enhancements (Not in Scope)

- Parallel scenario execution (current: sequential)
- Web dashboard for real-time monitoring
- Slack/email notifications on completion
- Historical trend analysis (compare to previous runs)
- Automated regression detection (alert if >10% slower)
