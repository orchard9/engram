# CLI Reference

Command-line interface for Engram cognitive graph database.

## engram start

Start the Engram server.

**Syntax:**
```bash
engram start [OPTIONS]
```

**Options:**
- `--http-port PORT` - HTTP API port (default: 7432)
- `--grpc-port PORT` - gRPC API port (default: 50051)
- `--config PATH` - Configuration file path

**Example:**
```bash
engram start --http-port 8080 --grpc-port 8081
```

## engram stop

Gracefully stop the Engram server.

**Syntax:**
```bash
engram stop
```

Sends SIGTERM to running server process and waits for graceful shutdown.

## engram status

Show server health and cluster status.

**Syntax:**
```bash
engram status [OPTIONS]
```

**Options:**
- `--space SPACE_ID` - Show metrics for specific memory space only

**Examples:**
```bash
# Overall health
engram status

# Specific space health
engram status --space production
```

**Output (single-node)**:
```
Engram Server Health Check
═══════════════════════════════════════
Process ID: 12345
HTTP Port: 7432
Process Status: Running
HTTP Health: Responding (15ms)

Per-Space Metrics:
┌────────────────────┬───────────┬──────────┬─────────────┬─────────────────┐
│ Space              │ Memories  │ Pressure │ WAL Lag (ms)│ Consolidation   │
├────────────────────┼───────────┼──────────┼─────────────┼─────────────────┤
│ default            │      1234 │      2.5%│        12.50│          5.20/s │
└────────────────────┴───────────┴──────────┴─────────────┴─────────────────┘
```

**Output (cluster)**:
```
Engram Server Health Check
═══════════════════════════════════════
Process ID: 12345
HTTP Port: 7432
Process Status: Running
HTTP Health: Responding (15ms)

Overall Status: healthy
Cluster Node: engram-node1 | alive 2 | suspect 0 | dead 0 | total 3

Per-Space Metrics:
[table as above]
```

## engram config

Manage configuration at runtime.

**Subcommands:**
- `config get KEY` - Get configuration value
- `config set KEY VALUE` - Set configuration value
- `config list` - List all configuration

**Examples:**
```bash
# Get specific key
engram config get feature_flags.spreading_api_beta

# Set feature flag
engram config set feature_flags.spreading_api_beta true

# List all config
engram config list
```

## engram space

Manage memory spaces (multi-tenancy).

**Subcommands:**
- `space list` - List all memory spaces
- `space create SPACE_ID` - Create new memory space
- `space delete SPACE_ID` - Delete memory space

**Examples:**
```bash
# List spaces
engram space list

# Create production space
engram space create production

# Delete test space
engram space delete test
```

## engram memory

Memory operations via CLI.

**Subcommands:**
- `memory create CONTENT` - Create new memory
- `memory list` - List memories
- `memory search QUERY` - Search memories

**Options:**
- `--space SPACE_ID` - Target specific memory space

**Examples:**
```bash
# Create memory
engram memory create "The mitochondria is the powerhouse of the cell"

# Create in specific space
engram memory create --space research "CRISPR gene editing"

# Search memories
engram memory search "mitochondria"

# List all memories
engram memory list
```

## engram query

Execute query language statements.

**Syntax:**
```bash
engram query "QUERY_STRING" [OPTIONS]
```

**Options:**
- `--format FORMAT` - Output format: json | table | raw (default: json)

**Examples:**
```bash
# Recall with confidence filter
engram query "RECALL what='mitochondria' CONFIDENCE > 0.8"

# Spreading activation
engram query "SPREAD FROM what='cellular biology' HOPS 2"

# Pattern completion
engram query "COMPLETE what='Einstein published theory' CONFIDENCE_THRESHOLD 0.7"

# Consolidation
engram query "CONSOLIDATE SPACE 'research' MIN_CLUSTER_SIZE 3"
```

## engram version

Show version information.

**Syntax:**
```bash
engram version
```

**Output:**
```
Engram v0.1.0
Rust Edition 2024
Features: hnsw_index, memory_mapped_persistence, psychological_decay, zig-kernels
```
