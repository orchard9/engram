# API Versioning & Compatibility Guide

Comprehensive versioning policy for Engram's REST and gRPC APIs. This guide ensures you can upgrade Engram versions confidently without breaking your applications.

## Table of Contents

- [Versioning Strategy](#versioning-strategy)
- [Compatibility Guarantees](#compatibility-guarantees)
- [Version Detection](#version-detection)
- [Migration Process](#migration-process)
- [Breaking vs Non-Breaking Changes](#breaking-vs-non-breaking-changes)
- [Deprecation Timeline](#deprecation-timeline)
- [Version Compatibility Matrix](#version-compatibility-matrix)

## Versioning Strategy

Engram uses semantic versioning for both the server and API protocols:

### Server Versioning

Format: `MAJOR.MINOR.PATCH` (e.g., `1.2.3`)

- **MAJOR**: Breaking API changes, major architecture changes
- **MINOR**: New features, backward-compatible changes
- **PATCH**: Bug fixes, performance improvements

### API Versioning

#### REST API

Versions embedded in URL path:
```
/api/v1/memories/remember
/api/v2/memories/remember
```

Current version: **v1**

#### gRPC API

Versions in proto package:
```protobuf
package engram.v1;
package engram.v2;
```

Current version: **v1** (`engram.v1`)

### Version Lifecycle

Each API version goes through these stages:

1. **Preview** (0-3 months): Early access, breaking changes possible
2. **Stable** (12+ months): Production-ready, backward compatible
3. **Deprecated** (6 months): Still works, migration encouraged
4. **Sunset** (3 months warning): Final removal date announced
5. **Removed**: Version no longer available

## Compatibility Guarantees

### What We Guarantee

#### Backward Compatibility Within Major Version

Within a major version (e.g., v1.0 → v1.9), we guarantee:

- **Existing endpoints remain functional**: No removing or renaming endpoints
- **Request format stays compatible**: Adding optional fields only
- **Response format is additive**: New fields added, existing fields unchanged
- **Error codes remain stable**: Same error codes for same situations
- **Proto field numbers never reused**: Deleted fields leave gaps

#### Example: v1.5 → v1.9 (Backward Compatible)

**v1.5 RememberRequest:**
```protobuf
message RememberRequest {
  string memory_space_id = 1;
  Memory memory = 2;
  bool auto_link = 10;
}
```

**v1.9 RememberRequest (backward compatible):**
```protobuf
message RememberRequest {
  string memory_space_id = 1;
  Memory memory = 2;
  bool auto_link = 10;
  float link_threshold = 11;      // NEW: Optional field (backward compatible)
  bool immediate_index = 12;       // NEW: Optional field (backward compatible)
}
```

Clients using v1.5 request format continue working with v1.9 server - new fields default to zero values.

### What Changes Without Breaking Compatibility

These changes are **non-breaking** and can happen in minor versions:

#### Adding Optional Fields

```protobuf
// v1.0
message Memory {
  string id = 1;
  repeated float embedding = 2;
}

// v1.1 (non-breaking: new optional field)
message Memory {
  string id = 1;
  repeated float embedding = 2;
  map<string, string> metadata = 3;  // NEW: Clients ignore if not set
}
```

#### Adding New Enum Values

```protobuf
// v1.0
enum MemoryType {
  MEMORY_TYPE_UNSPECIFIED = 0;
  MEMORY_TYPE_EPISODIC = 1;
  MEMORY_TYPE_SEMANTIC = 2;
}

// v1.1 (non-breaking: new enum value)
enum MemoryType {
  MEMORY_TYPE_UNSPECIFIED = 0;
  MEMORY_TYPE_EPISODIC = 1;
  MEMORY_TYPE_SEMANTIC = 2;
  MEMORY_TYPE_PROCEDURAL = 3;  // NEW: Old clients treat as UNSPECIFIED
}
```

#### Adding New RPCs

```protobuf
// v1.0
service EngramService {
  rpc Remember(RememberRequest) returns (RememberResponse);
  rpc Recall(RecallRequest) returns (RecallResponse);
}

// v1.1 (non-breaking: new RPC)
service EngramService {
  rpc Remember(RememberRequest) returns (RememberResponse);
  rpc Recall(RecallRequest) returns (RecallResponse);
  rpc Recognize(RecognizeRequest) returns (RecognizeResponse);  // NEW
}
```

#### Adding Response Fields

```json
// v1.0 response
{
  "memory_id": "mem_123",
  "storage_confidence": {"value": 0.95}
}

// v1.1 response (non-breaking: new field)
{
  "memory_id": "mem_123",
  "storage_confidence": {"value": 0.95},
  "linked_memories": ["mem_456", "mem_789"]  // NEW: Old clients ignore
}
```

### What Breaks Compatibility

These changes **require major version bump** (v1 → v2):

#### Renaming or Removing Fields

```protobuf
// v1
message Memory {
  string memory_id = 1;  // Field name
}

// v2 (BREAKING: renamed field)
message Memory {
  string id = 1;  // Different name, same field number = BREAKING
}
```

#### Changing Field Types

```protobuf
// v1
message Confidence {
  float value = 1;
}

// v2 (BREAKING: type change)
message Confidence {
  double value = 1;  // float → double = BREAKING
}
```

#### Removing Endpoints

```protobuf
// v1
service EngramService {
  rpc Remember(RememberRequest) returns (RememberResponse);
  rpc Recall(RecallRequest) returns (RecallResponse);
  rpc Forget(ForgetRequest) returns (ForgetResponse);
}

// v2 (BREAKING: removed RPC)
service EngramService {
  rpc Remember(RememberRequest) returns (RememberResponse);
  rpc Recall(RecallRequest) returns (RecallResponse);
  // Forget removed = BREAKING
}
```

#### Changing Required Fields

```protobuf
// v1
message RememberRequest {
  Memory memory = 1;  // Optional
}

// v2 (BREAKING: now required)
message RememberRequest {
  Memory memory = 1 [required];  // Making optional field required = BREAKING
}
```

## Version Detection

### Detect Server Version

#### REST API

```bash
# Check server version
curl http://localhost:8080/api/version

# Response
{
  "server_version": "1.5.2",
  "api_versions": ["v1"],
  "proto_version": "engram.v1",
  "build_date": "2024-10-27T10:00:00Z",
  "git_commit": "a1b2c3d4"
}
```

#### gRPC API (Server Reflection)

```python
import grpc
from grpc_reflection.v1alpha import reflection_pb2, reflection_pb2_grpc

channel = grpc.insecure_channel('localhost:50051')
stub = reflection_pb2_grpc.ServerReflectionStub(channel)

# List available services
request = reflection_pb2.ServerReflectionRequest(
    list_services=""
)

for response in stub.ServerReflectionInfo(iter([request])):
    for service in response.list_services_response.service:
        print(f"Service: {service.name}")
        # Output: engram.v1.EngramService
```

### Client Version Negotiation

#### Graceful Degradation Pattern

```python
class VersionAwareClient:
    def __init__(self, host):
        self.channel = grpc.insecure_channel(host)

        # Detect server version
        self.server_version = self._detect_version()

        # Use appropriate stub
        if self.server_version >= (2, 0):
            from engram.v2 import engram_pb2_grpc
            self.stub = engram_pb2_grpc.EngramServiceStub(self.channel)
        else:
            from engram.v1 import engram_pb2_grpc
            self.stub = engram_pb2_grpc.EngramServiceStub(self.channel)

    def _detect_version(self):
        """Detect server API version."""
        try:
            # Try v2 endpoint
            response = self.stub.Introspect(IntrospectRequest())
            if hasattr(response, 'api_version'):
                return tuple(map(int, response.api_version.split('.')))
        except:
            # Fallback to v1
            return (1, 0, 0)

    def remember(self, memory):
        """Version-agnostic remember."""
        if self.server_version >= (2, 0):
            # Use v2 fields if available
            return self._remember_v2(memory)
        else:
            return self._remember_v1(memory)
```

### Feature Detection vs Version Checking

Instead of checking versions, check for feature availability:

```python
def has_pattern_completion(client):
    """Check if server supports pattern completion."""
    try:
        # Try calling the method
        client.Complete(CompleteRequest())
        return True
    except grpc.RpcError as e:
        if e.code() == grpc.StatusCode.UNIMPLEMENTED:
            return False
        raise

# Use feature detection
if has_pattern_completion(client):
    response = client.Complete(pattern)
else:
    # Fallback to recall-based approach
    response = client.Recall(cue)
```

## Migration Process

### Step-by-Step Migration Guide

#### Phase 1: Preparation (Before Upgrade)

1. **Check current version**
   ```bash
   curl http://localhost:8080/api/version
   engram version
   ```

2. **Review changelog**
   ```bash
   # Download migration guide for target version
   curl https://docs.engram.dev/migrations/v1-to-v2.md
   ```

3. **Test in staging**
   ```bash
   # Run compatibility tests
   engram migrate check --from v1 --to v2
   ```

#### Phase 2: Client Updates

1. **Update proto definitions**
   ```bash
   # Download new proto files
   git fetch --tags
   git checkout v2.0.0

   # Regenerate client stubs
   cd proto
   ./generate_all.sh
   ```

2. **Update client code**
   ```python
   # Old v1 code
   from engram.v1 import engram_pb2

   memory = engram_pb2.Memory(
       memory_id="mem_123",  # v1 field name
       embedding=[...],
   )

   # New v2 code
   from engram.v2 import engram_pb2

   memory = engram_pb2.Memory(
       id="mem_123",  # v2 field name (renamed)
       embedding=[...],
   )
   ```

3. **Run tests against both versions**
   ```bash
   # Test against v1 server
   ENGRAM_SERVER_VERSION=v1 pytest

   # Test against v2 server
   ENGRAM_SERVER_VERSION=v2 pytest
   ```

#### Phase 3: Server Upgrade

1. **Backup data**
   ```bash
   # Export all memory spaces
   engram backup create --all-spaces --output backup_v1.tar.gz
   ```

2. **Upgrade server**
   ```bash
   # Stop v1 server
   systemctl stop engram

   # Install v2 binary
   wget https://releases.engram.dev/v2.0.0/engram-linux-amd64
   sudo install engram-linux-amd64 /usr/local/bin/engram

   # Run migration
   engram migrate run --from v1 --to v2 --dry-run
   engram migrate run --from v1 --to v2

   # Start v2 server
   systemctl start engram
   ```

3. **Verify migration**
   ```bash
   # Check system health
   engram introspect | jq '.health'

   # Verify memory count
   engram introspect | jq '.statistics.total_memories'

   # Test recall
   curl -X POST http://localhost:8080/api/v2/memories/recall \
     -d '{"cue": {"semantic": {"query": "test"}}}'
   ```

#### Phase 4: Gradual Rollout

1. **Blue-green deployment**
   ```
   [Load Balancer]
       ├─→ v1 server (80% traffic)  ← Existing clients
       └─→ v2 server (20% traffic)  ← Updated clients
   ```

2. **Monitor metrics**
   ```bash
   # Compare error rates
   engram metrics --metric error_rate --compare v1 v2

   # Compare latency
   engram metrics --metric recall_latency_p99 --compare v1 v2
   ```

3. **Shift traffic gradually**
   ```bash
   # Week 1: 80/20 (v1/v2)
   # Week 2: 50/50
   # Week 3: 20/80
   # Week 4: 0/100 (full v2)
   ```

## Breaking vs Non-Breaking Changes

### Examples from Hypothetical v1 → v2 Migration

#### Breaking Changes

| Change | v1 | v2 | Why Breaking |
|--------|----|----|--------------|
| Field rename | `memory_id` | `id` | Clients using old field name fail |
| Type change | `float confidence` | `Confidence message` | Wire format incompatible |
| Required field | Optional `embedding` | Required `embedding` | Old requests missing field rejected |
| Enum value removal | `FORGET_MODE_SUPPRESS` | *(removed)* | Old clients using removed value fail |
| Default change | `auto_link: true` | `auto_link: false` | Behavior changes unexpectedly |

#### Non-Breaking Changes

| Change | v1 | v2 | Why Safe |
|--------|----|----|----------|
| Add optional field | *(no field)* | `link_threshold: float` | Old clients ignore new field |
| Add enum value | 3 values | 4 values | Old servers treat new value as UNSPECIFIED |
| Add RPC method | 10 methods | 11 methods | Old clients don't call new method |
| Add response field | 3 fields | 4 fields | Old clients ignore extra field |
| Widen type | `int32` | `int64` | Larger type accommodates smaller values |

### Proto Evolution Best Practices

#### Safe Field Additions

```protobuf
// v1.0
message Memory {
  string id = 1;
  repeated float embedding = 2;
  // Field 3 reserved for future use
}

// v1.5 (safe addition)
message Memory {
  string id = 1;
  repeated float embedding = 2;
  map<string, string> metadata = 3;  // Using reserved slot
  Confidence confidence = 4;          // New field
}
```

#### Safe Deprecation

```protobuf
// v1.0
message RememberRequest {
  string memory_id = 1;  // Legacy field
}

// v1.5 (deprecate, don't remove)
message RememberRequest {
  string memory_id = 1 [deprecated = true];  // Mark deprecated
  string id = 2;  // New field, same purpose
}

// Server accepts both fields for compatibility
```

## Deprecation Timeline

Engram follows a structured deprecation process:

### Timeline

```
Month 0: Deprecation Announced
  ↓ (Feature still works, warnings logged)
Month 6: Migration Guide Published
  ↓ (Feature still works, loud warnings)
Month 12: Deprecated in Release Notes
  ↓ (Feature still works, error logs)
Month 15: Removal Warning (3 months notice)
  ↓ (Feature still works, countdown to removal)
Month 18: Feature Removed
```

### Deprecation Notices

#### In API Responses

```json
{
  "memory_id": "mem_123",
  "warnings": [
    {
      "code": "DEPRECATION",
      "message": "Field 'memory_id' deprecated, use 'id' instead",
      "deprecation_date": "2024-06-01",
      "removal_date": "2025-06-01",
      "migration_guide": "https://docs.engram.dev/migrations/memory-id-to-id"
    }
  ]
}
```

#### In Logs

```
[WARN] 2024-10-27 10:30:00 Using deprecated field 'memory_id' in RememberRequest
[WARN] Migration guide: https://docs.engram.dev/migrations/memory-id-to-id
[WARN] Removal planned for: 2025-06-01 (8 months remaining)
```

#### In Client Libraries

```python
import warnings

def remember(self, memory_id=None, id=None):
    """Store a memory."""
    if memory_id is not None:
        warnings.warn(
            "Parameter 'memory_id' is deprecated, use 'id' instead. "
            "Removal planned for v2.0 (June 2025). "
            "See: https://docs.engram.dev/migrations/memory-id-to-id",
            DeprecationWarning,
            stacklevel=2
        )
        id = memory_id  # Fallback to old parameter

    # Use 'id' going forward
    ...
```

## Version Compatibility Matrix

### Server → Client Compatibility

| Server Version | v1 Client | v2 Client | v3 Client |
|----------------|-----------|-----------|-----------|
| v1.0-v1.9 | Full support | Not available | Not available |
| v2.0-v2.9 | Degraded (v1 deprecated) | Full support | Not available |
| v3.0+ | Not supported | Degraded (v2 deprecated) | Full support |

**Degraded**: Core operations work, but new features unavailable.

### Proto Compatibility

```
engram.v1 → engram.v2 migration:

Breaking changes:
  - Memory.memory_id renamed to Memory.id
  - Confidence changed from float to message
  - ForgetMode.FORGET_MODE_SUPPRESS removed

Non-breaking additions:
  - RememberRequest.link_threshold added
  - MemoryType.MEMORY_TYPE_PROCEDURAL added
  - PatternCompletion RPC added
```

### Migration Scripts

#### Automated Field Migration

```python
#!/usr/bin/env python3
"""Migrate v1 memories to v2 format."""

from engram.v1 import engram_pb2 as v1_pb2
from engram.v2 import engram_pb2 as v2_pb2
import grpc

def migrate_memory_v1_to_v2(v1_memory):
    """Convert v1.Memory to v2.Memory."""
    return v2_pb2.Memory(
        # Renamed field
        id=v1_memory.memory_id,

        # Unchanged fields
        embedding=v1_memory.embedding,
        activation=v1_memory.activation,
        last_access=v1_memory.last_access,
        created_at=v1_memory.created_at,
        decay_rate=v1_memory.decay_rate,
        content=v1_memory.content,
        metadata=v1_memory.metadata,

        # Changed type: float → Confidence message
        confidence=v2_pb2.Confidence(
            value=v1_memory.confidence,
            reasoning="Migrated from v1"
        )
    )

def migrate_all_memories():
    """Migrate all memories from v1 to v2 server."""
    # Connect to v1 server (read-only)
    v1_channel = grpc.insecure_channel('v1-server:50051')
    v1_client = v1_pb2_grpc.EngramServiceStub(v1_channel)

    # Connect to v2 server (write)
    v2_channel = grpc.insecure_channel('v2-server:50051')
    v2_client = v2_pb2_grpc.EngramServiceStub(v2_channel)

    # Stream all memories from v1
    v1_request = v1_pb2.RecallRequest(
        cue=v1_pb2.Cue(threshold=v1_pb2.Confidence(value=0.0)),  # Get all
        max_results=1000000
    )

    for v1_memory in v1_client.Recall(v1_request).memories:
        # Convert to v2 format
        v2_memory = migrate_memory_v1_to_v2(v1_memory)

        # Store in v2 server
        v2_client.Remember(v2_pb2.RememberRequest(
            memory_space_id="migrated_from_v1",
            memory=v2_memory
        ))

        print(f"Migrated: {v1_memory.memory_id} → {v2_memory.id}")

if __name__ == "__main__":
    migrate_all_memories()
```

#### Batch Migration with Progress

```bash
#!/bin/bash
# migrate_v1_to_v2.sh - Batch migration with progress tracking

set -e

V1_SERVER="localhost:50051"
V2_SERVER="localhost:50052"
BATCH_SIZE=1000

# Get total memory count
total=$(engram introspect --server $V1_SERVER | jq '.statistics.total_memories')
echo "Migrating $total memories from v1 to v2..."

# Migrate in batches
migrated=0
while [ $migrated -lt $total ]; do
    echo "Progress: $migrated / $total ($(($migrated * 100 / $total))%)"

    # Export batch from v1
    engram export --server $V1_SERVER \
        --offset $migrated \
        --limit $BATCH_SIZE \
        --output /tmp/batch_$migrated.jsonl

    # Transform v1 → v2 format
    python3 transform_v1_to_v2.py /tmp/batch_$migrated.jsonl /tmp/batch_$migrated_v2.jsonl

    # Import to v2
    engram import --server $V2_SERVER \
        --input /tmp/batch_$migrated_v2.jsonl

    migrated=$(($migrated + $BATCH_SIZE))
    sleep 1  # Rate limiting
done

echo "Migration complete: $total memories migrated"

# Verify counts match
v1_count=$(engram introspect --server $V1_SERVER | jq '.statistics.total_memories')
v2_count=$(engram introspect --server $V2_SERVER | jq '.statistics.total_memories')

if [ "$v1_count" -eq "$v2_count" ]; then
    echo "Verification passed: counts match ($v1_count)"
else
    echo "ERROR: Count mismatch! v1=$v1_count, v2=$v2_count"
    exit 1
fi
```

## Frequently Asked Questions

### Can I run v1 and v2 clients against the same server?

No. The server API version must match the client. Use separate server instances for different API versions during migration:

```
v1 clients → v1 server (port 50051)
v2 clients → v2 server (port 50052)
```

After migration complete, decommission v1 server.

### How long are old API versions supported?

- **Stable versions**: Minimum 12 months after release
- **Deprecated versions**: 6 months after deprecation announcement
- **After sunset**: 0 support (upgrade required)

Example timeline:
```
2024-01: v1.0 released (stable)
2025-01: v2.0 released, v1.0 deprecated (12 months stable)
2025-07: v1.0 sunset announced (6 months deprecation)
2025-10: v1.0 removed (3 months sunset warning)
```

### What if I can't upgrade immediately?

1. **Pin server version**: Use specific version tag
   ```bash
   docker run engram/engram:1.9.5  # Pin to last v1 version
   ```

2. **Extend support contract**: Contact Engram for extended v1 support

3. **Incremental migration**: Migrate memory spaces one at a time

### Do I need to migrate all data at once?

No. You can migrate incrementally:

```bash
# Migrate critical memory spaces first
engram migrate run --space "production_memories" --from v1 --to v2

# Migrate less critical spaces later
engram migrate run --space "staging_memories" --from v1 --to v2
```

### What happens to embeddings during migration?

Embeddings are binary-compatible across versions - no re-embedding needed. Only metadata and schema change.

### Can I rollback after upgrading?

Yes, if you:
1. **Keep v1 backup**: Before upgrading
2. **Don't write new data**: Migration is one-way
3. **Rollback window**: Within 24 hours

```bash
# Rollback to v1 (within 24 hours)
systemctl stop engram
engram restore --backup backup_v1.tar.gz
systemctl start engram@v1
```

## Next Steps

- **Migration Tools**: [Automated Migration Scripts](/operations/migration-tools.md)
- **Changelog**: [API Change History](/changelog.md)
- **Upgrade Guide**: [Production Upgrade Checklist](/operations/upgrade-checklist.md)
- **Testing**: [Compatibility Testing Guide](/operations/compatibility-testing.md)
