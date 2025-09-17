# System Health

Monitor and introspect your Engram instance.

## Simple Health Check

Quick health status for monitoring systems.

**Endpoint:** `GET /health`

### Response

```json
{
  "status": "healthy",
  "service": "engram",
  "version": "0.1.0",
  "timestamp": "2025-09-17T01:30:00Z"
}
```

## Detailed System Health

Comprehensive health information including memory statistics and performance metrics.

**Endpoint:** `GET /api/v1/system/health`

### Response

```json
{
  "cognitive_load": {
    "capacity_remaining": "85%",
    "consolidation_queue": 0,
    "current": "low"
  },
  "memory_system": {
    "consolidation_active": true,
    "pattern_completion": "available",
    "spreading_activation": "normal",
    "total_memories": 42
  },
  "status": "healthy",
  "system_message": "All cognitive processes functioning normally"
}
```

## System Introspection

Internal statistics and performance metrics.

**Endpoint:** `GET /api/v1/system/introspect`

### Response

```json
{
  "memory_statistics": {
    "average_activation": 0.5,
    "consolidation_states": {
      "archived": 15,
      "consolidated": 20,
      "recent": 7
    },
    "total_nodes": 42
  },
  "performance_metrics": {
    "activation_efficiency": "high",
    "avg_recall_time_ms": 45,
    "memory_capacity_used": "15%"
  },
  "system_processes": {
    "dream_simulation": "offline",
    "memory_consolidation": "scheduled",
    "pattern_completion": "ready",
    "spreading_activation": "idle"
  }
}
```

## CLI Status Command

For command-line monitoring:

```bash
./target/debug/engram status
```

This provides a formatted view of the system health with visual indicators and helpful commands.