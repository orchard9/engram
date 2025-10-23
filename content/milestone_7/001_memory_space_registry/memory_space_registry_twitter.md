1/ Engram still boots a single global `MemoryStore`. Milestone 7 starts by adding a Memory Space Registry so every agent can own an isolated graph.

2/ The registry introduces a `MemorySpaceId` newtype, creates per-space stores on demand, and caches handles for low-latency routing.

3/ CLI and API layers will call `resolve_space()` before any store/recall, guaranteeing every request declares its tenant—or falls back to the configured default with a warning.

4/ Registry lifecycle hooks provision persistence roots up front, laying groundwork for per-space WAL and tiered storage workers.

5/ Operators get `engram space list/create` commands plus config knobs for default spaces. Rich errors guide misconfigured clients instead of silently mixing data.

6/ With the registry in place, we can partition persistence, tag metrics, and build isolation tests—unlocking true multi-agent memory on shared infrastructure.
