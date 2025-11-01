# ObservationQueue Flow Diagram

## Mermaid Diagram

```mermaid
graph TB
    subgraph Client["Client Application"]
        C1[gRPC Client]
        C2[WebSocket Client]
    end

    subgraph Server["Engram Server"]
        subgraph Handlers["Stream Handlers"]
            H1[ObserveStream Handler]
            H2[RecallStream Handler]
            H3[MemoryFlow Handler]
        end

        subgraph Session["Session Management"]
            SM[SessionManager<br/>DashMap]
            SS1[Session 1<br/>State: Active]
            SS2[Session 2<br/>State: Paused]
            SS3[Session 3<br/>State: Closed]
            SM --> SS1
            SM --> SS2
            SM --> SS3
        end

        subgraph Queue["ObservationQueue<br/>Lock-Free SegQueue"]
            QL[Low Priority<br/>25K capacity<br/>Background ops]
            QN[Normal Priority<br/>50K capacity<br/>Standard ops]
            QH[High Priority<br/>5K capacity<br/>Immediate ops]
        end

        subgraph Workers["Worker Pool"]
            W1[Worker 1<br/>Space A]
            W2[Worker 2<br/>Space B]
            W3[Worker 3<br/>Space A]
            W4[Worker 4<br/>Space C]
        end

        subgraph Storage["Storage Layer"]
            G1[HNSW Graph A<br/>generation=500]
            G2[HNSW Graph B<br/>generation=300]
            G3[HNSW Graph C<br/>generation=150]
        end
    end

    C1 -->|1. Init<br/>session_id| H1
    C1 -->|2. Observation<br/>seq=1,2,3...| H1
    C2 -->|Observation| H3

    H1 -->|Create Session| SM
    H1 -->|Validate Sequence| SS1
    H1 -->|Enqueue| QH
    H1 -->|Enqueue| QN
    H1 -->|Enqueue| QL

    H1 -->|ObservationAck<br/>ACCEPTED| C1
    H1 -->|StreamStatus<br/>BACKPRESSURE| C1

    QH -->|Dequeue Batch<br/>size=10-500| W1
    QN -->|Dequeue Batch| W2
    QN -->|Work Stealing<br/>threshold=1000| W3
    QL -->|Dequeue Batch| W4

    W1 -->|Insert<br/>generation=501| G1
    W2 -->|Insert<br/>generation=301| G2
    W3 -->|Insert<br/>generation=502| G1
    W4 -->|Insert<br/>generation=151| G3

    W1 -->|mark_committed<br/>generation=501| QN
    W2 -->|mark_committed<br/>generation=301| QN

    H2 -->|Snapshot<br/>gen=500| G1
    H2 -->|Incremental Results<br/>batch_size=10| C1

    style QH fill:#ff9999
    style QN fill:#99ccff
    style QL fill:#99ff99
    style G1 fill:#ffcc99
    style G2 fill:#ffcc99
    style G3 fill:#ffcc99
```

## ASCII Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                          CLIENT APPLICATION                         │
│  ┌─────────────────┐              ┌─────────────────┐              │
│  │  gRPC Client    │              │ WebSocket Client│              │
│  └────────┬────────┘              └────────┬────────┘              │
│           │                                 │                        │
└───────────┼─────────────────────────────────┼────────────────────────┘
            │                                 │
            │ Init + Observations             │ Observations
            ▼                                 ▼
┌─────────────────────────────────────────────────────────────────────┐
│                         ENGRAM SERVER                                │
│                                                                       │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │                    STREAM HANDLERS                           │   │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │   │
│  │  │ ObserveStream│  │ RecallStream │  │ MemoryFlow   │      │   │
│  │  │   Handler    │  │   Handler    │  │   Handler    │      │   │
│  │  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘      │   │
│  └─────────┼──────────────────┼──────────────────┼──────────────┘   │
│            │                  │                  │                   │
│            ▼                  │                  │                   │
│  ┌─────────────────────────┐ │                  │                   │
│  │  SESSION MANAGEMENT     │ │                  │                   │
│  │  ┌─────────────────┐   │ │                  │                   │
│  │  │ SessionManager  │   │ │                  │                   │
│  │  │   (DashMap)     │   │ │                  │                   │
│  │  ├─────────────────┤   │ │                  │                   │
│  │  │ Session 1: Active│  │ │                  │                   │
│  │  │ Session 2: Paused│  │ │                  │                   │
│  │  │ Session 3: Closed│  │ │                  │                   │
│  │  └─────────────────┘   │ │                  │                   │
│  └────────┬────────────────┘ │                  │                   │
│           │                  │                  │                   │
│           │ Validate Seq     │                  │                   │
│           ▼                  │                  │                   │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │             OBSERVATION QUEUE (Lock-Free SegQueue)          │   │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │   │
│  │  │   HIGH       │  │   NORMAL     │  │    LOW       │      │   │
│  │  │  Priority    │  │  Priority    │  │  Priority    │      │   │
│  │  │  5K cap      │  │  50K cap     │  │  25K cap     │      │   │
│  │  │  Immediate   │  │  Standard    │  │  Background  │      │   │
│  │  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘      │   │
│  └─────────┼──────────────────┼──────────────────┼──────────────┘   │
│            │                  │                  │                   │
│            │ Dequeue          │ Dequeue          │ Dequeue           │
│            │ batch=10-500     │ batch=10-500     │ batch=10-500      │
│            ▼                  ▼                  ▼                   │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │                      WORKER POOL                             │   │
│  │  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐        │   │
│  │  │Worker 1 │  │Worker 2 │  │Worker 3 │  │Worker 4 │        │   │
│  │  │Space A  │  │Space B  │  │Space A  │  │Space C  │        │   │
│  │  │         │  │         │  │(stealing)│  │         │        │   │
│  │  └────┬────┘  └────┬────┘  └────┬────┘  └────┬────┘        │   │
│  └───────┼────────────┼────────────┼────────────┼──────────────┘   │
│          │            │            │            │                   │
│          │ Insert     │ Insert     │ Insert     │ Insert            │
│          │ gen=501    │ gen=301    │ gen=502    │ gen=151           │
│          ▼            ▼            ▼            ▼                   │
│  ┌────────────────────────────────────────────────────────────┐    │
│  │                    STORAGE LAYER                            │    │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐         │    │
│  │  │ HNSW Graph A│  │ HNSW Graph B│  │ HNSW Graph C│         │    │
│  │  │ gen=500     │  │ gen=300     │  │ gen=150     │         │    │
│  │  └─────────────┘  └─────────────┘  └─────────────┘         │    │
│  └────────────────────────────────────────────────────────────┘    │
│                                                                      │
└──────────────────────────────────────────────────────────────────────┘

FLOW LEGEND:
  ────▶  Data flow
  ═══▶   Control flow
  ┌───┐  Component
  │   │
  └───┘

KEY CONCEPTS:
1. Lock-Free Queue: Three priority lanes (High/Normal/Low)
2. Space Partitioning: Workers process observations for specific memory spaces
3. Work Stealing: Workers steal batches when queue depth > 1000
4. Generation Tracking: Each insert increments generation counter
5. Backpressure: Triggered when queue depth > 85% capacity
```

## Flow Sequence

### 1. Observation Ingestion

```
Client                Handler              SessionManager        Queue               Worker              Graph
  │                      │                        │                │                   │                  │
  ├──Init──────────────▶│                        │                │                   │                  │
  │                      ├──Create Session───────▶│                │                   │                  │
  │                      │◀─────session_id────────┤                │                   │                  │
  │◀────InitAck─────────┤                        │                │                   │                  │
  │                      │                        │                │                   │                  │
  ├──Observation─seq=1─▶│                        │                │                   │                  │
  │                      ├──Validate Seq─────────▶│                │                   │                  │
  │                      │◀──────OK───────────────┤                │                   │                  │
  │                      ├──Enqueue─────────────────────────────▶│                   │                  │
  │                      │◀─────OK──────────────────────────────┤                   │                  │
  │◀────ObservationAck──┤                        │                │                   │                  │
  │                      │                        │                │                   │                  │
  │                      │                        │                ├──Dequeue Batch──▶│                  │
  │                      │                        │                │                   ├──Insert──────▶│
  │                      │                        │                │                   │◀───gen=N+1────┤
  │                      │                        │                │◀──Mark Committed──┤                  │
  │                      │                        │                │   gen=N+1         │                  │
```

### 2. Backpressure Activation

```
Client                Handler              Queue                Worker
  │                      │                   │                     │
  ├──Observation─seq=50─▶│                   │                     │
  │                      ├──Enqueue─────────▶│                     │
  │                      │                   │ (depth > 85%)       │
  │                      │◀───OVER_CAPACITY──┤                     │
  │◀────StreamStatus────┤                   │                     │
  │     BACKPRESSURE     │                   │                     │
  │                      │                   │                     │
  │  (client pauses)     │                   │                     │
  │        ...           │                   ├──Workers Drain─────▶│
  │        ...           │                   │                     │
  │        ...           │                   │◀────Batch Empty─────┤
  │                      │                   │ (depth < 50%)       │
  │◀────StreamStatus────┤◀──READY───────────┤                     │
  │     READY            │                   │                     │
  │  (client resumes)    │                   │                     │
```

### 3. Snapshot Isolation for Recall

```
Client                RecallHandler        Queue                Graph
  │                      │                   │                     │
  ├──RecallRequest──────▶│                   │                     │
  │  snapshot=true       │                   │                     │
  │                      ├──Get Generation──▶│                     │
  │                      │◀───gen=500────────┤                     │
  │                      │                   │                     │
  │                      ├──Search with Filter─────────────────▶│
  │                      │  (node.gen <= 500)                    │
  │                      │◀────Results [batch 1]─────────────────┤
  │◀────RecallResponse──┤                   │                     │
  │    more_results=true │                   │                     │
  │                      │◀────Results [batch 2]─────────────────┤
  │◀────RecallResponse──┤                   │                     │
  │    more_results=true │                   │                     │
  │                      │◀────Results [batch N]─────────────────┤
  │◀────RecallResponse──┤                   │                     │
  │    more_results=false│                   │                     │
```

## Performance Characteristics

| Component | Metric | Target | Actual |
|-----------|--------|--------|--------|
| Queue Enqueue | Latency P99 | < 1ms | ~0.5ms |
| Queue Dequeue | Latency P99 | < 1ms | ~0.3ms |
| HNSW Insert | Latency P99 | < 50ms | ~30ms |
| Session Validation | Latency P99 | < 100μs | ~50μs |
| Observation Ack | Latency P99 | < 10ms | ~5ms |
| Worker Throughput | obs/sec per worker | 12.5K | ~15K |
| Total Throughput | obs/sec (8 workers) | 100K | ~120K |
| Memory per Session | Bytes | < 1MB | ~800KB |
| Queue Capacity | Observations | 80K total | 80K (5K+50K+25K) |
