# Memory Reconsolidation Core: Architectural Perspectives

## Cognitive Architecture Designer

Memory reconsolidation, discovered by Nader, Schafe, & Le Doux (2000), fundamentally changed our understanding of memory stability. When a consolidated memory is reactivated, it enters a labile state for approximately 6 hours, during which it can be modified, strengthened, or even disrupted. This isn't a bug - it's a feature that allows memories to be updated with new information while maintaining their core associations.

The biological mechanism involves protein synthesis in the hippocampus and amygdala. Retrieval triggers a destabilization process where synaptic connections become temporarily malleable. New protein synthesis is required to restabilize the memory, creating a window during which the memory trace can be modified. Blocking protein synthesis during this window can prevent reconsolidation, effectively erasing the memory (Nader & Hardt, 2009).

From a cognitive architecture perspective, reconsolidation solves the stability-plasticity dilemma: how do we maintain stable long-term memories while remaining adaptable to new experiences? The answer is controlled lability triggered by reactivation. Memories aren't permanently fixed or constantly changing - they exist in a dynamic equilibrium where reactivation creates update opportunities.

Engram's implementation must capture three critical phases: (1) reactivation triggering labilization, (2) a time-limited window of malleability, and (3) reconsolidation that restabilizes the modified memory. The temporal dynamics are precise: labilization onset within minutes of reactivation, peak malleability at 1-2 hours, complete restabilization by 6 hours. These timescales are consistent across species from rodents to humans (Lee, 2009).

## Memory Systems Researcher

Lee (2009) provides a comprehensive review of reconsolidation phenomena, highlighting key empirical constraints our implementation must satisfy:

1. **Reactivation Boundary Conditions**: Not all retrieval triggers reconsolidation. The memory must be actively retrieved and brought into working memory, not just primed through spreading activation. This requires a reactivation strength threshold.

2. **Time-Limited Windows**: The labilization window is not indefinite. Nader & Hardt (2009) show that protein synthesis inhibitors are only effective within 6 hours of reactivation. After this window closes, the memory restabilizes and becomes resistant to modification.

3. **Strength-Dependent Labilization**: Stronger, more consolidated memories may require stronger reactivation to enter labile states (Suzuki et al., 2004). A well-consolidated memory from years ago might not labilize from weak retrieval.

Statistical validation requires careful experimental design. The classic protocol involves:
- Day 1: Initial learning (conditioning or association formation)
- Day 2-7: Consolidation period (no intervention)
- Day 8: Reactivation with optional intervention during labile window
- Day 9+: Test for memory retention

Our acceptance criteria must show:
1. Memories modified during labile window show 30-50% strength change (Cohen's d > 0.8)
2. Memories modified outside labile window show <10% change (no significant effect)
3. Labilization occurs within 5 minutes of reactivation (95% CI)
4. Restabilization completes within 6 hours (95% CI: [5-7 hours])

The statistical power must be sufficient to detect medium effect sizes (d = 0.5) with p < 0.01, requiring n > 30 for within-subjects designs.

## Rust Graph Engine Architect

Implementing reconsolidation requires tracking memory state across three phases: stable, labile, and reconsolidating. This state must be queryable without blocking concurrent operations. The architecture uses per-edge state machines with atomic transitions:

```rust
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum MemoryState {
    Stable,           // Normal consolidated state
    Labile,           // Reactivated, modifiable window open
    Reconsolidating,  // Restabilization in progress
}

pub struct EdgeMetadata {
    state: AtomicU8,  // Packed MemoryState representation
    labilization_time: AtomicU64,  // Timestamp when labile state entered
    original_strength: f32,  // Pre-labilization strength for comparison
    modification_count: AtomicU32,  // Track update frequency
}
```

The state transitions must be atomic to prevent race conditions:

```rust
impl EdgeMetadata {
    pub fn try_labilize(&self, now: Timestamp, threshold: f32) -> Result<bool> {
        // Atomic compare-exchange: Stable -> Labile
        let result = self.state.compare_exchange(
            MemoryState::Stable as u8,
            MemoryState::Labile as u8,
            Ordering::AcqRel,
            Ordering::Acquire,
        );

        if result.is_ok() {
            self.labilization_time.store(now.as_nanos(), Ordering::Release);
            Ok(true)
        } else {
            Ok(false)  // Already labile or reconsolidating
        }
    }
}
```

Performance targets: state transition in <20ns (atomic compare-exchange), state query in <5ns (atomic load). Memory overhead is 24 bytes per edge (state + timestamp + original strength + counter), acceptable for cognitive validation workloads.

## Systems Architecture Optimizer

The reconsolidation system creates interesting optimization challenges around time-based state transitions. We need to transition edges from Labile to Reconsolidating at precise times without polling every edge continuously.

Solution: priority queue of reconsolidation events, processed by a dedicated background thread:

```rust
pub struct ReconsolidationScheduler {
    /// Min-heap of (deadline, edge_id) pairs
    events: Arc<Mutex<BinaryHeap<Reverse<(Timestamp, EdgeId)>>>>,
    /// Signal for new events
    notify: Arc<Notify>,
}

impl ReconsolidationScheduler {
    pub fn schedule_reconsolidation(&self, edge: EdgeId, labilization_time: Timestamp) {
        let deadline = labilization_time + Duration::from_hours(6);

        let mut events = self.events.lock();
        events.push(Reverse((deadline, edge)));
        drop(events);

        self.notify.notify_one();
    }

    pub async fn run_event_loop(&self, graph: Arc<MemoryGraph>) {
        loop {
            let next_event = {
                let events = self.events.lock();
                events.peek().map(|Reverse((time, _))| *time)
            };

            if let Some(deadline) = next_event {
                let now = Timestamp::now();
                if deadline <= now {
                    // Process event
                    let (_, edge_id) = self.events.lock().pop().unwrap().0;
                    graph.begin_reconsolidation(edge_id);
                } else {
                    // Sleep until next deadline
                    tokio::time::sleep_until((deadline - now).into()).await;
                }
            } else {
                // No events, wait for notification
                self.notify.notified().await;
            }
        }
    }
}
```

This design ensures O(log n) scheduling overhead and precise timing without continuous polling. Memory overhead scales with the number of simultaneously labile edges, typically <1000 in realistic workloads.

Cache optimization: labilization timestamps should be cache-line aligned with edge state to ensure atomic state checks don't cause false sharing. The background scheduler thread should be pinned to a dedicated core to prevent scheduling jitter from affecting reconsolidation timing precision.
