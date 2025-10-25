# Reconsolidation Integration: Architectural Perspectives

## Cognitive Architecture Designer

Schiller et al. (2010) discovered that reconsolidation can be leveraged therapeutically: presenting new information during the labile window allows updating of maladaptive memories without extinction. This has profound implications for treating PTSD, phobias, and addiction - conditions where pathological memories drive behavior.

The integration with consolidation pipelines requires careful coordination. Engram's Milestone 6 consolidation system implements hippocampal-neocortical transfer: memories initially encoded in hippocampus gradually migrate to neocortex over days to weeks. Reconsolidation interacts with this process in complex ways: reactivating a partially consolidated memory can reset its consolidation state, requiring renewed transfer.

From a cognitive architecture perspective, reconsolidation-consolidation interaction creates a dynamic equilibrium. New memories undergo initial consolidation (hippocampal encoding, synaptic strengthening). Reactivation triggers reconsolidation (temporary labilization, opportunity for modification). Multiple reactivations with reinforcement accelerate consolidation (faster transfer to neocortex, increased stability). This cycle allows memories to stabilize through use while remaining adaptable to new information.

The temporal dynamics are critical. Consolidation operates on a timescale of hours to days (Milestone 6 uses exponential decay with tau = 24 hours). Reconsolidation operates on a timescale of minutes to hours (6-hour labile window). These processes must be coordinated without creating race conditions where consolidation and reconsolidation compete for the same memory trace.

## Memory Systems Researcher

Walker et al. (2003) demonstrated that sleep-dependent consolidation can be disrupted by daytime reactivation and reconsolidation. This reveals that consolidation and reconsolidation aren't independent processes - they interact and can interfere with each other.

Statistical validation of integration requires measuring both processes simultaneously:

1. **Consolidation Rate**: Track memory strength increase over 24-hour period without reactivation (baseline)
2. **Reconsolidation Impact**: Measure strength change when memory is reactivated at different consolidation stages
3. **Interaction Effects**: Test whether early-stage vs late-stage consolidated memories respond differently to reconsolidation

Acceptance criteria must show:
- Early consolidation (0-6 hours post-encoding): reactivation with modification produces 40-60% strength change
- Mid consolidation (6-24 hours): reactivation with modification produces 20-40% strength change
- Late consolidation (24+ hours): reactivation with modification produces 10-20% strength change

This gradient reflects that newly encoded memories are more malleable during initial consolidation, while well-consolidated memories show increased resistance to modification (Suzuki et al., 2004).

The experimental design must control for confounds: spacing effects, interference from other learning, and baseline memory decay. Within-subjects designs with counterbalanced conditions provide the statistical power needed to detect interaction effects (n > 40 for small-to-medium effects with p < 0.01).

## Rust Graph Engine Architect

Integrating reconsolidation with the existing consolidation pipeline requires careful state management. The consolidation scheduler (Milestone 6) already tracks memory age, consolidation strength, and transfer progress. Reconsolidation must coordinate with these existing mechanisms.

Architecture approach: extend edge metadata to include both consolidation state and reconsolidation state:

```rust
pub struct IntegratedMemoryState {
    // Consolidation state (from Milestone 6)
    consolidation_level: AtomicF32,  // 0.0 = unconsolidated, 1.0 = fully consolidated
    encoding_time: Timestamp,
    last_consolidation_update: AtomicU64,

    // Reconsolidation state (from Task 006)
    reconsolidation_state: AtomicU8,  // Stable/Labile/Reconsolidating
    labilization_time: AtomicU64,
    modification_count: AtomicU32,
}
```

The consolidation scheduler must check reconsolidation state before applying consolidation updates:

```rust
impl ConsolidationScheduler {
    pub async fn apply_consolidation_step(&self, edge: EdgeId) -> Result<()> {
        let state = self.graph.get_integrated_state(edge)?;

        // Check if memory is currently labile
        let recon_state = state.reconsolidation_state.load(Ordering::Acquire);
        if recon_state == MemoryState::Labile as u8 {
            // Memory is labile - pause consolidation until reconsolidation completes
            return Ok(());  // Skip this consolidation step
        }

        // Apply normal consolidation update
        let current_level = state.consolidation_level.load(Ordering::Acquire);
        let new_level = self.calculate_consolidation_increment(current_level);
        state.consolidation_level.store(new_level, Ordering::Release);

        Ok(())
    }
}
```

Performance targets: state coordination adds <10ns overhead to consolidation checks (single atomic load), no additional memory overhead (states packed into existing metadata).

## Systems Architecture Optimizer

The reconsolidation-consolidation integration creates interesting opportunities for zero-overhead abstractions. Both processes need precise temporal scheduling, both modify memory strength, and both interact with spreading activation. We can unify these through a single memory lifecycle manager.

Architecture: event-driven memory lifecycle with priority-based scheduling:

```rust
pub struct MemoryLifecycleManager {
    /// Unified event queue for consolidation and reconsolidation
    events: Arc<SegQueue<MemoryEvent>>,

    /// Background workers processing events
    workers: Vec<JoinHandle<()>>,

    /// Reference to memory graph
    graph: Arc<MemoryGraph>,
}

#[derive(Clone)]
enum MemoryEvent {
    ConsolidationStep {
        edge: EdgeId,
        scheduled_time: Timestamp,
        priority: u8,
    },
    BeginReconsolidation {
        edge: EdgeId,
        labilization_time: Timestamp,
        priority: u8,
    },
    CompleteReconsolidation {
        edge: EdgeId,
        scheduled_time: Timestamp,
        priority: u8,
    },
}

impl MemoryLifecycleManager {
    pub async fn run_worker(&self) {
        loop {
            if let Some(event) = self.events.pop() {
                match event {
                    MemoryEvent::ConsolidationStep { edge, .. } => {
                        self.apply_consolidation(edge).await;
                    }
                    MemoryEvent::BeginReconsolidation { edge, .. } => {
                        self.begin_reconsolidation(edge).await;
                    }
                    MemoryEvent::CompleteReconsolidation { edge, .. } => {
                        self.complete_reconsolidation(edge).await;
                    }
                }
            } else {
                tokio::time::sleep(Duration::from_millis(10)).await;
            }
        }
    }
}
```

This unified architecture eliminates duplication between consolidation and reconsolidation schedulers, reduces memory overhead by sharing event queues, and provides a single point of coordination for memory lifecycle management. The lock-free queue (SegQueue) ensures minimal contention when multiple workers process events concurrently.
