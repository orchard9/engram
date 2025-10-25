# Memory Reconsolidation: Research and Technical Foundation

## The Revolutionary Finding

Nader et al. (2000) overturned a century of memory theory with a simple experiment. They showed that retrieving a consolidated memory makes it temporarily labile again - subject to modification, enhancement, or disruption. This was thought impossible: consolidation was believed to be a one-way process.

The paradigm: rats learned fear conditioning (tone paired with shock). 24 hours later - well past consolidation - the memory was retrieved by presenting the tone alone. Immediately after retrieval, protein synthesis inhibitors were injected. Result: the fear memory was erased, as if it had never consolidated.

The implications are profound. Every time you recall a memory, it enters a window of reconsolidation where it can be updated, strengthened, or weakened. Memory is not static storage - it's dynamic reconstruction.

## Boundary Conditions

Reconsolidation doesn't happen every time you retrieve. Nader and colleagues identified specific conditions:

**Boundary Condition 1: Memory Age**
Lee (2009) showed recent memories (hours to days) reconsolidate readily. Very old memories (months to years) become resistant. The transition point varies by memory type and strength.

**Boundary Condition 2: Retrieval Duration**
Pedreira & Maldonado (2003) found brief retrieval triggers reconsolidation, but prolonged retrieval triggers extinction instead. The transition is sharp: 1-5 minutes often triggers reconsolidation, 10+ minutes triggers extinction.

**Boundary Condition 3: Novelty/Mismatch**
Sevenster et al. (2013) demonstrated reconsolidation requires prediction error or new information during retrieval. Simply reactivating an unchanged memory doesn't make it labile.

**Boundary Condition 4: Memory Strength**
Wang & Morris (2010) showed weak memories reconsolidate more readily than strong ones. Overlearned memories resist reconsolidation.

## Temporal Dynamics

Nader et al. (2000) characterized the reconsolidation window:
- Lability onset: immediate upon retrieval
- Maximum vulnerability: 0-3 hours post-retrieval
- Window closure: 6-8 hours post-retrieval
- Return to stability: 12-24 hours

During this window, the memory can be:
- Enhanced (by additional encoding)
- Updated (with new information)
- Weakened (by interference)
- Disrupted (by protein synthesis inhibition in biological systems)

## Computational Implementation

```rust
pub struct ReconsolidationWindow {
    node_id: NodeId,
    retrieval_time: Instant,
    original_strength: f32,
    lability_factor: f32,  // 0.0 = stable, 1.0 = fully labile
}

pub struct ReconsolidationEngine {
    active_windows: Vec<ReconsolidationWindow>,
    max_lability_hours: f32,      // 3 hours
    window_close_hours: f32,      // 6-8 hours
}

impl ReconsolidationEngine {
    pub fn check_reconsolidation_trigger(
        &self,
        memory: &Memory,
        retrieval_context: &Context,
    ) -> bool {
        // Boundary 1: Memory age (not too old)
        let age_days = memory.age().as_secs_f32() / 86400.0;
        if age_days > 90.0 {  // 3 months threshold
            return false;
        }

        // Boundary 2: Retrieval duration (brief, not extended)
        let retrieval_duration_secs = retrieval_context.duration.as_secs_f32();
        if retrieval_duration_secs < 60.0 || retrieval_duration_secs > 300.0 {
            return true;  // 1-5 minute sweet spot
        } else {
            return false;
        }

        // Boundary 3: Prediction error (novelty present)
        let prediction_error = self.compute_prediction_error(memory, retrieval_context);
        if prediction_error < 0.2 {
            return false;  // Too little novelty
        }

        // Boundary 4: Memory strength (not overlearned)
        if memory.strength > 5.0 {  // Threshold for overlearning
            return false;
        }

        true  // All conditions met
    }

    pub fn open_reconsolidation_window(&mut self, node_id: NodeId, strength: f32) {
        self.active_windows.push(ReconsolidationWindow {
            node_id,
            retrieval_time: Instant::now(),
            original_strength: strength,
            lability_factor: 1.0,  // Fully labile initially
        });
    }

    pub fn compute_lability(&self, window: &ReconsolidationWindow) -> f32 {
        let elapsed_hours = window.retrieval_time.elapsed().as_secs_f32() / 3600.0;

        if elapsed_hours < self.max_lability_hours {
            // Maximum lability phase
            1.0
        } else if elapsed_hours < self.window_close_hours {
            // Gradual closing
            let closure_progress = (elapsed_hours - self.max_lability_hours) /
                                  (self.window_close_hours - self.max_lability_hours);
            1.0 - closure_progress
        } else {
            // Window closed
            0.0
        }
    }

    pub fn modify_labile_memory(
        &mut self,
        node_id: NodeId,
        modifier: MemoryModifier,
    ) -> Result<(), Error> {
        let window = self.active_windows.iter_mut()
            .find(|w| w.node_id == node_id)
            .ok_or(Error::NotInReconsolidation)?;

        let lability = self.compute_lability(window);

        if lability < 0.1 {
            return Err(Error::WindowClosed);
        }

        // Apply modification proportional to lability
        match modifier {
            MemoryModifier::Strengthen(amount) => {
                window.original_strength += amount * lability;
            }
            MemoryModifier::Weaken(amount) => {
                window.original_strength -= amount * lability;
            }
            MemoryModifier::Update(new_associations) => {
                // Update only affects labile portion
                // Implementation depends on specific memory update mechanism
            }
        }

        Ok(())
    }
}
```

## Integration with Consolidation

Reconsolidation must integrate with the existing consolidation pipeline (M6):

1. **Retrieval triggers check:** When memory is retrieved, check boundary conditions
2. **Window opening:** If conditions met, open reconsolidation window
3. **Pipeline re-entry:** Labile memory re-enters STM→LTM consolidation process
4. **Modified consolidation:** Consolidation strength modulated by lability factor
5. **Window closure:** After 6-8 hours, memory returns to stable state

## Validation Criteria

**Target: Nader et al. (2000) and follow-up studies**

- Reconsolidation trigger rate: 30-50% of retrievals (depends on boundary conditions)
- Lability window: peak at 0-3h, closes by 6-8h
- Modification sensitivity: 2-3x higher during window than outside
- Boundary accuracy: age < 90 days, duration 1-5 min, strength < threshold

**Statistical Requirements:**
- N >= 1000 retrieval trials
- Test modification during vs outside window
- Effect size: d = 0.8-1.2 (large effects expected)
- Significance: p < 0.001

## Performance Budget

- Boundary condition check: < 5μs
- Window management: < 10μs
- Lability computation: < 1μs
- Total overhead: < 20μs per retrieval
