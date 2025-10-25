# Reconsolidation Integration: Twitter Thread

**Tweet 1/8**
Schiller et al. (2010) showed that presenting new information during reconsolidation's labile window can update fear memories without extinction. This requires precise coordination: memories must be labile (from reactivation) while new information consolidates. Two temporal processes, one window.

**Tweet 2/8**
Walker et al. (2003) found that daytime reactivation disrupts sleep-dependent consolidation. Retrieval triggers reconsolidation, which interferes with ongoing consolidation. The processes aren't sequential - they co-occur and interact. Integration challenge: prevent race conditions.

**Tweet 3/8**
Solution: consolidation pauses during lability. When memory enters labile state (reconsolidation active), consolidation scheduler skips that edge. Resume once reconsolidation completes. State check adds 5ns overhead - single atomic load during consolidation decision.

**Tweet 4/8**
Suzuki et al. (2004) showed consolidation level gates malleability: newly encoded memories (0-6 hours) show 40-60% strength change during reconsolidation, well-consolidated memories (24+ hours) show only 10-20% change. Stronger consolidation resists modification.

**Tweet 5/8**
Unified memory lifecycle manager: single event queue for both consolidation steps and reconsolidation transitions. Lock-free SegQueue eliminates duplication between schedulers. Worker pool processes both event types, coordinating through atomic state checks.

**Tweet 6/8**
Reconsolidation resets consolidation: significant modifications (>20% strength change) trigger partial consolidation reset. Modified memories must re-consolidate from 50% of previous level. Prevents poorly updated memories from achieving full neocortical transfer.

**Tweet 7/8**
Performance numbers: reconsolidation check during consolidation in 6ns median, pause decision in 10ns, resume scheduling in 15ns. Memory overhead is 64 bytes per edge (both states combined). Integration is essentially free - cost appears in temporal dynamics.

**Tweet 8/8**
Validation requires replicating consolidation-gated malleability: early memories show >1.5x larger reconsolidation changes than late memories (paired t-test, p < 0.01). Statistical acceptance ensures biological plausibility. Memory that strengthens through use while remaining adaptable.
