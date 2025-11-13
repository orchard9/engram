# Task 025: DRM False Memory Paradigm Validation

**Status**: Pending
**Duration**: 4 days
**Priority**: Medium
**Dependencies**: M17 Tasks 001-009 (dual memory types, spreading activation, blended recall)

## Objective

Validate that Engram's dual memory architecture reproduces the Deese-Roediger-McDermott (DRM) false memory effect through schema-based reconstruction and semantic spreading activation. Target: 40-55% false recall of critical lures that were never presented.

## Background

The DRM paradigm (Roediger & McDermott, 1995) demonstrates how semantic memory structures create false memories. When participants study word lists strongly associated with a non-presented "critical lure" (e.g., bed, rest, awake → SLEEP), they falsely remember the lure at rates comparable to veridical recall.

This validates that Engram's concept formation and spreading activation create schema-consistent biases during memory reconstruction, a hallmark of episodic-semantic interaction.

## Key Metrics

| Metric | Target Range | Source |
|--------|--------------|--------|
| **Critical lure false recall** | 40-55% | Roediger & McDermott Table 2 |
| **Veridical recall (studied items)** | 60-75% | Standard DRM data |
| **Related lure false alarm** | 20-30% | Unpresented but semantically related |
| **Unrelated lure false alarm** | <5% | Baseline false alarm rate |
| **Correlation with published data** | r > 0.70 | Across 15 DRM lists |

## Validation Criteria

### Pass Criteria
1. Critical lure false recall rate falls within 40-55% across all 15 lists
2. Pearson correlation r > 0.70 with Roediger & McDermott (1995) Table 2 data
3. Schema-consistent bias evident: related lures > unrelated lures by >15%
4. Veridical recall in 60-75% range (memory system functioning normally)
5. Performance regression <5% from M17 baseline

### Fail Criteria
- Critical lure rate <35% or >60% (mechanism too weak or too strong)
- Correlation r < 0.60 (poor fit to empirical data)
- Unrelated lure false alarms >10% (broken semantic filtering)
- Veridical recall <50% (overall memory system impaired)

## Test Design

### Study Phase
1. Present 15 DRM word lists (12 words each, 180 total)
2. Encoding: Create episodic memories with vector embeddings
3. Consolidation: Allow concept formation to extract semantic structure
4. Critical lures never presented but highly associated (e.g., SLEEP for bed/rest/awake list)

### Test Phase
1. Present 60 test items:
   - 15 studied words (veridical items)
   - 15 critical lures (non-studied, high association)
   - 15 related lures (non-studied, moderate association)
   - 15 unrelated lures (non-studied, no association)
2. Query: "Was this word in the study list?"
3. Measure: Recognition confidence (0.0-1.0)
4. Classify: >0.5 confidence = "yes" response

### Mechanism Validation
- **Spreading activation**: Critical lures activate during study via semantic spreading
- **Schema reconstruction**: Blended recall fills gaps with schema-consistent items
- **Confidence inflation**: Repeated activation during study inflates subjective confidence
- **Episode verification**: True episodes have higher activation than lures

## DRM Word Lists

Use standard 15 lists from Roediger & McDermott (1995):
1. SLEEP (bed, rest, awake, tired...)
2. CHAIR (table, sit, legs, desk...)
3. DOCTOR (nurse, sick, medicine, health...)
4. WINDOW (door, glass, pane, shade...)
5. SWEET (sour, candy, sugar, bitter...)
6. MOUNTAIN (hill, valley, climb, summit...)
7. SMELL (nose, breathe, aroma, hear...)
8. SOFT (hard, light, pillow, loud...)
9. COLD (hot, snow, warm, winter...)
10. NEEDLE (thread, pin, eye, sewing...)
11. SLOW (fast, lethargic, stop, listless...)
12. SMOKE (cigarette, puff, blaze, billows...)
13. TRASH (garbage, waste, can, refuse...)
14. ROUGH (smooth, bumpy, road, tough...)
15. ANGER (mad, fear, hate, rage...)

Dataset file: `engram-core/tests/cognitive_validation/datasets/drm_word_lists.json`

## Implementation Approach

### Test Structure
```rust
// engram-core/tests/cognitive_validation/drm_false_memory.rs

#[test]
fn test_drm_false_memory_paradigm() {
    // 1. Setup: Create graph with dual memory
    let graph = Arc::new(MemoryGraph::new());

    // 2. Study phase: Encode 15 DRM lists
    for list in drm_lists() {
        for word in list.studied_words {
            store_episode(graph, word, timestamp);
        }
        // Allow consolidation window
        advance_time(5.minutes);
    }

    // 3. Test phase: Recognition for 60 items
    let results = test_recognition(
        studied_items,   // 15 veridical
        critical_lures,  // 15 critical (high association)
        related_lures,   // 15 related (moderate)
        unrelated_lures  // 15 unrelated (baseline)
    );

    // 4. Validate metrics
    assert_in_range(results.critical_lure_rate, 0.40, 0.55);
    assert_gt(results.veridical_recall, 0.60);
    assert_lt(results.unrelated_false_alarms, 0.05);
    assert_correlation(results, roediger_mcdermott_data(), 0.70);
}
```

### Key Operations
1. **Embedding generation**: Sentence-transformers for word vectors
2. **Semantic clustering**: Critical lures cluster with studied items in concept space
3. **Blended recall**: Query activates both episodes and concepts
4. **Confidence scoring**: Activation strength → recognition confidence

## Statistical Analysis

### Primary Analysis
```rust
// Compute false memory rate for critical lures
let critical_lure_rate = critical_lure_yes / 15.0;

// Correlation with published data
let correlation = pearson_correlation(
    engram_rates_per_list,
    roediger_mcdermott_table2
);

// Effect size for lure type difference
let cohens_d = (critical_lure_rate - unrelated_lure_rate)
    / pooled_std_dev;
```

### Expected Results
- **Critical lures**: M = 0.47, SD = 0.12 (Roediger & McDermott)
- **Related lures**: M = 0.25, SD = 0.08
- **Unrelated lures**: M = 0.03, SD = 0.02
- **Effect size**: d > 3.0 (critical vs unrelated)

## Performance Requirements

- **Test duration**: <30 minutes for full 15-list paradigm
- **Memory usage**: <500MB for 180 episodes + concepts
- **Latency**: <100ms per recognition query
- **Regression**: <5% from M17 baseline

## Deliverables

1. Test implementation: `drm_false_memory.rs`
2. Dataset file: `drm_word_lists.json` with 15 lists
3. Statistical analysis module: Cohen's d, correlation, confidence intervals
4. Validation report: Markdown with graphs comparing Engram vs published data
5. Performance metrics: Latency, memory, regression check

## Success Validation

Run test with:
```bash
cargo test --test cognitive_validation drm_false_memory -- --nocapture

# Output should show:
# Critical lure false recall: 47% (target: 40-55%) ✓
# Veridical recall: 68% (target: 60-75%) ✓
# Correlation with R&M 1995: r = 0.73 (target: >0.70) ✓
# Performance regression: 2.1% (target: <5%) ✓
# PASSED
```

## References

### Primary Paper
- Roediger, H. L., & McDermott, K. B. (1995). Creating false memories: Remembering words not presented in lists. *Journal of Experimental Psychology: Learning, Memory, and Cognition*, 21(4), 803-814.

### Theoretical Context
- Deese, J. (1959). On the prediction of occurrence of particular verbal intrusions in immediate recall. *Journal of Experimental Psychology*, 58(1), 17-22.
- Gallo, D. A. (2010). False memories and fantastic beliefs: 15 years of the DRM illusion. *Memory & Cognition*, 38(7), 833-848.
- McClelland, J. L., et al. (1995). Why there are complementary learning systems in the hippocampus and neocortex. *Psychological Review*, 102(3), 419-457.

### Related Work
- Schacter, D. L., et al. (1998). False recognition and the right frontal lobe. *Neuropsychologia*, 36(4), 313-323.
- Roediger, H. L., et al. (2001). Factors that determine false recall: A multiple regression analysis. *Psychonomic Bulletin & Review*, 8(3), 385-407.

## Integration with M17

This task validates:
- **M17 Task 004**: Concept formation creates semantic schemas
- **M17 Task 005**: Binding strength influences false memory rates
- **M17 Task 009**: Blended recall combines episodes + concepts
- **M17 Task 007**: Spreading activation during encoding creates lure activation

False memories emerge naturally from dual memory architecture without explicit false memory mechanisms. This demonstrates the system captures core memory reconstruction phenomena.

## Task Completion Checklist

- [ ] Implement test in `drm_false_memory.rs`
- [ ] Create `drm_word_lists.json` with 15 standard lists
- [ ] Run test with M17 performance baseline check
- [ ] Validate critical lure rate 40-55%
- [ ] Validate correlation r > 0.70 with published data
- [ ] Generate validation report with graphs
- [ ] Run `make quality` - zero clippy warnings
- [ ] Update task file: `_pending` → `_in_progress` → `_complete`
- [ ] Commit with validation results in message
