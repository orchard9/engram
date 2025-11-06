# Task Creation Guide - Remaining Tasks

This guide provides templates for creating the remaining 10 task files (003-005, 007, 009-014) following the established patterns.

## Completed Task Files (Reference)

- ✅ **001_zero_overhead_metrics_pending.md** - Metrics infrastructure foundation
- ✅ **002_semantic_priming_pending.md** - Semantic priming engine
- ✅ **006_reconsolidation_core_pending.md** - Reconsolidation with exact boundaries
- ✅ **008_drm_false_memory_pending.md** - DRM paradigm validation

## Remaining Task Files (To Create)

### Phase 2: Cognitive Patterns

**003_associative_repetition_priming_pending.md** (2 days, P1)
- Associative priming via co-occurrence learning (McKoon & Ratcliff 1992)
- Repetition priming via exposure counting
- Integration with semantic priming (Task 002)
- Files: `/engram-core/src/cognitive/priming/{associative,repetition}.rs`

**004_proactive_interference_pending.md** (2 days, P0)
- Old memories interfere with new learning (Underwood 1957)
- Validation target: 20-30% accuracy reduction with 5+ prior lists
- Similarity-based interference detection
- Files: `/engram-core/src/cognitive/interference/proactive.rs`

**005_retroactive_fan_effect_pending.md** (2 days, P1)
- Retroactive interference (McGeoch 1942): 15-25% accuracy reduction
- Fan effect (Anderson 1974): 50-150ms RT increase per association
- Combined interference validation
- Files: `/engram-core/src/cognitive/interference/{retroactive,fan_effect}.rs`

**007_reconsolidation_integration_pending.md** (2 days, P1)
- Integrate reconsolidation (Task 006) with consolidation system (M6)
- Reconsolidated memories re-enter consolidation pipeline
- No conflicts between processes
- Files: `/engram-core/src/cognitive/reconsolidation/consolidation_integration.rs`

### Phase 3: Psychology Validation

**009_spacing_effect_validation_pending.md** (1 day, P1)
- Massed vs distributed practice (Cepeda et al. 2006)
- Validation target: 20-40% retention improvement
- Uses existing decay functions (M4)
- Files: `/engram-core/tests/psychology/spacing_effect.rs`

**010_interference_validation_suite_pending.md** (1 day, P1)
- Comprehensive validation: PI, RI, fan effect
- Statistical comparison to published data
- Per-phenomenon acceptance criteria
- Files: `/engram-core/tests/psychology/interference_validation.rs`

### Phase 4: Observability

**011_cognitive_tracing_pending.md** (2 days, P1)
- Structured event tracing for cognitive dynamics
- JSON export for visualization tools
- Priming, interference, reconsolidation events
- Files: `/engram-core/src/tracing/cognitive_events.rs`

**012_grafana_dashboard_pending.md** (1 day, P2)
- Prometheus/Grafana integration
- Panels: priming rates, interference magnitudes, reconsolidation hit rate
- Query optimization <100ms
- Files: `/docs/operations/grafana/cognitive_patterns_dashboard.json`

**013_integration_performance_pending.md** (2 days, P0)
- Integration testing: all patterns work together
- Performance validation: <1% overhead on production workload
- 10-minute soak test, no memory leaks
- Files: `/engram-core/tests/integration/cognitive_patterns_integration.rs`

**014_documentation_runbook_pending.md** (1 day, P2)
- API reference with psychology citations
- Explanation of biological foundations
- Operational tuning guide
- Files: `/docs/{reference,explanation,operations}/`

## Task File Template

```markdown
# Task XXX: [Task Name]

**Status:** Pending
**Priority:** P0/P1/P2
**Estimated Effort:** X days
**Dependencies:** [Task numbers or milestone]

## Objective

[1-2 paragraphs describing what this task delivers and why it matters]

## Integration Points

**Creates:**
- [New files with absolute paths]

**Uses:**
- [Existing files this depends on]

**Extends:**
- [Files to be modified]

## Detailed Specification

### 1. [Component Name]

```rust
// Code example showing API design and key types
```

### 2. [Another Component]

```rust
// More implementation details
```

### 3. Validation Tests

```rust
// Test examples showing acceptance criteria
```

## Acceptance Criteria

1. **[Category]:**
   - Specific measurable criteria
   - Empirical targets with tolerances
   - References to research papers

2. **Functional Requirements:**
   - What must work
   - Edge cases handled

3. **Performance:**
   - Latency budgets
   - Memory usage limits

4. **Testing:**
   - Test coverage requirements
   - Validation approach

## Testing Strategy

```bash
# Commands to run tests
cargo test ...
```

## Performance Requirements

- [Specific latency/throughput/memory budgets]

## Follow-ups

- [Tasks that depend on this]
- [Future work beyond scope]
```

## File Naming Convention

Format: `XXX_task_name_pending.md`

- XXX: Three-digit task number (001-014)
- task_name: Lowercase with underscores
- Status suffix: `_pending`, `_in_progress`, `_complete`

Examples:
- `003_associative_repetition_priming_pending.md`
- `010_interference_validation_suite_pending.md`

## Key Elements to Include

### 1. Empirical Grounding

Every cognitive task must cite specific research:
- Paper citation (Author Year)
- Empirical target (e.g., "60% false recall")
- Tolerance (e.g., "±10%")
- Statistical requirements (n, α, power)

### 2. Precise Specifications

Avoid vague requirements:
- ❌ "Should detect interference"
- ✅ "Detect proactive interference when similarity >0.7 and time_diff <24h"

### 3. Integration Points

Specify exact file paths:
- ❌ "Extends the priming module"
- ✅ "Extends `/engram-core/src/cognitive/priming/mod.rs`"

### 4. Acceptance Criteria

Must be measurable:
- ❌ "Works well"
- ✅ "P99 latency <100μs, verified by criterion benchmark"

## Creating a New Task File

1. **Copy template** from this guide
2. **Fill in specifics** from MILESTONE_13_SPECIFICATION.md
3. **Add code examples** showing API design
4. **Define tests** demonstrating acceptance criteria
5. **Cite research** for empirical validation
6. **Specify performance** budgets and constraints
7. **Review** against completed tasks for consistency

## Validation Checklist

Before considering a task file complete:

- [ ] Title and metadata complete (status, priority, effort, dependencies)
- [ ] Objective clearly states deliverable and importance
- [ ] Integration points specify exact file paths
- [ ] Code examples show key APIs and types
- [ ] Empirical targets cited from research papers
- [ ] Acceptance criteria are measurable
- [ ] Testing strategy includes commands to run
- [ ] Performance requirements quantified
- [ ] Follow-ups identify dependent tasks

## Psychology Paper Reference

Key papers to cite (from MILESTONE_13_SPECIFICATION.md):

- **Priming:** Neely (1977), Collins & Loftus (1975), McKoon & Ratcliff (1992)
- **Interference:** Anderson (1974), Underwood (1957), McGeoch (1942)
- **Spacing:** Cepeda et al. (2006), Bjork & Bjork (1992)
- **False Memory:** Roediger & McDermott (1995), Brainerd & Reyna (2002)
- **Reconsolidation:** Nader et al. (2000), Lee (2009)

Full bibliography in MILESTONE_13_SPECIFICATION.md Section 12.

## Need Help?

Consult these resources:

1. **MILESTONE_13_SPECIFICATION.md** - Complete technical details
2. **Completed task files** (001, 002, 006, 008) - Pattern examples
3. **PLANNING_SUMMARY.md** - Risk analysis and implementation guidance
4. **Agents:**
   - `memory-systems-researcher` - Biological plausibility
   - `rust-graph-engine-architect` - API design review
   - `verification-testing-lead` - Test design validation

---

**Goal:** Create 10 task files with same rigor as existing 4. Each task should be independently implementable by a competent Rust engineer without ambiguity.
