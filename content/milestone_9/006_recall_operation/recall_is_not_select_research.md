# Research: RECALL Is Not SELECT

## Research Questions

1. How does semantic recall differ from SQL SELECT fundamentally?
2. Why do cognitive queries return distributions, not rows?
3. How are confidence intervals first-class in probabilistic query results?
4. What role do evidence chains play in explaining recall results?
5. How does Engram balance recall completeness vs precision?

## Key Findings

### 1. Semantic Recall vs SQL SELECT

**SQL SELECT: Deterministic Set Retrieval**
```sql
SELECT * FROM memories WHERE confidence > 0.7;
-- Returns: Exactly those rows where confidence > 0.7
-- Semantics: Boolean filtering (in or out)
-- Guarantee: Repeatable, deterministic
```

**RECALL: Probabilistic Memory Retrieval**
```
RECALL episode WHERE confidence > 0.7
-- Returns: Episodes with probability of matching
-- Semantics: Confidence-weighted ranking
-- Guarantee: Best estimate given uncertainty
```

**Fundamental Differences**:

1. **Data Model**:
   - SQL: Discrete rows with exact values
   - RECALL: Continuous confidence distributions

2. **Filtering**:
   - SQL: Boolean (row matches or doesn't)
   - RECALL: Probabilistic (confidence that row matches)

3. **Results**:
   - SQL: Set of rows (unordered by default)
   - RECALL: Ranked distribution (ordered by confidence)

4. **Uncertainty**:
   - SQL: Not represented (assumes perfect knowledge)
   - RECALL: First-class (confidence intervals, uncertainty sources)

### 2. Confidence as First-Class Concept

**In SQL**: Confidence is just another column
```sql
CREATE TABLE memories (
    id UUID,
    content TEXT,
    confidence FLOAT
);

SELECT * FROM memories WHERE confidence > 0.7;
-- confidence is data, not metadata
```

**In RECALL**: Confidence is intrinsic to results
```rust
pub struct ProbabilisticQueryResult {
    pub episodes: Vec<(Episode, Confidence)>, // Confidence paired with data
    pub aggregate_confidence: ConfidenceInterval, // [lower, upper]
    pub uncertainty_sources: Vec<UncertaintySource>,
}
```

**Why This Matters**:
- SQL treats confidence as user data (can be wrong, stale, arbitrary)
- RECALL computes confidence from system state (temporal decay, retrieval ambiguity)
- SQL confidence doesn't compose (joining confident rows = ???)
- RECALL confidence propagates correctly (Bayesian update rules)

**Example**: Temporal Decay
```rust
// Episode stored 3 days ago with initial confidence 0.9
// Decay rate: 0.05 per day
// Current confidence: 0.9 * exp(-0.05 * 3) ≈ 0.77

// SQL: confidence column still says 0.9 (stale)
// RECALL: dynamically computes 0.77 at query time
```

### 3. Evidence Chains for Explainability

**SQL EXPLAIN**: Shows execution plan
```sql
EXPLAIN ANALYZE SELECT * FROM memories WHERE confidence > 0.7;

Seq Scan on memories  (cost=0.00..10.75 rows=5 width=32)
  Filter: (confidence > 0.7::double precision)
  Rows Removed by Filter: 3
```

**Tells you**: HOW the query ran (scan type, rows filtered)
**Doesn't tell you**: WHY specific rows returned, what their confidence means

**RECALL Evidence**: Shows result provenance
```rust
Evidence::Confidence(ConfidenceEvidence {
    base_confidence: 0.9,
    uncertainty_sources: vec![
        UncertaintySource::TemporalDecay { age: 3 days, decay_rate: 0.05 },
        UncertaintySource::RetrievalAmbiguity { similar_nodes: 5 },
    ],
    calibrated_confidence: 0.77,
    method: CalibrationMethod::Platt,
})
```

**Tells you**: WHY episode returned, how confidence computed, what uncertainties affect it

**Key Difference**: SQL optimizes for performance. RECALL optimizes for explainability.

### 4. Constraint Application

**SQL Constraints**: Boolean predicates
```sql
WHERE confidence > 0.7
  AND created_at > '2024-01-01'
  AND content LIKE '%meeting%'
```
Result: Rows where ALL constraints TRUE

**RECALL Constraints**: Confidence-weighted filters
```rust
pub enum Constraint {
    ConfidenceAbove(Confidence),
    CreatedAfter(SystemTime),
    SimilarTo { embedding: Vec<f32>, threshold: f32 },
}
```

**Semantic Similarity Constraint**:
```rust
Constraint::SimilarTo {
    embedding: query_embedding,
    threshold: 0.8,
}

// Not boolean! Returns confidence that episode is similar
// Confidence = cosine_similarity(episode.embedding, query_embedding)
// Filter: Keep if confidence >= threshold
```

**Temporal Constraint with Decay**:
```rust
Constraint::CreatedAfter(cutoff_time)

// Accounts for decay: older episodes less confident
// Not just "after cutoff or not" - gradual confidence decrease
```

**Key Insight**: RECALL constraints are probabilistic filters, not boolean predicates.

### 5. Completeness vs Precision Trade-off

**SQL**: Exact match semantics
- Returns all rows matching predicate
- No false positives (if predicate correct)
- No false negatives (if data complete)

**RECALL**: Best-effort retrieval
- Returns high-confidence matches
- May have false positives (low confidence included)
- May have false negatives (high confidence excluded due to ambiguity)

**Configurability**:
```rust
pub struct RecallQuery {
    pub confidence_threshold: Option<Confidence>, // Higher = more precision
    pub max_results: Option<usize>, // Limit for performance
    pub base_rate: Option<Confidence>, // Bayesian prior
}
```

**Tuning Examples**:
```
// High precision, low recall
RECALL episode WHERE confidence > 0.9

// High recall, low precision  
RECALL episode WHERE confidence > 0.5

// Balanced (default)
RECALL episode WHERE confidence > 0.7
```

**Unlike SQL**: User explicitly chooses precision/recall balance via confidence threshold.

### 6. Integration with ProbabilisticQueryExecutor

**From Milestone 6**: Calibrated confidence scores
```rust
pub trait ProbabilisticQueryExecutor {
    fn execute(
        &self,
        candidates: Vec<Episode>,
        activation_paths: &[ActivationPath],
        uncertainty_sources: Vec<UncertaintySource>,
    ) -> ProbabilisticQueryResult;
}
```

**RECALL uses this**:
```rust
impl QueryExecutor {
    fn execute_recall(
        &self,
        query: RecallQuery,
        space: &SpaceHandle,
    ) -> Result<ProbabilisticQueryResult> {
        // 1. Recall candidates
        let episodes = space.store.recall(&cue)?;

        // 2. Apply constraints (confidence filtering)
        let filtered = self.apply_constraints(episodes, &query.constraints)?;

        // 3. Probabilistic query execution (calibration)
        let result = self.probabilistic_executor.execute(
            filtered,
            &[], // No activation paths for direct recall
            vec![
                UncertaintySource::RetrievalAmbiguity {
                    similar_nodes: filtered.len(),
                },
            ],
        );

        Ok(result)
    }
}
```

**Calibration Benefits**:
- Raw confidence scores may be overconfident
- Platt scaling adjusts based on historical accuracy
- Result: Confidence = actual probability of correctness

## Synthesis

RECALL is fundamentally different from SELECT:

1. **Paradigm**: Probabilistic retrieval vs boolean filtering
2. **Results**: Confidence distributions vs discrete rows
3. **Uncertainty**: First-class vs absent
4. **Explainability**: Evidence chains vs execution plans
5. **Trade-offs**: User-controlled precision/recall vs exact match

This isn't just a syntactic difference. It's a different way of thinking about memory retrieval.

## References

1. "Probabilistic Databases", Suciu et al. (2011)
2. "Confidence Calibration in Deep Learning", Guo et al. (2017)
3. "Information Retrieval: Precision vs Recall", Manning et al. (2008)
4. "Bayesian Query Processing", Dalvi & Suciu (2007)
