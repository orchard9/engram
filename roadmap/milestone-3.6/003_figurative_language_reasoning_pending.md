# Task 003: Figurative Language Interpretation Engine

## Objective
Detect and interpret metaphors, similes, idioms, and analogies in queries and stored memories, mapping them to concrete semantic cues with explainable confidence while avoiding unsupported hallucinations.

## Priority
P1 (High Impact)

## Effort Estimate
2.5 days

## Dependencies
- Task 001: Multilingual Embedding Pipeline (vector semantics + language metadata)
- Task 002: Semantic Query Expansion & Recall Routing (normalized query contexts)

## Current Baseline
- No figurative-language detection; metaphors are treated as literal tokens and usually miss.
- No mechanism to attach interpretation metadata to cues or results.
- Existing `Cue` and `Episode` structures (engram-core/src/memory.rs) lack figurative annotation fields.
- `QueryContext` from Task 002 needs extension to carry figurative interpretations.

## Implementation Plan

### 1. Core Types and Structures

Create `engram-core/src/semantics/mod.rs` with:

```rust
pub mod figurative;
```

Create `engram-core/src/semantics/figurative.rs`:

```rust
use crate::Confidence;
use compact_str::CompactString;
use serde::{Deserialize, Serialize};
use std::sync::Arc;

/// Types of figurative language patterns
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum FigurativeType {
    Simile,
    Metaphor,
    Idiom,
    Analogy,
}

/// Detected figurative span in text
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FigurativeSpan {
    pub start: usize,
    pub end: usize,
    pub figurative_type: FigurativeType,
    pub confidence: Confidence,
    pub text: CompactString,
}

/// Literal interpretation of figurative language
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LiteralInterpretation {
    pub concept: CompactString,
    pub confidence: Confidence,
    pub rationale: &'static str, // Zero-copy from static lexicon
}

/// Complete figurative interpretation with multiple literal candidates
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FigurativeInterpretation {
    pub figurative_form: CompactString,
    pub literal_concepts: Vec<LiteralInterpretation>,
    pub language: LanguageCode,
    pub confidence: Confidence,
}

/// Language code for per-language processing
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum LanguageCode {
    EN,
    ES,
    ZH,
    // Add more as needed
}
```

### 2. Rule-Based Detector

Add to `engram-core/src/semantics/figurative.rs`:

```rust
use dashmap::DashMap;
use once_cell::sync::Lazy;

/// Static pattern database for explicit figurative language
static EXPLICIT_PATTERNS: Lazy<Vec<FigurativePattern>> = Lazy::new(|| {
    vec![
        FigurativePattern { marker: "like a", figurative_type: FigurativeType::Simile },
        FigurativePattern { marker: "as ", figurative_type: FigurativeType::Simile },
        // ... more patterns
    ]
});

pub struct FigurativePattern {
    pub marker: &'static str,
    pub figurative_type: FigurativeType,
}

/// Rule-based detector for explicit patterns
pub struct RuleBasedDetector {
    patterns: &'static [FigurativePattern],
    idiom_maps: Arc<DashMap<LanguageCode, &'static IdiomsLexicon>>,
}

impl RuleBasedDetector {
    pub fn new() -> Self {
        Self {
            patterns: &EXPLICIT_PATTERNS,
            idiom_maps: Arc::new(DashMap::new()),
        }
    }

    pub fn detect(&self, query: &str, lang: LanguageCode) -> Vec<FigurativeSpan> {
        let mut spans = Vec::new();

        // Fast path: explicit markers
        for pattern in self.patterns {
            if let Some(pos) = query.find(pattern.marker) {
                spans.push(FigurativeSpan {
                    start: pos,
                    end: pos + pattern.marker.len(),
                    figurative_type: pattern.figurative_type,
                    confidence: Confidence::exact(0.95),
                    text: query[pos..pos + pattern.marker.len()].into(),
                });
            }
        }

        // Idiom lexicon lookup
        if let Some(lexicon) = self.idiom_maps.get(&lang) {
            for idiom in lexicon.idioms {
                if let Some(pos) = query.find(idiom.text) {
                    spans.push(FigurativeSpan {
                        start: pos,
                        end: pos + idiom.text.len(),
                        figurative_type: FigurativeType::Idiom,
                        confidence: Confidence::exact(0.90),
                        text: idiom.text.into(),
                    });
                }
            }
        }

        spans
    }
}
```

### 3. Figurative Mapper with Lock-Free Design

Add to `engram-core/src/semantics/figurative.rs`:

```rust
use dashmap::DashMap;

#[cfg(feature = "hnsw_index")]
use crate::index::CognitiveHnswIndex;

pub struct FigurativeMapper {
    /// Lock-free idiom-to-concept mappings
    idiom_map: Arc<DashMap<CompactString, Vec<LiteralInterpretation>>>,

    /// HNSW index for embedding-based similarity (optional)
    #[cfg(feature = "hnsw_index")]
    concept_index: Arc<CognitiveHnswIndex>,
}

impl FigurativeMapper {
    pub fn new() -> Self {
        let idiom_map = Arc::new(DashMap::new());

        // Load curated mappings
        Self::load_default_idioms(&idiom_map);

        Self {
            idiom_map,
            #[cfg(feature = "hnsw_index")]
            concept_index: Arc::new(CognitiveHnswIndex::default()),
        }
    }

    fn load_default_idioms(map: &DashMap<CompactString, Vec<LiteralInterpretation>>) {
        // English idioms
        map.insert(
            "cold shoulder".into(),
            vec![LiteralInterpretation {
                concept: "ignore".into(),
                confidence: Confidence::exact(0.90),
                rationale: "idiom: cold shoulder -> ignore",
            }],
        );

        // Add more curated mappings
    }

    pub fn map_to_literal(
        &self,
        span: &FigurativeSpan,
        query: &str,
    ) -> Vec<LiteralInterpretation> {
        let figurative_text: CompactString = query[span.start..span.end].into();

        // Try exact lexicon match first (O(1) lookup)
        if let Some(interps) = self.idiom_map.get(&figurative_text) {
            return interps.clone();
        }

        // Fall back to embedding similarity if HNSW enabled
        #[cfg(feature = "hnsw_index")]
        {
            // Would use concept_index.search() here
            // For now, return empty
        }

        Vec::new()
    }
}
```

### 4. Integration with Query Context

Extend Task 002's `QueryContext` in query expansion module:

```rust
// In engram-core/src/semantics/query_expansion.rs (from Task 002)
pub struct QueryContext {
    pub raw_text: String,
    pub detected_language: LanguageCode,
    pub normalized_tokens: Vec<String>,
    pub synonyms: Vec<String>,
    pub abbreviations: Vec<String>,

    // NEW: Figurative interpretations
    pub figurative_interpretations: Vec<FigurativeInterpretation>,
}

impl QueryContext {
    /// Merge literal and figurative cues with confidence-based weighting
    pub fn merged_cues(&self) -> impl Iterator<Item = WeightedCue> + '_ {
        let literal = self.normalized_tokens.iter()
            .map(|token| WeightedCue {
                cue: Cue::semantic("literal".to_string(), token.clone(), Confidence::HIGH),
                weight: 1.0
            });

        let figurative = self.figurative_interpretations.iter()
            .filter(|fi| fi.confidence.raw() >= CONFIDENCE_THRESHOLD)
            .flat_map(|fi| fi.literal_concepts.iter())
            .map(|c| WeightedCue {
                cue: Cue::semantic("figurative".to_string(), c.concept.to_string(), c.confidence),
                weight: c.confidence.raw() * FIGURATIVE_PENALTY,
            });

        literal.chain(figurative)
    }
}

const CONFIDENCE_THRESHOLD: f32 = 0.7;
const FIGURATIVE_PENALTY: f32 = 0.85;

pub struct WeightedCue {
    pub cue: Cue,
    pub weight: f32,
}
```

### 5. Episode Metadata Extension

Extend `Episode` in `engram-core/src/memory.rs`:

```rust
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Episode {
    // ... existing fields ...

    /// Figurative language annotations (detected during ingestion)
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub figurative_annotations: Vec<FigurativeAnnotation>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FigurativeAnnotation {
    pub span: std::ops::Range<usize>,
    pub figurative_form: CompactString,
    pub literal_interpretations: smallvec::SmallVec<[LiteralInterpretation; 2]>,
}
```

### 6. Ingestion Pipeline Integration

Update `engram-core/src/store.rs` to annotate episodes during ingestion:

```rust
impl MemoryStore {
    pub fn store(&self, mut episode: Episode) -> StoreResult {
        // Detect figurative language if enabled
        if self.config.enable_figurative_detection {
            let detector = RuleBasedDetector::new();
            let mapper = FigurativeMapper::new();

            // Detect in 'what' field
            let spans = detector.detect(&episode.what, episode.language.unwrap_or(LanguageCode::EN));

            for span in spans {
                let interpretations = mapper.map_to_literal(&span, &episode.what);
                if !interpretations.is_empty() {
                    episode.figurative_annotations.push(FigurativeAnnotation {
                        span: span.start..span.end,
                        figurative_form: span.text,
                        literal_interpretations: interpretations.into(),
                    });
                }
            }
        }

        // ... rest of existing store logic ...
    }
}
```

### 7. Query Processing Integration

Update `engram-cli/src/api.rs` to process figurative language in queries:

```rust
// In recall_memories or search_memories_rest handlers
pub async fn recall_memories(
    State(state): State<ApiState>,
    Query(params): Query<RecallQuery>,
) -> Result<impl IntoResponse, ApiError> {
    // ... existing code ...

    let mut query_context = QueryContext::from_text(&query_text);

    // Add figurative detection
    if state.config.enable_figurative_detection {
        let detector = RuleBasedDetector::new();
        let mapper = FigurativeMapper::new();

        let spans = detector.detect(&query_text, query_context.detected_language);
        for span in spans {
            let interpretations = mapper.map_to_literal(&span, &query_text);
            if !interpretations.is_empty() {
                query_context.figurative_interpretations.push(FigurativeInterpretation {
                    figurative_form: span.text,
                    literal_concepts: interpretations,
                    language: query_context.detected_language,
                    confidence: span.confidence,
                });
            }
        }
    }

    // Use merged cues for recall
    let cues = query_context.merged_cues().collect::<Vec<_>>();

    // ... rest of recall logic ...
}
```

### 8. API Response Extension

Update `engram-cli/src/api.rs` response types:

```rust
#[derive(Debug, Serialize, ToSchema)]
pub struct MemoryResult {
    // ... existing fields ...

    /// Explanation if matched via figurative language
    pub figurative_match_explanation: Option<String>,
}

#[derive(Debug, Serialize, ToSchema)]
pub struct RecallResponse {
    // ... existing fields ...

    /// Figurative interpretations applied to query
    pub figurative_interpretations_applied: Vec<FigurativeMatchInfo>,
}

#[derive(Debug, Serialize, ToSchema)]
pub struct FigurativeMatchInfo {
    pub figurative_form: String,
    pub literal_interpretation: String,
    pub confidence: f32,
}
```

### 9. Configuration and Controls

Add to configuration system (config file or environment):

```toml
[figurative_language]
enabled = true
min_confidence = 0.7
enabled_languages = ["en", "es", "zh"]
disable_emergency = false  # Killswitch
```

Create `engram-core/src/config.rs` extension:

```rust
pub struct FigurativeLangConfig {
    pub enabled: bool,
    pub min_confidence: f32,
    pub enabled_languages: Vec<LanguageCode>,
}
```

### 10. Metrics and Observability

Add to `engram-core/src/metrics/mod.rs`:

```rust
pub struct FigurativeLanguageMetrics {
    pub detections_total: AtomicU64,
    pub interpretations_applied_total: AtomicU64,
    pub low_confidence_total: AtomicU64,
    pub cache_hit_ratio: AtomicF32,
}
```

Emit metrics during detection:

```rust
if interpretation.confidence.raw() < config.min_confidence {
    metrics.figurative.low_confidence_total.fetch_add(1, Ordering::Relaxed);
}
```

## File Structure

```
engram-core/src/
├── semantics/
│   ├── mod.rs              (NEW)
│   └── figurative.rs       (NEW - ~500 lines)
├── memory.rs               (MODIFY - add FigurativeAnnotation)
├── store.rs                (MODIFY - add detection in store())
└── config.rs               (MODIFY - add FigurativeLangConfig)

engram-core/resources/      (NEW directory)
├── idioms/
│   ├── en.toml            (English idioms)
│   ├── es.toml            (Spanish idioms)
│   └── zh.toml            (Chinese idioms)

engram-cli/src/
└── api.rs                  (MODIFY - query processing + response types)

engram-core/tests/
└── figurative_language_tests.rs  (NEW - integration tests)
```

## Idiom Lexicon Format

Create `engram-core/resources/idioms/en.toml`:

```toml
[[idioms]]
text = "cold shoulder"
literals = ["ignore", "rebuff"]
confidence = 0.9

[[idioms]]
text = "break the ice"
literals = ["start conversation", "initiate"]
confidence = 0.85

[[idioms]]
text = "time flies"
literals = ["passes quickly", "goes fast"]
confidence = 0.90
```

## Acceptance Criteria
- [ ] Figurative detector achieves ≥85% precision / ≥75% recall on curated multilingual figurative corpus.
- [ ] API responses include `figurative_interpretations_applied` array when interpretations contribute to recall.
- [ ] Operators can toggle figurative handling at runtime via config; disabled mode yields baseline literal behavior.
- [ ] Explainability metadata stored with each recall event for auditing.
- [ ] Integration tests pass for English, Spanish, Mandarin queries.
- [ ] P95 query latency overhead <5ms with figurative processing enabled.

## Testing Approach

### Unit Tests (engram-core/tests/figurative_language_tests.rs)

```rust
#[test]
fn test_rule_based_detection_english() {
    let detector = RuleBasedDetector::new();
    let query = "She gave me the cold shoulder";
    let spans = detector.detect(query, LanguageCode::EN);

    assert_eq!(spans.len(), 1);
    assert_eq!(spans[0].figurative_type, FigurativeType::Idiom);
    assert!(spans[0].confidence.raw() > 0.85);
}

#[test]
fn test_figurative_mapper_cold_shoulder() {
    let mapper = FigurativeMapper::new();
    let span = FigurativeSpan { /* ... */ };
    let interpretations = mapper.map_to_literal(&span, "cold shoulder");

    assert!(!interpretations.is_empty());
    assert!(interpretations.iter().any(|i| i.concept.contains("ignore")));
}

#[test]
fn test_confidence_threshold_filtering() {
    let context = QueryContext {
        figurative_interpretations: vec![
            FigurativeInterpretation { confidence: Confidence::exact(0.5), /* ... */ },
            FigurativeInterpretation { confidence: Confidence::exact(0.8), /* ... */ },
        ],
        /* ... */
    };

    let cues = context.merged_cues().collect::<Vec<_>>();
    // Should only include high-confidence interpretation
    assert_eq!(cues.iter().filter(|c| c.weight < 1.0).count(), 1);
}
```

### Integration Tests

```rust
#[tokio::test]
async fn test_figurative_query_recall() {
    let store = MemoryStore::new(/* ... */);

    // Store episode with literal content
    let episode = Episode::new(/* ... */, "She ignored me at the party", /* ... */);
    store.store(episode);

    // Query with figurative language
    let cue = Cue::semantic(
        "figurative_query".to_string(),
        "She gave me the cold shoulder".to_string(),
        Confidence::HIGH
    );

    let result = store.recall(&cue);
    assert!(!result.results.is_empty());
}
```

### Performance Benchmark

```rust
#[bench]
fn bench_figurative_detection(b: &mut Bencher) {
    let detector = RuleBasedDetector::new();
    b.iter(|| {
        detector.detect("She gave me the cold shoulder at work", LanguageCode::EN)
    });
}
```

## Risk Mitigation
- Keep figurative lexicons versioned in git; require PR review for new mappings.
- If classifier confidence uncertain, fall back to literal handling and emit monitoring event.
- Implement circuit breaker: if >10% of queries have low-confidence figurative matches, disable temporarily.
- Monitor `engram_figurative_low_confidence_total` metric; alert if exceeds threshold.

## Notes
- Use `compact_str::CompactString` for small strings to avoid heap allocations.
- Use `dashmap::DashMap` for concurrent access without locks.
- Leverage existing HNSW index (if feature enabled) for embedding-based similarity fallback.
- Coordinate with Task 001 for language detection integration.
- Coordinate with Task 002 for `QueryContext` extension.
- Consider future: ML-based implicit metaphor detection (lightweight distilled model).
- Consider future: Memory consolidation to strengthen frequently-used metaphorical mappings.

## Dependencies from chosen_libraries.md
- `compact_str` (already approved: small string optimization)
- `dashmap` (already used in codebase: lock-free concurrent hashmap)
- `smallvec` (already used: stack-allocated small vectors)
- Language detection: Coordinate with Task 001's approved library choice
