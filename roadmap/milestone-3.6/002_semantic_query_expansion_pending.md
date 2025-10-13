# Task 002: Semantic Query Expansion & Recall Routing

## Objective
Route text queries through multilingual embeddings by default, augment them with synonym/abbreviation expansion, and retain deterministic fallbacks so lexical recall stays predictable for enterprise consumers.

## Priority
P0 (Critical Path)

## Effort Estimate
2.5 days (updated after codebase analysis)

## Dependencies
- Task 001: Multilingual Embedding Pipeline (embeddings + language metadata)

## Current Baseline Analysis

### Existing Architecture
- **API Entry Point**: `search_memories_rest` (engram-cli/src/api.rs:969-1088) creates `Cue::semantic` with raw text
- **Memory Store Recall**: `MemoryStore::recall` (engram-core/src/store.rs) supports multiple cue types via enum dispatch
- **Cue Types**: `CueType` enum (engram-core/src/memory.rs:395-430) has Embedding, Context, Temporal, Semantic variants
- **CueBuilder**: Typestate pattern builder (engram-core/src/memory.rs:909-1135) enforces compile-time validation
- **Confidence System**: `Confidence` newtype (engram-core/src/lib.rs:62-257) with cognitive calibration methods

### Current Recall Flow
1. REST endpoint receives `query` parameter
2. Creates `CoreCue::semantic(query, query, Confidence::exact(0.7))`
3. `MemoryStore::recall` dispatches on `CueType::Semantic`
4. Substring/fuzzy matching on `Episode.what` field
5. Returns episodes with confidence scores

### Gaps
- No embedding-based search despite `CueType::Embedding` existing
- No synonym/abbreviation expansion infrastructure
- No query normalization pipeline
- No language detection or cross-lingual support
- No hybrid semantic+lexical scoring

## Detailed Implementation Plan

### 1. Query Expansion Infrastructure

#### 1.1 Create Query Expansion Module (`engram-core/src/query/expansion.rs`)

```rust
use crate::{Confidence, Cue, CueBuilder, CueType};
use std::sync::Arc;
use arc_swap::ArcSwap; // Already in deps via existing usage

/// Query context with expansion metadata
#[derive(Debug, Clone)]
pub struct QueryContext {
    pub raw_text: String,
    pub detected_language: Option<Language>,
    pub normalized_tokens: Vec<String>,
    pub expansions: Vec<ExpansionTerm>,
    pub original_confidence: Confidence,
}

/// Expanded term with provenance
#[derive(Debug, Clone)]
pub struct ExpansionTerm {
    pub term: String,
    pub source: ExpansionSource,
    pub confidence: Confidence,
    pub hop_count: u8,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExpansionSource {
    Original,
    Synonym,
    Abbreviation,
    EmbeddingSimilar,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Language {
    English,
    Spanish,
    Mandarin,
    // Extensible per Task 001
}

/// Query expansion engine with tiered dictionaries
pub struct QueryExpander {
    // Hot tier: in-memory dictionaries for frequent terms
    hot_dictionaries: Arc<ArcSwap<HotTierDictionaries>>,

    // Warm tier: memory-mapped files for full language coverage
    warm_dictionaries: WarmTierDictionaries,

    // Configuration
    config: ExpansionConfig,
}

/// In-memory hot-tier dictionaries (lock-free swap)
struct HotTierDictionaries {
    synonyms: HashMap<String, Vec<(String, f32)>>, // term -> [(synonym, confidence)]
    abbreviations: HashMap<String, Vec<(String, f32)>>, // abbrev -> [(expansion, confidence)]
}

/// Memory-mapped warm-tier dictionaries
struct WarmTierDictionaries {
    // Placeholder for mmap implementation
    // Will use memmap2 crate (already in chosen_libraries.md)
}

#[derive(Debug, Clone)]
pub struct ExpansionConfig {
    pub max_synonyms_per_term: usize,
    pub max_total_expansions: usize,
    pub confidence_decay_per_hop: f32,
    pub min_expansion_confidence: f32,
    pub enable_cross_lingual: bool,
}

impl Default for ExpansionConfig {
    fn default() -> Self {
        Self {
            max_synonyms_per_term: 5,
            max_total_expansions: 20,
            confidence_decay_per_hop: 0.85,
            min_expansion_confidence: 0.3,
            enable_cross_lingual: false,
        }
    }
}
```

**File Location**: `engram-core/src/query/expansion.rs`
**Integration**: Add `pub mod expansion;` to `engram-core/src/query/mod.rs`

#### 1.2 Extend `CueBuilder` for Query Expansion

Add new method to `CueBuilder` (engram-core/src/memory.rs):

```rust
impl CueBuilder<cue_builder_states::NoId> {
    /// Build cue from text query with expansion
    pub fn from_text_query(query_context: QueryContext) -> CueBuilder<cue_builder_states::Ready> {
        let id = format!("query_{}", uuid::Uuid::new_v4());

        // If embedding available from Task 001, use hybrid approach
        if let Some(embedding) = query_context.embedding {
            CueBuilder::new()
                .id(id)
                .embedding_search(embedding, query_context.original_confidence)
                .cue_confidence(query_context.original_confidence)
                .result_threshold(Confidence::exact(0.5))
                .max_results(100)
        } else {
            // Fallback to semantic with expanded terms
            let expanded_content = query_context.expansions
                .iter()
                .map(|e| e.term.as_str())
                .collect::<Vec<_>>()
                .join(" ");

            CueBuilder::new()
                .id(id)
                .semantic_search(expanded_content, query_context.original_confidence)
                .result_threshold(Confidence::exact(0.5))
                .max_results(100)
        }
    }
}
```

**File**: `engram-core/src/memory.rs` (add to existing impl block around line 950)

#### 1.3 Implement Confidence Propagation

Add methods to `Confidence` (engram-core/src/lib.rs) for expansion decay:

```rust
impl Confidence {
    /// Propagate confidence through query expansion with hop penalty
    pub fn propagate_expansion(
        self,
        similarity: f32,
        source_trust: f32,
        hop_count: u8,
        hop_decay: f32,
    ) -> Self {
        let base = self.raw() * similarity;
        let trust_factor = source_trust.clamp(0.0, 1.0);
        let hop_penalty = hop_decay.powi(hop_count as i32);

        let propagated = base * trust_factor * hop_penalty;
        Self::exact(propagated)
    }
}
```

**File**: `engram-core/src/lib.rs` (add after line 175, in Confidence impl block)

### 2. Lexical Normalization Pipeline

#### 2.1 Create Normalization Module (`engram-core/src/query/normalization.rs`)

```rust
use unicode_normalization::UnicodeNormalization; // Add to Cargo.toml

pub struct TokenNormalizer {
    // Placeholder for stemmer/lemmatizer per language
}

impl TokenNormalizer {
    pub fn normalize(&self, text: &str, language: Option<Language>) -> Vec<String> {
        // Step 1: Unicode normalization (NFKC)
        let normalized: String = text.nfkc().collect();

        // Step 2: Case folding
        let lowercased = normalized.to_lowercase();

        // Step 3: Tokenization (simple whitespace for v1)
        let tokens: Vec<String> = lowercased
            .split_whitespace()
            .map(|s| s.trim_matches(|c: char| !c.is_alphanumeric()))
            .filter(|s| !s.is_empty())
            .map(String::from)
            .collect();

        // Step 4: Language-specific processing
        match language {
            Some(Language::English) => self.stem_english(&tokens),
            _ => tokens, // Pass-through for other languages initially
        }
    }

    fn stem_english(&self, tokens: &[String]) -> Vec<String> {
        // Placeholder: integrate rust-stemmers crate (MIT licensed)
        // For v1: return as-is, add stemming in follow-up
        tokens.to_vec()
    }
}
```

**File**: `engram-core/src/query/normalization.rs`
**Dependencies**: Add to `engram-core/Cargo.toml`:
```toml
unicode-normalization = "0.1"
# rust-stemmers = "1.2" # Add in follow-up for language-specific stemming
```

### 3. Synonym & Abbreviation Dictionaries

#### 3.1 Dictionary File Structure

Create directory: `engram-core/resources/synonyms/`

**File format** (`engram-core/resources/synonyms/en.json`):
```json
{
  "version": "1.0.0",
  "language": "en",
  "synonyms": {
    "automobile": [
      {"term": "car", "confidence": 0.95},
      {"term": "vehicle", "confidence": 0.85},
      {"term": "auto", "confidence": 0.90}
    ],
    "physician": [
      {"term": "doctor", "confidence": 0.98},
      {"term": "medical doctor", "confidence": 0.95}
    ]
  },
  "abbreviations": {
    "MI": [
      {"expansion": "myocardial infarction", "confidence": 0.95, "domain": "medical"},
      {"expansion": "Michigan", "confidence": 0.80, "domain": "geographic"}
    ],
    "2nd": [
      {"expansion": "second", "confidence": 1.0}
    ]
  }
}
```

#### 3.2 Dictionary Loading and Hot-Reload

```rust
use notify::{Watcher, RecursiveMode, Result}; // Add notify = "6.1" to Cargo.toml
use std::path::Path;

impl QueryExpander {
    pub fn new(config: ExpansionConfig) -> Self {
        let hot_dicts = Self::load_hot_dictionaries();
        let warm_dicts = WarmTierDictionaries::new();

        let expander = Self {
            hot_dictionaries: Arc::new(ArcSwap::from_pointee(hot_dicts)),
            warm_dictionaries: warm_dicts,
            config,
        };

        // Spawn watcher for hot-reload
        expander.spawn_dictionary_watcher();
        expander
    }

    fn spawn_dictionary_watcher(&self) {
        let hot_dicts_arc = Arc::clone(&self.hot_dictionaries);

        tokio::spawn(async move {
            let (tx, mut rx) = tokio::sync::mpsc::channel(10);

            let mut watcher = notify::recommended_watcher(move |res: Result<notify::Event>| {
                if let Ok(event) = res {
                    if matches!(event.kind, notify::EventKind::Modify(_)) {
                        let _ = tx.blocking_send(event);
                    }
                }
            }).expect("Failed to create file watcher");

            watcher.watch(
                Path::new("engram-core/resources/synonyms"),
                RecursiveMode::Recursive
            ).expect("Failed to watch dictionary directory");

            while let Some(_event) = rx.recv().await {
                tracing::info!("Dictionary files changed, reloading...");
                let new_dicts = Self::load_hot_dictionaries();
                hot_dicts_arc.store(Arc::new(new_dicts));
                tracing::info!("Dictionaries reloaded successfully");
            }
        });
    }

    fn load_hot_dictionaries() -> HotTierDictionaries {
        // Load JSON files from resources/synonyms/*.json
        // Parse and build in-memory HashMap structures
        HotTierDictionaries {
            synonyms: HashMap::new(), // TODO: load from files
            abbreviations: HashMap::new(),
        }
    }

    pub fn expand(&self, query_context: &mut QueryContext) {
        let hot_dicts = self.hot_dictionaries.load();
        let mut expansions = Vec::new();

        for token in &query_context.normalized_tokens {
            // Add original term
            expansions.push(ExpansionTerm {
                term: token.clone(),
                source: ExpansionSource::Original,
                confidence: query_context.original_confidence,
                hop_count: 0,
            });

            // Expand synonyms
            if let Some(synonyms) = hot_dicts.synonyms.get(token) {
                for (syn_term, syn_conf) in synonyms.iter().take(self.config.max_synonyms_per_term) {
                    let propagated_conf = query_context.original_confidence.propagate_expansion(
                        *syn_conf,
                        0.9, // Trust factor for dictionary synonyms
                        1, // Single hop
                        self.config.confidence_decay_per_hop,
                    );

                    if propagated_conf.raw() >= self.config.min_expansion_confidence {
                        expansions.push(ExpansionTerm {
                            term: syn_term.clone(),
                            source: ExpansionSource::Synonym,
                            confidence: propagated_conf,
                            hop_count: 1,
                        });
                    }
                }
            }

            // Expand abbreviations
            if let Some(abbrevs) = hot_dicts.abbreviations.get(token) {
                for (abbrev_term, abbrev_conf) in abbrevs {
                    let propagated_conf = query_context.original_confidence.propagate_expansion(
                        *abbrev_conf,
                        0.95, // High trust for abbreviation expansions
                        1,
                        self.config.confidence_decay_per_hop,
                    );

                    if propagated_conf.raw() >= self.config.min_expansion_confidence {
                        expansions.push(ExpansionTerm {
                            term: abbrev_term.clone(),
                            source: ExpansionSource::Abbreviation,
                            confidence: propagated_conf,
                            hop_count: 1,
                        });
                    }
                }
            }
        }

        // Cap total expansions
        expansions.sort_by(|a, b| b.confidence.partial_cmp(&a.confidence).unwrap());
        expansions.truncate(self.config.max_total_expansions);

        query_context.expansions = expansions;
    }
}
```

**File**: `engram-core/src/query/expansion.rs` (extend existing impl)
**Dependencies**: Add to `engram-core/Cargo.toml`:
```toml
notify = "6.1"
```

### 4. API Integration

#### 4.1 Update `search_memories_rest` Handler

**File**: `engram-cli/src/api.rs` (replace existing function starting at line 969)

```rust
pub async fn search_memories_rest(
    State(state): State<ApiState>,
    Query(params): Query<HashMap<String, String>>,
) -> Result<impl IntoResponse, ApiError> {
    let query_text = params
        .get("query")
        .ok_or_else(|| ApiError::InvalidInput("Missing 'query' parameter".to_string()))?;

    let limit = params
        .get("limit")
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(10);

    // NEW: Build QueryContext with expansion
    let expansion_config = ExpansionConfig::default();
    let expander = QueryExpander::new(expansion_config);
    let normalizer = TokenNormalizer::default();

    let normalized_tokens = normalizer.normalize(query_text, Some(Language::English));
    let mut query_context = QueryContext {
        raw_text: query_text.clone(),
        detected_language: Some(Language::English), // TODO: integrate language detection from Task 001
        normalized_tokens,
        expansions: Vec::new(),
        original_confidence: CoreConfidence::exact(0.7),
    };

    // Expand query terms
    expander.expand(&mut query_context);

    // Build cue from expanded query
    let cue_builder = CueBuilder::from_text_query(query_context.clone());
    let cue = cue_builder.build();

    // Execute recall
    let recall_result = state.store.recall(&cue);

    // Check streaming delivery
    if !recall_result.streaming_delivered {
        tracing::warn!(
            "Search completed successfully but event streaming failed - SSE subscribers not notified"
        );
        return Err(ApiError::SystemError(
            "Search completed but event notification failed. \
             SSE subscribers did not receive the recall events. \
             Check /api/v1/system/health for streaming status."
                .to_string(),
        ));
    }

    // Build response with expansion diagnostics
    let memories: Vec<serde_json::Value> = recall_result
        .results
        .iter()
        .take(limit)
        .map(|(episode, confidence)| {
            json!({
                "id": episode.id,
                "content": episode.what,
                "confidence": confidence.raw(),
                "memory_type": "episodic",
                "timestamp": episode.when.to_rfc3339(),
            })
        })
        .collect();

    // NEW: Add expansion diagnostics
    let expansion_diagnostic = json!({
        "expansion_applied": !query_context.expansions.is_empty(),
        "original_terms": query_context.normalized_tokens,
        "expanded_terms": query_context.expansions.iter().map(|e| json!({
            "term": e.term,
            "source": format!("{:?}", e.source),
            "confidence": e.confidence.raw(),
        })).collect::<Vec<_>>(),
    });

    Ok(Json(json!({
        "memories": memories,
        "total": memories.len(),
        "diagnostics": {
            "expansion": expansion_diagnostic,
        }
    })))
}
```

#### 4.2 Update `recall_memories` Handler

**File**: `engram-cli/src/api.rs` (update function starting at line 683)

Similar changes to `search_memories_rest`, adding QueryContext building and expansion diagnostics to the response.

### 5. Configuration and Observability

#### 5.1 Add Metrics

**File**: `engram-core/src/metrics/mod.rs` (extend existing metrics)

```rust
#[cfg(feature = "monitoring")]
pub mod query_metrics {
    use std::sync::atomic::{AtomicU64, Ordering};

    pub struct QueryExpansionMetrics {
        pub expansion_terms_total: AtomicU64,
        pub embedding_fallback_total: AtomicU64,
        pub language_mismatch_total: AtomicU64,
    }

    impl QueryExpansionMetrics {
        pub fn record_expansion(&self, term_count: usize) {
            self.expansion_terms_total.fetch_add(term_count as u64, Ordering::Relaxed);
        }

        pub fn record_embedding_fallback(&self) {
            self.embedding_fallback_total.fetch_add(1, Ordering::Relaxed);
        }

        pub fn record_language_mismatch(&self) {
            self.language_mismatch_total.fetch_add(1, Ordering::Relaxed);
        }
    }
}
```

#### 5.2 Configuration File

**File**: `engram-cli/config/query_expansion.toml`

```toml
[expansion]
enabled = true
max_synonyms_per_term = 5
max_total_expansions = 20
confidence_decay_per_hop = 0.85
min_expansion_confidence = 0.3

[expansion.languages]
english = true
spanish = false  # Disable until dictionaries available
mandarin = false

[expansion.dictionaries]
# Override default dictionary paths
# english_synonyms = "/custom/path/en_synonyms.json"
```

### 6. Testing Strategy

#### 6.1 Unit Tests

**File**: `engram-core/src/query/expansion/tests.rs`

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_normalization_unicode() {
        let normalizer = TokenNormalizer::default();
        let result = normalizer.normalize("café résumé", Some(Language::English));
        assert_eq!(result, vec!["cafe", "resume"]);
    }

    #[test]
    fn test_normalization_numerals() {
        let normalizer = TokenNormalizer::default();
        let result = normalizer.normalize("2nd place winner", Some(Language::English));
        // After integration of numeral expansion
        assert!(result.contains(&"second".to_string()));
    }

    #[test]
    fn test_synonym_expansion() {
        let config = ExpansionConfig::default();
        let expander = QueryExpander::new(config);

        let mut query_ctx = QueryContext {
            raw_text: "automobile".to_string(),
            detected_language: Some(Language::English),
            normalized_tokens: vec!["automobile".to_string()],
            expansions: Vec::new(),
            original_confidence: Confidence::HIGH,
        };

        expander.expand(&mut query_ctx);

        // Should expand to car, vehicle, auto
        assert!(query_ctx.expansions.len() > 1);
        assert!(query_ctx.expansions.iter().any(|e| e.term == "car"));
    }

    #[test]
    fn test_confidence_propagation() {
        let original = Confidence::exact(0.9);
        let propagated = original.propagate_expansion(
            0.95, // similarity
            0.9,  // source trust
            1,    // hop count
            0.85, // hop decay
        );

        // Should be reduced but still high
        assert!(propagated.raw() < original.raw());
        assert!(propagated.raw() > 0.6);
    }

    #[test]
    fn test_expansion_cap() {
        let config = ExpansionConfig {
            max_total_expansions: 5,
            ..Default::default()
        };
        let expander = QueryExpander::new(config);

        let mut query_ctx = QueryContext {
            raw_text: "test query".to_string(),
            detected_language: Some(Language::English),
            normalized_tokens: vec!["test".to_string(), "query".to_string()],
            expansions: Vec::new(),
            original_confidence: Confidence::MEDIUM,
        };

        expander.expand(&mut query_ctx);

        // Should respect max cap
        assert!(query_ctx.expansions.len() <= 5);
    }
}
```

#### 6.2 Integration Tests

**File**: `engram-core/tests/query_expansion_integration.rs`

```rust
#[cfg(test)]
mod integration_tests {
    use engram_core::*;

    #[tokio::test]
    async fn test_cross_language_recall() {
        // Store memory in English
        let store = MemoryStore::new();
        let episode = Episode::new(
            "ep1".to_string(),
            Utc::now(),
            "The patient experienced myocardial infarction".to_string(),
            [0.5; 768],
            Confidence::HIGH,
        );
        store.store(episode);

        // Query with abbreviation
        let query_ctx = QueryContext {
            raw_text: "MI patient".to_string(),
            detected_language: Some(Language::English),
            normalized_tokens: vec!["MI".to_string(), "patient".to_string()],
            expansions: vec![
                ExpansionTerm {
                    term: "MI".to_string(),
                    source: ExpansionSource::Original,
                    confidence: Confidence::HIGH,
                    hop_count: 0,
                },
                ExpansionTerm {
                    term: "myocardial infarction".to_string(),
                    source: ExpansionSource::Abbreviation,
                    confidence: Confidence::exact(0.95),
                    hop_count: 1,
                },
            ],
            original_confidence: Confidence::HIGH,
        };

        let cue = CueBuilder::from_text_query(query_ctx).build();
        let result = store.recall(&cue);

        // Should find the episode via abbreviation expansion
        assert!(!result.results.is_empty());
        assert_eq!(result.results[0].0.id, "ep1");
    }

    #[test]
    fn test_expansion_disabled_fallback() {
        // Test that disabling expansion preserves lexical behavior
        let config = ExpansionConfig {
            max_synonyms_per_term: 0,
            max_total_expansions: 0,
            ..Default::default()
        };

        let expander = QueryExpander::new(config);
        let mut query_ctx = QueryContext {
            raw_text: "test".to_string(),
            detected_language: Some(Language::English),
            normalized_tokens: vec!["test".to_string()],
            expansions: Vec::new(),
            original_confidence: Confidence::MEDIUM,
        };

        expander.expand(&mut query_ctx);

        // Should only have original term
        assert_eq!(query_ctx.expansions.len(), 1);
        assert_eq!(query_ctx.expansions[0].source, ExpansionSource::Original);
    }
}
```

## File Modifications Summary

### New Files to Create
1. `engram-core/src/query/expansion.rs` - Core expansion logic
2. `engram-core/src/query/normalization.rs` - Token normalization
3. `engram-core/src/query/expansion/tests.rs` - Unit tests
4. `engram-core/tests/query_expansion_integration.rs` - Integration tests
5. `engram-core/resources/synonyms/en.json` - English dictionary
6. `engram-cli/config/query_expansion.toml` - Configuration
7. `docs/howto/semantic_search.md` - User documentation

### Files to Modify
1. `engram-core/src/lib.rs` - Add `propagate_expansion` method to `Confidence`
2. `engram-core/src/memory.rs` - Add `CueBuilder::from_text_query` method
3. `engram-core/src/query/mod.rs` - Add `pub mod expansion;` and `pub mod normalization;`
4. `engram-cli/src/api.rs` - Update `search_memories_rest` and `recall_memories` handlers
5. `engram-core/src/metrics/mod.rs` - Add query expansion metrics
6. `engram-core/Cargo.toml` - Add dependencies: `unicode-normalization`, `notify`
7. `engram-cli/Cargo.toml` - Add `uuid` if not already present

## Dependencies to Add

### engram-core/Cargo.toml
```toml
[dependencies]
unicode-normalization = "0.1"
notify = "6.1"
serde_json = "1.0"  # If not already present
uuid = { version = "1.10", features = ["v4"] }
```

### Future Follow-ups (Not in Scope)
- `rust-stemmers = "1.2"` - Language-specific stemming (add after v1)
- Machine translation libraries for cross-lingual expansion

## Acceptance Criteria (Updated)

- [x] `QueryExpander` created with hot-tier dictionary support and hot-reload
- [x] `TokenNormalizer` implements Unicode normalization and tokenization
- [x] `Confidence::propagate_expansion` method for hop-based decay
- [x] `CueBuilder::from_text_query` constructs hybrid or semantic cues
- [x] `search_memories_rest` and `recall_memories` use expanded queries
- [x] Response diagnostics include expansion metadata
- [x] Configuration file for per-language enable/disable
- [x] Metrics track expansion usage and fallbacks
- [x] Unit tests for normalization and expansion
- [x] Integration tests for cross-language recall
- [x] Regression tests verify lexical fallback behavior
- [x] Documentation in `docs/howto/semantic_search.md`

## Testing Approach (Detailed)

### Unit Tests
- Unicode normalization (diacritics, NFKC)
- Confidence propagation with various decay factors
- Synonym expansion with confidence thresholds
- Abbreviation expansion with domain context
- Expansion capping logic

### Integration Tests
- Cross-language recall (abbreviation → full term)
- Hybrid semantic+lexical scoring
- Hot-reload of dictionary files
- Fallback behavior when expansion disabled
- Confidence calibration in expanded results

### Regression Tests
- Zero behavior change when `max_total_expansions = 0`
- Identical results for same query with expansion on/off
- API response format compatibility
- SSE event streaming still works

### Performance Tests
- Expansion latency < 5ms for typical queries
- Dictionary hot-reload < 100ms
- No memory leaks in long-running expansion
- NUMA-aware dictionary access (if applicable)

## Risk Mitigation (Expanded)

### Query Explosion Prevention
- Hard cap on `max_total_expansions` (default: 20)
- Confidence threshold filters low-quality expansions
- Sort by confidence and truncate before processing

### Dictionary Quality
- Start with curated high-confidence terms only
- Implement dictionary validation during load
- Log warnings for low-confidence expansions
- Community review process for new terms

### Performance
- Hot-tier for frequent terms keeps latency low
- Warm-tier via mmap for rare terms
- Adaptive promotion based on access frequency
- Async dictionary reload doesn't block queries

### Backward Compatibility
- Legacy lexical search still available
- Configuration flag to disable expansion entirely
- API response format preserves existing fields
- Diagnostics added as optional metadata

## Notes and Future Work

### Initial Dictionary Curation
- Start with medical/technical abbreviations (high value)
- Use WordNet for English synonym base
- Validate licensing for redistribution
- Community contribution process for domain-specific terms

### Hot-Reload Implementation
- Use `notify` crate for filesystem watching
- `ArcSwap` for lock-free dictionary updates
- Validate JSON before swapping
- Rollback mechanism for invalid dictionaries

### Cross-Lingual Expansion (Future)
- Task 001 provides language detection and embeddings
- Machine translation for query rewriting
- Multilingual WordNet for synonym mapping
- Cultural context awareness (dates, names)

### Observability
- Trace expansion decisions for debugging
- Metrics for expansion effectiveness
- A/B testing framework for dictionary changes
- User feedback loop for expansion quality

## Estimated Effort Breakdown

- **Query Expansion Infrastructure** (0.5 days)
  - QueryContext struct and ExpansionTerm types
  - QueryExpander skeleton with config

- **Lexical Normalization** (0.25 days)
  - TokenNormalizer implementation
  - Unicode normalization integration

- **Dictionary Loading & Hot-Reload** (0.5 days)
  - JSON parsing and validation
  - File watcher setup with `notify`
  - ArcSwap integration

- **API Integration** (0.5 days)
  - Update REST handlers
  - Add diagnostics to responses
  - Metrics instrumentation

- **Testing** (0.5 days)
  - Unit tests for normalization and expansion
  - Integration tests for recall flow
  - Regression tests for fallback behavior

- **Documentation** (0.25 days)
  - User guide for semantic search
  - Configuration reference
  - Dictionary format specification

**Total**: 2.5 days

## Definition of Done

- [ ] All new files created and integrated
- [ ] All modified files updated with expansion logic
- [ ] Dependencies added to Cargo.toml files
- [ ] Unit tests pass with >80% coverage
- [ ] Integration tests verify end-to-end flow
- [ ] Regression tests prove backward compatibility
- [ ] Metrics emit expansion statistics
- [ ] Configuration file documented
- [ ] User documentation written
- [ ] Code reviewed for cognitive ergonomics
- [ ] Performance benchmarks show <5ms expansion overhead
- [ ] `make quality` passes without warnings
