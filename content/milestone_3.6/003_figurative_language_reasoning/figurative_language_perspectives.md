# Figurative Language Interpretation: Architectural Perspectives

## Cognitive Architecture Perspective

### The Challenge: Bridging Semantic Gaps in Memory Recall

Human memory doesn't store language literally. When someone says "She gave me the cold shoulder," our brains don't search for literal temperature or body parts. Instead, we automatically map figurative expressions onto their conceptual meanings through learned associations.

This is a fundamental challenge for cognitive architectures: how do we enable systems to understand that "cold shoulder" means "deliberate ignoring" without requiring perfect literal matches?

### Cognitive Science Foundation

Conceptual Metaphor Theory (Lakoff & Johnson, 1980) demonstrates that human thought is fundamentally metaphorical. We organize abstract concepts through mappings from concrete domains:
- TIME IS SPACE: "looking forward to", "behind schedule"
- ARGUMENT IS WAR: "attacked my position", "defended her claim"
- IDEAS ARE FOOD: "half-baked idea", "food for thought"

These aren't just linguistic quirks - they're how we structure cognition. A memory system aligned with human cognition must handle these mappings.

### Implementation Strategy for Engram

**Two-Stage Detection:**
1. Fast path: Rule-based detection for explicit markers and common idioms
2. Slow path: Contextual analysis for implicit metaphors using semantic type violations

**Cognitive Integration:**
- Figurative interpretations participate in spreading activation alongside literal cues
- Confidence weights reflect metaphor conventionality (dead metaphors = high confidence)
- Memory consolidation strengthens frequently-used metaphorical mappings over time

**Biological Plausibility:**
- Hippocampus (episodic memory) detects figurative language in incoming queries
- Neocortex (semantic memory) stores consolidated metaphorical mappings
- Complementary Learning Systems enable both fast detection and slow learning

### Key Insight

Figurative language isn't a peripheral feature - it's central to human-aligned memory retrieval. Systems that only handle literal language will never achieve natural interaction.

## Memory Systems Perspective

### The Problem: Episodic-Semantic Misalignment

Consider this scenario:
- Episodic memory: "My boss ignored me in the meeting this morning"
- Query: "When did Sarah give me the cold shoulder?"

Pure embedding similarity fails because "cold shoulder" and "ignored" occupy different vector spaces despite conceptual equivalence. This represents a fundamental mismatch between episodic encoding and semantic query patterns.

### Hippocampal-Neocortical Dynamics

The hippocampus (episodic memory) encodes experiences with rich contextual detail. The neocortex (semantic memory) stores abstracted patterns and schemas. Figurative language creates a bridge between these systems:

**During Encoding:**
- Detect figurative language in stored memories
- Annotate with literal interpretations
- Store under `EpisodeMetadata.figurative_annotations`

**During Retrieval:**
- Detect figurative language in queries
- Generate candidate literal interpretations
- Use both figurative and literal cues for spreading activation

**During Consolidation:**
- Strengthen frequently-used metaphorical mappings
- Move conventional metaphors from explicit lookup to implicit activation
- Prune unused figurative associations

### Schema Activation

Figurative expressions activate semantic schemas that guide retrieval:
- "Cold shoulder" -> SOCIAL_REJECTION schema
- "Break the ice" -> SOCIAL_INITIATION schema
- "Time flies" -> TEMPORAL_SPEED schema

Schema activation primes related concepts, improving recall accuracy and enabling analogical reasoning.

### Pattern Completion

When figurative cues are underspecified, pattern completion fills gaps:
- Query: "cold" (ambiguous)
- Context: social interaction memory
- Pattern completion: likely SOCIAL_REJECTION, not TEMPERATURE

This mirrors how humans use context to disambiguate metaphors.

### Key Insight

Figurative language handling enables proper interaction between episodic (experience) and semantic (concept) memory systems, essential for human-like retrieval.

## Rust Graph Engine Perspective

### The Challenge: Zero-Copy Figurative Processing

Figurative language interpretation adds computational overhead to the critical query path. We need to:
1. Detect figurative spans without allocation-heavy NLP pipelines
2. Map to literal concepts using cache-friendly lookups
3. Integrate with spreading activation without disrupting performance

### Lock-Free Figurative Mapping

**Design:**
```rust
pub struct FigurativeMapper {
    // Immutable mapping from figurative to literal concepts
    idiom_map: Arc<DashMap<CompactString, Vec<LiteralInterpretation>>>,

    // Embedding index for similarity-based mapping
    concept_index: Arc<HnswIndex>,

    // Per-language detectors (lock-free access)
    detectors: Arc<[LanguageDetector]>,
}

#[derive(Clone)]
pub struct LiteralInterpretation {
    concept: CompactString,
    confidence: f32,
    rationale: &'static str, // zero-copy from static lexicon
}
```

**Key Optimizations:**
- `DashMap` enables concurrent reads without locks
- `CompactString` reduces heap allocations for small strings
- Static string rationales avoid per-query allocations
- `Arc` sharing eliminates cloning overhead

### Integration with Spreading Activation

Figurative interpretations augment cue sets:

```rust
pub struct QueryContext {
    literal_cues: Vec<Cue>,
    figurative_interpretations: Vec<FigurativeInterpretation>,
}

impl QueryContext {
    pub fn merged_cues(&self) -> impl Iterator<Item = WeightedCue> + '_ {
        let literal = self.literal_cues.iter()
            .map(|c| WeightedCue { cue: c, weight: 1.0 });

        let figurative = self.figurative_interpretations.iter()
            .filter(|fi| fi.confidence >= THRESHOLD)
            .flat_map(|fi| fi.literal_concepts.iter())
            .map(|c| WeightedCue {
                cue: c,
                weight: fi.confidence * FIGURATIVE_PENALTY
            });

        literal.chain(figurative)
    }
}
```

**Design Principles:**
- Lazy iteration avoids materializing merged cue vectors
- Confidence-based filtering eliminates low-quality interpretations
- Weight penalties prevent figurative matches from dominating literal ones

### Cache Locality

Figurative detection is cache-friendly:
1. Rule-based detection operates on small string slices (L1 cache)
2. Idiom map uses hash-based lookup (cache-local for hot paths)
3. Embedding similarity batches operations to amortize cache misses

### Performance Budget

Target latency additions:
- Rule-based detection: <2ms (branch prediction friendly)
- Idiom lookup: <1ms (hash table, hot in cache)
- Embedding similarity: <3ms (SIMD-accelerated)
- Total: <6ms (within p95 budget)

### Key Insight

Figurative language processing can achieve <6ms overhead through zero-copy designs, lock-free data structures, and cache-aware algorithms.

## Systems Architecture Perspective

### The Problem: Scaling Figurative Processing

As memory systems grow to billions of episodes and hundreds of languages, figurative language handling must scale without becoming a bottleneck.

### Tiered Storage Strategy

**Hot Tier (In-Memory):**
- Frequently-used idioms and metaphor mappings
- Rule-based detector state
- Recent interpretation cache

**Warm Tier (Memory-Mapped):**
- Full idiom lexicons per language
- Precomputed figurative annotations for stored episodes
- ML model weights (quantized, memory-mapped)

**Cold Tier (Disk):**
- Historical interpretation logs for auditing
- Full training data for model updates
- Archived language-specific resources

### Amortized Annotation

Detect figurative language during ingestion, not query time:

```
Ingestion Path:
  Raw Episode -> Language Detection -> Figurative Detection -> Annotation
     |                                                           |
     +-----------------> Storage <------------------------------+
                            |
                      [Persisted with metadata]

Query Path:
  Query -> Figurative Detection -> Use Pre-Annotated Episodes
     |                                  (Fast: no re-detection)
     +------------> Spreading Activation
```

This shifts cost to the write path (acceptable) and keeps read path fast.

### Distributed Interpretation

For multi-node deployments:
- Each node maintains local idiom cache (eventually consistent)
- Centralized lexicon updates propagate via gossip protocol
- Interpretation decisions are deterministic given cache state (reproducible)

### Monitoring and Observability

Critical metrics:
```
engram_figurative_detection_duration_seconds (histogram)
engram_figurative_interpretation_applied_total (counter by language, confidence_band)
engram_figurative_low_confidence_total (counter, alerts for quality issues)
engram_figurative_cache_hit_ratio (gauge, optimization signal)
```

### Governance and Safety

**Versioned Lexicons:**
- Idiom mappings stored in git-tracked TOML files
- Schema: `"cold_shoulder" = { literal = ["ignore", "rebuff"], confidence = 0.9, languages = ["en"] }`
- Changes require PR review before deployment

**Confidence Thresholds:**
- Operator-configurable per deployment: `figurative_min_confidence: 0.7`
- Runtime toggles: `figurative_enabled_languages: ["en", "es"]`
- Emergency killswitch: `disable_figurative: true`

**Auditing:**
- Log all interpretations applied with confidence scores
- Store explanation metadata with recall results
- Enable post-hoc analysis of interpretation accuracy

### Key Insight

Scalable figurative language handling requires tiered storage, amortized annotation, deterministic interpretation, and strong governance controls.

## Synthesis: Choosing the Best Approach

Each perspective offers critical insights:

**Cognitive Architecture:** Emphasizes integration with spreading activation and consolidation - essential for human-aligned behavior.

**Memory Systems:** Highlights the role of figurative language in bridging episodic-semantic gaps and enabling schema-based retrieval.

**Rust Graph Engine:** Provides concrete zero-copy, lock-free implementation strategies that meet performance budgets.

**Systems Architecture:** Ensures scalability through tiered storage, amortized annotation, and strong governance.

### Recommended Implementation Path

1. Start with cognitive architecture perspective: define how figurative interpretations participate in spreading activation and consolidation
2. Layer in memory systems perspective: implement schema activation and pattern completion
3. Implement using Rust graph engine techniques: zero-copy, lock-free, cache-friendly
4. Scale with systems architecture approach: tiered storage, amortized annotation, monitoring

This synthesis ensures both biological plausibility and production readiness.
