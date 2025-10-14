# Figurative Language Interpretation Research

## Overview
This research explores approaches for detecting and interpreting figurative language (metaphors, similes, idioms, analogies) in memory systems, with focus on multilingual support, explainability, and integration with cognitive architectures.

## Core Problem

Figurative language creates a semantic disconnect between surface forms and intended meanings. Traditional information retrieval systems treat language literally, causing critical mismatches:
- Query: "She gave me the cold shoulder" fails to match memory: "She ignored me at the party"
- Query: "Time flies" fails to match memory: "The event passed quickly"
- Query: "Break the ice" fails to match memory: "Started the conversation"

In cognitive systems like Engram, this limitation prevents natural recall patterns and reduces human-aligned interaction quality.

## Research Areas

### 1. Computational Metaphor Detection

**Key Findings:**
- Rule-based approaches achieve high precision (>90%) for explicit markers: "like", "as...as", "seems like"
- Statistical methods using semantic type violations detect implicit metaphors: "time flies" violates temporal constraints
- Neural classifiers (BERT-based) achieve 78-85% F1 on metaphor detection tasks (Gao et al., 2018; Su et al., 2020)
- Hybrid approaches combining rules + ML outperform either alone (Tsvetkov et al., 2014)

**Citation:**
- Gao, G., Choi, E., Choi, Y., & Zettlemoyer, L. (2018). Neural metaphor detection in context. NAACL.
- Tsvetkov, Y., Boytsov, L., Gershman, A., Nyberg, E., & Dyer, C. (2014). Metaphor detection with cross-lingual model transfer. ACL.

### 2. Metaphor-to-Literal Mapping

**Conceptual Metaphor Theory (Lakoff & Johnson, 1980):**
- Metaphors map structured source domains onto target domains
- Example: ARGUMENT IS WAR -> "attacked my position", "defended her claim"
- Systematic mappings can be extracted and formalized

**Computational Approaches:**
- Word embeddings capture metaphorical relationships through vector arithmetic (Shutova et al., 2016)
- Metaphor interpretation as paraphrasing task: "cold shoulder" -> "deliberate ignoring" (Mao et al., 2018)
- Cross-domain similarity scoring using contextualized embeddings (Peters et al., 2018)

**Citation:**
- Lakoff, G., & Johnson, M. (1980). Metaphors We Live By. University of Chicago Press.
- Shutova, E., Teufel, S., & Korhonen, A. (2016). Statistical metaphor processing. Computational Linguistics, 39(2).

### 3. Multilingual Figurative Language

**Challenges:**
- Idioms are culture-specific: English "cold shoulder" vs. Chinese "chi bi geng" (eating closed-door soup)
- Metaphor systems vary: some languages use water metaphors for time, others use spatial metaphors
- Translation often loses figurative meaning

**Solutions:**
- Language-specific idiom lexicons (Wiktionary, culturally-grounded resources)
- Cross-lingual embedding spaces preserve some metaphorical structure (Vulić et al., 2020)
- Explicit language tags enable per-language interpretation pipelines

**Citation:**
- Vulić, I., Glavaš, G., Reichart, R., & Korhonen, A. (2020). Do we really need fully unsupervised cross-lingual embeddings? EMNLP.

### 4. Cognitive Grounding

**Memory and Metaphor:**
- Human memory organizes concepts through metaphorical mappings (Gibbs, 2006)
- Retrieval cues activate both literal and figurative associations
- Spreading activation propagates across metaphorical domains

**Implication for Engram:**
- Figurative interpretations should participate in spreading activation
- Confidence scoring reflects metaphor conventionality (dead metaphors -> high confidence)
- Memory consolidation could strengthen frequently-used metaphorical mappings

**Citation:**
- Gibbs, R. W. (2006). Embodiment and Cognitive Science. Cambridge University Press.

### 5. Explainability Requirements

**Transparency Needs:**
- Users need to understand why a figurative match was made
- Operators require auditable interpretation decisions
- Low-confidence interpretations should be flagged, not silently applied

**Design Principles:**
- Store interpretation rationale alongside matches: "matched via idiom: 'cold shoulder' -> 'ignoring'"
- Provide confidence thresholds with graceful degradation
- Enable human-in-the-loop validation for new mappings

## Technical Architecture Considerations

### Detection Pipeline
1. **Preprocessing:** Language identification, tokenization
2. **Rule-based layer:** Explicit simile/metaphor markers, idiom lexicon lookup
3. **ML fallback:** Lightweight classifier for implicit metaphors (quantized BERT or distilled model)
4. **Output:** `FigurativeSpan { text, type, confidence }`

### Mapping Pipeline
1. **Input:** Detected figurative span
2. **Candidate generation:** Embedding similarity, curated mapping table
3. **Ranking:** Confidence scoring based on conventionality, context fit
4. **Output:** `FigurativeInterpretation { literal_concept, confidence, rationale }`

### Integration with Query Flow
```
Query -> Language Detection -> Figurative Detection -> Concept Mapping
  |                                                           |
  +----> Literal Cues --------+                              |
                               v                              v
                          Cue Merging (weighted by confidence)
                               v
                        Spreading Activation
                               v
                       Ranked Results (with explanation metadata)
```

### Storage Considerations
- Precompute figurative annotations during ingestion (amortize cost)
- Store under `EpisodeMetadata.figurative_annotations`
- Index literal interpretations for faster recall routing

## Performance Constraints

**Latency Budget:**
- Figurative detection: <2ms (rule-based) + <3ms (ML fallback) = <5ms total
- Concept mapping: <1ms (cache lookups) + <2ms (embedding similarity) = <3ms total
- Total overhead: <8ms added to query path (acceptable for p95 <50ms target)

**Accuracy Targets:**
- Precision: >85% (avoid false positives that confuse recall)
- Recall: >75% (catch most conventional figurative language)
- Coverage: English (90%), Spanish/Mandarin (75%), other languages (60%)

## Safety and Governance

**Risk Mitigation:**
- Minimum confidence threshold (default: 0.7) to avoid spurious interpretations
- Operator controls to disable per language/domain
- Versioned lexicons with human review process
- Monitoring metrics: `engram_figurative_low_confidence_total`, `engram_figurative_interpretation_applied_total`

**Hallucination Prevention:**
- Never invent interpretations not in curated mappings or above ML confidence threshold
- Fall back to literal interpretation when uncertain
- Log all low-confidence interpretations for post-hoc analysis

## Integration with Cognitive Models

**Complementary Learning Systems:**
- Fast system (hippocampus-like): Detect figurative language in episodic queries
- Slow system (neocortex-like): Consolidate frequently-used metaphorical mappings
- Over time, conventional metaphors strengthen through replay

**Schema Theory:**
- Metaphors activate schemas that guide interpretation
- "Cold shoulder" activates SOCIAL_REJECTION schema
- Schema-based priming improves recall accuracy

## Future Research Directions

1. **Adaptive Mapping:** Learn user-specific metaphorical patterns from usage
2. **Contextual Disambiguation:** Resolve metaphor ambiguity using query context
3. **Cross-lingual Transfer:** Use multilingual embeddings to transfer metaphor knowledge
4. **Temporal Dynamics:** Model how metaphor meanings drift over time
5. **Multimodal Grounding:** Incorporate visual/sensory associations with metaphors

## Implementation Priorities

**Phase 1 (Current Task):**
- Rule-based detection for explicit markers
- Curated lexicons for common idioms (English, Spanish, Mandarin)
- Simple embedding-based mapping with confidence thresholds
- Basic explainability metadata

**Phase 2 (Future):**
- ML-based implicit metaphor detection
- Dynamic mapping learning from usage patterns
- Schema-based interpretation
- Integration with memory consolidation

## Key Takeaways

1. Figurative language handling is essential for natural memory recall
2. Hybrid rule-based + ML approaches balance precision and coverage
3. Explainability and confidence scoring are non-negotiable for trustworthiness
4. Multilingual support requires language-specific resources and pipelines
5. Integration with cognitive models (spreading activation, consolidation) enables learning
6. Performance overhead (<8ms) is acceptable for high-value feature

## References

- Lakoff, G., & Johnson, M. (1980). Metaphors We Live By. University of Chicago Press.
- Gao, G., Choi, E., Choi, Y., & Zettlemoyer, L. (2018). Neural metaphor detection in context. NAACL.
- Gibbs, R. W. (2006). Embodiment and Cognitive Science. Cambridge University Press.
- Shutova, E., Teufel, S., & Korhonen, A. (2016). Statistical metaphor processing. Computational Linguistics, 39(2).
- Tsvetkov, Y., Boytsov, L., Gershman, A., Nyberg, E., & Dyer, C. (2014). Metaphor detection with cross-lingual model transfer. ACL.
- Vulić, I., Glavaš, G., Reichart, R., & Korhonen, A. (2020). Do we really need fully unsupervised cross-lingual embeddings? EMNLP.
