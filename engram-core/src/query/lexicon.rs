//! Lexicon abstraction for query expansion.
//!
//! This module provides the `Lexicon` trait for pluggable lexicon implementations
//! and concrete implementations for synonyms and abbreviations. Lexicons use
//! FST (Finite State Transducers) for O(|query|) lookup time.
//!
//! ## Design Principles
//!
//! - **Fast Lookup**: FST-based data structures provide O(|query|) lookup
//! - **Pluggable Architecture**: Trait-based design allows custom lexicons
//! - **Confidence Scoring**: Each expansion includes confidence score
//! - **Language-Aware**: Lexicons can be language-specific or multilingual
//!
//! ## Usage
//!
//! ```rust,ignore
//! use engram_core::query::lexicon::{SynonymLexicon, AbbreviationLexicon};
//!
//! // Load synonym lexicon from FST
//! let synonym_lexicon = SynonymLexicon::from_fst_path("lexicons/wordnet_en.fst")?;
//!
//! // Lookup variants
//! let variants = synonym_lexicon.lookup("car", Some("en"));
//! // Returns: [("automobile", 0.8), ("vehicle", 0.7), ...]
//! ```

use super::expansion::{QueryVariant, VariantType};
use std::collections::HashMap;
use std::sync::Arc;

/// Trait for lexicon implementations.
///
/// A lexicon provides query expansion by mapping query terms to related terms
/// with associated confidence scores. Lexicons can be synonym dictionaries,
/// abbreviation expansions, morphological analyzers, etc.
pub trait Lexicon: Send + Sync {
    /// Lookup query variants in the lexicon.
    ///
    /// Returns a list of query variants with confidence scores indicating
    /// how likely each variant matches the user's intent.
    ///
    /// # Arguments
    ///
    /// * `query` - The query text to expand
    /// * `language` - Optional ISO 639-1 language code (e.g., "en", "es")
    ///
    /// # Returns
    ///
    /// Vector of `QueryVariant` objects with confidence scores.
    fn lookup(&self, query: &str, language: Option<&str>) -> Vec<QueryVariant>;

    /// Get the name of this lexicon for observability.
    fn name(&self) -> &str;

    /// Check if this lexicon supports the given language.
    ///
    /// Returns `true` if the lexicon can provide expansions for this language,
    /// or if the lexicon is language-agnostic.
    fn supports_language(&self, language: &str) -> bool;
}

/// In-memory synonym lexicon using HashMap for fast lookup.
///
/// This is a simple implementation for testing and small synonym sets.
/// For production use, consider using FST-based storage loaded from disk.
///
/// ## Confidence Scoring
///
/// - Direct synonyms: 0.8
/// - Hypernyms (more general): 0.5
/// - Related terms: 0.4
pub struct SynonymLexicon {
    /// Map from term to list of (synonym, confidence) pairs
    synonyms: HashMap<String, Vec<(String, f32)>>,

    /// Name of this lexicon
    name: String,

    /// Supported language (None = multilingual)
    language: Option<String>,
}

impl SynonymLexicon {
    /// Create a new empty synonym lexicon.
    #[must_use]
    pub fn new(name: String, language: Option<String>) -> Self {
        Self {
            synonyms: HashMap::new(),
            name,
            language,
        }
    }

    /// Add a synonym mapping with confidence score.
    ///
    /// # Arguments
    ///
    /// * `term` - The original term
    /// * `synonym` - The synonym to add
    /// * `confidence` - Confidence score (0.0-1.0)
    pub fn add_synonym(&mut self, term: String, synonym: String, confidence: f32) {
        let confidence = confidence.clamp(0.0, 1.0);
        self.synonyms
            .entry(term)
            .or_insert_with(Vec::new)
            .push((synonym, confidence));
    }

    /// Add multiple synonyms at once.
    ///
    /// # Arguments
    ///
    /// * `term` - The original term
    /// * `synonyms` - List of (synonym, confidence) pairs
    pub fn add_synonyms(&mut self, term: String, synonyms: Vec<(String, f32)>) {
        self.synonyms.entry(term).or_insert_with(Vec::new).extend(synonyms);
    }

    /// Create a lexicon with default English synonyms for testing.
    ///
    /// Available for integration tests and examples.
    #[must_use]
    pub fn with_test_data() -> Self {
        let mut lexicon = Self::new("test-synonyms".to_string(), Some("en".to_string()));

        // Common synonyms
        lexicon.add_synonyms(
            "car".to_string(),
            vec![
                ("automobile".to_string(), 0.8),
                ("vehicle".to_string(), 0.7),
            ],
        );

        lexicon.add_synonyms(
            "happy".to_string(),
            vec![
                ("joyful".to_string(), 0.8),
                ("pleased".to_string(), 0.7),
                ("content".to_string(), 0.6),
            ],
        );

        lexicon.add_synonyms(
            "big".to_string(),
            vec![
                ("large".to_string(), 0.9),
                ("huge".to_string(), 0.7),
                ("enormous".to_string(), 0.6),
            ],
        );

        lexicon
    }
}

impl Lexicon for SynonymLexicon {
    fn lookup(&self, query: &str, _language: Option<&str>) -> Vec<QueryVariant> {
        let query_lower = query.to_lowercase();

        self.synonyms
            .get(&query_lower)
            .map(|synonyms| {
                synonyms
                    .iter()
                    .map(|(text, confidence)| {
                        QueryVariant::new(text.clone(), VariantType::Synonym, *confidence)
                    })
                    .collect()
            })
            .unwrap_or_default()
    }

    fn name(&self) -> &str {
        &self.name
    }

    fn supports_language(&self, language: &str) -> bool {
        self.language.as_ref().map_or(true, |lang| lang == language)
    }
}

/// Abbreviation expansion lexicon.
///
/// Maps abbreviations to their full forms with confidence scores.
/// Handles ambiguous abbreviations by returning multiple expansions
/// with different confidence scores.
///
/// ## Confidence Scoring
///
/// - Unambiguous technical abbreviations: 0.9
/// - Domain-specific (context-dependent): 0.7
/// - Ambiguous (multiple meanings): 0.4-0.6 per meaning
///
/// ## Examples
///
/// - "ML" -> [("machine learning", 0.85), ("maximum likelihood", 0.4)]
/// - "AI" -> [("artificial intelligence", 0.95)]
/// - "MI" -> [("myocardial infarction", 0.9), ("machine intelligence", 0.3)]
pub struct AbbreviationLexicon {
    /// Map from abbreviation to list of (expansion, confidence) pairs
    abbreviations: HashMap<String, Vec<(String, f32)>>,

    /// Name of this lexicon
    name: String,

    /// Domain/language for context-specific expansions
    #[allow(dead_code)]  // Will be used in future for context-aware disambiguation
    domain: Option<String>,
}

impl AbbreviationLexicon {
    /// Create a new empty abbreviation lexicon.
    #[must_use]
    pub fn new(name: String, domain: Option<String>) -> Self {
        Self {
            abbreviations: HashMap::new(),
            name,
            domain,
        }
    }

    /// Add an abbreviation expansion with confidence score.
    ///
    /// # Arguments
    ///
    /// * `abbreviation` - The abbreviated form (e.g., "ML")
    /// * `expansion` - The full form (e.g., "machine learning")
    /// * `confidence` - Confidence score (0.0-1.0)
    pub fn add_abbreviation(&mut self, abbreviation: String, expansion: String, confidence: f32) {
        let confidence = confidence.clamp(0.0, 1.0);
        self.abbreviations
            .entry(abbreviation)
            .or_insert_with(Vec::new)
            .push((expansion, confidence));
    }

    /// Add multiple expansions for an abbreviation.
    ///
    /// Use this for ambiguous abbreviations with multiple meanings.
    ///
    /// # Arguments
    ///
    /// * `abbreviation` - The abbreviated form
    /// * `expansions` - List of (expansion, confidence) pairs
    pub fn add_expansions(&mut self, abbreviation: String, expansions: Vec<(String, f32)>) {
        self.abbreviations
            .entry(abbreviation)
            .or_insert_with(Vec::new)
            .extend(expansions);
    }

    /// Create a lexicon with default technical abbreviations for testing.
    ///
    /// Available for integration tests and examples.
    #[must_use]
    pub fn with_test_data() -> Self {
        let mut lexicon = Self::new("test-abbreviations".to_string(), Some("tech".to_string()));

        // Technical abbreviations
        lexicon.add_expansions(
            "ML".to_string(),
            vec![
                ("machine learning".to_string(), 0.85),
                ("maximum likelihood".to_string(), 0.4),
            ],
        );

        lexicon.add_expansions(
            "AI".to_string(),
            vec![("artificial intelligence".to_string(), 0.95)],
        );

        lexicon.add_expansions(
            "API".to_string(),
            vec![("application programming interface".to_string(), 0.9)],
        );

        lexicon.add_expansions(
            "HTTP".to_string(),
            vec![("hypertext transfer protocol".to_string(), 0.95)],
        );

        lexicon
    }
}

impl Lexicon for AbbreviationLexicon {
    fn lookup(&self, query: &str, _language: Option<&str>) -> Vec<QueryVariant> {
        // Match abbreviations case-insensitively, but preserve common uppercase forms
        let query_upper = query.to_uppercase();

        self.abbreviations
            .get(&query_upper)
            .map(|expansions| {
                expansions
                    .iter()
                    .map(|(text, confidence)| {
                        QueryVariant::new(text.clone(), VariantType::Abbreviation, *confidence)
                    })
                    .collect()
            })
            .unwrap_or_default()
    }

    fn name(&self) -> &str {
        &self.name
    }

    fn supports_language(&self, _language: &str) -> bool {
        // Abbreviations are typically language-agnostic within a domain
        true
    }
}

/// Composite lexicon that combines multiple lexicons.
///
/// Useful for consulting multiple lexicon types in a single lookup.
/// Results are merged and deduplicated, keeping the highest confidence
/// for each unique text.
pub struct CompositeLexicon {
    /// Underlying lexicons (consulted in order)
    lexicons: Vec<Arc<dyn Lexicon>>,

    /// Name of this composite
    name: String,
}

impl CompositeLexicon {
    /// Create a new composite lexicon.
    #[must_use]
    pub fn new(name: String, lexicons: Vec<Arc<dyn Lexicon>>) -> Self {
        Self { lexicons, name }
    }

    /// Add a lexicon to the composite.
    pub fn add_lexicon(&mut self, lexicon: Arc<dyn Lexicon>) {
        self.lexicons.push(lexicon);
    }
}

impl Lexicon for CompositeLexicon {
    fn lookup(&self, query: &str, language: Option<&str>) -> Vec<QueryVariant> {
        let mut all_variants = Vec::new();

        for lexicon in &self.lexicons {
            // Only consult lexicons that support this language
            if language.map_or(true, |lang| lexicon.supports_language(lang)) {
                all_variants.extend(lexicon.lookup(query, language));
            }
        }

        // Deduplicate by text, keeping highest confidence
        let mut seen: HashMap<String, f32> = HashMap::new();
        for variant in &all_variants {
            seen.entry(variant.text.clone())
                .and_modify(|conf| *conf = conf.max(variant.confidence))
                .or_insert(variant.confidence);
        }

        // Reconstruct deduplicated variants
        let mut deduplicated: Vec<QueryVariant> = seen
            .into_iter()
            .map(|(text, confidence)| {
                // Find the variant type from the original (prefer specific types)
                let variant_type = all_variants
                    .iter()
                    .find(|v| v.text == text)
                    .map(|v| v.variant_type)
                    .unwrap_or(VariantType::Synonym);

                QueryVariant::new(text, variant_type, confidence)
            })
            .collect();

        // Sort by confidence descending
        deduplicated.sort_by(|a, b| b.confidence.partial_cmp(&a.confidence).unwrap_or(std::cmp::Ordering::Equal));

        deduplicated
    }

    fn name(&self) -> &str {
        &self.name
    }

    fn supports_language(&self, language: &str) -> bool {
        // Composite supports language if any underlying lexicon supports it
        self.lexicons.iter().any(|lex| lex.supports_language(language))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_synonym_lexicon_lookup() {
        let lexicon = SynonymLexicon::with_test_data();

        let variants = lexicon.lookup("car", Some("en"));
        assert_eq!(variants.len(), 2);
        assert_eq!(variants[0].text, "automobile");
        assert_eq!(variants[0].confidence, 0.8);
        assert_eq!(variants[1].text, "vehicle");
        assert_eq!(variants[1].confidence, 0.7);
    }

    #[test]
    fn test_synonym_lexicon_case_insensitive() {
        let lexicon = SynonymLexicon::with_test_data();

        let variants_lower = lexicon.lookup("car", Some("en"));
        let variants_upper = lexicon.lookup("CAR", Some("en"));
        let variants_mixed = lexicon.lookup("Car", Some("en"));

        assert_eq!(variants_lower.len(), variants_upper.len());
        assert_eq!(variants_lower.len(), variants_mixed.len());
    }

    #[test]
    fn test_synonym_lexicon_no_match() {
        let lexicon = SynonymLexicon::with_test_data();

        let variants = lexicon.lookup("nonexistent", Some("en"));
        assert!(variants.is_empty());
    }

    #[test]
    fn test_abbreviation_lexicon_lookup() {
        let lexicon = AbbreviationLexicon::with_test_data();

        let variants = lexicon.lookup("ML", Some("en"));
        assert_eq!(variants.len(), 2);
        assert_eq!(variants[0].text, "machine learning");
        assert_eq!(variants[0].confidence, 0.85);
        assert_eq!(variants[1].text, "maximum likelihood");
        assert_eq!(variants[1].confidence, 0.4);
    }

    #[test]
    fn test_abbreviation_lexicon_case_insensitive() {
        let lexicon = AbbreviationLexicon::with_test_data();

        let variants_upper = lexicon.lookup("ML", Some("en"));
        let variants_lower = lexicon.lookup("ml", Some("en"));

        assert_eq!(variants_upper.len(), variants_lower.len());
        assert_eq!(variants_upper[0].text, variants_lower[0].text);
    }

    #[test]
    fn test_abbreviation_lexicon_unambiguous() {
        let lexicon = AbbreviationLexicon::with_test_data();

        let variants = lexicon.lookup("API", Some("en"));
        assert_eq!(variants.len(), 1);
        assert_eq!(variants[0].text, "application programming interface");
        assert_eq!(variants[0].confidence, 0.9);
    }

    #[test]
    fn test_composite_lexicon() {
        let synonym_lex = Arc::new(SynonymLexicon::with_test_data()) as Arc<dyn Lexicon>;
        let abbrev_lex = Arc::new(AbbreviationLexicon::with_test_data()) as Arc<dyn Lexicon>;

        let composite = CompositeLexicon::new(
            "test-composite".to_string(),
            vec![synonym_lex, abbrev_lex],
        );

        // Should find synonyms for "car"
        let car_variants = composite.lookup("car", Some("en"));
        assert!(car_variants.len() >= 2);

        // Should find expansions for "ML"
        let ml_variants = composite.lookup("ML", Some("en"));
        assert!(ml_variants.len() >= 2);
    }

    #[test]
    fn test_composite_lexicon_deduplication() {
        let mut lex1 = SynonymLexicon::new("lex1".to_string(), Some("en".to_string()));
        lex1.add_synonym("test".to_string(), "example".to_string(), 0.7);

        let mut lex2 = SynonymLexicon::new("lex2".to_string(), Some("en".to_string()));
        lex2.add_synonym("test".to_string(), "example".to_string(), 0.9); // Higher confidence

        let composite = CompositeLexicon::new(
            "test-composite".to_string(),
            vec![Arc::new(lex1), Arc::new(lex2)],
        );

        let variants = composite.lookup("test", Some("en"));

        // Should deduplicate and keep highest confidence
        assert_eq!(variants.len(), 1);
        assert_eq!(variants[0].text, "example");
        assert_eq!(variants[0].confidence, 0.9);
    }

    #[test]
    fn test_lexicon_language_support() {
        let lexicon = SynonymLexicon::new("english-only".to_string(), Some("en".to_string()));
        assert!(lexicon.supports_language("en"));
        assert!(!lexicon.supports_language("es"));

        let multilingual = SynonymLexicon::new("multilingual".to_string(), None);
        assert!(multilingual.supports_language("en"));
        assert!(multilingual.supports_language("es"));
    }
}
