//! Provenance tracking for embeddings to ensure auditability and reproducibility.
//!
//! This module implements metadata structures that track the origin of every embedding,
//! including model version, language, timestamp, and quality metrics. This enables:
//!
//! - Auditing which model generated which embeddings
//! - Migrating embeddings when upgrading models
//! - Debugging semantic recall issues
//! - Understanding embedding quality degradation
//!
//! All provenance data is serializable and stored alongside embeddings for long-term tracking.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::SystemTime;

/// Model version information for reproducibility and migration.
///
/// Every embedding must track which model generated it to enable:
/// - Model upgrades without re-embedding everything at once
/// - Debugging issues specific to model versions
/// - Auditability of semantic recall decisions
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ModelVersion {
    /// Model name (e.g., "sentence-transformers/paraphrase-multilingual-mpnet-base-v2")
    pub name: String,

    /// Model version (semantic version or git hash)
    pub version: String,

    /// Embedding vector dimension
    pub dimension: usize,
}

impl ModelVersion {
    /// Create a new model version.
    #[must_use]
    pub const fn new(name: String, version: String, dimension: usize) -> Self {
        Self {
            name,
            version,
            dimension,
        }
    }

    /// Check if this model version is compatible with another (same name and dimension).
    #[must_use]
    pub fn is_compatible_with(&self, other: &Self) -> bool {
        self.name == other.name && self.dimension == other.dimension
    }
}

/// Provenance metadata for an embedding, tracking its origin and quality.
///
/// This structure answers the questions:
/// - Who generated this embedding? (model name and version)
/// - When was it generated? (timestamp)
/// - What language is it? (ISO 639-1 code)
/// - How confident is the model? (quality score)
/// - What additional context is relevant? (extensible metadata)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbeddingProvenance {
    /// Model that generated this embedding
    pub model: ModelVersion,

    /// Language of the embedded text (ISO 639-1 code: en, es, zh, etc.)
    pub language: Option<String>,

    /// When this embedding was created
    #[serde(with = "system_time_serde")]
    pub created_at: SystemTime,

    /// Model's self-reported quality/confidence score (0.0-1.0)
    pub quality_score: Option<f32>,

    /// Extensible key-value metadata for additional provenance information
    #[serde(default)]
    pub metadata: HashMap<String, String>,
}

impl EmbeddingProvenance {
    /// Create new provenance with required fields.
    #[must_use]
    pub fn new(model: ModelVersion, language: Option<String>) -> Self {
        Self {
            model,
            language,
            created_at: SystemTime::now(),
            quality_score: None,
            metadata: HashMap::new(),
        }
    }

    /// Create provenance with a quality score.
    #[must_use]
    pub fn with_quality(model: ModelVersion, language: Option<String>, quality_score: f32) -> Self {
        Self {
            model,
            language,
            created_at: SystemTime::now(),
            quality_score: Some(quality_score.clamp(0.0, 1.0)),
            metadata: HashMap::new(),
        }
    }

    /// Add metadata key-value pair.
    #[must_use]
    pub fn with_metadata(mut self, key: String, value: String) -> Self {
        self.metadata.insert(key, value);
        self
    }

    /// Get metadata value by key.
    #[must_use]
    pub fn get_metadata(&self, key: &str) -> Option<&String> {
        self.metadata.get(key)
    }

    /// Check if this provenance indicates high-quality embedding.
    #[must_use]
    pub fn is_high_quality(&self) -> bool {
        self.quality_score.is_none_or(|score| score >= 0.7)
    }
}

/// Embedding vector paired with its provenance metadata.
///
/// This is the primary type returned by `EmbeddingProvider` implementations.
/// It ensures embeddings are never generated without provenance tracking.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbeddingWithProvenance {
    /// The embedding vector (dimension must match provenance.model.dimension)
    pub vector: Vec<f32>,

    /// Provenance metadata for this embedding
    pub provenance: EmbeddingProvenance,
}

impl EmbeddingWithProvenance {
    /// Create a new embedding with provenance.
    ///
    /// # Panics
    ///
    /// Panics if vector dimension doesn't match model dimension in provenance.
    #[must_use]
    pub fn new(vector: Vec<f32>, provenance: EmbeddingProvenance) -> Self {
        assert_eq!(
            vector.len(),
            provenance.model.dimension,
            "Embedding vector dimension {} must match model dimension {}",
            vector.len(),
            provenance.model.dimension
        );
        Self { vector, provenance }
    }

    /// Get the model version that generated this embedding.
    #[must_use]
    pub const fn model(&self) -> &ModelVersion {
        &self.provenance.model
    }

    /// Get the language of this embedding.
    #[must_use]
    pub fn language(&self) -> Option<&str> {
        self.provenance.language.as_deref()
    }

    /// Get the quality score if available.
    #[must_use]
    pub const fn quality_score(&self) -> Option<f32> {
        self.provenance.quality_score
    }
}

/// Custom serialization for SystemTime to handle cross-platform compatibility.
mod system_time_serde {
    use serde::{Deserialize, Deserializer, Serialize, Serializer};
    use std::time::{Duration, SystemTime, UNIX_EPOCH};

    pub fn serialize<S>(time: &SystemTime, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let duration = time
            .duration_since(UNIX_EPOCH)
            .map_err(serde::ser::Error::custom)?;
        duration.as_secs().serialize(serializer)
    }

    pub fn deserialize<'de, D>(deserializer: D) -> Result<SystemTime, D::Error>
    where
        D: Deserializer<'de>,
    {
        let secs = u64::deserialize(deserializer)?;
        Ok(UNIX_EPOCH + Duration::from_secs(secs))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_model_version_creation() {
        let model = ModelVersion::new("test-model".to_string(), "1.0.0".to_string(), 768);

        assert_eq!(model.name, "test-model");
        assert_eq!(model.version, "1.0.0");
        assert_eq!(model.dimension, 768);
    }

    #[test]
    fn test_model_version_compatibility() {
        let model1 = ModelVersion::new("model-a".to_string(), "1.0.0".to_string(), 768);
        let model2 = ModelVersion::new("model-a".to_string(), "1.0.1".to_string(), 768);
        let model3 = ModelVersion::new("model-b".to_string(), "1.0.0".to_string(), 768);
        let model4 = ModelVersion::new("model-a".to_string(), "1.0.0".to_string(), 512);

        // Same name and dimension => compatible
        assert!(model1.is_compatible_with(&model2));

        // Different name => incompatible
        assert!(!model1.is_compatible_with(&model3));

        // Different dimension => incompatible
        assert!(!model1.is_compatible_with(&model4));
    }

    #[test]
    fn test_embedding_provenance_creation() {
        let model = ModelVersion::new("test-model".to_string(), "1.0.0".to_string(), 768);
        let provenance = EmbeddingProvenance::new(model.clone(), Some("en".to_string()));

        assert_eq!(provenance.model, model);
        assert_eq!(provenance.language, Some("en".to_string()));
        assert!(provenance.quality_score.is_none());
        assert!(provenance.metadata.is_empty());
    }

    #[test]
    fn test_embedding_provenance_with_quality() {
        let model = ModelVersion::new("test-model".to_string(), "1.0.0".to_string(), 768);
        let provenance = EmbeddingProvenance::with_quality(model, Some("es".to_string()), 0.85);

        assert_eq!(provenance.quality_score, Some(0.85));
        assert!(provenance.is_high_quality());
    }

    #[test]
    fn test_embedding_provenance_quality_clamping() {
        let model = ModelVersion::new("test-model".to_string(), "1.0.0".to_string(), 768);

        // Test upper bound clamping
        let provenance1 = EmbeddingProvenance::with_quality(model.clone(), None, 1.5);
        assert_eq!(provenance1.quality_score, Some(1.0));

        // Test lower bound clamping
        let provenance2 = EmbeddingProvenance::with_quality(model, None, -0.5);
        assert_eq!(provenance2.quality_score, Some(0.0));
    }

    #[test]
    fn test_embedding_provenance_metadata() {
        let model = ModelVersion::new("test-model".to_string(), "1.0.0".to_string(), 768);
        let provenance = EmbeddingProvenance::new(model, Some("en".to_string()))
            .with_metadata("source".to_string(), "user_input".to_string())
            .with_metadata("context".to_string(), "search_query".to_string());

        assert_eq!(
            provenance.get_metadata("source"),
            Some(&"user_input".to_string())
        );
        assert_eq!(
            provenance.get_metadata("context"),
            Some(&"search_query".to_string())
        );
        assert_eq!(provenance.get_metadata("unknown"), None);
    }

    #[test]
    fn test_embedding_with_provenance_creation() {
        let model = ModelVersion::new("test-model".to_string(), "1.0.0".to_string(), 768);
        let provenance = EmbeddingProvenance::new(model, Some("en".to_string()));
        let vector = vec![0.1; 768];

        let embedding = EmbeddingWithProvenance::new(vector, provenance);

        assert_eq!(embedding.vector.len(), 768);
        assert_eq!(embedding.language(), Some("en"));
        assert_eq!(embedding.model().dimension, 768);
    }

    #[test]
    #[should_panic(expected = "Embedding vector dimension 512 must match model dimension 768")]
    fn test_embedding_with_provenance_dimension_mismatch() {
        let model = ModelVersion::new("test-model".to_string(), "1.0.0".to_string(), 768);
        let provenance = EmbeddingProvenance::new(model, Some("en".to_string()));
        let wrong_vector = vec![0.1; 512]; // Wrong dimension

        let _ = EmbeddingWithProvenance::new(wrong_vector, provenance);
    }

    #[test]
    fn test_provenance_serialization() {
        let model = ModelVersion::new("test-model".to_string(), "1.0.0".to_string(), 768);
        let provenance = EmbeddingProvenance::with_quality(model, Some("en".to_string()), 0.9)
            .with_metadata("key".to_string(), "value".to_string());

        // Test JSON serialization
        let json = serde_json::to_string(&provenance).unwrap_or_default();
        assert!(!json.is_empty(), "serialization should succeed");
        let deserialized: Result<EmbeddingProvenance, _> = serde_json::from_str(&json);
        assert!(deserialized.is_ok(), "deserialization should succeed");

        if let Ok(deserialized) = deserialized {
            assert_eq!(provenance.model, deserialized.model);
            assert_eq!(provenance.language, deserialized.language);
            assert_eq!(provenance.quality_score, deserialized.quality_score);
            assert_eq!(
                provenance.get_metadata("key"),
                deserialized.get_metadata("key")
            );
        }
    }

    #[test]
    fn test_embedding_with_provenance_serialization() {
        let model = ModelVersion::new("test-model".to_string(), "1.0.0".to_string(), 4);
        let provenance = EmbeddingProvenance::new(model, Some("en".to_string()));
        let vector = vec![0.1, 0.2, 0.3, 0.4];
        let embedding = EmbeddingWithProvenance::new(vector, provenance);

        // Test JSON serialization
        let json = serde_json::to_string(&embedding).unwrap_or_default();
        assert!(!json.is_empty(), "serialization should succeed");
        let deserialized: Result<EmbeddingWithProvenance, _> = serde_json::from_str(&json);
        assert!(deserialized.is_ok(), "deserialization should succeed");

        if let Ok(deserialized) = deserialized {
            assert_eq!(embedding.vector, deserialized.vector);
            assert_eq!(embedding.provenance.model, deserialized.provenance.model);
        }
    }

    #[test]
    fn test_is_high_quality() {
        let model = ModelVersion::new("test-model".to_string(), "1.0.0".to_string(), 768);

        // No quality score => assume high quality
        let provenance1 = EmbeddingProvenance::new(model.clone(), None);
        assert!(provenance1.is_high_quality());

        // Quality score >= 0.7 => high quality
        let provenance2 = EmbeddingProvenance::with_quality(model.clone(), None, 0.8);
        assert!(provenance2.is_high_quality());

        // Quality score < 0.7 => not high quality
        let provenance3 = EmbeddingProvenance::with_quality(model, None, 0.6);
        assert!(!provenance3.is_high_quality());
    }
}
