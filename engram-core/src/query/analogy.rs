//! Vector analogy operations for figurative language interpretation.
//!
//! This module provides pure mathematical operations for computing semantic analogies
//! using vector arithmetic. The classic example is:
//!
//! ```text
//! king - man + woman ≈ queen
//! ```
//!
//! ## Design Principles
//!
//! - **Pure Functions**: All operations are stateless and deterministic
//! - **Numerical Stability**: Careful handling of edge cases (zero vectors, normalization)
//! - **Type Safety**: Strong typing prevents dimension mismatches
//! - **Testability**: Easy to unit test without external dependencies
//!
//! ## Mathematical Foundation
//!
//! Semantic analogies leverage the property that word embeddings encode semantic
//! relationships as vector offsets. For example:
//!
//! - `fast - slow ≈ hot - cold` (intensity relationship)
//! - `Paris - France ≈ London - England` (capital-country relationship)
//!
//! We compute analogies by:
//! 1. Computing the offset between source terms: `offset = source2 - source1`
//! 2. Applying to target: `result = target + offset`
//! 3. Normalizing to unit length for similarity comparison

use std::fmt;

/// Error type for vector analogy operations.
#[derive(Debug, thiserror::Error)]
pub enum AnalogyError {
    /// Vector dimensions don't match
    #[error("dimension mismatch: expected {expected}, got {actual}")]
    DimensionMismatch {
        /// Expected dimension
        expected: usize,
        /// Actual dimension
        actual: usize,
    },

    /// Vector is zero or near-zero (cannot normalize)
    #[error("zero vector cannot be normalized")]
    ZeroVector,

    /// Numerical instability detected
    #[error("numerical instability: {reason}")]
    NumericalInstability {
        /// Reason for instability
        reason: String,
    },
}

/// A semantic analogy pattern detected in a query.
///
/// Represents queries like "X as Y" or "X like Y" where the relationship
/// between X and Y can be transferred to other concepts.
///
/// # Examples
///
/// - "fast as cheetah" → AnalogyPattern { target: "fast", relation: "as", source: "cheetah" }
/// - "brave like lion" → AnalogyPattern { target: "brave", relation: "like", source: "lion" }
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AnalogyPattern {
    /// Target concept (what we're describing)
    pub target: String,

    /// Relation type ("as", "like", "is")
    pub relation: AnalogyRelation,

    /// Source concept (what we're comparing to)
    pub source: String,
}

impl AnalogyPattern {
    /// Create a new analogy pattern.
    #[must_use]
    pub const fn new(target: String, relation: AnalogyRelation, source: String) -> Self {
        Self {
            target,
            relation,
            source,
        }
    }
}

impl fmt::Display for AnalogyPattern {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} {} {}", self.target, self.relation, self.source)
    }
}

/// Type of analogical relation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AnalogyRelation {
    /// Simile comparison ("as")
    As,

    /// Simile comparison ("like")
    Like,

    /// Metaphorical equivalence ("is")
    Is,
}

impl fmt::Display for AnalogyRelation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::As => write!(f, "as"),
            Self::Like => write!(f, "like"),
            Self::Is => write!(f, "is"),
        }
    }
}

/// Vector analogy engine for computing semantic relationships.
///
/// This provides pure mathematical operations on embedding vectors.
/// All operations are stateless and thread-safe.
pub struct AnalogyEngine;

impl AnalogyEngine {
    /// Subtract two vectors element-wise.
    ///
    /// Computes `a - b`, representing the semantic offset from b to a.
    ///
    /// # Arguments
    ///
    /// * `a` - First vector (minuend)
    /// * `b` - Second vector (subtrahend)
    ///
    /// # Returns
    ///
    /// Vector representing the difference `a - b`.
    ///
    /// # Errors
    ///
    /// Returns `DimensionMismatch` if vectors have different lengths.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let king = vec![0.1, 0.2, 0.3];
    /// let man = vec![0.05, 0.1, 0.15];
    /// let offset = AnalogyEngine::subtract(&king, &man)?;
    /// // offset ≈ [0.05, 0.1, 0.15]
    /// ```
    pub fn subtract(a: &[f32], b: &[f32]) -> Result<Vec<f32>, AnalogyError> {
        if a.len() != b.len() {
            return Err(AnalogyError::DimensionMismatch {
                expected: a.len(),
                actual: b.len(),
            });
        }

        Ok(a.iter().zip(b.iter()).map(|(x, y)| x - y).collect())
    }

    /// Add two vectors element-wise.
    ///
    /// Computes `a + b`, combining two semantic representations.
    ///
    /// # Arguments
    ///
    /// * `a` - First vector
    /// * `b` - Second vector (offset to apply)
    ///
    /// # Returns
    ///
    /// Vector representing the sum `a + b`.
    ///
    /// # Errors
    ///
    /// Returns `DimensionMismatch` if vectors have different lengths.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let woman = vec![0.04, 0.08, 0.12];
    /// let gender_offset = vec![0.05, 0.1, 0.15];
    /// let result = AnalogyEngine::add(&woman, &gender_offset)?;
    /// // result ≈ queen embedding
    /// ```
    pub fn add(a: &[f32], b: &[f32]) -> Result<Vec<f32>, AnalogyError> {
        if a.len() != b.len() {
            return Err(AnalogyError::DimensionMismatch {
                expected: a.len(),
                actual: b.len(),
            });
        }

        Ok(a.iter().zip(b.iter()).map(|(x, y)| x + y).collect())
    }

    /// Normalize a vector to unit length (L2 norm = 1).
    ///
    /// Essential for comparing vectors using cosine similarity after analogy operations.
    ///
    /// # Arguments
    ///
    /// * `v` - Vector to normalize
    ///
    /// # Returns
    ///
    /// Normalized vector with L2 norm = 1.
    ///
    /// # Errors
    ///
    /// Returns `ZeroVector` if the input vector has zero magnitude.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let v = vec![3.0, 4.0];  // magnitude = 5.0
    /// let normalized = AnalogyEngine::normalize(&v)?;
    /// // normalized ≈ [0.6, 0.8]
    /// ```
    pub fn normalize(v: &[f32]) -> Result<Vec<f32>, AnalogyError> {
        // Compute L2 norm (magnitude)
        let magnitude: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();

        // Check for zero vector
        if magnitude < 1e-8 {
            return Err(AnalogyError::ZeroVector);
        }

        // Normalize by dividing each component by magnitude
        Ok(v.iter().map(|x| x / magnitude).collect())
    }

    /// Compute semantic analogy: `target + (source2 - source1)`.
    ///
    /// This is the core operation for analogical reasoning. It computes:
    /// 1. The offset from source1 to source2
    /// 2. Applies that offset to target
    /// 3. Normalizes the result for similarity comparison
    ///
    /// # Arguments
    ///
    /// * `target` - Base concept to apply analogy to
    /// * `source1` - First term in source relation
    /// * `source2` - Second term in source relation
    ///
    /// # Returns
    ///
    /// Normalized vector representing the analogical result.
    ///
    /// # Errors
    ///
    /// Returns error if vectors have mismatched dimensions or result is zero.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// // "queen is to woman as king is to ___?"
    /// let queen_emb = embed("queen");
    /// let woman_emb = embed("woman");
    /// let king_emb = embed("king");
    /// let result = AnalogyEngine::compute_analogy(&king_emb, &woman_emb, &queen_emb)?;
    /// // result ≈ man embedding
    /// ```
    pub fn compute_analogy(
        target: &[f32],
        source1: &[f32],
        source2: &[f32],
    ) -> Result<Vec<f32>, AnalogyError> {
        // Step 1: Compute offset (source2 - source1)
        let offset = Self::subtract(source2, source1)?;

        // Step 2: Apply offset to target
        let result = Self::add(target, &offset)?;

        // Step 3: Normalize for similarity comparison
        Self::normalize(&result)
    }

    /// Compute cosine similarity between two vectors.
    ///
    /// Returns value in [-1, 1] where:
    /// - 1.0 = identical direction
    /// - 0.0 = orthogonal
    /// - -1.0 = opposite direction
    ///
    /// # Arguments
    ///
    /// * `a` - First vector (should be normalized)
    /// * `b` - Second vector (should be normalized)
    ///
    /// # Returns
    ///
    /// Cosine similarity value in [-1, 1].
    ///
    /// # Errors
    ///
    /// Returns `DimensionMismatch` if vectors have different lengths.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let v1 = vec![1.0, 0.0];
    /// let v2 = vec![0.0, 1.0];
    /// let similarity = AnalogyEngine::cosine_similarity(&v1, &v2)?;
    /// // similarity ≈ 0.0 (orthogonal)
    /// ```
    pub fn cosine_similarity(a: &[f32], b: &[f32]) -> Result<f32, AnalogyError> {
        if a.len() != b.len() {
            return Err(AnalogyError::DimensionMismatch {
                expected: a.len(),
                actual: b.len(),
            });
        }

        let dot_product: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        Ok(dot_product.clamp(-1.0, 1.0))
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)] // Tests are allowed to use unwrap

    use super::*;

    const EPSILON: f32 = 1e-6;

    fn assert_vec_close(actual: &[f32], expected: &[f32], epsilon: f32) {
        assert_eq!(actual.len(), expected.len(), "vector lengths differ");
        for (i, (a, e)) in actual.iter().zip(expected.iter()).enumerate() {
            assert!(
                (a - e).abs() < epsilon,
                "vector[{}]: expected {}, got {} (diff: {})",
                i,
                e,
                a,
                (a - e).abs()
            );
        }
    }

    #[test]
    fn test_vector_subtract() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![0.5, 1.0, 1.5];
        let result = AnalogyEngine::subtract(&a, &b).unwrap();
        assert_vec_close(&result, &[0.5, 1.0, 1.5], EPSILON);
    }

    #[test]
    fn test_vector_add() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![0.5, 1.0, 1.5];
        let result = AnalogyEngine::add(&a, &b).unwrap();
        assert_vec_close(&result, &[1.5, 3.0, 4.5], EPSILON);
    }

    #[test]
    fn test_dimension_mismatch() {
        let a = vec![1.0, 2.0];
        let b = vec![1.0, 2.0, 3.0];
        assert!(matches!(
            AnalogyEngine::subtract(&a, &b),
            Err(AnalogyError::DimensionMismatch { .. })
        ));
    }

    #[test]
    fn test_normalize() {
        let v = vec![3.0, 4.0]; // magnitude = 5.0
        let normalized = AnalogyEngine::normalize(&v).unwrap();
        assert_vec_close(&normalized, &[0.6, 0.8], EPSILON);

        // Verify unit length
        let magnitude: f32 = normalized.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((magnitude - 1.0).abs() < EPSILON);
    }

    #[test]
    fn test_normalize_zero_vector() {
        let zero = vec![0.0, 0.0, 0.0];
        assert!(matches!(
            AnalogyEngine::normalize(&zero),
            Err(AnalogyError::ZeroVector)
        ));
    }

    #[test]
    fn test_compute_analogy_simple() {
        // Simple analogy: [2,0] + ([4,0] - [1,0]) = [5,0]
        let target = vec![2.0, 0.0];
        let source1 = vec![1.0, 0.0];
        let source2 = vec![4.0, 0.0];

        let result = AnalogyEngine::compute_analogy(&target, &source1, &source2).unwrap();

        // Result should be normalized [5,0] → [1,0]
        assert_vec_close(&result, &[1.0, 0.0], EPSILON);
    }

    #[test]
    fn test_cosine_similarity() {
        // Identical vectors
        let v1 = vec![1.0, 0.0];
        let similarity = AnalogyEngine::cosine_similarity(&v1, &v1).unwrap();
        assert!((similarity - 1.0).abs() < EPSILON);

        // Orthogonal vectors
        let v2 = vec![0.0, 1.0];
        let similarity = AnalogyEngine::cosine_similarity(&v1, &v2).unwrap();
        assert!(similarity.abs() < EPSILON);

        // Opposite vectors
        let v3 = vec![-1.0, 0.0];
        let similarity = AnalogyEngine::cosine_similarity(&v1, &v3).unwrap();
        assert!((similarity - (-1.0)).abs() < EPSILON);
    }

    #[test]
    fn test_analogy_pattern_display() {
        let pattern = AnalogyPattern::new(
            "fast".to_string(),
            AnalogyRelation::As,
            "cheetah".to_string(),
        );
        assert_eq!(pattern.to_string(), "fast as cheetah");
    }

    #[test]
    fn test_analogy_relation_display() {
        assert_eq!(AnalogyRelation::As.to_string(), "as");
        assert_eq!(AnalogyRelation::Like.to_string(), "like");
        assert_eq!(AnalogyRelation::Is.to_string(), "is");
    }
}
