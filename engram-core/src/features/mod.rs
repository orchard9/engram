//! Feature abstraction layer for compile-time and runtime feature management
//!
//! This module provides a trait-based abstraction over feature flags, allowing
//! graceful degradation when features are disabled and runtime feature detection.
//!
//! ## Architecture Overview
//!
//! The feature provider pattern decouples optional functionality from core engine logic,
//! ensuring the system remains functional even when advanced features are disabled at
//! compile time. This follows the Null Object Pattern to eliminate feature flag checks
//! throughout the codebase.
//!
//! ## Key Components
//!
//! ### `FeatureProvider` Trait
//! The core abstraction that all feature providers must implement:
//! - `is_enabled()`: Runtime check if feature is available
//! - `name()`: Human-readable feature identifier 
//! - `description()`: Feature documentation
//! - `is_compatible_with()`: Inter-feature compatibility checking
//!
//! ### `FeatureRegistry`
//! Central registry managing all feature providers:
//! - Automatic provider registration based on compile-time flags
//! - Best-available provider selection with graceful fallback
//! - Runtime feature discovery and compatibility validation
//! - Feature status reporting
//!
//! ### Provider Interfaces
//! Each feature domain has a specific provider interface:
//! - **IndexProvider**: Vector similarity search (HNSW → linear search)
//! - **StorageProvider**: Persistence backends (memory-mapped → in-memory)
//! - **DecayProvider**: Psychological decay models (research-based → simple exponential)
//! - **MonitoringProvider**: Metrics collection (Prometheus → no-op)
//! - **CompletionProvider**: Pattern completion (neural → similarity-based)
//!
//! ## Usage Pattern
//!
//! ```rust
//! use engram_core::features::{FeatureRegistry, IndexProvider};
//!
//! // Get registry with all available providers
//! let registry = FeatureRegistry::new();
//!
//! // Get best available index provider (HNSW if available, linear search otherwise)
//! let index_provider = registry.get_best("index");
//! let index = index_provider
//!     .as_any()
//!     .downcast_ref::<Box<dyn IndexProvider>>()
//!     .unwrap()
//!     .create_index();
//!
//! // Index operations work the same regardless of implementation
//! index.build(&episodes)?;
//! let results = index.search(&query, k)?;
//! ```
//!
//! ## Graceful Degradation
//!
//! When optional features are disabled:
//! - **HNSW Index** → **Linear Search**: O(n) search with same API
//! - **Memory-Mapped Storage** → **In-Memory HashMap**: No persistence, same interface
//! - **Psychological Decay** → **Simple Exponential**: Basic time-based decay
//! - **Prometheus Monitoring** → **No-op**: All metrics calls become no-ops
//! - **Neural Completion** → **Similarity Matching**: Cosine similarity fallback
//!
//! ## Compile-Time Optimization
//!
//! Feature flags are resolved at compile time, so there's zero runtime overhead
//! for feature checking. The registry automatically selects the best available
//! implementation based on enabled features.
//!
//! ## Testing Strategy
//!
//! All providers implement the same interfaces, enabling:
//! - **Unit testing** with null implementations
//! - **Integration testing** across feature combinations
//! - **Behavioral verification** ensuring API compatibility
//! - **Performance testing** comparing implementations
//!
//! ## Extension Guidelines
//!
//! To add a new feature:
//! 1. Create provider trait in `features/{feature_name}.rs`
//! 2. Implement real provider with `#[cfg(feature = "...")]`
//! 3. Create null implementation in `features/null_impls.rs`
//! 4. Register both in `FeatureRegistry::new()`
//! 5. Add integration tests in `tests/feature_integration_tests.rs`

use std::any::Any;
use std::collections::HashMap;
use std::sync::Arc;

pub mod index;
pub mod storage;
pub mod decay;
pub mod monitoring;
pub mod completion;
pub mod null_impls;

// Re-export provider traits
pub use index::IndexProvider;
pub use storage::StorageProvider;
pub use decay::DecayProvider;
pub use monitoring::MonitoringProvider;
pub use completion::CompletionProvider;

/// Core trait for all feature providers
pub trait FeatureProvider: Send + Sync + Any {
    /// Check if this feature is enabled
    fn is_enabled(&self) -> bool;
    
    /// Get the name of this feature
    fn name(&self) -> &'static str;
    
    /// Get a human-readable description
    fn description(&self) -> &'static str;
    
    /// Check compatibility with other features
    fn is_compatible_with(&self, _other: &str) -> bool {
        // By default, all features are compatible
        true
    }
    
    /// As any for downcasting
    fn as_any(&self) -> &dyn Any;
}

/// Registry for all available features in the system
pub struct FeatureRegistry {
    providers: HashMap<&'static str, Arc<dyn FeatureProvider>>,
    compatibility_matrix: FeatureCompatibilityMatrix,
}

impl FeatureRegistry {
    /// Create a new feature registry with all available providers
    pub fn new() -> Self {
        let mut providers = HashMap::new();
        let compatibility_matrix = FeatureCompatibilityMatrix::default();
        
        // Register null implementations (always available)
        providers.insert("index_null", Arc::new(null_impls::NullIndexProvider::new()) as Arc<dyn FeatureProvider>);
        providers.insert("storage_null", Arc::new(null_impls::NullStorageProvider::new()) as Arc<dyn FeatureProvider>);
        providers.insert("decay_null", Arc::new(null_impls::NullDecayProvider::new()) as Arc<dyn FeatureProvider>);
        providers.insert("monitoring_null", Arc::new(null_impls::NullMonitoringProvider::new()) as Arc<dyn FeatureProvider>);
        providers.insert("completion_null", Arc::new(null_impls::NullCompletionProvider::new()) as Arc<dyn FeatureProvider>);
        
        // Override with real implementations if features are enabled
        #[cfg(feature = "hnsw_index")]
        {
            providers.insert("index", Arc::new(index::HnswIndexProvider::new()) as Arc<dyn FeatureProvider>);
        }
        
        #[cfg(feature = "memory_mapped_persistence")]
        {
            providers.insert("storage", Arc::new(storage::MmapStorageProvider::new()) as Arc<dyn FeatureProvider>);
        }
        
        #[cfg(feature = "psychological_decay")]
        {
            providers.insert("decay", Arc::new(decay::PsychologicalDecayProvider::new()) as Arc<dyn FeatureProvider>);
        }
        
        #[cfg(feature = "monitoring")]
        {
            providers.insert("monitoring", Arc::new(monitoring::PrometheusMonitoringProvider::new()) as Arc<dyn FeatureProvider>);
        }
        
        #[cfg(feature = "pattern_completion")]
        {
            providers.insert("completion", Arc::new(completion::PatternCompletionProvider::new()) as Arc<dyn FeatureProvider>);
        }
        
        Self {
            providers,
            compatibility_matrix,
        }
    }
    
    /// Get a feature provider by name
    pub fn get(&self, name: &str) -> Option<Arc<dyn FeatureProvider>> {
        self.providers.get(name).cloned()
    }
    
    /// Get a typed feature provider
    pub fn get_typed<T: FeatureProvider + 'static>(&self, name: &str) -> Option<Arc<T>> {
        self.providers.get(name).and_then(|provider| {
            provider.as_any().downcast_ref::<Arc<T>>().cloned()
        })
    }
    
    /// Get the best available provider for a feature type
    pub fn get_best(&self, feature_type: &str) -> Arc<dyn FeatureProvider> {
        // Try to get the real implementation first
        if let Some(provider) = self.get(feature_type) {
            if provider.is_enabled() {
                return provider;
            }
        }
        
        // Fall back to null implementation
        self.get(&format!("{}_null", feature_type))
            .expect("Null implementation should always exist")
    }
    
    /// Check if a set of features are compatible
    pub fn check_compatibility(&self, features: &[&str]) -> Result<(), String> {
        self.compatibility_matrix.check_compatibility(features)
    }
    
    /// Get all enabled features
    pub fn enabled_features(&self) -> Vec<&'static str> {
        self.providers
            .iter()
            .filter(|(name, provider)| !name.ends_with("_null") && provider.is_enabled())
            .map(|(name, _)| *name)
            .collect()
    }
    
    /// Get a summary of feature status
    pub fn status_summary(&self) -> String {
        let mut summary = String::from("Feature Status:\n");
        
        for feature_type in &["index", "storage", "decay", "monitoring", "completion"] {
            let provider = self.get_best(feature_type);
            summary.push_str(&format!(
                "  {}: {} ({})\n",
                feature_type,
                provider.name(),
                if provider.is_enabled() { "enabled" } else { "fallback" }
            ));
        }
        
        summary
    }
}

impl Default for FeatureRegistry {
    fn default() -> Self {
        Self::new()
    }
}

/// Matrix defining which features can work together
pub struct FeatureCompatibilityMatrix {
    incompatible_pairs: Vec<(&'static str, &'static str)>,
}

impl Default for FeatureCompatibilityMatrix {
    fn default() -> Self {
        Self {
            // Define any incompatible feature pairs here
            incompatible_pairs: vec![
                // Example: ("feature_a", "feature_b"),
            ],
        }
    }
}

impl FeatureCompatibilityMatrix {
    /// Check if a set of features are compatible
    pub fn check_compatibility(&self, features: &[&str]) -> Result<(), String> {
        for i in 0..features.len() {
            for j in (i + 1)..features.len() {
                for &(a, b) in &self.incompatible_pairs {
                    if (features[i] == a && features[j] == b) ||
                       (features[i] == b && features[j] == a) {
                        return Err(format!(
                            "Features '{}' and '{}' are incompatible",
                            features[i], features[j]
                        ));
                    }
                }
            }
        }
        Ok(())
    }
    
    /// Add an incompatible pair
    pub fn add_incompatible_pair(&mut self, feature_a: &'static str, feature_b: &'static str) {
        self.incompatible_pairs.push((feature_a, feature_b));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_feature_registry_creation() {
        let registry = FeatureRegistry::new();
        
        // Null implementations should always be available
        assert!(registry.get("index_null").is_some());
        assert!(registry.get("storage_null").is_some());
        assert!(registry.get("decay_null").is_some());
        assert!(registry.get("monitoring_null").is_some());
        assert!(registry.get("completion_null").is_some());
    }
    
    #[test]
    fn test_get_best_provider() {
        let registry = FeatureRegistry::new();
        
        // Should get null implementation if feature not enabled
        let index_provider = registry.get_best("index");
        assert!(index_provider.name().ends_with("null") || index_provider.is_enabled());
    }
    
    #[test]
    fn test_compatibility_checking() {
        let mut matrix = FeatureCompatibilityMatrix::default();
        matrix.add_incompatible_pair("feature_a", "feature_b");
        
        // Compatible features
        assert!(matrix.check_compatibility(&["feature_a", "feature_c"]).is_ok());
        
        // Incompatible features
        assert!(matrix.check_compatibility(&["feature_a", "feature_b"]).is_err());
    }
}