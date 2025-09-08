//! Null implementations for graceful feature degradation
//!
//! This module provides null object implementations for all feature providers,
//! ensuring the system can run even when optional features are disabled.

use super::*;
use crate::{Confidence, Episode, Memory};
use std::any::Any;
use std::path::Path;
use std::sync::Arc;
use std::time::Duration;

// ============================================================================
// Null Index Provider
// ============================================================================

/// Null index provider that falls back to linear search
pub struct NullIndexProvider;

impl NullIndexProvider {
    pub fn new() -> Self {
        Self
    }
}

impl FeatureProvider for NullIndexProvider {
    fn is_enabled(&self) -> bool {
        false
    }
    
    fn name(&self) -> &'static str {
        "index_null"
    }
    
    fn description(&self) -> &'static str {
        "Fallback linear search when HNSW is disabled"
    }
    
    fn as_any(&self) -> &dyn Any {
        self
    }
}

impl index::IndexProvider for NullIndexProvider {
    fn create_index(&self) -> Box<dyn index::Index> {
        Box::new(NullIndex::default())
    }
    
    fn get_config(&self) -> index::IndexConfig {
        index::IndexConfig::default()
    }
}

/// Null index implementation using linear search
#[derive(Default)]
struct NullIndex {
    episodes: Vec<Episode>,
}

impl index::Index for NullIndex {
    fn build(&mut self, episodes: &[Episode]) -> index::IndexResult<()> {
        self.episodes = episodes.to_vec();
        Ok(())
    }
    
    fn search(&self, query: &[f32; 768], k: usize) -> index::IndexResult<Vec<(String, f32)>> {
        use crate::compute::dispatch::DispatchVectorOps;
        use crate::compute::VectorOps;
        
        let processor = DispatchVectorOps::new();
        let mut results: Vec<(String, f32)> = self.episodes
            .iter()
            .map(|ep| {
                let similarity = processor.cosine_similarity_768(query, &ep.embedding);
                (ep.id.clone(), similarity)
            })
            .collect();
        
        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        results.truncate(k);
        
        Ok(results)
    }
    
    fn add(&mut self, episode: &Episode) -> index::IndexResult<()> {
        self.episodes.push(episode.clone());
        Ok(())
    }
    
    fn remove(&mut self, id: &str) -> index::IndexResult<()> {
        self.episodes.retain(|ep| ep.id != id);
        Ok(())
    }
    
    fn size(&self) -> usize {
        self.episodes.len()
    }
    
    fn clear(&mut self) {
        self.episodes.clear();
    }
}

// ============================================================================
// Null Storage Provider
// ============================================================================

/// Null storage provider that uses in-memory storage
pub struct NullStorageProvider;

impl NullStorageProvider {
    pub fn new() -> Self {
        Self
    }
}

impl FeatureProvider for NullStorageProvider {
    fn is_enabled(&self) -> bool {
        false
    }
    
    fn name(&self) -> &'static str {
        "storage_null"
    }
    
    fn description(&self) -> &'static str {
        "In-memory storage when memory-mapped persistence is disabled"
    }
    
    fn as_any(&self) -> &dyn Any {
        self
    }
}

impl storage::StorageProvider for NullStorageProvider {
    fn create_storage(&self, _path: &Path) -> Box<dyn storage::Storage> {
        Box::new(NullStorage::default())
    }
    
    fn get_config(&self) -> storage::StorageConfig {
        storage::StorageConfig::default()
    }
}

/// Null storage implementation using HashMap
#[derive(Default)]
struct NullStorage {
    episodes: std::collections::HashMap<String, Episode>,
}

impl storage::Storage for NullStorage {
    fn store(&mut self, episode: &Episode) -> storage::StorageResult<()> {
        self.episodes.insert(episode.id.clone(), episode.clone());
        Ok(())
    }
    
    fn retrieve(&self, id: &str) -> storage::StorageResult<Option<Episode>> {
        Ok(self.episodes.get(id).cloned())
    }
    
    fn delete(&mut self, id: &str) -> storage::StorageResult<()> {
        self.episodes.remove(id);
        Ok(())
    }
    
    fn list_ids(&self) -> storage::StorageResult<Vec<String>> {
        Ok(self.episodes.keys().cloned().collect())
    }
    
    fn flush(&mut self) -> storage::StorageResult<()> {
        // No-op for in-memory storage
        Ok(())
    }
    
    fn stats(&self) -> storage::StorageStats {
        storage::StorageStats {
            total_items: self.episodes.len(),
            total_bytes: self.episodes.len() * std::mem::size_of::<Episode>(),
            compression_ratio: 1.0,
        }
    }
}

// ============================================================================
// Null Decay Provider
// ============================================================================

/// Null decay provider that uses simple time-based decay
pub struct NullDecayProvider;

impl NullDecayProvider {
    pub fn new() -> Self {
        Self
    }
}

impl FeatureProvider for NullDecayProvider {
    fn is_enabled(&self) -> bool {
        false
    }
    
    fn name(&self) -> &'static str {
        "decay_null"
    }
    
    fn description(&self) -> &'static str {
        "Simple time-based decay when psychological models are disabled"
    }
    
    fn as_any(&self) -> &dyn Any {
        self
    }
}

impl decay::DecayProvider for NullDecayProvider {
    fn create_decay(&self) -> Box<dyn decay::Decay> {
        Box::new(NullDecay::default())
    }
    
    fn get_config(&self) -> decay::DecayConfig {
        decay::DecayConfig::default()
    }
}

/// Null decay implementation using simple exponential decay
#[derive(Default)]
struct NullDecay {
    params: decay::DecayParameters,
}

impl decay::Decay for NullDecay {
    fn calculate_decay(&self, elapsed: Duration) -> f32 {
        let hours = elapsed.as_secs_f32() / 3600.0;
        (-self.params.base_rate * hours).exp()
    }
    
    fn apply_decay(&self, episode: &mut Episode, elapsed: Duration) {
        let decay_factor = self.calculate_decay(elapsed);
        episode.decay_rate = decay_factor;
        
        let current_confidence = episode.encoding_confidence.raw();
        episode.encoding_confidence = Confidence::exact(current_confidence * decay_factor);
    }
    
    fn get_parameters(&self) -> decay::DecayParameters {
        self.params.clone()
    }
    
    fn set_parameters(&mut self, params: decay::DecayParameters) -> decay::DecayResult<()> {
        self.params = params;
        Ok(())
    }
}

// ============================================================================
// Null Monitoring Provider
// ============================================================================

/// Null monitoring provider that discards all metrics
pub struct NullMonitoringProvider;

impl NullMonitoringProvider {
    pub fn new() -> Self {
        Self
    }
}

impl FeatureProvider for NullMonitoringProvider {
    fn is_enabled(&self) -> bool {
        false
    }
    
    fn name(&self) -> &'static str {
        "monitoring_null"
    }
    
    fn description(&self) -> &'static str {
        "No-op monitoring when Prometheus is disabled"
    }
    
    fn as_any(&self) -> &dyn Any {
        self
    }
}

impl monitoring::MonitoringProvider for NullMonitoringProvider {
    fn create_monitoring(&self) -> Box<dyn monitoring::Monitoring> {
        Box::new(NullMonitoring)
    }
    
    fn get_config(&self) -> monitoring::MonitoringConfig {
        monitoring::MonitoringConfig::default()
    }
}

/// Null monitoring implementation that discards metrics
struct NullMonitoring;

impl monitoring::Monitoring for NullMonitoring {
    fn record_counter(&self, _name: &str, _value: u64, _labels: &[(String, String)]) {
        // No-op
    }
    
    fn record_gauge(&self, _name: &str, _value: f64, _labels: &[(String, String)]) {
        // No-op
    }
    
    fn record_histogram(&self, _name: &str, _value: f64, _labels: &[(String, String)]) {
        // No-op
    }
    
    fn start_timer(&self, _name: &str) -> Box<dyn monitoring::Timer> {
        Box::new(NullTimer)
    }
    
    fn get_metric(&self, _name: &str) -> monitoring::MonitoringResult<monitoring::MetricValue> {
        Ok(monitoring::MetricValue::Counter(0))
    }
}

struct NullTimer;

impl monitoring::Timer for NullTimer {
    fn stop(self: Box<Self>) {
        // No-op
    }
}

// ============================================================================
// Null Completion Provider
// ============================================================================

/// Null completion provider that uses simple similarity matching
pub struct NullCompletionProvider;

impl NullCompletionProvider {
    pub fn new() -> Self {
        Self
    }
}

impl FeatureProvider for NullCompletionProvider {
    fn is_enabled(&self) -> bool {
        false
    }
    
    fn name(&self) -> &'static str {
        "completion_null"
    }
    
    fn description(&self) -> &'static str {
        "Simple similarity matching when pattern completion is disabled"
    }
    
    fn as_any(&self) -> &dyn Any {
        self
    }
}

impl completion::CompletionProvider for NullCompletionProvider {
    fn create_completion(&self) -> Box<dyn completion::Completion> {
        Box::new(NullCompletion)
    }
    
    fn get_config(&self) -> completion::CompletionConfig {
        completion::CompletionConfig::default()
    }
}

/// Null completion implementation using simple similarity
struct NullCompletion;

impl completion::Completion for NullCompletion {
    fn complete(
        &self,
        partial: &Memory,
        candidates: &[Arc<Memory>],
        threshold: f32,
    ) -> completion::CompletionResult<Vec<completion::CompletionMatch>> {
        use crate::compute::dispatch::DispatchVectorOps;
        use crate::compute::VectorOps;
        
        let processor = DispatchVectorOps::new();
        let mut matches = Vec::new();
        
        for candidate in candidates {
            let similarity = processor.cosine_similarity_768(
                &partial.embedding,
                &candidate.embedding,
            );
            
            if similarity >= threshold {
                matches.push(completion::CompletionMatch {
                    memory: candidate.clone(),
                    confidence: Confidence::from_raw(similarity),
                    similarity,
                });
            }
        }
        
        matches.sort_by(|a, b| b.similarity.partial_cmp(&a.similarity).unwrap());
        Ok(matches)
    }
    
    fn predict_next(
        &self,
        _sequence: &[Memory],
        _candidates: &[Arc<Memory>],
    ) -> completion::CompletionResult<Vec<completion::PredictionResult>> {
        // Simple fallback: no sequence prediction
        Err(completion::CompletionError::CompletionFailed(
            "Sequence prediction not available in null implementation".to_string()
        ))
    }
    
    fn fill_gaps(
        &self,
        pattern: &Memory,
        _mask: &[bool],
    ) -> completion::CompletionResult<Memory> {
        // Simple fallback: return pattern as-is
        Ok(pattern.clone())
    }
}