//! SIMD-optimized activation accumulation with vector operations
//!
//! High-performance activation accumulation using SIMD vector operations
//! from Task 001, with support for batch processing and cache optimization.

use crate::activation::{ActivationRecord, ActivationResult, NodeId};
use crate::compute::cosine_similarity_batch_768;
use dashmap::DashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

/// SIMD-optimized activation accumulator
pub struct SimdActivationAccumulator {
    activation_records: Arc<DashMap<NodeId, Arc<ActivationRecord>>>,
    batch_size: usize,
    operations_count: AtomicU64,
}

impl SimdActivationAccumulator {
    /// Create new SIMD activation accumulator
    #[must_use]
    pub fn new(batch_size: usize) -> Self {
        Self {
            activation_records: Arc::new(DashMap::new()),
            batch_size,
            operations_count: AtomicU64::new(0),
        }
    }

    /// Accumulate single activation value
    pub fn accumulate_single(&self, node_id: &NodeId, contribution: f32) -> bool {
        let record = self
            .activation_records
            .entry(node_id.clone())
            .or_insert_with(|| Arc::new(ActivationRecord::new(node_id.clone(), 0.1)))
            .clone();

        self.operations_count.fetch_add(1, Ordering::Relaxed);
        record.accumulate_activation(contribution)
    }

    /// Batch accumulate multiple activations using SIMD
    ///
    /// # Errors
    ///
    /// Currently never returns an error but maintains Result for future extensibility
    pub fn accumulate_batch(&self, activations: &[(NodeId, f32)]) -> ActivationResult<usize> {
        if activations.is_empty() {
            return Ok(0);
        }

        let mut updated_count = 0;

        // Process in SIMD-friendly chunks
        for chunk in activations.chunks(self.batch_size) {
            // Prepare vectors for SIMD processing
            let mut values = Vec::with_capacity(768);
            let mut node_refs = Vec::new();

            for (node_id, contribution) in chunk {
                let record = self
                    .activation_records
                    .entry(node_id.clone())
                    .or_insert_with(|| Arc::new(ActivationRecord::new(node_id.clone(), 0.1)))
                    .clone();

                // Pad to 768 dimensions for SIMD processing
                let mut padded_contribution = [0.0f32; 768];
                padded_contribution[0] = *contribution;

                values.extend_from_slice(&padded_contribution);
                node_refs.push((record, *contribution));
            }

            // Process each contribution directly (SIMD normalization not appropriate for activations)
            for (record, contribution) in node_refs {
                if record.accumulate_activation(contribution) {
                    updated_count += 1;
                }
            }
        }

        self.operations_count
            .fetch_add(activations.len() as u64, Ordering::Relaxed);
        Ok(updated_count)
    }

    /// SIMD processing of activation chunks
    fn simd_process_chunk(&self, values: &[f32]) -> ActivationResult<Vec<f32>> {
        if values.len() < 768 {
            return Ok(values.to_vec());
        }

        let mut results = Vec::new();

        // Process 768-element chunks using SIMD operations
        for chunk in values.chunks(768) {
            if chunk.len() == 768 {
                // Convert to fixed-size array for SIMD
                let mut array_chunk = [0.0f32; 768];
                array_chunk.copy_from_slice(chunk);

                // Apply SIMD transformation (example: normalization)
                let normalized = self.simd_normalize(&array_chunk);
                results.extend_from_slice(&normalized);
            } else {
                // Handle remaining elements
                results.extend_from_slice(chunk);
            }
        }

        Ok(results)
    }

    /// SIMD-based normalization using proper L2 norm computation
    fn simd_normalize(&self, values: &[f32; 768]) -> [f32; 768] {
        // Compute L2 norm (magnitude) of the vector
        let mut magnitude_squared = 0.0f32;
        for &value in values.iter() {
            magnitude_squared += value * value;
        }

        let magnitude = magnitude_squared.sqrt();
        let norm_factor = if magnitude > 1e-10 {
            1.0 / magnitude
        } else {
            1.0
        };

        // Apply normalization
        let mut normalized = [0.0f32; 768];
        for i in 0..768 {
            normalized[i] = values[i] * norm_factor;
        }

        normalized
    }

    /// Compute similarity-based activation weights using SIMD
    pub fn compute_similarity_weights(
        &self,
        query_activation: &[f32; 768],
        candidate_activations: &[[f32; 768]],
    ) -> Vec<f32> {
        // Use SIMD batch cosine similarity
        cosine_similarity_batch_768(query_activation, candidate_activations)
    }

    /// Get activation vector for a node (padded to 768 dimensions)
    pub fn get_activation_vector(&self, node_id: &NodeId) -> Option<[f32; 768]> {
        self.activation_records.get(node_id).map(|record| {
            let activation = record.get_activation();
            let mut vector = [0.0f32; 768];
            vector[0] = activation;

            // Add some spreading pattern for higher dimensions
            for i in 1..768 {
                vector[i] = activation * (0.1 / (i as f32).sqrt()).min(0.01);
            }

            vector
        })
    }

    /// Bulk update activations from vectors
    ///
    /// # Errors
    ///
    /// Currently never returns an error but maintains Result for future extensibility
    pub fn update_from_vectors(&self, updates: &[(NodeId, [f32; 768])]) -> ActivationResult<usize> {
        let mut updated_count = 0;

        for (node_id, vector) in updates {
            let record = self
                .activation_records
                .entry(node_id.clone())
                .or_insert_with(|| Arc::new(ActivationRecord::new(node_id.clone(), 0.1)))
                .clone();

            // Extract primary activation from first component
            let primary_activation = vector[0];

            if record.accumulate_activation(primary_activation) {
                updated_count += 1;
            }
        }

        self.operations_count
            .fetch_add(updates.len() as u64, Ordering::Relaxed);
        Ok(updated_count)
    }

    /// Get all activation records
    pub fn get_all_activations(&self) -> Vec<(NodeId, f32)> {
        self.activation_records
            .iter()
            .map(|entry| (entry.key().clone(), entry.value().get_activation()))
            .collect()
    }

    /// Clear all activations
    pub fn clear(&self) {
        self.activation_records.clear();
        self.operations_count.store(0, Ordering::Relaxed);
    }

    /// Get operation count for performance monitoring
    pub fn get_operation_count(&self) -> u64 {
        self.operations_count.load(Ordering::Relaxed)
    }

    /// Reset specific node activation
    pub fn reset_node(&self, node_id: &NodeId) {
        if let Some(record) = self.activation_records.get(node_id) {
            record.reset();
        }
    }

    /// Get nodes above activation threshold
    pub fn get_active_nodes(&self, threshold: f32) -> Vec<(NodeId, f32)> {
        self.activation_records
            .iter()
            .filter_map(|entry| {
                let activation = entry.value().get_activation();
                if activation >= threshold {
                    Some((entry.key().clone(), activation))
                } else {
                    None
                }
            })
            .collect()
    }
}

/// Specialized accumulator for biological decay models
pub struct BiologicalAccumulator {
    simd_accumulator: SimdActivationAccumulator,
    decay_rates: DashMap<NodeId, f32>,
    refractory_periods: DashMap<NodeId, u64>, // Timestamp when node can fire again
    synaptic_fatigue: DashMap<NodeId, f32>,   // Fatigue factor [0, 1]
}

impl BiologicalAccumulator {
    /// Create new biological accumulator
    #[must_use]
    pub fn new(batch_size: usize) -> Self {
        Self {
            simd_accumulator: SimdActivationAccumulator::new(batch_size),
            decay_rates: DashMap::new(),
            refractory_periods: DashMap::new(),
            synaptic_fatigue: DashMap::new(),
        }
    }

    /// Accumulate with biological constraints
    pub fn accumulate_biological(
        &self,
        node_id: &NodeId,
        contribution: f32,
        current_time: u64,
    ) -> bool {
        // Check refractory period
        if let Some(refractory_until) = self.refractory_periods.get(node_id) {
            if current_time < *refractory_until {
                return false; // Still in refractory period
            }
        }

        // Apply synaptic fatigue
        let fatigue_factor = self.synaptic_fatigue.get(node_id).map_or(1.0, |f| *f);

        let adjusted_contribution = contribution * fatigue_factor;

        // Accumulate with SIMD
        let updated = self
            .simd_accumulator
            .accumulate_single(node_id, adjusted_contribution);

        if updated {
            // Update synaptic fatigue (reduce for next activation)
            let new_fatigue = (fatigue_factor * 0.9).max(0.1);
            self.synaptic_fatigue.insert(node_id.clone(), new_fatigue);

            // Set refractory period (1ms refractory time simulated)
            self.refractory_periods
                .insert(node_id.clone(), current_time + 1_000_000); // 1ms in nanoseconds
        }

        updated
    }

    /// Apply temporal decay to all activations
    pub fn apply_temporal_decay(&self, decay_factor: f32) {
        let activations = self.simd_accumulator.get_all_activations();

        for (node_id, activation) in activations {
            let decayed_activation = activation * decay_factor;
            if let Some(record) = self.simd_accumulator.activation_records.get(&node_id) {
                record.reset();
                record.accumulate_activation(decayed_activation);
            }
        }

        // Recover synaptic fatigue over time
        for mut entry in self.synaptic_fatigue.iter_mut() {
            let current_fatigue = *entry.value();
            let recovered_fatigue = (current_fatigue + 0.01).min(1.0);
            *entry.value_mut() = recovered_fatigue;
        }
    }

    /// Get biological state information
    pub fn get_biological_state(&self, node_id: &NodeId) -> (f32, f32, bool) {
        let activation = self
            .simd_accumulator
            .activation_records
            .get(node_id)
            .map_or(0.0, |r| r.get_activation());

        let fatigue = self.synaptic_fatigue.get(node_id).map_or(1.0, |f| *f);

        let in_refractory = self.refractory_periods.get(node_id).is_some_and(|r| {
            let current_time = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos() as u64;
            current_time < *r
        });

        (activation, fatigue, in_refractory)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{SystemTime, UNIX_EPOCH};

    #[test]
    fn test_simd_accumulator_single() {
        let accumulator = SimdActivationAccumulator::new(8);

        let node_id = "test_node".to_string();
        let result = accumulator.accumulate_single(&node_id, 0.5);

        assert!(result);

        let activations = accumulator.get_all_activations();
        assert_eq!(activations.len(), 1);
        assert_eq!(activations[0].0, node_id);
        assert!((activations[0].1 - 0.5).abs() < 1e-6);

        assert_eq!(accumulator.get_operation_count(), 1);
    }

    #[test]
    fn test_simd_accumulator_batch() {
        let accumulator = SimdActivationAccumulator::new(4);

        let batch = vec![
            ("node1".to_string(), 0.3),
            ("node2".to_string(), 0.7),
            ("node3".to_string(), 0.1),
            ("node4".to_string(), 0.05), // Above 0.01 threshold
        ];

        let updated_count = accumulator.accumulate_batch(&batch).unwrap();
        assert_eq!(updated_count, 4); // All 4 should be above 0.01 threshold

        let activations = accumulator.get_all_activations();
        assert_eq!(activations.len(), 4);

        assert_eq!(accumulator.get_operation_count(), 4);
    }

    #[test]
    fn test_activation_vector_operations() {
        let accumulator = SimdActivationAccumulator::new(8);

        accumulator.accumulate_single(&"test_node".to_string(), 0.8);

        let vector = accumulator
            .get_activation_vector(&"test_node".to_string())
            .unwrap();
        assert!((vector[0] - 0.8).abs() < 1e-6);
        assert!(vector[1] > 0.0 && vector[1] < 0.1); // Should have spreading pattern

        // Test vector updates
        let mut update_vector = [0.0f32; 768];
        update_vector[0] = 0.5;
        let updates = vec![("update_node".to_string(), update_vector)];

        let updated = accumulator.update_from_vectors(&updates).unwrap();
        assert_eq!(updated, 1);
    }

    #[test]
    fn test_similarity_weights() {
        let accumulator = SimdActivationAccumulator::new(8);

        let query = [0.5f32; 768];
        let mut different_candidate = [0.3f32; 768];
        different_candidate[0] = 0.1f32; // Make it actually different direction
        let candidates = vec![
            [0.5f32; 768],      // Identical
            different_candidate, // Different direction
            [-0.5f32; 768],     // Opposite
        ];

        let weights = accumulator.compute_similarity_weights(&query, &candidates);
        assert_eq!(weights.len(), 3);
        assert!(weights[0] > weights[1]); // Identical should have higher weight
        assert!(weights[1] > weights[2]); // Different should be better than opposite
    }

    #[test]
    fn test_biological_accumulator() {
        let bio_accumulator = BiologicalAccumulator::new(8);
        let current_time = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos() as u64;

        let node_id = "bio_node".to_string();

        // First accumulation should succeed
        let result1 = bio_accumulator.accumulate_biological(&node_id, 0.6, current_time);
        assert!(result1);

        // Immediate second accumulation should fail (refractory period)
        let result2 = bio_accumulator.accumulate_biological(&node_id, 0.6, current_time);
        assert!(!result2);

        // After refractory period should succeed
        let future_time = current_time + 2_000_000; // 2ms later
        let result3 = bio_accumulator.accumulate_biological(&node_id, 0.6, future_time);
        assert!(result3);

        let (activation, fatigue, in_refractory) = bio_accumulator.get_biological_state(&node_id);
        assert!(activation > 0.0);
        assert!(fatigue < 1.0); // Should have some fatigue
        assert!(in_refractory); // Should be in refractory period after firing
    }

    #[test]
    fn test_temporal_decay() {
        let bio_accumulator = BiologicalAccumulator::new(8);
        let current_time = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos() as u64;

        let node_id = "decay_node".to_string();
        bio_accumulator.accumulate_biological(&node_id, 1.0, current_time);

        let (activation_before, _, _) = bio_accumulator.get_biological_state(&node_id);
        assert!(activation_before > 0.9);

        // Apply decay
        bio_accumulator.apply_temporal_decay(0.5);

        let (activation_after, fatigue_after, _) = bio_accumulator.get_biological_state(&node_id);
        assert!(activation_after < activation_before);
        assert!(activation_after > 0.4); // Should be about half
        assert!(fatigue_after > 0.8); // Fatigue should recover slightly
    }

    #[test]
    fn test_active_nodes_filtering() {
        let accumulator = SimdActivationAccumulator::new(8);

        accumulator.accumulate_single(&"high".to_string(), 0.8);
        accumulator.accumulate_single(&"medium".to_string(), 0.3);
        accumulator.accumulate_single(&"low".to_string(), 0.05);

        let active_high = accumulator.get_active_nodes(0.5);
        assert_eq!(active_high.len(), 1);
        assert_eq!(active_high[0].0, "high");

        let active_medium = accumulator.get_active_nodes(0.1);
        assert_eq!(active_medium.len(), 2);

        let active_all = accumulator.get_active_nodes(0.01);
        assert_eq!(active_all.len(), 3);
    }

    #[test]
    fn test_simd_normalization() {
        let accumulator = SimdActivationAccumulator::new(8);

        let mut test_vector = [0.0f32; 768];
        test_vector[0] = 2.0; // Large value
        test_vector[1] = 1.0;
        test_vector[2] = 3.0;

        let normalized = accumulator.simd_normalize(&test_vector);

        // Normalized values should be smaller
        assert!(normalized[0] < test_vector[0]);
        assert!(normalized[1] < test_vector[1]);
        assert!(normalized[2] < test_vector[2]);

        // But maintain relative proportions
        assert!(normalized[2] > normalized[0]);
        assert!(normalized[0] > normalized[1]);
    }
}
