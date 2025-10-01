//! Lock-free `DashMap`-based memory backend for high-concurrency scenarios

use crate::Memory;
use crate::memory_graph::traits::{GraphBackend, MemoryBackend, MemoryError};
use atomic_float::AtomicF32;
use dashmap::DashMap;
use rayon::prelude::*;
use std::collections::{HashSet, VecDeque};
use std::sync::Arc;
use std::sync::atomic::Ordering;
use uuid::Uuid;

/// Lock-free `DashMap`-based memory backend
///
/// This backend is optimized for high-concurrency scenarios with minimal contention.
/// Uses [`DashMap`](dashmap::DashMap) for lock-free concurrent access and atomic operations for activation updates.
pub struct DashMapBackend {
    memories: Arc<DashMap<Uuid, Arc<Memory>>>,
    edges: Arc<DashMap<Uuid, Vec<(Uuid, f32)>>>,
    activation_cache: Arc<DashMap<Uuid, AtomicF32>>,
}

impl DashMapBackend {
    /// Create a new `DashMap` backend
    #[must_use]
    pub fn new() -> Self {
        Self {
            memories: Arc::new(DashMap::new()),
            edges: Arc::new(DashMap::new()),
            activation_cache: Arc::new(DashMap::new()),
        }
    }

    /// Create with pre-allocated capacity
    #[must_use]
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            memories: Arc::new(DashMap::with_capacity(capacity)),
            edges: Arc::new(DashMap::with_capacity(capacity)),
            activation_cache: Arc::new(DashMap::with_capacity(capacity)),
        }
    }
}

impl Default for DashMapBackend {
    fn default() -> Self {
        Self::new()
    }
}

impl MemoryBackend for DashMapBackend {
    fn store(&self, id: Uuid, memory: Memory) -> Result<(), MemoryError> {
        // Store initial activation in cache
        self.activation_cache
            .insert(id, AtomicF32::new(memory.activation()));
        self.memories.insert(id, Arc::new(memory));
        Ok(())
    }

    fn retrieve(&self, id: &Uuid) -> Result<Option<Arc<Memory>>, MemoryError> {
        Ok(self.memories.get(id).map(|entry| entry.value().clone()))
    }

    fn remove(&self, id: &Uuid) -> Result<Option<Arc<Memory>>, MemoryError> {
        let memory = self.memories.remove(id).map(|(_, v)| v);

        // Clean up associated data
        if memory.is_some() {
            self.activation_cache.remove(id);
            self.edges.remove(id);

            // Remove as target from other nodes (sequential operation - DashMap handles concurrency)
            for mut entry in self.edges.iter_mut() {
                entry.value_mut().retain(|(target, _)| target != id);
            }
        }

        Ok(memory)
    }

    fn search(&self, embedding: &[f32], k: usize) -> Result<Vec<(Uuid, f32)>, MemoryError> {
        if embedding.len() != 768 {
            return Err(MemoryError::InvalidEmbeddingDimension(embedding.len()));
        }

        // Similarity computation (DashMap is already thread-safe)
        let mut scores: Vec<(Uuid, f32)> = self
            .memories
            .iter()
            .map(|entry| {
                let (id, memory) = entry.pair();
                let similarity = cosine_similarity(embedding, &memory.embedding);
                (*id, similarity)
            })
            .collect();

        // Sort by similarity (highest first)
        scores.par_sort_unstable_by(|a, b| {
            b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
        });
        scores.truncate(k);

        Ok(scores)
    }

    fn update_activation(&self, id: &Uuid, activation: f32) -> Result<(), MemoryError> {
        match self.activation_cache.get(id) {
            Some(cached) => {
                cached.store(activation.clamp(0.0, 1.0), Ordering::Relaxed);
                Ok(())
            }
            None if self.memories.contains_key(id) => {
                self.activation_cache
                    .insert(*id, AtomicF32::new(activation.clamp(0.0, 1.0)));
                Ok(())
            }
            None => Err(MemoryError::NotFound(*id)),
        }
    }

    fn count(&self) -> usize {
        self.memories.len()
    }

    fn clear(&self) -> Result<(), MemoryError> {
        self.memories.clear();
        self.edges.clear();
        self.activation_cache.clear();
        Ok(())
    }

    fn all_ids(&self) -> Vec<Uuid> {
        self.memories.iter().map(|entry| *entry.key()).collect()
    }
}

impl GraphBackend for DashMapBackend {
    fn add_edge(&self, from: Uuid, to: Uuid, weight: f32) -> Result<(), MemoryError> {
        self.edges.entry(from).or_default().push((to, weight));
        Ok(())
    }

    fn remove_edge(&self, from: &Uuid, to: &Uuid) -> Result<bool, MemoryError> {
        let removed = self.edges.get_mut(from).is_some_and(|mut neighbors| {
            let initial_len = neighbors.len();
            neighbors.retain(|(target, _)| target != to);
            neighbors.len() < initial_len
        });
        Ok(removed)
    }

    fn get_neighbors(&self, id: &Uuid) -> Result<Vec<(Uuid, f32)>, MemoryError> {
        Ok(self
            .edges
            .get(id)
            .map(|entry| entry.value().clone())
            .unwrap_or_default())
    }

    fn traverse_bfs(&self, start: &Uuid, max_depth: usize) -> Result<Vec<Uuid>, MemoryError> {
        let mut visited = HashSet::new();
        let mut queue = VecDeque::new();
        let mut result = Vec::new();

        queue.push_back((*start, 0));
        visited.insert(*start);

        while let Some((node, depth)) = queue.pop_front() {
            if depth > max_depth {
                break;
            }

            result.push(node);

            if let Some(neighbors) = self.edges.get(&node) {
                for (neighbor, _) in neighbors.value() {
                    if visited.insert(*neighbor) {
                        queue.push_back((*neighbor, depth + 1));
                    }
                }
            }
        }

        Ok(result)
    }

    fn spread_activation(&self, source: &Uuid, decay: f32) -> Result<(), MemoryError> {
        // Optimized parallel activation spreading
        if let Some(neighbors) = self.edges.get(source) {
            let source_activation = self
                .activation_cache
                .get(source)
                .map_or(0.5, |a| a.load(Ordering::Relaxed));

            for (neighbor_id, weight) in neighbors.value() {
                if let Some(activation) = self.activation_cache.get(neighbor_id) {
                    loop {
                        let current = activation.load(Ordering::Relaxed);
                        let contribution = source_activation * weight * decay;
                        let new_activation = (current + contribution).min(1.0);

                        if activation
                            .compare_exchange_weak(
                                current,
                                new_activation,
                                Ordering::Relaxed,
                                Ordering::Relaxed,
                            )
                            .is_ok()
                        {
                            break;
                        }
                    }
                } else if self.memories.contains_key(neighbor_id) {
                    let contribution = source_activation * weight * decay;
                    self.activation_cache
                        .insert(*neighbor_id, AtomicF32::new(contribution.min(1.0)));
                }
            }
        }

        Ok(())
    }

    fn get_edge_weight(&self, from: &Uuid, to: &Uuid) -> Result<Option<f32>, MemoryError> {
        Ok(self.edges.get(from).and_then(|neighbors| {
            neighbors
                .value()
                .iter()
                .find(|(target, _)| target == to)
                .map(|(_, weight)| *weight)
        }))
    }

    fn all_edges(&self) -> Result<Vec<(Uuid, Uuid, f32)>, MemoryError> {
        let mut result = Vec::new();

        for entry in self.edges.iter() {
            let from = *entry.key();
            for (to, weight) in entry.value() {
                result.push((from, *to, *weight));
            }
        }

        Ok(result)
    }
}

/// Compute cosine similarity between two vectors
fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let dot_product: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();

    if norm_a == 0.0 || norm_b == 0.0 {
        0.0
    } else {
        dot_product / (norm_a * norm_b)
    }
}
