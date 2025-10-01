//! Simple `HashMap`-based memory backend for single-threaded use

use crate::memory::Memory;
use crate::memory_graph::traits::{GraphBackend, MemoryBackend, MemoryError};
use parking_lot::RwLock;
use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::Arc;
use uuid::Uuid;

/// Simple `HashMap`-based memory backend
///
/// This backend is suitable for single-threaded or low-contention scenarios.
/// Uses [`RwLock`](parking_lot::RwLock) for thread safety with minimal overhead.
pub struct HashMapBackend {
    memories: Arc<RwLock<HashMap<Uuid, Arc<Memory>>>>,
    edges: Arc<RwLock<HashMap<Uuid, Vec<(Uuid, f32)>>>>,
}

impl HashMapBackend {
    /// Create a new `HashMap` backend
    #[must_use]
    pub fn new() -> Self {
        Self {
            memories: Arc::new(RwLock::new(HashMap::new())),
            edges: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Create with pre-allocated capacity
    #[must_use]
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            memories: Arc::new(RwLock::new(HashMap::with_capacity(capacity))),
            edges: Arc::new(RwLock::new(HashMap::with_capacity(capacity))),
        }
    }
}

impl Default for HashMapBackend {
    fn default() -> Self {
        Self::new()
    }
}

impl MemoryBackend for HashMapBackend {
    fn store(&self, id: Uuid, memory: Memory) -> Result<(), MemoryError> {
        self.memories.write().insert(id, Arc::new(memory));
        Ok(())
    }

    fn retrieve(&self, id: &Uuid) -> Result<Option<Arc<Memory>>, MemoryError> {
        let memories = self.memories.read();
        Ok(memories.get(id).cloned())
    }

    fn remove(&self, id: &Uuid) -> Result<Option<Arc<Memory>>, MemoryError> {
        let memory = self.memories.write().remove(id);

        if memory.is_some() {
            let mut edges = self.edges.write();
            edges.remove(id);
            for neighbors in edges.values_mut() {
                neighbors.retain(|(target, _)| target != id);
            }
        }

        Ok(memory)
    }

    fn search(&self, embedding: &[f32], k: usize) -> Result<Vec<(Uuid, f32)>, MemoryError> {
        if embedding.len() != 768 {
            return Err(MemoryError::InvalidEmbeddingDimension(embedding.len()));
        }

        let mut scores: Vec<(Uuid, f32)> = self
            .memories
            .read()
            .iter()
            .map(|(id, memory)| {
                let similarity = cosine_similarity(embedding, &memory.embedding);
                (*id, similarity)
            })
            .collect();

        // Sort by similarity (highest first)
        scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scores.truncate(k);

        Ok(scores)
    }

    fn update_activation(&self, id: &Uuid, activation: f32) -> Result<(), MemoryError> {
        self.memories.read().get(id).map_or_else(
            || Err(MemoryError::NotFound(*id)),
            |memory| {
                memory.set_activation(activation.clamp(0.0, 1.0));
                Ok(())
            },
        )
    }

    fn count(&self) -> usize {
        self.memories.read().len()
    }

    fn clear(&self) -> Result<(), MemoryError> {
        self.memories.write().clear();
        self.edges.write().clear();
        Ok(())
    }

    fn all_ids(&self) -> Vec<Uuid> {
        self.memories.read().keys().copied().collect()
    }
}

impl GraphBackend for HashMapBackend {
    fn add_edge(&self, from: Uuid, to: Uuid, weight: f32) -> Result<(), MemoryError> {
        self.edges
            .write()
            .entry(from)
            .or_default()
            .push((to, weight));
        Ok(())
    }

    fn remove_edge(&self, from: &Uuid, to: &Uuid) -> Result<bool, MemoryError> {
        let removed;
        {
            let mut edges = self.edges.write();
            if let Some(neighbors) = edges.get_mut(from) {
                let initial_len = neighbors.len();
                neighbors.retain(|(target, _)| target != to);
                removed = neighbors.len() < initial_len;
            } else {
                removed = false;
            }
        }
        Ok(removed)
    }

    fn get_neighbors(&self, id: &Uuid) -> Result<Vec<(Uuid, f32)>, MemoryError> {
        Ok(self.edges.read().get(id).cloned().unwrap_or_default())
    }

    fn traverse_bfs(&self, start: &Uuid, max_depth: usize) -> Result<Vec<Uuid>, MemoryError> {
        let edges = self.edges.read();
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

            if let Some(neighbors) = edges.get(&node) {
                for (neighbor, _) in neighbors {
                    if visited.insert(*neighbor) {
                        queue.push_back((*neighbor, depth + 1));
                    }
                }
            }
        }

        Ok(result)
    }

    fn get_edge_weight(&self, from: &Uuid, to: &Uuid) -> Result<Option<f32>, MemoryError> {
        let weight = {
            let edges = self.edges.read();
            edges.get(from).and_then(|neighbors| {
                neighbors
                    .iter()
                    .find(|(target, _)| target == to)
                    .map(|(_, weight)| *weight)
            })
        };
        Ok(weight)
    }

    fn all_edges(&self) -> Result<Vec<(Uuid, Uuid, f32)>, MemoryError> {
        let mut result = Vec::new();
        {
            let edges = self.edges.read();
            for (from, neighbors) in edges.iter() {
                for (to, weight) in neighbors {
                    result.push((*from, *to, *weight));
                }
            }
        }

        Ok(result)
    }
}

/// Compute cosine similarity between two vectors
fn cosine_similarity(a: &[f32], b: &[f32; 768]) -> f32 {
    let dot_product: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();

    if norm_a == 0.0 || norm_b == 0.0 {
        0.0
    } else {
        dot_product / (norm_a * norm_b)
    }
}
