//! Infallible memory backend with graceful degradation
//! 
//! This backend never returns errors, instead degrading gracefully under pressure.
//! It returns reduced quality indicators when operations can't be fully completed.

use crate::memory_graph::traits::{GraphBackend, MemoryBackend, MemoryError};
use crate::memory::Memory;
use dashmap::DashMap;
use parking_lot::RwLock;
use std::collections::{BTreeMap, HashSet, VecDeque};
use std::sync::atomic::{AtomicUsize, Ordering};
use atomic_float::AtomicF32;
use std::sync::Arc;
use uuid::Uuid;

/// Quality indicator for infallible operations
#[derive(Debug, Clone, Copy)]
pub struct OperationQuality {
    /// Activation level (0.0 to 1.0) indicating operation quality
    pub activation: f32,
    /// Whether the operation was degraded
    pub degraded: bool,
    /// System pressure at time of operation
    pub pressure: f32,
}

impl OperationQuality {
    fn new(activation: f32, pressure: f32) -> Self {
        Self {
            activation,
            degraded: activation < 0.8,
            pressure,
        }
    }
}

/// Infallible memory backend that degrades gracefully
/// 
/// This backend follows cognitive design principles where operations never fail
/// but degrade in quality under system pressure, similar to human memory under stress.
pub struct InfallibleBackend {
    /// High-priority memories (hot cache)
    hot_memories: Arc<DashMap<Uuid, Arc<Memory>>>,
    
    /// Lower-priority memories (can be evicted)
    cold_memories: Arc<RwLock<BTreeMap<(OrderedFloat, Uuid), Arc<Memory>>>>,
    
    /// Graph edges
    edges: Arc<DashMap<Uuid, Vec<(Uuid, f32)>>>,
    
    /// Activation cache for fast updates
    activation_cache: Arc<DashMap<Uuid, AtomicF32>>,
    
    /// Current memory count
    memory_count: Arc<AtomicUsize>,
    
    /// Maximum capacity before eviction
    max_capacity: usize,
    
    /// System pressure (0.0 = no pressure, 1.0 = maximum pressure)
    system_pressure: Arc<AtomicF32>,
    
    /// Last operation quality for monitoring
    last_quality: Arc<RwLock<OperationQuality>>,
}

/// Wrapper for f32 that implements Ord for BTreeMap
#[derive(Clone, Copy, Debug)]
struct OrderedFloat(f32);

impl PartialEq for OrderedFloat {
    fn eq(&self, other: &Self) -> bool {
        self.0 == other.0
    }
}

impl Eq for OrderedFloat {}

impl PartialOrd for OrderedFloat {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        self.0.partial_cmp(&other.0)
    }
}

impl Ord for OrderedFloat {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.partial_cmp(other).unwrap_or(std::cmp::Ordering::Equal)
    }
}

impl InfallibleBackend {
    /// Create a new infallible backend
    #[must_use]
    pub fn new(max_capacity: usize) -> Self {
        Self {
            hot_memories: Arc::new(DashMap::new()),
            cold_memories: Arc::new(RwLock::new(BTreeMap::new())),
            edges: Arc::new(DashMap::new()),
            activation_cache: Arc::new(DashMap::new()),
            memory_count: Arc::new(AtomicUsize::new(0)),
            max_capacity,
            system_pressure: Arc::new(AtomicF32::new(0.0)),
            last_quality: Arc::new(RwLock::new(OperationQuality::new(1.0, 0.0))),
        }
    }
    
    /// Get current system pressure
    pub fn pressure(&self) -> f32 {
        self.system_pressure.load(Ordering::Relaxed)
    }
    
    /// Update system pressure based on current load
    fn update_pressure(&self) {
        let count = self.memory_count.load(Ordering::Relaxed);
        let pressure = (count as f32 / self.max_capacity as f32).min(1.0);
        self.system_pressure.store(pressure, Ordering::Relaxed);
    }
    
    /// Evict lowest-priority memories if over capacity
    fn maybe_evict(&self) {
        let count = self.memory_count.load(Ordering::Relaxed);
        if count > self.max_capacity {
            let to_evict = count - (self.max_capacity * 9 / 10); // Evict to 90% capacity
            
            let mut cold = self.cold_memories.write();
            for _ in 0..to_evict {
                if let Some(((_, id), _)) = cold.pop_first() {
                    self.memory_count.fetch_sub(1, Ordering::Relaxed);
                    self.activation_cache.remove(&id);
                    // Note: Not removing from hot_memories as it might still be there
                }
            }
        }
    }
    
    /// Record operation quality
    fn record_quality(&self, activation: f32) {
        let pressure = self.pressure();
        *self.last_quality.write() = OperationQuality::new(activation, pressure);
    }
}

impl Default for InfallibleBackend {
    fn default() -> Self {
        Self::new(10000)
    }
}

impl MemoryBackend for InfallibleBackend {
    fn store(&self, id: Uuid, memory: Memory) -> Result<(), MemoryError> {
        // Always succeeds, but quality varies
        let pressure = self.pressure();
        
        // Store in hot cache if low pressure, cold otherwise
        if pressure < 0.5 {
            self.hot_memories.insert(id, Arc::new(memory.clone()));
            self.record_quality(1.0);
        } else {
            // Store in cold storage with eviction priority
            let priority = OrderedFloat(memory.activation() * (1.0 - pressure));
            self.cold_memories.write().insert((priority, id), Arc::new(memory.clone()));
            self.record_quality(0.7);
        }
        
        // Update activation cache
        self.activation_cache.insert(id, AtomicF32::new(memory.activation()));
        
        // Update count and pressure
        self.memory_count.fetch_add(1, Ordering::Relaxed);
        self.update_pressure();
        
        // Evict if necessary (non-blocking)
        self.maybe_evict();
        
        Ok(()) // Always succeeds
    }
    
    fn retrieve(&self, id: &Uuid) -> Result<Option<Arc<Memory>>, MemoryError> {
        // Check hot cache first
        if let Some(memory) = self.hot_memories.get(id) {
            self.record_quality(1.0);
            return Ok(Some(memory.clone()));
        }
        
        // Check cold storage
        let cold = self.cold_memories.read();
        for ((_, mem_id), memory) in cold.iter() {
            if mem_id == id {
                self.record_quality(0.6); // Degraded quality for cold retrieval
                return Ok(Some(memory.clone()));
            }
        }
        
        self.record_quality(0.0);
        Ok(None) // Not found, but no error
    }
    
    fn remove(&self, id: &Uuid) -> Result<Option<Arc<Memory>>, MemoryError> {
        // Try to remove from hot cache
        let memory = self.hot_memories.remove(id).map(|(_, v)| v);
        
        // If not in hot, try cold
        let memory = memory.or_else(|| {
            let mut cold = self.cold_memories.write();
            cold.iter()
                .find(|((_, mem_id), _)| mem_id == id)
                .map(|(key, _)| key.clone())
                .and_then(|key| cold.remove(&key))
        });
        
        if memory.is_some() {
            self.memory_count.fetch_sub(1, Ordering::Relaxed);
            self.activation_cache.remove(id);
            self.edges.remove(id);
            self.update_pressure();
            self.record_quality(0.9);
        } else {
            self.record_quality(0.0);
        }
        
        Ok(memory) // Always succeeds
    }
    
    fn search(&self, embedding: &[f32], k: usize) -> Result<Vec<(Uuid, f32)>, MemoryError> {
        // Best-effort search - returns what it can find
        let mut results = Vec::new();
        
        // Search hot memories first (higher quality)
        for entry in self.hot_memories.iter() {
            let (id, memory) = entry.pair();
            if let Some(mem_emb) = &memory.embedding {
                if mem_emb.len() == embedding.len() {
                    let similarity = cosine_similarity(embedding, mem_emb);
                    results.push((*id, similarity));
                }
            }
        }
        
        // If not enough results, search cold storage
        if results.len() < k {
            let cold = self.cold_memories.read();
            for ((_, id), memory) in cold.iter().take(k * 2) { // Sample cold storage
                if let Some(mem_emb) = &memory.embedding {
                    if mem_emb.len() == embedding.len() {
                        let similarity = cosine_similarity(embedding, mem_emb);
                        results.push((*id, similarity));
                    }
                }
            }
        }
        
        // Sort and truncate
        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        results.truncate(k);
        
        let quality = if results.len() >= k { 1.0 } else { results.len() as f32 / k as f32 };
        self.record_quality(quality);
        
        Ok(results) // Always returns something, even if partial
    }
    
    fn update_activation(&self, id: &Uuid, activation: f32) -> Result<(), MemoryError> {
        self.activation_cache
            .entry(*id)
            .or_insert_with(|| AtomicF32::new(0.0))
            .store(activation.clamp(0.0, 1.0), Ordering::Relaxed);
        
        self.record_quality(0.9);
        Ok(()) // Always succeeds
    }
    
    fn count(&self) -> usize {
        self.memory_count.load(Ordering::Relaxed)
    }
    
    fn clear(&self) -> Result<(), MemoryError> {
        self.hot_memories.clear();
        self.cold_memories.write().clear();
        self.edges.clear();
        self.activation_cache.clear();
        self.memory_count.store(0, Ordering::Relaxed);
        self.system_pressure.store(0.0, Ordering::Relaxed);
        self.record_quality(1.0);
        Ok(())
    }
    
    fn all_ids(&self) -> Vec<Uuid> {
        let mut ids: HashSet<Uuid> = self.hot_memories.iter()
            .map(|entry| *entry.key())
            .collect();
        
        let cold = self.cold_memories.read();
        for ((_, id), _) in cold.iter() {
            ids.insert(*id);
        }
        
        ids.into_iter().collect()
    }
}

impl GraphBackend for InfallibleBackend {
    fn add_edge(&self, from: Uuid, to: Uuid, weight: f32) -> Result<(), MemoryError> {
        self.edges.entry(from)
            .or_insert_with(Vec::new)
            .push((to, weight));
        self.record_quality(1.0);
        Ok(())
    }
    
    fn remove_edge(&self, from: &Uuid, to: &Uuid) -> Result<bool, MemoryError> {
        let removed = if let Some(mut neighbors) = self.edges.get_mut(from) {
            let initial_len = neighbors.len();
            neighbors.retain(|(target, _)| target != to);
            neighbors.len() < initial_len
        } else {
            false
        };
        
        self.record_quality(if removed { 1.0 } else { 0.0 });
        Ok(removed)
    }
    
    fn get_neighbors(&self, id: &Uuid) -> Result<Vec<(Uuid, f32)>, MemoryError> {
        let neighbors = self.edges.get(id)
            .map(|entry| entry.value().clone())
            .unwrap_or_default();
        
        self.record_quality(if neighbors.is_empty() { 0.5 } else { 1.0 });
        Ok(neighbors)
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
                    if !visited.contains(neighbor) {
                        visited.insert(*neighbor);
                        queue.push_back((*neighbor, depth + 1));
                    }
                }
            }
        }
        
        self.record_quality(1.0);
        Ok(result)
    }
    
    fn get_edge_weight(&self, from: &Uuid, to: &Uuid) -> Result<Option<f32>, MemoryError> {
        let weight = self.edges.get(from)
            .and_then(|neighbors| {
                neighbors.value().iter()
                    .find(|(target, _)| target == to)
                    .map(|(_, weight)| *weight)
            });
        
        self.record_quality(if weight.is_some() { 1.0 } else { 0.0 });
        Ok(weight)
    }
    
    fn all_edges(&self) -> Result<Vec<(Uuid, Uuid, f32)>, MemoryError> {
        let edges: Vec<_> = self.edges.iter()
            .flat_map(|entry| {
                let from = *entry.key();
                entry.value().iter().map(move |(to, weight)| {
                    (from, *to, *weight)
                })
            })
            .collect();
        
        self.record_quality(1.0);
        Ok(edges)
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