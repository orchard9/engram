//! Lock-free HNSW node implementation with cache-optimal layout

use crate::{Confidence, Memory};
use atomic_float::AtomicF32;
use smallvec::SmallVec;
use std::sync::Arc;
use std::sync::atomic::{AtomicPtr, AtomicU8, AtomicU32, AtomicU64, Ordering};

/// L1-cache optimized node layout - 64-byte alignment for false sharing prevention
#[repr(C)]
#[repr(align(64))]
pub struct HnswNode {
    /// Dense node ID for vectorization
    pub node_id: u32,
    /// Lock-free layer updates
    pub layer_count: AtomicU8,
    /// ABA protection generation
    pub generation: AtomicU32,
    /// Lock-free activation updates
    pub activation: AtomicF32,
    /// Probabilistic weight
    pub confidence: Confidence,
    /// Epoch-based timestamp
    pub last_access_epoch: AtomicU64,
    /// Pointer to embedding (separate allocation for cache efficiency)
    pub embedding_ptr: AtomicPtr<[f32; 768]>,
    /// Lock-free connection updates
    pub connections_ptr: AtomicPtr<ConnectionBlock>,
    /// Reference to original memory
    pub memory: Arc<Memory>,
    /// Padding to fill cache line
    _padding: [u8; 7],
}

impl HnswNode {
    /// Create a new HNSW node from a memory
    pub fn from_memory(
        node_id: u32,
        memory: Arc<Memory>,
        layer_count: u8,
    ) -> Result<Self, super::HnswError> {
        // Validate embedding dimension
        if memory.embedding.len() != 768 {
            return Err(super::HnswError::InvalidDimension(memory.embedding.len()));
        }

        // Allocate embedding in separate memory region
        let embedding_box = Box::new(memory.embedding);
        let embedding_ptr = Box::into_raw(embedding_box);

        Ok(Self {
            node_id,
            layer_count: AtomicU8::new(layer_count),
            generation: AtomicU32::new(0),
            activation: AtomicF32::new(memory.activation()),
            confidence: memory.confidence,
            last_access_epoch: AtomicU64::new(0),
            embedding_ptr: AtomicPtr::new(embedding_ptr),
            connections_ptr: AtomicPtr::new(std::ptr::null_mut()),
            memory,
            _padding: [0u8; 7],
        })
    }

    /// Get the embedding for this node
    pub fn get_embedding(&self) -> &[f32; 768] {
        unsafe {
            let ptr = self.embedding_ptr.load(Ordering::Acquire);
            if ptr.is_null() {
                &self.memory.embedding
            } else {
                &*ptr
            }
        }
    }

    /// Update activation level atomically
    pub fn update_activation(&self, new_activation: f32) {
        self.activation.store(new_activation, Ordering::Relaxed);
        self.memory.set_activation(new_activation);
    }

    /// Get connections for a specific layer
    pub fn get_connections(&self, layer: usize) -> Option<&[HnswEdge]> {
        let connections_ptr = self.connections_ptr.load(Ordering::Acquire);
        if connections_ptr.is_null() {
            return None;
        }

        unsafe {
            let connections = &*connections_ptr;
            if layer < connections.layer_connections.len() {
                Some(&connections.layer_connections[layer])
            } else {
                None
            }
        }
    }

    /// Add a connection atomically
    pub fn add_connection(&self, layer: usize, edge: HnswEdge) -> Result<(), super::HnswError> {
        let connections_ptr = self.connections_ptr.load(Ordering::Acquire);

        if connections_ptr.is_null() {
            // Initialize connections if not present
            let mut connections = Box::new(ConnectionBlock::new());
            connections.add_edge(layer, edge)?;
            let new_ptr = Box::into_raw(connections);

            // Try to install the new connections
            match self.connections_ptr.compare_exchange(
                std::ptr::null_mut(),
                new_ptr,
                Ordering::Release,
                Ordering::Acquire,
            ) {
                Ok(_) => Ok(()),
                Err(_) => {
                    // Another thread initialized it, clean up and retry
                    unsafe {
                        Box::from_raw(new_ptr);
                    }
                    self.add_connection(layer, edge)
                }
            }
        } else {
            // Add to existing connections
            unsafe {
                let connections = &mut *connections_ptr;
                connections.add_edge(layer, edge)
            }
        }
    }
}

impl Drop for HnswNode {
    fn drop(&mut self) {
        // Clean up allocated embedding
        let embedding_ptr = self.embedding_ptr.load(Ordering::Acquire);
        if !embedding_ptr.is_null() {
            unsafe {
                Box::from_raw(embedding_ptr);
            }
        }

        // Clean up connections
        let connections_ptr = self.connections_ptr.load(Ordering::Acquire);
        if !connections_ptr.is_null() {
            unsafe {
                Box::from_raw(connections_ptr);
            }
        }
    }
}

/// Connection block with lock-free update semantics
#[repr(C)]
pub struct ConnectionBlock {
    /// Connections per layer (stack allocation for small M)
    pub layer_connections: Vec<SmallVec<[HnswEdge; 16]>>,
    /// Reference counting for safe reclamation
    pub ref_count: AtomicU32,
    /// Matches parent node generation
    pub generation: u32,
}

impl ConnectionBlock {
    /// Create a new connection block
    pub fn new() -> Self {
        Self {
            layer_connections: (0..16).map(|_| SmallVec::new()).collect(),
            ref_count: AtomicU32::new(1),
            generation: 0,
        }
    }

    /// Add an edge to a specific layer
    pub fn add_edge(&mut self, layer: usize, edge: HnswEdge) -> Result<(), super::HnswError> {
        if layer >= self.layer_connections.len() {
            return Err(super::HnswError::CorruptedGraph(format!(
                "Layer {} exceeds maximum",
                layer
            )));
        }

        self.layer_connections[layer].push(edge);
        Ok(())
    }

    /// Remove an edge from a specific layer
    pub fn remove_edge(&mut self, layer: usize, target_id: u32) -> bool {
        if layer >= self.layer_connections.len() {
            return false;
        }

        if let Some(pos) = self.layer_connections[layer]
            .iter()
            .position(|e| e.target_id == target_id)
        {
            self.layer_connections[layer].swap_remove(pos);
            true
        } else {
            false
        }
    }
}

/// SIMD-friendly edge representation - 16-byte aligned for vectorized operations
#[derive(Clone, Copy, Debug)]
#[repr(C, align(16))]
pub struct HnswEdge {
    /// Dense node ID
    pub target_id: u32,
    /// Precomputed distance for pruning
    pub cached_distance: f32,
    /// Atomic confidence value
    pub confidence_weight: f32,
    /// Packed flags and type
    pub edge_metadata: EdgeMetadata,
}

impl HnswEdge {
    /// Create a new edge
    pub fn new(target_id: u32, distance: f32, confidence: Confidence) -> Self {
        Self {
            target_id,
            cached_distance: distance,
            confidence_weight: confidence.raw(),
            edge_metadata: EdgeMetadata::default(),
        }
    }

    /// Get the confidence-weighted distance
    pub fn weighted_distance(&self) -> f32 {
        self.cached_distance * (1.0 - self.confidence_weight)
    }
}

#[derive(Clone, Copy, Debug, Default)]
#[repr(C)]
pub struct EdgeMetadata {
    /// Edge type discriminant
    pub edge_type: u8,
    /// Recent memory boost factor (0-255)
    pub temporal_boost: u8,
    /// How often this edge is traversed
    pub stability_score: u8,
    /// Reserved for future use
    _padding: u8,
}

// Ensure our structures have the expected sizes for cache optimization
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_node_cache_alignment() {
        // HnswNode should be exactly 64 bytes (one cache line)
        assert_eq!(std::mem::size_of::<HnswNode>(), 64);
        assert_eq!(std::mem::align_of::<HnswNode>(), 64);
    }

    #[test]
    fn test_edge_simd_alignment() {
        // HnswEdge should be 16-byte aligned for SIMD operations
        assert_eq!(std::mem::size_of::<HnswEdge>(), 16);
        assert_eq!(std::mem::align_of::<HnswEdge>(), 16);
    }

    #[test]
    fn test_node_creation() {
        use crate::MemoryBuilder;
        use chrono::Utc;

        // Create memory directly instead of through episode conversion
        let memory = Arc::new(crate::Memory::new(
            "test".to_string(),
            [0.5f32; 768],
            Confidence::HIGH,
        ));

        let node = HnswNode::from_memory(0, memory.clone(), 3).unwrap();
        assert_eq!(node.node_id, 0);
        assert_eq!(node.layer_count.load(Ordering::Relaxed), 3);
        assert_eq!(node.confidence, memory.confidence);
    }
}
