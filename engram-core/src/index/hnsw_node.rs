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
    /// ABA protection generation (for lock-free CAS operations)
    pub generation: AtomicU32,
    /// Lock-free activation updates
    pub activation: AtomicF32,
    /// Probabilistic weight
    pub confidence: Confidence,
    /// Sequence number for snapshot isolation (when observation was committed)
    pub sequence_number: AtomicU64,
    /// Epoch-based timestamp
    pub last_access_epoch: AtomicU64,
    /// Pointer to embedding (separate allocation for cache efficiency)
    pub embedding_ptr: AtomicPtr<[f32; 768]>,
    /// Lock-free connection updates
    pub connections_ptr: AtomicPtr<ConnectionBlock>,
    /// Reference to original memory
    pub memory: Arc<Memory>,
}

impl HnswNode {
    /// Create a new HNSW node from a memory
    ///
    /// # Errors
    ///
    /// Returns `HnswError::InvalidDimension` when the memory embedding does not
    /// match the expected dimensionality for the index.
    pub fn from_memory(
        node_id: u32,
        memory: Arc<Memory>,
        layer_count: u8,
    ) -> Result<Self, super::HnswError> {
        Self::from_memory_with_sequence(node_id, memory, layer_count, 0)
    }

    /// Create a new HNSW node from a memory with explicit sequence number.
    ///
    /// The sequence number is used for snapshot isolation - observations with
    /// sequence <= snapshot_generation are visible in snapshot-isolated recalls.
    ///
    /// # Errors
    ///
    /// Returns `HnswError::InvalidDimension` when the memory embedding does not
    /// match the expected dimensionality for the index.
    pub fn from_memory_with_sequence(
        node_id: u32,
        memory: Arc<Memory>,
        layer_count: u8,
        sequence_number: u64,
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
            sequence_number: AtomicU64::new(sequence_number),
            last_access_epoch: AtomicU64::new(0),
            embedding_ptr: AtomicPtr::new(embedding_ptr),
            connections_ptr: AtomicPtr::new(std::ptr::null_mut()),
            memory,
        })
    }

    /// Get the sequence number for this node (for snapshot isolation).
    #[must_use]
    pub fn get_sequence_number(&self) -> u64 {
        self.sequence_number.load(Ordering::Acquire)
    }

    /// Set the sequence number for this node (called when observation commits).
    pub fn set_sequence_number(&self, seq: u64) {
        self.sequence_number.store(seq, Ordering::Release);
    }

    /// Get the embedding for this node
    #[allow(unsafe_code)]
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
    #[allow(unsafe_code)]
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
    #[allow(unsafe_code)]
    ///
    /// # Errors
    ///
    /// Propagates allocation or structural errors when the connection block
    /// cannot be updated safely.
    pub fn add_connection(&self, layer: usize, edge: HnswEdge) -> Result<(), super::HnswError> {
        let connections_ptr = self.connections_ptr.load(Ordering::Acquire);

        if connections_ptr.is_null() {
            // Initialize connections if not present
            let mut connections = Box::new(ConnectionBlock::new());
            connections.add_edge(layer, edge)?;
            let new_ptr = Box::into_raw(connections);

            // Try to install the new connections
            if self
                .connections_ptr
                .compare_exchange(
                    std::ptr::null_mut(),
                    new_ptr,
                    Ordering::Release,
                    Ordering::Acquire,
                )
                .is_ok()
            {
                Ok(())
            } else {
                // Another thread initialized it, clean up and retry
                unsafe {
                    drop(Box::from_raw(new_ptr));
                }
                self.add_connection(layer, edge)
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
    #[allow(unsafe_code)]
    fn drop(&mut self) {
        // Clean up allocated embedding
        let embedding_ptr = self.embedding_ptr.load(Ordering::Acquire);
        if !embedding_ptr.is_null() {
            unsafe {
                drop(Box::from_raw(embedding_ptr));
            }
        }

        // Clean up connections
        let connections_ptr = self.connections_ptr.load(Ordering::Acquire);
        if !connections_ptr.is_null() {
            unsafe {
                drop(Box::from_raw(connections_ptr));
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

impl Default for ConnectionBlock {
    fn default() -> Self {
        Self::new()
    }
}

impl ConnectionBlock {
    /// Create a new connection block
    #[must_use]
    pub fn new() -> Self {
        Self {
            layer_connections: (0..16).map(|_| SmallVec::new()).collect(),
            ref_count: AtomicU32::new(1),
            generation: 0,
        }
    }

    /// Add an edge to a specific layer
    ///
    /// # Errors
    ///
    /// Returns `HnswError::CorruptedGraph` when the requested layer index is
    /// out of bounds for the current connection block.
    pub fn add_edge(&mut self, layer: usize, edge: HnswEdge) -> Result<(), super::HnswError> {
        if layer >= self.layer_connections.len() {
            return Err(super::HnswError::CorruptedGraph(format!(
                "Layer {layer} exceeds maximum"
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
    #[must_use]
    pub fn new(target_id: u32, distance: f32, confidence: Confidence) -> Self {
        Self {
            target_id,
            cached_distance: distance,
            confidence_weight: confidence.raw(),
            edge_metadata: EdgeMetadata::default(),
        }
    }

    /// Get the confidence-weighted distance
    #[must_use]
    pub fn weighted_distance(&self) -> f32 {
        self.cached_distance * (1.0 - self.confidence_weight)
    }
}

#[derive(Clone, Copy, Debug, Default)]
#[repr(C)]
/// Metadata for HNSW graph edges
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
    use std::fmt::Debug;

    type TestResult<T = ()> = Result<T, String>;

    fn ensure_eq<T>(actual: &T, expected: &T, context: &str) -> TestResult
    where
        T: PartialEq + Debug,
    {
        if actual == expected {
            Ok(())
        } else {
            Err(format!("{context}: expected {expected:?}, got {actual:?}"))
        }
    }

    #[test]
    fn test_node_cache_alignment() {
        // HnswNode should be 64-byte aligned (one cache line)
        // Size may be larger than 64 bytes due to added sequence_number field
        assert_eq!(std::mem::align_of::<HnswNode>(), 64);
        // Verify size is reasonable (not excessively large)
        assert!(std::mem::size_of::<HnswNode>() <= 128);
    }

    #[test]
    fn test_edge_simd_alignment() {
        // HnswEdge should be 16-byte aligned for SIMD operations
        assert_eq!(std::mem::size_of::<HnswEdge>(), 16);
        assert_eq!(std::mem::align_of::<HnswEdge>(), 16);
    }

    #[test]
    fn test_node_creation() -> TestResult {
        // use crate::MemoryBuilder;
        // use chrono::Utc;

        // Create memory directly instead of through episode conversion
        let memory = Arc::new(crate::Memory::new(
            "test".to_string(),
            [0.5f32; 768],
            Confidence::HIGH,
        ));

        let node = HnswNode::from_memory(0, memory.clone(), 3)
            .map_err(|err| format!("create node from memory: {err:?}"))?;
        ensure_eq(&node.node_id, &0, "node id matches")?;
        ensure_eq(&node.layer_count.load(Ordering::Relaxed), &3, "layer count")?;
        ensure_eq(&node.confidence, &memory.confidence, "confidence matches")
    }
}
