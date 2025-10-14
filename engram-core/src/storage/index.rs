//! Lock-free hash index implementation
//!
//! This module provides lock-free indexing for high-performance concurrent access.

use super::{StorageError, StorageResult};
use dashmap::DashMap;

/// Lock-free hash index backed by a concurrent `DashMap`.
pub struct LockFreeHashIndex {
    entries: DashMap<String, u64>,
}

impl LockFreeHashIndex {
    /// Create a new lock-free hash index
    #[must_use]
    pub fn new() -> Self {
        Self {
            entries: DashMap::new(),
        }
    }

    /// Insert key-value pair
    ///
    /// # Errors
    ///
    /// Returns [`StorageError`] when an empty key is provided.
    pub fn insert(&self, key: &str, value: u64) -> StorageResult<()> {
        if key.trim().is_empty() {
            return Err(StorageError::Configuration(
                "lock-free index requires non-empty keys".to_string(),
            ));
        }

        self.entries.insert(key.to_string(), value);
        Ok(())
    }

    /// Lookup value by key
    #[must_use]
    pub fn lookup(&self, key: &str) -> Option<u64> {
        if key.trim().is_empty() {
            return None;
        }

        self.entries.get(key).map(|entry| *entry.value())
    }
}

impl Default for LockFreeHashIndex {
    fn default() -> Self {
        Self::new()
    }
}
