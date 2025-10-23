//! Memory space registry implementation and handle management.

use chrono::{DateTime, Utc};
use dashmap::DashMap;
use std::fmt;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use tokio::sync::Mutex;

use crate::registry::error::MemorySpaceError;
use crate::{MemorySpaceId, MemoryStore};

/// Factory signature used to provision a [`MemoryStore`] for a given memory space.
type StoreFactory = dyn Fn(&MemorySpaceId, &SpaceDirectories) -> Result<Arc<MemoryStore>, MemorySpaceError>
    + Send
    + Sync;

/// Registry that manages creation, lookup, and lifecycle for memory spaces.
pub struct MemorySpaceRegistry {
    handles: DashMap<MemorySpaceId, Arc<SpaceHandle>>,
    init_lock: Mutex<()>,
    factory: Arc<StoreFactory>,
    persistence_root: PathBuf,
}

impl MemorySpaceRegistry {
    /// Create a new registry rooted at the provided persistence directory.
    pub fn new<P, F>(persistence_root: P, store_factory: F) -> Result<Self, MemorySpaceError>
    where
        P: Into<PathBuf>,
        F: Fn(&MemorySpaceId, &SpaceDirectories) -> Result<Arc<MemoryStore>, MemorySpaceError>
            + Send
            + Sync
            + 'static,
    {
        let root = persistence_root.into();
        if let Err(err) = std::fs::create_dir_all(&root) {
            return Err(MemorySpaceError::data_root(root, err));
        }

        Ok(Self {
            handles: DashMap::new(),
            init_lock: Mutex::new(()),
            factory: Arc::new(store_factory),
            persistence_root: root,
        })
    }

    /// Resolve or create the memory space identified by `space_id`.
    pub async fn create_or_get(
        &self,
        space_id: &MemorySpaceId,
    ) -> Result<Arc<SpaceHandle>, MemorySpaceError> {
        if let Some(existing) = self.handles.get(space_id) {
            return Ok(existing.value().clone());
        }

        let _guard = self.init_lock.lock().await;

        if let Some(existing) = self.handles.get(space_id) {
            return Ok(existing.value().clone());
        }

        let directories = self.ensure_directories(space_id)?;
        let store = (self.factory)(space_id, &directories)?;
        let handle = Arc::new(SpaceHandle::new(space_id.clone(), store, directories));
        self.handles.insert(space_id.clone(), handle.clone());

        tracing::info!(space = %space_id, path = %handle.directories.root.display(), "created memory space");
        Ok(handle)
    }

    /// Fetch an existing memory space handle.
    pub fn get(&self, space_id: &MemorySpaceId) -> Result<Arc<SpaceHandle>, MemorySpaceError> {
        self.handles
            .get(space_id)
            .map(|entry| entry.value().clone())
            .ok_or_else(|| MemorySpaceError::NotFound {
                id: space_id.clone(),
            })
    }

    /// List all registered memory spaces (ordered by identifier).
    pub fn list(&self) -> Vec<SpaceSummary> {
        let mut spaces: Vec<_> = self
            .handles
            .iter()
            .map(|entry| entry.value().summary())
            .collect();
        spaces.sort_by(|a, b| a.id.as_str().cmp(b.id.as_str()));
        spaces
    }

    /// Ensure the provided collection of spaces exists, creating any that are missing.
    pub async fn ensure_spaces<I>(&self, spaces: I) -> Result<(), MemorySpaceError>
    where
        I: IntoIterator<Item = MemorySpaceId>,
    {
        for id in spaces {
            self.create_or_get(&id).await?;
        }
        Ok(())
    }

    fn ensure_directories(
        &self,
        space_id: &MemorySpaceId,
    ) -> Result<SpaceDirectories, MemorySpaceError> {
        let root = self.persistence_root.join(space_id.as_str());
        let wal = root.join("wal");
        let hot = root.join("hot");
        let warm = root.join("warm");
        let cold = root.join("cold");

        for path in [&root, &wal, &hot, &warm, &cold] {
            if let Err(err) = std::fs::create_dir_all(path) {
                return Err(MemorySpaceError::persistence(space_id, path.clone(), err));
            }
        }

        Ok(SpaceDirectories {
            root,
            wal,
            hot,
            warm,
            cold,
        })
    }

    /// Base directory containing all memory space sub-directories.
    pub fn persistence_root(&self) -> &Path {
        &self.persistence_root
    }
}

/// Space handle containing runtime and persistence context for a specific tenant.
pub struct SpaceHandle {
    id: MemorySpaceId,
    store: Arc<MemoryStore>,
    directories: SpaceDirectories,
    created_at: DateTime<Utc>,
}

impl fmt::Debug for SpaceHandle {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("SpaceHandle")
            .field("id", &self.id)
            .field("directories", &self.directories)
            .field("created_at", &self.created_at)
            .finish_non_exhaustive()
    }
}

impl SpaceHandle {
    fn new(id: MemorySpaceId, store: Arc<MemoryStore>, directories: SpaceDirectories) -> Self {
        Self {
            id,
            store,
            directories,
            created_at: Utc::now(),
        }
    }

    /// Identifier for the space.
    #[must_use]
    pub const fn id(&self) -> &MemorySpaceId {
        &self.id
    }

    /// Clone of the underlying store reference for this space.
    #[must_use]
    pub fn store(&self) -> Arc<MemoryStore> {
        Arc::clone(&self.store)
    }

    /// Persistence directories allocated for this space.
    #[must_use]
    pub const fn directories(&self) -> &SpaceDirectories {
        &self.directories
    }

    /// Creation timestamp recorded when the space handle was initialised.
    #[must_use]
    pub const fn created_at(&self) -> DateTime<Utc> {
        self.created_at
    }

    pub(crate) fn summary(&self) -> SpaceSummary {
        SpaceSummary {
            id: self.id.clone(),
            root: self.directories.root.clone(),
            created_at: self.created_at,
        }
    }
}

impl fmt::Debug for MemorySpaceRegistry {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("MemorySpaceRegistry")
            .field("persistence_root", &self.persistence_root)
            .field("space_count", &self.handles.len())
            .finish_non_exhaustive()
    }
}

/// Persistence directories allocated for a memory space.
#[derive(Clone, Debug)]
pub struct SpaceDirectories {
    /// Root directory for the space.
    pub root: PathBuf,
    /// Write-ahead log directory.
    pub wal: PathBuf,
    /// Hot tier storage directory.
    pub hot: PathBuf,
    /// Warm tier storage directory.
    pub warm: PathBuf,
    /// Cold tier storage directory.
    pub cold: PathBuf,
}

/// Summary representation surfaced to CLI/diagnostics callers.
#[derive(Clone, Debug)]
pub struct SpaceSummary {
    /// Memory space identifier.
    pub id: MemorySpaceId,
    /// Root directory allocated for the space.
    pub root: PathBuf,
    /// Timestamp when the space was created/initialised.
    pub created_at: DateTime<Utc>,
}
