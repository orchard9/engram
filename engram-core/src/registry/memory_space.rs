//! Memory space registry implementation and handle management.

use chrono::{DateTime, Utc};
use dashmap::DashMap;
use std::fmt;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use tokio::sync::Mutex;

use crate::registry::error::MemorySpaceError;
use crate::{MemorySpaceId, MemoryStore};

#[cfg(feature = "memory_mapped_persistence")]
use crate::storage::{MemorySpacePersistence, PersistenceConfig, RecoveryReport};

/// Factory signature used to provision a [`MemoryStore`] for a given memory space.
type StoreFactory = dyn Fn(&MemorySpaceId, &SpaceDirectories) -> Result<Arc<MemoryStore>, MemorySpaceError>
    + Send
    + Sync;

/// Registry that manages creation, lookup, and lifecycle for memory spaces.
pub struct MemorySpaceRegistry {
    handles: DashMap<MemorySpaceId, Arc<SpaceHandle>>,
    #[cfg(feature = "memory_mapped_persistence")]
    persistence_handles: DashMap<MemorySpaceId, Arc<MemorySpacePersistence>>,
    #[cfg(feature = "memory_mapped_persistence")]
    persistence_config: PersistenceConfig,
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
        Self::with_persistence_config(
            persistence_root,
            store_factory,
            PersistenceConfig::default(),
        )
    }

    /// Create a new registry with custom persistence configuration.
    #[cfg(feature = "memory_mapped_persistence")]
    pub fn with_persistence_config<P, F>(
        persistence_root: P,
        store_factory: F,
        persistence_config: PersistenceConfig,
    ) -> Result<Self, MemorySpaceError>
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
            persistence_handles: DashMap::new(),
            persistence_config,
            init_lock: Mutex::new(()),
            factory: Arc::new(store_factory),
            persistence_root: root,
        })
    }

    /// Create a new registry without persistence configuration (when feature is disabled).
    #[cfg(not(feature = "memory_mapped_persistence"))]
    fn with_persistence_config<P, F>(
        persistence_root: P,
        store_factory: F,
        _persistence_config: PersistenceConfig,
    ) -> Result<Self, MemorySpaceError>
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

    /// Get or create a persistence handle for the given memory space.
    ///
    /// This method is idempotent and thread-safe. Multiple concurrent calls
    /// for the same space will return the same handle.
    #[cfg(feature = "memory_mapped_persistence")]
    pub async fn persistence_handle(
        &self,
        space_id: &MemorySpaceId,
    ) -> Result<Arc<MemorySpacePersistence>, MemorySpaceError> {
        if let Some(existing) = self.persistence_handles.get(space_id) {
            return Ok(existing.value().clone());
        }

        let _guard = self.init_lock.lock().await;

        if let Some(existing) = self.persistence_handles.get(space_id) {
            return Ok(existing.value().clone());
        }

        let directories = self.ensure_directories(space_id)?;
        let handle =
            MemorySpacePersistence::new(space_id.clone(), &self.persistence_config, &directories)
                .map_err(|e| MemorySpaceError::StoreInit {
                id: space_id.clone(),
                source: Box::new(e),
            })?;

        let handle = Arc::new(handle);
        self.persistence_handles
            .insert(space_id.clone(), handle.clone());

        tracing::info!(space = %space_id, "created persistence handle");
        Ok(handle)
    }

    /// Recover all memory spaces by scanning persistence directories and replaying WAL logs.
    ///
    /// This method scans the persistence root directory for existing space directories
    /// and attempts to recover each space's WAL. Recovery is logged per-space.
    #[cfg(feature = "memory_mapped_persistence")]
    pub async fn recover_all(&self) -> Result<Vec<RecoveryReport>, MemorySpaceError> {
        let mut reports = Vec::new();

        let entries = std::fs::read_dir(&self.persistence_root).map_err(|e| {
            MemorySpaceError::DataRootUnavailable {
                path: self.persistence_root.clone(),
                source: e,
            }
        })?;

        for entry_result in entries {
            let entry = entry_result.map_err(|e| MemorySpaceError::DataRootUnavailable {
                path: self.persistence_root.clone(),
                source: e,
            })?;

            let path = entry.path();
            if !path.is_dir() {
                continue;
            }

            let Some(space_id_str) = path.file_name().and_then(|n| n.to_str()) else {
                tracing::warn!(path = ?path, "skipping directory with invalid UTF-8 name");
                continue;
            };

            let space_id = match MemorySpaceId::try_from(space_id_str) {
                Ok(id) => id,
                Err(e) => {
                    tracing::warn!(
                        path = ?path,
                        error = ?e,
                        "skipping directory with invalid space ID"
                    );
                    continue;
                }
            };

            tracing::info!(space = %space_id, "recovering memory space");

            let persistence = self.persistence_handle(&space_id).await?;
            let wal_dir = path.join("wal");

            match persistence.recover(&wal_dir) {
                Ok(report) => {
                    tracing::info!(
                        space = %space_id,
                        recovered = report.recovered_entries,
                        corrupted = report.corrupted_entries,
                        duration = ?report.recovery_duration,
                        "space recovery completed"
                    );
                    reports.push(report);
                }
                Err(e) => {
                    tracing::error!(
                        space = %space_id,
                        error = ?e,
                        "space recovery failed"
                    );
                    return Err(MemorySpaceError::StoreInit {
                        id: space_id,
                        source: Box::new(e),
                    });
                }
            }
        }

        Ok(reports)
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
