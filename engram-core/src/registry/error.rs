//! Error types used by the memory space registry control plane.

use std::path::PathBuf;

use thiserror::Error;

use crate::MemorySpaceId;
use crate::types::MemorySpaceIdError;

/// Errors surfaced by the memory space registry and lifecycle manager.
#[derive(Debug, Error)]
pub enum MemorySpaceError {
    /// Identifier validation failed.
    #[error(transparent)]
    InvalidId(#[from] MemorySpaceIdError),

    /// Memory space requested is not known to the registry.
    #[error(
        "Memory space '{id}' not found\n  Expected: Previously created memory space id\n  Suggestion: Create the space via registry.create_or_get before use\n  Example: registry.create_or_get(&MemorySpaceId::try_from(\"tenant_a\")?)"
    )]
    NotFound {
        /// Identifier that was looked up.
        id: MemorySpaceId,
    },

    /// Persistence directories could not be provisioned.
    #[error(
        "Failed to prepare persistence directory '{path}' for space '{id}'\n  Expected: Writable filesystem path\n  Suggestion: Ensure Engram has permissions for the data root\n  Example: chmod +w {path}"
    )]
    Persistence {
        /// Memory space identifier.
        id: MemorySpaceId,
        /// Path that failed creation.
        path: PathBuf,
        #[source]
        /// Underlying IO error.
        source: std::io::Error,
    },

    /// Base data root could not be prepared; registry cannot operate.
    #[error(
        "Failed to prepare memory space data root '{path}'\n  Expected: Writable filesystem directory for registry\n  Suggestion: Verify Engram data directory permissions\n  Example: mkdir -p {path} && chown engram {path}"
    )]
    DataRootUnavailable {
        /// Data root that failed creation/validation.
        path: PathBuf,
        #[source]
        /// Underlying IO error.
        source: std::io::Error,
    },

    /// Store factory failed to construct a space-specific memory store.
    #[error(
        "Failed to initialise memory store for space '{id}'\n  Expected: Store factory to succeed\n  Suggestion: Inspect inner error for remediation steps\n  Example: ensure custom store factory handles persistence hooks"
    )]
    StoreInit {
        /// Identifier for which store initialisation failed.
        id: MemorySpaceId,
        #[source]
        /// Source error bubbling up from store construction.
        source: Box<dyn std::error::Error + Send + Sync>,
    },
}

impl MemorySpaceError {
    pub(crate) fn persistence(id: &MemorySpaceId, path: PathBuf, source: std::io::Error) -> Self {
        Self::Persistence {
            id: id.clone(),
            path,
            source,
        }
    }

    pub(crate) const fn data_root(path: PathBuf, source: std::io::Error) -> Self {
        Self::DataRootUnavailable { path, source }
    }
}
