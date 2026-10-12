//! Virtual File System - Directory Wrapper
//!
//! This module defines the [`Directory`] struct, which is the VFS-layer
//! in-memory representation of a mounted directory. It wraps an [`Inode`] and
//! adds the parent reference that the VFS needs for `..` traversal.
//!
//! [`Directory`] is distinct from the [`Inode`] it wraps. The [`Inode`]
//! represents the on-disk object and its metadata. The `Directory` represents
//! the VFS-layer view of that object while it is live in memory - its position
//! in the namespace, its relationship to adjacent directories, and any mount
//! point overlaid on top of it.
//!
//! Each child lookup reads a fresh driver snapshot because several wrappers
//! can refer to the same directory inode. The VFS mount table determines
//! whether path resolution crosses into another filesystem.

#![allow(dead_code)]

use alloc::sync::{Arc, Weak};

use super::inode::{Inode, DirectoryEntry};
use super::types::FileSystemError;

/// Ownership of a directory wrapper's parent in the namespace.
pub enum DirectoryParent {
    /// Keeps an ordinary parent alive for `..` traversal.
    Ordinary(Arc<Directory>),

    /// Avoids a cycle between a mount host and its mounted root.
    MountedRoot(Weak<Directory>),
}

/// The VFS-layer in-memory wrapper for a directory inode.
///
/// A [`Directory`] is created when the VFS resolves a directory inode during
/// path traversal. More than one wrapper can currently represent the same
/// inode. A wrapper remains alive as long as any
/// [`Arc<Directory>`] reference exists, held by the `VirtualFileSystem` root
/// pointer, a child wrapper, a working directory, or an active path resolution.
///
/// Ordinary directories hold a strong reference to their parent so `..` can
/// traverse from a retained wrapper after the path walk that created it ends.
/// The global root has no parent. A mounted root holds a weak host reference
/// because the mount table retains both wrappers.
pub struct Directory {
    /// The underlying inode for this directory.
    ///
    /// Carries the directory's inode number, owning filesystem instance,
    /// metadata, permissions, and timestamps. All driver calls related to
    /// this directory's on-disk state are dispatched through `inode.filesystem`.
    pub inode: Arc<Inode>,

    /// The parent wrapper used for `..` traversal.
    ///
    /// `None` only for the global root. Ordinary children retain their parent;
    /// mounted roots use a weak host reference to avoid a reference cycle.
    pub parent: Option<DirectoryParent>,

}

impl Directory {
    /// Constructs a new [`Directory`] wrapper around the given inode with no
    /// parent.
    ///
    /// Used when creating the global root directory during VFS initialization,
    /// before any parent or mount information is available.
    ///
    /// # Arguments
    ///
    /// * `inode` - The directory inode this wrapper represents.
    ///
    /// # Returns
    ///
    /// Returns a root wrapper with no parent.
    pub fn new_root(inode: Arc<Inode>) -> Self {
        Self {
            inode,
            parent:     None,
        }
    }

    /// Constructs a new [`Directory`] wrapper around the given inode with a
    /// strong parent reference.
    ///
    /// Used by the VFS when materializing a directory during path traversal
    /// path resolution. Retaining the parent keeps `..` valid after the
    /// path walk releases its own reference to that parent.
    ///
    /// # Arguments
    ///
    /// * `inode`  - The directory inode this wrapper represents.
    /// * `parent` - The ordinary parent directory in the namespace.
    ///
    /// # Returns
    ///
    /// Returns a wrapper that keeps its ordinary parent alive.
    pub fn new(inode: Arc<Inode>, parent: Arc<Directory>) -> Self {
        Self {
            inode,
            parent:     Some(DirectoryParent::Ordinary(parent)),
        }
    }

    /// Constructs a mounted root whose host retains the mounted root.
    ///
    /// The mount table retains the host while the filesystem is mounted. The
    /// reverse parent reference is weak to avoid a reference cycle.
    ///
    /// # Arguments
    ///
    /// * `inode` - The root inode of the mounted filesystem.
    /// * `host`  - The directory covered by the mount.
    ///
    /// # Returns
    ///
    /// Returns a mounted-root wrapper with a weak host reference.
    pub fn new_mounted_root(inode: Arc<Inode>, host: &Arc<Directory>) -> Self {
        Self {
            inode,
            parent:     Some(DirectoryParent::MountedRoot(Arc::downgrade(host))),
        }
    }

    /// Looks up a child entry by name in a fresh directory snapshot.
    ///
    /// The lookup is a linear scan of the driver's child list. This is
    /// acceptable at this stage. A name index could be added if directory
    /// lookup cost becomes a problem.
    /// The snapshot is dropped after the lookup, so retained wrappers cannot
    /// hold stale child lists.
    ///
    /// The special names `.` and `..` are not stored as real entries and
    /// will not be found by this method. The path resolver handles those
    /// components before calling here.
    ///
    /// # Arguments
    ///
    /// * `name` - The child name to search for. Must be a single path
    ///   component with no `/` separators.
    ///
    /// # Returns
    ///
    /// Returns the matching owned [`DirectoryEntry`] if found, `None` if no
    /// entry has that name, or `Err(FileSystemError)` if the driver read fails.
    pub fn lookup_child(
        &self,
        name: &str,
    ) -> Result<Option<DirectoryEntry>, FileSystemError> {
        Ok(self.inode.filesystem
            .read_dir(&self.inode)?
            .into_iter()
            .find(|entry| entry.name == name))
    }

    /// Checks if this directory is the global root of the namespace.
    ///
    /// A directory is the global root if and only if it has no parent - i.e.,
    /// `parent` is `None`. There is exactly one such directory in the VFS at
    /// any time: the root of the boot filesystem, mounted at `/`.
    ///
    /// # Returns
    ///
    /// Returns `true` when `parent` is `None`, and `false` otherwise.
    pub fn is_root(&self) -> bool {
        self.parent.is_none()
    }

    /// Returns the parent directory, upgrading a mounted root's weak host.
    ///
    /// This method should only be called when the caller knows a parent
    /// exists (i.e., `is_root()` is `false`). Ordinary parents remain alive
    /// through their child's strong reference. A mounted root's host remains
    /// alive through the mount table until unmount.
    ///
    /// # Returns
    ///
    /// Returns the parent, or `InternalError` if there is no parent or a
    /// mounted root's weak host reference cannot be upgraded.
    pub fn parent(&self) -> Result<Arc<Directory>, FileSystemError> {
        match self.parent.as_ref() {
            Some(DirectoryParent::Ordinary(parent)) => Ok(Arc::clone(parent)),
            Some(DirectoryParent::MountedRoot(host)) => {
                host.upgrade().ok_or(FileSystemError::InternalError)
            }
            None => Err(FileSystemError::InternalError),
        }
    }
}
