//! Virtual File System - Directory Wrapper
//!
//! This module defines the [`Directory`] struct, which is the VFS-layer
//! in-memory representation of a mounted directory. It wraps an [`Inode`] and
//! adds the state that the VFS needs to traverse the directory tree: a lazily
//! loaded child list, a weak reference to the parent directory, and an
//! optional mount-root pointer for crossing mount boundaries.
//!
//! [`Directory`] is distinct from the [`Inode`] it wraps. The [`Inode`]
//! represents the on-disk object and its metadata. The `Directory` represents
//! the VFS-layer view of that object while it is live in memory - its position
//! in the namespace, its relationship to adjacent directories, and any mount
//! point overlaid on top of it.
//!
//! A [`Directory`]'s `children` field starts as `None`. The first time the path
//! resolver needs to descend into the directory, it calls
//! [`Directory::ensure_children_loaded`], which dispatches to the filesystem
//! driver to populate the child list. Subsequent traversals reuse the
//! in-memory list without further driver calls. This is the primary lazy
//! loading seam; the future cache manager will extend it with eviction
//! support.
//!
//! When a filesystem is mounted on a directory, the VFS sets `mount_flag` on
//! the directory's [`Inode`] and populates `mount_root` on the [`Directory`]
//! wrapper with an [`Arc<Directory>`] pointing to the root of the mounted
//! filesystem. During downward path resolution, the resolver checks
//! `mount_root` and crosses the boundary transparently. During upward
//! resolution (`..`), the resolver detects that the current directory is a
//! mounted root and crosses back to the host directory.

#![allow(dead_code)]

use alloc::sync::{Arc, Weak};
use alloc::vec::Vec;

use super::inode::{Inode, DirectoryEntry};
use super::types::FileSystemError;

/// The VFS-layer in-memory wrapper for a directory inode.
///
/// A [`Directory`] is created by the VFS the first time a directory inode is
/// resolved during path traversal, and it remains alive as long as any
/// [`Arc<Directory>`] reference exists, held by a parent's child list, by the
/// `VirtualFileSystem` root pointer, or by an active path resolution on the
/// call stack.
///
/// The `children` field is the lazy-loaded list of this directory's immediate
/// children. It is `None` until the first traversal demand, at which point the
/// VFS populates it via the filesystem driver. This design keeps memory usage
/// proportional to the portion of the directory tree that has actually been
/// accessed, not the total size of the filesystem.
///
/// The `parent` field is a [`Weak<Directory>`] to avoid reference cycles. The
/// global root directory has `parent` set to `None`; all other directories
/// have a valid weak reference to their parent. Upgrading a `Weak` parent
/// reference must always succeed while the directory is live, as a failure
/// indicates a kernel invariant violation.
///
/// The `mount_root` field is `None` for ordinary directories. When a
/// filesystem is mounted on this directory, the VFS sets it to
/// `Some(Arc<Directory>)` pointing to the root of the mounted filesystem, and
/// sets `mount_flag` on the underlying `Inode`. The path resolver uses this
/// field to cross mount boundaries downward without a separate table lookup.
pub struct Directory {
    /// The underlying inode for this directory.
    ///
    /// Carries the directory's inode number, owning filesystem instance,
    /// metadata, permissions, timestamps, and VFS state flags (including
    /// `mount_flag`). All driver calls related to this directory's on-disk
    /// state are dispatched through `inode.filesystem`.
    pub inode: Arc<Inode>,

    /// The lazily loaded list of this directory's immediate children.
    ///
    /// `None` means the child list has not yet been loaded from disk. The
    /// first call to `ensure_children_loaded` dispatches to the filesystem
    /// driver and populates this field. Once `Some`, the list is the
    /// authoritative in-memory view of the directory's contents and is updated
    /// directly when children are created or deleted through the VFS.
    ///
    /// Each [`DirectoryEntry`] in the list starts with `inode` set to `None`
    /// and is promoted to `Some(Arc<Inode>)` lazily when the entry is
    /// individually resolved.
    pub children: Option<Vec<DirectoryEntry>>,

    /// A weak reference to this directory's parent in the namespace.
    ///
    /// `None` only for the global root directory. For all other directories,
    /// this must be a valid weak reference that can always be successfully
    /// upgraded while the [`Directory`] is live. The path resolver upgrades
    /// this reference when handling `..` components. Upgrade failure is
    /// treated as an internal kernel error.
    pub parent: Option<Weak<Directory>>,

    /// The root directory of a filesystem mounted on top of this directory,
    /// if any.
    ///
    /// `None` for ordinary directories. When a filesystem is mounted here,
    /// the VFS sets this to `Some(Arc<Directory>)` pointing to the mounted
    /// filesystem's root, and simultaneously sets `mount_flag` on `self.inode`
    /// as a fast-path signal. The path resolver checks `mount_flag` first; if
    /// set, it follows `mount_root` to cross the boundary rather than
    /// descending into this directory's own children.
    ///
    /// Set by `VirtualFileSystem::mount` and cleared by
    /// `VirtualFileSystem::unmount`.
    pub mount_root: Option<Arc<Directory>>,
}

impl Directory {
    /// Constructs a new [`Directory`] wrapper around the given inode with no
    /// children loaded, no parent, and no mount root.
    ///
    /// Used when creating the global root directory during VFS initialization,
    /// before any parent or mount information is available.
    ///
    /// # Arguments
    ///
    /// * `inode` - The directory inode this wrapper represents.
    pub fn new_root(inode: Arc<Inode>) -> Self {
        Self {
            inode,
            children:   None,
            parent:     None,
            mount_root: None,
        }
    }

    /// Constructs a new [`Directory`] wrapper around the given inode with a
    /// known parent and no children loaded.
    ///
    /// Used by the VFS when materializing a directory during path traversal
    /// or lazy child loading.
    ///
    /// # Arguments
    ///
    /// * `inode`  - The directory inode this wrapper represents.
    /// * `parent` - A weak reference to this directory's parent in the
    ///   namespace.
    pub fn new(inode: Arc<Inode>, parent: Weak<Directory>) -> Self {
        Self {
            inode,
            children:   None,
            parent:     Some(parent),
            mount_root: None,
        }
    }

    /// Ensures this directory's child list is loaded into memory, calling the
    /// filesystem driver if it has not been loaded yet.
    ///
    /// If `children` is already `Some`, this method returns immediately
    /// without calling the driver. Otherwise it calls
    /// [`super::driver::FileSystemInstance::read_dir`]
    /// on the owning filesystem instance, stores the result in `children`, and
    /// returns a shared reference to the populated list.
    ///
    /// This is the primary lazy loading seam for the directory tree. All path
    /// resolver code that needs to inspect a directory's children must call
    /// this method rather than accessing `children` directly.
    ///
    /// # Returns
    ///
    /// Returns a shared reference to the loaded `Vec<DirectoryEntry>`, or
    /// `Err(FileSystemError)` if the driver call fails.
    pub fn ensure_children_loaded(
        &mut self,
    ) -> Result<&Vec<DirectoryEntry>, FileSystemError> {
        if self.children.is_none() {
            let entries = self.inode.filesystem.read_dir(&self.inode)?;
            self.children = Some(entries);
        }

        // Safe to unwrap: we just assigned Some above if it was None.
        Ok(self.children.as_ref().unwrap())
    }

    /// Looks up a child entry by name in this directory's child list, loading
    /// the list from the filesystem driver first if it has not been loaded yet.
    ///
    /// The lookup is a linear scan of the child list. This is acceptable at
    /// this stage; a future optimization could introduce a name-keyed hash map
    /// as a secondary index on the child list. (TODO: improve the lookup alg.)
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
    /// Returns a mutable reference to the matching [`DirectoryEntry`] if found,
    /// `None` if no entry with that name exists, or `Err(FileSystemError)` if
    /// the child list could not be loaded from the driver.
    pub fn lookup_child(
        &mut self,
        name: &str,
    ) -> Result<Option<&mut DirectoryEntry>, FileSystemError> {
        self.ensure_children_loaded()?;

        // Safe to unwrap: ensure_children_loaded guarantees Some on success.
        Ok(self.children
            .as_mut()
            .unwrap()
            .iter_mut()
            .find(|entry| entry.name == name))
    }

    /// Adds a new child entry to this directory's in-memory child list.
    ///
    /// Called by the VFS after the filesystem driver has successfully created
    /// a new file, directory, or symlink on disk, to keep the in-memory child
    /// list consistent with the on-disk state.
    ///
    /// If the child list has not yet been loaded, this method loads it first
    /// before appending, so that the list remains complete. Callers must not
    /// add a child whose name already exists in the list; doing so would leave
    /// the directory in an inconsistent state.
    ///
    /// # Arguments
    ///
    /// * `entry` - The `DirectoryEntry` to append to the child list.
    ///
    /// # Returns
    ///
    /// Returns `Ok(())` on success, or `Err(FileSystemError)` if the child
    /// list could not be loaded from the driver.
    pub fn add_child(
        &mut self,
        entry: DirectoryEntry,
    ) -> Result<(), FileSystemError> {
        self.ensure_children_loaded()?;

        // Safe to unwrap: ensure_children_loaded guarantees Some on success.
        self.children.as_mut().unwrap().push(entry);
        Ok(())
    }

    /// Removes a child entry by name from this directory's in-memory child
    /// list.
    ///
    /// Called by the VFS after a directory entry has been unlinked on disk, to
    /// keep the in-memory child list consistent with the on-disk state.
    ///
    /// # Arguments
    ///
    /// * `name` - The name of the child entry to remove.
    ///
    /// # Returns
    ///
    /// Returns `Ok(true)` if the entry was found and removed, `Ok(false)` if
    /// no entry with that name existed, or `Err(FileSystemError)` if the child
    /// list could not be loaded from the driver. A `false` return when an
    /// unlink was believed to have succeeded on disk indicates an inconsistency
    /// and should be treated as an internal error by the caller.
    pub fn remove_child(
        &mut self,
        name: &str,
    ) -> Result<bool, FileSystemError> {
        self.ensure_children_loaded()?;

        let children = self.children.as_mut().unwrap();
        if let Some(pos) = children.iter().position(|e| e.name == name) {
            children.swap_remove(pos);
            return Ok(true);
        }

        Ok(false)
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

    /// Checks if a filesystem is currently mounted on this directory.
    ///
    /// This is a convenience wrapper around the `mount_flag` on the underlying
    /// inode. When `true`, the path resolver follows `mount_root` rather than
    /// descending into this directory's own children.
    ///
    /// # Returns
    ///
    /// Returns `true` when `self.inode.mount_flag` is set, and `false`
    /// otherwise.
    pub fn is_mount_point(&self) -> bool {
        self.inode.mount_flag
    }

    /// Upgrades the weak parent reference and returns the parent directory.
    ///
    /// This method should only be called when the caller knows a parent
    /// exists (i.e., `is_root()` is `false`). If the upgrade fails, it means
    /// the parent [`Directory`] was dropped while this directory was still
    /// live, which is a kernel invariant violation.
    ///
    /// # Returns
    ///
    /// Returns `Ok(Arc<Directory>)` if the parent reference could be upgraded,
    /// or `Err(FileSystemError::InternalError)` if this directory has no
    /// parent or the weak reference could not be upgraded.
    pub fn parent(&self) -> Result<Arc<Directory>, FileSystemError> {
        self.parent
            .as_ref()
            .and_then(|weak| weak.upgrade())
            .ok_or(FileSystemError::InternalError)
    }
}
