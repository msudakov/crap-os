//! Virtual File System - Inode and Directory Entry
//!
//! This module defines the two most fundamental VFS-layer data structures:
//! the in-memory inode, and the directory entry that binds a name to an inode
//! within a parent directory.
//!
//! Neither structure is filesystem-specific. Every filesystem driver maps its
//! own on-disk representation onto these generic types when satisfying the
//! contracts defined in `driver.rs`. All VFS-layer code above the driver
//! boundary (the path resolver, the directory wrapper, the open-file
//! machinery, and the `VirtualFileSystem` itself) works exclusively with
//! these types and never inspects driver-internal structures.
//!
//! An inode represents the filesystem object itself: its data, its metadata,
//! and its identity within a mounted filesystem instance. A directory entry
//! represents a named reference to an inode within a specific parent
//! directory. These are deliberately kept separate:
//!
//! - One inode may be referenced by multiple directory entries in the future
//!   when hardlink support is added (TODO: add hardlinks in the future).
//! - Removing a directory entry (unlinking) does not immediately destroy the
//!   inode; the inode survives until its link count reaches zero and all open
//!   handles are closed.
//! - The path resolver works with directory entries during traversal and only
//!   loads the full inode when it is needed.

#![allow(dead_code)]

use alloc::sync::Arc;
use alloc::string::String;

use super::types::{
    FileSystemError, InodeType, DirectoryEntryType, Permissions, Timestamps,
};
use super::driver::FileSystemInstance;

/// The VFS-layer in-memory representation of a filesystem object.
///
/// An [`Inode`] is the authoritative in-memory record for a single filesystem
/// object - a file, a directory, or a symlink. It is created by the VFS when
/// a filesystem driver returns inode data in response to a
/// [`super::driver::FileSystemInstance::load_inode`] call, and it remains alive
///  in memory for as long as any [`super::directory::Directory`] wrapper,
/// [`super::file::FileHandle`], or [`super::inode::DirectoryEntry`] holds an
/// `Arc<Inode>` reference to it.
///
/// The `filesystem` field is a back-reference to the
/// [`super::driver::FileSystemInstance`] that owns this inode. All driver
/// operations on this inode (reading data, syncing metadata, freeing blocks)
/// are dispatched through that reference. This means the VFS never needs to
/// look up which driver to call; the inode carries the answer.
///
/// Filesystem drivers may store additional, format-specific state in their
/// own internal tables keyed by `inode_number`, but that data never appears
/// on this struct. The `Inode` contains only what the VFS layer itself needs.
pub struct Inode {
    /// The inode's unique numeric identifier within its owning filesystem
    /// instance.
    ///
    /// Inode numbers are unique only within a single mounted filesystem. Two
    /// inodes on different volumes may share the same number. Code that needs
    /// a system-wide unique key should use [`super::types::OpenFileKey`],
    /// which pairs this number with a filesystem instance identifier.
    pub inode_number: u64,

    /// The mounted filesystem instance that owns and manages this inode.
    ///
    /// All driver calls related to this inode (data reads and writes, metadata
    /// syncs, block frees) are dispatched through this reference. The `Arc`
    /// ensures the filesystem instance remains alive for at least as long as
    /// any of its inodes are live in memory.
    pub filesystem: Arc<dyn FileSystemInstance>,

    /// The kind of filesystem object this inode represents.
    ///
    /// Determines how the VFS interprets the inode during path resolution and
    /// which wrapper type (`Directory`, `File`, or `Symlink` handling) is used
    /// above it.
    pub inode_type: InodeType,

    /// The size of this inode's data in bytes.
    ///
    /// For regular files, this is the file's current byte length. For symlinks,
    /// this is the byte length of the target path string. For directories, this
    /// field is filesystem-defined, and callers should not rely on it for
    /// anything other than passing it through to a stat snapshot.
    pub size: u64,

    /// The access-control permissions associated with this inode.
    ///
    /// Currently a placeholder pending the full ACL model design. All
    /// permission checks in the VFS treat any `Permissions` value as
    /// universally permissive until the real model is in place. See
    /// [`super::types::Permissions`].
    pub permissions: Permissions,

    /// Creation, last-modification, and last-access timestamps in Unix epoch
    /// milliseconds.
    ///
    /// Set at creation time via [`super::types::Timestamps::new`] and updated
    /// by the VFS on each relevant operation before the inode is synced back
    /// to disk.
    pub timestamps: Timestamps,

    /// The number of directory entries that currently reference this inode.
    ///
    /// Starts at `1` when the inode is created. Incremented when a new
    /// directory entry pointing to this inode is created (in future hardlink
    /// support), and decremented when a directory entry is removed. When
    /// `link_count` reaches `0` and no open file handles remain, the VFS
    /// instructs the filesystem driver to free the inode's data blocks and
    /// release the inode slot.
    pub link_count: u32,

    /// Indicates whether a filesystem has been mounted on this inode.
    ///
    /// Only meaningful when `inode_type` is [`InodeType::Directory`]. When
    /// `true`, the path resolver checks the owning
    /// [`super::directory::Directory`] wrapper's
    /// `mount_root` field and crosses the mount boundary rather than
    /// descending into this inode's own children.
    ///
    /// This flag is owned and mutated exclusively by the VFS during mount and
    /// unmount operations. Filesystem drivers never read or write it.
    pub mount_flag: bool,

    /// Indicates that this inode has been modified in memory but not yet
    /// synced to disk.
    ///
    /// Set to `true` by any VFS operation that updates `size`, `permissions`,
    /// or `timestamps`. Cleared when the VFS calls
    /// [`super::driver::FileSystemInstance::sync_inode`]
    /// and the driver confirms the write succeeded. The future cache manager
    /// will use this flag to identify inodes that must be flushed during
    /// writeback.
    pub dirty: bool,

    /// Indicates that all directory entries pointing to this inode have been
    /// removed, but one or more open file handles still hold a live `Arc`
    /// reference to it.
    ///
    /// While `pending_deletion` is `true`, the inode remains accessible
    /// through existing handles but cannot be opened by name (it has no
    /// directory entries). When the last handle closes and the `Arc<Inode>`
    /// reference count drops to zero, the VFS instructs the filesystem driver
    /// to free the inode's data blocks and release the inode slot. This
    /// implements POSIX unlink semantics, where a file can be deleted from
    /// the namespace while remaining readable and writable through any handles
    /// that were already open.
    pub pending_deletion: bool,
}

impl Inode {
    /// Constructs a new `Inode` with the supplied identity, type, and
    /// metadata.
    ///
    /// `mount_flag`, `dirty`, and `pending_deletion` are all initialized to
    /// `false`. The caller is responsible for supplying an accurate `size`
    /// for files and symlinks; for directories, `0` is an acceptable initial
    /// value.
    ///
    /// # Arguments
    ///
    /// * `inode_number` - The inode's unique identifier within `filesystem`.
    /// * `filesystem`   - The mounted filesystem instance that owns this inode.
    /// * `inode_type`   - The kind of filesystem object this inode represents.
    /// * `size`         - The initial data size in bytes.
    /// * `permissions`  - The access-control permissions for this inode.
    /// * `timestamps`   - The creation, modification, and access timestamps.
    /// * `link_count`   - The initial directory-entry reference count; normally
    ///   `1` for a newly created inode.
    pub fn new(
        inode_number: u64,
        filesystem: Arc<dyn FileSystemInstance>,
        inode_type: InodeType,
        size: u64,
        permissions: Permissions,
        timestamps: Timestamps,
        link_count: u32,
    ) -> Self {
        Self {
            inode_number,
            filesystem,
            inode_type,
            size,
            permissions,
            timestamps,
            link_count,
            mount_flag: false,
            dirty: false,
            pending_deletion: false,
        }
    }

    /// Checks if this inode represents a regular file.
    ///
    /// # Returns
    ///
    /// Returns `true` when `inode_type` is [`super::types::InodeType::File`],
    ///  and `false` otherwise.
    pub fn is_file(&self) -> bool {
        matches!(self.inode_type, InodeType::File)
    }

    /// Checks if this inode represents a directory.
    ///
    /// # Returns
    ///
    /// Returns `true` when `inode_type` is
    /// [`super::types::InodeType::Directory`], and `false` otherwise.
    pub fn is_directory(&self) -> bool {
        matches!(self.inode_type, InodeType::Directory)
    }

    /// Checks if this inode represents a symbolic link.
    ///
    /// # Returns
    ///
    /// Returns `true` when `inode_type` is [`super::types::InodeType::Symlink`]
    /// and `false` otherwise.
    pub fn is_symlink(&self) -> bool {
        matches!(self.inode_type, InodeType::Symlink)
    }

    /// Checks if a filesystem is mounted on this inode.
    ///
    /// Only meaningful for directory inodes. When `true`, the path resolver
    /// crosses the mount boundary rather than descending into this directory's
    /// own children.
    ///
    /// # Returns
    ///
    /// Returns `true` when `mount_flag` is set, and `false` otherwise.
    pub fn is_mount_point(&self) -> bool {
        self.mount_flag
    }

    /// Checks if this inode has in-memory changes that have not yet been
    /// written to disk.
    ///
    /// # Returns
    ///
    /// Returns `true` when `dirty` is set, and `false` otherwise.
    pub fn is_dirty(&self) -> bool {
        self.dirty
    }

    /// Checks if all directory entries referencing this inode have
    /// been removed and the inode is waiting for its last open handle to
    /// close before its data blocks are freed.
    ///
    /// # Returns
    ///
    /// Returns `true` when `pending_deletion` is set, and `false` otherwise.
    pub fn is_pending_deletion(&self) -> bool {
        self.pending_deletion
    }

    /// Constructs an [`super::types::InodeStat`] snapshot from this inode's
    /// current metadata.
    ///
    /// The snapshot captures the inode's metadata at the moment of the call.
    /// It does not hold any reference to the inode itself and can be freely
    /// copied into a userspace buffer by a future `stat()` syscall
    /// implementation.
    ///
    /// # Returns
    ///
    /// Returns this inode's metadata [`super::types::InodeStat`] snapshot.
    pub fn stat(&self) -> super::types::InodeStat {
        super::types::InodeStat {
            inode_number: self.inode_number,
            inode_type:   self.inode_type,
            size:         self.size,
            permissions:  self.permissions.clone(),
            timestamps:   self.timestamps,
            link_count:   self.link_count,
        }
    }
}

/// A named reference from a parent directory to a child inode.
///
/// A `DirectoryEntry` is the VFS-layer representation of a single slot in a
/// directory's child list. It records the child's name, its inode number, its
/// type (so the path resolver can branch without loading the full inode), and
/// an optional cached `Arc<Inode>` that is populated the first time the entry
/// is resolved.
///
/// Filesystem drivers return `DirectoryEntry` values from
/// [`super::driver::FileSystemInstance::lookup`] and
/// [`super::driver::FileSystemInstance::read_dir`].
/// At that point, the `inode` field is `None`; the VFS populates it lazily via
/// [`super::driver::FileSystemInstance::load_inode`]
/// the first time the entry is actually traversed or opened.
///
/// Storing the loaded `Arc<Inode>` on the entry avoids redundant driver calls
/// for frequently accessed children, and it forms the in-memory inode cache
/// for the directory tree.
pub struct DirectoryEntry {
    /// The name of this entry within its parent directory.
    ///
    /// This is the single path component (e.g., `bin`, `kernel.elf`)
    /// without any separators. Names must not be empty, must not contain
    /// `/`, and must not be `.` or `..` (those are synthesized by the
    /// path resolver, not stored as real entries).
    pub name: String,

    /// The inode number of the filesystem object this entry references,
    /// within the owning filesystem instance.
    ///
    /// Used to load the full inode on demand via
    /// [`super::driver::FileSystemInstance::load_inode`] when `inode` is
    /// `None`.
    pub inode_number: u64,

    /// The kind of object the referenced inode represents.
    ///
    /// Populated by the filesystem driver alongside `inode_number`. Allows
    /// the path resolver to make branching decisions without paying the cost of
    /// loading the full inode first.
    pub entry_type: DirectoryEntryType,

    /// The loaded in-memory inode for this entry, if it has been resolved.
    ///
    /// `None` when the entry was freshly returned by a filesystem driver and
    /// has not yet been traversed. Set to `Some` by the VFS the first time
    /// this entry is descended into or opened, by calling
    /// [`super::driver::FileSystemInstance::load_inode`] with `inode_number`.
    /// Once set, subsequent traversals reuse the cached `Arc<Inode>` without
    /// further driver calls.
    pub inode: Option<Arc<Inode>>,
}

impl DirectoryEntry {
    /// Constructs a new `DirectoryEntry` as returned by a filesystem driver:
    /// name, inode number, and type are known, but the inode has not yet been
    /// loaded.
    ///
    /// # Arguments
    ///
    /// * `name`         - The entry's name within its parent directory.
    /// * `inode_number` - The inode number of the referenced filesystem object
    ///   within the owning filesystem instance.
    /// * `entry_type`   - The kind of object the referenced inode represents.
    pub fn new(
        name: String,
        inode_number: u64,
        entry_type: DirectoryEntryType,
    ) -> Self {
        Self {
            name,
            inode_number,
            entry_type,
            inode: None,
        }
    }

    /// Constructs a new `DirectoryEntry` with a pre-loaded inode.
    ///
    /// Used by VFS-internal code when both the directory entry metadata and
    /// the inode are already available. E.g., when populating the in-memory
    /// directory tree during an eager boot load of well-known system
    /// directories.
    ///
    /// # Arguments
    ///
    /// * `name`         - The entry's name within its parent directory.
    /// * `inode_number` - The inode number of the referenced filesystem object
    ///   within the owning filesystem instance.
    /// * `entry_type`   - The kind of object the referenced inode represents.
    /// * `inode`        - The already-loaded inode for this entry.
    pub fn with_inode(
        name: String,
        inode_number: u64,
        entry_type: DirectoryEntryType,
        inode: Arc<Inode>,
    ) -> Self {
        Self {
            name,
            inode_number,
            entry_type,
            inode: Some(inode),
        }
    }

    /// Checks if the inode for this entry has already been loaded into memory.
    ///
    /// # Returns
    ///
    /// Returns `true` when `inode` is `Some`, and `false` when it is `None`.
    pub fn is_loaded(&self) -> bool {
        self.inode.is_some()
    }

    /// Attempts to load and cache this entry's inode by calling into the
    /// filesystem driver.
    ///
    /// If the inode is already cached in `self.inode`, this method returns
    /// immediately without calling the driver. Otherwise, it calls
    /// [`super::driver::FileSystemInstance::load_inode`] on the provided
    /// `filesystem` reference, caches the result in `self.inode`, and returns
    /// a clone of the `Arc<Inode>`.
    ///
    /// # Arguments
    ///
    /// * `filesystem` - The filesystem instance that owns this entry's inode.
    ///   Must be the same instance from which this `DirectoryEntry` was
    ///   originally obtained.
    ///
    /// # Returns
    ///
    /// Returns an `Arc<Inode>` for this entry, either from the cache or freshly
    /// loaded from the driver. Returns `Err(FileSystemError)` if the driver
    /// call fails.
    pub fn load_inode(
        &mut self,
        filesystem: &Arc<dyn FileSystemInstance>,
    ) -> Result<Arc<Inode>, FileSystemError> {
        if let Some(ref cached) = self.inode {
            return Ok(Arc::clone(cached));
        }

        let loaded = filesystem.load_inode(self.inode_number)?;
        self.inode = Some(Arc::clone(&loaded));
        Ok(loaded)
    }
}
