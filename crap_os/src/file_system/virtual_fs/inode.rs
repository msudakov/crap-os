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
//! - Regular-file and symlink inodes may be referenced by multiple directory
//!   entries through hard links. Directories retain a single parent.
//! - Removing a directory entry (unlinking) does not immediately destroy the
//!   inode; the inode survives until its link count reaches zero and all open
//!   handles are closed.
//! - The path resolver works with directory entries during traversal and only
//!   loads the full inode when it is needed.

#![allow(dead_code)]

use alloc::sync::Arc;
use alloc::string::String;
use crate::spinlock::SpinLock;

use super::types::{
    InodeType, DirectoryEntryType, Permissions, Timestamps,
};
use super::driver::FileSystemInstance;

/// The VFS-layer in-memory representation of a filesystem object.
///
/// An [`Inode`] is an in-memory record for a filesystem object: a file, a
/// directory, or a symlink. A driver may construct multiple records for the
/// same on-disk inode, but the VFS retains one shared record per filesystem
/// instance and inode number. Its table keeps that record alive until a
/// successful free or volume unmount; directory wrappers and file handles
/// may retain references longer.
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
/// Its mutable metadata is protected by a short-held lock because the record
/// can be shared through `Arc<Inode>`.
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
    /// Determines how the VFS interprets the inode during path resolution.
    /// Directories use namespace wrappers, regular files use open-file records,
    /// and symlinks are handled directly through their inodes.
    pub inode_type: InodeType,

    /// Mutable metadata shared by references to this in-memory inode.
    ///
    /// Hold the guard only while reading or changing fields, and release it
    /// before calling a filesystem driver.
    pub(crate) metadata: SpinLock<InodeMetadata>,
}

/// Metadata fields that can change while an [`Inode`] is shared.
pub(crate) struct InodeMetadata {
    /// The size of this inode's data in bytes.
    ///
    /// For regular files, this is the file's current byte length. For symlinks,
    /// this is the byte length of the target path string. For directories, this
    /// field is filesystem-defined, and callers should not rely on it for
    /// anything other than passing it through to a stat snapshot.
    pub(crate) size: u64,

    /// The access-control permissions associated with this inode.
    ///
    /// Currently a placeholder pending the full ACL model design. All
    /// permission checks in the VFS treat any [`Permissions`] value as
    /// universally permissive until the real model is in place. See
    /// [`super::types::Permissions`].
    pub(crate) permissions: Permissions,

    /// Creation, last-modification, and last-access timestamps in Unix epoch
    /// milliseconds.
    ///
    /// Set at creation time via [`super::types::Timestamps::new`] and updated
    /// by the VFS on each relevant operation before the inode is synced back
    /// to disk.
    pub(crate) timestamps: Timestamps,

    /// The number of directory entries that currently reference this inode.
    ///
    /// Starts at `1` when the inode is created. Incremented when a hard link
    /// is created, and decremented when a directory entry is removed. When
    /// `link_count` reaches `0` and no open file handles remain, the VFS
    /// instructs the filesystem driver to free the inode's data blocks and
    /// release the inode slot.
    pub(crate) link_count: u32,

    /// Indicates that this inode has been modified in memory but not yet
    /// synced to disk.
    ///
    /// Set to `true` by VFS reads and writes. The VFS clears it after
    /// [`super::driver::FileSystemInstance::sync_inode`] succeeds during
    /// volume unmount. A separate sync path for the boot volume is not yet
    /// implemented.
    pub(crate) dirty: bool,
}

impl Inode {
    /// Constructs a new [`Inode`] with the supplied identity, type, and
    /// metadata.
    ///
    /// The `dirty` field is initialized to `false`. The caller is responsible
    /// for supplying an accurate `size` for files and symlinks; for
    /// directories, `0` is an acceptable initial value.
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
            metadata: SpinLock::new(InodeMetadata {
                size,
                permissions,
                timestamps,
                link_count,
                dirty: false,
            }),
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

    /// Checks if this inode has in-memory changes that have not yet been
    /// written to disk.
    ///
    /// # Returns
    ///
    /// Returns `true` when `dirty` is set, and `false` otherwise.
    pub fn is_dirty(&self) -> bool {
        self.metadata.lock().dirty
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
        let metadata = self.metadata.lock();
        super::types::InodeStat {
            inode_number: self.inode_number,
            inode_type:   self.inode_type,
            size:         metadata.size,
            permissions:  metadata.permissions.clone(),
            timestamps:   metadata.timestamps,
            link_count:   metadata.link_count,
        }
    }
}

/// A named reference from a parent directory to a child inode.
///
/// A [`DirectoryEntry`] is the VFS-layer representation of a single slot in a
/// driver-provided directory listing. It records the child's name, its inode
/// number, and its type so the path resolver can branch without loading the
/// full inode. Inode references are retained by the VFS, not by entries.
///
/// Filesystem drivers return [`DirectoryEntry`] values from
/// [`super::driver::FileSystemInstance::lookup`] and
/// [`super::driver::FileSystemInstance::read_dir`]. Path resolution uses a
/// fresh entry and asks the VFS for its shared inode when the full inode is
/// needed.
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
    /// Used by the VFS to acquire the full inode on demand.
    pub inode_number: u64,

    /// The kind of object the referenced inode represents.
    ///
    /// Populated by the filesystem driver alongside `inode_number`. Allows
    /// the path resolver to make branching decisions without paying the cost of
    /// loading the full inode first.
    pub entry_type: DirectoryEntryType,
}

impl DirectoryEntry {
    /// Constructs a [`DirectoryEntry`] from a name, inode number, and type.
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
        }
    }
}
