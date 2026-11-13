//! Virtual File System - Core Types
//!
//! This module defines the foundational types shared across the entire VFS
//! layer. These types form the common vocabulary used by the virtual file
//! system, the path resolver, the inode and directory abstractions, and
//! every filesystem driver that implements the `FileSystemDriver` and
//! `FileSystemInstance` contracts.
//!
//! No filesystem-driver-specific logic lives here. All types in this module
//! are VFS-layer concepts, independent of any on-disk format. Filesystem
//! drivers consume these types when satisfying the VFS contracts defined in
//! `driver.rs`.
//!
//! The VFS uses the following types:
//!   - [`FileSystemError`]    : Canonical error type for all VFS operations.
//!   - [`InodeType`]          : Classifies an inode as a file, directory, or
//!                              symlink.
//!   - [`DirectoryEntryType`] : Classifies a directory entry, mirroring
//!                              `InodeType` at the entry level.
//!   - [`AccessMode`]         : Describes the access rights requested when
//!                              opening a file handle.
//!   - [`CreateMode`]         : Selects ordinary or exclusive create-and-open.
//!   - [`FileLockState`]      : Tracks the current write-lock state of an open
//!                              file's master record.
//!   - [`Timestamps`]         : Wall-clock timestamps recorded on every inode.
//!   - [`Permissions`]        : Placeholder for the future ACL-based permission
//!                              model.
//!   - [`InodeStat`]          : A point-in-time snapshot of inode metadata,
//!                              returned by `VirtualFileSystem::stat()` and
//!                              eventually copied into usermode `stat` buffers.
//!   - [`OpenFileKey`]        : Composite key used to look up a file's master
//!                              record in the VFS open-file table.

#![allow(dead_code)]

use crate::kernel_hashmap::KernelHash;

/// Canonical error type for all VFS and filesystem driver operations.
///
/// Every fallible function in the VFS layer returns `Result<T,
/// FileSystemError>`. Filesystem drivers map their own internal error
/// conditions onto these variants when returning from trait methods, so the
/// VFS and path resolver never need to know which driver is underneath.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FileSystemError {
    /// The on-disk format was not recognized by the driver, or the superblock
    /// failed validation (bad magic number, unsupported version, etc.).
    InvalidFormat,

    /// A required filesystem structure (superblock, inode table, block group
    /// descriptor, etc.) could not be read because it fell outside the
    /// device's reported capacity.
    CorruptStructure,

    /// The block device reported an unrecoverable read error.
    DiskReadError,

    /// The block device reported an unrecoverable write error.
    DiskWriteError,

    /// The filesystem is mounted read-only, and the requested operation would
    /// modify it.
    ReadOnlyFilesystem,

    /// The filesystem has no free blocks remaining.
    OutOfSpace,

    /// The filesystem has no free inode slots remaining.
    OutOfInodes,

    /// The requested path component or directory entry does not exist.
    NotFound,

    /// A path component that was expected to be a directory is not one.
    NotADirectory,

    /// A directory was supplied where a non-directory destination is required
    /// (e.g., renaming a regular file or symlink over a directory).
    IsADirectory,

    /// A path component that was expected to be a file is not one (e.g.,
    /// attempting to `read` a directory inode directly).
    NotAFile,

    /// A path component that was expected to be a symlink is not one.
    NotASymlink,

    /// A directory entry with the requested name already exists.
    AlreadyExists,

    /// The operation requires its source and destination to belong to the
    /// same file system instance, but they belong to different instances.
    CrossFileSystemOperation,

    /// Creating another hard link would exceed the VFS or driver limit
    /// on the number of directory entries naming one inode.
    TooManyHardLinks,

    /// A directory that was expected to be empty (e.g., for removal) still
    /// contains entries.
    DirectoryNotEmpty,

    /// An object is protected, is an active mount point, or is the global
    /// root. Retry after its holders release protection where possible.
    ResourceBusy,

    /// Symlink resolution exceeded
    /// [`crate::file_system::virtual_fs::resolver::MAX_SYMLINK_DEPTH`] hops,
    /// indicating a symlink loop or an exceedingly-deep chain.
    TooManySymlinks,

    /// The requested operation is prohibited by the rules for the target
    /// object, regardless of the caller's permissions (e.g., creating a
    /// directory hard link).
    OperationNotAllowed,

    /// The calling context does not have the required permissions for the
    /// requested operation.
    PermissionDenied,

    /// An attempt was made to write to a file handle opened with
    /// [`AccessMode::ReadOnly`], or to read from a handle opened with
    /// [`AccessMode::WriteOnly`].
    InvalidAccessMode,

    /// A write lock was requested on a file that is already held under an
    /// exclusive write lock by another handle.
    LockContention,

    /// An attempt was made to write through a handle while another handle
    /// holds an exclusive write lock on the same file.
    FileLocked,

    /// A file handle was closed, was not registered with this VFS, or did not
    /// belong to the expected open-file master record.
    InvalidHandle,

    /// A signed seek would place the file position before byte zero.
    InvalidOffset,

    /// A signed seek would place the file position above [`i64::MAX`] bytes.
    OffsetOverflow,

    /// The path string was empty, contained illegal characters, or exceeded
    /// the maximum supported length.
    InvalidPath,

    /// An operation was attempted on a file whose last directory entry was
    /// removed and whose storage awaits reclamation after its last close.
    PendingDeletion,

    /// A mount point could not be established because the target directory
    /// already has a filesystem mounted on it.
    AlreadyMounted,

    /// An unmount was attempted on a path that is not currently a mount point.
    NotMounted,

    /// No filesystem driver registered under the requested name was found in
    /// the VFS driver registry.
    UnknownDriver,

    /// An internal VFS invariant was violated (e.g., a `Weak` parent reference
    /// could not be upgraded). This represents a kernel bug and should be
    /// treated as fatal.
    InternalError,
}

/// Classifies an inode by the kind of filesystem object it represents.
///
/// The VFS recognizes exactly three inode kinds: regular files, directories,
/// and symlinks. Symlinks are handled directly through their inodes rather
/// than through a separate wrapper type.
///
/// This is deliberately narrow. Devices, pipes, and sockets are not
/// filesystem objects in this kernel's model and are managed by their own
/// subsystems outside the VFS.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InodeType {
    /// A regular file is a named sequence of bytes stored on disk.
    File,

    /// A directory is a named container of
    /// [`crate::file_system::virtual_fs::inode::DirectoryEntry`] records, each
    /// of which maps a name to a child inode.
    Directory,

    /// A symbolic link is a named inode whose data is a target path string.
    /// Followed transparently during path resolution unless the caller
    /// explicitly requests the symlink inode itself.
    Symlink,
}

/// Classifies a [`crate::file_system::virtual_fs::inode::DirectoryEntry`] by
/// the kind of inode it references.
///
/// Mirrors [`InodeType`] at the directory-entry level. Filesystem drivers
/// populate this field when returning entries from
/// [`crate::file_system::virtual_fs::driver::FileSystemInstance::lookup`] or
/// [`crate::file_system::virtual_fs::driver::FileSystemInstance::read_dir`], so
/// the VFS path resolver can make branching decisions without loading the
/// target inode first.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DirectoryEntryType {
    /// The entry points to a regular file inode.
    File,

    /// The entry points to a directory inode.
    Directory,

    /// The entry points to a symlink inode.
    Symlink,
}

/// Describes the access rights granted to a
/// [`crate::file_system::virtual_fs::file::FileHandle`] at open time.
///
/// The access mode is fixed when the handle is created and cannot be upgraded
/// afterwards. It is distinct from the file's write-lock state: a process may
/// hold a [`AccessMode::WriteOnly`] or [`AccessMode::ReadWrite`] handle without
/// ever requesting an exclusive write lock; the lock is a separate, dynamic
/// operation on top of an already-open write-capable handle.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AccessMode {
    /// The handle may only be used for read operations. Attempts to write
    /// through this handle return [`FileSystemError::InvalidAccessMode`].
    ReadOnly,

    /// The handle may only be used for write operations. Attempts to read
    /// through this handle return [`FileSystemError::InvalidAccessMode`].
    WriteOnly,

    /// The handle may be used for both read and write operations.
    ReadWrite,
}

/// Selects the base position for a signed seek.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SeekOrigin {
    /// Byte zero, corresponding to POSIX `SEEK_SET`.
    Start,

    /// The open handle's current position, corresponding to POSIX `SEEK_CUR`.
    Current,

    /// The current file length, corresponding to POSIX `SEEK_END`.
    End,
}

/// Selects what a successful open does to an existing regular file's data.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OpenContentsMode {
    /// Preserve the file's current length and contents.
    Preserve,

    /// Set the file length to zero. Requires a write-capable access mode.
    Truncate,
}

/// Controls file contents at open and the write position of the new handle.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct OpenOptions {
    /// Whether to preserve or truncate the contents of an existing file.
    pub contents_mode: OpenContentsMode,

    /// Selects the current end of the file for each write through this handle.
    /// The handle's access rights still determine whether it can write.
    pub append: bool,
}

/// Selects how create-and-open treats an occupied final pathname component.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CreateMode {
    /// Follow final symlinks and open an existing file, or create the missing
    /// final target as a regular file.
    IfMissing,

    /// Require a new final directory entry. Any occupied name, including a
    /// symlink, returns [`FileSystemError::AlreadyExists`].
    Exclusive,
}

impl AccessMode {
    /// Helper function to check if this access mode permits read operations.
    ///
    /// # Returns
    ///
    /// Returns `true` for [`AccessMode::ReadOnly`] and
    /// [`AccessMode::ReadWrite`]; `false` for [`AccessMode::WriteOnly`].
    pub fn can_read(self) -> bool {
        matches!(self, AccessMode::ReadOnly | AccessMode::ReadWrite)
    }

    /// Helper function to check if this access mode permits write operations.
    ///
    /// # Returns
    ///
    /// Returns `true` for [`AccessMode::WriteOnly`] and
    /// [`AccessMode::ReadWrite`]; `false` for [`AccessMode::ReadOnly`].
    pub fn can_write(self) -> bool {
        matches!(self, AccessMode::WriteOnly | AccessMode::ReadWrite)
    }
}

/// Optional namespace restrictions contributed by one registered open handle.
///
/// Ordinary opens request `NONE`. These options do not affect reads, writes,
/// timestamps, or write locks. Prefer `PROTECT_FULL` for namespace stability;
/// the individual options intentionally provide narrower guarantees.
///
/// Separate opens contribute independently. Any contribution blocks conflicting
/// mutations, including requests by its holder. An unprotected open cannot
/// override another handle's protection. Closing a handle releases only its own
/// contributions; cloning its Arc does not acquire another contribution.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NamespaceProtection(u8);

impl NamespaceProtection {
    /// Contributes no namespace restrictions.
    pub const NONE: Self = Self(0);

    /// Rejects unlink, rename, direct move, and replacement of every entry
    /// naming the opened inode. Also rejects truncating opens of the inode
    /// through any name. Additional hard links remain permitted and
    /// immediately share this protection. Ancestors remain independently
    /// mutable.
    pub const PROTECT_NAMES: Self = Self(1);

    /// Protects the resolved file's parent and captured ancestors through root,
    /// including mount hosts. Symlink entries used to reach the target are not
    /// protected. This option alone allows changing the file's own names. If
    /// it moves, the captured protection does not follow it to new ancestors.
    pub const PROTECT_ANCESTORS: Self = Self(2);

    /// Combines name and ancestor protection. All hard-link names are
    /// protected, but only the ancestor chain selected by this open is
    /// captured, not the ancestor chains of every alias. Prefer this
    /// combination when requesting protection unless the narrower behavior is
    /// specifically required.
    pub const PROTECT_FULL: Self =
        Self(Self::PROTECT_NAMES.0 | Self::PROTECT_ANCESTORS.0);

    /// Checks whether all requested restrictions are present.
    ///
    /// # Arguments
    ///
    /// * `requested` - Individual restriction or combination to inspect.
    ///
    /// # Returns
    ///
    /// Returns `true` if every requested bit is present. `NONE` always matches.
    pub const fn contains(self, requested: Self) -> bool {
        self.0 & requested.0 == requested.0
    }
}

impl core::ops::BitOr for NamespaceProtection {
    type Output = Self;

    /// Combines two sets of namespace restrictions without modifying them.
    ///
    /// # Arguments
    ///
    /// * `rhs` - Additional restrictions to include (right-hand-side of `OR`).
    ///
    /// # Returns
    ///
    /// Returns the union of both sets.
    fn bitor(self, rhs: Self) -> Self {
        Self(self.0 | rhs.0)
    }
}

/// Tracks the current write-lock state of an open file's master record.
///
/// Write locking is a dynamic, per-master-record concern, independent of how
/// many handles are open or what [`AccessMode`] they carry. A process that
/// holds a write-capable handle may request an exclusive write lock at any
/// time; while the lock is held, all other write-capable handles on the same
/// file receive [`FileSystemError::FileLocked`] if they attempt a write.
///
/// The VFS supports two states:
///
/// * [`FileLockState::Unlocked`]: writes from any write-capable handle are
///     permitted without coordination.
/// * [`FileLockState::WriteExclusive`]: one handle holds the lock; all other
///     write attempts on this file are rejected until the lock is released or
///     the holding handle is closed.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FileLockState {
    /// No exclusive lock is held; any write-capable handle may write freely.
    Unlocked,

    /// An exclusive write lock is held by the handle identified by
    /// [`holder_handle`]. No other handle may write to this file until the lock
    /// is released.
    ///
    /// The [`holder_handle`] field stores the VFS-wide identifier of the
    /// handle that acquired the lock. Process-local handle values are not
    /// unique across processes and cannot identify a lock holder here.
    WriteExclusive {
        /// The VFS-wide identifier of the lock holder.
        holder_handle: u64,
    },
}

/// Wall-clock timestamps recorded on every inode.
///
/// All three fields store Unix epoch milliseconds, derived from the kernel's
/// `globals::WALL_CLOCK`. Using milliseconds provides sub-second resolution
/// without the complexity of a separate nanosecond field, and the `u64` range
/// comfortably covers all realistic dates.
///
/// Timestamps are set and updated exclusively by VFS-layer code; filesystem
/// drivers receive a populated [`Timestamps`] value when creating or syncing an
/// inode and write it to disk as it is.
///
/// When the `WallClock` is not yet initialized (before the RTC anchor is
/// established during kernel boot), callers should substitute `0` as a
/// sentinel meaning "unknown". The `WallClock::now()` call will panic if
/// invoked before initialization, so callers in early-boot code must guard
/// appropriately.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Timestamps {
    /// Unix epoch milliseconds at which the inode was first created.
    pub created: u64,

    /// Unix epoch milliseconds at which the inode's data was last modified.
    /// Updated on every successful write operation.
    pub modified: u64,

    /// Unix epoch milliseconds at which the inode's data was last read.
    /// Updated on every successful read operation.
    pub accessed: u64,
}

impl Timestamps {
    /// Constructs a [`Timestamps`] value with all three fields set to `now`,
    /// suitable for use when creating a brand-new inode.
    ///
    /// # Arguments
    ///
    /// * `now` - Current Unix epoch milliseconds, obtained from
    ///   `WallClock::now()` via `globals::WALL_CLOCK` and `globals::HPET`.
    ///
    /// # Returns
    ///
    /// Returns a [`Timestamps`] where `created`, `modified`, and `accessed` are
    /// all equal to `now`.
    pub fn new(now: u64) -> Self {
        Self {
            created:  now,
            modified: now,
            accessed: now,
        }
    }

    /// Constructs a zeroed [`Timestamps`] value for use in early-boot contexts
    /// where the wall clock has not yet been initialized.
    ///
    /// All fields are set to `0`, which is treated throughout the VFS as a
    /// sentinel meaning "timestamp unknown". These values should be updated
    /// once the wall clock becomes available.
    ///
    /// # Returns
    ///
    /// Returns a [`Timestamps`] where `created`, `modified`, and `accessed`
    /// are all `0`.
    pub fn zero() -> Self {
        Self {
            created:  0,
            modified: 0,
            accessed: 0,
        }
    }
}

// =============================================================================
// Permissions (TODO: Placeholder)
// =============================================================================

/// Access-control permissions associated with an inode.
///
/// This is a placeholder type. The final implementation will be an ACL-based
/// model supporting per-user and per-group rights beyond the traditional Unix
/// owner/group/other triplet. The placeholder exists so that `Inode`,
/// `InodeStat`, and the filesystem driver traits can reference a concrete
/// `Permissions` type today without committing to its internal representation.
///
/// Until the ACL model is designed and implemented, all permission checks in
/// the VFS should be skipped or treated as universally permissive.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Permissions {
    // Reserved for the future ACL implementation. No fields are exposed until
    // the permission model is fully designed.
    _reserved: u64,
}

impl Permissions {
    /// Constructs a permissive [`Permissions`] value that grants all access.
    ///
    /// Used as a stand-in everywhere a [`Permissions`] value is required before
    /// the real ACL model is in place. All permission checks against this
    /// value must pass unconditionally.
    ///
    /// # Returns
    ///
    /// Returns a [`Permissions`] instance representing unrestricted access.
    pub fn all_permissive() -> Self {
        Self { _reserved: 0 }
    }
}

/// A point-in-time snapshot of an inode's metadata.
///
/// Returned by [`crate::file_system::virtual_fs::vfs::VirtualFileSystem::stat`]
/// rather than a raw `Arc<Inode>` reference, for two reasons: it avoids
/// unnecessarily widening the lifetime of the inode's `Arc`, and it maps
/// directly onto what a future `stat()` syscall will need to copy into a
/// userspace buffer.
///
/// Filesystem drivers do not produce [`InodeStat`] directly; the VFS constructs
/// one from a loaded `Inode` at the point `stat()` is called.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct InodeStat {
    /// The inode's unique identifier within its filesystem instance.
    pub inode_number: u64,

    /// The kind of object this inode represents.
    pub inode_type: InodeType,

    /// The total size of the inode's data in bytes.
    ///
    /// For regular files, this is the file length. For symlinks, this is the
    /// byte length of the target path string. For directories, this field is
    /// filesystem-defined and should not be relied upon by callers.
    pub size: u64,

    /// The access control permissions associated with this inode.
    pub permissions: Permissions,

    /// Creation, modification, and last-access timestamps.
    pub timestamps: Timestamps,

    /// The number of directory entries that reference this inode.
    ///
    /// For regular files and directories this starts at `1` on creation.
    /// At `0`, files survive until their last registered handle closes.
    /// Directories also survive retained directory wrappers and await explicit
    /// VFS reclamation. The filesystem driver frees storage only afterward.
    pub link_count: u32,
}

/// Composite key used to look up a file's master record in the VFS open-file
/// table.
///
/// Inode numbers are unique only within a single filesystem instance. When
/// multiple filesystems are mounted simultaneously, two different inodes on
/// different volumes may share the same inode number. The [`OpenFileKey`]
/// combines a per-instance `filesystem_id` with the `inode_number` to
/// guarantee global uniqueness across all mounted volumes.
///
/// The `filesystem_id` is assigned by the filesystem driver when it creates
/// a mounted instance and remains stable for that instance's lifetime.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct OpenFileKey {
    /// The unique identifier assigned to the mounted filesystem instance that
    /// owns this inode.
    pub filesystem_id: u64,

    /// The inode number within the owning filesystem instance.
    pub inode_number: u64,
}

impl OpenFileKey {
    /// Constructs a new [`OpenFileKey`] from a filesystem instance identifier
    /// and an inode number.
    ///
    /// # Arguments
    ///
    /// * `filesystem_id` - The unique identifier of the mounted filesystem
    ///   instance that owns the inode.
    /// * `inode_number`  - The inode's unique identifier within that filesystem
    ///   instance.
    ///
    /// # Returns
    ///
    /// Returns a new [`OpenFileKey`] combining both identifiers.
    pub fn new(filesystem_id: u64, inode_number: u64) -> Self {
        Self { filesystem_id, inode_number }
    }
}

/// Implements [`crate::kernel_hashmap::KernelHash`] for [`OpenFileKey`], so
/// that it can be used as a key in [`crate::kernel_hashmap::KernelHashMap`].
impl KernelHash for OpenFileKey {
    /// Computes the hash by running the algorithm over `filesystem_id` first
    /// (via its own [`crate::kernel_hashmap::KernelHash`] impl), then
    /// continuing the state over the bytes of `inode_number`. Chaining the two
    /// fields through a single pass (rather than XOR-ing two independent hashes
    /// together) ensures that field order contributes to the result. This way,
    /// `(a, b)` and `(b, a)` produce different hash values, and the risk of
    /// systematic collisions between transposed key pairs is avoided.
    /// 
    /// # Returns
    ///
    /// Returns the computed hash as [`u64`].
    fn kernel_hash(&self) -> u64 {
        const FNV_PRIME: u64 = 0x00000100000001b3;
        let mut hash = self.filesystem_id.kernel_hash();

        for &byte in &self.inode_number.to_ne_bytes() {
            hash ^= byte as u64;
            hash = hash.wrapping_mul(FNV_PRIME);
        }

        hash
    }
}
