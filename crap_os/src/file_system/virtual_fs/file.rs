//! Virtual File System - Open File State
//!
//! This module defines the two-tier open-file model used by the VFS:
//!
//! - [`File`] is the master record for an open file. There is exactly one
//!   `File` per unique inode that has at least one open handle anywhere in
//!   the system. It is owned by the object manager and persists from the
//!   moment the first handle to a given inode is opened until the last handle
//!   is closed. It tracks all live handles, enforces the write-lock state, and
//!   carries the `pending_deletion` flag used to implement POSIX unlink
//!   semantics.
//!
//! - [`FileHandle`] is the per-handle instance created for each individual
//!   `open()` call. It is owned by the calling process's handle table entry.
//!   It carries the per-handle state that is independent between handles to
//!   the same file: the current read/write position, and the access mode
//!   granted at open time. It holds a strong [`Arc<File>`] reference back to
//!   the master record, keeping the master record alive for at least as long
//!   as the handle exists.
//!
//! The object manager holds the sole [`Arc<File>`] that owns each master
//! record. Process handle tables hold [`Arc<FileHandle>`] values. The [`File`]
//! master record holds [`Weak<FileHandle>`] references to track active handles
//! without preventing their cleanup.
//!
//! When a process closes a handle or is terminated by the process manager,
//! the [`Arc<FileHandle>`] is dropped. When the last [`Arc<FileHandle>`] for a
//! given [`File`] is dropped, the object manager removes the [`Arc<File>`] from
//! its open-file table, dropping the master record. If `pending_deletion` is
//! set on the master record at that point, the VFS instructs the filesystem
//! driver to free the inode's data blocks.
//!
//! Write locking is a dynamic operation on top of an already-open
//! write-capable handle. Holding a `WriteOnly` or `ReadWrite` handle does not
//! automatically grant exclusive access; the process must explicitly request a
//! write lock. While a `WriteExclusive` lock is held, all other write attempts
//! on the same [`File`] return `FileSystemError::FileLocked`. The lock is
//! released when the holding handle is closed or when the process explicitly
//! releases it.

#![allow(dead_code)]

use alloc::sync::{Arc, Weak};
use alloc::vec::Vec;

use super::inode::Inode;
use super::types::{AccessMode, FileLockState, FileSystemError};

/// The master record for a file that has at least one open handle.
///
/// There is exactly one [`File`] per unique inode that is currently open
/// anywhere in the system. It is created the first time any process opens the
/// inode and destroyed when the last handle to it is closed.
///
/// The [`File`] does not directly track per-handle state such as read position
/// or access mode - those live on [`FileHandle`]. It tracks the things that are
/// shared across all handles to the same inode: the inode itself, the current
/// write-lock state, the set of live handles (as `Weak` references), and
/// whether the inode is pending deletion after all handles close.
///
/// Owned exclusively by the object manager's open-file table, keyed by
/// [`super::types::OpenFileKey`].
pub struct File {
    /// The inode this master record is open for.
    ///
    /// Shared with every [`FileHandle`] that references this [`File`]. The
    /// [`Arc`] keeps the inode live in memory for the duration of the open
    /// session, independently of whether the inode still has directory entries
    /// pointing to it.
    pub inode: Arc<Inode>,

    /// The current write-lock state of this file.
    ///
    /// `Unlocked` by default. A process holding a write-capable [`FileHandle`]
    /// may request an exclusive write lock, transitioning this state to
    /// `WriteExclusive`. While locked, all other write attempts through any
    /// handle on this [`File`] return `FileSystemError::FileLocked`. The lock
    /// is released when the holding handle is explicitly unlocked or closed.
    pub lock_state: FileLockState,

    /// Weak references to all live [`FileHandle`] instances for this file,
    /// across all processes.
    ///
    /// Weak references are used so that handle cleanup is driven by the
    /// process manager dropping [`Arc<FileHandle>`] values from process handle
    /// tables, not by the master record keeping handles alive. Stale [`Weak`]
    /// entries (where `upgrade()` returns `None`) accumulate only if a handle
    /// is closed without the VFS being notified, which must not happen under
    /// normal operation, as the process manager guarantees explicit cleanup on
    /// both normal and abnormal process exit.
    ///
    /// This list is used to count active handles (for determining when the
    /// master record should be destroyed) and to enforce write-lock semantics
    /// (for verifying that the lock holder's handle is still live before
    /// honouring a lock-release request).
    pub handles: Vec<Weak<FileHandle>>,

    /// Indicates that all directory entries referencing the underlying inode
    /// have been removed (via `VirtualFileSystem::delete`), but at least one
    /// open handle still holds an [`Arc<Inode>`] reference.
    ///
    /// While `true`, no new handles can be opened to this inode by name (it
    /// has no directory entries), but existing handles may continue to read
    /// and write it. When the last [`FileHandle`] is dropped, and this flag is
    /// `true`, the object manager calls the filesystem driver to free the
    /// inode's data blocks, completing the deletion. This implements POSIX
    /// unlink semantics.
    pub pending_deletion: bool,
}

impl File {
    /// Constructs a new `File` master record for the given inode.
    ///
    /// The initial lock state is `Unlocked`, the handle list is empty, and
    /// `pending_deletion` is `false`. The first [`FileHandle`] should be
    /// registered via `register_handle` immediately after construction.
    ///
    /// # Arguments
    ///
    /// * `inode` - The inode this master record represents.
    pub fn new(inode: Arc<Inode>) -> Self {
        Self {
            inode,
            lock_state:       FileLockState::Unlocked,
            handles:          Vec::new(),
            pending_deletion: false,
        }
    }

    /// Registers a new [`FileHandle`] with this master record.
    ///
    /// Appends a weak reference to `handle` to the `handles` list. Must be
    /// called each time a new [`Arc<FileHandle>`] is created for this file, so
    /// that the master record can track the total number of live handles and
    /// enforce write-lock semantics across all of them.
    ///
    /// # Arguments
    ///
    /// * `handle` - The newly created [`FileHandle`] to register.
    pub fn register_handle(&mut self, handle: &Arc<FileHandle>) {
        self.handles.push(Arc::downgrade(handle));
    }

    /// Removes any stale weak handle references from the `handles` list and
    /// returns the count of handles that are still live.
    ///
    /// A stale entry is one where [`Weak::upgrade()`] returns [`None`], meaning
    /// the corresponding [`Arc<FileHandle>`] has been dropped. Under normal
    /// operation, the process manager closes handles explicitly before they are
    /// dropped, but this method provides a reliable way to recount live handles
    /// without assuming the list is perfectly clean.
    ///
    /// # Returns
    ///
    /// Returns the number of live handles remaining after pruning stale
    /// entries.
    pub fn prune_and_count_handles(&mut self) -> usize {
        self.handles.retain(|weak| weak.upgrade().is_some());
        self.handles.len()
    }

    /// Gets the count of handles in the `handles` list.
    ///
    /// # Returns
    ///
    /// Returns the number of live handles in the list.
    pub fn count_handles(&mut self) -> usize {
        self.handles.len()
    }

    /// Checks if the given handle value currently holds the exclusive
    /// write lock on this file.
    ///
    /// # Arguments
    ///
    /// * `handle_value` - The process-local handle value to check against the
    ///   current lock holder.
    ///
    /// # Returns
    ///
    /// Returns `true` if `lock_state` is `WriteExclusive` and `holder_handle`
    /// equals `handle_value`, and `false` otherwise.
    pub fn is_lock_holder(&self, handle_value: u32) -> bool {
        matches!(
            self.lock_state,
            FileLockState::WriteExclusive { holder_handle }
            if holder_handle == handle_value
        )
    }

    /// Attempts to acquire an exclusive write lock on behalf of the handle
    /// identified by `handle_value`.
    ///
    /// Succeeds only when the file is currently `Unlocked`. If another handle
    /// already holds the lock, returns `FileSystemError::LockContention`
    /// without modifying the lock state.
    ///
    /// # Arguments
    ///
    /// * `handle_value` - The process-local handle value of the handle
    ///   requesting the lock. Stored in `FileLockState::WriteExclusive` so
    ///   that future lock-release requests can be validated against the
    ///   original holder.
    ///
    /// # Returns
    ///
    /// Returns `Ok(())` if the lock was successfully acquired, or
    /// `Err(FileSystemError::LockContention)` if the file is already locked
    /// by another handle.
    pub fn acquire_write_lock(
        &mut self,
        handle_value: u32,
    ) -> Result<(), FileSystemError> {
        match self.lock_state {
            FileLockState::Unlocked => {
                self.lock_state = FileLockState::WriteExclusive {
                    holder_handle: handle_value,
                };
                Ok(())
            }
            FileLockState::WriteExclusive { .. } => {
                Err(FileSystemError::LockContention)
            }
        }
    }

    /// Releases the exclusive write lock, provided the caller is the current
    /// lock holder.
    ///
    /// If the file is not locked, or the lock is held by a different handle
    /// than `handle_value`, the lock state is not modified, and an error is
    /// returned.
    ///
    /// # Arguments
    ///
    /// * `handle_value` - The process-local handle value of the handle
    ///   attempting to release the lock. Must match the `holder_handle`
    ///   recorded when the lock was acquired.
    ///
    /// # Returns
    ///
    /// Returns `Ok(())` if the lock was successfully released, or
    /// `Err(FileSystemError::InvalidHandle)` if the file is not locked or
    /// `handle_value` does not match the current lock holder.
    pub fn release_write_lock(
        &mut self,
        handle_value: u32,
    ) -> Result<(), FileSystemError> {
        if self.is_lock_holder(handle_value) {
            self.lock_state = FileLockState::Unlocked;
            return Ok(());
        }

        Err(FileSystemError::InvalidHandle)
    }

    /// Checks if a write operation through the given handle should be
    /// blocked by the current lock state.
    ///
    /// A write is blocked when the file is `WriteExclusive` and `handle_value`
    /// is not the lock holder. A write is always permitted when the file is
    /// `Unlocked`, or when `handle_value` is the lock holder.
    ///
    /// # Arguments
    ///
    /// * `handle_value` - The process-local handle value of the handle
    ///   attempting to write.
    ///
    /// # Returns
    ///
    /// Returns `true` if the write should be rejected with
    /// `FileSystemError::FileLocked`, and `false` if the write is permitted
    /// by the current lock state.
    pub fn is_write_blocked(&self, handle_value: u32) -> bool {
        matches!(
            self.lock_state,
            FileLockState::WriteExclusive { holder_handle }
            if holder_handle != handle_value
        )
    }
}

/// A per-handle instance representing a single `open()` call on a file.
///
/// Each time a process opens a file, the VFS creates a [`FileHandle`], and the
/// object manager registers it in the process's handle table. Multiple
/// processes, or the same process multiple times, may have independent
/// [`FileHandle`] instances for the same underlying inode; each carries its own
/// position and access mode.
///
/// The [`FileHandle`] holds a strong [`Arc<File>`] reference to the master
/// record, ensuring the master record stays alive for at least as long as the
/// handle exists. The master record holds a corresponding [`Weak<FileHandle>`]
/// back to this handle for tracking purposes.
///
/// Owned by the process's handle table entry. Dropped when the process closes
/// the handle explicitly or when the process manager tears down the process's
/// handle table on exit.
pub struct FileHandle {
    /// The master record for the file this handle is open on.
    ///
    /// Shared with all other handles to the same inode. Provides access to
    /// the inode, the write-lock state, and the pending-deletion flag.
    pub file: Arc<File>,

    /// The current byte offset for the next read or write operation.
    ///
    /// Starts at `0` when the handle is created. Advanced automatically after
    /// each successful read or write by the number of bytes transferred.
    /// May also be set explicitly via `VirtualFileSystem::seek`. Independent
    /// of the position of any other handle to the same file.
    pub position: u64,

    /// The access rights granted to this handle at open time.
    ///
    /// Fixed at creation and cannot be upgraded. The VFS checks this before
    /// every read and write operation: a `ReadOnly` handle may not write, and
    /// a `WriteOnly` handle may not read. Write-lock operations are only valid
    /// on handles whose `access_mode` permits writing.
    pub access_mode: AccessMode,
}

impl FileHandle {
    /// Constructs a new [`FileHandle`] for the given master record with the
    /// specified access mode.
    ///
    /// The initial position is `0`. The caller is responsible for registering
    /// this handle with the master record via [`File::register_handle`]
    /// immediately after wrapping it in an [`Arc`].
    ///
    /// # Arguments
    ///
    /// * `file`        - The master record for the file being opened.
    /// * `access_mode` - The access rights granted to this handle.
    pub fn new(file: Arc<File>, access_mode: AccessMode) -> Self {
        Self {
            file,
            position: 0,
            access_mode,
        }
    }

    /// Checks if this handle permits read operations.
    ///
    /// # Returns
    ///
    /// Returns `true` when `access_mode` is `ReadOnly` or `ReadWrite`, and
    /// `false` when it is `WriteOnly`.
    pub fn can_read(&self) -> bool {
        self.access_mode.can_read()
    }

    /// Checks if this handle permits write operations.
    ///
    /// # Returns
    ///
    /// Returns `true` when `access_mode` is `WriteOnly` or `ReadWrite`, and
    /// `false` when it is `ReadOnly`.
    pub fn can_write(&self) -> bool {
        self.access_mode.can_write()
    }

    /// Advances the handle's position by `bytes_transferred` bytes.
    ///
    /// Called internally by the VFS after each successful read or write to
    /// keep the position consistent with the amount of data transferred.
    /// Saturates at [`u64::MAX`] rather than wrapping, which is sufficient
    /// given that no real file can be that large.
    ///
    /// # Arguments
    ///
    /// * `bytes_transferred` - The number of bytes successfully read or
    ///   written in the most recent operation.
    pub fn advance_position(&mut self, bytes_transferred: usize) {
        self.position = self.position.saturating_add(bytes_transferred as u64);
    }

    /// Sets the handle's read/write position to an absolute byte offset.
    ///
    /// Does not validate the offset against the file's current size; the VFS
    /// `seek` implementation is responsible for any bounds checking required
    /// by the calling context.
    ///
    /// # Arguments
    ///
    /// * `offset` - The absolute byte offset to seek to.
    pub fn seek_to(&mut self, offset: u64) {
        self.position = offset;
    }
}
