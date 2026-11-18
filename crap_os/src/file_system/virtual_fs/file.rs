//! Virtual File System - Open File State
//!
//! This module defines the two-tier open-file model used by the VFS:
//!
//! - [`File`] is the master record for an open file. There is exactly one
//!   [`File`] per unique inode that has at least one open handle anywhere in
//!   the system. Its table entry persists from the first open until the last
//!   close. Retained handle references can keep the record allocated after
//!   removal from the table. It tracks registered handles, write-lock state,
//!   and whether the inode's last directory entry has been removed.
//!
//! - [`FileHandle`] is the per-handle instance created for each individual
//!   `open()` call. Its caller owns the returned `Arc`; a future process
//!   handle table can hold that reference. It carries the per-handle state that
//!   is independent between handles to the same file: the current read/write
//!   position, and the access mode granted at open time. It holds a strong
//!   [`Arc<File>`] reference back to the master record, keeping the master
//!   record alive for at least as long as the handle exists.
//!
//! The VFS open-file table holds an [`Arc<File>`] for each master record.
//! Callers hold [`Arc<FileHandle>`] values. The [`File`] master record holds
//! [`Weak<FileHandle>`] references to track active handles without preventing
//! their cleanup.
//!
//! The VFS registers each handle before returning from `open`. A successful
//! `close` removes that exact registration and invalidates all `Arc` clones of
//! the handle. When the last registration is removed, the VFS removes the
//! master record from its open-file table. If the master is pending deletion,
//! the VFS first asks the driver to free its storage; a driver failure leaves
//! the registration intact for a retry. Handle owners must call `close`
//! explicitly before dropping their last `Arc` reference.
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

use crate::spinlock::SpinLock;

use super::directory::DirectoryPathGuard;
use super::inode::{Inode, InodeNameGuard};
use super::types::{
    AccessMode, FileLockState, FileSystemError, NamespaceProtection
};

/// The master record for a file that has at least one open handle.
///
/// There is exactly one [`File`] per unique inode that is currently open
/// anywhere in the system. It is created the first time any process opens the
/// inode and removed from the VFS table when the last handle closes. Retained
/// handle references may keep the record allocated after table removal.
///
/// The [`File`] does not directly track per-handle state such as read position
/// or access mode. It holds the inode and protects the write-lock state and
/// registered handle list and deferred-deletion state shared by all opens of
/// that inode.
///
/// Owned by the VFS open-file table, keyed by [`super::types::OpenFileKey`].
pub struct File {
    /// The inode this master record is open for.
    ///
    /// Shared with every [`FileHandle`] that references this [`File`]. The
    /// [`Arc`] keeps the inode live in memory for the duration of the open
    /// session, independent of whether the inode still has directory entries
    /// pointing to it.
    pub inode: Arc<Inode>,

    /// Mutable book-keeping shared by all handles. VFS callers hold the global
    /// IRQ-safe lock before taking this short inner lock.
    state: SpinLock<FileState>,
}

/// Mutable book-keeping for the handles open on one inode.
struct FileState {
    /// Exclusive write-lock owner, identified by a VFS-wide handle ID.
    lock_state: FileLockState,

    /// Weak registrations for handles returned by successful VFS opens.
    handles: Vec<Weak<FileHandle>>,

    /// Set after the last directory entry is removed, until the last close
    /// successfully frees the inode through the filesystem driver.
    pending_deletion: bool,
}

impl File {
    /// Constructs a new [`File`] master record for the given inode.
    ///
    /// The initial write-lock state is `Unlocked`, the handle list is empty,
    /// and deletion is not pending. The VFS registers the first handle before
    /// returning from `open`.
    ///
    /// # Arguments
    ///
    /// * `inode` - The inode this master record represents.
    pub fn new(inode: Arc<Inode>) -> Self {
        Self {
            inode,
            state: SpinLock::new(FileState {
                lock_state: FileLockState::Unlocked,
                handles: Vec::new(),
                pending_deletion: false,
            }),
        }
    }

    /// Registers a new [`FileHandle`] with this master record.
    ///
    /// Appends a weak reference before the VFS returns a newly opened handle.
    ///
    /// # Arguments
    ///
    /// * `handle` - The newly created [`FileHandle`] to register.
    pub(super) fn register_handle(&self, handle: &Arc<FileHandle>) {
        self.state.lock().handles.push(Arc::downgrade(handle));
    }

    /// Checks whether the exact handle is registered with this master record.
    ///
    /// # Arguments
    ///
    /// * `handle` - The handle whose allocation identity is checked.
    ///
    /// # Returns
    ///
    /// Returns `true` if this exact handle has an entry in the master record,
    /// and `false` otherwise.
    pub(super) fn has_handle(&self, handle: &Arc<FileHandle>) -> bool {
        let ptr = Arc::as_ptr(handle);
        self.state.lock().handles.iter().any(|weak| weak.as_ptr() == ptr)
    }

    /// Counts the handles registered with this master record.
    ///
    /// # Returns
    ///
    /// Returns the number of registrations that have not been closed.
    pub(super) fn registered_handle_count(&self) -> usize {
        self.state.lock().handles.len()
    }

    /// Marks this open-file master for reclamation on its last close.
    ///
    /// Called after the filesystem driver removes the inode's last directory
    /// entry. The VFS serializes this operation with open and close.
    pub(super) fn mark_pending_deletion(&self) {
        self.state.lock().pending_deletion = true;
    }

    /// Checks whether this master must free its inode on the last close.
    ///
    /// # Returns
    ///
    /// Returns `true` when the inode has no directory entries and its storage
    /// awaits reclamation, `false` otherwise.
    pub(super) fn is_pending_deletion(&self) -> bool {
        self.state.lock().pending_deletion
    }

    /// Removes one registered handle and releases its write lock, if held.
    ///
    /// # Arguments
    ///
    /// * `handle` - The exact handle to unregister after a successful close.
    ///
    /// # Returns
    ///
    /// Returns the remaining registration count, or `InvalidHandle` if the
    /// handle was not registered.
    pub(super) fn unregister_handle(
        &self,
        handle: &Arc<FileHandle>,
    ) -> Result<usize, FileSystemError> {
        let mut state = self.state.lock();
        let ptr = Arc::as_ptr(handle);
        let index = state.handles.iter()
            .position(|weak| weak.as_ptr() == ptr)
            .ok_or(FileSystemError::InvalidHandle)?;
        state.handles.swap_remove(index);

        if matches!(
            &state.lock_state,
            FileLockState::WriteExclusive { holder_handle }
                if *holder_handle == handle.id
        ) {
            state.lock_state = FileLockState::Unlocked;
        }

        Ok(state.handles.len())
    }

    /// Checks whether a VFS-wide handle ID owns the exclusive write lock.
    ///
    /// # Arguments
    ///
    /// * `handle_id` - The VFS-wide identifier to check.
    ///
    /// # Returns
    ///
    /// Returns `true` if this ID owns the lock, and `false` otherwise.
    pub(super) fn is_lock_holder(&self, handle_id: u64) -> bool {
        let state = self.state.lock();
        matches!(
            &state.lock_state,
            FileLockState::WriteExclusive { holder_handle }
            if *holder_handle == handle_id
        )
    }

    /// Attempts to acquire an exclusive write lock on behalf of the handle
    /// identified by `handle_id`.
    ///
    /// Succeeds only when the file is currently `Unlocked`. If another handle
    /// already holds the lock, returns `FileSystemError::LockContention`
    /// without modifying the lock state.
    ///
    /// # Arguments
    ///
    /// * `handle_id` - The VFS-wide identifier of the requesting handle.
    ///
    /// # Returns
    ///
    /// Returns `Ok(())` if the lock was successfully acquired, or
    /// `Err(FileSystemError::LockContention)` if the file is already locked
    /// by another handle.
    pub(super) fn acquire_write_lock(
        &self,
        handle_id: u64,
    ) -> Result<(), FileSystemError> {
        let mut state = self.state.lock();
        if state.lock_state != FileLockState::Unlocked {
            return Err(FileSystemError::LockContention);
        }

        state.lock_state = FileLockState::WriteExclusive {
            holder_handle: handle_id,
        };

        Ok(())
    }

    /// Releases the exclusive write lock, provided the caller is the current
    /// lock holder.
    ///
    /// If the file is not locked, or the lock is held by a different handle
    /// than `handle_id`, the lock state is not modified, and an error is
    /// returned.
    ///
    /// # Arguments
    ///
    /// * `handle_id` - The VFS-wide identifier of the releasing handle.
    ///
    /// # Returns
    ///
    /// Returns `Ok(())` if the lock was successfully released, or
    /// `Err(FileSystemError::InvalidHandle)` if the file is not locked or
    /// `handle_id` does not match the current lock holder.
    pub(super) fn release_write_lock(
        &self,
        handle_id: u64,
    ) -> Result<(), FileSystemError> {
        let mut state = self.state.lock();
        if matches!(
            &state.lock_state,
            FileLockState::WriteExclusive { holder_handle }
                if *holder_handle == handle_id
        ) {
            state.lock_state = FileLockState::Unlocked;
            return Ok(());
        }

        Err(FileSystemError::InvalidHandle)
    }

    /// Checks if a write operation through the given handle should be
    /// blocked by the current lock state.
    ///
    /// A write is blocked when the file is `WriteExclusive` and `handle_id`
    /// is not the lock holder. A write is always permitted when the file is
    /// `Unlocked`, or when `handle_id` is the lock holder.
    ///
    /// # Arguments
    ///
    /// * `handle_id` - The VFS-wide identifier of the writing handle.
    ///
    /// # Returns
    ///
    /// Returns `true` if the write should be rejected with
    /// `FileSystemError::FileLocked`, and `false` if the write is permitted
    /// by the current lock state.
    pub(super) fn is_write_blocked(&self, handle_id: u64) -> bool {
        let state = self.state.lock();
        matches!(
            &state.lock_state,
            FileLockState::WriteExclusive { holder_handle }
            if *holder_handle != handle_id
        )
    }
}

/// A per-handle instance representing a single `open()` call on a file.
///
/// Each VFS open creates a [`FileHandle`] and returns its registered `Arc`
/// to the caller. Future process handle tables can own these references.
/// Multiple processes, or the same process multiple times, may have independent
/// [`FileHandle`] instances for the same underlying inode; each carries its own
/// position and access mode.
///
/// The [`FileHandle`] holds a strong [`Arc<File>`] reference to the master
/// record. The master record holds a corresponding [`Weak<FileHandle>`] until
/// VFS `close` unregisters this open handle. Clones of its `Arc` refer to the
/// same open handle and become invalid together when it is closed.
///
/// Owned by the caller's `Arc` references. The caller must explicitly close
/// the handle through the VFS before dropping its last reference.
pub struct FileHandle {
    /// The master record for the file this handle is open on.
    ///
    /// Shared with all other handles to the same inode. Provides access to
    /// the inode, shared write-lock state, and deferred-deletion state.
    pub file: Arc<File>,

    /// Identifier unique among VFS opens, including opens by other processes.
    pub(super) id: u64,

    /// Position and closed state shared by all `Arc` clones of this handle.
    ///
    /// The lock remains held through a synchronous driver transfer and its
    /// position update. VFS callers hold the global IRQ-safe lock first.
    pub(super) state: SpinLock<FileHandleState>,

    /// The access rights granted to this handle at open time.
    ///
    /// Fixed at creation and cannot be upgraded. The VFS checks this before
    /// every read and write operation: a `ReadOnly` handle may not write, and
    /// a `WriteOnly` handle may not read. Write-lock operations are only valid
    /// on handles whose `access_mode` permits writing.
    pub access_mode: AccessMode,

    /// Fixed at open. Each write selects the current inode size when set.
    pub(super) append: bool,
}

/// Namespace contributions owned by one open-handle registration.
///
/// Successful VFS close releases these tokens even when Arc clones remain.
/// Failed close retains them. Abandoning an incompletely constructed open
/// rolls back acquired protection through ordinary token destruction.
pub(super) struct OpenNamespaceProtection {
    /// Immutable options selected when this handle was opened.
    pub(super) flags: NamespaceProtection,

    /// One contribution protecting all names of the opened inode, if requested.
    pub(super) name_guard: Option<InodeNameGuard>,

    /// Captured parent and ancestor protection, if requested.
    pub(super) ancestor_guard: Option<DirectoryPathGuard>,
}

/// State shared by references to one open handle, not by separate opens.
pub(super) struct FileHandleState {
    /// Byte offset for the next read or write through this handle.
    pub(super) position: u64,

    /// Set after successful close so every remaining `Arc` clone is invalid.
    pub(super) closed: bool,

    /// Contributions owned by this registration until successful close.
    /// Unprotected opens have a value with both tokens absent. Closed handles
    /// contain `None`; retained Arc clones do not extend released protection.
    pub(super) namespace_protection: Option<OpenNamespaceProtection>,
}

impl FileHandle {
    /// Constructs a new [`FileHandle`] for the given master record with the
    /// specified access mode and prepared namespace protection.
    ///
    /// The initial position is `0`, and the handle starts open. The VFS wraps
    /// it in an [`Arc`] and registers it before returning from `open`.
    ///
    /// # Arguments
    ///
    /// * `file`                 - The master record for the file being opened.
    /// * `id`                   - Identifier assigned by the VFS to this open
    ///   handle.
    /// * `access_mode`          - The access rights granted to this handle.
    /// * `append`               - Whether each write selects the file end.
    /// * `namespace_protection` - Tokens acquired under VFS serialization.
    ///   Abandoning construction releases them without driver I/O.
    ///
    /// # Returns
    ///
    /// Returns an open handle that must be registered before publication.
    pub(super) fn new(
        file: Arc<File>,
        id: u64,
        access_mode: AccessMode,
        append: bool,
        namespace_protection: OpenNamespaceProtection,
    ) -> Self {
        Self {
            file,
            id,
            state: SpinLock::new(FileHandleState {
                position: 0,
                closed: false,
                namespace_protection: Some(namespace_protection),
            }),
            access_mode,
            append,
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

    /// Sets the handle's read/write position to an absolute byte offset.
    ///
    /// The state lock serializes this update with reads, writes, and close on
    /// the same handle. Seeking beyond the file's current size is permitted.
    ///
    /// # Arguments
    ///
    /// * `offset` - The absolute byte offset to seek to.
    ///
    /// # Returns
    ///
    /// Returns `Ok(())` after setting the position, or `InvalidHandle` if
    /// this handle has been closed.
    pub(super) fn seek_to(&self, offset: u64) -> Result<(), FileSystemError> {
        let mut state = self.state.lock();
        if state.closed {
            return Err(FileSystemError::InvalidHandle);
        }

        state.position = offset;

        Ok(())
    }
}
