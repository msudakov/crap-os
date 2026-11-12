//! # Virtual File System - File Operations
//!
//! This module implements the file and namespace operation methods on
//! [`super::vfs::VirtualFileSystem`], including opening files, transferring
//! data, querying metadata, and creating, removing, or renaming directory
//! entries.
//!
//! These methods coordinate VFS state with filesystem driver operations.
//! Path-based methods resolve the object or its parent directory; handle-based
//! methods use the registered open file. Each operation checks the applicable
//! type, access mode, and lock state, calls the driver when needed, and updates
//! the VFS state to reflect the result.
//!
//! ## Caller Responsibilities
//!
//! These methods operate at the kernel level and return kernel-side objects.
//! Callers are responsible for:
//!
//! - Retaining each VFS-registered [`super::file::FileHandle`] in an owning
//!   structure, such as a process handle table or kernel subsystem.
//! - Closing each handle through the VFS before dropping its last reference.
//!   A failed close leaves the handle open; its owner must retain it for retry.
//!
//! A future syscall layer is additionally responsible for:
//!
//! - Translating kernel-side [`super::types::FileSystemError`] values into
//!   appropriate userspace error codes.
//! - Copying [`super::types::InodeStat`] snapshots into userspace buffers
//!   for `stat`-family syscalls.
//!
//! ## Open File Table Lifecycle
//!
//! The open-file table in [`super::vfs::VirtualFileSystem`] holds one
//! [`super::file::File`] master record per inode that has at least one open
//! handle. [`super::vfs::VirtualFileSystem::open`] and
//! [`super::vfs::VirtualFileSystem::open_or_create`] insert a master record on
//! the first open and register every returned handle with its record.
//! [`super::vfs::VirtualFileSystem::close`] unregisters the exact handle and
//! removes the master record when no registrations remain, reclaiming the
//! inode first if the master is pending deletion.

#![allow(dead_code)]

use alloc::sync::Arc;
use alloc::string::ToString;
use core::sync::atomic::Ordering;

use super::directory::Directory;
use super::file::{File, FileHandle, OpenNamespaceProtection};
use super::inode::Inode;
use super::types::{
    AccessMode, CreateMode, DirectoryEntryType, FileSystemError, InodeStat,
    NamespaceProtection, OpenContentsMode, OpenFileKey, Permissions, SeekOrigin,
    Timestamps,
};
use super::vfs::VirtualFileSystem;
use crate::globals;

impl VirtualFileSystem {
    /// Opens a regular file without contributing namespace restrictions.
    ///
    /// Other handles' protection still applies. The caller holds global VFS
    /// serialization and explicitly closes the returned registration.
    ///
    /// # Arguments
    ///
    /// * `path`              - File pathname; final symlinks are followed.
    /// * `access_mode`       - Content access rights requested.
    /// * `working_directory` - Relative-path base, or root when absent.
    ///
    /// # Returns
    ///
    /// Returns a registered handle or the underlying open/resolution error.
    pub fn open(
        &mut self,
        path: &str,
        access_mode: AccessMode,
        working_directory: Option<Arc<Directory>>,
    ) -> Result<Arc<FileHandle>, FileSystemError> {
        self.open_with_namespace_protection(
            path, access_mode, working_directory, NamespaceProtection::NONE,
        )
    }

    /// Opens a regular file and atomically acquires namespace protection.
    ///
    /// The caller holds global VFS serialization throughout resolution, token
    /// acquisition, and registration. No namespace mutation can interleave.
    /// Prefer `PROTECT_FULL` for namespace stability. Ancestor-only protection
    /// stays on captured directories if the file later moves. All hard-link
    /// names share name protection, but only this resolved path's ancestors
    /// are captured. Symlink entries used to reach the target are not guarded.
    ///
    /// Reads, writes, timestamps, and write locking keep their own contracts.
    /// Protection does not prevent additional opens. The caller must close
    /// the registration before dropping its last Arc; a failed close retains
    /// protection for retry. Arc clones do not acquire independent tokens.
    ///
    /// # Arguments
    ///
    /// * `path`              - File pathname; final symlinks are followed.
    /// * `access_mode`       - Content access rights requested.
    /// * `working_directory` - Relative-path base, or root when absent.
    /// * `protection`        - Independent namespace restrictions, or `NONE`.
    ///
    /// # Returns
    ///
    /// Returns a registered handle owning the contributions. Returns `NotAFile`
    /// for other inode types, `NotFound` for a detached target, `ResourceBusy`
    /// for protection-counter exhaustion, `InternalError` for exhausted handle
    /// IDs or missing regular-file parent context, or a resolution error.
    /// A returned error publishes no handle and retains no new protection.
    pub fn open_with_namespace_protection(
        &mut self,
        path: &str,
        access_mode: AccessMode,
        working_directory: Option<Arc<Directory>>,
        protection: NamespaceProtection,
    ) -> Result<Arc<FileHandle>, FileSystemError> {
        self.open_with_options(
            path, access_mode, OpenContentsMode::Preserve, working_directory,
            protection,
        )
    }

    /// Opens an existing regular file with the requested contents and
    /// protection modes.
    ///
    /// The caller holds global VFS serialization and closes the returned handle
    /// through the VFS. On truncation, all names and handles observe the same
    /// empty inode. Existing handle offsets are preserved. The new handle
    /// starts at byte zero. Existing inode name protection blocks truncation
    /// through any alias. Protection requested by this open applies after
    /// preflight.
    ///
    /// # Arguments
    ///
    /// * `path`               - File pathname. Final symlinks are followed.
    /// * `access_mode`        - Content access rights requested for the handle.
    /// * `open_contents_mode` - Whether to preserve or truncate the existing
    ///   file.
    /// * `working_directory`  - Relative-path base, or root when absent.
    /// * `protection`         - Namespace contributions to own for this open.
    ///
    /// # Returns
    ///
    /// Returns a registered handle on success. Resolution, protection, access,
    /// write-lock, or driver errors preserve file contents and publish no
    /// handle. Existing name protection returns `ResourceBusy`. Any newly
    /// acquired protection is released on error.
    pub fn open_with_options(
        &mut self,
        path: &str,
        access_mode: AccessMode,
        open_contents_mode: OpenContentsMode,
        working_directory: Option<Arc<Directory>>,
        protection: NamespaceProtection,
    ) -> Result<Arc<FileHandle>, FileSystemError> {
        if open_contents_mode == OpenContentsMode::Truncate
            && !access_mode.can_write()
        {
            return Err(FileSystemError::InvalidAccessMode);
        }

        let resolved = self.resolve_entry_with_budget(
            path, working_directory, true, super::resolver::MAX_SYMLINK_DEPTH,
        )?;
        let inode = resolved.inode;

        if !inode.is_file() {
            return Err(FileSystemError::NotAFile);
        }

        let key = OpenFileKey::new(inode.filesystem.instance_id(),
            inode.inode_number);
        if inode.metadata.lock().link_count == 0
            || self.open_files.get(&key)
                .is_some_and(|master| master.is_pending_deletion())
        {
            return Err(FileSystemError::NotFound);
        }

        // Validate ID availability before acquiring tokens. Commit advancement
        // only when this open is ready to register.
        let next_handle_id = self.next_handle_id.checked_add(1)
            .ok_or(FileSystemError::InternalError)?;

        // Only protection from existing opens applies to this truncation.
        // Finish fallible token acquisition before committing the change.
        if open_contents_mode == OpenContentsMode::Truncate {
            self.ensure_open_truncation_allowed(&inode, &key)?;
        }

        let ancestor_guard = if protection.contains(
            NamespaceProtection::PROTECT_ANCESTORS)
        {
            let parent = resolved.file_parent
                .ok_or(FileSystemError::InternalError)?;
            Some(self.protect_directory_path(parent)?)
        } else {
            None
        };

        // A failed name acquisition drops the local ancestor token, rolling
        // back its contribution without a separate error-prone undo sequence.
        let name_guard = if protection.contains(
            NamespaceProtection::PROTECT_NAMES)
        {
            Some(inode.protect_names()?)
        } else {
            None
        };

        let namespace_protection = OpenNamespaceProtection {
            flags: protection, name_guard, ancestor_guard,
        };

        if open_contents_mode == OpenContentsMode::Truncate {
            Self::truncate_for_open(&inode)?;
        }

        Ok(self.register_open_handle(
            inode, access_mode, namespace_protection, next_handle_id,
        ))
    }

    /// Opens a regular file, creating its final target when absent.
    ///
    /// The caller holds global VFS serialization through traversal, creation,
    /// and handle registration. Parent symlinks and followed final symlinks
    /// consume one shared traversal budget. Existing contents are preserved.
    /// Creation permissions apply only when a new file is created. The handle
    /// contributes no namespace protection and must be closed through the VFS.
    ///
    /// # Arguments
    ///
    /// * `path`              - Absolute or relative file path. Trailing
    ///   separators are rejected, including in a followed symlink target.
    /// * `access_mode`       - Content access rights for the returned handle.
    /// * `create_mode`       - Whether an existing final name is permitted.
    /// * `permissions`       - Initial permissions if a file is created.
    /// * `working_directory` - Relative-path base, or root when absent.
    ///
    /// # Returns
    ///
    /// Returns a registered handle. Exclusive creation returns
    /// `AlreadyExists` for any occupied final name. Other errors include
    /// pathname, type, symlink-budget, handle-ID, and driver errors. A
    /// returned error publishes neither a new name nor a handle. Allocation
    /// failure follows the kernel allocator contract.
    pub fn open_or_create(
        &mut self,
        path: &str,
        access_mode: AccessMode,
        create_mode: CreateMode,
        permissions: Permissions,
        working_directory: Option<Arc<Directory>>,
    ) -> Result<Arc<FileHandle>, FileSystemError> {
        self.open_or_create_with_options(
            path, access_mode, create_mode, OpenContentsMode::Preserve,
            permissions, working_directory,
        )
    }

    /// Opens or creates a regular file with the requested contents mode.
    ///
    /// The caller holds global VFS serialization until registration completes.
    /// A newly created file starts empty. For an existing target, truncation
    /// changes the shared inode size and preserves existing handle offsets.
    ///
    /// # Arguments
    ///
    /// * `path`               - File pathname. Trailing separators are
    ///   rejected.
    /// * `access_mode`        - Content access rights requested for the handle.
    /// * `create_mode`        - Whether an occupied final name is permitted.
    /// * `open_contents_mode` - Whether to preserve or truncate an existing
    ///   target.
    /// * `permissions`        - Initial permissions when creating a file.
    /// * `working_directory`  - Relative-path base, or root when absent.
    ///
    /// # Returns
    ///
    /// Returns a registered handle that the caller must close. Exclusive
    /// creation returns `AlreadyExists` for an occupied final name. A returned
    /// error preserves the namespace and existing contents and registers no
    /// handle. Truncation requests require a write-capable access mode and
    /// return `ResourceBusy` when the existing inode has name protection.
    pub fn open_or_create_with_options(
        &mut self,
        path: &str,
        access_mode: AccessMode,
        create_mode: CreateMode,
        open_contents_mode: OpenContentsMode,
        permissions: Permissions,
        working_directory: Option<Arc<Directory>>,
    ) -> Result<Arc<FileHandle>, FileSystemError> {
        if open_contents_mode == OpenContentsMode::Truncate
            && !access_mode.can_write()
        {
            return Err(FileSystemError::InvalidAccessMode);
        }

        // Check handle capacity before the driver creates or truncates a file.
        let next_handle_id = self.next_handle_id.checked_add(1)
            .ok_or(FileSystemError::InternalError)?;
        let mut target_path = path.to_string();
        let mut base = working_directory;
        let mut symlink_budget = super::resolver::MAX_SYMLINK_DEPTH;

        loop {
            let (parent_path, name) = split_parent_and_name(&target_path)?;

            if target_path.ends_with('/') {
                return Err(FileSystemError::NotADirectory);
            }

            // The directory walker reports the budget left after following
            // intermediate links. The next final link consumes that same
            // budget instead of starting another independent traversal.
            let parent = self.walk_to_directory(
                parent_path, base.take(), &mut symlink_budget,
            )?;

            if name == "." || name == ".." {
                return Err(if create_mode == CreateMode::Exclusive {
                    FileSystemError::AlreadyExists
                } else {
                    FileSystemError::NotAFile
                });
            }

            let inode = match parent.lookup_child(name)? {
                Some(entry) => {
                    if create_mode == CreateMode::Exclusive {
                        return Err(FileSystemError::AlreadyExists);
                    }

                    if entry.entry_type == DirectoryEntryType::Symlink {
                        if symlink_budget == 0 {
                            return Err(FileSystemError::TooManySymlinks);
                        }
                        symlink_budget -= 1;

                        let link = self.get_or_load_inode(
                            &parent.inode.filesystem, entry.inode_number,
                        )?;
                        target_path = link.filesystem.read_symlink(&link)?;
                        // Absolute targets ignore this base in the next walk.
                        // Relative targets start in the link's parent.
                        base = Some(parent);
                        continue;
                    }

                    if entry.entry_type != DirectoryEntryType::File {
                        return Err(FileSystemError::NotAFile);
                    }

                    let inode = self.get_or_load_inode(
                        &parent.inode.filesystem, entry.inode_number,
                    )?;

                    if !inode.is_file() {
                        return Err(FileSystemError::NotAFile);
                    }

                    let key = OpenFileKey::new(
                        inode.filesystem.instance_id(), inode.inode_number,
                    );

                    if inode.metadata.lock().link_count == 0
                        || self.open_files.get(&key)
                            .is_some_and(|master| master.is_pending_deletion())
                    {
                        return Err(FileSystemError::NotFound);
                    }

                    if open_contents_mode == OpenContentsMode::Truncate {
                        self.ensure_open_truncation_allowed(&inode, &key)?;
                        Self::truncate_for_open(&inode)?;
                    }
                    inode
                }
                None => {
                    if parent.inode.metadata.lock().link_count == 0 {
                        return Err(FileSystemError::NotFound);
                    }

                    let timestamps = Timestamps::new(current_timestamp());
                    let inode = parent.inode.filesystem.create_file(
                        &parent.inode, name, permissions, timestamps,
                    )?;

                    // Driver creation has committed. Retention and
                    // registration have no recoverable error return.
                    self.retain_inode(inode)
                }
            };

            return Ok(self.register_open_handle(
                inode,
                access_mode,
                OpenNamespaceProtection {
                    flags: NamespaceProtection::NONE,
                    name_guard: None,
                    ancestor_guard: None,
                },
                next_handle_id,
            ));
        }
    }

    /// Checks existing name protection and the exclusive write lock.
    ///
    /// The caller holds global VFS serialization and checks before acquiring
    /// any protection for this open. The new handle cannot own an existing
    /// exclusive write lock.
    ///
    /// # Arguments
    ///
    /// * `inode` - Canonical regular-file inode to truncate.
    /// * `key`   - Open-file table key identifying that inode.
    ///
    /// # Returns
    ///
    /// Returns `Ok(())` if truncation is allowed, `ResourceBusy` if the
    /// inode has name protection, or `FileLocked` for an exclusive write lock.
    fn ensure_open_truncation_allowed(
        &self,
        inode: &Inode,
        key: &OpenFileKey,
    ) -> Result<(), FileSystemError> {
        if inode.names_are_protected() {
            return Err(FileSystemError::ResourceBusy);
        }

        if self.open_files.get(key)
            .is_some_and(|master| master.is_write_blocked(self.next_handle_id))
        {
            return Err(FileSystemError::FileLocked);
        }

        Ok(())
    }

    /// Commits truncation immediately before registering the checked open.
    ///
    /// The caller holds global VFS serialization and has completed all
    /// recoverable open checks and any requested token acquisition.
    ///
    /// # Arguments
    ///
    /// * `inode` - Canonical regular-file inode for the checked open.
    ///
    /// # Returns
    ///
    /// Returns `Ok(())` after updating the shared size and modification time.
    /// A driver error preserves the file's data and metadata.
    fn truncate_for_open(inode: &Inode) -> Result<(), FileSystemError> {
        inode.filesystem.truncate_to_zero(inode)?;

        let now = current_timestamp();
        let mut metadata = inode.metadata.lock();
        metadata.size = 0;
        metadata.timestamps.modified = now;
        metadata.dirty = true;

        Ok(())
    }

    /// Registers a checked handle without further recoverable operations.
    ///
    /// The caller holds global VFS serialization and has verified that
    /// `next_handle_id` is the checked successor of the current ID. Allocation
    /// failure follows the existing kernel allocator contract.
    ///
    /// # Arguments
    ///
    /// * `inode` - Canonical regular-file inode to hold open.
    /// * `access_mode` - Content access rights for the handle.
    /// * `namespace_protection` - Tokens already acquired for this handle.
    /// * `next_handle_id` - Checked successor to commit after registration.
    ///
    /// # Returns
    ///
    /// Returns the newly registered handle.
    fn register_open_handle(
        &mut self,
        inode: Arc<Inode>,
        access_mode: AccessMode,
        namespace_protection: OpenNamespaceProtection,
        next_handle_id: u64,
    ) -> Arc<FileHandle> {
        let key = OpenFileKey::new(
            inode.filesystem.instance_id(), inode.inode_number,
        );
        let master = self.open_files.entry(key)
            .or_insert_with(|| Arc::new(File::new(Arc::clone(&inode))));

        let handle = Arc::new(FileHandle::new(
            Arc::clone(master), self.next_handle_id, access_mode,
            namespace_protection,
        ));

        master.register_handle(&handle);
        self.next_handle_id = next_handle_id;
        handle
    }

    /// Reads up to `buffer.len()` bytes from `handle` into `buffer`.
    ///
    /// Reads from the handle's current position and advances it by the number
    /// of bytes successfully read. The read is dispatched to the filesystem
    /// driver via [`super::driver::FileSystemInstance::read`].
    ///
    /// After a successful read, `accessed` on the underlying inode's timestamps
    /// is updated, and the inode is marked dirty for a later sync.
    ///
    /// A read of zero bytes (e.g., because the handle is at the end of the
    /// file) is not an error; the driver returns `0` and the handle position
    /// is unchanged.
    /// The handle state is locked through the driver read and offset update so
    /// operations sharing this handle cannot read from the same offset. A
    /// closed handle returns `InvalidHandle` before calling the driver.
    ///
    /// # Arguments
    ///
    /// * `handle` - The file handle to read from. Must have been obtained from
    ///   `open` and must have [`AccessMode::ReadOnly`] or
    ///   [`AccessMode::ReadWrite`].
    /// * `buffer` - The destination buffer. Filled with up to `buffer.len()`
    ///   bytes starting from the handle's current position.
    ///
    /// # Returns
    ///
    /// Returns `Ok(usize)` with the number of bytes read and placed in
    /// `buffer`. Returns `Err(FileSystemError::InvalidAccessMode)` if the
    /// handle does not permit reads, `InvalidHandle` if it was closed, or
    /// other `Err(FileSystemError)` variants for I/O failures.
    pub fn read(
        &mut self,
        handle: &FileHandle,
        buffer: &mut [u8],
    ) -> Result<usize, FileSystemError> {
        let mut state = handle.state.lock();
        if state.closed {
            return Err(FileSystemError::InvalidHandle);
        }
        if !handle.can_read() {
            return Err(FileSystemError::InvalidAccessMode);
        }

        let inode = Arc::clone(&handle.file.inode);
        let bytes_read = inode.filesystem.read(&inode, state.position, buffer)?;
        state.position = state.position.saturating_add(bytes_read as u64);
        drop(state);

        // Update the accessed timestamp and mark the inode dirty.
        self.touch_accessed(&inode)?;

        Ok(bytes_read)
    }

    /// Writes `buffer.len()` bytes from `buffer` into the file at `handle`'s
    /// current position.
    ///
    /// Writes from the handle's current position and advances it by the number
    /// of bytes successfully written. The write is dispatched to the filesystem
    /// driver via [`super::driver::FileSystemInstance::write`].
    ///
    /// If another handle holds an exclusive write lock on the same file and
    /// `handle` is not the lock holder, the write is rejected with
    /// [`FileSystemError::FileLocked`].
    ///
    /// After a successful write, `modified` and `accessed` on the underlying
    /// inode's timestamps are updated, `size` is updated if the write extended
    /// the file, and the inode is marked dirty for a later sync.
    /// The handle state is locked through the driver write and offset update so
    /// operations sharing this handle cannot write from the same offset. A
    /// closed handle returns `InvalidHandle` before calling the driver.
    ///
    /// # Arguments
    ///
    /// * `handle`       - The file handle to write through. Must have
    ///   [`AccessMode::WriteOnly`] or [`AccessMode::ReadWrite`].
    /// * `buffer`       - The source buffer whose entire contents are written
    ///   starting at the handle's current position.
    ///
    /// # Returns
    ///
    /// Returns `Ok(usize)` with the number of bytes written, which equals
    /// `buffer.len()` on success. Returns
    /// `Err(FileSystemError::InvalidHandle)` if the handle was closed,
    /// `Err(FileSystemError::InvalidAccessMode)` if it does not permit writes,
    /// `Err(FileSystemError::FileLocked)` if another handle owns the exclusive
    /// write lock, or other `Err(FileSystemError)` variants for I/O failures.
    pub fn write(
        &mut self,
        handle: &FileHandle,
        buffer: &[u8],
    ) -> Result<usize, FileSystemError> {
        let mut state = handle.state.lock();
        if state.closed {
            return Err(FileSystemError::InvalidHandle);
        }
        if !handle.can_write() {
            return Err(FileSystemError::InvalidAccessMode);
        }

        // Check whether another handle holds an exclusive write lock.
        if handle.file.is_write_blocked(handle.id) {
            return Err(FileSystemError::FileLocked);
        }

        let inode = Arc::clone(&handle.file.inode);
        let bytes_written = inode.filesystem.write(
            &inode,
            state.position,
            buffer,
        )?;

        state.position = state.position.saturating_add(bytes_written as u64);

        // Update size if the write extended the file, then mark dirty.
        let new_end = state.position;
        drop(state);
        self.touch_modified(&inode, new_end)?;

        Ok(bytes_written)
    }

    /// Closes one registered file handle and invalidates all of its `Arc`
    /// clones. Separate opens of the same inode remain usable.
    ///
    /// If this was the last handle, and its open-file master is pending
    /// deletion (meaning all directory entries were removed while the file
    /// was open), this method calls
    /// [`super::driver::FileSystemInstance::free_inode`] to release the
    /// inode's data blocks, completing the deferred deletion.
    ///
    /// If the closing handle owns the exclusive write lock, unregistering it
    /// releases that lock. A failed deferred free leaves the handle open and
    /// registered so its owner can retry close. Namespace protection also
    /// remains effective on failure. Successful close releases this handle's
    /// contributions even if Arc clones remain; other opens keep their tokens.
    ///
    /// After success, the caller removes and drops its handle-table entry.
    ///
    /// # Arguments
    ///
    /// * `handle` - The exact [`Arc<FileHandle>`] returned by `open`, or a
    ///   clone of it. The caller retains this reference on failure.
    ///
    /// # Returns
    ///
    /// Returns `Ok(())` on success, `InvalidHandle` if the handle is closed or
    /// not registered with this VFS, or a driver error if deferred freeing
    /// fails. On a driver error, the handle remains open.
    pub fn close(
        &mut self,
        handle: &Arc<FileHandle>,
    ) -> Result<(), FileSystemError> {
        let key = OpenFileKey::new(handle.file.inode.filesystem.instance_id(),
            handle.file.inode.inode_number);

        let master = Arc::clone(
            self.open_files.get(&key)
                .ok_or(FileSystemError::InvalidHandle)?
        );

        // The key alone does not prove that this handle belongs to the master
        // currently in the table. A stale handle may name the same inode.
        if !Arc::ptr_eq(&master, &handle.file) {
            return Err(FileSystemError::InvalidHandle);
        }

        let mut handle_state = handle.state.lock();
        if handle_state.closed || !master.has_handle(handle) {
            return Err(FileSystemError::InvalidHandle);
        }

        // Free before committing the last close. On a driver error the handle
        // and table entry remain intact, so the caller can retry.
        if master.registered_handle_count() == 1
            && master.is_pending_deletion()
        {
            master.inode.filesystem.free_inode(&master.inode)?;
            // Only a successful free removes the canonical VFS inode.
            self.forget_inode(&key);
        }

        let remaining = master.unregister_handle(handle)?;
        handle_state.closed = true;
        let namespace_protection = handle_state.namespace_protection.take();
        drop(handle_state);

        // Release only after all fallible close work succeeds. Destructors
        // run outside the handle-state lock and perform no driver I/O.
        drop(namespace_protection);

        if remaining == 0 {
            self.open_files.remove(&key);
        }

        Ok(())
    }

    /// Sets the read/write position of `handle` to `offset` bytes from the
    /// beginning of the file.
    ///
    /// Does not validate `offset` against the file's current size. Seeking
    /// beyond the end of the file is permitted; a subsequent write will extend
    /// the file (with the gap zero-filled by the driver), and a subsequent
    /// read will return zero bytes. Seeking takes the handle's state lock so
    /// it cannot interleave with a read, write, or close on the same handle.
    ///
    /// # Arguments
    ///
    /// * `handle` - The file handle whose position should be updated.
    /// * `offset` - The absolute byte offset to seek to, measured from the
    ///   beginning of the file.
    ///
    /// # Returns
    ///
    /// Returns `Ok(())` after seeking, or `InvalidHandle` if the handle was
    /// closed.
    pub fn seek(&self, handle: &FileHandle, offset: u64
    ) -> Result<(), FileSystemError> {
        handle.seek_to(offset)
    }

    /// Repositions an open file handle by a signed byte displacement.
    ///
    /// The caller holds global VFS serialization. The handle state lock
    /// covers reading the current position and publishing the new one.
    /// Seeking from the end uses the canonical inode size. Seeking past
    /// the end is valid and changes only the offset.
    ///
    /// # Arguments
    ///
    /// * `handle` - Registered open handle whose position should change.
    /// * `offset` - Signed displacement in bytes from the selected origin.
    /// * `origin` - Whether the base is the start, current position, or end.
    ///
    /// # Returns
    ///
    /// Returns the new absolute byte position on success. A closed handle
    /// returns `InvalidHandle`, a negative result returns `InvalidOffset`,
    /// and a result greater than [`i64::MAX`] returns `OffsetOverflow`.
    /// On error, the previous offset is preserved.
    pub fn seek_signed(
        &self,
        handle: &FileHandle,
        offset: i64,
        origin: SeekOrigin,
    ) -> Result<i64, FileSystemError> {
        let mut state = handle.state.lock();
        if state.closed {
            return Err(FileSystemError::InvalidHandle);
        }

        let base = match origin {
            SeekOrigin::Start => 0,
            SeekOrigin::Current => state.position,
            SeekOrigin::End => handle.file.inode.metadata.lock().size,
        };

        // Widen before adding so a negative displacement can bring a base
        // above the signed range back within it.
        let position = i128::from(base) + i128::from(offset);
        if position < 0 {
            return Err(FileSystemError::InvalidOffset);
        }

        let position = i64::try_from(position)
            .map_err(|_| FileSystemError::OffsetOverflow)?;

        state.position = position as u64;
        Ok(position)
    }

    /// Returns a metadata snapshot for the inode at `path`.
    ///
    /// Resolves `path` to an inode and constructs an [`InodeStat`] snapshot
    /// from its current in-memory metadata. The snapshot does not hold a
    /// reference to the inode and can be freely copied into a userspace buffer
    /// by a `stat`-family syscall.
    ///
    /// If `path` resolves to a symlink, the symlink is followed and the
    /// target's metadata is returned. To stat the symlink inode itself, pass
    /// `follow_symlinks: false`.
    ///
    /// # Arguments
    ///
    /// * `path`              - The path of the inode to stat.
    /// * `working_directory` - The starting directory for relative paths.
    ///   `None` means use the global root.
    /// * `follow_symlinks`   - Whether to follow a symlink at the final path
    ///   component. `true` returns the target's stat; `false` returns the
    ///   symlink's own stat.
    ///
    /// # Returns
    ///
    /// Returns `Ok(InodeStat)` on success, or `Err(FileSystemError)` if the
    /// path cannot be resolved, or an I/O error occurs.
    pub fn stat(
        &self,
        path: &str,
        working_directory: Option<Arc<Directory>>,
        follow_symlinks: bool,
    ) -> Result<InodeStat, FileSystemError> {
        let inode = self.resolve(path, working_directory, follow_symlinks)?;
        Ok(inode.stat())
    }

    /// Creates a new regular file at `path`.
    ///
    /// Resolves the parent directory of `path`, verifies that no entry with
    /// the terminal name already exists, then calls
    /// [`super::driver::FileSystemInstance::create_file`] on the parent
    /// directory's filesystem driver. Each later lookup reads the driver's
    /// current directory state.
    ///
    /// # Arguments
    ///
    /// * `path`              - The path of the file to create. The parent
    ///   directory must already exist.
    /// * `permissions`       - The initial access-control permissions for the
    ///   new file.
    /// * `working_directory` - The starting directory for relative paths.
    ///   `None` means use the global root.
    ///
    /// # Returns
    ///
    /// Returns `Ok(Arc<Inode>)` for the newly created file inode on success,
    /// `Err(FileSystemError::AlreadyExists)` if an entry with that name
    /// already exists in the parent directory,
    /// `Err(FileSystemError::NotADirectory)` if `path` ends in `/`, or other
    /// `Err(FileSystemError)` variants for resolution and I/O failures.
    pub fn create_file(
        &mut self,
        path: &str,
        permissions: Permissions,
        working_directory: Option<Arc<Directory>>,
    ) -> Result<Arc<super::inode::Inode>,
        FileSystemError>
    {
        let (parent_path, file_name) = split_parent_and_name(path)?;
        if path.ends_with('/') {
            return Err(FileSystemError::NotADirectory);
        }

        let parent_dir = self.resolve_dir_with_budget(
            parent_path,
            working_directory,
            super::resolver::MAX_SYMLINK_DEPTH,
        )?;

        // Reject if the name already exists in the parent directory.
        if parent_dir.lookup_child(file_name)?.is_some() {
            return Err(FileSystemError::AlreadyExists);
        }

        if parent_dir.inode.metadata.lock().link_count == 0 {
            return Err(FileSystemError::NotFound);
        }

        let now = current_timestamp();
        let timestamps = Timestamps::new(now);

        let new_inode = parent_dir.inode.filesystem.create_file(
            &parent_dir.inode,
            file_name,
            permissions,
            timestamps,
        )?;

        Ok(self.retain_inode(new_inode))
    }

    /// Creates a new directory at `path`.
    ///
    /// Resolves the parent directory of `path`, verifies that no entry with
    /// the terminal name already exists, then calls
    /// [`super::driver::FileSystemInstance::create_directory`] on the parent
    /// directory's filesystem driver. Each later lookup reads the driver's
    /// current directory state.
    ///
    /// # Arguments
    ///
    /// * `path`              - The path of the directory to create. The parent
    ///   directory must already exist.
    /// * `permissions`       - The initial access-control permissions for the
    ///   new directory.
    /// * `working_directory` - The starting directory for relative paths.
    ///   `None` means use the global root.
    ///
    /// # Returns
    ///
    /// Returns `Ok(Arc<Inode>)` for the newly created directory inode on
    /// success, `Err(FileSystemError::AlreadyExists)` if an entry with that
    /// name already exists in the parent directory, or other
    /// `Err(FileSystemError)` variants for resolution and I/O failures.
    pub fn create_directory(
        &mut self,
        path: &str,
        permissions: Permissions,
        working_directory: Option<Arc<Directory>>,
    ) -> Result<Arc<super::inode::Inode>,
        FileSystemError>
    {
        let (parent_path, dir_name) = split_parent_and_name(path)?;

        let parent_dir = self.resolve_dir_with_budget(
            parent_path,
            working_directory,
            super::resolver::MAX_SYMLINK_DEPTH,
        )?;

        if parent_dir.lookup_child(dir_name)?.is_some() {
            return Err(FileSystemError::AlreadyExists);
        }

        if parent_dir.inode.metadata.lock().link_count == 0 {
            return Err(FileSystemError::NotFound);
        }

        let now = current_timestamp();
        let timestamps = Timestamps::new(now);

        let new_inode = parent_dir.inode.filesystem.create_directory(
            &parent_dir.inode,
            dir_name,
            permissions,
            timestamps,
        )?;

        Ok(self.retain_inode(new_inode))
    }

    /// Creates a new symbolic link at `path` pointing to `target`.
    ///
    /// Resolves the parent directory of `path`, verifies that no entry with
    /// the terminal name already exists, then calls
    /// [`super::driver::FileSystemInstance::create_symlink`] on the parent
    /// directory's filesystem driver. Each later lookup reads the driver's
    /// current directory state.
    ///
    /// The `target` string is stored verbatim as the symlink's data. It may
    /// be an absolute or relative path; the resolver interprets it at
    /// traversal time, not at creation time. No validation is performed on
    /// `target` beyond checking that it is non-empty.
    ///
    /// # Arguments
    ///
    /// * `path`              - The path at which to create the symlink. The
    ///   parent directory must already exist.
    /// * `target`            - The target path the symlink will point to.
    /// * `working_directory` - The starting directory for relative paths.
    ///   `None` means use the global root.
    ///
    /// # Returns
    ///
    /// Returns `Ok(Arc<Inode>)` for the newly created symlink inode on
    /// success, `Err(FileSystemError::AlreadyExists)` if an entry with that
    /// name already exists, `Err(FileSystemError::InvalidPath)` if `target`
    /// is empty, `Err(FileSystemError::NotADirectory)` if `path` ends in `/`,
    /// or other `Err(FileSystemError)` variants for resolution and I/O
    /// failures.
    pub fn create_symlink(
        &mut self,
        path: &str,
        target: &str,
        working_directory: Option<Arc<Directory>>,
    ) -> Result<Arc<super::inode::Inode>,
        FileSystemError>
    {
        if target.is_empty() {
            return Err(FileSystemError::InvalidPath);
        }

        let (parent_path, link_name) = split_parent_and_name(path)?;
        if path.ends_with('/') {
            return Err(FileSystemError::NotADirectory);
        }

        let parent_dir = self.resolve_dir_with_budget(
            parent_path,
            working_directory,
            super::resolver::MAX_SYMLINK_DEPTH,
        )?;

        if parent_dir.lookup_child(link_name)?.is_some() {
            return Err(FileSystemError::AlreadyExists);
        }

        if parent_dir.inode.metadata.lock().link_count == 0 {
            return Err(FileSystemError::NotFound);
        }

        let now = current_timestamp();
        let timestamps = Timestamps::new(now);

        let new_inode = parent_dir.inode.filesystem.create_symlink(
            &parent_dir.inode,
            link_name,
            target,
            timestamps,
        )?;

        Ok(self.retain_inode(new_inode))
    }

    /// Creates another directory entry for the inode at `source_path`.
    ///
    /// Both names share the same inode, data, metadata, and open-file master.
    /// Only regular files and symlinks can be linked, and the destination
    /// directory must belong to the source inode's file system instance.
    /// An existing destination is never replaced, even when it names the
    /// source inode.
    ///
    /// The caller holds the global VFS lock throughout. All VFS checks and
    /// count arithmetic precede the driver mutation. After the driver commits,
    /// only the canonical inode's in-memory link count is changed. The driver
    /// has already updated its backing count, so this change does not itself
    /// mark the inode dirty. Content timestamps remain unchanged.
    ///
    /// # Arguments
    ///
    /// * `source_path` - Existing absolute or relative source path.
    ///   Intermediate symlinks are followed. A trailing slash requires a
    ///   directory, which cannot be hard-linked.
    /// * `destination_path` - New absolute or relative name. Its parent must
    ///   exist. The final component must not be empty, `.` or `..`, contain
    ///   NUL, or be followed by a trailing slash.
    /// * `working_directory` - Base for both relative paths; `None` uses root.
    /// * `follow_symlinks` - Whether to follow a final source symlink. When
    ///   false, the new name refers to the symlink inode itself, even if its
    ///   target does not exist. Relative symlink targets are interpreted from
    ///   the directory containing whichever name is used to follow the link.
    ///
    /// # Returns
    ///
    /// Returns `Ok(())` after creating the name, `OperationNotAllowed` for a
    /// directory source, `CrossFileSystemOperation` for different file system
    /// instances, `AlreadyExists` for an occupied destination, or
    /// `TooManyHardLinks` at the count limit. Invalid names, resolution failures,
    /// and driver failures return their corresponding errors. A returned
    /// error leaves the destination entry and source link count unchanged.
    pub fn create_hard_link(
        &mut self,
        source_path: &str,
        destination_path: &str,
        working_directory: Option<Arc<Directory>>,
        follow_symlinks: bool,
    ) -> Result<(), FileSystemError> {
        let (destination_parent_path, destination_name) =
            split_parent_and_name(destination_path)?;
        if destination_path.ends_with('/') {
            return Err(FileSystemError::NotADirectory);
        }
        if destination_name == "." || destination_name == ".."
            || destination_name.contains('\0')
        {
            return Err(FileSystemError::InvalidPath);
        }

        let source = self.resolve(
            source_path, working_directory.clone(), follow_symlinks,
        )?;
        if source.is_directory() {
            return Err(FileSystemError::OperationNotAllowed);
        }

        let destination_parent = self.resolve_dir_with_budget(
            destination_parent_path,
            working_directory,
            super::resolver::MAX_SYMLINK_DEPTH,
        )?;
        if source.filesystem.instance_id()
            != destination_parent.inode.filesystem.instance_id()
        {
            return Err(FileSystemError::CrossFileSystemOperation);
        }
        // A retained wrapper must not allow creation in an unlinked directory.
        if destination_parent.inode.metadata.lock().link_count == 0 {
            return Err(FileSystemError::NotFound);
        }
        if destination_parent.lookup_child(destination_name)?.is_some() {
            return Err(FileSystemError::AlreadyExists);
        }

        let new_count = {
            let metadata = source.metadata.lock();
            if metadata.link_count == 0 {
                return Err(FileSystemError::NotFound);
            }
            metadata.link_count.checked_add(1)
                .ok_or(FileSystemError::TooManyHardLinks)?
        };

        // Do not hold the metadata lock while the driver commits its entry
        // and backing count. The global VFS lock serializes this operation.
        source.filesystem.link(
            &source, &destination_parent.inode, destination_name,
        )?;
        source.metadata.lock().link_count = new_count;

        Ok(())
    }

    /// Removes one name for a regular file or symlink without following it.
    ///
    /// The caller holds the global VFS lock throughout this operation.
    /// Other hard links and registered open handles preserve file lifetime.
    /// Directories, including mount hosts, return `OperationNotAllowed`.
    /// Name-protected regular files return `ResourceBusy` for every alias.
    /// Trailing slashes on files or symlinks return `NotADirectory`, even if a
    /// symlink targets a directory. This is an intentional contract.
    ///
    /// # Arguments
    ///
    /// * `path` - Entry to unlink. Empty paths return `NotFound`, root-only
    ///   paths return `ResourceBusy`, and final `.` or `..` components return
    ///   `OperationNotAllowed`, including with trailing separators.
    /// * `working_directory` - Base for relative paths; `None` uses root.
    ///   Intermediate `.` and `..` retain ordinary traversal semantics.
    ///
    /// # Returns
    ///
    /// Returns `Ok(())` after removing the name. A returned error leaves the
    /// name and backing link count unchanged.
    ///
    /// After the last link is removed, storage is reclaimed eagerly when no
    /// open-file master remains. Failed reclamation retains the zero-link
    /// inode for retry through
    /// [`super::vfs::VirtualFileSystem::reclaim_unlinked_inodes`].
    /// Unlink still succeeds because the name has already been removed.
    /// Open files retain the existing final-close reclamation behavior.
    pub fn unlink(
        &mut self,
        path: &str,
        working_directory: Option<Arc<Directory>>,
    ) -> Result<(), FileSystemError> {
        self.remove_directory_entry(path, working_directory, false)
    }

    /// Removes an empty directory's name while preserving retained wrappers.
    ///
    /// The caller holds the global VFS lock throughout this operation.
    /// Real directories permit trailing separators. Final symlinks are never
    /// followed, including with trailing separators. Protected directories and
    /// active mount hosts return `ResourceBusy` before any driver mutation.
    /// Removed wrappers retain their former parent for `..`, permit inspection,
    /// and cannot be used to create children, acquire guards, or mount volumes.
    /// Call [`super::vfs::VirtualFileSystem::reclaim_unlinked_inodes`]
    /// after releasing retained references to reclaim detached storage; removal
    /// itself performs no directory free.
    ///
    /// # Arguments
    ///
    /// * `path` - Directory name. Empty paths return `NotFound`, root-only
    ///   paths return `ResourceBusy`, and final `.` or `..` return
    ///   `OperationNotAllowed`, including with trailing separators.
    /// * `working_directory` - Base for relative paths; `None` uses root.
    ///   Intermediate `.` and `..` retain normal traversal semantics.
    ///
    /// # Returns
    ///
    /// Returns `Ok(())` after detachment, `NotADirectory` for files or symlinks,
    /// `DirectoryNotEmpty` for nonempty directories, or the applicable path,
    /// protection, mount, or driver error. Rejections leave the entry intact.
    pub fn remove_directory(
        &mut self,
        path: &str,
        working_directory: Option<Arc<Directory>>,
    ) -> Result<(), FileSystemError> {
        self.remove_directory_entry(path, working_directory, true)
    }

    /// Applies the common preflight and commit sequence for namespace removal.
    ///
    /// All type, protection, mount, and emptiness checks precede driver unlink.
    /// Directory reclamation is deferred to explicit VFS cleanup; file lifetime
    /// continues to use link counts and registered open-file masters.
    ///
    /// # Arguments
    ///
    /// * `path` - Unnormalized pathname whose final spelling is validated.
    /// * `working_directory` - Relative-path base, or root when absent.
    /// * `directory_only` - Selects directory removal rather than file unlink.
    ///
    /// # Returns
    ///
    /// Returns success after detachment, or a validation or driver unlink
    /// error before detachment. Failed eager reclamation retains the inode
    /// for explicit cleanup and does not change the successful result.
    fn remove_directory_entry(
        &mut self,
        path: &str,
        working_directory: Option<Arc<Directory>>,
        directory_only: bool,
    ) -> Result<(), FileSystemError> {
        if path.is_empty() {
            return Err(FileSystemError::NotFound);
        }

        let trimmed = path.trim_end_matches('/');
        if trimmed.is_empty() {
            return Err(FileSystemError::ResourceBusy);
        }

        let (parent_path, entry_name) = split_parent_and_name(path)?;
        if entry_name == "." || entry_name == ".." {
            return Err(FileSystemError::OperationNotAllowed);
        }

        let parent_dir = self.resolve_dir_with_budget(
            parent_path,
            working_directory,
            super::resolver::MAX_SYMLINK_DEPTH,
        )?;

        // Use the resolved parent for the final component. This preserves
        // relative-path semantics and does not follow a final symlink.
        let target_entry = parent_dir.lookup_child(entry_name)?
            .ok_or(FileSystemError::NotFound)?;
        let target_inode = self.get_or_load_inode(
            &parent_dir.inode.filesystem, target_entry.inode_number,
        )?;

        // A trailing slash names a directory, never a file or symlink.
        if path.ends_with('/') && !target_inode.is_directory() {
            return Err(FileSystemError::NotADirectory);
        }

        if directory_only && !target_inode.is_directory() {
            return Err(FileSystemError::NotADirectory);
        }

        if !directory_only && target_inode.is_directory() {
            return Err(FileSystemError::OperationNotAllowed);
        }

        ensure_namespace_mutation_allowed(&target_inode)?;

        // Reject deletion of a mount point directory.
        let target_key = OpenFileKey::new(
            target_inode.filesystem.instance_id(), target_inode.inode_number);
        if self.mounts.get(&target_key).is_some() {
            return Err(FileSystemError::ResourceBusy);
        }

        // Reject deletion of a non-empty directory.
        if target_inode.is_directory() {
            // Read the target directory to check whether it is empty.
            let child_inode_count = target_inode.filesystem
                .read_dir(&target_inode)?
                .len();

            if child_inode_count > 0 {
                return Err(FileSystemError::DirectoryNotEmpty);
            }
        }

        // Prepare the post-unlink decision before changing driver state.
        let remaining_links = {
            let metadata = target_inode.metadata.lock();
            metadata.link_count.checked_sub(1)
                .ok_or(FileSystemError::InternalError)?
        };
        let open_master = if remaining_links == 0 {
            let key = OpenFileKey::new(
                target_inode.filesystem.instance_id(),
                target_inode.inode_number,
            );
            self.open_files.get(&key).cloned()
        } else {
            None
        };

        // Remove the on-disk directory entry and decrement the link count.
        target_inode.filesystem.unlink(&parent_dir.inode, entry_name)?;

        // Preserve the actual former parent even when retained wrappers were
        // created before a move. This record is released only after free.
        if target_inode.is_directory() && remaining_links == 0 {
            self.relocated_parents.insert(target_key, Arc::clone(&parent_dir));
        }

        // Decrement this in-memory inode's link count after the driver commits.
        {
            let mut metadata = target_inode.metadata.lock();
            metadata.link_count = remaining_links;
        }

        // If the link count has reached zero, decide whether to free
        // immediately or defer until the last handle closes.
        if remaining_links == 0 && !target_inode.is_directory() {
            if let Some(master) = open_master {
                // The master retains the unlinked inode until the last close.
                master.mark_pending_deletion();
            } else {
                // Namespace removal has committed. Forget the inode only
                // after successful reclamation so a failed free remains
                // available to reclaim_unlinked_inodes.
                if target_inode.filesystem.free_inode(&target_inode).is_ok() {
                    self.forget_inode(&target_key);
                }
            }
        }

        Ok(())
    }

    /// Renames an entry, replacing an existing compatible destination.
    ///
    /// Regular files and symlinks may replace either non-directory type.
    /// Directories may replace empty directories only. Final symlinks are not
    /// followed. A same-inode destination is a successful no-op, even
    /// protected. The caller holds global VFS serialization throughout
    /// preflight, driver commit, and publication of canonical metadata. Driver
    /// errors must leave both names and backing link counts unchanged.
    ///
    /// Replaced inodes retain their other hard links and registered handles.
    /// Detached directories retain their actual former parent for retained
    /// wrapper traversal. Unopened displaced files and symlinks remain in the
    /// canonical table until explicit
    /// [`super::vfs::VirtualFileSystem::reclaim_unlinked_inodes`] succeeds.
    /// Cleanup failure is separate from an already successful rename.
    ///
    /// # Arguments
    ///
    /// * `old_path` - Existing source entry. Empty paths return `NotFound`,
    ///   root-only paths return `ResourceBusy`, and final `.` or `..` return
    ///   `OperationNotAllowed`. Intermediate special components still traverse.
    /// * `new_path` - Destination subject to the same special-name rules.
    ///   Trailing separators require an existing real directory. A missing
    ///   destination returns `NotADirectory` for a non-directory source and
    ///   `NotFound` for a directory source. Final symlinks with trailing
    ///   separators return `NotADirectory`, even for directory targets. This
    ///   is a deliberate CrapOS pathname contract.
    /// * `working_directory` - Relative-path base, or global root when absent.
    ///
    /// # Returns
    ///
    /// Returns success after commit, `CrossFileSystemOperation` across
    /// instances,`ResourceBusy` for protected entries or active mount hosts,
    /// `IsADirectory` or `NotADirectory` for type mismatches,
    /// `DirectoryNotEmpty` for nonempty destination directories, `InvalidPath`
    /// for moving a directory into its subtree or embedded NUL, or a pathname/
    /// driver error before commit. Kernel allocation failure follows the
    /// existing allocator contract and is not represented as a returned
    /// filesystem error.
    pub fn rename(
        &mut self,
        old_path: &str,
        new_path: &str,
        working_directory: Option<Arc<Directory>>,
    ) -> Result<(), FileSystemError> {
        let (old_parent_path, old_name) = rename_parent_and_name(old_path)?;
        let (new_parent_path, new_name) = rename_parent_and_name(new_path)?;

        let old_parent = self.resolve_dir_with_budget(
            old_parent_path, working_directory.clone(),
            super::resolver::MAX_SYMLINK_DEPTH,
        )?;

        let new_parent = self.resolve_dir_with_budget(
            new_parent_path, working_directory,
            super::resolver::MAX_SYMLINK_DEPTH,
        )?;

        if old_parent.inode.filesystem.instance_id()
            != new_parent.inode.filesystem.instance_id()
        {
            return Err(FileSystemError::CrossFileSystemOperation);
        }

        // Read separately so same-parent operands never require two guards
        // on the same metadata lock in a single expression.
        let old_attached = old_parent.inode.metadata.lock().link_count != 0;
        let new_attached = new_parent.inode.metadata.lock().link_count != 0;

        if !old_attached || !new_attached {
            return Err(FileSystemError::NotFound);
        }

        let source_entry = old_parent.lookup_child(old_name)?
            .ok_or(FileSystemError::NotFound)?;

        let source = self.get_or_load_inode(
            &old_parent.inode.filesystem, source_entry.inode_number,
        )?;

        let destination = match new_parent.lookup_child(new_name)? {
            Some(entry) => Some(self.get_or_load_inode(
                &new_parent.inode.filesystem, entry.inode_number,
            )?),
            None => None,
        };

        // Validate spelling before accepting a same-inode no-op. A destination
        // ending in slash must already denote a real directory, not just have
        // a directory source that could be moved to this name.
        if old_path.ends_with('/') && !source.is_directory() {
            return Err(FileSystemError::NotADirectory);
        }

        if new_path.ends_with('/') {
            match destination.as_ref() {
                Some(inode) if inode.is_directory() => {}
                Some(_) => return Err(FileSystemError::NotADirectory),
                None if !source.is_directory() => {
                    return Err(FileSystemError::NotADirectory);
                }
                None => return Err(FileSystemError::NotFound),
            }
        }

        if destination
            .as_ref().is_some_and(|inode| Arc::ptr_eq(inode, &source))
        {
            return Ok(());
        }

        self.ensure_rename_target_available(&source)?;

        if let Some(inode) = destination.as_ref() {
            self.ensure_rename_target_available(inode)?;

            if source.is_directory() && !inode.is_directory() {
                return Err(FileSystemError::NotADirectory);
            }

            if !source.is_directory() && inode.is_directory() {
                return Err(FileSystemError::IsADirectory);
            }

            if inode.is_directory() && !inode.filesystem.read_dir(inode)?
                .is_empty()
            {
                return Err(FileSystemError::DirectoryNotEmpty);
            }
        }

        if source.is_directory() {
            self.ensure_rename_outside_subtree(&source,
                Arc::clone(&new_parent))?;
        }

        // Prepare every fallible VFS step before the backing namespace changes.
        // Preserve the displaced master so publication needs no later lookup.
        let displaced = if let Some(inode) = destination {
            let remaining = inode.metadata.lock().link_count.checked_sub(1)
                .ok_or(FileSystemError::InternalError)?;
            let key = OpenFileKey::new(inode.filesystem.instance_id(),
                inode.inode_number);
            let master = if remaining == 0 {
                self.open_files.get(&key).cloned()
            } else { None };
            Some((inode, key, remaining, master))
        } else { None };

        source.filesystem.rename(
            &old_parent.inode, old_name, &new_parent.inode, new_name,
        )?;

        // Commit succeeded. No returned-error path or driver operation follows.
        if let Some((inode, key, remaining, master)) = displaced {
            inode.metadata.lock().link_count = remaining;
            if remaining == 0 {
                if inode.is_directory() {
                    self.relocated_parents.insert(key, Arc::clone(&new_parent));
                } else if let Some(master) = master {
                    master.mark_pending_deletion();
                }
                // Without a master, the canonical zero-link inode retains the
                // storage identity for explicit cleanup and failed-free retry.
            }
        }

        let same_parent =
            old_parent.inode.inode_number == new_parent.inode.inode_number;

        if source.is_directory() && !same_parent {
            let key = OpenFileKey::new(source.filesystem.instance_id(),
                source.inode_number);
            self.relocated_parents.insert(key, Arc::clone(&new_parent));
        }

        let now = current_timestamp();
        {
            let mut metadata = old_parent.inode.metadata.lock();
            metadata.timestamps.modified = now;
            metadata.dirty = true;
        }

        if !same_parent {
            let mut metadata = new_parent.inode.metadata.lock();
            metadata.timestamps.modified = now;
            metadata.dirty = true;
        }

        // The current metadata model has no POSIX status-change timestamp.
        Ok(())
    }

    /// Checks protection and mount status before changing an attachment.
    ///
    /// # Arguments
    ///
    /// * `inode` - Canonical source or displaced destination inode.
    ///
    /// # Returns
    ///
    /// Returns `ResourceBusy` for protection or an active mount host;
    /// otherwise returns success. Same-inode no-ops bypass this check.
    fn ensure_rename_target_available(&self, inode: &Arc<Inode>)
        -> Result<(), FileSystemError>
    {
        ensure_namespace_mutation_allowed(inode)?;

        let key = OpenFileKey::new(inode.filesystem.instance_id(),
            inode.inode_number);

        if self.mounts.get(&key).is_some() {
            return Err(FileSystemError::ResourceBusy);
        }

        Ok(())
    }

    /// Rejects moving a directory into itself or one of its descendants.
    ///
    /// Uses current relocated parents and mount records, rather than historical
    /// wrapper parents. The existing namespace must be acyclic; trusted callers
    /// must not construct arbitrary wrappers or mutate drivers behind the VFS.
    ///
    /// # Arguments
    ///
    /// * `source` - Canonical directory inode being moved.
    /// * `parent` - Proposed destination parent.
    ///
    /// # Returns
    ///
    /// Returns `InvalidPath` if ancestry contains `source`, success at global
    /// root, or the underlying parent-resolution error.
    fn ensure_rename_outside_subtree(
        &self, source: &Arc<Inode>, mut parent: Arc<Directory>,
    ) -> Result<(), FileSystemError> {
        loop {
            if Arc::ptr_eq(source, &parent.inode) {
                return Err(FileSystemError::InvalidPath);
            }

            if parent.is_root() {
                return Ok(());
            }

            let key = OpenFileKey::new(parent.inode.filesystem.instance_id(),
                parent.inode.inode_number);

            parent = if let Some(host_key) = self.mounted_root_index.get(&key) {
                Arc::clone(&self.mounts.get(host_key)
                    .ok_or(FileSystemError::InternalError)?.host_directory)
            } else if let Some(current) = self.relocated_parents.get(&key) {
                Arc::clone(current)
            } else { parent.parent()? };
        }
    }

    /// Updates the `accessed` timestamp on `inode` and marks it dirty.
    ///
    /// Called internally after a successful read operation. The `accessed`
    /// field is updated to the current wall-clock time. The inode is marked
    /// dirty on every successful call for a later sync.
    ///
    /// # Arguments
    ///
    /// * `inode` - The inode whose `accessed` timestamp should be updated.
    ///
    /// # Returns
    ///
    /// Returns `Ok(())` always. The signature returns `Result` for consistency
    /// with other internal helpers that may fail in future extensions.
    fn touch_accessed(
        &self,
        inode: &Arc<super::inode::Inode>,
    ) -> Result<(), FileSystemError> {
        let now = current_timestamp();
        let mut metadata = inode.metadata.lock();
        metadata.timestamps.accessed = now;
        metadata.dirty = true;

        Ok(())
    }

    /// Updates the `modified` and `accessed` timestamps on `inode`, extends
    /// `size` if `new_end` exceeds the current size, and marks the inode dirty.
    ///
    /// Called internally after a successful write operation. Both timestamps
    /// are updated to the current wall-clock time. If `new_end` is greater
    /// than the inode's current `size`, `size` is extended to `new_end`.
    ///
    /// # Arguments
    ///
    /// * `inode`   - The inode whose metadata should be updated.
    /// * `new_end` - The byte offset one past the last byte written, i.e.,
    ///   the handle position after the write. Used to extend `size` if the
    ///   write reached beyond the previous end of the file.
    ///
    /// # Returns
    ///
    /// Returns `Ok(())` always. The signature returns `Result` for consistency
    /// with other internal helpers that may fail in future extensions.
    fn touch_modified(
        &self,
        inode: &Arc<super::inode::Inode>,
        new_end: u64,
    ) -> Result<(), FileSystemError> {
        let now = current_timestamp();
        let mut metadata = inode.metadata.lock();
        metadata.timestamps.modified = now;
        metadata.timestamps.accessed = now;
        metadata.size = metadata.size.max(new_end);
        metadata.dirty = true;

        Ok(())
    }
}

/// Checks protection before changing an inode's existing namespace attachment.
///
/// The caller holds global VFS serialization and checks before driver mutation.
/// There is no holder bypass. Same-inode rename no-ops are handled before this
/// check because they change no namespace state. Hard-link creation does not
/// remove or change an existing attachment and does not use this check.
///
/// # Arguments
///
/// * `inode` - Canonical inode whose entry would be removed or changed.
///
/// # Returns
///
/// Returns `ResourceBusy` if name protection or directory path protection
/// blocks the operation; otherwise returns `Ok(())`.
fn ensure_namespace_mutation_allowed(inode: &Inode) -> Result<(), FileSystemError> {
    if inode.names_are_protected()
        || inode.directory_path_guards.load(Ordering::Acquire) != 0
    {
        return Err(FileSystemError::ResourceBusy);
    }
    Ok(())
}

/// Splits a rename operand after checking its final lexical component.
///
/// Intermediate `.` and `..` retain ordinary traversal. Trailing separators
/// are trimmed only to identify the final spelling; rename checks their type.
///
/// # Arguments
///
/// * `path` - Original source or destination pathname.
///
/// # Returns
///
/// Returns parent and name, `NotFound` for empty paths, `ResourceBusy` for
/// root, `OperationNotAllowed` for final `.`/`..`, or `InvalidPath` for
/// embedded NUL.
fn rename_parent_and_name(path: &str) -> Result<(&str, &str), FileSystemError> {
    if path.is_empty() {
        return Err(FileSystemError::NotFound);
    }

    if path.contains('\0') {
        return Err(FileSystemError::InvalidPath);
    }

    if path.trim_end_matches('/').is_empty() {
        return Err(FileSystemError::ResourceBusy);
    }

    let (parent, name) = split_parent_and_name(path)?;
    if name == "." || name == ".." {
        return Err(FileSystemError::OperationNotAllowed);
    }

    Ok((parent, name))
}

/// Splits a path string into the parent directory path and the terminal
/// component name.
///
/// Given `/foo/bar/baz`, returns `("/foo/bar", "baz")`. Given `/baz`, returns
/// `("/", "baz")`. Given a relative path `foo/bar`, returns `("foo", "bar")`.
///
/// Trailing slashes are removed to find the terminal name. Callers enforce
/// whether their operation permits a directory path ending in `/`. Returns
/// [`FileSystemError::InvalidPath`] if `path` is empty or consists entirely
/// of `/` separators with no terminal name.
///
/// # Arguments
///
/// * `path` - The path string to split. May be absolute or relative.
///
/// # Returns
///
/// Returns `Ok((&str, &str))` where the first element is the parent path and
/// the second is the terminal component name, or
/// `Err(FileSystemError::InvalidPath)` if the path is malformed.
fn split_parent_and_name(path: &str) -> Result<(&str, &str), FileSystemError> {
    if path.is_empty() {
        return Err(FileSystemError::InvalidPath);
    }

    // Trim trailing slashes to locate the name. The caller checks whether
    // the operation permits this path form before changing driver state.
    let trimmed = path.trim_end_matches('/');
    if trimmed.is_empty() {
        return Err(FileSystemError::InvalidPath);
    }

    match trimmed.rfind('/') {
        // No slash at all - relative path with a single component like `foo`.
        // The parent is the working directory (represented as `.`) and the
        // name is the entire trimmed string.
        None => Ok((".", trimmed)),

        // Slash at position 0 means the parent is the root `/`.
        Some(0) => {
            let name = &trimmed[1..];
            if name.is_empty() {
                Err(FileSystemError::InvalidPath)
            } else {
                Ok(("/", name))
            }
        }

        // Slash somewhere in the middle - split at the last slash.
        Some(pos) => Ok((&trimmed[..pos], &trimmed[pos + 1..])),
    }
}

/// Returns the current wall-clock time as Unix epoch milliseconds.
///
/// Reads the current time from the kernel's `WALL_CLOCK` and `HPET` globals.
/// Returns `0` if either global is not yet initialized, which can happen
/// during early-boot filesystem operations before the RTC anchor is
/// established.
///
/// # Returns
///
/// Returns the current Unix epoch milliseconds, or `0` if the wall clock is
/// not yet available.
fn current_timestamp() -> u64 {
    unsafe {
        let hpet = globals::HPET.lock();
        let wc   = globals::WALL_CLOCK.lock();

        match (hpet.as_ref(), wc.as_ref()) {
            (Some(h), Some(w)) => w.now(h),
            _                  => 0,
        }
    }
}
