//! # Virtual File System - File Operations
//!
//! This module implements the file operation methods on
//! [`super::vfs::VirtualFileSystem`]: the public interface that all kernel
//! code uses to open, read, write, close, seek, stat, create, delete, rename,
//! and create symlinks.
//!
//! These methods are the bridge between the abstract VFS layer and the
//! filesystem drivers below. Each method resolves its path argument to an
//! inode via the resolver, performs any necessary checks (type, permissions,
//! lock state, access mode), dispatches to the appropriate
//! [`super::driver::FileSystemInstance`] method, and updates the in-memory
//! state (open-file table, directory child lists, inode metadata) to reflect
//! the result.
//!
//! ## Caller Responsibilities
//!
//! These methods operate at the kernel level and return kernel-side objects.
//! The syscall layer above them is responsible for:
//!
//! - Registering the returned [`super::file::FileHandle`] with the object
//!   manager and inserting it into the calling process's handle table.
//! - Translating kernel-side [`super::types::FileSystemError`] values into
//!   appropriate userspace error codes.
//! - Copying [`super::types::InodeStat`] snapshots into userspace buffers
//!   for `stat`-family syscalls.
//!
//! ## Open File Table Lifecycle
//!
//! The open-file table in [`super::vfs::VirtualFileSystem`] holds one
//! [`super::file::File`] master record per inode that has at least one open
//! handle. [`super::vfs::VirtualFileSystem::open`] inserts a master record on
//! the first open and registers each subsequent handle with the existing
//! record. [`super::vfs::VirtualFileSystem::close`] decrements the handle
//! count and removes the master record when the last handle closes, triggering
//! any pending deletion if the inode's link count has reached zero.

#![allow(dead_code)]

use alloc::sync::Arc;

use super::directory::Directory;
use super::file::{File, FileHandle};
use super::inode::DirectoryEntry;
use super::types::{
    AccessMode, DirectoryEntryType, FileSystemError, InodeStat, InodeType,
    OpenFileKey, Permissions, Timestamps,
};
use super::vfs::VirtualFileSystem;
use crate::globals;

impl VirtualFileSystem {
    /// Opens the file at `path` and returns a new [`FileHandle`] for it.
    ///
    /// Resolves `path` to an inode, verifies it is a regular file, then
    /// either retrieves the existing [`super::file::File`] master record from
    /// the open-file table or creates a new one. A new [`FileHandle`] is
    /// created for this open call, registered with the master record, and
    /// returned to the caller.
    ///
    /// The returned [`FileHandle`] must be wrapped in an [`Arc`] by the caller
    /// and registered with the object manager before being placed in a process
    /// handle table. The [`Arc`] is what keeps the handle alive and what the
    /// master record holds a `Weak` reference to.
    ///
    /// If `path` resolves to a symlink, the symlink is followed and the
    /// target file is opened. To open the symlink inode itself, use a path
    /// that does not require following the final symlink, which is not a
    /// supported operation for `open`, as symlink inodes are not openable as
    /// files.
    ///
    /// # Arguments
    ///
    /// * `path`              - The path of the file to open.
    /// * `access_mode`       - The access rights to grant the new handle.
    /// * `working_directory` - The starting directory for relative paths.
    ///   `None` means use the global root.
    ///
    /// # Returns
    ///
    /// Returns `Ok(FileHandle)` on success. The caller must immediately wrap
    /// this in an [`Arc<FileHandle>`] and register it with the master record
    /// via [`super::file::File::register_handle`]. Returns
    /// `Err(FileSystemError::NotFound)` if the path does not exist,
    /// `Err(FileSystemError::NotAFile)` if it resolves to a non-file inode,
    /// or other `Err(FileSystemError)` variants for I/O and resolution
    /// failures.
    pub fn open(
        &mut self,
        path: &str,
        access_mode: AccessMode,
        working_directory: Option<Arc<Directory>>,
    ) -> Result<FileHandle, FileSystemError> {
        // Resolve the path, following symlinks so the caller gets the target
        // file rather than the symlink inode.
        let inode = self.resolve(path, working_directory, true)?;

        // Only regular files can be opened through this method.
        if !inode.is_file() {
            return Err(FileSystemError::NotAFile);
        }

        // Reject opens on inodes that are pending deletion. The inode has no
        // directory entries left and cannot be found by name, so this path
        // can only be reached through a race; we treat it as NotFound.
        if inode.pending_deletion {
            return Err(FileSystemError::NotFound);
        }

        let key = OpenFileKey::new(
            inode.filesystem.instance_id(),
            inode.inode_number,
        );

        // Retrieve the existing master record or create a new one.
        let master = self.open_files
            .entry(key)
            .or_insert_with(|| Arc::new(File::new(Arc::clone(&inode))));

        let handle = FileHandle::new(Arc::clone(master), access_mode);
        Ok(handle)
    }

    /// Reads up to `buffer.len()` bytes from `handle` into `buffer`.
    ///
    /// Reads from the handle's current position and advances it by the number
    /// of bytes successfully read. The read is dispatched to the filesystem
    /// driver via [`super::driver::FileSystemInstance::read`].
    ///
    /// After a successful read, `accessed` on the underlying inode's timestamps
    /// is updated, and the inode is marked dirty for a future sync.
    ///
    /// A read of zero bytes (e.g., because the handle is at the end of the
    /// file) is not an error; the driver returns `0` and the handle position
    /// is unchanged.
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
    /// handle does not permit reads, or other `Err(FileSystemError)` variants
    /// for I/O failures.
    pub fn read(
        &mut self,
        handle: &mut FileHandle,
        buffer: &mut [u8],
    ) -> Result<usize, FileSystemError> {
        if !handle.can_read() {
            return Err(FileSystemError::InvalidAccessMode);
        }

        let inode = Arc::clone(&handle.file.inode);
        let bytes_read = inode.filesystem.read(&inode, handle.position,buffer)?;

        handle.advance_position(bytes_read);

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
    /// the file, and the inode is marked dirty for a future sync.
    ///
    /// # Arguments
    ///
    /// * `handle`       - The file handle to write through. Must have
    ///   [`AccessMode::WriteOnly`] or [`AccessMode::ReadWrite`].
    /// * `handle_value` - The process-local handle value for `handle`, used
    ///   to check write-lock ownership. Supplied by the object manager at the
    ///   syscall boundary.
    /// * `buffer`       - The source buffer whose entire contents are written
    ///   starting at the handle's current position.
    ///
    /// # Returns
    ///
    /// Returns `Ok(usize)` with the number of bytes written, which equals
    /// `buffer.len()` on success. Returns
    /// `Err(FileSystemError::InvalidAccessMode)` if the handle does not permit
    /// writes, `Err(FileSystemError::FileLocked)` if the file is exclusively
    /// locked by another handle, or other `Err(FileSystemError)` variants for
    /// I/O failures.
    pub fn write(
        &mut self,
        handle: &mut FileHandle,
        handle_value: u32,
        buffer: &[u8],
    ) -> Result<usize, FileSystemError> {
        if !handle.can_write() {
            return Err(FileSystemError::InvalidAccessMode);
        }

        // Check whether another handle holds an exclusive write lock.
        if handle.file.is_write_blocked(handle_value) {
            return Err(FileSystemError::FileLocked);
        }

        let inode = Arc::clone(&handle.file.inode);
        let bytes_written = inode.filesystem.write(
            &inode,
            handle.position,
            buffer,
        )?;

        handle.advance_position(bytes_written);

        // Update size if the write extended the file, then mark dirty.
        let new_end = handle.position;
        self.touch_modified(&inode, new_end)?;

        Ok(bytes_written)
    }

    /// Closes a file handle, unregistering it from the master record and
    /// removing the master record from the open-file table if this was the
    /// last handle.
    ///
    /// If this was the last handle, and the underlying inode has
    /// `pending_deletion` set (meaning all directory entries were removed
    /// while the file was open), this method calls
    /// [`super::driver::FileSystemInstance::free_inode`] to release the
    /// inode's data blocks, completing the deferred deletion.
    ///
    /// If the closing handle currently holds an exclusive write lock, the lock
    /// is released before the handle is unregistered.
    ///
    /// The [`Arc<FileHandle>`] passed here should be the one held by the
    /// process's handle table entry. After this call returns, the caller must
    /// remove the handle table entry so that the [`Arc`] reference count drops
    /// to zero and the [`FileHandle`] is freed.
    ///
    /// # Arguments
    ///
    /// * `handle`       - The [`Arc<FileHandle>`] to close. Must have been
    ///   obtained from `open`.
    /// * `handle_value` - The process-local handle value, used to release any
    ///   write lock held by this handle.
    ///
    /// # Returns
    ///
    /// Returns `Ok(())` on success, or `Err(FileSystemError)` if a deferred
    /// free_inode call fails on a pending-deletion inode.
    pub fn close(
        &mut self,
        handle: Arc<FileHandle>,
        handle_value: u32,
    ) -> Result<(), FileSystemError> {
        let key = OpenFileKey::new(
            handle.file.inode.filesystem.instance_id(),
            handle.file.inode.inode_number,
        );

        {
            let master_ptr = Arc::as_ptr(&handle.file) as *mut File;
            let master = unsafe { &mut *master_ptr };

            if master.is_lock_holder(handle_value) {
                // Ignore the error here; if the handle is not the lock
                // holder, release_write_lock returns InvalidHandle, which
                // is harmless at close time.
                let _ = master.release_write_lock(handle_value);
            }

            // Prune stale weak references and count remaining live handles.
            // The handle we are closing is still alive (the caller holds the
            // Arc), so the live count will be at least 1 here.
            let live_count = master.prune_and_count_handles();

            // We check whether it will reach 0 by comparing against 1.
            if live_count <= 1 {
                // This is the last (or only) handle. Remove the master record
                // from the open-file table. Dropping the Arc<File> here will
                // destroy the master record once the caller also drops their
                // Arc<FileHandle>, which holds the last Arc<File> reference.
                if let Some(master_arc) = self.open_files.remove(&key) {
                    // If the inode is pending deletion, and the link count is
                    // zero, instruct the driver to free the inode's storage.
                    if master_arc.inode.pending_deletion
                        && master_arc.inode.link_count == 0
                    {
                        master_arc.inode.filesystem
                            .free_inode(&master_arc.inode)?;
                    }
                }
            }
        }

        Ok(())
    }

    /// Sets the read/write position of `handle` to `offset` bytes from the
    /// beginning of the file.
    ///
    /// Does not validate `offset` against the file's current size. Seeking
    /// beyond the end of the file is permitted; a subsequent write will extend
    /// the file (with the gap zero-filled by the driver), and a subsequent
    /// read will return zero bytes.
    ///
    /// # Arguments
    ///
    /// * `handle` - The file handle whose position should be updated.
    /// * `offset` - The absolute byte offset to seek to, measured from the
    ///   beginning of the file.
    pub fn seek(&self, handle: &mut FileHandle, offset: u64) {
        handle.seek_to(offset);
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
    /// directory's filesystem driver. On success, a new
    /// [`super::inode::DirectoryEntry`] is added to the parent directory's
    /// in-memory child list.
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
    /// already exists in the parent directory, or other `Err(FileSystemError)`
    /// variants for resolution and I/O failures.
    pub fn create_file(
        &mut self,
        path: &str,
        permissions: Permissions,
        working_directory: Option<Arc<Directory>>,
    ) -> Result<Arc<super::inode::Inode>,
        FileSystemError>
    {
        let (parent_path, file_name) = split_parent_and_name(path)?;

        let parent_dir = self.resolve_dir_with_budget(
            parent_path,
            working_directory,
            super::resolver::MAX_SYMLINK_DEPTH,
        )?;

        // Reject if the name already exists in the parent directory.
        {
            let dir_ptr = Arc::as_ptr(&parent_dir) as *mut
                super::directory::Directory;
            let dir = unsafe { &mut *dir_ptr };
            if dir.lookup_child(file_name)?.is_some() {
                return Err(FileSystemError::AlreadyExists);
            }
        }

        let now = current_timestamp();
        let timestamps = Timestamps::new(now);

        let new_inode = parent_dir.inode.filesystem.create_file(
            &parent_dir.inode,
            file_name,
            permissions,
            timestamps,
        )?;

        // Add the new entry to the parent's in-memory child list.
        let entry = DirectoryEntry::with_inode(
            alloc::string::ToString::to_string(file_name),
            new_inode.inode_number,
            DirectoryEntryType::File,
            Arc::clone(&new_inode),
        );

        {
            let dir_ptr = Arc::as_ptr(&parent_dir) as *mut
                super::directory::Directory;
            let dir = unsafe { &mut *dir_ptr };
            dir.add_child(entry)?;
        }

        Ok(new_inode)
    }

    /// Creates a new directory at `path`.
    ///
    /// Resolves the parent directory of `path`, verifies that no entry with
    /// the terminal name already exists, then calls
    /// [`super::driver::FileSystemInstance::create_directory`] on the parent
    /// directory's filesystem driver. On success, a new
    /// [`super::inode::DirectoryEntry`] is added to the parent directory's
    /// in-memory child list.
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

        {
            let dir_ptr = Arc::as_ptr(&parent_dir) as *mut
                super::directory::Directory;
            let dir = unsafe { &mut *dir_ptr };
            if dir.lookup_child(dir_name)?.is_some() {
                return Err(FileSystemError::AlreadyExists);
            }
        }

        let now = current_timestamp();
        let timestamps = Timestamps::new(now);

        let new_inode = parent_dir.inode.filesystem.create_directory(
            &parent_dir.inode,
            dir_name,
            permissions,
            timestamps,
        )?;

        let entry = DirectoryEntry::with_inode(
            alloc::string::ToString::to_string(dir_name),
            new_inode.inode_number,
            DirectoryEntryType::Directory,
            Arc::clone(&new_inode),
        );

        {
            let dir_ptr = Arc::as_ptr(&parent_dir) as *mut
                super::directory::Directory;
            let dir = unsafe { &mut *dir_ptr };
            dir.add_child(entry)?;
        }

        Ok(new_inode)
    }

    /// Creates a new symbolic link at `path` pointing to `target`.
    ///
    /// Resolves the parent directory of `path`, verifies that no entry with
    /// the terminal name already exists, then calls
    /// [`super::driver::FileSystemInstance::create_symlink`] on the parent
    /// directory's filesystem driver. On success, a new
    /// [`super::inode::DirectoryEntry`] is added to the parent directory's
    /// in-memory child list.
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
    /// is empty, or other `Err(FileSystemError)` variants for resolution and
    /// I/O failures.
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

        let parent_dir = self.resolve_dir_with_budget(
            parent_path,
            working_directory,
            super::resolver::MAX_SYMLINK_DEPTH,
        )?;

        {
            let dir_ptr = Arc::as_ptr(&parent_dir) as *mut
                super::directory::Directory;
            let dir = unsafe { &mut *dir_ptr };
            if dir.lookup_child(link_name)?.is_some() {
                return Err(FileSystemError::AlreadyExists);
            }
        }

        let now = current_timestamp();
        let timestamps = Timestamps::new(now);

        let new_inode = parent_dir.inode.filesystem.create_symlink(
            &parent_dir.inode,
            link_name,
            target,
            timestamps,
        )?;

        let entry = DirectoryEntry::with_inode(
            alloc::string::ToString::to_string(link_name),
            new_inode.inode_number,
            DirectoryEntryType::Symlink,
            Arc::clone(&new_inode),
        );

        {
            let dir_ptr = Arc::as_ptr(&parent_dir) as *mut
                super::directory::Directory;
            let dir = unsafe { &mut *dir_ptr };
            dir.add_child(entry)?;
        }

        Ok(new_inode)
    }

    /// Removes the directory entry for the filesystem object at `path`.
    ///
    /// Resolves `path` to its parent directory and terminal name, then calls
    /// [`super::driver::FileSystemInstance::unlink`] to remove the on-disk
    /// directory entry and decrement the inode's link count. The in-memory
    /// child list of the parent directory is updated to reflect the removal.
    ///
    /// If the inode's link count reaches zero, and no open handles exist for
    /// it, [`super::driver::FileSystemInstance::free_inode`] is called
    /// immediately to release the inode's data blocks. If open handles exist,
    /// `pending_deletion` is set on the inode and the data blocks are freed
    /// when the last handle closes via [`VirtualFileSystem::close`].
    ///
    /// Deleting a non-empty directory returns
    /// [`FileSystemError::DirectoryNotEmpty`]. Deleting a directory that is
    /// currently a mount point returns [`FileSystemError::AlreadyMounted`].
    ///
    /// # Arguments
    ///
    /// * `path`              - The path of the entry to delete.
    /// * `working_directory` - The starting directory for relative paths.
    ///   `None` means use the global root.
    ///
    /// # Returns
    ///
    /// Returns `Ok(())` on success, `Err(FileSystemError::NotFound)` if the
    /// path does not exist, `Err(FileSystemError::DirectoryNotEmpty)` if the
    /// target is a non-empty directory, or other `Err(FileSystemError)`
    /// variants for resolution and I/O failures.
    pub fn delete(
        &mut self,
        path: &str,
        working_directory: Option<Arc<Directory>>,
    ) -> Result<(), FileSystemError> {
        let (parent_path, entry_name) = split_parent_and_name(path)?;

        let parent_dir = self.resolve_dir_with_budget(
            parent_path,
            working_directory,
            super::resolver::MAX_SYMLINK_DEPTH,
        )?;

        // Resolve the target inode so we can inspect its type, link count,
        // and open-handle state.
        let target_inode = self.resolve(path, None, false)?;

        // Reject deletion of a mount point directory.
        if target_inode.mount_flag {
            return Err(FileSystemError::AlreadyMounted);
        }

        // Reject deletion of a non-empty directory.
        if target_inode.is_directory() {
            // Load the directory's children to check emptiness.
            // Look up the target as a child of parent to get its wrapper.
            // We only need the child count so a read_dir call suffices.
            let child_inode_count = target_inode.filesystem
                .read_dir(&target_inode)?
                .len();

            if child_inode_count > 0 {
                return Err(FileSystemError::DirectoryNotEmpty);
            }
        }

        // Remove the on-disk directory entry and decrement the link count.
        target_inode.filesystem.unlink(&parent_dir.inode, entry_name)?;

        // Remove the entry from the parent's in-memory child list.
        {
            let dir_ptr = Arc::as_ptr(&parent_dir) as *mut
                super::directory::Directory;
            let dir = unsafe { &mut *dir_ptr };
            dir.remove_child(entry_name)?;
        }

        // Decrement the in-memory link count.
        {
            let inode_ptr =
                Arc::as_ptr(&target_inode) as *mut
                super::inode::Inode;
            unsafe {
                (*inode_ptr).link_count
                    = (*inode_ptr).link_count.saturating_sub(1);
            }
        }

        // If the link count has reached zero, decide whether to free
        // immediately or defer until the last handle closes.
        if target_inode.link_count == 0 {
            let key = OpenFileKey::new(
                target_inode.filesystem.instance_id(),
                target_inode.inode_number,
            );

            if self.open_files.contains_key(&key) {
                // Open handles exist - mark for deferred deletion.
                let inode_ptr =
                    Arc::as_ptr(&target_inode) as *mut
                    super::inode::Inode;
                unsafe { (*inode_ptr).pending_deletion = true; }
            } else {
                // No open handles - free immediately.
                target_inode.filesystem.free_inode(&target_inode)?;
            }
        }

        Ok(())
    }

    /// Moves the directory entry at `old_path` to `new_path`, optionally
    /// changing its name.
    ///
    /// Resolves both paths to their respective parent directories and terminal
    /// names, then calls [`super::driver::FileSystemInstance::rename`] on the
    /// source inode's filesystem driver. The in-memory child lists of both
    /// parent directories are updated to reflect the move.
    ///
    /// `old_path` and `new_path` may share the same parent directory, in
    /// which case this is an in-place rename. Cross-filesystem renames (where
    /// the two paths reside on different mounted volumes) are not supported
    /// and return [`FileSystemError::InvalidPath`].
    ///
    /// If an entry already exists at `new_path`, the rename is rejected with
    /// [`FileSystemError::AlreadyExists`]. The caller is responsible for
    /// removing the destination first if an overwriting rename is desired.
    ///
    /// # Arguments
    ///
    /// * `old_path`          - The current path of the entry to rename.
    /// * `new_path`          - The desired path after the rename.
    /// * `working_directory` - The starting directory for relative paths.
    ///   `None` means use the global root.
    ///
    /// # Returns
    ///
    /// Returns `Ok(())` on success, `Err(FileSystemError::NotFound)` if
    /// `old_path` does not exist, `Err(FileSystemError::AlreadyExists)` if
    /// `new_path` already exists, `Err(FileSystemError::InvalidPath)` if the
    /// two paths are on different filesystems, or other `Err(FileSystemError)`
    /// variants for resolution and I/O failures.
    pub fn rename(
        &mut self,
        old_path: &str,
        new_path: &str,
        working_directory: Option<Arc<Directory>>,
    ) -> Result<(), FileSystemError> {
        let (old_parent_path, old_name) = split_parent_and_name(old_path)?;
        let (new_parent_path, new_name) = split_parent_and_name(new_path)?;

        let old_parent = self.resolve_dir_with_budget(
            old_parent_path,
            working_directory.clone(),
            super::resolver::MAX_SYMLINK_DEPTH,
        )?;

        let new_parent = self.resolve_dir_with_budget(
            new_parent_path,
            working_directory,
            super::resolver::MAX_SYMLINK_DEPTH,
        )?;

        // Reject cross-filesystem renames.
        if old_parent.inode.filesystem.instance_id()
            != new_parent.inode.filesystem.instance_id()
        {
            return Err(FileSystemError::InvalidPath);
        }

        // Reject if the destination name already exists.
        {
            let dir_ptr = Arc::as_ptr(&new_parent) as *mut
                super::directory::Directory;
            let dir = unsafe { &mut *dir_ptr };
            if dir.lookup_child(new_name)?.is_some() {
                return Err(FileSystemError::AlreadyExists);
            }
        }

        // Dispatch to the driver.
        old_parent.inode.filesystem.rename(
            &old_parent.inode,
            old_name,
            &new_parent.inode,
            new_name,
        )?;

        // Update the in-memory child lists. Remove the old entry from its
        // parent and add a new entry (carrying the same inode) to the new
        // parent.
        let moved_inode = self.resolve(old_path, None, false)?;

        {
            let dir_ptr = Arc::as_ptr(&old_parent) as *mut
                super::directory::Directory;
            let dir = unsafe { &mut *dir_ptr };
            dir.remove_child(old_name)?;
        }

        let entry_type = match moved_inode.inode_type {
            InodeType::File      => DirectoryEntryType::File,
            InodeType::Directory => DirectoryEntryType::Directory,
            InodeType::Symlink   => DirectoryEntryType::Symlink,
        };

        let new_entry = DirectoryEntry::with_inode(
            alloc::string::ToString::to_string(new_name),
            moved_inode.inode_number,
            entry_type,
            moved_inode,
        );

        {
            let dir_ptr = Arc::as_ptr(&new_parent) as *mut
                super::directory::Directory;
            let dir = unsafe { &mut *dir_ptr };
            dir.add_child(new_entry)?;
        }

        Ok(())
    }

    /// Updates the `accessed` timestamp on `inode` and marks it dirty.
    ///
    /// Called internally after a successful read operation. The `accessed`
    /// field is updated to the current wall-clock time. If the update causes
    /// `accessed` to change, `dirty` is set so the change will be persisted
    /// by a future sync.
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
        let inode_ptr = Arc::as_ptr(inode) as *mut super::inode::Inode;
    
        unsafe {
            (*inode_ptr).timestamps.accessed = now;
            (*inode_ptr).dirty = true;
        }
    
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
    /// Returns `Ok(())` always.
    fn touch_modified(
        &self,
        inode: &Arc<super::inode::Inode>,
        new_end: u64,
    ) -> Result<(), FileSystemError> {
        let now = current_timestamp();
        let inode_ptr = Arc::as_ptr(inode) as *mut super::inode::Inode;

        unsafe {
            (*inode_ptr).timestamps.modified = now;
            (*inode_ptr).timestamps.accessed = now;

            if new_end > (*inode_ptr).size {
                (*inode_ptr).size = new_end;
            }

            (*inode_ptr).dirty = true;
        }

        Ok(())
    }
}

/// Splits a path string into the parent directory path and the terminal
/// component name.
///
/// Given `/foo/bar/baz`, returns `("/foo/bar", "baz")`. Given `/baz`, returns
/// `("/", "baz")`. Given a relative path `foo/bar`, returns `("foo", "bar")`.
///
/// Returns [`FileSystemError::InvalidPath`] if `path` is empty, consists
/// entirely of `/` separators with no terminal name, or ends with a trailing
/// `/` (which would produce an empty terminal name).
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

    // Strip a single trailing slash if present to normalize paths like
    // "/foo/bar/". A path that is nothing but slashes has no terminal name.
    let trimmed = path.trim_end_matches('/');
    if trimmed.is_empty() {
        return Err(FileSystemError::InvalidPath);
    }

    match trimmed.rfind('/') {
        // No slash at all - relative path with a single component like "foo".
        // The parent is the working directory (represented as ".") and the
        // name is the entire trimmed string.
        None => Ok((".", trimmed)),

        // Slash at position 0 means the parent is the root "/".
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
