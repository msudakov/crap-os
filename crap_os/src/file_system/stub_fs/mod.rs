//! # Stub Filesystem
//!
//! This module provides an entirely in-memory filesystem driver used during
//! development to exercise the VFS layer before a real on-disk filesystem is
//! available. It implements [`FileSystemDriver`] and [`FileSystemInstance`]
//! with a hardcoded directory tree and no disk interaction.
//!
//! ## Hardcoded Tree
//!
//! The stub mounts the following namespace at whatever path the VFS assigns
//! it:
//!
//! ```text
//! /
//! ├── etc/
//! │   ├── config.txt        ("kernel=1\nversion=0.1\n")
//! │   └── hosts.txt         ("127.0.0.1 localhost\n")
//! ├── bin/
//! │   ├── shell             ("STUB_BINARY")
//! │   └── tools/
//! │       └── debug         ("STUB_BINARY")
//! ├── var/
//! │   └── log/
//! │       └── kernel.log    ("")
//! └── lib -> /etc
//! ```
//!
//! `kernel.log` starts empty and is used to test write-then-read behavior.
//! `lib` is a symlink to `/etc`, used to test symlink traversal.
//!
//! ## In-Memory Storage
//!
//! All file contents are stored in a `SpinLock`-protected flat table of
//! `(inode_number, Vec<u8>)` pairs inside the [`StubFsInstance`]. The
//! hardcoded content is initialized at mount time. Writes update the table
//! in place. Reads copy from the table into the caller's buffer.
//!
//! Directory structure and inode metadata are also stored in a flat table
//! of [`StubInode`] records, keyed by inode number.
//!
//! ## Inode Number Assignment
//!
//! Inode numbers are assigned statically at compile time:
//!
//! ```text
//! 1  - /               (root directory)
//! 2  - /etc
//! 3  - /etc/config.txt
//! 4  - /etc/hosts.txt
//! 5  - /bin
//! 6  - /bin/shell
//! 7  - /bin/tools
//! 8  - /bin/tools/debug
//! 9  - /var
//! 10 - /var/log
//! 11 - /var/log/kernel.log
//! 12 - /lib             (symlink -> /etc)
//! ```

pub mod stub_block_device;
pub mod vfs_tests;

extern crate alloc;

use alloc::sync::Arc;
use alloc::vec::Vec;
use alloc::string::{String, ToString};

use crate::file_system::virtual_fs::driver::{
    BlockDevice, FileSystemDriver, FileSystemInstance,
};
use crate::file_system::virtual_fs::inode::{Inode, DirectoryEntry};
use crate::file_system::virtual_fs::types::{
    DirectoryEntryType, FileSystemError, InodeType, Permissions, Timestamps,
};
use crate::spinlock::SpinLock;
use crate::globals;

// ----------------------------------------------------------------------------
// Static inode numbers
// ----------------------------------------------------------------------------

const INO_ROOT:           u64 = 1;
const INO_ETC:            u64 = 2;
const INO_ETC_CONFIG:     u64 = 3;
const INO_ETC_HOSTS:      u64 = 4;
const INO_BIN:            u64 = 5;
const INO_BIN_SHELL:      u64 = 6;
const INO_BIN_TOOLS:      u64 = 7;
const INO_BIN_TOOLS_DBG:  u64 = 8;
const INO_VAR:            u64 = 9;
const INO_VAR_LOG:        u64 = 10;
const INO_VAR_LOG_KERNEL: u64 = 11;
const INO_LIB:            u64 = 12;

/// The next inode number to assign when a new inode is created at runtime
/// (e.g., via `create_file`). Starts above the last hardcoded number.
const INO_DYNAMIC_START:  u64 = 13;

// ----------------------------------------------------------------------------
// StubInode - per-inode metadata record
// ----------------------------------------------------------------------------

/// The stub's internal per-inode metadata record.
///
/// Stored in [`StubFsInstance::inodes`], keyed by inode number. Holds
/// everything the stub needs to service VFS trait calls for a given inode:
/// its type, its parent's inode number (for directory walk-up), its
/// children's directory entries (for directories), and its symlink target
/// (for symlinks). File content is stored separately in
/// [`StubFsInstance::file_data`].
struct StubInode {
    /// The inode number uniquely identifying this record within the instance.
    inode_number: u64,

    /// The kind of filesystem object this inode represents.
    inode_type: InodeType,

    /// The inode number of this inode's parent directory.
    ///
    /// `0` for the root inode, which has no parent within this filesystem
    /// instance.
    parent_inode_number: u64,

    /// The direct children of this inode, if it is a directory.
    ///
    /// Empty for file and symlink inodes. Each entry records the child's
    /// name, inode number, and type, mirroring the
    /// [`DirectoryEntry`] type returned by the VFS driver trait.
    children: Vec<(String, u64, DirectoryEntryType)>,

    /// The target path, if this inode is a symlink.
    ///
    /// Empty string for file and directory inodes.
    symlink_target: String,

    /// The size of this inode's data in bytes.
    ///
    /// For files: the length of the content in [`StubFsInstance::file_data`].
    /// For symlinks: the byte length of [`symlink_target`].
    /// For directories: always `0`.
    ///
    /// [`symlink_target`]: StubInode::symlink_target
    size: u64,
}

impl StubInode {
    /// Constructs a new directory `StubInode`.
    ///
    /// # Arguments
    ///
    /// * `inode_number`        - The inode's unique identifier.
    /// * `parent_inode_number` - The parent directory's inode number. Pass
    ///   `0` for the root inode.
    fn new_directory(inode_number: u64, parent_inode_number: u64) -> Self {
        Self {
            inode_number,
            inode_type: InodeType::Directory,
            parent_inode_number,
            children: Vec::new(),
            symlink_target: String::new(),
            size: 0,
        }
    }

    /// Constructs a new file `StubInode` with the given initial size.
    ///
    /// # Arguments
    ///
    /// * `inode_number`        - The inode's unique identifier.
    /// * `parent_inode_number` - The parent directory's inode number.
    /// * `size`                - The initial content size in bytes.
    fn new_file(
        inode_number: u64,
        parent_inode_number: u64,
        size: u64,
    ) -> Self {
        Self {
            inode_number,
            inode_type: InodeType::File,
            parent_inode_number,
            children: Vec::new(),
            symlink_target: String::new(),
            size,
        }
    }

    /// Constructs a new symlink `StubInode` pointing to `target`.
    ///
    /// # Arguments
    ///
    /// * `inode_number`        - The inode's unique identifier.
    /// * `parent_inode_number` - The parent directory's inode number.
    /// * `target`              - The path this symlink points to.
    fn new_symlink(
        inode_number: u64,
        parent_inode_number: u64,
        target: &str,
    ) -> Self {
        let size = target.len() as u64;
        Self {
            inode_number,
            inode_type: InodeType::Symlink,
            parent_inode_number,
            children: Vec::new(),
            symlink_target: target.to_string(),
            size,
        }
    }
}

// ----------------------------------------------------------------------------
// StubFsInstance
// ----------------------------------------------------------------------------

/// The live in-memory state for a mounted stub filesystem instance.
///
/// Owns two flat tables protected by a [`SpinLock`]:
///
/// - `inodes` - all [`StubInode`] metadata records, indexed by position in
///   the `Vec` (not by inode number; use `find_inode` for lookup).
/// - `file_data` - all file and symlink content, as `(inode_number, Vec<u8>)`
///   pairs. Looked up by inode number.
///
/// A monotonic `next_inode` counter is used to assign inode numbers to
/// inodes created at runtime via `create_file`, `create_directory`, and
/// `create_symlink`.
pub struct StubFsInstance {
    /// The unique filesystem ID assigned by this driver at mount time.
    id: u64,

    /// All inode metadata records for this instance.
    inodes: SpinLock<Vec<StubInode>>,

    /// File and symlink content, keyed by inode number.
    file_data: SpinLock<Vec<(u64, Vec<u8>)>>,

    /// The next inode number to assign on dynamic creation.
    next_inode: SpinLock<u64>,
}

impl StubFsInstance {
    /// Constructs a new `StubFsInstance` with the hardcoded directory tree
    /// and file contents pre-populated.
    ///
    /// # Arguments
    ///
    /// * `id` - The unique filesystem ID assigned by this driver at mount time.
    fn new(id: u64) -> Self {
        // Build the flat inode table.
        let mut inodes: Vec<StubInode> = Vec::new();

        // Root directory
        let mut root = StubInode::new_directory(INO_ROOT, 0);
        root.children.push(("etc".to_string(), INO_ETC,  DirectoryEntryType::Directory));
        root.children.push(("bin".to_string(), INO_BIN,  DirectoryEntryType::Directory));
        root.children.push(("var".to_string(), INO_VAR,  DirectoryEntryType::Directory));
        root.children.push(("lib".to_string(), INO_LIB,  DirectoryEntryType::Symlink));
        inodes.push(root);

        // /etc
        let mut etc = StubInode::new_directory(INO_ETC, INO_ROOT);
        etc.children.push(("config.txt".to_string(), INO_ETC_CONFIG, DirectoryEntryType::File));
        etc.children.push(("hosts.txt".to_string(),  INO_ETC_HOSTS,  DirectoryEntryType::File));
        inodes.push(etc);

        // /etc/config.txt
        let config_content = b"kernel=1\nversion=0.1\n";
        inodes.push(StubInode::new_file(
            INO_ETC_CONFIG, INO_ETC, config_content.len() as u64,
        ));

        // /etc/hosts.txt
        let hosts_content = b"127.0.0.1 localhost\n";
        inodes.push(StubInode::new_file(
            INO_ETC_HOSTS, INO_ETC, hosts_content.len() as u64,
        ));

        // /bin
        let mut bin = StubInode::new_directory(INO_BIN, INO_ROOT);
        bin.children.push(("shell".to_string(), INO_BIN_SHELL, DirectoryEntryType::File));
        bin.children.push(("tools".to_string(), INO_BIN_TOOLS, DirectoryEntryType::Directory));
        inodes.push(bin);

        // /bin/shell
        let shell_content = b"STUB_BINARY";
        inodes.push(StubInode::new_file(
            INO_BIN_SHELL, INO_BIN, shell_content.len() as u64,
        ));

        // /bin/tools
        let mut tools = StubInode::new_directory(INO_BIN_TOOLS, INO_BIN);
        tools.children.push(("debug".to_string(), INO_BIN_TOOLS_DBG, DirectoryEntryType::File));
        inodes.push(tools);

        // /bin/tools/debug
        inodes.push(StubInode::new_file(
            INO_BIN_TOOLS_DBG, INO_BIN_TOOLS, shell_content.len() as u64,
        ));

        // /var
        let mut var = StubInode::new_directory(INO_VAR, INO_ROOT);
        var.children.push(("log".to_string(), INO_VAR_LOG, DirectoryEntryType::Directory));
        inodes.push(var);

        // /var/log
        let mut log = StubInode::new_directory(INO_VAR_LOG, INO_VAR);
        log.children.push(("kernel.log".to_string(), INO_VAR_LOG_KERNEL, DirectoryEntryType::File));
        inodes.push(log);

        // /var/log/kernel.log  (empty)
        inodes.push(StubInode::new_file(INO_VAR_LOG_KERNEL, INO_VAR_LOG, 0));

        // /lib -> /etc  (symlink)
        inodes.push(StubInode::new_symlink(INO_LIB, INO_ROOT, "/etc"));

        // Build the file data table.
        let mut file_data: Vec<(u64, Vec<u8>)> = Vec::new();
        file_data.push((INO_ETC_CONFIG,     config_content.to_vec()));
        file_data.push((INO_ETC_HOSTS,      hosts_content.to_vec()));
        file_data.push((INO_BIN_SHELL,      shell_content.to_vec()));
        file_data.push((INO_BIN_TOOLS_DBG,  shell_content.to_vec()));
        file_data.push((INO_VAR_LOG_KERNEL, Vec::new()));
        // Symlink target is stored in StubInode, not file_data.

        Self {
            id,
            inodes:     SpinLock::new(inodes),
            file_data:  SpinLock::new(file_data),
            next_inode: SpinLock::new(INO_DYNAMIC_START),
        }
    }

    /// Returns the current wall-clock timestamp in Unix epoch milliseconds,
    /// or `0` if the wall clock is not yet initialized.
    ///
    /// # Returns
    ///
    /// Returns the current Unix epoch milliseconds, or `0` if the wall clock
    /// is unavailable.
    fn now() -> u64 {
        unsafe {
            let hpet = globals::HPET.lock();
            let wc   = globals::WALL_CLOCK.lock();
            match (hpet.as_ref(), wc.as_ref()) {
                (Some(h), Some(w)) => w.now(h),
                _                  => 0,
            }
        }
    }

    /// Looks up the position of the [`StubInode`] with the given inode number
    /// in `inodes`, returning its index in the `Vec`.
    ///
    /// # Arguments
    ///
    /// * `inodes`       - The locked inode table to search.
    /// * `inode_number` - The inode number to find.
    ///
    /// # Returns
    ///
    /// Returns `Ok(index)` if found, or
    /// `Err(FileSystemError::NotFound)` if no inode with that number exists.
    fn find_inode_index(
        inodes: &Vec<StubInode>,
        inode_number: u64,
    ) -> Result<usize, FileSystemError> {
        inodes
            .iter()
            .position(|s| s.inode_number == inode_number)
            .ok_or(FileSystemError::NotFound)
    }

    /// Constructs a VFS [`Inode`] from a [`StubInode`] record.
    ///
    /// The returned [`Inode`]'s `filesystem` field points back to `instance`,
    /// so that the VFS can dispatch subsequent operations on this inode
    /// through the correct [`FileSystemInstance`].
    ///
    /// # Arguments
    ///
    /// * `stub`     - The stub inode record to convert.
    /// * `instance` - The filesystem instance that owns this inode.
    ///
    /// # Returns
    ///
    /// Returns a new [`Inode`] populated from `stub`.
    fn make_vfs_inode(
        stub: &StubInode,
        instance: Arc<dyn FileSystemInstance>,
    ) -> Arc<Inode> {
        let now = Self::now();
        Arc::new(Inode::new(
            stub.inode_number,
            instance,
            stub.inode_type,
            stub.size,
            Permissions::all_permissive(),
            crate::file_system::virtual_fs::types::Timestamps::new(now),
            1,
        ))
    }
}

// ----------------------------------------------------------------------------
// FileSystemInstance impl
// ----------------------------------------------------------------------------

impl FileSystemInstance for StubFsInstance {
    fn instance_id(&self) -> u64 {
        self.id
    }

    fn block_size(&self) -> u32 {
        4096
    }

    fn fs_type(&self) -> &'static str {
        "stubfs"
    }

    fn load_inode(
        &self,
        inode_number: u64,
    ) -> Result<Arc<Inode>, FileSystemError> {
        let inodes = self.inodes.lock();
        let idx    = Self::find_inode_index(&inodes, inode_number)?;
        // We need an Arc<dyn FileSystemInstance> pointing to self. We cannot
        // construct one here directly, so we store a pre-built Arc in the
        // instance and clone it. Since StubFsInstance is always heap-allocated
        // inside an Arc by the driver's mount(), we access it via a raw
        // pointer cast. This is safe because the Arc that owns self is alive
        // for the duration of this call.
        let instance_arc: Arc<dyn FileSystemInstance> = unsafe {
            let raw = self as *const StubFsInstance;
            Arc::increment_strong_count(raw);
            Arc::from_raw(raw)
        };
        Ok(Self::make_vfs_inode(&inodes[idx], instance_arc))
    }

    fn sync_inode(&self, _inode: &Inode) -> Result<(), FileSystemError> {
        // No disk to sync to; the in-memory state is always current.
        Ok(())
    }

    fn free_inode(&self, inode: &Inode) -> Result<(), FileSystemError> {
        let mut inodes = self.inodes.lock();
        let idx = Self::find_inode_index(&inodes, inode.inode_number)?;
        inodes.remove(idx);

        // Remove file data if present.
        let mut file_data = self.file_data.lock();
        if let Some(pos) = file_data
            .iter()
            .position(|(n, _)| *n == inode.inode_number)
        {
            file_data.remove(pos);
        }

        Ok(())
    }

    fn lookup(
        &self,
        directory: &Inode,
        name: &str,
    ) -> Result<DirectoryEntry, FileSystemError> {
        let inodes = self.inodes.lock();
        let idx    = Self::find_inode_index(&inodes, directory.inode_number)?;
        let dir    = &inodes[idx];

        for (child_name, child_ino, child_type) in &dir.children {
            if child_name == name {
                return Ok(DirectoryEntry::new(
                    child_name.clone(),
                    *child_ino,
                    *child_type,
                ));
            }
        }

        Err(FileSystemError::NotFound)
    }

    fn read_dir(
        &self,
        directory: &Inode,
    ) -> Result<Vec<DirectoryEntry>, FileSystemError> {
        let inodes = self.inodes.lock();
        let idx    = Self::find_inode_index(&inodes, directory.inode_number)?;
        let dir    = &inodes[idx];

        let entries = dir.children
            .iter()
            .map(|(name, ino, entry_type)| {
                DirectoryEntry::new(name.clone(), *ino, *entry_type)
            })
            .collect();

        Ok(entries)
    }

    fn read(
        &self,
        inode: &Inode,
        offset: u64,
        buffer: &mut [u8],
    ) -> Result<usize, FileSystemError> {
        let file_data = self.file_data.lock();
        let (_, data) = file_data
            .iter()
            .find(|(n, _)| *n == inode.inode_number)
            .ok_or(FileSystemError::NotFound)?;

        let start = offset as usize;
        if start >= data.len() {
            return Ok(0);
        }

        let available = &data[start..];
        let to_copy   = available.len().min(buffer.len());
        buffer[..to_copy].copy_from_slice(&available[..to_copy]);

        Ok(to_copy)
    }

    fn write(
        &self,
        inode: &Inode,
        offset: u64,
        buffer: &[u8],
    ) -> Result<usize, FileSystemError> {
        let mut file_data = self.file_data.lock();
        let (_, data) = file_data
            .iter_mut()
            .find(|(n, _)| *n == inode.inode_number)
            .ok_or(FileSystemError::NotFound)?;

        let start   = offset as usize;
        let end     = start + buffer.len();

        // Extend the backing Vec if the write reaches beyond the current end.
        if end > data.len() {
            data.resize(end, 0u8);
        }

        data[start..end].copy_from_slice(buffer);

        // Update the size field on the StubInode.
        let mut inodes = self.inodes.lock();
        if let Ok(idx) = Self::find_inode_index(&inodes, inode.inode_number) {
            if end as u64 > inodes[idx].size {
                inodes[idx].size = end as u64;
            }
        }

        Ok(buffer.len())
    }

    fn create_file(
        &self,
        parent: &Inode,
        name: &str,
        _permissions: Permissions,
        _timestamps: Timestamps,
    ) -> Result<Arc<Inode>, FileSystemError> {
        let new_ino = {
            let mut counter = self.next_inode.lock();
            let n = *counter;
            *counter += 1;
            n
        };

        {
            let mut inodes = self.inodes.lock();
            // Reject if name already exists in parent.
            let parent_idx = Self::find_inode_index(&inodes, parent.inode_number)?;
            if inodes[parent_idx].children.iter().any(|(n, _, _)| n == name) {
                return Err(FileSystemError::AlreadyExists);
            }
            inodes[parent_idx].children.push((
                name.to_string(),
                new_ino,
                DirectoryEntryType::File,
            ));
            inodes.push(StubInode::new_file(new_ino, parent.inode_number, 0));
        }

        {
            let mut file_data = self.file_data.lock();
            file_data.push((new_ino, Vec::new()));
        }

        self.load_inode(new_ino)
    }

    fn create_directory(
        &self,
        parent: &Inode,
        name: &str,
        _permissions: Permissions,
        _timestamps: Timestamps,
    ) -> Result<Arc<Inode>, FileSystemError> {
        let new_ino = {
            let mut counter = self.next_inode.lock();
            let n = *counter;
            *counter += 1;
            n
        };

        {
            let mut inodes = self.inodes.lock();
            let parent_idx = Self::find_inode_index(&inodes, parent.inode_number)?;
            if inodes[parent_idx].children.iter().any(|(n, _, _)| n == name) {
                return Err(FileSystemError::AlreadyExists);
            }
            inodes[parent_idx].children.push((
                name.to_string(),
                new_ino,
                DirectoryEntryType::Directory,
            ));
            inodes.push(StubInode::new_directory(new_ino, parent.inode_number));
        }

        self.load_inode(new_ino)
    }

    fn create_symlink(
        &self,
        parent: &Inode,
        name: &str,
        target: &str,
        _timestamps: Timestamps,
    ) -> Result<Arc<Inode>, FileSystemError> {
        let new_ino = {
            let mut counter = self.next_inode.lock();
            let n = *counter;
            *counter += 1;
            n
        };

        {
            let mut inodes = self.inodes.lock();
            let parent_idx = Self::find_inode_index(&inodes, parent.inode_number)?;
            if inodes[parent_idx].children.iter().any(|(n, _, _)| n == name) {
                return Err(FileSystemError::AlreadyExists);
            }
            inodes[parent_idx].children.push((
                name.to_string(),
                new_ino,
                DirectoryEntryType::Symlink,
            ));
            inodes.push(StubInode::new_symlink(
                new_ino,
                parent.inode_number,
                target,
            ));
        }

        self.load_inode(new_ino)
    }

    fn unlink(
        &self,
        parent: &Inode,
        name: &str,
    ) -> Result<(), FileSystemError> {
        let mut inodes = self.inodes.lock();
        let parent_idx = Self::find_inode_index(&inodes, parent.inode_number)?;

        let pos = inodes[parent_idx]
            .children
            .iter()
            .position(|(n, _, _)| n == name)
            .ok_or(FileSystemError::NotFound)?;

        inodes[parent_idx].children.remove(pos);

        Ok(())
    }

    fn rename(
        &self,
        old_parent: &Inode,
        old_name: &str,
        new_parent: &Inode,
        new_name: &str,
    ) -> Result<(), FileSystemError> {
        let mut inodes = self.inodes.lock();

        // Find and remove the entry from old_parent.
        let old_parent_idx = Self::find_inode_index(&inodes, old_parent.inode_number)?;
        let pos = inodes[old_parent_idx]
            .children
            .iter()
            .position(|(n, _, _)| n == old_name)
            .ok_or(FileSystemError::NotFound)?;

        let (_, child_ino, child_type) = inodes[old_parent_idx].children.remove(pos);

        // Add the entry to new_parent under the new name.
        let new_parent_idx = Self::find_inode_index(&inodes, new_parent.inode_number)?;

        if inodes[new_parent_idx].children.iter().any(|(n, _, _)| n == new_name) {
            // Put the entry back before returning the error so the inode
            // table stays consistent.
            inodes[old_parent_idx].children.push((
                old_name.to_string(),
                child_ino,
                child_type,
            ));
            return Err(FileSystemError::AlreadyExists);
        }

        inodes[new_parent_idx].children.push((
            new_name.to_string(),
            child_ino,
            child_type,
        ));

        // Update the child inode's parent pointer.
        if let Ok(child_idx) = Self::find_inode_index(&inodes, child_ino) {
            inodes[child_idx].parent_inode_number = new_parent.inode_number;
        }

        Ok(())
    }

    fn read_symlink(&self, inode: &Inode) -> Result<String, FileSystemError> {
        let inodes = self.inodes.lock();
        let idx    = Self::find_inode_index(&inodes, inode.inode_number)?;
        let stub   = &inodes[idx];

        if stub.inode_type != InodeType::Symlink {
            return Err(FileSystemError::NotASymlink);
        }

        Ok(stub.symlink_target.clone())
    }

    fn set_permissions(
        &self,
        _inode: &Inode,
        _permissions: Permissions,
    ) -> Result<(), FileSystemError> {
        // Permissions are not enforced in the stub; accept silently.
        Ok(())
    }

    fn set_timestamps(
        &self,
        _inode: &Inode,
        _timestamps: Timestamps,
    ) -> Result<(), FileSystemError> {
        // Timestamps are not persisted in the stub; accept silently.
        Ok(())
    }

    fn unmount(&self) -> Result<(), FileSystemError> {
        // Nothing to flush; all state is in memory and will be dropped when
        // the Arc<StubFsInstance> reference count reaches zero.
        Ok(())
    }
}

// ----------------------------------------------------------------------------
// StubFileSystemDriver
// ----------------------------------------------------------------------------

/// The stub filesystem driver.
///
/// A stateless factory that recognizes any block device as a valid stub
/// filesystem (it never inspects the device) and produces a
/// [`StubFsInstance`] pre-populated with the hardcoded directory tree.
///
/// Register this driver with the VFS at boot:
///
/// ```
/// let mut vfs = VIRTUAL_FILE_SYSTEM.lock();
/// vfs.as_mut().unwrap().register_driver(
///     Arc::new(StubFileSystemDriver)
/// );
/// ```
pub struct StubFileSystemDriver;

impl FileSystemDriver for StubFileSystemDriver {
    fn name(&self) -> &'static str {
        "stubfs"
    }

    /// Mounts a stub filesystem instance from any block device.
    ///
    /// The block device is not inspected or used; the stub is entirely
    /// in-memory. A new [`StubFsInstance`] is constructed with the hardcoded
    /// directory tree and returned as a live filesystem instance.
    ///
    /// # Arguments
    ///
    /// * `_device` - Ignored. The stub has no on-disk format.
    ///
    /// # Returns
    ///
    /// Returns `Ok(Arc<dyn FileSystemInstance>)` wrapping a new
    /// [`StubFsInstance`]. Never returns an error.
    fn mount(
        &self,
        _device: Arc<dyn BlockDevice>,
    ) -> Result<Arc<dyn FileSystemInstance>, FileSystemError> {
        // Assign a distinct ID to each stub instance. The VFS uses this ID
        // together with an inode number for mount and open-file keys.
        static NEXT_ID: crate::spinlock::SpinLock<u64> =
            crate::spinlock::SpinLock::new(1);

        let id = {
            let mut counter = NEXT_ID.lock();
            let n = *counter;
            *counter += 1;
            n
        };

        Ok(Arc::new(StubFsInstance::new(id)))
    }
}
