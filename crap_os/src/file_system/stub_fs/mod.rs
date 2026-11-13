//! # Stub Filesystem
//!
//! This module provides an entirely in-memory filesystem driver used during
//! development to exercise the VFS layer before a real on-disk filesystem is
//! available. It implements [`FileSystemDriver`] and [`FileSystemInstance`]
//! with a hardcoded directory tree and no disk interaction.
//! The VFS test harness can inject one-shot unmount, inode-free, staged
//! creation, and staged rename errors to check retry behavior without changing
//! backing state on failure. Namespace protection is enforced by the VFS, not
//! this driver.
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

/// One-shot unmount failure for the instance selected by the VFS test harness.
static FAIL_UNMOUNT_ONCE_FOR: SpinLock<Option<u64>> = SpinLock::new(None);

/// Makes the selected stub instance fail its next unmount without changing its state.
///
/// # Arguments
///
/// * `instance_id` - Identifier of the mounted stub instance to target.
fn fail_next_unmount_for(instance_id: u64) {
    *FAIL_UNMOUNT_ONCE_FOR.lock() = Some(instance_id);
}

/// One-shot free failure keyed by filesystem instance and inode number.
/// Untargeted frees leave the request armed, so independent mounts cannot
/// accidentally consume another test's failure.
static FAIL_FREE_ONCE_FOR: SpinLock<Option<(u64, u64)>> = SpinLock::new(None);

/// One-shot write failure keyed by filesystem instance and inode number.
static FAIL_WRITE_ONCE_FOR: SpinLock<Option<(u64, u64)>> = SpinLock::new(None);

/// Arms a failure before the next write changes the selected backing file.
///
/// The serial boot harness consumes the request before its fixture ends.
///
/// # Arguments
///
/// * `inode` - The backing inode whose next write should fail.
fn fail_next_write_for(inode: &Inode) {
    *FAIL_WRITE_ONCE_FOR.lock() =
        Some((inode.filesystem.instance_id(), inode.inode_number));
}

/// One-shot truncation failure keyed by filesystem instance and inode number.
static FAIL_TRUNCATE_ONCE_FOR: SpinLock<Option<(u64, u64)>> = SpinLock::new(None);

/// Arms a truncation failure before any backing data is changed.
///
/// The serial boot harness must consume the request before its fixture ends.
///
/// # Arguments
///
/// * `inode` - The backing inode whose next truncation should fail.
fn fail_next_truncate_for(inode: &Inode) {
    *FAIL_TRUNCATE_ONCE_FOR.lock() =
        Some((inode.filesystem.instance_id(), inode.inode_number));
}

/// One-shot regular-file creation failure after validation and before commit.
static FAIL_CREATE_ONCE_FOR: SpinLock<Option<u64>> = SpinLock::new(None);

/// Arms a staged regular-file creation failure for one stub instance.
///
/// The serial boot harness must consume the request before its fixture ends.
///
/// # Arguments
///
/// * `instance_id` - Mounted instance whose next valid creation fails.
fn fail_next_create_for(instance_id: u64) {
    *FAIL_CREATE_ONCE_FOR.lock() = Some(instance_id);
}

/// Arms a single backing-storage free failure without modifying the target.
///
/// The boot harness runs these injections serially. Arm only one request at
/// a time and ensure the targeted free is attempted before the fixture ends.
///
/// # Arguments
///
/// * `instance_id` - Mounted stub instance that owns the target inode.
/// * `inode_number` - Inode whose next free should return `DiskWriteError`.
fn fail_next_free_for(instance_id: u64, inode_number: u64) {
    *FAIL_FREE_ONCE_FOR.lock() = Some((instance_id, inode_number));
}

/// One-shot rename failure for a mounted instance, consumed after staging and
/// before commit. Same-inode no-ops do not consume it because they do no work.
static FAIL_RENAME_ONCE_FOR: SpinLock<Option<u64>> = SpinLock::new(None);

/// Arms one pre-commit rename error without changing backing state.
///
/// # Arguments
///
/// * `instance_id` - Mounted instance whose next actual rename should fail.
fn fail_next_rename_for(instance_id: u64) {
    *FAIL_RENAME_ONCE_FOR.lock() = Some(instance_id);
}

// ----------------------------------------------------------------------------
// StubInode - per-inode metadata record
// ----------------------------------------------------------------------------

/// The stub's internal per-inode metadata record.
///
/// Stored in [`StubFsInstance::inodes`], keyed by inode number. Holds
/// everything the stub needs to service VFS trait calls for a given inode:
/// its type, its backing link count, its
/// children's directory entries (for directories), and its symlink target
/// (for symlinks). File content is stored separately in
/// [`StubFsInstance::file_data`].
struct StubInode {
    /// The inode number uniquely identifying this record within the instance.
    inode_number: u64,

    /// The kind of filesystem object this inode represents.
    inode_type: InodeType,

    /// Number of directory entries naming this inode. The root starts at one
    /// as the stub's namespace anchor. Link and unlink update this backing
    /// count independently of the VFS's canonical in-memory metadata.
    link_count: u32,

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
    /// Constructs an empty directory record with one namespace link.
    ///
    /// Parent traversal belongs to the VFS directory wrappers and relocated
    /// parent table. The stub stores child entries without duplicating that
    /// parent relationship in an unused backing field.
    ///
    /// # Arguments
    ///
    /// * `inode_number` - Unique inode number within this instance.
    ///
    /// # Returns
    ///
    /// Returns a directory record with an empty child list and zero data size.
    fn new_directory(inode_number: u64) -> Self {
        Self {
            inode_number,
            inode_type: InodeType::Directory,
            link_count: 1,
            children: Vec::new(),
            symlink_target: String::new(),
            size: 0,
        }
    }

    /// Constructs regular-file metadata with one namespace link.
    ///
    /// Contents are stored separately in the instance's file-data table. Names
    /// in directory child lists supply all backing namespace attachments.
    ///
    /// # Arguments
    ///
    /// * `inode_number` - Unique inode number within this instance.
    /// * `size` - Initial content length in bytes, matching the backing data.
    ///
    /// # Returns
    ///
    /// Returns file metadata with the supplied size and no directory children.
    fn new_file(inode_number: u64, size: u64) -> Self {
        Self {
            inode_number,
            inode_type: InodeType::File,
            link_count: 1,
            children: Vec::new(),
            symlink_target: String::new(),
            size,
        }
    }

    /// Constructs a symlink record containing an owned copy of its target.
    ///
    /// The target stays in this record, not the regular-file data table. Its
    /// byte length becomes the initial size, and its link count starts at one.
    ///
    /// # Arguments
    ///
    /// * `inode_number` - Unique inode number within this instance.
    /// * `target` - Path text to store without resolving it.
    ///
    /// # Returns
    ///
    /// Returns a symlink record with owned target text and no child entries.
    fn new_symlink(inode_number: u64, target: &str) -> Self {
        let size = target.len() as u64;
        Self {
            inode_number,
            inode_type: InodeType::Symlink,
            link_count: 1,
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
///   the `Vec` (not by inode number; use
///   [`StubFsInstance::find_inode_index`] for lookup).
/// - `file_data` - regular-file contents as `(inode_number, Vec<u8>)` pairs,
///   looked up by inode number. Symlink targets stay in their inode records.
///
/// A monotonic `next_inode` counter is used to assign inode numbers to
/// inodes created at runtime via `create_file`, `create_directory`, and
/// `create_symlink`.
pub struct StubFsInstance {
    /// The unique filesystem ID assigned by this driver at mount time.
    id: u64,

    /// All inode metadata records for this instance.
    inodes: SpinLock<Vec<StubInode>>,

    /// Regular-file contents, keyed by inode number.
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
        let mut root = StubInode::new_directory(INO_ROOT);
        root.children.push(("etc".to_string(), INO_ETC,  DirectoryEntryType::Directory));
        root.children.push(("bin".to_string(), INO_BIN,  DirectoryEntryType::Directory));
        root.children.push(("var".to_string(), INO_VAR,  DirectoryEntryType::Directory));
        root.children.push(("lib".to_string(), INO_LIB,  DirectoryEntryType::Symlink));
        inodes.push(root);

        // /etc
        let mut etc = StubInode::new_directory(INO_ETC);
        etc.children.push(("config.txt".to_string(), INO_ETC_CONFIG, DirectoryEntryType::File));
        etc.children.push(("hosts.txt".to_string(),  INO_ETC_HOSTS,  DirectoryEntryType::File));
        inodes.push(etc);

        // /etc/config.txt
        let config_content = b"kernel=1\nversion=0.1\n";
        inodes.push(StubInode::new_file(
            INO_ETC_CONFIG, config_content.len() as u64,
        ));

        // /etc/hosts.txt
        let hosts_content = b"127.0.0.1 localhost\n";
        inodes.push(StubInode::new_file(
            INO_ETC_HOSTS, hosts_content.len() as u64,
        ));

        // /bin
        let mut bin = StubInode::new_directory(INO_BIN);
        bin.children.push(("shell".to_string(), INO_BIN_SHELL, DirectoryEntryType::File));
        bin.children.push(("tools".to_string(), INO_BIN_TOOLS, DirectoryEntryType::Directory));
        inodes.push(bin);

        // /bin/shell
        let shell_content = b"STUB_BINARY";
        inodes.push(StubInode::new_file(
            INO_BIN_SHELL, shell_content.len() as u64,
        ));

        // /bin/tools
        let mut tools = StubInode::new_directory(INO_BIN_TOOLS);
        tools.children.push(("debug".to_string(), INO_BIN_TOOLS_DBG, DirectoryEntryType::File));
        inodes.push(tools);

        // /bin/tools/debug
        inodes.push(StubInode::new_file(
            INO_BIN_TOOLS_DBG, shell_content.len() as u64,
        ));

        // /var
        let mut var = StubInode::new_directory(INO_VAR);
        var.children.push(("log".to_string(), INO_VAR_LOG, DirectoryEntryType::Directory));
        inodes.push(var);

        // /var/log
        let mut log = StubInode::new_directory(INO_VAR_LOG);
        log.children.push(("kernel.log".to_string(), INO_VAR_LOG_KERNEL, DirectoryEntryType::File));
        inodes.push(log);

        // /var/log/kernel.log  (empty)
        inodes.push(StubInode::new_file(INO_VAR_LOG_KERNEL, 0));

        // /lib -> /etc  (symlink)
        inodes.push(StubInode::new_symlink(INO_LIB, "/etc"));

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
            stub.link_count,
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
        // The returned inode needs an owning reference to this instance.
        // The private constructor is used only by mount(), which puts the
        // instance in an Arc. The caller keeps that allocation alive during
        // this method. Increment its strong count before reconstructing one
        // owned Arc from the pointer; the new Arc consumes that increment.
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

    /// Frees backing records, unless the harness armed a one-shot failure.
    ///
    /// Injection is checked before acquiring storage locks or removing any
    /// record. A simulated error preserves metadata and contents for retry.
    /// The VFS, not the driver, enforces namespace protection and lifetime.
    ///
    /// # Arguments
    ///
    /// * `inode` - Inode owned by this instance and eligible for reclamation
    ///   under the VFS lifetime contract.
    ///
    /// # Returns
    ///
    /// Returns `Ok(())` after removal, `DiskWriteError` once for the armed
    /// identity, or `NotFound` if its backing inode record does not exist.
    fn free_inode(&self, inode: &Inode) -> Result<(), FileSystemError> {
        {
            let mut fail_for = FAIL_FREE_ONCE_FOR.lock();
            if *fail_for == Some((self.id, inode.inode_number)) {
                *fail_for = None;
                return Err(FileSystemError::DiskWriteError);
            }
        }
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

        {
            let mut fail_for = FAIL_WRITE_ONCE_FOR.lock();
            if *fail_for == Some((self.id, inode.inode_number)) {
                *fail_for = None;
                return Err(FileSystemError::DiskWriteError);
            }
        }

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

    /// Clears regular-file data and sets the backing inode length to zero.
    ///
    /// The caller holds VFS serialization. All backing lookups and failure
    /// injection checks precede the mutation.
    ///
    /// # Arguments
    ///
    /// * `inode` - Regular-file inode owned by this stub instance.
    ///
    /// # Returns
    ///
    /// Returns `Ok(())` after truncating, `NotFound` if a backing record
    /// is missing, or `DiskWriteError` for the armed failure.
    fn truncate_to_zero(&self, inode: &Inode) -> Result<(), FileSystemError> {
        let mut file_data = self.file_data.lock();
        let (_, data) = file_data
            .iter_mut()
            .find(|(number, _)| *number == inode.inode_number)
            .ok_or(FileSystemError::NotFound)?;

        let mut inodes = self.inodes.lock();
        let index = Self::find_inode_index(&inodes, inode.inode_number)?;

        {
            let mut fail_for = FAIL_TRUNCATE_ONCE_FOR.lock();
            if *fail_for == Some((self.id, inode.inode_number)) {
                *fail_for = None;
                return Err(FileSystemError::DiskWriteError);
            }
        }

        *data = Vec::new();
        inodes[index].size = 0;
        Ok(())
    }

    fn create_file(
        &self,
        parent: &Inode,
        name: &str,
        _permissions: Permissions,
        _timestamps: Timestamps,
    ) -> Result<Arc<Inode>, FileSystemError> {
        let mut inodes = self.inodes.lock();
        let parent_idx = Self::find_inode_index(&inodes, parent.inode_number)?;
        if inodes[parent_idx].children.iter().any(|(n, _, _)| n == name) {
            return Err(FileSystemError::AlreadyExists);
        }

        let mut counter = self.next_inode.lock();
        let new_ino = *counter;
        let next_ino = new_ino.checked_add(1)
            .ok_or(FileSystemError::OutOfInodes)?;
        let backing = StubInode::new_file(new_ino, 0);
        let inode = Self::make_vfs_inode(
            &backing, Arc::clone(&parent.filesystem),
        );
        let entry_name = name.to_string();

        {
            let mut fail_for = FAIL_CREATE_ONCE_FOR.lock();
            if *fail_for == Some(self.id) {
                *fail_for = None;
                return Err(FileSystemError::DiskWriteError);
            }
        }

        // All returned-error checks precede publication. Allocation failure
        // follows the kernel allocator contract.
        inodes[parent_idx].children.push((
            entry_name, new_ino, DirectoryEntryType::File,
        ));
        inodes.push(backing);
        *counter = next_ino;
        drop(counter);
        drop(inodes);
        self.file_data.lock().push((new_ino, Vec::new()));

        Ok(inode)
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
            inodes.push(StubInode::new_directory(new_ino));
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
                new_ino, target,
            ));
        }

        self.load_inode(new_ino)
    }

    /// Adds a name and increments its backing count under the inode-table lock.
    ///
    /// All returned-error checks precede insertion. No inode or data record is
    /// created; the new child entry shares the existing inode and data.
    ///
    /// # Arguments
    ///
    /// * `source` - Live file or symlink inode owned by the receiving stub
    ///   instance, with a nonzero backing link count.
    /// * `destination_parent` - Live destination directory owned by the
    ///   receiving stub instance, with a nonzero backing link count.
    /// * `destination_name` - Name for the new entry within
    ///   `destination_parent`. No entry with this name may already exist;
    ///   the name must satisfy [`FileSystemInstance::link`].
    ///
    /// # Returns
    ///
    /// Returns `Ok(())` after insertion, `AlreadyExists` for an occupied name,
    /// `TooManyHardLinks` for count overflow, or an error for a missing record,
    /// an unlinked inode, or an invalid inode type. Returned errors leave the
    /// destination entry and backing count unchanged.
    fn link(
        &self,
        source: &Inode,
        destination_parent: &Inode,
        destination_name: &str,
    ) -> Result<(), FileSystemError> {
        let mut inodes = self.inodes.lock();
        let source_idx = Self::find_inode_index(&inodes, source.inode_number)?;
        let destination_parent_idx =
            Self::find_inode_index(&inodes, destination_parent.inode_number)?;

        let entry_type = match inodes[source_idx].inode_type {
            InodeType::File => DirectoryEntryType::File,
            InodeType::Symlink => DirectoryEntryType::Symlink,
            InodeType::Directory => {
                return Err(FileSystemError::OperationNotAllowed);
            }
        };
        if inodes[destination_parent_idx].inode_type != InodeType::Directory {
            return Err(FileSystemError::NotADirectory);
        }
        if inodes[source_idx].link_count == 0
            || inodes[destination_parent_idx].link_count == 0
        {
            return Err(FileSystemError::NotFound);
        }
        if inodes[destination_parent_idx].children.iter()
            .any(|(name, _, _)| name == destination_name)
        {
            return Err(FileSystemError::AlreadyExists);
        }
        let new_count = inodes[source_idx].link_count.checked_add(1)
            .ok_or(FileSystemError::TooManyHardLinks)?;

        // Allocate the entry before insertion. After it is inserted, only
        // the infallible backing-count assignment remains.
        let entry = (
            destination_name.to_string(), source.inode_number, entry_type,
        );
        inodes[destination_parent_idx].children.push(entry);
        inodes[source_idx].link_count = new_count;
        Ok(())
    }

    /// Removes one name and decrements its backing count without freeing data.
    ///
    /// The inode-table lock covers both changes. The VFS updates its own count
    /// after success and separately decides when to free the inode.
    ///
    /// # Arguments
    ///
    /// * `parent` - Directory containing the entry, owned by the receiving
    ///   stub instance.
    /// * `name` - Child entry to remove.
    ///
    /// # Returns
    ///
    /// Returns `Ok(())` after removing the entry and decrementing the count,
    /// `NotFound` for a missing record or name, or `InternalError` for a zero
    /// backing count. Returned errors leave both entry and count unchanged.
    fn unlink(
        &self,
        parent: &Inode,
        name: &str,
    ) -> Result<(), FileSystemError> {
        let mut inodes = self.inodes.lock();
        let parent_idx = Self::find_inode_index(&inodes, parent.inode_number)?;
        let pos = inodes[parent_idx].children.iter()
            .position(|(n, _, _)| n == name)
            .ok_or(FileSystemError::NotFound)?;
        let target_number = inodes[parent_idx].children[pos].1;
        let target_idx = Self::find_inode_index(&inodes, target_number)?;
        let remaining = inodes[target_idx].link_count.checked_sub(1)
            .ok_or(FileSystemError::InternalError)?;

        inodes[parent_idx].children.remove(pos);
        inodes[target_idx].link_count = remaining;
        Ok(())
    }

    /// Stages directory listings, then commits replacement under one table lock.
    ///
    /// All returned-error checks and allocations precede mutation. The trusted
    /// VFS caller enforces protection, mount, and ancestry rules. Backing type
    /// and emptiness checks also precede staging. Displaced storage remains
    /// allocated for the VFS's handle lifetime and explicit reclamation rules.
    ///
    /// # Arguments
    ///
    /// * `old_parent` - Existing source directory in this instance.
    /// * `old_name` - Source name to move.
    /// * `new_parent` - Destination directory in this instance.
    /// * `new_name` - Destination name, replacing a compatible entry if present.
    ///
    /// # Returns
    ///
    /// Returns success after commit, a backing lookup/type/count error before
    /// mutation, or one injected `DiskWriteError` after staging. Allocation
    /// failure follows the existing kernel allocator contract.
    fn rename(
        &self, old_parent: &Inode, old_name: &str,
        new_parent: &Inode, new_name: &str,
    ) -> Result<(), FileSystemError> {
        let mut inodes = self.inodes.lock();
        let old_idx = Self::find_inode_index(&inodes, old_parent.inode_number)?;
        let new_idx = Self::find_inode_index(&inodes, new_parent.inode_number)?;
        if inodes[old_idx].inode_type != InodeType::Directory
            || inodes[new_idx].inode_type != InodeType::Directory
        { return Err(FileSystemError::NotADirectory); }
        let source_pos = inodes[old_idx].children.iter()
            .position(|(name, _, _)| name == old_name)
            .ok_or(FileSystemError::NotFound)?;
        let (_, source_number, source_type) = &inodes[old_idx].children[source_pos];
        let source_number = *source_number;
        let source_type = *source_type;
        // Validate backing source existence before staging, even though no
        // separate parent record needs to be updated by this driver.
        Self::find_inode_index(&inodes, source_number)?;
        let displaced = match inodes[new_idx].children.iter()
            .find(|(name, _, _)| name == new_name)
        {
            Some((_, number, _)) if *number == source_number => return Ok(()),
            Some((_, number, _)) => {
                let idx = Self::find_inode_index(&inodes, *number)?;
                let source_directory = source_type == DirectoryEntryType::Directory;
                let destination_directory = inodes[idx].inode_type == InodeType::Directory;
                if source_directory && !destination_directory {
                    return Err(FileSystemError::NotADirectory);
                }
                if !source_directory && destination_directory {
                    return Err(FileSystemError::IsADirectory);
                }
                if destination_directory && !inodes[idx].children.is_empty() {
                    return Err(FileSystemError::DirectoryNotEmpty);
                }
                let remaining = inodes[idx].link_count.checked_sub(1)
                    .ok_or(FileSystemError::InternalError)?;
                Some((idx, remaining))
            }
            None => None,
        };
        let new_entry = (new_name.to_string(), source_number, source_type);
        let mut old_children = inodes[old_idx].children.clone();
        old_children.remove(source_pos);
        let new_children = if old_idx == new_idx {
            old_children.retain(|(name, _, _)| name != new_name);
            old_children.push(new_entry);
            None
        } else {
            let mut children = inodes[new_idx].children.clone();
            children.retain(|(name, _, _)| name != new_name);
            children.push(new_entry);
            Some(children)
        };
        {
            let mut fail_for = FAIL_RENAME_ONCE_FOR.lock();
            if *fail_for == Some(self.id) {
                *fail_for = None;
                return Err(FileSystemError::DiskWriteError);
            }
        }
        // No fallible operation follows. Backing inode indices stay valid:
        // only child vectors and metadata are changed, not the inode table.
        inodes[old_idx].children = old_children;
        if let Some(children) = new_children { inodes[new_idx].children = children; }
        if let Some((idx, remaining)) = displaced { inodes[idx].link_count = remaining; }
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

    /// Completes unmount, unless the test harness armed a one-shot failure.
    ///
    /// A simulated failure leaves this instance usable for a later retry.
    ///
    /// # Returns
    ///
    /// Returns `Err(DiskWriteError)` once for an armed instance, or `Ok(())`.
    fn unmount(&self) -> Result<(), FileSystemError> {
        // Exercise the VFS retry path without releasing any instance state.
        let mut fail_for = FAIL_UNMOUNT_ONCE_FOR.lock();
        if *fail_for == Some(self.id) {
            *fail_for = None;
            return Err(FileSystemError::DiskWriteError);
        }

        // Nothing to flush; the stub retains its in-memory state until the
        // last reference to this instance is dropped.
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
