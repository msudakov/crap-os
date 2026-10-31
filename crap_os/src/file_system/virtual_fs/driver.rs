//! Virtual File System - Filesystem Driver Traits
//!
//! This module defines the two traits that every filesystem driver must
//! implement to plug into the VFS:
//!
//! - [`FileSystemDriver`] is the stateless factory trait. A driver implements
//!   this trait once and registers itself with the VFS via
//!   [`super::vfs::VirtualFileSystem::register_driver`]. Native drivers
//!   validate a device's format before producing a live
//!   [`FileSystemInstance`]. The in-memory stub driver ignores its device.
//!
//! - [`FileSystemInstance`] is the stateful trait representing one mounted
//!   filesystem. Mount records and VFS inodes retain instance references;
//!   namespace and I/O operations dispatch through the inode's instance.
//!   The instance owns whatever state its driver needs to service requests.
//!
//! Splitting the factory from the live instance is deliberate. A driver such
//! as a FAT driver is a single zero-state object registered once. The same
//! driver can produce multiple independent instances if several FAT volumes
//! are mounted simultaneously. The VFS never needs to know which driver
//! produced a given instance; it holds only the [`Arc<dyn FileSystemInstance>`]
//! and calls through it.
//!
//! Both traits interact with storage through the [`BlockDevice`] trait defined
//! in this module. A [`BlockDevice`] represents any addressable block storage:
//! a physical disk partition, a RAM disk, or a virtual device. Drivers
//! receive a [`BlockDevice`] reference at mount time and use it for disk I/O
//! when their format requires storage. The VFS itself never reads from or
//! writes to a block device directly.
//!
//! All read and write operations at the VFS-to-driver boundary use byte
//! offsets and byte lengths. The driver is responsible for all internal block
//! arithmetic, such as translating a byte range into the block numbers that
//! must be read or written, handling partial first and last blocks, and
//! managing block allocation. The VFS does not currently use `block_size()`;
//! a future cache manager may need the driver's logical block size.

#![allow(dead_code)]

use alloc::sync::Arc;
use alloc::vec::Vec;
use alloc::string::String;

use super::inode::{Inode, DirectoryEntry};
use super::types::{FileSystemError, Permissions, Timestamps};

/// A raw block storage device that a filesystem driver reads from and writes
/// to.
///
/// The VFS passes a [`BlockDevice`] to the driver at mount time. Drivers that
/// perform disk I/O use it for storage access; the in-memory stub ignores it.
/// The VFS never calls [`BlockDevice`] methods directly.
///
/// Reads and writes are expressed in terms of logical block numbers and a
/// byte buffer. The block size is fixed per device and must match the value
/// returned by [`FileSystemInstance::block_size`] for the filesystem mounted
/// on it. The driver is responsible for ensuring alignment between its chosen
/// block size and the device's physical sector size.
///
/// `Send + Sync` are required because block devices may be shared across
/// kernel threads in SMP context.
pub trait BlockDevice: Send + Sync {
    /// Gets the size in bytes of a single logical block on this device.
    ///
    /// This value is fixed for the lifetime of the device and must be a
    /// power of two no smaller than the device's physical sector size. All
    /// calls to [`BlockDevice::read_block`] and [`BlockDevice::write_block`]
    /// must supply buffers whose length equals this value.
    /// 
    /// # Returns
    ///
    /// Returns the block size in bytes.
    fn block_size(&self) -> u32;

    /// Gets the total number of addressable logical blocks on this device.
    ///
    /// Block numbers passed to [`BlockDevice::read_block`] and
    /// [`BlockDevice::write_block`] must be strictly less than this value.
    ///
    /// # Returns
    ///
    /// Returns the total block count.
    fn block_count(&self) -> u64;

    /// Reads one logical block into `buffer`.
    ///
    /// `buffer` must be exactly [`BlockDevice::block_size`] bytes long. The
    /// driver fills it with the raw bytes of the block at `block_number`. If
    /// the device reports a read error, the contents of `buffer` are undefined.
    ///
    /// # Arguments
    ///
    /// * `block_number` - The zero-based logical block address to read; must
    ///   be less than [`BlockDevice::block_count`].
    /// * `buffer`       - The destination buffer; must be exactly
    ///   [`BlockDevice::block_size`] bytes long.
    ///
    /// # Returns
    ///
    /// Returns `Ok(())` on success, or `Err(FileSystemError::DiskReadError)`
    /// if the device reports a hardware error.
    fn read_block(
        &self,
        block_number: u64,
        buffer: &mut [u8],
    ) -> Result<(), FileSystemError>;

    /// Writes one logical block from `buffer` to the device.
    ///
    /// `buffer` must be exactly [`BlockDevice::block_size`] bytes long. The
    /// driver writes its contents to the block at `block_number`. If the device
    /// reports a write error, the on-disk state of that block is undefined.
    ///
    /// # Arguments
    ///
    /// * `block_number` - The zero-based logical block address to write; must
    ///   be less than [`BlockDevice::block_count`].
    /// * `buffer`       - The source buffer; must be exactly
    ///   [`BlockDevice::block_size`] bytes long.
    ///
    /// # Returns
    ///
    /// Returns `Ok(())` on success, or `Err(FileSystemError::DiskWriteError)`
    /// if the device reports a hardware error.
    fn write_block(
        &self,
        block_number: u64,
        buffer: &[u8],
    ) -> Result<(), FileSystemError>;
}

/// A stateless factory that produces a live [`FileSystemInstance`] from a
/// block device. Native implementations validate their on-disk format.
///
/// Implementations are registered with the VFS through
/// [`super::vfs::VirtualFileSystem::register_driver`]. The VFS calls
/// [`FileSystemDriver::mount`] when a caller requests that a volume be mounted
/// under a given driver name.
///
/// A single [`FileSystemDriver`] implementation may produce any number of
/// independent [`FileSystemInstance`] values if multiple volumes of the same
/// format are mounted simultaneously. The driver itself must be stateless;
/// all per-mount state lives in the returned instance.
///
/// `Send + Sync` are required because the driver registry is a global
/// structure accessible from any kernel context.
pub trait FileSystemDriver: Send + Sync {
    /// Returns the short identifying name for this driver.
    ///
    /// This name is used as the key in the VFS driver registry and must be
    /// unique across all registered drivers. It should be a lowercase ASCII
    /// string with no whitespace, e.g., `fat32`, `ext2`, etc.
    ///
    /// # Returns
    ///
    /// Returns a static string slice identifying this driver.
    fn name(&self) -> &'static str;

    /// Attempts to mount a filesystem of this driver's format from the given
    /// block device.
    ///
    /// A native driver reads and validates the device's superblock or
    /// equivalent on-disk structure, initializes the state needed to service
    /// requests, and returns a live [`FileSystemInstance`]. The test-only
    /// in-memory stub accepts any device without inspecting it.
    ///
    /// A native driver returns [`FileSystemError::InvalidFormat`] without
    /// modifying a device that has a different format. It should return
    /// [`FileSystemError::CorruptStructure`] for corrupt format structures.
    ///
    /// # Arguments
    ///
    /// * `device` - The block device to mount. A driver that performs disk
    ///   I/O retains it for the lifetime of the returned instance.
    ///
    /// # Returns
    ///
    /// Returns an [`Arc<dyn FileSystemInstance>`] representing the live mounted
    /// volume, or `Err(FileSystemError)` if the device could not be mounted.
    fn mount(
        &self,
        device: Arc<dyn BlockDevice>,
    ) -> Result<Arc<dyn FileSystemInstance>, FileSystemError>;
}

/// A live mounted filesystem instance that services VFS namespace and I/O
/// operations.
///
/// Mount records and VFS inodes retain [`Arc<dyn FileSystemInstance>`]
/// references, and VFS operations dispatch through this trait. The
/// implementation owns the per-instance state it needs, such as a native
/// format's superblock, allocation structures, and [`BlockDevice`] reference.
///
/// Methods are grouped by operation type below. A driver must keep its backing
/// state consistent when a mutation succeeds or reports an error. Native
/// drivers must also account for failures during persistent writes.
///
/// `Send + Sync` are required because the VFS may call into instances from
/// different kernel threads in SMP context.
pub trait FileSystemInstance: Send + Sync {

    // =========================================================================
    // Filesystem Metadata
    // =========================================================================

    /// Gets a unique identifier for this mounted instance.
    ///
    /// The driver assigns this ID when creating the instance. It must differ
    /// from every other live filesystem instance's ID. The VFS uses it as the
    /// `filesystem_id` component of [`super::types::OpenFileKey`] to
    /// distinguish inodes on this volume from identically numbered inodes on
    /// other mounted volumes.
    ///
    /// # Returns
    ///
    /// Returns the unique instance identifier assigned by the driver.
    fn instance_id(&self) -> u64;

    /// Gets the size in bytes of one logical block on this filesystem.
    ///
    /// Must equal the block size of the underlying [`BlockDevice`] and
    /// remain constant for the lifetime of the instance. The VFS does not
    /// currently query this value, but a future cache manager may use it.
    ///
    /// # Returns
    ///
    /// Returns the block size in bytes.
    fn block_size(&self) -> u32;

    /// Gets the short identifying name of the filesystem format.
    ///
    /// Should match the name returned by the [`FileSystemDriver`] that
    /// produced this instance, e.g., `fat32`, `ext2`, etc.
    ///
    /// # Returns
    ///
    /// Returns a static string slice naming the filesystem format.
    fn fs_type(&self) -> &'static str;

    // =========================================================================
    // Inode Operations
    // =========================================================================

    /// Loads the inode identified by `inode_number` into a VFS record.
    ///
    /// The driver reads its backing inode record and populates a VFS-layer
    /// [`Inode`] with the generic metadata fields (type, size, permissions,
    /// timestamps, link count), and returns it wrapped in an [`Arc`]. Any
    /// driver-private per-inode state is stored separately within the instance
    /// and keyed by `inode_number`.
    ///
    /// The returned [`Inode`]'s `filesystem` field must be set to an
    /// [`Arc<dyn FileSystemInstance>`] pointing back to this instance, so that
    /// subsequent VFS operations on the inode can dispatch through it without
    /// an external lookup. The driver does not need to return the same
    /// allocation on repeated calls. The VFS keeps one in-memory [`Inode`] per
    /// filesystem instance and inode number while that inode remains
    /// registered.
    ///
    /// # Arguments
    ///
    /// * `inode_number` - The inode's unique identifier within this filesystem
    ///   instance.
    ///
    /// # Returns
    ///
    /// Returns `Ok(Arc<Inode>)` on success, or `Err(FileSystemError)` if the
    /// inode number is unknown or the backing record cannot be read.
    fn load_inode(
        &self,
        inode_number: u64,
    ) -> Result<Arc<Inode>, FileSystemError>;

    /// Synchronizes the current in-memory state of `inode` with driver storage.
    ///
    /// The driver can obtain a consistent snapshot of the generic metadata
    /// (size, permissions, timestamps, and link count) with `inode.stat()` and
    /// persist it in its backing record. The in-memory stub has nothing to
    /// persist. The VFS does not hold the inode's metadata lock across driver
    /// calls. It clears the dirty flag only after this method succeeds during
    /// volume unmount.
    ///
    /// # Arguments
    ///
    /// * `inode` - The inode whose metadata should be persisted. The driver
    ///   uses `inode.inode_number` to locate its backing record.
    ///
    /// # Returns
    ///
    /// Returns `Ok(())` on success, or a driver error if synchronization
    /// fails.
    fn sync_inode(
        &self,
        inode: &Inode,
    ) -> Result<(), FileSystemError>;

    /// Frees the inode slot and all data blocks belonging to `inode`.
    ///
    /// Called by the VFS after unlink brings an inode's link count to zero.
    /// With no open handles, the VFS calls it immediately. Otherwise it waits
    /// for the last close. On success, the driver releases the inode slot and
    /// its data blocks.
    ///
    /// On deferred deletion, the final handle remains registered during this
    /// call so a failure can leave it open for a retry. The global VFS lock
    /// prevents a concurrent handle operation while storage is being freed.
    /// After success, the VFS closes the handle and forgets the inode.
    /// A failed immediate free has no VFS retry path yet.
    ///
    /// # Arguments
    ///
    /// * `inode` - The inode to free. The driver uses `inode.inode_number`
    ///   to locate its backing record and associated data.
    ///
    /// # Returns
    ///
    /// Returns `Ok(())` on success, or a driver error if deallocation fails.
    fn free_inode(
        &self,
        inode: &Inode,
    ) -> Result<(), FileSystemError>;

    // =========================================================================
    // Directory Operations
    // =========================================================================

    /// Looks up a single child by name within a directory inode.
    ///
    /// The driver searches the directory identified by `directory` for an
    /// entry whose name exactly matches `name`, and returns a
    /// [`DirectoryEntry`] containing its name, inode number, and type. The
    /// entry does not hold an inode reference. When traversal needs the full
    /// inode, the VFS acquires its shared inode by number, loading it through
    /// [`FileSystemInstance::load_inode`] if it is not already retained.
    /// The current resolver uses [`FileSystemInstance::read_dir`] for child
    /// lookups rather than calling this method.
    ///
    /// The special names `.` and `..` are handled entirely by the VFS path
    /// resolver and must not be stored as real entries on disk. Drivers must
    /// not return them from this method.
    ///
    /// # Arguments
    ///
    /// * `directory` - The directory inode to search within.
    /// * `name`      - The child name to look up. A single path component
    ///   with no `/` separators.
    ///
    /// # Returns
    ///
    /// Returns `Ok(DirectoryEntry)` if the name was found, or
    /// `Err(FileSystemError::NotFound)` if no matching entry exists. Returns
    /// other `Err(FileSystemError)` variants for I/O or structural errors.
    fn lookup(
        &self,
        directory: &Inode,
        name: &str,
    ) -> Result<DirectoryEntry, FileSystemError>;

    /// Returns all children of a directory inode as a list of directory
    /// entries.
    ///
    /// The driver reads the complete contents of the directory identified by
    /// `directory` and returns one [`DirectoryEntry`] per child. Each entry
    /// supplies a name, inode number, and type. Directory wrappers do not
    /// retain this list: the VFS reads a fresh listing for each child lookup
    /// and loads the child's inode separately when needed.
    ///
    /// The special entries `.` and `..` must not appear in the returned list;
    /// the VFS synthesizes them during path resolution.
    ///
    /// # Arguments
    ///
    /// * `directory` - The directory inode whose children should be listed.
    ///
    /// # Returns
    ///
    /// Returns `Ok(Vec<DirectoryEntry>)` containing one entry per child on
    /// success, or a driver error if the directory's backing state cannot be
    /// read.
    fn read_dir(
        &self,
        directory: &Inode,
    ) -> Result<Vec<DirectoryEntry>, FileSystemError>;

    // =========================================================================
    // File I/O
    // =========================================================================

    /// Reads up to `buffer.len()` bytes from `inode` starting at `offset`.
    ///
    /// The driver translates the byte range `[offset, offset + buffer.len())`
    /// into the appropriate block reads, handles partial first and last blocks,
    /// and copies the requested bytes into `buffer`. If the requested range
    /// extends beyond the end of the file, the driver reads only up to the
    /// end and returns the actual number of bytes placed in `buffer`.
    ///
    /// # Arguments
    ///
    /// * `inode`  - The file inode to read from; must have `inode_type`
    ///   [`super::types::InodeType::File`].
    /// * `offset` - The byte offset within the file at which to begin reading.
    /// * `buffer` - The destination buffer. The driver fills as much of this
    ///   as the file's remaining content allows.
    ///
    /// # Returns
    ///
    /// Returns `Ok(usize)` with the number of bytes actually read and placed
    /// in `buffer`, which may be less than `buffer.len()` if the read reached
    /// the end of the file. Returns `Err(FileSystemError)` on I/O error.
    fn read(
        &self,
        inode: &Inode,
        offset: u64,
        buffer: &mut [u8],
    ) -> Result<usize, FileSystemError>;

    /// Writes `buffer.len()` bytes to `inode` starting at `offset`.
    ///
    /// The driver translates the byte range `[offset, offset + buffer.len())`
    /// into the appropriate block writes, allocating new blocks as needed if
    /// the write extends beyond the file's current size. The driver updates
    /// its stored file length if the write extends the file.
    ///
    /// The VFS updates the inode's protected in-memory size and dirty flag
    /// after a successful write; the driver need not modify them directly.
    ///
    /// # Arguments
    ///
    /// * `inode`  - The file inode to write to; must have `inode_type`
    ///   [`super::types::InodeType::File`].
    /// * `offset` - The byte offset within the file at which to begin writing.
    /// * `buffer` - The source buffer whose entire contents are written.
    ///
    /// # Returns
    ///
    /// Returns `Ok(usize)` with the number of bytes actually written, which
    /// must equal `buffer.len()` on success. Returns `Err(FileSystemError)`
    /// on I/O error or if the filesystem has no free blocks remaining.
    fn write(
        &self,
        inode: &Inode,
        offset: u64,
        buffer: &[u8],
    ) -> Result<usize, FileSystemError>;

    // =========================================================================
    // Namespace Mutation
    // =========================================================================

    /// Creates a new regular file in the given parent directory.
    ///
    /// The driver allocates a new inode of type
    /// [`super::types::InodeType::File`], creates a backing record with the
    /// supplied metadata, and adds a directory entry for `name` in `parent`.
    /// The new inode's link count starts at `1`.
    ///
    /// # Arguments
    ///
    /// * `parent`      - The directory inode in which to create the new file.
    /// * `name`        - The name of the new file within `parent`; must not
    ///   already exist in the directory.
    /// * `permissions` - The initial access-control permissions for the new
    ///   inode.
    /// * `timestamps`  - The creation, modification, and access timestamps
    ///   for the new inode, provided by the VFS from the wall clock.
    ///
    /// # Returns
    ///
    /// Returns `Ok(Arc<Inode>)` for the newly created file inode on success,
    /// or `Err(FileSystemError)` if the name already exists, the filesystem
    /// has no free inode or block slots, or a disk write fails.
    fn create_file(
        &self,
        parent: &Inode,
        name: &str,
        permissions: Permissions,
        timestamps: Timestamps,
    ) -> Result<Arc<Inode>, FileSystemError>;

    /// Creates a new directory in the given parent directory.
    ///
    /// The driver allocates a new inode of type
    /// [`super::types::InodeType::Directory`], creates a backing record with
    /// the supplied metadata, and adds a directory entry for `name` in
    /// `parent`. The new inode's link count starts at `1`.
    ///
    /// # Arguments
    ///
    /// * `parent`      - The directory inode in which to create the new
    ///   directory.
    /// * `name`        - The name of the new directory within `parent`; must
    ///   not already exist in the directory.
    /// * `permissions` - The initial access-control permissions for the new
    ///   inode.
    /// * `timestamps`  - The creation, modification, and access timestamps
    ///   for the new inode, provided by the VFS from the wall clock.
    ///
    /// # Returns
    ///
    /// Returns `Ok(Arc<Inode>)` for the newly created directory inode on
    /// success, or `Err(FileSystemError)` if the name already exists, the
    /// filesystem has no free inode or block slots, or a disk write fails.
    fn create_directory(
        &self,
        parent: &Inode,
        name: &str,
        permissions: Permissions,
        timestamps: Timestamps,
    ) -> Result<Arc<Inode>, FileSystemError>;

    /// Creates a new symbolic link in the given parent directory.
    ///
    /// The driver allocates a new inode of type
    /// [`super::types::InodeType::Symlink`], stores `target` as the inode's
    /// data, creates a backing record with the supplied metadata, and adds a
    /// directory entry for `name` in `parent`. The new inode's link count
    /// starts at `1`.
    ///
    /// # Arguments
    ///
    /// * `parent`     - The directory inode in which to create the symlink.
    /// * `name`       - The name of the symlink within `parent`; must not
    ///   already exist in the directory.
    /// * `target`     - The target path string stored as the symlink's data.
    ///   May be absolute or relative; the VFS path resolver interprets it
    ///   during traversal.
    /// * `timestamps` - The creation, modification, and access timestamps
    ///   for the new inode, provided by the VFS from the wall clock.
    ///
    /// # Returns
    ///
    /// Returns `Ok(Arc<Inode>)` for the newly created symlink inode on
    /// success, or `Err(FileSystemError)` if the name already exists, the
    /// filesystem has no free inode or block slots, or a disk write fails.
    fn create_symlink(
        &self,
        parent: &Inode,
        name: &str,
        target: &str,
        timestamps: Timestamps,
    ) -> Result<Arc<Inode>, FileSystemError>;

    /// Removes a directory entry from its parent directory.
    ///
    /// The driver removes the entry named `name` from the directory identified
    /// by `parent`. A native driver also decrements the target's persistent
    /// link count. After success, the VFS decrements its retained inode's
    /// in-memory link count. The driver must leave the inode's storage intact;
    /// the VFS calls [`FileSystemInstance::free_inode`] separately when the
    /// count reaches zero.
    ///
    /// The name disappears immediately, while any existing open handles can
    /// access the data until the final close frees the inode.
    ///
    /// # Arguments
    ///
    /// * `parent` - The directory inode containing the entry to remove.
    /// * `name`   - The name of the entry to remove from `parent`.
    ///
    /// # Returns
    ///
    /// Returns `Ok(())` on success, or `Err(FileSystemError)` if the entry
    /// does not exist, `parent` is not a directory, or a disk write fails.
    fn unlink(
        &self,
        parent: &Inode,
        name: &str,
    ) -> Result<(), FileSystemError>;

    /// Moves a directory entry from one parent directory to another, optionally
    /// changing its name.
    ///
    /// The driver removes the entry named `old_name` from `old_parent`,
    /// creates a new entry named `new_name` in `new_parent` pointing to the
    /// same inode, and updates the driver's backing namespace. The VFS
    /// preflights both names, but does not roll back a partial driver mutation.
    /// `old_parent` and `new_parent` may refer to the same directory (i.e., an
    /// in-place rename).
    ///
    /// If an entry named `new_name` already exists in `new_parent`, the
    /// driver returns [`FileSystemError::AlreadyExists`]. The current VFS
    /// also rejects an existing destination; overwriting rename is not
    /// implemented.
    ///
    /// # Arguments
    ///
    /// * `old_parent` - The directory inode currently containing the entry.
    /// * `old_name`   - The current name of the entry within `old_parent`.
    /// * `new_parent` - The directory inode that will contain the entry after
    ///   the rename.
    /// * `new_name`   - The new name of the entry within `new_parent`.
    ///
    /// # Returns
    ///
    /// Returns `Ok(())` on success, or `Err(FileSystemError)` if the source
    /// entry does not exist, the destination name already exists, or a disk
    /// write fails.
    fn rename(
        &self,
        old_parent: &Inode,
        old_name: &str,
        new_parent: &Inode,
        new_name: &str,
    ) -> Result<(), FileSystemError>;

    // =========================================================================
    // Symlink Operations
    // =========================================================================

    /// Reads and returns the target path stored in a symlink inode.
    ///
    /// The driver reads the data of the inode identified by `inode`, which
    /// must have `inode_type` [`super::types::InodeType::Symlink`], and
    /// returns it as a [`String`]. The VFS path resolver uses this string to
    /// continue resolution after encountering a symlink component.
    ///
    /// # Arguments
    ///
    /// * `inode` - The symlink inode whose target path should be read; must
    ///   have `inode_type` [`super::types::InodeType::Symlink`].
    ///
    /// # Returns
    ///
    /// Returns `Ok(String)` containing the symlink's target path on success,
    /// or `Err(FileSystemError)` if `inode` is not a symlink or a disk read
    /// fails.
    fn read_symlink(
        &self,
        inode: &Inode,
    ) -> Result<String, FileSystemError>;

    // =========================================================================
    // Metadata
    // =========================================================================

    /// Replaces the permissions on the inode identified by `inode`.
    ///
    /// A native driver updates the inode's stored permission record with the
    /// supplied `permissions` value. A VFS caller must update the protected
    /// in-memory metadata after this method succeeds. The current VFS has no
    /// public permission-change operation.
    ///
    /// # Arguments
    ///
    /// * `inode`       - The inode whose permissions should be updated. The
    ///   driver uses `inode.inode_number` to locate its backing record.
    /// * `permissions` - The new permissions to apply.
    ///
    /// # Returns
    ///
    /// Returns `Ok(())` on success, or `Err(FileSystemError)` if a disk write
    /// fails.
    fn set_permissions(
        &self,
        inode: &Inode,
        permissions: Permissions,
    ) -> Result<(), FileSystemError>;

    /// Replaces the timestamps on the inode identified by `inode`.
    ///
    /// A native driver updates the inode's stored timestamp fields with the
    /// supplied `timestamps` value. A VFS caller must update the protected
    /// in-memory metadata after this method succeeds. The current VFS has no
    /// public timestamp-change operation.
    ///
    /// # Arguments
    ///
    /// * `inode`      - The inode whose timestamps should be updated. The
    ///   driver uses `inode.inode_number` to locate its backing record.
    /// * `timestamps` - The new timestamps to apply, in Unix epoch
    ///   milliseconds.
    ///
    /// # Returns
    ///
    /// Returns `Ok(())` on success, or `Err(FileSystemError)` if a disk write
    /// fails.
    fn set_timestamps(
        &self,
        inode: &Inode,
        timestamps: Timestamps,
    ) -> Result<(), FileSystemError>;

    // =========================================================================
    // Mount Lifecycle
    // =========================================================================

    /// Flushes all pending writes and releases resources held by this
    /// filesystem instance.
    ///
    /// Called by the VFS during unmount, after checking for open handles and
    /// syncing dirty VFS inodes. The driver must flush pending writes and
    /// metadata it owns before releasing resources needed for filesystem
    /// operations. On an error, the instance remains mounted and usable so the
    /// VFS can retry unmount. On success, the VFS removes the mount record and
    /// its retained inodes; other external inode references may remain.
    ///
    /// # Returns
    ///
    /// Returns `Ok(())` after a successful flush and teardown, or
    /// `Err(FileSystemError)` if flushing fails. On error, the driver must
    /// retain the state needed for normal operations and another unmount call.
    fn unmount(&self) -> Result<(), FileSystemError>;
}
