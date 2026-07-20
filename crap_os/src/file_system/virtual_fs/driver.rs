//! Virtual File System - Filesystem Driver Traits
//!
//! This module defines the two traits that every filesystem driver must
//! implement to plug into the VFS:
//!
//! - [`FileSystemDriver`] is the stateless factory trait. A driver implements
//!   this trait once and registers itself with the VFS at boot via
//!   [`super::vfs::VirtualFileSystem::register_driver`]. Its sole
//!   responsibility is recognizing a block device's on-disk format and
//!   producing a live [`FileSystemInstance`] from it.
//!
//! - [`FileSystemInstance`] is the stateful trait representing a single
//!   mounted volume. The VFS holds one [`Arc<dyn FileSystemInstance>`] per
//!   mounted filesystem and dispatches all namespace and I/O operations
//!   through it. The instance owns whatever in-memory state the driver needs
//!   to service requests: the superblock, allocation bitmaps, inode tables,
//!   and so on.
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
//! a physical disk partition, a RAM disk, or a future virtual device. Drivers
//! receive a [`BlockDevice`] reference at mount time and use it for all disk
//! I/O. The VFS itself never reads from or writes to a block device directly.
//!
//! All read and write operations at the VFS-to-driver boundary use byte
//! offsets and byte lengths. The driver is responsible for all internal block
//! arithmetic - translating a byte range into the block numbers that must be
//! read or written, handling partial first and last blocks, and managing block
//! allocation. The VFS never reasons about block boundaries or block size
//! except to query `block_size()` once at mount time for cache manager use.

use alloc::sync::Arc;
use alloc::vec::Vec;
use alloc::string::String;

use super::inode::{Inode, DirectoryEntry};
use super::types::{FileSystemError, Permissions, Timestamps};

/// A raw block storage device that a filesystem driver reads from and writes
/// to.
///
/// Every mounted filesystem receives a reference to a [`BlockDevice`] at mount
/// time. All disk I/O performed by the driver goes through this trait. The
/// VFS never calls [`BlockDevice`] methods directly; it is purely a
/// driver-to-storage interface.
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
    /// calls to [`read_block`] and [`write_block`] must supply buffers whose
    /// length equals this value.
    ///
    /// [`read_block`]: BlockDevice::read_block
    ///
    /// # Returns
    ///
    /// Returns the block size in bytes.
    fn block_size(&self) -> u32;

    /// Gets the total number of addressable logical blocks on this device.
    ///
    /// Block numbers passed to [`read_block`] and [`write_block`] must be
    /// strictly less than this value.
    ///
    /// [`read_block`]: BlockDevice::read_block
    ///
    /// # Returns
    ///
    /// Returns the total block count.
    fn block_count(&self) -> u64;

    /// Reads one logical block into `buffer`.
    ///
    /// `buffer` must be exactly [`block_size`] bytes long. The driver fills
    /// it with the raw bytes of the block at `block_number`. If the device
    /// reports a read error, the contents of `buffer` are undefined.
    ///
    /// [`block_size`]: BlockDevice::block_size
    ///
    /// # Arguments
    ///
    /// * `block_number` - The zero-based logical block address to read. Must
    ///   be less than [`block_count`].
    /// * `buffer`       - The destination buffer. Must be exactly
    ///   [`block_size`] bytes long.
    ///
    /// # Returns
    ///
    /// Returns `Ok(())` on success, or `Err(FileSystemError::DiskReadError)`
    /// if the device reports a hardware error.
    ///
    /// [`block_count`]: BlockDevice::block_count
    fn read_block(
        &self,
        block_number: u64,
        buffer: &mut [u8],
    ) -> Result<(), FileSystemError>;

    /// Writes one logical block from `buffer` to the device.
    ///
    /// `buffer` must be exactly [`block_size`] bytes long. The driver writes
    /// its contents to the block at `block_number`. If the device reports a
    /// write error, the on-disk state of that block is undefined.
    ///
    /// [`block_size`]: BlockDevice::block_size
    ///
    /// # Arguments
    ///
    /// * `block_number` - The zero-based logical block address to write. Must
    ///   be less than [`block_count`].
    /// * `buffer`       - The source buffer. Must be exactly [`block_size`]
    ///   bytes long.
    ///
    /// # Returns
    ///
    /// Returns `Ok(())` on success, or `Err(FileSystemError::DiskWriteError)`
    /// if the device reports a hardware error.
    ///
    /// [`block_count`]: BlockDevice::block_count
    fn write_block(
        &self,
        block_number: u64,
        buffer: &[u8],
    ) -> Result<(), FileSystemError>;
}

/// A stateless factory that recognizes an on-disk filesystem format and
/// produces a live [`FileSystemInstance`] from a block device.
///
/// Implementations of this trait are registered with the VFS once at kernel
/// boot via [`super::vfs::VirtualFileSystem::register_driver`] and remain
/// registered for the system's lifetime. The VFS calls [`mount`] when a
/// caller requests that a volume be mounted under a given driver name.
///
/// A single `FileSystemDriver` implementation may produce any number of
/// independent [`FileSystemInstance`] values if multiple volumes of the same
/// format are mounted simultaneously. The driver itself must be stateless;
/// all per-mount state lives in the returned instance.
///
/// `Send + Sync` are required because the driver registry is a global
/// structure accessible from any kernel context.
///
/// [`mount`]: FileSystemDriver::mount
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
    /// The driver reads the device's superblock or equivalent on-disk
    /// structure, validates the format magic number and version, initializes
    /// any in-memory state required to service subsequent requests, and
    /// returns a live [`FileSystemInstance`].
    ///
    /// If the device does not contain a filesystem of this driver's format,
    /// the driver must return [`FileSystemError::InvalidFormat`] without
    /// modifying the device. If the on-disk structures are present but
    /// corrupt, the driver should return [`FileSystemError::CorruptStructure`].
    ///
    /// # Arguments
    ///
    /// * `device` - The block device to mount. The driver retains this
    ///   reference for the lifetime of the returned instance and uses it for
    ///   all subsequent disk I/O.
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
/// The VFS holds one [`Arc<dyn FileSystemInstance>`] per mounted volume and
/// dispatches every operation on that volume through this trait. The
/// implementation owns all per-mount state: the superblock, block allocation
/// structures, open inode metadata, journal handles, and the [`BlockDevice`]
/// reference used for disk I/O.
///
/// Methods are grouped by operation type below. All methods that modify on-disk
/// state are expected to maintain the filesystem's internal consistency
/// invariants. Drivers that implement journaling or copy-on-write should
/// ensure that partial failures leave the volume in a recoverable state.
///
/// `Send + Sync` are required because the VFS may call into instances from
/// different kernel threads in SMP context.
pub trait FileSystemInstance: Send + Sync {

    // =========================================================================
    // Filesystem Metadata
    // =========================================================================

    /// Gets a unique identifier for this mounted instance.
    ///
    /// The VFS assigns this ID at mount time and uses it as the
    /// `filesystem_id` component of [`super::types::OpenFileKey`] to
    /// distinguish inodes on this volume from identically numbered inodes on
    /// other mounted volumes.
    ///
    /// # Returns
    ///
    /// Returns the unique instance identifier assigned by the VFS.
    fn instance_id(&self) -> u64;

    /// Gets the size in bytes of one logical block on this filesystem.
    ///
    /// Must equal the block size of the underlying [`BlockDevice`]. The VFS
    /// queries this value once at mount time and stores it in the
    /// [`super::mount::MountPoint`] for future cache manager use. It must
    /// remain constant for the lifetime of the instance.
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

    /// Loads the inode identified by `inode_number` from disk into memory.
    ///
    /// The driver reads the inode's on-disk record, populates a VFS-layer
    /// [`Inode`] with the generic metadata fields (type, size, permissions,
    /// timestamps, link count), and returns it wrapped in an [`Arc`]. Any
    /// driver-private per-inode state is stored separately within the instance
    /// and keyed by `inode_number`.
    ///
    /// The returned [`Inode`]'s `filesystem` field must be set to an
    /// [`Arc<dyn FileSystemInstance>`] pointing back to this instance, so that
    /// subsequent VFS operations on the inode can dispatch through it without
    /// an external lookup.
    ///
    /// # Arguments
    ///
    /// * `inode_number` - The inode's unique identifier within this filesystem
    ///   instance.
    ///
    /// # Returns
    ///
    /// Returns `Ok(Arc<Inode>)` on success, or `Err(FileSystemError)` if the
    /// inode number is out of range, the on-disk record is corrupt, or the
    /// block device reports a read error.
    fn load_inode(
        &self,
        inode_number: u64,
    ) -> Result<Arc<Inode>, FileSystemError>;

    /// Writes the current in-memory state of `inode` back to disk.
    ///
    /// The driver reads the generic metadata fields from `inode` (size,
    /// permissions, timestamps, link count) and persists them to the inode's
    /// on-disk record. The VFS calls this method whenever it clears the
    /// `dirty` flag on an inode after modifying its metadata.
    ///
    /// # Arguments
    ///
    /// * `inode` - The inode whose metadata should be persisted. The driver
    ///   uses `inode.inode_number` to locate the on-disk record.
    ///
    /// # Returns
    ///
    /// Returns `Ok(())` on success, or `Err(FileSystemError)` if the block
    /// device reports a write error.
    fn sync_inode(
        &self,
        inode: &Inode,
    ) -> Result<(), FileSystemError>;

    /// Frees the inode slot and all data blocks belonging to `inode`.
    ///
    /// Called by the VFS when an inode's link count reaches zero and all open
    /// handles to it have been closed, signaling that the inode's storage may
    /// be reclaimed. The driver must mark the inode slot as free in its
    /// allocation structure and return all data blocks to the free-block pool.
    ///
    /// This method must not be called while any open handle still holds a
    /// reference to the inode. The VFS guarantees this by deferring the call
    /// until the last [`super::file::FileHandle`] for the inode is dropped.
    ///
    /// # Arguments
    ///
    /// * `inode` - The inode to free. The driver uses `inode.inode_number`
    ///   to locate and release the on-disk record and associated data blocks.
    ///
    /// # Returns
    ///
    /// Returns `Ok(())` on success, or `Err(FileSystemError)` if the block
    /// device reports a write error during deallocation.
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
    /// entry whose name exactly matches `name`, and returns it as a
    /// [`DirectoryEntry`] with the `inode` field set to `None`. The VFS
    /// populates the `inode` field lazily via [`load_inode`] if and when the
    /// entry is traversed or opened.
    ///
    /// The special names `.` and `..` are handled entirely by the VFS path
    /// resolver and must not be stored as real entries on disk. Drivers must
    /// not return them from this method.
    ///
    /// [`load_inode`]: FileSystemInstance::load_inode
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
    /// `directory` and returns one [`DirectoryEntry`] per child, with each
    /// entry's `inode` field set to `None`. The VFS stores the returned list
    /// in the [`super::directory::Directory`] wrapper's `children` field as
    /// the result of lazy loading.
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
    /// success, or `Err(FileSystemError)` if the directory's on-disk
    /// structures could not be read.
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
    /// * `inode`  - The file inode to read from. Must have `inode_type`
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
    /// the inode's on-disk size field if the write extends the file.
    ///
    /// The VFS updates `inode.size` and sets `inode.dirty` after a successful
    /// write; the driver need not modify the in-memory `Inode` fields directly.
    ///
    /// # Arguments
    ///
    /// * `inode`  - The file inode to write to. Must have `inode_type`
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
    /// [`super::types::InodeType::File`], writes an initial on-disk record
    /// with the supplied metadata, and adds a directory entry for `name` in
    /// `parent`. The new inode's link count starts at `1`.
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
    /// [`super::types::InodeType::Directory`], writes an initial on-disk
    /// record with the supplied metadata, and adds a directory entry for
    /// `name` in `parent`. The new inode's link count starts at `1`.
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
    /// data, writes an initial on-disk record with the supplied metadata, and
    /// adds a directory entry for `name` in `parent`. The new inode's link
    /// count starts at `1`.
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

    /// Removes a directory entry from its parent directory on disk.
    ///
    /// The driver removes the entry named `name` from the directory identified
    /// by `parent` and decrements the target inode's on-disk link count. It
    /// must not free the inode's data blocks; that is done separately by
    /// [`free_inode`] once the VFS confirms that the link count has reached
    /// zero and all open handles are closed.
    ///
    /// This separation implements POSIX unlink semantics: the name disappears
    /// from the namespace immediately, but the inode's data remains accessible
    /// through any existing open handles until they are all closed.
    ///
    /// [`free_inode`]: FileSystemInstance::free_inode
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
    /// same inode, and updates any affected on-disk structures atomically
    /// where the filesystem's journaling or copy-on-write mechanism allows.
    /// `old_parent` and `new_parent` may refer to the same directory (an
    /// in-place rename).
    ///
    /// If an entry named `new_name` already exists in `new_parent`, the
    /// driver must return [`FileSystemError::AlreadyExists`]; the VFS is
    /// responsible for removing the destination first if the caller requested
    /// an overwriting rename.
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
    /// Returns `Ok(())` on success, or `Err(FileSystemError)` if either source
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
    /// * `inode` - The symlink inode whose target path should be read. Must
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
    /// The driver updates the inode's on-disk permission record with the
    /// supplied `permissions` value. The VFS sets `inode.dirty` after calling
    /// this method; a subsequent [`sync_inode`] call will persist the change
    /// to disk.
    ///
    /// [`sync_inode`]: FileSystemInstance::sync_inode
    ///
    /// # Arguments
    ///
    /// * `inode`       - The inode whose permissions should be updated. The
    ///   driver uses `inode.inode_number` to locate the on-disk record.
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
    /// The driver updates the inode's on-disk timestamp fields with the
    /// supplied `timestamps` value. The VFS sets `inode.dirty` after calling
    /// this method; a subsequent [`sync_inode`] call will persist the change
    /// to disk.
    ///
    /// [`sync_inode`]: FileSystemInstance::sync_inode
    ///
    /// # Arguments
    ///
    /// * `inode`      - The inode whose timestamps should be updated. The
    ///   driver uses `inode.inode_number` to locate the on-disk record.
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

    /// Flushes all pending writes and releases all resources held by this
    /// filesystem instance.
    ///
    /// Called by the VFS during unmount, after verifying that no open file
    /// handles remain on this volume. The driver must flush any dirty inodes
    /// or metadata to disk, release any heap allocations it owns, and drop its
    /// reference to the underlying [`BlockDevice`]. After this call returns,
    /// the VFS drops the [`Arc<dyn FileSystemInstance>`], which should be the
    /// last strong reference.
    ///
    /// # Returns
    ///
    /// Returns `Ok(())` if all pending state was successfully flushed, or
    /// `Err(FileSystemError)` if a disk write failed during flush. Even on
    /// error the instance is considered unmounted and will not receive further
    /// calls.
    fn unmount(&self) -> Result<(), FileSystemError>;
}
