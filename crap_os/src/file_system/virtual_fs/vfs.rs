//! Virtual File System
//!
//! This module defines [`VirtualFileSystem`], the top-level structure that
//! owns and coordinates the entire VFS namespace. It is the single entry
//! point for all filesystem operations in the kernel. Driver registration,
//! volume mounting and unmounting, path resolution, file I/O, and namespace
//! mutation all flow through it.
//!
//! The global instance is declared in `globals.rs`, and it is initialized
//! during kernel boot after the first filesystem driver is registered and the
//! boot volume is mounted.
//!
//! The [`VirtualFileSystem`] structure owns four collections:
//!
//! * `root` - the global root [`super::directory::Directory`], the `/` of the
//!   unified namespace.
//! * `drivers` - a registry of named [`super::driver::FileSystemDriver`]
//!   implementations, populated at boot.
//! * `mounts` - the primary mount table, keyed by host directory inode number,
//!   mapping each mount point to its [`super::mount::MountPoint`] object.
//! * `mounted_root_index` - a secondary index keyed by mounted-root inode
//!   number, used by the path resolver to cross mount boundaries upward during
//!   `..` traversal in O(1).
//! * `open_files` - the system-wide open-file table, mapping
//!   [`super::types::OpenFileKey`] to the [`super::file::File`] master record
//!   for every inode that currently has at least one open handle.
//! * `next_filesystem_id` - a monotonically incrementing counter used to
//!   assign a unique ID to each mounted filesystem instance.
//!
//! The [`VirtualFileSystem`] struct has no interior locking of its own. It is
//! wrapped in a [`crate::spinlock::StaticIrqSpinLock`] at the global level,
//! which serializes all access. This is sufficient for the current
//! single-threaded kernel context. TODO: finer-grained locking can be
//! introduced later without changing the public interface.

#![allow(dead_code)]

use alloc::sync::Arc;
use crate::kernel_hashmap::KernelHashMap;
use super::directory::Directory;
use super::driver::{BlockDevice, FileSystemDriver};
use super::file::File;
use super::inode::Inode;
use super::mount::MountPoint;
use super::types::{FileSystemError, OpenFileKey};

/// The top-level virtual file system structure.
///
/// Owns the global namespace root, the filesystem driver registry, the mount
/// table, and the system-wide open-file table. All kernel code that interacts
/// with files, directories, or symlinks does so through this struct, accessed
/// via the `VIRTUAL_FILE_SYSTEM` global in `globals.rs`.
///
/// See the module-level documentation for a description of each field and the
/// locking model.
pub struct VirtualFileSystem {
    /// The global root directory of the unified namespace (`/`).
    ///
    /// Every absolute path resolution starts here. Set once during
    /// initialization when the boot volume is mounted and never replaced
    /// afterwards. Relative path resolutions start from a per-process working
    /// directory, but that working directory is itself ultimately rooted here.
    pub(super) root: Arc<Directory>,

    /// The registry of named filesystem drivers.
    ///
    /// Keyed by the driver's short name string (e.g., `fat32`, etc.). Populated
    /// at kernel boot by calls to [`VirtualFileSystem::register_driver`]. The
    /// VFS looks up a driver by name when[`VirtualFileSystem::mount`] is
    /// called.
    drivers: KernelHashMap<&'static str, Arc<dyn FileSystemDriver>>,

    /// The primary mount table.
    ///
    /// Keyed by the inode number of the host directory (the directory in the
    /// parent filesystem on which a volume is mounted). Each value is the
    /// [`MountPoint`] object for that mount. Used during unmount and for
    /// mount enumeration.
    pub(super) mounts: KernelHashMap<u64, MountPoint>,

    /// A secondary index from mounted-root inode number to host directory
    /// inode number.
    ///
    /// Allows the path resolver to check if the current directory is the root
    /// of a mounted filesystem in O(1) during upward `..` traversal, without
    /// scanning the primary mount table. The value is the host directory's
    /// inode number, which can then be used to retrieve the full [`MountPoint`]
    /// from [`VirtualFileSystem::mounts`].
    pub(super) mounted_root_index: KernelHashMap<u64, u64>,

    /// The system-wide open-file table.
    ///
    /// Keyed by [`OpenFileKey`] (a combination of filesystem ID and inode
    /// number). Each value is the [`File`] master record for an inode that
    /// currently has at least one open handle somewhere in the system. Entries
    /// are inserted when the first handle to an inode is opened and removed
    /// when the last handle is closed.
    pub(super) open_files: KernelHashMap<OpenFileKey, Arc<File>>,

    /// Monotonically incrementing counter for assigning unique IDs to mounted
    /// filesystem instances.
    ///
    /// Incremented by one each time [`VirtualFileSystem::mount`] is called.
    /// The assigned ID is stored in the [`MountPoint`] and used as the
    /// `filesystem_id` component of [`OpenFileKey`] for all inodes on that
    /// volume.
    next_filesystem_id: u64,
}

impl VirtualFileSystem {
    /// Constructs a new [`VirtualFileSystem`] initialized with the given root
    /// directory.
    ///
    /// Called once during kernel boot after the boot filesystem driver has
    /// mounted the root volume and produced the root [`Directory`]. The
    /// driver registry, mount table, mounted-root index, and open-file table
    /// all start empty. The `next_filesystem_id` counter starts at `1`
    /// (reserving `0` as an invalid sentinel, consistent with the handle value
    /// convention).
    ///
    /// # Arguments
    ///
    /// * `root` - The root directory of the boot volume, representing `/` in
    ///   the unified namespace.
    pub fn new(root: Arc<Directory>) -> Self {
        Self {
            root,
            drivers:             KernelHashMap::new(),
            mounts:              KernelHashMap::new(),
            mounted_root_index:  KernelHashMap::new(),
            open_files:          KernelHashMap::new(),
            next_filesystem_id:  1,
        }
    }

    /// Registers a filesystem driver with the VFS under its declared name.
    ///
    /// The driver's name (returned by
    /// [`super::driver::FileSystemDriver::name`]) is used as the registry key.
    /// If a driver with the same name is already registered, the existing
    /// entry is replaced, and the old driver is dropped.
    ///
    /// Drivers are typically registered once during kernel boot, before any
    /// volumes are mounted.
    ///
    /// # Arguments
    ///
    /// * `driver` - The filesystem driver to register.
    pub fn register_driver(&mut self, driver: Arc<dyn FileSystemDriver>) {
        self.drivers.insert(driver.name(), driver);
    }

    /// Mounts a block device at the directory identified by `mount_path`,
    /// using the filesystem driver registered under `driver_name`.
    ///
    /// The sequence of operations is:
    ///
    /// 1. Resolve `mount_path` to a directory inode via the path resolver.
    /// 2. Verify the target is a directory and has no filesystem already
    ///    mounted on it.
    /// 3. Look up the named driver in the registry.
    /// 4. Call [`super::driver::FileSystemDriver::mount`] to produce a live
    ///    [`super::driver::FileSystemInstance`].
    /// 5. Assign the instance a unique filesystem ID.
    /// 6. Load the mounted filesystem's root inode via the new instance.
    /// 7. Wrap the root inode in a [`super::directory::Directory`] with a
    ///    weak parent reference to the host directory.
    /// 8. Set `mount_flag` on the host directory's inode.
    /// 9. Set `mount_root` on the host [`super::directory::Directory`] wrapper.
    /// 10. Insert a [`MountPoint`] into the primary mount table and the
    ///     mounted-root inode number into the secondary index.
    ///
    /// `mount_path` must refer to an existing directory. Mounting on a
    /// non-directory inode, on a path that does not exist, or on a directory
    /// that already has a filesystem mounted on it are all errors.
    ///
    /// # Arguments
    ///
    /// * `driver_name`      - The short name of the registered filesystem
    ///   driver to use (e.g., `fat32`).
    /// * `device`           - The block device containing the filesystem to
    ///   mount.
    /// * `mount_path`       - The absolute or relative path of the directory
    ///   to mount on.
    /// * `working_directory`- The working directory used as the base for
    ///   relative `mount_path` values. `None` means use the global root.
    ///
    /// # Returns
    ///
    /// Returns `Ok(())` on success, or `Err(FileSystemError)` if the path
    /// cannot be resolved, the driver is not found, the device cannot be
    /// mounted, or the target is already a mount point.
    pub fn mount(
        &mut self,
        driver_name: &'static str,
        device: Arc<dyn BlockDevice>,
        mount_path: &str,
        working_directory: Option<Arc<Directory>>,
    ) -> Result<(), FileSystemError> {
        // Resolve the mount path to its host directory. We need the Directory
        // wrapper (not just the inode), so we can set mount_root on it, but
        // resolve_to_directory gives us an Arc<Directory> directly.
        let host_dir = self.resolve_to_directory(
            mount_path,
            working_directory,
        )?;

        // Reject the mount if a filesystem is already mounted here.
        if host_dir.inode.mount_flag {
            return Err(FileSystemError::AlreadyMounted);
        }

        // Look up the requested driver.
        let driver = self.drivers
            .get(&driver_name)
            .ok_or(FileSystemError::UnknownDriver)?
            .clone();

        // Ask the driver to mount the device and produce a live instance.
        let instance = driver.mount(device)?;

        // Assign a unique filesystem ID to this instance.
        let filesystem_id = self.next_filesystem_id;
        self.next_filesystem_id += 1;

        // Load the mounted filesystem's root inode. Filesystem drivers
        // conventionally use inode number 1 for the root; if a driver uses
        // a different convention it should document this. We use 1 here as
        // the agreed boot-time contract for all drivers in this kernel.
        let root_inode = instance.load_inode(1)?;

        // Verify that the root inode is actually a directory.
        if !root_inode.is_directory() {
            return Err(FileSystemError::NotADirectory);
        }

        // Wrap the root inode in a Directory, with the host directory as
        // its parent so that .. at the mounted root crosses back up correctly.
        let mounted_root = Arc::new(Directory::new(
            root_inode,
            Arc::downgrade(&host_dir),
        ));

        // Set mount_flag on the host directory's inode. This is the fast-path
        // signal that the path resolver checks on every directory visit.
        //
        // Safety: we hold the VFS lock and have exclusive access to the inode.
        // No other thread can be reading or modifying mount_flag concurrently.
        unsafe {
            let inode_ptr = Arc::as_ptr(&host_dir.inode) as *mut Inode;
            (*inode_ptr).mount_flag = true;
        }

        // Set mount_root on the host Directory wrapper for the fast-path
        // single-pointer follow during downward traversal.
        //
        // Safety: same justification as above.
        unsafe {
            let dir_ptr = Arc::as_ptr(&host_dir) as *mut Directory;
            (*dir_ptr).mount_root = Some(Arc::clone(&mounted_root));
        }

        // Record the filesystem ID on the instance so inode loads can
        // embed it in OpenFileKey values. We achieve this by storing it in
        // the MountPoint and making it available via host_inode_number lookup.
        let host_inode_number = host_dir.inode.inode_number;
        let mounted_root_inode_number = mounted_root.inode.inode_number;

        let mount_point = MountPoint::new(
            filesystem_id,
            host_dir,
            mounted_root,
            instance,
        );

        // Insert into the primary mount table (host inode number -> MountPoint)
        // and the secondary index (mounted root inode number -> host inode
        // number) for O(1) upward boundary crossing during .. resolution.
        self.mounts.insert(host_inode_number, mount_point);
        self.mounted_root_index.insert(
            mounted_root_inode_number,
            host_inode_number,
        );

        Ok(())
    }

    /// Unmounts the filesystem mounted at the directory identified by
    /// `mount_path`.
    ///
    /// The sequence of operations is:
    ///
    /// 1. Resolve `mount_path` to a directory inode.
    /// 2. Verify a filesystem is currently mounted there.
    /// 3. Verify no open file handles exist on the mounted volume.
    /// 4. Remove the [`MountPoint`] from the primary mount table and the
    ///    mounted-root inode number from the secondary index.
    /// 5. Clear `mount_flag` on the host directory's inode and `mount_root`
    ///    on its [`super::directory::Directory`] wrapper.
    /// 6. Call [`super::driver::FileSystemInstance::unmount`] on the instance,
    ///    which flushes pending writes and releases driver resources.
    ///
    /// The `mount_path` must refer to a directory that currently has a
    /// filesystem mounted on it. Attempting to unmount a path that is not a
    /// mount point, or that has open file handles on the mounted volume, is an
    /// error.
    ///
    /// # Arguments
    ///
    /// * `mount_path`        - The absolute or relative path of the mount
    ///   point directory to unmount.
    /// * `working_directory` - The working directory used as the base for
    ///   relative `mount_path` values. `None` means use the global root.
    ///
    /// # Returns
    ///
    /// Returns `Ok(())` on success, or `Err(FileSystemError)` if the path
    /// cannot be resolved, the directory is not a mount point, open handles
    /// remain on the volume, or the driver's flush fails.
    pub fn unmount(
        &mut self,
        mount_path: &str,
        working_directory: Option<Arc<Directory>>,
    ) -> Result<(), FileSystemError> {
        // Resolve the path to the host directory inode.
        let host_dir = self.resolve_to_directory(
            mount_path,
            working_directory,
        )?;

        let host_inode_number = host_dir.inode.inode_number;

        // Verify a filesystem is actually mounted here.
        if !host_dir.inode.mount_flag {
            return Err(FileSystemError::NotMounted);
        }

        // Retrieve the MountPoint so that we can check for open handles and
        // obtain the mounted-root inode number for the secondary index.
        let mount_point = self.mounts
            .get(&host_inode_number)
            .ok_or(FileSystemError::NotMounted)?;

        let mounted_root_inode_number = mount_point.mounted_root_inode_number();
        let filesystem_id = mount_point.filesystem_id;

        // Reject the unmount if any open file handles exist on this volume.
        // We check the open-file table for any key whose filesystem_id matches.
        let has_open_files = self.open_files
            .keys()
            .any(|key| key.filesystem_id == filesystem_id);

        if has_open_files {
            return Err(FileSystemError::FileLocked);
        }

        // Remove the MountPoint from both tables before clearing the host
        // directory's flags, so that no concurrent path resolution (in an SMP
        // context) can observe a partially unmounted state.
        let mount_point = self.mounts
            .remove(&host_inode_number)
            .ok_or(FileSystemError::InternalError)?;

        self.mounted_root_index.remove(&mounted_root_inode_number);

        // Clear mount_flag on the host inode and mount_root on the host
        // Directory wrapper.
        //
        // Safety: we hold the VFS lock and have confirmed via the mount table
        // that no other context holds a reference to this mount point's state.
        unsafe {
            let inode_ptr = Arc::as_ptr(&host_dir.inode) as *mut Inode;
            (*inode_ptr).mount_flag = false;

            let dir_ptr = Arc::as_ptr(&host_dir) as *mut Directory;
            (*dir_ptr).mount_root = None;
        }

        // Call the driver's unmount to flush pending writes and release
        // resources. The MountPoint is dropped at the end of this scope,
        // which drops the Arc<dyn FileSystemInstance> and, if this was the
        // last strong reference, invokes the driver's cleanup.
        mount_point.instance.unmount()?;

        Ok(())
    }

    /// Resolves `path` to a [`Directory`] wrapper, returning an error if the
    /// resolved inode is not a directory.
    ///
    /// This is an internal helper used by [`VirtualFileSystem::mount`] and
    /// [`VirtualFileSystem::unmount`] to obtain a directory from a path string.
    /// It calls [`VirtualFileSystem::resolve`] and then verifies that the
    /// resulting inode has type [`InodeType::Directory`].
    ///
    /// # Arguments
    ///
    /// * `path`              - The path string to resolve.
    /// * `working_directory` - The working directory for relative paths.
    ///   `None` means use the global root.
    ///
    /// # Returns
    ///
    /// Returns `Ok(Arc<Directory>)` if the path resolves to a directory, or
    /// `Err(FileSystemError::NotADirectory)` if it resolves to a non-directory
    /// inode. Returns other `Err(FileSystemError)` variants for resolution
    /// failures.
    fn resolve_to_directory(
        &self,
        path: &str,
        working_directory: Option<Arc<Directory>>,
    ) -> Result<Arc<Directory>, FileSystemError> {
        let inode = self.resolve(path, working_directory.clone(), false)?;

        if !inode.is_directory() {
            return Err(FileSystemError::NotADirectory);
        }

        // The inode is a directory. To return an Arc<Directory>, we need the
        // Directory wrapper that owns this inode. We locate it by walking
        // the path again and returning the final Directory rather than its
        // inode.
        // TODO: This is a temporary approach; a future inode cache will make
        // Directory wrappers directly addressable by inode number.
        self.resolve_to_dir_wrapper(path, working_directory)
    }

    /// Resolves `path` to the [`Directory`] wrapper for the terminal
    /// component, without loading a separate inode.
    ///
    /// This internal helper mirrors the path resolver's logic but returns the
    /// [`Directory`] wrapper for the final component rather than its
    /// [`Inode`]. It is used by [`VirtualFileSystem::resolve_to_directory`]
    /// and by [`VirtualFileSystem::mount`] and [`VirtualFileSystem::unmount`],
    /// which need the wrapper to mutate `mount_flag` and `mount_root`.
    ///
    /// # Arguments
    ///
    /// * `path`              - The path string to resolve.
    /// * `working_directory` - The working directory for relative paths.
    ///   `None` means use the global root.
    ///
    /// # Returns
    ///
    /// Returns `Ok(Arc<Directory>)` for the directory at the end of `path`,
    /// or `Err(FileSystemError)` if resolution fails or the terminal component
    /// is not a directory.
    fn resolve_to_dir_wrapper(
        &self,
        path: &str,
        working_directory: Option<Arc<Directory>>,
    ) -> Result<Arc<Directory>, FileSystemError> {
        // Delegate to the full resolver implementation, which returns the
        // directory wrapper for the terminal component when the path resolves
        // to a directory. The full algorithm is in resolver.rs.
        self.resolve_dir_with_budget(
            path,
            working_directory,
            super::resolver::MAX_SYMLINK_DEPTH,
        )
    }
}
