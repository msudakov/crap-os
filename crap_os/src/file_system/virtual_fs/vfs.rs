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
//! The [`VirtualFileSystem`] structure owns the namespace and open-file state:
//!
//! * `root` - the global root [`super::directory::Directory`], the `/` of the
//!   unified namespace.
//! * `drivers` - a registry of named [`super::driver::FileSystemDriver`]
//!   implementations, populated at boot.
//! * `mounts` - the primary mount table, keyed by host filesystem and inode,
//!   mapping each mount point to its [`super::mount::MountPoint`] object.
//! * `mounted_root_index` - a secondary index keyed by mounted-root identity,
//!   used by the path resolver to cross mount boundaries upward during
//!   `..` traversal in O(1).
//! * `relocated_parents` - the current parents of directories moved across
//!   parent directories, keyed by filesystem and inode identity.
//! * `open_files` - the system-wide open-file table, mapping
//!   [`super::types::OpenFileKey`] to the [`super::file::File`] master record
//!   for every inode that currently has at least one open handle.
//!
//! The [`VirtualFileSystem`] struct has no interior locking of its own. It is
//! wrapped in a [`crate::spinlock::StaticIrqSpinLock`] at the global level,
//! which serializes all access. This is sufficient for the current
//! single-threaded kernel context. TODO: finer-grained locking can be
//! introduced later without changing the public interface.

#![allow(dead_code)]

use alloc::sync::Arc;
use crate::kernel_hashmap::KernelHashMap;
use super::directory::{Directory, DirectoryParent};
use super::driver::{BlockDevice, FileSystemDriver};
use super::file::File;
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
    /// Keyed by the instance ID and inode number of the host directory (the
    /// directory in the parent filesystem on which a volume is mounted). Each value is the
    /// [`MountPoint`] object for that mount. Used during unmount and for
    /// mount enumeration.
    pub(super) mounts: KernelHashMap<OpenFileKey, MountPoint>,

    /// A secondary index from mounted-root identity to host identity.
    ///
    /// Allows the path resolver to check if the current directory is the root
    /// of a mounted filesystem in O(1) during upward `..` traversal, without
    /// scanning the primary mount table. The value is the host directory's
    /// composite identity, which retrieves the full [`MountPoint`]
    /// from [`VirtualFileSystem::mounts`].
    pub(super) mounted_root_index: KernelHashMap<OpenFileKey, OpenFileKey>,

    /// Current parents of directories moved across parent directories.
    ///
    /// A retained [`Directory`] wrapper keeps the parent it had when created.
    /// The resolver checks this table before using that stored parent.
    pub(super) relocated_parents: KernelHashMap<OpenFileKey, Arc<Directory>>,

    /// The system-wide open-file table.
    ///
    /// Keyed by [`OpenFileKey`] (a combination of filesystem ID and inode
    /// number). Each value is the [`File`] master record for an inode that
    /// currently has at least one open handle somewhere in the system. Entries
    /// are inserted when the first handle to an inode is opened and removed
    /// when the last handle is closed.
    pub(super) open_files: KernelHashMap<OpenFileKey, Arc<File>>,

    /// The next VFS-wide identifier for a newly opened file handle.
    ///
    /// Each successful open receives a distinct identifier, including opens
    /// of the same inode by different processes. A failed overflow check
    /// prevents identifiers from being reused.
    pub(super) next_handle_id: u64,

}

impl VirtualFileSystem {
    /// Constructs a new [`VirtualFileSystem`] initialized with the given root
    /// directory.
    ///
    /// Called once during kernel boot after the boot filesystem driver has
    /// mounted the root volume and produced the root [`Directory`]. The
    /// driver registry, mount table, mounted-root index, and open-file table
    /// all start empty. Handle identifiers start at `1`, reserving `0` as an
    /// invalid sentinel.
    ///
    /// # Arguments
    ///
    /// * `root` - The root directory of the boot volume, representing `/` in
    ///   the unified namespace.
    ///
    /// # Returns
    ///
    /// Returns a VFS with empty tables and a fresh handle ID counter.
    pub fn new(root: Arc<Directory>) -> Self {
        Self {
            root,
            drivers:             KernelHashMap::new(),
            mounts:              KernelHashMap::new(),
            mounted_root_index:  KernelHashMap::new(),
            relocated_parents:   KernelHashMap::new(),
            open_files:          KernelHashMap::new(),
            next_handle_id:      1,
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
    /// 5. Read the driver's unique filesystem instance ID.
    /// 6. Load the mounted filesystem's root inode via the new instance.
    /// 7. Wrap the root inode in a [`super::directory::Directory`] with a
    ///    weak parent reference to the host directory.
    /// 8. Insert a [`MountPoint`] into the primary mount table and the
    ///     mounted-root identity into the secondary index.
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
        // Resolve to the covered host even when this path currently names a
        // mounted root. The table below determines whether it is occupied.
        let host_dir = self.resolve_mount_host(
            mount_path,
            working_directory,
        )?;

        // Reject the mount if a filesystem is already mounted here.
        let host_key = OpenFileKey::new(
            host_dir.inode.filesystem.instance_id(), host_dir.inode.inode_number,
        );
        if self.mounts.get(&host_key).is_some() {
            return Err(FileSystemError::AlreadyMounted);
        }

        // Look up the requested driver.
        let driver = self.drivers
            .get(&driver_name)
            .ok_or(FileSystemError::UnknownDriver)?
            .clone();

        // Ask the driver to mount the device and produce a live instance.
        let instance = driver.mount(device)?;

        // The driver supplies the instance ID used in VFS identity keys.
        let filesystem_id = instance.instance_id();

        // Load the mounted filesystem's root inode. Filesystem drivers
        // conventionally use inode number 1 for the root; if a driver uses
        // a different convention it should document this. We use 1 here as
        // the agreed boot-time contract for all drivers in this kernel.
        let root_inode = instance.load_inode(1)?;

        // Verify that the root inode is actually a directory.
        if !root_inode.is_directory() {
            return Err(FileSystemError::NotADirectory);
        }

        // The mount table retains the host. Keep the reverse parent reference
        // weak to avoid a reference cycle across the mount.
        let mounted_root = Arc::new(Directory::new_mounted_root(
            root_inode,
            &host_dir,
        ));

        // Retain the instance's ID in the mount record and index its root by
        // the same identity that inode and open-file lookups use.
        let mounted_root_key = OpenFileKey::new(
            mounted_root.inode.filesystem.instance_id(),
            mounted_root.inode.inode_number,
        );

        let mount_point = MountPoint::new(
            filesystem_id,
            host_dir,
            mounted_root,
            instance,
        );

        // Index both sides by filesystem instance and inode number. Separate
        // mounted volumes may assign the same inode numbers.
        self.mounts.insert(host_key, mount_point);
        self.mounted_root_index.insert(mounted_root_key, host_key);

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
    /// 4. Remove the [`MountPoint`] from the primary mount table, the
    ///    mounted-root identity from the secondary index, and relocated
    ///    parent entries for this filesystem instance.
    /// 5. Call [`super::driver::FileSystemInstance::unmount`] on the instance,
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
        let host_dir = self.resolve_mount_host(
            mount_path,
            working_directory,
        )?;

        let host_key = OpenFileKey::new(
            host_dir.inode.filesystem.instance_id(), host_dir.inode.inode_number,
        );

        // Verify a filesystem is actually mounted here.
        if self.mounts.get(&host_key).is_none() {
            return Err(FileSystemError::NotMounted);
        }

        // Retrieve the MountPoint so that we can check for open handles and
        // obtain the mounted-root identity for the secondary index.
        let mount_point = self.mounts
            .get(&host_key)
            .ok_or(FileSystemError::NotMounted)?;

        let mounted_root_key = OpenFileKey::new(
            mount_point.mounted_root.inode.filesystem.instance_id(),
            mount_point.mounted_root.inode.inode_number,
        );
        let filesystem_id = mount_point.filesystem_id;

        // Reject the unmount if any open file handles exist on this volume.
        // We check the open-file table for any key whose filesystem_id matches.
        let has_open_files = self.open_files
            .keys()
            .any(|key| key.filesystem_id == filesystem_id);

        if has_open_files {
            return Err(FileSystemError::FileLocked);
        }

        // The global VFS lock prevents another path walk from observing these
        // namespace table updates separately.
        let mount_point = self.mounts
            .remove(&host_key)
            .ok_or(FileSystemError::InternalError)?;

        self.mounted_root_index.remove(&mounted_root_key);
        // No relocated parent from the unmounted volume may outlive it.
        self.relocated_parents
            .retain(|key, _| key.filesystem_id != filesystem_id);

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

    /// Resolves a mount path to the directory covered by that mount.
    ///
    /// Ordinary path resolution crosses into a mounted filesystem. Mount and
    /// unmount instead need its host directory when the terminal component is
    /// already mounted. The mounted root's weak parent points to that host.
    ///
    /// # Arguments
    ///
    /// * `path` - Path of the mount host or its currently visible root.
    /// * `working_directory` - Starting directory for a relative path.
    ///
    /// # Returns
    ///
    /// Returns the mount host, or the resolved ordinary directory when no
    /// mount covers it. Resolution and parent-upgrade failures are returned.
    fn resolve_mount_host(
        &self,
        path: &str,
        working_directory: Option<Arc<Directory>>,
    ) -> Result<Arc<Directory>, FileSystemError> {
        let directory = self.resolve_to_directory(path, working_directory)?;
        if matches!(directory.parent.as_ref(), Some(DirectoryParent::MountedRoot(_))) {
            directory.parent()
        } else {
            Ok(directory)
        }
    }

    /// Resolves `path` to the [`Directory`] wrapper for the terminal
    /// component, without loading a separate inode.
    ///
    /// This internal helper mirrors the path resolver's logic but returns the
    /// [`Directory`] wrapper for the final component rather than its
    /// [`Inode`]. It is used by [`VirtualFileSystem::resolve_to_directory`]
    /// and by [`VirtualFileSystem::mount`] and [`VirtualFileSystem::unmount`],
    /// which need to identify the covered host directory.
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
