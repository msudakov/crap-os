//! Virtual File System - Mount Points
//!
//! This module defines [`MountPoint`], the first-class kernel object that
//! represents a filesystem mounted at a specific location in the namespace.
//!
//! Every call to [`super::vfs::VirtualFileSystem::mount`] produces one
//! `MountPoint` and registers it in the VFS mount table, keyed by the inode
//! identity of the host directory (filesystem instance and inode). Every call to
//! [`super::vfs::VirtualFileSystem::unmount`] removes the entry and drops the
//! [`MountPoint`], which in turn drops the [`Arc<dyn FileSystemInstance>`] and
//! triggers the driver's [`super::driver::FileSystemInstance::unmount`]
//! cleanup path.
//!
//! The VFS mount table is the shared source of mount state. Each directory
//! wrapper can consult it by host identity during downward traversal.
//! The table retains the host directory and mounted root for each mount.
//!
//! When the path resolver handles a `..` component and the current directory
//! is the root of a mounted filesystem, it must cross back up to the host
//! directory (this is the upward boundary crossing mentioned above). To support
//! this efficiently, the VFS maintains a secondary index from mounted-root
//! identity to host identity. This lets the resolver check if a given
//! directory is a mounted root (and, if so, what its host directory is) in O(1)
//! without scanning the entire mount table.

#![allow(dead_code)]

use alloc::sync::Arc;
use super::directory::Directory;
use super::driver::FileSystemInstance;

/// A kernel object representing a filesystem mounted at a specific location
/// in the VFS namespace.
///
/// A [`MountPoint`] ties together three things: the host directory that is
/// being overlaid (the directory in the parent filesystem on which this volume
/// is mounted), the root directory of the mounted filesystem (the directory
/// that the path resolver switches to when it crosses the boundary downward),
/// and the live [`FileSystemInstance`] that services all I/O on the mounted
/// volume.
///
/// [`MountPoint`] values are owned by the VFS mount table and are never exposed
/// directly to userspace. TODO: a future syscall interface will expose mount
/// information through a separate, userspace-safe snapshot type rather than
/// handing out references to this struct.
///
/// The `filesystem_id` field records the unique ID the driver assigned to the
/// mounted instance at mount time. It serves as the `filesystem_id` half of
/// [`super::types::OpenFileKey`] for all inodes on this volume, ensuring that
/// open-file table keys remain globally unique even when multiple volumes are
/// mounted simultaneously.
pub struct MountPoint {
    /// The unique identifier assigned to the mounted filesystem instance by
    /// its driver at mount time.
    ///
    /// Used as the `filesystem_id` component of
    /// [`super::types::OpenFileKey`] for all inodes belonging to this volume,
    /// and as the key for the secondary mounted-root index in the VFS.
    pub filesystem_id: u64,

    /// The directory in the parent filesystem on which this volume is mounted.
    ///
    /// The path resolver uses this reference when handling a `..` component
    /// at the root of the mounted filesystem - it crosses back up to this
    /// directory rather than staying at the mounted root. The VFS also uses
    /// this reference when resolving an unmount path to the covered host.
    pub host_directory: Arc<Directory>,

    /// The root directory of the mounted filesystem.
    ///
    /// The path resolver switches to this directory when the mount table
    /// contains the host directory's filesystem and inode identity.
    pub mounted_root: Arc<Directory>,

    /// The live filesystem instance servicing all I/O on this mounted volume.
    ///
    /// All VFS operations on inodes belonging to this volume dispatch through
    /// this reference. Dropped during unmount, which triggers the driver's
    /// [`super::driver::FileSystemInstance::unmount`] cleanup path once the
    /// [`Arc`] reference count reaches zero.
    pub instance: Arc<dyn FileSystemInstance>,
}

impl MountPoint {
    /// Constructs a new [`MountPoint`] from its three constituent parts and
    /// the filesystem ID supplied by the driver.
    ///
    /// Called exclusively by [`super::vfs::VirtualFileSystem::mount`] after
    /// the filesystem driver has successfully mounted the volume.
    ///
    /// # Arguments
    ///
    /// * `filesystem_id`  - The unique identifier the driver assigned to
    ///   `instance` at mount time.
    /// * `host_directory` - The directory in the parent filesystem being
    ///   overlaid by this mount.
    /// * `mounted_root`   - The root directory of the newly mounted filesystem.
    /// * `instance`       - The live filesystem instance for this volume.
    ///
    /// # Returns
    ///
    /// Returns a mount record containing the host, mounted root, and instance.
    pub fn new(
        filesystem_id: u64,
        host_directory: Arc<Directory>,
        mounted_root: Arc<Directory>,
        instance: Arc<dyn FileSystemInstance>,
    ) -> Self {
        Self {
            filesystem_id,
            host_directory,
            mounted_root,
            instance,
        }
    }
}
