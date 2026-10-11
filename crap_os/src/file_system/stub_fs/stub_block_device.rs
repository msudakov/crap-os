//! # Stub Block Device
//!
//! This module provides [`StubBlockDevice`], a no-op implementation of
//! [`crate::file_system::virtual_fs::driver::BlockDevice`] for use with
//! [`super::StubFileSystemDriver`].
//!
//! The stub filesystem stores all its data in memory and never performs real
//! disk I/O. [`StubBlockDevice`] exists only to satisfy the
//! [`crate::file_system::virtual_fs::driver::BlockDevice`] trait requirement
//! on [`crate::file_system::virtual_fs::driver::FileSystemDriver::mount`].
//! All of its methods are no-ops or return immediate errors, since the stub
//! driver never calls them.

use crate::file_system::virtual_fs::driver::BlockDevice;
use crate::file_system::virtual_fs::types::FileSystemError;

/// A no-op block device for use with the stub filesystem driver.
///
/// The stub filesystem is entirely in-memory and never reads from or writes
/// to a real block device. This struct satisfies the
/// [`BlockDevice`] trait boundary required by
/// [`crate::file_system::virtual_fs::driver::FileSystemDriver::mount`]
/// without performing any actual I/O.
///
/// Callers should never invoke [`read_block`] or [`write_block`] on this
/// device; doing so returns [`FileSystemError::DiskReadError`] and
/// [`FileSystemError::DiskWriteError`] respectively, since there is no
/// backing storage.
///
/// [`read_block`]: StubBlockDevice::read_block
/// [`write_block`]: StubBlockDevice::write_block
pub struct StubBlockDevice;

impl BlockDevice for StubBlockDevice {
    /// Returns the logical block size for this device.
    ///
    /// The stub filesystem uses 4096-byte blocks to match the VFS-wide
    /// convention, even though no actual block I/O is ever performed.
    ///
    /// # Returns
    ///
    /// Returns `4096`.
    fn block_size(&self) -> u32 {
        4096
    }

    /// Returns the total number of logical blocks on this device.
    ///
    /// The stub device has no real storage. This value is a placeholder that
    /// satisfies the trait; the stub driver never calls this method.
    ///
    /// # Returns
    ///
    /// Returns `0`.
    fn block_count(&self) -> u64 {
        0
    }

    /// Stub implementation - always returns an error.
    ///
    /// The stub filesystem never reads from a block device. If this method
    /// is called, it indicates a bug in the stub driver implementation.
    ///
    /// # Arguments
    ///
    /// * `_block_number` - Ignored.
    /// * `_buffer`       - Ignored.
    ///
    /// # Returns
    ///
    /// Returns `Err(FileSystemError::DiskReadError)` unconditionally.
    fn read_block(
        &self,
        _block_number: u64,
        _buffer: &mut [u8],
    ) -> Result<(), FileSystemError> {
        Err(FileSystemError::DiskReadError)
    }

    /// Stub implementation - always returns an error.
    ///
    /// The stub filesystem never writes to a block device. If this method
    /// is called, it indicates a bug in the stub driver implementation.
    ///
    /// # Arguments
    ///
    /// * `_block_number` - Ignored.
    /// * `_buffer`       - Ignored.
    ///
    /// # Returns
    ///
    /// Returns `Err(FileSystemError::DiskWriteError)` unconditionally.
    fn write_block(
        &self,
        _block_number: u64,
        _buffer: &[u8],
    ) -> Result<(), FileSystemError> {
        Err(FileSystemError::DiskWriteError)
    }
}
