//! # File System
//!
//! This module is the root of all filesystem-related code in the kernel. It
//! is organized into three layers, each as a submodule:
//!
//! - [`virtual_fs`] - the Virtual File System (VFS) layer. Defines the
//!   contracts (traits, types, and data structures) that all filesystem
//!   drivers must satisfy, and implements the path resolver, mount point
//!   machinery, open-file tracking, and the top-level
//!   [`virtual_fs::vfs::VirtualFileSystem`] struct that is the single entry
//!   point for all filesystem operations in the kernel.
//!
//! - [`stub_fs`] - a minimal in-memory filesystem driver used during
//!   development to exercise the VFS layer before a real on-disk filesystem
//!   is available. It implements [`virtual_fs::driver::FileSystemDriver`] and
//!   [`virtual_fs::driver::FileSystemInstance`] with a hardcoded in-memory
//!   directory tree and no disk interaction.
//!   TODO: remove this after stub_fs is no longer needed and native takes over.
//!
//! - [`crap_fs`] - the kernel's long-term native on-disk filesystem format.
//!   Designed specifically for this kernel and implemented against the VFS
//!   driver contracts.

pub mod virtual_fs;
//pub mod stub_fs;
//pub mod native_fs;
