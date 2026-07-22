//! # Virtual File System (VFS)
//!
//! This module implements the VFS layer: the kernel's unified, format-agnostic
//! interface to all filesystem operations. Every kernel subsystem that needs
//! to read or write files, resolve paths, manage directories, or mount
//! volumes does so through the [`vfs::VirtualFileSystem`] struct, accessed
//! via the `VIRTUAL_FILE_SYSTEM` global in `globals.rs`.
//!
//! ## Module Structure
//!
//! The VFS is split across several submodules, each with a focused
//! responsibility:
//!
//! - [`types`] - foundational enums and structs shared across all other VFS
//!   submodules: [`types::FileSystemError`], [`types::InodeType`],
//!   [`types::DirectoryEntryType`], [`types::AccessMode`],
//!   [`types::FileLockState`], [`types::Timestamps`], [`types::Permissions`],
//!   [`types::InodeStat`], and [`types::OpenFileKey`].
//!
//! - [`inode`] - the in-memory inode ([`inode::Inode`]) and directory entry
//!   ([`inode::DirectoryEntry`]) types. These are the two most fundamental
//!   VFS data structures; all filesystem drivers map their on-disk
//!   representations onto them.
//!
//! - [`directory`] - the [`directory::Directory`] wrapper, which sits above
//!   an [`inode::Inode`] of type [`types::InodeType::Directory`] and adds
//!   the VFS-layer state needed for tree traversal: a lazily loaded child
//!   list, a weak parent reference, and an optional mount-root pointer.
//!
//! - [`file`] - the two-tier open-file model: [`file::File`] is the master
//!   record (one per inode with at least one open handle, owned by the object
//!   manager), and [`file::FileHandle`] is the per-open-call instance (one
//!   per `open()` call, owned by the calling process's handle table entry).
//!
//! - [`driver`] - the [`driver::FileSystemDriver`] and
//!   [`driver::FileSystemInstance`] traits that every filesystem driver must
//!   implement, plus the [`driver::BlockDevice`] trait that drivers use for
//!   all disk I/O.
//!
//! - [`mount`] - the [`mount::MountPoint`] struct, a first-class kernel
//!   object representing a filesystem mounted at a specific location in the
//!   namespace.
//!
//! - [`vfs`] - the [`vfs::VirtualFileSystem`] struct: the top-level owner of
//!   the namespace root, the driver registry, the mount table, and the
//!   open-file table. All filesystem operations flow through it.
//!
//! - [`resolver`] - the path resolution algorithm, implemented as methods on
//!   [`vfs::VirtualFileSystem`]. Walks a path string component by component
//!   through the in-memory directory tree, handling `.`, `..`, symlink
//!   following (with a configurable depth limit), and mount boundary crossing
//!   in both directions.
//!
//! - [`operations`] - the file operation methods on [`vfs::VirtualFileSystem`]:
//!   `open`, `read`, `write`, `close`, `seek`, `stat`, `create_file`,
//!   `create_directory`, `create_symlink`, `delete`, and `rename`.
//!
//! ## Dependency Order
//!
//! The submodules form a strict layered dependency:
//!
//! types
//!   |__ inode                               (uses types)
//!         |__ directory                     (uses inode, types)
//!         |__ file                          (uses inode, types)
//!         |__ driver                        (uses inode, types)
//!               |__ mount                   (uses directory, driver)
//!                     |__ vfs               (uses all of the above)
//!                           |__ resolver    (extends vfs)
//!                           |__ operations  (extends vfs)
//!
//! The VFS defines contracts; it does not know which drivers exist.

pub mod types;
pub mod inode;
pub mod directory;
pub mod file;
pub mod driver;
pub mod mount;
pub mod vfs;
pub mod resolver;
pub mod operations;
