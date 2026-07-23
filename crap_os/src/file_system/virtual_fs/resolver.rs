//! # Virtual File System - Path Resolver
//!
//! This module implements the VFS path resolver: the algorithm that walks a
//! path string component by component through the in-memory directory tree
//! and produces the [`super::inode::Inode`] or
//! [`super::directory::Directory`] that the path refers to.
//!
//! The resolver is the most central piece of VFS logic. Every operation that
//! accepts a path (e.g., open, stat, create, delete, rename, mount, unmount)
//! begins with a resolver call.
//! 
//! ## Algorithm Overview
//!
//! 1. Determine the starting directory: the global root for absolute paths
//!    (those beginning with `/`), or `working_directory` for relative paths.
//! 2. Split the path on `/`, discarding empty components produced by leading,
//!    trailing, or consecutive separators.
//! 3. For each component in order:
//!    * `.`  - stay in the current directory and continue.
//!    * `..` - move to the parent directory, crossing mount boundaries upward
//!             if necessary.
//!    * Any other name - look it up in the current directory's child list,
//!      loading the list lazily from the driver if needed, then act on the
//!      result type (file, directory, or symlink).
//! 4. After processing the final component, return the resolved inode (or
//!    directory wrapper, depending on the entry point).
//!
//! ## Symlink Handling
//!
//! When the resolver encounters a symlink mid-path, it must follow it: the
//! symlink's target path is read from the driver and spliced into the
//! remaining resolution. If the target is absolute, resolution restarts from
//! the global root. If relative, it continues from the directory containing
//! the symlink.
//!
//! Symlinks at the final path component are handled based on the
//! `follow_symlinks` flag passed by the caller. Operations that want the
//! symlink object itself (e.g., reading or deleting the symlink) pass `false`;
//! operations that want the symlink's target (e.g., opening a file through a
//! symlink) pass `true`.
//!
//! To prevent infinite loops from circular symlink chains, the resolver
//! carries a `symlink_budget` counter. Each symlink followed decrements the
//! budget; when it reaches zero,
//! [`super::types::FileSystemError::TooManySymlinks`] is returned.
//! The initial budget is [`MAX_SYMLINK_DEPTH`].
//!
//! ## Mount Boundary Crossing
//!
//! The downward crossing (when the resolver moves into a directory that has a
//! filesystem mounted on it) is handled by checking
//! [`super::inode::Inode::mount_flag`] and following the
//! [`super::directory::Directory::mount_root`] pointer. This is a single
//! pointer follow with no table lookup.
//!
//! The Upward crossing (when the resolver handles `..` at the root of a mounted
//! filesystem) is handled by consulting the `mounted_root_index` in
//! [`super::vfs::VirtualFileSystem`], which maps mounted-root inode numbers
//! to host directory inode numbers in O(1).

#![allow(dead_code)]

use alloc::format;
use alloc::sync::Arc;
use alloc::vec::Vec;
use super::directory::Directory;
use super::inode::Inode;
use super::types::{DirectoryEntryType, FileSystemError};
use super::vfs::VirtualFileSystem;

/// The maximum number of symlink hops the resolver will follow before
/// returning [`super::types::FileSystemError::TooManySymlinks`].
///
/// Each symlink encountered during resolution decrements an internal counter
/// initialized to this value. When the counter reaches zero, the resolver
/// aborts with an error. A value of 16 is sufficient for all legitimate use
/// cases (real symlink chains are rarely deeper than 3-4 hops) while being
/// low enough to keep the worst-case recursive stack depth manageable.
pub const MAX_SYMLINK_DEPTH: u8 = 16;

impl VirtualFileSystem {
    /// Resolves `path` to the [`Inode`] it refers to.
    ///
    /// This is the primary entry point for path resolution. Every VFS
    /// operation that accepts a path string calls this method first to
    /// convert the path into an inode, then operates on the inode directly.
    ///
    /// Absolute paths (beginning with `/`) are resolved from the global root
    /// directory. Relative paths are resolved from `working_directory`, or
    /// from the global root if `working_directory` is `None`.
    ///
    /// The `follow_symlinks` flag controls behavior only when the final path
    /// component is a symlink:
    ///
    /// * `true`  - the symlink is followed, and the target's inode is returned.
    ///   Used by operations that want the object the symlink points to (e.g.,
    ///   open, stat).
    /// * `false` - the symlink inode itself is returned without following.
    ///   Used by operations that want the symlink object itself (e.g., delete,
    ///   read symlink target).
    ///
    /// Symlinks encountered at non-final components are always followed,
    /// regardless of this flag.
    ///
    /// # Arguments
    ///
    /// * `path`              - The path string to resolve. May be absolute or
    ///   relative, and may contain `.`, `..`, and symlink components.
    /// * `working_directory` - The starting directory for relative paths.
    ///   `None` means use the global root.
    /// * `follow_symlinks`   - Whether to follow a symlink at the final path
    ///   component.
    ///
    /// # Returns
    ///
    /// Returns `Ok(Arc<Inode>)` for the inode the path refers to, or
    /// `Err(FileSystemError)` if any component cannot be resolved, a symlink
    /// loop is detected, or an I/O error occurs.
    pub fn resolve(
        &self,
        path: &str,
        working_directory: Option<Arc<Directory>>,
        follow_symlinks: bool,
    ) -> Result<Arc<Inode>, FileSystemError> {
        self.resolve_with_budget(
            path,
            working_directory,
            follow_symlinks,
            MAX_SYMLINK_DEPTH,
        )
    }

    /// Resolves `path` to the [`Directory`] wrapper for the terminal
    /// directory component.
    ///
    /// Used internally by mount, unmount, and working-directory operations
    /// that need the [`Directory`] wrapper rather than just the inode. The
    /// path must resolve to a directory; resolving to a file or symlink
    /// returns [`FileSystemError::NotADirectory`].
    ///
    /// # Arguments
    ///
    /// * `path`              - The path string to resolve.
    /// * `working_directory` - The starting directory for relative paths.
    ///   `None` means use the global root.
    /// * `symlink_budget`    - The remaining number of symlink hops permitted.
    ///   Pass [`MAX_SYMLINK_DEPTH`] for the initial call.
    ///
    /// # Returns
    ///
    /// Returns `Ok(Arc<Directory>)` for the directory wrapper at the end of
    /// `path`, or `Err(FileSystemError)` if resolution fails or the terminal
    /// component is not a directory.
    pub fn resolve_dir_with_budget(
        &self,
        path: &str,
        working_directory: Option<Arc<Directory>>,
        symlink_budget: u8,
    ) -> Result<Arc<Directory>, FileSystemError> {
        let inode = self.resolve_with_budget(
            path,
            working_directory.clone(),
            false,
            symlink_budget,
        )?;

        if !inode.is_directory() {
            return Err(FileSystemError::NotADirectory);
        }

        // Walk the path again to obtain the Directory wrapper for the terminal
        // component.
        // TODO: Implement an inode cache that will make Directory wrappers
        // directly addressable by inode number.
        self.walk_to_directory(path, working_directory, symlink_budget)
    }

    /// The recursive core of the path resolver.
    ///
    /// Called by [`VirtualFileSystem::resolve`] with [`MAX_SYMLINK_DEPTH`] as
    /// the initial budget, and called recursively by itself each time a
    /// symlink is followed, with a decremented budget. The recursion depth is
    /// bounded by `symlink_budget` (at most [`MAX_SYMLINK_DEPTH`] frames deep).
    ///
    /// # Arguments
    ///
    /// * `path`              - The path string to resolve for this invocation.
    ///   On a recursive call this is the symlink target path, possibly
    ///   concatenated with remaining unresolved components.
    /// * `working_directory` - The starting directory. For absolute paths this
    ///   is ignored in favour of the global root. For relative paths, and for
    ///   relative symlink targets, this is the directory from which resolution
    ///   continues.
    /// * `follow_symlinks`   - Whether to follow a symlink at the final
    ///   component.
    /// * `symlink_budget`    - The remaining number of symlink hops permitted
    ///   before returning [`FileSystemError::TooManySymlinks`].
    ///
    /// # Returns
    ///
    /// Returns `Ok(Arc<Inode>)` for the resolved inode, or
    /// `Err(FileSystemError)` on any resolution failure.
    fn resolve_with_budget(
        &self,
        path: &str,
        working_directory: Option<Arc<Directory>>,
        follow_symlinks: bool,
        symlink_budget: u8,
    ) -> Result<Arc<Inode>, FileSystemError> {
        if path.is_empty() {
            return Err(FileSystemError::InvalidPath);
        }

        // Determine the starting directory. Absolute paths always start at
        // the global root; relative paths start from the supplied working
        // directory, falling back to root if none is provided.
        let mut current: Arc<Directory> = if path.starts_with('/') {
            Arc::clone(&self.root)
        } else {
            working_directory.unwrap_or_else(|| Arc::clone(&self.root))
        };

        // Split the path into components, discarding empty strings produced
        // by leading `/`, trailing `/`, or consecutive `//` separators.
        let components: Vec<&str> = path
            .split('/')
            .filter(|c| !c.is_empty())
            .collect();

        // If after filtering there are no components (e.g., path was "/" or
        // "///"), the path refers to the current directory itself.
        if components.is_empty() {
            return Ok(Arc::clone(&current.inode));
        }

        let last_index = components.len() - 1;

        for (index, &component) in components.iter().enumerate() {
            let is_last = index == last_index;

            // "." always refers to the current directory. No lookup needed;
            // just continue to the next component. If this is the last
            // component, the current directory's inode is the result.
            if component == "." {
                if is_last {
                    return Ok(Arc::clone(&current.inode));
                }
                continue;
            }

            // ".." moves to the parent directory. Three cases apply:
            if component == ".." {
                if current.is_root() {
                    // We are already at the global root; ".." is a no-op.
                    if is_last {
                        return Ok(Arc::clone(&current.inode));
                    }
                    continue;
                }

                // We are at the root of a mounted filesystem;  check whether
                // the current directory is a mounted root by looking up its
                // inode number in the secondary index.
                if let Some(&host_inode_number) = self
                    .mounted_root_index
                    .get(&current.inode.inode_number)
                {
                    // This directory is a mounted root. Cross upward to the
                    // host directory by retrieving it from the primary mount
                    // table.
                    let mount_point = self.mounts
                        .get(&host_inode_number)
                        .ok_or(FileSystemError::InternalError)?;

                    current = Arc::clone(&mount_point.host_directory);

                    if is_last {
                        return Ok(Arc::clone(&current.inode));
                    }
                    continue;
                }

                // Normal upward move via the weak parent reference. Upgrade
                // failure is a kernel invariant violation.
                current = current.parent()?;

                if is_last {
                    return Ok(Arc::clone(&current.inode));
                }
                continue;
            }

            // General component: look it up in the current directory's
            // child list. This may trigger lazy loading of the child list
            // from the filesystem driver if it has not been loaded yet.
            //
            // We need a mutable reference to the Directory to call
            // lookup_child (which may mutate the children field during lazy
            // loading). We obtain it through a raw pointer because Arc does
            // not hand out mutable references. This is safe because the VFS
            // lock serializes all access; no other thread can concurrently
            // mutate the Directory's children.
            let entry_type;
            let inode_number;
            {
                let dir_ptr = Arc::as_ptr(&current) as *mut Directory;
                let dir = unsafe { &mut *dir_ptr };

                match dir.lookup_child(component)? {
                    None => return Err(FileSystemError::NotFound),
                    Some(entry) => {
                        entry_type   = entry.entry_type;
                        inode_number = entry.inode_number;

                        // Eagerly load the inode into the entry's cache if it
                        // is not already loaded, so the Arc<Inode> is
                        // available for the match arms below.
                        entry.load_inode(&current.inode.filesystem)?;
                    }
                }
            }

            match entry_type {
                DirectoryEntryType::File => {
                    if !is_last {
                        // A file in a non-terminal position means the caller
                        // is trying to descend through a file as if it were a
                        // directory, which is an error.
                        return Err(FileSystemError::NotADirectory);
                    }

                    // Load the inode for the final file component and return
                    // it.
                    let inode = current.inode.filesystem
                        .load_inode(inode_number)?;
                    return Ok(inode);
                }

                DirectoryEntryType::Directory => {
                    // Load the child directory inode and wrap it in a
                    // Directory so we can descend into it.
                    let child_inode = current.inode.filesystem
                        .load_inode(inode_number)?;

                    let child_dir = Arc::new(Directory::new(
                        child_inode,
                        Arc::downgrade(&current),
                    ));

                    // Check for a downward mount boundary. If mount_flag is
                    // set on the child inode, the path resolver must cross
                    // into the mounted filesystem's root rather than
                    // descending into the child directory's own children.
                    if child_dir.inode.mount_flag {
                        // Retrieve the mounted root from the primary mount
                        // table via the child directory's inode number.
                        let mount_point = self.mounts
                            .get(&child_dir.inode.inode_number)
                            .ok_or(FileSystemError::InternalError)?;

                        current = Arc::clone(&mount_point.mounted_root);
                    } else {
                        current = child_dir;
                    }

                    if is_last {
                        return Ok(Arc::clone(&current.inode));
                    }
                }

                DirectoryEntryType::Symlink => {
                    if is_last && !follow_symlinks {
                        // The caller wants the symlink inode itself, not its
                        // target. Load and return it directly.
                        let inode = current.inode.filesystem
                            .load_inode(inode_number)?;
                        return Ok(inode);
                    }

                    // We need to follow the symlink. Deduct one hop from the
                    // budget before proceeding.
                    if symlink_budget == 0 {
                        return Err(FileSystemError::TooManySymlinks);
                    }
                    let remaining_budget = symlink_budget - 1;

                    // Read the symlink's target path from the driver.
                    let symlink_inode = current.inode.filesystem
                        .load_inode(inode_number)?;
                    let target = current.inode.filesystem
                        .read_symlink(&symlink_inode)?;

                    // Build the effective path by appending any remaining
                    // components after the symlink onto the target. For
                    // example, if the path is "/foo/link/bar/baz" and
                    // "link" resolves to a symlink with target "../qux",
                    // the effective path for the recursive call is
                    // "../qux/bar/baz".
                    let effective_path = if index < last_index {
                        let remaining: Vec<&str> =
                            components[index + 1..].to_vec();
                        let suffix = remaining.join("/");
                        format!("{}/{}", target, suffix)
                    } else {
                        target
                    };

                    // Determine the base for relative symlink targets. A
                    // relative target is resolved from the directory that
                    // contains the symlink (i.e., `current`), not from the
                    // caller's original working directory.
                    let symlink_base = if effective_path.starts_with('/') {
                        None  // Absolute target: resolver will use root.
                    } else {
                        Some(Arc::clone(&current))
                    };

                    // Recurse with the spliced path and decremented budget.
                    return self.resolve_with_budget(
                        &effective_path,
                        symlink_base,
                        follow_symlinks,
                        remaining_budget,
                    );
                }
            }
        }

        // The loop exhausted all components without returning early. This
        // happens only when `components` is non-empty but all components were
        // consumed via the continue paths (i.e., the path consisted entirely
        // of "." components). Return the current directory's inode.
        Ok(Arc::clone(&current.inode))
    }

    /// Walks `path` through the directory tree and returns the
    /// [`Directory`] wrapper for the terminal component.
    ///
    /// This internal helper is used by
    /// [`VirtualFileSystem::resolve_dir_with_budget`] to obtain the
    /// [`Directory`] wrapper when the caller needs it (e.g., for mount
    /// and unmount operations that must mutate wrapper fields). It duplicates
    /// part of the resolver's walking logic but returns the wrapper instead
    /// of the inode.
    ///
    /// TODO: Implement an inode cache that indexes [`Directory`] wrappers by
    /// inode number, so that the wrapper will be retrievable directly after
    /// resolving the inode.
    ///
    /// # Arguments
    ///
    /// * `path`              - The path string to walk. Must resolve to a
    ///   directory.
    /// * `working_directory` - The starting directory for relative paths.
    ///   `None` means use the global root.
    /// * `symlink_budget`    - The remaining symlink follow budget.
    ///
    /// # Returns
    ///
    /// Returns `Ok(Arc<Directory>)` for the directory at the end of `path`,
    /// or `Err(FileSystemError)` if resolution fails or the terminal component
    /// is not a directory.
    fn walk_to_directory(
        &self,
        path: &str,
        working_directory: Option<Arc<Directory>>,
        symlink_budget: u8,
    ) -> Result<Arc<Directory>, FileSystemError> {
        if path.is_empty() {
            return Err(FileSystemError::InvalidPath);
        }

        let mut current: Arc<Directory> = if path.starts_with('/') {
            Arc::clone(&self.root)
        } else {
            working_directory.clone().unwrap_or_else(|| Arc::clone(&self.root))
        };

        let components: Vec<&str> = path
            .split('/')
            .filter(|c| !c.is_empty())
            .collect();

        if components.is_empty() {
            return Ok(current);
        }

        let last_index = components.len() - 1;

        for (index, &component) in components.iter().enumerate() {
            let is_last = index == last_index;

            if component == "." {
                if is_last {
                    return Ok(current);
                }
                continue;
            }

            if component == ".." {
                if current.is_root() {
                    if is_last {
                        return Ok(current);
                    }
                    continue;
                }

                if let Some(&host_inode_number) = self
                    .mounted_root_index
                    .get(&current.inode.inode_number)
                {
                    let mount_point = self.mounts
                        .get(&host_inode_number)
                        .ok_or(FileSystemError::InternalError)?;
                    current = Arc::clone(&mount_point.host_directory);

                    if is_last {
                        return Ok(current);
                    }
                    continue;
                }

                current = current.parent()?;

                if is_last {
                    return Ok(current);
                }
                continue;
            }

            // Look up the child entry and resolve it to a Directory.
            let entry_type;
            let inode_number;
            {
                let dir_ptr = Arc::as_ptr(&current) as *mut Directory;
                let dir = unsafe { &mut *dir_ptr };

                match dir.lookup_child(component)? {
                    None => return Err(FileSystemError::NotFound),
                    Some(entry) => {
                        entry_type   = entry.entry_type;
                        inode_number = entry.inode_number;
                        entry.load_inode(&current.inode.filesystem)?;
                    }
                }
            }

            match entry_type {
                DirectoryEntryType::File => {
                    // A file in any position when we need a directory is an
                    // error.
                    return Err(FileSystemError::NotADirectory);
                }

                DirectoryEntryType::Directory => {
                    let child_inode = current.inode.filesystem
                        .load_inode(inode_number)?;

                    let child_dir = Arc::new(Directory::new(
                        child_inode,
                        Arc::downgrade(&current),
                    ));

                    if child_dir.inode.mount_flag {
                        let mount_point = self.mounts
                            .get(&child_dir.inode.inode_number)
                            .ok_or(FileSystemError::InternalError)?;
                        current = Arc::clone(&mount_point.mounted_root);
                    } else {
                        current = child_dir;
                    }

                    if is_last {
                        return Ok(current);
                    }
                }

                DirectoryEntryType::Symlink => {
                    if symlink_budget == 0 {
                        return Err(FileSystemError::TooManySymlinks);
                    }
                    let remaining_budget = symlink_budget - 1;

                    let symlink_inode = current.inode.filesystem
                        .load_inode(inode_number)?;
                    let target = current.inode.filesystem
                        .read_symlink(&symlink_inode)?;

                    let effective_path = if index < last_index {
                        let remaining: Vec<&str> =
                            components[index + 1..].to_vec();
                        let suffix = remaining.join("/");
                        format!("{}/{}", target, suffix)
                    } else {
                        target
                    };

                    let symlink_base = if effective_path.starts_with('/') {
                        None
                    } else {
                        Some(Arc::clone(&current))
                    };

                    return self.walk_to_directory(
                        &effective_path,
                        symlink_base,
                        remaining_budget,
                    );
                }
            }
        }

        Ok(current)
    }
}
