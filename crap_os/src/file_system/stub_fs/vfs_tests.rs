//! # VFS Test Harness
//!
//! This module contains the [`run_vfs_tests`] function, which exercises every
//! operation exposed by the [`crate::file_system::virtual_fs::vfs::VirtualFileSystem`]
//! against the stub filesystem driver. It is called from the kernel's main
//! entry point during early boot, before interrupts are enabled and the
//! prepared user-mode tasks begin executing.
//!
//! All output is written to the serial port via [`sprintln!`]. Any assertion
//! failure panics immediately with a descriptive message, so that the first
//! failure is always visible and unambiguous.
//!
//! ## Test Coverage
//!
//! The tests are grouped and run in dependency order:
//!
//! 1. **Mount** - registers the stub driver and mounts the root volume.
//! 2. **Stat** - resolves known paths and checks inode types and sizes.
//! 3. **Read** - opens known files and verifies their content byte-for-byte.
//! 4. **Write** - writes to `kernel.log`, reads back, and verifies.
//! 5. **Inode coherence** - checks identity and metadata across references.
//! 6. **Symlink traversal** - resolves paths through the `lib -> /etc` symlink.
//! 7. **Dot-dot traversal** - resolves `bin/tools/../shell` and verifies.
//! 8. **Create file** - creates a new file and verifies it can be opened.
//! 9. **Create directory** - creates a new directory and verifies it exists.
//! 10. **Create symlink** - creates a new symlink and verifies traversal.
//! 11. **Rename** - verifies in-place, relative, and cross-directory renames.
//! 12. **Unlink** - removes a file name and verifies it is gone, and checks
//!     successful removal after eager reclamation failure with cleanup retry.
//! 13. **POSIX deferred unlink** - checks two independent opens, lookup through
//!     a retained wrapper after unlink, final-close reclamation, and deletion
//!     relative to a working directory.
//! 14. **Wrapper consistency** - compares retained and fresh namespace views.
//! 15. **Mounted wrappers** - checks mount visibility, failed unmount recovery,
//!     and real unmount lookup.
//! 16. **Moved working directory** - checks `..` from a retained wrapper.
//! 17. **Trailing slashes** - requires a directory at the end of a path.
//! 18. **Hard links** - checks alias identity, data, counts, symlink policy,
//!     rejected mutations, rename between aliases, and final reclamation.
//! 19. **Directory removal and guards** - checks removal contracts, retained
//!     directory reclamation, ancestor protection, and mount boundaries.
//! 20. **File namespace protection** - checks alias-wide restrictions,
//!     additive handles, captured ancestry, symlinks, mount boundaries, and
//!     protection retention across a failed final close.
//! 21. **Replacement rename** - checks displaced lifetimes, compatible types,
//!     pathname and protection rules, pre-commit failure, and cleanup retry.
//! 22. **Create-and-open** - checks existing and new files, final symlinks,
//!     shared traversal limits, and creation and registration failures.
//! 23. **Teardown** - releases the global VFS owner after the tests.

extern crate alloc;

use alloc::sync::Arc;
use alloc::format;

use crate::file_system::stub_fs::stub_block_device::StubBlockDevice;
use crate::file_system::stub_fs::StubFileSystemDriver;
use crate::file_system::virtual_fs::directory::Directory;
use crate::file_system::virtual_fs::types::{
    AccessMode, CreateMode, FileSystemError, InodeType, Permissions,
    NamespaceProtection,
};
use crate::file_system::virtual_fs::vfs::VirtualFileSystem;
use crate::file_system::virtual_fs::file::FileHandle;
use crate::file_system::virtual_fs::inode::Inode;
use crate::globals;
use crate::file_system::virtual_fs::driver::FileSystemDriver;

use crate::sprintln;

/// Runs all VFS tests against the stub filesystem.
///
/// Initializes the global [`VirtualFileSystem`], registers the stub driver,
/// mounts the stub volume, runs all operation tests in order, then tears down
/// the root VFS instance.
/// All pass/fail output is written to the serial port.
///
/// Must be called after the serial port and wall clock are initialized, and
/// before any other filesystem use.
///
/// # Panics
///
/// Panics on the first failed test assertion, identifying the failing case.
pub fn run_vfs_tests() {
    sprintln!("[VFS TEST] Starting VFS test suite");
    sprintln!("[VFS TEST] ----------------------------------------");

    test_mount_and_register();
    test_stat();
    test_read();
    test_write();
    test_inode_coherence();
    test_symlink_traversal();
    test_dotdot_traversal();
    test_create_file();
    test_create_directory();
    test_create_symlink();
    test_rename();
    test_unlink();
    test_unlink_reclamation_retry();
    test_posix_deferred_unlink();
    test_retained_directory_namespace();
    test_mount_wrapper_consistency();
    test_moved_working_directory_parent();
    test_trailing_slash_paths();
    test_hard_links();
    test_directory_removal_and_guards();
    test_namespace_name_protection();
    test_namespace_combined_protection();
    test_namespace_symlinks_and_mounts();
    test_namespace_close_retry();
    test_replacement_file_lifetimes();
    test_replacement_directory_lifetimes();
    test_replacement_rejections();
    test_replacement_failure_atomicity();
    test_replacement_reclamation();
    test_open_or_create_basic();
    test_open_or_create_symlinks();
    test_open_or_create_failures();
    expect_ok(with_vfs(|vfs| vfs.reclaim_unlinked_inodes()), "final unlinked inode cleanup");
    teardown_vfs();

    sprintln!("[VFS TEST] ----------------------------------------");
    sprintln!("[VFS TEST] All tests passed");
}

// ----------------------------------------------------------------------------
// Helpers
// ----------------------------------------------------------------------------

/// Locks the global VFS and calls `f` with a mutable reference to it.
///
/// # Arguments
///
/// * `f` - A closure that receives `&mut VirtualFileSystem` and returns `T`.
///
/// # Returns
///
/// Returns the value produced by `f`.
///
/// # Panics
///
/// Panics if the VFS has not been initialized.
fn with_vfs<T, F: FnOnce(&mut VirtualFileSystem) -> T>(f: F) -> T {
    let mut guard = globals::VIRTUAL_FILE_SYSTEM.lock();
    let vfs = guard.as_mut().expect("[VFS TEST] VirtualFileSystem not initialized");
    f(vfs)
}

/// Asserts that `result` is `Ok`, panicking with `msg` if it is `Err`.
///
/// # Arguments
///
/// * `result` - The result to check.
/// * `msg`    - The panic message prefix if the result is `Err`.
///
/// # Returns
///
/// Returns the inner `Ok` value.
///
/// # Panics
///
/// Panics with `msg` if `result` is `Err`.
fn expect_ok<T>(result: Result<T, FileSystemError>, msg: &str) -> T {
    match result {
        Ok(v)  => v,
        Err(e) => panic!("[VFS TEST] FAIL - {}: {:?}", msg, e),
    }
}

/// Asserts that `result` is `Err` with the expected error variant, panicking
/// otherwise.
///
/// # Arguments
///
/// * `result`   - The result to check.
/// * `expected` - The expected error variant.
/// * `msg`      - The panic message prefix if the assertion fails.
///
/// # Panics
///
/// Panics if `result` is `Ok` or contains a different error variant.
fn expect_err<T>(
    result: Result<T, FileSystemError>,
    expected: FileSystemError,
    msg: &str,
) {
    match result {
        Err(e) if e == expected => {}
        Err(e) => panic!(
            "[VFS TEST] FAIL - {}: expected {:?}, got {:?}",
            msg, expected, e
        ),
        Ok(_) => panic!(
            "[VFS TEST] FAIL - {}: expected {:?}, got Ok",
            msg, expected
        ),
    }
}

/// Opens a file and returns the handle already registered by the VFS.
///
/// # Arguments
///
/// * `path`        - The path to open.
/// * `access_mode` - The access rights to request.
///
/// # Returns
///
/// Returns `Arc<FileHandle>` on success.
///
/// # Panics
///
/// Panics if the VFS cannot open `path`.
fn open_file(path: &str, access_mode: AccessMode) -> Arc<FileHandle> {
    expect_ok(
        with_vfs(|vfs| vfs.open(path, access_mode, None)),
        &alloc::format!("open({})", path),
    )
}

/// Closes a registered handle, invalidating all of its `Arc` clones.
///
/// # Arguments
///
/// * `handle` - The handle to close. The caller retains it on failure.
///
/// # Panics
///
/// Panics if the VFS cannot close `handle`.
fn close_file(handle: &Arc<FileHandle>) {
    let result = with_vfs(|vfs| vfs.close(handle));
    expect_ok(result, "close");
}

// ----------------------------------------------------------------------------
// Individual tests
// ----------------------------------------------------------------------------

/// Registers the stub driver and mounts the stub volume as the VFS root.
///
/// # Panics
///
/// Panics if mounting the stub volume or loading its root inode fails.
fn test_mount_and_register() {
    sprintln!("[VFS TEST] mount and register ...");

    // Mount the boot instance directly through the driver, then construct
    // the VFS around its real root. Only additional volumes use VFS mount().
    let stub_driver = Arc::new(StubFileSystemDriver);
    let device      = Arc::new(StubBlockDevice);

    // Mount the stub to produce a live instance, then build the root.
    let instance = stub_driver.mount(Arc::clone(&device) as Arc<dyn crate::file_system::virtual_fs::driver::BlockDevice>)
        .expect("[VFS TEST] stub driver mount() failed");

    // Load the root inode (inode number 1 by convention).
    let root_inode = instance.load_inode(1)
        .expect("[VFS TEST] failed to load root inode");

    let root_dir = Arc::new(Directory::new_root(root_inode));

    // Initialize the global VFS with this root directory.
    {
        let mut guard = globals::VIRTUAL_FILE_SYSTEM.lock();
        *guard = Some(VirtualFileSystem::new(Arc::clone(&root_dir)));
    }

    // Register the stub driver for additional mount-point tests. The root
    // instance itself has no host directory or mount-table entry.
    with_vfs(|vfs| {
        vfs.register_driver(Arc::new(StubFileSystemDriver));
    });

    sprintln!("[VFS TEST] mount and register ... PASS");
}

/// Verifies that known paths resolve to inodes with the expected types.
///
/// # Panics
///
/// Panics if a lookup fails or an inode has the wrong type or metadata.
fn test_stat() {
    sprintln!("[VFS TEST] stat ...");

    // Root should be a directory.
    let stat = expect_ok(
        with_vfs(|vfs| vfs.stat("/", None, false)),
        "stat(/)",
    );
    assert!(
        stat.inode_type == InodeType::Directory,
        "[VFS TEST] FAIL - stat(/): expected Directory"
    );

    // /etc should be a directory.
    let stat = expect_ok(
        with_vfs(|vfs| vfs.stat("/etc", None, false)),
        "stat(/etc)",
    );
    assert!(
        stat.inode_type == InodeType::Directory,
        "[VFS TEST] FAIL - stat(/etc): expected Directory"
    );

    // /etc/config.txt should be a file with the right size.
    let stat = expect_ok(
        with_vfs(|vfs| vfs.stat("/etc/config.txt", None, false)),
        "stat(/etc/config.txt)",
    );
    assert!(
        stat.inode_type == InodeType::File,
        "[VFS TEST] FAIL - stat(/etc/config.txt): expected File"
    );
    assert!(
        stat.size == b"kernel=1\nversion=0.1\n".len() as u64,
        "[VFS TEST] FAIL - stat(/etc/config.txt): wrong size"
    );

    // /lib should be a symlink when follow_symlinks is false.
    let stat = expect_ok(
        with_vfs(|vfs| vfs.stat("/lib", None, false)),
        "stat(/lib, no follow)",
    );
    assert!(
        stat.inode_type == InodeType::Symlink,
        "[VFS TEST] FAIL - stat(/lib, no follow): expected Symlink"
    );

    // /lib with follow_symlinks should resolve to /etc - a directory.
    let stat = expect_ok(
        with_vfs(|vfs| vfs.stat("/lib", None, true)),
        "stat(/lib, follow)",
    );
    assert!(
        stat.inode_type == InodeType::Directory,
        "[VFS TEST] FAIL - stat(/lib, follow): expected Directory"
    );

    sprintln!("[VFS TEST] stat ... PASS");
}

/// Verifies registered reads, independent opens, shared clones, and close.
///
/// # Panics
///
/// Panics when a read or handle-lifecycle check fails.
fn test_read() {
    sprintln!("[VFS TEST] read ...");

    // Separate opens have independent offsets, while Arc clones share one.
    let handle = open_file("/etc/config.txt", AccessMode::ReadOnly);
    let clone = Arc::clone(&handle);
    let independent = open_file("/etc/config.txt", AccessMode::ReadOnly);

    let mut prefix = [0u8; 7];
    let n = expect_ok(
        with_vfs(|vfs| vfs.read(handle.as_ref(), &mut prefix)),
        "read config prefix",
    );
    assert!(
        &prefix[..n] == b"kernel=",
        "[VFS TEST] FAIL - read config prefix"
    );

    let mut buf = [0u8; 64];
    let n = expect_ok(
        with_vfs(|vfs| vfs.read(clone.as_ref(), &mut buf)),
        "read config remainder through clone",
    );
    assert!(
        &buf[..n] == b"1\nversion=0.1\n",
        "[VFS TEST] FAIL - clone did not share the file position"
    );

    let n = expect_ok(
        with_vfs(|vfs| vfs.read(handle.as_ref(), &mut buf)),
        "read config at EOF",
    );
    assert!(n == 0, "[VFS TEST] FAIL - config read should reach EOF");

    expect_ok(
        with_vfs(|vfs| vfs.seek(handle.as_ref(), 0)),
        "seek config to start",
    );
    let n = expect_ok(
        with_vfs(|vfs| vfs.read(clone.as_ref(), &mut prefix)),
        "read config after seek through clone",
    );
    assert!(
        &prefix[..n] == b"kernel=",
        "[VFS TEST] FAIL - clone did not observe seek"
    );

    let n = expect_ok(
        with_vfs(|vfs| vfs.read(independent.as_ref(), &mut buf)),
        "read config through independent open",
    );
    assert!(
        &buf[..n] == b"kernel=1\nversion=0.1\n",
        "[VFS TEST] FAIL - independent open did not start at offset zero"
    );
    close_file(&independent);
    close_file(&handle);
    expect_err(
        with_vfs(|vfs| vfs.read(clone.as_ref(), &mut buf)),
        FileSystemError::InvalidHandle,
        "read through clone after close",
    );
    expect_err(
        with_vfs(|vfs| vfs.close(&clone)),
        FileSystemError::InvalidHandle,
        "close handle twice",
    );
    expect_err(
        with_vfs(|vfs| vfs.seek(clone.as_ref(), 0)),
        FileSystemError::InvalidHandle,
        "seek through clone after close",
    );

    // Read /etc/hosts.txt and verify content.
    let hosts = open_file("/etc/hosts.txt", AccessMode::ReadOnly);
    let n = expect_ok(
        with_vfs(|vfs| vfs.read(hosts.as_ref(), &mut buf)),
        "read /etc/hosts.txt",
    );
    assert!(
        &buf[..n] == b"127.0.0.1 localhost\n",
        "[VFS TEST] FAIL - read(/etc/hosts.txt): content mismatch"
    );
    close_file(&hosts);

    // Read /bin/shell and verify content.
    let shell = open_file("/bin/shell", AccessMode::ReadOnly);
    let mut shell_buf = [0u8; 16];
    let n = expect_ok(
        with_vfs(|vfs| vfs.read(shell.as_ref(), &mut shell_buf)),
        "read /bin/shell",
    );
    assert!(
        &shell_buf[..n] == b"STUB_BINARY",
        "[VFS TEST] FAIL - read(/bin/shell): content mismatch"
    );
    close_file(&shell);

    sprintln!("[VFS TEST] read ... PASS");
}

/// Writes to `/var/log/kernel.log`, then reads back and verifies the content.
///
/// # Panics
///
/// Panics if writing, reading, or closing fails.
fn test_write() {
    sprintln!("[VFS TEST] write ...");

    let write_data = b"boot ok\n";

    {
        let handle = expect_ok(
            with_vfs(|vfs| vfs.open("/var/log/kernel.log", AccessMode::ReadWrite, None)),
            "open /var/log/kernel.log for write",
        );
        let n = expect_ok(
            with_vfs(|vfs| vfs.write(handle.as_ref(), write_data)),
            "write /var/log/kernel.log",
        );
        assert!(
            n == write_data.len(),
            "[VFS TEST] FAIL - write: wrote {} bytes, expected {}",
            n, write_data.len()
        );
        close_file(&handle);
        expect_err(
            with_vfs(|vfs| vfs.write(handle.as_ref(), write_data)),
            FileSystemError::InvalidHandle,
            "write after close",
        );
    }

    // Re-open and read back.
    {
        let handle = expect_ok(
            with_vfs(|vfs| vfs.open("/var/log/kernel.log", AccessMode::ReadOnly, None)),
            "open /var/log/kernel.log for read-back",
        );
        let mut buf = [0u8; 32];
        let n = expect_ok(
            with_vfs(|vfs| vfs.read(handle.as_ref(), &mut buf)),
            "read-back /var/log/kernel.log",
        );
        assert!(
            &buf[..n] == write_data,
            "[VFS TEST] FAIL - write read-back: content mismatch"
        );
        close_file(&handle);
    }

    sprintln!("[VFS TEST] write ... PASS");
}

/// Checks that create, lookup, open, write, and unlink share one live inode.
///
/// # Panics
///
/// Panics if references disagree about identity, size, or link count, or if
/// the unlinked inode remains allocated after its final close.
fn test_inode_coherence() {
    sprintln!("[VFS TEST] inode coherence ...");

    let path = "/var/log/inode-coherence.tmp";
    let created = expect_ok(
        with_vfs(|vfs| vfs.create_file(
            path, Permissions::all_permissive(), None,
        )),
        "create file for inode coherence",
    );
    let initial = expect_ok(
        with_vfs(|vfs| vfs.resolve(path, None, false)),
        "resolve created inode",
    );
    assert!(Arc::ptr_eq(&created, &initial),
        "[VFS TEST] FAIL - create and lookup returned different inodes");

    let handle = open_file(path, AccessMode::ReadWrite);
    assert!(Arc::ptr_eq(&created, &handle.file.inode),
        "[VFS TEST] FAIL - open returned a different inode");
    let data = b"shared inode";
    let written = expect_ok(
        with_vfs(|vfs| vfs.write(handle.as_ref(), data)),
        "write file through canonical inode",
    );
    assert!(written == data.len(),
        "[VFS TEST] FAIL - inode coherence write was short");

    let fresh = expect_ok(
        with_vfs(|vfs| vfs.resolve(path, None, false)),
        "resolve inode after write",
    );
    assert!(Arc::ptr_eq(&created, &fresh),
        "[VFS TEST] FAIL - fresh lookup returned a different inode");
    assert!(created.stat().size == data.len() as u64
        && fresh.stat().size == data.len() as u64,
        "[VFS TEST] FAIL - retained and fresh inode sizes disagree");

    expect_ok(
        with_vfs(|vfs| vfs.unlink(path, None)),
        "unlink file while handle remains open",
    );
    assert!(created.stat().link_count == 0
        && handle.file.inode.stat().link_count == 0,
        "[VFS TEST] FAIL - retained inode kept its old link count");
    expect_err(
        with_vfs(|vfs| vfs.resolve(path, None, false)),
        FileSystemError::NotFound,
        "unlinked inode remains reachable by path",
    );

    close_file(&handle);
    expect_err(
        created.filesystem.load_inode(created.inode_number),
        FileSystemError::NotFound,
        "unlinked inode was not freed after final close",
    );

    sprintln!("[VFS TEST] inode coherence ... PASS");
}

/// Verifies that paths through the `lib -> /etc` symlink resolve correctly.
///
/// # Panics
///
/// Panics if resolution or reading fails.
fn test_symlink_traversal() {
    sprintln!("[VFS TEST] symlink traversal ...");

    // /lib/config.txt should resolve through lib -> /etc to /etc/config.txt.
    let stat = expect_ok(
        with_vfs(|vfs| vfs.stat("/lib/config.txt", None, true)),
        "stat(/lib/config.txt)",
    );
    assert!(
        stat.inode_type == InodeType::File,
        "[VFS TEST] FAIL - symlink traversal: /lib/config.txt not a File"
    );
    assert!(
        stat.size == b"kernel=1\nversion=0.1\n".len() as u64,
        "[VFS TEST] FAIL - symlink traversal: /lib/config.txt wrong size"
    );

    // Read through the symlink and verify content matches /etc/config.txt.
    {
        let handle = expect_ok(
            with_vfs(|vfs| vfs.open("/lib/config.txt", AccessMode::ReadOnly, None)),
            "open /lib/config.txt via symlink",
        );
        let mut buf = [0u8; 64];
        let n = expect_ok(
            with_vfs(|vfs| vfs.read(handle.as_ref(), &mut buf)),
            "read /lib/config.txt via symlink",
        );
        assert!(
            &buf[..n] == b"kernel=1\nversion=0.1\n",
            "[VFS TEST] FAIL - symlink traversal read: content mismatch"
        );
        close_file(&handle);
    }

    sprintln!("[VFS TEST] symlink traversal ... PASS");
}

/// Verifies that `..` resolves both during a walk and from a retained directory.
///
/// # Panics
///
/// Panics if a path crosses to the wrong parent.
fn test_dotdot_traversal() {
    sprintln!("[VFS TEST] dot-dot traversal ...");

    // /bin/tools/../shell should resolve to /bin/shell.
    let stat_direct = expect_ok(
        with_vfs(|vfs| vfs.stat("/bin/shell", None, false)),
        "stat(/bin/shell) direct",
    );
    let stat_dotdot = expect_ok(
        with_vfs(|vfs| vfs.stat("/bin/tools/../shell", None, false)),
        "stat(/bin/tools/../shell)",
    );
    assert!(
        stat_direct.inode_number == stat_dotdot.inode_number,
        "[VFS TEST] FAIL - dot-dot: inode numbers differ"
    );

    // A working-directory wrapper outlives the walk that constructed it.
    // Its parent must remain available for a later relative `..` lookup.
    let working_dir = expect_ok(
        with_vfs(|vfs| vfs.resolve_dir_with_budget(
            "/bin/tools",
            None,
            crate::file_system::virtual_fs::resolver::MAX_SYMLINK_DEPTH,
        )),
        "resolve /bin/tools as working directory",
    );
    let stat_relative = expect_ok(
        with_vfs(|vfs| vfs.stat("../shell", Some(working_dir), false)),
        "stat ../shell from /bin/tools",
    );
    assert!(
        stat_direct.inode_number == stat_relative.inode_number,
        "[VFS TEST] FAIL - dot-dot: retained working directory lost its parent"
    );

    // /etc/.. should resolve to root.
    let stat_root = expect_ok(
        with_vfs(|vfs| vfs.stat("/", None, false)),
        "stat(/)",
    );
    let stat_up = expect_ok(
        with_vfs(|vfs| vfs.stat("/etc/..", None, false)),
        "stat(/etc/..)",
    );
    assert!(
        stat_root.inode_number == stat_up.inode_number,
        "[VFS TEST] FAIL - dot-dot: /etc/.. did not resolve to root"
    );

    // /../.. should stay at root (cannot go above root).
    let stat_overflow = expect_ok(
        with_vfs(|vfs| vfs.stat("/../..", None, false)),
        "stat(/../..)",
    );
    assert!(
        stat_root.inode_number == stat_overflow.inode_number,
        "[VFS TEST] FAIL - dot-dot: /../.. did not clamp to root"
    );

    sprintln!("[VFS TEST] dot-dot traversal ... PASS");
}

/// Creates a new file under `/var/log/` and verifies its type, size, and
/// duplicate-name rejection.
///
/// # Panics
///
/// Panics if file creation or a follow-up lookup fails.
fn test_create_file() {
    sprintln!("[VFS TEST] create_file ...");

    expect_ok(
        with_vfs(|vfs| vfs.create_file(
            "/var/log/test.log",
            Permissions::all_permissive(),
            None,
        )),
        "create_file /var/log/test.log",
    );

    // Verify the file exists and is a File.
    let stat = expect_ok(
        with_vfs(|vfs| vfs.stat("/var/log/test.log", None, false)),
        "stat /var/log/test.log after create",
    );
    assert!(
        stat.inode_type == InodeType::File,
        "[VFS TEST] FAIL - create_file: /var/log/test.log not a File"
    );
    assert!(
        stat.size == 0,
        "[VFS TEST] FAIL - create_file: /var/log/test.log should be empty"
    );

    // Creating the same file again should return AlreadyExists.
    expect_err(
        with_vfs(|vfs| vfs.create_file(
            "/var/log/test.log",
            Permissions::all_permissive(),
            None,
        )),
        FileSystemError::AlreadyExists,
        "create_file duplicate",
    );

    sprintln!("[VFS TEST] create_file ... PASS");
}

/// Creates a new directory under `/bin/` and verifies it exists.
///
/// # Panics
///
/// Panics if directory creation or a follow-up lookup fails.
fn test_create_directory() {
    sprintln!("[VFS TEST] create_directory ...");

    expect_ok(
        with_vfs(|vfs| vfs.create_directory(
            "/bin/utils",
            Permissions::all_permissive(),
            None,
        )),
        "create_directory /bin/utils",
    );

    let stat = expect_ok(
        with_vfs(|vfs| vfs.stat("/bin/utils", None, false)),
        "stat /bin/utils after create",
    );
    assert!(
        stat.inode_type == InodeType::Directory,
        "[VFS TEST] FAIL - create_directory: /bin/utils not a Directory"
    );

    // Creating the same directory again should return AlreadyExists.
    expect_err(
        with_vfs(|vfs| vfs.create_directory(
            "/bin/utils",
            Permissions::all_permissive(),
            None,
        )),
        FileSystemError::AlreadyExists,
        "create_directory duplicate",
    );

    sprintln!("[VFS TEST] create_directory ... PASS");
}

/// Creates a new symlink and verifies that traversal through it works.
///
/// # Panics
///
/// Panics if creation, resolution, or reading fails.
fn test_create_symlink() {
    sprintln!("[VFS TEST] create_symlink ...");

    // Create /etc/cfg -> /etc/config.txt
    expect_ok(
        with_vfs(|vfs| vfs.create_symlink(
            "/etc/cfg",
            "/etc/config.txt",
            None,
        )),
        "create_symlink /etc/cfg -> /etc/config.txt",
    );

    // Without following, it should be a Symlink.
    let stat = expect_ok(
        with_vfs(|vfs| vfs.stat("/etc/cfg", None, false)),
        "stat /etc/cfg no follow",
    );
    assert!(
        stat.inode_type == InodeType::Symlink,
        "[VFS TEST] FAIL - create_symlink: /etc/cfg not a Symlink"
    );

    // With following, it should resolve to /etc/config.txt - a File.
    let stat = expect_ok(
        with_vfs(|vfs| vfs.stat("/etc/cfg", None, true)),
        "stat /etc/cfg follow",
    );
    assert!(
        stat.inode_type == InodeType::File,
        "[VFS TEST] FAIL - create_symlink: /etc/cfg follow not a File"
    );

    // Read through the new symlink and verify content.
    {
        let handle = expect_ok(
            with_vfs(|vfs| vfs.open("/etc/cfg", AccessMode::ReadOnly, None)),
            "open /etc/cfg via new symlink",
        );
        let mut buf = [0u8; 64];
        let n = expect_ok(
            with_vfs(|vfs| vfs.read(handle.as_ref(), &mut buf)),
            "read /etc/cfg via new symlink",
        );
        assert!(
            &buf[..n] == b"kernel=1\nversion=0.1\n",
            "[VFS TEST] FAIL - create_symlink read: content mismatch"
        );
        close_file(&handle);
    }

    sprintln!("[VFS TEST] create_symlink ... PASS");
}

/// Verifies in-place, relative, and cross-directory renames.
///
/// The relative and cross-directory cases restore their source paths so later
/// tests can continue to use the original stub filesystem fixtures.
///
/// # Panics
///
/// Panics if renaming, lookup, or reading fails.
fn test_rename() {
    sprintln!("[VFS TEST] rename ...");

    expect_ok(
        with_vfs(|vfs| vfs.rename("/etc/hosts.txt", "/etc/hosts.cfg", None)),
        "rename /etc/hosts.txt -> /etc/hosts.cfg",
    );

    // Old name should be gone.
    expect_err(
        with_vfs(|vfs| vfs.stat("/etc/hosts.txt", None, false)),
        FileSystemError::NotFound,
        "stat old name after rename",
    );

    // New name should exist and be a File with the same content.
    let stat = expect_ok(
        with_vfs(|vfs| vfs.stat("/etc/hosts.cfg", None, false)),
        "stat /etc/hosts.cfg after rename",
    );
    assert!(
        stat.inode_type == InodeType::File,
        "[VFS TEST] FAIL - rename: /etc/hosts.cfg not a File"
    );

    {
        let handle = expect_ok(
            with_vfs(|vfs| vfs.open("/etc/hosts.cfg", AccessMode::ReadOnly, None)),
            "open /etc/hosts.cfg after rename",
        );
        let mut buf = [0u8; 64];
        let n = expect_ok(
            with_vfs(|vfs| vfs.read(handle.as_ref(), &mut buf)),
            "read /etc/hosts.cfg after rename",
        );
        assert!(
            &buf[..n] == b"127.0.0.1 localhost\n",
            "[VFS TEST] FAIL - rename read: content mismatch"
        );
        close_file(&handle);
    }

    // Rename relative to a retained working directory, then restore the
    // absolute name so the fixture remains available at /etc/hosts.cfg.
    let etc_dir = expect_ok(
        with_vfs(|vfs| vfs.resolve_dir_with_budget(
            "/etc",
            None,
            crate::file_system::virtual_fs::resolver::MAX_SYMLINK_DEPTH,
        )),
        "resolve /etc as working directory for rename",
    );
    expect_ok(
        with_vfs(|vfs| vfs.rename("hosts.cfg", "hosts.relative", Some(etc_dir))),
        "rename hosts.cfg relative to /etc",
    );
    expect_err(
        with_vfs(|vfs| vfs.stat("/etc/hosts.cfg", None, false)),
        FileSystemError::NotFound,
        "stat old name after relative rename",
    );
    let relative_stat = expect_ok(
        with_vfs(|vfs| vfs.stat("/etc/hosts.relative", None, false)),
        "stat new name after relative rename",
    );
    assert!(
        relative_stat.inode_number == stat.inode_number,
        "[VFS TEST] FAIL - relative rename changed the inode"
    );
    expect_ok(
        with_vfs(|vfs| vfs.rename("/etc/hosts.relative", "/etc/hosts.cfg", None)),
        "restore /etc/hosts.cfg after relative rename",
    );

    // Move a file between directories and back. The inode and content must
    // survive both moves, and the old name must disappear after each one.
    let debug_stat = expect_ok(
        with_vfs(|vfs| vfs.stat("/bin/tools/debug", None, false)),
        "stat /bin/tools/debug before cross-directory rename",
    );
    expect_ok(
        with_vfs(|vfs| vfs.rename("/bin/tools/debug", "/var/log/debug", None)),
        "move /bin/tools/debug to /var/log/debug",
    );
    expect_err(
        with_vfs(|vfs| vfs.stat("/bin/tools/debug", None, false)),
        FileSystemError::NotFound,
        "stat old name after cross-directory rename",
    );
    let moved_stat = expect_ok(
        with_vfs(|vfs| vfs.stat("/var/log/debug", None, false)),
        "stat /var/log/debug after cross-directory rename",
    );
    assert!(
        moved_stat.inode_number == debug_stat.inode_number,
        "[VFS TEST] FAIL - cross-directory rename changed the inode"
    );
    let moved_handle = open_file("/var/log/debug", AccessMode::ReadOnly);
    let mut moved_data = [0u8; 16];
    let moved_len = expect_ok(
        with_vfs(|vfs| vfs.read(moved_handle.as_ref(), &mut moved_data)),
        "read /var/log/debug after cross-directory rename",
    );
    assert!(
        &moved_data[..moved_len] == b"STUB_BINARY",
        "[VFS TEST] FAIL - cross-directory rename changed file content"
    );
    close_file(&moved_handle);
    expect_ok(
        with_vfs(|vfs| vfs.rename("/var/log/debug", "/bin/tools/debug", None)),
        "restore /bin/tools/debug after cross-directory rename",
    );
    expect_err(
        with_vfs(|vfs| vfs.stat("/var/log/debug", None, false)),
        FileSystemError::NotFound,
        "stat moved name after restoring /bin/tools/debug",
    );
    let restored_stat = expect_ok(
        with_vfs(|vfs| vfs.stat("/bin/tools/debug", None, false)),
        "stat /bin/tools/debug after restoring cross-directory rename",
    );
    assert!(
        restored_stat.inode_number == debug_stat.inode_number,
        "[VFS TEST] FAIL - restoring cross-directory rename changed the inode"
    );

    sprintln!("[VFS TEST] rename ... PASS");
}

/// Unlinks `/bin/tools/debug` and verifies its name is gone.
///
/// # Panics
///
/// Panics if deletion fails or the old path remains reachable.
fn test_unlink() {
    sprintln!("[VFS TEST] unlink ...");

    expect_ok(
        with_vfs(|vfs| vfs.unlink("/bin/tools/debug", None)),
        "unlink /bin/tools/debug",
    );

    // The path should now return NotFound.
    expect_err(
        with_vfs(|vfs| vfs.stat("/bin/tools/debug", None, false)),
        FileSystemError::NotFound,
        "stat after unlink",
    );

    // Unlinking the absent name again should return NotFound.
    expect_err(
        with_vfs(|vfs| vfs.unlink("/bin/tools/debug", None)),
        FileSystemError::NotFound,
        "double unlink",
    );

    sprintln!("[VFS TEST] unlink ... PASS");
}

/// Checks successful unlink after failed eager reclamation and cleanup retry.
///
/// Each isolated fixture has no registered open handles. A targeted one-shot
/// free failure must be consumed during unlink, leaving the removed inode
/// available for explicit cleanup. Both regular files and dangling symlinks
/// exercise the non-directory reclamation path.
///
/// # Panics
///
/// Panics if unlink reports the reclamation error, preserves the name, loses
/// the retryable inode, skips eager reclamation, or repeats a completed free.
fn test_unlink_reclamation_retry() {
    sprintln!("[VFS TEST] unlink reclamation retry ...");

    for symlink in [false, true] {
        let mut vfs = new_stub_vfs();
        let path = "/etc/unlink-retry";
        let inode = if symlink {
            expect_ok(
                vfs.create_symlink(path, "/missing-unlink-target", None),
                "create unlink retry symlink",
            )
        } else {
            expect_ok(
                vfs.create_file(path, Permissions::all_permissive(), None),
                "create unlink retry file",
            )
        };

        assert_eq!(inode.stat().link_count, 1);
        super::fail_next_free_for(
            inode.filesystem.instance_id(),
            inode.inode_number,
        );

        expect_ok(
            vfs.unlink(path, None),
            "unlink succeeds despite eager free failure",
        );
        expect_err(
            vfs.stat(path, None, false),
            FileSystemError::NotFound,
            "failed eager free leaves name removed",
        );
        assert_eq!(inode.stat().link_count, 0);

        let backing = expect_ok(
            inode.filesystem.load_inode(inode.inode_number),
            "failed eager free preserves backing inode",
        );
        assert_eq!(backing.stat().link_count, 0);
        drop(backing);

        // This must succeed on the first cleanup call. An implementation
        // that skipped eager free would still have the failure armed.
        assert_eq!(
            expect_ok(
                vfs.reclaim_unlinked_inodes(),
                "retry failed eager reclamation",
            ),
            1,
        );
        expect_err(
            inode.filesystem.load_inode(inode.inode_number),
            FileSystemError::NotFound,
            "cleanup retry frees backing inode",
        );
        assert_eq!(
            expect_ok(
                vfs.reclaim_unlinked_inodes(),
                "cleanup after successful retry",
            ),
            0,
        );
    }

    sprintln!("[VFS TEST] unlink reclamation retry ... PASS");
}

/// Verifies that unlink preserves data for independent open handles, rejects
/// an open through a retained directory wrapper, and frees on the last close.
/// Also verifies deletion relative to a working directory.
///
/// # Panics
///
/// Panics if any deferred-deletion check fails.
fn test_posix_deferred_unlink() {
    sprintln!("[VFS TEST] POSIX deferred unlink ...");

    // Retain a directory wrapper, look up its entry, then open the file twice.
    let etc_dir = expect_ok(
        with_vfs(|vfs| vfs.resolve_dir_with_budget(
            "/etc",
            None,
            crate::file_system::virtual_fs::resolver::MAX_SYMLINK_DEPTH,
        )),
        "resolve /etc before deferred unlink",
    );
    expect_ok(
        with_vfs(|vfs| vfs.stat("config.txt", Some(Arc::clone(&etc_dir)), false)),
        "load config.txt in retained /etc wrapper",
    );
    let first = expect_ok(
        with_vfs(|vfs| vfs.open("/etc/config.txt", AccessMode::ReadOnly, None)),
        "first open /etc/config.txt before deferred unlink",
    );
    let last = expect_ok(
        with_vfs(|vfs| vfs.open("/etc/config.txt", AccessMode::ReadOnly, None)),
        "second open /etc/config.txt before deferred unlink",
    );

    // The open-file master must retain the deferred-free decision.
    expect_ok(
        with_vfs(|vfs| vfs.unlink("/etc/config.txt", None)),
        "unlink /etc/config.txt with open handle",
    );

    // The path should now be gone from the namespace.
    expect_err(
        with_vfs(|vfs| vfs.stat("/etc/config.txt", None, false)),
        FileSystemError::NotFound,
        "stat after deferred unlink",
    );
    expect_err(
        with_vfs(|vfs| vfs.open(
            "config.txt",
            AccessMode::ReadOnly,
            Some(Arc::clone(&etc_dir)),
        )),
        FileSystemError::NotFound,
        "open unlinked name through retained /etc wrapper",
    );

    // Both independent handles still read the original content.
    let mut buf = [0u8; 64];
    let n = expect_ok(
        with_vfs(|vfs| vfs.read(first.as_ref(), &mut buf)),
        "read through first handle after deferred unlink",
    );
    assert!(
        &buf[..n] == b"kernel=1\nversion=0.1\n",
        "[VFS TEST] FAIL - deferred unlink: content mismatch after unlink"
    );
    let n = expect_ok(
        with_vfs(|vfs| vfs.read(last.as_ref(), &mut buf)),
        "read through second handle after deferred unlink",
    );
    assert!(
        &buf[..n] == b"kernel=1\nversion=0.1\n",
        "[VFS TEST] FAIL - deferred unlink: second handle content mismatch"
    );

    // Only the final close may free the inode.
    expect_ok(
        with_vfs(|vfs| vfs.close(&first)),
        "close first handle after deferred unlink",
    );
    expect_ok(
        last.file.inode.filesystem.load_inode(last.file.inode.inode_number),
        "inode remains allocated until final close",
    );
    expect_err(
        with_vfs(|vfs| vfs.read(first.as_ref(), &mut buf)),
        FileSystemError::InvalidHandle,
        "read through closed first handle",
    );
    expect_ok(
        with_vfs(|vfs| vfs.close(&last)),
        "close final handle after deferred unlink",
    );
    expect_err(
        last.file.inode.filesystem.load_inode(last.file.inode.inode_number),
        FileSystemError::NotFound,
        "inode was not freed after deferred-unlink close",
    );

    // A relative unlink must resolve its target through the supplied working
    // directory, while an existing handle keeps the inode allocated.
    expect_ok(
        with_vfs(|vfs| vfs.create_file(
            "/var/log/deferred-relative.tmp",
            Permissions::all_permissive(),
            None,
        )),
        "create file for relative deferred unlink",
    );
    let log_dir = expect_ok(
        with_vfs(|vfs| vfs.resolve_dir_with_budget(
            "/var/log",
            None,
            crate::file_system::virtual_fs::resolver::MAX_SYMLINK_DEPTH,
        )),
        "resolve /var/log for relative deferred unlink",
    );
    let relative_handle = expect_ok(
        with_vfs(|vfs| vfs.open(
            "/var/log/deferred-relative.tmp",
            AccessMode::ReadOnly,
            None,
        )),
        "open file for relative deferred unlink",
    );
    expect_ok(
        with_vfs(|vfs| vfs.unlink(
            "deferred-relative.tmp",
            Some(log_dir),
        )),
        "unlink file relative to /var/log",
    );
    expect_err(
        with_vfs(|vfs| vfs.stat("/var/log/deferred-relative.tmp", None, false)),
        FileSystemError::NotFound,
        "stat after relative deferred unlink",
    );
    expect_ok(
        relative_handle.file.inode.filesystem
            .load_inode(relative_handle.file.inode.inode_number),
        "relative-unlinked inode remains until close",
    );
    expect_ok(
        with_vfs(|vfs| vfs.close(&relative_handle)),
        "close handle after relative deferred unlink",
    );
    expect_err(
        relative_handle.file.inode.filesystem
            .load_inode(relative_handle.file.inode.inode_number),
        FileSystemError::NotFound,
        "relative-unlinked inode freed after close",
    );

    sprintln!("[VFS TEST] POSIX deferred unlink ... PASS");
}

/// Checks namespace changes through an already loaded working directory.
///
/// # Panics
///
/// Panics when retained and fresh lookups disagree.
fn test_retained_directory_namespace() {
    sprintln!("[VFS TEST] retained directory namespace ...");

    let log_dir = expect_ok(
        with_vfs(|vfs| vfs.resolve_dir_with_budget(
            "/var/log", None,
            crate::file_system::virtual_fs::resolver::MAX_SYMLINK_DEPTH,
        )),
        "retain /var/log for namespace checks",
    );
    expect_ok(
        with_vfs(|vfs| vfs.stat("kernel.log", Some(Arc::clone(&log_dir)), false)),
        "visit a child through retained /var/log before mutation",
    );

    let created = expect_ok(
        with_vfs(|vfs| vfs.create_file(
            "/var/log/wrapper-new.tmp", Permissions::all_permissive(), None,
        )),
        "create file through a fresh path",
    );
    let relative = expect_ok(
        with_vfs(|vfs| vfs.stat("wrapper-new.tmp", Some(Arc::clone(&log_dir)), false)),
        "find created file through retained wrapper",
    );
    let absolute = expect_ok(
        with_vfs(|vfs| vfs.stat("/var/log/wrapper-new.tmp", None, false)),
        "find created file through fresh path",
    );
    assert!(relative.inode_number == created.inode_number
        && absolute.inode_number == created.inode_number,
        "[VFS TEST] FAIL - created file differs across wrappers");

    expect_ok(
        with_vfs(|vfs| vfs.rename(
            "/var/log/wrapper-new.tmp", "/var/log/wrapper-renamed.tmp", None,
        )),
        "rename file through fresh paths",
    );
    expect_err(
        with_vfs(|vfs| vfs.stat("wrapper-new.tmp", Some(Arc::clone(&log_dir)), false)),
        FileSystemError::NotFound,
        "old name through retained wrapper after rename",
    );
    let renamed = expect_ok(
        with_vfs(|vfs| vfs.stat("wrapper-renamed.tmp", Some(Arc::clone(&log_dir)), false)),
        "new name through retained wrapper after rename",
    );
    let fresh_renamed = expect_ok(
        with_vfs(|vfs| vfs.stat("/var/log/wrapper-renamed.tmp", None, false)),
        "new name through fresh path after rename",
    );
    assert!(renamed.inode_number == created.inode_number
        && fresh_renamed.inode_number == created.inode_number,
        "[VFS TEST] FAIL - renamed file differs across wrappers");

    expect_ok(
        with_vfs(|vfs| vfs.unlink("/var/log/wrapper-renamed.tmp", None)),
        "unlink file through fresh path",
    );
    expect_err(
        with_vfs(|vfs| vfs.stat("wrapper-renamed.tmp", Some(log_dir), false)),
        FileSystemError::NotFound,
        "unlinked name through retained wrapper",
    );
    expect_err(
        with_vfs(|vfs| vfs.stat("/var/log/wrapper-renamed.tmp", None, false)),
        FileSystemError::NotFound,
        "unlinked name through fresh path",
    );

    sprintln!("[VFS TEST] retained directory namespace ... PASS");
}

/// Checks mount visibility, failed unmount recovery, and unmount through
/// fresh and retained wrappers.
///
/// # Panics
///
/// Panics when a mount boundary is missed, a failed unmount removes the mount,
/// or a later unmount fails.
fn test_mount_wrapper_consistency() {
    sprintln!("[VFS TEST] mount wrapper consistency ...");

    let etc_dir = expect_ok(
        with_vfs(|vfs| vfs.resolve_dir_with_budget(
            "/etc", None,
            crate::file_system::virtual_fs::resolver::MAX_SYMLINK_DEPTH,
        )),
        "retain /etc before mount",
    );
    expect_ok(
        with_vfs(|vfs| vfs.stat("hosts.cfg", Some(Arc::clone(&etc_dir)), false)),
        "visit a child through retained /etc before covering it",
    );

    let device = Arc::new(StubBlockDevice);
    expect_ok(
        with_vfs(|vfs| vfs.mount("stubfs", device.clone(), "/etc", None)),
        "mount second stub instance on /etc",
    );
    // The second stub's own root appears at /etc. Its etc/ and bin/
    // directories therefore appear under /etc/etc and /etc/bin.
    let visible_root = expect_ok(
        with_vfs(|vfs| vfs.resolve("/etc", None, false)),
        "resolve mounted /etc root",
    );
    assert!(visible_root.filesystem.instance_id()
        != etc_dir.inode.filesystem.instance_id(),
        "[VFS TEST] FAIL - mount boundary missed: host instance {}, visible instance {}",
        etc_dir.inode.filesystem.instance_id(),
        visible_root.filesystem.instance_id());
    let visible_entries = expect_ok(
        visible_root.filesystem.read_dir(&visible_root),
        "read mounted /etc root",
    );
    assert!(visible_entries.iter().any(|entry| entry.name == "etc")
        && visible_entries.iter().any(|entry| entry.name == "bin"),
        "[VFS TEST] FAIL - mounted /etc root lacks stub root entries (instance {})",
        visible_root.filesystem.instance_id());
    let fresh = expect_ok(
        with_vfs(|vfs| vfs.resolve("/etc/bin/shell", None, false)),
        "fresh path enters mounted /etc",
    );
    let retained = expect_ok(
        with_vfs(|vfs| vfs.resolve("bin/shell", Some(Arc::clone(&etc_dir)), false)),
        "retained host enters mounted /etc",
    );
    assert!(fresh.filesystem.instance_id() == retained.filesystem.instance_id()
        && fresh.filesystem.instance_id() != etc_dir.inode.filesystem.instance_id(),
        "[VFS TEST] FAIL - wrappers see different mounted instances");
    expect_err(
        with_vfs(|vfs| vfs.stat("hosts.cfg", Some(Arc::clone(&etc_dir)), false)),
        FileSystemError::NotFound,
        "retained host hides covered children while mounted",
    );
    let nested = expect_ok(
        with_vfs(|vfs| vfs.resolve("/etc/etc/config.txt", None, false)),
        "same inode number inside mounted instance is not the host mount",
    );
    assert!(nested.filesystem.instance_id() == fresh.filesystem.instance_id(),
        "[VFS TEST] FAIL - inode number collision crossed the mount again");
    expect_err(
        with_vfs(|vfs| vfs.mount("stubfs", device.clone(), "/etc", None)),
        FileSystemError::AlreadyMounted,
        "second mount on /etc",
    );
    expect_err(
        with_vfs(|vfs| vfs.remove_directory("/etc", None)),
        FileSystemError::ResourceBusy,
        "remove covered /etc mount host",
    );
    let busy_handle = open_file("/etc/bin/shell", AccessMode::ReadOnly);
    expect_err(
        with_vfs(|vfs| vfs.unmount("/etc", None)),
        FileSystemError::FileLocked,
        "unmount with an open file on mounted volume",
    );
    close_file(&busy_handle);

    // A failed driver flush must leave both mount indexes intact. The stub
    // injects one failure for this mounted instance, then permits a retry.
    super::fail_next_unmount_for(visible_root.filesystem.instance_id());
    expect_err(
        with_vfs(|vfs| vfs.unmount("/etc", None)),
        FileSystemError::DiskWriteError,
        "unmount with a simulated flush failure",
    );
    expect_ok(
        with_vfs(|vfs| vfs.stat("/etc/bin/shell", None, false)),
        "fresh path still enters mount after failed unmount",
    );
    expect_ok(
        with_vfs(|vfs| vfs.stat("bin/shell", Some(Arc::clone(&etc_dir)), false)),
        "retained host still enters mount after failed unmount",
    );
    let parent_after_failure = expect_ok(
        with_vfs(|vfs| vfs.resolve("/etc/..", None, false)),
        "mounted root still crosses to host after failed unmount",
    );
    assert!(
        parent_after_failure.inode_number == etc_dir.inode.inode_number
            && parent_after_failure.filesystem.instance_id()
                == etc_dir.inode.filesystem.instance_id(),
        "[VFS TEST] FAIL - mounted-root parent index lost after failed unmount",
    );
    expect_err(
        with_vfs(|vfs| vfs.mount("stubfs", device.clone(), "/etc", None)),
        FileSystemError::AlreadyMounted,
        "mount remains registered after failed unmount",
    );
    expect_ok(
        with_vfs(|vfs| vfs.unmount("/etc", None)),
        "retry unmount /etc through fresh path",
    );
    expect_err(
        with_vfs(|vfs| vfs.stat("/etc/bin/shell", None, false)),
        FileSystemError::NotFound,
        "covered /etc restored after unmount",
    );

    expect_ok(
        with_vfs(|vfs| vfs.mount(
            "stubfs", device, ".", Some(Arc::clone(&etc_dir)),
        )),
        "mount through retained host",
    );
    expect_ok(
        with_vfs(|vfs| vfs.stat("bin/shell", Some(Arc::clone(&etc_dir)), false)),
        "retained host sees remount",
    );
    expect_ok(
        with_vfs(|vfs| vfs.unmount(".", Some(Arc::clone(&etc_dir)))),
        "unmount through retained host",
    );
    expect_ok(
        with_vfs(|vfs| vfs.stat("hosts.cfg", Some(etc_dir), false)),
        "retained host sees uncovered /etc again",
    );

    sprintln!("[VFS TEST] mount wrapper consistency ... PASS");
}

/// Checks `..` from a retained working directory after moving that directory.
///
/// The VFS records a cross-parent move in its relocated-parent table. Fresh
/// and retained wrappers must agree after moving to `/bin` and back to `/var`.
///
/// # Panics
///
/// Panics if either parent lookup fails or they disagree.
fn test_moved_working_directory_parent() {
    sprintln!("[VFS TEST] moved working directory parent ...");

    let retained_log = expect_ok(
        with_vfs(|vfs| vfs.resolve_dir_with_budget(
            "/var/log", None,
            crate::file_system::virtual_fs::resolver::MAX_SYMLINK_DEPTH,
        )),
        "retain /var/log before moving it",
    );
    expect_ok(
        with_vfs(|vfs| vfs.rename("/var/log", "/bin/log", None)),
        "move /var/log to /bin/log",
    );

    let moved = expect_ok(
        with_vfs(|vfs| vfs.stat("/bin/log", None, false)),
        "find moved directory through fresh path",
    );
    assert!(moved.inode_number == retained_log.inode.inode_number,
        "[VFS TEST] FAIL - moving /var/log changed its inode");

    let bin = expect_ok(
        with_vfs(|vfs| vfs.stat("/bin", None, false)),
        "stat expected new parent /bin",
    );
    let fresh_log = expect_ok(
        with_vfs(|vfs| vfs.resolve_dir_with_budget(
            "/bin/log", None,
            crate::file_system::virtual_fs::resolver::MAX_SYMLINK_DEPTH,
        )),
        "resolve moved directory through fresh path",
    );
    let fresh_parent = expect_ok(
        with_vfs(|vfs| vfs.stat("..", Some(fresh_log), false)),
        "resolve .. from fresh /bin/log wrapper",
    );
    assert!(fresh_parent.inode_number == bin.inode_number,
        "[VFS TEST] FAIL - fresh moved-directory parent is not /bin");

    let retained_parent = expect_ok(
        with_vfs(|vfs| vfs.stat("..", Some(Arc::clone(&retained_log)), false)),
        "resolve .. from retained moved-directory wrapper",
    );
    assert!(retained_parent.inode_number == bin.inode_number,
        "[VFS TEST] FAIL - retained moved-directory parent is inode {}, expected /bin inode {}",
        retained_parent.inode_number, bin.inode_number);

    expect_ok(
        with_vfs(|vfs| vfs.rename("/bin/log", "/var/log", None)),
        "restore /var/log after parent check",
    );
    let var = expect_ok(
        with_vfs(|vfs| vfs.stat("/var", None, false)),
        "stat restored parent /var",
    );
    let restored_parent = expect_ok(
        with_vfs(|vfs| vfs.stat("..", Some(retained_log), false)),
        "resolve .. from retained wrapper after restoring /var/log",
    );
    assert!(restored_parent.inode_number == var.inode_number,
        "[VFS TEST] FAIL - retained directory still has old parent after restoring /var/log");

    sprintln!("[VFS TEST] moved working directory parent ... PASS");
}

/// Checks that a trailing slash requires the final component to be a directory.
///
/// File creation and lookup must reject a trailing slash without creating or
/// removing a file. Directory creation, lookup, and removal may use one. A
/// trailing slash follows a final symlink during lookup only when its target
/// is a directory. Namespace mutation methods apply their own final-link rules.
///
/// # Panics
///
/// Panics if an operation returns the wrong result or a rejected operation
/// changes the namespace.
fn test_trailing_slash_paths() {
    sprintln!("[VFS TEST] trailing slash paths ...");

    expect_err(
        with_vfs(|vfs| vfs.create_file(
            "/var/log/trailing-file.tmp/",
            Permissions::all_permissive(),
            None,
        )),
        FileSystemError::NotADirectory,
        "create_file with trailing slash",
    );
    expect_err(
        with_vfs(|vfs| vfs.stat("/var/log/trailing-file.tmp", None, false)),
        FileSystemError::NotFound,
        "file was not created after trailing-slash rejection",
    );

    let log_dir = expect_ok(
        with_vfs(|vfs| vfs.resolve_dir_with_budget(
            "/var/log", None,
            crate::file_system::virtual_fs::resolver::MAX_SYMLINK_DEPTH,
        )),
        "resolve /var/log for relative trailing-slash creation",
    );
    expect_err(
        with_vfs(|vfs| vfs.create_file(
            "trailing-file.tmp/",
            Permissions::all_permissive(),
            Some(log_dir),
        )),
        FileSystemError::NotADirectory,
        "relative create_file with trailing slash",
    );

    expect_err(
        with_vfs(|vfs| vfs.create_symlink(
            "/etc/trailing-link/", "/etc/hosts.cfg", None,
        )),
        FileSystemError::NotADirectory,
        "create_symlink with trailing slash",
    );
    expect_err(
        with_vfs(|vfs| vfs.stat("/etc/trailing-link", None, false)),
        FileSystemError::NotFound,
        "symlink was not created after trailing-slash rejection",
    );

    // The deferred-unlink test removed config.txt. The rename test restored
    // hosts.cfg, so it remains a live regular-file fixture here.
    let regular_file = expect_ok(
        with_vfs(|vfs| vfs.stat("/etc/hosts.cfg", None, false)),
        "verify live regular-file fixture for trailing-slash checks",
    );
    assert!(regular_file.inode_type == InodeType::File,
        "[VFS TEST] FAIL - trailing-slash regular-file fixture is not a file");
    expect_err(
        with_vfs(|vfs| vfs.stat("/etc/hosts.cfg/", None, false)),
        FileSystemError::NotADirectory,
        "stat regular file with trailing slash",
    );
    expect_err(
        with_vfs(|vfs| vfs.open(
            "/etc/hosts.cfg/", AccessMode::ReadOnly, None,
        )),
        FileSystemError::NotADirectory,
        "open regular file with trailing slash",
    );
    expect_err(
        with_vfs(|vfs| vfs.unlink("/etc/hosts.cfg/", None)),
        FileSystemError::NotADirectory,
        "unlink regular file with trailing slash",
    );
    expect_err(
        with_vfs(|vfs| vfs.rename(
            "/etc/hosts.cfg/", "/etc/trailing-renamed.txt", None,
        )),
        FileSystemError::NotADirectory,
        "rename regular file from trailing-slash path",
    );
    expect_err(
        with_vfs(|vfs| vfs.rename(
            "/etc/hosts.cfg", "/etc/trailing-renamed.txt/", None,
        )),
        FileSystemError::NotADirectory,
        "rename regular file to trailing-slash path",
    );
    expect_ok(
        with_vfs(|vfs| vfs.stat("/etc/hosts.cfg", None, false)),
        "regular file remains after rejected trailing-slash mutations",
    );
    expect_err(
        with_vfs(|vfs| vfs.stat("/etc/trailing-renamed.txt", None, false)),
        FileSystemError::NotFound,
        "rename destination was not created after trailing-slash rejection",
    );

    let linked_directory = expect_ok(
        with_vfs(|vfs| vfs.stat("/lib/", None, false)),
        "stat symlink to directory with trailing slash",
    );
    assert!(linked_directory.inode_type == InodeType::Directory,
        "[VFS TEST] FAIL - trailing slash did not follow symlink to directory");
    expect_ok(
        with_vfs(|vfs| vfs.create_symlink(
            "/etc/trailing-file-link", "/etc/hosts.cfg", None,
        )),
        "create symlink to live file for trailing-slash lookup",
    );
    expect_err(
        with_vfs(|vfs| vfs.stat("/etc/trailing-file-link/", None, false)),
        FileSystemError::NotADirectory,
        "stat symlink to file with trailing slash",
    );
    expect_ok(
        with_vfs(|vfs| vfs.unlink("/etc/trailing-file-link", None)),
        "unlink symlink used by trailing-slash lookup",
    );

    expect_ok(
        with_vfs(|vfs| vfs.create_directory(
            "/bin/trailing-dir/", Permissions::all_permissive(), None,
        )),
        "create_directory with trailing slash",
    );
    let directory = expect_ok(
        with_vfs(|vfs| vfs.stat("/bin/trailing-dir/", None, false)),
        "stat directory with trailing slash",
    );
    assert!(directory.inode_type == InodeType::Directory,
        "[VFS TEST] FAIL - trailing-slash directory lookup returned another inode type");
    expect_ok(
        with_vfs(|vfs| vfs.remove_directory("/bin/trailing-dir/", None)),
        "remove directory with trailing slash",
    );

    sprintln!("[VFS TEST] trailing slash paths ... PASS");
}

/// Checks hard-link identity, failure behavior, and final-name reclamation.
///
/// Uses private fixtures so existing tests retain their expected namespace.
/// Direct driver loads verify backing counts independently of VFS retention.
///
/// # Panics
///
/// Panics if a link changes identity, a failed mutation changes counts or
/// entries, alias data differs, or storage is freed at the wrong lifetime.
fn test_hard_links() {
    sprintln!("[VFS TEST] hard links ...");

    with_vfs(|vfs| {
        let source = expect_ok(vfs.create_file(
            "/etc/hard-source", Permissions::all_permissive(), None,
        ), "create hard-link source");
        let cwd = expect_ok(vfs.resolve_dir_with_budget(
            "/etc", None,
            crate::file_system::virtual_fs::resolver::MAX_SYMLINK_DEPTH,
        ), "retain hard-link destination parent");

        expect_ok(vfs.create_hard_link(
            "hard-source",
            "../var/log/hard-alias",
            Some(Arc::clone(&cwd)),
            false,
        ), "create cross-directory relative hard link");

        let alias = expect_ok(vfs.resolve(
            "/var/log/hard-alias", None, false,
        ), "resolve hard-link alias");
        assert!(Arc::ptr_eq(&source, &alias),
            "[VFS TEST] FAIL - hard-link names must share the canonical inode");
        assert_eq!(source.stat().link_count, 2);

        let backing = expect_ok(source.filesystem.load_inode(
            source.inode_number,
        ), "reload backing count after link");
        assert_eq!(backing.stat().link_count, 2);

        let first = expect_ok(vfs.open(
            "/etc/hard-source", AccessMode::ReadWrite, None,
        ), "open first name");
        let second = expect_ok(vfs.open(
            "/var/log/hard-alias", AccessMode::ReadWrite, None,
        ), "open second name");

        assert!(Arc::ptr_eq(&first.file, &second.file),
            "[VFS TEST] FAIL - hard-link opens must share the file master");
        assert_eq!(
            expect_ok(vfs.write(&first, b"shared"), "write first name"),
            6,
        );

        let mut data = [0u8; 6];
        assert_eq!(
            expect_ok(vfs.read(&second, &mut data), "read second name"),
            6,
        );
        assert_eq!(&data, b"shared");
        assert_eq!(alias.stat().size, 6);

        expect_ok(vfs.rename(
            "/etc/hard-source", "/var/log/hard-alias", None,
        ), "rename between aliases is a no-op");
        assert_eq!(expect_ok(vfs.stat(
            "/etc/hard-source", None, false,
        ), "source survives same-inode rename").link_count, 2);
        assert_eq!(expect_ok(vfs.stat(
            "/var/log/hard-alias", None, false,
        ), "alias survives same-inode rename").link_count, 2);

        // VFS preflight failures must not publish another name or count.
        expect_err(vfs.create_hard_link(
            "/etc/hard-source", "/var/log/hard-alias", None, false,
        ), FileSystemError::AlreadyExists, "existing alias destination");

        expect_err(vfs.create_hard_link(
            "/etc/hard-source", "/etc/hard-source", None, false,
        ), FileSystemError::AlreadyExists, "hard link onto its own name");

        expect_err(vfs.create_hard_link(
            "/etc", "/etc/hard-directory", None, false,
        ), FileSystemError::OperationNotAllowed, "directory hard link");

        expect_err(vfs.create_hard_link(
            "/etc/hard-source", "/etc/hard-slash/", None, false,
        ), FileSystemError::NotADirectory, "destination trailing slash");

        expect_err(vfs.create_hard_link(
            "/etc/hard-source/", "/etc/hard-source-slash", None, false,
        ), FileSystemError::NotADirectory, "source trailing slash");

        for name in ["/etc/.", "/etc/..", "/etc/hard\0invalid"] {
            expect_err(vfs.create_hard_link(
                "/etc/hard-source", name, None, false,
            ), FileSystemError::InvalidPath, "invalid destination component");
        }

        expect_err(vfs.create_hard_link(
            "/etc/hard-missing", "/etc/hard-uncreated", None, false,
        ), FileSystemError::NotFound, "missing source");

        expect_err(vfs.create_hard_link(
            "/etc/hard-source", "/etc/hard-missing/child", None, false,
        ), FileSystemError::NotFound, "missing destination parent");

        // Exercise the driver's own failed-mutation path, bypassing VFS
        // preflight but leaving the canonical count untouched.
        expect_err(source.filesystem.link(
            &source, &cwd.inode, "hard-source",
        ), FileSystemError::AlreadyExists, "driver rejects duplicate name");
        assert_eq!(source.stat().link_count, 2);
        assert_eq!(expect_ok(source.filesystem.load_inode(
            source.inode_number,
        ), "backing count after rejected driver link").stat().link_count, 2);

        // Temporarily raise only the VFS count to exercise its arithmetic
        // limit. Restore it before any successful driver mutation.
        source.metadata.lock().link_count = u32::MAX;
        let overflow = vfs.create_hard_link(
            "/etc/hard-source", "/etc/hard-overflow", None, false,
        );
        source.metadata.lock().link_count = 2;
        expect_err(
            overflow,
            FileSystemError::TooManyHardLinks,
            "VFS count limit",
        );

        for path in [
            "/etc/hard-directory",
            "/etc/hard-slash",
            "/etc/hard-source-slash",
            "/etc/hard-uncreated",
            "/etc/hard-overflow",
        ] {
            expect_err(
                vfs.resolve(path, None, false),
                FileSystemError::NotFound,
                "rejected link left no entry",
            );
        }

        // First unlink leaves a live alias. Closing every current handle
        // must not free it; a later open through that alias still succeeds.
        expect_ok(vfs.unlink(
            "/etc/hard-source", None,
        ), "unlink first name");
        assert_eq!(source.stat().link_count, 1);
        assert_eq!(expect_ok(source.filesystem.load_inode(
            source.inode_number,
        ), "backing count after first unlink").stat().link_count, 1);

        expect_ok(vfs.close(&first), "close first handle with alias remaining");
        expect_ok(vfs.close(&second), "close second handle with alias remaining");

        let reopened = expect_ok(vfs.open(
            "/var/log/hard-alias", AccessMode::ReadWrite, None,
        ), "reopen surviving alias");
        let independent = expect_ok(vfs.open(
            "/var/log/hard-alias", AccessMode::ReadOnly, None,
        ), "second open before final unlink");

        expect_ok(vfs.unlink(
            "/var/log/hard-alias", None,
        ), "unlink final name");
        assert_eq!(source.stat().link_count, 0);

        expect_err(vfs.open(
            "/var/log/hard-alias", AccessMode::ReadOnly, None,
        ), FileSystemError::NotFound, "final name is gone");
        assert_eq!(expect_ok(source.filesystem.load_inode(
            source.inode_number,
        ), "unlinked backing inode remains open").stat().link_count, 0);

        let mut after_unlink = [0u8; 6];
        assert_eq!(expect_ok(
            vfs.read(&reopened, &mut after_unlink),
            "read after final unlink",
        ), 6);
        assert_eq!(&after_unlink, b"shared");

        expect_ok(vfs.close(&reopened), "close one unlinked open");
        expect_ok(
            source.filesystem.load_inode(source.inode_number),
            "storage survives a non-final close",
        );
        expect_ok(vfs.close(&independent), "final close reclaims inode");
        expect_err(
            source.filesystem.load_inode(source.inode_number),
            FileSystemError::NotFound,
            "backing inode reclaimed",
        );

        // No-open-handle final unlink must reclaim storage immediately too.
        let immediate = expect_ok(vfs.create_file(
            "/etc/hard-immediate", Permissions::all_permissive(), None,
        ), "create immediate-free fixture");

        expect_ok(vfs.create_hard_link(
            "/etc/hard-immediate",
            "/etc/hard-immediate-alias",
            None,
            false,
        ), "link immediate-free fixture");

        expect_ok(vfs.unlink(
            "/etc/hard-immediate", None,
        ), "first immediate unlink");
        expect_ok(vfs.unlink(
            "/etc/hard-immediate-alias", None,
        ), "last immediate unlink");
        expect_err(
            immediate.filesystem.load_inode(immediate.inode_number),
            FileSystemError::NotFound,
            "immediate inode reclaimed",
        );

        // Both source-symlink policies, including dangling targets.
        let target = expect_ok(vfs.create_file(
            "/etc/hard-target", Permissions::all_permissive(), None,
        ), "create symlink target");
        let symlink = expect_ok(vfs.create_symlink(
            "/etc/hard-symlink", "/etc/hard-target", None,
        ), "create source symlink");

        expect_ok(vfs.create_hard_link(
            "/etc/hard-symlink", "/etc/hard-symlink-alias", None, false,
        ), "link symlink itself");
        expect_ok(vfs.create_hard_link(
            "/etc/hard-symlink", "/etc/hard-target-alias", None, true,
        ), "link followed symlink target");

        assert!(Arc::ptr_eq(&symlink, &expect_ok(vfs.resolve(
            "/etc/hard-symlink-alias", None, false,
        ), "resolve symlink alias without following")));
        assert!(Arc::ptr_eq(&target, &expect_ok(vfs.resolve(
            "/etc/hard-target-alias", None, false,
        ), "resolve followed-target alias")));
        assert_eq!(symlink.stat().link_count, 2);
        assert_eq!(target.stat().link_count, 2);

        let dangling = expect_ok(vfs.create_symlink(
            "/etc/hard-dangling", "/etc/hard-absent-target", None,
        ), "create dangling symlink");
        expect_ok(vfs.create_hard_link(
            "/etc/hard-dangling", "/etc/hard-dangling-alias", None, false,
        ), "link dangling symlink inode");
        expect_err(vfs.create_hard_link(
            "/etc/hard-dangling", "/etc/hard-dangling-followed", None, true,
        ), FileSystemError::NotFound, "follow dangling source symlink");
        expect_err(vfs.create_hard_link(
            "/etc/hard-target", "/etc/hard-dangling", None, false,
        ), FileSystemError::AlreadyExists, "dangling destination is occupied");
        assert_eq!(dangling.stat().link_count, 2);

        // A destination on another mounted instance cannot name this inode.
        expect_ok(vfs.create_directory(
            "/etc/hard-mount", Permissions::all_permissive(), None,
        ), "create hard-link mount host");
        expect_ok(vfs.mount(
            "stubfs", Arc::new(StubBlockDevice), "/etc/hard-mount", None,
        ), "mount hard-link boundary fixture");

        expect_err(vfs.create_hard_link(
            "/etc/hard-target", "/etc/hard-mount/hard-cross", None, false,
        ), FileSystemError::CrossFileSystemOperation, "cross-instance hard link");
        expect_err(
            vfs.resolve("/etc/hard-mount/hard-cross", None, false),
            FileSystemError::NotFound,
            "no cross-instance destination created",
        );
        assert_eq!(target.stat().link_count, 2);

        expect_ok(vfs.unmount(
            "/etc/hard-mount", None,
        ), "unmount boundary fixture");
        expect_ok(vfs.remove_directory(
            "/etc/hard-mount", None,
        ), "remove mount host");

        for path in [
            "/etc/hard-symlink",
            "/etc/hard-symlink-alias",
            "/etc/hard-dangling",
            "/etc/hard-dangling-alias",
            "/etc/hard-target",
            "/etc/hard-target-alias",
        ] {
            expect_ok(vfs.unlink(path, None), "remove hard-link fixture");
        }
    });

    sprintln!("[VFS TEST] hard links ... PASS");
}

/// Releases the global test VFS after all regression groups complete.
///
/// The root has no host entry in the mount table. Mounted-volume unmount
/// behavior is exercised by the mount and replacement regressions. This
/// operation removes the global owner and drops it outside the global lock;
/// it does not assert that no independently retained references exist.
fn teardown_vfs() {
    let vfs = {
        let mut guard = globals::VIRTUAL_FILE_SYSTEM.lock();
        guard.take()
    };
    drop(vfs);
    sprintln!("[VFS TEST] VFS teardown complete");
}

/// Exercises distinct removal contracts, detached lifetimes, and path guards.
///
/// Runs against isolated fixtures after the hard-link tests. Driver loads
/// distinguish preserved backing storage from merely retained Arc objects.
/// Guards cover fresh and retained wrappers, moved ancestry, mount boundaries,
/// and independent holders. All fixtures are removed before suite teardown.
///
/// # Panics
///
/// Panics on the first unexpected result or inconsistent lifetime observation.
fn test_directory_removal_and_guards() {
    sprintln!("[VFS TEST] directory removal and path guards ...");
    with_vfs(|vfs| {
        let budget = crate::file_system::virtual_fs::resolver::MAX_SYMLINK_DEPTH;
        let permissions = Permissions::all_permissive();
        for path in ["/etc/remove-case", "/etc/remove-case/a", "/etc/remove-case/b",
            "/etc/remove-case/a/leaf"] {
            expect_ok(vfs.create_directory(path, permissions.clone(), None), "create removal fixture");
        }
        expect_ok(vfs.create_file("/etc/remove-case/file", permissions.clone(), None), "create file");
        expect_ok(vfs.create_symlink("/etc/remove-case/link", "/etc/remove-case/a",
            None), "create directory symlink");
        expect_ok(vfs.create_symlink("/etc/remove-case/dangling", "/absent-target",
            None), "create dangling symlink");

        for path in ["", "/", "////", ".", "..", "/etc/remove-case/a/.",
            "/etc/remove-case/a/..///"] {
            let error = if path.is_empty() { FileSystemError::NotFound }
                else if path.trim_matches('/').is_empty() { FileSystemError::ResourceBusy }
                else { FileSystemError::OperationNotAllowed };
            expect_err(vfs.unlink(path, None), error.clone(), "unlink special spelling");
            expect_err(vfs.remove_directory(path, None), error, "rmdir special spelling");
        }
        for path in ["/etc/remove-case/file", "/etc/remove-case/link", "/etc/remove-case/dangling",
            "/etc/remove-case/file///", "/etc/remove-case/link///", "/etc/remove-case/dangling///"] {
            expect_err(vfs.remove_directory(path, None), FileSystemError::NotADirectory, "rmdir type");
        }
        for path in ["/etc/remove-case/file///", "/etc/remove-case/link///", "/etc/remove-case/dangling///"] {
            expect_err(vfs.unlink(path, None), FileSystemError::NotADirectory, "unlink trailing slash");
        }
        for path in ["/etc/remove-case/a", "/etc/remove-case/a/leaf///"] {
            expect_err(vfs.unlink(path, None), FileSystemError::OperationNotAllowed, "unlink directory");
        }
        expect_err(vfs.remove_directory("/etc/remove-case/a", None),
            FileSystemError::DirectoryNotEmpty, "nonempty directory");

        let retained = expect_ok(vfs.resolve_dir_with_budget("/etc/remove-case/a/leaf", None, budget), "retain leaf");
        let other_wrapper = expect_ok(vfs.resolve_dir_with_budget("/etc/remove-case/a/leaf", None, budget), "second wrapper");
        let first = expect_ok(vfs.protect_directory_path(retained.clone()), "first guard");
        let second = expect_ok(vfs.protect_directory_path(other_wrapper.clone()), "second guard");
        for path in ["/etc/remove-case", "/etc/remove-case/a", "/etc/remove-case/a/leaf"] {
            expect_err(vfs.remove_directory(path, None), FileSystemError::ResourceBusy, "guarded removal");
            expect_err(vfs.rename(path, "/etc/guard-moved", None), FileSystemError::ResourceBusy, "guarded rename");
            expect_err(vfs.mount("stubfs", Arc::new(StubBlockDevice), path, None),
                FileSystemError::ResourceBusy, "guarded mount");
        }
        expect_ok(vfs.rename("/etc/remove-case/a/leaf", "/etc/remove-case/a/leaf", None), "guarded no-op");
        expect_ok(vfs.create_file("/etc/remove-case/a/leaf/child", permissions.clone(), None), "guard allows content creation");
        expect_ok(vfs.rename("/etc/remove-case/a/leaf/child", "/etc/remove-case/a/leaf/renamed", None), "guard allows child rename");
        expect_ok(vfs.unlink("/etc/remove-case/a/leaf/renamed", None), "guard allows child unlink");
        drop(first);
        expect_err(vfs.remove_directory("/etc/remove-case/a/leaf", None), FileSystemError::ResourceBusy, "second guard remains");
        drop(second);

        // Move before reacquiring protection: stale wrapper parents must not
        // protect the old ancestry or allow mutation of the new ancestry.
        expect_ok(vfs.rename("/etc/remove-case/a/leaf", "/etc/remove-case/b/leaf", None), "move unprotected leaf");
        let moved_guard = expect_ok(vfs.protect_directory_path(retained.clone()), "guard moved wrapper");
        expect_err(vfs.rename("/etc/remove-case/b", "/etc/remove-case/b2", None), FileSystemError::ResourceBusy, "new ancestor protected");
        expect_ok(vfs.remove_directory("/etc/remove-case/a", None), "old ancestor unprotected");
        drop(moved_guard);
        expect_ok(vfs.remove_directory("/etc/remove-case/b/leaf///", None), "remove retained leaf");
        assert_eq!(retained.inode.stat().link_count, 0);
        assert!(expect_ok(retained.inode.filesystem.read_dir(&retained.inode), "detached listing").is_empty());
        expect_err(vfs.resolve("/etc/remove-case/b/leaf", None, false), FileSystemError::NotFound, "name detached");
        expect_err(vfs.protect_directory_path(retained.clone()), FileSystemError::NotFound, "cannot guard detached leaf");
        expect_err(vfs.create_file("new", permissions.clone(), Some(retained.clone())), FileSystemError::NotFound, "cannot create in detached leaf");
        expect_err(vfs.create_directory("new", permissions.clone(), Some(retained.clone())), FileSystemError::NotFound, "cannot mkdir in detached leaf");
        expect_err(vfs.create_symlink("new", "/etc", Some(retained.clone())), FileSystemError::NotFound, "cannot symlink in detached leaf");
        expect_err(vfs.create_hard_link("/etc/remove-case/file", "new", Some(retained.clone()), false), FileSystemError::NotFound, "cannot link into detached leaf");
        expect_err(vfs.rename("/etc/remove-case/file", "new", Some(retained.clone())), FileSystemError::NotFound, "cannot move into detached leaf");
        expect_err(vfs.mount("stubfs", Arc::new(StubBlockDevice), ".", Some(retained.clone())), FileSystemError::NotFound, "cannot mount detached leaf");
        let parent = expect_ok(vfs.resolve("..", Some(retained.clone()), false), "former parent traversal");
        let b = expect_ok(vfs.resolve("/etc/remove-case/b", None, false), "current former parent");
        assert!(Arc::ptr_eq(&parent, &b));
        let leaf_inode = retained.inode.clone();
        drop(retained);
        expect_ok(vfs.reclaim_unlinked_inodes(), "cleanup while second wrapper remains");
        expect_ok(leaf_inode.filesystem.load_inode(leaf_inode.inode_number), "second wrapper retains storage");
        drop(other_wrapper);
        expect_ok(vfs.reclaim_unlinked_inodes(), "cleanup after final wrapper");
        expect_err(leaf_inode.filesystem.load_inode(leaf_inode.inode_number), FileSystemError::NotFound, "detached storage reclaimed");
        expect_ok(vfs.unlink("/etc/remove-case/b/../file", None), "intermediate traversal");
        expect_ok(vfs.unlink("/etc/remove-case/link", None), "unlink directory symlink itself");
        expect_ok(vfs.unlink("/etc/remove-case/dangling", None), "unlink dangling symlink");

        expect_ok(vfs.mount("stubfs", Arc::new(StubBlockDevice), "/etc/remove-case/b", None), "mount guard fixture");
        let inside = expect_ok(vfs.resolve_dir_with_budget("/etc/remove-case/b/etc", None, budget), "mounted directory");
        let mount_guard = expect_ok(vfs.protect_directory_path(inside.clone()), "guard across mount");
        expect_err(vfs.unmount("/etc/remove-case/b", None), FileSystemError::ResourceBusy, "guard blocks unmount");
        expect_err(vfs.rename("/etc/remove-case", "/etc/remove-case-moved", None), FileSystemError::ResourceBusy, "guard crosses host ancestry");
        expect_err(vfs.unlink("/etc/remove-case/b", None), FileSystemError::OperationNotAllowed, "unlink mount host");
        expect_err(vfs.remove_directory("/etc/remove-case/b", None), FileSystemError::ResourceBusy, "rmdir mount host");
        drop(mount_guard);
        drop(inside);
        expect_ok(vfs.unmount("/etc/remove-case/b", None), "unmount after guard release");
        expect_err(vfs.remove_directory("/etc/remove-case/b/../b/.", None), FileSystemError::OperationNotAllowed, "final dot remains rejected");
        expect_ok(vfs.remove_directory("/etc/remove-case/b/../b///", None), "intermediate rmdir traversal");
        expect_ok(vfs.remove_directory("/etc/remove-case", None), "remove fixture root");
        drop(parent);
        drop(b);
        expect_ok(vfs.reclaim_unlinked_inodes(), "reclaim fixture ancestry");
    });
    sprintln!("[VFS TEST] directory removal and path guards ... PASS");
}


/// Creates an isolated stub-backed VFS for a regression group.
///
/// Each fixture has its own instance identity and original stub tree. Tests
/// have exclusive ownership of the local VFS and explicitly close every open
/// registration before dropping it. No global VFS state is replaced.
///
/// # Returns
///
/// Returns a fresh VFS with the stub driver registered for additional mounts.
///
/// # Panics
///
/// Panics if the stub instance or its root cannot be acquired.
fn new_stub_vfs() -> VirtualFileSystem {
    let driver = Arc::new(StubFileSystemDriver);
    let instance = expect_ok(driver.mount(Arc::new(StubBlockDevice)), "isolated stub mount");
    let root = expect_ok(instance.load_inode(1), "isolated stub root");
    let mut vfs = VirtualFileSystem::new(Arc::new(Directory::new_root(root)));
    vfs.register_driver(driver);
    vfs
}

/// Checks name protection across aliases, independent opens, and Arc clones.
///
/// Rejected mutations must preserve both names, backing link counts, and inode
/// identity. Content transfers and creation of additional aliases remain legal.
/// Closing one registration invalidates its clones and releases only its own
/// protection; an ordinary open does not override another holder's restriction.
///
/// # Panics
///
/// Panics on an unexpected result, identity change, count, or content transfer.
fn test_namespace_name_protection() {
    sprintln!("[VFS TEST] namespace name protection ...");
    let mut vfs = new_stub_vfs();
    let file = expect_ok(vfs.create_file("/etc/protected", Permissions::all_permissive(), None), "create protected file");
    expect_ok(vfs.create_hard_link("/etc/protected", "/var/alias", None, false), "create alias");
    let first = expect_ok(vfs.open_with_namespace_protection("/etc/protected", AccessMode::ReadWrite,
        None, NamespaceProtection::PROTECT_NAMES), "first protected open");
    let clone = first.clone();
    let second = expect_ok(vfs.open_with_namespace_protection("/var/alias", AccessMode::ReadOnly,
        None, NamespaceProtection::PROTECT_NAMES), "independent alias open");
    let ordinary = expect_ok(vfs.open("/var/alias", AccessMode::ReadWrite, None), "ordinary open alongside protection");
    assert_eq!(expect_ok(vfs.write(&ordinary, b"shared"), "ordinary write remains allowed"), 6);
    let mut buffer = [0u8; 6];
    assert_eq!(expect_ok(vfs.read(&first, &mut buffer), "protected read remains allowed"), 6);
    assert_eq!(&buffer, b"shared");
    expect_ok(vfs.create_hard_link("/etc/protected", "/etc/new-alias", None, false), "alias creation under protection");
    for path in ["/etc/protected", "/var/alias", "/etc/new-alias"] {
        expect_err(vfs.unlink(path, None), FileSystemError::ResourceBusy, "every alias protected against unlink");
        expect_err(vfs.rename(path, "/etc/moved", None), FileSystemError::ResourceBusy, "every alias protected against rename or move");
        let observed = expect_ok(vfs.resolve(path, None, false), "rejected mutation preserves name");
        assert!(Arc::ptr_eq(&observed, &file));
    }
    assert_eq!(file.stat().link_count, 3);
    assert_eq!(expect_ok(file.filesystem.load_inode(file.inode_number), "backing count after rejected mutations").stat().link_count, 3);
    expect_ok(vfs.rename("/etc/protected", "/var/alias", None), "protected same-inode no-op");
    expect_ok(vfs.create_file("/etc/other", Permissions::all_permissive(), None), "create competing source");
    expect_err(vfs.rename("/etc/other", "/etc/protected", None), FileSystemError::ResourceBusy, "protected replacement destination preflight");
    expect_ok(vfs.resolve("/etc/other", None, false), "replacement rejection preserves source");
    expect_ok(vfs.close(&first), "close first registration");
    expect_err(vfs.close(&clone), FileSystemError::InvalidHandle, "clone does not own another registration");
    expect_err(vfs.unlink("/var/alias", None), FileSystemError::ResourceBusy, "second registration still protects aliases");
    expect_ok(vfs.close(&second), "close independent registration");
    // The closed Arc clone still exists, but cannot retain protection.
    expect_ok(vfs.rename("/etc/protected", "/etc/renamed", None), "final protected close releases names");
    expect_ok(vfs.unlink("/etc/new-alias", None), "unlink new alias after release");
    expect_ok(vfs.unlink("/var/alias", None), "unlink original alias after release");
    expect_ok(vfs.unlink("/etc/renamed", None), "ordinary handle permits final unlink");
    expect_ok(file.filesystem.load_inode(file.inode_number), "ordinary handle retains unlinked backing inode");
    expect_ok(vfs.close(&ordinary), "close ordinary final handle");
    expect_err(file.filesystem.load_inode(file.inode_number), FileSystemError::NotFound, "last close reclaims backing inode");
    sprintln!("[VFS TEST] namespace name protection ... PASS");
}

/// Checks independent name and ancestor contributions and the FULL combination.
///
/// Name-only protection permits ancestor renames. Ancestor-only protection
/// permits direct file movement and remains on its captured chain. Full
/// protection covers all names but only the selected alias's ancestor chain.
///
/// # Panics
///
/// Panics if a restriction is missing, overbroad, or survives successful close.
fn test_namespace_combined_protection() {
    sprintln!("[VFS TEST] namespace combined protection ...");
    let mut vfs = new_stub_vfs();
    let p = Permissions::all_permissive();
    expect_ok(vfs.create_file("/etc/protected", p, None), "create additive fixture");
    let names = expect_ok(vfs.open_with_namespace_protection("/etc/protected", AccessMode::ReadOnly,
        None, NamespaceProtection::PROTECT_NAMES), "name-only open");
    expect_ok(vfs.rename("/etc", "/settings", None), "name-only permits ancestor rename");
    let ancestors = expect_ok(vfs.open_with_namespace_protection("/settings/protected", AccessMode::ReadOnly,
        None, NamespaceProtection::PROTECT_ANCESTORS), "ancestor-only independent open");
    expect_err(vfs.rename("/settings/protected", "/var/moved", None), FileSystemError::ResourceBusy, "name holder blocks direct movement");
    expect_err(vfs.rename("/settings", "/settings2", None), FileSystemError::ResourceBusy, "ancestor holder blocks parent movement");
    expect_ok(vfs.close(&names), "release only names");
    expect_ok(vfs.rename("/settings/protected", "/var/moved", None), "ancestor-only permits direct move");
    expect_ok(vfs.rename("/var", "/storage", None), "protection does not follow moved file");
    expect_err(vfs.rename("/settings", "/settings2", None), FileSystemError::ResourceBusy, "captured ancestor remains protected");
    expect_ok(vfs.close(&ancestors), "release captured chain");
    expect_ok(vfs.rename("/settings", "/settings2", None), "ancestor release permits rename");
    expect_ok(vfs.create_hard_link("/storage/moved", "/settings2/alias", None, false), "create alias on another chain");
    assert_eq!(NamespaceProtection::PROTECT_FULL,
        NamespaceProtection::PROTECT_NAMES | NamespaceProtection::PROTECT_ANCESTORS);
    let full = expect_ok(vfs.open_with_namespace_protection("/storage/moved", AccessMode::ReadOnly,
        None, NamespaceProtection::PROTECT_FULL), "full open");
    expect_err(vfs.rename("/storage", "/storage2", None), FileSystemError::ResourceBusy, "full protects opened ancestry");
    expect_err(vfs.unlink("/settings2/alias", None), FileSystemError::ResourceBusy, "full protects other alias name");
    expect_ok(vfs.rename("/settings2", "/settings3", None), "full does not protect other alias ancestry");
    expect_ok(vfs.close(&full), "release full protection");
    expect_ok(vfs.rename("/storage", "/storage2", None), "full ancestor release");
    expect_ok(vfs.unlink("/settings3/alias", None), "full name release");
    sprintln!("[VFS TEST] namespace combined protection ... PASS");
}

/// Checks actual target ancestry through symlinks and across mounted volumes.
///
/// Lexical symlink parents must remain mutable. Mount-host and unmount checks
/// must observe the target's captured ancestry, while unrelated filesystem
/// instances remain independent even when their inode numbers coincide.
/// Failed resolution must not publish handles or leave ancestor protection.
///
/// # Panics
///
/// Panics if target selection, protection scope, failure cleanup, or unmount
/// results differ from the contract.
fn test_namespace_symlinks_and_mounts() {
    sprintln!("[VFS TEST] namespace symlinks and mounts ...");
    let mut vfs = new_stub_vfs();
    let p = Permissions::all_permissive();
    let target = expect_ok(vfs.create_file("/var/log/target", p.clone(), None), "create symlink target");
    expect_ok(vfs.create_symlink("/etc/entry", "../var/log/target", None), "relative final symlink");
    let handle = expect_ok(vfs.open_with_namespace_protection("/etc/entry", AccessMode::ReadOnly,
        None, NamespaceProtection::PROTECT_FULL), "protected open through final symlink");
    assert!(Arc::ptr_eq(&handle.file.inode, &target));
    expect_err(vfs.rename("/var/log", "/var/log2", None), FileSystemError::ResourceBusy, "actual target parent protected");
    expect_err(vfs.mount("stubfs", Arc::new(StubBlockDevice), "/var", None), FileSystemError::ResourceBusy, "actual ancestor cannot be covered");
    expect_ok(vfs.unlink("/etc/entry", None), "symlink entry itself not protected");
    expect_ok(vfs.rename("/etc", "/settings", None), "lexical parent not protected");
    expect_ok(vfs.close(&handle), "close symlink-resolved handle");
    expect_ok(vfs.rename("/var/log", "/var/log2", None), "target parent released");
    expect_ok(vfs.create_symlink("/settings/loop", "loop", None), "create loop");
    expect_err(vfs.open_with_namespace_protection("/settings/loop", AccessMode::ReadOnly,
        None, NamespaceProtection::PROTECT_FULL), FileSystemError::TooManySymlinks, "failed protected symlink open");
    expect_err(vfs.open_with_namespace_protection("/settings", AccessMode::ReadOnly,
        None, NamespaceProtection::PROTECT_FULL), FileSystemError::NotAFile, "directory open rejected");
    expect_ok(vfs.rename("/settings", "/settings2", None), "failed opens leave ancestry unprotected");

    // Relative resolution through an intermediate symlink must capture the
    // same target ancestry as the final-symlink case, not the caller's base.
    expect_ok(vfs.create_symlink("/settings2/bridge", "/var/log2", None), "intermediate directory symlink");
    let base = expect_ok(vfs.resolve_dir_with_budget("/settings2", None,
        crate::file_system::virtual_fs::resolver::MAX_SYMLINK_DEPTH), "relative open base");
    let relative = expect_ok(vfs.open_with_namespace_protection("bridge/target", AccessMode::ReadOnly,
        Some(base), NamespaceProtection::PROTECT_FULL), "relative intermediate-symlink open");
    assert!(Arc::ptr_eq(&relative.file.inode, &target));
    expect_err(vfs.rename("/var/log2", "/var/log3", None), FileSystemError::ResourceBusy, "relative target ancestry protected");
    expect_err(vfs.unlink("/var/log2/target", None), FileSystemError::ResourceBusy, "relative target name protected");
    expect_ok(vfs.close(&relative), "close relative protected open");

    expect_ok(vfs.mount("stubfs", Arc::new(StubBlockDevice), "/bin/tools", None), "mount protected-open fixture");
    let mounted = expect_ok(vfs.open_with_namespace_protection("/bin/tools/etc/config.txt", AccessMode::ReadOnly,
        None, NamespaceProtection::PROTECT_FULL), "full open inside mount");
    let host_file = expect_ok(vfs.resolve("/settings2/config.txt", None, false), "host inode-number peer");
    assert_eq!(host_file.inode_number, mounted.file.inode.inode_number);
    assert_ne!(host_file.filesystem.instance_id(), mounted.file.inode.filesystem.instance_id());
    expect_ok(vfs.unlink("/settings2/config.txt", None), "same inode number on other instance remains unprotected");
    expect_err(vfs.unmount("/bin/tools", None), FileSystemError::ResourceBusy, "full open blocks unmount");
    expect_err(vfs.rename("/bin", "/programs", None), FileSystemError::ResourceBusy, "captured ancestry crosses mount host");
    expect_ok(vfs.close(&mounted), "release mounted protection");
    expect_ok(vfs.unmount("/bin/tools", None), "unmount after protected close");
    sprintln!("[VFS TEST] namespace symlinks and mounts ... PASS");
}

/// Checks preservation of an ancestor-only token after failed final close.
///
/// Ancestor-only protection permits unlinking the file, making the last close
/// reclaim storage. A targeted driver failure must leave the handle readable,
/// its captured ancestors protected, and its contents intact for a close retry.
///
/// # Panics
///
/// Panics if failure commits any close state or successful retry fails to
/// release storage and protection. The armed failure is consumed in this test.
fn test_namespace_close_retry() {
    sprintln!("[VFS TEST] namespace close retry ...");
    let mut vfs = new_stub_vfs();
    let inode = expect_ok(vfs.create_file("/etc/retry", Permissions::all_permissive(), None), "create retry file");
    let handle = expect_ok(vfs.open_with_namespace_protection("/etc/retry", AccessMode::ReadWrite,
        None, NamespaceProtection::PROTECT_ANCESTORS), "open ancestor-only retry file");
    let clone = handle.clone();
    assert_eq!(expect_ok(vfs.write(&handle, b"keep"), "write before unlink"), 4);
    expect_ok(vfs.unlink("/etc/retry", None), "ancestor-only permits unlink");
    super::fail_next_free_for(inode.filesystem.instance_id(), inode.inode_number);
    // An unrelated immediate free must neither fail nor consume the request.
    expect_ok(vfs.create_file("/etc/other", Permissions::all_permissive(), None), "create unrelated free target");
    expect_ok(vfs.unlink("/etc/other", None), "unrelated free preserves armed failure");
    expect_err(vfs.close(&handle), FileSystemError::DiskWriteError, "inject final-close free failure");
    expect_err(vfs.rename("/etc", "/settings", None), FileSystemError::ResourceBusy, "failed close retains ancestors");
    expect_err(vfs.remove_directory("/etc", None), FileSystemError::ResourceBusy, "failed close retains removal protection");
    expect_ok(inode.filesystem.load_inode(inode.inode_number), "failed free preserves backing inode");
    expect_ok(vfs.seek(&clone, 0), "failed close leaves clone usable");
    let mut buffer = [0u8; 4];
    assert_eq!(expect_ok(vfs.read(&clone, &mut buffer), "read after failed close"), 4);
    assert_eq!(&buffer, b"keep");
    expect_ok(vfs.close(&clone), "retry close through clone");
    expect_err(vfs.read(&handle, &mut buffer), FileSystemError::InvalidHandle, "successful retry invalidates all clones");
    expect_err(inode.filesystem.load_inode(inode.inode_number), FileSystemError::NotFound, "successful retry reclaims storage");
    expect_ok(vfs.rename("/etc", "/settings", None), "successful retry releases ancestry");
    sprintln!("[VFS TEST] namespace close retry ... PASS");
}


/// Creates a regular file with known contents for replacement regressions.
///
/// The temporary writer is explicitly closed before returning the inode.
///
/// # Arguments
///
/// * `vfs` - Exclusively owned isolated fixture.
/// * `path` - Unoccupied pathname below an existing directory.
/// * `contents` - Initial bytes to write from offset zero.
///
/// # Returns
///
/// Returns the canonical inode with no registered open handle.
///
/// # Panics
///
/// Panics if creation, the complete write, or close fails.
fn replacement_file(vfs: &mut VirtualFileSystem, path: &str, contents: &[u8]) -> Arc<Inode> {
    let inode = expect_ok(vfs.create_file(path, Permissions::all_permissive(), None), "replacement create file");
    let handle = expect_ok(vfs.open(path, AccessMode::ReadWrite, None), "replacement initial writer");
    assert_eq!(expect_ok(vfs.write(&handle, contents), "replacement initial contents"), contents.len());
    expect_ok(vfs.close(&handle), "replacement close initial writer");
    inode
}

/// Verifies a pathname still resolves to the expected canonical inode.
///
/// # Arguments
///
/// * `vfs` - Isolated fixture whose namespace is inspected.
/// * `path` - Entry to resolve without following a final symlink.
/// * `inode` - Canonical identity expected at that name.
///
/// # Panics
///
/// Panics if lookup fails or the name identifies another inode.
fn replacement_identity(vfs: &VirtualFileSystem, path: &str, inode: &Arc<Inode>) {
    let found = expect_ok(vfs.resolve(path, None, false), "replacement lookup identity");
    assert!(Arc::ptr_eq(&found, inode), "[VFS TEST] replacement changed unexpected identity: {}", path);
}

/// Reads the whole expected file through an existing handle from offset zero.
///
/// # Arguments
///
/// * `vfs` - Exclusively owned fixture servicing the handle.
/// * `handle` - Registered readable handle whose offset may change.
/// * `contents` - Exact expected bytes; an extra read byte detects excess data.
///
/// # Panics
///
/// Panics if seek, read, byte count, or contents are incorrect.
fn replacement_contents(vfs: &mut VirtualFileSystem, handle: &Arc<FileHandle>, contents: &[u8]) {
    expect_ok(vfs.seek(handle, 0), "replacement seek");
    let mut bytes = alloc::vec![0u8; contents.len() + 1];
    let length = expect_ok(vfs.read(handle, &mut bytes), "replacement read");
    assert_eq!(length, contents.len());
    assert_eq!(&bytes[..length], contents);
}

/// Checks unchanged source links and displaced files with aliases and opens.
///
/// Independent old handles must keep accessing the displaced inode. New opens
/// see the source inode at the destination, and writes to the displaced file
/// must not modify it. Both replacement and later unlink respect final close.
///
/// # Panics
///
/// Panics on unexpected identity, content, count, or storage lifetime.
fn test_replacement_file_lifetimes() {
    sprintln!("[VFS TEST] replacement file lifetimes ...");
    let mut vfs = new_stub_vfs();
    let source = replacement_file(&mut vfs, "/etc/source", b"SOURCE");
    let victim = replacement_file(&mut vfs, "/var/log/destination", b"VICTIM");
    expect_ok(vfs.create_hard_link("/etc/source", "/bin/source-alias", None, false), "source alias");
    expect_ok(vfs.create_hard_link("/var/log/destination", "/etc/victim-alias", None, false), "victim alias");
    let source_handle = expect_ok(vfs.open("/etc/source", AccessMode::ReadOnly, None), "open rename source");
    let first = expect_ok(vfs.open("/var/log/destination", AccessMode::ReadWrite, None), "first victim open");
    let second = expect_ok(vfs.open("/etc/victim-alias", AccessMode::ReadOnly, None), "second victim open");
    expect_ok(vfs.rename("/etc/source", "/var/log/destination", None), "cross-parent replacement");
    expect_err(vfs.resolve("/etc/source", None, false), FileSystemError::NotFound, "source name removed");
    replacement_identity(&vfs, "/var/log/destination", &source);
    replacement_identity(&vfs, "/bin/source-alias", &source);
    replacement_identity(&vfs, "/etc/victim-alias", &victim);
    assert_eq!(source.stat().link_count, 2);
    assert_eq!(victim.stat().link_count, 1);
    assert_eq!(expect_ok(victim.filesystem.load_inode(victim.inode_number), "backing victim count").stat().link_count, 1);
    expect_ok(vfs.seek(&first, 0), "seek displaced writer");
    assert_eq!(expect_ok(vfs.write(&first, b"CHANGED"), "write displaced inode"), 7);
    replacement_contents(&mut vfs, &second, b"CHANGED");
    replacement_contents(&mut vfs, &source_handle, b"SOURCE");
    let fresh = expect_ok(vfs.open("/var/log/destination", AccessMode::ReadOnly, None), "fresh destination open");
    assert!(Arc::ptr_eq(&fresh.file.inode, &source));
    replacement_contents(&mut vfs, &fresh, b"SOURCE");
    expect_ok(vfs.unlink("/etc/victim-alias", None), "remove displaced final alias");
    expect_ok(vfs.close(&first), "first victim close");
    expect_ok(vfs.reclaim_unlinked_inodes(), "cleanup skips registered displaced inode");
    expect_ok(victim.filesystem.load_inode(victim.inode_number), "second open preserves storage");
    expect_ok(vfs.close(&second), "last displaced close");
    expect_err(victim.filesystem.load_inode(victim.inode_number), FileSystemError::NotFound, "final close frees displaced inode");
    expect_ok(vfs.close(&fresh), "close fresh destination");
    expect_ok(vfs.close(&source_handle), "close moved source");

    // With no surviving alias, rename itself must mark the old open master
    // pending deletion. No later unlink is available to perform that step.
    let incoming = replacement_file(&mut vfs, "/etc/incoming", b"new");
    let old = replacement_file(&mut vfs, "/etc/old", b"old");
    let old_handle = expect_ok(vfs.open("/etc/old", AccessMode::ReadOnly, None), "open last-name victim");
    expect_ok(vfs.rename("/etc/incoming", "/etc/old", None), "replace last-name open victim");
    replacement_identity(&vfs, "/etc/old", &incoming);
    assert_eq!(old.stat().link_count, 0);
    replacement_contents(&mut vfs, &old_handle, b"old");
    expect_ok(vfs.reclaim_unlinked_inodes(), "cleanup skips pending master");
    super::fail_next_free_for(old.filesystem.instance_id(), old.inode_number);
    expect_err(vfs.close(&old_handle), FileSystemError::DiskWriteError, "replacement final-close free failure");
    replacement_identity(&vfs, "/etc/old", &incoming);
    replacement_contents(&mut vfs, &old_handle, b"old");
    expect_ok(vfs.close(&old_handle), "replacement pending-deletion final close retry");
    expect_err(old.filesystem.load_inode(old.inode_number), FileSystemError::NotFound, "rename marked final-close reclamation");
    sprintln!("[VFS TEST] replacement file lifetimes ... PASS");
}

/// Checks replacement of empty directories while old wrappers remain live.
///
/// Both a moved source wrapper and an overwritten destination wrapper must
/// observe their correct parent. The destination wrapper predates an earlier
/// move, exercising replacement's actual former-parent record rather than
/// accidentally relying on its original stored parent.
///
/// # Panics
///
/// Panics on incorrect traversal, detached mutation, or premature reclamation.
fn test_replacement_directory_lifetimes() {
    sprintln!("[VFS TEST] replacement directory lifetimes ...");
    let mut vfs = new_stub_vfs();
    let p = Permissions::all_permissive();
    let source = expect_ok(vfs.create_directory("/etc/incoming", p.clone(), None), "incoming directory");
    replacement_file(&mut vfs, "/etc/incoming/child", b"child");
    let victim = expect_ok(vfs.create_directory("/etc/old-location", p.clone(), None), "destination directory");
    let budget = crate::file_system::virtual_fs::resolver::MAX_SYMLINK_DEPTH;
    let old_wrapper = expect_ok(vfs.resolve_dir_with_budget("/etc/old-location", None, budget), "retain old destination");
    let source_wrapper = expect_ok(vfs.resolve_dir_with_budget("/etc/incoming", None, budget), "retain source directory");
    expect_ok(vfs.rename("/etc/old-location", "/var/log/target", None), "move destination before replacement");
    expect_ok(vfs.rename("/etc/incoming///", "/var/log/target///", None), "replace retained empty directory");
    replacement_identity(&vfs, "/var/log/target", &source);
    expect_ok(vfs.resolve("/var/log/target/child", None, false), "incoming directory contents survive");
    assert_eq!(source.stat().link_count, 1);
    assert_eq!(victim.stat().link_count, 0);
    let parent = expect_ok(vfs.resolve("/var/log", None, false), "replacement parent");
    for wrapper in [&old_wrapper, &source_wrapper] {
        let actual = expect_ok(vfs.resolve("..", Some(Arc::clone(wrapper)), false), "retained wrapper parent");
        assert!(Arc::ptr_eq(&actual, &parent));
    }
    assert!(expect_ok(victim.filesystem.read_dir(&victim), "detached directory listing").is_empty());
    expect_err(vfs.create_file("new", p.clone(), Some(old_wrapper.clone())), FileSystemError::NotFound, "no creation in displaced directory");
    expect_err(vfs.protect_directory_path(old_wrapper.clone()), FileSystemError::NotFound, "no guard on displaced directory");
    expect_ok(vfs.reclaim_unlinked_inodes(), "retained directory cleanup");
    expect_ok(victim.filesystem.load_inode(victim.inode_number), "wrapper preserves displaced directory storage");
    drop(old_wrapper);
    expect_ok(vfs.reclaim_unlinked_inodes(), "final displaced wrapper cleanup");
    expect_err(victim.filesystem.load_inode(victim.inode_number), FileSystemError::NotFound, "displaced directory reclaimed");
    expect_ok(vfs.create_directory("/etc/a", p.clone(), None), "same-parent source directory");
    expect_ok(vfs.create_directory("/etc/b", p, None), "same-parent destination directory");
    expect_ok(vfs.rename("/etc/a", "/etc/b", None), "same-parent directory replacement");
    expect_ok(vfs.reclaim_unlinked_inodes(), "cleanup unopened directory replacement");
    drop(source_wrapper);
    sprintln!("[VFS TEST] replacement directory lifetimes ... PASS");
}

/// Exercises type, path, subtree, mount, and optional protection rejection.
///
/// Rejections preserve source/destination identities. Same-inode aliases are
/// still successful no-ops, and failed special spellings never normalize into
/// a different directory entry and mutate it.
///
/// # Panics
///
/// Panics if an operation returns the wrong error or changes rejected names.
fn test_replacement_rejections() {
    sprintln!("[VFS TEST] replacement rejections ...");
    let mut vfs = new_stub_vfs();
    let p = Permissions::all_permissive();
    let file = replacement_file(&mut vfs, "/etc/file", b"file");
    for path in ["/etc/tree", "/etc/tree/sub", "/etc/empty", "/etc/nonempty"] {
        expect_ok(vfs.create_directory(path, p.clone(), None), "rejection directory");
    }
    replacement_file(&mut vfs, "/etc/nonempty/child", b"keep");
    expect_ok(vfs.create_symlink("/etc/link", "/missing", None), "dangling rejection link");
    for (old, new, error) in [
        ("/etc/file", "/etc/empty", FileSystemError::IsADirectory),
        ("/etc/link", "/etc/empty", FileSystemError::IsADirectory),
        ("/etc/empty", "/etc/file", FileSystemError::NotADirectory),
        ("/etc/empty", "/etc/link", FileSystemError::NotADirectory),
        ("/etc/empty", "/etc/nonempty", FileSystemError::DirectoryNotEmpty),
        ("/etc/tree", "/etc/tree/sub/new", FileSystemError::InvalidPath),
        ("/etc/tree", "/etc/tree/new", FileSystemError::InvalidPath),
        ("/etc/file/", "/etc/new", FileSystemError::NotADirectory),
        ("/etc/link/", "/etc/new", FileSystemError::NotADirectory),
        ("/etc/file", "/etc/new/", FileSystemError::NotADirectory),
        ("/etc/link", "/etc/new/", FileSystemError::NotADirectory),
        ("/etc/file", "/etc/link/", FileSystemError::NotADirectory),
        ("/etc/empty", "/etc/absent/", FileSystemError::NotFound),
    ] {
        let before = expect_ok(vfs.resolve(old.trim_end_matches('/'), None, false), "rejection source before");
        let destination_before = vfs.resolve(new.trim_end_matches('/'), None, false).ok();
        expect_err(vfs.rename(old, new, None), error, "replacement preflight rejection");
        replacement_identity(&vfs, old.trim_end_matches('/'), &before);
        if let Some(inode) = destination_before {
            replacement_identity(&vfs, new.trim_end_matches('/'), &inode);
        } else {
            expect_err(vfs.resolve(new.trim_end_matches('/'), None, false), FileSystemError::NotFound, "rejected destination remains absent");
        }
    }
    for path in ["", "/", "////", ".", "..", "/etc/empty/.", "/etc/empty/..///"] {
        let error = if path.is_empty() { FileSystemError::NotFound }
            else if path.trim_matches('/').is_empty() { FileSystemError::ResourceBusy }
            else { FileSystemError::OperationNotAllowed };
        expect_err(vfs.rename(path, "/etc/new", None), error.clone(), "special source rejected");
        expect_err(vfs.rename("/etc/file", path, None), error, "special destination rejected");
        replacement_identity(&vfs, "/etc/file", &file);
    }
    expect_err(vfs.rename("/etc/file\0", "/etc/new", None), FileSystemError::InvalidPath, "NUL source rejected");
    expect_err(vfs.rename("/etc/file", "/etc/new\0", None), FileSystemError::InvalidPath, "NUL destination rejected");
    let protected = expect_ok(vfs.open_with_namespace_protection("/etc/file", AccessMode::ReadOnly,
        None, NamespaceProtection::PROTECT_NAMES), "protected replacement source");
    let other = replacement_file(&mut vfs, "/etc/other", b"other");
    expect_err(vfs.rename("/etc/file", "/etc/other", None), FileSystemError::ResourceBusy, "source protection");
    expect_err(vfs.rename("/etc/other", "/etc/file", None), FileSystemError::ResourceBusy, "destination protection");
    replacement_identity(&vfs, "/etc/other", &other);
    replacement_identity(&vfs, "/etc/file", &file);
    expect_ok(vfs.create_hard_link("/etc/file", "/etc/alias", None, false), "protected no-op alias");
    expect_ok(vfs.rename("/etc/file", "/etc/alias", None), "protected same-inode no-op");
    assert_eq!(file.stat().link_count, 2);
    expect_ok(vfs.close(&protected), "release file protection");
    let wrapper = expect_ok(vfs.resolve_dir_with_budget("/etc/empty", None,
        crate::file_system::virtual_fs::resolver::MAX_SYMLINK_DEPTH), "protected directory wrapper");
    let guard = expect_ok(vfs.protect_directory_path(wrapper), "protected directory destination");
    expect_err(vfs.rename("/etc/tree", "/etc/empty", None), FileSystemError::ResourceBusy, "directory destination protection");
    expect_err(vfs.rename("/etc/empty", "/etc/new", None), FileSystemError::ResourceBusy, "directory source protection");
    drop(guard);
    expect_ok(vfs.mount("stubfs", Arc::new(StubBlockDevice), "/bin/tools", None), "mount rename boundary");
    expect_err(vfs.rename("/etc/file", "/bin/tools/etc/config.txt", None), FileSystemError::CrossFileSystemOperation, "cross-instance replacement");
    expect_err(vfs.rename("/bin/tools", "/bin/new", None), FileSystemError::ResourceBusy, "mount source rejection");
    expect_err(vfs.rename("/etc/empty", "/bin/tools", None), FileSystemError::ResourceBusy, "mount destination rejection");
    expect_ok(vfs.unmount("/bin/tools", None), "unmount rejected replacement fixture");

    // A relative destination may start from a wrapper retained before a move.
    // The subtree walk must consult relocated parents or it could create a
    // cycle by moving the new ancestor into that retained descendant.
    for path in ["/etc/retained", "/etc/retained/leaf", "/var/log/container"] {
        expect_ok(vfs.create_directory(path, p.clone(), None), "relocated ancestry fixture");
    }
    let leaf = expect_ok(vfs.resolve_dir_with_budget("/etc/retained/leaf", None,
        crate::file_system::virtual_fs::resolver::MAX_SYMLINK_DEPTH), "retain descendant before move");
    expect_ok(vfs.rename("/etc/retained", "/var/log/container/retained", None), "move retained subtree");
    expect_err(vfs.rename("/var/log/container", "new", Some(leaf)), FileSystemError::InvalidPath, "relocated ancestry rejects cycle");
    expect_ok(vfs.resolve("/var/log/container/retained/leaf", None, false), "rejected cycle preserves namespace");
    sprintln!("[VFS TEST] replacement rejections ... PASS");
}

/// Injects failure after driver staging for same-parent and cross-parent cases.
///
/// Both listings, link counts, contents, and retained-directory parents must
/// survive the returned error. A retry must perform the same replacement and
/// must not leave duplicate entries or change the source's link count.
///
/// # Panics
///
/// Panics on a mutation before successful commit or incorrect retry behavior.
fn test_replacement_failure_atomicity() {
    sprintln!("[VFS TEST] replacement failure atomicity ...");
    for (directory, cross_parent) in [(false, false), (false, true), (true, false), (true, true)] {
        let mut vfs = new_stub_vfs();
        let destination = if cross_parent { "/var/log/destination" } else { "/etc/destination" };
        let (source, victim) = if directory {
            let p = Permissions::all_permissive();
            (expect_ok(vfs.create_directory("/etc/source", p.clone(), None), "atomic source directory"),
             expect_ok(vfs.create_directory(destination, p, None), "atomic destination directory"))
        } else {
            (replacement_file(&mut vfs, "/etc/source", b"source"),
             replacement_file(&mut vfs, destination, b"victim"))
        };
        let wrapper = if directory {
            Some(expect_ok(vfs.resolve_dir_with_budget("/etc/source", None,
                crate::file_system::virtual_fs::resolver::MAX_SYMLINK_DEPTH), "atomic retained source"))
        } else { None };
        super::fail_next_rename_for(source.filesystem.instance_id());
        expect_err(vfs.rename("/etc/source", destination, None), FileSystemError::DiskWriteError, "staged driver failure");
        replacement_identity(&vfs, "/etc/source", &source);
        replacement_identity(&vfs, destination, &victim);
        for inode in [&source, &victim] {
            assert_eq!(inode.stat().link_count, 1);
            assert_eq!(expect_ok(inode.filesystem.load_inode(inode.inode_number), "atomic backing count").stat().link_count, 1);
        }
        assert_eq!(expect_ok(vfs.reclaim_unlinked_inodes(), "no premature orphan after rename error"), 0);
        if let Some(wrapper) = wrapper.as_ref() {
            let parent = expect_ok(vfs.resolve("..", Some(wrapper.clone()), false), "source parent after failed move");
            let etc = expect_ok(vfs.resolve("/etc", None, false), "old parent identity");
            assert!(Arc::ptr_eq(&parent, &etc));
        } else {
            for (path, contents) in [("/etc/source", &b"source"[..]), (destination, &b"victim"[..])] {
                let handle = expect_ok(vfs.open(path, AccessMode::ReadOnly, None), "atomic content open");
                replacement_contents(&mut vfs, &handle, contents);
                expect_ok(vfs.close(&handle), "atomic content close");
            }
        }
        expect_ok(vfs.rename("/etc/source", destination, None), "retry staged replacement");
        replacement_identity(&vfs, destination, &source);
        let (parent_path, name) = if cross_parent { ("/var/log", "destination") } else { ("/etc", "destination") };
        let parent_inode = expect_ok(vfs.resolve(parent_path, None, false), "retry destination parent");
        let listing = expect_ok(parent_inode.filesystem.read_dir(&parent_inode), "retry backing listing");
        assert_eq!(listing.iter().filter(|entry| entry.name == name).count(), 1);
        expect_err(vfs.resolve("/etc/source", None, false), FileSystemError::NotFound, "source absent after retry");
        assert_eq!(source.stat().link_count, 1);
        assert_eq!(victim.stat().link_count, 0);
        expect_ok(vfs.reclaim_unlinked_inodes(), "reclaim displaced inode after retry");
        expect_err(victim.filesystem.load_inode(victim.inode_number), FileSystemError::NotFound, "retry victim reclaimed");
        drop(wrapper);
    }
    sprintln!("[VFS TEST] replacement failure atomicity ... PASS");
}

/// Checks compatible symlink replacement and retryable post-commit cleanup.
///
/// Unopened displaced files and symlinks keep backing storage until explicit
/// cleanup. Injected free failures must not affect the committed namespace.
/// Unmount also retries cleanup before detaching its instance.
///
/// # Panics
///
/// Panics if rename follows a final symlink, reclamation changes a committed
/// name, or successful cleanup/unmount fails to release displaced storage.
fn test_replacement_reclamation() {
    sprintln!("[VFS TEST] replacement reclamation ...");
    for (source_link, victim_link) in [(false, false), (true, false), (false, true), (true, true)] {
        let mut vfs = new_stub_vfs();
        let source = if source_link {
            expect_ok(vfs.create_symlink("/etc/source", "/etc/config.txt", None), "replacement source symlink")
        } else { replacement_file(&mut vfs, "/etc/source", b"source") };
        let victim = if victim_link {
            expect_ok(vfs.create_symlink("/etc/destination", "/dangling", None), "replacement dangling destination")
        } else { replacement_file(&mut vfs, "/etc/destination", b"victim") };
        let unaffected = expect_ok(vfs.resolve("/etc/config.txt", None, false), "symlink target before replacement");
        super::fail_next_free_for(victim.filesystem.instance_id(), victim.inode_number);
        expect_ok(vfs.rename("/etc/source", "/etc/destination", None), "rename succeeds without freeing victim");
        replacement_identity(&vfs, "/etc/destination", &source);
        assert_eq!(victim.stat().link_count, 0);
        expect_ok(victim.filesystem.load_inode(victim.inode_number), "unopened displaced storage retained");
        expect_err(vfs.reclaim_unlinked_inodes(), FileSystemError::DiskWriteError, "post-commit free error");
        replacement_identity(&vfs, "/etc/destination", &source);
        replacement_identity(&vfs, "/etc/config.txt", &unaffected);
        expect_ok(victim.filesystem.load_inode(victim.inode_number), "failed free stays retryable");
        assert_eq!(expect_ok(vfs.reclaim_unlinked_inodes(), "post-commit free retry"), 1);
        expect_err(victim.filesystem.load_inode(victim.inode_number), FileSystemError::NotFound, "unopened victim reclaimed");
        if source_link {
            assert_eq!(expect_ok(source.filesystem.read_symlink(&source), "moved symlink contents"), "/etc/config.txt");
        }
    }
    let mut vfs = new_stub_vfs();
    expect_ok(vfs.mount("stubfs", Arc::new(StubBlockDevice), "/bin/tools", None), "mount cleanup fixture");
    let source = replacement_file(&mut vfs, "/bin/tools/etc/source", b"new");
    let victim = replacement_file(&mut vfs, "/bin/tools/etc/destination", b"old");
    expect_ok(vfs.rename("/bin/tools/etc/source", "/bin/tools/etc/destination", None), "mounted replacement");
    super::fail_next_free_for(victim.filesystem.instance_id(), victim.inode_number);
    expect_err(vfs.unmount("/bin/tools", None), FileSystemError::DiskWriteError, "unmount preserves mount on cleanup error");
    replacement_identity(&vfs, "/bin/tools/etc/destination", &source);
    expect_ok(vfs.unmount("/bin/tools", None), "retry unmount reclaims displaced inode");
    expect_err(victim.filesystem.load_inode(victim.inode_number), FileSystemError::NotFound, "unmount freed displaced storage");
    sprintln!("[VFS TEST] replacement reclamation ... PASS");
}

/// Checks both create-and-open modes on new and existing regular files.
///
/// A second ordinary open must retain the created file's contents and
/// canonical identity. Exclusive creation rejects every occupied final name
/// without changing the existing file or directory.
///
/// # Panics
///
/// Panics if registration, content preservation, identity, or exclusivity
/// differs from the stated contract.
fn test_open_or_create_basic() {
    sprintln!("[VFS TEST] create-and-open basic ...");
    let mut vfs = new_stub_vfs();
    let path = "/etc/opened-on-create";
    let created = expect_ok(vfs.open_or_create(
        path, AccessMode::ReadWrite, CreateMode::IfMissing,
        Permissions::all_permissive(), None,
    ), "create and open missing file");
    assert_eq!(expect_ok(vfs.write(&created, b"keep"), "write created file"), 4);
    expect_ok(vfs.close(&created), "close created file");

    let reopened = expect_ok(vfs.open_or_create(
        path, AccessMode::ReadOnly, CreateMode::IfMissing,
        Permissions::all_permissive(), None,
    ), "ordinary create-and-open existing file");
    assert!(Arc::ptr_eq(&created.file.inode, &reopened.file.inode));
    let mut contents = [0u8; 4];
    assert_eq!(expect_ok(vfs.read(&reopened, &mut contents),
        "read existing contents"), 4);
    assert_eq!(&contents, b"keep");
    expect_err(vfs.open_or_create(
        path, AccessMode::ReadWrite, CreateMode::Exclusive,
        Permissions::all_permissive(), None,
    ), FileSystemError::AlreadyExists, "exclusive existing file");
    expect_err(vfs.open_or_create(
        "/etc", AccessMode::ReadWrite, CreateMode::Exclusive,
        Permissions::all_permissive(), None,
    ), FileSystemError::AlreadyExists, "exclusive existing directory");
    expect_ok(vfs.close(&reopened), "close reopened file");

    let exclusive = expect_ok(vfs.open_or_create(
        "/etc/exclusive-new", AccessMode::ReadWrite,
        CreateMode::Exclusive, Permissions::all_permissive(), None,
    ), "exclusive create missing file");
    let ordinary = expect_ok(vfs.open("/etc/exclusive-new",
        AccessMode::ReadOnly, None), "ordinary open of exclusive creation");
    assert!(Arc::ptr_eq(&exclusive.file.inode, &ordinary.file.inode));
    expect_ok(vfs.close(&ordinary), "close ordinary open");
    expect_ok(vfs.close(&exclusive), "close exclusive creation");
    sprintln!("[VFS TEST] create-and-open basic ... PASS");
}

/// Checks final-symlink creation and one shared traversal budget.
///
/// Relative and absolute dangling targets are created in their resolved
/// directories. A parent symlink and sixteen final symlinks together exceed
/// the same budget that permits the sixteen final links alone.
///
/// # Panics
///
/// Panics if symlink creation targets, exclusive rejection, loop handling,
/// parent resolution, or budget accounting differ from the contract.
fn test_open_or_create_symlinks() {
    sprintln!("[VFS TEST] create-and-open symlinks ...");
    let mut vfs = new_stub_vfs();
    expect_ok(vfs.create_symlink("/etc/absolute-link", "/var/created-absolute", None),
        "absolute dangling symlink");
    expect_err(vfs.open_or_create(
        "/etc/absolute-link", AccessMode::ReadWrite,
        CreateMode::Exclusive, Permissions::all_permissive(), None,
    ), FileSystemError::AlreadyExists, "exclusive dangling symlink");
    expect_err(vfs.stat("/var/created-absolute", None, false),
        FileSystemError::NotFound, "exclusive did not create target");
    let absolute = expect_ok(vfs.open_or_create(
        "/etc/absolute-link", AccessMode::ReadWrite,
        CreateMode::IfMissing, Permissions::all_permissive(), None,
    ), "create through absolute symlink");
    let target = expect_ok(vfs.resolve("/var/created-absolute", None, false),
        "resolve created absolute target");
    assert!(Arc::ptr_eq(&absolute.file.inode, &target));
    expect_ok(vfs.close(&absolute), "close absolute target");
    let live = expect_ok(vfs.open_or_create(
        "/etc/absolute-link", AccessMode::ReadOnly,
        CreateMode::IfMissing, Permissions::all_permissive(), None,
    ), "ordinary creation opens existing symlink target");
    assert!(Arc::ptr_eq(&live.file.inode, &target));
    expect_ok(vfs.close(&live), "close live symlink target");

    expect_ok(vfs.create_symlink("/etc/relative-link", "../var/created-relative", None),
        "relative dangling symlink");
    expect_ok(vfs.create_symlink("/etc/chain-link", "relative-link", None),
        "chained symlink");
    let relative = expect_ok(vfs.open_or_create(
        "/etc/chain-link", AccessMode::ReadWrite,
        CreateMode::IfMissing, Permissions::all_permissive(), None,
    ), "create through chained relative symlinks");
    assert!(Arc::ptr_eq(&relative.file.inode,
        &expect_ok(vfs.resolve("/var/created-relative", None, false),
            "resolve created relative target")));
    expect_ok(vfs.close(&relative), "close relative target");
    expect_err(vfs.open_or_create(
        "/etc/relative-link", AccessMode::ReadOnly,
        CreateMode::Exclusive, Permissions::all_permissive(), None,
    ), FileSystemError::AlreadyExists, "exclusive live symlink");

    expect_err(vfs.open_or_create(
        "/missing-parent/file", AccessMode::ReadWrite,
        CreateMode::IfMissing, Permissions::all_permissive(), None,
    ), FileSystemError::NotFound, "missing parent remains absent");
    expect_err(vfs.open_or_create(
        "/etc/config.txt/child", AccessMode::ReadWrite,
        CreateMode::IfMissing, Permissions::all_permissive(), None,
    ), FileSystemError::NotADirectory, "non-directory parent rejected");
    expect_err(vfs.open_or_create(
        "/etc/trailing/", AccessMode::ReadWrite,
        CreateMode::IfMissing, Permissions::all_permissive(), None,
    ), FileSystemError::NotADirectory, "trailing slash rejects file creation");

    for index in 0..16 {
        let link = format!("/etc/budget-{}", index);
        let next = if index == 15 {
            "/etc/budget-file".into()
        } else {
            format!("/etc/budget-{}", index + 1)
        };
        expect_ok(vfs.create_symlink(&link, &next, None),
            "create symlink budget chain");
    }
    let budget_file = expect_ok(vfs.open_or_create(
        "/etc/budget-0", AccessMode::ReadWrite,
        CreateMode::IfMissing, Permissions::all_permissive(), None,
    ), "sixteen final symlinks fit budget");
    expect_ok(vfs.close(&budget_file), "close budget chain target");
    expect_ok(vfs.create_symlink("/etc/parent-alias", "/etc", None),
        "symlink in parent path");
    expect_err(vfs.open_or_create(
        "/etc/parent-alias/budget-0", AccessMode::ReadWrite,
        CreateMode::IfMissing, Permissions::all_permissive(), None,
    ), FileSystemError::TooManySymlinks,
        "parent and final symlinks share budget");
    expect_ok(vfs.create_symlink("/etc/loop-a", "/etc/loop-b", None),
        "first loop link");
    expect_ok(vfs.create_symlink("/etc/loop-b", "/etc/loop-a", None),
        "second loop link");
    expect_err(vfs.open_or_create(
        "/etc/loop-a", AccessMode::ReadWrite,
        CreateMode::IfMissing, Permissions::all_permissive(), None,
    ), FileSystemError::TooManySymlinks, "symlink loop is bounded");

    expect_ok(vfs.mount("stubfs", Arc::new(StubBlockDevice),
        "/bin/tools", None), "mount create-and-open target");
    expect_ok(vfs.create_symlink("/etc/mounted-link",
        "/bin/tools/etc/mounted-created", None),
        "link to mounted destination");
    let mounted = expect_ok(vfs.open_or_create(
        "/etc/mounted-link", AccessMode::ReadWrite,
        CreateMode::IfMissing, Permissions::all_permissive(), None,
    ), "create through link into mounted instance");
    let mounted_target = expect_ok(vfs.resolve(
        "/bin/tools/etc/mounted-created", None, false,
    ), "resolve mounted created target");
    assert!(Arc::ptr_eq(&mounted.file.inode, &mounted_target));
    expect_ok(vfs.close(&mounted), "close mounted target");
    expect_ok(vfs.unmount("/bin/tools", None), "unmount create-and-open target");
    sprintln!("[VFS TEST] create-and-open symlinks ... PASS");
}

/// Checks pre-commit creation errors and successful retry.
///
/// A staged driver failure leaves both the directory entry and the next
/// inode slot untouched, allowing a retry to use that same slot.
///
/// # Panics
///
/// Panics if a failed create publishes a name or inode, advances the slot,
/// or prevents a successful retry.
fn test_open_or_create_failures() {
    sprintln!("[VFS TEST] create-and-open failures ...");
    let mut vfs = new_stub_vfs();
    let baseline = expect_ok(vfs.create_file(
        "/etc/create-baseline", Permissions::all_permissive(), None,
    ), "create failure baseline");
    let next_inode_number = baseline.inode_number + 1;
    super::fail_next_create_for(baseline.filesystem.instance_id());
    expect_err(vfs.open_or_create(
        "/etc/create-retry", AccessMode::ReadWrite,
        CreateMode::IfMissing, Permissions::all_permissive(), None,
    ), FileSystemError::DiskWriteError, "staged creation failure");
    expect_err(vfs.resolve("/etc/create-retry", None, false),
        FileSystemError::NotFound, "failed create left name absent");
    expect_err(baseline.filesystem.load_inode(next_inode_number),
        FileSystemError::NotFound, "failed create left backing inode absent");
    let retry = expect_ok(vfs.open_or_create(
        "/etc/create-retry", AccessMode::ReadWrite,
        CreateMode::IfMissing, Permissions::all_permissive(), None,
    ), "retry creation");
    assert_eq!(retry.file.inode.inode_number, next_inode_number);
    expect_ok(vfs.close(&retry), "close retry creation");

    sprintln!("[VFS TEST] create-and-open failures ... PASS");
}
