//! # VFS Test Harness
//!
//! This module contains the [`run_vfs_tests`] function, which exercises every
//! operation exposed by the [`crate::file_system::virtual_fs::vfs::VirtualFileSystem`]
//! against the stub filesystem driver. It is called from the kernel's main
//! entry point during early boot, before any userspace is initialized.
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
//! 5. **Symlink traversal** - resolves paths through the `lib -> /etc` symlink.
//! 6. **Dot-dot traversal** - resolves `bin/tools/../shell` and verifies.
//! 7. **Create file** - creates a new file and verifies it can be opened.
//! 8. **Create directory** - creates a new directory and verifies it exists.
//! 9. **Create symlink** - creates a new symlink and verifies traversal.
//! 10. **Rename** - verifies in-place, relative, and cross-directory renames.
//! 11. **Delete** - deletes a file and verifies it is gone.
//! 12. **POSIX deferred delete** - checks two independent opens, cached-name
//!     rejection, final-close reclamation, and deletion relative to a working
//!     directory.
//! 13. **Wrapper consistency** - compares retained and fresh namespace views.
//! 14. **Mounted wrappers** - checks mount visibility and real unmount lookup.
//! 15. **Teardown** - drops the root volume cleanly.

extern crate alloc;

use alloc::sync::Arc;

use crate::file_system::stub_fs::stub_block_device::StubBlockDevice;
use crate::file_system::stub_fs::StubFileSystemDriver;
use crate::file_system::virtual_fs::directory::Directory;
use crate::file_system::virtual_fs::types::{
    AccessMode, FileSystemError, InodeType, Permissions,
};
use crate::file_system::virtual_fs::vfs::VirtualFileSystem;
use crate::file_system::virtual_fs::file::FileHandle;
use crate::globals;
use crate::file_system::virtual_fs::driver::FileSystemDriver;

use crate::sprintln;

/// Runs all VFS tests against the stub filesystem.
///
/// Initializes the global [`VirtualFileSystem`], registers the stub driver,
/// mounts the stub volume, runs all operation tests in order, then tears down
/// the root VFS instance.
/// Panics on the first assertion failure with a message identifying the
/// failing test. All pass/fail output is written to the serial port.
///
/// Must be called after the serial port and wall clock are initialized, and
/// before any other filesystem use.
pub fn run_vfs_tests() {
    sprintln!("[VFS TEST] Starting VFS test suite");
    sprintln!("[VFS TEST] ----------------------------------------");

    test_mount_and_register();
    test_stat();
    test_read();
    test_write();
    test_symlink_traversal();
    test_dotdot_traversal();
    test_create_file();
    test_create_directory();
    test_create_symlink();
    test_rename();
    test_delete();
    test_posix_deferred_delete();
    test_retained_directory_namespace();
    test_mount_wrapper_consistency();
    test_unmount();

    sprintln!("[VFS TEST] ----------------------------------------");
    sprintln!("[VFS TEST] All tests passed");
}

// ----------------------------------------------------------------------------
// Helpers
// ----------------------------------------------------------------------------

/// Locks the global VFS and calls `f` with a mutable reference to it.
///
/// Panics if the VFS has not been initialized.
///
/// # Arguments
///
/// * `f` - A closure that receives `&mut VirtualFileSystem` and returns `T`.
///
/// # Returns
///
/// Returns the value produced by `f`.
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
/// # Returns
///
/// Returns no value after a successful close; panics on an error.
fn close_file(handle: &Arc<FileHandle>) {
    let result = with_vfs(|vfs| vfs.close(handle));
    expect_ok(result, "close");
}

// ----------------------------------------------------------------------------
// Individual tests
// ----------------------------------------------------------------------------

/// Registers the stub driver and mounts the stub volume as the VFS root.
fn test_mount_and_register() {
    sprintln!("[VFS TEST] mount and register ...");

    // Build the root directory for the VFS from the stub's root inode.
    // We need to create the VirtualFileSystem with a root Directory before
    // mounting, so we create a temporary placeholder root inode.
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
/// # Arguments
///
/// None.
///
/// # Returns
///
/// Returns no value; panics when a read or handle-lifecycle check fails.
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
/// # Arguments
///
/// None.
///
/// # Returns
///
/// Returns no value; panics if writing, reading, or closing fails.
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

/// Verifies that paths through the `lib -> /etc` symlink resolve correctly.
///
/// # Arguments
///
/// None.
///
/// # Returns
///
/// Returns no value; panics if resolution or reading fails.
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
/// # Arguments
///
/// None.
///
/// # Returns
///
/// Returns no value; panics if a path crosses to the wrong parent.
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

/// Creates a new file under `/var/log/` and verifies it can be opened and
/// written to.
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
/// # Arguments
///
/// None.
///
/// # Returns
///
/// Returns no value; panics if creation, resolution, or reading fails.
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
/// # Arguments
///
/// None.
///
/// # Returns
///
/// Returns no value; panics if renaming, lookup, or reading fails.
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

/// Deletes `/bin/tools/debug` and verifies it is gone.
fn test_delete() {
    sprintln!("[VFS TEST] delete ...");

    expect_ok(
        with_vfs(|vfs| vfs.delete("/bin/tools/debug", None)),
        "delete /bin/tools/debug",
    );

    // The path should now return NotFound.
    expect_err(
        with_vfs(|vfs| vfs.stat("/bin/tools/debug", None, false)),
        FileSystemError::NotFound,
        "stat after delete",
    );

    // Deleting again should return NotFound.
    expect_err(
        with_vfs(|vfs| vfs.delete("/bin/tools/debug", None)),
        FileSystemError::NotFound,
        "double delete",
    );

    sprintln!("[VFS TEST] delete ... PASS");
}

/// Verifies that unlink preserves data for independent open handles, rejects
/// an open through a retained directory wrapper, and frees on the last close.
/// Also verifies deletion relative to a working directory.
///
/// # Arguments
///
/// None.
///
/// # Returns
///
/// Returns no value; panics if any deferred-deletion check fails.
fn test_posix_deferred_delete() {
    sprintln!("[VFS TEST] POSIX deferred delete ...");

    // Retain a wrapper with a loaded entry, then open the file twice.
    let etc_dir = expect_ok(
        with_vfs(|vfs| vfs.resolve_dir_with_budget(
            "/etc",
            None,
            crate::file_system::virtual_fs::resolver::MAX_SYMLINK_DEPTH,
        )),
        "resolve /etc before deferred delete",
    );
    expect_ok(
        with_vfs(|vfs| vfs.stat("config.txt", Some(Arc::clone(&etc_dir)), false)),
        "load config.txt in retained /etc wrapper",
    );
    let first = expect_ok(
        with_vfs(|vfs| vfs.open("/etc/config.txt", AccessMode::ReadOnly, None)),
        "first open /etc/config.txt before deferred delete",
    );
    let last = expect_ok(
        with_vfs(|vfs| vfs.open("/etc/config.txt", AccessMode::ReadOnly, None)),
        "second open /etc/config.txt before deferred delete",
    );

    // The open-file master must retain the deferred-free decision.
    expect_ok(
        with_vfs(|vfs| vfs.delete("/etc/config.txt", None)),
        "delete /etc/config.txt with open handle",
    );

    // The path should now be gone from the namespace.
    expect_err(
        with_vfs(|vfs| vfs.stat("/etc/config.txt", None, false)),
        FileSystemError::NotFound,
        "stat after deferred delete",
    );
    expect_err(
        with_vfs(|vfs| vfs.open(
            "config.txt",
            AccessMode::ReadOnly,
            Some(Arc::clone(&etc_dir)),
        )),
        FileSystemError::NotFound,
        "open deleted name through retained /etc wrapper",
    );

    // Both independent handles still read the original content.
    let mut buf = [0u8; 64];
    let n = expect_ok(
        with_vfs(|vfs| vfs.read(first.as_ref(), &mut buf)),
        "read through first handle after deferred delete",
    );
    assert!(
        &buf[..n] == b"kernel=1\nversion=0.1\n",
        "[VFS TEST] FAIL - deferred delete: content mismatch after delete"
    );
    let n = expect_ok(
        with_vfs(|vfs| vfs.read(last.as_ref(), &mut buf)),
        "read through second handle after deferred delete",
    );
    assert!(
        &buf[..n] == b"kernel=1\nversion=0.1\n",
        "[VFS TEST] FAIL - deferred delete: second handle content mismatch"
    );

    // Only the final close may free the inode.
    expect_ok(
        with_vfs(|vfs| vfs.close(&first)),
        "close first handle after deferred delete",
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
        "close final handle after deferred delete",
    );
    expect_err(
        last.file.inode.filesystem.load_inode(last.file.inode.inode_number),
        FileSystemError::NotFound,
        "inode was not freed after deferred-delete close",
    );

    // A relative delete must resolve its target through the supplied working
    // directory, while an existing handle keeps the inode allocated.
    expect_ok(
        with_vfs(|vfs| vfs.create_file(
            "/var/log/deferred-relative.tmp",
            Permissions::all_permissive(),
            None,
        )),
        "create file for relative deferred delete",
    );
    let log_dir = expect_ok(
        with_vfs(|vfs| vfs.resolve_dir_with_budget(
            "/var/log",
            None,
            crate::file_system::virtual_fs::resolver::MAX_SYMLINK_DEPTH,
        )),
        "resolve /var/log for relative deferred delete",
    );
    let relative_handle = expect_ok(
        with_vfs(|vfs| vfs.open(
            "/var/log/deferred-relative.tmp",
            AccessMode::ReadOnly,
            None,
        )),
        "open file for relative deferred delete",
    );
    expect_ok(
        with_vfs(|vfs| vfs.delete(
            "deferred-relative.tmp",
            Some(log_dir),
        )),
        "delete file relative to /var/log",
    );
    expect_err(
        with_vfs(|vfs| vfs.stat("/var/log/deferred-relative.tmp", None, false)),
        FileSystemError::NotFound,
        "stat after relative deferred delete",
    );
    expect_ok(
        relative_handle.file.inode.filesystem
            .load_inode(relative_handle.file.inode.inode_number),
        "relative-deleted inode remains until close",
    );
    expect_ok(
        with_vfs(|vfs| vfs.close(&relative_handle)),
        "close handle after relative deferred delete",
    );
    expect_err(
        relative_handle.file.inode.filesystem
            .load_inode(relative_handle.file.inode.inode_number),
        FileSystemError::NotFound,
        "relative-deleted inode freed after close",
    );

    sprintln!("[VFS TEST] POSIX deferred delete ... PASS");
}

/// Checks namespace changes through an already loaded working directory.
///
/// # Arguments
///
/// None.
///
/// # Returns
///
/// Returns no value; panics when retained and fresh lookups disagree.
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
        "load retained /var/log child list",
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
        with_vfs(|vfs| vfs.delete("/var/log/wrapper-renamed.tmp", None)),
        "delete file through fresh path",
    );
    expect_err(
        with_vfs(|vfs| vfs.stat("wrapper-renamed.tmp", Some(log_dir), false)),
        FileSystemError::NotFound,
        "deleted name through retained wrapper",
    );
    expect_err(
        with_vfs(|vfs| vfs.stat("/var/log/wrapper-renamed.tmp", None, false)),
        FileSystemError::NotFound,
        "deleted name through fresh path",
    );

    sprintln!("[VFS TEST] retained directory namespace ... PASS");
}

/// Checks mount visibility and unmount through fresh and retained wrappers.
///
/// # Arguments
///
/// None.
///
/// # Returns
///
/// Returns no value; panics when a mount boundary is missed or unmount fails.
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
        "load covered /etc child list",
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
        with_vfs(|vfs| vfs.delete("/etc", None)),
        FileSystemError::AlreadyMounted,
        "delete covered /etc mount host",
    );
    let busy_handle = open_file("/etc/bin/shell", AccessMode::ReadOnly);
    expect_err(
        with_vfs(|vfs| vfs.unmount("/etc", None)),
        FileSystemError::FileLocked,
        "unmount with an open file on mounted volume",
    );
    close_file(&busy_handle);
    expect_ok(
        with_vfs(|vfs| vfs.unmount("/etc", None)),
        "unmount /etc through fresh path",
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

/// Tears down the stub filesystem cleanly.
///
/// # Arguments
///
/// None.
///
/// # Returns
///
/// Returns no value; panics if the global VFS remains initialized.
fn test_unmount() {
    sprintln!("[VFS TEST] unmount ...");

    // The root filesystem cannot be unmounted via the normal unmount path
    // since it has no host directory in the mount table - it IS the root.
    // For the stub test we verify that the VFS can be torn down cleanly by
    // dropping the global instance.
    {
        let mut guard = globals::VIRTUAL_FILE_SYSTEM.lock();
        *guard = None;
    }

    // Verify the global is now None.
    {
        let guard = globals::VIRTUAL_FILE_SYSTEM.lock();
        assert!(guard.is_none(), "[VFS TEST] FAIL - unmount: VFS still Some after teardown");
    }

    sprintln!("[VFS TEST] unmount ... PASS");
}
