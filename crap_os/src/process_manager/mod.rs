//! Process Manager Module - Top-Level Authority for Process Lifetime
//!
//! The [`ProcessManager`] is the single owner of all live [`Process`] instances
//! in the kernel. It is responsible for creating processes, registering them
//! in the global process list, and providing lookup facilities.

pub mod process;
pub mod thread;
pub mod manager;

// Re-export public APIs
pub use manager::ProcessManager;
pub use thread::Thread;

/// A no-op entry point used as the mandatory initial thread when creating a
/// Process via `ProcessManager::create_process`.
///
/// Every Process must have at least one thread at creation time to maintain
/// the invariant that a threadless process never exists. However, during early
/// kernel initialization, there may not yet be a meaningful entry point to
/// assign to the initial thread of a kernel process (such as the System
/// process). This stub serves as a placeholder in those cases.
///
/// When scheduled, this thread will simply return immediately, causing
/// `task_exit` to be called via `task_entry_stub`, which will mark the task
/// as `Dead` and queue the dead task reaper.
///
/// # Arguments
/// 
/// * `_arg` - Unused, present only to satisfy the `fn(u64)` signature required
///            by `Task::new` and the scheduler entry point convention.
#[allow(dead_code)]
pub(crate) fn nop_thread_stub(_arg: u64) {}
