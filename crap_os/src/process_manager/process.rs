//! Process Representation and Thread Management Module
//!
//! A [`Process`] is the unit of resource ownership. It owns a collection of
//! [`Thread`]s and holds the page table root PML4 (`cr3`) that will define its
//! address space when user mode is introduced. Every thread belongs to exactly
//! one process, and a process must always have at least one live thread.
//!
//! The ownership model is represented as follows:
//!
//!   ProcessManager
//!         |
//!       (Arc)
//!         |
//!         V
//!      Process
//!         |
//!   (IrqSpinLock)
//!         |
//!       (Arc)
//!         |
//!         V
//!    IrqSpinLock<Thread>  ---(Weak)-->  Process
//!         |
//!      task_id --(scheduler lookup)-->  Task
//!                                         |
//!                                       (Weak)
//!                                         V
//!                                  IrqSpinLock<Thread>
//!
//! - [`ProcessManager`] holds a strong `Arc` to each process
//! - [`Process`] holds a strong `Arc` to each of its threads via a locked list
//! - [`Thread`] holds a `Weak` back-reference to its owning process
//! - No strong ownership cycles exist at any level
//!
//! Processes are created by [`ProcessManager::create_process`], which
//! constructs the process and immediately spawns its main thread via
//! [`Process::spawn_thread`]. A process with no threads is an intermediate
//! construction/termination state only, as the non-empty invariant is
//! established by the time [`create_process`] returns.
//!
//! Each process stores a `cr3` value representing the physical address of its
//! PML4 page table root. This field is reserved for future user mode support;
//! all current kernel processes share the same address space, so `cr3` is
//! stored, but not yet acted upon.

use core::sync::atomic::{AtomicU64, Ordering};
use alloc::sync::Arc;
use alloc::vec::Vec;
use crate::spinlock::IrqSpinLock;
use super::thread::{Thread, ThreadId};
use crate::task_scheduler::{insert_and_queue_task, SchedulerError};
use crate::task_scheduler::task::Task;

/// Uniquely identifies a process within the process manager.
///
/// `ProcessId` is a simple monotonically increasing counter. IDs are never
/// reused, and each new process receives a globally unique ID for its entire
/// lifetime.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct ProcessId(u64);

impl ProcessId {
    /// The `ProcessId` of the idle process.
    ///
    /// The idle process is a permanent singleton created during scheduler
    /// initialization. It owns the idle thread and task, which represent
    /// the initial kernel execution context. Its ID is fixed at 0 and is
    /// never reassigned.
    pub const IDLE: ProcessId = ProcessId(0);

    /// Allocates the next unique `ProcessId`.
    ///
    /// Uses a thread-safe atomic counter starting at 1, incrementing with
    /// [`Ordering::Relaxed`] since the ID only needs to be unique, and no
    /// ordering guarantees relative to other data are required.
    #[inline]
    pub fn next() -> Self {
        static COUNTER: AtomicU64 = AtomicU64::new(1);
        Self(COUNTER.fetch_add(1, Ordering::Relaxed))
    }

    /// Returns the underlying `u64` value, primarily used for debug output
    /// and logging.
    /// 
    /// # Returns
    /// 
    /// Returns the underlying `u64` value.
    #[inline]
    pub fn as_u64(self) -> u64 {
        self.0
    }
}

/// A process: the unit of resource ownership and address space.
///
/// A process owns a list of [`Thread`]s and holds the page table root for
/// its address space. It is always heap-allocated behind an `Arc<Process>`,
/// allowing shared ownership between the [`ProcessManager`] and the `Weak`
/// back-references held by each of its threads.
///
/// # Invariants
///
/// - A process must never be empty, and it must own at least one thread at all
///   times after construction. [`ProcessManager::create_process`] guarantees
///   this by always spawning a main thread before returning.
/// - The thread list is protected by an [`IrqSpinLock`]. Callers must not
///   hold a [`Thread`] lock while acquiring the thread list lock, as this
///   would violate the established lock ordering and risk deadlock.
///
/// # Lock Ordering
///
/// When both locks must be held simultaneously, always acquire in this order:
/// 1. `Process::threads` (outer)
/// 2. `IrqSpinLock<Thread>` (inner)
pub struct Process {
    /// Unique identifier for this process, never reused or recycled.
    pub id: ProcessId,

    /// Human-readable name for this process, used in debug output and logging.
    pub name: &'static str,

    /// Physical address of this process's page table root (CR3 register value).
    ///
    /// Reserved for future user mode support. All current kernel processes
    /// share the same address space, so this field is stored at construction
    /// but not yet acted upon.
    pub cr3: u64,

    /// The list of threads owned by this process.
    ///
    /// Protected by an [`IrqSpinLock`] since thread spawn and removal may
    /// occur from task context while other cores or interrupt paths may
    /// concurrently read the list. Each entry is an `Arc<IrqSpinLock<Thread>>`;
    /// the process holds strong ownership, while the scheduler's task
    /// back-references are `Weak`.
    pub threads: IrqSpinLock<Vec<Arc<IrqSpinLock<Thread>>>>,
}

#[allow(dead_code)]
impl Process {
    /// Creates a new empty process with the given name and page table root.
    /// 
    /// # Arguments
    /// 
    /// * `name` - Human-readable name for this process.
    /// * `cr3`  - PML4 page table root physical address.
    /// 
    /// # Returns
    ///
    /// Returns the process wrapped in [`Arc`], with no threads yet. Callers
    /// must immediately spawn at least one thread via [`spawn_thread`] to
    /// satisfy the non-empty invariant. In practice, this is always done by
    /// [`ProcessManager::create_process`], which is the intended entry point
    /// for process creation.
    pub(crate) fn new(name: &'static str, cr3: u64) -> Arc<Self> {
        Arc::new(Process {
            id: ProcessId::next(),
            name,
            cr3,
            threads: IrqSpinLock::new(Vec::new()),
        })
    }

    /// Creates the idle process, which owns the idle thread and task.
    ///
    /// Uses [`ProcessId::IDLE`] instead of allocating a new ID. The idle
    /// process is a permanent singleton. It is created once during scheduler
    /// initialization and never removed.
    /// 
    /// # Arguments
    /// 
    /// * `cr3`  - PML4 page table root physical address.
    /// 
    /// # Returns
    ///
    /// Returns the idle process wrapped in [`Arc`]. Like [`new`], it starts
    /// with an empty thread list; the idle thread is added immediately after by
    /// [`ProcessManager::init_idle_process`].
    pub(crate) fn new_idle(cr3: u64) -> Arc<Self> {
        Arc::new(Process {
            id: ProcessId::IDLE,
            name: "Idle",
            cr3,
            threads: IrqSpinLock::new(Vec::new()),
        })
    }

    /// Spawns a new thread in this process.
    ///
    /// Creates a [`Thread`], constructs and queues its backing [`Task`] in the
    /// scheduler, writes the assigned [`TaskId`] back into the thread, and
    /// registers the thread in this process's thread list. The task is inserted
    /// and queued before the thread is registered in the process thread list.
    /// 
    /// # Arguments
    /// 
    /// * `name`  - Human-readable name for this thread.
    /// * `entry` - The thread's entry function to call.
    /// * `arg`   - Single u64 argument to pass to the thread's entry function.
    ///             Pass `0` if not needed.
    ///
    /// # Returns
    ///
    /// Returns a strong `Arc` reference to the new thread on success (allowing
    /// the caller to hold a reference independently of the process thread list)
    /// if successful, or [`SchedulerError`] if the scheduler task table or run
    /// queue is full, in which case no thread or task is registered.
    pub fn spawn_thread(
        self: &Arc<Self>,
        name: &'static str,
        entry: fn(u64),
        arg: u64,
    ) -> Result<Arc<IrqSpinLock<Thread>>, SchedulerError> {
        // Create the new thread object
        let thread = Thread::new(name, Arc::downgrade(self));

        // Create the new task object
        let task = Task::new(entry, arg, Arc::downgrade(&thread));

        // Make the new task immediately eligible for scheduling, but the task
        // will not actually begin executing until the timer ISR next calls
        // `schedule()` and selects it from the head of the ready queue.
        let task_id = insert_and_queue_task(task)?;
        
        // Set the thread's `TaskId` reference, returned by the scheduler
        thread.lock().task_id = Some(task_id);

        // Register the new thread with the parent process
        self.threads.lock().push(Arc::clone(&thread));

        // Return the thread reference
        Ok(thread)
    }

    /// Removes a thread from this process's thread list by [`ThreadId`].
    ///
    /// Called during process cleanup after a thread has fully exited and its
    /// backing task has been cleaned up by the reaper. Once removed, the
    /// process no longer holds a strong reference to the thread, and the `Arc`
    /// will be dropped if no other owner exists.
    /// 
    /// # Arguments
    /// 
    /// * `thread_id` - ID of the thread to be removed from this process's
    ///                 thread list.
    pub fn remove_thread(&self, thread_id: ThreadId) {
        self.threads
            .lock()
            .retain(|thread| thread.lock().id != thread_id);
    }
}
