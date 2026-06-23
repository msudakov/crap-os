//! Process Thread Representation and Lifecycle Management Module
//!
//! A [`Thread`] is the unit of process membership, belonging to exactly one
//! [`Process`], and wraps exactly one scheduler `Task`. Threads are the bridge
//! between the scheduler, which owns execution context, and the process
//! manager, which owns lifetime and ownership.
//!
//! The ownership model can be summarized as follows:
//!
//!   Process  --(Arc)-->  IrqSpinLock<Thread>  --(Weak)-->  Process
//!                                 |
//!                                 V
//!                          task_id: TaskId  --(lookup)-->  Task
//!
//!   Task  ---(Weak)--->  IrqSpinLock<Thread>
//!
//! - [`Process`] holds a strong `Arc` to each of its threads via thread list
//! - [`Thread`] holds a `Weak` back-reference to its owning [`Process`]
//! - [`Task`] holds a `Weak` back-reference to its [`Thread`]
//! - [`Thread`] references its [`Task`] by [`TaskId`] for forward lookup
//!
//! Thus, no strong ownership cycles exist, and the process is the ultimate
//! owner. When the last `Arc<Process>` is dropped, the thread list is dropped
//! with it; however, this is just a property of an `Arc`, and the termination
//! and cleanup happen in the reverse order, from the bottom up
//! (from Task -> to Thread -> to Process).
//!
//! The [`ThreadState`] and [`TaskState`] states are parallel but distinct:
//!   - `Active`  :  Alive and schedulable; Ready/Running in `TaskState`
//!   - `Waiting` :  Voluntarily blocked, awaiting a call to [`wake`]
//!   - `Dying`   :  Exit is initiated (async), and the reaper is queued
//!   - `Dead`    :  Reaper and cleanup have completed, and it is safe to drop
//!
//! The scheduler owns `TaskState`; the process manager owns `ThreadState`.
//! They are kept in sync at transition points by the scheduler, reaper,
//! and exit path, but there is no automatic enforcement between them.

use core::sync::atomic::{AtomicU64, Ordering};
use alloc::sync::{Arc, Weak};
use crate::spinlock::IrqSpinLock;
use super::process::Process;
use crate::task_scheduler::TaskId;

/// Uniquely identifies a thread within the process manager.
///
/// Unlike [`TaskId`], which is slot-based and recycled, `ThreadId` is a
/// simple monotonically increasing counter. Thread IDs are never reused, and
/// each new thread gets a globally unique ID for its entire lifetime.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct ThreadId(u64);

impl ThreadId {
    /// The `ThreadId` of the idle thread.
    ///
    /// Assigned to the idle thread during scheduler initialization. Like
    /// [`TaskId::IDLE`], this is a fixed constant that is never reassigned
    /// or recycled.
    pub const IDLE: ThreadId = ThreadId(0);

    /// Allocates the next unique `ThreadId`.
    ///
    /// Uses a thread-safe atomic counter starting at 1 and incrementing with
    /// [`Ordering::Relaxed`], since the ID only needs to be unique, and no
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

/// Represents the lifecycle state of a thread from the process manager's
/// perspective.
///
/// The state can transition as follows:
///
///               |---------------------------|
///               V                           |
///  [spawn] -> Active -> Waiting -> Active again (via `yield_blocked` / `wake`)
///               |
///               V
///            Dying  ----->  Dead
///            (exit called)  (reaper finished)
///
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum ThreadState {
    /// The thread is alive and its task is schedulable.
    /// Fine-grained Ready/Running distinctions live in `TaskState`.
    Active,

    /// The thread has voluntarily blocked via `yield_blocked` and is
    /// waiting to be explicitly woken via `wake`.
    Waiting,

    /// The thread has begun exiting. Its task has also been marked as
    /// `TaskState::Dying`, and the reaper has been queued. But cleanup is not
    /// yet complete, and it is not safe to drop the thread yet.
    Dying,

    /// The reaper has run, and tombstone cleanup has finished. At this point,
    /// [`Thread::task_id`] has been set to `None` by the scheduler, and the
    /// thread is safe to drop. 
    Dead,
}

/// A thread: the unit of process membership and lifecycle tracking.
///
/// A `Thread` wraps exactly one scheduler [`Task`] and belongs to exactly one
/// [`Process`]. It is the bridge between the scheduler (which owns execution
/// context) and the process manager (which owns ownership and lifetime).
///
/// `Thread` instances are always heap-allocated behind an
/// `Arc<IrqSpinLock<Thread>>`, allowing shared ownership across the process
/// thread list and the task's back-reference. Their relationship to processes
/// and tasks is visualized as follows:
///
///   Process
///     |__ Arc<IrqSpinLock<Thread>>  <---------------- thread list entry
///                   |
///                   |-- task_id: Option<TaskId>       forward ref (by ID)
///                   +-- Weak<Process>                 back-ref to owner
///
///   Scheduler
///     |__ TaskSlot
///           |__ Task
///                 |__ thread: Weak<IrqSpinLock<Thread>>  back-ref to thread
///
/// The `Thread` holds a forward reference to its task by [`TaskId`], and the
/// [`Task`] holds a [`Weak`] back-reference to its thread. Neither side holds
/// a strong owning reference to the other - the scheduler owns the `Task`
/// exclusively, and the `Process` owns the `Thread` exclusively.
#[allow(dead_code)]
pub struct Thread {
    /// Unique identifier for this thread, never reused or recycled.
    pub id: ThreadId,

    /// The [`TaskId`] of the scheduler task backing this thread.
    ///
    /// Set to `Some` by [`Process::spawn_thread`] immediately after the task
    /// is inserted into the scheduler. Set back to `None` by
    /// [`Scheduler::remove_task`] during tombstone cleanup, indicating that the
    /// backing task has been fully removed from the scheduler.
    ///
    /// `None` is only expected in two situations:
    /// - Briefly during thread construction, before the task is inserted/queued
    /// - After reaper has completed tombstone cleanup, and the thread is `Dead`
    pub task_id: Option<TaskId>,

    /// Human-readable name for this thread, used in debug output and logging.
    pub name: &'static str,

    /// The current lifecycle state of this thread, kept in sync with the
    /// backing [`TaskState`] by the scheduler and the reaper.
    pub state: ThreadState,

    /// Weak back-reference to the owning [`Process`].
    ///
    /// Deliberately weak to avoid a reference cycle: `Process` holds a strong
    /// `Arc` to each of its threads, so the thread must not hold a strong
    /// reference back. Upgrading this `Weak` should always succeed while the
    /// process is alive, as a dangling upgrade indicates a lifecycle ordering
    /// violation.
    pub process: Weak<Process>,
}

/// # Safety
///
/// `Thread` contains a `Weak<Process>`, which requires `Process: Send + Sync`
/// for the derived `Send` to hold automatically. This impl asserts that
/// `Thread` is safe to send across thread boundaries, which holds as long as
/// all accesses to the inner fields go through the enclosing.
unsafe impl Send for Thread {}

#[allow(dead_code)]
impl Thread {
    /// Creates a new thread belonging to the given process.
    /// 
    /// # Arguments
    /// 
    /// * `name`    - Human-readable name for this thread.
    /// * `process` - `Weak` back-reference to the thread's parent [`Process`].
    /// 
    /// # Returns
    ///
    /// Returns the thread wrapped in `Arc<IrqSpinLock<Thread>>` for shared
    /// ownership between the process thread list and the task back-reference.
    pub fn new(
        name: &'static str,
        process: Weak<Process>,
    ) -> Arc<IrqSpinLock<Thread>> {
        let thread = Arc::new(IrqSpinLock::new(Thread {
            id: ThreadId::next(),
            task_id: None,  // Starts as None, gets filled when task is inserted
            name,
            state: ThreadState::Active,  // Thread is Active, task is Ready
            process,
        }));

        thread
    }

    /// Creates the idle thread, which wraps the idle task representing the
    /// initial kernel execution context.
    ///
    /// Uses [`ThreadId::IDLE`] instead of allocating a new ID. Like the idle
    /// task, the idle thread is permanent, as it is never removed or reaped.
    /// 
    /// # Arguments
    /// 
    /// * `process` - `Weak` back-reference to the thread's idle [`Process`].
    /// 
    /// # Returns
    ///
    /// Returns the thread wrapped in `Arc<IrqSpinLock<Thread>>` for shared
    /// ownership between the process thread list and the task back-reference.
    pub fn new_idle(process: Weak<Process>) -> Arc<IrqSpinLock<Thread>> {
        let thread = Arc::new(IrqSpinLock::new(Thread {
            id: ThreadId::IDLE,
            task_id: None,
            name: "Idle",
            state: ThreadState::Active,
            process,
        }));
        thread
    }

    /// Initiates exit of this thread by queueing the `task_killer` `SystemTask`
    ///  to mark the backing task as `TaskState::Dying` and signal a forced
    /// reschedule if it is currently running.
    /// 
    /// If thread is already [`ThreadState::Dying`] or [`ThreadState::Dead`],
    /// this is a no-op to prevent double-exit. Can be called by the [`Thread`]
    /// itself or externally via [`exit_thread`].
    pub fn exit(&mut self) {
        if self.state == ThreadState::Dead || self.state == ThreadState::Dying 
            || self.task_id.is_none() {
                return
        }

        // `SystemTask`routines accept a single u64 argument, so we use a helper
        // function to encode the two components of `TaskId` into a single u64.
        let task_id_u64 = crate::helper_functions::compress_task_id(
            self.task_id.unwrap());

        // Queue the task killer `SystemTask` to prepare the thread's task for
        // termination and drop. Reaper and tombstone cleanup will be called by
        // this `SystemTask`.
        crate::system_core::queue_system_task(
            crate::system_core::system_tasks::task_killer, task_id_u64);
    }
}

/// Initiates a thread's exit sequence.
///
/// This is the public entry point for exiting a thread from outside the
/// thread itself. It acquires the [`IrqSpinLock`] and delegates to
/// [`Thread::exit`].
/// 
/// # Arguments
/// 
/// * `thread` - the given thread to terminate.
#[allow(dead_code)]
pub fn exit_thread(thread: Arc<IrqSpinLock<Thread>>) {
    thread.lock().exit();
}
