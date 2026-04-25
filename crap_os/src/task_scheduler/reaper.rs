//! Task Reaper and Tombstone Cleanup
//!
//! This module implements the two-phase task death pipeline, which ensures that
//! task stacks are never freed while the CPU might still be executing on them.
//! The pipeline works as follows:
//! 
//! Task exits or is killed  -> Timer ISR tick N   -->  Timer ISR tick N+1
//! |                           |                       |
//! |_ mark task Dying          |_ tombstone_cleanup(): |_ tombstone_cleanup()
//! |_ queue_task_reaper(id) -> |   (processes tasks    |   <-- finds task in
//! |_ call schedule()          |    marked Dead from   |   TOMBSTONE_QUEUE,
//!                             |    previous tick;     |   calls remove_task(),
//!                             |    none yet)          |   which drops _stack
//!                             |                       |
//!                             |_ reap_dying_tasks():  |_ reap_dying_tasks()
//!                             |   <-- finds task in   |   (nothing new)
//!                             |       REAPER_QUEUE,   |
//!                             |       Dying -> Dead,  |
//!                             |       pushes to       |
//!                             |       TOMBSTONE_QUEUE |
//!                             |                       |
//!                             |_ schedule() (switch)  |_ ...
//!
//! The two-tick delay between a task being switched away from and its stack
//! being freed is the key invariant. When `switch_to` executes its final
//! `ret`, the CPU is still briefly executing on the outgoing task's stack.
//! By the time `tombstone_cleanup` sees a task in the TOMBSTONE_QUEUE, at
//! least one full timer tick has elapsed, guaranteeing the CPU has moved to
//! a different stack.
//! 
//! # Lock ordering
//!
//! Both `reap_dying_tasks` and `tombstone_cleanup` acquire REAPER_QUEUE (or
//! TOMBSTONE_QUEUE) and then SCHEDULER while holding the first lock. This
//! nested acquisition is safe only as long as a consistent global lock ordering
//! is maintained everywhere, and no other code path acquires these locks in the
//! reverse order, which would lead to a deadlock.

use super::scheduler::SCHEDULER;
use super::task::{TaskState, TaskId};
use crate::spinlock::StaticIrqSpinLock;

/// Maximum number of tasks that can be pending in the reaper queue at once.
/// Tasks are pushed here when they exit or are killed and transition to
/// `Dying`, and drained each timer tick by `reap_dying_tasks`. 64 slots is
/// more than enough for a single tick.
const REAPER_QUEUE_SIZE: usize = 64;

/// Bitmask for wrapping the reaper queue's head/tail indices. Only valid
/// because REAPER_QUEUE_SIZE is a power of two, and `index & MASK` is
/// equivalent to `index % SIZE` but without the division.
const REAPER_QUEUE_MASK: usize = REAPER_QUEUE_SIZE - 1;

/// Maximum number of tasks that can be pending tombstone cleanup at once.
/// Tasks are pushed here by `reap_dying_tasks` when they transition from
/// `Dying` to `Dead`, and drained on the following timer tick by
/// `tombstone_cleanup`. Same sizing rationale as `REAPER_QUEUE_SIZE`.
const TOMBSTONE_QUEUE_SIZE: usize = 64;

/// Bitmask for wrapping the tombstone queue's head/tail indices. This holds
/// the same power-of-two rationale as `REAPER_QUEUE_MASK`.
const TOMBSTONE_QUEUE_MASK: usize = TOMBSTONE_QUEUE_SIZE - 1;

/// Errors that can be returned by reaper operations.
#[allow(dead_code)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReaperError {
    /// The reaper queue is full, and the dying task's ID could not be enqueued.
    /// This should never happen in normal operation, as it means more than
    /// `REAPER_QUEUE_SIZE` tasks died within a single timer tick.
    ReaperQueueFull,

    /// The tombstone queue is full. A `Dead` task's ID could not be enqueued
    /// for cleanup. This should never happen in normal operation, as it means
    /// more than `TOMBSTONE_QUEUE_SIZE` tasks were reaped within a single
    /// timer tick.
    TombstoneQueueFull,

    /// The referenced `TaskId` does not exist in the task table. Returned by
    /// `wake()` when the target task has already been cleaned up, or when an
    /// invalid ID is passed.
    UnknownTask,
}

/// A fixed-size FIFO queue of `TaskId`s for tasks that are `Dying` and need
/// to be transitioned to `Dead` on the next timer tick.
///
/// Pushed to by `queue_task_reaper` (called from where a task exits or is
/// killed). Drained by `reap_dying_tasks` (called from the timer ISR).
pub(super) struct ReaperQueue {
    /// Ring buffer of pending task IDs. Slots are `None` when empty.
    queue: [Option<TaskId>; REAPER_QUEUE_SIZE],

    /// Index of the next slot to pop from (oldest entry).
    head: usize,

    /// Index of the next slot to push into (one past the newest entry).
    tail: usize,

    /// Number of entries currently in the queue. Used to distinguish full
    /// from empty (both have head == tail when using a ring buffer).
    queue_len: usize,
}

impl ReaperQueue {
    /// Creates a new, empty reaper queue. `const fn` so it can be used in
    /// static initializers.
    const fn new() -> Self {
        Self {
            queue: [const { None }; REAPER_QUEUE_SIZE],
            head: 0,
            tail: 0,
            queue_len:  0,
        }
    }

    /// Pushes a `TaskId` onto the back of the queue.
    /// 
    /// # Arguments
    /// 
    /// * `task_id` - ID of the task to enqueue.
    /// 
    /// # Returns
    ///
    /// Returns `Err(ReaperQueueFull)` if the queue has no space. The caller
    /// should treat this as a fatal scheduler error, as a dying task that
    /// cannot be enqueued will never be reaped and its resources will leak.
    fn push(&mut self, task_id: TaskId) -> Result<(), ReaperError> {
        if self.queue_len == REAPER_QUEUE_SIZE {
            return Err(ReaperError::ReaperQueueFull);
        }

        self.queue[self.tail] = Some(task_id);

        // Wrap tail using bitwise AND instead of modulo. Safe because
        // REAPER_QUEUE_SIZE is a power of two.
        self.tail = (self.tail + 1) & REAPER_QUEUE_MASK;
        self.queue_len += 1;
        Ok(())
    }

    /// Pops a `TaskId` from the front of the queue.
    ///
    /// # Returns
    /// 
    /// Returns `Some` task ID from the front of the queue, or `None` if the
    /// queue is empty.
    fn pop(&mut self) -> Option<TaskId> {
        if self.queue_len == 0 {
            return None;
        }

        // `take()` moves the value out and leaves None in the slot, which is
        // important for keeping the backing array clean.
        let task_id = self.queue[self.head].take();

        // Wrap head using bitwise AND
        self.head = (self.head + 1) & REAPER_QUEUE_MASK;
        self.queue_len -= 1;
        task_id
    }
}

/// A fixed-size FIFO queue of `TaskId`s for tasks that are `Dead` and ready
/// to have their resources (including their kernel stack) freed.
///
/// Pushed to by `reap_dying_tasks` when it transitions a task from
/// `Dying` -> `Dead`. Drained by `tombstone_cleanup` on the following (N + 1)
/// timer tick, providing the one-tick delay that guarantees the CPU is no
/// longer on the task's stack.
pub(super) struct TombstoneQueue {
    /// Ring buffer of pending task IDs awaiting resource cleanup.
    queue: [Option<TaskId>; TOMBSTONE_QUEUE_SIZE],

    /// Index of the next slot to pop from (oldest entry).
    head: usize,

    /// Index of the next slot to push into (one past the newest entry).
    tail: usize,

    /// Number of entries currently in the queue.
    queue_len: usize,
}

impl TombstoneQueue {
    /// Creates a new, empty tombstone queue. `const fn` for static use.
    const fn new() -> Self {
        Self {
            queue: [const { None }; TOMBSTONE_QUEUE_SIZE],
            head: 0,
            tail: 0,
            queue_len:  0,
        }
    }

    /// Pushes a `TaskId` onto the back of the tombstone queue.
    /// 
    /// # Arguments
    /// 
    /// * `task_id` - ID of the task to enqueue.
    /// 
    /// # Returns
    ///
    /// Returns `Err(TombstoneQueueFull)` if full. Same severity as
    /// `ReaperQueue::push`, as a task that cannot be enqueued here will never
    /// have its stack freed, causing a heap leak.
    fn push(&mut self, task_id: TaskId) -> Result<(), ReaperError> {
        if self.queue_len == TOMBSTONE_QUEUE_SIZE {
            return Err(ReaperError::TombstoneQueueFull);
        }

        self.queue[self.tail] = Some(task_id);

        // Wrap tail using bitwise AND
        self.tail = (self.tail + 1) & TOMBSTONE_QUEUE_MASK;
        self.queue_len += 1;
        Ok(())
    }

    /// Pops a `TaskId` from the front of the queue.
    ///
    /// # Returns
    /// 
    /// Returns `Some` task ID from the front of the queue, or `None` if the
    /// queue is empty.
    fn pop(&mut self) -> Option<TaskId> {
        if self.queue_len == 0 {
            return None;
        }

        // `take()` moves the value out and leaves None in the slot, which is
        // important for keeping the backing array clean.
        let task_id = self.queue[self.head].take();

        // Wrap head using bitwise AND
        self.head = (self.head + 1) & TOMBSTONE_QUEUE_MASK;
        self.queue_len -= 1;
        task_id
    }
}

/// Global reaper queue. Pushed to on task termination (either for exit or
/// kill), and drained by `reap_dying_tasks` (in ISR context).
pub(super) static REAPER_QUEUE: StaticIrqSpinLock<ReaperQueue> =
    StaticIrqSpinLock::new(ReaperQueue::new());

/// Global tombstone queue. Written by `reap_dying_tasks` and drained by
/// `tombstone_cleanup`, both in ISR context on consecutive timer ticks.
pub(super) static TOMBSTONE_QUEUE: StaticIrqSpinLock<TombstoneQueue> =
    StaticIrqSpinLock::new(TombstoneQueue::new());

/// Enqueues a `Dying` task for reaping on the next timer tick.
///
/// The task must already be in the `Dying` state when this is called, as
/// `reap_dying_tasks` will ignore any task that is not marked `Dying` when it
/// drains this queue.
/// 
/// # Arguments
/// 
/// * `task_id` - ID of the task to push to the reaper queue. 
///
/// # Errors
///
/// Returns `Err(ReaperQueueFull)` if the queue has no space. The caller
/// should treat this as fatal and unrecoverable error.
pub fn queue_task_reaper(task_id: TaskId) -> Result<(), ReaperError> {
    REAPER_QUEUE.lock().push(task_id)?;
    Ok(())
}

/// Drains the reaper queue, transitioning all `Dying` tasks to `Dead` and
/// enqueuing them for tombstone cleanup on the next tick.
///
/// Called from the timer ISR on every tick (after `tombstone_cleanup`), so that
/// any tasks promoted to `Dead` here are not immediately freed; they will be
/// freed on the next tick's `tombstone_cleanup` pass.
pub fn reap_dying_tasks() {
    // Acquire the reaper queue first. If it's empty, return immediately
    // to avoid taking the scheduler lock unnecessarily.
    let mut queue = REAPER_QUEUE.lock();
    if queue.queue_len == 0 {
        return;
    }

    // At least one `Dying` task exists. Acquire the scheduler lock to update
    // task states.
    let mut scheduler = SCHEDULER.lock();

    loop {
        let task_id = match queue.pop() {
            Some(id) => id,
            None => break,  // Queue is empty; drain is complete.
        };

        if let Some(task) = scheduler.get_task_mut(task_id) {
            // Only reap tasks that are actually Dying. If a task ID is in
            // the queue, but the task is in some other state, something has
            // gone wrong elsewhere, and we skip it rather than corrupting
            // state.
            if task.state == TaskState::Dying {
                // Advance state to Dead. The task will not be rescheduled
                // from this point forward, as schedule() never re-queues
                // Dead tasks.
                task.state = TaskState::Dead;

                // Push to the tombstone queue for stack cleanup on the
                // next timer tick. The extra tick of delay ensures the
                // CPU is no longer executing on this task's stack before
                // remove_task() drops it.
                if TOMBSTONE_QUEUE.lock().push(task_id.clone()).is_err() {
                    // Tombstone queue full; this task's stack will leak.
                    // This is a serious error, but we cannot panic here
                    // (as we're in an ISR), so we log and continue.
                    crate::hardware_manager::sprint(
                        "[REAPER] Failed to queue tombstone cleanup...\n");
                }
            }
        }
    }
}

/// Drains the tombstone queue, permanently removing all `Dead` tasks from the
/// task table and freeing their resources (including their kernel stacks).
///
/// Called from the timer ISR on every tick, before `reap_dying_tasks`. This
/// ordering ensures that tasks promoted to `Dead` in the current tick's
/// `reap_dying_tasks` pass are not freed until the next tick, thus providing
/// the minimum one-tick delay between a task being switched away from and
/// its stack being dropped.
///
/// Dropping the `Task` here will drop its `_stack: Box<[u8]>`, returning the
/// kernel stack memory to the heap. This is the only place where task stacks
/// are freed.
pub fn tombstone_cleanup() {
    // Acquire the tombstone queue. If empty, return without touching the
    // scheduler lock.
    let mut queue = TOMBSTONE_QUEUE.lock();
    let dead_task_count = queue.queue_len as u64;

    if dead_task_count == 0 {
        return;
    }

    // At least one `Dead` task is ready for cleanup. Acquire the scheduler
    // lock to remove the task slots.
    let mut scheduler = SCHEDULER.lock();
    
    loop {
        let task_id = match queue.pop() {
            Some(id) => id,
            None => break,  // Queue is empty; drain is complete.
        };

        // Remove the task slot from the scheduler table. This drops the
        // `Task` struct, which drops `_stack`, freeing the kernel stack.
        // By the time we get here, at least one full timer tick has elapsed
        // since the task was switched away from, so the CPU is guaranteed
        // to no longer be executing on this stack.
        scheduler.remove_task(task_id);
    }

    // TODO: for debugging purposes; can clean up later on.
    // Log how many tasks were cleaned up, so we can verify during testing.
    if crate::globals::DEBUG_LEVEL == crate::DebugLevel::INFO {
        crate::helper_functions::print_u64_field(
            "\n[REAPER] Tombstone cleanup reaped dead task(s): ",
            dead_task_count);
    }
}
