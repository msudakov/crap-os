//! Kernel Task Scheduler
//!
//! This module implements the kernel's task scheduler. It uses preemptive
//! scheduling, maintains a table of all live tasks and a round-robin ready
//! queue, and performs context switches in response to timer interrupts.
//!
//! The task table (`tasks: [TaskSlot; MAX_TASKS]`) is a flat array of
//! [`TaskSlot`] entries. The task table has the following structure:
//! 
//! +----------------------------------------------------------+
//! | Task | Task | Task | Task | Task | ... |      Task       |
//! | Slot | Slot | Slot | Slot | Slot | ... |      Slot       |
//! |  0   |  1   |  2   |  3   |  4   | ... | (MAX_TASKS - 1) |
//! +------/      \--------------------------------------------+
//!       /        \
//!      /          \
//!     /            \
//!    /              \
//!   /                \   
//!   | Option<Task>   |
//!   | Generation (u8)|
//! 
//! Each `TaskSlot` contains an optional `Task`, where a `Some` slot holds a
//! live task, regardless of its state, while a `None` slot is free. It also
//! tracks its generation value; generation here means "life cycle" rather than
//! "creation". This is explained in more detail in the struct itself. The ready
//! queue is a power-of-two ring buffer (`queue:[TaskId; QUEUE_SIZE]`) of
//! [`TaskId`]s representing tasks in the `Ready` state.
//!
//! The scheduling algorithm is a first-in-first-out round robin: the task at
//! the head of the queue gets the next time slice. After the slice, if the task
//! is still `Running`, it is moved to `Ready` and appended to the tail, giving
//! all other ready tasks a turn before it runs again. This is the simplest fair
//! scheduling algorithm with O(1) enqueue and dequeue operations. There is no
//! priority system yet; tasks preempted by the timer, tasks woken from being
//! `Blocked`, newly-spawned tasks, etc., are all treated equally.
//!
//! All mutable scheduler state is protected by a single `StaticIrqSpinLock`.
//! Because it is an IrqSpinLock, acquiring it also disables hardware interrupts
//! on the current CPU for the duration of the critical section. This prevents
//! the timer ISR from re-entering `schedule()` while we are in the middle of
//! modifying the task table or queue.
//!
//! CRITICAL RULE: The lock must NEVER be held across a call to `switch_to`
//! because of:
//!   - Deadlock danger: the incoming task will try to acquire the same lock on
//!     its next scheduler interaction; if we hold it during the switch, it can
//!     never acquire it.
//!   - Wrong stack danger: `IrqSpinLockGuard::drop` restores RFLAGS
//!     (re-enabling interrupts) by executing `popfq`. If the guard is dropped
//!     after the switch, `popfq` runs on the incoming task's stack with the
//!     outgoing task's saved flags, thus corrupting the incoming task's
//!     interrupt state.

use core::sync::atomic::{Ordering, AtomicBool};
use alloc::sync::Weak;
use super::switcher::switch_to;
use super::task::{Task, TaskId, TaskState};
use super::queue_task_reaper;
use crate::spinlock::{IrqSpinLock, StaticIrqSpinLock};
use crate::globals::{SYS_FLAG_KERNEL_INIT_COMPLETE, CPU_TICKS_REMAINING,
    TASK_QUANTUM_TICKS, CPU_FORCE_RESCHEDULE};
use crate::process_manager::thread::{Thread, ThreadState};
use crate::processor_control::{CpuId};

/// Maximum number of simultaneously live tasks (in any [`TaskState`]) in
/// [`TaskSlot`]s.
const MAX_TASKS: usize = 256;

/// Capacity of the ready-queue ring buffer; measured in entries, where each
/// entry is a `TaskId`.
/// 
/// This size must be a power of two, so that the `& QUEUE_MASK` wrap operation
/// is correct. If the ready-queue is full, `queue_push` returns
/// `SchedulerError::QueueFull` error.
const QUEUE_SIZE: usize = 256;

/// Bitmask applied to ring-buffer indices to implement power-of-two wrapping.
///
/// Because `QUEUE_SIZE` is a power of two, `index & QUEUE_MASK` is equivalent
/// to `index % QUEUE_SIZE`, but implemented as a single AND instruction rather
/// than a division.
const QUEUE_MASK: usize = QUEUE_SIZE - 1;

/// A single slot in the scheduler's task table.
///
/// The task table is a fixed-size array of [`TaskSlot`]s. Each slot either
/// holds a live [`Task`] or is empty, and carries a generation counter that
/// is incremented each time the slot is freed. This allows [`TaskId`]s to
/// be validated against the current generation, making stale references to
/// recycled slots safely detectable.
///
/// The layout of the task table (tasks: [TaskSlot; MAX_TASKS]) is as follows:
///
///  Index | slot_generation | task
/// -------|-----------------|--------------------------------------
///    0   |        0        | Some(idle_task)  <--- permanent, never freed
///    1   |        g        | Some(Task { id: {1, g}, ... })  <--- occupied
///    2   |        g        | None                            <--- available
///    3   |        g        | Some(Task { id: {3, g}, ... })  <--- occupied
///   ...  |       ...       | ...                             ...
///   255  |        g        | None                            <--- available
///
/// When a task is removed from a slot, `slot_generation` is incremented with
/// wrapping arithmetic. Any [`TaskId`] referencing this slot with the old
/// generation will no longer match and will be treated as stale by [`get_task`]
/// and [`get_task_mut`]. Wrapping after 255 removals from a single slot is
/// considered safe in practice. Slot 0 is permanently occupied by the idle
/// task, and its generation never changes. All other slots are fair game for
/// the rest of the system.
struct TaskSlot {
    /// The task currently occupying this slot, or `None` if the slot is free.
    task: Option<Task>,

    /// Generation counter for this slot.
    ///
    /// Compared against [`TaskId::slot_generation`] during every lookup to
    /// detect stale references. Incremented with [`u8::wrapping_add`] each
    /// time a task is removed from this slot. Initialized to 0 for all slots,
    /// matching [`TaskId::IDLE`] for slot 0 and [`TaskId::PENDING`] sentinel
    /// detection for all others.
    slot_generation: u8,
}

impl TaskSlot {
    /// Creates a new empty slot with generation 0.
    ///
    /// This is `const fn` to allow use in the static [`Scheduler::tasks`] array
    /// initializer.
    const fn new() -> Self {
        Self {
            task: None,
            slot_generation: 0,
        }
    }
}

/// All mutable scheduler state, combined into a single struct, so it can be
/// protected by a single `StaticIrqSpinLock`.
///
/// The following must hold true:
///   - Every task in `queue` has state `Ready` and exists in `tasks`.
///   - Exactly one task has state `Running` at any time; its ID is `current`.
///   - A task that is `Blocked`, `Dying,` or `Dead` is not in `queue`.
///   - A `Dying` or `Dead` task remains in `tasks` until the scheduler performs
///     tombstone cleanup via the reaper (drops it and frees its stack).
///   - `queue_len` always equals the number of IDs between `head` and `tail`
///     in the ring. Specifically: `queue_len == 0` when the queue is empty;
///     and, `queue_len == QUEUE_SIZE` when the queue is full.
///
/// The scheduler's spinlock is always acquired through `SCHEDULER.lock()`, and
/// it must never held across `switch_to`. We define it as `pub(super)` because
/// it needs to be accessible from `task_exit` in `task.rs`.
pub(super) struct Scheduler {
    /// Flat array of [`TaskSlot`]s, where each slot either holds a live
    /// [`Task`] or is empty, and carries a generation counter that is
    /// incremented each time the slot is freed.
    tasks: [TaskSlot; MAX_TASKS],

    /// Round-robin ready queue, implemented as a power-of-two ring buffer.
    ///
    /// Contains the [`TaskId`]s of all tasks currently in `TaskState::Ready`,
    /// ordered by how long they have been waiting (oldest task is at `head`).
    /// Tasks are dequeued/consumed from `head` and enqueued/produced at `tail`.
    queue: [TaskId; QUEUE_SIZE],

    /// Index of the next entry (in the range `[0, QUEUE_SIZE)`) to dequeue; it
    /// is the consumer pointer.
    head: usize,

    /// Index of the next free slot (in the range `[0, QUEUE_SIZE)`) to enqueue;
    /// it is the producer pointer.
    tail: usize,

    /// Number of task IDs currently stored in the ring buffer.
    ///
    /// Tracking this separately avoids the classic ring-buffer ambiguity where
    /// `head == tail` could mean either the queue is empty or it is full.
    /// With this tracker:
    ///  - `queue_len == 0`          -> queue is empty (even if `head == tail`)
    ///  - `queue_len == QUEUE_SIZE` -> queue is full  (even if `head == tail`)
    queue_len: usize,

    /// The `TaskId` of the task that is currently executing on the CPU.
    ///
    /// Updated by `schedule()` before each context switch. The task whose ID
    /// is stored here has `TaskState::Running` and is not in `queue`. `init()`
    /// initializes this to `TaskId::IDLE`; and, it is set to the actual
    /// running task ID at the end of every `schedule()` call.
    /// 
    /// We define it as `pub(super)` because it needs to be accessible from
    /// `task_exit` in `task.rs`.
    pub(super) current: TaskId,
}

#[allow(dead_code)]
impl Scheduler {
    /// Creates the initial empty scheduler state at compile time.
    const fn new() -> Self {
        Self {
            // Each slot is independently initialized to `None`.
            tasks: [const { TaskSlot::new() }; MAX_TASKS],

            // Fill the queue with `IDLE` as a safe placeholder value. Entries
            // are only meaningful between `head` and `head + queue_len`.
            queue: [TaskId::IDLE; QUEUE_SIZE],
            head: 0,
            tail: 0,
            queue_len: 0,
            current: TaskId::IDLE,
        }
    }

    /// Appends a task ID to the tail of the ready queue.
    /// 
    /// # Arguments
    /// 
    /// * `task_id` - The task ID to be appended.
    /// 
    /// # Returns
    ///
    /// Returns `Err(QueueFull)` if the ring buffer is at capacity, which the
    /// caller must handle.
    fn queue_push(&mut self, task_id: TaskId) -> Result<(), SchedulerError> {
        if self.queue_len == QUEUE_SIZE {
            return Err(SchedulerError::QueueFull);
        }
        // Write to the current tail slot, then advance tail with
        // power-of-two wrap.
        self.queue[self.tail] = task_id;
        self.tail = (self.tail + 1) & QUEUE_MASK;
        self.queue_len += 1;
        Ok(())
    }

    /// Removes and returns the task ID at the head of the ready queue.
    ///
    /// # Returns
    /// 
    /// Returns `None` if the queue is empty (i.e., no tasks are `Ready`).
    /// In that case, the scheduler will switch to the idle task.
    fn queue_pop(&mut self) -> Option<TaskId> {
        if self.queue_len == 0 {
            return None;
        }
        // Read from the current head slot, then advance head with
        // power-of-two wrap.
        let task_id = self.queue[self.head];
        self.head = (self.head + 1) & QUEUE_MASK;
        self.queue_len -= 1;
        Some(task_id)
    }

    /// Inserts the idle task directly into slot 0 of the task table.
    ///
    /// Slot 0 is permanently reserved for the idle task and is never freed
    /// or reassigned. This bypasses the normal [`insert_and_queue_task`] path
    /// entirely. The idle task is not queued in the Ready queue, since it is
    /// only switched to when the queue is empty.
    /// 
    /// # Arguments
    /// 
    /// * `idle_task` - Idle task to insert into the task table.
    fn insert_idle_task(&mut self, idle_task: Task) {
        self.tasks[0].slot_generation = 0;
        self.tasks[0].task = Some(idle_task);
    }

    /// Finds a free slot, assigns a [`TaskId`], marks the task `Ready`,
    /// inserts it into the task table, and enqueues it for scheduling.
    ///
    /// This is the single entry point for making a new task known to the
    /// scheduler and immediately eligible for scheduling. The slot index and
    /// current slot generation together form the task's [`TaskId`], which is
    /// written into [`Task::id`] before insertion.
    ///
    /// # Arguments
    /// 
    /// * `task` - The [`Task`] to be inserted and added to the Ready queue.
    ///
    /// # Returns
    ///
    /// Returns the [`TaskId`] assigned to the task on success (so the caller
    /// can store it in [`Thread::task_id`]), [`SchedulerError::TaskTableFull`]
    /// if all 255 non-idle slots are occupied, or [`SchedulerError::QueueFull`]
    /// if the run queue has no space. The latter error should never happen.
    fn insert_and_queue_task(
        &mut self,
        mut task: Task,
    ) -> Result<TaskId, SchedulerError> {
        // We loop over every slot, looking for the first empty one
        for (i, slot) in self.tasks.iter_mut().enumerate() {
            // Once a free slot is located, we proceed to use it
            if slot.task.is_none() {
                // The new `TaskId` gets comprised of the slot index and the
                // current generation value of the slot.
                let task_id = TaskId {
                    slot_index: i,
                    slot_generation: slot.slot_generation
                };

                // Clone the new ID value and assign it to the mutable Task
                task.id = task_id.clone();

                // Mark the new task as `Ready`
                task.state = TaskState::Ready;

                // Populate this slot with the new task
                slot.task = Some(task);

                // Push the new task to the Ready queue. This should never fail
                // because the length of the queue is the same as the size of
                // the task table, so there can never be more tasks in the queue
                // than there are in the table. If this fails for any reason,
                // we want to fault loudly.
                self.queue_push(task_id).expect(
                    "[SCHEDULER] Task inserted, but failed to enqueue...");

                return Ok(task_id);
            }
        }
        Err(SchedulerError::TaskTableFull)
    }

    /// Obtains a shared reference to a `Task` by its ID.
    /// 
    /// Because of how [`TaskSlot`]s are implemented with slot index and slot
    /// generation, this fetch operation is very efficient and executes in
    /// constant time (O(1)).
    /// 
    /// # Arguments
    /// 
    /// * `task_id` - The task ID to get a shared reference for.
    /// 
    /// # Returns
    /// 
    /// Returns a shared reference to the located task, or `None` if no
    /// such task exists in the table.
    fn get_task(&self, task_id: TaskId) -> Option<&Task> {
        self.tasks
            .get(task_id.slot_index)
            .filter(|slot| slot.slot_generation == task_id.slot_generation
                && slot.task.is_some())
            .and_then(|slot| slot.task.as_ref())
    }

    /// Obtains a mutable reference to a `Task` by its ID.
    /// 
    /// Because of how [`TaskSlot`]s are implemented with slot index and slot
    /// generation, this fetch operation is very efficient and executes in
    /// constant time (O(1)). We declare it as `pub(super)` because this
    /// function needs to be accessible from `task_exit` in `task.rs`.
    /// 
    /// # Arguments
    /// 
    /// * `task_id` - The task ID to get a mutable reference for.
    /// 
    /// # Returns
    /// 
    /// Returns a mutable reference to the located task, or `None` if no
    /// such task exists in the table.
    pub(super) fn get_task_mut(&mut self, task_id: TaskId) -> Option<&mut Task> {
        self.tasks
            .get_mut(task_id.slot_index)
            .filter(|slot| slot.slot_generation == task_id.slot_generation
                && slot.task.is_some())
            .and_then(|slot| slot.task.as_mut())
    }

    /// Removes a `Task` by its ID from the table, dropping it and freeing its
    /// heap-allocated stack. The linear scan runs in time O(MAX_TASKS). 
    /// 
    /// # Arguments
    /// 
    /// * `task_id` - ID of the `Task` to remove from the `tasks` table.
    /// 
    /// # Caution
    /// 
    /// The caller is responsible for ensuring the task is not currently running
    /// on any CPU before calling this; dropping a task whose stack is in use
    /// would result in a use-after-free bug and vulnerability.
    

    /// Removes a task from its slot (dropping it and freeing its heap-allocated
    /// stack), updates its backing thread, and increments the slot's generation
    /// counter.
    /// 
    /// Because of how [`TaskSlot`]s are implemented with slot index and slot
    /// generation, this fetch operation is very efficient and executes in
    /// constant time (O(1)). It is a no-op if no task with the ID exists.
    /// Called by [`tombstone_cleanup`] after reaper execution to free slots
    /// occupied by [`TaskState::Dead`] tasks.
    /// 
    /// # Arguments
    /// 
    /// * `task_id` - ID of the [`Task`] to remove from the task table.
    /// 
    /// # Caution
    /// 
    /// The caller is responsible for ensuring the task is not currently running
    /// on any CPU before calling this; dropping a task whose stack is in use
    /// would result in a use-after-free bug and vulnerability.
    /// 
    /// # Panics
    ///
    /// Panics if the task's [`Weak`] thread reference cannot be upgraded.
    /// A task being reaped must still have a live thread, as a dangling `Weak`
    /// at this point indicates a lifecycle ordering violation.
    pub(super) fn remove_task(&mut self, task_id: TaskId) {
        // Double check that the task is still current and is not stale
        if let Some(task) = self.get_task_mut(task_id) {
            // Upgrade the thread reference or panic if the `Weak` is dangling
            let thread = task.thread.upgrade().unwrap();
            {
                // Acquire thread lock
                let mut locked_thread = thread.lock();

                // Unset the `task_id` link field on the parent thread,
                // signaling that the task has been fully unregistered from the
                // scheduler.
                locked_thread.task_id = None;

                // Mark the parent thread as dead before dropping the task
                locked_thread.state = ThreadState::Dead;
            }

            // Dropping the Task here drops its Box<[u8]> stack allocation.
            // This is safe because the task is Dead, and schedule() has
            // already switched away from it and will never switch back. This
            // makes the slot available for future insertions.
            self.tasks[task_id.slot_index].task = None;

            // Increment the slot's generation, invalidating all existing
            // `TaskId`s that referenced this slot
            self.tasks[task_id.slot_index].slot_generation =
                self.tasks[task_id.slot_index].slot_generation.wrapping_add(1);
        }
    }
}

/// Contains errors that can be returned by fallible scheduler operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SchedulerError {
    /// The ready queue ring buffer is at capacity.
    /// There are already `QUEUE_SIZE` tasks in the `Ready` state. The caller
    /// should wait for some tasks to complete or block before retrying.
    QueueFull,

    /// The task table is at capacity.
    /// All `MAX_TASKS` slots are occupied by live tasks. The caller can free a
    /// slot by letting a task run to completion, thus letting the tombstone
    /// cleanup to drop it.
    TaskTableFull,

    /// The referenced `TaskId` does not exist in the task table.
    /// Returned by `wake()` when the target task has already been cleaned up,
    /// or when an invalid ID is passed.
    UnknownTask,
}

// =============================================================================
// Global scheduler instance
// =============================================================================

/// The single global scheduler, protected by an IRQ-safe spinlock.
///
/// `StaticIrqSpinLock` satisfies `Sync` (required for `static`) and disables
/// hardware interrupts for the duration of every lock acquisition, preventing
/// the timer ISR from re-entering the scheduler while the lock is held by a
/// task-context caller (e.g., `spawn` or `wake`).
/// 
/// We declare it as `pub(super)` because it needs to be accessible from
/// `task_exit` in `task.rs`.
pub(super) static SCHEDULER: StaticIrqSpinLock<Scheduler> =
    StaticIrqSpinLock::new(Scheduler::new());

// =============================================================================
// Public API - exported at the module level
// =============================================================================

/// Initializes the scheduler, registers the current execution context as the
/// idle task (`TaskId::IDLE`), and initializes the BSP's per-CPU quantum slot.
///
/// The idle task is the kernel's own `_start` routine that is already running
/// on the higher-half stack. We register it as `TaskId::IDLE`, so that the
/// scheduler can:
///   - Save its RSP the first time the timer ISR preempts it;
///   - Resume it later when needed by restoring that RSP.
/// 
/// # Arguments
/// 
/// * `thread` - `Weak` back-reference to the task's parent `Thread`.
pub fn init_idle(thread: Weak<IrqSpinLock<Thread>>) {
    let mut scheduler = SCHEDULER.lock();

    // Create the idle task descriptor.
    //
    // `Task::new_idle()` does not allocate a real stack; it creates a tiny
    // placeholder `Box<[u8]>`, so the `_stack` field is non-null. The idle
    // task's actual stack is the kernel's own boot stack, which lives for the
    // kernel's entire lifetime.
    //
    // `saved_rsp` starts at 0 here. The first time the timer ISR calls
    // `schedule()` and switches away from the idle task, `switch_to` will
    // write the real RSP value into `saved_rsp` before loading the next
    // task's RSP. When the idle task is eventually scheduled again,
    // `switch_to` will restore that RSP, and `_start` will resume as if
    // `schedule()` had simply returned.
    let idle_task = Task::new_idle(thread);

    // Insert the idle task into the table. The idle task is not enqueued in
    // the ready queue; the scheduler falls back to it implicitly when the queue
    // is empty.
    scheduler.insert_idle_task(idle_task);

    // `current` is set to IDLE here because the idle task is currently running
    scheduler.current = TaskId::IDLE;

    // Initialize the BSP's quantum slot and flag slot. The idle task runs until
    // the first real task is ready; giving it a full quantum is the safest
    // default.
    let cpu = CpuId::current();
    unsafe {
        CPU_TICKS_REMAINING.init(CpuId::current(), TASK_QUANTUM_TICKS);
        CPU_FORCE_RESCHEDULE.init(cpu, AtomicBool::new(false));
    }
}

/// Inserts a new kernel task into the Scheduler's tasks table, and enqueues it
/// on the ready queue.
///
/// # Arguments
/// 
/// * `task` - New [`Task`] to insert and schedule for execution.
///
/// # Returns
/// 
/// Returns the inserted task's new [`TaskId`] on success or [`SchedulerError`]
/// if either the task table or the ready queue is full.
pub fn insert_and_queue_task(task: Task) -> Result<TaskId, SchedulerError> {
    let mut scheduler = SCHEDULER.lock();

    // Enqueue the task ID in the ready queue. If the queue is full, the
    // task remains in the table as `Ready`, but is unreachable by the
    // scheduler until a slot frees up. In practice, this should not happen
    // if QUEUE_SIZE == MAX_TASKS.
    let task_id = scheduler.insert_and_queue_task(task)?;

    Ok(task_id)
}

/// Performs a preemptive context switch, selecting the next `Ready` task from
/// the head of the round-robin queue.
///
/// This is the core of the scheduler. It is called at the end of every timer
/// interrupt tick. From the outgoing task's perspective, it is a function call
/// that simply takes a long time to return. It returns the next time that task
/// is selected by the scheduler, and the task switch executes again with it as
/// the incoming task.
pub unsafe fn schedule() {
    // First, we must disable interrupts. This is technically unnecessary when
    // called from the timer ISR, as the interrupts are already disabled in
    // that context. It is a safe no-op in that case, and we save on passing a
    // special flag argument to track the call site. However, this is critical
    // when the call site is `task::task_exit` or `scheduler::yield_blocked`.
    // When called from a task context, if the interrupts are left enabled and
    // a timer interrupt fires while we're inside `switch_to`, we'll be risking
    // a re-entrance deadlock. That may be very difficult to troubleshoot. We
    // disable interrupts for the entire duration. This prevents a timer IRQ
    // from firing between the lock drop and switch_to, which would cause a
    // recursive schedule() call with stale RSP values and corrupt scheduler
    // state.
    let flags = crate::helper_functions::disable_interrupts_save();

    // Critical section: read and update scheduler state.
    // We compute everything `switch_to` needs while the lock is held, extract
    // raw values (pointer + integer), then release the lock before the switch.
    // We have to use raw values instead of references because a reference into
    // `scheduler` would borrow the `MutexGuard`, keeping the lock alive. We
    // need the lock released before `switch_to`, so we must copy the data out.
    let (old_rsp_ptr, new_rsp, new_cr3) = {
        let mut scheduler = SCHEDULER.lock();

        // Dequeue the next ready task.
        // `queue_pop` returns `None` if no tasks are in the ready queue,
        // meaning every other task is either `Blocked`, `Dying`, `Dead`, or
        // there is only the idle task. In that case, we switch to idle task.
        let next_id = loop {
            // Pop the next task ID from the ready queue
            let task_id = match scheduler.queue_pop() {
                Some(id) => id,
                None => TaskId::IDLE,
            };

            // If the result from above is `None`, we proceed scheduling the
            // idle task, since there is nothing else in the queue.
            if task_id == TaskId::IDLE {
                break task_id;
            }

            // If the result from above is `Some`, it is still possible that
            // the popped task has been forcefully killed while in the Ready
            // queue. We check for this by explicitly looking only for a task
            // in `Ready` state. If this is not the case here, we loop back to
            // the top and try again.
            if let Some(next_task) = scheduler.get_task_mut(task_id) {
                if next_task.state == TaskState::Ready {
                    // Transition the incoming task to Running
                    next_task.state = TaskState::Running;
                    break task_id;
                }
            }
        };

        // Handle the outgoing task.
        // If the outgoing task is still `Running` (i.e., it has not blocked
        // or terminated itself during this time slice), move it to `Ready`
        // and push it to the tail of the queue, so it will run again after all
        // currently-queued tasks have had their turn. If the task is `Blocked`,
        // it will be re-enqueued by `wake()` when the awaited event fires. If
        // the task is `Dying` or `Dead`, the reaper and tombstone cleanup will
        // drop it on a future pass.
        let outgoing_id = scheduler.current;
        if let Some(outgoing) = scheduler.get_task_mut(outgoing_id) {
            if outgoing.state == TaskState::Running {
                outgoing.state = TaskState::Ready;
                
                // Only re-insert the idle task if the kernel initialization
                // has not yet finished, as the idle task may have more steps
                // to execute and must be re-queued to run again.
                let queue_idle_task = !SYS_FLAG_KERNEL_INIT_COMPLETE.load(
                    Ordering::Relaxed);
                
                // If the push fails (queue full), we silently drop the outgoing
                // task from the ready set rather than panic inside an ISR. The
                // task remains in the task table with state `Ready` and can be
                // re-enqueued the next time `wake()` or another scheduling pass
                // processes it. This is a last-resort defence; it should not
                // occur under normal operation if QUEUE_SIZE == MAX_TASKS.
                if outgoing_id != TaskId::IDLE || queue_idle_task {
                    let _ = scheduler.queue_push(outgoing_id);
                }
            }
            // Blocked, Dying, or Dead: do not re-enqueue. The task will be
            // woken or cleaned up through separate mechanisms (`wake`,
            // `remove_task`, etc).
        }

        scheduler.current = next_id;  // Update the current task with incoming

        // Extract raw RSP data for calling `switch_to`, which needs:
        //   old_rsp_ptr - a *mut u64 pointing at `outgoing_task.saved_rsp`,
        //                 so it can write the outgoing RSP there.
        //   new_rsp     - the `incoming_task.saved_rsp` value to restore.
        //
        // We obtain a raw pointer to `saved_rsp` rather than a reference, so
        // we can use it after the lock guard is dropped. The pointer remains
        // valid as long as the task stays in the `tasks` array - which it will,
        // because we only remove tasks explicitly via `remove_task`, and that
        // never happens to a task that is `Running`.
        let old_rsp_ptr = scheduler
            .get_task_mut(outgoing_id)
            .map(|task| &mut task.saved_rsp as *mut u64)
            .unwrap_or(core::ptr::null_mut());

        // Get a reference to the incoming task for CR3 and stack top settings
        let next_task = scheduler.get_task(next_id).unwrap();

        // Only update TSS.rsp[0] if the incoming task can actually run in ring
        // 3. Kernel-only tasks (including idle) never transition to ring 3, so
        // the TSS kernel stack is irrelevant for them. Skipping the update also
        // avoids clobbering a valid TSS entry with a meaningless value when
        // switching between two kernel tasks.
        if next_task.is_user_task {
            crate::processor_control::gdt::set_kernel_stack(next_task.kernel_stack_top);
        }

        // Fetch the new CR3 and saved RSP variables to pass to the switcher,
        // and the remaining quantum ticks for the new task.
        let new_rsp = next_task.saved_rsp;
        let new_cr3 = next_task.cr3;
        let new_quantum = next_task.ticks_remaining;

        // Load the incoming task's quantum into this CPU's per-CPU slot, so the
        // timer ISR sees the correct remaining ticks without acquiring the
        // scheduler lock.
        unsafe { *CPU_TICKS_REMAINING.get_mut(CpuId::current()) = new_quantum };

        (old_rsp_ptr, new_rsp, new_cr3)

        // The lock is released here.
    };

    // Under normal operation, these can never be null/zero. Just a sanity check
    // to avoid a null pointer dereference inside `switch_to` if something
    // has gone wrong, like a race during early boot, or a corrupted task table.
    if old_rsp_ptr.is_null() || new_rsp == 0 {
        if old_rsp_ptr.is_null() {
            crate::hardware_manager::serial::print(
                "\n[SCHEDULER] Outgoing old RSP pointer is null...\n");
        }
        else {
            crate::hardware_manager::serial::print(
                "\n[SCHEDULER] Incoming new RSP is zero...\n");
        }
        return;
    }

    // Perform the context switch. No lock is held at this point, but the
    // interrupts must still be disabled across the call to `switch_to`.
    //  The assembly code in `switch_to` will:
    //    - Push callee-saved registers onto the outgoing task's stack;
    //    - Write RSP into the outgoing task's `saved_rsp`;
    //    - Load RSP from the incoming task's stack;
    //    - Conditionally reload CR3
    //    - Pop callee-saved registers from the incoming task's stack;
    //    - Execute `ret`, jumping into the incoming task's execution context.
    //
    // From the outgoing task's perspective, this function call "returns" the
    // next time it is selected as the incoming task in a future `schedule()`
    // invocation. From the incoming task's perspective it either:
    //   - Returns from a prior `switch_to` call (previously-run task), or
    //   - Enters `task_entry_stub` for the first time (brand-new task).
    unsafe { switch_to(old_rsp_ptr, new_rsp, new_cr3) };

    // Finally, when the outgoing task gets back here, the interrupts are
    // restored to what they were. And if the task doesn't return here (i.e., it
    // finishes and exits or faults and gets reaped that way), that is fine too.
    crate::helper_functions::restore_interrupts(flags);
}

/// Called from the APIC timer ISR on every tick. It performs two jobs:
///
/// Job 1: Sleep queue advancement.
/// Calls [`super::sleep_queue::tick_sleep_queue`] to decrement the head of the
/// sleep delta queue and wake any tasks whose sleep duration has expired. This
/// happens before quantum accounting, so that any task woken this tick is
/// immediately [`TaskState::Ready`] and eligible to be scheduled in the same
/// tick.
///
/// Job 2: Quantum accounting and preemption signaling.
/// Increments the current task's `ticks_executed` counter unconditionally,
/// then decrements `ticks_remaining`. When `ticks_remaining` reaches 1 (the
/// last tick of the quantum), it is reset to [`TASK_QUANTUM_TICKS`] and the
/// function returns `true` to signal the ISR that it should invoke
/// [`schedule`] to preempt the current task.
///
/// We need to use `<= 1` (and not strictly `< 1`) as the preemption threshold,
/// because, at `remaining = 1`, this is the last tick of the quantum.
/// Decrementing to 0 and checking on the next tick would give the task one
/// extra tick beyond its quantum. The logic flows as follows:
///   - After tick 1: remaining = 4, 4 <= 1? No  → decrement → remaining = 3
///   - After tick 2: remaining = 3, 3 <= 1? No  → decrement → remaining = 2
///   - After tick 3: remaining = 2, 2 <= 1? No  → decrement → remaining = 1
///   - After tick 4: remaining = 1, 1 <= 1? Yes → preempt, reset to 4
///
/// # Returns
/// 
/// Returns `true` if the scheduler should preempt the current task, `false` if
/// the current task should continue running.
pub fn on_timer_tick() -> bool {
    // Advance the sleep delta queue. Any tasks woken here will be Ready and
    // eligible for scheduling within this same tick.
    super::sleep_queue::tick_sleep_queue();

    // Read and decrement the per-CPU quantum counter; no scheduler lock needed.
    // This is the only value the timer ISR hot path touches on every tick.
    let cpu = CpuId::current();
    let remaining = CPU_TICKS_REMAINING.get(cpu);

    // `ticks_executed` still lives on the Task and requires the scheduler lock.
    // We only pay this cost once per tick, not per-CPU per-tick in the future.
    {
        let mut scheduler = SCHEDULER.lock();
        let current_id = scheduler.current;
        if let Some(task) = scheduler.get_task_mut(current_id) {
            task.ticks_executed.fetch_add(1, Ordering::Relaxed);
        }
    }

    if *remaining <= 1 {
        // Quantum expired. Reset the per-CPU counter to a full quantum so
        // the incoming task starts fresh. The task's own ticks_remaining
        // field will be overwritten by schedule() when it loads the next
        // task's quantum into this slot.
        unsafe { *CPU_TICKS_REMAINING.get_mut(cpu) = TASK_QUANTUM_TICKS };
        true
    }
    else {
        unsafe { *CPU_TICKS_REMAINING.get_mut(cpu) = remaining - 1 };
        false
    }

    /*let mut scheduler = SCHEDULER.lock();
    let current_id = scheduler.current;

    if let Some(task) = scheduler.get_task_mut(current_id) {
        // Always account for this tick regardless of preemption outcome
        task.ticks_executed.fetch_add(1, Ordering::Relaxed);

        // Check and decrement the quantum countdown
        if task.ticks_remaining <= 1 {
            // Quantum expired; reset to full quantum and request preemption.
            // The reset happens here rather than in `schedule()`, so the task
            // gets a fresh full quantum the next time it runs, regardless of
            // how `schedule()` selects the next task.
            task.ticks_remaining = crate::globals::TASK_QUANTUM_TICKS;
            return true;
        }

        // Quantum still has time remaining, so we decrement it and continue.
        task.ticks_remaining -= 1;
        false
    }
    else {
        // No task is currently registered as running.
        // Signal the ISR to call schedule(), so it can find one.
        true
    }*/
}

/// Blocks the calling task and immediately yields the CPU to the next ready
/// task.
///
/// Transitions the current task to `TaskState::Blocked` (and its parent thread
/// to `ThreadState::Waiting`), removes it from the ready queue, then calls
/// `schedule()` as if a timer tick had just fired. Because the task's state is
/// `Blocked`, `schedule()` will not re-enqueue it, and the task will remain
/// suspended until `wake(id)` is called for it.
///
/// The calling task resumes from this function when another task or ISR calls
/// `wake(id)`, and the scheduler eventually selects it from the ready queue.
/// This must be called from task context; it must not be called from an ISR, as
/// the calling task must have a valid `saved_rsp` that `schedule()` can save
/// the outgoing RSP into.
pub fn yield_blocked() {
    // Transition the current task to `Blocked` state under the lock
    {
        let mut scheduler = SCHEDULER.lock();
        let this_id = scheduler.current;
        if let Some(task) = scheduler.get_task_mut(this_id) {
            // Reset the quantum so the task gets a fresh timeslice on wakeup
            task.ticks_remaining = crate::globals::TASK_QUANTUM_TICKS;

            task.state = TaskState::Blocked;
            // No queue removal is needed because the task was `Running`, so it
            // was not in the ready queue to begin with. `schedule()` will not
            // re-enqueue it because its state will not be `Running`.

            // We also mark the task's parent thread as waiting
            task.thread.upgrade().unwrap().lock().state = ThreadState::Waiting;
        }
        // Lock is released here
    };

    // Force an immediate reschedule. Calling `schedule()` directly here (rather
    // than waiting for the next timer tick) gives up the remainder of this
    // task's time slice immediately, which is the expected semantics for a
    // voluntary block.
    unsafe { schedule() };
}

/// Transitions a `Blocked` task back to `Ready` and enqueues it for scheduling,
/// also marking its parent `Thread` as `ThreadState::Active`.
///
/// Typically called by an ISR or another task when an event the blocked task
/// was waiting for has occurred (e.g., data arrived in a ring buffer, a timer
/// fired, a lock became available).
///
/// If the task is already `Ready` or `Running`, this function succeeds silently
/// without double-enqueuing. Calling `wake` on an already-ready task is
/// harmless. This makes it safe to call from ISRs that may fire multiple times
/// before the task drains the event (e.g., a keyboard ISR).
/// 
/// # Arguments
/// 
/// * `task_id` - ID of the task to wake.
///
/// # Thread/ISR safety
/// 
/// Safe to call from both task context and interrupt context. The
/// `IrqSpinLock` handles the difference in interrupt-enable state correctly.
///
/// # Returns
/// 
/// Returns `Ok(())` on success (including the case of waking a non-blocked
/// task), or `Err(UnknownTask)` if no task with this ID exists.
pub fn wake(task_id: TaskId) -> Result<(), SchedulerError> {
    let mut scheduler = SCHEDULER.lock();
    let task = scheduler
        .get_task_mut(task_id)
        .ok_or(SchedulerError::UnknownTask)?;

    if task.state == TaskState::Blocked {
        // Mark the task as ready
        task.state = TaskState::Ready;

        // Mark the task's parent thread as active
        task.thread.upgrade().unwrap().lock().state = ThreadState::Active;

        // Enqueue the now-ready task at the tail of the round-robin queue.
        scheduler.queue_push(task_id)?;
    }

    Ok(())
}

/// Looks up the ID of the task currently executing on the CPU.
/// 
/// The returned ID is a snapshot; by the time the caller inspects it, a context
/// switch may have occurred, and a different task may be running.
///
/// # Returns
/// 
/// Returns the `TaskId` of the task currently executing on the CPU.
/// 
/// # Thread/ISR safety
/// 
/// Acquires the scheduler lock for a single field read, then immediately
/// releases it. Safe to call from both task context and ISR context.
#[inline]
pub fn get_current_task_id() -> TaskId {
    SCHEDULER.lock().current
}

/// Marks a task as [`TaskState::Dying`], marks the backing thread as
/// [`ThreadState::Dying`] to signal to the process manager that this thread is
/// on its way out, and queues a forced reschedule if the task is currently
/// running.
///
/// Dying and Dead tasks are not re-enqueued by [`schedule`], so the task will
/// not run again after being switched out. Slot cleanup is deferred to the
/// reaper and the tombstone cleanup on a later clock tick.
/// 
/// # Arguments
/// 
/// * `task_id_u64` - Encoded index and generation components of a [`TaskId`].
///
/// # Panics
///
/// Panics if the task's [`Weak`] thread reference cannot be upgraded.
/// A task being killed must still have a live thread, as a dangling `Weak`
/// at this point indicates a lifecycle ordering violation.
pub fn kill_task(task_id_u64: u64) {
    // `SystemTask`routines accept a single u64 argument, so we use a helper
    // function to decode and expand the two components of `TaskId`, which were
    // encoded by the caller into the u64 parameter to this function.
    let task_id = crate::helper_functions::expand_task_id(task_id_u64);
    
    // This checks if the task is currently executing on this CPU
    let is_current = {
        let mut scheduler = SCHEDULER.lock();

        // Make sure the task is not stale
        if let Some(task) = scheduler.get_task_mut(task_id) {
            // Set the task state to dying
            task.state = TaskState::Dying;

            // We also mark the task's parent thread as dying
            task.thread.upgrade().unwrap().lock().state = ThreadState::Dying;

            // Fill `is_current`
            scheduler.current == task_id
        }
        else {
            false
        }
    };  // The lock is released here

    // Queue task reaper to mark the task as dead on the next tick
    if queue_task_reaper(task_id).is_err() {
        crate::hardware_manager::sprint(
            "\n[REAPER] Failed to queue task reaper...\n");
    }

    if is_current {
        // Signal the timer ISR to reschedule regardless of task quantum once
        // the system task queue is fully drained.
        CPU_FORCE_RESCHEDULE.current().store(true, Ordering::SeqCst);
    }
}

/// Marks the currently-running task as `Dying` and queues the task reaper to
/// run on the next timer tick.
/// 
/// Called from exception handlers when a recoverable fault is attributed to the
/// current task. This is the abnormal termination counterpart to `task_exit()`
/// in `task.rs`, and both paths converge on the same `Dying` task state and the
/// same reaper, so the cleanup machinery does not need to distinguish between
/// normal and abnormal task termination.
///
/// We do not call `schedule()` from here; the caller is responsible for that,
/// since the call site (an exception handler) may need to do additional work,
/// such as logging, before switching away. The lock is released before
/// returning, so the caller can call `schedule()` without holding it.
pub fn kill_current_task() {
    {
        let mut scheduler = SCHEDULER.lock();
        let this_id = scheduler.current;
        if let Some(task) = scheduler.get_task_mut(this_id) {
            task.state = TaskState::Dying;

            // We also mark the task's parent thread as dying
            task.thread.upgrade().unwrap().lock().state = ThreadState::Dying;

            // Queue task reaper to mark the task as Dead on the next tick
            if queue_task_reaper(this_id).is_err() {
                crate::hardware_manager::sprint(
                    "\n[REAPER] Failed to queue task reaper...\n");
            }
        }
    }  // The lock is released here
}

/// Blocks the current task for at least `seconds` seconds.
///
/// The task is inserted into the global sleep delta queue with a tick countdown
/// derived from `seconds`, then immediately marked [`TaskState::Blocked`]. It
/// will be woken and re-enqueued automatically by the timer ISR once its delta
/// expires.
/// 
/// The system sleep is only granular down to 1 second because the woken tasks
/// are re-enqueued at the tail of the Ready queue and, depending on the size
/// of the queue, they may be delayed in resuming execution by many milliseconds.
/// E.g., with the current quantum of 4ms, if the queue contained 100 tasks, the
/// woken task would be late by around 0.4 or 0.5 seconds. So, the smallest
/// sleep timer allowed for granularity is 1 second, where the lateness wouldn't
/// be as noticeable. Otherwise, if we tell a task to sleep for 100 ms, and it
/// is always late by 400 ms (so it sleeps for 0.5 s), which is no good.
///
/// The task's `ticks_remaining` is reset to [`TASK_QUANTUM_TICKS`] before
/// yielding. This ensures that when the task is eventually woken and
/// rescheduled, it receives a fresh full quantum rather than whatever
/// (potentially very small) remainder it had at the time it called `sleep()`.
///
/// # Arguments
///
/// * `seconds` - The number of seconds to sleep. Passing `seconds == 0` is a
///               valid no-op sleep: the task simply yields the remainder of its
///               current quantum and is immediately re-enqueued without ever
///               entering the sleep queue.
#[allow(dead_code)]
pub fn sleep(seconds: u32) {
    let task_id = get_current_task_id();

    if seconds == 0 {
        // Zero-duration sleep: skip the queue entirely and just yield.
        // The task remains Ready and will be re-enqueued normally.
        unsafe { schedule() };
        return;
    }

    // Acquiring both locks back-to-back with IRQs disabled across the entire
    // window prevents a race condition that would make the task forever blocked
    // if triggered. This way, the ISR cannot consume the queue entry before the
    // task is marked Blocked.
    {
        // Lock the sleep queue first. This simultaneously disables IRQs via
        // the StaticIrqSpinLock, opening the atomic window we need.
        let mut sleep_guard = super::sleep_queue::SLEEP_QUEUE.lock();

        // Insert directly through the guard rather than via a helper function
        // to avoid a second acquisition of SLEEP_QUEUE.
        sleep_guard.insert(task_id, (seconds * 1000) as u64);

        // With IRQs still disabled, acquire the scheduler lock and transition
        // the task to Blocked. The ISR cannot fire between these two steps,
        // so the queue entry we just inserted is guaranteed to still be live
        // when the task's state becomes Blocked.
        let mut sched = SCHEDULER.lock();
        if let Some(task) = sched.get_task_mut(task_id) {
            // Reset the quantum so the task gets a fresh timeslice on wakeup
            task.ticks_remaining = crate::globals::TASK_QUANTUM_TICKS;

            // Mark the task Blocked so schedule() will not re-enqueue it
            task.state = TaskState::Blocked;

            // Mirror the block in the owning Thread, so any code inspecting
            // thread state sees a consistent view in the Process Manager.
            task.thread.upgrade().unwrap().lock().state =
                crate::process_manager::thread::ThreadState::Waiting;
        }
        // Both guards drop here, restoring IRQs
    }

    // Yield the CPU. Because the task is now Blocked, schedule() will not
    // push it back onto the run queue, and it will remain off the queue until
    // `tick_sleep_queue()` calls `wake()` on the task when its delta expires.
    unsafe { schedule() };
}
