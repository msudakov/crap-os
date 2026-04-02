// =============================================================================
// Kernel Task Scheduler
// =============================================================================
//
// This module implements the kernel's task scheduler. It uses preemptive
// scheduling, maintains a table of all live tasks and a round-robin ready
// queue, and performs context switches in response to timer interrupts.
//
// The task table (`tasks: [Option<Task>; MAX_TASKS]`) is a flat array of
// optional `Task` values; a `Some` slot holds a live task, regardless of its
// state, while a `None` slot is free. The ready queue is a power-of-two ring
// buffer (`queue: [TaskId; QUEUE_SIZE]`) of `TaskId` values representing tasks
// in the `Ready` state.
//
// The scheduling algorithm is a first-in-first-out round robin: the task at the
// head of the queue gets the next time slice. After the slice, if the task is
// still `Running`, it is moved to `Ready` and appended to the tail, giving all
// other ready tasks a turn before it runs again. This is the simplest fair
// scheduling algorithm with O(1) enqueue and dequeue operations. There is no
// priority system yet; tasks preempted by the timer, tasks woken from being
// `Blocked`, newly-spawned tasks, etc., are all treated equally.
//
// All mutable scheduler state is protected by a single `StaticIrqSpinLock`.
// Because it is an IrqSpinLock, acquiring it also disables hardware interrupts
// on the current CPU for the duration of the critical section. This prevents
// the timer ISR from re-entering `schedule()` while we are in the middle of
// modifying the task table or queue.
//
// CRITICAL RULE: The lock must NEVER be held across a call to `switch_to`
// because of:
//   - Deadlock danger: the incoming task will try to acquire the same lock on
//     its next scheduler interaction; if we hold it during the switch, it can
//     never acquire it.
//   - Wrong stack danger: `IrqSpinLockGuard::drop` restores RFLAGS (re-enabling
//     interrupts) by executing `popfq`. If the guard is dropped after the
//     switch, `popfq` runs on the incoming task's stack with the outgoing
//     task's saved flags, thus corrupting the incoming task's interrupt state.

use super::switcher::switch_to;
use super::task::{Task, TaskId, TaskState};
use crate::spinlock::StaticIrqSpinLock;

/// Maximum number of simultaneously live tasks (both ready and blocked).
///
/// Each occupied slot in the task table costs one `Option<Task>` (~48 bytes of
/// BSS), plus the task's 16 KB heap-allocated stack. With 256 tasks, it gives:
///   - 256 * 48 = ~12 KB of BSS;
///   - Up to 256 * 16 KB = 4 MB of heap, which is only actually allocated
///     when a task is spawned, not all at initialization time.
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

/// All mutable scheduler state, combined into a single struct, so it can be
/// protected by a single `StaticIrqSpinLock`.
///
/// The following must hold true:
///   - Every task in `queue` has state `Ready` and exists in `tasks`.
///   - Exactly one task has state `Running` at any time; its ID is `current`.
///   - A task that is `Blocked` or `Dead` is not in `queue`.
///   - A `Dead` task remains in `tasks` until `schedule()` performs tombstone
///     cleanup (drops it and frees its stack).
///   - `queue_len` always equals the number of IDs between `head` and `tail`
///     in the ring. Specifically: `queue_len == 0` when the queue is empty;
///     and, `queue_len == QUEUE_SIZE` when the queue is full.
///
/// The scheduler's spinlock is always acquired through `SCHEDULER.lock()`, and
/// it must never held across `switch_to`. We define it as `pub(super)` because
/// it needs to be accessible from `task_exit` in `task.rs`.
pub(super) struct Scheduler {
    /// Flat array of optional tasks, where each index is a storage slot with no
    /// semantic meaning; tasks are located by scanning for a matching `TaskId`,
    /// not by indexing directly.
    ///
    /// Slot occupancy:
    ///   `None`     - free slot, available for a new task.
    ///   `Some(t)`  - live task `t`; may be in any `TaskState`.
    tasks: [Option<Task>; MAX_TASKS],

    /// Round-robin ready queue, implemented as a power-of-two ring buffer.
    ///
    /// Contains the `TaskId`s of all tasks currently in `TaskState::Ready`,
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
    ///
    /// # `const fn` and non-`Copy` arrays
    /// 
    /// `Option<Task>` is not `Copy` (because `Task` contains a `Box`), so we
    /// cannot write `[None; MAX_TASKS]` in a `const` context with a runtime
    /// value. The `[const { None }; MAX_TASKS]` syntax evaluates `None` as a
    /// constant expression per-element, which is allowed because `Option<Task>`
    /// has a valid all-zeros `None` representation at compile time.
    /// `[TaskId::IDLE; QUEUE_SIZE]` works because `TaskId` is `Copy`.
    const fn new() -> Self {
        Self {
            // Each slot is independently initialized to `None`.
            tasks: [const { None }; MAX_TASKS],
            // Fill the queue with `IDLE` as a safe placeholder value.
            // Entries are only meaningful between `head` and `head+queue_len`.
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
    /// caller must handle. But, the task ID is not lost silently, as the task
    /// remains in `tasks` and can be re-enqueued when space becomes
    /// available (e.g., after a dead task is cleaned up).
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
    /// In that case, the caller should keep the current task running rather
    /// than attempting a context switch.
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

    /// Inserts a `Task` into the first available `None` slot in `tasks`.
    /// 
    /// # Arguments
    /// 
    /// * `task` - The `Task` to be inserted.
    /// 
    /// # Returns
    ///
    /// Returns the slot index on success, or `Err(TaskTableFull)` if every
    /// slot is already occupied. The slot index is not meaningful outside of
    /// this function, as tasks are later found by ID scan instead of an index.
    /// 
    /// This runs in time O(MAX_TASKS) to scan the table linearly for a free
    /// slot. It's acceptable here, because spawning is not on the hot path.
    fn insert_task(&mut self, task: Task) -> Result<usize, SchedulerError> {
        for (i, slot) in self.tasks.iter_mut().enumerate() {
            if slot.is_none() {
                *slot = Some(task);
                return Ok(i);
            }
        }
        Err(SchedulerError::TaskTableFull)
    }

    /// Obtains a shared reference to a `Task` by its ID.
    /// 
    /// The linear scan runs in time O(MAX_TASKS).
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
        self.tasks.iter()
            .filter_map(|slot| slot.as_ref())  // Skip the `None` slots
            .find(|task| task.id == task_id)
    }

    /// Obtains a mutable reference to a `Task` by its ID.
    /// 
    /// The linear scan runs in time O(MAX_TASKS). We declare it as `pub(super)`
    /// because this function needs to be accessible from `task_exit` in
    /// `task.rs`.
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
        self.tasks.iter_mut()
            .filter_map(|slot| slot.as_mut())  // Skip the `None` slots
            .find(|task| task.id == task_id)
    }

    /// Removes a `Task` by its ID from the table, dropping it and freeing its
    /// heap-allocated stack. The linear scan runs in time O(MAX_TASKS). It is
    /// a no-op if no task with the ID exists.
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
    fn remove_task(&mut self, task_id: TaskId) {
        for slot in self.tasks.iter_mut() {
            if slot.as_ref().map_or(false, |task| task.id == task_id) {
                *slot = None;  // `Task::drop` runs here, freeing the stack
                return;
            }
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

/// Initializes the scheduler and registers the current execution context as
/// the "idle task" (`TaskId::IDLE`).
///
/// The idle task is not a separately-allocated thread, but the kernel's own
/// `_start` routine that is already running on the higher-half stack. We
/// register it as `TaskId::IDLE`, so that the scheduler can:
///   - Save its RSP the first time the timer ISR preempts it;
///   - Resume it later when needed by restoring that RSP.
/// 
/// Call this exactly once from the `_start` routine, before enabling
/// interrupts, but after:
///   - The kernel heap is initialized, as task spawning needs heap allocation;
///   - The GDT and IDT are loaded.
pub fn init() {
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
    let idle_task = Task::new_idle();

    // `current` is set to IDLE here because the idle task is currently running
    scheduler.current = TaskId::IDLE;

    // Insert the idle task into the table. The idle task is not enqueued in
    // the ready queue; the scheduler falls back to it implicitly when the queue
    // is empty.
    scheduler.insert_task(idle_task)
        .expect("[SCHEDULER] Failed to insert idle task: task table full");
}

/// Creates a new kernel task that will execute `entry(arg)` and enqueues it
/// on the ready queue.
///
/// Allocates a `TASK_STACK_SIZE` bytes of stack from the kernel heap,
/// writes the initial context frame, and makes the task immediately eligible
/// for scheduling. The task will not actually begin executing until the timer
/// ISR next calls `schedule()` and selects it from the head of the queue.
///
/// # Arguments
/// 
/// * `entry` - The task's entry function.
/// * `arg`   - An opaque `u64` passed as the sole argument to `entry` function.
///
/// # Returns
/// 
/// Returns the `TaskId` of the newly-spawned task on success, or a
/// `SchedulerError` if either the task table or the ready queue is full.
///
/// # Panics
/// 
/// Panics if the kernel heap cannot satisfy the stack allocation.
///
/// # Locking
/// 
/// `Task::new` (the heap allocation) is done before acquiring the scheduler
/// lock, minimizing the time the lock is held. The lock is only held for the
/// brief table-insertion and queue-push operations.
pub fn spawn(entry: fn(u64), arg: u64) -> Result<TaskId, SchedulerError> {
    // Allocate the task and its stack outside the lock. Heap allocation may
    // block briefly (if the heap needs to grow) or involve page-mapping
    // operations. Doing this with the scheduler lock held would block the
    // timer ISR and any other task trying to call `wake()` or `spawn()` for
    // the duration, which would unnecessarily increase interrupt latency.
    let task = Task::new(entry, arg);
    let task_id = task.id;

    {
        let mut scheduler = SCHEDULER.lock();  // Acquire lock here

        // Insert the task into the table first; if the table is full, we want
        // to return an error before modifying the queue.
        scheduler.insert_task(task)?;

        // Enqueue the task ID in the ready queue. If the queue is full, the
        // task remains in the table as `Ready`, but is unreachable by the
        // scheduler until a slot frees up. In practice, this should not happen
        // if QUEUE_SIZE == MAX_TASKS.
        scheduler.queue_push(task_id)?;

        // Explicitly confirm Ready state (`Task::new()` already sets this, but
        // being explicit makes the invariant clear at the call site).
        if let Some(task) = scheduler.get_task_mut(task_id) {
            task.state = TaskState::Ready;
        }
    }  // Lock released here

    Ok(task_id)
}

/// Performs a preemptive context switch, selecting the next `Ready` task from
/// the head of the round-robin queue.
///
/// This is the core of the scheduler. It is called at the end of every timer
/// interrupt tick. The sequence is:
///   1. Lock the scheduler state.
///   2. Pop the next `TaskId` from the ready queue. If the queue is empty,
///      return immediately (keep running the current task).
///   3. Re-enqueue the outgoing task at the tail if it is still `Running`.
///      (If it is `Blocked` or `Dead`, do not re-enqueue.)
///   4. Transition the incoming task to `Running` state, and update `current`.
///   5. Extract the raw RSP pointer/value needed for `switch_to`.
///   6. Release the lock before calling `switch_to`; this is critical.
///   7. Call `switch_to(old_rsp_ptr, new_rsp)`.
///
/// From the outgoing task's perspective, step 7 is a function call that simply
/// takes a long time to return. It returns the next time that task is selected
/// by the scheduler and step 7 executes again with it as the incoming task.
///
/// # Safety
/// 
/// Must only be called from the timer ISR, with interrupts already disabled
/// by the ISR entry. The `IrqSpinLock` keeps them disabled throughout the
/// critical section. `switch_to` itself runs without any lock held.
pub unsafe fn schedule() {
    // Step 1: Critical section - read and update scheduler state
    //--------------------------------------------------------------------------
    // We compute everything `switch_to` needs while the lock is held, extract
    // raw values (pointer + integer), then release the lock before the switch.
    // We have to use raw values instead of references because a reference into
    // `scheduler` would borrow the `MutexGuard`, keeping the lock alive. We
    // need the lock released before `switch_to`, so we must copy the data out.
    let (old_rsp_ptr, new_rsp) = {
        let mut scheduler = SCHEDULER.lock();

        // Step 2: Dequeue the next ready task
        //----------------------------------------------------------------------
        // `queue_pop` returns `None` if no tasks are in the ready queue,
        // meaning every other task is either `Blocked`, `Dead`, or there is
        // only the idle task. In that case, we keep running the current task.
        let next_id = match scheduler.queue_pop() {
            Some(id) => id,
            None => return,  // Nothing to switch to; stay on the current task
        };

        // Step 3: Handle the outgoing task
        //----------------------------------------------------------------------
        // If the outgoing task is still `Running` (i.e., it has not blocked
        // or terminated itself during this time slice), move it to `Ready`
        // and push it to the tail of the queue, so it will run again after all
        // currently-queued tasks have had their turn. If the task is `Blocked`,
        // it will be re-enqueued by `wake()` when  the awaited event fires. If
        // the task is `Dead`, tombstone cleanup will drop it on a future pass.
        let outgoing_id = scheduler.current;
        if let Some(outgoing) = scheduler.get_task_mut(outgoing_id) {
            if outgoing.state == TaskState::Running {
                outgoing.state = TaskState::Ready;
                // If the push fails (queue full), we silently drop the outgoing
                // task from the ready set rather than panic inside an ISR. The
                // task remains in the task table with state `Ready` and can be
                // re-enqueued the next time `wake()` or another scheduling pass
                // processes it. This is a last-resort defence; it should not
                // occur under normal operation if QUEUE_SIZE == MAX_TASKS.
                let _ = scheduler.queue_push(outgoing_id);
            }
            // Blocked or Dead: do not re-enqueue. The task will be woken or
            // cleaned up through separate mechanisms (`wake`, `remove_task`).
        }

        // Step 4: Transition the incoming task to Running
        if let Some(next_task) = scheduler.get_task_mut(next_id) {
            next_task.state = TaskState::Running;
        }
        scheduler.current = next_id;

        // Step 5: Extract raw RSP data for calling `switch_to`
        //----------------------------------------------------------------------
        // `switch_to` needs:
        //   old_rsp_ptr - a *mut u64 pointing at outgoing_task.saved_rsp,
        //                 so it can write the outgoing RSP there.
        //   new_rsp     - the incoming_task.saved_rsp value to restore.
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

        let new_rsp = scheduler
            .get_task(next_id)
            .map(|task| task.saved_rsp)
            .unwrap_or(0);

        (old_rsp_ptr, new_rsp)

        // Step 6: The lock is released here
        //----------------------------------------------------------------------
        // The guard's `drop` calls `restore_interrupts(saved_flags)`. Since
        // we are inside the timer ISR, interrupts were already disabled on
        // entry; `saved_flags` has IF=0, so they remain disabled after the
        // guard drops. We must not re-enable interrupts before `switch_to`
        // completes.
    };

    // Under normal operation, these can never be null/zero. Just a sanity check
    // to avoid a null pointer dereference inside `switch_to` if something
    // has gone wrong, like a race during early boot, or a corrupted task table.
    if old_rsp_ptr.is_null() || new_rsp == 0 {
        return;
    }

    // Step 7: Perform the context switch
    //--------------------------------------------------------------------------
    // No lock is held at this point. The assembly code in `switch_to` will:
    //   1. Push callee-saved registers onto the outgoing task's stack;
    //   2. Write RSP into the outgoing task's `saved_rsp`;
    //   3. Load RSP from the incoming task's stack;
    //   4. Pop callee-saved registers from the incoming task's stack;
    //   5. Execute `ret`, jumping into the incoming task's execution context.
    //
    // From the outgoing task's perspective, this function call "returns" the
    // next time it is selected as the incoming task in a future `schedule()`
    // invocation. From the incoming task's perspective it either:
    //   - Returns from a prior `switch_to` call (previously-run task), or
    //   - Enters `task_entry_stub` for the first time (brand-new task).
    unsafe { switch_to(old_rsp_ptr, new_rsp) };
}

/// Blocks the calling task and immediately yields the CPU to the next ready
/// task.
///
/// Transitions the current task to `TaskState::Blocked`, removes it from the
/// ready queue, then calls `schedule()` as if a timer tick had just fired.
/// Because the task's state is `Blocked`, `schedule()` will not re-enqueue it,
/// and the task will remain suspended until `wake(id)` is called for it.
///
/// The calling task resumes from this function when another task or ISR calls
/// `wake(id)`, and the scheduler eventually selects it from the ready queue.
/// This must be called from task context; it must not be called from an ISR, as
/// the calling task must have a valid `saved_rsp` that `schedule()` can save
/// the outgoing RSP into.
pub fn yield_blocked() {
    // Transition the current task to `Blocked` state under the lock
    let current_id = {
        let mut scheduler = SCHEDULER.lock();
        let this_id = scheduler.current;
        if let Some(task) = scheduler.get_task_mut(this_id) {
            task.state = TaskState::Blocked;
            // No queue removal is needed because the task was `Running`, so it
            // was not in the ready queue to begin with. `schedule()` will not
            // re-enqueue it because its state will not be `Running`.
        }
        this_id
        // Lock is released here
    };
    let _ = current_id;  // ID recorded above; suppress unused-variable warning.

    // Force an immediate reschedule. Calling `schedule()` directly here (rather
    // than waiting for the next timer tick) gives up the remainder of this
    // task's time slice immediately, which is the expected semantics for a
    // voluntary block.
    unsafe { schedule() };
}

/// Transitions a `Blocked` task back to `Ready` and enqueues it for scheduling.
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
        task.state = TaskState::Ready;
        // Enqueue the now-ready task at the tail of the round-robin queue.
        scheduler.queue_push(task_id)?;
    }

    Ok(())
}

/// Looks up the ID of the task currently executing on the CPU.
/// 
/// The returned ID is a snapshot; by the time the caller inspects it, a context
/// switch may have occurred and a different task may be running.
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
