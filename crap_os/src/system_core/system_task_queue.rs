//! System Task Queue
//!
//! This module provides the kernel's deferred work mechanism: a fixed-size,
//! statically allocated ring buffer of [`SystemTask`] entries that are drained
//! at the start of every timer interrupt, after EOI and before the normal task
//! scheduler runs.
//!
//! This sits between hardware IRQ context and normal scheduled tasks in the
//! kernel's priority hierarchy. SystemTasks run with interrupts disabled,
//! are never preempted by normal tasks, and must always run to completion
//! without blocking or yielding.

use crate::spinlock::StaticIrqSpinLock;

/// Maximum number of [`SystemTask`] entries that can be queued at once.
/// Since SystemTasks are short-lived and drained every timer tick, 32 slots is
/// enough for now. Normally, only a handful will ever be live simultaneously.
/// If this limit is hit, this constant should be reconsidered.
const MAX_SYSTEM_TASKS: usize = 32;

/// A single unit of deferred kernel work.
/// 
/// A `SystemTask` is a function pointer and its argument, enqueued for
/// execution at the next timer interrupt. It is is a short, bounded,
/// non-blocking unit of kernel work that runs at elevated priority, after
/// hardware IRQs, but before any normal scheduled task gets the CPU.
/// SystemTasks are drained at the start of every timer interrupt, with
/// interrupts disabled, before `schedule()` is called.
///
/// Rules for `SystemTask` functions:
///   - Must run to completion quickly (no spinning and no sleeping);
///   - Must be as efficient as possible;
///   - Must not block or yield;
///   - Must acquire and release any spinlocks within their own body, and never
///     hold a lock across a return;
///   - Must not call `schedule()` themselves.
pub struct SystemTask {
    /// The function to execute. Takes a single u64 argument for flexibility:
    /// can carry a task ID, a pointer cast to u64, a status code, etc.
    function: fn(u64),

    /// The argument forwarded to `function` at call time. Interpretation is
    /// entirely up to the function; the queue itself treats it as opaque.
    func_arg:  u64,
}

/// The fixed-size ring buffer that backs the `SystemTask` queue.
///
/// Entries are pushed at `tail` and popped from `head`. Both indices advance
/// modulo [`MAX_SYSTEM_TASKS`], wrapping around to zero when they reach the
/// end of the array. This gives O(1) push and pop with no allocation and no
/// data movement.
struct SystemTaskQueue {
    /// The backing store. Each slot is `None` when empty and `Some(SystemTask)`
    /// when occupied. Slots are set back to `None` on pop so no stale function
    /// pointers linger in memory after a task has been consumed.
    system_tasks: [Option<SystemTask>; MAX_SYSTEM_TASKS],

    /// Index of the next slot to read from (pop). Advances by one, modulo
    /// [`MAX_SYSTEM_TASKS`], each time an entry is consumed.
    head: usize,

    /// Index of the next slot to write to (push). Advances by one, modulo
    /// [`MAX_SYSTEM_TASKS`], each time an entry is enqueued.
    tail: usize,

    /// Number of live entries currently in the ring buffer.
    ///
    /// Tracking this separately avoids the classic ring-buffer ambiguity where
    /// `head == tail` could mean either the queue is empty or it is full.
    /// With this tracker:
    ///  - `queue_len == 0`          -> queue is empty (even if `head == tail`)
    ///  - `queue_len == QUEUE_SIZE` -> queue is full  (even if `head == tail`)
    queue_len: usize,
}

impl SystemTaskQueue {
    /// Constructs an empty `SystemTaskQueue`.
    /// 
    /// All slots are `None`, and `head`, `tail`, and `queue_len` are all zero.
    /// This is a `const fn`, so it can be used to initialize the `static`
    /// instance at compile time with no runtime cost.
    const fn new() -> Self {
        Self {
            // The `const { None }` block is required to initialize a non-Copy
            // array of Options in a const context without needing a Default
            // implementation on `SystemTask`.
            system_tasks: [const { None }; MAX_SYSTEM_TASKS],
            head: 0,
            tail: 0,
            queue_len:  0,
        }
    }

    /// Pushes a `SystemTask` onto the back of the ring buffer, advances `tail`
    /// by one modulo `MAX_SYSTEM_TASKS`, and increments `queue_len`.
    /// 
    /// # Arguments
    /// 
    /// * `task` - New `SystemTask` to enqueue.
    /// 
    /// # Returns
    /// 
    /// Returns `Err` with a static message if the buffer is full, so the caller
    /// can log and continue rather than panicking, since this may be called
    /// from IRQ context where panicking is not safe.
    fn push(&mut self, task: SystemTask) -> Result<(), &'static str> {
        if self.queue_len == MAX_SYSTEM_TASKS {
            return Err("[SYSTEM_TASK] Queue full, dropping SystemTask");
        }

        self.system_tasks[self.tail] = Some(task);
        self.tail = (self.tail + 1) % MAX_SYSTEM_TASKS;  // Wrap tail around
        self.queue_len += 1;
        Ok(())
    }

    /// Pops a `SystemTask` from the front of the ring buffer, advances `head`
    /// by one modulo `MAX_SYSTEM_TASKS`, and decrements `queue_len`.
    /// 
    /// # Returns
    /// 
    /// Returns the popped off `SystemTask` if successful, or `None` if the
    /// ring buffer is empty.
    fn pop(&mut self) -> Option<SystemTask> {
        if self.queue_len == 0 {
            return None;
        }

        // `take()` moves the value out and leaves None in the slot, which is
        // important for keeping the backing array clean, as we don't want a
        // consumed task's function pointer sitting in memory indefinitely.
        let task = self.system_tasks[self.head].take();
        self.head = (self.head + 1) % MAX_SYSTEM_TASKS;
        self.queue_len -= 1;
        task
    }
}

/// The single global SystemTask queue, protected by an IRQ-safe spinlock, so it
/// can be enqueued from both normal task context and ISR context safely.
static SYSTEM_TASK_QUEUE: StaticIrqSpinLock<SystemTaskQueue> =
    StaticIrqSpinLock::new(SystemTaskQueue::new());

/// Enqueue a `SystemTask` for execution at the next timer tick.
///
/// Safe to call from any context: normal kernel tasks, hardware ISRs, or
/// exception handlers. The `StaticIrqSpinLock` ensures atomic access across
/// all call sites. If the queue is full, a warning is printed to serial, and
/// the task is silently dropped. This should never occur during normal
/// operation; if it does, `MAX_SYSTEM_TASKS` should be reconsidered.
/// 
/// # Arguments
/// 
/// * `function` - Target `SystemTask` function to enqueue.
/// * `func_arg` - The u64 opaque function argument to pass to the function.
pub fn queue_system_task(function: fn(u64), func_arg: u64) {
    let mut queue = SYSTEM_TASK_QUEUE.lock();
    if let Err(error_message) = queue.push(SystemTask { function, func_arg }) {
        // We can't panic here, as we may be in IRQ context. Log and move on.
        crate::hardware_manager::sprint(error_message);
        crate::hardware_manager::sprint("\n");
    }
}

/// Drain all currently queued `SystemTask`s in FIFO order.
///
/// Called exclusively by the timer IRQ handler (`handler_apic_timer`), after
/// EOI and before `schedule()`. At this point, interrupts are already disabled,
/// as we are inside an IRQ handler, so no new tasks can be enqueued by an ISR
/// mid-drain. However, a `SystemTask` itself may enqueue additional
/// SystemTasks internally without an issue.
pub fn drain_system_tasks() {
    loop {
        // Pop one task at a time, releasing the lock between each execution.
        // This keeps the queue lock held for the absolute minimum time and
        // allows a `SystemTask` to safely enqueue further SystemTasks without
        // deadlocking on the queue lock itself.
        let system_task = {
            let mut queue = SYSTEM_TASK_QUEUE.lock();
            queue.pop()
        };  // The lock is released here, before the function is called

        match system_task {
            Some(task) => (task.function)(task.func_arg),  // Invoke function
            None => break,  // Queue is empty; drain is complete.
        }
    }
}
