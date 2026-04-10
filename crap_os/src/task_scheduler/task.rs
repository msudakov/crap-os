//! Kernel Task Representation
//!
//! This module defines the data types that represent a kernel task and the
//! instrumentation needed to construct a new task's initial stack frame, so
//! that the generic context switcher (`switcher.rs`) can resume it for the very
//! first time without any special handling.
//!
//! A task is an independent thread of kernel execution. Each task has:
//!   - A unique numeric ID (`TaskId`);
//!   - A lifecycle state (`TaskState`) visible to the scheduler;
//!   - A private stack, heap-allocated as a `Box<[u8]>`;
//!   - A saved stack pointer (`saved_rsp`) that the context switcher uses to
//!     restore the task's register state when it is next scheduled.
//!
//! All tasks run at ring 0, for now, with interrupts enabled once they start
//! executing.

use core::sync::atomic::{AtomicU64, Ordering};
use alloc::boxed::Box;

/// TaskId is an opaque, unique identifier for a kernel task.
///
/// Wraps a `u64` for copy and comparison. The only reserved value is `0`,
/// which is held by `TaskId::IDLE` for the bootstrap execution context.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct TaskId(u64);

impl TaskId {
    /// The ID of the idle task, which is the pseudo-task representing the
    /// initial kernel execution context that exists before `scheduler::init`
    /// is called. This ID is never issued by `alloc_task_id`, where the
    /// counter starts at 1.
    pub const IDLE: TaskId = TaskId(0);

    /// Unwraps the `u64` value. Mainly used for debug output and logging.
    /// 
    /// # Returns
    /// 
    /// Returns the underlying `u64` value.
    #[allow(dead_code)]
    #[inline]
    pub fn as_u64(self) -> u64 {
        self.0
    }
}

/// Global monotonically increasing counter used to mint unique `TaskId`s. It
/// starts at 1, so that 0 is always reserved for `TaskId::IDLE`.
static NEXT_TASK_ID: AtomicU64 = AtomicU64::new(1);

/// Allocates the next unique `TaskId`.
///
/// Each call atomically increments `NEXT_TASK_ID` and wraps the previous value
/// in a `TaskId`. Called exactly once per `Task::new` invocation.
/// 
/// # Returns
/// 
/// Returns the next unique `TaskId`.
#[inline]
fn alloc_task_id() -> TaskId {
    // `Relaxed` ordering is intentional and sufficient. We only need each
    // `fetch_add` to return a distinct value, and we do not need to synchronize
    // any other memory operations with respect to the counter itself. No other
    // thread observes the counter value for anything other than its uniqueness.
    TaskId(NEXT_TASK_ID.fetch_add(1, Ordering::Relaxed))
}

// The `TaskState` defined below tracks the following state transitions:
//
//   +---------+                        +---------+
//   |         |   scheduler picks it   |         |
//   |  Ready  | ---------------------> | Running |
//   |         | <--------------------- |         |
//   +---------+  preempted / yields    +----|----+
//        ^                                  | blocks on event
//        |                                  v
//        |    event arrives (wake)    +---------+
//        ---------------------------- | Blocked |
//                                     +----|----+
//                                          | (and eventually...)
//                                          v
//                                    +----------+
//                                    |   Dead   |
//                                    +----------+

/// The lifecycle state of a task as tracked by the scheduler.
#[allow(dead_code)]
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum TaskState {
    /// The task is in the run queue and eligible to be scheduled onto a CPU.
    /// A newly created task starts in this state.
    Ready,

    /// The task is currently executing on the CPU.
    /// At any given time, at most one task per CPU core is in this state.
    Running,

    /// The task is waiting for an external event.
    /// A `Blocked` task is not in the run queue. It will not be selected
    /// by the scheduler until some other task or interrupt handler calls
    /// `scheduler::wake(id)` to transition it back to `Ready`.
    Blocked,

    /// The task has finished executing (its entry function returned, or it
    /// explicitly terminated itself). The `Task` struct (and its stack
    /// allocation) remains alive until the scheduler performs a "tombstone
    /// cleanup" on the next scheduling pass, at which point the `Task` is
    /// dropped and the stack is freed.
    Dead,
}

/// Size in bytes of the per-task kernel stack.
///
/// 16 KB is a reasonable default for kernel tasks: it's large enough to
/// accommodate typical kernel call depths, and it is small enough that
/// spawning hundreds of tasks does not exhaust the kernel heap.
///
/// It must be a multiple of 16 to satisfy the System V AMD64 ABI's requirement
/// that RSP be 16-byte aligned on function entry.
pub const TASK_STACK_SIZE: usize = 16 * 1024;  // 16 KB

// =============================================================================
// Initial stack frame layout
// =============================================================================
//
// The context switcher saves and restores a task's register state by pushing
// and popping a fixed set of callee-saved GPRs onto/from the task's own stack.
// When we create a new task, we have to manually write this exact frame
// structure onto the new stack, so that the switcher can resume the new task
// identically to how it would resume any already-running task, without any
// special first-time-resume path.
//
// This is the exact frame layout we need (where addresses increase upward, and
// stack grows downward):
//
//   +----------------------------------+  <- stack_top (highest address)
//   |  (unused / guard space)          |
//   |----------------------------------|
//   |  rip  = task_entry_stub ptr      |  <- "return address" consumed by ret
//   |----------------------------------|
//   |  rbp  = 0                        |  <- popped last  (rbp)
//   |  rbx  = 0                        |
//   |  r12  = entry fn ptr             |  <- popped 4th   (r12)
//   |  r13  = arg u64                  |  <- popped 3rd   (r13)
//   |  r14  = 0                        |
//   |  r15  = 0                        |  <- popped first (r15)
//   +----------------------------------+  <- saved_rsp points here
//
// When `switch_to` resumes this task for the first time, it:
//   1. Restores RSP to `saved_rsp` (pointing at the bottom of the frame);
//   2. Pops R15, R14, R13, R12, RBX, RBP in that order;
//   3. Executes `ret`, which pops `rip` = `task_entry_stub`, and jumps there.
//
// `task_entry_stub` then moves R12 (entry fn ptr) into an appropriate
// register and R13 (arg) into RDI, and calls the actual task entry function.
//
// Since `switch_to` is called like a regular function from the scheduler, only
// these callee-saved six registers need to be on the software frame. The
// caller-saved registers (RAX, RCX, RDX, RSI, RDI, R8–R11) belong to the task
// itself and are undefined at the point of a first-time resume. This works,
// because `task_entry_stub` only uses R12 and R13 before calling the task
// function, which will establish its own register state immediately.
//
// We store entry/arg in R12/R13 rather than RDI/RSI because RDI and RSI are
// caller-saved, so they are not part of the software context frame and would
// not be preserved across `switch_to`. R12 and R13 are callee-saved, appear on
// the frame, and are correctly restored before `task_entry_stub` runs.

/// The software context frame written at the top of a new task's stack.
///
/// `repr(C, packed)` ensures the fields are laid out in declaration order with
/// no padding, matching the exact byte sequence that `switch_to` pops via
/// `pop r15; pop r14; pop r13; pop r12; pop rbx; pop rbp; ret`.
#[repr(C, packed)]
struct InitialFrame {
    /// Popped first into R15. Zero - no meaningful initial value for R15.
    r15: u64,

    /// Popped into R14. Zero.
    r14: u64,

    /// Popped into R13. Holds the `arg: u64` parameter passed to `Task::new`.
    /// `task_entry_stub` moves this into RDI before calling the entry function.
    r13: u64,

    /// Popped into R12. Holds the entry function pointer
    /// (`entry: fn(u64)`). `task_entry_stub` calls this via `call r12`.
    r12: u64,

    /// Popped into RBX. Zero.
    rbx: u64,

    /// Popped into RBP. Zero - marks this as the outermost frame to debuggers
    /// and stack unwinders (a zero RBP conventionally signals "no caller").
    rbp: u64,

    /// Consumed by `ret` as the return address, it jumps to `task_entry_stub`.
    /// This is not popped into any GPR; `ret` pops it directly into RIP.
    rip: u64,
}

/// Naked trampoline that every new task executes first when it is scheduled
/// for the very first time.
///
/// This trampoline exists because, after `switch_to` pops the `InitialFrame`
/// and executes `ret`, RSP and all callee-saved registers are in the state we
/// wrote into the frame. But, the actual task entry function expects its
/// argument in RDI (per the System V ABI), not in R13. This stub bridges that
/// gap: it moves R13 into RDI and then calls the entry function through R12.
///
/// Upon entry to this stub, R12 contains the entry function pointer
/// (`fn(u64)`), R13 has the `u64` argument, and all other caller-saved
/// registers are undefined (the new task owns them).
///
/// We use a naked function here (with no compiler-generated prologue or
/// epilogue) because we need precise control over which instructions execute
/// and in what order, as any compiler-inserted `push rbp` or frame setup would
/// corrupt the carefully constructed stack state.
///
/// # Safety
/// 
/// This function must never be called directly; it is only ever jumped to by
/// `switch_to` via the `rip` field of an `InitialFrame`. The register
/// preconditions described above must hold.
#[unsafe(naked)]
unsafe extern "C" fn task_entry_stub() {
    core::arch::naked_asm!(
        // Re-enable hardware interrupts for the new task. Interrupts are
        // disabled while `switch_to` runs (it is called from the scheduler,
        // which holds an IrqSpinLock that disables interrupts for the
        // duration). We must re-enable them before entering the task
        // body, so that the new task can receive timer interrupts, keyboard
        // interrupts, and so on.
        "sti",

        // Move the task's argument from R13 into RDI.
        // RDI is the first integer argument register in the System V AMD64 ABI.
        // After this, calling `entry(arg)` via R12 will pass `arg` correctly.
        "mov rdi, r13",

        // Call the task's entry function through R12
        "call r12",

        // Finite tasks return here and exit cleanly, while diverging
        // tasks (essentially, fn(u64) -> !) never reach this point.
        "call {exit}",

        // Defensive trap: unreachable under correct usage.
        // `ud2` raises #UD (Invalid Opcode exception, vector 6), which will
        // be caught by the IDT's #UD handler and produce a kernel panic.
        "ud2",

        exit = sym task_exit,  // Call tombstone cleanup routines
    );
}

/// A kernel task (kernel-mode cooperative/preemptive thread).
///
/// Each `Task` owns its execution stack for its entire lifetime. The stack
/// is heap-allocated so the kernel heap must be initialized before any task
/// (other than the idle task) is created.
///
/// Tasks are created via `Task::new` and managed by the scheduler in
/// `scheduler.rs`. The scheduler holds `Task` values behind an `IrqSpinLock`
/// so that task state is consistent even when modified from interrupt context.
///
/// When a `Task` is dropped (e.g., after tombstone cleanup), `Box<[u8]>` drops
/// its allocation, freeing the stack back to the kernel heap. The scheduler
/// must ensure the task is not currently running on any CPU when it is dropped.
pub struct Task {
    /// Unique, immutable identifier for this task. It is assigned at creation
    /// time and never changed.
    pub id: TaskId,

    /// Current scheduler-visible lifecycle state. It is modified by the
    /// scheduler on run-queue operations and by interrupt handlers via
    /// `scheduler::wake`.
    pub state: TaskState,

    /// Saved stack pointer: the value of RSP at the moment this task was
    /// last switched away from.
    ///
    /// Updated by the context switcher each time this task is preempted or
    /// voluntarily yields, and read back when the task is next resumed.
    ///
    /// The value is a byte offset into `_stack` (specifically, it points at
    /// the bottom of an `InitialFrame` or a live `switch_to` frame inside
    /// the stack allocation). It is stored as `u64` rather than `*mut u64` for
    /// these reasons:
    ///   - `u64` is `Send`; raw pointers are not. This allows `Task` to
    ///      implement `Send` with a single `unsafe impl` rather than requiring
    ///      a wrapper type.
    ///   - The pointer arithmetic is done entirely inside `unsafe` assembly in
    ///      the switcher, so there is no benefit to maintaining the Rust type.
    ///
    /// Invariant: between scheduler invocations (i.e., when no `switch_to`
    /// is in progress), `saved_rsp` always points to a valid frame inside
    /// `_stack` for `Ready` and `Blocked` tasks. For the currently `Running`
    /// task, `saved_rsp` may be stale (it holds the value from the previous
    /// switch-away and will be overwritten by the next `switch_to`).
    pub saved_rsp: u64,

    /// The number of system clock ticks remaining in the quantum of this task.
    /// 
    /// This starts out at max quantum ticks and gets decremented every clock
    /// tick. When it reaches 1 (not 0, because that check happens at the end
    /// of the task's tick period), the task is scheduled for preemption.
    /// 
    /// TODO: migrate this to Process Control structure when per-CPU storage is
    /// implemented. On SMP, this should live on the running core, not the task,
    /// to avoid needing the scheduler lock in the timer ISR hot path.
    pub ticks_remaining: u32,

    /// The total number of system clock ticks this task has consumed; used for
    /// accounting purposes.
    pub ticks_executed: AtomicU64,

    /// Heap-allocated stack storage.
    ///
    /// `Box<[u8]>` is used rather than `Box<[u8; TASK_STACK_SIZE]>` because:
    ///   - A fixed-size array box would require the size to be known at the
    ///     call site; a slice box allows the size to be a runtime value.
    ///   - The fat pointer (`ptr` + `len`) means we always know the stack's
    ///     base address and length without storing them separately.
    ///
    /// The field is prefixed `_` to signal that it is never accessed by name
    /// after construction. Its sole purpose is to own the allocation, so that
    /// it is freed (via `Drop`) when the `Task` is dropped.
    _stack: Box<[u8]>,
}

// SAFETY: `Task` contains `saved_rsp: u64` which is semantically a pointer
// into `_stack`. Rust does not know this, so it cannot verify that the pointer
// is only accessed from one thread at a time. We assert `Send` manually, so
// that:
//   - The scheduler holds all `Task` values behind a single `IrqSpinLock`,
//     so at most one CPU core can access a given `Task` at any time.
//   - `switch_to` is the only code that reads/writes `saved_rsp`, and it is
//     always called with the scheduler lock held ( and interrupts disabled).
unsafe impl Send for Task {}

impl Task {
    /// Constructs the idle task that represents the kernel's initial execution
    /// context (the thread that runs the `_start` routine). Unlike `Task::new`,
    /// this does not allocate a real stack. The idle task's "stack" is the
    /// kernel's own boot stack.
    pub(crate) fn new_idle() -> Self {
        // A minimal placeholder allocation so `_stack` is never a null Box.
        // The idle task's real stack is the kernel's higher-half boot stack,
        // which the bootloader set up and which persists for the kernel's
        // lifetime. This allocation is never executed on.
        let stack: Box<[u8]> = {
            let mut v = alloc::vec::Vec::with_capacity(16);
            v.resize(16, 0u8);
            v.into_boxed_slice()
        };

        // `saved_rsp` starts at 0. The first call to `switch_to` that switches
        // away from the idle task will overwrite `saved_rsp` with the real RSP
        // value at that moment, making the idle task resumable. The state is
        // initialized to `Running` because this task is the currently
        // executing context at the time `new_idle` is called.
        Task {
            id:              TaskId::IDLE,
            state:           TaskState::Running,
            saved_rsp:       0,
            ticks_remaining: crate::globals::TASK_QUANTUM_TICKS,
            ticks_executed:  AtomicU64::new(0),
            _stack:          stack,
        }
    }

    /// Creates a new kernel task that will call `entry(arg)` when first
    /// scheduled.
    ///
    /// Allocates a `TASK_STACK_SIZE`-byte stack from the kernel heap, writes
    /// an `InitialFrame` at the top of the stack, and sets `saved_rsp` to point
    /// at the bottom of that frame so the switcher can resume the task
    /// correctly. The new task starts in `TaskState::Ready` and will not run
    /// until the scheduler places it on the run queue and eventually calls
    /// `switch_to`.
    ///
    /// # Arguments
    /// 
    /// * `entry` - The task's entry function.
    /// * `arg`   - An opaque `u64` passed as the sole argument to `entry`. We
    ///   can use it as a type-erased pointer, an integer, or ignore it if the
    ///   task needs no parameter.
    ///
    /// # Panics
    /// 
    /// Panics if the kernel heap cannot satisfy the stack allocation. This may
    /// happen if `TASK_STACK_SIZE` bytes are unavailable, and the heap cannot
    /// grow.
    pub fn new(entry: fn(u64), arg: u64) -> Self {
        // This allocates the stack for the new task.
        // We use Vec::with_capacity + resize + into_boxed_slice rather than
        // `vec![0u8; TASK_STACK_SIZE]` because the latter may not zero-
        // initialize in all Rust versions/configurations, while resize(N, 0u8)
        // always does. Zero-initializing is important here, as it ensures no
        // stale heap data is visible to the new task through uninitialized
        // stack reads.
        let mut stack: Box<[u8]> = {
            let mut v = alloc::vec::Vec::with_capacity(TASK_STACK_SIZE);
            v.resize(TASK_STACK_SIZE, 0u8);
            v.into_boxed_slice()
        };

        // Compute the frame pointer. We place the `InitialFrame` just below
        // the top of the stack, aligned to 16 bytes, so that after `switch_to`
        // pops the frame and executes `ret`, RSP lands on a 16-byte aligned
        // address as required by the ABI. The `& !0xF` mask clears the low 4
        // bits, rounding down to 16 bytes.
        let stack_top = stack.as_mut_ptr() as usize + TASK_STACK_SIZE;
        let frame_ptr = ((stack_top - core::mem::size_of::<InitialFrame>())
            & !0xF) as *mut InitialFrame;

        // Write the `InitialFrame`.
        // For R15, R14, RBX: no meaningful initial value; zero is conventional.
        // R13 is the task argument; `task_entry_stub` moves this into RDI.
        // R12 is the entry function pointer; `task_entry_stub` calls this.
        // RBP of zero marks this as the outermost frame for stack unwinders.
        // RIP is address of the trampoline; `switch_to`'s `ret` jumps here.
        //
        // SAFETY:
        //   - `frame_ptr` was computed from `stack.as_mut_ptr()`, an address
        //     within the live `stack` allocation.
        //   - The subtraction and alignment ensure `frame_ptr` is at least
        //     `size_of::<InitialFrame>()` bytes below `stack_top`, so the
        //     entire write falls within the allocation bounds.
        //   - `stack` is zero-initialized, so there are no pre-existing invalid
        //     values at `frame_ptr` that we would be reading through later.
        unsafe {
            frame_ptr.write(InitialFrame {
                r15: 0,
                r14: 0,
                r13: arg,
                r12: entry as u64,
                rbx: 0,
                rbp: 0,
                rip: task_entry_stub as unsafe extern "C" fn() as u64,
            });
        }

        // `saved_rsp` gets the address of the bottom of the frame (where R15
        // lives), which is the value `switch_to` will load into RSP when
        // resuming this task.
        let saved_rsp = frame_ptr as u64;

        Task {
            id:              alloc_task_id(),
            state:           TaskState::Ready,
            saved_rsp,
            ticks_remaining: crate::globals::TASK_QUANTUM_TICKS,
            ticks_executed:  AtomicU64::new(0),
            _stack:          stack,
        }
    }
}

/// Handles a task's normal return and exit.
/// 
/// This gets called automatically by `task_entry_stub` when a task's entry
/// function returns normally. It marks the current task as `Dead` and
/// immediately yields to the scheduler. The actual stack and task table
/// cleanup (tombstone cleanup) is deferred to the `dead_task_reaper`
/// `SystemTask`, which runs at the next timer tick. This function never
/// returns.
pub fn task_exit() -> ! {
    // Mark this task as `Dead`. We do this in a block, so the scheduler lock
    // is dropped before we call `schedule()` (which will acquire it again
    // internally), and we must not hold it across that call.
    {
        let mut scheduler = super::scheduler::SCHEDULER.lock();
        let this_id = scheduler.current;
        if let Some(task) = scheduler.get_task_mut(this_id) {
            task.state = TaskState::Dead;
        }
        // The lock is dropped here
    }

    // Enqueue the `dead_task_reaper` `SystemTask`, so tombstone cleanup runs
    // on the next timer tick. The reaper will find this task's slot marked
    // `Dead` and free it.
    crate::system_core::queue_system_task(
        crate::system_core::system_tasks::dead_task_reaper, 0);

    // With the status set to `Dead`, we now hand off to the next ready task.
    // This task will never be rescheduled, because `Dead` tasks are not
    // re-queued. The reaper will free this task's stack at the next timer tick,
    // by which point we are no longer running on it.
    unsafe { super::scheduler::schedule() };

    // Truly unreachable, as `schedule()` switches the stack away and never
    // returns to a `Dead` task. If we somehow land here, fault loudly.
    unreachable!("task_exit: schedule() returned to a dead task");
}
