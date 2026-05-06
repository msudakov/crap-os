//! Task Representation Module
//!
//! This module defines the data types that represent a task and the
//! instrumentation needed to construct a new task's initial stack frame, so
//! that the generic context switcher (`switcher.rs`) can resume it for the very
//! first time without any special handling.
//!
//! A task is an independent unit of execution. Each task has:
//!   - An ID struct (`TaskId`), which is explained below;
//!   - A lifecycle state (`TaskState`) visible to the scheduler;
//!   - A private stack, heap-allocated as a `Box<[u8]>`;
//!   - A saved stack pointer (`saved_rsp`) that the context switcher uses to
//!     restore the task's register state when it is next scheduled;
//!   - A `Thread` object as a `Weak` back-reference that ties this execution
//!     unit to a `Process` object in the Process Manager.
//!
//! All tasks start executing with interrupts enabled.

use core::sync::atomic::AtomicU64;
use alloc::sync::Weak;
use alloc::boxed::Box;
use alloc::vec::Vec;
use crate::spinlock::IrqSpinLock;
use crate::task_scheduler::queue_task_reaper;
use crate::process_manager::thread::{Thread, ThreadState};
use crate::memory_manager::AddressSpace;
use crate::gdt::{USER_CS_RPL3, USER_DS_RPL3};

/// Uniquely identifies a task within the scheduler's task table.
///
/// A `TaskId` is a combination of a slot index and a generation counter,
/// which together allow O(1) lookup while guarding against stale references
/// to recycled slots (ABA problem).
///
/// A `TaskId` is considered valid if and only if:
///   - `tasks[slot_index].slot_generation == slot_generation`
///   - `tasks[slot_index].task.is_some()`
///
/// If either condition is false, the slot has been recycled since this
/// `TaskId` was issued, and the reference is stale. This plays a part in
/// disregarding stale task IDs that are in the scheduler's Ready queue after
/// a task has been killed.
#[derive(Clone, Copy, PartialEq, Eq)]
pub struct TaskId {
    /// Index into the scheduler's `tasks: [TaskSlot; MAX_TASKS]` array.
    /// Provides direct O(1) access without any search or hashing. Slot 0 is
    /// permanently reserved for the idle task and is never reused.
    pub slot_index: usize,

    /// Generation counter for the slot at the time this `TaskId` was issued.
    /// Incremented (with wrapping) each time a task is removed from this slot,
    /// invalidating all previously issued `TaskId`s that referenced it.
    /// Since slot 0 is never freed, its generation stays at 0 permanently.
    /// For slots 1..MAX_TASKS, wrapping after 255 reuses is considered safe
    /// in practice; the probability of having 255 task lifetimes go through a
    /// single slot while some references to it are still in the Ready queue is
    /// negligible.
    pub slot_generation: u8,
}

impl TaskId {
    /// The `TaskId` of the idle task.
    ///
    /// The idle task occupies slot 0 with generation 0 permanently. It
    /// represents the initial kernel execution context and is created once
    /// during scheduler initialization. Its slot is never freed, so no other
    /// task will ever be assigned `slot_index: 0`, and the generation on
    /// slot 0 will never be incremented.
    pub const IDLE: TaskId = TaskId {
        slot_index: 0,
        slot_generation: 0,
    };

    /// A sentinel `TaskId` representing a task that has been created but not
    /// yet inserted into the scheduler's task table.
    ///
    /// `Task::new` cannot know its slot index at construction time, as the slot
    /// is assigned by `insert_and_queue_task` during insertion. Until then,
    /// the task's `id` field holds this sentinel to make the uninitialized
    /// state explicit and distinguishable from any real task, including idle.
    ///
    /// `slot_index: usize::MAX` is used as the sentinel because it is not a
    /// valid index into `tasks[MAX_TASKS]`, so any accidental lookup will
    /// safely return `None` rather than aliasing a real slot.
    pub const PENDING: TaskId = TaskId {
        slot_index: usize::MAX,
        slot_generation: 0,
    };
}

// The `TaskState` defined below tracks the following state transitions:
//
//   +---------+                        +---------+
//   |         |   scheduler picks it   |         |
//   |  Ready  | ---------------------> | Running |
//   |         | <--------------------- |         |
//   +---------+   preempted / yields   +----|----+
//        ^                                  | blocks on event, timer, etc.
//        |                                  v
//        |    event arrives (wake)     +---------+
//        ----------------------------- | Blocked |
//                                      +---------+
//                                           |
//                                           | (and eventually...)
//                                           V
//                                     +-----------+
//                                     |   Dying   |
//                                     +-----------+
//                                           |
//                                           |
//                                           V
//                                     +-----------+
//                                     |    Dead   |
//                                     +-----------+

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
    /// explicitly terminated itself) or faulted. The `Task` struct (and its
    /// stack allocation) remains alive until the task reaper runs and marks
    /// it as [`TaskState::Dead`] on the next timer tick.
    Dying,

    /// The reaper has marked the task as dead, and the scheduler will perform
    /// tombstone cleanup on the next scheduling pass, at which point the
    /// `Task` is dropped and the stack is freed. During the tombstone cleanup,
    /// the task's parent thread will also be marked as [`ThreadState::Dead`],
    /// just before the task is dropped.
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
// Kernel initial stack frame layout
// =============================================================================
//
// The context switcher saves and restores a task's register state by pushing
// and popping a fixed set of callee-saved GPRs onto/from the task's own stack.
// When we create a new kernel task, we have to manually write this exact frame
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

/// The software context frame written at the top of a new kernel task's stack.
///
/// `repr(C, packed)` ensures the fields are laid out in declaration order with
/// no padding, matching the exact byte sequence that `switch_to` pops via
/// `pop r15; pop r14; pop r13; pop r12; pop rbx; pop rbp; ret`.
#[repr(C, packed)]
struct KernelInitialFrame {
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

/// Naked trampoline that every new kernel task executes first when it is
/// scheduled for the very first time.
///
/// This trampoline exists because, after `switch_to` pops the
/// `KernelInitialFrame` and executes `ret`, RSP and all callee-saved registers
/// are in the state we wrote into the frame. But, the actual task entry
/// function expects its argument in RDI (per the System V ABI), not in R13.
/// This stub bridges that gap: it moves R13 into RDI and then calls the entry
/// function through R12.
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
/// `switch_to` via the `rip` field of an `KernelInitialFrame`. The register
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

// =============================================================================
// User initial stack frame layout
// =============================================================================
// 
// When a kernel thread starts, `switch_to` lands on a `KernelInitialFrame` and
// `rets` into `kernel_task_entry_stub`, which calls the entry function
// directly. User threads can't work that way, as we can't just call into ring
// 3. The only architectural mechanism to enter ring 3 is `iretq`, which
// atomically restores `RIP`, `CS`, `RFLAGS`, `RSP`, and `SS` from a frame on
// the kernel stack and transitions privilege.
// 
// So, the initial kernel stack for a user thread needs to look like this when
// it's first scheduled:
// 
// high address ->  +-----------------+
//                  | SS              |  <- USER_DS | 3
//                  | RSP (user)      |  <- top of user stack
//                  | RFLAGS          |  <- IF=1, IOPL=0, reserved bits correct
//                  | CS              |  <- USER_CS | 3
//                  | RIP             |  <- user entry point
//                  |-----------------|  <- iretq reads from here upward
//                  | r15             |
//                  | r14             |
//                  | r13             |
//                  | r12             |
//                  | rbx             |
//                  | rbp             |  <- saved_rsp points here
// low address  ->  +-----------------+     (bottom of KernelInitialFrame)

/// The RFLAGS value we hand to a fresh user thread via iretq.
///
/// Bit 9 (IF)       = 1: interrupts enabled in user mode from the start.
/// Bit 1 (reserved) = 1: always must be set per the x86-64 spec.
/// All other bits clear: no trap flag, IOPL=0 (no direct I/O port access),
/// no direction flag, no alignment check.
const USER_RFLAGS: u64 = (1 << 9) | (1 << 1);

/// Initial stack layout for a new user thread.
///
/// `switch_to` pops the callee-saved registers (bottom six fields) and
/// `ret`s into `user_task_entry_stub`. The stub executes `iretq`, which
/// consumes the top five fields and transitions to ring 3 at `rip` with
/// `rsp` pointing to the user stack.
///
/// Field order is from low address (bottom of struct) to high address (top),
/// matching the push/pop order on the stack.
#[repr(C, packed)]
struct UserInitialFrame {
    // Callee-saved registers, popped by `switch_to`; this is the same layout
    // as in `KernelInitialFrame`.
    r15: u64,
    r14: u64,
    r13: u64,
    r12: u64,
    rbx: u64,
    rbp: u64,

    // ret target: lands here after `switch_to` pops registers
    rip: u64,  // address of `user_task_entry_stub`

    // These are consumed by `iretq` inside `user_task_entry_stub`
    iretq_rip:    u64,  // user entry point virtual address
    iretq_cs:     u64,  // USER_CS | 3
    iretq_rflags: u64,  // IF=1, reserved bit=1
    iretq_rsp:    u64,  // top of user stack (virtual address)
    iretq_ss:     u64,  // USER_DS | 3
}

/// Naked entry trampoline for new user-mode tasks.
///
/// This runs in ring 0 on the task's kernel stack immediately after the
/// first `switch_to` lands on it. Its only job is to execute `iretq`, which
/// atomically:
///   - Loads RIP from the kernel stack  -> user entry point
///   - Loads CS  from the kernel stack  -> USER_CS | RPL3 (ring 3)
///   - Loads RFLAGS                     -> IF=1
///   - Loads RSP from the kernel stack  -> user stack top
///   - Loads SS  from the kernel stack  -> USER_DS | RPL3 (ring 3)
///   - Transitions the CPU to ring 3
///
/// After `iretq` the kernel stack is idle, and it will be reused the next time
/// this thread is interrupted or makes a syscall (the CPU pushes a new `iretq`
/// frame onto it at that point, and `TSS.rsp[0]` ensures it finds this stack).
#[unsafe(naked)]
unsafe extern "C" fn user_task_entry_stub() {
    core::arch::naked_asm!(
        // At this point, we are in ring 0, on the task's kernel stack.
        // The UserInitialFrame's `iretq` fields (rip, cs, rflags, rsp, and ss)
        // are sitting at [rsp+0..rsp+39] exactly as `iretq` expects them.
        // We now re-enable interrupts and transfer the execution to ring 3.
        "sti",
        "iretq",
    );
}

/// A task is the underlying execution unit of process threads.
///
/// Each `Task` owns its execution stack for its entire lifetime. The stack
/// is heap-allocated, so the kernel heap must be initialized before any task
/// is created.
///
/// Tasks are created via [`Task::new_kernel`] or [`Task::new_user`] and managed
/// by the scheduler in `scheduler.rs`. The scheduler holds `Task` values behind
/// an `IrqSpinLock`, so that task state is consistent even when modified from
/// interrupt context.
///
/// When a `Task` is dropped (e.g., after tombstone cleanup), `Box<[u8]>` drops
/// its allocation, freeing the stack back to the kernel heap. The scheduler
/// must ensure the task is not currently running on any CPU when it is dropped.
pub struct Task {
    /// The unique identifier for this task within the scheduler's task table.
    ///
    /// It gets initialized to [`TaskId::PENDING`] at construction time, since
    /// the slot index is not yet known. The real `TaskId` is assigned by
    /// `insert_and_queue_task` once the task has been placed into a slot.
    ///
    /// After insertion, `id` is stable and never changes for the lifetime of
    /// the task. On removal, the slot's generation is incremented, rendering
    /// this `TaskId` stale and causing any subsequent lookup via
    /// [`Scheduler::get_task`] to return `None`.
    ///
    /// The idle task is the only exception to the above, and its `id` is set
    /// directly to [`TaskId::IDLE`] by [`Task::new_idle`] and inserted via
    /// `insert_idle_task`, bypassing the normal pending-to-assigned lifecycle.
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
    /// the bottom of a `KernelInitialFrame`, a `UserInitialFrame`, or a live
    /// `switch_to` frame inside the stack allocation). It is stored as `u64`
    /// rather than `*mut u64` for these reasons:
    ///   - `u64` is `Send`; raw pointers are not. This allows `Task` to
    ///     implement `Send` with a single `unsafe impl` rather than requiring
    ///     a wrapper type.
    ///   - The pointer arithmetic is done entirely inside `unsafe` assembly in
    ///     the switcher, so there is no benefit to maintaining the Rust type.
    ///
    /// Invariant: between scheduler invocations (i.e., when no `switch_to`
    /// is in progress), `saved_rsp` always points to a valid frame inside
    /// `_stack` for `Ready` and `Blocked` tasks. For the currently `Running`
    /// task, `saved_rsp` may be stale (it holds the value from the previous
    /// switch-away and will be overwritten by the next `switch_to`).
    pub saved_rsp: u64,

    /// The fixed top of the kernel stack allocated for this task, used to
    /// populate TSS.rsp[0], so the CPU has somewhere to land on ring 3 -> ring
    /// 0 transitions. Every task, including user tasks, has one of these,
    /// because even user tasks need a kernel stack to handle their interrupts
    /// and syscalls.
    pub kernel_stack_top: u64,

    /// Used to signal the scheduler and context switcher whether `TSS.rsp[0]`
    /// needs to be updated on context switch. If this is true, `TSS.rsp[0]` is
    /// updated on switch; otherwise (i.e., it's a kernel task), the update is
    /// skipped on switch.
    pub is_user_task: bool,
    
    /// The read-only PML4 address value cached from the parent process's
    /// address space. It is used for fast checking whether `TSS.rsp[0]` needs
    /// to be updated when the scheduler's
    /// [`crate::task_scheduler::scheduler::schedule`] routine calls
    /// [`crate::task_scheduler::switcher::switch_to`]. Having this value cached
    /// here saves critical time during context switches.
    pub cr3: u64,

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

    /// Back-reference to the `Thread` object that "owns" this `Task` for
    /// process management purposes.
    /// 
    /// This is a `Weak` reference to avoid an ownership cycle. It is set
    /// atomically during `Process::spawn_thread` before the thread is
    /// published to the process thread list.
    pub thread: Weak<IrqSpinLock<Thread>>,

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
    /// 
    /// # Arguments
    /// 
    /// * `thread` - `Weak` back-reference to the task's parent `Thread`.
    pub(crate) fn new_idle(thread: Weak<IrqSpinLock<Thread>>) -> Self {
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
        //
        // `kernel_stack_top` is set to 0, as the idle task never runs in ring
        // 3; so, TSS.rsp[0] is always overwritten before any user task runs.
        // Thus, initializing it to 0 here is safe.
        Task {
            id:               TaskId::IDLE,
            state:            TaskState::Running,
            saved_rsp:        0,
            kernel_stack_top: 0,
            is_user_task:     false,
            cr3:              0,
            ticks_remaining:  crate::globals::TASK_QUANTUM_TICKS,
            ticks_executed:   AtomicU64::new(0),
            thread,
            _stack:           stack,
        }
    }

    /// Creates a new kernel task that will call `entry(arg)` when first
    /// scheduled; the task is immediately associated with its parent `Thread`.
    ///
    /// Allocates a `TASK_STACK_SIZE`-byte stack from the kernel heap, writes
    /// an `KernelInitialFrame` at the top of the stack, and sets `saved_rsp` to
    /// point at the bottom of that frame so the switcher can resume the task
    /// correctly. The new task starts in `TaskState::Ready` and will not run
    /// until the scheduler places it on the run queue and eventually calls
    /// `switch_to`.
    ///
    /// # Arguments
    /// 
    /// * `entry`  - The task's entry function.
    /// * `arg`    - An opaque `u64` passed as the sole argument to `entry`. We
    ///              can use it as a type-erased pointer, an integer, or ignore
    ///              it if the task needs no parameter.
    /// * `thread` - `Weak` back-reference to the task's parent `Thread`.
    ///
    /// # Panics
    /// 
    /// Panics if the kernel heap cannot satisfy the stack allocation. This may
    /// happen if `TASK_STACK_SIZE` bytes are unavailable, and the heap cannot
    /// grow.
    pub fn new_kernel(
        entry: fn(u64),
        arg: u64,
        thread: Weak<IrqSpinLock<Thread>>,
    ) -> Self {
        // Allocate the stack for the new task.
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

        // Compute the frame pointer. We place the `KernelInitialFrame` just
        // below the top of the stack, aligned to 16 bytes, so that after
        // `switch_to` pops the frame and executes `ret`, RSP lands on a 16-byte
        // aligned address as required by the ABI. The `& !0xF` mask clears the
        // low 4 bits, rounding down to 16 bytes.
        //
        // `stack_top` is the highest usable address in this task's kernel
        // stack, aligned to 16 bytes. This is what `TSS.rsp[0]` must point to,
        // so that the CPU lands on a valid stack when entering the kernel from
        // ring 3.
        let stack_top = (stack.as_mut_ptr() as usize + TASK_STACK_SIZE) & !0xF;
        let frame_ptr = (
            (stack_top - core::mem::size_of::<KernelInitialFrame>()) & !0xF)
            as *mut KernelInitialFrame;

        // Write the `KernelInitialFrame`.
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
        //     `size_of::<KernelInitialFrame>()` bytes below `stack_top`, so the
        //     entire write falls within the allocation bounds.
        //   - `stack` is zero-initialized, so there are no pre-existing invalid
        //     values at `frame_ptr` that we would be reading through later.
        unsafe {
            frame_ptr.write(KernelInitialFrame {
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
            id: TaskId::PENDING, // Gets replaced with proper ID after insertion
            state: TaskState::Ready,
            saved_rsp,
            kernel_stack_top: stack_top as u64,
            is_user_task: false,
            cr3: 0,  // Kernel task, so no CR3 switch needed
            ticks_remaining: crate::globals::TASK_QUANTUM_TICKS,
            ticks_executed: AtomicU64::new(0),
            thread,
            _stack: stack,
        }
    }

    /// Creates a new user-mode task.
    ///
    /// Allocates a kernel stack for interrupt/syscall handling and sets up a
    /// [`UserInitialFrame`], so that the first `switch_to` lands in
    /// [`user_task_entry_stub`], which `iretq`s into ring 3 at `user_entry`.
    /// A user stack is already allocated and provided inside `address_space`.
    ///
    /// # Arguments
    ///
    /// * `user_entry`    - Virtual address of the user-mode entry point. This
    ///                     is a raw address in the process's virtual address
    ///                     space, not a Rust function pointer.
    /// * `address_space` - The process's address space, where the user stack
    ///                     is mapped.
    /// * `pml4_phys`     - Physical address of the process's PML4, cached into
    ///                     the task for use by `switch_to`.
    /// * `thread`        - `Weak` reference to the owning thread.
    pub fn new_user(
        user_entry: u64,
        address_space: &AddressSpace,
        pml4_phys: u64,
        thread: Weak<IrqSpinLock<Thread>>,
    ) -> Self {
        // Allocate the kernel stack. This stack is used whenever the CPU
        // enters ring 0 on behalf of this thread (interrupts, syscalls, etc).
        // TSS.rsp[0] is updated to kernel_stack_top on every switch to
        // this task so the CPU always finds a valid kernel stack.
        let mut kernel_stack: Box<[u8]> = {
            let mut stack_vector = Vec::with_capacity(TASK_STACK_SIZE);
            stack_vector.resize(TASK_STACK_SIZE, 0u8);
            stack_vector.into_boxed_slice()
        };
        let kernel_stack_top =
            (kernel_stack.as_mut_ptr() as usize + TASK_STACK_SIZE) & !0xF;

        // Build the UserInitialFrame at the top of the kernel stack.
        // The frame must be 16-byte aligned per the ABI.
        let frame_ptr = ((kernel_stack_top
            - core::mem::size_of::<UserInitialFrame>()) & !0xF)
            as *mut UserInitialFrame;

        // Write the `UserInitialFrame`
        unsafe {
            frame_ptr.write(UserInitialFrame {
                // Callee-saved registers are all zero for a fresh task
                r15: 0,
                r14: 0,
                r13: 0,
                r12: 0,
                rbx: 0,
                rbp: 0,

                // `switch_to` will `ret` to `user_task_entry_stub`, which
                // executes `iretq` using the fields below.
                rip: user_task_entry_stub as *const () as u64,

                // iretq frame, consumed atomically by `iretq` in the stub.
                iretq_rip:    user_entry,
                iretq_cs:     USER_CS_RPL3,
                iretq_rflags: USER_RFLAGS,
                iretq_rsp:    address_space.user_stack_top,  // user stack start
                iretq_ss:     USER_DS_RPL3,
            });
        }

        Task {
            id:               TaskId::PENDING,
            state:            TaskState::Ready,
            saved_rsp:        frame_ptr as u64,
            kernel_stack_top: kernel_stack_top as u64,
            is_user_task:     true,
            cr3:              pml4_phys,
            ticks_remaining:  crate::globals::TASK_QUANTUM_TICKS,
            ticks_executed:   AtomicU64::new(0),
            thread,
            _stack:           kernel_stack,
        }
    }
}

/// Handles a task's normal return and exit.
/// 
/// This gets called automatically by `task_entry_stub` when a task's entry
/// function returns normally. It marks the current task as `Dying`, marks its
/// parent `Thread` as `Dying`, and immediately yields to the scheduler. The
/// actual stack and task table cleanup (tombstone cleanup) are deferred by two
/// timer ticks. This function never returns.
pub fn task_exit() -> ! {
    // Disable interrupts for the entire exit sequence to close the race
    // window between marking this task Dying and calling schedule(). If a
    // timer fires between those two points, schedule() will see a Dying task
    // as current and produce a null old_rsp_ptr, corrupting the switch.
    // schedule() will re-enable interrupts after the switch via
    // restore_interrupts(), so we don't need to restore them here.
    unsafe { core::arch::asm!("cli", options(nomem, nostack)) };

    // Mark this task as `Dying`. We do this in a block, so the scheduler lock
    // is dropped before we call `schedule()` (which will acquire it again
    // internally), and we must not hold it across that call.
    {
        let mut scheduler = super::scheduler::SCHEDULER.lock();
        let this_id = scheduler.current;
        if let Some(task) = scheduler.get_task_mut(this_id) {
            task.state = TaskState::Dying;

            // We also mark the task's parent thread as dying
            task.thread.upgrade().unwrap().lock().state = ThreadState::Dying;

            // Queue task reaper to mark the task as dead on the next tick
            if queue_task_reaper(this_id).is_err() {
                crate::hardware_manager::sprint(
                    "\n[REAPER] Failed to queue task reaper...\n");
            }
        }
        // The lock is dropped here
    }

    // With the status set to `Dying`, we now hand off to the next ready task.
    // This task will never be rescheduled, because `Dying` tasks are not
    // re-queued. The reaper will free this task's stack at the next timer tick,
    // by which point we are no longer running on it.
    unsafe { super::scheduler::schedule() };

    // Truly unreachable, as `schedule()` switches the stack away and never
    // returns to a `Dying` task. If we somehow land here, fault loudly.
    unreachable!("task_exit: schedule() returned to a dying/dead task");
}
