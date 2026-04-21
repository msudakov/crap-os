//! CPU Context Switcher
//!
//! This module contains the single function `switch_to`, which is the only
//! place in the entire kernel where CPU register state is manually saved and
//! restored. Everything else in the Task Scheduler is ordinary Rust code
//! that never touches registers directly.
//!
//! A context switch involves pausing one task's execution and resuming
//! another's. From each task's perspective, it looks like an ordinary function
//! call to `switch_to` that takes an unusually long time to return, because
//! between the call and the return, the CPU executed some other task entirely.
//!
//! The mechanism that makes this transparent is saving and restoring a minimal
//! snapshot of the CPU's registers. When we switch away from task A:
//!   1. We push A's callee-saved registers onto A's own stack;
//!   2. We record A's stack pointer so we can find these registers later.
//!
//! When we later switch back to task A:
//!   1. We restore A's stack pointer;
//!   2. We pop A's callee-saved registers back from A's stack;
//!   3. We execute `ret`, which pops A's return address and jumps there,
//!      resuming A exactly where it left off, as if `switch_to` just returned.
//!
//! The function is declared `extern "C"`, so the compiler treats it exactly
//! like any other function call under the System V AMD64 ABI. Under the ABI,
//! we must save and restore callee-saved registers (RBX, RBP, R12–R15). The
//! frame layout is as follows:
//!
//!   +----------------------------------+  <- stack_top (highest address)
//!   |  (unused / guard space)          |
//!   |----------------------------------|
//!   |  rip  = task_entry_stub ptr      |  <- "return address" consumed by ret
//!   |----------------------------------|
//!   |  rbp  = 0                        |  <- pushed first, popped last  (rbp)
//!   |  rbx  = 0                        |
//!   |  r12  = entry fn ptr (new tasks) |  <- pushed 3rd, popped 4th     (r12)
//!   |  r13  = arg u64 (for new tasks)  |  <- pushed 4th, popped 3rd     (r13)
//!   |  r14  = 0                        |
//!   |  r15  = 0                        |  <- pushed last, popped first (r15)
//!   +----------------------------------+  <- saved_rsp points here
//!
//! We don't need to handle the caller-saved registers because the caller is
//! responsible for them, and the callee is free to clobber them. And, the
//! compiler already saves these before calling `switch_to`, if needed. So, we
//! don't need to worry about them.
//!
//! This is deliberately minimal (48 bytes per task context) and avoids
//! saving the FPU/SSE/AVX state (XMM/YMM registers), which works as long
//! as the kernel does not use floating-point or SIMD in task context. If FP
//! support is ever added, each task will need an additional FP state save area.
//!
//! The function handles both cases of brand-new tasks vs. previously-run tasks
//! transparently. This unified design means it needs no "first-run" task check
//! and contains no branches at all; the hot path is a straight-line sequence
//! of instructions:
//!
//!   - For a previously-run task:
//!     Its stack already contains a real context frame pushed by the last
//!     `switch_to` call that switched away from it. Popping that frame
//!     restores the task's registers exactly as they were, and `ret` jumps
//!     back into `schedule()` at the instruction after the `call switch_to`.
//!
//!   - For a brand-new task (never yet scheduled):
//!     `Task::new` (in task.rs) pre-builds a synthetic `InitialFrame` at the
//!     top of the task's stack with the following:
//!       - R12 = entry function pointer
//!       - R13 = entry argument
//!       - RBP, RBX, R14, R15 = 0 (no meaningful initial values)
//!       - A fake "return address" above the frame pointing at
//!         `task_entry_stub`
//!     When `switch_to` pops this synthetic frame and executes `ret`, it jumps
//!     to `task_entry_stub`, which moves R12/R13 into the ABI argument
//!     registers (RDI) and calls the real entry function.


/// Switches the execution context from an "outgoing" task to an "incoming"
/// task.
/// 
/// # Arguments
/// 
/// * `old_rsp_ptr` - (RDI) Pointer to the `saved_rsp` field of the outgoing
///   task's `Task` struct. The function stores the outgoing task's RSP here
///   after pushing the frame, so the scheduler can later pass it back as
///   `new_rsp` to resume this task.
///
/// * `new_rsp`     -  (RSI) The value, previously written to `saved_rsp` for
///   the incoming task, to restore, so that popping the frame retrieves that
///   task's registers correctly.
/// 
/// # Safety
/// 
/// This function has no compiler-generated prologue or epilogue whatsoever.
/// The entire function body must be a single `naked_asm!` block. This is
/// required here for two reasons:
///   1. Any compiler-inserted `push rbp` / `sub rsp, N` prologue would corrupt
///      the frame layout that the pop sequence depends on;
///   2. We need `ret` to be the very last instruction, so it pops the correct
///      return address from whichever stack (old or new) is active at that
///      point.
/// Because the function body is pure assembly with no Rust-level control flow,
/// the compiler cannot reason about its safety, and we must ensure all of it
/// works correctly.
#[unsafe(naked)]
pub unsafe extern "C" fn switch_to(old_rsp_ptr: *mut u64, new_rsp: u64) {
    core::arch::naked_asm!(
        // =====================================================================
        // Phase 1: Save the outgoing task's context
        // =====================================================================
        //
        // We push all six callee-saved registers onto the current (outgoing)
        // task's stack. After these six pushes, RSP points at the bottom of the
        // context frame (the R15 slot), which is exactly the value we will
        // store in `saved_rsp` and later restore from.
        //
        // Push order: highest address first (rbp), lowest address last (r15).
        // The corresponding pops in Phase 2 must be in the exact reverse order.
        "push rbp",  // rbp: frame pointer / general purpose
        "push rbx",  // rbx: general purpose
        "push r12",  // r12: holds entry fn ptr for new tasks
        "push r13",  // r13: holds entry arg for new tasks
        "push r14",  // r14: general purpose
        "push r15",  // r15: general purpose

        // Store the outgoing task's RSP (now pointing at the bottom of the
        // frame we just built) into old_rsp_ptr, which is held in RDI (per
        // System V ABI). After this write, the outgoing task is fully
        // suspended: its register state is on its stack, and `saved_rsp`
        // records where to find it.
        "mov [rdi], rsp",

        // =====================================================================
        // Phase 2: Restore the incoming task's context
        // =====================================================================
        //
        // From this instruction forward, we are operating in the context of the
        // incoming task. The outgoing task is logically paused and will not
        // resume until `switch_to` is called again with its `saved_rsp`. We
        // load the incoming task's saved RSP from the second argument (RSI).
        // This switches the stack pointer to the incoming task's stack, where
        // that task's previously-pushed context frame (or the synthetic
        // InitialFrame for a new task) is waiting.
        "mov rsp, rsi",

        // Next, we pop the callee-saved registers in the reverse push order.
        // For a previously-run task, these restore the exact register values
        // the task had when it was last switched out. For a brand-new task,
        // these load the values `Task::new` pre-wrote into the `InitialFrame`:
        //     R15 = 0  (no initial value)
        //     R14 = 0  (no initial value)
        //     R13 = task argument (u64), is read by `task_entry_stub`
        //     R12 = entry function pointer, is called by `task_entry_stub`
        //     RBX = 0  (no initial value)
        //     RBP = 0  (marks outermost frame for stack unwinders)
        "pop r15",
        "pop r14",
        "pop r13",
        "pop r12",
        "pop rbx",
        "pop rbp",

        // =====================================================================
        // Phase 3: Transfer control to the incoming task
        // =====================================================================
        //
        // This `ret` pops the 8-byte word now at the top of the stack (the word
        // immediately above the frame we just popped) into RIP and jumps there.
        // What that word contains depends on which kind of task is resuming:
        //
        //   - For a previously-run task:
        //     RSP now points at the return address the compiler emitted for the
        //     `call switch_to` instruction inside `schedule()`. `ret` jumps
        //     back into `schedule()` at the instruction immediately after that
        //     call, as if `switch_to` had just returned normally.
        //
        //   - For a brand-new task:
        //     `Task::new` placed the address of `task_entry_stub` at this
        //     position in the InitialFrame (the `rip` field). So, `ret` jumps
        //     to `task_entry_stub` in this case.
        "ret",
    );
}
