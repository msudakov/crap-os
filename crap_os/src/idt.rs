//! Interrupt Descriptor Table (IDT)
//!
//! This module owns the full 256-entry IDT and all CPU exception handlers for
//! vectors 0–31. It also implements the needed hardware IRQ vectors (e.g. APIC)
//! in the 32–255 range, and all unused ones get a generic "unhandled IRQ" stub
//! that halts.
//!
//! Each IDT entry points to a `#[naked]` trampoline function, which does the
//!  following:
//!   1. Pushes a dummy error code (0) on vectors that don't push one
//!      automatically, so every handler receives a uniform stack layout;
//!   2. Saves all general-purpose registers (RAX..R15) onto the stack;
//!   3. Passes a pointer to the resulting `InterruptFrame` to a safe
//!      handler function via the System V AMD64 calling convention (RDI);
//!   4. On return from the handler, restores all GPRs and executes IRETQ.
//!
//! After the trampoline saves the registers, the stack is laid out as follows
//! on handler entry:
//!
//!   Higher addresses  (SS pushed first by CPU)
//!   +----------------------+
//!   |  SS         (+80)    |  -+
//!   |  RSP        (+72)    |   |
//!   |  RFLAGS     (+64)    |   |  Pushed by CPU automatically
//!   |  CS         (+56)    |   |
//!   |  RIP        (+48)    |  -+
//!   |  error_code (+40)    |  <- CPU (for #DF, #PF, #GP, etc.) or 0 (our stub)
//!   |  R15        (+32)    |  -+
//!   |  R14        (+24)    |   |
//!   |  R13        (+16)    |   |
//!   |  R12        (+ 8)    |   |  Saved by trampoline (PUSH order: R15..RAX)
//!   |  R11        (+ 0)    |   |
//!   |  R10        (-8 )    |   |
//!   |  R9         (-16)    |   |
//!   |  R8         (-24)    |   |
//!   |  RBP        (-32)    |   |
//!   |  RDI        (-40)    |   |
//!   |  RSI        (-48)    |   |
//!   |  RDX        (-56)    |   |
//!   |  RCX        (-64)    |   |
//!   |  RBX        (-72)    |   |
//!   |  RAX        (-80)    |  -+  <- RSP on handler entry, passed as &frame
//!   +----------------------+
//!   Lower addresses
//!
//! The `InterruptFrame` struct mirrors this layout in declaration order so
//! that `&frame` (where frame: *const InterruptFrame = RSP) gives safe Rust
//! access to every field.
//!
//! The double-fault handler (#DF, vector 8) is installed with IST=1, which
//! points to the dedicated `DOUBLE_FAULT_STACK` defined in gdt.rs. All other
//! handlers use IST=0 (no stack switch; they run on whatever stack was active
//! when the exception fired).

use core::sync::atomic::Ordering;
use core::arch::asm;        // used in non-naked handlers (cr2 read, cpu_halt)
use core::arch::naked_asm;  // used inside #[naked] trampolines
use crate::gdt::KERNEL_CS;
use crate::globals::IDT;
use crate::hardware_manager;
use crate::helper_functions::print_u64_field;

// =============================================================================
// InterruptFrame
// =============================================================================

/// The full CPU + GPR context captured by every interrupt trampoline.
///
/// Fields are ordered to match the stack layout described above, from lowest
/// address (RAX, pushed last) to highest address (SS, pushed first by CPU).
/// The struct is `#[repr(C)]` so the compiler cannot reorder fields and the
/// pointer cast from RSP is well-defined.
#[repr(C)]
pub struct InterruptFrame {
    // General-purpose registers saved by the trampoline (low -> high address)
    pub rax: u64,
    pub rbx: u64,
    pub rcx: u64,
    pub rdx: u64,
    pub rsi: u64,
    pub rdi: u64,
    pub rbp: u64,
    pub r8:  u64,
    pub r9:  u64,
    pub r10: u64,
    pub r11: u64,
    pub r12: u64,
    pub r13: u64,
    pub r14: u64,
    pub r15: u64,

    // Error code: pushed by the CPU for some exceptions; 0 for the rest.
    pub error_code: u64,

    // Fields pushed automatically by the CPU on exception entry.
    pub rip:    u64,
    pub cs:     u64,
    pub rflags: u64,
    pub rsp:    u64,
    pub ss:     u64,
}

// =============================================================================
// IDT Entry Encoding
// =============================================================================

/// A single 16-byte IDT gate descriptor (interrupt or trap gate).
///
/// `repr(C)` keeps field order stable. `repr(packed)` is not needed here
/// because the fields are naturally aligned when laid out in declaration order
/// on a 16-byte boundary, but we ensure the containing array is aligned to
/// 16 bytes via the IDT array itself.
#[repr(C)]
#[derive(Copy, Clone)]
pub struct IdtEntry {
    offset_low:  u16,   // Bits 15:0 of the handler virtual address
    selector:    u16,   // Code segment selector (must be KERNEL_CS = 0x08)
    ist_and_rsv: u8,    // Bits 2:0 = IST index (0 = no switch); bits 7:3 = 0
    type_attr:   u8,    // Gate type, DPL, and Present bit
    offset_mid:  u16,   // Bits 31:16 of the handler virtual address
    offset_high: u32,   // Bits 63:32 of the handler virtual address
    _reserved:   u32,   // Must be zero
}

// Gate type and attributes byte values. Bit layout of `type_attr`:
//   Bits 3:0 - gate type
//   Bit  4   - 0 (storage segment; must be 0 for interrupt/trap gates)
//   Bits 6:5 - DPL (descriptor privilege level, 0 = ring 0 only)
//   Bit  7   - Present
const INTERRUPT_GATE: u8 = 0x8E;  // P=1, DPL=0, type=0b1110 (interrupt gate)
const TRAP_GATE:      u8 = 0x8F;  // P=1, DPL=0, type=0b1111 (trap gate)

impl IdtEntry {
    /// Returns a zeroed, non-present IDT entry.
    pub const fn missing() -> Self {
        Self {
            offset_low:  0,
            selector:    0,
            ist_and_rsv: 0,
            type_attr:   0,
            offset_mid:  0,
            offset_high: 0,
            _reserved:   0,
        }
    }

    /// Builds an interrupt gate entry.
    ///
    /// An interrupt gate automatically clears IF (disables interrupts) on
    /// entry, preventing re-entrant interrupts. This is correct for all
    /// hardware exception handlers and IRQ stubs.
    ///
    /// # Arguments
    /// 
    /// * `handler` - Virtual address of the `#[naked]` trampoline function.
    /// * `ist`     - IST slot (1–7) for an alternate stack, or 0 for none.
    pub fn interrupt_gate(handler: u64, ist: u8) -> Self {
        Self {
            offset_low:  (handler & 0xFFFF) as u16,
            selector:    KERNEL_CS,
            ist_and_rsv: ist & 0x7,
            type_attr:   INTERRUPT_GATE,
            offset_mid:  ((handler >> 16) & 0xFFFF) as u16,
            offset_high: (handler >> 32) as u32,
            _reserved:   0,
        }
    }

    /// Builds a trap gate entry.
    ///
    /// A trap gate does not clear IF on entry, so interrupts remain enabled
    /// while the handler runs. Used for exceptions like breakpoints (#BP) that
    /// are expected to return and where nested interrupts are acceptable.
    ///
    /// # Arguments
    /// 
    /// * `handler` - Virtual address of the `#[naked]` trampoline function.
    /// * `ist`     - IST slot (1–7) for an alternate stack, or 0 for none.
    pub fn trap_gate(handler: u64, ist: u8) -> Self {
        Self {
            offset_low:  (handler & 0xFFFF) as u16,
            selector:    KERNEL_CS,
            ist_and_rsv: ist & 0x7,
            type_attr:   TRAP_GATE,
            offset_mid:  ((handler >> 16) & 0xFFFF) as u16,
            offset_high: (handler >> 32) as u32,
            _reserved:   0,
        }
    }
}

/// The kernel's 256-entry Interrupt Descriptor Table.
///
/// 256 * 16 bytes = 4096 bytes (exactly one page). `align(16)` satisfies
/// the architectural requirement that the IDT base be 8-byte aligned; 16-byte
/// alignment additionally ensures every entry starts on a naturally aligned
/// boundary for clean cache-line behaviour.
#[repr(C, align(16))]
pub struct Idt {
    pub entries: [IdtEntry; 256],
}

/// The 10-byte IDTR pseudo-descriptor loaded via `lidt`.
#[repr(C, packed)]
struct IdtDescriptor {
    limit: u16,
    base:  u64,
}

// =============================================================================
// Trampoline Macro
// =============================================================================
//
// This macro handles two types of naked functions, depending on error code:
//
// - make_exception_stub!(name, vector, handler)
//     For exceptions that do not push an error code (vectors 0, 1, 2, 3, 4,
//     5, 6, 7, 9, 15, 16, 18, 19, 20, 21, 28, 29, 31). The stub pushes a
//     dummy 0 so every handler receives a uniform frame.
//
// - make_exception_stub_error_code!(name, vector, handler)
//     For exceptions that push an error code (vectors 8, 10, 11, 12, 13,
//     14, 17, 30). The stub skips the dummy push.
//
// In both cases, the macro emits a `#[naked]` extern "C" function whose sole
// content is an `asm!` block that:
//   1. (Optionally) pushes a dummy error code.
//   2. Saves RAX–R15 in the order that matches `InterruptFrame`.
//   3. Passes RSP (pointing at the bottom of the frame) in RDI.
//   4. Calls the safe Rust handler.
//   5. Restores RAX–R15 and executes IRETQ.
//
// The GPR save/restore order is:
// push r15, r14, r13, r12, r11, r10, r9, r8, rbp, rdi, rsi, rdx, rcx, rbx, rax
// High registers go first, so RAX ends up at the lowest address, matching the
// InterruptFrame field order.

macro_rules! make_exception_stub {
    ($name:ident, $handler:ident) => {
        #[unsafe(naked)]
        pub unsafe extern "C" fn $name() {
            naked_asm!(
                // Push dummy error code (0) - this exception has none.
                "push 0",
                // Save all GPRs onto the stack in high-to-low register order
                "push r15",
                "push r14",
                "push r13",
                "push r12",
                "push r11",
                "push r10",
                "push r9",
                "push r8",
                "push rbp",
                "push rdi",
                "push rsi",
                "push rdx",
                "push rcx",
                "push rbx",
                "push rax",
                // RDI contains pointer to the InterruptFrame (System V arg 1)
                "mov rdi, rsp",
                // Save the frame pointer below the aligned stack so we can
                // recover RSP after the call. The handler (which is a normal C
                // function) is free to clobber RDI, so we cannot rely on
                // it surviving the call.
                "mov rbx, rsp",
                // 16-byte align the stack before the CALL
                "and rsp, -16",
                "call {handler}",
                // Restore RSP from the copy we saved in RBX
                "mov rsp, rbx",
                // Restore GPRs in reverse push order
                "pop rax",
                "pop rbx",
                "pop rcx",
                "pop rdx",
                "pop rsi",
                "pop rdi",
                "pop rbp",
                "pop r8",
                "pop r9",
                "pop r10",
                "pop r11",
                "pop r12",
                "pop r13",
                "pop r14",
                "pop r15",
                // Skip the error code slot (either dummy or real)
                "add rsp, 8",
                // Return from interrupt: pops RIP, CS, RFLAGS, RSP, and SS
                "iretq",
                handler = sym $handler,
            );
        }
    };
}

/// Same as `make_exception_stub!` but for exceptions that push an error code.
/// We do not push a dummy 0 here - the CPU already put the real error code on
/// the stack, so the frame layout is identical.
macro_rules! make_exception_stub_error_code {
    ($name:ident, $handler:ident) => {
        #[unsafe(naked)]
        pub unsafe extern "C" fn $name() {
            naked_asm!(
                // No dummy push: CPU already pushed the error code. Everything
                // else if the same.
                "push r15",
                "push r14",
                "push r13",
                "push r12",
                "push r11",
                "push r10",
                "push r9",
                "push r8",
                "push rbp",
                "push rdi",
                "push rsi",
                "push rdx",
                "push rcx",
                "push rbx",
                "push rax",
                "mov rdi, rsp",
                "mov rbx, rsp",
                "and rsp, -16",
                "call {handler}",
                "mov rsp, rbx",
                "pop rax",
                "pop rbx",
                "pop rcx",
                "pop rdx",
                "pop rsi",
                "pop rdi",
                "pop rbp",
                "pop r8",
                "pop r9",
                "pop r10",
                "pop r11",
                "pop r12",
                "pop r13",
                "pop r14",
                "pop r15",
                "add rsp, 8",
                "iretq",
                handler = sym $handler,
            );
        }
    };
}

// =============================================================================
// Trampoline Functions (one per vector 0–31, plus a generic IRQ stub)
// =============================================================================
//
// Vectors that push an error code (hardware-mandated):
//   8  (#DF  double fault)
//   10 (#TS  invalid TSS)
//   11 (#NP  segment not present)
//   12 (#SS  stack fault)
//   13 (#GP  general protection)
//   14 (#PF  page fault)
//   17 (#AC  alignment check)
//   30 (#SX  security exception)
//
// All others use make_exception_stub! which pushes a dummy 0.

make_exception_stub!(stub_de,  handler_divide_error);         // #DE  vec 0
make_exception_stub!(stub_db,  handler_debug);                // #DB  vec 1
make_exception_stub!(stub_nmi, handler_nmi);                  // NMI  vec 2
make_exception_stub!(stub_bp,  handler_breakpoint);           // #BP  vec 3
make_exception_stub!(stub_of,  handler_overflow);             // #OF  vec 4
make_exception_stub!(stub_br,  handler_bound_range);          // #BR  vec 5
make_exception_stub!(stub_ud,  handler_invalid_opcode);       // #UD  vec 6
make_exception_stub!(stub_nm,  handler_device_not_available); // #NM  vec 7
make_exception_stub_error_code!(
    stub_df, handler_double_fault);                           // #DF  vec 8
make_exception_stub!(stub_cso, handler_coproc_seg_overrun);   // --   vec 9
make_exception_stub_error_code!(
    stub_ts, handler_invalid_tss);                            // #TS  vec 10
make_exception_stub_error_code!(
    stub_np, handler_segment_not_present);                    // #NP  vec 11
make_exception_stub_error_code!(
    stub_ss, handler_stack_fault);                            // #SS  vec 12
make_exception_stub_error_code!(
    stub_gp, handler_general_protection);                     // #GP  vec 13
make_exception_stub_error_code!(stub_pf, handler_page_fault); // #PF  vec 14
make_exception_stub!(stub_rsv15, handler_reserved);           // --   vec 15
make_exception_stub!(stub_mf, handler_x87_fpu);               // #MF  vec 16
make_exception_stub_error_code!(
    stub_ac, handler_alignment_check);                        // #AC vec 17
make_exception_stub!(stub_mc,  handler_machine_check);        // #MC  vec 18
make_exception_stub!(stub_xm,  handler_simd_exception);       // #XM  vec 19
make_exception_stub!(stub_ve,  handler_virtualization);       // #VE  vec 20
make_exception_stub!(stub_cp,  handler_control_protection);   // #CP  vec 21
make_exception_stub!(stub_rsv22, handler_reserved);           // --   vec 22
make_exception_stub!(stub_rsv23, handler_reserved);           // --   vec 23
make_exception_stub!(stub_rsv24, handler_reserved);           // --   vec 24
make_exception_stub!(stub_rsv25, handler_reserved);           // --   vec 25
make_exception_stub!(stub_rsv26, handler_reserved);           // --   vec 26
make_exception_stub!(stub_rsv27, handler_reserved);           // --   vec 27
make_exception_stub!(stub_hv,  handler_hypervisor_injection); // #HV  vec 28
make_exception_stub!(stub_vc,  handler_vmm_communication);    // #VC  vec 29
make_exception_stub_error_code!(
    stub_sx, handler_security_exception);                     // #SX vec 30
make_exception_stub!(stub_rsv31, handler_reserved);           // --   vec 31

// Hardware IRQ handlers (APIC)
make_exception_stub!(stub_apic_timer, handler_apic_timer);    // vec 0x20
make_exception_stub!(
    stub_apic_keyboard, handler_apic_keyboard);               // vec 0x21
make_exception_stub!(
    stub_apic_spurious, handler_apic_spurious);               // vec 0xFF

/// Generic stub for hardware IRQ vectors 32–255. Pushes vector 0xFF as a
/// placeholder and halts - these will be replaced by real handlers when the
/// PIC/APIC is initialised.
#[unsafe(naked)]
pub unsafe extern "C" fn stub_irq_generic() {
    naked_asm!(
        "push 0",
        "push r15", "push r14", "push r13", "push r12",
        "push r11", "push r10", "push r9",  "push r8",
        "push rbp",  "push rdi", "push rsi", "push rdx",
        "push rcx",  "push rbx", "push rax",
        "mov rdi, rsp",
        "mov rbx, rsp",
        "and rsp, -16",
        "call {handler}",
        "mov rsp, rbx",
        "pop rax",  "pop rbx",  "pop rcx", "pop rdx",
        "pop rsi",  "pop rdi",  "pop rbp",
        "pop r8",   "pop r9",   "pop r10", "pop r11",
        "pop r12",  "pop r13",  "pop r14", "pop r15",
        "add rsp, 8",
        "iretq",
        handler = sym handler_unhandled_irq,
    );
}

// =============================================================================
// Safe Rust Exception Handlers
// =============================================================================
//
// These are called from the trampolines above via a standard System V call
// (frame pointer in RDI). They must not panic or allocate during a fatal
// exception handler (especially #DF). If the heap or lock state is corrupt,
// those operations will either deadlock or double-fault again.
//
// For the "print and halt" handlers, we call `serial::print()` directly rather
// than going through the SERIAL spinlock. We do this because the spinlock may
// already be held on the faulting CPU, and, during a #DF, the heap and globals
// may be in an inconsistent state

/// Prints a minimal exception diagnostic to the serial port (bypassing the
/// spinlock) and halts the CPU forever. This helper macro centralises the
/// print-and-halt pattern. This is intentionally a macro rather than a
/// function so the inline `asm!` in the `hlt` loop does not require an extra
/// stack frame during a fault.
macro_rules! exception_halt {
    ($msg:expr, $frame:expr) => {{
        hardware_manager::sprint(concat!("\n[EXCEPTION] ", $msg, "\n"));
        print_frame($frame);
        cpu_halt();
    }};
}

/// Dumps the key fields of an `InterruptFrame` to the serial port.
///
/// Called from the `exception_halt!` macro. Uses `serial::print()` directly
/// to avoid any dependency on the global spinlock-protected `SERIAL` singleton,
/// which may be locked or corrupted at the time of the exception.
/// 
/// # Arguments
/// 
/// * `frame` - The given `InterruptFrame` to print fields from.
fn print_frame(frame: &InterruptFrame) {
    hardware_manager::sprint("----- Register Dump -----\n");
    print_u64_field("  RIP    = ", frame.rip);
    print_u64_field("  CS     = ", frame.cs);
    print_u64_field("  RFLAGS = ", frame.rflags);
    print_u64_field("  RSP    = ", frame.rsp);
    print_u64_field("  SS     = ", frame.ss);
    print_u64_field("  ERR    = ", frame.error_code);
    print_u64_field("  RAX    = ", frame.rax);
    print_u64_field("  RBX    = ", frame.rbx);
    print_u64_field("  RCX    = ", frame.rcx);
    print_u64_field("  RDX    = ", frame.rdx);
    print_u64_field("  RSI    = ", frame.rsi);
    print_u64_field("  RDI    = ", frame.rdi);
    print_u64_field("  RBP    = ", frame.rbp);
    print_u64_field("  R8     = ", frame.r8);
    print_u64_field("  R9     = ", frame.r9);
    print_u64_field("  R10    = ", frame.r10);
    print_u64_field("  R11    = ", frame.r11);
    print_u64_field("  R12    = ", frame.r12);
    print_u64_field("  R13    = ", frame.r13);
    print_u64_field("  R14    = ", frame.r14);
    print_u64_field("  R15    = ", frame.r15);
    hardware_manager::sprint("----- End Dump -----\n");
}

/// Disables interrupts and halts the CPU in an infinite loop.
///
/// `#[inline(never)]` prevents the compiler from inlining this into each
/// handler site.
#[inline(never)]
fn cpu_halt() -> ! {
    loop {
        unsafe {
            asm!(
                "cli",
                "hlt",
                options(nomem, nostack),
            );
        }
    }
}

/// Triage point for recoverable CPU exceptions. Determines whether the fault
/// originated in a normal task context or in kernel/idle context, and responds
/// accordingly.
///
/// If a normal task is to blame:
///   - The fault details are logged;
///   - The task is marked `Dead` via `kill_current_task()`;
///   - The `dead_task_reaper` SystemTask is enqueued for tombstone cleanup;
///   - `schedule()` is called to immediately switch to the next ready task, and
///   this call never returns to the faulting task.
///
/// If the fault occurred in kernel or idle context:
///   - The kernel halts unconditionally, as this indicates a kernel bug.
///
/// This function is diverging because all execution paths either switch away
/// via `schedule()` or halt the CPU, and neither returns to the caller.
/// 
/// # Arguments
/// 
/// * `reason` - Message string detailing why the task is being killed.
/// * `frame`  - The `InterruptFrame` from the calling exception handler.
fn try_kill_current_task(reason: &str, frame: &InterruptFrame) -> ! {
    let task_id = crate::task_scheduler::get_current_task_id();

    if task_id != crate::task_scheduler::TaskId::IDLE {
        // The fault came from a normal task, so we kill it and move on, while
        // logging enough detail to diagnose the fault post-mortem if needed.
        print_u64_field("\n[TASK FAULT] Task ", task_id.slot_index as u64);
        hardware_manager::sprint(" killed due to: ");
        hardware_manager::sprint(reason);
        hardware_manager::sprint("\n");
        print_frame(frame);

        // Mark the task as `Dead` and enqueue the reaper. This releases the
        // scheduler lock before returning, so `schedule()` below does not
        // deadlock trying to acquire it.
        crate::task_scheduler::kill_current_task();

        // Switch away from the faulting task immediately. The exception stub's
        // iretq frame is orphaned on the dead task's stack, and the reaper will
        // free that stack on the next timer tick, long after we have switched
        // away from it.
        unsafe { crate::task_scheduler::schedule() };

        // `schedule()` switches the stack and never returns to a Dead task. If
        // we ever land here, fault loudly.
        unreachable!(
            "try_kill_current_task: schedule() returned to a dead task");
    } else {
        // The fault occurred in kernel or idle context; this is a kernel bug,
        // and not a task fault. We must halt unconditionally.
        hardware_manager::sprint("\n[KERNEL FAULT] ");
        hardware_manager::sprint(reason);
        hardware_manager::sprint(" in kernel context. Halting...\n");
        print_frame(frame);
        cpu_halt();
    }
}

/// Vector 0: #DE Divide Error
extern "C" fn handler_divide_error(frame: &InterruptFrame) {
    try_kill_current_task("#DE Divide Error", frame);
}

/// Vector 1: #DB Debug Exception
/// 
/// #DB is a trap (IF stays enabled); used by software debuggers. For now we
/// just print and halt; a future debugger integration can replace this.
extern "C" fn handler_debug(frame: &InterruptFrame) {
    exception_halt!("#DB Debug Exception", frame);
}

/// Vector 2: NMI Non-Maskable Interrupt
///
/// NMI cannot be masked with CLI. It signals catastrophic hardware errors,
/// where we must not re-enable interrupts and must simply halt.
extern "C" fn handler_nmi(frame: &InterruptFrame) {
    exception_halt!("NMI Non-Maskable Interrupt", frame);
}

/// Vector 3: #BP Breakpoint
///
/// #BP is a trap gate (IF stays enabled). INT 3 is the standard debugger
/// break mechanism. This implementation prints the frame and continues
/// (returns normally), which causes execution to resume at the instruction
/// after the INT 3. A future debugger can intercept this and spin waiting for
/// a "continue" command instead.
extern "C" fn handler_breakpoint(frame: &InterruptFrame) {
    hardware_manager::sprint("\n[BREAKPOINT] INT3 hit\n");
    print_frame(frame);
    // This intentionally returns (does not halt), so that execution continues
}

/// Vector 4: #OF Overflow
extern "C" fn handler_overflow(frame: &InterruptFrame) {
    try_kill_current_task("#OF Overflow", frame);
}

/// Vector 5: #BR BOUND Range Exceeded
extern "C" fn handler_bound_range(frame: &InterruptFrame) {
    try_kill_current_task("#BR BOUND Range Exceeded", frame);
}

/// Vector 6: #UD Invalid Opcode
extern "C" fn handler_invalid_opcode(frame: &InterruptFrame) {
    try_kill_current_task("#UD Invalid Opcode", frame);
}

/// Vector 7: #NM Device Not Available (no FPU/coprocessor)
extern "C" fn handler_device_not_available(frame: &InterruptFrame) {
    try_kill_current_task("#NM Device Not Available (FPU)", frame);
}

/// Vector 8: #DF Double Fault
/// 
/// The double-fault handler must not return. Its error code is always 0.
/// It runs on IST1 (the dedicated double-fault stack), so even a corrupt
/// kernel stack is safe.
extern "C" fn handler_double_fault(frame: &InterruptFrame) {
    // Unconditionally print via direct serial access - at this point the
    // kernel stack and heap may be corrupted. We cannot trust any spinlock
    // or dynamic allocation.
    hardware_manager::sprint(
        "\n\n[DOUBLE FAULT] Kernel double fault - halting.\n");
    hardware_manager::sprint("Error code (always 0): 0\n");
    print_frame(frame);
    cpu_halt();
}

/// Vector 9: Coprocessor Segment Overrun (legacy, never fires on modern CPUs)
extern "C" fn handler_coproc_seg_overrun(frame: &InterruptFrame) {
    exception_halt!("Coprocessor Segment Overrun (legacy)", frame);
}

/// Vector 10: #TS Invalid TSS
extern "C" fn handler_invalid_tss(frame: &InterruptFrame) {
    hardware_manager::sprint("\n[EXCEPTION] #TS Invalid TSS\n");
    hardware_manager::sprint("  Selector causing fault = ");
    print_u64_field("", frame.error_code);
    print_frame(frame);
    cpu_halt();
}

/// Vector 11: #NP Segment Not Present
extern "C" fn handler_segment_not_present(frame: &InterruptFrame) {
    hardware_manager::sprint("\n[EXCEPTION] #NP Segment Not Present\n");
    hardware_manager::sprint("  Selector = ");
    print_u64_field("", frame.error_code);
    print_frame(frame);
    cpu_halt();
}

/// Vector 12: #SS Stack-Segment Fault
extern "C" fn handler_stack_fault(frame: &InterruptFrame) {
    hardware_manager::sprint("\n[EXCEPTION] #SS Stack-Segment Fault\n");
    print_u64_field("  Selector = ", frame.error_code);

    // Will print the full frame
    try_kill_current_task("#SS Stack-Segment Fault", frame);
}

/// Vector 13: #GP General Protection Fault
/// 
/// The error code encodes the segment selector involved (or 0 for non-segment
/// faults). Bit 0 set = external event; bit 1 set = IDT selector; bit 2 =
/// LDT; bits 15:3 = selector index.
extern "C" fn handler_general_protection(frame: &InterruptFrame) {
    hardware_manager::sprint("\n[EXCEPTION] #GP General Protection Fault\n");
    let ec = frame.error_code;

    if ec == 0 {
        hardware_manager::sprint("  (no specific segment involved)\n");
    }
    else {
        print_u64_field("  Error code (selector info): ", ec);
    }

    // Will print the full frame
    try_kill_current_task("#GP General Protection Fault", frame);
}

/// Vector 14: #PF Page Fault
/// 
/// The faulting linear address is in CR2. The error code flags are:
///   bit 0  P  - 0: non-present page  1: protection violation
///   bit 1  W  - 0: read              1: write
///   bit 2  U  - 0: supervisor        1: user
///   bit 3  R  - 1: reserved PTE bit set
///   bit 4  I  - 1: instruction fetch
///   bit 5  PK - 1: protection-key violation
///   bit 6  SS - 1: shadow-stack access
extern "C" fn handler_page_fault(frame: &InterruptFrame) {
    // Read the faulting virtual address from CR2
    let cr2: u64;
    unsafe { asm!("mov {}, cr2", out(reg) cr2, options(nomem, nostack)) };
    let ec = frame.error_code;

    hardware_manager::sprint("\n[EXCEPTION] #PF Page Fault\n");
    hardware_manager::sprint("  Faulting address: ");
    print_u64_field("", cr2);
    hardware_manager::sprint("  Error code:       ");
    print_u64_field("", ec);

    // Decode the error code for easier debugging
    if ec & (1 << 0) == 0 {
        hardware_manager::sprint("  Reason: non-present page\n");
    }
    else {
        hardware_manager::sprint("  Reason: protection violation\n");
    }

    if ec & (1 << 1) != 0 {
        hardware_manager::sprint("  Access: write\n");
    }
    else {
        hardware_manager::sprint("  Access: read\n");
    }

    if ec & (1 << 2) != 0 {
        hardware_manager::sprint("  Mode:   user\n");
    }
    else {
        hardware_manager::sprint("  Mode:   supervisor\n");
    }

    if ec & (1 << 3) != 0 {
        hardware_manager::sprint("  Note:   reserved PTE bit set\n");
    }

    if ec & (1 << 4) != 0 {
        hardware_manager::sprint("  Note:   instruction fetch\n");
    }

    try_kill_current_task("#PF Page Fault", frame); // Will print the full frame
}

/// Vector 15: Reserved
extern "C" fn handler_reserved(frame: &InterruptFrame) {
    exception_halt!("Reserved / Unimplemented Exception", frame);
}

/// Vector 16: #MF x87 FPU Floating-Point Error
extern "C" fn handler_x87_fpu(frame: &InterruptFrame) {
    try_kill_current_task("#MF x87 FPU Floating-Point Error", frame);
}

/// Vector 17: #AC Alignment Check
extern "C" fn handler_alignment_check(frame: &InterruptFrame) {
    hardware_manager::sprint("\n[EXCEPTION] #AC Alignment Check\n");
    print_u64_field("  Error code: ", frame.error_code);

    // Will print the full frame
    try_kill_current_task("#AC Alignment Check", frame);
}

/// Vector 18: #MC Machine Check
/// 
/// Machine Check is a non-maskable, non-resumable hardware error. We cannot
/// trust memory at this point and must halt.
extern "C" fn handler_machine_check(_frame: &InterruptFrame) {
    // Avoid dereferencing the frame, as the hardware state may be compromised
    hardware_manager::sprint("\n[EXCEPTION] #MC Machine Check - halting.\n");
    cpu_halt();
}

/// Vector 19: #XM SIMD Floating-Point Exception
extern "C" fn handler_simd_exception(frame: &InterruptFrame) {
    try_kill_current_task("#XM SIMD Floating-Point Exception", frame);
}

/// Vector 20: #VE Virtualization Exception (VMX EPT violation)
extern "C" fn handler_virtualization(frame: &InterruptFrame) {
    exception_halt!("#VE Virtualization Exception", frame);
}

/// Vector 21: #CP Control Protection Exception (CET shadow-stack violation)
extern "C" fn handler_control_protection(frame: &InterruptFrame) {
    exception_halt!("#CP Control Protection Exception", frame);
}

// Vectors 22–27: Reserved
// Handled by handler_reserved above (shared stub)

/// Vector 28: #HV Hypervisor Injection Exception (AMD SEV-ES)
extern "C" fn handler_hypervisor_injection(frame: &InterruptFrame) {
    exception_halt!("#HV Hypervisor Injection Exception", frame);
}

/// Vector 29: #VC VMM Communication Exception (AMD SEV-ES)
extern "C" fn handler_vmm_communication(frame: &InterruptFrame) {
    exception_halt!("#VC VMM Communication Exception", frame);
}

/// Vector 30: #SX Security Exception
extern "C" fn handler_security_exception(frame: &InterruptFrame) {
    hardware_manager::sprint("\n[EXCEPTION] #SX Security Exception\n");
    print_u64_field("  Error code: ", frame.error_code);
    print_frame(frame);
    cpu_halt();
}

/// Unhandled hardware IRQ vectors
extern "C" fn handler_unhandled_irq(_frame: &InterruptFrame) {
    hardware_manager::sprint("[IRQ] Unhandled hardware interrupt from APIC!\n");
}

/// Vector 0x20: APIC Timer
extern "C" fn handler_apic_timer(_frame: &InterruptFrame) {
    // Increment tick count first so any task that reads TIMER_TICKS after
    // being woken this tick sees the updated value.
    crate::globals::TIMER_TICKS.fetch_add(1, Ordering::Relaxed);

    // Process the system clock tick and check if the task that is currently
    // running has exhausted its quantum and should be preempted.
    let mut should_schedule = crate::task_scheduler::on_timer_tick();

    // Drain all pending `SystemTask`s before yielding to the next normal task.
    // This is the sole drain point; system tasks run here at elevated
    // priority, with interrupts disabled (as we are inside an IRQ handler),
    // before any normal task gets the CPU.
    crate::system_core::drain_system_tasks();

    // Check if a SystemTask has set the force reschedule flag, and reset it
    if crate::globals::SYS_FLAG_FORCE_RESCHEDULE.load(Ordering::Relaxed) {
        should_schedule = true;
        crate::globals::SYS_FLAG_FORCE_RESCHEDULE.store(
            false, Ordering::SeqCst);
    }

    // Send EOI before schedule(). This re-arms the APIC for the next tick.
    unsafe { crate::hardware_manager::eoi(); }

    if should_schedule {
        // Hand control to the scheduler. If there is another ready task, this
        // call does not return until we are scheduled again. If no other task
        // is ready, it returns immediately and we fall through to iretq
        // normally.
        unsafe { crate::task_scheduler::schedule(); }
    }
}

/// Vector 0x21: PS/2 Keyboard (routed via I/O APIC IRQ 1)
extern "C" fn handler_apic_keyboard(_frame: &InterruptFrame) {
    // We must read the scancode before EOI, or the keyboard controller may not
    // assert another IRQ for the next keypress.
    let scancode: u8;
    unsafe {
        // Read scancode into local variable
        core::arch::asm!("in al, 0x60", out("al") scancode,
            options(nomem, nostack));

        // Send End-of-Interrupt (EOI) to the APIC
        hardware_manager::eoi();
    }

    // Push raw scancode into the ring and wake the keyboard task.
    // process_scancode is now called by the keyboard task, not here.
    hardware_manager::keyboard_push_scancode(scancode);
}

/// Vector 0xFF: Spurious Interrupt
///
/// A spurious interrupt occurs when the CPU acknowledges an interrupt that
/// was already resolved. Per the APIC spec, we must not send EOI for a
/// spurious interrupt, as doing so would signal completion of a real interrupt
/// that is still pending, corrupting the APIC's internal state.
extern "C" fn handler_apic_spurious(_frame: &InterruptFrame) {
    // Intentionally no EOI.
}

// =============================================================================
// IDT Initialization
// =============================================================================

/// Populates all 256 IDT entries and loads the table into the CPU via `lidt`.
///
/// The call order is important here:
///   1. `gdt::init_gdt()` must be called first (the IDT entries reference the
///      KERNEL_CS selector, which must be valid in the GDT);
///   2. `idt::init_idt()` installs the handlers;
///   3. The caller may then execute `sti` to enable interrupts.
///
/// # Safety
/// 
/// Must be called once, during single-threaded kernel init, before `sti`.
pub unsafe fn init_idt() {
    unsafe {
        // `&raw mut` produces a raw pointer directly without forming a
        // mutable reference.
        let idt = &raw mut IDT.entries;

        // CPU Exception Vectors 0–21:
        //
        // Breakpoint (#BP) and Debug (#DB) use trap gates so IF stays enabled,
        // allowing debugger tooling to work. All other exceptions use interrupt
        // gates (IF cleared on entry).
        // The double-fault handler uses IST=1 for a guaranteed-valid stack.
        (*idt)[0]  = IdtEntry::interrupt_gate(stub_de    as unsafe extern "C" fn() as u64, 0);
        (*idt)[1]  = IdtEntry::trap_gate     (stub_db    as unsafe extern "C" fn() as u64, 0);
        (*idt)[2]  = IdtEntry::interrupt_gate(stub_nmi   as unsafe extern "C" fn() as u64, 0);
        (*idt)[3]  = IdtEntry::trap_gate     (stub_bp    as unsafe extern "C" fn() as u64, 0);
        (*idt)[4]  = IdtEntry::interrupt_gate(stub_of    as unsafe extern "C" fn() as u64, 0);
        (*idt)[5]  = IdtEntry::interrupt_gate(stub_br    as unsafe extern "C" fn() as u64, 0);
        (*idt)[6]  = IdtEntry::interrupt_gate(stub_ud    as unsafe extern "C" fn() as u64, 0);
        (*idt)[7]  = IdtEntry::interrupt_gate(stub_nm    as unsafe extern "C" fn() as u64, 0);
        (*idt)[8]  = IdtEntry::interrupt_gate(stub_df    as unsafe extern "C" fn() as u64, 1);
        (*idt)[9]  = IdtEntry::interrupt_gate(stub_cso   as unsafe extern "C" fn() as u64, 0);
        (*idt)[10] = IdtEntry::interrupt_gate(stub_ts    as unsafe extern "C" fn() as u64, 0);
        (*idt)[11] = IdtEntry::interrupt_gate(stub_np    as unsafe extern "C" fn() as u64, 0);
        (*idt)[12] = IdtEntry::interrupt_gate(stub_ss    as unsafe extern "C" fn() as u64, 0);
        (*idt)[13] = IdtEntry::interrupt_gate(stub_gp    as unsafe extern "C" fn() as u64, 0);
        (*idt)[14] = IdtEntry::interrupt_gate(stub_pf    as unsafe extern "C" fn() as u64, 0);
        (*idt)[15] = IdtEntry::interrupt_gate(stub_rsv15 as unsafe extern "C" fn() as u64, 0);
        (*idt)[16] = IdtEntry::interrupt_gate(stub_mf    as unsafe extern "C" fn() as u64, 0);
        (*idt)[17] = IdtEntry::interrupt_gate(stub_ac    as unsafe extern "C" fn() as u64, 0);
        (*idt)[18] = IdtEntry::interrupt_gate(stub_mc    as unsafe extern "C" fn() as u64, 0);
        (*idt)[19] = IdtEntry::interrupt_gate(stub_xm    as unsafe extern "C" fn() as u64, 0);
        (*idt)[20] = IdtEntry::interrupt_gate(stub_ve    as unsafe extern "C" fn() as u64, 0);
        (*idt)[21] = IdtEntry::interrupt_gate(stub_cp    as unsafe extern "C" fn() as u64, 0);

        // Reserved Vectors 22–27:
        (*idt)[22] = IdtEntry::interrupt_gate(stub_rsv22 as unsafe extern "C" fn() as u64, 0);
        (*idt)[23] = IdtEntry::interrupt_gate(stub_rsv23 as unsafe extern "C" fn() as u64, 0);
        (*idt)[24] = IdtEntry::interrupt_gate(stub_rsv24 as unsafe extern "C" fn() as u64, 0);
        (*idt)[25] = IdtEntry::interrupt_gate(stub_rsv25 as unsafe extern "C" fn() as u64, 0);
        (*idt)[26] = IdtEntry::interrupt_gate(stub_rsv26 as unsafe extern "C" fn() as u64, 0);
        (*idt)[27] = IdtEntry::interrupt_gate(stub_rsv27 as unsafe extern "C" fn() as u64, 0);

        // Vectors 28–31:
        (*idt)[28] = IdtEntry::interrupt_gate(stub_hv    as unsafe extern "C" fn() as u64, 0);
        (*idt)[29] = IdtEntry::interrupt_gate(stub_vc    as unsafe extern "C" fn() as u64, 0);
        (*idt)[30] = IdtEntry::interrupt_gate(stub_sx    as unsafe extern "C" fn() as u64, 0);
        (*idt)[31] = IdtEntry::interrupt_gate(stub_rsv31 as unsafe extern "C" fn() as u64, 0);

        // Hardware IRQ Vectors 32–255:
        let mut i = 32usize;
        while i < 256 {
            (*idt)[i] = IdtEntry::interrupt_gate(
                stub_irq_generic as unsafe extern "C" fn() as u64, 0);
            i += 1;
        }

        // APIC-specific vectors replace the generic stubs set by the loop above
        (*idt)[0x20] = IdtEntry::interrupt_gate(
            stub_apic_timer    as unsafe extern "C" fn() as u64, 0);
        (*idt)[0x21] = IdtEntry::interrupt_gate(
            stub_apic_keyboard as unsafe extern "C" fn() as u64, 0);
        (*idt)[0xFF] = IdtEntry::interrupt_gate(
            stub_apic_spurious as unsafe extern "C" fn() as u64, 0);

        // Build and load the IDTR pseudo-descriptor
        let idtr = IdtDescriptor {
            limit: (core::mem::size_of::<Idt>() - 1) as u16,
            base:  core::ptr::addr_of!(IDT) as u64,
        };

        asm!(
            "lidt [{idtr}]",
            idtr = in(reg) &idtr as *const IdtDescriptor as u64,
        );
    }
}
