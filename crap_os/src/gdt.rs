//! Global Descriptor Table (GDT) and Task State Segment (TSS)
//!
//! This module defines the systems's permanent GDT and TSS, which replace the
//! temporary 3-entry GDT set up by the Memory Manager as a placeholder.
//!
//! The permanent GDT adds a TSS descriptor, which gives us the following:
//!   1. A dedicated stack for the double-fault exception handler via Interrupt
//!      Stack Table (IST), so that a fault caused by a corrupted or overflowed
//!      stack does not cause an additional fault (which would lead to triple
//!      fault and imminent CPU reset).
//!   2. A kernel-mode stack pointer (RSP0) for transitions from ring 3 to
//!      ring 0.
//!
//! The GDT has the following layout:
//!
//!  Index | Offset | Selector | Description
//!  ---------------------------------------------------------------------------
//!  0     | 0x00   | 0x00     | Null descriptor (required; CPU ignores slot 0)
//!  1     | 0x08   | 0x08     | 64-bit ring 0 kernel code (execute/read)
//!  2     | 0x10   | 0x10     | 64-bit ring 0 kernel data (read/write)
//!  3     | 0x18   | 0x18     | TSS ring 0 descriptor (low 8 bytes)
//!  4     | 0x20   | -        | TSS ring 0 descriptor (high 8 bytes)
//!  5     | 0x28   | 0x2B     | 64-bit ring 3 user data | (OR'd) 0x3 RPL
//!  6     | 0x30   | 0x33     | 64-bit ring 3 user code | (OR'd) 0x3 RPL
//!
//! Two 8-byte slots at index 3 and 4 form the single 16-byte TSS descriptor.
//! The selector 0x18 refers to both, and index 4 (0x20) is not itself a
//! selector. The selectors at indices 5 and 6 for user mode differ from their
//! respective offsets (unlike kernel segments) because they have to be OR'd
//! with the RPL value of 0x3. Kernel-mode selectors are basically OR'd with 0.
//!
//! The IST1 (Interrupt Stack Table 1) points to the top of DOUBLE_FAULT_STACK.
//! When a double-fault fires, the CPU atomically switches RSP to IST1 before
//! invoking the handler, guaranteeing a known-good stack even if the original
//! RSP was invalid.
//!
//! We need a static mut for the GDT and TSS because the CPU accesses both
//! tables by raw virtual address, via GDTR for the GDT and the TSS base field
//! in the GDT for the TSS. There is no Rust reference involved here; therefore,
//! no spinlock can protect them. But thankfully, spinlocks are not needed.
//!
//! Both tables are written exactly once, during single-threaded init before any
//! interrupts fire. They are read-only for the rest of the kernel's lifetime;
//! the CPU reads them on exception entry, but never writes through a Rust
//! reference. The GDT and TSS are required to have stable virtual addresses for
//! the lifetime of the kernel. A spinlock-protected heap allocation would
//! satisfy the stability requirement, but it would add unnecessary locking
//! overhead.

use crate::globals::{TSS, GDT, DOUBLE_FAULT_STACK};

// =============================================================================
// Double-Fault IST Stack
// =============================================================================

/// Size of the dedicated double-fault IST stack in bytes.
///
/// The 16 KB stack size is reserved for stack traces in the future and calling
/// into formatting code, both of which can use significant stack space. But,
/// the double-fault handler itself is very small, as it just prints diagnostics
/// and then halts the CPU.
pub const DOUBLE_FAULT_IST_SIZE: usize = 4096 * 4;  // 16 KB

/// Physical storage wrapper for the double-fault IST stack.
///
/// The x86-64 System V ABI requires RSP to be 16-byte aligned on entry to any
/// function. The IST pointer stored in the TSS must point to the top (highest
/// address) of this array.
///
/// # Safety
/// 
/// Written (zeroed) exactly once during `init_gdt()`, in single-threaded init.
/// After that, only the CPU writes to this memory (as a hardware stack on
/// double-fault entry). Rust code never touches it again.
#[repr(C, align(16))]
pub struct IstStack(pub [u8; DOUBLE_FAULT_IST_SIZE]);

// =============================================================================
// Task State Segment (TSS)
// =============================================================================

/// The x86-64 Task State Segment structure.
///
/// In 64-bit long mode, the CPU no longer uses hardware task-switching, which
/// was a 32-bit mechanism. The TSS survives in 64-bit mode solely to
/// provide two things:
///
///   1. Privilege-level stack pointers (RSP0–RSP2): when an exception or
///      interrupt occurs while the CPU is in ring 3 (user mode), the CPU reads
///      the appropriate RSPn field from the TSS and switches the stack before
///      pushing the interrupt frame. This is how the kernel always has a valid
///      stack to handle interrupts, even when user code may have corrupted RSP.
///
///   2. Interrupt Stack Table (IST): Seven additional stack pointers
///      (IST1–IST7). An IDT entry can specify an IST slot (1–7); when that
///      interrupt fires, the CPU unconditionally switches RSP to the IST
///      pointer regardless of what privilege level was interrupted. This is
///      used for the double-fault handler to guarantee a known-good stack even
///      if RSP itself is corrupted.
#[repr(C, packed)]
pub struct Tss {
    /// Reserved by the architecture; the CPU never reads this field.
    _reserved0:  u32,

    /// RSP0–RSP2: kernel stack pointers for ring 0/1/2.
    ///
    /// The CPU loads RSP from `rsp[0]` (RSP0) when transitioning from ring 3
    /// to ring 0 (e.g., on every syscall or user-mode interrupt). RSP1 and
    /// RSP2 are for transitions to rings 1 and 2 respectively.
    pub rsp: [u64; 3],

    /// Reserved; the CPU never reads this field.
    _reserved1:  u64,

    /// IST1–IST7: interrupt stack table pointers.
    ///
    /// Indexed 0–6 in this array (corresponding to IST1–IST7 in the Intel
    /// manual). Each entry holds the top (highest address) of a dedicated
    /// stack.
    ///
    /// An IDT gate entry's IST field (bits 2:0 of the flags byte) selects
    /// which IST pointer the CPU uses:
    ///   0   = do not use IST; instead, use the normal RSPn mechanism
    ///   1   = use IST1 (our `ist[0]`), used for the double-fault handler
    ///   2-7 = use IST7 (our `ist[6]`)
    pub ist: [u64; 7],

    /// Reserved; the CPU never reads this field.
    _reserved2:  u64,

    /// Reserved; the CPU never reads this field.
    _reserved3:  u16,

    /// Byte offset from the start of the TSS to the I/O Permission Bitmap.
    ///
    /// The IOPB is a hardware feature that lets the kernel grant or deny
    /// individual in/out instructions to user-mode code on a per-port basis.
    /// We do not use it, so this is set to `size_of::<Tss>()` (one past the
    /// end of the struct), which the CPU interprets as "no IOPB present" and
    /// denies all I/O port access from user mode.
    pub iopb_offset: u16,
}

impl Tss {
    /// Creates a zeroed TSS with `iopb_offset` set to `size_of::<Tss>()`.
    ///
    /// All RSP and IST entries are zero until populated by `init_gdt()`.
    /// `const fn` allows this to be used as a `static` initializer.
    pub const fn new() -> Self {
        Self {
            _reserved0: 0,
            rsp: [0; 3],
            _reserved1: 0,
            ist: [0; 7],
            _reserved2: 0,
            _reserved3: 0,
            iopb_offset: core::mem::size_of::<Tss>() as u16,  // "No IOPB"
        }
    }
}

// =============================================================================
// GDT Descriptor Encoding
// =============================================================================

/// A single 8-byte GDT entry (segment descriptor).
///
/// On x86-64, most segment descriptors are 8 bytes. The TSS descriptor is
/// the notable exception: it is a system segment descriptor that is 16 bytes
/// (two consecutive `GdtEntry` slots, addressed by a single selector).
///
/// `repr(transparent)` means this struct has exactly the same layout as its
/// single `u64` field, so we can safely cast between `GdtEntry` and `u64`.
#[repr(transparent)]
#[derive(Copy, Clone)]
pub struct GdtEntry(pub u64);

impl GdtEntry {
    /// The null descriptor (GDT slot 0).
    ///
    /// The x86-64 architecture requires slot 0 to be all zeros. Loading a
    /// null selector (0x00) into a data-segment register is legal, but loading
    /// it into CS triggers a GP (General Protection) fault.
    pub const NULL: Self = Self(0);

    /// 64-bit ring-0 code segment descriptor. Raw value: `0x00AF9A000000FFFF`.
    pub const KERNEL_CODE64: Self = Self(0x00AF9A000000FFFF);

    /// 64-bit ring-0 data segment descriptor. Raw value: `0x00CF92000000FFFF`.
    pub const KERNEL_DATA64: Self = Self(0x00CF92000000FFFF);

    /// 64-bit ring-3 code segment descriptor. Raw value: `0x00AFFA000000FFFF`.
    /// Same as KERNEL_CODE64 but with DPL=3 (bits 45-46 set):
    /// FA = `1111_1010` -> Present | DPL=3 | Code | Execute/Read
    pub const USER_CODE64: Self = Self(0x00AFFA000000FFFF);

    /// 64-bit ring-3 data segment descriptor. Raw value: `0x00CFF2000000FFFF`.
    /// Same as `KERNEL_DATA64`, but with DPL=3 (bits 45-46 set):
    /// F2 = `1111_0010` -> Present | DPL=3 | Data | Read/Write
    pub const USER_DATA64: Self = Self(0x00CFF2000000FFFF);

    /// Encodes the low 8-byte half of a 64-bit TSS descriptor.
    ///
    /// Unlike code/data descriptors, TSS descriptors are system segment
    /// descriptors and occupy two consecutive 8-byte GDT slots. The
    /// TSS base address is split across the two halves because the low half
    /// re-uses the legacy 8-byte descriptor format (whose base field is only
    /// 32 bits wide); the high half carries the upper 32 bits of the base.
    ///
    /// # Arguments
    /// 
    /// * `base`  - Virtual address of the `TSS` struct.
    /// * `limit` - `size_of::<Tss>() - 1` (the CPU adds 1 internally).
    pub fn tss_low(base: u64, limit: u32) -> Self {
        // Extract the sub-fields of `limit` and `base` needed for the low half

        // Low 16 bits of the limit (placed in descriptor bits 15:0)
        let limit_low = (limit & 0xFFFF) as u64;
        // High 4 bits of the limit (placed in descriptor bits 51:48)
        let limit_high = ((limit >> 16) & 0xF) as u64;

        // Base bits [23:0] - fits in the lower three bytes of the base field.
        // The mask 0x00FF_FFFF keeps only the bottom 24 bits.
        let base_low = (base & 0x00FF_FFFF) as u64;

        // Base bits [31:24] - the next byte, placed in descriptor bits 63:56
        let base_mid = ((base >> 24) & 0xFF) as u64;

        // Assemble all sub-fields using bit shifts and OR.
        // The shifts place each sub-field at its correct bit position in u64.
        let value =
            limit_low             // Bits 15:0  - limit[15:0]
            | (base_low << 16)    // Bits 39:16 - base[23:0]
            | (0b1001_u64 << 40)  // Bits 43:40 - 64-bit available TSS
            | (1_u64 << 47)       // Bit  47    - P=1 (present)
            | (limit_high << 48)  // Bits 51:48 - limit[19:16]
            | (base_mid << 56);   // Bits 63:56 - base[31:24]

        Self(value)
    }

    /// Encodes the high 8-byte half of a 64-bit TSS descriptor.
    ///
    /// The high half exists solely to extend the base address from 32 to 64
    /// bits. Its layout is:
    ///
    ///  Bits of u64  | Content
    ///  -------------------------------------------------------------------
    ///  31:0         | base[63:32] - upper 32 bits of the TSS base address
    ///  63:32        | Reserved; must be zero (CPU may raise #GP if non-zero)
    ///
    /// # Arguments
    /// 
    /// * `base` - The same virtual address passed to `tss_low`.
    pub fn tss_high(base: u64) -> Self {
        // Shift away the lower 32 bits that were encoded in the low half.
        // The result naturally fits in 32 bits, and the upper 32 bits are 0.
        Self(base >> 32)
    }
}

// =============================================================================
// Global Descriptor Table
// =============================================================================

/// The kernel's permanent 5-entry GDT.
///
/// GDT Entries are:
///   [0] - Null descriptor        (selector 0x00, required by architecture)
///   [1] - 64-bit kernel code     (selector 0x08)
///   [2] - 64-bit kernel data     (selector 0x10)
///   [3] - TSS descriptor low     (selector 0x18,  the lower 8 bytes of TSS)
///   [4] - TSS descriptor high    (not a selector; the upper 8 bytes of TSS)
///   [5] - 64-bit user data       (selector 0x2B = 0x28 | 0x03)
///   [6] - 64-bit user code       (selector 0x33 = 0x30 | 0x03)
///
/// The x86-64 requirement is that the GDT be 8-byte aligned.
#[repr(C, align(8))]
pub struct Gdt {
    pub entries: [GdtEntry; 7],
}

// =============================================================================
// Selector Constants
// =============================================================================
//
// A segment selector is a 16-bit value interpreted as:
//   Bits [15:3] - Index into the GDT (or LDT).
//   Bit  [2]    - TI (Table Indicator): 0 = GDT, 1 = LDT.
//   Bits [1:0]  - RPL (Requested Privilege Level): 0 = ring 0, 3 = ring 3.
//
// All kernel selectors use TI=0 (GDT) and RPL=0 (ring 0).

/// Kernel code segment selector.
/// GDT index 1 -> byte offset 8 -> 0x08. Used in CS after lgdt.
pub const KERNEL_CS: u16 = 0x08;

/// Kernel data segment selector.
/// GDT index 2 -> byte offset 16 -> 0x10. Loaded into DS/ES/SS/FS/GS.
pub const KERNEL_DS: u16 = 0x10;

/// TSS selector.
/// GDT index 3 -> byte offset 24 -> 0x18. Loaded into TR via `ltr`.
pub const TSS_SELECTOR: u16 = 0x18;

/// User data segment selector.
/// GDT index 5 -> byte offset 40 -> 0x28 | RPL 3 (OR'd with 0x03).
pub const USER_DS: u16 = 0x2B;

/// User code segment selector.
/// GDT index 6 -> byte offset 48 -> 0x30 | RPL 3 (OR'd with 0x03).
pub const USER_CS: u16 = 0x33;

/// The 10-byte GDTR descriptor structure consumed by the `lgdt` instruction.
///
/// `lgdt` expects a pointer to exactly this layout in memory:
///   bytes [0..1] - limit: (size of GDT in bytes) − 1
///   bytes [2..9] - base:  64-bit virtual address of the GDT array
#[repr(C, packed)]
struct GdtDescriptor {
    /// Size of the GDT in bytes, minus 1. The CPU adds 1 to get the true size,
    /// so a 5-entry GDT (40 bytes) has limit = 39.
    limit: u16,

    /// Virtual address of the first byte of the GDT array. The CPU uses this
    /// to translate GDT indices (from segment selectors) into descriptor addresses.
    base:  u64,
}

/// Initializes and loads the permanent GDT and TSS.
///
/// After this function returns:
///   - The CPU uses our 5-entry GDT for all segment descriptor lookups
///   - CS, DS, ES, SS, FS, GS are all loaded with our kernel selectors
///   - The Task Register (TR) points to our TSS
///   - Double-fault exceptions will switch to the IST1 stack before firing
///
/// The following steps are performed:
///   1. Set `TSS.ist[0]` (IST1) to the top of `DOUBLE_FAULT_STACK`
///   2. Patch the two TSS descriptor slots in `GDT` with the TSS address/limit
///   3. Populate user-mode descriptors
///   4. Build a `GdtDescriptor` pointing at `GDT`
///   5. Execute `lgdt` to load the GDTR, activating the new GDT
///   6. Perform a far return (`retfq`) to atomically load CS = `KERNEL_CS`
///   7. Reload DS, ES, SS, FS, GS with `KERNEL_DS`
///   8. Execute `ltr` to load the Task Register with `TSS_SELECTOR`
///
/// # Safety
/// 
/// The caller must ensure:
///   - Called exactly once, during single-threaded kernel initialization.
///   - Interrupts are disabled (via CLI) before calling. The IDT has not been
///     loaded yet, so any interrupt or exception between step 4 and the end of
///     this function would likely cause a triple fault.
///   - No other code is concurrently modifying `GDT`, `TSS`, or
///     `DOUBLE_FAULT_STACK`.
pub unsafe fn init_gdt() {
    // -------------------------------------------------------------------------
    // Step 1: Initialize IST1 in the TSS to the top of the double-fault stack.
    //
    // `core::ptr::addr_of!` is used instead of a normal `&` reference to
    // avoid creating a Rust shared reference to a `static mut`. Under the
    // `static_mut_refs` lint, even creating a `&` to a `static mut` for the
    // purpose of obtaining its address is considered unsound because the
    // reference could alias a mutable use elsewhere. `addr_of!` produces a raw
    // `*const` pointer directly from the place expression without going
    // through a reference.
    // -------------------------------------------------------------------------
    let df_stack_base = unsafe {
        core::ptr::addr_of!(DOUBLE_FAULT_STACK.0) as u64
    };
    let df_stack_top  = df_stack_base + DOUBLE_FAULT_IST_SIZE as u64;

    // IST slots are 1-indexed in Intel documentation (IST1–IST7), but
    // our array is 0-indexed, so IST1 = ist[0].
    unsafe { TSS.ist[0] = df_stack_top };

    // -------------------------------------------------------------------------
    // Step 2: Encode the TSS address and size into the GDT.
    //
    // The TSS descriptor is a 16-byte system descriptor formed by two
    // consecutive 8-byte GDT slots ([3] = low half, [4] = high half).
    // Both halves must be written before `ltr` is executed; writing them
    // after would leave the GDT in a partially invalid state.
    //
    // `size_of::<Tss>() - 1` is the limit value (CPU adds 1 to get the size).
    // -------------------------------------------------------------------------
    let tss_base  = core::ptr::addr_of!(TSS) as u64;
    let tss_limit = (core::mem::size_of::<Tss>() - 1) as u32;

    unsafe {
        GDT.entries[3] = GdtEntry::tss_low(tss_base, tss_limit);
        GDT.entries[4] = GdtEntry::tss_high(tss_base);
    }

    // -------------------------------------------------------------------------
    // Step 3: Populate user-mode descriptors.
    //
    // `USER_DATA64` must be at index 5 and `USER_CODE64` at index 6. SYSRET
    // requires that the user `CS = (STAR MSR upper 16 bits) + 16`, and user
    // `SS = (STAR MSR upper 16 bits) + 8`.
    unsafe {
        GDT.entries[5] = GdtEntry::USER_DATA64;
        GDT.entries[6] = GdtEntry::USER_CODE64;
    }

    // -------------------------------------------------------------------------
    // Steps 4–6: Build the GDTR, load it, and reload CS via far return.
    //
    // The `limit` field is `size_of::<Gdt>() - 1` = 5 * 8 − 1 = 39.
    //
    // Far return (RETFQ) is the standard way to reload CS in long mode:
    //   * We push the new CS selector onto the stack.
    //   * We push the address of the instruction after RETFQ onto the stack.
    //   * RETFQ atomically pops RIP (the return address) then CS (the selector).
    //   * Execution continues at label "3:" with the new CS active.
    //
    // The GdtDescriptor is a local stack variable. It is safe to point GDTR
    // at a stack address because the CPU caches the GDT base/limit in an
    // internal register after `lgdt`; the memory is only read during `lgdt`
    // itself, not on every descriptor access.
    // -------------------------------------------------------------------------
    let gdtr = GdtDescriptor {
        limit: (core::mem::size_of::<Gdt>() - 1) as u16,
        base:  core::ptr::addr_of!(GDT) as u64,
    };

    core::arch::asm!(
        // Step 4 & 5: Load the GDTR from the gdtr struct on the stack.
        // After this instruction, the CPU uses our new GDT for all subsequent
        // segment descriptor lookups, but CS still holds the old selector.
        "lgdt [{gdtr}]",

        // Step 6: Far return to atomically reload CS:
        "push {cs}",              // the new code segment selector
        "lea {tmp}, [rip + 3f]",  // RIP-relative: compute address of "3:"
        "push {tmp}",             // the return RIP
        "retfq",                  // far return: pops RIP, then CS

        // Execution resumes here with CS = KERNEL_CS (0x08).
        "3:",

        // Step 7: Reload all data-segment registers.
        // These are not updated by the far return above; RETFQ only affects CS.
        // The `mov reg, ax` form is used because segment registers can only be
        // loaded from a general-purpose register, not from an immediate value.
        "mov ax, {ds}",
        "mov ds, ax",   // Data segment
        "mov es, ax",   // Extra segment (legacy; used by string instructions)
        "mov ss, ax",   // Stack segment (affects RSP privilege checking)
        "mov fs, ax",   // FS (used for TLS/per-CPU data in future)
        "mov gs, ax",   // GS (used for per-CPU kernel data in future)

        // Operand constraints:
        gdtr = in(reg) &gdtr as *const GdtDescriptor as u64,
    
        // `const` allows the selector values to be encoded as immediates.
        cs = const KERNEL_CS as u64,
        ds = const KERNEL_DS as u64,
    
        // `lateout` means `tmp` may alias an input register; it's written
        // after all inputs are consumed. `_` discards the final value.
        tmp  = lateout(reg) _,
    
        // RAX is clobbered as a result.
        out("rax") _,
    );

    // -------------------------------------------------------------------------
    // Step 8: Load the Task Register (TR) with the TSS selector.
    //
    // `ltr` does two things:
    //   1. Loads the Task Register with the given selector, making the CPU
    //      use the pointed-to TSS for RSP0/IST lookups on exception entry;
    //   2. Marks the TSS descriptor in the GDT as "busy" (sets bit 41 of the
    //      descriptor's type field from 0 to 1: 0b1001 -> 0b1011).
    //
    // The "busy" mark exists so the CPU can detect invalid nested task-switches
    // (a relic of 32-bit hardware task switching, harmless in 64-bit mode).
    // It must be set before any exception or interrupt fires, otherwise the
    // CPU has no TSS to load IST pointers from.
    // -------------------------------------------------------------------------
    core::arch::asm!(
        "ltr ax",
        in("ax") TSS_SELECTOR,
    );
}

/// Updates `TSS.rsp[0]` to point to the given kernel stack top.
///
/// This must be called on every context switch to the incoming task, before
/// that task starts executing. The CPU reads `TSS.rsp[0]` on every ring-3  to
/// ring-0 transition (interrupts, syscalls, etc.) to find the kernel stack, so
/// it must always reflect the currently running task's kernel stack.
/// 
/// # Arguments
/// 
/// * `stack_top` - The given kernel stack top to update `TSS.rsp[0]` with.
#[inline]
pub fn set_kernel_stack(stack_top: u64) {
    unsafe { TSS.rsp[0] = stack_top };

    // Verification: read back and confirm the write took effect.
    // Remove once step 6 (user threads) is working.
    unsafe {
        let rsp0 = core::ptr::addr_of!(TSS.rsp[0]).read_unaligned();
        debug_assert_eq!(
            rsp0, stack_top,
            "TSS.rsp[0] write did not take effect"
        );
    }
}
