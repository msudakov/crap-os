//! Local APIC (Advanced Programmable Interrupt Controller) and I/O APIC Driver
//!
//! This module configures and operates the two APIC components, replacing the
//! legacy 8259 PIC pair that UEFI leaves active by default. The two APICs
//! coordinate interrupts like this:
//!
//!  +-------------------------------------------------------------------------+
//!  |  Hardware device (keyboard, timer, PCI, etc.)                           |
//!  |         |                                                               |
//!  |         |  IRQ line                                                     |
//!  |         V                                                               |
//!  |   +----------+  Redirection    +------------+  IPI / vector   +-----+   |
//!  |   | I/O APIC | ------------->  | Local APIC | --------------> | CPU |   |
//!  |   +----------+                 +------------+                 +-----+   |
//!  +-------------------------------------------------------------------------+
//!
//!  I/O APIC (one per motherboard interrupt controller, MMIO at 0xFEC00000):
//!    - Receives external hardware IRQ lines from devices;
//!    - Routes each IRQ to a chosen Local APIC vector via a 64-bit redirection
//!      table entry (one entry per IRQ pin);
//!    - Programmed indirectly through an index/data register pair (IOREGSEL /
//!      IOWIN) rather than direct register access.
//!
//!  Local APIC (one per CPU core, MMIO at 0xFEE00000 by default):
//!    - Receives interrupt messages from the I/O APIC and delivers them to the
//!      core as vectors (0x00–0xFF);
//!    - Hosts the per-CPU APIC timer, used for periodic scheduling ticks;
//!    - Must be explicitly signalled at the end of each interrupt via the EOI
//!      register, and forgetting this silently masks all further interrupts of
//!      the same or lower priority class;
//!    - Manages spurious interrupts, which are phantom interrupts the hardware
//!      can generate; the APIC spec requires a dedicated "spurious vector"
//!      handler.
//!
//! ----------------------------------------------------------------------------
//! Access model
//! ----------------------------------------------------------------------------
//!
//! Both APICs are memory-mapped I/O (MMIO) devices. All reads and writes must
//! use `read_volatile` / `write_volatile` to prevent the compiler from
//! optimizing, reordering, or combining accesses to their registers.
//!
//! After `init_apic`, LAPIC_BASE and IOAPIC_BASE hold virtual addresses,
//! translated through the kernel's direct physical map. They are initially
//! set to their default physical values as a compile-time fallback, but are
//! overwritten on the first call to `init_apic` with the ACPI-derived addresses
//! converted to virtual.
//!
//! ----------------------------------------------------------------------------
//! Interrupt vector layout
//! ----------------------------------------------------------------------------
//!
//!   0x00–0x1F  CPU exceptions           (defined by Intel architecture)
//!   0x20       APIC timer               (VECTOR_TIMER)
//!   0x21       PS/2 keyboard, I/O IRQ 1 (VECTOR_KEYBOARD)
//!   0xFF       Spurious interrupt       (VECTOR_SPURIOUS - required by APIC)
//!
//! Vectors 0x22–0xFE are free for future devices.
//!
//! ----------------------------------------------------------------------------
//! Initialization sequence
//! ----------------------------------------------------------------------------
//!
//!   1. Call `disable_pic_8259()` to reinitialize and mask the legacy PIC, so
//!      its IRQs do not conflict with APIC vectors;
//!   2. Call `init_apic(lapic_phys, ioapic_phys)` with the addresses from ACPI;
//!   3. Load the IDT with handlers for VECTOR_TIMER, VECTOR_KEYBOARD, and
//!      VECTOR_SPURIOUS;
//!   4. Call `configure_timer(count)` to start the periodic APIC timer;
//!   5. Enable the keyboard by calling `ioapic_unmask_irq(1)` from the keyboard
//!      driver once it is ready to handle interrupts;
//!   6. Execute `sti` to globally enable interrupts.

use crate::memory_manager::MemoryManager;
use core::ptr::{read_volatile, write_volatile};
use core::sync::atomic::{AtomicBool, Ordering};

/// Vector delivered to the CPU when the APIC timer fires.
pub const VECTOR_TIMER: u8 = 0x20;

/// Vector delivered to the CPU when the PS/2 keyboard fires (I/O APIC IRQ 1).
pub const VECTOR_KEYBOARD: u8 = 0x21;

/// Spurious interrupt vector, required by the APIC specification.
///
/// The APIC can generate a spurious interrupt (no real hardware event) if the
/// CPU masks an interrupt at just the wrong moment. The APIC spec mandates a
/// dedicated vector for these. The spurious handler must NOT call `eoi()`, so
/// spurious interrupts are never acknowledged by the APIC itself.
pub const VECTOR_SPURIOUS: u8 = 0xFF;

// =============================================================================
// Local APIC register offsets (ref. Intel SDM Vol. 3A, Sec. 10.4, "Local APIC")
// =============================================================================
//
// The Local APIC exposes its registers as 32-bit values at fixed byte offsets
// from LAPIC_BASE. Despite being 32-bit registers, they are spaced 16 bytes
// apart (aligned to 128-bit / 16-byte boundaries) on the original xAPIC
// hardware. We therefore express offsets in bytes and cast to *const/*mut u32.

/// Local APIC ID register (read-only).
/// Bits [27:24] (for xAPIC) contain the APIC hardware ID of this core.
const _LAPIC_REG_ID: u32 = 0x020;

/// Local APIC Version register (read-only).
/// Bits [7:0] = version, bits [23:16] = (max_LVT_entry - 1).
const _LAPIC_REG_VER: u32 = 0x030;

/// Local APIC Task Priority Register (TPR).
///
/// Controls which interrupt priority classes are accepted by this LAPIC.
/// Priority = vector[7:4]; the LAPIC only delivers interrupts whose priority
/// is strictly greater than TPR[7:4]. Setting TPR = 0 accepts all vectors, as
/// priority 0 is the lowest possible threshold.
const LAPIC_TPR: u32 = 0x080;

/// End-Of-Interrupt register (write-only).
///
/// Writing any value (conventionally 0) to this register tells the Local APIC
/// that the current interrupt has been fully handled. The APIC then:
///   - Clears the corresponding bit in its In-Service Register (ISR);
///   - Allows lower-priority interrupts to be delivered;
///   - For level-triggered I/O APIC entries: sends a de-assert message to
///     the I/O APIC, so it can accept the next assertion of that IRQ line.
///
/// This register must be written at the end of every APIC interrupt handler
/// except the spurious vector handler.
const LAPIC_EOI: u32 = 0x0B0;

/// Spurious Interrupt Vector Register.
///
/// Bit 8 (APIC software-enable): when 0, the APIC is in software-disabled
/// state and does not deliver interrupts. Must be set to 1 to enable.
/// Bits [7:0]: the spurious interrupt vector number (must end in 0xF).
const LAPIC_SPURIOUS: u32 = 0x0F0;

/// LVT (Local Vector Table) Timer Register.
///
/// Controls how the APIC timer fires:
///  Bits  [7:0]  - interrupt vector delivered when the timer fires.
///  Bit   [16]   - mask: 1 = timer interrupt disabled, 0 = enabled.
///  Bits [18:17] - timer mode: 00 = one-shot, 01 = periodic, 10 = TSC-deadline.
const LAPIC_LVT_TIMER: u32 = 0x320;

/// Timer Initial Count register.
///
/// The APIC timer counts down from this value to zero using the bus/core
/// crystal clock divided by the Timer Divide Configuration (DCR) value. When
/// it reaches zero:
///   - One-shot mode: fires once, stops.
///   - Periodic mode: fires, reloads Initial Count, and continues.
/// Writing this register (re-)starts the countdown.
const LAPIC_TIMER_INITIAL_COUNT: u32 = 0x380;

/// APIC register offset for the timer's current (live countdown) value.
/// 
/// This is a read-only snapshot of how far the timer has counted down
/// from its initial value. It is distinct from LAPIC_TIMER_INITIAL_COUNT
/// (0x380), which is write-only and sets the starting value.
const LAPIC_TIMER_CURRENT_COUNT: u32 = 0x390;

/// Timer Divide Configuration Register (DCR).
///
/// Sets the divisor applied to the bus clock before feeding the APIC timer
/// counter. The divisor controls the timer resolution vs. range trade-off:
/// a larger divisor gives coarser ticks but a longer maximum interval.
const LAPIC_TIMER_DCR: u32 = 0x3E0;

/// LVT timer mode bits for one-shot operation.
/// 
/// In one-shot mode, the timer counts down from INITIAL_COUNT to zero once and
/// then stops. We use this during calibration, so the counter never wraps or
/// reloads, and we just measure how far it fell in a known wall-clock window.
const LVT_TIMER_ONE_SHOT: u32 = 0;

/// LVT mask bit (bit 16).
/// 
/// When set, the timer interrupt is suppressed, and the countdown still runs,
/// but no interrupt vector is delivered. We mask during calibration, so the
/// half-initialized one-shot timer doesn't accidentally fire into the normal
/// timer ISR and corrupt scheduler state while we are mid-measurement.
const LVT_TIMER_MASKED: u32 = 1 << 16;

/// How many milliseconds to run the calibration measurement window.
/// 
/// 10ms is long enough that APIC counter granularity error is negligible (a
/// few ticks of error over ~630,000 ticks is < 0.001%), and it is short enough
/// to not meaningfully delay boot.
const CALIBRATION_MS: u64 = 10;

// =============================================================================
// Local APIC register flag bits
// =============================================================================

/// Bit 8 of the Spurious Interrupt Vector Register.
/// When set, the Local APIC is software-enabled and will deliver interrupts.
/// When clear, all interrupt delivery is suppressed regardless of other
/// settings.
const LAPIC_SPURIOUS_ENABLE: u32 = 1 << 8;

/// LVT timer mode bit for periodic mode.
/// When set in LAPIC_LVT_TIMER bits [18:17], the timer reloads and fires again
/// after each expiry instead of stopping after the first expiry.
const LVT_TIMER_PERIODIC: u32 = 1 << 17;

/// LVT mask bit (bit 16 of any LVT register).
/// When set, the corresponding interrupt source is masked (disabled).
/// When clear, the interrupt is unmasked (enabled).
const LVT_MASKED: u32 = 1 << 16;

/// DCR value for divide-by-16.
///
/// The mapping of DCR register values to actual divisors is non-linear
/// (ref. Intel SDM Vol. 3A Table 10-2); 0x3 corresponds to ÷16.
/// With a typical ~100 MHz bus clock: tick interval ≈ 16 / 100,000,000 s
/// per count unit, so an initial count of 1,000,000 -> ~160 ms per interrupt.
const TIMER_DIVIDE_BY_16: u32 = 0x3;

// =============================================================================
// The I/O APIC uses indirect register access, where we write the register index
// to IOREGSEL (IOAPIC_BASE + 0x00), then read/write the data via IOWIN
// (IOAPIC_BASE + 0x10). The indices below are used with ioapic_read/write.
// The below IOAPIC_REG_ID, IOAPIC_REG_VER, IOAPIC_REG_REDIRECT_TABLE, and
// IOAPIC_MASKED I/O APIC register indices are found in the Intel I/O APIC
// specification, Section 3, "Register Description".
// =============================================================================

/// I/O APIC ID register index. Bits [27:24] contain the I/O APIC hardware ID.
const _IOAPIC_REG_ID: u32 = 0x00;

/// I/O APIC Version register index.
/// Bits  [7:0]  - I/O APIC version (typically, it is 0x11 or 0x20).
/// Bits [23:16] - max_redir_entry: the number of redirection table entries
///                supported by this I/O APIC, minus 1. Used in `init_io_apic`
///                to determine how many entries to mask at startup.
const IOAPIC_REG_VER: u32 = 0x01;

/// Index of the low 32-bit word of redirection table entry 0.
///
/// The redirection table starts at index 0x10. Each 64-bit entry occupies
/// two consecutive 32-bit register slots:
///   Low  word: index 0x10 + 2 * IRQ number
///   High word: index 0x11 + 2 * IRQ number
///
/// So IRQ 0 -> low=0x10, high=0x11;
///    IRQ 1 -> low=0x12, high=0x13;
const IOAPIC_REG_REDIRECT_TABLE: u32 = 0x10;

/// Mask bit in the low 32-bit word of an I/O APIC redirection table entry.
/// Bit 16: 1 = interrupt masked (disabled), 0 = unmasked (enabled).
const IOAPIC_MASKED: u32 = 1 << 16;

/// Virtual base address of the Local APIC MMIO register block.
/// Defaults to the standard physical address 0xFEE00000; overwritten in
/// `init_apic` with the ACPI-provided address translated to virtual.
static mut LAPIC_BASE: u64 = 0xFEE00000;

/// Virtual base address of the I/O APIC MMIO register block.
/// Defaults to the standard physical address 0xFEC00000; overwritten in
/// `init_apic` with the ACPI-provided address translated to virtual.
static mut IOAPIC_BASE: u64 = 0xFEC00000;

/// Set to `true` after `init_apic` completes successfully.
/// Uses `Ordering::Release` on the store so all preceding MMIO writes are
/// visible to any thread that observes `APIC_INITIALIZED == true` with an
/// `Acquire` load.
static APIC_INITIALIZED: AtomicBool = AtomicBool::new(false);

/// Reads the 32-bit Local APIC register at a byte offset from LAPIC_BASE.
///
/// # Arguments
/// 
/// * `reg_offset` - Byte offset from the LAPIC_BASE address.
/// 
/// # Returns
/// 
/// Returns the 32-bit value from the Local APIC register.
///
/// # Safety
/// 
/// LAPIC_BASE must be a valid mapped virtual address (set by `init_apic`).
#[allow(dead_code)]
#[inline]
unsafe fn lapic_read(reg_offset: u32) -> u32 {
    // Compute the absolute virtual address of the register
    let addr = unsafe { (LAPIC_BASE + reg_offset as u64) as *const u32 };
    unsafe { read_volatile(addr) }
}

/// Writes a value to the 32-bit Local APIC register at a byte offset.
/// 
/// # Arguments
/// 
/// * `reg_offset` - Byte offset from the LAPIC_BASE address.
/// * `value`      - Value to write.
///
/// # Safety
/// 
/// LAPIC_BASE must be a valid mapped virtual address (set by `init_apic`).
#[inline]
unsafe fn lapic_write(reg_offset: u32, value: u32) {
    let addr = unsafe { (LAPIC_BASE + reg_offset as u64) as *mut u32 };
    unsafe { write_volatile(addr, value) };
}

/// Reads the 32-bit I/O APIC register identified by an index.
///
/// The I/O APIC uses an indirect access protocol:
///   1. Write the register index to IOREGSEL (IOAPIC_BASE + 0x00).
///   2. Read the register value from IOWIN   (IOAPIC_BASE + 0x10).
///
/// # Arguments
/// 
/// * `reg_index` - Register index to read from.
/// 
/// # Returns
/// 
/// Returns the 32-bit value from the I/O APIC register.
/// 
/// # Safety
///
/// IOAPIC_BASE must be a valid mapped virtual address (set by `init_apic`).
#[inline]
unsafe fn ioapic_read(reg_index: u32) -> u32 {
    // Write the register index to the IOREGSEL register (offset 0x00)
    let regsel = unsafe { IOAPIC_BASE as *mut u32 };
    unsafe { write_volatile(regsel, reg_index) };

    // Read the data from the IOWIN register (offset 0x10)
    let iowin = unsafe { (IOAPIC_BASE + 0x10) as *const u32 };
    unsafe { read_volatile(iowin) }
}

/// Writes a value to the 32-bit I/O APIC register identified by an index,
/// using the same two-step indirect protocol as `ioapic_read`: write index to
/// IOREGSEL, then write the value to IOWIN.
///
/// # Arguments
///
/// * `reg_index`  - Register index to write to.
/// * `value`      - Value to write.
///
/// # Safety
///
/// IOAPIC_BASE must be a valid mapped virtual address.
#[inline]
unsafe fn ioapic_write(reg_index: u32, val: u32) {
    // Select the target register
    let regsel = unsafe { IOAPIC_BASE as *mut u32 };
    unsafe { write_volatile(regsel, reg_index) };

    // Write the new value
    let iowin = unsafe { (IOAPIC_BASE + 0x10) as *mut u32 };
    unsafe { write_volatile(iowin, val) };
}

/// Writes a full 64-bit I/O APIC redirection table entry as two 32-bit writes.
///
/// The high word contains the destination APIC ID; the low word contains the
/// vector, delivery mode, and the mask bit. We write the high word first to
/// avoid a window where the low word is updated with a new (unmasked) vector
/// while the high word still holds a stale destination. That could briefly
/// route the interrupt to the wrong CPU.
///
/// # Arguments
/// * `irq`   - The I/O APIC IRQ pin number (0-based).
/// * `value` - The 64-bit redirection table entry to write.
///
/// # Safety
/// IOAPIC_BASE must be valid. `irq` must be within the I/O APIC's supported
/// range (checked by the caller - this function does not range-check).
#[inline]
unsafe fn ioapic_write_redirect(irq: u8, value: u64) {
    // Low-word register index: IOAPIC_REG_REDIRECT_TABLE + 2 * irq
    let reg_low = IOAPIC_REG_REDIRECT_TABLE + 2 * irq as u32;
    let reg_high = reg_low + 1;  // High-word register index: reg_low + 1

    // Write high word first (destination field) to avoid stale routing.
    unsafe { ioapic_write(reg_high, (value >> 32) as u32) };

    // Write low word last (vector + control bits including mask).
    unsafe { ioapic_write(reg_low, value as u32) };
}

/// Re-initializes and fully masks the legacy dual-8259 PIC.
///
/// UEFI firmware leaves the 8259 PIC enabled and mapped to IRQ vectors that
/// overlap with our APIC vector assignments. If the PIC is not remapped and
/// masked before enabling the APIC, spurious PIC interrupts would be delivered
/// as CPU exceptions (vectors 0x00–0x0F overlap with CPU faults in the default
/// PIC mapping, causing instant triple faults or mis-handled exceptions).
///
/// Simply masking the PIC's IMR (Interrupt Mask Register) is not sufficient on
/// all hardware, as some BIOS/UEFI implementations leave the PIC in an
/// undefined state. The safest approach is a full reinitialization sequence
/// (ICW1–ICW4) with a harmless vector remapping, followed by masking all lines.
///
/// Sending ICW1 to the PIC's command port puts it into initialization mode.
/// The PIC then expects three more bytes in order (ICW2, ICW3, ICW4) on its
/// data port before it resumes normal operation.
///
/// This is the I/O port map:
///   0x20 - Master PIC command port
///   0x21 - Master PIC data / IMR port
///   0xA0 - Slave  PIC command port
///   0xA1 - Slave  PIC data / IMR port
///
/// # Safety
/// 
/// Must be called before enabling the APIC, with interrupts disabled, to
/// prevent any PIC interrupt from firing between PIC re-init and masking.
pub unsafe fn disable_pic_8259() {
    // ICW1: Begin initialization of both PICs
    // 0x11 = ICW1_INIT (bit 4) | ICW1_ICW4 (bit 0): edge-triggered, cascade,
    // expect ICW4 to follow. Sent to the command ports of both master and
    // slave.
    core::arch::asm!("out 0x20, al", in("al") 0x11u8);  // Master: start init
    core::arch::asm!("out 0xA0, al", in("al") 0x11u8);  // Slave:  start init

    // ICW2: Set interrupt vector base for each PIC
    // Maps master PIC IRQs 0–7  -> vectors 0x20–0x27 (harmless; we mask them).
    // Maps slave PIC IRQs  8–15 -> vectors 0x28–0x2F (harmless; we mask them).
    // These ranges are above the CPU exception vectors (0x00–0x1F) and do not
    // conflict with our APIC assignments, so if a PIC interrupt fires before
    // the mask is applied it will hit an unused IDT entry rather than an
    // exception.
    core::arch::asm!("out 0x21, al", in("al") 0x20u8);  // Master base: 0x20
    core::arch::asm!("out 0xA1, al", in("al") 0x28u8);  // Slave  base: 0x28

    // ICW3: Configure cascade (master/slave connection)
    // Master: bit mask indicating which IRQ line (IRQ 2) is connected to slave.
    //   0x04 = 0b00000100: slave is on IRQ line 2.
    // Slave:  its cascade identity (the IRQ line it is connected to on master).
    //   0x02 = binary cascade identity 2.
    core::arch::asm!("out 0x21, al", in("al") 0x04u8);  // Master: slave on IRQ2
    core::arch::asm!("out 0xA1, al", in("al") 0x02u8);  // Slave: cascade ID=2

    // ICW4: Set 8086/8088 operation mode
    // 0x01 = 8086 mode (non-buffered, normal EOI, not special fully-nested).
    // Required by modern x86 systems; 8085 mode (0x00) is for legacy hardware.
    core::arch::asm!("out 0x21, al", in("al") 0x01u8);  // Master: 8086 mode
    core::arch::asm!("out 0xA1, al", in("al") 0x01u8);  // Slave:  8086 mode

    // OCW1: Mask all IRQ lines on both PICs
    // Writing 0xFF to the data port masks all 8 IRQ lines of each PIC. After
    // this, no 8259 interrupt will reach the CPU regardless of what hardware
    // asserts an IRQ line.
    core::arch::asm!("out 0x21, al", in("al") 0xFFu8);  // Master: mask all
    core::arch::asm!("out 0xA1, al", in("al") 0xFFu8);  // Slave:  mask all
}

/// Performs the initial configuration of the Local APIC.
///
/// After this function:
///   - The APIC is software-enabled and will deliver interrupts;
///   - All interrupt priority levels are accepted (TPR = 0);
///   - The spurious vector is set to VECTOR_SPURIOUS (0xFF);
///   - The timer LVT entry is masked (timer is off until `configure_timer`).
unsafe fn init_local_apic() {
    unsafe {
        // Set TPR (Task Priority Register) to 0, so the APIC accepts interrupts
        // of any priority level. A non-zero TPR would suppress all vectors
        // whose priority class (vector[7:4]) is <= TPR[7:4].
        lapic_write(LAPIC_TPR, 0);

        // Write the Spurious Interrupt Vector Register:
        //   Bits [7:0] = VECTOR_SPURIOUS (0xFF) - the spurious vector number;
        //                Must have bits [3:0] all set; 0xFF satisfies this.
        //   Bit  [8]   = LAPIC_SPURIOUS_ENABLE  - software-enables the APIC.
        //
        // This single write both assigns the spurious vector and brings the
        // APIC out of software-disabled state.
        lapic_write(LAPIC_SPURIOUS,
            VECTOR_SPURIOUS as u32 | LAPIC_SPURIOUS_ENABLE);

        // Mask the timer LVT entry so no timer interrupt fires until
        // `configure_timer` is called with the desired interval.
        lapic_write(LAPIC_LVT_TIMER, LVT_MASKED);
    }
}

/// Performs the initial configuration of the I/O APIC.
///
/// After this function:
///   - All redirection table entries are masked (no IRQs will fire).
///   - IRQ 1 (PS/2 keyboard) is programmed to deliver VECTOR_KEYBOARD on
///     LAPIC 0, edge-triggered, active-high - but remains masked.
///     Call `ioapic_unmask_irq(1)` when the keyboard driver is ready.
unsafe fn init_io_apic() {
    // Determine the number of supported redirection entries.
    // The I/O APIC Version Register reports the maximum redirection entry index
    // in bits [23:16] as (max_entries - 1). Adding 1 gives the total count.
    // Most I/O APICs support 24 entries (IRQ pins 0–23), but we read the
    // hardware value rather than assuming.
    let ver = unsafe { ioapic_read(IOAPIC_REG_VER) };
    let max_redir = ((ver >> 16) & 0xFF) + 1;  // Total entries supported.

    // Mask all redirection table entries.
    // We start from a clean state by masking every entry, regardless of what
    // the firmware may have programmed. This prevents any external IRQ from
    // firing until we explicitly unmask the lines we want to handle. Only the
    // low 32-bit word contains the mask bit; we read-modify-write to avoid
    // disturbing the high word (destination APIC ID) unnecessarily.
    for irq in 0..max_redir as u8 {
        let reg_lo = IOAPIC_REG_REDIRECT_TABLE + 2 * irq as u32;
        let lo = unsafe { ioapic_read(reg_lo) };

        // Set bit 16 (mask) while preserving all other bits.
        unsafe { ioapic_write(reg_lo, lo | IOAPIC_MASKED) };
    }

    // Program IRQ 1 (PS/2 keyboard) redirection table entry. The 64-bit
    // redirection entry bit-field has the following layout:
    //
    //  Bits    | Field            | Value | Meaning
    //  ------------------------------------------------------------------------
    //  [7:0]   | Vector           | 0x21  | CPU vector delivered on interrupt
    //  [10:8]  | Delivery mode    | 0b000 | Fixed: deliver to destination LAPIC
    //  [11]    | Dest mode        | 0     | Physical: destination is an APIC ID
    //  [12]    | Delivery status  | 0     | Read-only; 0 = idle
    //  [13]    | Polarity         | 0     | Active high (PS/2 keyboard is AH)
    //  [14]    | Remote IRR       | 0     | Read-only; only relevant for level
    //  [15]    | Trigger mode     | 0     | Edge-triggered (PS/2 uses edge)
    //  [16]    | Mask             | 1     | Masked - keyboard driver unmasks
    //  [63:56] | Destination      | 0     | LAPIC ID 0 = bootstrap processor
    //
    // Edge-triggered vs level-triggered:
    //   PS/2 keyboard uses edge-triggered signalling: the IRQ line pulses
    //   briefly when a key event occurs. Edge mode means the I/O APIC fires
    //   once per pulse. Level-triggered (for PCI devices, USB controllers,
    //   etc.) keeps the line asserted until the driver acknowledges the event,
    //   requiring the driver to clear the hardware condition before EOI.
    //
    // We start this entry masked. The keyboard driver calls
    // `ioapic_unmask_irq(1)` after installing its IDT handler, ensuring no
    // keyboard interrupt can fire before the handler is ready.
    let redir_keyboard: u64 =
        (VECTOR_KEYBOARD as u64)    // Bits   [7:0]: vector 0x21
        | (0b000_u64 << 8)          // Bits  [10:8]: Fixed delivery mode
        | (0_u64     << 11)         // Bit     [11]: Physical destination mode
        | (0_u64     << 13)         // Bit     [13]: Active-high polarity
        | (0_u64     << 15)         // Bit     [15]: Edge-triggered
        | (IOAPIC_MASKED as u64)    // Bit     [16]: Start masked
        | (0u64      << 56);        // Bits [63:56]: Destination LAPIC ID 0

    unsafe { ioapic_write_redirect(1, redir_keyboard) };
}

// =============================================================================
// Public API
// =============================================================================

/// Initializes the Local APIC and I/O APIC.
///
/// Translates the ACPI-supplied physical base addresses to virtual addresses,
/// then configures both APICs to a known safe state.
///
/// After this function returns:
///   - The Local APIC is software-enabled and will deliver interrupts;
///   - All I/O APIC redirection table entries are masked, and no external IRQs
///     will fire until explicitly unmasked by the relevant driver;
///   - I/O APIC IRQ 1 (PS/2 keyboard) is programmed but still masked;
///     call `ioapic_unmask_irq(1)` to enable it;
///   - The APIC timer LVT entry is masked; call `configure_timer` to start it.
///
/// # Arguments
/// 
/// * `local_apic_phys` - Physical base address from ACPI MADT. Pass 0 to
///                        fall back to the architectural default (0xFEE00000).
/// * `io_apic_phys`    - Physical base address from ACPI MADT. Pass 0 to
///                        fall back to the architectural default (0xFEC00000).
///
/// # Safety
/// 
/// Must be called exactly once, with interrupts disabled (CLI), during
/// single-threaded kernel initialization. The physical map must be active
/// (i.e., `init_higher_half` must have completed), so that `phys_to_virt`
/// yields valid, accessible addresses.
pub unsafe fn init_apic(local_apic_phys: u64, io_apic_phys: u64) {
    unsafe {
        // Translate the APIC MMIO base addresses from physical to virtual.
        //
        // We must store virtual addresses here, not physical ones. After
        // `remove_identity_maps()` is called in the Memory Manager, the
        // lower-half identity map is torn down, so physical addresses are no
        // longer valid as virtual addresses. The direct physical map
        // (KERNEL_PHYS_MAP_BASE + phys) remains valid for the kernel's
        // lifetime. If the caller passes 0, fall back to the architectural
        // default physical address before translating.
        LAPIC_BASE = if local_apic_phys != 0 {
            MemoryManager::phys_to_virt(local_apic_phys)
        }
        else {
            MemoryManager::phys_to_virt(0xFEE00000)
        };

        IOAPIC_BASE = if io_apic_phys != 0 {
            MemoryManager::phys_to_virt(io_apic_phys)
        }
        else {
            MemoryManager::phys_to_virt(0xFEC00000)
        };

        init_local_apic();
        init_io_apic();
    }

    // Signal that APIC is fully initialized. `Release` ordering ensures all
    // preceding MMIO writes are visible to any thread that observes this flag
    // with an `Acquire` load.
    APIC_INITIALIZED.store(true, Ordering::Release);
}

/// Configures the APIC timer in periodic mode and starts the countdown.
///
/// The timer counts down from `initial_count` to zero using the internal bus
/// clock divided by the DCR divisor (set to ÷16 here). In periodic mode,
/// it reloads and continues automatically, firing `VECTOR_TIMER` on each
/// expiry.
/// 
/// Without calibration against a known-frequency source (PIT or HPET), the
/// tick rate is hardware-dependent. Rough estimate with a ÷16 DCR:
///   - 1 GHz bus clock: count of 1,000,000 -> ~16 ms per tick (~62 Hz).
///   - Higher bus clocks -> shorter intervals per count unit.
/// 
/// TODO: Later on, need to calibrate against the PIT and measure how many APIC
/// timer counts elapse in a known PIT interval, then compute the desired
/// initial count.
/// 
/// # Arguments
/// 
/// * `initial_count` - Initial number of ticks to start the timer from.
///
/// # Safety
/// 
/// Must be called after `init_apic`. Interrupts should be disabled (or the
/// IDT entry for `VECTOR_TIMER` must already be installed) before calling, to
/// prevent a timer interrupt firing before the handler is in place.
pub unsafe fn configure_timer(initial_count: u32) {
    unsafe {
        // Set the clock divisor to ÷16
        lapic_write(LAPIC_TIMER_DCR, TIMER_DIVIDE_BY_16);

        // Configure the LVT timer entry:
        //   Bits [7:0]  = VECTOR_TIMER (0x20) - the vector to deliver
        //   Bit  [17]   = LVT_TIMER_PERIODIC  - reload and repeat
        //   Bit  [16]   = 0 (LVT_MASKED not set) - unmasked, will fire
        lapic_write(LAPIC_LVT_TIMER, VECTOR_TIMER as u32 | LVT_TIMER_PERIODIC);

        // Write the Initial Count register and start the timer
        lapic_write(LAPIC_TIMER_INITIAL_COUNT, initial_count);
    }
}

/// Calibrates the APIC timer against the HPET and returns the number of APIC
/// timer ticks per millisecond.
///
/// The APIC timer runs from the CPU's internal bus or crystal clock, whose
/// frequency is not standardized and varies across hardware and hypervisors.
/// Without calibration, any hardcoded initial count value produces wildly
/// different wall-clock intervals depending on the platform. For example, a
/// count of 1,000,000 might be 1ms on one machine and 4ms on another.
///
/// The HPET counter period is reported in femtoseconds in the ACPI table and
/// is guaranteed to be accurate by the hardware/firmware. We don't need to
/// measure it, and we can trust it directly. This makes the HPET a good fixed
/// reference against which to measure the APIC timer's unknown frequency.
///
/// # Arguments
/// 
/// * `hpet` - The HPET information structure.
///
/// # Returns
/// 
/// Returns APIC timer ticks per millisecond at divide-by-16.
pub unsafe fn calibrate_timer(hpet: &crate::hardware_manager::HpetInfo) -> u32 {
    // Match the divide configuration used by `configure_timer()`, so the
    // returned value is directly usable without any post-calibration scaling.
    unsafe { lapic_write(LAPIC_TIMER_DCR, TIMER_DIVIDE_BY_16) };

    // Configure the APIC timer in one-shot mode, masked, divide-by-16. The
    // counter runs down, but delivers no interrupt. This prevents a spurious
    // vector 0x20 from firing into the scheduler ISR while we are mid-
    // calibration with an otherwise uninitialized timer state.
    unsafe {
        lapic_write(
            LAPIC_LVT_TIMER,
            VECTOR_TIMER as u32 | LVT_TIMER_ONE_SHOT | LVT_TIMER_MASKED,
        )
    };

    // Arm it with the maximum possible initial count (0xFFFF_FFFF), so it will
    // not expire during the measurement window. At ~63,000 ticks/ms, this gives
    // us roughly 68 seconds before the counter would reach zero, which is more
    // than enough.
    unsafe { lapic_write(LAPIC_TIMER_INITIAL_COUNT, 0xFFFF_FFFF) };

    // Convert the calibration window from milliseconds to HPET ticks, which
    // are measured in femtoseconds (1ms = 1_000_000_000_000 fs).
    // This is exact - no floating point, no approximation.
    let hpet_ticks_per_cal =
        (CALIBRATION_MS * 1_000_000_000_000) / hpet.period_fs as u64;

    // Spin on the HPET main counter until CALIBRATION_MS has elapsed. The
    // `wrapping_sub` handles the theoretically possible (but practically
    // irrelevant at 100MHz+) case of the counter wrapping mid-spin.
    let hpet_start = unsafe { hpet.read_counter() };
    loop {
        let elapsed = unsafe { hpet.read_counter() }.wrapping_sub(hpet_start);
        if elapsed >= hpet_ticks_per_cal {
            break;
        }
    }

    // Read the APIC timer's current countdown value. Since it started at
    // 0xFFFFFFFF and has been counting down, the number of ticks elapsed
    // is the difference between the initial value and the current value.
    let apic_end = unsafe { lapic_read(LAPIC_TIMER_CURRENT_COUNT) };
    let apic_ticks_elapsed = 0xFFFF_FFFF_u32.wrapping_sub(apic_end);
    let apic_ticks_per_ms = apic_ticks_elapsed / CALIBRATION_MS as u32;

    // Disarm the timer, and `configure_timer()` will re-arm it in periodic
    // mode with the calibrated value once we return to main.
    unsafe { lapic_write(LAPIC_TIMER_INITIAL_COUNT, 0) };

    apic_ticks_per_ms
}

/// Signals End-Of-Interrupt (EOI) to the Local APIC by writing 0 to the EOI
/// register, which is the required protocol; any other value may cause a #GP
/// on some APIC implementations.
///
/// Must be called at the end of every APIC interrupt handler except the
/// spurious vector handler. Omitting the EOI silently prevents any further
/// interrupt of equal or lower priority from being delivered to this CPU. This
/// type of bug can be difficult to diagnose.
#[inline]
pub unsafe fn eoi() {
    unsafe { lapic_write(LAPIC_EOI, 0) };
}

/// Masks (disables) the specified I/O APIC IRQ line.
///
/// Reads the low word of the redirection table entry, sets bit 16 (the mask
/// bit), and writes it back. The interrupt will not be delivered to any CPU
/// until `ioapic_unmask_irq` is called for this line.
///
/// # Arguments
/// 
/// * `irq` - The I/O APIC IRQ pin number to mask (0-based, typically 0–23).
#[allow(dead_code)]
pub unsafe fn ioapic_mask_irq(irq: u8) {
    let reg = IOAPIC_REG_REDIRECT_TABLE + 2 * irq as u32;
    let lo = unsafe { ioapic_read(reg) };

    // Set bit 16 (IOAPIC_MASKED) while preserving all other fields.
    unsafe { ioapic_write(reg, lo | IOAPIC_MASKED) };
}

/// Unmasks (enables) the specified I/O APIC IRQ line.
///
/// Reads the low word of the redirection table entry, clears bit 16 (the mask
/// bit), and writes it back. From this point, the hardware can deliver
/// interrupts on this IRQ line to the CPU.
///
/// # Arguments
/// 
/// * `irq` - The I/O APIC IRQ pin number to unmask (0-based, typically 0–23).
pub unsafe fn ioapic_unmask_irq(irq: u8) {
    let reg = IOAPIC_REG_REDIRECT_TABLE + 2 * irq as u32;
    let lo = unsafe { ioapic_read(reg) };

    // Clear bit 16 (IOAPIC_MASKED) using a bitwise AND with the complement.
    unsafe { ioapic_write(reg, lo & !IOAPIC_MASKED) };
}
