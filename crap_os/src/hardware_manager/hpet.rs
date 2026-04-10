//! # HPET - High Precision Event Timer
//!
//! This module handles discovery, parsing, and access to the HPET hardware.
//! The High Precision Event Timer is a system-level timer defined by Intel
//! and Microsoft in the IA-PC HPET Architecture Specification. It provides:
//!   - A single monotonically increasing main counter register;
//!   - A hardware-reported counter period in femtoseconds
//!         (exact, no measurement needed);
//!   - One or more comparator channels that can fire interrupts;
//!   - A minimum frequency of 10 MHz (period <= 100ns).
//!
//! We use the HPET as a reference clock for calibrating the APIC timer during
//! kernel initialization. We do not use the HPET comparators or interrupts,
//! and only read the main counter register to measure wall-clock elapsed time.
//!
//! The HPET is a global, stable, high-resolution reference that lets us
//! determine the APIC timer's unknown frequency once at boot. After
//! calibration, the APIC timer drives all scheduling, and the HPET main counter
//! remains available for any future timekeeping needs.
//!
//! The HPET base address is obtained from the HPET ACPI table, which is located
//! by walking the XSDT/RSDT from the RSDP. The table contains a Generic Address
//! Structure (GAS) block that gives us the physical MMIO base address. We map
//! that page and access registers via volatile reads/writes through the fixed
//! kernel physical map offset. All HPET registers are memory-mapped at fixed
//! offsets from the base address.

use core::ptr;
use crate::memory_manager::MemoryManager;
use super::acpi::{find_acpi_table, SdtHeader};

/// This is the HPET-specific body that immediately follows the standard
/// `SdtHeader` in the ACPI HPET table.
///
/// This layout is defined by the HPET specification and the ACPI
/// specification's description of the HPET table. Total size is 20 bytes. The
/// address of the HPET MMIO registers is given by the Generic Address Structure
/// (GAS), embedded in this table. We only support system memory GAS entries.
#[repr(C, packed)]
struct HpetAcpiTable {
    /// Hardware revision ID reported by the HPET block.
    hardware_rev_id: u8,

    /// Packed bitfield describing the HPET block's comparator configuration:
    ///   Bits [4:0] - number of comparators minus one (e.g., 2 = 3 comparators)
    ///   Bit  [5]   - 1 if the main counter is 64-bit capable, 0 if 32-bit only
    ///   Bit  [6]   - reserved
    ///   Bit  [7]   - 1 if legacy replacement (IRQ0/IRQ8) routing is supported
    comparator_info: u8,

    /// PCI vendor ID of the HPET block's logic.
    pci_vendor_id: u16,

    /// Address space identifier in the Generic Address Structure (GAS), where
    /// 0 = system memory (MMIO), and 1 = system I/O (port I/O). The HPET spec
    /// requires system memory for the main counter and configuration registers.
    /// We reject anything other than 0.
    address_space_id: u8,

    /// GAS - width of the register in bits.
    /// We always access HPET registers as 64-bit values as required by the
    /// specification.
    register_bit_width: u8,

    /// GAS - bit offset within the register. It is always 0 for HPET.
    register_bit_offset: u8,

    /// GAS - reserved byte, always 0.
    _reserved: u8,

    /// GAS - physical base address of the HPET MMIO register block. This is
    /// the address of the first HPET register (General Capabilities and ID,
    /// offset 0x000). We map this page and use it as `base_virt` after
    /// translating through `phys_to_virt()`. It is typically 0xFED00000 on
    /// x86-64 systems.
    base_address: u64,

    /// HPET sequence number, used when multiple HPET blocks are present. Most
    /// systems have only one (hpet_number == 0).
    hpet_number: u8,

    /// Minimum tick value in HPET clock periods that the main counter is
    /// guaranteed to advance between successive reads, reported by firmware.
    /// A value of 0 (common in hypervisors) is technically spec-violating,
    /// but harmless since we never use HPET comparators.
    minimum_tick: u16,

    /// Page protection and OEM attribute byte. It describes memory protection
    /// guarantees for the HPET page.
    page_protection: u8,
}

// =============================================================================
// MMIO Register Offsets and Configuration Bits
// =============================================================================

/// General Capabilities and ID Register (read-only, 64-bit).
///
/// Layout:
/// Bits [63:32] - COUNTER_CLK_PERIOD: main counter tick period in femtoseconds.
///                This is the most important field; it tells us exactly how
///                fast the counter runs without any measurement. The spec
///                requires this to be non-zero and <= 100,000,000 fs (100ns),
///                guaranteeing a minimum frequency of 10 MHz.
/// Bits [31:16] - VENDOR_ID: PCI vendor ID
/// Bits [15:13] - reserved
/// Bit  [13]    - COUNT_SIZE_CAP: 1 if main counter is 64-bit, 0 if 32-bit
/// Bits [12:8]  - NUM_TIM_CAP: number of comparator timers minus one
/// Bits [7:0]   - REV_ID: hardware revision
const HPET_REG_CAPABILITIES: usize = 0x000;

/// General Configuration Register (read/write, 64-bit).
///
/// Layout (relevant bits):
/// Bit [0] - ENABLE_CNF: overall enable for the main counter and comparators.
///           Must be 1 for the main counter to advance. Some firmware leaves
///           this cleared at boot, so we check and set it if needed.
/// Bit [1] - LEG_RT_CNF: legacy replacement routing enable. We never set this.
/// All other bits are reserved or comparator-specific.
const HPET_REG_CONFIG: usize = 0x010;

/// Main Counter Value Register (read/write, 64-bit).
///
/// Contains the current value of the free-running main counter. Increments
/// by 1 every COUNTER_CLK_PERIOD femtoseconds. This is the register we read
/// during APIC timer calibration and spin-waits.
const HPET_REG_MAIN_COUNTER: usize = 0x0F0;

/// Overall enable bit in the General Configuration Register (bit 0).
/// When set, the main counter increments and comparator interrupts can fire.
/// When clear, the main counter is frozen.
const HPET_CONFIG_ENABLE: u64 = 1 << 0;

// =============================================================================
// Public Interface
// =============================================================================

/// Collected information about the HPET, populated once during `parse_hpet()`
/// and stored in a global for use throughout the kernel.
///
/// After `parse_hpet()` returns `Some(HpetInfo)`, the HPET main counter is
/// guaranteed to be running and readable via `read_counter()`.
#[allow(dead_code)]
pub struct HpetInfo {
    /// Virtual address of the HPET MMIO base, derived from the physical base
    /// address in the ACPI table via `MemoryManager::phys_to_virt()`.
    /// All register accesses are computed as offsets from this address.
    pub base_virt: u64,

    /// Main counter period in femtoseconds (fs), read directly from the
    /// General Capabilities register at boot. This value is exact and
    /// hardware-reported, so no measurement or approximation is involved.
    pub period_fs: u32,

    /// Number of comparator timers available in this HPET block. Derived from
    /// bits [4:0] of the `comparator_info` byte in the ACPI table (stored as
    /// `num_comparators_minus_one` + 1). It is typically 3 on most systems.
    /// Currently unused.
    pub num_timers: u8,

    /// True if the main counter register is 64-bit wide, false if 32-bit.
    /// Derived from bit [5] of the General Capabilities register.
    /// A 64-bit counter at 100 MHz would take ~5,849 years to wrap.
    /// A 32-bit counter at 100 MHz wraps every ~42 seconds, so it's relevant
    /// for long spin-waits if this is ever false.
    pub counter_64bit: bool,

    /// Minimum tick value from the ACPI table. Represents the minimum number
    /// of femtoseconds between successive reads of the main counter that are
    /// guaranteed to show advancement. A value of 0 (common in hypervisors)
    /// is technically spec-violating, but harmless for our read-only usage.
    pub minimum_tick: u16,
}

#[allow(dead_code)]
impl HpetInfo {
    /// Converts a duration in femtoseconds to the equivalent number of
    /// HPET main counter ticks.
    ///
    /// # Arguments
    /// 
    /// * `fs` - Duration in femtoseconds to convert.
    /// 
    /// # Returns
    /// 
    /// Returns the converted number of HPET ticks.
    #[inline]
    pub fn fs_to_ticks(&self, fs: u64) -> u64 {
        fs / self.period_fs as u64
    }

    /// Converts a number of HPET main counter ticks to femtoseconds.
    ///
    /// # Arguments
    /// 
    /// * `ticks` - Number of HPET ticks to convert.
    /// 
    /// # Returns
    /// 
    /// Returns the converted number of femtoseconds.
    #[inline]
    pub fn ticks_to_fs(&self, ticks: u64) -> u64 {
        ticks * self.period_fs as u64
    }

    /// Converts a number of HPET main counter ticks to nanoseconds, rounded
    /// down.
    ///
    /// Divides by 1,000,000 to convert from femtoseconds to nanoseconds.
    /// For sub-nanosecond HPET periods (period_fs < 1,000,000, i.e. > 1 GHz)
    /// this loses sub-ns precision, which is acceptable for all current use
    /// cases.
    /// 
    /// # Arguments
    /// 
    /// * `ticks` - Number of HPET ticks to convert.
    /// 
    /// # Returns
    /// 
    /// Returns the converted number of nanoseconds.
    #[inline]
    pub fn ticks_to_ns(&self, ticks: u64) -> u64 {
        ticks * self.period_fs as u64 / 1000000
    }

    /// Reads the current value of the HPET main counter register directly
    /// from MMIO.
    /// 
    /// # Returns
    /// 
    /// Returns the current value of the HPET main counter register.
    ///
    /// # Safety
    /// 
    /// Caller must ensure `parse_hpet()` has completed successfully and the
    /// HPET MMIO page is mapped. Both are guaranteed if `HpetInfo` was
    /// obtained from `parse_hpet()`.
    #[inline]
    pub unsafe fn read_counter(&self) -> u64 {
        unsafe {
            ptr::read_volatile(
                (self.base_virt + HPET_REG_MAIN_COUNTER as u64) as *const u64
            )
        }
    }

    /// Spin-waits for at least `ns` nanoseconds using the HPET main counter
    /// as the time reference.
    ///
    /// Converts the requested nanosecond duration to HPET ticks, then polls
    /// `read_counter()` until the counter has advanced by at least that many
    /// ticks from the start value. Uses `wrapping_sub` for the elapsed
    /// calculation to correctly handle counter wraparound on 32-bit HPET
    /// implementations, where the counter rolls over every ~42 seconds at
    /// 100 MHz. On 64-bit counters, wraparound is not a practical concern,
    /// but the wrapping arithmetic is still valid.
    /// 
    /// # Arguments
    /// 
    /// * `ns` - The number of nanoseconds to spin-wait for.
    ///
    /// # Returns
    /// 
    /// Returns the actual number of ticks elapsed, which will be >= the
    /// requested duration, due to loop overhead and timer granularity.
    ///
    /// # Safety
    /// 
    /// The same safety requirements as `read_counter()` above.
    pub unsafe fn spin_wait_ns(&self, ns: u64) -> u64 {
        // Convert nanoseconds to femtoseconds to use fs_to_ticks.
        // 1ns = 1_000_000 fs.
        let ticks_to_wait = self.fs_to_ticks(ns * 1_000_000);
        let start = unsafe { self.read_counter() };
        loop {
            let now = unsafe { self.read_counter() };
            // wrapping_sub gives correct elapsed ticks even if the counter
            // has wrapped around zero between start and now.
            let elapsed = now.wrapping_sub(start);
            if elapsed >= ticks_to_wait {
                return elapsed;
            }
        }
    }
}

/// Locates the HPET via its ACPI table, maps its MMIO region, reads its
/// capabilities, ensures the main counter is running, and returns an
/// `HpetInfo` structure describing the hardware.
///
/// # Arguments
/// 
/// * `rsdp_virt` - RSDP virtual address from the bootloader (already
///       translated through the kernel's direct physical map).
/// 
/// # Returns
/// 
/// Returns `Some(HpetInfo)` on success, or `None` if:
///   - No HPET ACPI table is present;
///   - The GAS address space is not system memory;
///   - The base address is zero;
///   - The reported counter period is zero or exceeds 100ns (spec violation).
///
/// # Safety
/// 
/// The RSDP virtual address must be valid and point to a correctly formed ACPI
/// RSDP structure. The HPET MMIO page (typically 0xFED00000) must be mapped in
/// the page tables before this function is called, as accessing an unmapped
/// MMIO page will cause a page fault.
pub unsafe fn parse_hpet(rsdp_virt: u64) -> Option<HpetInfo> {
    // Walk the XSDT/RSDT and locate the HPET ACPI table by its 4-byte
    // signature "HPET".
    let hpet_sdt = unsafe { find_acpi_table(rsdp_virt, b"HPET")? };

    // Parse the HPET-specific table body that follows the `SdtHeader` to
    // extract the physical MMIO base address from the embedded GAS block. We
    // advance past the `SdtHeader` by adding its size to the table pointer.
    let table_body = unsafe {
        &*((hpet_sdt as usize + size_of::<SdtHeader>()) as *const HpetAcpiTable)
    };

    // Validate that the GAS describes a system memory (MMIO) mapping. We only
    // support MMIO access, which is `address_space_id` of `0`. The ID of `1`
    // would indicate I/O port access, which the HPET spec does not use for the
    // main register block. Any other value indicates malformed firmware, and
    // we must refuse to proceed.
    let address_space_id = unsafe {
        ptr::read_unaligned(core::ptr::addr_of!((*table_body).address_space_id))
    };
    if address_space_id != 0 {
        return None;
    }

    // Extract and validate the physical MMIO base address. A zero base address
    // indicates either absent hardware or malformed firmware. The standard
    // HPET base address on x86-64 is `0xFED00000`, though firmware is free to
    // place it elsewhere within the MMIO address space.
    let base_phys = unsafe {
        ptr::read_unaligned(core::ptr::addr_of!((*table_body).base_address))
    };
    if base_phys == 0 {
        return None;
    }

    // Extract the `minimum_tick` and `comparator_info` fields from the ACPI
    // table.
    let minimum_tick = unsafe {
        ptr::read_unaligned(core::ptr::addr_of!((*table_body).minimum_tick))
    };
    let comparator_info = unsafe {
        ptr::read_unaligned(core::ptr::addr_of!((*table_body).comparator_info))
    };

    // bits [4:0] of `comparator_info` are (`num_comparators` - 1), so we add 1
    // for the actual count. Bit [5] indicates whether the main counter is
    // 64-bit.
    let num_timers = (comparator_info & 0x1F) + 1;
    let counter_64bit = (comparator_info & (1 << 5)) != 0;

    // Map the physical base to a virtual address. The caller is responsible
    // for ensuring the corresponding physical page has been mapped in the page
    // tables. The HPET register block fits within a single 4KB page.
    let base_virt = MemoryManager::phys_to_virt(base_phys);

    // Read the General Capabilities and ID register.
    // This 64-bit register is at MMIO offset 0x000 (the base address itself).
    // The upper 32 bits contain COUNTER_CLK_PERIOD - the main counter's tick
    // period in femtoseconds. This is the single most important value the HPET
    // provides: it tells us exactly how fast the counter runs without any
    // measurement on our part.
    let caps = unsafe {
        ptr::read_volatile(
            (base_virt + HPET_REG_CAPABILITIES as u64) as *const u64)
    };
    let period_fs = (caps >> 32) as u32;

    // Validate the period against the HPET specification requirements:
    //   - Must be non-zero (a zero period would imply infinite frequency)
    //   - Must be <= 100,000,000 fs (100ns), guaranteeing >= 10 MHz
    // A period outside this range indicates either absent/broken hardware or
    // a hypervisor reporting a nonsensical value. Either way, we must not
    // proceed, as a bad `period_fs` would silently miscalibrate the APIC timer.
    if period_fs == 0 || period_fs > 100_000_000 {
        return None;
    }

    // Ensure the main counter is running.
    // The `ENABLE_CNF` bit (bit 0) of the General Configuration register
    // controls whether the main counter increments. Some firmware initializes
    // this to `0` (counter halted). We read the current config, check the bit,
    // and set it if needed.
    let config_ptr = (base_virt + HPET_REG_CONFIG as u64) as *mut u64;
    let config = unsafe { ptr::read_volatile(config_ptr) };
    if config & HPET_CONFIG_ENABLE == 0 {
        // Counter is halted, so we enable it. The main counter will begin
        // incrementing immediately after this write.
        unsafe { ptr::write_volatile(config_ptr, config | HPET_CONFIG_ENABLE) };
    }

    Some(HpetInfo {
        base_virt,
        period_fs,
        num_timers,
        counter_64bit,
        minimum_tick,
    })
}
