//! ACPI (Advanced Configuration and Power Interface) Table Parser
//!
//! This module walks the ACPI table hierarchy to locate the two physical
//! addresses the APIC driver needs to initialize interrupt routing:
//! - Local APIC base address, which is usually at 0xFEE00000:
//!     The Local APIC (LAPIC) is a per-CPU controller. Each core uses its own
//!     LAPIC to receive interrupts, send inter-processor interrupts (IPIs), and
//!     manage the local timer. The base address is the same for all CPU cores.
//!
//! - I/O APIC base address, which is usually at 0xFEC00000:
//!     The I/O APIC is a single chip (or one per cluster) that receives
//!     external hardware interrupts (PCI, PS/2, timers, etc.) and routes
//!     them to one or more LAPICs. We'll only track the first I/O APIC.
//! 
//! The `find_acpi_table` function also helps locate other needed ACPI tables,
//! such as HPET that is used for APIC timer calibration.
//!
//! The ACPI table hierarchy is comprised of the following components: RSDP,
//! XSDT/RSDT, MADT (the APIC), and others. Visually, it looks like this:
//!   RSDP
//!    |_ XSDT or RSDT
//!         |- FACP (Fixed ACPI Description Table)
//!         |- APIC <- this is the MADT, and it contains what we need
//!         |- HPET
//!         |_ ... (other tables, which we can ignore for now)
//! 
//!  - RSDP (Root System Description Pointer):
//!      Supplied by the firmware and passed through by our bootloader.
//!      Contains the physical address of either the XSDT (for ACPI 2.0+) or
//!      RSDT (for ACPI 1.0), depending on the revision field.
//!
//!  - XSDT/RSDT (Extended/Root System Description Table)
//!      An array of physical pointers (64-bit for XSDT, 32-bit for RSDT),
//!      each pointing to another SDT. We scan this array looking for "APIC",
//!      which is the MADT signature.
//!
//!  - MADT (Multiple APIC Description Table, signature "APIC")
//!      Contains the default Local APIC address, followed by a variable-length
//!      list of interrupt controller structures - one per LAPIC, I/O APIC,
//!      interrupt source override, NMI source, etc. We scan these entries to
//!      find the first I/O APIC entry (type 1).
//!
//! All ACPI structures are defined as `repr(C, packed)`. The ACPI spec lays
//! them out with no implicit padding, and they may be placed at arbitrary byte
//! offsets in firmware-provided memory, so alignment cannot be assumed.
//!
//! Rust forbids creating a reference (`&T`) to an unaligned field of a packed
//! struct, because the reference itself would be misaligned and any read
//! through it is undefined behaviour on architectures that require alignment.
//! To avoid this, we always use `ptr::read_unaligned` (via `addr_of!`) when
//! reading multi-byte fields from packed structs. Single `u8` fields can be
//! read through a field access because they are always aligned.

use core::ptr;
use crate::memory_manager::MemoryManager;
use crate::processor_control::topology::{CpuInfo, CpuTopology};

/// ACPI 2.0 Root System Description Pointer (RSDP) - the entry point into the
/// ACPI table tree.
///
/// The firmware places this structure at a well-known physical address and
/// passes it to the OS during boot via the `BootInfo` struct. The first 20
/// bytes are the ACPI 1.0 structure; the remaining fields were added in ACPI
/// 2.0. The two versions can be distinguished using `revision`:
///   - revision == 0 for ACPI 1.0 means we use `rsdt_addr` (32-bit);
///   - revision >= 2 for ACPI 2.0+ means we use `xsdt_addr` (64-bit) when
///       non-zero.
#[repr(C, packed)]
struct Rsdp {
    /// ASCII signature, always `b"RSD PTR "` (with the trailing space). The
    /// space is important, as it pads the field to 8 bytes.
    signature: [u8; 8],

    /// ACPI 1.0 checksum. The sum of all bytes in the first 20 bytes of this
    /// structure (the ACPI 1.0 portion) must equal 0 mod 256. We do not verify
    /// this checksum because the bootloader is assumed to have validated the
    /// RSDP before passing its address to us.
    checksum: u8,

    /// OEM-assigned identifier string; it is not null-terminated.
    oem_id: [u8; 6],

    /// ACPI revision:
    ///   0 = ACPI 1.0 (use `rsdt_addr`, as `xsdt_addr` does not exist).
    ///   2 = ACPI 2.0+ (use `xsdt_addr`).
    revision: u8,

    /// Physical address of the RSDT (Root System Description Table).
    /// Used only on ACPI 1.0 (revision == 0), or as a fallback. It is
    /// 32-bit because ACPI 1.0 predates systems with more than 4 GiB of RAM.
    rsdt_addr: u32,

    /// Total byte length of this RSDP structure (including the 2.0 fields).
    length: u32,

    /// Physical address of the XSDT (Extended System Description Table).
    /// 64-bit, preferred over `rsdt_addr` on ACPI 2.0+ systems.
    /// May be 0 on some broken firmware; fall back to `rsdt_addr` in that case.
    xsdt_addr: u64,

    /// ACPI 2.0 extended checksum. Covers the entire RSDP structure
    /// (all bytes including the 2.0 extension fields). We do not verify this.
    ext_checksum: u8,

    /// Padding - reserved by the ACPI spec, must be zero.
    _reserved:  [u8; 3],
}

/// The common header that every ACPI System Description Table begins with.
///
/// All ACPI tables (RSDT, XSDT, MADT, FADT, HPET, etc.) start with this
/// identical 36-byte header. The `signature` field identifies the specific
/// table type, and `length` covers the entire table including this header.
///
/// We cast raw virtual pointers to `*const SdtHeader` to identify a table by
/// its signature before deciding how to parse the rest.
#[repr(C, packed)]
pub(super) struct SdtHeader {
    /// 4-byte ASCII table identifier:
    ///   `b"APIC"` = MADT (Multiple APIC Description Table)
    ///   `b"FACP"` = FADT (Fixed ACPI Description Table)
    ///   `b"XSDT"` = Extended System Description Table
    ///   `b"RSDT"` = Root System Description Table
    pub signature: [u8; 4],

    /// Total length of this table in bytes, including this header and all
    /// following data. Used to compute the end address when walking entries.
    pub length: u32,

    /// Table-specific revision number. Its meaning varies per table type.
    pub revision: u8,

    /// Checksum such that the sum of all bytes in the table equals 0 mod 256.
    /// It is not verified by this implementation.
    pub checksum: u8,

    /// OEM identifier (not null-terminated, padded with spaces or zeros).
    pub oem_id: [u8; 6],

    /// OEM-assigned table identifier (e.g., the system model name).
    pub oem_table_id: [u8; 8],

    /// OEM-assigned revision number for this specific table instance.
    pub oem_revision: u32,

    /// Vendor ID of the tool that created the table (e.g., BIOS/UEFI vendor).
    pub creator_id: u32,

    /// Revision of the creation tool.
    pub creator_revision: u32,
}

/// The MADT-specific header that immediately follows the generic `SdtHeader`.
///
/// The MADT begins with the `SdtHeader` (signature = `"APIC"`), then this
/// 8-byte structure, then a variable-length array of interrupt controller
/// entries (each starting with a `MadtEntryHeader`).
#[repr(C, packed)]
struct MadtHeader {
    /// Default physical base address of the Local APIC. This is the address
    /// used before any Local APIC Address Override entries are processed.
    /// On most modern x86-64 systems, this is 0xFEE00000.
    local_apic_addr: u32,

    /// System flags. Bit 0 (PCAT_COMPAT): set if the system also has a
    /// legacy dual-8259 PIC that must be masked/disabled before enabling APIC
    /// mode. We do not use this flag directly, as the APIC driver handles PIC
    /// masking unconditionally.
    flags: u32,
}

/// The 2-byte tag at the start of every MADT interrupt controller entry.
///
/// All MADT entries begin with this header, which gives the entry type and
/// total length (in bytes, including these 2 bytes). We read this first to
/// know how to interpret the remaining bytes and how far to advance the cursor.
#[repr(C, packed)]
struct MadtEntryHeader {
    /// Identifies the type of this interrupt controller structure:
    ///   0 = Processor Local APIC
    ///   1 = I/O APIC (this is the type we care about)
    ///   2 = Interrupt Source Override
    ///   3 = NMI Source
    ///   4 = Local APIC NMI
    ///   5 = Local APIC Address Override
    ///   ... Other types
    entry_type: u8,

    /// Total byte length of this entry, including this 2-byte header.
    /// Must be >= 2; a value < 2 indicates a malformed table.
    length: u8,
}

/// MADT entry type 0: Processor Local APIC.
///
/// One entry exists per logical CPU in the system. The `apic_id` field is the
/// hardware APIC ID used for interrupt routing and IPI targeting. The `flags`
/// field indicates whether the CPU is usable:
///   bit 0 (ENABLED):        firmware has enabled this CPU.
///   bit 1 (ONLINE_CAPABLE): CPU can be brought online at runtime (hot-plug).
///
/// A CPU with neither bit set should be ignored entirely.
#[repr(C, packed)]
struct MadtLocalApic {
    /// Standard 2-byte MADT entry header (entry_type = 0, length = 8).
    header: MadtEntryHeader,

    /// ACPI Processor UID. Correlates this entry with ACPI processor objects
    /// in the DSDT/SSDT. Not used for interrupt routing.
    acpi_uid: u8,

    /// Hardware APIC ID. This is what the LAPIC ID register reports for this
    /// CPU, and what the I/O APIC redirection table destination field expects
    /// when targeting this CPU in physical destination mode.
    apic_id: u8,

    /// Processor flags:
    ///   bit 0 = ENABLED:        CPU is enabled and may be started.
    ///   bit 1 = ONLINE_CAPABLE: CPU supports firmware-managed hot-plug.
    flags: u32,
}

/// MADT entry type 1: I/O APIC.
///
/// Describes one I/O APIC in the system. A machine may have more than one
/// I/O APIC (common in multi-socket servers); each has its own base address
/// and covers a contiguous range of Global System Interrupts (GSIs) starting
/// at `gsi_base`. We only use the first one found.
#[repr(C, packed)]
struct MadtIoApic {
    /// Standard 2-byte MADT entry header (entry_type = 1, length = 12).
    header: MadtEntryHeader,

    /// Hardware ID of this I/O APIC, assigned by the firmware. Used when
    /// programming redirection table entries to specify the target I/O APIC
    /// in systems with multiple I/O APICs. This is not used for now.
    io_apic_id: u8,

    /// Reserved and must be zero. Used for alignment within the ACPI table.
    _reserved: u8,

    /// Physical base address of this I/O APIC's memory-mapped register file.
    /// The I/O APIC exposes two 32-bit registers at this address:
    ///   offset 0x00 = IOREGSEL (index register: write the register index here)
    ///   offset 0x10 = IOWIN    (data window: read/write the selected register)
    io_apic_addr: u32,

    /// Global System Interrupt base for this I/O APIC.
    /// The GSI number for pin N of this I/O APIC is `gsi_base + N`.
    /// For the first (and usually only) I/O APIC, this is typically 0.
    /// A second I/O APIC might have gsi_base = 24 if the first handles 24 pins.
    gsi_base: u32,
}

/// The APIC-related information extracted from the ACPI tables. This info is
/// used by the APIC driver.
#[allow(dead_code)]
pub struct ApicInfo {
    /// Physical base address of the Local APIC register block. We need to map
    /// this address (usually 0xFEE00000) into the kernel's virtual address
    /// space before accessing any LAPIC registers.
    pub local_apic_phys: u64,

    /// Physical base address of the first I/O APIC's register file. We need to
    /// map this address (usually 0xFEC00000) before programming IRQ routing.
    pub io_apic_phys: u64,

    /// Global System Interrupt base of the first I/O APIC, which is usually 0.
    /// The APIC driver adds this to a pin index to get the GSI number used in
    /// interrupt source override entries.
    pub io_apic_gsi_base: u32,
}

/// Walks the RSDP and searches the XSDT (ACPI 2.0+) or RSDT (ACPI 1.0) for a
/// table matching the given 4-byte signature.
/// 
/// # Arguments
/// 
/// * `rsdp_virt` - RSDP virtual address from the bootloader (already translated
///     through the kernel's direct physical map).
/// * `sig`       - The 4-byte signature to search for.
/// 
/// # Returns
/// 
/// Returns a virtual pointer to the table's SdtHeader on success, or `None` if
/// the RSDP signature is invalid or the MADT cannot be found.
pub(super) unsafe fn find_acpi_table(
    rsdp_virt: u64,
    sig: &[u8; 4],
) -> Option<*const SdtHeader> {
    // Cast the virtual address to a reference to our Rsdp struct
    let rsdp = unsafe { &*(rsdp_virt as *const Rsdp) };

    // Verify the RSDP signature before trusting any other fields;
    // `b"RSD PTR "` includes the trailing space, and all 8 bytes must match.
    if &rsdp.signature != b"RSD PTR " {
        return None;
    }

    // Choose between XSDT (64-bit pointers for ACPI 2.0+) and RSDT (32-bit
    // pointers for ACPI 1.0) based on the revision field. We prefer XSDT when
    // available because RSDT physical addresses are 32-bit and could fail to
    // represent tables above 4 GiB (though this is very rare).
    if rsdp.revision >= 2 && rsdp.xsdt_addr != 0 {
        // ACPI 2.0+: use the 64-bit XSDT
        unsafe {
            find_table_in_xsdt(
                MemoryManager::phys_to_virt(rsdp.xsdt_addr), sig)
        }
    } else {
        // ACPI 1.0: use the 32-bit RSDT
        unsafe {
            find_table_in_rsdt(
                MemoryManager::phys_to_virt(rsdp.rsdt_addr as u64), sig)
        }
    }
}

/// Parses the ACPI tables starting from the RSDP.
///
/// Performs a single pass through the MADT, simultaneously collecting:
///   - APIC hardware addresses (Local APIC base, I/O APIC base, and GSI base)
///   - CPU topology (one `CpuInfo` per Processor Local APIC entry, type 0)
///
/// The `bsp_apic_id` field of the returned `CpuTopology` is set from the
/// `bsp_apic_id` argument (passed from `MemoryMapInfo`, which the bootloader
/// populated via CPUID leaf 1).
///
/// # Arguments
/// 
/// * `rsdp_virt`    - RSDP virtual address from the bootloader (already
///     translated through the kernel's direct physical map). ACPI tables
///     themselves store physical addresses internally; this function translates
///     those physical addresses to virtual ones using the Memory Manager before
///     dereferencing them.
/// * `bsp_apic_id`  - APIC ID of the Bootstrap Processor, as read by the
///     bootloader via CPUID leaf 1, EBX bits [31:24].
/// 
/// # Returns
/// 
/// Returns `Some((ApicInfo, CpuTopology))` on success, or `None` if the RSDP
/// signature is invalid or the MADT cannot be found.
/// 
/// # Safety
/// 
/// The caller must ensure that:
/// - `rsdp_virt` is a valid, mapped virtual address pointing to a genuine RSDP.
/// - All physical addresses referenced by the ACPI tables are covered by the
///     kernel's direct physical map (i.e., `phys_to_virt` can safely translate
///     them). This is guaranteed as long as the physical map covers the first
///     4 GiB, as ACPI tables are always below 4 GiB on x86-64.
pub unsafe fn parse_acpi(
    rsdp_virt: u64,
    bsp_apic_id: u32,
) -> Option<(ApicInfo, CpuTopology)> {
    // Locate the APIC table (MADT) pointer
    let madt_ptr = unsafe { find_acpi_table(rsdp_virt, b"APIC")? };

    // Single pass: collect APIC addresses and CPU topology simultaneously
    unsafe { parse_madt(madt_ptr, bsp_apic_id) }
}

/// Scans the XSDT entry array for a table whose 4-byte signature matches `sig`.
///
/// The XSDT body (after the `SdtHeader`) is a packed array of 64-bit physical
/// addresses. Each address points to another SDT. We dereference each pointer
/// (after translating to virtual), check its signature, and return the first
/// match.
///
/// # Arguments
/// 
/// * `xsdt_virt` - Virtual address of the XSDT's `SdtHeader`.
/// * `sig`       - The 4-byte signature to search for (e.g., `b"APIC"`).
///
/// # Returns
/// 
/// Returns a raw virtual pointer to the matching table's `SdtHeader`, or `None`
/// if no table with the requested signature exists in the XSDT.
unsafe fn find_table_in_xsdt(xsdt_virt: u64, sig: &[u8; 4]
) -> Option<*const SdtHeader> {
    // Interpret start of the XSDT as a generic SDT header to read its length
    let hdr = unsafe { &*(xsdt_virt as *const SdtHeader) };

    // Read `length` using read_unaligned because `SdtHeader` is packed and the
    // field may not be at a 4-byte-aligned offset in memory.
    let table_len = unsafe {
        ptr::read_unaligned(core::ptr::addr_of!((*hdr).length)) as usize
    };

    // The 64-bit pointer array begins immediately after the 36-byte SdtHeader
    let entries_start = xsdt_virt as usize + size_of::<SdtHeader>();

    // The array ends at the last byte covered by `table_len`
    let entries_end = xsdt_virt as usize + table_len;

    // `ptr_addr` is our cursor - it advances by 8 bytes (one u64 pointer) each
    // iteration.
    let mut ptr_addr = entries_start;

    // We stop when fewer than 8 bytes remain (incomplete entry)
    while ptr_addr + 8 <= entries_end {
        // Read the 64-bit physical address of the next child table through
        // a raw pointer cast.
        let entry_phys = unsafe { *(ptr_addr as *const u64) };
        ptr_addr += 8;  // Advance past this entry regardless of validity

        // A null pointer in the XSDT is a firmware bug, but can occur;
        // we skip it to avoid dereferencing address 0.
        if entry_phys == 0 {
            continue;
        }

        // Translate the physical address to virtual before reading the header.
        // The direct physical map guarantees this is valid for any address in
        // the first 4 GiB, which covers all ACPI tables.
        let entry_hdr = unsafe {
            &*(MemoryManager::phys_to_virt(entry_phys) as *const SdtHeader)
        };

        // The signature is a 4-byte array; we compare it directly with `sig`.
        // We can compare through a reference here because `[u8; 4]` has
        // alignment 1, so there is no unaligned-reference UB risk.
        if &entry_hdr.signature == sig {
            return Some(entry_hdr as *const SdtHeader);
        }
    }

    None
}

/// Scans the RSDT entry array for a table whose 4-byte signature matches `sig`.
///
/// Functionally identical to `find_table_in_xsdt`, except the entry array
/// contains 32-bit physical addresses rather than 64-bit ones.
///
/// # Arguments
/// 
/// * `rsdt_virt` - Virtual address of the RSDT's `SdtHeader`.
/// * `sig`       - The 4-byte signature to search for (e.g., `b"APIC"`).
///
/// # Returns
/// 
/// Returns a raw virtual pointer to the matching table's `SdtHeader`, or `None`.
/// if no table with the requested signature exists in the RSDT.
unsafe fn find_table_in_rsdt(rsdt_virt: u64, sig: &[u8; 4]
) -> Option<*const SdtHeader> {
    let hdr = unsafe { &*(rsdt_virt as *const SdtHeader) };

    let table_len = unsafe {
        ptr::read_unaligned(core::ptr::addr_of!((*hdr).length)) as usize
    };

    // Entry array begins after the SdtHeader; each entry is 4 bytes (u32)
    let entries_start = rsdt_virt as usize + size_of::<SdtHeader>();
    let entries_end   = rsdt_virt as usize + table_len;
    let mut ptr_addr  = entries_start;

    while ptr_addr + 4 <= entries_end {
        // Read a 32-bit physical pointer and widen it to 64 bits for
        // `phys_to_virt` to work.
        let entry_phys = unsafe { *(ptr_addr as *const u32) as u64 };
        ptr_addr += 4;  // Advance past this 32-bit entry

        if entry_phys == 0 {
            continue;  // Skip the null entry
        }

        let entry_hdr = unsafe {
            &*(MemoryManager::phys_to_virt(entry_phys) as *const SdtHeader)
        };

        if &entry_hdr.signature == sig {
            return Some(entry_hdr as *const SdtHeader);
        }
    }

    None
}

/// Performs a single pass through the MADT, collecting both APIC hardware
/// addresses and the full CPU topology.
///
/// The MADT table has the following layout (all sizes in bytes):
///
///   +-------------------------------------+
///   | SdtHeader (36 bytes)                |  Common header, signature = "APIC"
///   |-------------------------------------|
///   | MadtHeader (8 bytes)                |  local_apic_addr + flags
///   |-------------------------------------|
///   | Entry 0: MadtEntryHeader (2 bytes)  |  type + length
///   |   ... entry-specific data ...       |
///   |-------------------------------------|
///   | Entry 1: MadtEntryHeader (2 bytes)  |
///   |   ... entry-specific data ...       |
///   |-------------------------------------|
///   | ... (variable number of entries) ...|
///   +-------------------------------------+
///
/// Entry types handled:
///  0 = Processor Local APIC -> appended to `CpuTopology`
///  1 = I/O APIC             -> captured as `io_apic_phys` / `io_apic_gsi_base`
///
/// All other entry types are skipped by advancing the cursor by `entry.length`.
///
/// # Arguments
/// 
/// * `madt_sdt`    - Virtual pointer to the start of the MADT (its SdtHeader).
/// * `bsp_apic_id` - APIC ID of the BSP (from bootloader CPUID read), stored
///     into `CpuTopology::bsp_apic_id`, so that callers can identify the BSP
///     entry.
///
/// # Returns
/// 
/// Returns `Some((ApicInfo, CpuTopology))` on success, or `None` if no I/O
/// APIC entry was found (which would make APIC-mode interrupt routing
/// impossible).
unsafe fn parse_madt(
    madt_sdt: *const SdtHeader,
    bsp_apic_id: u32,
) -> Option<(ApicInfo, CpuTopology)> {
    // Read the total table length from the SDT header, used to compute the
    // address of the last byte of the entry array.
    let sdt_len = unsafe {
        ptr::read_unaligned(core::ptr::addr_of!((*madt_sdt).length)) as usize
    };

    // Read the default Local APIC address from the MADT-specific header.
    // The MadtHeader starts immediately after the 36-byte SdtHeader.
    let madt_hdr_ptr = (
        madt_sdt as usize + size_of::<SdtHeader>()) as *const MadtHeader;
    let local_apic_addr = unsafe {
        ptr::read_unaligned(core::ptr::addr_of!(
            (*madt_hdr_ptr).local_apic_addr)) as u64
    };

    // Walk the variable-length entry list. Entries start after both the
    // SdtHeader and the MadtHeader.
    let entries_start = madt_sdt as usize + size_of::<SdtHeader>()
        + size_of::<MadtHeader>();
    let entries_end = madt_sdt as usize + sdt_len;

    let mut cursor = entries_start;
    let mut io_apic_phys: u64 = 0;
    let mut io_apic_gsi_base: u32 = 0;

    // Initialize the topology with the BSP APIC ID from the bootloader.
    // CPU entries will be appended as we encounter type-0 entries below.
    let mut topology = CpuTopology::new();
    topology.bsp_apic_id = bsp_apic_id;

    while cursor + size_of::<MadtEntryHeader>() <= entries_end {
        // Read the 2-byte entry header to determine type and length.
        // Both fields are u8, so alignment is not a concern here.
        let entry = unsafe { &*(cursor as *const MadtEntryHeader) };
        let entry_len = entry.length as usize;

        // A well-formed entry must be at least 2 bytes (just the header).
        // A length of 0 or 1 would cause an infinite loop, so we treat it
        // as a corrupt table and stop parsing.
        if entry_len < 2 {
            break;
        }

        match entry.entry_type {
            // Type 0: Processor Local APIC - one entry per logical CPU.
            // Minimum valid length is 8 bytes (header + acpi_uid + apic_id +
            // flags). Entries shorter than this are malformed, so we skip them.
            0 if entry_len >= 8 => {
                let lapic = unsafe { &*(cursor as *const MadtLocalApic) };

                let apic_id = lapic.apic_id;
                let acpi_uid = lapic.acpi_uid;
                let flags = unsafe {
                    ptr::read_unaligned(core::ptr::addr_of!(lapic.flags))
                };
                let enabled = (flags & 0x1) != 0;
                let online_capable = (flags & 0x2) != 0;

                // Only record CPUs the firmware considers usable. A CPU with
                // neither ENABLED nor ONLINE_CAPABLE set cannot be started.
                if enabled || online_capable {
                    topology.push(CpuInfo {
                        apic_id: apic_id as u32,
                        acpi_uid,
                        enabled,
                        online_capable,
                    });
                }
            }

            // Type 1: I/O APIC - capture the first one found.
            // We do not currently support multi-I/O-APIC systems; the first
            // entry is sufficient for all IRQ routing on any single-socket
            // desktop or workstation target.
            1 if entry_len >= 12 => {
                // Only record the first I/O APIC; ignore subsequent ones.
                if io_apic_phys == 0 {
                    let io = unsafe { &*(cursor as *const MadtIoApic) };
                    io_apic_phys = unsafe {
                        ptr::read_unaligned(
                            core::ptr::addr_of!(io.io_apic_addr)) as u64
                    };
                    io_apic_gsi_base = unsafe {
                        ptr::read_unaligned(core::ptr::addr_of!(io.gsi_base))
                    };
                }
            }

            // All other entry types (ISOs, NMIs, overrides, x2APIC, etc.)
            // are intentionally ignored for now. The cursor still advances
            // by `entry_len` below, so they are correctly skipped.
            _ => {}
        }

        cursor += entry_len;
    }

    // If no I/O APIC entry was found, we cannot configure external interrupts.
    if io_apic_phys == 0 {
        return None;
    }

    let apic_info = ApicInfo {
        local_apic_phys: local_apic_addr,
        io_apic_phys,
        io_apic_gsi_base,
    };

    Some((apic_info, topology))
}
