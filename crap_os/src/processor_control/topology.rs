//! CPU Topology
//!
//! This module defines the data structures that describe the physical CPU
//! topology of the system, as enumerated from the ACPI MADT during early boot.
//!
//! [`CpuTopology`] is the single source of truth for all CPU-related
//! information. It is populated once during kernel initialization from the
//! MADT, and then treated as read-only for the rest of the kernel's lifetime.
//! This structure informs the rest of the system how many CPUs exist, what a
//! CPU's APIC ID is, which CPU is the BSP, etc.

/// Maximum number of logical CPUs this kernel will support.
///
/// 64 covers any realistic desktop or workstation target. We can later raise it
/// for many-core server builds. It is a compile-time constant so `CpuTopology`
/// can live in BSS with no heap allocation required.
pub const MAX_CPUS: usize = 64;

/// Information about a single logical CPU, as reported by the ACPI MADT
/// Processor Local APIC entry (type 0).
#[derive(Copy, Clone)]
pub struct CpuInfo {
    /// Hardware APIC ID. Used to address this CPU when sending IPIs and
    /// when programming I/O APIC redirection table destination fields.
    /// On xAPIC systems, this fits in a u8, but it stored as u32 for x2APIC
    /// forward-compatibility.
    pub apic_id: u32,

    /// ACPI Processor UID. The MADT-assigned identifier for this processor.
    /// Distinct from the APIC ID; used when correlating with other ACPI
    /// tables (e.g., SSDT, _MAT). Not needed for interrupt routing.
    pub acpi_uid: u8,

    /// MADT flags bit 0. When true, the firmware has this CPU enabled and
    /// it may be brought online. When false, the CPU is present in the MADT,
    /// but the firmware does not consider it usable.
    pub enabled: bool,

    /// MADT flags bit 1. When true, the CPU is capable of being brought
    /// online at runtime even if `enabled` is currently false (firmware-
    /// managed hot-plug).
    pub online_capable: bool,
}

impl CpuInfo {
    /// Returns a zeroed, disabled placeholder used to fill unused slots in
    /// the `CpuTopology` array before the MADT walk completes.
    const fn empty() -> Self {
        Self {
            apic_id: 0,
            acpi_uid: 0,
            enabled: false,
            online_capable: false,
        }
    }
}

/// System-wide CPU topology, populated once from the ACPI MADT during early
/// kernel initialization.
///
/// After `CpuTopology::new()` returns, this structure is effectively read-only.
/// No locks are needed to read it; it must never be mutated after the init path
/// completes.
pub struct CpuTopology {
    /// APIC ID of the Bootstrap Processor (BSP). Read from CPUID leaf 1 in
    /// the bootloader and passed through `MemoryMapInfo`. The BSP is the CPU
    /// that executes the kernel entry point; all other CPUs are APs.
    pub bsp_apic_id: u32,

    /// Number of valid entries in `cpus`. Always <= [`MAX_CPUS`].
    pub cpu_count: usize,

    /// Per-CPU information array. Only indices `0..cpu_count` are valid;
    /// the rest are `CpuInfo::empty()` placeholders.
    pub cpus: [CpuInfo; MAX_CPUS],
}

impl CpuTopology {
    /// Creates a new, empty `CpuTopology`. It should be called before the MADT
    /// walk. Entries are filled in by `acpi::parse_acpi`.
    pub const fn new() -> Self {
        Self {
            bsp_apic_id: 0,
            cpu_count: 0,
            cpus: [CpuInfo::empty(); MAX_CPUS],
        }
    }

    /// Appends a CPU entry discovered during the MADT walk.
    ///
    /// Silently drops entries beyond `MAX_CPUS`, but this should never happen
    /// on any target this kernel is designed for.
    /// 
    /// # Arguments
    /// 
    /// * `cpu_info` - CPU information structure to add to the CPU topology.
    pub fn push(&mut self, cpu_info: CpuInfo) {
        if self.cpu_count < MAX_CPUS {
            self.cpus[self.cpu_count] = cpu_info;
            self.cpu_count += 1;
        }
    }

    /// Looks up and returns the CPU designated as BSP, if any.
    /// 
    /// # Returns
    /// 
    /// Returns the `CpuInfo` for the BSP, identified by matching `bsp_apic_id`
    /// against the entries collected from the MADT, or `None` if the BSP's APIC
    /// ID does not appear in the MADT (which would indicate a firmware bug or a
    /// mismatch between CPUID and the MADT).
    pub fn bsp(&self) -> Option<&CpuInfo> {
        self.cpus[..self.cpu_count]
            .iter()
            .find(|cpu| cpu.apic_id == self.bsp_apic_id)
    }

    /// Returns an iterator over all valid (enabled or online-capable) CPU
    /// entries. Skips placeholder slots beyond `cpu_count`.
    pub fn iter(&self) -> impl Iterator<Item = &CpuInfo> {
        self.cpus[..self.cpu_count].iter()
    }

    /// Returns an iterator over only the CPUs that are enabled and usable.
    pub fn get_enabled_cpus(&self) -> impl Iterator<Item = &CpuInfo> {
        self.cpus[..self.cpu_count]
            .iter()
            .filter(|cpu| cpu.enabled || cpu.online_capable)
    }

    /// Returns the number of CPUs that are enabled or online-capable, i.e.
    /// the number that can actually be brought online.
    pub fn get_usable_cpu_count(&self) -> usize {
        self.get_enabled_cpus().count()
    }
}
