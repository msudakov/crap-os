// =============================================================================
// Memory Manager Module
// =============================================================================
// 
// The Memory Manager module is responsible for all physical and virtual memory
// operations in the system.
//
// Higher-half virtual address space layout:
//
//   0xFFFF800000000000  Physical direct map base (64TB window, covers all RAM)
//   0xFFFF900000000000  Framebuffer              (permanent virtual window)
//   0xFFFFA00000000000  Kernel heap              64 MB max heap size
//   0xFFFFFFFF80000000  Kernel image + stack     Sits in the top 2GB of the
//                                                48-bit canonical address space
//
// UEFI runtime services are intentionally NOT mapped. ExitBootServices() was
// already called in the bootloader, making SetVirtualAddressMap() illegal.
// Shutdown and reset will be handled via ACPI and legacy port 0x64,
// respectively, with zero firmware involvement at runtime.

pub mod pmm;
pub mod vmm;
pub mod kernel_heap;
pub mod memory_manager;

// Re-export deeper structs
pub use memory_manager::MemoryManager;

pub const PRESENT: u64 = 1 << 0;   // Must be 1 for the entry to be valid
pub const WRITABLE: u64 = 1 << 1;  // If 1, writes are allowed; if 0, read-only
// pub const USER: u64 = 1 << 2;     // If 1, user-mode access is allowed
pub const PWT: u64 = 1 << 3;       // Page Write-Through
pub const PCD: u64 = 1 << 4;       // Page Cache Disable
pub const NX: u64 = 1 << 63;       // No-Execute bit

/// Memory map structure received from the bootloader.
#[repr(C)]
#[derive(Copy, Clone)]
pub struct MemoryMapInfo {
    pub memory_map_addr: u64,
    pub memory_map_size: u64,
    pub descriptor_size: u64,
    pub descriptor_ver: u32,
    pub kernel_load_addr : u64,
    pub kernel_image_size: u64,
    pub stack_base_addr: u64,
    pub stack_size: u64,
    pub rsdp_addr: u64,
}

/// UEFI memory region types. After ExitBootServices(), only
/// EfiRuntimeServicesCode/Data and EfiACPIMemoryNVS will remain reserved.
/// Everything else is either already free or reclaimable by the kernel. Efi
/// services code/data will be reclaimed at the end of initialization.
#[repr(u32)]
#[allow(dead_code)]
#[derive(PartialEq, Eq, PartialOrd, Copy, Clone)]
pub enum EfiMemoryType {
    EfiReservedMemoryType      = 0x00000000,  // Reserve permanently
    EfiLoaderCode              = 0x00000001,  // Reclaimable after boot
    EfiLoaderData              = 0x00000002,  // Reclaimable after boot
    EfiBootServicesCode        = 0x00000003,  // Reclaimable after EBS
    EfiBootServicesData        = 0x00000004,  // Reclaimable after EBS
    EfiRuntimeServicesCode     = 0x00000005,  // Must stay mapped (UEFI runtime)
    EfiRuntimeServicesData     = 0x00000006,  // Must stay mapped (UEFI runtime)
    EfiConventionalMemory      = 0x00000007,  // Free/usable RAM
    EfiUnusableMemory          = 0x00000008,  // Don't touch
    EfiACPIReclaimMemory       = 0x00000009,  // Reclaimable after ACPI init
    EfiACPIMemoryNVS           = 0x0000000A,  // Reserve permanently
    EfiMemoryMappedIO          = 0x0000000B,  // MMIO - don't touch
    EfiMemoryMappedIOPortSpace = 0x0000000C,  // MMIO - don't touch
    EfiPalCode                 = 0x0000000D,  // Processor Abstraction Layer
    EfiPersistentMemory        = 0x0000000E,  // Don't touch
    EfiMaxMemoryType           = 0x0000000F,  // Reserved
}

/// Memory descriptor structure provided by the UEFI bootloader.
#[repr(C)]
#[derive(Copy, Clone)]
pub struct EfiMemoryDescriptor {
    pub region_type: EfiMemoryType,
    pub padding: u32,
    pub physical_start: u64,
    pub virtual_start: u64,
    pub num_pages: u64,
    pub attribute: u64
}

impl EfiMemoryDescriptor {
    /// Instantiates a new `EfiMemoryDescriptor` from raw address value.
    ///
    /// # Arguments
    ///
    /// * `address` - Raw address value of the start of the structure.
    pub fn new(address: u64) -> Self {
        let ptr = address as *const EfiMemoryDescriptor;
        unsafe { return *ptr }
    }
}
