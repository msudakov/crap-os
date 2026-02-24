/*
    CrapOS Memory Manager Module
*/

#[repr(C)]
pub struct MemoryMapInfo {
    pub memory_map_addr: u64,
    pub memory_map_size: u64,
    pub descriptor_size: u64,
    pub descriptor_ver: u32,
}

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
    EfiMemoryMappedIO          = 0x0000000B,  // MMIO — don't touch
    EfiMemoryMappedIOPortSpace = 0x0000000C,  // // MMIO — don't touch
    EfiPalCode                 = 0x0000000D,  // Processor Abstraction Layer
    EfiPersistentMemory        = 0x0000000E,  // Don't touch
    EfiMaxMemoryType           = 0x0000000F,  // Reserved
}

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
