/*
    CrapOS Memory Manager Module
*/

#[repr(C)]
pub struct MemoryMapInfo {
    memory_map_addr: u64,
    memory_map_size: u64,
    descriptor_size: u64,
    descriptor_ver: u32,
}

#[repr(C)]
pub struct EfiMemoryDescriptor {
    region_type: u32,
    padding: u32,
    physical_start: u64,
    virtual_start: u64,
    num_pages: u64,
    attribute: u64
}
