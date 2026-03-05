// =============================================================================
// CrapOS Memory Manager Module
// =============================================================================
// 
// This module is responsible for all physical and virtual memory operations
// in the system.
//
// Higher-half virtual address space layout:
//
//   0xFFFF800000000000  Physical direct map base (64TB window, covers all RAM)
//   0xFFFF900000000000  Framebuffer              (permanent virtual window)
//   0xFFFFFFFF80000000  Kernel image + stack     Sits in the top 2GB of the
//                                                48-bit canonical address space
//
// UEFI runtime services are intentionally NOT mapped. ExitBootServices() was
// already called in the bootloader, making SetVirtualAddressMap() illegal.
// Shutdown and reset are handled via ACPI and legacy port 0x64 respectively,
// with zero firmware involvement at runtime.

use crate::globals;


const PRESENT: u64 = 1 << 0;  // Must be 1 for the entry to be valid
const WRITABLE: u64 = 1 << 1; // If 1, writes are allowed; if 0, read-only
// const USER: u64 = 1 << 2;     // If 1, user-mode access is allowed
const NX: u64 = 1 << 63;

// Kernel start and end tags collected from the linker.
unsafe extern "C" {
    static __kernel_phys_start: u8;
    static __kernel_phys_end: u8;
}

// This is the structure received from the bootloader
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
}

/// UEFI memory region types. After ExitBootServices(), only
/// EfiRuntimeServicesCode/Data and EfiACPIMemoryNVS must remain reserved.
/// Everything else is either already free or reclaimable by the kernel.
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
    EfiMemoryMappedIOPortSpace = 0x0000000C,  // MMIO — don't touch
    EfiPalCode                 = 0x0000000D,  // Processor Abstraction Layer
    EfiPersistentMemory        = 0x0000000E,  // Don't touch
    EfiMaxMemoryType           = 0x0000000F,  // Reserved
}

// Memory descriptor structure provided by the UEFI bootloader
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

// =============================================================================
// Physical Memory Manager
// =============================================================================
//
// The main job of the Physical Memory Manager is to allocate, free, and keep
// track of physical memory pages in RAM. But, it has to do it really-really
// efficiently because the entire system, including the Virtual Memory Manager,
// depends on it for this one task. It has to be as fast as possible. 
//
// There are several methods of keeping track of physical pages. For example,
// the bitmap method is able to locate a new free page in time O(n) when
// unoptimized and down to O(log n) with some optimizations. However, this
// implementation uses a simpler method that is able to fetch a new page
// in runtime of O(1), or in deterministic time.
//
// Specifically, it uses a stack-type (LIFO) singly-linked list.
// Besides a counter for the number of free pages remaining in RAM, its
// `free_list_head` member always points to the first/next free physical page
// to be delivered when requested. In turn, each free physical page is
// modified to have its first 8 bytes hold the address of the next page, and
// so on.
//
// Each free page points to the next. Every time a page is freed and recycled
// back to the manager, the PMM will take the current head address, place it
// in the first 8 bytes of the newly-freed page to bump the old top page down,
// and then update the head to point to the newly-freed page. And when a page
// is allocated, the reverse takes place: the PMM follows the head to the
// soon-to-be allocated page to read its first 8 bytes and find the
// next-in-line page for later allocations. It then updates the head address
// to point to the following page and returns the requsted page to the caller.

#[allow(dead_code)]
pub struct PhysicalMemoryManager {
    kernel_load_addr: u64,       // Where the bootloader mapped the kernel image
    kernel_image_size: u64,      // Size of the kernel
    kernel_stack_base_addr: u64, // Base address of the initial kernel stack
    kernel_stack_size: u64,      // Size of the initial kernel stack
    framebuffer_addr: u64,       // Where the framebuffer is located
    framebuffer_size: u64,       // Framebuffer total size (its width x height)
    memory_map_addr: u64,        // Memory map structure address from bootloader
    memory_map_size: u64,        // Total memory map size
    memory_map_desc_size: u64,   // Size of a memory map descriptor structure
    free_list_head: Option<u64>, // Physical address of the next free page
    free_pages: u64,             // Total number of remaining free pages

    higher_half: bool,           // True if running in higher-half kernel
}

impl PhysicalMemoryManager {
    /// Instantiates and initializes the Physical Memory Manager.
    ///
    /// # Arguments
    ///
    /// * `framebuffer_info` - Framebuffer info structure from the bootloader.
    /// * `memory_map` - Memory map information structure from the bootloader.
    pub fn init(framebuffer_info: &crate::FramebufferInfo,
        memory_map: &MemoryMapInfo,
    ) -> Self {
        // Need to divide BPP by 8 because it historically represents
        // bits-per-pixel instead of bytes.
        let fb_size = (framebuffer_info.framebuffer_height as u64) *
            (framebuffer_info.framebuffer_width as u64) *
            (framebuffer_info.framebuffer_bpp as u64 / 8);

        // Instantiate the Physical Memory Manager
        let mut pmm = PhysicalMemoryManager {
            framebuffer_addr: framebuffer_info.framebuffer_addr,
            framebuffer_size: fb_size,
            memory_map_addr: memory_map.memory_map_addr,
            memory_map_size: memory_map.memory_map_size,
            memory_map_desc_size: memory_map.descriptor_size,
            kernel_load_addr: memory_map.kernel_load_addr,
            kernel_image_size: memory_map.kernel_image_size,
            kernel_stack_base_addr: memory_map.stack_base_addr,
            kernel_stack_size: memory_map.stack_size,
            free_list_head: None,  // Singly-linked list of free page frames
            free_pages: 0,         // Counter of the free page frames
            higher_half: false,
        };

        // Initialize the PMM
        let mut descriptor_addr = pmm.memory_map_addr;
        let num_segments = pmm.memory_map_size / pmm.memory_map_desc_size;

        // Traverse the memory map, map out usable regions, and free all pages
        for _ in 0..num_segments {
            let memory_descriptor = EfiMemoryDescriptor::new(descriptor_addr);
            descriptor_addr += pmm.memory_map_desc_size;

            // Only seed the free list with conventional memory for now.
            // Boot services and loader memory are reclaimed later in
            // reclaim_boot_memory(), after page tables are fully established.
            if memory_descriptor.region_type != EfiMemoryType::EfiConventionalMemory {
                continue;
            }

            for i in 0..memory_descriptor.num_pages {
                let page_start_addr = memory_descriptor.physical_start+i*0x1000;
                let page_end_addr = page_start_addr + 0x1000 - 1;

                if page_start_addr == 0 {
                    continue;  // Skipping physical page 0 (null page)
                }

                // Skip pages that collide with regions we depend on
                // Detect collision on the UEFI memory map region
                if page_overlaps(page_start_addr, page_end_addr,
                    pmm.memory_map_addr,
                    pmm.memory_map_addr + pmm.memory_map_size
                ) {
                    continue;
                }
                
                // Detect collision on the mapped kernel image memory region
                let kernel_start = core::ptr::addr_of!(__kernel_phys_start) as u64;
                let kernel_end = core::ptr::addr_of!(__kernel_phys_end) as u64;
                if page_overlaps(page_start_addr, page_end_addr,
                    kernel_start,
                    kernel_end
                ) {
                    continue;
                }

                // Detect collision on the mapped kernel stack memory region
                if page_overlaps(page_start_addr, page_end_addr,
                    pmm.kernel_stack_base_addr,
                    pmm.kernel_stack_base_addr + pmm.kernel_stack_size
                ) {
                    continue;
                }

                // Detect collision on the mapped framebuffer memory region
                if page_overlaps(page_start_addr, page_end_addr,
                    pmm.framebuffer_addr,
                    pmm.framebuffer_addr + pmm.framebuffer_size
                ) {
                    continue;
                }

                // Insert the free page into the singly-linked list
                pmm.free_page(page_start_addr);
            }
        }
        pmm
    }

    /// Writes the current head into the first 8 bytes of the given page, then
    /// updates the head with the address of this page.
    ///
    /// # Arguments
    ///
    /// * `address` - Address of the physical page to free and re-insert.
    ///
    /// # Safety
    /// 
    /// Dereferences a raw pointer by address value.
    /*fn free_page(&mut self, address: u64) {
        let ptr = address as *mut u64;
        unsafe { *ptr =  self.free_list_head.unwrap_or(0) };
        self.free_list_head = Some(address);
        self.free_pages += 1;
    }*/
    fn free_page(&mut self, address: u64) {
        let virt = if self.higher_half {
            address + globals::KERNEL_PHYSICAL_MAP_BASE
        } else {
            address
        };
        let ptr = virt as *mut u64;
        unsafe { *ptr = self.free_list_head.unwrap_or(0) };
        self.free_list_head = Some(address);
        self.free_pages += 1;
    }

    /// Fetches the next free physical page and allocates it by updating the
    /// head to the allocated page's next pointer in its first 8 bytes.
    ///
    /// # Returns
    ///
    /// Returns the address value of the allocated page. Returns None if out of
    /// free physical memory.
    ///
    /// # Safety
    /// 
    /// Dereferences a raw pointer by address value.
    /*fn alloc_page(&mut self) -> Option<u64> {
        let head = self.free_list_head?;
        let next = unsafe { *(head as *const u64) };

        self.free_list_head = if next == 0 {
            None
        }
        else {
            Some(next)
        };

        self.free_pages -= 1;
        Some(head)
    }*/
    fn alloc_page(&mut self) -> Option<u64> {
        let head = self.free_list_head?;
        let virt = if self.higher_half {
            head + globals::KERNEL_PHYSICAL_MAP_BASE
        } else {
            head
        };
        let next = unsafe { *(virt as *const u64) };
        self.free_list_head = if next == 0 { None } else { Some(next) };
        self.free_pages -= 1;
        Some(head)
    }
}

/// Checks if a given memory page overlaps with a memory region that should be
/// treated as untouchable.
/// 
/// This ensures that the physical memory manager never delivers a "free" page
/// that is used by the firmware or by any part of the system itself.
///
/// # Arguments
///
/// * `page_start`   - Start address of the page to check.
/// * `page_end`     - End address of the page to check.
/// * `region_start` - Start address of a critical region to check against.
/// * `region_end`   - End address of a critical region to check against.
/// 
/// # Returns
/// 
/// True if the given page is found to be overlapping with the given memory
/// region, false otherwise.
fn page_overlaps(page_start: u64, page_end: u64, region_start: u64,
    region_end: u64
) -> bool {
    page_start <= region_end && page_end >= region_start
}

// =============================================================================
// Virtual Memory Manager
// =============================================================================

// On x86-64 systems with 4-level paging, which is provided by UEFI, the
// paging hierarchy is PML4 -> PDPT -> PD -> PT -> physical page frame, where:
//   * PML4 - Page Map Level 4, contains 512 PML4Es
//   * PDPT - Page Directory Pointer Table, contains 512 PDPTEs 
//   * PD   - Page Directory, contains 512 PDEs
//   * PT   - Page Table, contains 512 64-bit PTEs
//
// Each of these tables is 4KB, containing 512 entries, with each entry being
// 8 bytes.
//
// This structure maps 48-bit virtual addresses to physical addresses. A
// 48-bit virtual address is split into 9-bit indices for each mapping level,
// plus a 12-bit page offset: 9-bit PML4 index, 9-bit PDPT index, 9-bit PD
// index, 9-bit PT index, and 12-bit offset. A single PML4 entry (PML4E) can
// map 512 GB of memory, making the total addressable space per PML4 table 256
// TB, which is more than enough for our purposes here. PML5 allows for larger
// (57-bit) virtual address spaces.
//
// To facilitate address translation, the Memory Management Unit (MMU), located
// in the CPU chip package, uses the CR3 register to locate the PML4. It then
// traverses the levels to resolve the final physical address.


/// Looks up or creates a page table by its address and an index from previous
/// level.
/// 
/// # Arguments
///
/// * `pmm`   - Physical memory manager.
/// * `table` - Address of the page table to look up or create if not exists.
/// * `index` - Page table index from previous lookup level.
/// 
/// # Returns
/// 
/// Returns a pointer to a new page table if created, or an existing table.
/// 
/// # Safety
/// 
/// Dereferences raw pointers.
fn get_or_create_table(pmm: &mut PhysicalMemoryManager, table: *mut u64,
    index: u64
) -> *mut u64 {
    unsafe {
        let entry = table.add(index as usize);

        if *entry & PRESENT == 0 {
            let new_table = pmm.alloc_page().expect(
                "[CRITICAL] OOM during page table walk!");
            zero_out_page(new_table);
            *entry = new_table | PRESENT | WRITABLE;
        }

        (*entry & !0xFFF) as *mut u64
    }
}

/// Maps a virtual page to a physical memory page with the 4-level PML4
/// MMU addressing specification.
/// 
/// # Arguments
///
/// * `pmm`           - Physical memory manager.
/// * `pml4_addr`     - Address of the current PML4 from the CR3 register.
/// * `virtual_addr`  - Virtual page address to map.
/// * `physical_addr` - Physical page address to map to.
/// * `flags`         - Virtual page flags to include in the mapping.
/// 
/// # Safety
/// 
/// Dereferences raw pointers.
fn map_page(pmm: &mut PhysicalMemoryManager, pml4_addr: *mut u64,
    virtual_addr: u64, physical_addr: u64, flags: u64
) {
    let pml4_index = (virtual_addr >> 39) & 0x1FF;
    let pdpt_index = (virtual_addr >> 30) & 0x1FF;
    let pd_index   = (virtual_addr >> 21) & 0x1FF;
    let pt_index   = (virtual_addr >> 12) & 0x1FF;

    let pdpt = get_or_create_table(pmm, pml4_addr, pml4_index);
    let pd   = get_or_create_table(pmm, pdpt, pdpt_index);
    let pt   = get_or_create_table(pmm, pd, pd_index);

    unsafe {
        let pte = pt.add(pt_index as usize);
        *pte = (physical_addr & !0xFFF) | flags | PRESENT;
    }
}

/// Refreshes TLB by reloading CR3 using inline assembly.
/// 
/// Writing the CR3 register back to itself on x86 processors acts as a command
/// to the CPU to flush (invalidate) all non-global Translation Lookaside Buffer
/// (TLB) entries. Even though the value in the register does not change, the
/// hardware interprets the write to CR3 operation as a signal that the paging
/// structures may have changed, requiring the TLB to be refreshed.
/// 
/// # Safety
/// 
/// Uses inline assembly to write to the CR3 register.
#[allow(dead_code)]
fn refresh_tlb() {
    unsafe {
        core::arch::asm!(
            "mov rax, cr3",
            "mov cr3, rax",
            out("rax") _,
            options(nostack, preserves_flags)
        );
    }
}

/// Instructs the MMU to invalidate a virtual memory page and refresh its
/// virtual mappings.
/// 
/// # Arguments
///
/// * `address` - Address of a virtual page to invalidate and refresh.
/// 
/// # Safety
/// 
/// Uses inline assembly to invoke `invlpg` on a virtual address.
#[allow(dead_code)]
fn invalidate_page(address: u64) {
    unsafe {
        core::arch::asm!("invlpg [{}]", in(reg) address, options(nostack,
            preserves_flags));
    }
}

/// Zeroes out a given page.
/// 
/// # Arguments
///
/// * `address` - Address of the page to zero out.
/// 
/// # Safety
/// 
/// Performs a `write_volatile` on raw memory locations.
fn zero_out_page(address: u64) {
    let page_cursor = address as *mut u64;
    for i in 0..512 {
        unsafe { page_cursor.add(i).write_volatile(0u64) };
    }
}

/// Initial page table setup, called before the higher-half jump.
/// 
/// Builds a new PML4 that identity-maps just enough of physical memory to keep
/// the CPU alive while we execute the CR3 switch, then immediately jumps to
/// the higher half. The identity maps are torn down later by
/// remove_identity_maps().
/// 
/// # Arguments
///
/// * `pmm` - Physical memory manager.
/// * `framebuffer_info` - Framebuffer info structure from the bootloader.
/// * `memory_map` - Memory map information structure from the bootloader.
/// 
/// # Returns
/// 
/// Returns the new PML4 table.
/// 
/// # Safety
/// 
/// Uses inline assembly to write the new PML4 address to the CR3 register and
/// then jump to the higher-half.
pub fn init_page_tables(pmm: &mut PhysicalMemoryManager,
    framebuffer_info: &crate::FramebufferInfo, memory_map: &MemoryMapInfo
) -> *mut u64 {
    // Allocate and zero out the new PML4
    let new_pml4 = pmm.alloc_page().expect("[CRITICAL] OOM allocating PML4!");
    zero_out_page(new_pml4);
    let pml4 = new_pml4 as *mut u64;

    let mut descriptor_addr = memory_map.memory_map_addr;
    let num_segments = memory_map.memory_map_size / memory_map.descriptor_size;

    for _ in 0..num_segments {
        let descriptor = EfiMemoryDescriptor::new(descriptor_addr);
        descriptor_addr += memory_map.descriptor_size;

        // Identity map all conventional and boot-related regions so the CPU
        // does not fault during the brief window between CR3 load and the
        // higher-half jump. These identity maps are removed in
        // init_higher_half().
        if !(descriptor.region_type == EfiMemoryType::EfiConventionalMemory ||
            descriptor.region_type == EfiMemoryType::EfiLoaderCode ||
            descriptor.region_type == EfiMemoryType::EfiLoaderData ||
            descriptor.region_type == EfiMemoryType::EfiBootServicesCode ||
            descriptor.region_type == EfiMemoryType::EfiBootServicesData
        ) {
            continue;
        }

        for i in 0..descriptor.num_pages {
            let phys = descriptor.physical_start + i * 0x1000;
            map_page(pmm, pml4, phys, phys, PRESENT | WRITABLE);
        }
    }

    // Identity map + higher-half map: kernel image
    let kernel_start = core::ptr::addr_of!(__kernel_phys_start) as u64;
    let kernel_end = core::ptr::addr_of!(__kernel_phys_end) as u64;
    let mut addr = kernel_start;
    while addr < kernel_end {
        map_page(pmm, pml4, addr, addr, PRESENT | WRITABLE);
        map_page(pmm, pml4, addr + globals::KERNEL_VIRTUAL_BASE, addr,
            PRESENT | WRITABLE);
        addr += 0x1000;
    }

    // Identity map + higher-half map: kernel stack (NX — never execute stack)
    let stack_start = memory_map.stack_base_addr & !0xFFF;
    let stack_end = memory_map.stack_base_addr + memory_map.stack_size;
    let mut addr = stack_start;
    while addr < stack_end {
        map_page(pmm, pml4, addr, addr, PRESENT | WRITABLE | NX);
        map_page(pmm, pml4, addr + globals::KERNEL_VIRTUAL_BASE, addr,
            PRESENT | WRITABLE | NX);
        addr += 0x1000;
    }

    // Identity map: framebuffer (low address only; higher-half map is set
    // up later in init_higher_half() at KERNEL_FRAMEBUFFER_VIRTUAL_BASE). Need
    // to divide BPP by 8 because it historically represents bits-per-pixel
    // instead of bytes.
    let fb_size = (framebuffer_info.framebuffer_height as u64)
        * (framebuffer_info.framebuffer_width as u64)
        * (framebuffer_info.framebuffer_bpp as u64 / 8);
    let fb_start = framebuffer_info.framebuffer_addr & !0xFFF;
    let fb_end = framebuffer_info.framebuffer_addr + fb_size;
    let mut addr = fb_start;
    while addr < fb_end {
        map_page(pmm, pml4, addr, addr, PRESENT | WRITABLE);
        addr += 0x1000;
    }

    // Identity map: UEFI memory map buffer
    let map_start = memory_map.memory_map_addr & !0xFFF;
    let map_end = memory_map.memory_map_addr + memory_map.memory_map_size;
    let mut addr = map_start;
    while addr < map_end {
        map_page(pmm, pml4, addr, addr, PRESENT | WRITABLE);
        addr += 0x1000;
    }

    pml4
}

// =============================================================================
// Higher-half transition (called after init_page_tables)
// =============================================================================

/// Builds a direct map where every physical address P is accessible at
/// (KERNEL_PHYSICAL_MAP_BASE + P).
/// 
/// This is a standard approach that lets the rest of the kernel avoid reasoning
/// about raw physical addresses above the PMM layer. We use the functions
/// `MemoryManager::phys_to_virt()` / `virt_to_phys()` to convert between them.
/// The EfiRuntimeServicesCode regions are mapped without NX since the firmware
/// may store executable trampolines there, even though we never call them.
/// All other regions get NX as a defensive measure.
/// 
/// # Arguments
///
/// * `pmm` - Physical memory manager.
/// * `pml4` - PML4 table.
/// * `memory_map` - Memory map information structure from the bootloader.
fn build_direct_map(pmm: &mut PhysicalMemoryManager, pml4: *mut u64,
    memory_map: &MemoryMapInfo,
) {
    let phys_map_base = globals::KERNEL_PHYSICAL_MAP_BASE;
    let mut descriptor_addr = memory_map.memory_map_addr;
    let num_segments = memory_map.memory_map_size / memory_map.descriptor_size;

    for _ in 0..num_segments {
        let desc = EfiMemoryDescriptor::new(descriptor_addr);
        descriptor_addr += memory_map.descriptor_size;

        // MMIO and unknown types are intentionally excluded — they have
        // device-specific mapping requirements and must never be touched here.
        let should_map = matches!(
            desc.region_type,
            EfiMemoryType::EfiConventionalMemory   |
            EfiMemoryType::EfiLoaderCode           |
            EfiMemoryType::EfiLoaderData           |
            EfiMemoryType::EfiBootServicesCode     |
            EfiMemoryType::EfiBootServicesData     |
            EfiMemoryType::EfiRuntimeServicesCode  |
            EfiMemoryType::EfiRuntimeServicesData  |
            EfiMemoryType::EfiACPIReclaimMemory    |
            EfiMemoryType::EfiACPIMemoryNVS
        );
        if !should_map { continue; }

        let flags = match desc.region_type {
            EfiMemoryType::EfiRuntimeServicesCode => PRESENT | WRITABLE,
            _                                     => PRESENT | WRITABLE | NX,
        };

        for i in 0..desc.num_pages {
            let phys = desc.physical_start + i * 0x1000;
            map_page(pmm, pml4, phys_map_base + phys, phys, flags);
        }
    }
}





/// Maps the framebuffer to its permanent virtual address at FB_VIRT_BASE.
/// After this, the framebuffer driver should update its base pointer and
/// never reference the old identity-mapped address again.
fn map_framebuffer_higher_half(
    pmm:             &mut PhysicalMemoryManager,
    pml4:            *mut u64,
    framebuffer_info: &crate::FramebufferInfo,
) {
    let fb_size = (framebuffer_info.framebuffer_height as u64)
        * (framebuffer_info.framebuffer_width  as u64)
        * (framebuffer_info.framebuffer_bpp    as u64 / 8);

    let fb_phys_start = framebuffer_info.framebuffer_addr & !0xFFF;
    let fb_phys_end   = (framebuffer_info.framebuffer_addr + fb_size + 0xFFF)
        & !0xFFF;

    let mut phys = fb_phys_start;
    let mut virt = globals::KERNEL_FRAMEBUFFER_VIRTUAL_BASE;
    while phys < fb_phys_end {
        map_page(pmm, pml4, virt, phys, PRESENT | WRITABLE);
        phys += 0x1000;
        virt += 0x1000;
    }
}

/// Removes all lower-half identity maps by zeroing PML4 entries 0–255, then
/// flushes the TLB. After this point no address below 0xFFFF800000000000 is
/// valid. This is the point of no return for identity-mapped access.
fn remove_identity_maps(pml4_phys: u64) {
    // PML4 entries 0–255 span the entire lower half of the canonical address
    // space (0x0000_0000_0000_0000 – 0x0000_7FFF_FFFF_FFFF). Entries 256–511
    // are the higher half and must be left intact.
    
    let pml4_virt = (pml4_phys + globals::KERNEL_PHYSICAL_MAP_BASE) as *mut u64;
    //for i in 0..256usize {
    for i in 0..256usize {
        //crate::serial::print("[INFO] TEST X\n");
        unsafe { pml4_virt.add(i).write_volatile(0u64) };
    }
    
    //refresh_tlb();
    //crate::serial::print("[INFO] TEST 4.3\n");
    // Inline the CR3 reload here rather than calling refresh_tlb().
    // A function call after zeroing the PML4 entries may cause the
    // compiler to emit a stack access before the TLB is flushed,
    // hitting a now-unmapped low address and triple faulting.
    unsafe {
        core::arch::asm!(
            "mov rax, cr3",
            "mov cr3, rax",
            out("rax") _,
        );
    }
    // Nothing after this — no stack access, no function epilogue,
    // no register restore. The caller's code resumes in higher half
    // with a clean TLB.
}

/// Reclaims bootloader and boot-services memory back into the PMM free list.
///
/// Must be called after remove_identity_maps() — by this point the CPU no
/// longer needs those pages mapped, and we're safe to recycle them.
///
/// The following are intentionally NOT reclaimed here:
///   - EfiRuntimeServicesCode/Data  (firmware reserved — do not touch)
///   - EfiACPIMemoryNVS             (ACPI firmware tables — permanent)
///   - EfiACPIReclaimMemory         (reclaim separately after ACPI init)
///   - The memory map buffer itself (still being iterated)
///   - The kernel image and stack
///
/// EfiConventionalMemory pages are already in the PMM from init() and must
/// not be double-freed.
/*#[allow(dead_code)]
fn reclaim_boot_memory(
    pmm:        &mut PhysicalMemoryManager,
    memory_map: &MemoryMapInfo,
) {
    let mut descriptor_addr = memory_map.memory_map_addr;
    let num_segments = memory_map.memory_map_size / memory_map.descriptor_size;

    let map_start    = memory_map.memory_map_addr;
    let map_end      = memory_map.memory_map_addr + memory_map.memory_map_size;
    let kernel_start = core::ptr::addr_of!(__kernel_phys_start) as u64;
    let kernel_end   = core::ptr::addr_of!(__kernel_phys_end)   as u64;

    for _ in 0..num_segments {
        let desc = EfiMemoryDescriptor::new(descriptor_addr);
        descriptor_addr += memory_map.descriptor_size;

        // Only these types become unconditionally free after EBS
        let reclaimable = matches!(
            desc.region_type,
            EfiMemoryType::EfiLoaderCode       |
            EfiMemoryType::EfiLoaderData       |
            EfiMemoryType::EfiBootServicesCode |
            EfiMemoryType::EfiBootServicesData
        );
        if !reclaimable { continue; }

        for i in 0..desc.num_pages {
            let page_start = desc.physical_start + i * 0x1000;
            let page_end   = page_start + 0x1000 - 1;

            // Guard: never free the memory map buffer we're currently reading
            if page_overlaps(page_start, page_end, map_start, map_end) {
                continue;
            }
            // Guard: never free the kernel image
            if page_overlaps(page_start, page_end, kernel_start, kernel_end) {
                continue;
            }
            // Guard: never free the kernel stack
            if page_overlaps(
                page_start, page_end,
                pmm.kernel_stack_base_addr,
                pmm.kernel_stack_base_addr + pmm.kernel_stack_size,
            ) { continue; }

            pmm.free_page(page_start);
        }
    }
}*/




/// Reloads the GDTR and IDTR with their direct-map virtual addresses.
/// Must be called after build_direct_map() and before remove_identity_maps().
fn reload_gdt_and_idt() {
    // Read the current physical base addresses from the GDTR and IDTR
    let mut gdt_desc = [0u8; 10]; // 2 bytes limit + 8 bytes base
    let mut idt_desc = [0u8; 10];

    unsafe {
        core::arch::asm!(
        "sgdt [{gdt}]",
        "sidt [{idt}]",
        gdt = in(reg) gdt_desc.as_mut_ptr(),
        idt = in(reg) idt_desc.as_mut_ptr(),
        )
    };

    // Extract the 8-byte base address from bytes [2..10]
    let gdt_phys = u64::from_le_bytes(gdt_desc[2..10].try_into().unwrap());
    let idt_phys = u64::from_le_bytes(idt_desc[2..10].try_into().unwrap());
    let gdt_limit = u16::from_le_bytes(gdt_desc[0..2].try_into().unwrap());
    let idt_limit = u16::from_le_bytes(idt_desc[0..2].try_into().unwrap());

    // Compute the direct-map virtual addresses
    let gdt_virt = gdt_phys + globals::KERNEL_PHYSICAL_MAP_BASE;
    let idt_virt = idt_phys + globals::KERNEL_PHYSICAL_MAP_BASE;

    // Build new descriptors with the virtual base addresses and reload
    let mut new_gdt = [0u8; 10];
    let mut new_idt = [0u8; 10];

    new_gdt[0..2].copy_from_slice(&gdt_limit.to_le_bytes());
    new_gdt[2..10].copy_from_slice(&gdt_virt.to_le_bytes());
    new_idt[0..2].copy_from_slice(&idt_limit.to_le_bytes());
    new_idt[2..10].copy_from_slice(&idt_virt.to_le_bytes());

    unsafe {
        core::arch::asm!(
        "lgdt [{gdt}]",
        "lidt [{idt}]",
        gdt = in(reg) new_gdt.as_ptr(),
        idt = in(reg) new_idt.as_ptr(),
    )
    };
}




// Structure for the Global Descriptor Table
#[repr(C, align(8))]
struct Gdt {
    null:    u64,
    code64:  u64,
    data64:  u64,
}

/*
    When the _start routine is invoked from the bootloader, we're still
    running under UEFI's GDT and CS segment. This creates our own GDT that the
    kernel loads at the very start of the routine.
*/
static GDT: Gdt = Gdt {
    null:   0x0000000000000000,
    code64: 0x00AF9A000000FFFF,  // 64-bit code, ring 0
    data64: 0x00CF92000000FFFF,  // 64-bit data, ring 0
};

/// Replaces the bootloader's GDT with the kernel's GDT.
/// 
/// # Safety
/// 
/// Uses inline assembly to load the OS kernel's own GDT.
fn load_gdt() {
    unsafe {
        // Build a 10-byte GDTR directly on the stack as a byte array
        // to avoid any alignment or packed struct issues
        let mut gdtr = [0u8; 10];
        let limit = (core::mem::size_of::<Gdt>() - 1) as u16;
        let base  = &GDT as *const Gdt as u64;

        gdtr[0..2].copy_from_slice(&limit.to_le_bytes());
        gdtr[2..10].copy_from_slice(&base.to_le_bytes());

        core::arch::asm!(
            "lgdt [{gdtr}]",
            "push 0x8",
            "lea rax, [rip + 3f]",
            "push rax",
            "retfq",
            "3:",
            "mov ax, 0x10",
            "mov ds, ax",
            "mov es, ax",
            "mov ss, ax",
            "mov fs, ax",
            "mov gs, ax",
            gdtr = in(reg) gdtr.as_ptr(),
            out("rax") _,
        )
    };
}






// =============================================================================
// Global Memory Manager Interface
// =============================================================================
//
// This interface encapsulates the needed physical and virtual memory management
// structures and functionality, and exports them to the rest of the system.
// This interface is implemented with IRQ-safe spinlock in the system globals.

pub struct MemoryManager {
    pmm: PhysicalMemoryManager,
    pml4: *mut u64,  // virtual (direct-map) address, used after higher-half
    pml4_phys: u64,  // physical address, used during early init
}

#[allow(dead_code)]
impl MemoryManager {
    /// Instantiates and initializes physical memory manager, maps available
    /// physical memory, and then initializes virtual page tables.
    /// 
    /// # Arguments
    ///
    /// * `framebuffer_info` - Framebuffer info structure from the bootloader.
    /// * `memory_map` - Memory map information structure from the bootloader.
    pub fn init(
        pmm: PhysicalMemoryManager,
        pml4_phys: u64,
    ) -> Self {
        // First, we need to set the EFER.NXE (No-Execute Enable) bit in the
        // IA32_EFER MSR (Model Specific Register). This allows the use of the
        // NX bit in memory regions, which is critical for secutiry.
        let efer_msr: u64 = 0xC0000080;
        let mut low: u32;
        let mut high: u32;
        unsafe {
            // Read the current value of the MSR
            core::arch::asm!(
                "rdmsr",
                in("ecx") efer_msr,
                out("eax") low,
                out("edx") high
            );
            
            // Set NXE bit (bit 11, which is in the low 32 bits) via bitwise OR
            low |= 1 << 11;
            
            // Write the modified value back to the MSR
            core::arch::asm!(
                "wrmsr",
                in("ecx") efer_msr,
                in("eax") low,
                in("edx") high
            );
        }

        Self {
            pmm,
            pml4: pml4_phys as *mut u64,  // starts as physical, updated below
            pml4_phys,
        }
    }





    /// Phase 2 — Completes the higher-half transition. Call this immediately
    /// after `init()`, once the CPU is running with the new PML4 in CR3.
    ///
    /// This function:
    ///   1. Builds the full physical direct map at PHYS_MAP_BASE
    ///   2. Maps the framebuffer to its permanent address at FB_VIRT_BASE
    ///   3. Removes all lower-half identity maps (point of no return)
    ///   4. Reclaims bootloader and boot-services memory into the PMM
    ///
    /// After this returns, no low virtual address is valid. All framebuffer
    /// writes must use FB_VIRT_BASE. Physical addresses are reachable via
    /// phys_to_virt(). UEFI is completely gone from the address space.
    pub fn init_higher_half(
        &mut self,
        framebuffer_info: &crate::FramebufferInfo,
        memory_map:       &MemoryMapInfo,
    ) {
        build_direct_map(&mut self.pmm, self.pml4, memory_map);
        self.pmm.higher_half = true;

        map_framebuffer_higher_half(&mut self.pmm, self.pml4, framebuffer_info);

        // Switch pml4 pointer to its direct-map virtual address now that
        // the direct map exists. All subsequent page table operations use
        // this virtual address, including remove_identity_maps.
        self.pml4 = (self.pml4_phys + globals::KERNEL_PHYSICAL_MAP_BASE) as *mut u64;

        // Reload GDT and IDT via their direct-map virtual addresses before
        // wiping the low identity maps. The CPU reads these on every fault
        // and segment operation — if they're at low addresses when the
        // identity maps are gone, any exception causes a triple fault.
        reload_gdt_and_idt();

        load_gdt();
        crate::serial::print("[INFO] Loaded GDT\n");

        // TODO: implement removing identify maps and reclaiming space


        // All new mappings must be established before the identity maps are
        // removed. Reversing this order would cause an immediate page fault.
        remove_identity_maps(self.pml4_phys);

        // Boot memory reclaim must come last — we are still reading the
        // memory map buffer throughout all of the above.
        reclaim_boot_memory(&mut self.pmm, memory_map);

    }

    



    /// Delegates to the inner `PhysicalMemoryManager::alloc_page()` and
    /// allocates a physical page.
    pub fn alloc_page(&mut self) -> Option<u64> {
        self.pmm.alloc_page()
    }

    /// Delegates to the inner `PhysicalMemoryManager::free_page()` and frees
    /// a physical page.
    pub fn free_page(&mut self, addr: u64) {
        self.pmm.free_page(addr)
    }

    /// Delegates to the `map_page` and maps a physical page to a virtual page.
    pub fn map_page(&mut self, virtual_addr: u64, physical_addr: u64, flags: u64) {
        map_page(&mut self.pmm, self.pml4, virtual_addr, physical_addr, flags);
    }

    // TODO: implement in the future when needed
    //pub fn unmap_page(&mut self, virtual_addr: u64) {
    //    // future
    //}

    /// Converts a physical address to its direct-map virtual address.
    /// Valid for any address covered by build_direct_map().
    #[inline(always)]
    pub fn phys_to_virt(phys: u64) -> u64 {
        phys + globals::KERNEL_PHYSICAL_MAP_BASE
    }

    /// Converts a direct-map virtual address back to its physical address.
    #[inline(always)]
    pub fn virt_to_phys(virt: u64) -> u64 {
        virt - globals::KERNEL_PHYSICAL_MAP_BASE
    }
}

// Implements unsafe Send for spinlock management
unsafe impl Send for MemoryManager {}


/// Reclaims EfiACPIReclaimMemory pages. Call this after your ACPI
    /// subsystem has finished parsing all tables it needs.

    /// Reclaims bootloader and boot-services memory back into the PMM free list.
///
/// Must be called after remove_identity_maps() — by this point the CPU no
/// longer needs those pages mapped, and we're safe to recycle them.
///
/// The following are intentionally NOT reclaimed here:
///   - EfiRuntimeServicesCode/Data  (firmware reserved — do not touch)
///   - EfiACPIMemoryNVS             (ACPI firmware tables — permanent)
///   - EfiACPIReclaimMemory         (reclaim separately after ACPI init)
///   - The memory map buffer itself (still being iterated)
///   - The kernel image and stack
///
/// EfiConventionalMemory pages are already in the PMM from init() and must
/// not be double-freed.
    #[inline(never)]
fn reclaim_boot_memory(
    pmm:        &mut PhysicalMemoryManager,
    memory_map: &MemoryMapInfo,
    ) {
        let mut descriptor_addr = memory_map.memory_map_addr
            + globals::KERNEL_PHYSICAL_MAP_BASE;
        let num_segments = memory_map.memory_map_size / memory_map.descriptor_size;

        let map_start    = memory_map.memory_map_addr;
        let map_end      = memory_map.memory_map_addr + memory_map.memory_map_size;
        let kernel_start = core::ptr::addr_of!(__kernel_phys_start) as u64;
        let kernel_end   = core::ptr::addr_of!(__kernel_phys_end)   as u64;

        for _ in 0..num_segments {
            let desc = EfiMemoryDescriptor::new(descriptor_addr);
            descriptor_addr += memory_map.descriptor_size;

            let reclaimable = matches!(
                desc.region_type,
                EfiMemoryType::EfiLoaderCode       |
                EfiMemoryType::EfiLoaderData       |
                EfiMemoryType::EfiBootServicesCode |
                EfiMemoryType::EfiBootServicesData
            );
            if !reclaimable { continue; }

            for i in 0..desc.num_pages {
                let page_start = desc.physical_start + i * 0x1000;
                let page_end   = page_start + 0x1000 - 1;

                if page_start == 0 { continue; }

                if page_overlaps(page_start, page_end, map_start, map_end) {
                    continue;
                }
                if page_overlaps(page_start, page_end, kernel_start, kernel_end) {
                    continue;
                }
                if page_overlaps(
                    page_start, page_end,
                    pmm.kernel_stack_base_addr,
                    pmm.kernel_stack_base_addr + pmm.kernel_stack_size,
                ) { continue; }

                pmm.free_page(page_start);
            }
        }
    }


// TODO: For sanity checks only. Delete later
/*pub fn test_vmm() {
    unsafe {
        let cookie: u64 = 0xDEADBEEFCAFEBABE;
        let phys_addr = globals::MEMORY_MANAGER.lock().as_mut().unwrap().alloc_page().unwrap();
        crate::sprintln!("\n[TEST] Got physical address 0x{:016X}", phys_addr);
        let virt_addr: u64 = 0x0000010000000000;
        crate::sprintln!("[TEST] Chosen virtual address 0x{:016X}", virt_addr);
        let phys_ptr = phys_addr as *const u64;
        crate::sprintln!("[TEST] Read from physical address 0x{:016X}", *phys_ptr);
        let cr3: u64;
        core::arch::asm!("mov {}, cr3", out(reg) cr3);
        crate::sprintln!("[TEST] CR3 / PML4 address 0x{:016X}", cr3);
        globals::MEMORY_MANAGER.lock().as_mut().unwrap().map_page(virt_addr, phys_addr, PRESENT | WRITABLE);
        crate::sprintln!("[TEST] Mapped new virtual page");
        let ptr = virt_addr as *mut u64;
        *ptr = cookie;
        crate::sprintln!("[TEST] Wrote cookie to virtual address");
        crate::sprintln!("[TEST] Read cookie from virtual address 0x{:016X}", *ptr);
        crate::sprintln!("[TEST] Read cookie from physical address 0x{:016X}", *phys_ptr);
    }
}*/
