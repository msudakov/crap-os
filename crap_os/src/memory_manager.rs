// =============================================================================
// Memory Manager Module
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

const PAGE_SIZE: u64 = 0x1000;  // Default page size of 4096 bytes

pub const PRESENT: u64 = 1 << 0;   // Must be 1 for the entry to be valid
pub const WRITABLE: u64 = 1 << 1;  // If 1, writes are allowed; if 0, read-only
// pub const USER: u64 = 1 << 2;     // If 1, user-mode access is allowed
pub const NX: u64 = 1 << 63;

// Kernel physical start and physical end tags collected from the linker.
unsafe extern "C" {
    static __kernel_phys_start: u8;
    static __kernel_phys_end: u8;
}

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

/// GDT (Global Descriptor Table). The GDT must exist and the segment selectors
/// loaded into the segment registers must reference valid descriptors. The
/// `align(8)` ensures the table is 8-byte aligned, which is required by the
/// the x86_64 architecture.
#[repr(C, align(8))]
struct Gdt {
    null:    u64,  // The null descriptor
    code64:  u64,  // Entry 1 (selector 0x08): 64-bit kernel code segment
    data64:  u64,  // Entry 2 (selector 0x10): 64-bit kernel data segment
}

// The kernel's statically allocated GDT, stored in the `.rodata` section.
///
/// Using a `static` (rather than stack-allocated) ensures the table remains
/// alive for the entire life of the kernel and is not accidentally freed.
/// When the _start routine is invoked from the bootloader, we're still running
/// under UEFI's GDT and CS segment. This creates our own GDT that the kernel
/// loads at the very start of the routine.
static GDT: Gdt = Gdt {
    null:   0x0000000000000000,
    code64: 0x00AF9A000000FFFF,  // 64-bit code, ring 0
    data64: 0x00CF92000000FFFF,  // 64-bit data, ring 0
};

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
// implementation uses a simpler method that is able to fetch a new page and
// also release a page in runtime of O(1), or in deterministic time.
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

/// Physical Memory Manager strcuture.
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
    kernel_start: u64,           // From `__kernel_phys_start` linker tag
    kernel_end: u64,             // From `__kernel_phys_end` linker tag
    is_higher_half: bool,        // True if running in higher-half kernel
    free_list_head: Option<u64>, // Physical address of the next free page
    free_pages: u64,             // Total number of remaining free pages
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

        let kernel_start = core::ptr::addr_of!(__kernel_phys_start) as u64;
        let kernel_end = core::ptr::addr_of!(__kernel_phys_end) as u64;

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
            kernel_start: kernel_start,
            kernel_end: kernel_end,
            is_higher_half: false,  // Default this to false and flip later on
            free_list_head: None,   // Singly-linked list of free page frames
            free_pages: 0,          // Counter of the free page frames
        };

        // Initialize the descriptor pointer and compute the number of segments
        let mut descriptor_addr = pmm.memory_map_addr;
        let num_segments = pmm.memory_map_size / pmm.memory_map_desc_size;

        // Traverse the memory map, map out usable regions, and free all pages
        for _ in 0..num_segments {
            let memory_descriptor = EfiMemoryDescriptor::new(descriptor_addr);
            descriptor_addr += pmm.memory_map_desc_size;

            // Only seed the free list with conventional memory for now.
            // Boot services and loader memory are reclaimed later in
            // reclaim_boot_memory(), after page tables are fully established.
            if memory_descriptor.region_type !=
                EfiMemoryType::EfiConventionalMemory { continue; }

            for i in 0..memory_descriptor.num_pages {
                let page_start_addr =
                    memory_descriptor.physical_start + i * PAGE_SIZE;
                let page_end_addr = page_start_addr + PAGE_SIZE - 1;

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
                if page_overlaps(page_start_addr, page_end_addr, kernel_start,
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
    /// Computes the virtual address with the physical map base offset if the
    /// kernel is running in the higher-half virtual memory space.
    ///
    /// # Arguments
    ///
    /// * `address` - Address of the physical page to free and re-insert.
    ///
    /// # Safety
    /// 
    /// Dereferences a raw pointer by address value.
    fn free_page(&mut self, address: u64) {
        let virt = if self.is_higher_half {
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
    /// Computes the virtual address with the physical map base offset if the
    /// kernel is running in the higher-half virtual memory space.
    ///
    /// # Returns
    ///
    /// Returns the address value of the allocated page. Returns None if out of
    /// free physical memory.
    ///
    /// # Safety
    /// 
    /// Dereferences a raw pointer by address value.
    fn alloc_page(&mut self) -> Option<u64> {
        let head = self.free_list_head?;
        let virt = if self.is_higher_half {
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
//
// On x86-64 systems with 4-level paging, the paging hierarchy is:
//   PML4 -> PDPT -> PD -> PT -> physical page frame, where:
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
            let new_table_phys = pmm.alloc_page().expect(
                "[CRITICAL] OOM during page table walk!");
            let new_table_virt = if pmm.is_higher_half {
                new_table_phys + globals::KERNEL_PHYSICAL_MAP_BASE
            } else {
                new_table_phys
            };
            zero_out_page(new_table_virt);
            *entry = new_table_phys | PRESENT | WRITABLE;
        }

        let phys = *entry & !0xFFF;
        let virt = if pmm.is_higher_half {
            phys + globals::KERNEL_PHYSICAL_MAP_BASE
        } else {
            phys
        };
        virt as *mut u64
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

/// Walks the page table hierarchy for a given virtual address and returns
/// the physical address it maps to, or None if any level is absent.
///
/// # Arguments
///
/// * `pml4_addr`    - Pointer to the root PML4 table in CR3.
/// * `virtual_addr` - The virtual address to resolve.
///
/// # Returns
///
/// Returns the physical address the virtual address maps to, or `None` if it
/// is unmapped.
///
/// # Safety
///
/// Dereferences raw pointers derived from the page-table walk.
fn get_physical_addr(pml4_addr: *mut u64, virtual_addr: u64) -> Option<u64> {
    let pml4_index = (virtual_addr >> 39) & 0x1FF;
    let pdpt_index = (virtual_addr >> 30) & 0x1FF;
    let pd_index   = (virtual_addr >> 21) & 0x1FF;
    let pt_index   = (virtual_addr >> 12) & 0x1FF;
    let offset     =  virtual_addr        & 0xFFF;

    unsafe {
        // PML4 -> PDPT
        let pml4e = pml4_addr.add(pml4_index as usize);
        if *pml4e & PRESENT == 0 {
            return None;
        }
        let pdpt = ((*pml4e & !0xFFF) + globals::KERNEL_PHYSICAL_MAP_BASE)
            as *mut u64;

        // PDPT -> PD
        let pdpte = pdpt.add(pdpt_index as usize);
        if *pdpte & PRESENT == 0 {
            return None;
        }
        let pd = ((*pdpte & !0xFFF) + globals::KERNEL_PHYSICAL_MAP_BASE)
            as *mut u64;

        // PD -> PT
        let pde = pd.add(pd_index as usize);
        if *pde & PRESENT == 0 {
            return None;
        }
        let pt = ((*pde & !0xFFF) + globals::KERNEL_PHYSICAL_MAP_BASE)
            as *mut u64;

        // PT -> PTE
        let pte = pt.add(pt_index as usize);
        if *pte & PRESENT == 0 {
            return None;
        }

        Some((*pte & !0xFFF) | offset)
    }
}

/// Scans a given table for any entries present in it.
///
/// Used by `unmap_page` for page table reclamation after unmapping.
/// 
/// # Arguments
///
/// * `table` - Virtual pointer to the page table to scan.
/// 
/// # Returns
/// 
/// Returns `true` if all 512 entries in the given page table are absent, false
/// otherwise.
///
/// # Safety
///
/// Dereferences raw pointers.
fn is_table_empty(table: *mut u64) -> bool {
    for i in 0..512 {  // Max 512 entries to investigate
        if unsafe { *table.add(i) } & PRESENT != 0 {
            return false;
        }
    }
    true
}

/// Unmaps a single virtual page, zeroes its PTE, invalidates the TLB entry,
/// and reclaims any intermediate page tables (PT, PD, PDPT) that become
/// empty as a result.
///
/// It does not free the physical frame the mapping pointed to; that is the
/// caller's responsibility. The Memory Manager's function `unmap_and_free_page`
/// can be used for the common case where we might want to both unmap and free
/// a page.
///
/// # Arguments
///
/// * `pmm`          - Physical memory manager, used to free emptied tables.
/// * `pml4_addr`    - Virtual pointer to the root PML4 table.
/// * `virtual_addr` - The virtual address whose mapping should be removed.
///
/// # Returns
/// 
/// Returns `true` if a live mapping was found and removed, `false` if the
/// address was already unmapped at any level of the walk.
/// 
/// # Safety
///
/// Dereferences raw pointers derived from the page-table walk.
fn unmap_page(pmm: &mut PhysicalMemoryManager, pml4_addr: *mut u64,
    virtual_addr: u64
) -> bool {
    let pml4_index = (virtual_addr >> 39) & 0x1FF;
    let pdpt_index = (virtual_addr >> 30) & 0x1FF;
    let pd_index   = (virtual_addr >> 21) & 0x1FF;
    let pt_index   = (virtual_addr >> 12) & 0x1FF;

    unsafe {
        // First, we need to walk down the hierarchy collecting pointers to
        // each level's entry.

        // PML4 -> PDPT
        let pml4e = pml4_addr.add(pml4_index as usize);
        if *pml4e & PRESENT == 0 {
            return false;
        }
        let pdpt = ((*pml4e & !0xFFF) + globals::KERNEL_PHYSICAL_MAP_BASE)
            as *mut u64;

        // PDPT -> PD
        let pdpte = pdpt.add(pdpt_index as usize);
        if *pdpte & PRESENT == 0 {
            return false;
        }
        let pd = ((*pdpte & !0xFFF) + globals::KERNEL_PHYSICAL_MAP_BASE)
            as *mut u64;

        // PD -> PT
        let pde = pd.add(pd_index as usize);
        if *pde & PRESENT == 0 {
            return false;
        }
        let pt = ((*pde & !0xFFF) + globals::KERNEL_PHYSICAL_MAP_BASE)
            as *mut u64;

        // PT -> PTE
        let pte = pt.add(pt_index as usize);
        if *pte & PRESENT == 0 {
            return false;
        }

        // Zero the PTE and flush this TLB slot
        pte.write_volatile(0u64);
        invalidate_page(virtual_addr);

        // After zeroing the PTE, we walk back up the hierarchy to reclaim
        // tables that are now empty. We have to check at each level whether
        // the table is now entirely empty before freeing it.

        // PT: if empty, free it and clear its entry in the PD
        if is_table_empty(pt) {
            let pt_phys = *pde & !0xFFF;
            pde.write_volatile(0u64);
            pmm.free_page(pt_phys);

            // PD: if empty after losing the PT, free it and clear its PDPT
            // entry
            if is_table_empty(pd) {
                let pd_phys = *pdpte & !0xFFF;
                pdpte.write_volatile(0u64);
                pmm.free_page(pd_phys);

                // PDPT: if empty after losing the PD, free it and clear its
                // PML4 entry
                if is_table_empty(pdpt) {
                    let pdpt_phys = *pml4e & !0xFFF;
                    pml4e.write_volatile(0u64);
                    pmm.free_page(pdpt_phys);
                }
            }
        }

        true
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
            "mov rax, cr3",  // Read current CR3 (PML4 address) into rax
            "mov cr3, rax",  // Write it back; this flushes the TLB.
            out("rax") _,    // Tell the compiler RAX is clobbered (no output).
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

    // Initialize the descriptor pointer and compute the number of segments
    let mut descriptor_addr = memory_map.memory_map_addr;
    let num_segments = memory_map.memory_map_size / memory_map.descriptor_size;

    for _ in 0..num_segments {
        let descriptor = EfiMemoryDescriptor::new(descriptor_addr);
        descriptor_addr += memory_map.descriptor_size;

        // Identity map all conventional and boot-related regions so the CPU
        // does not fault during the brief window between CR3 load and the
        // higher-half jump. These identity maps are removed in
        // init_higher_half().
        let should_map = matches!(
            descriptor.region_type,
            EfiMemoryType::EfiConventionalMemory |
            EfiMemoryType::EfiLoaderCode         |
            EfiMemoryType::EfiLoaderData         |
            EfiMemoryType::EfiBootServicesCode   |
            EfiMemoryType::EfiBootServicesData
        );
        if !should_map { continue; }

        for i in 0..descriptor.num_pages {
            let phys = descriptor.physical_start + i * PAGE_SIZE;
            map_page(pmm, pml4, phys, phys, PRESENT | WRITABLE);
        }
    }

    // Identity map + higher-half map: kernel image
    let mut addr = pmm.kernel_start;
    while addr < pmm.kernel_end {
        map_page(pmm, pml4, addr, addr, PRESENT | WRITABLE);
        map_page(pmm, pml4, addr + globals::KERNEL_VIRTUAL_BASE, addr,
            PRESENT | WRITABLE);
        addr += PAGE_SIZE;
    }

    // Identity map + higher-half map: kernel stack with NX bit
    let stack_start = memory_map.stack_base_addr & !0xFFF;
    let stack_end = memory_map.stack_base_addr + memory_map.stack_size;
    let mut addr = stack_start;
    while addr < stack_end {
        map_page(pmm, pml4, addr, addr, PRESENT | WRITABLE | NX);
        map_page(pmm, pml4, addr + globals::KERNEL_VIRTUAL_BASE, addr,
            PRESENT | WRITABLE | NX);
        addr += PAGE_SIZE;
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
        addr += PAGE_SIZE;
    }

    // Identity map: UEFI memory map buffer
    let map_start = memory_map.memory_map_addr & !0xFFF;
    let map_end = memory_map.memory_map_addr + memory_map.memory_map_size;
    let mut addr = map_start;
    while addr < map_end {
        map_page(pmm, pml4, addr, addr, PRESENT | WRITABLE);
        addr += PAGE_SIZE;
    }

    pml4
}

// =============================================================================
// Higher-Half Kernel Transition
// This part is called after init_page_tables, and it handles the critical
// transition from a "low" (identity-mapped) memory layout set up by the UEFI
// bootloader, to a "higher-half" kernel layout where the kernel and all
// physical memory are accessed through high virtual addresses.
// =============================================================================

/// Builds a direct physical map, a contiguous window of virtual address space,
/// where every physical address P is accessible at
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
/// * `pmm` - The physical memory manager, used to allocate page-table pages.
/// * `pml4` - Raw pointer to the root of the current active page table (PML4).
/// * `memory_map` - The UEFI memory map describing all physical memory regions.
fn build_direct_map(pmm: &mut PhysicalMemoryManager, pml4: *mut u64,
    memory_map: &MemoryMapInfo
) {
    // The virtual base at which physical address 0 will appear.
    // E.g., physical PAGE_SIZE -> virtual KERNEL_PHYSICAL_MAP_BASE + PAGE_SIZE.
    let phys_map_base = globals::KERNEL_PHYSICAL_MAP_BASE;

    // Initialize the descriptor pointer and compute the number of segments
    let mut descriptor_addr = memory_map.memory_map_addr;
    let num_segments = memory_map.memory_map_size / memory_map.descriptor_size;

    for _ in 0..num_segments {
        let descriptor = EfiMemoryDescriptor::new(descriptor_addr);
        descriptor_addr += memory_map.descriptor_size;

        // MMIO and unknown types are intentionally excluded. They have
        // device-specific mapping requirements and must never be touched here.
        let should_map = matches!(
            descriptor.region_type,
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

        // Choose page-table flags appropriate for this region type. Runtime
        // services code needs to be executable so UEFI functions can run.
        // Everything else is marked NX as a security measure.
        let flags = match descriptor.region_type {
            EfiMemoryType::EfiRuntimeServicesCode => PRESENT | WRITABLE,
            _                                     => PRESENT | WRITABLE | NX,
        };

        // Map every 4 KB page in this descriptor's physical range
        for i in 0..descriptor.num_pages {
            let phys = descriptor.physical_start + i * PAGE_SIZE;
            map_page(pmm, pml4, phys_map_base + phys, phys, flags);
        }
    }
}

/// Maps the linear GPU framebuffer to its permanent virtual address in the
/// kernel's higher-half virtual address space.
///
/// After the identity maps are removed, the kernel cannot access the
/// framebuffer through its physical address. This function creates a virtual
/// mapping at `KERNEL_FRAMEBUFFER_VIRTUAL_BASE` so the kernel's graphics code
/// can continue drawing to the screen.
///
/// # Arguments
/// 
/// * `pmm` - Physical memory manager (for allocating page-table pages).
/// * `pml4` - Root of the active page table.
/// * `framebuffer_info` - UEFI-provided information about the framebuffer.
fn map_framebuffer_higher_half(pmm: &mut PhysicalMemoryManager, pml4: *mut u64,
    framebuffer_info: &crate::FramebufferInfo
) {
    // Calculate the total byte size of the framebuffer. Height * width gives
    // the total number of pixels; then, multiplying by (bpp / 8) converts
    // bits-per-pixel to bytes-per-pixel.
    let fb_size = (framebuffer_info.framebuffer_height as u64)
        * (framebuffer_info.framebuffer_width as u64)
        * (framebuffer_info.framebuffer_bpp as u64 / 8);

    // Round the framebuffer's physical start address down to a 4 KB boundary.
    // `& !0xFFF` clears the low 12 bits, aligning downward. This ensures we
    // don't miss the partial first page if the framebuffer doesn't start on a
    // page boundary.
    let fb_phys_start = framebuffer_info.framebuffer_addr & !0xFFF;

    // Round the framebuffer's physical end address up to a 4 KiB boundary.
    // Adding `0xFFF` before masking ensures we round up rather than truncating.
    // This guarantees we don't miss the partial last page.
    let fb_phys_end   = (framebuffer_info.framebuffer_addr + fb_size + 0xFFF)
        & !0xFFF;

    // Walk through every 4 KB page in the framebuffer's physical range and
    // create a virtual mapping for each one.
    let mut phys = fb_phys_start;
    let mut virt = globals::KERNEL_FRAMEBUFFER_VIRTUAL_BASE;
    while phys < fb_phys_end {
        map_page(pmm, pml4, virt, phys, PRESENT | WRITABLE);
        phys += PAGE_SIZE;
        virt += PAGE_SIZE;
    }
}

/// Updates the GDT and IDT descriptor registers so they point to the virtual
/// (higher-half) addresses of those tables, rather than their old physical
/// (identity-mapped) addresses. This must be called after `build_direct_map()`
/// and before `remove_identity_maps()`.
///
/// The GDT (Global Descriptor Table) and IDT (Interrupt Descriptor Table) are
/// located in physical memory. Before the higher-half switch, the CPU's GDTR
/// and IDTR registers contained their physical addresses (which worked because
/// identity mapping made phys == virt).
/// 
/// # Safety
/// 
/// Uses inline assembly to store and reload GDTR and IDTR.
fn reload_gdt_and_idt() {
    // Buffers to hold the 10-byte (2 bytes limit + 8 bytes base) GDTR and IDTR
    // pseudo-descriptor structures.
    let mut gdt_desc = [0u8; 10];
    let mut idt_desc = [0u8; 10];

    unsafe {
        core::arch::asm!(
            "sgdt [{gdt}]",     // Store current GDTR into memory at gdt_desc
            "sidt [{idt}]",     // Store current IDTR into memory at idt_desc
            gdt = in(reg) gdt_desc.as_mut_ptr(),  // Pass pointer to gdt_desc
            idt = in(reg) idt_desc.as_mut_ptr(),  // Pass pointer to idt_desc
        )
    };

    // Decode the current (physical-address) descriptors by extracting the
    // 64-bit physical base address from bytes 2–9
    let gdt_phys = u64::from_le_bytes(gdt_desc[2..10].try_into().unwrap());
    let idt_phys = u64::from_le_bytes(idt_desc[2..10].try_into().unwrap());

    // Extract the 16-bit limit of the GDT and IDT (bytes 0–1)
    let gdt_limit = u16::from_le_bytes(gdt_desc[0..2].try_into().unwrap());
    let idt_limit = u16::from_le_bytes(idt_desc[0..2].try_into().unwrap());

    // Compute the direct-map virtual addresses
    let gdt_virt = gdt_phys + globals::KERNEL_PHYSICAL_MAP_BASE;
    let idt_virt = idt_phys + globals::KERNEL_PHYSICAL_MAP_BASE;

    // Build the new 10-byte pseudo-descriptors with virtual base addresses 
    let mut new_gdt = [0u8; 10];
    let mut new_idt = [0u8; 10];

    // Write the limit (unchanged) into bytes 0–1
    new_gdt[0..2].copy_from_slice(&gdt_limit.to_le_bytes());
    new_idt[0..2].copy_from_slice(&idt_limit.to_le_bytes());

    // Write the new virtual base address into bytes 2–9
    new_gdt[2..10].copy_from_slice(&gdt_virt.to_le_bytes());
    new_idt[2..10].copy_from_slice(&idt_virt.to_le_bytes());

    // Load the updated descriptors into GDTR and IDTR
    unsafe {
            core::arch::asm!(
            "lgdt [{gdt}]",  // Load the new GDT descriptor
            "lidt [{idt}]",  // Load the new IDT descriptor
            gdt = in(reg) new_gdt.as_ptr(), // Pointer to the new GDT descriptor
            idt = in(reg) new_idt.as_ptr(), // Pointer to the new IDT descriptor
        )
    };
}

/// Installs the kernel's statically defined GDT and performs a far return
/// to reload the Code Segment (CS) register with the new 64-bit code selector.
///
/// The far return is needed because  the CS register cannot be changed with a
/// normal `MOV` instruction. The only ways to update CS in long mode are:
///   - A far CALL / far JMP / far RET (which load a new CS:RIP pair atomically)
///   - An interrupt return (IRETQ)
///
/// The trick used here is to push a fake "return address" (new CS selector +
/// new RIP) onto the stack and execute `RETFQ` (64-bit far return), which pops
/// CS and RIP simultaneously, effectively performing a long jump to the
/// instruction after `RETFQ` with the new CS loaded.
///
/// # Safety
/// 
/// Uses inline assembly to load the kernel's own GDT.
fn load_gdt() {
    unsafe {
        // Build a 10-byte GDTR pseudo-descriptor on the stack as a byte array
        // to avoid any alignment or packed struct issues.
        let mut gdtr = [0u8; 10];

        // The reason for `- 1` is that the CPU adds 1 when interpreting the
        // size of GDT structure.
        let limit = (core::mem::size_of::<Gdt>() - 1) as u16;

        // Using the virtual address of the static GDT object
        let base  = &GDT as *const Gdt as u64;

        // Pack limit (bytes 0–1) and base (bytes 2–9) in little-endian order
        gdtr[0..2].copy_from_slice(&limit.to_le_bytes());
        gdtr[2..10].copy_from_slice(&base.to_le_bytes());

        core::arch::asm!(
            "lgdt [{gdtr}]",  // Load the new GDT from our gdtr buffer
            "push 0x8",       // Push the new code segment selector
            "lea rax, [rip + 3f]",  // Compute the address of label "3:"
            "push rax",  // Push it as the return address
            "retfq",     // Far return: execution continues at label 3 below
            "3:",        // Execution resumes here with the new CS loaded
            "mov ax, 0x10",  // Reload the GDT index 2 segment register
            "mov ds, ax",    // Reload the data segment
            "mov es, ax",    // Reload the extra segment
            "mov ss, ax",    // Reload the stack segment
            "mov fs, ax",    // Reload FS
            "mov gs, ax",    // Reload GS
            gdtr = in(reg) gdtr.as_ptr(),  // Pointer to the GDTR buffer
            out("rax") _,    // RAX is clobbered; discard the output
        )
    };
}

/// Removes the identity-mapped lower half of the virtual address space by
/// zeroing out the lower 256 PML4 entries to unmap everything below the
/// halfway point.
///
/// After modifying the page tables, CR3 must be reloaded so the CPU discards
/// any cached translations in TLB. After this point no address below
/// 0xFFFF800000000000 is valid. This is the point of no return for
/// identity-mapped access. PML4 entries 0–255 span the entire lower half of
/// the canonical address space (0x0000000000000000 – 0x00007FFFFFFFFFFF).
/// Entries 256–511 are the higher half and must be left intact.
///
/// # Arguments
/// * `pml4_phys` - The physical address of the PML4 table. We need the physical
///   address to compute the new virtual address through the direct map.
/// 
/// # Safety
/// 
/// Uses `write_volatile` to prevent the compiler from optimising these away.
fn remove_identity_maps(pml4_phys: u64) {
    // Convert the physical PML4 address to a virtual address through the
    // direct physical map that we just built.
    let pml4_virt = (pml4_phys + globals::KERNEL_PHYSICAL_MAP_BASE) as *mut u64;
    
    // Zero out the first 256 PML4 entries (the lower half).
    for i in 0..256usize {
        unsafe { pml4_virt.add(i).write_volatile(0u64) };
    }

    // Flush the entire TLB by writing the current CR3 value back to CR3.
    // CR3 holds the physical address of the active PML4; rewriting it forces
    // the CPU to discard all cached page-table walks. We need to use an
    // inline CR3 reload here rather than calling refresh_tlb(). A function call
    // after zeroing the PML4 entries may cause the compiler to emit a stack
    // access before the TLB is flushed, hitting a now-unmapped low address and
    // triple faulting.
    unsafe {
        core::arch::asm!(
            "mov rax, cr3",  // Read current CR3 (PML4 address) into rax
            "mov cr3, rax",  // Write it back; this flushes the TLB.
            out("rax") _,    // Tell the compiler RAX is clobbered (no output).
        );
    }
}

/// Returns all bootloader and UEFI boot-services memory to the physical memory
/// allocator so the kernel can reuse those pages as general-purpose RAM.
///
/// These pages could not be freed earlier because they contained data the
/// kernel needed during initialisation, but they are now safe to reclaim as the
/// last step in `init_higher_half`. The following are intentionally not
/// reclaimed here:
///   - EfiRuntimeServicesCode/Data  (firmware reserved; do not touch)
///   - EfiACPIMemoryNVS             (ACPI firmware tables; permanent)
///   - EfiACPIReclaimMemory         (reclaim separately after ACPI init)
///   - The memory map buffer itself (still being iterated)
///   - The kernel image and stack
///
/// The EfiConventionalMemory pages are already in the PMM from init() and must
/// not be double-freed.
///
/// # Arguments
/// 
/// * `pmm` - The physical memory manager to free pages back into.
/// * `memory_map` - The UEFI memory map (now accessed through the direct map
///   using its virtual address).
///
/// # Safety
/// 
/// This function is marked `#[inline(never)]` to ensure the compiler does not
/// inline it into `init_higher_half`. Inlining could cause the compiler to
/// keep local variables (like the loop counter or descriptor pointer) in
/// registers or on the stack across the point where we free the stack pages
/// themselves, which could be catastrophic. By keeping it a separate
/// function call, the stack frame is set up and torn down cleanly.
#[inline(never)]
fn reclaim_boot_memory(pmm: &mut PhysicalMemoryManager,
    memory_map: &MemoryMapInfo
) {
    // Obtain the virtual address of the first descriptor
    let mut descriptor_addr = memory_map.memory_map_addr
        + globals::KERNEL_PHYSICAL_MAP_BASE;
    
    // Number of descriptor entries in the map
    let num_segments = memory_map.memory_map_size / memory_map.descriptor_size;

    // Physical address range occupied by the memory map array itself. We must
    // not free pages that contain the map while we are iterating over it.
    let map_start    = memory_map.memory_map_addr;
    let map_end      = memory_map.memory_map_addr + memory_map.memory_map_size;

    for _ in 0..num_segments {
        // Parse the current descriptor and advance the address for next loop
        let desc = EfiMemoryDescriptor::new(descriptor_addr);
        descriptor_addr += memory_map.descriptor_size;

        // We only reclaim loader and boot-services pages
        let reclaimable = matches!(
            desc.region_type,
            EfiMemoryType::EfiLoaderCode       |
            EfiMemoryType::EfiLoaderData       |
            EfiMemoryType::EfiBootServicesCode |
            EfiMemoryType::EfiBootServicesData
        );
        if !reclaimable { continue; }

        // Iterate over each 4 KB page within this descriptor's range
        for i in 0..desc.num_pages {
            // Physical start and end addresses of this page (inclusive)
            let page_start = desc.physical_start + i * PAGE_SIZE;
            let page_end = page_start + PAGE_SIZE - 1;

            if page_start == 0 { continue; }  // Always skip the null/zero page

            // Skip pages that overlap with the UEFI memory map itself
            if page_overlaps(page_start, page_end, map_start, map_end) {
                continue;
            }

            // Skip pages that overlap with the kernel image
            if page_overlaps(page_start, page_end, pmm.kernel_start,
                pmm.kernel_end) {
                continue;
            }

            // Skip pages that overlap with the kernel stack.
            if page_overlaps(page_start, page_end, pmm.kernel_stack_base_addr,
                pmm.kernel_stack_base_addr + pmm.kernel_stack_size,
            ) { 
                continue;
            }

            // All safety checks passed; return this page to the free list
            pmm.free_page(page_start);
        }
    }
}

// =============================================================================
// MemoryManager: Public Interface
// =============================================================================
//
// This interface encapsulates the needed physical and virtual memory management
// structures and functionality, and exports them to the rest of the system.
// This interface is implemented with IRQ-safe spinlock in the system globals.

/// The top-level Memory Manager. It owns the physical memory allocator (PMM)
/// and the root of the page-table hierarchy (PML4).
///
/// After construction via [`MemoryManager::init`], the caller must invoke
/// [`MemoryManager::init_higher_half`] to complete the transition to the
/// higher-half kernel virtual address space.
pub struct MemoryManager {
    /// The physical memory manager tracks which physical pages are free/used
    /// and services `alloc_page` / `free_page` requests.
    pmm: PhysicalMemoryManager,

    /// Virtual address of the PML4 page-table root. It changes after the
    /// higher-half switch from a physical address to a virtual address through
    /// the direct map.
    pml4: *mut u64,

    /// Physical address of the PML4 root, used during early init. It stays
    /// constant throughout, even after the higher-half switch. It is needed
    /// by `remove_identity_maps`.
    pml4_phys: u64,
}

#[allow(dead_code)]
impl MemoryManager {
    /// Creates a new `MemoryManager` and enables the No-Execute (NX) bit
    /// in the CPU's EFER (Extended Feature Enable Register) MSR (Model-Specific
    /// Register).
    ///
    /// The NX bit (bit 11 of EFER) enables page-level execute-disable support.
    /// When set, individual page-table entries can mark pages as non-executable
    /// (using the NX flag in the PTE), which is a key security feature that
    /// prevents data pages from being executed as code.
    ///
    /// # Arguments
    /// 
    /// * `pmm` - An already-initialised physical memory manager.
    /// * `pml4_phys` - The physical address of the root PML4 table. It is
    ///   updated to virtual address stored in `pml4` during `init_higher_half`.
    /// 
    /// # Safety
    /// 
    /// Uses inline assembly to set the NXE bit in the CPU's EFER MSR.
    pub fn init(
        pmm: PhysicalMemoryManager,
        pml4_phys: u64,
    ) -> Self {
        // Fixed MSR address for the EFER, defined by the x86_64 architecture
        let efer_msr: u64 = 0xC0000080;

        // Variables to hold the low (EAX) and high (EDX) 32-bit halves of the
        // 64-bit MSR value. MSRs are accessed as two 32-bit halves on x86.
        let mut low: u32;
        let mut high: u32;

        unsafe {
            // Read the MSR specified in ECX into EDX:EAX. After this,
            // `low` = bits 31:0 of EFER, `high` = bits 63:32.
            core::arch::asm!(
                "rdmsr",
                in("ecx") efer_msr,  // MSR number to read
                out("eax") low,      // Low 32 bits -> low
                out("edx") high,     // High 32 bits -> high
            );

            // Set bit 11 of the low half (the NXE bit)
            low |= 1 << 11;
            
            // Write the modified value back to the MSR
            core::arch::asm!(
                "wrmsr",
                in("ecx") efer_msr,  // MSR number to write
                in("eax") low,       // Low 32 bits (with NXE now set)
                in("edx") high,      // High 32 bits (unchanged)
            );
        }

        Self {
            pmm,
            // At this point identity mapping is in effect, so the physical
            // address can be used directly as a pointer. The `pml4` starts as
            // physical and is updated yo virtual in `init_higher_half`.
            pml4: pml4_phys as *mut u64,
            pml4_phys,
        }
    }

    /// Completes the transition to the higher-half kernel layout. This function
    /// must be called once after `init`, and it performs these steps in order:
    ///
    /// 1. Builds the direct physical map
    /// 2. Updates the PMM to use virtual addresses
    /// 3. Maps the framebuffer into the high virtual framebuffer region
    /// 4. Updates `self.pml4` to point to the virtual address of the PML4
    /// 5. Reloads GDT and IDT with their new virtual base addresses
    /// 6. Loads the kernel's own minimal GDT
    /// 7. Removes the bootloader's identity maps from the lower half
    /// 8. Reclaims boot-time memory back to the PMM
    ///
    /// # Arguments
    /// 
    /// * `framebuffer_info` - UEFI framebuffer information struct.
    /// * `memory_map` - UEFI memory map with regions, types, page counts.
    /// 
    /// After this returns, no low virtual address is valid. Physical addresses
    /// are reachable via `phys_to_virt()`. UEFI is completely gone from the
    /// address space.
    pub fn init_higher_half(&mut self,
        framebuffer_info: &crate::FramebufferInfo, memory_map: &MemoryMapInfo
    ) {
        // Step 1: Map all usable physical memory into the high virtual window.
        // After this, every physical address P can be read/written at
        // KERNEL_PHYSICAL_MAP_BASE + P once switched to higher-half addressing.
        build_direct_map(&mut self.pmm, self.pml4, memory_map);

        // Step 2: Tell the PMM that it should now use higher-half (virtual)
        // addresses. Future allocations will return virtual pointers rather
        // than physical ones.
        self.pmm.is_higher_half = true;

        // Step 3: Map the GPU framebuffer into the kernel's virtual address
        // space so the graphics subsystem can still write to it after identity
        // maps are removed.
        map_framebuffer_higher_half(&mut self.pmm, self.pml4, framebuffer_info);

        // Step 4: Update `self.pml4` from a physical address pointer to a
        // virtual address pointer through the direct map. All subsequent
        // `map_page` calls will dereference this pointer and must use the
        // virtual address.
        self.pml4 = (
            self.pml4_phys + globals::KERNEL_PHYSICAL_MAP_BASE) as *mut u64;

        // Step 5: Reload GDTR and IDTR so they point to the virtual
        // (higher-half) addresses of those tables. If we skipped this step, the
        // CPU would try to handle interrupts/faults using a now-unmapped
        // physical address and would triple-fault.
        reload_gdt_and_idt();

        // Step 6: Load the kernel's own minimal 3-entry GDT (null, code, data).
        // Also reloads CS, DS, ES, SS, FS, and GS with the correct selectors.
        load_gdt();

        // Step 7: Zero out the lower 256 PML4 entries (the identity-mapped
        // region) and flush the TLB. After this, virtual addresses below
        // KERNEL_PHYSICAL_MAP_BASE are unmapped.
        remove_identity_maps(self.pml4_phys);

        // Step 8: Walk the memory map again and return all loader/boot-services
        // pages to the PMM as free pages. Those pages were previously reserved
        // (we couldn't free them until we finished using the memory map and
        // page tables that lived in those regions).
        reclaim_boot_memory(&mut self.pmm, memory_map);
    }

    /// Converts a physical address to a virtual address through the direct map.
    ///
    /// Valid only after `init_higher_half` has been called.
    ///
    /// # Arguments
    /// 
    /// * `phys` - A physical address.
    /// 
    /// # Returns
    ///
    /// Returns the corresponding virtual address in the kernel's direct map
    /// window.
    #[inline(always)]
    pub fn phys_to_virt(phys: u64) -> u64 {
        phys + globals::KERNEL_PHYSICAL_MAP_BASE
    }

    /// Converts a virtual address (in the direct map window) back to a physical
    /// address.
    ///
    /// Valid only for addresses that lie within the direct physical map.
    ///
    /// # Arguments
    /// 
    /// * `virt` - A virtual address.
    ///
    /// # Returns
    /// 
    /// Returns the underlying physical address.
    #[inline(always)]
    pub fn virt_to_phys(virt: u64) -> u64 {
        virt - globals::KERNEL_PHYSICAL_MAP_BASE
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
    pub fn map_page(&mut self, virtual_addr: u64, physical_addr: u64,
        flags: u64
    ) {
        map_page(&mut self.pmm, self.pml4, virtual_addr, physical_addr, flags);
    }

    /// Delegates to the `get_physical_addr` and resolves a virtual address to
    /// its mapped physical address.
    pub fn get_physical_addr(&self, virtual_addr: u64) -> Option<u64> {
        get_physical_addr(self.pml4, virtual_addr)
    }

    /// Unmaps the virtual page, reclaiming any intermediate page tables that
    /// become empty as a result.
    ///
    /// It does not free the underlying physical frame. The function
    /// `unmap_and_free_page` can be used for the common case where want want
    /// both. Some of the use cases when we would not want to free the physical
    /// frame after unmapping a virtual page are:
    ///   - Copy-on-write technique
    ///   - Shared memory
    ///   - Reference-counted frames
    ///   - Remapping
    ///
    /// # Returns
    /// 
    /// Returns `true` if a mapping existed and was removed, `false` if the
    /// address was already unmapped at any level of the walk.
    pub fn unmap_page(&mut self, virtual_addr: u64) -> bool {
        unmap_page(&mut self.pmm, self.pml4, virtual_addr)
    }

    /// Same as `unmap_page`, except this function also returns its physical
    /// frame to the PMM.
    ///
    /// This is the common case. But, `unmap_page` can be used directly if we
    /// need to handle the physical frame ourselves (e.g., shared mappings, COW,
    /// etc).
    pub fn unmap_and_free_page(&mut self, virtual_addr: u64) {
        if let Some(phys) = self.get_physical_addr(virtual_addr) {
            self.unmap_page(virtual_addr);
            self.pmm.free_page(phys);
        }
    }

    /// Gets the number of free physical pages from the Physical Memory Manager.
    /// 
    /// # Returns
    /// 
    /// Returns the number of free physical frames in the PMM.
    pub fn free_page_count(&self) -> u64 {
        self.pmm.free_pages
    }
}

// Implements unsafe Send for spinlock management
unsafe impl Send for MemoryManager {}
