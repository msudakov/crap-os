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
//   0xFFFFA00000000000  Kernel heap              64 MB max heap size
//   0xFFFFFFFF80000000  Kernel image + stack     Sits in the top 2GB of the
//                                                48-bit canonical address space
//
// UEFI runtime services are intentionally NOT mapped. ExitBootServices() was
// already called in the bootloader, making SetVirtualAddressMap() illegal.
// Shutdown and reset will be handled via ACPI and legacy port 0x64,
// respectively, with zero firmware involvement at runtime.

use core::alloc::{GlobalAlloc, Layout};
use crate::globals::{PAGE_SIZE, KERNEL_PHYSICAL_MAP_BASE, KERNEL_VIRTUAL_BASE,
    KERNEL_FRAMEBUFFER_VIRTUAL_BASE, MEMORY_MANAGER, KERNEL_HEAP};
use core::ptr;

pub const PRESENT: u64 = 1 << 0;   // Must be 1 for the entry to be valid
pub const WRITABLE: u64 = 1 << 1;  // If 1, writes are allowed; if 0, read-only
// pub const USER: u64 = 1 << 2;     // If 1, user-mode access is allowed
pub const NX: u64 = 1 << 63;

// Kernel physical start and physical end tags collected from the linker
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
// to point to the following page and returns the requested page to the caller.

/// Physical Memory Manager structure.
#[allow(dead_code)]
pub struct PhysicalMemoryManager {
    kernel_load_addr: u64,       // Where the bootloader mapped the kernel image
    kernel_image_size: u64,      // Size of the kernel
    kernel_stack_base_addr: u64, // Base address of the initial kernel stack
    kernel_stack_size: u64,      // Size of the initial kernel stack
    framebuffer_addr: u64,       // Where the framebuffer is located
    framebuffer_size: u64,       // Framebuffer total size
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
        // bits-per-pixel instead of bytes-per-pixel.
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
                EfiMemoryType::EfiConventionalMemory {
                    continue;
            }

            // Loop through every page in the descriptor segment
            for i in 0..memory_descriptor.num_pages {
                let page_start_addr =
                    memory_descriptor.physical_start + i * PAGE_SIZE;
                let page_end_addr = page_start_addr + PAGE_SIZE - 1;

                if page_start_addr == 0 {
                    continue;  // Skipping physical page 0 (null page)
                }

                // Skip pages that collide with regions we depend on.
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
            address + KERNEL_PHYSICAL_MAP_BASE
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
            head + KERNEL_PHYSICAL_MAP_BASE
        } else {
            head
        };

        let next = unsafe { *(virt as *const u64) };
        self.free_list_head = if next == 0 {
            None
        } else {
            Some(next)
        };

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
/// `table` must be a valid, aligned pointer to a mapped 512-entry page table.
/// `index` must be in range [0, 511]. The PMM must be in a consistent state.
unsafe fn get_or_create_table(pmm: &mut PhysicalMemoryManager, table: *mut u64,
    index: u64
) -> *mut u64 {
    unsafe {
        let entry = table.add(index as usize);

        if *entry & PRESENT == 0 {
            let new_table_phys = pmm.alloc_page().expect(
                "[CRITICAL] OOM during page table walk!");
            let new_table_virt = if pmm.is_higher_half {
                new_table_phys + KERNEL_PHYSICAL_MAP_BASE
            } else {
                new_table_phys
            };
            zero_out_page(new_table_virt);
            *entry = new_table_phys | PRESENT | WRITABLE;
        }

        let phys = *entry & !0xFFF;
        let virt = if pmm.is_higher_half {
            phys + KERNEL_PHYSICAL_MAP_BASE
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
/// `pml4_addr` must point to a valid, mapped PML4 table. `physical_addr` must
/// be a real, PMM-owned physical page address. Caller must ensure the virtual
/// address is not already mapped to a different frame unless intentionally
/// remapping.
unsafe fn map_page(pmm: &mut PhysicalMemoryManager, pml4_addr: *mut u64,
    virtual_addr: u64, physical_addr: u64, flags: u64
) {
    let pml4_index = (virtual_addr >> 39) & 0x1FF;
    let pdpt_index = (virtual_addr >> 30) & 0x1FF;
    let pd_index   = (virtual_addr >> 21) & 0x1FF;
    let pt_index   = (virtual_addr >> 12) & 0x1FF;

    let pdpt = unsafe { get_or_create_table(pmm, pml4_addr, pml4_index) };
    let pd   = unsafe { get_or_create_table(pmm, pdpt, pdpt_index) };
    let pt   = unsafe { get_or_create_table(pmm, pd, pd_index) };

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
/// `pml4_addr` must point to a valid, mapped PML4 table. The page table
/// hierarchy it points to must not be concurrently modified.
unsafe fn get_physical_addr(pml4_addr: *mut u64, virtual_addr: u64
) -> Option<u64> {
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
        let pdpt = ((*pml4e & !0xFFF) + KERNEL_PHYSICAL_MAP_BASE)
            as *mut u64;

        // PDPT -> PD
        let pdpte = pdpt.add(pdpt_index as usize);
        if *pdpte & PRESENT == 0 {
            return None;
        }
        let pd = ((*pdpte & !0xFFF) + KERNEL_PHYSICAL_MAP_BASE)
            as *mut u64;

        // PD -> PT
        let pde = pd.add(pd_index as usize);
        if *pde & PRESENT == 0 {
            return None;
        }
        let pt = ((*pde & !0xFFF) + KERNEL_PHYSICAL_MAP_BASE)
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
/// `table` must be a valid, aligned pointer to a mapped 512-entry page table
/// that remains valid and unmodified for the duration of the scan.
unsafe fn is_table_empty(table: *mut u64) -> bool {
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
        let pdpt = ((*pml4e & !0xFFF) + KERNEL_PHYSICAL_MAP_BASE)
            as *mut u64;

        // PDPT -> PD
        let pdpte = pdpt.add(pdpt_index as usize);
        if *pdpte & PRESENT == 0 {
            return false;
        }
        let pd = ((*pdpte & !0xFFF) + KERNEL_PHYSICAL_MAP_BASE)
            as *mut u64;

        // PD -> PT
        let pde = pd.add(pd_index as usize);
        if *pde & PRESENT == 0 {
            return false;
        }
        let pt = ((*pde & !0xFFF) + KERNEL_PHYSICAL_MAP_BASE)
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
/// `address` must be a valid virtual address pointing to at least 4096 bytes
/// of exclusively-owned, writable, mapped memory.
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
            unsafe { map_page(pmm, pml4, phys, phys, PRESENT | WRITABLE) };
        }
    }

    // Identity map + higher-half map: kernel image
    let mut addr = pmm.kernel_start;
    while addr < pmm.kernel_end {
        unsafe { map_page(pmm, pml4, addr, addr, PRESENT | WRITABLE) };
        unsafe {
            map_page(pmm, pml4, addr + KERNEL_VIRTUAL_BASE, addr,
            PRESENT | WRITABLE)
        };
        addr += PAGE_SIZE;
    }

    // Identity map + higher-half map: kernel stack with NX bit
    let stack_start = memory_map.stack_base_addr & !0xFFF;
    let stack_end = memory_map.stack_base_addr + memory_map.stack_size;
    let mut addr = stack_start;
    while addr < stack_end {
        unsafe { map_page(pmm, pml4, addr, addr, PRESENT | WRITABLE | NX) };
        unsafe {
            map_page(pmm, pml4, addr + KERNEL_VIRTUAL_BASE, addr,
            PRESENT | WRITABLE | NX)
        };
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
        unsafe { map_page(pmm, pml4, addr, addr, PRESENT | WRITABLE) };
        addr += PAGE_SIZE;
    }

    // Identity map: UEFI memory map buffer
    let map_start = memory_map.memory_map_addr & !0xFFF;
    let map_end = memory_map.memory_map_addr + memory_map.memory_map_size;
    let mut addr = map_start;
    while addr < map_end {
        unsafe { map_page(pmm, pml4, addr, addr, PRESENT | WRITABLE) };
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
/// about raw physical addresses above the PMM layer. We can use the functions
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
    let phys_map_base = KERNEL_PHYSICAL_MAP_BASE;

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
            unsafe { map_page(pmm, pml4, phys_map_base + phys, phys, flags) };
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

    // Round the framebuffer's physical end address up to a 4 KB boundary.
    // Adding `0xFFF` before masking ensures we round up rather than truncating.
    // This guarantees we don't miss the partial last page.
    let fb_phys_end   = (framebuffer_info.framebuffer_addr + fb_size + 0xFFF)
        & !0xFFF;

    // Walk through every 4 KB page in the framebuffer's physical range and
    // create a virtual mapping for each one.
    let mut phys = fb_phys_start;
    let mut virt = KERNEL_FRAMEBUFFER_VIRTUAL_BASE;
    while phys < fb_phys_end {
        unsafe { map_page(pmm, pml4, virt, phys, PRESENT | WRITABLE) };
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
    let gdt_virt = gdt_phys + KERNEL_PHYSICAL_MAP_BASE;
    let idt_virt = idt_phys + KERNEL_PHYSICAL_MAP_BASE;

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

/// Removes the identity-mapped lower half of the virtual address space by
/// zeroing out the lower 256 PML4 entries to unmap everything below the
/// halfway point.
///
/// After modifying the page tables, CR3 must be reloaded so the CPU discards
/// any cached translations in TLB. After this point, no address below
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
    let pml4_virt = (pml4_phys + KERNEL_PHYSICAL_MAP_BASE) as *mut u64;
    
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
/// manager so the kernel can reuse those pages as general-purpose RAM.
///
/// These pages could not be freed earlier because they contained data the
/// kernel needed during initialization, but they are now safe to reclaim as the
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
        + KERNEL_PHYSICAL_MAP_BASE;
    
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

            if page_start == 0 { continue; }  // Skipping the null/zero page

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
// MemoryManager - Public Interface
// =============================================================================
//
// This interface encapsulates the needed physical and virtual memory management
// structures and functionality, and exports them to the rest of the system.
// This interface is implemented with IRQ-safe spinlock in the system globals.

/// The top-level Memory Manager. It owns the physical memory allocator (PMM)
/// and the root of the page-table hierarchy (PML4).
///
/// After construction via `MemoryManager::init`, the caller must invoke
/// `MemoryManager::init_higher_half` to complete the transition to the
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
    /// * `pmm` - An already-initialized physical memory manager.
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
            // physical and is updated to virtual in `init_higher_half`.
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

        // Step 2: Instruct the PMM that it should now use higher-half (virtual)
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
            self.pml4_phys + KERNEL_PHYSICAL_MAP_BASE) as *mut u64;

        // Step 5: Reload GDTR and IDTR so they point to the virtual
        // (higher-half) addresses of those tables. If we skipped this step, the
        // CPU would try to handle interrupts/faults using a now-unmapped
        // physical address and would triple-fault.
        reload_gdt_and_idt();

        // Step 6: Load the kernel's own minimal 3-entry GDT (null, code, data).
        // This was a placeholder step, before we implemented a better GDT. It
        // is no longer needed, and the proper GDT is initialized right after
        // the Memory Manager finishes its init steps. This is kept here for
        // completeness.

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
        phys + KERNEL_PHYSICAL_MAP_BASE
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
        virt - KERNEL_PHYSICAL_MAP_BASE
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
    /// 
    /// # Safety
    ///
    /// `virtual_addr` must be a canonical higher-half address not already in
    /// use by the kernel. `physical_addr` must be a PMM-owned physical page.
    /// Mapping a page that is already mapped to a different frame silently
    /// overwrites the PTE.
    pub unsafe fn map_page(&mut self, virtual_addr: u64, physical_addr: u64,
        flags: u64
    ) {
        unsafe {
            map_page(&mut self.pmm, self.pml4, virtual_addr, physical_addr,
                flags);
        }
    }

    /// Delegates to the `get_physical_addr` and resolves a virtual address to
    /// its mapped physical address.
    /// 
    /// # Safety
    ///
    /// The page table hierarchy must not be concurrently modified while this
    /// function is executing.
    pub unsafe fn get_physical_addr(&self, virtual_addr: u64) -> Option<u64> {
        unsafe { get_physical_addr(self.pml4, virtual_addr) }
    }

    /// Unmaps the virtual page, reclaiming any intermediate page tables that
    /// become empty as a result.
    ///
    /// It does not free the underlying physical frame. The function
    /// `unmap_and_free_page` can be used for the common case where we want
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
    /// 
    /// # Safety
    ///
    /// `virtual_addr` must be a canonical address. Unmapping a page that is
    /// still in use (e.g., part of a live stack or the heap) causes immediate
    /// undefined behaviour on next access.
    pub unsafe fn unmap_page(&mut self, virtual_addr: u64) -> bool {
        unmap_page(&mut self.pmm, self.pml4, virtual_addr)
    }

    /// Same as `unmap_page`, except this function also returns its physical
    /// frame to the PMM.
    ///
    /// This is the common case. But, `unmap_page` can be used directly if we
    /// need to handle the physical frame ourselves (e.g., shared mappings, COW,
    /// etc).
    /// 
    /// # Safety
    ///
    /// Same as `unmap_page`. Additionally, the freed physical frame must not be
    /// referenced by any other mapping (e.g., shared memory, COW) or it will be
    /// returned to the PMM while still live.
    pub unsafe fn unmap_and_free_page(&mut self, virtual_addr: u64) {
        unsafe {
            if let Some(phys) = self.get_physical_addr(virtual_addr) {
                self.unmap_page(virtual_addr);
                self.pmm.free_page(phys);
            }
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

// =============================================================================
// Kernel Heap Allocator
// =============================================================================
//
// This implementation is a linked free-list allocator backed by the kernel VMM.
// It implements `GlobalAlloc`, so Rust's `alloc` crate (Box, Vec, String, etc.)
// works transparently once registered with `#[global_allocator]`.
//
// The heap grows on demand from a fixed virtual address range
// `[base, base + max_size)`. Physical pages are mapped into that range in
// chunks via "grow" calls. Free memory is tracked as a singly-linked list of
// `FreeBlock` nodes, kept sorted by ascending address. Keeping the list sorted
// allows adjacent freed blocks to be merged (coalesced) in a single pass,
// preventing fragmentation over time.
//
// Each allocation stores an `AllocHeader` immediately before the returned
// pointer, plus a copy of the `data_offset` field in the 8 bytes directly
// before the returned data pointer. This redundant copy allows `deallocate` to
// recover the block start without reading the full header, which is crucial
// when the pointer is the only information the caller provides on `free`.
//
// > This is the memory layout of an allocated block:
//   [ AllocHeader | <alignment padding> | data_offset | <data> ]
//     ^                                                  ^^^^
//     block_ptr                                      pointer returned to caller
//
// > This is another view of the same layout of an allocated block:
//
//  +----------------------------------+  <- block_ptr (raw page-aligned base)
//  |  AllocHeader                     |
//  |    total_size: usize  (8 bytes)  |
//  |    data_offset: usize (8 bytes)  |
//  |----------------------------------|
//  |  alignment padding (0+ bytes)    |
//  |----------------------------------|
//  |  data_offset copy  (8 bytes)     |  <- data_ptr - 8
//  |----------------------------------|  <- data_ptr (returned to caller)
//  |  data  (layout.size bytes)       |
//  +----------------------------------+  <- block_ptr + total_size
//
// > This is the memory layout of a free block:
//   [ FreeBlock { size, *next } | ... unused space ... ]
//     ^
//     block_ptr
//
// > This is another view of the same layout of a free block:
//
//  +----------------------------------+  <- FreeBlock pointer
//  |  size: usize   (8 bytes)         |  Total bytes available in this block
//  |  next: *mut FreeBlock (8 bytes)  |  Pointer to next free block
//  +----------------------------------+
//
// FreeBlock::size is the total size of the free region, including the
// FreeBlock header itself. FreeBlock::next is the next free block in
// address-sorted order, or null if this is the tail of the list.
//
// > Free list ordering:
//     The free list is kept sorted by ascending block address. This makes
//     coalescing O(1) per `free` (just check immediate neighbors) at the cost
//     of O(n) sorted insertion, which is acceptable since `free` is not on a
//     hot path in a kernel heap.
//
// > Lock ordering (this must never be inverted!):
//     KERNEL_HEAP -> MEMORY_MANAGER
//
// This way, `grow()` acquires the MEMORY_MANAGER lock while the caller holds
// the KERNEL_HEAP lock. No code path may hold MEMORY_MANAGER and then trigger
// a heap allocation. Otherwise, a deadlock would be imminent.
//
// Two IRQ-safe spinlocks are necessary here, and this is what happens during a
// heap allocation:
//
// caller
// |_ GlobalHeapAllocator::alloc()
//    |_ KERNEL_HEAP.heap.lock()      -> this acquires heap lock
//    .  |_ KernelHeap::allocate()
//    .     |_ if free list has space -> no second lock needed, returns directly
//    .     |_ if free list exhausted:
//    .        |_ KernelHeap::grow()
//    .        .  |_ MEMORY_MANAGER.lock()  -> this acquires MM lock
//    .        .  .  |_ mm.alloc_page()
//    .        .  .  |_ mm.map_page()
//    .        .  |_ MEMORY_MANAGER unlocked
//    .        |_ retries free list
//    |_ KERNEL_HEAP unlocked

/// A node in the sorted free-block linked list, stored inline inside free
/// memory.
///
/// This struct is written directly into the first bytes of the free memory it
/// describes, so no separate allocation is needed. That means a free block must
/// be at least `FreeBlock::SIZE` bytes (16 bytes on x64) to hold this header.
#[repr(C)]
struct FreeBlock {
    /// Total number of bytes in this free region, including the bytes
    /// occupied by this `FreeBlock` header itself.
    size: usize,

    /// Pointer to the next `FreeBlock` in the sorted free list, or null
    /// if this is the last node.
    next: *mut FreeBlock,
}

impl FreeBlock {
    /// The size of the `FreeBlock` header in bytes (16 bytes on x64).
    ///
    /// This is used as the minimum block size; any region smaller than this
    /// cannot store the free-list node and therefore cannot be tracked.
    const SIZE: usize = core::mem::size_of::<FreeBlock>();
}

/// Header written at the very start of each allocated block at `block_ptr`,
/// before the alignment padding and data. It contains the metadata stored
/// before each live allocation.
///
/// When the caller calls `deallocate(data_ptr)`, we use the `data_offset` copy
/// stored just before `data_ptr` to walk back to `block_ptr`, then read this
/// header to get the block's total size, so we can return it to the free list.
/// The `data_offset` is always the last field, so it sits at exactly
/// `size_of::<usize>()` bytes before the `data_ptr`, regardless of how much
/// padding was inserted for alignment. This lets `deallocate` recover
/// `block_ptr` with a single pointer subtraction without needing to recompute
/// alignment arithmetic.
#[repr(C)]
struct AllocHeader {
    /// Total size of the allocated block in bytes, from `block_ptr` to the end
    /// of the data region. Used by `deallocate` to know how many bytes to
    /// return to the free list.
    total_size:  usize,
    
    /// Byte offset from `block_ptr` to the start of the callee-visible data.
    /// Equals `AllocHeader::SIZE` rounded up to the allocation's alignment.
    /// It is stored here and, again, at 8 bytes before `data_ptr`, so 
    /// `deallocate` can recover `block_ptr = data_ptr - data_offset`.
    data_offset: usize,
}

impl AllocHeader {
    /// The size of the `AllocHeader` in bytes (16 bytes on x64).
    ///
    /// Used as the base for computing `data_offset`: the data starts at least
    /// `AllocHeader::SIZE` bytes into the block, then rounded up further to
    /// satisfy the allocation's alignment requirement.
    const SIZE: usize = core::mem::size_of::<AllocHeader>();
}

// =============================================================================
// KernelHeap - the heap allocator state
// =============================================================================

/// The kernel heap allocator.
///
/// Manages a virtual address range, committing physical pages into it on
/// demand and tracking free memory with a sorted linked free list.
///
/// # Thread Safety
/// 
/// `KernelHeap` itself is NOT thread-safe, and all methods take `&mut self`.
/// Thread safety is provided by the `LockedHeap` wrapper, which protects the
/// heap behind an IRQ-safe spinlock.
pub struct KernelHeap {
    /// The virtual base address of the heap region. All heap memory lives at
    /// addresses >= `base`.
    base: u64,

    /// The maximum number of bytes the heap is allowed to grow to.
    /// `base + max_size` is the exclusive upper bound of the virtual range.
    max_size: u64,

    /// How many bytes have been committed (backed by physical pages) so far.
    /// New pages are always appended at `base + committed`.
    committed: u64,

    /// Head of the sorted free-block linked list. It is `null` when the heap
    /// has no free space (e.g., when it is freshly created, and before `init`
    /// is called).
    free_list: *mut FreeBlock,
}

// SAFETY: The heap is always accessed through the global IRQ spinlock,
// which disables interrupts for the duration of every heap operation. This
// implements unsafe Send for spinlock management.
unsafe impl Send for KernelHeap {}

impl KernelHeap {
    /// Creates a new, empty `KernelHeap` covering the virtual range
    /// `[base, base + max_size)`, with `base` inclusive and `base + max_size`
    /// exclusive.
    ///
    /// No physical memory is mapped and no free blocks exist yet. Call
    /// `KernelHeap::init` before making any allocations. The `const fn`
    /// allows this to be used in `static` initializers.
    /// 
    /// #Arguments
    /// 
    /// * `base` - The virtual base address of the heap region.
    /// * `max_size` - The maximum size of the heap in bytes.
    pub const fn new(base: u64, max_size: u64) -> Self {
        Self {
            base,
            max_size,
            committed: 0,
            free_list: ptr::null_mut(),  // No free blocks yet.
        }
    }

    /// Commits `initial_pages` pages from the PMM into the heap's virtual
    /// region and inserts them into the free list, making the heap ready for
    /// allocations.
    ///
    /// Must be called once after the MemoryManager has been placed in the
    /// global and is initialized.
    pub fn init(&mut self, initial_pages: u64) {
        self.grow(initial_pages);
    }

    /// Maps `num_pages` new physical pages into the next available virtual
    /// address range in the heap and adds them as a single free block.
    ///
    /// Pages are appended starting at `base + committed`, then `committed`
    /// is advanced by `pages * PAGE_SIZE`.
    ///
    /// Panics if the grow would exceed `max_size`, or if the physical memory
    /// manager is out of pages.
    /// 
    /// # Arguments
    /// 
    /// * `num_pages` - The number of physical pages to insert into the heap.
    fn grow(&mut self, num_pages: u64) {
        let num_bytes = num_pages * PAGE_SIZE;  // Convert page count to bytes
        
        // Enforce the hard upper bound on the heap's virtual range
        assert!(
            self.committed + num_bytes <= self.max_size,
            "[KERNEL_HEAP] Out of heap memory: grow() would exceed max size"
        );

        // The virtual address at which the new region starts. New regions are
        // always appended immediately after previously committed memory.
        let region_virtual_addr = self.base + self.committed;

        {
            // Lock the global memory manager, which also disables interrupts
            // so an interrupt handler cannot try to allocate concurrently
            // while we're in mid-grow.
            let mut mm_guard = MEMORY_MANAGER.lock();
            let mm = mm_guard.as_mut()
                .expect("[KERNEL_HEAP] MemoryManager is not initialized");

            // Allocate and map one physical page at a time
            for i in 0..num_pages {
                // Compute the virtual address for this page
                let virt_addr = region_virtual_addr + i * PAGE_SIZE;

                // Allocate a free physical page from the PMM
                let phys_addr = mm.alloc_page()
                    .expect("[KERNEL_HEAP] PMM out of memory during heap grow");

                // Map virtual to physical in the kernel's page tables.
                unsafe {
                    mm.map_page(virt_addr, phys_addr, PRESENT | WRITABLE | NX)
                };
            }
        } // MEMORY_MANAGER lock dropped here, and interrupts are re-enabled

        // Record that these bytes are now committed
        self.committed += num_bytes;

        // Add the entire newly committed region to the free list as one big
        // block. This will merge it with adjacent blocks if possible.
        unsafe {
            self.insert_free_block_sorted(region_virtual_addr as *mut u8,
                num_bytes as usize);
        }
    }

    /// Inserts a raw memory region into the sorted free list and immediately
    /// attempts to coalesce it with its neighbors. The free list is kept in
    /// ascending order of pointer value. This makes coalescing of adjacent
    /// blocks possible in O(1) after insertion.
    ///
    /// # Coalescing Algorithm
    /// 
    /// After inserting the block between `prev` and `next`:
    ///   1. If the end of the new block exactly meets the start of `next`,
    ///      merge the new block and `next` into one larger block.
    ///   2. If the end of `prev` exactly meets the start of the new block,
    ///      merge `prev` and the new block into one larger block.
    ///
    /// # Arguments
    /// * `ptr`        - Start of the memory region to add to the free list.
    /// * `total_size` - Total number of bytes in the region.
    /// 
    /// # Safety
    ///
    /// `ptr` must point to at least `FreeBlock::SIZE` bytes of valid,
    /// exclusively-owned, writable memory. `total_size` must be the true byte
    /// count of the entire region starting at `ptr`.
    unsafe fn insert_free_block_sorted(&mut self, ptr: *mut u8,
        total_size: usize
    ) {
        assert!(
            total_size >= FreeBlock::SIZE,
            "[KERNEL_HEAP] block is too small to hold FreeBlock header"
        );

        // Find the correct insertion position by walking the list until we find
        // the first node whose address is >= `ptr`.
        // After the loop:
        //   `prev` = the node that should precede our new block
        //      (if null, it means insert at head);
        //   `next` = the node that should follow our new block
        //      (if null, it means insert at tail).
        let mut prev: *mut FreeBlock = ptr::null_mut();
        let mut next = self.free_list;

        while !next.is_null() && (next as usize) < (ptr as usize) {
            prev = next;
            // Advance to the next node by reading the `next` pointer stored
            // inside the current free block.
            next = unsafe { (*next).next };
        }

        // Write the FreeBlock header into the region. Reuse the first
        // `FreeBlock::SIZE` bytes of the freed region to store the list node.
        // No separate allocation is needed.
        let block = ptr as *mut FreeBlock;
        unsafe { (*block).size = total_size };  // Block covers total_size bytes
        unsafe { (*block).next = next };  // Link to the following free block

        // Splice the new node into the list
        if prev.is_null() {
            // New block is before all existing blocks, it becomes the new head
            self.free_list = block;
        }
        else {
            // Link the previous node's `next` pointer to our new block
            unsafe { (*prev).next = block };
        }

        // Coalesce with the next block (forward merge). If the byte immediately
        // after our new block is where `next` starts, they are physically
        // adjacent and can be merged into one larger block.
        if !next.is_null() {
            // The address of the byte just past our new block
            let block_end = (block as usize) + unsafe { (*block).size };
            
            if block_end == (next as usize) {
                // Merge by absorbing `next` into `block`. Add `next`'s size to
                // our size so the combined block covers both regions.
                unsafe { (*block).size += (*next).size };

                // Skip over `next` in the list as it no longer exists as a
                // separate node.
                unsafe { (*block).next  = (*next).next };
            }
        }

        // Coalesce with the previous block (backward merge). If `prev` ends
        // exactly where our new block begins, merge `prev` and our block. At
        // this point, `block` may already be the result of a forward merge
        // from above, so the backward merge can produce a 3-way merge.
        if !prev.is_null() {
            // The address of the byte just past `prev`.
            let prev_end = (prev as usize) + unsafe { (*prev).size };

            if prev_end == (block as usize) {
                // Merge by absorbing `block` into `prev`. `prev` grows to cover
                // our block and any forward-merged region.
                unsafe { (*prev).size += (*block).size };

                // Remove `block` from the list by making `prev` point to
                // `block`'s `next`.
                unsafe { (*prev).next  = (*block).next };
            }
        }
    }

    /// Computes the `total_size` and `data_offset` for an allocation described
    /// by `layout`.
    /// 
    /// # Arguments
    /// 
    /// * `layout`  -  Layout of a memory block.
    ///
    /// # Alignment Computation
    /// 
    /// The standard formula for rounding `x` up to the next multiple of `align`
    /// (when `align` is a power of two) is:
    ///   `(x + align - 1) & !(align - 1)`
    /// where:
    ///   `x` = `AllocHeader::SIZE`  (how many header bytes precede the data)
    ///   `align` = `layout.align()` (the alignment required by Rust allocator)
    ///
    /// This ensures `data_ptr = block_ptr + data_offset` is correctly aligned
    /// for the requested type, satisfying the contract of the `GlobalAlloc`
    /// trait.
    /// 
    /// # Returns
    /// 
    /// `(total_size, data_offset)`, where:
    ///   - `data_offset` is the number of bytes from `block_ptr` to the data
    ///     start, calculated by rounding `AllocHeader::SIZE` up to
    ///     `layout.align()`.
    ///   - `total_size` is the total bytes consumed from the free list for
    ///     this allocation.
    #[inline]
    fn allocation_sizes(layout: &Layout) -> (usize, usize) {
        let align = layout.align();

        // Round the header size up to the nearest multiple of `align`. This is
        // the byte offset from `block_ptr` to where data begins.
        let data_offset = (AllocHeader::SIZE + align - 1) & !(align - 1);

        // Total bytes needed: header + padding region + data.
        let total_size  = data_offset + layout.size();

        (total_size, data_offset)
    }

    /// Allocates a block of memory, satisfying a given `layout`.
    ///
    /// Performs a first-fit search through the sorted free list. We return
    /// the first free block that is large enough, rather than searching for
    /// the best-fitting or smallest-fitting block. This runs in the order O(n)
    /// (in the number of free blocks) and tends to leave larger blocks
    /// intact at the end of the list. If no block is large enough, `grow` is
    /// called to commit more physical pages, which adds a new large free block
    /// and retries from the list head.
    /// 
    /// # Arguments
    /// 
    /// * `layout`  -  Memory block layout to satisfy with this allocation.
    /// 
    /// # Returns
    ///
    /// Returns a pointer to the start of the accessible data region,
    /// aligned as required by `layout`, or a null pointer on failure
    /// (only possible if `max_size` is exhausted and the PMM is also full).
    pub unsafe fn allocate(&mut self, layout: Layout) -> *mut u8 {
        // Compute how much raw memory we need and where the data will start
        let (total_size, data_offset) = Self::allocation_sizes(&layout);

        // `previous` and `current` are the two-pointer hand-over-hand walk used
        // to splice a node out of the singly-linked list when we find a fit.
        let mut previous: *mut FreeBlock = ptr::null_mut();
        let mut current  = self.free_list;  // Start at the head of free list

        loop {
            if current.is_null() {
                // Reached the end of the free list without finding a fit.
                // Grow the heap by at least `total_size` bytes, rounded up to
                // whole pages. We request a minimum of 4 pages (16 KB) to
                // amortize the cost of growing, as it avoids growing 1 page at
                // a time for many small allocations.
                let pages_needed =
                    ((total_size as u64 + PAGE_SIZE - 1) / PAGE_SIZE).max(4);
                self.grow(pages_needed);

                // After growing, restart the search from the beginning. The
                // new free block may have been coalesced anywhere in the list.
                current  = self.free_list;
                previous = ptr::null_mut();
                continue;
            }

            // Read the size of the current free block
            let available = unsafe { (*current).size };

            if available >= total_size {
                // This block is big enough, and we can use it

                // Read `next` before we overwrite the block's memory with the
                // header
                let next = unsafe { (*current).next };

                // Splice `current` out of the free list
                if previous.is_null() {
                    // `current` is the head of the list
                    self.free_list = next;
                }
                else {
                    // `current` is in the middle or tail, so bypass it
                    unsafe { (*previous).next = next };
                }

                // Raw pointer to the start of the block we just removed
                let block_ptr = current as *mut u8;

                // Check if there is enough space left over after the allocation
                // to create a new free block from the remainder
                let remaining = available - total_size;

                if remaining >= FreeBlock::SIZE {
                    // There is enough left over to form a valid free block.
                    // The leftover region starts immediately after this
                    // allocation.
                    let leftover_ptr = unsafe { block_ptr.add(total_size) };

                    // Re-insert the leftover into the sorted free list.
                    // This may also coalesce it with the next free block.
                    unsafe {
                        self.insert_free_block_sorted(leftover_ptr, remaining);
                    }
                }
                // If remaining < FreeBlock::SIZE, the leftover is too small to
                // track, and it becomes internal fragmentation (wasted bytes at
                // the end of this allocation). The `total_size` already
                // accounts for the data so the caller sees no side effect.

                // Write the AllocHeader at the start of the block
                let header = block_ptr as *mut AllocHeader;
                // Block size for dealloc
                unsafe { (*header).total_size  = total_size };
                // Header span for dealloc
                unsafe { (*header).data_offset = data_offset };

                // Compute and return the data pointer. The data starts
                // `data_offset` bytes into the block.
                let data_ptr = unsafe { block_ptr.add(data_offset) };

                // Write a copy of `data_offset` in the 8 bytes immediately
                // before `data_ptr`. This is the "back-pointer" that
                // `deallocate` uses to recover `block_ptr` from `data_ptr`
                // without a global lookup. This overlaps with the alignment
                // padding region, which is otherwise unused.
                // Since data_offset >= AllocHeader::SIZE + any padding, there
                // is always room for this usize before `data_ptr`.
                unsafe {
                    *(data_ptr.sub(core::mem::size_of::<usize>()) as *mut usize)
                        = data_offset
                };

                return data_ptr;
            }

            // This block is too small, and we need to advance to the next one
            previous = current;
            current  = unsafe { (*current).next };
        }
    }

    /// Frees a previously allocated block and returns it to the free list.
    ///
    /// The caller only provides `ptr` (the `data_ptr` returned by `allocate`).
    /// We need to find `block_ptr` (the raw start of the block) and its
    /// `total_size`.
    /// 
    /// This is the algorithm to recover the block boundaries from just `ptr`:
    ///  1. Read `data_offset` from `ptr - sizeof(usize)`, which was written
    ///     there by `allocate` as a back-pointer;
    ///  2. Set `block_ptr = ptr - data_offset`;
    ///  3. Cast `block_ptr` to `*const AllocHeader` and read `total_size`;
    ///  4. Call `insert_free_block_sorted(block_ptr, total_size)` to return
    ///     the memory to the free list and coalesce with any neighbors.
    ///
    /// # Arguments
    /// 
    /// * `ptr`  -  Previously-allocated pointer to be freed.
    /// 
    /// # Safety
    /// 
    /// `ptr` must be a pointer previously returned by `allocate` on this heap
    /// and not yet freed. Double-free or freeing an invalid pointer causes
    /// undefined behaviour.
    pub unsafe fn deallocate(&mut self, ptr: *mut u8) {
        // 1. Recover `data_offset` from the 8 bytes just before the
        // pointer
        let data_offset = unsafe {
            *(ptr.sub(core::mem::size_of::<usize>()) as *const usize)
        };

        // 2. Walk back from the pointer to the raw block start
        let block_ptr = unsafe { ptr.sub(data_offset) };

        // 3: Read `total_size` from the `AllocHeader` at the block start
        let header = block_ptr as *const AllocHeader;
        let total_size = unsafe { (*header).total_size };

        // 4: Return the entire block to the free list
        unsafe {
            self.insert_free_block_sorted(block_ptr, total_size)
        }
    }
}

// =============================================================================
// LockedHeap - thread-safe wrapper around KernelHeap
// =============================================================================

/// A `KernelHeap` wrapped in an `IrqSpinLock` for safe concurrent access.
///
/// All kernel heap operations go through this type. The `IrqSpinLock`
/// ensures that only one thread (or CPU core) accesses the heap at a time, and
/// that interrupts are disabled while the lock is held, preventing a deadlock
/// if an interrupt handler tries to allocate memory while an allocation is
/// already in progress on the same core.
pub struct LockedHeap {
    /// The protected heap is public so the global allocator shim can access it,
    /// but any access still requires locking.
    pub heap: crate::spinlock::StaticIrqSpinLock<KernelHeap>
}

impl LockedHeap {
    /// Constructs a `LockedHeap` wrapping a new, empty `KernelHeap`.
    ///
    /// `const fn` so this can initialize a `static` variable at compile time.
    pub const fn new(base: u64, max_size: u64) -> Self {
        Self {
            heap: crate::spinlock::StaticIrqSpinLock::new(
                KernelHeap::new(base, max_size)
            )
        }
    }
}

// =============================================================================
// GlobalHeapAllocator - Rust GlobalAlloc integration
// =============================================================================

/// A zero-sized type that implements Rust's `GlobalAlloc` trait.
///
/// By registering this as the `#[global_allocator]`, all Rust allocations in
/// the kernel (Box, Vec, String, Arc, etc.) are routed through `KERNEL_HEAP`.
///
/// The struct itself holds no state; all state lives in the `KERNEL_HEAP`
/// static, accessed via `globals::KERNEL_HEAP`.
pub struct GlobalHeapAllocator;

unsafe impl GlobalAlloc for GlobalHeapAllocator {
    /// Called by Rust's runtime whenever memory is allocated.
    ///
    /// It locks the heap, runs the first-fit allocator, and returns the
    /// received pointer. If the heap needs to grow to satisfy the request,
    /// `grow` is called automatically inside `allocate`.
    /// 
    /// # Arguments
    /// 
    /// * `layout`  -  Memory block layout to satisfy with this allocation.
    ///
    /// # Returns
    /// 
    /// Returns the pointer to allocated dynamic memory.
    /// 
    /// # Safety
    /// 
    /// The caller (Rust's allocator infrastructure) guarantees that `layout`
    /// has non-zero size and a power-of-two alignment.
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        unsafe { KERNEL_HEAP.heap.lock().allocate(layout) }
    }

    /// Called by Rust's runtime whenever memory is freed.
    ///
    /// It locks the heap and returns the block to the free list. The passed
    /// `layout` is actually ignored because Rust's `GlobalAlloc` contract
    /// requires the caller to pass the same `layout` that was used to allocate
    /// the pointer. But, our implementation stores `total_size` and
    /// `data_offset` in the block header, so we can recover all the information
    /// we need from `ptr` alone. The `layout` argument is therefore unused, but
    /// it has to be accepted to satisfy the trait.
    /// 
    /// # Arguments
    /// 
    /// * `ptr`     -  Pointer to the memory allocation to be freed.
    /// * `layout`  -  Layout that was used during allocation, but is ignored.
    ///
    /// # Safety
    /// 
    /// `ptr` must have been returned by `alloc` with a compatible `layout`
    /// and must not have been freed already.
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        // Explicitly ignore layout, as all info is in the block header
        let _ = layout;
        unsafe { KERNEL_HEAP.heap.lock().deallocate(ptr) }
    }
}
