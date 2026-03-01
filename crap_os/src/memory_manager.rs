// =============================================================================
// CrapOS Memory Manager Module
// =============================================================================
// 
// This module is responsible for all physical and virtual memory operations
// in the system.

use crate::{globals, sprintln};

const PRESENT: u64 = 1 << 0;  // Must be 1 for the entry to be valid
const WRITABLE: u64 = 1 << 1; // If 1, writes are allowed; if 0, read-only
// const USER: u64 = 1 << 2;     // If 1, user-mode access is allowed
// TODO: Not implementing the NX/Execute Disabled bit (bit 63) for now

// This is the structure received from the bootloader
#[repr(C)]
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

// UEFI conventional memory region types by code
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

struct PhysicalMemoryManager {
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
}

impl PhysicalMemoryManager {
    /// Instantiates and initializes the Physical Memory Manager.
    ///
    /// # Arguments
    ///
    /// * `framebuffer_info` - Framebuffer info structure from the bootloader.
    /// * `memory_map` - Memory map information structure from the bootloader.
    fn init(framebuffer_info: &crate::FramebufferInfo,
        memory_map: &MemoryMapInfo,
    ) -> Self {
        let fb_size = (framebuffer_info.framebuffer_height as u64) *
            (framebuffer_info.framebuffer_width as u64) *
            (framebuffer_info.framebuffer_bpp as u64);

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
        };

        // Initialize the PMM
        let mut descriptor_addr = pmm.memory_map_addr;
        let num_segments = pmm.memory_map_size / pmm.memory_map_desc_size;

        // Traverse the memory map, map out usable regions, and free all pages
        for _ in 0..num_segments {
            let memory_descriptor = EfiMemoryDescriptor::new(descriptor_addr);
            descriptor_addr += pmm.memory_map_desc_size;

            // We'll start working only with basic available memory without
            // reclaiming boot loader and services memory for now.
            if memory_descriptor.region_type != EfiMemoryType::EfiConventionalMemory {
                continue;
            }

            for i in 0..memory_descriptor.num_pages {
                let page_start_addr = memory_descriptor.physical_start+i*0x1000;
                let page_end_addr = page_start_addr + 0x1000 - 1;

                // Check the page address for collisions with existing
                // allocations. This should not happen as the bootloader
                // should have accounted for most of this, but this is
                // needed as a sanity check.

                if page_start_addr == 0 {
                    continue;  // Skipping physical page 0 (null page)
                }

                // Detect collision on the UEFI memory map region
                if page_overlaps(page_start_addr, page_end_addr,
                    pmm.memory_map_addr,
                    pmm.memory_map_addr + pmm.memory_map_size
                ) {
                    continue;
                }
                
                // Detect collision on the mapped kernel image memory region
                if page_overlaps(page_start_addr, page_end_addr,
                    pmm.kernel_load_addr,
                    pmm.kernel_load_addr + pmm.kernel_image_size
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
    fn free_page(&mut self, address: u64) {
        let ptr = address as *mut u64;
        unsafe { *ptr =  self.free_list_head.unwrap_or(0) };
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
    fn alloc_page(&mut self) -> Option<u64> {
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

/// Instructs the MMU to invalidate a virtual memory page and refresh its
/// virtual mappings.
/// 
/// # Arguments
///
/// * `pmm` - Physical memory manager.
/// * `framebuffer_info` - Framebuffer info structure from the bootloader.
/// * `memory_map` - Memory map information structure from the bootloader.
/// 
/// # Safety
/// 
/// Uses inline assembly to write the new PML4 address to the CR3 register.
fn init_page_tables(pmm: &mut PhysicalMemoryManager,
    framebuffer_info: &crate::FramebufferInfo, memory_map: &MemoryMapInfo
) -> *mut u64 {
    // Allocate and zero out the new PML4
    let new_pml4 = pmm.alloc_page().expect("[CRITICAL] OOM allocating PML4!");
    zero_out_page(new_pml4);
    let pml4 = new_pml4 as *mut u64;

    // Identity map all EfiConventionalMemory regions
    let mut descriptor_addr = memory_map.memory_map_addr;
    let num_segments = memory_map.memory_map_size / memory_map.descriptor_size;

    for _ in 0..num_segments {
        let descriptor = EfiMemoryDescriptor::new(descriptor_addr);
        descriptor_addr += memory_map.descriptor_size;

        if descriptor.region_type != EfiMemoryType::EfiConventionalMemory {
            continue;
        }

        for i in 0..descriptor.num_pages {
            let phys = descriptor.physical_start + i * 0x1000;
            map_page(pmm, pml4, phys, phys, PRESENT | WRITABLE);
        }
    }

    // Identity map the kernel image
    let kernel_start = memory_map.kernel_load_addr & !0xFFF;
    let kernel_end = memory_map.kernel_load_addr + memory_map.kernel_image_size;
    let mut addr = kernel_start;
    while addr < kernel_end {
        map_page(pmm, pml4, addr, addr, PRESENT | WRITABLE);
        addr += 0x1000;
    }

    // Identity map the kernel stack
    let stack_start = memory_map.stack_base_addr & !0xFFF;
    let stack_end = memory_map.stack_base_addr + memory_map.stack_size;
    let mut addr = stack_start;
    while addr < stack_end {
        map_page(pmm, pml4, addr, addr, PRESENT | WRITABLE);
        addr += 0x1000;
    }

    // Identity map the framebuffer
    let fb_size = (framebuffer_info.framebuffer_height as u64)
        * (framebuffer_info.framebuffer_width as u64)
        * (framebuffer_info.framebuffer_bpp as u64);
    let fb_start = framebuffer_info.framebuffer_addr & !0xFFF;
    let fb_end = framebuffer_info.framebuffer_addr + fb_size;
    let mut addr = fb_start;
    while addr < fb_end {
        map_page(pmm, pml4, addr, addr, PRESENT | WRITABLE);
        addr += 0x1000;
    }

    // Identity map the UEFI memory map buffer
    let map_start = memory_map.memory_map_addr & !0xFFF;
    let map_end = memory_map.memory_map_addr + memory_map.memory_map_size;
    let mut addr = map_start;
    while addr < map_end {
        map_page(pmm, pml4, addr, addr, PRESENT | WRITABLE);
        addr += 0x1000;
    }

    // Switch CR3 to the new PML4
    unsafe {
        core::arch::asm!(
            "mov cr3, {}",
            in(reg) new_pml4,
            options(nostack, preserves_flags)
    )};

    sprintln!("[INFO] Switched to new page tables");
    pml4
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
    pml4: *mut u64,
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
        framebuffer_info: &crate::FramebufferInfo,
        memory_map: &MemoryMapInfo,
    ) -> Self {
        let mut pmm = PhysicalMemoryManager::init(framebuffer_info, memory_map);
        let pml4 = init_page_tables(&mut pmm, framebuffer_info, memory_map);
        Self { pmm, pml4 }
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
}

// Implements unsafe Send for spinlock management
unsafe impl Send for MemoryManager {}



// TODO: For sanity checks only. Delete later
pub fn test_vmm() {
    unsafe {
        let cookie: u64 = 0xDEADBEEFCAFEBABE;
        let phys_addr = globals::MEMORY_MANAGER.lock().as_mut().unwrap().alloc_page().unwrap();
        sprintln!("\n[TEST] Got physical address 0x{:016X}", phys_addr);
        let virt_addr: u64 = 0x0000010000000000;
        sprintln!("[TEST] Chosen virtual address 0x{:016X}", virt_addr);
        let phys_ptr = phys_addr as *const u64;
        sprintln!("[TEST] Read from physical address 0x{:016X}", *phys_ptr);
        let cr3: u64;
        core::arch::asm!("mov {}, cr3", out(reg) cr3);
        sprintln!("[TEST] CR3 / PML4 address 0x{:016X}", cr3);
        globals::MEMORY_MANAGER.lock().as_mut().unwrap().map_page(virt_addr, phys_addr, PRESENT | WRITABLE);
        sprintln!("[TEST] Mapped new virtual page");
        let ptr = virt_addr as *mut u64;
        *ptr = cookie;
        sprintln!("[TEST] Wrote cookie to virtual address");
        sprintln!("[TEST] Read cookie from virtual address 0x{:016X}", *ptr);
        sprintln!("[TEST] Read cookie from physical address 0x{:016X}", *phys_ptr);
    }
}
