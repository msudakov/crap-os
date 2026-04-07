//! Physical Memory Manager
//!
//! The main job of the Physical Memory Manager is to allocate, free, and keep
//! track of physical memory pages in RAM. But, it has to do it really-really
//! efficiently because the entire system, including the Virtual Memory Manager,
//! depends on it for this one task. It has to be as fast as possible. 
//!
//! There are several methods of keeping track of physical pages. For example,
//! the bitmap method is able to locate a new free page in time O(n) when
//! unoptimized and down to O(log n) with some optimizations. However, this
//! implementation uses a simpler method that is able to fetch a new page and
//! also release a page in runtime of O(1), or in deterministic time.
//!
//! Specifically, it uses a stack-type (LIFO) singly-linked list.
//! Besides a counter for the number of free pages remaining in RAM, its
//! `free_list_head` member always points to the first/next free physical page
//! to be delivered when requested. In turn, each free physical page is
//! modified to have its first 8 bytes hold the address of the next page, and
//! so on.
//!
//! Each free page points to the next. Every time a page is freed and recycled
//! back to the manager, the PMM will take the current head address, place it
//! in the first 8 bytes of the newly-freed page to bump the old top page down,
//! and then update the head to point to the newly-freed page. And when a page
//! is allocated, the reverse takes place: the PMM follows the head to the
//! soon-to-be allocated page to read its first 8 bytes and find the
//! next-in-line page for later allocations. It then updates the head address
//! to point to the following page and returns the requested page to the caller.

use crate::memory_manager::{MemoryMapInfo, EfiMemoryDescriptor, EfiMemoryType};
use crate::globals::{PAGE_SIZE, KERNEL_PHYSICAL_MAP_BASE,
    __kernel_phys_start, __kernel_phys_end};

/// Physical Memory Manager structure.
#[allow(dead_code)]
pub struct PhysicalMemoryManager {
    kernel_load_addr: u64,       // Where the bootloader mapped the kernel image
    kernel_image_size: u64,      // Size of the kernel
    pub kernel_stack_base_addr: u64, // Base address of the initial kernel stack
    pub kernel_stack_size: u64,  // Size of the initial kernel stack
    framebuffer_addr: u64,       // Where the framebuffer is located
    framebuffer_size: u64,       // Framebuffer total size
    memory_map_addr: u64,        // Memory map structure address from bootloader
    memory_map_size: u64,        // Total memory map size
    memory_map_desc_size: u64,   // Size of a memory map descriptor structure
    pub kernel_start: u64,       // From `__kernel_phys_start` linker tag
    pub kernel_end: u64,         // From `__kernel_phys_end` linker tag
    pub is_higher_half: bool,    // True if running in higher-half kernel
    free_list_head: Option<u64>, // Physical address of the next free page
    pub free_pages: u64,         // Total number of remaining free pages
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
    pub fn free_page(&mut self, address: u64) {
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
    pub fn alloc_page(&mut self) -> Option<u64> {
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
pub fn page_overlaps(page_start: u64, page_end: u64, region_start: u64,
    region_end: u64
) -> bool {
    page_start <= region_end && page_end >= region_start
}
