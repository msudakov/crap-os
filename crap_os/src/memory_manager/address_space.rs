//! User Process Address Space Management
//!
//! This module provides the `AddressSpace` type, which owns the PML4 (page
//! table root) for a user process and exposes safe operations on it. The x86-64
//! address space is split at the PML4 boundary into two halves.
//!   - User space (private per-process, starts empty):
//!       0x0000_0000_0000_0000 - 0x0000_7FFF_FFFF_FFFF  (PML4 indices 0–255)
//!   - Kernel space (shared across all processes, copied from kernel PML4)
//!       0xFFFF_8000_0000_0000 - 0xFFFF_FFFF_FFFF_FFFF  (PML4 indices 256–511)
//!
//! Every user process gets its own PML4. The lower half (indices 0–255) is
//! initially empty and is populated as the process maps memory. The upper half
//! (indices 256–511) is seeded from the kernel's own PML4 at the time the
//! address space is created, so all kernel mappings (direct physical map,
//! kernel code, framebuffer, MMIO, etc.) are immediately visible in every
//! process without any per-process mapping work.
//!
//! Sharing the upper half means a kernel mapping change after process creation
//! is not automatically reflected in existing processes, as the upper-half
//! entries are copied, not aliased. For static kernel mappings (which is what
//! we have right now), this is fine. When dynamic kernel mappings are added in
//! the future (e.g., kernel modules), those would need to be propagated to all
//! live address spaces.
//!
//! The USER flag (bit 2) must be set on all page table entries that user-mode
//! code is permitted to access. The kernel upper-half entries are copied
//! without the USER flag, so user-mode code cannot read or write kernel memory
//! even though the mappings are present in the PML4. The hardware enforces
//! this, and any ring-3 access to a supervisor-only page triggers a #PF.

use crate::memory_manager::PhysicalMemoryManager;
use crate::memory_manager::vmm::map_page;
use crate::globals::{KERNEL_PHYSICAL_MAP_BASE, PAGE_SIZE};

/// Owns the PML4 (top-level page table) for a single user process.
///
/// TODO: 
/// When dropped, the PML4 page itself is currently leaked, as a full teardown
/// (walking the page table and freeing every intermediate table and mapped
/// user page) will be added when process exit is implemented. For now,
/// `AddressSpace` is only created and never destroyed.
pub struct AddressSpace {
    /// Physical address of this process's PML4 page. This is the value loaded
    /// into CR3 when the process is scheduled.
    pub pml4_phys: u64,
}

impl AddressSpace {
    /// Creates a new user address space.
    ///
    /// Allocates a fresh PML4 page, zeroes the lower half (user space, indices
    /// 0–255), and copies the kernel's upper-half PML4 entries (indices
    /// 256–511) so that all kernel mappings are immediately accessible from
    /// this address space.
    ///
    /// # Arguments
    ///
    /// * `pmm`              - Physical memory manager used to allocate the PML4
    ///                        page.
    /// * `kernel_pml4_phys` - Physical address of the kernel's own PML4, used
    ///                        as the source for the upper-half copy.
    ///
    /// # Returns
    ///
    /// A new `AddressSpace`, whose `pml4_phys` is ready to be loaded into CR3.
    ///
    /// # Panics
    ///
    /// Panics with `[CRITICAL] OOM` if the physical memory manager cannot
    /// allocate a page for the new PML4.
    pub fn new(pmm: &mut PhysicalMemoryManager, kernel_pml4_phys: u64) -> Self {
        // Allocate one physical page for the new PML4
        let pml4_phys = pmm.alloc_page()
            .expect("[CRITICAL] OOM allocating user PML4");

        // Compute the virtual address of the new PML4 via the direct physical
        // map, so we can write to it.
        let pml4_virt = pml4_phys + KERNEL_PHYSICAL_MAP_BASE;
        let pml4 = pml4_virt as *mut u64;

        // Zero the entire PML4, clearing both halves
        for i in 0..512 {
            unsafe { pml4.add(i).write_volatile(0u64) };
        }

        // Copy kernel upper-half PML4 entries (indices 256–511) from the
        // kernel's PML4 into the new PML4. This gives the process access to
        // all kernel mappings without requiring any per-process kernel mapping.
        //
        // These entries do not have the USER bit set, so ring-3 code cannot
        // dereference them, as the hardware will fault on any such attempt.
        // They exist solely so that when the CPU is running kernel code on
        // behalf of this process (e.g., handling a syscall or interrupt), the
        // kernel's own virtual address space is reachable without a CR3 switch.
        let kernel_pml4_virt = kernel_pml4_phys + KERNEL_PHYSICAL_MAP_BASE;
        let kernel_pml4 = kernel_pml4_virt as *const u64;

        for i in 256..512 {
            let kernel_entry = unsafe { kernel_pml4.add(i).read_volatile() };
            unsafe { pml4.add(i).write_volatile(kernel_entry) };
        }

        AddressSpace { pml4_phys }
    }

    /// Maps a single page in this address space.
    ///
    /// Thin wrapper around the VMM's `map_page` that uses this address space's
    /// PML4 as the root. Intermediate page tables are allocated from `pmm` as
    /// needed.
    ///
    /// # Arguments
    ///
    /// * `pmm`           - Physical memory manager for intermediate table
    ///                     allocation.
    /// * `virtual_addr`  - The virtual address to map; must be page-aligned.
    /// * `physical_addr` - The physical address to map to, which also must be
    ///                     page-aligned.
    /// * `flags`         - Page table entry flags (PRESENT, WRITABLE, USER,
    ///                     NX, etc.). The caller is responsible for including
    ///                     the USER flag for any page that user-mode code must
    ///                     be able to access.
    /// * `is_user_page`  - True for user mappings, false for kernel mappings
    ///                     within a user address space.
    pub fn map_page(
        &self,
        pmm: &mut PhysicalMemoryManager,
        virtual_addr: u64,
        physical_addr: u64,
        flags: u64,
        is_user_page: bool,
    ) {
        let pml4 = (self.pml4_phys + KERNEL_PHYSICAL_MAP_BASE) as *mut u64;
        unsafe { map_page(pmm, pml4, virtual_addr, physical_addr, flags,
            is_user_page) };
    }

    /// Wraps an already-loaded PML4 physical address in an `AddressSpace`.
    ///
    /// Used by kernel processes (including idle and system) that were created
    /// during system initialization and already have a valid PML4 loaded in
    /// CR3. No allocation is performed, and no entries are modified, as this
    /// is purely a type-level wrapper.
    /// 
    /// # Arguments
    ///
    /// * `pml4_phys` - Physical address of the kernel's own PML4.
    pub fn from_existing(pml4_phys: u64) -> Self {
        AddressSpace { pml4_phys }
    }
}
