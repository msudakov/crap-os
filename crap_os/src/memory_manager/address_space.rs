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

use super::{PRESENT, WRITABLE, USER, NX};
use crate::globals::{KERNEL_PHYSICAL_MAP_BASE, PAGE_SIZE};

/// Owns the PML4 (top-level page table) for a single user process.
///
/// TODO: 
/// When dropped, the PML4 page itself is currently leaked, as a full teardown
/// (walking the page table and freeing every intermediate table and mapped
/// user page) will be added when process exit is implemented. For now,
/// `AddressSpace` is only created and never destroyed.
#[allow(dead_code)]
pub struct AddressSpace {
    /// Physical address of this process's PML4 page. This is the value loaded
    /// into CR3 when the process is scheduled.
    pub pml4_phys: u64,

    /// Address of the top of the user stack. This is the initial RSP handed to
    /// the user thread (the top of the mapped region).
    pub user_stack_top: u64,

    /// Size of the user stack in the number of pages.
    pub user_stack_pages: u64,
}

impl AddressSpace {
    // Allocate the user stack. We allocate physical pages and map them
    // into the user address space at a fixed virtual address for now.
    // A proper VA allocator will replace this hardcoded address later.
    //
    // The user stack grows downward from USER_STACK_TOP, so the initial
    // RSP handed to the user thread is USER_STACK_TOP (the top of the
    // mapped region). We map USER_STACK_PAGES pages below that address.
    //
    // TODO: implement proper VA allocator
    const USER_STACK_PAGES: u64 = 4;  // 16 KB user stack is enough for now
    const USER_STACK_TOP: u64 = 0x7FFFFFFF0000;

    /// Creates and initializes a new user address space, using the default
    /// user stack top and number of stack pages.
    ///
    /// Allocates a fresh PML4 page, zeroes the lower half (user space, indices
    /// 0–255), and copies the kernel's upper-half PML4 entries (indices
    /// 256–511) so that all kernel mappings are immediately accessible from
    /// this address space.
    ///
    /// # Arguments
    ///
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
    pub fn init(kernel_pml4_phys: u64) -> Self {
        // Allocate one physical page for the new PML4
        let pml4_phys = {
            let mut mm_guard = crate::globals::MEMORY_MANAGER.lock();
            let mm = mm_guard.as_mut().unwrap();
            mm.alloc_page().expect("[CRITICAL] OOM allocating user PML4")
        };

        // Compute the virtual address of the new PML4 via the direct physical
        // map, so we can write to it. We could use the `zero_out_page` helper
        // function of VMM, but we'll need to write more than just zeroes, so
        // might as well do it manually here.
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
            let entry = unsafe { kernel_pml4.add(i).read_volatile() };
            unsafe { pml4.add(i).write_volatile(entry) };
        }

        // Next, we map the user stack into the new PML4. We re-acquire the
        // memory manager lock here because `vmm::map_page` needs the PMM for
        // intermediate table allocation. The new PML4 is fully initialized at
        // this point, so `map_page` will correctly build page tables rooted at
        // `pml4_phys`.
        {
            let mut mm_guard = crate::globals::MEMORY_MANAGER.lock();
            let mm = mm_guard.as_mut().unwrap();
            for i in 0..Self::USER_STACK_PAGES {
                // Allocate a physical page for this stack page
                let phys = mm.alloc_page()
                    .expect("[CRITICAL] OOM allocating user stack page");

                // Map it into the user address space, growing downward from
                // USER_STACK_TOP. Page 0 is directly below USER_STACK_TOP,
                // page 1 is one page below that, and so on.
                //
                // Flags: PRESENT | WRITABLE | USER | NX
                //   USER: ring-3 accessible
                //   NX:   stack pages should never be executable
                let virt = Self::USER_STACK_TOP - (i + 1) * PAGE_SIZE;
                let flags = PRESENT | WRITABLE | USER | NX;

                unsafe { mm.map_page(pml4_phys, virt, phys, flags) };
            }
        }

        AddressSpace {
            pml4_phys,
            user_stack_top: Self::USER_STACK_TOP,
            user_stack_pages: Self::USER_STACK_PAGES,
        }
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
    /// * `kernel_pml4_phys` - Physical address of the kernel's own PML4.
    pub fn from_existing(kernel_pml4_phys: u64) -> Self {
        AddressSpace {
            pml4_phys: kernel_pml4_phys,
            user_stack_top: Self::USER_STACK_TOP,
            user_stack_pages: Self::USER_STACK_PAGES,
        }
    }
}
