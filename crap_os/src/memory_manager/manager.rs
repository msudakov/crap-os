//! MemoryManager - Public Interface
//!
//! This interface encapsulates the needed physical and virtual memory management
//! structures and functionality, and exports them to the rest of the system.
//! This interface is implemented with IRQ-safe spinlock in the system globals.

use crate::memory_manager::MemoryMapInfo;
use crate::memory_manager::PhysicalMemoryManager;
use crate::memory_manager::vmm;
use crate::globals::KERNEL_PHYSICAL_MAP_BASE;

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
        vmm::build_direct_map(&mut self.pmm, self.pml4, memory_map);

        // Step 2: Instruct the PMM that it should now use higher-half (virtual)
        // addresses. Future allocations will return virtual pointers rather
        // than physical ones.
        self.pmm.is_higher_half = true;

        // Step 3: Map the GPU framebuffer into the kernel's virtual address
        // space so the graphics subsystem can still write to it after identity
        // maps are removed.
        vmm::map_framebuffer_higher_half(&mut self.pmm, self.pml4, framebuffer_info);

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
        vmm::reload_gdt_and_idt();

        // Step 6: Load the kernel's own minimal 3-entry GDT (null, code, data).
        // This was a placeholder step, before we implemented a better GDT. It
        // is no longer needed, and the proper GDT is initialized right after
        // the Memory Manager finishes its init steps. This is kept here for
        // completeness.

        // Step 7: Zero out the lower 256 PML4 entries (the identity-mapped
        // region) and flush the TLB. After this, virtual addresses below
        // KERNEL_PHYSICAL_MAP_BASE are unmapped.
        vmm::remove_identity_maps(self.pml4_phys);

        // Step 8: Walk the memory map again and return all loader/boot-services
        // pages to the PMM as free pages. Those pages were previously reserved
        // (we couldn't free them until we finished using the memory map and
        // page tables that lived in those regions).
        vmm::reclaim_boot_memory(&mut self.pmm, memory_map);
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
            vmm::map_page(&mut self.pmm, self.pml4, virtual_addr, physical_addr,
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
        unsafe { vmm::get_physical_addr(self.pml4, virtual_addr) }
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
        vmm::unmap_page(&mut self.pmm, self.pml4, virtual_addr)
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
