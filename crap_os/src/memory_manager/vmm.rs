//! Virtual Memory Manager
//!
//! On x86-64 systems with 4-level paging, the paging hierarchy is:
//!   PML4 -> PDPT -> PD -> PT -> physical page frame, where:
//!   * PML4 - Page Map Level 4, contains 512 PML4Es
//!   * PDPT - Page Directory Pointer Table, contains 512 PDPTEs 
//!   * PD   - Page Directory, contains 512 PDEs
//!   * PT   - Page Table, contains 512 64-bit PTEs
//!
//! Each of these tables is 4KB, containing 512 entries, with each entry being
//! 8 bytes.
//!
//! This structure maps 48-bit virtual addresses to physical addresses. A
//! 48-bit virtual address is split into 9-bit indices for each mapping level,
//! plus a 12-bit page offset: 9-bit PML4 index, 9-bit PDPT index, 9-bit PD
//! index, 9-bit PT index, and 12-bit offset. A single PML4 entry (PML4E) can
//! map 512 GB of memory, making the total addressable space per PML4 table 256
//! TB, which is more than enough for our purposes here. PML5 allows for larger
//! (57-bit) virtual address spaces.
//!
//! To facilitate address translation, the Memory Management Unit (MMU), located
//! in the CPU chip package, uses the CR3 register to locate the PML4. It then
//! traverses the levels to resolve the final physical address.

use crate::memory_manager::{PRESENT, WRITABLE, USER, PWT, PCD, NX};
use crate::memory_manager::{MemoryMapInfo, EfiMemoryDescriptor, EfiMemoryType};
use crate::memory_manager::{PhysicalMemoryManager, page_overlaps};
use crate::globals::{KERNEL_PHYSICAL_MAP_BASE, KERNEL_VIRTUAL_BASE, PAGE_SIZE,
    KERNEL_FRAMEBUFFER_VIRTUAL_BASE};

/// Looks up or creates a page table by its address and an index from previous
/// level.
/// 
/// # Arguments
///
/// * `pmm`                - Physical memory manager.
/// * `table`              - Address of the page table to look up or create
///                          if not exists.
/// * `index`              - Page table index from previous lookup level.
/// * `intermediate_flags` - Flags used by user-space page table walks to 
///                          propagate USER bit through all intermediate levels.
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
unsafe fn get_or_create_table(
    pmm: &mut PhysicalMemoryManager,
    table: *mut u64,
    index: u64,
    intermediate_flags: u64,
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

            // Use caller-supplied flags so user-space page table walks
            // propagate USER through all intermediate levels.
            *entry = new_table_phys | intermediate_flags;
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
/// * `is_user_page`  - If true, any newly-created intermediate tables (PDPT,
///     PD, PT) will be marked with PRESENT | WRITABLE | USER. This is required
///     for user-mode pages, as the CPU checks the USER flag at every level of
///     the walk, not just the final PTE. Existing intermediate tables are not
///     modified, so the caller must ensure they were originally created with
///     the correct flags. If set to false, intermediate tables are marked
///     PRESENT | WRITABLE (supervisor only), which is correct for all kernel
///     mappings.
/// 
/// # Safety
/// 
/// Dereferences raw pointers.
/// `pml4_addr` must point to a valid, mapped PML4 table. `physical_addr` must
/// be a real, PMM-owned physical page address. Caller must ensure the virtual
/// address is not already mapped to a different frame unless intentionally
/// remapping.
pub unsafe fn map_page(
    pmm: &mut PhysicalMemoryManager,
    pml4_addr: *mut u64,
    virtual_addr: u64,
    physical_addr: u64,
    flags: u64,
    is_user_page: bool,
) {
    let intermediate_flags = if is_user_page {
        PRESENT | WRITABLE | USER
    } else {
        PRESENT | WRITABLE
    };

    let pml4_index = (virtual_addr >> 39) & 0x1FF;
    let pdpt_index = (virtual_addr >> 30) & 0x1FF;
    let pd_index   = (virtual_addr >> 21) & 0x1FF;
    let pt_index   = (virtual_addr >> 12) & 0x1FF;

    let pdpt = unsafe {
        get_or_create_table(pmm, pml4_addr, pml4_index, intermediate_flags)
    };
    let pd   = unsafe {
        get_or_create_table(pmm, pdpt, pdpt_index, intermediate_flags)
    };
    let pt   = unsafe {
        get_or_create_table(pmm, pd, pd_index, intermediate_flags)
    };

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
pub unsafe fn get_physical_addr(pml4_addr: *mut u64, virtual_addr: u64
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
pub fn unmap_page(pmm: &mut PhysicalMemoryManager, pml4_addr: *mut u64,
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
pub(super) fn zero_out_page(address: u64) {
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
            unsafe {
                map_page(pmm, pml4, phys, phys, PRESENT | WRITABLE, false)
            };
        }
    }

    // Identity map + higher-half map: kernel image
    let mut addr = pmm.kernel_start;
    while addr < pmm.kernel_end {
        unsafe {
            map_page(pmm, pml4, addr, addr, PRESENT | WRITABLE, false);
            map_page(pmm, pml4, addr + KERNEL_VIRTUAL_BASE, addr,
                PRESENT | WRITABLE, false)
        };
        addr += PAGE_SIZE;
    }

    // Identity map + higher-half map: kernel stack with NX bit
    let stack_start = memory_map.stack_base_addr & !0xFFF;
    let stack_end = memory_map.stack_base_addr + memory_map.stack_size;
    let mut addr = stack_start;
    while addr < stack_end {
        unsafe {
            map_page(pmm, pml4, addr, addr, PRESENT | WRITABLE | NX, false);
            map_page(pmm, pml4, addr + KERNEL_VIRTUAL_BASE, addr,
            PRESENT | WRITABLE | NX, false);
        }
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
        unsafe { map_page(pmm, pml4, addr, addr, PRESENT | WRITABLE, false) };
        addr += PAGE_SIZE;
    }

    // Identity map: UEFI memory map buffer
    let map_start = memory_map.memory_map_addr & !0xFFF;
    let map_end = memory_map.memory_map_addr + memory_map.memory_map_size;
    let mut addr = map_start;
    while addr < map_end {
        unsafe { map_page(pmm, pml4, addr, addr, PRESENT | WRITABLE, false) };
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
pub fn build_direct_map(pmm: &mut PhysicalMemoryManager, pml4: *mut u64,
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
            unsafe {
                map_page(pmm, pml4, phys_map_base + phys, phys, flags, false)
            };
        }
    }

    // Map APIC MMIO regions through the direct physical map.
    //
    // These are not in the UEFI memory map as conventional memory, so the
    // loop above skips them. We map them explicitly here so the APIC driver
    // can reach them at KERNEL_PHYSICAL_MAP_BASE + phys_addr.
    let apic_regions: [(u64, u64); 2] = [
        (0xFEE00000, 0x1000),  // Local APIC
        (0xFEC00000, 0x1000),  // I/O APIC
    ];

    for (phys_base, size) in apic_regions {
        let mut phys = phys_base;
        while phys < phys_base + size {
            unsafe {
                map_page(pmm, pml4, KERNEL_PHYSICAL_MAP_BASE + phys, phys,
                    PRESENT | WRITABLE | PWT | PCD, false);
            }
            phys += PAGE_SIZE;
        }
    }

    // Map HPET MMIO region through the direct physical map. It is also not in
    // the UEFI memory map as conventional memory, so we need to map it
    // separately. One entire page is enough for what we need out of it.
    let hpet_mmio_addr = 0xFED00000;
    unsafe {
        map_page(pmm, pml4, KERNEL_PHYSICAL_MAP_BASE + hpet_mmio_addr,
            hpet_mmio_addr, PRESENT | WRITABLE | PWT | PCD, false)
    };
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
pub fn map_framebuffer_higher_half(
    pmm: &mut PhysicalMemoryManager,
    pml4: *mut u64,
    framebuffer_info: &crate::FramebufferInfo,
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
        unsafe { map_page(pmm, pml4, virt, phys, PRESENT | WRITABLE, false) };
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
pub fn reload_gdt_and_idt() {
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
pub fn remove_identity_maps(pml4_phys: u64) {
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
pub fn reclaim_boot_memory(pmm: &mut PhysicalMemoryManager,
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
