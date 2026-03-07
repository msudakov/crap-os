/**
 * @file  craploader.c
 * @brief CrapLoader: UEFI Bootloader for CrapOS
 *
 * This file implements the UEFI bootloader responsible for loading the kernel
 * binary from disk, setting up initial page tables, acquiring hardware info
 * (framebuffer, memory map), and jumping into the higher-half kernel.
 *
 * =============================================================================
 * HIGH-LEVEL BOOT SEQUENCE
 * =============================================================================
 *
 *  1. Initialise the GNU-EFI library and serial port (COM1) for debug output.
 *  2. Acquire the GOP (Graphics Output Protocol) to locate the linear
 *     framebuffer and record its address, resolution, and pixel format.
 *  3. Locate UEFI protocols needed to open files from EFI System Partition:
 *       - LoadedImageProtocol: tells us which device we booted from;
 *       - SimpleFileSystem:    gives us a file handle to the ESP root;
 *  4. Open "kernel.bin" from the ESP root, query its size, and load it
 *     into physical memory at 1 MB (0x100000).
 *  5. Allocate the kernel stack (page-aligned) anywhere in RAM.
 *  6. Snapshot the UEFI memory map (needed to build page tables).
 *  7. Build the initial page table hierarchy (PML4) in a pre-allocated pool:
 *       a. Identity-map the PT node pool itself;
 *       b. Identity-map all conventional/boot/loader memory;
 *       c. Identity-map + higher-half-map the kernel image;
 *       d. Identity-map + higher-half-map the kernel stack (NX);
 *       e. Identity-map the framebuffer;
 *       f. Identity-map the UEFI memory map buffer;
 *       g. Map the first 4 GB of physical RAM at KERNEL_PHYS_MAP_BASE
 *          using 2 MB huge pages (the "direct physical map");
 *  8. Obtain the final memory map and call ExitBootServices; after this
 *     point, UEFI boot services are gone, and only serial output is available.
 *  9. Install a temporary minimal GDT (the UEFI firmware GDT may be
 *     reclaimed on some implementations after ExitBootServices).
 * 10. Enable the NX (No-Execute) bit in the IA32_EFER MSR.
 * 11. Load CR3 with our new PML4, putting the paging under our control.
 * 12. Switch RSP to the higher-half virtual stack address.
 * 13. Jump to the kernel entry point at its higher-half virtual address.
 *
 * =============================================================================
 * VIRTUAL MEMORY LAYOUT (must exactly match kernel's globals)
 * =============================================================================
 *
 *  0x0000000000000000 – 0x00007FFFFFFFFFFF  User space
 *  0xFFFF800000000000         Direct physical map base (first 4 GB mapped here)
 *  0xFFFF900000000000         Framebuffer virtual base
 *  0xFFFFFFFF80000000         Kernel image virtual base
 *
 * =============================================================================
 */

#include <efi.h>
#include <efilib.h>


#define COM1_PORT 0x3F8  // COM1 port base address
#define KERNEL_STACK_PAGES 32  // 32 * 4096 = 128 KB; lower it to 64 later on
#define KERNEL_STACK_SIZE  (KERNEL_STACK_PAGES * 0x1000)

// Virtual memory layout, must match kernel globals exactly
#define KERNEL_VIRTUAL_BASE          0xFFFFFFFF80000000ULL
#define KERNEL_PHYSICAL_BASE         0x100000ULL
#define KERNEL_VIRTUAL_OFFSET        (KERNEL_VIRTUAL_BASE- KERNEL_PHYSICAL_BASE)
#define KERNEL_PHYS_MAP_BASE         0xFFFF800000000000ULL
#define KERNEL_FRAMEBUFFER_VIRT_BASE 0xFFFF900000000000ULL

// Page table flags
#define PT_PRESENT  (1ULL << 0)
#define PT_WRITABLE (1ULL << 1)
#define PT_HUGE     (1ULL << 7)  // For 2MB pages in PD
#define PT_NX       (1ULL << 63)

// Debug message levels
enum DebugLevel
{
    DEBUG = 1,
    INFO = 2,
    WARNING = 3,
    ERROR = 4,
    CRITICAL = 5
};

// Framebuffer information structure to pass to kernel
typedef struct {
    UINT64 framebuffer_addr;
    UINT32 framebuffer_width;
    UINT32 framebuffer_height;
    UINT32 framebuffer_pitch;
    UINT32 framebuffer_bpp;
} FramebufferInfo;

// Memory map information structure to pass to kernel
typedef struct {
    UINT64 memory_map_addr;
    UINT64 memory_map_size;
    UINT64 descriptor_size;
    UINT32 descriptor_ver;
    UINT64 kernel_load_addr;
    UINT64 kernel_image_size;
    UINT64 stack_base_addr;
    UINT64 stack_size;
} MemoryMapInfo;

// Boot information structure to pass to kernel
typedef struct {
    UINT64 magic;
    FramebufferInfo* framebuffer_info;
    MemoryMapInfo* memory_map_info;
} BootInfo;

// Page pool structure for page mapping
typedef struct {
    UINT64  base;  // Physical base address of the pool
    UINTN   used;  // Number of pages used so far
    UINTN   cap;   // Total number of pages in the pool
} PagePool;

const enum DebugLevel DEBUG_LEVEL = INFO;  // Set debug message level

// Kernel entry point that receives boot info
typedef void (*kernel_entry)(BootInfo *);

/**
  Reads one byte of data from a specified I/O port address.

  @param[in]  port   Serial port base address.
  
  @return The UINT8 value received from the specified serial port.
**/
static inline unsigned char inb(unsigned short port) {
    unsigned char ret;
    __asm__ volatile ( "inb %1, %0" : "=a"(ret) : "Nd"(port) );
    return ret;
}

/**
  Writes one byte of data to a specified I/O port address.

  @param[in]  port    Serial port base address.
  @param[in]  value   Byte value to write to serial port.
**/
static inline void outb(unsigned short port, unsigned char value) {
    __asm__ volatile ( "outb %0, %1" : : "a"(value), "Nd"(port) );
}

/**
  Initializes serial port for debugging purposes.
**/
void init_serial() {
    outb(COM1_PORT + 1, 0x00);    // Disable interrupts on COM1
    outb(COM1_PORT + 3, 0x80);    // Enable DLAB (set baud rate divisor)
    outb(COM1_PORT + 0, 0x03);    // Set divisor 3 (low byte) 38400 baud
    outb(COM1_PORT + 1, 0x00);    //               (high byte)
    outb(COM1_PORT + 3, 0x03);    // 8 bits, no parity, one stop bit
    outb(COM1_PORT + 2, 0xC7);    // Enable FIFO, clear them (14-byte threshold)
    outb(COM1_PORT + 4, 0x0B);    // IRQs enabled, RTS/DSR set
}

/**
  Writes a byte to COM1 serial port.

  @param[in]  c   Byte to write.
**/
void serial_write_byte(char c) {
    // Wait for transmit buffer to be empty
    while ((inb(COM1_PORT + 5) & 0x20) == 0);

    // Send the byte
    outb(COM1_PORT, c);
}

/**
  Counts the number of characters in a string (its length), until it encounters
    a null terminator.

  @param[in]  str   Message string to write.

  @return The int value of the message length.
**/
int get_strlen(const char* str) {
    int len = 0;
    while (*str++ != '\0') {
        len++;
    }

    return len;
}

/**
  Writes a given message to COM1 serial port.

  @param[in]  str   Message string to write.
**/
void serial_write(const char* str) {
    int strLen = get_strlen(str);

    for (int i = 0; i < strLen; i++) {
        serial_write_byte(str[i]);
    }
}

/**
  Prints a debug message to console if its debug level is equal to or greater
    than the global debug message level.

  @param[in]  debug_level   Specified debug level of the given message.
  @param[in]  message       Debug message to print.
**/
void print_debug(enum DebugLevel debug_level, CHAR16* message) {
    if (debug_level < DEBUG_LEVEL || message == NULL) {
        return;
    }

    Print(message);
}

/**
  Prints an unsigned 64-bit integer value (e.g., memory address) as hex string.

  @param[in]  value   Value to print as hex.
**/
void serial_write_hex(UINT64 value) {
    char buf[17];
    buf[16] = '\0';

    for (int i = 15; i >= 0; i--) {
        int nibble = value & 0xF;
        buf[i] = nibble < 10 ? '0' + nibble : 'a' + nibble - 10;
        value >>= 4;
    }
    serial_write(buf);
}

/**
  Computes the index into a given page table level for a virtual address.

  PML4 index: bits 47:39
  PDPT index: bits 38:30
  PD   index: bits 29:21
  PT   index: bits 20:12

  @param[in]  virt    Virtual address to index.
  @param[in]  level   Page table level (4=PML4, 3=PDPT, 2=PD, 1=PT).

  @return The 9-bit index for this level.
**/
static UINT64 pt_index(UINT64 virt, int level) {
    return (virt >> (12 + 9 * (level - 1))) & 0x1FF;
}

/**
 * Allocates one 4 KB page from the page-table node pool.
 *
 * Uses a bump-pointer strategy: simply returns the next unused page in
 * the pre-allocated contiguous pool block, then increments the counter.
 * The allocated page is zeroed out so that all page-table entries default to
 * "not present" (bit 0 = 0).
 *
 * @param[in]  pool   Pointer to the PagePool to allocate from.
 * 
 * @return Physical address of the newly allocated, zeroed page, or 0 if the
 *  pool is exhausted.
 */
static UINT64 pool_alloc(PagePool *pool) {
    if (pool->used >= pool->cap) {
        return 0;  // Pool exhausted
    }

    // Calculate the physical address of the next free page
    UINT64 page = pool->base + pool->used * 0x1000;
    pool->used++;  // Advance the bump pointer

    // Zero all 512 8-byte entries so every PTE starts as "not present"
    UINT64 *p = (UINT64 *)page;
    for (int i = 0; i < 512; i++) {
        p[i] = 0;
    }

    return page;
}

/**
 * Maps a single 4 KB physical page to a virtual address in the page table
 * rooted at `pml4`, using the pool allocator for any new table nodes needed.
 *
 * Walks the four levels of the page table hierarchy (PML4 > PDPT > PD > PT),
 * allocating a new zeroed page for each level that has no entry yet, and
 * finally writes the leaf PTE with the physical address and flags.
 *
 * @param[in]  pool           The pool from which to allocate new PT node pages.
 * @param[in]  pml4           Physical address of the root PML4 page table.
 * @param[in]  virtual_addr   The virtual address to create a mapping for.
 * @param[in]  physical_addr  The physical address of the page to map.
 * @param[in]  flags          Page-table entry flags.
 */
static void map_page_pool(PagePool *pool, UINT64 pml4, UINT64 virtual_addr,
    UINT64 physical_addr, UINT64 flags)
{
    // ---------- Level 4: PML4 ----------

    // Compute a pointer to the PML4 entry for this virtual address
    UINT64 *pml4e = (UINT64 *)(pml4 + pt_index(virtual_addr, 4) * 8);

    // If this PML4 entry is not yet present, allocate a new PDPT page
    if (!(*pml4e & PT_PRESENT)) {
        UINT64 pdpt = pool_alloc(pool);
        *pml4e = pdpt | PT_PRESENT | PT_WRITABLE;
    }

    // Extract the physical address of the PDPT from the PML4 entry
    UINT64 pdpt_addr = *pml4e & ~0xFFFULL;

    // ---------- Level 3: PDPT ----------

    // Compute a pointer to the relevant PDPT entry
    UINT64 *pdpte = (UINT64 *)(pdpt_addr + pt_index(virtual_addr, 3) * 8);

    // Allocate a new PD if no PD exists for this virtual address range
    if (!(*pdpte & PT_PRESENT)) {
        UINT64 pd = pool_alloc(pool);
        *pdpte = pd | PT_PRESENT | PT_WRITABLE;
    }

    // Strip flags to get the PD physical address
    UINT64 pd_addr = *pdpte & ~0xFFFULL;

    // ---------- Level 2: PD ----------

    // Compute a pointer to the relevant PD entry
    UINT64 *pde = (UINT64 *)(pd_addr + pt_index(virtual_addr, 2) * 8);

    // Allocate a new PT if no PT exists for this 2 MB virtual range
    if (!(*pde & PT_PRESENT)) {
        UINT64 pt = pool_alloc(pool);
        *pde = pt | PT_PRESENT | PT_WRITABLE;
    }

    // Strip flags to get the PT physical address
    UINT64 pt_addr = *pde & ~0xFFFULL;

    // ---------- Level 1: PT (leaf) ----------

    // Pointer to the 4 KB leaf page table entry
    UINT64 *pte = (UINT64 *)(pt_addr + pt_index(virtual_addr, 1) * 8);

    // Write the leaf entry: physical address OR'd with the caller-supplied
    // flags. This is page-aligned, with low 12 bits cleared.
    *pte = (physical_addr & ~0xFFFULL) | flags;
}

/**
 * Builds the initial kernel page tables using a pre-allocated pool
 * of 4 KB pages for all intermediate and leaf page-table nodes.
 *
 * This function is called before ExitBootServices because it needs
 * `AllocatePages`. The tables it produces are loaded into CR3 just before
 * jumping to the kernel.
 *
 * See the numbered steps inside for a detailed breakdown of what is mapped
 * and why.
 *
 * @param[in] BS               UEFI Boot Services table pointer.
 * @param[in] MemoryMap        Pointer to the EFI memory map array.
 * @param[in] MemoryMapSize    Total byte size of the memory map array.
 * @param[in] DescriptorSize   Byte size of one EFI_MEMORY_DESCRIPTOR entry.
 * @param[in] KernelBuffer     Physical address at which kernel.bin is loaded.
 * @param[in] KernelSize       Size of the kernel image in bytes.
 * @param[in] KernelStackBase  Physical base address of kernel stack allocation.
 * @param[in] FramebufferBase  Physical base address of the GPU framebuffer.
 * @param[in] FramebufferSize  Size of the framebuffer in bytes.
 * @param[in] PoolPages        Number of pool pages to use.
 *
 * @return Physical address of the PML4 root page table, or 0 on failure.
 */
static UINT64 build_page_tables(
    EFI_BOOT_SERVICES       *BS,
    EFI_MEMORY_DESCRIPTOR   *MemoryMap,
    UINTN                    MemoryMapSize,
    UINTN                    DescriptorSize,
    EFI_PHYSICAL_ADDRESS     KernelBuffer,
    UINTN                    KernelSize,
    EFI_PHYSICAL_ADDRESS     KernelStackBase,
    UINT64                   FramebufferBase,
    UINT64                   FramebufferSize,
    UINTN                    PoolPages)
{
    // -------------------------------------------------------------------------
    // Step 1: Allocate the page-table node pool upfront.
    //
    // We pre-allocate one large contiguous block and carve individual
    // 4 KB pages out of it with the bump allocator (pool_alloc).
    // -------------------------------------------------------------------------
    EFI_PHYSICAL_ADDRESS pool_base = 0;
    EFI_STATUS status = uefi_call_wrapper(BS->AllocatePages, 4,
        AllocateAnyPages, EfiLoaderData, PoolPages, &pool_base);
    if (EFI_ERROR(status)) {
        serial_write("[-] Failed to allocate PT pool\n\r");
        return 0;
    }

    // Zero the entire pool so all page-table entries start as "not present"
    UINT64 *p = (UINT64 *)pool_base;
    for (UINTN i = 0; i < PoolPages * 512; i++) {
        p[i] = 0;
    }

    // Initialise the bump allocator descriptor
    PagePool pool = {
        .base = (UINT64)pool_base,  // Pool starts at the allocated address
        .used = 0,                  // No pages consumed yet
        .cap  = PoolPages,         // Capacity = 700 pages
    };

    // -------------------------------------------------------------------------
    // Step 2: Allocate the PML4 root page from the pool.
    //
    // The PML4 is the top-level page table (level 4). Its physical address is
    // loaded into CR3 to activate our page tables.
    // -------------------------------------------------------------------------
    UINT64 pml4 = pool_alloc(&pool);
    if (!pml4) {
        serial_write("[-] Pool exhausted allocating PML4\n\r");
        return 0;
    }

    // -------------------------------------------------------------------------
    // Step 3: Identity-map the entire PT node pool, so that
    // virtual == physical address.
    //
    // We must do this before using `pool_alloc` for anything else. The reason
    // is that when CR3 is loaded, the CPU starts using our new page tables for
    // all memory accesses, including its own walks through the page table
    // tree. If the pool pages themselves are not mapped, the very act of
    // walking the page tables causes a page fault because the CPU cannot read
    // the table entries. And at this early stage, that would lead to a triple
    // fault.
    //
    // It is safe to call `map_page_pool` here (which itself calls `pool_alloc`)
    // because the pool's physical memory already exists and is zeroed. Also,
    // `map_page_pool` only allocates new nodes for entries not yet present,
    // and the base address of each pool page is a known flat offset from
    // pool_base, so the allocator correctly handles recursive use.
    // -------------------------------------------------------------------------
    UINT64 pool_end = pool.base + PoolPages * 0x1000;
    for (UINT64 addr = pool.base; addr < pool_end; addr += 0x1000) {
        map_page_pool(&pool, pml4, addr, addr, PT_PRESENT | PT_WRITABLE);
    }

    // -------------------------------------------------------------------------
    // Step 4: Identity-map all UEFI conventional and boot-services memory.
    //
    // The CPU must be able to execute code and access data in the regions
    // UEFI is currently using. Until ExitBootServices is called and we
    // stop using UEFI, these pages must remain accessible at their current
    // (physical == virtual) addresses.
    //
    // We walk every descriptor in the memory map and map the regions of types
    // EfiConventionalMemory, EfiLoaderCode/Data, and EfiBootServicesCode/Data.
    // -------------------------------------------------------------------------
    UINTN num_descs = MemoryMapSize / DescriptorSize;
    EFI_MEMORY_DESCRIPTOR *desc = MemoryMap;  // Current descriptor pointer

    for (UINTN i = 0; i < num_descs; i++) {
        int should_map = (
            desc->Type == EfiConventionalMemory  ||  // Free RAM
            desc->Type == EfiLoaderCode          ||  // Bootloader code
            desc->Type == EfiLoaderData          ||  // Bootloader data
            desc->Type == EfiBootServicesCode    ||  // UEFI BS code
            desc->Type == EfiBootServicesData        // UEFI BS data
        );
        if (should_map) {
            // Map every 4 KB page in this descriptor's physical range
            for (UINTN pg = 0; pg < desc->NumberOfPages; pg++) {
                UINT64 phys = desc->PhysicalStart + pg * 0x1000;
                map_page_pool(&pool, pml4, phys, phys,
                    PT_PRESENT | PT_WRITABLE);
            }
        }

        // Advance the pointer to the next descriptor
        desc = (EFI_MEMORY_DESCRIPTOR *)((UINT8 *)desc + DescriptorSize);
    }

    // -------------------------------------------------------------------------
    // Step 5: Map the kernel image with both identity map and higher-half map.
    //
    // We need two mappings for the kernel image:
    //
    //   a) Identity map (physical == virtual):
    //      Required so that execution can continue seamlessly right after
    //      CR3 is loaded, before we jump to the higher-half entry point.
    //      The instruction pointer (RIP) still points to a low address at
    //      that moment, so those addresses must remain valid.
    //
    //   b) Higher-half map (virtual = physical + KERNEL_VIRTUAL_BASE):
    //      The kernel is compiled with all symbols linked at
    //      KERNEL_VIRTUAL_BASE + offset. The kernel entry point and all
    //      kernel functions live at higher-half virtual addresses.
    // -------------------------------------------------------------------------
    UINT64 kernel_phys = (UINT64)KernelBuffer;
    UINT64 kernel_end  = kernel_phys + KernelSize;
    for (UINT64 addr = kernel_phys; addr < kernel_end; addr += 0x1000) {
        map_page_pool(&pool, pml4, addr, addr,
            PT_PRESENT | PT_WRITABLE);  // Identity map
        map_page_pool(&pool, pml4, addr + KERNEL_VIRTUAL_BASE, addr,
            PT_PRESENT | PT_WRITABLE);  // Higher-half map
    }

    // -------------------------------------------------------------------------
    // Step 6: Map the kernel stack with both identity map and higher-half map,
    // using the NX flag.
    //
    // Same dual-mapping rationale as the kernel image. The stack is marked
    // NX (No-Execute) because code should never be executed from the stack;
    // this is a standard security hardening measure.
    //
    // We align the stack start down to a page boundary with `& ~0xFFF`
    // to ensure we map the entire first partial page just in case that
    // KernelStackBase is not page-aligned, which it should be.
    // -------------------------------------------------------------------------
    UINT64 stack_start = KernelStackBase & ~0xFFFULL;  // Page-align down
    UINT64 stack_end   = KernelStackBase + KERNEL_STACK_SIZE;
    for (UINT64 addr = stack_start; addr < stack_end; addr += 0x1000) {
        map_page_pool(&pool, pml4, addr, addr,
            PT_PRESENT | PT_WRITABLE | PT_NX);  // Identity map with NX
        map_page_pool(&pool, pml4, addr + KERNEL_VIRTUAL_BASE, addr,
            PT_PRESENT | PT_WRITABLE | PT_NX);  // Higher-half map with NX
    }

    // -------------------------------------------------------------------------
    // Step 7: Identity-map the framebuffer.
    //
    // Only an identity map here since the kernel's memory manager will set up
    // the proper higher-half framebuffer mapping (at 
    // KERNEL_FRAMEBUFFER_VIRT_BASE) during its own initialisation. We need the
    // identity map so that the kernel can access the framebuffer before its
    // VMM is fully running. The framebuffer start is page-aligned downward for
    // safety.
    // -------------------------------------------------------------------------
    UINT64 fb_start = FramebufferBase & ~0xFFFULL;  // Page-align down
    UINT64 fb_end   = FramebufferBase + FramebufferSize;
    for (UINT64 addr = fb_start; addr < fb_end; addr += 0x1000) {
        map_page_pool(&pool, pml4, addr, addr, PT_PRESENT | PT_WRITABLE);
    }

    // -------------------------------------------------------------------------
    // Step 8: Identity-map the UEFI memory map buffer.
    //
    // The MemoryMap pointer passed into this function points to a buffer
    // in UEFI-managed memory. The kernel reads this buffer during its own
    // memory manager initialisation to discover usable physical memory.
    // It must remain accessible after CR3 is switched to our new tables. Its
    // start is page-aligned downward to catch any partial leading page.
    // -------------------------------------------------------------------------
    UINT64 map_start = (UINT64)MemoryMap & ~0xFFFULL;  // Page-align down
    UINT64 map_end   = (UINT64)MemoryMap + MemoryMapSize;
    for (UINT64 addr = map_start; addr < map_end; addr += 0x1000) {
        map_page_pool(&pool, pml4, addr, addr, PT_PRESENT | PT_WRITABLE);
    }

    // -------------------------------------------------------------------------
    // Step 9: Direct physical map for the first 4 GB at KERNEL_PHYS_MAP_BASE,
    // using 2 MB huge pages.
    //
    // This creates a "window" in the higher half where every physical address
    // P can be accessed at KERNEL_PHYS_MAP_BASE + P. The kernel uses this
    // to access arbitrary physical memory (page tables, hardware, ACPI tables)
    // by virtual address after identity maps are removed.
    //
    // Physical range covered: 0x0 to 0xFFFFFFFF (4 GB).
    // -------------------------------------------------------------------------
    
    // Get a pointer to the PML4 entry for KERNEL_PHYS_MAP_BASE
    UINT64 *pml4e = (UINT64 *)(pml4 + pt_index(KERNEL_PHYS_MAP_BASE, 4) * 8);

    // Allocate the single PDPT page for this region
    UINT64 pdpt_page = pool_alloc(&pool);
    if (!pdpt_page) {
        serial_write("[-] Pool exhausted allocating phys map PDPT\n\r");
        return 0;
    }

    // Store the PDPT address in the PML4 entry
    *pml4e = pdpt_page | PT_PRESENT | PT_WRITABLE;

    // Iterate over 4 PDPT entries (one per GB)
    for (UINT64 pdpt_i = 0; pdpt_i < 4; pdpt_i++) {
        // Allocate one PD page per GB of physical address space
        UINT64 pd_page = pool_alloc(&pool);
        if (!pd_page) {
            serial_write("[-] Pool exhausted allocating phys map PD\n\r");
            return 0;
        }

        // Write the PDPT entry: PD address
        UINT64 *pdpte = (UINT64 *)(pdpt_page + pdpt_i * 8);
        *pdpte = pd_page | PT_PRESENT | PT_WRITABLE;

        // Fill all 512 PD entries with 2 MB huge page mappings
        for (UINT64 pd_i = 0; pd_i < 512; pd_i++) {
            // Physical address for this 2 MB page
            UINT64 phys = (pdpt_i * 512 + pd_i) * 0x200000ULL;

            // PD entry pointer: pd_page base + pd_i * 8 bytes per entry
            UINT64 *pde = (UINT64 *)(pd_page + pd_i * 8);

            // Set the PD entry as a 2 MB huge page mapping
            *pde = phys | PT_PRESENT | PT_WRITABLE | PT_HUGE;
        }
    }

    // Return the physical address of the PML4 root
    return pml4;
}

// =============================================================================
// UEFI entry point
// =============================================================================

/**
 * UEFI application entry point; the firmware calls this function.
 *
 * @param[in] ImageHandle  UEFI handle identifying this loaded image.
 * @param[in] SystemTable  Pointer to the UEFI System Table, which provides
 *                         access to all UEFI services (boot services, runtime
 *                         services, protocols, console I/O, etc.).
 *
 * @return EFI_SUCCESS on clean exit (unreachable in normal flow), or an EFI
 *  error code if a fatal error occurs before ExitBootServices.
 */
EFI_STATUS
EFIAPI
efi_main(EFI_HANDLE ImageHandle, EFI_SYSTEM_TABLE *SystemTable) {
    EFI_STATUS Status;
    EFI_INPUT_KEY Key;
    EFI_FILE_PROTOCOL *Root;
    EFI_FILE_PROTOCOL *KernelFile;
    EFI_LOADED_IMAGE_PROTOCOL *LoadedImage;
    EFI_SIMPLE_FILE_SYSTEM_PROTOCOL *FileSystem;
    EFI_GRAPHICS_OUTPUT_PROTOCOL *Gop;
    UINTN KernelSize;
    EFI_PHYSICAL_ADDRESS KernelBuffer;
    EFI_PHYSICAL_ADDRESS KernelStackBase = 0;
    EFI_PHYSICAL_ADDRESS KernelStackTop;
    UINTN MapKey;
    UINTN DescriptorSize;
    UINT32 DescriptorVersion;
    UINTN MemoryMapSize = 0;
    EFI_MEMORY_DESCRIPTOR *MemoryMap = NULL;
    FramebufferInfo *FramebufferInfoStruct;
    MemoryMapInfo *MemoryMapInfoStruct;
    BootInfo *BootInfoStruct;
    UINT64 Pml4 = 0;
    
    InitializeLib(ImageHandle, SystemTable);  // Initialize the GNU-EFI library
    init_serial();  // Initialize serial port for debugging

    print_debug(INFO, L"[+] Hello from CrapLoader!\n\r");
    //uefi_call_wrapper(SystemTable->BootServices->Stall, 1, 1000000);

    // Allocate boot info structures
    FramebufferInfoStruct = AllocatePool(sizeof(FramebufferInfo));
    MemoryMapInfoStruct = AllocatePool(sizeof(MemoryMapInfo));
    BootInfoStruct = AllocatePool(sizeof(BootInfo));
    BootInfoStruct->magic = 0xDEADBEEFB007CAFE;

    print_debug(DEBUG, L"[*] Getting Graphics Output Protocol...\n\r");
    Status = uefi_call_wrapper(SystemTable->BootServices->LocateProtocol, 3,
        &gEfiGraphicsOutputProtocolGuid, NULL, (void**)&Gop);
    
    if (EFI_ERROR(Status)) {
        Print(L"[-] WARNING: Could not get GOP (status: %r)\n\r", Status);
        print_debug(WARNING, L"[*] Will try VGA text mode fallback\n\r");
        FramebufferInfoStruct->framebuffer_addr = 0xB8000;  // VGA text buffer
        FramebufferInfoStruct->framebuffer_width = 80;
        FramebufferInfoStruct->framebuffer_height = 25;
        FramebufferInfoStruct->framebuffer_pitch = 160;
        FramebufferInfoStruct->framebuffer_bpp = 16;
    }
    else {
        // Get framebuffer info
        FramebufferInfoStruct->framebuffer_addr = Gop->Mode->FrameBufferBase;

        FramebufferInfoStruct->framebuffer_width =
            Gop->Mode->Info->HorizontalResolution;
        FramebufferInfoStruct->framebuffer_height =
            Gop->Mode->Info->VerticalResolution;
        FramebufferInfoStruct->framebuffer_pitch =
            Gop->Mode->Info->PixelsPerScanLine * 4;

        FramebufferInfoStruct->framebuffer_bpp = 32;
        print_debug(DEBUG, L"[+] Located Graphics Output Protocol\n\r");
    }
    BootInfoStruct->framebuffer_info = FramebufferInfoStruct;
    //uefi_call_wrapper(SystemTable->BootServices->Stall, 1, 1000000);

    Status = uefi_call_wrapper(SystemTable->BootServices->HandleProtocol, 3,
        ImageHandle, &gEfiLoadedImageProtocolGuid, (void**)&LoadedImage);
    if (EFI_ERROR(Status)) {
        print_debug(ERROR, L"[-] Failed to get LoadedImage protocol\n\r");
        Print(L"[!FATAL!] Press any key to exit...\n\r");
        while (uefi_call_wrapper(SystemTable->ConIn->ReadKeyStroke, 2,
            SystemTable->ConIn, &Key) != EFI_SUCCESS);
        return Status;
    }
    print_debug(DEBUG, L"[+] Located LoadedImage protocol\n\r");

    Status = uefi_call_wrapper(SystemTable->BootServices->HandleProtocol, 3,
        LoadedImage->DeviceHandle, &gEfiSimpleFileSystemProtocolGuid,
        (void**)&FileSystem);
    if (EFI_ERROR(Status)) {
        print_debug(ERROR, L"[-] Failed to get FileSystem protocol\n\r");
        Print(L"[!FATAL!] Press any key to exit...\n\r");
        while (uefi_call_wrapper(SystemTable->ConIn->ReadKeyStroke, 2,
            SystemTable->ConIn, &Key) != EFI_SUCCESS);
        return Status;
    }
    print_debug(DEBUG, L"[+] Located FileSystem protocol\n\r");

    Status = uefi_call_wrapper(FileSystem->OpenVolume, 2, FileSystem, &Root);
    if (EFI_ERROR(Status)) {
        print_debug(ERROR, L"[-] Failed to open root volume\n\r");
        Print(L"[!FATAL!] Press any key to exit...\n\r");
        while (uefi_call_wrapper(SystemTable->ConIn->ReadKeyStroke, 2,
            SystemTable->ConIn, &Key) != EFI_SUCCESS);
        return Status;
    }
    print_debug(DEBUG, L"[+] Opened the root directory\n\r");

    Status = uefi_call_wrapper(Root->Open, 5, Root, &KernelFile, L"kernel.bin",
        EFI_FILE_MODE_READ, 0);
    if (EFI_ERROR(Status)) {
        print_debug(ERROR, L"[-] Failed to open kernel.bin\n\r");
        Print(L"[!FATAL!] Press any key to exit...\n\r");
        while (uefi_call_wrapper(SystemTable->ConIn->ReadKeyStroke, 2,
            SystemTable->ConIn, &Key) != EFI_SUCCESS);
        return Status;
    }
    print_debug(DEBUG, L"[+] Opened the kernel file\n\r");
    
    // Get the kernel file size
    EFI_FILE_INFO *FileInfo;
    UINTN FileInfoSize = SIZE_OF_EFI_FILE_INFO + 200;
    FileInfo = AllocatePool(FileInfoSize);
    Status = uefi_call_wrapper(KernelFile->GetInfo, 4, KernelFile,
        &gEfiFileInfoGuid, &FileInfoSize, FileInfo);
    if (EFI_ERROR(Status)) {
        print_debug(ERROR, L"[-] Failed to get kernel file info\n\r");
        Print(L"[!FATAL!] Press any key to exit...\n\r");
        while (uefi_call_wrapper(SystemTable->ConIn->ReadKeyStroke, 2,
            SystemTable->ConIn, &Key) != EFI_SUCCESS);
        return Status;
    }
    KernelSize = FileInfo->FileSize;
    FreePool(FileInfo);
    Print(L"[+] Kernel file size: %d bytes\n\r", KernelSize);
    //uefi_call_wrapper(SystemTable->BootServices->Stall, 1, 1000000);
    
    // Allocate memory for the kernel
    KernelBuffer = 0x100000;  // Load at 1MB
    Status = uefi_call_wrapper(SystemTable->BootServices->AllocatePages, 4,
        AllocateAddress, EfiLoaderCode, (KernelSize + 0xFFF) / 0x1000,
        &KernelBuffer);
    if (EFI_ERROR(Status)) {
        print_debug(ERROR, L"[-] Failed to allocate memory for kernel\n\r");
        Print(L"[!FATAL!] Press any key to exit...\n\r");
        while (uefi_call_wrapper(SystemTable->ConIn->ReadKeyStroke, 2,
            SystemTable->ConIn, &Key) != EFI_SUCCESS);
        return Status;
    }
    print_debug(DEBUG, L"[+] Allocated memory for kernel\n\r");
    
    // Read the kernel into memory
    Status = uefi_call_wrapper(KernelFile->Read, 3, KernelFile, &KernelSize,
        (void*)KernelBuffer);
    if (EFI_ERROR(Status)) {
        print_debug(ERROR, L"[-] Failed to read kernel\n\r");
        Print(L"[!FATAL!] Press any key to exit...\n\r");
        while (uefi_call_wrapper(SystemTable->ConIn->ReadKeyStroke, 2,
            SystemTable->ConIn, &Key) != EFI_SUCCESS);
        return Status;
    }
    print_debug(DEBUG, L"[+] Successfully read kernel into memory\n\r");
    
    // Close the kernel file
    uefi_call_wrapper(KernelFile->Close, 1, KernelFile);
    uefi_call_wrapper(Root->Close, 1, Root);

    Print(L"[+] Kernel loaded at 0x%x\n\r", KernelBuffer);
    //uefi_call_wrapper(SystemTable->BootServices->Stall, 1, 1000000);

    // Allocate initial kernel stack
    Status = uefi_call_wrapper(SystemTable->BootServices->AllocatePages, 4,
        AllocateAnyPages, EfiLoaderData, KERNEL_STACK_PAGES, &KernelStackBase);
    if (EFI_ERROR(Status)) {
        print_debug(ERROR, L"[-] Failed to allocate kernel stack\n\r");
        Print(L"[!FATAL!] Press any key to exit...\n\r");
        while (uefi_call_wrapper(SystemTable->ConIn->ReadKeyStroke, 2,
            SystemTable->ConIn, &Key) != EFI_SUCCESS);
        return Status;
    }
    
    // Zero out allocated mmory just in case
    SetMem((void*)KernelStackBase, KERNEL_STACK_SIZE, 0);

    // Stack grows downward, so RSP should start at the top of the allocation.
    // We need to subtract 8 bytes to keep the RSP 16-byte aligned on entry.
    // This is because the x86-64 ABI requires RSP % 16 == 8 just before a call
    // instruction, but since we're jumping directly rather than using CALL,
    // we want RSP % 16 == 0 at _start).
    KernelStackTop = KernelStackBase + KERNEL_STACK_SIZE - 8;

    // Record stack details for the kernel's memory manager
    MemoryMapInfoStruct->stack_base_addr = KernelStackBase;
    MemoryMapInfoStruct->stack_size = KERNEL_STACK_SIZE;

    print_debug(DEBUG, L"[+] Allocated kernel stack\n\r");
    Print(L"[+] Stack: base=0x%lx  top=0x%lx  size=%d bytes\n\r",
        KernelStackBase, KernelStackTop, KERNEL_STACK_SIZE);

    // We have to call GetMemoryMap twice. This first call gets the
    // initial size of the memory map structure. It may or may not change
    // before the second and final call.
    Status = uefi_call_wrapper(SystemTable->BootServices->GetMemoryMap, 5,
        &MemoryMapSize, NULL, &MapKey, &DescriptorSize, &DescriptorVersion);

    // Allocate memory map buffer, with some extra space. It is possible for
    // the memory map to grow in size between the initial call and the second
    // call. To be safe, we add the size of 4 additional memory descriptors,
    // which should be enough in all cases. We'll also add 32 more descriptors
    // for pages table nodes.
    MemoryMapSize += 36 * DescriptorSize;
    MemoryMap = AllocatePool(MemoryMapSize);

    // Finalize memory map struct fields
    MemoryMapInfoStruct->descriptor_size = DescriptorSize;
    MemoryMapInfoStruct->descriptor_ver = DescriptorVersion;
    MemoryMapInfoStruct->memory_map_addr = (UINT64)MemoryMap;
    MemoryMapInfoStruct->kernel_load_addr = KernelBuffer;
    MemoryMapInfoStruct->kernel_image_size = KernelSize;
    BootInfoStruct->memory_map_info = MemoryMapInfoStruct;



    Print(L"[*] Crap loader is starting to load CrapOS\n\r");
    Print(L"[!] Press any key to cancel. It's about to hit the fan...\n\r");
    //uefi_call_wrapper(SystemTable->BootServices->Stall, 1, 1000000);
    Print(L"[!] Start in 3 seconds...\n\r");
    //uefi_call_wrapper(SystemTable->BootServices->Stall, 1, 1000000);
    Print(L"[!] Start in 2 seconds...\n\r");
    //uefi_call_wrapper(SystemTable->BootServices->Stall, 1, 1000000);
    Print(L"[!] Start in 1 second...\n\r");
    //uefi_call_wrapper(SystemTable->BootServices->Stall, 1, 1000000);



    // -------------------------------------------------------------------------
    // Build page tables using a preliminary memory map snapshot.
    //
    // We have to build the page tables now, before ExitBootServices, because
    // `build_page_tables` calls `BS->AllocatePages` for the PT pool, which
    // is only available while boot services are active.
    //
    // We take a preliminary snapshot of the memory map to use as input
    // for the identity-mapping pass. The map may change slightly between
    // this snapshot and the final `ExitBootServices` call, but that is
    // acceptable: we have already added extra buffer space, and the
    // pages we need are already mapped.
    // -------------------------------------------------------------------------
    print_debug(INFO, L"[*] Building page tables...\n\r");

    // Calculate the framebuffer size in bytes for the page-table builder
    UINT64 fb_size = (UINT64)FramebufferInfoStruct->framebuffer_height
        * (UINT64)FramebufferInfoStruct->framebuffer_width
        * (UINT64)(FramebufferInfoStruct->framebuffer_bpp / 8);

    // Take a preliminary memory map snapshot for the identity-map pass
    UINTN PreMapSize = MemoryMapSize;  // In: buffer capacity, out: actual size
    Status = uefi_call_wrapper(SystemTable->BootServices->GetMemoryMap, 5,
        &PreMapSize, MemoryMap, &MapKey, &DescriptorSize, &DescriptorVersion);
    if (EFI_ERROR(Status)) {
        print_debug(ERROR, L"[-] Failed to get preliminary memory map\n\r");
        Print(L"[!FATAL!] Press any key to exit...\n\r");
        while (uefi_call_wrapper(SystemTable->ConIn->ReadKeyStroke, 2,
            SystemTable->ConIn, &Key) != EFI_SUCCESS);
        return Status;
    }

    // Calculate pool size based on actual memory map
    UINTN num_descs = PreMapSize / DescriptorSize;
    UINTN total_pages_to_map = 0;
    EFI_MEMORY_DESCRIPTOR *desc = MemoryMap;
    for (UINTN i = 0; i < num_descs; i++) {
        int should_map = (
            desc->Type == EfiConventionalMemory  ||
            desc->Type == EfiLoaderCode          ||
            desc->Type == EfiLoaderData          ||
            desc->Type == EfiBootServicesCode    ||
            desc->Type == EfiBootServicesData
        );
        if (should_map) total_pages_to_map += desc->NumberOfPages;
        desc = (EFI_MEMORY_DESCRIPTOR *)((UINT8 *)desc + DescriptorSize);
    }

    // Each PT node covers 512 pages. Add overhead for PDPTs, PDs, direct
    // physical map, higher-half mappings, and generous slack.
    UINTN pool_pages = (total_pages_to_map / 512) + 256;

    // Build the full page table hierarchy; returns 0 on allocation failure.
    Pml4 = build_page_tables(
        SystemTable->BootServices,
        MemoryMap,
        PreMapSize,
        DescriptorSize,
        KernelBuffer,
        KernelSize,
        KernelStackBase,
        FramebufferInfoStruct->framebuffer_addr,
        fb_size,
        pool_pages
    );
    if (!Pml4) {
        print_debug(ERROR, L"[-] Failed to build page tables (OOM)\n\r");
        Print(L"[!FATAL!] Press any key to exit...\n\r");
        while (uefi_call_wrapper(SystemTable->ConIn->ReadKeyStroke, 2,
            SystemTable->ConIn, &Key) != EFI_SUCCESS);
        return EFI_OUT_OF_RESOURCES;
    }
    Print(L"[+] Page tables built. PML4 at 0x%lx\n\r", Pml4);

    // Compute the kernel stack top and entry point at their higher-half virtual
    // addresses, used after we load our new PML4 and jump to the kernel.
    KernelStackTop = KernelStackBase + KERNEL_STACK_SIZE - 8;
    UINT64 KernelStackTopVirt = KernelStackTop + KERNEL_VIRTUAL_BASE;
    UINT64 KernelEntryVirt = KERNEL_VIRTUAL_BASE + KERNEL_PHYSICAL_BASE;

    // -------------------------------------------------------------------------
    // Get the final memory map and exit boot services.
    //
    // ExitBootServices terminates UEFI boot services permanently. After
    // this call, no UEFI boot service functions can be used, the UEFI
    // console is gone, and only runtime services and direct hardware access
    // remain available.
    //
    // The MapKey returned by GetMemoryMap must match exactly the one
    // passed to ExitBootServices. If any UEFI activity changed the map
    // between the two calls, ExitBootServices returns EFI_INVALID_PARAMETER
    // and we must retry. We allow up to 10 retries.
    //
    // CRITICAL: No UEFI boot service calls are made between GetMemoryMap
    // and ExitBootServices as any such call would likely invalidate the MapKey.
    // -------------------------------------------------------------------------
    Print(L"[*] Exiting boot services...\n\r");

    UINTN retries = 0;
    while (retries < 10) {
        // Reset to buffer capacity each retry
        UINTN CurrentMapSize = MemoryMapSize;

        // Refresh the memory map and obtain a fresh MapKey
        Status = uefi_call_wrapper(SystemTable->BootServices->GetMemoryMap, 5,
            &CurrentMapSize, MemoryMap, &MapKey, &DescriptorSize,
            &DescriptorVersion);
        if (EFI_ERROR(Status)) {
            retries++;
            continue;
        }

        // Attempt to exit boot services with the MapKey we just received. If
        // the map hasn't changed since GetMemoryMap, this should succeed. If
        // it returns EFI_INVALID_PARAMETER, the map has changed, and we must
        // try again.
        Status = uefi_call_wrapper(SystemTable->BootServices->ExitBootServices,
            2, ImageHandle, MapKey);
        if (Status == EFI_SUCCESS) {
            // Record the final actual map size for the kernel
            MemoryMapInfoStruct->memory_map_size = CurrentMapSize;
            break;
        }
        retries++;
    }

    if (EFI_ERROR(Status)) {
        // ExitBootServices failed even after retries: fatal, cannot proceed
        print_debug(ERROR, L"[-] Failed to exit boot services\n\r");
        Print(L"[!FATAL!] Press any key to exit...\n\r");
        while (uefi_call_wrapper(SystemTable->ConIn->ReadKeyStroke, 2,
            SystemTable->ConIn, &Key) != EFI_SUCCESS);
        return Status;
    }

    // -------------------------------------------------------------------------
    // From this point on, UEFI boot services are gone, and only serial output
    // is available for debugging. No UEFI functions may be called. Next, we
    // need to install our own minimal GDT before loading CR3.
    //
    // On some firmware implementations (notably VMware), UEFI may reclaim
    // the firmware's own GDT pages after ExitBootServices. If we then load
    // a PML4 that no longer maps the firmware GDT, any subsequent segment
    // descriptor access would fault.
    //
    // So, we pre-emptively install our own GDT from static memory, marked
    // `__attribute__((aligned(8)))` to meet x86 alignment requirements. The
    // GDTR pseudo-descriptor is also static so its address is stable.
    //
    // The GDT has the same structure as the kernel's own GDT (3 entries):
    //   [0x00] Null descriptor: required by x86 architecture.
    //   [0x08] 64-bit code segment: ring 0, execute/read with L flag set.
    //   [0x10] 64-bit data segment: ring 0, read/write.
    //
    // After `lgdt`, we reload all segment registers. CS (code segment) requires
    // a far return trick (same as `load_gdt` in kernel). DS/ES/FS/GS/SS (data
    // segments) can be updated with `mov` directly.
    // -------------------------------------------------------------------------
    serial_write("[*] Jumping to higher-half kernel...\n\r");

    // Static storage so the GDT survives the inline asm block
    static UINT64 gdt[] __attribute__((aligned(8))) = {
        0x0000000000000000ULL,  // [0x00] null descriptor
        0x00af9a000000ffffULL,  // [0x08] 64-bit code segment
        0x00af92000000ffffULL,  // [0x10] 64-bit data segment
    };

    // GDTR pseudo-descriptor: 6 bytes packed (2-byte limit + 8-byte base).
    // The __attribute__((packed)) prevents the compiler from inserting padding
    // between the limit and base fields.
    static struct {
        UINT16 limit;              // Size of GDT in bytes, minus 1
        UINT64 base;               // Linear address of the GDT
    } __attribute__((packed)) gdtr = {
        .limit = sizeof(gdt) - 1,  // 3 entries * 8 bytes - 1 = 23
        .base  = (UINT64)gdt,
    };

    __asm__ __volatile__(
        "lgdt %0\n\t"         // Load the new GDTR from memory
        "mov $0x10, %%ax\n\t" // Data segment selector: GDT index 2, ring 0
        "mov %%ax, %%ds\n\t"  // Reload DS (data segment)
        "mov %%ax, %%es\n\t"  // Reload ES (extra segment)
        "mov %%ax, %%fs\n\t"  // Reload FS
        "mov %%ax, %%gs\n\t"  // Reload GS
        "mov %%ax, %%ss\n\t"  // Reload SS (stack segment)
        "pushq $0x08\n\t"     // Push CS selector onto the stack
        "lea 1f(%%rip), %%rax\n\t" // Compute address of label 1 as RIP-relative
        "pushq %%rax\n\t"     // Push return RIP (address of label 1) onto stack
        "lretq\n\t"           // Far return: pops RIP then CS; reloads CS=0x08
        "1:\n\t"              // Execution resumes here with new CS loaded
        : : "m"(gdtr) : "rax", "memory"  // GDTR passed as memory operand;
                                         // rax is clobbered; compiler barrier,
                                         // no memory accesses reordered
    );

    // Enable the NXE (No-Execute Enable) bit in IA32_EFER. Our page tables use
    // PT_NX (bit 63 of PTEs) to mark the stack as non-executable. However, the
    // CPU only honors that flag if the NXE (No-Execute Enable) bit (bit 11) is
    // set in the EFER MSR. QEMU's OVMF typically sets NXE, but VMware firmware
    // does not. We have to set it unconditionally to be safe on all platforms.
    __asm__ __volatile__(
        "mov $0xC0000080, %%ecx\n\t"  // EFER MSR address
        "rdmsr\n\t"                   // Read EFER: EAX = bits 31:0
        "or $0x800, %%eax\n\t"        // Set bit 11 (NXE) in the low 32 bits
        "wrmsr\n\t"                   // Write modified value back to EFER
        : : : "eax", "ecx", "edx"     // Clobbers: these registers are modified
    );

    // Load CR3 with our new PML4 physical address. Writing CR3 activates our
    // new page tables immediately. From this instruction onwards, all
    // virtual-to-physical translations use our PML4.
    __asm__ __volatile__(
        "mov %0, %%cr3\n\t"       // Load Pml4 physical address into CR3
        : : "r"(Pml4) : "memory"  // memory barrier: all pending memory writes
                                  // are committed before the page tables are
                                  // switched
    );

    // -------------------------------------------------------------------------
    // Switch to the higher-half kernel stack and jump to the kernel entry.
    // -------------------------------------------------------------------------

    // Load all needed values into callee-saved registers before changing RSP.
    // After we update RSP, the current stack frame becomes inaccessible because
    // it is at the old physical RSP location, which is now in a different
    // virtual address. If the compiler spills anything to the stack after the
    // RSP switch, it will corrupt or read garbage memory.
    register UINT64 rsp_val asm("rbx") = KernelStackTopVirt;  // New RSP value
    register UINT64 rdi_val asm("r12") = (UINT64)BootInfoStruct;  // Kernel arg
    register UINT64 jmp_val asm("r13") = KernelEntryVirt;  // Kernel entry VMA

    // We zero RBP to mark this as the outermost stack frame. This is by
    // convention: a zero RBP tells debuggers and stack unwinders that there is
    // no caller frame above this point.
    __asm__ __volatile__ (
        "mov %0, %%rsp\n\t"     // Switch RSP to the higher-half virtual stack
        "mov %1, %%rdi\n\t"     // RDI = BootInfoStruct (first argument per ABI)
        "xor %%rbp, %%rbp\n\t"  // Zero RBP: mark as top of call stack
        "jmp *%2\n\t"           // Indirect jump to the kernel entry VMA
        : : "r"(rsp_val), "r"(rdi_val), "r"(jmp_val) : "memory"
    );

    // Will never reach here. The kernel never returns to the bootloader. The
    // `hlt` loop is a defensive safety net in case something goes wrong and
    // execution somehow falls through.
    while(1) {
        __asm__ __volatile__("hlt");
    }

    // Will never reach here. Needed to satisfies the compiler's requirement for
    // a return statement
    return EFI_SUCCESS;
}
