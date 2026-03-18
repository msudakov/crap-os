// =============================================================================
// CrapOS Main System Module
// =============================================================================

#![no_std]   // This is an OS kernel; there is no standard library for now
#![no_main]  // Not depending on a runtime, so cannot use main as entry point

mod globals;
mod spinlock;
mod macros;
mod system_routines;
mod hardware_manager;
mod memory_manager;
pub mod gdt;
pub mod idt;
mod tests;

use hardware_manager::framebuffer::FramebufferInfo;
use memory_manager::kernel_heap::GlobalHeapAllocator;

// Need to explicitly linke the built-in alloc crate in a no_std environment
extern crate alloc;

// Register a zero-sized type that implements `GlobalAlloc` by delegating to
// `globals::KERNEL_HEAP`.
#[global_allocator]
static GA: GlobalHeapAllocator = GlobalHeapAllocator;

/// System-wide error code values.
#[repr(u32)]
#[derive(PartialEq)]
pub enum ErrorCode {
    StatusOK = 0x00000000,
    BufferTooSmall = 0x00000001,
}

/// Preset level of debugging messages printed by the system.
#[repr(i32)]
#[allow(dead_code)]
#[derive(PartialEq, Eq, PartialOrd)]
pub enum DebugLevel {
    DEBUG = 1,
    INFO = 2,
    WARNING = 3,
    ERROR = 4,
    CRITICAL = 5
}

/// The BootInfo structure passed to the _start routine by the bootloader when
/// KernelEntry is jumped to and execution is transferred to the kernel.
/// This must match the structure in the C bootloader exactly.
#[repr(C)]
#[derive(Copy, Clone)]
pub struct BootInfo {
    magic: u64,
    framebuffer_info: *const hardware_manager::framebuffer::FramebufferInfo,
    memory_map_info: *const memory_manager::MemoryMapInfo,
}

/// Kernel entry point routine.
/// 
/// Since we're not depending on a runtime or an OS in this bare-metal binary,
/// we can't use the main function as an entry point. This `_start` routine must
/// be exported instead. Also, it is critical that Rust does not mangle the name
/// of the exported routine, thus `no_mangle` is a must.
///
/// # Arguments
///
/// * `boot_info` - Raw pointer to a `BootInfo` structure from the bootloader.
#[unsafe(no_mangle)]
pub extern "C" fn _start(boot_info: *const BootInfo) -> ! {
    // Disable maskable hardware interrupts, until we initialize IDT
    unsafe { core::arch::asm!("cli", options(nostack, preserves_flags)) };

    // Initialize serial port. This is needed here (before the memory manager
    // initialization and the jump to higher-half kernel space) to print debug
    // messages. These writes are not spinlocked at this early stage yet.
    hardware_manager::serial::init(globals::COM1_PORT);
    
    // Validate boot_info pointer, then dereference boot_info
    if boot_info.is_null() {loop{unsafe{core::arch::asm!("hlt")}}}
    let info = unsafe { *boot_info };  // Copy the struct instead of reference

    // Sanity check for magic value
    if info.magic != 0xDEADBEEFB007CAFE {loop{unsafe{core::arch::asm!("hlt")}}}

    // Dereference framebuffer_info by copying the struct instead of reference
    let framebuffer = unsafe { *info.framebuffer_info };
    if framebuffer.framebuffer_addr == 0 {loop{unsafe{core::arch::asm!("hlt")}}}

    // Dereference memory_map_info by copying the struct instead of reference
    let memory_map  = unsafe { *info.memory_map_info };

    // Initialize the physical memory manager. This enumerates and maps physical
    // pages to enable page tables in the next step.
    let mut pmm = memory_manager::pmm::PhysicalMemoryManager::init(
        &framebuffer, &memory_map);

    // Initialize page tabes and store PML4, which is used in the next step to
    // replace CR3 and then inline-jump to higher-half virtual space.
    let pml4 = memory_manager::vmm::init_page_tables(&mut pmm, &framebuffer,
        &memory_map);

    // Inline switch the CR3 for the new PML4
    unsafe {
        core::arch::asm!(
            "mov cr3, {pml4}",
            pml4 = in(reg) pml4,
        );
    }

    // At this point, our RIP and RSP are both in the higher-half space. First,
    // we must re-copy these from the pre-jump variables into new stack slots
    // that are allocated on the post-jump higher-half stack.
    // The pre-jump variables (framebuffer and memory_map) are at physical
    // stack addresses and must not be referenced after this point.
    let framebuffer = framebuffer;
    let memory_map = memory_map;

    // Properly initialize the memory manager from the higher-half kernel space.
    // This uses the physical manager passed as a struct and the PML4 address of
    // the new page tables we created before the jump.
    let mut memory_manager = memory_manager::MemoryManager::init(
        pmm, pml4 as u64);

    // Complete virtual memory initialization. This includes building direct
    // physical map, mapping the framebuffer to the new kernel space, reloading
    // the GDT and the IDT, removing old identity maps, and finally reclaiming
    // boot memory.
    memory_manager.init_higher_half(&framebuffer, &memory_map);
    {
        let mut global_mm = globals::MEMORY_MANAGER.lock();
        *global_mm = Some(memory_manager);
    }

    // We can now safely print basic messages via serial port
    hardware_manager::serial::print("[INFO] Initialized higher-half kernel\n");

    // Initialize serial port writer for global macros
    {
        let mut writer = globals::SERIAL.lock();
        *writer = Some(hardware_manager::serial::SerialWriter::new(
            globals::COM1_PORT,
        ));
    }

    // We can now use serial port IRQ-safe global spinlock macros
    sprint_debug!(DebugLevel::INFO, "[INFO] Serial initialized successfully");

    // Initialie kernel heap and pre-map 16 pages (64 KB)
    globals::KERNEL_HEAP.heap.lock().init(16);

    unsafe { crate::gdt::init_gdt(); }  // Initialize Global Descriptor Table
    unsafe { crate::idt::init_idt(); }  // Initialize Interrupt Descriptor Table

    // IDT is initialized; it is safe to re-enable maskable hardware interrupts
    unsafe { core::arch::asm!("sti", options(nomem, nostack)); }

    // Initialize framebuffer writer for global macros
    {
        let mut writer = globals::FRAMEBUFFER.lock();
        *writer = Some(hardware_manager::framebuffer::FramebufferWriter::new(
            globals::KERNEL_FRAMEBUFFER_VIRTUAL_BASE as *mut u32,
            framebuffer.framebuffer_width,
            framebuffer.framebuffer_height,
        ));
        writer.as_mut().unwrap().clear_screen();
    }
    sprint_debug!(DebugLevel::INFO, "[INFO] Graphics initialized successfully");

    // Draw OS banner
    fbprintln!("Hello and welcome to:\n");
    globals::FRAMEBUFFER.lock().as_mut().unwrap().draw_banner();
    sprint_debug!(DebugLevel::DEBUG, "[DEBUG] Text drawn");

    sprint_debug!(DebugLevel::INFO, "[INFO] Initialized higher-half kernel");
    sprintln!("[+] Kernel higher-half virtual space initialization complete!");
    fbprintln!("[+] Kernel higher-half virtual space initialization complete!\n");
    fbprintln!("  - Kernel virtual base address is: 0x{:X}\n",
    globals::KERNEL_VIRTUAL_BASE);
    fbprintln!("  - Kernel physical map base address is: 0x{:X}\n",
    globals::KERNEL_PHYSICAL_MAP_BASE);

    // Testing the address pointer of framebuffer
    let mut fb_ptr: usize = 0;
    {
        let mut writer = globals::FRAMEBUFFER.lock();
        if let Some(ref mut writer) = *writer {
            let raw_ptr: *mut u32 = writer.framebuffer;
            fb_ptr = raw_ptr as usize;
            sprintln!("Kernel framebuffer address is: 0x{:X}", fb_ptr);
        }
    }
    fbprintln!("  - Kernel framebuffer address is: 0x{:X}\n", fb_ptr);

    // Testing memory manager and heap allocator
    sprintln!("[*] Running general Memory Manager tests...");
    fbprintln!("[*] Running general Memory Manager tests...");
    tests::memory::test_memory_manager();
    sprintln!("[+] All Memory Manager tests passed!\n");
    fbprintln!("[+] All Memory Manager tests passed!\n");
    sprintln!("[*] Running MM heap allocator tests...");
    fbprintln!("[*] Running MM heap allocator tests...");
    tests::memory::test_heap_allocator();
    sprintln!("[+] All MM heap allocator tests passed!\n");
    fbprintln!("[+] All MM heap allocator tests passed!\n");


    // Done for now, loop HALT forever
    loop {
        unsafe { core::arch::asm!("hlt") };
    } 
}

/// Manual panic handler for when we need to crash.
/// 
/// Normally, we'd use the panic handler provided by the standard library, but
/// this is a bare-metal no-dependency binary of the OS kernel. So, we have to
/// implement our own handler.
///
/// # Arguments
///
/// * `info` - Panic info structure for displaying debugging information.
/// 
/// # Safety
/// 
/// Crashes the system and halts the CPU.
#[panic_handler]
fn panic(info: &core::panic::PanicInfo) -> ! {
    // Print this without acquiring the spinlock
    hardware_manager::serial::print("\n!!! KERNEL PANIC !!!\n");

    if let Some(location) = info.location() {
        sprintln!("Panic occurred in file '{}' at line {}", location.file(),
            location.line());
    }

    sprintln!("Panic Message: {}", info.message());

    loop {
        // Halt the CPU
        unsafe { core::arch::asm!("hlt"); }
    }
}
