// =============================================================================
// CrapOS Main System Module
// =============================================================================

#![no_std]   // This is an OS kernel; there is no standard library for now
#![no_main]  // Not depending on a runtime, so cannot use main as entry point

mod globals;
mod spinlock;
mod macros;
mod system_routines;
mod serial;
mod framebuffer;
mod memory_manager;

use framebuffer::FramebufferInfo;
use memory_manager::MemoryMapInfo;
use memory_manager::PhysicalMemoryManager;
use globals::MEMORY_MANAGER;
use crate::memory_manager::MemoryManager;


#[repr(u32)]
#[derive(PartialEq)]
pub enum ErrorCode {
    StatusOK = 0x00000000,
    BufferTooSmall = 0x00000001,
}

// Preset level of debugging messages sent by the kernel
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

/*
    These are the BootInfo structures that are passed to the _start routine by
    the bootloader when KernelEntry is called and execution is transferred to
    the kernel. They must match the structures in the C bootloader exactly.
*/
#[repr(C)]
pub struct BootInfo {
    magic: u64,
    framebuffer_info: *const FramebufferInfo,
    memory_map_info: *const MemoryMapInfo,
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
#[unsafe(link_section = ".text._start")]
pub extern "C" fn _start(boot_info: *const BootInfo) -> ! {
    // Disbale maskable hardware interrupts, until we implement IDT
    unsafe { core::arch::asm!("cli", options(nostack, preserves_flags)) };

    // Initialize serial port. This is needed here, before the memory manager
    // initialization and the jump to higher-half kernel space, to print debug
    // messages. These writes are not spinlocked at this early staged yet.
    crate::serial::init(globals::COM1_PORT);
    
    // Validate boot_info pointer
    if boot_info.is_null() {loop{unsafe{core::arch::asm!("hlt")}}}

    let info = unsafe { &*boot_info };  // Dereference boot_info

    // Sanity check for magic value
    if info.magic != 0xDEADBEEFB007CAFE {loop{unsafe{core::arch::asm!("hlt")}}}

    // Dereference framebuffer_info
    let framebuffer = unsafe { *info.framebuffer_info };
    if framebuffer.framebuffer_addr == 0 {loop{unsafe{core::arch::asm!("hlt")}}}

    // Dereference memory_map_info
    let memory_map  = unsafe { *info.memory_map_info };

    // Initialize the physical memory manager
    let mut pmm = PhysicalMemoryManager::init(&framebuffer, &memory_map);

    // Initialize page tabes and store PML4
    let pml4 = memory_manager::init_page_tables(&mut pmm, &framebuffer,
        &memory_map);

    // Inline switch the CR3 for the new PML4, then jump to higher half kernel.
    // Execution continues at the label 3, with RSP and RIP both in higher half.
    unsafe {
        core::arch::asm!(
            "mov cr3, {pml4}",        // Switch CR3 to the new PML4
            "mov rax, rsp",           // RSP -> RAX
            "add rax, {base}",        // Increment stack pointer by kernel base
            "mov rsp, rax",           // RAX -> RSP updates RSP with new base
            "lea rax, [rip + 3f]",    // Increment RIP with new base and offset
            "add rax, {base}",        // Position jump address in RAX
            "jmp rax",                // Jump to higher half
            "3:",                     // Numeric label to keep compiler happy
            pml4 = in(reg) pml4,
            base = const globals::KERNEL_VIRTUAL_BASE,
        )
    };

    // Re-copy these from the pre-jump variables into new stack slots
    // that are allocated on the post-jump higher-half stack.
    // The pre-jump variables (framebuffer and memory_map) are at physical
    // stack addresses and must not be referenced after this point.
    let framebuffer  = framebuffer;
    let memory_map   = memory_map;

    // Properly initialize the memory manager from the higher-half kernel space
    let mut memory_manager = MemoryManager::init(pmm, pml4 as u64);
    memory_manager.init_higher_half(&framebuffer, &memory_map);
    {
        let mut global_mm = MEMORY_MANAGER.lock();
        *global_mm = Some(memory_manager);
    }

    // Can now safely print basic messages via serial port
    crate::serial::print("[INFO] Initialized higher half kernel\n");

    // Instantiate serial port writer for macros
    {
        let mut writer = globals::SERIAL.lock();
        *writer = Some(serial::SerialWriter::new(
            globals::COM1_PORT,
        ));
    }

    // Can now use serial port IRQ-safe global spinlock macros
    sprint_debug!(DebugLevel::INFO, "[INFO] Serial initialized successfully");

    // Instantiate and initialize framebuffer writer
    {
        let mut writer = globals::FRAMEBUFFER.lock();
        *writer = Some(framebuffer::FramebufferWriter::new(
            globals::KERNEL_FRAMEBUFFER_VIRTUAL_BASE as *mut u32,
            framebuffer.framebuffer_width,
            framebuffer.framebuffer_height,
        ));
        writer.as_mut().unwrap().clear_screen();
    }
    sprint_debug!(DebugLevel::INFO, "[INFO] Graphics initialized successfully");

    // Draw OS banner
    fbprintln!("Hello and {} to:\n", "welcome");
    globals::FRAMEBUFFER.lock().as_mut().unwrap().draw_banner();
    sprint_debug!(DebugLevel::DEBUG, "[DEBUG] Text drawn");

    sprint_debug!(DebugLevel::INFO, "[INFO] Initialized higher half kernel");
    sprintln!("[+] Kernel higher-half virtual space initialization complete!");
    fbprintln!("[+] Kernel higher-half virtual space initialization complete!\n");

    fbprintln!("  - Kernel virtual base address is: 0x{:X}\n", globals::KERNEL_VIRTUAL_BASE);
    fbprintln!("  - Kernel physical map base address is: 0x{:X}\n", globals::KERNEL_PHYSICAL_MAP_BASE);

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

    
    // Done for now.. loop forever and ever
    loop {
        //core::arch::asm!("hlt");
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
    crate::serial::print("\n!!! KERNEL PANIC !!!\n");

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
