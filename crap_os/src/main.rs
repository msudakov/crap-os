#![no_std]   // This is an OS kernel; there is no standard library for now
#![no_main]  // Not depending on a runtime, so cannot use main as entry point

mod globals;
mod spinlock;
mod macros;
mod system_routines;
mod serial;
mod framebuffer;
mod memory_manager;

use serial::print_debug;
use serial::print;
use serial::println;
use framebuffer::FramebufferInfo;
use memory_manager::MemoryMapInfo;
use memory_manager::PhysicalMemoryManager;


// Structure for the Global Descriptor Table
#[repr(C, align(8))]
struct Gdt {
    null:    u64,
    code64:  u64,
    data64:  u64,
}

// Structure for the GDT register
#[repr(C, packed)]
struct Gdtr {
    limit: u16,
    base:  u64,
}

/*
    When the _start routine is invoked from the bootloader, we're still
    running under UEFI's GDT and CS segment. This creates our own GDT that the
    kernel loads at the very start of the routine.
*/
static GDT: Gdt = Gdt {
    null:   0x0000000000000000,
    code64: 0x00AF9A000000FFFF,  // 64-bit code, ring 0
    data64: 0x00CF92000000FFFF,  // 64-bit data, ring 0
};

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
pub const DEBUG_LEVEL: DebugLevel = DebugLevel::INFO;

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

/// Replaces the bootloader's GDT with the kernel's GDT.
/// 
/// # Safety
/// 
/// Uses inline assembly to load the OS kernel's own GDT.
fn load_gdt() {
    let gdtr = Gdtr {
        limit: (core::mem::size_of::<Gdt>() - 1) as u16,
        base:  &GDT as *const Gdt as u64,
    };
    unsafe {
        core::arch::asm!(
            "lgdt [{0}]",
            "push 0x8",
            "lea rax, [{1}]",
            "push rax",
            "retfq",
            "mov ax, 0x10",
            "mov ds, ax",
            "mov es, ax",
            "mov ss, ax",
            in(reg) &gdtr,
            label {()},
            options(nostack)
    )};
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
    load_gdt();  // Must be executed before anything else

    unsafe {
        // Clear the interrupt flag to disable maskable interrupts
        core::arch::asm!("cli");

        serial::init_serial();  // Initialize serial port for debugging
        print_debug(DebugLevel::INFO, "[INFO] Kernel started");

        if boot_info.is_null() {  // Validate boot_info pointer
            print_debug(DebugLevel::CRITICAL, "[ERROR] boot_info is null");
            loop { core::arch::asm!("hlt"); }
        }
    }

    let info = unsafe { &*boot_info };  // Dereference boot_info

    // Sanity check for magic value
    if info.magic != 0xDEADBEEFB007CAFE {
        print_debug(DebugLevel::CRITICAL,
            "[CRITICAL] Magic value did not match");
        unsafe { core::arch::asm!("hlt") };
    }
    print_debug(DebugLevel::DEBUG, "[DEBUG] Magic value matched");
    print_debug(DebugLevel::DEBUG, "[DEBUG] Got BootInfo structure");

    // Dereference framebuffer_info
    let framebuffer = unsafe { &*info.framebuffer_info };
    print_debug(DebugLevel::DEBUG, "[DEBUG] Framebuffer info read");

    if framebuffer.framebuffer_addr == 0 {  // Validate framebuffer address
        print_debug(DebugLevel::ERROR, "[ERROR] framebuffer address is 0");
        loop { unsafe { core::arch::asm!("hlt") }}
    }
    print_debug(DebugLevel::INFO, "[INFO] Validated framebuffer addr");

    // Dereference memory_map_info
    let memory_map = unsafe { &*info.memory_map_info };

    //let mut writer = FramebufferWriter::new(
    //    framebuffer.framebuffer_addr as *mut u32,
    //    framebuffer.framebuffer_width,
    //    framebuffer.framebuffer_height,
    //);
    //writer.clear_screen();
    {
        let mut writer = globals::FRAMEBUFFER.lock();
        *writer = Some(framebuffer::FramebufferWriter::new(
            framebuffer.framebuffer_addr as *mut u32,
            framebuffer.framebuffer_width,
            framebuffer.framebuffer_height,
        ));
        writer.as_mut().unwrap().clear_screen();
    }
    print_debug(DebugLevel::INFO, "[INFO] Graphics initialized successfully");
    
    // Draw OS banner
    fbprintln!("Hello and {} to:\n", "welcome");
    globals::FRAMEBUFFER.lock().as_mut().unwrap().draw_banner();
    print_debug(DebugLevel::DEBUG, "[DEBUG] Text drawn");


    print_debug(DebugLevel::INFO, "[INFO] Mapping available physical memory");
    let mut pmm = PhysicalMemoryManager::init(&framebuffer, &memory_map);
    print_debug(DebugLevel::INFO, "[INFO] Available physical memory mapped");
    print_debug(DebugLevel::INFO, "[INFO] Mapping virtual memory...");
    let _pml4 =  memory_manager::init_page_tables(&mut pmm, &framebuffer, &memory_map);
    print_debug(DebugLevel::INFO, "[INFO] Virtual memory mapped");
    print_debug(DebugLevel::INFO, "[INFO] Testing virtual memory...");
    memory_manager::test_vmm(&mut pmm);

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
    print("\n!!! KERNEL PANIC !!!\n");
    
    if let Some(location) = info.location() {
        print("Location: ");
        println(location.file());
    }

    loop {
        // Halt the CPU
        unsafe { core::arch::asm!("hlt"); }
    }
}
