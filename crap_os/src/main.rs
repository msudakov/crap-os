#![no_std]   // This is an OS kernel; there is no standard library for now
#![no_main]  // Not depending on a runtime, so cannot use main as entry point

mod stdlib;
mod serial_printer;
mod framebuffer;
mod memory_manager;

use core::fmt::Write;
use serial_printer::print_debug;
use serial_printer::print;
use serial_printer::println;
use framebuffer::FramebufferInfo;
use framebuffer::FramebufferWriter;
use memory_manager::MemoryMapInfo;
use memory_manager::PhysicalMemoryManager;


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
    unsafe {
        // Clear the interrupt flag to disable maskable interrupts
        core::arch::asm!("cli");

        serial_printer::init_serial();  // Initialize serial port for debugging
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

    let mut writer = FramebufferWriter::new(
        framebuffer.framebuffer_addr as *mut u32,
        framebuffer.framebuffer_width,
        framebuffer.framebuffer_height,
    );
    writer.clear_screen();
    print_debug(DebugLevel::INFO, "[INFO] Graphics initialized successfully");
    
    // Writer example that allows basic string formatting
    write!(writer, "Hello and {} to:\n\n", "welcome").unwrap();

    // Draw OS banner
    writer.draw_banner();
    print_debug(DebugLevel::DEBUG, "[DEBUG] Text drawn");

    /*
    writer.println("Mapping available physical memory (type 0x7):\n");
    print_debug(DebugLevel::INFO, "[INFO] Mapping available physical memory (type 0x7):\n");
    writer.println(
        "Type:       Physical Start:     Virtual Start:      Num Pages:          Attributes:");
    println("Type:       Physical Start:     Virtual Start:      Num Pages:          Attributes:");

    let mut descriptor_addr = memory_map.memory_map_addr;
    let num_segments = memory_map.memory_map_size / memory_map.descriptor_size;

    for _ in 0..num_segments {
        let memory_descriptor = EfiMemoryDescriptor::new(descriptor_addr);
        descriptor_addr += memory_map.descriptor_size;

        // We'll start working only with basic available memory without
        // recliaming boot loader and services memory for now.
        if memory_descriptor.region_type != EfiMemoryType::EfiConventionalMemory {
            continue;
        }
        let region_type = crate::stdlib::u32_to_hex_bytes(
            memory_descriptor.region_type as u32);
        writer.print_bytes(&region_type);
        writer.print("  ");
        print_bytes(&region_type);
        print("  ");

        let physical_start = crate::stdlib::u64_to_hex_bytes(
            memory_descriptor.physical_start);
        writer.print_bytes(&physical_start);
        writer.print("  ");
        print_bytes(&physical_start);
        print("  ");

        let virtual_start = crate::stdlib::u64_to_hex_bytes(
            memory_descriptor.virtual_start);
        writer.print_bytes(&virtual_start);
        writer.print("  ");
        print_bytes(&virtual_start);
        print("  ");

        let num_pages = crate::stdlib::u64_to_hex_bytes(
            memory_descriptor.num_pages);
        writer.print_bytes(&num_pages);
        writer.print("  ");
        print_bytes(&num_pages);
        print("  ");

        let attribute = crate::stdlib::u64_to_hex_bytes(
            memory_descriptor.attribute);
        writer.print_bytes(&attribute);
        writer.println("");
        print_bytes(&attribute);
        println("");
    }

    writer.println("[+] Available physical memory mapped!");
    print_debug(DebugLevel::INFO, "[INFO] Available physical memory mapped!");
    */

    print_debug(DebugLevel::INFO, "[INFO] Mapping available physical memory (type 0x7):\n");

    let mut pmm = PhysicalMemoryManager::init(&framebuffer, &memory_map);

    /*let cookie: u64 = 0xDEADBEEFCAFEBABE;
    let page1 = pmm.alloc_page();
    match page1 {
        Some(addr) => {
            let ptr = (addr + 0x10) as *mut u64;
            unsafe {
                use core::ptr;
                ptr::write_volatile(ptr, cookie);
            }
            println("[OK]: Wrote to address 1");
            pmm.free_page(addr);
            println("[OK]: Freed page 1");
        }
        None => {
            println("[ERROR]: failed to allocate 1");
        }
    }*/

    let cr3: u64;
    unsafe { core::arch::asm!("mov {}, cr3", out(reg) cr3) };
    let cre3_hex = crate::stdlib::u64_to_hex_bytes(cr3);
    println("CR3 (PML4):");
    serial_printer::print_bytes(&cre3_hex);
    println("");

    print_debug(DebugLevel::INFO, "[INFO] Available physical memory mapped!");
    

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
        print(location.file());
        print("\n");
    }

    loop {
        // Halt the CPU
        unsafe { core::arch::asm!("hlt"); }
    }
}
