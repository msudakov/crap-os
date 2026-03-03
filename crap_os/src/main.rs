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
use globals::MEMORY_MANAGER;


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





/*#[repr(C, packed)]
#[derive(Copy, Clone)]
struct IdtEntry {
    offset_low:  u16,
    selector:    u16,
    ist:         u8,
    attributes:  u8,
    offset_mid:  u16,
    offset_high: u32,
    _reserved:   u32,
}

#[repr(C, packed)]
struct Idtr {
    limit: u16,
    base:  u64,
}

// A simple handler that just halts
unsafe extern "C" fn exception_handler() -> ! {
    unsafe { core::arch::asm!("cli", "hlt", options(noreturn, nostack)) };
}

static mut IDT: [IdtEntry; 32] = [IdtEntry {
    offset_low:  0,
    selector:    0,
    ist:         0,
    attributes:  0,
    offset_mid:  0,
    offset_high: 0,
    _reserved:   0,
}; 32];

pub fn init_idt() {
    let handler = exception_handler as u64;
    unsafe {
        let idt_ptr = core::ptr::addr_of_mut!(IDT);
        for i in 0..32 {
            (*idt_ptr)[i] = IdtEntry {
                offset_low:  handler as u16,
                selector:    0x8,
                ist:         0,
                attributes:  0x8E,
                offset_high: (handler >> 32) as u32,
                offset_mid:  (handler >> 16) as u16,
                _reserved:   0,
            };
        }

        let idtr = Idtr {
            limit: (core::mem::size_of::<[IdtEntry; 32]>() - 1) as u16,
            base:  idt_ptr as u64,
        };

        core::arch::asm!(
            "lidt [{0}]",
            in(reg) &idtr,
            options(nostack)
        );
    }
}*/






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
    // Disbale maskable hardware interrupts, until we implement IDT
    unsafe { core::arch::asm!("cli", options(nostack, preserves_flags)) };
    
    load_gdt();  // Load our own Global Descriptor Table (GDT)
    crate::serial::init(globals::COM1_PORT);  // Initialize serial port writer

    //init_idt();  // TODO: delete later when we implement our own IDT

    crate::serial::print("[INFO] CrapOS kernel started\n");

    // Validate boot_info pointer
    if boot_info.is_null() {
        crate::serial::print("[ERROR] boot_info is null\n");
        loop { unsafe { core::arch::asm!("hlt") }};
    }

    let info = unsafe { &*boot_info };  // Dereference boot_info

    // Sanity check for magic value
    if info.magic != 0xDEADBEEFB007CAFE {
        crate::serial::print("[CRITICAL] Magic value did not match\n");
        unsafe { core::arch::asm!("hlt") };
    }

    // Dereference framebuffer_info
    let framebuffer = unsafe { &*info.framebuffer_info };
    if framebuffer.framebuffer_addr == 0 {  // Validate framebuffer address
        crate::serial::print("\n!!! KERNEL PANIC !!!\n");
        loop { unsafe { core::arch::asm!("hlt") }}
    }
    crate::serial::print("[INFO] Validated framebuffer address\n");

    // Dereference memory_map_info
    let memory_map = unsafe { &*info.memory_map_info };

    crate::serial::print("[INFO] Mapping physical and virtual memory...\n");
    
    // Initialize PMM and page tables, then perform the higher-half jump.
    // This must happen outside any lock because it relocates the stack.
    let mm = memory_manager::MemoryManager::init(&framebuffer, &memory_map);
    {
        // Now that we're stable in the higher half, store it in the global
        let mut memory_manager = MEMORY_MANAGER.lock();
        *memory_manager = Some(mm);
    }
    crate::serial::print("[INFO] Memory mapped and initialized\n");

    // Instantiate serial port writer
    {
        let mut writer = globals::SERIAL.lock();
        *writer = Some(serial::SerialWriter::new(
            globals::COM1_PORT,
        ));
        // Can now use serial port IRQ-safe global spinlock
    }

    // Instantiate and initialize framebuffer writer
    {
        let mut writer = globals::FRAMEBUFFER.lock();
        *writer = Some(framebuffer::FramebufferWriter::new(
            framebuffer.framebuffer_addr as *mut u32,
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

    // TODO: delete this test later
    sprint_debug!(DebugLevel::INFO, "[INFO] Testing virtual memory...");
    memory_manager::test_vmm();
    sprintln!("[+] Memory tested!");

    // Testing the address pointer of framebuffer
    {
        let mut writer = globals::FRAMEBUFFER.lock();
        if let Some(ref mut writer) = *writer {
            let raw_ptr: *mut u32 = writer.framebuffer;
            sprintln!("The framebuffer address is: 0x{:X}", raw_ptr as usize);
        }
    }
    

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
