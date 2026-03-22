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

use hardware_manager::FramebufferInfo;
use memory_manager::MemoryManager;
use memory_manager::GlobalHeapAllocator;

// Need to explicitly link the built-in alloc crate in a no_std environment
extern crate alloc;

// Register a zero-sized type that implements `GlobalAlloc` by delegating to
// `globals::KERNEL_HEAP`.
#[global_allocator]
static GLOBAL_ALLOCATOR: GlobalHeapAllocator = GlobalHeapAllocator;

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
    framebuffer_info: *const hardware_manager::FramebufferInfo,
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
    hardware_manager::serial_init(globals::COM1_PORT);
    
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

    // Parse ACPI info to find APIC addresses. This must be done before Memory
    // Manager init sequence.
    let rsdp_virt = memory_map.rsdp_addr + globals::KERNEL_PHYSICAL_MAP_BASE;
    let apic_info = unsafe {
        hardware_manager::parse_acpi(rsdp_virt).expect("ACPI/MADT not found")
    };

    // Initialize the physical memory manager. This enumerates and maps physical
    // pages to enable page tables in the next step.
    let mut pmm = memory_manager::PhysicalMemoryManager::init(
        &framebuffer, &memory_map);

    // Initialize page tabes and store PML4, which is used in the next step to
    // replace CR3 and then inline-jump to higher-half virtual space.
    let pml4 = memory_manager::init_page_tables(&mut pmm, &framebuffer,
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
    let apic_info = apic_info;

    // Properly initialize the memory manager from the higher-half kernel space.
    // This uses the physical manager passed as a struct and the PML4 address of
    // the new page tables we created before the jump.
    let mut memory_manager = MemoryManager::init(
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
    hardware_manager::sprint("[INFO] Initialized higher-half kernel\n");

    // Initialize serial port writer for global macros
    {
        let mut writer = globals::SERIAL.lock();
        *writer = Some(hardware_manager::SerialWriter::new(
            globals::COM1_PORT,
        ));
    }

    // We can now use serial port IRQ-safe global spinlock macros
    sprint_debug!(DebugLevel::INFO, "[INFO] Serial initialized successfully");

    // Initialie kernel heap and pre-map 16 pages (64 KB)
    globals::KERNEL_HEAP.heap.lock().init(16);

    unsafe { crate::gdt::init_gdt(); }  // Initialize Global Descriptor Table
    unsafe { crate::idt::init_idt(); }  // Initialize Interrupt Descriptor Table

    // Initialize framebuffer writer for global macros
    {
        let mut writer = globals::FRAMEBUFFER.lock();
        *writer = Some(hardware_manager::FramebufferWriter::new(
            globals::KERNEL_FRAMEBUFFER_VIRTUAL_BASE as *mut u32,
            framebuffer.framebuffer_width,
            framebuffer.framebuffer_height,
        ));
        writer.as_mut().unwrap().clear_screen();
    }
    sprint_debug!(DebugLevel::INFO, "[INFO] Graphics initialized successfully");

    // If the firmware leaves the 8259 PIC enabled (which UEFI typically does),
    // we must disable it before enabling the APIC, or the PIC will fire
    // spurious IRQs on vectors 0x08–0x0F that collide with our CPU exceptions.
    unsafe { hardware_manager::disable_pic_8259() };
    sprint_debug!(DebugLevel::DEBUG, "[INFO] Disabled legacy PIC 8259");

    // Initialize and configure APICs
    unsafe {
        // Initialize Local APIC and I/O APIC
        hardware_manager::init_apic(apic_info.local_apic_phys,
            apic_info.io_apic_phys,
        );
        sprint_debug!(DebugLevel::DEBUG, "[DEBUG] APICs have been initialized");

        // Configure the APIC timer (tune initial_count as needed for testing)
        hardware_manager::configure_timer(1000000);
        sprint_debug!(DebugLevel::DEBUG, "[DEBUG] Timer interrupt initialized");

        // Unmask the keyboard IRQ in the I/O APIC
        hardware_manager::ioapic_unmask_irq(1);
        sprint_debug!(DebugLevel::DEBUG, "[DEBUG] Keyboard interrupts ready");
    }

    // IDT is initialized, and the APIC is set up with the registered interrupt
    // handlers. It is now safe to re-enable maskable hardware interrupts.
    unsafe { core::arch::asm!("sti", options(nomem, nostack)); }

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

    sprintln!("[+] ACPI and APICs initialized successfully!");
    fbprintln!("[+] ACPI and APICs initialized successfully!\n");
    sprintln!("[+] IDT ready, interrupts enabled...");
    fbprintln!("[+] IDT ready, interrupts enabled...\n");

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

    // Testing timer interrupts
    sprintln!("[*] Testing IRQ timer interrupts...");
    fbprintln!("[*] Testing IRQ timer interrupts...");
    let mut last_tick = system_routines::get_timer_ticks();
    let mut counter = 0;
    loop {
        let current = system_routines::get_timer_ticks();
        if current != last_tick {
            sprint!(".");
            fbprint!(".");
            last_tick = current;
            counter += 1
        }

        if counter > 30 {
            break;
        }

        // Halt until the next interrupt to avoid spinning the CPU at 100%
        unsafe { core::arch::asm!("hlt", options(nomem, nostack)); }
    }
    sprintln!("\n[+] IRQ timer interrupt test complete!\n");
    fbprintln!("\n[+] IRQ timer interrupt test complete!\n");

    // Testing keyboard interrupts
    sprintln!("[*] Testing keyboard interrupts. Type some stuff...");
    fbprintln!("[*] Testing keyboard interrupts. Type some stuff...");

    // Halt until the next interrupt to avoid spinning the CPU at 100%
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
    hardware_manager::sprint("\n!!! KERNEL PANIC !!!\n");

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
