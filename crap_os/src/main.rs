//! CrapOS Main System Module

#![no_std]   // This is an OS kernel; there is no standard library for now
#![no_main]  // Not depending on a runtime, so cannot use main as entry point

mod globals;
mod spinlock;
mod macros;
mod processor_control;
mod helper_functions;
mod hardware_manager;
mod memory_manager;
mod system_core;
mod task_scheduler;
mod process_manager;
mod crypto;
mod tests;

use hardware_manager::FramebufferInfo;
use memory_manager::{MemoryManager, GlobalHeapAllocator};
use process_manager::nop_thread_stub;
use crate::task_scheduler::sleep;

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
    let (apic_info, cpu_topology) = unsafe {
        hardware_manager::parse_acpi(rsdp_virt, memory_map.bsp_apic_id).expect(
            "ACPI/MADT not found")
    };
    let hpet_info = unsafe {
        hardware_manager::parse_hpet(rsdp_virt).expect(
            "HPET not found or invalid. Cannot calibrate APIC timer...")
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
    let hpet_info = hpet_info;
    let cr3: u64 = pml4 as u64;

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



    // CPU parsing test
    {
        sprintln!("[INFO] CPU topology from ACPI MADT:");
        sprintln!("[INFO]   BSP APIC ID (bootloader CPUID): {}", cpu_topology.bsp_apic_id);
        sprintln!("[INFO]   Total CPUs in MADT: {}", cpu_topology.cpu_count);
        sprintln!("[INFO]   Usable CPUs (enabled or online-capable): {}", cpu_topology.get_usable_cpu_count());
 
        for cpu in cpu_topology.iter() {
            sprintln!("[INFO]   CPU | APIC ID: {:#04x} | ACPI UID: {:#04x} | Enabled: {} | Online-capable: {} | BSP: {}",
                cpu.apic_id,
                cpu.acpi_uid,
                cpu.enabled,
                cpu.online_capable,
                cpu.apic_id == cpu_topology.bsp_apic_id,
            );
        }
        if cpu_topology.bsp().is_none() {
            panic!("BSP APIC ID {:#x} not found in MADT — firmware bug or struct layout mismatch", cpu_topology.bsp_apic_id);
        }
    }

    // Initialize CPU topology
    processor_control::init_cpu_topology(cpu_topology);



    // Initialie kernel heap and pre-map 16 pages (64 KB)
    globals::KERNEL_HEAP.heap.lock().init(16);

    // Initialize Global Descriptor Table
    unsafe { processor_control::gdt::init_gdt(); }

    // Initialize Interrupt Descriptor Table
    unsafe { crate::processor_control::init_idt(); }
    unsafe { crate::processor_control::load_idt(); }

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

    // Initialize Local APIC and I/O APIC; also needed for timer calibration
    unsafe {
        hardware_manager::init_apic(apic_info.local_apic_phys,
            apic_info.io_apic_phys,
        );
        sprint_debug!(DebugLevel::DEBUG, "[DEBUG] APICs have been initialized");
    }

    // Initialize High Precision Event Timer (HPET) and calibrate APIC timer
    {
        // Initialize HPET
        let mut hpet = globals::HPET.lock();
        *hpet = Some(hpet_info);

        // Calculate the number of APIC timer ticks per millisecond
        let apic_ticks_per_ms = unsafe {
            hardware_manager::calibrate_timer(hpet.as_ref().unwrap())
        };
        sprint_debug!(DebugLevel::INFO, "[INFO] Calibrated APIC timer");
        sprintln!("[INFO] APIC timer: {} ticks/ms", apic_ticks_per_ms);

        // Configure the APIC timer (currently 1ms tick rate)
        unsafe { hardware_manager::configure_timer(apic_ticks_per_ms) };
        sprint_debug!(DebugLevel::DEBUG, "[DEBUG] Timer interrupt initialized");

        // Initialize cryptographically-secure random number generator
        unsafe { crypto::init_cpu(&hpet.as_ref().unwrap()); }
        // To initialize RNG on an AP as part of bring up:
        // unsafe { crypto::init_cpu(&HPET); } // Use global HpetInfo
    }

    // Unmask the keyboard IRQ in the I/O APIC
    unsafe { hardware_manager::ioapic_unmask_irq(1) };
    sprint_debug!(DebugLevel::DEBUG, "[DEBUG] Keyboard interrupts ready");

    // Initialize the Task Scheduler, register the main kernel _start
    // routine as the idle task for the scheduler, and create the system idle
    // process and the first idle thread (this thread).
    let _idle_process = globals::PROCESS_MANAGER.init_idle_process(cr3);

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

    /*sprintln!("[+] ACPI and APICs initialized successfully!");
    fbprintln!("[+] ACPI and APICs initialized successfully!\n");
    sprintln!("[+] IDT ready, interrupts enabled...");
    fbprintln!("[+] IDT ready, interrupts enabled...\n");*/

    // Testing memory manager and heap allocator
    /*sprintln!("[*] Running general Memory Manager tests...");
    fbprintln!("[*] Running general Memory Manager tests...");
    tests::memory::test_memory_manager();
    sprintln!("[+] All Memory Manager tests passed!\n");
    fbprintln!("[+] All Memory Manager tests passed!\n");
    sprintln!("[*] Running MM heap allocator tests...");
    fbprintln!("[*] Running MM heap allocator tests...");
    tests::memory::test_heap_allocator();
    sprintln!("[+] All MM heap allocator tests passed!\n");
    fbprintln!("[+] All MM heap allocator tests passed!\n");*/


    // Create and initialize the System process
    let system_process = globals::PROCESS_MANAGER.create_kernel_process(
        "System",
        cr3,
        nop_thread_stub,
        0
    ).expect("[FATAL ERROR] Failed to create System process");

    // Spawn keyboard buffer reader thread in the System process
    system_process.spawn_kernel_thread("Keyboard reader", task_keyboard, 0).expect(
        "Failed to spawn keyboard thread");
    
    // Testing keyboard interrupts
    sprintln!("[*] Testing keyboard interrupts. Type some stuff...");
    fbprintln!("[*] Testing keyboard interrupts. Type some stuff...");

    // Create Test process
    let test_process_1 = globals::PROCESS_MANAGER.create_kernel_process(
        "Test Proc 1",
        cr3,
        task_a,
        0
    ).expect("Failed to create test process");
    test_process_1.spawn_kernel_thread("P1 T2", task_b, 0).expect("failed to spawn task B");
    
    //let test_process_2 = globals::PROCESS_MANAGER.create_kernel_process(
    //    "Test Proc 2",
    //    cr3,
    //    task_c,
    //    0
    //).expect("Failed to create test process");
    test_process_1.spawn_kernel_thread("Fault task", task_fault, 0).expect("failed to spawn Fault Task");
    let thread_c = test_process_1.spawn_kernel_thread("Task C", task_c, 0).expect("failed to spawn task C");
    crate::process_manager::thread::exit_thread(thread_c);

    // Testing user-mode processes. This is very clunky and is only here for
    // testing purposes...
    // This opcode sequence is: push rax; pop rax; jmp -4
    // It exercises the user stack without doing anything privileged
    const USER_PAYLOAD: &[u8] = &[0x50, 0x58, 0xEB, 0xFC];
    const USER_CODE_VIRT: u64 = 0x0000_0000_0040_0000;
    // Allocate and populate the code page
    let code_phys = {
        let mut mm_guard = globals::MEMORY_MANAGER.lock();
        let mm = mm_guard.as_mut().unwrap();
        let phys = mm.alloc_page()
            .expect("[FATAL] OOM allocating user code page");
        let virt = phys + globals::KERNEL_PHYSICAL_MAP_BASE;
        unsafe {
            core::ptr::copy_nonoverlapping(
                USER_PAYLOAD.as_ptr(),
                virt as *mut u8,
                USER_PAYLOAD.len(),
            );
        }
        phys
    };
    // Create the process - AddressSpace::init runs here, PML4 is fresh
    let user_process = globals::PROCESS_MANAGER.create_user_process(
        "User Test",
        cr3,
        USER_CODE_VIRT,
    ).expect("Could not create user process");
    // Map the code page into the process's address space now that
    // we have the PML4. This must happen before interrupts are enabled so the
    // thread cannot be scheduled before the mapping is in place.
    {
        let mut mm_guard = globals::MEMORY_MANAGER.lock();
        let mm = mm_guard.as_mut().unwrap();
        unsafe {
            mm.map_page(
                user_process.pml4_phys(),
                USER_CODE_VIRT,
                code_phys,
                crate::memory_manager::PRESENT | crate::memory_manager::USER,
            );
        }
    }
    



    // Signal the Task Scheduler that the kernel has completed its
    // initialization sequence. After this, the idle task (this task) will only
    // be selected to run if no other tasks are available and ready to run.
    globals::SYS_FLAG_KERNEL_INIT_COMPLETE.store(true,
        core::sync::atomic::Ordering::SeqCst);
    
    // IDT is initialized, and the APIC is set up with the registered interrupt
    // handlers. The System process is also initialized, and it is now safe to
    // re-enable maskable hardware interrupts.
    unsafe { core::arch::asm!("sti", options(nomem, nostack)); }


    //globals::PROCESS_MANAGER.print_processes();

    // Enter halt loop on the idle task
    let mut count = 0;
    loop {
        //crate::hardware_manager::sprint("+");
        count += 1;
        if count == 10 {
            count = 0;
            //globals::PROCESS_MANAGER.print_processes();
        }
        unsafe { core::arch::asm!("hlt", options(nomem, nostack)); }
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
    hardware_manager::sprint("\n!!! KERNEL PANIC !!!\n");

    if let Some(location) = info.location() {
        hardware_manager::sprint("Panic occurred in file: ");
        hardware_manager::sprint(location.file());
        hardware_manager::sprint("\n");

        if let Some(message) = info.message().as_str() {
            hardware_manager::sprint("Panic Message: ");
            hardware_manager::sprint(message);
            hardware_manager::sprint("\n");
        }

        sprintln!("Panic occurred in file '{}' at line {}", location.file(),
            location.line());
    }

    sprintln!("Panic Message: {}", info.message());

    loop {
        // Halt the CPU
        unsafe { core::arch::asm!("hlt"); }
    }
}

/// Registers a keyboard buffer reader task with the ISR, then loops through
/// the buffer and processes all scancodes in it; finally, blocks itself via
/// `yield_blocked()` until the next it is it woken.
/// 
/// # Arguments
///
/// * `_arg` - Unused argument; accepted for conformity.
fn task_keyboard(_arg: u64) {
    // Register this task, so the keyboard ISR knows who to wake
    hardware_manager::keyboard_set_task_id(
        task_scheduler::get_current_task_id());

    loop {
        // Drain everything currently in the ring buffer
        while let Some(scancode) = hardware_manager::keyboard_pop_scancode() {
            if let Some(ascii) = hardware_manager::process_scancode(scancode) {
                // TODO:
                // Placeholder system shutdown control sequence (CTRL+ALT+ESC),
                // which returns as 0xFF for now.
                if ascii == 0xFF {
                    fbprint!("SHUTDOWN...");
                    continue;
                }

                // Convert the single byte to a str slice and print it to
                // serial and framebuffer for now.
                let buf = [ascii];
                if let Ok(s) = core::str::from_utf8(&buf) {
                    sprint!("{}", s);
                    fbprint!("{}", s);
                }
            }
        }

        // Buffer is empty, block until the ISR wakes us on the next key event
        task_scheduler::yield_blocked();
    }
}

#[allow(dead_code)]
fn task_a(_arg: u64) {
    loop {
        //crate::hardware_manager::sprint("\n* HELLO from Process 1 Thread 1");
        //fbprint!("\n* HELLO from Process 1 Thread 1");
        crate::hardware_manager::sprint("A");
        sleep(1);
    }
}

#[allow(dead_code)]
fn task_b(_arg: u64) {
    //loop {
    for _ in 0..3 {
        crate::hardware_manager::sprint("Hello, world");
        //crate::hardware_manager::sprint("\n# HOWDY from Process 1 Thread 2");
        //fbprint!("\n# HOWDY from Process 1 Thread 2");
        sleep(1);
    }
}

#[allow(dead_code)]
fn task_fault(_arg: u64) {
    for _ in 0..1_000_000 {
        unsafe { core::arch::asm!("nop"); }
    }
    // Deliberately trigger a #DE divide by zero
    unsafe {
        core::arch::asm!(
            "xor rdx, rdx",
            "xor rax, rax",
            "xor rcx, rcx",
            "div rcx",
            options(nomem, nostack)
        );
    }
}

#[allow(dead_code)]
fn task_c(_arg: u64) {
    loop {
        //crate::hardware_manager::sprint("\n% HOLA from Process 2 Thread 1");
        //fbprint!("\n% HOLA from Process 2 Thread 1");
        crate::hardware_manager::sprint(".");
        sleep(1);
    }
}
