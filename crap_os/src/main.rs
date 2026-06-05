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






    // --- BLAKE2b tests ---
    // --- RFC 7693 Appendix A known-answer tests ---
    /*
    BLAKE2b-512(""):
  0x786A02F742015903C6C6FD852552D272912F4740E15847618A86E217F71F5419D25E1031AFEE585313896444934EB04B903A685B1448B755D56F701AFE9BE2CE
BLAKE2b-512("abc"):
  0xBA80A53F981C4D0D6A2797B69F12F6E94C212F14685AC4B74B12BB6FDBFFA2D17D87C5392AAB792DC252D5DE4533CC9518D38AA8DBF1925AB92386EDD4009923
BLAKE2b-256("abc"):
  0xBDDD813C634239723171EF3FEE98579B94964E3BB1CB3E427262C8C068D52319
BLAKE2b-20("abc"):
  0x44229FC0EF
BLAKE2b-512 MAC (RFC 7693 Appendix C vector):
  0x8D6CF87C08380D2D1506EEE46FD4222D21D8C04E585FBFD08269C98F702833A156326A0724656400EE09351D57B440175E2A5DE93CC5F80DB6DAF83576CF75FA
BLAKE2b-256 MAC:
  0x33A1301490C833CAE224B593D17D0FA5B5F09650ACAB559F44B2AA3D5971D330
BLAKE2b-384("Hello", key="secret key"):
  0xA520837C8D7E1948D7C02B7A93DB7B4349A69A4506947BB94A62A70420CE02052C2226F2F70378FBE48C764185389B85
blake2b(out_len=0)  -> None (correct)
blake2b(key=empty) -> None (correct)
blake2b(out_len=65) -> None (correct)
BLAKE2b-512("a" * 1000):
  0xD6A69459FE93FC6B9537ED4336E5099E0DCCA3E97290A412500ED7A0DAFFB03D80CF3650A20E0591F748E10C3C534945EE83D5F2C9722F1A68D98B8C01AF23FD
     */
     /*
    // Test 1: Unkeyed BLAKE2b-512 of empty input.
    // Expected: 786a02f742015903c6c6fd852552d272912f4740e15847618a86e217f71f5419
    //           d25e1031afee585313896444934eb04b903a685b1448b755d56f701afe9be2ce
    let digest = crypto::blake2b_512(b"");
    hardware_manager::sprint("BLAKE2b-512(\"\"):\n  ");
    hardware_manager::sprint(&helper_functions::bytes_to_hex(&digest));
    hardware_manager::sprint("\n");

    // Test 2: Unkeyed BLAKE2b-512 of "abc".
    // Expected: ba80a53f981c4d0d6a2797b69f12f6e94c212f14685ac4b74b12bb6fdbffa2d1
    //           7d87c5392aab792dc252d5de4533cc9518d38aa8dbf1925ab92386edd4009923
    let digest = crypto::blake2b_512(b"abc");
    hardware_manager::sprint("BLAKE2b-512(\"abc\"):\n  ");
    hardware_manager::sprint(&helper_functions::bytes_to_hex(&digest));
    hardware_manager::sprint("\n");

    // Test 3: Unkeyed BLAKE2b-256 of "abc".
    // Expected: bddd813c634239723171ef3fee98579b94964e3bb1cb3e427262c8c068d52319
    let digest = crypto::blake2b_256(b"abc");
    hardware_manager::sprint("BLAKE2b-256(\"abc\"):\n  ");
    hardware_manager::sprint(&helper_functions::bytes_to_hex(&digest));
    hardware_manager::sprint("\n");

    // Test 4: Variable output length — BLAKE2b-20 (5 bytes) of "abc".
    // A non-standard length to verify the truncation path works correctly.
    // Expected: 3345524a bf6bbe18 0c96b8ab (first 5 bytes of the 64-byte state)
    let digest = crypto::blake2b_variable(b"abc", 5);
    hardware_manager::sprint("BLAKE2b-20(\"abc\"):\n  ");
    hardware_manager::sprint(&helper_functions::bytes_to_hex(&digest));
    hardware_manager::sprint("\n");

    // Test 5: Keyed BLAKE2b-512 MAC with a 64-byte key of 0x00..0x3f
    // and input 0x00..0xbf. This is the RFC 7693 Appendix C keyed test vector.
    // Expected: 142709d62e28fcccd0af97fad0f8465b971e82201dc51070faa0372aa43e9248
    //           4be1c1e73ba10906d5d1853db6a4106e0a7bf9800d373d6dee2d46d62ef2a461
    let key: [u8; 64] = core::array::from_fn(|i| i as u8);
    let msg: [u8; 192] = core::array::from_fn(|i| i as u8);
    let mac = crypto::blake2b_mac_512(&msg, &key);
    hardware_manager::sprint("BLAKE2b-512 MAC (RFC 7693 Appendix C vector):\n  ");
    match mac {
        Some(ref m) => hardware_manager::sprint(&helper_functions::bytes_to_hex(m)),
        None        => hardware_manager::sprint("None (unexpected!)"),
    }
    hardware_manager::sprint("\n");

    // Test 6: Keyed BLAKE2b-256 MAC.
    // Same key and message as Test 5, truncated to 32 bytes.
    let mac = crypto::blake2b_mac_256(&msg, &key);
    hardware_manager::sprint("BLAKE2b-256 MAC:\n  ");
    match mac {
        Some(ref m) => hardware_manager::sprint(&helper_functions::bytes_to_hex(m)),
        None        => hardware_manager::sprint("None (unexpected!)"),
    }
    hardware_manager::sprint("\n");

    // Test 7: Full blake2b() entry point — variable length and key together.
    // 48-byte output, keyed, input "Hello".
    let mac = crypto::blake2b(
        b"Hello",
        48,
        Some(b"secret key"),
    );
    hardware_manager::sprint("BLAKE2b-384(\"Hello\", key=\"secret key\"):\n  ");
    match mac {
        Some(ref m) => hardware_manager::sprint(&helper_functions::bytes_to_hex(m)),
        None        => hardware_manager::sprint("None (unexpected!)"),
    }
    hardware_manager::sprint("\n");

    // Test 8: Edge case — invalid parameters should return None.
    let bad_out_len = crypto::blake2b(b"Hello", 0,    None);
    let bad_key_len = crypto::blake2b(b"Hello", 32,   Some(b""));
    let out_too_big = crypto::blake2b(b"Hello", 65,   None);
    hardware_manager::sprint("blake2b(out_len=0)  -> ");
    hardware_manager::sprint(if bad_out_len.is_none() { "None (correct)\n" } else { "Some (WRONG!)\n" });
    hardware_manager::sprint("blake2b(key=empty) -> ");
    hardware_manager::sprint(if bad_key_len.is_none() { "None (correct)\n" } else { "Some (WRONG!)\n" });
    hardware_manager::sprint("blake2b(out_len=65) -> ");
    hardware_manager::sprint(if out_too_big.is_none() { "None (correct)\n" } else { "Some (WRONG!)\n" });

    // Test 9: Long input spanning multiple 128-byte compression blocks.
    // Verifies the update() loop correctly handles block boundaries.
    let long_input = [0x61u8; 1000]; // 1000 × 'a'
    let digest = crypto::blake2b_512(&long_input);
    hardware_manager::sprint("BLAKE2b-512(\"a\" * 1000):\n  ");
    hardware_manager::sprint(&helper_functions::bytes_to_hex(&digest));
    hardware_manager::sprint("\n");
    */

    // RFC 7693 §A.1 — Unkeyed BLAKE2b-512 of the empty input.
    /*let digest = crypto::blake2b_512(b"");
    let expected: [u8; 64] = [
        0x78, 0x6a, 0x02, 0xf7, 0x42, 0x01, 0x59, 0x03,
        0xc6, 0xc6, 0xfd, 0x85, 0x25, 0x52, 0xd2, 0x72,
        0x91, 0x2f, 0x47, 0x40, 0xe1, 0x58, 0x47, 0x61,
        0x8a, 0x86, 0xe2, 0x17, 0xf7, 0x1f, 0x54, 0x19,
        0xd2, 0x5e, 0x10, 0x31, 0xaf, 0xee, 0x58, 0x53,
        0x13, 0x89, 0x64, 0x44, 0x93, 0x4e, 0xb0, 0x4b,
        0x90, 0x3a, 0x68, 0x5b, 0x14, 0x48, 0xb7, 0x55,
        0xd5, 0x6f, 0x70, 0x1a, 0xfe, 0x9b, 0xe2, 0xce,
    ];
    assert_eq!(digest, expected,
        "RFC 7693 empty-input BLAKE2b-512 mismatch");*/
    
    // RFC 7693 §A.1 — Unkeyed BLAKE2b-512 of b"abc".
    /*let digest = crypto::blake2b_512(b"abc");
    let expected: [u8; 64] = [
        0xba, 0x80, 0xa5, 0x3f, 0x98, 0x1c, 0x4d, 0x0d,
        0x6a, 0x27, 0x97, 0xb6, 0x9f, 0x12, 0xf6, 0xe9,
        0x4c, 0x21, 0x2f, 0x14, 0x68, 0x5a, 0xc4, 0xb7,
        0x4b, 0x12, 0xbb, 0x6f, 0xdb, 0xff, 0xa2, 0xd1,
        0x7d, 0x87, 0xc5, 0x39, 0x2a, 0xab, 0x79, 0x2d,
        0xc2, 0x52, 0xd5, 0xde, 0x45, 0x33, 0xcc, 0x95,
        0x18, 0xd3, 0x8a, 0xa8, 0xdb, 0xf1, 0x92, 0x5a,
        0xb9, 0x23, 0x86, 0xed, 0xd4, 0x00, 0x99, 0x23,
    ];
    assert_eq!(digest, expected,
        "RFC 7693 'abc' BLAKE2b-512 mismatch");*/
    
    // TODO: FAIL 1
    // Keyed BLAKE2b-512 from RFC 7693 Appendix A.
    // Key: 64 bytes, values 0x00 through 0x3f
    /*let key: alloc::vec::Vec<u8> = (0u8..64).collect();
    let input: alloc::vec::Vec<u8> = (0u8..250).collect();
    let result = crypto::blake2b(&input, 64, Some(&key)).expect("blake2b should not return None");
    // RFC 7693 Table 2, row for input length 250
    let expected: [u8; 64] = [
        0x6e, 0xce, 0x5e, 0xce, 0x92, 0x2d, 0x60, 0x1e,
        0xe7, 0x72, 0x00, 0xcf, 0xa6, 0xde, 0x36, 0x11,
        0x28, 0x75, 0x20, 0x19, 0x08, 0x77, 0x09, 0x3e,
        0x3d, 0x3b, 0x04, 0x01, 0x31, 0x07, 0x23, 0x84,
        0x23, 0xfe, 0x76, 0xe2, 0x25, 0xa8, 0x8d, 0x0d,
        0x43, 0xdc, 0x4d, 0x44, 0x36, 0x42, 0x16, 0x9a,
        0x52, 0x40, 0x47, 0xe5, 0x0b, 0x2d, 0xed, 0x55,
        0x51, 0xf5, 0x20, 0xf2, 0x90, 0x25, 0xfb, 0x78,
    ];
    assert_eq!(result.as_slice(), &expected,
        "RFC 7693 keyed 250-byte BLAKE2b-512 mismatch");*/
    
    // TODO: unknown vec! macro
    // RFC 7693 keyed vector — input length 1 (single byte 0x00).
    /*let key: alloc::vec::Vec<u8> = (0u8..64).collect();
    let input: alloc::vec::Vec<u8> = alloc::vec::vec![0x00];
    let result = crypto::blake2b(&input, 64, Some(&key)).expect("blake2b returned None");
    // RFC 7693 Table 2, row for input length 1
    let expected: [u8; 64] = [
        0x33, 0xd0, 0x82, 0x5d, 0xdd, 0xf8, 0xe6, 0x07,
        0x51, 0x58, 0x55, 0x0b, 0x50, 0x35, 0x55, 0x21,
        0x8a, 0xc8, 0x30, 0xc9, 0xbc, 0xb8, 0x44, 0x19,
        0xf1, 0x80, 0x49, 0xde, 0x54, 0x94, 0x04, 0xf4,
        0xfa, 0xef, 0xd7, 0xd9, 0x71, 0x63, 0x8e, 0x77,
        0x37, 0x9c, 0x4f, 0xda, 0x53, 0x85, 0x40, 0x39,
        0x2d, 0x8a, 0x1b, 0x93, 0x01, 0xde, 0x9d, 0xb2,
        0xa3, 0x33, 0x13, 0x82, 0x6c, 0xf1, 0xcd, 0x41,
    ];
    assert_eq!(result.as_slice(), &expected,
        "RFC 7693 keyed 1-byte BLAKE2b-512 mismatch");*/
    
    // TODO: FAIL test
    // RFC 7693 keyed vector — input length 128 (exactly one full block).
    /*let key: alloc::vec::Vec<u8> = (0u8..64).collect();
    let input: alloc::vec::Vec<u8> = (0u8..128).collect();
    let result = crypto::blake2b(&input, 64, Some(&key)).expect("blake2b returned None");
    // RFC 7693 Table 2, row for input length 128
    let expected: [u8; 64] = [
        0x01, 0x4a, 0x95, 0xb9, 0x04, 0xdd, 0x21, 0xe5,
        0x0d, 0xba, 0x26, 0x58, 0x91, 0x18, 0x76, 0xc6,
        0x37, 0xfe, 0x30, 0x26, 0xf8, 0x7f, 0xb5, 0x52,
        0x13, 0x97, 0x35, 0xef, 0xfc, 0x7a, 0xa7, 0x53,
        0x05, 0xa2, 0x68, 0x07, 0xc0, 0xf5, 0x5a, 0x21,
        0xfc, 0xc2, 0x64, 0x2f, 0x12, 0xa5, 0xb1, 0x70,
        0x9e, 0x21, 0x4a, 0xe6, 0x35, 0xb8, 0x4b, 0x64,
        0x98, 0xbe, 0x48, 0x33, 0x91, 0x37, 0x22, 0x42,
    ];
    assert_eq!(result.as_slice(), &expected,
        "RFC 7693 keyed 128-byte BLAKE2b-512 mismatch");*/
    
    // TODO: FAIL test
    // RFC 7693 keyed vector — input length 255 (near-maximum test).
    /*let key: alloc::vec::Vec<u8> = (0u8..64).collect();
    let input: alloc::vec::Vec<u8> = (0u8..=254).collect(); // 255 bytes
    let result = crypto::blake2b(&input, 64, Some(&key)).expect("blake2b returned None");
    // RFC 7693 Table 2, row for input length 255
    let expected: [u8; 64] = [
        0x44, 0x35, 0x90, 0x51, 0x04, 0x94, 0x4c, 0x20,
        0x3f, 0x2e, 0x2d, 0x42, 0xab, 0x38, 0xa1, 0x49,
        0x7e, 0xcb, 0x48, 0x15, 0xad, 0xe7, 0x3f, 0x6e,
        0xae, 0xa2, 0x5d, 0x34, 0xa4, 0x2c, 0x37, 0x90,
        0x05, 0x32, 0xec, 0xef, 0xa3, 0x4e, 0x1e, 0x91,
        0x56, 0xf0, 0x30, 0x9e, 0x38, 0x3d, 0xd9, 0x82,
        0x6f, 0x8b, 0x3e, 0x6e, 0x08, 0xc0, 0x2e, 0x73,
        0x45, 0x64, 0xca, 0x12, 0x46, 0xa3, 0x69, 0xa5,
    ];
    assert_eq!(result.as_slice(), &expected,
        "RFC 7693 keyed 255-byte BLAKE2b-512 mismatch");*/
    
    // Unkeyed BLAKE2b-256 of empty input.
    let digest = crypto::blake2b_256(b"");
    let expected: [u8; 32] = [
        0x0e, 0x57, 0x51, 0xc0, 0x26, 0xe5, 0x43, 0xb2,
        0xe8, 0xab, 0x2e, 0xb0, 0x60, 0x99, 0xda, 0xa1,
        0xd1, 0xe5, 0xdf, 0x47, 0x77, 0x8f, 0x77, 0x87,
        0xfa, 0xab, 0x45, 0xcd, 0xf1, 0x2f, 0xe3, 0xa8,
    ];
    assert_eq!(digest, expected, "BLAKE2b-256 empty-input mismatch");

    // TODO: FAILED test (unkeyed)
    // Unkeyed BLAKE2b-256 of b"abc".
    let digest = crypto::blake2b_256(b"abc");
    let expected: [u8; 32] = [
        0xbd, 0xdd, 0x81, 0x3c, 0x63, 0x42, 0x39, 0x72,
        0x31, 0x71, 0xef, 0x3f, 0xee, 0x98, 0x57, 0x9b,
        0x94, 0x96, 0x4e, 0x3b, 0xb1, 0xcb, 0x3e, 0x42,
        0x72, 0x62, 0xc8, 0xc0, 0x68, 0xd5, 0x23, 0x19,
    ];
    assert_eq!(digest, expected, "BLAKE2b-256 'abc' mismatch");

    // TODO: FAIL test
    /*// Keyed BLAKE2b-512 MAC of b"" with a 32-byte all-zero key.
    let key = [0u8; 32];
    let result = crypto::blake2b_mac_512(b"", &key).expect("MAC should not fail");
    // Computed from the BLAKE2 reference implementation.
    let expected: [u8; 64] = [
        0x78, 0x6a, 0x02, 0xf7, 0x42, 0x01, 0x59, 0x03,
        0xc6, 0xc6, 0xfd, 0x85, 0x25, 0x52, 0xd2, 0x72,
        0x91, 0x2f, 0x47, 0x40, 0xe1, 0x58, 0x47, 0x61,
        0x8a, 0x86, 0xe2, 0x17, 0xf7, 0x1f, 0x54, 0x19,
        0xd2, 0x5e, 0x10, 0x31, 0xaf, 0xee, 0x58, 0x53,
        0x13, 0x89, 0x64, 0x44, 0x93, 0x4e, 0xb0, 0x4b,
        0x90, 0x3a, 0x68, 0x5b, 0x14, 0x48, 0xb7, 0x55,
        0xd5, 0x6f, 0x70, 0x1a, 0xfe, 0x9b, 0xe2, 0xce,
    ];
    assert_eq!(result, expected, "blake2b_mac_512 zero-key empty-input mismatch");*/

    // `blake2b_mac_512` and `blake2b` with `Some(key)` must agree.
    /*let key = b"test-key-32-bytes-long-here!!!!";
    let input = b"The quick brown fox jumps over the lazy dog";
    let via_mac = crypto::blake2b_mac_512(input, key).expect("mac should succeed");
    let via_api = crypto::blake2b(input, 64, Some(key))
        .expect("blake2b should succeed");
    assert_eq!(&via_mac[..], via_api.as_slice(),
        "blake2b_mac_512 and blake2b(key=Some) disagree");*/
    
    // `blake2b_mac_256` and `blake2b` with `Some(key)` must agree.
    /*let key = b"another-key";
    let input = b"hello world";
    let via_mac = crypto::blake2b_mac_256(input, key).expect("mac should succeed");
    let via_api = crypto::blake2b(input, 32, Some(key))
        .expect("blake2b should succeed");
    assert_eq!(&via_mac[..], via_api.as_slice(),
        "blake2b_mac_256 and blake2b(key=Some) disagree");*/
    
    // Hashing the same bytes with different keys must give different outputs.
    /*let input = b"same input";
    let key1 = b"key one";
    let key2 = b"key two";
    let mac1 = crypto::blake2b_mac_512(input, key1).unwrap();
    let mac2 = crypto::blake2b_mac_512(input, key2).unwrap();
    assert_ne!(mac1, mac2, "Different keys produced the same MAC");*/

    // Keyed hash must differ from unkeyed hash of the same input
    /*let input = b"some data";
    let key = b"secret";
    let keyed   = crypto::blake2b(&input[..], 64, Some(key)).unwrap();
    let unkeyed = crypto::blake2b(&input[..], 64, None).unwrap();
    assert_ne!(keyed, unkeyed,
        "Keyed and unkeyed BLAKE2b produced the same output");*/
    
    // Different `out_len` values must produce different digests, *not* merely
    // the longer digest truncated.
    /*let input = b"domain separation test";
    let digest_32 = crypto::blake2b(input, 32, None).unwrap();
    let digest_64 = crypto::blake2b(input, 64, None).unwrap();
    // The 32-byte output must NOT equal the first 32 bytes of the 64-byte output.
    assert_ne!(digest_32.as_slice(), &digest_64[..32],
        "BLAKE2b outputs of different lengths should NOT be prefixes of each other");*/

    // `blake2b_512` and `blake2b_512_slice` are identical
    /*let input = b"consistency check input";
    let a = crypto::blake2b_512(input);
    let b = crypto::blake2b_512_slice(input);
    assert_eq!(a, b, "blake2b_512 and blake2b_512_slice disagree");*/

    // `blake2b_variable` with an in-range length must match `blake2b`.
    /*let input = b"variable length test";
    for out_len in [1usize, 16, 32, 48, 63, 64] {
        let via_var = crypto::blake2b_variable(input, out_len);
        let via_api = crypto::blake2b(input, out_len, None).unwrap();
        assert_eq!(via_var, via_api,
            "blake2b_variable({out_len}) disagrees with blake2b");
    }*/

    // One-liner tests
    /*let result = crypto::blake2b_variable(b"anything", 0);
        assert_eq!(result.len(), 1,
            "Expected clamped length 1, got {}", result.len());
    let result = crypto::blake2b_variable(b"anything", 65);
        assert_eq!(result.len(), 64,
            "Expected clamped length 64, got {}", result.len());
    assert!(crypto::blake2b(b"x", 0, None).is_none(),
            "Expected None for out_len=0");
    assert!(crypto::blake2b(b"x", 65, None).is_none(),
            "Expected None for out_len=65");
    assert!(crypto::blake2b(b"x", 32, Some(b"")).is_none(),
            "Expected None for empty key");
    let long_key = [0u8; 65];
        assert!(crypto::blake2b(b"x", 32, Some(&long_key)).is_none(),
            "Expected None for 65-byte key");
    assert!(crypto::blake2b_mac_512(b"data", b"").is_none(),
            "Expected None for empty key");
    let long_key = [0xffu8; 65];
        assert!(crypto::blake2b_mac_256(b"data", &long_key).is_none(),
            "Expected None for 65-byte key");*/
    
    // Input of exactly BLOCK_SIZE (128 bytes) must be handled correctly.
    /*let input = [0x42u8; 128];
    let digest = crypto::blake2b_512(&input);
    // Verify against blake2b() for consistency; also check length.
    let via_api = crypto::blake2b(&input, 64, None).unwrap();
    assert_eq!(&digest[..], via_api.as_slice());
    // Non-trivial: result must differ from the empty-input hash.
    let empty = crypto::blake2b_512(b"");
    assert_ne!(digest, empty);*/

    // Input of exactly BLOCK_SIZE + 1 (129 bytes) crosses a block boundary.
    /*let input = [0x37u8; 129];
    let digest = crypto::blake2b_512(&input);
    let via_api = crypto::blake2b(&input, 64, None).unwrap();
    assert_eq!(&digest[..], via_api.as_slice());*/

    // Input of exactly 2 * BLOCK_SIZE (256 bytes).
    /*let input = [0x55u8; 256];
    let a = crypto::blake2b_512(&input);
    let b = crypto::blake2b(&input, 64, None).unwrap();
    assert_eq!(&a[..], b.as_slice());*/

    // Calling `update` in many small chunks must give the same result as
    // one large call, since the streaming API is stateful.
    // Build a non-trivial 300-byte input.
    /*let input: alloc::vec::Vec<u8> = (0u8..=255).chain(0u8..44).collect();
    assert_eq!(input.len(), 300);
    let oneshot = crypto::blake2b(&input, 64, None).unwrap();*/

    // Manually drive the state in three chunks to exercise update looping.
    /*let mut state = crypto::Blake2bState::new(64, None);
    state.update(&input[..100]);
    state.update(&input[100..200]);
    state.update(&input[200..]);
    let chunked = state.finalize();
    assert_eq!(oneshot, chunked,
        "One-shot and chunked updates produced different digests");*/

    // Two identical calls must always produce identical output.
    /*let input = b"determinism test vector";
    let a = crypto::blake2b_512(input);
    let b = crypto::blake2b_512(input);
    assert_eq!(a, b);*/

    // A one-bit change in input must produce a completely different digest
    // (avalanche effect sanity check).
    /*let input_a = b"hello world";
    let mut input_b = *input_a;
    input_b[0] ^= 0x01; // flip one bit in 'h' → 'i'
    let digest_a = crypto::blake2b_512(input_a);
    let digest_b = crypto::blake2b_512(&input_b);
    assert_ne!(digest_a, digest_b,
        "One-bit change should produce a different digest");
    // Count differing bytes — roughly half should differ (birthday bound).
    let differing = digest_a.iter().zip(digest_b.iter())
        .filter(|(a, b)| a != b)
        .count();
    assert!(differing >= 24,
        "Too few bytes differ ({differing}/64) — possible avalanche failure");*/

    

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
