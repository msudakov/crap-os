#![no_std]   // This is an OS kernel; there is no standard library for now
#![no_main]  // Not depending on a runtime, so cannot use main as entry point


// Standard base address for COM1 serial port, used for serial port functions
const COM1_PORT: u16 = 0x3F8;

// Simple 8x8 bitmap font for ASCII characters 32-126, used for drawing text
const FONT: [[u8; 8]; 95] = [
    [0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00], // Space (32)
    [0x18, 0x3C, 0x3C, 0x18, 0x18, 0x00, 0x18, 0x00], // !
    [0x36, 0x36, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00], // "
    [0x36, 0x36, 0x7F, 0x36, 0x7F, 0x36, 0x36, 0x00], // #
    [0x0C, 0x3E, 0x03, 0x1E, 0x30, 0x1F, 0x0C, 0x00], // $
    [0x00, 0x63, 0x33, 0x18, 0x0C, 0x66, 0x63, 0x00], // %
    [0x1C, 0x36, 0x1C, 0x6E, 0x3B, 0x33, 0x6E, 0x00], // &
    [0x06, 0x06, 0x03, 0x00, 0x00, 0x00, 0x00, 0x00], // '
    [0x18, 0x0C, 0x06, 0x06, 0x06, 0x0C, 0x18, 0x00], // (
    [0x06, 0x0C, 0x18, 0x18, 0x18, 0x0C, 0x06, 0x00], // )
    [0x00, 0x66, 0x3C, 0xFF, 0x3C, 0x66, 0x00, 0x00], // *
    [0x00, 0x0C, 0x0C, 0x3F, 0x0C, 0x0C, 0x00, 0x00], // +
    [0x00, 0x00, 0x00, 0x00, 0x00, 0x0C, 0x0C, 0x06], // ,
    [0x00, 0x00, 0x00, 0x3F, 0x00, 0x00, 0x00, 0x00], // -
    [0x00, 0x00, 0x00, 0x00, 0x00, 0x0C, 0x0C, 0x00], // .
    [0x60, 0x30, 0x18, 0x0C, 0x06, 0x03, 0x01, 0x00], // /
    [0x3E, 0x63, 0x73, 0x7B, 0x6F, 0x67, 0x3E, 0x00], // 0
    [0x0C, 0x0E, 0x0C, 0x0C, 0x0C, 0x0C, 0x3F, 0x00], // 1
    [0x1E, 0x33, 0x30, 0x1C, 0x06, 0x33, 0x3F, 0x00], // 2
    [0x1E, 0x33, 0x30, 0x1C, 0x30, 0x33, 0x1E, 0x00], // 3
    [0x38, 0x3C, 0x36, 0x33, 0x7F, 0x30, 0x78, 0x00], // 4
    [0x3F, 0x03, 0x1F, 0x30, 0x30, 0x33, 0x1E, 0x00], // 5
    [0x1C, 0x06, 0x03, 0x1F, 0x33, 0x33, 0x1E, 0x00], // 6
    [0x3F, 0x33, 0x30, 0x18, 0x0C, 0x0C, 0x0C, 0x00], // 7
    [0x1E, 0x33, 0x33, 0x1E, 0x33, 0x33, 0x1E, 0x00], // 8
    [0x1E, 0x33, 0x33, 0x3E, 0x30, 0x18, 0x0E, 0x00], // 9
    [0x00, 0x0C, 0x0C, 0x00, 0x00, 0x0C, 0x0C, 0x00], // :
    [0x00, 0x0C, 0x0C, 0x00, 0x00, 0x0C, 0x0C, 0x06], // ;
    [0x18, 0x0C, 0x06, 0x03, 0x06, 0x0C, 0x18, 0x00], // <
    [0x00, 0x00, 0x3F, 0x00, 0x00, 0x3F, 0x00, 0x00], // =
    [0x06, 0x0C, 0x18, 0x30, 0x18, 0x0C, 0x06, 0x00], // >
    [0x1E, 0x33, 0x30, 0x18, 0x0C, 0x00, 0x0C, 0x00], // ?
    [0x3E, 0x63, 0x7B, 0x7B, 0x7B, 0x03, 0x1E, 0x00], // @
    [0x0C, 0x1E, 0x33, 0x33, 0x3F, 0x33, 0x33, 0x00], // A
    [0x3F, 0x66, 0x66, 0x3E, 0x66, 0x66, 0x3F, 0x00], // B
    [0x3C, 0x66, 0x03, 0x03, 0x03, 0x66, 0x3C, 0x00], // C
    [0x1F, 0x36, 0x66, 0x66, 0x66, 0x36, 0x1F, 0x00], // D
    [0x7F, 0x46, 0x16, 0x1E, 0x16, 0x46, 0x7F, 0x00], // E
    [0x7F, 0x46, 0x16, 0x1E, 0x16, 0x06, 0x0F, 0x00], // F
    [0x3C, 0x66, 0x03, 0x03, 0x73, 0x66, 0x7C, 0x00], // G
    [0x33, 0x33, 0x33, 0x3F, 0x33, 0x33, 0x33, 0x00], // H
    [0x1E, 0x0C, 0x0C, 0x0C, 0x0C, 0x0C, 0x1E, 0x00], // I
    [0x78, 0x30, 0x30, 0x30, 0x33, 0x33, 0x1E, 0x00], // J
    [0x67, 0x66, 0x36, 0x1E, 0x36, 0x66, 0x67, 0x00], // K
    [0x0F, 0x06, 0x06, 0x06, 0x46, 0x66, 0x7F, 0x00], // L
    [0x63, 0x77, 0x7F, 0x7F, 0x6B, 0x63, 0x63, 0x00], // M
    [0x63, 0x67, 0x6F, 0x7B, 0x73, 0x63, 0x63, 0x00], // N
    [0x1C, 0x36, 0x63, 0x63, 0x63, 0x36, 0x1C, 0x00], // O
    [0x3F, 0x66, 0x66, 0x3E, 0x06, 0x06, 0x0F, 0x00], // P
    [0x1E, 0x33, 0x33, 0x33, 0x3B, 0x1E, 0x38, 0x00], // Q
    [0x3F, 0x66, 0x66, 0x3E, 0x36, 0x66, 0x67, 0x00], // R
    [0x1E, 0x33, 0x07, 0x0E, 0x38, 0x33, 0x1E, 0x00], // S
    [0x3F, 0x2D, 0x0C, 0x0C, 0x0C, 0x0C, 0x1E, 0x00], // T
    [0x33, 0x33, 0x33, 0x33, 0x33, 0x33, 0x3F, 0x00], // U
    [0x33, 0x33, 0x33, 0x33, 0x33, 0x1E, 0x0C, 0x00], // V
    [0x63, 0x63, 0x63, 0x6B, 0x7F, 0x77, 0x63, 0x00], // W
    [0x63, 0x63, 0x36, 0x1C, 0x1C, 0x36, 0x63, 0x00], // X
    [0x33, 0x33, 0x33, 0x1E, 0x0C, 0x0C, 0x1E, 0x00], // Y
    [0x7F, 0x63, 0x31, 0x18, 0x4C, 0x66, 0x7F, 0x00], // Z
    [0x1E, 0x06, 0x06, 0x06, 0x06, 0x06, 0x1E, 0x00], // [
    [0x03, 0x06, 0x0C, 0x18, 0x30, 0x60, 0x40, 0x00], // \
    [0x1E, 0x18, 0x18, 0x18, 0x18, 0x18, 0x1E, 0x00], // ]
    [0x08, 0x1C, 0x36, 0x63, 0x00, 0x00, 0x00, 0x00], // ^
    [0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xFF], // _
    [0x0C, 0x0C, 0x18, 0x00, 0x00, 0x00, 0x00, 0x00], // `
    [0x00, 0x00, 0x1E, 0x30, 0x3E, 0x33, 0x6E, 0x00], // a
    [0x07, 0x06, 0x06, 0x3E, 0x66, 0x66, 0x3B, 0x00], // b
    [0x00, 0x00, 0x1E, 0x33, 0x03, 0x33, 0x1E, 0x00], // c
    [0x38, 0x30, 0x30, 0x3e, 0x33, 0x33, 0x6E, 0x00], // d
    [0x00, 0x00, 0x1E, 0x33, 0x3f, 0x03, 0x1E, 0x00], // e
    [0x1C, 0x36, 0x06, 0x0f, 0x06, 0x06, 0x0F, 0x00], // f
    [0x00, 0x00, 0x6E, 0x33, 0x33, 0x3E, 0x30, 0x1F], // g
    [0x07, 0x06, 0x36, 0x6E, 0x66, 0x66, 0x67, 0x00], // h
    [0x0C, 0x00, 0x0E, 0x0C, 0x0C, 0x0C, 0x1E, 0x00], // i
    [0x30, 0x00, 0x30, 0x30, 0x30, 0x33, 0x33, 0x1E], // j
    [0x07, 0x06, 0x66, 0x36, 0x1E, 0x36, 0x67, 0x00], // k
    [0x0E, 0x0C, 0x0C, 0x0C, 0x0C, 0x0C, 0x1E, 0x00], // l
    [0x00, 0x00, 0x33, 0x7F, 0x7F, 0x6B, 0x63, 0x00], // m
    [0x00, 0x00, 0x1F, 0x33, 0x33, 0x33, 0x33, 0x00], // n
    [0x00, 0x00, 0x1E, 0x33, 0x33, 0x33, 0x1E, 0x00], // o
    [0x00, 0x00, 0x3B, 0x66, 0x66, 0x3E, 0x06, 0x0F], // p
    [0x00, 0x00, 0x6E, 0x33, 0x33, 0x3E, 0x30, 0x78], // q
    [0x00, 0x00, 0x3B, 0x6E, 0x66, 0x06, 0x0F, 0x00], // r
    [0x00, 0x00, 0x3E, 0x03, 0x1E, 0x30, 0x1F, 0x00], // s
    [0x08, 0x0C, 0x3E, 0x0C, 0x0C, 0x2C, 0x18, 0x00], // t
    [0x00, 0x00, 0x33, 0x33, 0x33, 0x33, 0x6E, 0x00], // u
    [0x00, 0x00, 0x33, 0x33, 0x33, 0x1E, 0x0C, 0x00], // v
    [0x00, 0x00, 0x63, 0x6B, 0x7F, 0x7F, 0x36, 0x00], // w
    [0x00, 0x00, 0x63, 0x36, 0x1C, 0x36, 0x63, 0x00], // x
    [0x00, 0x00, 0x33, 0x33, 0x33, 0x3E, 0x30, 0x1F], // y
    [0x00, 0x00, 0x3F, 0x19, 0x0C, 0x26, 0x3F, 0x00], // z
    [0x38, 0x0C, 0x0C, 0x07, 0x0C, 0x0C, 0x38, 0x00], // {
    [0x18, 0x18, 0x18, 0x00, 0x18, 0x18, 0x18, 0x00], // |
    [0x07, 0x0C, 0x0C, 0x38, 0x0C, 0x0C, 0x07, 0x00], // }
    [0x6E, 0x3B, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00], // ~
];

// Preset level of debugging messages sent by the kernel
#[repr(i32)]
#[derive(PartialEq, PartialOrd)]
enum DebugLevel {
    DEBUG = 1,
    INFO = 2,
    WARNING = 3,
    ERROR = 4,
    CRITICAL = 5
}
const DEBUG_LEVEL: DebugLevel = DebugLevel::INFO;

/*
    This is the BootInfo structure that is passed to the _start routine by the
    bootloader when EntryPoint is called and execution is transferred to the
    kernel. This must match the structure in the C bootloader exactly.
*/
#[repr(C)]
pub struct BootInfo {
    framebuffer_addr: u64,
    framebuffer_width: u32,
    framebuffer_height: u32,
    framebuffer_pitch: u32,
    framebuffer_bpp: u32,
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

        init_serial();  // Initialize serial port for debugging
        print_debug(DebugLevel::WARNING, b"[WARN] Kernel started\n");

        if boot_info.is_null() {  // Validate boot_info pointer
            print_debug(DebugLevel::CRITICAL, b"[ERROR] boot_info is null\n");
            loop { core::arch::asm!("hlt"); }
        }
        let info = &*boot_info;  // Dereference boot_info
        
        print_debug(DebugLevel::DEBUG, b"[DEBUG] Got BootInfo structure\n");
        
        // Get frame buffer
        let fb_addr = info.framebuffer_addr;
        let fb_width = info.framebuffer_width;
        let fb_height = info.framebuffer_height;

        print_debug(DebugLevel::DEBUG, b"[DEBUG] Framebuffer info read\n");
        
        if fb_addr == 0 {  // Validate framebuffer address
            print_debug(DebugLevel::ERROR,
                b"[ERROR] framebuffer address is 0\n");
            loop { core::arch::asm!("hlt"); }
        }
        print_debug(DebugLevel::INFO, b"[INFO] Validated framebuffer addr\n");
        
        // Cast frame buffer address value as raw pointer
        let framebuffer = fb_addr as *mut u32;
        
        // Clear screen to black
        let total_pixels = fb_width * fb_height;
        for i in 0..total_pixels {
            //framebuffer.offset(i as isize).write_volatile(0x00000033);
            framebuffer.offset(i as isize).write_volatile(0x00000000);
        }
        print_debug(DebugLevel::DEBUG, b"[DEBUG] Screen cleared\n");

        /*
        // Draw a white rectangle in the center
        let rect_x = fb_width / 2 - 100;
        let rect_y = fb_height / 2 - 50;
        let rect_w = 200;
        let rect_h = 100;
        for y in rect_y..(rect_y + rect_h) {
            for x in rect_x..(rect_x + rect_w) {
                let offset = (y * fb_width + x) as isize;
                framebuffer.offset(offset).write_volatile(0x00FFFFFF);
            }
        }
        */
        
        // Draw text and banner on the screen
        draw_string(framebuffer, fb_width, 10, 10, b"HELLO FROM", 0x004AF262,
            0x00000000);
        draw_banner(framebuffer, fb_width, 10, 40);

        print_debug(DebugLevel::DEBUG, b"[DEBUG] Text drawn\n");
        print_debug(DebugLevel::INFO,
            b"[INFO] Graphics initialized successfully\n");
        
        // Done for now.. loop forever and ever
        loop {
            //core::arch::asm!("hlt");
        }
    }
}

/// Draws a single character into a frame buffer.
///
/// Draws a character at position (x, y) with given foreground and background
/// colors.
/// 
/// # Arguments
///
/// * `framebuffer` - Pointer to frame buffer to draw in.
/// * `fb_width` - Width of the specified frame buffer.
/// * `pos_x` - X coordinate of position to draw at.
/// * `pos_y` - Y coordinate of position to draw at.
/// * `char` - Character to draw.
/// * `fg_color` - Character foreground color.
/// * `bg_color` - Character background color.
///
/// # Safety
/// 
/// Performs a `write_volatile` on the frame buffer memory location.
fn draw_char(
    framebuffer: *mut u32,
    fb_width: u32,
    pos_x: u32,
    pos_y: u32,
    char: u8,
    fg_color: u32,
    bg_color: u32,
) {
    // Only handle standard printable ASCII characters (32-126)
    if char < 32 || char > 126 {
        return;
    }

    // Subtract the lowest character value of 32 to get its index in the array
    let font_index = (char - 32) as usize;
    let glyph = &FONT[font_index];

    // Loop through each of the 8 bytes of the bitmap index
    for row in 0..8 {
        let byte = glyph[row];
        for col in 0..8 {
            let pixel_x = pos_x + col;
            let pixel_y = pos_y + (row as u32);  // Used as usize above
            
            // Determine whether to draw the pixel in bg or fg color
            let color = if byte & (1 << col) != 0 {
                fg_color
            }
            else {
                bg_color
            };

            // Determine the absolute offset to draw the pixel at
            let offset = (pixel_y * fb_width + pixel_x) as isize;

            // Draw the pixel
            unsafe {
                framebuffer.offset(offset).write_volatile(color);
            }
        }
    }
}

/// Draws a string into a frame buffer.
///
/// Draws a string at position (x, y) with given foreground and background
/// colors.
/// 
/// # Arguments
///
/// * `framebuffer` - Pointer to frame buffer to draw in.
/// * `fb_width` - Width of the specified frame buffer.
/// * `pos_x` - X coordinate of position to draw at.
/// * `pos_y` - Y coordinate of position to draw at.
/// * `str` - String text to draw.
/// * `fg_color` - String foreground color.
/// * `bg_color` - String background color.
fn draw_string(
    framebuffer: *mut u32,
    fb_width: u32,
    pos_x: u32,
    pos_y: u32,
    str: &[u8],
    fg_color: u32,
    bg_color: u32,
) {
    for (i, &char) in str.iter().enumerate() {
        draw_char(framebuffer, fb_width, pos_x + (i as u32 * 8), pos_y, char,
            fg_color, bg_color);
    }
}

/// Draws an ASCII banner of the OS.
/// 
/// # Arguments
///
/// * `fb` - Pointer to frame buffer to draw in.
/// * `fbw` - Width of the specified frame buffer.
/// * `x` - Starting X coordinate of position to draw at.
/// * `y` - Starting Y coordinate of position to draw at.
fn draw_banner(
    fb: *mut u32,
    fbw: u32,
    x: u32,
    y: u32,
) {
    let fg: u32 = 0x004AF262;
    let bg: u32 = 0x00000000;

    draw_string(fb, fbw, x, y + 000, b"+------------------------------------------------+", fg, bg);
    draw_string(fb, fbw, x, y + 010, b"|                                                |", fg, bg);
    draw_string(fb, fbw, x, y + 020, b"|   #####                       #######  #####   |", fg, bg);
    draw_string(fb, fbw, x, y + 030, b"|  #     # #####    ##   #####  #     # #     #  |", fg, bg);
    draw_string(fb, fbw, x, y + 040, b"|  #       #    #  #  #  #    # #     # #        |", fg, bg);
    draw_string(fb, fbw, x, y + 050, b"|  #       #    # #    # #    # #     #  #####   |", fg, bg);
    draw_string(fb, fbw, x, y + 060, b"|  #       #####  ###### #####  #     #       #  |", fg, bg);
    draw_string(fb, fbw, x, y + 070, b"|  #     # #   #  #    # #      #     # #     #  |", fg, bg);
    draw_string(fb, fbw, x, y + 080, b"|   #####  #    # #    # #      #######  #####   |", fg, bg);
    draw_string(fb, fbw, x, y + 090, b"|                                                |", fg, bg);
    draw_string(fb, fbw, x, y + 100, b"+------------------------------------------------+", fg, bg);
}

/// Reads one byte of data from a specified I/O port address.
///
/// # Arguments
///
/// * `port` - Serial port base address.
///
/// # Returns
///
/// One byte value received from the specified serial port.
/// 
/// # Safety
/// 
/// Executes inline assembly.
#[inline(always)]
fn inb(port: u16) -> u8 {
    let value: u8;
    unsafe {
        core::arch::asm!(
            "in al, dx",
            out("al") value,
            in("dx") port,
            options(nomem, nostack, preserves_flags)
        );
    }
    value
}

/// Writes one byte of data to a specified I/O port address.
///
/// # Arguments
///
/// * `port` - Serial port base address.
/// * `value` - Byte value to write to serial port.
/// 
/// # Safety
/// 
/// Executes inline assembly.
#[inline(always)]
unsafe fn outb(port: u16, value: u8) {
    unsafe {
        core::arch::asm!(
            "out dx, al",
            in("dx") port,
            in("al") value,
            options(nomem, nostack, preserves_flags)
        );
    }
}

/// Initializes serial port for debugging purposes.
/// 
/// # Safety
/// 
/// Calls inline functions for direct assembly execution.
fn init_serial() {
    unsafe {
        // Disable interrupts on COM1
        outb(COM1_PORT + 1, 0x00);
        
        // Enable DLAB (set baud rate divisor)
        outb(COM1_PORT + 3, 0x80);
        
        // Set divisor to 3 (38400 baud)
        outb(COM1_PORT + 0, 0x03);
        outb(COM1_PORT + 1, 0x00);
        
        // 8 bits, no parity, one stop bit
        outb(COM1_PORT + 3, 0x03);
        
        // Enable FIFO, clear them, with 14-byte threshold
        outb(COM1_PORT + 2, 0xC7);
        
        // IRQs enabled, RTS/DSR set
        outb(COM1_PORT + 4, 0x0B);
    }
}

/// Writes a given byte string to serial port.
///
/// # Arguments
///
/// * `str` - Byte string to write.
/// 
/// # Safety
/// 
/// Calls inline functions for direct assembly execution.
fn serial_write(str: &[u8]) {
    unsafe {
        for &byte in str {
            // Wait for transmit buffer to be empty
            while (inb(COM1_PORT + 5) & 0x20) == 0 {}
            
            // Send the byte
            outb(COM1_PORT, byte);
        }
    }
}

fn print_debug(debug_level: DebugLevel, str: &[u8]) {
    if debug_level < DEBUG_LEVEL {
        return;
    }

    serial_write(str);
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
    serial_write(b"\n!!! KERNEL PANIC !!!\n");
    
    if let Some(location) = info.location() {
        serial_write(b"Location: ");
        serial_write(location.file().as_bytes());
        serial_write(b"\n");
    }

    loop {
        // Halt the CPU
        unsafe { core::arch::asm!("hlt"); }
    }
}
