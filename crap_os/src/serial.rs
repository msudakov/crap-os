// CrapOS Serial Printer Module
//
// This module contains the needed functionality to write messages to a serial
// port, mainly used for development and debugging.

// Standard base address for COM1 serial port, used for serial port functions
const COM1_PORT: u16 = 0x3F8;




//pub static WRITER: Writer = Writer {
    //column_position: 0,
    //color_code: ColorCode::new(Color::Yellow, Color::Black),
    //buffer: unsafe { &mut *(0xb8000 as *mut Buffer) },
//};

//pub struct SerialWriter {
//    pub port: u16,
//}




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
pub fn init_serial() {
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
#[allow(dead_code)]
pub fn serial_write(str: &[u8]) {
    unsafe {
        for &byte in str {
            // Wait for transmit buffer to be empty
            while (inb(COM1_PORT + 5) & 0x20) == 0 {}
            
            // Send the byte
            outb(COM1_PORT, byte);
        }
    }
}

/// Prints out a string slice inline with previous output, without a newline
/// character.
/// 
/// # Arguments
///
/// * `str` - String slice to print.
#[allow(dead_code)]
pub fn print(str: &str) {
    serial_write(str.as_bytes());
}

/// Prints out a string slice inline with previous output, followed by a
/// newline character.
/// 
/// # Arguments
///
/// * `str` - String slice to print.
#[allow(dead_code)]
pub fn println(str: &str) {
    print(str);
    print("\n");
}

/// Prints a debug message to serial port.
/// 
/// The message is only printed if its given level is equal to or higher than
/// the hardcoded global variable for overall debugging level.
///
/// # Arguments
///
/// * `debug_level` - Specified debug level of the given message.
/// * `message` - Debug message to print.
#[allow(dead_code)]
pub fn print_debug(debug_level: crate::DebugLevel, message: &str) {
    if debug_level < crate::DEBUG_LEVEL {
        return;
    }

    println(message);
}

/// Handles the printing of byte slices or arrays passed as slices.
/// 
/// # Arguments
///
/// * `bytes` - Byte slice to print as string.
/// 
/// # Safety
/// 
/// The bytes passed to this function must all be printable characters.
#[allow(dead_code)]
pub fn print_bytes(bytes: &[u8]) {
    let byte_string: &str = unsafe {
        core::str::from_utf8_unchecked(bytes)
    };
    print(byte_string);
}

/// Helper function to print an address value or another u64 value with a label.
/// 
/// # Arguments
///
/// * `label`   - String label to print before the value.
/// * `address` - Memory address or another u64 value to print.
#[allow(dead_code)]
pub fn print_addr_with_label(label: &str, address: u64) {
    let addr_hex_str = crate::system_routines::u64_to_hex_bytes(address);
    print(label);
    print(": ");
    print_bytes(&addr_hex_str);
    println("");
}
