//! Serial Printer Module
//! 
//! This module contains the needed functionality to write messages to a serial
//! port, mainly used for development and debugging.

pub struct SerialWriter {
    pub port: u16,
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
fn outb(port: u16, value: u8) {
    unsafe {
        core::arch::asm!(
            "out dx, al",
            in("dx") port,
            in("al") value,
            options(nomem, nostack, preserves_flags)
        );
    }
}

/// Initialize a serial port for transmission.
///
/// # Arguments
///
/// * `port` - Serial port base address.
pub fn init(port: u16) {
    outb(port + 1, 0x00);  // Disable interrupts on COM1
    outb(port + 3, 0x80);  // Enable DLAB (set baud rate divisor)
    outb(port + 0, 0x03);  // Set divisor to 3 (38400 baud)
    outb(port + 1, 0x00);  // Set divisor to 3 (38400 baud)
    outb(port + 3, 0x03);  // 8 bits, no parity, one stop bit
    outb(port + 2, 0xC7);  // Enable FIFO and clear with 14-byte threshold
    outb(port + 4, 0x0B);  // IRQs enabled, RTS/DSR set
}

/// Rust format string implementation for SerialWriter.
impl core::fmt::Write for SerialWriter {
    fn write_str(&mut self, s: &str) -> core::fmt::Result {
        self.serial_write(s);
        Ok(())
    }
}

#[allow(dead_code)]
impl SerialWriter {
    /// Instantiates a new `SerialWriter`.
    ///
    /// # Arguments
    ///
    /// * `port` - Serial port base address.
    pub fn new(port: u16) -> Self {
        Self {
            port: port,
        }
    }

    /// Writes a given byte string to serial port.
    ///
    /// # Arguments
    ///
    /// * `str` - String to write.
    fn serial_write(&mut self, str: &str) {
        for char in str.bytes() {
            // Wait for transmit buffer to be empty
            while (inb(self.port + 5) & 0x20) == 0 {}

            // Send the byte
            outb(self.port, char);
        }
    }

    /// Prints a debug message to serial port.
    /// 
    /// The message is only printed if its given level is equal to or higher than
    /// the hardcoded global variable for overall debugging level.
    ///
    /// # Arguments
    ///
    /// * `debug_level` - Specified debug level of the given message.
    /// * `msg` - Debug message to print.
    pub fn print_debug(&mut self, debug_level: crate::DebugLevel, msg: &str) {
        if debug_level < crate::globals::DEBUG_LEVEL {
            return;
        }

        self.serial_write(msg);
        self.serial_write("\n");
    }
}

// Implements unsafe Send for spinlock management
unsafe impl Send for SerialWriter {}

/// Separate function to print using an already-initialized serial port without
/// relying on spinlocks. This is for cases where we need to troubleshoot
/// spinlocks themselves.
///
/// # Arguments
///
/// * `msg` - Debug message to print.
#[allow(dead_code)]
pub fn print(msg: &str) {
    for char in msg.bytes() {
        // Wait for transmit buffer to be empty
        while (inb(crate::globals::COM1_PORT + 5) & 0x20) == 0 {}

        // Send the byte
        outb(crate::globals::COM1_PORT, char);
    }
}
