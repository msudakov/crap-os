//! This file contains system helper routines

use core::sync::atomic::Ordering;
use crate::globals::HEX_CHARS_UPPER;

/// Converts a u64 to a fixed-width hex byte string.
/// 
/// The hex string in the output byte buffer will be prefixed with `0x`.
///
/// # Arguments
///
/// * `value` - The u64 value to convert.
/// 
/// # Returns
/// 
/// The 18-char-wide byte array, prefixed with `0x`.
#[allow(dead_code)]
pub fn u64_to_hex_bytes(value: u64) -> [u8; 18] {
    let mut buffer = [b'0'; 18];

    // Start filling from the end of the 16 hex digits section
    let mut i = 18 - 1;
    let mut temp_value = value;

    // Write the 16 hex digits (2 chars per byte of u64)
    for _ in 0..16 {
        // Get the last 4 bits (nibble) and map to hex char
        buffer[i] = HEX_CHARS_UPPER[(temp_value & 0xF) as usize];
        temp_value >>= 4;
        i -= 1;
    }

    // Add the "0x" prefix
    buffer[0] = b'0';
    buffer[1] = b'x';
    buffer
}

/// Converts a u32 to a fixed-width hex byte string.
/// 
/// The hex string in the output byte buffer will be prefixed with `0x`.
///
/// # Arguments
///
/// * `value` - The u32 value to convert.
/// 
/// # Returns
/// 
/// The 10-char-wide byte array, prefixed with `0x`.
#[allow(dead_code)]
pub fn u32_to_hex_bytes(value: u32) -> [u8; 10] {
    let mut buffer = [b'0'; 10];

    // Start filling from the end of the 16 hex digits section
    let mut i = 10 - 1;
    let mut temp_value = value;

    // Write the 8 hex digits (2 chars per byte of u64)
    for _ in 0..8 {
        // Get the last 4 bits (nibble) and map to hex char
        buffer[i] = HEX_CHARS_UPPER[(temp_value & 0xF) as usize];
        temp_value >>= 4;
        i -= 1;
    }

    // Add the "0x" prefix
    buffer[0] = b'0';
    buffer[1] = b'x';
    buffer
}

/// Converts a u32 to a decimal string for logging where we cannot use
/// `format!` or any allocating machinery.
/// 
/// # Arguments
///
/// * `value`  - The u32 mutable value to convert.
/// * `buffer` - Mutable buffer to use during conversion.
/// 
/// # Returns
/// 
/// Returns the decimal string representation of the given value.
#[allow(dead_code)]
pub fn u32_to_dec_str(mut value: u32, buffer: &mut [u8; 10]) -> &str {
    if value == 0 {
        buffer[9] = b'0';
        return core::str::from_utf8(&buffer[9..]).unwrap();
    }

    let mut i = 10usize;
    while value > 0 {
        i -= 1;
        buffer[i] = b'0' + (value % 10) as u8;
        value /= 10;
    }

    core::str::from_utf8(&buffer[i..]).unwrap()
}

/// Converts a u64 to a decimal string for logging where we cannot use 
/// `format!` or any allocating machinery.
/// 
/// # Arguments
///
/// * `value`  - The u64 mutable value to convert.
/// * `buffer` - Mutable buffer to use during conversion.
/// 
/// # Returns
/// 
/// Returns the decimal string representation of the given value.
#[allow(dead_code)]
pub fn u64_to_dec_str(mut value: u64, buffer: &mut [u8; 20]) -> &str {
    if value == 0 {
        buffer[19] = b'0';
        return core::str::from_utf8(&buffer[19..]).unwrap();
    }

    let mut i = 20usize;
    while value > 0 {
        i -= 1;
        buffer[i] = b'0' + (value % 10) as u8;
        value /= 10;
    }

    core::str::from_utf8(&buffer[i..]).unwrap()
}

/// Formats and prints one `u64` value as a labelled hex line to the serial
/// port, bypassing the spinlock.
/// 
/// # Arguments
/// 
/// * `label` - The label to accompany the given value.
/// * `value` - The `u64`` value to print.
pub fn print_u64_field(label: &str, value: u64) {
    crate::hardware_manager::sprint(label);
    let hex = crate::system_routines::u64_to_hex_bytes(value);
    crate::hardware_manager::sprint(
        unsafe { 
            core::str::from_utf8_unchecked(&hex)
        }
    );
    crate::hardware_manager::sprint("\n");
}

/// Saves the current CPU flags register (which includes the interrupt-enable
/// flag `IF`) and then disables hardware interrupts by executing `CLI` (Clear
/// Interrupt Flag).
///
/// # Returns
/// 
/// Returns the saved flags value as a `usize` so that `restore_interrupts` can
/// later put them back, re-enabling interrupts if and only if they were enabled
/// before this call.
#[inline]
pub fn disable_interrupts_save() -> usize {
    // `flags` will receive the RFLAGS value captured before CLI
    let flags: usize;
    unsafe {
        core::arch::asm!(
            "pushfq",       // Push the 64-bit RFLAGS register onto the stack
            "pop {flags}",  // Pop it back into our output variable
            "cli",          // Clear the interrupt-enable flag (bit 9 of RFLAGS)
            flags = out(reg) flags,
            options(nomem, preserves_flags)
        );
    }
    flags
}

/// Restores the CPU flags register from a previously saved value.
///
/// If `flags` has the interrupt-enable bit (`IF` bit 9) set, this effectively
/// re-enables interrupts. If it was clear, interrupts stay disabled.
#[inline]
pub fn restore_interrupts(flags: usize) {
    unsafe {
        core::arch::asm!(
            "push {flags}",        // Push saved RFLAGS value onto the stack
            "popfq",               // Pop it back into RFLAGS (restoring IF bit)
            flags = in(reg) flags,
            options(nomem, preserves_flags)
        );
    }
}

/// Fetches the current value of the monotonic tick counter.
///
/// The tick rate depends on the `initial_count` passed to `configure_timer`.
/// It is not calibrated to wall-clock time by default.
/// 
/// # Returns
/// 
/// Returns the current value of the monotonic tick counter.
#[allow(dead_code)]
pub fn get_timer_ticks() -> u64 {
    crate::globals::TIMER_TICKS.load(Ordering::Relaxed)
}
