//! This file contains system helper routines

use core::sync::atomic::Ordering;
use crate::globals::{HEX_CHARS_UPPER, BASE64_CHARS};
use crate::task_scheduler::TaskId;

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
    let hex = u64_to_hex_bytes(value);
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

/// Encodes the `slot_index` and `slot_generation` fields of `TaskId` into a
/// single u64 value.
/// 
/// This is useful when we need to pass a `TaskId` struct as a single u64
/// function argument value into a `SystemTask`. This works by encoding the u8
/// value of `slot_generation` into the highest 8 bits of the u64, whose lower
/// bits contain the `slot_index` value. Because the number of `TaskSlot`
/// elements in the task table will never be this high, there is no possibility
/// of practical collision of the two values in the middle.
/// 
/// First, we cast the u8 to u64. Then, we shift left by 56 bits (64 total bits
/// - 8 bits). And lastly, we bitwise OR with target slot index value.
/// 
/// # Arguments
/// 
/// * `task_id` - The `TaskId` struct to encode.
/// 
/// # Returns
/// 
/// Returns the single u64 value with encoded components of `TaskId` inside it.
pub fn compress_task_id(task_id: TaskId) -> u64 {
    (task_id.slot_index as u64) | ((task_id.slot_generation as u64) << 56)
}

/// Does the opposite of `compress_task_id` and expands it back to the proper
/// `TaskId` struct.
/// 
/// # Arguments
/// 
/// * `encoded_task_id` - The u64 value with encoded components of a `TaskId`.
/// 
/// # Returns
/// 
/// Decoded `TaskId` structure.
pub fn expand_task_id(encoded_task_id: u64) -> TaskId {
    let mut slot_index = encoded_task_id as usize;

    // Extract slot generation from the highest 8 bits of the value
    let slot_generation = (slot_index >> (usize::BITS - 8)) as u8;

    // Shift left 8 to discard the top, then right 8 to restore position.
    slot_index = (slot_index << 8) >> 8;

    // Restore the original task id
    TaskId {slot_index, slot_generation}
}

/// Converts a byte slice into an owned hex string prefixed with `0x`,
/// with each byte rendered as exactly two uppercase hex digits.
///
/// An empty slice returns the string `"0x"` with no hex digits following
/// the prefix.
///
/// # Arguments
///
/// * `bytes` - The byte slice to encode.
///
/// # Returns
///
/// Returns an owned `String` of length `2 + (bytes.len() * 2)`.
#[allow(dead_code)]
pub fn bytes_to_hex(bytes: &[u8]) -> alloc::string::String {
    // Pre-allocate the exact capacity: "0x" prefix + 2 chars per byte
    let mut hex = alloc::string::String::with_capacity(2 + bytes.len() * 2);

    hex.push('0');
    hex.push('x');

    for &byte in bytes {
        // Encode the high nibble first, then the low nibble
        hex.push(HEX_CHARS_UPPER[(byte >> 4) as usize] as char);
        hex.push(HEX_CHARS_UPPER[(byte & 0xF) as usize] as char);
    }

    hex
}

/// Decodes a single ASCII hex character into its 4-bit nibble value.
///
/// Accepts `0`–`9`, `a`–`f`, and `A`–`F`.
/// 
/// # Arguments
///
/// * `byte` - The single ASCII hex character to decode.
/// 
/// # Returns
/// 
/// Returns 4-bit nibble value for an ASCII hex character on success, or `None`
/// in the case of an invalid value.
#[inline]
fn hex_nibble(byte: u8) -> Option<u8> {
    match byte {
        b'0'..=b'9' => Some(byte - b'0'),
        b'a'..=b'f' => Some(byte - b'a' + 10),
        b'A'..=b'F' => Some(byte - b'A' + 10),
        _           => None,
    }
}

/// Converts a hex string into an owned byte vector.
///
/// Accepts both uppercase and lowercase hex digits (`0`–`9`, `a`–`f`,
/// `A`–`F`), with or without a `0x` / `0X` prefix. An odd number of hex
/// digits is handled by implicitly zero-padding the leading nibble, so
/// `"0xabc"` is treated as `"0x0abc"` and decodes to `[0x0a, 0xbc]`.
///
/// # Arguments
///
/// * `hex` - The hex string to decode.
///
/// # Returns
///
/// Returns `Some(Vec<u8>)` containing the decoded bytes on success,
/// or `None` if:
///   - The input is empty or contains only the `0x` / `0X` prefix with no
///     hex digits following it;
///   - Any character in the digit portion is not a valid hex digit.
#[allow(dead_code)]
pub fn hex_to_bytes(hex: &str) -> Option<alloc::vec::Vec<u8>> {
    // Strip the optional "0x" / "0X" prefix before processing
    let hex = if hex.starts_with("0x") || hex.starts_with("0X") {
        &hex[2..]
    } else {
        hex
    };

    // A bare prefix with no digits, or a completely empty string, produces
    // no bytes and is considered invalid.
    if hex.is_empty() {
        return None;
    }

    let hex_chars = hex.as_bytes();

    // Pre-allocate the exact output capacity. An odd digit count gets one
    // extra byte for the implicit leading zero nibble.
    let byte_len = (hex_chars.len() + 1) / 2;
    let mut bytes = alloc::vec::Vec::with_capacity(byte_len);

    // For an odd-length input, synthesize the first byte from a leading zero
    // nibble and the first actual digit, then advance past that digit.
    // For even-length input, `start` remains 0 and this block is skipped.
    let mut i = 0;
    if hex_chars.len() % 2 != 0 {
        bytes.push(hex_nibble(hex_chars[0])?);
        i = 1;
    }

    // Decode the remaining characters in two-character (full byte) steps.
    while i < hex_chars.len() {
        let high = hex_nibble(hex_chars[i])?;
        let low  = hex_nibble(hex_chars[i + 1])?;
        bytes.push((high << 4) | low);
        i += 2;
    }

    Some(bytes)
}

/// Encodes a byte slice into a Base64 string, without padding.
///
/// The PHC string format used by Argon2 and other password hashing schemes
/// specifies Base64 without trailing `=` padding characters. This function
/// omits padding accordingly. The output length is `ceil(input.len() * 4 / 3)`.
///
/// # Arguments
///
/// * `input` - The byte slice to encode. May be any length, including empty.
///
/// # Returns
///
/// Returns an owned `String` containing the Base64-encoded representation of
/// `input`, without padding.
#[allow(dead_code)]
pub fn base64_encode_unpadded(input: &[u8]) -> alloc::string::String {
    // Each group of 3 input bytes produces 4 Base64 characters. The output
    // length is ceil(n * 4 / 3), which equals (n + 2) / 3 * 4 before stripping
    // padding, but we omit padding so the exact length is (n * 4 + 2) / 3.
    let result_length = (input.len() * 4 + 2) / 3;
    let mut result = alloc::string::String::with_capacity(result_length);
 
    // Process full 3-byte groups
    let mut i = 0;
    while i + 3 <= input.len() {
        // Pack three bytes into a 24-bit integer, then extract four 6-bit
        // indices into BASE64_CHARS.
        let n = ((input[i] as u32) << 16)
              | ((input[i + 1] as u32) << 8)
              |  (input[i + 2] as u32);
        result.push(BASE64_CHARS[(n >> 18) as usize & 0x3F] as char);
        result.push(BASE64_CHARS[(n >> 12) as usize & 0x3F] as char);
        result.push(BASE64_CHARS[(n >>  6) as usize & 0x3F] as char);
        result.push(BASE64_CHARS[ n        as usize & 0x3F] as char);
        i += 3;
    }
 
    // Handle the remaining 1 or 2 bytes without emitting padding characters
    let remaining = input.len() - i;
    if remaining == 1 {
        // One byte remaining: produces 2 Base64 characters (12 bits, 4 unused)
        let n = (input[i] as u32) << 16;
        result.push(BASE64_CHARS[(n >> 18) as usize & 0x3F] as char);
        result.push(BASE64_CHARS[(n >> 12) as usize & 0x3F] as char);
    }
    else if remaining == 2 {
        // Two bytes remaining: produces 3 Base64 characters (18 bits, 2 unused)
        let n = ((input[i] as u32) << 16) | ((input[i + 1] as u32) << 8);
        result.push(BASE64_CHARS[(n >> 18) as usize & 0x3F] as char);
        result.push(BASE64_CHARS[(n >> 12) as usize & 0x3F] as char);
        result.push(BASE64_CHARS[(n >>  6) as usize & 0x3F] as char);
    }
 
    result
}

/// Encodes a byte slice into a padded Base64 string (RFC 4648), with trailing
/// `=` padding characters to make the output length a multiple of 4.
///
/// This is the standard Base64 encoding used by most protocols (MIME, PEM,
/// HTTP Basic Auth, etc.).
///
/// # Arguments
///
/// * `input` - The byte slice to encode. May be any length, including empty.
///
/// # Returns
///
/// Returns an owned `String` whose length is always a multiple of 4, containing
/// the Base64-encoded representation of `input` with `=` padding as needed.
#[allow(dead_code)]
pub fn base64_encode(input: &[u8]) -> alloc::string::String {
    // Start with the unpadded encoding, then append 0, 1, or 2 `=` characters
    // so the total length is a multiple of 4.
    let mut result = base64_encode_unpadded(input);

    let remainder = result.len() % 4;
    if remainder != 0 {
        for _ in 0..(4 - remainder) {
            result.push('=');
        }
    }

    result
}

/// Decodes a single Base64 character to its 6-bit value.
/// 
/// # Arguments
///
/// * `byte` - The Base64 character to decode.
/// 
/// # Returns
///
/// Returns `None` for any byte that is not a member of the standard Base64
/// alphabet. Padding characters (`=`) are not handled here; they are stripped
/// before [`base64_decode`] calls this function.
#[inline]
fn base64_val(byte: u8) -> Option<u8> {
    match byte {
        b'A'..=b'Z' => Some(byte - b'A'),
        b'a'..=b'z' => Some(byte - b'a' + 26),
        b'0'..=b'9' => Some(byte - b'0' + 52),
        b'+'        => Some(62),
        b'/'        => Some(63),
        _           => None,
    }
}

/// Decodes a Base64 string (RFC 4648) into an owned byte vector.
///
/// Accepts input with or without trailing `=` padding. Whitespace is not
/// accepted; any character outside the standard Base64 alphabet (including
/// whitespace, newlines, and hyphens) causes the function to return `None`.
///
/// # Arguments
///
/// * `input` - The Base64 string to decode.
///
/// # Returns
///
/// Returns `Some(Vec<u8>)` containing the decoded bytes on success,
/// or `None` if:
///   - Any character is outside the Base64 alphabet (`A`–`Z`, `a`–`z`,
///     `0`–`9`, `+`, `/`, `=`);
///   - Padding characters (`=`) appear in an invalid position (i.e., not at
///     the end, or more than two consecutive padding characters).
#[allow(dead_code)]
pub fn base64_decode(input: &str) -> Option<alloc::vec::Vec<u8>> {
    // Strip trailing padding characters before processing; we handle
    // unpadded input equally well.
    let input = input.trim_end_matches('=');
    let bytes = input.as_bytes();
 
    // The output length is floor(n * 3 / 4) where n is the unpadded length
    let result_length = bytes.len() * 3 / 4;
    let mut result = alloc::vec::Vec::with_capacity(result_length);
 
    // Decode in 4-character groups. A trailing group of 2 or 3 characters is
    // handled separately after the main loop.
    let mut i = 0;
    while i + 4 <= bytes.len() {
        let a = base64_val(bytes[i])?;
        let b = base64_val(bytes[i + 1])?;
        let c = base64_val(bytes[i + 2])?;
        let d = base64_val(bytes[i + 3])?;
 
        let n = ((a as u32) << 18)
              | ((b as u32) << 12)
              | ((c as u32) <<  6)
              |  (d as u32);
 
        result.push((n >> 16) as u8);
        result.push((n >>  8) as u8);
        result.push( n        as u8);
        i += 4;
    }
 
    // Handle the remaining 2 or 3 unpadded characters
    let remaining = bytes.len() - i;
    if remaining == 2 {
        // 2 chars -> 1 byte (12 bits encoded, 4 are padding zeros)
        let a = base64_val(bytes[i])?;
        let b = base64_val(bytes[i + 1])?;
        let n = ((a as u32) << 18) | ((b as u32) << 12);
        result.push((n >> 16) as u8);
    }
    else if remaining == 3 {
        // 3 chars ->  2 bytes (18 bits encoded, 2 are padding zeros)
        let a = base64_val(bytes[i])?;
        let b = base64_val(bytes[i + 1])?;
        let c = base64_val(bytes[i + 2])?;
        let n = ((a as u32) << 18) | ((b as u32) << 12) | ((c as u32) << 6);
        result.push((n >> 16) as u8);
        result.push((n >>  8) as u8);
    }
    else if remaining == 1 {
        // A single remaining character is always invalid in Base64: every
        // valid Base64 character encodes 6 bits, so the minimum meaningful
        // unit that produces at least one output byte is 2 characters (12
        // bits, of which 8 are data). One character encodes only 6 bits,
        // which is not enough for a full byte.
        return None;
    }
 
    Some(result)
}
