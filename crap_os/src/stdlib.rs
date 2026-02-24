/*
    CrapOS Standard Library Module
*/

const HEX_CHARS_UPPER: &[u8; 16] = b"0123456789ABCDEF";

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
