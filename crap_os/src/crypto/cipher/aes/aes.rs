//! Shared Primitives of the AES Block Cipher for All AES-Based Modes
//!
//! This module implements the AES-256 block cipher as specified in FIPS 197.
//! It exposes only the building blocks that higher-level mode modules need:
//!
//!   - [`KeySchedule`]: expanded round-key state derived from a 256-bit key.
//!   - [`aes256_encrypt_block`]: encrypts a single 16-byte block in place using
//!     the provided [`KeySchedule`]. This is the only AES primitive exposed
//!     publicly; decryption is handled by mode-specific logic.
//!
//! This design avoids heap allocation, and the state lives on the stack or in
//! caller-supplied buffers. This implementation is 256-bit only; AES-128 and
//! AES-192 are not implemented. The key schedule expansion is therefore fixed
//! at 15 round keys (14 rounds + the initial AddRoundKey), each 16 bytes,
//! giving a 240-byte expanded key.
//!
//! The 256-byte S-box and its inverse are stored as `const` arrays. This is
//! the standard software AES approach; it is not timing-side-channel-free on
//! all microarchitectures (cache-timing attacks exist). For a kernel context
//! without untrusted co-tenants on the same core this is acceptable; a future
//! hardened variant could use bitsliced AES.

#![allow(dead_code)]

/// Number of rounds for AES-256.
pub const AES256_ROUNDS: usize = 14;

/// Number of round keys = rounds + 1.
pub const AES256_ROUND_KEYS: usize = AES256_ROUNDS + 1;

/// AES block size in bytes.
pub const BLOCK_SIZE: usize = 16;

/// AES-256 key size in bytes.
pub const KEY256_SIZE: usize = 32;

/// The AES SubBytes forward substitution table.
///
/// `SBOX[i]` gives the substituted byte for input byte `i`. Generated from
/// the multiplicative inverse in GF(2^8) followed by an affine transform.
#[rustfmt::skip]
pub const SBOX: [u8; 256] = [
    0x63, 0x7c, 0x77, 0x7b, 0xf2, 0x6b, 0x6f, 0xc5,
    0x30, 0x01, 0x67, 0x2b, 0xfe, 0xd7, 0xab, 0x76,
    0xca, 0x82, 0xc9, 0x7d, 0xfa, 0x59, 0x47, 0xf0,
    0xad, 0xd4, 0xa2, 0xaf, 0x9c, 0xa4, 0x72, 0xc0,
    0xb7, 0xfd, 0x93, 0x26, 0x36, 0x3f, 0xf7, 0xcc,
    0x34, 0xa5, 0xe5, 0xf1, 0x71, 0xd8, 0x31, 0x15,
    0x04, 0xc7, 0x23, 0xc3, 0x18, 0x96, 0x05, 0x9a,
    0x07, 0x12, 0x80, 0xe2, 0xeb, 0x27, 0xb2, 0x75,
    0x09, 0x83, 0x2c, 0x1a, 0x1b, 0x6e, 0x5a, 0xa0,
    0x52, 0x3b, 0xd6, 0xb3, 0x29, 0xe3, 0x2f, 0x84,
    0x53, 0xd1, 0x00, 0xed, 0x20, 0xfc, 0xb1, 0x5b,
    0x6a, 0xcb, 0xbe, 0x39, 0x4a, 0x4c, 0x58, 0xcf,
    0xd0, 0xef, 0xaa, 0xfb, 0x43, 0x4d, 0x33, 0x85,
    0x45, 0xf9, 0x02, 0x7f, 0x50, 0x3c, 0x9f, 0xa8,
    0x51, 0xa3, 0x40, 0x8f, 0x92, 0x9d, 0x38, 0xf5,
    0xbc, 0xb6, 0xda, 0x21, 0x10, 0xff, 0xf3, 0xd2,
    0xcd, 0x0c, 0x13, 0xec, 0x5f, 0x97, 0x44, 0x17,
    0xc4, 0xa7, 0x7e, 0x3d, 0x64, 0x5d, 0x19, 0x73,
    0x60, 0x81, 0x4f, 0xdc, 0x22, 0x2a, 0x90, 0x88,
    0x46, 0xee, 0xb8, 0x14, 0xde, 0x5e, 0x0b, 0xdb,
    0xe0, 0x32, 0x3a, 0x0a, 0x49, 0x06, 0x24, 0x5c,
    0xc2, 0xd3, 0xac, 0x62, 0x91, 0x95, 0xe4, 0x79,
    0xe7, 0xc8, 0x37, 0x6d, 0x8d, 0xd5, 0x4e, 0xa9,
    0x6c, 0x56, 0xf4, 0xea, 0x65, 0x7a, 0xae, 0x08,
    0xba, 0x78, 0x25, 0x2e, 0x1c, 0xa6, 0xb4, 0xc6,
    0xe8, 0xdd, 0x74, 0x1f, 0x4b, 0xbd, 0x8b, 0x8a,
    0x70, 0x3e, 0xb5, 0x66, 0x48, 0x03, 0xf6, 0x0e,
    0x61, 0x35, 0x57, 0xb9, 0x86, 0xc1, 0x1d, 0x9e,
    0xe1, 0xf8, 0x98, 0x11, 0x69, 0xd9, 0x8e, 0x94,
    0x9b, 0x1e, 0x87, 0xe9, 0xce, 0x55, 0x28, 0xdf,
    0x8c, 0xa1, 0x89, 0x0d, 0xbf, 0xe6, 0x42, 0x68,
    0x41, 0x99, 0x2d, 0x0f, 0xb0, 0x54, 0xbb, 0x16,
];

/// The AES SubBytes inverse substitution table.
///
/// `INV_SBOX[i]` is the inverse of `SBOX[i]`. Used during decryption
/// (InvSubBytes). Exposed so that mode-level decryption implementations
/// can use it without re-declaring it.
#[rustfmt::skip]
pub const INV_SBOX: [u8; 256] = [
    0x52, 0x09, 0x6a, 0xd5, 0x30, 0x36, 0xa5, 0x38,
    0xbf, 0x40, 0xa3, 0x9e, 0x81, 0xf3, 0xd7, 0xfb,
    0x7c, 0xe3, 0x39, 0x82, 0x9b, 0x2f, 0xff, 0x87,
    0x34, 0x8e, 0x43, 0x44, 0xc4, 0xde, 0xe9, 0xcb,
    0x54, 0x7b, 0x94, 0x32, 0xa6, 0xc2, 0x23, 0x3d,
    0xee, 0x4c, 0x95, 0x0b, 0x42, 0xfa, 0xc3, 0x4e,
    0x08, 0x2e, 0xa1, 0x66, 0x28, 0xd9, 0x24, 0xb2,
    0x76, 0x5b, 0xa2, 0x49, 0x6d, 0x8b, 0xd1, 0x25,
    0x72, 0xf8, 0xf6, 0x64, 0x86, 0x68, 0x98, 0x16,
    0xd4, 0xa4, 0x5c, 0xcc, 0x5d, 0x65, 0xb6, 0x92,
    0x6c, 0x70, 0x48, 0x50, 0xfd, 0xed, 0xb9, 0xda,
    0x5e, 0x15, 0x46, 0x57, 0xa7, 0x8d, 0x9d, 0x84,
    0x90, 0xd8, 0xab, 0x00, 0x8c, 0xbc, 0xd3, 0x0a,
    0xf7, 0xe4, 0x58, 0x05, 0xb8, 0xb3, 0x45, 0x06,
    0xd0, 0x2c, 0x1e, 0x8f, 0xca, 0x3f, 0x0f, 0x02,
    0xc1, 0xaf, 0xbd, 0x03, 0x01, 0x13, 0x8a, 0x6b,
    0x3a, 0x91, 0x11, 0x41, 0x4f, 0x67, 0xdc, 0xea,
    0x97, 0xf2, 0xcf, 0xce, 0xf0, 0xb4, 0xe6, 0x73,
    0x96, 0xac, 0x74, 0x22, 0xe7, 0xad, 0x35, 0x85,
    0xe2, 0xf9, 0x37, 0xe8, 0x1c, 0x75, 0xdf, 0x6e,
    0x47, 0xf1, 0x1a, 0x71, 0x1d, 0x29, 0xc5, 0x89,
    0x6f, 0xb7, 0x62, 0x0e, 0xaa, 0x18, 0xbe, 0x1b,
    0xfc, 0x56, 0x3e, 0x4b, 0xc6, 0xd2, 0x79, 0x20,
    0x9a, 0xdb, 0xc0, 0xfe, 0x78, 0xcd, 0x5a, 0xf4,
    0x1f, 0xdd, 0xa8, 0x33, 0x88, 0x07, 0xc7, 0x31,
    0xb1, 0x12, 0x10, 0x59, 0x27, 0x80, 0xec, 0x5f,
    0x60, 0x51, 0x7f, 0xa9, 0x19, 0xb5, 0x4a, 0x0d,
    0x2d, 0xe5, 0x7a, 0x9f, 0x93, 0xc9, 0x9c, 0xef,
    0xa0, 0xe0, 0x3b, 0x4d, 0xae, 0x2a, 0xf5, 0xb0,
    0xc8, 0xeb, 0xbb, 0x3c, 0x83, 0x53, 0x99, 0x61,
    0x17, 0x2b, 0x04, 0x7e, 0xba, 0x77, 0xd6, 0x26,
    0xe1, 0x69, 0x14, 0x63, 0x55, 0x21, 0x0c, 0x7d,
];

/// Round constant table (Rcon) for AES key schedule.
///
/// `RCON[i]` is the round constant for key schedule round `i`, defined as
/// x^(i-1) in GF(2^8) with the irreducible polynomial 0x11b. Only the first
/// 7 values are needed for AES-256 (which has 7 key schedule iterations after
/// the initial two 128-bit halves). We include 10 for clarity and future use.
#[rustfmt::skip]
const RCON: [u8; 10] = [
    0x01, 0x02, 0x04, 0x08, 0x10,
    0x20, 0x40, 0x80, 0x1b, 0x36,
];

/// Multiplies a byte by 2 in GF(2^8) modulo the AES irreducible polynomial
/// x^8 + x^4 + x^3 + x + 1 (0x11b).
///
/// Left-shifts by 1 and XORs with 0x1b if the high bit was set, implementing
/// the conditional reduction without branching.
/// 
/// # Arguments
///
/// * `x` - The byte to multiply.
/// 
/// # Returns
///
/// Returns the result of the computation.
#[inline(always)]
pub fn gmul2(x: u8) -> u64 {
    let hi = x >> 7;
    ((x << 1) ^ (hi * 0x1b)) as u64
}

/// Multiplies a byte by 3 in GF(2^8): xtime(x) XOR x.
/// 
/// # Arguments
///
/// * `x` - The byte to multiply.
/// 
/// # Returns
///
/// Returns the result of the computation.
#[inline(always)]
pub fn gmul3(x: u8) -> u64 {
    gmul2(x) ^ (x as u64)
}

/// Expanded AES-256 round keys.
///
/// Holds the 15 round keys derived from a 256-bit (32-byte) master key by
/// the AES-256 key schedule. Each round key is 16 bytes (one AES block).
/// The array is stored flat: `round_keys[r * 16 .. r * 16 + 16]` is the round
/// key for round `r`.
pub struct KeySchedule {
    /// Flat array of all 15 * 16 = 240 round-key bytes.
    pub round_keys: [u8; AES256_ROUND_KEYS * BLOCK_SIZE],
}

impl KeySchedule {
    /// Expands a 256-bit AES key into a full [`KeySchedule`].
    ///
    /// Implements the AES-256 key expansion. The first two round keys are the
    /// raw key halves; the remaining 13 are derived by the key schedule
    /// recurrence.
    ///
    /// # Arguments
    ///
    /// * `key` - The 32-byte (256-bit) AES key.
    pub fn new(key: &[u8; KEY256_SIZE]) -> Self {
        let mut rk = [0u8; AES256_ROUND_KEYS * BLOCK_SIZE];

        // The first 32 bytes of the expanded key are the key itself
        rk[..32].copy_from_slice(key);

        // AES-256 key schedule: Nk=8, Nr=14, produces 15 round keys (60 words)
        // We work in 4-byte words. Words 0..7 are the raw key.
        // Words i (8 <= i < 60) are derived from earlier words.
        for i in 8..60usize {
            let prev_word: [u8; 4] = rk[(i - 1) * 4..(i - 1) * 4 + 4]
                .try_into()
                .unwrap();
            let mut temp = prev_word;

            if i % 8 == 0 {
                // RotWord: rotate left by one byte
                temp = [temp[1], temp[2], temp[3], temp[0]];
                // SubWord
                temp = [SBOX[temp[0] as usize], SBOX[temp[1] as usize],
                        SBOX[temp[2] as usize], SBOX[temp[3] as usize]];
                // XOR with Rcon (only the first byte of Rcon is non-zero)
                temp[0] ^= RCON[i / 8 - 1];
            }
            else if i % 8 == 4 {
                // AES-256 extra SubWord at every Nk/2 position
                temp = [SBOX[temp[0] as usize], SBOX[temp[1] as usize],
                        SBOX[temp[2] as usize], SBOX[temp[3] as usize]];
            }

            let base = i * 4;
            let prev_base = (i - 8) * 4;
            rk[base]     = rk[prev_base]     ^ temp[0];
            rk[base + 1] = rk[prev_base + 1] ^ temp[1];
            rk[base + 2] = rk[prev_base + 2] ^ temp[2];
            rk[base + 3] = rk[prev_base + 3] ^ temp[3];
        }

        Self { round_keys: rk }
    }

    /// Gets the round key a round as a 16-byte slice.
    ///
    /// # Arguments
    ///
    /// * `r` - The given AES round.
    /// 
    /// # Returns
    ///
    /// Returns the round key for the round `r`.
    /// 
    /// # Panics
    ///
    /// Panics if `r >= AES256_ROUND_KEYS`.
    #[inline(always)]
    pub fn round_key(&self, r: usize) -> &[u8; BLOCK_SIZE] {
        self.round_keys[r * BLOCK_SIZE..(r + 1) * BLOCK_SIZE]
            .try_into()
            .unwrap()
    }
}

/// Encrypts a single 16-byte AES block in place using the given key schedule.
///
/// Implements the full AES-256 cipher: AddRoundKey, then 13 rounds of
/// SubBytes + ShiftRows + MixColumns + AddRoundKey, then a final
/// round of SubBytes + ShiftRows + AddRoundKey (no MixColumns in the last
/// round).
///
/// The state is maintained as a 4*4 byte matrix in column-major order (the
/// AES "state array" layout): bytes 0, 4, 8, 12 are column 0; bytes 1, 5, 9,
/// 13 are column 1; and so on.
///
/// # Arguments
///
/// * `block`    - 16-byte block to encrypt, modified in place.
/// * `schedule` - Expanded key schedule produced by [`KeySchedule::new`].
pub fn aes256_encrypt_block(
    block: &mut [u8; BLOCK_SIZE],
    schedule: &KeySchedule,
) {
    // Initial AddRoundKey
    add_round_key(block, schedule.round_key(0));

    // Rounds 1–13: full rounds with MixColumns
    for r in 1..AES256_ROUNDS {
        sub_bytes(block);
        shift_rows(block);
        mix_columns(block);
        add_round_key(block, schedule.round_key(r));
    }

    // Final round: no MixColumns
    sub_bytes(block);
    shift_rows(block);
    add_round_key(block, schedule.round_key(AES256_ROUNDS));
}

/// SubBytes: applies the AES S-box to every byte of the state.
#[inline]
fn sub_bytes(state: &mut [u8; BLOCK_SIZE]) {
    for b in state.iter_mut() {
        *b = SBOX[*b as usize];
    }
}

/// ShiftRows: cyclically shifts the rows of the AES state array left.
///
/// Row 0 is unchanged; row 1 shifts left by 1; row 2 by 2; row 3 by 3.
/// The state is stored in column-major order, so row `r` consists of bytes
/// at indices r, r+4, r+8, r+12.
/// 
/// # Arguments
///
/// * `state` - The current state of the AES cipher.
#[inline]
fn shift_rows(state: &mut [u8; BLOCK_SIZE]) {
    // Row 1: shift left by 1
    let tmp = state[1];
    state[1] = state[5];
    state[5] = state[9];
    state[9] = state[13];
    state[13] = tmp;

    // Row 2: shift left by 2
    state.swap(2, 10);
    state.swap(6, 14);

    // Row 3: shift left by 3 (= shift right by 1)
    let tmp = state[15];
    state[15] = state[11];
    state[11] = state[7];
    state[7]  = state[3];
    state[3]  = tmp;
}

/// MixColumns: applies the AES MixColumns transformation to each column.
///
/// Each column is treated as a four-term polynomial over GF(2^8) and
/// multiplied by the fixed AES polynomial a(x) = {03}x^3 + {01}x^2 +
/// {01}x + {02}.
/// 
/// # Arguments
///
/// * `state` - The current state of the AES cipher.
#[inline]
fn mix_columns(state: &mut [u8; BLOCK_SIZE]) {
    for col in 0..4 {
        let base = col * 4;
        let s0 = state[base];
        let s1 = state[base + 1];
        let s2 = state[base + 2];
        let s3 = state[base + 3];

        state[base]     = (gmul2(s0) ^ gmul3(s1) ^ s2 as u64 ^ s3 as u64) as u8;
        state[base + 1] = (s0 as u64 ^ gmul2(s1) ^ gmul3(s2) ^ s3 as u64) as u8;
        state[base + 2] = (s0 as u64 ^ s1 as u64 ^ gmul2(s2) ^ gmul3(s3)) as u8;
        state[base + 3] = (gmul3(s0) ^ s1 as u64 ^ s2 as u64 ^ gmul2(s3)) as u8;
    }
}

/// AddRoundKey: XORs a round key into the state byte-by-byte.
/// 
/// # Arguments
///
/// * `state`     - The current state of the AES cipher.
/// * `round_key` - The key to add (XOR) for a new cipher round.
#[inline]
fn add_round_key(state: &mut [u8; BLOCK_SIZE], round_key: &[u8; BLOCK_SIZE]) {
    for i in 0..BLOCK_SIZE {
        state[i] ^= round_key[i];
    }
}
