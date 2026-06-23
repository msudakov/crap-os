//! AES-256-GCM Authenticated Encryption
//!
//! This module implements AES-256 in Galois/Counter Mode (GCM) as specified
//! in NIST SP 800-38D. It provides authenticated encryption with a 128-bit
//! (16-byte) authentication tag, a 96-bit (12-byte) randomly generated IV,
//! and AES-256 as the underlying block cipher.
//!
//! All fixed-length metadata is placed at the front so the decrypt side can
//! slice it off without knowing the ciphertext length in advance:
//!
//! +--------------------------------------------------------------------------+
//! | Tag (16 bytes, GHASH)  |  IV (12 bytes, random)  |  Ciphertext (n bytes) |
//! +--------------------------------------------------------------------------+
//!
//! The total overhead per message is 28 bytes (16 tag + 12 IV). This cipher
//! mode offers the following security properties and guarantees:
//!
//! - Confidentiality: AES-256 CTR mode. The keystream block for counter
//!   value `ctr` is `AES_K(IV || ctr)` where `ctr` is a 32-bit big-endian
//!   integer starting at 2 (counter 1 is reserved for the authentication tag
//!   key generation).
//!
//! - Authenticity: GHASH over the ciphertext. The authentication tag is
//!   `GHASH_H(C) XOR E_K(IV || 0^31 || 1)`, where H = AES_K(0^128) and the
//!   XOR mask is the CTR=1 keystream block.
//!
//! - IV uniqueness: The IV is generated internally by [`encrypt`] via
//!   [`get_random_bytes`], eliminating IV-reuse risk at the call site. Callers
//!   must never supply their own IV.
//!
//! - Tag size: Full 128-bit tag. Truncated tags are not supported.
//!
//! Additional Authenticated Data (AAD) is not implemented. The GHASH input
//! is `len(C)` in the final block and the ciphertext only; the AAD length
//! field in the final GHASH block is always zero. If AAD support is needed
//! in the future, it can be added without changing the ciphertext format.

#![allow(dead_code)]

use super::aes::{aes256_encrypt_block, KeySchedule, BLOCK_SIZE, KEY256_SIZE};
use super::super::super::rng::get_random_bytes;
use alloc::vec::Vec;

/// Length of the GCM authentication tag in bytes (128 bits, full GHASH output).
pub const TAG_SIZE: usize = 16;

/// Length of the GCM IV (nonce) in bytes. SP 800-38D recommends 96-bit IVs as
/// the standard form for GCM; they avoid the extra GHASH derivation step
/// required for non-96-bit nonces.
pub const IV_SIZE: usize = 12;

/// Combined byte overhead added to each message: tag + IV.
pub const GCM_OVERHEAD: usize = TAG_SIZE + IV_SIZE;

/// Errors returned by AES-256-GCM operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AesGcmError {
    /// The authentication tag did not match. The ciphertext may have been
    /// tampered with, truncated, or decrypted with the wrong key. The output
    /// buffer is zeroed before this error is returned to prevent accidental
    /// use of unauthenticated plaintext.
    AuthenticationFailure,

    /// The input buffer is shorter than the minimum valid ciphertext length
    /// ([`GCM_OVERHEAD`] bytes), making it impossible to extract a tag and IV.
    CiphertextTooShort,
}

/// GHASH state, holding the running hash value and the pre-computed hash
/// subkey H.
///
/// H is the AES encryption of the all-zero block under the message key:
/// `H = AES_K(0^128)`. It is derived once per encryption/decryption
/// operation and reused for all GHASH multiplications.
struct Ghash {
    /// Running hash accumulator; starts at the all-zero block.
    y: [u8; BLOCK_SIZE],

    /// Hash subkey H = AES_K(0^128).
    h: [u8; BLOCK_SIZE],
}

impl Ghash {
    /// Initializes a new GHASH instance from the hash subkey `H`.
    ///
    /// # Arguments
    ///
    /// * `h` - The 16-byte hash subkey, computed as `AES_K(0^128)`.
    ///
    /// # Returns
    ///
    /// Returns a [`Ghash`] instance with the accumulator initialized to the
    /// all-zero block and the subkey set to `h`.
    fn new(h: [u8; BLOCK_SIZE]) -> Self {
        Self { y: [0u8; BLOCK_SIZE], h }
    }

    /// Updates the GHASH accumulator with one 16-byte block.
    ///
    /// Computes `Y = (Y XOR block) * H` in GF(2^128) with the GCM reduction
    /// polynomial x^128 + x^7 + x^2 + x + 1 (0xe1 << 120).
    ///
    /// # Arguments
    ///
    /// * `block` - The 16-byte input block to fold into the accumulator.
    fn update(&mut self, block: &[u8; BLOCK_SIZE]) {
        // Y = Y XOR block
        for i in 0..BLOCK_SIZE {
            self.y[i] ^= block[i];
        }

        // Y = Y * H in GF(2^128)
        self.y = gf128_mul(&self.y, &self.h);
    }

    /// Feeds an arbitrary-length byte slice into GHASH, padding the final
    /// partial block with zeros if needed.
    ///
    /// # Arguments
    ///
    /// * `data` - Input bytes to process. Any length is accepted; a partial
    ///   final block is zero-padded to 16 bytes before processing.
    fn update_bytes(&mut self, data: &[u8]) {
        let mut chunks = data.chunks_exact(BLOCK_SIZE);
        for chunk in chunks.by_ref() {
            let block: &[u8; BLOCK_SIZE] = chunk.try_into().unwrap();
            self.update(block);
        }

        let remainder = chunks.remainder();
        if !remainder.is_empty() {
            let mut padded = [0u8; BLOCK_SIZE];
            padded[..remainder.len()].copy_from_slice(remainder);
            self.update(&padded);
        }
    }

    /// Gets the current GHASH output (the accumulator value).
    ///
    /// # Returns
    ///
    /// Returns the 16-byte GHASH accumulator as a byte array. Does not reset
    /// the accumulator; call sites should discard the [`Ghash`] instance after
    /// calling this.
    fn finalize(&self) -> [u8; BLOCK_SIZE] {
        self.y
    }
}

/// Multiplies two 128-bit values in GF(2^128) using the GCM reduction
/// polynomial x^128 + x^7 + x^2 + x + 1.
///
/// Implements the "right-to-left comb" bit-serial algorithm. Both operands
/// are treated as big-endian 128-bit integers, consistent with the GCM
/// specification.
///
/// # Arguments
///
/// * `x` - First 128-bit operand, as a big-endian byte array.
/// * `y` - Second 128-bit operand, as a big-endian byte array.
///
/// # Returns
///
/// Returns the 128-bit product `x * y` in GF(2^128), as a big-endian byte
/// array.
fn gf128_mul(x: &[u8; 16], y: &[u8; 16]) -> [u8; 16] {
    // Reduction polynomial: the low 64 bits of x^128 + x^7 + x^2 + x + 1
    // when represented as a 128-bit big-endian value are
    // 0xe100_0000_0000_0000 in the high word. In the right-shift variant
    // used here, the constant is 0xe1 shifted into position.
    const R: u64 = 0xe100_0000_0000_0000u64;

    // Represent X and Y as two u64 words (big-endian: hi = bits 127..64)
    let x_hi = u64::from_be_bytes(x[0..8].try_into().unwrap());
    let x_lo = u64::from_be_bytes(x[8..16].try_into().unwrap());
    let y_hi = u64::from_be_bytes(y[0..8].try_into().unwrap());
    let y_lo = u64::from_be_bytes(y[8..16].try_into().unwrap());

    let mut z_hi: u64 = 0;
    let mut z_lo: u64 = 0;
    let mut v_hi = x_hi;
    let mut v_lo = x_lo;

    // Process each bit of Y from MSB to LSB
    for i in 0..128u32 {
        // Extract bit i of Y (MSB-first: bit 0 is the MSB of y_hi)
        let bit = if i < 64 {
            (y_hi >> (63 - i)) & 1
        }
        else {
            (y_lo >> (127 - i)) & 1
        };

        // Conditionally XOR V into Z
        if bit == 1 {
            z_hi ^= v_hi;
            z_lo ^= v_lo;
        }

        // V = V * x^-1 in GF(2^128): right-shift V by 1, then reduce if the
        // bit shifted out (the LSB before the shift) was 1.
        let carry = v_lo & 1;
        v_lo = (v_lo >> 1) | (v_hi << 63);
        v_hi >>= 1;
        if carry == 1 {
            v_hi ^= R;
        }
    }

    let mut result = [0u8; 16];
    result[0..8].copy_from_slice(&z_hi.to_be_bytes());
    result[8..16].copy_from_slice(&z_lo.to_be_bytes());

    result
}

/// Builds the initial counter block J0 for a 96-bit IV.
///
/// For 96-bit IVs, J0 = IV || 0^31 || 1 (i.e., the IV occupies the first 12
/// bytes, and the counter field is the 32-bit big-endian integer 1 in the last
/// 4 bytes).
///
/// # Arguments
///
/// * `iv` - The 12-byte (96-bit) IV.
///
/// # Returns
///
/// Returns the 16-byte initial counter block J0.
#[inline]
fn make_j0(iv: &[u8; IV_SIZE]) -> [u8; BLOCK_SIZE] {
    let mut j0 = [0u8; BLOCK_SIZE];
    j0[..IV_SIZE].copy_from_slice(iv);
    j0[15] = 0x01;  // Counter = 1, big-endian

    j0
}

/// Increments the 32-bit counter in the last 4 bytes of a counter block,
/// wrapping on overflow. The first 12 bytes (the IV portion) are unchanged.
///
/// # Arguments
///
/// * `block` - The 16-byte counter block to increment in place. Only the
///   last 4 bytes (the big-endian counter field) are modified.
#[inline]
fn inc32(block: &mut [u8; BLOCK_SIZE]) {
    let ctr = u32::from_be_bytes(block[12..16].try_into().unwrap());
    block[12..16].copy_from_slice(&ctr.wrapping_add(1).to_be_bytes());
}

/// Generates a CTR-mode keystream block for a given counter block and key
/// schedule, returning the AES-encrypted counter value.
///
/// The counter block is incremented in place after the keystream block is
/// generated, ready for the next call.
///
/// # Arguments
///
/// * `counter`  - Current counter block (modified in place: incremented after
///   use).
/// * `schedule` - AES-256 key schedule.
///
/// # Returns
///
/// Returns the 16-byte keystream block for the current counter value.
#[inline]
fn ctr_keystream_block(
    counter: &mut [u8; BLOCK_SIZE],
    schedule: &KeySchedule,
) -> [u8; BLOCK_SIZE] {
    let mut block = *counter;
    aes256_encrypt_block(&mut block, schedule);
    inc32(counter);

    block
}

/// XOR-encrypts/decrypts `data` in place using AES-256-CTR, starting from
/// counter block `ctr_block`.
///
/// The starting counter is incremented to CTR=2 before bulk encryption,
/// reserving CTR=1 for the tag mask (per SP 800-38D). The caller passes in
/// a counter block already set to CTR=2.
///
/// CTR mode is its own inverse: the same function encrypts and decrypts.
///
/// # Arguments
///
/// * `data`      - Buffer to encrypt or decrypt in place.
/// * `ctr_block` - Counter block for the first data block (counter = 2).
/// * `schedule`  - AES-256 key schedule.
#[inline]
fn ctr_apply_keystream(
    data: &mut [u8],
    ctr_block: &mut [u8; BLOCK_SIZE],
    schedule: &KeySchedule,
) {
    let mut remaining = data;

    // Full 16-byte blocks
    while remaining.len() >= BLOCK_SIZE {
        let ks = ctr_keystream_block(ctr_block, schedule);
        for i in 0..BLOCK_SIZE {
            remaining[i] ^= ks[i];
        }
        remaining = &mut remaining[BLOCK_SIZE..];
    }

    // Trailing partial block
    if !remaining.is_empty() {
        let ks = ctr_keystream_block(ctr_block, schedule);
        for (i, b) in remaining.iter_mut().enumerate() {
            *b ^= ks[i];
        }
    }
}

/// Computes the GCM authentication tag.
///
/// Implements SP 800-38D paragraph 7.1 step 6:
///
///   1. `H = AES_K(0^128)` — hash subkey.
///   2. GHASH over the ciphertext (no AAD in this implementation).
///   3. Final GHASH block: `len(A) || len(C)` as two 64-bit big-endian
///      integers. Since AAD is always empty, `len(A) = 0`.
///   4. Tag = `GHASH_H(ciphertext, lengths) XOR E_K(J0)`, where J0 is the
///      IV || 0x00000001 counter block (CTR=1 keystream, the tag mask).
///
/// # Arguments
///
/// * `ciphertext` - The fully-encrypted ciphertext bytes.
/// * `j0`         - The initial counter block J0 (IV || counter=1).
/// * `schedule`   - AES-256 key schedule.
///
/// # Returns
///
/// Returns the 16-byte authentication tag.
fn compute_tag(
    ciphertext: &[u8],
    j0: &[u8; BLOCK_SIZE],
    schedule: &KeySchedule,
) -> [u8; TAG_SIZE] {
    // Derive H = AES_K(0^128)
    let mut h_block = [0u8; BLOCK_SIZE];
    aes256_encrypt_block(&mut h_block, schedule);
    let mut ghash = Ghash::new(h_block);

    // GHASH over ciphertext
    ghash.update_bytes(ciphertext);

    // GHASH final length block: len(A) || len(C) in bits, as two 64-bit
    // big-endian integers. len(A) = 0 (no AAD).
    let mut len_block = [0u8; BLOCK_SIZE];
    let c_bit_len = (ciphertext.len() as u64) * 8;
    len_block[8..16].copy_from_slice(&c_bit_len.to_be_bytes());
    ghash.update(&len_block);

    let s = ghash.finalize();

    // Tag mask: E_K(J0), where J0 has counter = 1
    let mut tag_mask = *j0;
    aes256_encrypt_block(&mut tag_mask, schedule);

    // Tag = S XOR E_K(J0)
    let mut tag = [0u8; TAG_SIZE];
    for i in 0..TAG_SIZE {
        tag[i] = s[i] ^ tag_mask[i];
    }

    tag
}

/// Compares two 16-byte slices in constant time, returning `true` if they
/// are equal.
///
/// Accumulates the XOR of all byte pairs into a single `u8` with OR; any
/// non-zero result means at least one byte differed. The comparison visits
/// all 16 bytes regardless of where the first difference occurs, preventing
/// timing-based tag-comparison attacks.
///
/// # Arguments
///
/// * `a` - First 16-byte value to compare.
/// * `b` - Second 16-byte value to compare.
///
/// # Returns
///
/// Returns `true` if `a` and `b` are identical, `false` otherwise.
#[inline]
fn constant_time_eq_16(a: &[u8; TAG_SIZE], b: &[u8; TAG_SIZE]) -> bool {
    let mut diff: u8 = 0;
    for i in 0..TAG_SIZE {
        diff |= a[i] ^ b[i];
    }
    diff == 0
}

/// Encrypts `plaintext` with AES-256-GCM using `key`.
///
/// A fresh 96-bit IV is drawn from the CSPRNG for every call. The output
/// is a single byte buffer with the layout:
///
/// [ tag (16 bytes) | IV (12 bytes) | ciphertext (plaintext.len() bytes) ]
///
/// Total output length is `plaintext.len() + GCM_OVERHEAD` (28 bytes overhead).
///
/// # Arguments
///
/// * `key`       - AES-256 key (32 bytes).
/// * `plaintext` - Data to encrypt. May be empty; the tag still authenticates
///   the key and IV in that case.
///
/// # Returns
///
/// Returns the authenticated ciphertext as an owned `[u8; TAG_SIZE + IV_SIZE]`
/// header plus ciphertext, all in a `Vec<u8>`.
///
/// # Panics
///
/// Panics if `init_cpu` has not been called for the current CPU (required by
/// the CSPRNG for IV generation).
pub fn encrypt(key: &[u8; KEY256_SIZE], plaintext: &[u8]) -> Vec<u8> {
    // Generate a fresh IV via CSPRNG
    let mut iv = [0u8; IV_SIZE];
    get_random_bytes(&mut iv);

    let schedule = KeySchedule::new(key);

    // Build J0 (IV || counter=1)
    let j0 = make_j0(&iv);

    // Starting counter for data encryption is counter=2 (counter=1 is reserved
    // for the tag mask)
    let mut ctr_block = j0;
    inc32(&mut ctr_block);  // Advance to counter=2

    // Allocate output: tag (16) + IV (12) + ciphertext (n)
    let plaintext_len = plaintext.len();
    let mut output = alloc::vec![0u8; TAG_SIZE + IV_SIZE + plaintext_len];

    // Write IV into output[TAG_SIZE..TAG_SIZE+IV_SIZE]
    output[TAG_SIZE..TAG_SIZE + IV_SIZE].copy_from_slice(&iv);

    // Copy plaintext into the ciphertext region, then encrypt in place
    output[TAG_SIZE + IV_SIZE..].copy_from_slice(plaintext);
    ctr_apply_keystream(&mut output[TAG_SIZE + IV_SIZE..], &mut ctr_block,
        &schedule);

    // Compute tag over the ciphertext (not the plaintext)
    let tag = compute_tag(&output[TAG_SIZE + IV_SIZE..], &j0, &schedule);
    output[..TAG_SIZE].copy_from_slice(&tag);

    output
}

/// Decrypts and authenticates an AES-256-GCM ciphertext produced by
/// [`encrypt`].
///
/// Expects the input format `[ tag (16) | IV (12) | ciphertext (n) ]`. The
/// authentication tag is verified in constant time before any plaintext is
/// returned. If verification fails, the output buffer is zeroed and
/// [`AesGcmError::AuthenticationFailure`] is returned, preventing any
/// unauthenticated plaintext from reaching the caller.
///
/// # Arguments
///
/// * `key`        - AES-256 key (32 bytes). Must match the key used during
///   encryption.
/// * `ciphertext` - The authenticated ciphertext, exactly as produced by
///   [`encrypt`].
///
/// # Returns
///
/// `Ok(Vec<u8>)` containing the plaintext on success, or an [`AesGcmError`]
/// on failure.
///
/// # Errors
///
/// - [`AesGcmError::CiphertextTooShort`] if `ciphertext.len() < GCM_OVERHEAD`.
/// - [`AesGcmError::AuthenticationFailure`] if the tag does not match.
pub fn decrypt(
    key: &[u8; KEY256_SIZE],
    ciphertext: &[u8],
) -> Result<Vec<u8>, AesGcmError> {
    if ciphertext.len() < GCM_OVERHEAD {
        return Err(AesGcmError::CiphertextTooShort);
    }

    // Parse input
    let received_tag: &[u8; TAG_SIZE] = ciphertext[..TAG_SIZE]
        .try_into()
        .unwrap();
    let iv: &[u8; IV_SIZE] = ciphertext[TAG_SIZE..TAG_SIZE + IV_SIZE]
        .try_into()
        .unwrap();
    let ct_bytes = &ciphertext[GCM_OVERHEAD..];

    let schedule = KeySchedule::new(key);
    let j0 = make_j0(iv);

    // Recompute tag over the raw ciphertext bytes (before decryption)
    let expected_tag = compute_tag(ct_bytes, &j0, &schedule);

    // Constant-time tag comparison — must happen before decryption
    if !constant_time_eq_16(received_tag, &expected_tag) {
        return Err(AesGcmError::AuthenticationFailure);
    }

    // Tag verified; decrypt the ciphertext
    let mut plaintext = ct_bytes.to_vec();
    let mut ctr_block = j0;
    inc32(&mut ctr_block);  // Advance to counter=2 (matches encryption)
    ctr_apply_keystream(&mut plaintext, &mut ctr_block, &schedule);

    Ok(plaintext)
}
