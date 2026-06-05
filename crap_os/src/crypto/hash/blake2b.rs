//! BLAKE2b Cryptographic Hash and MAC Function
//!
//! This module implements BLAKE2b as specified in RFC 7693. BLAKE2b is a
//! cryptographic hash function optimized for 64-bit platforms. It is used as
//! the internal primitive for Argon2id password hashing.
//!
//! BLAKE2b produces digests of any length from 1 to 64 bytes. Common output
//! lengths and their typical use cases:
//!
//!   | Output length | Security level | Typical use                          |
//!   |---------------|----------------|--------------------------------------|
//!   | 32 bytes      | 128-bit        | General purpose, key derivation      |
//!   | 48 bytes      | 192-bit        | When SHA-384 compatibility is needed |
//!   | 64 bytes      | 256-bit        | Maximum security, Argon2id seed      |
//!
//! BLAKE2b supports an optional key of 1–64 bytes, turning it into a
//! Message Authentication Code (MAC) without any additional construction.
//! Unlike HMAC, the key is incorporated directly into the initial hash state,
//! making keyed BLAKE2b slightly faster than HMAC-SHA256 while providing
//! equivalent security. Use [`blake2b`] with a `Some(key)` argument for MAC
//! computation.
//!
//! Warning: Keyed BLAKE2b is not the same as HMAC-BLAKE2b!
//!
//! BLAKE2b has the following security properties:
//!   - Collision resistance: 2^(out_len * 4) operations (128-bit for 32-byte
//!     output, 256-bit for 64-byte output)
//!   - Pre-image resistance: 2^(out_len * 8) operations
//!   - Length-extension immune: BLAKE2b's finalization flag prevents the
//!     length-extension attacks that affect SHA-2
//!   - Side-channel considerations: this implementation does not use
//!     hardware-specific constant-time guarantees; avoid branching on secret
//!     data in callers

/// BLAKE2b initialization vector.
///
/// These are the same as the SHA-512 IV: the first 64 bits of the fractional
/// parts of the square roots of the first eight prime numbers.
const IV: [u64; 8] = [
    0x6a09e667f3bcc908, 0xbb67ae8584caa73b, 0x3c6ef372fe94f82b,
    0xa54ff53a5f1d36f1, 0x510e527fade682d1, 0x9b05688c2b3e6c1f,
    0x1f83d9abfb41bd6b, 0x5be0cd19137e2179,
];

/// BLAKE2b sigma permutation table.
///
/// Each of the 12 rounds uses a different permutation of the 16 message word
/// indices. Rounds 10 and 11 reuse the permutations from rounds 0 and 1.
#[rustfmt::skip]
const SIGMA: [[usize; 16]; 12] = [
    [ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15],
    [14, 10,  4,  8,  9, 15, 13,  6,  1, 12,  0,  2, 11,  7,  5,  3],
    [11,  8, 12,  0,  5,  2, 15, 13, 10, 14,  3,  6,  7,  1,  9,  4],
    [ 7,  9,  3,  1, 13, 12, 11, 14,  2,  6,  5, 10,  4,  0, 15,  8],
    [ 9,  0,  5,  7,  2,  4, 10, 15, 14,  1, 11, 12,  6,  8,  3, 13],
    [ 2, 12,  6, 10,  0, 11,  8,  3,  4, 13,  7,  5, 15, 14,  1,  9],
    [12,  5,  1, 15, 14, 13,  4, 10,  0,  7,  6,  3,  9,  2,  8, 11],
    [13, 11,  7, 14, 12,  1,  3,  9,  5,  0, 15,  4,  8,  6,  2, 10],
    [ 6, 15, 14,  9, 11,  3,  0,  8, 12,  2, 13,  7,  1,  4, 10,  5],
    [10,  2,  8,  4,  7,  6,  1,  5, 15, 11,  9, 14,  3, 12, 13,  0],
    [ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15],
    [14, 10,  4,  8,  9, 15, 13,  6,  1, 12,  0,  2, 11,  7,  5,  3],
];

/// Maximum key length in bytes.
const MAX_KEY_LEN: usize = 64;

/// Maximum output (digest) length in bytes.
const MAX_OUT_LEN: usize = 64;

/// BLAKE2b block size in bytes.
const BLOCK_SIZE: usize = 128;

/// Internal BLAKE2b hash state.
///
/// Holds the 8-word chaining value, the two-word counter, the finalization
/// flags, and a partial input block buffer. Exposed only through the public
/// API functions; callers never construct this directly.
struct Blake2bState {
    /// Chaining value: 8 * 64-bit words, initialized from IV XOR parameter
    /// block and updated after each compression.
    h: [u64; 8],

    /// Message byte counter: total bytes absorbed so far, split into low and
    /// high 64-bit words. BLAKE2b supports inputs up to 2^128 bytes.
    t: [u64; 2],

    /// Finalization flags. `f[0]` is set to `u64::MAX` on the last block.
    /// `f[1]` is used for tree hashing (not needed here, always 0).
    f: [u64; 2],

    /// Partial block buffer. Filled as input arrives; compressed when full
    /// or when `finalize` is called.
    buf: [u8; BLOCK_SIZE],

    /// Number of bytes currently in `buf`.
    buf_len: usize,

    /// Output length in bytes (1–64). Stored to produce the correctly
    /// truncated digest in `finalize`.
    out_len: usize,
}

impl Blake2bState {
    /// Initializes a new BLAKE2b state.
    ///
    /// Constructs the parameter block, XORs it with the IV to produce the
    /// initial chaining value, and optionally absorbs the key block as the
    /// first input block.
    ///
    /// # Arguments
    ///
    /// * `out_len` - Desired digest length in bytes (1–64).
    /// * `key`     - Optional key slice (1–64 bytes). `None` for unkeyed mode.
    fn new(out_len: usize, key: Option<&[u8]>) -> Self {
        let key_len = key.map_or(0, |k| k.len());

        // Parameter block p0.
        // For sequential hashing (our use case), only the first word is
        // non-trivial; all others default to zero or their fan-out/depth
        // defaults.
        //
        // Byte layout of the first parameter word:
        //   [7:0]   digest_length  (out_len as u8)
        //   [15:8]  key_length     (key_len as u8)
        //   [23:16] fanout         (1 for sequential)
        //   [31:24] depth          (1 for sequential)
        //   [63:32] leaf_length    (0 for sequential)
        let p0: u64 = (out_len  as u64)
                    | ((key_len as u64) << 8)
                    | (1u64              << 16)   // fanout = 1
                    | (1u64              << 24);  // depth  = 1

        // XOR the first IV word with p0; all others remain as-is since the
        // remaining parameter words are all zero for sequential hashing.
        let mut h = IV;
        h[0] ^= p0;

        let mut state = Blake2bState {
            h,
            t: [0u64; 2],
            f: [0u64; 2],
            buf: [0u8; BLOCK_SIZE],
            buf_len: 0,
            out_len,
        };

        // If a key is provided, pad it to a full block and absorb it as the
        // first input block.
        if let Some(k) = key {
            let mut key_block = [0u8; BLOCK_SIZE];
            key_block[..k.len()].copy_from_slice(k);
            state.update(&key_block);
        }

        state
    }

    /// Absorbs `input` bytes into the hash state.
    ///
    /// Fills the internal buffer and compresses full blocks as they become
    /// available. Partial final blocks are held in the buffer until
    /// `finalize` is called.
    /// 
    /// # Arguments
    ///
    /// * `input` - Buffer of bytes to absorb into the hash state.
    fn update(&mut self, input: &[u8]) {
        let mut offset = 0;

        while offset < input.len() {
            // If the buffer is full and there is more input, compress it now.
            // We must not compress the last block here, as it must be
            // compressed with the finalization flag set, which only `finalize`
            // does.
            if self.buf_len == BLOCK_SIZE {
                self.increment_counter(BLOCK_SIZE as u64);
                self.compress();
                self.buf_len = 0;
            }

            // Copy as much input as fits into the buffer
            let space   = BLOCK_SIZE - self.buf_len;
            let to_copy = (input.len() - offset).min(space);
            self.buf[self.buf_len..self.buf_len + to_copy]
                .copy_from_slice(&input[offset..offset + to_copy]);
            self.buf_len += to_copy;
            offset       += to_copy;
        }
    }

    /// Finalizes the hash, pads the last block with zeros, sets the
    /// finalization flag, compresses once, and extracts `out_len` bytes from
    /// the resulting chaining value.
    /// 
    /// # Returns
    /// 
    /// Returns the digest as a `Vec<u8>` of `out_len` bytes.
    fn finalize(mut self) -> alloc::vec::Vec<u8> {
        // Increment the counter by the number of bytes remaining in the buffer
        // (the last, possibly partial, block). Zero-pad the rest of the buffer.
        self.increment_counter(self.buf_len as u64);
        for i in self.buf_len..BLOCK_SIZE {
            self.buf[i] = 0;
        }

        // Set the last-block finalization flag (f[0] = all ones)
        self.f[0] = u64::MAX;
        self.compress();

        // Serialize the chaining value to bytes (little-endian) and truncate
        // to the requested output length.
        let mut out = alloc::vec![0u8; self.out_len];
        let mut all_bytes = [0u8; 64];
        for (i, &word) in self.h.iter().enumerate() {
            all_bytes[i*8..i*8+8].copy_from_slice(&word.to_le_bytes());
        }
        out.copy_from_slice(&all_bytes[..self.out_len]);
        out
    }

    /// Increments the 128-bit byte counter by `n`.
    /// 
    /// # Arguments
    ///
    /// * `n` - The number to increment the counter by, with wrapping.
    #[inline]
    fn increment_counter(&mut self, n: u64) {
        self.t[0] = self.t[0].wrapping_add(n);
        if self.t[0] < n {
            self.t[1] = self.t[1].wrapping_add(1);
        }
    }

    /// Applies the BLAKE2b compression function to the current buffer.
    ///
    /// Constructs the 16-word message schedule from the 128-byte buffer,
    /// initializes the 16-word working vector from the current state, runs 12
    /// rounds of G mixing, then folds the result back into the chaining value.
    fn compress(&mut self) {
        // Interpret the 128-byte buffer as 16 little-endian u64 message words
        let mut m = [0u64; 16];
        for (i, word) in m.iter_mut().enumerate() {
            *word = u64::from_le_bytes(
                self.buf[i*8..i*8+8].try_into().unwrap()
            );
        }

        // Initialize the 16-word working vector `v`.
        //   v[0..7]  = current chaining value h[0..7]
        //   v[8..15] = IV, with v[12] and v[13] XOR'd with the counter,
        //              and v[14] XOR'd with the finalization flag.
        let mut v = [0u64; 16];
        v[..8].copy_from_slice(&self.h);
        v[8..16].copy_from_slice(&IV);
        v[12] ^= self.t[0];
        v[13] ^= self.t[1];
        v[14] ^= self.f[0];
        v[15] ^= self.f[1];

        // 12 rounds of 8 G-function calls each
        for round in 0..12 {
            let s = &SIGMA[round];

            // Column step
            g(&mut v, 0, 4,  8, 12, m[s[ 0]], m[s[ 1]]);
            g(&mut v, 1, 5,  9, 13, m[s[ 2]], m[s[ 3]]);
            g(&mut v, 2, 6, 10, 14, m[s[ 4]], m[s[ 5]]);
            g(&mut v, 3, 7, 11, 15, m[s[ 6]], m[s[ 7]]);

            // Diagonal step
            g(&mut v, 0, 5, 10, 15, m[s[ 8]], m[s[ 9]]);
            g(&mut v, 1, 6, 11, 12, m[s[10]], m[s[11]]);
            g(&mut v, 2, 7,  8, 13, m[s[12]], m[s[13]]);
            g(&mut v, 3, 4,  9, 14, m[s[14]], m[s[15]]);
        }

        // Fold the working vector back into the chaining value.
        // h[i] ^= v[i] ^ v[i+8] for i in 0..8.
        for i in 0..8 {
            self.h[i] ^= v[i] ^ v[i + 8];
        }
    }
}

/// The BLAKE2b G mixing function.
///
/// Mixes two message words `x` and `y` into four words of the working vector
/// at positions `a`, `b`, `c`, `d`. The rotation constants (32, 24, 16, 63)
/// are specific to BLAKE2b and differ from BLAKE2s.
/// 
/// # Arguments
///
/// * `v` - The working vector.
/// * `a` - The first word position of the working vector.
/// * `b` - The second word position of the working vector.
/// * `c` - The third word position of the working vector.
/// * `d` - The fourth word position of the working vector.
/// * `x` - The first word to mix.
/// * `y` - The second word to mix.
#[inline(always)]
fn g(
    v: &mut [u64; 16],
    a: usize,
    b: usize,
    c: usize,
    d: usize,
    x: u64,
    y: u64
) {
    v[a] = v[a].wrapping_add(v[b]).wrapping_add(x);
    v[d] = (v[d] ^ v[a]).rotate_right(32);
    v[c] = v[c].wrapping_add(v[d]);
    v[b] = (v[b] ^ v[c]).rotate_right(24);
    v[a] = v[a].wrapping_add(v[b]).wrapping_add(y);
    v[d] = (v[d] ^ v[a]).rotate_right(16);
    v[c] = v[c].wrapping_add(v[d]);
    v[b] = (v[b] ^ v[c]).rotate_right(63);
}

// =============================================================================
// Public API
// =============================================================================

/// Computes a BLAKE2b digest of variable output length, with an optional key.
///
/// This is the primary entry point for all BLAKE2b operations. All other
/// functions in this module are convenience wrappers around this one.
///
/// # Arguments
///
/// * `input`   - The byte slice to hash. May be any length, including empty.
/// * `out_len` - Desired output length in bytes. Must be in 1..=64.
/// * `key`     - Optional key for MAC mode. If `Some`, must be 1..=64 bytes.
///               Pass `None` for unkeyed hashing.
///
/// # Returns
///
/// Returns a `Vec<u8>` of exactly `out_len` bytes containing the digest,
/// or `None` if `out_len` is 0 or > 64, or if the key length is 0 or > 64.
pub fn blake2b(
    input:   &[u8],
    out_len: usize,
    key:     Option<&[u8]>,
) -> Option<alloc::vec::Vec<u8>> {
    if out_len == 0 || out_len > MAX_OUT_LEN {
        return None;
    }
    if let Some(k) = key {
        if k.is_empty() || k.len() > MAX_KEY_LEN {
            return None;
        }
    }

    let mut state = Blake2bState::new(out_len, key);
    state.update(input);
    Some(state.finalize())
}

/// Computes an unkeyed BLAKE2b-512 digest (64-byte output).
///
/// Convenience wrapper for the most common use case. Used internally by
/// Argon2id for the H0 seed and H' variable-length hash.
///
/// # Arguments
///
/// * `input` - The byte slice to hash.
///
/// # Returns
///
/// Returns the 64-byte BLAKE2b-512 digest as a fixed-size array.
pub fn blake2b_512(input: &[u8]) -> [u8; 64] {
    let mut state = Blake2bState::new(64, None);
    state.update(input);
    let out = state.finalize();
    let mut result = [0u8; 64];
    result.copy_from_slice(&out);

    result
}

/// Identical to [`blake2b_512`] but accepts a slice reference.
///
/// Provided as a named alias, so Argon2id's `h_prime_output` function can call
/// it uniformly regardless of whether the input is a fixed array or a slice.
///
/// # Arguments
///
/// * `input` - The byte slice to hash.
///
/// # Returns
///
/// Returns the 64-byte BLAKE2b-512 digest as a fixed-size array.
pub fn blake2b_512_slice(input: &[u8]) -> [u8; 64] {
    blake2b_512(input)
}

/// Computes an unkeyed BLAKE2b digest of a variable output length.
///
/// Used by Argon2id's H' function to produce output blocks of arbitrary
/// length during memory initialization and digest extraction.
///
/// # Arguments
///
/// * `input`   - The byte slice to hash.
/// * `out_len` - Desired output length in bytes (1–64). Lengths outside this
///               range are clamped: 0 is treated as 1, and values above 64
///               are treated as 64. This clamping behaviour is intentional
///               for the Argon2id H' use case; other callers should use
///               `blake2b` directly if they need an error on invalid lengths.
///
/// # Returns
///
/// Returns a `Vec<u8>` of exactly `out_len` bytes.
pub fn blake2b_variable(input: &[u8], out_len: usize) -> alloc::vec::Vec<u8> {
    let clamped = out_len.clamp(1, MAX_OUT_LEN);
    let mut state = Blake2bState::new(clamped, None);
    state.update(input);

    state.finalize()
}

/// Computes an unkeyed BLAKE2b-256 digest (32-byte output).
///
/// A convenient choice for general-purpose hashing where 128-bit collision
/// resistance is sufficient and a 32-byte output is preferred (e.g., hash
/// table keys, content-addressed identifiers).
///
/// # Arguments
///
/// * `input` - The byte slice to hash.
///
/// # Returns
///
/// Returns the 32-byte BLAKE2b-256 digest as a fixed-size array.
pub fn blake2b_256(input: &[u8]) -> [u8; 32] {
    let mut state = Blake2bState::new(32, None);
    state.update(input);
    let out = state.finalize();

    let mut result = [0u8; 32];
    result.copy_from_slice(&out);

    result
}

/// Computes a keyed BLAKE2b-512 MAC (64-byte output).
///
/// Equivalent to HMAC-SHA512 in security level, but faster and simpler. The
/// key is incorporated directly into the initial hash state rather than being
/// processed in two HMAC passes.
///
/// # Arguments
///
/// * `input` - The message bytes to authenticate.
/// * `key`   - The MAC key. Must be 1–64 bytes. A key of exactly 32 bytes
///             provides 256-bit MAC security; a 64-byte key provides 512-bit.
///
/// # Returns
///
/// Returns `Some([u8; 64])` containing the 64-byte MAC on success, or `None`
/// if the key is empty or longer than 64 bytes.
pub fn blake2b_mac_512(input: &[u8], key: &[u8]) -> Option<[u8; 64]> {
    if key.is_empty() || key.len() > MAX_KEY_LEN {
        return None;
    }

    let mut state = Blake2bState::new(64, Some(key));
    state.update(input);
    let out = state.finalize();

    let mut result = [0u8; 64];
    result.copy_from_slice(&out);

    Some(result)
}

/// Computes a keyed BLAKE2b-256 MAC (32-byte output).
///
/// A compact MAC suitable for message authentication where a 32-byte tag is
/// preferred. Provides 128-bit MAC security with a sufficiently long key.
///
/// # Arguments
///
/// * `input` - The message bytes to authenticate.
/// * `key`   - The MAC key; must be 1–64 bytes.
///
/// # Returns
///
/// Returns `Some([u8; 32])` containing the 32-byte MAC on success, or `None`
/// if the key is empty or longer than 64 bytes.
pub fn blake2b_mac_256(input: &[u8], key: &[u8]) -> Option<[u8; 32]> {
    if key.is_empty() || key.len() > MAX_KEY_LEN {
        return None;
    }

    let mut state = Blake2bState::new(32, Some(key));
    state.update(input);
    let out = state.finalize();

    let mut result = [0u8; 32];
    result.copy_from_slice(&out);

    Some(result)
}
