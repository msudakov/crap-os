//! The SHA-3 512 Basic Hashing Algorithm

#![allow(dead_code)]

/// The Keccak-f[1600] round constants (FIPS 202, Section 3.4, Algorithm 5).
///
/// There are 24 rounds in Keccak-f[1600]. Each constant is derived from a
/// linear feedback shift register and breaks the symmetry of the otherwise
/// fully symmetric Keccak permutation.
#[rustfmt::skip]
const KECCAK_RC: [u64; 24] = [
    0x0000000000000001, 0x0000000000008082, 0x800000000000808a,
    0x8000000080008000, 0x000000000000808b, 0x0000000080000001,
    0x8000000080008081, 0x8000000000008009, 0x000000000000008a,
    0x0000000000000088, 0x0000000080008009, 0x000000008000000a,
    0x000000008000808b, 0x800000000000008b, 0x8000000000008089,
    0x8000000000008003, 0x8000000000008002, 0x8000000000000080,
    0x000000000000800a, 0x800000008000000a, 0x8000000080008081,
    0x8000000000008080, 0x0000000080000001, 0x8000000080008008,
];

/// The Keccak-f[1600] rho rotation offsets (FIPS 202, Section 3.2.2).
///
/// The offset for lane (x, y) is stored at index `x + 5 * y`. The (0,0) lane
/// (index 0) has a rotation of 0 and is omitted from the rho step by
/// convention, but including it here (as 0) simplifies the indexing.
#[rustfmt::skip]
const KECCAK_RHO: [u32; 25] = [
     0,  1, 62, 28, 27,
    36, 44,  6, 55, 20,
     3, 10, 43, 25, 39,
    41, 45, 15, 21,  8,
    18,  2, 61, 56, 14,
];

/// The Keccak-f[1600] pi permutation.
///
/// Maps lane index `i` to its new position after the pi step. Stored as a
/// lookup table to avoid the modular arithmetic on every round. Derived from
/// the pi definition: new position `(x + 3 * y) mod 5, x` where `x = i % 5`,
/// `y = i / 5`.
#[rustfmt::skip]
const KECCAK_PI: [usize; 25] = [
     0, 10, 20,  5, 15,
    16,  1, 11, 21,  6,
     7, 17,  2, 12, 22,
    23,  8, 18,  3, 13,
    14, 24,  9, 19,  4,
];

/// Applies the Keccak-f[1600] permutation to the 25-word (1600-bit) state
/// in place.
///
/// This is the core of SHA-3 / SHAKE. It consists of 24 rounds, each comprising
/// five steps: theta, rho, pi, chi, iota, as defined in FIPS 202, Section 3.
#[inline]
fn keccak_f1600(state: &mut [u64; 25]) {
    for &rc in &KECCAK_RC {
        // Theta
        // Each column's parity is XOR'd into the two adjacent columns.
        let mut c = [0u64; 5];
        for x in 0..5 {
            c[x] = state[x] ^ state[x+5] ^state[x+10] ^state[x+15] ^state[x+20];
        }
        let mut d = [0u64; 5];
        for x in 0..5 {
            d[x] = c[(x + 4) % 5] ^ c[(x + 1) % 5].rotate_left(1);
        }
        for x in 0..5 {
            for y in 0..5 {
                state[x + 5*y] ^= d[x];
            }
        }

        // Rho and pi combined
        // rho rotates each lane by its offset; pi then permutes the lane
        // positions. Combining them avoids a second temporary array.
        let mut temp = [0u64; 25];
        for i in 0..25 {
            temp[KECCAK_PI[i]] = state[i].rotate_left(KECCAK_RHO[i]);
        }

        // Chi
        // Non-linear step: each bit is XOR'd with a function of two
        // neighbours in the same row.
        for y in 0..5 {
            for x in 0..5 {
                state[x + 5*y] = temp[x + 5*y]
                    ^ ((!temp[(x+1)%5 + 5*y]) & temp[(x+2)%5 + 5*y]);
            }
        }

        // Iota
        // XOR the round constant into lane (0, 0) to break round symmetry.
        state[0] ^= rc;
    }
}

/// Computes the SHA3-512 digest of `input` and returns the 512-bit (64-byte)
/// result.
///
/// SHA3-512 is based on the Keccak sponge construction (FIPS 202) and is
/// structurally distinct from the SHA-2 family. It provides:
///   - Collision resistance: 256-bit security level against collision attacks;
///   - Pre-image resistance: 512-bit security level;
///   - Length-extension immunity: the sponge construction absorbs the input
///     into a hidden capacity portion of the state, making length-extension
///     attacks structurally impossible. This is a concrete advantage over
///     SHA-256 when a bare digest (not HMAC) is used as a MAC.
///
/// SHA3-512 produces larger digests than SHA-256 (64 vs 32 bytes) and is
/// somewhat slower on software implementations without hardware acceleration.
/// Prefer it when the highest security margin or length-extension immunity
/// is required.
///
/// # Arguments
///
/// * `input` - The byte slice to hash. May be any length, including empty.
///
/// # Returns
///
/// Returns the 64-byte (512-bit) SHA3-512 digest as a raw byte array, as
/// specified by FIPS 202.
pub fn sha3_512(input: &[u8]) -> [u8; 64] {
    // SHA3-512 parameters (FIPS 202, Section 7):
    //   rate     r = 1600 - 2 * 512 = 576 bits = 72 bytes
    //   capacity c = 2 * 512 = 1024 bits
    //   output   n = 512 bits = 64 bytes
    const RATE: usize = 72;

    // The sponge state is 1600 bits = 25 * 64-bit lanes, stored in
    // little-endian lane order as required by the Keccak spec.
    let mut state = [0u64; 25];

    // Absorb phase
    //
    // Process full-rate (72-byte) blocks. XOR each block into the first
    // RATE bytes of the state (the "rate" portion), then apply the
    // Keccak-f[1600] permutation.
    let mut offset = 0;
    while offset + RATE <= input.len() {
        for i in 0..RATE / 8 {
            state[i] ^= u64::from_le_bytes(
                input[offset + i*8 .. offset + i*8 + 8].try_into().unwrap()
            );
        }
        keccak_f1600(&mut state);
        offset += RATE;
    }

    // Padding (FIPS 202, Section 4, multi-rate padding)
    //
    // SHA-3 uses the domain suffix 0x06 followed by standard Keccak padding
    // (0x80 at the last byte of the rate block). This distinguishes SHA-3
    // from raw Keccak (which uses 0x01) and SHAKE (which uses 0x1f).
    let mut last_block = [0u8; RATE];
    let remaining = &input[offset..];
    last_block[..remaining.len()].copy_from_slice(remaining);
    last_block[remaining.len()] = 0x06;  // SHA-3 domain separation byte
    last_block[RATE - 1] |= 0x80;        // Final padding bit

    for i in 0..RATE / 8 {
        state[i] ^= u64::from_le_bytes(
            last_block[i*8 .. i*8 + 8].try_into().unwrap()
        );
    }

    // Squeeze phase
    //
    // Apply the final permutation, then read the first 64 bytes of the state
    // as the digest. Since RATE (72) > output size (64), a single squeeze
    // block is sufficient.
    keccak_f1600(&mut state);

    let mut digest = [0u8; 64];
    for i in 0..8 {
        digest[i*8 .. i*8+8].copy_from_slice(&state[i].to_le_bytes());
    }

    digest
}
