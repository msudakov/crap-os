//! The MD5 Basic Hashing Algorithm
//! 
//! # Security Warning
//! 
//! MD5 is included solely for non-security checksum use cases. It must never
//! be used in any new security-sensitive context.

/// Per-round shift amounts, as specified in RFC 1321, Section 3.4.
/// Indexed as `MD5_SHIFTS[round]` where `round` is 0..63.
#[rustfmt::skip]
const MD5_SHIFTS: [u32; 64] = [
    7, 12, 17, 22,  7, 12, 17, 22,  7, 12, 17, 22,  7, 12, 17, 22,
    5,  9, 14, 20,  5,  9, 14, 20,  5,  9, 14, 20,  5,  9, 14, 20,
    4, 11, 16, 23,  4, 11, 16, 23,  4, 11, 16, 23,  4, 11, 16, 23,
    6, 10, 15, 21,  6, 10, 15, 21,  6, 10, 15, 21,  6, 10, 15, 21,
];

/// Precomputed table of `floor(2^32 * abs(sin(i+1)))` for i in 0..63.
/// Defined in RFC 1321, Section 3.4.
#[rustfmt::skip]
const MD5_K: [u32; 64] = [
    0xd76aa478, 0xe8c7b756, 0x242070db, 0xc1bdceee,
    0xf57c0faf, 0x4787c62a, 0xa8304613, 0xfd469501,
    0x698098d8, 0x8b44f7af, 0xffff5bb1, 0x895cd7be,
    0x6b901122, 0xfd987193, 0xa679438e, 0x49b40821,
    0xf61e2562, 0xc040b340, 0x265e5a51, 0xe9b6c7aa,
    0xd62f105d, 0x02441453, 0xd8a1e681, 0xe7d3fbc8,
    0x21e1cde6, 0xc33707d6, 0xf4d50d87, 0x455a14ed,
    0xa9e3e905, 0xfcefa3f8, 0x676f02d9, 0x8d2a4c8a,
    0xfffa3942, 0x8771f681, 0x6d9d6122, 0xfde5380c,
    0xa4beea44, 0x4bdecfa9, 0xf6bb4b60, 0xbebfbc70,
    0x289b7ec6, 0xeaa127fa, 0xd4ef3085, 0x04881d05,
    0xd9d4d039, 0xe6db99e5, 0x1fa27cf8, 0xc4ac5665,
    0xf4292244, 0x432aff97, 0xab9423a7, 0xfc93a039,
    0x655b59c3, 0x8f0ccc92, 0xffeff47d, 0x85845dd1,
    0x6fa87e4f, 0xfe2ce6e0, 0xa3014314, 0x4e0811a1,
    0xf7537e82, 0xbd3af235, 0x2ad7d2bb, 0xeb86d391,
];

/// Computes the MD5 digest of `input` and returns the 128-bit (16-byte) result.
///
/// # Security warning
///
/// MD5 is cryptographically broken and must not be used for any
/// security-sensitive purpose. Practical collision attacks exist, and MD5 is
/// also vulnerable to length-extension attacks.
///
/// Acceptable uses:
///   - Non-security checksums (detecting accidental data corruption);
///   - Interoperability with legacy systems that require MD5;
///   - Fingerprinting where collision resistance is not required.
///
/// # Arguments
///
/// * `input` - The byte slice to hash. May be any length, including empty.
///
/// # Returns
///
/// Returns the 16-byte (128-bit) MD5 digest as a raw byte array, in
/// little-endian word order as specified by RFC 1321.
pub fn md5(input: &[u8]) -> [u8; 16] {
    // Initial hash state (RFC 1321, Section 3.3). These are the low-order
    // bytes of the fractional parts of the square roots of the first primes,
    // stored in little-endian order.
    let mut a0: u32 = 0x67452301;
    let mut b0: u32 = 0xefcdab89;
    let mut c0: u32 = 0x98badcfe;
    let mut d0: u32 = 0x10325476;

    // Pre-processing: padding (RFC 1321, Section 3.1 & 3.2)
    //
    // The message is padded to a length congruent to 448 bits (mod 512),
    // then the original bit length is appended as a 64-bit little-endian
    // integer, producing a total padded length that is a multiple of 512 bits
    // (64 bytes).
    //
    // Padding: append a single 0x80 byte, then as many 0x00 bytes as needed.
    let bit_len = (input.len() as u64).wrapping_mul(8);
    let mut msg: alloc::vec::Vec<u8> = alloc::vec::Vec::with_capacity(
        input.len() + 64
    );
    msg.extend_from_slice(input);
    msg.push(0x80);
    while msg.len() % 64 != 56 {
        msg.push(0x00);
    }
    // Append original bit length as a 64-bit little-endian integer
    msg.extend_from_slice(&bit_len.to_le_bytes());

    // Process each 512-bit (64-byte) block
    for block in msg.chunks_exact(64) {
        // Interpret the block as 16 little-endian u32 words
        let mut m = [0u32; 16];
        for (i, word) in m.iter_mut().enumerate() {
            *word = u32::from_le_bytes(block[i*4..i*4+4].try_into().unwrap());
        }

        let (mut a, mut b, mut c, mut d) = (a0, b0, c0, d0);

        for i in 0u32..64 {
            let (f, g) = match i {
                0..=15  => ((b & c) | (!b & d), i),
                16..=31 => ((d & b) | (!d & c), (5 * i + 1) % 16),
                32..=47 => (b ^ c ^ d, (3 * i + 5) % 16),
                _       => (c ^ (b | !d), (7 * i) % 16),
            };

            let temp = d;
            d = c;
            c = b;
            b = b.wrapping_add(
                (a.wrapping_add(f)
                  .wrapping_add(MD5_K[i as usize])
                  .wrapping_add(m[g as usize]))
                .rotate_left(MD5_SHIFTS[i as usize])
            );
            a = temp;
        }

        a0 = a0.wrapping_add(a);
        b0 = b0.wrapping_add(b);
        c0 = c0.wrapping_add(c);
        d0 = d0.wrapping_add(d);
    }

    // Produce the 16-byte digest in little-endian word order
    let mut digest = [0u8; 16];
    digest[0..4].copy_from_slice(&a0.to_le_bytes());
    digest[4..8].copy_from_slice(&b0.to_le_bytes());
    digest[8..12].copy_from_slice(&c0.to_le_bytes());
    digest[12..16].copy_from_slice(&d0.to_le_bytes());

    digest
}
