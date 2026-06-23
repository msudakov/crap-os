//! The SHA-1 Basic Hashing Algorithm
//! 
//! # Security Warning
//! 
//! SHA-1 is included solely for non-security checksum use cases. It must never
//! be used in any new security-sensitive context.

#![allow(dead_code)]

/// Computes the SHA-1 digest of `input` and returns the 160-bit (20-byte)
/// result.
///
/// # Security warning
///
/// SHA-1 is cryptographically broken and must not be used for any
/// security-sensitive purpose. Practical chosen-prefix collision attacks have
/// been demonstrated, and SHA-1 is also vulnerable to length-extension attacks.
/// 
/// Acceptable uses:
///   - Non-security checksums
///   - Interoperability with legacy protocols that mandate SHA-1 (e.g., older
///     TLS cipher suites, legacy SSH key fingerprints);
///   - HMAC-SHA1 in non-security contexts (HMAC partially mitigates the
///     collision weakness, but the construction is still deprecated).
///
/// # Arguments
///
/// * `input` - The byte slice to hash. May be any length, including empty.
///
/// # Returns
///
/// Returns the 20-byte (160-bit) SHA-1 digest as a raw byte array in big-endian
/// word order, as specified by FIPS 180-4.
pub fn sha1(input: &[u8]) -> [u8; 20] {
    // Initial hash values (FIPS 180-4, Section 5.3.1).
    let mut h: [u32; 5] = [
        0x67452301, 0xefcdab89, 0x98badcfe, 0x10325476, 0xc3d2e1f0,
    ];

    // Pre-processing: padding (FIPS 180-4, Section 5.1.1)
    //
    // Append 0x80, then zero bytes until length ≡ 56 (mod 64), then the
    // original bit length as a 64-bit big-endian integer.
    let bit_len = (input.len() as u64).wrapping_mul(8);
    let mut msg: alloc::vec::Vec<u8> = alloc::vec::Vec::with_capacity(
        input.len() + 64
    );
    msg.extend_from_slice(input);
    msg.push(0x80);
    while msg.len() % 64 != 56 {
        msg.push(0x00);
    }
    msg.extend_from_slice(&bit_len.to_be_bytes());

    // Process each 512-bit (64-byte) block
    for block in msg.chunks_exact(64) {
        // Expand the 16 message words into 80 words (message schedule)
        let mut w = [0u32; 80];
        for i in 0..16 {
            w[i] = u32::from_be_bytes(block[i*4..i*4+4].try_into().unwrap());
        }
        for i in 16..80 {
            w[i] = (w[i-3] ^ w[i-8] ^ w[i-14] ^ w[i-16]).rotate_left(1);
        }

        let (mut a, mut b, mut c, mut d, mut e) =
            (h[0], h[1], h[2], h[3], h[4]);

        for i in 0..80 {
            let (f, k) = match i {
                0..=19  => ((b & c) | (!b & d), 0x5a827999u32),
                20..=39 => (b ^ c ^ d, 0x6ed9eba1u32),
                40..=59 => ((b & c) | (b & d) | (c & d), 0x8f1bbcdcu32),
                _       => (b ^ c ^ d, 0xca62c1d6u32),
            };

            let temp = a.rotate_left(5)
                .wrapping_add(f)
                .wrapping_add(e)
                .wrapping_add(k)
                .wrapping_add(w[i]);
            e = d;
            d = c;
            c = b.rotate_left(30);
            b = a;
            a = temp;
        }

        h[0] = h[0].wrapping_add(a);
        h[1] = h[1].wrapping_add(b);
        h[2] = h[2].wrapping_add(c);
        h[3] = h[3].wrapping_add(d);
        h[4] = h[4].wrapping_add(e);
    }

    // Produce the 20-byte digest in big-endian word order
    let mut digest = [0u8; 20];
    for (i, word) in h.iter().enumerate() {
        digest[i*4..i*4+4].copy_from_slice(&word.to_be_bytes());
    }

    digest
}
