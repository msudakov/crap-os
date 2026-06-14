//! The HMAC-SHA-256 Algorithm

#![allow(dead_code)]

use super::super::sha256;

/// Computes the HMAC-SHA-256 of `message` under `key` and returns the
/// 256-bit (32-byte) authentication tag.
///
/// HMAC is defined in RFC 2104 and standardised for SHA-256 in FIPS 198-1.
/// The construction is:
///
/// 
/// HMAC(K, m) = SHA256((K' XOR opad) || SHA256((K' XOR ipad) || m))
/// 
///
/// where `||` is concatenation, `K'` is the key block-normalised to 64 bytes:
///   - if `|K| >  64`: `K' = SHA256(K)` zero-padded to 64 bytes;
///   - if `|K| <= 64`: `K'` is `K` zero-padded to 64 bytes.
///
/// Security properties:
///   - Provides message integrity and authentication under a shared secret.
///   - Secure against length-extension attacks (unlike bare SHA-256).
///   - Security level ≈ min(|K|, 256) bits, assuming a secret key.
///
/// # Arguments
///
/// * `key`     - The secret key. Any length is accepted; keys longer than
///               64 bytes are hashed down to 32 bytes first (per RFC 2104 §2).
///               For full 256-bit security, use a 32-byte (256-bit) key.
/// * `message` - The byte slice to authenticate. May be any length, including
///               empty.
///
/// # Returns
///
/// Returns the 32-byte (256-bit) HMAC-SHA-256 tag as a raw byte array.
pub fn hmac_sha256(key: &[u8], message: &[u8]) -> [u8; 32] {
    const BLOCK_LEN: usize = 64;

    // Normalise the key to exactly BLOCK_LEN bytes.
    //
    // Keys longer than the block size are hashed; shorter keys are
    // zero-padded. The result is stored as a fixed-size array, so no heap
    // allocation is required beyond what sha256() already needs internally.
    let mut k_prime = [0u8; BLOCK_LEN];
    if key.len() > BLOCK_LEN {
        // Hash the oversized key down to 32 bytes, then zero-pad to 64
        let hashed = sha256(key);
        k_prime[..32].copy_from_slice(&hashed);
        // k_prime[32..] is already zeroed
    }
    else {
        k_prime[..key.len()].copy_from_slice(key);
        // k_prime[key.len()..] is already zeroed
    }

    // Derive the inner and outer padded keys.
    //
    // ipad = 0x36 repeated; opad = 0x5C repeated.
    // XOR k_prime with each pad in a single pass to keep stack usage flat.
    let mut k_ipad = [0u8; BLOCK_LEN];
    let mut k_opad = [0u8; BLOCK_LEN];
    for i in 0..BLOCK_LEN {
        k_ipad[i] = k_prime[i] ^ 0x36;
        k_opad[i] = k_prime[i] ^ 0x5C;
    }

    // Inner hash: SHA256((K' XOR ipad) || message).
    //
    // Concatenate on the heap (same pattern as sha256's own padding step).
    let mut inner_input: alloc::vec::Vec<u8> =
        alloc::vec::Vec::with_capacity(BLOCK_LEN + message.len());
    inner_input.extend_from_slice(&k_ipad);
    inner_input.extend_from_slice(message);
    let inner_hash = sha256(&inner_input);

    // Outer hash: SHA256((K' XOR opad) || inner_hash)
    let mut outer_input: alloc::vec::Vec<u8> =
        alloc::vec::Vec::with_capacity(BLOCK_LEN + 32);
    outer_input.extend_from_slice(&k_opad);
    outer_input.extend_from_slice(&inner_hash);

    let tag = sha256(&outer_input);

    // Scrub key material from the stack.
    //
    // k_prime, k_ipad, and k_opad contain key-derived data and must be
    // cleared before they go out of scope. Rust does not guarantee that
    // dropping a value zeroes its backing memory, and without an explicit
    // wipe the bytes may linger in the stack frame and be readable by
    // subsequent code or a memory-safety violation elsewhere.
    //
    // We use `write_volatile` to prevent the compiler from eliding the
    // stores as "dead writes": because the arrays are never read again,
    // a plain assignment would be optimised away entirely.
    for byte in k_prime.iter_mut()
              .chain(k_ipad.iter_mut())
              .chain(k_opad.iter_mut())
    {
        unsafe { core::ptr::write_volatile(byte, 0u8) };
    }

    tag
}
