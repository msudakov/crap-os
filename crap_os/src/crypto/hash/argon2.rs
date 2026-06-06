//! Argon2id Password Hashing
//!
//! This module implements the Argon2id variant of the Argon2 memory-hard
//! password hashing function, as specified in RFC 9106. Argon2id is the
//! recommended choice for password hashing by OWASP, NIST SP 800-63B, and the
//! Password Hashing Competition (PHC).
//!
//! Argon2id is a hybrid of two Argon2 variants:
//!
//!   - Argon2d fills memory blocks using data-dependent addressing, providing
//!     strong GPU resistance but some vulnerability to side-channel timing
//!     attacks.
//!   - Argon2i uses data-independent addressing, providing side-channel
//!     resistance at the cost of slightly weaker GPU resistance.
//!   - Argon2id uses Argon2i addressing for the first half of the first pass,
//!     and Argon2d addressing for all subsequent passes. This hybrid provides
//!     both side-channel resistance and strong GPU/ASIC resistance, and is the
//!     recommended variant for general password hashing.
//!
//! The algorithm is parameterised by three cost factors:
//!
//!   - `m` (memory): the number of 1 KB memory blocks to allocate. Higher
//!     values increase memory usage for the attacker. OWASP recommends a
//!     minimum of 19 MB (19456 blocks) for interactive logins.
//!   - `t` (time / iterations): the number of passes over the memory.
//!     Higher values increase time cost. OWASP recommends a minimum of 2.
//!   - `p` (parallelism): the number of independent lanes. This implementation
//!     is single-threaded; `p` is accepted as a parameter for PHC string
//!     compatibility, but the work is always performed sequentially across all
//!     lanes.
//!
//! Both `hash_password` and `hash_password_with_params` return a PHC string,
//! the standard serialization format used by password databases in format:
//!     $argon2id$v=19$m=19456,t=2,p=1$<salt_base64>$<hash_base64>
//! 
//! The PHC string is self-describing: it carries the algorithm, version, all
//! cost parameters, the salt, and the hash digest in a single string. It can
//! be stored directly in a user database and passed back to `verify_password`
//! for verification without any additional bookkeeping.
//!
//! # Security properties
//!
//!   - Memory hardness: filling a large memory array forces attackers to
//!     pay the full memory cost for each guess, making parallel GPU/ASIC
//!     attacks expensive.
//!   - Time hardness: multiple passes over the memory array increase the
//!     time cost per guess without additional memory.
//!   - Side-channel resistance: the Argon2id hybrid uses data-independent
//!     addressing for the first half-pass, preventing timing side channels
//!     during the most sensitive phase.

#![allow(dead_code)]

use crate::helper_functions::{base64_encode, base64_decode};
use super::blake2b::{blake2b_512, blake2b_512_slice, h_prime};

/// Argon2 version number, as encoded in the PHC string.
/// Version 19 (0x13) is the current and only widely-deployed version.
const ARGON2_VERSION: u32 = 19;

/// Argon2 type identifier for the Argon2id variant.
const ARGON2_TYPE_ID: u32 = 2;

/// Size of one Argon2 memory block in bytes.
const BLOCK_SIZE: usize = 1024;

/// Number of u64 words in one block (1024 / 8).
const BLOCK_WORDS: usize = BLOCK_SIZE / 8;

/// OWASP-recommended minimum memory cost: 19 MB = 19456 * 1 KB blocks.
pub const DEFAULT_M_COST: u32 = 19456;

/// OWASP-recommended minimum time cost (iteration count).
pub const DEFAULT_T_COST: u32 = 2;

/// Default parallelism. This implementation is single-threaded, so p=1
/// produces the same output as a multi-threaded implementation with p=1.
/// Callers that need p>1 output for cross-system compatibility can set this
/// via [`hash_password_with_params`].
pub const DEFAULT_P_COST: u32 = 1;

/// Output hash length in bytes. 32 bytes (256 bits) is the standard choice
/// and the OWASP recommendation for Argon2id.
pub const HASH_LEN: usize = 32;

/// Errors returned by Argon2id operations.
#[derive(Debug)]
pub enum Argon2Error {
    /// The memory cost `m` is too low. RFC 9106 requires m >= 8 * p.
    MemoryCostTooLow,

    /// The time cost `t` is zero. RFC 9106 requires t >= 1.
    TimeCostZero,

    /// The parallelism `p` is zero. RFC 9106 requires p >= 1.
    ParallelismZero,

    /// The salt is shorter than 8 bytes, which is the RFC 9106 minimum.
    SaltTooShort,

    /// The password is empty. While RFC 9106 technically allows empty
    /// passwords, we reject them as a defence-in-depth measure.
    PasswordEmpty,

    /// A PHC string passed to `verify_password` is malformed.
    InvalidPhcString,

    /// The password did not match the stored hash.
    PasswordMismatch,
}

/// Runs the full Argon2id algorithm and returns the raw [`HASH_LEN`]-byte
/// digest.
///
/// This is the internal entry point used by both [`hash_password_with_params`]
/// and [`verify_password`]. It implements RFC 9106 Section 3 in full:
///   1. Compute the initial hash H0 via BLAKE2b;
///   2. Derive the first two blocks of each lane from H0;
///   3. Fill the remaining blocks using Argon2id addressing (data-independent
///      for the first half of the first pass, data-dependent thereafter);
///   4. XOR all final-column blocks into a single block;
///   5. Extract the digest from the final block via BLAKE2b.
/// 
/// # Arguments
///
/// * `password` - The raw password bytes to hash. Must not be empty; enforced
///                by [`validate_params`] before this function is called.
/// * `salt`     - The salt bytes. Must be at least 8 bytes; enforced by
///                [`validate_params`] before this function is called.
/// * `m_cost`   - Memory cost in 1 KB blocks. Determines the total size of
///                the memory array: `m_cost` blocks of `BLOCK_SIZE` bytes each,
///                rounded down to the nearest multiple of `4 * p_cost`.
/// * `t_cost`   - Time cost (number of passes over the memory array). Each
///                additional pass increases the time cost without increasing
///                memory usage.
/// * `p_cost`   - Parallelism (number of independent lanes). Determines the
///                memory layout: the array is divided into `p_cost` lanes of
///                equal length, each processed independently. This
///                implementation is single-threaded; lanes are filled
///                sequentially rather than in parallel.
///
/// # Returns
///
/// Returns `[u8; HASH_LEN]` containing the raw 32-byte Argon2id digest.
fn argon2id_raw(
    password: &[u8],
    salt: &[u8],
    m_cost: u32,
    t_cost: u32,
    p_cost: u32,
) -> [u8; HASH_LEN] {
    // Step 1: Compute H0.
    // H0 is a 64-byte BLAKE2b digest of all Argon2 parameters concatenated in
    // a specific order. It seeds the entire memory-filling process.
    let h0 = compute_h0(password, salt, m_cost, t_cost, p_cost);

    // Compute memory layout.
    // The total number of blocks is rounded down to the nearest multiple of
    // (4 * p_cost), as required by the RFC. Each lane contains `lane_len`
    // blocks, divided into 4 equal segments.
    let p = p_cost as usize;
    let total_blocks = (m_cost as usize / (4 * p)) * (4 * p);
    let lane_len = total_blocks / p;
    let segment_len = lane_len / 4;

    // Allocate the memory array as a flat Vec of BLOCK_WORDS-word blocks.
    let mut memory: alloc::vec::Vec<[u64; BLOCK_WORDS]> =
        alloc::vec![[0u64; BLOCK_WORDS]; total_blocks];

    // Step 2: Initialise the first two blocks of each lane.
    // Blocks B[l][0] and B[l][1] are derived from H0 using the variable-length
    // hash function H'. The input is H0 concatenated with two 32-bit
    // little-endian integers: the constant 0 or 1 (for block index), and the
    // lane index.
    for l in 0..p {
        let block0 = h_prime_1024(&h0, 0, l as u32);
        let block1 = h_prime_1024(&h0, 1, l as u32);
        memory[l * lane_len] = block0;
        memory[l * lane_len + 1] = block1;
    }

    // Step 3: Fill all remaining blocks.
    // Argon2id uses Argon2i (data-independent) addressing for segments 0 and 1
    // (first half) of pass 0, and Argon2d (data-dependent) addressing for all
    // other segments and passes.
    for pass in 0..t_cost as usize {
        for slice in 0..4 {
            for lane in 0..p {
                fill_segment(
                    &mut memory,
                    pass,
                    lane,
                    slice,
                    lane_len,
                    segment_len,
                    p,
                    t_cost as usize,
                );
            }
        }
    }

    // Step 4: Finalise - XOR all last-column blocks.
    // XOR together the last block of every lane to produce the final block B_f.
    let mut final_block = [0u64; BLOCK_WORDS];
    for l in 0..p {
        let last = &memory[l * lane_len + (lane_len - 1)];
        for w in 0..BLOCK_WORDS {
            final_block[w] ^= last[w];
        }
    }

    // Step 5: Extract the digest via H'(HASH_LEN, B_f).
    let final_bytes = block_to_bytes(&final_block);
    let digest = h_prime(&final_bytes, HASH_LEN);

    let mut out = [0u8; HASH_LEN];
    out.copy_from_slice(&digest[..HASH_LEN]);
    out
}

/// Computes the 64-byte H0 seed block (RFC 9106, Section 3.3).
///
/// H0 = BLAKE2b-512(p || T || m || t || v || y || len(P) || P || len(S) || S
///                    || len(K) || K || len(X) || X),
///
/// where:
///   p = parallelism (u32 LE)     T = output length (u32 LE)
///   m = memory cost (u32 LE)     t = time cost (u32 LE)
///   v = version (u32 LE)         y = type (u32 LE, 2 for Argon2id)
///   K = secret key (empty here)  X = associated data (empty here)
/// 
/// # Arguments
///
/// * `password` - The raw password bytes. Included in H0 as a length-prefixed
///                field (`len(P) || P`). A single bit change produces a
///                completely different H0.
/// * `salt`     - The salt bytes. Included as a length-prefixed field
///                (`len(S) || S`). Ensures that two identical passwords
///                produce different H0 values and therefore different digests,
///                preventing precomputation attacks.
/// * `m_cost`   - Memory cost parameter, encoded as a u32 little-endian word.
/// * `t_cost`   - Time cost parameter, encoded as a u32 little-endian word.
/// * `p_cost`   - Parallelism parameter, encoded as a u32 little-endian word.
///                This is the first field in the H0 input, per RFC 9106.
///
/// # Returns
///
/// Returns the 64-byte BLAKE2b-512 digest of all concatenated input fields,
/// used as the seed for all subsequent memory block derivation.
fn compute_h0(
    password: &[u8],
    salt:     &[u8],
    m_cost:   u32,
    t_cost:   u32,
    p_cost:   u32,
) -> [u8; 64] {
    let mut input: alloc::vec::Vec<u8> = alloc::vec::Vec::with_capacity(
        8 * 4 + password.len() + salt.len()
    );

    // Fixed-width parameters in order, all little-endian u32
    input.extend_from_slice(&p_cost.to_le_bytes());
    input.extend_from_slice(&(HASH_LEN as u32).to_le_bytes());
    input.extend_from_slice(&m_cost.to_le_bytes());
    input.extend_from_slice(&t_cost.to_le_bytes());
    input.extend_from_slice(&ARGON2_VERSION.to_le_bytes());
    input.extend_from_slice(&ARGON2_TYPE_ID.to_le_bytes());

    // Variable-length fields: each prefixed with its length as a u32 LE
    input.extend_from_slice(&(password.len() as u32).to_le_bytes());
    input.extend_from_slice(password);
    input.extend_from_slice(&(salt.len() as u32).to_le_bytes());
    input.extend_from_slice(salt);

    // Secret key: empty (length 0, no bytes follow)
    input.extend_from_slice(&0u32.to_le_bytes());

    // Associated data: empty
    input.extend_from_slice(&0u32.to_le_bytes());

    blake2b_512(&input)
}

/// Produces a 1024-byte (128-word) memory block using H'.
///
/// Derives the initial blocks `B[lane][0]` and `B[lane][1]` for each lane
/// from the H0 seed. The input to H' is constructed as:
///
///   H'(1024, H0 || LE32(block_idx) || LE32(lane))
///
/// where `block_idx` is 0 for the first block of the lane and 1 for the
/// second. This ensures every initial block in every lane is derived from
/// the same H0 seed but is unique by construction, as no two (block_idx, lane)
/// pairs produce the same block.
///
/// # Arguments
///
/// * `h0`        - The 64-byte H0 seed block produced by `compute_h0`. All
///                 lanes share the same H0; the `block_idx` and `lane`
///                 arguments differentiate their initial blocks.
/// * `block_idx` - Index of the block within the lane being initialised.
///                 Must be 0 (first block) or 1 (second block). Values
///                 outside this range are not used by Argon2id but are not
///                 explicitly rejected; they would produce valid but unused
///                 blocks.
/// * `lane`      - Index of the lane being initialised, in `0..p_cost`.
///                 Combined with `block_idx`, uniquely identifies which of
///                 the `2 * p_cost` initial blocks is being derived.
///
/// # Returns
///
/// Returns a `[u64; BLOCK_WORDS]` array containing the 1024-byte derived block,
/// interpreted as 128 little-endian 64-bit words, ready for use as
/// `memory[lane * lane_len + block_idx]`.
fn h_prime_1024(
    h0: &[u8; 64],
    block_idx: u32,
    lane: u32,
) -> [u64; BLOCK_WORDS] {
    let mut input = [0u8; 64 + 4 + 4];
    input[..64].copy_from_slice(h0);
    input[64..68].copy_from_slice(&block_idx.to_le_bytes());
    input[68..72].copy_from_slice(&lane.to_le_bytes());

    let bytes = h_prime(&input, BLOCK_SIZE);
    bytes_to_block(&bytes)
}

/// Fills one segment of the memory array.
///
/// A segment is one quarter of a lane. The fill order is:
///   Pass 0, slices 0 and 1, all lanes: Argon2i (data-independent) addressing.
///   All other (pass, slice) combinations:  Argon2d (data-dependent)
///   addressing.
///
/// For each block position in the segment, a reference block index is
/// computed using the appropriate addressing mode, and the new block is
/// produced by applying the G compression function to the previous block
/// and the reference block.
/// 
/// # Arguments
///
/// * `memory`       - The flat memory array shared across all lanes. Indexed
///                    as `memory[lane * lane_len + block_offset]`. This
///                    function reads from arbitrary positions (the previous
///                    block and the reference block) and writes to the current
///                    position within the segment being filled.
/// * `pass`         - Current pass index, in `0..t_cost`. Pass 0 uses
///                    Argon2i addressing for the first two slices; all
///                    subsequent passes use Argon2d addressing throughout.
/// * `lane`         - Index of the lane being filled, in `0..p_cost`. Each
///                    lane occupies a contiguous region of `lane_len` blocks
///                    in `memory`, starting at `lane * lane_len`.
/// * `slice`        - Index of the segment within the lane, in `0..4`. Each
///                    lane is divided into exactly 4 equal segments; this
///                    argument identifies which quarter is being filled.
/// * `lane_len`     - Number of blocks per lane, equal to `total_blocks /
///                    p_cost`. Each lane's blocks occupy indices
///                    `lane * lane_len .. (lane + 1) * lane_len` in `memory`.
/// * `segment_len`  - Number of blocks per segment, equal to `lane_len / 4`.
///                    This is the number of block positions this call will
///                    fill, minus any skipped initial blocks on pass 0.
/// * `p`            - Total number of lanes (parallelism), equal to `p_cost`
///                    as `usize`. Used by `index_alpha` to compute the size
///                    of the reference set, which spans blocks across all
///                    lanes in passes after pass 0.
/// * `t_cost`       - Total number of passes. Passed through to
///                    `compute_pseudo_rands` for use in the data-independent
///                    pseudo-random keystream construction.
fn fill_segment(
    memory:      &mut alloc::vec::Vec<[u64; BLOCK_WORDS]>,
    pass:        usize,
    lane:        usize,
    slice:       usize,
    lane_len:    usize,
    segment_len: usize,
    p:           usize,
    t_cost:      usize,
) {
    // Argon2id: use data-independent addressing for pass 0, slices 0 and 1
    let data_independent = pass == 0 && slice < 2;

    // Precompute the pseudo-random block stream for data-independent mode.
    // In Argon2i mode, reference indices are derived from a keystream generated
    // by compressing a sequence of counter blocks, so no data from the memory
    // array is read while computing the index. This prevents timing side
    // channels in the first half of the first pass.
    let pseudo_rands: alloc::vec::Vec<u64> = if data_independent {
        compute_pseudo_rands(pass, lane, slice, lane_len, segment_len, p,t_cost)
    }
    else {
        alloc::vec::Vec::new()  // Unused in data-dependent mode
    };

    let seg_start = lane * lane_len + slice * segment_len;

    for s in 0..segment_len {
        let abs_idx = seg_start + s;

        // The first two blocks of each lane are already initialised from H0;
        // skip them on the first pass.
        if pass == 0 && abs_idx < lane * lane_len + 2 {
            continue;
        }

        // Previous block index: wraps within the lane
        let prev_idx = if abs_idx == lane * lane_len {
            lane * lane_len + lane_len - 1  // Wrap to last block in lane
        }
        else {
            abs_idx - 1
        };

        // Determine the pseudo-random value used to select the reference block
        let pseudo_rand = if data_independent {
            pseudo_rands[s]
        }
        else {
            // Argon2d: use the first u64 of the previous block as the index
            memory[prev_idx][0]
        };

        let ref_idx = index_alpha(
            pseudo_rand, pass, lane, slice, lane_len, segment_len, p, abs_idx,
        );

        // Produce the new block: G(prev_block XOR ref_block) XOR prev_block.
        // SAFETY: `prev_idx` and `ref_idx` are always distinct from `abs_idx`
        // by the construction in `index_alpha`. We need to read two blocks and
        // write a third; since all three indices are distinct, we use
        // split_at_mut to satisfy the borrow checker.
        let new_block = {
            let prev = memory[prev_idx];
            let refb = memory[ref_idx];
            compress(&prev, &refb)
        };
        memory[abs_idx] = new_block;
    }
}

/// Computes the pseudo-random keystream for one Argon2i segment.
///
/// In data-independent mode, reference indices are derived from a stream of
/// pseudo-random u64 values. Each value is the first word of a compressed
/// "input block" that encodes the current position (pass, lane, slice,
/// counter). This makes the reference pattern fully deterministic from the
/// parameters alone, with no dependence on memory contents.
/// 
/// # Arguments
///
/// * `pass`         - Current pass index, in `0..t_cost`. Encoded directly
///                    into the input block so that the same (lane, slice,
///                    counter) position produces a different pseudo-random
///                    value on each pass, preventing keystream reuse across
///                    passes.
/// * `lane`         - Index of the lane being filled, in `0..p_cost`. Encoded
///                    into the input block to ensure each lane has an
///                    independent pseudo-random keystream even within the
///                    same (pass, slice).
/// * `slice`        - Index of the segment within the lane, in `0..4`.
///                    Encoded into the input block alongside `pass` and `lane`
///                    to fully qualify the position of this keystream within
///                    the overall filling order.
/// * `lane_len`     - Number of blocks per lane. Currently unused directly in
///                    this function but retained for symmetry with
///                    `fill_segment`'s signature and potential future use in
///                    extended addressing modes.
/// * `segment_len`  - Number of pseudo-random values to produce, equal to the
///                    number of block positions in this segment. One value is
///                    generated per position; the caller indexes into the
///                    returned `Vec` by segment position `s`.
/// * `p`            - Total number of lanes (parallelism). Encoded into the
///                    input block as part of the position description, ensuring
///                    the keystream differs across configurations with
///                    different parallelism values even if all other parameters
///                    match.
/// * `t_cost`       - Total number of passes. Encoded into the input block
///                    alongside the other position fields to fully bind the
///                    keystream to the complete set of Argon2id cost
///                    parameters.
///
/// # Returns
///
/// Returns a `Vec<u64>` of exactly `segment_len` pseudo-random values. Each
/// value is the first `u64` word of a compressed input block and is passed to
/// `index_alpha` to select the reference block for the corresponding position
/// in the segment.
fn compute_pseudo_rands(
    pass:        usize,
    lane:        usize,
    slice:       usize,
    lane_len:    usize,
    segment_len: usize,
    p:           usize,
    t_cost:      usize,
) -> alloc::vec::Vec<u64> {
    let _ = (lane_len, p, t_cost);  // Used indirectly via segment_len
    let mut result = alloc::vec::Vec::with_capacity(segment_len);

    for s in 0..segment_len {
        // Build the input block for this counter position.
        // Layout: (pass, lane, slice, s+1, t_cost, p, ARGON2_TYPE_ID, zeros...)
        let mut input_block = [0u64; BLOCK_WORDS];
        input_block[0] = pass as u64;
        input_block[1] = lane as u64;
        input_block[2] = slice as u64;
        input_block[3] = s as u64 + 1;
        input_block[4] = t_cost as u64;
        input_block[5] = p as u64;
        input_block[6] = ARGON2_TYPE_ID as u64;

        // G(zero_block, input_block) gives the pseudo-random block; its first
        // word seeds the reference index selection.
        let zero_block = [0u64; BLOCK_WORDS];
        let pr_block   = compress(&zero_block, &input_block);
        result.push(pr_block[0]);
    }

    result
}

/// Computes the reference block index for one block fill step.
///
/// Implements the index mapping from RFC 9106, Section 3.4. The reference
/// block is drawn from the "reference set" (the set of already-filled blocks
/// visible to the current position) using `pseudo_rand` as a non-uniform
/// index into that set. The mapping applies a quadratic distribution that
/// favours recently-written blocks, increasing the effective memory hardness
/// by making older blocks less likely to be evicted from cache before they
/// are referenced.
///
/// The reference set size varies by pass and slice:
///   - Pass 0, slice 0: only blocks before the current position in this lane.
///   - Pass 0, slice > 0: all filled blocks in previous slices of this lane,
///     plus blocks before the current position in the current slice.
///   - Pass > 0: all blocks in all lanes except the segment currently being
///     filled, giving a reference set that spans the entire memory array
///     minus one segment.
///
/// # Arguments
///
/// * `pseudo_rand`  - A 64-bit pseudo-random value used to select the
///                    reference block from the reference set. In
///                    data-independent mode this comes from
///                    `compute_pseudo_rands`; in data-dependent mode it is
///                    the first `u64` word of the previous block.
/// * `pass`         - Current pass index, in `0..t_cost`. Determines which
///                    formula is used to compute the reference set size -
///                    pass 0 has a smaller reference set that grows as blocks
///                    are filled; later passes have access to the full array.
/// * `lane`         - Index of the lane being filled, in `0..p_cost`. Used
///                    to compute the absolute block index of the start of
///                    this lane and to identify the current lane's blocks
///                    within the global reference set on passes after pass 0.
/// * `slice`        - Index of the current segment within the lane, in `0..4`.
///                    Used on pass 0 to determine how many previously filled
///                    segments are available as reference candidates.
/// * `lane_len`     - Number of blocks per lane. Used to compute absolute
///                    block indices from lane-relative offsets, and to
///                    determine the reference set size on passes after pass 0.
/// * `segment_len`  - Number of blocks per segment, equal to `lane_len / 4`.
///                    Used to compute the number of already-filled blocks
///                    within the current slice up to the current position.
/// * `p`            - Total number of lanes (parallelism). Used on passes
///                    after pass 0 to include blocks from all other lanes in
///                    the reference set, giving a reference set size of
///                    approximately `p * lane_len - segment_len`.
/// * `abs_idx`      - Absolute index of the block currently being filled in
///                    the flat `memory` array, equal to
///                    `lane * lane_len + slice * segment_len + s`. Used to
///                    determine how many blocks have been filled before this
///                    position within the current slice.
///
/// # Returns
///
/// Returns the absolute index into the flat `memory` array of the selected
/// reference block. Guaranteed to be distinct from `abs_idx` (a block is never
/// its own reference) and to point to an already-filled block, by the
/// construction of the reference set and the quadratic mapping.
fn index_alpha(
    pseudo_rand:  u64,
    pass:         usize,
    lane:         usize,
    slice:        usize,
    lane_len:     usize,
    segment_len:  usize,
    p:            usize,
    abs_idx:      usize,
) -> usize {
    // Compute the size of the reference set
    let reference_area = if pass == 0 {
        if slice == 0 {
            abs_idx - lane * lane_len - 1
        }
        else {
            slice * segment_len + (abs_idx - lane*lane_len-slice*segment_len)-1
        }
    }
    else {
        lane_len - segment_len + (abs_idx - lane *lane_len-slice*segment_len)-1
            + (p - 1) * lane_len
    };

    if reference_area == 0 {
        return lane * lane_len;  // Fallback: first block of this lane
    }

    // Quadratic distribution mapping:
    //     phi = reference_area * (x^2 / 2^64 + x / 2^32),
    // where x is the low 32 bits of pseudo_rand. This favours recently-written
    // blocks.
    let x  = pseudo_rand & 0xFFFF_FFFF;
    let y  = (x * x) >> 32;
    let z  = (x + y) >> 1;
    let phi = reference_area - 1 - (z as usize % reference_area);

    // Map phi into an absolute block index. The reference set starts at the
    // beginning of the previous slice (or wraps for pass 0 slice 0).
    let start = if pass == 0 {
        lane * lane_len
    }
    else {
        lane * lane_len + ((slice + 1) % 4) * segment_len
    };

    (start + phi) % (p * lane_len)
}

/// The Argon2 block compression function G:
///
///     G(X, Y) = P(P(X XOR Y, row-wise), column-wise) XOR (X XOR Y),
///
/// where P is the Argon2 permutation - 8 applications of the modified
/// Blake2b G mixing function applied to the 16 words of each row or column
/// of the 8*16 matrix view of the 1024-byte block.
///
/// The computation proceeds in three steps:
///   1. Compute R = X XOR Y, the bitwise XOR of the two input blocks.
///   2. Compute Z by applying P to each of the 8 rows of R, then applying
///      P to each of the 8 columns of the result.
///   3. Return Z XOR R, mixing the permuted result back with the pre-image
///      to prevent the compression from being trivially invertible.
///
/// The modified G mixing function used inside P differs from the standard
/// Blake2b G function in that two of the four addition steps are augmented
/// with a 64*64 -> 128-bit multiplication term (via `trunc32`), providing
/// additional non-linearity and resistance to GPU-based attacks.
///
/// # Arguments
///
/// * `x` - The first input block, typically the previous block in the lane
///         (`memory[prev_idx]`). Treated as a 128-word (1024-byte) array of
///         little-endian `u64` values. Not modified by this function.
/// * `y` - The second input block, the reference block selected by
///         `index_alpha` (`memory[ref_idx]`). Treated identically to `x`.
///         Not modified by this function.
///
/// # Returns
///
/// Returns a new `[u64; BLOCK_WORDS]` block containing the result of G(X, Y),
/// written into `memory[abs_idx]` by the caller. The output is fully
/// determined by `x` and `y`; this function has no side effects.
fn compress(
    x: &[u64; BLOCK_WORDS],
    y: &[u64; BLOCK_WORDS],
) -> [u64; BLOCK_WORDS] {
    // R = X XOR Y
    let mut r = [0u64; BLOCK_WORDS];
    for i in 0..BLOCK_WORDS {
        r[i] = x[i] ^ y[i];
    }

    let mut z = r;

    // Apply P to each of the 8 rows (each row is 16 u64 words = 128 bytes)
    for row in 0..8 {
        let base = row * 16;
        p_col(&mut z, base, base+1, base+2, base+3, base+4, base+5, base+6,
            base+7, base+8, base+9, base+10, base+11, base+12, base+13, base+14,
            base+15);
    }

    // Apply P to each of the 8 columns
    for col in 0..8 {
        p_col(&mut z,
            col,      col+8,  col+16, col+24,
            col+32,   col+40, col+48, col+56,
            col+64,   col+72, col+80, col+88,
            col+96,   col+104,col+112,col+120);
    }

    // Output = Z XOR R
    for i in 0..BLOCK_WORDS {
        z[i] ^= r[i];
    }

    z
}

/// Applies the Argon2 permutation P to 16 u64 words addressed by index.
///
/// P is defined as 8 applications of the modified Blake2b G mixing function,
/// organized in a specific "diagonal" pattern within the 4*4 sub-block. The
/// 16 word indices address the words in the full 128-word block.
#[allow(clippy::too_many_arguments)]
#[inline]
fn p_col(
    v: &mut [u64; BLOCK_WORDS],
    v0: usize,  v1: usize,  v2: usize,  v3: usize,
    v4: usize,  v5: usize,  v6: usize,  v7: usize,
    v8: usize,  v9: usize, v10: usize, v11: usize,
   v12: usize, v13: usize, v14: usize, v15: usize,
) {
    // Four column G-mix calls
    g_mix(v, v0, v4, v8,  v12);
    g_mix(v, v1, v5, v9,  v13);
    g_mix(v, v2, v6, v10, v14);
    g_mix(v, v3, v7, v11, v15);

    // Four diagonal G-mix calls
    g_mix(v, v0, v5, v10, v15);
    g_mix(v, v1, v6, v11, v12);
    g_mix(v, v2, v7, v8,  v13);
    g_mix(v, v3, v4, v9,  v14);
}

/// The modified Blake2b G mixing function used in the Argon2 permutation.
///
/// Differs from standard Blake2b G in that it uses 64-bit multiplication
/// instead of XOR for two of the mixing steps, providing non-linearity. All
/// rotations are the same as Blake2b.
#[inline]
fn g_mix(
    v: &mut [u64; BLOCK_WORDS],
    a: usize,
    b: usize,
    c: usize,
    d: usize,
) {
    v[a] = v[a].wrapping_add(v[b])
               .wrapping_add(2u64.wrapping_mul(trunc32(v[a]).wrapping_mul(trunc32(v[b]))));
    v[d] = (v[d] ^ v[a]).rotate_right(32);
    v[c] = v[c].wrapping_add(v[d])
               .wrapping_add(2u64.wrapping_mul(trunc32(v[c]).wrapping_mul(trunc32(v[d]))));
    v[b] = (v[b] ^ v[c]).rotate_right(24);
    v[a] = v[a].wrapping_add(v[b])
               .wrapping_add(2u64.wrapping_mul(trunc32(v[a]).wrapping_mul(trunc32(v[b]))));
    v[d] = (v[d] ^ v[a]).rotate_right(16);
    v[c] = v[c].wrapping_add(v[d])
               .wrapping_add(2u64.wrapping_mul(trunc32(v[c]).wrapping_mul(trunc32(v[d]))));
    v[b] = (v[b] ^ v[c]).rotate_right(63);
}

/// Extracts the low 32 bits of a u64 as a u64, used in the modified G mixing.
#[inline(always)]
fn trunc32(x: u64) -> u64 { x & 0xFFFF_FFFF }

/// Interprets a 1024-byte slice as a `[u64; BLOCK_WORDS]` block.
///
/// Each consecutive 8-byte group in `bytes` is decoded as one little-endian
/// `u64` word, producing a 128-word array. This is the canonical byte-to-block
/// conversion used throughout Argon2id whenever a byte sequence produced by
/// H' needs to be stored in the memory array.
///
/// # Arguments
///
/// * `bytes` - The byte slice to interpret. Must be exactly [`BLOCK_SIZE`]
///             bytes long; passing a shorter slice will panic on the
///             `try_into` call inside the conversion loop.
///
/// # Returns
///
/// Returns a `[u64; BLOCK_WORDS]` array of 128 little-endian `u64` words
/// representing the same data as `bytes`. The inverse of `block_to_bytes`.
fn bytes_to_block(bytes: &[u8]) -> [u64; BLOCK_WORDS] {
    let mut block = [0u64; BLOCK_WORDS];
    for (i, word) in block.iter_mut().enumerate() {
        *word = u64::from_le_bytes(bytes[i*8..i*8+8].try_into().unwrap());
    }
    block
}

/// Serialises a `[u64; BLOCK_WORDS]` block to a 1024-byte `Vec<u8>`.
///
/// Each `u64` word in `block` is encoded as 8 consecutive little-endian bytes,
/// producing a flat byte representation of the block. This is the canonical
/// block-to-byte conversion used when a memory block needs to be passed to a
/// byte-oriented function such as `blake2b::h_prime` during final digest
/// extraction.
///
/// # Arguments
///
/// * `block` - The 128-word block to serialise. Each word is written in
///             little-endian byte order. The inverse of [`bytes_to_block`]:
///             `bytes_to_block(&block_to_bytes(b))` always returns `*b`.
///
/// # Returns
///
/// Returns a `Vec<u8>` of exactly [`BLOCK_SIZE`] bytes containing the
/// little-endian byte representation of `block`.
fn block_to_bytes(block: &[u64; BLOCK_WORDS]) -> alloc::vec::Vec<u8> {
    let mut bytes = alloc::vec![0u8; BLOCK_SIZE];
    for (i, &word) in block.iter().enumerate() {
        bytes[i*8..i*8+8].copy_from_slice(&word.to_le_bytes());
    }
    bytes
}

/// Encodes the salt, hash digest, and cost parameters into a PHC string.
///
/// Produces the standard PHC (Password Hashing Competition) serialization
/// format used by password databases to store a self-describing hash record.
/// The format carries everything needed to verify a password or reproduce the
/// hash without any external bookkeeping.
///
/// Output format:
/// `$argon2id$v=19$m=<m_cost>,t=<t_cost>,p=<p_cost>$<salt_b64>$<hash_b64>`,
///
/// where `<salt_b64>` and `<hash_b64>` are unpadded Base64 encodings of the
/// raw salt and digest bytes respectively.
///
/// # Arguments
///
/// * `salt`     - The raw salt bytes that were used during hashing. Encoded
///                as unpadded Base64 in the fourth field of the PHC string.
///                Must be the same salt that was passed to [`argon2id_raw`],
///                as it is required for verification.
/// * `hash`     - The raw 32-byte digest produced by [`argon2id_raw`]. Encoded
///                as unpadded Base64 in the fifth and final field of the PHC
///                string.
/// * `m_cost`   - Memory cost parameter encoded in the third field as `m=N`.
///                Stored so that [`verify_password`] can reproduce the hash
///                with identical parameters without caller intervention.
/// * `t_cost`   - Time cost parameter encoded in the third field as `t=N`.
///                Stored for the same reason as `m_cost`.
/// * `p_cost`   - Parallelism parameter encoded in the third field as `p=N`.
///                Stored for the same reason as `m_cost`.
///
/// # Returns
///
/// Returns an owned `String` containing the complete PHC-format hash record,
/// suitable for direct storage in a user database. The string is fully
/// self-describing and can be passed to [`verify_password`] without any
/// additional context.
fn encode_phc(
    salt: &[u8],
    hash: &[u8; HASH_LEN],
    m_cost: u32,
    t_cost: u32,
    p_cost: u32,
) -> alloc::string::String {
    let mut s = alloc::string::String::new();
    s.push_str("$argon2id$v=19$m=");
    s.push_str(&u32_to_alloc_string(m_cost));
    s.push_str(",t=");
    s.push_str(&u32_to_alloc_string(t_cost));
    s.push_str(",p=");
    s.push_str(&u32_to_alloc_string(p_cost));
    s.push('$');
    s.push_str(&base64_encode(salt));
    s.push('$');
    s.push_str(&base64_encode(hash));
    s
}

/// Parses a PHC string and extracts the salt, hash digest, and cost parameters.
///
/// The inverse of [`encode_phc`]. Splits the PHC string on `$` delimiters,
/// validates the algorithm identifier and version field, parses the three
/// cost parameters from the parameter field, and Base64-decodes the salt
/// and hash digest fields. Used by [`verify_password`] to recover all
/// information needed to re-hash a candidate password with identical
/// parameters.
///
/// Expected format:
/// `$argon2id$v=19$m=<m_cost>,t=<t_cost>,p=<p_cost>$<salt_b64>$<hash_b64>`
///
/// # Arguments
///
/// * `phc` - The PHC string to parse, as produced by [`encode_phc`]. Must
///           begin with `$argon2id$v=19$`, contain exactly three
///           comma-separated cost parameters in the third field, and have
///           valid unpadded Base64 in the salt and hash fields. Any
///           deviation from this structure returns
///           `Err(Argon2Error::InvalidPhcString)`.
///
/// # Returns
///
/// `Ok((salt, hash, m_cost, t_cost, p_cost))` on success, where:
///   - `salt`   is the decoded raw salt bytes as a `Vec<u8>`;
///   - `hash`   is the decoded raw digest bytes as a `Vec<u8>`;
///   - `m_cost` is the memory cost as a `u32`;
///   - `t_cost` is the time cost as a `u32`;
///   - `p_cost` is the parallelism as a `u32`.
///
/// `Err(Argon2Error::InvalidPhcString)` if any of the following are true:
///   - The string does not split into exactly 6 `$`-delimited fields;
///   - The algorithm identifier field is not `"argon2id"`;
///   - The version field is not `"v=19"`;
///   - The parameter field does not contain exactly three comma-separated
///     `key=value` pairs with the keys `m`, `t`, and `p` in that order;
///   - Any parameter value is not a valid decimal `u32`;
///   - The salt or hash field contains characters outside the Base64 alphabet.
fn decode_phc(
    phc: &str,
) -> Result<(alloc::vec::Vec<u8>,alloc::vec::Vec<u8>,u32,u32,u32),Argon2Error> {
    // Expected format: $argon2id$v=19$m=M,t=T,p=P$SALT$HASH
    let parts: alloc::vec::Vec<&str> = phc.split('$').collect();
    if parts.len() != 6 || parts[0] != "" || parts[1] != "argon2id" {
        return Err(Argon2Error::InvalidPhcString);
    }

    // parts[2] = "v=19" - version check
    if parts[2] != "v=19" {
        return Err(Argon2Error::InvalidPhcString);
    }

    // parts[3] = "m=M,t=T,p=P"
    let params: alloc::vec::Vec<&str> = parts[3].split(',').collect();
    if params.len() != 3 {
        return Err(Argon2Error::InvalidPhcString);
    }
    let m_cost = parse_param(params[0], "m=")
        .ok_or(Argon2Error::InvalidPhcString)?;
    let t_cost = parse_param(params[1], "t=")
        .ok_or(Argon2Error::InvalidPhcString)?;
    let p_cost = parse_param(params[2], "p=")
        .ok_or(Argon2Error::InvalidPhcString)?;

    let salt = base64_decode(parts[4])
        .ok_or(Argon2Error::InvalidPhcString)?;
    let hash = base64_decode(parts[5])
        .ok_or(Argon2Error::InvalidPhcString)?;

    Ok((salt, hash, m_cost, t_cost, p_cost))
}

/// Parses a `"key=VALUE"` parameter string and returns the `u32` value.
///
/// Strips the expected `prefix` from the start of `s` and delegates to
/// [`parse_u32`] to decode the remaining decimal string. Used by [`decode_phc`]
/// to extract the individual cost parameter values from the comma-separated
/// parameter field of a PHC string.
///
/// # Arguments
///
/// * `s`      - The parameter string to parse, e.g. `"m=19456"`. Must begin
///              with `prefix` followed immediately by a decimal integer with
///              no surrounding whitespace. Returns `None` if `s` does not
///              start with `prefix` or if the value portion is not a valid
///              decimal `u32`.
/// * `prefix` - The expected key prefix including the `=` separator, e.g.
///              `"m="`, `"t="`, or `"p="`. Must match the start of `s`
///              exactly; the comparison is case-sensitive.
///
/// # Returns
///
/// Returns `Some(value)` if `s` begins with `prefix` and the remainder is a
/// valid decimal `u32`, or `None` if the prefix does not match or the value
/// cannot be parsed. `None` is propagated by [`decode_phc`] as
/// `Err(Argon2Error::InvalidPhcString)`.
fn parse_param(s: &str, prefix: &str) -> Option<u32> {
    let val = s.strip_prefix(prefix)?;

    parse_u32(val)
}

/// Parses a decimal `u32` from a string without using `std` or `core::fmt`.
///
/// Iterates over the bytes of `s` one character at a time, rejecting any
/// byte outside the ASCII decimal digit range (`'0'`–`'9'`). Accumulates the
/// result using checked arithmetic to detect overflow without panicking.
/// Used by [`parse_param`] to decode cost parameter values from a PHC string
/// in a `no_std` context where `str::parse::<u32>()` is unavailable.
///
/// # Arguments
///
/// * `s` - The string to parse. Must contain only ASCII decimal digits (`0`–
///         `9`) with no leading or trailing whitespace, no sign character,
///         and no underscores or other separators. An empty string returns
///         `None`. A value that would overflow `u32` (i.e., greater than
///         `4_294_967_295`) returns `None` via `checked_mul` / `checked_add`
///         rather than wrapping or panicking.
///
/// # Returns
///
/// Returns `Some(value)` if `s` is a non-empty string of ASCII decimal digits
/// representing a value in `0..=u32::MAX`, or `None` if `s` is empty,
/// contains any non-digit character, or the decoded value overflows `u32`.
fn parse_u32(s: &str) -> Option<u32> {
    if s.is_empty() {
        return None;
    }

    let mut result: u32 = 0;

    for b in s.bytes() {
        if b < b'0' || b > b'9' { return None; }
        result = result.checked_mul(10)?.checked_add((b - b'0') as u32)?;
    }

    Some(result)
}

/// Converts a `u32` to an owned decimal `String` without using `format!` or
/// any other `std` formatting machinery.
///
/// Builds the decimal representation manually by repeatedly extracting the
/// least significant digit via remainder and division, writing digits into a
/// fixed-size stack buffer in reverse order, then constructing a `String`
/// from the filled portion of the buffer. Used by `encode_phc` to serialize
/// the `m_cost`, `t_cost`, and `p_cost` cost parameters into the PHC string
/// without depending on `std::fmt`.
///
/// # Arguments
///
/// * `v` - The `u32` value to convert. All values in `0..=u32::MAX` are
///         handled correctly, including `0` which returns `"0"` rather than
///         an empty string. The maximum value `4_294_967_295` produces a
///         10-character string, which fits exactly in the 10-byte stack
///         buffer used internally.
///
/// # Returns
///
/// Returns an owned `String` containing the decimal representation of `v`, with
/// no leading zeroes (except for the value `0` itself), no sign character, and
/// no surrounding whitespace.
fn u32_to_alloc_string(mut v: u32) -> alloc::string::String {
    if v == 0 {
        return alloc::string::String::from("0");
    }

    let mut buf = [0u8; 10];
    let mut i = 10usize;

    while v > 0 {
        i -= 1;
        buf[i] = b'0' + (v % 10) as u8;
        v /= 10;
    }

    alloc::string::String::from(
        core::str::from_utf8(&buf[i..]).unwrap()
    )
}

/// Validates all Argon2id input parameters against the constraints defined in
/// the RFC before any memory allocation or hashing work is performed.
///
/// Centralises all parameter checking so that [`argon2id_raw`] can assume its
/// inputs are well-formed and focus purely on the hashing algorithm. Every
/// public entry point ([`hash_password`] and [`hash_password_with_params`])
/// calls this function before delegating to [`argon2id_raw`].
///
/// # Arguments
///
/// * `password` - The raw password bytes to validate. Rejected if empty as a
///                defence-in-depth measure, even though the RFC technically
///                permits empty passwords. An empty password is almost always
///                a caller bug.
/// * `salt`     - The salt bytes to validate. Must be at least 8 bytes per
///                the RFC. Shorter salts provide insufficient uniqueness and
///                are rejected unconditionally.
/// * `m_cost`   - Memory cost in 1 KB blocks. Must satisfy `m_cost >= 8 *
///                p_cost` per the RFC, which ensures each lane has at least
///                8 blocks - the minimum needed for the segment and slice
///                structure to be well-defined.
/// * `t_cost`   - Time cost (number of passes). Must be >= 1 per the RFC.
///                A value of 0 would mean no memory filling occurs and the
///                output would be derived solely from H0, providing no memory
///                hardness.
/// * `p_cost`   - Parallelism (number of lanes). Must be >= 1 per the RFC.
///                A value of 0 would produce a degenerate memory layout with
///                no lanes and is rejected unconditionally.
///
/// # Returns
///
/// Returns `Ok(())` if all parameters satisfy their constraints, or the first
/// applicable error from `Argon2Error` if any constraint is violated:
///   - `Argon2Error::PasswordEmpty`     if `password` is empty;
///   - `Argon2Error::SaltTooShort`      if `salt.len() < 8`;
///   - `Argon2Error::TimeCostZero`      if `t_cost == 0`;
///   - `Argon2Error::ParallelismZero`   if `p_cost == 0`;
///   - `Argon2Error::MemoryCostTooLow`  if `m_cost < 8 * p_cost`.
fn validate_params(
    password: &[u8],
    salt:     &[u8],
    m_cost:   u32,
    t_cost:   u32,
    p_cost:   u32,
) -> Result<(), Argon2Error> {
    if password.is_empty()  { return Err(Argon2Error::PasswordEmpty);    }
    if salt.len() < 8       { return Err(Argon2Error::SaltTooShort);     }
    if t_cost == 0          { return Err(Argon2Error::TimeCostZero);     }
    if p_cost == 0          { return Err(Argon2Error::ParallelismZero);  }
    if m_cost < 8 * p_cost  { return Err(Argon2Error::MemoryCostTooLow); }

    Ok(())
}

/// Compares two byte slices for equality in constant time.
///
/// Iterates over all bytes of both slices regardless of where the first
/// mismatch occurs, ensuring the time taken is determined solely by the
/// length of the inputs and not by their contents. This prevents timing
/// side channels that would otherwise allow an attacker to learn how many
/// bytes of a computed hash matched a stored hash by measuring response
/// latency - a realistic attack against naive equality checks in password
/// verification loops.
///
/// # Arguments
///
/// * `a` - The first byte slice to compare. In [`verify_password`], this is
///         the freshly computed Argon2id digest for the candidate password.
/// * `b` - The second byte slice to compare. In [`verify_password`], this is
///         the stored digest decoded from the PHC string. Must be the same
///         length as `a`; if the lengths differ, the function returns `false`
///         immediately without inspecting any bytes. This length check is
///         not a timing leak because digest lengths are public information
///         fixed by the algorithm parameters.
///
/// # Returns
///
/// Returns `true` if `a` and `b` are identical in both length and content,
/// `false` otherwise. The return value is derived by accumulating the bitwise
/// OR of all per-byte XOR differences into a single byte - a non-zero
/// accumulator indicates at least one differing byte, and a zero accumulator
/// confirms all bytes matched.
#[inline]
fn constant_time_eq(a: &[u8], b: &[u8]) -> bool {
    if a.len() != b.len() {
        return false;
    }

    let mut diff: u8 = 0;

    for (x, y) in a.iter().zip(b.iter()) {
        diff |= x ^ y;
    }

    diff == 0
}

// =============================================================================
// Public API
// =============================================================================

/// Hashes a password using Argon2id with OWASP-recommended default parameters.
///
/// Uses `m = 19456` (19 MB), `t = 2` iterations, `p = 1` lane. These are
/// the 2023 OWASP minimum recommendations for interactive authentication. For
/// higher-security contexts (e.g., encrypting long-term key material), use
/// [`hash_password_with_params`] with larger `m` and `t` values.
///
/// # Arguments
///
/// * `password` - The raw password bytes to hash. Must not be empty.
/// * `salt`     - A unique, randomly generated salt. Must be at least 8 bytes;
///                16 bytes (128 bits) is recommended. Generate with
///                [`crate::crypto::rng::get_random_bytes`].
///
/// # Returns
///
/// Returns `Ok(String)` containing the PHC-format hash string on success, or
/// `Err(Argon2Error)` if any input constraint is violated. Example PHC output:
///     `$argon2id$v=19$m=19456,t=2,p=1$<salt_base64>$<hash_base64>`
pub fn hash_password(
    password: &[u8],
    salt: &[u8],
) -> Result<alloc::string::String, Argon2Error> {
    hash_password_with_params(
        password,
        salt,
        DEFAULT_M_COST,
        DEFAULT_T_COST,
        DEFAULT_P_COST,
    )
}

/// Hashes a password using Argon2id with caller-specified cost parameters.
///
/// # Arguments
///
/// * `password` - The raw password bytes to hash. Must not be empty.
/// * `salt`     - A unique, randomly generated salt. Must be at least 8 bytes.
/// * `m_cost`   - Memory cost in KB blocks. Must be >= 8 * p_cost.
///                Each block is 1024 bytes, so `m_cost = 19456` uses ~19 MB.
/// * `t_cost`   - Time cost (number of passes). Must be >= 1.
/// * `p_cost`   - Parallelism (number of lanes). Must be >= 1. This
///                implementation processes lanes sequentially; p affects the
///                memory layout and output but not the thread count.
///
/// # Returns
///
/// Returns `Ok(String)` with the PHC-format hash, or `Err(Argon2Error)` if any
/// parameter violates the RFC 9106 constraints.
pub fn hash_password_with_params(
    password: &[u8],
    salt: &[u8],
    m_cost: u32,
    t_cost: u32,
    p_cost: u32,
) -> Result<alloc::string::String, Argon2Error> {
    validate_params(password, salt, m_cost, t_cost, p_cost)?;

    let hash = argon2id_raw(password, salt, m_cost, t_cost, p_cost);

    Ok(encode_phc(salt, &hash, m_cost, t_cost, p_cost))
}

/// Verifies a password against a stored PHC hash string.
///
/// Parses the PHC string to recover the salt and cost parameters, re-hashes
/// the candidate password with those parameters, and compares the result
/// against the stored digest using a constant-time comparison to prevent
/// timing side channels.
///
/// # Arguments
///
/// * `password` - The candidate password bytes to verify.
/// * `phc`      - The stored PHC hash string produced by [`hash_password`] or
///                [`hash_password_with_params`].
///
/// # Returns
///
/// `Ok(())` if the password matches, or `Err(Argon2Error)` if the PHC string
/// is malformed or the password does not match.
pub fn verify_password(
    password: &[u8],
    phc: &str,
) -> Result<(), Argon2Error> {
    let (salt, stored_hash, m_cost, t_cost, p_cost) = decode_phc(phc)?;

    let computed = argon2id_raw(password, &salt, m_cost, t_cost, p_cost);

    // Constant-time comparison: iterate over all bytes regardless of the
    // first mismatch, so the time taken does not reveal how many bytes matched.
    if !constant_time_eq(&computed, &stored_hash) {
        return Err(Argon2Error::PasswordMismatch);
    }

    Ok(())
}
