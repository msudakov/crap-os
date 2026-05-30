//! Cryptographically Secure Random Number Generator
//!
//! This module provides `get_random_bytes`, a CSPRNG backed by x86-64 hardware
//! entropy sources. It is `no_std`-compatible, does not allocate, and scales
//! to SMP by giving each logical CPU its own independent entropy pool via
//! [`PerCpu<CpuRngState>`]. There is no shared mutable state and no inter-CPU
//! contention on the hot path.
//!
//! Three entropy sources are used, in priority order:
//!
//!   1. RDSEED (preferred seed source): reads directly from the CPU's
//!      on-die hardware entropy conditioner (thermal noise, ring oscillator
//!      jitter, etc.) before any DRBG stage. Carries the highest entropy
//!      quality, but may stall under load and requires retries. Used only
//!      during pool (re)seeding.
//!
//!   2. RDRAND (bulk generation): reads from a hardware DRBG
//!      (SP 800-90A CTR_DRBG) that is itself continuously reseeded from the
//!      same entropy source as RDSEED. Guaranteed to succeed within a bounded
//!      number of retries under normal operation (at most 10, per Intel spec;
//!      we use 16 to be safe). Used for every call to `get_random_bytes` once
//!      the pool is seeded.
//!
//!   3. TSC + HPET jitter fallback: used on hardware that lacks RDRAND /
//!      RDSEED (pre-Ivy Bridge CPUs, some hypervisors that hide the feature
//!      bits). Mixes `RDTSC` samples with HPET counter reads through the
//!      wyHash finalizer. Not cryptographically strong in isolation, but
//!      meaningfully better than a fixed seed and the best available option
//!      without hardware support.
//!
//! Each logical CPU has its own [`CpuRngState`] slot in a
//! `static PerCpu<CpuRngState>`. A slot holds a 256-bit (32-byte) entropy pool
//! and a plain `u32` reseed counter. Because only the owning CPU ever touches
//! its slot, no atomics or locks are needed for the pool state itself. The only
//! synchronization concern is preventing an interrupt handler on the same core
//! from re-entering [`get_random_bytes`] mid-draw; we address this by saving
//! and restoring the interrupt-enable flag around the draw (the same pattern
//! used by `IrqSpinLock`, but without the lock).
//!
//! Every [`RESEED_INTERVAL`] calls, the pool is refreshed by XOR-folding in 32
//! fresh bytes from the hardware source. Between reseeds, each 8-byte output
//! chunk is bound to the pool state (output = hw_word XOR pool_word), so
//! learning the RDRAND output stream does not reveal the pool, and vice versa.

use core::arch::asm;
use crate::processor_control::{CpuId, PerCpu};
use crate::helper_functions::{disable_interrupts_save, restore_interrupts};

/// Number of [`get_random_bytes`] calls between automatic pool reseeds.
///
/// Each reseed draws 32 fresh bytes from the hardware source and XOR-folds
/// them into the pool. Lower values increase forward secrecy at the cost of
/// more RDSEED/RDRAND calls per unit of output.
const RESEED_INTERVAL: u32 = 64;

/// Maximum retry count for RDRAND and RDSEED before falling back to jitter.
///
/// Intel's specification guarantees RDRAND succeeds within 10 retries under
/// normal conditions. We use 16 as a conservative margin.
const HW_RETRY_LIMIT: u32 = 16;

/// Per-CPU entropy pool state.
///
/// One instance lives in each CPU's slot in [`RNG_STATE`]. All fields are
/// accessed only by the owning CPU, so no atomics or locks are needed here.
pub struct CpuRngState {
    /// 256-bit entropy pool. Mixed with hardware output on every draw and
    /// reseeded from hardware every [`RESEED_INTERVAL`] calls.
    pool: [u8; 32],

    /// Counts [`get_random_bytes`] calls since the last reseed. Plain `u32`
    /// because only one CPU ever reads or writes this field.
    call_count: u32,

    /// Byte offset (0, 8, 16, 24) into the pool for the next draw's binding
    /// word. Advances by 8 on each draw and wraps at 32, ensuring successive
    /// draws bind against different pool regions.
    pool_cursor: usize,
}

#[allow(dead_code)]
impl CpuRngState {
    /// Returns a zeroed, uninitialized state.
    ///
    /// A zeroed pool is safe, as it is always XOR-folded with hardware entropy
    /// before first use by [`init_cpu`].
    const fn init() -> Self {
        Self {
            pool: [0u8; 32],
            call_count: 0,
            pool_cursor: 0,
        }
    }
}

/// Per-CPU RNG state table, indexed by APIC ID via [`CpuId`].
///
/// Each CPU initializes its own slot via [`init_cpu`] and thereafter accesses
/// it exclusively through [`CpuId::current()`]. No cross-CPU access ever
/// occurs.
static RNG_STATE: PerCpu<CpuRngState> = PerCpu::new();

/// Checks for RDRAND support: CPUID leaf 1, ECX bit 30. The result does not
/// change at runtime; callers on hot paths should call this once and cache it
/// locally.
/// 
/// # Returns
/// 
/// Returns `true` if the executing CPU advertises RDRAND support.
#[inline]
fn has_rdrand() -> bool {
    let ecx: u32;

    unsafe {
        asm!(
            "mov {save}, rbx",
            "cpuid",
            "mov rbx, {save}",
            save = out(reg) _,
            inout("eax") 1u32 => _,
            out("ecx") ecx,
            out("edx") _,
            options(nostack, nomem, preserves_flags),
        );
    }

    (ecx >> 30) & 1 == 1
}

/// Checks for RDSEED support: CPUID leaf 7, sub-leaf 0, EBX bit 18.
/// 
/// # Returns
/// 
/// Returns `true` if the executing CPU advertises RDSEED support.
#[inline]
fn has_rdseed() -> bool {
    let ebx: u32;

    unsafe {
        asm!(
            "mov {save}, rbx",
            "xor ecx, ecx",
            "cpuid",
            "mov {out:e}, ebx",
            "mov rbx, {save}",
            save = out(reg) _,
            out = out(reg) ebx,
            inout("eax") 7u32 => _,
            out("ecx") _,
            out("edx") _,
            options(nostack, nomem, preserves_flags),
        );
    }

    (ebx >> 18) & 1 == 1
}

/// Attempts to read one 64-bit value from RDRAND, retrying up to
/// [`HW_RETRY_LIMIT`] times.
///
/// RDRAND sets CF=1 on success, CF=0 on transient failure. `setc` captures CF
/// into a byte register without branching inside the `asm!` block.
///
/// # Returns
///
/// `Some(value)` on success, `None` if all retries fail (should not occur on
/// healthy hardware under normal conditions).
#[inline]
fn rdrand64() -> Option<u64> {
    let mut value: u64;
    let mut ok: u8;

    for _ in 0..HW_RETRY_LIMIT {
        unsafe {
            asm!(
                "rdrand {val}",
                "setc {ok}",
                val = out(reg) value,
                ok = out(reg_byte) ok,
                options(nostack, nomem),
            );
        }

        if ok != 0 {
            return Some(value);
        }

        core::hint::spin_loop();
    }

    None
}

/// Attempts to read one 64-bit value from RDSEED, retrying up to
/// [`HW_RETRY_LIMIT`] times.
///
/// RDSEED samples the CPU's raw entropy conditioner directly and may stall
/// more often than RDRAND under heavy multi-core load, as all cores share
/// the same physical entropy source. The retry-with-pause pattern gives the
/// hardware time to accumulate a fresh sample between attempts.
///
/// # Returns
///
/// `Some(value)` on success, `None` after all retries fail.
#[inline]
fn rdseed64() -> Option<u64> {
    let mut value: u64;
    let mut ok: u8;

    for _ in 0..HW_RETRY_LIMIT {
        unsafe {
            asm!(
                "rdseed {val}",
                "setc {ok}",
                val = out(reg) value,
                ok = out(reg_byte) ok,
                options(nostack, nomem),
            );
        }

        if ok != 0 {
            return Some(value);
        }

        core::hint::spin_loop();
    }

    None
}

/// Reads the 64-bit invariant TSC.
///
/// Constructed from the EDX:EAX pair that `RDTSC` produces. Used as a
/// timing jitter source on hardware without RDRAND/RDSEED.
/// 
/// # Returns
/// 
/// Returns the 64-bit invariant TSC.
#[inline]
fn rdtsc() -> u64 {
    let lo: u32;
    let hi: u32;

    unsafe {
        asm!(
            "rdtsc",
            out("eax") lo,
            out("edx") hi,
            options(nostack, nomem, preserves_flags),
        );
    }

    ((hi as u64) << 32) | (lo as u64)
}

/// Produces a 64-bit entropy estimate from TSC/HPET timing jitter.
///
/// This is the fallback path for hardware without RDRAND or RDSEED. It
/// exploits the measurable, non-reproducible jitter between two independent
/// clock domains:
///
///   - TSC: the CPU's invariant timestamp counter, ticking at a fixed
///     multiple of the core clock;
///   - HPET: an external crystal oscillator on a completely separate clock
///     domain, accessed over the memory bus.
///
/// The latency of the HPET MMIO read varies with pipeline state, cache
/// occupancy, memory bus arbitration, and SMI activity - none of which are
/// controllable or observable by an attacker in the general case.
///
/// The three samples (TSC before, HPET, TSC after) are mixed through the
/// wyHash finalizer, which provides strong avalanche: a single bit change in
/// any input affects all 64 output bits. The two magic constants are the
/// published wyHash primes, chosen for their verified mixing properties.
///
/// Note: this is not cryptographically strong in isolation! On an idle system
/// with a deterministic workload, the jitter may be low. Treat it as a
/// weak-but-non-zero entropy source that is meaningfully better than a
/// compile-time constant or a pure TSC.
/// 
/// # Arguments
///
/// * `hpet` - Reference to the initialized `HpetInfo`.
/// 
/// # Returns
/// 
/// Returns a 64-bit entropy estimate from TSC/HPET timing jitter.
///
/// # Safety
///
/// `hpet` must satisfy `HpetInfo::read_counter`'s safety requirements:
/// `parse_hpet` must have completed, and the HPET MMIO page must be mapped.
#[inline]
unsafe fn jitter_sample(hpet: &crate::hardware_manager::HpetInfo) -> u64 {
    let tsc_before = rdtsc();
    let hpet_val = unsafe { hpet.read_counter() };
    let tsc_after = rdtsc();

    // wyHash magic primes (from the wyHash reference implementation).
    const P0: u64 = 0x2d358dccaa6c78a5;
    const P1: u64 = 0x8bb84b93962eacc9;

    /// Multiply two u64s, XOR-fold the 128-bit result to 64 bits.
    /// This is the core non-linear mixing step of wyHash.
    #[inline(always)]
    fn wymix(a: u64, b: u64) -> u64 {
        let r = (a as u128).wrapping_mul(b as u128);
        (r as u64) ^ ((r >> 64) as u64)
    }

    // Fold three samples into two words, then run two finalizer rounds.
    let a = wymix(tsc_before ^ P0, hpet_val ^ P1);
    let b = wymix(tsc_after  ^ P0, hpet_val.rotate_left(17) ^ P1);

    a ^ b
}

/// Gets a 64-bit word from the best available entropy source.
/// 
/// Source priority:
///   1. RDSEED (if `prefer_seed` is true and RDSEED is available)
///   2. RDRAND (if available)
///   3. TSC/HPET jitter (if `hpet` is `Some`)
///   4. TSC only (absolute last resort)
///
/// `prefer_seed` should be `true` during pool seeding (where quality matters
/// most) and `false` during bulk output generation (where RDRAND is fast
/// enough and RDSEED may stall).
/// 
/// # Arguments
///
/// * `prefer_seed` - Pass `true` for seeding preference if RDSEED is available.
/// * `hpet`        - Reference to the initialized `HpetInfo`, if any.
/// 
/// # Returns
/// 
/// Returns a 64-bit word from the best available entropy source.
#[inline]
unsafe fn best_hardware_word(
    prefer_seed: bool,
    hpet: Option<&crate::hardware_manager::HpetInfo>,
) -> u64 {
    // Jitter closure - only constructed if we reach the fallback path
    let jitter = || -> u64 {
        match hpet {
            Some(h) => unsafe { jitter_sample(h) },
            None => rdtsc(),
        }
    };

    if prefer_seed && has_rdseed() {
        // RDSEED preferred for seeding; fall through to RDRAND on exhaustion
        rdseed64()
            .or_else(rdrand64)
            .unwrap_or_else(jitter)
    }
    else if has_rdrand() {
        rdrand64().unwrap_or_else(jitter)
    }
    else {
        jitter()
    }
}

/// XOR-mixes all 32 bytes of `src` into `dst` in-place.
///
/// XOR-mixing is information-preserving: if either argument contains genuine
/// entropy, the output retains at least as much entropy as the better of the
/// two. Used to fold fresh seed material into an existing pool without
/// discarding accumulated state.
/// 
/// # Arguments
///
/// * `dst` - Target destination buffer for the XOR operation and storage.
/// * `src` - Source buffer for the XOR operation.
#[inline]
fn xor_mix(dst: &mut [u8; 32], src: &[u8; 32]) {
    // Iterate over 8-byte (u64) chunks to give the compiler the best chance
    // of emitting a handful of XOR instructions rather than 32 byte ops.
    for i in 0..4 {
        let offset = i * 8;
        let d = u64::from_le_bytes(dst[offset..offset + 8].try_into().unwrap());
        let s = u64::from_le_bytes(src[offset..offset + 8].try_into().unwrap());
        dst[offset..offset + 8].copy_from_slice(&(d ^ s).to_le_bytes());
    }
}

/// Fills a 32-byte buffer with entropy from the best available hardware source.
///
/// Uses `prefer_seed = true`, so RDSEED is attempted for each word when
/// available. Called during both initial seeding and periodic reseeds.
/// 
/// # Arguments
///
/// * `out`  - Destination buffer to store entropy.
/// * `hpet` - Reference to the initialized `HpetInfo`, if any.
///
/// # Safety
///
/// `hpet` must satisfy `HpetInfo::read_counter`'s safety requirements if
/// `Some`.
#[inline]
unsafe fn fill_seed_buffer(
    out: &mut [u8; 32],
    hpet: Option<&crate::hardware_manager::HpetInfo>,
) {
    for chunk in 0..4usize {
        let word = unsafe { best_hardware_word(true, hpet) };
        out[chunk * 8..chunk * 8 + 8].copy_from_slice(&word.to_le_bytes());
    }
}

/// Draws 8 bytes from the calling CPU's pool with interrupt-safe protection.
///
/// Disables interrupts for the duration of the draw to prevent an ISR on the
/// same core from re-entering [`get_random_bytes`] and observing a partially
/// advanced cursor or pool state. Interrupts are restored immediately after
/// the draw regardless of how the function exits.
///
/// The draw itself:
///   - Reads 8 bytes (one u64) from the pool at `pool_cursor`;
///   - Draws one hardware word from RDRAND (or jitter fallback);
///   - Outputs `hardware_word ^ pool_word`, binding the output to both sources;
///   - Advances `pool_cursor` by 8, wrapping at 32.
///
/// An attacker who knows the RDRAND output cannot predict the output without
/// also knowing the pool state, and vice versa.
///
/// # Arguments
///
/// * `cpu` - The `CpuId` of the calling CPU.
#[inline]
fn draw_8_bytes(cpu: CpuId) -> [u8; 8] {
    // Disable interrupts for the duration of the draw. This prevents an ISR
    // on this core from re-entering get_random_bytes mid-draw. We do not need
    // a spinlock here because we are the only CPU that can touch this slot.
    let flags = disable_interrupts_save();

    // SAFETY: We are on `cpu`, interrupts are disabled, and no ISR can
    // preempt us. No other CPU ever writes this slot. Exclusive access holds.
    let state = unsafe { RNG_STATE.get_mut(cpu) };

    // Read the 8-byte pool word at the current cursor position
    let cursor = state.pool_cursor;
    let pool_word = u64::from_le_bytes(
        state.pool[cursor..cursor + 8].try_into().unwrap()
    );

    // Advance cursor, wrapping at the end of the 32-byte pool
    state.pool_cursor = (cursor + 8) & 0x18;  // equivalent to % 32

    // Draw a hardware word. No HPET here (not cached globally); RDRAND or
    // TSC-only fallback.
    //
    // SAFETY: No HPET reference; jitter_sample will not be called.
    let hardware_word = unsafe { best_hardware_word(false, None) };

    // Bind: output is hardware_word XOR pool_word. Neither source alone
    // determines the output.
    let output = hardware_word ^ pool_word;

    // Restore interrupts and return the drawn 8 bytes
    restore_interrupts(flags);
    output.to_le_bytes()
}

// =============================================================================
// Public API
// =============================================================================

/// Initializes the RNG state for the calling CPU.
///
/// Must be called exactly once per CPU, from that CPU's own init path, after
/// the HPET has been initialized. For the BSP, this is during early kernel
/// init; for each AP, it is called inside its bring-up sequence before the AP
/// is considered ready.
///
/// After this call, [`get_random_bytes`] is safe to use on the calling CPU.
///
/// # Arguments
///
/// * `hpet` - Reference to the initialized `HpetInfo`. Used for the jitter
///   fallback path on hardware without RDRAND/RDSEED, and to improve initial
///   seed quality on all hardware. Safe to pass even when RDRAND is available.
///
/// # Safety
///
/// - `hpet` must satisfy `HpetInfo::read_counter`'s safety requirements.
/// - Must be called from the CPU that owns the slot being initialized (i.e.,
///   [`CpuId::current()`] must match the slot being written). This is the
///   normal boot contract: each CPU initializes its own per-CPU state.
/// - Must not be called concurrently with any other access to this CPU's
///   [`RNG_STATE`] slot (guaranteed during single-threaded CPU bring-up).
pub unsafe fn init_cpu(hpet: &crate::hardware_manager::HpetInfo) {
    let cpu = CpuId::current();

    // Build the initial pool from hardware entropy before writing the slot,
    // so we never store an all-zero pool in the live state.
    let mut initial_pool = [0u8; 32];
    unsafe { fill_seed_buffer(&mut initial_pool, Some(hpet)) };

    // Initialize the per-CPU slot. `PerCpu::init` writes the value and sets
    // the initialized flag with Release ordering, ensuring the pool contents
    // are visible to any subsequent Acquire load in `get` or `get_mut`.
    unsafe {
        RNG_STATE.init(cpu, CpuRngState {
            pool:         initial_pool,
            call_count:   0,
            pool_cursor:  0,
        })
    };
}

/// Fills a buffer with cryptographically secure random bytes.
///
/// This function:
///   1. Snapshots [`CpuId::current()`] once and holds it for the entire call,
///      avoiding repeated CPUID serialization on the hot path.
///   2. Checks whether a periodic reseed is due (`call_count % RESEED_INTERVAL
///      == 0`) and, if so, XOR-folds 32 fresh hardware bytes into the pool.
///   3. Fills `buffer` in 8-byte chunks by drawing one hardware word per chunk
///      and XOR-binding it against the current pool word at `pool_cursor`.
///      The cursor advances by 8 bytes (wrapping at 32) on each draw, so
///      successive draws bind against distinct pool regions.
///   4. Handles a trailing partial chunk (when `buffer.len()` is not a multiple
///      of 8) by drawing one extra word and copying only the needed bytes.
///
/// Each 8-byte draw is delegated to [`draw_8_bytes`], which disables interrupts
/// for the duration of the draw to prevent an ISR on the same core from
/// re-entering this function mid-draw and observing or corrupting the
/// `pool_cursor` / pool state.
///
/// # Arguments
///
/// * `buffer` - Destination buffer. Any length is accepted. An empty slice is
///   a no-op.
///
/// # Panics
///
/// Panics if `init_cpu` has not been called for the current CPU (forwarded
/// from `PerCpu::get_mut`'s assertion on the initialized flag).
pub fn get_random_bytes(buffer: &mut [u8]) {
    if buffer.is_empty() {
        return;
    }

    // Snapshot the CPU ID once. `CpuId::current()` executes CPUID, a
    // serializing instruction. Calling it once per `get_random_bytes`
    // invocation (rather than per 8-byte draw) keeps the overhead constant
    // regardless of buffer length.
    let cpu = CpuId::current();

    // Periodic reseed: check before the first draw so the pool is fresh
    // at the start of this call when the counter wraps.
    {
        // SAFETY: We are on `cpu`, and interrupts are about to be managed per
        // draw. Reading call_count here races with nothing because only this
        // CPU writes it.
        let state = unsafe { RNG_STATE.get_mut(cpu) };
        if state.call_count % RESEED_INTERVAL == 0 {
            let mut fresh = [0u8; 32];
            // No HPET reference here (not stored globally). RDRAND is the
            // reseed source between init_cpu calls. For HPET-quality periodic
            // reseeds, call `reseed_cpu(hpet)` explicitly.
            unsafe { fill_seed_buffer(&mut fresh, None) };
            xor_mix(&mut state.pool, &fresh);
        }
        state.call_count = state.call_count.wrapping_add(1);
    }

    // Fill the buffer in 8-byte chunks
    let mut remaining = buffer;
    while remaining.len() >= 8 {
        let word = draw_8_bytes(cpu);
        remaining[..8].copy_from_slice(&word);
        remaining = &mut remaining[8..];
    }

    // Trailing partial chunk (0..7 bytes)
    if !remaining.is_empty() {
        let word = draw_8_bytes(cpu);
        remaining.copy_from_slice(&word[..remaining.len()]);
    }
}

/// Forces an immediate reseed of the calling CPU's pool using the HPET jitter
/// source.
///
/// Produces higher-quality entropy than the automatic RDRAND-only reseed path
/// in [`get_random_bytes`]. Call this from a context where `HpetInfo` is
/// available - e.g., immediately after [`init_cpu`] or after a suspend/
/// resume cycle.
///
/// # Arguments
///
/// * `hpet` - Reference to the initialized `HpetInfo`.
///
/// # Safety
///
/// Same requirements as [`init_cpu`]: must be called from the CPU whose slot
/// is being reseeded, and `hpet` must satisfy `HpetInfo::read_counter`'s
/// safety contract.
#[allow(dead_code)]
pub unsafe fn reseed_cpu(hpet: &crate::hardware_manager::HpetInfo) {
    let cpu = CpuId::current();
    let mut fresh = [0u8; 32];
    unsafe { fill_seed_buffer(&mut fresh, Some(hpet)) };

    let flags = disable_interrupts_save();
    let state  = unsafe { RNG_STATE.get_mut(cpu) };
    xor_mix(&mut state.pool, &fresh);
    restore_interrupts(flags);
}

/// Convenience wrapper around [`get_random_bytes`] that allocates and returns
/// an owned `Vec<u8>` of `length` cryptographically secure random bytes.
///
/// Prefer the slice form `get_random_bytes(&mut buffer)` in contexts where the
/// output length is known at compile time or heap allocation should be
/// avoided. This wrapper exists for call sites where an owned buffer is
/// needed directly, such as key generation or nonce/IV construction.
///
/// # Arguments
///
/// * `length` - Number of random bytes to generate.
///
/// # Returns
///
/// An owned `Vec<u8>` of length `length` filled with cryptographically secure
/// random bytes.
///
/// # Panics
///
/// Panics if [`init_cpu`] has not been called for the current CPU.
#[allow(dead_code)]
pub fn get_random_bytes_vec(length: usize) -> alloc::vec::Vec<u8> {
    let mut buffer = alloc::vec![0u8; length];
    get_random_bytes(&mut buffer);

    buffer
}
