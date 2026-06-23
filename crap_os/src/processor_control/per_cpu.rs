//! Per-CPU Storage Primitive
//!
//! [`PerCpu<T>`] provides a fixed-size array of `T` slots, one per logical
//! CPU, indexed by [`CpuId`]. It is the foundation for all per-CPU state in
//! the kernel: scheduler run queues, current-task pointers, tick counters, TSS,
//! etc.
//!
//! Slots are stored as [`MaybeUninit<T>`] and tracked by a per-slot
//! [`AtomicBool`] initialized flag. This avoids requiring `T: Default` and
//! makes the uninitialized state explicit rather than relying on a sentinel
//! value.
//!
//! The struct is `Sync` (safe to share across CPUs) because:
//!   - Each CPU only writes to its own slot (enforced by convention and by
//!     the `unsafe` contract on `init` and `get_mut`).
//!   - `get` provides a shared reference only after the slot is confirmed
//!     initialized, and only to the caller's own slot in normal usage.
//!
//! # SMP safety rules
//!
//! - `init` must be called exactly once per CPU, from that CPU's own init
//!   path, before any call to `get` or `get_mut` for that slot.
//! - `get_mut` is only safe to call for your own CPU's slot, and only when
//!   no other reference to that slot is alive. During single-threaded init,
//!   this is trivially true.
//! - Cross-CPU writes (one CPU mutating another CPU's slot) are never safe
//!   through this API. Use atomics or message-passing for that.

use core::cell::UnsafeCell;
use core::mem::MaybeUninit;
use core::sync::atomic::{AtomicBool, Ordering};
use super::topology::MAX_CPUS;

/// CPU Identifier
///
/// [`CpuId`] is a newtype over a `u32` APIC ID that serves as the index into
/// all per-CPU data structures. It is the only way to address a specific CPU
/// throughout the kernel; passing raw integers is intentionally not supported.
///
/// [`CpuId`] is `Copy` and cheap to pass by value. The inner `u32` is the xAPIC
/// hardware ID, which fits in a `u8` on all current x86-64 systems, but is
/// stored as `u32` for x2APIC forward-compatibility (x2APIC IDs are 32-bit).
/// 
/// We use the CPUID instead of the LAPIC ID register because the latter (MMIO
/// at `0xFEE00000 + 0x020`) requires `init_apic` to have run, and the MMIO
/// window to be mapped. CPUID leaf 1 EBX bits [31:24] return the same initial
/// APIC ID unconditionally and are always available, making `CpuId::current()`
/// safe to call even during the earliest init path.
#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub struct CpuId(u32);
 
impl CpuId {
    /// Returns the `CpuId` of the CPU executing this call.
    ///
    /// Reads the initial APIC ID from CPUID leaf 1, EBX bits [31:24].
    /// This value is assigned by firmware and does not change at runtime.
    ///
    /// # Cost
    ///
    /// CPUID is a serializing instruction and has non-trivial cost (~20–100
    /// cycles depending on the microarchitecture). Callers on hot paths should
    /// cache the result locally rather than calling this repeatedly.
    #[inline]
    pub fn current() -> Self {
        let apic_id: u32;
        
        // SAFETY: CPUID is always available on x86-64. Leaf 1 is guaranteed
        // to exist on any processor that supports the x86-64 ISA. RBX is used
        // internally by LLVM and cannot be used as an operand for inline asm;
        // so, we let the compiler pick a scratch register and move it manually.
        unsafe {
            core::arch::asm!(
                "mov {save}, rbx",   // Save rbx to a 64-bit compiler reg
                "cpuid",             // Clobbers eax/ebx/ecx/edx
                "shr ebx, 24",       // Shift APIC ID bits [31:24] down to [7:0]
                "movzx {out:e}, bl", // Zero-extend the low byte into output reg
                "mov rbx, {save}",   // Restore rbx
                save = out(reg) _,   // Compiler picks a 64-bit scratch reg
                out  = out(reg) apic_id,
                in("eax") 1u32,
                out("ecx") _,
                out("edx") _,
                options(nostack, nomem),
            );
        }

        CpuId(apic_id)
    }
 
    /// Constructs a `CpuId` from a raw APIC ID value.
    ///
    /// Used when iterating over [`super::topology::CpuTopology`] entries to
    /// construct `CpuId`s for each known CPU during init. We should use
    /// [`CpuId::current()`] in all other contexts.
    #[inline]
    pub const fn from_apic_id(apic_id: u32) -> Self {
        CpuId(apic_id)
    }
 
    /// Returns the raw APIC ID value.
    ///
    /// Should only be needed when interfacing with hardware (e.g., programming
    /// an I/O APIC redirection table entry destination field). We should pass
    /// `CpuId` by value everywhere else.
    #[inline]
    pub const fn apic_id(self) -> u32 {
        self.0
    }
}

/// A per-CPU storage cell holding one `T` per logical CPU, indexed by
/// [`CpuId`].
///
/// Lives in a `static`; all methods take `&self`. Interior mutability is
/// provided by `UnsafeCell`, with safety enforced by the calling convention
/// (each CPU only touches its own slot).
pub struct PerCpu<T> {
    /// The actual per-CPU values, uninitialized until `init` is called for
    /// that slot. `UnsafeCell` provides interior mutability; `MaybeUninit`
    /// avoids requiring `T: Default` and makes the uninitialized state
    /// explicit.
    slots: [UnsafeCell<MaybeUninit<T>>; MAX_CPUS],

    /// Tracks which slots have been initialized. Indexed identically to
    /// `slots`. Using an atomic allows `get` to check initialization without
    /// holding any lock.
    initialized: [AtomicBool; MAX_CPUS],
}

// SAFETY: `PerCpu<T>` can be shared across CPUs as long as `T: Send`.
// Each CPU accesses only its own slot, and `init` uses `Acquire`/`Release`
// ordering to ensure the written value is visible before the initialized flag
// is observed as true.
unsafe impl<T: Send> Sync for PerCpu<T> {}
unsafe impl<T: Send> Send for PerCpu<T> {}

// Helper macro to construct the two large arrays as const. Rust does not yet
// support const array initialization with non-Copy types via a closure, so we
// use a macro to repeat the initializer MAX_CPUS times. Both
// `UnsafeCell<MaybeUninit<T>>` and `AtomicBool` are valid in const contexts.
macro_rules! repeat_64 {
    ($expr:expr) => {[
        $expr,$expr,$expr,$expr,$expr,$expr,$expr,$expr,  //  0.. 7
        $expr,$expr,$expr,$expr,$expr,$expr,$expr,$expr,  //  8..15
        $expr,$expr,$expr,$expr,$expr,$expr,$expr,$expr,  // 16..23
        $expr,$expr,$expr,$expr,$expr,$expr,$expr,$expr,  // 24..31
        $expr,$expr,$expr,$expr,$expr,$expr,$expr,$expr,  // 32..39
        $expr,$expr,$expr,$expr,$expr,$expr,$expr,$expr,  // 40..47
        $expr,$expr,$expr,$expr,$expr,$expr,$expr,$expr,  // 48..55
        $expr,$expr,$expr,$expr,$expr,$expr,$expr,$expr,  // 56..63
    ]};
}

impl<T> PerCpu<T> {
    /// Creates a new and uninitialized `PerCpu<T>`.
    ///
    /// All slots are in the uninitialized state. `init` must be called for
    /// each CPU's slot before any access via `get` or `get_mut`. The function
    /// is `const`, so this can be used to initialize `static` variables.
    pub const fn new() -> Self {
        Self {
            slots: repeat_64!(UnsafeCell::new(MaybeUninit::uninit())),
            initialized: repeat_64!(AtomicBool::new(false)),
        }
    }

    /// Initializes the slot for `cpu`, writing `value` into it.
    ///
    /// Must be called exactly once per CPU, from that CPU's own init path,
    /// before any call to `get` or `get_mut` for that slot.
    /// 
    /// # Arguments
    /// 
    /// * `cpu`   - The [`CpuId`] of the CPU slot to initialize.
    /// * `value` - Value to write to the per-cpu storage slot.
    ///
    /// # Panics
    ///
    /// Panics if `cpu.apic_id()` is >= `MAX_CPUS`, or if this slot has
    /// already been initialized (double-init would be a bug).
    ///
    /// # Safety
    ///
    /// The caller must ensure no concurrent access to this slot is occurring
    /// at the time of the call. During normal BSP/AP bring-up, this is
    /// guaranteed, as each CPU initializes only its own slot while that slot
    /// is otherwise quiescent.
    pub unsafe fn init(&self, cpu: CpuId, value: T) {
        let index = cpu.apic_id() as usize;
        assert!(
            index < MAX_CPUS, "CpuId {} exceeds MAX_CPUS ({})", index, MAX_CPUS);
        assert!(
            !self.initialized[index].load(Ordering::Relaxed),
            "PerCpu slot {} already initialized",
            index
        );

        // SAFETY: We are the only writer to this slot (enforced by the caller's
        // contract). `MaybeUninit::write` does not read the old value, so there
        // is no UB from writing to an uninitialized cell.
        unsafe { (*self.slots[index].get()).write(value) };

        // Release ordering: ensures the write to `slots[index]` is visible to
        // any CPU that subsequently observes `initialized[index]` as true via
        // an Acquire load in `get`.
        self.initialized[index].store(true, Ordering::Release);
    }

    /// Gets a shared reference to the slot for a given CPU.
    /// 
    /// # Arguments
    /// 
    /// * `cpu` - The [`CpuId`] of the CPU slot to reference.
    /// 
    /// # Returns
    /// 
    /// Returns a shared reference to the slot for `cpu`.
    ///
    /// # Panics
    ///
    /// Panics if `cpu.apic_id()` is >= `MAX_CPUS`, or if the slot has not
    /// yet been initialized via `init`.
    ///
    /// # Lifetime
    ///
    /// The returned reference is tied to `&self`, so it cannot outlive the
    /// `PerCpu`. For a `static PerCpu`, this is effectively `'static`.
    #[inline]
    pub fn get(&self, cpu: CpuId) -> &T {
        let index = cpu.apic_id() as usize;
        assert!(
            index < MAX_CPUS,
            "CpuId {} exceeds MAX_CPUS ({})",
            index,
            MAX_CPUS
        );

        // Acquire ordering: pairs with the Release store in `init`, ensuring
        // we see the fully written value.
        assert!(
            self.initialized[index].load(Ordering::Acquire),
            "PerCpu slot {} accessed before init",
            index
        );

        // SAFETY: `initialized[index]` is true, so `slots[index]` has been
        // fully written by `init`. No mutable reference to this slot can exist
        // as long as the caller obeys the SMP safety rules (own-slot writes
        // only, no concurrent `get_mut`).
        unsafe { (*self.slots[index].get()).assume_init_ref() }
    }

    /// Gets a mutable reference to the slot for a given CPU.
    /// 
    /// # Arguments
    /// 
    /// * `cpu` - The [`CpuId`] of the CPU slot to reference.
    /// 
    /// # Returns
    /// 
    /// Returns a mutable reference to the slot for `cpu`.
    ///
    /// # Panics
    ///
    /// Panics if `cpu.apic_id()` is >= `MAX_CPUS`, or if the slot has not
    /// yet been initialized via `init`.
    ///
    /// # Safety
    ///
    /// The caller must ensure:
    /// 1. No other reference (shared or mutable) to this slot is alive.
    /// 2. This is called from the CPU that owns the slot, or during a
    ///    single-threaded init phase where no other CPU can access the slot.
    ///
    /// Violating either condition is undefined behaviour (data race).
    #[inline]
    pub unsafe fn get_mut(&self, cpu: CpuId) -> &mut T {
        let index = cpu.apic_id() as usize;
        assert!(
            index < MAX_CPUS,
            "CpuId {} exceeds MAX_CPUS ({})",
            index,
            MAX_CPUS
        );
        assert!(
            self.initialized[index].load(Ordering::Acquire),
            "PerCpu slot {} accessed before init",
            index
        );

        // SAFETY: Caller guarantees exclusive access to this slot.
        unsafe { (*self.slots[index].get()).assume_init_mut() }
    }

    /// Checks if the slot for a given CPU has been initialized.
    /// 
    /// Useful for defensive checks and debug assertions. Does not provide
    /// any ordering guarantee beyond the `Acquire` on the flag itself.
    /// 
    /// # Arguments
    /// 
    /// * `cpu` - The [`CpuId`] of the CPU slot to check.
    /// 
    /// # Returns
    /// 
    /// Returns `true` if the slot for `cpu` has been initialized, `false`
    /// otherwise.
    /// 
    /// TODO: Remove dead_code marker when ths gets used later on.
    #[allow(dead_code)]
    #[inline]
    pub fn is_initialized(&self, cpu: CpuId) -> bool {
        let index = cpu.apic_id() as usize;
        if index >= MAX_CPUS {
            return false;
        }

        self.initialized[index].load(Ordering::Acquire)
    }

    /// Convenience wrapper: equivalent to `self.get(CpuId::current())`.
    ///
    /// Reads the current CPU's APIC ID via CPUID on each call. The result
    /// should be cached if calling from a hot path.
    #[inline]
    pub fn current(&self) -> &T {
        self.get(CpuId::current())
    }

    /// Convenience wrapper: equivalent to
    /// `unsafe { self.get_mut(CpuId::current()) }`.
    ///
    /// # Safety
    ///
    /// Same contract as [`get_mut`](Self::get_mut).
    #[inline]
    pub unsafe fn current_mut(&self) -> &mut T {
        unsafe { self.get_mut(CpuId::current()) }
    }
}
