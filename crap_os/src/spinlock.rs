//! Spinlock and IRQ Safety Module
//!
//! This module contains spinlock implementations for synchronizing critical
//! globals. These are mutual-exclusion primitives that busy-wait (spin) until
//! the lock is available. They are suitable for short critical sections in a
//! kernel context. Plain spinlocks are not safe to use in both thread and
//! interrupt context simultaneously. If an interrupt fires while the lock is
//! held on a current CPU, and the interrupt handler also tries to acquire
//! the same lock, we would deadlock - bad news bear.
//!
//! SpinLock<T> is the core primitive here. It spins on an AtomicBool using a
//! test-and-test-and-set (TTAS) loop for cache efficiency. IrqSpinLock<T> then
//! wraps SpinLock and disables CPU hardware interrupts while held, preventing
//! deadlock between a thread and an interrupt service routine competing for
//! the same lock. StaticSpinLock<T> and StaticIrqSpinLock<T> are thin wrappers
//! that manually implement Sync so these can be safe to put in a static
//! variable.
//!
//! Each lock has a corresponding RAII (Resource Acquisition Is Initialization)
//! guard type that automatically releases the lock (for IrqSpinLock, it also
//! restores interrupt state) when it is dropped.

// Needed for interior mutability; it tells the compiler that the value inside
// may be mutated through a shared reference.
use core::cell::UnsafeCell;

// This emits a CPU pause or yield hint (e.g., x86 PAUSE) that improves
// performance and power usage inside a busy-wait spin loop.
use core::hint;

// These let us write guard to access the inner T transparently, just like
// a normal reference.
use core::ops::{Deref, DerefMut};

// The AtomicBool is a boolean that can be read and written atomically across
// threads without a data race. The Ordering controls how surrounding memory
// operations are ordered relative to the atomic operation.
use core::sync::atomic::{AtomicBool, Ordering};

// `PhantomData<*mut ()>` is a zero-sized marker that only influences the
// compiler's variance, Send, and Sync analysis. A raw pointer `*mut ()` is
// neither Send nor Sync, so embedding `PhantomData<*mut ()>` in a struct makes
// that struct likewise !Send and !Sync unless we explicitly say otherwise.
use core::marker::PhantomData;

// =============================================================================
// SpinLock<T>
// =============================================================================

/// A simple, non-reentrant spinlock protecting a value of type T.
///
/// When a thread wants access to T, it atomically sets `locked` to true.
/// If `locked` is already true, the thread busy-waits (spins) until it
/// becomes false, then tries again.
pub struct SpinLock<T> {
    locked: AtomicBool,   // Represents whether the lock is currently being held
    data: UnsafeCell<T>,  // The data being protected by SpinLock
}

// SAFETY: SpinLock uses an atomic to guard exclusive access to T. As long as
// `T: Send` (i.e., it is safe to move T between threads), SpinLock is safe
// to both send to another thread (Send) and share by reference across threads
// (Sync).
unsafe impl<T: Send> Send for SpinLock<T> {}
unsafe impl<T: Send> Sync for SpinLock<T> {}

#[allow(dead_code)]
impl<T> SpinLock<T> {
    /// Creates a new unlocked SpinLock that wraps some data. This const
    /// function is evaluated at compile time, which is necessary to initialize
    /// static variables.
    /// 
    /// # Arguments
    /// 
    /// * `data` - Data to be protected by SpinLock.
    #[inline]
    pub const fn new(data: T) -> Self {
        Self {
            locked: AtomicBool::new(false), // Start unlocked
            data: UnsafeCell::new(data),    // Wrap data for interior mutability
        }
    }

    /// Acquires the lock, spinning until it becomes available.
    ///
    /// The Spin strategy uses a "test-and-test-and-set" (TTAS) pattern for
    /// cache efficiency:
    /// 1. Try an atomic compare-exchange to go from false to true.
    /// 2. If that fails, busy-wait reading `locked` with a relaxed load and
    ///    a `spin_loop()` hint until it looks unlocked, then retry step 1.
    ///
    /// This is more cache-friendly than spinning exclusively on the expensive
    /// atomic compare-exchange because a plain load can be satisfied by the
    /// local CPU cache without causing cache-line invalidations on other cores.
    /// 
    /// # Returns
    /// 
    /// Returns a SpinLockGuard that releases the lock when dropped.
    /// 
    /// # Deadlock Warning
    /// Calling `lock()` from an interrupt handler while the lock is already
    /// held on the interrupted thread will spin forever. 
    #[inline]
    pub fn lock(&self) -> SpinLockGuard<'_, T> {
        loop {
            // Fast path attempt to grab the lock immediately
            if let Ok(guard) = self.try_lock() {
                return guard;
            }

            // Lock was busy. Spin with a relaxed load until it looks free,
            // then loop back to the attempt at the top.
            while self.locked.load(Ordering::Relaxed) {
                // This emits the x86 PAUSE instruction (or its equivalent),
                // which reduces power consumption and avoids pipeline hazards
                // caused by tight spin loops.
                hint::spin_loop();
            }
        }
    }

    /// Attempts to acquire the lock once without spinning.
    ///
    /// Uses `compare_exchange` with `Acquire` ordering on success so that
    /// all memory writes performed before the lock was released are visible
    /// to us after we acquire it. The `Relaxed` on failure is fine here
    /// because we don't need any ordering guarantees when we don't take the
    /// lock.
    /// 
    /// # Returns
    /// 
    /// Returns `Ok(guard)` if the lock was free and is now acquired, or
    /// `Err(())` if the lock was already held by another thread.
    #[inline]
    pub fn try_lock(&self) -> Result<SpinLockGuard<'_, T>, ()> {
        self.locked
            // This is done atomically: if locked == false, set it to true and
            // return Ok(false). If it was already true, return Err(true); 
            // someone else has the lock.
            .compare_exchange(false, true, Ordering::Acquire, Ordering::Relaxed)
            // On success, build the guard. The `_not_send: PhantomData` marks
            // the guard as !Send, so it cannot be moved to another thread (the
            // lock must be released on the same CPU/thread that acquired it).
            .map(|_| SpinLockGuard { lock: self, _not_send: PhantomData })
            // Discard the actual boolean value as callers only need an Ok/Err.
            .map_err(|_| ())
    }

    /// Spot-checks if the lock is currently held. This is inherently racy, and
    /// by the time we inspect the result, the state may have changed. This is
    /// useful only for diagnostic and debugging purposes.
    /// 
    /// # Returns
    /// 
    /// Returns true if the lock is currently held by something, false if the
    /// lock is free to be acquired.
    #[inline]
    pub fn is_locked(&self) -> bool {
        // Relaxed here is sufficient, as we just want the current value with
        // no ordering guarantees relative to other operations.
        self.locked.load(Ordering::Relaxed)
    }

    /// Forcibly releases the lock without going through the normal RAII guard.
    ///
    /// # Safety
    /// 
    /// The caller must be the current lock owner and must ensure no
    /// SpinLockGuard for this lock is still alive. Calling this incorrectly
    /// can allow two threads to both believe they hold the lock simultaneously.
    /// Misuse can cause data races.
    #[inline]
    pub unsafe fn force_unlock(&self) {
        // Release ordering ensures all writes we made while holding the lock
        // are visible to the next thread that acquires it.
        self.locked.store(false, Ordering::Release);
    }

    /// Consumes the SpinLock and returns the inner value.
    ///
    /// This is safe because consuming `self` proves no other thread can still
    /// hold a reference to this lock.
    /// 
    /// # Returns
    /// 
    /// Returns the inner value of `T`.
    #[inline]
    pub fn into_inner(self) -> T {
        // `UnsafeCell::into_inner` unwraps the cell and returns the `T``.
        self.data.into_inner()
    }

    /// Gets a mutable reference to the inner value.
    ///
    /// This is safe because a `&mut SpinLock<T>` guarantees exclusive access,
    /// and no other thread could possibly hold the lock at the same time.
    /// 
    /// # Returns
    /// 
    /// Returns a mutable reference to the inner value.
    #[inline]
    pub fn get_mut(&mut self) -> &mut T {
        // UnsafeCell::get_mut` is safe to call when we have exclusive (`&mut`)
        // access.
        self.data.get_mut()
    }
}

// =============================================================================
// SpinLockGuard<'_, T>
// =============================================================================

/// Allows writing `*guard` or `guard.field` to access the protected `T` through
/// an immutable shared reference.
impl<T> Deref for SpinLockGuard<'_, T> {
    // The guard transparently looks like a `T`.
    type Target = T;

    #[inline]
    fn deref(&self) -> &T {
        // SAFETY: We hold the lock, so no other thread can mutate `data` right
        // now. The raw pointer returned by `UnsafeCell::get()` is valid for
        // the lifetime of the guard.
        unsafe { &*self.lock.data.get() }
    }
}

/// Allows writing `*guard = value` or `guard.field = value` to mutate the
/// protected `T` through a mutable reference.
impl<T> DerefMut for SpinLockGuard<'_, T> {
    #[inline]
    fn deref_mut(&mut self) -> &mut T {
        // SAFETY: We hold the lock exclusively, so mutable access is safe.
        // There is only one live mutable reference to the guard at this point.
        unsafe { &mut *self.lock.data.get() }
    }
}

/// Automatically releases the lock when the guard goes out of scope; this is
/// the RAII pattern.
impl<T> Drop for SpinLockGuard<'_, T> {
    #[inline]
    fn drop(&mut self) {
        // Release ordering ensures all writes in the critical section are
        // visible before the lock becomes available to the next acquirer.
        self.lock.locked.store(false, Ordering::Release);
    }
}

// =============================================================================
// SpinLockGuard<'a, T>
// =============================================================================

/// RAII guard returned by `SpinLock::lock()` or `SpinLock::try_lock()`.
///
/// Holding this guard means the lock is acquired. The lock is released
/// automatically when this value is dropped.
pub struct SpinLockGuard<'a, T> {
    // A reference back to the SpinLock that created this guard. The lifetime
    // `'a` ensures the guard cannot outlive the lock.
    lock: &'a SpinLock<T>,

    // Zero-sized marker that makes `SpinLockGuard` not `Send`. This prevents a
    // guard from being transferred to another thread. The lock must be released
    // on the same thread (for IRQ locks, also on the same CPU) that acquired
    // it. `PhantomData<*mut ()>` is used because `*mut ()` is !Send + !Sync,
    // which is exactly the property we want to inherit.
    _not_send: PhantomData<*mut ()>,
}

// =============================================================================
// StaticSpinLock<T>
// =============================================================================

/// A wrapper around SpinLock designed for static variables. Rust requires types
/// placed in `static` to implement `Sync`. The base `SpinLock<T>` already
/// implements `Sync` for `T: Send`, but sometimes we need a distinct type to
/// make intent explicit, or we have a `T` that is `Send` but whose `Sync` impl
/// is manually waived. `StaticSpinLock` purposefully does not implement `Send`.
/// A `static` item never needs to be moved between threads.
pub struct StaticSpinLock<T>(SpinLock<T>);

// `Sync` is required for statics. The lock can be safely accessed by
// multiple threads simultaneously, as the internal atomics make this work.
unsafe impl<T: Send> Sync for StaticSpinLock<T> {}

#[allow(dead_code)]
impl<T> StaticSpinLock<T> {
    /// Creates a new, unlocked StaticSpinLock. The const fn ensures it can be
    /// used to initialise static items at compile time.
    #[inline]
    pub const fn new(data: T) -> Self {
        Self(SpinLock::new(data))
    }

    /// Delegates to the inner `SpinLock::lock()` and spins until the lock is
    /// free, returning the RAII guard.
    #[inline]
    pub fn lock(&self) -> SpinLockGuard<'_, T> {
        self.0.lock()
    }

    /// Delegates to the inner `SpinLock::try_lock()` in a non-blocking attempt.
    #[inline]
    pub fn try_lock(&self) -> Result<SpinLockGuard<'_, T>, ()> {
        self.0.try_lock()
    }

    /// Delegates to the inner `SpinLock::is_locked()`.
    #[inline]
    pub fn is_locked(&self) -> bool {
        self.0.is_locked()
    }
}

// =============================================================================
// IrqSpinLock<T>
// =============================================================================

/// A spinlock that additionally disables CPU hardware interrupts while held.
///
/// This is required when both a normal thread and an interrupt service routine
/// (ISR) can both try to acquire the same lock. If a thread holds the lock and
/// an interrupt fires that also tries to take the lock, we'd get a deadlock
/// because the ISR cannot run to completion and the thread never gets CPU time
/// to release the lock. By disabling interrupts before spinning, we ensure
/// the ISR cannot preempt us while we own the lock.
pub struct IrqSpinLock<T>(SpinLock<T>);

// Same reasoning as for SpinLock above: it is safe to share across threads as
// long as `T: Send``.
unsafe impl<T: Send> Send for IrqSpinLock<T> {}
unsafe impl<T: Send> Sync for IrqSpinLock<T> {}

#[allow(dead_code)]
impl<T> IrqSpinLock<T> {
    /// Creates a new unlocked IrqSpinLock that wraps a SpinLock. This const
    /// function is evaluated at compile time, which is necessary to initialize
    /// static variables.
    /// 
    /// # Arguments
    /// 
    /// * `data` - Data to be protected by the inner SpinLock.
    #[inline]
    pub const fn new(data: T) -> Self {
        Self(SpinLock::new(data))
    }

    /// Disables hardware interrupts, then acquires the inner spinlock,
    /// spinning until it is available.
    /// 
    /// # Returns
    /// 
    /// Returns an IrqSpinLockGuard that, when dropped, releases the
    /// SpinLock and restores the interrupt-enable state to what it was
    /// before this call.
    #[inline]
    pub fn lock(&self) -> IrqSpinLockGuard<'_, T> {
        // Capture current interrupt state and disable interrupts atomically.
        // `flags` holds the x86 RFLAGS value before CLI; specifically bit 9
        // (IF) tells us whether interrupts were enabled.
        let flags = crate::system_routines::disable_interrupts_save();
        
        // Now that interrupts are off, acquire the SpinLock normally. This way,
        // no interrupt handler can sneak in and try to acquire the same
        // lock between the above two steps because we disabled interrupts
        // first.
        IrqSpinLockGuard {
            guard: self.0.lock(),
            flags,  // Save the old flags here so we can restore them on drop
        }
    }

    /// Tries to acquire the lock without spinning. Interrupts are disabled
    /// only briefly if the lock cannot be obtained.
    /// 
    /// If the lock is unavailable, interrupts are restored before returning
    /// `None` so callers are not left in an interrupt-disabled state
    /// unexpectedly.
    /// 
    /// # Returns
    ///
    /// Returns `Some(guard)` if successful, or `None` if the lock was already
    /// held by something else.
    #[inline]
    pub fn try_lock(&self) -> Option<IrqSpinLockGuard<'_, T>> {
        // Disable interrupts first; we must not be preempted by an ISR between
        // checking the lock and acquiring it.
        let flags = crate::system_routines::disable_interrupts_save();
        
        match self.0.try_lock() {
            Ok(guard) => Some(IrqSpinLockGuard { guard, flags }),
            Err(_) => {
                // Lock is busy. Restore interrupt state before returning so
                // the caller is not stuck with interrupts disabled.
                crate::system_routines::restore_interrupts(flags);
                None
            }
        }
    }
}

// =============================================================================
// IrqSpinLockGuard<'a, T>
// =============================================================================

/// RAII guard returned by `IrqSpinLock::lock()` or `IrqSpinLock::try_lock()`.
///
/// While this guard is alive:
///   - The underlying SpinLock is held, and no other thread can access the data
///   - Hardware interrupts are disabled on the current CPU
///
/// When dropped, it releases the SpinLock and restores the interrupt-enable
/// state to whatever it was before `lock()` was called.
pub struct IrqSpinLockGuard<'a, T> {
    // The underlying SpinLock guard. Dropping this releases the AtomicBool.
    guard: SpinLockGuard<'a, T>,

    // The saved x86 RFLAGS value captured immediately before interrupts were
    // disabled. Passed to `restore_interrupts()` on drop.
    flags: usize,
}

/// Transparent read access to the protected `T` through the guard.
impl<T> Deref for IrqSpinLockGuard<'_, T> {
    type Target = T;
    fn deref(&self) -> &T {
        // Delegates to SpinLockGuard's Deref, which performs the unsafe
        // dereference.
        &self.guard
    }
}

/// Transparent write access to the protected `T` through the guard.
impl<T> DerefMut for IrqSpinLockGuard<'_, T> {
    fn deref_mut(&mut self) -> &mut T {
        // Delegates to SpinLockGuard's DerefMut.
        &mut self.guard
    }
}

/// Releases the SpinLock and restores interrupt state when the guard is
/// dropped.
///
/// Drop order matters here: `self.guard` is dropped implicitly by Rust after
/// this `drop` body runs (fields are dropped in declaration order), which
/// stores `false` to `locked` with `Release` ordering. Then, we call
/// `restore_interrupts` to re-enable interrupts if they were enabled before we
/// took the lock.
///
/// Interrupts are restored after the SpinLock is released (because the guard
/// field is dropped first due to declaration order), which is the safe
/// ordering. We don't want an interrupt to fire and try to acquire the same
/// lock before we've actually released it.
impl<T> Drop for IrqSpinLockGuard<'_, T> {
    fn drop(&mut self) {
        // Explicitly drop the spinlock guard before re-enabling interrupts.
        // If we restore interrupts first, a timer IRQ can fire and attempt
        // to acquire the same spinlock before we've fully released it, causing
        // a deadlock on a single-core system.
        unsafe { core::ptr::drop_in_place(&mut self.guard) };

        // With the guard dropped, it is now safe to restore interrupts
        crate::system_routines::restore_interrupts(self.flags);
    }
}

// =============================================================================
// StaticIrqSpinLock<T>
// =============================================================================

/// A wrapper around IrqSpinLock for static variables.
///
/// Similar to StaticSpinLock, but for the interrupt-safe variant. Implements
/// `Sync` so it can live in a static.
pub struct StaticIrqSpinLock<T>(IrqSpinLock<T>);

// `Sync` is required for static. This is safe because IrqSpinLock is
// internally sound.
unsafe impl<T: Send> Sync for StaticIrqSpinLock<T> {}

#[allow(dead_code)]
impl<T> StaticIrqSpinLock<T> {
    /// Creates a new unlocked StaticIrqSpinLock that wraps an IrqSpinLock.
    /// This const function is evaluated at compile time, which is necessary to
    /// initialize static variables.
    /// 
    /// # Arguments
    /// 
    /// * `data` - Data to be protected by the inner IrqSpinLock.
    #[inline]
    pub const fn new(data: T) -> Self {
        Self(IrqSpinLock::new(data))
    }

    /// Disables interrupts and acquires the lock, spinning until available.
    #[inline]
    pub fn lock(&self) -> IrqSpinLockGuard<'_, T> {
        self.0.lock()
    }

    /// Non-blocking attempt to acquire the lock.
    /// 
    /// # Returns
    ///
    /// Returns guard if successful, or `None` if the lock was already held by
    /// something else (and interrupts are restored before returning in that
    /// case).
    #[inline]
    pub fn try_lock(&self) -> Option<IrqSpinLockGuard<'_, T>> {
        self.0.try_lock()
    }
}
