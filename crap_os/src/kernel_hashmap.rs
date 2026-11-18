//! Kernel Hash Map
//!
//! This module provides [`KernelHashMap`], a general-purpose, no-std
//! open-addressing hash map for use throughout the kernel, and
//! [`KernelHash`], the hashing trait that all key types must implement. The
//! implementation only requires the `alloc` crate for heap allocation of the
//! backing slot array.
//!
//! The implementation has the following design specifications:
//!
//! * Open addressing with linear probing: all entries live in a single
//!   contiguous heap-allocated array. Collisions are resolved by scanning
//!   forward from the home slot. This is cache-friendly and avoids the
//!   per-entry heap allocation overhead of chaining.
//!
//! * FNV-1a hash function is a simple, fast, non-cryptographic hash with no
//!   external dependencies, with one XOR and one multiply per byte. Suitable
//!   for kernel key types (integers, short strings) where hash-flooding
//!   resistance is not required.
//!
//! * Tombstone deletion: removed entries are marked as `Tombstone` rather than
//!   `Empty`, so that probe chains formed during insertion remain intact after
//!   removal. A subsequent lookup for a key that probed through the now-removed
//!   slot will still find its target.
//!
//! * Load factor of 0.7: the table resizes (doubles capacity and rehashes)
//!   when the number of occupied-plus-tombstone slots exceeds 70% of total
//!   capacity. Tombstones count toward the load factor because they consume
//!   probe-chain slots.
//!
//! * Power-of-two capacity: capacity is always a power of two, so the home
//!   slot for any hash value is computed with a single bitmask instead of a
//!   modulo division.
//!
//! * Minimum capacity of 8: the first insertion triggers an allocation
//!   of 8 slots. An empty map holds no heap allocation at all.
//!
//! Key types must implement [`KernelHash`] and [`Eq`]. Built-in
//! implementations are provided for [`u32`], [`u64`], and [`&'static str`].
//! Composite key types (e.g.,
//! [`crate::file_system::virtual_fs::types::OpenFileKey`]) must implement
//! [`KernelHash`] manually; the recommended approach is to chain FNV-1a over
//! each field's bytes in sequence so that field order contributes to the hash
//! and `(a, b)` and `(b, a)` produce different values.

#![allow(dead_code)]

extern crate alloc;

use alloc::boxed::Box;
use alloc::vec::Vec;
use core::mem;

/// A non-cryptographic hash trait for kernel key types.
///
/// All key types used with [`KernelHashMap`] must implement this trait.
/// Implementations should be cheap and deterministic: the same value must
/// always produce the same hash within a single boot session. Hash values
/// need not be stable across reboots or kernel versions.
///
/// Built-in implementations are provided for [`u32`], [`u64`], and
/// [`&'static str`], all using the FNV-1a algorithm. Composite key structs
/// should chain FNV-1a over each field's bytes in declaration order; see
/// [`crate::file_system::virtual_fs::types::OpenFileKey`] for an example.
pub trait KernelHash {
    /// Computes a 64-bit hash of this value.
    ///
    /// # Returns
    ///
    /// Returns a `u64` hash. The value is used only as a table index; the
    /// [`KernelHashMap`] masks it to the current capacity internally.
    fn kernel_hash(&self) -> u64;
}

// FNV-1a 64-bit constants. The offset basis and prime are specified by the
// FNV standard and must not be changed.
const FNV_OFFSET_BASIS: u64 = 0xcbf29ce484222325;
const FNV_PRIME: u64 = 0x00000100000001b3;

/// Computes the FNV-1a 64-bit hash of a byte slice.
///
/// Processes each byte with one XOR and one wrapping multiply, starting from
/// [`FNV_OFFSET_BASIS`]. The result is sensitive to both byte values and byte
/// order, which makes it suitable for hashing multi-field composite keys when
/// fields are fed in sequence.
///
/// # Arguments
///
/// * `bytes` - The raw bytes to hash.
///
/// # Returns
///
/// Returns the FNV-1a 64-bit hash of `bytes`.
#[inline]
fn fnv1a_hash_bytes(bytes: &[u8]) -> u64 {
    let mut hash = FNV_OFFSET_BASIS;

    for &byte in bytes {
        hash ^= byte as u64;
        hash  = hash.wrapping_mul(FNV_PRIME);
    }

    hash
}

impl KernelHash for u32 {
    /// Returns the FNV-1a hash of this `u32` value's native-endian bytes.
    ///
    /// # Returns
    ///
    /// Returns the FNV-1a 64-bit hash of the four bytes of `self`.
    #[inline]
    fn kernel_hash(&self) -> u64 {
        fnv1a_hash_bytes(&self.to_ne_bytes())
    }
}

impl KernelHash for u64 {
    /// Returns the FNV-1a hash of this `u64` value's native-endian bytes.
    ///
    /// # Returns
    ///
    /// Returns the FNV-1a 64-bit hash of the eight bytes of `self`.
    #[inline]
    fn kernel_hash(&self) -> u64 {
        fnv1a_hash_bytes(&self.to_ne_bytes())
    }
}

impl KernelHash for &'static str {
    /// Returns the FNV-1a hash of this string's UTF-8 bytes.
    ///
    /// # Returns
    ///
    /// Returns the FNV-1a 64-bit hash of the UTF-8 byte representation of
    /// `self`.
    #[inline]
    fn kernel_hash(&self) -> u64 {
        fnv1a_hash_bytes(self.as_bytes())
    }
}

/// A single slot in the [`KernelHashMap`]'s backing array.
///
/// Each slot is in one of three states:
///
/// - `Empty` - never been written to. A probe chain terminates here: if a
///   lookup reaches an `Empty` slot, the key is absent.
/// - `Tombstone` - previously held a key-value pair that was removed. Probe
///   chains must not terminate at a `Tombstone`; lookups continue past it so
///   that keys inserted after the removed entry (and that landed further along
///   the same probe chain) remain reachable.
/// - `Occupied(K, V)` - holds a live key-value pair.
enum Slot<K, V> {
    Empty,
    Tombstone,
    Occupied(K, V),
}

impl<K, V> Slot<K, V> {
    /// Checks if this slot is `Empty`.
    ///
    /// Used during rehashing to identify free slots in the new backing array.
    ///
    /// # Returns
    ///
    /// Returns `true` for `Empty`, and `false` for `Tombstone` or `Occupied`.
    fn is_empty(&self) -> bool {
        matches!(self, Slot::Empty)
    }
}

/// Numerator of the maximum load factor fraction (7/10 = 0.70).
///
/// The table resizes when `(occupied + tombstone) * LOAD_FACTOR_DEN >
/// capacity * LOAD_FACTOR_NUM`. Tombstones count because they occupy probe
/// chain slots and degrade lookup performance just as occupied slots do.
const LOAD_FACTOR_NUM: usize = 7;

/// Denominator of the maximum load factor fraction (7/10 = 0.70).
const LOAD_FACTOR_DEN: usize = 10;

/// The minimum backing array capacity, and the capacity of the first
/// allocation. Must be a power of two.
const MIN_CAPACITY: usize = 8;

/// A general-purpose, no-std open-addressing hash map for kernel use.
///
/// [`KernelHashMap<K, V>`] maps keys of type `K` to values of type `V`.
/// Keys must implement [`KernelHash`] and [`Eq`]; values may be any type.
///
/// The backing storage is a heap-allocated boxed slice of [`Slot`] entries.
/// An empty map holds no heap allocation; the first insertion allocates
/// [`MIN_CAPACITY`] slots.
pub struct KernelHashMap<K, V> {
    /// The backing slot array. Length is always a power of two, or zero for
    /// a freshly constructed empty map.
    slots: Box<[Slot<K, V>]>,

    /// The number of `Occupied` slots. This is the map's logical length as
    /// seen by callers.
    len: usize,

    /// The number of `Occupied` plus `Tombstone` slots. Used to compute the
    /// effective load factor, since tombstones consume probe-chain capacity
    /// even though they hold no live data.
    used: usize,
}

impl<K: KernelHash + Eq, V> KernelHashMap<K, V> {
    /// Maps a hash value to a starting slot index for a table of `cap` slots.
    ///
    /// `cap` must be a power of two. The home index is the low-order bits of
    /// `hash`, masked to `cap - 1`. This is equivalent to `hash % cap` but
    /// requires only a bitwise AND.
    ///
    /// # Arguments
    ///
    /// * `hash` - The hash value to map.
    /// * `cap`  - The current backing array length. Must be a power of two
    ///   and greater than zero.
    ///
    /// # Returns
    ///
    /// Returns the home slot index for `hash` in a table of `cap` slots.
    #[inline]
    fn slot_index(hash: u64, cap: usize) -> usize {
        (hash as usize) & (cap - 1)
    }

    /// Probes the table for `key` using linear probing.
    ///
    /// Starts at the home slot for `key`'s hash and scans forward (wrapping
    /// at the end of the array) until either the key is found in an
    /// `Occupied` slot or an `Empty` slot is reached, which signals that the
    /// key is absent.
    ///
    /// `Tombstone` slots are skipped during the search but the index of the
    /// first tombstone encountered is remembered. If the key is absent, the
    /// first tombstone index is returned as the preferred insertion point,
    /// so that insertions reuse tombstone slots and keep the array compact.
    ///
    /// This method must only be called when the backing array is non-empty.
    ///
    /// # Arguments
    ///
    /// * `key` - The key to search for.
    ///
    /// # Returns
    ///
    /// Returns `(index, found)` where `index` is the slot to use for the
    /// result of this probe (either the `Occupied` slot holding `key` if
    /// found, or the best available insertion slot if not) and `found` is
    /// `true` if and only if `key` was present in the table.
    fn probe(&self, key: &K) -> (usize, bool) {
        let cap = self.slots.len();
        let hash = key.kernel_hash();
        let mut index = Self::slot_index(hash, cap);
        let mut first_tombstone: Option<usize> = None;

        loop {
            match &self.slots[index] {
                Slot::Empty => {
                    // Probe chain ends here; key is absent. Prefer the first
                    // tombstone slot for insertion if one was seen, so the
                    // new entry lands as close to its home slot as possible.
                    return (first_tombstone.unwrap_or(index), false);
                }

                Slot::Tombstone => {
                    if first_tombstone.is_none() {
                        first_tombstone = Some(index);
                    }
                }

                Slot::Occupied(k, _) => {
                    if k == key {
                        return (index, true);
                    }
                }
            }
            index = (index + 1) & (cap - 1);
        }
    }

    /// Rebuilds the backing array with a new capacity, discarding all
    /// tombstones.
    ///
    /// Allocates a fresh slot array of `new_cap` slots, moves every
    /// `Occupied` entry into it at its new home position (computed by linear
    /// probing with no tombstones present), then replaces `self.slots` with
    /// the new array. After rehashing, `used` equals `len` because all
    /// tombstones have been discarded.
    ///
    /// `new_cap` must be a power of two and must be large enough to hold all
    /// `self.len` live entries below the load factor threshold.
    ///
    /// # Arguments
    ///
    /// * `new_cap` - The new backing array length. Must be a power of two.
    fn rehash(&mut self, new_cap: usize) {
        // Allocate the new slot array filled entirely with Empty slots.
        let mut new_slots: Vec<Slot<K, V>> = Vec::with_capacity(new_cap);
        for _ in 0..new_cap {
            new_slots.push(Slot::Empty);
        }
        let mut new_slots = new_slots.into_boxed_slice();

        // Move every live entry from the old array into the new one. We
        // replace self.slots with an empty slice first so that we can take
        // ownership of the old slots via Vec::from without a clone.
        let old_slots = mem::replace(&mut self.slots, Box::new([]));
        for slot in Vec::from(old_slots).into_iter() {
            if let Slot::Occupied(k, v) = slot {
                let hash = k.kernel_hash();
                let mut index = Self::slot_index(hash, new_cap);

                // Linear probe in the new (tombstone-free) array to find an
                // empty slot. This always terminates because new_cap is large
                // enough to hold all entries below the load factor.
                loop {
                    if new_slots[index].is_empty() {
                        new_slots[index] = Slot::Occupied(k, v);
                        break;
                    }
                    index = (index + 1) & (new_cap - 1);
                }
            }
        }

        self.slots = new_slots;
        // Tombstones have been eliminated; used now equals len.
        self.used  = self.len;
    }

    /// Grows the backing array if the current load factor exceeds the
    /// threshold.
    ///
    /// Called before every insertion. If the table is empty (zero capacity),
    /// allocates [`MIN_CAPACITY`] slots. If the effective load (occupied plus
    /// tombstone slots) exceeds 70% of current capacity, doubles the capacity
    /// and rehashes.
    fn maybe_grow(&mut self) {
        let cap = self.slots.len();

        if cap == 0 || self.used * LOAD_FACTOR_DEN > cap * LOAD_FACTOR_NUM {
            let new_cap = if cap == 0 {
                MIN_CAPACITY
            } else {
                cap * 2
            };

            self.rehash(new_cap);
        }
    }
}

// =============================================================================
// Entry API
// =============================================================================

/// The result of a [`KernelHashMap::entry`] call.
///
/// Represents either an occupied slot (the key is already present) or a
/// vacant slot (the key is absent and ready for insertion). The primary use
/// is [`Entry::or_insert_with`], which inserts a lazily-computed default
/// value when the entry is vacant and returns a mutable reference to the
/// value in either case.
pub enum Entry<'a, K, V> {
    /// The key is present in the map.
    Occupied(OccupiedEntry<'a, K, V>),

    /// The key is absent from the map; this entry is ready for insertion.
    Vacant(VacantEntry<'a, K, V>),
}

impl<'a, K: KernelHash + Eq, V> Entry<'a, K, V> {
    /// Inserts a default value produced by `f` if the entry is vacant, then
    /// returns a mutable reference to the value.
    ///
    /// If the entry is already occupied, `f` is not called and the existing
    /// value is returned by mutable reference. If the entry is vacant, `f`
    /// is called once to produce a value, the value is inserted, and a
    /// mutable reference to the newly inserted value is returned.
    ///
    /// # Arguments
    ///
    /// * `f` - A closure that produces the default value. Called at most once,
    ///   and only when the entry is vacant.
    ///
    /// # Returns
    ///
    /// Returns a mutable reference to the value for this key, whether it was
    /// already present or just inserted.
    pub fn or_insert_with<F: FnOnce() -> V>(self, f: F) -> &'a mut V {
        match self {
            Entry::Occupied(e) => e.into_mut(),
            Entry::Vacant(e)   => e.insert(f()),
        }
    }
}

/// An [`Entry`] for a key that is already present in the map.
pub struct OccupiedEntry<'a, K, V> {
    slot: &'a mut Slot<K, V>,
}

impl<'a, K, V> OccupiedEntry<'a, K, V> {
    /// Converts this occupied entry into a mutable reference to its value.
    ///
    /// # Returns
    ///
    /// Returns a mutable reference to the value stored in this entry.
    pub fn into_mut(self) -> &'a mut V {
        if let Slot::Occupied(_, v) = self.slot {
            v
        } else {
            unreachable!()
        }
    }
}

/// An [`Entry`] for a key that is absent from the map.
pub struct VacantEntry<'a, K, V> {
    /// The key that will be inserted.
    key:  K,

    /// The slot in the backing array where the new entry will be placed.
    slot: &'a mut Slot<K, V>,

    /// Mutable reference to the map's `len` counter, updated on insertion.
    len:  &'a mut usize,

    /// Mutable reference to the map's `used` counter, updated on insertion
    /// if the slot was previously `Empty` (not `Tombstone`).
    used: &'a mut usize,
}

impl<'a, K, V> VacantEntry<'a, K, V> {
    /// Inserts `value` into this vacant entry and returns a mutable reference
    /// to the newly stored value.
    ///
    /// # Arguments
    ///
    /// * `value` - The value to store for this entry's key.
    ///
    /// # Returns
    ///
    /// Returns a mutable reference to the value that was just inserted.
    pub fn insert(self, value: V) -> &'a mut V {
        // If the slot was a Tombstone, `used` is already counted for it and
        // must not be incremented again.
        let was_tombstone = matches!(self.slot, Slot::Tombstone);
        *self.slot = Slot::Occupied(self.key, value);
        *self.len += 1;

        if !was_tombstone {
            *self.used += 1;
        }

        if let Slot::Occupied(_, v) = self.slot {
            v
        } else {
            unreachable!()
        }
    }
}

/// Default
impl<K: KernelHash + Eq, V> Default for KernelHashMap<K, V> {
    fn default() -> Self {
        Self::new()
    }
}

/// Utility function that finds the smallest power of two that is greater than
/// or equal to `n`.
///
/// Returns `1` for `n` of zero or one. Used by [`KernelHashMap::shrink`] to
/// compute a right-sized replacement capacity after bulk removal.
///
/// # Arguments
///
/// * `n` - The minimum value the result must meet or exceed.
///
/// # Returns
///
/// Returns the smallest power of two >= `n`.
fn next_power_of_two(n: usize) -> usize {
    if n <= 1 {
        return 1;
    }

    let mut p = 1usize;
    while p < n {
        p <<= 1;
    }

    p
}

// =============================================================================
// Public API
// =============================================================================

impl<K: KernelHash + Eq, V> KernelHashMap<K, V> {
    /// Constructs a new, empty [`KernelHashMap`].
    ///
    /// No heap allocation is performed until the first insertion.
    pub fn new() -> Self {
        Self {
            slots: Box::new([]),
            len:   0,
            used:  0,
        }
    }

    /// Gets the number of key-value pairs currently stored in the map.
    ///
    /// # Returns
    ///
    /// Returns the count of live entries. Tombstones left by
    /// [`KernelHashMap::remove`] are not counted.
    pub fn len(&self) -> usize {
        self.len
    }

    /// Checks if the map contains no key-value pairs.
    ///
    /// # Returns
    ///
    /// Returns `true` when [`KernelHashMap::len`] is zero, and `false`
    /// otherwise.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Inserts a key-value pair into the map.
    ///
    /// If the key was already present, the existing value is replaced and the
    /// previous value is returned. If the key was absent, the pair is inserted
    /// and `None` is returned.
    ///
    /// If the insertion causes the load factor to exceed the threshold, the
    /// backing array is grown and rehashed before the new entry is placed.
    ///
    /// # Arguments
    ///
    /// * `key`   - The key to insert or update.
    /// * `value` - The value to associate with `key`.
    ///
    /// # Returns
    ///
    /// Returns `Some(old_value)` if the key was already present and has been
    /// replaced, or `None` if the key was newly inserted.
    pub fn insert(&mut self, key: K, value: V) -> Option<V> {
        self.maybe_grow();
        let (index, found) = self.probe(&key);

        if found {
            // Key exists; replace value in place and return the old one.
            if let Slot::Occupied(_, v) = &mut self.slots[index] {
                return Some(mem::replace(v, value));
            }
        }

        // Inserting into an Empty or Tombstone slot. Tombstone -> Occupied
        // does not increase `used` because the tombstone already counted.
        let was_tombstone = matches!(self.slots[index], Slot::Tombstone);
        self.slots[index] = Slot::Occupied(key, value);
        self.len += 1;

        if !was_tombstone {
            self.used += 1;
        }

        None
    }

    /// Gets a shared reference to the value associated with `key`.
    ///
    /// # Arguments
    ///
    /// * `key` - The key to look up.
    ///
    /// # Returns
    ///
    /// Returns `Some(&value)` if `key` is present, or `None` if it is absent.
    pub fn get(&self, key: &K) -> Option<&V> {
        if self.slots.is_empty() {
            return None;
        }

        let (index, found) = self.probe(key);
        if found {
            if let Slot::Occupied(_, v) = &self.slots[index] {
                return Some(v);
            }
        }

        None
    }

    /// Gets a mutable reference to the value associated with `key`.
    ///
    /// # Arguments
    ///
    /// * `key` - The key to look up.
    ///
    /// # Returns
    ///
    /// Returns `Some(&mut value)` if `key` is present, or `None` if it is
    /// absent.
    pub fn get_mut(&mut self, key: &K) -> Option<&mut V> {
        if self.slots.is_empty() {
            return None;
        }

        let (index, found) = self.probe(key);
        if found {
            if let Slot::Occupied(_, v) = &mut self.slots[index] {
                return Some(v);
            }
        }

        None
    }

    /// Checks if the map contains an entry for `key`.
    ///
    /// # Arguments
    ///
    /// * `key` - The key whose presence to test.
    ///
    /// # Returns
    ///
    /// Returns `true` if `key` is present, and `false` otherwise.
    pub fn contains_key(&self, key: &K) -> bool {
        self.get(key).is_some()
    }

    /// Removes the entry for `key` and returns its value.
    ///
    /// The vacated slot is marked `Tombstone` rather than `Empty` so that
    /// probe chains passing through it remain intact. The tombstone counts
    /// toward the load factor and will be eliminated at the next rehash.
    ///
    /// # Arguments
    ///
    /// * `key` - The key to remove.
    ///
    /// # Returns
    ///
    /// Returns `Some(value)` if `key` was present and has been removed, or
    /// `None` if `key` was not found.
    pub fn remove(&mut self, key: &K) -> Option<V> {
        if self.slots.is_empty() {
            return None;
        }

        let (index, found) = self.probe(key);
        if !found {
            return None;
        }

        // Replace the `Occupied` slot with a `Tombstone`. `len` decrements but
        // `used` stays the same: the tombstone still occupies a probe-chain
        // slot and must be counted toward load.
        let old = mem::replace(&mut self.slots[index], Slot::Tombstone);
        self.len -= 1;

        if let Slot::Occupied(_, v) = old {
            Some(v)
        } else {
            unreachable!()
        }
    }

    /// Gets an iterator over all keys in the map.
    ///
    /// Iteration order is unspecified and may vary between insertions.
    ///
    /// # Returns
    ///
    /// Returns an iterator yielding shared references to each key.
    pub fn keys(&self) -> impl Iterator<Item = &K> {
        self.slots.iter().filter_map(|slot| {
            if let Slot::Occupied(k, _) = slot {
                Some(k)
            } else {
                None
            }
        })
    }

    /// Gets an iterator over all key-value pairs in the map.
    ///
    /// Iteration order is unspecified and may vary between insertions.
    ///
    /// # Returns
    ///
    /// Returns an iterator yielding `(&key, &value)` tuples for each live
    /// entry.
    pub fn iter(&self) -> impl Iterator<Item = (&K, &V)> {
        self.slots.iter().filter_map(|slot| {
            if let Slot::Occupied(k, v) = slot {
                Some((k, v))
            } else {
                None
            }
        })
    }

    /// Retains only the entries for which `predicate` returns `true`.
    ///
    /// Each entry for which `predicate` returns `false` is replaced with a
    /// `Tombstone`. If a large fraction of entries is removed, consider
    /// calling [`KernelHashMap::shrink`] afterward to reclaim memory and
    /// restore probe performance.
    ///
    /// # Arguments
    ///
    /// * `predicate` - A closure that receives `(&key, &mut value)` and
    ///   returns `true` to keep the entry or `false` to remove it.
    pub fn retain<F>(&mut self, mut predicate: F)
    where
        F: FnMut(&K, &mut V) -> bool,
    {
        for slot in self.slots.iter_mut() {
            if let Slot::Occupied(k, v) = slot {
                if !predicate(k, v) {
                    *slot = Slot::Tombstone;
                    self.len -= 1;
                    // `used` is unchanged: the tombstone still occupies the
                    // slot and counts toward probe-chain density.
                }
            }
        }
    }

    /// Rehashes the map into the smallest power-of-two capacity that keeps
    /// the current entries below the load factor threshold.
    ///
    /// Useful after a large [`KernelHashMap::retain`] call that removed many
    /// entries. Has no effect if the current capacity is already optimal.
    pub fn shrink(&mut self) {
        let new_cap = next_power_of_two(
            (self.len * LOAD_FACTOR_DEN / LOAD_FACTOR_NUM).max(MIN_CAPACITY),
        );

        if new_cap < self.slots.len() {
            self.rehash(new_cap);
        }
    }

    /// Resolves an [`Entry`] for the given key, allowing conditional insertion.
    ///
    /// If the key is already present, returns [`Entry::Occupied`]. If the key
    /// is absent, returns [`Entry::Vacant`]. The most common use is
    /// `.entry(key).or_insert_with(|| value)`, which inserts a default value
    /// only when the key is absent and returns a mutable reference to the
    /// (now-present) value in either case.
    ///
    /// Calling this method may trigger a grow-and-rehash if the table is near
    /// its load factor threshold, so it takes `&mut self` even in the
    /// occupied case.
    ///
    /// # Arguments
    ///
    /// * `key` - The key to look up or prepare for insertion.
    ///
    /// # Returns
    ///
    /// Returns [`Entry::Occupied`] if the key is present, or
    /// [`Entry::Vacant`] if it is absent.
    pub fn entry(&mut self, key: K) -> Entry<'_, K, V> {
        self.maybe_grow();
        let (index, found) = self.probe(&key);

        if found {
            Entry::Occupied(OccupiedEntry {
                slot: &mut self.slots[index],
            })
        } else {
            Entry::Vacant(VacantEntry {
                key,
                slot: &mut self.slots[index],
                len:  &mut self.len,
                used: &mut self.used,
            })
        }
    }
}
