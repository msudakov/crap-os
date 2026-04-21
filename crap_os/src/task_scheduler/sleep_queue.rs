//! Sleep Queue Module
//!
//! This module implements a delta queue for managing sleeping tasks. When a
//! task calls [`scheduler::sleep`], it is inserted here with a tick countdown,
//! marked [`TaskState::Blocked`], and removed from the run queue. The APIC
//! timer ISR drives this queue indirectly via [`scheduler::on_timer_tick`],
//! which calls [`tick_sleep_queue`] on every system clock tick.
//!
//! The sleep timers are stored using delta queue encoding. Rather than storing
//! the absolute wakeup tick for each entry, each entry stores only the
//! difference in ticks from the entry before it. E.g., if three tasks sleep for
//! 10ms, 15ms, and 20ms respectively, the queue stores:
//!
//! [ {delta: 10, task: A}, {delta: 5, task: B}, {delta: 5, task: C} ]
//!
//! The absolute wakeup time for any entry is the sum of all deltas from the
//! head up to and including that entry. This encoding means the timer ISR only
//! ever needs to decrement the head entry on each tick, keeping the hot timer
//! path O(1), regardless of how many tasks are sleeping.
//!
//! Insertion is O(n) in the number of sleeping tasks, but this only happens in
//! thread context (inside [`scheduler::sleep`]), not in the ISR, so the cost
//! is acceptable.
//!
//! The queue is protected by a [`StaticIrqSpinLock`], which disables IRQs on
//! acquisition. This is necessary because the queue is accessed from both the
//! thread and the ISR contexts that would otherwise race.

use crate::spinlock::StaticIrqSpinLock;
use crate::task_scheduler::task::TaskId;
use crate::task_scheduler::scheduler;

/// Maximum number of tasks that can be simultaneously sleeping. If this limit
/// is reached, [`SleepQueue::insert`] will print a message over the serial port
/// and just return. But 256 matches the scheduler's `MAX_TASKS` constant for
/// now, so a full task table can never overflow the sleep queue.
const MAX_SLEEP_ENTRIES: usize = 256;

/// A single entry in the sleep delta queue.
struct SleepEntry {
    /// Ticks remaining relative to the previous entry in the queue (delta
    /// encoding). The absolute wakeup time for entry[i] is the sum of
    /// `delta_ticks` for all entries from index 0 through i inclusive.
    ///
    /// On each timer tick, only the head entry's `delta_ticks` is decremented.
    /// All other entries are implicitly accounted for by the delta chain, and
    /// when the head expires and is popped, the next entry's countdown begins
    /// naturally, since it already stores only the remaining time after its
    /// predecessor would have expired.
    delta_ticks: u64,

    /// The ID of the task to wake when this entry's delta reaches zero.
    task_id: TaskId,
}

/// The delta queue data structure that holds all currently sleeping tasks.
/// Entries are stored in ascending order of absolute wakeup time, maintained
/// by [`SleepQueue::insert`].
pub struct SleepQueue {
    /// Fixed-size flat array of optional [`SleepEntry`] slots. Active entries
    /// occupy indices `[0..queue_len)` contiguously; all slots from `queue_len`
    /// onward are `None`. This avoids heap allocation and keeps the structure
    /// usable in a static `const` context.
    entries: [Option<SleepEntry>; MAX_SLEEP_ENTRIES],

    /// Number of active sleep entries currently stored in the queue.
    /// Always satisfies `queue_len <= MAX_SLEEP_ENTRIES`.
    queue_len: usize,
}

impl SleepQueue {
    /// Creates an empty [`SleepQueue`] at compile time, suitable for use as a
    /// static initializer. All entry slots are initialized to `None` and
    /// `queue_len` is set to 0.
    const fn new() -> Self {
        Self {
            entries: [const { None }; MAX_SLEEP_ENTRIES],
            queue_len: 0,
        }
    }

    /// Inserts a new sleep entry for `task_id` that should wake after `ticks`
    /// milliseconds, maintaining the delta-queue invariant throughout.
    ///
    /// The queue is walked from head to tail, subtracting each entry's
    /// `delta_ticks` from `ticks` as we go, until we find the correct
    /// insertion position. At that point, `ticks` holds the delta relative to
    /// the predecessor (or the absolute tick count if inserting at the head).
    /// The successor entry's delta is reduced by the new entry's delta to
    /// preserve its absolute wakeup time.
    /// 
    /// If the queue is full (i.e., already contains [`MAX_SLEEP_ENTRIES`]
    /// entries), this is also a no-op.
    ///
    /// # Arguments
    ///
    /// * `task_id` - The [`TaskId`] of the task to create a new sleep entry for.
    /// * `ticks`   - The number of milliseconds the task should sleep. A value
    ///               of 0 is a no-op; the caller should simply yield instead.
    pub fn insert(&mut self, task_id: TaskId, ticks: u64) {
        if ticks == 0 {
            // Zero-duration sleep: no entry needed. The caller is expected to
            // handle this case before calling insert(), but we guard here for
            // safety.
            return;
        }

        // TODO: Implement better error reporting when a queue is full
        if self.queue_len == MAX_SLEEP_ENTRIES {
            crate::hardware_manager::serial::print(
                "[SleepQueue] Failed to insert task sleep entry: queue full\n");
            return;
        }

        // `remaining` tracks how many ticks are left to account for as we walk
        // the existing entries. It starts as the full requested sleep duration
        // and is reduced by each entry's delta as we pass it, converging on the
        // correct delta for the new entry at its insertion point.
        let mut remaining = ticks;
        
        // Default to appending at the tail if no earlier insertion point is
        // found (i.e., this entry expires after all existing entries).
        let mut insert_pos = self.queue_len;

        for i in 0..self.queue_len {
            let entry = self.entries[i].as_ref().unwrap();
            if remaining <= entry.delta_ticks {
                // The new entry expires before `entry[i]`, so we insert here.
                // `remaining` is now the correct delta relative to `entry[i-1]`
                // (or the absolute tick count if `i == 0`).
                insert_pos = i;
                break;
            }

            // The new entry outlasts `entry[i]`, so we subtract its delta and
            // keep walking.
            remaining -= entry.delta_ticks;
        }

        // Shift all entries from `insert_pos` to the tail 1 slot to the
        // right to make room for the new entry.
        let mut i = self.queue_len;
        while i > insert_pos {
            self.entries[i] = self.entries[i - 1].take();
            i -= 1;
        }

        // Write the new entry at the insertion point with its computed delta
        self.entries[insert_pos] = Some(SleepEntry {
            delta_ticks: remaining,
            task_id,
        });
        self.queue_len += 1;

        // Fix up the successor entry's delta. Because we are inserting before
        // it, its delta (which was previously relative to entry[insert_pos-1])
        // must now be relative to the new entry, instead. Subtracting the new
        // entry's delta preserves its absolute wakeup time.
        if insert_pos + 1 < self.queue_len {
            if let Some(next) = self.entries[insert_pos + 1].as_mut() {
                next.delta_ticks -= remaining;
            }
        }
    }


    /// Advances the sleep queue by one timer tick and wakes all tasks whose
    /// sleep duration has expired.
    ///
    /// This is called every system clock tick by [`tick_sleep_queue`] from
    /// [`scheduler::on_timer_tick`]. Only the head entry's `delta_ticks` is
    /// decremented, as all other entries are implicitly advanced by the delta
    /// chain. Once the head reaches zero, it is popped, and [`scheduler::wake`]
    /// is called for its task, transitioning it from [`TaskState::Blocked`] to
    /// [`TaskState::Ready`] and re-enqueueing it in the run queue. This
    /// continues in a loop since multiple tasks may share the same absolute
    /// wakeup tick (i.e., the next entry also has `delta_ticks == 0` after the
    /// head is popped).
    ///
    /// # Returns
    ///
    /// Returns the number of tasks woken this tick. Usually 0 or 1; can be
    /// greater if multiple tasks were inserted with the same absolute wakeup
    /// time.
    fn tick(&mut self) -> usize {
        if self.queue_len == 0 {
            // Fast path: nothing sleeping, nothing to do.
            return 0;
        }

        // Decrement only the head entry's delta. All other deltas are
        // implicitly correct by the delta-chain invariant and do not need
        // to be touched.
        if let Some(head) = self.entries[0].as_mut() {
            if head.delta_ticks > 0 {
                head.delta_ticks -= 1;
            }
        }

        // Pop and wake all head entries whose delta has reached zero. Multiple
        // tasks can expire on the same tick if they were inserted with the same
        // absolute wakeup time, resulting in adjacent entries both having
        // `delta_ticks == 0`.
        let mut woken = 0;
        while self.queue_len > 0 {
            let delta = self.entries[0]
                        .as_ref()
                        .map(|entry| entry.delta_ticks)
                        .unwrap_or(1);
            
            // A non-zero head delta means no more tasks are due this tick.
            // Because of the delta-chain invariant, all subsequent entries
            // also have a non-zero absolute remaining time, so we can stop.
            if delta != 0 {
                break;
            }

            // Pop the expired head entry
            let entry = self.entries[0].take().unwrap();

            // Shift all remaining entries one slot left to close the gap
            for i in 0..self.queue_len - 1 {
                self.entries[i] = self.entries[i + 1].take();
            }
            self.queue_len -= 1;  // Decrement length after popping head entry

            // Transition the task from Blocked to Ready and re-enqueue it in
            // the scheduler's run queue, so it will be picked up in the next
            // scheduling pass. The task's thread state is also restored to
            // Active inside `wake()`.
            let _ = scheduler::wake(entry.task_id);
            woken += 1;
        }

        woken
    }
}

/// The global sleep queue instance, protected by an [`StaticIrqSpinLock`].
pub static SLEEP_QUEUE: StaticIrqSpinLock<SleepQueue> =
    StaticIrqSpinLock::new(SleepQueue::new());

/// Advances the global sleep queue by one tick and wakes any tasks whose sleep
/// duration has expired.
///
/// Called unconditionally by [`scheduler::on_timer_tick`] on every APIC timer
/// interrupt, before quantum accounting. Woken tasks are immediately
/// transitioned to [`TaskState::Ready`] and re-enqueued, making them eligible
/// for scheduling in the same tick that woke them.
///
/// # Returns
///
/// Returns the number of tasks woken this tick.
pub fn tick_sleep_queue() -> usize {
    SLEEP_QUEUE.lock().tick()
}
