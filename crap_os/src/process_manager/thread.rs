use alloc::sync::{Arc, Weak};
use core::sync::atomic::{AtomicU64, Ordering};
use crate::spinlock::IrqSpinLock;
use super::process::Process;
use crate::task_scheduler::TaskId;

// ————————————————————————————————————————————————————————————————————
// ThreadId
// ————————————————————————————————————————————————————————————————————

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct ThreadId(u64);

impl ThreadId {
    pub const IDLE: ThreadId = ThreadId(0);

    #[inline]
    pub fn next() -> Self {
        static COUNTER: AtomicU64 = AtomicU64::new(1);
        Self(COUNTER.fetch_add(1, Ordering::Relaxed))
    }

    /// Unwraps the `u64` value. Mainly used for debug output and logging.
    /// 
    /// # Returns
    /// 
    /// Returns the underlying `u64` value.
    #[inline]
    pub fn as_u64(self) -> u64 {
        self.0
    }
}

// ————————————————————————————————————————————————————————————————————
// ThreadState
// ————————————————————————————————————————————————————————————————————

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum ThreadState {
    Active,    // alive and schedulable (Ready or Running lives in Task)
    Waiting,   // voluntarily blocked via yield_blocked, awaiting wake
    Dying,     // has called thread_exit, awaiting reaper
    Dead,      // reaper has finished, safe to drop
}

// ————————————————————————————————————————————————————————————————————
// Thread
// ————————————————————————————————————————————————————————————————————

pub struct Thread {
    
    pub id: ThreadId,
    pub task_id: Option<TaskId>,
    pub name: &'static str,
    pub state: ThreadState,

    // Non-owning back-reference to the parent process.
    // Weak avoids a cycle: Process owns Thread, Thread does not own Process.
    pub process: Weak<Process>,
}

unsafe impl Send for Thread {}

impl Thread {
    /// Allocate a new kernel thread belonging to `process`.
    /// `entry` is a plain `fn(u64)` — the `arg` slot is available
    /// for the caller to pass context (e.g. a pointer to a config struct).
    /// For threads that need no argument, pass `0`.
    pub fn new(
        name: &'static str,
        //entry: fn(u64),
        //arg: u64,
        process: Weak<Process>,
    ) -> Arc<IrqSpinLock<Thread>> {
        let thread = Arc::new(IrqSpinLock::new(Thread {
            id: ThreadId::next(),
            task_id: None,
            name,
            state: ThreadState::Active,
            process,
        }));

        thread
    }

    pub fn new_idle(process: Weak<Process>) -> Arc<IrqSpinLock<Thread>> {
        let thread = Arc::new(IrqSpinLock::new(Thread {
            id: ThreadId::IDLE,
            task_id: None,
            name: "Idle",
            state: ThreadState::Active,
            process,
        }));
        thread
    }
}
