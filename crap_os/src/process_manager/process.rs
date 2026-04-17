use alloc::sync::Arc;
use alloc::vec::Vec;
use core::sync::atomic::{AtomicU64, Ordering};
use crate::spinlock::IrqSpinLock;
use super::thread::{Thread, ThreadId};
use crate::task_scheduler::{queue_task, SchedulerError};
use crate::task_scheduler::task::Task;

// ————————————————————————————————————————————————————————————————————
// ProcessId
// ————————————————————————————————————————————————————————————————————

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct ProcessId(u64);

impl ProcessId {
    pub const IDLE: ProcessId = ProcessId(0);

    #[inline]
    pub fn next() -> Self {
        static COUNTER: AtomicU64 = AtomicU64::new(1);
        Self(COUNTER.fetch_add(1, Ordering::Relaxed))
    }

    #[inline]
    pub fn as_u64(self) -> u64 {
        self.0
    }
}

// ————————————————————————————————————————————————————————————————————
// Process
// ————————————————————————————————————————————————————————————————————

pub struct Process {
    pub id: ProcessId,
    pub name: &'static str,
    pub cr3: u64,

    // IrqSpinLock: the scheduler (IRQ context) may walk this list
    // to remove a dying thread. The inner SpinLock<Thread> protects
    // individual thread state during scheduling.
    pub threads: IrqSpinLock<Vec<Arc<IrqSpinLock<Thread>>>>,
}

impl Process {
    pub(crate) fn new(name: &'static str, cr3: u64) -> Arc<Self> {
        Arc::new(Process {
            id: ProcessId::next(),
            name,
            cr3,
            threads: IrqSpinLock::new(Vec::new()),
        })
    }

    pub(crate) fn new_idle(cr3: u64) -> Arc<Self> {
        Arc::new(Process {
            id: ProcessId::IDLE,
            name: "Idle",
            cr3,
            threads: IrqSpinLock::new(Vec::new()),
        })
    }

    //pub fn add_idle_thread()
    

    /// Spawn a new kernel thread in this process and return a reference to it.
    /// The returned Arc is also stored in self.threads — the process owns it,
    /// the caller gets a shared view.
    pub fn spawn_thread(
        self: &Arc<Self>,
        name: &'static str,
        entry: fn(u64),
        arg: u64,
    ) -> Result<Arc<IrqSpinLock<Thread>>, SchedulerError> {
        let thread = Thread::new(name, Arc::downgrade(self));
        let task = Task::new(entry, arg, Arc::downgrade(&thread));
        let task_id = task.id;
        
        // Make the new task immediately eligible for scheduling, but the task
        // will not actually begin executing until the timer ISR next calls
        // `schedule()` and selects it from the head of the ready queue.
        queue_task(task)?;
        
        thread.lock().task_id = Some(task_id);
        self.threads.lock().push(Arc::clone(&thread));

        Ok(thread)
    }

    /// Remove a thread by ID. Called by the reaper after a thread has died.
    pub fn remove_thread(&self, id: ThreadId) {
        self.threads
            .lock()
            .retain(|t| t.lock().id != id);
    }
}
