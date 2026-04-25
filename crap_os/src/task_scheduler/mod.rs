//! Task Scheduler Module
//! 
//! This module is responsible for all scheduling and tracking of tasks.
//! A task is the fundamental, schedulable unit of execution recognized
//! by the kernel, representing a context (registers, stack, address space) that
//! the scheduler manages and switches. A task is what the scheduler actually
//! selects to run; it is the executable component of a thread.

mod switcher;
pub mod task;
pub mod scheduler;
pub mod reaper;
pub(super) mod sleep_queue;

// Re-export public APIs
pub use scheduler::{SchedulerError, init_idle, on_timer_tick, schedule, wake,
    insert_and_queue_task, yield_blocked, get_current_task_id, sleep,
    kill_current_task};
pub use task::TaskId;
pub use reaper::{queue_task_reaper, reap_dying_tasks, tombstone_cleanup};
