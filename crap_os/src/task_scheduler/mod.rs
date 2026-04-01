// =============================================================================
// Task Scheduler Module
// =============================================================================
// 
// This module is responsible for all scheduling and tracking of system tasks.
// A system task is the fundamental, schedulable unit of execution recognized
// by the kernel, representing a context (registers, stack, address space) that
// the scheduler manages and switches. A task is what the scheduler actually
// selects to run; it is the executable component of a thread.

mod task;
mod switcher;
pub mod scheduler;

// Re-export public APIs
pub use scheduler::{init, schedule, spawn, wake, yield_blocked,
    get_current_task_id};
pub use task::TaskId;
