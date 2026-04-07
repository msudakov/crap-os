//! System Tasks Module
//!
//! This module contains concrete `SystemTask` functions executed by the
//! [`super::system_task_queue`] machinery. Each function in this module is a
//! discrete unit of deferred kernel work that is short, non-blocking, and
//! designed to run at elevated priority between hardware IRQ handling and
//! normal task scheduling.

/// This `SystemTask` is enqueued by `task_exit()` whenever a task terminates,
/// and fires on the next timer tick.
/// 
/// It calls into the scheduler to perform tombstone cleanup: sweeping the task
/// table for Dead tasks and freeing their resources.
/// 
/// # Arguments
/// 
/// * `arg` - This parameter is unused and exists for signature consistency.
pub fn dead_task_reaper(_arg: u64) {
    crate::task_scheduler::tombstone_cleanup();
}
