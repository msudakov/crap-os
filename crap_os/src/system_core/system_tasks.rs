//! System Tasks Module
//!
//! This module contains concrete `SystemTask` functions executed by the
//! [`super::system_task_queue`] machinery. Each function in this module is a
//! discrete unit of deferred kernel work that is short, non-blocking, and
//! designed to run at elevated priority between hardware IRQ handling and
//! normal task scheduling.

/// This `SystemTask` is enqueued by an exiting thread whenever its task is
/// targeted for forceful termination, and fires on the next timer tick.
/// 
/// # Arguments
/// 
/// * `task_id_u64` - Compressed `TaskId` value of the target task as a single u64.
pub fn task_killer(task_id_u64: u64) {
    crate::task_scheduler::scheduler::kill_task(task_id_u64);
}
