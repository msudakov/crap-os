// =============================================================================
// System Core Module
// =============================================================================
// 
// The System Core module is the kernel's infrastructure layer. It sits above
// raw Hardware Manager and above the Task Scheduler, coordinating between them.
// It owns the `SystemTaskQueue` and anything else that is kernel policy, but
// not a hardware mechanism.


pub mod system_task_queue;
pub mod system_tasks;

// Re-export public APIs
pub use system_task_queue::{queue_system_task, drain_system_tasks};
