//! ProcessManager - Public Interface
//! 
//! This module is the intended entry point for all process creation, and
//! callers should not construct [`Process`] instances directly. The overall
//! ownership model is reflected by the following schematic:
//!
//!   ProcessManager
//!         |
//!   (IrqSpinLock)
//!         |
//!       (Arc)
//!         V
//!      Process  <---(Weak)---  Thread
//!         |
//!   (IrqSpinLock)
//!         |
//!       (Arc)
//!         V
//!  IrqSpinLock<Thread>
//!
//! The [`ProcessManager`] holds the only strong `Arc` references to each
//! process at the manager level. Callers that receive an `Arc<Process>` from
//! [`create_process`] may hold their own strong references, but the manager's
//! entry in the process list is the canonical owner.
//!
//! The idle process is a special singleton initialized via
//! [`init_idle_process`] during early kernel startup, before the scheduler
//! begins running. It owns the idle thread and task, which represent the
//! initial kernel execution context. It is permanent and never removed from
//! the process list.

use alloc::sync::Arc;
use alloc::vec::Vec;
use super::process::Process;
use crate::spinlock::IrqSpinLock;
use crate::task_scheduler::{SchedulerError, TaskId};
use crate::sprintln;
use crate::fbprintln;

/// The top-level manager for all processes.
///
/// Holds the global list of live processes and provides the entry points for
/// process creation and lookup. There is one global instance of
/// [`ProcessManager`], initialized at kernel startup.
pub struct ProcessManager {
    /// The list of all live processes, including the idle process.
    ///
    /// Protected by an [`IrqSpinLock`] since process creation and lookup
    /// may occur concurrently from different task contexts. Each entry is
    /// an `Arc<Process>`; the manager holds strong ownership, while threads
    /// within each process hold `Weak` back-references.
    processes: IrqSpinLock<Vec<Arc<Process>>>,
}

#[allow(dead_code)]
impl ProcessManager {
    /// Creates a new, empty [`ProcessManager`].
    ///
    /// This is a `const fn` to allow static initialization. The process list
    /// starts empty, and [`init_idle_process`] must be called during kernel
    /// startup before any scheduling begins, followed by [`create_process`]
    /// for any system processes that should run at boot.
    pub const fn new() -> Self {
        ProcessManager {
            processes: IrqSpinLock::new(Vec::new()),
        }
    }

    /// Initializes the idle process and registers it in the process list.
    ///
    /// Creates the idle process and its idle thread, assigns [`TaskId::IDLE`]
    /// to the thread, and initializes the scheduler's idle task via
    /// [`crate::task_scheduler::init_idle`]. Must be called exactly once
    /// during early kernel startup, before the scheduler begins dispatching.
    /// 
    /// # Arguments
    /// 
    /// * `cr3` - PML4 page table root physical address.
    ///
    /// # Returns
    ///
    /// Returns a strong `Arc` reference to the idle process, which the system
    /// may hold for the lifetime of the kernel.
    pub fn init_idle_process(&self, cr3: u64) -> Arc<Process> {
        // Create idle process and thread objects
        let idle_process = Process::new_idle(cr3);
        let idle_thread = super::Thread::new_idle(
            Arc::downgrade(&idle_process));
        
        // Create idle task reference in the thread through `task_id`
        idle_thread.lock().task_id = Some(TaskId::IDLE);

        // Register idle thread in the idle process's thread list
        idle_process.threads.lock().push(Arc::clone(&idle_thread));

        // Register idle process with the process manager
        self.processes.lock().push(Arc::clone(&idle_process));

        // Initialize the idle task with the task scheduler
        crate::task_scheduler::init_idle(Arc::downgrade(&idle_thread));

        idle_process
    }

    /// Creates a new kernel process with a single main thread and registers it
    /// in the process list, and the main thread is immediately queued in the
    /// scheduler and eligible for scheduling as soon as this function returns.
    ///
    /// # Arguments
    /// 
    /// * `name`       - Human-readable name for this process.
    /// * `cr3`        - PML4 page table root physical address.
    /// * `main_entry` - The main thread's entry function to call.
    /// * `main_arg`   - Single u64 argument to pass to the main thread's entry
    ///                  function. Pass `0` if not needed.
    /// 
    /// # Returns
    ///
    /// Returns a strong `Arc` reference to the new process on success, or
    /// [`SchedulerError`] if the scheduler task table or run queue
    /// is full.
    pub fn create_kernel_process(
        &self,
        name: &'static str,
        cr3: u64,
        main_entry: fn(u64),
        main_arg: u64,
    ) -> Result<Arc<Process>, SchedulerError> {
        // Create the new process instance and try to spawn the main thread
        let process = Process::new_kernel(name, cr3);
        process.spawn_kernel_thread("main", main_entry, main_arg)?;
        
        // Only register the new process if the thread was spawned successfully
        self.processes.lock().push(Arc::clone(&process));

        Ok(process)
    }

    /// Creates a new user process with a fresh isolated address space and a
    /// single main thread, registers it in the process list, and queues the
    /// main thread in the scheduler immediately.
    ///
    /// Unlike `create_kernel_process`, this allocates a brand-new PML4 for
    /// the process and copies the kernel's upper-half entries into it. The
    /// caller does not need to set up any mappings, as the main thread's code
    /// page and user stack are mapped internally by `spawn_user_thread`.
    ///
    /// # Arguments
    ///
    /// * `name`             - Human-readable name for this process.
    /// * `kernel_pml4_phys` - Physical address of the kernel's own PML4, used
    ///                        as the source for the upper-half copy. Pass the
    ///                        same CR3 value used by all kernel processes.
    /// * `user_entry`       - Virtual address of the user-mode entry point in
    ///                        the new process's address space.
    ///
    /// # Returns
    ///
    /// Returns a strong `Arc` reference to the new process on success, or
    /// [`SchedulerError`] if the scheduler task table or run queue is full.
    pub fn create_user_process(
        &self,
        name: &'static str,
        kernel_pml4_phys: u64,
        user_entry: u64,
    ) -> Result<Arc<Process>, SchedulerError> {
        // Allocate a fresh address space and create the process. No user
        // mappings exist yet - `spawn_user_thread` maps the user stack below.
        let process = Process::new_user(name, kernel_pml4_phys);

        // Spawn the main thread. This maps the user stack into the process's
        // address space and sets up the iretq frame. The thread is queued in
        // the scheduler and eligible to run as soon as this returns.
        process.spawn_user_thread("main", user_entry)?;

        // Only register the process if the thread spawned successfully.
        // If spawn_user_thread returned an error, the process Arc is dropped
        // here and its address space is cleaned up without polluting the
        // process list with a threadless process.
        self.processes.lock().push(Arc::clone(&process));

        Ok(process)
    }

    /// Looks up a process by name; useful for debug output and logging.
    /// 
    /// # Arguments
    /// 
    /// * `name` - Process name string to find in the process list.
    /// 
    /// # Returns
    ///
    /// Returns an `Arc` of the process as `Some` if found, or `None` otherwise.
    pub fn find_by_name(&self, name: &str) -> Option<Arc<Process>> {
        self.processes
            .lock()
            .iter()
            .find(|process| process.name == name)
            .cloned()
    }

    /// Prints out current process and thread status
    pub fn print_processes(&self) {
        let processes = self.processes.lock();
        for proc in processes.iter() {
            sprintln!("\n========== PROCESS INFORMATION ==========");
            fbprintln!("\n========== PROCESS INFORMATION ==========");
            sprintln!("Process: {} (ID: {}, PML4: {:#X})", proc.name, proc.id.as_u64(), proc.pml4_phys());
            fbprintln!("Process: {} (ID: {}, PML4: {:#X})", proc.name, proc.id.as_u64(), proc.pml4_phys());

            {
                let threads = proc.threads.lock();
                sprintln!("Threads: {}", threads.len());
                fbprintln!("Threads: {}", threads.len());
                for thread in threads.iter() {
                    let locked_thread = thread.lock();
                    sprintln!("    Thread {} (ID: {}, State: {:?})", locked_thread.name, locked_thread.id.as_u64(), locked_thread.state);
                    fbprintln!("    Thread {} (ID: {}, State: {:?})", locked_thread.name, locked_thread.id.as_u64(), locked_thread.state);
                }
            }
        }
    }
}
