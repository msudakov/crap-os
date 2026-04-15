use alloc::sync::Arc;
use alloc::vec::Vec;
use crate::spinlock::IrqSpinLock;
use super::process::Process;
use crate::task_scheduler::{SchedulerError, TaskId};
use crate::{sprintln, fbprintln};

pub struct ProcessManager {
    // IrqSpinLock: process list may be consulted from IRQ context
    // (e.g. during a fault that needs to identify the current process).
    processes: IrqSpinLock<Vec<Arc<Process>>>,
}

impl ProcessManager {
    pub const fn new() -> Self {
        ProcessManager {
            processes: IrqSpinLock::new(Vec::new()),
        }
    }

    pub fn init_idle_process(&self, cr3: u64) -> Arc<Process> {
        let idle_process = Process::new_idle(cr3);
        let idle_thread = super::Thread::new_idle(Arc::downgrade(&idle_process));
        
        idle_thread.lock().task_id = Some(TaskId::IDLE);
        idle_process.threads.lock().push(Arc::clone(&idle_thread));
        self.processes.lock().push(Arc::clone(&idle_process));

        crate::task_scheduler::init_idle(Arc::downgrade(&idle_thread));
        
        idle_process
    }

    pub fn create_process(
        &self,
        name: &'static str,
        cr3: u64,
        main_entry: fn(u64),
        main_arg: u64,
    ) -> Result<Arc<Process>, SchedulerError> {
        let process = Process::new(name, cr3);
        process.spawn_thread("main", main_entry, main_arg)?;
        self.processes.lock().push(Arc::clone(&process));
        Ok(process)
    }

    pub fn print_processes(&self) {
        let processes = self.processes.lock();
        for proc in processes.iter() {
            sprintln!("\n========== PROCESS INFORMATION ==========");
            fbprintln!("\n========== PROCESS INFORMATION ==========");
            sprintln!("Process: {} (ID: {}, PML4: {:#X})", proc.name, proc.id.as_u64(), proc.cr3);
            fbprintln!("Process: {} (ID: {}, PML4: {:#X})", proc.name, proc.id.as_u64(), proc.cr3);

            {
                let threads = proc.threads.lock();
                sprintln!("Threads: {}", threads.len());
                fbprintln!("Threads: {}", threads.len());
                for thread in threads.iter() {
                    let locked_thread = thread.lock();
                    sprintln!("    Thread {} (ID: {}, Task: {}, State: {:?})", locked_thread.name, locked_thread.id.as_u64(), locked_thread.task_id.unwrap().as_u64(), locked_thread.state);
                    fbprintln!("    Thread {} (ID: {}, Task: {}, State: {:?})", locked_thread.name, locked_thread.id.as_u64(), locked_thread.task_id.unwrap().as_u64(), locked_thread.state);
                }
            }
        }
    }

    /// Look up a process by name. Useful during early init before you
    /// have a richer handle system.
    pub fn find_by_name(&self, name: &str) -> Option<Arc<Process>> {
        self.processes
            .lock()
            .iter()
            .find(|p| p.name == name)
            .cloned()
    }
}
