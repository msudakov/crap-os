//! System Globals
//! 
//! This file contains global system resources, some of which must be
//! synchronized and protected from races and deadlocks.

use core::sync::atomic::{AtomicU64, AtomicBool};
use crate::{DebugLevel};
use crate::spinlock::StaticIrqSpinLock;
use crate::hardware_manager::{SerialWriter, FramebufferWriter};
use crate::memory_manager::{MemoryManager, LockedHeap};
use crate::process_manager::{ProcessManager};
use crate::hardware_manager::hpet::HpetInfo;
use crate::processor_control::per_cpu::PerCpu;

// =============================================================================
// Basic Globals
// =============================================================================

/// Standard base address for COM1 serial port, used for serial port functions.
pub const COM1_PORT: u16 = 0x3F8;

/// System-wide debug message level.
pub const DEBUG_LEVEL: DebugLevel = DebugLevel::INFO;

/// Upper hexadecimal character set.
pub const HEX_CHARS_UPPER: &[u8; 16] = b"0123456789ABCDEF";

/// Standard Base64 alphabet.
pub const BASE64_CHARS: &[u8; 64] =
    b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

// Base addresses for the higher-half kernel mappings
pub const KERNEL_PHYSICAL_MAP_BASE: u64        = 0xFFFF800000000000;
pub const KERNEL_FRAMEBUFFER_VIRTUAL_BASE: u64 = 0xFFFF900000000000;
pub const KERNEL_HEAP_BASE: u64                = 0xFFFFA00000000000; //64 MB max
pub const KERNEL_VIRTUAL_BASE: u64             = 0xFFFFFFFF80000000;

/// Default page size of 4096 bytes.
pub const PAGE_SIZE: u64 = 0x1000;

// Kernel physical start and physical end tags collected from the linker
unsafe extern "C" {
    pub static __kernel_phys_start: u8;
    pub static __kernel_phys_end: u8;
}

/// This defines task quantum (runtime duration before preemption) for use in
/// the Task Scheduler. It is the number of system clock ticks for each task
/// quantum.
pub const TASK_QUANTUM_TICKS: u32 = 4;  // 4ms at 1ms tick rate


// =============================================================================
// Synchronized Globals
// =============================================================================

/// Writer singleton for serial port.
pub static SERIAL: StaticIrqSpinLock<Option<SerialWriter>> =
    StaticIrqSpinLock::new(None);

/// Writer singleton for framebuffer.
pub static FRAMEBUFFER: StaticIrqSpinLock<Option<FramebufferWriter>> =
    StaticIrqSpinLock::new(None);

/// Memory manager singleton.
pub static MEMORY_MANAGER: StaticIrqSpinLock<Option<MemoryManager>> =
    StaticIrqSpinLock::new(None);

/// Kernel heap singleton for 64 MB max.
pub static KERNEL_HEAP: LockedHeap = LockedHeap::new(
    KERNEL_HEAP_BASE, 64 * 1024 * 1024);

/// Global Process Manager singleton.
pub static PROCESS_MANAGER: ProcessManager = ProcessManager::new();

/// Global monotonic tick counter, incremented by the BSP's timer ISR on every
/// tick. Used as a wall-clock reference for sleep and timeout calculations. On
/// SMP, this remains BSP-only.
///
/// `AtomicU64` makes the counter safe to read from any context (interrupt
/// handlers, kernel threads) without a lock. `Relaxed` ordering is acceptable
/// for reads because the counter is not used to establish a happens-before
/// relationship with other shared data, so callers only need the numeric value.
pub static TIMER_TICKS: AtomicU64 = AtomicU64::new(0);

/// Global High Precision Event Timer (HPET) information structure, used in
/// calibrating the system clock timer.
pub static HPET: StaticIrqSpinLock<Option<HpetInfo>> =
    StaticIrqSpinLock::new(None);

/// Atomic boolean flag to track if the kernel's initialization sequence has
/// finished or still ongoing. This is used when deciding whether the
/// idle task should be re-inserted into the `Ready` task queue to be re-
/// scheduled when the timer ISR fires and task preemption occurs. After the
/// kernel is initialized, the idle task only ever runs if no other task is
/// ready to be executed in the Task Scheduler's queue.
pub static SYS_FLAG_KERNEL_INIT_COMPLETE: AtomicBool = AtomicBool::new(false);

/// Atomic boolean flag used to signal the timer ISR on a given CPU to disregard
/// the result of task quantum check and force the scheduler to run regardless.
/// This is set by SystemTask routines tied to task and process management
/// (e.g., when a task is being killed).
pub static CPU_FORCE_RESCHEDULE: PerCpu<AtomicBool> = PerCpu::new();

/// Per-CPU quantum countdown. Holds the number of timer ticks remaining in
/// the current task's time slice on each CPU. Decremented by the timer ISR
/// directly, with no scheduler lock needed.
///
/// Initialized for the BSP during scheduler init. Each AP initializes its
/// own slot during AP bring-up.
pub static CPU_TICKS_REMAINING: PerCpu<u32> = PerCpu::new();
