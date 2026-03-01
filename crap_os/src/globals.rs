// =============================================================================
// System Globals
// =============================================================================
// 
// This file contains global system resources, some of which must be
// synchronized and protected from races and deadlocks.

use crate::spinlock::StaticIrqSpinLock;
use crate::serial::SerialWriter;
use crate::framebuffer::FramebufferWriter;
use crate::DebugLevel;
use crate::memory_manager::MemoryManager;

// =============================================================================
// Basic Globals
// =============================================================================

// Standard base address for COM1 serial port, used for serial port functions
pub const COM1_PORT: u16 = 0x3F8;

// System-wide debug level
pub const DEBUG_LEVEL: DebugLevel = DebugLevel::INFO;

// Upper hexadecimal character set
pub const HEX_CHARS_UPPER: &[u8; 16] = b"0123456789ABCDEF";

// =============================================================================
// Synchronized Globals
// =============================================================================

// Writer singleton for serial port
pub static SERIAL: StaticIrqSpinLock<Option<SerialWriter>> =
    StaticIrqSpinLock::new(None);

// Writer singleton for framebuffer
pub static FRAMEBUFFER: StaticIrqSpinLock<Option<FramebufferWriter>> =
    StaticIrqSpinLock::new(None);

// Memory manager singleton
pub static MEMORY_MANAGER: StaticIrqSpinLock<Option<MemoryManager>> =
    StaticIrqSpinLock::new(None);
