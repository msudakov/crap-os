// This file contains global system resources, some of which must be
// synchronized and protected from races and deadlocks.

// =============================================================================
// Basic Globals
// =============================================================================


// =============================================================================
// Synchronized Globals
// =============================================================================

use crate::spinlock::StaticIrqSpinLock;
use crate::framebuffer::FramebufferWriter;

pub static FRAMEBUFFER: StaticIrqSpinLock<Option<FramebufferWriter>> =
    StaticIrqSpinLock::new(None);
