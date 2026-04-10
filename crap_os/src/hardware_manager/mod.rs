//! Hardware Manager Module
//! 
//! The Hardware Manager module is responsible for all abstraction and
//! interoperation of the system's hardware components.

pub mod serial;
pub mod framebuffer;
pub mod acpi;
pub mod apic;
pub mod hpet;
pub mod keyboard;

// Re-export public APIs
pub use serial::SerialWriter;
pub use serial::print as sprint;
pub use serial::init as serial_init;
pub use framebuffer::{FramebufferInfo, FramebufferWriter};
pub use keyboard::{keyboard_push_scancode, keyboard_pop_scancode,
    process_scancode, keyboard_set_task_id};
pub use acpi::parse_acpi;
pub use hpet::{HpetInfo, parse_hpet};
pub use apic::{init_apic, eoi, disable_pic_8259, configure_timer,
    calibrate_timer, ioapic_unmask_irq};
