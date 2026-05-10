//! Processor Control Module
//!
//! This module owns everything that is specific to the CPU itself: topology
//! enumeration, per-CPU state, descriptor tables (GDT, IDT), and AP bring-up
//! and IPI dispatch.

pub mod topology;

pub use topology::{CpuInfo, CpuTopology, MAX_CPUS};
