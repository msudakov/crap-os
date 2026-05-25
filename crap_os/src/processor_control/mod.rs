//! Processor Control Module
//!
//! This module owns everything that is specific to the CPU itself: topology
//! enumeration, per-CPU state, descriptor tables (GDT, IDT), and AP bring-up
//! and IPI dispatch.

pub mod topology;
pub mod per_cpu;
pub mod gdt;
pub mod idt;

pub use topology::{
    init_cpu_topology,
    /*CpuInfo,
    CpuTopology,
    MAX_CPUS,
    CPU_TOPOLOGY,*/
    };
pub use per_cpu::{CpuId, PerCpu};
pub use gdt::{
    // Runtime API
    set_kernel_stack,
    /*init_gdt,
    ap_init_gdt,*/
    // Types needed by other modules
    /*Tss,
    Gdt,
    IstStack,*/
    // Per-CPU statics
    /*CPU_TSS,
    CPU_GDT,
    CPU_DOUBLE_FAULT_STACK,*/
    // Selector constants referenced from idt.rs and task.rs
    /*KERNEL_CS,
    KERNEL_DS,
    TSS_SELECTOR,
    USER_CS,
    USER_DS,
    USER_CS_RPL3,
    USER_DS_RPL3,
    DOUBLE_FAULT_IST_SIZE,*/
};
pub use idt::{init_idt, load_idt};
