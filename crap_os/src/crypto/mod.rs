//! Cryptographic Primitives
//!
//! This module is the kernel's cryptographic foundation, providing the
//! building blocks for any operation that requires randomness, secrecy,
//! integrity, or authenticity. All implementations are `no_std`-compatible,
//! allocation-optional, and written from scratch without third-party crates,
//! operating directly on hardware entropy sources and CPU instructions where
//! applicable.
//!
//! # Security model
//!
//! All cryptographic code in this module assumes a cooperative hardware
//! environment (i.e., the CPU and firmware are trusted) and is not designed to
//! defend against physical side-channel attacks such as power analysis or fault
//! injection. Software-level side-channel hygiene (constant-time comparisons,
//! secret-independent memory access patterns) is applied where relevant and
//! noted in the documentation of each submodule.

pub mod rng;

pub use rng::{init_cpu, get_random_bytes_vec};
