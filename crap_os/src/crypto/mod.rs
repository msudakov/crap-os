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
#![allow(unused_imports)]

pub mod rng;
mod hash;
mod hmac;

// Re-export public APIs
pub use rng::{init_cpu, get_random_bytes_vec, get_pseudo_random_bytes_vec};

pub use hash::{md5, sha1, sha256, sha3_512};
pub use hash::{hash_password, hash_password_with_params, verify_password};
pub use hmac::hmac_sha256;
