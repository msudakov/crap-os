//! Symmetric and Asymmetric Cipher Constructions
//!
//! This module collects block ciphers, stream ciphers, and public-key
//! constructions along with their modes of operation. All public-facing
//! encrypt and decrypt functions are re-exported from the top-level [`crypto`]
//! module; internal primitives (key schedules, round functions, block
//! operations, field arithmetic) are intentionally not exposed outside the
//! crate.

mod aes;

// Re-export public APIs
pub use aes::aes_gcm::{encrypt as aes_gcm_encrypt, decrypt as aes_gcm_decrypt};
