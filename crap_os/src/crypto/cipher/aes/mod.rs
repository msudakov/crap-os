//! AES (Advanced Encryption Standard) Block Cipher and Modes of Operation.
//!
//! This module provides AES-256 as both a raw block cipher primitive and as
//! the foundation for higher-level authenticated encryption modes. It contains
//! the following submodules:
//!
//! - [`aes`]: Core AES-256 block cipher. Key schedule expansion, the round
//!   functions (SubBytes, ShiftRows, MixColumns, AddRoundKey), and the
//!   single-block encryption primitive. Internal to this crate; not re-exported
//!   as part of the public API.
//!
//! - [`aes_gcm`]: AES-256-GCM authenticated encryption. Builds on the core
//!   block cipher to provide AEAD with a 128-bit authentication tag and a
//!   randomly generated 96-bit IV. See [`aes_gcm::encrypt`] and
//!   [`aes_gcm::decrypt`] for the public-facing API.

mod aes;
pub mod aes_gcm;

// Re-export public APIs
pub use aes_gcm::{encrypt, decrypt};
