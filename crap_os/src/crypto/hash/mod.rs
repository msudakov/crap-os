//! Non-Keyed Cryptographic and General-Purpose Hash Functions
//!
//! This module provides stateless hashing functions that operate on raw byte
//! slices and return fixed-size byte arrays. Each function processes the entire
//! input in one call; for streaming or incremental hashing, a future update
//! will add incremental wrappers.

mod md5;
mod sha1;
mod sha2;
mod sha3;
mod blake2b;
mod argon2;

// Re-export public APIs
pub use md5::md5;
pub use sha1::sha1;
pub use sha2::sha256;
pub use sha3::sha3_512;
pub use argon2::{hash_password, hash_password_with_params, verify_password};
