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
pub mod blake2b;

// Re-export public APIs
pub use md5::md5;
pub use sha1::sha1;
pub use sha2::sha256;
pub use sha3::sha3_512;

pub use blake2b::blake2b_512;
pub use blake2b::blake2b_256;
pub use blake2b::blake2b_variable;
pub use blake2b::blake2b_mac_512;
pub use blake2b::blake2b_mac_256;
pub use blake2b::blake2b;
pub use blake2b::blake2b_512_slice;
