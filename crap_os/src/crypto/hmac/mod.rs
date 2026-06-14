//! HMAC (Hash-based Message Authentication Code) Functions
//!
//! This module contains cryptographic HMAC construction functions, which
//! produce a fixed-size authentication tag over an arbitrary-length message
//! and secret key.

mod sha256;

// Re-export public APIs
pub use sha256::hmac_sha256;
