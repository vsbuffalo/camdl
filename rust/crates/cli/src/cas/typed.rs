//! [`ContentHash`] — a SHA-256 newtype the CLI uses to keep content hashes
//! type-distinct from arbitrary strings. Constructed from canonicalized input
//! bytes via [`ContentHash::from_bytes`]; used by `profile` and `pfilter` at the sites that fold an input slice into a level hash.

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

// ─── ContentHash ─────────────────────────────────────────────────────────────

/// 64-char hex SHA-256 of canonicalized content inputs. Newtype so the
/// type system distinguishes content hashes from arbitrary strings (a
/// `String` parameter that happens to hold a hash will not type-check).
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ContentHash(String);

impl ContentHash {
    /// Hex over arbitrary bytes — the standard way to construct a
    /// ContentHash from canonicalized input.
    pub fn from_bytes(bytes: &[u8]) -> Self {
        Self(hex_of(&Sha256::digest(bytes)))
    }

    /// Full 64-char hex digest.
    pub fn full(&self) -> &str { &self.0 }
}

impl std::fmt::Display for ContentHash {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

fn hex_of(bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut out = String::with_capacity(bytes.len() * 2);
    for b in bytes {
        out.push(HEX[(b >> 4) as usize] as char);
        out.push(HEX[(b & 0xf) as usize] as char);
    }
    out
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn content_hash_from_bytes_is_deterministic() {
        let a = ContentHash::from_bytes(b"hello");
        let b = ContentHash::from_bytes(b"hello");
        assert_eq!(a, b);
        assert_eq!(a.full().len(), 64);
    }

}
