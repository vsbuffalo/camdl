//! `FiniteF64` — the *resolved user input* float policy.
//!
//! A resolved input (param value, dt, t_start/t_end, a bound) must be a
//! finite number, and two spellings of zero should hit the same cache. So
//! `FiniteF64`:
//!
//! - rejects `NaN`/`±Inf` at construction (a non-finite param is a
//!   [`NonFiniteFloat`] error, surfaced *before* any hashing — totality is
//!   preserved because the failure is a value, not a panic), and
//! - normalizes `-0.0 → +0.0`, so `--param x=-0` and `--param x=0` resolve
//!   to the same identity.
//!
//! This is the *opposite* policy from structural IR floats
//! ([`crate::CanonicalHasher::write_f64_bits`]), which keep `±0.0` and NaN
//! payloads distinct to match the IR's `ConstExpr::PartialEq`. The two are
//! one hasher with a field-level policy: routing an IR float through
//! `FiniteF64` would erase a distinction the IR treats as real *and* reject
//! NaN-bearing consts at hash time (a totality break), so the policy is
//! chosen by the field's *type*, not by a runtime flag.

use crate::hash::{CanonicalHasher, ContentAddressed};

/// A resolved, finite `f64` with `-0.0` normalized to `+0.0`. Construct via
/// [`FiniteF64::new`]; the invariant (finite, no negative zero) holds for
/// every value of the type, so its `ContentAddressed` impl can hash the raw
/// bits unconditionally.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FiniteF64(f64);

/// A resolved float was `NaN` or `±Inf`. Surfaced before hashing — the
/// [`crate::ResolveError`] taxonomy wraps this as `NonFiniteParam`.
#[derive(Debug, Clone, Copy, thiserror::Error)]
#[error("non-finite resolved value: {0}")]
pub struct NonFiniteFloat(pub f64);

impl FiniteF64 {
    /// Construct from a raw `f64`. Returns [`NonFiniteFloat`] for `NaN` or
    /// `±Inf`; normalizes `-0.0 → +0.0` otherwise.
    pub fn new(x: f64) -> Result<Self, NonFiniteFloat> {
        if !x.is_finite() {
            return Err(NonFiniteFloat(x));
        }
        // `x == 0.0` is true for both `+0.0` and `-0.0`; assigning the
        // literal `+0.0` normalizes the sign. All other values pass through.
        let normalized = if x == 0.0 { 0.0 } else { x };
        Ok(FiniteF64(normalized))
    }

    /// The underlying finite, sign-normalized value.
    pub fn get(self) -> f64 {
        self.0
    }
}

impl ContentAddressed for FiniteF64 {
    fn hash_into(&self, h: &mut CanonicalHasher) {
        // The invariant guarantees finite + non-negative-zero, so raw bits
        // are canonical: `+0.0` → 0, and `-0.0` is unreachable.
        h.write_f64_bits(self.0);
    }
}
