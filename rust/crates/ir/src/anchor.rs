//! Observation anchors (gh#616) — the symbolic end (or start) of observed data,
//! plus a compile-time-folded constant offset.
//!
//! An anchor is deliberately NOT an [`crate::expr::Expr`]: it may appear only in
//! the three positions the DSL grants it (`simulate { to }` / a preset's, a
//! forcing `breakpoints` entry, and `value_at`'s time argument), so it cannot
//! leak into a propensity. Its VALUE is data-dependent, resolved once per run
//! from the run's bound observation streams; its OFFSET is not, having been
//! folded to model time units by the compiler. That split is what keeps the
//! compiled model data-independent.
//!
//! Wire form (see `ocaml/lib/ir/serde.ml`, `anchored_time_to_json` — the two
//! sides must agree): a zero offset is the bare string `"last_obs"` /
//! `"first_obs"`; a non-zero offset is `{"anchor": …, "offset": …}`. The bare
//! string is exactly what `value_at(…, last_obs)` emitted before this feature,
//! so the pre-gh#616 corpus is byte-identical.

use serde::{Deserialize, Serialize};
use std::fmt;

/// Which end of the observed record an anchor names.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ObsAnchor {
    /// The minimum observation time over the run's bound streams.
    #[serde(rename = "first_obs")]
    First,
    /// The maximum observation time over the run's bound streams.
    #[serde(rename = "last_obs")]
    Last,
}

impl ObsAnchor {
    /// The DSL/wire spelling — also the name a diagnostic should print.
    pub fn as_str(self) -> &'static str {
        match self {
            ObsAnchor::First => "first_obs",
            ObsAnchor::Last => "last_obs",
        }
    }
}

impl fmt::Display for ObsAnchor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// An observation anchor plus a signed constant offset **in model time units**.
/// `offset == 0.0` is a bare anchor.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[serde(from = "AnchoredTimeWire", into = "AnchoredTimeWire")]
pub struct AnchoredTime {
    pub anchor: ObsAnchor,
    pub offset: f64,
}

// Bitwise equality on the offset, matching `ConstExpr`: two IR nodes that differ
// only in zero sign or NaN payload should be observably distinct, and this type
// feeds run identity.
impl PartialEq for AnchoredTime {
    fn eq(&self, other: &Self) -> bool {
        self.anchor == other.anchor && self.offset.to_bits() == other.offset.to_bits()
    }
}
impl Eq for AnchoredTime {}

impl AnchoredTime {
    /// A bare anchor (zero offset).
    pub fn bare(anchor: ObsAnchor) -> Self {
        AnchoredTime { anchor, offset: 0.0 }
    }

    /// The resolved model time, given the anchor's resolved value.
    pub fn resolve(&self, anchor_time: f64) -> f64 {
        anchor_time + self.offset
    }
}

/// How an anchored time reads to a human: `last_obs`, `last_obs + 28`,
/// `first_obs - 7` (offset in model time units).
impl fmt::Display for AnchoredTime {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.offset == 0.0 {
            write!(f, "{}", self.anchor)
        } else if self.offset > 0.0 {
            write!(f, "{} + {}", self.anchor, self.offset)
        } else {
            write!(f, "{} - {}", self.anchor, -self.offset)
        }
    }
}

// ── Wire form ─────────────────────────────────────────────────────────────────

/// The two accepted JSON spellings. Serialization always produces the canonical
/// one (bare string at zero offset), so a decode→encode round-trip is stable.
#[derive(Serialize, Deserialize)]
#[serde(untagged)]
enum AnchoredTimeWire {
    Bare(ObsAnchor),
    Offset {
        anchor: ObsAnchor,
        #[serde(default)]
        offset: f64,
    },
}

impl From<AnchoredTimeWire> for AnchoredTime {
    fn from(w: AnchoredTimeWire) -> Self {
        match w {
            AnchoredTimeWire::Bare(anchor) => AnchoredTime { anchor, offset: 0.0 },
            AnchoredTimeWire::Offset { anchor, offset } => AnchoredTime { anchor, offset },
        }
    }
}

impl From<AnchoredTime> for AnchoredTimeWire {
    fn from(a: AnchoredTime) -> Self {
        if a.offset == 0.0 {
            AnchoredTimeWire::Bare(a.anchor)
        } else {
            AnchoredTimeWire::Offset { anchor: a.anchor, offset: a.offset }
        }
    }
}

impl std::str::FromStr for ObsAnchor {
    type Err = ();
    /// The single place a string becomes an anchor — shared by the IR decoder
    /// and the CLI's `--to` / `condition_from` spec parser, so the accepted
    /// spellings cannot diverge between the DSL and the CLI.
    fn from_str(s: &str) -> Result<Self, ()> {
        match s {
            "first_obs" => Ok(ObsAnchor::First),
            "last_obs" => Ok(ObsAnchor::Last),
            _ => Err(()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The canonical emission: a bare anchor is the bare string — byte-identical
    /// to what `value_at(…, last_obs)` emitted before gh#616.
    #[test]
    fn zero_offset_serialises_as_the_bare_string() {
        let a = AnchoredTime::bare(ObsAnchor::Last);
        assert_eq!(serde_json::to_string(&a).unwrap(), r#""last_obs""#);
        let f = AnchoredTime::bare(ObsAnchor::First);
        assert_eq!(serde_json::to_string(&f).unwrap(), r#""first_obs""#);
    }

    #[test]
    fn nonzero_offset_serialises_as_the_object() {
        let a = AnchoredTime { anchor: ObsAnchor::Last, offset: 28.0 };
        assert_eq!(
            serde_json::to_string(&a).unwrap(),
            r#"{"anchor":"last_obs","offset":28.0}"#
        );
        let b = AnchoredTime { anchor: ObsAnchor::First, offset: -7.0 };
        assert_eq!(
            serde_json::to_string(&b).unwrap(),
            r#"{"anchor":"first_obs","offset":-7.0}"#
        );
    }

    /// Both spellings decode; the object form with a zero offset normalises back
    /// to the bare string, so re-encoding is canonical.
    #[test]
    fn both_wire_forms_decode() {
        let bare: AnchoredTime = serde_json::from_str(r#""last_obs""#).unwrap();
        assert_eq!(bare, AnchoredTime::bare(ObsAnchor::Last));

        let obj: AnchoredTime =
            serde_json::from_str(r#"{"anchor":"first_obs","offset":-7.0}"#).unwrap();
        assert_eq!(obj, AnchoredTime { anchor: ObsAnchor::First, offset: -7.0 });

        let zero_obj: AnchoredTime =
            serde_json::from_str(r#"{"anchor":"last_obs","offset":0.0}"#).unwrap();
        assert_eq!(serde_json::to_string(&zero_obj).unwrap(), r#""last_obs""#);
    }

    #[test]
    fn unknown_anchor_is_rejected() {
        assert!(serde_json::from_str::<AnchoredTime>(r#""mid_obs""#).is_err());
        assert!(serde_json::from_str::<AnchoredTime>(r#"{"anchor":"mid_obs"}"#).is_err());
    }

    #[test]
    fn from_str_matches_the_wire_spelling() {
        use std::str::FromStr;
        assert_eq!(ObsAnchor::from_str("first_obs"), Ok(ObsAnchor::First));
        assert_eq!(ObsAnchor::from_str("last_obs"), Ok(ObsAnchor::Last));
        assert!(ObsAnchor::from_str("lastobs").is_err());
        // The parser and the emitter agree in both directions.
        for a in [ObsAnchor::First, ObsAnchor::Last] {
            assert_eq!(ObsAnchor::from_str(a.as_str()), Ok(a));
        }
    }

    #[test]
    fn resolve_applies_the_offset() {
        let a = AnchoredTime { anchor: ObsAnchor::Last, offset: 28.0 };
        assert_eq!(a.resolve(91.0), 119.0);
        let b = AnchoredTime { anchor: ObsAnchor::First, offset: -7.0 };
        assert_eq!(b.resolve(3.0), -4.0);
    }

    #[test]
    fn display_reads_like_the_dsl() {
        assert_eq!(AnchoredTime::bare(ObsAnchor::Last).to_string(), "last_obs");
        assert_eq!(
            AnchoredTime { anchor: ObsAnchor::Last, offset: 28.0 }.to_string(),
            "last_obs + 28"
        );
        assert_eq!(
            AnchoredTime { anchor: ObsAnchor::First, offset: -7.0 }.to_string(),
            "first_obs - 7"
        );
    }
}
