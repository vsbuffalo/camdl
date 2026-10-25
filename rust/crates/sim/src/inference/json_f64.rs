//! An `f64` in JSON that keeps its non-finite values.
//!
//! JSON has no literal for `+∞`, `−∞` or `NaN`, and `serde_json` writes every
//! one of them as `null`, which it then refuses to read back as an `f64`. One
//! non-finite number therefore makes the whole document unreadable, and the
//! `null` it leaves cannot say which of the three it was. An `R̂` of `+∞` — a
//! sampler whose chains each sat at their own single value — did exactly this
//! to `diagnostics.json`, and `fit summary` then dropped every record in it,
//! the refused chains' `bad_init` records included.
//!
//! Here a finite value is the JSON number `serde_json` has always written, to
//! the byte, so an ordinary file does not change; a non-finite one is the
//! string `"inf"`, `"-inf"` or `"NaN"` (`f64`'s own `Display` spelling). The
//! reader accepts a number or exactly one of those strings, and refuses `null`
//! with [`LEGACY_NULL`]: a `null` was written for all three non-finite values
//! alike, so admitting it as any one of them would be a guess.
//!
//! Use `#[serde(with = "crate::inference::json_f64")]` on an `f64`,
//! [`option`] on an `Option<f64>` (where `None` stays `null`), and [`map`] on
//! a `BTreeMap<String, f64>`.
//!
//! JSON only: the reader dispatches on the token it finds (`deserialize_any`),
//! which a non-self-describing format such as bincode cannot serve. No type
//! that uses this is written as bincode.

use serde::de::{self, Unexpected, Visitor};
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use std::fmt;

/// The error a `null` in one of these fields reads as. Written by every
/// camdl binary before non-finite numbers were spelled out, for `+∞`, `−∞`
/// and `NaN` alike — so which of the three it was is not in the file.
pub const LEGACY_NULL: &str = "null where a number was expected: the file was written before \
     camdl spelled a non-finite number as \"inf\", \"-inf\" or \"NaN\", and null stood for all \
     three, so the value is not recoverable from it";

/// The string a non-finite value is written as.
fn spelling(v: f64) -> &'static str {
    if v.is_nan() {
        "NaN"
    } else if v > 0.0 {
        "inf"
    } else {
        "-inf"
    }
}

pub fn serialize<S: Serializer>(v: &f64, s: S) -> Result<S::Ok, S::Error> {
    if v.is_finite() {
        s.serialize_f64(*v)
    } else {
        s.serialize_str(spelling(*v))
    }
}

pub fn deserialize<'de, D: Deserializer<'de>>(d: D) -> Result<f64, D::Error> {
    d.deserialize_any(NumberVisitor)
}

struct NumberVisitor;

impl<'de> Visitor<'de> for NumberVisitor {
    type Value = f64;

    fn expecting(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(r#"a number, or one of "inf", "-inf", "NaN""#)
    }

    fn visit_f64<E: de::Error>(self, v: f64) -> Result<f64, E> {
        Ok(v)
    }

    // An integer token reads as the `f64` it names, as it does for a plain
    // `f64` field: a hand-written `"threshold": 1` is not an error.
    fn visit_u64<E: de::Error>(self, v: u64) -> Result<f64, E> {
        Ok(v as f64)
    }

    fn visit_i64<E: de::Error>(self, v: i64) -> Result<f64, E> {
        Ok(v as f64)
    }

    fn visit_str<E: de::Error>(self, v: &str) -> Result<f64, E> {
        match v {
            "inf" => Ok(f64::INFINITY),
            "-inf" => Ok(f64::NEG_INFINITY),
            "NaN" => Ok(f64::NAN),
            other => Err(E::invalid_value(Unexpected::Str(other), &self)),
        }
    }

    fn visit_unit<E: de::Error>(self) -> Result<f64, E> {
        Err(E::custom(LEGACY_NULL))
    }
}

/// The value as a type, for the containers below.
struct Number(f64);

impl Serialize for Number {
    fn serialize<S: Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        serialize(&self.0, s)
    }
}

impl<'de> Deserialize<'de> for Number {
    fn deserialize<D: Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        deserialize(d).map(Number)
    }
}

/// `Option<f64>`: `None` is `null`, as it always was; `Some` is a number or a
/// non-finite spelling. A `null` reads as `None` — which is also what a file
/// written before this module read as, so a `Some(+∞)` such a file recorded
/// as `null` still reads as `None`; that loss is in the file, not the reader.
/// Pair with `#[serde(default)]` so a missing key stays `None`, as it is for
/// an `Option` field without a `with`.
pub mod option {
    use super::*;

    pub fn serialize<S: Serializer>(v: &Option<f64>, s: S) -> Result<S::Ok, S::Error> {
        match v {
            Some(x) => s.serialize_some(&Number(*x)),
            None => s.serialize_none(),
        }
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(d: D) -> Result<Option<f64>, D::Error> {
        d.deserialize_any(OptionVisitor)
    }

    struct OptionVisitor;

    impl<'de> Visitor<'de> for OptionVisitor {
        type Value = Option<f64>;

        fn expecting(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            f.write_str(r#"null, a number, or one of "inf", "-inf", "NaN""#)
        }

        fn visit_unit<E: de::Error>(self) -> Result<Option<f64>, E> {
            Ok(None)
        }

        fn visit_f64<E: de::Error>(self, v: f64) -> Result<Option<f64>, E> {
            NumberVisitor.visit_f64(v).map(Some)
        }

        fn visit_u64<E: de::Error>(self, v: u64) -> Result<Option<f64>, E> {
            NumberVisitor.visit_u64(v).map(Some)
        }

        fn visit_i64<E: de::Error>(self, v: i64) -> Result<Option<f64>, E> {
            NumberVisitor.visit_i64(v).map(Some)
        }

        fn visit_str<E: de::Error>(self, v: &str) -> Result<Option<f64>, E> {
            NumberVisitor.visit_str(v).map(Some)
        }
    }
}

/// `BTreeMap<String, f64>`, each value as above.
pub mod map {
    use super::*;
    use std::collections::BTreeMap;

    pub fn serialize<S: Serializer>(m: &BTreeMap<String, f64>, s: S) -> Result<S::Ok, S::Error> {
        s.collect_map(m.iter().map(|(k, v)| (k, Number(*v))))
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(d: D) -> Result<BTreeMap<String, f64>, D::Error> {
        let m = BTreeMap::<String, Number>::deserialize(d)?;
        Ok(m.into_iter().map(|(k, v)| (k, v.0)).collect())
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    #[derive(Debug, serde::Serialize, serde::Deserialize)]
    struct Holder {
        #[serde(with = "super")]
        x: f64,
        #[serde(with = "super::option", default)]
        o: Option<f64>,
        #[serde(with = "super::map")]
        m: BTreeMap<String, f64>,
    }

    #[derive(Debug, serde::Serialize, serde::Deserialize)]
    struct Plain {
        x: f64,
        o: Option<f64>,
        m: BTreeMap<String, f64>,
    }

    /// A finite value is the bytes `serde_json` writes for a plain `f64` —
    /// including the shortest-round-trip spellings of small and large values.
    #[test]
    fn finite_values_are_written_as_the_plain_number() {
        for x in [0.0, -0.0, 1.0, 0.25, 6.678e-5, 1.0e21, -3.5, f64::MAX, f64::MIN_POSITIVE] {
            let m = BTreeMap::from([("beta".to_string(), x)]);
            for o in [None, Some(x)] {
                let with = serde_json::to_string(&Holder { x, o, m: m.clone() }).unwrap();
                let plain = serde_json::to_string(&Plain { x, o, m: m.clone() }).unwrap();
                assert_eq!(with, plain, "a finite {x} must not change on disk");
            }
        }
    }

    #[test]
    fn non_finite_values_are_spelled_and_read_back() {
        for (x, s) in [(f64::INFINITY, "inf"), (f64::NEG_INFINITY, "-inf"), (f64::NAN, "NaN")] {
            let h = Holder { x, o: Some(x), m: BTreeMap::from([("beta".to_string(), x)]) };
            let json = serde_json::to_string(&h).unwrap();
            assert_eq!(json, format!(r#"{{"x":"{s}","o":"{s}","m":{{"beta":"{s}"}}}}"#));
            let back: Holder = serde_json::from_str(&json).unwrap();
            for v in [back.x, back.o.unwrap(), back.m["beta"]] {
                assert!(v.to_bits() == x.to_bits() || (v.is_nan() && x.is_nan()),
                    "{s} must read back as itself, got {v}");
            }
        }
    }

    /// `null` cannot be read as a number: it stood for `+∞`, `−∞` and `NaN`
    /// alike. In an `Option` it is `None`, as it always was.
    #[test]
    fn null_is_refused_as_a_number_and_is_none_as_an_option() {
        let e = serde_json::from_str::<Holder>(r#"{"x":null,"m":{}}"#).unwrap_err();
        assert!(e.to_string().contains("written before"), "{e}");
        let h: Holder = serde_json::from_str(r#"{"x":1,"o":null,"m":{}}"#).unwrap();
        assert_eq!((h.x, h.o), (1.0, None), "an integer token is a number; null is None");
        let h: Holder = serde_json::from_str(r#"{"x":1.5,"m":{}}"#).unwrap();
        assert_eq!(h.o, None, "a missing optional key is None");
    }

    /// Only the three spellings this writes are numbers. Any other string is
    /// refused, rather than parsed — a finite number written as a string is
    /// not something camdl produces, and reading one would hide the writer
    /// that did.
    #[test]
    fn other_strings_are_refused() {
        for s in ["1.5", "infinity", "nan", "Inf", ""] {
            let json = format!(r#"{{"x":"{s}","m":{{}}}}"#);
            assert!(serde_json::from_str::<Holder>(&json).is_err(), "{s:?} must be refused");
        }
    }
}
