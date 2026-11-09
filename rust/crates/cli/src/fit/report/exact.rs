//! A report as JSON in which every number is kept exactly, including the
//! non-finite ones.
//!
//! `serde_json` writes NaN, `+∞` and `−∞` all as `null`, so two reports that
//! differ only in which of the three a field holds serialize equal. The
//! run-equals-loader check (proposal `docs/dev/proposals/2026-11-08-cli-reporting.md`,
//! "The report") compares the report `fit run` built in memory with the one
//! `fit summary` builds from the leaf, and must not read a NaN as an infinity.
//! [`exact_json`] serializes as `serde_json::to_value` does, except that a
//! non-finite float becomes the string `"NaN"`, `"inf"` or `"-inf"`, the
//! spelling `sim::inference::json_f64` already writes.
//!
//! [`capture`] is how the comparison sees both reports: when
//! `CAMDL_REPORT_CAPTURE` names a file, `fit run` and `fit summary` each append
//! the report they built to it, one JSON line per method. Nothing else reads
//! the variable, and it changes nothing a run computes or writes.

use serde::ser::{self, Serialize};
use serde_json::{Map, Value};

/// The environment variable naming the file [`capture`] appends to.
pub const CAPTURE_ENV: &str = "CAMDL_REPORT_CAPTURE";

/// `value` as JSON, with non-finite floats spelled as strings.
pub fn exact_json<T: Serialize + ?Sized>(value: &T) -> Result<Value, serde_json::Error> {
    value.serialize(Exact)
}

/// Append `report` to the file `CAMDL_REPORT_CAPTURE` names, as one line
/// `{"by": by, "leaf": …, "report": …}`; nothing when the variable is unset. A
/// capture that cannot be written is said on stderr and does not stop the
/// command: it is a check's input, not the command's output.
pub fn capture(by: &str, report: &super::build::MethodReport) {
    use std::io::Write;
    let Some(path) = std::env::var_os(CAPTURE_ENV) else { return };
    let line = exact_json(report).map(|r| {
        serde_json::json!({
            "by": by,
            "leaf": report.stored.as_ref().map(|s| s.leaf.clone()),
            "report": r,
        })
    });
    let written = line
        .map_err(|e| e.to_string())
        .and_then(|line| {
            std::fs::OpenOptions::new().create(true).append(true).open(&path)
                .and_then(|mut f| writeln!(f, "{line}"))
                .map_err(|e| e.to_string())
        });
    if let Err(e) = written {
        eprintln!("warning: {CAPTURE_ENV}: cannot record the report in {}: {e}",
            std::path::Path::new(&path).display());
    }
}

fn float(v: f64) -> Value {
    if v.is_nan() {
        Value::from("NaN")
    } else if v == f64::INFINITY {
        Value::from("inf")
    } else if v == f64::NEG_INFINITY {
        Value::from("-inf")
    } else {
        Value::from(v)
    }
}

/// The serializer: `serde_json`'s value serializer with exact floats.
struct Exact;

type Error = serde_json::Error;

impl ser::Serializer for Exact {
    type Ok = Value;
    type Error = Error;
    type SerializeSeq = Seq;
    type SerializeTuple = Seq;
    type SerializeTupleStruct = Seq;
    type SerializeTupleVariant = Variant<Seq>;
    type SerializeMap = Obj;
    type SerializeStruct = Obj;
    type SerializeStructVariant = Variant<Obj>;

    fn serialize_bool(self, v: bool) -> Result<Value, Error> { Ok(Value::Bool(v)) }
    fn serialize_i8(self, v: i8) -> Result<Value, Error> { Ok(Value::from(v)) }
    fn serialize_i16(self, v: i16) -> Result<Value, Error> { Ok(Value::from(v)) }
    fn serialize_i32(self, v: i32) -> Result<Value, Error> { Ok(Value::from(v)) }
    fn serialize_i64(self, v: i64) -> Result<Value, Error> { Ok(Value::from(v)) }
    fn serialize_u8(self, v: u8) -> Result<Value, Error> { Ok(Value::from(v)) }
    fn serialize_u16(self, v: u16) -> Result<Value, Error> { Ok(Value::from(v)) }
    fn serialize_u32(self, v: u32) -> Result<Value, Error> { Ok(Value::from(v)) }
    fn serialize_u64(self, v: u64) -> Result<Value, Error> { Ok(Value::from(v)) }
    fn serialize_f32(self, v: f32) -> Result<Value, Error> { Ok(float(v as f64)) }
    fn serialize_f64(self, v: f64) -> Result<Value, Error> { Ok(float(v)) }
    fn serialize_char(self, v: char) -> Result<Value, Error> { Ok(Value::from(v.to_string())) }
    fn serialize_str(self, v: &str) -> Result<Value, Error> { Ok(Value::from(v)) }
    fn serialize_bytes(self, v: &[u8]) -> Result<Value, Error> {
        Ok(Value::Array(v.iter().map(|b| Value::from(*b)).collect()))
    }
    fn serialize_none(self) -> Result<Value, Error> { Ok(Value::Null) }
    fn serialize_some<T: Serialize + ?Sized>(self, v: &T) -> Result<Value, Error> { v.serialize(self) }
    fn serialize_unit(self) -> Result<Value, Error> { Ok(Value::Null) }
    fn serialize_unit_struct(self, _: &'static str) -> Result<Value, Error> { Ok(Value::Null) }
    fn serialize_unit_variant(self, _: &'static str, _: u32, variant: &'static str) -> Result<Value, Error> {
        Ok(Value::from(variant))
    }
    fn serialize_newtype_struct<T: Serialize + ?Sized>(self, _: &'static str, v: &T) -> Result<Value, Error> {
        v.serialize(self)
    }
    fn serialize_newtype_variant<T: Serialize + ?Sized>(
        self, _: &'static str, _: u32, variant: &'static str, v: &T,
    ) -> Result<Value, Error> {
        let mut m = Map::new();
        m.insert(variant.to_string(), v.serialize(Exact)?);
        Ok(Value::Object(m))
    }
    fn serialize_seq(self, len: Option<usize>) -> Result<Seq, Error> {
        Ok(Seq(Vec::with_capacity(len.unwrap_or(0))))
    }
    fn serialize_tuple(self, len: usize) -> Result<Seq, Error> { self.serialize_seq(Some(len)) }
    fn serialize_tuple_struct(self, _: &'static str, len: usize) -> Result<Seq, Error> {
        self.serialize_seq(Some(len))
    }
    fn serialize_tuple_variant(
        self, _: &'static str, _: u32, variant: &'static str, len: usize,
    ) -> Result<Variant<Seq>, Error> {
        Ok(Variant { name: variant, inner: Seq(Vec::with_capacity(len)) })
    }
    fn serialize_map(self, _: Option<usize>) -> Result<Obj, Error> {
        Ok(Obj { map: Map::new(), key: None })
    }
    fn serialize_struct(self, _: &'static str, _: usize) -> Result<Obj, Error> {
        self.serialize_map(None)
    }
    fn serialize_struct_variant(
        self, _: &'static str, _: u32, variant: &'static str, _: usize,
    ) -> Result<Variant<Obj>, Error> {
        Ok(Variant { name: variant, inner: Obj { map: Map::new(), key: None } })
    }
}

struct Seq(Vec<Value>);

impl Seq {
    fn push<T: Serialize + ?Sized>(&mut self, v: &T) -> Result<(), Error> {
        self.0.push(v.serialize(Exact)?);
        Ok(())
    }
}

impl ser::SerializeSeq for Seq {
    type Ok = Value;
    type Error = Error;
    fn serialize_element<T: Serialize + ?Sized>(&mut self, v: &T) -> Result<(), Error> { self.push(v) }
    fn end(self) -> Result<Value, Error> { Ok(Value::Array(self.0)) }
}

impl ser::SerializeTuple for Seq {
    type Ok = Value;
    type Error = Error;
    fn serialize_element<T: Serialize + ?Sized>(&mut self, v: &T) -> Result<(), Error> { self.push(v) }
    fn end(self) -> Result<Value, Error> { Ok(Value::Array(self.0)) }
}

impl ser::SerializeTupleStruct for Seq {
    type Ok = Value;
    type Error = Error;
    fn serialize_field<T: Serialize + ?Sized>(&mut self, v: &T) -> Result<(), Error> { self.push(v) }
    fn end(self) -> Result<Value, Error> { Ok(Value::Array(self.0)) }
}

struct Obj {
    map: Map<String, Value>,
    key: Option<String>,
}

/// A map key as `serde_json` writes one: a string as itself, a number or a
/// bool as its text.
fn key_text(v: Value) -> Result<String, Error> {
    match v {
        Value::String(s) => Ok(s),
        Value::Number(n) => Ok(n.to_string()),
        Value::Bool(b) => Ok(b.to_string()),
        other => Err(ser::Error::custom(format!("a map key must be a string or a number, not {other}"))),
    }
}

impl ser::SerializeMap for Obj {
    type Ok = Value;
    type Error = Error;
    fn serialize_key<T: Serialize + ?Sized>(&mut self, k: &T) -> Result<(), Error> {
        self.key = Some(key_text(k.serialize(Exact)?)?);
        Ok(())
    }
    fn serialize_value<T: Serialize + ?Sized>(&mut self, v: &T) -> Result<(), Error> {
        let k = self.key.take().ok_or_else(|| <Error as ser::Error>::custom("a map value with no key"))?;
        self.map.insert(k, v.serialize(Exact)?);
        Ok(())
    }
    fn end(self) -> Result<Value, Error> { Ok(Value::Object(self.map)) }
}

impl ser::SerializeStruct for Obj {
    type Ok = Value;
    type Error = Error;
    fn serialize_field<T: Serialize + ?Sized>(&mut self, k: &'static str, v: &T) -> Result<(), Error> {
        self.map.insert(k.to_string(), v.serialize(Exact)?);
        Ok(())
    }
    fn end(self) -> Result<Value, Error> { Ok(Value::Object(self.map)) }
}

/// A tuple or struct variant: its fields under the variant's name.
struct Variant<T> {
    name: &'static str,
    inner: T,
}

impl<T> Variant<T> {
    fn wrap(name: &'static str, v: Value) -> Value {
        let mut m = Map::new();
        m.insert(name.to_string(), v);
        Value::Object(m)
    }
}

impl ser::SerializeTupleVariant for Variant<Seq> {
    type Ok = Value;
    type Error = Error;
    fn serialize_field<T: Serialize + ?Sized>(&mut self, v: &T) -> Result<(), Error> { self.inner.push(v) }
    fn end(self) -> Result<Value, Error> { Ok(Self::wrap(self.name, Value::Array(self.inner.0))) }
}

impl ser::SerializeStructVariant for Variant<Obj> {
    type Ok = Value;
    type Error = Error;
    fn serialize_field<T: Serialize + ?Sized>(&mut self, k: &'static str, v: &T) -> Result<(), Error> {
        ser::SerializeStruct::serialize_field(&mut self.inner, k, v)
    }
    fn end(self) -> Result<Value, Error> { Ok(Self::wrap(self.name, Value::Object(self.inner.map))) }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeMap;

    #[derive(serde::Serialize)]
    #[serde(tag = "kind", rename_all = "snake_case")]
    enum Tagged {
        Point { x: f64 },
    }

    #[derive(serde::Serialize)]
    enum Outer {
        Unit,
        New(f64),
        Pair(f64, f64),
        Rec { y: f64 },
    }

    /// The three non-finite values are told apart; everything finite reads
    /// as `serde_json::to_value` writes it.
    #[test]
    fn non_finite_floats_are_told_apart_and_the_rest_is_serde_json() {
        let v = exact_json(&[f64::NAN, f64::INFINITY, f64::NEG_INFINITY, 1.5]).unwrap();
        assert_eq!(v, serde_json::json!(["NaN", "inf", "-inf", 1.5]));
        assert_ne!(exact_json(&f64::NAN).unwrap(), exact_json(&f64::INFINITY).unwrap());

        let mut by_chain = BTreeMap::new();
        by_chain.insert(3usize, 0.25);
        let finite = (
            by_chain,
            Tagged::Point { x: 2.0 },
            [Outer::Unit, Outer::New(1.0), Outer::Pair(1.0, 2.0), Outer::Rec { y: 3.0 }],
            Some("s"),
            None::<u8>,
        );
        assert_eq!(exact_json(&finite).unwrap(), serde_json::to_value(&finite).unwrap());
    }
}
