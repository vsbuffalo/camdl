//! Canonical JSON for values a stored identity holds as text: object keys
//! sorted at every depth, so a value's text does not depend on the order a
//! map was filled in, and a gate that refuses a non-finite float before
//! serde writes it as `null`.

/// `v` with the keys of every object, at every depth, in sorted order, so its
/// text does not depend on the order a map was filled in (nor on whether
/// `serde_json`'s `preserve_order` feature is on).
pub(crate) fn canonical(v: &serde_json::Value) -> serde_json::Value {
    match v {
        serde_json::Value::Object(m) => {
            let mut keys: Vec<&String> = m.keys().collect();
            keys.sort();
            let mut sorted = serde_json::Map::new();
            for k in keys {
                sorted.insert(k.clone(), canonical(&m[k]));
            }
            serde_json::Value::Object(sorted)
        }
        serde_json::Value::Array(a) => serde_json::Value::Array(a.iter().map(canonical).collect()),
        other => other.clone(),
    }
}

/// Serialize `value` whole, then remove each `exclude`d key: a top-level key
/// (`output_dir`), or a dotted path to a nested one (`data.tables`), which is
/// removed where the path exists and is nothing to remove where it does not
/// (a config with no `[data]`).
/// A table left empty by the removal is removed too. A field added to the
/// value's type is in the result unless it is named here, so an omission is
/// written down where a reviewer sees it. Finiteness is not checked here;
/// a caller that keys the result calls [`ensure_finite`] on `value` first.
pub(crate) fn serialize_minus<T: serde::Serialize>(
    value: &T,
    exclude: &[&str],
) -> Result<serde_json::Value, String> {
    let mut v = serde_json::to_value(value)
        .map_err(|e| format!("cannot serialize identity level for hashing: {e}"))?;
    if !exclude.is_empty() {
        match v {
            serde_json::Value::Object(ref mut m) => {
                for key in exclude {
                    let mut parts: Vec<&str> = key.split('.').collect();
                    let last = parts.pop().expect("split yields at least one part");
                    let mut obj = Some(&mut *m);
                    for p in parts {
                        obj = obj.and_then(|o| o.get_mut(p)).and_then(|v| v.as_object_mut());
                    }
                    if let Some(o) = obj {
                        o.remove(last);
                    }
                    // A table left empty by the subtraction goes too: a config
                    // whose `[simulate]` held only `from` must key like one
                    // with no `[simulate]` at all.
                    if let Some((parent, _)) = key.rsplit_once('.') {
                        let mut parts: Vec<&str> = parent.split('.').collect();
                        let leaf = parts.pop().expect("rsplit_once left a parent");
                        let mut obj = Some(&mut *m);
                        for p in parts {
                            obj = obj.and_then(|o| o.get_mut(p)).and_then(|v| v.as_object_mut());
                        }
                        if let Some(o) = obj {
                            if o.get(leaf).and_then(|v| v.as_object()).is_some_and(|t| t.is_empty()) {
                                o.remove(leaf);
                            }
                        }
                    }
                }
            }
            // `exclude` on a non-object is a programming error, not a data
            // one: the caller asked to subtract a key from something that has
            // none, so the subtraction silently did nothing.
            _ => return Err(
                "serialize_minus: `exclude` given for a value that does not \
                 serialize to a JSON object".to_string()),
        }
    }
    Ok(v)
}

// ── Non-finite finiteness gate ───────────────────────────────────────────────
//
// serde_json 1.0.150 serializes a non-finite f64/f32 (NaN / ±Inf) as `null`
// in BOTH `to_value` (→ `Value::Null`) and the text serializer (→ writes
// `null`) — neither errors. `inf`/`nan` are valid TOML floats, so two configs
// or methods differing only in a non-finite value would both collapse to the
// same `null` and hash equal: a silent collision. This generic gate rejects
// any non-finite float anywhere in a `Serialize` value *before* it reaches the
// nulling serializer, field-agnostically (a future float field is covered
// automatically) — the include-by-default posture applied to finiteness.

#[derive(Debug)]
struct NonFinite(String);

impl std::fmt::Display for NonFinite {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}
impl std::error::Error for NonFinite {}
impl serde::ser::Error for NonFinite {
    fn custom<T: std::fmt::Display>(msg: T) -> Self {
        NonFinite(msg.to_string())
    }
}

/// A `Serializer` that produces nothing and only fails on a non-finite float.
#[derive(Clone, Copy)]
struct FiniteCheck;

fn nonfinite(kind: &str) -> NonFinite {
    NonFinite(format!("non-finite {kind} (NaN/Inf) in a hashed fit input"))
}

impl serde::Serializer for FiniteCheck {
    type Ok = ();
    type Error = NonFinite;
    type SerializeSeq = Self;
    type SerializeTuple = Self;
    type SerializeTupleStruct = Self;
    type SerializeTupleVariant = Self;
    type SerializeMap = Self;
    type SerializeStruct = Self;
    type SerializeStructVariant = Self;

    fn serialize_f32(self, v: f32) -> Result<(), NonFinite> {
        if v.is_finite() { Ok(()) } else { Err(nonfinite("f32")) }
    }
    fn serialize_f64(self, v: f64) -> Result<(), NonFinite> {
        if v.is_finite() { Ok(()) } else { Err(nonfinite("f64")) }
    }

    fn serialize_bool(self, _: bool) -> Result<(), NonFinite> { Ok(()) }
    fn serialize_i8(self, _: i8) -> Result<(), NonFinite> { Ok(()) }
    fn serialize_i16(self, _: i16) -> Result<(), NonFinite> { Ok(()) }
    fn serialize_i32(self, _: i32) -> Result<(), NonFinite> { Ok(()) }
    fn serialize_i64(self, _: i64) -> Result<(), NonFinite> { Ok(()) }
    fn serialize_u8(self, _: u8) -> Result<(), NonFinite> { Ok(()) }
    fn serialize_u16(self, _: u16) -> Result<(), NonFinite> { Ok(()) }
    fn serialize_u32(self, _: u32) -> Result<(), NonFinite> { Ok(()) }
    fn serialize_u64(self, _: u64) -> Result<(), NonFinite> { Ok(()) }
    fn serialize_char(self, _: char) -> Result<(), NonFinite> { Ok(()) }
    fn serialize_str(self, _: &str) -> Result<(), NonFinite> { Ok(()) }
    fn serialize_bytes(self, _: &[u8]) -> Result<(), NonFinite> { Ok(()) }
    fn serialize_none(self) -> Result<(), NonFinite> { Ok(()) }
    fn serialize_some<T: serde::Serialize + ?Sized>(self, v: &T) -> Result<(), NonFinite> {
        v.serialize(self)
    }
    fn serialize_unit(self) -> Result<(), NonFinite> { Ok(()) }
    fn serialize_unit_struct(self, _: &'static str) -> Result<(), NonFinite> { Ok(()) }
    fn serialize_unit_variant(self, _: &'static str, _: u32, _: &'static str) -> Result<(), NonFinite> { Ok(()) }
    fn serialize_newtype_struct<T: serde::Serialize + ?Sized>(self, _: &'static str, v: &T) -> Result<(), NonFinite> {
        v.serialize(self)
    }
    fn serialize_newtype_variant<T: serde::Serialize + ?Sized>(self, _: &'static str, _: u32, _: &'static str, v: &T) -> Result<(), NonFinite> {
        v.serialize(self)
    }
    fn serialize_seq(self, _: Option<usize>) -> Result<Self, NonFinite> { Ok(self) }
    fn serialize_tuple(self, _: usize) -> Result<Self, NonFinite> { Ok(self) }
    fn serialize_tuple_struct(self, _: &'static str, _: usize) -> Result<Self, NonFinite> { Ok(self) }
    fn serialize_tuple_variant(self, _: &'static str, _: u32, _: &'static str, _: usize) -> Result<Self, NonFinite> { Ok(self) }
    fn serialize_map(self, _: Option<usize>) -> Result<Self, NonFinite> { Ok(self) }
    fn serialize_struct(self, _: &'static str, _: usize) -> Result<Self, NonFinite> { Ok(self) }
    fn serialize_struct_variant(self, _: &'static str, _: u32, _: &'static str, _: usize) -> Result<Self, NonFinite> { Ok(self) }
}

impl serde::ser::SerializeSeq for FiniteCheck {
    type Ok = ();
    type Error = NonFinite;
    fn serialize_element<T: serde::Serialize + ?Sized>(&mut self, v: &T) -> Result<(), NonFinite> { v.serialize(FiniteCheck) }
    fn end(self) -> Result<(), NonFinite> { Ok(()) }
}
impl serde::ser::SerializeTuple for FiniteCheck {
    type Ok = ();
    type Error = NonFinite;
    fn serialize_element<T: serde::Serialize + ?Sized>(&mut self, v: &T) -> Result<(), NonFinite> { v.serialize(FiniteCheck) }
    fn end(self) -> Result<(), NonFinite> { Ok(()) }
}
impl serde::ser::SerializeTupleStruct for FiniteCheck {
    type Ok = ();
    type Error = NonFinite;
    fn serialize_field<T: serde::Serialize + ?Sized>(&mut self, v: &T) -> Result<(), NonFinite> { v.serialize(FiniteCheck) }
    fn end(self) -> Result<(), NonFinite> { Ok(()) }
}
impl serde::ser::SerializeTupleVariant for FiniteCheck {
    type Ok = ();
    type Error = NonFinite;
    fn serialize_field<T: serde::Serialize + ?Sized>(&mut self, v: &T) -> Result<(), NonFinite> { v.serialize(FiniteCheck) }
    fn end(self) -> Result<(), NonFinite> { Ok(()) }
}
impl serde::ser::SerializeMap for FiniteCheck {
    type Ok = ();
    type Error = NonFinite;
    fn serialize_key<T: serde::Serialize + ?Sized>(&mut self, k: &T) -> Result<(), NonFinite> { k.serialize(FiniteCheck) }
    fn serialize_value<T: serde::Serialize + ?Sized>(&mut self, v: &T) -> Result<(), NonFinite> { v.serialize(FiniteCheck) }
    fn end(self) -> Result<(), NonFinite> { Ok(()) }
}
impl serde::ser::SerializeStruct for FiniteCheck {
    type Ok = ();
    type Error = NonFinite;
    fn serialize_field<T: serde::Serialize + ?Sized>(&mut self, _: &'static str, v: &T) -> Result<(), NonFinite> { v.serialize(FiniteCheck) }
    fn end(self) -> Result<(), NonFinite> { Ok(()) }
}
impl serde::ser::SerializeStructVariant for FiniteCheck {
    type Ok = ();
    type Error = NonFinite;
    fn serialize_field<T: serde::Serialize + ?Sized>(&mut self, _: &'static str, v: &T) -> Result<(), NonFinite> { v.serialize(FiniteCheck) }
    fn end(self) -> Result<(), NonFinite> { Ok(()) }
}

/// Reject any non-finite float anywhere in `v` — call before hashing, since
/// the JSON serializer would otherwise silently collapse it to `null`.
pub(crate) fn ensure_finite<T: serde::Serialize>(v: &T) -> Result<(), String> {
    v.serialize(FiniteCheck)
        .map_err(|e| format!("cannot hash fit input: {e}"))
}

