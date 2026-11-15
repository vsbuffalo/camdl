//! Comparing two declarations in their document form.
//!
//! A declaration is compared as the JSON document the `declared` crate writes
//! for it, with the members that exist only for display left out. That form is
//! the published one, every type in it refuses unknown fields, and a member
//! added to a declaration later is compared without any change here: leaving a
//! member out of the comparison takes an entry in [`DISPLAY_ONLY`], so the
//! default for a new member is that it counts.
//!
//! On an expression this is the proposal's rule, "two declared expressions are
//! the same when they parse to the same tree": of a `DeclaredExpr`'s four
//! members only `tree` is compared.

use serde_json::Value;

use declared::KatexLoss;

use crate::{FieldChange, FieldValue};

/// Members of a declaration that say nothing about the model: an expression's
/// rendering, the reasons the rendering is incomplete, and the list of names
/// read (all three derived from its `tree`); a declaration's text as written;
/// and the expanded names it produced.
///
/// `tests/golden_documents.rs` checks, on every compiled golden, that these
/// names occur only where the list above says they do.
pub const DISPLAY_ONLY: [&str; 5] = ["katex", "katex_loss", "reads", "source_text", "expands_to"];

fn is_display_only(key: &str) -> bool {
    DISPLAY_ONLY.contains(&key)
}

/// Whether two documents say the same thing: equal once the display-only
/// members are left out.
pub fn same(left: &Value, right: &Value) -> bool {
    match (left, right) {
        (Value::Object(l), Value::Object(r)) => {
            let mut l_members = l.iter().filter(|(k, _)| !is_display_only(k));
            let mut r_members = r.iter().filter(|(k, _)| !is_display_only(k));
            loop {
                match (l_members.next(), r_members.next()) {
                    (None, None) => return true,
                    (Some((lk, lv)), Some((rk, rv))) if lk == rk && same(lv, rv) => {}
                    _ => return false,
                }
            }
        }
        (Value::Array(l), Value::Array(r)) => {
            l.len() == r.len() && l.iter().zip(r).all(|(a, b)| same(a, b))
        }
        _ => left == right,
    }
}

/// A copy of a document without its display-only members.
fn without_display(value: &Value) -> Value {
    match value {
        Value::Object(members) => Value::Object(
            members
                .iter()
                .filter(|(k, _)| !is_display_only(k))
                .map(|(k, v)| (k.clone(), without_display(v)))
                .collect(),
        ),
        Value::Array(items) => Value::Array(items.iter().map(without_display).collect()),
        other => other.clone(),
    }
}

/// A name in an expression, as written in the document: the object
/// `{"name": <string>, "kind": <string>}` and nothing else.
fn as_name_ref(value: &Value) -> Option<(String, String)> {
    let members = value.as_object()?;
    if members.len() != 2 {
        return None;
    }
    Some((members.get("name")?.as_str()?.to_string(), members.get("kind")?.as_str()?.to_string()))
}

/// Every name a declaration reads, with the kind the compiler resolved it to,
/// once each, in order of first appearance. This walks every tree in the
/// declaration, including those (a prior's arguments, a staging law's) that
/// carry no `reads` list of their own.
pub fn names(value: &Value) -> Vec<(String, String)> {
    fn walk(value: &Value, out: &mut Vec<(String, String)>) {
        if let Some(name) = as_name_ref(value) {
            if !out.contains(&name) {
                out.push(name);
            }
            return;
        }
        match value {
            Value::Object(members) => members.values().for_each(|v| walk(v, out)),
            Value::Array(items) => items.iter().for_each(|v| walk(v, out)),
            _ => {}
        }
    }
    let mut out = Vec::new();
    walk(value, &mut out);
    out
}

/// The variant tags of `declared::ExprTree`. An object with one of these as
/// its only member is treated as an expression and reported whole.
///
/// This decides only how finely a change is located, never whether there is
/// one: `Destination::Sum` shares the tag `sum`, and is then reported whole
/// where it could have been reported arm by arm.
const EXPR_TAGS: [&str; 14] = [
    "const", "unit", "ident", "index", "bin_op", "un_op", "sum", "cond", "call", "list", "range",
    "obs_access", "run_member", "str",
];

fn is_declared_expr(value: &Value) -> bool {
    value.as_object().is_some_and(|m| m.contains_key("tree") && m.contains_key("katex"))
}

fn is_bare_tree(value: &Value) -> bool {
    value.as_object().is_some_and(|m| {
        m.len() == 1 && m.keys().next().is_some_and(|k| EXPR_TAGS.contains(&k.as_str()))
    })
}

fn field_value(value: &Value) -> FieldValue {
    if value.is_null() {
        return FieldValue::Absent;
    }
    if is_declared_expr(value) {
        let katex = value.get("katex").and_then(Value::as_str).unwrap_or_default().to_string();
        let katex_loss: Vec<KatexLoss> = value
            .get("katex_loss")
            .and_then(|l| serde_json::from_value(l.clone()).ok())
            .unwrap_or_default();
        return FieldValue::Expr { katex, katex_loss };
    }
    FieldValue::Other { value: without_display(value) }
}

/// Where two declarations differ, as the paths of the members that differ.
///
/// The walk descends while both sides have the same shape (objects with the
/// same members, lists of the same length) and stops at an expression, which is
/// reported whole. Empty exactly when [`same`] holds.
pub fn changes(left: &Value, right: &Value) -> Vec<FieldChange<FieldValue>> {
    fn walk(path: &str, left: &Value, right: &Value, out: &mut Vec<FieldChange<FieldValue>>) {
        if same(left, right) {
            return;
        }
        let leaf = |v: &Value| is_declared_expr(v) || is_bare_tree(v);
        if !leaf(left) && !leaf(right) {
            match (left, right) {
                (Value::Object(l), Value::Object(r)) => {
                    let keys = |m: &serde_json::Map<String, Value>| -> Vec<String> {
                        m.keys().filter(|k| !is_display_only(k)).cloned().collect()
                    };
                    let l_keys = keys(l);
                    if l_keys == keys(r) {
                        for key in &l_keys {
                            let child = if path.is_empty() { key.clone() } else { format!("{path}.{key}") };
                            walk(&child, &l[key], &r[key], out);
                        }
                        return;
                    }
                }
                (Value::Array(l), Value::Array(r)) if l.len() == r.len() => {
                    for (i, (a, b)) in l.iter().zip(r).enumerate() {
                        walk(&format!("{path}[{i}]"), a, b, out);
                    }
                    return;
                }
                _ => {}
            }
        }
        out.push(FieldChange {
            field: path.to_string(),
            left: field_value(left),
            right: field_value(right),
        });
    }
    let mut out = Vec::new();
    walk("", left, right, &mut out);
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn display_members_do_not_count_and_every_other_member_does() {
        let a = json!({"tree": {"const": {"value": 1.0}}, "katex": "1", "katex_loss": [], "reads": []});
        let b = json!({"tree": {"const": {"value": 1.0}}, "katex": "one", "katex_loss": ["number_rounded"], "reads": []});
        let c = json!({"tree": {"const": {"value": 2.0}}, "katex": "1", "katex_loss": [], "reads": []});
        assert!(same(&a, &b));
        assert!(!same(&a, &c));
        assert!(changes(&a, &b).is_empty());
        assert_eq!(changes(&a, &c).len(), 1);
    }

    #[test]
    fn a_member_on_one_side_only_is_a_difference() {
        assert!(!same(&json!({"a": 1}), &json!({"a": 1, "b": 2})));
        assert!(!same(&json!({"a": 1, "b": 2}), &json!({"a": 1})));
        assert!(same(&json!({"a": 1, "source_text": "x"}), &json!({"a": 1})));
    }

    #[test]
    fn a_list_that_only_extends_another_is_a_difference() {
        let short = json!({"levels": ["child", "adult"]});
        let long = json!({"levels": ["child", "adult", "elder"]});
        assert!(!same(&short, &long));
        assert!(!same(&long, &short));
        let found = changes(&short, &long);
        assert_eq!(found.len(), 1);
        assert_eq!(found[0].field, "levels");
    }

    #[test]
    fn a_change_is_located_by_its_path_and_an_expression_is_reported_whole() {
        let expr = |v: f64| json!({"tree": {"const": {"value": v}}, "katex": v.to_string(), "katex_loss": [], "reads": []});
        let a = json!({"name": "r", "dynamics": {"rate": {"rate": expr(1.0)}}, "from": [{"compartment": "S"}]});
        let b = json!({"name": "r", "dynamics": {"rate": {"rate": expr(2.0)}}, "from": [{"compartment": "E"}]});
        let found = changes(&a, &b);
        let paths: Vec<&str> = found.iter().map(|c| c.field.as_str()).collect();
        assert_eq!(paths, ["dynamics.rate.rate", "from[0].compartment"]);
        assert_eq!(found[0].left, FieldValue::Expr { katex: "1".into(), katex_loss: vec![] });
    }

    #[test]
    fn names_are_read_from_every_tree_once_each_in_order() {
        let doc = json!({
            "args": [{"name": "mean", "value": {"ident": {"name": {"name": "mu", "kind": "parameter"}}}}],
            "body": {"tree": {"bin_op": {"op": "mul",
                "lhs": {"ident": {"name": {"name": "N", "kind": "definition"}}},
                "rhs": {"ident": {"name": {"name": "mu", "kind": "parameter"}}}}},
              "katex": "", "katex_loss": [], "reads": [{"name": "N", "kind": "definition"}]}
        });
        assert_eq!(
            names(&doc),
            [("mu".to_string(), "parameter".to_string()), ("N".to_string(), "definition".to_string())]
        );
    }
}
