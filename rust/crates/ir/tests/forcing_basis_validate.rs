//! `ir::validate` and `forcing_basis` (proposal 2026-10-27-estimable-bsplines
//! §5.1): the node is legal only inside a gradient map, and there only naming a
//! spline forcing and one of its basis functions. Each case edits the
//! `bspline_uniform` golden (forcing `contact`, 6 basis functions; transition
//! `infection` reads it).

use serde_json::{json, Value};

fn golden() -> Value {
    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/../../../ocaml/golden/bspline_uniform.ir.json");
    serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap()
}

fn validate(doc: &Value) -> Result<(), Vec<String>> {
    let model = ir::from_str(&serde_json::to_string(doc).unwrap()).expect("parse");
    ir::validate::validate(&model).map_err(|es| es.iter().map(|e| e.to_string()).collect())
}

fn basis(name: &str, index: u32) -> Value {
    json!({"forcing_basis": {"name": name, "index": index}})
}

fn with_rate_grad(entry: Value) -> Value {
    let mut doc = golden();
    let tr = doc["model"]["transitions"].as_array_mut().unwrap().iter_mut()
        .find(|t| t["name"] == "infection").unwrap();
    tr["rate_grad"]["beta"] = json!({"grad": entry});
    doc
}

#[test]
fn a_valid_basis_in_a_rate_gradient_passes() {
    validate(&with_rate_grad(basis("contact", 5))).expect("basis 5 of a 6-function spline is valid");
}

#[test]
fn a_basis_outside_a_gradient_is_refused() {
    let mut doc = golden();
    doc["model"]["transitions"][0]["rate"] = basis("contact", 0);
    let errs = validate(&doc).unwrap_err();
    assert!(errs.iter().any(|e| e.contains("forcing_basis(contact, 0) appears outside a gradient map")), "{errs:?}");
}

#[test]
fn a_bad_target_is_refused() {
    for (entry, want) in [
        (basis("contact", 6), "basis 6 of forcing 'contact', which has 6 basis functions"),
        (basis("nowhere", 0), "names no forcing in this model"),
    ] {
        let errs = validate(&with_rate_grad(entry)).unwrap_err();
        assert!(errs.iter().any(|e| e.contains(want)), "expected {want:?}: {errs:?}");
    }
    // A non-spline forcing has no basis.
    let mut doc = with_rate_grad(basis("pulse", 0));
    doc["model"]["time_functions"].as_array_mut().unwrap().push(json!({
        "name": "pulse",
        "kind": {"piecewise": {"breakpoints": [{"const": 10.0}], "values": [{"const": 1.0}, {"const": 2.0}]}},
        "dim": [0, 0],
    }));
    let errs = validate(&doc).unwrap_err();
    assert!(errs.iter().any(|e| e.contains("which is not a spline and has no basis")), "{errs:?}");
}
