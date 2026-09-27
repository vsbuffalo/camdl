//! gh#938 sibling — a `#[lineage]` transition's parent-pool weight
//! expressions are name-checked when the model is built.
//!
//! The lineage event recorder (`lineage/event_log.rs`) evaluates each
//! parent-pool weight with the name-keyed `propensity::eval_expr` at the first
//! lineage event, so a wrong name used to surface mid-simulation, and only on a
//! run that records lineage. `CompiledModel::new` now refuses it, naming the
//! transition and the unknown name, as it does for rate expressions.

use std::path::PathBuf;

use ir::expr::Expr;
use sim::compiled_model::CompiledModel;

fn yule() -> ir::Model {
    let manifest = std::env::var("CARGO_MANIFEST_DIR").unwrap();
    let path = PathBuf::from(&manifest).join("tests/fixtures/yule_lineage.ir.json");
    let contents = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("could not read {path:?}: {e}"));
    let mut m: ir::Model =
        ir::from_str(&contents).unwrap_or_else(|e| panic!("failed to parse yule_lineage: {e}"));
    // The fixture leaves `lambda` without a value; give it one so the only
    // defect a test sees is the one it injects.
    for p in &mut m.parameters {
        if p.name == "lambda" {
            p.value = p.value.with_value(0.4);
        }
    }
    m
}

fn weights(m: &mut ir::Model) -> &mut Vec<(String, Expr)> {
    &mut m.transitions[0]
        .lineage
        .as_mut()
        .expect("yule_lineage's birth transition carries a lineage annotation")
        .parent_pool_weights
}

fn refusal(m: ir::Model) -> String {
    match CompiledModel::new(m) {
        Ok(_) => panic!("CompiledModel::new must refuse the model; it built"),
        Err(e) => {
            assert!(e.is_structural(), "a build refusal must be structural, got {e:?}");
            e.to_string()
        }
    }
}

#[test]
fn well_formed_lineage_fixture_builds() {
    CompiledModel::new(yule()).expect("the unmodified fixture must build");
}

#[test]
fn unknown_name_in_parent_pool_weight_is_refused_at_build() {
    let mut m = yule();
    weights(&mut m)[0].1 = Expr::param("ghost");
    let msg = refusal(m);
    for needle in ["ghost", "'birth'"] {
        assert!(msg.contains(needle), "refusal must name `{needle}`; got: {msg}");
    }
}

#[test]
fn unknown_parent_pool_compartment_is_refused_at_build() {
    let mut m = yule();
    weights(&mut m)[0].0 = "Ghost".into();
    let msg = refusal(m);
    for needle in ["Ghost", "'birth'"] {
        assert!(msg.contains(needle), "refusal must name `{needle}`; got: {msg}");
    }
}
