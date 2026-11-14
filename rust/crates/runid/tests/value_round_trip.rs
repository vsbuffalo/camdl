//! A compiled model must never pass through a generic JSON value.
//!
//! Source: `ocaml/golden/init_dependency_order.ir.json`, a compiled golden
//! (primary), whose initial conditions are declared in the order D, S, E, I, R.
//!
//! In this workspace `serde_json::Value` keeps an object's keys sorted (the
//! `preserve_order` feature is off). The order of a model's
//! `initial_conditions` is part of its identity, so reading a compiled model
//! into a `Value` and writing it back yields a different model. The first test
//! pins that as a fact, so nobody has to rediscover it; the second pins that
//! the typed reader and writer do preserve the order and the identity.
//!
//! The rule that follows, for a bundle, a diff, `publish` and anything else
//! that moves a compiled model: copy it as bytes with a digest, and never
//! re-serialize it.

use std::path::Path;

use runid::inputs::model_ir_hash;

const GOLDEN: &str = "../../../ocaml/golden/init_dependency_order.ir.json";

fn golden_text() -> String {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join(GOLDEN);
    std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()))
}

fn initial_condition_order(model: &ir::Model) -> Vec<String> {
    model.initial_conditions.0.keys().cloned().collect()
}

#[test]
fn a_generic_value_round_trip_changes_the_model() {
    let text = golden_text();
    let original = ir::from_str(&text).expect("the golden parses");
    let declared_order = initial_condition_order(&original);

    // The fixture has to be able to show the hazard: three or more initial
    // conditions, declared in an order that is not the sorted one.
    let mut sorted = declared_order.clone();
    sorted.sort();
    assert!(declared_order.len() >= 3, "fixture needs at least three initial conditions");
    assert_ne!(declared_order, sorted, "fixture must declare its initial conditions out of sorted order");

    let through_value: serde_json::Value = serde_json::from_str(&text).expect("the golden is JSON");
    let rewritten = serde_json::to_string(&through_value).expect("serialize");
    let reread = ir::from_str(&rewritten).expect("the rewritten text still parses as a model");

    assert_eq!(
        initial_condition_order(&reread),
        sorted,
        "a generic value writes the initial conditions back in sorted order"
    );
    assert_ne!(
        model_ir_hash(&reread),
        model_ir_hash(&original),
        "and that is a different model: its identity changed"
    );
}

#[test]
fn the_typed_round_trip_preserves_order_and_identity() {
    let text = golden_text();
    let original = ir::envelope_from_str(&text).expect("the golden parses");

    let rewritten = serde_json::to_string(&original).expect("serialize the envelope");
    let reread = ir::envelope_from_str(&rewritten).expect("the rewritten envelope parses");

    assert_eq!(initial_condition_order(&reread.model), initial_condition_order(&original.model));
    assert_eq!(model_ir_hash(&reread.model), model_ir_hash(&original.model));
    assert_eq!(reread.model, original.model);
    assert_eq!(reread.declared, original.declared, "the declared section survives too");

    // The path a command takes when it changes a model and writes it back.
    let reemitted = ir::reemit_with_model(&text, &original.model).expect("re-emit");
    let again = ir::envelope_from_str(&reemitted).expect("the re-emitted envelope parses");
    assert_eq!(model_ir_hash(&again.model), model_ir_hash(&original.model));
}
