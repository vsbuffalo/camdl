//! Deserialise every `ir/golden/*.ir.json` model, round-trip it
//! (serialise → deserialise → structural equality), and validate it.
//!
//! Run with:  cd rust && cargo test --test golden_deser
//!
//! This guards the Rust serde over a corpus of hand-authored models (the same
//! files are used as input models by many integration tests): a serialise/
//! deserialise asymmetry — a field emitted one way and read another — surfaces
//! here as a round-trip inequality. The corpus carries no `bindings` field, so
//! `binding_bearing_model_round_trips` below adds a model that does (model-level
//! `bindings` + `BindingRef`), and
//! `expr.rs::roundtrips_every_variant_and_a_deep_nesting` covers the
//! `Reduce`/`BindingRef` Expr variants.

use std::fs;
use std::path::Path;

fn golden_dir() -> std::path::PathBuf {
    // Works whether run from `rust/` or from the repo root.
    let candidates = [
        Path::new("../ir/golden"),
        Path::new("ir/golden"),
    ];
    for c in &candidates {
        if c.is_dir() {
            return c.to_path_buf();
        }
    }
    panic!("cannot locate ir/golden directory (tried ../ir/golden and ir/golden)");
}

fn deser_golden(name: &str) {
    let dir = golden_dir();
    let path = dir.join(format!("{}.ir.json", name));
    let contents = fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("failed to read {}: {}", path.display(), e));

    let model = ir::from_str(&contents)
        .unwrap_or_else(|e| panic!("failed to deserialise {}: {}", name, e));

    // Round-trip: serialise and deserialise again; structural equality must hold.
    let json2 = ir::to_string_pretty(&model)
        .unwrap_or_else(|e| panic!("failed to serialise {}: {}", name, e));
    let model2 = ir::from_str(&json2)
        .unwrap_or_else(|e| panic!("round-trip deserialise failed for {}: {}", name, e));

    assert_eq!(model, model2, "round-trip equality failed for {}", name);

    // Basic sanity: version field
    assert_eq!(model.version, "0.3", "unexpected version in {}", name);

    // Run validation
    ir::validate::validate(&model)
        .unwrap_or_else(|errs| {
            let msgs: Vec<_> = errs.iter().map(|e| e.to_string()).collect();
            panic!("validation errors in {}:\n  {}", name, msgs.join("\n  "));
        });
}

#[test] fn golden_sir_basic()         { deser_golden("sir_basic"); }
#[test] fn golden_sir_demography()    { deser_golden("sir_demography"); }
#[test] fn golden_sir_vaccination()   { deser_golden("sir_vaccination"); }
#[test] fn golden_pure_death()        { deser_golden("pure_death"); }
#[test] fn golden_birth_death()       { deser_golden("birth_death"); }
#[test] fn golden_two_state()         { deser_golden("two_state"); }
#[test] fn golden_cholera_siwr()      { deser_golden("cholera_siwr"); }
#[test] fn golden_seir_age()          { deser_golden("seir_age"); }
#[test] fn golden_sir_placebo_ekrng() { deser_golden("sir_placebo_ekrng"); }
#[test] fn golden_sir_spatial_sum()   { deser_golden("sir_spatial_sum"); }
// First anchored fixture in the corpus: origin + add_calendar_months/years
// + date_range. Provides cross-language regression coverage for the typed-
// time surface in 2026-05-22-typed-time-and-dsl-ergonomics.md (Phase 1+2).
#[test] fn golden_sirv_anchored_calendar() { deser_golden("sirv_anchored_calendar"); }

/// Fix B deser coverage that `ir/golden/` (frozen v0.3, no bindings) cannot
/// provide: a binding-bearing model must survive a JSON round-trip with its
/// model-level `bindings` and the `BindingRef` nodes that read them intact.
/// Sourced from `ocaml/golden/` (v0.6) since that is where bindings exist.
#[test]
fn binding_bearing_model_round_trips() {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../ocaml/golden/sir_reservoir_mixed.ir.json");
    let contents = fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("failed to read {}: {}", path.display(), e));

    let model = ir::from_str(&contents)
        .unwrap_or_else(|e| panic!("failed to deserialise binding model: {}", e));

    // Non-vacuous guards: the fixture must actually exercise the binding paths,
    // else a future fixture swap could silently make this test prove nothing.
    assert!(
        !model.bindings.is_empty(),
        "fixture must carry model-level bindings"
    );

    let json2 = ir::to_string_pretty(&model).expect("serialise binding model");
    assert!(
        json2.contains("binding_ref"),
        "fixture must reference a binding from a rate (BindingRef), else the \
         reference path is untested"
    );

    let model2 = ir::from_str(&json2).expect("round-trip deserialise binding model");
    assert_eq!(model, model2, "binding-bearing model round-trip changed structure");
    assert!(!model2.bindings.is_empty(), "bindings dropped on round-trip");
}
