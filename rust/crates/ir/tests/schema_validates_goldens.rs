//! `ir/schema.json` is the written-down form of the IR's wire shape, but no
//! code reads it: the OCaml and Rust serde implementations are each written by
//! hand, so the schema drifts the moment a change to them is not mirrored here.
//! This test makes that drift a failure. It validates the `model` object of
//! every `ocaml/golden/*.ir.json` (what the compiler emits) and every
//! `ir/golden/*.ir.json` (the frozen Rust-side set) against the schema, and checks that the schema rejects shapes it does not declare, so a
//! passing run means the schema covers the goldens rather than accepting
//! anything.

use std::path::{Path, PathBuf};

use serde_json::Value;

fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../..")
}

fn compile_schema() -> (boon::Schemas, boon::SchemaIndex) {
    let path = repo_root().join("ir/schema.json");
    let text = std::fs::read_to_string(&path).expect("read ir/schema.json");
    let schema: Value = serde_json::from_str(&text).expect("ir/schema.json is not valid JSON");
    // Register the schema under its own `$id`, so its `#/definitions/…` refs
    // resolve without the compiler trying to fetch anything.
    let id = schema["$id"].as_str().expect("ir/schema.json has a string $id").to_owned();
    let mut compiler = boon::Compiler::new();
    compiler.add_resource(&id, schema).expect("register ir/schema.json");
    let mut schemas = boon::Schemas::new();
    let index = compiler
        .compile(&id, &mut schemas)
        .unwrap_or_else(|e| panic!("ir/schema.json does not compile as a JSON schema: {e:#}"));
    (schemas, index)
}

fn golden_models() -> Vec<(PathBuf, Value)> {
    let mut files: Vec<PathBuf> = ["ocaml/golden", "ir/golden"]
        .iter()
        .flat_map(|d| {
            std::fs::read_dir(repo_root().join(d))
                .unwrap_or_else(|e| panic!("read {d}: {e}"))
                .map(|e| e.expect("dir entry").path())
        })
        .filter(|p| p.to_string_lossy().ends_with(".ir.json"))
        .collect();
    files.sort();
    files
        .into_iter()
        .map(|p| {
            let text = std::fs::read_to_string(&p).expect("read golden");
            let mut env: Value = serde_json::from_str(&text).expect("golden is not valid JSON");
            let model = env
                .get_mut("model")
                .unwrap_or_else(|| panic!("{} has no `model` key", p.display()))
                .take();
            (p, model)
        })
        .collect()
}

#[test]
fn every_golden_validates_against_ir_schema() {
    let (schemas, index) = compile_schema();
    let goldens = golden_models();
    // Non-vacuous: the glob found the corpus.
    assert!(goldens.len() >= 60, "expected the ocaml/golden + ir/golden corpus, found {} files", goldens.len());

    let failures: Vec<String> = goldens
        .iter()
        .filter_map(|(path, model)| {
            schemas
                .validate(model, index)
                .err()
                .map(|e| format!("{}:\n{e:#}", path.display()))
        })
        .collect();
    assert!(
        failures.is_empty(),
        "{} of {} goldens do not match ir/schema.json — update the schema to the \
         current IR serde shape:\n\n{}",
        failures.len(),
        goldens.len(),
        failures.join("\n\n")
    );
}

/// The schema must reject what it does not declare. Without this, a schema
/// loosened to `{}` would pass the test above.
#[test]
fn ir_schema_rejects_undeclared_shapes() {
    let (schemas, index) = compile_schema();
    let (_, model) = golden_models()
        .into_iter()
        .find(|(_, m)| !m["time_functions"].as_array().unwrap().is_empty())
        .expect("a golden with a forcing");
    assert!(schemas.validate(&model, index).is_ok());

    // An unknown top-level field.
    let mut m = model.clone();
    m["not_an_ir_field"] = Value::Bool(true);
    assert!(schemas.validate(&m, index).is_err(), "unknown model field accepted");

    // An unknown forcing kind.
    let mut m = model.clone();
    m["time_functions"][0]["kind"] = serde_json::json!({ "wavelet": { "period": { "const": 1.0 } } });
    assert!(schemas.validate(&m, index).is_err(), "unknown time-function kind accepted");

    // An unknown expression node.
    let mut m = model.clone();
    m["transitions"][0]["rate"] = serde_json::json!({ "bogus": 1.0 });
    assert!(schemas.validate(&m, index).is_err(), "unknown expr node accepted");
}
