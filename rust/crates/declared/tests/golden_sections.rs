//! The `declared` section of every compiled golden model, read through this
//! crate alone.
//!
//! Source: the committed `*.ir.json` goldens under `ocaml/golden` and
//! `tests/fixtures/*/ir` (primary: what the compiler emits). This test never
//! uses the `ir` crate: it takes the `declared` member out of the raw JSON, so
//! it also shows the section is readable without the runtime's reader for
//! compiled models, whatever `ir_version` the envelope carries.
//!
//! Checked on each section: it deserializes through the typed model (every
//! type refuses unknown fields, so this is a shape check); it serializes back
//! to the same JSON; no name is unresolved; and each expression's `reads` is
//! exactly the names in its tree. The ordered form of the last check, and the
//! check that each kind names something the section declares, are in
//! `ocaml/test/test_declared_emit.ml`.

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

use declared::{DeclaredModel, SCHEMA_VERSION};
use serde_json::Value;

fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../..")
}

fn golden_files() -> Vec<PathBuf> {
    let root = repo_root();
    let mut dirs = vec![root.join("ocaml/golden")];
    for entry in std::fs::read_dir(root.join("tests/fixtures")).expect("read tests/fixtures") {
        let ir = entry.expect("directory entry").path().join("ir");
        if ir.is_dir() {
            dirs.push(ir);
        }
    }
    let mut files: Vec<PathBuf> = dirs
        .iter()
        .flat_map(|d| std::fs::read_dir(d).unwrap_or_else(|e| panic!("read {}: {e}", d.display())))
        .map(|e| e.expect("directory entry").path())
        .filter(|p| p.to_string_lossy().ends_with(".ir.json"))
        .collect();
    files.sort();
    files
}

fn envelope(path: &Path) -> Value {
    let text = std::fs::read_to_string(path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
    serde_json::from_str(&text).unwrap_or_else(|e| panic!("{} is not JSON: {e}", path.display()))
}

/// A name reference is the object `{"name": <string>, "kind": <string>}`.
fn as_name_ref(v: &Value) -> Option<(String, String)> {
    let o = v.as_object()?;
    if o.len() != 2 {
        return None;
    }
    Some((o.get("name")?.as_str()?.to_string(), o.get("kind")?.as_str()?.to_string()))
}

fn names_in(v: &Value, out: &mut BTreeSet<(String, String)>) {
    if let Some(r) = as_name_ref(v) {
        out.insert(r);
        return;
    }
    match v {
        Value::Object(o) => o.values().for_each(|x| names_in(x, out)),
        Value::Array(a) => a.iter().for_each(|x| names_in(x, out)),
        _ => {}
    }
}

/// Visit every top-level expression `{tree, katex, katex_loss, reads}` under `v`.
fn for_each_expr(v: &Value, f: &mut dyn FnMut(&Value, &Value)) {
    match v {
        Value::Object(o) => {
            if o.len() == 4
                && o.contains_key("tree")
                && o.contains_key("katex")
                && o.contains_key("katex_loss")
                && o.contains_key("reads")
            {
                f(&o["tree"], &o["reads"]);
            }
            o.values().for_each(|x| for_each_expr(x, f));
        }
        Value::Array(a) => a.iter().for_each(|x| for_each_expr(x, f)),
        _ => {}
    }
}

#[test]
fn every_golden_section_reads_through_this_crate() {
    let files = golden_files();
    assert!(files.len() >= 90, "expected the whole golden set, found {}", files.len());
    let mut n_exprs = 0usize;
    for path in &files {
        let name = path.display();
        let env = envelope(path);
        let section = env.get("declared").unwrap_or_else(|| panic!("{name}: no `declared` section"));

        let model: DeclaredModel = serde_json::from_value(section.clone())
            .unwrap_or_else(|e| panic!("{name}: the declared section does not deserialize: {e}"));
        assert_eq!(model.schema_version.get(), SCHEMA_VERSION, "{name}: schema_version");
        let back = serde_json::to_value(&model).expect("serialize");
        assert_eq!(&back, section, "{name}: the section does not round-trip through the typed model");

        let mut all = BTreeSet::new();
        names_in(section, &mut all);
        if let Some((n, _)) = all.iter().find(|(_, kind)| kind == "unresolved") {
            panic!("{name}: the name '{n}' is unresolved");
        }

        for_each_expr(section, &mut |tree, reads| {
            n_exprs += 1;
            let mut in_tree = BTreeSet::new();
            names_in(tree, &mut in_tree);
            let listed: Vec<(String, String)> = reads
                .as_array()
                .unwrap_or_else(|| panic!("{name}: `reads` is not a list"))
                .iter()
                .map(|r| as_name_ref(r).unwrap_or_else(|| panic!("{name}: malformed `reads` entry {r}")))
                .collect();
            let as_set: BTreeSet<(String, String)> = listed.iter().cloned().collect();
            assert_eq!(listed.len(), as_set.len(), "{name}: `reads` repeats a name: {listed:?}");
            assert_eq!(as_set, in_tree, "{name}: `reads` is not the names of its tree");
        });

        // Every declaration records what it expanded to, and together the
        // lists are exactly the compiled model's names, each claimed once.
        // The compiled names are read from the envelope's JSON, not through
        // the runtime's reader.
        let compiled = |key: &str| -> Vec<String> {
            let mut names: Vec<String> = env["model"][key]
                .as_array()
                .unwrap_or_else(|| panic!("{name}: model.{key} is not a list"))
                .iter()
                .map(|x| x["name"].as_str().unwrap_or_else(|| panic!("{name}: unnamed {key}")).to_string())
                .collect();
            names.sort();
            names
        };
        let claimed = |what: &str, lists: Vec<&Option<Vec<String>>>| -> Vec<String> {
            let mut names: Vec<String> = lists
                .into_iter()
                .flat_map(|l| l.clone().unwrap_or_else(|| panic!("{name}: a {what} has no expands_to")))
                .collect();
            names.sort();
            names
        };
        assert_eq!(
            claimed("compartment", model.compartments.iter().map(|c| &c.expands_to).collect()),
            compiled("compartments"),
            "{name}: compartments"
        );
        assert_eq!(
            claimed("transition", model.transitions.iter().map(|t| &t.expands_to).collect()),
            compiled("transitions"),
            "{name}: transitions"
        );
        assert_eq!(
            claimed("observation stream", model.observations.iter().map(|o| &o.expands_to).collect()),
            compiled("observations"),
            "{name}: observations"
        );

        // A committed artifact holds no absolute path.
        for f in &model.compile_inputs.files {
            assert!(!f.path.starts_with('/'), "{name}: absolute path {}", f.path);
        }
    }
    assert!(n_exprs > 1000, "the walk found only {n_exprs} expressions");
}

/// The section does not depend on the envelope's IR version: a document taken
/// from an envelope stamped with a version no runtime reads still deserializes
/// here. (That the runtime's own reader refuses such an envelope is pinned in
/// `rust/crates/ir/tests/declared_section.rs`.)
#[test]
fn a_section_reads_from_an_envelope_of_another_ir_version() {
    let path = repo_root().join("ocaml/golden/sir_basic.ir.json");
    let mut env = envelope(&path);
    env["ir_version"] = Value::String("0.1".into());
    env["model"] = Value::String("not a model this build could read".into());
    let section = env.get("declared").expect("declared section").clone();
    let model: DeclaredModel = serde_json::from_value(section).expect("reads through the declared crate alone");
    assert_eq!(model.schema_version.get(), SCHEMA_VERSION);
    assert!(!model.compartments.is_empty());
}
