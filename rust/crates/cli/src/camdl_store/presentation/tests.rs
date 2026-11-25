//! The presentation a reader of a stored result is given, by [`shown_for`],
//! and the one `camdl path` and `camdl cat` resolve, over a store the release
//! binary writes: a fit run, a prediction from it, a profile point of it and
//! a simulation, each named by its label and by its run id.
//!
//! The binary is `rust/target/release/camdl`, built by `make build` from the
//! source this test was built from; it finds `camdlc` on `PATH`.

use std::path::{Path, PathBuf};
use std::process::Command;

use store::{Handle, Key, Store};

use super::*;
use crate::resolve_file::{resolve_file, Narrow, Source};

fn doc_model(doc: &str, fitted: bool) -> String {
    let observed = if fitted {
        r#"
observations {
  cases {
    columns       { time : time, cases : count }
    covers        = closing_at(time, 1 'days)
    projected     = incidence(infection)
    emit_schedule = every 1 'days
    cases ~ poisson(rate = projected)
  }
}"#
    } else {
        "scenarios { baseline { set = { beta = 0.3  gamma = 0.1  N0 = 1000  I0 = 5 } } }"
    };
    format!(
        r#"
time_unit = 'days
compartments {{ S, I, R }}
parameters {{
  #' {doc}
  beta  : rate         in [0.05, 1.0]  ~ log_normal(mu = -1.0, sigma = 0.5)
  gamma : rate         in [0.01, 0.5]  ~ log_normal(mu = -2.0, sigma = 0.5)
  N0    : count
  I0    : count
}}
let N = S + I + R
transitions {{
  infection : S --> I  @ beta * S * I / N
  recovery  : I --> R  @ gamma * I
}}
init {{ S = N0 - I0  I = I0 }}
{observed}
simulate {{ from = 0 'days  to = 20 'days }}
"#
    )
}

const FIT_TOML: &str = r#"
[model]
camdl = "model.camdl"
[data]
file = "cases.tsv"
[estimate]
beta = { bounds = [0.1, 0.9], start = 0.4 }
[fixed]
gamma = 0.2
N0 = 1000
I0 = 5
[method]
algorithm = "if2"
backend = "chain_binomial"
chains = 1
particles = 50
iterations = 2
cooling = 0.7
starts = "from_prior"
"#;

fn camdl(dir: &Path, args: &[&str]) {
    let bin = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../target/release/camdl");
    assert!(bin.exists(), "release camdl missing: {} - run `make build`", bin.display());
    let out = Command::new(bin)
        .current_dir(dir)
        .env_remove("CLICOLOR_FORCE")
        .env("CAMDL_SKIP_VERSION_CHECK", "1")
        .env("CAMDL_STORE", dir.join("st"))
        .args(args)
        .output()
        .expect("spawn camdl");
    assert!(out.status.success(), "camdl {args:?}: {}", String::from_utf8_lossy(&out.stderr));
}

/// A store holding a fit run `@a` of a file documenting `beta` as `fit`, a
/// prediction `@p` from it, a profile `@pr` of it run from a file documenting
/// `beta` as `profile`, and a simulation `@s` of a file documenting `beta`
/// as `sim`.
fn store_of_four() -> (tempfile::TempDir, Store) {
    let tmp = tempfile::tempdir().unwrap();
    let dir = tmp.path();
    std::fs::write(dir.join("model.camdl"), doc_model("fit", true)).unwrap();
    std::fs::write(dir.join("profiled.camdl"), doc_model("profile", true)).unwrap();
    std::fs::write(dir.join("sim.camdl"), doc_model("sim", false)).unwrap();
    std::fs::write(dir.join("fit.toml"), FIT_TOML).unwrap();
    let mut tsv = String::from("time\tcases\n");
    for (t, c) in [3, 6, 9, 14, 20, 27, 33, 38, 40, 39, 36, 31, 26, 21, 16, 12, 9, 7, 5, 4].iter().enumerate() {
        tsv.push_str(&format!("{}\t{c}\n", t + 1));
    }
    std::fs::write(dir.join("cases.tsv"), tsv).unwrap();
    camdl(dir, &["fit", "run", "fit.toml", "--seed", "1", "--label", "a", "--progress", "none"]);
    camdl(dir, &["predict", "@a", "--label", "p", "--progress", "none"]);
    camdl(
        dir,
        &[
            "profile", "profiled.camdl", "--fit", "@a", "--sweep", "beta=lin(0.3,0.6,2)", "--label", "pr", "--particles", "50",
            "--iterations", "2", "--seed", "1", "--progress", "none",
        ],
    );
    camdl(dir, &["simulate", "sim.camdl", "--scenario", "baseline", "--seed", "1", "--label", "s", "-o", "sim.tsv"]);
    let store = crate::camdl_store::open_existing(&dir.join("st")).expect("the store");
    (tmp, store)
}

/// The first result of kind `kind` the invocation `@label` listed.
fn result_of(store: &Store, label: &str, kind: &str) -> Key {
    let inv = store.invocation(&Handle::Label { name: label.into(), back: 0 }).unwrap();
    inv.entries().into_iter().find(|(k, _, _)| *k == kind).map(|(_, key, _)| key).unwrap_or_else(|| panic!("@{label} lists no {kind}"))
}

/// The presentation a reader of `result`, named by `handle`, is given, and
/// the one `camdl path HANDLE model_presentation:docs` resolves.
fn both(store: &Store, handle: &Handle, result: Key) -> (Option<Key>, Option<Key>) {
    let located = store.locate(result).unwrap();
    let model = model_of(store, &located).expect("the result's compiled model");
    let shown = match shown_for(store, handle, result, model) {
        Shown::Recorded { presentation, .. } => Some(presentation.id().key()),
        Shown::NotRecorded { .. } => None,
        Shown::Unreadable { run_id, why } => panic!("{handle}: {run_id} unreadable: {why}"),
    };
    let path = resolve_file(store, &Source::Handle(handle), &Narrow::default(), "model_presentation:docs").ok().map(|f| f.located.run_id());
    (shown, path)
}

/// The documentation of `beta` the presentation `key` holds.
fn beta_doc(store: &Store, key: Key) -> String {
    let presentation: Stored<ModelPresentation> = store.resolve(&RunRef::new(Handle::IdPrefix(key.to_hex()))).unwrap();
    let docs: PresentedDocs = serde_json::from_slice(&super::super::read_stored(&presentation, DOCS).unwrap()).unwrap();
    docs.docs.parameters["beta"].text.clone().unwrap()
}

/// A reader of a stored result and `camdl path` are given one presentation,
/// the one of the file the naming invocation ran: by label and by run id,
/// for a fit run, a prediction, a profile point and a simulation.
#[test]
fn a_results_presentation_is_the_one_camdl_path_resolves() {
    let (_tmp, store) = store_of_four();
    for (label, kind, doc) in [("a", "fit_run", "fit"), ("p", "prediction", "fit"), ("pr", "profile_point", "profile"), ("s", "sim", "sim")] {
        let result = result_of(&store, label, kind);
        for handle in [Handle::Label { name: label.into(), back: 0 }, Handle::IdPrefix(result.to_hex())] {
            let (shown, path) = both(&store, &handle, result);
            assert_eq!(shown, path, "{handle} ({kind}): the reader's presentation and camdl path's");
            assert_eq!(shown.map(|k| beta_doc(&store, k)).as_deref(), Some(doc), "{handle} ({kind})");
        }
    }
}

/// A prediction whose invocation record lists no compiled model, as a record
/// written before a prediction listed its model does, has no presentation
/// for either reader: no other invocation is read for it, although one
/// later recorded a presentation of the same compiled model.
#[test]
fn a_record_naming_no_presentation_gives_none_to_either_reader() {
    let (tmp, store) = store_of_four();
    let seq = store.invocation(&Handle::Label { name: "p".into(), back: 0 }).unwrap().seq();
    let path: PathBuf = tmp.path().join("st/.invocations").join(format!("{seq}.json"));
    let mut record: serde_json::Value = serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
    record["results"].as_array_mut().unwrap().retain(|r| r["kind"] != "compiled_model");
    std::fs::write(&path, serde_json::to_vec_pretty(&record).unwrap()).unwrap();
    // A store reads the records once, so the edited record is read by a store
    // opened after it, as by the next command.
    let store = crate::camdl_store::open_existing(&tmp.path().join("st")).expect("the store");

    let result = result_of(&store, "p", "prediction");
    for handle in [Handle::Label { name: "p".into(), back: 0 }, Handle::IdPrefix(result.to_hex())] {
        assert_eq!(both(&store, &handle, result), (None, None), "{handle}");
    }
}
