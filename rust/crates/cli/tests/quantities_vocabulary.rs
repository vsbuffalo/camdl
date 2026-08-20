//! `camdl simulate --quantities FILE`: a reporting vocabulary supplied at the
//! point of use, in place of the model's own `quantities {}` block.
//!
//! Proposal `docs/dev/proposals/2026-08-19-quantities-as-a-separable-layer.md`.
//! Four properties, each of which has a concrete failure mode if it does not
//! hold:
//!
//! 1. **Replacement, not merge.** The vocabulary's quantities are emitted and
//!    the model's own are not. A merge would mean the reporting table silently
//!    depended on which of two files declared a name first.
//! 2. **The vocabulary keys the artifact.** Two vocabularies over one run write
//!    two tables at two addresses; the same vocabulary twice writes one. Without
//!    the key the second run overwrites the first at one path and a reader
//!    cannot tell which formulas produced the numbers it is holding — the class
//!    fixed twice already (gh#626 `--to`, gh#641 `--init-state`).
//! 3. **An in-place edit re-keys.** The key is the file's BYTES, not its path.
//!    Correcting a formula and re-running must produce a new table, not a cache
//!    hit on the old one.
//! 4. **The run's identity does not move.** `quantities` are excluded from
//!    `Model::hash_into` (`runid`'s `ir_quantities_excluded_from_hash`), so the
//!    trajectory a vocabulary is read off is byte-identically the same run.

use std::path::{Path, PathBuf};
use std::process::Command;

fn binary() -> PathBuf {
    PathBuf::from(env!("CARGO_BIN_EXE_camdl"))
}

/// A closed SIR whose OWN `quantities {}` block declares `model_peak`. Any test
/// that supplies a vocabulary asserts `model_peak` is absent from the output —
/// that is what "the model's own block is not consulted" means operationally.
const MODEL: &str = r#"
time_unit = 'days
compartments { S, I, R }

parameters {
  beta  : positive in [0.01, 5.0]
  gamma : positive in [0.01, 1.0]
}

let N  = S + I + R
let R0 = beta / gamma

transitions {
  infection : S --> I @ beta * S * I / N
  recovery  : I --> R @ gamma * I
}

init { S = 990  I = 10  R = 0 }

quantities {
  model_peak = max(I / N)
}

simulate { from = 0  to = 70 }

scenarios {
  baseline {
    label = "base"
    set = { beta = 0.3  gamma = 0.1 }
  }
}
"#;

/// Vocabulary A. `R0` is a param-bearing `let`, which the compiler INLINES —
/// it appears nowhere in the compiled IR. Reporting it here pins that the
/// vocabulary is resolved against the model SOURCE's symbols: resolved against
/// the IR it would be rejected as an undeclared name.
const VOCAB_A: &str = "quantities {\n  attack_rate = final((1000 - S) / 1000)\n  r_zero      = final(R0)\n}\n";

/// Vocabulary B — a different reporting question over the same model.
const VOCAB_B: &str = "quantities {\n  peak_time = time_of_max(I)\n}\n";

fn run(args: &[&str]) -> std::process::Output {
    Command::new(binary())
        .args(args)
        .env("CAMDL_SKIP_VERSION_CHECK", "1")
        .output()
        .expect("spawn camdl")
}

struct Fixture {
    _tmp: tempfile::TempDir,
    root: PathBuf,
    model: PathBuf,
}

fn fixture() -> Fixture {
    let tmp = tempfile::tempdir().unwrap();
    let root = tmp.path().to_path_buf();
    let model = root.join("sir.camdl");
    std::fs::write(&model, MODEL).unwrap();
    Fixture { _tmp: tmp, root, model }
}

impl Fixture {
    fn vocab(&self, name: &str, body: &str) -> PathBuf {
        let p = self.root.join(name);
        std::fs::write(&p, body).unwrap();
        p
    }

    /// One `simulate` run. `vocab = None` uses the model's own block.
    fn simulate(&self, out: &Path, vocab: Option<&Path>, store: &Path) -> std::process::Output {
        let mut args: Vec<String> = vec![
            "simulate".into(),
            self.model.to_string_lossy().into_owned(),
            "--scenario".into(),
            "baseline".into(),
            "--backend".into(),
            "chain_binomial".into(),
            "--seed".into(),
            "7".into(),
            "--output-dir".into(),
            store.to_string_lossy().into_owned(),
            "--quantities-out".into(),
            out.to_string_lossy().into_owned(),
        ];
        if let Some(v) = vocab {
            args.push("--quantities".into());
            args.push(v.to_string_lossy().into_owned());
        }
        let refs: Vec<&str> = args.iter().map(|s| s.as_str()).collect();
        run(&refs)
    }
}

fn assert_ok(out: &std::process::Output, what: &str) {
    assert!(
        out.status.success(),
        "{what} failed:\nstdout={}\nstderr={}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr),
    );
}

/// Every `quantities*` directory and `quantities*.json` manifest under `dir`,
/// sorted — the *addresses* a run wrote its reporting tables to.
fn quantity_addresses(dir: &Path) -> Vec<String> {
    let mut names: Vec<String> = std::fs::read_dir(dir)
        .unwrap()
        .flatten()
        .map(|e| e.file_name().to_string_lossy().into_owned())
        .filter(|n| n.starts_with("quantities"))
        .collect();
    names.sort();
    names
}

/// The single sim leaf's `run_id` under a CAS store.
fn only_run_id(store: &Path) -> String {
    let mut found: Vec<PathBuf> = Vec::new();
    let mut stack = vec![store.to_path_buf()];
    while let Some(d) = stack.pop() {
        for e in std::fs::read_dir(&d).into_iter().flatten().flatten() {
            let p = e.path();
            if p.is_dir() {
                if p.join("run.json").exists() {
                    found.push(p.clone());
                }
                stack.push(p);
            }
        }
    }
    assert_eq!(found.len(), 1, "expected exactly one sim leaf under {}", store.display());
    let meta: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(found[0].join("run.json")).unwrap()).unwrap();
    meta["run_id"].as_str().unwrap().to_string()
}

// ── 1. Replacement, not merge ───────────────────────────────────────────────

#[test]
fn a_vocabulary_replaces_the_models_own_block() {
    let f = fixture();
    let va = f.vocab("va.camdl", VOCAB_A);
    let out = f.root.join("out");
    let store = f.root.join("store");
    assert_ok(&f.simulate(&out, Some(&va), &store), "simulate --quantities");

    let addrs = quantity_addresses(&out);
    let dir = addrs.iter().find(|n| !n.ends_with(".json")).expect("a quantities dir");
    let qdir = out.join(dir);
    assert!(qdir.join("attack_rate.tsv").exists(), "the vocabulary's quantity is emitted");
    assert!(
        qdir.join("r_zero.tsv").exists(),
        "a param-bearing `let` (R0) resolves — the vocabulary is compiled against \
         the model SOURCE, where inlined lets still exist"
    );
    assert!(
        !qdir.join("model_peak.tsv").exists(),
        "the model's own quantities block must NOT be consulted (replacement, not merge)"
    );
}

/// The control: with no `--quantities`, the model's own block is what runs, at
/// the historical `quantities/` address.
#[test]
fn without_a_vocabulary_the_models_own_block_runs_at_the_historical_address() {
    let f = fixture();
    let out = f.root.join("out");
    let store = f.root.join("store");
    assert_ok(&f.simulate(&out, None, &store), "simulate");
    assert_eq!(quantity_addresses(&out), vec!["quantities", "quantities.json"]);
    assert!(out.join("quantities").join("model_peak.tsv").exists());
}

// ── 2. The vocabulary keys the artifact ─────────────────────────────────────

#[test]
fn two_vocabularies_write_two_addresses_and_one_vocabulary_writes_one() {
    let f = fixture();
    let va = f.vocab("va.camdl", VOCAB_A);
    let vb = f.vocab("vb.camdl", VOCAB_B);
    let out = f.root.join("out");
    let store = f.root.join("store");

    assert_ok(&f.simulate(&out, Some(&va), &store), "simulate with A");
    let after_a = quantity_addresses(&out);
    assert_eq!(after_a.len(), 2, "one dir + one manifest: {after_a:?}");

    // The SAME vocabulary again must collide — a re-run is the same artifact.
    assert_ok(&f.simulate(&out, Some(&va), &store), "simulate with A again");
    assert_eq!(
        quantity_addresses(&out),
        after_a,
        "the same vocabulary twice is ONE address (a correct collision)"
    );

    // A DIFFERENT vocabulary must not.
    assert_ok(&f.simulate(&out, Some(&vb), &store), "simulate with B");
    let after_b = quantity_addresses(&out);
    assert_eq!(
        after_b.len(),
        4,
        "two vocabularies must occupy two addresses, not overwrite one: {after_b:?}"
    );

    // And each address holds ITS OWN vocabulary's table.
    let dir_a = after_a.iter().find(|n| !n.ends_with(".json")).unwrap();
    let dir_b = after_b
        .iter()
        .find(|n| !n.ends_with(".json") && *n != dir_a)
        .expect("B's own directory");
    assert!(out.join(dir_a).join("attack_rate.tsv").exists());
    assert!(out.join(dir_b).join("peak_time.tsv").exists());
    assert!(!out.join(dir_a).join("peak_time.tsv").exists());
}

/// The manifest records which vocabulary produced the table, by path and by
/// full digest — the 8-hex directory says two tables differ, this says how.
#[test]
fn the_manifest_records_the_vocabulary_that_produced_it() {
    let f = fixture();
    let va = f.vocab("va.camdl", VOCAB_A);
    let out = f.root.join("out");
    let store = f.root.join("store");
    assert_ok(&f.simulate(&out, Some(&va), &store), "simulate with A");

    let manifest_name = quantity_addresses(&out)
        .into_iter()
        .find(|n| n.ends_with(".json"))
        .expect("a manifest");
    let m: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(out.join(&manifest_name)).unwrap()).unwrap();
    let v = &m["vocabulary"];
    assert!(
        v["file"].as_str().unwrap().ends_with("va.camdl"),
        "manifest names the vocabulary file: {v}"
    );
    assert_eq!(v["sha256"].as_str().unwrap().len(), 64, "…and pins its full digest");
    assert!(
        manifest_name.contains(&v["sha256"].as_str().unwrap()[..8]),
        "the address is the digest's prefix: {manifest_name} vs {v}"
    );

    // The model's own block writes no `vocabulary` key — historical bytes.
    let out2 = f.root.join("out2");
    let store2 = f.root.join("store2");
    assert_ok(&f.simulate(&out2, None, &store2), "simulate");
    let m2: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(out2.join("quantities.json")).unwrap())
            .unwrap();
    assert!(m2.get("vocabulary").is_none(), "no override ⇒ no provenance key");
}

// ── 3. An in-place edit re-keys ─────────────────────────────────────────────

#[test]
fn an_in_place_edit_of_the_vocabulary_rekeys_the_artifact() {
    let f = fixture();
    let v = f.vocab("v.camdl", VOCAB_A);
    let out = f.root.join("out");
    let store = f.root.join("store");
    assert_ok(&f.simulate(&out, Some(&v), &store), "simulate before the edit");
    let before = quantity_addresses(&out);

    // Correct a formula in place. Same path, different bytes.
    std::fs::write(&v, "quantities {\n  attack_rate = final((990 - S) / 990)\n}\n").unwrap();
    assert_ok(&f.simulate(&out, Some(&v), &store), "simulate after the edit");
    let after = quantity_addresses(&out);

    assert_eq!(after.len(), 4, "the edited vocabulary is a new address: {after:?}");
    for name in &before {
        assert!(after.contains(name), "the pre-edit artifact is still there: {name}");
    }
}

// ── 4. The run's identity does not move ─────────────────────────────────────

#[test]
fn the_sim_run_id_is_byte_identical_with_and_without_a_vocabulary() {
    let f = fixture();
    let va = f.vocab("va.camdl", VOCAB_A);
    let vb = f.vocab("vb.camdl", VOCAB_B);

    let store_plain = f.root.join("store_plain");
    let store_a = f.root.join("store_a");
    let store_b = f.root.join("store_b");
    assert_ok(&f.simulate(&f.root.join("o0"), None, &store_plain), "simulate");
    assert_ok(&f.simulate(&f.root.join("o1"), Some(&va), &store_a), "simulate with A");
    assert_ok(&f.simulate(&f.root.join("o2"), Some(&vb), &store_b), "simulate with B");

    let plain = only_run_id(&store_plain);
    assert_eq!(
        plain,
        only_run_id(&store_a),
        "a reporting vocabulary must not re-key the trajectory it reports on"
    );
    assert_eq!(plain, only_run_id(&store_b), "…for any vocabulary");
}

// ── Refusals ────────────────────────────────────────────────────────────────

/// A name the model does not declare is a hard error naming BOTH the name and
/// the vocabulary file. Naming only the name leaves the author unable to tell
/// which of the two files is wrong.
#[test]
fn a_missing_symbol_names_the_symbol_and_the_file() {
    let f = fixture();
    let bad = f.vocab("bad.camdl", "quantities {\n  cfr = final(f_cfr)\n}\n");
    let out = f.simulate(&f.root.join("out"), Some(&bad), &f.root.join("store"));
    assert!(!out.status.success(), "an undeclared name must fail the run");
    let err = String::from_utf8_lossy(&out.stderr);
    assert!(err.contains("f_cfr"), "the error names the symbol: {err}");
    assert!(err.contains("bad.camdl"), "the error names the vocabulary file: {err}");
}

/// A vocabulary is resolved by the compiler against the model's symbols, so it
/// cannot be applied to an already-compiled `.ir.json`. Refuse rather than run
/// with the model's own quantities under a flag that says otherwise.
#[test]
fn a_vocabulary_on_compiled_ir_is_refused() {
    let f = fixture();
    let va = f.vocab("va.camdl", VOCAB_A);
    // A committed golden IR — the refusal fires before any compile, so the
    // model's content is irrelevant; what matters is that the path is `.ir.json`.
    let manifest = std::env::var("CARGO_MANIFEST_DIR").unwrap();
    let ir = Path::new(&manifest).join("../../../ocaml/golden/sir_basic.ir.json");
    assert!(ir.exists(), "golden IR missing: {}", ir.display());

    let out = run(&[
        "simulate",
        ir.to_str().unwrap(),
        "--scenario",
        "baseline",
        "--seed",
        "7",
        "--output-dir",
        f.root.join("store").to_str().unwrap(),
        "--quantities-out",
        f.root.join("out").to_str().unwrap(),
        "--quantities",
        va.to_str().unwrap(),
    ]);
    assert!(!out.status.success(), "applying a vocabulary to compiled IR must be refused");
    let err = String::from_utf8_lossy(&out.stderr);
    assert!(
        err.contains("SOURCE") || err.contains("source"),
        "the refusal explains that the vocabulary needs the model source: {err}"
    );
}

/// `--quantities` without `--quantities-out` would compile a vocabulary and
/// then emit nothing. clap refuses it at the boundary.
#[test]
fn a_vocabulary_without_an_output_directory_is_refused() {
    let f = fixture();
    let va = f.vocab("va.camdl", VOCAB_A);
    let out = run(&[
        "simulate",
        f.model.to_str().unwrap(),
        "--scenario",
        "baseline",
        "--seed",
        "7",
        "--output-dir",
        f.root.join("store").to_str().unwrap(),
        "--quantities",
        va.to_str().unwrap(),
    ]);
    assert!(!out.status.success(), "--quantities alone emits nothing; refuse it");
}
