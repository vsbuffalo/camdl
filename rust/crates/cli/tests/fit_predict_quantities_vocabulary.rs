//! `camdl fit predict <fit> --quantities FILE`: report a fit that already exists
//! with a corrected reporting vocabulary.
//!
//! Proposal `docs/dev/proposals/2026-08-19-quantities-as-a-separable-layer.md`,
//! answering gh#618. The obstacle it removes is specific: `fit predict` reads
//! the model IR ARCHIVED in the fit leaf, so editing a `quantities {}` formula
//! in the source has no effect on a fit that has already run. It does not
//! orphan the fit — it simply does nothing, which is the worse failure, because
//! the wrong number keeps being reported at exit 0.
//!
//! What is pinned here:
//!
//! - a vocabulary applied to a fit produces its table, and the model's own
//!   block is not consulted;
//! - two vocabularies on one fit land at two content addresses; the same
//!   vocabulary twice lands at one;
//! - an in-place edit of the vocabulary re-keys the table;
//! - the FIT's identity does not move — same `run_id`, same segment, byte-for-
//!   byte, with and without `--quantities`;
//! - a name the model does not declare is a hard error naming the name AND the
//!   file;
//! - a model source that has drifted from the fit is refused rather than used.

use std::path::{Path, PathBuf};
use std::process::Command;

fn binary() -> PathBuf {
    PathBuf::from(env!("CARGO_BIN_EXE_camdl"))
}

/// A closed SIR with a weekly NegBinomial stream and its OWN `quantities {}`
/// block declaring `model_peak`. `R0` is a param-bearing `let` — inlined by the
/// compiler and absent from the IR — so a vocabulary that reports it proves the
/// vocabulary is resolved against the model SOURCE.
const MODEL: &str = r#"
time_unit = 'days

compartments { S, I, R }

parameters {
  beta  : rate         in [0.05, 1.0]  ~ log_normal(mu = -1.0, sigma = 0.5)
  gamma : rate         in [0.01, 0.5]  ~ log_normal(mu = -2.0, sigma = 0.5)
  N0    : count
  I0    : count
  rho   : probability  in [0.1, 0.9]   ~ beta(alpha = 2.0, beta = 5.0)
  k     : positive     in [1.0, 100.0] ~ half_normal(sigma = 10.0)
}

let N  = S + I + R
let R0 = beta / gamma

transitions {
  infection : S --> I  @ beta * S * I / N
  recovery  : I --> R  @ gamma * I
}

init {
  S = N0 - I0
  I = I0
}

observations {
  weekly_cases {
    columns       { time : time, weekly_cases : count }
    projected     = incidence(infection)
    emit_schedule = every 7 'days
    weekly_cases  ~ neg_binomial(mean = rho * projected, r = k)
  }
}

quantities {
  model_peak = max(I / N)
}

simulate {
  from = 0 'days
  to   = 80 'days
}
"#;

const DATA: &str = "time\tweekly_cases\n7\t16\n14\t166\n21\t626\n28\t1303\n35\t1260\n42\t1023\n49\t327\n56\t91\n";

const FIT_TOML: &str = r#"output_dir = "results"

[model]
camdl = "model.camdl"

[data.observations]
weekly_cases = "weekly_cases.tsv"

[estimate]
beta  = { bounds = [0.05, 1.0], start = 0.4 }
gamma = { bounds = [0.01, 0.5], start = 0.15 }

[fixed]
N0  = 10000
I0  = 10
rho = 0.6
k   = 10.0

[stages.posterior]
algorithm = "mh"
backend = "ode"
chains = 2
iterations = 60
burn_in = 20
thin = 1
"#;

/// Vocabulary A: an attack rate plus `R0`, the inlined `let`.
const VOCAB_A: &str = "quantities {\n  attack_rate = final((10000 - S) / 10000)\n  r_zero      = final(R0)\n}\n";

/// Vocabulary B: a different reporting question over the same fit.
const VOCAB_B: &str = "quantities {\n  peak_time = time_of_max(I)\n}\n";

fn run(dir: &Path, args: &[&str]) -> std::process::Output {
    Command::new(binary())
        .args(args)
        .current_dir(dir)
        .env("CAMDL_SKIP_VERSION_CHECK", "1")
        .output()
        .expect("spawn camdl")
}

fn assert_ok(out: &std::process::Output, what: &str) {
    assert!(
        out.status.success(),
        "{what} failed:\nstdout={}\nstderr={}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr),
    );
}

struct Fit {
    _tmp: tempfile::TempDir,
    root: PathBuf,
    segment: PathBuf,
}

/// Run one small `mh` fit and return its project root + fit segment.
fn fitted(tag: &str) -> Fit {
    let tmp = tempfile::tempdir().unwrap();
    let root = tmp.path().to_path_buf();
    std::fs::write(root.join("model.camdl"), MODEL).unwrap();
    std::fs::write(root.join("weekly_cases.tsv"), DATA).unwrap();
    std::fs::write(root.join("fit.toml"), FIT_TOML).unwrap();
    let out = run(&root, &["fit", "run", "fit.toml", "--seed", "1"]);
    assert_ok(&out, &format!("fit run ({tag})"));

    let fits = root.join("results").join("fits");
    let segment = std::fs::read_dir(&fits)
        .unwrap()
        .flatten()
        .map(|e| e.path())
        .find(|p| p.is_dir())
        .expect("a fit segment");
    Fit { _tmp: tmp, root, segment }
}

impl Fit {
    fn vocab(&self, name: &str, body: &str) -> PathBuf {
        let p = self.root.join(name);
        std::fs::write(&p, body).unwrap();
        p
    }

    fn predict(&self, vocab: Option<&str>) -> std::process::Output {
        let mut args: Vec<&str> = vec![
            "fit",
            "predict",
            self.segment.to_str().unwrap(),
            "--horizon",
            "free_forward",
        ];
        if let Some(v) = vocab {
            args.push("--quantities");
            args.push(v);
        }
        run(&self.root, &args)
    }

    /// The `quantities*` addresses the fit segment currently holds, sorted.
    fn addresses(&self) -> Vec<String> {
        let mut names: Vec<String> = std::fs::read_dir(&self.segment)
            .unwrap()
            .flatten()
            .map(|e| e.file_name().to_string_lossy().into_owned())
            .filter(|n| n.starts_with("quantities"))
            .collect();
        names.sort();
        names
    }

    /// Every `run.json` under this fit — one per stage leaf — as
    /// (segment-relative path, bytes), sorted. This IS the fit's identity on
    /// disk: the leaf addresses plus the records they hold.
    fn identity_snapshot(&self) -> Vec<(String, String)> {
        let mut out: Vec<(String, String)> = Vec::new();
        let mut stack = vec![self.segment.clone()];
        while let Some(d) = stack.pop() {
            for e in std::fs::read_dir(&d).into_iter().flatten().flatten() {
                let p = e.path();
                if p.is_dir() {
                    stack.push(p);
                } else if p.file_name().is_some_and(|n| n == "run.json") {
                    let rel = p
                        .strip_prefix(&self.segment)
                        .unwrap()
                        .to_string_lossy()
                        .into_owned();
                    out.push((rel, std::fs::read_to_string(&p).unwrap()));
                }
            }
        }
        assert!(!out.is_empty(), "a fit must have at least one leaf run record");
        out.sort();
        out
    }
}

// ── The vocabulary replaces the model's own block ───────────────────────────

#[test]
fn a_vocabulary_applied_to_a_fit_produces_its_table_and_not_the_models() {
    let f = fitted("replace");
    let va = f.vocab("va.camdl", VOCAB_A);
    assert_ok(
        &f.predict(Some(va.to_str().unwrap())),
        "fit predict --quantities",
    );

    let addrs = f.addresses();
    let dir = addrs
        .iter()
        .find(|n| !n.ends_with(".json"))
        .expect("a quantities directory");
    assert!(
        dir.starts_with("quantities-"),
        "a supplied vocabulary writes a keyed address, got {dir}"
    );
    let qdir = f.segment.join(dir);
    assert!(qdir.join("attack_rate.tsv").exists(), "the vocabulary's quantity is emitted");
    assert!(
        qdir.join("r_zero.tsv").exists(),
        "a param-bearing `let` (R0) resolves — the vocabulary is compiled against \
         the model SOURCE, where it still exists; it is inlined away in the archived IR"
    );
    assert!(
        !qdir.join("model_peak.tsv").exists(),
        "the model's own quantities block must NOT be consulted"
    );
    assert!(
        !f.segment.join("quantities").join("model_peak.tsv").exists(),
        "…and this predict must not have written the model's block anywhere"
    );
}

/// The control: no `--quantities` reports the model's own block at the
/// historical address, exactly as before.
#[test]
fn without_a_vocabulary_the_models_own_block_is_reported() {
    let f = fitted("control");
    assert_ok(&f.predict(None), "fit predict");
    assert_eq!(f.addresses(), vec!["quantities", "quantities.json"]);
    assert!(f.segment.join("quantities").join("model_peak.tsv").exists());
}

// ── The vocabulary keys the artifact ────────────────────────────────────────

#[test]
fn two_vocabularies_on_one_fit_are_two_addresses_and_one_is_one() {
    let f = fitted("addresses");
    let va = f.vocab("va.camdl", VOCAB_A);
    let vb = f.vocab("vb.camdl", VOCAB_B);

    assert_ok(&f.predict(Some(va.to_str().unwrap())), "predict with A");
    let after_a = f.addresses();
    assert_eq!(after_a.len(), 2, "one dir + one manifest: {after_a:?}");

    assert_ok(&f.predict(Some(va.to_str().unwrap())), "predict with A again");
    assert_eq!(
        f.addresses(),
        after_a,
        "the same vocabulary twice is ONE address (a correct collision)"
    );

    assert_ok(&f.predict(Some(vb.to_str().unwrap())), "predict with B");
    let after_b = f.addresses();
    assert_eq!(
        after_b.len(),
        4,
        "two vocabularies must not overwrite each other: {after_b:?}"
    );

    let dir_a = after_a.iter().find(|n| !n.ends_with(".json")).unwrap();
    let dir_b = after_b
        .iter()
        .find(|n| !n.ends_with(".json") && *n != dir_a)
        .expect("B's own directory");
    assert!(f.segment.join(dir_a).join("attack_rate.tsv").exists());
    assert!(f.segment.join(dir_b).join("peak_time.tsv").exists());
    assert!(!f.segment.join(dir_a).join("peak_time.tsv").exists());
}

#[test]
fn an_in_place_edit_of_the_vocabulary_rekeys_the_table() {
    let f = fitted("edit");
    let v = f.vocab("v.camdl", VOCAB_A);
    assert_ok(&f.predict(Some(v.to_str().unwrap())), "predict before the edit");
    let before = f.addresses();

    // Correct a formula in place — the workflow the whole feature exists for.
    std::fs::write(&v, "quantities {\n  attack_rate = final((10000 - S) / 9990)\n}\n").unwrap();
    assert_ok(&f.predict(Some(v.to_str().unwrap())), "predict after the edit");
    let after = f.addresses();

    assert_eq!(after.len(), 4, "the corrected vocabulary is a new address: {after:?}");
    for name in &before {
        assert!(after.contains(name), "the pre-edit table is still there: {name}");
    }
}

/// The manifest names the vocabulary that produced the table and pins its
/// digest, so a table can be traced to its formulas.
#[test]
fn the_manifest_records_the_vocabulary() {
    let f = fitted("manifest");
    let va = f.vocab("va.camdl", VOCAB_A);
    assert_ok(&f.predict(Some(va.to_str().unwrap())), "predict with A");

    let name = f
        .addresses()
        .into_iter()
        .find(|n| n.ends_with(".json"))
        .expect("a manifest");
    let m: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(f.segment.join(&name)).unwrap()).unwrap();
    assert!(m["vocabulary"]["file"].as_str().unwrap().ends_with("va.camdl"), "{m}");
    assert_eq!(m["vocabulary"]["sha256"].as_str().unwrap().len(), 64);
}

// ── The fit's identity does not move ────────────────────────────────────────

#[test]
fn the_fit_run_id_is_byte_identical_with_and_without_a_vocabulary() {
    let f = fitted("identity");
    let before = f.identity_snapshot();
    let segment_before = f.segment.clone();

    let va = f.vocab("va.camdl", VOCAB_A);
    assert_ok(&f.predict(Some(va.to_str().unwrap())), "predict with A");
    let vb = f.vocab("vb.camdl", VOCAB_B);
    assert_ok(&f.predict(Some(vb.to_str().unwrap())), "predict with B");
    assert_ok(&f.predict(None), "predict with the model's own block");

    assert_eq!(f.segment, segment_before, "the fit's segment must not move");
    assert_eq!(
        f.identity_snapshot(),
        before,
        "a reporting vocabulary must leave every leaf address and run record \
         byte-identical — quantities are excluded from model identity, and \
         predict must not re-key the fit it reads"
    );
}

// ── Refusals ────────────────────────────────────────────────────────────────

#[test]
fn a_missing_symbol_names_the_symbol_and_the_file() {
    let f = fitted("missing");
    let bad = f.vocab("bad.camdl", "quantities {\n  cfr = final(f_cfr)\n}\n");
    let out = f.predict(Some(bad.to_str().unwrap()));
    assert!(!out.status.success(), "an undeclared name must fail the predict");
    let err = String::from_utf8_lossy(&out.stderr);
    assert!(err.contains("f_cfr"), "the error names the symbol: {err}");
    assert!(err.contains("bad.camdl"), "the error names the vocabulary file: {err}");
}

/// The vocabulary is compiled against the model SOURCE, but the fit ran on the
/// ARCHIVED IR. If the source has drifted the two are different models, and
/// reporting the fit through formulas resolved against the wrong one is a
/// plausible-looking wrong answer. `model_ir_hash` excludes quantities, so the
/// check asks exactly "same model apart from its reporting layer?".
#[test]
fn a_drifted_model_source_is_refused_not_used() {
    let f = fitted("drift");
    let va = f.vocab("va.camdl", VOCAB_A);
    // A structural edit to the source AFTER the fit: a new compartment.
    let drifted = MODEL.replace("compartments { S, I, R }", "compartments { S, I, R, D }");
    assert_ne!(drifted, MODEL);
    std::fs::write(f.root.join("model.camdl"), &drifted).unwrap();

    let out = f.predict(Some(va.to_str().unwrap()));
    assert!(!out.status.success(), "a drifted source must be refused");
    let err = String::from_utf8_lossy(&out.stderr);
    assert!(
        err.contains("no longer the model the fit ran on"),
        "the refusal says the source drifted: {err}"
    );

    // And the control: editing ONLY the model's quantities block is NOT drift —
    // that is the whole point of keying the check on a quantities-free hash.
    let reporting_only = MODEL.replace("model_peak = max(I / N)", "model_peak = max(I)");
    assert_ne!(reporting_only, MODEL);
    std::fs::write(f.root.join("model.camdl"), &reporting_only).unwrap();
    assert_ok(
        &f.predict(Some(va.to_str().unwrap())),
        "a quantities-only source edit must NOT count as drift",
    );
}
