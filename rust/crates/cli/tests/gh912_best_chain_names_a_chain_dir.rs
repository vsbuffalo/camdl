//! gh#912: the `best_chain` a leaf stores is the number of a `chain_N/`
//! directory that exists.
//!
//! Three surfaces name the winning chain and a reader will line them up:
//! the method's own stderr line (`… (chain N)`), `fit_state.toml`'s
//! `best_chain`, and `run.json`'s `inputs.best_chain` (which is what
//! `camdl show` and `camdl list --format json` print). They were not the
//! same number: the two files carried the runner's 0-based index while the
//! stderr line and every `chain_N/` directory were numbered from one, so on
//! a two-chain fit the stored value named the chain that did NOT win, and a
//! stored `0` named a directory no method writes.
//!
//! What this pins, per method, on a real leaf:
//!
//!   * the three surfaces agree on one number;
//!   * that number is in `1..=n_chains`;
//!   * `chain_<best_chain>/` is a directory on the leaf, so a reader can open
//!     the winner's output with no arithmetic;
//!   * on IF2, `mle_params.toml`'s `[provenance] chain` — 1-indexed since it
//!     was written — agrees with it, so the leaf does not name two chains.
//!
//! Both fits are the committed spatial polio AFP+ES fixture
//! (`tests/fixtures/polio_afp_es/`) shrunk to a test-sized method block: a
//! real multi-stream problem, small enough to run in the default gate.

use std::path::{Path, PathBuf};
use std::process::Command;

fn camdl_bin() -> PathBuf {
    let manifest = std::env::var("CARGO_MANIFEST_DIR")
        .expect("CARGO_MANIFEST_DIR set under cargo test");
    let p = Path::new(&manifest).join("../../target/release/camdl");
    assert!(
        p.exists(),
        "release camdl binary missing: {} — run `make build-rust` or `make test`",
        p.display()
    );
    p
}

/// The dune-built compiler. The fixture model is compiled with THIS binary
/// (a stale `camdlc` on PATH predates the stratified-observation header), and
/// the fit runs against the resulting `.ir.json`, so no `CAMDLC` override is
/// needed at fit time.
fn camdlc_bin() -> PathBuf {
    let manifest = std::env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR");
    let p = Path::new(&manifest).join("../../../ocaml/_build/default/bin/camdlc.exe");
    assert!(p.exists(), "camdlc.exe missing: {} — run `make build-ocaml`", p.display());
    p
}

fn fixture(rel: &str) -> PathBuf {
    let manifest = std::env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR");
    Path::new(&manifest).join("../../../tests/fixtures").join(rel)
}

fn compile_model(dir: &Path) -> PathBuf {
    let ir = dir.join("polio_afp_es_2patch.ir.json");
    let out = Command::new(camdlc_bin())
        .arg(fixture("polio_afp_es_2patch.camdl"))
        .output()
        .expect("spawn camdlc");
    assert!(out.status.success(), "camdlc failed: {}", String::from_utf8_lossy(&out.stderr));
    std::fs::write(&ir, &out.stdout).unwrap();
    ir
}

/// One of the committed polio fit configs, shrunk to a test-sized
/// `[method]` block.
///
/// Everything above the fixture's own `[method]` header is kept verbatim —
/// the model, the two observation streams, the `[estimate]` bounds/priors and
/// the `[fixed]` truth values are the committed problem, not a restatement of
/// it. Only two things change: `method_block` replaces the fixture's method,
/// and the three paths are rewritten to absolute, since this copy lives
/// outside the fixture directory its relative paths resolve against.
fn shrunk_config(
    fixture_toml: &str, ir: &Path, dir: &Path, name: &str, method_block: &str,
) -> PathBuf {
    let src = fixture(fixture_toml);
    let fixture_dir = src.parent().expect("fixture parent").to_path_buf();
    let text = std::fs::read_to_string(&src)
        .unwrap_or_else(|e| panic!("read {}: {e}", src.display()));

    let mut body = String::new();
    let mut in_observations = false;
    for line in text.lines() {
        let t = line.trim_start();
        if t.starts_with('[') {
            if t.starts_with("[method]") {
                break;
            }
            in_observations = t.starts_with("[data.observations]");
        }
        let key = t.split('=').next().unwrap_or("").trim();
        let rel = || {
            t.split_once('=')
                .map(|(_, v)| v.trim().trim_matches('"').to_string())
                .expect("a key = value line")
        };
        if key == "camdl" {
            body.push_str(&format!("camdl = \"{}\"\n", ir.display()));
        } else if in_observations && t.contains('=') && !t.starts_with('#') {
            body.push_str(&format!(
                "{key} = \"{}\"\n", fixture_dir.join(rel()).display()));
        } else {
            body.push_str(line);
            body.push('\n');
        }
    }
    assert!(body.contains("[estimate]") && body.contains("[data.observations]"),
        "the fixture's problem must survive the shrink:\n{body}");
    body.push_str(method_block);

    let p = dir.join(format!("{name}.toml"));
    std::fs::write(&p, body).unwrap();
    p
}

/// Run `camdl fit run` on a config, returning its stderr.
fn run_fit(config: &Path, out_root: &Path) -> String {
    let r = Command::new(camdl_bin())
        .current_dir(config.parent().expect("config parent"))
        // The fixture declares no `output_dir` (one would write into the
        // committed fixture directory), so the destination is pinned here —
        // which also keeps the test off a developer's ambient value.
        .env("CAMDL_OUTPUT_DIR", out_root)
        .env("CAMDL_SKIP_VERSION_CHECK", "1")
        .args(["fit", "run", &config.to_string_lossy()])
        .output()
        .expect("spawn camdl fit run");
    let stderr = String::from_utf8_lossy(&r.stderr).to_string();
    assert!(r.status.success(), "fit run failed:\nstderr={stderr}");
    stderr
}

/// The single `fit_stage` leaf under `<out_root>/fits/`.
fn stage_leaf(out_root: &Path) -> PathBuf {
    let mut stack = vec![out_root.join("fits")];
    while let Some(d) = stack.pop() {
        let rj = d.join("run.json");
        if rj.is_file() {
            if let Ok(v) = serde_json::from_str::<serde_json::Value>(
                &std::fs::read_to_string(&rj).unwrap_or_default())
            {
                if v.get("kind").and_then(|k| k.as_str()) == Some("fit_stage") {
                    return d;
                }
            }
        }
        if let Ok(es) = std::fs::read_dir(&d) {
            for e in es.flatten() {
                if e.path().is_dir() { stack.push(e.path()); }
            }
        }
    }
    panic!("no fit_stage leaf under {}", out_root.join("fits").display());
}

fn fit_state_best_chain(leaf: &Path) -> i64 {
    let p = leaf.join("fit_state.toml");
    let text = std::fs::read_to_string(&p)
        .unwrap_or_else(|e| panic!("read {}: {e}", p.display()));
    for line in text.lines() {
        if let Some((k, v)) = line.split_once('=') {
            if k.trim() == "best_chain" {
                return v.trim().parse()
                    .unwrap_or_else(|e| panic!("best_chain parse in {}: {e}", p.display()));
            }
        }
    }
    panic!("no best_chain in {}", p.display());
}

fn run_json_best_chain(leaf: &Path) -> i64 {
    let p = leaf.join("run.json");
    let v: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(&p).unwrap_or_else(|e| panic!("read {}: {e}", p.display())))
        .expect("parse run.json");
    v["inputs"]["best_chain"].as_i64().unwrap_or_else(|| {
        panic!("inputs.best_chain absent or not an integer in {}: {}",
            p.display(), v["inputs"])
    })
}

/// The chain number in the method's own stderr line — the line containing
/// `marker`, from which `(chain N)` is read.
fn stderr_chain(stderr: &str, marker: &str) -> i64 {
    let line = stderr.lines().find(|l| l.contains(marker)).unwrap_or_else(|| {
        panic!("no stderr line containing {marker:?}:\n{stderr}")
    });
    let after = line.split_once("(chain ").unwrap_or_else(|| {
        panic!("no `(chain N)` on the {marker:?} line: {line:?}")
    }).1;
    let digits: String = after.chars().take_while(char::is_ascii_digit).collect();
    digits.parse().unwrap_or_else(|e| panic!("chain number parse in {line:?}: {e}"))
}

/// `[provenance] chain` from an `mle_params.toml`. The params section above
/// it is bare `key = value` lines, so the scan starts at the table header.
fn mle_provenance_chain(leaf: &Path) -> i64 {
    let p = leaf.join("mle_params.toml");
    let text = std::fs::read_to_string(&p)
        .unwrap_or_else(|e| panic!("read {}: {e}", p.display()));
    let mut in_prov = false;
    for line in text.lines() {
        let t = line.trim();
        if t.starts_with('[') { in_prov = t.starts_with("[provenance]"); continue; }
        if !in_prov { continue; }
        if let Some((k, v)) = t.split_once('=') {
            if k.trim() == "chain" {
                return v.trim().parse()
                    .unwrap_or_else(|e| panic!("chain parse in {}: {e}", p.display()));
            }
        }
    }
    panic!("no [provenance] chain in {}", p.display());
}

/// The shared assertion: the three surfaces agree, the number is a real chain,
/// and its directory is on the leaf.
fn assert_best_chain_names_a_chain_dir(
    method: &str, leaf: &Path, stderr: &str, marker: &str, n_chains: i64,
) -> i64 {
    let printed = stderr_chain(stderr, marker);
    let stored = fit_state_best_chain(leaf);
    let recorded = run_json_best_chain(leaf);

    assert_eq!(stored, printed,
        "{method}: fit_state.toml says best_chain = {stored} but the run printed \
         chain {printed} — the stored value must be the 1-based chain number the \
         run named, not the 0-based index (gh#912).\nleaf: {}", leaf.display());
    assert_eq!(recorded, printed,
        "{method}: run.json inputs.best_chain = {recorded} but the run printed \
         chain {printed} — `camdl show` and `camdl list --format json` print this \
         value raw, so it must be the 1-based chain number (gh#912).\nleaf: {}",
        leaf.display());

    assert!((1..=n_chains).contains(&stored),
        "{method}: best_chain = {stored} is not a chain of a {n_chains}-chain fit \
         (1..={n_chains}) — 0 names a directory no method writes (gh#912)");

    let chain_dir = leaf.join(format!("chain_{stored}"));
    assert!(chain_dir.is_dir(),
        "{method}: best_chain = {stored} but {} is not a directory — the stored \
         number must name the winning chain's output (gh#912). Leaf holds: {:?}",
        chain_dir.display(),
        std::fs::read_dir(leaf).unwrap().flatten()
            .map(|e| e.file_name()).collect::<Vec<_>>());

    stored
}

const PGAS_METHOD: &str = "\
[method]
algorithm = \"pgas\"
backend   = \"chain_binomial\"
chains    = 2
particles = 50
sweeps    = 12
burn_in   = 4
thin      = 1
";

const IF2_METHOD: &str = "\
[method]
algorithm  = \"if2\"
backend    = \"chain_binomial\"
chains     = 2
particles  = 50
iterations = 3
cooling    = 0.5
loglik_eval = { n_particles = 1000, n_replicates = 2 }
dt_check = { enabled = false }
";

/// PGAS: the sampler picks the chain whose sweeps reached the highest
/// complete-data log-likelihood and prints it; the leaf must store that chain.
#[test]
fn pgas_best_chain_is_the_chain_the_run_named() {
    let tmp = tempfile::tempdir().unwrap();
    let ir = compile_model(tmp.path());
    let cfg = shrunk_config("polio_afp_es/pgas.toml", &ir, tmp.path(), "pgas_small", PGAS_METHOD);
    let out_root = tmp.path().join("results_pgas");

    let stderr = run_fit(&cfg, &out_root);
    let leaf = stage_leaf(&out_root);
    assert_best_chain_names_a_chain_dir(
        "pgas", &leaf, &stderr, "best complete-data ll:", 2);
}

/// IF2: the same three surfaces, plus the fourth artifact on an optimizer
/// leaf — `mle_params.toml`'s `[provenance] chain`, which has been 1-indexed
/// since it was written. Two artifacts of one leaf must not name two chains.
#[test]
fn if2_best_chain_agrees_with_the_mle_provenance_chain() {
    let tmp = tempfile::tempdir().unwrap();
    let ir = compile_model(tmp.path());
    let cfg = shrunk_config("polio_afp_es/fit.toml", &ir, tmp.path(), "if2_small", IF2_METHOD);
    let out_root = tmp.path().join("results_if2");

    let stderr = run_fit(&cfg, &out_root);
    let leaf = stage_leaf(&out_root);
    let stored = assert_best_chain_names_a_chain_dir("if2", &leaf, &stderr, "best ll=", 2);

    let prov = mle_provenance_chain(&leaf);
    assert_eq!(prov, stored,
        "if2: mle_params.toml [provenance] chain = {prov} but fit_state.toml / \
         run.json say best_chain = {stored} — one leaf must not name two chains \
         (gh#912).\nleaf: {}", leaf.display());
}
