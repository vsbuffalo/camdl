//! gh#785: `cross_chain_compat.json` on disk, beside `diagnostics.json`.
//!
//! `M[i][j] = log p(x_j | θ_i)` — chain `i`'s parameters scoring chain `j`'s
//! latent path — is what separates augmentation locking (each PGAS chain
//! pinned to the path its own parameters produced) from the marginal posterior
//! genuinely having several modes. The in-crate unit tests pin what the numbers
//! MEAN; this pins what a downstream reader (`camdl-scope`) actually finds:
//!
//! * the file exists for a PGAS stage, with the documented keys;
//! * `chains` is 1-based, matching the `chain_N/` directories beside it, and
//!   describes `M`'s actual rows;
//! * the DIAGONAL agrees with what the sampler itself recorded for that chain's
//!   final sweep — `transition_ll + initial_state_ll` in `chain_N/trace.tsv`.
//!   Without that, the matrix could drift from the sampler's own numbers and
//!   nothing would say so;
//! * both derived numbers are the stated functions of `M`;
//! * the file is ABSENT for a PMMH stage. PMMH accepts on a marginal likelihood
//!   estimate and stores no latent path, so there is nothing to score. An empty
//!   matrix would read as "measured, found nothing" — a different and false
//!   claim.
//!
//! Skipped when the release binary or camdlc isn't present.

use std::path::{Path, PathBuf};
use std::process::Command;

fn camdl_bin() -> PathBuf {
    let manifest = std::env::var("CARGO_MANIFEST_DIR")
        .expect("CARGO_MANIFEST_DIR set under cargo test");
    let p = Path::new(&manifest).join("../../target/release/camdl");
    assert!(
        p.exists(),
        "release camdl binary missing: {} - run `make build-rust` or `make test` (gh#105)",
        p.display()
    );
    p
}

fn camdlc_bin() -> Option<PathBuf> {
    let manifest = std::env::var("CARGO_MANIFEST_DIR").ok()?;
    let p = Path::new(&manifest).join("../../../ocaml/_build/default/bin/camdlc.exe");
    if p.exists() { Some(p) } else { None }
}

struct Tmp(PathBuf);
impl Tmp { fn path(&self) -> &Path { &self.0 } }
impl Drop for Tmp { fn drop(&mut self) { let _ = std::fs::remove_dir_all(&self.0); } }
fn tempdir(tag: &str) -> Tmp {
    let ns = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos();
    let base = std::env::temp_dir().join(format!(
        "camdl_gh785_{}_{}_{}", tag, std::process::id(), ns));
    std::fs::create_dir_all(&base).unwrap();
    Tmp(base)
}

/// A tiny SIR with a DECLARED initial-state law (`I ~ poisson(rate = I0)`), so
/// `log p(x₀ | θ)` is a live term of the matrix rather than a constant zero —
/// the case gh#785 calls out as where the augmentation coupling actually lives.
fn write_fixture(dir: &Path) -> (PathBuf, PathBuf) {
    let camdlc = camdlc_bin().expect("camdlc.exe present");
    let src = r#"
time_unit = 'days
compartments { S, I, R }
parameters {
  beta  : rate  in [0.001, 5.0]
  gamma : rate  in [0.01, 1.0]
  N0    : count in [100, 10000]
  I0    : count in [1, 500]
}
transitions {
  infection : S --> I @ beta * S * I / N0
  recovery  : I --> R @ gamma * I
}
observations {
  cases {
    columns       { time : time, cases : count }
    projected  = prevalence(I)
    emit_schedule = every 1 'days
    cases ~ poisson(rate = projected)
  }
}
init {
  S = N0 - I0
  I ~ poisson(rate = I0)
}
simulate { from = 0 'days  to = 20 'days }
"#;
    let model_path = dir.join("sir.camdl");
    std::fs::write(&model_path, src).unwrap();
    let ir_path = dir.join("sir.ir.json");
    let out = Command::new(&camdlc).arg(&model_path).output().unwrap();
    assert!(out.status.success(),
        "camdlc failed: {}", String::from_utf8_lossy(&out.stderr));
    std::fs::write(&ir_path, &out.stdout).unwrap();

    let data_path = dir.join("cases.tsv");
    std::fs::write(&data_path, "time\tcases\n\
        1\t11\n2\t14\n3\t18\n4\t24\n5\t31\n6\t40\n7\t48\n8\t55\n9\t60\n10\t58\n\
        11\t52\n12\t45\n13\t38\n14\t31\n15\t25\n16\t20\n17\t16\n18\t13\n19\t10\n20\t8\n")
        .unwrap();

    (ir_path, data_path)
}

/// Sweeps per chain. Long enough that at least one chain's θ moves after
/// burn-in — see `theta_moved_after_burn_in`, which the diagonal assertion
/// depends on to be able to see anything.
const SWEEPS: usize = 60;
const BURN_IN: usize = 10;

/// A fit.toml whose single stage runs `algorithm` with `chains` chains.
fn write_fit_toml(
    dir: &Path, ir: &Path, data: &Path, algorithm: &str, chains: usize,
) -> (PathBuf, PathBuf) {
    let out = dir.join(format!("results_{algorithm}"));
    // PGAS counts sweeps, PMMH counts iterations; everything else is shared.
    let budget = if algorithm == "pgas" {
        format!("sweeps = {SWEEPS}")
    } else {
        format!("iterations = {SWEEPS}")
    };
    let toml = format!(r#"
output_dir = "{out}"
[model]
camdl = "{ir}"
[data.observations]
cases = "{data}"
[config]
dt = 1.0
[estimate]
beta  = {{ bounds = [0.01, 5.0], prior = {{ log_normal = {{ mu = -0.3, sigma = 0.5 }} }}, start = 0.8 }}
gamma = {{ bounds = [0.01, 1.0], prior = {{ log_normal = {{ mu = -1.2, sigma = 0.5 }} }}, start = 0.3 }}
[fixed]
N0 = 1000
I0 = 10
[stages.post]
algorithm = "{algorithm}"
backend = "chain_binomial"
chains = {chains}
particles = 40
{budget}
burn_in = {BURN_IN}
"#,
        out = out.display(), ir = ir.display(), data = data.display(),
    );
    let p = dir.join(format!("fit_{algorithm}.toml"));
    std::fs::write(&p, toml).unwrap();
    (p, out)
}

/// The stage leaf directory under `<out>/fits/` — the one holding
/// `diagnostics.json` and the `chain_N/` dirs.
fn stage_leaf(out: &Path) -> PathBuf {
    let mut stack = vec![out.join("fits")];
    while let Some(d) = stack.pop() {
        let rj = d.join("run.json");
        if rj.is_file() {
            if let Ok(v) = serde_json::from_str::<serde_json::Value>(
                &std::fs::read_to_string(&rj).unwrap_or_default(),
            ) {
                if v.get("kind").and_then(|k| k.as_str()) == Some("fit_stage") {
                    return d;
                }
            }
        }
        if let Ok(es) = std::fs::read_dir(&d) {
            for e in es.flatten() { if e.path().is_dir() { stack.push(e.path()); } }
        }
    }
    panic!("no fit_stage leaf under {}", out.join("fits").display());
}

fn run_fit(bin: &Path, fit: &Path) {
    let r = Command::new(bin)
        .arg("fit").arg("run").arg(fit).arg("--seed").arg("7")
        .output().expect("spawn camdl");
    assert!(r.status.success(),
        "fit run failed: {}", String::from_utf8_lossy(&r.stderr));
}

/// One chain's `trace.tsv` as a header plus one row of fields per sweep. The
/// callback that writes it fires on EVERY sweep, so the last row is the final
/// sweep — the one the matrix's diagonal is computed at.
fn trace(stage: &Path, chain_1based: usize) -> (Vec<String>, Vec<Vec<String>>) {
    let path = stage.join(format!("chain_{chain_1based}/trace.tsv"));
    let text = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
    let mut lines = text.lines().filter(|l| !l.trim().is_empty());
    let header: Vec<String> =
        lines.next().expect("trace header").split('\t').map(String::from).collect();
    let rows: Vec<Vec<String>> = lines
        .map(|l| l.split('\t').map(String::from).collect::<Vec<_>>())
        .collect();
    assert!(!rows.is_empty(), "chain {chain_1based} wrote no trace rows");
    for r in &rows {
        assert_eq!(r.len(), header.len(), "trace row width must match the header");
    }
    (header, rows)
}

fn column<'a>(header: &[String], row: &'a [String], name: &str) -> &'a str {
    let i = header.iter().position(|c| c == name)
        .unwrap_or_else(|| panic!("trace.tsv has no `{name}` column; header: {header:?}"));
    &row[i]
}

/// `transition_ll + initial_state_ll` for the final sweep — the sampler's own
/// accounting for the `(θ, x)` pair the matrix's diagonal must reproduce.
fn last_sweep_path_term(header: &[String], rows: &[Vec<String>]) -> f64 {
    let last = rows.last().unwrap();
    let g = |name: &str| column(header, last, name).parse::<f64>()
        .unwrap_or_else(|e| panic!("`{name}` = {:?} does not parse ({e})",
            column(header, last, name)));
    g("transition_ll") + g("initial_state_ll")
}

/// Did this chain's θ change between the first retained sweep and the final
/// one? The diagonal assertion is only capable of catching a mispaired `(θ, x)`
/// on a chain whose θ actually moved — on a frozen chain every sweep's θ is the
/// same vector and any pairing gives the same number. Asserted rather than
/// assumed, so a fixture that silently freezes turns the assertion vacuous
/// LOUDLY.
fn theta_moved_after_burn_in(header: &[String], rows: &[Vec<String>]) -> bool {
    let first = &rows[BURN_IN];
    let last = rows.last().unwrap();
    ["beta", "gamma"].iter().any(|p| column(header, first, p) != column(header, last, p))
}

#[test]
fn pgas_writes_the_cross_chain_compatibility_matrix() {
    let bin = camdl_bin();
    if camdlc_bin().is_none() { return }
    let tmp = tempdir("pgas");
    let (ir, data) = write_fixture(tmp.path());
    let (fit, out) = write_fit_toml(tmp.path(), &ir, &data, "pgas", 3);
    run_fit(&bin, &fit);

    let stage = stage_leaf(&out);
    let path = stage.join("cross_chain_compat.json");
    assert!(path.is_file(),
        "a multi-chain PGAS stage must write cross_chain_compat.json beside \
         diagnostics.json; {} holds: {:?}",
        stage.display(),
        std::fs::read_dir(&stage).unwrap()
            .flatten().map(|e| e.file_name()).collect::<Vec<_>>());
    let v: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&path).unwrap()).expect("parse artifact");

    // ── The declared conventions ─────────────────────────────────────────
    assert_eq!(v["includes_initial_state"], serde_json::json!(true));
    assert_eq!(v["terms"], serde_json::json!("transition + initial_state"));
    assert!(v["draw"].as_str().expect("draw is a string").contains("sweep"),
        "the artifact must say which draw it was computed at; got {:?}", v["draw"]);
    let numbering = v["chain_numbering"].as_str().expect("chain_numbering is a string");
    assert!(numbering.contains("1-based") && numbering.contains("chain_"),
        "the artifact must state its chain numbering; got {numbering:?}");

    // ── `chains` describes M's actual rows, 1-based ───────────────────────
    let chains: Vec<usize> = serde_json::from_value(v["chains"].clone()).expect("chains");
    assert_eq!(chains, vec![1, 2, 3],
        "chain ids must be the 1-based `chain_N/` names, not the 0-based draws.tsv key");
    for &c in &chains {
        assert!(stage.join(format!("chain_{c}")).is_dir(),
            "chains names chain {c}, but there is no chain_{c}/ directory beside the artifact");
    }
    let m: Vec<Vec<f64>> = serde_json::from_value(v["M"].clone()).expect("M");
    assert_eq!(m.len(), chains.len(), "M must have one row per named chain");
    for row in &m {
        assert_eq!(row.len(), chains.len(), "M must be square");
        assert!(row.iter().all(|x| x.is_finite()),
            "every entry is a log path-density and must be finite: {row:?}");
    }

    // ── The diagonal is the sampler's own number ─────────────────────────
    // M[i][i] = log p(x_i | θ_i) at the final sweep, which is exactly what that
    // sweep's trace row recorded as `transition_ll + initial_state_ll`. The
    // trace writes 4 decimal places for each of the two terms, so the tolerance
    // is the rounding, not a fudge.
    let mut any_moved = false;
    for (i, &c) in chains.iter().enumerate() {
        let (header, rows) = trace(&stage, c);
        any_moved |= theta_moved_after_burn_in(&header, &rows);
        let recorded = last_sweep_path_term(&header, &rows);
        assert!(
            (m[i][i] - recorded).abs() < 1e-3,
            "M[{i}][{i}] = {} must equal chain {c}'s final-sweep \
             transition_ll + initial_state_ll = {} from chain_{c}/trace.tsv — \
             a diagonal that drifts from the sampler's own accounting means the \
             matrix is not scoring the pair the chain actually held",
            m[i][i], recorded,
        );
    }
    assert!(
        any_moved,
        "no chain's θ moved between the first retained sweep and the last, so the \
         diagonal check above cannot distinguish the paired (θ, x) from any other \
         pairing — the fixture has gone degenerate and needs more sweeps or a \
         longer series before this test means anything"
    );

    // ── The derived numbers are functions of M ───────────────────────────
    let k = chains.len();
    let diag: f64 = (0..k).map(|i| m[i][i]).sum::<f64>() / k as f64;
    let off: f64 = (0..k).flat_map(|i| (0..k).filter(move |&j| j != i).map(move |j| (i, j)))
        .map(|(i, j)| m[i][j]).sum::<f64>() / (k * k - k) as f64;
    let asym = (0..k).flat_map(|i| (0..k).map(move |j| (i, j)))
        .map(|(i, j)| (m[i][j] - m[j][i]).abs())
        .fold(0.0_f64, f64::max);
    let reported_dom = v["diagonal_dominance"].as_f64().expect("diagonal_dominance");
    let reported_asym = v["asymmetry"].as_f64().expect("asymmetry");
    assert!((reported_dom - (diag - off)).abs() < 1e-9,
        "diagonal_dominance must be mean(diag) − mean(offdiag) = {}; got {reported_dom}",
        diag - off);
    assert!((reported_asym - asym).abs() < 1e-9,
        "asymmetry must be max |M[i][j] − M[j][i]| = {asym}; got {reported_asym}");
}

/// PMMH stores no latent path, so the artifact must be ABSENT — not an empty
/// matrix, which a consumer would read as "measured, found nothing".
#[test]
fn pmmh_writes_no_cross_chain_compatibility_matrix() {
    let bin = camdl_bin();
    if camdlc_bin().is_none() { return }
    let tmp = tempdir("pmmh");
    let (ir, data) = write_fixture(tmp.path());
    let (fit, out) = write_fit_toml(tmp.path(), &ir, &data, "pmmh", 2);
    run_fit(&bin, &fit);

    let stage = stage_leaf(&out);
    assert!(stage.join("diagnostics.json").is_file(),
        "sanity: the PMMH stage must have completed and written diagnostics.json");
    assert!(!stage.join("cross_chain_compat.json").exists(),
        "PMMH has no stored latent path to score, so cross_chain_compat.json must not \
         exist at all — an empty matrix would read as a measurement");
}
