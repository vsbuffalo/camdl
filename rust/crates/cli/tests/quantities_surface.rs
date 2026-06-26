//! Comprehensive surface test for generated quantities (proposal 2026-06-25):
//! one showcase model exercising EVERY reduction kind, both shapes (series +
//! scalar), stratification, reduction arithmetic, a state-dependent threshold,
//! and a never-firing (censored) timing — simulated deterministically and
//! checked against captured golden values PLUS relational invariants (which catch
//! a logic bug independent of the exact realization).

use std::path::{Path, PathBuf};
use std::process::Command;

fn binary() -> PathBuf {
    let manifest = std::env::var("CARGO_MANIFEST_DIR").unwrap();
    Path::new(&manifest).join("../../target/release/camdl")
}

/// A 2-patch SIR whose `quantities {}` block touches the whole v1 surface. The
/// baseline scenario fixes params (R0 = 2.5 / 1.8) for a deterministic run.
const MODEL: &str = r#"
time_unit = 'days

dimensions { patch = [a, b] }
compartments { S, I, R }
stratify(by = patch)

parameters {
  gamma     : positive in [0.01, 1.0]
  N[patch]  : positive in [100, 100000]
  R0[patch] : positive in [0.5, 10.0]
  thr       : count    in [1, 100000]
}

let beta[p in patch] = R0[p] * gamma
let I_total = sum(p in patch, I[p])
let N_tot   = sum(p in patch, S[p] + I[p] + R[p])

transitions {
  infection[p in patch] : S[p] --> I[p] @ beta[p] * S[p] * I[p] / (S[p] + I[p] + R[p])
  recovery[p in patch]  : I[p] --> R[p] @ gamma * I[p]
}

init {
  S[a] = 990  I[a] = 10  R[a] = 0
  S[b] = 980  I[b] = 20  R[b] = 0
}

quantities {
  prevalence[p in patch] = I[p] / (S[p] + I[p] + R[p])
  total_prev             = I_total / N_tot
  peak_prev   = max(I_total / N_tot)
  trough_prev = min(I_total / N_tot)
  mean_prev   = mean(I_total / N_tot)
  final_R[p in patch] = final(R[p])
  high_days   = count_above(I_total, thr)
  low_days    = count_below(I_total, thr)
  person_days = integral(I_total)
  peak_t   = time_of_max(I_total)
  trough_t = time_of_min(I_total)
  onset    = first_above(I_total, thr)
  fadeout  = last_above(I_total, thr)
  first_lo = first_below(I_total, thr)
  last_lo  = last_below(I_total, thr)
  big_t    = first_above(I_total, 0.5 * N_tot)
  never    = first_above(I_total, 9000000)
  outbreak_dur = fadeout - onset
  half_dur     = outbreak_dur / 2
}

simulate { from = 0  to = 200 }

scenarios {
  baseline {
    label = "showcase"
    set = { gamma = 0.1  N[a] = 1000  N[b] = 1000  R0[a] = 2.5  R0[b] = 1.8  thr = 50 }
  }
}
"#;

fn run(bin: &Path, args: &[&str]) -> std::process::Output {
    Command::new(bin)
        .args(args)
        .env("CAMDL_SKIP_VERSION_CHECK", "1")
        .output()
        .expect("spawn camdl")
}

/// The point value of an unstratified scalar quantity (header `value`, then the
/// single value row).
fn scalar(qdir: &Path, name: &str) -> String {
    let txt = std::fs::read_to_string(qdir.join("quantities").join(format!("{name}.tsv")))
        .unwrap_or_else(|e| panic!("read {name}.tsv: {e}"));
    let mut lines = txt.lines();
    assert_eq!(lines.next(), Some("value"), "{name}: point scalar header");
    lines.next().unwrap_or_else(|| panic!("{name}: missing value row")).trim().to_string()
}

fn scalar_f(qdir: &Path, name: &str) -> f64 {
    scalar(qdir, name).parse().unwrap_or_else(|_| panic!("{name} not a number"))
}

#[test]
fn simulate_emits_the_full_quantity_surface() {
    let bin = binary();
    assert!(bin.exists(), "release camdl missing: {} — run `make build-rust`", bin.display());
    let tmp = tempfile::tempdir().unwrap();
    let model = tmp.path().join("showcase.camdl");
    std::fs::write(&model, MODEL).unwrap();
    let qdir = tmp.path().join("q");

    let out = run(&bin, &[
        "simulate", model.to_str().unwrap(),
        "--scenario", "baseline", "--seed", "1", "--backend", "chain_binomial",
        "--output-dir", tmp.path().join("results").to_str().unwrap(),
        "--quantities-out", qdir.to_str().unwrap(),
    ]);
    assert!(out.status.success(), "simulate failed:\n{}", String::from_utf8_lossy(&out.stderr));

    // ── Captured golden values (deterministic: chain_binomial, ChaCha8, seed 1) ──
    assert_eq!(scalar(&qdir, "peak_prev"), "0.186");
    assert_eq!(scalar(&qdir, "trough_prev"), "0");
    assert_eq!(scalar(&qdir, "mean_prev"), "0.044214");
    assert_eq!(scalar(&qdir, "high_days"), "84");
    assert_eq!(scalar(&qdir, "low_days"), "115");
    assert_eq!(scalar(&qdir, "person_days"), "17759");
    assert_eq!(scalar(&qdir, "peak_t"), "44");
    assert_eq!(scalar(&qdir, "trough_t"), "156");
    assert_eq!(scalar(&qdir, "onset"), "8");
    assert_eq!(scalar(&qdir, "fadeout"), "91");
    assert_eq!(scalar(&qdir, "first_lo"), "0");
    assert_eq!(scalar(&qdir, "last_lo"), "200");
    assert_eq!(scalar(&qdir, "outbreak_dur"), "83");
    assert_eq!(scalar(&qdir, "half_dur"), "41.5");

    // ── Censoring: a crossing that never happens is NA, not a fabricated time ──
    assert_eq!(scalar(&qdir, "big_t"), "NA", "I_total never reaches 50% of N");
    assert_eq!(scalar(&qdir, "never"), "NA", "threshold 9e6 never crossed");

    // ── Relational invariants (independent of the exact realization) ──
    assert_eq!(scalar_f(&qdir, "outbreak_dur"), scalar_f(&qdir, "fadeout") - scalar_f(&qdir, "onset"));
    assert_eq!(scalar_f(&qdir, "half_dur"), scalar_f(&qdir, "outbreak_dur") / 2.0);
    let (onset, peak, fade) = (scalar_f(&qdir, "onset"), scalar_f(&qdir, "peak_t"), scalar_f(&qdir, "fadeout"));
    assert!(onset <= peak && peak <= fade, "onset {onset} <= peak {peak} <= fadeout {fade}");
    let (peak_p, trough_p, mean_p) = (scalar_f(&qdir, "peak_prev"), scalar_f(&qdir, "trough_prev"), scalar_f(&qdir, "mean_prev"));
    assert!(trough_p <= mean_p && mean_p <= peak_p, "trough <= mean <= peak prevalence");
    let n_snap = 201.0; // t = 0..=200 at dt = 1
    assert!(scalar_f(&qdir, "high_days") + scalar_f(&qdir, "low_days") <= n_snap);

    // ── Stratified scalar: one row per patch ──
    let final_r = std::fs::read_to_string(qdir.join("quantities/final_R.tsv")).unwrap();
    let mut frl = final_r.lines();
    assert_eq!(frl.next(), Some("patch\tvalue"), "stratified scalar header");
    assert_eq!(frl.next(), Some("a\t887"));
    assert_eq!(frl.next(), Some("b\t764"));

    // ── Series shapes: total (time+value), stratified (time+patch+value) ──
    let total = std::fs::read_to_string(qdir.join("quantities/total_prev.tsv")).unwrap();
    assert_eq!(total.lines().next(), Some("time\tvalue"), "series header");
    assert_eq!(total.lines().count(), 1 + 201, "one row per snapshot");
    let prev = std::fs::read_to_string(qdir.join("quantities/prevalence.tsv")).unwrap();
    assert_eq!(prev.lines().next(), Some("time\tpatch\tvalue"), "stratified series header");

    // ── Manifest: one entry per logical quantity; every Time / Derived-of-Time
    //    quantity is statically censorable ──
    let manifest: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(qdir.join("quantities.json")).unwrap()).unwrap();
    assert_eq!(manifest["schema"], "camdl.quantities/v1");
    let qs = manifest["quantities"].as_array().unwrap();
    assert_eq!(qs.len(), 19, "19 logical quantities (stratified families count once)");
    let censorable: std::collections::BTreeSet<&str> =
        qs.iter().filter(|q| q["censoring"].is_object()).map(|q| q["name"].as_str().unwrap()).collect();
    let expected: std::collections::BTreeSet<&str> = [
        "peak_t", "trough_t", "onset", "fadeout", "first_lo", "last_lo", "big_t", "never",
        "outbreak_dur", "half_dur", // Derived transitively referencing a Time scalar
    ].into_iter().collect();
    assert_eq!(censorable, expected, "exactly the Time + Derived-of-Time quantities are censorable");
}
