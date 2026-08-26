//! Comprehensive surface + correctness test for generated quantities
//! (proposal 2026-06-25), driven by the committed fixture
//! `tests/fixtures/quantities/quantities_showcase.camdl`.
//!
//! Two tests over one deterministic run (chain_binomial, ChaCha8, seed 1,
//! scenario `baseline`):
//!
//! * `simulate_emits_the_full_quantity_surface` — every reduction kind, both
//!   shapes, stratification, reduction arithmetic, censoring, and the manifest,
//!   pinned against captured golden values + relational invariants.
//! * `quantities_match_independent_recomputation` — the CORRECTNESS check:
//!   re-derives every reduction from the emitted trajectory + obs series by an
//!   independent fold and asserts it equals the evaluator's output. This catches
//!   a fold-logic bug that captured-value pins (which only catch drift) would
//!   not. The obs reductions reduce the SAME `y_sim` the run drew (no redraw),
//!   so they must equal a fold over the emitted `--obs` series.

use std::path::{Path, PathBuf};
use std::process::Command;

fn binary() -> PathBuf {
    let manifest = std::env::var("CARGO_MANIFEST_DIR").unwrap();
    Path::new(&manifest).join("../../target/release/camdl")
}

/// The committed showcase fixture (repo-root `tests/fixtures/`).
fn fixture() -> PathBuf {
    let manifest = std::env::var("CARGO_MANIFEST_DIR").unwrap();
    Path::new(&manifest).join("../../../tests/fixtures/quantities/quantities_showcase.camdl")
}

/// Run the deterministic showcase, emitting the trajectory (`-o`), the obs
/// series (`--obs`), and the quantities dir. Returns `(qdir, traj, obs)`.
fn run_showcase(tmp: &Path) -> (PathBuf, PathBuf, PathBuf) {
    let bin = binary();
    assert!(bin.exists(), "release camdl missing: {} — run `make build-rust`", bin.display());
    let fixture = fixture();
    assert!(fixture.exists(), "fixture missing: {}", fixture.display());
    let qdir = tmp.join("q");
    let traj = tmp.join("traj.tsv");
    let obs = tmp.join("obs.tsv");
    let out = Command::new(&bin)
        .args([
            "simulate",
            fixture.to_str().unwrap(),
            "--scenario",
            "baseline",
            "--seed",
            "1",
            "--backend",
            "chain_binomial",
            "-o",
            traj.to_str().unwrap(),
            "--obs",
            obs.to_str().unwrap(),
            "--output-dir",
            tmp.join("results").to_str().unwrap(),
            "--quantities-out",
            qdir.to_str().unwrap(),
        ])
        .env("CAMDL_SKIP_VERSION_CHECK", "1")
        .output()
        .expect("spawn camdl");
    assert!(out.status.success(), "simulate failed:\n{}", String::from_utf8_lossy(&out.stderr));
    (qdir, traj, obs)
}

/// The point value of an unstratified scalar quantity (header `value`, then the
/// single value row).
fn scalar(qdir: &Path, name: &str) -> String {
    let txt = std::fs::read_to_string(qdir.join("quantities").join(format!("{name}.tsv")))
        .unwrap_or_else(|e| panic!("read {name}.tsv: {e}"));
    let mut lines = txt.lines();
    // This fixture runs with `--scenario baseline`, so it has a scenario axis
    // and the design coordinate leads the row (gh#562).
    assert_eq!(
        lines.next(),
        Some("scenario\tvalue"),
        "{name}: point scalar header carries the scenario coordinate"
    );
    lines
        .next()
        .unwrap_or_else(|| panic!("{name}: missing value row"))
        .rsplit('\t')
        .next()
        .expect("value field")
        .trim()
        .to_string()
}

fn scalar_f(qdir: &Path, name: &str) -> f64 {
    scalar(qdir, name).parse().unwrap_or_else(|_| panic!("{name} not a number"))
}

#[test]
fn simulate_emits_the_full_quantity_surface() {
    let tmp = tempfile::tempdir().unwrap();
    let (qdir, _traj, _obs) = run_showcase(tmp.path());

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
    // v1.1 observation source — captured golden values
    assert_eq!(scalar(&qdir, "peak_cases"), "85");
    assert_eq!(scalar(&qdir, "total_cases"), "4221");
    assert_eq!(scalar(&qdir, "cases_onset"), "28");

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
    // The run passes `--scenario baseline`, so the design coordinate leads every
    // header ahead of the stratum dims and the value (gh#562).
    let final_r = std::fs::read_to_string(qdir.join("quantities/final_R.tsv")).unwrap();
    let mut frl = final_r.lines();
    assert_eq!(frl.next(), Some("scenario\tpatch\tvalue"), "stratified scalar header");
    assert_eq!(frl.next(), Some("baseline\ta\t887"));
    assert_eq!(frl.next(), Some("baseline\tb\t764"));

    // ── Series shapes: total (time+value), stratified (time+patch+value) ──
    let total = std::fs::read_to_string(qdir.join("quantities/total_prev.tsv")).unwrap();
    assert_eq!(total.lines().next(), Some("scenario\ttime\tvalue"), "series header");
    assert_eq!(total.lines().count(), 1 + 201, "one row per snapshot");
    let prev = std::fs::read_to_string(qdir.join("quantities/prevalence.tsv")).unwrap();
    assert_eq!(
        prev.lines().next(),
        Some("scenario\ttime\tpatch\tvalue"),
        "stratified series header"
    );

    // ── Manifest: one entry per logical quantity; every Time / Derived-of-Time
    //    quantity is statically censorable ──
    let manifest: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(qdir.join("quantities.json")).unwrap()).unwrap();
    assert_eq!(manifest["schema"], "camdl.quantities/v1");
    let qs = manifest["quantities"].as_array().unwrap();
    assert_eq!(qs.len(), 25, "25 logical quantities (21 state + 4 obs; stratified families count once)");
    let censorable: std::collections::BTreeSet<&str> =
        qs.iter().filter(|q| q["censoring"].is_object()).map(|q| q["name"].as_str().unwrap()).collect();
    let expected: std::collections::BTreeSet<&str> = [
        "peak_t", "trough_t", "onset", "fadeout", "first_lo", "last_lo", "big_t", "never",
        "outbreak_dur", "half_dur", // Derived transitively referencing a Time scalar
        "cases_onset",              // obs-source Time reduction
        "prev_at_50", "cases_at_28", "late", // value_at censors out-of-window
    ].into_iter().collect();
    assert_eq!(
        censorable, expected,
        "exactly the Time, Derived-of-Time, and value_at quantities are censorable"
    );
}

// ── Independent recomputation oracle ─────────────────────────────────────────
// These mirror `sim::quantity`'s fold semantics EXACTLY (read off the evaluator):
// max/min over finite values, mean over all, strict-inequality counts/crossings,
// argmax first-on-ties, trapezoid integral. The test asserts the quantity output
// equals an independent fold of the SAME emitted series — so a fold-logic bug
// surfaces here even when the captured value happens to look plausible.

fn max_finite(s: &[f64]) -> f64 {
    s.iter().copied().filter(|v| v.is_finite()).fold(f64::NEG_INFINITY, f64::max)
}
fn min_finite(s: &[f64]) -> f64 {
    s.iter().copied().filter(|v| v.is_finite()).fold(f64::INFINITY, f64::min)
}
fn mean_all(s: &[f64]) -> f64 {
    s.iter().sum::<f64>() / s.len() as f64
}
fn count_strict(s: &[f64], thr: f64, gt: bool) -> usize {
    s.iter().filter(|&&v| v.is_finite() && if gt { v > thr } else { v < thr }).count()
}
/// First index of the maximum, strict `>` (ties keep the first), non-finite skipped.
fn argmax_first(s: &[f64]) -> Option<usize> {
    let mut best: Option<(usize, f64)> = None;
    for (i, &v) in s.iter().enumerate() {
        if v.is_finite() && best.is_none_or(|(_, b)| v > b) {
            best = Some((i, v));
        }
    }
    best.map(|(i, _)| i)
}
fn argmin_first(s: &[f64]) -> Option<usize> {
    let mut best: Option<(usize, f64)> = None;
    for (i, &v) in s.iter().enumerate() {
        if v.is_finite() && best.is_none_or(|(_, b)| v < b) {
            best = Some((i, v));
        }
    }
    best.map(|(i, _)| i)
}
/// First/last time the (finite) series crosses the threshold (strict); `None`
/// if it never does (the evaluator right-censors → `NA`).
fn cross_time(s: &[f64], t: &[f64], thr: f64, gt: bool, first: bool) -> Option<f64> {
    let mut hit = None;
    for (i, &v) in s.iter().enumerate() {
        if v.is_finite() && if gt { v > thr } else { v < thr } {
            hit = Some(t[i]);
            if first {
                break;
            }
        }
    }
    hit
}
fn trapezoid(s: &[f64], t: &[f64]) -> f64 {
    let n = s.len().min(t.len());
    (0..n.saturating_sub(1)).map(|i| 0.5 * (s[i] + s[i + 1]) * (t[i + 1] - t[i])).sum()
}

/// Parse a trajectory TSV (skips `#` comment lines), returning `(header, rows)`
/// where each row is the column values keyed by name.
fn read_tsv(path: &Path) -> (Vec<String>, Vec<Vec<f64>>) {
    let txt = std::fs::read_to_string(path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
    let mut lines = txt.lines().filter(|l| !l.starts_with('#') && !l.trim().is_empty());
    let header: Vec<String> = lines.next().expect("header").split('\t').map(|s| s.to_string()).collect();
    let rows: Vec<Vec<f64>> = lines
        .map(|l| l.split('\t').map(|c| c.parse::<f64>().unwrap_or(f64::NAN)).collect())
        .collect();
    (header, rows)
}

fn column(header: &[String], rows: &[Vec<f64>], name: &str) -> Vec<f64> {
    let idx = header.iter().position(|c| c == name).unwrap_or_else(|| panic!("no column `{name}` in {header:?}"));
    rows.iter().map(|r| r[idx]).collect()
}

/// A scalar quantity is `NA` (censored) iff the emitted value parses as such.
fn scalar_opt(qdir: &Path, name: &str) -> Option<f64> {
    match scalar(qdir, name).as_str() {
        "NA" => None,
        s => Some(s.parse().unwrap_or_else(|_| panic!("{name}: not a number or NA"))),
    }
}

fn assert_eq_f(label: &str, got: f64, want: f64) {
    assert!((got - want).abs() < 1e-6, "{label}: quantity={got} != recomputed={want}");
}

#[test]
fn quantities_match_independent_recomputation() {
    let tmp = tempfile::tempdir().unwrap();
    let (qdir, traj, obs) = run_showcase(tmp.path());

    // Rebuild the state series the evaluator folded, from the emitted trajectory.
    let (h, rows) = read_tsv(&traj);
    let t = column(&h, &rows, "t");
    let (sa, sb) = (column(&h, &rows, "S_a"), column(&h, &rows, "S_b"));
    let (ia, ib) = (column(&h, &rows, "I_a"), column(&h, &rows, "I_b"));
    let (ra, rb) = (column(&h, &rows, "R_a"), column(&h, &rows, "R_b"));
    let n = t.len();
    let i_total: Vec<f64> = (0..n).map(|i| ia[i] + ib[i]).collect();
    let n_tot: Vec<f64> = (0..n).map(|i| sa[i] + sb[i] + ia[i] + ib[i] + ra[i] + rb[i]).collect();
    let prev: Vec<f64> = (0..n).map(|i| i_total[i] / n_tot[i]).collect();
    let thr = 50.0;

    // Value reductions over the prevalence / I_total series.
    assert_eq_f("final_prev≈total_prev.last", *prev.last().unwrap(), prev[n - 1]);
    assert_eq_f("peak_prev", scalar_f(&qdir, "peak_prev"), max_finite(&prev));
    assert_eq_f("trough_prev", scalar_f(&qdir, "trough_prev"), min_finite(&prev));
    assert_eq_f("mean_prev", scalar_f(&qdir, "mean_prev"), mean_all(&prev));
    assert_eq_f("high_days", scalar_f(&qdir, "high_days"), count_strict(&i_total, thr, true) as f64);
    assert_eq_f("low_days", scalar_f(&qdir, "low_days"), count_strict(&i_total, thr, false) as f64);
    assert_eq_f("person_days", scalar_f(&qdir, "person_days"), trapezoid(&i_total, &t));

    // Time reductions.
    assert_eq_f("peak_t", scalar_f(&qdir, "peak_t"), t[argmax_first(&i_total).unwrap()]);
    assert_eq_f("trough_t", scalar_f(&qdir, "trough_t"), t[argmin_first(&i_total).unwrap()]);
    assert_eq!(scalar_opt(&qdir, "onset"), cross_time(&i_total, &t, thr, true, true), "onset");
    assert_eq!(scalar_opt(&qdir, "fadeout"), cross_time(&i_total, &t, thr, true, false), "fadeout");
    assert_eq!(scalar_opt(&qdir, "first_lo"), cross_time(&i_total, &t, thr, false, true), "first_lo");
    assert_eq!(scalar_opt(&qdir, "last_lo"), cross_time(&i_total, &t, thr, false, false), "last_lo");
    // never-crossing thresholds censor to NA, not a fabricated time.
    assert_eq!(scalar_opt(&qdir, "big_t"), cross_time(&i_total, &t, 0.5 * n_tot[0], true, true), "big_t");
    assert_eq!(scalar_opt(&qdir, "never"), None, "never");

    // value_at: LOCF read — the value at the last output time <= the anchor;
    // censored past the window (proposal 2026-08-17). Recomputed with an
    // INDEPENDENT scan (not partition_point) over the emitted trajectory.
    let locf = |series: &[f64], times: &[f64], anchor: f64| -> Option<f64> {
        let mut out = None;
        for i in 0..times.len() {
            if times[i] <= anchor {
                out = Some(series[i]);
            }
        }
        if anchor < times[0] || anchor > *times.last().unwrap() { None } else { out }
    };
    assert_eq_f("prev_at_50", scalar_f(&qdir, "prev_at_50"), locf(&i_total, &t, 50.0).unwrap());
    // `late` reads past t_end=200 — censored to NA, never clamped to final().
    assert_eq!(scalar_opt(&qdir, "late"), None, "late must censor, not clamp");

    // Derived arithmetic over prior scalars.
    let (onset, fadeout) = (scalar_f(&qdir, "onset"), scalar_f(&qdir, "fadeout"));
    assert_eq_f("outbreak_dur", scalar_f(&qdir, "outbreak_dur"), fadeout - onset);
    assert_eq_f("half_dur", scalar_f(&qdir, "half_dur"), (fadeout - onset) / 2.0);

    // ── Observation source: reduce the SAME y_sim the run published ──────────
    // The obs reductions must equal a fold over the emitted `--obs` series — the
    // load-bearing "no redraw" invariant, at the whole-surface level.
    let (oh, orows) = read_tsv(&obs);
    let ot = column(&oh, &orows, "time");
    let cases = column(&oh, &orows, "cases");
    assert_eq_f("peak_cases", scalar_f(&qdir, "peak_cases"), max_finite(&cases));
    assert_eq_f("total_cases", scalar_f(&qdir, "total_cases"), trapezoid(&cases, &ot));
    assert_eq!(scalar_opt(&qdir, "cases_onset"), cross_time(&cases, &ot, thr, true, true), "cases_onset");
    // value_at over the obs source reads on the STREAM's own time axis
    // (every 14 days), so the anchor 28 lands exactly on its second emission.
    let locf_obs = {
        let mut out = None;
        for i in 0..ot.len() {
            if ot[i] <= 28.0 {
                out = Some(cases[i]);
            }
        }
        out.unwrap()
    };
    assert_eq_f("cases_at_28", scalar_f(&qdir, "cases_at_28"), locf_obs);
}
