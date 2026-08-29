//! Cross-validate camdl's prequential scoring-rule kernels against external
//! references (gh#628, gh#629): sample CRPS (edf and fair estimators), the
//! mixture log score, and the randomized PIT.
//!
//! The fixture `prequential_scores_ref.tsv` is committed, so this test is
//! offline and CI never needs R. Regenerate with
//! `Rscript scripts/gen_prequential_scores_fixture.R`.
//!
//! Two evidence axes, per row (`axis` column):
//!
//! * **Exact agreement** (`tol = 1e-12`): the edf CRPS against
//!   `scoringRules::crps_sample(method = "edf")` — an independent published
//!   implementation of the same estimator, so agreement is to machine
//!   precision; the fair CRPS and randomized PIT against their definitional
//!   forms (the O(S²) pairwise sum; the literal atom split), computed in R
//!   with no shared shortcut with camdl's order-statistic and counting code;
//!   the mixture log score against R density values combined by an
//!   independently written log-mean-exp.
//! * **Closed form** (`crps_closed_form[_hist]`): large committed ensembles
//!   drawn from Poisson / negative-binomial / normal predictives, scored
//!   against `crps_pois` / `crps_nbinom` / `crps_norm` analytic values
//!   within a stated Monte Carlo tolerance the generator verifies before
//!   writing. This axis knows nothing about sample estimators at all, so it
//!   checks the estimator, not the arithmetic. Count ensembles ride as
//!   `value:count` histograms (CRPS is permutation-invariant, so the
//!   ensemble reconstructs exactly at n = 1e5 for a few dozen tokens).
//!
//! The generator does not get to assert its own reference forms: before
//! writing it checks the fair/edf normalization identity on every exact
//! row, the point-mass mixture against the parametric density, and the
//! randomized PIT's calibration on a Poisson forecast, aborting on failure.

use sim::inference::prequential::{
    crps_sample, crps_sample_fair, log_score_plug_in, pit_sample_randomized,
};
use std::path::{Path, PathBuf};

fn fixture(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures").join(name)
}

struct Row {
    case: String,
    axis: String,
    y: f64,
    oracle_a: f64,
    oracle_b: f64,
    oracle_c: f64,
    tol: f64,
    samples: Vec<f64>,
}

fn parse_opt(s: &str) -> f64 {
    if s == "NA" { f64::NAN } else { s.parse().expect("numeric field") }
}

fn load() -> Vec<Row> {
    let path = fixture("prequential_scores_ref.tsv");
    let text = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
    let mut rows = Vec::new();
    for line in text.lines() {
        if line.starts_with('#') || line.starts_with("case\t") || line.trim().is_empty() {
            continue;
        }
        let f: Vec<&str> = line.split('\t').collect();
        assert_eq!(f.len(), 8, "malformed fixture row: {line}");
        let samples: Vec<f64> = if f[1] == "crps_closed_form_hist" {
            // value:count histogram — expand to the full ensemble.
            f[7].split(',').flat_map(|vc| {
                let (v, c) = vc.split_once(':').expect("value:count entry");
                let v: f64 = v.parse().expect("histogram value");
                let c: usize = c.parse().expect("histogram count");
                std::iter::repeat_n(v, c)
            }).collect()
        } else {
            f[7].split(',').map(|v| v.parse().expect("sample value")).collect()
        };
        rows.push(Row {
            case: f[0].to_string(),
            axis: f[1].to_string(),
            y: parse_opt(f[2]),
            oracle_a: parse_opt(f[3]),
            oracle_b: parse_opt(f[4]),
            oracle_c: parse_opt(f[5]),
            tol: f[6].parse().expect("tol"),
            samples,
        });
    }
    assert!(rows.len() >= 20, "fixture unexpectedly short: {} rows", rows.len());
    rows
}

#[test]
fn scoring_kernels_match_external_oracles() {
    let mut by_axis: std::collections::BTreeMap<String, usize> = Default::default();
    for r in load() {
        *by_axis.entry(r.axis.clone()).or_default() += 1;
        match r.axis.as_str() {
            "crps_exact" => {
                let edf = crps_sample(&r.samples, r.y);
                let fair = crps_sample_fair(&r.samples, r.y);
                assert!((edf - r.oracle_a).abs() <= r.tol,
                    "{}: edf CRPS {edf} vs scoringRules {}", r.case, r.oracle_a);
                assert!((fair - r.oracle_b).abs() <= r.tol,
                    "{}: fair CRPS {fair} vs pairwise reference {}", r.case, r.oracle_b);
            }
            "pit_exact" => {
                for (v, want, label) in [
                    (0.0, r.oracle_a, "v=0"),
                    (0.5, r.oracle_b, "v=0.5"),
                    (1.0, r.oracle_c, "v=1"),
                ] {
                    let got = pit_sample_randomized(&r.samples, r.y, v);
                    assert!((got - want).abs() <= r.tol,
                        "{}: randomized PIT at {label}: {got} vs {want}", r.case);
                }
            }
            "logscore" => {
                // The kernel's uniform-weight path: unnormalized zero
                // log-weights, per-particle log-likelihoods from the fixture.
                let lw = vec![0.0; r.samples.len()];
                let got = log_score_plug_in(&r.samples, &lw);
                assert!((got - r.oracle_a).abs() <= r.tol.max(r.oracle_a.abs() * 1e-14),
                    "{}: mixture log score {got} vs reference {}", r.case, r.oracle_a);
            }
            "crps_closed_form" | "crps_closed_form_hist" => {
                let edf = crps_sample(&r.samples, r.y);
                let fair = crps_sample_fair(&r.samples, r.y);
                assert!((edf - r.oracle_a).abs() <= r.tol,
                    "{}: edf CRPS {edf} vs analytic {} (tol {})",
                    r.case, r.oracle_a, r.tol);
                assert!((fair - r.oracle_a).abs() <= r.tol,
                    "{}: fair CRPS {fair} vs analytic {} (tol {})",
                    r.case, r.oracle_a, r.tol);
            }
            other => panic!("unknown fixture axis: {other}"),
        }
    }
    // Every evidence axis must actually be present — a truncated or
    // mis-generated fixture must not pass vacuously.
    for axis in ["crps_exact", "pit_exact", "logscore",
                 "crps_closed_form", "crps_closed_form_hist"] {
        assert!(by_axis.contains_key(axis), "fixture carries no {axis} rows");
    }
}
