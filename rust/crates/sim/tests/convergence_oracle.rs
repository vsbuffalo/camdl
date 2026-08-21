//! Cross-validate camdl's rank-normalized convergence diagnostics against the
//! R package `posterior` 1.7.0 — the implementation maintained by the authors
//! of Vehtari, Gelman, Simpson, Carpenter & Bürkner (2021), _Bayesian
//! Analysis_ 16(2):667-718, and the same algorithm Stan reports.
//!
//! Both halves of the fixture are committed: `convergence_chains.tsv` holds
//! the draws both sides score, `convergence_posterior_ref.tsv` holds
//! posterior's statistics on exactly those draws. Regenerate with
//! `Rscript scripts/gen_convergence_posterior_fixture.R`; CI never needs R.
//!
//! The oracle is external on purpose. A second camdl-side implementation of
//! the same paper would agree with itself about anything it misread — that
//! ranks are averaged within tied groups, that the Blom offset denominator is
//! `S − 2·(3/8) + 1` and not `S − 2·(3/8)`, that Geyer's truncated estimator
//! keeps ρ̂₀ when the first pair sum is non-positive. Each of those was checked
//! against `posterior`'s source and is worth a case here.
//!
//! Case coverage, and what each is for:
//!
//! | case                 | what it pins                                        |
//! | -------------------- | --------------------------------------------------- |
//! | `ar1_mixed`          | the ordinary well-mixed baseline                     |
//! | `within_chain_drift` | split-R̂ vs classic: 1.428 vs 1.001 on one fit       |
//! | `scale_disagree`     | the FOLDED branch — same location, different spread  |
//! | `rwm_ties`           | exact repeats from rejected proposals (PMMH)         |
//! | `heavy_tail`         | a right-skewed marginal where classic R̂ misreads    |
//! | `odd_draws`          | the split convention drops the middle draw           |
//! | `antithetic`         | negative autocorrelation; the τ̂ cap fires           |
//! | `one_stuck_chain`    | a chain with exactly zero within-chain variance      |
//! | `lower_bound_pileup` | ~half the draws tied at a bound                      |
//! | `upper_bound_pileup` | a constant 95% indicator: tail-ESS is undefined      |
//! | `two_chains_short`   | the smallest accepted input; the `max_t == 0` branch |
//! | `all_constant`       | the degenerate-variance refusal                      |

use sim::inference::convergence::{rank_convergence, ConvergenceError};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

fn fixture(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures").join(name)
}

/// `case → chains[chain][draw]`.
fn load_chains() -> BTreeMap<String, Vec<Vec<f64>>> {
    let path = fixture("convergence_chains.tsv");
    let text = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("cannot read {}: {}", path.display(), e));
    let mut out: BTreeMap<String, Vec<Vec<f64>>> = BTreeMap::new();
    for line in text.lines() {
        if line.starts_with('#') || line.starts_with("case\t") || line.trim().is_empty() {
            continue;
        }
        let f: Vec<&str> = line.split('\t').collect();
        assert_eq!(f.len(), 4, "expected 4 columns, got {:?}", f);
        let chain: usize = f[1].parse().unwrap();
        let draw: usize = f[2].parse().unwrap();
        let value: f64 = f[3].parse().unwrap();
        let case = out.entry(f[0].to_string()).or_default();
        if case.len() <= chain {
            case.resize(chain + 1, Vec::new());
        }
        assert_eq!(case[chain].len(), draw, "draws must arrive in file order");
        case[chain].push(value);
    }
    out
}

/// `(case, statistic) → value`, `None` where posterior reports `NA`.
fn load_reference() -> BTreeMap<(String, String), Option<f64>> {
    let path = fixture("convergence_posterior_ref.tsv");
    let text = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("cannot read {}: {}", path.display(), e));
    let mut out = BTreeMap::new();
    for line in text.lines() {
        if line.starts_with('#') || line.starts_with("case\t") || line.trim().is_empty() {
            continue;
        }
        let f: Vec<&str> = line.split('\t').collect();
        assert_eq!(f.len(), 3, "expected 3 columns, got {:?}", f);
        let v = if f[2] == "NA" { None } else { Some(f[2].parse::<f64>().unwrap()) };
        out.insert((f[0].to_string(), f[1].to_string()), v);
    }
    out
}

fn reference(r: &BTreeMap<(String, String), Option<f64>>, case: &str, stat: &str) -> Option<f64> {
    *r.get(&(case.to_string(), stat.to_string()))
        .unwrap_or_else(|| panic!("fixture has no `{stat}` for case `{case}`"))
}

/// Relative agreement. `1e-9` is roughly the accuracy of the Acklam inverse
/// normal CDF camdl uses where R uses its own `qnorm`, and is two orders
/// tighter than any difference a convention error would produce.
fn assert_close(case: &str, stat: &str, got: f64, want: f64) {
    let rel = (got - want).abs() / want.abs().max(1.0);
    assert!(
        rel < 1e-9,
        "{case}/{stat}: camdl = {got}, posterior 1.7.0 = {want} (rel {rel:.3e})"
    );
}

#[test]
fn rhat_matches_posterior() {
    let (chains, r) = (load_chains(), load_reference());
    let mut checked = 0;
    for (case, cs) in &chains {
        match (rank_convergence(cs), reference(&r, case, "rhat")) {
            (Ok(d), Some(want)) => {
                assert_close(case, "rhat", d.rhat, want);
                checked += 1;
            }
            (Err(e), None) => {
                // posterior returns NA exactly where camdl refuses by name.
                assert!(matches!(e, ConvergenceError::ConstantDraws { .. }),
                    "{case}: posterior NA, camdl refused with {e}");
            }
            (Ok(d), None) => panic!("{case}: posterior reports NA, camdl reports {}", d.rhat),
            (Err(e), Some(want)) => panic!("{case}: posterior reports {want}, camdl refused: {e}"),
        }
    }
    assert!(checked >= 11, "expected ≥11 scored cases, got {checked}");
}

/// The bulk and folded halves must each match, not just their maximum — a
/// `max` can hide a wrong component whenever the other one is larger.
/// `scale_disagree` is the case that separates them: chains that agree on
/// location (`rhat_bulk` ≈ 1.00) and disagree on spread (`rhat` = 1.31).
#[test]
fn folded_and_bulk_rhat_are_separately_correct() {
    let (chains, r) = (load_chains(), load_reference());
    for (case, cs) in &chains {
        let Ok(d) = rank_convergence(cs) else { continue };
        let want = reference(&r, case, "rhat").expect("scored case has a reference");
        assert_close(case, "rhat", d.rhat_bulk.max(d.rhat_folded), want);
        assert!(d.rhat_bulk.is_finite(), "{case}: bulk R̂ must be finite");
    }
    let d = rank_convergence(&chains["scale_disagree"]).unwrap();
    assert!(d.rhat_bulk < 1.01,
        "chains agreeing on location must have bulk R̂ ≈ 1, got {}", d.rhat_bulk);
    assert!(d.rhat_folded > 1.3,
        "chains disagreeing on SCALE must be caught by the folded statistic, got {}",
        d.rhat_folded);
    assert!((d.rhat - d.rhat_folded).abs() < 1e-12,
        "the headline must be the folded value here");
}

#[test]
fn ess_bulk_matches_posterior() {
    let (chains, r) = (load_chains(), load_reference());
    let mut checked = 0;
    for (case, cs) in &chains {
        let Ok(d) = rank_convergence(cs) else { continue };
        let want = reference(&r, case, "ess_bulk").expect("scored case has a reference");
        assert_close(case, "ess_bulk", d.ess_bulk, want);
        checked += 1;
    }
    assert!(checked >= 11, "expected ≥11 scored cases, got {checked}");
}

#[test]
fn ess_tail_matches_posterior_including_where_it_is_undefined() {
    let (chains, r) = (load_chains(), load_reference());
    let mut undefined = 0;
    for (case, cs) in &chains {
        let Ok(d) = rank_convergence(cs) else { continue };
        match reference(&r, case, "ess_tail") {
            Some(want) => assert_close(case, "ess_tail", d.ess_tail, want),
            None => {
                assert!(d.ess_tail.is_nan(),
                    "{case}: posterior has no tail-ESS, camdl reports {}", d.ess_tail);
                undefined += 1;
            }
        }
    }
    assert_eq!(undefined, 1,
        "exactly one fixture case (upper_bound_pileup) has an undefined tail-ESS");
}

/// The whole point of the change. `within_chain_drift` is the pattern measured
/// on the ebola 8-chain PGAS fit in gh#84: chain MEANS agree, so the classic
/// Gelman & Rubin statistic reads 1.0008 — inside the healthy band
/// `docs/workflow.md` publishes — while each chain drifts across its own run
/// and the rank-normalized split statistic reads 1.4280.
#[test]
fn split_and_rank_normalization_catch_what_classic_rhat_misses() {
    let (chains, r) = (load_chains(), load_reference());
    let classic = reference(&r, "within_chain_drift", "rhat_classic").unwrap();
    let want = reference(&r, "within_chain_drift", "rhat").unwrap();
    assert!(classic < 1.05, "fixture premise: classic R̂ is inside the healthy band");
    assert!(want > 1.4, "fixture premise: the rank statistic is not");
    let d = rank_convergence(&chains["within_chain_drift"]).unwrap();
    assert_close("within_chain_drift", "rhat", d.rhat, want);
}

/// Bulk-ESS reported as a fraction of the draws it came from — the number that
/// says when to distrust the estimator itself (gh#84, requirement 4).
#[test]
fn ess_bulk_ratio_is_the_reported_fraction() {
    let chains = load_chains();
    let d = rank_convergence(&chains["within_chain_drift"]).unwrap();
    assert_eq!(d.n_draws_total, 4 * 250);
    assert!((d.ess_bulk_ratio() - d.ess_bulk / 1000.0).abs() < 1e-12);
    // 8.3 of 1000 draws: the integrated autocorrelation time is a tenth of the
    // whole run, which is the regime the ratio exists to expose.
    assert!(d.ess_bulk_ratio() < 0.01,
        "drifting chains must report a tiny ESS/N, got {}", d.ess_bulk_ratio());
}
