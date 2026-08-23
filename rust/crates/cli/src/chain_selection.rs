//! Read-side chain selection (`--exclude-chains`).
//!
//! A completed Bayesian fit is a cloud of posterior draws concatenated across
//! MCMC chains, written to `draws.tsv` with a leading `chain` key column. When a
//! near-unidentified parameter carves out a flat ridge, a minority of chains can
//! wander into a non-representative side mode; the per-chain diagnostics (gh#406)
//! name them. This module is the escape hatch: drop the diagnosed chains from the
//! posterior cloud on the READ side, at `fit predict` / `fit summary` /
//! `fit table` time — never re-running the fit, never re-keying its CAS identity.
//!
//! ## The one filter, once
//!
//! [`ChainSelection::apply_keyed`] is the single place a chain filter is applied
//! to a draws cloud. Every read-side consumer that offers `--exclude-chains`
//! routes its cloud through it (predict via [`crate::posterior_draws`], summary's
//! subset recompute), so the drop set is validated and applied identically —
//! there is no per-command re-implementation to drift.
//!
//! ## Parse at the boundary
//!
//! The CLI string `"3,5"` is parsed ONCE, by [`ChainSelection::parse_exclude`],
//! into a validated set of 1-based chain ids. Downstream holds the typed
//! [`ChainSelection`] and never re-parses a string.
//!
//! ## 1-based UI, 0-based file
//!
//! Chains are named 1-based everywhere the user sees them (`chain_N/` dirs, the
//! per-chain summary table, `--exclude-chains 3,5`). The `chain` column IN
//! `draws.tsv` is 0-based (it is the join key to `trajectories.tsv`;
//! `fit/pgas.rs`). This module owns the single conversion: a user id `k` matches
//! a draw row whose `chain` field is `k - 1`.
//!
//! ## Loaded gun (why the guardrails)
//!
//! Post-hoc chain exclusion can bias a posterior (cherry-picking modes). R̂ > 1.1
//! with chains in different modes is the model telling you something true. So a
//! selection is never silent: [`warn_active_selection`] prints a loud,
//! non-quietable warning naming the dropped chains, and the primary remedy
//! (is a parameter unidentified?) is surfaced first.

use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;

use crate::KeyedDraw;

/// A validated, 1-based set of MCMC chain ids to DROP from a posterior cloud.
///
/// Constructed only through [`ChainSelection::parse_exclude`] (the boundary
/// parser) — its existence proves the ids are positive integers and the set is
/// non-empty. Whether the ids actually exist in a given fit is checked at apply
/// time (that needs the fit's chain set), by [`ChainSelection::apply_keyed`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ChainSelection {
    /// 1-based chain ids to exclude. Non-empty by construction.
    excluded: BTreeSet<usize>,
}

/// What a chain selection actually did to a cloud — the provenance record.
///
/// All ids are 1-based (the user-facing convention). `n_total` is the number of
/// distinct chains that were present BEFORE filtering, so `kept.len() +
/// (excluded that were present) == n_total`. Stamped into `predictive.json` and
/// printed in the `fit summary` header so a chain-subset artifact is never
/// mistakable for a full-cloud one.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SubsetInfo {
    /// Chains dropped (1-based, sorted). Exactly the requested exclusion set —
    /// validated to be present, so every id here really was in the fit.
    pub excluded: Vec<usize>,
    /// Chains retained (1-based, sorted).
    pub kept: Vec<usize>,
    /// Distinct chains present before filtering (`kept.len() + excluded.len()`).
    pub n_total: usize,
}

impl ChainSelection {
    /// Parse `--exclude-chains 3,5` → the drop set `{3, 5}`.
    ///
    /// Rejects (hard error, never a silent shrug): an empty list, a non-integer
    /// token, and `0` (chains are 1-based, so `0` is always a mistake). A
    /// repeated id collapses into the set — `3,3` means "drop chain 3".
    pub fn parse_exclude(raw: &str) -> Result<Self, String> {
        let mut excluded = BTreeSet::new();
        for tok in raw.split(',') {
            let t = tok.trim();
            if t.is_empty() {
                return Err(format!(
                    "--exclude-chains: empty chain id in '{raw}' \
                     (expected a comma-separated list of 1-based ids, e.g. 3,5)"
                ));
            }
            let id: usize = t.parse().map_err(|_| {
                format!(
                    "--exclude-chains: '{t}' is not a chain id \
                     (expected a positive integer, e.g. 3,5)"
                )
            })?;
            if id == 0 {
                return Err(
                    "--exclude-chains: chain ids are 1-based; 0 is not a valid chain".to_string(),
                );
            }
            excluded.insert(id);
        }
        if excluded.is_empty() {
            return Err(
                "--exclude-chains: no chain ids given (expected e.g. --exclude-chains 3,5)"
                    .to_string(),
            );
        }
        Ok(ChainSelection { excluded })
    }

    /// `"3,5"` — the excluded ids as the warning renders them.
    pub fn excluded_csv(&self) -> String {
        self.excluded
            .iter()
            .map(|c| c.to_string())
            .collect::<Vec<_>>()
            .join(",")
    }

    /// The cohort-level warning (`fit table`, `compare`): the same drop set is
    /// applied to every fit, so there is no single kept/total to report — name
    /// the requested ids. Printed once, to stderr, before the derivations run.
    pub fn warn_requested(&self) {
        eprintln!(
            "\x1b[33mwarning:\x1b[0m --exclude-chains will drop chain(s) {} from each fit's \
             posterior cloud.",
            self.excluded_csv()
        );
        eprintln!("         {BIAS_CAVEAT_A}");
        eprintln!("         {BIAS_CAVEAT_B}");
    }

    /// The per-fit warning (`compare --exclude-chains @fit:ids`): this drop set
    /// targets one named fit, so name it. Printed once per fit, to stderr,
    /// before the derivations run. The bias caveat is left to the caller so it
    /// prints once for a multi-fit selection.
    pub fn warn_requested_for_fit(&self, fit: &str) {
        eprintln!(
            "\x1b[33mwarning:\x1b[0m --exclude-chains will drop chain(s) {} from fit '{fit}'.",
            self.excluded_csv()
        );
    }

    /// Drop the selected chains from a keyed draws cloud.
    ///
    /// This is THE chain filter — the single definition every read-side consumer
    /// shares. Steps, in order:
    ///
    /// 1. Determine the present chain set from the rows (distinct `chain + 1`,
    ///    1-based). A cloud with no `chain` column (an old param-only
    ///    `draws.tsv`) cannot be chain-selected → hard error.
    /// 2. Every requested id must be present → else `chain 7 not in this fit
    ///    (chains 1..6)`.
    /// 3. Drop the rows of the excluded chains.
    /// 4. An empty result (every chain excluded) → hard error.
    ///
    /// Returns the retained rows and the [`SubsetInfo`] provenance record.
    pub fn apply_keyed(
        &self,
        draws: Vec<KeyedDraw>,
    ) -> Result<(Vec<KeyedDraw>, SubsetInfo), String> {
        // 1. Present chains (1-based). A `None` chain means the file has no
        // `chain` column, so no chain can be named — refuse rather than silently
        // treat the whole cloud as one nameless chain.
        let mut present: BTreeSet<usize> = BTreeSet::new();
        for d in &draws {
            match d.chain {
                Some(c) => {
                    present.insert(c + 1);
                }
                None => {
                    return Err(
                        "--exclude-chains: this fit's draws.tsv has no `chain` column, \
                         so its chains cannot be selected (it predates keyed draws, or \
                         is a hand-built param-only file)"
                            .to_string(),
                    );
                }
            }
        }

        // 2. Validate every requested id is present.
        let present_hi = present.iter().copied().max().unwrap_or(0);
        for &id in &self.excluded {
            if !present.contains(&id) {
                return Err(format!(
                    "--exclude-chains: chain {id} not in this fit (chains 1..{present_hi})"
                ));
            }
        }

        // 3. Drop the excluded chains.
        let retained: Vec<KeyedDraw> = draws
            .into_iter()
            .filter(|d| {
                // Every row has Some(chain) here (step 1 refused None).
                let one_based = d.chain.map(|c| c + 1).unwrap_or(0);
                !self.excluded.contains(&one_based)
            })
            .collect();

        // 4. Refuse an empty posterior.
        if retained.is_empty() {
            return Err(
                "--exclude-chains leaves an empty posterior (every chain was excluded)".to_string(),
            );
        }

        let excluded: Vec<usize> = self.excluded.iter().copied().collect();
        let kept: Vec<usize> = present.iter().copied().filter(|c| !self.excluded.contains(c)).collect();
        let info = SubsetInfo {
            excluded,
            kept,
            n_total: present.len(),
        };
        Ok((retained, info))
    }
}

impl SubsetInfo {
    /// The JSON provenance object stamped into `predictive.json` and any other
    /// artifact produced under a selection: `{excluded, kept, n_total}`.
    pub fn to_json(&self) -> serde_json::Value {
        serde_json::json!({
            "excluded": self.excluded,
            "kept": self.kept,
            "n_total": self.n_total,
        })
    }

    /// `"3,5"` — the excluded ids as the header/warning render them.
    pub fn excluded_csv(&self) -> String {
        self.excluded
            .iter()
            .map(|c| c.to_string())
            .collect::<Vec<_>>()
            .join(",")
    }
}

/// The convergence diagnostics recomputed over a chain SUBSET — the shared
/// output of applying `--exclude-chains` and re-running the fit's own
/// [`compute_rhat_ess`](crate::fit::runner::compute_rhat_ess) on the retained
/// chains.
///
/// `fit summary` consumes the whole struct (it also derives posterior means from
/// [`kept`](Self::kept)); `fit predict` reduces it to a band label (max R̂ / min
/// ESS). Both routing through [`recompute_subset_diagnostics`] is what keeps the
/// two from ever disagreeing on the same fit + selection — the divergence that
/// let a chain-subset predictive band carry the polluted full-cloud R̂.
pub struct SubsetDiagnostics {
    /// Provenance: which chains were dropped / kept.
    pub info: SubsetInfo,
    /// The retained draw rows (chain-keyed), for any per-row derivation the
    /// caller needs (e.g. `fit summary`'s posterior means).
    pub kept: Vec<KeyedDraw>,
    /// Per-param Gelman–Rubin R̂ over the retained chains — FINITE entries only.
    /// A constant / fixed param yields a non-finite R̂ and is omitted, so a `max`
    /// over this map is the max R̂ across the *estimated* params (mirrors
    /// `PosteriorDiagnostics::max_rhat`).
    pub per_param: BTreeMap<String, crate::fit::method_result::ParamConvergence>,

    /// Retained draw count (rows kept).
    pub n_samples: usize,
    /// Retained chain count.
    pub n_chains: usize,
}

/// Drop the selected chains from a stage's `draws.tsv` and recompute the
/// per-param R̂ / ESS over the RETAINED chains, using the SAME
/// [`compute_rhat_ess`](crate::fit::runner::compute_rhat_ess) the fit used at
/// completion and the SAME [`apply_keyed`](ChainSelection::apply_keyed) filter.
///
/// This is the one recompute both `fit summary --exclude-chains` and
/// `fit predict --exclude-chains` call — so the diagnostics a chain-subset band
/// carries are the diagnostics the summary reports for the same subset, never
/// the stored full-cloud value that includes the dropped chains.
///
/// `estimated` is the **estimated (non-pinned)** parameter names — and it is
/// required, because "score every column and drop the non-finite results" is
/// not a safe default.
///
/// `draws.tsv` carries estimated params first and then the model's PINNED ones,
/// constant across every row by construction. Scoring all of them and filtering
/// non-finite results at insertion made one filter do two jobs: it correctly
/// hid a pinned parameter, which has no meaningful R̂, and it incorrectly hid an
/// ESTIMATED parameter frozen by a sampler that never accepted a move. The
/// benign half worked by accident and the pathological half disappeared with
/// it. Iterating only the estimated set separates them: a `ConstantDraws`
/// refusal in this map now means exactly one thing — an estimated parameter
/// that never moved.
pub fn recompute_subset_diagnostics(
    draws_path: &Path,
    selection: &ChainSelection,
    estimated: &[String],
) -> Result<SubsetDiagnostics, String> {
    let keyed = crate::load_draws_tsv_keyed(&draws_path.to_string_lossy())?;
    let (kept, info) = selection.apply_keyed(keyed)?;

    // Group the retained rows by 0-based chain (BTreeMap → ascending order).
    let mut grouped: BTreeMap<usize, Vec<&KeyedDraw>> = BTreeMap::new();
    for d in &kept {
        if let Some(c) = d.chain {
            grouped.entry(c).or_default().push(d);
        }
    }

    // Score the estimated params only. A pinned parameter is never offered to
    // the estimator, so it never enters these maps and cannot be confused with
    // an estimated one that failed to move.
    let param_names: &[String] = estimated;

    let mut per_param = BTreeMap::new();
    for p in param_names {
        let chains: Vec<Vec<f64>> = grouped
            .values()
            .map(|rows| rows.iter().filter_map(|r| r.params.get(p).copied()).collect())
            .collect();
        // EVERY estimated param gets an entry, whatever the estimator
        // managed — a parameter must never leave the fit by failing to be
        // diagnosed.
        per_param.insert(p.clone(), d_to_param(&crate::fit::runner::compute_rhat_ess(&chains)));
    }

    Ok(SubsetDiagnostics {
        info,
        n_samples: kept.len(),
        n_chains: grouped.len(),
        kept,
        per_param,
    })
}

/// Project one parameter's [`RhatEss`](crate::fit::runner::RhatEss) into the
/// sum type the diagnostics carry, so "could not be scored" travels with the
/// parameter instead of removing it.
fn d_to_param(
    d: &crate::fit::runner::RhatEss,
) -> crate::fit::method_result::ParamConvergence {
    use crate::fit::method_result::{ParamConvergence, Stat};
    match d.rank() {
        None => {
            let e = d.refusal().expect("an unscored param carries its refusal");
            ParamConvergence::NotScored {
                reason: e.refusal(),
                detail: Some(e.clone()),
            }
        }
        Some(r) => ParamConvergence::Scored {
            rhat: Stat::from_f64(r.rhat),
            rhat_bulk: Stat::from_f64(r.rhat_bulk),
            rhat_folded: Stat::from_f64(r.rhat_folded),
            rhat_classic: Stat::from_f64(d.rhat_classic()),
            ess_bulk: Stat::from_f64(r.ess_bulk),
            ess_tail: Stat::from_f64(r.ess_tail),
            all_chains_frozen: r.all_chains_frozen,
        },
    }
}

/// The two load-bearing caveat lines, single-sourced so the per-fit
/// ([`warn_active_selection`]) and cohort ([`ChainSelection::warn_requested`])
/// warnings state the bias risk identically.
const BIAS_CAVEAT_A: &str =
    "Post-hoc chain exclusion BIASES the posterior toward the retained mode — it is not a \
     convergence fix.";
const BIAS_CAVEAT_B: &str =
    "Excluded chains that disagree are often the sign of an unidentified parameter; see \
     `camdl fit summary` for the per-chain diagnostics.";

/// Print the two-line bias caveat once. For the per-fit `compare` form, which
/// names each fit's drop set separately ([`ChainSelection::warn_requested_for_fit`])
/// and then states the shared bias risk a single time.
pub fn eprint_bias_caveat() {
    eprintln!("         {BIAS_CAVEAT_A}");
    eprintln!("         {BIAS_CAVEAT_B}");
}

/// The loud, non-quietable warning printed to stderr on EVERY active selection.
///
/// The failure mode this guards is silent: a cherry-picked posterior read as if
/// it were a normal one. So the warning cannot be a warning the tooling can
/// suppress — it prints whenever a selection actually dropped a chain. It names
/// the dropped chains and states the bias direction (toward the retained mode),
/// and points at `fit summary` for the per-chain diagnostics (gh#406).
pub fn warn_active_selection(info: &SubsetInfo) {
    eprintln!(
        "\x1b[33mwarning:\x1b[0m --exclude-chains dropped chain(s) {} \
         ({} of {} chains kept).",
        info.excluded_csv(),
        info.kept.len(),
        info.n_total
    );
    eprintln!("         {BIAS_CAVEAT_A}");
    eprintln!("         {BIAS_CAVEAT_B}");
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    /// A keyed draw on 0-based chain `c`, draw `d`, with a single param `beta`.
    fn kd(c: usize, d: usize, beta: f64) -> KeyedDraw {
        KeyedDraw {
            chain: Some(c),
            draw: Some(d),
            params: HashMap::from([("beta".to_string(), beta)]),
        }
    }

    #[test]
    fn parse_basic_list() {
        // Parse produces the canonical drop set; apply it to a 5-chain cloud and
        // read the concrete excluded ids back off the provenance record.
        let sel = ChainSelection::parse_exclude("3,5").unwrap();
        let draws: Vec<KeyedDraw> = (0..5).map(|c| kd(c, 0, 0.1)).collect();
        let (_, info) = sel.apply_keyed(draws).unwrap();
        assert_eq!(info.excluded, vec![3, 5]);
    }

    #[test]
    fn parse_dedups_and_sorts() {
        assert_eq!(
            ChainSelection::parse_exclude("5,3,5").unwrap(),
            ChainSelection::parse_exclude("3,5").unwrap(),
            "repeats collapse and order is canonical",
        );
    }

    #[test]
    fn parse_rejects_zero() {
        let e = ChainSelection::parse_exclude("0").unwrap_err();
        assert!(e.contains("1-based"), "0 rejected with a 1-based hint: {e}");
    }

    #[test]
    fn parse_rejects_non_integer() {
        let e = ChainSelection::parse_exclude("3,x").unwrap_err();
        assert!(e.contains("not a chain id"), "got: {e}");
    }

    #[test]
    fn parse_rejects_empty_and_blank() {
        assert!(ChainSelection::parse_exclude("").is_err());
        assert!(ChainSelection::parse_exclude("3,,5").is_err(), "empty token in the middle");
    }

    #[test]
    fn apply_drops_the_named_chain_1based() {
        // 0-based chains 0,1,2 → 1-based 1,2,3. Exclude 3 → drop chain 0-based 2.
        let draws = vec![
            kd(0, 0, 0.1), kd(0, 1, 0.11),
            kd(1, 0, 0.2), kd(1, 1, 0.21),
            kd(2, 0, 9.0), kd(2, 1, 9.1), // the outlier chain (1-based 3)
        ];
        let sel = ChainSelection::parse_exclude("3").unwrap();
        let (kept, info) = sel.apply_keyed(draws).unwrap();
        assert_eq!(kept.len(), 4, "chain 3 (2 rows) dropped");
        assert!(kept.iter().all(|d| d.chain != Some(2)), "no 0-based chain 2 rows remain");
        assert_eq!(info.excluded, vec![3]);
        assert_eq!(info.kept, vec![1, 2]);
        assert_eq!(info.n_total, 3);
    }

    #[test]
    fn apply_errors_on_absent_chain_with_range() {
        let draws = vec![kd(0, 0, 0.1), kd(1, 0, 0.2)]; // chains 1..2
        let sel = ChainSelection::parse_exclude("7").unwrap();
        let e = sel.apply_keyed(draws).unwrap_err();
        assert!(e.contains("chain 7 not in this fit"), "names the bad id: {e}");
        assert!(e.contains("chains 1..2"), "states the valid range: {e}");
    }

    #[test]
    fn apply_errors_when_all_excluded() {
        let draws = vec![kd(0, 0, 0.1), kd(1, 0, 0.2)]; // chains 1,2
        let sel = ChainSelection::parse_exclude("1,2").unwrap();
        let e = sel.apply_keyed(draws).unwrap_err();
        assert!(e.contains("empty posterior"), "refuses an empty cloud: {e}");
    }

    #[test]
    fn apply_errors_without_chain_column() {
        // A param-only cloud (chain = None) cannot be chain-selected.
        let draws = vec![
            KeyedDraw { chain: None, draw: None, params: HashMap::from([("beta".to_string(), 0.1)]) },
        ];
        let sel = ChainSelection::parse_exclude("1").unwrap();
        let e = sel.apply_keyed(draws).unwrap_err();
        assert!(e.contains("no `chain` column"), "refuses a keyless cloud: {e}");
    }

    #[test]
    fn subset_info_json_and_csv() {
        let info = SubsetInfo { excluded: vec![3, 5], kept: vec![1, 2, 4, 6], n_total: 6 };
        assert_eq!(info.excluded_csv(), "3,5");
        let j = info.to_json();
        assert_eq!(j["excluded"], serde_json::json!([3, 5]));
        assert_eq!(j["kept"], serde_json::json!([1, 2, 4, 6]));
        assert_eq!(j["n_total"], serde_json::json!(6));
    }

    /// Review blocker 1, the half that makes `ConstantDraws` decidable.
    ///
    /// `draws.tsv` carries estimated parameters and then the model's PINNED
    /// ones, constant across every row. Scoring every column and dropping the
    /// non-finite results made one filter do two jobs — correctly hiding a
    /// pinned parameter, which has no meaningful R̂, and incorrectly hiding an
    /// ESTIMATED parameter that a stuck sampler froze. Iterating only the
    /// estimated set separates them.
    #[test]
    fn a_frozen_estimated_param_is_a_pathology_a_pinned_one_is_not_scored() {
        let dir = std::env::temp_dir().join("camdl_frozen_vs_pinned_test");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let draws = dir.join("draws.tsv");

        // `beta` mixes. `frozen` never moved — every chain sits at its own
        // single value, the 0%-acceptance deadlock. `sigma` is pinned: constant
        // at one value across every chain, because the model fixes it.
        let mut text = String::from("chain\tdraw\tbeta\tfrozen\tsigma\n");
        // Three chains written; chain 3 (1-based) is excluded, so the scored
        // subset is chains 0 and 1 — `ChainSelection` has no empty form.
        for chain in 0..3 {
            for draw in 0..40 {
                let beta = 0.3 + 0.01 * ((draw * 7 + chain * 3) % 11) as f64;
                let frozen = if chain == 0 { 0.5 } else { 0.9 };
                text.push_str(&format!("{chain}\t{draw}\t{beta}\t{frozen}\t6.3\n"));
            }
        }
        std::fs::write(&draws, text).unwrap();
        let keep_all = ChainSelection::parse_exclude("3").expect("drop the spare chain");

        // Estimated = beta + frozen. `sigma` is pinned and must not appear at
        // all — not in the R̂ map, and not as a refusal either.
        let sub = recompute_subset_diagnostics(
            &draws, &keep_all, &["beta".to_string(), "frozen".to_string()])
            .expect("recompute");
        assert!(sub.per_param.contains_key("beta"), "beta is assessable");
        assert!(!sub.per_param.contains_key("sigma"),
            "a pinned parameter is never scored: {:?}", sub.per_param.keys());

        // The frozen ESTIMATED parameter is PRESENT — never dropped for having
        // failed to be diagnosed — and is classified as a pathology.
        let frozen = sub.per_param.get("frozen")
            .unwrap_or_else(|| panic!("frozen must stay in the map: {:?}", sub.per_param.keys()));
        assert!(frozen.is_pathology(),
            "a sampler that never accepted a move is a failure, not a shrug: {frozen:?}");
        assert!(frozen.why_no_rhat().is_some(),
            "and it carries the reason: {frozen:?}");
        assert!(!frozen.rhat().cell(3, "—").contains('e'),
            "never a shape-determined 1e15: {:?}", frozen.rhat());

        let _ = std::fs::remove_dir_all(&dir);
    }
}
