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

use std::collections::BTreeSet;

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

    /// The cohort-level warning (`fit table`): the same drop set is applied to
    /// every fit, so there is no single kept/total to report — name the
    /// requested ids. Printed once, to stderr, before the derivations run.
    pub fn warn_requested(&self) {
        eprintln!(
            "\x1b[33mwarning:\x1b[0m --exclude-chains will drop chain(s) {} from each fit's \
             posterior when deriving a quantity.",
            self.excluded_csv()
        );
        eprintln!("         {BIAS_CAVEAT_A}");
        eprintln!("         {BIAS_CAVEAT_B}");
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

/// The two load-bearing caveat lines, single-sourced so the per-fit
/// ([`warn_active_selection`]) and cohort ([`ChainSelection::warn_requested`])
/// warnings state the bias risk identically.
const BIAS_CAVEAT_A: &str =
    "Post-hoc chain exclusion BIASES the posterior toward the retained mode — it is not a \
     convergence fix.";
const BIAS_CAVEAT_B: &str =
    "Excluded chains that disagree are often the sign of an unidentified parameter; see \
     `camdl fit summary` for the per-chain diagnostics.";

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
}
