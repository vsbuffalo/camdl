//! What `camdl score` reads off a prediction computed again with its outcomes
//! bound, and the scoring rules it applies to each row
//! (`docs/dev/proposals/2026-11-23-prediction-and-scores.md`, "What a score
//! is" and "What score computes, by mode").
//!
//! The prediction's own computation ([`crate::fit::predict`]) hands back one
//! [`ScoredCell`] per predicted row whose observed value is known: every
//! predictive sample it drew there, and, in the same layout, the observation
//! model's log density of the observed value at the state behind each sample.
//! Under one-step and forecast prediction it also hands back one
//! [`JointCell`] per time, holding each sample's log density of every
//! stream's value observed there together. Nothing here runs a model.
//!
//! A cell's rules are computed from its samples alone, so a score is a pure
//! function of the cells and the [`ScoreRules`] settings:
//!
//! - the log score `log p̂(y) = logsumexp_s ℓ_s − log S` over the `S = M·N`
//!   samples (`M` draws, `N` particles or trajectories per draw): the log of
//!   the mean density, never the mean of per-draw logs;
//! - the CRPS by the fair estimator ([`score::trace::crps_sample_fair`]);
//! - the WIS and its parts on the forecast hubs' quantile set
//!   ([`score::interval`]), from the samples' type-7 quantiles;
//! - the randomized PIT ([`score::trace::pit_sample_randomized`]);
//! - whether the observed value lies in each central interval.

use std::collections::BTreeMap;

use crate::fit::observed_rows::ObservedRow;
use crate::fit::predict::RowRole;

/// Which of the three modes a scored row was predicted in.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) enum ScoreMode {
    FreeForward,
    OneStep,
    Forecast,
}

impl ScoreMode {
    /// The mode's name, as `score.json` and `camdl predict --mode` spell it.
    pub(crate) fn as_str(self) -> &'static str {
        match self {
            ScoreMode::FreeForward => "free_forward",
            ScoreMode::OneStep => "one_step",
            ScoreMode::Forecast => "forecast",
        }
    }
}

/// Which of the two modes that predict the data's own rows predicted a row.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) enum DataRowMode {
    FreeForward,
    OneStep,
}

impl DataRowMode {
    /// The value of the `mode` column of `scored_rows.tsv`.
    pub(crate) fn as_str(self) -> &'static str {
        self.score_mode().as_str()
    }

    pub(crate) fn score_mode(self) -> ScoreMode {
        match self {
            DataRowMode::FreeForward => ScoreMode::FreeForward,
            DataRowMode::OneStep => ScoreMode::OneStep,
        }
    }
}

/// What a scored row is a prediction of: a row of the data, or a target
/// after a forecast origin. A score holds rows of one kind.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) enum CellKind {
    /// A data row, predicted free-forward or one step ahead.
    DataRow { mode: DataRowMode },
    /// A forecast target: the origin it is forecast from, how many of its
    /// stream's periods ahead of the origin it is, from 1, and the time from
    /// the origin to its end.
    Target { origin: f64, periods_ahead: i64, lead: f64 },
}

impl CellKind {
    pub(crate) fn mode(&self) -> ScoreMode {
        match self {
            CellKind::DataRow { mode } => mode.score_mode(),
            CellKind::Target { .. } => ScoreMode::Forecast,
        }
    }
}

/// What a joint log score is of: every stream's row ending at one stop of a
/// one-step prediction, or a forecast's targets.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) enum JointKind {
    /// The rows of every stream that end at one stop of the one-step filter.
    DataRow,
    /// The targets of a forecast from `origin`: those ending at one time,
    /// which share the `lead`, or with `lead` `None` every target of the
    /// forecast.
    Target { origin: f64, lead: Option<f64> },
}

impl JointKind {
    pub(crate) fn mode(&self) -> ScoreMode {
        match self {
            JointKind::DataRow => ScoreMode::OneStep,
            JointKind::Target { .. } => ScoreMode::Forecast,
        }
    }
}

/// The outcomes `camdl score` binds into the prediction's computation, read
/// from the outcome files by the loader the prediction reads its data with.
#[derive(Debug, Clone, Default)]
pub(crate) struct OutcomeBinding {
    /// Per leaf stream: every row of its outcome file, in file order. The
    /// first `conditioning[leaf]` are the rows the prediction conditioned on,
    /// equal to them by row digest.
    pub rows: BTreeMap<String, Vec<ObservedRow>>,
    pub conditioning: BTreeMap<String, usize>,
    /// Per stream source: the outcome file, which the one-step pass binds.
    pub files: BTreeMap<String, String>,
}

impl OutcomeBinding {
    /// The rows of `leaf` after the ones the prediction conditioned on; empty
    /// for a leaf with no outcome file.
    pub(crate) fn after_conditioning(&self, leaf: &str) -> &[ObservedRow] {
        match (self.rows.get(leaf), self.conditioning.get(leaf)) {
            (Some(rows), Some(&n)) => &rows[n.min(rows.len())..],
            _ => &[],
        }
    }
}

/// One predicted row or target with its observed value.
#[derive(Debug, Clone)]
pub(crate) struct ScoredCell {
    pub scenario: String,
    pub sweep: Vec<(String, f64)>,
    /// The observation block, a leaf of `source`.
    pub leaf: String,
    pub source: String,
    pub stratum: Vec<(String, String)>,
    /// What the row measures: a period `[start, stop)`, or an instant with
    /// `start == stop`.
    pub start: f64,
    pub stop: f64,
    pub kind: CellKind,
    pub role: RowRole,
    pub observed: f64,
    /// The predictive samples drawn at the row from the states that reached
    /// it alive. Empty when no draw's state did.
    pub samples: Vec<f64>,
    /// The log density of `observed` at each state the row's score averages
    /// over: every particle of every draw that reached the row, a particle
    /// the filter killed and a draw whose filter degenerated at the row at
    /// −∞. Empty when no draw reached it, and the row is then unscored.
    /// Under free-forward and forecast prediction it is aligned with
    /// `samples`.
    pub log_liks: Vec<f64>,
}

/// The log density of every stream observed at one time together, or of a
/// forecast's every target together.
#[derive(Debug, Clone)]
pub(crate) struct JointCell {
    pub scenario: String,
    pub sweep: Vec<(String, f64)>,
    pub kind: JointKind,
    pub start: f64,
    pub stop: f64,
    pub role: RowRole,
    /// The leaves whose values the joint density is of.
    pub streams: Vec<String>,
    /// `log_liks[s]`: sample `s`'s log density of those values together.
    pub log_liks: Vec<f64>,
}

/// What the recomputed prediction hands the score.
#[derive(Debug, Clone, Default)]
pub(crate) struct Scored {
    pub cells: Vec<ScoredCell>,
    pub joints: Vec<JointCell>,
    /// Rows or streams that were not scored, each with the reason.
    pub notes: Vec<String>,
    /// Whether the prediction ran on the ODE backend, whose free-forward
    /// density along a draw's trajectory is exact.
    pub ode: bool,
    /// Whether the prediction's parameters were one point (a fixed point, or
    /// an optimizer run's winner) rather than a posterior.
    pub plug_in: bool,
}

impl Scored {
    pub(crate) fn extend(&mut self, other: Scored) {
        self.cells.extend(other.cells);
        self.joints.extend(other.joints);
        self.notes.extend(other.notes);
        self.ode |= other.ode;
        self.plug_in |= other.plug_in;
    }
}

/// The settings of the rules: the two central interval levels whose coverage
/// is reported, and the seed of the PIT's uniform draws.
#[derive(Debug, Clone, Copy)]
pub(crate) struct ScoreRules {
    pub intervals: (f64, f64),
    pub pit_seed: u64,
}

/// The rules computed for one cell.
#[derive(Debug, Clone, Copy)]
pub(crate) struct RowScore {
    pub log_score: f64,
    pub crps: f64,
    pub wis: score::interval::Wis,
    pub pit: f64,
    /// Whether the observed value lies in the first and the second central
    /// interval of [`ScoreRules::intervals`], ends included.
    pub inside: (bool, bool),
    pub mean: f64,
    /// The quantiles at [`crate::quantile::QUANTILE_LEVELS`].
    pub quantiles: [f64; 5],
}

/// `logsumexp(ll) − log(len)`, refusing NaN and an empty set.
pub(crate) fn log_mean_exp(ll: &[f64]) -> Result<f64, String> {
    crate::fit::forecast_score::log_mean_exp(ll)
}

/// The rules of one cell that has samples, with `v` the PIT's uniform draw.
/// The log score is over `log_liks`; the other rules are over `samples`.
pub(crate) fn score_cell(cell: &ScoredCell, rules: &ScoreRules, v: f64) -> Result<RowScore, String> {
    let at = || format!("stream '{}' at t = {}", cell.leaf, cell.stop);
    if cell.samples.is_empty() || cell.log_liks.len() < cell.samples.len() {
        return Err(format!(
            "internal error: {}: {} samples and {} log densities",
            at(),
            cell.samples.len(),
            cell.log_liks.len()
        ));
    }
    if let Some(x) = cell.samples.iter().find(|x| !x.is_finite()) {
        return Err(format!("{}: a predictive sample is not finite ({x})", at()));
    }
    let y = cell.observed;
    let log_score = log_mean_exp(&cell.log_liks).map_err(|e| format!("{}: {e}", at()))?;
    let mut sorted = cell.samples.clone();
    sorted.sort_by(f64::total_cmp);
    let q = |p: f64| score::trace::sample_quantile_sorted(&sorted, p);
    let hub: Vec<f64> = score::interval::HUB_LEVELS.iter().map(|&p| q(p)).collect();
    let (median, intervals) = score::interval::intervals_of(&score::interval::HUB_LEVELS, &hub)
        .ok_or("internal error: the hub quantile set is not symmetric")?;
    let wis = score::interval::weighted_interval_score(median, &intervals, y)
        .ok_or_else(|| format!("internal error: {}: the quantiles are not ordered", at()))?;
    let within = |level: f64| {
        let tail = (1.0 - level) / 2.0;
        q(tail) <= y && y <= q(1.0 - tail)
    };
    let levels = crate::quantile::QUANTILE_LEVELS;
    Ok(RowScore {
        log_score,
        crps: score::trace::crps_sample_fair(&cell.samples, y),
        wis,
        pit: score::trace::pit_sample_randomized(&cell.samples, y, v),
        inside: (within(rules.intervals.0), within(rules.intervals.1)),
        mean: cell.samples.iter().sum::<f64>() / cell.samples.len() as f64,
        quantiles: [q(levels[0].0), q(levels[1].0), q(levels[2].0), q(levels[3].0), q(levels[4].0)],
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cell(samples: Vec<f64>, log_liks: Vec<f64>, y: f64) -> ScoredCell {
        ScoredCell {
            scenario: "(none)".into(),
            sweep: Vec::new(),
            leaf: "cases".into(),
            source: "cases".into(),
            stratum: Vec::new(),
            start: 0.0,
            stop: 7.0,
            kind: CellKind::Target { origin: 0.0, periods_ahead: 1, lead: 7.0 },
            role: RowRole::Future,
            observed: y,
            samples,
            log_liks,
        }
    }

    const RULES: ScoreRules = ScoreRules { intervals: (0.5, 0.9), pit_seed: 1 };

    /// Two posterior draws whose observation densities of the outcome are
    /// 0.02 and 0.18: the log score is the log of their mean density, 0.1,
    /// and differs from the mean of their logs, which by Jensen's
    /// inequality is smaller.
    #[test]
    fn the_log_score_of_a_mixture_is_the_log_of_the_mean_density() {
        let c = cell(vec![3.0, 5.0], vec![0.02_f64.ln(), 0.18_f64.ln()], 4.0);
        let s = score_cell(&c, &RULES, 0.5).unwrap();
        assert!((s.log_score - 0.1_f64.ln()).abs() < 1e-14, "{}", s.log_score);
        let mean_of_logs = (0.02_f64.ln() + 0.18_f64.ln()) / 2.0;
        assert!(s.log_score > mean_of_logs + 0.5, "log of mean {} vs mean of logs {mean_of_logs}", s.log_score);
    }

    /// The CRPS, PIT and coverage are the kernels' on the samples, and the
    /// WIS is the hub-set score of the samples' quantiles.
    #[test]
    fn the_sample_rules_are_the_kernels() {
        let samples: Vec<f64> = (0..101).map(f64::from).collect();
        let c = cell(samples.clone(), vec![0.0; 101], 30.0);
        let s = score_cell(&c, &RULES, 0.25).unwrap();
        assert_eq!(s.crps, score::trace::crps_sample_fair(&samples, 30.0));
        assert_eq!(s.pit, score::trace::pit_sample_randomized(&samples, 30.0, 0.25));
        assert_eq!(s.inside, (true, true));
        assert_eq!(s.quantiles, [5.0, 25.0, 50.0, 75.0, 95.0]);
        let out = score_cell(&cell(samples, vec![0.0; 101], 20.0), &RULES, 0.25).unwrap();
        assert_eq!(out.inside, (false, true), "20 is outside [25, 75] and inside [5, 95]");
        assert!(out.wis.overprediction > 0.0 && out.wis.underprediction == 0.0);
    }

    #[test]
    fn a_cell_whose_layouts_disagree_is_an_internal_error() {
        assert!(score_cell(&cell(vec![1.0, 2.0], vec![0.0], 1.0), &RULES, 0.5).is_err());
        assert!(score_cell(&cell(vec![], vec![0.0], 1.0), &RULES, 0.5).is_err());
        assert!(score_cell(&cell(vec![1.0, f64::NAN], vec![0.0, 0.0], 1.0), &RULES, 0.5).is_err());
    }
}
