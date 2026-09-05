//! Convergence of the *latent path* itself — R̂ and ESS of every state
//! variable at every substep, across the chains, computed over the posterior
//! paths a PGAS stage saved (gh#822).
//!
//! # Why the parameter R̂ is not enough
//!
//! A PGAS chain reports a joint draw `(θ, x_{0:T})`, and `fit summary`'s R̂
//! covers only `θ`. The path can be frozen while the parameters look settled:
//! on the three-province Ebola model (`bvd_province_hier3_ksmooth`, 19,200
//! particles) four independent chains held four *different* early latent paths
//! for the whole run — the exposed count in one province at day 5 read 198,
//! 132, 180 and 141 across the chains, each chain constant at its own value —
//! while the identified parameters agreed to R̂ ≈ 1. Each chain was reporting
//! one draw of the early path, and `θ` was conditioned on a prefix the sampler
//! never revisited. The trajectory-renewal profile
//! (`super::path_renewal`) says the sampler *did not move* there; this profile
//! says whether that mattered — whether the chains disagree about the states
//! it did not move.
//!
//! # The reduction
//!
//! For one cell — one column (compartment, `flow_*` or `inc_*`) at one substep
//! — the saved paths of each chain form a vector, and those vectors are handed
//! to [`rank_convergence`], the same rank-normalized split-R̂ / bulk-ESS
//! estimator the parameter table uses. One implementation, so a per-state
//! number and a per-parameter number mean the same thing. The estimator's own
//! refusals are read as findings, because for a latent state they are:
//!
//! - **`Constant`** — the pooled draws never moved (a compartment that is
//!   structurally zero, a flow that never fired). Nothing to assess.
//! - **`FrozenDisagree`** — every chain is internally constant, and the chains
//!   differ. R̂ is `+∞` there; the number is not informative but the *status*
//!   is: this is the frozen-prefix signature above, one cell at a time.
//! - **`Mixed`** — the chains moved; R̂ and ESS are computed and kept.
//!
//! The ESS is over the paths that were *saved* (`n_trajectories`, spread over
//! the retained draws by `draw_stride`), not over every sweep, so it is a lower
//! bound on the sweep-level ESS and the two are not comparable across runs that
//! saved different counts.
//!
//! # Resolved in time, then reduced
//!
//! The cells are binned into the same ten fixed tenths of the substep series
//! as the renewal profile ([`RENEWAL_BINS`]), so the two can be read side by
//! side: renewal 0.00 in `b0` with `frozen-disagree` 0.9 in `b0` is a
//! sampler that never moved over states the chains disagree about; renewal
//! 0.00 with `constant` 0.9 is a sampler that never moved over states that
//! are pinned. `agree_from` is the earliest substep after which no column is
//! `FrozenDisagree` — the horizon behind which the chains' paths are
//! independent draws of *something*, in front of which they are each a single
//! sample.

use io::trajectories::{PosteriorDraw, TrajColumnSpec};
use sim::inference::convergence::{
    rank_convergence, ConvergenceError, RankConvergence, DEGENERATE_REL_TOL,
    MIN_DRAWS_FOR_INFORMATIVE_ESS,
};
use sim::inference::pgas::RENEWAL_BINS;
use sim::state::Flows;
use std::path::Path;

/// The `pgas_summary.json` key the block lives under.
pub const LATENT_CONVERGENCE_KEY: &str = "latent_convergence";
/// The per-cell long-form table written beside the summary.
pub const LATENT_CONVERGENCE_TSV: &str = "latent_convergence.tsv";

/// One chain's saved posterior paths as a dense `[draw][substep][column]`
/// block, in the column order the `trajectories.tsv` writer emits (integer
/// compartments, real compartments, `flow_*`, `inc_*`).
///
/// Flattened once, at the point the chain drops its `PosteriorDraw`s, so the
/// stage retains eight bytes per cell per draw rather than the snapshot
/// structures.
#[derive(Debug, Clone, PartialEq)]
pub struct ChainPaths {
    /// The chain's label as its directory and the per-chain tables carry it
    /// (`chain_N`, 1-based).
    pub chain: usize,
    pub n_draws: usize,
    pub n_substeps: usize,
    pub n_cols: usize,
    /// Substep times, from the first draw's snapshots.
    pub times: Vec<f64>,
    values: Vec<f64>,
}

impl ChainPaths {
    /// Build from in-memory draws. `Ok(None)` when the chain saved no path.
    ///
    /// The shape is validated the way the TSV writer validates it: every draw
    /// has the same substep count, every snapshot the column counts the spec
    /// declares. A violation is a corrupt record and is refused, not padded.
    pub fn from_draws(
        draws: &[PosteriorDraw],
        columns: &TrajColumnSpec,
    ) -> Result<Option<Self>, String> {
        let Some(first) = draws.first() else { return Ok(None) };
        let n_substeps = first.path.snapshots.len();
        let n_int = columns.int_comps.len();
        let n_real = columns.real_comps.len();
        let n_flow = columns.flows.len();
        let n_inc = columns.incidence.len();
        let n_cols = n_int + n_real + n_flow + n_inc;
        let mut values = Vec::with_capacity(draws.len() * n_substeps * n_cols);
        for d in draws {
            if d.path.snapshots.len() != n_substeps {
                return Err(format!(
                    "latent convergence: chain {} draw {} has {} substeps, first draw has {}",
                    d.chain, d.draw, d.path.snapshots.len(), n_substeps
                ));
            }
            if n_inc > 0 && d.incidence.len() != n_substeps {
                return Err(format!(
                    "latent convergence: chain {} draw {}: incidence has {} rows but path has {} snapshots",
                    d.chain, d.draw, d.incidence.len(), n_substeps
                ));
            }
            for (s, snap) in d.path.snapshots.iter().enumerate() {
                if snap.int_state.counts.len() != n_int
                    || snap.real_state.values.len() != n_real
                    || snap.flows.len() != n_flow
                {
                    return Err(format!(
                        "latent convergence: chain {} draw {} substep {}: snapshot shape \
                         ({} int, {} real, {} flows) does not match the column spec \
                         ({n_int}, {n_real}, {n_flow})",
                        d.chain, d.draw, s,
                        snap.int_state.counts.len(), snap.real_state.values.len(), snap.flows.len(),
                    ));
                }
                values.extend(snap.int_state.counts.iter().map(|&c| c as f64));
                values.extend(snap.real_state.values.iter().copied());
                match &snap.flows {
                    Flows::Int(fs) => values.extend(fs.iter().map(|&f| f as f64)),
                    Flows::Real(fs) => values.extend(fs.iter().copied()),
                }
                if n_inc > 0 {
                    let row = &d.incidence[s];
                    if row.len() != n_inc {
                        return Err(format!(
                            "latent convergence: chain {} draw {} substep {}: incidence row has \
                             {} entries, spec declares {n_inc}",
                            d.chain, d.draw, s, row.len()
                        ));
                    }
                    values.extend(row.iter().copied());
                }
            }
        }
        let times = first.path.snapshots.iter().map(|s| s.t).collect();
        Ok(Some(ChainPaths { chain: first.chain + 1, n_draws: draws.len(), n_substeps, n_cols, times, values }))
    }

    /// Read one chain's saved paths back from its `trajectories.tsv`, with
    /// the data column names in file order.
    ///
    /// The columns are whatever the header names beyond the `chain draw time
    /// [date]` key — the file is the authority on its own layout, so no model
    /// is needed to score it. Rows are grouped by `draw` in file order (sweep
    /// order), and every draw must span the same substeps as the first, in the
    /// same order; anything else is a corrupt record and is refused.
    /// `Ok(None)` is a file with a header and no rows.
    pub fn from_trajectories_tsv(path: &Path) -> Result<Option<(Self, Vec<String>)>, String> {
        let txt = std::fs::read_to_string(path)
            .map_err(|e| format!("cannot read {}: {e}", path.display()))?;
        let mut lines = txt.lines().filter(|l| !l.starts_with('#'));
        let header: Vec<&str> = lines
            .next()
            .ok_or_else(|| format!("empty trajectories file: {}", path.display()))?
            .split('\t')
            .collect();
        let pos = |name: &str| {
            header.iter().position(|h| *h == name).ok_or_else(|| {
                format!("{}: no `{name}` column in the header", path.display())
            })
        };
        let (chain_i, draw_i, time_i) = (pos("chain")?, pos("draw")?, pos("time")?);
        let key = |h: &str| matches!(h, "chain" | "draw" | "time" | "date");
        let data_i: Vec<usize> = (0..header.len()).filter(|&i| !key(header[i])).collect();
        let columns: Vec<String> = data_i.iter().map(|&i| header[i].to_string()).collect();
        let n_cols = columns.len();

        // One entry per saved draw, in file order; the draw index only has to
        // change to start a new path (the writer emits each path contiguously).
        let mut draws: Vec<(usize, Vec<f64>, Vec<f64>)> = Vec::new();
        // The file's `chain` column is the 0-based id the writer received;
        // the label is the 1-based one its directory carries.
        let mut chain: Option<usize> = None;
        for (lineno, line) in lines.enumerate() {
            if line.trim().is_empty() {
                continue;
            }
            let f: Vec<&str> = line.split('\t').collect();
            let field = |i: usize| {
                f.get(i).ok_or_else(|| {
                    format!("{}: row {} has {} fields, header has {}",
                        path.display(), lineno + 1, f.len(), header.len())
                })
            };
            let num = |i: usize| -> Result<f64, String> {
                field(i)?.parse::<f64>().map_err(|e| {
                    format!("{}: row {} column `{}`: {e}", path.display(), lineno + 1, header[i])
                })
            };
            let draw: usize = field(draw_i)?.parse().map_err(|e| {
                format!("{}: row {} `draw`: {e}", path.display(), lineno + 1)
            })?;
            let t = num(time_i)?;
            let c: usize = field(chain_i)?.parse().map_err(|e| {
                format!("{}: row {} `chain`: {e}", path.display(), lineno + 1)
            })?;
            match chain {
                None => chain = Some(c + 1),
                Some(prev) if prev != c + 1 => {
                    return Err(format!(
                        "{}: row {} is chain {c}, the file's first row chain {}",
                        path.display(), lineno + 1, prev - 1
                    ));
                }
                Some(_) => {}
            }
            if draws.last().map_or(true, |(d, _, _)| *d != draw) {
                draws.push((draw, Vec::new(), Vec::new()));
            }
            let (_, times, values) = draws.last_mut().expect("just pushed");
            times.push(t);
            for &i in &data_i {
                values.push(num(i)?);
            }
        }
        let Some((_, times, _)) = draws.first() else { return Ok(None) };
        let n_substeps = times.len();
        let times = times.clone();
        let mut values = Vec::with_capacity(draws.len() * n_substeps * n_cols);
        for (draw, t, v) in &draws {
            if *t != times {
                return Err(format!(
                    "{}: draw {draw} spans {} substeps, the first draw {}; the paths do not share one substep grid",
                    path.display(), t.len(), n_substeps
                ));
            }
            values.extend_from_slice(v);
        }
        let chain = chain.expect("a row was read");
        Ok(Some((ChainPaths { chain, n_draws: draws.len(), n_substeps, n_cols, times, values }, columns)))
    }

    /// A dense block from already-flattened values; `values` is
    /// `[draw][substep][column]`.
    #[cfg(test)]
    pub fn from_values(
        chain: usize,
        n_draws: usize,
        n_substeps: usize,
        n_cols: usize,
        times: Vec<f64>,
        values: Vec<f64>,
    ) -> Self {
        assert_eq!(values.len(), n_draws * n_substeps * n_cols, "dense block shape");
        assert_eq!(times.len(), n_substeps, "one time per substep");
        ChainPaths { chain, n_draws, n_substeps, n_cols, times, values }
    }

    #[inline]
    fn value(&self, draw: usize, substep: usize, col: usize) -> f64 {
        self.values[(draw * self.n_substeps + substep) * self.n_cols + col]
    }
}

/// What the chains' saved paths say about one state at one substep.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LatentStatus {
    /// Every chain is internally constant, and the chains disagree: each chain
    /// holds one draw of this state and never revisited it.
    FrozenDisagree,
    /// The pooled draws never moved: the state is pinned (structurally, or by
    /// the data) and there is nothing to assess.
    Constant,
    /// The chains moved; R̂ and ESS are in `conv`.
    Mixed,
}

impl LatentStatus {
    pub fn as_str(self) -> &'static str {
        match self {
            LatentStatus::FrozenDisagree => "frozen_disagree",
            LatentStatus::Constant => "constant",
            LatentStatus::Mixed => "mixed",
        }
    }
}

/// One column at one substep.
#[derive(Debug, Clone, PartialEq)]
pub struct LatentCell {
    pub substep: usize,
    pub time: f64,
    pub column: usize,
    pub status: LatentStatus,
    /// Pooled mean over every chain's saved draws.
    pub mean: f64,
    /// Sample standard deviation of the per-chain means (the `B/n` of R̂).
    pub between_sd: f64,
    /// Root of the mean within-chain variance (the `W` of R̂).
    pub within_sd: f64,
    pub chain_mean_min: f64,
    pub chain_mean_max: f64,
    /// Chains that are internally constant at this cell, whatever the status.
    pub n_frozen_chains: usize,
    /// `Some` iff `status == Mixed`.
    pub conv: Option<RankConvergence>,
}

/// The cells of one tenth of the substep series, reduced.
#[derive(Debug, Clone, PartialEq)]
pub struct LatentBin {
    pub n_cells: usize,
    pub frac_frozen_disagree: f64,
    pub frac_constant: f64,
    pub frac_mixed: f64,
    /// Mean over the non-`Constant` cells of the fraction of chains that are
    /// internally constant there. A `Mixed` cell can still be three chains
    /// holding one draw each and a fourth that moved once; `frac_mixed`
    /// cannot see that, this can. `None` when every cell is `Constant`.
    pub frozen_chain_frac: Option<f64>,
    /// Over the `Mixed` cells; `None` when the bin has none.
    pub rhat_max: Option<f64>,
    /// Over the `Mixed` cells with a finite ESS; `None` when the bin has none.
    pub ess_bulk_min: Option<f64>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct LatentConvergence {
    pub n_chains: usize,
    /// Which chains, by their `chain_N` label — a refused start has no saved
    /// path and is not among them.
    pub chain_ids: Vec<usize>,
    /// Draws used per chain — the smallest saved count across chains; each
    /// chain keeps its first `n_draws` (the convention
    /// `super::row_convergence::partition_by_chain` uses, for the same reason:
    /// R̂ needs one count per chain and the retained block must stay
    /// contiguous in sweep order).
    pub n_draws: usize,
    /// What each chain actually saved, so a truncation is visible.
    pub chain_n_saved: Vec<usize>,
    pub n_substeps: usize,
    pub columns: Vec<String>,
    pub cells: Vec<LatentCell>,
    pub bins: Vec<LatentBin>,
    /// The earliest substep `s` such that no cell at any substep `≥ s` is
    /// `FrozenDisagree`. `Some(0)` when no cell is; `None` when the final
    /// substep still holds one.
    pub agree_from: Option<usize>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum LatentError {
    TooFewChains { n_chains: usize },
    TooFewDraws { n_draws: usize },
    ShapeMismatch { chain: usize, what: String },
    NonFinite { substep: usize, column: String },
}

impl std::fmt::Display for LatentError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            LatentError::TooFewChains { n_chains } => write!(
                f, "latent-path convergence needs at least 2 chains with saved paths, have {n_chains}"
            ),
            LatentError::TooFewDraws { n_draws } => write!(
                f, "latent-path convergence needs at least 4 saved paths per chain, have {n_draws} \
                    (raise n_trajectories)"
            ),
            LatentError::ShapeMismatch { chain, what } => {
                write!(f, "latent-path convergence: chain {} {what}", chain + 1)
            }
            LatentError::NonFinite { substep, column } => write!(
                f, "latent-path convergence: non-finite value in `{column}` at substep {substep}"
            ),
        }
    }
}

/// The bin a substep falls in — the expression `RenewalBins::record` uses, so
/// a renewal bin and a latent bin cover the same substeps.
fn bin_of(substep: usize, n_substeps: usize) -> usize {
    (substep * RENEWAL_BINS / n_substeps).min(RENEWAL_BINS - 1)
}

fn is_constant(v: &[f64]) -> bool {
    let (lo, hi) = v.iter().fold((f64::INFINITY, f64::NEG_INFINITY), |(l, h), &x| {
        (l.min(x), h.max(x))
    });
    let scale = (v.iter().sum::<f64>() / v.len() as f64).abs().max(f64::MIN_POSITIVE);
    hi - lo <= DEGENERATE_REL_TOL * scale
}

fn sample_var(v: &[f64], mean: f64) -> f64 {
    if v.len() < 2 {
        return 0.0;
    }
    v.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (v.len() - 1) as f64
}

/// Classify and reduce one cell from its per-chain draw vectors.
fn reduce_cell(
    substep: usize,
    time: f64,
    column: usize,
    per_chain: &[Vec<f64>],
) -> Result<LatentCell, ConvergenceError> {
    let n_chains = per_chain.len();
    let chain_means: Vec<f64> = per_chain
        .iter()
        .map(|c| c.iter().sum::<f64>() / c.len() as f64)
        .collect();
    let mean = chain_means.iter().sum::<f64>() / n_chains as f64;
    let between_sd = sample_var(&chain_means, mean).sqrt();
    let within_sd = (per_chain
        .iter()
        .zip(&chain_means)
        .map(|(c, &m)| sample_var(c, m))
        .sum::<f64>()
        / n_chains as f64)
        .sqrt();
    let chain_mean_min = chain_means.iter().cloned().fold(f64::INFINITY, f64::min);
    let chain_mean_max = chain_means.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let n_frozen_chains = per_chain.iter().filter(|c| is_constant(c)).count();

    let (status, conv) = match rank_convergence(per_chain) {
        Err(ConvergenceError::ConstantDraws { .. }) => (LatentStatus::Constant, None),
        Ok(rc) if rc.all_chains_frozen => (LatentStatus::FrozenDisagree, None),
        Ok(rc) => (LatentStatus::Mixed, Some(rc)),
        Err(e) => return Err(e),
    };
    Ok(LatentCell {
        substep, time, column, status, mean, between_sd, within_sd,
        chain_mean_min, chain_mean_max, n_frozen_chains, conv,
    })
}

/// The whole table: every column at every substep, across `chains`.
///
/// `columns` names the dense block's columns in order. Chains with no saved
/// path are the caller's to drop before calling.
pub fn latent_convergence(
    chains: &[ChainPaths],
    columns: &[String],
) -> Result<LatentConvergence, LatentError> {
    let n_chains = chains.len();
    if n_chains < 2 {
        return Err(LatentError::TooFewChains { n_chains });
    }
    let n_substeps = chains[0].n_substeps;
    let n_cols = chains[0].n_cols;
    if n_cols != columns.len() {
        return Err(LatentError::ShapeMismatch {
            chain: 0,
            what: format!("has {n_cols} columns, {} names given", columns.len()),
        });
    }
    for (c, ch) in chains.iter().enumerate() {
        if ch.n_substeps != n_substeps || ch.n_cols != n_cols {
            return Err(LatentError::ShapeMismatch {
                chain: c,
                what: format!(
                    "has {} substeps × {} columns, chain 1 has {n_substeps} × {n_cols}",
                    ch.n_substeps, ch.n_cols
                ),
            });
        }
    }
    let chain_ids: Vec<usize> = chains.iter().map(|c| c.chain).collect();
    let chain_n_saved: Vec<usize> = chains.iter().map(|c| c.n_draws).collect();
    let n_draws = chain_n_saved.iter().copied().min().unwrap_or(0);
    if n_draws < 4 {
        return Err(LatentError::TooFewDraws { n_draws });
    }

    // Cells are independent; reduce substeps in parallel (the ESS
    // autocovariance is the cost, and a long series times a wide state is
    // many cells).
    use rayon::prelude::*;
    let per_substep: Vec<Result<Vec<LatentCell>, LatentError>> = (0..n_substeps)
        .into_par_iter()
        .map(|s| {
            let time = chains[0].times[s];
            let mut per_chain: Vec<Vec<f64>> = vec![Vec::with_capacity(n_draws); n_chains];
            let mut out = Vec::with_capacity(n_cols);
            for col in 0..n_cols {
                for (c, ch) in chains.iter().enumerate() {
                    per_chain[c].clear();
                    per_chain[c].extend((0..n_draws).map(|d| ch.value(d, s, col)));
                }
                let cell = reduce_cell(s, time, col, &per_chain).map_err(|e| match e {
                    ConvergenceError::NonFiniteDraw { .. } => LatentError::NonFinite {
                        substep: s,
                        column: columns[col].clone(),
                    },
                    // Chain count, draw count and equal lengths were checked
                    // above; constant draws are a status, not an error.
                    other => LatentError::ShapeMismatch { chain: 0, what: other.to_string() },
                })?;
                out.push(cell);
            }
            Ok(out)
        })
        .collect();
    let mut cells = Vec::with_capacity(n_substeps * n_cols);
    for r in per_substep {
        cells.extend(r?);
    }

    let bins = reduce_bins(&cells, n_substeps, n_chains);
    let agree_from = agree_from(&cells, n_substeps);
    Ok(LatentConvergence {
        n_chains,
        chain_ids,
        n_draws,
        chain_n_saved,
        n_substeps,
        columns: columns.to_vec(),
        cells,
        bins,
        agree_from,
    })
}

/// Every chain's saved paths in a finished PGAS stage, read back from
/// `chain_N/trajectories.tsv` in chain order, with the shared column names —
/// the input [`latent_convergence`] takes, so a stage that ran before the
/// block existed (or whose report scrolled by) can be scored from what it
/// wrote. Chains without a `trajectories.tsv` (a refused start, or a stage
/// that saved no paths) are skipped, as the stage skips them; `Ok(None)` is a
/// stage with no such file at all. Chains whose headers disagree are refused.
pub fn read_stage_paths(stage_dir: &Path) -> Result<Option<(Vec<ChainPaths>, Vec<String>)>, String> {
    let mut files: Vec<(usize, std::path::PathBuf)> = std::fs::read_dir(stage_dir)
        .map_err(|e| format!("cannot read {}: {e}", stage_dir.display()))?
        .flatten()
        .filter_map(|e| {
            let n: usize = e.file_name().to_str()?.strip_prefix("chain_")?.parse().ok()?;
            let p = e.path().join("trajectories.tsv");
            p.is_file().then_some((n, p))
        })
        .collect();
    files.sort_by_key(|(n, _)| *n);
    let mut chains = Vec::new();
    let mut columns: Option<Vec<String>> = None;
    for (n, p) in files {
        let Some((block, cols)) = ChainPaths::from_trajectories_tsv(&p)? else { continue };
        match &columns {
            None => columns = Some(cols),
            Some(prev) if *prev != cols => {
                return Err(format!(
                    "chain {n}: {} names columns {:?}, chain {}'s names {:?}",
                    p.display(), cols, chains.len(), prev
                ));
            }
            Some(_) => {}
        }
        chains.push(block);
    }
    Ok(columns.map(|c| (chains, c)))
}

fn reduce_bins(cells: &[LatentCell], n_substeps: usize, n_chains: usize) -> Vec<LatentBin> {
    struct Acc {
        n: usize,
        frozen: usize,
        constant: usize,
        mixed: usize,
        /// Sum over non-constant cells of `n_frozen_chains`.
        frozen_chains: usize,
        rhat_max: Option<f64>,
        ess_min: Option<f64>,
    }
    let mut acc: Vec<Acc> = (0..RENEWAL_BINS)
        .map(|_| Acc { n: 0, frozen: 0, constant: 0, mixed: 0, frozen_chains: 0, rhat_max: None, ess_min: None })
        .collect();
    for cell in cells {
        let a = &mut acc[bin_of(cell.substep, n_substeps)];
        a.n += 1;
        if cell.status != LatentStatus::Constant {
            a.frozen_chains += cell.n_frozen_chains;
        }
        match cell.status {
            LatentStatus::FrozenDisagree => a.frozen += 1,
            LatentStatus::Constant => a.constant += 1,
            LatentStatus::Mixed => {
                a.mixed += 1;
                if let Some(rc) = &cell.conv {
                    if rc.rhat.is_finite() {
                        a.rhat_max = Some(a.rhat_max.map_or(rc.rhat, |m| m.max(rc.rhat)));
                    }
                    if rc.ess_bulk.is_finite() {
                        a.ess_min = Some(a.ess_min.map_or(rc.ess_bulk, |m| m.min(rc.ess_bulk)));
                    }
                }
            }
        }
    }
    acc.into_iter()
        .map(|a| {
            let frac = |k: usize| if a.n == 0 { f64::NAN } else { k as f64 / a.n as f64 };
            let non_constant = a.frozen + a.mixed;
            let frozen_chain_frac = (non_constant > 0)
                .then(|| a.frozen_chains as f64 / (non_constant * n_chains) as f64);
            LatentBin {
                n_cells: a.n,
                frac_frozen_disagree: frac(a.frozen),
                frac_constant: frac(a.constant),
                frac_mixed: frac(a.mixed),
                frozen_chain_frac,
                rhat_max: a.rhat_max,
                ess_bulk_min: a.ess_min,
            }
        })
        .collect()
}

fn agree_from(cells: &[LatentCell], n_substeps: usize) -> Option<usize> {
    let last_frozen = cells
        .iter()
        .filter(|c| c.status == LatentStatus::FrozenDisagree)
        .map(|c| c.substep)
        .max();
    match last_frozen {
        None => Some(0),
        Some(s) if s + 1 < n_substeps => Some(s + 1),
        Some(_) => None,
    }
}

impl LatentConvergence {
    /// The end-of-stage block, printed directly under the renewal profile so
    /// the two rows of tenths line up.
    ///
    /// Composed from the four pieces below rather than written out here, so
    /// `camdl fit summary` — which reorders them and gates the glossary behind
    /// `--explain` — reads the same numbers from the same code as the
    /// end-of-stage print.
    pub fn report(&self) -> String {
        let mut s = String::from("\n");
        s.push_str(&self.headline());
        s.push_str(&self.bins_table());
        s.push_str(&self.glossary());
        s.push_str(&self.findings());
        s
    }

    /// The one line naming what was scored: how many chains, which, how many
    /// saved paths each, and the shape of the block.
    pub fn headline(&self) -> String {
        format!(
            "latent-path convergence (gh#822; {} chain(s) [{}] × {} saved path(s), {} substeps × {} columns):\n",
            self.n_chains,
            self.chain_ids.iter().map(|c| c.to_string()).collect::<Vec<_>>().join(", "),
            self.n_draws,
            self.n_substeps,
            self.columns.len(),
        )
    }

    /// The per-bin reduction: one row per statistic, one column per tenth of
    /// the horizon.
    pub fn bins_table(&self) -> String {
        let cell = |v: Option<f64>| match v {
            Some(x) if x.is_finite() => format!("{x:>6.3}"),
            _ => "    NA".to_string(),
        };
        let frac = |v: f64| if v.is_finite() { Some(v) } else { None };
        // Below `MIN_DRAWS_FOR_INFORMATIVE_ESS` the estimator returns
        // `n_chains * (n_draws / 2)` for ANY input: it splits each chain in
        // half and a half that short cannot run its autocorrelation
        // truncation. The row would print one constant across every bin and
        // say nothing about mixing, so it is omitted rather than printed with
        // a disclaimer — a number beside R-hat reads as a measurement however
        // it is captioned. R-hat is unaffected; it uses no autocorrelation.
        let ess_row = if self.n_draws >= MIN_DRAWS_FOR_INFORMATIVE_ESS {
            format!("\x20 ESS min (mixed)  {}\n",
                self.bins.iter().map(|b| match b.ess_bulk_min {
                    Some(e) => format!("{e:>6.0}"),
                    None => "    NA".to_string(),
                }).collect::<Vec<_>>().join(" "))
        } else {
            format!(
                "\x20 (ESS omitted: {} saved path(s) per chain is below the {} the \
                 estimator needs; it would report {} in every bin whatever the paths did. \
                 Raise n_trajectories.)\n",
                self.n_draws, MIN_DRAWS_FOR_INFORMATIVE_ESS,
                self.n_chains * (self.n_draws / 2))
        };
        let labels: Vec<String> = (0..RENEWAL_BINS).map(|b| format!("    b{b}")).collect();
        format!(
            "\x20 bin              {}\n\
             \x20 frozen-disagree  {}\n\
             \x20 constant         {}\n\
             \x20 chains frozen    {}\n\
             \x20 R̂ max (mixed)    {}\n{}",
            labels.join(" "),
            self.bins.iter().map(|b| cell(frac(b.frac_frozen_disagree))).collect::<Vec<_>>().join(" "),
            self.bins.iter().map(|b| cell(frac(b.frac_constant))).collect::<Vec<_>>().join(" "),
            self.bins.iter().map(|b| cell(b.frozen_chain_frac)).collect::<Vec<_>>().join(" "),
            self.bins.iter().map(|b| cell(b.rhat_max)).collect::<Vec<_>>().join(" "),
            ess_row,
        )
    }

    /// What the two rows a reader misreads most actually count. Definitions
    /// only — no number from this fit appears here, which is why `fit summary`
    /// can hold it behind `--explain` without withholding evidence.
    pub fn glossary(&self) -> String {
        String::from(
            "  frozen-disagree: fraction of (state, substep) cells where every chain is \
             constant at its own value — each chain holds one draw there\n\
             \x20 chains frozen: over the non-constant cells, the fraction of chains that \
             never moved — a `mixed` cell may still be all chains but one holding one draw\n",
        )
    }

    /// What this fit's paths actually did: where the chains start agreeing, the
    /// single worst frozen cell, and the scope the ESS was taken over.
    /// Findings, not definitions — they are shown whatever `--explain` says.
    pub fn findings(&self) -> String {
        let mut s = String::new();
        match self.agree_from {
            Some(0) => s.push_str("  no frozen-disagree cell: the chains' paths mix everywhere\n"),
            Some(a) => s.push_str(&format!(
                "  chains agree from substep {a} of {}: before it some state is one draw per chain\n",
                self.n_substeps
            )),
            None => s.push_str(&format!(
                "  some state is frozen-disagree at the final substep {}: no substep where the chains' paths mix throughout\n",
                self.n_substeps - 1
            )),
        }
        // The single worst frozen cell, by between-chain spread relative to
        // its mean — the concrete number a reader can look up in the paths.
        if let Some(worst) = self
            .cells
            .iter()
            .filter(|c| c.status == LatentStatus::FrozenDisagree && c.mean.abs() > 0.0)
            .max_by(|a, b| {
                (a.between_sd / a.mean.abs())
                    .partial_cmp(&(b.between_sd / b.mean.abs()))
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
        {
            s.push_str(&format!(
                "  widest frozen disagreement: `{}` at substep {} (t={}) — chain means {}..{}, pooled mean {:.1}\n",
                self.columns[worst.column],
                worst.substep,
                worst.time,
                fmt_num(worst.chain_mean_min),
                fmt_num(worst.chain_mean_max),
                worst.mean,
            ));
        }
        s.push_str(&format!(
            "  ESS is over the {} saved path(s) per chain, not every sweep; table: {}\n",
            self.n_draws, LATENT_CONVERGENCE_TSV
        ));
        s
    }

    /// The `pgas_summary.json` block: the per-bin reduction, the horizon, and
    /// the shape it was computed over. The per-cell table stays in the TSV.
    pub fn summary_block(&self) -> serde_json::Value {
        let opt = |v: Option<f64>| match v {
            Some(x) if x.is_finite() => serde_json::json!(x),
            _ => serde_json::Value::Null,
        };
        let frac = |x: f64| if x.is_finite() { serde_json::json!(x) } else { serde_json::Value::Null };
        serde_json::json!({
            "n_chains": self.n_chains,
            "chain_ids": self.chain_ids,
            "n_draws": self.n_draws,
            "chain_n_saved": self.chain_n_saved,
            "n_substeps": self.n_substeps,
            "n_columns": self.columns.len(),
            "bin_span": format!("1/{RENEWAL_BINS} of the substep series"),
            "bins": self.bins.iter().map(|b| serde_json::json!({
                "n_cells": b.n_cells,
                "frac_frozen_disagree": frac(b.frac_frozen_disagree),
                "frac_constant": frac(b.frac_constant),
                "frac_mixed": frac(b.frac_mixed),
                "frozen_chain_frac": opt(b.frozen_chain_frac),
                "rhat_max": opt(b.rhat_max),
                "ess_bulk_min": opt(b.ess_bulk_min),
            })).collect::<Vec<_>>(),
            "agree_from": self.agree_from,
            "ess_over": "saved paths (n_trajectories), not sweeps",
            "table": LATENT_CONVERGENCE_TSV,
        })
    }

    /// The per-cell table, long form: one row per (substep, column).
    pub fn write_tsv(&self, path: &Path) -> Result<(), String> {
        use std::io::Write;
        let f = std::fs::File::create(path).map_err(|e| format!("{}: {e}", path.display()))?;
        let mut w = std::io::BufWriter::new(f);
        let na = |v: f64| if v.is_finite() { format!("{v:.6}") } else { "NA".to_string() };
        writeln!(
            w,
            "substep\ttime\tcolumn\tstatus\tmean\tbetween_sd\twithin_sd\tchain_mean_min\t\
             chain_mean_max\tn_frozen_chains\trhat\trhat_bulk\trhat_folded\tess_bulk\tess_tail"
        )
        .map_err(|e| e.to_string())?;
        for c in &self.cells {
            let (rhat, bulk, folded, ess_b, ess_t) = match &c.conv {
                Some(rc) => (na(rc.rhat), na(rc.rhat_bulk), na(rc.rhat_folded), na(rc.ess_bulk), na(rc.ess_tail)),
                None => ("NA".into(), "NA".into(), "NA".into(), "NA".into(), "NA".into()),
            };
            writeln!(
                w,
                "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}",
                c.substep,
                c.time,
                self.columns[c.column],
                c.status.as_str(),
                na(c.mean),
                na(c.between_sd),
                na(c.within_sd),
                na(c.chain_mean_min),
                na(c.chain_mean_max),
                c.n_frozen_chains,
                rhat, bulk, folded, ess_b, ess_t,
            )
            .map_err(|e| e.to_string())?;
        }
        w.flush().map_err(|e| e.to_string())
    }
}

fn fmt_num(x: f64) -> String {
    if x.fract() == 0.0 && x.abs() < 1e15 {
        format!("{}", x as i64)
    } else {
        format!("{x:.3}")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A block where `f(chain, draw, substep, col)` gives the value.
    fn block(
        n_chains: usize,
        n_draws: usize,
        n_substeps: usize,
        n_cols: usize,
        f: impl Fn(usize, usize, usize, usize) -> f64,
    ) -> Vec<ChainPaths> {
        (0..n_chains)
            .map(|c| {
                let mut values = Vec::new();
                for d in 0..n_draws {
                    for s in 0..n_substeps {
                        for k in 0..n_cols {
                            values.push(f(c, d, s, k));
                        }
                    }
                }
                let times = (0..n_substeps).map(|s| s as f64).collect();
                ChainPaths::from_values(c + 1, n_draws, n_substeps, n_cols, times, values)
            })
            .collect()
    }

    fn names(n: usize) -> Vec<String> {
        (0..n).map(|k| format!("c{k}")).collect()
    }

    /// A deterministic, non-degenerate cloud: distinct per draw, overlapping
    /// across chains.
    fn cloud(c: usize, d: usize, s: usize, k: usize) -> f64 {
        let x = ((c * 7 + d * 13 + s * 3 + k * 5) % 17) as f64;
        10.0 + x + 0.1 * (c as f64)
    }

    #[test]
    fn classification_matches_the_oracle() {
        // Column 0: each chain constant at its own value → FrozenDisagree.
        // Column 1: everything equal → Constant.
        // Column 2: a moving cloud → Mixed, with R̂ equal to rank_convergence
        // on the same vectors.
        let chains = block(3, 8, 2, 3, |c, d, s, k| match k {
            0 => 100.0 + 10.0 * c as f64,
            1 => 42.0,
            _ => cloud(c, d, s, k),
        });
        let lc = latent_convergence(&chains, &names(3)).unwrap();
        assert_eq!(lc.cells.len(), 2 * 3);
        let at = |s: usize, k: usize| lc.cells.iter().find(|c| c.substep == s && c.column == k).unwrap();

        let frozen = at(0, 0);
        assert_eq!(frozen.status, LatentStatus::FrozenDisagree);
        assert_eq!(frozen.n_frozen_chains, 3);
        assert!(frozen.conv.is_none());
        assert_eq!(frozen.chain_mean_min, 100.0);
        assert_eq!(frozen.chain_mean_max, 120.0);
        assert_eq!(frozen.mean, 110.0);
        assert_eq!(frozen.within_sd, 0.0);
        assert!((frozen.between_sd - 10.0).abs() < 1e-12, "sd of {{100,110,120}} is 10");

        let constant = at(1, 1);
        assert_eq!(constant.status, LatentStatus::Constant);
        assert_eq!(constant.n_frozen_chains, 3);
        assert!(constant.conv.is_none());

        let mixed = at(1, 2);
        assert_eq!(mixed.status, LatentStatus::Mixed);
        assert_eq!(mixed.n_frozen_chains, 0);
        let vectors: Vec<Vec<f64>> = (0..3)
            .map(|c| (0..8).map(|d| cloud(c, d, 1, 2)).collect())
            .collect();
        let oracle = rank_convergence(&vectors).unwrap();
        assert_eq!(mixed.conv.as_ref().unwrap(), &oracle);
        // between/within are the R̂ ingredients — check them against a direct
        // computation, so a swap of the two would be caught.
        let means: Vec<f64> = vectors.iter().map(|v| v.iter().sum::<f64>() / 8.0).collect();
        let gm = means.iter().sum::<f64>() / 3.0;
        let b = (means.iter().map(|m| (m - gm).powi(2)).sum::<f64>() / 2.0).sqrt();
        let w = (vectors.iter().zip(&means)
            .map(|(v, m)| v.iter().map(|x| (x - m).powi(2)).sum::<f64>() / 7.0)
            .sum::<f64>() / 3.0).sqrt();
        assert!((mixed.between_sd - b).abs() < 1e-12);
        assert!((mixed.within_sd - w).abs() < 1e-12);
        assert!((mixed.mean - gm).abs() < 1e-12);
    }

    /// `ESS min (mixed)` read exactly 40 in all ten bins of a real 8-chain,
    /// 10-path PGAS stage. That is not a coincidence and not a floor the
    /// reduction imposes: below
    /// [`MIN_DRAWS_FOR_INFORMATIVE_ESS`] the estimator returns
    /// `n_chains * (n_draws / 2)` for any input at all, so the row counts
    /// draws and says nothing about mixing. The number matches the reference
    /// implementation and is left alone; what the block owes the reader is the
    /// sentence saying so.
    #[test]
    fn the_ess_row_is_omitted_when_it_would_be_a_constant_not_a_measurement() {
        // The shape the real stage had: 8 chains, 10 saved paths each.
        let chains = block(8, 10, 12, 2, cloud);
        let lc = latent_convergence(&chains, &names(2)).unwrap();
        let ess: Vec<f64> = lc.cells.iter()
            .filter_map(|c| c.conv.as_ref())
            .map(|c| c.ess_bulk)
            .collect();
        assert!(!ess.is_empty(), "the fixture must produce assessable cells");
        assert!(ess.iter().all(|&e| e == 40.0),
            "every mixed cell reports 8 x (10/2) = 40 whatever the paths did: {ess:?}");
        let report = lc.report();
        assert!(!report.contains("ESS min (mixed)"),
            "a row that reports 40 whatever the paths did is not printed:\n{report}");
        assert!(report.contains("ESS omitted: 10 saved path(s) per chain"),
            "the omission is stated, with the count that caused it:\n{report}");
        assert!(report.contains("Raise n_trajectories"),
            "and what to change to get a measurement:\n{report}");

        // At the threshold the estimator runs, so the caveat must NOT fire —
        // otherwise it would read as a permanent disclaimer rather than a
        // statement about this stage.
        let enough = block(8, MIN_DRAWS_FOR_INFORMATIVE_ESS, 12, 2, cloud);
        let lc = latent_convergence(&enough, &names(2)).unwrap();
        let report = lc.report();
        assert!(report.contains("ESS min (mixed)"),
            "with enough saved paths the row is a measurement and is printed:\n{report}");
        assert!(!report.contains("ESS omitted"),
            "and the omission notice must not fire:\n{report}");
        let ess: Vec<f64> = lc.cells.iter()
            .filter_map(|c| c.conv.as_ref())
            .map(|c| c.ess_bulk)
            .collect();
        assert!(ess.iter().any(|&e| e != ess[0]),
            "and it varies across cells: {ess:?}");
    }

    /// The four pieces ARE the end-of-stage block, in that order.
    ///
    /// `camdl fit summary` reorders them and holds the glossary behind
    /// `--explain`; the end-of-stage print must keep emitting exactly what it
    /// always did. Pinning the composition is what stops one caller's layout
    /// change from silently rewriting the other's output.
    #[test]
    fn the_report_is_exactly_headline_table_glossary_findings() {
        let chains = block(4, 12, 20, 2, cloud);
        let lc = latent_convergence(&chains, &names(2)).unwrap();
        assert_eq!(
            lc.report(),
            format!("\n{}{}{}{}", lc.headline(), lc.bins_table(), lc.glossary(), lc.findings()),
        );
        // Each piece carries its own content and no other's, so a caller
        // picking two of them cannot pick up a third by accident.
        assert!(lc.headline().contains("4 chain(s)"));
        assert!(lc.bins_table().contains("frozen-disagree  "));
        assert!(lc.glossary().contains("frozen-disagree: fraction of"));
        assert!(!lc.glossary().contains("ESS is over the"));
        assert!(lc.findings().contains("ESS is over the"));
        assert!(!lc.findings().contains("frozen-disagree: fraction of"));
    }

    #[test]
    fn bins_account_for_every_cell_and_cover_the_series_in_tenths() {
        // 23 substeps × 2 columns; column 0 frozen for the first 7 substeps,
        // then a cloud; column 1 constant throughout.
        let chains = block(2, 6, 23, 2, |c, d, s, k| match k {
            0 if s < 7 => 5.0 + c as f64,
            0 => cloud(c, d, s, k),
            _ => 0.0,
        });
        let lc = latent_convergence(&chains, &names(2)).unwrap();
        assert_eq!(lc.bins.len(), RENEWAL_BINS);
        let total: usize = lc.bins.iter().map(|b| b.n_cells).sum();
        assert_eq!(total, 23 * 2);
        for b in &lc.bins {
            assert!(b.n_cells > 0);
            let sum = b.frac_frozen_disagree + b.frac_constant + b.frac_mixed;
            assert!((sum - 1.0).abs() < 1e-12, "fractions partition the bin: {sum}");
        }
        // Substeps 0..7 span bins 0..=3 (7*10/23 = 3); bin 0 holds substeps
        // 0,1,2 (3*10/23 = 1 → substep 3 is bin 1).
        assert_eq!(bin_of(2, 23), 0);
        assert_eq!(bin_of(3, 23), 1);
        assert_eq!(bin_of(6, 23), 2);
        assert_eq!(bin_of(7, 23), 3);
        assert_eq!(bin_of(22, 23), 9);
        // Bin 0: substeps 0,1,2 → column 0 frozen (3 cells), column 1 constant (3).
        assert_eq!(lc.bins[0].n_cells, 6);
        assert_eq!(lc.bins[0].frac_frozen_disagree, 0.5);
        assert_eq!(lc.bins[0].frac_constant, 0.5);
        assert_eq!(lc.bins[0].rhat_max, None);
        // Bin 3: substeps 7,8,9 → column 0 mixed, column 1 constant.
        assert_eq!(bin_of(9, 23), 3);
        assert_eq!(lc.bins[3].n_cells, 6);
        assert_eq!(lc.bins[3].frac_frozen_disagree, 0.0);
        assert_eq!(lc.bins[3].frac_mixed, 0.5);
        assert!(lc.bins[3].rhat_max.unwrap().is_finite());
        // Last bin: substeps 21,22 (20*10/23 = 8) → all mixed or constant.
        assert_eq!(bin_of(20, 23), 8);
        assert_eq!(lc.bins[9].n_cells, 4);
        assert_eq!(lc.bins[9].frac_frozen_disagree, 0.0);
    }

    #[test]
    fn frozen_chain_frac_sees_the_one_chain_that_moved() {
        // 4 chains, 10 substeps, 3 columns.
        //   column 0: chains 0..2 constant at their own value, chain 3 a cloud
        //             → `mixed` with three frozen chains (0.75 of them);
        //   column 1: every chain constant at its own value → frozen-disagree
        //             (all four frozen, 1.0);
        //   column 2: constant everywhere → excluded from the mean.
        let chains = block(4, 6, 10, 3, |c, d, s, k| match k {
            0 if c < 3 => 100.0 + c as f64,
            0 => cloud(c, d, s, k),
            1 => 5.0 + c as f64,
            _ => 0.0,
        });
        let lc = latent_convergence(&chains, &names(3)).unwrap();
        for (s, cell) in lc.cells.iter().enumerate() {
            let expect = match cell.column {
                0 => (LatentStatus::Mixed, 3),
                1 => (LatentStatus::FrozenDisagree, 4),
                _ => (LatentStatus::Constant, 4),
            };
            assert_eq!((cell.status, cell.n_frozen_chains), expect, "cell {s}");
        }
        // Every bin holds one cell of each column: (3 + 4) / (2 × 4).
        for b in &lc.bins {
            assert_eq!(b.n_cells, 3);
            assert_eq!(b.frac_mixed, 1.0 / 3.0);
            assert!((b.frozen_chain_frac.unwrap() - 7.0 / 8.0).abs() < 1e-12);
        }
        // All-constant → no non-constant cell to average over.
        let all_constant = block(4, 6, 10, 1, |_, _, _, _| 1.0);
        let lc = latent_convergence(&all_constant, &names(1)).unwrap();
        assert!(lc.bins.iter().all(|b| b.frozen_chain_frac.is_none()));
    }

    #[test]
    fn agree_from_is_the_substep_after_the_last_frozen_cell() {
        let frozen_prefix = |k_frozen: usize| {
            block(2, 6, 10, 2, move |c, d, s, k| {
                if k == 0 && s < k_frozen { 5.0 + c as f64 } else { cloud(c, d, s, k) }
            })
        };
        let lc = latent_convergence(&frozen_prefix(4), &names(2)).unwrap();
        assert_eq!(lc.agree_from, Some(4), "frozen at 0..4 → agree from 4");
        let lc = latent_convergence(&frozen_prefix(0), &names(2)).unwrap();
        assert_eq!(lc.agree_from, Some(0));
        let lc = latent_convergence(&frozen_prefix(10), &names(2)).unwrap();
        assert_eq!(lc.agree_from, None, "frozen through the final substep");
        // A frozen cell in the middle only — agree_from is after it, not 0.
        let chains = block(2, 6, 10, 1, |c, d, s, k| {
            if s == 6 { 5.0 + c as f64 } else { cloud(c, d, s, k) }
        });
        let lc = latent_convergence(&chains, &names(1)).unwrap();
        assert_eq!(lc.agree_from, Some(7));
    }

    #[test]
    fn refusals_name_their_cause() {
        let one = block(1, 6, 3, 1, cloud);
        assert_eq!(
            latent_convergence(&one, &names(1)).unwrap_err(),
            LatentError::TooFewChains { n_chains: 1 }
        );
        let short = block(2, 3, 3, 1, cloud);
        assert_eq!(
            latent_convergence(&short, &names(1)).unwrap_err(),
            LatentError::TooFewDraws { n_draws: 3 }
        );
        let mismatch = vec![block(1, 6, 3, 1, cloud).remove(0), block(1, 6, 4, 1, cloud).remove(0)];
        assert!(matches!(
            latent_convergence(&mismatch, &names(1)).unwrap_err(),
            LatentError::ShapeMismatch { chain: 1, .. }
        ));
        let nan = block(2, 6, 3, 1, |c, d, s, k| if s == 2 && d == 1 { f64::NAN } else { cloud(c, d, s, k) });
        assert_eq!(
            latent_convergence(&nan, &names(1)).unwrap_err(),
            LatentError::NonFinite { substep: 2, column: "c0".into() }
        );
    }

    #[test]
    fn unequal_saved_counts_truncate_to_the_common_first_block() {
        let mut chains = block(2, 8, 3, 1, cloud);
        // Chain 1 saved only 6 paths (six, so the ESS is finite and the
        // equality below is exact rather than NaN-blind).
        let short = block(1, 6, 3, 1, |_, d, s, k| cloud(1, d, s, k)).remove(0);
        chains[1] = short;
        let lc = latent_convergence(&chains, &names(1)).unwrap();
        assert_eq!(lc.n_draws, 6);
        assert_eq!(lc.chain_n_saved, vec![8, 6]);
        // The result equals the reduction over chain 0's FIRST 6 draws.
        let vectors: Vec<Vec<f64>> = (0..2).map(|c| (0..6).map(|d| cloud(c, d, 0, 0)).collect()).collect();
        let oracle = rank_convergence(&vectors).unwrap();
        assert_eq!(lc.cells[0].conv.as_ref().unwrap(), &oracle);
    }

    #[test]
    fn from_draws_flattens_in_the_writer_column_order() {
        use sim::state::{IntState, RealState, Snapshot, Trajectory};
        let snap = |t: f64, i: i64| Snapshot {
            t,
            int_state: IntState { counts: vec![i, i + 1] },
            real_state: RealState { values: vec![0.5] },
            flows: Flows::Int(vec![7]),
        };
        let draw = |draw: usize, base: i64| {
            let mut path = Trajectory::new();
            path.push(snap(0.0, base));
            path.push(snap(1.0, base + 10));
            PosteriorDraw { chain: 0, draw, path, incidence: vec![vec![3.0], vec![4.0]] }
        };
        let columns = TrajColumnSpec {
            int_comps: vec!["S".into(), "I".into()],
            real_comps: vec!["R".into()],
            flows: vec!["flow_a".into()],
            incidence: vec!["inc_x".into()],
        };
        let cp = ChainPaths::from_draws(&[draw(0, 100), draw(2, 200)], &columns).unwrap().unwrap();
        assert_eq!((cp.n_draws, cp.n_substeps, cp.n_cols), (2, 2, 5));
        assert_eq!(cp.times, vec![0.0, 1.0]);
        // draw 0, substep 1: S=110, I=111, R=0.5, flow_a=7, inc_x=4
        assert_eq!((0..5).map(|k| cp.value(0, 1, k)).collect::<Vec<_>>(), vec![110.0, 111.0, 0.5, 7.0, 4.0]);
        assert_eq!(cp.value(1, 0, 0), 200.0);
        assert!(ChainPaths::from_draws(&[], &columns).unwrap().is_none());
        // A draw with a different substep count is refused.
        let mut bad = draw(1, 0);
        bad.path.snapshots.pop();
        assert!(ChainPaths::from_draws(&[draw(0, 1), bad], &columns).is_err());
    }

    #[test]
    fn tsv_and_summary_carry_every_cell_and_bin() {
        let chains = block(2, 6, 5, 2, |c, d, s, k| if k == 0 && s < 2 { 1.0 + c as f64 } else { cloud(c, d, s, k) });
        let lc = latent_convergence(&chains, &names(2)).unwrap();
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join(LATENT_CONVERGENCE_TSV);
        lc.write_tsv(&p).unwrap();
        let text = std::fs::read_to_string(&p).unwrap();
        let lines: Vec<&str> = text.lines().collect();
        assert_eq!(lines.len(), 1 + 5 * 2);
        assert!(lines[0].starts_with("substep\ttime\tcolumn\tstatus\t"));
        assert_eq!(text.matches("\tfrozen_disagree\t").count(), 2);
        let block = lc.summary_block();
        assert_eq!(block["bins"].as_array().unwrap().len(), RENEWAL_BINS);
        assert_eq!(block["agree_from"], serde_json::json!(2));
        assert_eq!(block["n_draws"], serde_json::json!(6));
        let report = lc.report();
        assert!(report.contains("chains agree from substep 2 of 5"));
        assert!(report.contains("widest frozen disagreement: `c0` at substep"));
    }

    /// A `trajectories.tsv` as the writer lays it out: comment line, key
    /// columns (with `date`), then the data columns; each draw contiguous.
    fn traj_tsv(chain: usize, draws: &[(usize, &[(f64, [f64; 2])])]) -> String {
        let mut s = String::from("# camdl-trajectories v1\tmodel=x\tmethod=pgas\tgranularity=substep\n");
        s.push_str("chain\tdraw\ttime\tdate\tS\tflow_a\n");
        for (d, rows) in draws {
            for (t, v) in rows.iter() {
                s.push_str(&format!("{chain}\t{d}\t{t}\t2026-01-01\t{}\t{}\n", v[0], v[1]));
            }
        }
        s
    }

    #[test]
    fn trajectories_tsv_reads_back_as_the_dense_block() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("trajectories.tsv");
        std::fs::write(&p, traj_tsv(3, &[
            (10, &[(0.0, [9.0, 0.0]), (0.5, [8.0, 1.0]), (1.0, [6.0, 2.0])]),
            (20, &[(0.0, [9.0, 0.0]), (0.5, [7.0, 2.0]), (1.0, [7.0, 0.0])]),
        ])).unwrap();
        let (block, cols) = ChainPaths::from_trajectories_tsv(&p).unwrap().unwrap();
        assert_eq!(cols, vec!["S".to_string(), "flow_a".to_string()]);
        assert_eq!(block.chain, 4, "the file's 0-based chain id, labelled as its `chain_N` directory");
        assert_eq!((block.n_draws, block.n_substeps, block.n_cols), (2, 3, 2));
        assert_eq!(block.times, vec![0.0, 0.5, 1.0]);
        assert_eq!(block.value(0, 2, 0), 6.0);
        assert_eq!(block.value(1, 1, 1), 2.0);
        assert_eq!(block.value(1, 2, 0), 7.0);

        // Header only: a chain that saved no path.
        std::fs::write(&p, "chain\tdraw\ttime\tS\n").unwrap();
        assert!(ChainPaths::from_trajectories_tsv(&p).unwrap().is_none());

        // A draw on a different substep grid is refused, not padded.
        std::fs::write(&p, traj_tsv(1, &[
            (10, &[(0.0, [1.0, 0.0]), (1.0, [1.0, 0.0])]),
            (20, &[(0.0, [1.0, 0.0])]),
        ])).unwrap();
        let e = ChainPaths::from_trajectories_tsv(&p).unwrap_err();
        assert!(e.contains("draw 20 spans 1 substeps"), "{e}");

        // No `draw` column: not a trajectories file.
        std::fs::write(&p, "chain\ttime\tS\n1\t0\t1\n").unwrap();
        assert!(ChainPaths::from_trajectories_tsv(&p).unwrap_err().contains("`draw`"));

        // Two chains in one file is not one chain's file.
        let mut two = traj_tsv(0, &[(10, &[(0.0, [1.0, 0.0])])]);
        two.push_str("1\t10\t0\t2026-01-01\t1\t0\n");
        std::fs::write(&p, two).unwrap();
        assert!(ChainPaths::from_trajectories_tsv(&p).unwrap_err().contains("is chain 1"));
    }

    #[test]
    fn stage_paths_come_in_chain_order_and_skip_chains_without_a_path() {
        let dir = tempfile::tempdir().unwrap();
        let rows: &[(f64, [f64; 2])] = &[(0.0, [1.0, 0.0]), (1.0, [2.0, 1.0])];
        // chain_10 must sort after chain_2; chain_3 refused its start and has
        // no trajectories.tsv; chain_4 wrote a header only.
        for (n, s) in [(10usize, 10.0), (2, 2.0), (1, 1.0)] {
            let d = dir.path().join(format!("chain_{n}"));
            std::fs::create_dir(&d).unwrap();
            let rows: Vec<(f64, [f64; 2])> = rows.iter().map(|(t, v)| (*t, [v[0] * s, v[1]])).collect();
            // The writer puts the 0-based id in the file under a 1-based dir.
            std::fs::write(d.join("trajectories.tsv"),
                traj_tsv(n - 1, &[(5, &rows), (6, &rows), (7, &rows), (8, &rows)])).unwrap();
        }
        std::fs::create_dir(dir.path().join("chain_3")).unwrap();
        std::fs::write(dir.path().join("chain_3/trace.tsv"), "sweep\tloglik\n").unwrap();
        std::fs::create_dir(dir.path().join("chain_4")).unwrap();
        std::fs::write(dir.path().join("chain_4/trajectories.tsv"), "chain\tdraw\ttime\tS\tflow_a\n").unwrap();

        let (chains, cols) = read_stage_paths(dir.path()).unwrap().unwrap();
        assert_eq!(cols, vec!["S".to_string(), "flow_a".to_string()]);
        let first_s: Vec<f64> = chains.iter().map(|c| c.value(0, 1, 0)).collect();
        assert_eq!(first_s, vec![2.0, 4.0, 20.0], "chain_1, chain_2, chain_10");
        let labels: Vec<usize> = chains.iter().map(|c| c.chain).collect();
        assert_eq!(labels, vec![1, 2, 10]);
        let lc = latent_convergence(&chains, &cols).unwrap();
        assert_eq!(lc.chain_ids, vec![1, 2, 10]);
        assert!(lc.report().contains("3 chain(s) [1, 2, 10]"), "{}", lc.report());
        assert_eq!(lc.summary_block()["chain_ids"], serde_json::json!([1, 2, 10]));

        // A chain whose header names other columns is refused.
        std::fs::write(dir.path().join("chain_4/trajectories.tsv"),
            "chain\tdraw\ttime\tS\tflow_b\n4\t5\t0\t1\t0\n").unwrap();
        assert!(read_stage_paths(dir.path()).unwrap_err().contains("chain 4"));

        // No chain wrote a path at all.
        let empty = tempfile::tempdir().unwrap();
        std::fs::create_dir(empty.path().join("chain_1")).unwrap();
        assert!(read_stage_paths(empty.path()).unwrap().is_none());
    }
}
