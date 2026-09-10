//! Per-chain log-likelihood outlier diagnostics for the Bayesian samplers.
//!
//! `fit summary`'s aggregate R̂/ESS answer "did the chains agree?" but not
//! "WHICH chain disagreed." When a minority of MCMC chains wander into a
//! non-representative side mode (a near-unidentified parameter carving out a
//! flat ridge), the user is left to open each `chain_N/trace.tsv` by hand to
//! find the stragglers. A per-chain robust z-score names them.
//!
//! Read-side only. Everything here is computed at `fit summary` time from the
//! committed trace files; nothing flows into a `run_id` / CAS key. Adding a
//! diagnostic does not re-key any fit.
//!
//! One shared seam, and one obligation on everyone who feeds it. The scout
//! gate message (`gating::format_decibans_spread_verdict`) and the summary
//! table (`fit_summary`) both name outliers via the same
//! [`chain_loglik_mod_zscores`] / [`outlier_labels`], so "what counts as an
//! outlier" is defined in exactly one place. That function scores whatever
//! numbers it is handed: **the caller owns the claim that they are comparable
//! across chains.** The scout gate hands it IF2 clean-eval marginals
//! `log p(y | θ)`; this module's reader hands it whatever
//! [`LoglikType::chain_agreement_column`] nominates. Anything else — most of
//! all a per-chain quantity evaluated at that chain's own latent path — is not
//! an input to this statistic.
//!
//! ## Which column, and why not by position (gh#667)
//!
//! Every Bayesian sampler streams its trace through
//! [`super::trace_writer::TraceWriter`], whose fixed layout is
//! `<index> <loglik> log_posterior …`. That made trace column index 1 look
//! like a safe structural invariant to key on, and this module keyed on it.
//! It is not safe: the *name* in that slot is `log_likelihood` for pmmh / mh /
//! nuts — a marginal `log p(y | θ)`, comparable — but `log_complete_data_ll`
//! for PGAS, the joint `log p(y, X | θ)` over the data **and the sampled
//! latent path**, which is not. Two different quantities in one slot, and the
//! substitution was invisible precisely because nothing named the column.
//!
//! Scoring chains on the PGAS value ranks them mostly by the latent-path term
//! `log p(X | θ)`, a density at one sampled path: a θ whose path distribution
//! is more concentrated scores higher on every typical path without fitting
//! the data any better. On the 60,000-sweep fit behind gh#667 the between-
//! chain spread was 522 nats in the path term and 9 nats in the observation
//! term — the flag would have named chains that were sampling correctly.
//!
//! So the reader keys on the column NAME, chosen per sampler by
//! [`LoglikType::chain_agreement_column`]: `obs_ll` (`log p(y | X, θ)`) for
//! PGAS, `log_likelihood` for the marginal samplers. PGAS's own target and its
//! transition term are still read and displayed — see
//! [`CompleteDataMeans`] — they are just never ranked on.
//!
//! ## The statistic: robust modified z-score (Iglewicz & Hoaglin 1993)
//!
//! We flag on the **modified z-score**
//!
//! ```text
//!   M_i = 0.6745 · (x_i − median(x)) / MAD ,    MAD = median(|x_i − median(x)|)
//! ```
//!
//! centred on the **median** and scaled by the **median absolute deviation**,
//! both of which ignore the outliers themselves. This is deliberately *not* the
//! classic `(x − mean)/SD` z-score: mean/SD let a minority of co-stuck chains
//! inflate the spread and mask their own flag (and cap a single outlier's |z|
//! at √(N−1), unreachable for small N). The robust version has neither problem
//! — a handful of chains in a side mode are flagged no matter how many share
//! it, because the median/MAD describe the *bulk*, not the whole set. The
//! `0.6745` (≈ Φ⁻¹(0.75)) rescales MAD to a consistent σ-estimator, so the
//! threshold stays comparable to a normal-theory z.
//!
//! `MAD == 0` (≥ half the chains share a numeric value, so the median has zero
//! robust spread — a degenerate case that does not arise for real MCMC means
//! but can in synthetic inputs) falls back to Iglewicz–Hoaglin's documented
//! alternate, the **mean absolute deviation**:
//!
//! ```text
//!   M_i = (x_i − mean(x)) / (1.253314 · MeanAD) ,   MeanAD = mean(|x_i − mean(x)|)
//! ```
//!
//! and if `MeanAD` is also 0 (every chain identical) there are no outliers.
//! Non-finite chain means (an unreadable trace) never poison the median/MAD —
//! they are excluded from the centre/scale and rendered as `—`, so a broken
//! chain is visible rather than silently reading as "clean."

use std::path::Path;

use super::loglik::{LoglikType, TRACE_COL_COMPLETE_DATA_LL, TRACE_COL_TRANSITION_LL};

/// The `TraceWriter` layout's third column, present in every sampler's trace.
/// Its presence is the structural check that a file is a trace at all, and it
/// is itself read by the degeneracy screen (gh#608).
const TRACE_COL_LOG_POSTERIOR: &str = "log_posterior";

/// Flag a chain when its **modified z-score** exceeds this in magnitude. 3.5 is
/// the Iglewicz–Hoaglin (1993) recommended cutoff: with the 0.6745 rescaling,
/// MAD is a consistent σ-estimator, so 3.5 modified-SDs is the robust analogue
/// of a 3.5-σ classic z. Named (not inlined) so the gate and the summary share
/// one threshold.
pub const CHAIN_LOGLIK_OUTLIER_MODZ: f64 = 3.5;

/// One chain's mean log-likelihood and its robust modified z-score against the
/// between-chain spread.
#[derive(Debug, Clone, PartialEq)]
pub struct ChainZScore {
    /// 1-based chain number for display (`chain 1` … `chain N`), matching the
    /// user-facing convention used everywhere else in fit output.
    pub chain: usize,
    /// Mean log-likelihood over this chain's post-burn-in draws. `NaN` when the
    /// chain had no readable trace rows (rendered `—`, never counted "clean").
    pub mean_loglik: f64,
    /// Modified z-score (median/MAD, or mean/MeanAD when MAD == 0). `0.0` for a
    /// single chain or perfect agreement; `NaN` when `mean_loglik` is non-finite.
    pub mod_z: f64,
    /// `|mod_z| > CHAIN_LOGLIK_OUTLIER_MODZ`. Always `false` for a non-finite
    /// `mod_z` (a NaN is surfaced as `—`, not swallowed as "not an outlier").
    pub is_outlier: bool,
}

/// Per-chain robust modified z-scores from per-chain mean logliks, in chain
/// order (index 0 = chain 1). See the module docs for the statistic. A single
/// chain, or perfect agreement, yields all-zero scores and nothing flagged —
/// the negative-control path.
pub fn chain_loglik_mod_zscores(means: &[f64]) -> Vec<ChainZScore> {
    let n = means.len();
    if n == 0 {
        return Vec::new();
    }
    // Robust centre + scale from the FINITE means only: a NaN from an unreadable
    // trace must neither poison the median/MAD nor read as "clean."
    let finite: Vec<f64> = means.iter().copied().filter(|m| m.is_finite()).collect();
    let (center, scale) = robust_center_scale(&finite);
    means
        .iter()
        .enumerate()
        .map(|(i, &m)| {
            // Finite mean + a positive, finite scale → a genuine z. Otherwise
            // (non-finite mean, or every finite chain identical so scale == 0)
            // there is no z to compute: NaN mean → NaN z (rendered `—`);
            // identical finite chains → 0.0.
            let mod_z = if m.is_finite() && scale.is_finite() && scale > 0.0 {
                (m - center) / scale
            } else if m.is_finite() {
                0.0
            } else {
                f64::NAN
            };
            // Explicit finiteness guard: a NaN z is NOT an outlier verdict, it is
            // "could not assess" — surfaced via `mean_loglik`/`mod_z` = `—`, never
            // silently passed as "not an outlier."
            let is_outlier = mod_z.is_finite() && mod_z.abs() > CHAIN_LOGLIK_OUTLIER_MODZ;
            ChainZScore {
                chain: i + 1,
                mean_loglik: m,
                mod_z,
                is_outlier,
            }
        })
        .collect()
}

/// `(center, scale)` for the modified z-score over the finite chain means.
/// Primary: `(median, MAD/0.6745)`. Fallback when `MAD == 0`:
/// `(mean, 1.253314·MeanAD)`. `scale == 0.0` (all values identical, or fewer
/// than one finite value) means "no spread" → every z is 0, nothing flagged.
fn robust_center_scale(finite: &[f64]) -> (f64, f64) {
    let n = finite.len();
    if n == 0 {
        return (0.0, 0.0);
    }
    let med = median_of(finite);
    let mad = median_of(&finite.iter().map(|x| (x - med).abs()).collect::<Vec<_>>());
    if mad > 0.0 {
        // M_i = 0.6745·(x − med)/MAD  ⇔  center = med, scale = MAD / 0.6745.
        (med, mad / 0.6745)
    } else {
        // MAD == 0: ≥ half the means share a value. Iglewicz–Hoaglin's alternate
        // scale is the mean absolute deviation: M_i = (x − mean)/(1.253314·MeanAD).
        let mean = finite.iter().sum::<f64>() / n as f64;
        let mean_ad = finite.iter().map(|x| (x - mean).abs()).sum::<f64>() / n as f64;
        (mean, 1.253314 * mean_ad)
    }
}

/// Median of a set of finite values (clones + sorts; the chain count is tiny).
/// Even length averages the two central order statistics. `NaN` for an empty set.
fn median_of(vals: &[f64]) -> f64 {
    let n = vals.len();
    if n == 0 {
        return f64::NAN;
    }
    let mut sorted = vals.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    if n % 2 == 1 {
        sorted[n / 2]
    } else {
        (sorted[n / 2 - 1] + sorted[n / 2]) / 2.0
    }
}

/// The flagged chains as `"chain N"` labels, worst-|mod_z| first. Empty when no
/// chain exceeds the threshold.
pub fn outlier_labels(scores: &[ChainZScore]) -> Vec<String> {
    let mut flagged: Vec<&ChainZScore> = scores.iter().filter(|s| s.is_outlier).collect();
    flagged.sort_by(|a, b| {
        b.mod_z
            .abs()
            .partial_cmp(&a.mod_z.abs())
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    flagged.iter().map(|s| format!("chain {}", s.chain)).collect()
}

/// What the per-chain table shows, and what it is allowed to rank on.
///
/// [`scored`](Self::scored) is the ONLY field the outlier statistic sees; the
/// rest is display. Splitting them in the type is the point — it is what stops
/// the sampler's own objective from being mistaken for a cross-chain
/// comparison again (gh#667).
#[derive(Debug, Clone, PartialEq)]
pub struct ChainLoglikMeans {
    /// The trace column [`scored`](Self::scored) was read from, for labelling
    /// the table and for saying which column is missing when it is.
    pub scored_column: &'static str,
    /// Per-chain mean of `scored_column` over the retained draws, in chain
    /// order (index 0 = chain 1). `NaN` when a chain has no readable rows.
    pub scored: Vec<f64>,
    /// `true` when no discovered trace carries `scored_column` at all — a
    /// stage whose chains cannot be compared, which the caller must SAY rather
    /// than render as a table of dashes.
    pub scored_column_absent: bool,
    /// PGAS only: the sampler's own target and its latent-path term, shown
    /// beside the scored column and never ranked on. `None` for the marginal
    /// samplers, where the scored column already IS the target.
    pub complete_data: Option<CompleteDataMeans>,
}

/// PGAS's complete-data target and its latent-path term, per chain — the two
/// quantities gh#667 removed from the ranking but deliberately kept visible.
/// `log_complete_data_ll` is the sampler's own Gibbs target, so a chain that
/// has fallen off the joint support shows up here; `transition_ll` is what
/// makes the diagnosis obvious, because a wide spread there next to a tight
/// spread in `obs_ll` is the entropy effect rather than a fit difference.
#[derive(Debug, Clone, PartialEq)]
pub struct CompleteDataMeans {
    /// Per-chain mean `log p(y, X | θ)` (`log_complete_data_ll`).
    pub complete: Vec<f64>,
    /// Per-chain mean `log p(X | θ)` (`transition_ll`).
    pub transition: Vec<f64>,
}

/// Read every `chain_*/trace.tsv` under `stage_dir` and return the per-chain
/// post-burn-in means the summary table needs, ordered by ascending chain
/// number.
///
/// `kind` is the stage's log-likelihood class, which decides *by name* which
/// column the chains may be compared on — see
/// [`LoglikType::chain_agreement_column`] and this module's header for why a
/// column *position* is the wrong key (gh#667).
///
/// Returns `None` when no `chain_*/trace.tsv` files exist (the sampler wrote no
/// per-chain trace — e.g. an optimizer-only stage) so the caller can say so
/// rather than silently show an empty table.
///
/// ## Post-burn-in selection
///
/// The post-burn-in kept draws are a suffix of the trace, and `draws.tsv`
/// records exactly how many there are per chain, so we take the **last `K_c`
/// rows** of chain `c`'s trace (`K_c` = that chain's row count in `draws.tsv`):
///
/// - **nuts** — the trace is written post-warmup only, `draws.tsv` has the same
///   `n_samples` rows: last `K_c` == all rows (exact).
/// - **pmmh / mh** — the trace carries a warm-up prefix (`step < burn_in`, one
///   row per step) then the thinned sampling suffix; `draws.tsv` counts the
///   suffix, so the last `K_c` rows are exactly the post-burn-in draws (exact).
/// - **pgas** — the trace carries every sweep (warm-up + all sampling,
///   unthinned); `draws.tsv` counts the thinned post-burn-in draws, so the last
///   `K_c` sweeps are a post-burn-in tail window (`K_c ≤ n_sweeps − burn_in`,
///   so warm-up is never included — a subset of the post-burn-in region when
///   thinning is on).
///
/// When `draws.tsv` is absent or unreadable (a partial / hand-built stage) we
/// fall back to the whole trace: without the posterior manifest we cannot strip
/// warm-up, so the mean is noisier but still honest. This never contaminates a
/// well-formed fit.
pub fn read_chain_mean_logliks(stage_dir: &Path, kind: LoglikType) -> Option<ChainLoglikMeans> {
    let mut chain_dirs = discover_chain_dirs(stage_dir);
    if chain_dirs.is_empty() {
        return None;
    }
    chain_dirs.sort_by_key(|(n, _)| *n);

    // Per-chain kept-draw counts from draws.tsv, in ascending chain-id order.
    // draws.tsv's `chain` column is 0-based and contiguous, so the i-th count
    // pairs with the i-th chain dir once both are sorted ascending.
    let draw_counts = read_draw_counts(&stage_dir.join("draws.tsv"));

    let scored_column = kind.chain_agreement_column();
    // The complete-data split is PGAS's alone: the marginal samplers have no
    // latent path to separate out, and their scored column IS their target.
    // Exhaustive, not `!is_marginal()` — a future non-marginal class (an
    // observation-conditional, say) would not thereby acquire PGAS's columns.
    let wants_split = match kind {
        LoglikType::CompleteData => true,
        LoglikType::If2 | LoglikType::Marginal | LoglikType::OdeMarginal => false,
    };
    let names: Vec<&str> = if wants_split {
        vec![scored_column, TRACE_COL_COMPLETE_DATA_LL, TRACE_COL_TRANSITION_LL]
    } else {
        vec![scored_column]
    };

    let mut scored = Vec::with_capacity(chain_dirs.len());
    let mut complete = Vec::with_capacity(chain_dirs.len());
    let mut transition = Vec::with_capacity(chain_dirs.len());
    // "Absent" means a readable trace exists and none of them names the scored
    // column — distinct from "no readable trace at all", which surfaces as NaN
    // means. Starts `false` so an unreadable stage never claims the column is
    // missing.
    let mut any_header_read = false;
    let mut any_scored_column = false;

    for (i, (_, dir)) in chain_dirs.iter().enumerate() {
        let cols = read_trace_cols(&dir.join("trace.tsv"), &names);
        if let Some(cols) = cols.as_ref() {
            any_header_read = true;
            if cols[0].is_some() {
                any_scored_column = true;
            }
        }
        let k = draw_counts.get(i).copied();
        let mean_of = |slot: usize| -> f64 {
            let Some(Some(all)) = cols.as_ref().map(|c| &c[slot]) else {
                return f64::NAN;
            };
            retained_mean(all, k)
        };
        scored.push(mean_of(0));
        if wants_split {
            complete.push(mean_of(1));
            transition.push(mean_of(2));
        }
    }

    Some(ChainLoglikMeans {
        scored_column,
        scored,
        scored_column_absent: any_header_read && !any_scored_column,
        complete_data: wants_split.then_some(CompleteDataMeans { complete, transition }),
    })
}

/// Mean of the retained tail of one chain's column: the last `k` rows when
/// `draws.tsv` gave a count that fits, else the whole trace (no manifest means
/// we cannot strip warm-up, so the mean is noisier but still honest). `NaN`
/// for an empty selection — never silently 0.
fn retained_mean(all: &[f64], k: Option<usize>) -> f64 {
    let selected: &[f64] = match k {
        Some(k) if k > 0 && k <= all.len() => &all[all.len() - k..],
        _ => all,
    };
    if selected.is_empty() {
        f64::NAN
    } else {
        selected.iter().sum::<f64>() / selected.len() as f64
    }
}

/// Collect `(chain_number, dir)` for every `chain_<N>` subdirectory of
/// `stage_dir` with a parseable trailing integer. The on-disk base differs by
/// sampler (nuts numbers from 0, pmmh/pgas from 1); sorting by the parsed
/// number gives a stable order either way, and the display index is assigned
/// sequentially by the caller.
fn discover_chain_dirs(stage_dir: &Path) -> Vec<(usize, std::path::PathBuf)> {
    let mut dirs = Vec::new();
    let Ok(entries) = std::fs::read_dir(stage_dir) else {
        return dirs;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if !path.is_dir() {
            continue;
        }
        let Some(name) = path.file_name().and_then(|n| n.to_str()) else {
            continue;
        };
        if let Some(rest) = name.strip_prefix("chain_") {
            if let Ok(n) = rest.parse::<usize>() {
                dirs.push((n, path));
            }
        }
    }
    dirs
}

/// Read the NAMED columns of one per-chain trace, row-aligned, as f64 rows.
///
/// `None` when the file is missing/unreadable or does not look like a
/// [`super::trace_writer::TraceWriter`] trace (its header must carry
/// `log_posterior`); that guard is what separates "not a trace" from "a trace
/// without this column". Otherwise one slot per requested name, in order, with
/// `None` for a name the header does not have — so a caller can tell a missing
/// column from an empty one.
///
/// A row is skipped for EVERY column when any *present* requested field is
/// short or unparseable, which keeps the returned vectors aligned with one
/// another (a caller reading `obs_ll` beside `transition_ll` compares the same
/// sweeps).
fn read_trace_cols(trace_path: &Path, names: &[&str]) -> Option<Vec<Option<Vec<f64>>>> {
    let contents = std::fs::read_to_string(trace_path).ok()?;
    let mut lines = contents.lines();
    let header = lines.next()?;
    let cols: Vec<&str> = header.split('\t').collect();
    if !cols.contains(&TRACE_COL_LOG_POSTERIOR) {
        return None;
    }
    let idx: Vec<Option<usize>> = names
        .iter()
        .map(|n| cols.iter().position(|c| c == n))
        .collect();
    let mut out: Vec<Option<Vec<f64>>> =
        idx.iter().map(|i| i.map(|_| Vec::new())).collect();
    let mut row: Vec<f64> = Vec::with_capacity(names.len());
    for line in lines {
        if line.trim().is_empty() {
            continue;
        }
        let fields: Vec<&str> = line.split('\t').collect();
        row.clear();
        let complete = idx.iter().flatten().all(|&c| {
            match fields.get(c).and_then(|s| s.parse::<f64>().ok()) {
                Some(v) => {
                    row.push(v);
                    true
                }
                None => false,
            }
        });
        if !complete {
            continue;
        }
        for (slot, v) in out.iter_mut().filter_map(|s| s.as_mut()).zip(row.iter()) {
            slot.push(*v);
        }
    }
    Some(out)
}

/// One chain's stuck-state screen (gh#608, ebola F8): over the RETAINED
/// draws (the same last-K_c rule as [`read_chain_mean_logliks`]), how many
/// rows record a non-finite log-posterior as the chain's CURRENT state. A
/// Metropolis-within-Gibbs chain whose current state has zero posterior
/// density is incoherent (gh#607) — ANY such retained row marks the chain
/// degenerate, and its draws contaminate `draws.tsv` and every pooled
/// number. Exclusion stays explicit (gh#419): this screen only makes the
/// contamination impossible to miss.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ChainNegInf {
    /// 1-based display index, aligned with the per-chain loglik table.
    pub chain: usize,
    /// Retained rows whose log-posterior is non-finite.
    pub n_neginf: usize,
    /// Retained rows for this chain.
    pub n_retained: usize,
}

/// One chain's point-mass screen (gh#635, ebola item 1): over `draws.tsv`,
/// how many DISTINCT parameter vectors the chain retained. A chain that
/// never accepted a θ-move keeps exactly one — a point mass at its start,
/// which the −inf screen (gh#608) cannot see when that start has FINITE
/// density. Its draws still enter every pooled number (3 such chains were
/// 37.5% of a pooled cloud downstream, R̂ 5.14, masquerading as a mode).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ChainUniqueDraws {
    /// 1-based display index, aligned with the per-chain loglik table.
    pub chain: usize,
    /// Distinct parameter vectors among the retained draws.
    pub n_unique: usize,
    /// Retained draws for this chain.
    pub n_draws: usize,
}

/// Per-chain distinct-parameter-vector counts from `draws.tsv`. The `chain`
/// column is 0-based; vectors are compared as their exact on-disk text (the
/// writer round-trips full precision, so textual equality is value
/// equality). `None` when draws.tsv is missing/unreadable.
pub fn read_chain_unique_draws(stage_dir: &Path) -> Option<Vec<ChainUniqueDraws>> {
    let contents = std::fs::read_to_string(stage_dir.join("draws.tsv")).ok()?;
    let mut lines = contents.lines();
    let header = lines.next()?;
    let mut cols = header.split('\t');
    if cols.next() != Some("chain") {
        return None;
    }
    use std::collections::{BTreeMap, HashSet};
    let mut per_chain: BTreeMap<usize, (HashSet<&str>, usize)> = BTreeMap::new();
    for line in lines {
        if line.trim().is_empty() {
            continue;
        }
        let Some((chain_tok, rest)) = line.split_once('\t') else { continue };
        let Ok(c) = chain_tok.parse::<usize>() else { continue };
        // Strip the draw index (column 2): the PARAMS are what repeat.
        let params = rest.split_once('\t').map(|(_, p)| p).unwrap_or(rest);
        let e = per_chain.entry(c).or_default();
        e.0.insert(params);
        e.1 += 1;
    }
    if per_chain.is_empty() {
        return None;
    }
    Some(per_chain.into_iter().map(|(c, (set, n))| ChainUniqueDraws {
        chain: c + 1,
        n_unique: set.len(),
        n_draws: n,
    }).collect())
}

/// Per-chain [`ChainNegInf`] screen over the retained trace rows. `None`
/// when no chain traces exist (same discovery as the mean-loglik reader).
pub fn read_chain_neginf(stage_dir: &Path) -> Option<Vec<ChainNegInf>> {
    let mut chain_dirs = discover_chain_dirs(stage_dir);
    if chain_dirs.is_empty() {
        return None;
    }
    chain_dirs.sort_by_key(|(n, _)| *n);
    let draw_counts = read_draw_counts(&stage_dir.join("draws.tsv"));
    let mut out = Vec::with_capacity(chain_dirs.len());
    for (i, (_, dir)) in chain_dirs.iter().enumerate() {
        let cols = read_trace_cols(&dir.join("trace.tsv"), &[TRACE_COL_LOG_POSTERIOR]);
        let all: &[f64] = match cols.as_ref().and_then(|c| c[0].as_ref()) {
            Some(v) => v,
            None => &[],
        };
        let selected: &[f64] = match draw_counts.get(i).copied() {
            Some(k) if k > 0 && k <= all.len() => &all[all.len() - k..],
            _ => all,
        };
        out.push(ChainNegInf {
            chain: i + 1,
            n_neginf: selected.iter().filter(|v| !v.is_finite()).count(),
            n_retained: selected.len(),
        });
    }
    Some(out)
}

/// Read `draws.tsv` and return the per-chain kept-draw count, ordered by
/// ascending 0-based `chain` id. Empty when the file is missing or has no
/// `chain` column.
fn read_draw_counts(draws_path: &Path) -> Vec<usize> {
    let Ok(contents) = std::fs::read_to_string(draws_path) else {
        return Vec::new();
    };
    let mut lines = contents.lines();
    let Some(header) = lines.next() else {
        return Vec::new();
    };
    let cols: Vec<&str> = header.split('\t').collect();
    let Some(chain_col) = cols.iter().position(|c| *c == "chain") else {
        return Vec::new();
    };
    let mut counts: std::collections::BTreeMap<usize, usize> = std::collections::BTreeMap::new();
    for line in lines {
        if line.trim().is_empty() {
            continue;
        }
        let fields: Vec<&str> = line.split('\t').collect();
        if let Some(id) = fields.get(chain_col).and_then(|s| s.parse::<usize>().ok()) {
            *counts.entry(id).or_insert(0) += 1;
        }
    }
    counts.into_values().collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn flags_a_single_clear_outlier() {
        // Five well-mixed chains near -100 (jittered, as real chains are), one
        // stuck far below at -400.
        let means = vec![-100.0, -101.0, -99.0, -100.5, -100.2, -400.0];
        let scores = chain_loglik_mod_zscores(&means);
        assert_eq!(scores.len(), 6);
        let stuck = &scores[5];
        assert_eq!(stuck.chain, 6);
        assert!(stuck.is_outlier, "stuck chain must flag: mod_z = {}", stuck.mod_z);
        assert!(stuck.mod_z < -CHAIN_LOGLIK_OUTLIER_MODZ, "mod_z strongly negative: {}", stuck.mod_z);
        for sc in &scores[..5] {
            assert!(!sc.is_outlier, "well-mixed chain {} wrongly flagged (mod_z={})", sc.chain, sc.mod_z);
        }
        assert_eq!(outlier_labels(&scores), vec!["chain 6"]);
    }

    /// THE point of the robust upgrade: two co-stuck chains among six now BOTH
    /// flag (the classic mean/SD z masked them — they inflated the SD and pulled
    /// their own |z| below threshold). Good chains carry realistic jitter so the
    /// MAD is non-zero (real MCMC means are never numerically identical).
    #[test]
    fn four_good_two_stuck_flags_both_stuck_chains() {
        // 4 good (jittered) at ≈ -1204, 2 stuck at ≈ -1461.
        let means = vec![-1204.0, -1205.0, -1203.0, -1206.0, -1461.0, -1462.0];
        let scores = chain_loglik_mod_zscores(&means);
        assert!(scores[4].is_outlier, "stuck chain 5 must flag: mod_z={}", scores[4].mod_z);
        assert!(scores[5].is_outlier, "stuck chain 6 must flag: mod_z={}", scores[5].mod_z);
        assert!(
            scores[..4].iter().all(|s| !s.is_outlier),
            "good chains must not flag: {:?}",
            scores[..4].iter().map(|s| s.mod_z).collect::<Vec<_>>()
        );
        let labels = outlier_labels(&scores);
        assert!(labels.contains(&"chain 5".to_string()), "must name chain 5: {labels:?}");
        assert!(labels.contains(&"chain 6".to_string()), "must name chain 6: {labels:?}");
        assert_eq!(labels.len(), 2, "exactly the two stuck chains: {labels:?}");
    }

    #[test]
    fn flags_nothing_when_well_mixed() {
        // Negative control: all chains agree — no outlier, and the scores are
        // small and finite (not a vacuous "everything is NaN" pass).
        let means = vec![-100.0, -100.3, -99.8, -100.1, -99.9, -100.2];
        let scores = chain_loglik_mod_zscores(&means);
        assert!(scores.iter().all(|s| !s.is_outlier), "well-mixed set must flag nothing");
        assert!(scores.iter().all(|s| s.mod_z.is_finite()), "scores must be finite");
        assert!(scores.iter().all(|s| s.mod_z.abs() < CHAIN_LOGLIK_OUTLIER_MODZ));
        assert!(outlier_labels(&scores).is_empty());
    }

    #[test]
    fn single_chain_has_no_spread() {
        let scores = chain_loglik_mod_zscores(&[-42.0]);
        assert_eq!(scores.len(), 1);
        assert_eq!(scores[0].mod_z, 0.0);
        assert!(!scores[0].is_outlier);
    }

    /// MAD == 0 (majority of chains numerically identical) must not divide by
    /// zero: it falls back to the MeanAD scale, which flags the lone outlier
    /// with a FINITE score (never NaN).
    #[test]
    fn mad_zero_meanad_fallback_flags_outlier_without_nan() {
        // Nine identical good chains at -50, one outlier at -500 → MAD = 0.
        let mut means = vec![-50.0; 9];
        means.push(-500.0);
        let scores = chain_loglik_mod_zscores(&means);
        assert!(scores.iter().all(|s| s.mod_z.is_finite()), "MeanAD fallback must yield finite scores, not NaN");
        assert!(scores[9].is_outlier, "the outlier must flag via MeanAD: mod_z={}", scores[9].mod_z);
        assert!(scores[..9].iter().all(|s| !s.is_outlier), "identical good chains must not flag");
        assert_eq!(outlier_labels(&scores), vec!["chain 10"]);
    }

    #[test]
    fn all_identical_chains_yield_zero_not_nan() {
        // Both MAD and MeanAD are 0 → scale 0 → z = 0, no outliers, no NaN.
        let scores = chain_loglik_mod_zscores(&[-10.0, -10.0, -10.0, -10.0]);
        assert!(scores.iter().all(|s| s.mod_z == 0.0), "identical chains → mod_z = 0, not NaN");
        assert!(scores.iter().all(|s| !s.is_outlier));
    }

    /// A chain with an unreadable trace (NaN mean) must not read as a clean
    /// "not an outlier" with a real score — its mod_z is NaN (rendered `—`), and
    /// it does not poison the finite chains' median/MAD.
    #[test]
    fn non_finite_mean_is_not_silently_clean() {
        let means = vec![-50.0, -51.0, -49.0, f64::NAN, -400.0];
        let scores = chain_loglik_mod_zscores(&means);
        assert!(scores[3].mod_z.is_nan(), "broken chain's mod_z must be NaN, not a fake number");
        assert!(!scores[3].is_outlier, "a NaN score is not an outlier VERDICT (it is 'unassessable')");
        // The real outlier among the finite chains is still flagged.
        assert!(scores[4].is_outlier, "the -400 outlier must still flag: mod_z={}", scores[4].mod_z);
    }

    /// End-to-end reader: synthesize a six-chain stage dir (`chain_N/trace.tsv`,
    /// one stuck) + a `draws.tsv` manifest, with a warm-up prefix in each trace
    /// that MUST be stripped. Good chains carry jitter (distinct means) so the
    /// robust MAD is non-zero.
    #[test]
    fn reader_strips_warmup_and_names_the_stuck_chain() {
        let dir = crate::test_support::unique_temp_dir("chain_diag_reader");
        std::fs::create_dir_all(&dir).unwrap();

        let warmup = "1\t-900.0\t-905.0\n2\t-880.0\t-885.0\n";
        // dir numbering starts at 1 (pmmh/pgas convention). Distinct good means.
        let write_trace = |c: usize, kept: &str| {
            let cd = dir.join(format!("chain_{c}"));
            std::fs::create_dir_all(&cd).unwrap();
            let body = format!("step\tlog_likelihood\tlog_posterior\n{warmup}{kept}");
            std::fs::write(cd.join("trace.tsv"), body).unwrap();
        };
        write_trace(1, "3\t-50.0\t-52.0\n4\t-50.0\t-52.0\n5\t-50.0\t-52.0\n"); // mean -50.0
        write_trace(2, "3\t-50.5\t-52.0\n4\t-50.5\t-52.0\n5\t-50.5\t-52.0\n"); // mean -50.5
        write_trace(3, "3\t-49.5\t-51.0\n4\t-49.5\t-51.0\n5\t-49.5\t-51.0\n"); // mean -49.5
        write_trace(4, "3\t-50.2\t-52.0\n4\t-50.2\t-52.0\n5\t-50.2\t-52.0\n"); // mean -50.2
        write_trace(5, "3\t-49.8\t-51.0\n4\t-49.8\t-51.0\n5\t-49.8\t-51.0\n"); // mean -49.8
        write_trace(6, "3\t-300.0\t-302.0\n4\t-301.0\t-303.0\n5\t-299.0\t-301.0\n"); // stuck ≈ -300

        let mut draws = String::from("chain\tdraw\tbeta\n");
        for c in 0..6 {
            for d in 0..3 {
                draws.push_str(&format!("{c}\t{d}\t0.5\n"));
            }
        }
        std::fs::write(dir.join("draws.tsv"), draws).unwrap();

        // A marginal sampler: trace column 1 is `log_likelihood` = log p(y | θ),
        // already comparable across chains, so gh#667 leaves this path alone.
        let means = read_chain_mean_logliks(&dir, LoglikType::Marginal)
            .expect("chain traces present");
        assert_eq!(means.scored_column, "log_likelihood");
        assert!(means.complete_data.is_none(), "no latent-path split for a marginal sampler");
        let means = means.scored;
        assert_eq!(means.len(), 6);
        // Warm-up (-900/-880) excluded → good chains ≈ -50, not dragged to -890.
        for (i, m) in means.iter().take(5).enumerate() {
            assert!((m - (-50.0)).abs() < 1.0, "chain {} mean {} ≈ -50 (warm-up stripped)", i + 1, m);
        }
        assert!((means[5] - (-300.0)).abs() < 1.0, "chain 6 (stuck) mean {} ≈ -300", means[5]);

        let scores = chain_loglik_mod_zscores(&means);
        assert!(scores[5].is_outlier, "stuck chain 6 must flag: mod_z={}", scores[5].mod_z);
        assert!(scores[..5].iter().all(|s| !s.is_outlier), "good chains must not flag");
        assert_eq!(outlier_labels(&scores), vec!["chain 6"]);

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn reader_returns_none_without_chain_dirs() {
        let dir = crate::test_support::unique_temp_dir("chain_diag_empty");
        std::fs::create_dir_all(&dir).unwrap();
        assert!(read_chain_mean_logliks(&dir, LoglikType::Marginal).is_none(),
            "no chain_* dirs → None");
        std::fs::remove_dir_all(&dir).ok();
    }

    /// A six-chain PGAS stage whose complete-data ranking and observation-only
    /// ranking DISAGREE, so the fixture discriminates between them.
    ///
    /// - chain 3 has a hugely concentrated latent path (`transition_ll` −800 vs
    ///   ≈ −3000 elsewhere) and a perfectly ordinary data fit.
    /// - chain 6 has an ordinary latent path and reproduces the data ≈450 nats
    ///   worse than every other chain.
    ///
    /// Ranking on `log_complete_data_ll` flags chain 3 — a chain that fits the
    /// data as well as any other. Ranking on `obs_ll` flags chain 6.
    fn write_disagreeing_pgas_stage(dir: &std::path::Path) {
        std::fs::create_dir_all(dir).unwrap();
        // (transition_ll, obs_ll) per chain; complete = transition + obs.
        let chains = [
            (-2832.6, -952.0),
            (-2933.0, -952.8),
            (-800.0, -951.7),    // concentrated path, ordinary data fit
            (-3100.9, -952.9),
            (-3172.6, -952.3),
            (-3000.0, -1400.0),  // ordinary path, BAD data fit
        ];
        for (i, (trans, obs)) in chains.iter().enumerate() {
            let cd = dir.join(format!("chain_{}", i + 1));
            std::fs::create_dir_all(&cd).unwrap();
            let complete = trans + obs;
            let mut body = String::from(
                "sweep\tlog_complete_data_ll\tlog_posterior\ttransition_ll\tobs_ll\n",
            );
            // Two warm-up rows the last-K_c rule must strip (draws.tsv keeps 3).
            for s in 0..2 {
                body.push_str(&format!("{s}\t-9000\t-9100\t-8000\t-1000\n"));
            }
            for s in 2..5 {
                body.push_str(&format!(
                    "{s}\t{complete}\t{}\t{trans}\t{obs}\n",
                    complete - 1.0
                ));
            }
            std::fs::write(cd.join("trace.tsv"), body).unwrap();
        }
        let mut draws = String::from("chain\tdraw\tbeta\n");
        for c in 0..6 {
            for d in 0..3 {
                draws.push_str(&format!("{c}\t{d}\t0.{}{}\n", c, d));
            }
        }
        std::fs::write(dir.join("draws.tsv"), draws).unwrap();
    }

    /// gh#667: a PGAS chain is scored on `obs_ll` = `log p(y | X, θ)` — "does
    /// this chain reproduce the data" — NOT on `log_complete_data_ll`, whose
    /// latent-path term is a density at one sampled path and rewards a
    /// concentrated path distribution rather than a better fit.
    #[test]
    fn pgas_chains_are_scored_on_obs_ll_not_the_complete_data_target() {
        let dir = crate::test_support::unique_temp_dir("chain_diag_gh667");
        write_disagreeing_pgas_stage(&dir);

        let means = read_chain_mean_logliks(&dir, LoglikType::CompleteData)
            .expect("six chain traces");
        assert_eq!(means.scored_column, "obs_ll",
            "PGAS chains are compared on log p(y | X, θ)");
        assert!(!means.scored_column_absent);
        assert_eq!(means.scored.len(), 6);

        // The scored quantity IS obs_ll (warm-up stripped: −1000 never enters).
        assert!((means.scored[5] - (-1400.0)).abs() < 1e-6,
            "chain 6 must be scored on its obs_ll (−1400), got {}", means.scored[5]);
        assert!((means.scored[2] - (-951.7)).abs() < 1e-6,
            "chain 3 must be scored on its obs_ll (−951.7), got {}", means.scored[2]);

        // …and the flag follows it: the chain that fits the DATA worst.
        let scores = chain_loglik_mod_zscores(&means.scored);
        assert_eq!(outlier_labels(&scores), vec!["chain 6"],
            "the badly-fitting chain is the outlier: {:?}",
            scores.iter().map(|s| s.mod_z).collect::<Vec<_>>());
        assert!(!scores[2].is_outlier,
            "chain 3 fits the data like every other chain and must NOT flag \
             (mod_z = {})", scores[2].mod_z);

        // The fixture genuinely discriminates: the complete-data column the old
        // reader ranked on flags the OTHER chain, and only it. It is still READ
        // (kept visible in the table) — it is just no longer the ranking key.
        let split = means.complete_data.as_ref().expect("PGAS carries the split");
        assert!((split.complete[2] - (-1751.7)).abs() < 1e-6,
            "chain 3's complete-data target is still read: {}", split.complete[2]);
        assert!((split.transition[2] - (-800.0)).abs() < 1e-6,
            "chain 3's transition term is still read: {}", split.transition[2]);
        let cd_scores = chain_loglik_mod_zscores(&split.complete);
        assert_eq!(outlier_labels(&cd_scores), vec!["chain 3"],
            "precondition: ranking on log_complete_data_ll flags the \
             concentrated-path chain, not the badly-fitting one: {:?}",
            cd_scores.iter().map(|s| s.mod_z).collect::<Vec<_>>());

        std::fs::remove_dir_all(&dir).ok();
    }

    /// gh#667: the split shown beside the scored column is the real
    /// decomposition — 522 nats of between-chain spread in `transition_ll`
    /// against 9 in `obs_ll` is what makes "this is an entropy effect, not a
    /// fit difference" legible. Uses the six chains of the real 60,000-sweep
    /// fit in the issue, whose complete-data spread is genuinely dominated by
    /// the latent-path term.
    #[test]
    fn transition_and_obs_spreads_are_read_separately() {
        let dir = crate::test_support::unique_temp_dir("chain_diag_gh667_split");
        std::fs::create_dir_all(&dir).unwrap();
        // The issue's table: (transition_ll, obs_ll) for the 6 finite chains.
        let chains = [
            (-2832.6, -952.0), (-2933.0, -952.8), (-2972.0, -951.7),
            (-3100.9, -958.0), (-3172.6, -952.9), (-3354.4, -960.4),
        ];
        for (i, (trans, obs)) in chains.iter().enumerate() {
            let cd = dir.join(format!("chain_{}", i + 1));
            std::fs::create_dir_all(&cd).unwrap();
            let complete = trans + obs;
            std::fs::write(cd.join("trace.tsv"), format!(
                "sweep\tlog_complete_data_ll\tlog_posterior\ttransition_ll\tobs_ll\n\
                 0\t{complete}\t{}\t{trans}\t{obs}\n", complete - 1.0)).unwrap();
        }
        let means = read_chain_mean_logliks(&dir, LoglikType::CompleteData).expect("traces");
        let split = means.complete_data.as_ref().expect("PGAS split");
        let range = |v: &[f64]| {
            v.iter().cloned().fold(f64::NEG_INFINITY, f64::max)
                - v.iter().cloned().fold(f64::INFINITY, f64::min)
        };
        assert!((range(&split.transition) - 521.8).abs() < 0.1,
            "latent-path spread ≈ 522 nats, got {}", range(&split.transition));
        assert!((range(&means.scored) - 8.7).abs() < 0.1,
            "observation spread ≈ 9 nats, got {}", range(&means.scored));

        // The two columns rank the chains differently on the REAL fit too, not
        // only on a constructed cohort: chain 6 is 8.7 nats below best on data
        // fit and 522 below best on the latent path, and only the observation
        // column names it.
        let obs_flagged = outlier_labels(&chain_loglik_mod_zscores(&means.scored));
        let cd_flagged = outlier_labels(&chain_loglik_mod_zscores(&split.complete));
        assert_eq!(obs_flagged, vec!["chain 6"], "obs_ll names the worst DATA fit");
        assert!(cd_flagged.is_empty(),
            "the complete-data column, whose spread is 60× larger, names NOBODY \
             — its own scale swamps the differences it would flag: {cd_flagged:?}");
        // NOTE (gh#664): that obs_ll flag fires at |mod-z| = 5.09 on a spread of
        // 8.7 nats, because the modified z-score is scale-free — it has no
        // notion of how many nats matter. gh#667 fixes WHICH quantity is
        // scored; making the threshold mean something is gh#664, and this
        // assertion pins the pre-gh#664 behaviour so that change shows up here.

        std::fs::remove_dir_all(&dir).ok();
    }

    /// Negative control for gh#667: when the two rankings AGREE — one chain is
    /// off on the latent path AND reproduces the data far worse — the flag
    /// still lands on it. Narrowing the scored quantity must not cost the
    /// diagnostic the cases it already caught.
    #[test]
    fn a_chain_bad_on_both_terms_is_still_flagged() {
        let dir = crate::test_support::unique_temp_dir("chain_diag_gh667_agree");
        std::fs::create_dir_all(&dir).unwrap();
        // Five chains agree; chain 6 is worse on BOTH terms.
        let chains = [
            (-2900.0, -952.0), (-2910.0, -952.8), (-2890.0, -951.7),
            (-2905.0, -952.9), (-2895.0, -952.3), (-3900.0, -1400.0),
        ];
        for (i, (trans, obs)) in chains.iter().enumerate() {
            let cd = dir.join(format!("chain_{}", i + 1));
            std::fs::create_dir_all(&cd).unwrap();
            let complete = trans + obs;
            std::fs::write(cd.join("trace.tsv"), format!(
                "sweep\tlog_complete_data_ll\tlog_posterior\ttransition_ll\tobs_ll\n\
                 0\t{complete}\t{}\t{trans}\t{obs}\n", complete - 1.0)).unwrap();
        }
        let means = read_chain_mean_logliks(&dir, LoglikType::CompleteData).expect("traces");
        let split = means.complete_data.as_ref().expect("PGAS split");
        // Precondition: this cohort is one both columns agree on…
        assert_eq!(outlier_labels(&chain_loglik_mod_zscores(&split.complete)), vec!["chain 6"]);
        // …and the scored column still names it.
        assert_eq!(outlier_labels(&chain_loglik_mod_zscores(&means.scored)), vec!["chain 6"]);

        std::fs::remove_dir_all(&dir).ok();
    }

    /// gh#667: the reader keys on the column NAME. A PGAS trace whose position-1
    /// column is `log_complete_data_ll` must be read on `obs_ll` — the position
    /// is not consulted, so the two quantities cannot be substituted for each
    /// other. The fixture makes them numerically distinguishable on purpose.
    #[test]
    fn reader_keys_on_the_name_so_position_1_cannot_stand_in() {
        let dir = crate::test_support::unique_temp_dir("chain_diag_pgas");
        let cd = dir.join("chain_1");
        std::fs::create_dir_all(&cd).unwrap();
        std::fs::write(
            cd.join("trace.tsv"),
            "sweep\tlog_complete_data_ll\tlog_posterior\ttransition_ll\tobs_ll\n\
             0\t-77.0\t-79.0\t-55.0\t-22.0\n1\t-77.0\t-79.0\t-55.0\t-22.0\n",
        )
        .unwrap();
        let means = read_chain_mean_logliks(&dir, LoglikType::CompleteData).expect("one chain");
        assert!((means.scored[0] - (-22.0)).abs() < 1e-9,
            "must score obs_ll (-22), not the position-1 column (-77): {}", means.scored[0]);
        let split = means.complete_data.as_ref().expect("PGAS split");
        assert!((split.complete[0] - (-77.0)).abs() < 1e-9);
        assert!((split.transition[0] - (-55.0)).abs() < 1e-9);
        std::fs::remove_dir_all(&dir).ok();
    }

    /// A PGAS trace with no `obs_ll` column cannot be scored at all — the
    /// reader says the column is absent rather than falling back to whatever
    /// sits in position 1, which is exactly the substitution gh#667 removed.
    #[test]
    fn missing_scored_column_is_reported_not_silently_substituted() {
        let dir = crate::test_support::unique_temp_dir("chain_diag_no_obs");
        let cd = dir.join("chain_1");
        std::fs::create_dir_all(&cd).unwrap();
        std::fs::write(
            cd.join("trace.tsv"),
            "sweep\tlog_complete_data_ll\tlog_posterior\n0\t-77.0\t-79.0\n1\t-77.0\t-79.0\n",
        )
        .unwrap();
        let means = read_chain_mean_logliks(&dir, LoglikType::CompleteData).expect("one chain");
        assert!(means.scored_column_absent, "the absence must be REPORTED");
        assert!(means.scored[0].is_nan(),
            "no obs_ll → no score; never the complete-data value: {}", means.scored[0]);
        std::fs::remove_dir_all(&dir).ok();
    }

    /// A file that is not a `TraceWriter` trace at all (no `log_posterior`)
    /// yields NaN means, and is NOT mistaken for "a trace missing a column".
    #[test]
    fn a_non_trace_file_is_not_a_missing_column() {
        let dir = crate::test_support::unique_temp_dir("chain_diag_not_trace");
        let cd = dir.join("chain_1");
        std::fs::create_dir_all(&cd).unwrap();
        std::fs::write(cd.join("trace.tsv"), "a\tb\tc\n1\t2\t3\n").unwrap();
        let means = read_chain_mean_logliks(&dir, LoglikType::Marginal).expect("one chain dir");
        assert!(!means.scored_column_absent,
            "an unreadable trace is not evidence about which columns a trace has");
        assert!(means.scored[0].is_nan());
        std::fs::remove_dir_all(&dir).ok();
    }
}
