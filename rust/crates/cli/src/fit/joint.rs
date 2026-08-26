//! The keyed-joint `(θ, X)` read: pair each posterior draw with its smoothed
//! latent trajectory, classified by [`LatentPath`]. This is the reader the
//! conditioned counterfactual fork (the engine seam + contrasts) consumes.
//!
//! The join is PARTIAL by design (the (θ,X) spec): only the path-saved subset is
//! `Sampled`; the rest is `NotSaved`. The forkable count is surfaced so a
//! contrast bands honestly over the subset, never silently over fewer draws than
//! the parameter posterior.
//!
//! Both resolvers take the [`PosteriorDrawsRef`] — the draws authority — rather
//! than a `(fit_ref, stage)` pair, and read the cloud through
//! [`PosteriorDrawsRef::load_keyed_with_info`]. That is what makes a read-side
//! chain selection (`--exclude-chains`) reach the joint: the filter is applied
//! once, where the ref carries it, so a `(θ, X)` consumer cannot band over a
//! chain the same command reports as excluded (gh#695).
//!
//! gh#322; docs/dev/proposals/2026-06-28-keyed-joint-param-trajectory-output.md.

use std::collections::{BTreeSet, HashMap};
use std::path::Path;

use crate::chain_selection::SubsetInfo;
use crate::posterior_draws::PosteriorDrawsRef;
use crate::run_meta::InferenceBackend;

/// Whether (and how) a posterior draw's latent state X is recoverable for a
/// conditioned fork — classified by the latent ARTIFACT, not the method name.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LatentPath {
    /// Deterministic backend (ODE): X = integrate(θ) exactly. Forkable; the path
    /// is recomputed from θ on demand, nothing stored.
    Deterministic,
    /// A stochastic backend with a stored smoothed path for this draw. Forkable;
    /// the path is loaded LAZILY by its `(chain, draw)` locator at fork time
    /// (step 3) — not held eagerly, since even the partial subset of
    /// substep-resolution paths is large at national scale.
    Sampled { chain: usize, draw: usize },
    /// A stochastic fit with NO saved path for this draw — every PMMH/PF fit
    /// today, and any PGAS draw outside the saved (`traj_stride`) subset. NOT
    /// forkable; a contrast skips it (and the skipped count is surfaced).
    NotSaved,
}

impl LatentPath {
    /// Forkable = a usable latent state exists (deterministic, or a stored path).
    pub fn is_forkable(&self) -> bool {
        !matches!(self, LatentPath::NotSaved)
    }
}

/// One paired posterior draw: its parameter vector (θ) and its latent
/// classification (where to recover the smoothed `X(T*)` for a conditioned fork).
///
/// Both fields are consumed live by the counterfactual contrasts reducer
/// ([`crate::fit::contrasts::emit_contrasts`]): per forkable draw it reads
/// `params` to resolve each arm's θ (the 5-tier resolver) and `latent` to locate
/// the saved path to fork from. `fit summary` additionally reports the
/// [`JointEnsemble`] counts.
pub struct JointDraw {
    pub params: HashMap<String, f64>,
    pub latent: LatentPath,
}

/// The keyed-joint posterior: every draw paired with its [`LatentPath`], plus
/// the forkable-subset count — the honest denominator for a contrast band.
pub struct JointEnsemble {
    /// Per-draw pairing — see [`JointDraw`]; consumed by the contrasts reducer.
    pub draws: Vec<JointDraw>,
    /// Draws with a usable latent state (`Deterministic` + `Sampled`).
    pub n_forkable: usize,
    /// The posterior draws this ensemble was resolved from — **after** any chain
    /// selection. With `--exclude-chains` active it is the RETAINED count, so
    /// `n_forkable`/`n_total` is the honest denominator pair for the cloud the
    /// caller actually bands over, not for a cloud that includes dropped chains.
    pub n_total: usize,
    /// The chain selection that produced this cloud (`--exclude-chains`), when
    /// one was active; `None` is the full cloud. Carried so a consumer NAMES the
    /// subset in its own output rather than reporting a subset count as if it
    /// were the whole posterior (gh#695).
    pub selection: Option<SubsetInfo>,
}

/// Classify each keyed draw against the set of `(chain, draw)` keys that have a
/// saved trajectory. Pure (no I/O): the join + classification logic, factored so
/// it is unit-testable without writing fixtures to disk.
///
/// `is_ode`: a deterministic backend — every draw is `Deterministic` (X is
/// recomputed from θ), so `traj_keys` is irrelevant.
///
/// `selection` is the provenance of the cloud handed in: `keyed` is expected to
/// be POST-filter, so the counts derived here describe the retained cloud and
/// the record travels with them.
pub fn classify_joint(
    keyed: Vec<crate::KeyedDraw>,
    traj_keys: &BTreeSet<(usize, usize)>,
    is_ode: bool,
    selection: Option<SubsetInfo>,
) -> JointEnsemble {
    let n_total = keyed.len();
    let mut n_forkable = 0usize;
    let draws = keyed
        .into_iter()
        .map(|d| {
            let latent = if is_ode {
                LatentPath::Deterministic
            } else {
                match (d.chain, d.draw) {
                    (Some(c), Some(dr)) if traj_keys.contains(&(c, dr)) => {
                        LatentPath::Sampled { chain: c, draw: dr }
                    }
                    _ => LatentPath::NotSaved,
                }
            };
            if latent.is_forkable() {
                n_forkable += 1;
            }
            JointDraw { params: d.params, latent }
        })
        .collect();
    JointEnsemble { draws, n_forkable, n_total, selection }
}

/// Resolve a fit's keyed-joint `(θ, X)`: inner-join `draws.tsv` to the smoothed
/// `trajectories.tsv` on `(chain, draw)` and classify each draw.
///
/// - ODE fit → every draw is `Deterministic` (recompute X from θ).
/// - chain-binomial fit → `Sampled` iff the draw's `(chain, draw)` has a saved
///   trajectory, else `NotSaved` (the partial join).
///
/// The cloud is read through the ref's own
/// [`load_keyed_with_info`](PosteriorDrawsRef::load_keyed_with_info), so a
/// selection attached with
/// [`with_selection`](PosteriorDrawsRef::with_selection) is applied HERE — a
/// `(θ, X)` consumer cannot fork a chain the same command excluded (gh#695).
/// `trajectory_keys` still scans every `chain_*/` dir, which is harmless: a key
/// belonging to a dropped chain has no retained draw to match.
pub fn resolve_joint(pref: &PosteriorDrawsRef) -> Result<JointEnsemble, String> {
    let (keyed, sel_info) = pref.load_keyed_with_info()?;
    let stage_dir = pref
        .draws_path
        .parent()
        .ok_or_else(|| format!("draws path has no parent: {}", pref.draws_path.display()))?;

    let is_ode = pref.backend == Some(InferenceBackend::Ode);
    let traj_keys = if is_ode { BTreeSet::new() } else { trajectory_keys(stage_dir) };
    Ok(classify_joint(keyed, &traj_keys, is_ode, sel_info))
}

// ── The terminal-origin specialization (gh#697) ─────────────────────────────

/// One paired posterior draw at the forecast origin: its parameter vector θ_i
/// and the terminal state X_i(T) of its own saved latent path.
///
/// θ and X live in **one struct**, not two parallel vectors, because the
/// pairing is the whole point: `--init-state fit` exists so that draw i runs at
/// its own parameters from its own inferred state, and a shuffled pairing still
/// produces a plausible-looking cloud. Keeping them together means a shuffle
/// has to be written deliberately rather than introduced by an index slip.
#[derive(Debug, Clone)]
pub struct ForecastDraw {
    pub params: HashMap<String, f64>,
    /// The `(chain, draw)` key this pair came from — provenance for the error
    /// messages, and part of what keys the run (see `runid::inputs::
    /// InitStateRow`).
    pub chain: usize,
    pub draw: usize,
    /// Integer compartment counts at the origin, in model order.
    pub counts: Vec<i64>,
    /// Real compartment values at the origin, in model order.
    pub reals: Vec<f64>,
}

/// The paired `(θ_i, X_i(T))` ensemble a terminal-origin forecast runs from,
/// with the honest denominator alongside it.
pub struct ForecastEnsemble {
    /// The model time every path ends at — the forecast origin. Verified equal
    /// across draws by [`resolve_forecast_ensemble`], never assumed. `NaN` when
    /// `draws` is empty (there is no origin), which the caller never reads: an
    /// empty ensemble is refused first, with `n_total` in the message.
    pub origin_t: f64,
    /// The forkable draws, in `draws.tsv` order.
    pub draws: Vec<ForecastDraw>,
    /// All posterior draws (the parameter-posterior size) — the denominator the
    /// forkable subset must be reported against.
    pub n_total: usize,
    /// Provenance for the run's report: which stage's cloud this is, and the
    /// `draws.tsv` it was read from. The unpaired `--draws posterior` path
    /// prints the same two, so a user can tell the clouds apart in a log.
    pub stage: String,
    pub draws_path: std::path::PathBuf,
}

/// Resolve a fit's terminal-origin paired ensemble: every posterior draw that
/// has a saved latent path, joined to the last snapshot of that path.
///
/// At the terminal observation time the smoothing distribution equals the
/// filtering distribution (no future data remains to condition on), so the last
/// row of each stored path is a draw from `p(x_T | y_{1:T})` carrying its own θ
/// — the prediction distribution's origin in Särkkä's taxonomy. Interior
/// origins are deliberately out of scope (gh#641): iterating forward from a
/// *smoothing* draw has no cell in that taxonomy.
///
/// Errors, never guesses, when: the fit ran on a deterministic backend (no
/// stored paths — ODE forking is the separate gh#325 follow-up); or the saved
/// paths do not all end at the same time (one shared origin is what makes the
/// cloud a single forecast, so a disagreement is named rather than averaged).
/// A fit with *no* saved paths returns an empty `draws` with the real
/// `n_total`, so the caller can refuse with both numbers in hand.
///
/// Like [`resolve_joint`], the cloud comes through the ref's own load method, so
/// whatever chain selection the ref carries is already applied when the pairing
/// happens. `simulate` attaches none today (it has no `--exclude-chains`), so
/// this is a no-op there — and the day it grows one, attaching it to the ref is
/// the whole change (gh#695).
pub fn resolve_forecast_ensemble(
    pref: &PosteriorDrawsRef,
    columns: &io::trajectories::TrajColumnSpec,
) -> Result<ForecastEnsemble, String> {
    if pref.backend == Some(InferenceBackend::Ode) {
        return Err("--init-state fit: this fit ran on the ode backend, which stores no \
             latent paths — X is recomputed from θ, and the re-integration seam a \
             forecast would need is not wired (gh#325).\n  \
             Fix: forecast from a chain_binomial (PGAS) fit, or run the ODE forward \
             from the model's own t_start with --draws posterior.".to_string());
    }
    let (keyed, sel_info) = pref.load_keyed_with_info()?;
    let stage_dir = pref
        .draws_path
        .parent()
        .ok_or_else(|| format!("draws path has no parent: {}", pref.draws_path.display()))?;

    // One pass per chain file: every saved path's terminal snapshot, keyed by
    // (chain, draw). This is also the "has a saved path" predicate the
    // classifier below uses, so the forkable count and the states it resolves
    // can never disagree.
    let mut terminal: HashMap<(usize, usize), io::trajectories::TerminalState> = HashMap::new();
    let mut chain_files: Vec<std::path::PathBuf> = Vec::new();
    if let Ok(entries) = std::fs::read_dir(stage_dir) {
        for e in entries.flatten() {
            let traj = e.path().join("trajectories.tsv");
            if traj.is_file() {
                chain_files.push(traj);
            }
        }
    }
    chain_files.sort();
    for traj in &chain_files {
        for ts in io::trajectories::read_terminal_states(traj, columns)? {
            terminal.insert((ts.chain, ts.draw), ts);
        }
    }

    let traj_keys: BTreeSet<(usize, usize)> = terminal.keys().copied().collect();
    let joint = classify_joint(keyed, &traj_keys, false, sel_info);
    let n_total = joint.n_total;

    let mut draws: Vec<ForecastDraw> = Vec::with_capacity(joint.n_forkable);
    let mut origin: Option<(f64, usize, usize)> = None;
    for d in joint.draws {
        let LatentPath::Sampled { chain, draw } = d.latent else {
            continue;
        };
        let ts = terminal
            .get(&(chain, draw))
            .expect("classified Sampled from the terminal map's own keys");
        match origin {
            None => origin = Some((ts.t, chain, draw)),
            Some((t0, c0, d0)) => {
                if (ts.t - t0).abs() > io::trajectories::SNAPSHOT_TIME_TOL {
                    return Err(format!(
                        "--init-state fit: the saved latent paths do not share one \
                         terminal time — (chain {c0}, draw {d0}) ends at t = {t0} but \
                         (chain {chain}, draw {draw}) ends at t = {}. A forecast cloud \
                         has ONE origin; forking these together would band states taken \
                         at different instants.\n  \
                         Fix: re-fit so every chain runs the same observation window.",
                        ts.t
                    ));
                }
            }
        }
        draws.push(ForecastDraw {
            params: d.params,
            chain,
            draw,
            counts: ts.int_state.counts.clone(),
            reals: ts.real_state.values.clone(),
        });
    }

    Ok(ForecastEnsemble {
        origin_t: origin.map(|(t, _, _)| t).unwrap_or(f64::NAN),
        draws,
        n_total,
        stage: pref.stage.clone(),
        draws_path: pref.draws_path.clone(),
    })
}

/// Collect every `(chain, draw)` key from the stage's `chain_*/trajectories.tsv`
/// files. Skips the leading `# camdl-trajectories …` comment line and dedups
/// the per-snapshot rows to one key per saved draw.
///
/// This is the "has a saved smoothing path" predicate for the whole crate —
/// [`classify_joint`] and the `quantities/` conditioned read (gh#722) both fold
/// through it, so the forkable count a command reports and the paths it can
/// actually open cannot disagree.
pub(crate) fn trajectory_keys(stage_dir: &Path) -> BTreeSet<(usize, usize)> {
    let mut keys = BTreeSet::new();
    let Ok(entries) = std::fs::read_dir(stage_dir) else {
        return keys;
    };
    for e in entries.flatten() {
        let traj = e.path().join("trajectories.tsv");
        if !traj.is_file() {
            continue;
        }
        let Ok(txt) = std::fs::read_to_string(&traj) else {
            continue;
        };
        let mut lines = txt.lines().filter(|l| !l.starts_with('#'));
        let Some(header) = lines.next() else {
            continue;
        };
        let cols: Vec<&str> = header.split('\t').collect();
        let (Some(ci), Some(di)) = (
            cols.iter().position(|c| *c == "chain"),
            cols.iter().position(|c| *c == "draw"),
        ) else {
            continue;
        };
        for l in lines {
            if l.trim().is_empty() {
                continue;
            }
            let f: Vec<&str> = l.split('\t').collect();
            if let (Some(Ok(c)), Some(Ok(dr))) = (
                f.get(ci).map(|s| s.parse::<usize>()),
                f.get(di).map(|s| s.parse::<usize>()),
            ) {
                keys.insert((c, dr));
            }
        }
    }
    keys
}

#[cfg(test)]
mod tests {
    use super::*;

    fn draw(chain: Option<usize>, draw: Option<usize>) -> crate::KeyedDraw {
        crate::KeyedDraw { chain, draw, params: HashMap::from([("beta".to_string(), 0.3)]) }
    }

    #[test]
    fn stochastic_partial_join_classifies_saved_vs_notsaved() {
        // 4 draws across 2 chains; only (0,20) and (1,21) have a saved path.
        let keyed = vec![
            draw(Some(0), Some(20)),
            draw(Some(0), Some(21)),
            draw(Some(1), Some(20)),
            draw(Some(1), Some(21)),
        ];
        let traj: BTreeSet<(usize, usize)> = [(0, 20), (1, 21)].into_iter().collect();
        let j = classify_joint(keyed, &traj, false, None);
        assert_eq!(j.n_total, 4);
        assert_eq!(j.n_forkable, 2, "only the 2 path-saved draws are forkable");
        assert_eq!(j.draws[0].latent, LatentPath::Sampled { chain: 0, draw: 20 });
        assert_eq!(j.draws[1].latent, LatentPath::NotSaved);
        assert_eq!(j.draws[2].latent, LatentPath::NotSaved);
        assert_eq!(j.draws[3].latent, LatentPath::Sampled { chain: 1, draw: 21 });
    }

    #[test]
    fn ode_is_all_deterministic_regardless_of_traj() {
        let keyed = vec![draw(Some(0), Some(5)), draw(Some(0), Some(6))];
        // ODE ignores traj keys entirely — every draw recomputes X from θ.
        let j = classify_joint(keyed, &BTreeSet::new(), true, None);
        assert_eq!(j.n_forkable, 2);
        assert!(j.draws.iter().all(|d| d.latent == LatentPath::Deterministic));
    }

    #[test]
    fn keyless_stochastic_draws_are_notsaved() {
        // A pre-key draws.tsv (no chain/draw) → unjoinable → NotSaved.
        let keyed = vec![draw(None, None), draw(None, None)];
        let j = classify_joint(keyed, &BTreeSet::new(), false, None);
        assert_eq!(j.n_forkable, 0);
        assert!(j.draws.iter().all(|d| d.latent == LatentPath::NotSaved));
    }
}
