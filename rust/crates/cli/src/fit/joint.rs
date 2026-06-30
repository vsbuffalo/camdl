//! The keyed-joint `(θ, X)` read: pair each posterior draw with its smoothed
//! latent trajectory, classified by [`LatentPath`]. This is the reader the
//! conditioned counterfactual fork (the engine seam + contrasts) consumes.
//!
//! The join is PARTIAL by design (the (θ,X) spec): only the path-saved subset is
//! `Sampled`; the rest is `NotSaved`. The forkable count is surfaced so a
//! contrast bands honestly over the subset, never silently over fewer draws than
//! the parameter posterior.
//!
//! gh#322; docs/dev/proposals/2026-06-28-keyed-joint-param-trajectory-output.md.

use std::collections::{BTreeSet, HashMap};
use std::path::Path;

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
    /// All posterior draws (the parameter-posterior size).
    pub n_total: usize,
}

/// Classify each keyed draw against the set of `(chain, draw)` keys that have a
/// saved trajectory. Pure (no I/O): the join + classification logic, factored so
/// it is unit-testable without writing fixtures to disk.
///
/// `is_ode`: a deterministic backend — every draw is `Deterministic` (X is
/// recomputed from θ), so `traj_keys` is irrelevant.
pub fn classify_joint(
    keyed: Vec<crate::KeyedDraw>,
    traj_keys: &BTreeSet<(usize, usize)>,
    is_ode: bool,
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
    JointEnsemble { draws, n_forkable, n_total }
}

/// Resolve a fit's keyed-joint `(θ, X)`: inner-join `draws.tsv` to the smoothed
/// `trajectories.tsv` on `(chain, draw)` and classify each draw.
///
/// - ODE fit → every draw is `Deterministic` (recompute X from θ).
/// - chain-binomial fit → `Sampled` iff the draw's `(chain, draw)` has a saved
///   trajectory, else `NotSaved` (the partial join).
pub fn resolve_joint(fit_ref: &str, stage: Option<&str>) -> Result<JointEnsemble, String> {
    let pref = crate::posterior_draws::resolve_posterior_draws(fit_ref, stage)?;
    let keyed = crate::load_draws_tsv_keyed(&pref.draws_path.to_string_lossy())?;
    let stage_dir = pref
        .draws_path
        .parent()
        .ok_or_else(|| format!("draws path has no parent: {}", pref.draws_path.display()))?;

    let is_ode = pref.backend == Some(InferenceBackend::Ode);
    let traj_keys = if is_ode { BTreeSet::new() } else { trajectory_keys(stage_dir) };
    Ok(classify_joint(keyed, &traj_keys, is_ode))
}

/// Collect every `(chain, draw)` key from the stage's `chain_*/trajectories.tsv`
/// files. Skips the leading `# camdl-trajectories …` comment line and dedups
/// the per-snapshot rows to one key per saved draw.
fn trajectory_keys(stage_dir: &Path) -> BTreeSet<(usize, usize)> {
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
        let j = classify_joint(keyed, &traj, false);
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
        let j = classify_joint(keyed, &BTreeSet::new(), true);
        assert_eq!(j.n_forkable, 2);
        assert!(j.draws.iter().all(|d| d.latent == LatentPath::Deterministic));
    }

    #[test]
    fn keyless_stochastic_draws_are_notsaved() {
        // A pre-key draws.tsv (no chain/draw) → unjoinable → NotSaved.
        let keyed = vec![draw(None, None), draw(None, None)];
        let j = classify_joint(keyed, &BTreeSet::new(), false);
        assert_eq!(j.n_forkable, 0);
        assert!(j.draws.iter().all(|d| d.latent == LatentPath::NotSaved));
    }
}
