//! gh#785: the cross-chain path/parameter compatibility matrix,
//! `M[i][j] = log p(x_j | θ_i)`.
//!
//! ## What it answers
//!
//! PGAS alternates two conditional moves: `X | θ, y` (conditional SMC) and
//! `θ | X, y`. When those two blocks are strongly dependent, a chain can
//! acquire a latent path that is compatible with its own parameters and then
//! stop leaving that pair, while BOTH conditional moves still look healthy in
//! isolation. Nothing else camdl reports separates that from the marginal
//! posterior genuinely having several modes: `trajectory_renewal` says how much
//! of the path was refreshed per sweep, not whether the refreshed path is
//! compatible with any OTHER chain's parameters.
//!
//! `M` asks that question directly. Take one representative draw per chain —
//! chain `i`'s parameter vector `θ_i` and the latent path `x_i` it was paired
//! with at that sweep — and score every path under every parameter vector.
//! Chain `i`'s parameters making chain `i`'s path far more probable than any
//! other chain's, symmetrically, is the signature of the augmentation pinning
//! each chain in place. Off-diagonal entries comparable to the diagonal say the
//! chains are exchangeable in the augmented space and whatever separates them
//! lives in the marginal instead.
//!
//! ## Which terms
//!
//! `p(x | θ)` here is the PATH density and nothing else:
//!
//! ```text
//!     M[i][j] = log p(x₀ⱼ | θᵢ) + Σ_s log p(x_{j,s} | x_{j,s-1}, θᵢ)
//!             = initial_state + transition
//! ```
//!
//! The observation term `log p(y | x)` is deliberately EXCLUDED. It is a
//! property of the path and the data, not of the parameters scoring the path,
//! so it would add a `j`-dependent constant to every entry of column `j` and
//! shift the diagonal without changing what the matrix is asking. A reader
//! comparing a diagonal entry against `trace.tsv`'s `log_posterior` (which
//! carries the observation term and the prior) will therefore see a different
//! number, by design; the diagonal is exactly that sweep's
//! `transition_ll + initial_state_ll`.
//!
//! The initial-state term is load-bearing rather than incidental: in a model
//! whose `init {}` declares a law over the seed (`I ~ poisson(rate = I0)`),
//! that term is a large part of what couples θ to the path.
//!
//! ## Chain numbering
//!
//! `chains` is **1-based**, matching the `chain_N/` directories the reader is
//! looking at while reading this file — not the 0-based `chain` key inside
//! `draws.tsv` / `trajectories.tsv` / `chain_starts.tsv`. The convention is
//! stated in the artifact itself (`chain_numbering`) so no consumer has to
//! infer it.
//!
//! ## Which chains appear
//!
//! Only chains that actually sampled. A chain refused at its start (gh#607,
//! `BadInit`) has no path and no draw, so it is omitted from `chains` and from
//! `M` entirely rather than contributing a row of nulls a consumer would have
//! to special-case. `chains[i]` always names the chain that produced row `i`
//! and column `i` of `M`, so the matrix is square and complete for the chains
//! it lists.

use serde::{Deserialize, Serialize};

use sim::compiled_model::CompiledModel;
use sim::error::SimError;
use sim::inference::multi_stream_obs::MultiStreamObsModel;
use sim::inference::particle_filter::Observation;
use sim::inference::pgas::{complete_data_loglik, ObsAtSubstep, PGASTrajectory};

/// One chain's representative `(θ, x)` draw.
///
/// The two halves MUST come from the same sweep. A `θ` from one sweep paired
/// with an `x` from another is a cross term, and putting it on the diagonal
/// would make the whole diagnostic read backwards — a locked chain would look
/// exchangeable. The producer takes both from the cold rung's state at the end
/// of the final sweep, which is one instant.
pub struct ChainDraw<'a> {
    /// 1-based chain id, matching the `chain_N/` output directory.
    pub chain: usize,
    /// The full model parameter vector (estimated and fixed alike), natural
    /// scale — the vector `complete_data_loglik` expects.
    pub params: &'a [f64],
    /// The latent path this parameter vector was paired with.
    pub trajectory: &'a PGASTrajectory,
}

/// The `cross_chain_compat.json` artifact.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CrossChainCompat {
    /// Chain ids in `M`'s row and column order. 1-based (see
    /// [`Self::chain_numbering`]). Always describes `M`'s actual rows: a chain
    /// that never sampled is absent from both.
    pub chains: Vec<usize>,
    /// `M[i][j] = log p(x_j | θ_i)` in nats — row `i` is chain `chains[i]`'s
    /// parameters scoring every chain's path.
    #[serde(rename = "M")]
    pub m: Vec<Vec<f64>>,
    /// Always `true`: `log p(x₀ | θ)` is part of every entry.
    pub includes_initial_state: bool,
    /// Which components of the complete-data log-likelihood `M` is built from.
    pub terms: String,
    /// Which draw per chain each row/column was computed at.
    pub draw: String,
    /// How to read `chains` — stated in the file so a consumer never has to
    /// guess between the 1-based `chain_N/` directories and the 0-based `chain`
    /// column in `draws.tsv`.
    pub chain_numbering: String,
    /// `mean(diag) − mean(offdiag)`, in nats. Large and positive is the
    /// signature of each chain being pinned to its own path.
    pub diagonal_dominance: f64,
    /// `max |M[i][j] − M[j][i]|`, in nats. Separates "chain `j`'s path is
    /// improbable under everything" (which shows up symmetrically) from a
    /// genuine one-way pairing.
    pub asymmetry: f64,
}

/// What [`CrossChainCompat::terms`] says. A constant so the producer and any
/// test asserting the artifact's contract cannot drift.
pub const TERMS: &str = "transition + initial_state";

/// What [`CrossChainCompat::draw`] says.
pub const DRAW: &str = "final sweep (θ and x taken from the same sweep)";

/// What [`CrossChainCompat::chain_numbering`] says.
pub const CHAIN_NUMBERING: &str = "1-based, matching the chain_N/ directories";

impl CrossChainCompat {
    /// Score every chain's path under every chain's parameters.
    ///
    /// `k²` calls to [`complete_data_loglik`], the same density the sampler
    /// evaluates every sweep — this is a read-only diagnostic computed after
    /// the fact and consumes no randomness, so it cannot move any number the
    /// fit already reported.
    ///
    /// Returns `Ok(None)` when fewer than two chains have a draw: `M` is then a
    /// 1×1 matrix whose only entry is the chain's own path term, which
    /// `trace.tsv` already carries, and neither derived number is defined.
    pub fn compute(
        model: &CompiledModel,
        draws: &[ChainDraw<'_>],
        observations: &[Observation],
        dt: f64,
        obs_model: &MultiStreamObsModel,
        obs_at_substep: &ObsAtSubstep,
    ) -> Result<Option<Self>, SimError> {
        if draws.len() < 2 {
            return Ok(None);
        }
        let mut m = Vec::with_capacity(draws.len());
        for row in draws {
            let mut r = Vec::with_capacity(draws.len());
            for col in draws {
                let c = complete_data_loglik(
                    model, col.trajectory, row.params, observations, dt,
                    obs_model, obs_at_substep,
                )?;
                // The path density only: `p(x | θ)`, not `p(y, x | θ)`. See the
                // module docs — `c.observation` is `p(y | x)`, which answers a
                // different question from "is this path compatible with these
                // parameters", and would only add a column-wise constant.
                r.push(c.transition + c.initial_state);
            }
            m.push(r);
        }
        Ok(Some(Self::from_matrix(
            draws.iter().map(|d| d.chain).collect(),
            m,
        )))
    }

    /// Attach the derived numbers to an already-computed matrix.
    ///
    /// Split out from [`Self::compute`] so the two summary statistics — the
    /// numbers a reader actually acts on — are exercisable without a model.
    ///
    /// # Panics
    /// If `m` is not square, if `chains` does not name exactly one chain per
    /// row, or if there are fewer than two chains. All three are producer bugs
    /// that would emit an artifact a consumer cannot index — the last because
    /// both derived numbers need an off-diagonal to exist, so the artifact
    /// would have to carry a `NaN` a reader has to special-case.
    pub fn from_matrix(chains: Vec<usize>, m: Vec<Vec<f64>>) -> Self {
        let k = chains.len();
        assert!(k >= 2, "cross-chain matrix needs at least two chains, got {k}");
        assert_eq!(m.len(), k, "cross-chain matrix has {} rows for {k} chains", m.len());
        for (i, row) in m.iter().enumerate() {
            assert_eq!(
                row.len(), k,
                "cross-chain matrix row {i} has {} entries for {k} chains", row.len());
        }
        let mut diag_sum = 0.0;
        let mut off_sum = 0.0;
        let mut asymmetry: f64 = 0.0;
        for (i, row) in m.iter().enumerate() {
            for (j, &v) in row.iter().enumerate() {
                if i == j {
                    diag_sum += v;
                } else {
                    off_sum += v;
                    // The transposed entry: `θ_j` scoring `x_i`, the other
                    // direction of the same pair.
                    asymmetry = asymmetry.max((v - m[j][i]).abs());
                }
            }
        }
        // Non-zero because `k >= 2` is asserted above.
        let n_off = k * k - k;
        let diagonal_dominance = diag_sum / k as f64 - off_sum / n_off as f64;
        Self {
            chains,
            m,
            includes_initial_state: true,
            terms: TERMS.to_string(),
            draw: DRAW.to_string(),
            chain_numbering: CHAIN_NUMBERING.to_string(),
            diagonal_dominance,
            asymmetry,
        }
    }

    /// The two lines printed beside the end-of-stage diagnostics.
    pub fn report(&self) -> String {
        format!(
            "  cross-chain diagonal dominance: {:.1} nats \
             (mean own-path − mean cross-path, {} chains)\n  \
             cross-chain asymmetry: {:.1} nats (max |M[i][j] − M[j][i]|)\n",
            self.diagonal_dominance, self.chains.len(), self.asymmetry,
        )
    }

    /// Write the artifact. Errors are returned rather than swallowed so the
    /// caller decides — a diagnostic that silently fails to be written is a
    /// diagnostic nobody knows to look for.
    pub fn write(&self, path: &std::path::Path) -> Result<(), String> {
        let json = serde_json::to_string_pretty(self)
            .map_err(|e| format!("cannot serialize cross-chain compat matrix: {e}"))?;
        std::fs::write(path, json + "\n")
            .map_err(|e| format!("cannot write {}: {e}", path.display()))
    }
}

#[cfg(test)]
mod tests {
    //! The fixture is the `sir_basic` golden with a Poisson prevalence stream
    //! attached, so the observation term is non-zero and a matrix that
    //! accidentally included it is visible as a number rather than as a
    //! plausible one.
    //!
    //! Each synthetic "chain" is a `(θ, x)` pair built by simulating a
    //! reference path AT that θ — the same producer PGAS conditions on. That
    //! makes the displacement experiment faithful: a chain whose parameters and
    //! path agree is what an augmentation-locked chain looks like.

    use std::sync::Arc;

    use sim::compiled_model::CompiledModel;
    use sim::inference::multi_stream_obs::{
        dense_cells, BoundObs, MultiStreamObsModel, StreamProjection, StreamSpec,
    };
    use sim::inference::particle_filter::Observation;
    use sim::inference::pgas::{build_obs_at_substep, simulate_reference};
    use sim::rng::StatefulRng;

    use super::*;

    const DT: f64 = 1.0;
    const T_END: f64 = 60.0;
    /// Index of `I` in `compartments { S, I, R }`.
    const I_IDX: usize = 1;

    /// A Poisson stream on prevalence of `I`, so the fixture carries a real
    /// observation term. `sir_basic` declares no `observations {}` block.
    fn poisson_prevalence_block() -> ir::observation::ObservationModel {
        use ir::expr::*;
        use ir::observation::*;
        let rate = Expr::Projected(ProjectedExpr { projected: () });
        ObservationModel {
            name: "prevalence".into(),
            source: "prevalence".into(),
            columns: vec![
                ObsColumn { name: "time".into(), role: ColumnRole::Time },
                ObsColumn {
                    name: "prevalence".into(),
                    role: ColumnRole::Value(ir::parameter::ParamKind::Count),
                },
            ],
            scored: "prevalence".into(),
            emit_schedule: Some(ObservationSchedule::AtTimes(vec![])),
            stratum: vec![],
            projection: Projection::CurrentPop("I".into()),
            projection_state_grad: Default::default(),
            likelihood: Likelihood::Poisson(PoissonLikelihood { rate: ir::Diffable::new(rate) }),
        }
    }

    fn compiled() -> Arc<CompiledModel> {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../../ocaml/golden/sir_basic.ir.json");
        let json = std::fs::read_to_string(&path)
            .unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
        let mut m = ir::from_str(&json).expect("parse sir_basic golden");
        m.observations = vec![poisson_prevalence_block()];
        m.simulation.t_end = T_END;
        // `I ~ poisson(rate = I0)` in place of `sir_basic`'s deterministic
        // `I = I0`. Without a declared law `initial_state_logpdf` returns 0 for
        // every (θ, x), and dropping the initial-state term from `M` would be
        // invisible — the exact case gh#785 says is load-bearing, since a
        // parameter-dependent initial condition is a large part of what couples
        // θ to the path.
        let i0 = ir::expr::Expr::param("I0");
        m.initial_conditions.0.insert(
            "I".into(),
            ir::model::InitSpec::Count(ir::model::InitCountLaw::Poisson(
                ir::observation::PoissonLikelihood { rate: ir::Diffable::new(i0) },
            )),
        );
        for p in &mut m.parameters {
            if p.value.resolved_value().is_none() {
                let v = match p.name.as_str() {
                    "beta" => 0.3,
                    "gamma" => 0.1,
                    "N0" => 1000.0,
                    "I0" => 10.0,
                    _ => 0.5,
                };
                p.value = p.value.with_value(v);
            }
        }
        Arc::new(CompiledModel::new(m).expect("compile sir_basic"))
    }

    /// The full model parameter vector with `beta` set to `b`.
    fn params_at(model: &CompiledModel, b: f64) -> Vec<f64> {
        let mut p = model.default_params.clone();
        p[model.param_index["beta"]] = b;
        p
    }

    /// As [`params_at`], also moving `I0` — the rate of the declared
    /// initial-state law, so the `log p(x₀ | θ)` term varies by row as well as
    /// by column.
    fn params_at_seeded(model: &CompiledModel, b: f64, i0: f64) -> Vec<f64> {
        let mut p = params_at(model, b);
        p[model.param_index["I0"]] = i0;
        p
    }

    /// A reference path simulated at `params` — the `(θ, x)` pairing a
    /// parameter-locked chain would hold.
    fn path_at(model: &CompiledModel, params: &[f64], seed: u64) -> sim::inference::pgas::PGASTrajectory {
        let mut rng = StatefulRng::new(seed);
        simulate_reference(model, params, T_END, DT, sim::rng::BinomialAlgorithm::default(),
                           &mut rng).expect("reference path")
    }

    /// Weekly prevalence observations read off a baseline path, plus the
    /// matching observation model and substep map.
    fn observed(model: &Arc<CompiledModel>) -> (Vec<Observation>, MultiStreamObsModel, ObsAtSubstep) {
        let base = path_at(model, &params_at(model, 0.3), 11);
        let mut obs = Vec::new();
        for (s, rec) in base.substeps.iter().enumerate() {
            let t = ((s + 1) as f64) * DT;
            if (t.round() as i64) % 7 == 0 {
                obs.push(Observation { time: t, value: rec.counts_after[I_IDX] as f64 });
            }
        }
        assert!(obs.len() >= 4, "fixture must carry several observations");
        let obs_model = MultiStreamObsModel::new(
            BoundObs::bind(vec![StreamSpec::dense(
                StreamProjection::IntCompSum(vec![I_IDX]),
                model.model.observations[0].clone(),
                dense_cells(obs.iter().map(|o| o.value).collect()),
                obs.iter().map(|o| o.time).collect(),
            )])
            .expect("bind observations").0,
            model.clone(),
        )
        .expect("observation model");
        let map = build_obs_at_substep(&obs, model.model.simulation.t_start, DT)
            .expect("obs→substep map");
        (obs, obs_model, map)
    }

    /// Build the matrix for a set of `(chain id, beta, path seed)` chains, each
    /// chain's path simulated at its own beta.
    fn matrix_for(chains: &[(usize, f64, u64)]) -> (Arc<CompiledModel>, CrossChainCompat) {
        let model = compiled();
        let (obs, obs_model, map) = observed(&model);
        let params: Vec<Vec<f64>> = chains.iter().map(|&(_, b, _)| params_at(&model, b)).collect();
        let paths: Vec<_> = chains.iter().zip(&params)
            .map(|(&(_, _, seed), p)| path_at(&model, p, seed))
            .collect();
        let draws: Vec<ChainDraw<'_>> = chains.iter().zip(&params).zip(&paths)
            .map(|((&(id, _, _), p), x)| ChainDraw { chain: id, params: p, trajectory: x })
            .collect();
        let compat = CrossChainCompat::compute(&model, &draws, &obs, DT, &obs_model, &map)
            .expect("compute")
            .expect("two or more chains");
        (model, compat)
    }

    // ── The derived numbers ──────────────────────────────────────────────

    /// `diagonal_dominance` and `asymmetry` on a matrix small enough to check
    /// by hand. The 2×2 is deliberately asymmetric so a transposed or
    /// symmetrised implementation cannot pass.
    #[test]
    fn derived_numbers_are_the_stated_formulas() {
        //      θ₁      θ₂
        // x₁  -10.0   -50.0     (row = the parameters doing the scoring)
        // x₂   -3.0   -20.0
        let c = CrossChainCompat::from_matrix(vec![1, 2], vec![
            vec![-10.0, -50.0],
            vec![-3.0, -20.0],
        ]);
        // mean(diag) = -15, mean(offdiag) = (-50 + -3)/2 = -26.5
        assert!((c.diagonal_dominance - 11.5).abs() < 1e-12,
            "diagonal_dominance = mean(diag) − mean(offdiag); got {}", c.diagonal_dominance);
        assert!((c.asymmetry - 47.0).abs() < 1e-12,
            "asymmetry = max |M[i][j] − M[j][i]| = |−50 − (−3)|; got {}", c.asymmetry);
        assert_eq!(c.chains, vec![1, 2]);
        assert!(c.includes_initial_state);
        assert_eq!(c.terms, TERMS);
    }

    /// A single chain has no cross term to measure, so no artifact is produced
    /// — `chains` must never describe rows `M` does not have.
    #[test]
    fn one_chain_produces_no_matrix() {
        let model = compiled();
        let (obs, obs_model, map) = observed(&model);
        let p = params_at(&model, 0.3);
        let x = path_at(&model, &p, 7);
        let draws = vec![ChainDraw { chain: 1, params: &p, trajectory: &x }];
        assert!(
            CrossChainCompat::compute(&model, &draws, &obs, DT, &obs_model, &map)
                .expect("compute").is_none(),
            "a one-chain fit has no off-diagonal, so there is nothing to report"
        );
    }

    // ── What each entry is ───────────────────────────────────────────────

    /// Every entry is the PATH density of the COLUMN's path under the ROW's
    /// parameters: `transition + initial_state`, and no observation term.
    ///
    /// Three separable failures, none of which a property test can see:
    ///
    /// * indexing the other way round transposes `M`, which leaves the diagonal
    ///   and BOTH derived numbers untouched;
    /// * including `p(y | x)` shifts every entry by a column-wise constant,
    ///   which cancels out of `diagonal_dominance` almost exactly;
    /// * dropping `log p(x₀ | θ)` is invisible in any model with a
    ///   deterministic `init {}`, which is why the fixture declares a law.
    ///
    /// So each entry is checked against a direct call, and each of the two
    /// terms is separately shown to be present or absent.
    #[test]
    fn each_entry_is_the_column_path_under_the_row_parameters() {
        let model = compiled();
        let (obs, obs_model, map) = observed(&model);
        // Deliberately unequal betas AND seeds: the two cross terms then differ
        // from each other, so a transposed implementation gives different
        // numbers, and `log p(x₀ | θ)` differs by row as well as by column.
        let params: Vec<Vec<f64>> = [(0.25, 10.0), (0.55, 40.0)].iter()
            .map(|&(b, i0)| params_at_seeded(&model, b, i0))
            .collect();
        let paths: Vec<_> = params.iter().enumerate()
            .map(|(i, p)| path_at(&model, p, 100 + i as u64))
            .collect();
        let draws: Vec<ChainDraw<'_>> = (0..2)
            .map(|i| ChainDraw { chain: i + 1, params: &params[i], trajectory: &paths[i] })
            .collect();
        let c = CrossChainCompat::compute(&model, &draws, &obs, DT, &obs_model, &map)
            .expect("compute").expect("two chains");

        let mut any_obs = false;
        let mut any_init = false;
        for i in 0..2 {
            for j in 0..2 {
                let oracle = complete_data_loglik(
                    &model, &paths[j], &params[i], &obs, DT, &obs_model, &map,
                ).expect("oracle");
                assert_eq!(
                    c.m[i][j], oracle.transition + oracle.initial_state,
                    "M[{i}][{j}] must be log p(x_{j} | θ_{i}) = transition + initial_state \
                     (row indexes the PARAMETERS)"
                );
                if oracle.observation.abs() > 1.0 {
                    any_obs = true;
                    assert_ne!(
                        c.m[i][j], oracle.total,
                        "M[{i}][{j}] must exclude the observation term p(y | x) = {}",
                        oracle.observation
                    );
                }
                if oracle.initial_state.abs() > 1.0 {
                    any_init = true;
                    assert_ne!(
                        c.m[i][j], oracle.transition,
                        "M[{i}][{j}] must include the initial-state term \
                         log p(x₀ | θ) = {} — `includes_initial_state` claims it does",
                        oracle.initial_state
                    );
                }
            }
        }
        assert!(any_obs,
            "the fixture must produce a non-zero observation term for its exclusion to bite");
        assert!(any_init,
            "the fixture must produce a non-zero initial-state term for its inclusion to bite");
        assert_ne!(c.m[0][1], c.m[1][0],
            "the fixture's two cross terms must differ, or a transpose is undetectable");
        assert!(c.includes_initial_state);
    }

    // ── The property the diagnostic exists for ───────────────────────────

    /// Near-identical chains are exchangeable in the augmented space: their
    /// parameters score each other's paths about as well as their own, so
    /// diagonal dominance is small. Displace one chain's `(θ, x)` pair — the
    /// picture of a chain locked onto its own augmentation — and dominance
    /// rises by orders of magnitude.
    ///
    /// This is the whole claim of gh#785: `M` separates augmentation locking
    /// from the chains merely differing.
    #[test]
    fn a_displaced_chain_raises_diagonal_dominance() {
        // Three chains agreeing to within 1% on beta, each with its own path.
        let (_, near) = matrix_for(&[(1, 0.300, 21), (2, 0.303, 22), (3, 0.297, 23)]);
        // The same first two chains; the third displaced to a beta four times
        // larger, with the path that beta actually produces.
        let (_, split) = matrix_for(&[(1, 0.300, 21), (2, 0.303, 22), (3, 1.200, 23)]);

        eprintln!("near-identical: dominance {:.1} nats, asymmetry {:.1} nats",
            near.diagonal_dominance, near.asymmetry);
        eprintln!("one displaced:  dominance {:.1} nats, asymmetry {:.1} nats",
            split.diagonal_dominance, split.asymmetry);

        for c in [&near, &split] {
            assert_eq!(c.chains, vec![1, 2, 3], "chains must describe M's actual rows");
            assert_eq!(c.m.len(), 3);
            for row in &c.m {
                assert_eq!(row.len(), 3, "M must be square");
                assert!(row.iter().all(|v| v.is_finite()),
                    "every entry must be a finite log-density: {row:?}");
            }
            assert!(c.diagonal_dominance.is_finite());
        }

        // Chains that agree on θ score each other's paths about as well as
        // their own. The residual is the path-to-path spread at a common θ,
        // which is small next to the displaced case below.
        assert!(
            near.diagonal_dominance.abs() < 20.0,
            "chains agreeing on beta to 1% must be near-exchangeable in the augmented \
             space; dominance was {:.1} nats",
            near.diagonal_dominance
        );
        assert!(
            split.diagonal_dominance > 20.0 * near.diagonal_dominance.abs().max(1.0),
            "displacing one chain must raise diagonal dominance well clear of the \
             near-identical case; got {:.1} vs {:.1} nats",
            split.diagonal_dominance, near.diagonal_dominance
        );

        // The displaced chain is pinned to its own path in BOTH directions:
        // its parameters score the other paths badly, and its path scores
        // badly under the others' parameters.
        let d = 2usize;
        for j in 0..3 {
            if j == d { continue }
            assert!(split.m[d][d] > split.m[d][j] + 50.0,
                "the displaced chain's own path must be far more probable under its own \
                 parameters ({:.1}) than chain {}'s path ({:.1})",
                split.m[d][d], j + 1, split.m[d][j]);
            assert!(split.m[j][j] > split.m[j][d] + 50.0,
                "chain {}'s parameters must score the displaced path far below their own \
                 ({:.1} vs {:.1})", j + 1, split.m[j][j], split.m[j][d]);
        }
    }

    // ── The artifact ─────────────────────────────────────────────────────

    /// The on-disk shape a downstream reader (camdl-scope) parses: the keys,
    /// the 1-based chain numbering, and `M` indexable by `chains`' positions.
    #[test]
    fn the_artifact_states_its_conventions() {
        let (_, c) = matrix_for(&[(1, 0.30, 31), (2, 0.60, 32)]);
        let v: serde_json::Value =
            serde_json::from_str(&serde_json::to_string(&c).unwrap()).unwrap();
        assert_eq!(v["chains"], serde_json::json!([1, 2]),
            "chain ids are 1-based, matching the chain_N/ directories");
        assert_eq!(v["includes_initial_state"], serde_json::json!(true));
        assert_eq!(v["terms"], serde_json::json!(TERMS));
        assert_eq!(v["draw"], serde_json::json!(DRAW));
        assert_eq!(v["chain_numbering"], serde_json::json!(CHAIN_NUMBERING));
        let m = v["M"].as_array().expect("M is an array of rows");
        assert_eq!(m.len(), 2);
        assert!(m.iter().all(|r| r.as_array().unwrap().len() == 2));
        assert!(v["diagonal_dominance"].is_number());
        assert!(v["asymmetry"].is_number());
    }
}
