//! Deterministic ODE gradient — `det_grad` (gh#275 Phase 1).
//!
//! Returns `(log p(y | θ), ∇_θ log p(y | θ))` for the deterministic ODE
//! likelihood, the object gradient-based Bayesian sampling (`nuts` on `ode`,
//! Phase 2) and any future gradient-MLE consume. The prior term and the
//! unconstrained-space change-of-variables term are added by the sampler target
//! (reused verbatim from PGAS); `det_grad` is the data-likelihood factor only.
//!
//! Structure (proposal `2026-06-26-ode-nuts-gradient-spine.md`): integrate the
//! augmented `(x, flow, S, acc_sens)` system once
//! ([`crate::ode::integrate_obs_sensitivity`]), then score value + gradient from
//! the per-obs sensitivity records
//! ([`MultiStreamObsModel::ode_loglik_and_grad`]). The gradient is the sum of two
//! orthogonal factors: the θ-direct term (PGAS's obs-grad seam, `projected` fixed)
//! and the trajectory chain `(∂logp/∂projected)·(∂projected/∂θ)` fed by the
//! forward sensitivities. No finite differences (those live only in the §1f
//! gradient-check oracle); no runtime autodiff.

use crate::compiled_model::CompiledModel;
use crate::config::OdeConfig;
use crate::error::SimError;
use crate::inference::MultiStreamObsModel;

/// Evaluate `(log p(y | θ), ∇_θ log p(y | θ))` for the deterministic ODE
/// likelihood at `params`, differentiating with respect to the estimated
/// parameters `estimated_to_model` (each entry a model parameter index).
///
/// `obs_times` must lie on the model's output grid (the recorder scores on the
/// output grid, matching `compute_ode_loglik`). `dt` is the fixed RK4 step.
///
/// Refuses (hard error) the cases v1 cannot differentiate soundly, each with a
/// reason: an adaptive integrator, a `dt`-in-rate model, a scheduled effect
/// ([`crate::ode::integrate_obs_sensitivity`]); a parameterized initial condition
/// (needs the `ic_grad` seed, below); a `DerivedExpr` prevalence projection or a
/// `projected`-transforming likelihood argument
/// ([`MultiStreamObsModel::ode_loglik_and_grad`]).
pub fn det_grad(
    compiled: &CompiledModel,
    obs_model: &MultiStreamObsModel,
    obs_times: &[f64],
    dt: f64,
    params: &[f64],
    estimated_to_model: &[usize],
    // gh#396: when `Some`, replace the `[origin, t_end]` transient with the
    // periodic equilibrium solved at `T_eq` — the equilibrium state `X*(θ)` and its
    // exact sensitivity `∂X*/∂θ` seed the `[T_eq, t_end]` integration, so the long
    // spin-up is never integrated. `None` → today's path (from the model `t_start`).
    warm_start: Option<&crate::ode_equilibrium::WarmStart>,
) -> Result<(f64, Vec<f64>), SimError> {
    // §1h capability gate: the ONE place that answers "can this model be fit by the
    // ODE gradient?" — refusing (with the compiler's own reason where one exists)
    // an unsupported rate/obs/σ² gradient, a nonsmooth ∂rate/∂state, an adaptive
    // integrator, a `dt`-in-rate model, a scheduled effect, a nonsmooth or
    // real-compartment initial condition, a DerivedExpr projection, and a
    // `projected`-transforming likelihood argument. Run before any gradient is
    // taken, so a refusal is a single actionable message, not a mid-integration
    // failure.
    let estimated: std::collections::HashSet<&str> = estimated_to_model
        .iter()
        .map(|&i| compiled.model.parameters[i].name.as_str())
        .collect();
    crate::inference::gradient_capability::preflight_gradient_ode(compiled, params, &estimated)?;

    // Forward-sensitivity seed S(t_start) = ∂(initial_state)/∂θ (`ic_grad`): zero
    // for an explicit (constant) initial condition, nonzero for a parameterized
    // one whose expression involves an estimated parameter (gh#275 §1c C-seed).
    let ic_grad = compiled.ic_grad_seed(params, estimated_to_model)?;

    // gh#396: warm-start replaces the transient with the equilibrium at T_eq;
    // otherwise integrate from the model's t_start with the ic_grad seed.
    let (x0, state_sens_0, t_start) = match warm_start {
        Some(ws) => {
            let eq = crate::ode_equilibrium::solve_equilibrium(
                compiled,
                params,
                estimated_to_model,
                &ic_grad,
                ws.t_eq,
                ws.period,
                dt,
            )?;
            log::debug!(
                "ode warm-start: equilibrium in {} Newton iters, {} conserved direction(s)",
                eq.iters,
                eq.n_conserved
            );
            (Some(eq.x_star), eq.x_star_sens, ws.t_eq)
        }
        None => (None, ic_grad, compiled.model.simulation.t_start),
    };

    let cfg = OdeConfig {
        t_start,
        t_end: compiled.model.simulation.t_end,
        dt,
    };

    let records = crate::ode::integrate_obs_sensitivity(
        compiled,
        params,
        estimated_to_model,
        x0.as_deref(),
        &state_sens_0,
        &cfg,
        obs_times,
    )?;

    obs_model.ode_loglik_and_grad(&records, params, estimated_to_model)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::inference::multi_stream_obs::{StreamProjection, StreamSpec};
    use crate::inference::{dense_cells, BoundObs, MultiStreamObsModel};
    use ir::deriv::DerivEntry;
    use ir::expr::{BinOp, ConstExpr, Expr, ParamExpr, ProjectedExpr};
    use ir::observation::{Likelihood, PoissonLikelihood};
    use ir::Diffable;
    use std::collections::HashMap;
    use std::sync::Arc;

    fn projected() -> Expr {
        Expr::Projected(ProjectedExpr { projected: () })
    }

    /// `seir_observations` made ODE-gradient-testable, using the compiler-EMITTED
    /// gradients throughout — the full pipeline, not hand-installed derivatives:
    ///
    /// - `rate_state_grad` (J_x) is emitted: `infection = beta·S·I/N` with
    ///   `N = S+E+I+R` (a hoisted `PopSum` binding), so `∂rate/∂{S,E,I,R}` carries
    ///   the full product/quotient rule through the binding.
    /// - `weekly_cases` is the NATIVE `negbin(mean = rho·incidence, dispersion = k)`
    ///   — a reporting-rate model. Its emitted `mean_grad[rho] = incidence` (factor 1
    ///   for the obs param `rho`) and its emitted `mean_proj = rho` (`∂mean/∂projected`,
    ///   the WrtProjected factor-2 coefficient) are BOTH exercised. `k` is a second
    ///   factor-1 obs param.
    /// - `detection` is rewritten to `poisson(rate = I)` — a prevalence stream whose
    ///   argument IS `projected`, exercising the `IntCompSum` factor-2 chain
    ///   (`∂g/∂x·S`). Its `proj_grad` is `∂(projected)/∂projected = 1`, matching what
    ///   the compiler emits for a bare `Projected` argument.
    ///
    /// The initial condition is made explicit (`S(t_start)=∂init/∂θ=0`, det_grad v1).
    fn seir_ode_grad_fixture() -> ir::Model {
        let manifest = std::env::var("CARGO_MANIFEST_DIR").unwrap();
        let path = std::path::PathBuf::from(&manifest)
            .join("../../../ocaml/golden/seir_observations.ir.json");
        let mut model: ir::Model = ir::from_str(
            &std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("read: {e}")),
        )
        .unwrap_or_else(|e| panic!("parse: {e}"));

        // The emitted J_x and the native weekly_cases mean-projection derivative must
        // be present — this test validates them end-to-end.
        assert!(
            model.transitions.iter().any(|t| !t.rate_state_grad.0.is_empty()),
            "seir_observations must carry emitted rate_state_grad (run make update-golden)"
        );
        let weekly = model.observations.iter().find(|o| o.name == "weekly_cases").unwrap();
        if let Likelihood::NegBinomial(nb) = &weekly.likelihood {
            assert!(
                nb.mean.proj_grad.is_some(),
                "native weekly_cases mean (rho·projected) must carry an emitted proj_grad \
                 (run make update-golden)"
            );
        } else {
            panic!("weekly_cases must be the native NegBinomial");
        }

        // Explicit (constant) initial condition: ∂init/∂θ = 0. The golden now
        // carries emitted `ic_grad` for its own parameterized IC, so clear it to
        // stay consistent with this forced Explicit IC (the estimated-IC oracle
        // below sets its own IC + ic_grad).
        model.initial_conditions = ir::model::InitialConditions::Explicit(HashMap::from([
            ("S".to_string(), 9990.0),
            ("E".to_string(), 0.0),
            ("I".to_string(), 10.0),
            ("R".to_string(), 0.0),
        ]));
        model.ic_grad = HashMap::new();

        // Keep the native weekly_cases (rho·incidence). Rewrite detection to a
        // prevalence poisson whose argument IS `projected` (proj_grad = 1).
        for om in &mut model.observations {
            if om.name == "detection" {
                om.likelihood = Likelihood::Poisson(PoissonLikelihood {
                    rate: Diffable {
                        expr: projected(),
                        grad: HashMap::new(),
                        proj_grad: Some(DerivEntry::Grad(Expr::Const(ConstExpr { value: 1.0 }))),
                    },
                });
            }
        }
        model
    }

    fn set_defaults(model: &mut ir::Model) {
        let defaults = [
            ("beta", 0.6), ("sigma", 0.2), ("gamma", 0.1), ("k", 8.0),
            ("rho", 0.5), ("p_detect", 0.8), ("N0", 10000.0), ("I0", 10.0),
        ];
        for p in &mut model.parameters {
            if p.value.resolved_value().is_none() {
                let v = defaults.iter().find(|(n, _)| *n == p.name).map(|(_, v)| *v).unwrap_or(0.5);
                p.value = p.value.with_value(v);
            }
        }
    }

    fn build_obs_model(
        compiled: Arc<CompiledModel>,
        obs_times: &[f64],
        per_stream: Vec<Vec<f64>>,
    ) -> MultiStreamObsModel {
        let model = compiled.model.clone();
        let specs: Vec<StreamSpec> = model
            .observations
            .iter()
            .enumerate()
            .map(|(si, om)| {
                let projection = StreamProjection::from_ir(&om.projection, &compiled, &om.name).unwrap();
                StreamSpec {
                    projection,
                    ir_model: om.clone(),
                    observations: dense_cells(per_stream[si].clone()),
                    obs_times: obs_times.to_vec(),
                    aux: vec![],
                }
            })
            .collect();
        MultiStreamObsModel::new(BoundObs::bind(specs).unwrap().0, compiled).unwrap()
    }

    /// The det_grad FD oracle (gh#275 §1f): `∇_symbolic` vs a central finite
    /// difference of the loglik, over an incidence stream and a prevalence stream,
    /// exercising factor-1 (`k`) and factor-2 (`beta`, `gamma`, through both an
    /// `inc_sens` incidence chain and a `state_sens` prevalence chain).
    #[test]
    fn det_grad_matches_finite_difference_seir_incidence_and_prevalence() {
        let mut model = seir_ode_grad_fixture();
        set_defaults(&mut model);
        model.simulation.t_end = 60.0;
        let compiled = Arc::new(CompiledModel::new(model).unwrap());

        let n = compiled.param_index.len();
        let mut params = vec![0.0; n];
        for p in &compiled.model.parameters {
            params[compiled.param_index[p.name.as_str()]] = p.value.resolved_value().unwrap();
        }

        let dt = 1.0;
        let obs_times: Vec<f64> = (1..=8).map(|w| (w * 7) as f64).collect(); // 7..56
        let est = vec![
            compiled.param_index["beta"],
            compiled.param_index["gamma"],
            compiled.param_index["k"],
            compiled.param_index["rho"],
        ];

        // Synthesize obs data = the projected value at the true params (near the
        // mode → well-conditioned FD). Uses the recorder + each stream's resolved
        // projection, exactly as det_grad reads them.
        let cfg = OdeConfig { t_start: compiled.model.simulation.t_start, t_end: 60.0, dt };
        let (int_s0, _) = compiled.initial_state(&params).unwrap();
        let seed = vec![0.0; int_s0.counts.len() * est.len()];
        let recs =
            crate::ode::integrate_obs_sensitivity(&compiled, &params, &est, None, &seed, &cfg, &obs_times)
                .unwrap();
        let projections: Vec<StreamProjection> = compiled
            .model
            .observations
            .iter()
            .map(|om| StreamProjection::from_ir(&om.projection, &compiled, &om.name).unwrap())
            .collect();
        let per_stream: Vec<Vec<f64>> = projections
            .iter()
            .map(|proj| {
                recs.iter()
                    .map(|r| {
                        let v: f64 = match proj {
                            StreamProjection::FlowSum(idxs) => idxs.iter().map(|&i| r.inc[i]).sum(),
                            StreamProjection::IntCompSum(idxs) => {
                                idxs.iter().map(|&i| r.counts[i]).sum()
                            }
                            StreamProjection::Expr(_) => panic!("fixture has no Expr projection"),
                        };
                        v.round()
                    })
                    .collect()
            })
            .collect();
        // Sanity: incidence and prevalence data are materially nonzero.
        assert!(per_stream[0].iter().sum::<f64>() > 1.0, "weekly incidence data must be nonzero");
        assert!(per_stream[1].iter().sum::<f64>() > 1.0, "detection prevalence data must be nonzero");

        let obs_model = build_obs_model(compiled.clone(), &obs_times, per_stream);

        let (ll, grad) = det_grad(&compiled, &obs_model, &obs_times, dt, &params, &est, None).unwrap();
        assert!(ll.is_finite(), "det_grad loglik must be finite, got {ll}");
        assert_eq!(grad.len(), est.len());

        let eps = 1e-6;
        let names = ["beta", "gamma", "k", "rho"];
        for (i, &midx) in est.iter().enumerate() {
            let mut pp = params.clone();
            let mut pm = params.clone();
            pp[midx] += eps;
            pm[midx] -= eps;
            let (llp, _) = det_grad(&compiled, &obs_model, &obs_times, dt, &pp, &est, None).unwrap();
            let (llm, _) = det_grad(&compiled, &obs_model, &obs_times, dt, &pm, &est, None).unwrap();
            let fd = (llp - llm) / (2.0 * eps);
            let rel = if fd.abs() > 1e-8 {
                (grad[i] - fd).abs() / fd.abs()
            } else {
                (grad[i] - fd).abs()
            };
            assert!(
                rel < 1e-4,
                "det_grad ∂/∂{} = {} vs FD {} (rel err {:.2e})",
                names[i],
                grad[i],
                fd,
                rel
            );
        }
        // Non-vacuity: beta (factor-2 through both chains) and k (factor-1) must
        // carry materially nonzero gradients, or the test proves little.
        assert!(grad[0].abs() > 1e-6, "∂/∂beta should be materially nonzero");
        assert!(grad[2].abs() > 1e-6, "∂/∂k should be materially nonzero");
    }

    /// The det_grad FD oracle for a **DerivedExpr (nonlinear) projection**
    /// (gh#275 §1h). The detection stream projects `√I + √R` — a nonlinear function
    /// of TWO compartments — so its observation factor-2 term is
    /// `∂projected/∂θ = (0.5/√I)·S[I] + (0.5/√R)·S[R]`: the compiler-shaped
    /// projection gradient `∂proj/∂x_j`, evaluated at the trajectory, weighting the
    /// forward sensitivity `S` and SUMMED over the compartments. This is the whole
    /// point — a linear `IntCompSum` uses weight `1`; a DerivedExpr uses the
    /// state-dependent `∂proj/∂x`, and the multi-compartment case exercises the
    /// `Σ_j` accumulation (catching an entry-ordering / stride bug). `∂(√x)/∂x =
    /// 0.5/√x` is unambiguous, so the hand-installed `{I: 0.5/√I, R: 0.5/√R}` is
    /// verified end-to-end here. (`√`, not `I²`: it keeps the projected value
    /// O(100) so the central FD does not lose precision to cancellation.)
    ///
    /// RED-CHECK (verified during development): zeroing the projection gradient
    /// drops the stream's entire factor-2, and analytic `∂/∂beta` lands at rel err
    /// ≈ 1e-2 vs the central FD — well above the 1e-4 gate. (Not larger because the
    /// co-located incidence stream still contributes correctly, so beta's total is
    /// only partly wrong; the `√I+√R` term is what closes the gap.)
    #[test]
    fn det_grad_matches_finite_difference_derivedexpr_projection() {
        use ir::expr::UnOp;
        let mut model = seir_ode_grad_fixture();
        set_defaults(&mut model);
        model.simulation.t_end = 60.0;
        // Rewrite `detection` to a MULTI-compartment DerivedExpr projection √I + √R,
        // poisson(rate=projected). Two state-dependent terms exercise the factor-2
        // `Σ_j (∂proj/∂x_j)·S[j]` accumulation over more than one compartment —
        // catching an entry-ordering or stride bug the single-term case can't.
        let sqrt = |c: &str| Expr::un_op(UnOp::Sqrt, Expr::pop(c));
        let half_over_sqrt = |c: &str| DerivEntry::Grad(Expr::bin_op(
            BinOp::Div, Expr::Const(ConstExpr { value: 0.5 }), sqrt(c)));
        for om in &mut model.observations {
            if om.name == "detection" {
                om.projection = ir::observation::Projection::DerivedExpr(
                    Expr::bin_op(BinOp::Add, sqrt("I"), sqrt("R")));
                // ∂(√I + √R)/∂I = 0.5/√I, ∂/∂R = 0.5/√R
                om.projection_state_grad = ir::deriv::CompGradMap(HashMap::from([
                    ("I".to_string(), half_over_sqrt("I")),
                    ("R".to_string(), half_over_sqrt("R")),
                ]));
                om.likelihood = Likelihood::Poisson(PoissonLikelihood {
                    rate: Diffable {
                        expr: projected(),
                        grad: HashMap::new(),
                        proj_grad: Some(DerivEntry::Grad(Expr::Const(ConstExpr { value: 1.0 }))),
                    },
                });
            }
        }
        let compiled = Arc::new(CompiledModel::new(model).unwrap());
        let n = compiled.param_index.len();
        let mut params = vec![0.0; n];
        for p in &compiled.model.parameters {
            params[compiled.param_index[p.name.as_str()]] = p.value.resolved_value().unwrap();
        }
        let i_idx = compiled.global_to_int[compiled.comp_index["I"]].unwrap();
        let r_idx = compiled.global_to_int[compiled.comp_index["R"]].unwrap();

        let dt = 1.0;
        let obs_times: Vec<f64> = (1..=8).map(|w| (w * 7) as f64).collect();
        let est = vec![compiled.param_index["beta"], compiled.param_index["gamma"]];

        // Synthesize obs data at the true params: weekly = incidence, detection = √I+√R.
        let cfg = OdeConfig { t_start: compiled.model.simulation.t_start, t_end: 60.0, dt };
        let seed = vec![0.0; compiled.initial_state(&params).unwrap().0.counts.len() * est.len()];
        let recs =
            crate::ode::integrate_obs_sensitivity(&compiled, &params, &est, None, &seed, &cfg, &obs_times)
                .unwrap();
        let flow_idx = 0usize; // weekly_cases is the incidence stream (FlowSum)
        let weekly: Vec<f64> = recs.iter().map(|r| r.inc[flow_idx].round()).collect();
        let detection: Vec<f64> = recs.iter()
            .map(|r| (r.counts[i_idx].max(0.0).sqrt() + r.counts[r_idx].max(0.0).sqrt()).round())
            .collect();
        assert!(detection.iter().sum::<f64>() > 1.0, "√I+√R detection data must be nonzero");

        let obs_model = build_obs_model(compiled.clone(), &obs_times, vec![weekly, detection]);
        let (ll, grad) = det_grad(&compiled, &obs_model, &obs_times, dt, &params, &est, None).unwrap();
        assert!(ll.is_finite(), "det_grad loglik must be finite, got {ll}");

        let eps = 1e-6;
        let names = ["beta", "gamma"];
        for (i, &midx) in est.iter().enumerate() {
            let mut pp = params.clone();
            let mut pm = params.clone();
            pp[midx] += eps;
            pm[midx] -= eps;
            let (llp, _) = det_grad(&compiled, &obs_model, &obs_times, dt, &pp, &est, None).unwrap();
            let (llm, _) = det_grad(&compiled, &obs_model, &obs_times, dt, &pm, &est, None).unwrap();
            let fd = (llp - llm) / (2.0 * eps);
            let rel = if fd.abs() > 1e-8 {
                (grad[i] - fd).abs() / fd.abs()
            } else {
                (grad[i] - fd).abs()
            };
            assert!(
                rel < 1e-4,
                "det_grad ∂/∂{} = {} vs FD {} (rel err {:.2e})",
                names[i], grad[i], fd, rel
            );
        }
        // Non-vacuity: beta drives I, so the I² factor-2 must be materially nonzero.
        assert!(grad[0].abs() > 1e-6, "∂/∂beta should be materially nonzero (the projection factor-2)");
    }

    /// The det_grad FD oracle for an **estimated initial-condition parameter**
    /// (gh#275 §1c C-seed, Risk #1). A parameterized IC `S = N0 − I0`, `I = I0`
    /// makes the initial epidemic size a function of the estimated `I0`; its
    /// forward sensitivity must be seeded at `S(t_start) = ∂init/∂I0`
    /// (`ic_grad[S][I0] = −1`, `ic_grad[I][I0] = +1`) and propagate through `J_x`
    /// into every downstream observation. Without the seed the whole `∂/∂I0` chain
    /// is identically zero, silently collapsing the initial-size marginal.
    ///
    /// RED-CHECK (verified during development): forcing `ic_grad_seed` to return
    /// zeros makes analytic `∂/∂I0 = 0` while the central FD is ≈ +0.39 (the value
    /// path is smooth in `I0` because the ODE gradient path uses the *continuous*
    /// initial state), so the assertion fails with 100% relative error — the seed
    /// is exactly what closes the gap.
    #[test]
    fn det_grad_matches_finite_difference_estimated_initial_condition() {
        let mut model = seir_ode_grad_fixture();
        set_defaults(&mut model);
        model.simulation.t_end = 60.0;

        // Parameterized IC: I0 sets the initial infected count and is drawn out of
        // S so the total population is conserved. N0 stays fixed.
        let param = |n: &str| Expr::Param(ParamExpr { param: n.to_string() });
        model.initial_conditions = ir::model::InitialConditions::Parameterized(HashMap::from([
            ("S".to_string(), Expr::bin_op(BinOp::Sub, param("N0"), param("I0"))),
            ("E".to_string(), Expr::Const(ConstExpr { value: 0.0 })),
            ("I".to_string(), param("I0")),
            ("R".to_string(), Expr::Const(ConstExpr { value: 0.0 })),
        ]));
        // Emitted ∂init/∂I0: −1 for S (= ∂(N0−I0)/∂I0), +1 for I. N0 is fixed → no
        // column. (What the OCaml WrtParam-over-init pass will emit.)
        let grad1 = |v: f64| DerivEntry::Grad(Expr::Const(ConstExpr { value: v }));
        model.ic_grad = HashMap::from([
            ("S".to_string(), HashMap::from([("I0".to_string(), grad1(-1.0))])),
            ("I".to_string(), HashMap::from([("I0".to_string(), grad1(1.0))])),
        ]);

        let compiled = Arc::new(CompiledModel::new(model).unwrap());
        let n = compiled.param_index.len();
        let mut params = vec![0.0; n];
        for p in &compiled.model.parameters {
            params[compiled.param_index[p.name.as_str()]] = p.value.resolved_value().unwrap();
        }

        let dt = 1.0;
        let obs_times: Vec<f64> = (1..=8).map(|w| (w * 7) as f64).collect();
        // Estimate a rate param and the IC param together.
        let est = vec![compiled.param_index["beta"], compiled.param_index["I0"]];

        // Synthesize obs data at the true params (value path uses continuous init).
        let cfg = OdeConfig { t_start: compiled.model.simulation.t_start, t_end: 60.0, dt };
        let seed0 = compiled.ic_grad_seed(&params, &est).unwrap();
        let recs =
            crate::ode::integrate_obs_sensitivity(&compiled, &params, &est, None, &seed0, &cfg, &obs_times)
                .unwrap();
        let projections: Vec<StreamProjection> = compiled
            .model
            .observations
            .iter()
            .map(|om| StreamProjection::from_ir(&om.projection, &compiled, &om.name).unwrap())
            .collect();
        let per_stream: Vec<Vec<f64>> = projections
            .iter()
            .map(|proj| {
                recs.iter()
                    .map(|r| {
                        let v: f64 = match proj {
                            StreamProjection::FlowSum(idxs) => idxs.iter().map(|&i| r.inc[i]).sum(),
                            StreamProjection::IntCompSum(idxs) => {
                                idxs.iter().map(|&i| r.counts[i]).sum()
                            }
                            StreamProjection::Expr(_) => panic!("fixture has no Expr projection"),
                        };
                        v.round()
                    })
                    .collect()
            })
            .collect();
        assert!(per_stream[0].iter().sum::<f64>() > 1.0, "incidence data must be nonzero");

        let obs_model = build_obs_model(compiled.clone(), &obs_times, per_stream);
        let (ll, grad) = det_grad(&compiled, &obs_model, &obs_times, dt, &params, &est, None).unwrap();
        assert!(ll.is_finite(), "det_grad loglik must be finite, got {ll}");

        let eps = 1e-4; // I0 ~ 10, so a larger step keeps the FD well-conditioned.
        let names = ["beta", "I0"];
        for (i, &midx) in est.iter().enumerate() {
            let mut pp = params.clone();
            let mut pm = params.clone();
            pp[midx] += eps;
            pm[midx] -= eps;
            let (llp, _) = det_grad(&compiled, &obs_model, &obs_times, dt, &pp, &est, None).unwrap();
            let (llm, _) = det_grad(&compiled, &obs_model, &obs_times, dt, &pm, &est, None).unwrap();
            let fd = (llp - llm) / (2.0 * eps);
            let rel = if fd.abs() > 1e-8 {
                (grad[i] - fd).abs() / fd.abs()
            } else {
                (grad[i] - fd).abs()
            };
            assert!(
                rel < 1e-3,
                "det_grad ∂/∂{} = {} vs FD {} (rel err {:.2e})",
                names[i],
                grad[i],
                fd,
                rel
            );
        }
        // Non-vacuity: the IC-parameter gradient must be materially nonzero — this
        // is the whole point (a zero seed would make it identically zero).
        assert!(grad[1].abs() > 1e-3, "∂/∂I0 should be materially nonzero (the seed drives it)");
    }
}
