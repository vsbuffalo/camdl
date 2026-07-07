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
) -> Result<(f64, Vec<f64>), SimError> {
    let d = estimated_to_model.len();

    // §1h capability gate: the ONE place that answers "can this model be fit by the
    // ODE gradient?" — refusing (with the compiler's own reason where one exists)
    // an unsupported rate/obs/σ² gradient, a nonsmooth ∂rate/∂state, an adaptive
    // integrator, a `dt`-in-rate model, a scheduled effect, a parameterized initial
    // condition (the `ic_grad` seed is a follow-up), a DerivedExpr projection, and a
    // `projected`-transforming likelihood argument. Run before any gradient is
    // taken, so a refusal is a single actionable message, not a mid-integration
    // failure. The seed below is therefore zero: the gate has established the
    // initial condition is explicit (`∂init/∂θ = 0`).
    let estimated: std::collections::HashSet<&str> = estimated_to_model
        .iter()
        .map(|&i| compiled.model.parameters[i].name.as_str())
        .collect();
    crate::inference::gradient_capability::preflight_gradient_ode(compiled, params, &estimated)?;

    let (int_s0, _) = compiled.initial_state(params)?;
    let state_sens_0 = vec![0.0f64; int_s0.counts.len() * d];

    let cfg = OdeConfig {
        t_start: compiled.model.simulation.t_start,
        t_end: compiled.model.simulation.t_end,
        dt,
    };

    let records = crate::ode::integrate_obs_sensitivity(
        compiled,
        params,
        estimated_to_model,
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
    use ir::expr::{ConstExpr, Expr, ParamExpr, ProjectedExpr};
    use ir::observation::{Likelihood, NegBinomialLikelihood, PoissonLikelihood};
    use ir::Diffable;
    use std::collections::HashMap;
    use std::sync::Arc;

    fn projected() -> Expr {
        Expr::Projected(ProjectedExpr { projected: () })
    }
    fn grad1(param: &str, e: Expr) -> HashMap<String, DerivEntry> {
        HashMap::from([(param.to_string(), DerivEntry::Grad(e))])
    }

    /// `seir_observations` made ODE-gradient-testable. Its `rate_state_grad` is
    /// the compiler-EMITTED J_x — `infection = beta·S·I/N` with `N = S+E+I+R` (a
    /// hoisted `PopSum` binding), so `∂rate/∂{S,E,I,R}` carries the full
    /// product/quotient rule through the binding, exactly the WrtPop autodiff the
    /// emission wires. The initial condition is made explicit (so
    /// `S(t_start)=∂init/∂θ=0`, det_grad v1), and the two obs streams are rewritten
    /// so their `projected`-bearing argument IS exactly `projected`:
    ///
    /// - `weekly_cases`: `negbin(mean = incidence, dispersion = k)` — the incidence
    ///   `FlowSum` factor-2 (`k` also exercises factor-1, the θ-direct term).
    /// - `detection`: `poisson(rate = I)` — the prevalence `IntCompSum` factor-2
    ///   (`∂g/∂x · S`).
    fn seir_ode_grad_fixture() -> ir::Model {
        let manifest = std::env::var("CARGO_MANIFEST_DIR").unwrap();
        let path = std::path::PathBuf::from(&manifest)
            .join("../../../ocaml/golden/seir_observations.ir.json");
        let mut model: ir::Model = ir::from_str(
            &std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("read: {e}")),
        )
        .unwrap_or_else(|e| panic!("parse: {e}"));

        // The emitted J_x must be present — this test validates it end-to-end.
        assert!(
            model.transitions.iter().any(|t| !t.rate_state_grad.0.is_empty()),
            "seir_observations must carry emitted rate_state_grad (run make update-golden)"
        );

        // Explicit (constant) initial condition: ∂init/∂θ = 0.
        model.initial_conditions = ir::model::InitialConditions::Explicit(HashMap::from([
            ("S".to_string(), 9990.0),
            ("E".to_string(), 0.0),
            ("I".to_string(), 10.0),
            ("R".to_string(), 0.0),
        ]));

        // Rewrite obs likelihoods so `projected` enters directly.
        for om in &mut model.observations {
            match om.name.as_str() {
                "weekly_cases" => {
                    om.likelihood = Likelihood::NegBinomial(NegBinomialLikelihood {
                        mean: Diffable { expr: projected(), grad: HashMap::new() },
                        dispersion: Diffable {
                            expr: Expr::Param(ParamExpr { param: "k".to_string() }),
                            grad: grad1("k", Expr::Const(ConstExpr { value: 1.0 })),
                        },
                    });
                }
                "detection" => {
                    om.likelihood = Likelihood::Poisson(PoissonLikelihood {
                        rate: Diffable { expr: projected(), grad: HashMap::new() },
                    });
                }
                other => panic!("unexpected obs stream {other}"),
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
        ];

        // Synthesize obs data = the projected value at the true params (near the
        // mode → well-conditioned FD). Uses the recorder + each stream's resolved
        // projection, exactly as det_grad reads them.
        let cfg = OdeConfig { t_start: compiled.model.simulation.t_start, t_end: 60.0, dt };
        let (int_s0, _) = compiled.initial_state(&params).unwrap();
        let seed = vec![0.0; int_s0.counts.len() * est.len()];
        let recs =
            crate::ode::integrate_obs_sensitivity(&compiled, &params, &est, &seed, &cfg, &obs_times)
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

        let (ll, grad) = det_grad(&compiled, &obs_model, &obs_times, dt, &params, &est).unwrap();
        assert!(ll.is_finite(), "det_grad loglik must be finite, got {ll}");
        assert_eq!(grad.len(), est.len());

        let eps = 1e-6;
        let names = ["beta", "gamma", "k"];
        for (i, &midx) in est.iter().enumerate() {
            let mut pp = params.clone();
            let mut pm = params.clone();
            pp[midx] += eps;
            pm[midx] -= eps;
            let (llp, _) = det_grad(&compiled, &obs_model, &obs_times, dt, &pp, &est).unwrap();
            let (llm, _) = det_grad(&compiled, &obs_model, &obs_times, dt, &pm, &est).unwrap();
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
}
