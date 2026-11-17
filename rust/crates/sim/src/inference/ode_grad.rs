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
/// output grid, matching `compute_ode_loglik`). `dt` is the fixed RK4 step;
/// `burnin_dt` is the coarse step for the unscored warm-up `[t_start, first_obs)`
/// (gh#396 follow-on) — `burnin_dt <= dt` disables it (fine step throughout). The
/// coarse region integrates state and sensitivity together, so the returned
/// gradient stays consistent with the coarsely-computed value.
///
/// Scheduled effects are integrated through: the RK4 step is split exactly at
/// each effect instant and the state jump is applied together with its
/// sensitivity jump `S⁺ = ∂Φ/∂x·S⁻ + ∂Φ/∂θ` (gh#275 §1g,
/// [`crate::effects::SensBlock`]), using the compiler-emitted `∂amount/∂θ` of
/// each action.
///
/// Refuses (hard error) the cases that cannot be differentiated soundly, each
/// with a reason: an adaptive integrator, a `dt`-in-rate model, an effect the
/// sensitivity cannot cross (a reactive or θ-timed one, a `transfer(count = …)`,
/// a state-dependent amount), an initial condition the `ic_grad` seed cannot
/// express, a nonsmooth `DerivedExpr` prevalence projection or
/// `projected`-transforming likelihood argument — all in
/// [`crate::inference::gradient_capability::preflight_gradient_ode`].
#[allow(clippy::too_many_arguments)]
pub fn det_grad(
    compiled: &CompiledModel,
    obs_model: &MultiStreamObsModel,
    obs_times: &[f64],
    dt: f64,
    burnin_dt: f64,
    params: &[f64],
    estimated_to_model: &[usize],
) -> Result<(f64, Vec<f64>), SimError> {
    // §1h capability gate: the ONE place that answers "can this model be fit by the
    // ODE gradient?" — refusing (with the compiler's own reason where one exists)
    // an unsupported rate/obs/σ² gradient, a nonsmooth ∂rate/∂state, an adaptive
    // integrator, a `dt`-in-rate model, a scheduled effect the sensitivity cannot
    // cross, a nonsmooth or real-compartment initial condition, a DerivedExpr
    // projection, and a
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
    let state_sens_0 = compiled.ic_grad_seed(params, estimated_to_model)?;

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
        burnin_dt,
    )?;

    obs_model.ode_loglik_and_grad(&records, params, estimated_to_model)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::inference::multi_stream_obs::{StreamProjection, StreamSpec, StreamTimes};
    use crate::inference::{compute_ode_loglik, dense_cells, BoundObs, MultiStreamObsModel};
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
        model.initial_conditions = ir::model::InitialConditions::constants([
            ("S".to_string(), 9990.0),
            ("E".to_string(), 0.0),
            ("I".to_string(), 10.0),
            ("R".to_string(), 0.0),
        ]);
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
                    times: StreamTimes::contiguous_for(&projection, compiled.model.simulation.t_start, obs_times.to_vec()).unwrap(),
                    projection,
                    ir_model: om.clone(),
                    observations: dense_cells(per_stream[si].clone()),
                    aux: vec![],
                }
            })
            .collect();
        let t_start = compiled.model.simulation.t_start;
        MultiStreamObsModel::new(BoundObs::bind(t_start, specs).unwrap().0, compiled).unwrap()
    }

    /// Everything `det_grad` needs for the SEIR incidence+prevalence scenario:
    /// the compiled model, the bound observation model (data synthesized at the
    /// true parameters), the parameter vector, the estimated-parameter model
    /// indices (`beta`, `gamma`, `k`, `rho`), the observation times and the RK4
    /// step. Shared by the finite-difference oracle and the cross-process
    /// reproducibility pin so both score exactly the same object.
    #[allow(clippy::type_complexity)]
    fn seir_det_grad_scenario(
    ) -> (Arc<CompiledModel>, MultiStreamObsModel, Vec<f64>, Vec<usize>, Vec<f64>, f64) {
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
        let (int_s0, _) = compiled.initial_state_mean(&params).unwrap();
        let seed = vec![0.0; int_s0.counts.len() * est.len()];
        let recs =
            crate::ode::integrate_obs_sensitivity(&compiled, &params, &est, &seed, &cfg, &obs_times, cfg.dt)
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
                            StreamProjection::FlowRatio { .. } => {
                                panic!("fixture has no FlowRatio projection")
                            }
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
        (compiled, obs_model, params, est, obs_times, dt)
    }

    /// The det_grad FD oracle (gh#275 §1f): `∇_symbolic` vs a central finite
    /// difference of the loglik, over an incidence stream and a prevalence stream,
    /// exercising factor-1 (`k`) and factor-2 (`beta`, `gamma`, through both an
    /// `inc_sens` incidence chain and a `state_sens` prevalence chain).
    #[test]
    fn det_grad_matches_finite_difference_seir_incidence_and_prevalence() {
        let (compiled, obs_model, params, est, obs_times, dt) = seir_det_grad_scenario();

        let (ll, grad) = det_grad(&compiled, &obs_model, &obs_times, dt, dt, &params, &est).unwrap();
        assert!(ll.is_finite(), "det_grad loglik must be finite, got {ll}");
        assert_eq!(grad.len(), est.len());

        let eps = 1e-6;
        let names = ["beta", "gamma", "k", "rho"];
        for (i, &midx) in est.iter().enumerate() {
            let mut pp = params.clone();
            let mut pm = params.clone();
            pp[midx] += eps;
            pm[midx] -= eps;
            let (llp, _) = det_grad(&compiled, &obs_model, &obs_times, dt, dt, &pp, &est).unwrap();
            let (llm, _) = det_grad(&compiled, &obs_model, &obs_times, dt, dt, &pm, &est).unwrap();
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

    /// Environment marker that turns this test binary into the child half of
    /// [`det_grad_is_bit_identical_across_processes`].
    const GH682_CHILD: &str = "CAMDL_GH682_DET_GRAD_CHILD";

    /// gh#682: `det_grad` must return the SAME BITS in every process.
    ///
    /// The compiler-emitted `∂rate/∂compartment` map is a `HashMap`, and Rust's
    /// default `RandomState` is seeded per process, so the resolved entry order —
    /// which is the summation order of `Σ_j ∂rate/∂x_j · S[j,p]` in
    /// `ode::sensitivity_derivs` — used to vary from run to run. Floating-point
    /// addition is not associative, so the returned gradient varied in its last
    /// bits, and `nuts` on the `ode` backend and the gradient MLE did not
    /// reproduce: four invocations of one identical `nl-lbfgs` config took 94, 95,
    /// 96 and 106 objective evaluations.
    ///
    /// A single process cannot observe this, so the test re-invokes its own test
    /// binary with `GH682_CHILD` set; each child scores the same scenario, prints
    /// the loglik and gradient as raw `f64` bits, and the parent requires every
    /// child to agree exactly. Eight children: a random order over the four
    /// compartments of the `infection` rate agrees across all eight only by
    /// accident.
    #[test]
    fn det_grad_is_bit_identical_across_processes() {
        if std::env::var_os(GH682_CHILD).is_some() {
            let (compiled, obs_model, params, est, obs_times, dt) = seir_det_grad_scenario();
            let (ll, grad) =
                det_grad(&compiled, &obs_model, &obs_times, dt, dt, &params, &est).unwrap();
            let mut line = format!("GH682-BITS {:016x}", ll.to_bits());
            for g in &grad {
                line.push_str(&format!(" {:016x}", g.to_bits()));
            }
            println!("{line}");
            return;
        }

        let exe = std::env::current_exe().expect("the running test binary's path");
        let mut seen: Vec<String> = Vec::new();
        for run in 0..8 {
            let out = std::process::Command::new(&exe)
                .args([
                    "--exact",
                    "inference::ode_grad::tests::det_grad_is_bit_identical_across_processes",
                    "--nocapture",
                ])
                .env(GH682_CHILD, "1")
                .output()
                .expect("spawn the child test process");
            let stdout = String::from_utf8_lossy(&out.stdout).into_owned();
            assert!(
                out.status.success(),
                "child run {run} failed ({}):\n{stdout}\n{}",
                out.status,
                String::from_utf8_lossy(&out.stderr)
            );
            let line = stdout
                .lines()
                .find(|l| l.starts_with("GH682-BITS "))
                .unwrap_or_else(|| panic!("child run {run} printed no bits line:\n{stdout}"))
                .to_string();
            seen.push(line);
        }

        // Non-vacuity: the scenario must carry a real gradient, not four zeros
        // that would agree whatever the summation order.
        let fields: Vec<&str> = seen[0].split_whitespace().skip(1).collect();
        assert_eq!(fields.len(), 5, "one loglik + four gradient components");
        assert!(
            fields[1..].iter().any(|h| {
                let v = f64::from_bits(u64::from_str_radix(h, 16).unwrap());
                v.abs() > 1e-6
            }),
            "the scenario must produce a materially nonzero gradient: {}",
            seen[0]
        );

        let distinct: std::collections::BTreeSet<&String> = seen.iter().collect();
        assert_eq!(
            distinct.len(),
            1,
            "det_grad returned {} distinct results over 8 processes (gh#682 — the \
             ∂rate/∂compartment summation order is not pinned):\n{}",
            distinct.len(),
            seen.join("\n")
        );
    }

    /// gh#682, the in-process half: every resolved gradient map must be ordered by
    /// its index, because that order is the summation order the ODE forward
    /// sensitivity uses and `HashMap` iteration order is not stable.
    ///
    /// One compile would not catch a shuffle: `RandomState` re-keys per `HashMap`,
    /// so every fresh deserialize of the golden orders differently. Twelve compiles
    /// make an accidental pass (a random order over five compartments is sorted
    /// about one time in 120) vanishingly unlikely.
    ///
    /// `seir_vaccine_seasonal` rather than the `det_grad` fixture: its widest
    /// `∂rate/∂θ` map carries three parameters and its widest `∂rate/∂compartment`
    /// map five compartments, so neither check is vacuous. (Every `seir_observations`
    /// rate reads a single parameter, which is sorted whatever the hash seed.)
    #[test]
    fn resolved_gradient_maps_are_index_ordered() {
        let manifest = std::env::var("CARGO_MANIFEST_DIR").unwrap();
        let path = std::path::PathBuf::from(&manifest)
            .join("../../../ocaml/golden/seir_vaccine_seasonal.ir.json");
        let src = std::fs::read_to_string(&path).unwrap();

        let mut widest_state = 0usize;
        let mut widest_param = 0usize;
        for compile in 0..12 {
            let mut model: ir::Model = ir::from_str(&src).unwrap();
            for p in &mut model.parameters {
                if p.value.resolved_value().is_none() {
                    p.value = p.value.with_value(0.5);
                }
            }
            let compiled = CompiledModel::new(model).unwrap();
            for (tr_idx, cg) in compiled.resolved.rate_state_grads_indexed.iter().enumerate() {
                let idxs: Vec<usize> = cg.0.iter().map(|(j, _)| *j).collect();
                widest_state = widest_state.max(idxs.len());
                assert!(
                    idxs.windows(2).all(|w| w[0] < w[1]),
                    "compile {compile}: rate_state_grad for transition {tr_idx} is not \
                     ordered by compartment index: {idxs:?}"
                );
            }
            for (tr_idx, g) in compiled.resolved.rate_grads_indexed.iter().enumerate() {
                let idxs: Vec<usize> = g.iter().map(|(p, _)| *p).collect();
                widest_param = widest_param.max(idxs.len());
                assert!(
                    idxs.windows(2).all(|w| w[0] < w[1]),
                    "compile {compile}: rate_grad for transition {tr_idx} is not ordered by \
                     parameter index: {idxs:?}"
                );
            }
        }
        // Non-vacuity: a map of one entry is sorted whatever the hash seed.
        assert!(widest_state >= 2, "no multi-entry ∂rate/∂compartment map was checked");
        assert!(widest_param >= 2, "no multi-entry ∂rate/∂θ map was checked");
    }

    /// The COARSE-`burnin_dt` FD oracle (gh#396 follow-on) — the correctness gate
    /// for the burn-in feature. Integrate the augmented `(x, S)` system with a
    /// coarse RK4 step on the unscored warm-up `[t_start, first_obs)` and the exact
    /// `dt` on the scored window, then check that the returned gradient is the exact
    /// derivative of the *coarsely-computed* value: `det_grad(burnin_dt=K).grad` vs
    /// a central finite difference of `det_grad(burnin_dt=K).value`. This proves the
    /// central claim — the sensitivity flows through the coarse region so value and
    /// gradient stay mutually consistent (unlike a frozen checkpoint, whose
    /// `∂x*/∂θ ≡ 0` would make the gradient miss the warm-up entirely).
    ///
    /// A slow epidemic (`beta = 0.15`, `R0 = 1.5`) keeps incidence/prevalence
    /// materially nonzero across the scored window and the coarse `dt = 5` steps
    /// well within RK4 stability for the fastest (5-day latent) compartment, so no
    /// clamp fires (the clamp refusal is a separate test). `first_obs = 50` gives a
    /// real `[0, 50)` warm-up = ten coarse steps that land exactly on 50.
    ///
    /// The `ll_coarse != ll_fine` assertion is the non-vacuity guard: it proves the
    /// coarse path actually integrated differently than the fine one, so the
    /// gradient it matches is genuinely the coarse gradient (a bug that silently
    /// ignored `burnin_dt` would leave `grad` = the fine gradient, which would then
    /// NOT match the coarse FD). RED-CHECK (verified during development): resetting
    /// `S` to zero at `cond_from` — the frozen-checkpoint failure mode — drops the
    /// warm-up sensitivity and `∂/∂beta` lands far above the 1e-4 gate vs the FD.
    #[test]
    fn det_grad_matches_finite_difference_under_coarse_burnin_dt() {
        let mut model = seir_ode_grad_fixture();
        set_defaults(&mut model);
        model.simulation.t_start = 0.0;
        model.simulation.t_end = 150.0;
        let compiled = Arc::new(CompiledModel::new(model).unwrap());

        let n = compiled.param_index.len();
        let mut params = vec![0.0; n];
        for p in &compiled.model.parameters {
            params[compiled.param_index[p.name.as_str()]] = p.value.resolved_value().unwrap();
        }
        // Slow epidemic so it stays active in the scored window and the coarse
        // steps stay well within RK4 stability.
        params[compiled.param_index["beta"]] = 0.15;

        let dt = 1.0;
        let burnin_dt = 5.0;
        // first obs = 50 ⇒ warm-up [0, 50) integrated coarsely (10 steps of 5).
        let obs_times: Vec<f64> = (5..=15).map(|w| (w * 10) as f64).collect(); // 50..150
        let est = vec![
            compiled.param_index["beta"],
            compiled.param_index["gamma"],
            compiled.param_index["k"],
            compiled.param_index["rho"],
        ];

        // Synthesize data with the SAME coarse integration at the true params, so the
        // FD is well-conditioned (data near the coarse-likelihood mode).
        let cfg = OdeConfig { t_start: 0.0, t_end: 150.0, dt };
        let (int_s0, _) = compiled.initial_state_mean(&params).unwrap();
        let seed = vec![0.0; int_s0.counts.len() * est.len()];
        let recs = crate::ode::integrate_obs_sensitivity(
            &compiled, &params, &est, &seed, &cfg, &obs_times, burnin_dt,
        )
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
                            StreamProjection::FlowRatio { .. } => {
                                panic!("fixture has no FlowRatio projection")
                            }
                        };
                        v.round()
                    })
                    .collect()
            })
            .collect();
        assert!(per_stream[0].iter().sum::<f64>() > 1.0, "coarse incidence data must be nonzero");
        assert!(per_stream[1].iter().sum::<f64>() > 1.0, "coarse prevalence data must be nonzero");

        let obs_model = build_obs_model(compiled.clone(), &obs_times, per_stream);

        // The coarse gradient + value.
        let (ll_coarse, grad) =
            det_grad(&compiled, &obs_model, &obs_times, dt, burnin_dt, &params, &est).unwrap();
        assert!(ll_coarse.is_finite(), "coarse det_grad loglik must be finite, got {ll_coarse}");

        // Non-vacuity: the coarse path must integrate DIFFERENTLY than the fine one —
        // otherwise `grad` is the fine gradient and matching the coarse FD proves
        // nothing.
        let (ll_fine, _) =
            det_grad(&compiled, &obs_model, &obs_times, dt, dt, &params, &est).unwrap();
        assert!(
            (ll_coarse - ll_fine).abs() > 1e-6,
            "coarse burn-in must change the value vs fine dt (ll_coarse {ll_coarse} vs \
             ll_fine {ll_fine}); the test is vacuous otherwise"
        );

        // The gradient is the exact derivative of the coarse value.
        let eps = 1e-6;
        let names = ["beta", "gamma", "k", "rho"];
        for (i, &midx) in est.iter().enumerate() {
            let mut pp = params.clone();
            let mut pm = params.clone();
            pp[midx] += eps;
            pm[midx] -= eps;
            let (llp, _) =
                det_grad(&compiled, &obs_model, &obs_times, dt, burnin_dt, &pp, &est).unwrap();
            let (llm, _) =
                det_grad(&compiled, &obs_model, &obs_times, dt, burnin_dt, &pm, &est).unwrap();
            let fd = (llp - llm) / (2.0 * eps);
            let rel = if fd.abs() > 1e-8 {
                (grad[i] - fd).abs() / fd.abs()
            } else {
                (grad[i] - fd).abs()
            };
            assert!(
                rel < 1e-4,
                "coarse det_grad ∂/∂{} = {} vs FD {} (rel err {:.2e})",
                names[i], grad[i], fd, rel
            );
        }
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
        let seed = vec![0.0; compiled.initial_state_mean(&params).unwrap().0.counts.len() * est.len()];
        let recs =
            crate::ode::integrate_obs_sensitivity(&compiled, &params, &est, &seed, &cfg, &obs_times, cfg.dt)
                .unwrap();
        let flow_idx = 0usize; // weekly_cases is the incidence stream (FlowSum)
        let weekly: Vec<f64> = recs.iter().map(|r| r.inc[flow_idx].round()).collect();
        let detection: Vec<f64> = recs.iter()
            .map(|r| (r.counts[i_idx].max(0.0).sqrt() + r.counts[r_idx].max(0.0).sqrt()).round())
            .collect();
        assert!(detection.iter().sum::<f64>() > 1.0, "√I+√R detection data must be nonzero");

        let obs_model = build_obs_model(compiled.clone(), &obs_times, vec![weekly, detection]);
        let (ll, grad) = det_grad(&compiled, &obs_model, &obs_times, dt, dt, &params, &est).unwrap();
        assert!(ll.is_finite(), "det_grad loglik must be finite, got {ll}");

        let eps = 1e-6;
        let names = ["beta", "gamma"];
        for (i, &midx) in est.iter().enumerate() {
            let mut pp = params.clone();
            let mut pm = params.clone();
            pp[midx] += eps;
            pm[midx] -= eps;
            let (llp, _) = det_grad(&compiled, &obs_model, &obs_times, dt, dt, &pp, &est).unwrap();
            let (llm, _) = det_grad(&compiled, &obs_model, &obs_times, dt, dt, &pm, &est).unwrap();
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
        model.initial_conditions = ir::model::InitialConditions::exprs([
            ("S".to_string(), Expr::bin_op(BinOp::Sub, param("N0"), param("I0"))),
            ("E".to_string(), Expr::Const(ConstExpr { value: 0.0 })),
            ("I".to_string(), param("I0")),
            ("R".to_string(), Expr::Const(ConstExpr { value: 0.0 })),
        ]);
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
            crate::ode::integrate_obs_sensitivity(&compiled, &params, &est, &seed0, &cfg, &obs_times, cfg.dt)
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
                            StreamProjection::FlowRatio { .. } => {
                                panic!("fixture has no FlowRatio projection")
                            }
                        };
                        v.round()
                    })
                    .collect()
            })
            .collect();
        assert!(per_stream[0].iter().sum::<f64>() > 1.0, "incidence data must be nonzero");

        let obs_model = build_obs_model(compiled.clone(), &obs_times, per_stream);
        let (ll, grad) = det_grad(&compiled, &obs_model, &obs_times, dt, dt, &params, &est).unwrap();
        assert!(ll.is_finite(), "det_grad loglik must be finite, got {ll}");

        let eps = 1e-4; // I0 ~ 10, so a larger step keeps the FD well-conditioned.
        let names = ["beta", "I0"];
        for (i, &midx) in est.iter().enumerate() {
            let mut pp = params.clone();
            let mut pm = params.clone();
            pp[midx] += eps;
            pm[midx] -= eps;
            let (llp, _) = det_grad(&compiled, &obs_model, &obs_times, dt, dt, &pp, &est).unwrap();
            let (llm, _) = det_grad(&compiled, &obs_model, &obs_times, dt, dt, &pm, &est).unwrap();
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

    // ── gh#680: per-stream incidence binning under heterogeneous cadences ──────
    //
    // The value path (`compute_ode_loglik`, used by `mh`) bins incidence in two
    // levels: a blanket per-transition `cum_flows` tally, plus a per-stream `acc`
    // that is folded before scoring and zeroed only for the streams scheduled at
    // THAT union index. The gradient path must bin identically — value AND
    // sensitivity — or a stream on a longer cadence is scored against a shorter
    // modelled window.
    //
    // `compute_ode_loglik` is the oracle: the two paths target the same posterior,
    // so their logliks must agree, and the analytic gradient must match a central
    // finite difference OF THE VALUE PATH. An FD of `det_grad`'s own value cannot
    // see this bug — a mis-binned value is still smooth in θ, and its own forward
    // sensitivity is its exact derivative.

    /// Two `FlowSum` (incidence) streams over DISJOINT transition sets, each
    /// `poisson(rate = projected)`, so the whole gradient is factor 2 — the
    /// `inc_sens` chain the per-stream binning feeds. Explicit integer initial
    /// condition so the value path (rounded `initial_state`) and the gradient path
    /// (`initial_state_continuous`) start from the same state and their flows are
    /// directly comparable.
    ///
    /// `flow_sets` names each stream's transitions. One name lowers to
    /// `Projection::CumulativeFlow` → a single-index `FlowSum`; two or more lower to
    /// `Projection::CumulativeFlowSum` → a MULTI-index `FlowSum`, which is what an
    /// un-indexed `incidence()` over a stratified transition family produces
    /// (`StreamProjection::from_ir`, §25.4).
    fn two_incidence_stream_model(t_end: f64, flow_sets: [&[&str]; 2]) -> ir::Model {
        let manifest = std::env::var("CARGO_MANIFEST_DIR").unwrap();
        let path = std::path::PathBuf::from(&manifest)
            .join("../../../ocaml/golden/seir_observations.ir.json");
        let mut model: ir::Model =
            ir::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();
        model.initial_conditions = ir::model::InitialConditions::constants([
            ("S".to_string(), 9990.0),
            ("E".to_string(), 0.0),
            ("I".to_string(), 10.0),
            ("R".to_string(), 0.0),
        ]);
        model.ic_grad = HashMap::new();
        model.simulation.t_end = t_end;

        let base = model.observations[0].clone();
        let mk = |name: &str, transitions: &[&str]| {
            let mut om = base.clone();
            om.name = name.to_string();
            om.source = name.to_string();
            om.projection = match transitions {
                [one] => ir::observation::Projection::CumulativeFlow(one.to_string()),
                many => ir::observation::Projection::CumulativeFlowSum(
                    many.iter().map(|s| s.to_string()).collect(),
                ),
            };
            om.projection_state_grad = Default::default();
            om.likelihood = Likelihood::Poisson(PoissonLikelihood {
                rate: Diffable {
                    expr: projected(),
                    grad: HashMap::new(),
                    // rate IS projected → ∂rate/∂projected = 1.
                    proj_grad: Some(DerivEntry::Grad(Expr::Const(ConstExpr { value: 1.0 }))),
                },
            });
            om
        };
        model.observations =
            vec![mk("cases_a", flow_sets[0]), mk("cases_b", flow_sets[1])];
        model
    }

    /// Synthetic counts for one stream under the VALUE path's binning: accumulate
    /// the stream's real flow — summed over ALL the transitions its projection
    /// selects — across snapshots, and close the bin at each of THIS stream's own
    /// observation times (never at a sibling's).
    fn windowed_counts(traj: &crate::state::Trajectory, trs: &[usize], times: &[f64]) -> Vec<f64> {
        let mut out = Vec::with_capacity(times.len());
        let mut acc = 0.0;
        let mut next = 0usize;
        for (i, snap) in traj.snapshots.iter().enumerate() {
            if i > 0 {
                acc += trs.iter().map(|&tr| snap.flows.as_real()[tr]).sum::<f64>();
            }
            if next < times.len() && (snap.t - times[next]).abs() < 1e-9 {
                out.push(acc.round());
                acc = 0.0;
                next += 1;
            }
        }
        assert_eq!(out.len(), times.len(), "every obs time must land on a snapshot");
        out
    }

    /// Build the two-stream fixture at `(times_a, times_b)` and return everything
    /// both paths need. Data is generated at the default θ; the likelihood is then
    /// evaluated at a DIFFERENT θ so the gradient is materially nonzero and the
    /// finite difference is well conditioned.
    #[allow(clippy::type_complexity)]
    fn two_stream_fixture(
        times_a: &[f64],
        times_b: &[f64],
        dt: f64,
        flow_sets: [&[&str]; 2],
    ) -> (Arc<CompiledModel>, MultiStreamObsModel, Vec<f64>, Vec<f64>, Vec<usize>) {
        let t_end = times_a.last().copied().unwrap().max(times_b.last().copied().unwrap());
        let mut model = two_incidence_stream_model(t_end, flow_sets);
        set_defaults(&mut model);
        let compiled = Arc::new(CompiledModel::new(model).unwrap());

        let n = compiled.param_index.len();
        let mut truth = vec![0.0; n];
        for p in &compiled.model.parameters {
            truth[compiled.param_index[p.name.as_str()]] = p.value.resolved_value().unwrap();
        }

        let cfg = OdeConfig { t_start: compiled.model.simulation.t_start, t_end, dt };
        let traj = crate::ode::run_ode(&compiled, &truth, &cfg, None, None).unwrap();
        let tr = |name: &str| {
            compiled.model.transitions.iter().position(|t| t.name == name).unwrap()
        };
        let trs = |names: &[&str]| names.iter().map(|n| tr(n)).collect::<Vec<_>>();
        let data_a = windowed_counts(&traj, &trs(flow_sets[0]), times_a);
        let data_b = windowed_counts(&traj, &trs(flow_sets[1]), times_b);
        assert!(data_a.iter().sum::<f64>() > 100.0, "cases_a data must be substantial");
        assert!(data_b.iter().sum::<f64>() > 10.0, "cases_b data must be substantial");

        let specs: Vec<StreamSpec> = compiled
            .model
            .observations
            .iter()
            .zip([(&data_a, times_a), (&data_b, times_b)])
            .map(|(om, (data, times))| StreamSpec {
                times: StreamTimes::contiguous_for(&StreamProjection::from_ir(&om.projection, &compiled, &om.name).unwrap(), compiled.model.simulation.t_start, times.to_vec()).unwrap(),
                projection: StreamProjection::from_ir(&om.projection, &compiled, &om.name).unwrap(),
                ir_model: om.clone(),
                observations: dense_cells(data.clone()),
                aux: vec![],
            })
            .collect();
        let obs_model = MultiStreamObsModel::new(
            BoundObs::bind(compiled.model.simulation.t_start, specs).unwrap().0,
            compiled.clone(),
        ).unwrap();

        // The UNION axis both paths score on.
        let mut union: Vec<f64> = times_a.iter().chain(times_b.iter()).copied().collect();
        union.sort_by(|a, b| a.partial_cmp(b).unwrap());
        union.dedup();

        // Evaluate away from the data-generating θ: a materially nonzero gradient.
        let mut params = truth.clone();
        params[compiled.param_index["beta"]] = 0.66;
        params[compiled.param_index["gamma"]] = 0.09;
        let est = vec![compiled.param_index["beta"], compiled.param_index["gamma"]];

        (compiled, obs_model, union, params, est)
    }

    /// `det_grad` (the `nuts` path) against `compute_ode_loglik` (the `mh` path)
    /// on a TWO-CADENCE model: the value must agree, and the analytic gradient must
    /// match a central finite difference of the value path. gh#680: the gradient
    /// path zeroed its incidence tally at every union observation time, so the
    /// 14-day stream was scored against a 7-day modelled window.
    #[test]
    fn det_grad_bins_multi_cadence_incidence_like_the_value_path() {
        let dt = 1.0;
        let times_a: Vec<f64> = (1..=8).map(|w| (w * 7) as f64).collect(); // 7 d
        let times_b: Vec<f64> = (1..=4).map(|w| (w * 14) as f64).collect(); // 14 d
        let (compiled, obs_model, union, params, est) =
            two_stream_fixture(&times_a, &times_b, dt, [&["infection"], &["recovery"]]);
        assert_eq!(union.len(), 8, "the union axis is the 7-day grid");
        assert_eq!(obs_model.n_interval_streams(), 2, "both streams are incidence bins");

        let (ll_grad, grad) =
            det_grad(&compiled, &obs_model, &union, dt, dt, &params, &est).unwrap();
        let ll_value =
            compute_ode_loglik(&compiled, &obs_model, &union, dt, &params, dt).unwrap();
        assert!(ll_grad.is_finite() && ll_value.is_finite());
        assert!(
            (ll_grad - ll_value).abs() < 1e-6,
            "gradient-path loglik {ll_grad} disagrees with value-path loglik {ll_value} \
             (Δ = {:.6e}) — the two paths must bin multi-cadence incidence identically",
            ll_grad - ll_value
        );

        let eps = 1e-6;
        let names = ["beta", "gamma"];
        for (i, &midx) in est.iter().enumerate() {
            let mut pp = params.clone();
            let mut pm = params.clone();
            pp[midx] += eps;
            pm[midx] -= eps;
            let llp = compute_ode_loglik(&compiled, &obs_model, &union, dt, &pp, dt).unwrap();
            let llm = compute_ode_loglik(&compiled, &obs_model, &union, dt, &pm, dt).unwrap();
            let fd = (llp - llm) / (2.0 * eps);
            let rel = (grad[i] - fd).abs() / fd.abs().max(1e-8);
            assert!(
                rel < 1e-4,
                "det_grad ∂/∂{} = {} vs value-path FD {} (rel err {:.2e})",
                names[i], grad[i], fd, rel
            );
        }
        // Non-vacuity: both directions must carry real signal, or the agreement
        // above is agreement about nothing.
        assert!(grad[0].abs() > 1.0, "∂/∂beta should be materially nonzero, got {}", grad[0]);
        assert!(grad[1].abs() > 1.0, "∂/∂gamma should be materially nonzero, got {}", grad[1]);
    }

    /// The multi-index case of the same binning: a stream whose `FlowSum` selects
    /// TWO transitions, observed on a cadence four times longer than its sibling's.
    ///
    /// `Projection::CumulativeFlowSum` — an un-indexed `incidence()` over a
    /// stratified transition family — lowers to a multi-index `FlowSum`
    /// (`StreamProjection::from_ir`), and the models that reach for it are exactly
    /// the several-incidence-stream models this per-stream binning governs. Every
    /// other case here is single-index, so on its own none of them pins that the bin
    /// sums the right SET of transitions over the right WINDOW: a fold that took only
    /// `flow_indices[0]`, or a scorer that re-summed `rec.inc` over `idxs` after
    /// binning, would pass them all.
    ///
    /// `cases_a = incidence(infection) + incidence(progression)` at 28 d against
    /// `cases_b = incidence(recovery)` at 7 d. The two folded transitions run at
    /// materially different rates, so dropping or transposing one moves the projected
    /// value; the 4:1 cadence ratio means a blanket reset would score `cases_a`'s
    /// four-week count against one week of modelled flow.
    #[test]
    fn det_grad_bins_a_multi_index_flowsum_at_a_second_cadence() {
        let dt = 1.0;
        let times_a: Vec<f64> = (1..=2).map(|q| (q * 28) as f64).collect(); // 28 d
        let times_b: Vec<f64> = (1..=8).map(|w| (w * 7) as f64).collect(); // 7 d
        let (compiled, obs_model, union, params, est) = two_stream_fixture(
            &times_a,
            &times_b,
            dt,
            [&["infection", "progression"], &["recovery"]],
        );
        assert_eq!(union.len(), 8, "the union axis is the 7-day grid");
        assert_eq!(obs_model.n_interval_streams(), 2, "both streams are incidence bins");
        // The slot map really is multi-index, and holds the two named transitions —
        // `CumulativeFlowSum` resolving to one index, or to the wrong pair, would
        // make everything below agree about the wrong projection.
        let tr = |name: &str| {
            compiled.model.transitions.iter().position(|t| t.name == name).unwrap()
        };
        let selected = obs_model.incidence_streams();
        assert_eq!(
            selected[0],
            ("cases_a".to_string(), vec![tr("infection"), tr("progression")]),
            "cases_a must select BOTH transitions — this test exists for the multi-index \
             FlowSum that `CumulativeFlowSum` lowers to; got {:?}",
            selected
        );
        assert_eq!(
            selected[1],
            ("cases_b".to_string(), vec![tr("recovery")]),
            "cases_b is the single-index sibling on the shorter cadence; got {:?}",
            selected
        );

        let (ll_grad, grad) =
            det_grad(&compiled, &obs_model, &union, dt, dt, &params, &est).unwrap();
        let ll_value =
            compute_ode_loglik(&compiled, &obs_model, &union, dt, &params, dt).unwrap();
        assert!(ll_grad.is_finite() && ll_value.is_finite());
        assert!(
            (ll_grad - ll_value).abs() < 1e-6,
            "gradient-path loglik {ll_grad} disagrees with value-path loglik {ll_value} \
             (Δ = {:.6e}) — a multi-index FlowSum must bin the same transition set over \
             the same per-stream window on both paths",
            ll_grad - ll_value
        );

        let eps = 1e-6;
        let names = ["beta", "gamma"];
        for (i, &midx) in est.iter().enumerate() {
            let mut pp = params.clone();
            let mut pm = params.clone();
            pp[midx] += eps;
            pm[midx] -= eps;
            let llp = compute_ode_loglik(&compiled, &obs_model, &union, dt, &pp, dt).unwrap();
            let llm = compute_ode_loglik(&compiled, &obs_model, &union, dt, &pm, dt).unwrap();
            let fd = (llp - llm) / (2.0 * eps);
            let rel = (grad[i] - fd).abs() / fd.abs().max(1e-8);
            assert!(
                rel < 1e-4,
                "det_grad ∂/∂{} = {} vs value-path FD {} (rel err {:.2e})",
                names[i], grad[i], fd, rel
            );
        }
        // Non-vacuity: both directions must carry real signal, or the agreement
        // above is agreement about nothing.
        assert!(grad[0].abs() > 1.0, "∂/∂beta should be materially nonzero, got {}", grad[0]);
        assert!(grad[1].abs() > 1.0, "∂/∂gamma should be materially nonzero, got {}", grad[1]);
    }

    /// NEGATIVE CONTROL for gh#680: the same two-stream fixture with ONE cadence.
    /// Here the union index and every stream's schedule coincide, so per-stream
    /// binning and a blanket reset are the same operation — the common path. The
    /// printed full-precision value/gradient must be unchanged by the multi-cadence
    /// fix (bit-identical), and the value/FD agreement must hold as before.
    #[test]
    fn det_grad_single_cadence_is_unchanged() {
        let dt = 1.0;
        let times: Vec<f64> = (1..=8).map(|w| (w * 7) as f64).collect();
        let (compiled, obs_model, union, params, est) =
            two_stream_fixture(&times, &times, dt, [&["infection"], &["recovery"]]);
        assert_eq!(union, times, "single cadence: the union axis IS the shared grid");

        let (ll_grad, grad) =
            det_grad(&compiled, &obs_model, &union, dt, dt, &params, &est).unwrap();
        let ll_value =
            compute_ode_loglik(&compiled, &obs_model, &union, dt, &params, dt).unwrap();
        eprintln!(
            "gh#680 single-cadence control: ll = {:.17e} grad = [{:.17e}, {:.17e}]",
            ll_grad, grad[0], grad[1]
        );
        assert!(
            (ll_grad - ll_value).abs() < 1e-6,
            "single-cadence gradient-path loglik {ll_grad} disagrees with value-path \
             {ll_value} (Δ = {:.6e})",
            ll_grad - ll_value
        );

        let eps = 1e-6;
        let names = ["beta", "gamma"];
        for (i, &midx) in est.iter().enumerate() {
            let mut pp = params.clone();
            let mut pm = params.clone();
            pp[midx] += eps;
            pm[midx] -= eps;
            let llp = compute_ode_loglik(&compiled, &obs_model, &union, dt, &pp, dt).unwrap();
            let llm = compute_ode_loglik(&compiled, &obs_model, &union, dt, &pm, dt).unwrap();
            let fd = (llp - llm) / (2.0 * eps);
            let rel = (grad[i] - fd).abs() / fd.abs().max(1e-8);
            assert!(
                rel < 1e-4,
                "det_grad ∂/∂{} = {} vs value-path FD {} (rel err {:.2e})",
                names[i], grad[i], fd, rel
            );
        }
        assert!(grad[0].abs() > 1.0, "∂/∂beta should be materially nonzero, got {}", grad[0]);
        assert!(grad[1].abs() > 1.0, "∂/∂gamma should be materially nonzero, got {}", grad[1]);
    }

    /// Proposal 2026-09-09: a `FlowRatio` stream on the ODE gradient path.
    ///
    /// `cases_a = incidence(infection) / (incidence(infection) + incidence(recovery))`
    /// at 7 d, scored `binomial(n = 200, p = projected)`, beside
    /// `cases_b = incidence(recovery)` at 14 d. The ratio owns two bins and its
    /// sibling one — three bins for two streams — and the ratio's
    /// `∂projected/∂θ` is the quotient rule over the two sensitivity bins,
    /// `(b·∂a − a·∂b) / b²`. The value path (`compute_ode_loglik`, which reads
    /// the real bins) and the gradient path must agree on the value, and the
    /// analytic gradient must match a central finite difference of the value
    /// path. A reader that dropped the `a·∂b` term, or divided by `b` rather
    /// than `b²`, fails the finite-difference check.
    #[test]
    fn det_grad_differentiates_a_flow_ratio_by_the_quotient_rule() {
        let dt = 1.0;
        let times_a: Vec<f64> = (1..=8).map(|w| (w * 7) as f64).collect(); // 7 d
        let times_b: Vec<f64> = (1..=4).map(|w| (w * 14) as f64).collect(); // 14 d
        let t_end = 56.0;
        let mut model = two_incidence_stream_model(t_end, [&["infection"], &["recovery"]]);
        set_defaults(&mut model);
        // The first stream becomes the ratio, scored k of a fixed n.
        model.observations[0].projection = ir::observation::Projection::FlowRatio {
            numerator: vec!["infection".into()],
            denominator: vec!["infection".into(), "recovery".into()],
        };
        model.observations[0].likelihood =
            Likelihood::Binomial(ir::observation::BinomialLikelihood {
                n: Expr::Const(ConstExpr { value: 200.0 }),
                p: Diffable {
                    expr: projected(),
                    grad: HashMap::new(),
                    // p IS projected → ∂p/∂projected = 1.
                    proj_grad: Some(DerivEntry::Grad(Expr::Const(ConstExpr { value: 1.0 }))),
                },
            });
        let compiled = Arc::new(CompiledModel::new(model).unwrap());

        let n = compiled.param_index.len();
        let mut truth = vec![0.0; n];
        for p in &compiled.model.parameters {
            truth[compiled.param_index[p.name.as_str()]] = p.value.resolved_value().unwrap();
        }
        let cfg = OdeConfig { t_start: compiled.model.simulation.t_start, t_end, dt };
        let traj = crate::ode::run_ode(&compiled, &truth, &cfg, None, None).unwrap();
        let tr = |name: &str| {
            compiled.model.transitions.iter().position(|t| t.name == name).unwrap()
        };
        let (inf, rec) = (tr("infection"), tr("recovery"));
        // Data at the truth: k = round(200 · window fraction) for the ratio, a
        // count for its sibling. The likelihood is evaluated away from the
        // truth below, so the gradient is materially nonzero.
        let a = windowed_counts(&traj, &[inf], &times_a);
        let b = windowed_counts(&traj, &[rec], &times_a);
        let data_a: Vec<f64> =
            a.iter().zip(&b).map(|(&a, &b)| (200.0 * a / (a + b)).round()).collect();
        let data_b = windowed_counts(&traj, &[rec], &times_b);

        let specs: Vec<StreamSpec> = compiled
            .model
            .observations
            .iter()
            .zip([(&data_a, &times_a), (&data_b, &times_b)])
            .map(|(om, (data, times))| {
                let projection =
                    StreamProjection::from_ir(&om.projection, &compiled, &om.name).unwrap();
                StreamSpec {
                    times: StreamTimes::contiguous_for(
                        &projection, compiled.model.simulation.t_start, times.to_vec(),
                    ).unwrap(),
                    projection,
                    ir_model: om.clone(),
                    observations: dense_cells(data.clone()),
                    aux: vec![],
                }
            })
            .collect();
        let obs_model = MultiStreamObsModel::new(
            BoundObs::bind(compiled.model.simulation.t_start, specs).unwrap().0,
            compiled.clone(),
        )
        .unwrap();
        assert_eq!(obs_model.n_interval_streams(), 3, "two bins for the ratio, one for its sibling");
        assert_eq!(
            obs_model.incidence_streams(),
            vec![
                ("cases_a_numerator".to_string(), vec![inf]),
                ("cases_a_denominator".to_string(), vec![inf, rec]),
                ("cases_b".to_string(), vec![rec]),
            ],
            "the ratio's two bins are named, numerator first, and the sibling follows"
        );

        let mut union: Vec<f64> = times_a.iter().chain(times_b.iter()).copied().collect();
        union.sort_by(|a, b| a.partial_cmp(b).unwrap());
        union.dedup();
        let mut params = truth.clone();
        params[compiled.param_index["beta"]] = 0.66;
        params[compiled.param_index["gamma"]] = 0.09;
        let est = vec![compiled.param_index["beta"], compiled.param_index["gamma"]];

        let (ll_grad, grad) =
            det_grad(&compiled, &obs_model, &union, dt, dt, &params, &est).unwrap();
        let ll_value =
            compute_ode_loglik(&compiled, &obs_model, &union, dt, &params, dt).unwrap();
        assert!(ll_grad.is_finite() && ll_value.is_finite());
        assert!(
            (ll_grad - ll_value).abs() < 1e-6,
            "gradient-path loglik {ll_grad} disagrees with value-path loglik {ll_value} \
             (Δ = {:.6e}) — the ratio must be read from the same two bins on both paths",
            ll_grad - ll_value
        );

        let eps = 1e-6;
        let names = ["beta", "gamma"];
        for (i, &midx) in est.iter().enumerate() {
            let mut pp = params.clone();
            let mut pm = params.clone();
            pp[midx] += eps;
            pm[midx] -= eps;
            let llp = compute_ode_loglik(&compiled, &obs_model, &union, dt, &pp, dt).unwrap();
            let llm = compute_ode_loglik(&compiled, &obs_model, &union, dt, &pm, dt).unwrap();
            let fd = (llp - llm) / (2.0 * eps);
            let rel = (grad[i] - fd).abs() / fd.abs().max(1e-8);
            assert!(
                rel < 1e-4,
                "det_grad ∂/∂{} = {} vs value-path FD {} (rel err {:.2e})",
                names[i], grad[i], fd, rel
            );
        }
        // Non-vacuity: both directions must carry real signal.
        assert!(grad[0].abs() > 1.0, "∂/∂beta should be materially nonzero, got {}", grad[0]);
        assert!(grad[1].abs() > 1.0, "∂/∂gamma should be materially nonzero, got {}", grad[1]);
    }

    /// gh#952: the ODE forward sensitivities of a model with a power of a
    /// compartment that starts at zero. `sir_pow_zero_base`'s infection rate
    /// contains `R^alpha` with `R(0) = 0`, so the state Jacobian `∂rate/∂R` and
    /// `∂rate/∂alpha` are evaluated at a zero base on the first step. The
    /// textbook power rule emits `R^alpha·(alpha·(1/R))` and `R^alpha·log(R)`
    /// there — NaN at runtime — and the NaN reaches every parameter's
    /// sensitivity column through the Jacobian. Each sensitivity must be finite
    /// and match a central finite difference of the solved state.
    #[test]
    fn ode_sensitivity_finite_and_matches_fd_at_a_zero_base_power() {
        let manifest = std::env::var("CARGO_MANIFEST_DIR").unwrap();
        let path = std::path::PathBuf::from(&manifest)
            .join("../../../tests/fixtures/gradient/ir/sir_pow_zero_base.ir.json");
        let mut model: ir::Model = ir::from_str(
            &std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("read: {e}")),
        )
        .unwrap_or_else(|e| panic!("parse: {e}"));
        for p in &mut model.parameters {
            p.value = p.value.with_value(match p.name.as_str() {
                "beta" => 0.5, "gamma" => 0.1, "alpha" => 1.5,
                other => panic!("unexpected parameter {other}"),
            });
        }
        let compiled = CompiledModel::new(model).unwrap();

        let names = ["beta", "gamma", "alpha"];
        let est: Vec<usize> = names.iter().map(|n| compiled.param_index[*n]).collect();
        let mut params = vec![0.0; compiled.param_index.len()];
        for p in &compiled.model.parameters {
            params[compiled.param_index[p.name.as_str()]] = p.value.resolved_value().unwrap();
        }
        let r = compiled.model.compartments.iter().position(|c| c.name == "R").unwrap();
        assert_eq!(compiled.initial_state_mean(&params).unwrap().0.counts[r], 0,
            "non-vacuity: R must start at exactly zero");

        // R^alpha grows superlinearly, so a short horizon and a fine step keep
        // the solve away from the clamp guard (a different refusal).
        let dt = 0.1;
        let cfg = OdeConfig { t_start: 0.0, t_end: 10.0, dt };
        let obs_times: Vec<f64> = (1..=10).map(|t| t as f64).collect();
        let d = est.len();
        let seed = vec![0.0; compiled.model.compartments.len() * d];
        let solve = |p: &[f64]| {
            crate::ode::integrate_obs_sensitivity(&compiled, p, &est, &seed, &cfg, &obs_times, dt)
                .unwrap()
        };

        let recs = solve(&params);
        for rec in &recs {
            assert!(rec.state_sens.iter().chain(&rec.inc_sens).all(|s| s.is_finite()),
                "t = {}: a sensitivity is non-finite: state_sens {:?}", rec.t, rec.state_sens);
        }

        // ∂R(t_end)/∂θ against a central FD of the solved state.
        let last = recs.len() - 1;
        for (k, (&midx, name)) in est.iter().zip(names).enumerate() {
            let eps = 1e-6 * params[midx].abs();
            let mut pp = params.clone();
            let mut pm = params.clone();
            pp[midx] += eps;
            pm[midx] -= eps;
            let fd = (solve(&pp)[last].counts[r] - solve(&pm)[last].counts[r]) / (2.0 * eps);
            let got = recs[last].state_sens[r * d + k];
            let rel = (got - fd).abs() / fd.abs().max(1e-8);
            eprintln!("  ∂R(10)/∂{name} = {got:.6} vs FD {fd:.6} (rel err {rel:.2e})");
            assert!(rel < 1e-4, "∂R(10)/∂{name} = {got} vs FD {fd} (rel err {rel:.2e})");
        }
    }

    // ── gh#275 §1g: scheduled effects — the event-jump sensitivity ─────────────
    //
    // The ODE gradient path applies every effect the run fires together with its
    // sensitivity jump `S⁺ = ∂Φ/∂x·S⁻ + ∂Φ/∂θ` (`effects::SensBlock`) and splits
    // the RK4 step exactly at each effect instant. The oracle has two halves:
    //
    //  1. The gradient path integrates the SAME discrete computation as the value
    //     path (`run_ode`, which `compute_ode_loglik` scores): the incidence it
    //     records over every observation interval is bit-identical to the sum of
    //     `run_ode`'s snapshot flows over that interval, and its continuous state
    //     rounds to the snapshot state. The two logliks then agree to 1e-9
    //     relative — not bit for bit, because a stream on a longer cadence sums
    //     its bins in a different association on the two paths (gh#680).
    //  2. det_grad's GRADIENT matches a central finite difference OF THE VALUE
    //     PATH for every estimated parameter, at a point where effects fall
    //     mid-step (dt = 1, instants at .5) and at t_start (the origin batch).
    //
    // An FD of det_grad's own value could not see a mis-applied jump (a wrong
    // jump is still smooth in θ, and S would be its own exact derivative); the
    // value path is the independent reference.

    use ir::intervention::{
        Action, AddAction, FireSource, FractionTransfer, Intervention, InterventionKind,
        InterventionSchedule, SetAction,
    };

    fn fixed_param(name: &str, value: f64) -> ir::parameter::Parameter {
        ir::parameter::Parameter {
            name: name.into(),
            value: ir::parameter::ParamValue::Fixed { value },
            param_kind: None,
            param_dim: None,
        }
    }

    fn event(name: &str, times: Vec<f64>, actions: Vec<Action>) -> Intervention {
        Intervention {
            name: name.into(),
            base_name: None,
            fire: FireSource::Scheduled(InterventionSchedule::AtTimes(times)),
            actions,
            kind: InterventionKind::Event,
        }
    }

    fn transfer(src: &str, dst: &str, fraction: Expr, grad: &[(&str, Expr)]) -> Action {
        Action::FractionTransfer(FractionTransfer {
            src: src.into(),
            dst: dst.into(),
            fraction,
            fraction_grad: grad.iter().map(|(p, e)| (p.to_string(), DerivEntry::Grad(e.clone()))).collect(),
        })
    }

    /// Two incidence streams (`infection` weekly, `recovery` fortnightly) over a
    /// 56-day SEIR carrying `interventions` and `extra_params`. The data are
    /// generated by the VALUE path (`run_ode`) at the default θ, so they carry the
    /// effects; the likelihood is then evaluated at `theta` (name → value, the
    /// rest at their defaults) so the gradient is materially nonzero, and
    /// differentiated in `est_names`.
    /// Bind `(data, times)` per stream (in `model.observations` order) to
    /// `compiled` — so the same data can be scored under a second compiled model.
    fn bind_obs(compiled: &Arc<CompiledModel>, data: &[(Vec<f64>, Vec<f64>)]) -> MultiStreamObsModel {
        let t_start = compiled.model.simulation.t_start;
        let specs: Vec<StreamSpec> = compiled
            .model
            .observations
            .iter()
            .zip(data)
            .map(|(om, (values, times))| {
                let projection = StreamProjection::from_ir(&om.projection, compiled, &om.name).unwrap();
                StreamSpec {
                    times: StreamTimes::contiguous_for(&projection, t_start, times.clone()).unwrap(),
                    projection,
                    ir_model: om.clone(),
                    observations: dense_cells(values.clone()),
                    aux: vec![],
                }
            })
            .collect();
        MultiStreamObsModel::new(BoundObs::bind(t_start, specs).unwrap().0, compiled.clone()).unwrap()
    }

    #[allow(clippy::type_complexity)]
    fn event_fixture(
        interventions: Vec<Intervention>,
        extra_params: Vec<ir::parameter::Parameter>,
        theta: &[(&str, f64)],
        est_names: &[&str],
    ) -> (Arc<CompiledModel>, MultiStreamObsModel, Vec<f64>, Vec<f64>, Vec<usize>, Vec<(Vec<f64>, Vec<f64>)>) {
        let dt = 1.0;
        let times_a: Vec<f64> = (1..=8).map(|w| (w * 7) as f64).collect();
        let times_b: Vec<f64> = (1..=4).map(|w| (w * 14) as f64).collect();
        let mut model = two_incidence_stream_model(56.0, [&["infection"], &["recovery"]]);
        model.parameters.extend(extra_params);
        model.interventions = interventions;
        set_defaults(&mut model);
        let compiled = Arc::new(CompiledModel::new(model).unwrap());

        let n = compiled.param_index.len();
        let mut truth = vec![0.0; n];
        for p in &compiled.model.parameters {
            truth[compiled.param_index[p.name.as_str()]] = p.value.resolved_value().unwrap();
        }
        let t_start = compiled.model.simulation.t_start;
        let cfg = OdeConfig { t_start, t_end: 56.0, dt };
        let traj = crate::ode::run_ode(&compiled, &truth, &cfg, None, None).unwrap();
        let tr = |name: &str| compiled.model.transitions.iter().position(|t| t.name == name).unwrap();
        let data_a = windowed_counts(&traj, &[tr("infection")], &times_a);
        let data_b = windowed_counts(&traj, &[tr("recovery")], &times_b);
        assert!(data_a.iter().sum::<f64>() > 100.0, "cases_a data must be substantial");
        assert!(data_b.iter().sum::<f64>() > 10.0, "cases_b data must be substantial");
        let data = vec![(data_a, times_a.clone()), (data_b, times_b.clone())];
        let obs_model = bind_obs(&compiled, &data);

        let mut union: Vec<f64> = times_a.iter().chain(times_b.iter()).copied().collect();
        union.sort_by(|a, b| a.partial_cmp(b).unwrap());
        union.dedup();

        let mut params = truth;
        for (name, v) in theta {
            params[compiled.param_index[*name]] = *v;
        }
        let est = est_names.iter().map(|n| compiled.param_index[*n]).collect();
        (compiled, obs_model, union, params, est, data)
    }

    /// The two halves of the oracle. Returns `(name, analytic, fd, rel)` per
    /// estimated parameter, printed as a table.
    fn event_oracle(
        label: &str,
        compiled: &CompiledModel,
        obs_model: &MultiStreamObsModel,
        union: &[f64],
        params: &[f64],
        est: &[usize],
        names: &[&str],
    ) -> Vec<(String, f64, f64, f64)> {
        let dt = 1.0;
        // Half 1: the recorded state and incidence against `run_ode`'s snapshots.
        let cfg = OdeConfig { t_start: compiled.model.simulation.t_start, t_end: compiled.model.simulation.t_end, dt };
        let seed0 = compiled.ic_grad_seed(params, est).unwrap();
        let recs = crate::ode::integrate_obs_sensitivity(compiled, params, est, &seed0, &cfg, union, dt).unwrap();
        let traj = crate::ode::run_ode(compiled, params, &cfg, None, None).unwrap();
        let n_tr = compiled.model.transitions.len();
        let mut prev_t = cfg.t_start;
        for rec in &recs {
            let snap = traj.snapshots.iter().find(|s| (s.t - rec.t).abs() < 1e-9)
                .unwrap_or_else(|| panic!("{label}: run_ode has no snapshot at t = {}", rec.t));
            for (i, &c) in rec.counts.iter().enumerate() {
                let v = snap.int_state.counts[i] as f64;
                assert!((c - v).abs() <= 0.5 + 1e-9,
                    "{label}: t = {}: continuous count {c} does not round to the value path's {v} (compartment {i})", rec.t);
            }
            // The flow since the previous union obs: the sequential sum of the
            // snapshot flows in (prev_t, t], in the same association the recorder
            // accumulates — bit for bit.
            for tr in 0..n_tr {
                let mut acc = 0.0;
                for s in &traj.snapshots {
                    if s.t > prev_t + 1e-9 && s.t <= rec.t + 1e-9 {
                        acc += s.flows.as_real()[tr];
                    }
                }
                assert_eq!(rec.inc[tr].to_bits(), acc.to_bits(),
                    "{label}: t = {}: recorded incidence {:.17e} of transition {tr} is not the value \
                     path's {acc:.17e} bit for bit — the two paths must apply the same effects to \
                     the same discrete computation", rec.t, rec.inc[tr]);
            }
            prev_t = rec.t;
        }

        let (ll_grad, grad) = det_grad(compiled, obs_model, union, dt, dt, params, est).unwrap();
        let ll_value = compute_ode_loglik(compiled, obs_model, union, dt, params, dt).unwrap();
        assert!(ll_grad.is_finite() && ll_value.is_finite());
        assert!(
            (ll_grad - ll_value).abs() <= 1e-9 * ll_value.abs(),
            "{label}: the gradient path's loglik {ll_grad:.17e} is not the value path's \
             {ll_value:.17e} (Δ = {:.3e})", ll_grad - ll_value
        );
        let mut rows = Vec::new();
        eprintln!("{label}: ll = {ll_grad:.6}");
        eprintln!("  {:<8} {:>18} {:>18} {:>10}", "param", "analytic", "value-path FD", "rel err");
        for (i, &midx) in est.iter().enumerate() {
            let eps = 1e-6 * params[midx].abs().max(1.0);
            let mut pp = params.to_vec();
            let mut pm = params.to_vec();
            pp[midx] += eps;
            pm[midx] -= eps;
            let llp = compute_ode_loglik(compiled, obs_model, union, dt, &pp, dt).unwrap();
            let llm = compute_ode_loglik(compiled, obs_model, union, dt, &pm, dt).unwrap();
            let fd = (llp - llm) / (2.0 * eps);
            let rel = (grad[i] - fd).abs() / fd.abs().max(1e-8);
            eprintln!("  {:<8} {:>18.8} {:>18.8} {:>10.2e}", names[i], grad[i], fd, rel);
            rows.push((names[i].to_string(), grad[i], fd, rel));
        }
        rows
    }

    /// Fixed-fraction transfers at fixed times — the yearly ageing / school-entry
    /// transfer of an age-structured model — at mid-step instants and at t_start
    /// (the origin batch). `∂Φ/∂θ = 0` here, so this isolates the `∂Φ/∂x·S⁻`
    /// half of the jump: `S[S,:] *= (1 − f)`, `S[R,:] += f·S[S,:]`, and the step
    /// split at each instant.
    #[test]
    fn det_grad_through_fixed_fraction_transfers_matches_the_value_path_and_its_fd() {
        let ageing = |times: Vec<f64>| {
            event("ageing", times, vec![transfer("S", "R", Expr::const_(0.15), &[])])
        };
        let theta = [("beta", 0.66), ("gamma", 0.09)];
        let names = ["beta", "gamma"];
        let (compiled, obs_model, union, params, est, data) =
            event_fixture(vec![ageing(vec![0.0, 10.5, 24.5, 38.5])], vec![], &theta, &names);
        let rows = event_oracle("fixed-fraction transfers", &compiled, &obs_model, &union, &params, &est, &names);
        for (name, got, fd, rel) in &rows {
            assert!(rel < &1e-4, "∂/∂{name} = {got} vs value-path FD {fd} (rel err {rel:.2e})");
            assert!(got.abs() > 1.0, "∂/∂{name} should be materially nonzero, got {got}");
        }

        // Non-vacuity: the events materially change the likelihood of the SAME
        // data — otherwise the agreement above is about a model whose effects do
        // nothing. The ageing-free model scores the event-generated data elsewhere.
        let (ll_with, _) = det_grad(&compiled, &obs_model, &union, 1.0, 1.0, &params, &est).unwrap();
        let (quiet, _, _, _, _, _) = event_fixture(vec![], vec![], &theta, &names);
        let quiet_obs = bind_obs(&quiet, &data);
        let (ll_without, _) = det_grad(&quiet, &quiet_obs, &union, 1.0, 1.0, &params, &est).unwrap();
        assert!(
            (ll_with - ll_without).abs() > 1.0,
            "the transfers must move the likelihood of the same data: with {ll_with}, without {ll_without}"
        );
    }

    /// Parameterised amounts, every action kind, and coincident drains: an
    /// `add(I, seed)` import at t_start and mid-step, two same-instant transfers
    /// into `R` (one with the product fraction `cov·eff`, exercising a non-constant
    /// `∂f/∂θ` and the ascending-delta combination of a compartment two actions
    /// touch), and a `set E = e_set`. Differentiated in the rate parameters AND
    /// the four amount parameters — the `∂Φ/∂θ` half of the jump.
    #[test]
    fn det_grad_through_parameterised_amount_events_matches_the_value_path_and_its_fd() {
        let extra = vec![
            fixed_param("seed", 5.0),
            fixed_param("cov", 0.3),
            fixed_param("eff", 0.8),
            fixed_param("e_set", 20.0),
        ];
        let interventions = vec![
            event("import", vec![0.0, 14.0, 30.5], vec![Action::Add(AddAction {
                compartment: "I".into(),
                count: Expr::param("seed"),
                count_grad: HashMap::from([("seed".to_string(), DerivEntry::Grad(Expr::const_(1.0)))]),
            })]),
            event("sia", vec![21.0, 35.5], vec![transfer(
                "S", "R",
                Expr::bin_op(BinOp::Mul, Expr::param("cov"), Expr::param("eff")),
                &[("cov", Expr::param("eff")), ("eff", Expr::param("cov"))],
            )]),
            // Coincident with `sia` at 21.0: a second drain into R in the same phase.
            event("clear", vec![21.0], vec![transfer("E", "R", Expr::const_(0.2), &[])]),
            event("reset", vec![28.5], vec![Action::Set(SetAction {
                compartment: "E".into(),
                value: Expr::param("e_set"),
                value_grad: HashMap::from([("e_set".to_string(), DerivEntry::Grad(Expr::const_(1.0)))]),
            })]),
        ];
        let theta = [("beta", 0.66), ("gamma", 0.09), ("seed", 6.0), ("cov", 0.25), ("eff", 0.7), ("e_set", 25.0)];
        let names = ["beta", "gamma", "seed", "cov", "eff", "e_set"];
        let (compiled, obs_model, union, params, est, _) = event_fixture(interventions, extra, &theta, &names);
        let rows = event_oracle("parameterised amounts", &compiled, &obs_model, &union, &params, &est, &names);
        for (name, got, fd, rel) in &rows {
            assert!(rel < &1e-4, "∂/∂{name} = {got} vs value-path FD {fd} (rel err {rel:.2e})");
            assert!(got.abs() > 1e-2, "∂/∂{name} should be materially nonzero, got {got}");
        }
    }

    /// Negative control: the oracle has teeth. With the import's `count_grad`
    /// deliberately wrong (½ instead of 1) the gate still admits the model (a
    /// `Grad` entry is present), det_grad's value is still the value path's, and
    /// the FD exposes the wrong jump source term in `∂/∂seed` — and only there.
    #[test]
    fn a_wrong_amount_gradient_is_caught_by_the_value_path_fd() {
        let interventions = vec![event("import", vec![0.0, 14.0, 30.5], vec![Action::Add(AddAction {
            compartment: "I".into(),
            count: Expr::param("seed"),
            count_grad: HashMap::from([("seed".to_string(), DerivEntry::Grad(Expr::const_(0.5)))]),
        })])];
        let theta = [("beta", 0.66), ("seed", 6.0)];
        let names = ["beta", "seed"];
        let (compiled, obs_model, union, params, est, _) =
            event_fixture(interventions, vec![fixed_param("seed", 5.0)], &theta, &names);
        let rows = event_oracle("wrong count_grad", &compiled, &obs_model, &union, &params, &est, &names);
        assert!(rows[0].3 < 1e-4, "∂/∂beta is unaffected by the import's amount gradient: {:?}", rows[0]);
        assert!(
            rows[1].3 > 0.3,
            "a halved ∂count/∂seed must show as a ~50% error in ∂/∂seed, got rel err {:.2e}",
            rows[1].3
        );
    }

    /// A null jump at an existing boundary leaves value and gradient bit-identical
    /// to the effect-free model: `transfer(fraction = 0)` at an output time adds a
    /// boundary the walk already stops at, moves no state (`x += −0`) and no
    /// sensitivity (`0·S`). The jump machinery itself perturbs nothing.
    #[test]
    fn an_inert_event_at_an_output_boundary_leaves_det_grad_bit_identical() {
        let theta = [("beta", 0.66), ("gamma", 0.09)];
        let names = ["beta", "gamma"];
        let (quiet, quiet_obs, union, params, est, data) = event_fixture(vec![], vec![], &theta, &names);
        let inert = vec![event("noop", vec![7.0, 21.0], vec![transfer("S", "R", Expr::const_(0.0), &[])])];
        let (loud, _, _, _, _, _) = event_fixture(inert, vec![], &theta, &names);
        assert!(
            !crate::effect_plan::EffectPlan::build(
                &loud, &params, 0.0, 56.0,
                crate::effect_plan::FireClock::Exact, crate::effect_plan::Origin::Init,
            ).unwrap().is_empty(),
            "the inert event must actually fire"
        );
        let loud_obs = bind_obs(&loud, &data);
        let (ll_q, g_q) = det_grad(&quiet, &quiet_obs, &union, 1.0, 1.0, &params, &est).unwrap();
        let (ll_l, g_l) = det_grad(&loud, &loud_obs, &union, 1.0, 1.0, &params, &est).unwrap();
        assert_eq!(ll_q.to_bits(), ll_l.to_bits(), "loglik moved: {ll_q:.17e} vs {ll_l:.17e}");
        for (i, (a, b)) in g_q.iter().zip(&g_l).enumerate() {
            assert_eq!(a.to_bits(), b.to_bits(), "∂/∂{} moved: {a:.17e} vs {b:.17e}", names[i]);
        }
    }
}
