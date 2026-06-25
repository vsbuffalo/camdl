//! Finite-difference regression test for the gamma-multiplier-density
//! gradient term in `complete_data_loglik_grad` (gh#20).
//!
//! Simulates an SIR-with-overdispersion trajectory at fixed σ², then
//! compares the analytic gradient (which now includes the
//! d/dθ log Γ(g; dt/σ², σ²/dt) chain rule through `log_gamma_density_grad_substep`)
//! against a central finite difference of the complete-data log-likelihood.
//! The acceptance bar is 1e-4 relative — the same bar the per-distribution
//! gradient helpers in `obs_loglik.rs` are tested at.
//!
//! These tests bypass `pgas::run_pgas`'s preflight gate (which still
//! blocks obs-likelihood params with NUTS until gh#76 lands) by calling
//! `complete_data_loglik_grad` directly — the same function NUTS would
//! invoke. σ² estimation with PGAS+NUTS is unblocked after this commit.

use std::sync::Arc;
use sim::compiled_model::CompiledModel;
use sim::inference::pgas::{IVPMapping, simulate_reference, complete_data_loglik, build_obs_at_substep};
use sim::inference::pgas_grad::complete_data_loglik_grad;
use sim::inference::MultiStreamObsModel;
use sim::inference::particle_filter::Observation;
use sim::rng::StatefulRng;

fn load_model(path: &str) -> ir::Model {
    let json = std::fs::read_to_string(path)
        .unwrap_or_else(|e| panic!("cannot read {}: {}", path, e));
    ir::from_str(&json)
        .unwrap_or_else(|e| panic!("cannot parse {}: {}", path, e))
}

fn set_param_defaults(model: &mut ir::Model, defaults: &[(&str, f64)]) {
    for p in &mut model.parameters {
        if p.value.resolved_value().is_none() {
            if let Some(&(_, v)) = defaults.iter().find(|(n, _)| *n == p.name) {
                p.value = p.value.with_value(v);
            } else {
                p.value = p.value.with_value(0.5);
            }
        }
    }
}

fn build_params_and_names(compiled: &CompiledModel) -> (Vec<f64>, Vec<String>) {
    let n = compiled.param_index.len();
    let mut params = vec![0.0; n];
    for p in &compiled.model.parameters {
        if let Some(v) = p.value.resolved_value() {
            params[compiled.param_index[p.name.as_str()]] = v;
        }
    }
    let mut names = vec![String::new(); n];
    for p in &compiled.model.parameters {
        names[compiled.param_index[p.name.as_str()]] = p.name.clone();
    }
    (params, names)
}

fn fd_check(
    compiled: &Arc<CompiledModel>,
    trajectory: &sim::inference::pgas::PGASTrajectory,
    observations: &[Observation],
    obs_model: &MultiStreamObsModel,
    params: &[f64],
    param_names: &[String],
    estimated_indices: &[usize],
    params_to_check: &[usize],
    dt: f64,
    rel_tol: f64,
    name: &str,
) {
    let d = estimated_indices.len();
    let n_model_params = compiled.model.parameters.len();
    let mut model_to_estimated: Vec<Option<usize>> = vec![None; n_model_params];
    for (est_idx, &model_idx) in estimated_indices.iter().enumerate() {
        model_to_estimated[model_idx] = Some(est_idx);
    }
    let rate_grads_for_run = sim::inference::pgas_grad::resolve_rate_grad_for_run(
        &compiled.resolved.rate_grads_indexed,
        &model_to_estimated,
    );
    let ivp_mappings: Vec<IVPMapping> = vec![];
    let oas = build_obs_at_substep(observations, compiled.model.simulation.t_start, dt).unwrap();
    let estimated_to_model: Vec<usize> = estimated_indices.to_vec();

    let (ll, grad) = complete_data_loglik_grad(
        compiled, trajectory, params, observations, dt,
        obs_model, &ivp_mappings,
        d, &rate_grads_for_run, &oas,
        &estimated_to_model,
    ).unwrap();
    assert!(ll.is_finite(), "[{}] log-likelihood must be finite, got {}", name, ll);
    eprintln!("[{}] LL = {:.4}", name, ll);

    for &est_idx in params_to_check {
        let model_idx = estimated_indices[est_idx];
        let p_val = params[model_idx];
        let eps = (1e-5 * p_val.abs()).max(1e-8);

        let mut p_plus = params.to_vec();
        let mut p_minus = params.to_vec();
        p_plus[model_idx] += eps;
        p_minus[model_idx] -= eps;

        let ll_plus = complete_data_loglik(
            compiled, trajectory, &p_plus, observations, dt,
            obs_model, &ivp_mappings, &oas,
        ).unwrap().total;
        let ll_minus = complete_data_loglik(
            compiled, trajectory, &p_minus, observations, dt,
            obs_model, &ivp_mappings, &oas,
        ).unwrap().total;
        let fd = (ll_plus - ll_minus) / (2.0 * eps);

        let analytic = grad[est_idx];
        let rel_err = if fd.abs() > 1e-10 {
            (analytic - fd).abs() / fd.abs()
        } else {
            (analytic - fd).abs()
        };

        eprintln!(
            "[{}] d(ll)/d({:12}) = {:14.6e} (analytic) vs {:14.6e} (fd), rel_err = {:.2e}",
            name, param_names[model_idx], analytic, fd, rel_err
        );

        assert!(
            rel_err < rel_tol,
            "[{}] gradient mismatch for {}: analytic={:.6e}, fd={:.6e}, rel_err={:.2e} (tol={:.0e})",
            name, param_names[model_idx], analytic, fd, rel_err, rel_tol
        );
    }
}

fn run_gh20_check(sigma_se: f64, seed: u64, dt: f64) {
    let mut model = load_model("../../../ocaml/golden/sir_overdispersion.ir.json");
    set_param_defaults(&mut model, &[
        ("beta", 0.3), ("gamma", 0.1),
        ("sigma_se", sigma_se),
        ("N0", 1000.0), ("I0", 10.0),
    ]);
    let compiled = Arc::new(CompiledModel::new(model).unwrap());
    let (params, param_names) = build_params_and_names(&compiled);

    let t_end = compiled.model.simulation.t_end;
    let mut rng = StatefulRng::new(seed);
    let trajectory = simulate_reference(&compiled, &params, t_end, dt, &mut rng).unwrap();

    let total_gammas: usize = trajectory.substeps.iter().map(|s| s.gammas.len()).sum();
    assert!(total_gammas > 0,
        "gh#20 fixture must produce a trajectory with overdispersed gammas; got 0");

    let observations: Vec<Observation> = vec![];
    let obs_model = MultiStreamObsModel::empty(compiled.clone());

    let n_params = compiled.param_index.len();
    let estimated_indices: Vec<usize> = (0..n_params).collect();
    let sigma_idx = compiled.param_index["sigma_se"];
    let beta_idx = compiled.param_index["beta"];
    let gamma_idx = compiled.param_index["gamma"];

    fd_check(
        &compiled, &trajectory, &observations, &obs_model,
        &params, &param_names, &estimated_indices,
        // sigma_se is the new term added by gh#20.
        // beta and gamma are regression checks — gh#20 must not break the
        // existing rate-density gradient.
        &[sigma_idx, beta_idx, gamma_idx],
        dt, 1e-4,
        &format!("gh20_sigma_se={}_seed={}", sigma_se, seed),
    );
}

#[test]
fn gh20_gamma_density_grad_matches_fd_small_sigma() {
    // Small σ²: shape = dt/σ² is large, gammas tightly concentrated near 1.
    run_gh20_check(0.01, 42, 1.0);
}

#[test]
fn gh20_gamma_density_grad_matches_fd_medium_sigma() {
    run_gh20_check(0.1, 43, 1.0);
}

#[test]
fn gh20_gamma_density_grad_matches_fd_large_sigma() {
    // Large σ²: shape ≈ 1, broader gamma distribution — exercises the
    // digamma + ln(g) terms in a regime where each contribution is non-trivial.
    run_gh20_check(1.0, 44, 1.0);
}

/// Stage-3 magnitude gate (exact-PGAS oracle, magnitude arm).
///
/// The gamma-multiplier density `shape = dt/σ²`, `scale = σ²/dt` (and the
/// binomial `p = 1 - exp(-rate·dt)`) consume `dt` as a MAGNITUDE — the dominant
/// exact-PGAS exposure: a shortened substep (`dt_substep ≈ 0.9 ≠ 1.0`) is an
/// O(10%) change to these terms, far larger than the slowly-varying time path
/// the seasonal gate covers. Every `dt=1.0` test above leaves this partly
/// vacuous (`shape = 1/σ²`, `p = 1-exp(-rate)`), so a `dt`-magnitude site that
/// silently used `1.0` instead of the realized `dt_substep` would pass them.
///
/// A uniform NON-UNIT `dt` makes the magnitude bite while keeping the snap
/// invariant intact (`rec.dt_substep == dt` holds for a uniform grid, so the
/// 2b consumer asserts pass). It exercises exactly the per-substep density and
/// gradient math that exact-PGAS feeds a shortened `rec.dt_substep` into — the
/// per-substep math sees one `dt_substep` at a time and is blind to whether
/// neighbours differ, so uniform-non-unit fully validates the magnitude path.
/// `dt = 0.9125` is the proposal's named shortened-substep value.
///
/// FD-vs-analytic at 1e-4, on σ_se (the new gamma term) plus beta/gamma
/// (rate-density regression), across σ regimes — the same coverage as the
/// `dt=1.0` battery, now at `dt≠1`.
#[test]
fn stage3_gamma_grad_matches_fd_shortened_dt_small_sigma() {
    run_gh20_check(0.01, 42, 0.9125);
}

#[test]
fn stage3_gamma_grad_matches_fd_shortened_dt_medium_sigma() {
    run_gh20_check(0.1, 43, 0.9125);
}

#[test]
fn stage3_gamma_grad_matches_fd_shortened_dt_large_sigma() {
    run_gh20_check(1.0, 44, 0.9125);
}

/// A second non-unit `dt` (half-step) so the magnitude gate is not pinned to a
/// single value — `shape`/`scale`/`p` all move with `dt`, and two distinct `dt`
/// values catch a term that is correct at one `dt` by coincidence.
#[test]
fn stage3_gamma_grad_matches_fd_half_dt() {
    run_gh20_check(0.1, 45, 0.5);
}

/// gh#76 cleanup, concern (D). Multi-overdispersed-transition lockstep.
///
/// The `sir_overdispersion.ir.json` fixture has ONE overdispersed
/// transition per source group — value (`pgas::complete_data_loglik`)
/// and gradient (`pgas_grad::log_gamma_density_grad_substep`) maintain
/// independent `gamma_idx` counters that walk `model.source_groups` in
/// the same order. With one overdispersed transition per group, the two
/// counters are guaranteed to track each other byte-for-byte even if
/// the per-transition advance logic drifts.
///
/// `sir_two_overdispersed.ir.json` has TWO overdispersed transitions
/// out of the same source compartment (S → I and S → V), each with its
/// own σ² parameter (`sigma_inf` and `sigma_loss`). The chain-binomial
/// substep records both gammas back-to-back in `rec.gammas`. If the
/// gradient's iteration drifts vs the value's — e.g. advances `gamma_idx`
/// once per source group rather than once per overdispersed transition,
/// or skips overdispersed transitions that the value didn't skip —
/// σ²-gradient terms get attributed to the wrong σ² and FD-vs-analytic
/// disagrees.
///
/// FD-tests both σ² gradients at multiple σ values to exercise the
/// digamma + ln(g) terms across the regimes used in practice. Acceptance
/// bar 1e-4 relative, same as the single-overdispersed test.
fn run_gh76_cleanup_two_overdisp_check(sigma_inf: f64, sigma_loss: f64, seed: u64, dt: f64) {
    let mut model = load_model("../../../ocaml/golden/sir_two_overdispersed.ir.json");
    set_param_defaults(&mut model, &[
        ("beta", 0.3), ("gamma", 0.1), ("mu", 0.05),
        ("sigma_inf", sigma_inf), ("sigma_loss", sigma_loss),
        ("N0", 1000.0), ("I0", 10.0),
    ]);
    let compiled = Arc::new(CompiledModel::new(model).unwrap());
    let (params, param_names) = build_params_and_names(&compiled);

    let t_end = compiled.model.simulation.t_end;
    let mut rng = StatefulRng::new(seed);
    let trajectory = simulate_reference(&compiled, &params, t_end, dt, &mut rng).unwrap();

    // Sanity: each substep should record exactly 2 gammas (one per
    // overdispersed transition out of S), at least when S > 0 with both
    // rates above RATE_EPSILON. At least some substeps must have 2.
    let max_gammas_in_substep: usize = trajectory.substeps.iter().map(|s| s.gammas.len()).max().unwrap_or(0);
    assert!(max_gammas_in_substep >= 2,
        "two-overdispersed fixture must produce ≥ 2 gammas in at least one substep; \
         max observed = {}. If this fails the test is degenerate (only one transition \
         is firing) and won't actually exercise the lockstep.", max_gammas_in_substep);

    let observations: Vec<Observation> = vec![];
    let obs_model = MultiStreamObsModel::empty(compiled.clone());

    let n_params = compiled.param_index.len();
    let estimated_indices: Vec<usize> = (0..n_params).collect();
    let sigma_inf_idx  = compiled.param_index["sigma_inf"];
    let sigma_loss_idx = compiled.param_index["sigma_loss"];
    let beta_idx       = compiled.param_index["beta"];
    let gamma_idx      = compiled.param_index["gamma"];
    let mu_idx         = compiled.param_index["mu"];

    fd_check(
        &compiled, &trajectory, &observations, &obs_model,
        &params, &param_names, &estimated_indices,
        // sigma_inf, sigma_loss are the two σ² gradients — the iterator
        // lockstep check. If gamma_idx drifts in the gradient, these
        // get attributed to the wrong σ² and FD disagrees.
        // beta, gamma, mu are regression checks on the rate-density gradient.
        &[sigma_inf_idx, sigma_loss_idx, beta_idx, gamma_idx, mu_idx],
        dt, 1e-4,
        &format!("gh76_two_overdisp_sigma_inf={}_sigma_loss={}_seed={}",
                 sigma_inf, sigma_loss, seed),
    );
}

#[test]
fn gh76_cleanup_two_overdisp_grad_matches_fd_small_sigma() {
    run_gh76_cleanup_two_overdisp_check(0.01, 0.01, 42, 1.0);
}

#[test]
fn gh76_cleanup_two_overdisp_grad_matches_fd_medium_sigma() {
    run_gh76_cleanup_two_overdisp_check(0.1, 0.1, 43, 1.0);
}

#[test]
fn gh76_cleanup_two_overdisp_grad_matches_fd_asymmetric_sigma() {
    // Different σ²'s — the gradient must distinguish them. If the
    // iterator drifts so σ_inf and σ_loss get swapped, this test catches
    // it (the two gradients would be flipped).
    run_gh76_cleanup_two_overdisp_check(0.05, 0.3, 44, 1.0);
}

#[test]
fn gh76_cleanup_two_overdisp_grad_matches_fd_large_sigma() {
    run_gh76_cleanup_two_overdisp_check(1.0, 1.0, 45, 1.0);
}

// ── gh#197: the value/gradient ENERGY oracle (the "spine oracle") ───────────
//
// NUTS integrates with energy = `complete_data_loglik_grad(θ).0` and force =
// its `grad`. That `.0` MUST equal `complete_data_loglik(θ).total` — the same
// scalar computed two ways (the value-only callers vs the gradient path).
// gh#197: the gradient path adds the gamma-multiplier density GRADIENT but omits
// its VALUE from `.0`, so the NUTS energy is low by Σ log Γ(g; dt/σ², σ²/dt) on
// any overdispersed model — a silently biased σ² posterior, and NUTS then
// targets a different distribution than MH-within-Gibbs / the replica-exchange
// swap (both score with `complete_data_loglik().total`, gamma-inclusive).
//
// The non-gamma terms are already identical between the two paths, so the whole
// gap is the omitted gamma value; we assert energy == value BIT-EXACTLY. RED
// before the fix (low by ~70 nats on these fixtures); GREEN after.
//
// The FD tests above do NOT catch this — they check grad == ∂(value fn), which
// holds (the gamma gradient IS present). This is the orthogonal invariant: the
// returned energy vs the true value.
fn run_spine_oracle(sigma_se: f64, seed: u64, dt: f64) {
    let mut model = load_model("../../../ocaml/golden/sir_overdispersion.ir.json");
    set_param_defaults(&mut model, &[
        ("beta", 0.3), ("gamma", 0.1),
        ("sigma_se", sigma_se),
        ("N0", 1000.0), ("I0", 10.0),
    ]);
    let compiled = Arc::new(CompiledModel::new(model).unwrap());
    let (params, _names) = build_params_and_names(&compiled);

    let t_end = compiled.model.simulation.t_end;
    let mut rng = StatefulRng::new(seed);
    let trajectory = simulate_reference(&compiled, &params, t_end, dt, &mut rng).unwrap();

    let total_gammas: usize = trajectory.substeps.iter().map(|s| s.gammas.len()).sum();
    assert!(total_gammas > 0,
        "spine-oracle fixture must produce overdispersed gammas; got 0");

    // No observations: the gamma value lives in the TRANSITION density, so the
    // oracle exposes gh#197 with an empty obs model — the obs term is then 0 on
    // both paths, isolating the gamma gap.
    let observations: Vec<Observation> = vec![];
    let obs_model = MultiStreamObsModel::empty(compiled.clone());

    let n_model_params = compiled.model.parameters.len();
    let estimated_indices: Vec<usize> = (0..compiled.param_index.len()).collect();
    let mut model_to_estimated: Vec<Option<usize>> = vec![None; n_model_params];
    for (e, &m) in estimated_indices.iter().enumerate() { model_to_estimated[m] = Some(e); }
    let rate_grads = sim::inference::pgas_grad::resolve_rate_grad_for_run(
        &compiled.resolved.rate_grads_indexed, &model_to_estimated);
    let ivp_mappings: Vec<IVPMapping> = vec![];
    let oas = build_obs_at_substep(
        &observations, compiled.model.simulation.t_start, dt).unwrap();
    let d = estimated_indices.len();

    let value = complete_data_loglik(
        &compiled, &trajectory, &params, &observations, dt,
        &obs_model, &ivp_mappings, &oas,
    ).unwrap().total;

    let (energy, _grad) = complete_data_loglik_grad(
        &compiled, &trajectory, &params, &observations, dt,
        &obs_model, &ivp_mappings, d, &rate_grads, &oas, &estimated_indices,
    ).unwrap();

    assert_eq!(
        energy.to_bits(), value.to_bits(),
        "spine oracle (gh#197): complete_data_loglik_grad(θ).0 = {energy} must equal \
         complete_data_loglik(θ).total = {value} f64-exact — the NUTS energy must be \
         the true target. gap = {:.6} nats (= omitted Σ gamma log-density)",
        value - energy,
    );
}

#[test]
fn spine_oracle_energy_equals_value_small_sigma() { run_spine_oracle(0.01, 42, 1.0); }

#[test]
fn spine_oracle_energy_equals_value_medium_sigma() { run_spine_oracle(0.1, 43, 1.0); }

#[test]
fn spine_oracle_energy_equals_value_large_sigma() { run_spine_oracle(1.0, 44, 1.0); }

// ── gh#200 (+ gh#3-ungrouped): deterministic source-less inflow ──────────────
//
// A DETERMINISTIC source-less inflow (`--> S @ k`) is ungrouped (no source
// group). The value fn exact-counts it (`flow == round(rate·dt)` → no density
// term); the grad path scored `poisson_logpmf(flow, rate·dt)` — a spurious term
// in the NUTS energy, and it skipped on `rate <= 0.0` not `RATE_EPSILON`. Built
// programmatically because no golden has a deterministic-draw transition.
//
// The model pairs a stochastic recovery `S --> R @ γ·S` (the grouped density
// BOTH paths compute identically — so GREEN is a non-trivial nonzero match) with
// the deterministic birth (the divergence). Spine oracle: RED by the birth's
// poisson term pre-fix, GREEN bit-exact after.
#[test]
fn spine_oracle_deterministic_inflow_not_poisson_scored() {
    use std::collections::HashMap;
    use ir::{
        expr::{Expr, ParamExpr, PopExpr, BinOp, BinOpExpr, BinOpWrap},
        model::{Compartment, CompartmentKind, InitialConditions, OutputConfig,
                OutputSchedule, RegularOutputSchedule, SimulationConfig},
        parameter::{ParamValue, Parameter},
        transition::{DrawMethod, StoichiometryEntry, Transition},
        Model,
    };

    let m = Model {
        name: "det_inflow_spine".into(),
        version: "0.3".into(),
        time_unit: "days".into(),
        description: None,
        origin: None, origin_rata_die: None,
        compartments: vec![
            Compartment { name: "S".into(), kind: CompartmentKind::Integer },
            Compartment { name: "R".into(), kind: CompartmentKind::Integer },
        ],
        transitions: vec![
            // Stochastic, source-bearing → grouped. Shared density both paths compute.
            Transition {
                name: "recovery".into(),
                stoichiometry: vec![
                    StoichiometryEntry("S".into(), -1),
                    StoichiometryEntry("R".into(), 1),
                ],
                rate: Expr::BinOp(BinOpWrap { bin_op: BinOpExpr {
                    op: BinOp::Mul,
                    left: Box::new(Expr::Param(ParamExpr { param: "gamma".into() })),
                    right: Box::new(Expr::Pop(PopExpr { pop: "S".into() })),
                }}),
                metadata: None,
                draw_method: DrawMethod::Poisson,
                rate_grad: Default::default(), lineage: None,
            },
            // Deterministic, SOURCE-LESS inflow → ungrouped. The gh#200 trigger.
            Transition {
                name: "birth".into(),
                stoichiometry: vec![StoichiometryEntry("S".into(), 1)],
                rate: Expr::Param(ParamExpr { param: "k".into() }),
                metadata: None,
                draw_method: DrawMethod::Deterministic,
                rate_grad: Default::default(), lineage: None,
            },
        ],
        ode_equations: vec![], time_functions: vec![], tables: vec![], interventions: vec![],
        observations: vec![],
        bindings: vec![],
        per_eval_bindings: vec![],
        parameters: vec![
            Parameter { name: "gamma".into(), value: ParamValue::Fixed { value: 0.1 }, param_kind: None, param_dim: None, doc: None },
            Parameter { name: "k".into(), value: ParamValue::Fixed { value: 5.0 }, param_kind: None, param_dim: None, doc: None },
        ],
        initial_conditions: InitialConditions::Explicit({
            let mut h = HashMap::new();
            h.insert("S".into(), 1000.0); h.insert("R".into(), 0.0); h
        }),
        output: OutputConfig {
            times: OutputSchedule::Regular(RegularOutputSchedule { start: 0.0, step: 1.0, end: 20.0 }),
            format: "tsv".into(), trajectory: true, observations: false,
        },
        simulation: SimulationConfig {
            t_start: 0.0, t_end: 20.0, time_semantics: "continuous".into(),
            dt: Some(1.0), rng_seed: Some(7),
            integrator: Default::default(),
        },
        presets: vec![], model_structure: None, balance: None, identity_tracked_compartments: vec![],
    };
    let compiled = Arc::new(CompiledModel::new(m).unwrap());
    let (params, _names) = build_params_and_names(&compiled);

    let mut rng = StatefulRng::new(7);
    let trajectory = simulate_reference(&compiled, &params, 20.0, 1.0, &mut rng).unwrap();

    // Sanity: the deterministic birth actually fired (else the oracle is vacuous).
    let birth_idx = compiled.model.transitions.iter()
        .position(|t| t.name == "birth").unwrap();
    let birth_flow: u64 = trajectory.substeps.iter().map(|s| s.flows[birth_idx]).sum();
    assert!(birth_flow > 0, "deterministic birth must fire (got 0 flow)");

    let observations: Vec<Observation> = vec![];
    let obs_model = MultiStreamObsModel::empty(compiled.clone());
    let estimated_indices: Vec<usize> = (0..compiled.param_index.len()).collect();
    let mut model_to_estimated: Vec<Option<usize>> = vec![None; compiled.model.parameters.len()];
    for (e, &mi) in estimated_indices.iter().enumerate() { model_to_estimated[mi] = Some(e); }
    let rate_grads = sim::inference::pgas_grad::resolve_rate_grad_for_run(
        &compiled.resolved.rate_grads_indexed, &model_to_estimated);
    let ivp_mappings: Vec<IVPMapping> = vec![];
    let oas = build_obs_at_substep(
        &observations, compiled.model.simulation.t_start, 1.0).unwrap();
    let d = estimated_indices.len();

    let value = complete_data_loglik(
        &compiled, &trajectory, &params, &observations, 1.0,
        &obs_model, &ivp_mappings, &oas,
    ).unwrap().total;
    let (energy, _grad) = complete_data_loglik_grad(
        &compiled, &trajectory, &params, &observations, 1.0,
        &obs_model, &ivp_mappings, d, &rate_grads, &oas, &estimated_indices,
    ).unwrap();

    assert_eq!(
        energy.to_bits(), value.to_bits(),
        "spine oracle (gh#200): a deterministic source-less inflow must NOT be \
         Poisson-scored on the grad path. energy = {energy}, value = {value}, \
         gap = {:.6} nats (= spurious Σ poisson_logpmf of the deterministic birth)",
        value - energy,
    );
}

/// Multi-gamma coverage for the spine oracle. `sir_two_overdispersed` has TWO
/// overdispersed transitions out of the same source → 2 gammas per substep. The
/// grad path must add the gamma values in the SAME left-fold order as the value
/// fn (`((td)+g1)+g2`); a pre-summed `(td)+(g1+g2)` differs by a ULP (f64 add is
/// non-associative), which `to_bits()` catches. The single-overdispersed
/// `spine_oracle_*` tests above (1 gamma/substep) cannot reach this case.
#[test]
fn spine_oracle_two_overdispersed_multi_gamma_bit_exact() {
    let mut model = load_model("../../../ocaml/golden/sir_two_overdispersed.ir.json");
    set_param_defaults(&mut model, &[
        ("beta", 0.3), ("gamma", 0.1), ("mu", 0.05),
        ("sigma_inf", 0.2), ("sigma_loss", 0.15), // asymmetric → two distinct gammas
        ("N0", 1000.0), ("I0", 10.0),
    ]);
    let compiled = Arc::new(CompiledModel::new(model).unwrap());
    let (params, _names) = build_params_and_names(&compiled);

    let t_end = compiled.model.simulation.t_end;
    let mut rng = StatefulRng::new(46);
    let trajectory = simulate_reference(&compiled, &params, t_end, 1.0, &mut rng).unwrap();

    // Confirm we actually hit the multi-gamma case (≥2 gammas in some substep).
    let max_g = trajectory.substeps.iter().map(|s| s.gammas.len()).max().unwrap_or(0);
    assert!(max_g >= 2,
        "fixture must produce ≥2 gammas/substep to exercise the summation-order \
         path; got max {max_g}");

    let observations: Vec<Observation> = vec![];
    let obs_model = MultiStreamObsModel::empty(compiled.clone());
    let estimated_indices: Vec<usize> = (0..compiled.param_index.len()).collect();
    let mut model_to_estimated: Vec<Option<usize>> = vec![None; compiled.model.parameters.len()];
    for (e, &mi) in estimated_indices.iter().enumerate() { model_to_estimated[mi] = Some(e); }
    let rate_grads = sim::inference::pgas_grad::resolve_rate_grad_for_run(
        &compiled.resolved.rate_grads_indexed, &model_to_estimated);
    let ivp_mappings: Vec<IVPMapping> = vec![];
    let oas = build_obs_at_substep(
        &observations, compiled.model.simulation.t_start, 1.0).unwrap();
    let d = estimated_indices.len();

    let value = complete_data_loglik(
        &compiled, &trajectory, &params, &observations, 1.0,
        &obs_model, &ivp_mappings, &oas,
    ).unwrap().total;
    let (energy, _grad) = complete_data_loglik_grad(
        &compiled, &trajectory, &params, &observations, 1.0,
        &obs_model, &ivp_mappings, d, &rate_grads, &oas, &estimated_indices,
    ).unwrap();

    assert_eq!(
        energy.to_bits(), value.to_bits(),
        "spine oracle (multi-gamma): energy = {energy}, value = {value}, \
         gap = {:.3e} nats — a non-zero gap here means the grad pre-summed the \
         per-substep gammas instead of left-folding them into log_p.",
        value - energy,
    );
}
