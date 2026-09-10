//! A/B gate for the gh#272 loop-invariant code-motion (LICM) pass — the
//! byte-identical soundness proof the pass's claim rests on.
//!
//! LICM (`ocaml/lib/ir/licm.ml`) extracts maximal param/table-only invariant
//! subexpressions (the in-model gravity kernel's `exp(-gamma_k*log(dratio))`
//! terms and its normalization sum) out of the dynamics rates into
//! `per_eval_bindings`, replacing each with a `PerEvalRef`. It claims to be
//! *trajectory-preserving* (the runtime evaluates the hoisted binding to the
//! same value the inlined subtree would). This gate makes that a test, on a
//! model where the pass actually fires.
//!
//! Two committed fixtures compiled from the SAME source (`licm_ab.camdl`, a
//! 4-patch in-model gravity kernel with a guarded FOI):
//!   - `licm_ab_off.ir.json` — `CAMDL_NO_LICM=1 camdlc` (kernel inlined)
//!   - `licm_ab_on.ir.json`  — `camdlc` (kernel hoisted; LICM is default-on)
//!
//! See the source header for the exact regeneration commands. The fixtures are
//! static IR (the test does not recompile), so the default-flag flip is
//! decoupled from this gate.
//!
//! Two assertions:
//!   1. NON-VACUITY — the ON fixture has `per_eval_bindings` and `PerEvalRef`
//!      nodes (the pass fired) and is strictly smaller; the OFF fixture has none.
//!      A green test on a no-op pass proves nothing; this guards it.
//!   2. SOUNDNESS — for every supported backend at a fixed seed, the hoisted and
//!      inlined models simulate to a byte-identical trajectory (same FNV-1a
//!      hash). This is what "trajectory-preserving" means.

use std::path::PathBuf;
use ir::expr::Expr;
use sim::{
    compiled_model::CompiledModel,
    config::{ChainBinomialConfig, GillespieConfig, OdeConfig, SimConfig},
    simulate::Simulate,
    ChainBinomialSim, GillespieSim, OdeSim,
};

const SEED: u64 = 42;

fn fixtures_dir() -> PathBuf {
    let manifest = std::env::var("CARGO_MANIFEST_DIR").unwrap();
    PathBuf::from(&manifest).join("tests/fixtures")
}

fn load(name: &str) -> (ir::Model, usize) {
    let path = fixtures_dir().join(format!("{}.ir.json", name));
    let contents = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("could not read {:?}: {}", path, e));
    let model = ir::from_str(&contents)
        .unwrap_or_else(|e| panic!("failed to parse {}: {}", name, e));
    (model, contents.len())
}

/// Count `PerEvalRef` nodes across all transition rates + rate_grads + ODE
/// derivatives (the surfaces LICM rewrites).
fn count_per_eval_refs(m: &ir::Model) -> usize {
    fn in_expr(e: &Expr) -> usize {
        match e {
            Expr::PerEvalRef(_) => 1,
            Expr::BinOp(w) => in_expr(&w.bin_op.left) + in_expr(&w.bin_op.right),
            Expr::UnOp(w) => in_expr(&w.un_op.arg),
            Expr::Cond(w) => in_expr(&w.cond.pred) + in_expr(&w.cond.then) + in_expr(&w.cond.else_),
            Expr::TableLookup(w) => w.table_lookup.indices.iter().map(in_expr).sum(),
            Expr::Reduce(w) => w.reduce.iter().map(in_expr).sum(),
            Expr::UncheckedDim(w) => in_expr(&w.unchecked_dim.inner),
            _ => 0,
        }
    }
    let mut n = 0;
    for t in &m.transitions {
        n += in_expr(&t.rate);
        for g in t.rate_grad.values() {
            if let ir::deriv::DerivEntry::Grad(e) = g {
                n += in_expr(e);
            }
        }
    }
    for eq in &m.ode_equations {
        n += in_expr(&eq.derivative);
    }
    n
}

/// FNV-1a/64 over the full trajectory numeric content — the same hash the
/// trajectory-baseline / constant-fold gates use.
fn trajectory_hash(traj: &sim::state::Trajectory) -> u64 {
    let mut h: u64 = 0xcbf29ce484222325;
    let mut mix = |bytes: &[u8]| {
        for &b in bytes {
            h ^= b as u64;
            h = h.wrapping_mul(0x100000001b3);
        }
    };
    for snap in &traj.snapshots {
        mix(&snap.t.to_bits().to_le_bytes());
        for &c in &snap.int_state.counts {
            mix(&c.to_le_bytes());
        }
        for &v in &snap.real_state.values {
            mix(&v.to_bits().to_le_bytes());
        }
        match &snap.flows {
            sim::state::Flows::Int(fs) => {
                for &f in fs {
                    mix(&f.to_le_bytes());
                }
            }
            sim::state::Flows::Real(fs) => {
                for &f in fs {
                    mix(&f.to_bits().to_le_bytes());
                }
            }
        }
    }
    h
}

#[test]
fn gate_licm_is_byte_identical() {
    sim::eval_stats::set_allow_degenerate_rates(true);

    let (off, off_bytes) = load("licm_ab_off");
    let (on, on_bytes) = load("licm_ab_on");

    // ── 1. NON-VACUITY ──────────────────────────────────────────────────────
    let off_refs = count_per_eval_refs(&off);
    let on_refs = count_per_eval_refs(&on);
    assert_eq!(
        off_refs, 0,
        "OFF fixture already has PerEvalRef nodes — regenerated with LICM on?"
    );
    assert!(
        on_refs > 0 && !on.per_eval_bindings.is_empty(),
        "LICM did not fire on the ON fixture: {on_refs} PerEvalRef nodes, \
         {} per_eval_bindings. Regenerate from licm_ab.camdl (see its header).",
        on.per_eval_bindings.len()
    );
    assert!(
        on_bytes < off_bytes,
        "LICM did not shrink the IR (off={off_bytes} bytes, on={on_bytes} bytes)"
    );
    eprintln!(
        "non-vacuity: PerEvalRef {off_refs} -> {on_refs}; \
         per_eval_bindings={}; IR {off_bytes} -> {on_bytes} bytes",
        on.per_eval_bindings.len()
    );

    // ── 2. SOUNDNESS ────────────────────────────────────────────────────────
    let compiled_off = CompiledModel::new(off.clone()).expect("OFF model failed to compile");
    let compiled_on = CompiledModel::new(on.clone()).expect("ON model failed to compile");

    let params_off = compiled_off.default_params.clone();
    let params_on = compiled_on.default_params.clone();

    let t_start = off.simulation.t_start;
    let t_end = off.simulation.t_end;
    assert_eq!(t_end, on.simulation.t_end, "fixtures disagree on t_end");

    let backends: &[(&str, SimConfig)] = &[
        ("gillespie", SimConfig::Gillespie(GillespieConfig { t_start, t_end, output_dt: None })),
        ("chain_binomial", SimConfig::ChainBinomial(ChainBinomialConfig { t_start, t_end, dt: 1.0 })),
        ("ode", SimConfig::Ode(OdeConfig { t_start, t_end, dt: 1.0 })),
    ];

    let required = compiled_off.required_capabilities();
    let mut checked = 0usize;
    for (backend, config) in backends {
        let sim: &dyn Simulate = match *backend {
            "gillespie" => &GillespieSim,
            "ode" => &OdeSim,
            _ => &ChainBinomialSim,
        };
        if !(required - sim.capabilities()).is_empty() {
            continue;
        }
        let traj_off = sim
            .run(&compiled_off, &params_off, SEED, config)
            .unwrap_or_else(|e| panic!("OFF {backend} sim failed: {e:?}"));
        let traj_on = sim
            .run(&compiled_on, &params_on, SEED, config)
            .unwrap_or_else(|e| panic!("ON {backend} sim failed: {e:?}"));

        let h_off = trajectory_hash(&traj_off);
        let h_on = trajectory_hash(&traj_on);
        assert_eq!(
            h_off, h_on,
            "TRAJECTORY DIVERGED on {backend}: LICM is NOT byte-identical \
             (off 0x{h_off:016x} != on 0x{h_on:016x}). This is a soundness bug \
             in the hoist (e.g. a variant subtree mis-classified as invariant), \
             not a golden update."
        );
        eprintln!("{backend}: byte-identical (hash 0x{h_off:016x})");
        checked += 1;
    }
    assert!(checked >= 3, "expected at least 3 backends checked, got {checked}");

    // ── 3. GRADIENT VALUE-IDENTITY ──────────────────────────────────────────
    // `rate_grad` is hoisted too (LICM rewrites it), but the forward backends
    // above never evaluate it. Evaluate every transition's rate_grad expression
    // at the initial state + default params, off vs on, and assert bitwise-equal
    // — the direct check that hoisting the gradient surface is value-preserving
    // (the gradient eval-equality gate the proposal calls for). Each context
    // points at its own model so a `PerEvalRef` resolves against the right
    // `per_eval_bindings`.
    let (int_off, real_off) = compiled_off.initial_state_mean(&params_off).expect("off init state");
    let (int_on, real_on) = compiled_on.initial_state_mean(&params_on).expect("on init state");
    let ctx_off = sim::propensity::EvalCtx {
        model: &compiled_off, int_s: &int_off, real_s: &real_off, params: &params_off,
        t: t_start, dt: 1.0, projected: None, aux: None, int_float_override: None, per_eval: None,
    };
    let ctx_on = sim::propensity::EvalCtx {
        model: &compiled_on, int_s: &int_on, real_s: &real_on, params: &params_on,
        t: t_start, dt: 1.0, projected: None, aux: None, int_float_override: None, per_eval: None,
    };
    // `rate_grad` is a HashMap in the IR, so its per-transition term ORDER is
    // non-deterministic — compare by param index (the value), not by position.
    use std::collections::HashMap;
    let mut grad_terms = 0usize;
    for (ti, (g_off, g_on)) in compiled_off.resolved.rate_grads_indexed.iter()
        .zip(compiled_on.resolved.rate_grads_indexed.iter())
        .enumerate()
    {
        let off_map: HashMap<usize, u64> = g_off.iter()
            .filter_map(|(p, entry)| match entry {
                sim::resolved_expr::ResolvedDerivEntry::Grad(e) =>
                    Some((*p, sim::resolved_expr::eval_resolved(e, &ctx_off).to_bits())),
                sim::resolved_expr::ResolvedDerivEntry::Unsupported { .. } => None,
            }).collect();
        let on_map: HashMap<usize, u64> = g_on.iter()
            .filter_map(|(p, entry)| match entry {
                sim::resolved_expr::ResolvedDerivEntry::Grad(e) =>
                    Some((*p, sim::resolved_expr::eval_resolved(e, &ctx_on).to_bits())),
                sim::resolved_expr::ResolvedDerivEntry::Unsupported { .. } => None,
            }).collect();
        assert_eq!(
            off_map.len(), on_map.len(),
            "rate_grad term count differs at transition {ti} (off={}, on={})",
            off_map.len(), on_map.len()
        );
        for (p, &v_off) in &off_map {
            let v_on = *on_map.get(p)
                .unwrap_or_else(|| panic!("param idx {p} present in OFF rate_grad but not ON, transition {ti}"));
            assert_eq!(
                v_off, v_on,
                "GRADIENT DIVERGED transition {ti} param idx {p}: off bits=0x{v_off:016x} \
                 on bits=0x{v_on:016x} — LICM hoisting of rate_grad is not value-preserving"
            );
            grad_terms += 1;
        }
    }
    assert!(grad_terms > 0, "no rate_grad terms evaluated — fixture has no gradients?");
    eprintln!("gradient: {grad_terms} rate_grad terms byte-identical off vs on");

    // ── 4. STAGED-SCRATCH A/B ───────────────────────────────────────────────
    // Design C runtime mechanism: a `PerEvalRef` reads `ctx.per_eval[slot]` when
    // the caller has staged the prologue, and falls through to on-demand eval
    // (`eval_resolved(per_eval_bindings[slot], ..)`) when it has not. Both must
    // yield the SAME value — the staged scratch is exactly what on-demand eval
    // would compute, just hoisted out of the loop. This isolates the runtime
    // read mechanism from the hoist-soundness (§2-3) and the end-to-end identity:
    // §2 already proves the ON model (run_ode stages the scratch) is byte-
    // identical to the fully-inlined OFF model, so staged-vs-inlined is covered
    // end-to-end; this section pins staged-vs-on-demand at expression granularity.
    //
    // Stage the prologue on the ON model at the initial (state, t), then for every
    // rate / rate_grad / ODE-derivative expression assert eval-with-scratch ==
    // eval-with-None, bitwise. `ctx_on` (above) is the on-demand path (per_eval:
    // None). Non-vacuity: the ON model has per-eval bindings (§1), so the scratch
    // is non-empty and the rate surface carries `PerEvalRef` nodes that exercise
    // both arms.
    let scratch = sim::resolved_expr::eval_per_eval_scratch(
        &compiled_on, &params_on, t_start, 1.0);
    assert!(
        !scratch.is_empty(),
        "ON model staged an empty per-eval scratch — nothing to A/B (pass did not fire?)"
    );
    let ctx_staged = sim::propensity::EvalCtx {
        model: &compiled_on, int_s: &int_on, real_s: &real_on, params: &params_on,
        t: t_start, dt: 1.0, projected: None, aux: None, int_float_override: None,
        per_eval: Some(&scratch),
    };
    let mut pe_checks = 0usize;
    let mut check = |e: &sim::resolved_expr::ResolvedExpr, what: &str| {
        let v_staged = sim::resolved_expr::eval_resolved(e, &ctx_staged).to_bits();
        let v_demand = sim::resolved_expr::eval_resolved(e, &ctx_on).to_bits();
        assert_eq!(
            v_staged, v_demand,
            "STAGED != ON-DEMAND ({what}): staged bits=0x{v_staged:016x} \
             demand bits=0x{v_demand:016x} — the staged scratch diverges from \
             on-demand evaluation (a slot-indexing / prefix-slicing bug)"
        );
        pe_checks += 1;
    };
    for (ti, rate) in compiled_on.resolved.rates.iter().enumerate() {
        check(rate, &format!("rate transition {ti}"));
    }
    for g in &compiled_on.resolved.rate_grads_indexed {
        for (_p, entry) in g {
            if let sim::resolved_expr::ResolvedDerivEntry::Grad(e) = entry {
                check(e, "rate_grad");
            }
        }
    }
    for d in &compiled_on.resolved.ode_derivatives {
        check(d, "ode_derivative");
    }
    assert!(pe_checks > 0, "no expressions evaluated in the staged-scratch A/B");
    eprintln!(
        "staged-scratch A/B: {} bindings staged; {pe_checks} exprs staged == on-demand",
        scratch.len()
    );
}

// ── Inference producer A/B: staged scratch == on-demand through step_one ───────
//
// Phase 2 (gh#272) threads the staged per-eval scratch through the stochastic
// inference producer. PF, IF2, PGAS, and PMMH all advance particles via
// `ProcessModel::step`, which delegates to `chain_binomial::step_one` — the one
// seam every stochastic cell shares. Stepping the hoisted (ON) model with its
// staged scratch and the inlined (OFF) model on-demand, under the SAME seed,
// must yield byte-identical particle counts AND flow accumulators at every
// substep. This is the inference-path analogue of §2's forward byte-identity:
// the proof that staging the scratch through the producer is value-preserving,
// so PGAS/IF2/PF reach fixed-kernel parity *correctly*, not just faster.
//
// A wrong-θ staging bug (the IF2 silent-wrong risk: serving one particle's
// kernel to another's θ) would change the rates, the draws, and hence the
// counts — caught here. Counts are an integer trajectory, so this is exact, not
// tolerance-based. The full PF/IF2/PGAS loglik is a deterministic function of
// these producer states plus the (per_eval-free) observation scoring, so its
// byte-identity follows from this seam's.
#[test]
fn gate_licm_inference_producer_byte_identical() {
    use std::sync::Arc;
    use sim::inference::{ChainBinomialProcess, ProcessModel};
    use sim::rng::StatefulRng;
    sim::eval_stats::set_allow_degenerate_rates(true);

    let (off, _) = load("licm_ab_off");
    let (on, _) = load("licm_ab_on");
    let compiled_off = Arc::new(CompiledModel::new(off).expect("OFF model failed to compile"));
    let compiled_on = Arc::new(CompiledModel::new(on).expect("ON model failed to compile"));

    let params_off = compiled_off.default_params.clone();
    let params_on = compiled_on.default_params.clone();
    let t_start = 0.0_f64;
    let dt = 1.0_f64;

    // The producers stage from these. Non-vacuity: ON stages a NON-EMPTY kernel
    // scratch (the producer takes the staged arm); OFF stages nothing (the
    // producer takes the on-demand arm). Without this, byte-identity is vacuous.
    let pe_off = sim::resolved_expr::stage_per_eval(&compiled_off, &params_off, t_start, dt);
    let pe_on = sim::resolved_expr::stage_per_eval(&compiled_on, &params_on, t_start, dt);
    assert!(pe_off.is_none(), "OFF fixture unexpectedly has per_eval bindings");
    assert!(
        pe_on.as_ref().is_some_and(|s| !s.is_empty()),
        "ON fixture staged an empty scratch — the LICM pass did not fire"
    );

    let proc_off = ChainBinomialProcess::new(compiled_off.clone());
    let proc_on = ChainBinomialProcess::new(compiled_on.clone());
    let mut rng_off = StatefulRng::new(SEED);
    let mut rng_on = StatefulRng::new(SEED);
    let mut s_off = proc_off.initial_state_draw(&params_off, &mut rng_off).expect("off init state");
    let mut s_on = proc_on.initial_state_draw(&params_on, &mut rng_on).expect("on init state");
    let mut scr_off = proc_off.new_scratch();
    let mut scr_on = proc_on.new_scratch();

    let n_steps = 120usize; // the fixture's sim window (120 days at dt=1)
    for s in 0..n_steps {
        let t = t_start + s as f64;
        // ON threads the staged scratch (exactly what bootstrap_filter / run_if2 /
        // csmc_as do); OFF takes the on-demand arm. The fixture has no events or
        // interventions, so `due_effects` is empty.
        proc_on
            .step(&mut s_on, &params_on, t, dt, pe_on.as_deref(), &mut rng_on, &mut scr_on, &[])
            .expect("on producer step");
        proc_off
            .step(&mut s_off, &params_off, t, dt, pe_off.as_deref(), &mut rng_off, &mut scr_off, &[])
            .expect("off producer step");
        assert_eq!(
            s_off.counts, s_on.counts,
            "PRODUCER COUNTS DIVERGED at substep {s}: LICM-on staged scratch != LICM-off \
             on-demand through ProcessModel::step (step_one) — a θ-granularity / wiring bug \
             in the Phase 2 staging, not a golden update."
        );
        assert_eq!(
            s_off.flow_accumulators, s_on.flow_accumulators,
            "PRODUCER FLOWS DIVERGED at substep {s}: counts matched but flow accumulators did \
             not — the density/incidence path would diverge."
        );
    }
    eprintln!(
        "inference producer: {n_steps} substeps byte-identical (ON staged vs OFF on-demand), \
         counts + flow accumulators"
    );
}

// ── PGAS loglik A/B: the staged scratch is value-preserving on the full ────────
// ── Bayesian path (producer + density + NUTS gradient) ────────────────────────
//
// Phase 2 stages the per-eval scratch through THREE PGAS surfaces:
//   - the CSMC producer  (simulate_reference_on_grid → step_one),
//   - the transition density (complete_data_loglik → log_transition_density_substep),
//   - the NUTS gradient  (complete_data_loglik_grad → log_transition_density_grad).
// Drive all three on the hoisted (ON) and inlined (OFF) fixtures at the same
// seed / params / grid, and assert the PGAS complete-data log-likelihood AND its
// gradient are byte-identical off vs on. This is the result-level standing gate:
// toggling `--no-licm` must not move a PGAS fit's numbers. (Measured on the real
// SLE-14 model the loglik is identical to the last decimal — −25771.0 both — at
// a 5.9× speedup; this pins that equality permanently on a self-contained
// fixture.) gh#272 Phase 2.
#[test]
fn gate_licm_pgas_loglik_byte_identical() {
    use std::sync::Arc;
    use sim::inference::pgas::{
        build_substep_grid, complete_data_loglik, simulate_reference_on_grid, ObsAtSubstep,
    };
    use sim::inference::pgas_grad::{complete_data_loglik_grad, resolve_rate_grad_for_run};
    use sim::inference::particle_filter::Observation;
    use sim::inference::MultiStreamObsModel;
    use sim::rng::StatefulRng;
    use sim::schedule::StepPolicy;
    sim::eval_stats::set_allow_degenerate_rates(true);

    let (off, _) = load("licm_ab_off");
    let (on, _) = load("licm_ab_on");
    let compiled_off = Arc::new(CompiledModel::new(off).expect("OFF model failed to compile"));
    let compiled_on = Arc::new(CompiledModel::new(on).expect("ON model failed to compile"));
    assert!(
        compiled_off.resolved.per_eval_bindings.is_empty()
            && !compiled_on.resolved.per_eval_bindings.is_empty(),
        "fixtures not in the expected LICM off/on state"
    );
    let params_off = compiled_off.default_params.clone();
    let params_on = compiled_on.default_params.clone();
    let dt = 1.0_f64;
    let t_start = 0.0_f64;

    // Build the PGAS reference trajectory, the complete-data log-likelihood, and
    // its analytic gradient on one model. Observation TIMES only tile the substep
    // grid (the obs model is `empty` — no scoring), so this isolates exactly the
    // rate / density / rate_grad surfaces LICM rewrites.
    let obs_times = [30.0_f64, 60.0, 90.0, 120.0];
    let run = |compiled: &Arc<CompiledModel>, params: &[f64]| -> (f64, f64, Vec<f64>) {
        let observations: Vec<Observation> =
            obs_times.iter().map(|&t| Observation { time: t, value: 0.0 }).collect();
        let grid =
            build_substep_grid(t_start, dt, &observations, &[], StepPolicy::Exact).unwrap();
        let mut rng = StatefulRng::new(SEED);
        let traj =
            simulate_reference_on_grid(compiled, params, dt, &grid.steps, None, &mut rng).unwrap();
        let obs_model = MultiStreamObsModel::empty(compiled.clone());
        let no_obs: Vec<Observation> = vec![];
        let no_map = ObsAtSubstep::new();
        let comps =
            complete_data_loglik(compiled, &traj, params, &no_obs, dt, &obs_model, &no_map)
                .unwrap();
        let n_params = params.len();
        let model_to_estimated: Vec<Option<usize>> = (0..n_params).map(Some).collect();
        let estimated_to_model: Vec<usize> = (0..n_params).collect();
        let rate_grads =
            resolve_rate_grad_for_run(&compiled.resolved.rate_grads_indexed, &model_to_estimated);
        let (ll, grad) = complete_data_loglik_grad(
            compiled, &traj, params, &no_obs, dt, &obs_model, n_params, &rate_grads, &no_map,
            &estimated_to_model,
        )
        .unwrap();
        (comps.transition, ll, grad)
    };

    let (td_off, ll_off, grad_off) = run(&compiled_off, &params_off);
    let (td_on, ll_on, grad_on) = run(&compiled_on, &params_on);

    assert!(td_off.is_finite() && ll_off.is_finite(), "OFF PGAS loglik must be finite");
    assert_eq!(
        td_off.to_bits(),
        td_on.to_bits(),
        "PGAS TRANSITION DENSITY DIVERGED: off={td_off} on={td_on} — staging the scratch \
         through simulate_reference_on_grid / log_transition_density_substep moved the PGAS \
         likelihood. A θ-granularity / wiring bug, NOT a golden update."
    );
    assert_eq!(
        ll_off.to_bits(),
        ll_on.to_bits(),
        "PGAS GRADIENT-PATH LL DIVERGED: off={ll_off} on={ll_on}"
    );
    assert_eq!(grad_off.len(), grad_on.len(), "gradient dimensionality differs off vs on");
    let mut nonzero = 0usize;
    for (i, (&go, &gn)) in grad_off.iter().zip(grad_on.iter()).enumerate() {
        assert_eq!(
            go.to_bits(),
            gn.to_bits(),
            "PGAS GRADIENT DIVERGED at param {i}: off={go} on={gn} — the staged scratch on the \
             rate_grad path (complete_data_loglik_grad) is not value-preserving."
        );
        if go != 0.0 {
            nonzero += 1;
        }
    }
    // Non-vacuity: the gradient surface is genuinely exercised (licm_ab carries
    // rate_grad for its fitted params), so this isn't an all-zero trivial match.
    assert!(nonzero > 0, "gradient is identically zero — the rate_grad A/B is vacuous");
    eprintln!(
        "PGAS A/B: transition-density + complete-data LL + {}-dim gradient ({nonzero} nonzero) \
         all byte-identical ON vs OFF (td={td_off}, ll={ll_off})",
        grad_off.len()
    );
}
