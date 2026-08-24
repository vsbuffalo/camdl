//! The initial-state seam's contract: mean / draw / logpdf / logpdf_grad.
//!
//! `CompiledModel::initial_state` split into the four questions an
//! initial-state law has to answer (proposal
//! `docs/dev/proposals/2026-08-23-initial-state-parameters.md`, staging step
//! 3). Two of the four — `initial_state_logpdf` and
//! `initial_state_logpdf_grad` — have no consumer until staging step 4 wires
//! PGAS's Binomial IVP term through them, so this file is what exercises them
//! in the meantime, and it pins the parts of the contract step 4 will build on.
//!
//! What is asserted here is NOT "0.0 == 0.0". It is:
//!
//! 1. **The draw does not consume randomness while no `init {}` entry declares
//!    a law.** That is the whole basis for calling step 3 value-preserving:
//!    every stochastic forward path (chain-binomial, Gillespie, the bootstrap
//!    and correlated filters, IF2, PGAS's reference walk) now hands its own
//!    stream to `initial_state_draw`, and if that call started consuming, every
//!    one of those trajectories would silently shift. When step 4 lands a law,
//!    this test should keep passing for a LAW-FREE model and the law-bearing
//!    case gets its own test — the RNG-order change must be a deliberate,
//!    visible decision, not a baseline diff someone discovers later.
//! 2. **`logpdf_grad` is indexed by MODEL parameter, length `params.len()`.**
//!    Step 4's callers (`complete_data_loglik_grad`, ODE-NUTS) work in an
//!    ESTIMATED-parameter basis and map through their own `estimated_to_model`.
//!    A length mismatch there is an out-of-bounds panic at best and a gradient
//!    attributed to the wrong parameter at worst.
//!
//! Two `init {}` shapes are covered — one whose entries are all literals and
//! one whose entries are expressions over parameters. They are the same
//! `InitSpec::Deterministic` variant, but they take different paths through the
//! producer (constant placement vs `eval_expr` against the partially built
//! state), so both are exercised.

use std::sync::Arc;

use ir::expr::Expr;
use ir::model::InitSpec;
use sim::{compiled_model::CompiledModel, rng::StatefulRng};

/// Whether every `init {}` entry is a bare literal. The predicate the expander
/// used to fold into the IR itself, before `InitialConditions` became one
/// ordered map of per-compartment specs.
fn all_entries_are_literals(model: &ir::Model) -> bool {
    model
        .initial_conditions
        .iter()
        .all(|(_, InitSpec::Deterministic(e))| matches!(e, Expr::Const(_)))
}

const SEED: u64 = 20260823;

fn load(rel: &str) -> ir::Model {
    let path = format!("{}/../../../{}", env!("CARGO_MANIFEST_DIR"), rel);
    let json = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("read {path}: {e}"));
    ir::from_str(&json).unwrap_or_else(|e| panic!("parse {path}: {e:?}"))
}

/// SEIR with `init { S = N0 - I0; I = I0 }` — entries that are expressions over
/// parameters. Required parameters carry no value in the IR, so fill them the
/// way `gradient_check.rs` does for this same fixture.
fn parameterized_model() -> (Arc<CompiledModel>, Vec<f64>) {
    let mut model = load("tests/fixtures/gradient/ir/seir_seasonal_lagged.ir.json");
    assert!(
        !all_entries_are_literals(&model),
        "fixture must exercise expression evaluation, not constant placement"
    );
    for p in &mut model.parameters {
        if p.value.resolved_value().is_none() {
            p.value = p.value.with_value(match p.name.as_str() {
                "beta" => 0.3,
                "sigma" => 0.2,
                "gamma" => 0.1,
                "alpha" => 0.15,
                "phi_season" => 90.0,
                "N0" => 1_000_000.0,
                "I0" => 10.0,
                _ => 0.5,
            });
        }
    }
    let compiled = Arc::new(CompiledModel::new(model).expect("compile parameterized fixture"));
    let params = compiled.default_params.clone();
    (compiled, params)
}

/// SIR with constant initial counts — every entry a literal.
fn explicit_model() -> (Arc<CompiledModel>, Vec<f64>) {
    let model = load("tests/fixtures/corner_cases/ir/dt_rate.ir.json");
    assert!(
        all_entries_are_literals(&model),
        "fixture must exercise constant placement"
    );
    let compiled = Arc::new(CompiledModel::new(model).expect("compile explicit fixture"));
    let params = compiled.default_params.clone();
    (compiled, params)
}

/// Non-vacuity: an all-zero initial state would make every equality below
/// trivially true.
fn assert_state_is_populated(int_counts: &[i64], who: &str) {
    assert!(
        int_counts.iter().any(|&c| c > 0),
        "{who}: initial state is all zeros — the assertions below would be vacuous"
    );
}

#[test]
fn a_parameterized_init_actually_reads_its_parameters() {
    // Guards the guard: if `N0` did not move the initial state, the
    // Parameterized fixture would be an Explicit one in disguise and the
    // mean/draw comparison would say nothing about expression evaluation.
    let (compiled, params) = parameterized_model();
    let (base, _) = compiled.initial_state_mean(&params).expect("mean");
    assert_state_is_populated(&base.counts, "parameterized");

    let n0 = compiled.param_index["N0"];
    let mut bumped = params.clone();
    bumped[n0] += 1000.0;
    let (moved, _) = compiled.initial_state_mean(&bumped).expect("mean at bumped N0");
    assert_ne!(
        base.counts, moved.counts,
        "N0 does not move the initial state — this fixture does not exercise \
         parameterized IC evaluation"
    );
}

#[test]
fn the_draw_equals_the_mean_and_leaves_the_stream_untouched() {
    for (who, (compiled, params)) in
        [("parameterized", parameterized_model()), ("explicit", explicit_model())]
    {
        let (mean_int, mean_real) = compiled.initial_state_mean(&params).expect("mean");
        assert_state_is_populated(&mean_int.counts, who);

        let mut rng = StatefulRng::new(SEED);
        let (draw_int, draw_real) = compiled
            .initial_state_draw(&params, &mut rng)
            .expect("draw");

        // No `init {}` entry can declare a law yet, so the draw IS the mean.
        assert_eq!(draw_int.counts, mean_int.counts, "{who}: draw != mean (int)");
        assert_eq!(draw_real.values, mean_real.values, "{who}: draw != mean (real)");

        // …and it consumed nothing, so the caller's stream is where it was.
        // This is the load-bearing claim: every stochastic forward path now
        // routes its own stream through `initial_state_draw`, and a draw that
        // consumed here would shift every one of those trajectories.
        let mut untouched = StatefulRng::new(SEED);
        let after_draw: Vec<f64> = (0..8).map(|_| rng.uniform()).collect();
        let never_drawn: Vec<f64> = (0..8).map(|_| untouched.uniform()).collect();
        assert_eq!(
            after_draw, never_drawn,
            "{who}: initial_state_draw consumed from the RNG — every forward \
             path's trajectory would shift"
        );
    }
}

#[test]
fn the_density_and_its_gradient_agree_that_there_is_no_law() {
    for (who, (compiled, params)) in
        [("parameterized", parameterized_model()), ("explicit", explicit_model())]
    {
        let (int_s, real_s) = compiled.initial_state_mean(&params).expect("mean");
        assert_state_is_populated(&int_s.counts, who);

        let lp = compiled.initial_state_logpdf(&int_s.counts, &real_s.values, &params);
        assert!(lp.is_finite(), "{who}: logpdf is not finite ({lp})");
        assert_eq!(lp, 0.0, "{who}: a deterministic init {{}} contributes no density");

        let grad = compiled.initial_state_logpdf_grad(&int_s.counts, &real_s.values, &params);

        // The shape contract staging step 4 depends on: MODEL-parameter basis,
        // so a caller in an estimated-parameter basis can index it through its
        // own `estimated_to_model` without an out-of-bounds or a
        // wrong-parameter attribution.
        assert_eq!(
            grad.len(),
            params.len(),
            "{who}: logpdf_grad must be indexed by model parameter"
        );
        assert!(grad.iter().all(|g| g.is_finite()), "{who}: logpdf_grad has a non-finite entry");
        assert!(
            grad.iter().all(|&g| g == 0.0),
            "{who}: ∂/∂θ of a deterministic initial state must be zero"
        );

        // Non-vacuity on the length assertion: a model with no parameters would
        // make `grad.len() == params.len()` true at zero.
        assert!(!params.is_empty(), "{who}: fixture has no parameters");
    }
}
