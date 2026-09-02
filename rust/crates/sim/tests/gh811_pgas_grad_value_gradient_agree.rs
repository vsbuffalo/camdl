//! gh#811: when the PGAS density path rejects a substep, its gradient must be
//! the zero gradient of that constant floor — never NaN.
//!
//! `log_transition_density_grad` scores each substep with the shared
//! `binom_logpmf` and differentiates it with a hand-rolled `dbinom_dp` written
//! beside it. The two must agree about the rejected region, and gh#810 made
//! them disagree: it taught `binom_logpmf` to return `-inf` for a NaN
//! probability (correct — a NaN poisons every downstream sum, while `-inf`
//! kills one particle) without giving the hand-rolled derivative the matching
//! guard. Before that change both returned NaN: wrong, but consistently wrong
//! and therefore loud. After it, the value rejects the point while the
//! gradient hands NaN to the NUTS momentum update — a silent corruption.
//!
//! A NaN probability is reachable because `prob_q_from_rate_dt_clamped` passes
//! NaN straight through (`f64::clamp` returns NaN for NaN), so any rate
//! expression that evaluates to NaN arrives here as a NaN `p_total`.
//!
//! This is the gh#197 / gh#200 defect class — a value/gradient divergence in
//! the PGAS density path — and the house contract for it is already written
//! one branch below the defect: `return Ok((f64::NEG_INFINITY, vec![0.0; d]))`.

use sim::compiled_model::CompiledModel;
use sim::inference::pgas::simulate_reference;
use sim::inference::pgas_grad::{log_transition_density_grad, resolve_rate_grad_for_run};
use sim::rng::StatefulRng;

const DT: f64 = 1.0;
const T_END: f64 = 6.0;

fn sir() -> CompiledModel {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../../ocaml/golden/sir_overdispersion.ir.json");
    let json = std::fs::read_to_string(&path).expect("read sir_overdispersion golden");
    let mut m = ir::from_str(&json).expect("parse sir_overdispersion");
    m.simulation.t_start = 0.0;
    m.simulation.t_end = T_END;
    for p in &mut m.parameters {
        let v = match p.name.as_str() {
            "beta" => 1.2,
            "gamma" => 0.5,
            "N0" => 500.0,
            "I0" => 20.0,
            // whatever the overdispersion scale is called in this golden
            _ => 0.2,
        };
        p.value = ir::parameter::ParamValue::Fixed { value: v };
    }
    CompiledModel::new(m).expect("compile")
}

/// The invariant, stated as the house contract: a non-finite density has a
/// zero gradient. Never a NaN one.
#[test]
fn a_rejected_substep_has_a_zero_gradient_not_a_nan_one() {
    let compiled = sir();
    let good = compiled.default_params.clone();
    let t_start = compiled.model.simulation.t_start;

    let mut rng = StatefulRng::new(7);
    let reference =
        simulate_reference(&compiled, &good, T_END, DT, &mut rng).expect("reference");

    let d = good.len();
    let model_to_estimated: Vec<Option<usize>> = (0..good.len()).map(Some).collect();
    let rate_grads = resolve_rate_grad_for_run(
        &compiled.resolved.rate_grads_indexed, &model_to_estimated);

    // Parameters and computed propensities are BOTH finiteness-guarded
    // upstream (`propensity.rs:532` and `:651`), so a NaN cannot arrive that
    // way. The gamma multipliers are not: `pgas_grad.rs:160` reads
    // `gammas[gamma_idx]` and multiplies it into the rate with no guard, and
    // gammas are REPLAYED from a stored trajectory rather than freshly drawn.
    // That is the route this pins.

    let mut checked = 0usize;
    for (s, rec) in reference.substeps.iter().enumerate() {
        let t = t_start + s as f64 * DT;
        let mut nan_gammas = rec.gammas.clone();
        if nan_gammas.is_empty() { continue; }
        nan_gammas[0] = f64::NAN;

        let (lp, grad) = log_transition_density_grad(
            &compiled, &rec.counts_before, &rec.flows, &nan_gammas,
            &good, t, DT, None, d, &rate_grads,
        ).expect("grad call must not error");

        assert!(!lp.is_nan(),
            "substep {s}: the density itself came back NaN — gh#810 should have \
             made this -inf");

        if !lp.is_finite() {
            checked += 1;
            let bad: Vec<(usize, f64)> = grad.iter().copied().enumerate()
                .filter(|(_, g)| *g != 0.0)
                .collect();
            assert!(bad.is_empty(),
                "substep {s}: the density rejected this point (log_p = {lp}) but the \
                 gradient is not the zero gradient of that floor — components {bad:?}. \
                 A NaN here propagates into the NUTS momentum update, where the -inf \
                 value would merely have killed the particle. This is the \
                 gh#197/gh#200 divergence class.");
        }
    }
    assert!(checked > 0,
        "no substep was rejected, so this test proved nothing — the NaN parameter \
         is not reaching the density path and the fixture needs rethinking");
}
