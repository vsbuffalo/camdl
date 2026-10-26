//! The post-advance tail of the shared within-substep effect lifecycle.
//!
//! Every backend applies the effects due at one instant in the same order
//! (language spec §13.10):
//!
//! ```text
//! INFLOW events (from the START-OF-STEP SNAPSHOT)
//!     -> ADVANCE (chain-binomial's transition draw; none at a Gillespie or ODE boundary)
//!     -> DRAIN-AND-`set` events (on the resulting state)
//!     -> INTERVENE (on the post-event state, declaration order)
//!     -> BALANCE (last, chain-only)
//! ```
//!
//! The first three stages are [`crate::effects::apply_event_phases`]; the last
//! two are [`apply_post_advance`] here. The ADVANCE kernel itself (transition
//! draws) stays per-backend, because the draw algorithms differ
//! (Euler-multinomial vs RK4 vs SSA) — that is the seam: unify the effect
//! bookkeeping, keep the kernels distinct.

use crate::{
    compiled_model::{CompiledModel, ResolvedBalance},
    error::SimError,
    intervention::apply_effect_batch,
    propensity::EvalCtx,
    resolved_expr::eval_resolved,
    state::{IntState, RealState},
};

/// → `FixedStepLifecycle::apply_post_advance`. Stages 3-4: INTERVENE then BALANCE on the
/// CURRENT post-advance state, in fixed order. One function so no backend can reorder them.
/// NOTE: for tau/ode/gillespie this is a one-call passthrough today (balance is chain-only).
///
/// INTERVENE applies the `intervention_idx` batch [`due_effects`](crate::effects::due_effects)
/// derived for this substep (reading the current post-advance state); BALANCE
/// then overwrites the target compartment so the population budget holds,
/// reading the post-intervention state. The balance target is exempt from the
/// negative-count check by construction (its negativity is a separate signal,
/// warned about here, not erred). RNG-free.
///
/// `intervention_idx` is the scheduled (`!always_active`) interventions due at
/// `t + dt`, in declaration order — the seam no longer re-derives due-ness here
/// (the duplication removed by the scheduling-spine §B). `dt` is `dt_actual` —
/// the realized substep length, driving the balance / effect-amount evaluation.
/// See docs/dev/proposals/2026-06-07-scheduling-spine-v2.md §A/§B.
#[allow(clippy::too_many_arguments)]
pub fn apply_post_advance(
    model: &CompiledModel,
    intervention_idx: &[usize],
    current: &mut IntState,
    real: &mut RealState,
    params: &[f64],
    t: f64,
    dt: f64,
    balance: Option<&ResolvedBalance>,
) -> Result<(), SimError> {
    let t_end = t + dt;

    // Stage 3: INTERVENE on the current post-advance state.
    if !intervention_idx.is_empty() {
        apply_effect_batch(
            t_end, model, intervention_idx, dt, current, real, params,
        )?;
    }

    // Stage 4: BALANCE — overwrite the target compartment so the population
    // budget holds. All other compartments are finalized at this point.
    if let Some(bal) = balance {
        let ctx = EvalCtx {
            model, int_s: current, real_s: real,
            params, t: t_end, dt, projected: None, aux: None, int_float_override: None, per_eval: None,
        };
        let val = eval_resolved(&bal.expr, &ctx);
        let bal_count = val.round() as i64;
        if bal_count < 0 {
            log::warn!("balance compartment went negative ({}) at t={:.1} — \
                        model may be inconsistent at these parameters", bal_count, t_end);
        }
        current.counts[bal.local_int_idx] = bal_count;
    }

    // Post-INTERVENE/BALANCE negative-count check. The pre-advance scan in
    // each kernel catches negatives from the transition draws (and fused event
    // deltas), but it runs *before* this function — so an INTERVENE `set` to a
    // value below zero slips past it. Catch it here, after the canonical
    // INTERVENE+BALANCE tail. The balance target is exempt: its negativity is a
    // separate, already-warned signal (above), not a count error.
    let balance_target = balance.map(|b| b.local_int_idx);
    for (local, &count) in current.counts.iter().enumerate() {
        if count < 0 && Some(local) != balance_target {
            return Err(SimError::NegativeCount {
                compartment: model.int_compartment_name(local),
                attempted_value: count,
                t: t_end,
                cause: crate::error::NegativeCountCause::InterventionNegative,
            });
        }
    }

    Ok(())
}
