//! StepClock oracle (scheduling-spine-v2 §A): a transition RATE that references
//! `dt` (`Expr::Dt`, gh#54) must be evaluated at the REALIZED substep length
//! (`EvalCtx.dt = dt_actual`), not the nominal grid `dt`, end-to-end across the
//! forward producer (`step_one` via `simulate_reference_on_grid`) and the PGAS
//! complete-data density — under `StepPolicy::Exact`, where off-grid
//! observations shorten substeps so `dt_actual ≠ grid_dt`.
//!
//! Why this gate exists (the coverage hole it closes): every other dt-sensitive
//! gate misses the `Expr::Dt`-in-a-rate path under clipping.
//!   - `gate_pgas_density_baseline` runs `simulate_reference` at a FIXED `dt`
//!     (uniform substeps, `dt_actual == grid_dt` always) — it never clips.
//!   - `pgas_exact_tiling` DOES clip, but `seir_vaccine_seasonal`'s
//!     `dt_substep` sensitivity comes from the `1-exp(-rate·dt)` KERNEL and the
//!     seasonal `t0`, NOT from a rate expression that reads `dt`.
//! So before this gate, the exact path StepClock's `EvalCtx.dt = dt_actual`
//! decision governs — a rate FORMULA that contains `dt` — was unguarded in the
//! clipped regime. Feeding the grid `dt` there instead would silently change
//! the likelihood, hence the posterior.
//!
//! Fixture: `tests/fixtures/corner_cases/dt_rate.camdl` — the infection hazard
//! carries an explicit `(dt / tau)` factor, so the realized substep length
//! enters the rate expression linearly.

use std::sync::Arc;

use sim::compiled_model::CompiledModel;
use sim::inference::pgas::{
    build_substep_grid, complete_data_loglik, log_transition_density_substep,
    simulate_reference_on_grid, ObsAtSubstep,
};
use sim::inference::particle_filter::Observation;
use sim::inference::MultiStreamObsModel;
use sim::propensity::eval_propensities;
use sim::rng::StatefulRng;
use sim::schedule::StepPolicy;
use sim::state::{IntState, RealState};

const SEED: u64 = 11;
const DT: f64 = 1.0;

/// Off-grid observation times with VARIED fractional gaps, so each Exact window
/// re-anchors off the dt=1 grid and produces a genuinely shortened final
/// substep (equal gaps would re-align after the first remainder).
const OBS_TIMES: &[f64] = &[3.5, 7.2, 11.8, 15.3, 19.6];

fn load_dt_rate() -> (Arc<CompiledModel>, Vec<f64>) {
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../../tests/fixtures/corner_cases/ir/dt_rate.ir.json"
    );
    let json = std::fs::read_to_string(path).unwrap_or_else(|e| panic!("read {path}: {e}"));
    let model = ir::from_str(&json).expect("parse dt_rate IR");
    let compiled = Arc::new(CompiledModel::new(model).expect("compile dt_rate"));
    let params = compiled.default_params.clone();
    (compiled, params)
}

fn obs_at(times: &[f64]) -> Vec<Observation> {
    times.iter().map(|&t| Observation { time: t, value: 0.0 }).collect()
}

fn reference_trajectory(
    compiled: &Arc<CompiledModel>,
    params: &[f64],
) -> sim::inference::pgas::PGASTrajectory {
    let t_start = compiled.model.simulation.t_start;
    let observations = obs_at(OBS_TIMES);
    let grid = build_substep_grid(t_start, DT, &observations, &[], StepPolicy::Exact)
        .expect("build exact substep grid");
    let mut rng = StatefulRng::new(SEED);
    simulate_reference_on_grid(compiled, params, DT, &grid.steps, None, &mut rng)
        .expect("simulate_reference_on_grid on dt_rate")
}

/// (1) Integration / consumer-consistency: the full producer → records →
/// `complete_data_loglik` pipeline runs on an `Expr::Dt`-rate model under Exact
/// clipping, stays finite, and scores it from the REALIZED `(t0, dt_substep)`
/// records (== the realized recompute, != the uniform `s·dt` reconstruction).
///
/// NB: the realized-vs-uniform Δ here is driven by the realized records broadly
/// — the kernel's `1-exp(-rate·dt_substep)`, the substep time `t0`, AND the
/// rate's `dt` — so it is NOT an isolated `Expr::Dt` discriminator (freezing
/// `Expr::Dt` leaves this test green; the kernel/t0 difference remains). The
/// clean, isolated `Expr::Dt`/StepClock oracle is test (2) below. This test
/// pins that the consumer reads the realized records (cf. pgas_exact_tiling's
/// arm (c), here on the dt-referencing model) and that the pipeline is
/// NaN-free under clipping.
#[test]
fn dt_rate_density_reads_realized_records_under_exact_clip() {
    let (compiled, params) = load_dt_rate();
    let t_start = compiled.model.simulation.t_start;
    let traj = reference_trajectory(&compiled, &params);

    // Non-vacuity: off-grid obs genuinely shortened substeps.
    let n_short = traj.substeps.iter().filter(|r| r.dt_substep != DT).count();
    eprintln!("  shortened substeps: {n_short} / {}", traj.substeps.len());
    assert!(n_short >= 1, "off-grid obs must shorten ≥1 substep (got {n_short})");
    for r in traj.substeps.iter().filter(|r| r.dt_substep != DT) {
        assert!(
            r.dt_substep > 0.0 && r.dt_substep < DT,
            "shortened dt_substep {} must be in (0, dt)",
            r.dt_substep
        );
    }

    // Recompute the transition density two ways over the SAME trajectory:
    //   realized: EvalCtx.dt = rec.dt_substep (the clipped substep)
    //   uniform : EvalCtx.dt = grid dt for every substep (the StepClock regression)
    let recompute = |use_uniform: bool| -> f64 {
        let mut total = 0.0;
        for (s, rec) in traj.substeps.iter().enumerate() {
            let (t, dt_s) = if use_uniform {
                (t_start + s as f64 * DT, DT)
            } else {
                (rec.t0, rec.dt_substep)
            };
            total += log_transition_density_substep(
                &compiled, &rec.counts_before, &rec.flows, &rec.gammas, &params, t, dt_s, None,
            )
            .expect("finite per-substep density");
        }
        total
    };
    let d_realized = recompute(false);
    let d_uniform = recompute(true);
    assert!(d_realized.is_finite(), "realized density must be finite");

    // The consumer reads rec.(t0, dt_substep): its transition component equals the
    // realized recompute bit-for-bit (no overdispersion in this model → no gamma
    // term). Obs-independent — no obs scoring.
    let obs_model = MultiStreamObsModel::empty(compiled.clone());
    let comps = complete_data_loglik(
        &compiled,
        &traj,
        &params,
        &[],
        DT,
        &obs_model,
        &ObsAtSubstep::new(),
    )
    .expect("complete_data_loglik");
    assert_eq!(
        comps.transition.to_bits(),
        d_realized.to_bits(),
        "complete_data_loglik must score the (dt/tau) rate at the realized dt_substep \
         ({} vs recompute {})",
        comps.transition,
        d_realized
    );

    // Non-vacuity: the realized records reconstruct a materially different
    // density than the uniform `s·dt` reconstruction, so "reads realized
    // records" is a load-bearing claim (a consumer that ignored them would
    // change the density). This Δ is NOT `Expr::Dt`-specific — see the header;
    // the isolated rate-eval guard is test (2).
    eprintln!(
        "  d_realized={d_realized:.6e}  d_uniform={d_uniform:.6e}  Δ={:.3e}",
        d_realized - d_uniform
    );
    assert!(
        (d_realized - d_uniform).abs() > 1e-6 * d_uniform.abs().max(1.0),
        "realized records must reconstruct a different density than uniform s·dt (Δ={:.3e})",
        d_realized - d_uniform
    );
}

/// (2) Mechanism level: the infection propensity scales EXACTLY with
/// `EvalCtx.dt` (because its rate is `… · dt/tau`), while the recovery
/// propensity (`gamma·I`, no `dt`) is invariant. This isolates the `Expr::Dt`
/// evaluation from the kernel/density, so a regression that froze the rate at
/// the grid dt cannot hide behind the `1-exp(-rate·dt)` term.
#[test]
fn dt_rate_propensity_scales_with_eval_ctx_dt() {
    let (compiled, params) = load_dt_rate();
    let traj = reference_trajectory(&compiled, &params);

    let inf = compiled
        .model
        .transitions
        .iter()
        .position(|t| t.name == "infection")
        .expect("infection transition");
    let rec_idx = compiled
        .model
        .transitions
        .iter()
        .position(|t| t.name == "recovery")
        .expect("recovery transition");

    // First shortened substep with a live epidemic (infection propensity > 0).
    let real_s = RealState::new(0);
    let mut chosen = None;
    for rec in traj.substeps.iter().filter(|r| r.dt_substep != DT) {
        let mut int_s = IntState::new(rec.counts_before.len());
        int_s.counts.copy_from_slice(&rec.counts_before);
        let mut prop_grid = Vec::new();
        eval_propensities(&compiled, &int_s, &real_s, &params, rec.t0, DT, None, &mut prop_grid)
            .expect("eval at grid dt");
        if prop_grid[inf] > 0.0 {
            chosen = Some((rec.t0, rec.dt_substep, int_s, prop_grid));
            break;
        }
    }
    let (t0, short_dt, int_s, prop_grid) =
        chosen.expect("a shortened substep with nonzero infection propensity");
    assert!(short_dt > 0.0 && short_dt < DT, "need a genuinely shortened substep");

    let mut prop_short = Vec::new();
    eval_propensities(&compiled, &int_s, &real_s, &params, t0, short_dt, None, &mut prop_short)
        .expect("eval at realized dt");

    // infection rate = beta·S·I/N · (dt/tau): linear in EvalCtx.dt, so the only
    // difference between the two evals is the dt factor → exact ratio.
    let ratio = prop_short[inf] / prop_grid[inf];
    eprintln!(
        "  short_dt={short_dt}  infection: grid={:.6e} realized={:.6e} ratio={ratio:.12}",
        prop_grid[inf], prop_short[inf]
    );
    assert!(
        (ratio - short_dt / DT).abs() < 1e-12,
        "infection propensity must scale as dt_actual/grid_dt: ratio {ratio} vs {}",
        short_dt / DT
    );

    // ABSOLUTE pin (closes the ratio's blind spot): the realized propensity must
    // EQUAL the rate computed from first principles at the realized dt —
    // beta·S·I/N·(dt_actual/tau). The ratio alone only proves "linear in the dt
    // argument" and would survive a `Dt → k·ctx.dt` overload, or a read of the
    // wrong-but-proportional dt field (the grid_dt-vs-dt_actual mixup this branch
    // exists to prevent). The absolute value nails `Expr::Dt == ctx.dt`.
    let comp = |name: &str| {
        compiled.model.compartments.iter().position(|c| c.name == name)
            .unwrap_or_else(|| panic!("compartment {name} not found"))
    };
    let s = int_s.counts[comp("S")] as f64;
    let i = int_s.counts[comp("I")] as f64;
    let n: f64 = int_s.counts.iter().map(|&c| c as f64).sum();
    let beta = params[compiled.param_index["beta"]];
    let tau = params[compiled.param_index["tau"]];
    let expected_short = beta * s * i / n * (short_dt / tau);
    let rel = (prop_short[inf] - expected_short).abs() / expected_short.abs().max(1e-12);
    assert!(
        rel < 1e-9,
        "infection propensity must EQUAL beta·S·I/N·(dt_actual/tau) = {expected_short:.6e}, \
         got {:.6e} (rel {rel:.2e}) — pins Expr::Dt == ctx.dt, not merely linear in it",
        prop_short[inf]
    );

    // recovery rate = gamma·I: references no dt → bit-identical across the two evals.
    assert_eq!(
        prop_short[rec_idx].to_bits(),
        prop_grid[rec_idx].to_bits(),
        "recovery rate must be dt-independent (grid {} vs realized {})",
        prop_grid[rec_idx],
        prop_short[rec_idx]
    );
}
