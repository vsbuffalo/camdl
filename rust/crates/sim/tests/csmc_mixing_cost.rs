//! gh#718: what the defect-2 gate costs, in the unit that decides it.
//!
//! Gating ancestor sampling on "an ancestry was actually drawn" reduces how
//! often the reference trajectory can detach from its own past. That is a loss
//! of STATISTICAL EFFICIENCY per sweep — successive sweeps return more similar
//! trajectories, so there are fewer effectively-independent draws.
//!
//! It is not obviously a loss overall, because the same change makes each sweep
//! CHEAPER: `fill_ancestor_log_weights` and `splice_log_ratio` are skipped
//! wherever the gate closes, which on a weekly-observed model is most substeps.
//!
//! The quantity that combines the two is **ESS per second** — effective sample
//! size, which discounts for autocorrelation, divided by wall-clock. This test
//! reports it, alongside the pieces it is made of, so the tradeoff can be read
//! rather than assumed:
//!
//! - `renewal` — the fraction of the returned trajectory taken from a
//!   non-reference particle, averaged over sweeps. The direct measure of how
//!   much of the path ancestor sampling replaced.
//! - `renewal_by_bin` — the same resolved in time, which is what distinguishes
//!   "the path is stuck everywhere" from "the path is stuck early", the failure
//!   mode ancestor sampling exists to prevent.
//! - `ESS/sweep` and `ESS/sec` for a functional of the latent trajectory.
//!
//! This is a MEASUREMENT, not a gate: it asserts only that the chain moved at
//! all, so it cannot fail spuriously on a slow machine. Run it with
//! `--nocapture` and read the table.

use std::sync::Arc;
use std::time::Instant;

use sim::compiled_model::CompiledModel;
use sim::inference::dense_cells;
use sim::inference::multi_stream_obs::{BoundObs, MultiStreamObsModel, StreamProjection, StreamSpec};
use sim::inference::particle_filter::Observation;
use sim::inference::pgas::{
    build_obs_at_substep, csmc_as, simulate_reference, EffectFiring, ObsAtSubstep, PGASTrajectory,
};
use sim::rng::StatefulRng;

const DT: f64 = 1.0;
const SEED: u64 = 20260823;
const I_IDX: usize = 1;

fn prevalence_obs_block() -> ir::observation::ObservationModel {
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

fn model(t_end: f64) -> Arc<CompiledModel> {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../../ocaml/golden/sir_overdispersion.ir.json");
    let json = std::fs::read_to_string(&path).expect("read sir_overdispersion golden");
    let mut m = ir::from_str(&json).expect("parse");
    m.observations = vec![prevalence_obs_block()];
    m.simulation.t_start = 0.0;
    m.simulation.t_end = t_end;
    for p in &mut m.parameters {
        let v = match p.name.as_str() {
            "beta" => 0.3,
            "gamma" => 0.1,
            "sigma_se" => 0.1,
            "N0" => 1000.0,
            "I0" => 10.0,
            other => panic!("unexpected parameter {other}"),
        };
        p.value = ir::parameter::ParamValue::Fixed { value: v };
    }
    Arc::new(CompiledModel::new(m).expect("compile"))
}

/// Autocorrelation-corrected effective sample size, initial-positive-sequence
/// estimator (Geyer 1992): sum the autocovariance until a pair of consecutive
/// lags sums non-positive. Same family as the one the fit summary reports.
fn ess(x: &[f64]) -> f64 {
    let n = x.len();
    if n < 8 {
        return n as f64;
    }
    let mean = x.iter().sum::<f64>() / n as f64;
    let var = x.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / n as f64;
    if var <= 0.0 {
        return 1.0;
    }
    let acf = |k: usize| -> f64 {
        x[..n - k].iter().zip(&x[k..]).map(|(a, b)| (a - mean) * (b - mean)).sum::<f64>()
            / n as f64
            / var
    };
    let mut sum = 0.0;
    let mut k = 1usize;
    while k + 1 < n {
        let pair = acf(k) + acf(k + 1);
        if pair <= 0.0 {
            break;
        }
        sum += pair;
        k += 2;
    }
    n as f64 / (1.0 + 2.0 * sum)
}

/// One PGAS trajectory chain at fixed θ: repeated `csmc_as` sweeps, each
/// conditioned on the last one's output. That is exactly the X-update PGAS
/// performs; holding θ fixed isolates the trajectory kernel from the NUTS step.
#[test]
fn report_the_mixing_cost_of_the_ancestor_sampling_gate() {
    let n_substeps = 80usize;
    let cadence = std::env::var("CSMC_MIXING_CADENCE")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(7);
    let n_sweeps = std::env::var("CSMC_MIXING_SWEEPS")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(400);
    let n_particles = std::env::var("CSMC_MIXING_NP")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(256);

    let compiled = model(n_substeps as f64 * DT);
    let params = compiled.default_params.clone();
    let mut rng = StatefulRng::new(SEED);
    let reference = simulate_reference(&compiled, &params, n_substeps as f64 * DT, DT, &mut rng)
        .expect("reference");

    // Observations every `cadence` substeps — the knob that sets how many
    // substeps draw an ancestry, and so how often the gate opens.
    let obs: Vec<Observation> = (0..n_substeps)
        .filter(|s| (s + 1) % cadence == 0)
        .map(|s| Observation {
            time: ((s + 1) as f64) * DT,
            value: reference.substeps[s].counts_after[I_IDX] as f64,
        })
        .collect();
    assert!(obs.len() >= 4, "need several observations");

    let obs_model = MultiStreamObsModel::new(
        BoundObs::bind(vec![StreamSpec::dense(
            StreamProjection::IntCompSum(vec![I_IDX]),
            compiled.model.observations[0].clone(),
            dense_cells(obs.iter().map(|o| o.value).collect()),
            obs.iter().map(|o| o.time).collect(),
        )])
        .unwrap()
        .0,
        compiled.clone(),
    )
    .unwrap();
    let obs_at_substep: ObsAtSubstep =
        build_obs_at_substep(&obs, compiled.model.simulation.t_start, DT).expect("obs_at_substep");

    // The chain.
    let mut traj: PGASTrajectory = reference.clone();
    let mut renewal_sum = 0.0;
    let mut renewal_bins_sum = [0.0f64; 10];
    let mut renewal_bins_n = [0usize; 10];
    let (mut opportunities, mut proposed, mut accepted) = (0usize, 0usize, 0usize);
    // Functional: prevalence at the LAST substep. The terminal state is the
    // hardest coordinate for the sampler to renew, so it is the honest one.
    let mut series: Vec<f64> = Vec::with_capacity(n_sweeps);

    let t0 = Instant::now();
    for i in 0..n_sweeps {
        let (next, d) = csmc_as(
            &compiled,
            &params,
            &obs,
            &traj,
            n_particles,
            DT,
            &obs_model,
            SEED.wrapping_add(i as u64).wrapping_mul(0x9e3779b97f4a7c15),
            &obs_at_substep,
            EffectFiring::default(),
        )
        .expect("csmc_as");
        renewal_sum += d.trajectory_renewal;
        for (b, &r) in d.renewal_by_bin.iter().enumerate() {
            if r.is_finite() {
                renewal_bins_sum[b] += r;
                renewal_bins_n[b] += 1;
            }
        }
        opportunities += d.n_resampled;
        proposed += d.n_as_proposed;
        accepted += d.n_as_accepted;
        series.push(next.substeps[n_substeps - 1].counts_after[I_IDX] as f64);
        traj = next;
    }
    let secs = t0.elapsed().as_secs_f64();

    let e = ess(&series);
    let bins: Vec<String> = (0..10)
        .map(|b| {
            if renewal_bins_n[b] > 0 {
                format!("{:.2}", renewal_bins_sum[b] / renewal_bins_n[b] as f64)
            } else {
                "NA".into()
            }
        })
        .collect();

    eprintln!();
    eprintln!("=== CSMC mixing, {n_sweeps} sweeps, {n_particles} particles, \
               {n_substeps} substeps, observation every {cadence} ===");
    eprintln!("  AS opportunities / sweep : {:.2}", opportunities as f64 / n_sweeps as f64);
    eprintln!("  AS proposed / sweep      : {:.2}", proposed as f64 / n_sweeps as f64);
    eprintln!("  AS accepted / sweep      : {:.2}", accepted as f64 / n_sweeps as f64);
    eprintln!(
        "  AS acceptance rate       : {:.1}%",
        100.0 * accepted as f64 / proposed.max(1) as f64
    );
    eprintln!("  trajectory_renewal       : {:.4}", renewal_sum / n_sweeps as f64);
    eprintln!("  renewal_by_bin           : [{}]", bins.join(" "));
    eprintln!("  wall clock               : {secs:.2} s  ({:.1} ms/sweep)",
              1000.0 * secs / n_sweeps as f64);
    eprintln!("  ESS (terminal prevalence): {e:.1} of {n_sweeps} sweeps");
    eprintln!("  ESS / sweep              : {:.4}", e / n_sweeps as f64);
    eprintln!("  ESS / second             : {:.2}", e / secs);
    eprintln!();

    // Measurement, not gate: assert only that the chain is alive, so this
    // cannot go red on a loaded machine.
    assert!(e > 1.0, "the trajectory chain did not move at all (ESS {e:.2})");
    assert!(
        series.iter().any(|&v| v != series[0]),
        "the terminal state never changed across {n_sweeps} sweeps"
    );
}
