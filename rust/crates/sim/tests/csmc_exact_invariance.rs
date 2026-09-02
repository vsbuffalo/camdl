//! gh#718: does one `csmc_as` sweep leave `p(X | θ, y)` invariant?
//!
//! Every previous check on this kernel pins a PIECE of it —
//! `csmc_splice_ratio_oracle` pins `splice_log_ratio` against
//! `complete_data_loglik`, `pgas_ancestor_weight` pins the Eq.-(17) weight,
//! `csmc_splice_continuity` pins that the returned path does not jump. None of
//! them asks the only question that matters: applied to a draw from the target,
//! does the kernel return a draw from the target?
//!
//! # Why this fixture can answer it exactly
//!
//! The measurement in gh#718 needed sampling-importance-resampling for a ground
//! truth, so every comparison carried the ground truth's own Monte-Carlo error
//! and the verdict rested on a 11-SE gap. Here there is no ground-truth error at
//! all. The model is small enough that the **entire support is enumerable**: a
//! plain (non-overdispersed) SIR has no continuous noise, so a trajectory is
//! determined by its per-substep integer flows, and with `N₀ = 6` over 4
//! substeps that is a few thousand paths. Enumerate them, score each with
//! `complete_data_loglik`, normalise — and `π` is exact to floating point.
//!
//! The test is then the definition of invariance, run directly:
//!
//! ```text
//!   X₀ ~ π   (exact categorical draw over the enumerated support)
//!   X₁ = csmc_as(X₀, true)
//!   H₀:  X₁ ~ π
//! ```
//!
//! `X₀` is an exact draw, so under `H₀` the tally of `X₁` over the support is
//! multinomial with the known probabilities `π` — a goodness-of-fit test with no
//! nuisance parameters and no competing hypothesis about relaxation rate. A
//! kernel that is not invariant fails it at a rate set only by `M`.
//!
//! # Non-vacuity
//!
//! Three ways this could pass for the wrong reason, each asserted against:
//!
//! - the ancestor-sampling MH step never fires (then the sweep is nearly plain
//!   particle Gibbs and the splice path is untested) — the diagnostics counters
//!   are summed and required to be non-trivial;
//! - the returned path leaves the enumerated support (then the tally is silently
//!   incomplete) — an escape is a hard failure, not a dropped sample;
//! - `π` is concentrated on one path (then any kernel passes) — the effective
//!   support size is asserted.

use std::collections::HashMap;
use std::sync::Arc;

use sim::compiled_model::CompiledModel;
use sim::inference::dense_cells;
use sim::inference::multi_stream_obs::{BoundObs, MultiStreamObsModel, StreamProjection, StreamSpec};
use sim::inference::particle_filter::Observation;
use sim::inference::pgas::{
    build_obs_at_substep, complete_data_loglik, csmc_as, EffectFiring, ObsAtSubstep,
    PGASTrajectory, SubstepRecord,
};
use sim::rng::StatefulRng;

const DT: f64 = 1.0;
const N_SUBSTEPS: usize = 4;
/// Two substeps with the ONLY observation at the end: the single ancestor-
/// sampling move that matters runs on uniform (identity) ancestry, and no later
/// resampling step can restore the reference lineage to the free particles.
const TRAP: [(usize, f64); 1] = [(1, 3.0)];
const TRAP_SUBSTEPS: usize = 2;
/// Local compartment indices in `sir_basic` (compartments are `S, I, R`).
const S_IDX: usize = 0;
const I_IDX: usize = 1;
const R_IDX: usize = 2;
/// Transition indices in `sir_basic` (`infection`, then `recovery`).
const TR_INFECTION: usize = 0;
const TR_RECOVERY: usize = 1;

/// Which substeps carry a prevalence reading, and what it reads.
///
/// Two schedules, because they exercise different code. DENSE observes every
/// substep, so `csmc_as` resamples at every one. SPARSE leaves substeps
/// unobserved, where the weights are uniform and `csmc_as` SKIPS resampling
/// entirely — a path the dense schedule never reaches.
const DENSE: [(usize, f64); 4] = [(0, 3.0), (1, 3.0), (2, 2.0), (3, 2.0)];
const SPARSE: [(usize, f64); 2] = [(1, 3.0), (3, 2.0)];

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

/// A deliberately tiny SIR. `sir_basic` has plain binomial draws — no
/// `overdispersed(...)`, hence no continuous gamma multipliers, hence a
/// finite trajectory support. The bounds are widened because the golden's
/// `N0 ∈ [100, 100000]` would otherwise exclude the population that makes
/// enumeration possible.
fn model(n_substeps: usize) -> Arc<CompiledModel> {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../../ocaml/golden/sir_basic.ir.json");
    let json = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("read sir_basic golden {path:?}: {e}"));
    let mut m = ir::from_str(&json).expect("parse sir_basic");
    m.observations = vec![prevalence_obs_block()];
    m.simulation.t_start = 0.0;
    m.simulation.t_end = n_substeps as f64 * DT;
    for p in &mut m.parameters {
        let v = match p.name.as_str() {
            "beta" => 1.2,
            "gamma" => 0.5,
            "N0" => 6.0,
            "I0" => 2.0,
            other => panic!("unexpected parameter {other} in sir_basic"),
        };
        // `Fixed`, not `with_value`: an `Estimated` parameter keeps the golden's
        // bounds, and `N0 ∈ [100, 100000]` would reject the population that
        // makes the support enumerable.
        p.value = ir::parameter::ParamValue::Fixed { value: v };
    }
    let compiled = Arc::new(CompiledModel::new(m).expect("compile sir_basic"));
    assert_eq!(
        compiled.model.compartments.iter().map(|c| c.name.as_str()).collect::<Vec<_>>(),
        vec!["S", "I", "R"],
        "the enumeration hardcodes the stoichiometry; compartment order changed"
    );
    assert_eq!(
        compiled.model.transitions.iter().map(|t| t.name.as_str()).collect::<Vec<_>>(),
        vec!["infection", "recovery"],
        "the enumeration hardcodes the stoichiometry; transition order changed"
    );
    compiled
}

struct Fixture {
    compiled: Arc<CompiledModel>,
    params: Vec<f64>,
    obs: Vec<Observation>,
    obs_model: MultiStreamObsModel,
    obs_at_substep: ObsAtSubstep,
    initial_counts: Vec<i64>,
    n_substeps: usize,
}

fn fixture() -> Fixture {
    fixture_with(&DENSE, N_SUBSTEPS)
}

fn fixture_with(schedule: &[(usize, f64)], n_substeps: usize) -> Fixture {
    let compiled = model(n_substeps);
    let params = compiled.default_params.clone();
    let (init, _) = compiled.initial_state_mean(&params).expect("initial state");
    let initial_counts = init.counts.clone();

    let obs: Vec<Observation> = schedule
        .iter()
        .map(|&(s, v)| Observation { time: ((s + 1) as f64) * DT, value: v })
        .collect();

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
    assert_eq!(
        obs_at_substep.len(),
        schedule.len(),
        "every scheduled observation must land on a substep"
    );

    Fixture { compiled, params, obs, obs_model, obs_at_substep, initial_counts, n_substeps }
}

/// The flows of every substep, flattened — the identity of a trajectory on this
/// model, since the initial state is deterministic and `counts_after` is
/// `counts_before + A·flows`.
fn key(traj: &PGASTrajectory) -> Vec<u64> {
    traj.substeps.iter().flat_map(|r| r.flows.iter().copied()).collect()
}

fn advance(before: &[i64], k_inf: u64, k_rec: u64) -> Vec<i64> {
    let mut after = before.to_vec();
    after[S_IDX] -= k_inf as i64;
    after[I_IDX] += k_inf as i64 - k_rec as i64;
    after[R_IDX] += k_rec as i64;
    after
}

/// Every trajectory the model can produce: at each substep the infection flow
/// is bounded by `S` and the recovery flow by `I`, so the support is the
/// product of those ranges along each path.
fn enumerate_paths(initial: &[i64], n_substeps: usize) -> Vec<PGASTrajectory> {
    let mut out: Vec<PGASTrajectory> = Vec::new();
    let mut stack: Vec<(Vec<i64>, Vec<SubstepRecord>)> = vec![(initial.to_vec(), Vec::new())];
    while let Some((state, recs)) = stack.pop() {
        let s = recs.len();
        if s == n_substeps {
            out.push(PGASTrajectory { initial_counts: initial.to_vec(), substeps: recs });
            continue;
        }
        for k_inf in 0..=(state[S_IDX] as u64) {
            for k_rec in 0..=(state[I_IDX] as u64) {
                let mut flows = vec![0u64; 2];
                flows[TR_INFECTION] = k_inf;
                flows[TR_RECOVERY] = k_rec;
                let after = advance(&state, k_inf, k_rec);
                let mut next = recs.clone();
                next.push(SubstepRecord {
                    counts_before: state.clone(),
                    counts_after: after.clone(),
                    flows,
                    gammas: Vec::new(),
                    t0: s as f64 * DT,
                    dt_substep: DT,
                });
                stack.push((after, next));
            }
        }
    }
    out
}

/// The exact smoothing target over the enumerated support: `π(X) ∝ p(X, y)`,
/// read from the same `complete_data_loglik` the θ-move conditions on.
fn exact_target(f: &Fixture) -> (Vec<PGASTrajectory>, Vec<f64>, HashMap<Vec<u64>, usize>) {
    let all = enumerate_paths(&f.initial_counts, f.n_substeps);
    let mut paths = Vec::new();
    let mut logp = Vec::new();
    for traj in all {
        let ll = complete_data_loglik(
            &f.compiled,
            &traj,
            &f.params,
            &f.obs,
            DT,
            &f.obs_model,
            &f.obs_at_substep,
        )
        .expect("complete_data_loglik")
        .total;
        if ll.is_finite() {
            paths.push(traj);
            logp.push(ll);
        }
    }
    let max = logp.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let mut w: Vec<f64> = logp.iter().map(|l| (l - max).exp()).collect();
    let z: f64 = w.iter().sum();
    for x in w.iter_mut() {
        *x /= z;
    }
    let index: HashMap<Vec<u64>, usize> =
        paths.iter().enumerate().map(|(i, t)| (key(t), i)).collect();
    assert_eq!(index.len(), paths.len(), "enumeration produced duplicate flow keys");
    (paths, w, index)
}

fn draw_categorical(p: &[f64], u: f64) -> usize {
    let mut c = 0.0;
    for (i, &q) in p.iter().enumerate() {
        c += q;
        if u < c {
            return i;
        }
    }
    p.len() - 1
}

/// The control that makes the invariance test above mean anything: `π` is
/// built from `complete_data_loglik`, which scores a path with
/// `log_transition_density_substep`. If that density is not the law
/// `step_one` actually draws from, then `π` is not the kernel's target and a
/// failure above would say nothing about ancestor sampling.
///
/// So measure it: draw the substep many times from a fixed state and compare
/// the empirical flow frequencies against the density, cell by cell.
#[test]
fn the_scored_density_is_the_producers_own_law() {
    use sim::chain_binomial::{step_one, StepScratch};
    use sim::inference::pgas::log_transition_density_substep;
    use sim::state::RealState;

    let f = fixture();
    let m = 400_000usize;
    let state = f.initial_counts.clone();
    let n_tr = f.compiled.model.transitions.len();

    let mut tally: HashMap<Vec<u64>, u64> = HashMap::new();
    let mut scratch = StepScratch::new(&f.compiled);
    for i in 0..m {
        let mut rng = StatefulRng::new_stream(0xabcd_0000u64.wrapping_add(i as u64), 0);
        let mut counts = state.clone();
        let mut flows = vec![0u64; n_tr];
        let mut real = RealState::new(f.compiled.real_local_to_global.len());
        step_one(
            &f.compiled, &mut counts, &mut flows, &mut real, &f.params, 0.0, DT, None, &mut rng,
            &mut scratch,
        )
        .expect("step_one");
        *tally.entry(flows).or_insert(0) += 1;
    }

    let mf = m as f64;
    let mut worst = (0.0f64, Vec::new(), 0.0, 0.0);
    let mut total_p = 0.0;
    for k_inf in 0..=(state[S_IDX] as u64) {
        for k_rec in 0..=(state[I_IDX] as u64) {
            let flows = vec![k_inf, k_rec];
            let ld = log_transition_density_substep(
                &f.compiled, &state, &flows, &[], &f.params, 0.0, DT, None,
            )
            .expect("density");
            let p = ld.exp();
            total_p += p;
            let obs = *tally.get(&flows).unwrap_or(&0) as f64 / mf;
            if p * mf < 25.0 {
                continue;
            }
            let z = (obs - p) / (p * (1.0 - p) / mf).sqrt();
            if z.abs() > worst.0 {
                worst = (z.abs(), flows.clone(), p, obs);
            }
        }
    }
    eprintln!(
        "density vs producer: Σp={total_p:.6}, worst cell |z|={:.2} at flows {:?} \
         (density {:.6}, empirical {:.6})",
        worst.0, worst.1, worst.2, worst.3
    );
    assert!(
        (total_p - 1.0).abs() < 1e-9,
        "the scored density does not sum to 1 over the substep's support (Σp={total_p}) — \
         it is not a probability mass function of the flows"
    );
    assert!(
        worst.0 < 5.0,
        "the density `complete_data_loglik` scores is NOT the law `step_one` draws from: \
         worst cell |z|={:.2} at flows {:?} (density {:.6} vs empirical {:.6} over {m} draws). \
         Every likelihood in the inference stack is then scoring the wrong thing.",
        worst.0, worst.1, worst.2, worst.3
    );
}

#[test]
fn one_sweep_leaves_the_smoothing_target_invariant() {
    check_invariance(&DENSE, N_SUBSTEPS, true, "dense (observation on every substep)");
}

/// The same question on a schedule with UNOBSERVED substeps, where the weights
/// are uniform and `csmc_as` skips resampling altogether. The dense schedule
/// never reaches that branch, so without this case the skip is unexercised.
#[test]
fn one_sweep_is_invariant_when_some_substeps_skip_resampling() {
    check_invariance(&SPARSE, N_SUBSTEPS, true, "sparse (some substeps skip resampling)");
}

fn check_invariance(
    schedule: &[(usize, f64)],
    n_substeps: usize,
    expect_splices: bool,
    label: &str,
) {
    let f = fixture_with(schedule, n_substeps);
    let (paths, pi, index) = exact_target(&f);

    // Non-vacuity (3): a target concentrated on one path passes for free.
    let ess = 1.0 / pi.iter().map(|p| p * p).sum::<f64>();
    eprintln!("support: {} paths, effective support size {:.1}", paths.len(), ess);
    assert!(ess > 8.0, "target is too concentrated to test anything (ESS {ess:.1})");

    let m: usize = std::env::var("CSMC_INVARIANCE_M")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(200_000);
    let n_particles: usize = std::env::var("CSMC_INVARIANCE_NP")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(5);

    let mut rng = StatefulRng::new(20260823);
    let mut tally = vec![0u64; paths.len()];
    eprintln!("--- {label} ---");
    let (mut n_proposed, mut n_accepted) = (0usize, 0usize);

    for i in 0..m {
        let x0 = draw_categorical(&pi, rng.uniform());
        let (x1, diag) = csmc_as(
            &f.compiled,
            &f.params,
            &f.obs,
            &paths[x0],
            n_particles,
            DT,
            &f.obs_model,
            0x5eed_0000_0000_0000u64.wrapping_add(i as u64),
            &f.obs_at_substep,
            EffectFiring::default(),
            true,
        )
        .expect("csmc_as");
        n_proposed += diag.n_as_proposed;
        n_accepted += diag.n_as_accepted;

        // Non-vacuity (2): an off-support return is a failure, never a drop.
        let k = key(&x1);
        let idx = *index.get(&k).unwrap_or_else(|| {
            panic!("csmc_as returned a path outside the model's support: flows {k:?}")
        });
        tally[idx] += 1;
    }

    // Non-vacuity (1): the splice path must actually have been exercised.
    eprintln!(
        "AS proposals {} ({:.2}/sweep), accepted {} ({:.1}%)",
        n_proposed,
        n_proposed as f64 / m as f64,
        n_accepted,
        100.0 * n_accepted as f64 / n_proposed.max(1) as f64
    );
    if expect_splices {
        assert!(
            n_accepted > m / 10,
            "ancestor sampling accepted only {n_accepted} splices over {m} sweeps — \
             the splice path this fixture exists to check is barely exercised"
        );
    } else {
        // gh#718 defect 2: this fixture has no substep where an ancestry is
        // drawn, so the correct behaviour is that ancestor sampling never runs.
        // Asserting that is what makes the fixture a REGRESSION test: remove the
        // `did_resample` gate and these counters go non-zero, firing here
        // immediately rather than waiting for the statistic below to accumulate
        // enough draws to notice.
        assert_eq!(
            n_proposed, 0,
            "{label}: ancestor sampling proposed {n_proposed} moves on a sweep that never \
             drew an ancestry — the did_resample gate is not in place (gh#718 defect 2)"
        );
        assert_eq!(n_accepted, 0, "{label}: a move was ACCEPTED with no ancestry drawn");
    }

    // Goodness of fit. Under H₀ the tally is multinomial(M, π), so bin `i` has
    // mean `M·π_i` and variance `M·π_i(1−π_i)`.
    let mf = m as f64;
    let mut chi2 = 0.0;
    let mut df = 0usize;
    let mut worst = (0.0f64, 0usize);
    for i in 0..paths.len() {
        let e = mf * pi[i];
        if e < 25.0 {
            continue;
        }
        df += 1;
        let z = (tally[i] as f64 - e) / (e * (1.0 - pi[i])).sqrt();
        chi2 += z * z;
        if z.abs() > worst.0 {
            worst = (z.abs(), i);
        }
    }
    assert!(df > 5, "too few well-populated bins ({df}) to test — raise CSMC_INVARIANCE_M");
    // Wilson–Hilferty: χ²_df standardised to an approximate N(0,1).
    let z_agg = ((chi2 / df as f64).powf(1.0 / 3.0) - (1.0 - 2.0 / (9.0 * df as f64)))
        / (2.0 / (9.0 * df as f64)).sqrt();

    eprintln!(
        "M={m} np={n_particles} bins={df} chi2={chi2:.1} z_agg={z_agg:.2} \
         worst bin |z|={:.2} (path {})",
        worst.0, worst.1
    );

    assert!(
        z_agg < 6.0,
        "{label}: csmc_as does not leave p(X | θ, y) invariant: goodness-of-fit against the \
         EXACT enumerated target gives χ²={chi2:.1} on {df} bins (z={z_agg:.2}), \
         worst single bin |z|={:.2}. π here has no Monte-Carlo error, so this is \
         not a ground-truth artefact (gh#718).",
        worst.0
    );
}

/// The sharpest available case for one specific mechanism: ancestor sampling
/// applied on top of IDENTITY free ancestry, with no later resampling step to
/// restore the reference's lineage to the free particles.
///
/// Two substeps, observation only at the end. Substep 0 has uniform weights but
/// every particle shares the deterministic initial state, so its ancestor move
/// is vacuous. Substep 1 also has uniform weights — the free particles have
/// diverged by then, so its ancestor move is real, it runs on identity
/// ancestry, and nothing after it can resample. That is the camdl analogue of
/// the exactly-enumerated counterexample in the gh#718 review.
#[test]
fn one_sweep_is_invariant_with_no_resampling_step_at_all() {
    check_invariance(&TRAP, TRAP_SUBSTEPS, false, "trap (no substep ever draws an ancestry)");
}
