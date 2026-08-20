//! gh#607: does the ancestor-sampling accept/reject ratio equal the thing it
//! claims to be?
//!
//! `splice_log_ratio` is the whole correctness argument of the Metropolis step:
//! Lindsten, Jordan & Schön (2014, JMLR 15:2145-2184) Eq. (21) collapses to
//! `α = S_{i'}/S_N`, and `splice_log_ratio` computes `log S` centred on the
//! reference. Nothing structural detects a wrong `α` — a continuity assertion
//! and a byte-identity digest both hold for any scale factor on it. So pin it
//! against the ONE function the θ-move actually conditions on,
//! `complete_data_loglik`.
//!
//! The identity being asserted. Let `spliced` be the reference trajectory whose
//! substeps from `s` on are shifted by the constant offset `Δ` (exactly the path
//! an accepted splice produces), sharing the reference's prefix. Then
//!
//! ```text
//!   splice_log_ratio(s, Δ) = [cdl(spliced) − cdl(reference)]
//!                            − [td(x'_{s-1}+Δ) − td(x'_{s-1})]
//! ```
//!
//! The prefix terms cancel because the two trajectories share it; the substep-`s`
//! transition factor is subtracted because it cancels against the proposal
//! weight instead (it is already in `ancestor_log_w`), which is why
//! `splice_log_ratio`'s transition sum starts at `s+1`.
//!
//! Two observation kinds, because they exercise different halves:
//!
//! - **Interval** (`FlowSum`, incidence). The projection reads flows, which the
//!   offset does not touch, so `Δ` alone cannot move the observation term. What
//!   this pins instead is the ACCUMULATOR SEED: `complete_data_loglik` sums the
//!   bin from its start, so the substep chosen below sits strictly INSIDE a bin
//!   and the ratio must carry the partial sum already banked. Seeding at zero —
//!   scoring the straddled bin on the suffix flows alone — fails this.
//! - **Instant** (`IntCompSum`, prevalence). The projection reads compartment
//!   counts at the observation instant, so every observation term moves with
//!   `Δ`. This pins the shifted-state observation evaluation.
//!
//! What these do NOT pin, stated rather than papered over: the explicit
//! gamma-multiplier scoring. σ² may not reference compartment state, so with the
//! term SET unchanged the multiplier density is bit-identical on both sides and
//! contributes exactly zero to the ratio. The case where it does not cancel is a
//! term-set change, and this fixture's single overdispersed group turns any such
//! change into a rejection — which is what `gamma_desync_is_a_zero_density`
//! below pins.

use std::sync::Arc;

use sim::compiled_model::CompiledModel;
use sim::inference::dense_cells;
use sim::inference::multi_stream_obs::{BoundObs, MultiStreamObsModel, StreamProjection, StreamSpec};
use sim::inference::particle_filter::Observation;
use sim::inference::pgas::{
    build_obs_at_substep, complete_data_loglik, log_transition_density_substep, reference_baseline,
    simulate_reference, splice_log_ratio, ObsAtSubstep, PGASTrajectory,
};
use sim::rng::StatefulRng;

const DT: f64 = 1.0;
const SEED: u64 = 20260819;
const S_IDX: usize = 0;
const I_IDX: usize = 1;
/// Absolute tolerance in nats. The two sides sum ~80 substeps of densities in
/// different orders, so they agree to floating-point round-off, not bitwise.
const TOL: f64 = 1e-9;

/// Which projection the scored stream uses.
enum Kind {
    /// `incidence(infection)` — Interval; reads flows.
    Interval,
    /// `I` — Instant; reads compartment counts at the observation instant.
    Prevalence,
}

fn obs_block(kind: &Kind) -> ir::observation::ObservationModel {
    use ir::expr::*;
    use ir::observation::*;
    let rate = Expr::Projected(ProjectedExpr { projected: () });
    ObservationModel {
        name: "y".into(),
        source: "y".into(),
        columns: vec![
            ObsColumn { name: "time".into(), role: ColumnRole::Time },
            ObsColumn { name: "y".into(), role: ColumnRole::Value(ir::parameter::ParamKind::Count) },
        ],
        scored: "y".into(),
        emit_schedule: Some(ObservationSchedule::AtTimes(vec![])),
        stratum: vec![],
        projection: match kind {
            Kind::Interval => Projection::CumulativeFlow("infection".into()),
            Kind::Prevalence => Projection::CurrentPop("I".into()),
        },
        projection_state_grad: Default::default(),
        likelihood: Likelihood::Poisson(PoissonLikelihood { rate: ir::Diffable::new(rate) }),
    }
}

fn model(kind: &Kind) -> Arc<CompiledModel> {
    let json = std::fs::read_to_string("../../../ocaml/golden/sir_overdispersion.ir.json")
        .expect("read sir_overdispersion golden");
    let mut m = ir::from_str(&json).expect("parse");
    m.observations = vec![obs_block(kind)];
    for p in &mut m.parameters {
        if p.value.resolved_value().is_none() {
            let v = match p.name.as_str() {
                "beta" => 0.3,
                "gamma" => 0.1,
                "sigma_se" => 0.1,
                "N0" => 1000.0,
                "I0" => 10.0,
                _ => 0.5,
            };
            p.value = p.value.with_value(v);
        }
    }
    Arc::new(CompiledModel::new(m).expect("compile"))
}

struct Fixture {
    compiled: Arc<CompiledModel>,
    params: Vec<f64>,
    reference: PGASTrajectory,
    obs: Vec<Observation>,
    obs_model: MultiStreamObsModel,
    obs_at_substep: ObsAtSubstep,
}

fn fixture(kind: Kind) -> Fixture {
    let compiled = model(&kind);
    let params = compiled.default_params.clone();
    let t_end = compiled.model.simulation.t_end;
    let mut rng = StatefulRng::new(SEED);
    let reference = simulate_reference(&compiled, &params, t_end, DT, &mut rng).expect("reference");

    let mut cum: u64 = 0;
    let mut obs: Vec<Observation> = Vec::new();
    for (s, rec) in reference.substeps.iter().enumerate() {
        cum += rec.flows[0];
        let t = ((s + 1) as f64) * DT;
        if (t.round() as i64) % 7 == 0 {
            let value = match kind {
                Kind::Interval => cum as f64,
                Kind::Prevalence => rec.counts_after[I_IDX] as f64,
            };
            obs.push(Observation { time: t, value });
            cum = 0;
        }
    }
    assert!(obs.len() >= 3, "need several observation intervals");

    let projection = match kind {
        Kind::Interval => StreamProjection::FlowSum(vec![0]),
        Kind::Prevalence => StreamProjection::IntCompSum(vec![I_IDX]),
    };
    let obs_model = MultiStreamObsModel::new(
        BoundObs::bind(vec![StreamSpec::dense(
            projection,
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

    Fixture { compiled, params, reference, obs, obs_model, obs_at_substep }
}

/// The reference's own flow accumulation entering substep `s` — the partial bin
/// a splice at `s` inherits. Mirrors `complete_data_loglik`'s fold/reset
/// lifecycle, stopping before `s`.
fn accumulation_entering(f: &Fixture, s: usize) -> (Vec<u64>, Vec<u64>) {
    let n_tr = f.compiled.model.transitions.len();
    let mut cum = vec![0u64; n_tr];
    let mut acc = vec![0u64; f.obs_model.n_interval_streams()];
    for t in 0..s {
        for (i, &fl) in f.reference.substeps[t].flows.iter().enumerate() {
            cum[i] += fl;
        }
        if let Some(&obs_idx) = f.obs_at_substep.get(&t) {
            f.obs_model.fold_into_acc(&cum, &mut acc);
            cum.fill(0);
            f.obs_model.reset_due_acc(obs_idx, &mut acc);
        }
    }
    (cum, acc)
}

/// The reference with every substep from `s` on shifted by `offset` — the path
/// an accepted splice actually returns.
fn spliced(reference: &PGASTrajectory, s: usize, offset: &[i64]) -> PGASTrajectory {
    let mut out = reference.clone();
    for rec in out.substeps.iter_mut().skip(s) {
        for i in 0..offset.len() {
            rec.counts_before[i] += offset[i];
            rec.counts_after[i] += offset[i];
        }
    }
    out
}

fn cdl(f: &Fixture, traj: &PGASTrajectory) -> f64 {
    complete_data_loglik(
        &f.compiled, traj, &f.params, &f.obs, DT, &f.obs_model, &[], &f.obs_at_substep,
    )
    .expect("complete_data_loglik")
    .total
}

/// A substep strictly inside an observation bin, so the straddled bin's partial
/// accumulation is non-zero and the seed is load-bearing.
fn substep_inside_a_bin(f: &Fixture) -> usize {
    let s = 10;
    assert!(
        f.obs_at_substep.get(&(s - 1)).is_none() && f.obs_at_substep.get(&s).is_none(),
        "substep {s} must sit strictly inside a bin for this test to pin the seed"
    );
    let (cum, _) = accumulation_entering(f, s);
    assert!(
        cum.iter().sum::<u64>() > 0,
        "the straddled bin must have banked flow already, or the seed is not exercised"
    );
    s
}

fn check_ratio_matches_complete_data_loglik(kind: Kind, label: &str) {
    let f = fixture(kind);
    let s = substep_inside_a_bin(&f);
    let n_comp = f.reference.initial_counts.len();

    // A population-conserving offset: one more susceptible, one fewer removed.
    // Small enough to stay admissible over the whole suffix (asserted below).
    let mut offset = vec![0i64; n_comp];
    offset[S_IDX] = 1;
    offset[n_comp - 1] = -1;

    let spliced_traj = spliced(&f.reference, s, &offset);
    let ll_ref = cdl(&f, &f.reference);
    let ll_spliced = cdl(&f, &spliced_traj);
    assert!(
        ll_ref.is_finite() && ll_spliced.is_finite(),
        "{label}: both paths must be on-support or the test is vacuous \
         (ref={ll_ref}, spliced={ll_spliced})"
    );

    // The substep-s transition factor cancels against the proposal weight, so
    // it is not part of the ratio — subtract it from the trajectory difference.
    let rec = &f.reference.substeps[s];
    let td_ref = log_transition_density_substep(
        &f.compiled, &rec.counts_before, &rec.flows, &rec.gammas, &f.params,
        rec.t0, rec.dt_substep, None,
    )
    .unwrap();
    let shifted_before: Vec<i64> =
        rec.counts_before.iter().zip(&offset).map(|(c, d)| c + d).collect();
    let td_spliced = log_transition_density_substep(
        &f.compiled, &shifted_before, &rec.flows, &rec.gammas, &f.params,
        rec.t0, rec.dt_substep, None,
    )
    .unwrap();
    assert!(td_ref.is_finite() && td_spliced.is_finite(), "{label}: substep-s density");

    let expected = (ll_spliced - ll_ref) - (td_spliced - td_ref);

    let baseline = reference_baseline(
        &f.compiled, &f.reference, &f.params, &f.obs_model, &f.obs_at_substep, None,
    )
    .expect("baseline");
    let (cum_seed, acc_seed) = accumulation_entering(&f, s);
    let got = splice_log_ratio(
        &f.compiled, &f.reference, &f.params, &f.obs_model, &f.obs_at_substep, None,
        &baseline, s, &offset, &cum_seed, &acc_seed,
    )
    .expect("splice_log_ratio");

    // Non-vacuity: a ratio of zero would satisfy the assertion trivially if the
    // offset changed nothing.
    assert!(
        expected.abs() > 1e-3,
        "{label}: the offset must actually move the density, else this passes \
         for free (expected={expected})"
    );
    assert!(
        (got - expected).abs() < TOL,
        "{label}: splice_log_ratio = {got} but the complete-data likelihood of \
         the spliced path says {expected} (gap {:.3e} nats). The accept/reject \
         ratio is not the density the θ-move conditions on — gh#607.",
        (got - expected).abs()
    );
}

#[test]
fn ratio_matches_complete_data_loglik_on_an_interval_stream() {
    check_ratio_matches_complete_data_loglik(Kind::Interval, "interval");
}

#[test]
fn ratio_matches_complete_data_loglik_on_a_prevalence_stream() {
    check_ratio_matches_complete_data_loglik(Kind::Prevalence, "prevalence");
}

/// The offset's own admissibility is not the ratio's business, but a splice that
/// lands off-support must score `−∞` on BOTH sides — the ratio and the
/// complete-data likelihood must refuse the same paths.
#[test]
fn an_off_support_offset_is_refused_by_both() {
    let f = fixture(Kind::Interval);
    let s = substep_inside_a_bin(&f);
    let n_comp = f.reference.initial_counts.len();

    // Drain the infectious compartment: the recorded recovery flows can no
    // longer come out of it.
    let mut offset = vec![0i64; n_comp];
    offset[I_IDX] = -f.reference.substeps[s].counts_before[I_IDX];
    assert!(offset[I_IDX] < 0, "fixture must have infectious individuals at s");

    let spliced_traj = spliced(&f.reference, s, &offset);
    assert_eq!(cdl(&f, &spliced_traj), f64::NEG_INFINITY, "cdl must refuse it");

    let baseline = reference_baseline(
        &f.compiled, &f.reference, &f.params, &f.obs_model, &f.obs_at_substep, None,
    )
    .expect("baseline");
    let (cum_seed, acc_seed) = accumulation_entering(&f, s);
    let got = splice_log_ratio(
        &f.compiled, &f.reference, &f.params, &f.obs_model, &f.obs_at_substep, None,
        &baseline, s, &offset, &cum_seed, &acc_seed,
    )
    .expect("splice_log_ratio");
    assert_eq!(
        got,
        f64::NEG_INFINITY,
        "the ratio must refuse a splice the complete-data likelihood refuses"
    );
}

/// The gamma multipliers are bound POSITIONALLY, in `step_one`'s push order, and
/// the density walk skips a source group whose compartment the offset has
/// emptied — WITHOUT advancing the gamma index. Every later overdispersed group
/// then reads the wrong multiplier, silently, because the read falls back to
/// `1.0` past the end of the slice.
///
/// The state chosen here is the exact trap: `S` emptied, but the recorded
/// infection flow at that substep is ZERO, so no `k > n` and no zero-rate
/// conflict fires — every other guard passes and the density comes back finite
/// and wrong unless the multiplier count itself is checked.
#[test]
fn gamma_desync_is_a_zero_density() {
    let f = fixture(Kind::Interval);

    // A substep with a recorded multiplier but no realized infection flow.
    let s = f
        .reference
        .substeps
        .iter()
        .position(|r| !r.gammas.is_empty() && r.flows[0] == 0 && r.counts_before[S_IDX] > 0)
        .expect("need a substep with a recorded gamma and zero infection flow");
    let rec = &f.reference.substeps[s];

    // Sanity: at its own state the record is on-support.
    let at_own = log_transition_density_substep(
        &f.compiled, &rec.counts_before, &rec.flows, &rec.gammas, &f.params,
        rec.t0, rec.dt_substep, None,
    )
    .unwrap();
    assert!(at_own.is_finite(), "negative control: the record scores finitely at its own state");

    // Empty the source the multiplier was drawn for. The group is now skipped,
    // so the state consumes zero multipliers while the record holds one.
    let mut emptied = rec.counts_before.clone();
    emptied[S_IDX] = 0;
    let at_emptied = log_transition_density_substep(
        &f.compiled, &emptied, &rec.flows, &rec.gammas, &f.params,
        rec.t0, rec.dt_substep, None,
    )
    .unwrap();
    assert_eq!(
        at_emptied,
        f64::NEG_INFINITY,
        "a state that consumes 0 of the {} recorded gamma multipliers cannot have \
         produced this record, so its density is zero — returning a finite number \
         here pairs later overdispersed groups with the wrong multiplier (gh#607)",
        rec.gammas.len()
    );
}
