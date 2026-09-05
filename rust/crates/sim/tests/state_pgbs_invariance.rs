//! Exact invariance of the state-space backward kernel (`csmc_bs`) —
//! the spike note's two toy gates, in the `csmc_exact_invariance` mold:
//!
//! ```text
//!   X₀ ~ π   (exact categorical draw over the enumerated support)
//!   X₁ = csmc_bs(X₀)
//!   H₀:  X₁ ~ π
//! ```
//!
//! Gate 1 (unique inversion): a tiny SIR — state deltas pin the flows, so
//! the Z-path and the flow record are in bijection and the tally runs over
//! flow keys exactly as the innovation kernel's test does.
//!
//! Gate 2 (ambiguous flows — the novel collapsed object): a diamond
//! A→B→D / A→C→D where many flow records share one Z-path. The kernel draws
//! the Z-path with flows marginalized and then RECONSTRUCTS flows from the
//! lattice conditional, so the returned flow record must be distributed as
//! the full flow-path posterior π_F — tallying flow keys therefore tests the
//! backward stitch AND the reconstruction jointly, which is strictly
//! stronger than tallying Z-paths.
//!
//! Non-vacuity, as in the parent harness: off-support returns are hard
//! failures, the effective support size is asserted, and for gate 2 the
//! enumerated support is asserted to contain Z-paths with MULTIPLE flow
//! records (else the ambiguity the gate exists for is absent).

use std::collections::HashMap;
use std::sync::Arc;

use ir::{
    expr::{BinOp, Expr},
    model::{
        Compartment, CompartmentKind, InitialConditions, OutputConfig, OutputSchedule,
        SimulationConfig,
    },
    parameter::Parameter,
    transition::{DrawMethod, StoichiometryEntry, Transition},
    Model,
};
use sim::compiled_model::CompiledModel;
use sim::inference::dense_cells;
use sim::inference::multi_stream_obs::{
    BoundObs, MultiStreamObsModel, StreamProjection, StreamSpec,
};
use sim::inference::particle_filter::Observation;
use sim::inference::pgas::{
    build_obs_at_substep, complete_data_loglik, EffectFiring, ObsAtSubstep, PGASTrajectory,
    SubstepRecord,
};
use sim::inference::state_pgbs::csmc_bs;
use sim::inference::state_transition::StateTransitionAnalysis;
use sim::rng::StatefulRng;

const DT: f64 = 1.0;

fn prevalence_obs_block(comp: &str) -> ir::observation::ObservationModel {
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
        projection: Projection::CurrentPop(comp.into()),
        projection_state_grad: Default::default(),
        likelihood: Likelihood::Poisson(PoissonLikelihood { rate: ir::Diffable::new(rate) }),
    }
}

struct Fixture {
    compiled: Arc<CompiledModel>,
    params: Vec<f64>,
    obs: Vec<Observation>,
    obs_model: MultiStreamObsModel,
    obs_at_substep: ObsAtSubstep,
    analysis: StateTransitionAnalysis,
    initial_counts: Vec<i64>,
    n_substeps: usize,
}

/// Flow-key identity of a trajectory (initial state is deterministic in both
/// fixtures, so flows identify the full record).
fn key(traj: &PGASTrajectory) -> Vec<u64> {
    traj.substeps.iter().flat_map(|r| r.flows.iter().copied()).collect()
}

/// Enumerate every flow path (per-substep flows bounded by source counts),
/// score with `complete_data_loglik`, normalize.
fn exact_target(
    f: &Fixture,
    max_flow: impl Fn(&[i64], usize) -> u64,
) -> (Vec<PGASTrajectory>, Vec<f64>, HashMap<Vec<u64>, usize>) {
    let n_tr = f.compiled.model.transitions.len();
    let mut stoich = vec![vec![0i64; n_tr]; f.initial_counts.len()];
    for (j, s) in f.compiled.transition_stoich.iter().enumerate() {
        for &(local, delta) in s {
            stoich[local][j] += delta;
        }
    }
    let advance = |state: &[i64], flows: &[u64]| -> Vec<i64> {
        (0..state.len())
            .map(|c| state[c] + (0..n_tr).map(|j| stoich[c][j] * flows[j] as i64).sum::<i64>())
            .collect()
    };

    let mut out: Vec<PGASTrajectory> = Vec::new();
    let mut stack: Vec<(Vec<i64>, Vec<SubstepRecord>)> =
        vec![(f.initial_counts.clone(), Vec::new())];
    while let Some((state, recs)) = stack.pop() {
        let s = recs.len();
        if s == f.n_substeps {
            out.push(PGASTrajectory {
                initial_counts: f.initial_counts.clone(),
                substeps: recs,
            });
            continue;
        }
        // odometer over per-transition flows bounded by max_flow
        let caps: Vec<u64> = (0..n_tr).map(|j| max_flow(&state, j)).collect();
        let mut flows = vec![0u64; n_tr];
        'enumerate: loop {
            let after = advance(&state, &flows);
            if after.iter().all(|&x| x >= 0) {
                let mut next = recs.clone();
                next.push(SubstepRecord {
                    counts_before: state.clone(),
                    counts_after: after.clone(),
                    flows: flows.clone(),
                    gammas: Vec::new(),
                    t0: s as f64 * DT,
                    dt_substep: DT,
                });
                stack.push((after, next));
            }
            let mut i = 0;
            loop {
                if i == n_tr {
                    break 'enumerate;
                }
                flows[i] += 1;
                if flows[i] <= caps[i] {
                    break;
                }
                flows[i] = 0;
                i += 1;
            }
        }
    }

    let mut paths = Vec::new();
    let mut logp = Vec::new();
    for traj in out {
        let ll = complete_data_loglik(
            &f.compiled, &traj, &f.params, &f.obs, DT, &f.obs_model, &f.obs_at_substep,
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
    assert_eq!(index.len(), paths.len(), "duplicate flow keys in enumeration");
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

fn check_invariance(f: &Fixture, paths: &[PGASTrajectory], pi: &[f64],
                    index: &HashMap<Vec<u64>, usize>, label: &str) {
    let ess = 1.0 / pi.iter().map(|p| p * p).sum::<f64>();
    eprintln!("--- {label}: support {} paths, ESS {ess:.1} ---", paths.len());
    assert!(ess > 8.0, "target too concentrated to test anything (ESS {ess:.1})");

    let m: usize = std::env::var("BS_INVARIANCE_M")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(120_000);
    let n_particles = 5usize;

    let mut rng = StatefulRng::new(20260902);
    let mut tally = vec![0u64; paths.len()];
    let mut renewal_sum = 0.0;
    for i in 0..m {
        let x0 = draw_categorical(pi, rng.uniform());
        let (x1, diag) = csmc_bs(
            &f.compiled,
            &f.params,
            &paths[x0],
            n_particles,
            DT,
            &f.obs_model,
            0xb5ed_0000_0000_0000u64.wrapping_add(i as u64),
            &f.obs_at_substep,
            EffectFiring::default(),
            &f.analysis,
            sim::rng::BinomialAlgorithm::Btpe,
        )
        .expect("csmc_bs");
        renewal_sum += diag.trajectory_renewal;
        let k = key(&x1);
        let idx = *index.get(&k).unwrap_or_else(|| {
            panic!("csmc_bs returned a path outside the support: flows {k:?}")
        });
        tally[idx] += 1;
    }
    eprintln!("mean renewal {:.3}", renewal_sum / m as f64);
    assert!(
        renewal_sum / m as f64 > 0.05,
        "{label}: the kernel never moves — invariance would be vacuous"
    );

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
    assert!(df > 5, "too few well-populated bins ({df}) — raise BS_INVARIANCE_M");
    let z_agg = ((chi2 / df as f64).powf(1.0 / 3.0) - (1.0 - 2.0 / (9.0 * df as f64)))
        / (2.0 / (9.0 * df as f64)).sqrt();
    eprintln!("M={m} bins={df} chi2={chi2:.1} z_agg={z_agg:.2} worst |z|={:.2}", worst.0);
    assert!(
        z_agg < 6.0,
        "{label}: csmc_bs does not leave p(X | θ, y) invariant \
         (χ²={chi2:.1} on {df} bins, z={z_agg:.2}, worst |z|={:.2})",
        worst.0
    );
}

// ── Gate 1: unique inversion (tiny SIR from the golden) ─────────────────────

fn sir_fixture() -> Fixture {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../../ocaml/golden/sir_basic.ir.json");
    let json = std::fs::read_to_string(&path).expect("read sir_basic golden");
    let mut m = ir::from_str(&json).expect("parse sir_basic");
    m.observations = vec![prevalence_obs_block("I")];
    m.simulation.t_start = 0.0;
    let n_substeps = 4usize;
    m.simulation.t_end = n_substeps as f64 * DT;
    for p in &mut m.parameters {
        let v = match p.name.as_str() {
            "beta" => 1.2,
            "gamma" => 0.5,
            "N0" => 6.0,
            "I0" => 2.0,
            other => panic!("unexpected parameter {other}"),
        };
        p.value = ir::parameter::ParamValue::Fixed { value: v };
    }
    let compiled = Arc::new(CompiledModel::new(m).expect("compile sir_basic"));
    let params = compiled.default_params.clone();
    let (init, _) = compiled.initial_state_mean(&params).expect("initial state");
    let initial_counts = init.counts.clone();

    let schedule = [(0usize, 3.0), (1, 3.0), (2, 2.0), (3, 2.0)];
    let obs: Vec<Observation> = schedule
        .iter()
        .map(|&(s, v)| Observation { time: ((s + 1) as f64) * DT, value: v })
        .collect();
    let obs_model = MultiStreamObsModel::new(
        BoundObs::bind(vec![StreamSpec::dense(
            StreamProjection::IntCompSum(vec![1]),
            compiled.model.observations[0].clone(),
            dense_cells(obs.iter().map(|o| o.value).collect()),
            obs.iter().map(|o| o.time).collect(),
        )])
        .unwrap()
        .0,
        compiled.clone(),
    )
    .unwrap();
    let obs_at_substep =
        build_obs_at_substep(&obs, compiled.model.simulation.t_start, DT).expect("obs map");
    let analysis = StateTransitionAnalysis::from_model(&compiled, &obs_model).expect("analysis");
    assert_eq!(analysis.n_free_dims(), 0, "SIR must invert uniquely");
    Fixture { compiled, params, obs, obs_model, obs_at_substep, analysis, initial_counts, n_substeps }
}

#[test]
fn one_bs_sweep_is_invariant_on_the_unique_inversion_toy() {
    let f = sir_fixture();
    let (paths, pi, index) = exact_target(&f, |state, j| {
        // infection bounded by S, recovery by I
        if j == 0 { state[0].max(0) as u64 } else { state[1].max(0) as u64 }
    });
    check_invariance(&f, &paths, &pi, &index, "unique-inversion SIR");
}

// ── Gate 2: ambiguous flows (the diamond) ───────────────────────────────────

fn diamond_compiled() -> Arc<CompiledModel> {
    let comps = [("A", 3.0), ("B", 1.0), ("C", 1.0), ("D", 0.0)];
    let trs = [
        ("ab", "A", "B", "r1", 0.5),
        ("ac", "A", "C", "r2", 0.4),
        ("bd", "B", "D", "r3", 0.6),
        ("cd", "C", "D", "r4", 0.7),
    ];
    let transitions = trs
        .iter()
        .map(|(name, from, to, p, _)| Transition {
            rate_state_grad: Default::default(),
            name: (*name).into(),
            stoichiometry: vec![
                StoichiometryEntry((*from).into(), -1),
                StoichiometryEntry((*to).into(), 1),
            ],
            rate: Expr::bin_op(BinOp::Mul, Expr::param(*p), Expr::pop(*from)),
            metadata: None,
            draw_method: DrawMethod::Poisson,
            rate_grad: Default::default(),
            lineage: None,
        })
        .collect();
    let m = Model {
        ic_grad: Default::default(),
        name: "bs_diamond".into(),
        version: "0.3".into(),
        time_unit: "days".into(),
        description: None,
        origin: None,
        origin_rata_die: None,
        compartments: comps
            .iter()
            .map(|(n, _)| Compartment { name: (*n).into(), kind: CompartmentKind::Integer })
            .collect(),
        transitions,
        ode_equations: vec![],
        time_functions: vec![],
        tables: vec![],
        interventions: vec![],
        observations: vec![prevalence_obs_block("D")],
        bindings: vec![],
        per_eval_bindings: vec![],
        parameters: trs
            .iter()
            .map(|(_, _, _, p, v)| Parameter {
                name: (*p).into(),
                value: ir::parameter::ParamValue::Fixed { value: *v },
                param_kind: None,
                param_dim: None,
            })
            .collect(),
        initial_conditions: InitialConditions::constants(
            comps.iter().map(|(n, v)| ((*n).into(), *v)).collect::<HashMap<_, _>>(),
        ),
        output: OutputConfig {
            times: OutputSchedule::AtTimes(vec![1.0]),
            format: "tsv".into(),
            trajectory: true,
            observations: false,
        },
        simulation: SimulationConfig {
            t_start: 0.0,
            t_end: 3.0,
            time_semantics: "continuous".into(),
            dt: Some(DT),
            rng_seed: Some(1),
            integrator: Default::default(),
            t_end_anchor: None,
        },
        presets: vec![],
        model_structure: None,
        balance: None,
        identity_tracked_compartments: vec![],
        quantities: vec![],
        contrasts: vec![],
    };
    Arc::new(CompiledModel::new(m).expect("compile diamond"))
}

fn diamond_fixture() -> Fixture {
    let compiled = diamond_compiled();
    let params = compiled.default_params.clone();
    let initial_counts = vec![3i64, 1, 1, 0];
    let n_substeps = 3usize;

    let schedule = [(0usize, 1.0), (1, 1.0), (2, 2.0)];
    let obs: Vec<Observation> = schedule
        .iter()
        .map(|&(s, v)| Observation { time: ((s + 1) as f64) * DT, value: v })
        .collect();
    let obs_model = MultiStreamObsModel::new(
        BoundObs::bind(vec![StreamSpec::dense(
            StreamProjection::IntCompSum(vec![3]), // prevalence of D
            compiled.model.observations[0].clone(),
            dense_cells(obs.iter().map(|o| o.value).collect()),
            obs.iter().map(|o| o.time).collect(),
        )])
        .unwrap()
        .0,
        compiled.clone(),
    )
    .unwrap();
    let obs_at_substep =
        build_obs_at_substep(&obs, compiled.model.simulation.t_start, DT).expect("obs map");
    let analysis = StateTransitionAnalysis::from_model(&compiled, &obs_model).expect("analysis");
    assert_eq!(analysis.n_free_dims(), 1, "the diamond must carry its ambiguity");
    Fixture { compiled, params, obs, obs_model, obs_at_substep, analysis, initial_counts, n_substeps }
}

#[test]
fn one_bs_sweep_is_invariant_on_the_ambiguous_diamond() {
    let f = diamond_fixture();
    let (paths, pi, index) = exact_target(&f, |state, j| match j {
        0 | 1 => state[0].max(0) as u64,
        2 => state[1].max(0) as u64,
        _ => state[2].max(0) as u64,
    });
    // Non-vacuity: the support must contain Z-paths with MULTIPLE flow
    // records, or the collapsed sum this gate exists for is never exercised.
    let mut z_multiplicity: HashMap<Vec<i64>, usize> = HashMap::new();
    for t in &paths {
        let zkey: Vec<i64> =
            t.substeps.iter().flat_map(|r| r.counts_after.iter().copied()).collect();
        *z_multiplicity.entry(zkey).or_insert(0) += 1;
    }
    let ambiguous = z_multiplicity.values().filter(|&&n| n > 1).count();
    assert!(
        ambiguous > 5,
        "only {ambiguous} Z-paths have multiple flow records — the fixture is \
         not exercising the collapsed marginal"
    );
    check_invariance(&f, &paths, &pi, &index, "ambiguous-flow diamond");
}


// ── Gate 3: ambiguity + a load-bearing accumulator ──────────────────────────
//
// The incidence stream sums {ab, ac} — a total the diamond's null direction
// preserves (H·v = 0), so the flow ambiguity SURVIVES while the kernel's
// accumulator machinery becomes load-bearing: bins are folded per substep,
// the first bin spans two substeps (carry across an unobserved boundary),
// the reference slot's accumulator path is pinned, and the backward weights'
// `d_acc` must respect the reset convention. Neither earlier gate touches
// any of that (both are prevalence-only, n_streams = 0).

fn diamond_acc_fixture() -> Fixture {
    let compiled = diamond_compiled();
    let params = compiled.default_params.clone();
    let initial_counts = vec![3i64, 1, 1, 0];
    let n_substeps = 3usize;

    // Stream 1: prevalence of D at every substep (weights vary everywhere).
    let prev_sched = [(0usize, 1.0), (1, 1.0), (2, 2.0)];
    let prev_obs: Vec<Observation> = prev_sched
        .iter()
        .map(|&(s, v)| Observation { time: ((s + 1) as f64) * DT, value: v })
        .collect();
    // Stream 2: incidence of ab+ac, bins closing at t = 2 (substeps 0–1) and
    // t = 3 (substep 2).
    let inc_times = vec![2.0, 3.0];
    let inc_values = vec![2.0, 1.0];

    let mut inc_block = prevalence_obs_block("D");
    inc_block.name = "a_exits".into();
    inc_block.source = "a_exits".into();

    let obs_model = MultiStreamObsModel::new(
        BoundObs::bind(vec![
            StreamSpec::dense(
                StreamProjection::IntCompSum(vec![3]),
                compiled.model.observations[0].clone(),
                dense_cells(prev_obs.iter().map(|o| o.value).collect()),
                prev_obs.iter().map(|o| o.time).collect(),
            ),
            StreamSpec::dense(
                StreamProjection::FlowSum(vec![0, 1]), // ab + ac
                inc_block,
                dense_cells(inc_values),
                inc_times.clone(),
            ),
        ])
        .unwrap()
        .0,
        compiled.clone(),
    )
    .unwrap();

    // Union observation axis: every time either stream scores.
    let obs: Vec<Observation> = [1.0, 2.0, 3.0]
        .iter()
        .map(|&t| Observation { time: t, value: 0.0 })
        .collect();
    let obs_at_substep =
        build_obs_at_substep(&obs, compiled.model.simulation.t_start, DT).expect("obs map");
    let analysis = StateTransitionAnalysis::from_model(&compiled, &obs_model).expect("analysis");
    assert_eq!(
        analysis.n_free_dims(),
        1,
        "the ab+ac stream must NOT kill the diamond's ambiguity (H·v = 0)"
    );
    Fixture { compiled, params, obs, obs_model, obs_at_substep, analysis, initial_counts, n_substeps }
}

#[test]
fn one_bs_sweep_is_invariant_with_a_load_bearing_accumulator() {
    let f = diamond_acc_fixture();
    let (paths, pi, index) = exact_target(&f, |state, j| match j {
        0 | 1 => state[0].max(0) as u64,
        2 => state[1].max(0) as u64,
        _ => state[2].max(0) as u64,
    });
    let mut z_multiplicity: HashMap<Vec<i64>, usize> = HashMap::new();
    for t in &paths {
        let zkey: Vec<i64> =
            t.substeps.iter().flat_map(|r| r.counts_after.iter().copied()).collect();
        *z_multiplicity.entry(zkey).or_insert(0) += 1;
    }
    let ambiguous = z_multiplicity.values().filter(|&&n| n > 1).count();
    assert!(
        ambiguous > 5,
        "only {ambiguous} counts-paths have multiple flow records — ambiguity gone"
    );
    check_invariance(&f, &paths, &pi, &index, "diamond + accumulator");
}
