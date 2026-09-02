//! Oracle tests for the state-transition density
//! (`sim::inference::state_transition`, spike note 2026-09-02).
//!
//! The module's one job is `p(Z'|Z) = Σ_{F≥0: SF=ΔX, HF=ΔA} p(F|Z)`. Every
//! test here checks it against BRUTE FORCE: enumerate all flow vectors up to
//! a cap, filter by the edge constraints, and log-sum-exp the EXISTING
//! innovation-conditional density. The brute force shares no code with the
//! module's linear algebra or lattice enumeration, so agreement pins the
//! collapse, the RREF solve, the bounded enumeration, and the merged-class
//! split-marginalization correction all at once.
//!
//! Toys cover the three structures the spike note names: unique inversion (a
//! chain), a genuine ambiguity diamond (nonzero nullspace — the novel
//! collapsed sum), an identical-stoichiometry merged class, and an
//! accumulator (`H`) constraint that removes ambiguity. A frequency test
//! checks the density against `step_one` itself, and the reconstruction draw
//! is checked for constraint satisfaction and correct frequencies.

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
use sim::inference::pgas::log_transition_density_substep;
use sim::inference::state_transition::{
    log_state_transition_density, sample_edge_flows, StateTransitionAnalysis,
};
use sim::rng::StatefulRng;

const DT: f64 = 1.0;
const T0: f64 = 0.0;

/// Build a compiled model from (compartments, transitions as
/// `(name, from, to, rate_param)`), all Poisson-drawn, rate = param · pop(from).
fn model(comps: &[(&str, f64)], trs: &[(&str, &str, &str, &str, f64)]) -> Arc<CompiledModel> {
    let transitions = trs
        .iter()
        .map(|(name, from, to, _p, _v)| Transition {
            rate_state_grad: Default::default(),
            name: (*name).into(),
            stoichiometry: vec![
                StoichiometryEntry((*from).into(), -1),
                StoichiometryEntry((*to).into(), 1),
            ],
            rate: Expr::bin_op(BinOp::Mul, Expr::param(trs.iter().find(|t| t.0 == *name).unwrap().3), Expr::pop(*from)),
            metadata: None,
            draw_method: DrawMethod::Poisson,
            rate_grad: Default::default(),
            lineage: None,
        })
        .collect();
    let mut params: Vec<Parameter> = Vec::new();
    for (_, _, _, p, v) in trs {
        if !params.iter().any(|q| q.name == **p) {
            params.push(Parameter {
                name: (*p).into(),
                value: ir::parameter::ParamValue::Fixed { value: *v },
                param_kind: None,
                param_dim: None,
            });
        }
    }
    let m = Model {
        ic_grad: Default::default(),
        name: "state_transition_oracle".into(),
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
        observations: vec![],
        bindings: vec![],
        per_eval_bindings: vec![],
        parameters: params,
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
            t_end: 1.0,
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
    Arc::new(CompiledModel::new(m).expect("compile oracle toy"))
}

/// An obs model with the given incidence streams (`FlowSum` over transition
/// names); empty ⇒ no interval streams and `H` has no rows.
fn obs_model(compiled: &Arc<CompiledModel>, streams: &[(&str, &[&str])]) -> MultiStreamObsModel {
    let mut prev = ir::observation::ObservationModel {
        name: String::new(),
        source: String::new(),
        columns: vec![
            ir::observation::ObsColumn { name: "time".into(), role: ir::observation::ColumnRole::Time },
            ir::observation::ObsColumn {
                name: "v".into(),
                role: ir::observation::ColumnRole::Value(ir::parameter::ParamKind::Count),
            },
        ],
        scored: "v".into(),
        emit_schedule: Some(ir::observation::ObservationSchedule::AtTimes(vec![])),
        stratum: vec![],
        projection: ir::observation::Projection::CurrentPop(
            compiled.model.compartments[0].name.clone(),
        ),
        projection_state_grad: Default::default(),
        likelihood: ir::observation::Likelihood::Poisson(ir::observation::PoissonLikelihood {
            rate: ir::Diffable::new(Expr::Projected(ir::expr::ProjectedExpr { projected: () })),
        }),
    };
    let tidx: HashMap<&str, usize> = compiled
        .model
        .transitions
        .iter()
        .enumerate()
        .map(|(i, t)| (t.name.as_str(), i))
        .collect();
    let specs: Vec<StreamSpec> = streams
        .iter()
        .map(|(name, trs)| {
            prev.name = (*name).into();
            prev.source = (*name).into();
            StreamSpec::dense(
                StreamProjection::FlowSum(trs.iter().map(|t| tidx[t]).collect()),
                prev.clone(),
                dense_cells(vec![0.0]),
                vec![1.0],
            )
        })
        .collect();
    if specs.is_empty() {
        // At least one instant stream so BoundObs::bind has content.
        prev.name = "prev".into();
        prev.source = "prev".into();
        let spec = StreamSpec::dense(
            StreamProjection::IntCompSum(vec![0]),
            prev.clone(),
            dense_cells(vec![0.0]),
            vec![1.0],
        );
        return MultiStreamObsModel::new(
            BoundObs::bind(vec![spec]).unwrap().0,
            compiled.clone(),
        )
        .unwrap();
    }
    MultiStreamObsModel::new(BoundObs::bind(specs).unwrap().0, compiled.clone()).unwrap()
}

/// Brute-force oracle: log Σ over all flow vectors `f ∈ [0, cap]^n_tr` with
/// `S·f = ΔX` and `H·f = ΔA` of `exp(log_transition_density_substep(f))`.
fn brute_force(
    compiled: &CompiledModel,
    h_rows: &[Vec<usize>],
    counts_before: &[i64],
    d_counts: &[i64],
    d_acc: &[i64],
    params: &[f64],
    cap: u64,
) -> f64 {
    let n_tr = compiled.model.transitions.len();
    let n_comp = counts_before.len();
    let mut flows = vec![0u64; n_tr];
    let mut terms: Vec<f64> = Vec::new();
    let mut stoich = vec![vec![0i64; n_tr]; n_comp];
    for (j, s) in compiled.transition_stoich.iter().enumerate() {
        for &(local, delta) in s {
            stoich[local][j] += delta;
        }
    }
    loop {
        let matches = (0..n_comp).all(|c| {
            (0..n_tr).map(|j| stoich[c][j] * flows[j] as i64).sum::<i64>() == d_counts[c]
        }) && h_rows.iter().zip(d_acc).all(|(row, &da)| {
            row.iter().map(|&j| flows[j] as i64).sum::<i64>() == da
        });
        if matches {
            if let Ok(td) = log_transition_density_substep(
                compiled, counts_before, &flows, &[], params, T0, DT, None,
            ) {
                if td > f64::NEG_INFINITY {
                    terms.push(td);
                }
            }
        }
        // odometer
        let mut i = 0;
        loop {
            if i == n_tr {
                let mx = terms.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                if terms.is_empty() {
                    return f64::NEG_INFINITY;
                }
                return mx + terms.iter().map(|v| (v - mx).exp()).sum::<f64>().ln();
            }
            flows[i] += 1;
            if flows[i] <= cap {
                break;
            }
            flows[i] = 0;
            i += 1;
        }
    }
}

fn assert_close(a: f64, b: f64, what: &str) {
    if a == f64::NEG_INFINITY && b == f64::NEG_INFINITY {
        return;
    }
    assert!(
        (a - b).abs() < 1e-9,
        "{what}: module {a} vs brute force {b} (diff {})",
        (a - b).abs()
    );
}

// ── 1. Unique inversion: a chain has nullspace zero ─────────────────────────

#[test]
fn chain_density_matches_brute_force_and_has_no_free_dims() {
    let compiled = model(
        &[("X", 20.0), ("Y", 5.0), ("Z", 0.0)],
        &[("drain", "X", "Y", "mu", 0.3), ("out", "Y", "Z", "nu", 0.2)],
    );
    let om = obs_model(&compiled, &[]);
    let an = StateTransitionAnalysis::from_model(&compiled, &om).unwrap();
    assert_eq!(an.n_free_dims(), 0, "a chain must have a unique flow inversion");
    let params = compiled.default_params.clone();
    let before = [20i64, 5, 0];
    for d in [[-3i64, 2, 1], [0, 0, 0], [-1, 1, 0], [-2, -1, 3], [1, -1, 0]] {
        let got =
            log_state_transition_density(&compiled, &an, &before, &d, &[], &params, T0, DT, None)
                .unwrap();
        let want = brute_force(&compiled, &[], &before, &d, &[], &params, 8);
        assert_close(got, want, &format!("chain edge {d:?}"));
    }
}

// ── 2. Diamond: nonzero nullspace, the collapsed sum ────────────────────────

fn diamond() -> Arc<CompiledModel> {
    model(
        &[("A", 12.0), ("B", 4.0), ("C", 3.0), ("D", 0.0)],
        &[
            ("ab", "A", "B", "r1", 0.4),
            ("ac", "A", "C", "r2", 0.3),
            ("bd", "B", "D", "r3", 0.5),
            ("cd", "C", "D", "r4", 0.6),
        ],
    )
}

#[test]
fn diamond_density_matches_brute_force_over_the_lattice() {
    let compiled = diamond();
    let om = obs_model(&compiled, &[]);
    let an = StateTransitionAnalysis::from_model(&compiled, &om).unwrap();
    assert_eq!(an.n_free_dims(), 1, "the diamond has exactly one ambiguity direction");
    let params = compiled.default_params.clone();
    let before = [12i64, 4, 3, 0];
    for d in [
        [-4i64, 1, 1, 2],
        [-2, 2, 0, 0],
        [-3, -1, 1, 3],
        [0, 0, 0, 0],
        [-5, 3, 2, 0],
        [-1, 1, 1, -1],
    ] {
        let got =
            log_state_transition_density(&compiled, &an, &before, &d, &[], &params, T0, DT, None)
                .unwrap();
        let want = brute_force(&compiled, &[], &before, &d, &[], &params, 9);
        assert_close(got, want, &format!("diamond edge {d:?}"));
    }
}

// ── 3. Accumulator constraint kills the diamond's ambiguity ────────────────

#[test]
fn an_incidence_stream_on_one_arm_makes_the_diamond_unique() {
    let compiled = diamond();
    let om = obs_model(&compiled, &[("obs_ab", &["ab"])]);
    let an = StateTransitionAnalysis::from_model(&compiled, &om).unwrap();
    assert_eq!(
        an.n_free_dims(),
        0,
        "observing one arm's flow pins the whole diamond"
    );
    let params = compiled.default_params.clone();
    let before = [12i64, 4, 3, 0];
    let h_rows: Vec<Vec<usize>> = vec![vec![0]]; // transition "ab" is index 0
    for (d, da) in [([-4i64, 1, 1, 2], [3i64]), ([-4, 1, 1, 2], [1]), ([-2, 2, 0, 0], [2])] {
        let got =
            log_state_transition_density(&compiled, &an, &before, &d, &da, &params, T0, DT, None)
                .unwrap();
        let want = brute_force(&compiled, &h_rows, &before, &d, &da, &params, 9);
        assert_close(got, want, &format!("constrained diamond edge {d:?} acc {da:?}"));
    }
}

// ── 4. Merged class: identical stoichiometry marginalizes exactly ──────────

#[test]
fn identical_stoichiometry_pair_marginalizes_exactly() {
    let compiled = model(
        &[("A", 15.0), ("B", 0.0)],
        &[("fast", "A", "B", "r1", 0.5), ("slow", "A", "B", "r2", 0.1)],
    );
    let om = obs_model(&compiled, &[]);
    let an = StateTransitionAnalysis::from_model(&compiled, &om).unwrap();
    assert_eq!(an.n_free_dims(), 0, "the merged pair leaves no free dimension");
    assert_eq!(an.merged_classes().len(), 1);
    let params = compiled.default_params.clone();
    let before = [15i64, 0];
    for d in [[-4i64, 4], [0, 0], [-1, 1], [-7, 7]] {
        let got =
            log_state_transition_density(&compiled, &an, &before, &d, &[], &params, T0, DT, None)
                .unwrap();
        let want = brute_force(&compiled, &[], &before, &d, &[], &params, 8);
        assert_close(got, want, &format!("merged-pair edge {d:?}"));
    }
}

// ── 5. The density agrees with step_one's own frequencies ──────────────────

#[test]
fn density_matches_forward_simulation_frequencies_on_the_diamond() {
    use sim::chain_binomial::{step_one, StepScratch};
    let compiled = diamond();
    let om = obs_model(&compiled, &[]);
    let an = StateTransitionAnalysis::from_model(&compiled, &om).unwrap();
    let params = compiled.default_params.clone();
    let before = [12i64, 4, 3, 0];
    let n_tr = compiled.model.transitions.len();

    let n_draws = 40_000usize;
    let mut freq: HashMap<[i64; 4], usize> = HashMap::new();
    let mut rng = StatefulRng::new(97);
    let mut scratch = StepScratch::new(&compiled);
    let mut real = sim::state::RealState::new(0);
    for _ in 0..n_draws {
        let mut counts = before.to_vec();
        let mut flows = vec![0u64; n_tr];
        step_one(
            &compiled, &mut counts, &mut flows, &mut real, &params, T0, DT, None,
            &mut rng, &mut scratch,
        )
        .unwrap();
        let d = [
            counts[0] - before[0],
            counts[1] - before[1],
            counts[2] - before[2],
            counts[3] - before[3],
        ];
        *freq.entry(d).or_insert(0) += 1;
    }
    // Compare the most frequent end states: empirical vs density, generous
    // tolerance (Monte-Carlo noise at 40k draws).
    let mut checked = 0;
    for (d, n) in freq.iter().filter(|(_, &n)| n > 400) {
        let emp = *n as f64 / n_draws as f64;
        let got = log_state_transition_density(
            &compiled, &an, &before, d, &[], &params, T0, DT, None,
        )
        .unwrap()
        .exp();
        assert!(
            (got - emp).abs() < 0.20 * emp.max(0.02),
            "state {d:?}: density {got:.4} vs empirical {emp:.4}"
        );
        checked += 1;
    }
    assert!(checked >= 5, "too few high-frequency states to make this test meaningful");
}

// ── 6. Reconstruction: constraints hold and frequencies match the terms ────

#[test]
fn sampled_edge_flows_satisfy_constraints_and_follow_the_conditional() {
    let compiled = diamond();
    let om = obs_model(&compiled, &[]);
    let an = StateTransitionAnalysis::from_model(&compiled, &om).unwrap();
    let params = compiled.default_params.clone();
    let before = [12i64, 4, 3, 0];
    let d = [-4i64, 1, 1, 2];

    let mut rng = StatefulRng::new(4242);
    let mut freq: HashMap<Vec<u64>, usize> = HashMap::new();
    let n_draws = 30_000usize;
    for _ in 0..n_draws {
        let f = sample_edge_flows(
            &compiled, &an, &before, &d, &[], &params, T0, DT, None, &mut rng,
        )
        .unwrap()
        .expect("edge is feasible");
        // constraint check: S·F = ΔX
        let mut got = [0i64; 4];
        for (j, s) in compiled.transition_stoich.iter().enumerate() {
            for &(local, delta) in s {
                got[local] += delta * f[j] as i64;
            }
        }
        assert_eq!(got, d, "sampled flows must reproduce the edge delta");
        *freq.entry(f).or_insert(0) += 1;
    }
    // Frequencies must match p(F|Z)/Σ p(F'|Z) over the compatible set.
    let total_log = log_state_transition_density(
        &compiled, &an, &before, &d, &[], &params, T0, DT, None,
    )
    .unwrap();
    for (f, n) in freq.iter().filter(|(_, &n)| n > 300) {
        let emp = *n as f64 / n_draws as f64;
        let lp = log_transition_density_substep(
            &compiled, &before, f, &[], &params, T0, DT, None,
        )
        .unwrap();
        let cond = (lp - total_log).exp();
        assert!(
            (cond - emp).abs() < 0.15 * cond.max(0.02),
            "flow {f:?}: conditional {cond:.4} vs empirical {emp:.4}"
        );
    }
}

// ── 7. The prototype class gate refuses loudly ─────────────────────────────

#[test]
fn overdispersed_models_are_refused_with_the_spike_reference() {
    let compiled = model(
        &[("A", 10.0), ("B", 0.0)],
        &[("ab", "A", "B", "r1", 0.4)],
    );
    // Rebuild with an overdispersed draw method.
    let mut m = (*compiled.model).clone();
    m.transitions[0].draw_method = DrawMethod::Overdispersed {
        sigma_sq: Expr::const_(0.5),
        sigma_sq_grad: Default::default(),
    };
    let od = Arc::new(CompiledModel::new(m).unwrap());
    let om = obs_model(&od, &[]);
    let Err(err) = StateTransitionAnalysis::from_model(&od, &om) else {
        panic!("overdispersed model must be refused");
    };
    assert!(
        err.to_string().contains("overdispersed"),
        "refusal must name the unsupported construct: {err}"
    );
}
