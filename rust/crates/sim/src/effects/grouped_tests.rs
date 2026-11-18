//! The grouped transfer: the allocation function as a pure function, then the
//! action through the discrete resolver, the ODE's continuous arena and the
//! sensitivity rule, and beside other actions at one instant.
//!
//! The fixture has a staged compartment written as three integer cells `E1`,
//! `E2`, `E3`, two unstaged integer compartments `R` and `Q`, and the same
//! shape in real compartments (`X1`, `X2`, `X3`, `Y`).

use super::*;
use ir::{
    deriv::DerivEntry,
    expr::Expr,
    intervention::{
        AbsoluteTransfer, AddAction, FireSource, FractionTransfer, InterventionKind,
        InterventionSchedule, SetAction,
    },
    model::{
        Compartment, CompartmentKind, InitialConditions, OutputConfig, OutputSchedule,
        SimulationConfig,
    },
    parameter::Parameter,
    transition::{DrawMethod, StoichiometryEntry, Transition},
    Model,
};

const INT_NAMES: [&str; 5] = ["E1", "E2", "E3", "R", "Q"];
const REAL_NAMES: [&str; 4] = ["X1", "X2", "X3", "Y"];

/// One effect per entry of `effects`, each firing at t = 1. Parameter `p` is
/// 0.4 (model index 0).
fn model(effects: Vec<(&str, Vec<Action>)>) -> CompiledModel {
    let compartments = INT_NAMES
        .iter()
        .map(|n| Compartment { name: (*n).into(), kind: CompartmentKind::Integer })
        .chain(REAL_NAMES.iter().map(|n| Compartment { name: (*n).into(), kind: CompartmentKind::Real }))
        .collect();
    let m = Model {
        constants: vec![],
        ic_grad: Default::default(),
        name: "grouped_test".into(),
        version: "0.1".into(),
        time_unit: "days".into(),
        description: None,
        origin: None,
        origin_rata_die: None,
        compartments,
        transitions: vec![Transition {
            rate_state_grad: Default::default(),
            name: "none".into(),
            stoichiometry: vec![StoichiometryEntry("R".into(), -1), StoichiometryEntry("Q".into(), 1)],
            rate: Expr::const_(0.0),
            metadata: None,
            draw_method: DrawMethod::Poisson,
            rate_grad: Default::default(),
            lineage: None,
        }],
        ode_equations: vec![],
        time_functions: vec![],
        tables: vec![],
        interventions: effects
            .into_iter()
            .map(|(name, actions)| Intervention {
                name: name.into(),
                base_name: None,
                fire: FireSource::Scheduled(InterventionSchedule::AtTimes(vec![1.0])),
                actions,
                kind: InterventionKind::Event,
            })
            .collect(),
        observations: vec![],
        bindings: vec![],
        per_eval_bindings: vec![],
        parameters: vec![Parameter {
            name: "p".into(),
            value: ir::parameter::ParamValue::Fixed { value: 0.4 },
            param_kind: None,
            param_dim: None,
        }],
        initial_conditions: InitialConditions::constants(
            INT_NAMES.iter().chain(REAL_NAMES.iter()).map(|n| ((*n).to_string(), 0.0)),
        ),
        output: OutputConfig {
            times: OutputSchedule::AtTimes(vec![0.0, 1.0]),
            format: "tsv".into(),
            trajectory: true,
            observations: false,
        },
        simulation: SimulationConfig {
            t_start: 0.0,
            t_end: 1.0,
            time_semantics: "continuous".into(),
            dt: Some(1.0),
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
    CompiledModel::new(m).unwrap()
}

fn pairs(cells: &[(&str, &str)]) -> Vec<(String, String)> {
    cells.iter().map(|(s, d)| ((*s).to_string(), (*d).to_string())).collect()
}

/// The stages of `E` into the one cell `R`.
const E_TO_R: [(&str, &str); 3] = [("E1", "R"), ("E2", "R"), ("E3", "R")];
const X_TO_Y: [(&str, &str); 3] = [("X1", "Y"), ("X2", "Y"), ("X3", "Y")];

fn grouped_fraction(cells: &[(&str, &str)], fraction: Expr) -> Action {
    let fraction_grad = match &fraction {
        Expr::Param(_) => [("p".to_string(), DerivEntry::Grad(Expr::const_(1.0)))].into_iter().collect(),
        _ => Default::default(),
    };
    Action::GroupedTransfer(GroupedTransfer {
        pairs: pairs(cells),
        amount: GroupedAmount::Fraction { fraction, fraction_grad },
    })
}

fn grouped_count(cells: &[(&str, &str)], count: f64) -> Action {
    Action::GroupedTransfer(GroupedTransfer {
        pairs: pairs(cells),
        amount: GroupedAmount::Count { count: Expr::const_(count) },
    })
}

fn fraction(src: &str, dst: &str, fraction: Expr) -> Action {
    let fraction_grad = match &fraction {
        Expr::Param(_) => [("p".to_string(), DerivEntry::Grad(Expr::const_(1.0)))].into_iter().collect(),
        _ => Default::default(),
    };
    Action::FractionTransfer(FractionTransfer { src: src.into(), dst: dst.into(), fraction, fraction_grad })
}

fn add(comp: &str, n: f64) -> Action {
    Action::Add(AddAction { compartment: comp.into(), count: Expr::const_(n), count_grad: Default::default() })
}

/// Apply the effects `due` as events at one instant on the integer backends'
/// state. `int` is `[E1, E2, E3, R, Q]`, `real` is `[X1, X2, X3, Y]`.
fn events_discrete(
    m: &CompiledModel,
    due: &[usize],
    int: [i64; 5],
    real: [f64; 4],
) -> Result<(Vec<i64>, Vec<f64>), SimError> {
    let mut int_s = IntState::from_vec(int.to_vec());
    let mut real_s = RealState::from_vec(real.to_vec());
    let mut deltas = EffectDeltas::default();
    {
        let mut arena = DiscreteArena { int: &mut int_s, real: &mut real_s, deltas: &mut deltas };
        apply_event_phases(m, due, &mut arena, &m.default_params, 1.0, 1.0, None)?;
    }
    Ok((int_s.counts, real_s.values))
}

/// The same as scheduled interventions: sequential, in declaration order.
fn interventions_discrete(m: &CompiledModel, due: &[usize], int: [i64; 5]) -> Vec<i64> {
    let mut int_s = IntState::from_vec(int.to_vec());
    let mut real_s = RealState::from_vec(vec![0.0; 4]);
    for &i in due {
        apply_intervention_effects(
            m, i, &m.model.interventions[i], &mut int_s, &mut real_s, &m.default_params, 1.0, 1.0,
        )
        .unwrap();
    }
    int_s.counts
}

/// Apply the effects `due` as events on the ODE's `f64` state.
fn events_continuous(
    m: &CompiledModel,
    due: &[usize],
    int: [f64; 5],
    real: [f64; 4],
) -> Result<(Vec<f64>, Vec<f64>), SimError> {
    let (mut iv, mut rv) = (int.to_vec(), real.to_vec());
    {
        let mut arena = ContinuousArena { int: &mut iv, real: &mut rv, sens: None };
        apply_event_phases(m, due, &mut arena, &m.default_params, 1.0, 1.0, None)?;
    }
    Ok((iv, rv))
}

// ── Allocation, as a pure function ──────────────────────────────────────────

#[test]
fn allocation_worked_case_five_people_at_two_fifths() {
    // 5 people in stages (3, 1, 1), M = floor(5 · 0.4) = 2. Floors (1, 0, 0);
    // remainders 6 mod 5 = 1, 2, 2; the one left over goes to the larger
    // remainder, and of the two tied stages to the earlier one.
    assert_eq!(allocate_by_occupancy(&[3, 1, 1], 2), Some(vec![1, 1, 0]));
}

#[test]
fn allocation_ties_go_to_the_earlier_stage() {
    assert_eq!(allocate_by_occupancy(&[1, 1, 1], 1), Some(vec![1, 0, 0]));
    assert_eq!(allocate_by_occupancy(&[1, 1, 1], 2), Some(vec![1, 1, 0]));
    assert_eq!(allocate_by_occupancy(&[2, 2], 1), Some(vec![1, 0]));
    // A larger remainder beats an earlier position.
    assert_eq!(allocate_by_occupancy(&[1, 2], 1), Some(vec![0, 1]));
    // An empty stage never receives the individual left over.
    assert_eq!(allocate_by_occupancy(&[0, 1, 1], 1), Some(vec![0, 1, 0]));
}

#[test]
fn allocation_of_nothing_and_of_everything() {
    assert_eq!(allocate_by_occupancy(&[0, 0, 0], 0), Some(vec![0, 0, 0]));
    assert_eq!(allocate_by_occupancy(&[4, 0, 7], 0), Some(vec![0, 0, 0]));
    assert_eq!(allocate_by_occupancy(&[4, 0, 7], 11), Some(vec![4, 0, 7]));
    assert_eq!(allocate_by_occupancy(&[], 0), Some(vec![]));
}

#[test]
fn allocation_refuses_what_it_cannot_divide() {
    assert_eq!(allocate_by_occupancy(&[3, 1, 1], 6), None, "more than the stages hold");
    assert_eq!(allocate_by_occupancy(&[3, 1, 1], -1), None, "a negative amount");
    assert_eq!(allocate_by_occupancy(&[3, -1, 1], 1), None, "a negative stage count");
    assert_eq!(allocate_by_occupancy(&[0, 0], 1), None, "an amount from empty stages");
}

#[test]
fn allocation_does_not_overflow_at_large_counts() {
    // m · n_i exceeds i64 here; the arithmetic is in i128.
    let big = i64::MAX / 4;
    let shares = allocate_by_occupancy(&[big, big, big], 3 * big - 1).unwrap();
    assert_eq!(shares.iter().map(|&s| s as i128).sum::<i128>(), 3 * big as i128 - 1);
    assert!(shares.iter().all(|&s| s <= big));
}

/// A small deterministic generator (SplitMix64), so the property check below
/// needs no seed plumbing and is the same on every run.
struct SplitMix(u64);
impl SplitMix {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    fn below(&mut self, n: u64) -> u64 {
        self.next() % n
    }
}

#[test]
fn allocation_conserves_and_respects_every_stage_over_random_inputs() {
    let mut rng = SplitMix(0x5EED);
    for case in 0..20_000 {
        let k = 1 + rng.below(8) as usize;
        // Mix scales, with empty stages common.
        let scale = [2u64, 5, 50, 1_000, 1_000_000][rng.below(5) as usize];
        let counts: Vec<i64> = (0..k)
            .map(|_| if rng.below(4) == 0 { 0 } else { rng.below(scale) as i64 })
            .collect();
        let total: i64 = counts.iter().sum();
        let m = rng.below(total as u64 + 1) as i64;
        let shares = allocate_by_occupancy(&counts, m)
            .unwrap_or_else(|| panic!("case {case}: no allocation of {m} among {counts:?}"));
        assert_eq!(shares.len(), k, "case {case}");
        assert_eq!(shares.iter().sum::<i64>(), m, "case {case}: {m} among {counts:?} gave {shares:?}");
        for (i, (&s, &n)) in shares.iter().zip(&counts).enumerate() {
            assert!(
                (0..=n).contains(&s),
                "case {case}: stage {i} holds {n} and gives {s} ({m} among {counts:?})"
            );
            // Within one individual of the exact proportional share.
            if total > 0 {
                let exact = m as f64 * n as f64 / total as f64;
                assert!((s as f64 - exact).abs() < 1.0, "case {case}: stage {i} share {s} vs {exact}");
            }
        }
    }
}

// ── The discrete resolver ───────────────────────────────────────────────────

#[test]
fn integer_fraction_floors_the_compartment_total_once() {
    let m = model(vec![("move", vec![grouped_fraction(&E_TO_R, Expr::const_(0.4))])]);
    // floor(5 · 0.4) = 2, taken as (1, 1, 0).
    let (int, _) = events_discrete(&m, &[0], [3, 1, 1, 0, 0], [0.0; 4]).unwrap();
    assert_eq!(int, vec![2, 0, 1, 2, 0]);
}

#[test]
fn integer_fraction_moves_people_separate_stage_transfers_would_not() {
    // Three stages of 10 at a fraction of 0.2/3: one floor of 30 · 0.0667 = 2
    // for the compartment, where a floor per stage gives 0 three times.
    let f = 0.2 / 3.0;
    let grouped = model(vec![("move", vec![grouped_fraction(&E_TO_R, Expr::const_(f))])]);
    let (int, _) = events_discrete(&grouped, &[0], [10, 10, 10, 0, 0], [0.0; 4]).unwrap();
    assert_eq!(int[3], 2, "the grouped transfer moves floor(30 · 0.2/3) = 2");
    assert_eq!(int[..3].iter().sum::<i64>(), 28);

    let separate = model(vec![(
        "move",
        E_TO_R.iter().map(|(s, d)| fraction(s, d, Expr::const_(f))).collect(),
    )]);
    let (int, _) = events_discrete(&separate, &[0], [10, 10, 10, 0, 0], [0.0; 4]).unwrap();
    assert_eq!(int[3], 0, "a plain fraction transfer still floors per cell");
}

#[test]
fn integer_fraction_is_clamped_to_the_unit_interval() {
    for (f, moved) in [(-0.5, 0), (0.0, 0), (1.0, 5), (1.7, 5)] {
        let m = model(vec![("move", vec![grouped_fraction(&E_TO_R, Expr::const_(f))])]);
        let (int, _) = events_discrete(&m, &[0], [3, 1, 1, 0, 0], [0.0; 4]).unwrap();
        assert_eq!(int[3], moved, "fraction {f}");
        assert_eq!(int[..3].iter().sum::<i64>(), 5 - moved, "fraction {f}");
    }
}

#[test]
fn integer_count_rounds_then_caps_at_the_compartment_total() {
    // round(2.4) = 2 of 5, as (1, 1, 0).
    let m = model(vec![("move", vec![grouped_count(&E_TO_R, 2.4)])]);
    let (int, _) = events_discrete(&m, &[0], [3, 1, 1, 0, 0], [0.0; 4]).unwrap();
    assert_eq!(int, vec![2, 0, 1, 2, 0]);
    // More than the stages hold: all 5 move, and no stage goes negative.
    let m = model(vec![("move", vec![grouped_count(&E_TO_R, 40.0)])]);
    let (int, _) = events_discrete(&m, &[0], [3, 1, 1, 0, 0], [0.0; 4]).unwrap();
    assert_eq!(int, vec![0, 0, 0, 5, 0]);
}

#[test]
fn an_empty_compartment_moves_nothing() {
    for action in [grouped_fraction(&E_TO_R, Expr::const_(0.9)), grouped_count(&E_TO_R, 7.0)] {
        let m = model(vec![("move", vec![action])]);
        let (int, _) = events_discrete(&m, &[0], [0, 0, 0, 4, 0], [0.0; 4]).unwrap();
        assert_eq!(int, vec![0, 0, 0, 4, 0]);
    }
    for action in [grouped_fraction(&X_TO_Y, Expr::const_(0.9)), grouped_count(&X_TO_Y, 7.0)] {
        let m = model(vec![("move", vec![action])]);
        let (_, real) = events_discrete(&m, &[0], [0; 5], [0.0, 0.0, 0.0, 4.0]).unwrap();
        assert_eq!(real, vec![0.0, 0.0, 0.0, 4.0]);
        let (_, real) = events_continuous(&m, &[0], [0.0; 5], [0.0, 0.0, 0.0, 4.0]).unwrap();
        assert_eq!(real, vec![0.0, 0.0, 0.0, 4.0]);
    }
}

#[test]
fn a_negative_count_is_refused_on_every_path() {
    let m = model(vec![("move", vec![grouped_count(&E_TO_R, -2.0)])]);
    let e = events_discrete(&m, &[0], [3, 1, 1, 0, 0], [0.0; 4]).unwrap_err();
    assert!(e.to_string().contains("negative count -2"), "{e}");
    let e = events_continuous(&m, &[0], [3.0, 1.0, 1.0, 0.0, 0.0], [0.0; 4]).unwrap_err();
    assert!(e.to_string().contains("negative count -2"), "{e}");
    let m = model(vec![("move", vec![grouped_count(&X_TO_Y, -2.0)])]);
    let e = events_discrete(&m, &[0], [0; 5], [3.0, 1.0, 1.0, 0.0]).unwrap_err();
    assert!(e.to_string().contains("negative count -2"), "{e}");
}

#[test]
fn stages_keep_their_destination_cell() {
    // A move between strata of one staged compartment: stage i to stage i.
    // Here the "other stratum" is spelled R, Q, R to make the pairing visible.
    let cells = [("E1", "R"), ("E2", "Q"), ("E3", "R")];
    let m = model(vec![("move", vec![grouped_fraction(&cells, Expr::const_(1.0))])]);
    let (int, _) = events_discrete(&m, &[0], [3, 4, 5, 0, 0], [0.0; 4]).unwrap();
    assert_eq!(int, vec![0, 0, 0, 8, 4]);
}

#[test]
fn real_cells_take_the_real_valued_rules() {
    // Fraction: x_i = n_i · f, as separate fraction transfers.
    let m = model(vec![("move", vec![grouped_fraction(&X_TO_Y, Expr::const_(0.4))])]);
    let (_, real) = events_discrete(&m, &[0], [0; 5], [3.0, 1.0, 1.0, 0.0]).unwrap();
    assert_eq!(real, vec![3.0 - 3.0 * 0.4, 1.0 - 0.4, 1.0 - 0.4, 3.0 * 0.4 + 0.4 + 0.4]);
    // Count: x_i = min(c, N) · n_i / N.
    let m = model(vec![("move", vec![grouped_count(&X_TO_Y, 2.5)])]);
    let (_, real) = events_discrete(&m, &[0], [0; 5], [3.0, 1.0, 1.0, 0.0]).unwrap();
    assert_eq!(real[..3], [3.0 - 2.5 * 3.0 / 5.0, 1.0 - 2.5 / 5.0, 1.0 - 2.5 / 5.0]);
    assert!((real[3] - 2.5).abs() < 1e-12);
    // A count above the total moves the total.
    let m = model(vec![("move", vec![grouped_count(&X_TO_Y, 9.0)])]);
    let (_, real) = events_discrete(&m, &[0], [0; 5], [3.0, 1.0, 1.0, 0.0]).unwrap();
    assert_eq!(real, vec![0.0, 0.0, 0.0, 5.0]);
}

#[test]
fn cells_of_both_representations_are_refused() {
    let m = model(vec![("move", vec![grouped_fraction(&[("E1", "R"), ("X1", "Y")], Expr::const_(0.5))])]);
    let e = events_discrete(&m, &[0], [3, 1, 1, 0, 0], [1.0; 4]).unwrap_err();
    assert!(e.to_string().contains("must be the same representation"), "{e}");
    let m = model(vec![("move", vec![grouped_fraction(&[("E1", "Y"), ("E2", "Y")], Expr::const_(0.5))])]);
    let e = events_continuous(&m, &[0], [3.0, 1.0, 1.0, 0.0, 0.0], [1.0; 4]).unwrap_err();
    assert!(e.to_string().contains("must be the same representation"), "{e}");
}

// ── Stage-count invariance ──────────────────────────────────────────────────

/// The same compartment total, held in one cell or spread over three stages,
/// gives up the same number under the same amount.
#[test]
fn the_total_moved_does_not_depend_on_the_stage_count() {
    let splits: [[i64; 3]; 5] = [[80, 0, 0], [27, 27, 26], [1, 1, 78], [0, 80, 0], [10, 30, 40]];
    for f in [0.0, 0.2 / 3.0, 0.25, 0.4, 0.999, 1.0] {
        // One stage: the compartment is the single cell E1, a plain transfer.
        let plain = model(vec![("move", vec![fraction("E1", "R", Expr::const_(f))])]);
        let (int, _) = events_discrete(&plain, &[0], [80, 0, 0, 0, 0], [0.0; 4]).unwrap();
        let expected = int[3];
        let staged = model(vec![("move", vec![grouped_fraction(&E_TO_R, Expr::const_(f))])]);
        for split in splits {
            let (int, _) =
                events_discrete(&staged, &[0], [split[0], split[1], split[2], 0, 0], [0.0; 4]).unwrap();
            assert_eq!(int[3], expected, "fraction {f}, stages {split:?}");
            assert_eq!(int[..3].iter().sum::<i64>(), 80 - expected);
            assert!(int[..3].iter().all(|&n| n >= 0));
        }
    }
    for c in [0.0, 0.4, 7.0, 7.5, 80.0, 200.0] {
        let plain = model(vec![(
            "move",
            vec![Action::AbsoluteTransfer(AbsoluteTransfer {
                src: "E1".into(),
                dst: "R".into(),
                count: Expr::const_(c),
            })],
        )]);
        let (int, _) = events_discrete(&plain, &[0], [80, 0, 0, 0, 0], [0.0; 4]).unwrap();
        let expected = int[3];
        let staged = model(vec![("move", vec![grouped_count(&E_TO_R, c)])]);
        for split in splits {
            let (int, _) =
                events_discrete(&staged, &[0], [split[0], split[1], split[2], 0, 0], [0.0; 4]).unwrap();
            assert_eq!(int[3], expected, "count {c}, stages {split:?}");
            assert!(int[..3].iter().all(|&n| n >= 0));
        }
    }
}

#[test]
fn the_real_total_moved_does_not_depend_on_the_stage_count() {
    let splits: [[f64; 3]; 4] = [[80.0, 0.0, 0.0], [26.5, 26.5, 27.0], [0.25, 1.0, 78.75], [10.0, 30.0, 40.0]];
    let total_moved = |m: &CompiledModel, split: [f64; 3]| -> (f64, f64) {
        let state = [split[0], split[1], split[2], 0.0];
        let (_, discrete) = events_discrete(m, &[0], [0; 5], state).unwrap();
        let (_, continuous) = events_continuous(m, &[0], [0.0; 5], state).unwrap();
        (discrete[3], continuous[3])
    };
    for f in [0.0, 0.2 / 3.0, 0.4, 1.0] {
        let staged = model(vec![("move", vec![grouped_fraction(&X_TO_Y, Expr::const_(f))])]);
        for split in splits {
            let (d, c) = total_moved(&staged, split);
            assert!((d - 80.0 * f).abs() < 1e-11, "fraction {f}, stages {split:?}: {d}");
            assert!((c - 80.0 * f).abs() < 1e-11, "fraction {f}, stages {split:?}: {c}");
        }
    }
    for count in [0.0, 7.5, 80.0, 200.0] {
        let staged = model(vec![("move", vec![grouped_count(&X_TO_Y, count)])]);
        for split in splits {
            let (d, c) = total_moved(&staged, split);
            assert!((d - count.min(80.0)).abs() < 1e-11, "count {count}, stages {split:?}: {d}");
            assert!((c - count.min(80.0)).abs() < 1e-11, "count {count}, stages {split:?}: {c}");
        }
    }
}

// ── The ODE's continuous state ──────────────────────────────────────────────

#[test]
fn the_ode_applies_the_real_valued_rules_to_integer_compartments() {
    // No floor and no allocation: x_i = n_i · f on the f64 state.
    let m = model(vec![("move", vec![grouped_fraction(&E_TO_R, Expr::const_(0.4))])]);
    let (int, _) = events_continuous(&m, &[0], [3.0, 1.0, 1.0, 0.0, 0.0], [0.0; 4]).unwrap();
    assert_eq!(int[..3], [3.0 - 3.0 * 0.4, 1.0 - 0.4, 1.0 - 0.4]);
    assert!((int[3] - 2.0).abs() < 1e-12);
    let m = model(vec![("move", vec![grouped_count(&E_TO_R, 2.0)])]);
    let (int, _) = events_continuous(&m, &[0], [3.0, 1.0, 1.0, 0.0, 0.0], [0.0; 4]).unwrap();
    assert_eq!(int[..3], [3.0 - 2.0 * 3.0 / 5.0, 1.0 - 2.0 / 5.0, 1.0 - 2.0 / 5.0]);
    assert!((int[3] - 2.0).abs() < 1e-12);
}

#[test]
fn a_grouped_fraction_equals_separate_fraction_transfers_on_the_ode() {
    // Values and forward sensitivities, to the bit, as an event and as a
    // sequential intervention.
    let grouped = model(vec![("move", vec![grouped_fraction(&E_TO_R, Expr::param("p"))])]);
    let separate = model(vec![(
        "move",
        E_TO_R.iter().map(|(s, d)| fraction(s, d, Expr::param("p"))).collect(),
    )]);
    let start = [3.0, 1.25, 7.5, 2.0, 0.0];
    // One sensitivity column (∂/∂p), a row per integer compartment.
    let sens0 = [0.5, -0.25, 2.0, 1.0, 0.0];
    let run = |m: &CompiledModel, as_event: bool| -> (Vec<f64>, Vec<f64>) {
        let (mut iv, mut rv, mut s) = (start.to_vec(), vec![0.0; 4], sens0.to_vec());
        {
            let mut arena = ContinuousArena {
                int: &mut iv,
                real: &mut rv,
                sens: Some(SensBlock { state: &mut s, param_model_idx: &[0] }),
            };
            if as_event {
                apply_event_phases(m, &[0], &mut arena, &m.default_params, 1.0, 1.0, None).unwrap();
            } else {
                apply_interventions_continuous(m, &[0], &mut arena, &m.default_params, 1.0, 1.0).unwrap();
            }
        }
        (iv, s)
    };
    // Both forms make the same changes in the same order on either path, so
    // they agree to the bit.
    let (gv, gs) = run(&grouped, true);
    let (sv, ss) = run(&separate, true);
    assert_eq!(gv, sv, "values, as an event");
    assert_eq!(gs, ss, "sensitivities, as an event");

    let (gv, gs) = run(&grouped, false);
    let (sv, ss) = run(&separate, false);
    assert_eq!(gv, sv, "values, as an intervention");
    assert_eq!(gs, ss, "sensitivities, as an intervention");

    // And the rule itself, by hand: with f = 0.4 active in [0, 1],
    // dx_i = f · S[s_i] + x_i · ∂f/∂p.
    let f = 0.4;
    for i in 0..3 {
        let dx = f * sens0[i] + start[i];
        assert_eq!(gs[i], sens0[i] - dx, "stage {i}");
    }
}

#[test]
fn a_grouped_count_has_no_sensitivity_rule() {
    let m = model(vec![("move", vec![grouped_count(&E_TO_R, 2.0)])]);
    let (mut iv, mut rv, mut s) = (vec![3.0, 1.0, 1.0, 0.0, 0.0], vec![0.0; 4], vec![0.0; 5]);
    let mut arena = ContinuousArena {
        int: &mut iv,
        real: &mut rv,
        sens: Some(SensBlock { state: &mut s, param_model_idx: &[0] }),
    };
    let e = apply_event_phases(&m, &[0], &mut arena, &m.default_params, 1.0, 1.0, None).unwrap_err();
    assert!(e.to_string().contains("carries no amount gradient"), "{e}");
}

// ── Beside other actions at one instant ─────────────────────────────────────

#[test]
fn as_an_event_a_grouped_drain_reads_the_state_after_inflows() {
    // The inflow phase comes first whatever the declaration order: E1 gains
    // 5, and the drain then moves floor(10 · 0.4) = 4 of the 10, as (3, 1, 0)
    // from stages (8, 1, 1): floors (3, 0, 0), remainders 2, 4, 4.
    for actions in [
        vec![grouped_fraction(&E_TO_R, Expr::const_(0.4)), add("E1", 5.0)],
        vec![add("E1", 5.0), grouped_fraction(&E_TO_R, Expr::const_(0.4))],
    ] {
        let m = model(vec![("ev", actions)]);
        let (int, _) = events_discrete(&m, &[0], [3, 1, 1, 0, 0], [0.0; 4]).unwrap();
        assert_eq!(int, vec![5, 0, 1, 4, 0]);
        let (int, _) = events_continuous(&m, &[0], [3.0, 1.0, 1.0, 0.0, 0.0], [0.0; 4]).unwrap();
        assert!((int[3] - 4.0).abs() < 1e-12, "the ODE moves 10 · 0.4");
    }
}

#[test]
fn as_an_event_drains_of_one_phase_read_the_same_state() {
    // A grouped drain of E and a plain drain of R, in one phase: the plain
    // drain reads R before the grouped inflow, in either order.
    for actions in [
        vec![grouped_fraction(&E_TO_R, Expr::const_(0.4)), fraction("R", "Q", Expr::const_(0.5))],
        vec![fraction("R", "Q", Expr::const_(0.5)), grouped_fraction(&E_TO_R, Expr::const_(0.4))],
    ] {
        let m = model(vec![("ev", actions)]);
        let (int, _) = events_discrete(&m, &[0], [3, 1, 1, 10, 0], [0.0; 4]).unwrap();
        assert_eq!(int, vec![2, 0, 1, 10 - 5 + 2, 5]);
    }
}

#[test]
fn as_an_intervention_actions_apply_in_declaration_order() {
    // Sequential: the second action reads what the first left.
    let m = model(vec![(
        "iv",
        vec![grouped_fraction(&E_TO_R, Expr::const_(0.4)), fraction("R", "Q", Expr::const_(0.5))],
    )]);
    // 2 into R (12), then half of 12 to Q.
    assert_eq!(interventions_discrete(&m, &[0], [3, 1, 1, 10, 0]), vec![2, 0, 1, 6, 6]);
    let m = model(vec![(
        "iv",
        vec![fraction("R", "Q", Expr::const_(0.5)), grouped_fraction(&E_TO_R, Expr::const_(0.4))],
    )]);
    assert_eq!(interventions_discrete(&m, &[0], [3, 1, 1, 10, 0]), vec![2, 0, 1, 7, 5]);
    // Two grouped drains in sequence never overdraw: the second divides what
    // the first left.
    let m = model(vec![(
        "iv",
        vec![grouped_fraction(&E_TO_R, Expr::const_(0.5)), grouped_fraction(&E_TO_R, Expr::const_(0.5))],
    )]);
    assert_eq!(interventions_discrete(&m, &[0], [1, 1, 0, 0, 0]), vec![0, 1, 0, 1, 0]);
}

#[test]
fn a_grouped_drain_beside_a_plain_drain_is_checked_like_two_fraction_transfers() {
    let want_plain = {
        let m = model(vec![
            ("a", vec![fraction("E2", "R", Expr::const_(0.6))]),
            ("b", vec![fraction("E2", "Q", Expr::const_(0.6))]),
        ]);
        events_discrete(&m, &[0, 1], [3, 1, 1, 0, 0], [0.0; 4]).unwrap_err().to_string()
    };
    assert!(want_plain.contains("`transfer` fractions from compartment 'E2' sum to 1.2"), "{want_plain}");
    // A grouped drain beside a plain drain of one of its stages: the same
    // refusal when the fractions sum above 1, on both arenas.
    let m = model(vec![
        ("a", vec![grouped_fraction(&E_TO_R, Expr::const_(0.6))]),
        ("b", vec![fraction("E2", "Q", Expr::const_(0.6))]),
    ]);
    let e = events_discrete(&m, &[0, 1], [3, 1, 1, 0, 0], [0.0; 4]).unwrap_err();
    assert_eq!(e.to_string(), want_plain);
    let e = events_continuous(&m, &[0, 1], [3.0, 1.0, 1.0, 0.0, 0.0], [0.0; 4]).unwrap_err();
    assert_eq!(e.to_string(), want_plain);
    // And applied when they sum to at most 1: both read (40, 20, 20); the
    // grouped one takes 20 as (10, 5, 5), the plain one floor(20 · 0.5) = 10.
    let m = model(vec![
        ("a", vec![grouped_fraction(&E_TO_R, Expr::const_(0.25))]),
        ("b", vec![fraction("E2", "Q", Expr::const_(0.5))]),
    ]);
    let (int, _) = events_discrete(&m, &[0, 1], [40, 20, 20, 0, 0], [0.0; 4]).unwrap();
    assert_eq!(int, vec![30, 5, 15, 20, 10]);
}

/// Two grouped transfers due at one instant that share a source cell are
/// refused whatever the state and whatever their fractions: each divides its
/// own total among the stages, and both can give the individual left over to
/// the same stage (from (1, 1), two halves both take stage 1).
#[test]
fn two_grouped_drains_sharing_a_source_cell_are_refused_up_front() {
    let to_q = [("E1", "Q"), ("E2", "Q"), ("E3", "Q")];
    let want = "more than one `transfer` out of a staged compartment draws from 'E1' at t=1: \
                events 'a', 'b'";
    for (fa, fb) in [(0.5, 0.5), (0.25, 0.5), (0.6, 0.6), (0.0, 0.0)] {
        let m = model(vec![
            ("a", vec![grouped_fraction(&E_TO_R, Expr::const_(fa))]),
            ("b", vec![grouped_fraction(&to_q, Expr::const_(fb))]),
        ]);
        // The state does not matter: refused where nothing could overdraw too.
        for int in [[1, 1, 0, 0, 0], [40, 20, 20, 0, 0], [0, 0, 0, 0, 0]] {
            let e = events_discrete(&m, &[0, 1], int, [0.0; 4]).unwrap_err();
            assert!(e.to_string().contains(want), "{fa} + {fb} from {int:?}: {e}");
            let e = events_continuous(&m, &[0, 1], int.map(|n| n as f64), [0.0; 4]).unwrap_err();
            assert!(e.to_string().contains(want), "{fa} + {fb} from {int:?} on the ODE: {e}");
        }
    }
    // One shared cell is enough, and the refusal names it.
    let m = model(vec![
        ("a", vec![grouped_fraction(&[("E1", "R"), ("E2", "R")], Expr::const_(0.2))]),
        ("b", vec![grouped_fraction(&[("E2", "Q"), ("E3", "Q")], Expr::const_(0.2))]),
    ]);
    let e = events_discrete(&m, &[0, 1], [40, 20, 20, 0, 0], [0.0; 4]).unwrap_err();
    assert!(e.to_string().contains("draws from 'E2' at t=1: events 'a', 'b'"), "{e}");
    // Grouped transfers of different compartments do not collide.
    let m = model(vec![
        ("a", vec![grouped_fraction(&[("E1", "R"), ("E2", "R")], Expr::const_(0.5))]),
        ("b", vec![grouped_fraction(&[("E3", "Q"), ("R", "Q")], Expr::const_(0.5))]),
    ]);
    let (int, _) = events_discrete(&m, &[0, 1], [4, 2, 6, 8, 0], [0.0; 4]).unwrap();
    assert_eq!(int, vec![2, 1, 3, 4 + 3, 3 + 4]);
    // Nor do they as scheduled interventions, which apply one after another.
    let m = model(vec![
        ("a", vec![grouped_fraction(&E_TO_R, Expr::const_(0.5))]),
        ("b", vec![grouped_fraction(&to_q, Expr::const_(0.5))]),
    ]);
    assert_eq!(interventions_discrete(&m, &[0, 1], [1, 1, 0, 0, 0]), vec![0, 1, 0, 1, 0]);
}

/// A grouped fraction transfer and a plain fraction transfer from one of its
/// stages, with fractions summing to at most 1, never take more than the
/// stage holds: the plain one takes floor(n · f'), and the grouped one at most
/// ceil(n · f), which is below n − floor(n · f') + 1.
#[test]
fn a_grouped_and_a_plain_fraction_drain_never_overdraw_a_stage() {
    let mut rng = SplitMix(0xD2A1);
    let mut tight = 0u32;
    for case in 0..200_000 {
        let k = 1 + rng.below(6) as usize;
        let scale = [2u64, 3, 7, 10, 100, 100_000][rng.below(6) as usize];
        let counts: Vec<i64> = (0..k)
            .map(|_| if rng.below(5) == 0 { 0 } else { rng.below(scale) as i64 })
            .collect();
        let total: i64 = counts.iter().sum();
        // Fractions as the runtime sees them: f64s whose sum passes the
        // overdraw check. Rationals with small denominators land the
        // products on and beside whole numbers, where rounding bites.
        let den = [2u64, 3, 4, 5, 7, 10, 100, 1 << 20][rng.below(8) as usize];
        let a = rng.below(den + 1);
        let b = rng.below(den - a + 1);
        let (f, f_plain) = match rng.below(3) {
            0 => (a as f64 / den as f64, b as f64 / den as f64),
            1 => (a as f64 / den as f64, 1.0 - a as f64 / den as f64),
            _ => {
                let f = rng.next() as f64 / u64::MAX as f64;
                (f, (1.0 - f) * (rng.next() as f64 / u64::MAX as f64))
            }
        };
        let mut sorted = [f, f_plain];
        sorted.sort_by(|x, y| x.total_cmp(y));
        if sorted[0] + sorted[1] > 1.0 + 2.0 * f64::EPSILON {
            continue; // the overdraw check refuses this pair
        }
        // The grouped transfer's total, as `resolve_action` forms it.
        let moved = (((total as f64) * f.clamp(0.0, 1.0)).floor() as i64).min(total);
        let shares = allocate_by_occupancy(&counts, moved).unwrap();
        for (i, (&n, &g)) in counts.iter().zip(&shares).enumerate() {
            // A plain fraction transfer of this stage, as `resolve_action` forms it.
            let plain = ((n as f64) * f_plain.clamp(0.0, 1.0)).floor() as i64;
            assert!(
                g + plain <= n,
                "case {case}: stage {i} holds {n}; grouped f = {f} takes {g}, plain f' = {f_plain} \
                 takes {plain} (stages {counts:?})"
            );
            tight += u32::from(n > 0 && g + plain == n);
        }
    }
    assert!(tight > 1_000, "the bound is reached often ({tight}), so the check is not slack");
}

#[test]
fn a_grouped_count_beside_another_drain_is_refused_whatever_the_state() {
    let m = model(vec![
        ("a", vec![grouped_count(&E_TO_R, 1.0)]),
        ("b", vec![fraction("E3", "Q", Expr::const_(0.1))]),
    ]);
    let e = events_discrete(&m, &[0, 1], [300, 100, 100, 0, 0], [0.0; 4]).unwrap_err();
    assert!(
        e.to_string().contains("more than one `transfer` from compartment 'E3'")
            && e.to_string().contains("at least one moves a fixed count"),
        "{e}"
    );
}

#[test]
fn a_set_on_a_stage_beside_a_grouped_transfer_is_refused() {
    let set = Action::Set(SetAction { compartment: "E2".into(), value: Expr::const_(5.0), value_grad: Default::default() });
    let m = model(vec![("a", vec![set]), ("b", vec![grouped_fraction(&E_TO_R, Expr::const_(0.5))])]);
    let e = events_discrete(&m, &[0, 1], [3, 1, 1, 0, 0], [0.0; 4]).unwrap_err();
    assert!(e.to_string().contains("a `set` and a `transfer` on compartment 'E2'"), "{e}");
    // A destination cell counts too.
    let set = Action::Set(SetAction { compartment: "R".into(), value: Expr::const_(5.0), value_grad: Default::default() });
    let m = model(vec![("a", vec![set]), ("b", vec![grouped_fraction(&E_TO_R, Expr::const_(0.5))])]);
    let e = events_discrete(&m, &[0, 1], [3, 1, 1, 0, 0], [0.0; 4]).unwrap_err();
    assert!(e.to_string().contains("a `set` and a `transfer` on compartment 'R'"), "{e}");
}
