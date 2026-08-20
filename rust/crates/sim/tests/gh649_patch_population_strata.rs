//! gh#649 — `patch_population` must read the stratum from the model's declared
//! dimensions, never from the compartment *name*.
//!
//! `patch_population` returns N₀, the Binomial denominator in the IVP
//! initial-value density `Binom(S₀; N₀, s₀)`. Three consumers read it: the
//! complete-data log-likelihood value (`pgas::complete_data_loglik`), its
//! gradient (`pgas_grad::complete_data_loglik_grad`), and the free-particle
//! initial draw in `csmc_as`. The Binomial MLE of `s₀` is `k/N₀`, so a wrong
//! N₀ biases the posterior for every initial-condition parameter — silently.
//!
//! The two shapes a name suffix gets wrong:
//!
//!   1. **More than one dimension.** `S_p1_child` has suffix `child`, which
//!      matches `S_p2_child` too, so the sum runs across patches.
//!   2. **A dimension value containing `_`.** `patch = [north_kivu,
//!      south_kivu]` gives every compartment the suffix `kivu`, so N₀ becomes
//!      the whole model population. This is DRC province naming.
//!
//! Plus the negative controls that pin what must NOT move: an unstratified
//! model still returns the total population; a single-dimension model with
//! underscore-free values returns exactly what it returned before; and a
//! partially stratified model (`stratify(by = …, only = [E])`, the Erlang-stage
//! shape of `ocaml/golden/seir_erlang.camdl`) keeps both its answers.

use std::collections::HashMap;

use ir::{
    model::{
        Compartment, CompartmentKind, Dimension, InitialConditions, ModelStructure, OutputConfig,
        OutputSchedule, SimulationConfig,
    },
    Model,
};
use sim::{compiled_model::CompiledModel, inference::pgas::patch_population};

/// A compiled model with `names` as its integer compartments and an optional
/// `model_structure`. No transitions: `patch_population` reads only the
/// compartment list and the declared dimensions.
fn model_with(names: &[&str], structure: Option<ModelStructure>) -> CompiledModel {
    let model = Model {
        ic_grad: Default::default(),
        name: "gh649".into(),
        version: "0".into(),
        time_unit: "days".into(),
        description: None,
        origin: None,
        origin_rata_die: None,
        compartments: names
            .iter()
            .map(|n| Compartment { name: (*n).into(), kind: CompartmentKind::Integer })
            .collect(),
        transitions: vec![],
        ode_equations: vec![],
        time_functions: vec![],
        tables: vec![],
        interventions: vec![],
        observations: vec![],
        bindings: vec![],
        per_eval_bindings: vec![],
        parameters: vec![],
        initial_conditions: InitialConditions::Explicit(HashMap::new()),
        output: OutputConfig {
            times: OutputSchedule::AtTimes(vec![]),
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
        model_structure: structure,
        balance: None,
        identity_tracked_compartments: vec![],
        quantities: vec![],
        contrasts: vec![],
    };
    CompiledModel::new(model).expect("model compiles")
}

/// `dims`: declared dimensions in order. `comp_dims`: base compartment → the
/// dimension names it is stratified by (the expander emits an entry for every
/// base, with an empty list when it is unstratified).
fn structure(dims: &[(&str, &[&str])], comp_dims: &[(&str, &[&str])]) -> ModelStructure {
    ModelStructure {
        dimensions: dims
            .iter()
            .map(|(n, vs)| Dimension {
                name: (*n).into(),
                values: vs.iter().map(|v| (*v).to_string()).collect(),
            })
            .collect(),
        compartment_dims: comp_dims
            .iter()
            .map(|(n, ds)| ((*n).to_string(), ds.iter().map(|d| (*d).to_string()).collect()))
            .collect(),
        base_compartments: comp_dims.iter().map(|(n, _)| (*n).to_string()).collect(),
        transmission_transitions: vec![],
        infectious_compartments: vec![],
    }
}

/// Index of `name` in `names`, so each assertion names the compartment it means.
fn idx(names: &[&str], name: &str) -> usize {
    names.iter().position(|n| *n == name).expect("compartment present")
}

// ── Failure shape 1: more than one dimension ────────────────────────────────

/// `patch = [p1, p2]` × `age = [child, adult]`, S and I stratified by both.
/// The stratum of `S_p1_child` is the (p1, child) cell — S_p1_child + I_p1_child
/// — not every `*_child` compartment across both patches.
#[test]
fn two_dimensions_do_not_sum_a_stratum_across_patches() {
    // Cartesian order, patch major (the order the OCaml expander emits).
    let names = [
        "S_p1_child", "S_p1_adult", "S_p2_child", "S_p2_adult",
        "I_p1_child", "I_p1_adult", "I_p2_child", "I_p2_adult",
    ];
    let m = model_with(
        &names,
        Some(structure(
            &[("patch", &["p1", "p2"]), ("age", &["child", "adult"])],
            &[("S", &["patch", "age"]), ("I", &["patch", "age"])],
        )),
    );
    // Deliberately distinct counts: no two strata share a sum, so a wrong
    // grouping cannot land on the right number by coincidence.
    let counts: Vec<i64> = vec![
        100, 200, 300, 400, // S_p1_child, S_p1_adult, S_p2_child, S_p2_adult
        7, 11, 13, 17, //     I_p1_child, I_p1_adult, I_p2_child, I_p2_adult
    ];
    assert_eq!(counts.iter().sum::<i64>(), 1048, "total population");

    // (p1, child) = S_p1_child + I_p1_child = 100 + 7.
    assert_eq!(patch_population(&m, &counts, idx(&names, "S_p1_child")), 107);
    // (p2, adult) = S_p2_adult + I_p2_adult = 400 + 17.
    assert_eq!(patch_population(&m, &counts, idx(&names, "I_p2_adult")), 417);
    // The suffix answer for S_p1_child would be every `*_child`: 100+300+7+13.
    assert_ne!(patch_population(&m, &counts, idx(&names, "S_p1_child")), 420);
}

// ── Failure shape 2: a dimension value containing `_` ───────────────────────

/// `patch = [north_kivu, south_kivu]` — DRC province naming. Every expanded
/// compartment name ends in `_kivu`, so a suffix groups the entire model into
/// one stratum.
#[test]
fn dimension_value_containing_underscore_keeps_patches_apart() {
    let names = [
        "S_north_kivu", "S_south_kivu",
        "I_north_kivu", "I_south_kivu",
        "R_north_kivu", "R_south_kivu",
    ];
    let m = model_with(
        &names,
        Some(structure(
            &[("patch", &["north_kivu", "south_kivu"])],
            &[("S", &["patch"]), ("I", &["patch"]), ("R", &["patch"])],
        )),
    );
    let counts: Vec<i64> = vec![
        900, 1900, // S_north_kivu, S_south_kivu
        10, 20, //   I_north_kivu, I_south_kivu
        90, 80, //   R_north_kivu, R_south_kivu
    ];
    assert_eq!(counts.iter().sum::<i64>(), 3000, "total population");

    assert_eq!(patch_population(&m, &counts, idx(&names, "S_north_kivu")), 1000);
    assert_eq!(patch_population(&m, &counts, idx(&names, "I_south_kivu")), 2000);
    // The suffix answer is the whole model for every compartment.
    assert_ne!(patch_population(&m, &counts, idx(&names, "S_north_kivu")), 3000);
}

// ── Negative controls: what must NOT move ───────────────────────────────────

/// No dimensions declared: every compartment is in the one stratum, so N₀ is
/// the total population. This is the current answer and must stay the answer.
#[test]
fn unstratified_model_returns_the_total_population() {
    let names = ["S", "I", "R"];
    let m = model_with(&names, None);
    let counts: Vec<i64> = vec![990, 5, 5];
    for name in names {
        assert_eq!(patch_population(&m, &counts, idx(&names, name)), 1000, "{name}");
    }
}

/// One dimension, underscore-free values — the case the suffix rule got right.
/// This fix must be behaviour-preserving here.
#[test]
fn single_dimension_without_underscores_is_unchanged() {
    let names = ["S_a", "S_b", "I_a", "I_b", "R_a", "R_b"];
    let m = model_with(
        &names,
        Some(structure(
            &[("patch", &["a", "b"])],
            &[("S", &["patch"]), ("I", &["patch"]), ("R", &["patch"])],
        )),
    );
    let counts: Vec<i64> = vec![900, 1900, 10, 20, 90, 80];
    // patch a = S_a + I_a + R_a = 900 + 10 + 90; patch b = 1900 + 20 + 80.
    assert_eq!(patch_population(&m, &counts, idx(&names, "S_a")), 1000);
    assert_eq!(patch_population(&m, &counts, idx(&names, "R_a")), 1000);
    assert_eq!(patch_population(&m, &counts, idx(&names, "S_b")), 2000);
}

/// An unstratified model whose compartment name happens to contain `_`. There
/// are no dimensions, so there is one stratum and N₀ is the total population —
/// the same root cause as the two failures above, read off the model structure
/// rather than the name.
#[test]
fn underscore_in_a_name_without_dimensions_is_not_a_stratum() {
    let names = ["S", "I", "R", "n_vax"];
    let m = model_with(&names, None);
    let counts: Vec<i64> = vec![900, 50, 30, 20];
    assert_eq!(patch_population(&m, &counts, idx(&names, "n_vax")), 1000);
    assert_eq!(patch_population(&m, &counts, idx(&names, "S")), 1000);
}

/// Partial stratification — `stratify(by = latent_stage, only = [E])`, the
/// Erlang-latent-stage shape of `ocaml/golden/seir_erlang.camdl`. S, I and R
/// carry no dimension; E is split into e1/e2/e3.
///
/// A compartment names the stratum given by *its own* dimensions. `S` carries
/// none, so it names the whole model and N₀ is the total population. `E_e1`
/// carries `latent_stage = e1`, and no other compartment carries that
/// dimension at all, so its stratum is itself. Both are today's answers and
/// both must survive.
#[test]
fn partial_stratification_keeps_both_answers() {
    let names = ["S", "I", "R", "E_e1", "E_e2", "E_e3"];
    let m = model_with(
        &names,
        Some(structure(
            &[("latent_stage", &["e1", "e2", "e3"])],
            &[("S", &[]), ("I", &[]), ("R", &[]), ("E", &["latent_stage"])],
        )),
    );
    let counts: Vec<i64> = vec![900, 50, 30, 5, 7, 8];
    assert_eq!(counts.iter().sum::<i64>(), 1000, "total population");

    // S is in no stratum, so its denominator is the whole population.
    assert_eq!(patch_population(&m, &counts, idx(&names, "S")), 1000);
    assert_eq!(patch_population(&m, &counts, idx(&names, "R")), 1000);
    // E_e1 is the only compartment carrying latent_stage = e1.
    assert_eq!(patch_population(&m, &counts, idx(&names, "E_e1")), 5);
    assert_eq!(patch_population(&m, &counts, idx(&names, "E_e2")), 7);
}
