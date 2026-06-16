//! Gate properties for the Resolve bridge, proven end-to-end through the
//! real CLI→runid mapping: gh#147 (a horizon change re-keys the run_id),
//! presentation normalization (`--format`/`time_semantics` are inert), the
//! resolved-`process_seed` rule (lone vs sweep-point with the same base seed
//! → distinct paths), and non-finite params surfacing as `ResolveError`.

use std::collections::HashMap;

use super::*;
use crate::args::types::Backend;
use ir::model::{
    InitialConditions, Model, OutputConfig, OutputSchedule, RegularOutputSchedule, SimulationConfig,
};

fn tiny_model() -> Model {
    Model {
        name: "sir".into(),
        version: "1".into(),
        time_unit: "days".into(),
        description: None,
        origin: None,
        origin_rata_die: None,
        compartments: vec![],
        transitions: vec![],
        ode_equations: vec![],
        time_functions: vec![],
        tables: vec![],
        interventions: vec![],
        observations: vec![],
        parameters: vec![],
        bindings: vec![],
        initial_conditions: InitialConditions::Explicit(Default::default()),
        output: OutputConfig {
            times: OutputSchedule::Regular(RegularOutputSchedule { start: 0.0, step: 1.0, end: 100.0 }),
            format: "tsv".into(),
            trajectory: true,
            observations: false,
        },
        simulation: SimulationConfig {
            t_start: 0.0,
            t_end: 100.0,
            time_semantics: "continuous".into(),
            dt: Some(1.0),
            rng_seed: None,
            integrator: Default::default(),
        },
        presets: vec![],
        model_structure: None,
        balance: None,
        identity_tracked_compartments: vec![],
    }
}

fn ctx<'a>(
    model: &'a Model,
    output: &'a OutputSchedule,
    params: &'a HashMap<String, f64>,
    t_end: f64,
    base_seed: u64,
    process_seed: u64,
) -> TrajectoryCtx<'a> {
    TrajectoryCtx {
        model,
        model_stem: "sir",
        ir_version: "0.7",
        engine_version: "0.3.0+test",
        backend: Backend::ChainBinomial,
        dt: 1.0,
        t_start: 0.0,
        t_end,
        output,
        allow_degenerate_rates: false,
        no_flows: false,
        columns: columns_empty(),
        base_params: params,
        table_digests: vec![],
        enable: &[],
        disable: &[],
        scen_params: params_empty(),
        param_label: "base",
        scenario_label: "baseline",
        base_seed,
        process_seed,
    }
}

fn params_empty() -> &'static HashMap<String, f64> {
    use std::sync::OnceLock;
    static EMPTY: OnceLock<HashMap<String, f64>> = OnceLock::new();
    EMPTY.get_or_init(HashMap::new)
}

fn columns_empty() -> &'static std::collections::BTreeSet<String> {
    use std::sync::OnceLock;
    static EMPTY: OnceLock<std::collections::BTreeSet<String>> = OnceLock::new();
    EMPTY.get_or_init(std::collections::BTreeSet::new)
}

#[test]
fn horizon_change_re_keys_the_run_id() {
    // gh#147: the old model_hash omitted output cadence / t_end. Here the
    // horizon lives in the config level; a t_end change must change the
    // config level hash and the run_id (but not the model level).
    let model = tiny_model();
    let out = OutputSchedule::Regular(RegularOutputSchedule { start: 0.0, step: 1.0, end: 100.0 });
    let p = HashMap::new();
    let a = resolve_trajectory(&ctx(&model, &out, &p, 100.0, 1, 1)).unwrap();
    let b = resolve_trajectory(&ctx(&model, &out, &p, 200.0, 1, 1)).unwrap();

    assert_eq!(a.levels[0].hash, b.levels[0].hash, "model level unchanged");
    assert_ne!(a.levels[1].hash, b.levels[1].hash, "config level (t_end) must differ");
    assert_ne!(a.run_id, b.run_id, "a t_end change must re-key the run_id (gh#147)");
}

#[test]
fn presentation_fields_are_inert() {
    // Two models differing only in output.format / time_semantics must
    // produce the same model digest — those fields are normalized out, so
    // --format and time rendering stay provenance.
    let mut m1 = tiny_model();
    let mut m2 = tiny_model();
    m1.output.format = "tsv".into();
    m1.simulation.time_semantics = "continuous".into();
    m2.output.format = "parquet".into();
    m2.simulation.time_semantics = "calendar".into();
    assert_eq!(
        model_digest(&m1, "0.7", "0.3.0").content_hash(),
        model_digest(&m2, "0.7", "0.3.0").content_hash(),
        "output.format / time_semantics must not affect the model digest"
    );

    // Negative control: a real structural change (a different name) does.
    let mut m3 = tiny_model();
    m3.name = "different".into();
    assert_ne!(
        model_digest(&m1, "0.7", "0.3.0").content_hash(),
        model_digest(&m3, "0.7", "0.3.0").content_hash()
    );
}

#[test]
fn lone_vs_sweep_point_same_base_seed_distinct_paths() {
    // The resolved-seed rule, end-to-end: a lone `--seed 42` (process_seed =
    // 42) and a sweep-point with base 42 (process_seed = mixed) share the
    // readable `seed_42` label but must produce distinct run_ids AND distinct
    // store paths. Non-vacuous: both carry base_seed = 42, so this passes
    // only because the seed level hashes process_seed, not the base.
    let model = tiny_model();
    let out = OutputSchedule::Regular(RegularOutputSchedule { start: 0.0, step: 1.0, end: 100.0 });
    let p = HashMap::new();
    let mixed = crate::util::mix_cell_seed(42, 1, 0);
    let lone = resolve_trajectory(&ctx(&model, &out, &p, 100.0, 42, 42)).unwrap();
    let sweep = resolve_trajectory(&ctx(&model, &out, &p, 100.0, 42, mixed)).unwrap();

    assert_ne!(lone.run_id, sweep.run_id, "distinct process_seed → distinct run_id");

    let root = std::path::Path::new("/results");
    let pa = runid::store_path(root, ArtifactKind::Sim, &lone.levels);
    let pb = runid::store_path(root, ArtifactKind::Sim, &sweep.levels);
    assert!(pa.to_string_lossy().contains("/seed_42-"));
    assert!(pb.to_string_lossy().contains("/seed_42-"));
    assert_ne!(pa, pb, "lone vs sweep-point with the same base seed → DISTINCT paths");
}

#[test]
fn run_id_is_composed_from_the_level_hashes() {
    // The leaf identity is the factored tuple: run_id = hash(kind, [level
    // hashes in path order]) — the contract the store/reader rely on.
    let model = tiny_model();
    let out = OutputSchedule::Regular(RegularOutputSchedule { start: 0.0, step: 1.0, end: 100.0 });
    let p = HashMap::new();
    let r = resolve_trajectory(&ctx(&model, &out, &p, 100.0, 5, 5)).unwrap();
    assert_eq!(r.levels.len(), 5, "model/config/params/scenario/seed");
    let names: Vec<&str> = r.levels.iter().map(|l| l.name.as_str()).collect();
    assert_eq!(names, ["model", "config", "params", "scenario", "seed"]);
    let hs: Vec<_> = r.levels.iter().map(|l| l.hash).collect();
    assert_eq!(r.run_id, runid::run_id(ArtifactKind::Sim, &hs));
}

#[test]
fn non_finite_param_is_a_resolve_error() {
    let model = tiny_model();
    let out = OutputSchedule::Regular(RegularOutputSchedule { start: 0.0, step: 1.0, end: 100.0 });
    let mut p = HashMap::new();
    p.insert("beta".to_string(), f64::NAN);
    let r = resolve_trajectory(&ctx(&model, &out, &p, 100.0, 1, 1));
    assert!(
        matches!(r, Err(ResolveError::NonFiniteParam(_))),
        "a NaN resolved param must surface as ResolveError before any hashing"
    );
}

#[test]
fn scenario_delta_re_keys_only_the_scenario_level() {
    // An enable/disable change re-keys the scenario level (and run_id), not
    // the model/config/params/seed levels.
    let model = tiny_model();
    let out = OutputSchedule::Regular(RegularOutputSchedule { start: 0.0, step: 1.0, end: 100.0 });
    let p = HashMap::new();
    let base = resolve_trajectory(&ctx(&model, &out, &p, 100.0, 1, 1)).unwrap();

    let mut c = ctx(&model, &out, &p, 100.0, 1, 1);
    let enable = ["vax".to_string()];
    c.enable = &enable;
    c.scenario_label = "with_vax";
    let withvax = resolve_trajectory(&c).unwrap();

    assert_eq!(base.levels[0].hash, withvax.levels[0].hash, "model unchanged");
    assert_eq!(base.levels[1].hash, withvax.levels[1].hash, "config unchanged");
    assert_eq!(base.levels[2].hash, withvax.levels[2].hash, "params unchanged");
    assert_ne!(base.levels[3].hash, withvax.levels[3].hash, "scenario must differ");
    assert_ne!(base.run_id, withvax.run_id);
}
