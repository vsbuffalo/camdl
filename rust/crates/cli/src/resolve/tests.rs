//! Gate properties for the Resolve bridge, proven end-to-end through the
//! real CLI→runid mapping: gh#147 (a horizon change re-keys the run_id),
//! presentation normalization (`--format`/`time_semantics` are inert), the
//! resolved-`process_seed` rule (lone vs sweep-point with the same base seed
//! → distinct paths), and non-finite params surfacing as `ResolveError`.
//!
//! The presentation-normalization gate is **cross-kind** (gh#442): it is
//! asserted for every CAS kind whose identity folds a model — `sim`, `fit`,
//! `pfilter`, `survey`, `sim_ensemble`, `profile` — not just the two that
//! route through this module. See the gh#442 section at the bottom.

use std::collections::HashMap;

use super::*;
use crate::args::types::ForwardBackend;
use ir::model::{
    InitialConditions, Model, OutputConfig, OutputSchedule, RegularOutputSchedule, SimulationConfig,
};

fn tiny_model() -> Model {
    Model {
        ic_grad: Default::default(),
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
        per_eval_bindings: vec![],
        initial_conditions: InitialConditions::Explicit(Default::default()),
        output: OutputConfig {
            times: OutputSchedule::Regular(RegularOutputSchedule { start: 0.0, step: 1.0 }),
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
        identity_tracked_compartments: vec![], quantities: vec![], contrasts: vec![],
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
        backend: ForwardBackend::ChainBinomial,
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

/// The `model`-level digest hash under fixed versions — the unit the
/// presentation / gradient inertness tests compare. Goes through the real
/// `ModelDigest::from_model`, which owns the presentation strip (gh#442).
fn model_digest_hash(m: &Model) -> ContentHash {
    ModelDigest::from_model(m, "0.7".into(), EngineVersion("0.3.0".into())).content_hash()
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
    // gh#147: the horizon lives in the config level (not the model level); a
    // t_end change must change the
    // config level hash and the run_id (but not the model level).
    let model = tiny_model();
    let out = OutputSchedule::Regular(RegularOutputSchedule { start: 0.0, step: 1.0 });
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
        model_digest_hash(&m1),
        model_digest_hash(&m2),
        "output.format / time_semantics must not affect the model digest"
    );

    // Negative control: a real structural change (a different name) does.
    let mut m3 = tiny_model();
    m3.name = "different".into();
    assert_ne!(
        model_digest_hash(&m1),
        model_digest_hash(&m3)
    );
}

#[test]
fn gradient_maps_are_inert() {
    // Model identity is gradient-independent (proposal
    // 2026-07-16-gradient-maps-out-of-run-identity.md): the compiler-derived
    // gradient maps are not folded into the model hash, so the SAME model
    // compiled full (with gradients, for nuts/ode) and lean (`camdlc
    // --no-state-grad`, for simulate/mh) shares ONE model digest — the re-key-free
    // guarantee that unblocks gh#439 A2. Proven end-to-end through
    // `ModelDigest::from_model`, the identity used for every `model` level.
    use ir::deriv::{CompGradMap, DerivEntry};
    use ir::expr::{BinOp, Expr};
    use ir::transition::{DrawMethod, StoichiometryEntry, Transition};

    let tx = |with_grad: bool| -> Transition {
        let mut rate_grad = std::collections::HashMap::new();
        let mut rate_state_grad = std::collections::HashMap::new();
        if with_grad {
            rate_grad.insert("beta".to_string(), DerivEntry::Grad(Expr::pop("I")));
            rate_state_grad.insert("I".to_string(), DerivEntry::Grad(Expr::param("beta")));
        }
        Transition {
            name: "infection".into(),
            stoichiometry: vec![
                StoichiometryEntry("S".into(), -1),
                StoichiometryEntry("I".into(), 1),
            ],
            rate: Expr::bin_op(BinOp::Mul, Expr::param("beta"), Expr::pop("I")),
            metadata: None,
            draw_method: DrawMethod::Poisson,
            rate_grad,
            rate_state_grad: CompGradMap(rate_state_grad),
            lineage: None,
        }
    };

    let mut lean = tiny_model();
    lean.transitions = vec![tx(false)];
    let mut full = tiny_model();
    full.transitions = vec![tx(true)];
    assert_eq!(
        model_digest_hash(&lean),
        model_digest_hash(&full),
        "gradient maps must not affect the model digest — lean and full compiles of \
         the same model share one identity (proposal 2026-07-16)"
    );

    // Negative control: the rate EXPRESSION itself is identity — changing it
    // (still gradient-free) must re-key.
    let mut changed = tiny_model();
    let mut t = tx(false);
    t.rate = Expr::param("beta");
    changed.transitions = vec![t];
    assert_ne!(
        model_digest_hash(&lean),
        model_digest_hash(&changed),
        "the rate expression itself is identity (only its gradient is stripped)"
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
    let out = OutputSchedule::Regular(RegularOutputSchedule { start: 0.0, step: 1.0 });
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
    let out = OutputSchedule::Regular(RegularOutputSchedule { start: 0.0, step: 1.0 });
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
    let out = OutputSchedule::Regular(RegularOutputSchedule { start: 0.0, step: 1.0 });
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
    let out = OutputSchedule::Regular(RegularOutputSchedule { start: 0.0, step: 1.0 });
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

// ── gh#241 PR F: input-surface differential harness (sim/batch) ───────────────
//
// The identity guarantee made executable: a SEMANTIC input mutation MUST re-key
// the `run_id`; a PRESENTATION / provenance mutation MUST NOT. This complements
// the anti-drift golden (which pins absolute encodings) with parametric
// sensitivity + inertness — so a future change that leaks a presentation field
// into identity, or drops a semantic one, fails here, not silently in the field.

/// All resolved trajectory inputs, owned, so each case clones the base and
/// mutates exactly one field. `run_id()` resolves through the real
/// `resolve_trajectory` path.
#[derive(Clone)]
struct SimInputs {
    model: Model,
    model_stem: String,
    backend: ForwardBackend,
    dt: f64,
    t_start: f64,
    t_end: f64,
    output: OutputSchedule,
    allow_degenerate_rates: bool,
    no_flows: bool,
    columns: std::collections::BTreeSet<String>,
    base_params: HashMap<String, f64>,
    enable: Vec<String>,
    disable: Vec<String>,
    scen_params: HashMap<String, f64>,
    param_label: String,
    scenario_label: String,
    base_seed: u64,
    process_seed: u64,
}

impl SimInputs {
    fn base() -> Self {
        SimInputs {
            model: tiny_model(),
            model_stem: "sir".into(),
            backend: ForwardBackend::ChainBinomial,
            dt: 1.0,
            t_start: 0.0,
            t_end: 100.0,
            output: OutputSchedule::Regular(RegularOutputSchedule { start: 0.0, step: 1.0 }),
            allow_degenerate_rates: false,
            no_flows: false,
            columns: std::collections::BTreeSet::new(),
            base_params: HashMap::new(),
            enable: vec![],
            disable: vec![],
            scen_params: HashMap::new(),
            param_label: "base".into(),
            scenario_label: "baseline".into(),
            base_seed: 1,
            process_seed: 1,
        }
    }

    fn run_id(&self) -> ContentHash {
        resolve_trajectory(&TrajectoryCtx {
            model: &self.model,
            model_stem: &self.model_stem,
            ir_version: "0.7",
            engine_version: "0.3.0+test",
            backend: self.backend,
            dt: self.dt,
            t_start: self.t_start,
            t_end: self.t_end,
            output: &self.output,
            allow_degenerate_rates: self.allow_degenerate_rates,
            no_flows: self.no_flows,
            columns: &self.columns,
            base_params: &self.base_params,
            table_digests: vec![],
            enable: &self.enable,
            disable: &self.disable,
            scen_params: &self.scen_params,
            param_label: &self.param_label,
            scenario_label: &self.scenario_label,
            base_seed: self.base_seed,
            process_seed: self.process_seed,
        })
        .expect("resolve")
        .run_id
    }
}

#[test]
fn differential_semantic_inputs_rekey_the_run_id() {
    let base_rid = SimInputs::base().run_id();
    // Each closure mutates exactly one SEMANTIC field; the run_id MUST change.
    let cases: Vec<(&str, Box<dyn Fn(&mut SimInputs)>)> = vec![
        ("backend",          Box::new(|i: &mut SimInputs| i.backend = ForwardBackend::Gillespie)),
        ("dt",               Box::new(|i| i.dt = 0.5)),
        ("t_start",          Box::new(|i| i.t_start = 10.0)),
        ("t_end",            Box::new(|i| i.t_end = 200.0)),
        ("output_step",      Box::new(|i| i.output =
            OutputSchedule::Regular(RegularOutputSchedule { start: 0.0, step: 2.0 }))),
        ("base_param",       Box::new(|i| { i.base_params.insert("beta".into(), 0.5); })),
        ("scenario_enable",  Box::new(|i| i.enable = vec!["vacc".into()])),
        ("scenario_disable", Box::new(|i| i.disable = vec!["aging".into()])),
        ("scen_param",       Box::new(|i| { i.scen_params.insert("beta".into(), 0.7); })),
        ("process_seed",     Box::new(|i| i.process_seed = 2)),
        ("no_flows",         Box::new(|i| i.no_flows = true)),
        ("columns",          Box::new(|i| { i.columns.insert("S".into()); })),
        ("model_structure",  Box::new(|i| i.model.name = "different".into())),
    ];
    for (name, mutate) in cases {
        let mut i = SimInputs::base();
        mutate(&mut i);
        assert_ne!(i.run_id(), base_rid, "semantic input `{name}` must re-key the run_id");
    }
}

#[test]
fn differential_presentation_inputs_are_inert() {
    let base_rid = SimInputs::base().run_id();
    // Each mutates exactly one PRESENTATION / provenance field; run_id MUST hold.
    // (`base_seed` is `#[run_input(provenance)]`; the level *labels* never enter
    // the hash — `run_id` is built from level hashes only.)
    let cases: Vec<(&str, Box<dyn Fn(&mut SimInputs)>)> = vec![
        ("model_stem",     Box::new(|i: &mut SimInputs| i.model_stem = "renamed".into())),
        ("param_label",    Box::new(|i| i.param_label = "p1".into())),
        ("scenario_label", Box::new(|i| i.scenario_label = "sc1".into())),
        ("base_seed",      Box::new(|i| i.base_seed = 99)), // process_seed held fixed
    ];
    for (name, mutate) in cases {
        let mut i = SimInputs::base();
        mutate(&mut i);
        assert_eq!(i.run_id(), base_rid, "presentation input `{name}` must NOT affect the run_id");
    }
}

// ── gh#442: presentation normalization is cross-kind ──────────────────────────
//
// Model identity used to be normalized only on the paths routing through
// `resolve::model_digest` (sim, fit). `pfilter` / `survey` / `sim_ensemble` /
// `profile` called `ModelDigest::from_model` on the RAW model, so
// `output.format` and `simulation.time_semantics` were inert for sim+fit and
// LOAD-BEARING for those four — the same model keyed differently depending only
// on `--format`. Normalization now lives inside `ModelDigest::from_model`, the
// one constructor every identity path routes through, so the property below
// cannot hold for some kinds and silently fail for others.
//
// Two gates here:
//   1. `presentation_fields_are_inert_on_every_cas_kind` — the property.
//   2. `cas_identity_pins` — the absolute run_id hexes, so the deliberate gh#442
//      re-key of the four batch kinds is the last unannounced move any of them
//      makes, and so the sim/fit keys (which this refactor must NOT touch) stay
//      pinned to their pre-gh#442 values.

use crate::fit::config_v2::FitConfigV2;
use crate::pfilter_cas::{resolve_pfilter, PfilterCtx};
use crate::profile_cas::{resolve_profile_point, ProfilePointCtx};
use crate::sim_ensemble_cas::{resolve_sim_ensemble, EnsembleCell, EnsembleCtx};
use crate::survey_cas::{resolve_survey, SurveyCtx};

const IRV: &str = "0.7";
const ENGV: &str = "0.3.0+test";

/// A minimal fit config with no `[data]` streams, so `fit_level_digest` needs
/// no files on disk and the model is its only model-bearing input.
fn fit_config() -> FitConfigV2 {
    toml::from_str(
        "[model]\ncamdl = \"models/sir.camdl\"\n\
         [estimate]\nbeta = { bounds = [0.01, 2.0] }\n\
         [fixed]\nN0 = 1000000\n\
         [stages.mle]\nalgorithm = \"if2\"\nbackend = \"chain_binomial\"\n\
         chains = 4\nparticles = 1000\niterations = 50\ncooling = 0.70\n",
    )
    .expect("fixture fit config must parse")
}

/// Resolve `model` through EVERY model-bearing CAS identity path. Returns
/// `(kind, model-bearing level hash, leaf run_id)` in a fixed order. Every
/// non-model input is held constant, so any hash motion is the model's.
fn all_kind_identities(model: &Model) -> Vec<(&'static str, ContentHash, ContentHash)> {
    let mut out = Vec::new();

    // sim — `resolve_trajectory`; levels[0] is the `model` level.
    let out_sched = OutputSchedule::Regular(RegularOutputSchedule { start: 0.0, step: 1.0 });
    let params: HashMap<String, f64> = HashMap::new();
    let sim = resolve_trajectory(&ctx(model, &out_sched, &params, 100.0, 1, 1)).expect("sim");
    out.push(("sim", sim.levels[0].hash, sim.run_id));

    // fit — `resolve_fit_stage`; levels[0] is the `fit` level (folds the model
    // digest alongside the data/config digests).
    let cfg = fit_config();
    let stage = cfg.stages.get("mle").expect("fixture stage").clone();
    let fit = crate::fit::cas::resolve_fit_stage(&crate::fit::cas::FitStageCtx {
        model,
        fit_stem: "sir",
        ir_version: IRV,
        engine_version: ENGV,
        config: &cfg,
        data_paths: &indexmap::IndexMap::new(),
        stage_name: "mle",
        stage: &stage,
        ordinal: 1,
        seed: 7,
        deps: vec![],
    })
    .expect("fit");
    out.push(("fit", fit.levels[0].hash, fit.run_id));

    let data: Vec<(String, String)> =
        vec![("cases".to_string(), ContentHash::digest_bytes(b"cases").to_hex())];

    // pfilter — levels[0] is the `model` level.
    let pf_params = [("beta".to_string(), 0.3_f64)];
    let pf = resolve_pfilter(&PfilterCtx {
        model,
        ir_version: IRV,
        engine_version: ENGV,
        stem: "sir",
        data: &data,
        params: &pf_params,
        particles: 100,
        replicates: 1,
        dt: 1.0,
        obs_block: "",
        flow_indices: &[],
        seed: 7,
    })
    .expect("pfilter");
    out.push(("pfilter", pf.levels[0].hash, pf.run_id));

    // survey — levels[0] is the `model` level.
    let bounds = [("beta".to_string(), 0.01_f64, 2.0_f64)];
    let fixed = [("N0".to_string(), 1000.0_f64)];
    let sv = resolve_survey(&SurveyCtx {
        model,
        ir_version: IRV,
        engine_version: ENGV,
        stem: "sir",
        data: &data,
        eval_method: "pfilter",
        eval_particles: 100,
        eval_replicates: 1,
        bounds: &bounds,
        fixed: &fixed,
        scenario: None,
        n_points: 32,
        seed: 7,
    })
    .expect("survey");
    out.push(("survey", sv.levels[0].hash, sv.run_id));

    // sim_ensemble — levels[0] is the `model` level.
    let cells = [EnsembleCell {
        scenario_label: "baseline".into(),
        process_seed: 1,
        draw_idx: 0,
        sim_run_id: ContentHash::from_bytes([9; 32]),
        traj_digest: ContentHash::from_bytes([10; 32]),
    }];
    let ens = resolve_sim_ensemble(&EnsembleCtx {
        model,
        ir_version: IRV,
        engine_version: ENGV,
        stem: "sir",
        backend: ForwardBackend::ChainBinomial,
        dt: 1.0,
        base_params: &params,
        cells: &cells,
    })
    .expect("sim_ensemble");
    out.push(("sim_ensemble", ens.levels[0].hash, ens.run_id));

    // profile — levels[0] is the `profile` base level (folds the model digest).
    let focal = [("beta".to_string(), 0.3_f64)];
    let grid = [("beta".to_string(), vec![0.1_f64, 0.3, 0.5])];
    let base_config = serde_json::json!({ "fixed": { "N0": 1000.0 } });
    let method_config = serde_json::json!({ "algorithm": "if2" });
    let pr = resolve_profile_point(&ProfilePointCtx {
        model,
        ir_version: IRV,
        engine_version: ENGV,
        stem: "sir",
        method_name: "if2",
        data: &data,
        base_config: &base_config,
        method_config: &method_config,
        focal: &focal,
        grid: &grid,
        seed: 7,
        start_index: 0,
        deps: vec![],
    })
    .expect("profile");
    out.push(("profile", pr.levels[0].hash, pr.run_id));

    out
}

#[test]
fn presentation_fields_are_inert_on_every_cas_kind() {
    // Two models identical but for the two pure-presentation fields.
    let mut tsv = tiny_model();
    tsv.output.format = "tsv".into();
    tsv.simulation.time_semantics = "continuous".into();
    let mut parquet = tiny_model();
    parquet.output.format = "parquet".into();
    parquet.simulation.time_semantics = "calendar".into();

    for ((k, lh_a, rid_a), (_, lh_b, rid_b)) in
        all_kind_identities(&tsv).into_iter().zip(all_kind_identities(&parquet))
    {
        assert_eq!(
            lh_a, lh_b,
            "[{k}] output.format / time_semantics must NOT affect the model-bearing \
             level hash — they are presentation, not identity (gh#442)"
        );
        assert_eq!(rid_a, rid_b, "[{k}] ... and therefore must NOT re-key the run_id");
    }

    // Each field on its own must be inert too, so a half-fix (one field
    // stripped, the other not) still fails here.
    for field in ["output.format", "simulation.time_semantics"] {
        let mut only = tiny_model();
        if field == "output.format" {
            only.output.format = "parquet".into();
        } else {
            only.simulation.time_semantics = "calendar".into();
        }
        for ((k, _, rid_base), (_, _, rid_only)) in
            all_kind_identities(&tiny_model()).into_iter().zip(all_kind_identities(&only))
        {
            assert_eq!(rid_base, rid_only, "[{k}] `{field}` alone must not re-key the run_id");
        }
    }

    // Negative control: a genuinely structural edit MUST re-key every kind —
    // otherwise the assertions above could pass vacuously (e.g. if the model
    // stopped contributing to identity at all).
    let mut renamed = tiny_model();
    renamed.name = "different".into();
    for ((k, lh_a, rid_a), (_, lh_b, rid_b)) in
        all_kind_identities(&tiny_model()).into_iter().zip(all_kind_identities(&renamed))
    {
        assert_ne!(lh_a, lh_b, "[{k}] a model rename MUST move the model-bearing level");
        assert_ne!(rid_a, rid_b, "[{k}] a model rename MUST re-key the run_id");
    }
}

#[test]
fn cas_identity_pins() {
    // Absolute run_id pins for every model-bearing kind, over `tiny_model()`
    // with every other input fixed. Two jobs:
    //
    //   - `sim` / `fit` are pinned to their PRE-gh#442 values. gh#442 sanctions
    //     a re-key of the four batch kinds and nothing else; if this refactor
    //     (or a later one) moves sim or fit, that is collateral, and it fails
    //     here rather than silently in the field.
    //   - `pfilter` / `survey` / `sim_ensemble` / `profile` are pinned to their
    //     POST-gh#442 values — the deliberate re-key, recorded so it can never
    //     recur silently.
    //
    // A move here is a deliberate, reviewed re-key: say which kinds move and
    // why, then re-pin.
    let expected: &[(&str, &str)] = &[
        // Unchanged by gh#442 (verified: this is the value the pre-fix build
        // produced for the same fixture).
        ("sim", "4893a3eeab75a4216b8a365d8ebb445ee8577d26c31d28514432f6bbdda73342"),
        ("fit", "c2707d3d973cbdaf9c0d5afc553264ca59f6002c60055b5957f31aa2431f673f"),
        // Re-keyed by gh#442: these four hashed the RAW model, so their `model`
        // level folded `output.format = "tsv"` / `time_semantics = "continuous"`.
        ("pfilter", "08df1f4d17f0b2a4428202b055bc1cc7c9cc5c6573fe2c3387e95bfdbe55ad3e"),
        ("survey", "7815cf6ee9be362f1de0f444e8d6e60c262e53d70d729459599d9bdfc6c89bc8"),
        ("sim_ensemble", "cdc28e4c0e8c7335b4903460e705cb3f606e89bd42076621811f43e0951ef123"),
        ("profile", "a9f236851af534936300f1a048ad3c26b41a27976d154c519ed68ebb1e6a9873"),
    ];
    // Compared as whole lists so a failure reports EVERY kind that moved, not
    // just the first — the re-key scope is the thing under review.
    let actual: Vec<(&str, String)> =
        all_kind_identities(&tiny_model()).into_iter().map(|(k, _, rid)| (k, rid.to_hex())).collect();
    let expected: Vec<(&str, String)> =
        expected.iter().map(|(k, h)| (*k, (*h).to_string())).collect();
    assert_eq!(
        actual, expected,
        "a run_id moved — this RE-KEYS the store for that kind. If deliberate, \
         state which kinds re-key and why, then re-pin (gh#442)."
    );
}

#[test]
fn gh442_did_not_re_key_sim_or_fit() {
    // The no-collateral gate, as a property rather than a captured constant.
    //
    // The PRE-gh#442 sim/fit path was `ModelDigest::from_model(&normalize(m))`
    // — the caller stripped, then hashed. Resolving an already-stripped model
    // therefore reproduces exactly what the old code computed. Post-fix,
    // `from_model` strips internally, so resolving the RAW model must give the
    // same answer. Equality on `sim` and `fit` is the statement "gh#442 moved
    // no sim or fit key"; it holds because the strip is idempotent (assigns a
    // constant), which is pinned in
    // `runid::ir_hash::tests::normalization_is_idempotent_and_sim_fit_bytes_unchanged`.
    //
    // Non-vacuous: `tiny_model()` carries `format = "tsv"` and
    // `time_semantics = "continuous"`, so the two models genuinely differ.
    let raw = tiny_model();
    assert!(!raw.output.format.is_empty() && !raw.simulation.time_semantics.is_empty());
    let mut pre_stripped = tiny_model();
    pre_stripped.output.format = String::new();
    pre_stripped.simulation.time_semantics = String::new();

    let now = all_kind_identities(&raw);
    let old_path = all_kind_identities(&pre_stripped);
    for ((k, _, rid_now), (_, _, rid_old)) in now.into_iter().zip(old_path) {
        // Every kind agrees post-fix — but `sim` / `fit` agreeing is the load-
        // bearing claim: those two took the pre-strip path before gh#442, so
        // this says their stored keys did not move.
        assert_eq!(
            rid_now, rid_old,
            "[{k}] resolving a pre-stripped model (the pre-gh#442 sim/fit code path) \
             must give the same run_id as resolving the raw model — for sim/fit this \
             IS the no-collateral-re-key guarantee"
        );
    }
}
