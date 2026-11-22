//! Gate properties of the old store's identities: the model-identity gates
//! are cross-kind, asserted for every kind on `runid` whose identity folds a
//! model (`fit`, `profile`), and the fail-closed reuse rule of the writer
//! seam.


use super::*;
    use runid::ContentAddressed;
use ir::model::{
    InitialConditions, Model, OutputConfig, OutputSchedule, RegularOutputSchedule, SimulationConfig,
};

pub(crate) fn tiny_model() -> Model {
    Model {
        constants: vec![],
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
        initial_conditions: InitialConditions::default(),
        output: OutputConfig {
            times: OutputSchedule::Regular(RegularOutputSchedule { start: 0.0, step: 1.0 }),
            trajectory: true,
            observations: false,
        },
        simulation: SimulationConfig {
            t_start: 0.0,
            t_end: 100.0,
            dt: Some(1.0),
            rng_seed: None,
            integrator: Default::default(),
            t_end_anchor: None, t_start_anchor: None,
        },
        presets: vec![],
        model_structure: None,
        balance: None,
        identity_tracked_compartments: vec![], quantities: vec![], contrasts: vec![],
    }
}



/// The `model`-level digest hash under fixed versions, the unit the gradient
/// inertness tests compare. Goes through the real `ModelDigest::from_model`.
fn model_digest_hash(m: &Model) -> ContentHash {
    runid::inputs::ModelDigest::from_model(m, "0.7".into(), runid::inputs::EngineIdentity::of_build("0.3.0")).content_hash()
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





// ── gh#241 PR F: input-surface differential harness ───────────────────────────
//
// The identity guarantee made executable: a SEMANTIC input mutation MUST re-key
// the `run_id`; a PRESENTATION / provenance mutation MUST NOT. This complements
// the anti-drift golden (which pins absolute encodings) with parametric
// sensitivity + inertness — so a future change that leaks a presentation field
// into identity, or drops a semantic one, fails here, not silently in the field.





// ── Model identity across every CAS kind ──────────────────────────────────────
//
// Every kind builds its model level through `ModelDigest::from_model`. Two
// gates here:
//   1. `model_name_is_inert_on_every_cas_kind`: the model's name moves no
//      kind's identity, and a structural edit moves every kind's.
//   2. `cas_identity_pins`: the absolute run_id hexes, so any re-key of any
//      kind is deliberate and reviewed.

use crate::fit::config_v2::FitConfig;
use crate::profile_cas::{resolve_profile_point, ProfilePointCtx};

const IRV: &str = "0.7";
const ENGV: &str = "0.3.0+test";

/// A minimal fit config with no `[data]` streams, so `fit_level_digest` needs
/// no files on disk and the model is its only model-bearing input. `starts`
/// is spelled so the method identity is takeable without a model to resolve
/// the default against.
fn fit_config() -> FitConfig {
    FitConfig::from_toml_str(
        "[model]\ncamdl = \"models/sir.camdl\"\n\
         [estimate]\nbeta = { bounds = [0.01, 2.0] }\n\
         [fixed]\nN0 = 1000000\n\
         [method]\nalgorithm = \"if2\"\nbackend = \"chain_binomial\"\n\
         chains = 4\nparticles = 1000\niterations = 50\ncooling = 0.70\n\
         starts = \"uniform_unconstrained\"\n",
    )
    .expect("fixture fit config must parse")
}

/// Resolve `model` through EVERY model-bearing CAS identity path. Returns
/// `(kind, model-bearing level hash, leaf run_id)` in a fixed order. Every
/// non-model input is held constant, so any hash motion is the model's.
fn all_kind_identities(model: &Model) -> Vec<(&'static str, ContentHash, ContentHash)> {
    let mut out = Vec::new();

    // fit — `resolve_fit_method`; levels[0] is the `fit` level (folds the model
    // digest alongside the data/config digests).
    let cfg = fit_config();
    let method = cfg.inference.method.clone().expect("fixture method");
    let fit = crate::fit::cas::resolve_fit_method(&crate::fit::cas::FitMethodCtx {
        model,
        fit_stem: "sir",
        ir_version: IRV,
        engine: &runid::inputs::EngineIdentity::of_build(ENGV),
        problem: &cfg.problem,
        data: &Default::default(),
        method: &method,
        seed: 7,
        deps: vec![],
    })
    .expect("fit");
    out.push(("fit", fit.levels[0].hash, fit.run_id));

    let data: Vec<(String, String)> =
        vec![("cases".to_string(), ContentHash::digest_bytes(b"cases").to_hex())];



    // profile — levels[0] is the `profile` base level (folds the model digest).
    let focal = [("beta".to_string(), 0.3_f64)];
    let grid = [("beta".to_string(), vec![0.1_f64, 0.3, 0.5])];
    // Hashed through the same helper production uses. Serializing a `Value`
    // yields that same `Value`, so these digests are identical to the
    // pre-helper `digest_value` ones — which is exactly why the pinned
    // run_ids below must not move.
    let base_config = crate::fit::cas::canonical_config_hash(
        &serde_json::json!({ "fixed": { "N0": 1000.0 } }), &[]).unwrap();
    let method_config = crate::fit::cas::canonical_config_hash(
        &serde_json::json!({ "algorithm": "if2" }), &[]).unwrap();
    let pr = resolve_profile_point(&ProfilePointCtx {
        model,
        ir_version: IRV,
        engine: &runid::inputs::EngineIdentity::of_build(ENGV),
        stem: "sir",
        method_name: "if2",
        data: &data,
        base_config,
        method_config,
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
fn model_name_is_inert_on_every_cas_kind() {
    // The model's `name` is the source file's stem, a label: it is inert in
    // every kind's identity, like the leaf directory label it also feeds.
    let mut renamed = tiny_model();
    renamed.name = "different".into();
    for ((k, lh_a, rid_a), (_, lh_b, rid_b)) in
        all_kind_identities(&tiny_model()).into_iter().zip(all_kind_identities(&renamed))
    {
        assert_eq!(lh_a, lh_b, "[{k}] the model name must not move the model-bearing level");
        assert_eq!(rid_a, rid_b, "[{k}] the model name must not re-key the run_id");
    }

    // Negative control: a genuinely structural edit MUST re-key every kind —
    // otherwise the assertions above could pass vacuously (e.g. if the model
    // stopped contributing to identity at all).
    let mut longer = tiny_model();
    longer.simulation.t_end = 200.0;
    for ((k, lh_a, rid_a), (_, lh_b, rid_b)) in
        all_kind_identities(&tiny_model()).into_iter().zip(all_kind_identities(&longer))
    {
        assert_ne!(lh_a, lh_b, "[{k}] a structural edit MUST move the model-bearing level");
        assert_ne!(rid_a, rid_b, "[{k}] a structural edit MUST re-key the run_id");
    }
}

#[test]
fn cas_identity_pins() {
    // Absolute run_id pins for every model-bearing kind, over `tiny_model()`
    // with every other input fixed. The job: a move here is a deliberate,
    // reviewed re-key — say which kinds move and why, then re-pin. Anything
    // that moves a kind NOT named in the reason is collateral, and it fails
    // here rather than silently in the field.
    //
    // gh#733 (ir/VERSION 0.34) moved ALL SIX at once, and it is the first
    // change that does. `initial_conditions` stopped being a three-variant enum
    // over the whole block and became one ordered map of per-compartment specs,
    // so its `ContentAddressed` bytes changed — a new `ir::model::InitSpec`
    // header, and a length-prefixed walk in DECLARATION order rather than the
    // sorted `write_str_map`. Every kind's `model` level folds
    // `initial_conditions`, so every kind re-keys. That the scope is "all six"
    // rather than "some" is itself the claim this list makes checkable: a
    // change to a field EVERY kind hashes cannot leave any of them behind.
    let expected: &[(&str, &str)] = &[
        // `fit` carries TWO re-keys composed: this one, and the
        // `ivp` -> `perturb_only_at_t0` rename, whose renamed TOML key
        // changes the fit canonical JSON's hashed bytes even at its `false`
        // default. The other five carry only this one.
        //
        // gh#833 then moved `sim` ALONE: `SimConfig` schema_version 4 -> 5,
        // because the sim leaf's `traj.tsv` gained `t_start`/`t_stop` and a
        // key must not name bytes a recompute no longer reproduces. The other
        // five kinds hold no trajectory table under a `SimConfig` key, so
        // they stay — that scope is the claim this pin checks.
        //
        // The `[stages]` → `[method]` split (proposal 2026-09-08, §8 item 22)
        // then moved `fit` ALONE, twice over: the fit level's canonical config
        // JSON lost the `fit_starts` leaf it had serialized as `null` since the
        // key was introduced, and the method level's payload — the algorithm
        // serialized whole, minus its extension dimension — dropped the four
        // chain-start fields (`init`, `init_mle`, `survey_path`,
        // `survey_top_k_n`) for the one `starts` rule, so every method leaf
        // re-keys. `pfilter`, `survey` and `profile` read the same problem
        // half through `Problem::load` and hash nothing that changed; `sim`
        // and `sim_ensemble` hash no fit config at all. That the other five
        // stay is the claim this pin checks.
        //
        // gh#573 then moved `sim` ALONE: `ResolvedScenario` schema_version
        // 1 -> 2 gained `scale`, so the scenario level hashes the effective
        // delta. Only `sim` hashes a `ResolvedScenario` here; `survey` keys its
        // scenario by name (beside the model digest, which covers every
        // preset), and this fixture's `sim_ensemble` folds fixed cell ids. A
        // real ensemble re-keys through its cells' `run_id`s.
        //
        // gh#583 then moved ALL SIX: the `ir` hash policy went to SV = 3, which
        // hashes the integrator, the anchored horizons and an observation
        // model's `stratum` / `covers` unconditionally rather than only when
        // non-default. The SV header and `SimulationConfig` are folded into
        // every kind's `model` level, so no kind can stay.
        //
        // gh#583 (item C) then moved `sim_ensemble` ALONE: its `params` level
        // is now the set of its cells' own `params` level hashes rather than a
        // base parameter map rebuilt beside them. The other five kinds hash no
        // ensemble level. (A real `sim` leaf also re-keys where its resolved
        // values differ from the old base map — a scenario that sets values, a
        // draw shadowed by `--param`, a grid cell over model defaults — but
        // this fixture hands `resolve_trajectory` its params directly.)
        //
        // gh#583 (item F) then moved `fit` ALONE: the fit level's canonical
        // config JSON drops the `[provenance]` key, which it had serialized as
        // `null` even when absent — so every fit re-keys, noted or not. No
        // other kind hashes the fit config blob.
        //
        // Proposal 2026-10-24-batch-to-simulate (§3 "Ensemble identity", §8)
        // then moved `sim_ensemble` ALONE: its `grid` level's per-cell tuple
        // became `(point_id, process_seed, sim_run_id)` — the arm's label, a
        // display string, left the key — and the level gained a plain
        // `design` field (`null` here). Every stored multi-cell ensemble
        // re-keys once and is rebuilt from cells the store already holds. No
        // other kind hashes an ensemble level, and no `sim` leaf changes.
        //
        // IR 0.41 (typed constants leave `parameters`) then moved EVERY kind:
        // `Model::hash_into` folds the new `constants` list unconditionally,
        // so its length prefix enters every kind's `model` level — the same
        // deliberate, version-bumped re-key as `per_eval_bindings` at 0.19.
        //
        // Deleting `[estimate].transform` (support-keyed transforms, D2)
        // moved `fit` only: the hashed fit config carried the field as
        // `"transform": null` on every estimate entry, and it is gone. Every
        // accepted override was redundant, so no stored output changes.
        //
        // The `ir` hash policy going to SV = 4 then moved EVERY kind: the
        // model's `name` (the source file's stem) left `Model::hash_into`, and
        // the SV header is folded into every kind's `model` level.
        //
        // The `[stages]` -> `[method]` rename of the `runid` input types then
        // moved `fit` and `profile` ALONE: `StageConfig`, `StageLevel` and
        // `FitStageInput` became `MethodConfig`, `MethodLevel` and
        // `FitMethodInput`, and `ProfileStage` became `ProfileMethod`. The
        // derive hashes each type's fully-qualified name as its domain tag, so
        // the same field bytes take a new tag. Only fits hash the first three
        // and only profiles hash the fourth; the other four kinds stay.
        //
        // HASH_VERSION 1 -> 2 (gh#1018) then moved EVERY kind: the version is
        // folded into every root hash and every `run_id`, so no kind can stay.
        // It declares two changes to what feeds identity — IR floats parsed
        // exactly, and a `from_mle` source keyed on its `start_values` — and
        // neither changes the bytes this fixture hashes (`tiny_model()` is
        // built in code, not parsed, and no fit here starts from_mle).
        //
        // HASH_VERSION 2 -> 3 (gh#1046) moved EVERY kind again, for the same
        // reason. It declares one change to what feeds identity: a model is
        // hashed by the bodies of its hoisted subtrees, not by their numbering,
        // and the table of them is no longer hashed as a list.
        //
        // The temporal-architecture migration's Stage 4 then moved `sim`
        // ALONE: `SimConfig` schema_version 5 -> 6, because the sim leaf's
        // `run.json` gained the model's `calendar` block. Only `sim` hashes a
        // `SimConfig` and records one; the other five stay.
        // HASH_VERSION 3 -> 4 (D43) moved EVERY kind once more. It declares one
        // change to what feeds identity: a fit's method level no longer carries
        // a resolved observation alignment. With the version held at 3 that
        // change moved `fit` alone (checked when the bump was made).
        //
        // HASH_VERSION 4 -> 5 (temporal-architecture Stage 2, D58) moved
        // EVERY kind again. It declares two changes: training rows decided by
        // period end (D38, D40), and the forcing-extrapolation override keyed
        // in the sim, pfilter, profile and survey identities. Off, as here, it
        // is omitted from the pfilter, survey and profile config levels; the
        // `sim` and `sim_ensemble` config level (`SimConfig`) carries it as a
        // field, so their bytes change beyond the version. `fit` also gained
        // the partition rule version in its `fit` level (`FitDigest`); no
        // other kind hashes a `FitDigest`.
        //
        // HASH_VERSION 5 -> 6 (D72) moved EVERY kind once more, for the
        // version alone: no level's encoding changed. It declares that the
        // bootstrap filter and IF2 no longer weight, resample, perturb or cool
        // at a stop with no observed value.
        //
        // HASH_VERSION 6 -> 7 (D77) moved every kind once more, for the
        // version alone. It declares that the bootstrap filter and IF2 no
        // longer resample on exactly equal weights, and that a binomial
        // `0` of `0` row is bound as a hole.
        //
        // HASH_VERSION 7 -> 8 (proposal 2026-11-17 §1b, §1c) moved every kind
        // once more, for the version alone: this fixture's fit is IF2, which
        // reads no gradient, and its pfilter and profile hold
        // `allow_degenerate_rates` at its default.
        //
        // ir/VERSION 0.48 (`simulation.t_start_anchor`) moved every kind once:
        // the model hash now folds the start anchor (absent here, one byte), so
        // every model level's bytes changed. Real runs re-key with the IR
        // version folded into their model digest anyway; this fixture pins a
        // fixed IR version, so the encoding is what moved it.
        //
        // The profile `seed` level hashing the job's process seed, derived
        // from the point's position in the name-sorted grid, moved `profile`
        // alone: this fixture's point is the second value of its grid, so
        // its process seed is no longer the base seed.
        //
        // ir/VERSION 0.49 (`output.format` and `simulation.time_semantics`
        // leave the IR) moved every kind once: the model hash no longer
        // writes the two empty strings it hashed for them, so every model
        // level's bytes changed.
        //
        // `sim` and `sim_ensemble` leave the list: `simulate` stores its
        // cells on the run store, whose ids are not runid's.
        ("fit", "937d67de229bc67c23bf862cdc345332dd145f1b66283d4e2635f6022851979f"),
        ("profile", "a3182ce7cbb9eae3f2f5f9bc2a782d2d6d4a5a19ca3921624f1f73ecdff5857b"),
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


/// gh#901: a profile-point leaf's levels are named
/// `profile / point / method / seed / start` — `method`, not `stage`, the same
/// word the fit store's middle level and the rest of the tool use.
///
/// The rename is identity-neutral, and the test says so in the only two places
/// it could bite. The `run_id` folds level *hashes* — no name, no label — and
/// the on-disk segment is `{label}-{hash8}`, also nameless. Both are asserted
/// here, so a future change that made a level's name identity-bearing fails
/// this test rather than silently orphaning every profile already on disk.
#[test]
fn a_profile_points_method_level_is_named_method_and_naming_it_rekeys_nothing() {
    let model = tiny_model();
    let base_config = crate::fit::cas::canonical_config_hash(
        &serde_json::json!({ "fixed": { "N0": 1000.0 } }), &[]).unwrap();
    let method_config = crate::fit::cas::canonical_config_hash(
        &serde_json::json!({ "algorithm": "if2" }), &[]).unwrap();
    let focal = [("beta".to_string(), 0.3_f64)];
    let grid = [("beta".to_string(), vec![0.1_f64, 0.3, 0.5])];
    let pr = resolve_profile_point(&ProfilePointCtx {
        model: &model,
        ir_version: IRV,
        engine: &runid::inputs::EngineIdentity::of_build(ENGV),
        stem: "sir",
        method_name: "if2",
        data: &[],
        base_config,
        method_config,
        focal: &focal,
        grid: &grid,
        seed: 7,
        start_index: 0,
        deps: vec![],
    })
    .expect("profile point resolves");

    let names: Vec<&str> = pr.levels.iter().map(|l| l.name.as_str()).collect();
    assert_eq!(names, vec!["profile", "point", "method", "seed", "start"],
        "the level a profile's sub-fit method rides is named `method`");

    let hashes: Vec<ContentHash> = pr.levels.iter().map(|l| l.hash).collect();
    assert_eq!(pr.run_id, runid::run_id(runid::ArtifactKind::ProfilePoint, &hashes),
        "run_id folds level hashes only — a level's NAME is not in it, so \
         renaming one cannot re-key a leaf");

    let method_level = pr.levels.iter().find(|l| l.name == "method").unwrap();
    assert_eq!(runid::segment(method_level),
        format!("if2-{}", method_level.hash.short8()),
        "the path segment is {{label}}-{{hash8}}; renaming the level moves no \
         directory on disk");
}

/// A profile point's run id and process seed depend on the set of grid cells,
/// not on the order its axes, its focal values or a listed axis's values are
/// given in. Distinct (point, start) pairs get distinct process seeds, also
/// past 1000 starts.
#[test]
fn profile_point_identity_and_seed_ignore_sweep_order() {
    let model = tiny_model();
    let base_config = crate::fit::cas::canonical_config_hash(
        &serde_json::json!({ "fixed": { "N0": 1000.0 } }), &[]).unwrap();
    let method_config = crate::fit::cas::canonical_config_hash(
        &serde_json::json!({ "algorithm": "if2" }), &[]).unwrap();
    let engine = runid::inputs::EngineIdentity::of_build(ENGV);
    let resolve = |focal: &[(String, f64)], grid: &[(String, Vec<f64>)], start: u32| {
        resolve_profile_point(&ProfilePointCtx {
            model: &model, ir_version: IRV, engine: &engine, stem: "sir",
            method_name: "if2", data: &[], base_config, method_config,
            focal, grid, seed: 7, start_index: start, deps: vec![],
        }).expect("profile point resolves")
    };
    let b = |v: f64| ("beta".to_string(), v);
    let g = |v: f64| ("gamma".to_string(), v);
    let grid_bg = [("beta".to_string(), vec![0.1, 0.3, 0.5]), ("gamma".to_string(), vec![0.05, 0.25])];
    let grid_gb = [("gamma".to_string(), vec![0.25, 0.05]), ("beta".to_string(), vec![0.5, 0.1, 0.3])];

    let a = resolve(&[b(0.3), g(0.05)], &grid_bg, 1);
    let z = resolve(&[g(0.05), b(0.3)], &grid_gb, 1);
    assert_eq!(a.run_id, z.run_id, "axis, focal and value order must not change the run id");
    assert_eq!(a.process_seed, z.process_seed, "nor the process seed");

    let mut seeds = std::collections::HashSet::new();
    for bv in [0.1, 0.3, 0.5] {
        for gv in [0.05, 0.25] {
            for start in [0, 1, 1000, 1001] {
                seeds.insert(resolve(&[b(bv), g(gv)], &grid_bg, start).process_seed);
            }
        }
    }
    assert_eq!(seeds.len(), 24, "every (point, start) must get its own process seed");
}

// ── Run ids of compiled goldens, pinned ──────────────────────────────────────
//
// Invariant 9 of `docs/dev/proposals/2026-11-13-model-structure-diff.md`, on
// the run-id side. `tests/fixtures/golden_run_ids.tsv` holds, for five compiled
// goldens, the model-bearing level hash and the leaf `run_id` of every CAS
// kind (`all_kind_identities`), recorded at the commit named in its header:
// the last one before the `declared` section of the IR envelope existed. This
// test recomputes them from the goldens as they are now and compares, so a
// change to a compiled golden that moves any run id fails here.
//
// What the pin does and does not say. Every non-model input is fixed, and so
// are the two version strings a run id folds beside the model (`IRV`, `ENGV`).
// A run id on a user's machine folds the real `ir/VERSION` and the engine
// build, so it changes at every IR version bump and every engine build by
// design. The pin holds those constant to isolate the one thing the declared
// section could have disturbed: what the compiled model contributes.
//
// Regenerated only by `make update-golden-run-ids`, and only when a re-key of
// these models is the subject of the change.

const GOLDEN_RUN_IDS_TABLE: &str = "tests/fixtures/golden_run_ids.tsv";

/// Plain; stratified with an indexed parameter, a table and a forcing; a `via`
/// staged transition; interventions and scenarios; an observation model.
const PINNED_GOLDENS: &[&str] = &[
    "sir_basic",
    "seir_spatial_5_inference",
    "ebola_outcome_hyper",
    "polio_age",
    "seir_observations",
];

fn golden_run_id_rows() -> Vec<String> {
    let golden_dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../../ocaml/golden");
    let mut rows = Vec::new();
    for name in PINNED_GOLDENS {
        let path = golden_dir.join(format!("{name}.ir.json"));
        let text = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
        let model = ir::from_str(&text).unwrap_or_else(|e| panic!("parse {}: {e}", path.display()));
        for (kind, level, run_id) in all_kind_identities(&model) {
            rows.push(format!("{name}\t{kind}\t{}\t{}", level.to_hex(), run_id.to_hex()));
        }
    }
    rows
}

#[test]
fn golden_run_ids_match_the_reference_table() {
    let table = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join(GOLDEN_RUN_IDS_TABLE);
    let rows = golden_run_id_rows();

    if let Ok(commit) = std::env::var("CAMDL_CAPTURE_GOLDEN_RUN_IDS") {
        let mut out = String::new();
        out.push_str("# Model-bearing level hash and run_id of every CAS kind, for five compiled goldens.\n");
        out.push_str("# Source: ocaml/golden/<golden>.ir.json (primary), resolved by all_kind_identities in\n");
        out.push_str("#   rust/crates/cli/src/resolve/tests.rs with every non-model input fixed, including\n");
        out.push_str("#   the IR version and engine strings a real run id folds.\n");
        out.push_str(&format!("# Recorded at commit: {commit}\n"));
        out.push_str("# Regenerate: make update-golden-run-ids\n");
        out.push_str("golden\tkind\tlevel_hash\trun_id\n");
        for row in &rows {
            out.push_str(row);
            out.push('\n');
        }
        std::fs::write(&table, out).unwrap_or_else(|e| panic!("write {}: {e}", table.display()));
        return;
    }

    let text = std::fs::read_to_string(&table).unwrap_or_else(|e| panic!("read {}: {e}", table.display()));
    let recorded: Vec<&str> = text
        .lines()
        .filter(|line| !line.starts_with('#') && !line.starts_with("golden\t"))
        .collect();
    assert_eq!(recorded.len(), rows.len(), "the table has one row per golden and kind");
    for (was, now) in recorded.iter().zip(&rows) {
        assert_eq!(
            was, now,
            "a pinned golden run id moved (columns: golden, kind, level hash, run id). \
             If re-keying these models is the subject of the change, say so and run \
             `make update-golden-run-ids`; otherwise this is collateral and must be fixed."
        );
    }
}

/// Fail closed (proposal 2026-11-17 §1a): a build whose engine fingerprint is
/// incomplete may share its `ENGINE_ID` with another build's code, so it never
/// serves a stored leaf. It computes, and the leaf it would have read is set
/// aside in `.quarantine/`, never deleted.
#[test]
fn a_build_with_an_incomplete_engine_identity_never_reads_a_stored_leaf() {
    use crate::version::EngineReuse;
    let levels = vec![
        crate::fit::cas::level("fit", "sir", crate::fit::cas::structural_level_hash(&"fit".to_string())),
        crate::fit::cas::level("method", "if2", crate::fit::cas::structural_level_hash(&"method".to_string())),
        crate::fit::cas::level("seed", "seed_1", crate::fit::cas::structural_level_hash(&"seed".to_string())),
    ];
    let artifact = ResolvedArtifact {
        kind: ArtifactKind::FitMethod,
        run_id: runid::run_id(ArtifactKind::FitMethod, &levels.iter().map(|l| l.hash).collect::<Vec<_>>()),
        levels,
        display_inputs: serde_json::Value::Null,
    };
    let root = crate::test_support::unique_temp_dir("fail_closed_reuse");
    let store = FsCasStore::new(&root);
    let calendar = ir::caltime::CalendarAxis::parse("days", None).unwrap();
    let meta = RecordMeta::new("0.7", "sir.camdl", None, calendar,
        &crate::forcing_extrapolation::ForcingPolicy::new(false));
    let claim = begin_resolved_write(&store, &root, &artifact, &meta).unwrap();
    claim.write("fit_state.toml", b"loglik = 0.0\n").unwrap();
    claim.finalize(serde_json::Value::Null).unwrap();

    let reusable = check_reuse_with(&store, &root, &artifact, WritePolicy::Reuse, EngineReuse::Reusable).unwrap();
    assert!(matches!(reusable, ReuseVerdict::CacheHit { .. }), "a determined build reuses the stored leaf");

    let unreusable = EngineReuse::Unreusable { missing: "rustc" };
    assert_eq!(engine_write_policy(WritePolicy::Reuse, unreusable), WritePolicy::Force);
    let verdict = check_reuse_with(&store, &root, &artifact, WritePolicy::Reuse, unreusable).unwrap();
    assert!(matches!(verdict, ReuseVerdict::MustRun), "an undetermined build computes");
    assert!(
        runid::find_leaf(&root, ArtifactKind::FitMethod, &artifact.levels).is_none(),
        "the stored leaf is no longer servable"
    );
    assert!(root.join(".quarantine").read_dir().unwrap().next().is_some(), "it was quarantined, not deleted");
    let _ = std::fs::remove_dir_all(&root);
}

/// Proposal 2026-11-17 §1b: two compiles of one forward model whose gradient
/// maps differ — what a camdlc with an edited autodiff emits — are one model,
/// and so one run for a method that never reads a gradient, and two runs for
/// a method that does. Asserted on the whole fit `run_id`, through
/// `resolve_fit_method`, for every method on each side.
#[test]
fn gradient_maps_re_key_the_fits_that_read_them_and_no_other() {
    use ir::deriv::{CompGradMap, DerivEntry};
    use ir::expr::{BinOp, Expr};
    use ir::transition::{DrawMethod, StoichiometryEntry, Transition};

    let compiled_with = |d_rate_d_beta: Expr| -> Model {
        let mut m = tiny_model();
        m.transitions = vec![Transition {
            name: "infection".into(),
            stoichiometry: vec![StoichiometryEntry("S".into(), -1), StoichiometryEntry("I".into(), 1)],
            rate: Expr::bin_op(BinOp::Mul, Expr::param("beta"), Expr::pop("I")),
            metadata: None,
            draw_method: DrawMethod::Poisson,
            rate_grad: std::collections::HashMap::from([("beta".to_string(), DerivEntry::Grad(d_rate_d_beta))]),
            rate_state_grad: CompGradMap(Default::default()),
            lineage: None,
        }];
        m
    };
    // The right derivative of beta·I, and a wrong one.
    let right = compiled_with(Expr::pop("I"));
    let wrong = compiled_with(Expr::bin_op(BinOp::Mul, Expr::const_(2.0), Expr::pop("I")));
    assert_eq!(model_digest_hash(&right), model_digest_hash(&wrong), "one forward model");

    let cfg = fit_config();
    let run_id = |model: &Model, method_toml: &str| {
        let method: crate::fit::config_v2::Method =
            toml::from_str(&format!("{method_toml}\nstarts = \"uniform_unconstrained\"\n")).expect("method toml");
        crate::fit::cas::resolve_fit_method(&crate::fit::cas::FitMethodCtx {
            model,
            fit_stem: "sir",
            ir_version: IRV,
            engine: &runid::inputs::EngineIdentity::of_build(ENGV),
            problem: &cfg.problem,
            data: &Default::default(),
            method: &method,
            seed: 7,
            deps: vec![],
        })
        .expect("fit identity")
        .run_id
    };
    let pgas = "algorithm = \"pgas\"\nbackend = \"chain_binomial\"\nchains = 2\nparticles = 100\nsweeps = 10";
    let reads = [
        ("pgas with nuts", format!("{pgas}\nuse_nuts = true")),
        ("nl-lbfgs", "algorithm = \"nl-lbfgs\"\nbackend = \"ode\"\nchains = 1".to_string()),
    ];
    let never_reads = [
        ("pgas --no-nuts", format!("{pgas}\nuse_nuts = false")),
        ("if2", "algorithm = \"if2\"\nbackend = \"chain_binomial\"\nchains = 2\nparticles = 100\n\
                 iterations = 10\ncooling = 0.7".to_string()),
        ("nl-sbplx", "algorithm = \"nl-sbplx\"\nbackend = \"ode\"\nchains = 1".to_string()),
        ("nl-bobyqa", "algorithm = \"nl-bobyqa\"\nbackend = \"ode\"\nchains = 1".to_string()),
    ];
    for (what, toml) in &reads {
        assert_ne!(run_id(&right, toml), run_id(&wrong, toml), "{what} reads the gradient maps: it must re-key");
    }
    for (what, toml) in &never_reads {
        assert_eq!(run_id(&right, toml), run_id(&wrong, toml), "{what} reads no gradient: it must not re-key");
    }
}
