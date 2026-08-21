//! Sanity tests for the derived digest/leaf types: the derive flows the
//! canonical rules through real shapes (FiniteF64, BTreeMap, nested digests,
//! provenance-skip, the hand-written `Deps` set ordering).

use std::collections::{BTreeMap, BTreeSet};

use super::*;
use crate::float::FiniteF64;
use crate::hash::{ContentAddressed, ContentHash};
use crate::kind::ArtifactKind;

fn fid(b: u8) -> FiniteF64 {
    FiniteF64::new(b as f64).unwrap()
}

// ── Anti-drift encoding golden (gh#241 §4.2 / review attack 5) ───────────────
//
// `macro_eq` pins derive≡hand (a *relative* check) — it cannot catch a change to
// the canonical encoding itself (field order, framing, HASH_VERSION,
// schema_version, the enum/FiniteF64/BTreeMap rules) mirrored into both sides.
// These pin the *absolute* hex of representative leaf types + the composed
// run_ids, so any such change fails loudly here. A move is a deliberate,
// reviewed re-key (bump HASH_VERSION / a type's schema_version and re-pin),
// never silent.

#[cfg(test)]
fn golden_fixtures() -> Vec<(&'static str, ContentHash)> {
    let model = ModelDigest {
        ir: ContentHash::from_bytes([5; 32]),
        ir_version: "0.7".into(),
        engine: EngineVersion("0.3.0".into()),
    };
    let config = SimConfig {
        backend: Backend::ChainBinomial,
        dt: fid(1),
        t_start: fid(0),
        t_end: fid(100),
        output: ResolvedOutputSchedule::Regular { start: fid(0), step: fid(1) },
        calendar: CalendarMode::Numeric,
        allow_degenerate_rates: false,
        no_flows: false,
        columns: BTreeSet::new(),
        init_state: None,
    };
    let mut values: BTreeMap<ParamId, FiniteF64> = BTreeMap::new();
    values.insert(ParamId("beta".into()), fid(2));
    let params = ResolvedParams { values, tables: vec![DataDigest(ContentHash::from_bytes([8; 32]))] };
    let scenario = ResolvedScenario {
        enabled: [InterventionId("vacc".into())].into_iter().collect(),
        disabled: BTreeSet::new(),
        patch: BTreeMap::new(),
    };
    let seed = Seed { process_seed: 42, base_seed: 7 };
    let fit = FitDigest {
        model: model.clone(),
        data: vec![DataDigest(ContentHash::from_bytes([1; 32]))],
        holdout_data: vec![DataDigest(ContentHash::from_bytes([2; 32]))],
        fit_toml: ContentHash::from_bytes([3; 32]),
        engine: EngineVersion("0.3.0".into()),
    };
    let stage = StageConfig {
        config: ContentHash::from_bytes([7; 32]),
        obs_block: String::new(),
        flow_indices: vec![],
        target_length: 50,
        obs_alignment: ResolvedObsAlignment::Snap,
    };
    let sim_levels = vec![
        model.content_hash(),
        config.content_hash(),
        params.content_hash(),
        scenario.content_hash(),
        seed.content_hash(),
    ];
    let fit_levels = vec![fit.content_hash(), stage.content_hash(), seed.content_hash()];
    vec![
        ("Backend::ChainBinomial", Backend::ChainBinomial.content_hash()),
        ("CalendarMode::Numeric", CalendarMode::Numeric.content_hash()),
        ("ResolvedObsAlignment::Snap", ResolvedObsAlignment::Snap.content_hash()),
        ("Seed", seed.content_hash()),
        ("ModelDigest", model.content_hash()),
        ("SimConfig", config.content_hash()),
        ("ResolvedParams", params.content_hash()),
        ("ResolvedScenario", scenario.content_hash()),
        ("FitDigest", fit.content_hash()),
        ("StageConfig", stage.content_hash()),
        ("run_id(Sim)", crate::run_id(ArtifactKind::Sim, &sim_levels)),
        ("run_id(FitStage)", crate::run_id(ArtifactKind::FitStage, &fit_levels)),
    ]
}

#[test]
fn canonical_encoding_is_pinned() {
    // Pinned literals. A change here is a deliberate, reviewed re-key: bump
    // HASH_VERSION (whole store) or the relevant type's schema_version, then
    // re-pin. It must never move silently.
    let expected: &[(&str, &str)] = &[
        ("Backend::ChainBinomial", "cf36b902f013ab2f0c2e5f074e66878a495990872c50bce0aa2683d1001e3d5c"),
        ("CalendarMode::Numeric", "ababcfd6a697dafc2981da21816f0555f18d5633454985ab7ff76d970c00646a"),
        ("ResolvedObsAlignment::Snap", "c8d06c17fd493405f2f220666705f80193775801258a0be28ae36fb8d475a809"),
        ("Seed", "dd2fb5245233d07fc6a715d0e7683b52767252050a29e5dbbb9921e1ba61397d"),
        ("ModelDigest", "50b2c476d23c4a7923f414bed47b0fc59757e17b18048ba82af36d89267e9447"),
        ("SimConfig", "2da6a6c529f909dfef38be64e58722ff8d30a8c760ebcf70b5828476f3bb1d6d"),
        ("ResolvedParams", "3cae27d97f964a1a6e654228dcf0ced7407f2937792ea7b2c20b724628d1ec10"),
        ("ResolvedScenario", "bdc6a70dd99429b0adc3646f4f089279a69b6101ece3ab5bb3ebf31b7a32c0ca"),
        ("FitDigest", "a87aad94ad68c799fd558a872ebeb7de8507c3da867c6553ad198af38882d7e7"),
        ("StageConfig", "f6eb2654d2393f1365ba8610b3a80a5a5772e752b54179dc208f82b995f067df"),
        ("run_id(Sim)", "5ff91d4772e77bd6f896a0e7ff516968219acc1a48f0e75aa176f395f5e1582a"),
        ("run_id(FitStage)", "882fceab6e6120667091cc2f4c02a8a035645e46f4d81b86643ea591ee101836"),
    ];
    let actual = golden_fixtures();
    assert_eq!(actual.len(), expected.len(), "fixture/expected count drift");
    for ((name, h), (ename, ehex)) in actual.iter().zip(expected) {
        assert_eq!(name, ename, "fixture order drift");
        assert_eq!(
            &h.to_hex(),
            ehex,
            "canonical encoding of `{name}` changed — this RE-KEYS the store. If \
             intentional, bump HASH_VERSION (or the type's schema_version) and re-pin; \
             never let this move silently (gh#241 §4.2)."
        );
    }
}

#[test]
fn seed_hashes_process_seed_not_base() {
    // The seed level hashes the resolved process_seed; the base seed is
    // provenance (path label only). Two seeds with the same process_seed but
    // different base must hash identically; differing process_seed must not.
    let a = Seed { process_seed: 42, base_seed: 7 };
    let b = Seed { process_seed: 42, base_seed: 999 };
    let c = Seed { process_seed: 43, base_seed: 7 };
    assert_eq!(a.content_hash(), b.content_hash(), "base_seed is provenance");
    assert_ne!(a.content_hash(), c.content_hash(), "process_seed is semantic");
}

#[test]
fn resolved_params_is_value_order_invariant() {
    // BTreeMap is canonical by construction; build the same logical map two
    // ways and confirm equal hashes, plus a value-change negative control.
    let mk = |order_ab: bool| {
        let mut values: BTreeMap<ParamId, FiniteF64> = BTreeMap::new();
        if order_ab {
            values.insert(ParamId("beta".into()), fid(2));
            values.insert(ParamId("alpha".into()), fid(1));
        } else {
            values.insert(ParamId("alpha".into()), fid(1));
            values.insert(ParamId("beta".into()), fid(2));
        }
        ResolvedParams { values, tables: vec![] }
    };
    assert_eq!(mk(true).content_hash(), mk(false).content_hash());

    let mut changed = mk(true);
    changed.values.insert(ParamId("beta".into()), fid(3));
    assert_ne!(mk(true).content_hash(), changed.content_hash());
}

fn artifact_ref(run_id_byte: u8, digest_byte: u8) -> ArtifactRef {
    ArtifactRef {
        run_id: ContentHash::from_bytes([run_id_byte; 32]),
        kind: ArtifactKind::Sim,
        artifact: "traj.tsv".into(),
        digest: ContentHash::from_bytes([digest_byte; 32]),
    }
}

#[test]
fn deps_is_a_set_sorted_by_run_id() {
    // Reordering independent upstreams must not change the deps hash.
    let r1 = artifact_ref(1, 10);
    let r2 = artifact_ref(2, 20);
    let ab = Deps(vec![r1.clone(), r2.clone()]);
    let ba = Deps(vec![r2, r1]);
    assert_eq!(ab.content_hash(), ba.content_hash(), "deps order must not matter");

    // A different upstream identity changes the hash.
    let diff = Deps(vec![artifact_ref(1, 10), artifact_ref(3, 30)]);
    assert_ne!(ab.content_hash(), diff.content_hash());
}

#[test]
fn artifact_ref_kind_is_provenance_digest_is_semantic() {
    // kind is display-only (skipped); the consumed file's digest is folded in.
    let mut a = artifact_ref(1, 10);
    let mut b = artifact_ref(1, 10);
    a.kind = ArtifactKind::Sim;
    b.kind = ArtifactKind::FitStage;
    assert_eq!(a.content_hash(), b.content_hash(), "ArtifactRef.kind is provenance");

    let c = artifact_ref(1, 99);
    assert_ne!(a.content_hash(), c.content_hash(), "consumed digest is semantic");
}

#[test]
fn fit_digest_holdout_content_is_keyed() {
    // gh#190: editing a holdout file's *content* (same path) changes
    // `holdout_data` and must change the FitDigest — a stale fit (and its
    // held-out predictive score) must not be reused under an unchanged run_id.
    let model = ModelDigest {
        ir: ContentHash::from_bytes([5; 32]),
        ir_version: "0.9".into(),
        engine: EngineVersion("0.3.0".into()),
    };
    let base = FitDigest {
        model: model.clone(),
        data: vec![DataDigest(ContentHash::from_bytes([1; 32]))],
        holdout_data: vec![DataDigest(ContentHash::from_bytes([2; 32]))],
        fit_toml: ContentHash::from_bytes([3; 32]),
        engine: EngineVersion("0.3.0".into()),
    };
    // Only the holdout stream's content digest differs (same training data,
    // same fit.toml blob → same holdout *path*).
    let mut edited = base.clone();
    edited.holdout_data = vec![DataDigest(ContentHash::from_bytes([9; 32]))];
    assert_ne!(
        base.content_hash(),
        edited.content_hash(),
        "a holdout file's content must be folded into the fit identity (gh#190) — \
         editing it (same path) must change the run_id"
    );

    // No spurious sensitivity: identical holdout content → identical hash.
    assert_eq!(base.content_hash(), base.clone().content_hash());
}

#[test]
fn stage_config_obs_alignment_is_keyed() {
    // gh#189: the resolved obs alignment (snap vs exact) drives the posterior,
    // so two otherwise-identical stage configs differing only in it must get
    // distinct stage hashes — exact and snap cannot collide in the CAS store.
    let base = StageConfig {
        config: ContentHash::from_bytes([7; 32]),
        obs_block: String::new(),
        flow_indices: vec![],
        target_length: 50,
        obs_alignment: ResolvedObsAlignment::Snap,
    };
    let mut flipped = base.clone();
    flipped.obs_alignment = ResolvedObsAlignment::Exact;
    assert_ne!(
        base.content_hash(),
        flipped.content_hash(),
        "the resolved obs_alignment must be folded into the stage identity (gh#189) — \
         a snap fit and an exact fit at the same config must not collide"
    );
    assert_eq!(base.content_hash(), base.clone().content_hash());
}

#[test]
fn trajectory_input_display_is_provenance() {
    let model = ModelDigest {
        ir: ContentHash::from_bytes([5; 32]),
        ir_version: "0.7".into(),
        engine: EngineVersion("0.3.0".into()),
    };
    let config = SimConfig {
        backend: Backend::ChainBinomial,
        dt: fid(1),
        t_start: fid(0),
        t_end: fid(100),
        output: ResolvedOutputSchedule::Regular { start: fid(0), step: fid(1) },
        calendar: CalendarMode::Numeric,
        allow_degenerate_rates: false,
        no_flows: false,
        columns: BTreeSet::new(),
        init_state: None,
    };
    let params = ResolvedParams { values: BTreeMap::new(), tables: vec![] };
    let scenario = ResolvedScenario {
        enabled: BTreeSet::new(),
        disabled: BTreeSet::new(),
        patch: BTreeMap::new(),
    };
    let seed = Seed { process_seed: 1, base_seed: 1 };

    let base = TrajectoryInput {
        model: model.clone(),
        config: config.clone(),
        params: params.clone(),
        scenario: scenario.clone(),
        seed,
        display: RunProvenance { argv: vec!["camdl".into()], label: None },
    };
    let other = TrajectoryInput {
        model,
        config,
        params,
        scenario,
        seed,
        display: RunProvenance {
            argv: vec!["totally".into(), "different".into()],
            label: Some("a label".into()),
        },
    };
    assert_eq!(
        base.content_hash(),
        other.content_hash(),
        "the provenance display field must not affect identity"
    );

    // Negative control: a semantic change (degenerate-rates flag) does.
    let mut flipped = base.clone();
    flipped.config.allow_degenerate_rates = true;
    assert_ne!(base.content_hash(), flipped.content_hash());
}

#[test]
fn output_view_is_keyed_into_config() {
    // gh#156: `--no-flows` / `--columns` change the leaf's *bytes* (a column
    // subset), so a content-addressed leaf cannot share a `run_id` with the
    // full one — the view rides the config-level identity.
    let base = SimConfig {
        backend: Backend::ChainBinomial,
        dt: fid(1),
        t_start: fid(0),
        t_end: fid(100),
        output: ResolvedOutputSchedule::Regular { start: fid(0), step: fid(1) },
        calendar: CalendarMode::Numeric,
        allow_degenerate_rates: false,
        no_flows: false,
        columns: BTreeSet::new(),
        init_state: None,
    };

    let mut no_flows = base.clone();
    no_flows.no_flows = true;
    assert_ne!(base.content_hash(), no_flows.content_hash(),
        "--no-flows must re-key the config level");

    let mut cols = base.clone();
    cols.columns = ["S".to_string(), "I".to_string()].into_iter().collect();
    assert_ne!(base.content_hash(), cols.content_hash(),
        "--columns must re-key the config level");

    // The allow-list is a set: insertion order is identity-inert (emitted
    // column order follows the model, not the list).
    let mut cols_rev = base.clone();
    cols_rev.columns = ["I".to_string(), "S".to_string()].into_iter().collect();
    assert_eq!(cols.content_hash(), cols_rev.content_hash(),
        "the --columns allow-list is order-invariant");
}

#[test]
fn init_state_is_keyed_into_config() {
    // gh#641: a run seeded from a filtered-state file computes a different
    // trajectory from the same model at the same θ and seed. If the file's
    // bytes did not re-key, the store would serve a stale forecast whenever
    // the state changed under an unchanged model — the exact silent-wrong the
    // CAS exists to prevent.
    let base = SimConfig {
        backend: Backend::ChainBinomial,
        dt: fid(1),
        t_start: fid(0),
        t_end: fid(100),
        output: ResolvedOutputSchedule::Regular { start: fid(0), step: fid(1) },
        calendar: CalendarMode::Numeric,
        allow_degenerate_rates: false,
        no_flows: false,
        columns: BTreeSet::new(),
        init_state: None,
    };
    let seeded = |ensemble: u8, row: u64| {
        let mut c = base.clone();
        c.init_state = Some(InitStateDigest {
            ensemble: DataDigest(ContentHash::from_bytes([ensemble; 32])),
            row,
        });
        c
    };

    // Seeding at all re-keys: `init {}` and a restored state are different runs.
    assert_ne!(base.content_hash(), seeded(1, 0).content_hash(),
        "--init-state must re-key the config level");

    // TWO DIFFERENT ORIGIN ENSEMBLES → DISTINCT IDENTITY. The headline
    // requirement, and the same one for both sources: a state file's bytes and
    // a fit's resolved (θ, X) join both arrive here as a content digest.
    assert_ne!(seeded(1, 0).content_hash(), seeded(2, 0).content_hash(),
        "a different origin ensemble must produce a distinct config-level hash");

    // Two rows of the SAME ensemble → distinct identity. Needed on top of the
    // seed level because two cells can share a process_seed (`--seeds 7,7`)
    // while restoring different states; without the row in the key those cells
    // collide and the store serves one trajectory for both.
    assert_ne!(seeded(1, 0).content_hash(), seeded(1, 1).content_hash(),
        "a different restored row must produce a distinct config-level hash");

    // Same ensemble, same row → same identity (the cache still works).
    assert_eq!(seeded(1, 3).content_hash(), seeded(1, 3).content_hash());
}

/// gh#697: the fit-sourced origin ensemble is keyed by its own CONTENT — the
/// origin time, and every selected draw's `(chain, draw)` key plus restored
/// values, in selection order.
///
/// Content rather than provenance (the fit's `run_id` + stage + selection
/// rule) because provenance keying means enumerating every knob that changes
/// which draws are selected: miss one and two different clouds collide on one
/// cache entry, which is the silent-wrong the store exists to prevent. Each
/// axis below is one way the restored states can change without the model, θ,
/// seed or horizon changing.
#[test]
fn a_fit_sourced_origin_ensemble_is_keyed_by_its_content() {
    let row = |chain: u64, draw: u64, s: i64| InitStateRow {
        chain,
        draw,
        counts: vec![s, 10, 0],
        reals: vec![FiniteF64::new(0.5).unwrap()],
    };
    let base = InitStateEnsemble {
        origin_t: FiniteF64::new(56.0).unwrap(),
        rows: vec![row(0, 20, 9000), row(1, 25, 8800)],
    };

    // Identical content → identical digest: re-running the same command against
    // the same fit must HIT the cache, not miss it.
    let same = InitStateEnsemble {
        origin_t: FiniteF64::new(56.0).unwrap(),
        rows: vec![row(0, 20, 9000), row(1, 25, 8800)],
    };
    assert_eq!(base.content_hash(), same.content_hash(),
        "an unchanged paired ensemble must re-key to itself");

    // A changed restored VALUE — a re-fit under an unchanged model.
    let bumped = InitStateEnsemble {
        origin_t: base.origin_t,
        rows: vec![row(0, 20, 9001), row(1, 25, 8800)],
    };
    assert_ne!(base.content_hash(), bumped.content_hash(),
        "a changed origin state must re-key");

    // A changed ORDER — the same states assigned to different draws. The cloud
    // looks identical and every trajectory differs; without this, a shuffled
    // pairing would be served the correct pairing's cached leaves.
    let reordered = InitStateEnsemble {
        origin_t: base.origin_t,
        rows: vec![row(1, 25, 8800), row(0, 20, 9000)],
    };
    assert_ne!(base.content_hash(), reordered.content_hash(),
        "a reordered ensemble is a different (θ, X) assignment and must re-key");

    // A changed (chain, draw) KEY with identical values — the same states
    // attributed to different posterior draws.
    let rekeyed = InitStateEnsemble {
        origin_t: base.origin_t,
        rows: vec![row(0, 21, 9000), row(1, 25, 8800)],
    };
    assert_ne!(base.content_hash(), rekeyed.content_hash(),
        "the same state under a different draw key must re-key");

    // A changed SUBSET — one more forkable draw is a different cloud.
    let bigger = InitStateEnsemble {
        origin_t: base.origin_t,
        rows: vec![row(0, 20, 9000), row(1, 25, 8800), row(1, 30, 8700)],
    };
    assert_ne!(base.content_hash(), bigger.content_hash(),
        "a different forkable subset must re-key");

    // A changed ORIGIN TIME with identical states — the same counts read at a
    // different instant is a different forecast.
    let later = InitStateEnsemble {
        origin_t: FiniteF64::new(63.0).unwrap(),
        rows: base.rows.clone(),
    };
    assert_ne!(base.content_hash(), later.content_hash(),
        "a different forecast origin must re-key");

    // Domain separation from the file source: a `--init-state FILE` digest is
    // SHA-256 over the file's raw bytes, this is the structural hash of a
    // distinct named type, so the two cannot alias without a SHA-256 collision.
    assert_ne!(
        base.content_hash(),
        ContentHash::digest_bytes(b"# camdl-final-state v1\tt=56\n"),
        "the two origin-ensemble producers must not share a digest by construction"
    );
}
