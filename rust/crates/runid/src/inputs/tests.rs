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
        engine: EngineIdentity::of_build("0.3.0"),
    };
    let config = SimConfig {
        backend: Backend::ChainBinomial,
        dt: fid(1),
        t_start: fid(0),
        t_end: fid(100),
        output: ResolvedOutputSchedule::Regular { start: fid(0), step: fid(1) },
        calendar: CalendarMode::Numeric,
        allow_degenerate_rates: false,
        allow_forcing_extrapolation: false,
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
        scale: [(ParamId("beta".into()), vec![fid(2)])].into_iter().collect(),
    };
    let seed = Seed { process_seed: 42, base_seed: 7 };
    let fit = FitDigest {
        model: model.clone(),
        data: vec![DataDigest(ContentHash::from_bytes([1; 32]))],
        holdout_data: vec![DataDigest(ContentHash::from_bytes([2; 32]))],
        tables: vec![],
        fit_toml: ContentHash::from_bytes([3; 32]),
        engine: EngineIdentity::of_build("0.3.0"),
        partition_rule_version: 1,
    };
    let method = MethodConfig {
        config: ContentHash::from_bytes([7; 32]),
        obs_block: String::new(),
        flow_indices: vec![],
        target_length: 50,
    };
    let sim_levels = vec![
        model.content_hash(),
        config.content_hash(),
        params.content_hash(),
        scenario.content_hash(),
        seed.content_hash(),
    ];
    let fit_levels = vec![fit.content_hash(), method.content_hash(), seed.content_hash()];
    vec![
        ("Backend::ChainBinomial", Backend::ChainBinomial.content_hash()),
        ("CalendarMode::Numeric", CalendarMode::Numeric.content_hash()),
        ("Seed", seed.content_hash()),
        ("ModelDigest", model.content_hash()),
        ("SimConfig", config.content_hash()),
        ("ResolvedParams", params.content_hash()),
        ("ResolvedScenario", scenario.content_hash()),
        ("FitDigest", fit.content_hash()),
        ("MethodConfig", method.content_hash()),
        ("run_id(Sim)", crate::run_id(ArtifactKind::Sim, &sim_levels)),
        ("run_id(FitMethod)", crate::run_id(ArtifactKind::FitMethod, &fit_levels)),
    ]
}

#[test]
fn canonical_encoding_is_pinned() {
    // Pinned literals. A change here is a deliberate, reviewed re-key: bump
    // HASH_VERSION (whole store) or the relevant type's schema_version, then
    // re-pin. It must never move silently.
    //
    // HASH_VERSION 1 -> 2 (gh#1018): every pin below moved once, because the
    // version is folded into every root hash. No type's encoding changed.
    //
    // HASH_VERSION 2 -> 3 (gh#1046): every pin moved once more, for the same
    // reason. With the version held at 2 and the gh#1046 change applied, every
    // pin here was unchanged (checked when the bump was made): the change is
    // to how a model is hashed, and no fixture here holds a model.
    //
    // HASH_VERSION 3 -> 4 (D43): `MethodConfig` lost its resolved
    // `obs_alignment` field. With the version held at 3 and the field removed,
    // only `MethodConfig` and `run_id(FitMethod)` moved (checked when the bump
    // was made); with the bump, every pin moved once more.
    //
    // HASH_VERSION 4 -> 5 (temporal-architecture Stage 2, D58): every pin
    // moved once more for the version. `SimConfig` also gained
    // `allow_forcing_extrapolation` (false here), so its bytes and
    // `run_id(Sim)` change beyond the version, and `FitDigest` gained
    // `partition_rule_version` (1 here), so its bytes and `run_id(FitMethod)`
    // do too.
    //
    // HASH_VERSION 5 -> 6 (D72): every pin moved once more, for the version
    // alone. No type's encoding changed.
    //
    // HASH_VERSION 6 -> 7 (D77): the version alone again.
    //
    // HASH_VERSION 7 -> 8 (proposal 2026-11-17 §1b, §1c): the version alone.
    // No type's encoding changed.
    let expected: &[(&str, &str)] = &[
        ("Backend::ChainBinomial", "cb8273887c7fa7c86f9d068061486e1d936ed19699404f90fe23c823a77f8849"),
        ("CalendarMode::Numeric", "a3d65259d18f488945468a1e0f2064a3b6fc489f58a280917cfa050d25637963"),
        ("Seed", "ed089f152398835769e079739cc894257cfe888f3bebda829cf9359364ec1798"),
        ("ModelDigest", "7da049a55db9fb6013c76e50df42060695b3c0afd4afab0d234df26d4212f318"),
        // gh#833: `SimConfig` schema_version 4 -> 5 — the sim leaf's
        // `traj.tsv` gained `t_start`/`t_stop`, so the bytes under an
        // unchanged key changed; the key follows. `run_id(Sim)` moves with it.
        // Temporal-architecture Stage 4: `SimConfig` schema_version 5 -> 6 —
        // the sim leaf's `run.json` gained `calendar`, by the same rule.
        ("SimConfig", "1fe2d8054860b76121515916953a3d4f3b178cc1644b39c0735717bd17d871b7"),
        ("ResolvedParams", "d3304a5d14875263848cc3de40f7693b98e64f0d532c9668ce0154c8a0f61e0f"),
        // gh#573: `ResolvedScenario` schema_version 1 -> 2 — added `scale`
        // (the fixture now carries one factor, so the encoding of a non-empty
        // scale is pinned too). `run_id(Sim)` moves with it.
        ("ResolvedScenario", "70f54a6326061e3bf051a3e194f77fc838355dad6d417687d026a46220e82c34"),
        ("FitDigest", "27d26e9de8ee08b0acc6aa1312325a1d61e486234de9af04a17a1eb1d9d49b50"),
        // The `[stages]` -> `[method]` rename: the type is `MethodConfig`, not
        // `StageConfig`, and the derive hashes the fully-qualified type name as
        // its domain tag. Same fields, same bytes, new tag; `run_id(FitMethod)`
        // moves with it. Nothing else here moves.
        ("MethodConfig", "c38bc16d49909053ea81929aacbe727ad6c7606d54eb279bb36a97688d6a219d"),
        ("run_id(Sim)", "b0e9103c574bb0a78d4c812dd1c5108eec595dd407620c13162e7b303e131ce6"),
        ("run_id(FitMethod)", "a80b43d5e8604cfb6402a862aff713b70c9ba973b452ba8fec8469f66351554f"),
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
    b.kind = ArtifactKind::FitMethod;
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
        engine: EngineIdentity::of_build("0.3.0"),
    };
    let base = FitDigest {
        model: model.clone(),
        data: vec![DataDigest(ContentHash::from_bytes([1; 32]))],
        holdout_data: vec![DataDigest(ContentHash::from_bytes([2; 32]))],
        tables: vec![],
        fit_toml: ContentHash::from_bytes([3; 32]),
        engine: EngineIdentity::of_build("0.3.0"),
        partition_rule_version: 1,
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

    // A bound external file's content is fit identity too, and so is its
    // absence: binding one where there was none, or changing its bytes,
    // re-keys.
    let mut one_table = base.clone();
    one_table.tables = vec![DataDigest(ContentHash::from_bytes([4; 32]))];
    let mut other_table = base.clone();
    other_table.tables = vec![DataDigest(ContentHash::from_bytes([6; 32]))];
    assert_ne!(base.content_hash(), one_table.content_hash(), "a bound table must enter the fit identity");
    assert_ne!(one_table.content_hash(), other_table.content_hash(), "a table's bytes must key the fit");
}

#[test]
fn sim_config_semantic_flags_rekey() {
    let base = SimConfig {
        backend: Backend::ChainBinomial,
        dt: fid(1),
        t_start: fid(0),
        t_end: fid(100),
        output: ResolvedOutputSchedule::Regular { start: fid(0), step: fid(1) },
        calendar: CalendarMode::Numeric,
        allow_degenerate_rates: false,
        allow_forcing_extrapolation: false,
        no_flows: false,
        columns: BTreeSet::new(),
        init_state: None,
    };

    // The degenerate-rates flag changes collapse handling, so it re-keys.
    let mut flipped = base.clone();
    flipped.allow_degenerate_rates = true;
    assert_ne!(base.content_hash(), flipped.content_hash());

    // So does the forcing-extrapolation override (D44): the run may hold a
    // forcing's end value, so it is a different run.
    let mut held = base.clone();
    held.allow_forcing_extrapolation = true;
    assert_ne!(base.content_hash(), held.content_hash());
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
        allow_forcing_extrapolation: false,
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
        allow_forcing_extrapolation: false,
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
/// Content rather than provenance (the fit's `run_id` + method + selection
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

/// The engine identity hashes the bytes the engine string hashed before the
/// switches had a field of their own (`EngineVersion(String)`, a derived
/// newtype): each hex below is that type's hash of the string beside it,
/// recorded at ee9ba86c0 before the change. A default run, and a run with
/// switches, keep their identities.
#[test]
fn the_engine_identity_hashes_the_pre_field_engine_string() {
    let switch = |name: &str, value: Option<&str>| EngineSwitch { name: name.into(), value: value.map(Into::into) };
    let cases = [
        (
            EngineIdentity::of_build("0123456789abcdef0123456789abcdef"),
            "0123456789abcdef0123456789abcdef",
            "f6112779267e4043d6c401084be628e43551ceb5284bd2bd7eb5537557b7bef3",
        ),
        (
            EngineIdentity { build: "e".into(), switches: vec![switch("CAMDL_EVAL_UNRESOLVED", None)] },
            "e+21:CAMDL_EVAL_UNRESOLVED",
            "ab19e0561fd43b075c7c99b5bfa764d333d2b95f56aabd5df70dda67abe3d870",
        ),
        (
            EngineIdentity {
                build: "e".into(),
                switches: vec![switch("CAMDL_EVAL_UNRESOLVED", None), switch("CAMDL_NO_BINDING_CACHE", None)],
            },
            "e+21:CAMDL_EVAL_UNRESOLVED+22:CAMDL_NO_BINDING_CACHE",
            "f07175f8c28634d0bdc8d56e4ca961c728e67468c36a2b0e3c14fb3b71c1af71",
        ),
        (
            EngineIdentity { build: "e".into(), switches: vec![switch("CAMDL_EXPERIMENTAL_FOO", Some("2"))] },
            "e+22:CAMDL_EXPERIMENTAL_FOO=1:2",
            "e40728f645d6ba8ae0ee1738f82b3b59572e08de9a8146429f67a7897861876a",
        ),
    ];
    for (engine, spelling, hex) in cases {
        assert_eq!(engine.hashed_spelling(), spelling);
        assert_eq!(engine.content_hash().to_hex(), hex, "{spelling}");
    }
}

/// Two switch sets never give one engine identity, whatever the values hold.
#[test]
fn the_engine_identity_is_injective_in_the_switches() {
    let switch = |name: &str, value: Option<&str>| EngineSwitch { name: name.into(), value: value.map(Into::into) };
    let id = |switches: Vec<EngineSwitch>| EngineIdentity { build: "e".into(), switches }.content_hash();
    assert_ne!(
        id(vec![switch("CAMDL_EXPERIMENTAL_A", Some("1+22:CAMDL_EXPERIMENTAL_B=1:2"))]),
        id(vec![switch("CAMDL_EXPERIMENTAL_A", Some("1")), switch("CAMDL_EXPERIMENTAL_B", Some("2"))]),
    );
    assert_ne!(
        id(vec![switch("CAMDL_EXPERIMENTAL_A", Some("1=2"))]),
        id(vec![switch("CAMDL_EXPERIMENTAL_A=1", Some("2"))]),
    );
    assert_ne!(id(vec![]), id(vec![switch("CAMDL_EVAL_UNRESOLVED", None)]));
}
