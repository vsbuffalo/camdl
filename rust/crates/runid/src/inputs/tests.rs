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
        output: ResolvedOutputSchedule::Regular { start: fid(0), step: fid(1), end: fid(100) },
        calendar: CalendarMode::Numeric,
        allow_degenerate_rates: false,
        no_flows: false,
        columns: BTreeSet::new(),
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
        output: ResolvedOutputSchedule::Regular { start: fid(0), step: fid(1), end: fid(100) },
        calendar: CalendarMode::Numeric,
        allow_degenerate_rates: false,
        no_flows: false,
        columns: BTreeSet::new(),
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
