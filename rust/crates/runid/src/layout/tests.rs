//! Layout tests: the exact 5-level sim path shape, and the two identity
//! properties the refactor exists to fix, proven *at the path level* — a
//! `t_end`/output change re-keys the config segment (gh#147), and a
//! lone-run vs sweep-point with the same base seed map to **distinct
//! paths** (the resolved-`process_seed` rule), not just distinct run_ids.

use std::path::Path;

use super::*;
use crate::float::FiniteF64;
use crate::hash::{ContentAddressed, ContentHash};
use crate::inputs::{Backend, CalendarMode, ResolvedOutputSchedule, Seed, SimConfig};
use crate::record::LevelId;

fn fid(x: f64) -> FiniteF64 {
    FiniteF64::new(x).unwrap()
}

fn level(name: &str, label: &str, h: ContentHash) -> LevelId {
    LevelId { name: name.into(), label: label.into(), hash: h, schema_version: 1 }
}

/// A resolved sim config that differs only in horizon (`t_end` + output end).
fn config(t_end: f64) -> SimConfig {
    SimConfig {
        backend: Backend::ChainBinomial,
        dt: fid(1.0),
        t_start: fid(0.0),
        t_end: fid(t_end),
        output: ResolvedOutputSchedule::Regular { start: fid(0.0), step: fid(1.0), end: fid(t_end) },
        calendar: CalendarMode::Numeric,
        allow_degenerate_rates: false,
    }
}

/// The five sim levels in path order, with the config + seed levels keyed on
/// their real digests (model/params/scenario hashes stubbed for the test).
fn sim_levels(
    model: ContentHash,
    cfg: &SimConfig,
    params: ContentHash,
    scenario_label: &str,
    scenario: ContentHash,
    seed: Seed,
) -> Vec<LevelId> {
    vec![
        level("model", "sir_basic", model),
        level("config", "chain_binomial-dt1", cfg.content_hash()),
        level("params", "base", params),
        level("scenario", scenario_label, scenario),
        level("seed", &format!("seed_{}", seed.base_seed), seed.content_hash()),
    ]
}

fn stub(byte: u8) -> ContentHash {
    ContentHash::from_bytes([byte; 32])
}

#[test]
fn sim_path_has_the_factored_five_level_shape() {
    let root = Path::new("/results");
    let seed = Seed { process_seed: 42, base_seed: 42 };
    let levels = sim_levels(stub(1), &config(100.0), stub(2), "baseline", stub(3), seed);

    let got = store_path(root, ArtifactKind::Sim, &levels);
    let expected = root
        .join("sims")
        .join(format!("sir_basic-{}", levels[0].hash.short8()))
        .join(format!("chain_binomial-dt1-{}", levels[1].hash.short8()))
        .join(format!("base-{}", levels[2].hash.short8()))
        .join(format!("baseline-{}", levels[3].hash.short8()))
        .join(format!("seed_42-{}", levels[4].hash.short8()));
    assert_eq!(got, expected);
}

#[test]
fn t_end_change_re_keys_the_config_segment() {
    // gh#147: the model_hash allowlist omitted output cadence / t_end, so two
    // models differing only there collided. Here the horizon lives in the
    // config level; changing it must change the config segment (and the path).
    let root = Path::new("/results");
    let seed = Seed { process_seed: 1, base_seed: 1 };
    let a = sim_levels(stub(1), &config(100.0), stub(2), "baseline", stub(3), seed);
    let b = sim_levels(stub(1), &config(200.0), stub(2), "baseline", stub(3), seed);

    assert_eq!(a[0].hash, b[0].hash, "model level unchanged");
    assert_ne!(a[1].hash, b[1].hash, "config level (t_end/output) must differ");
    assert_ne!(
        store_path(root, ArtifactKind::Sim, &a),
        store_path(root, ArtifactKind::Sim, &b),
        "a t_end change must produce a distinct store path (gh#147)"
    );
}

#[test]
fn lone_vs_sweep_point_same_base_seed_are_distinct_paths() {
    // The resolved-seed rule: the seed level hashes the resolved process_seed,
    // not the base --seed. A lone `--seed 42` (process_seed = 42) and the
    // `beta=2` point of a sweep with `--seed 42` (process_seed = 42 ^ M_DRAW)
    // share the readable label `seed_42` but MUST land in distinct paths — if
    // the level hashed the base seed they'd collide on disk (a silent wrong
    // answer the run.json gate cannot catch, since both compute the same hash).
    let root = Path::new("/results");
    let lone = Seed { process_seed: 42, base_seed: 42 };
    let sweep_point = Seed { process_seed: 42 ^ 0x9e37_79b9, base_seed: 42 };

    let a = sim_levels(stub(1), &config(100.0), stub(2), "baseline", stub(3), lone);
    let b = sim_levels(stub(1), &config(100.0), stub(2), "baseline", stub(3), sweep_point);

    let pa = store_path(root, ArtifactKind::Sim, &a);
    let pb = store_path(root, ArtifactKind::Sim, &b);

    // Both keep the readable `seed_42` label…
    assert!(pa.to_string_lossy().contains("/seed_42-"));
    assert!(pb.to_string_lossy().contains("/seed_42-"));
    // …yet the segments — and the paths — differ via the process_seed hash.
    assert_ne!(pa, pb, "lone vs sweep-point with the same base seed must be DISTINCT paths");
}

#[test]
fn empty_scenario_uses_baseline_label_with_a_real_hash() {
    // The empty-delta scenario shows the readable label `baseline`, but the
    // segment carries the real hash of the empty delta — never a literal zero.
    let root = Path::new("/results");
    let seed = Seed { process_seed: 7, base_seed: 7 };
    let levels = sim_levels(stub(1), &config(100.0), stub(2), "baseline", stub(0xab), seed);
    let p = store_path(root, ArtifactKind::Sim, &levels);
    assert!(p.to_string_lossy().contains(&format!("/baseline-{}/", stub(0xab).short8())));
}

// ── label rendering + kind dirs ──────────────────────────────────────────────

#[test]
fn path_label_preserves_compound_labels() {
    assert_eq!(path_label("chain_binomial-dt1"), "chain_binomial-dt1");
    assert_eq!(path_label("01-scout"), "01-scout");
    assert_eq!(path_label("seed_42"), "seed_42");
    assert_eq!(path_label("baseline"), "baseline");
    // Unsafe characters are mapped to `_`, and the result is lowercased.
    assert_eq!(path_label("With SIA!"), "with_sia_");
    assert_eq!(path_label("R0=3.0"), "r0_3.0");
}

#[test]
fn kind_store_dirs() {
    assert_eq!(ArtifactKind::Sim.store_dir(), "sims");
    assert_eq!(ArtifactKind::FitStage.store_dir(), "fits");
    assert_eq!(ArtifactKind::Pfilter.store_dir(), "pfilters");
    assert_eq!(ArtifactKind::Survey.store_dir(), "surveys");
    assert_eq!(ArtifactKind::ProfilePoint.store_dir(), "profiles");
}

#[test]
fn segment_is_label_dash_hash8_with_no_disambiguator() {
    let lvl = level("config", "chain_binomial-dt1", stub(0xcd));
    let seg = segment(&lvl);
    assert_eq!(seg, format!("chain_binomial-dt1-{}", stub(0xcd).short8()));
    // The base form never carries a `~` — that is the store's collision
    // disambiguator, appended at commit time, not by Layout.
    assert!(!seg.contains('~'));
}
