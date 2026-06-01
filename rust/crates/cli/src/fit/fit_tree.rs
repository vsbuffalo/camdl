//! Canonical fit-directory walker.
//!
//! One module, two functions, used by every consumer that needs to
//! enumerate fit-stage runs:
//!
//! - [`walk_fit_dir`] — given one fit_dir (`results/fits/<stem>-<hash>/`)
//!   returns one [`StageNode`] per completed `RunKind::FitStage` run
//!   found anywhere underneath. Independent of layout convention beyond
//!   "every stage writes a `run.json`."
//! - [`walk_fits_root`] — given the top-level `results/fits/` returns
//!   one [`FitDirEntry`] per fit_dir, each carrying its already-parsed
//!   top-level [`FitMeta`] for filter use without a second `run.json`
//!   read per row.
//!
//! Replaces three current walkers (the buggy v1 walker in
//! `fit_summary.rs::format_text`, `grid_summary::iter_cells`, and
//! `browse::resolve_stage_by_hash`). See
//! `docs/dev/proposals/2026-04-28-fit-experiment-management.md` §1.
//!
//! `StageNode` is method-agnostic by construction: it does not carry a
//! `fit_state_path` field (an IF2-only artifact). Consumers that need
//! the typed result load it via `MethodResult::load_from`.

use std::io;
use std::path::{Path, PathBuf};

use crate::run_meta::{FitMeta, Run, RunKind};

/// One discovered fit-stage run inside a fit_dir.
///
/// `axes` carries the (data_kind, fit_seed, sweep_slug) triple
/// extracted from the path relative to the enclosing fit_dir. When the
/// path doesn't fit the canonical v2 layout, `axes` is `None` — the
/// stage is still surfaced (its `run.json` is real) but consumers that
/// need axis grouping skip it.
#[derive(Debug, Clone)]
pub struct StageNode {
    /// Absolute (or whatever the caller passed) path to the stage
    /// directory containing `run.json`.
    pub stage_dir: PathBuf,
    /// The stage's parsed `run.json` (kind = `FitStage`).
    pub run: Run,
    /// Path-derived axes — `None` when the stage is found outside the
    /// canonical `<fit_dir>/{real,synthetic}/...` layout.
    pub axes: Option<StageAxes>,
}

/// (data_kind, fit_seed, sweep_slug) triple extracted from a stage's
/// path relative to its fit_dir.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StageAxes {
    pub data_kind: DataKind,
    pub fit_seed: u64,
    /// `None` when no `--sweep` was active for this fit. When the sweep
    /// produced a single point with no slug, also `None`.
    pub sweep_slug: Option<String>,
}

/// Whether this stage fit a real-data dataset or one of N synthetic
/// replicates.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DataKind {
    Real,
    Synthetic { ds_idx: usize },
}

/// One fit_dir entry returned by [`walk_fits_root`]. The `fit_meta`
/// field is pre-parsed so `fit table`'s outer loop can filter by
/// model_hash / label / declared stages without a second `run.json`
/// read per row.
#[derive(Debug, Clone)]
pub struct FitDirEntry {
    pub fit_dir: PathBuf,
    /// Top-level `run.json` — `kind` is `RunKind::Fit`.
    pub run: Run,
    /// Convenience: the kind payload. Always equals `run.kind`'s
    /// `Fit(_)` variant; carrying it as a separate field lets callers
    /// skip the matched destructure.
    pub fit_meta: FitMeta,
}

/// Walk one fit directory and return every `RunKind::FitStage` run
/// found underneath. Filesystem layout is intentionally not
/// constrained beyond "every stage writes a `run.json`": this walker
/// works on real-data fits, synthetic-replicate fits, sweep cells,
/// any combination of those, and on user-defined non-canonical
/// layouts.
///
/// Returns runs in lexicographic order on `stage_dir` (deterministic
/// for tests). Malformed `run.json` files are skipped with a stderr
/// log (don't panic — a stale write or unrelated `run.json` should
/// not crash a list command).
pub fn walk_fit_dir(fit_dir: &Path) -> io::Result<Vec<StageNode>> {
    let mut out = Vec::new();
    if !fit_dir.exists() {
        return Err(io::Error::new(
            io::ErrorKind::NotFound,
            format!("fit_dir not found: {}", fit_dir.display()),
        ));
    }
    visit_dir(fit_dir, fit_dir, &mut out);
    out.sort_by(|a, b| a.stage_dir.cmp(&b.stage_dir));
    Ok(out)
}

/// Walk `results/fits/` (or whatever root the caller passes) and
/// return one entry per fit_dir. Top-level `run.json` files that
/// aren't `RunKind::Fit` are skipped (e.g. someone manually parking
/// a sim run under `fits/` won't accidentally surface as a fit).
///
/// Returns entries in lexicographic order on `fit_dir`.
pub fn walk_fits_root(root: &Path) -> io::Result<Vec<FitDirEntry>> {
    let mut out = Vec::new();
    if !root.exists() {
        // Empty root → empty list. `walk_fit_dir` errors on a missing
        // dir because that's a single-fit lookup; this is a directory
        // listing, and "no fits yet" is a normal state.
        return Ok(out);
    }
    // gh#147 (M3.2): a CAS fit is a segment dir `fits/{stem}-{h8}/` holding
    // its stage-leaf subdirs (`{NN-stage}-{h8}/seed_N-{h8}/run.json`, kind
    // FitStage) plus the fit-level sidecar. `read_fit_segment` derives ONE
    // fit-level `RunKind::Fit` entry per segment from those leaves — the same
    // adapter `table_row::build_row` (via `Run::read`) consumes, so the table
    // row and its source entry agree. A dir with no FitStage leaves is not a
    // fit segment and is skipped silently.
    let entries = std::fs::read_dir(root)?;
    for e in entries.flatten() {
        let p = e.path();
        if !p.is_dir() {
            continue;
        }
        if let Some(run) = crate::run_meta::read_fit_segment(&p) {
            if let RunKind::Fit(meta) = &run.kind {
                out.push(FitDirEntry {
                    fit_dir: p.clone(),
                    run: run.clone(),
                    fit_meta: meta.clone(),
                });
            }
        }
    }
    out.sort_by(|a, b| a.fit_dir.cmp(&b.fit_dir));
    Ok(out)
}

// ── internals ───────────────────────────────────────────────────────

/// Depth-first recursion. At every directory, try to read `run.json`;
/// if present and a FitStage, push a node. Always recurse into
/// children — a fit_dir contains a top-level `Fit` run.json AND
/// per-stage `FitStage` run.json files, so we can't stop at the first
/// run.json we find.
fn visit_dir(fit_dir: &Path, here: &Path, out: &mut Vec<StageNode>) {
    let run_json = here.join("run.json");
    if run_json.is_file() {
        // `Run::read` transparently synthesizes a legacy `Run` from a CAS
        // fit-stage `RunRecord` (gh#147 M3.2), so this one path discovers
        // both content-addressed and any remaining legacy fit stages.
        match Run::read(here) {
            Ok(run) => {
                if matches!(run.kind, RunKind::FitStage(_)) {
                    let axes = derive_axes(fit_dir, here);
                    out.push(StageNode {
                        stage_dir: here.to_path_buf(),
                        run,
                        axes,
                    });
                }
            }
            Err(err) => {
                eprintln!(
                    "warning: walk_fit_dir: skipping {} (cannot read run.json: {})",
                    here.display(),
                    err
                );
            }
        }
    }
    let entries = match std::fs::read_dir(here) {
        Ok(e) => e,
        Err(_) => return,
    };
    for e in entries.flatten() {
        let p = e.path();
        if p.is_dir() {
            visit_dir(fit_dir, &p, out);
        }
    }
}

/// Extract `StageAxes` from a stage path relative to its fit_dir.
/// Canonical v2 layouts:
///
/// ```text
/// <fit_dir>/real/fit_<seed>/<stage>/
/// <fit_dir>/real/fit_<seed>/<sweep_slug>/<stage>/
/// <fit_dir>/synthetic/ds_<NN>/fit_<seed>/<stage>/
/// <fit_dir>/synthetic/ds_<NN>/fit_<seed>/<sweep_slug>/<stage>/
/// ```
///
/// Returns `None` when the path doesn't fit either shape (e.g. a
/// non-canonical user layout or a profile sub-tree). The walker still
/// surfaces those stages — `axes` is None on the node and consumers
/// that need axis grouping skip them.
fn derive_axes(fit_dir: &Path, stage_dir: &Path) -> Option<StageAxes> {
    let rel = stage_dir.strip_prefix(fit_dir).ok()?;
    let parts: Vec<&str> = rel
        .components()
        .filter_map(|c| c.as_os_str().to_str())
        .collect();

    // The last component is the stage name; we don't need it for axes
    // (it's already on `Run.kind.stage`). What precedes it must match
    // one of the four shapes above.
    if parts.len() < 3 {
        return None;
    }
    let body = &parts[..parts.len() - 1];

    // Real layouts: ["real", "fit_<seed>"] or ["real", "fit_<seed>", "<slug>"]
    if body.first() == Some(&"real") {
        let fit_seed = parse_fit_seed(body.get(1)?)?;
        let sweep_slug = match body.len() {
            2 => None,
            3 => Some(body[2].to_string()),
            _ => return None,
        };
        return Some(StageAxes {
            data_kind: DataKind::Real,
            fit_seed,
            sweep_slug,
        });
    }

    // Synthetic layouts:
    //   ["synthetic", "ds_<NN>", "fit_<seed>"]
    //   ["synthetic", "ds_<NN>", "fit_<seed>", "<slug>"]
    if body.first() == Some(&"synthetic") {
        let ds_idx = parse_ds_idx(body.get(1)?)?;
        let fit_seed = parse_fit_seed(body.get(2)?)?;
        let sweep_slug = match body.len() {
            3 => None,
            4 => Some(body[3].to_string()),
            _ => return None,
        };
        return Some(StageAxes {
            data_kind: DataKind::Synthetic { ds_idx },
            fit_seed,
            sweep_slug,
        });
    }

    None
}

fn parse_fit_seed(s: &str) -> Option<u64> {
    s.strip_prefix("fit_")?.parse().ok()
}

fn parse_ds_idx(s: &str) -> Option<usize> {
    s.strip_prefix("ds_")?.parse().ok()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::run_meta::{FitMeta, FitStageMeta, RunKind, RunStatus, SimulateMeta};
    use std::collections::HashMap;

    /// Allocate a unique tempdir for one test. Cleaned up by `Drop`.
    struct TempDir(PathBuf);
    impl TempDir {
        fn path(&self) -> &Path {
            &self.0
        }
    }
    impl Drop for TempDir {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }
    fn tempdir(tag: &str) -> TempDir {
        let ns = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let p = std::env::temp_dir().join(format!(
            "camdl_fittree_{}_{}_{}",
            tag,
            std::process::id(),
            ns
        ));
        std::fs::create_dir_all(&p).unwrap();
        TempDir(p)
    }

    fn fit_run(parent_hash: &str) -> Run {
        Run {
            hash: parent_hash.into(),
            version: "0.1.0+test".into(),
            created_at: "2026-04-27T00:00:00Z".into(),
            argv: vec!["camdl".into(), "fit".into(), "run".into()],
            status: RunStatus::Completed { wall_time_seconds: 1.0 },
            label: None,
            kind: RunKind::Fit(FitMeta {
                model: "sir.camdl".into(),
                model_hash: "f00d".repeat(16),
                fit_toml_path: "fit.toml".into(),
                fit_toml_hash: "cafe".repeat(16),
                data_hashes: HashMap::new(),
                estimated: vec!["beta".into()],
                fixed: HashMap::new(),
                stages_declared: vec!["mle".into()],
                ic_free: false,
                resolved_priors: Vec::new(),
                parameters_provenance: Default::default(),
                        }),
        }
    }

    /// gh#147 (M3.2): write a CAS fit segment at `seg` — one FitStage leaf
    /// (`01-mle-<h8>/seed_1-<h8>/run.json`) plus the fit-level sidecar — the
    /// shape `walk_fits_root` reads. `fit_h8` (8 hex) seeds the shared
    /// `fit`-level hash so each segment gets a distinct `run.hash`.
    fn write_cas_fit_seg(seg: &std::path::Path, fit_h8: &str) {
        let leaf = seg.join("01-mle-1fb03eee").join("seed_1-06cbd6b3");
        std::fs::create_dir_all(&leaf).unwrap();
        let fit_hash = format!("{fit_h8}{}", "0".repeat(64 - fit_h8.len()));
        let run_id = format!("{:0<64}", format!("{fit_h8}01"));
        let rec = format!(
            r#"{{"format_version":1,"kind":"fit_stage","run_id":"{run_id}","hash_version":1,"ir_version":"0.7","engine_version":"0.1.0+test","levels":[{{"name":"fit","label":"fit","hash":"{fit_hash}","schema_version":1}},{{"name":"stage","label":"01-mle","hash":"1fb03eee00000000000000000000000000000000000000000000000000000000","schema_version":1}},{{"name":"seed","label":"seed_1","hash":"06cbd6b300000000000000000000000000000000000000000000000000000000","schema_version":1}}],"status":"completed","artifacts":{{}},"inputs":{{"stage":"mle","method":"if2","backend":"chain_binomial","seed":1,"n_chains":2}},"provenance":{{"created_at":"2026-04-27T00:00:00Z","argv":["camdl","fit","run"]}}}}"#
        );
        std::fs::write(leaf.join("run.json"), rec).unwrap();
        std::fs::write(
            seg.join("fit.meta.json"),
            r#"{"model_hash":"f00d","model_path":"sir.camdl","fit_toml_path":"fit.toml"}"#,
        )
        .unwrap();
    }

    fn stage_run(parent_hash: &str, stage: &str, method: crate::run_meta::MethodKind, seed: u64) -> Run {
        Run {
            hash: format!("{}-{}-{}", parent_hash, stage, seed)
                .chars()
                .cycle()
                .take(64)
                .collect(),
            version: "0.1.0+test".into(),
            created_at: "2026-04-27T00:00:00Z".into(),
            argv: vec!["camdl".into(), "fit".into(), "run".into()],
            status: RunStatus::Completed { wall_time_seconds: 1.0 },
            label: None,
            kind: RunKind::FitStage(FitStageMeta {
                fit_hash: parent_hash.into(),
                stage: stage.into(),
                method,
                backend: crate::run_meta::Backend::ChainBinomial,
                seed,
                n_chains: 2,
                algorithm: serde_json::Value::Null,
                best_loglik: Some(-100.0),
                best_chain: Some(0),
                starts_from: None,
                derived_from: None,
                parent_profile_hash: None,
                profile_point_idx: None,
                profile_start_idx: None,
                parameters_provenance: Default::default(),
                init_provenance: None,
                        }),
        }
    }

    /// Place a stage `run.json` and an empty marker file at
    /// `<fit_dir>/<rel>/run.json`.
    fn place_stage(fit_dir: &Path, rel: &str, run: &Run) {
        let dir = fit_dir.join(rel);
        std::fs::create_dir_all(&dir).unwrap();
        run.write(&dir).unwrap();
    }

    #[test]
    fn walks_real_only_single_fit() {
        let tmp = tempdir("real_only");
        let fit_dir = tmp.path().join("fit_he-deadbeef");
        std::fs::create_dir_all(&fit_dir).unwrap();

        let parent = "deadbeef".repeat(8);
        fit_run(&parent).write(&fit_dir).unwrap();
        place_stage(
            &fit_dir,
            "real/fit_1/scout",
            &stage_run(&parent, "scout", crate::run_meta::MethodKind::If2, 1),
        );
        place_stage(
            &fit_dir,
            "real/fit_1/refine",
            &stage_run(&parent, "refine", crate::run_meta::MethodKind::If2, 1),
        );

        let nodes = walk_fit_dir(&fit_dir).unwrap();
        assert_eq!(nodes.len(), 2);
        // Both nodes carry the canonical real/fit_1 axes, no sweep.
        for node in &nodes {
            let axes = node.axes.as_ref().expect("real fit must produce axes");
            assert_eq!(axes.data_kind, DataKind::Real);
            assert_eq!(axes.fit_seed, 1);
            assert_eq!(axes.sweep_slug, None);
        }
        let stage_names: Vec<&str> = nodes
            .iter()
            .map(|n| match &n.run.kind {
                RunKind::FitStage(m) => m.stage.as_str(),
                _ => unreachable!(),
            })
            .collect();
        assert!(stage_names.contains(&"scout"));
        assert!(stage_names.contains(&"refine"));
    }

    #[test]
    fn walks_synthetic_layout() {
        let tmp = tempdir("synthetic");
        let fit_dir = tmp.path().join("fit_syn-cafebabe");
        std::fs::create_dir_all(&fit_dir).unwrap();
        let parent = "cafebabe".repeat(8);
        fit_run(&parent).write(&fit_dir).unwrap();
        for ds in 1..=2usize {
            for fs in [11u64, 22] {
                let rel = format!("synthetic/ds_{:02}/fit_{}/mle", ds, fs);
                place_stage(&fit_dir, &rel, &stage_run(&parent, "mle", crate::run_meta::MethodKind::If2, fs));
            }
        }
        let nodes = walk_fit_dir(&fit_dir).unwrap();
        assert_eq!(nodes.len(), 4, "2 datasets × 2 seeds = 4 nodes");
        // Spot-check one cell's axes.
        let ds01_22 = nodes
            .iter()
            .find(|n| {
                let axes = n.axes.as_ref().unwrap();
                matches!(axes.data_kind, DataKind::Synthetic { ds_idx: 1 })
                    && axes.fit_seed == 22
            })
            .expect("ds_01 × fit_22 cell");
        assert_eq!(ds01_22.axes.as_ref().unwrap().sweep_slug, None);
    }

    #[test]
    fn walks_sweep_slug_layout() {
        let tmp = tempdir("sweep");
        let fit_dir = tmp.path().join("fit_sweep-12345678");
        std::fs::create_dir_all(&fit_dir).unwrap();
        let parent = "12345678".repeat(8);
        fit_run(&parent).write(&fit_dir).unwrap();
        place_stage(
            &fit_dir,
            "real/fit_1/R0_1.000/mle",
            &stage_run(&parent, "mle", crate::run_meta::MethodKind::If2, 1),
        );
        place_stage(
            &fit_dir,
            "real/fit_1/R0_2.000/mle",
            &stage_run(&parent, "mle", crate::run_meta::MethodKind::If2, 1),
        );
        let nodes = walk_fit_dir(&fit_dir).unwrap();
        assert_eq!(nodes.len(), 2);
        let slugs: Vec<String> = nodes
            .iter()
            .map(|n| n.axes.as_ref().unwrap().sweep_slug.clone().unwrap())
            .collect();
        assert!(slugs.iter().any(|s| s == "R0_1.000"));
        assert!(slugs.iter().any(|s| s == "R0_2.000"));
    }

    #[test]
    fn walks_mixed_method_stages() {
        let tmp = tempdir("mixed");
        let fit_dir = tmp.path().join("fit_mixed-aaaabbbb");
        std::fs::create_dir_all(&fit_dir).unwrap();
        let parent = "aaaabbbb".repeat(8);
        fit_run(&parent).write(&fit_dir).unwrap();
        place_stage(
            &fit_dir,
            "real/fit_1/scout",
            &stage_run(&parent, "scout", crate::run_meta::MethodKind::If2, 1),
        );
        place_stage(
            &fit_dir,
            "real/fit_1/pgas",
            &stage_run(&parent, "pgas", crate::run_meta::MethodKind::Pgas, 1),
        );
        let nodes = walk_fit_dir(&fit_dir).unwrap();
        assert_eq!(nodes.len(), 2);
        let methods: Vec<&str> = nodes
            .iter()
            .map(|n| match &n.run.kind {
                RunKind::FitStage(m) => m.method.as_str(),
                _ => unreachable!(),
            })
            .collect();
        assert!(methods.contains(&"if2"));
        assert!(methods.contains(&"pgas"));
    }

    #[test]
    fn skips_malformed_run_json_does_not_panic() {
        let tmp = tempdir("malformed");
        let fit_dir = tmp.path().join("fit_bad-deadbeef");
        let stage_dir = fit_dir.join("real/fit_1/scout");
        std::fs::create_dir_all(&stage_dir).unwrap();
        // Top-level fit run.json is OK.
        fit_run(&"deadbeef".repeat(8)).write(&fit_dir).unwrap();
        // Stage run.json is garbage.
        std::fs::write(stage_dir.join("run.json"), "{ not valid json }").unwrap();

        let nodes = walk_fit_dir(&fit_dir).unwrap();
        assert!(
            nodes.is_empty(),
            "malformed stage run.json should be skipped, got {:?}",
            nodes
        );
    }

    #[test]
    fn missing_fit_dir_errors() {
        let err = walk_fit_dir(Path::new("/definitely/does/not/exist/here")).unwrap_err();
        assert_eq!(err.kind(), io::ErrorKind::NotFound);
    }

    #[test]
    fn surfaces_non_canonical_layout_with_axes_none() {
        // A user might park a fit-stage run.json under a one-off path
        // (e.g. for a debugging copy). The walker should still surface
        // it — `axes` is None signals "not in canonical layout."
        let tmp = tempdir("noncanon");
        let fit_dir = tmp.path().join("fit_x-aabbccdd");
        std::fs::create_dir_all(&fit_dir).unwrap();
        let parent = "aabbccdd".repeat(8);
        fit_run(&parent).write(&fit_dir).unwrap();
        place_stage(
            &fit_dir,
            "weird_layout/scout",
            &stage_run(&parent, "scout", crate::run_meta::MethodKind::If2, 1),
        );
        let nodes = walk_fit_dir(&fit_dir).unwrap();
        assert_eq!(nodes.len(), 1);
        assert!(nodes[0].axes.is_none(),
            "non-canonical layout must produce axes=None");
    }

    #[test]
    fn walks_fits_root_returns_one_entry_per_fit() {
        let tmp = tempdir("root");
        let fits_root = tmp.path().join("fits");
        std::fs::create_dir_all(&fits_root).unwrap();

        let dirs = ["fit_a-11111111", "fit_b-22222222"];
        for name in dirs.iter() {
            let d = fits_root.join(name);
            // gh#147 (M3.2): a CAS fit segment (one FitStage leaf + the
            // fit-level sidecar) — the shape `walk_fits_root`/`read_fit_segment`
            // reads now, not a fit-wide `RunKind::Fit` record.
            let h8 = name.split('-').next_back().unwrap();
            write_cas_fit_seg(&d, h8);
        }
        // A non-fit run.json (Simulate kind) under fits/ — should be
        // skipped.
        let intruder = fits_root.join("not_a_fit");
        std::fs::create_dir_all(&intruder).unwrap();
        let sim_run = Run {
            hash: "deadbeef".repeat(8),
            version: "0.1.0+test".into(),
            created_at: "t".into(),
            argv: vec![],
            status: RunStatus::Running,
            label: None,
            kind: RunKind::Simulate(SimulateMeta {
                model: "x".into(),
                model_hash: "h".repeat(64),
                scenario: "baseline".into(),
                sim_hash: "s".repeat(64),
                scen_hash: "c".repeat(64),
                seed: 1,
                backend: crate::args::types::Backend::Gillespie,
                dt: 1.0,
                sweep_point: HashMap::new(),
                from_fit_hash: None,
                parameters_provenance: Default::default(),
                        }),
        };
        sim_run.write(&intruder).unwrap();

        let entries = walk_fits_root(&fits_root).unwrap();
        let names: Vec<&str> = entries
            .iter()
            .map(|e| e.fit_dir.file_name().unwrap().to_str().unwrap())
            .collect();
        assert_eq!(names, ["fit_a-11111111", "fit_b-22222222"]);
        // FitMeta is pre-parsed (derived from the stage leaves) and accessible
        // without a re-read; `stages_declared` comes from the leaves' stage
        // levels. gh#147 (M3.2): `estimated`/`fixed`/priors are config detail
        // not carried on the leaves, so they default empty — config-diff reads
        // the archived `fit.toml.original` for those.
        assert_eq!(entries[0].fit_meta.stages_declared, vec!["mle"]);
        // The full Run is also exposed (callers want created_at /
        // wall_time without a third re-read).
        assert!(matches!(entries[0].run.kind, RunKind::Fit(_)));
    }

    #[test]
    fn walks_fits_root_empty_when_no_root() {
        let entries =
            walk_fits_root(Path::new("/definitely/not/here/either/fits")).unwrap();
        assert!(entries.is_empty());
    }
}
