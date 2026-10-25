//! gh#147 (M-ensemble). The `SimEnsemble` CAS identity: map a multi-cell
//! `simulate` (`--replicates`/`--seeds`/multi-scenario/`--draws`) into the
//! `runid` factored levels (`model` / `config` / `params` / `grid`) and its
//! leaf `run_id`.
//!
//! A multi-cell `simulate` writes one per-cell [`Sim`](runid::ArtifactKind::Sim)
//! leaf AND a combined wide-format trajectory
//! TSV that interleaves every cell with `replicate`/`scenario`/`draw` columns.
//! That combined TSV is the *ensemble* artifact: a derived view over the N
//! leaves. Its identity must be a pure function of everything that determines
//! the combined bytes:
//!   - **model** — the pure model IR digest (constant across cells).
//!   - **config** — backend + dt (shared by every cell).
//!   - **params** — the set of the cells' own `params` level hashes (sorted,
//!     deduplicated): which resolved parameter vectors the ensemble ran. Read
//!     off the cells' identities rather than rebuilt beside them, so it cannot
//!     describe different values from the ones the cells hashed (gh#583).
//!   - **grid** — a digest over the SORTED cell list plus the run's resolved
//!     design. Each cell contributes `(point_id, process_seed, sim_run_id)`;
//!     the `sim_run_id` already encodes that cell's model/config/params/
//!     scenario/seed, so a changed `--draws` param value (same draw index)
//!     re-keys the ensemble, and the arm's label — a display string, not
//!     identity — is deliberately absent: two ensembles of identical cells
//!     hash alike whether an arm was named by a preset or derived from flags,
//!     and renaming a preset re-keys nothing at this level (it does re-key the
//!     ensemble through every cell's `sim_run_id`, since the name is in the
//!     model digest). The cell COUNT is folded in
//!     explicitly (`n_cells`): 3 replicates vs 4 is a different combined TSV,
//!     so a different ensemble (count-in-the-key, the n_trajectories collision
//!     class). Sorting makes the digest order-independent. The `design` is the
//!     resolved `--sweep` / `--sweep-file` grid of a sweep job ([`ResolvedDesign`])
//!     and `None` otherwise; it is what makes `--sweep beta=0.2,0.3` and
//!     `--sweep beta=0.3,0.2` — the same cell set, two different combined
//!     files — two ensembles.
//!
//! Mirrors [`crate::survey_cas`] / [`crate::pfilter_cas`] for the level/digest
//! conventions.

use runid::inputs::{EngineVersion, ModelDigest};
use runid::{run_id, ArtifactKind, ContentHash, LevelId};

use crate::fit::cas::canonical_config_hash;

/// One cell of a multi-cell `simulate`, contributing to the ensemble's `grid`
/// digest and to its `deps` (the cell's `Sim` leaf).
#[derive(Debug, Clone)]
pub struct EnsembleCell {
    /// The scenario label rendered into the combined TSV's `scenario` column.
    /// Display only: it enters the ensemble's `run.json` summary, never its
    /// `grid` digest.
    pub scenario_label: String,
    /// The resolved process seed driving this cell's trajectory.
    pub process_seed: u64,
    /// The id the cell is written under on the parameter-point axis: a sweep
    /// cell's `point_id` (a sweep file's own column, else the 0-based design
    /// index), the draw index of a `--draws` cell, 0 for a single point. The
    /// written id, not the row position, is what the `grid` digest sorts
    /// over: it is the `point_id` the ensemble's own rows carry, so the
    /// identity names what the stored bytes say. Two files with the same rows
    /// and different ids are already two designs through the file digest, so
    /// no distinctness hangs on the choice, and a file without the column
    /// hashes exactly as before (written id = row).
    pub point_id: usize,
    /// The cell's `Sim` leaf `run_id` — its full identity (model/config/params/
    /// scenario/seed). Folding it into `grid` makes a per-draw param change
    /// re-key the ensemble.
    pub sim_run_id: ContentHash,
    /// SHA-256 of the cell's `traj.tsv` — the `deps` edge's consumed-artifact
    /// digest (which upstream file the combined TSV was built from).
    pub traj_digest: ContentHash,
    /// The cell's `params` level hash (its resolved parameter values).
    pub params_level: ContentHash,
}

/// A fully-resolved ensemble leaf: the four factored levels (in path order) and
/// the leaf `run_id` composed from their hashes.
pub struct ResolvedEnsemble {
    pub levels: Vec<LevelId>,
    pub run_id: ContentHash,
}

/// Inputs to [`resolve_sim_ensemble`], all resolved by the caller.
pub struct EnsembleCtx<'a> {
    pub model: &'a ir::Model,
    pub ir_version: &'a str,
    pub engine_version: &'a str,
    /// Provenance label for the `model` path segment (the model stem).
    pub stem: &'a str,
    pub backend: crate::args::types::ForwardBackend,
    pub dt: f64,
    /// The full set of cells the run expands to.
    pub cells: &'a [EnsembleCell],
    /// The resolved design of a sweep job; `None` for every other job.
    pub design: Option<&'a ResolvedDesign>,
}

/// The resolved design axis of a `simulate` sweep job: what `--sweep` or
/// `--sweep-file` expanded to, recorded in the ensemble's `run.json` and
/// folded into its `grid` identity. Serialized as it is — every field enters
/// the digest.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(tag = "source", rename_all = "snake_case")]
pub enum ResolvedDesign {
    /// `--sweep PARAM=GRID`, repeatable. `params` is name-sorted; the design
    /// points are their Cartesian product with the first name varying slowest,
    /// so `point_id` is a position in that product.
    Sweep { params: Vec<SweptParam>, n_points: usize },
    /// `--sweep-file FILE`: a TSV whose header names parameters and whose rows
    /// are the design points. `point_ids` is the id each row was written
    /// under, in row order: the file's own `point_id` column when it has one,
    /// else `0..n_points`. The file enters by its content digest, never its
    /// path.
    File { params: Vec<String>, digest: String, n_points: usize, point_ids: Vec<usize> },
}

/// One swept parameter and the values it takes, in grid order.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct SweptParam {
    pub name: String,
    pub values: Vec<f64>,
}

use crate::fit::cas::{level, structural_level_hash};

/// Compact `dt` rendering for the `config` segment label (`1`, `0.5`, …).
fn fmt_dt(dt: f64) -> String {
    if (dt.round() - dt).abs() < 1e-9 {
        format!("{}", dt.round() as i64)
    } else {
        format!("{}", dt)
    }
}

/// Resolve an ensemble leaf's identity: the four factored levels and the
/// `run_id` derived from their hashes.
/// The `config` level: backend + step size the ensemble was produced under.
/// A struct rather than a `json!` literal so the level is include-by-default —
/// see `PfilterConfigLevel` for the full argument.
#[derive(serde::Serialize)]
struct EnsembleConfigLevel<'a> {
    backend: &'a str,
    dt: f64,
}

/// The `grid` level: the sorted cell list, the explicit count (so N vs N+1
/// replicates is a different ensemble) and the resolved design (`null` for a
/// job without one — serialized as it is, so a design-less job and a sweep job
/// over the same cells are two ensembles).
#[derive(serde::Serialize)]
struct EnsembleGridLevel<'a> {
    n_cells: usize,
    cells: &'a [(usize, u64, String)],
    design: Option<&'a ResolvedDesign>,
}

pub fn resolve_sim_ensemble(ctx: &EnsembleCtx) -> Result<ResolvedEnsemble, String> {
    // params level — the distinct `params` level hashes of the cells, sorted:
    // the parameter vectors the ensemble ran, as the cells themselves hashed
    // them. A sequence with no field set to forget, so it takes no wrapper
    // struct.
    let mut params_levels: Vec<String> =
        ctx.cells.iter().map(|c| c.params_level.to_hex()).collect();
    params_levels.sort();
    params_levels.dedup();

    // grid level — the sorted cell list + the explicit cell count + the
    // design. Each cell is (point_id, seed, sim_run_id); sorting is
    // order-independent. The count is folded so N vs N+1 replicates is a
    // different ensemble.
    let mut cells_sorted: Vec<(usize, u64, String)> = ctx
        .cells
        .iter()
        .map(|c| (c.point_id, c.process_seed, c.sim_run_id.to_hex()))
        .collect();
    cells_sorted.sort();
    let grid_level =
        EnsembleGridLevel { n_cells: ctx.cells.len(), cells: &cells_sorted, design: ctx.design };

    let model_digest = ModelDigest::from_model(
        ctx.model,
        ctx.ir_version.to_string(),
        EngineVersion(ctx.engine_version.to_string()),
    );

    let config_label = format!("{}-dt{}", ctx.backend.as_str(), fmt_dt(ctx.dt));
    let config_level = EnsembleConfigLevel { backend: ctx.backend.as_str(), dt: ctx.dt };

    let grid_label = format!("cells-n{}", ctx.cells.len());

    let levels = vec![
        level("model", ctx.stem, structural_level_hash(&model_digest)),
        level("config", &config_label, canonical_config_hash(&config_level, &[])?),
        level("params", "base", canonical_config_hash(&params_levels, &[])?),
        level("grid", &grid_label, canonical_config_hash(&grid_level, &[])?),
    ];
    let level_hashes: Vec<ContentHash> = levels.iter().map(|l| l.hash).collect();
    let rid = run_id(ArtifactKind::SimEnsemble, &level_hashes);
    Ok(ResolvedEnsemble { levels, run_id: rid })
}

/// Build the `deps` (`ArtifactRef` per cell, edge to the `Sim` leaf's
/// `traj.tsv`). `Deps` is hashed as a set sorted by `run_id`, so cell order is
/// irrelevant to the consumer's identity.
pub fn ensemble_deps(cells: &[EnsembleCell]) -> Vec<runid::inputs::ArtifactRef> {
    cells
        .iter()
        .map(|c| runid::inputs::ArtifactRef {
            run_id: c.sim_run_id,
            kind: ArtifactKind::Sim,
            artifact: "traj.tsv".to_string(),
            digest: c.traj_digest,
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cell(scenario: &str, seed: u64, point_id: usize, rid: u8) -> EnsembleCell {
        EnsembleCell {
            scenario_label: scenario.to_string(),
            process_seed: seed,
            point_id,
            sim_run_id: ContentHash::from_bytes([rid; 32]),
            traj_digest: ContentHash::from_bytes([rid ^ 0xff; 32]),
            params_level: ContentHash::from_bytes([rid ^ 0x0f; 32]),
        }
    }

    /// The `grid` level exactly as `resolve_sim_ensemble` builds it, without an
    /// `ir::Model` fixture (mirrors `survey_cas::tests`).
    fn grid_level(cells: &[EnsembleCell], design: Option<&ResolvedDesign>) -> crate::fit::cas::LevelHash {
        let mut s: Vec<(usize, u64, String)> =
            cells.iter().map(|c| (c.point_id, c.process_seed, c.sim_run_id.to_hex())).collect();
        s.sort();
        canonical_config_hash(&EnsembleGridLevel { n_cells: cells.len(), cells: &s, design }, &[]).unwrap()
    }

    fn beta_design(values: Vec<f64>) -> ResolvedDesign {
        ResolvedDesign::Sweep {
            params: vec![SweptParam { name: "beta".to_string(), values }],
            n_points: 2,
        }
    }

    /// The identity pin for the `grid` level (proposal
    /// 2026-10-24-batch-to-simulate §3 "Ensemble identity", §8). The level is
    /// `(point_id, process_seed, sim_run_id)` per cell, the count, and the
    /// design — the arm's label is not in it. These bytes re-keyed every stored
    /// multi-cell ensemble once; a change here re-keys them again and is a
    /// release-notes event, never collateral.
    #[test]
    fn grid_level_identity_is_pinned() {
        let cells = [cell("baseline", 1, 0, 1), cell("baseline", 2, 0, 2)];
        assert_eq!(
            grid_level(&cells, None).into_inner().to_hex(),
            "1bb7c8ce76ab09cae24ecbdc22b9ca69cc3defc3844d4295da4232e60fbdba27",
            "the design-less grid level moved; every stored multi-cell ensemble re-keys"
        );
        assert_eq!(
            grid_level(&cells, Some(&beta_design(vec![0.2, 0.3]))).into_inner().to_hex(),
            "28ab1670e60b47b6459615cac3d768cbc2c15ab153197e4c87600b05649f9fca",
            "the sweep grid level moved; every stored sweep ensemble re-keys"
        );
        let file = ResolvedDesign::File {
            params: vec!["beta".to_string()],
            digest: "0".repeat(64),
            n_points: 2,
            point_ids: vec![0, 1],
        };
        assert_eq!(
            grid_level(&cells, Some(&file)).into_inner().to_hex(),
            "6f24d516083b51736a01b71dde45ccd14b440c191a62b4e2e261b373be471211",
            "the sweep-file grid level moved; every stored sweep-file ensemble re-keys"
        );
    }

    /// The config level digests exactly as the `json!` literal it replaced.
    #[test]
    fn config_level_is_byte_identical_to_the_literal_it_replaced() {
        let cfg = EnsembleConfigLevel { backend: "chain_binomial", dt: 1.0 };
        assert_eq!(
            canonical_config_hash(&cfg, &[]).unwrap(),
            canonical_config_hash(&serde_json::json!({ "backend": "chain_binomial", "dt": 1.0 }), &[]).unwrap(),
            "the config struct must reproduce the literal's digest");
    }

    /// Count-in-the-key: 3 cells vs 4 cells (one extra replicate-seed) MUST
    /// produce different `grid` hashes — the combined TSV has more rows, so it
    /// is a different ensemble (the n_trajectories collision class).
    #[test]
    fn cell_count_is_in_the_key() {
        let three = [cell("baseline", 1, 0, 1), cell("baseline", 2, 0, 2), cell("baseline", 3, 0, 3)];
        let four = [
            cell("baseline", 1, 0, 1), cell("baseline", 2, 0, 2),
            cell("baseline", 3, 0, 3), cell("baseline", 4, 0, 4),
        ];
        assert_ne!(
            grid_level(&three, None), grid_level(&four, None),
            "3 vs 4 cells must produce distinct grid hashes (cell count in the key)"
        );
    }

    /// The grid digest is order-independent (sorted) and content-sensitive,
    /// and the arm's label is not content: a preset-named arm and a
    /// flag-derived arm over identical cells are one ensemble.
    #[test]
    fn grid_is_order_independent_and_content_sensitive_and_label_free() {
        let a = [cell("baseline", 1, 0, 1), cell("baseline", 2, 0, 2)];
        let a_rev = [cell("baseline", 2, 0, 2), cell("baseline", 1, 0, 1)];
        assert_eq!(grid_level(&a, None), grid_level(&a_rev, None), "cell order must not change the grid hash");

        // A changed per-cell sim_run_id (e.g. a --draws param value changed)
        // re-keys the grid even at the same (point, seed).
        let a_diff = [cell("baseline", 1, 0, 9), cell("baseline", 2, 0, 2)];
        assert_ne!(grid_level(&a, None), grid_level(&a_diff, None),
            "a changed cell sim_run_id must change the grid hash");

        // A changed point_id re-keys: the same cells laid out differently on
        // the point axis is a different combined file.
        let a_pt = [cell("baseline", 1, 1, 1), cell("baseline", 2, 0, 2)];
        assert_ne!(grid_level(&a, None), grid_level(&a_pt, None),
            "a changed point_id must change the grid hash");

        // A changed scenario LABEL does not: the cell's run_id already pins the
        // scenario's content, and the label is display.
        let a_scen = [cell("baseline+vaccination", 1, 0, 1), cell("vax", 2, 0, 2)];
        assert_eq!(grid_level(&a, None), grid_level(&a_scen, None),
            "the arm label is not identity; identical cells under two names are one ensemble");
    }

    /// The design is in the key: a sweep job and a design-less job over the
    /// same cells differ, and so do two sweeps whose grids list the same values
    /// in different orders (the same cell set, two different combined files).
    #[test]
    fn design_is_in_the_key() {
        let cells = [cell("baseline", 1, 0, 1), cell("baseline", 1, 1, 2)];
        let none = grid_level(&cells, None);
        let fwd = grid_level(&cells, Some(&beta_design(vec![0.2, 0.3])));
        let rev = grid_level(&cells, Some(&beta_design(vec![0.3, 0.2])));
        assert_ne!(none, fwd, "a sweep job must not share an ensemble with a design-less job");
        assert_ne!(fwd, rev, "`--sweep beta=0.2,0.3` and `--sweep beta=0.3,0.2` are two ensembles");
        assert_eq!(fwd, grid_level(&cells, Some(&beta_design(vec![0.2, 0.3]))));
    }

    /// Deps fold each cell's Sim leaf as a `traj.tsv` edge.
    #[test]
    fn deps_reference_each_sim_leaf() {
        let cells = [cell("baseline", 1, 0, 1), cell("baseline", 2, 0, 2)];
        let deps = ensemble_deps(&cells);
        assert_eq!(deps.len(), 2);
        for d in &deps {
            assert_eq!(d.kind, ArtifactKind::Sim);
            assert_eq!(d.artifact, "traj.tsv");
        }
        assert!(deps.iter().any(|d| d.run_id == ContentHash::from_bytes([1; 32])));
        assert!(deps.iter().any(|d| d.run_id == ContentHash::from_bytes([2; 32])));
    }
}
