//! gh#147 (M-ensemble). The `SimEnsemble` CAS identity: map a multi-cell
//! `simulate` (`--replicates`/`--seeds`/multi-scenario/`--draws`) into the
//! `runid` factored levels (`model` / `config` / `params` / `grid`) and its
//! leaf `run_id`.
//!
//! A multi-cell `simulate` writes one per-cell [`Sim`](runid::ArtifactKind::Sim)
//! leaf (byte-identical to `batch run`) AND a combined wide-format trajectory
//! TSV that interleaves every cell with `replicate`/`scenario`/`draw` columns.
//! That combined TSV is the *ensemble* artifact: a derived view over the N
//! leaves. Its identity must be a pure function of everything that determines
//! the combined bytes:
//!   - **model** — the pure model IR digest (constant across cells).
//!   - **config** — backend + dt (shared by every cell).
//!   - **params** — the resolved base parameter map (the values shared across
//!     cells before any per-draw override; the per-draw deltas ride in `grid`
//!     via each cell's `Sim` run_id).
//!   - **grid** — a digest over the SORTED cell list. Each cell contributes
//!     `(scenario_label, process_seed, draw_idx, sim_run_id)`; the `sim_run_id`
//!     already encodes that cell's model/config/params/scenario/seed, so a
//!     changed `--draws` param value (same draw index) re-keys the ensemble.
//!     The cell COUNT is folded in explicitly (`n_cells`): 3 replicates vs 4
//!     is a different combined TSV, so a different ensemble (count-in-the-key,
//!     the n_trajectories collision class). Sorting makes the digest
//!     order-independent.
//!
//! Mirrors [`crate::survey_cas`] / [`crate::pfilter_cas`] for the level/digest
//! conventions.

use runid::inputs::{EngineVersion, ModelDigest};
use runid::{run_id, ArtifactKind, ContentAddressed, ContentHash, LevelId};

use crate::fit::cas::canonical_config_hash;

/// One cell of a multi-cell `simulate`, contributing to the ensemble's `grid`
/// digest and to its `deps` (the cell's `Sim` leaf).
#[derive(Debug, Clone)]
pub struct EnsembleCell {
    /// The scenario label rendered into the combined TSV's `scenario` column.
    pub scenario_label: String,
    /// The resolved process seed driving this cell's trajectory.
    pub process_seed: u64,
    /// The 0-based draw index (the `draw` column; 0 when not a `--draws` run).
    pub draw_idx: usize,
    /// The cell's `Sim` leaf `run_id` — its full identity (model/config/params/
    /// scenario/seed). Folding it into `grid` makes a per-draw param change
    /// re-key the ensemble.
    pub sim_run_id: ContentHash,
    /// SHA-256 of the cell's `traj.tsv` — the `deps` edge's consumed-artifact
    /// digest (which upstream file the combined TSV was built from).
    pub traj_digest: ContentHash,
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
    /// Resolved base parameter map (name → value), shared across cells.
    pub base_params: &'a std::collections::HashMap<String, f64>,
    /// The full set of cells the run expands to.
    pub cells: &'a [EnsembleCell],
}

use crate::fit::cas::level;

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

/// The `grid` level: the sorted cell list plus the explicit count, so N vs
/// N+1 replicates is a different ensemble.
#[derive(serde::Serialize)]
struct EnsembleGridLevel<'a> {
    n_cells: usize,
    cells: &'a [(&'a str, u64, usize, String)],
}

pub fn resolve_sim_ensemble(ctx: &EnsembleCtx) -> Result<ResolvedEnsemble, String> {
    // params level — the resolved base map, sorted + finiteness-gated (a
    // non-finite would null-collapse and collide distinct values).
    let mut params_sorted: Vec<(&str, f64)> =
        ctx.base_params.iter().map(|(k, v)| (k.as_str(), *v)).collect();
    params_sorted.sort_by(|a, b| a.0.cmp(b.0));
    // The `params` level is the sorted (name, value) list itself — a sequence
    // with no field set to forget, so it takes no wrapper struct (and wrapping
    // it would re-key: an object is not an array). `canonical_config_hash`
    // gates finiteness on the raw values, which `json!` would already have
    // collapsed for a NaN.

    // grid level — the sorted cell list + the explicit cell count. Each cell is
    // (scenario, seed, draw, sim_run_id); sorting is order-independent. The
    // count is folded so N vs N+1 replicates is a different ensemble.
    let mut cells_sorted: Vec<(&str, u64, usize, String)> = ctx
        .cells
        .iter()
        .map(|c| {
            (
                c.scenario_label.as_str(),
                c.process_seed,
                c.draw_idx,
                c.sim_run_id.to_hex(),
            )
        })
        .collect();
    cells_sorted.sort();
    let grid_level = EnsembleGridLevel { n_cells: ctx.cells.len(), cells: &cells_sorted };

    let model_digest = ModelDigest::from_model(
        ctx.model,
        ctx.ir_version.to_string(),
        EngineVersion(ctx.engine_version.to_string()),
    );

    let config_label = format!("{}-dt{}", ctx.backend.as_str(), fmt_dt(ctx.dt));
    let config_level = EnsembleConfigLevel { backend: ctx.backend.as_str(), dt: ctx.dt };

    let grid_label = format!("cells-n{}", ctx.cells.len());

    let levels = vec![
        level("model", ctx.stem, model_digest.content_hash()),
        level("config", &config_label, canonical_config_hash(&config_level, &[])?),
        level("params", "base", canonical_config_hash(&params_sorted, &[])?),
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
    use crate::fit::cas::digest_value;

    fn cell(scenario: &str, seed: u64, draw: usize, rid: u8) -> EnsembleCell {
        EnsembleCell {
            scenario_label: scenario.to_string(),
            process_seed: seed,
            draw_idx: draw,
            sim_run_id: ContentHash::from_bytes([rid; 32]),
            traj_digest: ContentHash::from_bytes([rid ^ 0xff; 32]),
        }
    }

    /// The `grid` level is `digest_value` over the same blob `resolve` builds;
    /// unit-testing it pins the collision-freeness of the cell-set identity
    /// without an `ir::Model` fixture (mirrors `survey_cas::tests`).
    fn grid_level(cells: &[EnsembleCell]) -> ContentHash {
        let mut s: Vec<(&str, u64, usize, String)> = cells
            .iter()
            .map(|c| (c.scenario_label.as_str(), c.process_seed, c.draw_idx, c.sim_run_id.to_hex()))
            .collect();
        s.sort();
        digest_value(&serde_json::json!({ "n_cells": cells.len(), "cells": s }))
    }

    /// Byte-neutrality of the struct rewrite: `EnsembleGridLevel` /
    /// `EnsembleConfigLevel` must digest EXACTLY as the `json!` literals they
    /// replaced — `grid_level` above IS that literal — or every stored
    /// ensemble leaf silently re-keys.
    #[test]
    fn levels_are_byte_identical_to_the_literals_they_replaced() {
        let cells = [cell("baseline", 1, 0, 1), cell("baseline", 2, 0, 2)];
        let mut sorted: Vec<(&str, u64, usize, String)> = cells
            .iter()
            .map(|c| (c.scenario_label.as_str(), c.process_seed, c.draw_idx, c.sim_run_id.to_hex()))
            .collect();
        sorted.sort();
        let grid = EnsembleGridLevel { n_cells: cells.len(), cells: &sorted };
        assert_eq!(canonical_config_hash(&grid, &[]).unwrap(), grid_level(&cells),
            "the grid struct must reproduce the literal's digest");

        let cfg = EnsembleConfigLevel { backend: "chain_binomial", dt: 1.0 };
        assert_eq!(
            canonical_config_hash(&cfg, &[]).unwrap(),
            digest_value(&serde_json::json!({ "backend": "chain_binomial", "dt": 1.0 })),
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
            grid_level(&three), grid_level(&four),
            "3 vs 4 cells must produce distinct grid hashes (cell count in the key)"
        );
    }

    /// The grid digest is order-independent (sorted) but content-sensitive.
    #[test]
    fn grid_is_order_independent_and_content_sensitive() {
        let a = [cell("baseline", 1, 0, 1), cell("baseline", 2, 0, 2)];
        let a_rev = [cell("baseline", 2, 0, 2), cell("baseline", 1, 0, 1)];
        assert_eq!(grid_level(&a), grid_level(&a_rev), "cell order must not change the grid hash");

        // A changed per-cell sim_run_id (e.g. a --draws param value changed)
        // re-keys the grid even at the same (scenario, seed, draw).
        let a_diff = [cell("baseline", 1, 0, 9), cell("baseline", 2, 0, 2)];
        assert_ne!(grid_level(&a), grid_level(&a_diff),
            "a changed cell sim_run_id must change the grid hash");

        // A changed scenario label re-keys.
        let a_scen = [cell("vax", 1, 0, 1), cell("baseline", 2, 0, 2)];
        assert_ne!(grid_level(&a), grid_level(&a_scen),
            "a changed scenario label must change the grid hash");
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
