//! Reading the content-addressed (`runid::RunRecord`) CAS store.
//!
//! The walk itself is [`runid::discover`] (`runid::leaf`): the presence of a
//! `run.json` (or a `.lock`) is the only discovery signal — no hardcoded level
//! depth — and an unreadable candidate comes back as an explicit
//! [`runid::UnreadableLeaf`] entry. The functions here are the per-kind
//! accessors every command has used, kept with their existing output: each
//! drops the unreadable entries (naming a renamed-`kind` record on stderr, as
//! before) and the leaf state. Their callers move onto `runid::discover`
//! directly, after which these go. The per-kind projections live in `browse`
//! and `fit::fit_view`.

use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::time::SystemTime;

use runid::{ArtifactKind, ContentHash, RunRecord, Scope, Unreadable, UnreadableLeaf};

use crate::cas_index;

// The fit-level provenance sidecar (`fit.meta.json`) lives in
// `run_meta::FitSidecar` (it carries `run_meta` provenance types —
// `ResolvedPriorEntry`, `ParameterProvenance`), with `write_fit_sidecar` /
// `read_fit_sidecar` beside it there.

/// Bounds a caller can put on a per-kind walk, decided from the leaf's cheap
/// header and its directory mtime — before the full `RunRecord` parse. The
/// kind-free part of [`runid::Scope`], whose docs give the rules.
///
/// Every bound is **one-sided**: it may only rule out a leaf the caller was
/// going to discard anyway, so a bounded walk returns exactly the subset of
/// the unbounded one the caller would have kept. Nothing here can change
/// *which* rows a command prints — only what it costs to find them.
#[derive(Default, Clone, Copy)]
pub struct Bounds<'a> {
    /// Skip leaves whose `run_id` is in this set — the per-cell members an
    /// already-discovered ensemble row represents.
    pub skip: Option<&'a HashSet<ContentHash>>,
    /// Skip leaves whose directory mtime proves they were written before this
    /// instant. `camdl list --since` passes `now - duration`: an older mtime
    /// bounds `created_at` below the cutoff, so the row would be filtered out.
    pub not_before: Option<SystemTime>,
}

/// Say once per path, on stderr, that a skipped `run.json` was written by an
/// older camdl under a since-renamed `kind` ([`runid::renamed_kind`]). Any
/// other unreadable record is skipped silently, as before.
fn warn_if_renamed_kind(leaf: &UnreadableLeaf) {
    use std::sync::{Mutex, OnceLock};
    static WARNED: OnceLock<Mutex<HashSet<PathBuf>>> = OnceLock::new();
    if !matches!(leaf.reason, Unreadable::RenamedKind { .. }) {
        return;
    }
    let run_json = leaf.dir.join("run.json");
    let first = WARNED
        .get_or_init(|| Mutex::new(HashSet::new()))
        .lock()
        .map(|mut seen| seen.insert(run_json.clone()))
        .unwrap_or(true);
    if first {
        crate::out::detached()
            .say(format_args!("warning: skipping {}: {}.", run_json.display(), leaf.reason));
    }
}

/// The readable leaves of a [`runid::discover_in_walk_order`] walk, in the
/// order it met them, as `(dir, record)`. Unreadable entries are dropped here
/// (renamed kinds named on stderr), which is what every caller of this module
/// has always been given.
fn readable(under: &Path, scope: Scope) -> Vec<(PathBuf, RunRecord)> {
    runid::discover_in_walk_order(under, scope)
        .into_iter()
        .filter_map(|entry| match entry {
            Ok(leaf) => Some((leaf.dir, leaf.record)),
            Err(unreadable) => {
                warn_if_renamed_kind(&unreadable);
                None
            }
        })
        .collect()
}

/// Every `RunRecord` under `subtree`, of every kind — the unfiltered walk.
pub fn walk_records(subtree: &Path) -> Vec<(PathBuf, RunRecord)> {
    readable(subtree, Scope::default())
}

/// Every leaf of `kind` under that kind's store partition (`sims/`, `fits/`,
/// …) that `bounds` admits. One walk behind all six per-kind accessors below,
/// so the partition dir and the kind can never disagree — both come from
/// [`ArtifactKind::store_dir`].
fn walk_kind(root: &Path, kind: ArtifactKind, bounds: Bounds) -> Vec<Leaf> {
    let scope = Scope { kind: Some(kind), skip: bounds.skip, not_before: bounds.not_before };
    readable(&root.join(kind.store_dir()), scope)
        .into_iter()
        .map(|(dir, record)| Leaf { dir, record })
        .collect()
}

/// A new-format leaf record with its directory, plus convenience accessors
/// that read the factored level labels (provenance) for display.
#[derive(Debug, Clone)]
pub struct Leaf {
    pub dir: PathBuf,
    pub record: RunRecord,
}

impl Leaf {
    /// A level's readable label by level name (`"model"`, `"scenario"`, …).
    pub fn level_label(&self, name: &str) -> &str {
        self.record
            .levels
            .iter()
            .find(|l| l.name == name)
            .map(|l| l.label.as_str())
            .unwrap_or("")
    }

    pub fn run_id_hex(&self) -> String {
        self.record.run_id.to_hex()
    }

    /// The base seed parsed from the `seed_{n}` label (0 if absent/unparsed).
    pub fn seed(&self) -> u64 {
        self.level_label("seed")
            .strip_prefix("seed_")
            .and_then(|s| s.parse().ok())
            .unwrap_or(0)
    }

    /// `traj.tsv` size in bytes (0 if absent).
    pub fn traj_bytes(&self) -> u64 {
        std::fs::metadata(self.dir.join("traj.tsv")).map(|m| m.len()).unwrap_or(0)
    }
}

/// Resolve leaves of `kind` whose `run_id` hex starts with `prefix`, using the
/// derived index as an accelerator with `run.json` as the source of truth.
///
/// The fast path consults [`cas_index::resolve_prefix`], every hit of which is
/// verified against the live `run.json` (a stale/repointed entry is dropped —
/// never resolved to a dead path). On an index miss (no index, no matching
/// entry, or every candidate stale) it falls back to the full per-kind walk —
/// which finds out-of-band-added leaves the index lacks — and then repairs the
/// index from a fresh full-tree walk (best-effort; a cache, never a gate).
fn resolve_prefix_indexed(
    root: &Path,
    kind: ArtifactKind,
    prefix: &str,
    walk: impl Fn(&Path) -> Vec<Leaf>,
) -> Vec<Leaf> {
    if let Some(hits) = cas_index::resolve_prefix(root, kind, prefix) {
        return hits;
    }
    // Index miss → authoritative full walk (out-of-band leaves are found
    // here), then repair the index so the next lookup is fast.
    let hits: Vec<Leaf> =
        walk(root).into_iter().filter(|s| s.run_id_hex().starts_with(prefix)).collect();
    let _ = cas_index::rebuild(root);
    hits
}

/// All `sims/` leaves of kind `Sim` (new-format trajectory runs).
pub fn walk_sim_leaves(root: &Path) -> Vec<Leaf> {
    walk_kind(root, ArtifactKind::Sim, Bounds::default())
}

/// All `sims/` leaves of kind `Sim` EXCEPT those whose `run_id` is in
/// `members` — the per-cell leaves an already-discovered `SimEnsemble` row
/// represents, which `camdl list` must not print a second time.
///
/// Separate from [`walk_sim_leaves`] because the exclusion has to happen
/// *inside* the walk: on the store in gh#699, 461,282 of 550,647 sim leaves
/// are ensemble members, so discovering them only to drop them afterwards is
/// the whole cost of the command. A member is recognized from its `run_id`
/// alone, which the walk reads from the record's header without parsing it.
pub fn walk_sim_leaves_bounded(root: &Path, bounds: Bounds) -> Vec<Leaf> {
    walk_kind(root, ArtifactKind::Sim, bounds)
}

/// New-format sims whose `run_id` hex matches `prefix` (for `show`/`cat`
/// prefix resolution; combined with the legacy `run.hash` matches in
/// `browse::resolve_any` so a user can address any run during M2→M3).
pub fn resolve_sim_prefix(root: &Path, prefix: &str) -> Vec<Leaf> {
    resolve_prefix_indexed(root, ArtifactKind::Sim, prefix, walk_sim_leaves)
}

/// All `fits/` leaves of kind `FitMethod` (new-format fit-method runs, M3.2).
pub fn walk_fit_leaves(root: &Path) -> Vec<Leaf> {
    walk_kind(root, ArtifactKind::FitMethod, Bounds::default())
}

/// New-format fit methods whose `run_id` hex matches `prefix` (for `show`/`cat`
/// prefix resolution alongside `resolve_sim_prefix`).
pub fn resolve_fit_prefix(root: &Path, prefix: &str) -> Vec<Leaf> {
    resolve_prefix_indexed(root, ArtifactKind::FitMethod, prefix, walk_fit_leaves)
}

/// All `profiles/` leaves of kind `ProfilePoint` (new-format profile-point
/// mini-fits, M3.3). Each is one `(grid point × seed × start)` cell under the
/// factored `profile/point/method/seed/start` tree.
pub fn walk_profile_leaves(root: &Path) -> Vec<Leaf> {
    walk_kind(root, ArtifactKind::ProfilePoint, Bounds::default())
}

/// New-format profile points whose `run_id` hex matches `prefix` (for
/// `show`/`cat` prefix resolution alongside `resolve_sim_prefix`).
pub fn resolve_profile_prefix(root: &Path, prefix: &str) -> Vec<Leaf> {
    resolve_prefix_indexed(root, ArtifactKind::ProfilePoint, prefix, walk_profile_leaves)
}

/// All `pfilters/` leaves of kind `Pfilter` (new-format particle-filter evals,
/// M3.3). Each is one `(model × config × params × seed)` standalone eval —
/// a single leaf, no grid.
pub fn walk_pfilter_leaves(root: &Path) -> Vec<Leaf> {
    walk_kind(root, ArtifactKind::Pfilter, Bounds::default())
}

/// New-format pfilter evals whose `run_id` hex matches `prefix` (for
/// `show`/`cat` prefix resolution alongside `resolve_sim_prefix`).
pub fn resolve_pfilter_prefix(root: &Path, prefix: &str) -> Vec<Leaf> {
    resolve_prefix_indexed(root, ArtifactKind::Pfilter, prefix, walk_pfilter_leaves)
}

/// All `surveys/` leaves of kind `Survey` (new-format likelihood-landscape
/// surveys, M3.3). Each is one `(model × config × box × seed)` LHS landscape —
/// a single leaf, the N points are within it (not an axis).
pub fn walk_survey_leaves(root: &Path) -> Vec<Leaf> {
    walk_kind(root, ArtifactKind::Survey, Bounds::default())
}

/// New-format surveys whose `run_id` hex matches `prefix` (for `show`/`cat`
/// prefix resolution alongside `resolve_sim_prefix`).
pub fn resolve_survey_prefix(root: &Path, prefix: &str) -> Vec<Leaf> {
    resolve_prefix_indexed(root, ArtifactKind::Survey, prefix, walk_survey_leaves)
}

/// All `ensembles/` leaves of kind `SimEnsemble` (the combined-across-cells
/// wide-format TSV of a multi-cell `simulate`). Each references its N per-cell
/// `Sim` leaves via `deps`; the combined TSV is its `ensemble.tsv` artifact.
pub fn walk_sim_ensemble_leaves(root: &Path) -> Vec<Leaf> {
    walk_kind(root, ArtifactKind::SimEnsemble, Bounds::default())
}

/// New-format ensembles whose `run_id` hex matches `prefix` (for `show`/`cat`
/// prefix resolution alongside `resolve_sim_prefix`).
pub fn resolve_sim_ensemble_prefix(root: &Path, prefix: &str) -> Vec<Leaf> {
    resolve_prefix_indexed(root, ArtifactKind::SimEnsemble, prefix, walk_sim_ensemble_leaves)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn record(kind: &str, run_id: &str) -> String {
        format!(
            r#"{{"format_version":1,"kind":"{kind}","run_id":"{run_id}","hash_version":1,
                "ir_version":"0.31","engine_version":"0.1.0+test",
                "levels":[{{"name":"seed","label":"seed_1","hash":"{run_id}","schema_version":1}}],
                "status":"completed","artifacts":{{}},
                "provenance":{{"created_at":"2026-08-01T12:00:00Z","argv":[]}}}}"#
        )
    }

    fn plant(dir: &Path, bytes: &str) {
        std::fs::create_dir_all(dir).unwrap();
        std::fs::write(dir.join("run.json"), bytes).unwrap();
    }

    fn id(stem: &str) -> String {
        format!("{:0<64}", stem)
    }

    /// The accessors keep their output while the walk reports unreadable
    /// candidates: garbage, a truncated record, and a lock-only directory are
    /// entries of `runid::discover`, and none of them is a row here.
    #[test]
    fn the_accessors_drop_unreadable_candidates() {
        let tmp = tempfile::tempdir().unwrap();
        let root = tmp.path();
        let sims = root.join("sims");
        plant(&sims.join("good"), &record("sim", &id("a1")));
        plant(&sims.join("garbage"), "{ not json");
        let cut = record("sim", &id("b2"));
        plant(&sims.join("truncated"), &cut[..cut.len() / 2]);
        std::fs::create_dir_all(sims.join("lock-only")).unwrap();
        std::fs::write(sims.join("lock-only/.lock"), "1\n").unwrap();

        let discovered = runid::discover(root, Scope::default());
        assert_eq!(discovered.len(), 4, "{discovered:?}");
        assert_eq!(discovered.iter().filter(|e| e.is_err()).count(), 3);

        let rows: Vec<String> = walk_records(root).iter().map(|(_, r)| r.run_id.to_hex()).collect();
        assert_eq!(rows, vec![id("a1")]);
        let rows: Vec<String> = walk_sim_leaves(root).iter().map(Leaf::run_id_hex).collect();
        assert_eq!(rows, vec![id("a1")]);
    }

    /// `walk_sim_leaves_bounded` hands its bounds to the walk: the skipped
    /// members are not rows, and the unbounded walk still returns them.
    #[test]
    fn the_bounded_accessor_passes_its_bounds_to_the_walk() {
        let tmp = tempfile::tempdir().unwrap();
        let root = tmp.path();
        let members = [id("aa01"), id("aa02")];
        for (i, m) in members.iter().enumerate() {
            plant(&root.join(format!("sims/m/draw{i}")), &record("sim", m));
        }
        plant(&root.join("sims/solo"), &record("sim", &id("ff01")));
        let skip: HashSet<ContentHash> =
            members.iter().map(|m| ContentHash::from_hex(m).unwrap()).collect();

        let rows: Vec<String> =
            walk_sim_leaves_bounded(root, Bounds { skip: Some(&skip), not_before: None })
                .iter()
                .map(Leaf::run_id_hex)
                .collect();
        assert_eq!(rows, vec![id("ff01")]);
        assert_eq!(walk_sim_leaves(root).len(), 3);
    }
}
