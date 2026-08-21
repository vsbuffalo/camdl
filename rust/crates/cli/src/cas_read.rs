//! Reading the content-addressed (`runid::RunRecord`) CAS store.
//!
//! A generic, Layout-driven walk: the presence of a parseable `RunRecord`
//! `run.json` is the only discovery signal — no hardcoded level depth. Every
//! run kind (sim / fit-stage / profile-point / pfilter / survey) is discovered
//! here; the per-kind projections live in `browse` and `fit::fit_view`.

use std::collections::HashSet;
use std::path::{Path, PathBuf};

use runid::{ArtifactKind, ContentHash, RunRecord};

use crate::cas_index;

/// Full `RunRecord` parses performed by [`walk_records`] in this process.
///
/// The point of the gated walks below is that a leaf the caller is going to
/// discard never costs a full parse — a claim about *work*, not about the rows
/// returned, so counting the parses is the only way to assert it. Compiled out
/// of the shipped binary.
#[cfg(test)]
pub(crate) static FULL_PARSES: std::sync::atomic::AtomicUsize =
    std::sync::atomic::AtomicUsize::new(0);

// The fit-level provenance sidecar (`fit.meta.json`) lives in
// `run_meta::FitSidecar` (it carries `run_meta` provenance types —
// `ResolvedPriorEntry`, `ParameterProvenance`), with `write_fit_sidecar` /
// `read_fit_sidecar` beside it there.

/// Recursively collect every `(dir, RunRecord)` under `subtree` whose dir holds
/// a parseable `RunRecord` `run.json`. Hidden dirs (`.staging`, `.quarantine`)
/// are skipped. Descends through leaves too, but a leaf's declared child
/// sub-artifacts under `obs/…` carry an `obs.json` (not a `run.json`), so they
/// are not surfaced here as standalone records — they're reached as the
/// trajectory leaf's `children`.
pub fn walk_records(subtree: &Path) -> Vec<(PathBuf, RunRecord)> {
    let mut out = Vec::new();
    if !subtree.exists() {
        return out;
    }
    let mut stack = vec![subtree.to_path_buf()];
    while let Some(dir) = stack.pop() {
        let rj = dir.join("run.json");
        if rj.is_file() {
            if let Ok(bytes) = std::fs::read(&rj) {
                #[cfg(test)]
                FULL_PARSES.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                if let Ok(rec) = serde_json::from_slice::<RunRecord>(&bytes) {
                    out.push((dir.clone(), rec));
                }
            }
        }
        if let Ok(entries) = std::fs::read_dir(&dir) {
            for e in entries.flatten() {
                let p = e.path();
                if !p.is_dir() {
                    continue;
                }
                let name = e.file_name();
                if name.to_string_lossy().starts_with('.') {
                    continue; // .staging / .quarantine
                }
                stack.push(p);
            }
        }
    }
    out
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
    walk_records(&root.join("sims"))
        .into_iter()
        .filter(|(_, r)| r.kind == ArtifactKind::Sim)
        .map(|(dir, record)| Leaf { dir, record })
        .collect()
}

/// All `sims/` leaves of kind `Sim` EXCEPT those whose `run_id` is in
/// `members` — the per-cell leaves an already-discovered `SimEnsemble` row
/// represents, which `camdl list` must not print a second time.
///
/// Separate from [`walk_sim_leaves`] because the exclusion has to happen
/// *inside* the walk: on the store in gh#699, 461,282 of 550,647 sim leaves
/// are ensemble members, so discovering them only to drop them afterwards is
/// the whole cost of the command.
pub fn walk_sim_leaves_excluding(root: &Path, members: &HashSet<ContentHash>) -> Vec<Leaf> {
    walk_sim_leaves(root).into_iter().filter(|l| !members.contains(&l.record.run_id)).collect()
}

/// New-format sims whose `run_id` hex matches `prefix` (for `show`/`cat`
/// prefix resolution; combined with the legacy `run.hash` matches in
/// `browse::resolve_any` so a user can address any run during M2→M3).
pub fn resolve_sim_prefix(root: &Path, prefix: &str) -> Vec<Leaf> {
    resolve_prefix_indexed(root, ArtifactKind::Sim, prefix, walk_sim_leaves)
}

/// All `fits/` leaves of kind `FitStage` (new-format fit-stage runs, M3.2).
pub fn walk_fit_leaves(root: &Path) -> Vec<Leaf> {
    walk_records(&root.join("fits"))
        .into_iter()
        .filter(|(_, r)| r.kind == ArtifactKind::FitStage)
        .map(|(dir, record)| Leaf { dir, record })
        .collect()
}

/// New-format fit stages whose `run_id` hex matches `prefix` (for `show`/`cat`
/// prefix resolution alongside `resolve_sim_prefix`).
pub fn resolve_fit_prefix(root: &Path, prefix: &str) -> Vec<Leaf> {
    resolve_prefix_indexed(root, ArtifactKind::FitStage, prefix, walk_fit_leaves)
}

/// All `profiles/` leaves of kind `ProfilePoint` (new-format profile-point
/// mini-fits, M3.3). Each is one `(grid point × seed × start)` cell under the
/// factored `profile/point/stage/seed/start` tree.
pub fn walk_profile_leaves(root: &Path) -> Vec<Leaf> {
    walk_records(&root.join("profiles"))
        .into_iter()
        .filter(|(_, r)| r.kind == ArtifactKind::ProfilePoint)
        .map(|(dir, record)| Leaf { dir, record })
        .collect()
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
    walk_records(&root.join("pfilters"))
        .into_iter()
        .filter(|(_, r)| r.kind == ArtifactKind::Pfilter)
        .map(|(dir, record)| Leaf { dir, record })
        .collect()
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
    walk_records(&root.join("surveys"))
        .into_iter()
        .filter(|(_, r)| r.kind == ArtifactKind::Survey)
        .map(|(dir, record)| Leaf { dir, record })
        .collect()
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
    walk_records(&root.join("ensembles"))
        .into_iter()
        .filter(|(_, r)| r.kind == ArtifactKind::SimEnsemble)
        .map(|(dir, record)| Leaf { dir, record })
        .collect()
}

/// New-format ensembles whose `run_id` hex matches `prefix` (for `show`/`cat`
/// prefix resolution alongside `resolve_sim_prefix`).
pub fn resolve_sim_ensemble_prefix(root: &Path, prefix: &str) -> Vec<Leaf> {
    resolve_prefix_indexed(root, ArtifactKind::SimEnsemble, prefix, walk_sim_ensemble_leaves)
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use std::sync::atomic::Ordering;
    use std::sync::{Mutex, MutexGuard};

    /// Serializes the tests that read [`FULL_PARSES`] — the counter is process
    /// -wide and `cargo test` runs test functions on parallel threads.
    static PARSE_COUNT: Mutex<()> = Mutex::new(());

    /// Take the parse-count lock and zero the counter. Hold the guard for the
    /// duration of the walk being measured.
    fn measuring() -> MutexGuard<'static, ()> {
        let guard = PARSE_COUNT.lock().unwrap_or_else(|e| e.into_inner());
        FULL_PARSES.store(0, Ordering::Relaxed);
        guard
    }

    fn full_parses() -> usize {
        FULL_PARSES.load(Ordering::Relaxed)
    }

    /// A 64-hex `run_id` from a short readable stem.
    pub fn id(stem: &str) -> String {
        format!("{:0<64}", stem)
    }

    /// Plant a `Sim` leaf at `root/sims/<sub>/` whose `run.json` carries
    /// `run_id` plus a kilobyte-scale `inputs` payload, the shape a real
    /// per-draw leaf has (that payload is exactly what the gated walk must not
    /// pay to materialize).
    pub fn plant_sim(root: &Path, sub: &str, run_id: &str) -> PathBuf {
        let dir = root.join("sims").join(sub);
        std::fs::create_dir_all(&dir).unwrap();
        let params: Vec<String> =
            (0..40).map(|i| format!("\"theta_{i}\": {}.5", i)).collect();
        let rec = format!(
            r#"{{
                "format_version": 1,
                "kind": "sim",
                "run_id": "{run_id}",
                "hash_version": 1,
                "ir_version": "0.31",
                "engine_version": "0.1.0+test",
                "levels": [
                    {{"name":"model","label":"m","hash":"{run_id}","schema_version":1}},
                    {{"name":"scenario","label":"baseline","hash":"{run_id}","schema_version":1}},
                    {{"name":"seed","label":"seed_1","hash":"{run_id}","schema_version":1}}
                ],
                "status": "completed",
                "artifacts": {{}},
                "inputs": {{"backend":"chain_binomial","dt":1.0,"params":{{{}}}}},
                "provenance": {{"created_at":"2026-08-01T12:00:00Z","argv":[]}}
            }}"#,
            params.join(",")
        );
        std::fs::write(dir.join("run.json"), rec).unwrap();
        dir
    }

    /// Plant a `SimEnsemble` leaf whose `deps` name `members` as its per-cell
    /// `Sim` leaves — the record `camdl list` reads to learn which sim leaves
    /// its own row already represents.
    pub fn plant_ensemble(root: &Path, sub: &str, run_id: &str, members: &[String]) -> PathBuf {
        let dir = root.join("ensembles").join(sub);
        std::fs::create_dir_all(&dir).unwrap();
        let deps: Vec<String> = members
            .iter()
            .map(|m| {
                format!(
                    r#"{{"run_id":"{m}","kind":"sim","artifact":"traj.tsv","digest":"{m}"}}"#
                )
            })
            .collect();
        let rec = format!(
            r#"{{
                "format_version": 1,
                "kind": "sim_ensemble",
                "run_id": "{run_id}",
                "hash_version": 1,
                "ir_version": "0.31",
                "engine_version": "0.1.0+test",
                "levels": [
                    {{"name":"model","label":"m","hash":"{run_id}","schema_version":1}},
                    {{"name":"grid","label":"cells-n{}","hash":"{run_id}","schema_version":1}}
                ],
                "deps": [{}],
                "status": "completed",
                "artifacts": {{}},
                "provenance": {{"created_at":"2026-08-01T12:05:00Z","argv":[]}}
            }}"#,
            members.len(),
            deps.join(",")
        );
        std::fs::write(dir.join("run.json"), rec).unwrap();
        dir
    }

    /// A store shaped like the one in gh#699: most sim leaves are members of an
    /// ensemble that already represents them. Returns (member ids, free ids).
    pub fn ensemble_heavy_store(root: &Path, n_members: usize, n_free: usize)
        -> (Vec<String>, Vec<String>)
    {
        let members: Vec<String> = (0..n_members).map(|i| id(&format!("aa{i:04x}"))).collect();
        for (i, m) in members.iter().enumerate() {
            plant_sim(root, &format!("m-1111/draw{i}-2222/baseline-3333/seed_1-4444"), m);
        }
        let free: Vec<String> = (0..n_free).map(|i| id(&format!("ff{i:04x}"))).collect();
        for (i, f) in free.iter().enumerate() {
            plant_sim(root, &format!("solo-5555/cfg{i}-6666/baseline-3333/seed_1-4444"), f);
        }
        plant_ensemble(root, "m-1111/cells-n-7777", &id("e0"), &members);
        (members, free)
    }

    /// gh#699. `camdl list` suppresses every sim leaf an ensemble row already
    /// represents — so those leaves must never cost a full `RunRecord` parse.
    /// The walk may only materialize the leaves that survive to be printed.
    #[test]
    #[ignore = "red until walk_sim_leaves_excluding gates before the full parse (gh#699)"]
    fn ensemble_members_are_skipped_without_a_full_parse() {
        let tmp = tempfile::tempdir().unwrap();
        let root = tmp.path();
        let (members, free) = ensemble_heavy_store(root, 40, 5);

        let member_ids: HashSet<ContentHash> =
            members.iter().map(|m| ContentHash::from_hex(m).unwrap()).collect();

        let _guard = measuring();
        let leaves = walk_sim_leaves_excluding(root, &member_ids);

        // Correctness: exactly the non-member leaves, unchanged.
        let mut got: Vec<String> = leaves.iter().map(|l| l.run_id_hex()).collect();
        got.sort();
        let mut want = free.clone();
        want.sort();
        assert_eq!(got, want, "the printed row set must be the non-member leaves");

        // Cost: one full parse per printed row, not one per leaf in the tree.
        assert_eq!(
            full_parses(),
            free.len(),
            "ensemble members must not be fully parsed ({} sim leaves in the tree)",
            members.len() + free.len()
        );
    }
}
