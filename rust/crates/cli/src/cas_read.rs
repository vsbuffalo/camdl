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

#[cfg(test)]
thread_local! {
    /// Full `RunRecord` parses performed by the walk **on this thread**.
    ///
    /// The point of the gated walks below is that a leaf the caller is going
    /// to discard never costs a full parse — a claim about *work*, not about
    /// the rows returned, which are required to stay identical — so counting
    /// the parses is the only way to assert it. Thread-local because the test
    /// harness gives each test its own thread and the walk is single-threaded,
    /// so each test sees exactly its own count with no synchronization.
    /// Compiled out of the shipped binary.
    pub(crate) static FULL_PARSES: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

// The fit-level provenance sidecar (`fit.meta.json`) lives in
// `run_meta::FitSidecar` (it carries `run_meta` provenance types —
// `ResolvedPriorEntry`, `ParameterProvenance`), with `write_fit_sidecar` /
// `read_fit_sidecar` beside it there.

/// The two identity fields a walk reads out of a `run.json` before deciding
/// whether the record is worth materializing.
struct RunHeader {
    kind: ArtifactKind,
    run_id: ContentHash,
}

/// Read `kind` + `run_id` out of `run.json` bytes WITHOUT materializing the
/// rest of the record.
///
/// `RunRecord` serializes `format_version`, `kind`, `run_id` first (its field
/// declaration order, `runid::record`), and this stops at the second of the
/// two — so on a leaf record of several kilobytes it tokenizes roughly a
/// hundred bytes, and `levels`, `artifacts`, `output_schema`, `inputs` and
/// `provenance` are never even scanned, let alone allocated. Field order is a
/// *speed* assumption only: a record that spells those two fields last still
/// reads correctly here, just without the early exit.
///
/// `None` for anything that is not a JSON object carrying both fields —
/// exactly the inputs on which the full `RunRecord` parse also fails, so
/// gating on this can never drop a leaf [`walk_records`] would have kept.
fn run_header(bytes: &[u8]) -> Option<RunHeader> {
    use serde::de::{Error as _, IgnoredAny, MapAccess, Visitor};

    /// The header is handed back through `out` rather than as the visitor's
    /// value because the visitor **aborts** once it has both fields, and an
    /// aborted deserialization has no value to return.
    struct HeaderVisitor<'a> {
        out: &'a mut Option<RunHeader>,
    }

    impl<'de> Visitor<'de> for HeaderVisitor<'_> {
        type Value = ();

        fn expecting(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
            f.write_str("a run.json object carrying `kind` and `run_id`")
        }

        fn visit_map<A: MapAccess<'de>>(self, mut map: A) -> Result<(), A::Error> {
            let mut kind: Option<ArtifactKind> = None;
            let mut run_id: Option<ContentHash> = None;
            while let Some(key) = map.next_key::<String>()? {
                match key.as_str() {
                    "kind" => kind = Some(map.next_value()?),
                    "run_id" => run_id = Some(map.next_value()?),
                    _ => {
                        map.next_value::<IgnoredAny>()?;
                    }
                }
                if let (Some(kind), Some(run_id)) = (kind, run_id) {
                    *self.out = Some(RunHeader { kind, run_id });
                    // Stop: everything after `run_id` in a leaf record is the
                    // kilobytes we are here not to read. serde_json still wants
                    // the closing brace, so the only way out mid-document is an
                    // error the caller discards.
                    return Err(A::Error::custom("header read complete"));
                }
            }
            Ok(())
        }
    }

    let mut out = None;
    let mut de = serde_json::Deserializer::from_slice(bytes);
    // The error is deliberately ignored: it is either the early-exit abort
    // (with `out` set) or a malformed record (with `out` still `None`).
    let _ = serde::Deserializer::deserialize_map(&mut de, HeaderVisitor { out: &mut out });
    out
}

/// What a walk may rule out from a leaf's cheap header, before paying for the
/// full `RunRecord` parse.
///
/// Both fields are one-sided: they may only rule out a leaf the caller was
/// going to discard anyway, so a gated walk returns exactly the subset of
/// [`walk_records`]'s result the caller would have kept. Nothing here can
/// change *which* rows a command prints — only what it costs to find them.
struct LeafGate<'a> {
    /// Keep only leaves of this kind; `None` keeps every kind.
    kind: Option<ArtifactKind>,
    /// Skip leaves whose `run_id` is in this set (the per-cell members an
    /// ensemble row already represents).
    members: Option<&'a HashSet<ContentHash>>,
}

impl LeafGate<'_> {
    /// Materialize every record found — the unfiltered walk.
    const ANY: LeafGate<'static> = LeafGate { kind: None, members: None };

    /// Whether reading the cheap header can rule anything out. When it cannot,
    /// reading it would be pure overhead.
    fn is_selective(&self) -> bool {
        self.kind.is_some() || self.members.is_some_and(|m| !m.is_empty())
    }

    fn admits(&self, header: &RunHeader) -> bool {
        self.kind.is_none_or(|k| k == header.kind)
            && !self.members.is_some_and(|m| m.contains(&header.run_id))
    }
}

/// Recursively collect every `(dir, RunRecord)` under `subtree` whose dir holds
/// a parseable `RunRecord` `run.json` **that `gate` admits**. Hidden dirs
/// (`.staging`, `.quarantine`) are skipped. Descends through leaves too, but a
/// leaf's declared child sub-artifacts under `obs/…` carry an `obs.json` (not a
/// `run.json`), so they are not surfaced here as standalone records — they're
/// reached as the trajectory leaf's `children`.
fn walk_gated(subtree: &Path, gate: &LeafGate) -> Vec<(PathBuf, RunRecord)> {
    let mut out = Vec::new();
    if !subtree.exists() {
        return out;
    }
    let mut stack = vec![subtree.to_path_buf()];
    while let Some(dir) = stack.pop() {
        // One `read_dir` answers both questions about this directory — which
        // children to descend into, and whether it holds a `run.json` — from
        // the entry types the directory listing already carries.
        let run_json = match std::fs::read_dir(&dir) {
            Ok(entries) => scan_dir(entries, &mut stack),
            // A directory that cannot be listed may still hold a readable
            // `run.json` (mode 0111); fall back to asking for it by name.
            Err(_) => {
                let rj = dir.join("run.json");
                rj.is_file().then_some(rj)
            }
        };
        let Some(rj) = run_json else { continue };
        let Ok(bytes) = std::fs::read(&rj) else { continue };
        // The cheap header decides whether this record is worth materializing.
        // A header that does not read could not have parsed as a full
        // `RunRecord` either, so treating it as not-admitted drops exactly
        // what the full parse dropped.
        let admitted =
            !gate.is_selective() || run_header(&bytes).is_some_and(|h| gate.admits(&h));
        if admitted {
            #[cfg(test)]
            FULL_PARSES.with(|n| n.set(n.get() + 1));
            if let Ok(rec) = serde_json::from_slice::<RunRecord>(&bytes) {
                out.push((dir, rec));
            }
        }
    }
    out
}

/// Sort one directory listing into "descend into this" (pushed onto `stack`)
/// and "this is the leaf's `run.json`" (returned), reading each entry's type
/// from the listing itself.
///
/// `DirEntry::file_type` answers from the directory entry on every filesystem
/// camdl targets (APFS, ext4, XFS all populate `d_type`), where
/// `Path::is_dir`/`is_file` cost a `stat` per entry — 274k of them on the
/// synthetic 40k-leaf store, and the walk's dominant term.
///
/// A symlink is the exception: `file_type` reports the *link*, while
/// `Path::is_dir`/`is_file` report the target, and following it is the
/// behaviour a store with a symlinked subtree relies on. So a symlinked entry
/// — and only a symlinked entry — is still resolved with a `stat`.
fn scan_dir(entries: std::fs::ReadDir, stack: &mut Vec<PathBuf>) -> Option<PathBuf> {
    let mut run_json = None;
    for e in entries.flatten() {
        let Ok(file_type) = e.file_type() else { continue };
        let name = e.file_name();
        let is_run_json = name == "run.json";
        let (is_dir, is_file) = if file_type.is_symlink() {
            let p = e.path();
            (p.is_dir(), is_run_json && p.is_file())
        } else {
            (file_type.is_dir(), is_run_json && file_type.is_file())
        };
        if is_dir {
            if name.to_string_lossy().starts_with('.') {
                continue; // .staging / .quarantine
            }
            stack.push(e.path());
        } else if is_file {
            run_json = Some(e.path());
        }
    }
    run_json
}

/// Every `RunRecord` under `subtree`, of every kind — the unfiltered walk.
pub fn walk_records(subtree: &Path) -> Vec<(PathBuf, RunRecord)> {
    walk_gated(subtree, &LeafGate::ANY)
}

/// Every leaf of `kind` under that kind's store partition (`sims/`, `fits/`,
/// …), optionally skipping `members`. One walk behind all six per-kind
/// accessors below, so the partition dir and the kind can never disagree —
/// both come from [`ArtifactKind::store_dir`].
fn walk_kind(
    root: &Path,
    kind: ArtifactKind,
    members: Option<&HashSet<ContentHash>>,
) -> Vec<Leaf> {
    let gate = LeafGate { kind: Some(kind), members };
    walk_gated(&root.join(kind.store_dir()), &gate)
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
    walk_kind(root, ArtifactKind::Sim, None)
}

/// All `sims/` leaves of kind `Sim` EXCEPT those whose `run_id` is in
/// `members` — the per-cell leaves an already-discovered `SimEnsemble` row
/// represents, which `camdl list` must not print a second time.
///
/// Separate from [`walk_sim_leaves`] because the exclusion has to happen
/// *inside* the walk: on the store in gh#699, 461,282 of 550,647 sim leaves
/// are ensemble members, so discovering them only to drop them afterwards is
/// the whole cost of the command. A member is recognized from its `run_id`
/// alone, which [`run_header`] reads without parsing the record.
pub fn walk_sim_leaves_excluding(root: &Path, members: &HashSet<ContentHash>) -> Vec<Leaf> {
    walk_kind(root, ArtifactKind::Sim, Some(members))
}

/// New-format sims whose `run_id` hex matches `prefix` (for `show`/`cat`
/// prefix resolution; combined with the legacy `run.hash` matches in
/// `browse::resolve_any` so a user can address any run during M2→M3).
pub fn resolve_sim_prefix(root: &Path, prefix: &str) -> Vec<Leaf> {
    resolve_prefix_indexed(root, ArtifactKind::Sim, prefix, walk_sim_leaves)
}

/// All `fits/` leaves of kind `FitStage` (new-format fit-stage runs, M3.2).
pub fn walk_fit_leaves(root: &Path) -> Vec<Leaf> {
    walk_kind(root, ArtifactKind::FitStage, None)
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
    walk_kind(root, ArtifactKind::ProfilePoint, None)
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
    walk_kind(root, ArtifactKind::Pfilter, None)
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
    walk_kind(root, ArtifactKind::Survey, None)
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
    walk_kind(root, ArtifactKind::SimEnsemble, None)
}

/// New-format ensembles whose `run_id` hex matches `prefix` (for `show`/`cat`
/// prefix resolution alongside `resolve_sim_prefix`).
pub fn resolve_sim_ensemble_prefix(root: &Path, prefix: &str) -> Vec<Leaf> {
    resolve_prefix_indexed(root, ArtifactKind::SimEnsemble, prefix, walk_sim_ensemble_leaves)
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    /// Zero this thread's parse counter before the walk being measured.
    fn measuring() {
        FULL_PARSES.with(|n| n.set(0));
    }

    fn full_parses() -> usize {
        FULL_PARSES.with(|n| n.get())
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
    fn ensemble_members_are_skipped_without_a_full_parse() {
        let tmp = tempfile::tempdir().unwrap();
        let root = tmp.path();
        let (members, free) = ensemble_heavy_store(root, 40, 5);

        let member_ids: HashSet<ContentHash> =
            members.iter().map(|m| ContentHash::from_hex(m).unwrap()).collect();

        measuring();
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

    /// `--kind sim` is the explicit request for the per-cell level, so the
    /// unexcluded walk must still return the member leaves. The exclusion is
    /// opt-in, never a property of the store.
    #[test]
    fn the_unexcluded_walk_still_returns_member_leaves() {
        let tmp = tempfile::tempdir().unwrap();
        let root = tmp.path();
        let (members, free) = ensemble_heavy_store(root, 6, 2);

        let mut got: Vec<String> =
            walk_sim_leaves(root).iter().map(|l| l.run_id_hex()).collect();
        got.sort();
        let mut want: Vec<String> = members.into_iter().chain(free).collect();
        want.sort();
        assert_eq!(got, want);
    }

    /// The header read stops at `run_id`: a record whose bytes AFTER `run_id`
    /// are unparseable still yields a header, while the full `RunRecord` parse
    /// of the same bytes fails. That gap is what the walk is buying — a
    /// `serde_json::from_slice::<RunHeader>` would tokenize the whole document
    /// and fail here too, so this test fails if the early exit is lost.
    #[test]
    fn run_header_stops_at_run_id() {
        let bytes = format!(
            r#"{{"format_version":1,"kind":"sim","run_id":"{}","levels":[ NOT JSON"#,
            id("abcd")
        );
        let hdr = run_header(bytes.as_bytes()).expect("header reads before the garbage");
        assert_eq!(hdr.kind, ArtifactKind::Sim);
        assert_eq!(hdr.run_id.to_hex(), id("abcd"));
        assert!(
            serde_json::from_slice::<RunRecord>(bytes.as_bytes()).is_err(),
            "the full record parse must fail on the same bytes"
        );
    }

    /// The walk follows symlinks, both to a subtree and to a `run.json`.
    /// Reading entry types out of the directory listing (`DirEntry::file_type`)
    /// instead of `stat`ing each path is what makes the walk cheap, but it
    /// reports the *link*, not the target — so a symlinked subtree would
    /// silently stop being walked, and a symlinked `run.json` would stop being
    /// a leaf. Both are still resolved.
    #[test]
    #[cfg(unix)]
    fn symlinked_subtrees_and_records_are_still_walked() {
        let tmp = tempfile::tempdir().unwrap();
        let root = tmp.path();

        // A leaf reached only through a symlinked directory.
        let elsewhere = tmp.path().join("elsewhere");
        plant_sim(&elsewhere, "m-1111/cfg-2222", &id("beef"));
        std::fs::create_dir_all(root.join("sims")).unwrap();
        std::os::unix::fs::symlink(
            elsewhere.join("sims").join("m-1111"),
            root.join("sims").join("linked-3333"),
        )
        .unwrap();

        // A leaf whose `run.json` is itself a symlink to a record elsewhere.
        let donor = plant_sim(&elsewhere, "donor-4444/cfg-5555", &id("cafe"));
        let via_link = root.join("sims").join("via-link-6666");
        std::fs::create_dir_all(&via_link).unwrap();
        std::os::unix::fs::symlink(donor.join("run.json"), via_link.join("run.json")).unwrap();

        let mut got: Vec<String> =
            walk_sim_leaves(root).iter().map(|l| l.run_id_hex()).collect();
        got.sort();
        assert_eq!(got, vec![id("beef"), id("cafe")]);
    }

    /// Anything the header read rejects, the full parse rejects too — the
    /// property that makes gating on the header safe. A `run.json` with no
    /// `run_id` is dropped by both.
    #[test]
    fn a_record_without_identity_is_dropped_by_both_reads() {
        let bytes = br#"{"format_version":1,"kind":"sim","status":"completed"}"#;
        assert!(run_header(bytes).is_none());
        assert!(serde_json::from_slice::<RunRecord>(bytes).is_err());
    }
}
