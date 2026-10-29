//! Fit handles: resolve a user-typed fit reference to the run or problem
//! folder it names.
//!
//! The modeler loop is `fit → {summarize, predict, scenario, compare}`,
//! referencing the fit each time. A fit is referenced four ways, and resolution
//! is the fallible boundary: a bare hex string and a relative path are
//! genuinely ambiguous, so we classify by cheap syntactic priority
//! (`@name` → label; `*.toml` → config; an existing dir → run dir; else →
//! hash prefix) and let resolution fail with a typed [`ResolveError`] — never a
//! silent guess. Ambiguity is a typed, git-style listed outcome, not a
//! heuristic pick.
//!
//! This subsumes the per-verb `resolve_segment` (which handled only RunDir /
//! Config) and adds `@label` (the one run whose `run.json` carries the label,
//! gh#896) and hash-prefix (a run's `run_id`, else the fit-level hash — the
//! `fit_id` that `fit table` and `camdl list` display).
//!
//! Phase 1b of
//! `docs/dev/proposals/2026-06-27-sealed-fit-packets-handles-and-override-algebra.md`.

use crate::fit::config_v2::FitConfig;
use std::path::{Path, PathBuf};

/// A parsed fit reference, before resolution. Classification is syntactic and
/// total; the I/O (and the ambiguity) lives in resolution.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FitRef {
    /// `@name` — the leading `@` sigil is the label marker (stripped here);
    /// names the one run labelled `name`.
    Label(String),
    /// A `fit.toml` config → its unique run (or an ambiguity list).
    Config(PathBuf),
    /// An existing fit results (segment) directory.
    RunDir(PathBuf),
    /// A fit-level hash prefix (what `fit table` / `list` display as `fit_id`).
    HashPrefix(String),
}

impl FitRef {
    /// Classify a raw handle by syntactic priority: `@` → Label; `*.toml` →
    /// Config; an existing directory → RunDir; else → HashPrefix. The order is
    /// the contract — a `fit.toml` that also happens to name an existing
    /// directory is still a Config (the `.toml` suffix wins), and anything left
    /// is treated as a hex prefix (which resolution validates).
    pub fn classify(s: &str) -> FitRef {
        if let Some(name) = s.strip_prefix('@') {
            return FitRef::Label(name.to_string());
        }
        if Path::new(s).extension().map(|e| e.eq_ignore_ascii_case("toml")).unwrap_or(false) {
            return FitRef::Config(PathBuf::from(s));
        }
        if Path::new(s).is_dir() {
            return FitRef::RunDir(PathBuf::from(s));
        }
        FitRef::HashPrefix(s.to_string())
    }
}

/// Typed resolution failure — surfaced verbatim; ambiguity is listed git-style.
#[derive(Debug)]
pub enum ResolveError {
    /// No fit matched the handle.
    NotFound(String),
    /// The handle matched more than one fit — the candidates are listed so the
    /// user can disambiguate (a run directory, or a longer hash prefix).
    Ambiguous { handle: String, candidates: Vec<PathBuf> },
    /// The handle names several runs where one is needed (gh#896). Each run is
    /// listed with what tells it apart, and the message states the axes on
    /// which the runs actually differ — computed from their records, never
    /// assumed.
    AmbiguousRuns { handle: String, runs: Vec<crate::cas_read::Leaf> },
}

impl std::fmt::Display for ResolveError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ResolveError::NotFound(msg) => write!(f, "{msg}"),
            ResolveError::Ambiguous { handle, candidates } => {
                writeln!(f, "{} resolves to {} fits:", handle, candidates.len())?;
                for c in candidates {
                    writeln!(f, "    {}", c.display())?;
                }
                write!(
                    f,
                    "  Pass a run directory or a longer hash prefix to disambiguate."
                )
            }
            ResolveError::AmbiguousRuns { handle, runs } => {
                writeln!(
                    f,
                    "{handle} names {} runs, which differ in {}:",
                    runs.len(),
                    differing_axes(runs)
                )?;
                for r in runs {
                    writeln!(f, "    {}", run_line(r))?;
                }
                write!(f, "  Name one run by its leaf directory or a run-id prefix.")
            }
        }
    }
}

/// The level of a fit leaf's identity named `name`, as `(label, hash hex)`.
fn level<'a>(r: &'a crate::cas_read::Leaf, name: &str) -> Option<(&'a str, String)> {
    r.record
        .levels
        .iter()
        .find(|l| l.name == name)
        .map(|l| (l.label.as_str(), l.hash.to_hex()))
}

/// What sets a set of runs of one handle apart, read from their identity
/// levels and build: `[method]` settings, fit seed, the problem (model, data,
/// parameters), or the camdl build — whichever actually vary.
fn differing_axes(runs: &[crate::cas_read::Leaf]) -> String {
    let varies = |key: &dyn Fn(&crate::cas_read::Leaf) -> Option<String>| {
        let first = runs.first().map(key);
        runs.iter().any(|r| Some(key(r)) != first)
    };
    let mut axes: Vec<&str> = Vec::new();
    if varies(&|r| level(r, "fit").map(|l| l.1)) {
        axes.push("the problem (model, data or parameters)");
    }
    if varies(&|r| level(r, "method").map(|l| l.1)) {
        axes.push("`[method]` settings or `--starts`");
    }
    if varies(&|r| level(r, "seed").map(|l| l.1)) {
        axes.push("fit seed");
    }
    if varies(&|r| Some(r.record.engine_version.clone())) {
        axes.push("the camdl build that ran them");
    }
    match axes.as_slice() {
        [] => "nothing their records show".to_string(),
        [one] => one.to_string(),
        [init @ .., last] => format!("{} and {last}", init.join(", ")),
    }
}

/// One listed run: run id, method level, seed, chains, when, label, where.
fn run_line(r: &crate::cas_read::Leaf) -> String {
    let id = r.run_id_hex();
    let method = level(r, "method")
        .map(|(label, hash)| format!("{label}-{}", &hash[..8.min(hash.len())]))
        .unwrap_or_default();
    let chains = r
        .record
        .inputs
        .get("n_chains")
        .and_then(|v| v.as_u64())
        .map(|n| format!("{n} chains"))
        .unwrap_or_default();
    let when = r.record.provenance.created_at.clone().unwrap_or_default();
    let label = r
        .record
        .provenance
        .label
        .as_deref()
        .map(|l| format!("@{l}"))
        .unwrap_or_default();
    let parts = [
        format!("run {}", &id[..12.min(id.len())]),
        method,
        format!("seed {}", r.seed()),
        chains,
        when,
        label,
        r.dir.display().to_string(),
    ];
    parts.into_iter().filter(|p| !p.is_empty()).collect::<Vec<_>>().join("  ")
}

/// One stored fit run, named by a handle: a method leaf
/// (`fits/<stem>-<h8>/<algorithm>-<h8>/seed_<n>-<h8>/`) and the identity its
/// `run.json` records.
#[derive(Debug, Clone)]
pub struct LeafRef {
    pub dir: PathBuf,
}

impl LeafRef {
    fn of(leaf: crate::cas_read::Leaf) -> LeafRef {
        LeafRef { dir: leaf.dir }
    }
}

/// What a handle names once resolved: one run, or a problem folder that holds
/// one or more runs.
#[derive(Debug, Clone)]
pub enum FitTarget {
    Leaf(LeafRef),
    Segment(PathBuf),
}

/// The completed fit leaves directly under a problem folder, in directory
/// order.
fn completed_leaves(segment: &Path) -> Vec<crate::cas_read::Leaf> {
    let mut leaves: Vec<crate::cas_read::Leaf> = crate::cas_read::walk_records(segment)
        .into_iter()
        .filter(|(_, r)| {
            r.kind == runid::ArtifactKind::FitStage && r.status == runid::RunStatus::Completed
        })
        .map(|(dir, record)| crate::cas_read::Leaf { dir, record })
        .collect();
    leaves.sort_by(|a, b| a.dir.cmp(&b.dir));
    leaves
}

/// The fit leaf at `dir`, when `dir` is one.
fn leaf_at(dir: &Path) -> Option<crate::cas_read::Leaf> {
    let bytes = std::fs::read(dir.join("run.json")).ok()?;
    let record: runid::RunRecord = serde_json::from_slice(&bytes).ok()?;
    (record.kind == runid::ArtifactKind::FitStage)
        .then(|| crate::cas_read::Leaf { dir: dir.to_path_buf(), record })
}

/// Resolve a handle to what it names: a leaf directory, a run-id prefix or
/// `@label` names one run; a problem folder or a fit-id prefix names that
/// folder; a `fit.toml` names the folder whose archived config it matches.
pub fn resolve_target(s: &str) -> Result<FitTarget, ResolveError> {
    match FitRef::classify(s) {
        FitRef::RunDir(dir) => match leaf_at(&dir) {
            Some(leaf) => Ok(FitTarget::Leaf(LeafRef::of(leaf))),
            None if dir.is_dir() => Ok(FitTarget::Segment(dir)),
            None => Err(ResolveError::NotFound(format!("no such fit directory: {}", dir.display()))),
        },
        FitRef::HashPrefix(prefix) => {
            let root = crate::run_paths::output_root(None, None);
            let mut leaves = crate::cas_read::resolve_fit_prefix(&root, &prefix);
            match leaves.len() {
                0 => {
                    let seg = resolve_by(&prefix, |seg| {
                        crate::fit::fit_view::FitView::read(seg)
                            .map(|v| v.fit_hash.starts_with(&prefix))
                            .unwrap_or(false)
                    })?;
                    Ok(FitTarget::Segment(seg))
                }
                1 => Ok(FitTarget::Leaf(LeafRef::of(leaves.remove(0)))),
                _ => {
                    leaves.sort_by(|a, b| a.dir.cmp(&b.dir));
                    Err(ResolveError::AmbiguousRuns { handle: s.to_string(), runs: leaves })
                }
            }
        }
        FitRef::Label(name) => {
            let root = crate::run_paths::output_root(None, None);
            Ok(FitTarget::Leaf(resolve_label(&root, &name)?))
        }
        FitRef::Config(path) => Ok(FitTarget::Segment(resolve_config(&path)?.0)),
    }
}

/// Every fit run under `root` carrying `label`, whatever its status, in
/// directory order. At most one non-failed run holds a label: `fit run
/// --label` and `camdl label` refuse to give a held label to another run.
pub fn runs_labelled(root: &Path, label: &str) -> Vec<crate::cas_read::Leaf> {
    let mut runs: Vec<crate::cas_read::Leaf> = crate::cas_read::walk_fit_leaves(root)
        .into_iter()
        .filter(|l| l.record.provenance.label.as_deref() == Some(label))
        .collect();
    runs.sort_by(|a, b| a.dir.cmp(&b.dir));
    runs
}

/// `@name` → the one completed run under `root` labelled `name`.
pub fn resolve_label(root: &Path, name: &str) -> Result<LeafRef, ResolveError> {
    let mut runs: Vec<crate::cas_read::Leaf> = runs_labelled(root, name)
        .into_iter()
        .filter(|l| l.record.status == runid::RunStatus::Completed)
        .collect();
    match runs.len() {
        0 => Err(ResolveError::NotFound(format!(
            "no completed fit run labelled @{name} under {}",
            root.join("fits").display()
        ))),
        1 => Ok(LeafRef::of(runs.remove(0))),
        _ => Err(ResolveError::AmbiguousRuns { handle: format!("@{name}"), runs }),
    }
}

/// Resolve a handle to the one run it names. A problem folder names a run
/// only while it holds one; holding several, it is refused with the runs
/// listed and what sets them apart.
pub fn resolve_leaf(s: &str) -> Result<LeafRef, ResolveError> {
    match resolve_target(s)? {
        FitTarget::Leaf(leaf) => Ok(leaf),
        FitTarget::Segment(segment) => {
            let mut leaves = completed_leaves(&segment);
            match leaves.len() {
                0 => Err(ResolveError::NotFound(format!(
                    "{} holds no completed fit run",
                    segment.display()
                ))),
                1 => Ok(LeafRef::of(leaves.remove(0))),
                _ => Err(ResolveError::AmbiguousRuns { handle: s.to_string(), runs: leaves }),
            }
        }
    }
}

/// Refuse a problem folder in which one method has several runs, listing
/// them: a reader that shows one run per method would otherwise choose
/// among them by directory order.
pub fn refuse_several_runs_of_one_method(handle: &str, segment: &Path) -> Result<(), ResolveError> {
    let leaves = completed_leaves(segment);
    let method = |l: &crate::cas_read::Leaf| level(l, "method").map(|m| m.0.to_string());
    let repeated: Vec<crate::cas_read::Leaf> = leaves
        .iter()
        .filter(|l| leaves.iter().filter(|o| method(o) == method(l)).count() > 1)
        .cloned()
        .collect();
    if repeated.is_empty() {
        Ok(())
    } else {
        Err(ResolveError::AmbiguousRuns { handle: handle.to_string(), runs: repeated })
    }
}

/// A resolved fit: its problem folder, the run the handle named (when it
/// named one), and the loaded config.
///
/// The proposal's full `SealedFit` (model IR + posterior ensemble + diagnostics)
/// is Phase 4; the verbs today consume the segment + config, plus the leaf a
/// run handle names (gh#896).
pub struct ResolvedFit {
    /// The problem folder: archived model IR, `fit.meta.json`, and where
    /// `fit predict` writes.
    pub segment: PathBuf,
    /// The run a leaf directory, run-id prefix or `@label` named; `None` for a
    /// problem-folder handle.
    pub leaf: Option<PathBuf>,
    pub config: FitConfig,
}

impl ResolvedFit {
    /// Where a reader takes the run from: the named leaf, else the problem
    /// folder (whose readers choose among its runs).
    pub fn run_dir(&self) -> &Path {
        self.leaf.as_deref().unwrap_or(&self.segment)
    }
}

/// Resolve a raw fit handle to its segment, the run it names, and the config
/// that governs its data.
pub fn resolve_fit(s: &str) -> Result<ResolvedFit, ResolveError> {
    let (target, live_config) = match FitRef::classify(s) {
        FitRef::Config(path) => {
            let (seg, cfg) = resolve_config(&path)?;
            (FitTarget::Segment(seg), Some(cfg))
        }
        _ => (resolve_target(s)?, None),
    };
    let (segment, leaf) = match target {
        FitTarget::Segment(seg) => (seg, None),
        FitTarget::Leaf(l) => {
            let seg = crate::fit::fit_view::leaf_segment(&l.dir).ok_or_else(|| {
                ResolveError::NotFound(format!("{} is not inside a fit's directory", l.dir.display()))
            })?;
            (seg, Some(l.dir))
        }
    };
    let config = match live_config {
        // The Config (`fit.toml`) handle named the live config — its relative
        // data/model paths already resolve against the config's own directory.
        Some(c) => c,
        // A segment handle (run-dir / @label / hash) has no caller-supplied
        // config; recover one whose data paths resolve correctly.
        None => load_config_for_segment(&segment).map_err(ResolveError::NotFound)?,
    };
    Ok(ResolvedFit { segment, leaf, config })
}

/// Recover the config for a segment reached by run-dir / `@label` / hash.
///
/// The archived `fit.toml.original` is the authoritative record of the config
/// that produced this fit — the file the user still has on disk may have drifted
/// since. So the archive is what we read, always; there is no "prefer the live
/// original" branch, because a live original that still matched would parse to
/// the same config anyway.
///
/// What the archive cannot carry is its own *location*. Data archival is not yet
/// part of the sealed packet (Phase 1 archives only the model IR), so the data
/// still lives where the config says — at paths written relative to the
/// **original fit.toml directory**, not to the segment. Anchoring them at the
/// archive's own location turned `../data/streams/x.tsv` into
/// `results/fits/data/streams/x.tsv` and broke `compare` for every project whose
/// configs use relative paths (gh#652). The sidecar records `fit_toml_path`, so
/// that directory is known even when the file itself is gone or changed, and
/// [`FitConfig::load_anchored_at`] resolves against it.
///
/// A fit run entirely from the CLI has no `fit.toml` at all: `write_fit_sidecar`
/// records an empty `fit_toml_path` and archives nothing. That is a real gap, not
/// a lookup miss, so it gets its own named error rather than a wrong path.
fn load_config_for_segment(segment: &Path) -> Result<FitConfig, String> {
    let side = crate::run_meta::read_fit_sidecar(segment).ok_or_else(|| {
        format!(
            "{} is not a complete fit directory: no fit.meta.json",
            segment.display()
        )
    })?;
    if side.fit_toml_path.is_empty() {
        return Err(format!(
            "{} was produced without a fit.toml (a CLI-only run), so it carries \
             no config to recover: no archived config and no recorded config \
             directory to resolve its data paths against.\n  \
             Pass the fit toml, or re-run the fit from one.",
            segment.display()
        ));
    }
    let archived = segment.join("fit.toml.original");
    FitConfig::load_anchored_at(&archived, Path::new(&side.fit_toml_path)).map_err(|e| {
        format!(
            "{} is a fit directory but its archived config could not be read: {e}\n  \
             (expected {})",
            segment.display(),
            archived.display()
        )
    })
}

/// The canonical identity of a fit.toml's text — what the config MEANS, with
/// comments, whitespace, and float spelling discarded (gh#653). Parsed with the
/// production parser, so a config this cannot parse cannot match a fit either.
fn config_identity(text: &str) -> Result<String, String> {
    let config = FitConfig::from_toml_str(text)?;
    crate::fit::cas::config_identity_hash(&config)
}

/// The Config (`fit.toml`) branch: match the config's MEANING against the
/// archived config of every fit segment under the root it names. Returns the
/// matched segment and the live config (loaded once here).
///
/// The match used to be a SHA-256 over the config's raw bytes, recorded in each
/// sidecar as `fit_toml_hash`. That conflates two questions: which exact bytes
/// produced this fit (provenance — the raw hash answers it, and the sidecar
/// still records it) and whether a config means the same thing (identity — the
/// raw hash gets it wrong, because reflowing a comment, reordering two
/// top-level tables, or writing `1.00` for `1.0` all change the bytes and none
/// change the meaning). A comment reflow across a set of national configs
/// orphaned hours of completed sampling: the fits were in the store, and
/// `compare` reported "no completed fit found" (gh#653).
///
/// Retroactive by construction: the identity is computed from each segment's
/// archived `fit.toml.original` at lookup time, so fits already on disk resolve
/// under this rule without re-stamping, and old and new fits take one code path.
/// A segment whose archive is missing or unparseable therefore cannot be
/// matched — so those segments are named in the not-found message rather than
/// silently passed over.
fn resolve_config(path: &Path) -> Result<(PathBuf, FitConfig), ResolveError> {
    let text = std::fs::read_to_string(path)
        .map_err(|e| ResolveError::NotFound(format!("cannot read fit config {}: {e}", path.display())))?;
    let wanted = config_identity(&text)
        .map_err(|e| ResolveError::NotFound(format!("cannot load fit config {}: {e}", path.display())))?;
    let config = FitConfig::load(&path.to_string_lossy())
        .map_err(|e| ResolveError::NotFound(format!("cannot load fit config {}: {e}", path.display())))?;
    let cas_root = crate::run_paths::output_root(None, config.problem.output_dir.as_deref());
    let fits_dir = cas_root.join("fits");
    let mut matches: Vec<PathBuf> = Vec::new();
    let mut unreadable: Vec<String> = Vec::new();
    if let Ok(entries) = std::fs::read_dir(&fits_dir) {
        for entry in entries.flatten() {
            let seg = entry.path();
            if !seg.is_dir() || crate::run_meta::read_fit_sidecar(&seg).is_none() {
                continue;
            }
            let archived = seg.join("fit.toml.original");
            let outcome = std::fs::read_to_string(&archived)
                .map_err(|e| e.to_string())
                .and_then(|t| config_identity(&t));
            match outcome {
                Ok(h) if h == wanted => matches.push(seg),
                Ok(_) => {}
                Err(e) => unreadable.push(format!("    {}: {e}", seg.display())),
            }
        }
    }
    matches.sort();
    match matches.len() {
        0 => {
            let mut msg = format!(
                "no completed fit found for {} under {}.\n  Run `camdl fit run {}` first.",
                path.display(),
                fits_dir.display(),
                path.display()
            );
            if !unreadable.is_empty() {
                unreadable.sort();
                msg.push_str(&format!(
                    "\n  {} fit director{} could not be checked (no readable \
                     fit.toml.original):\n{}",
                    unreadable.len(),
                    if unreadable.len() == 1 { "y" } else { "ies" },
                    unreadable.join("\n")
                ));
            }
            Err(ResolveError::NotFound(msg))
        }
        1 => Ok((matches.remove(0), config)),
        _ => Err(ResolveError::Ambiguous { handle: path.display().to_string(), candidates: matches }),
    }
}

/// Scan `fits/` segments under the default output root and return the unique
/// one matching `pred`. Used by the fit-id prefix branch, which has
/// no config to name a custom root — they resolve against the conventional
/// `results/` root (or `CAMDL_OUTPUT_DIR`); a fit under a custom `output_dir`
/// is reached by its run dir or `fit.toml`.
fn resolve_by(
    handle: &str,
    pred: impl Fn(&Path) -> bool,
) -> Result<PathBuf, ResolveError> {
    let fits_dir = crate::run_paths::output_root(None, None).join("fits");
    let mut matches: Vec<PathBuf> = Vec::new();
    if let Ok(entries) = std::fs::read_dir(&fits_dir) {
        for entry in entries.flatten() {
            let seg = entry.path();
            if seg.is_dir() && pred(&seg) {
                matches.push(seg);
            }
        }
    }
    matches.sort();
    match matches.len() {
        0 => Err(ResolveError::NotFound(format!(
            "no fit found for {handle} under {}",
            fits_dir.display()
        ))),
        1 => Ok(matches.remove(0)),
        _ => Err(ResolveError::Ambiguous { handle: handle.to_string(), candidates: matches }),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn classify_prioritizes_label_sigil() {
        assert_eq!(FitRef::classify("@jigawa-baseline"), FitRef::Label("jigawa-baseline".into()));
    }

    #[test]
    fn classify_recognizes_toml_config() {
        assert_eq!(FitRef::classify("fit.toml"), FitRef::Config(PathBuf::from("fit.toml")));
        assert_eq!(FitRef::classify("a/b/run.TOML"), FitRef::Config(PathBuf::from("a/b/run.TOML")));
    }

    #[test]
    fn classify_falls_through_to_hash_prefix() {
        // A hex string that is neither a label, a .toml, nor an existing dir.
        assert_eq!(FitRef::classify("b4aa952d"), FitRef::HashPrefix("b4aa952d".into()));
    }

    #[test]
    fn classify_existing_dir_is_run_dir() {
        let tmp = std::env::temp_dir().join(format!("camdl_handle_classify_{}", std::process::id()));
        let _ = std::fs::create_dir_all(&tmp);
        let s = tmp.to_string_lossy().into_owned();
        assert_eq!(FitRef::classify(&s), FitRef::RunDir(tmp.clone()));
        let _ = std::fs::remove_dir_all(&tmp);
    }

    /// A portable fit.toml: every path relative, reaching UP out of its own
    /// directory (`fits/x.toml` → `../data/...`), which is what a project that
    /// keeps its configs in one place and its data in another looks like.
    const PORTABLE_TOML: &str = r#"
[model]
camdl = "../models/sir.camdl"

[data.observations]
cases = "../data/streams/cases.tsv"

[estimate]
beta = { bounds = [0.01, 2.0] }

[fixed]
gamma = 0.2

[method]
algorithm = "if2"
backend = "chain_binomial"
chains = 1
particles = 10
iterations = 2
cooling = 0.7
"#;

    /// `<tmp>/{fits,data/streams,results/fits/seg}` with the portable config at
    /// `fits/a.toml` and a segment whose sidecar records it. Returns
    /// `(tmp, toml_path, segment)`.
    fn portable_project(slug: &str) -> (PathBuf, PathBuf, PathBuf) {
        let tmp = std::env::temp_dir()
            .join(format!("camdl_handle_{slug}_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&tmp);
        let toml_dir = tmp.join("fits");
        let seg = tmp.join("results/fits/sir-0000beef");
        std::fs::create_dir_all(&toml_dir).unwrap();
        std::fs::create_dir_all(tmp.join("data/streams")).unwrap();
        std::fs::create_dir_all(&seg).unwrap();
        std::fs::write(tmp.join("data/streams/cases.tsv"), "time\tcases\n1\t2\n").unwrap();
        let toml_path = toml_dir.join("a.toml");
        std::fs::write(&toml_path, PORTABLE_TOML).unwrap();
        (tmp, toml_path, seg)
    }

    #[test]
    fn archived_config_anchors_at_the_recorded_toml_dir() {
        // gh#652: the archived config's `../data/streams/cases.tsv` must resolve
        // against the fit.toml's ORIGINAL directory, not against the segment the
        // archive happens to sit in. Anchored at the segment it would name
        // `results/fits/data/streams/cases.tsv`, which never existed — the whole
        // reason `compare <run-dir>` could not derive a prequential.
        let (tmp, toml_path, seg) = portable_project("anchor");
        let side = crate::run_meta::FitSidecar {
            fit_toml_path: toml_path.to_string_lossy().into_owned(),
            ..Default::default()
        };
        crate::run_meta::write_fit_sidecar(&seg, &toml_path, &side).unwrap();
        // The live original is GONE — the archive is all that is left, which is
        // exactly the case the segment-anchored resolution got wrong.
        std::fs::remove_file(&toml_path).unwrap();

        let cfg = resolve_fit(&seg.to_string_lossy())
            .unwrap_or_else(|e| panic!("resolve_fit on the segment: {e}"))
            .config;
        let data = Path::new(&cfg.problem.data.as_ref().unwrap().observations["cases"]);
        assert!(
            data.is_file(),
            "the archived config's data path must resolve to the real file; got {}",
            data.display()
        );
        assert_eq!(
            std::fs::canonicalize(data).unwrap(),
            std::fs::canonicalize(tmp.join("data/streams/cases.tsv")).unwrap()
        );
        // The model path anchors the same way.
        assert_eq!(
            Path::new(&cfg.problem.model.camdl),
            tmp.join("fits/../models/sir.camdl"),
        );
        let _ = std::fs::remove_dir_all(&tmp);
    }

    #[test]
    fn cli_only_fit_has_a_named_error_not_a_wrong_path() {
        // A fit run with no fit.toml archives no config and records no config
        // directory (`fit_toml_path` is ""), so there is nothing to anchor. That
        // is a named gap, not a silent miss or a plausible-looking wrong path.
        let (tmp, _toml, seg) = portable_project("clionly");
        crate::run_meta::write_fit_sidecar(
            &seg,
            Path::new(""),
            &crate::run_meta::FitSidecar::default(),
        )
        .unwrap();
        let err = resolve_fit(&seg.to_string_lossy())
            .err()
            .expect("a CLI-only fit cannot yield a config")
            .to_string();
        assert!(
            err.contains("produced without a fit.toml"),
            "the error names the cause; got: {err}"
        );
        let _ = std::fs::remove_dir_all(&tmp);
    }

    /// Stand up `<tmp>/fits/a.toml` plus a segment under `<tmp>/results/fits/`
    /// whose archived config is `archived`, and resolve `live` as a `fit.toml`
    /// handle against it. Returns the resolution outcome and the segment.
    fn resolve_live_against_archive(
        slug: &str,
        archived: &str,
        live: &str,
    ) -> (Result<PathBuf, ResolveError>, PathBuf, PathBuf) {
        let (tmp, toml_path, seg) = portable_project(slug);
        std::fs::write(&toml_path, archived).unwrap();
        let side = crate::run_meta::FitSidecar {
            fit_toml_path: toml_path.to_string_lossy().into_owned(),
            fit_toml_hash: crate::hashing::sha256_hex(archived.as_bytes()),
            ..Default::default()
        };
        // Archives `fit.toml.original` from the config on disk, as a fit does.
        crate::run_meta::write_fit_sidecar(&seg, &toml_path, &side).unwrap();
        std::fs::write(&toml_path, live).unwrap();
        (resolve_config(&toml_path).map(|r| r.0), seg, tmp)
    }

    /// The archived config, written portably (`output_dir` points at the
    /// results tree the segment lives in, so the lookup scans it).
    const ARCHIVED: &str = r#"# fit the outbreak
output_dir = "../results"

[model]
camdl = "../models/sir.camdl"

[data.observations]
cases = "../data/streams/cases.tsv"

[estimate]
beta = { bounds = [0.01, 2.0] }

[fixed]
gamma = 0.2

[method]
algorithm = "if2"
backend = "chain_binomial"
chains = 1
particles = 10
iterations = 2
cooling = 0.7
"#;

    #[test]
    fn a_meaning_preserving_edit_still_resolves() {
        // gh#653: the same config, reflowed. The comment is rewrapped, the
        // top-level tables are reordered, `0.2` is spelled `0.20`, and the
        // whitespace differs — every byte-level difference, no semantic one.
        // Under a raw-bytes hash this orphaned hours of completed sampling.
        let reflowed = r#"# fit
# the outbreak
output_dir = "../results"

[estimate]
beta = { bounds = [0.01, 2.0] }

[fixed]
gamma = 0.20

[data.observations]
cases = "../data/streams/cases.tsv"

[model]
camdl   = "../models/sir.camdl"

[method]
algorithm  = "if2"
backend    = "chain_binomial"
chains     = 1
particles  = 10
iterations = 2
cooling    = 0.70
"#;
        let (got, seg, tmp) = resolve_live_against_archive("reflow", ARCHIVED, reflowed);
        let got = got.unwrap_or_else(|e| panic!("a reflowed config must still resolve: {e}"));
        assert_eq!(
            std::fs::canonicalize(&got).unwrap(),
            std::fs::canonicalize(&seg).unwrap(),
            "resolved {}, expected {}",
            got.display(),
            seg.display()
        );
        let _ = std::fs::remove_dir_all(&tmp);
    }

    #[test]
    fn a_changed_particle_count_does_not_resolve() {
        // NEGATIVE CONTROL. Canonicalising too aggressively would silently hand
        // back the posterior of a DIFFERENT fit — worse than the bug fixed here.
        // A stage knob is a real difference and must not resolve.
        let changed = ARCHIVED.replace("particles = 10", "particles = 20");
        assert_ne!(changed, ARCHIVED);
        let (got, _seg, tmp) = resolve_live_against_archive("particles", ARCHIVED, &changed);
        let err = got.expect_err("a changed particle count is a different fit").to_string();
        assert!(err.contains("no completed fit found"), "got: {err}");
        let _ = std::fs::remove_dir_all(&tmp);
    }

    #[test]
    fn a_changed_data_path_does_not_resolve() {
        // NEGATIVE CONTROL. Same shape, different input file.
        let changed = ARCHIVED.replace("streams/cases.tsv", "streams/cases_v2.tsv");
        assert_ne!(changed, ARCHIVED);
        let (got, _seg, tmp) = resolve_live_against_archive("datapath", ARCHIVED, &changed);
        let err = got.expect_err("a changed data path is a different fit").to_string();
        assert!(err.contains("no completed fit found"), "got: {err}");
        let _ = std::fs::remove_dir_all(&tmp);
    }

    #[test]
    fn a_segment_with_no_archived_config_is_named_not_skipped() {
        // The identity of a stored fit is computed from its archived config, so
        // a segment without one cannot be matched. Say which segment and why —
        // a silent skip reads as "that fit was never run".
        let (tmp, toml_path, seg) = portable_project("noarchive");
        std::fs::write(&toml_path, ARCHIVED).unwrap();
        let side = crate::run_meta::FitSidecar {
            fit_toml_path: toml_path.to_string_lossy().into_owned(),
            ..Default::default()
        };
        crate::run_meta::write_fit_sidecar(&seg, &toml_path, &side).unwrap();
        std::fs::remove_file(seg.join("fit.toml.original")).unwrap();

        let err = resolve_config(&toml_path).map(|r| r.0)
            .expect_err("no archive, no match")
            .to_string();
        assert!(
            err.contains("could not be checked") && err.contains("sir-0000beef"),
            "the unmatchable segment is named; got: {err}"
        );
        let _ = std::fs::remove_dir_all(&tmp);
    }

    /// A fit leaf record whose `method` and `seed` levels hash to the given
    /// one-character stems, for the ambiguity wording tests.
    fn leaf_with(method: char, seed: char, dir: &str) -> crate::cas_read::Leaf {
        let h = |c: char| c.to_string().repeat(64);
        let record: runid::RunRecord = serde_json::from_str(&format!(
            r#"{{"format_version":1,"kind":"fit_stage","run_id":"{run}","hash_version":1,"ir_version":"0.7","engine_version":"0.1.0+test","levels":[{{"name":"fit","label":"demo","hash":"{fit}","schema_version":1}},{{"name":"method","label":"if2","hash":"{m}","schema_version":1}},{{"name":"seed","label":"seed_1","hash":"{s}","schema_version":1}}],"status":"completed","artifacts":{{}},"inputs":{{"method":"if2","n_chains":2}},"provenance":{{}}}}"#,
            run = format!("{method}{seed}").repeat(32),
            fit = h('0'),
            m = h(method),
            s = h(seed),
        ))
        .unwrap();
        crate::cas_read::Leaf { dir: PathBuf::from(dir), record }
    }

    #[test]
    fn ambiguity_names_only_the_axes_that_differ() {
        let method_only = ResolveError::AmbiguousRuns {
            handle: "@x".into(),
            runs: vec![leaf_with('a', '1', "fits/a"), leaf_with('b', '1', "fits/b")],
        }
        .to_string();
        assert!(method_only.contains("@x names 2 runs, which differ in `[method]` settings or `--starts`:"),
            "{method_only}");
        assert!(!method_only.contains("seed,") && !method_only.contains("fit seed"), "{method_only}");

        let seed_only = ResolveError::AmbiguousRuns {
            handle: "@x".into(),
            runs: vec![leaf_with('a', '1', "fits/a"), leaf_with('a', '2', "fits/b")],
        }
        .to_string();
        assert!(seed_only.contains("which differ in fit seed:"), "{seed_only}");

        let both = ResolveError::AmbiguousRuns {
            handle: "@x".into(),
            runs: vec![leaf_with('a', '1', "fits/a"), leaf_with('b', '2', "fits/b")],
        }
        .to_string();
        assert!(both.contains("which differ in `[method]` settings or `--starts` and fit seed:"),
            "{both}");
        assert!(both.contains("fits/a") && both.contains("fits/b"), "lists both: {both}");
    }

    #[test]
    fn ambiguous_lists_candidates() {
        let e = ResolveError::Ambiguous {
            handle: "@dup".into(),
            candidates: vec![PathBuf::from("results/fits/a"), PathBuf::from("results/fits/b")],
        };
        let msg = e.to_string();
        assert!(msg.contains("@dup resolves to 2 fits"));
        assert!(msg.contains("results/fits/a") && msg.contains("results/fits/b"));
    }
}
