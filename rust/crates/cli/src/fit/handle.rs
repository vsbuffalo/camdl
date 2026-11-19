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
    /// A `fit.toml` config → the run made from it as it now reads (or an
    /// ambiguity list).
    Config(PathBuf),
    /// An existing directory: a run's leaf, or a problem folder (segment).
    RunDir(PathBuf),
    /// A run's `run_id` prefix, else a fit-level hash prefix (what `fit table`
    /// / `list` display as `fit_id`).
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
    /// The handle names a run whose results are to be read, and the run did
    /// not complete: the message names it and its state
    /// ([`crate::leaf_use::require_completed`]).
    NotCompleted(String),
    /// A `fit.toml` whose runs were all made by another engine — another
    /// camdl build, or this build with other engine switches on: none is the
    /// computation `fit run` of the file would make here, so none is used in
    /// its place. Each is listed with how its engine differs and how to name
    /// it.
    OtherEngines { handle: String, runs: Vec<crate::cas_read::Leaf> },
    /// The handle names a fit folder whose stored config cannot be read as the
    /// fit ran it: no `fit.resolved.toml`, one that does not parse, a data file
    /// gone, or a `[fixed]` other than the one the fit recorded. Unlike
    /// [`Self::NotFound`] (no fit there), a reader that can run without a fit
    /// must not pass over it, since the fit is there and would run otherwise.
    StoredConfig(String),
}

impl std::fmt::Display for ResolveError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ResolveError::NotFound(msg) | ResolveError::NotCompleted(msg) | ResolveError::StoredConfig(msg) => {
                write!(f, "{msg}")
            }
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
            ResolveError::OtherEngines { handle, runs } => {
                writeln!(
                    f,
                    "no completed run of {handle} was made by this camdl build ({}) with the \
                     engine switches set here; {} run(s) of it by another build or with other \
                     engine switches:",
                    crate::version::VERSION_SHORT,
                    runs.len()
                )?;
                for r in runs {
                    writeln!(f, "    {}  {}", build_of(r), run_line(r))?;
                }
                write!(
                    f,
                    "  A run another engine made is not used in place of this one's. Name one \
                     by its leaf directory (the last field of its line), or by its run-id prefix \
                     when the store is ./results; or run `camdl fit run {handle}`."
                )
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
    if varies(&|r| Some(switch_names(&r.record.engine_switches))) {
        axes.push("the engine switches set when they ran");
    }
    match axes.as_slice() {
        [] => "nothing their records show".to_string(),
        [one] => one.to_string(),
        [init @ .., last] => format!("{} and {last}", init.join(", ")),
    }
}

/// The engine switches a record names, by name and value, comma-separated;
/// `no engine switch` for none.
fn switch_names(switches: &[runid::inputs::EngineSwitch]) -> String {
    if switches.is_empty() {
        return "no engine switch".to_string();
    }
    switches
        .iter()
        .map(|s| match &s.value {
            Some(v) => format!("{}={v}", s.name),
            None => s.name.clone(),
        })
        .collect::<Vec<_>>()
        .join(", ")
}

/// Whether `r` was made by the engine `fit run` keys in this process: this
/// build ([`crate::version::ENGINE_ID`]), with the engine switches its
/// identity keys set as they are now ([`crate::fit::fit_engine`]).
/// `run.json`'s `engine_version` is the build alone; the switches are its
/// `engine_switches`.
fn made_by_this_engine(r: &crate::cas_read::Leaf) -> bool {
    let engine = crate::fit::fit_engine();
    r.record.engine_version == engine.build && r.record.engine_switches == engine.switches
}

/// How the engine that made `r` differs from this one, for a run another
/// engine made: for this build, the engine switches it ran with; otherwise
/// `stale: built by X` (another commit), `built from uncommitted changes: X`
/// (the same commit), or, when the readable versions agree, the build
/// configuration (features, profile, toolchain or flags).
fn build_of(r: &crate::cas_read::Leaf) -> String {
    if r.record.engine_version == crate::version::ENGINE_ID {
        return format!("made with other engine switches: {}", switch_names(&r.record.engine_switches));
    }
    let Some(v) = r.record.provenance.camdl_version.as_deref() else {
        return "built by a camdl that recorded no version".to_string();
    };
    match crate::version::build_relation(v, crate::version::VERSION_SHORT) {
        crate::version::BuildRelation::Stale => format!("stale: built by {v}"),
        crate::version::BuildRelation::UncommittedChanges => format!("built from uncommitted changes: {v}"),
        crate::version::BuildRelation::Same => {
            format!("{}: {v}", crate::version::ANOTHER_CONFIGURATION)
        }
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
            r.kind == runid::ArtifactKind::FitMethod && r.status == runid::RunStatus::Completed
        })
        .map(|(dir, record)| crate::cas_read::Leaf { dir, record })
        .collect();
    leaves.sort_by(|a, b| a.dir.cmp(&b.dir));
    leaves
}

/// The fit leaf at `dir`, when `dir` is one; `Ok(None)` when it holds no
/// record or a record of another kind. A record that cannot be read is an
/// error naming it: read as "no leaf here", it would send the directory on
/// as a problem folder, and the reader would be told the folder "holds no
/// completed fit run" about a run whose record is damaged.
fn leaf_at(dir: &Path) -> Result<Option<crate::cas_read::Leaf>, ResolveError> {
    let leaf = crate::leaf_use::read(dir).map_err(ResolveError::NotFound)?;
    Ok(leaf
        .filter(|l| l.record.kind == runid::ArtifactKind::FitMethod)
        .map(|l| crate::cas_read::Leaf { dir: l.dir, record: l.record }))
}

/// Resolve a handle to what it names: a leaf directory, a run-id prefix,
/// `@label` or a `fit.toml` names one run; a problem folder or a fit-id prefix
/// names that folder.
pub fn resolve_target(s: &str) -> Result<FitTarget, ResolveError> {
    match FitRef::classify(s) {
        FitRef::RunDir(dir) => match leaf_at(&dir)? {
            Some(leaf) => Ok(FitTarget::Leaf(LeafRef::of(leaf))),
            None if dir.is_dir() => Ok(FitTarget::Segment(dir)),
            None => Err(ResolveError::NotFound(format!("no such fit directory: {}", dir.display()))),
        },
        FitRef::HashPrefix(prefix) => {
            resolve_hash_prefix(&crate::run_paths::output_root(None, None), &prefix)
        }
        FitRef::Label(name) => {
            let root = crate::run_paths::output_root(None, None);
            Ok(FitTarget::Leaf(resolve_label(&root, &name)?))
        }
        FitRef::Config(path) => Ok(FitTarget::Leaf(resolve_config(&path)?.0)),
    }
}

/// A hex prefix under `root`: a run's `run_id`, or a problem folder's fit id.
/// Both namespaces are searched; a prefix matching in both is refused with
/// every match listed rather than resolved to the run (gh#896), and several
/// matches in one namespace are refused as before.
pub(crate) fn resolve_hash_prefix(root: &Path, prefix: &str) -> Result<FitTarget, ResolveError> {
    let mut leaves = crate::cas_read::resolve_fit_prefix(root, prefix);
    leaves.sort_by(|a, b| a.dir.cmp(&b.dir));
    let fits_dir = root.join("fits");
    let mut folders: Vec<PathBuf> = std::fs::read_dir(&fits_dir)
        .into_iter()
        .flatten()
        .flatten()
        .map(|e| e.path())
        .filter(|seg| {
            seg.is_dir()
                && crate::fit::fit_view::FitView::read(seg)
                    .is_some_and(|v| v.fit_hash.starts_with(prefix))
        })
        .collect();
    folders.sort();
    match (leaves.len(), folders.len()) {
        (0, 0) => Err(ResolveError::NotFound(format!(
            "no fit found for {prefix} under {}",
            fits_dir.display()
        ))),
        (1, 0) => Ok(FitTarget::Leaf(LeafRef::of(leaves.remove(0)))),
        (_, 0) => Err(ResolveError::AmbiguousRuns { handle: prefix.to_string(), runs: leaves }),
        (0, 1) => Ok(FitTarget::Segment(folders.remove(0))),
        _ => Err(ResolveError::Ambiguous {
            handle: prefix.to_string(),
            candidates: leaves.into_iter().map(|l| l.dir).chain(folders).collect(),
        }),
    }
}

/// Every fit run under `root` carrying `label`, whatever its status, in
/// directory order. Several can carry one label; [`label_history`] orders
/// the completed ones, and its first is the label's holder.
pub fn runs_labelled(root: &Path, label: &str) -> Vec<crate::cas_read::Leaf> {
    let mut runs: Vec<crate::cas_read::Leaf> = crate::cas_read::walk_fit_leaves(root)
        .into_iter()
        .filter(|l| l.record.provenance.label.as_deref() == Some(label))
        .collect();
    runs.sort_by(|a, b| a.dir.cmp(&b.dir));
    runs
}

/// The runs that hold or have held `label`: the fit runs under `root` that
/// carry it and whose live state is completed, most recently labelled first
/// (`label_history(..)[0]` is the label's holder, `[n]` is `@label~n`).
/// A run's labelling time is its `label_set_at`, or its `created_at` when it
/// was labelled before runs carried the stamp; ties are broken by `run_id`.
/// A run whose live state is interrupted, failed or running holds nothing.
pub fn label_history(root: &Path, label: &str) -> Vec<crate::cas_read::Leaf> {
    let mut runs: Vec<crate::cas_read::Leaf> = runs_labelled(root, label)
        .into_iter()
        .filter(|l| matches!(runid::store::leaf_state(&l.dir, &l.record), runid::store::LeafState::Completed))
        .collect();
    let key = |l: &crate::cas_read::Leaf| label_order_key(label_set_at(l).as_deref());
    runs.sort_by(|a, b| key(b).cmp(&key(a)).then_with(|| a.record.run_id.to_hex().to_string().cmp(&b.record.run_id.to_hex().to_string())));
    runs
}

/// When a fit run was given its label: `label_set_at`, else (a run labelled
/// before runs carried the stamp) its `created_at`.
pub fn label_set_at(leaf: &crate::cas_read::Leaf) -> Option<String> {
    let p = &leaf.record.provenance;
    p.label_set_at.clone().or_else(|| p.created_at.clone())
}

/// A stamp as a string that orders by time: `created_at` is written to the
/// second (`…:05Z`) and `label_set_at` to the millisecond (`…:05.120Z`), so
/// both are brought to the millisecond before they are compared.
fn label_order_key(stamp: Option<&str>) -> String {
    let s = stamp.unwrap_or("").trim_end_matches('Z');
    if s.is_empty() || s.contains('.') { s.to_string() } else { format!("{s}.000") }
}

/// A label handle's name and its history position: `m0` → (`m0`, 0),
/// `m0~2` → (`m0`, 2). `~` is outside the label alphabet, so it cannot be part
/// of a label; `m0~` and `m0~x` are refused naming the form.
pub fn parse_label_handle(name: &str) -> Result<(&str, usize), ResolveError> {
    match name.split_once('~') {
        None => Ok((name, 0)),
        Some((label, n)) => match n.parse::<usize>() {
            Ok(k) if !label.is_empty() && n.bytes().all(|b| b.is_ascii_digit()) => Ok((label, k)),
            _ => Err(ResolveError::NotFound(format!(
                "`@{name}` is not a fit handle: a label's earlier holders are named \
                 @<label>~<N>, N a whole number (@{label}~1 is the holder before @{label})"
            ))),
        },
    }
}

/// `@name` or `@name~N` → the run under `root` that holds the label, or its
/// Nth earlier holder ([`label_history`]). Prints the resolution line on
/// stderr, once per handle per process.
pub fn resolve_label(root: &Path, name: &str) -> Result<LeafRef, ResolveError> {
    let (label, n) = parse_label_handle(name)?;
    let mut history = label_history(root, label);
    if history.is_empty() {
        return Err(ResolveError::NotFound(format!(
            "no completed fit run labelled @{label} under {}{}",
            root.join("fits").display(),
            folder_label_hint(root, label)
        )));
    }
    if n >= history.len() {
        let mut msg = format!(
            "@{name} names no run: @{label} has {} earlier holder{} (@{label}~0 is @{label}):",
            history.len() - 1,
            if history.len() == 2 { "" } else { "s" }
        );
        for (i, l) in history.iter().enumerate() {
            let id = l.run_id_hex();
            msg.push_str(&format!(
                "\n    @{label}~{i}  run {}  labelled {}  {}",
                &id[..12.min(id.len())],
                label_set_at(l).unwrap_or_else(|| "at an unrecorded time".into()),
                crate::fit::run_view::RunVerdict::of(l).cell(),
            ));
        }
        return Err(ResolveError::NotFound(msg));
    }
    let earlier = history.len() - 1;
    let leaf = history.swap_remove(n);
    say_resolution(name, &leaf, earlier);
    Ok(LeafRef::of(leaf))
}

/// The line every verb that resolves `@L` or `@L~N` prints to stderr:
/// `@m0 → run 7c01e2 (results/fits/…)`, then `(2 earlier holders)` when the
/// label has any. A label can move between two invocations, so the run it
/// named is said every time, once per handle per process.
fn say_resolution(name: &str, leaf: &crate::cas_read::Leaf, earlier: usize) {
    static SAID: std::sync::Mutex<Vec<String>> = std::sync::Mutex::new(Vec::new());
    let mut said = SAID.lock().unwrap_or_else(|e| e.into_inner());
    if said.iter().any(|s| s == name) {
        return;
    }
    said.push(name.to_string());
    let id = leaf.run_id_hex();
    let cwd = std::env::current_dir().unwrap_or_default();
    let path = leaf.dir.strip_prefix(&cwd).unwrap_or(&leaf.dir).display().to_string();
    let tail = match earlier {
        0 => String::new(),
        1 => " (1 earlier holder)".to_string(),
        n => format!(" ({n} earlier holders)"),
    };
    crate::out::detached().say(format_args!("@{name} → run {} ({path}){tail}", &id[..6.min(id.len())]));
}

/// For a label no run carries: the problem folders whose `fit.meta.json` still
/// holds it — where a camdl before gh#896 stored a fit's label — with the
/// command that gives it to each of their runs. Empty when there are none.
fn folder_label_hint(root: &Path, name: &str) -> String {
    let mut out = String::new();
    let mut folders: Vec<PathBuf> = std::fs::read_dir(root.join("fits"))
        .into_iter()
        .flatten()
        .flatten()
        .map(|e| e.path())
        .filter(|seg| {
            crate::run_meta::read_fit_sidecar(seg).ok().flatten().and_then(|s| s.label).as_deref() == Some(name)
        })
        .collect();
    folders.sort();
    for seg in folders {
        out.push_str(&format!(
            "\n  {} records @{name} in its fit.meta.json, where an older camdl kept a \
             fit's label; a label is now kept on a run. Give it to the run you mean:",
            seg.display()
        ));
        for leaf in completed_leaves(&seg) {
            let id = leaf.run_id_hex();
            out.push_str(&format!(
                "\n    camdl label {} \"{name}\"    # {}",
                &id[..12.min(id.len())],
                leaf.dir.display()
            ));
        }
    }
    out
}

/// Resolve a handle to the one run it names. A problem folder names a run
/// only while it holds one; holding several, it is refused with the runs
/// listed and what sets them apart.
pub fn resolve_leaf(s: &str) -> Result<LeafRef, ResolveError> {
    match resolve_target(s)? {
        FitTarget::Leaf(leaf) => Ok(leaf),
        FitTarget::Segment(segment) => only_run(s, &segment),
    }
}

/// A handle to the one run whose posterior draws a reader replays (gh#896): a
/// run handle names its run; a problem folder names its one completed run
/// that wrote a `draws.tsv`, so an optimizer run beside a sampler run is not
/// a rival. Several such runs are refused with them listed. Returns the run
/// and, when the folder also held runs without draws, a note saying which
/// run was used.
pub fn resolve_run_with_draws(s: &str) -> Result<(LeafRef, Option<String>), ResolveError> {
    match resolve_target(s)? {
        FitTarget::Leaf(leaf) => {
            require_completed(&leaf.dir)?;
            Ok((leaf, None))
        }
        FitTarget::Segment(segment) => {
            let all = completed_leaves(&segment);
            let total = all.len();
            let mut with_draws: Vec<crate::cas_read::Leaf> =
                all.into_iter().filter(|l| l.dir.join("draws.tsv").is_file()).collect();
            match with_draws.len() {
                0 => Err(ResolveError::NotFound(format!(
                    "{} holds no completed run with posterior draws (draws.tsv); \
                     a sampler (pgas, pmmh, mh) writes them",
                    segment.display()
                ))),
                1 => {
                    let leaf = with_draws.remove(0);
                    let note = (total > 1).then(|| {
                        format!(
                            "{} holds {total} runs; using the one with posterior draws, {}",
                            segment.display(),
                            leaf.dir.display()
                        )
                    });
                    Ok((LeafRef::of(leaf), note))
                }
                _ => Err(ResolveError::AmbiguousRuns { handle: s.to_string(), runs: with_draws }),
            }
        }
    }
}

/// Refuse a run, named by a handle for its results to be read, that did not
/// complete. A run directory, a run-id prefix and a hash prefix name a run
/// at any state; `@label`, a `fit.toml` and a problem folder resolve only to
/// completed runs, and pass.
fn require_completed(dir: &Path) -> Result<(), ResolveError> {
    match crate::leaf_use::read(dir).map_err(ResolveError::NotFound)? {
        Some(leaf) => crate::leaf_use::require_completed(&leaf).map_err(ResolveError::NotCompleted),
        None => Err(ResolveError::NotFound(format!("{} holds no run.json", dir.display()))),
    }
}

/// The one completed run in a problem folder, or a refusal: none, or several
/// listed with what sets them apart.
fn only_run(handle: &str, segment: &Path) -> Result<LeafRef, ResolveError> {
    let mut leaves = completed_leaves(segment);
    match leaves.len() {
        0 => Err(ResolveError::NotFound(format!(
            "{} holds no completed fit run",
            segment.display()
        ))),
        1 => Ok(LeafRef::of(leaves.remove(0))),
        _ => Err(ResolveError::AmbiguousRuns { handle: handle.to_string(), runs: leaves }),
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
            let (leaf, cfg) = resolve_config(&path)?;
            (FitTarget::Leaf(leaf), Some(cfg))
        }
        _ => (resolve_target(s)?, None),
    };
    let (segment, leaf) = match target {
        FitTarget::Segment(seg) => (seg, None),
        FitTarget::Leaf(l) => {
            let seg = crate::fit::fit_view::leaf_segment(&l.dir)
                .map_err(ResolveError::NotFound)?
                .ok_or_else(|| {
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
        None => load_config_for_segment(&segment)?,
    };
    Ok(ResolvedFit { segment, leaf, config })
}

/// [`resolve_fit`] for a verb that reads one run (`fit predict`, `compare`):
/// a problem-folder handle names its one run, and holding several it is
/// refused with the runs listed rather than one chosen by a rule (gh#896).
/// The run must have completed: one that failed, is running or was
/// interrupted is refused with its state. The returned `leaf` is always
/// `Some`.
pub fn resolve_fit_run(s: &str) -> Result<ResolvedFit, ResolveError> {
    let mut resolved = resolve_fit(s)?;
    match &resolved.leaf {
        Some(dir) => require_completed(dir)?,
        None => resolved.leaf = Some(only_run(s, &resolved.segment)?.dir),
    }
    Ok(resolved)
}

/// The fit problem a `--fit` handle names, for a command that reads a fit's
/// problem but none of its results (`pfilter`, `profile`, `survey`): the
/// config as the user wrote it, the configuration resolved from it, where its
/// relative paths are anchored, and the configuration parsed.
///
/// The handle forms are `fit predict`'s. A `fit.toml` names the problem the
/// file states as it now reads (gh#896: a `fit.toml` names the run the file
/// alone makes), whether or not a fit of it is stored, so a `pfilter --fit`
/// before any fit runs keeps working; a problem a fit run's command line
/// changed (`fit run --allow-forcing-extrapolation`) is not the file's, and is
/// not read through it. A leaf directory, `@label`, run-id or fit-id prefix,
/// or problem folder names a stored fit, whose `fit.resolved.toml` is read: the
/// config the fit ran under, a flag's problem value written in (gh#1268) and a
/// sweep point's own `[fixed]`.
///
/// One thing the resolved config does not hold: the run's own method. Its
/// `[method]` is the file's, while a folder holds every method run on its
/// problem and a method flag (`--starts`, a sampler flag) is recorded in the
/// leaf; `profile`, the one reader of the fit's `[method]` (its nl-* budget
/// and `starts`), reads the file's.
pub(crate) struct FitProblem {
    /// The handle as typed, for messages.
    pub handle: String,
    /// The config as the user wrote it, read once: the fit file, or the stored
    /// fit's `fit.toml.original`. What a reader that archives the config
    /// (`profile`) stores as its own `fit.toml.original`, and hashes.
    pub original: String,
    /// The stored fit's `fit.resolved.toml`, what `profile` stores as its own;
    /// `None` for a fit file, whose resolution needs the model (a
    /// `from_scenario` is expanded from it), so `profile` builds it
    /// ([`crate::fit::run_config::resolved_config_text`]) with the `[fixed]`
    /// values it resolved.
    pub resolved: Option<String>,
    /// The path the config's relative paths were written against: the file
    /// itself, or the stored fit's source config (`<segment>/<source_config>`).
    /// Where a reader that archives the config (`profile`'s sidecar) records
    /// it as written.
    pub anchor: PathBuf,
    /// The configuration, parsed and anchored at `anchor`.
    pub config: FitConfig,
}

impl FitProblem {
    /// The handle's file name, for messages.
    pub fn name(&self) -> String {
        Path::new(&self.handle)
            .file_name()
            .map(|n| n.to_string_lossy().into_owned())
            .unwrap_or_else(|| self.handle.clone())
    }
}

/// Resolve a `--fit` handle to the problem it names ([`FitProblem`]).
pub(crate) fn resolve_fit_problem(s: &str) -> Result<FitProblem, ResolveError> {
    // A `*.toml`, or any existing regular file, is a fit config: these
    // readers took a config of any file name before they took handles.
    let config_path = match FitRef::classify(s) {
        FitRef::Config(path) => Some(path),
        _ if Path::new(s).is_file() => Some(PathBuf::from(s)),
        _ => None,
    };
    if let Some(path) = config_path {
        let text = std::fs::read_to_string(&path)
            .map_err(|e| ResolveError::NotFound(format!("cannot read fit config {}: {e}", path.display())))?;
        let config = FitConfig::parse_anchored_at(&text, &path, &path)
            .map_err(|e| ResolveError::NotFound(format!("cannot load fit config {}: {e}", path.display())))?;
        return Ok(FitProblem { handle: s.to_string(), original: text, resolved: None, anchor: path, config });
    }
    let segment = match resolve_target(s)? {
        FitTarget::Segment(seg) => seg,
        FitTarget::Leaf(l) => crate::fit::fit_view::leaf_segment(&l.dir)
            .map_err(ResolveError::NotFound)?
            .ok_or_else(|| ResolveError::NotFound(format!("{} is not inside a fit's directory", l.dir.display())))?,
    };
    let (resolved, anchor, config) = read_segment_config(&segment)?;
    let archived = segment.join(crate::run_meta::ARCHIVED_CONFIG);
    let original = std::fs::read_to_string(&archived)
        .map_err(|e| ResolveError::NotFound(format!("cannot read {}: {e}", archived.display())))?;
    Ok(FitProblem { handle: s.to_string(), original, resolved: Some(resolved), anchor, config })
}

/// Recover the config for a segment reached by run-dir / `@label` / hash.
///
/// The stored `fit.resolved.toml` is the authoritative record of the config
/// this fit ran under — the file the user still has on disk may have drifted
/// since, and holds neither a command-line override nor a sweep point's values.
/// So the resolved config is what we read, always; there is no "prefer the
/// live original" branch. A folder written before fit folders stored one is
/// refused, naming the re-run that writes it; the verbatim `fit.toml.original`
/// beside it is never read in its place.
///
/// What the stored config cannot carry is its own *location*. Data archival is not yet
/// part of the sealed packet (Phase 1 archives only the model IR), so the data
/// still lives where the config says — at paths written relative to the
/// **source config's directory**, not to the segment. Anchoring them at the
/// archive's own location turned `../data/streams/x.tsv` into
/// `results/fits/data/streams/x.tsv` and broke `compare` for every project whose
/// configs use relative paths (gh#652). The sidecar records `source_config`,
/// relative to the segment, so that directory is known even when the file
/// itself is gone or changed — and known from any working directory, which a
/// path recorded relative to where `fit run` was started was not (gh#1009).
/// [`FitConfig::load_anchored_at`] resolves against `<segment>/<source_config>`.
///
/// A data file the recovered config names but that is not there is refused
/// here, naming the stream, the path it was looked for at, the rule that put it
/// there, and the content hash the fit recorded for it — so the reader can find
/// the file it needs rather than a downstream verb failing on a bare path.
///
/// A fit run entirely from the CLI has no `fit.toml` at all: `write_fit_sidecar`
/// records no `source_config` and archives nothing. That is a real gap, not a
/// lookup miss, so it gets its own named error rather than a wrong path.
///
/// Errors come in two kinds. A directory with no `fit.meta.json`, and a
/// CLI-only fit, hold no config to recover: [`ResolveError::NotFound`]. A fit
/// folder whose config cannot be read as the fit ran it — its `fit.meta.json`
/// unreadable, no `fit.resolved.toml` or one that does not parse, a data file
/// gone, or a `[fixed]` that is not the one the fit recorded running at — is
/// [`ResolveError::StoredConfig`], which no reader may pass over.
fn load_config_for_segment(segment: &Path) -> Result<FitConfig, ResolveError> {
    read_segment_config(segment).map(|(_, _, config)| config)
}

/// [`load_config_for_segment`], with the resolved config's text it parsed and
/// the path that text's relative paths are anchored at.
fn read_segment_config(segment: &Path) -> Result<(String, PathBuf, FitConfig), ResolveError> {
    let side = crate::run_meta::read_fit_sidecar(segment)
        .map_err(ResolveError::StoredConfig)?
        .ok_or_else(|| {
            ResolveError::NotFound(format!("{} is not a complete fit directory: no fit.meta.json", segment.display()))
        })?;
    let Some(source_config) = side.source_config.as_deref() else {
        return Err(ResolveError::NotFound(format!(
            "{} was produced without a fit.toml (a CLI-only run), so it carries \
             no config to recover: no archived config and no recorded config \
             directory to resolve its data paths against.\n  \
             Pass the fit toml, or re-run the fit from one.",
            segment.display()
        )));
    };
    let resolved = crate::run_meta::require_resolved_config(segment).map_err(ResolveError::StoredConfig)?;
    let anchor = segment.join(source_config);
    let unreadable = |e: String| {
        ResolveError::StoredConfig(format!(
            "{} is a fit directory but its resolved config could not be read: {e}\n  \
             (expected {})",
            segment.display(),
            resolved.display()
        ))
    };
    let text = std::fs::read_to_string(&resolved)
        .map_err(|e| unreadable(format!("cannot read {}: {e}", resolved.display())))?;
    let config = FitConfig::parse_anchored_at(&text, &resolved, &anchor).map_err(unreadable)?;
    refuse_fixed_not_run(&config, &side, segment).map_err(ResolveError::StoredConfig)?;
    refuse_missing_data(&config, &side, &anchor).map_err(ResolveError::StoredConfig)?;
    Ok((text, anchor, config))
}

/// `Err` naming every parameter whose value in the stored config's `[fixed]`
/// is not the value `fit.meta.json` records the fit running at (`fixed`),
/// with both values: a config that would not compute the stored fit. A name
/// the record holds that the config's `[fixed]` lacks passes unless the config
/// estimates it. The config's `[fixed]` is resolved as the fit resolves it.
fn refuse_fixed_not_run(
    config: &FitConfig,
    side: &crate::run_meta::FitSidecar,
    segment: &Path,
) -> Result<(), String> {
    let problem = &config.problem;
    let stored = problem.fixed.resolve()?;
    let differs: Vec<String> = side
        .fixed
        .iter()
        .filter_map(|(name, &ran)| match stored.get(name) {
            Some(&v) if v != ran => Some(format!("{name} = {v} in the stored config, {ran} in the run")),
            None if problem.estimate.contains_key(name) => {
                Some(format!("{name} estimated in the stored config, fixed at {ran} in the run"))
            }
            _ => None,
        })
        .collect();
    if differs.is_empty() {
        return Ok(());
    }
    Err(format!(
        "{}: the stored config's [fixed] is not what the fit ran at ({}), as recorded in its \
         fit.meta.json, so it would not compute the stored fit.",
        segment.display(),
        differs.join("; ")
    ))
}

/// `Err` naming every data file the recovered config binds that is not on
/// disk: the stream, where it was looked for, and the hash `fit.meta.json`
/// recorded for it (`data_hashes` / `holdout_hashes`, keyed by source).
///
/// Reads `[data]` without the model: which files are bound needs none (the
/// shorthand's one file, or the keys as written), and the shorthand's hash is
/// every recorded value, whether an older sidecar keyed it by leaf or a newer
/// one by source. Resolving `StreamBindings` here would load (or compile) the
/// model on every handle resolution to answer a question it does not change.
fn refuse_missing_data(
    config: &FitConfig,
    side: &crate::run_meta::FitSidecar,
    anchor: &Path,
) -> Result<(), String> {
    let Some(data) = &config.problem.data else { return Ok(()) };
    let hash_of = |streams: &[&str]| -> String {
        let hashes: std::collections::BTreeSet<&str> = streams.iter()
            .filter_map(|s| side.data_hashes.get(*s).map(String::as_str))
            .collect();
        if hashes.is_empty() {
            "no hash recorded".to_string()
        } else {
            format!("recorded hash {}", hashes.into_iter().collect::<Vec<_>>().join(", "))
        }
    };
    let mut missing: Vec<String> = Vec::new();
    if let Some(file) = &data.file {
        if !Path::new(file).is_file() {
            // The shorthand binds every stream to the one file; each stream's
            // recorded hash is that file's.
            let streams: Vec<&str> = side.data_hashes.keys().map(String::as_str).collect();
            missing.push(format!("[data] file: looked for {file} ({})", hash_of(&streams)));
        }
    }
    for (stream, path) in &data.observations {
        if !Path::new(path).is_file() {
            missing.push(format!("stream `{stream}`: looked for {path} ({})", hash_of(&[stream])));
        }
    }
    for (stream, path) in data.holdout.iter().flatten() {
        if !Path::new(path).is_file() {
            let recorded = side.holdout_hashes.get(stream)
                .map_or_else(|| "no hash recorded".to_string(), |h| format!("recorded hash {h}"));
            missing.push(format!("holdout stream `{stream}`: looked for {path} ({recorded})"));
        }
    }
    if missing.is_empty() {
        return Ok(());
    }
    Err(format!(
        "the fit's data is not where its config says:\n  {}\n  \
         The config's relative data paths resolve against the source config's \
         directory, {} (`source_config` in fit.meta.json, relative to the fit folder). \
         Restore the file there, or pass a fit.toml whose paths name it.",
        missing.join("\n  "),
        anchor.parent().unwrap_or(anchor).display()
    ))
}

/// The canonical identity of a fit.toml's text — what the config MEANS, with
/// comments, whitespace, and float spelling discarded (gh#653). Parsed with the
/// production parser, so a config this cannot parse cannot match a fit either.
/// `fit run` records it on every leaf it writes or reuses, in
/// `inputs.config_identities`.
pub(crate) fn config_identity(text: &str) -> Result<String, String> {
    let config = FitConfig::from_toml_str(text)?;
    crate::fit::cas::config_identity_hash(&config)
}

/// The Config (`fit.toml`) branch: the runs made from this config's MEANING —
/// the completed fit leaves whose recorded `inputs.config_identities` include
/// the file's — under the root it names. Returns the one run and the live config
/// (loaded once here).
///
/// Each run records the identity of the config it was read from, so the file
/// in hand names the runs made from it as it now reads, not the problem
/// folder's latest archived config (gh#896). Resolution is exact (proposal
/// 2026-11-17 §1d): the answer is the run this engine made — its
/// `engine_version` is this binary's `ENGINE_ID` and its `engine_switches`
/// the switches `fit run` keys in this process, so it is the run `fit run` of
/// this file would reuse. Runs another engine made (another build, or this
/// one with other switches) are never chosen in its place, however recent:
/// with none by this engine they are listed ([`ResolveError::OtherEngines`]),
/// and beside this engine's they are noted. Several runs by this engine
/// (several fit seeds, a `--starts` override) are listed.
///
/// Identity is the canonical hash of the parsed value tree, not the raw bytes
/// (gh#653): reflowing a comment, reordering two top-level tables, or writing
/// `1.00` for `1.0` changes the bytes and not the meaning. A comment reflow
/// across a set of national configs once orphaned hours of completed
/// sampling. The raw-bytes hash is still recorded in each sidecar as
/// `fit_toml_hash`, as provenance.
///
/// A stored run that records no config identity (written before runs carried
/// one) cannot be matched by file, so the not-found message counts them
/// rather than passing over them silently.
fn resolve_config(path: &Path) -> Result<(LeafRef, FitConfig), ResolveError> {
    let text = std::fs::read_to_string(path)
        .map_err(|e| ResolveError::NotFound(format!("cannot read fit config {}: {e}", path.display())))?;
    let wanted = config_identity(&text)
        .map_err(|e| ResolveError::NotFound(format!("cannot load fit config {}: {e}", path.display())))?;
    let config = FitConfig::load(&path.to_string_lossy())
        .map_err(|e| ResolveError::NotFound(format!("cannot load fit config {}: {e}", path.display())))?;
    let cas_root = crate::run_paths::output_root(None, config.problem.output_dir.as_deref());
    let fits_dir = cas_root.join("fits");
    let completed: Vec<crate::cas_read::Leaf> = crate::cas_read::walk_fit_leaves(&cas_root)
        .into_iter()
        .filter(|l| l.record.status == runid::RunStatus::Completed)
        .collect();
    let recorded = |l: &crate::cas_read::Leaf| -> Option<Vec<String>> {
        l.record.inputs.get("config_identities").and_then(|v| v.as_array()).map(|a| {
            a.iter().filter_map(|x| x.as_str().map(String::from)).collect()
        })
    };
    let unrecorded = completed.iter().filter(|l| recorded(l).is_none()).count();
    // Runs read from this file whose command-line options changed what they
    // computed (`--allow-forcing-extrapolation`, `--starts`, a sampler flag,
    // `--sweep`, another seed): none is the file's run, so none answers for
    // it, but the refusal names them.
    let mut with_options: Vec<std::path::PathBuf> = completed
        .iter()
        .filter(|l| {
            l.record.inputs.get("config_identities_with_options").and_then(|v| v.as_array())
                .is_some_and(|a| a.iter().any(|x| x.as_str() == Some(wanted.as_str())))
        })
        .map(|l| l.dir.clone())
        .collect();
    with_options.sort();
    let mut matches: Vec<crate::cas_read::Leaf> = completed
        .into_iter()
        .filter(|l| recorded(l).is_some_and(|ids| ids.contains(&wanted)))
        .collect();
    matches.sort_by(|a, b| a.dir.cmp(&b.dir));
    let (mut this_build, other_builds): (Vec<_>, Vec<_>) = matches
        .into_iter()
        .partition(made_by_this_engine);
    match (this_build.len(), other_builds.len()) {
        (0, 0) => {
            let mut msg = format!(
                "no completed fit run of {} as it now reads, under {}.\n  Run `camdl fit run {}` first.",
                path.display(),
                fits_dir.display(),
                path.display()
            );
            if !with_options.is_empty() {
                msg.push_str(&format!(
                    "\n  {} run(s) were made from this file with command-line options that \
                     change what a fit computes (such as --allow-forcing-extrapolation, \
                     --starts or a sampler flag), so none is the run the file alone makes. \
                     Name one by its leaf directory:",
                    with_options.len()
                ));
                for d in &with_options {
                    msg.push_str(&format!("\n    {}", d.display()));
                }
            }
            if unrecorded > 0 {
                msg.push_str(&format!(
                    "\n  {unrecorded} stored run(s) record no config identity (written before \
                     runs carried one) and cannot be matched by file; name one by its leaf \
                     directory or run_id."
                ));
            }
            Err(ResolveError::NotFound(msg))
        }
        (0, _) => Err(ResolveError::OtherEngines { handle: path.display().to_string(), runs: other_builds }),
        (1, others) => {
            if others > 0 {
                crate::out::detached().say(format_args!(
                    "note: {} also matches {others} run(s) by another camdl build or with other \
                     engine switches, which are not \
                     used; name one by its run-id prefix or leaf directory to read it",
                    path.display(),
                ));
            }
            Ok((LeafRef::of(this_build.remove(0)), config))
        }
        _ => Err(ResolveError::AmbiguousRuns { handle: path.display().to_string(), runs: this_build }),
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

    /// Write `seg` as `fit run` writes a fit folder made from the file at
    /// `toml_path` with no option: the sidecar, the file as read and its
    /// resolved config.
    fn write_fit_folder(
        seg: &Path,
        toml_path: &Path,
        side: &crate::run_meta::FitSidecar,
    ) -> std::io::Result<()> {
        let text = std::fs::read_to_string(toml_path)?;
        let fixed = FitConfig::from_toml_str(&text).expect("a fit file").problem.fixed.resolve().expect("[fixed]");
        let no_flag = crate::fit::run_config::ProblemFlags { allow_forcing_extrapolation: false };
        let resolved = crate::fit::run_config::resolved_config_text(&text, no_flag, &fixed)
            .expect("a resolved config");
        crate::run_meta::write_fit_sidecar_with_config(seg, toml_path, &text, &resolved, Path::new(""), side)
    }

    /// A stored config whose `[fixed]` is not what `fit.meta.json` records the
    /// fit running at is refused, as a stored config, naming the parameter and
    /// both values; one that agrees resolves.
    #[test]
    fn a_stored_config_whose_fixed_is_not_the_runs_is_refused() {
        let (tmp, toml_path, seg) = portable_project("fixed_not_run");
        let side = |gamma: f64| crate::run_meta::FitSidecar {
            fixed: std::collections::BTreeMap::from([("gamma".to_string(), gamma)]),
            ..Default::default()
        };
        write_fit_folder(&seg, &toml_path, &side(0.2)).unwrap();
        resolve_fit(&seg.to_string_lossy()).unwrap_or_else(|e| panic!("the record agrees: {e}"));
        write_fit_folder(&seg, &toml_path, &side(0.3)).unwrap();
        let err = resolve_fit(&seg.to_string_lossy()).err().expect("the record disagrees");
        assert!(matches!(err, ResolveError::StoredConfig(_)), "{err}");
        let err = err.to_string();
        assert!(err.contains("gamma = 0.2 in the stored config, 0.3 in the run"), "{err}");
        let _ = std::fs::remove_dir_all(&tmp);
    }

    /// A fit folder an older camdl wrote, with `fit.toml.original` but no
    /// `fit.resolved.toml`, is refused, naming the re-run that writes it; the
    /// verbatim file is never read in its place.
    #[test]
    fn a_fit_folder_without_a_resolved_config_is_refused_naming_the_rerun() {
        let (tmp, toml_path, seg) = portable_project("no_resolved");
        write_fit_folder(&seg, &toml_path, &Default::default()).unwrap();
        resolve_fit(&seg.to_string_lossy()).unwrap_or_else(|e| panic!("a current folder resolves: {e}"));
        std::fs::remove_file(seg.join(crate::run_meta::RESOLVED_CONFIG)).unwrap();
        assert!(seg.join(crate::run_meta::ARCHIVED_CONFIG).is_file());
        let err = resolve_fit(&seg.to_string_lossy()).err().expect("no resolved config");
        assert!(matches!(err, ResolveError::StoredConfig(_)), "a fit folder, not a miss: {err}");
        let err = err.to_string();
        assert!(err.contains("has no fit.resolved.toml"), "{err}");
        assert!(err.contains("written by an older camdl"), "{err}");
        assert!(err.contains("Re-run the fit"), "{err}");
        let _ = std::fs::remove_dir_all(&tmp);
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
            ..Default::default()
        };
        write_fit_folder(&seg, &toml_path, &side).unwrap();
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
            seg.join("../../../fits/../models/sir.camdl"),
        );
        let _ = std::fs::remove_dir_all(&tmp);
    }

    /// gh#1009: the source config is recorded relative to the fit folder, so
    /// it names the same file from any working directory — `fits/a.toml` is
    /// three levels up from `results/fits/sir-0000beef/`.
    #[test]
    fn the_source_config_is_recorded_relative_to_the_fit_folder() {
        let (tmp, toml_path, seg) = portable_project("relative");
        write_fit_folder(&seg, &toml_path, &Default::default()).unwrap();
        let side = crate::run_meta::read_fit_sidecar(&seg).unwrap().unwrap();
        assert_eq!(side.source_config.as_deref(), Some("../../../fits/a.toml"));
        assert_eq!(side.schema_version, crate::run_meta::FIT_META_SCHEMA_VERSION);
        assert_eq!(std::fs::canonicalize(seg.join("../../../fits/a.toml")).unwrap(),
            std::fs::canonicalize(&toml_path).unwrap());
        let _ = std::fs::remove_dir_all(&tmp);
    }

    /// A data file the recovered config binds but that is gone is refused
    /// with the stream, the path looked for, and the hash the fit recorded.
    #[test]
    fn a_missing_data_file_is_named_with_where_it_was_looked_for_and_its_hash() {
        let (tmp, toml_path, seg) = portable_project("missing_data");
        let side = crate::run_meta::FitSidecar {
            data_hashes: std::collections::BTreeMap::from([("cases".to_string(), "b39b0a43".to_string())]),
            ..Default::default()
        };
        write_fit_folder(&seg, &toml_path, &side).unwrap();
        std::fs::remove_file(tmp.join("data/streams/cases.tsv")).unwrap();
        let err = resolve_fit(&seg.to_string_lossy()).err().expect("the data is gone").to_string();
        assert!(err.contains("stream `cases`"), "{err}");
        assert!(err.contains("data/streams/cases.tsv"), "names where it looked: {err}");
        assert!(err.contains("recorded hash b39b0a43"), "{err}");
        assert!(err.contains("source_config"), "says how the path was resolved: {err}");
        let _ = std::fs::remove_dir_all(&tmp);
    }

    /// A missing `[data.holdout]` file is named with the hash the fit recorded
    /// for it (`holdout_hashes`), and as unrecorded by an older sidecar.
    #[test]
    fn a_missing_holdout_file_is_named_with_its_recorded_hash() {
        for (slug, holdout_hashes, expect) in [
            ("missing_holdout_new", vec![("cases", "c0ffee11")], "recorded hash c0ffee11"),
            ("missing_holdout_old", vec![], "no hash recorded"),
        ] {
            let (tmp, toml_path, seg) = portable_project(slug);
            std::fs::write(&toml_path, format!(
                "{PORTABLE_TOML}\n[data.holdout]\ncases = \"../data/streams/holdout.tsv\"\n")).unwrap();
            let side = crate::run_meta::FitSidecar {
                holdout_hashes: holdout_hashes.into_iter()
                    .map(|(k, v)| (k.to_string(), v.to_string())).collect(),
                ..Default::default()
            };
            write_fit_folder(&seg, &toml_path, &side).unwrap();
            let err = resolve_fit(&seg.to_string_lossy()).err().expect("the holdout is gone").to_string();
            assert!(err.contains("holdout stream `cases`") && err.contains("holdout.tsv")
                    && err.contains(expect), "{slug}: {err}");
            let _ = std::fs::remove_dir_all(&tmp);
        }
    }

    /// An older `fit.meta.json` is refused with its version named, never read
    /// as if its `fit_toml_path` were a `source_config`.
    #[test]
    fn an_older_fit_meta_is_refused_naming_its_version() {
        let (tmp, _toml, seg) = portable_project("old_schema");
        std::fs::write(seg.join("fit.meta.json"),
            r#"{"model_path":"sir.camdl","fit_toml_path":"fits/a.toml"}"#).unwrap();
        let err = resolve_fit(&seg.to_string_lossy()).err().expect("an old record").to_string();
        assert!(err.contains("schema version 1") && err.contains("reads version 3"), "{err}");
        // A version-2 record could omit the calendar axis; it is refused by
        // name too, not read with the axis missing (D59).
        std::fs::write(seg.join("fit.meta.json"),
            r#"{"schema_version":2,"source_model":"sir.camdl","source_config":"fit.toml"}"#).unwrap();
        let err = resolve_fit(&seg.to_string_lossy()).err().expect("a v2 record").to_string();
        assert!(err.contains("schema version 2") && err.contains("reads version 3"), "{err}");
        let _ = std::fs::remove_dir_all(&tmp);
    }

    #[test]
    fn cli_only_fit_has_a_named_error_not_a_wrong_path() {
        // A fit run with no fit.toml archives no config and records no config
        // location (`source_config` is absent), so there is nothing to anchor. That
        // is a named gap, not a silent miss or a plausible-looking wrong path.
        let (tmp, _toml, seg) = portable_project("clionly");
        crate::run_meta::write_fit_sidecar(
            &seg,
            Path::new(""),
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

    /// Plant one completed fit run under `seg` whose record lists
    /// `identities` as the configs it was reached from (`None`: a run written
    /// before runs recorded any). Returns the leaf directory.
    fn plant_run(seg: &Path, identities: Option<&[String]>) -> PathBuf {
        let leaf = seg.join("if2-00000000").join("seed_1-00000000");
        std::fs::create_dir_all(&leaf).unwrap();
        let inputs = match identities {
            Some(ids) => serde_json::json!({ "method": "if2", "config_identities": ids }),
            None => serde_json::json!({ "method": "if2" }),
        };
        let rec = serde_json::json!({
            "format_version": 1, "kind": "fit_method", "run_id": "ab".repeat(32),
            "hash_version": 1, "calendar": {"days_per_unit": 1.0, "origin": null, "time_unit": "days"}, "ir_version": "0.7", "engine_version": crate::version::ENGINE_ID,
            "levels": [], "status": "completed", "artifacts": {}, "inputs": inputs,
        });
        std::fs::write(leaf.join("run.json"), rec.to_string()).unwrap();
        leaf
    }

    /// A run directory whose `run.json` is not a run record is an error
    /// naming the leaf and why, not a problem folder that "holds no completed
    /// fit run"; a directory with no record is still a problem folder.
    #[test]
    fn a_run_dir_with_an_unreadable_record_is_an_error_naming_it() {
        let tmp = crate::test_support::unique_temp_dir("handle_unreadable_leaf");
        let leaf = plant_run(&tmp.join("fits").join("toy-00000000"), None);
        std::fs::write(leaf.join("run.json"), "{ truncated").unwrap();
        let err = resolve_target(&leaf.to_string_lossy()).expect_err("an unreadable record").to_string();
        assert!(err.starts_with(&format!("{}: its run.json is not a run record", leaf.display())), "{err}");
        assert!(!err.contains("holds no completed fit run"), "{err}");
        let segment = leaf.parent().unwrap().parent().unwrap();
        assert!(matches!(resolve_target(&segment.to_string_lossy()), Ok(FitTarget::Segment(_))));
        let _ = std::fs::remove_dir_all(&tmp);
    }

    /// Stand up `<tmp>/fits/a.toml` plus a segment under `<tmp>/results/fits/`
    /// holding one run made from `archived`, and resolve `live` as a
    /// `fit.toml` handle against it. Returns the resolved run's directory,
    /// the planted run's, and the temp root.
    fn resolve_live_against_archive(
        slug: &str,
        archived: &str,
        live: &str,
    ) -> (Result<PathBuf, ResolveError>, PathBuf, PathBuf) {
        let (tmp, toml_path, seg) = portable_project(slug);
        std::fs::write(&toml_path, archived).unwrap();
        let side = crate::run_meta::FitSidecar {
            fit_toml_hash: crate::hashing::sha256_hex(archived.as_bytes()),
            ..Default::default()
        };
        write_fit_folder(&seg, &toml_path, &side).unwrap();
        let leaf = plant_run(&seg, Some(&[config_identity(archived).unwrap()]));
        std::fs::write(&toml_path, live).unwrap();
        (resolve_config(&toml_path).map(|r| r.0.dir), leaf, tmp)
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
        // A method knob is a real difference and must not resolve.
        let changed = ARCHIVED.replace("particles = 10", "particles = 20");
        assert_ne!(changed, ARCHIVED);
        let (got, _seg, tmp) = resolve_live_against_archive("particles", ARCHIVED, &changed);
        let err = got.expect_err("a changed particle count is a different fit").to_string();
        assert!(err.contains("no completed fit run of"), "got: {err}");
        let _ = std::fs::remove_dir_all(&tmp);
    }

    #[test]
    fn a_changed_data_path_does_not_resolve() {
        // NEGATIVE CONTROL. Same shape, different input file.
        let changed = ARCHIVED.replace("streams/cases.tsv", "streams/cases_v2.tsv");
        assert_ne!(changed, ARCHIVED);
        let (got, _seg, tmp) = resolve_live_against_archive("datapath", ARCHIVED, &changed);
        let err = got.expect_err("a changed data path is a different fit").to_string();
        assert!(err.contains("no completed fit run of"), "got: {err}");
        let _ = std::fs::remove_dir_all(&tmp);
    }

    #[test]
    fn a_run_with_no_recorded_config_is_counted_not_skipped() {
        // A run is matched by the config identity it records, so one written
        // before runs recorded any cannot be matched by file. Say so — a
        // silent skip reads as "that fit was never run".
        let (tmp, toml_path, seg) = portable_project("noidentity");
        std::fs::write(&toml_path, ARCHIVED).unwrap();
        plant_run(&seg, None);

        let err = resolve_config(&toml_path).map(|r| r.0)
            .expect_err("no recorded identity, no match")
            .to_string();
        assert!(
            err.contains("1 stored run(s) record no config identity"),
            "the unmatchable run is counted; got: {err}"
        );
        let _ = std::fs::remove_dir_all(&tmp);
    }

    #[test]
    fn a_config_names_the_run_made_from_it_not_the_folders_latest() {
        // gh#896: two runs in one folder, made from two versions of one file.
        // The file as it now reads names the run made from it, whatever the
        // folder's archive holds.
        let (tmp, toml_path, seg) = portable_project("tworuns");
        let older = ARCHIVED.replace("particles = 10", "particles = 20");
        std::fs::write(&toml_path, &older).unwrap();
        let side = crate::run_meta::FitSidecar {
            ..Default::default()
        };
        write_fit_folder(&seg, &toml_path, &side).unwrap();
        let a = plant_run(&seg, Some(&[config_identity(ARCHIVED).unwrap()]));
        let b = seg.join("if2-11111111").join("seed_1-00000000");
        std::fs::create_dir_all(&b).unwrap();
        let rec = std::fs::read_to_string(a.join("run.json")).unwrap()
            .replace(&config_identity(ARCHIVED).unwrap(), &config_identity(&older).unwrap())
            .replace(&"ab".repeat(32), &"cd".repeat(32));
        std::fs::write(b.join("run.json"), rec).unwrap();

        for (text, want) in [(ARCHIVED, &a), (older.as_str(), &b)] {
            std::fs::write(&toml_path, text).unwrap();
            let got = resolve_config(&toml_path).map(|r| r.0.dir)
                .unwrap_or_else(|e| panic!("resolves: {e}"));
            assert_eq!(std::fs::canonicalize(&got).unwrap(), std::fs::canonicalize(want).unwrap());
        }
        let _ = std::fs::remove_dir_all(&tmp);
    }

    /// A fit leaf record whose `method` and `seed` levels hash to the given
    /// one-character stems, for the ambiguity wording tests.
    fn leaf_with(method: char, seed: char, dir: &str) -> crate::cas_read::Leaf {
        let h = |c: char| c.to_string().repeat(64);
        let record: runid::RunRecord = serde_json::from_str(&format!(
            r#"{{"format_version":1,"kind":"fit_method","run_id":"{run}","hash_version":1,"calendar":{{"days_per_unit":1.0,"origin":null,"time_unit":"days"}},"ir_version":"0.7","engine_version":"0.1.0+test","levels":[{{"name":"fit","label":"demo","hash":"{fit}","schema_version":1}},{{"name":"method","label":"if2","hash":"{m}","schema_version":1}},{{"name":"seed","label":"seed_1","hash":"{s}","schema_version":1}}],"status":"completed","artifacts":{{}},"inputs":{{"method":"if2","n_chains":2}},"provenance":{{}}}}"#,
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

    /// gh#896: a hex prefix matching both a run's id and a problem folder's
    /// fit id is refused with both listed, not resolved to the run.
    #[test]
    fn a_prefix_matching_a_run_and_a_fit_id_is_refused() {
        let root = crate::test_support::unique_temp_dir("handle_prefix_both");
        let seg = root.join("fits").join("demo-abcd0000");
        let leaf = seg.join("if2-11111111").join("seed_1-22222222");
        std::fs::create_dir_all(&leaf).unwrap();
        crate::run_meta::write_fit_sidecar(
            &seg, Path::new("nonexistent.toml"), Path::new(""), &crate::run_meta::FitSidecar::default()).unwrap();
        let rec = serde_json::json!({
            "format_version": 1, "kind": "fit_method", "run_id": format!("abcd9{}", "0".repeat(59)),
            "hash_version": 1, "calendar": {"days_per_unit": 1.0, "origin": null, "time_unit": "days"}, "ir_version": "0.7", "engine_version": "0.1.0+test",
            "levels": [
                {"name": "fit", "label": "demo", "hash": format!("abcd0{}", "0".repeat(59)), "schema_version": 1},
                {"name": "method", "label": "if2", "hash": "1".repeat(64), "schema_version": 1},
                {"name": "seed", "label": "seed_1", "hash": "2".repeat(64), "schema_version": 1}],
            "status": "completed", "artifacts": {}, "inputs": {"method": "if2"},
        });
        std::fs::write(leaf.join("run.json"), rec.to_string()).unwrap();

        let err = resolve_hash_prefix(&root, "abcd").expect_err("both namespaces match").to_string();
        assert!(err.contains("abcd resolves to 2 fits"), "{err}");
        assert!(err.contains(&leaf.display().to_string()) && err.contains(&seg.display().to_string()),
            "lists the run and the folder: {err}");
        // A longer prefix in one namespace resolves.
        assert!(matches!(resolve_hash_prefix(&root, "abcd9").unwrap(), FitTarget::Leaf(_)));
        assert!(matches!(resolve_hash_prefix(&root, "abcd0").unwrap(), FitTarget::Segment(_)));
        std::fs::remove_dir_all(&root).ok();
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

    /// Plant a completed run of `config` under `seg` at `<method_dir>/seed_1-…`,
    /// recorded as made by the build `engine` whose readable version is
    /// `camdl_version`. Returns the leaf directory.
    fn plant_run_built_by(seg: &Path, method_dir: &str, run_id: &str, config: &str, engine: &str, camdl_version: &str) -> PathBuf {
        let leaf = seg.join(method_dir).join("seed_1-00000000");
        std::fs::create_dir_all(&leaf).unwrap();
        let rec = serde_json::json!({
            "format_version": 1, "kind": "fit_method", "run_id": run_id,
            "hash_version": 1, "calendar": {"days_per_unit": 1.0, "origin": null, "time_unit": "days"},
            "ir_version": "0.7", "engine_version": engine,
            "levels": [], "status": "completed", "artifacts": {},
            "inputs": { "method": "if2", "config_identities": [config_identity(config).unwrap()] },
            "provenance": { "camdl_version": camdl_version },
        });
        std::fs::write(leaf.join("run.json"), rec.to_string()).unwrap();
        leaf
    }

    /// `0.1.0+abc1234` → `0.1.0+abc1234.dirty-00000000`: the running build's
    /// commit with other uncommitted changes.
    fn same_commit_dirty() -> String {
        let base = crate::version::VERSION_SHORT.split(".dirty-").next().unwrap();
        format!("{base}.dirty-00000000")
    }

    /// Proposal 2026-11-17 §1d: a `fit.toml` names the run this build made
    /// from it, exactly. A run of the same file by another build is another
    /// computation: it is never used in its place, even when it is the only
    /// one; it is listed with how it differs and how to name it.
    #[test]
    fn a_config_whose_only_run_is_another_builds_is_refused_naming_it() {
        let (tmp, toml_path, seg) = portable_project("other_build_only");
        std::fs::write(&toml_path, ARCHIVED).unwrap();
        let stale = plant_run_built_by(&seg, "if2-00000000", &"ab".repeat(32), ARCHIVED,
            &"0".repeat(32), "0.0.1+0000000");
        let err = resolve_config(&toml_path).map(|r| r.0.dir)
            .expect_err("a run by another build is not this file's run under this build")
            .to_string();
        assert!(err.contains("no completed run of") && err.contains("by this camdl build"), "{err}");
        let _ = stale;
        assert!(err.contains("if2-00000000/seed_1-00000000"), "lists the run's leaf: {err}");
        assert!(err.contains("run abababababab"), "names its run id: {err}");
        assert!(err.contains("stale: built by 0.0.1+0000000"), "says how its build differs: {err}");
        assert!(err.contains("run-id prefix") && err.contains("leaf directory"), "says how to name one: {err}");

        // The same commit with other uncommitted changes is named as such.
        let dirty = plant_run_built_by(&seg, "if2-11111111", &"cd".repeat(32), ARCHIVED,
            &"1".repeat(32), &same_commit_dirty());
        let err = resolve_config(&toml_path).map(|r| r.0.dir).expect_err("two other builds").to_string();
        assert!(err.contains(&format!("built from uncommitted changes: {}", same_commit_dirty())), "{err}");
        let _ = dirty;
        assert!(err.contains("if2-11111111/seed_1-00000000") && err.contains("if2-00000000/seed_1-00000000"), "{err}");
        let _ = std::fs::remove_dir_all(&tmp);
    }

    /// The run this build made is the answer, whatever other builds left
    /// beside it.
    #[test]
    fn a_config_resolves_to_this_builds_run_beside_another_builds() {
        let (tmp, toml_path, seg) = portable_project("this_build_and_other");
        std::fs::write(&toml_path, ARCHIVED).unwrap();
        plant_run_built_by(&seg, "if2-00000000", &"ab".repeat(32), ARCHIVED, &"0".repeat(32), "0.0.1+0000000");
        let mine = plant_run_built_by(&seg, "if2-11111111", &"cd".repeat(32), ARCHIVED,
            crate::version::ENGINE_ID, crate::version::VERSION_SHORT);
        let got = resolve_config(&toml_path).map(|r| r.0.dir).unwrap_or_else(|e| panic!("resolves: {e}"));
        assert_eq!(std::fs::canonicalize(got).unwrap(), std::fs::canonicalize(mine).unwrap());
        let _ = std::fs::remove_dir_all(&tmp);
    }

    /// A run another build made whose readable version is this build's (same
    /// commit, same uncommitted changes) differs in its build configuration,
    /// dependencies included: an uncommitted `Cargo.lock` change moves
    /// `ENGINE_ID` and not the label. The advice leads with the leaf
    /// directory, which names a run from any store; a run-id prefix is looked
    /// for under `./results` only.
    #[test]
    fn another_builds_run_with_this_builds_label_names_its_configuration() {
        let (tmp, toml_path, seg) = portable_project("same_label_other_engine");
        std::fs::write(&toml_path, ARCHIVED).unwrap();
        plant_run_built_by(&seg, "if2-00000000", &"ab".repeat(32), ARCHIVED,
            &"0".repeat(32), crate::version::VERSION_SHORT);
        let err = resolve_config(&toml_path).map(|r| r.0.dir).expect_err("another build").to_string();
        assert!(err.contains("another build configuration") && err.contains("dependencies"), "{err}");
        let dir = err.find("leaf directory").unwrap_or_else(|| panic!("{err}"));
        let prefix = err.find("run-id prefix").unwrap_or_else(|| panic!("{err}"));
        assert!(dir < prefix, "the leaf directory is the advice that always works: {err}");
        let _ = std::fs::remove_dir_all(&tmp);
    }

    /// A run this build made with an engine switch on is not the run `fit run`
    /// of the file makes in a process without it, and it is not another
    /// build's: `run.json`'s `engine_version` is the build alone, and the
    /// listing names the switch.
    #[test]
    fn this_builds_run_with_an_engine_switch_is_named_by_its_switch() {
        let (tmp, toml_path, seg) = portable_project("this_build_switch");
        std::fs::write(&toml_path, ARCHIVED).unwrap();
        let leaf = plant_run_built_by(&seg, "if2-00000000", &"ab".repeat(32), ARCHIVED,
            crate::version::ENGINE_ID, crate::version::VERSION_SHORT);
        let record = leaf.join("run.json");
        let mut rec: serde_json::Value = serde_json::from_slice(&std::fs::read(&record).unwrap()).unwrap();
        rec["engine_switches"] = serde_json::json!([{ "name": "CAMDL_EVAL_UNRESOLVED" }]);
        std::fs::write(&record, rec.to_string()).unwrap();
        assert!(crate::env::get().engine_switches().is_empty(), "the test process sets no engine switch");
        let err = resolve_config(&toml_path).map(|r| r.0.dir).expect_err("another engine").to_string();
        assert!(err.contains("made with other engine switches: CAMDL_EVAL_UNRESOLVED"), "{err}");
        assert!(!err.contains("stale") && !err.contains("another build configuration"), "not another build: {err}");
        let _ = std::fs::remove_dir_all(&tmp);
    }
}
