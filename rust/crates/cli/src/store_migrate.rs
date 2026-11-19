//! `camdl dev migrate`: bring the folders of a store an older camdl wrote up
//! to the current store layout ([`runid::STORE_LAYOUT`]), in place.
//!
//! Every folder of a store records the layout it was written at: a leaf in
//! its `run.json` (`store_layout`), a fit or profile folder in its
//! `fit.meta.json`. A record without the field is layout 0. A reader of a
//! stored fit's configuration refuses a folder at any layout but the current
//! one ([`check_layout`]); this module is the only code that reads an older
//! one.
//!
//! **Steps.** [`STEPS`] lists one [`Step`] per layout number, `n` to `n + 1`,
//! in order. A step has a precondition ([`Step::check`]), a derivation that
//! reads only what the folder already records ([`Step::derive`], which returns
//! the files to write), and a verification of what it derived
//! ([`Step::verify`]), run before anything is written. A folder is taken
//! through every step from its layout to the current one.
//!
//! **Never a guess.** A folder whose records are missing or disagree is
//! refused with the reason and left as it was; the others are migrated, and
//! the command exits non-zero.
//!
//! **Atomic per step.** Each file a step writes goes to a temporary file in
//! the same directory and is renamed over its name
//! ([`runid::write_file_atomic`]); the folder's layout is raised last, in its
//! record, the same way. A crash leaves the folder at its old layout, with at
//! most a file the step writes again on the next run (it verifies a file
//! already there against what it derives).
//!
//! **Marked.** A file a migration writes opens with a header naming the step
//! and the records it was derived from, so it is told apart from one the
//! command that made the folder wrote.
//!
//! **Adding a step.** Raise [`runid::STORE_LAYOUT`], append a [`Step`] whose
//! `from` is the previous layout, and make every reader of what the new
//! layout adds check it ([`check_layout`]). **Retiring steps.** Before 1.0 a
//! step older than the previous release may be dropped from the front of
//! [`STEPS`]; a folder older than the oldest step kept is refused with its
//! layout number ([`oldest_migratable`]).

use std::path::{Path, PathBuf};

use runid::{ArtifactKind, STORE_LAYOUT};

use crate::run_meta::{FitSidecar, ARCHIVED_CONFIG, RESOLVED_CONFIG};

/// `Err` naming `folder` when its recorded layout `found` is not the current
/// one: an older folder with the command that brings it up to date, a newer
/// one with the camdl that can read it.
pub fn check_layout(folder: &Path, found: u32) -> Result<(), String> {
    match found.cmp(&STORE_LAYOUT) {
        std::cmp::Ordering::Equal => Ok(()),
        std::cmp::Ordering::Less => Err(format!(
            "{} is at store layout {found}, older than this camdl's layout {STORE_LAYOUT}: it was \
             written by an older camdl, and lacks what a reader here needs ({}). Run \
             `camdl dev migrate {}`, which adds it from what the folder records.",
            folder.display(),
            STEPS.iter().filter(|s| s.from >= found).map(|s| s.adds).collect::<Vec<_>>().join("; "),
            folder.display()
        )),
        std::cmp::Ordering::Greater => Err(format!(
            "{} is at store layout {found}, written by a newer camdl than this one, which reads \
             layout {STORE_LAYOUT}. Read it with that camdl or a newer one.",
            folder.display()
        )),
    }
}

/// The oldest layout [`STEPS`] can migrate from.
pub fn oldest_migratable() -> u32 {
    STEPS.first().map_or(STORE_LAYOUT, |s| s.from)
}

/// One folder of a store, as its record says.
#[derive(Debug)]
pub enum Folder {
    /// A leaf: a directory holding a `run.json`.
    Leaf { dir: PathBuf, record: runid::RunRecord },
    /// A fit or profile folder: a directory holding a `fit.meta.json`, the
    /// record of a folder with no `run.json`.
    Record { dir: PathBuf, side: FitSidecar },
}

impl Folder {
    pub fn dir(&self) -> &Path {
        match self {
            Folder::Leaf { dir, .. } | Folder::Record { dir, .. } => dir,
        }
    }

    pub fn layout(&self) -> u32 {
        match self {
            Folder::Leaf { record, .. } => record.store_layout,
            Folder::Record { side, .. } => side.store_layout,
        }
    }
}

/// A file a step writes, relative to the folder.
#[derive(Debug, Clone, PartialEq)]
pub struct Write {
    pub name: &'static str,
    pub text: String,
}

/// One step of the migration, from layout `from` to `from + 1`.
pub struct Step {
    pub from: u32,
    /// What the step adds, in a phrase, for messages.
    pub adds: &'static str,
    /// The precondition: `Err` when the folder cannot take this step.
    pub check: fn(&Folder) -> Result<(), String>,
    /// The files the step writes, derived from what the folder records alone.
    pub derive: fn(&Folder) -> Result<Vec<Write>, String>,
    /// `Err` when what [`Self::derive`] produced is not what the folder's
    /// records say it must be. Run before anything is written.
    pub verify: fn(&Folder, &[Write]) -> Result<(), String>,
}

/// Every step, in order. `STEPS[i].from + 1 == STEPS[i + 1].from`, and the
/// last step ends at [`runid::STORE_LAYOUT`].
pub static STEPS: &[Step] = &[Step {
    from: 0,
    adds: "a fit folder's fit.resolved.toml, the configuration its fits ran under",
    check: step0::check,
    derive: step0::derive,
    verify: step0::verify,
}];

/// Layout 0 to 1: a fit folder that keeps its fit file (`fit.toml.original`)
/// gains `fit.resolved.toml`.
///
/// What each folder kind gets:
///
/// - A fit folder (its leaves are fit runs) with a fit file: the resolved
///   configuration, built by the function `fit run` builds it with
///   ([`crate::fit::run_config::resolved_config_text_with`]) from three
///   records: `fit.toml.original` (checked against `fit_toml_hash`), the
///   `[fixed]` values `fit.meta.json`'s partition records the folder running
///   at, and whether its runs' `run.json` record forcing extrapolation as
///   allowed (the one command-line value of the problem half,
///   [`crate::fit::run_config::ProblemFlags`]). It is verified as a reader
///   parses it, against the recorded `[fixed]` and estimated names, and, when
///   the folder already holds the file, against that file.
/// - A fit folder made without a fit file (a CLI-only fit) holds no
///   configuration: the layout is raised alone.
/// - A profile folder: the layout is raised alone. No reader takes a
///   configuration from a profile folder, and its resolved configuration
///   cannot be derived from what it records (a profile of a stored fit stores
///   that fit's, with the fit's flags rather than its own), so none is
///   written; a profile folder migrated from layout 0 keeps only its
///   `fit.toml.original`.
/// - A leaf: the layout is raised alone; nothing a leaf holds changed.
mod step0 {
    use super::*;

    /// The header of a `fit.resolved.toml` this step writes.
    pub(super) const HEADER: &str = "\
# fit.resolved.toml: the configuration the fits in this folder ran under.
# Written by `camdl dev migrate` (store layout 0 to 1), not by `camdl fit run`,
# from what this folder recorded: fit.toml.original (the fit file as read; its
# SHA-256 is fit_toml_hash in fit.meta.json), the [fixed] values fit.meta.json
# records the fit running at (its partition), and whether the runs' run.json
# record forcing extrapolation as allowed (--allow-forcing-extrapolation, the
# one command-line value of the problem half). Below this header it is the
# text `camdl fit run` writes for the same records: relative [model], [data]
# and output_dir paths resolve against the fit file's directory (source_config
# in fit.meta.json), a reader binds each [data.tables] input to its copy in
# inputs/, [estimate], [data.observations] and [data.holdout] keep the file's
# order and every other table is sorted, and [method] is the file's.
";

    /// What a folder is, for this step.
    enum Kind<'a> {
        /// A fit folder with a fit file: the records the derivation reads.
        Fit { dir: &'a Path, side: &'a FitSidecar, source_config: &'a str },
        /// Nothing to write: a leaf, a CLI-only fit, a profile folder.
        RaiseOnly,
    }

    fn kind(folder: &Folder) -> Result<Kind<'_>, String> {
        let (dir, side) = match folder {
            Folder::Leaf { .. } => return Ok(Kind::RaiseOnly),
            Folder::Record { dir, side } => (dir.as_path(), side),
        };
        let Some(source_config) = side.source_config.as_deref() else { return Ok(Kind::RaiseOnly) };
        let kinds: std::collections::HashSet<ArtifactKind> = runid::discover_in_walk_order(dir, runid::Scope::default())
            .into_iter()
            .filter_map(|l| l.ok().map(|l| l.record.kind))
            .collect();
        let profile = kinds.contains(&ArtifactKind::ProfilePoint);
        let fit = kinds.contains(&ArtifactKind::FitMethod);
        match (fit, profile) {
            (true, true) => Err("it holds both fit runs and profile points, so it is neither a fit folder \
                                 nor a profile folder"
                .to_string()),
            (false, true) => Ok(Kind::RaiseOnly),
            _ => Ok(Kind::Fit { dir, side, source_config }),
        }
    }

    pub(super) fn check(folder: &Folder) -> Result<(), String> {
        let Kind::Fit { dir, side, .. } = kind(folder)? else { return Ok(()) };
        let original = dir.join(ARCHIVED_CONFIG);
        let bytes = std::fs::read(&original).map_err(|e| {
            format!("fit.meta.json names a fit file (source_config) but {} cannot be read: {e}", original.display())
        })?;
        if side.fit_toml_hash.is_empty() {
            return Err("fit.meta.json records no fit_toml_hash, so fit.toml.original cannot be checked \
                        to be the file the fit read"
                .to_string());
        }
        let found = crate::hashing::sha256_hex(&bytes);
        if found != side.fit_toml_hash {
            return Err(format!(
                "fit.toml.original is not the file the fit read: its SHA-256 is {found}, and \
                 fit.meta.json records {} (fit_toml_hash)",
                side.fit_toml_hash
            ));
        }
        Ok(())
    }

    /// The `[fixed]` values the folder's fits ran at, from its partition: the
    /// parameters its `[fixed]` declared (inline, from a file or from a
    /// scenario) and a sweep point's design parameters, each at its recorded
    /// value. A fit whose roles formed no partition recorded `fixed` as its
    /// `[fixed]` resolved, which is read instead.
    fn ran_fixed(side: &FitSidecar) -> Result<indexmap::IndexMap<String, f64>, String> {
        use crate::roles::{ParamRecord, PartitionField, RoleKind};
        match &side.partition {
            Some(PartitionField::Resolved(rec)) => {
                let mut out = indexmap::IndexMap::new();
                for (name, p) in &rec.parameters {
                    let ParamRecord::Fixed { job_role, declared, value, .. } = p else { continue };
                    let from_config = declared.declared_by == "fit_fixed" || *job_role == Some(RoleKind::Design);
                    if !from_config {
                        continue;
                    }
                    let v = value.ok_or_else(|| {
                        format!("fit.meta.json's partition records `{name}` as fixed by the fit config with no value")
                    })?;
                    out.insert(name.clone(), v);
                }
                Ok(out)
            }
            Some(PartitionField::Unresolved { .. }) => Ok(side.fixed.iter().map(|(k, v)| (k.clone(), *v)).collect()),
            None => Err("fit.meta.json records no partition, so which fixed values came from the fit config \
                         is not recorded"
                .to_string()),
        }
    }

    /// Whether the runs in `dir` were allowed to extrapolate forcings, as
    /// every fit run's `run.json` records it (`inputs.forcing_extrapolation`,
    /// written only when allowed). All of a folder's runs share its problem,
    /// so they must agree. `None` when the folder holds no readable fit run.
    fn runs_allow_forcing_extrapolation(dir: &Path) -> Result<Option<bool>, String> {
        let scope = runid::Scope { kind: Some(ArtifactKind::FitMethod), ..Default::default() };
        let mut seen: Option<(bool, PathBuf)> = None;
        for leaf in runid::discover_in_walk_order(dir, scope).into_iter().flatten() {
            let allowed = leaf.record.inputs.get("forcing_extrapolation")
                .and_then(|f| f.get("allowed"))
                .and_then(serde_json::Value::as_bool)
                .unwrap_or(false);
            match &seen {
                Some((first, at)) if *first != allowed => {
                    return Err(format!(
                        "its runs disagree on whether forcing extrapolation was allowed: {} says {first}, \
                         {} says {allowed}",
                        at.display(),
                        leaf.dir.display()
                    ));
                }
                Some(_) => {}
                None => seen = Some((allowed, leaf.dir.clone())),
            }
        }
        Ok(seen.map(|(a, _)| a))
    }

    pub(super) fn derive(folder: &Folder) -> Result<Vec<Write>, String> {
        let Kind::Fit { dir, side, .. } = kind(folder)? else { return Ok(Vec::new()) };
        let original = std::fs::read_to_string(dir.join(ARCHIVED_CONFIG))
            .map_err(|e| format!("cannot read {}: {e}", dir.join(ARCHIVED_CONFIG).display()))?;
        let file = crate::fit::config_v2::FitConfig::from_toml_str(&original)
            .map_err(|e| format!("fit.toml.original does not parse: {e}"))?;
        let file_allows = file.problem.config.allow_forcing_extrapolation;
        let runs_allow = runs_allow_forcing_extrapolation(dir)?;
        let flag = match (file_allows, runs_allow) {
            (true, Some(false)) => {
                return Err("fit.toml.original allows forcing extrapolation and its runs record it as \
                            not allowed"
                    .to_string());
            }
            (true, _) => false,
            (false, Some(allowed)) => allowed,
            (false, None) => {
                return Err("it holds no readable fit run, so whether the command line allowed forcing \
                            extrapolation is not recorded"
                    .to_string());
            }
        };
        let flags = crate::fit::run_config::ProblemFlags { allow_forcing_extrapolation: flag };
        let fixed = ran_fixed(side)?;
        let text = crate::fit::run_config::resolved_config_text_with(HEADER, &original, flags, &fixed)?;
        Ok(vec![Write { name: RESOLVED_CONFIG, text }])
    }

    pub(super) fn verify(folder: &Folder, writes: &[Write]) -> Result<(), String> {
        let Kind::Fit { dir, side, source_config } = kind(folder)? else {
            return if writes.is_empty() { Ok(()) } else { Err("a folder with nothing to derive derived a file".into()) };
        };
        let [Write { text, .. }] = writes else {
            return Err(format!("expected to derive {RESOLVED_CONFIG} alone, derived {} files", writes.len()));
        };
        // As a reader parses it, anchored where the fit file was.
        let path = dir.join(RESOLVED_CONFIG);
        let config = crate::fit::config_v2::FitConfig::parse_anchored_at(text, &path, &dir.join(source_config))
            .map_err(|e| format!("the derived {RESOLVED_CONFIG} does not parse: {e}"))?;
        crate::fit::handle::refuse_fixed_not_run(&config, side, dir)?;
        let derived_fixed = config.problem.fixed.resolve()?;
        let recorded: std::collections::BTreeSet<&str> = side.estimated.iter().map(String::as_str).collect();
        let derived: std::collections::BTreeSet<&str> = config.problem.estimate.keys().map(String::as_str).collect();
        if recorded != derived {
            return Err(format!(
                "the derived configuration estimates {derived:?}, and fit.meta.json records {recorded:?}"
            ));
        }
        // Every value the fit file writes inline in [fixed] is in the derived
        // [fixed], at the file's value unless a sweep point set it.
        let original = std::fs::read_to_string(dir.join(ARCHIVED_CONFIG)).map_err(|e| e.to_string())?;
        let file = crate::fit::config_v2::FitConfig::from_toml_str(&original).map_err(|e| e.to_string())?;
        let design = design_names(side);
        for (name, v) in &file.problem.fixed.values {
            match derived_fixed.get(name) {
                Some(d) if d == v || design.contains(name) => {}
                Some(d) => {
                    return Err(format!(
                        "fit.toml.original fixes {name} = {v} and the partition records {d}, with no sweep point \
                         setting it"
                    ));
                }
                None => return Err(format!("fit.toml.original fixes {name}, and the partition records no value for it")),
            }
        }
        // A file already there: one the fit wrote, or this step on a run that
        // stopped before raising the layout. Either says what was derived.
        if let Ok(existing) = std::fs::read_to_string(&path) {
            let body = |t: &str| -> String { t.lines().skip_while(|l| l.starts_with('#')).collect::<Vec<_>>().join("\n") };
            if body(&existing) != body(text) {
                return Err(format!(
                    "{} already exists and says something other than its records: re-deriving it gives \
                     another configuration",
                    path.display()
                ));
            }
        }
        Ok(())
    }

    /// The design parameters of the folder's sweep point, from its partition.
    fn design_names(side: &FitSidecar) -> std::collections::BTreeSet<String> {
        use crate::roles::{ParamRecord, PartitionField, RoleKind};
        match &side.partition {
            Some(PartitionField::Resolved(rec)) => rec.parameters.iter()
                .filter(|(_, p)| matches!(p, ParamRecord::Fixed { job_role: Some(RoleKind::Design), .. }))
                .map(|(n, _)| n.clone())
                .collect(),
            _ => Default::default(),
        }
    }
}

/// What happened to one folder.
#[derive(Debug, PartialEq)]
pub enum Outcome {
    /// Already at the current layout.
    Current,
    /// Taken from `from` to the current layout, writing `wrote`. Under
    /// `--dry-run`, what would be.
    Migrated { from: u32, wrote: Vec<&'static str> },
    /// Left as it was, for `reason`.
    Refused { reason: String },
    /// A record the store treats as stale (an older record format or hash
    /// version): recomputed rather than read, so it has no layout to raise.
    Stale { reason: String },
}

/// Every folder at or under `path`, with a reason for each directory that
/// looks like one but whose record cannot be read.
pub fn folders(path: &Path) -> Vec<Result<Folder, (PathBuf, Outcome)>> {
    let mut out = Vec::new();
    let mut stack = vec![path.to_path_buf()];
    while let Some(dir) = stack.pop() {
        let Ok(entries) = std::fs::read_dir(&dir) else { continue };
        let mut children: Vec<PathBuf> = entries
            .flatten()
            .filter(|e| e.file_type().is_ok_and(|t| t.is_dir()))
            .filter(|e| {
                let n = e.file_name();
                !n.to_string_lossy().starts_with('.') && n != runid::ATTEMPTS_DIR
            })
            .map(|e| e.path())
            .collect();
        children.sort();
        children.reverse();
        stack.extend(children);
        if dir.join("fit.meta.json").is_file() {
            out.push(match crate::run_meta::read_fit_sidecar(&dir) {
                Ok(Some(side)) => Ok(Folder::Record { dir, side }),
                Ok(None) => continue,
                Err(e) => Err((dir, Outcome::Refused { reason: e })),
            });
        } else if dir.join("run.json").is_file() || dir.join(".lock").is_file() {
            out.push(match runid::read_leaf(&dir) {
                Ok(Some(leaf)) => {
                    let r = &leaf.record;
                    if r.format_version != runid::FORMAT_VERSION || r.hash_version != runid::HASH_VERSION {
                        Err((dir, Outcome::Stale {
                            reason: format!(
                                "run.json is format {} / hash version {}; this camdl recomputes such a run \
                                 rather than reading it",
                                r.format_version, r.hash_version
                            ),
                        }))
                    } else {
                        Ok(Folder::Leaf { dir, record: leaf.record })
                    }
                }
                Ok(None) => continue,
                Err(u) => match u.reason {
                    runid::Unreadable::RenamedKind { .. } | runid::Unreadable::PredatesCalendar => {
                        Err((dir, Outcome::Stale { reason: u.reason.to_string() }))
                    }
                    other => Err((dir, Outcome::Refused { reason: other.to_string() })),
                },
            });
        }
    }
    out
}

/// Take `folder` through every step from its layout to the current one, or
/// under `dry_run` derive and verify the first and write nothing.
pub fn migrate(folder: Folder, dry_run: bool) -> Outcome {
    let from = folder.layout();
    if from == STORE_LAYOUT {
        return Outcome::Current;
    }
    if from > STORE_LAYOUT {
        return Outcome::Refused {
            reason: format!("it is at store layout {from}, written by a newer camdl (this one writes {STORE_LAYOUT})"),
        };
    }
    if from < oldest_migratable() {
        return Outcome::Refused {
            reason: format!(
                "it is at store layout {from}, older than the oldest this camdl migrates ({}); re-run what \
                 made it, or migrate it with an older camdl first",
                oldest_migratable()
            ),
        };
    }
    let mut folder = folder;
    let mut wrote = Vec::new();
    for step in STEPS.iter().filter(|s| s.from >= from) {
        if folder.layout() != step.from {
            return Outcome::Refused {
                reason: format!("it is at layout {}, and the next step starts at {}", folder.layout(), step.from),
            };
        }
        let writes = match (step.check)(&folder)
            .and_then(|()| (step.derive)(&folder))
            .and_then(|w| (step.verify)(&folder, &w).map(|()| w))
        {
            Ok(w) => w,
            Err(reason) => return Outcome::Refused { reason: format!("layout {} to {}: {reason}", step.from, step.from + 1) },
        };
        wrote.extend(writes.iter().map(|w| w.name));
        if dry_run {
            // A later step would see this one's writes; under --dry-run the
            // first step is all that is derived.
            return Outcome::Migrated { from, wrote };
        }
        folder = match apply(folder, step.from + 1, &writes) {
            Ok(f) => f,
            Err(reason) => return Outcome::Refused { reason: format!("layout {} to {}: {reason}", step.from, step.from + 1) },
        };
    }
    Outcome::Migrated { from, wrote }
}

/// Write `writes` into the folder, each whole or not at all, and then raise
/// its recorded layout to `to`; the folder as its record now reads.
fn apply(folder: Folder, to: u32, writes: &[Write]) -> Result<Folder, String> {
    let dir = folder.dir().to_path_buf();
    for w in writes {
        // A file already there was verified against the derived one
        // ([`Step::verify`]): kept as written, so a file the folder's own
        // command wrote keeps its header.
        let path = dir.join(w.name);
        if path.exists() {
            continue;
        }
        runid::write_file_atomic(&path, w.text.as_bytes()).map_err(|e| e.to_string())?;
    }
    match folder {
        Folder::Leaf { dir, .. } => {
            runid::update_record(&dir, |r| {
                if r.store_layout + 1 != to {
                    return Err(format!("its run.json changed to layout {} while migrating", r.store_layout));
                }
                r.store_layout = to;
                Ok(true)
            })
            .map_err(|e| e.to_string())?;
            let leaf = runid::read_leaf(&dir).map_err(|u| u.reason.to_string())?
                .ok_or_else(|| format!("{} has no run.json after migrating", dir.display()))?;
            Ok(Folder::Leaf { dir, record: leaf.record })
        }
        Folder::Record { dir, mut side } => {
            side.store_layout = to;
            crate::run_meta::store_fit_sidecar(&dir, &side).map_err(|e| e.to_string())?;
            let side = crate::run_meta::read_fit_sidecar(&dir)?
                .ok_or_else(|| format!("{} has no fit.meta.json after migrating", dir.display()))?;
            if side.store_layout != to {
                return Err(format!("fit.meta.json reads layout {} after writing {to}", side.store_layout));
            }
            Ok(Folder::Record { dir, side })
        }
    }
}

/// `camdl dev migrate <path> [--dry-run]`.
pub fn cmd_migrate(a: &crate::args::MigrateArgs) {
    let out = crate::out::Out::new(false);
    if !a.path.is_dir() {
        out.fail(format!("error: {} is not a directory", a.path.display()));
    }
    let (mut current, mut migrated, mut stale, mut refused) = (0usize, 0usize, 0usize, 0usize);
    let verb = if a.dry_run { "would migrate" } else { "migrated" };
    for found in folders(&a.path) {
        let (dir, outcome) = match found {
            Ok(folder) => {
                let dir = folder.dir().to_path_buf();
                (dir, migrate(folder, a.dry_run))
            }
            Err(pair) => pair,
        };
        match outcome {
            Outcome::Current => current += 1,
            Outcome::Migrated { from, wrote } => {
                migrated += 1;
                let files = if wrote.is_empty() { String::new() } else { format!(", writing {}", wrote.join(", ")) };
                out.report_line(format_args!("{verb} {}: store layout {from} to {STORE_LAYOUT}{files}", dir.display()));
            }
            Outcome::Stale { reason } => {
                stale += 1;
                out.report_line(format_args!("left {}: {reason}", dir.display()));
            }
            Outcome::Refused { reason } => {
                refused += 1;
                out.report_line(format_args!("refused {}: {reason}", dir.display()));
            }
        }
    }
    out.report_line(format_args!(
        "{migrated} folder(s) {verb}, {current} already at layout {STORE_LAYOUT}, {stale} stale left as they \
         are, {refused} refused{}",
        if a.dry_run { " (dry run: nothing written)" } else { "" }
    ));
    if refused > 0 {
        std::process::exit(1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_steps_run_from_the_oldest_kept_to_the_current_layout() {
        for pair in STEPS.windows(2) {
            assert_eq!(pair[0].from + 1, pair[1].from, "the steps are consecutive");
        }
        assert_eq!(STEPS.last().map(|s| s.from + 1), Some(STORE_LAYOUT), "the last step ends at the current layout");
    }

    #[test]
    fn a_layout_other_than_the_current_one_is_refused_by_name() {
        let dir = Path::new("results/fits/sir-0123abcd");
        assert!(check_layout(dir, STORE_LAYOUT).is_ok());
        let older = check_layout(dir, 0).unwrap_err();
        assert!(older.contains("store layout 0") && older.contains("camdl dev migrate results/fits/sir-0123abcd"),
            "{older}");
        assert!(older.contains("fit.resolved.toml"), "names what the folder lacks: {older}");
        let newer = check_layout(dir, STORE_LAYOUT + 1).unwrap_err();
        assert!(newer.contains("written by a newer camdl"), "{newer}");
        assert!(!newer.contains("migrate"), "{newer}");
    }
}
