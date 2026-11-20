//! `camdl dev migrate`: bring the folders of a store an older camdl wrote up
//! to the current store layout ([`runid::STORE_LAYOUT`]), in place. The
//! layout, the command's guarantees (derive only from what a folder records,
//! refuse rather than guess, atomic per step, a marked header) and how to add
//! or retire a step are specified in `docs/camdl-run-spec.md` §2.7; this
//! module is their one implementation, and the only code that reads a folder
//! at an older layout. Readers refuse one through [`check_layout`].

use std::path::{Path, PathBuf};

use runid::{ArtifactKind, STORE_LAYOUT};

use crate::run_meta::{FitSidecar, OriginalError, ARCHIVED_CONFIG, RESOLVED_CONFIG};

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

/// Layout 0 to 1: a fit folder that keeps its fit file gains
/// `fit.resolved.toml`, built by the function `fit run` builds it with
/// ([`crate::fit::run_config::resolved_config_text_with`]); every other folder
/// has its layout raised alone. What each kind gets, and from which records,
/// is the run spec's §2.7, "Step 0 to 1".
mod step0 {
    use super::*;

    /// The header of a `fit.resolved.toml` this step writes.
    pub(super) const HEADER: &str = "\
# fit.resolved.toml: the configuration the fits in this folder ran under.
# Written by `camdl dev migrate` (store layout 0 to 1), not by `camdl fit run`,
# from what this folder recorded: fit.toml.original (the fit file as read; its
# SHA-256 is fit_toml_hash in fit.meta.json), the [fixed] values fit.meta.json
# records the fit running at (its partition), and whether the runs' run.json
# record forcing extrapolation as allowed. Below this header it is the text
# `camdl fit run` writes for the same records (docs/camdl-run-spec.md, 2.7).
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
        let kinds = crate::cas_read::kinds_under(dir);
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
        match crate::run_meta::read_verified_original(dir, &side.fit_toml_hash) {
            Ok(_) => Ok(()),
            Err(OriginalError::Unreadable { path, error }) => Err(format!(
                "fit.meta.json names a fit file (source_config) but {} cannot be read: {error}",
                path.display()
            )),
            Err(OriginalError::NoRecordedHash { .. }) => Err("fit.meta.json records no fit_toml_hash, so \
                fit.toml.original cannot be checked to be the file the fit read"
                .to_string()),
            Err(OriginalError::Mismatch { found, recorded, bytes }) => {
                if stored_with_the_flag_folded_in(dir, &bytes)? {
                    return refuse_a_folded_original();
                }
                Err(format!(
                    "fit.toml.original is not the file the fit read: its SHA-256 is {found}, and \
                     fit.meta.json records {recorded} (fit_toml_hash)"
                ))
            }
        }
    }

    /// Whether `original` (a `fit.toml.original` whose hash is not the
    /// recorded `fit_toml_hash`) is the file as main stored it for a fit run
    /// with `--allow-forcing-extrapolation`, from c60d0401c until the
    /// resolved config: the file parsed, `[config] allow_forcing_extrapolation
    /// = true` set, and the table written back, so the text is its own
    /// re-serialisation, it allows extrapolation, and the folder's runs record
    /// it as allowed. `fit_toml_hash` hashes the file as read, which no record
    /// in the folder holds.
    fn stored_with_the_flag_folded_in(dir: &Path, original: &[u8]) -> Result<bool, String> {
        let Ok(text) = std::str::from_utf8(original) else { return Ok(false) };
        let Ok(doc) = text.parse::<toml::Table>() else { return Ok(false) };
        let rewritten = toml::to_string(&doc).is_ok_and(|t| t == text);
        let allows = doc.get("config")
            .and_then(|c| c.get("allow_forcing_extrapolation"))
            .and_then(toml::Value::as_bool)
            == Some(true);
        Ok(rewritten && allows && runs_allow_forcing_extrapolation(dir)? == Some(true))
    }

    /// What step 0 to 1 does with a folder whose `fit.toml.original` main
    /// stored with `--allow-forcing-extrapolation` folded in
    /// ([`stored_with_the_flag_folded_in`]). The file the fit read is not in
    /// the folder, so it cannot be checked against `fit_toml_hash`, and the
    /// folder is refused. A rescue of such folders (deriving the resolved
    /// configuration from the folded file, if the maintainer accepts that
    /// the file is not checked) belongs here and nowhere else.
    fn refuse_a_folded_original() -> Result<(), String> {
        Err("it was written by a camdl that folded --allow-forcing-extrapolation into the fit \
             file it stored: its fit.toml.original is that file re-written with the flag set, \
             while fit_toml_hash records the file as the fit read it, which the folder does not \
             keep, so the stored file cannot be checked to be the fit's. Re-run the fit with this \
             camdl."
            .to_string())
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
        // The steps are consecutive from the oldest kept (checked above and
        // by `the_steps_run_from_the_oldest_kept_to_the_current_layout`), and
        // each raises the layout by one.
        debug_assert_eq!(folder.layout(), step.from);
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
        Folder::Record { dir, side: walked } => {
            // The record as it is now, not as the walk read it: another writer
            // may have rewritten it since.
            let mut side = crate::run_meta::read_fit_sidecar(&dir)?
                .ok_or_else(|| format!("{} has no fit.meta.json", dir.display()))?;
            if side.store_layout != walked.store_layout {
                return Err(format!(
                    "its fit.meta.json changed to layout {} while migrating",
                    side.store_layout
                ));
            }
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

    /// Raising a fit folder's layout rewrites its `fit.meta.json` as it is
    /// then, not as the walk read it: a field another writer changed in
    /// between (here `estimated`) survives.
    #[test]
    fn raising_the_layout_rewrites_the_sidecar_as_it_is_on_disk() {
        let dir = crate::test_support::unique_temp_dir("migrate_reread");
        std::fs::create_dir_all(&dir).unwrap();
        crate::run_meta::store_fit_sidecar(&dir, &FitSidecar::default()).unwrap();
        let side = crate::run_meta::read_fit_sidecar(&dir).unwrap().unwrap();
        assert_eq!(side.store_layout, 0);
        let walked = Folder::Record { dir: dir.clone(), side };
        let mut changed = crate::run_meta::read_fit_sidecar(&dir).unwrap().unwrap();
        changed.estimated = vec!["written meanwhile".to_string()];
        crate::run_meta::store_fit_sidecar(&dir, &changed).unwrap();
        let after = match apply(walked, 1, &[]).unwrap() {
            Folder::Record { side, .. } => side,
            Folder::Leaf { .. } => unreachable!(),
        };
        assert_eq!(after.store_layout, 1);
        assert_eq!(after.estimated, vec!["written meanwhile".to_string()]);
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// A fit whose roles formed no partition recorded `partition` as
    /// `unresolved` and its `[fixed]` resolved as `fixed`; step 0 to 1 takes
    /// the folder's `[fixed]` from that field, and verifies it.
    #[test]
    fn an_unresolved_partition_takes_the_fixed_values_the_record_holds() {
        let root = crate::test_support::unique_temp_dir("migrate_unresolved");
        let seg = root.join("fits").join("fit-00000000");
        std::fs::create_dir_all(&seg).unwrap();
        // The file allows forcing extrapolation itself, so no run record is
        // needed to say whether the command line did.
        let original = "[model]\ncamdl = \"m.camdl\"\n[estimate]\nbeta = { bounds = [0.05, 2.0] }\n\
                        [fixed]\ngamma = 0.2\nrho = 0.5\n[config]\nallow_forcing_extrapolation = true\n";
        std::fs::write(root.join("fit.toml"), original).unwrap();
        std::fs::write(seg.join(ARCHIVED_CONFIG), original).unwrap();
        let meta = serde_json::json!({
            "schema_version": crate::run_meta::FIT_META_SCHEMA_VERSION,
            "source_config": "../../fit.toml",
            "model_identity": "",
            "fit_toml_hash": crate::hashing::sha256_hex(original.as_bytes()),
            "estimated": ["beta"],
            "fixed": { "gamma": 0.2, "rho": 0.5 },
            "partition": { "status": "unresolved", "reason": "a role the rules refuse" },
            "calendar": { "days_per_unit": 1.0, "origin": null, "time_unit": "days" },
        });
        std::fs::write(seg.join("fit.meta.json"), meta.to_string()).unwrap();
        let side = crate::run_meta::read_fit_sidecar(&seg).unwrap().unwrap();
        assert!(matches!(side.partition, Some(crate::roles::PartitionField::Unresolved { .. })));
        let folder = Folder::Record { dir: seg.clone(), side };
        step0::check(&folder).unwrap();
        let writes = step0::derive(&folder).unwrap();
        step0::verify(&folder, &writes).unwrap();
        let derived = crate::fit::config_v2::FitConfig::from_toml_str(&writes[0].text).unwrap();
        let fixed = derived.problem.fixed.resolve().unwrap();
        assert_eq!((fixed.get("gamma"), fixed.get("rho"), fixed.len()), (Some(&0.2), Some(&0.5), 2), "{}", writes[0].text);

        // A record whose `fixed` disagrees with the file's inline value is
        // refused, not written.
        let mut wrong = meta.clone();
        wrong["fixed"]["rho"] = serde_json::json!(0.6);
        std::fs::write(seg.join("fit.meta.json"), wrong.to_string()).unwrap();
        let side = crate::run_meta::read_fit_sidecar(&seg).unwrap().unwrap();
        let folder = Folder::Record { dir: seg.clone(), side };
        let writes = step0::derive(&folder).unwrap();
        let err = step0::verify(&folder, &writes).unwrap_err();
        assert!(err.contains("rho"), "{err}");
        let _ = std::fs::remove_dir_all(&root);
    }

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
