//! `camdl list`, `camdl show`, `camdl cat` — browse the content-addressable
//! store written by `camdl simulate`.
//!
//! All three walk `./results/sims/` by default. For alpha, walk is
//! unindexed — fast enough for thousands of runs. A persistent index
//! can be added later if needed.

use std::path::{Path, PathBuf};
use std::time::SystemTime;

use crate::out::paint::{self as style, Style};

use crate::cas_read;
use crate::util::fmt_relative_time;

// ── Entry types ──────────────────────────────────────────────────────────────

/// A new-format (`runid::RunRecord`) simulate leaf, prepared for the `list`
/// display. The kind-Sim filter happens in [`cas_read::walk_sim_leaves`].
struct SimRow {
    leaf: cas_read::Leaf,
    /// Path relative to the current working directory, copy-paste ready.
    rel_path: String,
    /// When the run was written (from `provenance.created_at`; falls back to
    /// filesystem mtime).
    created: SystemTime,
}

/// Discover the new-format sim leaves under `root/sims/` for `list`, bounded
/// by what the row filters below can already rule out.
fn discover_sim_rows(root: &str, bounds: cas_read::Bounds) -> Vec<SimRow> {
    let cwd = std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
    cas_read::walk_sim_leaves_bounded(Path::new(root), bounds)
        .into_iter()
        .map(|leaf| {
            let created = leaf
                .record
                .provenance
                .created_at
                .as_deref()
                .and_then(parse_iso8601)
                .unwrap_or_else(|| {
                    std::fs::metadata(&leaf.dir)
                        .and_then(|m| m.modified())
                        .unwrap_or(SystemTime::UNIX_EPOCH)
                });
            let rel_path = pathdiff_str(&leaf.dir, &cwd);
            SimRow { leaf, rel_path, created }
        })
        .collect()
}

/// A new-format pfilter-eval leaf, prepared for the `list` display.
struct PfilterRow {
    leaf: cas_read::Leaf,
    rel_path: String,
    created: SystemTime,
}

/// Discover the new-format pfilter leaves under `root/pfilters/` for `list`.
fn discover_pfilter_rows(root: &str) -> Vec<PfilterRow> {
    let cwd = std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
    cas_read::walk_pfilter_leaves(Path::new(root))
        .into_iter()
        .map(|leaf| {
            let created = leaf.record.provenance.created_at.as_deref()
                .and_then(parse_iso8601)
                .unwrap_or_else(|| {
                    std::fs::metadata(&leaf.dir).and_then(|m| m.modified())
                        .unwrap_or(SystemTime::UNIX_EPOCH)
                });
            let rel_path = pathdiff_str(&leaf.dir, &cwd);
            PfilterRow { leaf, rel_path, created }
        })
        .collect()
}

/// A new-format `SimEnsemble` leaf, prepared for the `list` display. The
/// kind-SimEnsemble filter happens in [`cas_read::walk_sim_ensemble_leaves`].
struct EnsembleRow {
    leaf: cas_read::Leaf,
    rel_path: String,
    created: SystemTime,
}

/// Discover the new-format ensemble leaves under `root/ensembles/` for `list`.
fn discover_ensemble_rows(root: &str) -> Vec<EnsembleRow> {
    let cwd = std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
    cas_read::walk_sim_ensemble_leaves(Path::new(root))
        .into_iter()
        .map(|leaf| {
            let created = leaf.record.provenance.created_at.as_deref()
                .and_then(parse_iso8601)
                .unwrap_or_else(|| {
                    std::fs::metadata(&leaf.dir).and_then(|m| m.modified())
                        .unwrap_or(SystemTime::UNIX_EPOCH)
                });
            let rel_path = pathdiff_str(&leaf.dir, &cwd);
            EnsembleRow { leaf, rel_path, created }
        })
        .collect()
}

// ── cmd_list ─────────────────────────────────────────────────────────────────

/// `--kind` filter: which of sims / fits / profiles / surveys / pfilters /
/// ensembles to surface. `All` is the default and includes every kind.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum KindFilter { Sim, Fit, Profile, Pfilter, Survey, Ensemble, All }

impl KindFilter {
    fn includes_sims(self)      -> bool { matches!(self, Self::Sim      | Self::All) }
    fn includes_fits(self)      -> bool { matches!(self, Self::Fit      | Self::All) }
    fn includes_profiles(self)  -> bool { matches!(self, Self::Profile  | Self::All) }
    fn includes_pfilters(self)  -> bool { matches!(self, Self::Pfilter  | Self::All) }
    fn includes_surveys(self)   -> bool { matches!(self, Self::Survey   | Self::All) }
    fn includes_ensembles(self) -> bool { matches!(self, Self::Ensemble | Self::All) }
}

pub fn cmd_list(a: &crate::args::ListArgs) {
    let out = crate::out::Out::new(false);
    // --parent=HASH: enumerate the grid-point × start runs of one
    // specific profile. Takes precedence over the default sim/fit
    // enumeration because it's a more specific request; the other
    // filters (since, limit, format) still apply.
    if let Some(parent_hash) = a.parent.as_ref() {
        list_profile_children(&out, &a.resolved_root().to_string_lossy(), parent_hash, a);
        return;
    }

    let root = a.resolved_root().to_string_lossy();
    let filter_since: Option<std::time::Duration> = a.since.as_ref().map(|d| d.0);
    let filter_kind = match a.kind.as_str() {
        "sim" | "simulate"      => KindFilter::Sim,
        "fit"                   => KindFilter::Fit,
        "profile" | "profiles"  => KindFilter::Profile,
        "pfilter" | "pfilters"  => KindFilter::Pfilter,
        "survey" | "surveys"    => KindFilter::Survey,
        "ensemble" | "ensembles" => KindFilter::Ensemble,
        _                       => KindFilter::All,
    };
    let format_json = a.format.as_deref() == Some("json");

    // Ensemble grouping: a multi-cell `simulate` writes N per-cell `Sim` leaves
    // AND one `SimEnsemble` that references them (deps). In any view that shows
    // ensembles, the ensemble row REPRESENTS its members — don't also print one
    // row per replicate/cell (that's the messy N-row spam). `--kind sim`
    // (ensembles excluded) keeps showing the individual leaves, since that's an
    // explicit request for the per-cell level.
    //
    // The ensembles are read FIRST because their `deps` are what let the sim
    // walk skip a member leaf without paying to parse it: on the store in
    // gh#699 the ensembles are 88 leaves in 262 directories and the members
    // they represent are 461,282 of 550,647 sim leaves. Membership is
    // determined over the whole ensemble set, before any `--limit`/`--since`
    // truncation, so the collapse does not depend on which ensembles print.
    let ensembles = if !filter_kind.includes_ensembles() {
        Vec::new()
    } else {
        discover_ensemble_rows(&root)
    };
    let ensemble_member_ids: std::collections::HashSet<runid::ContentHash> = ensembles
        .iter()
        .flat_map(|e| e.leaf.record.deps.iter().map(|d| d.run_id))
        .collect();

    // `--since` bounds DISCOVERY, not just the printed rows. A leaf
    // directory's mtime is an upper bound on the `created_at` of the record
    // inside it (the record is stamped before `run.json` is written into that
    // directory), so an mtime older than the cutoff proves the row would be
    // filtered out below — and the leaf need never be read. Only sims are
    // bounded this way; that is where the leaves are (550,647 of 550,881 in
    // gh#699), and the ensembles must stay unbounded because their `deps` have
    // to name every member regardless of the ensemble's own age.
    let since_cutoff = filter_since.and_then(|dur| SystemTime::now().checked_sub(dur));

    let runs = if !filter_kind.includes_sims() {
        Vec::new()
    } else {
        discover_sim_rows(
            &root,
            cas_read::Bounds { skip: Some(&ensemble_member_ids), not_before: since_cutoff },
        )
    };
    let fits = if !filter_kind.includes_fits() {
        Vec::new()
    } else {
        discover_fits(&root).unwrap_or_else(|e| { out.fail(format!("error: {}", e)); })
    };
    let profiles = if !filter_kind.includes_profiles() {
        Vec::new()
    } else {
        discover_profiles(&root).unwrap_or_else(|e| { out.fail(format!("error: {}", e)); })
    };
    let surveys = if !filter_kind.includes_surveys() {
        Vec::new()
    } else {
        discover_survey_rows(&root)
    };
    let pfilters = if !filter_kind.includes_pfilters() {
        Vec::new()
    } else {
        discover_pfilter_rows(&root)
    };

    let now = SystemTime::now();

    let mut filtered_runs: Vec<SimRow> = runs.into_iter()
        .filter(|r| a.model.as_deref().is_none_or(|m| r.leaf.level_label("model").contains(m)))
        .filter(|r| a.scenario.as_deref().is_none_or(|s| r.leaf.level_label("scenario") == s))
        .filter(|r| match filter_since {
            Some(dur) => now.duration_since(r.created).is_ok_and(|d| d <= dur),
            None => true,
        })
        .collect();
    filtered_runs.sort_by(|x, y| y.created.cmp(&x.created));

    let filtered_fits: Vec<FitEntry> = fits.into_iter()
        .filter(|f| a.model.as_deref().is_none_or(|m| f.view.source_model.as_deref().is_some_and(|s| s.contains(m))))
        .filter(|_| a.scenario.is_none())
        .filter(|f| match filter_since {
            Some(dur) => now.duration_since(f.created).is_ok_and(|d| d <= dur),
            None => true,
        })
        .collect();
    // Fits are listed by run, grouped by problem folder (newest folder first,
    // runs oldest first within it); `--limit` counts runs.
    let mut fit_folders: Vec<FitFolder> = filtered_fits
        .into_iter()
        .map(|entry| {
            let problem = crate::fit::run_view::Problem::of(&entry.fit_dir);
            FitFolder { entry, problem }
        })
        .collect();
    fit_folders.sort_by(|x, y| y.entry.created.cmp(&x.entry.created));
    let mut fit_rows: Vec<FitRunRow> = fit_run_rows(Path::new(&*root), &fit_folders)
        .into_iter()
        .filter(|r| a.history || r.history_index.is_none_or(|n| n == 0))
        .collect();
    fit_rows.sort_by(|x, y| x.folder.cmp(&y.folder).then(x.created.cmp(&y.created)));

    let mut filtered_profiles: Vec<ProfileEntry> = profiles.into_iter()
        .filter(|p| a.model.as_deref().is_none_or(|m| p.model.contains(m)))
        .filter(|_| a.scenario.is_none())
        .filter(|p| match filter_since {
            Some(dur) => now.duration_since(p.created).is_ok_and(|d| d <= dur),
            None => true,
        })
        .collect();
    filtered_profiles.sort_by(|x, y| y.created.cmp(&x.created));

    let mut filtered_surveys: Vec<SurveyRow> = surveys.into_iter()
        .filter(|s| a.model.as_deref().is_none_or(|m| s.leaf.level_label("model").contains(m)))
        .filter(|_| a.scenario.is_none())
        .filter(|s| match filter_since {
            Some(dur) => now.duration_since(s.created).is_ok_and(|d| d <= dur),
            None => true,
        })
        .collect();
    filtered_surveys.sort_by(|x, y| y.created.cmp(&x.created));

    let mut filtered_pfilters: Vec<PfilterRow> = pfilters.into_iter()
        .filter(|p| a.model.as_deref().is_none_or(|m| p.leaf.level_label("model").contains(m)))
        .filter(|_| a.scenario.is_none())
        .filter(|p| match filter_since {
            Some(dur) => now.duration_since(p.created).is_ok_and(|d| d <= dur),
            None => true,
        })
        .collect();
    filtered_pfilters.sort_by(|x, y| y.created.cmp(&x.created));

    let mut filtered_ensembles: Vec<EnsembleRow> = ensembles.into_iter()
        .filter(|e| a.model.as_deref().is_none_or(|m| e.leaf.level_label("model").contains(m)))
        .filter(|_| a.scenario.is_none())
        .filter(|e| match filter_since {
            Some(dur) => now.duration_since(e.created).is_ok_and(|d| d <= dur),
            None => true,
        })
        .collect();
    filtered_ensembles.sort_by(|x, y| y.created.cmp(&x.created));

    if !a.all {
        filtered_runs.truncate(a.limit);
        fit_rows.truncate(a.limit);
        filtered_profiles.truncate(a.limit);
        filtered_surveys.truncate(a.limit);
        filtered_pfilters.truncate(a.limit);
        filtered_ensembles.truncate(a.limit);
    }

    if format_json {
        print_sim_json(&out, &filtered_runs);
        for row in &fit_rows {
            let line = fit_list_row_json(row, &fit_folders[row.folder].problem);
            out.report_line(format_args!("{}", serde_json::to_string(&line).unwrap_or_default()));
        }
        print_profiles_json(&out, &filtered_profiles);
        print_survey_json(&out, &filtered_surveys);
        print_pfilter_json(&out, &filtered_pfilters);
        print_ensemble_json(&out, &filtered_ensembles);
    } else {
        let any_other = !fit_rows.is_empty()
            || !filtered_profiles.is_empty()
            || !filtered_surveys.is_empty()
            || !filtered_pfilters.is_empty()
            || !filtered_ensembles.is_empty();
        if !filtered_ensembles.is_empty() {
            out.say(format_args!("{}", style::stderr().paint("ensembles", Style::Bold)));
            print_ensemble_table(&out, &filtered_ensembles, now);
            out.say("");
        }
        if !fit_rows.is_empty() {
            out.say(format_args!("{}", style::stderr().paint("fits", Style::Bold)));
            print_fit_runs_table(&out, &fit_folders, &fit_rows, a.runs);
            out.say("");
        }
        if !filtered_profiles.is_empty() {
            out.say(format_args!("{}", style::stderr().paint("profiles", Style::Bold)));
            print_profiles_table(&out, &filtered_profiles, now);
            out.say("");
        }
        if !filtered_surveys.is_empty() {
            out.say(format_args!("{}", style::stderr().paint("surveys", Style::Bold)));
            print_survey_table(&out, &filtered_surveys, now);
            out.say("");
        }
        if !filtered_pfilters.is_empty() {
            out.say(format_args!("{}", style::stderr().paint("pfilters", Style::Bold)));
            print_pfilter_table(&out, &filtered_pfilters, now);
            out.say("");
        }
        if !filtered_runs.is_empty() || !any_other {
            if any_other { out.say(format_args!("{}", style::stderr().paint("sims", Style::Bold))); }
            print_sim_table(&out, &filtered_runs, now);
        }
    }
}

/// Enumerate the (grid point × seed × start) leaves of one profile, identified
/// by its profile-base hash prefix (the `profile` level). Walks the new-format
/// `ProfilePoint` leaves under `<root>/profiles/` and prints those whose base
/// hash matches, in (point, seed, start) order.
fn list_profile_children(
    out: &crate::out::Out,
    root: &str,
    parent_hash_prefix: &str,
    a: &crate::args::ListArgs,
) {
    let cwd = std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
    let mut matches: Vec<cas_read::Leaf> = cas_read::walk_profile_leaves(Path::new(root))
        .into_iter()
        .filter(|leaf| leaf.record.levels.first()
            .map(|l| l.hash.to_hex().starts_with(parent_hash_prefix))
            .unwrap_or(false))
        .collect();

    if matches.is_empty() {
        out.say(format_args!("no profile-point runs found with profile-base hash prefix '{}'",
            parent_hash_prefix));
        return;
    }

    // Natural grid-traversal order: by (point, seed, start) level label.
    matches.sort_by(|x, y| {
        (x.level_label("point"), x.level_label("seed"), x.level_label("start"))
            .cmp(&(y.level_label("point"), y.level_label("seed"), y.level_label("start")))
    });

    let limit = if a.all { matches.len() } else { a.limit.min(matches.len()) };

    if a.format.as_deref() == Some("json") {
        let slice: Vec<&runid::RunRecord> = matches.iter().take(limit).map(|l| &l.record).collect();
        match serde_json::to_string_pretty(&slice) {
            Ok(s)  => out.report_line(format_args!("{}", s)),
            Err(e) => out.say(format_args!("json error: {}", e)),
        }
        return;
    }

    out.say(format_args!("{}", style::stderr().paint("profile-point leaves", Style::Bold)));
    out.say(format_args!("  {:<16} {:<8} {:<8} {:>14}  path",
        "point", "seed", "start", "best_loglik"));
    for leaf in matches.iter().take(limit) {
        let ll = leaf.record.inputs.as_object()
            .and_then(|o| o.get("best_loglik"))
            .and_then(|v| v.as_f64())
            .map(|x| format!("{:.2}", x))
            .unwrap_or_else(|| "—".into());
        let rel = pathdiff_str(&leaf.dir, &cwd);
        out.say(format_args!("  {:<16} {:<8} {:<8} {:>14}  {}",
            leaf.level_label("point"), leaf.level_label("seed"),
            leaf.level_label("start"), ll, style::stderr().paint(&rel.to_string(), Style::Dim)));
    }
    if matches.len() > limit {
        out.say(format_args!("  ... {} more (use --all to show)", matches.len() - limit));
    }
}

// ── cmd_show ─────────────────────────────────────────────────────────────────

pub fn cmd_show(a: &crate::args::ShowArgs) {
    let out = crate::out::Out::new(false);
    let root = a.root.to_string_lossy();
    let json = show_json(&out, a.format.as_deref());
    let resolved = resolve_any(&root, &a.target).unwrap_or_else(|e| {
        out.fail(format!("error: {}", e));
    });
    if json {
        let value = match &resolved {
            Resolved::Fit { leaf, rel_path, .. } => fit_show_json(leaf, rel_path),
            Resolved::FitEnvelope { segment, .. } => fit_folder_json(segment),
            Resolved::Sim { leaf, rel_path, .. } => run_show_json(leaf, "sim", rel_path),
            Resolved::Profile { leaf, rel_path, .. } => run_show_json(leaf, "profile_point", rel_path),
            Resolved::Pfilter { leaf, rel_path, .. } => run_show_json(leaf, "pfilter", rel_path),
            Resolved::Survey { leaf, rel_path, .. } => run_show_json(leaf, "survey", rel_path),
            Resolved::SimEnsemble { leaf, rel_path, .. } => run_show_json(leaf, "sim_ensemble", rel_path),
        };
        out.report_line(format_args!("{}", serde_json::to_string_pretty(&value).unwrap_or_default()));
        return;
    }
    match Ok::<Resolved, String>(resolved) {
        Ok(Resolved::Sim { leaf, rel_path, created }) => show_sim_record(&out, &leaf, &rel_path, created),
        Ok(Resolved::Fit { leaf, rel_path, created }) => show_fit_run(&out, &leaf, &rel_path, created),
        Ok(Resolved::Profile { leaf, rel_path, created }) => show_profile_record(&out, &leaf, &rel_path, created),
        Ok(Resolved::Pfilter { leaf, rel_path, created }) => show_pfilter_record(&out, &leaf, &rel_path, created),
        Ok(Resolved::Survey { leaf, rel_path, created }) => show_survey_record(&out, &leaf, &rel_path, created),
        Ok(Resolved::SimEnsemble { leaf, rel_path, created }) => show_sim_ensemble_record(&out, &leaf, &rel_path, created),
        Ok(Resolved::FitEnvelope { segment, run, rel_path }) => show_fit_envelope(&out, &segment, run.as_deref(), &rel_path),
        Err(e) => {
            out.fail(format!("error: {}", e));
        }
    }
}

/// Render a new-format (`RunRecord`) sim: the factored levels, the run_id
/// address, and provenance. Mirrors the legacy `show_simulate` layout.
fn show_sim_record(out: &crate::out::Out, leaf: &cas_read::Leaf, rel_path: &str, created: SystemTime) {
    let rec = &leaf.record;
    out.report_line(format_args!("{}", style::stdout().paint("path", Style::Dim))); out.report_line(format_args!("  {}", style::stdout().paint(rel_path, Style::Code)));
    out.report_line(format_args!("{}", style::stdout().paint("kind", Style::Dim))); out.report_line("  sim");
    if let Some(ref l) = rec.provenance.label {
        out.report_line(format_args!("{}", style::stdout().paint("label", Style::Dim))); out.report_line(format_args!("  {}", l));
    }
    out.report_line(format_args!("{}", style::stdout().paint("model", Style::Dim))); out.report_line(format_args!("  {}", leaf.level_label("model")));
    out.report_line(format_args!("{}", style::stdout().paint("scenario", Style::Dim))); out.report_line(format_args!("  {}", leaf.level_label("scenario")));
    out.report_line(format_args!("{}", style::stdout().paint("seed", Style::Dim))); out.report_line(format_args!("  {}", leaf.seed()));
    out.report_line(format_args!("{}", style::stdout().paint("config", Style::Dim))); out.report_line(format_args!("  {}", leaf.level_label("config")));
    // The resolved parameter values the `params` level hashed (recorded in
    // `run.json.inputs`, display-only — gh#583 item I).
    if let Some(params) = rec.inputs.get("params").and_then(|v| v.as_object()) {
        out.report_line(format_args!("{}", style::stdout().paint("params", Style::Dim)));
        for (name, value) in params {
            out.report_line(format_args!("  {:<16} {}", name, value));
        }
    }
    out.report_line(format_args!("{}", style::stdout().paint("run_id", Style::Dim))); out.report_line(format_args!("  {}", style::stdout().paint(&rec.run_id.to_hex().to_string(), Style::Dim)));
    out.report_line(format_args!("{}", style::stdout().paint("levels", Style::Dim)));
    for lvl in &rec.levels {
        out.report_line(format_args!("  {:<9} {}-{}", lvl.name, lvl.label, style::stdout().paint(&lvl.hash.short8().to_string(), Style::Dim)));
    }
    out.report_line(format_args!("{}", style::stdout().paint("trajectory", Style::Dim)));
    out.report_line(format_args!("  {} bytes", leaf.traj_bytes()));
    out.report_line(format_args!("{}", style::stdout().paint("created", Style::Dim)));
    out.report_line(format_args!("  {}  ({})",
        rec.provenance.created_at.as_deref().unwrap_or("?"),
        fmt_relative_time(created, SystemTime::now())));
    out.report_line(format_args!("{}", style::stdout().paint("engine", Style::Dim))); out.report_line(format_args!("  {}", engine_line(rec)));
    out.report_line(format_args!("{}", style::stdout().paint("argv", Style::Dim)));
    out.report_line(format_args!("  {}", rec.provenance.argv.join(" ")));
}

/// The directories `fit predict` wrote for a fit envelope, each with the
/// segment-relative prefix its files are listed under (gh#896). For one run —
/// the named one, or the folder's only one — the directory of a default-options
/// predict by this engine ([`crate::fit::fit_view::default_predictions_dir`])
/// is listed bare (prefix ""), and every other `predict/<run id>-<key>/` under
/// its own prefix, so a bare path always means the default options and never
/// names a different artifact as predicts come and go. For a folder holding
/// several runs, every run's directories are listed under their prefixes. A
/// listed path is one `cat --stream` takes.
fn envelope_outputs(segment: &Path, run: Option<&str>) -> Vec<(PathBuf, String)> {
    let ids: Vec<String> = match run {
        Some(id) => vec![id.to_string()],
        None => crate::fit::fit_view::FitView::read(segment)
            .map(|v| v.methods.into_iter().map(|s| s.run_id).collect())
            .unwrap_or_default(),
    };
    let single = ids.len() == 1;
    let mut out: Vec<(PathBuf, String)> = Vec::new();
    for id in &ids {
        let bare = crate::fit::fit_view::default_predictions_dir(segment, id);
        let stem = crate::fit::fit_view::predictions_label(id);
        let mut dirs: Vec<PathBuf> = std::fs::read_dir(segment.join("predict"))
            .into_iter()
            .flatten()
            .flatten()
            .map(|e| e.path())
            .filter(|d| {
                let n = d.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
                d.is_dir() && n.starts_with(&format!("{stem}-")) && !(single && *d == bare)
            })
            .collect();
        dirs.sort();
        if single {
            out.push((bare.clone(), String::new()));
        }
        for d in dirs {
            let rel = d.strip_prefix(segment).unwrap_or(&d).display().to_string();
            out.push((d, format!("{rel}/")));
        }
    }
    out
}

/// Every address of one artifact family under a fit segment, sorted: the bare
/// `family` plus every `family-<key>` sibling.
///
/// `fit predict` keys the address of a report by what produced it — the
/// reporting vocabulary (`quantities-<key8>`, proposal 2026-08-19) and the chain
/// selection (`predictive-excl3,5`, gh#795) — so the pooled artifact and each
/// keyed one coexist. A fixed-name listing would show only the pooled one and
/// read as "the subset was never generated".
///
/// `dirs` selects directories (the TSVs) rather than the `<address>.json`
/// manifests beside them. Manifests are returned with their `.json` suffix, so
/// either list can be printed as-is.
///
/// Ordered by ADDRESS, not by filename, so the pooled artifact leads both lists.
/// Sorting manifests by filename would put `predictive.json` after
/// `predictive-excl4.json` (`-` sorts before `.`) while the directories put
/// `predictive/` first — the same fit reading in two different orders.
fn keyed_children(segment: &Path, family: &str, dirs: bool) -> Vec<String> {
    let prefix = format!("{family}-");
    let stem_of = |n: &str| -> String {
        if dirs { n.to_string() } else { n.strip_suffix(".json").unwrap_or("").to_string() }
    };
    let mut out: Vec<(String, String)> = std::fs::read_dir(segment)
        .into_iter()
        .flatten()
        .flatten()
        .filter(|e| e.path().is_dir() == dirs)
        .map(|e| e.file_name().to_string_lossy().into_owned())
        .filter_map(|n| {
            let stem = stem_of(&n);
            (stem == family || stem.starts_with(&prefix)).then_some((stem, n))
        })
        .collect();
    out.sort();
    out.into_iter().map(|(_, n)| n).collect()
}

/// Render a fit SEGMENT as its *output envelope*: the fit-level identity (label,
/// fit hash, declared methods) plus the discoverable output files the fit
/// produced (predictive / observed / quantities artifacts + top-level
/// manifests). A read-side projection of the segment directory — there is no
/// fit-wide `run.json`, so this is the fit's `show` view. Mirrors the visual
/// style of [`show_fit_record`].
fn show_fit_envelope(out: &crate::out::Out, segment: &Path, run: Option<&str>, rel_path: &str) {
    out.report_line(format_args!("{}", style::stdout().paint("path", Style::Dim))); out.report_line(format_args!("  {}", style::stdout().paint(rel_path, Style::Code)));
    out.report_line(format_args!("{}", style::stdout().paint("kind", Style::Dim))); out.report_line("  fit");
    // Fit-level hash + declared methods from the folded view (when the segment
    // holds method leaves; an output-only / incomplete segment skips these).
    // Labels are its runs' (gh#896), never identity-bearing.
    if let Some(view) = crate::fit::fit_view::FitView::read(segment) {
        if let Some(label) = view.label() {
            out.report_line(format_args!("{}", style::stdout().paint("label", Style::Dim))); out.report_line(format_args!("  {}", label));
        }
        let short = if view.fit_hash.len() >= 8 { &view.fit_hash[..8] } else { &view.fit_hash };
        out.report_line(format_args!("{}", style::stdout().paint("fit", Style::Dim))); out.report_line(format_args!("  {}", style::stdout().paint(short, Style::Dim)));
        if !view.methods_declared.is_empty() {
            out.report_line(format_args!("{}", style::stdout().paint("methods", Style::Dim)));
            out.report_line(format_args!("  {}", view.methods_declared.join(", ")));
        }
    }
    show_folder_runs(out, segment);
    // Output envelope: the predict/quantities artifacts the fit produced,
    // grouped by subdir, each as a segment-relative path the user can pass to
    // `cat --stream`. Absent subdirs are skipped (a not-yet-predicted fit just
    // shows fewer outputs).
    out.report_line(format_args!("{}", style::stdout().paint("outputs", Style::Dim)));
    let mut any = false;
    // None of these is one fixed name. `fit predict --quantities FILE` writes
    // `quantities-<key8>/` (proposal 2026-08-19), one per reporting vocabulary;
    // `fit predict --exclude-chains 3,5` writes `predictive-excl3,5/`,
    // `contrasts-excl3,5/` and `quantities…-excl3,5/` (gh#795), one per chain
    // subset. Discover each family rather than listing fixed names, or an
    // artifact that WAS written would read here as never generated. `observed/`
    // is unkeyed — it is the fit's data, the same under any selection.
    for (outputs, prefix) in envelope_outputs(segment, run) {
        let mut subs: Vec<String> = keyed_children(&outputs, "predictive", true);
        subs.push("observed".into());
        subs.extend(keyed_children(&outputs, "contrasts", true));
        subs.extend(keyed_children(&outputs, "quantities", true));
        for sub in &subs {
            let sub = sub.as_str();
            if let Ok(entries) = std::fs::read_dir(outputs.join(sub)) {
                let mut files: Vec<String> = entries
                    .flatten()
                    .filter_map(|e| {
                        let p = e.path();
                        if p.extension().is_some_and(|x| x == "tsv") {
                            p.file_name()
                                .map(|f| format!("{}{}/{}", prefix, sub, f.to_string_lossy()))
                        } else {
                            None
                        }
                    })
                    .collect();
                files.sort();
                for f in files {
                    out.report_line(format_args!("  {}", f));
                    any = true;
                }
            }
        }
        let mut manifests: Vec<String> = keyed_children(&outputs, "predictive", false);
        // gh#892: `fit predict` writes `report.json` on every run — the list of
        // deterministic failures, empty when nothing refused — keyed by chain
        // selection like the predictive itself (`report-excl3,5.json`). It is the
        // file that says whether a predictive is complete, so it belongs beside
        // the predictive rather than being reachable only by knowing it is there.
        manifests.extend(keyed_children(&outputs, "report", false));
        manifests.extend(keyed_children(&outputs, "quantities", false));
        for manifest in &manifests {
            if outputs.join(manifest).exists() {
                out.report_line(format_args!("  {}{}", prefix, manifest));
                any = true;
            }
        }
    }
    if segment.join("fit.meta.json").exists() {
        out.report_line("  fit.meta.json");
        any = true;
    }
    if !any {
        out.report_line(format_args!("  {}", style::stdout().paint("(none — run `camdl fit predict` to generate)", Style::Dim)));
    }
}

/// Render a new-format (`RunRecord`) profile-point leaf: the five factored
/// levels (`profile`/`point`/`method`/`seed`/`start`), the run_id address, the
/// `--label` (read from the profile-base `fit.meta.json` sidecar — its single
/// authoritative home, NOT copied per leaf), and the recorded `inputs`
/// including the per-leaf provenance (gh#83/85 parameter resolution, per-chain
/// init, suppressed-warnings waiver). Mirrors `show_fit_record`.
fn show_profile_record(out: &crate::out::Out, leaf: &cas_read::Leaf, rel_path: &str, created: SystemTime) {
    let rec = &leaf.record;
    out.report_line(format_args!("{}", style::stdout().paint("path", Style::Dim))); out.report_line(format_args!("  {}", style::stdout().paint(rel_path, Style::Code)));
    out.report_line(format_args!("{}", style::stdout().paint("kind", Style::Dim))); out.report_line("  profile_point");
    // The label lives once on the profile-base sidecar; walk up the leaf's
    // ancestors to the first segment carrying a `fit.meta.json`.
    match leaf.dir.ancestors().find_map(|d| crate::run_meta::read_fit_sidecar(d).transpose()) {
        Some(Ok(sc)) => if let Some(ref l) = sc.label {
            out.report_line(format_args!("{}", style::stdout().paint("label", Style::Dim))); out.report_line(format_args!("  {}", l));
        },
        Some(Err(e)) => out.say(format_args!("warning: {e}")),
        None => {}
    }
    for lvl_name in ["profile", "point", "method", "seed", "start"] {
        out.report_line(format_args!("{}", style::stdout().paint(lvl_name, Style::Dim)));
        out.report_line(format_args!("  {}", leaf.level_label(lvl_name)));
    }
    out.report_line(format_args!("{}", style::stdout().paint("run_id", Style::Dim))); out.report_line(format_args!("  {}", style::stdout().paint(&rec.run_id.to_hex().to_string(), Style::Dim)));
    out.report_line(format_args!("{}", style::stdout().paint("levels", Style::Dim)));
    for lvl in &rec.levels {
        out.report_line(format_args!("  {:<9} {}-{}", lvl.name, lvl.label, style::stdout().paint(&lvl.hash.short8().to_string(), Style::Dim)));
    }
    if !rec.deps.is_empty() {
        out.report_line(format_args!("{}", style::stdout().paint("deps", Style::Dim)));
        for d in &rec.deps {
            out.report_line(format_args!("  {} {} ({})", style::stdout().paint(&d.run_id.short8().to_string(), Style::Dim), d.artifact, style::stdout().paint(&d.digest.short8().to_string(), Style::Dim)));
        }
    }
    if let Some(obj) = rec.inputs.as_object() {
        for key in ["method", "grid_point", "start", "best_loglik", "wall_time_seconds"] {
            if let Some(v) = obj.get(key) {
                out.report_line(format_args!("{}", style::stdout().paint(key, Style::Dim))); out.report_line(format_args!("  {}", v));
            }
        }
        // Per-leaf provenance (display-only, never hashed). Surface the audit
        // trail the old per-run ProfileMeta carried so write→read→visible holds.
        if let Some(prov) = obj.get("provenance").and_then(|v| v.as_object()) {
            if let Some(pp) = prov.get("parameters_provenance").and_then(|v| v.as_object()) {
                if !pp.is_empty() {
                    out.report_line(format_args!("{}", style::stdout().paint("parameter provenance", Style::Dim)));
                    let mut names: Vec<&String> = pp.keys().collect();
                    names.sort();
                    for n in names {
                        let src = pp[n].get("source").and_then(|v| v.as_str()).unwrap_or("?");
                        let role = pp[n].get("role").and_then(|v| v.as_str()).unwrap_or("?");
                        out.report_line(format_args!("  {:<16} {} ({})", n, src, role));
                    }
                }
            }
            if let Some(ip) = prov.get("init_provenance") {
                if let Some(method) = ip.get("method").and_then(|v| v.as_str()) {
                    let n_chains = ip.get("chains").and_then(|v| v.as_array()).map(|a| a.len()).unwrap_or(0);
                    out.report_line(format_args!("{}", style::stdout().paint("init provenance", Style::Dim)));
                    out.report_line(format_args!("  method {} ({} chain{})", method, n_chains,
                        if n_chains == 1 { "" } else { "s" }));
                }
            }
        }
    }
    out.report_line(format_args!("{}", style::stdout().paint("created", Style::Dim)));
    out.report_line(format_args!("  {}  ({})",
        rec.provenance.created_at.as_deref().unwrap_or("?"),
        fmt_relative_time(created, SystemTime::now())));
    out.report_line(format_args!("{}", style::stdout().paint("engine", Style::Dim))); out.report_line(format_args!("  {}", engine_line(rec)));
}

/// Render a new-format (`RunRecord`) pfilter-eval leaf: the four factored
/// levels (`model`/`config`/`params`/`seed`), the run_id address, and the
/// recorded loglik result + scored point from `inputs`. A pfilter eval is a
/// single leaf (no grid); the `--label`, if any, is on the leaf provenance.
fn show_pfilter_record(out: &crate::out::Out, leaf: &cas_read::Leaf, rel_path: &str, created: SystemTime) {
    let rec = &leaf.record;
    out.report_line(format_args!("{}", style::stdout().paint("path", Style::Dim))); out.report_line(format_args!("  {}", style::stdout().paint(rel_path, Style::Code)));
    out.report_line(format_args!("{}", style::stdout().paint("kind", Style::Dim))); out.report_line("  pfilter");
    if let Some(ref l) = rec.provenance.label {
        out.report_line(format_args!("{}", style::stdout().paint("label", Style::Dim))); out.report_line(format_args!("  {}", l));
    }
    for lvl_name in ["model", "config", "params", "seed"] {
        out.report_line(format_args!("{}", style::stdout().paint(lvl_name, Style::Dim)));
        out.report_line(format_args!("  {}", leaf.level_label(lvl_name)));
    }
    out.report_line(format_args!("{}", style::stdout().paint("run_id", Style::Dim))); out.report_line(format_args!("  {}", style::stdout().paint(&rec.run_id.to_hex().to_string(), Style::Dim)));
    out.report_line(format_args!("{}", style::stdout().paint("levels", Style::Dim)));
    for lvl in &rec.levels {
        out.report_line(format_args!("  {:<9} {}-{}", lvl.name, lvl.label, style::stdout().paint(&lvl.hash.short8().to_string(), Style::Dim)));
    }
    if let Some(obj) = rec.inputs.as_object() {
        for key in ["loglik", "loglik_sd", "n_replicates", "n_particles"] {
            if let Some(v) = obj.get(key) {
                if v.is_null() { continue; }
                out.report_line(format_args!("{}", style::stdout().paint(key, Style::Dim))); out.report_line(format_args!("  {}", v));
            }
        }
        if let Some(params) = obj.get("params").and_then(|v| v.as_array()) {
            if !params.is_empty() {
                out.report_line(format_args!("{}", style::stdout().paint("scored point", Style::Dim)));
                for pair in params {
                    if let Some(p) = pair.as_array() {
                        if p.len() == 2 {
                            out.report_line(format_args!("  {:<16} {}",
                                p[0].as_str().unwrap_or("?"), p[1]));
                        }
                    }
                }
            }
        }
    }
    out.report_line(format_args!("{}", style::stdout().paint("created", Style::Dim)));
    out.report_line(format_args!("  {}  ({})",
        rec.provenance.created_at.as_deref().unwrap_or("?"),
        fmt_relative_time(created, SystemTime::now())));
    out.report_line(format_args!("{}", style::stdout().paint("engine", Style::Dim))); out.report_line(format_args!("  {}", engine_line(rec)));
}

/// Render a new-format (`RunRecord`) survey leaf: the four factored levels
/// (`model`/`config`/`box`/`seed`), the run_id address, the recorded eval
/// config + estimated set + best-loglik from `inputs`, and the landscape /
/// summary / HTML artifact sizes. A survey is a single leaf (the N LHS points
/// live within it, not as an axis). Mirrors `show_pfilter_record`.
fn show_survey_record(out: &crate::out::Out, leaf: &cas_read::Leaf, rel_path: &str, created: SystemTime) {
    let rec = &leaf.record;
    out.report_line(format_args!("{}", style::stdout().paint("path", Style::Dim))); out.report_line(format_args!("  {}", style::stdout().paint(rel_path, Style::Code)));
    out.report_line(format_args!("{}", style::stdout().paint("kind", Style::Dim))); out.report_line("  survey");
    if let Some(ref l) = rec.provenance.label {
        out.report_line(format_args!("{}", style::stdout().paint("label", Style::Dim))); out.report_line(format_args!("  {}", l));
    }
    for lvl_name in ["model", "config", "box", "seed"] {
        out.report_line(format_args!("{}", style::stdout().paint(lvl_name, Style::Dim)));
        out.report_line(format_args!("  {}", leaf.level_label(lvl_name)));
    }
    out.report_line(format_args!("{}", style::stdout().paint("run_id", Style::Dim))); out.report_line(format_args!("  {}", style::stdout().paint(&rec.run_id.to_hex().to_string(), Style::Dim)));
    out.report_line(format_args!("{}", style::stdout().paint("levels", Style::Dim)));
    for lvl in &rec.levels {
        out.report_line(format_args!("  {:<9} {}-{}", lvl.name, lvl.label, style::stdout().paint(&lvl.hash.short8().to_string(), Style::Dim)));
    }
    if let Some(obj) = rec.inputs.as_object() {
        if let Some(est) = obj.get("estimated").and_then(|v| v.as_array()) {
            let names: Vec<&str> = est.iter().filter_map(|v| v.as_str()).collect();
            out.report_line(format_args!("{}", style::stdout().paint("estimated", Style::Dim))); out.report_line(format_args!("  {}", names.join(", ")));
        }
        for key in ["eval_method", "eval_particles", "eval_replicates", "n_points", "best_loglik"] {
            if let Some(v) = obj.get(key) {
                if v.is_null() { continue; }
                out.report_line(format_args!("{}", style::stdout().paint(key, Style::Dim))); out.report_line(format_args!("  {}", v));
            }
        }
    }
    let landscape = leaf.dir.join("landscape.tsv");
    if landscape.exists() {
        let bytes = std::fs::metadata(&landscape).map(|m| m.len()).unwrap_or(0);
        out.report_line(format_args!("{}", style::stdout().paint("landscape", Style::Dim)));
        out.report_line(format_args!("  landscape.tsv ({} bytes)", bytes));
    }
    let summary = leaf.dir.join("summary.json");
    if summary.exists() {
        if let Ok(s) = std::fs::read_to_string(&summary) {
            if let Ok(j) = serde_json::from_str::<serde_json::Value>(&s) {
                if let Some(se_q) = j.get("loglik_se_quartiles") {
                    out.report_line(format_args!("{}", style::stdout().paint("loglik_se quartiles", Style::Dim)));
                    out.report_line(format_args!("  {}", se_q));
                }
            }
        }
    }
    let html = leaf.dir.join("landscape.html");
    if html.exists() {
        let bytes = std::fs::metadata(&html).map(|m| m.len()).unwrap_or(0);
        out.report_line(format_args!("{}", style::stdout().paint("rendered", Style::Dim)));
        out.report_line(format_args!("  landscape.html ({} bytes)", bytes));
    }
    out.report_line(format_args!("{}", style::stdout().paint("created", Style::Dim)));
    out.report_line(format_args!("  {}  ({})",
        rec.provenance.created_at.as_deref().unwrap_or("?"),
        fmt_relative_time(created, SystemTime::now())));
    out.report_line(format_args!("{}", style::stdout().paint("engine", Style::Dim))); out.report_line(format_args!("  {}", engine_line(rec)));
}

/// Render a new-format (`RunRecord`) `SimEnsemble` leaf: the four factored
/// levels (`model`/`config`/`params`/`grid`), the run_id address, the cell
/// count + scenario/seed/draw summary from `inputs`, the `deps` (the N per-cell
/// `Sim` leaves the combined TSV is derived from), and the `ensemble.tsv` size.
fn show_sim_ensemble_record(out: &crate::out::Out, leaf: &cas_read::Leaf, rel_path: &str, created: SystemTime) {
    let rec = &leaf.record;
    out.report_line(format_args!("{}", style::stdout().paint("path", Style::Dim))); out.report_line(format_args!("  {}", style::stdout().paint(rel_path, Style::Code)));
    out.report_line(format_args!("{}", style::stdout().paint("kind", Style::Dim))); out.report_line("  sim_ensemble");
    if let Some(ref l) = rec.provenance.label {
        out.report_line(format_args!("{}", style::stdout().paint("label", Style::Dim))); out.report_line(format_args!("  {}", l));
    }
    for lvl_name in ["model", "config", "params", "grid"] {
        out.report_line(format_args!("{}", style::stdout().paint(lvl_name, Style::Dim)));
        out.report_line(format_args!("  {}", leaf.level_label(lvl_name)));
    }
    out.report_line(format_args!("{}", style::stdout().paint("run_id", Style::Dim))); out.report_line(format_args!("  {}", style::stdout().paint(&rec.run_id.to_hex().to_string(), Style::Dim)));
    out.report_line(format_args!("{}", style::stdout().paint("levels", Style::Dim)));
    for lvl in &rec.levels {
        out.report_line(format_args!("  {:<9} {}-{}", lvl.name, lvl.label, style::stdout().paint(&lvl.hash.short8().to_string(), Style::Dim)));
    }
    if let Some(obj) = rec.inputs.as_object() {
        for key in ["n_cells", "n_scenarios", "n_seeds", "n_draws", "n_points"] {
            if let Some(v) = obj.get(key) {
                if v.is_null() { continue; }
                out.report_line(format_args!("{}", style::stdout().paint(key, Style::Dim))); out.report_line(format_args!("  {}", v));
            }
        }
        if let Some(sc) = obj.get("scenarios").and_then(|v| v.as_array()) {
            let names: Vec<&str> = sc.iter().filter_map(|v| v.as_str()).collect();
            if !names.is_empty() {
                out.report_line(format_args!("{}", style::stdout().paint("scenarios", Style::Dim))); out.report_line(format_args!("  {}", names.join(", ")));
            }
        }
    }
    // The deps are the N per-cell Sim leaves the combined TSV concatenates.
    if !rec.deps.is_empty() {
        out.report_line(format_args!("{}", style::stdout().paint("deps", Style::Dim)));
        out.report_line(format_args!("  {} per-cell sim leaves", rec.deps.len()));
        for d in &rec.deps {
            out.report_line(format_args!("  {} {} ({})", style::stdout().paint(&d.run_id.short8().to_string(), Style::Dim), d.artifact, style::stdout().paint(&d.digest.short8().to_string(), Style::Dim)));
        }
    }
    let ensemble = leaf.dir.join("ensemble.tsv");
    if ensemble.exists() {
        let bytes = std::fs::metadata(&ensemble).map(|m| m.len()).unwrap_or(0);
        out.report_line(format_args!("{}", style::stdout().paint("ensemble", Style::Dim)));
        out.report_line(format_args!("  ensemble.tsv ({} bytes)", bytes));
    }
    out.report_line(format_args!("{}", style::stdout().paint("created", Style::Dim)));
    out.report_line(format_args!("  {}  ({})",
        rec.provenance.created_at.as_deref().unwrap_or("?"),
        fmt_relative_time(created, SystemTime::now())));
    out.report_line(format_args!("{}", style::stdout().paint("engine", Style::Dim))); out.report_line(format_args!("  {}", engine_line(rec)));
}

// ── cmd_cat ──────────────────────────────────────────────────────────────────

pub fn cmd_cat(a: &crate::args::CatArgs) {
    let out = crate::out::Out::new(false);
    let selection = crate::chain_selection::ChainSelection::from_flags(
        a.chains.as_deref(),
        a.exclude_chains.as_deref(),
    )
    .unwrap_or_else(|e| {
        out.fail(format!("error: {e}"));
    });
    let root = a.root.to_string_lossy();
    let resolved = resolve_any(&root, &a.target).unwrap_or_else(|e| {
        out.fail(format!("error: {}", e));
    });

    use std::io::Write as _;

    // A chain selection reads a fit run's own stream; every other kind is
    // refused rather than printed whole.
    if let Some(sel) = &selection {
        if !matches!(resolved, Resolved::Fit { .. }) {
            out.fail(format!(
                "error: {} selects rows of a fit run's chain-keyed stream (draws.tsv); \
                 `{}` does not name a fit run",
                sel.flag(), a.target
            ));
        }
    }

    // Every kind is new-format (`RunRecord`); each arm emits and returns.
    match resolved {
        Resolved::Sim { leaf, rel_path, .. } => {
            let bytes = if let Some(ref stream) = a.stream {
                let path = find_obs_stream(&leaf.dir, stream).unwrap_or_else(|| {
                    out.fail(format!("error: no observation stream '{}' in {}", stream, rel_path));
                });
                std::fs::read(&path).unwrap_or_else(|e| {
                    out.fail(format!("error reading {}: {}", path.display(), e));
                })
            } else {
                std::fs::read(leaf.dir.join("traj.tsv")).unwrap_or_else(|e| {
                    out.fail(format!("error reading traj.tsv: {}", e));
                })
            };
            let _ = out.data().write_all(&bytes);
        }
        // New-format fit method: default to the θ̂ summary; `--stream NAME`
        // cats a named file from the leaf (e.g. `draws.tsv`,
        // `chain_1/trace.tsv`).
        Resolved::Fit { leaf, rel_path, .. } => {
            let name = a.stream.as_deref().unwrap_or("fit_state.toml");
            let path = run_stream_path(&leaf, name).unwrap_or_else(|| leaf.dir.join(name));
            let bytes = std::fs::read(&path).unwrap_or_else(|e| {
                out.fail(format!(
                    "error reading {} in {}: {}\n  \
                     run `camdl show {}` to list the run's outputs",
                    name, rel_path, e, a.target
                ));
            });
            let bytes = match &selection {
                None => bytes,
                Some(sel) => select_chain_rows(&out, &leaf, name, sel, &bytes)
                    .unwrap_or_else(|e| out.fail(format!("error: {e}")))
                    .into_bytes(),
            };
            let _ = out.data().write_all(&bytes);
        }
        // New-format profile point: default to the per-(point, seed, start)
        // `mle.toml`; `--stream NAME` cats a named file from the leaf.
        Resolved::Profile { leaf, rel_path, .. } => {
            let name = a.stream.as_deref().unwrap_or("mle.toml");
            let path = leaf.dir.join(name);
            let bytes = std::fs::read(&path).unwrap_or_else(|e| {
                out.fail(format!("error reading {} in {}: {}", name, rel_path, e));
            });
            let _ = out.data().write_all(&bytes);
        }
        // New-format pfilter eval: default to the `loglik.toml` summary;
        // `--stream NAME` cats a named saved artifact from the leaf.
        Resolved::Pfilter { leaf, rel_path, .. } => {
            let name = a.stream.as_deref().unwrap_or("loglik.toml");
            let path = leaf.dir.join(name);
            let bytes = std::fs::read(&path).unwrap_or_else(|e| {
                out.fail(format!("error reading {} in {}: {}", name, rel_path, e));
            });
            let _ = out.data().write_all(&bytes);
        }
        // New-format survey: default to `landscape.tsv`; `--stream NAME`
        // cats a named file from the leaf (e.g. `summary.json`,
        // `landscape.html`).
        Resolved::Survey { leaf, rel_path, .. } => {
            let name = a.stream.as_deref().unwrap_or("landscape.tsv");
            let path = leaf.dir.join(name);
            let bytes = std::fs::read(&path).unwrap_or_else(|e| {
                out.fail(format!("error reading {} in {}: {}", name, rel_path, e));
            });
            let _ = out.data().write_all(&bytes);
        }
        // New-format ensemble: default to `ensemble.tsv` (the combined
        // wide-format trajectory — byte-identical to the `-o` mirror).
        Resolved::SimEnsemble { leaf, rel_path, .. } => {
            let name = a.stream.as_deref().unwrap_or("ensemble.tsv");
            let path = leaf.dir.join(name);
            let bytes = std::fs::read(&path).unwrap_or_else(|e| {
                out.fail(format!("error reading {} in {}: {}", name, rel_path, e));
            });
            let _ = out.data().write_all(&bytes);
        }
        // Fit ENVELOPE: `--stream <rel>` cats an output file under the segment
        // (e.g. `predictive/weekly_cases.tsv`); no `--stream` defaults to the
        // `fit.meta.json` summary record. A bare relative path without an
        // extension also matches the `.tsv` sibling (so `--stream
        // predictive/weekly_cases` finds `predictive/weekly_cases.tsv`).
        Resolved::FitEnvelope { segment, run, rel_path } => {
            let rel = a.stream.as_deref().unwrap_or("fit.meta.json");
            // A folder that holds several runs names none of them: a stream
            // that is not the folder's own file is an ambiguity, listed as
            // `fit summary` lists it.
            if run.is_none() && !segment.join(rel).exists() {
                let runs: Vec<cas_read::Leaf> = folder_runs(&segment)
                    .into_iter()
                    .filter(|l| l.record.status == runid::RunStatus::Completed)
                    .collect();
                if runs.len() > 1 {
                    out.fail(format!("error: {}", crate::fit::handle::ResolveError::AmbiguousRuns {
                        handle: a.target.clone(),
                        runs,
                    }));
                }
            }
            // A stream is looked up in the run's predictions directory first
            // (gh#896), then in the segment (`fit.meta.json`, or a
            // `predict/<run id>-<key>/…` path when the folder holds several runs).
            let mut bases: Vec<PathBuf> = envelope_outputs(&segment, run.as_deref())
                .into_iter()
                .filter(|(_, prefix)| prefix.is_empty())
                .map(|(dir, _)| dir)
                .collect();
            bases.push(segment.clone());
            let candidates = |base: &Path| {
                let mut c = vec![base.join(rel)];
                if Path::new(rel).extension().is_none() {
                    c.push(base.join(format!("{}.tsv", rel)));
                }
                c
            };
            let path = bases
                .iter()
                .flat_map(|b| candidates(b))
                .find(|p| p.exists())
                .unwrap_or_else(|| segment.join(rel));
            let bytes = std::fs::read(&path).unwrap_or_else(|e| {
                out.fail(format!(
                    "error reading {} in fit {}: {}\n  \
                     run `camdl show {}` to list the available output streams",
                    rel, rel_path, e, a.target
                ));
            });
            let _ = out.data().write_all(&bytes);
        }
    }
}

/// Where a run handle's stream `name` is: the run's leaf, then its
/// default-options predictions under the problem folder's `predict/`,
/// then the problem folder itself. An extensionless name also matches its
/// `.tsv` sibling. `None` when no candidate exists.
fn run_stream_path(leaf: &cas_read::Leaf, name: &str) -> Option<PathBuf> {
    let mut bases = vec![leaf.dir.clone()];
    if let Ok(Some(segment)) = crate::fit::fit_view::leaf_segment(&leaf.dir) {
        let id = leaf.record.run_id.to_hex().to_string();
        bases.push(crate::fit::fit_view::default_predictions_dir(&segment, &id));
        bases.push(segment);
    }
    bases.iter().find_map(|b| {
        let mut c = vec![b.join(name)];
        if Path::new(name).extension().is_none() {
            c.push(b.join(format!("{name}.tsv")));
        }
        c.into_iter().find(|p| p.is_file())
    })
}

/// The rows of a fit leaf's stream `name` that a chain selection keeps
/// (`cat --chains` / `--exclude-chains`). Only a stream whose recorded
/// `output_schema` has a `chain` column can be selected — in practice
/// `draws.tsv`; any other is refused by name. The selection is resolved
/// against the file's own chains, warned about on stderr, and the kept rows
/// are the file's own bytes.
fn select_chain_rows(
    out: &crate::out::Out,
    leaf: &cas_read::Leaf,
    name: &str,
    sel: &crate::chain_selection::ChainSelection,
    bytes: &[u8],
) -> Result<String, String> {
    let has_chain_column = leaf.record.output_schema.get(name).is_some_and(|t| {
        t.columns.iter().any(|c| c.role == runid::record::ColumnRole::Chain)
    });
    if !has_chain_column {
        return Err(format!(
            "{} selects rows by chain, and the stream `{name}` has no `chain` column in \
             this run's recorded output schema; it applies to a stream such as draws.tsv",
            sel.flag()
        ));
    }
    let text = std::str::from_utf8(bytes).map_err(|_| format!("`{name}` is not UTF-8 text"))?;
    let present = crate::chain_selection::chains_in_draws(&leaf.dir.join(name))?;
    let info = sel.resolve(&present)?;
    crate::chain_selection::warn_active_selection(
        out, sel.polarity(), &info, crate::out::paint::stderr());
    crate::chain_selection::filter_tsv_rows(text, &info)
}

/// Locate `<sim_dir>/obs/<obs_subdir>/<stream>.tsv`, taking the first
/// match across `obs_subdir/`. Returns `None` if no stream by that
/// name exists.
fn find_obs_stream(sim_dir: &Path, stream: &str) -> Option<PathBuf> {
    let obs_root = sim_dir.join("obs");
    if !obs_root.exists() { return None; }
    let entries = std::fs::read_dir(&obs_root).ok()?;
    for entry in entries.flatten() {
        let file = entry.path().join(format!("{}.tsv", stream));
        if file.exists() { return Some(file); }
    }
    None
}

// ── Internals: discovery + resolution ────────────────────────────────────────
//
// New-format `sims/` are discovered generically via [`discover_sim_rows`]
// (data-driven depth through [`cas_read`]). The legacy fit/profile/survey
// discovery below is M3-DELETION-BOUND (gh#147) — see [`load_run_common`].

/// A discovered cached fit.
#[derive(Debug, Clone)]
struct FitEntry {
    view: crate::fit::fit_view::FitView,
    /// The segment directory itself, so a leaf's path can be rendered in the
    /// same frame as `rel_path` (gh#895).
    fit_dir: PathBuf,
    rel_path: String,
    created: SystemTime,
}

// ── Profile listings ─────────────────────────────────────────────────────────

/// A discovered profile, summarized from its `ProfilePoint` leaves under
/// `<root>/profiles/<base>/.../`. One entry per profile (grouped by the
/// `profile`-level base hash); carries the display fields `camdl list` needs.
#[derive(Debug, Clone)]
struct ProfileEntry {
    /// Profile-base hash (the `profile` level / `levels[0]`) — the
    /// `camdl show <hash>`/`list --parent <hash>` address for this profile.
    hash: String,
    /// User `--label` from the profile-base `fit.meta.json` sidecar (its
    /// single authoritative home; `None` when unset).
    label: Option<String>,
    /// The profile-base segment, cwd-relative.
    rel_path: String,
    /// Latest leaf's creation time.
    created: SystemTime,
    /// Display-only model path (from the profile-base sidecar).
    model: String,
    /// Comma-separated focal param names (e.g. "beta,gamma"), reconstructed
    /// from the distinct `point`-level labels.
    focal: String,
    /// Grid shape (e.g. "11×9 starts=4"), reconstructed from the distinct
    /// `point` labels (per-axis value counts) and distinct `start` levels.
    shape: String,
    /// Number of seed replicates (distinct `seed` levels).
    n_seeds: usize,
}

/// Discover profiles for `camdl list`: walk the new-format `ProfilePoint`
/// leaves under `<root>/profiles/` and fold them into one `ProfileEntry` per
/// profile, keyed by the `profile`-level base hash. The grid shape and focal
/// names are reconstructed from the leaves' `point` labels; the `--label` and
/// model come from the profile-base `fit.meta.json` sidecar.
fn discover_profiles(root: &str) -> Result<Vec<ProfileEntry>, String> {
    use std::collections::{HashMap, HashSet};
    let cwd = std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."));

    // One `ProfileEntry` per profile: group new-format `ProfilePoint` leaves
    // by their `profile`-level (base) hash. Each leaf is one (point × seed ×
    // start) cell; the entry summarizes the cells.
    let mut groups: HashMap<String, Vec<cas_read::Leaf>> = HashMap::new();
    for leaf in cas_read::walk_profile_leaves(Path::new(root)) {
        let base = leaf.record.levels.first().map(|l| l.hash.to_hex()).unwrap_or_default();
        groups.entry(base).or_default().push(leaf);
    }

    let mut out = Vec::new();
    for (base_hash, leaves) in groups {
        let mut point_labels: Vec<String> = Vec::new();
        let mut seed_hashes: HashSet<String> = HashSet::new();
        let mut start_hashes: HashSet<String> = HashSet::new();
        let mut created = SystemTime::UNIX_EPOCH;
        let mut base_seg: Option<PathBuf> = None;
        let level_hash = |leaf: &cas_read::Leaf, name: &str| -> Option<String> {
            leaf.record.levels.iter().find(|l| l.name == name).map(|l| l.hash.to_hex())
        };
        for leaf in &leaves {
            point_labels.push(leaf.level_label("point").to_string());
            if let Some(h) = level_hash(leaf, "seed") { seed_hashes.insert(h); }
            if let Some(h) = level_hash(leaf, "start") { start_hashes.insert(h); }
            let c = leaf_created(leaf);
            if c > created { created = c; }
            // The profile-base segment is four levels up (start/seed/method/
            // point → base); used for the sidecar (label/model) and rel path.
            if base_seg.is_none() {
                base_seg = leaf.dir.ancestors().nth(4).map(|p| p.to_path_buf());
            }
        }
        let sidecar = match base_seg.as_deref().map(crate::run_meta::read_fit_sidecar) {
            Some(Ok(s)) => s,
            Some(Err(e)) => {
                crate::out::detached().say(format_args!("warning: {e}"));
                None
            }
            None => None,
        };
        let label = sidecar.as_ref().and_then(|s| s.label.clone());
        let model = sidecar.as_ref()
            .and_then(|s| s.source_model.clone())
            .unwrap_or_else(|| "?".to_string());
        let (focal, shape) = summarize_grid(&point_labels, start_hashes.len());
        let rel_path = base_seg.as_deref()
            .map(|p| pathdiff_str(p, &cwd))
            .unwrap_or_default();
        out.push(ProfileEntry {
            hash: base_hash, label, rel_path, created, model, focal, shape,
            n_seeds: seed_hashes.len(),
        });
    }
    out.sort_by(|a, b| b.created.cmp(&a.created));
    Ok(out)
}

/// Reconstruct a profile's focal-param list + grid shape from the distinct
/// `point`-level labels. Each label is `name=val[__name2=val2]` (written by
/// `profile_cas::resolve_profile_point`); the per-axis distinct-value counts
/// give the grid dims. E.g. an 11×9 grid with 4 starts → `("beta,gamma",
/// "11×9 starts=4")`.
fn summarize_grid(point_labels: &[String], n_starts: usize) -> (String, String) {
    use std::collections::{HashMap, HashSet};
    let mut names: Vec<String> = Vec::new();
    let mut values: HashMap<String, HashSet<String>> = HashMap::new();
    let mut distinct_points: HashSet<&str> = HashSet::new();
    for lbl in point_labels {
        distinct_points.insert(lbl.as_str());
        for pair in lbl.split("__") {
            if let Some((name, val)) = pair.split_once('=') {
                if !names.iter().any(|n| n == name) { names.push(name.to_string()); }
                values.entry(name.to_string()).or_default().insert(val.to_string());
            }
        }
    }
    let focal = names.join(",");
    let shape = if names.is_empty() {
        format!("{} pts starts={}", distinct_points.len(), n_starts)
    } else {
        let dims: Vec<String> = names.iter()
            .map(|n| values.get(n).map(|s| s.len()).unwrap_or(0).to_string())
            .collect();
        format!("{} starts={}", dims.join("×"), n_starts)
    };
    (focal, shape)
}

// ── Survey listings ──────────────────────────────────────────────────────────

/// One discovered survey run. Surveys live at
/// `<root>/surveys/<stem>-<hash[:8]>/` with a `run.json` of kind
/// A new-format (`runid::RunRecord`) survey leaf, prepared for the `list`
/// display. The kind-Survey filter happens in [`cas_read::walk_survey_leaves`].
struct SurveyRow {
    leaf: cas_read::Leaf,
    rel_path: String,
    created: SystemTime,
}

/// Discover the new-format survey leaves under `root/surveys/` for `list`.
fn discover_survey_rows(root: &str) -> Vec<SurveyRow> {
    let cwd = std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
    cas_read::walk_survey_leaves(Path::new(root))
        .into_iter()
        .map(|leaf| {
            let created = leaf.record.provenance.created_at.as_deref()
                .and_then(parse_iso8601)
                .unwrap_or_else(|| {
                    std::fs::metadata(&leaf.dir).and_then(|m| m.modified())
                        .unwrap_or(SystemTime::UNIX_EPOCH)
                });
            let rel_path = pathdiff_str(&leaf.dir, &cwd);
            SurveyRow { leaf, rel_path, created }
        })
        .collect()
}

fn print_survey_table(out: &crate::out::Out, surveys: &[SurveyRow], now: SystemTime) {
    let mut t = comfy_table::Table::new();
    t.set_content_arrangement(comfy_table::ContentArrangement::Dynamic);
    t.set_header(vec!["model", "estimate", "n_points", "eval", "top_loglik", "age", "path"]);
    for s in surveys {
        let obj = s.leaf.record.inputs.as_object();
        let estimated = obj
            .and_then(|o| o.get("estimated"))
            .and_then(|v| v.as_array())
            .map(|a| a.iter().filter_map(|x| x.as_str()).collect::<Vec<_>>().join(","))
            .unwrap_or_default();
        let n_points = obj.and_then(|o| o.get("n_points")).and_then(|v| v.as_u64())
            .map(|n| n.to_string()).unwrap_or_else(|| "—".into());
        // The `config` level label is `{method}-P{particles}-r{replicates}`.
        let eval = s.leaf.level_label("config").to_string();
        let ll = obj.and_then(|o| o.get("best_loglik")).and_then(|v| v.as_f64())
            .map(|x| format!("{:.2}", x)).unwrap_or_else(|| "—".into());
        let age = fmt_relative_time(s.created, now);
        t.add_row(vec![
            s.leaf.level_label("model").to_string(),
            estimated,
            n_points,
            eval,
            ll,
            age,
            s.rel_path.clone(),
        ]);
    }
    out.report_line(format_args!("{t}"));
}

fn print_survey_json(out: &crate::out::Out, surveys: &[SurveyRow]) {
    let recs: Vec<&runid::RunRecord> = surveys.iter().map(|s| &s.leaf.record).collect();
    match serde_json::to_string_pretty(&recs) {
        Ok(s) => out.report_line(format_args!("{}", s)),
        Err(e) => out.say(format_args!("json error: {}", e)),
    }
}

/// Walk `root/fits/` one level deep — each immediate child is a fit
/// directory (`<stem>-<hash[:8]>/`). Method-level run.json records live
/// deeper and are not surfaced by `camdl list`.
///
/// Implementation: delegates to `fit_tree::walk_fits_root` for
/// canonical fit-dir discovery, then layers on the per-entry display
/// metadata (`rel_path`, `created` mtime) browse needs that the
/// canonical walker doesn't carry.
fn discover_fits(root: &str) -> Result<Vec<FitEntry>, String> {
    let fits_dir = Path::new(root).join("fits");
    let cwd = std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
    let entries = crate::fit::fit_tree::walk_fits_root(&fits_dir)
        .map_err(|e| format!("cannot read {}: {}", fits_dir.display(), e))?;
    Ok(entries
        .into_iter()
        .map(|e| {
            // `walk_fits_root` already parsed the segment into a `FitView`;
            // reuse it rather than re-reading. `created` and `rel_path` are
            // display-only, computed from the view's `created_at` + dir path.
            let created = parse_iso8601(&e.view.created_at)
                .unwrap_or_else(|| std::fs::metadata(&e.fit_dir)
                    .and_then(|m| m.modified())
                    .unwrap_or(SystemTime::UNIX_EPOCH));
            let rel_path = pathdiff_str(&e.fit_dir, &cwd);
            FitEntry { view: e.view, fit_dir: e.fit_dir, rel_path, created }
        })
        .collect())
}

/// A resolved run — always a new-format (`runid::RunRecord`) leaf, dispatched
/// by `ArtifactKind`.
#[derive(Debug)]
enum Resolved {
    Sim { leaf: cas_read::Leaf, rel_path: String, created: SystemTime },
    /// `FitMethod` leaf under `fits/`.
    Fit { leaf: cas_read::Leaf, rel_path: String, created: SystemTime },
    /// `ProfilePoint` leaf under `profiles/`.
    Profile { leaf: cas_read::Leaf, rel_path: String, created: SystemTime },
    /// `Pfilter`-eval leaf under `pfilters/`.
    Pfilter { leaf: cas_read::Leaf, rel_path: String, created: SystemTime },
    /// `Survey` leaf under `surveys/`.
    Survey { leaf: cas_read::Leaf, rel_path: String, created: SystemTime },
    /// `SimEnsemble` leaf under `ensembles/` (the combined multi-cell TSV).
    SimEnsemble { leaf: cas_read::Leaf, rel_path: String, created: SystemTime },
    /// A fit SEGMENT under `fits/` — the fit's *output envelope*, not a single
    /// leaf. A fit has no fit-wide `run.json` (its identity is the `fit`-level
    /// hash carried by every method leaf + the `fit.meta.json` sidecar), so this
    /// is a directory projection: it discovers the predict/quantities artifacts
    /// the fit produced. Reached by `@label`, by the segment dir, or by the
    /// fit-level hash prefix (the `fit_id` that `fit table`/`list` display).
    ///
    /// `run` is the run a `@label` named; `fit predict` writes each run's
    /// outputs under `<segment>/predict/<run id>-<key>/` (gh#896), so the envelope's
    /// output files are that run's, or the folder's one run's.
    FitEnvelope { segment: PathBuf, run: Option<String>, rel_path: String },
}

/// Resolve a user-supplied key to a single new-format leaf. Accepts either a
/// path to a `run.json`-containing leaf directory, or a hash prefix where
/// `{prefix}/{scenario}[/{seed_N}]` narrows sims further. An ambiguous prefix
/// errors, listing all candidates with their kinds.
fn resolve_any(root: &str, key: &str) -> Result<Resolved, String> {
    let cwd = std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."));

    // `@label` and a `fit.toml` are fit handles only (no other kind carries a
    // `@` sigil or is named by a config), so they resolve through the fit
    // handle resolver every fit verb uses: `@label` to the run carrying it,
    // a `fit.toml` to its problem folder.
    if let Some(resolved) = resolve_fit_handle(Path::new(root), key) {
        return resolved;
    }

    // Path form: read the leaf `run.json` (`RunRecord`) directly.
    let as_path = Path::new(key);
    // A fit SEGMENT directory addressed by path has the `fit.meta.json` sidecar
    // but no leaf `run.json` (the fit identity is a path segment, not a leaf
    // record). Project its output envelope. Checked before the `run.json` form.
    if as_path.is_dir()
        && as_path.join("fit.meta.json").exists()
        && !as_path.join("run.json").exists()
    {
        let rel_path = pathdiff_str(as_path, &cwd);
        return Ok(Resolved::FitEnvelope { segment: as_path.to_path_buf(), run: None, rel_path });
    }
    if as_path.is_dir() && as_path.join("run.json").exists() {
        let rec = match runid::read_leaf(as_path) {
            Ok(Some(leaf)) => leaf.record,
            Ok(None) => return Err(format!("{} holds no run.json", as_path.display())),
            Err(u) => {
                return Err(format!("could not read run.json at {}: {}", as_path.display(), u.reason))
            }
        };
        let kind = rec.kind;
        let leaf = cas_read::Leaf { dir: as_path.to_path_buf(), record: rec };
        let created = leaf_created(&leaf);
        let rel_path = pathdiff_str(as_path, &cwd);
        return Ok(match kind {
            runid::ArtifactKind::FitMethod => Resolved::Fit { leaf, rel_path, created },
            runid::ArtifactKind::ProfilePoint => Resolved::Profile { leaf, rel_path, created },
            runid::ArtifactKind::Pfilter => Resolved::Pfilter { leaf, rel_path, created },
            runid::ArtifactKind::Survey => Resolved::Survey { leaf, rel_path, created },
            runid::ArtifactKind::SimEnsemble => Resolved::SimEnsemble { leaf, rel_path, created },
            _ => Resolved::Sim { leaf, rel_path, created },
        });
    }

    // Hash-prefix form.
    let parts: Vec<&str> = key.split('/').collect();
    let hash_prefix = parts[0];
    let scen_filter = parts.get(1).copied();
    let seed_filter: Option<u64> = parts.get(2)
        .and_then(|s| s.strip_prefix("seed_"))
        .or_else(|| parts.get(2).copied())
        .and_then(|s| s.parse().ok());

    // New-format sims: match the run_id hex prefix, narrow by scenario/seed.
    let mut sim_matches: Vec<(cas_read::Leaf, String, SystemTime)> = Vec::new();
    for leaf in cas_read::resolve_sim_prefix(Path::new(root), hash_prefix) {
        if scen_filter.is_some_and(|s| s != leaf.level_label("scenario")) { continue; }
        if seed_filter.is_some_and(|s| s != leaf.seed()) { continue; }
        let created = leaf_created(&leaf);
        let rel = pathdiff_str(&leaf.dir, &cwd);
        sim_matches.push((leaf, rel, created));
    }

    // Fit runs and problem folders: the fit handle resolver's own prefix
    // rule (a run's `run_id`, else a folder's fit id; a prefix matching both
    // is ambiguous), so a prefix names the same fit here as in every fit verb.
    let mut fit_matches: Vec<(cas_read::Leaf, String, SystemTime)> = Vec::new();
    let mut fit_envelope_matches: Vec<(PathBuf, String)> = Vec::new();
    match crate::fit::handle::resolve_hash_prefix(Path::new(root), hash_prefix) {
        Ok(crate::fit::handle::FitTarget::Leaf(l)) => {
            if let Ok(Resolved::Fit { leaf, rel_path, created }) = fit_leaf(&l.dir, &cwd) {
                fit_matches.push((leaf, rel_path, created));
            }
        }
        Ok(crate::fit::handle::FitTarget::Segment(seg)) => {
            let rel = pathdiff_str(&seg, &cwd);
            fit_envelope_matches.push((seg, rel));
        }
        Err(crate::fit::handle::ResolveError::NotFound(_)) => {}
        Err(crate::fit::handle::ResolveError::AmbiguousRuns { runs, .. }) => {
            for leaf in runs {
                let created = leaf_created(&leaf);
                let rel = pathdiff_str(&leaf.dir, &cwd);
                fit_matches.push((leaf, rel, created));
            }
        }
        Err(crate::fit::handle::ResolveError::Ambiguous { candidates, .. }) => {
            for dir in candidates {
                let rel = pathdiff_str(&dir, &cwd);
                match fit_leaf(&dir, &cwd) {
                    Ok(Resolved::Fit { leaf, rel_path, created }) => fit_matches.push((leaf, rel_path, created)),
                    _ => fit_envelope_matches.push((dir, rel)),
                }
            }
        }
        Err(e) => return Err(e.to_string()),
    }

    // New-format profile points (M3.3): match the run_id hex prefix under
    // profiles/. A profile-base prefix is the umbrella view — use `list
    // --parent <base>` for that; here a single leaf is addressed.
    let mut profile_matches: Vec<(cas_read::Leaf, String, SystemTime)> = Vec::new();
    for leaf in cas_read::resolve_profile_prefix(Path::new(root), hash_prefix) {
        let created = leaf_created(&leaf);
        let rel = pathdiff_str(&leaf.dir, &cwd);
        profile_matches.push((leaf, rel, created));
    }

    // New-format pfilter evals (M3.3): match the run_id hex prefix under
    // pfilters/.
    let mut pfilter_matches: Vec<(cas_read::Leaf, String, SystemTime)> = Vec::new();
    for leaf in cas_read::resolve_pfilter_prefix(Path::new(root), hash_prefix) {
        let created = leaf_created(&leaf);
        let rel = pathdiff_str(&leaf.dir, &cwd);
        pfilter_matches.push((leaf, rel, created));
    }

    // New-format surveys (M3.3): match the run_id hex prefix under surveys/.
    let mut survey_matches: Vec<(cas_read::Leaf, String, SystemTime)> = Vec::new();
    for leaf in cas_read::resolve_survey_prefix(Path::new(root), hash_prefix) {
        let created = leaf_created(&leaf);
        let rel = pathdiff_str(&leaf.dir, &cwd);
        survey_matches.push((leaf, rel, created));
    }

    // New-format sim ensembles: match the run_id hex prefix under ensembles/.
    let mut ensemble_matches: Vec<(cas_read::Leaf, String, SystemTime)> = Vec::new();
    for leaf in cas_read::resolve_sim_ensemble_prefix(Path::new(root), hash_prefix) {
        let created = leaf_created(&leaf);
        let rel = pathdiff_str(&leaf.dir, &cwd);
        ensemble_matches.push((leaf, rel, created));
    }

    // Fit ENVELOPES: a fit segment whose fit-level hash (the `fit_id` that
    // `fit table`/`list` display, NOT a method-leaf run_id) starts with the
    // prefix. This resolves the whole fit's output envelope. Folded into the
    // same 0/1/many resolution below, so a prefix that collides with a method
    // leaf (`fit_matches`) surfaces as ambiguous rather than silently picked.

    match sim_matches.len() + fit_matches.len() + profile_matches.len()
        + pfilter_matches.len() + survey_matches.len() + ensemble_matches.len()
        + fit_envelope_matches.len() {
        0 => Err(format!("no run matches '{}' in {}", key, root)),
        1 => {
            if let Some((leaf, rel_path, created)) = sim_matches.into_iter().next() {
                Ok(Resolved::Sim { leaf, rel_path, created })
            } else if let Some((leaf, rel_path, created)) = fit_matches.into_iter().next() {
                Ok(Resolved::Fit { leaf, rel_path, created })
            } else if let Some((leaf, rel_path, created)) = profile_matches.into_iter().next() {
                Ok(Resolved::Profile { leaf, rel_path, created })
            } else if let Some((leaf, rel_path, created)) = pfilter_matches.into_iter().next() {
                Ok(Resolved::Pfilter { leaf, rel_path, created })
            } else if let Some((leaf, rel_path, created)) = survey_matches.into_iter().next() {
                Ok(Resolved::Survey { leaf, rel_path, created })
            } else if let Some((leaf, rel_path, created)) = ensemble_matches.into_iter().next() {
                Ok(Resolved::SimEnsemble { leaf, rel_path, created })
            } else {
                let (segment, rel_path) = fit_envelope_matches.into_iter().next().unwrap();
                Ok(Resolved::FitEnvelope { segment, run: None, rel_path })
            }
        }
        n => {
            let mut msg = format!("'{}' is ambiguous, matches {} entries:\n", key, n);
            for (_, rel, _) in &sim_matches {
                msg.push_str(&format!("  {:<14} {}\n", "sim", rel));
            }
            for (_, rel, _) in &fit_matches {
                msg.push_str(&format!("  {:<14} {}\n", "fit_method", rel));
            }
            for (_, rel, _) in &profile_matches {
                msg.push_str(&format!("  {:<14} {}\n", "profile_point", rel));
            }
            for (_, rel, _) in &pfilter_matches {
                msg.push_str(&format!("  {:<14} {}\n", "pfilter", rel));
            }
            for (_, rel, _) in &survey_matches {
                msg.push_str(&format!("  {:<14} {}\n", "survey", rel));
            }
            for (_, rel, _) in &ensemble_matches {
                msg.push_str(&format!("  {:<14} {}\n", "sim_ensemble", rel));
            }
            for (_, rel) in &fit_envelope_matches {
                msg.push_str(&format!("  {:<14} {}\n", "fit", rel));
            }
            msg.push_str("refine by appending /<scenario> and/or /<seed_N>, \
                         or pass a longer hash prefix");
            Err(msg)
        }
    }
}

/// Created-time for a new-format leaf (provenance timestamp, else dir mtime).
fn leaf_created(leaf: &cas_read::Leaf) -> SystemTime {
    leaf.record
        .provenance
        .created_at
        .as_deref()
        .and_then(parse_iso8601)
        .unwrap_or_else(|| {
            std::fs::metadata(&leaf.dir)
                .and_then(|m| m.modified())
                .unwrap_or(SystemTime::UNIX_EPOCH)
        })
}

// ── Output formatting ────────────────────────────────────────────────────────

fn print_sim_table(out: &crate::out::Out, rows: &[SimRow], now: SystemTime) {
    use comfy_table::{Table, Cell, ContentArrangement, presets::NOTHING};

    if rows.is_empty() {
        out.say(format_args!("{}", style::stderr().paint("(no cached runs)", Style::Dim)));
        return;
    }

    // NOTHING preset: plain aligned columns, no borders. Reads like `ls -l`
    // and scans cleanly for 20+ rows without box-art visual fatigue.
    let mut table = Table::new();
    table
        .load_preset(NOTHING)
        .set_content_arrangement(ContentArrangement::Dynamic)
        .set_header(vec![
            Cell::new("CREATED").add_attribute(comfy_table::Attribute::Bold),
            Cell::new("RUN_ID").add_attribute(comfy_table::Attribute::Bold),
            Cell::new("LABEL").add_attribute(comfy_table::Attribute::Bold),
            Cell::new("MODEL").add_attribute(comfy_table::Attribute::Bold),
            Cell::new("SCENARIO").add_attribute(comfy_table::Attribute::Bold),
            Cell::new("SEED").add_attribute(comfy_table::Attribute::Bold),
            Cell::new("PARAMS").add_attribute(comfy_table::Attribute::Bold),
            Cell::new("SIZE").add_attribute(comfy_table::Attribute::Bold),
            Cell::new("PATH").add_attribute(comfy_table::Attribute::Bold),
        ]);

    for r in rows {
        let rel_time   = fmt_relative_time(r.created, now);
        let model      = model_display_name(r.leaf.level_label("model"));
        let size       = format_size(r.leaf.traj_bytes());
        // The address is the run_id (the path is keyed by the factored level
        // hashes; the run_id is what `show`/`cat` resolve).
        let hash_short = short_hash_cell(&r.leaf.run_id_hex());
        let label_cell = label_cell(&r.leaf.record.provenance.label);
        // The params level label carries the sweep point (`beta=0.2`) or
        // `base` for an unswept run.
        let params     = r.leaf.level_label("params").to_string();
        table.add_row(vec![
            Cell::new(rel_time).fg(comfy_table::Color::Yellow),
            hash_short,
            label_cell,
            Cell::new(model),
            Cell::new(r.leaf.level_label("scenario")).fg(comfy_table::Color::Green),
            Cell::new(r.leaf.seed()),
            Cell::new(params).add_attribute(comfy_table::Attribute::Dim),
            Cell::new(size),
            Cell::new(&r.rel_path).fg(comfy_table::Color::Cyan),
        ]);
    }

    out.report_line(format_args!("{table}"));
}

/// Compact model identifier for the list's MODEL column. Full absolute
/// paths (`/home/x/camdl/ocaml/golden/sir_basic.ir.json`)
/// are unreadable at table width. Strip the directory and the standard
/// extensions — a reader recognizes the model by its basename.
fn model_display_name(path: &str) -> String {
    // Take the last path component after either separator.
    let base = path.rsplit(['/', '\\']).next().unwrap_or(path);
    // Strip `.ir.json` first (longer suffix), then fall back to `.camdl`.
    if let Some(stem) = base.strip_suffix(".ir.json") { return stem.to_string(); }
    if let Some(stem) = base.strip_suffix(".camdl")   { return stem.to_string(); }
    base.to_string()
}

fn print_sim_json(out: &crate::out::Out, rows: &[SimRow]) {
    for r in rows {
        let json = serde_json::to_string(&r.leaf.record).unwrap_or_default();
        out.report_line(format_args!("{}", json));
    }
}

fn print_pfilter_json(out: &crate::out::Out, rows: &[PfilterRow]) {
    for r in rows {
        let json = serde_json::to_string(&r.leaf.record).unwrap_or_default();
        out.report_line(format_args!("{}", json));
    }
}

fn print_pfilter_table(out: &crate::out::Out, rows: &[PfilterRow], now: SystemTime) {
    use comfy_table::{Table, Cell, ContentArrangement, presets::NOTHING};
    let mut table = Table::new();
    table
        .load_preset(NOTHING)
        .set_content_arrangement(ContentArrangement::Dynamic)
        .set_header(vec![
            Cell::new("CREATED").add_attribute(comfy_table::Attribute::Bold),
            Cell::new("RUN_ID").add_attribute(comfy_table::Attribute::Bold),
            Cell::new("LABEL").add_attribute(comfy_table::Attribute::Bold),
            Cell::new("MODEL").add_attribute(comfy_table::Attribute::Bold),
            Cell::new("CONFIG").add_attribute(comfy_table::Attribute::Bold),
            Cell::new("SEED").add_attribute(comfy_table::Attribute::Bold),
            Cell::new("LOGLIK").add_attribute(comfy_table::Attribute::Bold),
            Cell::new("PATH").add_attribute(comfy_table::Attribute::Bold),
        ]);
    for r in rows {
        let rel_time = fmt_relative_time(r.created, now);
        let model    = model_display_name(r.leaf.level_label("model"));
        let loglik   = r.leaf.record.inputs.get("loglik")
            .and_then(|v| v.as_f64())
            .map(|x| format!("{:.2}", x))
            .unwrap_or_else(|| "—".into());
        table.add_row(vec![
            Cell::new(rel_time).fg(comfy_table::Color::Yellow),
            short_hash_cell(&r.leaf.run_id_hex()),
            label_cell(&r.leaf.record.provenance.label),
            Cell::new(model),
            Cell::new(r.leaf.level_label("config")).add_attribute(comfy_table::Attribute::Dim),
            Cell::new(r.leaf.seed()),
            Cell::new(loglik).fg(comfy_table::Color::Magenta),
            Cell::new(&r.rel_path).fg(comfy_table::Color::Cyan),
        ]);
    }
    out.report_line(format_args!("{table}"));
}

fn print_ensemble_table(out: &crate::out::Out, rows: &[EnsembleRow], now: SystemTime) {
    use comfy_table::{Table, Cell, ContentArrangement, presets::NOTHING};
    let mut table = Table::new();
    table
        .load_preset(NOTHING)
        .set_content_arrangement(ContentArrangement::Dynamic)
        .set_header(vec![
            Cell::new("CREATED").add_attribute(comfy_table::Attribute::Bold),
            Cell::new("RUN_ID").add_attribute(comfy_table::Attribute::Bold),
            Cell::new("LABEL").add_attribute(comfy_table::Attribute::Bold),
            Cell::new("MODEL").add_attribute(comfy_table::Attribute::Bold),
            Cell::new("CONFIG").add_attribute(comfy_table::Attribute::Bold),
            Cell::new("CELLS").add_attribute(comfy_table::Attribute::Bold),
            Cell::new("SIZE").add_attribute(comfy_table::Attribute::Bold),
            Cell::new("PATH").add_attribute(comfy_table::Attribute::Bold),
        ]);
    for r in rows {
        let rel_time = fmt_relative_time(r.created, now);
        let model    = model_display_name(r.leaf.level_label("model"));
        let n_cells  = r.leaf.record.inputs.get("n_cells")
            .and_then(|v| v.as_u64())
            .map(|n| n.to_string())
            // Fall back to the dep count (one dep per cell) if inputs lacks it.
            .unwrap_or_else(|| r.leaf.record.deps.len().to_string());
        let size = format_size(
            std::fs::metadata(r.leaf.dir.join("ensemble.tsv")).map(|m| m.len()).unwrap_or(0));
        table.add_row(vec![
            Cell::new(rel_time).fg(comfy_table::Color::Yellow),
            short_hash_cell(&r.leaf.run_id_hex()),
            label_cell(&r.leaf.record.provenance.label),
            Cell::new(model),
            Cell::new(r.leaf.level_label("config")).add_attribute(comfy_table::Attribute::Dim),
            Cell::new(n_cells).fg(comfy_table::Color::Green),
            Cell::new(size),
            Cell::new(&r.rel_path).fg(comfy_table::Color::Cyan),
        ]);
    }
    out.report_line(format_args!("{table}"));
}

fn print_ensemble_json(out: &crate::out::Out, rows: &[EnsembleRow]) {
    for r in rows {
        let json = serde_json::to_string(&r.leaf.record).unwrap_or_default();
        out.report_line(format_args!("{}", json));
    }
}

/// Schema discriminator for the `camdl list --kind fit --format json` row,
/// mirroring `table_row`'s `{name, version}` form so a consumer keys on the
/// contract rather than on the shape it happens to see.
///
/// v1 is the first tagged vintage and is also the gh#895 shape: the per-leaf
/// array is `methods`, and each entry carries the leaf's `method_hash`,
/// `run_id` and `path` so two leaves of one algorithm in one segment are
/// distinguishable and addressable.
///
/// v2: `fit_toml_path` and `model` (relative to the directory the fit was run
/// from) are `source_config` and `source_model`, relative to the fit folder,
/// or `null` (gh#1009).
const FIT_LIST_ROW_SCHEMA_NAME: &str = "fit_list_row";
const FIT_LIST_ROW_SCHEMA_VERSION: u32 = 3;

fn print_profiles_json(out: &crate::out::Out, profiles: &[ProfileEntry]) {
    for p in profiles {
        let json = serde_json::json!({
            "hash":    p.hash,
            "label":   p.label,
            "model":   p.model,
            "focal":   p.focal,
            "shape":   p.shape,
            "n_seeds": p.n_seeds,
            "path":    p.rel_path,
        });
        out.report_line(format_args!("{}", serde_json::to_string(&json).unwrap_or_default()));
    }
}

fn print_profiles_table(out: &crate::out::Out, profiles: &[ProfileEntry], now: SystemTime) {
    use comfy_table::{Table, Cell, ContentArrangement, presets::NOTHING};
    let mut table = Table::new();
    table
        .load_preset(NOTHING)
        .set_content_arrangement(ContentArrangement::Dynamic)
        .set_header(vec![
            Cell::new("CREATED").add_attribute(comfy_table::Attribute::Bold),
            Cell::new("HASH").add_attribute(comfy_table::Attribute::Bold),
            Cell::new("LABEL").add_attribute(comfy_table::Attribute::Bold),
            Cell::new("MODEL").add_attribute(comfy_table::Attribute::Bold),
            Cell::new("FOCAL").add_attribute(comfy_table::Attribute::Bold),
            Cell::new("SHAPE").add_attribute(comfy_table::Attribute::Bold),
            Cell::new("SEEDS").add_attribute(comfy_table::Attribute::Bold),
            Cell::new("PATH").add_attribute(comfy_table::Attribute::Bold),
        ]);
    for p in profiles {
        let rel_time = fmt_relative_time(p.created, now);
        let model    = model_display_name(&p.model);
        let seeds_cell = if p.n_seeds == 1 {
            Cell::new("1")
        } else {
            // Multi-seed profile: highlight so the sensitivity-spread
            // surface is easy to spot in long listings.
            Cell::new(p.n_seeds.to_string())
                .fg(comfy_table::Color::Green)
                .add_attribute(comfy_table::Attribute::Bold)
        };
        let hash_short = short_hash_cell(&p.hash);
        let label_cell = label_cell(&p.label);
        table.add_row(vec![
            Cell::new(rel_time).fg(comfy_table::Color::Yellow),
            hash_short,
            label_cell,
            Cell::new(model),
            Cell::new(&p.focal).fg(comfy_table::Color::Magenta),
            Cell::new(&p.shape).add_attribute(comfy_table::Attribute::Dim),
            seeds_cell,
            Cell::new(&p.rel_path).fg(comfy_table::Color::Cyan),
        ]);
    }
    out.report_line(format_args!("{table}"));
}

/// 8-char hash prefix cell — what `camdl show <hash>` and
/// `camdl label <hash>` accept.
fn short_hash_cell(hash: &str) -> comfy_table::Cell {
    let n = hash.len().min(8);
    comfy_table::Cell::new(&hash[..n]).add_attribute(comfy_table::Attribute::Dim)
}

/// Render the LABEL cell uniformly across kinds: the trimmed label or
/// a dim "<unlabelled>" placeholder.
fn label_cell(label: &Option<String>) -> comfy_table::Cell {
    match label {
        Some(l) => comfy_table::Cell::new(l),
        None => comfy_table::Cell::new("<unlabelled>")
            .add_attribute(comfy_table::Attribute::Dim),
    }
}

fn format_size(bytes: u64) -> String {
    if bytes < 1024 { format!("{}B", bytes) }
    else if bytes < 1024 * 1024 { format!("{}K", bytes / 1024) }
    else if bytes < 1024 * 1024 * 1024 { format!("{}M", bytes / 1024 / 1024) }
    else { format!("{}G", bytes / 1024 / 1024 / 1024) }
}

// ── Parsers (stdlib only) ────────────────────────────────────────────────────

/// Parse a duration like "1h", "30m", "2d", "1w". Returns Err on unknown
/// suffix or parse failure.
#[cfg(test)]
fn parse_duration(s: &str) -> Result<std::time::Duration, String> {
    let s = s.trim();
    if s.is_empty() { return Err("empty duration".into()); }
    let (num_str, unit) = s.split_at(s.len() - 1);
    let n: u64 = num_str.parse()
        .map_err(|_| format!("bad duration '{}', expected <number><unit> (e.g. 1h, 2d)", s))?;
    let secs = match unit {
        "s" => n,
        "m" => n * 60,
        "h" => n * 3600,
        "d" => n * 86400,
        "w" => n * 86400 * 7,
        other => return Err(format!("unknown duration unit '{}', expected s/m/h/d/w", other)),
    };
    Ok(std::time::Duration::from_secs(secs))
}

/// Parse `YYYY-MM-DDTHH:MM:SSZ` back to SystemTime.
fn parse_iso8601(s: &str) -> Option<SystemTime> {
    // Format: 2026-04-16T14:23:11Z
    if s.len() != 20 || !s.ends_with('Z') { return None; }
    let year: i32 = s[0..4].parse().ok()?;
    let month: u32 = s[5..7].parse().ok()?;
    let day: u32 = s[8..10].parse().ok()?;
    let hour: u32 = s[11..13].parse().ok()?;
    let minute: u32 = s[14..16].parse().ok()?;
    let second: u32 = s[17..19].parse().ok()?;
    let secs = ir::caltime::unix_epoch_days(year as i64, month as i64, day as i64) * 86400
        + (hour * 3600 + minute * 60 + second) as i64;
    if secs < 0 { return None; }
    Some(SystemTime::UNIX_EPOCH + std::time::Duration::from_secs(secs as u64))
}

/// Produce a path relative to `base` (usually CWD), falling back to the
/// absolute string if the strip fails.
fn pathdiff_str(path: &Path, base: &Path) -> String {
    match path.strip_prefix(base) {
        Ok(rel) => rel.to_string_lossy().into_owned(),
        Err(_)  => path.to_string_lossy().into_owned(),
    }
}


// ── Fit runs: `show` and `cat` through the fit handle resolver ───────────────

/// Resolve a fit handle the way every fit verb does (`fit/handle.rs`), under
/// `root`: `@label` names the run carrying it; a `fit.toml` names the problem
/// folder of the run made from it. `None` when `key` is neither form, so the
/// caller resolves it as a path or hash prefix across every kind.
fn resolve_fit_handle(root: &Path, key: &str) -> Option<Result<Resolved, String>> {
    use crate::fit::handle::{FitRef, FitTarget};
    let cwd = std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
    match FitRef::classify(key) {
        FitRef::Label(name) => Some(
            crate::fit::handle::resolve_label(root, &name)
                .map_err(|e| e.to_string())
                .and_then(|leaf| fit_leaf(&leaf.dir, &cwd)),
        ),
        FitRef::Config(_) => Some(match crate::fit::handle::resolve_target(key) {
            Ok(FitTarget::Leaf(leaf)) => crate::fit::fit_view::leaf_segment(&leaf.dir)
                .and_then(|s| s.ok_or_else(|| format!("{} is not inside a fit's directory", leaf.dir.display())))
                .map(|segment| {
                    let rel_path = pathdiff_str(&segment, &cwd);
                    Resolved::FitEnvelope { segment, run: None, rel_path }
                }),
            Ok(FitTarget::Segment(segment)) => {
                let rel_path = pathdiff_str(&segment, &cwd);
                Ok(Resolved::FitEnvelope { segment, run: None, rel_path })
            }
            Err(e) => Err(e.to_string()),
        }),
        FitRef::RunDir(_) | FitRef::HashPrefix(_) => None,
    }
}

/// The fit run whose leaf is `dir`, as a [`Resolved::Fit`].
fn fit_leaf(dir: &Path, cwd: &Path) -> Result<Resolved, String> {
    let record = match runid::read_leaf(dir) {
        Ok(Some(leaf)) => leaf.record,
        Ok(None) => return Err(format!("{} holds no run.json", dir.display())),
        Err(u) => return Err(format!("could not read run.json at {}: {}", dir.display(), u.reason)),
    };
    let leaf = cas_read::Leaf { dir: dir.to_path_buf(), record };
    let created = leaf_created(&leaf);
    Ok(Resolved::Fit { rel_path: pathdiff_str(dir, cwd), leaf, created })
}

/// The output format `show` was asked for. `--format` takes `human` (the
/// default; `text` is the same) or `json`; anything else is refused.
fn show_json(out: &crate::out::Out, format: Option<&str>) -> bool {
    match format {
        None | Some("human") | Some("text") => false,
        Some("json") => true,
        Some(other) => out.fail(format!(
            "error: --format {other}: `camdl show` writes `human` (the default) or `json`"
        )),
    }
}

/// `show` of one fit run: who it is, the problem it belongs to, how its chains
/// started, its stored verdict, and the outputs it holds.
fn show_fit_run(out: &crate::out::Out, leaf: &cas_read::Leaf, rel_path: &str, created: SystemTime) {
    use crate::fit::run_view::{self, short};
    let rec = &leaf.record;
    let p = |k: &str| style::stdout().paint(k, Style::Dim);
    let segment = crate::fit::fit_view::leaf_segment(&leaf.dir).ok().flatten();
    let problem = segment.as_deref().map(run_view::Problem::of);
    let id = rec.run_id.to_hex().to_string();
    let label = rec.provenance.label.as_deref().map(|l| format!("  (@{l})")).unwrap_or_default();
    out.report_line(format_args!("{}     {}{label}", p("run"), short(&id, 12)));
    out.report_line(format_args!("{}    fit_method", p("kind")));
    if let Some(problem) = &problem {
        out.report_line(format_args!("{} {}", p("problem"), problem.line()));
    }
    out.report_line(format_args!("{}  {}", p("method"), method_line(leaf)));
    out.report_line(format_args!("{}  {}", p("starts"), run_view::Starts::of(rec).cell()));
    out.report_line(format_args!("{}  {}", p("status"), run_view::status_word(leaf)));
    out.report_line(format_args!("{} {}", p("verdict"), run_view::RunVerdict::of(leaf).cell()));
    if let Some(v) = rec.inputs.get("best_loglik").filter(|v| !v.is_null()) {
        out.report_line(format_args!("{} {}", p("best loglik"), v));
    }
    out.report_line(format_args!("{}    {}", p("path"), style::stdout().paint(rel_path, Style::Code)));
    out.report_line(format_args!("{} {}  ({})", p("created"),
        rec.provenance.created_at.as_deref().unwrap_or("?"),
        fmt_relative_time(created, SystemTime::now())));
    out.report_line(format_args!("{}", p("outputs")));
    let outputs = segment.as_deref().map(|s| run_view::outputs(leaf, s)).unwrap_or_else(|| {
        rec.artifacts.keys().map(|k| run_view::Output { path: k.clone(), role: "file".into() }).collect()
    });
    let width = outputs.iter().map(|o| o.path.chars().count()).max().unwrap_or(0);
    for o in &outputs {
        out.report_line(format_args!("  {:<width$}  {}", o.path, p(&o.role)));
    }
    if outputs.is_empty() {
        out.report_line(format_args!("  {}", p("(none recorded)")));
    }
}

/// `mh · 6 chains · 100000 iterations · seed 1`, from the run's record.
fn method_line(leaf: &cas_read::Leaf) -> String {
    let inputs = &leaf.record.inputs;
    let mut parts = vec![leaf.level_label("method").to_string()];
    if let Some(n) = inputs.get("n_chains").and_then(|v| v.as_u64()) {
        parts.push(format!("{n} chain{}", if n == 1 { "" } else { "s" }));
    }
    let algo = inputs.get("algorithm");
    for (key, word) in [("iterations", "iterations"), ("sweeps", "sweeps")] {
        if let Some(n) = algo.and_then(|a| a.get(key)).and_then(|v| v.as_u64()) {
            parts.push(format!("{n} {word}"));
        }
    }
    parts.push(format!("seed {}", leaf.seed()));
    parts.join(" · ")
}

/// The `fit_show` (schema v1) object for one fit run.
fn fit_show_json(leaf: &cas_read::Leaf, rel_path: &str) -> serde_json::Value {
    use crate::fit::run_view;
    let rec = &leaf.record;
    let segment = crate::fit::fit_view::leaf_segment(&leaf.dir).ok().flatten();
    let outputs: Vec<serde_json::Value> = segment
        .as_deref()
        .map(|s| run_view::outputs(leaf, s))
        .unwrap_or_default()
        .into_iter()
        .map(|o| serde_json::json!({ "path": o.path, "role": o.role }))
        .collect();
    serde_json::json!({
        "schema": { "name": run_view::FIT_SHOW_SCHEMA_NAME, "version": run_view::FIT_SHOW_SCHEMA_VERSION },
        "run_id": rec.run_id.to_hex().to_string(),
        "label": rec.provenance.label,
        "label_set_at": rec.provenance.label.as_ref().and_then(|_| crate::fit::handle::label_set_at(leaf)),
        "history_index": history_index(leaf, segment.as_deref()),
        "problem": segment.as_deref().map(|s| run_view::Problem::of(s).to_json()),
        "method": leaf.level_label("method"),
        "seed": leaf.seed(),
        "chains": rec.inputs.get("n_chains"),
        "starts": run_view::Starts::of(rec).to_json(),
        "status": run_view::status_word(leaf),
        "verdict": run_view::RunVerdict::of(leaf).to_json(),
        "path": rel_path,
        "outputs": outputs,
    })
}

/// A fit run's place in its label's history (`0` for the holder, `n` for
/// `@label~n`); `None` without a label, or for a run that holds no place
/// (not completed). The store root is the problem folder's grandparent
/// (`<root>/fits/<folder>`).
fn history_index(leaf: &cas_read::Leaf, segment: Option<&Path>) -> Option<usize> {
    let label = leaf.record.provenance.label.as_deref()?;
    let root = segment?.parent()?.parent()?;
    crate::fit::handle::label_history(root, label).iter().position(|l| l.dir == leaf.dir)
}

/// The fit runs directly under a problem folder, completed or not, in
/// directory order.
fn folder_runs(segment: &Path) -> Vec<cas_read::Leaf> {
    let mut runs: Vec<cas_read::Leaf> = cas_read::walk_records(segment)
        .into_iter()
        .filter(|(_, r)| r.kind == runid::ArtifactKind::FitMethod)
        .map(|(dir, record)| cas_read::Leaf { dir, record })
        .collect();
    runs.sort_by(|a, b| a.dir.cmp(&b.dir));
    runs
}

/// The handle that selects `leaf`: `@label` when it carries one, else a
/// 12-character prefix of its run id.
fn run_handle(leaf: &cas_read::Leaf) -> String {
    match &leaf.record.provenance.label {
        Some(l) => format!("@{l}"),
        None => crate::fit::run_view::short(&leaf.record.run_id.to_hex().to_string(), 12).to_string(),
    }
}

/// The runs block of a problem folder's `show`: one line per run, with the
/// handle that selects it.
fn show_folder_runs(out: &crate::out::Out, segment: &Path) {
    use crate::fit::run_view;
    let runs = folder_runs(segment);
    if runs.is_empty() {
        return;
    }
    out.report_line(format_args!("{}", style::stdout().paint("runs", Style::Dim)));
    for leaf in &runs {
        out.report_line(format_args!(
            "  {}  {}  {}  {}  {}",
            run_handle(leaf),
            leaf.level_label("method"),
            format_args!("seed {}", leaf.seed()),
            run_view::status_word(leaf),
            run_view::RunVerdict::of(leaf).cell(),
        ));
    }
}

/// `show --format json` on a problem folder: `{schema, problem, runs}`.
fn fit_folder_json(segment: &Path) -> serde_json::Value {
    use crate::fit::run_view;
    let cwd = std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
    let runs: Vec<serde_json::Value> = folder_runs(segment)
        .iter()
        .map(|l| fit_show_json(l, &pathdiff_str(&l.dir, &cwd)))
        .collect();
    serde_json::json!({
        "schema": { "name": run_view::FIT_SHOW_SCHEMA_NAME, "version": run_view::FIT_SHOW_SCHEMA_VERSION },
        "problem": run_view::Problem::of(segment).to_json(),
        "runs": runs,
    })
}

/// `show --format json` for every other kind: `run_show` (schema v1), the
/// leaf's kind and path beside its whole `run.json` record.
fn run_show_json(leaf: &cas_read::Leaf, kind: &str, rel_path: &str) -> serde_json::Value {
    serde_json::json!({
        "schema": { "name": "run_show", "version": 1 },
        "kind": kind,
        "path": rel_path,
        "record": leaf.record,
    })
}


// ── Fit listing: one row per run, grouped by problem folder ──────────────────

/// One fit run as `camdl list` shows it.
struct FitRunRow {
    leaf: cas_read::Leaf,
    /// The problem folder the run sits in, and its index in the listing.
    folder: usize,
    /// The run's place in its label's history: `Some(0)` for the label's
    /// current holder, `Some(n)` for `@label~n`, `None` without a label or
    /// for a run that cannot hold one (not completed).
    history_index: Option<usize>,
    rel_path: String,
    created: SystemTime,
}

/// The problem folders `list` shows, with their runs.
struct FitFolder {
    entry: FitEntry,
    problem: crate::fit::run_view::Problem,
}

/// Every fit run under the listed folders, each with its label's history
/// index (`fit::handle::label_history`).
fn fit_run_rows(root: &Path, folders: &[FitFolder]) -> Vec<FitRunRow> {
    let cwd = std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
    let mut histories: std::collections::BTreeMap<String, Vec<PathBuf>> = Default::default();
    let mut rows = Vec::new();
    for (i, f) in folders.iter().enumerate() {
        for leaf in folder_runs(&f.entry.fit_dir) {
            let history_index = leaf.record.provenance.label.clone().and_then(|label| {
                let order = histories.entry(label.clone()).or_insert_with(|| {
                    crate::fit::handle::label_history(root, &label)
                        .into_iter()
                        .map(|l| l.dir)
                        .collect()
                });
                order.iter().position(|d| d == &leaf.dir)
            });
            let created = leaf_created(&leaf);
            let rel_path = pathdiff_str(&leaf.dir, &cwd);
            rows.push(FitRunRow { leaf, folder: i, history_index, rel_path, created });
        }
    }
    rows
}

/// The label cell of a run: `m0`, `m0~1` for a superseded holder, empty
/// without a label.
fn run_label_cell(row: &FitRunRow) -> String {
    match (&row.leaf.record.provenance.label, row.history_index) {
        (Some(l), Some(0)) | (Some(l), None) => l.clone(),
        (Some(l), Some(n)) => format!("{l}~{n}"),
        (None, _) => String::new(),
    }
}

/// A run's `fit_list_row` (schema v3) object.
fn fit_list_row_json(row: &FitRunRow, problem: &crate::fit::run_view::Problem) -> serde_json::Value {
    use crate::fit::run_view;
    let rec = &row.leaf.record;
    serde_json::json!({
        "schema": { "name": FIT_LIST_ROW_SCHEMA_NAME, "version": FIT_LIST_ROW_SCHEMA_VERSION },
        "kind": "fit",
        "problem": problem.to_json(),
        "label": rec.provenance.label,
        "label_set_at": rec.provenance.label.as_ref().and_then(|_| crate::fit::handle::label_set_at(&row.leaf)),
        "history_index": row.history_index,
        "run_id": rec.run_id.to_hex().to_string(),
        "method": row.leaf.level_label("method"),
        "seed": row.leaf.seed(),
        "starts": run_view::Starts::of(rec).to_json(),
        "status": run_view::status_word(&row.leaf),
        "verdict": run_view::RunVerdict::of(&row.leaf).to_json(),
        "path": row.rel_path,
    })
}

/// One line of a folder's table: a run, or several seeds of one method
/// collapsed.
struct TableLine {
    label: String,
    run: String,
    method: String,
    seed: String,
    starts: String,
    status: String,
    verdict: String,
}

/// `1-3`, `1,2,5`: a sorted seed set, ranges joined.
fn seed_range(seeds: &[u64]) -> String {
    let mut s: Vec<u64> = seeds.to_vec();
    s.sort_unstable();
    s.dedup();
    if s.len() > 1 && s.windows(2).all(|w| w[1] == w[0] + 1) {
        return format!("{}-{}", s[0], s[s.len() - 1]);
    }
    s.iter().map(u64::to_string).collect::<Vec<_>>().join(",")
}

/// The combined verdict of collapsed seeds: the one cell when they agree,
/// else each verdict's leading words with a count (`2 converged, 1 not
/// converged`, `1 unreadable`).
fn combined_verdict(cells: &[String]) -> String {
    if cells.windows(2).all(|w| w[0] == w[1]) {
        return cells.first().cloned().unwrap_or_default();
    }
    let mut counts: Vec<(String, usize)> = Vec::new();
    for c in cells {
        let head = c.split([':', '(']).next().unwrap_or(c).trim().to_string();
        match counts.iter_mut().find(|(h, _)| *h == head) {
            Some((_, n)) => *n += 1,
            None => counts.push((head, 1)),
        }
    }
    counts.into_iter().map(|(h, n)| format!("{n} {h}")).collect::<Vec<_>>().join(", ")
}

/// The table lines of one folder's runs. Unless `expand`, unlabelled runs of
/// one method that differ only in seed collapse to one line with the seed
/// range and a combined verdict; a labelled run always has its own line.
fn folder_lines(rows: &[&FitRunRow], expand: bool) -> Vec<TableLine> {
    use crate::fit::run_view::{self, short};
    let line = |r: &FitRunRow| TableLine {
        label: run_label_cell(r),
        run: short(&r.leaf.record.run_id.to_hex().to_string(), 6).to_string(),
        method: r.leaf.level_label("method").to_string(),
        seed: r.leaf.seed().to_string(),
        starts: run_view::Starts::of(&r.leaf.record).cell(),
        status: run_view::status_word(&r.leaf).to_string(),
        verdict: run_view::RunVerdict::of(&r.leaf).cell(),
    };
    if expand {
        return rows.iter().map(|r| line(r)).collect();
    }
    // Group unlabelled runs by everything but the seed: the method level.
    let method_key = |r: &FitRunRow| {
        r.leaf.record.levels.iter().find(|l| l.name == "method")
            .map(|l| format!("{}-{}", l.label, l.hash.to_hex()))
    };
    let mut out: Vec<TableLine> = Vec::new();
    let mut grouped: Vec<(Option<String>, Vec<&FitRunRow>)> = Vec::new();
    for r in rows {
        if r.leaf.record.provenance.label.is_some() {
            grouped.push((None, vec![r]));
            continue;
        }
        let key = method_key(r);
        match grouped.iter_mut().find(|(k, _)| k.is_some() && *k == key) {
            Some((_, g)) => g.push(r),
            None => grouped.push((key.or_else(|| Some(r.leaf.dir.display().to_string())), vec![r])),
        }
    }
    for (_, g) in grouped {
        if g.len() == 1 {
            out.push(line(g[0]));
            continue;
        }
        let seeds: Vec<u64> = g.iter().map(|r| r.leaf.seed()).collect();
        let cells: Vec<String> = g.iter().map(|r| run_view::RunVerdict::of(&r.leaf).cell()).collect();
        let statuses: Vec<String> = g.iter().map(|r| run_view::status_word(&r.leaf).to_string()).collect();
        let first = line(g[0]);
        out.push(TableLine {
            run: format!("{} runs", g.len()),
            seed: seed_range(&seeds),
            status: combined_verdict(&statuses),
            verdict: combined_verdict(&cells),
            ..first
        });
    }
    out
}

/// The fits section of `camdl list`: per problem folder, a header line and a
/// table of its runs.
fn print_fit_runs_table(out: &crate::out::Out, folders: &[FitFolder], rows: &[FitRunRow], expand: bool) {
    use comfy_table::{presets::NOTHING, Attribute, Cell, ContentArrangement, Table};
    let mut unlabelled = 0usize;
    for (i, f) in folders.iter().enumerate() {
        let mine: Vec<&FitRunRow> = rows.iter().filter(|r| r.folder == i).collect();
        if mine.is_empty() {
            continue;
        }
        unlabelled += mine.iter().filter(|r| r.leaf.record.provenance.label.is_none()).count();
        out.report_line(format_args!(
            "{} {} · {}",
            style::stdout().paint("problem", Style::Bold),
            f.problem.line(),
            style::stdout().paint(&f.entry.rel_path, Style::Code)
        ));
        let mut table = Table::new();
        table.load_preset(NOTHING).set_content_arrangement(ContentArrangement::Disabled);
        table.set_header(
            ["LABEL", "RUN", "METHOD", "SEED", "STARTS", "STATUS", "VERDICT"]
                .map(|h| Cell::new(h).add_attribute(Attribute::Bold)),
        );
        for l in folder_lines(&mine, expand) {
            table.add_row(vec![
                Cell::new(l.label),
                Cell::new(l.run).add_attribute(Attribute::Dim),
                Cell::new(l.method).fg(comfy_table::Color::Green),
                Cell::new(l.seed),
                Cell::new(l.starts),
                Cell::new(l.status),
                Cell::new(l.verdict),
            ]);
        }
        for line in table.to_string().lines() {
            out.report_line(format_args!(" {}", line.trim_end()));
        }
    }
    crate::fit::fit_table::emit_unlabelled_warning(out, unlabelled);
}

// ── Tests ────────────────────────────────────────────────────────────────────

/// The `engine` line of a record: the readable version that ran it and the
/// engine's build fingerprint (`run.json`'s `engine_version`), e.g.
/// `0.4.0+abc1234.dirty-3f9c21e0 (engine 9514b53cefba5cfc…)`.
fn engine_line(rec: &runid::RunRecord) -> String {
    let id = &rec.engine_version;
    match rec.provenance.camdl_version.as_deref() {
        Some(v) if v != id => format!("{v} (engine {})", id),
        _ => id.clone(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_duration_ok() {
        use std::time::Duration;
        assert_eq!(parse_duration("30s").unwrap(), Duration::from_secs(30));
        assert_eq!(parse_duration("5m").unwrap(), Duration::from_secs(300));
        assert_eq!(parse_duration("2h").unwrap(), Duration::from_secs(7200));
        assert_eq!(parse_duration("1d").unwrap(), Duration::from_secs(86400));
        assert_eq!(parse_duration("1w").unwrap(), Duration::from_secs(86400 * 7));
    }

    #[test]
    fn parse_duration_bad() {
        assert!(parse_duration("").is_err());
        assert!(parse_duration("abc").is_err());
        assert!(parse_duration("5y").is_err()); // y not supported; use weeks for alpha
        assert!(parse_duration("1.5h").is_err());
    }

    #[test]
    fn parse_iso8601_roundtrip() {
        use crate::cas::iso8601_utc;
        let times = [
            std::time::UNIX_EPOCH,
            std::time::UNIX_EPOCH + std::time::Duration::from_secs(946684800), // 2000-01-01
            std::time::UNIX_EPOCH + std::time::Duration::from_secs(1776297600), // 2026-04-16
            std::time::UNIX_EPOCH + std::time::Duration::from_secs(1709210096), // 2024-02-29T12:34:56Z
        ];
        for t in times {
            let s = iso8601_utc(t);
            let parsed = parse_iso8601(&s).expect("should parse");
            assert_eq!(parsed, t, "round-trip failed for {}", s);
        }
    }

    #[test]
    fn format_size_buckets() {
        assert_eq!(format_size(500), "500B");
        assert_eq!(format_size(2048), "2K");
        assert_eq!(format_size(5 * 1024 * 1024), "5M");
    }

    #[test]
    fn model_display_name_strips_dir_and_extension() {
        // Absolute path + .ir.json → basename without extension
        assert_eq!(
            model_display_name("/home/x/camdl/ocaml/golden/sir_basic.ir.json"),
            "sir_basic"
        );
        // .camdl extension also stripped
        assert_eq!(model_display_name("../models/seir.camdl"), "seir");
        // No extension → bare basename
        assert_eq!(model_display_name("/tmp/custom"), "custom");
        // Bare basename unchanged (still strips known extension)
        assert_eq!(model_display_name("sir.ir.json"), "sir");
    }

    // ── Transitional reader: new-format (RunRecord) sim resolution ──────────

    /// Write a new-format sim leaf (RunRecord run.json + traj.tsv) at its
    /// factored `store_path`. `salt` varies the seed-level hash so two records
    /// land at distinct paths; `run_id` is set directly to exercise prefix
    /// resolution.
    fn write_sim_record(
        root: &Path,
        run_id: runid::ContentHash,
        seed: u64,
        salt: u8,
    ) -> PathBuf {
        let h = |b: u8| runid::ContentHash::from_bytes([b; 32]);
        let lvl = |name: &str, label: String, b: u8| runid::LevelId {
            name: name.into(), label, hash: h(b), schema_version: 1,
        };
        let levels = vec![
            lvl("model", "sir".into(), 1),
            lvl("config", "chain_binomial-dt1".into(), 2),
            lvl("params", "base".into(), 3),
            lvl("scenario", "baseline".into(), 4),
            lvl("seed", format!("seed_{seed}"), salt),
        ];
        let dir = runid::store_path(root, runid::ArtifactKind::Sim, &levels);
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("traj.tsv"), "t\tS\n0\t100\n").unwrap();
        let rec = runid::RunRecord {
            format_version: runid::FORMAT_VERSION,
            kind: runid::ArtifactKind::Sim,
            run_id,
            hash_version: runid::HASH_VERSION,
            ir_version: "0.7".into(),
            engine_version: "test".into(),
            levels,
            deps: vec![],
            status: runid::RunStatus::Completed,
            failure: None,
            artifacts: Default::default(),
            output_schema: Default::default(),
            children: Default::default(),
            calendar: ir::caltime::CalendarAxis::parse("days", None).unwrap(),
            inputs: serde_json::Value::Null,
            provenance: runid::Provenance::default(),
        };
        std::fs::write(dir.join("run.json"), serde_json::to_string(&rec).unwrap()).unwrap();
        dir
    }

    #[test]
    fn resolve_sim_by_run_id_prefix_and_path() {
        let tmp = tempfile::tempdir().unwrap();
        let dir = write_sim_record(
            tmp.path(),
            runid::ContentHash::from_bytes([0xab; 32]),
            42,
            10,
        );
        let root = tmp.path().to_str().unwrap();

        // run_id hex prefix.
        match resolve_any(root, "abab").unwrap() {
            Resolved::Sim { leaf, .. } => assert_eq!(leaf.seed(), 42),
            _ => panic!("expected new-format Sim"),
        }
        // /scenario narrowing.
        match resolve_any(root, "abab/baseline").unwrap() {
            Resolved::Sim { leaf, .. } => assert_eq!(leaf.seed(), 42),
            _ => panic!("expected Sim"),
        }
        // Path form.
        match resolve_any(root, dir.to_str().unwrap()).unwrap() {
            Resolved::Sim { .. } => {}
            _ => panic!("expected Sim from path"),
        }
        // No match.
        assert!(resolve_any(root, "ffff").is_err());
    }

    #[test]
    fn resolve_sim_ambiguous_prefix_lists_candidates() {
        // Two sims whose run_ids share the prefix "ab" but diverge after.
        let tmp = tempfile::tempdir().unwrap();
        write_sim_record(tmp.path(), runid::ContentHash::from_bytes([0xab; 32]), 1, 10);
        let mut b = [0xab; 32];
        b[1] = 0xcd; // hex "abcd…"
        write_sim_record(tmp.path(), runid::ContentHash::from_bytes(b), 2, 20);
        let root = tmp.path().to_str().unwrap();

        // "ab" matches both → ambiguous, with the sim kind label listed.
        let err = resolve_any(root, "ab").expect_err("ambiguous prefix must reject");
        assert!(err.contains("ambiguous"), "got: {}", err);
        assert!(err.contains("matches 2"), "got: {}", err);
        assert!(err.contains("sim"), "expected kind label: got {}", err);

        // "abab" uniquely resolves the first.
        match resolve_any(root, "abab").unwrap() {
            Resolved::Sim { leaf, .. } => assert_eq!(leaf.seed(), 1),
            _ => panic!("expected Sim"),
        }
    }
}
