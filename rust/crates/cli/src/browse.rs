//! `camdl list`, `camdl show`, `camdl cat` — browse the content-addressable
//! store written by `camdl simulate --cas` and `camdl batch run`.
//!
//! All three walk `./results/sims/` by default. For alpha, walk is
//! unindexed — fast enough for thousands of runs. A persistent index
//! can be added later if needed.

use std::path::{Path, PathBuf};
use std::time::SystemTime;

use owo_colors::OwoColorize;

use crate::cas_read;
use crate::run_meta::{Run, RunKind};
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

/// Discover the new-format sim leaves under `root/sims/` for `list`.
fn discover_sim_rows(root: &str) -> Vec<SimRow> {
    let cwd = std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
    cas_read::walk_sim_leaves(Path::new(root))
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

/// Shared preamble for the **legacy** `run_meta::Run` kinds (fit/profile/
/// survey): read `run.json` and derive the display time + cwd-relative path.
/// Returns `None` when the directory isn't a (legacy) run.
///
/// M3-DELETION-BOUND (gh#147): the transitional reader dispatches new-format
/// `sims/` through [`cas_read`] and the legacy kinds through this path. When
/// M3 migrates the fit/profile/survey *writers* to `RunRecord`, delete this
/// helper and all `discover_fits`/`discover_profiles`/`discover_surveys` /
/// `ResolvedRun` machinery in the same change — the generic walker subsumes
/// them. The dual path is debt with a due date, not a keeper.
fn load_run_common(dir: &Path, cwd: &Path) -> Option<(Run, SystemTime, String)> {
    let run = Run::read(dir).ok()?;
    let created = parse_iso8601(&run.created_at)
        .unwrap_or_else(|| std::fs::metadata(dir)
            .and_then(|m| m.modified())
            .unwrap_or(SystemTime::UNIX_EPOCH));
    let rel_path = pathdiff_str(dir, cwd);
    Some((run, created, rel_path))
}

// ── cmd_list ─────────────────────────────────────────────────────────────────

/// `--kind` filter: which of sims / fits / profiles / surveys to
/// surface. `All` is the default and includes all four.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum KindFilter { Sim, Fit, Profile, Survey, All }

impl KindFilter {
    fn includes_sims(self)     -> bool { matches!(self, Self::Sim     | Self::All) }
    fn includes_fits(self)     -> bool { matches!(self, Self::Fit     | Self::All) }
    fn includes_profiles(self) -> bool { matches!(self, Self::Profile | Self::All) }
    fn includes_surveys(self)  -> bool { matches!(self, Self::Survey  | Self::All) }
}

pub fn cmd_list(a: &crate::args::ListArgs) {
    // --parent=HASH: enumerate the grid-point × start runs of one
    // specific profile. Takes precedence over the default sim/fit
    // enumeration because it's a more specific request; the other
    // filters (since, limit, format) still apply.
    if let Some(parent_hash) = a.parent.as_ref() {
        list_profile_children(&a.root.to_string_lossy(), parent_hash, a);
        return;
    }

    let root = a.root.to_string_lossy();
    let filter_since: Option<std::time::Duration> = a.since.as_ref().map(|d| d.0);
    let filter_kind = match a.kind.as_str() {
        "sim" | "simulate"      => KindFilter::Sim,
        "fit"                   => KindFilter::Fit,
        "profile" | "profiles"  => KindFilter::Profile,
        "survey" | "surveys"    => KindFilter::Survey,
        _                       => KindFilter::All,
    };
    let format_json = a.format.as_deref() == Some("json");

    let runs = if !filter_kind.includes_sims() {
        Vec::new()
    } else {
        discover_sim_rows(&root)
    };
    let fits = if !filter_kind.includes_fits() {
        Vec::new()
    } else {
        discover_fits(&root).unwrap_or_else(|e| { eprintln!("error: {}", e); std::process::exit(1); })
    };
    let profiles = if !filter_kind.includes_profiles() {
        Vec::new()
    } else {
        discover_profiles(&root).unwrap_or_else(|e| { eprintln!("error: {}", e); std::process::exit(1); })
    };
    let surveys = if !filter_kind.includes_surveys() {
        Vec::new()
    } else {
        discover_surveys(&root).unwrap_or_else(|e| { eprintln!("error: {}", e); std::process::exit(1); })
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

    let mut filtered_fits: Vec<FitEntry> = fits.into_iter()
        .filter(|f| a.model.as_deref().is_none_or(|m| f.meta.model.contains(m)))
        .filter(|_| a.scenario.is_none())
        .filter(|f| match filter_since {
            Some(dur) => now.duration_since(f.created).is_ok_and(|d| d <= dur),
            None => true,
        })
        .collect();
    filtered_fits.sort_by(|x, y| y.created.cmp(&x.created));

    let mut filtered_profiles: Vec<ProfileEntry> = profiles.into_iter()
        .filter(|p| a.model.as_deref().is_none_or(|m| p.model.contains(m)))
        .filter(|_| a.scenario.is_none())
        .filter(|p| match filter_since {
            Some(dur) => now.duration_since(p.created).is_ok_and(|d| d <= dur),
            None => true,
        })
        .collect();
    filtered_profiles.sort_by(|x, y| y.created.cmp(&x.created));

    let mut filtered_surveys: Vec<SurveyEntry> = surveys.into_iter()
        .filter(|s| a.model.as_deref().is_none_or(|m| s.model.contains(m)))
        .filter(|_| a.scenario.is_none())
        .filter(|s| match filter_since {
            Some(dur) => now.duration_since(s.created).is_ok_and(|d| d <= dur),
            None => true,
        })
        .collect();
    filtered_surveys.sort_by(|x, y| y.created.cmp(&x.created));

    if !a.all {
        filtered_runs.truncate(a.limit);
        filtered_fits.truncate(a.limit);
        filtered_profiles.truncate(a.limit);
        filtered_surveys.truncate(a.limit);
    }

    if format_json {
        print_sim_json(&filtered_runs);
        print_fits_json(&filtered_fits);
        print_profiles_json(&filtered_profiles);
        print_surveys_json(&filtered_surveys);
    } else {
        let any_other = !filtered_fits.is_empty()
            || !filtered_profiles.is_empty()
            || !filtered_surveys.is_empty();
        if !filtered_fits.is_empty() {
            eprintln!("{}", "fits".bold());
            print_fits_table(&filtered_fits, now);
            eprintln!();
        }
        if !filtered_profiles.is_empty() {
            eprintln!("{}", "profiles".bold());
            print_profiles_table(&filtered_profiles, now);
            eprintln!();
        }
        if !filtered_surveys.is_empty() {
            eprintln!("{}", "surveys".bold());
            print_surveys_table(&filtered_surveys, now);
            eprintln!();
        }
        if !filtered_runs.is_empty() || !any_other {
            if any_other { eprintln!("{}", "sims".bold()); }
            print_sim_table(&filtered_runs, now);
        }
    }
}

/// Enumerate the grid-point × start children of one profile, identified
/// by a hash prefix. Scans `<root>/profiles/*/points/*/start_*/run.json`
/// and prints those whose `parent_profile_hash` starts with the given
/// prefix. Minimal output — a richer "loglik + wall_time per point" view
/// is a v2 follow-up per the profile-CAS proposal.
fn list_profile_children(
    root: &str,
    parent_hash_prefix: &str,
    a: &crate::args::ListArgs,
) {
    use crate::run_meta::{Run, RunKind};

    let root_path = std::path::Path::new(root);
    let profiles_root = root_path.join("profiles");
    if !profiles_root.exists() {
        eprintln!("no profiles under {}", profiles_root.display());
        return;
    }

    // Pass 1: find any ReplicateSet umbrellas whose `parent_hash` (the
    // umbrella's own Run.hash) or `inner_content_hash` (the seed-free
    // hash shared across replicate children) matches the prefix.
    // Multi-seed profile umbrellas store each per-seed child's profile
    // content hash as a `parent_profile_hash` on the deeper FitStage
    // leaves, so we need to expand the user-supplied umbrella prefix
    // into the set of per-seed hashes before the leaf walk.
    let mut expanded_prefixes: Vec<String> = vec![parent_hash_prefix.to_string()];
    for dir in walkdir_all(&profiles_root) {
        let rj = dir.join("run.json");
        if !rj.exists() { continue; }
        let Ok(text) = std::fs::read_to_string(&rj) else { continue; };
        let Ok(run) = serde_json::from_str::<Run>(&text) else { continue; };
        if let RunKind::ReplicateSet(ref m) = run.kind {
            let umbrella_matches =
                run.hash.starts_with(parent_hash_prefix)
                || m.inner_content_hash.starts_with(parent_hash_prefix);
            if !umbrella_matches { continue; }
            // For each child, peek at its run.json to get the per-seed
            // profile content hash and add it to the expanded set.
            for key in &m.keys {
                let child_dir = dir.join("replicates").join(key);
                let crj = child_dir.join("run.json");
                let Ok(ctext) = std::fs::read_to_string(&crj) else { continue; };
                let Ok(crun) = serde_json::from_str::<Run>(&ctext) else { continue; };
                if matches!(crun.kind, RunKind::Profile(_)) {
                    expanded_prefixes.push(crun.hash);
                }
            }
        }
    }

    let mut matches: Vec<(std::path::PathBuf, Run)> = Vec::new();
    for dir in walkdir_all(&profiles_root) {
        let rj = dir.join("run.json");
        if !rj.exists() { continue; }
        let Ok(text) = std::fs::read_to_string(&rj) else { continue; };
        let Ok(run) = serde_json::from_str::<Run>(&text) else { continue; };
        if let RunKind::FitStage(ref m) = run.kind {
            let parent = m.parent_profile_hash.as_deref();
            if parent.is_some_and(|h| {
                expanded_prefixes.iter().any(|p| h.starts_with(p))
            }) {
                matches.push((dir, run));
            }
        }
    }

    if matches.is_empty() {
        eprintln!("no grid-point runs found with parent hash prefix '{}'", parent_hash_prefix);
        return;
    }

    // Sort by (point_idx, start_idx) for natural grid-traversal order.
    matches.sort_by_key(|(_, run)| match &run.kind {
        RunKind::FitStage(m) => (m.profile_point_idx.unwrap_or(usize::MAX),
                                  m.profile_start_idx.unwrap_or(usize::MAX)),
        _ => (usize::MAX, usize::MAX),
    });

    let limit = if a.all { matches.len() } else { a.limit.min(matches.len()) };

    if a.format.as_deref() == Some("json") {
        // Minimal JSON array for scripting. Full `Run` round-trip.
        let slice: Vec<&Run> = matches.iter().take(limit).map(|(_, r)| r).collect();
        match serde_json::to_string_pretty(&slice) {
            Ok(s)  => println!("{}", s),
            Err(e) => eprintln!("json error: {}", e),
        }
        return;
    }

    eprintln!("{}", "profile grid-point starts".bold());
    eprintln!("  {:<6} {:<6} {:>14} {:>10}  {}",
        "point", "start", "best_loglik", "wall_s", "path");
    for (dir, run) in matches.iter().take(limit) {
        let RunKind::FitStage(ref m) = run.kind else { continue; };
        let point = m.profile_point_idx.map(|n| n.to_string()).unwrap_or("?".into());
        let start = m.profile_start_idx.map(|n| n.to_string()).unwrap_or("?".into());
        let ll = m.best_loglik
            .map(|x| format!("{:.2}", x))
            .unwrap_or_else(|| "—".into());
        let wall = match run.status.wall_time_seconds() {
            Some(t) => format!("{:.1}", t),
            None    => "running".to_string(),
        };
        let rel = dir.strip_prefix(root_path)
            .unwrap_or(dir)
            .display()
            .to_string();
        eprintln!("  {:<6} {:<6} {:>14} {:>10}  {}", point, start, ll, wall, rel.dimmed());
    }
    if matches.len() > limit {
        eprintln!("  ... {} more (use --all to show)", matches.len() - limit);
    }
}

// ── cmd_show ─────────────────────────────────────────────────────────────────

pub fn cmd_show(a: &crate::args::ShowArgs) {
    let root = a.root.to_string_lossy();
    match resolve_any(&root, &a.target) {
        Ok(Resolved::Sim { leaf, rel_path, created }) => show_sim_record(&leaf, &rel_path, created),
        Ok(Resolved::Legacy(r)) => show(&r),
        Err(e) => {
            eprintln!("error: {}", e);
            std::process::exit(1);
        }
    }
}

/// Render a new-format (`RunRecord`) sim: the factored levels, the run_id
/// address, and provenance. Mirrors the legacy `show_simulate` layout.
fn show_sim_record(leaf: &cas_read::Leaf, rel_path: &str, created: SystemTime) {
    let rec = &leaf.record;
    println!("{}", "path".bright_black()); println!("  {}", rel_path.cyan());
    println!("{}", "kind".bright_black()); println!("  sim");
    if let Some(ref l) = rec.provenance.label {
        println!("{}", "label".bright_black()); println!("  {}", l);
    }
    println!("{}", "model".bright_black()); println!("  {}", leaf.level_label("model"));
    println!("{}", "scenario".bright_black()); println!("  {}", leaf.level_label("scenario"));
    println!("{}", "seed".bright_black()); println!("  {}", leaf.seed());
    println!("{}", "config".bright_black()); println!("  {}", leaf.level_label("config"));
    println!("{}", "run_id".bright_black()); println!("  {}", rec.run_id.to_hex().dimmed());
    println!("{}", "levels".bright_black());
    for lvl in &rec.levels {
        println!("  {:<9} {}-{}", lvl.name, lvl.label, lvl.hash.short8().dimmed());
    }
    println!("{}", "trajectory".bright_black());
    println!("  {} bytes", leaf.traj_bytes());
    println!("{}", "created".bright_black());
    println!("  {}  ({})",
        rec.provenance.created_at.as_deref().unwrap_or("?"),
        fmt_relative_time(created, SystemTime::now()));
    println!("{}", "engine".bright_black()); println!("  {}", rec.engine_version);
    println!("{}", "argv".bright_black());
    println!("  {}", rec.provenance.argv.join(" "));
}

/// Kind-agnostic show entry point. One match on `run.kind`; per-kind
/// renderers below. Adding a new `RunKind` variant gets a compiler
/// error here until a renderer is wired in.
fn show(r: &ResolvedRun) {
    match &r.run.kind {
        RunKind::Simulate(_)     => show_simulate(r),
        RunKind::Fit(_)          => show_fit(r),
        RunKind::FitStage(_)     => show_fit_stage(r),
        RunKind::Profile(_)      => show_profile_leaf(r),
        RunKind::ReplicateSet(_) => show_replicate_set(r),
        RunKind::Survey(_)       => show_survey(r),
    }
}

/// Header shared by every kind: path, kind label, optional label,
/// timing/version/argv. Keeps the per-kind renderers focused on
/// kind-specific fields.
fn show_header(r: &ResolvedRun) {
    println!("{}", "path".bright_black()); println!("  {}", r.rel_path.cyan());
    println!("{}", "kind".bright_black()); println!("  {}", kind_label(&r.run.kind));
    if let Some(ref l) = r.run.label {
        println!("{}", "label".bright_black()); println!("  {}", l);
    }
}

fn show_footer(r: &ResolvedRun) {
    println!("{}", "created".bright_black());
    println!("  {}  ({})", r.run.created_at,
        fmt_relative_time(r.created, SystemTime::now()));
    println!("{}", "version".bright_black()); println!("  {}", r.run.version);
    println!("{}", "wall time".bright_black());
    match r.run.status.wall_time_seconds() {
        Some(t) => println!("  {:.1}s", t),
        None    => println!("  (running)"),
    }
    println!("{}", "argv".bright_black());
    println!("  {}", r.run.argv.join(" "));
}

fn show_simulate(r: &ResolvedRun) {
    let RunKind::Simulate(m) = &r.run.kind else { unreachable!() };
    show_header(r);
    println!("{}", "model".bright_black()); println!("  {}", m.model);
    println!("{}", "scenario".bright_black()); println!("  {}", m.scenario);
    println!("{}", "seed".bright_black()); println!("  {}", m.seed);
    println!("{}", "backend".bright_black());
    println!("  {} (dt = {})", m.backend, m.dt);
    println!("{}", "hashes".bright_black());
    println!("  sim   {}", m.sim_hash.dimmed());
    println!("  scen  {}", m.scen_hash.dimmed());
    println!("  model {}", m.model_hash.dimmed());
    if let Some(fh) = &m.from_fit_hash {
        println!("  from-fit {}", fh.dimmed());
    }
    let traj_bytes = std::fs::metadata(r.abs_path.join("traj.tsv"))
        .map(|m| m.len()).unwrap_or(0);
    println!("{}", "trajectory".bright_black());
    println!("  {} bytes", traj_bytes);
    show_footer(r);
}

fn show_fit(r: &ResolvedRun) {
    let RunKind::Fit(m) = &r.run.kind else { unreachable!() };
    show_header(r);
    println!("{}", "model".bright_black()); println!("  {}", m.model);
    println!("{}", "fit.toml".bright_black()); println!("  {}", m.fit_toml_path);
    println!("{}", "estimate".bright_black()); println!("  {}", m.estimated.join(", "));
    if !m.fixed.is_empty() {
        let mut fx: Vec<_> = m.fixed.iter().collect();
        fx.sort_by_key(|(k, _)| k.to_string());
        let items: Vec<String> = fx.iter().map(|(k, v)| format!("{}={}", k, v)).collect();
        println!("{}", "fixed".bright_black()); println!("  {}", items.join(", "));
    }
    println!("{}", "stages".bright_black());
    println!("  {}", m.stages_declared.join(", "));
    println!("{}", "hashes".bright_black());
    println!("  fit      {}", r.run.hash.dimmed());
    println!("  model    {}", m.model_hash.dimmed());
    println!("  fit.toml {}", m.fit_toml_hash.dimmed());
    show_footer(r);
}

fn show_fit_stage(r: &ResolvedRun) {
    let RunKind::FitStage(m) = &r.run.kind else { unreachable!() };
    show_header(r);
    println!("{}", "stage".bright_black());
    println!("  {} (method: {})", m.stage, m.method);
    println!("{}", "seed".bright_black()); println!("  {}", m.seed);
    println!("{}", "chains".bright_black()); println!("  {}", m.n_chains);
    if let Some(ll) = m.best_loglik {
        let chain = m.best_chain.map(|c| format!(" (chain {})", c + 1)).unwrap_or_default();
        println!("{}", "best loglik".bright_black());
        println!("  {:.2}{}", ll, chain);
    }
    if !m.algorithm.is_null() {
        println!("{}", "algorithm".bright_black());
        let pretty = serde_json::to_string_pretty(&m.algorithm).unwrap_or_default();
        for line in pretty.lines() { println!("  {}", line.dimmed()); }
    }
    if let Some(sf) = &m.starts_from {
        let h = sf.stage_hash.as_deref().unwrap_or("?");
        let short = &h[..h.len().min(16)];
        println!("{}", "starts from".bright_black());
        println!("  {} ({})", sf.stage, short.dimmed());
    }
    if let Some(ref hash) = m.parent_profile_hash {
        let short = &hash[..hash.len().min(16)];
        println!("{}", "parent profile".bright_black());
        println!("  {}", short.dimmed());
        if let (Some(pi), Some(si)) = (m.profile_point_idx, m.profile_start_idx) {
            println!("  point {} / start {}", pi, si);
        }
    }
    if let Some(ref df) = m.derived_from {
        println!("{}", "derived from".bright_black());
        println!("  {}", df);
    }
    println!("{}", "hashes".bright_black());
    println!("  stage {}", r.run.hash.dimmed());
    println!("  fit   {}", m.fit_hash.dimmed());
    show_footer(r);
}

fn show_profile_leaf(r: &ResolvedRun) {
    let RunKind::Profile(m) = &r.run.kind else { unreachable!() };
    show_header(r);
    println!("{}", "model".bright_black()); println!("  {}", m.model);
    println!("{}", "focal params".bright_black());
    println!("  {}", m.focal_params.join(", "));
    println!("{}", "grid".bright_black());
    for axis in &m.grid {
        let n = axis.values.len();
        let preview = if n <= 6 {
            axis.values.iter().map(|v| format!("{}", v)).collect::<Vec<_>>().join(", ")
        } else {
            let head: Vec<String> = axis.values.iter().take(3).map(|v| format!("{}", v)).collect();
            let tail: Vec<String> = axis.values.iter().rev().take(2).rev().map(|v| format!("{}", v)).collect();
            format!("{}, …, {}", head.join(", "), tail.join(", "))
        };
        println!("  {}: {} values [{}]", axis.param, n, preview);
    }
    println!("{}", "starts".bright_black()); println!("  {} per grid point", m.n_starts);
    println!("{}", "total jobs".bright_black()); println!("  {}", m.total_jobs);
    println!("{}", "seed".bright_black()); println!("  {}", m.seed_base);
    let profile_tsv = r.abs_path.join("profile.tsv");
    if profile_tsv.exists() {
        let bytes = std::fs::metadata(&profile_tsv).map(|m| m.len()).unwrap_or(0);
        println!("{}", "rollup".bright_black());
        println!("  profile.tsv ({} bytes)", bytes);
    }
    println!("{}", "hashes".bright_black());
    println!("  profile        {}", r.run.hash.dimmed());
    println!("  model          {}", m.model_hash.dimmed());
    println!("  if2 config     {}", m.if2_config_hash.dimmed());
    println!("  base params    {}", m.base_params_hash.dimmed());
    show_footer(r);
}

fn show_replicate_set(r: &ResolvedRun) {
    let RunKind::ReplicateSet(m) = &r.run.kind else { unreachable!() };
    show_header(r);
    println!("{}", "umbrella".bright_black());
    println!("  {} of {}", m.child_kind, m.dim_name);
    println!("{}", "children".bright_black());
    for k in &m.keys {
        let child_dir = r.abs_path.join("replicates").join(k);
        let exists_marker = if child_dir.join("run.json").exists() { "✓" } else { "·" };
        println!("  {} {}", exists_marker, k);
    }
    let summary = r.abs_path.join("summary.tsv");
    if summary.exists() {
        let bytes = std::fs::metadata(&summary).map(|m| m.len()).unwrap_or(0);
        println!("{}", "summary".bright_black());
        println!("  {} ({} bytes)", summary.display(), bytes);
    } else {
        println!("{}", "summary".bright_black());
        println!("  {} (not yet written)", "summary.tsv".dimmed());
    }
    println!("{}", "hashes".bright_black());
    println!("  parent {}", r.run.hash.dimmed());
    println!("  inner  {}", m.inner_content_hash.dimmed());
    show_footer(r);
}

fn show_survey(r: &ResolvedRun) {
    let RunKind::Survey(m) = &r.run.kind else { unreachable!() };
    show_header(r);
    println!("{}", "model".bright_black()); println!("  {}", m.model);
    println!("{}", "estimated".bright_black());
    println!("  {}", m.estimated.join(", "));
    println!("{}", "bounds".bright_black());
    let mut bounds: Vec<(&String, &(f64, f64))> = m.bounds.iter().collect();
    bounds.sort_by(|a, b| a.0.cmp(b.0));
    for (name, (lo, hi)) in &bounds {
        println!("  {}: [{}, {}]", name, lo, hi);
    }
    if !m.fixed.is_empty() {
        let mut fx: Vec<(&String, &f64)> = m.fixed.iter().collect();
        fx.sort_by(|a, b| a.0.cmp(b.0));
        let items: Vec<String> = fx.iter().map(|(k, v)| format!("{}={}", k, v)).collect();
        println!("{}", "fixed".bright_black()); println!("  {}", items.join(", "));
    }
    if let Some(ref s) = m.scenario {
        println!("{}", "scenario".bright_black()); println!("  {}", s);
    }
    println!("{}", "n_points".bright_black()); println!("  {}", m.n_points);
    println!("{}", "eval".bright_black());
    match m.eval_method {
        crate::run_meta::SurveyEvalMethod::Pfilter =>
            println!("  pfilter ({} particles × {} replicates)",
                m.eval_particles, m.eval_replicates),
        crate::run_meta::SurveyEvalMethod::Simulate =>
            println!("  simulate (single trajectory per point)"),
        // SurveyMeta only stores resolved methods — `Auto` is
        // resolved in `cmd_survey` before persistence.
        crate::run_meta::SurveyEvalMethod::Auto =>
            println!("  auto (unresolved — bug; SurveyMeta should never store Auto)"),
    }
    println!("{}", "seed".bright_black()); println!("  {}", m.seed);
    let landscape = r.abs_path.join("landscape.tsv");
    if landscape.exists() {
        let bytes = std::fs::metadata(&landscape).map(|m| m.len()).unwrap_or(0);
        println!("{}", "landscape".bright_black());
        println!("  landscape.tsv ({} bytes)", bytes);
    }
    let summary = r.abs_path.join("summary.json");
    if summary.exists() {
        // Inline the top-loglik / SE-quartile fields if available.
        if let Ok(s) = std::fs::read_to_string(&summary) {
            if let Ok(j) = serde_json::from_str::<serde_json::Value>(&s) {
                if let Some(top) = j.get("top_loglik").and_then(|v| v.as_f64()) {
                    println!("{}", "top loglik".bright_black());
                    println!("  {:.2}", top);
                }
                if let Some(se_q) = j.get("loglik_se_quartiles") {
                    println!("{}", "loglik_se quartiles".bright_black());
                    println!("  {}", se_q);
                }
            }
        }
    }
    let html = r.abs_path.join("landscape.html");
    if html.exists() {
        let bytes = std::fs::metadata(&html).map(|m| m.len()).unwrap_or(0);
        println!("{}", "rendered".bright_black());
        println!("  landscape.html ({} bytes)", bytes);
    }
    println!("{}", "hashes".bright_black());
    println!("  survey {}", r.run.hash.dimmed());
    println!("  model  {}", m.model_hash.dimmed());
    show_footer(r);
}

// ── cmd_cat ──────────────────────────────────────────────────────────────────

pub fn cmd_cat(a: &crate::args::CatArgs) {
    let root = a.root.to_string_lossy();
    let resolved = resolve_any(&root, &a.target).unwrap_or_else(|e| {
        eprintln!("error: {}", e); std::process::exit(1);
    });

    use std::io::Write as _;

    // New-format sim: emit traj.tsv (or an obs stream) from the leaf dir.
    let resolved = match resolved {
        Resolved::Sim { leaf, rel_path, .. } => {
            let bytes = if let Some(ref stream) = a.stream {
                let path = find_obs_stream(&leaf.dir, stream).unwrap_or_else(|| {
                    eprintln!("error: no observation stream '{}' in {}", stream, rel_path);
                    std::process::exit(1);
                });
                std::fs::read(&path).unwrap_or_else(|e| {
                    eprintln!("error reading {}: {}", path.display(), e); std::process::exit(1);
                })
            } else {
                std::fs::read(leaf.dir.join("traj.tsv")).unwrap_or_else(|e| {
                    eprintln!("error reading traj.tsv: {}", e); std::process::exit(1);
                })
            };
            let _ = std::io::stdout().write_all(&bytes);
            return;
        }
        Resolved::Legacy(r) => r,
    };

    match &resolved.run.kind {
        // Legacy sims no longer exist (sims are RunRecord), but the match
        // stays exhaustive; a path-form cat of an old sim run.json reads here.
        RunKind::Simulate(_) => {
            let bytes = if let Some(ref stream) = a.stream {
                let path = find_obs_stream(&resolved.abs_path, stream).unwrap_or_else(|| {
                    eprintln!("error: no observation stream '{}' in {}", stream, resolved.rel_path);
                    std::process::exit(1);
                });
                std::fs::read(&path).unwrap_or_else(|e| {
                    eprintln!("error reading {}: {}", path.display(), e); std::process::exit(1);
                })
            } else {
                std::fs::read(resolved.abs_path.join("traj.tsv")).unwrap_or_else(|e| {
                    eprintln!("error reading traj.tsv: {}", e); std::process::exit(1);
                })
            };
            let _ = std::io::stdout().write_all(&bytes);
        }
        RunKind::ReplicateSet(_) => {
            let summary = resolved.abs_path.join("summary.tsv");
            if !summary.exists() {
                eprintln!("error: 'camdl cat' on a replicate-set umbrella expects \
                    summary.tsv, which has not been written yet for {}.",
                    resolved.rel_path);
                std::process::exit(1);
            }
            let bytes = std::fs::read(&summary).unwrap_or_else(|e| {
                eprintln!("error reading {}: {}", summary.display(), e);
                std::process::exit(1);
            });
            let _ = std::io::stdout().write_all(&bytes);
        }
        RunKind::Profile(_) => {
            let profile_tsv = resolved.abs_path.join("profile.tsv");
            if !profile_tsv.exists() {
                eprintln!("error: 'camdl cat' on a profile leaf expects \
                    profile.tsv, which has not been written yet for {}.",
                    resolved.rel_path);
                std::process::exit(1);
            }
            let bytes = std::fs::read(&profile_tsv).unwrap_or_else(|e| {
                eprintln!("error reading {}: {}", profile_tsv.display(), e);
                std::process::exit(1);
            });
            let _ = std::io::stdout().write_all(&bytes);
        }
        RunKind::Fit(_) => {
            eprintln!("error: 'camdl cat' on a fit has no single-file target.\n  \
                       {} is a fit directory. For stage output, pass the stage\n  \
                       path directly, e.g. `camdl cat {}/real/fit_<seed>/<stage>/mle_params.toml`.",
                      resolved.rel_path, resolved.rel_path);
            std::process::exit(1);
        }
        RunKind::FitStage(_) => {
            eprintln!("error: 'camdl cat' on a fit-stage has no canonical \
                       single-file target. {} is a stage directory; pass a \
                       specific file path (mle_params.toml, draws.tsv, …) \
                       directly.",
                      resolved.rel_path);
            std::process::exit(1);
        }
        RunKind::Survey(_) => {
            let landscape = resolved.abs_path.join("landscape.tsv");
            if !landscape.exists() {
                eprintln!("error: 'camdl cat' on a survey expects \
                    landscape.tsv, which has not been written yet for {}.",
                    resolved.rel_path);
                std::process::exit(1);
            }
            let bytes = std::fs::read(&landscape).unwrap_or_else(|e| {
                eprintln!("error reading {}: {}", landscape.display(), e);
                std::process::exit(1);
            });
            let _ = std::io::stdout().write_all(&bytes);
        }
    }
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
    run: Run,
    meta: crate::run_meta::FitMeta,
    rel_path: String,
    created: SystemTime,
}

// ── Profile listings ─────────────────────────────────────────────────────────

/// A discovered profile run, single- or multi-seed. Profiles live at
/// `<root>/profiles/<stem>-<hash[:8]>/` with a `run.json` of kind
/// `Profile` (single-seed) or `ReplicateSet` (multi-seed umbrella).
/// Both shapes carry the same display fields needed by `camdl list`.
#[derive(Debug, Clone)]
struct ProfileEntry {
    run: Run,
    rel_path: String,
    created: SystemTime,
    /// Display-only model path. From ProfileMeta for single-seed; from
    /// the first child's run.json for replicate-set umbrellas.
    model: String,
    /// Comma-separated focal param names (e.g. "beta,gamma").
    focal: String,
    /// Grid shape (e.g. "11×9 starts=4"). For replicate-set umbrellas
    /// the grid is shared across children.
    shape: String,
    /// Number of seed replicates. 1 for single-seed; N for multi-seed.
    n_seeds: usize,
}

/// Walk `<root>/profiles/` one level deep. Each immediate child is a
/// profile-umbrella directory (`<stem>-<hash[:8]>/`) with a `run.json`
/// of kind `ReplicateSet { child_kind: "profile" }`. Single-seed
/// profiles are the trivial N=1 case of the same shape — there is no
/// longer a `RunKind::Profile`-at-top-level path. Display fields
/// (model/focal/shape) are read from the first child's run.json.
fn discover_profiles(root: &str) -> Result<Vec<ProfileEntry>, String> {
    let profiles_root = Path::new(root).join("profiles");
    if !profiles_root.exists() { return Ok(Vec::new()); }
    let cwd = std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
    let entries = std::fs::read_dir(&profiles_root)
        .map_err(|e| format!("cannot read {}: {}", profiles_root.display(), e))?;
    let mut out = Vec::new();
    for entry in entries.flatten() {
        let dir = entry.path();
        if !dir.is_dir() { continue; }
        let Some((run, created, rel_path)) = load_run_common(&dir, &cwd) else { continue; };
        let RunKind::ReplicateSet(m) = &run.kind else { continue };
        if m.child_kind != "profile" { continue }
        let child_dir = dir.join("replicates")
            .join(m.keys.first().cloned().unwrap_or_default());
        let (model, focal, shape) = std::fs::read_to_string(child_dir.join("run.json"))
            .ok()
            .and_then(|s| serde_json::from_str::<Run>(&s).ok())
            .and_then(|child| match child.kind {
                RunKind::Profile(cm) => Some((
                    cm.model,
                    cm.focal_params.join(","),
                    format_grid_shape(&cm.grid, cm.n_starts),
                )),
                _ => None,
            })
            .unwrap_or_else(|| (
                "?".to_string(),
                "?".to_string(),
                "?".to_string(),
            ));
        let n_seeds = m.keys.len();
        out.push(ProfileEntry {
            model, focal, shape, n_seeds,
            run, rel_path, created,
        });
    }
    Ok(out)
}

/// Format a profile grid shape for the listing column. e.g.
/// 11×9 grid with 4 starts → "11×9 starts=4".
fn format_grid_shape(
    grid: &[crate::run_meta::GridAxis],
    n_starts: usize,
) -> String {
    if grid.is_empty() {
        return format!("(empty) starts={}", n_starts);
    }
    let dims: Vec<String> = grid.iter().map(|g| g.values.len().to_string()).collect();
    format!("{} starts={}", dims.join("×"), n_starts)
}

// ── Survey listings ──────────────────────────────────────────────────────────

/// One discovered survey run. Surveys live at
/// `<root>/surveys/<stem>-<hash[:8]>/` with a `run.json` of kind
/// `Survey(SurveyMeta)`. Display-only fields surfaced in `camdl list`.
#[derive(Debug, Clone)]
struct SurveyEntry {
    run: Run,
    rel_path: String,
    created: SystemTime,
    /// Display model path (from `SurveyMeta.model`).
    model: String,
    /// Comma-separated estimated parameter names.
    estimated: String,
    /// "pfilter Px×Rk" or "simulate".
    eval: String,
    /// Number of LHS points.
    n_points: usize,
    /// Best loglik in `landscape.tsv`. `None` when the artifact is
    /// missing (interrupted run).
    top_loglik: Option<f64>,
}

/// Walk `<root>/surveys/` one level deep. Each child dir is a
/// survey-run directory.
fn discover_surveys(root: &str) -> Result<Vec<SurveyEntry>, String> {
    let surveys_root = Path::new(root).join("surveys");
    if !surveys_root.exists() { return Ok(Vec::new()); }
    let cwd = std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
    let entries = std::fs::read_dir(&surveys_root)
        .map_err(|e| format!("cannot read {}: {}", surveys_root.display(), e))?;
    let mut out = Vec::new();
    for entry in entries.flatten() {
        let dir = entry.path();
        if !dir.is_dir() { continue; }
        let Some((run, created, rel_path)) = load_run_common(&dir, &cwd) else { continue; };
        let RunKind::Survey(m) = &run.kind else { continue };
        let eval = match m.eval_method {
            crate::run_meta::SurveyEvalMethod::Pfilter =>
                format!("pfilter {}p×{}r", m.eval_particles, m.eval_replicates),
            crate::run_meta::SurveyEvalMethod::Simulate => "simulate".to_string(),
            crate::run_meta::SurveyEvalMethod::Auto => "auto".to_string(),
        };
        // Read top loglik from summary.json when present.
        let top_loglik = std::fs::read_to_string(dir.join("summary.json"))
            .ok()
            .and_then(|s| serde_json::from_str::<serde_json::Value>(&s).ok())
            .and_then(|j| j.get("top_loglik").and_then(|v| v.as_f64()));
        out.push(SurveyEntry {
            model: m.model.clone(),
            estimated: m.estimated.join(","),
            eval,
            n_points: m.n_points,
            top_loglik,
            run, rel_path, created,
        });
    }
    Ok(out)
}

fn print_surveys_table(surveys: &[SurveyEntry], now: SystemTime) {
    let mut t = comfy_table::Table::new();
    t.set_content_arrangement(comfy_table::ContentArrangement::Dynamic);
    t.set_header(vec!["model", "estimate", "n_points", "eval", "top_loglik", "age", "path"]);
    for s in surveys {
        let age = fmt_relative_time(s.created, now);
        let ll = s.top_loglik
            .map(|x| format!("{:.2}", x))
            .unwrap_or_else(|| "—".into());
        t.add_row(vec![
            s.model.clone(),
            s.estimated.clone(),
            s.n_points.to_string(),
            s.eval.clone(),
            ll,
            age,
            s.rel_path.clone(),
        ]);
    }
    println!("{t}");
}

fn print_surveys_json(surveys: &[SurveyEntry]) {
    let runs: Vec<&Run> = surveys.iter().map(|s| &s.run).collect();
    match serde_json::to_string_pretty(&runs) {
        Ok(s) => println!("{}", s),
        Err(e) => eprintln!("json error: {}", e),
    }
}

/// Walk `root/fits/` one level deep — each immediate child is a fit
/// directory (`<stem>-<hash[:8]>/`). Stage-level run.json records live
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
            // `walk_fits_root` already parsed run.json; reuse its
            // `run` rather than re-reading the file. `created` and
            // `rel_path` are display-only and computed from the
            // already-parsed `run.created_at` plus the dir path.
            let created = parse_iso8601(&e.run.created_at)
                .unwrap_or_else(|| std::fs::metadata(&e.fit_dir)
                    .and_then(|m| m.modified())
                    .unwrap_or(SystemTime::UNIX_EPOCH));
            let rel_path = pathdiff_str(&e.fit_dir, &cwd);
            FitEntry { run: e.run, meta: e.fit_meta, rel_path, created }
        })
        .collect())
}

/// One resolved run, kind-agnostic. Kind-specific data lives inside
/// `run.kind` (a `RunKind` tagged union); renderers dispatch on the
/// variant rather than carrying a parallel enum here. This single
/// shape applies to every `RunKind` — sim, fit, fit-stage, profile,
/// replicate-set — so `camdl show` and `camdl cat` can route
/// uniformly.
#[derive(Debug, Clone)]
struct ResolvedRun {
    run: Run,
    abs_path: PathBuf,
    rel_path: String,
    created: SystemTime,
}

/// A resolved run: a new-format sim (`RunRecord`) or a legacy kind (`Run`).
/// The transitional reader resolves across both during M2→M3.
#[derive(Debug)]
enum Resolved {
    Sim { leaf: cas_read::Leaf, rel_path: String, created: SystemTime },
    Legacy(ResolvedRun),
}

/// Resolve a user-supplied key to a single run, spanning both the new-format
/// `sims/` (matched on `run_id` hex prefix) and the legacy fit/profile/survey
/// subtrees (matched on `Run.hash` prefix). Accepts either a path to a
/// `run.json`-containing directory (new or legacy format), or a hash prefix
/// where `{prefix}/{scenario}[/{seed_N}]` narrows sims further. An ambiguous
/// prefix errors, listing all candidates with their kinds.
fn resolve_any(root: &str, key: &str) -> Result<Resolved, String> {
    let cwd = std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."));

    // Path form: read run.json directly — try the new RunRecord first, then
    // fall back to a legacy Run.
    let as_path = Path::new(key);
    if as_path.is_dir() && as_path.join("run.json").exists() {
        if let Ok(bytes) = std::fs::read(as_path.join("run.json")) {
            if let Ok(rec) = serde_json::from_slice::<runid::RunRecord>(&bytes) {
                let leaf = cas_read::Leaf { dir: as_path.to_path_buf(), record: rec };
                let created = leaf_created(&leaf);
                return Ok(Resolved::Sim { leaf, rel_path: pathdiff_str(as_path, &cwd), created });
            }
        }
        let (run, created, rel_path) = load_run_common(as_path, &cwd)
            .ok_or_else(|| format!("could not read run.json at {}", as_path.display()))?;
        return Ok(Resolved::Legacy(ResolvedRun {
            run, rel_path, created, abs_path: as_path.to_path_buf(),
        }));
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

    // Legacy kinds: match Run.hash prefix under fits/profiles/surveys.
    let mut legacy_matches: Vec<ResolvedRun> = Vec::new();
    for top in ["fits", "profiles", "surveys"] {
        let subroot = Path::new(root).join(top);
        if !subroot.exists() { continue; }
        for dir in walkdir_all(&subroot) {
            if !dir.join("run.json").exists() { continue; }
            let Some((run, created, rel_path)) = load_run_common(&dir, &cwd) else { continue; };
            if !run.hash.starts_with(hash_prefix) { continue; }
            legacy_matches.push(ResolvedRun { run, rel_path, created, abs_path: dir });
        }
    }

    match sim_matches.len() + legacy_matches.len() {
        0 => Err(format!("no run matches '{}' in {}", key, root)),
        1 => {
            if let Some((leaf, rel_path, created)) = sim_matches.into_iter().next() {
                Ok(Resolved::Sim { leaf, rel_path, created })
            } else {
                Ok(Resolved::Legacy(legacy_matches.into_iter().next().unwrap()))
            }
        }
        n => {
            let mut msg = format!("'{}' is ambiguous, matches {} entries:\n", key, n);
            for (_, rel, _) in &sim_matches {
                msg.push_str(&format!("  {:<14} {}\n", "sim", rel));
            }
            for r in &legacy_matches {
                msg.push_str(&format!("  {:<14} {}\n", kind_label(&r.run.kind), r.rel_path));
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

/// Short tag for the disambiguation listing (`camdl show <ambiguous>`)
/// — same vocabulary as the `kind` discriminator in run.json.
fn kind_label(kind: &RunKind) -> &'static str {
    match kind {
        RunKind::Simulate(_)     => "sim",
        RunKind::Fit(_)          => "fit",
        RunKind::FitStage(_)     => "fit-stage",
        RunKind::Profile(_)      => "profile",
        RunKind::ReplicateSet(_) => "replicate-set",
        RunKind::Survey(_)       => "survey",
    }
}

/// Find the fit-stage directory whose `run.json` has `Run.hash`
/// starting with `hash_prefix`. Walks every
/// `<root>/fits/**/run.json` file — stage-level (FitStage kind)
/// only; the top-level `Run::Fit` at the fit root is skipped.
///
/// Returns `Ok(path)` for exactly one match, `Err` on zero or
/// multiple matches (with the candidates enumerated in the
/// multiple-match error). Used by `--starts-from <hash>` to let
/// users reference a stage by git-style short hash without
/// knowing the directory layout.
pub fn resolve_stage_by_hash(root: &str, hash_prefix: &str)
    -> Result<std::path::PathBuf, String>
{
    let fits = std::path::Path::new(root).join("fits");
    if !fits.exists() {
        return Err(format!("no fits/ tree under {}", root));
    }
    let mut matches = Vec::new();
    for entry in walkdir_all(&fits) {
        let run_json = entry.join("run.json");
        if !run_json.is_file() { continue; }
        let Ok(run) = Run::read(&entry) else { continue; };
        // We only want FitStage runs, not the top-level Fit run.
        if !matches!(run.kind, RunKind::FitStage(_)) { continue; }
        if run.hash.starts_with(hash_prefix) {
            matches.push(entry.clone());
        }
    }
    match matches.len() {
        0 => Err(format!("no fit stage matching hash prefix '{}' under {}",
            hash_prefix, root)),
        1 => Ok(matches.into_iter().next().unwrap()),
        n => {
            let mut msg = format!(
                "hash prefix '{}' is ambiguous, matches {} stages:\n",
                hash_prefix, n);
            for p in &matches {
                msg.push_str(&format!("  {}\n", p.display()));
            }
            msg.push_str("refine by passing a longer hash prefix");
            Err(msg)
        }
    }
}

/// Walk a directory tree returning every directory encountered. Depth-
/// unbounded; used by `resolve_stage_by_hash`. Dedicated because the
/// walkdir crate isn't a direct dep of this module and we only need
/// the simplest possible recursion.
fn walkdir_all(root: &std::path::Path) -> Vec<std::path::PathBuf> {
    let mut out = Vec::new();
    let mut stack = vec![root.to_path_buf()];
    while let Some(dir) = stack.pop() {
        if let Ok(entries) = std::fs::read_dir(&dir) {
            for e in entries.flatten() {
                let p = e.path();
                if p.is_dir() {
                    out.push(p.clone());
                    stack.push(p);
                }
            }
        }
    }
    out
}

// ── Output formatting ────────────────────────────────────────────────────────

fn print_sim_table(rows: &[SimRow], now: SystemTime) {
    use comfy_table::{Table, Cell, ContentArrangement, presets::NOTHING};

    if rows.is_empty() {
        eprintln!("{}", "(no cached runs)".dimmed());
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

    println!("{table}");
}

/// Compact model identifier for the list's MODEL column. Full absolute
/// paths (`/Users/vsb/projects/work/camdl/ocaml/golden/sir_basic.ir.json`)
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

fn print_sim_json(rows: &[SimRow]) {
    for r in rows {
        let json = serde_json::to_string(&r.leaf.record).unwrap_or_default();
        println!("{}", json);
    }
}

fn print_fits_json(fits: &[FitEntry]) {
    for f in fits {
        let json = serde_json::to_string(&f.run).unwrap_or_default();
        println!("{}", json);
    }
}

fn print_fits_table(fits: &[FitEntry], now: SystemTime) {
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
            Cell::new("ESTIMATE").add_attribute(comfy_table::Attribute::Bold),
            Cell::new("STAGES").add_attribute(comfy_table::Attribute::Bold),
            Cell::new("PATH").add_attribute(comfy_table::Attribute::Bold),
        ]);
    let mut unlabelled = 0usize;
    for f in fits {
        let rel_time = fmt_relative_time(f.created, now);
        let model    = model_display_name(&f.meta.model);
        let estimate = {
            let joined = f.meta.estimated.join(",");
            if joined.chars().count() > 30 {
                let mut s: String = joined.chars().take(29).collect(); s.push('…'); s
            } else { joined }
        };
        let stages = f.meta.stages_declared.join(",");
        if f.run.label.is_none() { unlabelled += 1; }
        let hash_short = short_hash_cell(&f.run.hash);
        let label_cell = label_cell(&f.run.label);
        table.add_row(vec![
            Cell::new(rel_time).fg(comfy_table::Color::Yellow),
            hash_short,
            label_cell,
            Cell::new(model),
            Cell::new(estimate).add_attribute(comfy_table::Attribute::Dim),
            Cell::new(stages).fg(comfy_table::Color::Green),
            Cell::new(&f.rel_path).fg(comfy_table::Color::Cyan),
        ]);
    }
    println!("{table}");
    crate::fit::fit_table::emit_unlabelled_warning(unlabelled);
}

fn print_profiles_json(profiles: &[ProfileEntry]) {
    for p in profiles {
        let json = serde_json::to_string(&p.run).unwrap_or_default();
        println!("{}", json);
    }
}

fn print_profiles_table(profiles: &[ProfileEntry], now: SystemTime) {
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
        let hash_short = short_hash_cell(&p.run.hash);
        let label_cell = label_cell(&p.run.label);
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
    println!("{table}");
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
    let secs = days_from_civil(year, month, day) * 86400
        + (hour * 3600 + minute * 60 + second) as i64;
    if secs < 0 { return None; }
    Some(SystemTime::UNIX_EPOCH + std::time::Duration::from_secs(secs as u64))
}

/// Howard Hinnant's days_from_civil (inverse of the one in cas.rs).
fn days_from_civil(y: i32, m: u32, d: u32) -> i64 {
    let y = if m <= 2 { y - 1 } else { y } as i64;
    let era = if y >= 0 { y } else { y - 399 } / 400;
    let yoe = (y - era * 400) as u32;
    let doy = (153 * (if m > 2 { m - 3 } else { m + 9 }) + 2) / 5 + d - 1;
    let doe = yoe * 365 + yoe/4 - yoe/100 + doy;
    era * 146097 + doe as i64 - 719468
}

/// Produce a path relative to `base` (usually CWD), falling back to the
/// absolute string if the strip fails.
fn pathdiff_str(path: &Path, base: &Path) -> String {
    match path.strip_prefix(base) {
        Ok(rel) => rel.to_string_lossy().into_owned(),
        Err(_)  => path.to_string_lossy().into_owned(),
    }
}

// ── Tests ────────────────────────────────────────────────────────────────────

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
            model_display_name("/Users/vsb/projects/work/camdl/ocaml/golden/sir_basic.ir.json"),
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
            artifacts: Default::default(),
            children: Default::default(),
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
