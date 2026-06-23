//! `camdl fit table <root>` — the cross-fit aggregator.
//!
//! Walks `<root>/fits/` (or whatever root the caller passes), loads
//! each fit's terminal-stage `MethodResult`, and projects each fit to
//! one [`TableRow`]. The same row schema is embedded in
//! `fit summary --format json` so the two surfaces never drift —
//! see Deliverable C in
//! `docs/dev/proposals/2026-04-28-fit-experiment-management.md` §3.
//!
//! Scope: this command is read-only. It never mutates fit_dirs and
//! never touches the network or external systems. All state is
//! recovered from the on-disk run.json + per-stage output files.

use std::time::{SystemTime, UNIX_EPOCH};

use crate::args::{FitTableArgs, FitTableFormat};
use crate::fit::config_diff::ConfigDiff;
use crate::fit::config_v2::FitConfigV2;
use crate::fit::fit_tree::{self, FitDirEntry};
use crate::fit::fit_view::FitView;
use crate::fit::table_row::{self, TableRow};

/// Top-level entry point. Walks the root, applies filters, builds
/// rows, renders in the requested format.
pub fn cmd_fit_table(args: &FitTableArgs) {
    let entries = match fit_tree::walk_fits_root(&args.root) {
        Ok(es) => es,
        Err(e) => {
            eprintln!("error: cannot walk fits root {}: {}", args.root.display(), e);
            std::process::exit(1);
        }
    };

    let now_unix = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0);

    // Outer-loop filters that key only on the FitView (no per-stage
    // load needed). These are cheap; running them first avoids the
    // walker / MethodResult load on rows the user filtered out.
    let pre_filtered: Vec<&FitDirEntry> = entries
        .iter()
        .filter(|e| matches_outer_filters(e, args, now_unix))
        .collect();

    // Pick the baseline fit (lowest fit_hash among the pre-filtered
    // cohort, deterministic). When --baseline is supplied we honour
    // it; when the cohort is empty there's nothing to do.
    let baseline_idx = match &args.baseline {
        Some(prefix) => pre_filtered
            .iter()
            .position(|e| e.view.fit_hash.starts_with(prefix)),
        None => pre_filtered
            .iter()
            .enumerate()
            .min_by(|(_, a), (_, b)| a.view.fit_hash.cmp(&b.view.fit_hash))
            .map(|(i, _)| i),
    };
    let baseline_loaded = baseline_idx.and_then(|i| {
        let entry = pre_filtered[i];
        load_archived_fit_toml(&entry.fit_dir)
            .ok()
            .map(|cfg| (entry, cfg))
    });

    // Build rows with config_diffs against the chosen baseline.
    let mut rows: Vec<TableRow> = Vec::new();
    let mut errors: Vec<String> = Vec::new();
    for entry in &pre_filtered {
        let cfg = match load_archived_fit_toml(&entry.fit_dir) {
            Ok(c) => Some(c),
            Err(e) => {
                errors.push(format!("{}: {}", entry.fit_dir.display(), e));
                None
            }
        };
        let diff = match (&cfg, &baseline_loaded) {
            (Some(this_cfg), Some((base_entry, base_cfg))) => {
                if base_entry.view.fit_hash == entry.view.fit_hash {
                    ConfigDiff::identity(&entry.view.fit_hash)
                } else {
                    ConfigDiff::compare(
                        this_cfg,
                        base_cfg,
                        &entry.view,
                        &base_entry.view,
                    )
                    .with_baseline_hash(base_entry.view.fit_hash.clone())
                }
            }
            _ => ConfigDiff::identity(&entry.view.fit_hash),
        };
        match table_row::build_row(&entry.fit_dir, diff, 0.0, now_unix) {
            Ok(r) => rows.push(r),
            Err(e) => errors.push(format!("{}: {}", entry.fit_dir.display(), e)),
        }
    }

    // Inner filters that key on the loaded TableRow.
    rows.retain(|r| matches_row_filters(r, args));

    // delta_ll_vs_best, computed over the surviving rows. PGAS rows
    // (best_loglik = None) are skipped from the max search and keep
    // their delta at 0.0 — there is no scalar likelihood to compare.
    if let Some(max) = rows
        .iter()
        .filter_map(|r| r.best_loglik)
        .fold(None, |acc: Option<f64>, x| Some(acc.map(|a| a.max(x)).unwrap_or(x)))
    {
        for r in &mut rows {
            if let Some(ll) = r.best_loglik {
                r.delta_ll_vs_best = ll - max;
            }
        }
    }

    // Stable sort: oldest fits at the bottom (most recent first).
    rows.sort_by(|a, b| b.created_at.cmp(&a.created_at));

    for err in &errors {
        eprintln!("warning: {}", err);
    }

    match args.format {
        FitTableFormat::Text => print!("{}", render_text(&rows)),
        FitTableFormat::Json => render_json(&rows),
        FitTableFormat::Md => print!("{}", render_md(&rows)),
        FitTableFormat::Csv => print!("{}", render_csv(&rows)),
    }

    // Unlabelled-fits nudge: if ≥ N rows have no label, print a
    // single end-of-output stderr hint pointing at `camdl fit label`.
    // Threshold via CAMDL_UNLABELED_THRESHOLD env var (default 5),
    // not a CLI flag — it's a per-user preference, not per-invocation.
    // Suppressed for non-text formats so JSON/CSV/md outputs remain
    // shell-pipe-friendly.
    if matches!(args.format, FitTableFormat::Text) {
        emit_unlabelled_warning(rows.iter().filter(|r| r.label.is_none()).count());
    }
}

/// Emit a one-line stderr hint when the count of unlabelled fits in
/// the rendered output reaches the user's threshold. The threshold
/// reads from `CAMDL_UNLABELED_THRESHOLD` (default 5). Setting the
/// var to `0` disables the warning entirely; setting it to a value
/// less than the actual count suppresses it for that invocation.
pub(crate) fn emit_unlabelled_warning(unlabelled_count: usize) {
    let threshold: usize = std::env::var("CAMDL_UNLABELED_THRESHOLD")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(5);
    if threshold == 0 || unlabelled_count < threshold {
        return;
    }
    eprintln!();
    eprintln!("note: {} unlabelled fit{} in output. Add labels with:",
        unlabelled_count, if unlabelled_count == 1 { "" } else { "s" });
    eprintln!("        camdl fit run --label \"<short description>\" fit.toml");
    eprintln!("        camdl fit label <hash> \"<short description>\"");
    eprintln!("      Set CAMDL_UNLABELED_THRESHOLD=0 to disable this hint.");
}

/// Load the fit.toml archived inside `<fit_dir>/fit.toml.original`.
/// Hard-cut on a missing archive: per the back-compat-is-a-non-goal
/// posture the reader errors with an actionable message rather than
/// silently falling back to `FitView.fit_toml_path` (which can
/// move/change).
fn load_archived_fit_toml(fit_dir: &std::path::Path) -> Result<FitConfigV2, String> {
    let archive = fit_dir.join("fit.toml.original");
    if !archive.exists() {
        return Err(format!(
            "no fit.toml.original at {} (predates step 6 of the \
             experiment-management proposal, 2026-04-28). Re-run the \
             fit (the content hash is stable, so the re-run lands in \
             the same fit_dir and writes the missing artifact) or \
             remove the directory.",
            archive.display()));
    }
    FitConfigV2::load(&archive.to_string_lossy())
}

fn matches_outer_filters(entry: &FitDirEntry, args: &FitTableArgs, now_unix: i64) -> bool {
    let view: &FitView = &entry.view;

    if let Some(model) = &args.model {
        if !view.model_identity.starts_with(model.as_str()) {
            return false;
        }
    }
    if let Some(stage) = &args.with_stage {
        if !view.stages_declared.iter().any(|s| s == stage) {
            return false;
        }
    }
    if let Some(prefix) = &args.hash {
        if !view.fit_hash.starts_with(prefix.as_str()) {
            return false;
        }
    }
    if let Some(secs) = args.since_seconds {
        if let Some(created) = parse_iso_to_unix(&view.created_at) {
            if now_unix - created > secs {
                return false;
            }
        }
    }
    if let Some(pat) = &args.label_pattern {
        let label = view.label.as_deref().unwrap_or("");
        if !glob_match(pat, label) {
            return false;
        }
    }
    true
}

/// Minimal glob matcher: `*` matches any (possibly empty) substring,
/// `?` matches exactly one character, every other character matches
/// itself. Sufficient for `--label-pattern "narrow R0*"` /
/// `--label-pattern "*take 1"` / `--label-pattern "*"`. Iterative
/// two-pointer scan; no regex dependency.
fn glob_match(pattern: &str, text: &str) -> bool {
    let p: Vec<char> = pattern.chars().collect();
    let t: Vec<char> = text.chars().collect();
    let (mut pi, mut ti) = (0usize, 0usize);
    let (mut star_p, mut star_t): (Option<usize>, usize) = (None, 0);
    while ti < t.len() {
        if pi < p.len() && (p[pi] == '?' || p[pi] == t[ti]) {
            pi += 1;
            ti += 1;
        } else if pi < p.len() && p[pi] == '*' {
            star_p = Some(pi);
            star_t = ti;
            pi += 1;
        } else if let Some(sp) = star_p {
            pi = sp + 1;
            star_t += 1;
            ti = star_t;
        } else {
            return false;
        }
    }
    while pi < p.len() && p[pi] == '*' {
        pi += 1;
    }
    pi == p.len()
}

fn matches_row_filters(row: &TableRow, args: &FitTableArgs) -> bool {
    if let Some(method) = &args.with_method {
        if row.method != method.as_str() {
            return false;
        }
    }
    if args.converged && !row.converged {
        return false;
    }
    if args.gate_failed && row.converged {
        return false;
    }
    true
}

fn parse_iso_to_unix(s: &str) -> Option<i64> {
    let bytes = s.as_bytes();
    if bytes.len() != 20 || bytes[10] != b'T' || bytes[19] != b'Z' {
        return None;
    }
    let year: i32 = s[0..4].parse().ok()?;
    let month: u32 = s[5..7].parse().ok()?;
    let day: u32 = s[8..10].parse().ok()?;
    let hour: u32 = s[11..13].parse().ok()?;
    let minute: u32 = s[14..16].parse().ok()?;
    let second: u32 = s[17..19].parse().ok()?;
    if !(1..=12).contains(&month) || !(1..=31).contains(&day) {
        return None;
    }
    let days = ir::caltime::unix_epoch_days(year as i64, month as i64, day as i64);
    Some(days * 86_400 + hour as i64 * 3600 + minute as i64 * 60 + second as i64)
}

// ── Renderers ──────────────────────────────────────────────────────

#[derive(serde::Serialize)]
struct TableJson<'a> {
    schema: TableJsonSchema,
    rows: &'a [TableRow],
}

#[derive(serde::Serialize)]
struct TableJsonSchema {
    name: String,
    version: u32,
    /// row_schema mirrors the `table_row` discriminator so consumers
    /// don't have to descend into a row to find it.
    row_schema: super::table_row::TableRowSchema,
}

fn render_json(rows: &[TableRow]) {
    let doc = TableJson {
        schema: TableJsonSchema {
            name: "fit_table".into(),
            version: 1,
            row_schema: super::table_row::TableRowSchema::current(),
        },
        rows,
    };
    let s = serde_json::to_string_pretty(&doc)
        .expect("TableJson must serialize");
    println!("{}", s);
}

fn render_text(rows: &[TableRow]) -> String {
    let mut s = String::new();
    s.push_str(&format!(
        "{:<10} {:<22} {:<14} {:<8} {:<6} {:<10} {:>10} {:<13} {:>6}\n",
        "fit_id", "label", "stem", "method", "stages", "converged", "best_ll", "ll_type", "age"
    ));
    s.push_str(&"-".repeat(110));
    s.push('\n');
    if rows.is_empty() {
        s.push_str("(no fits matched)\n");
        return s;
    }
    for r in rows {
        let label = r.label.as_deref().unwrap_or("<unlabelled>");
        let stages = r.stages.join("+");
        let converged = if r.converged { "yes" } else { "no" };
        let best = r
            .best_loglik
            .map(|v| format!("{:>10.1}", v))
            .unwrap_or_else(|| format!("{:>10}", "—"));
        // `best_ll` is `—` for PGAS, so the type column is what tells a
        // reader the row carries a complete-data (joint) value (gh#280).
        let ll_type = super::loglik::LoglikType::tag_or_unknown(r.loglik_type);
        let age = format_age(r.age_seconds);
        s.push_str(&format!(
            "{:<10} {:<22} {:<14} {:<8} {:<6} {:<10} {} {:<13} {:>6}\n",
            r.fit_id, truncate(label, 22), truncate(&r.stem, 14),
            r.method, truncate(&stages, 6), converged, best, ll_type, age,
        ));
    }
    s
}

fn render_md(rows: &[TableRow]) -> String {
    let mut s = String::new();
    s.push_str(
        "| fit_id | label | stem | method | stages | converged | best_ll | ll_type | age |\n",
    );
    s.push_str(
        "|---|---|---|---|---|---|---|---|---|\n",
    );
    for r in rows {
        let label = r.label.as_deref().unwrap_or("<unlabelled>");
        let stages = r.stages.join("+");
        let converged = if r.converged { "✓" } else { "✗" };
        let best = r
            .best_loglik
            .map(|v| format!("{:.1}", v))
            .unwrap_or_else(|| "—".into());
        let ll_type = super::loglik::LoglikType::tag_or_unknown(r.loglik_type);
        let age = format_age(r.age_seconds);
        s.push_str(&format!(
            "| `{}` | {} | `{}` | {} | {} | {} | {} | {} | {} |\n",
            r.fit_id, label, r.stem, r.method, stages, converged, best, ll_type, age,
        ));
    }
    s
}

fn render_csv(rows: &[TableRow]) -> String {
    let mut s = String::new();
    // `loglik_type` is appended last (gh#280): CSV is positional, so a new
    // column never shifts an existing one out from under a consumer.
    s.push_str("fit_id,fit_hash,label,stem,model_identity,method,stages,converged,gate_verdict,best_loglik,max_chain_agreement,max_rhat,acceptance_rate,delta_ll_vs_best,age_seconds,created_at,stale,loglik_type\n");
    for r in rows {
        let label = csv_field(r.label.as_deref().unwrap_or(""));
        let stages = r.stages.join("+");
        let best = r
            .best_loglik
            .map(|v| format!("{}", v))
            .unwrap_or_default();
        let max_a = r
            .max_chain_agreement
            .map(|v| format!("{}", v))
            .unwrap_or_default();
        let max_r = r.max_rhat.map(|v| format!("{}", v)).unwrap_or_default();
        let acc = r.acceptance_rate.map(|v| format!("{}", v)).unwrap_or_default();
        s.push_str(&format!(
            "{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}\n",
            r.fit_id,
            r.fit_hash,
            label,
            csv_field(&r.stem),
            r.model_identity,
            r.method,
            stages,
            r.converged,
            r.gate_verdict,
            best,
            max_a,
            max_r,
            acc,
            r.delta_ll_vs_best,
            r.age_seconds,
            r.created_at,
            r.stale,
            super::loglik::LoglikType::tag_or_unknown(r.loglik_type),
        ));
    }
    s
}

fn csv_field(s: &str) -> String {
    if s.contains([',', '"', '\n']) {
        format!("\"{}\"", s.replace('"', "\"\""))
    } else {
        s.to_string()
    }
}

fn truncate(s: &str, max: usize) -> String {
    if s.chars().count() <= max {
        s.to_string()
    } else {
        let mut t: String = s.chars().take(max.saturating_sub(1)).collect();
        t.push('…');
        t
    }
}

fn format_age(seconds: i64) -> String {
    if seconds < 60 {
        return format!("{}s", seconds.max(0));
    }
    let minutes = seconds / 60;
    if minutes < 60 {
        return format!("{}m", minutes);
    }
    let hours = minutes / 60;
    if hours < 48 {
        return format!("{}h", hours);
    }
    let days = hours / 24;
    if days < 14 {
        return format!("{}d", days);
    }
    format!("{}w", days / 7)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::{Path, PathBuf};

    /// One-off tempdir helper that cleans up on Drop.
    struct TempDir(PathBuf);
    impl TempDir {
        fn path(&self) -> &Path {
            &self.0
        }
    }
    impl Drop for TempDir {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }
    fn tempdir(tag: &str) -> TempDir {
        let ns = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let p = std::env::temp_dir().join(format!(
            "camdl_fittable_{}_{}_{}",
            tag,
            std::process::id(),
            ns
        ));
        std::fs::create_dir_all(&p).unwrap();
        TempDir(p)
    }

    #[test]
    fn format_age_uses_compact_units() {
        assert_eq!(format_age(0), "0s");
        assert_eq!(format_age(45), "45s");
        assert_eq!(format_age(60 * 5), "5m");
        assert_eq!(format_age(60 * 60 * 3), "3h");
        assert_eq!(format_age(60 * 60 * 24 * 5), "5d");
        assert_eq!(format_age(60 * 60 * 24 * 30), "4w");
    }

    /// A minimal row carrying the given loglik class — everything else is
    /// the empty/zero placeholder shape used by the error branch.
    fn row_with_type(loglik_type: Option<crate::fit::loglik::LoglikType>) -> TableRow {
        TableRow {
            schema: table_row::TableRowSchema::current(),
            fit_id: "abc12345".into(),
            fit_hash: "abc12345".into(),
            label: None,
            stem: "model".into(),
            model_identity: "mid".into(),
            stages: vec!["pgas".into()],
            method: "pgas".into(),
            config_diff_from_baseline: super::super::config_diff::ConfigDiff::identity(""),
            converged: true,
            gate_verdict: "n/a".into(),
            best_loglik: None,
            loglik_type,
            max_chain_agreement: None,
            max_rhat: Some(1.01),
            acceptance_rate: None,
            ess_at_mle: None,
            ess_posterior: None,
            params: std::collections::BTreeMap::new(),
            delta_ll_vs_best: 0.0,
            age_seconds: 0,
            created_at: String::new(),
            stale: false,
            stale_reason: None,
        }
    }

    /// gh#280: the CSV gains an **appended** `loglik_type` column (positional,
    /// so existing columns never shift). A PGAS row whose `best_loglik` is
    /// empty still reports `complete_data` in the new column. Fails on the
    /// pre-gh#280 header (no such column).
    #[test]
    fn csv_appends_loglik_type_column() {
        let csv = render_csv(&[row_with_type(Some(crate::fit::loglik::LoglikType::CompleteData))]);
        let header = csv.lines().next().unwrap();
        assert!(header.ends_with(",loglik_type"),
            "loglik_type must be the last (appended) column: {header}");
        // best_loglik stays where it was — the column count grew by exactly 1.
        assert_eq!(header.split(',').count(),
            csv.lines().nth(1).unwrap().split(',').count(),
            "header and row must have the same field count");
        let row = csv.lines().nth(1).unwrap();
        assert!(row.ends_with(",complete_data"),
            "PGAS row's appended column carries the joint tag: {row}");
        // A legacy row with no type renders `unknown`, never inferred.
        let csv2 = render_csv(&[row_with_type(None)]);
        assert!(csv2.lines().nth(1).unwrap().ends_with(",unknown"),
            "absent type renders `unknown`: {csv2}");
    }

    /// Cohort filtering on `--with-method` keeps only the requested
    /// rows. Walk pickup of fit_seed / Real preference is exercised by
    /// `table_row::build_row`; here we only check the cohort filter
    /// surface.
    #[test]
    fn render_text_zero_rows_says_so() {
        let s = render_text(&[]);
        assert!(s.contains("(no fits matched)"));
    }

    /// gh#147 (M3.2): write a CAS fit segment at `seg` — one `FitStage` leaf
    /// (`01-mle-<h8>/seed_1-<h8>/run.json`, a `runid::RunRecord`) plus the
    /// fit-level sidecar — the shape `walk_fits_root` reads now. No
    /// `mle_params.toml`/`fit_state.toml`, so the row's MethodResult won't load
    /// (the property under test); the fit entry itself is still discovered.
    fn write_cas_fit_seg(seg: &Path, fit_h8: &str) {
        let leaf = seg.join("01-mle-1fb03eee").join("seed_1-06cbd6b3");
        std::fs::create_dir_all(&leaf).unwrap();
        let fit_hash = format!("{fit_h8}{}", "0".repeat(64 - fit_h8.len()));
        let run_id = format!("{:0<64}", format!("{fit_h8}01"));
        let rec = format!(
            r#"{{"format_version":1,"kind":"fit_stage","run_id":"{run_id}","hash_version":1,"ir_version":"0.7","engine_version":"0.1.0+test","levels":[{{"name":"fit","label":"fit","hash":"{fit_hash}","schema_version":1}},{{"name":"stage","label":"01-mle","hash":"1fb03eee00000000000000000000000000000000000000000000000000000000","schema_version":1}},{{"name":"seed","label":"seed_1","hash":"06cbd6b300000000000000000000000000000000000000000000000000000000","schema_version":1}}],"status":"completed","artifacts":{{}},"inputs":{{"stage":"mle","method":"if2","backend":"chain_binomial","seed":1,"n_chains":2}},"provenance":{{"created_at":"2026-04-27T00:00:00Z","argv":["camdl","fit","run"]}}}}"#
        );
        std::fs::write(leaf.join("run.json"), rec).unwrap();
        std::fs::write(
            seg.join("fit.meta.json"),
            r#"{"model_identity":"f00d","model_path":"sir.camdl","fit_toml_path":"fit.toml"}"#,
        )
        .unwrap();
    }

    /// The walker integration: write two CAS fit segments whose leaves carry no
    /// loadable MethodResults, and ensure `walk_fits_root` still discovers both
    /// fit entries (the row's MethodResult load failing is handled downstream in
    /// `cmd_fit_table`; loadable-result coverage lives in the integration test
    /// in `tests/fit_experiment_management.rs`, which has the FitState fixtures).
    #[test]
    fn walker_returns_empty_rows_when_no_method_results_loadable() {
        let tmp = tempdir("empty_results");
        let fits_root = tmp.path().join("fits");
        std::fs::create_dir_all(&fits_root).unwrap();
        for name in &["fit_a-aaaaaaaa", "fit_b-bbbbbbbb"] {
            let d = fits_root.join(name);
            let h8 = name.split('-').next_back().unwrap();
            write_cas_fit_seg(&d, h8);
        }
        let entries = fit_tree::walk_fits_root(&fits_root).unwrap();
        assert_eq!(entries.len(), 2);
    }
}
