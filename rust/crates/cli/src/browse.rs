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

/// A filter run on the run store, prepared for `list` and `show`.
struct PfilterRow {
    stored: store::Stored<crate::pfilter_store::Pfilter>,
    labels: crate::pfilter_store::PfilterLabels,
    /// The label of the latest invocation that asked for it, if it had one.
    label: Option<String>,
    rel_path: String,
    created: SystemTime,
}

impl PfilterRow {
    fn loglik(&self) -> Option<crate::pfilter_store::Loglik> {
        crate::camdl_store::read_stored(&self.stored, crate::pfilter_store::LOGLIK_FILE)
            .and_then(|b| crate::pfilter_store::Loglik::from_bytes(&b))
            .ok()
    }

    fn json(&self) -> serde_json::Value {
        serde_json::json!({
            "kind": "pfilter",
            "run_id": self.stored.id().key().to_hex(),
            "label": self.label,
            "labels": self.labels,
            "path": self.rel_path,
            "identity": self.stored.identity(),
            "loglik": self.loglik(),
        })
    }
}

/// The run store's root a reading command uses: `--store`, else
/// `CAMDL_STORE`, else the default.
fn store_root(flag: Option<&Path>) -> PathBuf {
    flag.map(Path::to_path_buf).unwrap_or_else(crate::camdl_store::read_root)
}

/// The old results root a reading command searches: the one it is given
/// (`--root`, the `list` positional, or `CAMDL_OUTPUT_DIR`), else
/// `./results`. When a run store is named (`--store`, or `CAMDL_STORE`) and no
/// old root is, there is none: a `./results` in the working directory is
/// never read beside a store the command was pointed at.
pub(crate) fn old_root(named: Option<&Path>, store: Option<&Path>) -> Option<PathBuf> {
    match named {
        Some(r) => Some(r.to_path_buf()),
        None if store.is_some() || crate::env::get().vars.store.is_some() => None,
        None => Some(PathBuf::from(format!("./{}", crate::run_paths::DEFAULT_OUTPUT_ROOT))),
    }
}

/// The store `target` is read from: the one a result's folder lies in, else
/// [`store_root`]'s.
fn store_root_for(flag: Option<&Path>, target: &str) -> PathBuf {
    match flag {
        Some(f) => f.to_path_buf(),
        None => crate::fit_store::store_holding(Path::new(target)).unwrap_or_else(crate::camdl_store::read_root),
    }
}

/// Every completed filter run the run store's invocation records list, once,
/// with the label of the latest invocation that asked for it.
fn discover_pfilter_rows(root: &Path) -> Vec<PfilterRow> {
    let Some(store) = crate::camdl_store::open_existing(root) else { return Vec::new() };
    let cwd = std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
    let mut rows: std::collections::BTreeMap<store::Key, PfilterRow> = std::collections::BTreeMap::new();
    for inv in store.invocations() {
        let Ok(members) = inv.members::<crate::pfilter_store::Pfilter>() else { continue };
        for (id, labels) in members {
            if let Some(row) = rows.get_mut(&id.key()) {
                if inv.label().is_some() {
                    row.label = inv.label().map(str::to_string);
                }
                continue;
            }
            let Ok(Some(stored)) = store.get(&id) else { continue };
            let created = std::fs::metadata(stored.folder()).and_then(|m| m.modified()).unwrap_or(SystemTime::UNIX_EPOCH);
            let rel_path = pathdiff_str(stored.folder(), &cwd);
            rows.insert(id.key(), PfilterRow { stored, labels, label: inv.label().map(str::to_string), rel_path, created });
        }
    }
    rows.into_values().collect()
}

/// The filter run a handle names on the run store: `None` when the handle
/// names no filter run there, so the caller looks in the old store.
fn resolve_stored_pfilter(root: &Path, key: &str) -> Option<Result<PfilterRow, String>> {
    let store = crate::camdl_store::open_existing(root)?;
    let handle: store::RunRef<crate::pfilter_store::Pfilter> = key.parse().ok()?;
    let stored = match store.resolve(&handle) {
        Ok(s) => s,
        Err(store::Error::NotFound { .. } | store::Error::WrongKind { .. } | store::Error::NoRecord { .. }) => return None,
        Err(e) => return Some(Err(e.to_string())),
    };
    let row = discover_pfilter_rows(root).into_iter().find(|r| r.stored.id() == stored.id());
    Some(row.ok_or_else(|| format!("{key} names a filter run no invocation record lists")))
}

/// The design table of the invocation a label names, when that invocation is
/// a `pfilter --sweep-file` design on the run store: `None` for any other
/// handle or invocation, so the caller resolves it as a single result.
fn stored_design_table(root: &Path, target: &str) -> Option<Result<Vec<u8>, String>> {
    let handle: store::Handle = target.parse().ok()?;
    if !matches!(handle, store::Handle::Label { .. }) {
        return None;
    }
    let store = crate::camdl_store::open_existing(root)?;
    let invocation = store.invocation(&handle).ok()?;
    let members = invocation.members::<crate::pfilter_store::Pfilter>().ok()?;
    if !members.iter().any(|(_, labels)| labels.point.is_some()) {
        return None;
    }
    let mut table = Vec::new();
    Some(
        invocation
            .render_to::<crate::pfilter_store::DesignTable>(&store, &crate::pfilter_store::DesignTableOptions { run_id: true }, &mut table)
            .map(|()| table)
            .map_err(|e| e.to_string()),
    )
}

/// A file of a stored filter run, or of one of its by-products: the trace,
/// the health table, the filtering and filtered-stream tables, the final
/// state, the saved paths or the prequential scores. A name without an
/// extension also stands for `<name>.tsv`. A by-product is found through the
/// invocation records, by the filter run its identity holds.
fn stored_pfilter_file(root: &Path, row: &PfilterRow, name: &str) -> Result<Vec<u8>, String> {
    match stored_pfilter_file_named(root, row, name) {
        Err(_) if !name.contains('.') => stored_pfilter_file_named(root, row, &format!("{name}.tsv")),
        r => r,
    }
}

fn stored_pfilter_file_named(root: &Path, row: &PfilterRow, name: &str) -> Result<Vec<u8>, String> {
    use store::ContentAddressed;
    use crate::pfilter_store as pf;
    if row.stored.result_files().contains(&name) {
        return crate::camdl_store::read_stored(&row.stored, name);
    }
    let store = crate::camdl_store::open_existing(root).ok_or("no run store")?;
    let run = row.stored.identity().content_hash();
    let mut names: Vec<String> = row.stored.result_files().iter().map(|s| s.to_string()).collect();
    for inv in store.invocations().iter().rev() {
        macro_rules! look {
            ($t:ty, $flag:literal) => {
                for (id, ()) in inv.members::<$t>().unwrap_or_default() {
                    let Ok(Some(s)) = store.get(&id) else {
                        // A by-product of this run that is stored but fails its
                        // manifest check is not served; say which, and how to
                        // make it again.
                        if let Ok(Some(e)) = store.inspect(&id) {
                            if e.identity().filter.content_hash() == run {
                                if let store::EntryState::Damaged { what } = e.state() {
                                    return Err(format!(
                                        "the filter run's {} is damaged ({what}), so it is not read; run the \
                                         pfilter again with {} and --force to store it again",
                                        <$t as store::Artifact>::KIND, $flag
                                    ));
                                }
                            }
                        }
                        continue;
                    };
                    if s.identity().filter.content_hash() != run {
                        continue;
                    }
                    if s.result_files().contains(&name) {
                        return crate::camdl_store::read_stored(&s, name);
                    }
                    names.extend(s.result_files().iter().map(|f| f.to_string()));
                }
            };
        }
        look!(pf::Trace, "--trace");
        look!(pf::PfHealth, "--pf-health");
        look!(pf::Filtering, "--save-filtering");
        look!(pf::FilteredStreams, "--save-filtered-streams");
        look!(pf::FinalState, "--save-final-state");
        look!(pf::SavedPaths, "--save-paths");
        look!(pf::Prequential, "--save-prequential");
    }
    names.sort();
    names.dedup();
    Err(format!("the filter run holds no file `{name}`; its files and its by-products' files: {}", names.join(", ")))
}

/// `camdl show` of a filter run on the run store.
fn show_stored_pfilter(out: &crate::out::Out, row: &PfilterRow) {
    let field = |name: &str, value: &dyn std::fmt::Display| {
        out.report_line(format_args!("{}", style::stdout().paint(name, Style::Dim)));
        out.report_line(format_args!("  {value}"));
    };
    field("path", &style::stdout().paint(&row.rel_path, Style::Code));
    field("kind", &"pfilter");
    if let Some(l) = &row.label {
        field("label", l);
    }
    let id = row.stored.identity();
    field("model", &row.labels.model);
    field("config", &crate::pfilter_store::config_label(id.config.particles, id.config.replicates, id.config.dt.get()));
    field("seed", &id.seed.seed);
    field("run_id", &style::stdout().paint(&row.stored.id().key().to_hex(), Style::Dim));
    if let Some(ll) = row.loglik() {
        field("loglik", &ll.loglik.unwrap_or(f64::NEG_INFINITY));
        if let Some(sd) = ll.loglik_sd {
            field("loglik_sd", &sd);
        }
        if let Some(failure) = &ll.failure {
            field("n_failed", &ll.n_failed);
            field("failure", failure);
        }
        field("n_replicates", &ll.n_replicates);
        field("n_particles", &ll.n_particles);
    }
    if !id.params.values.is_empty() {
        out.report_line(format_args!("{}", style::stdout().paint("scored point", Style::Dim)));
        for (name, v) in &id.params.values {
            out.report_line(format_args!("  {:<16} {}", name, v.get()));
        }
    }
    field("created", &fmt_relative_time(row.created, SystemTime::now()));
}

/// A fit run on the run store, prepared for `list`.
struct StoredFitRow {
    stored: store::Stored<crate::fit_store::FitRun>,
    labels: crate::fit_store::RunLabels,
    /// The model the problem was posed on, by name.
    model: String,
    /// The label of the latest invocation that asked for it, if it had one.
    label: Option<String>,
    rel_path: String,
    created: SystemTime,
}

impl StoredFitRow {
    /// The run's `fit_state.toml` best log-likelihood, if it records one.
    fn best_loglik(&self) -> Option<f64> {
        let bytes = crate::camdl_store::read_stored(&self.stored, "fit_state.toml").ok()?;
        let doc: toml::Table = String::from_utf8(bytes).ok()?.parse().ok()?;
        doc.get("best_loglik").and_then(|v| v.as_float())
    }

    fn json(&self) -> serde_json::Value {
        let id = self.stored.identity();
        serde_json::json!({
            "kind": "fit_run",
            "run_id": self.stored.id().key().to_hex(),
            "problem_id": id.chain.problem.id().key().to_hex(),
            "label": self.label,
            "method": self.labels.method,
            "seed": self.labels.seed,
            "model": self.model,
            "path": self.rel_path,
            "best_loglik": self.best_loglik(),
        })
    }
}

/// The model a stored problem was posed on, by the name the invocation that
/// stored it gave it.
fn problem_model_name(store: &store::Store, problem: &store::Id<crate::fit_store::FitProblem>) -> String {
    store
        .invocation(&store::Handle::IdPrefix(problem.key().to_hex()))
        .ok()
        .and_then(|inv| inv.members::<crate::fit_store::FitProblem>().ok())
        .and_then(|m| m.into_iter().find(|(id, _)| id == problem).map(|(_, l)| l.model))
        .unwrap_or_default()
}

/// Every completed fit run the run store's invocation records list, once,
/// with the label of the latest invocation that asked for it.
fn discover_stored_fit_rows(root: &Path) -> Vec<StoredFitRow> {
    let Some(store) = crate::camdl_store::open_existing(root) else { return Vec::new() };
    let cwd = std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
    let mut rows: std::collections::BTreeMap<store::Key, StoredFitRow> = std::collections::BTreeMap::new();
    for inv in store.invocations() {
        let Ok(members) = inv.members::<crate::fit_store::FitRun>() else { continue };
        for (id, labels) in members {
            if let Some(row) = rows.get_mut(&id.key()) {
                if inv.label().is_some() {
                    row.label = inv.label().map(str::to_string);
                }
                continue;
            }
            let Ok(Some(stored)) = store.get(&id) else { continue };
            let created = std::fs::metadata(stored.folder()).and_then(|m| m.modified()).unwrap_or(SystemTime::UNIX_EPOCH);
            let rel_path = pathdiff_str(stored.folder(), &cwd);
            let model = problem_model_name(&store, stored.identity().chain.problem.id());
            rows.insert(id.key(), StoredFitRow { stored, labels, model, label: inv.label().map(str::to_string), rel_path, created });
        }
    }
    rows.into_values().collect()
}

fn print_stored_fit_table(out: &crate::out::Out, rows: &[StoredFitRow], now: SystemTime) {
    use comfy_table::{Table, Cell, ContentArrangement, presets::NOTHING};
    let mut table = Table::new();
    table
        .load_preset(NOTHING)
        .set_content_arrangement(ContentArrangement::Dynamic)
        .set_header(vec![
            Cell::new("CREATED").add_attribute(comfy_table::Attribute::Bold),
            Cell::new("RUN_ID").add_attribute(comfy_table::Attribute::Bold),
            Cell::new("FIT_ID").add_attribute(comfy_table::Attribute::Bold),
            Cell::new("LABEL").add_attribute(comfy_table::Attribute::Bold),
            Cell::new("MODEL").add_attribute(comfy_table::Attribute::Bold),
            Cell::new("METHOD").add_attribute(comfy_table::Attribute::Bold),
            Cell::new("SEED").add_attribute(comfy_table::Attribute::Bold),
            Cell::new("BEST_LL").add_attribute(comfy_table::Attribute::Bold),
            Cell::new("PATH").add_attribute(comfy_table::Attribute::Bold),
        ]);
    for r in rows {
        let ll = r.best_loglik().map(|v| format!("{v:.2}")).unwrap_or_else(|| "—".into());
        table.add_row(vec![
            Cell::new(fmt_relative_time(r.created, now)).fg(comfy_table::Color::Yellow),
            short_hash_cell(&r.stored.id().key().to_hex()),
            short_hash_cell(&r.stored.identity().chain.problem.id().key().to_hex()),
            label_cell(&r.label),
            Cell::new(&r.model),
            Cell::new(&r.labels.method),
            Cell::new(r.labels.seed),
            Cell::new(ll).fg(comfy_table::Color::Magenta),
            Cell::new(&r.rel_path).fg(comfy_table::Color::Cyan),
        ]);
    }
    out.report_line(format_args!("{table}"));
}

/// What a handle names among a fit's results on the run store.
enum StoredFitResult {
    Run(store::Stored<crate::fit_store::FitRun>, RunExtras),
    Problem(store::Stored<crate::fit_store::FitProblem>),
    Prediction(store::Stored<crate::fit_store::Prediction>),
}

/// What the store records of a fit run beside its own files: the label of
/// the latest invocation that asked for it, and its stored predictions, each
/// with its folder relative to the run's.
struct RunExtras {
    label: Option<String>,
    predictions: Vec<(String, store::Stored<crate::fit_store::Prediction>)>,
}

impl RunExtras {
    fn of(store: &store::Store, run: &store::Stored<crate::fit_store::FitRun>) -> RunExtras {
        let mut label = None;
        let mut seen = std::collections::BTreeSet::new();
        let mut predictions = Vec::new();
        for inv in store.invocations() {
            if let Ok(members) = inv.members::<crate::fit_store::FitRun>() {
                if members.iter().any(|(id, _)| id == run.id()) && inv.label().is_some() {
                    label = inv.label().map(str::to_string);
                }
            }
            let Ok(members) = inv.members::<crate::fit_store::Prediction>() else { continue };
            for (id, _) in members {
                if !seen.insert(id.key()) {
                    continue;
                }
                let Ok(Some(p)) = store.get(&id) else { continue };
                let crate::fit_store::ParameterSourceRecord::Posterior { draws, .. } = &p.identity().source;
                if draws.id() == run.id() {
                    let rel = p.folder().strip_prefix(run.folder()).unwrap_or(p.folder()).display().to_string();
                    predictions.push((format!("{rel}/"), p));
                }
            }
        }
        predictions.sort_by(|a, b| a.0.cmp(&b.0));
        RunExtras { label, predictions }
    }

    /// The prediction file `name` (`predict/<mode>-<id>/<file>`) names.
    fn prediction_file<'a>(&'a self, name: &'a str) -> Option<(&'a store::Stored<crate::fit_store::Prediction>, &'a str)> {
        self.predictions.iter().find_map(|(dir, p)| name.strip_prefix(dir.as_str()).map(|f| (p, f)))
    }
}

impl StoredFitResult {
    fn kind(&self) -> &'static str {
        match self {
            StoredFitResult::Run(..) => "fit_run",
            StoredFitResult::Problem(_) => "fit",
            StoredFitResult::Prediction(_) => "prediction",
        }
    }

    fn folder(&self) -> &Path {
        match self {
            StoredFitResult::Run(s, _) => s.folder(),
            StoredFitResult::Problem(s) => s.folder(),
            StoredFitResult::Prediction(s) => s.folder(),
        }
    }

    fn run_id(&self) -> String {
        match self {
            StoredFitResult::Run(s, _) => s.id().key().to_hex(),
            StoredFitResult::Problem(s) => s.id().key().to_hex(),
            StoredFitResult::Prediction(s) => s.id().key().to_hex(),
        }
    }

    fn result_files(&self) -> Vec<String> {
        let names = match self {
            StoredFitResult::Run(s, _) => s.result_files(),
            StoredFitResult::Problem(s) => s.result_files(),
            StoredFitResult::Prediction(s) => s.result_files(),
        };
        names.into_iter().map(str::to_string).collect()
    }

    fn identity(&self) -> serde_json::Value {
        match self {
            StoredFitResult::Run(s, _) => serde_json::to_value(s.identity()),
            StoredFitResult::Problem(s) => serde_json::to_value(s.identity()),
            StoredFitResult::Prediction(s) => serde_json::to_value(s.identity()),
        }
        .unwrap_or(serde_json::Value::Null)
    }

    fn read(&self, name: &str) -> Result<Vec<u8>, String> {
        match self {
            StoredFitResult::Run(s, _) => crate::camdl_store::read_stored(s, name),
            StoredFitResult::Problem(s) => crate::camdl_store::read_stored(s, name),
            StoredFitResult::Prediction(s) => crate::camdl_store::read_stored(s, name),
        }
    }

    fn json(&self) -> serde_json::Value {
        let cwd = std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
        serde_json::json!({
            "kind": self.kind(),
            "run_id": self.run_id(),
            "path": pathdiff_str(self.folder(), &cwd),
            "identity": self.identity(),
            "files": self.result_files(),
        })
    }

    /// `camdl show`: what the result is, where it is, and its files.
    fn show(&self, out: &crate::out::Out) {
        let cwd = std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
        let field = |name: &str, value: &dyn std::fmt::Display| {
            out.report_line(format_args!("{}", style::stdout().paint(name, Style::Dim)));
            out.report_line(format_args!("  {value}"));
        };
        field("path", &style::stdout().paint(&pathdiff_str(self.folder(), &cwd), Style::Code));
        field("kind", &self.kind());
        field("run_id", &style::stdout().paint(&self.run_id(), Style::Dim));
        if let StoredFitResult::Run(s, extras) = self {
            if let Some(l) = &extras.label {
                field("label", l);
            }
            let id = s.identity();
            field("problem", &id.chain.problem.id().key().to_hex());
            field("method", &id.chain.method.algorithm);
            field("seed", &id.chain.seed.seed);
            field("length", &id.length);
            if let Some(base) = &id.resumed_from {
                field("resumed from", &base.id().key().to_hex());
            }
        }
        if let StoredFitResult::Prediction(s) = self {
            let crate::fit_store::ParameterSourceRecord::Posterior { draws, .. } = &s.identity().source;
            field("draws of", &draws.id().key().to_hex());
        }
        out.report_line(format_args!("{}", style::stdout().paint("files", Style::Dim)));
        for f in self.result_files() {
            out.report_line(format_args!("  {f}"));
        }
        let StoredFitResult::Run(_, extras) = self else { return };
        if extras.predictions.is_empty() {
            return;
        }
        // Each prediction's files, by the path `cat --stream` takes.
        out.report_line(format_args!("{}", style::stdout().paint("outputs", Style::Dim)));
        for (dir, p) in &extras.predictions {
            for f in p.result_files() {
                out.report_line(format_args!("  {dir}{f}"));
            }
        }
        let forecasts: Vec<(&String, Result<crate::fit::forecast_output::ForecastSummary, String>)> = extras
            .predictions
            .iter()
            .filter(|(_, p)| p.result_files().contains(&crate::fit::forecast_output::FORECAST_MANIFEST))
            .map(|(dir, p)| {
                let summary = crate::camdl_store::read_stored(p, crate::fit::forecast_output::FORECAST_MANIFEST)
                    .and_then(|b| crate::fit::forecast_output::ForecastSummary::parse(&b));
                (dir, summary)
            })
            .collect();
        if forecasts.is_empty() {
            return;
        }
        out.report_line(format_args!("{}", style::stdout().paint("forecasts", Style::Dim)));
        for (dir, summary) in forecasts {
            out.report_line(format_args!("  {dir}"));
            match summary {
                Ok(s) => {
                    for (label, value) in s.lines() {
                        out.report_line(format_args!("    {}{} {value}",
                            style::stdout().paint(label, Style::Dim), " ".repeat(10 - label.len())));
                    }
                }
                Err(e) => out.report_line(format_args!("    {}", style::stdout().paint(&e, Style::Dim))),
            }
        }
    }

    /// `camdl cat`: the named result file, else the result's summary file;
    /// a chain selection keeps the rows of the chains it names.
    fn cat(
        &self,
        stream: Option<&str>,
        selection: Option<&crate::chain_selection::ChainSelection>,
        out: &crate::out::Out,
    ) -> Result<Vec<u8>, String> {
        let default = match self {
            StoredFitResult::Run(..) => "fit_state.toml",
            StoredFitResult::Problem(_) => crate::fit_store::PROBLEM_FILE,
            StoredFitResult::Prediction(_) => "report.json",
        };
        let files = self.result_files();
        let name = stream.unwrap_or(default);
        // A run's prediction file, by the path `show` lists it under.
        if let (StoredFitResult::Run(_, extras), Some(stream)) = (self, stream) {
            if let Some((p, file)) = extras.prediction_file(stream) {
                return crate::camdl_store::read_stored(p, file);
            }
        }
        // An extensionless name also matches its `.tsv` sibling.
        let found = if files.iter().any(|f| f == name) {
            name.to_string()
        } else if Path::new(name).extension().is_none() && files.iter().any(|f| *f == format!("{name}.tsv")) {
            format!("{name}.tsv")
        } else {
            return Err(format!("{} holds no file `{name}`; its files: {}", self.run_id(), files.join(", ")));
        };
        let bytes = self.read(&found)?;
        match selection {
            None => Ok(bytes),
            Some(sel) => {
                if !matches!(self, StoredFitResult::Run(..)) || found != "draws.tsv" {
                    return Err(format!(
                        "{} selects rows by chain, and applies to a fit run's draws.tsv",
                        sel.flag()
                    ));
                }
                select_chain_rows(out, &found, sel, &bytes).map(String::into_bytes)
            }
        }
    }
}

/// The fit run, problem or prediction a handle names on the run store:
/// `None` when it names none there, so the caller looks in the old store.
fn resolve_stored_fit(root: &Path, key: &str) -> Option<Result<StoredFitResult, String>> {
    let store = crate::camdl_store::open_existing(root)?;
    let handle: store::Handle = key.parse().ok()?;
    let not_here = |e: &store::Error| {
        matches!(e, store::Error::NotFound { .. } | store::Error::WrongKind { .. } | store::Error::NoRecord { .. })
    };
    match store.resolve::<crate::fit_store::FitRun>(&store::RunRef::new(handle.clone())) {
        Ok(s) => {
            if key.starts_with('@') {
                crate::out::detached().say(format_args!("{key} → run {} ({})",
                    &s.id().key().to_hex()[..6], crate::fit::report::display_path(s.folder())));
            }
            let extras = RunExtras::of(&store, &s);
            return Some(Ok(StoredFitResult::Run(s, extras)));
        }
        Err(e) if not_here(&e) => {}
        Err(e) => return Some(Err(e.to_string())),
    }
    match store.resolve::<crate::fit_store::FitProblem>(&store::RunRef::new(handle.clone())) {
        Ok(s) => return Some(Ok(StoredFitResult::Problem(s))),
        Err(e) if not_here(&e) => {}
        Err(e) => return Some(Err(e.to_string())),
    }
    match store.resolve::<crate::fit_store::Prediction>(&store::RunRef::new(handle)) {
        Ok(s) => Some(Ok(StoredFitResult::Prediction(s))),
        // A fit result the store holds in another state is refused with it.
        Err(e) if not_here(&e) => crate::fit_store::not_completed(&store, key).map(Err),
        Err(e) => Some(Err(e.to_string())),
    }
}

/// A simulation cell on the run store, prepared for `list` and `show`.
struct StoredSimRow {
    stored: store::Stored<crate::sim_store::Sim>,
    labels: crate::sim_store::CellLabels,
    /// The label of the latest invocation that asked for it, if it had one.
    label: Option<String>,
    rel_path: String,
    created: SystemTime,
}

impl StoredSimRow {
    fn json(&self) -> serde_json::Value {
        serde_json::json!({
            "kind": "sim",
            "run_id": self.stored.id().key().to_hex(),
            "label": self.label,
            "labels": self.labels,
            "path": self.rel_path,
            "identity": self.stored.identity(),
            "files": self.stored.result_files(),
            "failure": crate::sim_store::Failure::of(&self.stored).ok().flatten(),
        })
    }
}

/// Every completed simulation cell the run store's invocation records list,
/// once, with the label of the latest invocation that asked for it.
fn discover_stored_sim_rows(root: &Path) -> Vec<StoredSimRow> {
    let Some(store) = crate::camdl_store::open_existing(root) else { return Vec::new() };
    let cwd = std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
    let mut rows: std::collections::BTreeMap<store::Key, StoredSimRow> = std::collections::BTreeMap::new();
    for inv in store.invocations() {
        let Ok(members) = inv.members::<crate::sim_store::Sim>() else { continue };
        for (id, labels) in members {
            if let Some(row) = rows.get_mut(&id.key()) {
                if inv.label().is_some() {
                    row.label = inv.label().map(str::to_string);
                }
                continue;
            }
            let Ok(Some(stored)) = store.get(&id) else { continue };
            let created = std::fs::metadata(stored.folder()).and_then(|m| m.modified()).unwrap_or(SystemTime::UNIX_EPOCH);
            let rel_path = pathdiff_str(stored.folder(), &cwd);
            rows.insert(id.key(), StoredSimRow { stored, labels, label: inv.label().map(str::to_string), rel_path, created });
        }
    }
    rows.into_values().collect()
}

/// The simulation cell a handle names on the run store: `None` when the
/// handle names no cell there.
fn resolve_stored_sim(root: &Path, key: &str) -> Option<Result<StoredSimRow, String>> {
    let store = crate::camdl_store::open_existing(root)?;
    // A label names an invocation: it names a cell only when that invocation
    // asked for exactly one, and nothing here when it asked for none.
    if let Ok(h @ store::Handle::Label { .. }) = key.parse::<store::Handle>() {
        let invocation = store.invocation(&h).ok()?;
        let members = invocation.members::<crate::sim_store::Sim>().ok()?;
        return match members.as_slice() {
            [] => None,
            [(id, _)] => {
                let row = discover_stored_sim_rows(root).into_iter().find(|r| r.stored.id() == id);
                Some(row.ok_or_else(|| format!("{key} names a cell that is not stored")))
            }
            many => Some(Err(format!(
                "{key} names {} cells, where one is needed; `camdl cat {key}` renders their table",
                many.len()
            ))),
        };
    }
    let handle: store::RunRef<crate::sim_store::Sim> = key.parse().ok()?;
    let stored = match store.resolve(&handle) {
        Ok(s) => s,
        Err(store::Error::NotFound { .. } | store::Error::WrongKind { .. } | store::Error::NoRecord { .. }) => return None,
        Err(e) => return Some(Err(e.to_string())),
    };
    let row = discover_stored_sim_rows(root).into_iter().find(|r| r.stored.id() == stored.id());
    Some(row.ok_or_else(|| format!("{key} names a simulation cell no invocation record lists")))
}

/// The combined table of the `simulate` invocation a label names, rendered
/// from its stored cells: `None` for any other handle or invocation.
fn stored_sim_table(root: &Path, target: &str) -> Option<Result<Vec<u8>, String>> {
    let handle: store::Handle = target.parse().ok()?;
    if !matches!(handle, store::Handle::Label { .. }) {
        return None;
    }
    let store = crate::camdl_store::open_existing(root)?;
    let invocation = store.invocation(&handle).ok()?;
    if invocation.members::<crate::sim_store::Sim>().ok()?.is_empty() {
        return None;
    }
    let mut table = Vec::new();
    let options = crate::sim_store::CombinedTableOptions { dates: None };
    Some(
        invocation
            .render_to::<crate::sim_store::CombinedTable>(&store, &options, &mut table)
            .map(|()| table)
            .map_err(|e| e.to_string()),
    )
}

/// A file of a stored cell, or of one of its by-products: the event log, an
/// observation stream's `<stream>.tsv` (or the stream's name), the quantities. A by-product is found
/// through the invocation records, by the record of the cell its identity
/// holds.
fn stored_sim_file(root: &Path, row: &StoredSimRow, name: &str) -> Result<Vec<u8>, String> {
    match stored_sim_file_named(root, row, name) {
        Err(_) if !name.contains('.') => stored_sim_file_named(root, row, &format!("{name}.tsv")),
        r => r,
    }
}

fn stored_sim_file_named(root: &Path, row: &StoredSimRow, name: &str) -> Result<Vec<u8>, String> {
    use store::ContentAddressed;
    if row.stored.result_files().contains(&name) {
        return crate::camdl_store::read_stored(&row.stored, name);
    }
    let store = crate::camdl_store::open_existing(root).ok_or("no run store")?;
    let cell = row.stored.identity().content_hash();
    let mut names: Vec<String> = row.stored.result_files().iter().map(|s| s.to_string()).collect();
    for inv in store.invocations().iter().rev() {
        macro_rules! look {
            ($t:ty, $flag:literal) => {
                for (id, ()) in inv.members::<$t>().unwrap_or_default() {
                    let Ok(Some(s)) = store.get(&id) else {
                        // A by-product of this cell that is stored but fails
                        // its manifest check is not served; say which, and how
                        // to make it again.
                        if let Ok(Some(e)) = store.inspect(&id) {
                            if e.identity().sim.content_hash() == cell {
                                if let store::EntryState::Damaged { what } = e.state() {
                                    return Err(format!(
                                        "the cell's {} is damaged ({what}), so it is not read; run the \
                                         simulate again with {} and --force to store it again",
                                        <$t as store::Artifact>::KIND, $flag
                                    ));
                                }
                            }
                        }
                        continue;
                    };
                    if s.identity().sim.content_hash() != cell {
                        continue;
                    }
                    if s.result_files().contains(&name) {
                        return crate::camdl_store::read_stored(&s, name);
                    }
                    names.extend(s.result_files().iter().map(|f| f.to_string()));
                }
            };
        }
        look!(crate::sim_store::EventLog, "--event-log");
        look!(crate::sim_store::Obs, "--obs");
        look!(crate::sim_store::SimQuantities, "--quantities-out");
    }
    names.sort();
    names.dedup();
    Err(format!("the cell holds no file `{name}`; its files and its by-products' files: {}", names.join(", ")))
}

/// `camdl show` of a simulation cell on the run store.
fn show_stored_sim(out: &crate::out::Out, row: &StoredSimRow) {
    let field = |name: &str, value: &dyn std::fmt::Display| {
        out.report_line(format_args!("{}", style::stdout().paint(name, Style::Dim)));
        out.report_line(format_args!("  {value}"));
    };
    field("path", &style::stdout().paint(&row.rel_path, Style::Code));
    field("kind", &"sim");
    if let Some(l) = &row.label {
        field("label", l);
    }
    let id = row.stored.identity();
    field("model", &row.labels.model);
    field("config", &crate::sim_store::config_label(&id.config));
    field("scenario", &row.labels.scenario);
    field("seed", &row.labels.seed_label());
    field("process seed", &id.seed.seed);
    field("run_id", &style::stdout().paint(&row.stored.id().key().to_hex(), Style::Dim));
    if let Ok(Some(f)) = crate::sim_store::Failure::of(&row.stored) {
        field("failure", &format!("{}: {}", f.stage, f.reason));
    }
    if !id.params.values.is_empty() {
        out.report_line(format_args!("{}", style::stdout().paint("parameters", Style::Dim)));
        for (name, v) in &id.params.values {
            out.report_line(format_args!("  {:<16} {}", name, v.get()));
        }
    }
    field("files", &row.stored.result_files().join(", "));
    field("created", &fmt_relative_time(row.created, SystemTime::now()));
}

fn print_stored_sim_table(out: &crate::out::Out, rows: &[StoredSimRow], now: SystemTime) {
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
            Cell::new("SCENARIO").add_attribute(comfy_table::Attribute::Bold),
            Cell::new("SEED").add_attribute(comfy_table::Attribute::Bold),
            Cell::new("PATH").add_attribute(comfy_table::Attribute::Bold),
        ]);
    for r in rows {
        table.add_row(vec![
            Cell::new(fmt_relative_time(r.created, now)).fg(comfy_table::Color::Yellow),
            short_hash_cell(&r.stored.id().key().to_hex()),
            label_cell(&r.label),
            Cell::new(&r.labels.model),
            Cell::new(crate::sim_store::config_label(&r.stored.identity().config)).add_attribute(comfy_table::Attribute::Dim),
            Cell::new(&r.labels.scenario),
            Cell::new(r.labels.seed_label()),
            Cell::new(&r.rel_path).fg(comfy_table::Color::Cyan),
        ]);
    }
    out.report_line(format_args!("{table}"));
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

/// `--kind` filter: which of sims / fits / profiles / pfilters / ensembles
/// to surface. `All` is the default and includes every kind.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum KindFilter { Sim, Fit, Profile, Pfilter, Ensemble, All }

impl KindFilter {
    fn includes_sims(self)      -> bool { matches!(self, Self::Sim      | Self::All) }
    fn includes_fits(self)      -> bool { matches!(self, Self::Fit      | Self::All) }
    fn includes_profiles(self)  -> bool { matches!(self, Self::Profile  | Self::All) }
    fn includes_pfilters(self)  -> bool { matches!(self, Self::Pfilter  | Self::All) }
    fn includes_ensembles(self) -> bool { matches!(self, Self::Ensemble | Self::All) }
}

pub fn cmd_list(a: &crate::args::ListArgs) {
    let out = crate::out::Out::new(false);
    // --parent=HASH: enumerate the grid-point × start runs of one
    // specific profile. Takes precedence over the default sim/fit
    // enumeration because it's a more specific request; the other
    // filters (since, limit, format) still apply.
    let legacy = old_root(a.root_flag.as_deref().or(a.root.as_deref()), a.store.as_deref());
    if let Some(parent_hash) = a.parent.as_ref() {
        let Some(root) = legacy.as_ref() else {
            out.fail("error: --parent lists a profile's runs, which are kept under the results root; \
                name it with --root DIR or CAMDL_OUTPUT_DIR");
        };
        list_profile_children(&out, &root.to_string_lossy(), parent_hash, a);
        return;
    }

    let filter_since: Option<std::time::Duration> = a.since.as_ref().map(|d| d.0);
    let filter_kind = match a.kind.as_str() {
        "sim" | "simulate"      => KindFilter::Sim,
        "fit"                   => KindFilter::Fit,
        "profile" | "profiles"  => KindFilter::Profile,
        "pfilter" | "pfilters"  => KindFilter::Pfilter,
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
    let ensembles = match legacy.as_ref() {
        Some(root) if filter_kind.includes_ensembles() => discover_ensemble_rows(&root.to_string_lossy()),
        _ => Vec::new(),
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

    let runs = match legacy.as_ref() {
        Some(root) if filter_kind.includes_sims() => discover_sim_rows(
            &root.to_string_lossy(),
            cas_read::Bounds { skip: Some(&ensemble_member_ids), not_before: since_cutoff },
        ),
        _ => Vec::new(),
    };
    let fits = if !filter_kind.includes_fits() {
        Vec::new()
    } else {
        discover_stored_fit_rows(&store_root(a.store.as_deref()))
    };
    let profiles = match legacy.as_ref() {
        Some(root) if filter_kind.includes_profiles() => {
            discover_profiles(&root.to_string_lossy()).unwrap_or_else(|e| { out.fail(format!("error: {}", e)); })
        }
        _ => Vec::new(),
    };
    let mut stored_profiles = if !filter_kind.includes_profiles() {
        Vec::new()
    } else {
        crate::profile_store::discover_invocations(&store_root(a.store.as_deref()))
    };
    let pfilters = if !filter_kind.includes_pfilters() {
        Vec::new()
    } else {
        discover_pfilter_rows(&store_root(a.store.as_deref()))
    };
    let stored_sims = if !filter_kind.includes_sims() {
        Vec::new()
    } else {
        discover_stored_sim_rows(&store_root(a.store.as_deref()))
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

    let mut fit_rows: Vec<StoredFitRow> = fits.into_iter()
        .filter(|r| a.model.as_deref().is_none_or(|m| r.model.contains(m)))
        .filter(|_| a.scenario.is_none())
        .filter(|r| match filter_since {
            Some(dur) => now.duration_since(r.created).is_ok_and(|d| d <= dur),
            None => true,
        })
        .collect();
    fit_rows.sort_by(|x, y| y.created.cmp(&x.created));

    let mut filtered_profiles: Vec<ProfileEntry> = profiles.into_iter()
        .filter(|p| a.model.as_deref().is_none_or(|m| p.model.contains(m)))
        .filter(|_| a.scenario.is_none())
        .filter(|p| match filter_since {
            Some(dur) => now.duration_since(p.created).is_ok_and(|d| d <= dur),
            None => true,
        })
        .collect();
    filtered_profiles.sort_by(|x, y| y.created.cmp(&x.created));

    stored_profiles.retain(|p| a.model.as_deref().is_none_or(|m| p.model.contains(m))
        && a.scenario.is_none()
        && filter_since.is_none_or(|dur| now.duration_since(p.created).is_ok_and(|d| d <= dur)));
    stored_profiles.sort_by(|x, y| y.created.cmp(&x.created));

    let mut filtered_pfilters: Vec<PfilterRow> = pfilters.into_iter()
        .filter(|p| a.model.as_deref().is_none_or(|m| p.labels.model.contains(m)))
        .filter(|_| a.scenario.is_none())
        .filter(|p| match filter_since {
            Some(dur) => now.duration_since(p.created).is_ok_and(|d| d <= dur),
            None => true,
        })
        .collect();
    filtered_pfilters.sort_by(|x, y| y.created.cmp(&x.created));

    let mut filtered_stored_sims: Vec<StoredSimRow> = stored_sims.into_iter()
        .filter(|r| a.model.as_deref().is_none_or(|m| r.labels.model.contains(m)))
        .filter(|r| a.scenario.as_deref().is_none_or(|s| r.labels.scenario == s))
        .filter(|r| match filter_since {
            Some(dur) => now.duration_since(r.created).is_ok_and(|d| d <= dur),
            None => true,
        })
        .collect();
    filtered_stored_sims.sort_by(|x, y| y.created.cmp(&x.created));

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
        filtered_pfilters.truncate(a.limit);
        stored_profiles.truncate(a.limit);
        filtered_stored_sims.truncate(a.limit);
        filtered_ensembles.truncate(a.limit);
    }

    if format_json {
        print_sim_json(&out, &filtered_runs);
        for row in &fit_rows {
            out.report_line(format_args!("{}", serde_json::to_string(&row.json()).unwrap_or_default()));
        }
        print_profiles_json(&out, &filtered_profiles);
        print_pfilter_json(&out, &filtered_pfilters);
        for p in &stored_profiles {
            out.report_line(format_args!("{}", serde_json::to_string(&p.json()).unwrap_or_default()));
        }
        for r in &filtered_stored_sims {
            out.report_line(format_args!("{}", serde_json::to_string(&r.json()).unwrap_or_default()));
        }
        print_ensemble_json(&out, &filtered_ensembles);
    } else {
        let any_other = !fit_rows.is_empty()
            || !filtered_profiles.is_empty()
            || !filtered_pfilters.is_empty()
            || !stored_profiles.is_empty()
            || !filtered_stored_sims.is_empty()
            || !filtered_ensembles.is_empty();
        if !filtered_ensembles.is_empty() {
            out.say(format_args!("{}", style::stderr().paint("ensembles", Style::Bold)));
            print_ensemble_table(&out, &filtered_ensembles, now);
            out.say("");
        }
        if !fit_rows.is_empty() {
            out.say(format_args!("{}", style::stderr().paint("fits", Style::Bold)));
            print_stored_fit_table(&out, &fit_rows, now);
            out.say("");
        }
        if !filtered_profiles.is_empty() {
            out.say(format_args!("{}", style::stderr().paint("profiles", Style::Bold)));
            print_profiles_table(&out, &filtered_profiles, now);
            out.say("");
        }
        if !stored_profiles.is_empty() {
            out.say(format_args!("{}", style::stderr().paint("profiles", Style::Bold)));
            print_stored_profile_table(&out, &stored_profiles, now);
            out.say("");
        }
        if !filtered_pfilters.is_empty() {
            out.say(format_args!("{}", style::stderr().paint("pfilters", Style::Bold)));
            print_pfilter_table(&out, &filtered_pfilters, now);
            out.say("");
        }
        if !filtered_stored_sims.is_empty() {
            out.say(format_args!("{}", style::stderr().paint("sims", Style::Bold)));
            print_stored_sim_table(&out, &filtered_stored_sims, now);
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
    let json = show_json(&out, a.format.as_deref());
    if let Some(row) = resolve_stored_pfilter(&store_root_for(a.store.as_deref(), &a.target), &a.target) {
        let row = row.unwrap_or_else(|e| out.fail(format!("error: {e}")));
        if json {
            out.report_line(format_args!("{}", serde_json::to_string_pretty(&row.json()).unwrap_or_default()));
        } else {
            show_stored_pfilter(&out, &row);
        }
        return;
    }
    if let Some(point) = crate::profile_store::resolve_point(&store_root_for(a.store.as_deref(), &a.target), &a.target) {
        let point = point.unwrap_or_else(|e| out.fail(format!("error: {e}")));
        let cwd = std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
        let rel_path = pathdiff_str(point.folder(), &cwd);
        if json {
            let value = serde_json::json!({
                "kind": <crate::profile_store::ProfilePoint as store::Artifact>::KIND,
                "run_id": point.id().key().to_hex(),
                "path": rel_path,
                "identity": point.identity(),
            });
            out.report_line(format_args!("{}", serde_json::to_string_pretty(&value).unwrap_or_default()));
        } else {
            crate::profile_store::show_point(&out, &point, &rel_path);
        }
        return;
    }
    if let Some(row) = resolve_stored_sim(&store_root_for(a.store.as_deref(), &a.target), &a.target) {
        let row = row.unwrap_or_else(|e| out.fail(format!("error: {e}")));
        if json {
            out.report_line(format_args!("{}", serde_json::to_string_pretty(&row.json()).unwrap_or_default()));
        } else {
            show_stored_sim(&out, &row);
        }
        return;
    }
    if let Some(found) = resolve_stored_fit(&store_root_for(a.store.as_deref(), &a.target), &a.target) {
        let found = found.unwrap_or_else(|e| out.fail(format!("error: {e}")));
        if json {
            out.report_line(format_args!("{}", serde_json::to_string_pretty(&found.json()).unwrap_or_default()));
        } else {
            found.show(&out);
        }
        return;
    }
    let resolved = resolve_named(old_root(a.root.as_deref(), a.store.as_deref()).as_deref(), &a.target, a.store.as_deref())
        .unwrap_or_else(|e| out.fail(format!("error: {}", e)));
    if json {
        let value = match &resolved {
            Resolved::Sim { leaf, rel_path, .. } => run_show_json(leaf, "sim", rel_path),
            Resolved::Profile { leaf, rel_path, .. } => run_show_json(leaf, "profile_point", rel_path),
            Resolved::SimEnsemble { leaf, rel_path, .. } => run_show_json(leaf, "sim_ensemble", rel_path),
        };
        out.report_line(format_args!("{}", serde_json::to_string_pretty(&value).unwrap_or_default()));
        return;
    }
    match Ok::<Resolved, String>(resolved) {
        Ok(Resolved::Sim { leaf, rel_path, created }) => show_sim_record(&out, &leaf, &rel_path, created),
        Ok(Resolved::Profile { leaf, rel_path, created }) => show_profile_record(&out, &leaf, &rel_path, created),
        Ok(Resolved::SimEnsemble { leaf, rel_path, created }) => show_sim_ensemble_record(&out, &leaf, &rel_path, created),
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
    use std::io::Write as _;

    // A label naming a `pfilter --sweep-file` invocation: its design table,
    // rendered from the stored points.
    if a.stream.is_none() && selection.is_none() {
        if let Some(table) = stored_design_table(&store_root_for(a.store.as_deref(), &a.target), &a.target) {
            let table = table.unwrap_or_else(|e| out.fail(format!("error: {e}")));
            let _ = out.data().write_all(&table);
            return;
        }
    }

    // A `profile` invocation on the run store, by its label: the profile
    // table, rendered from its stored points.
    if a.stream.is_none() && selection.is_none() {
        if let Some(table) = crate::profile_store::invocation_table(&store_root_for(a.store.as_deref(), &a.target), &a.target) {
            let table = table.unwrap_or_else(|e| out.fail(format!("error: {e}")));
            let _ = out.data().write_all(&table);
            return;
        }
    }

    // A profile point on the run store: `mle.toml`, or a named result file.
    if let Some(point) = crate::profile_store::resolve_point(&store_root_for(a.store.as_deref(), &a.target), &a.target) {
        let point = point.unwrap_or_else(|e| out.fail(format!("error: {e}")));
        if let Some(sel) = &selection {
            out.fail(format!("error: {} selects rows of a fit run's chain-keyed stream; `{}` names a profile point",
                sel.flag(), a.target));
        }
        let name = a.stream.as_deref().unwrap_or(crate::profile_store::MLE_FILE);
        let bytes = crate::camdl_store::read_stored(&point, name).unwrap_or_else(|e| out.fail(format!("error: {e}")));
        let _ = out.data().write_all(&bytes);
        return;
    }

    // A `simulate` invocation on the run store, by its label: the combined
    // table, rendered from its stored cells.
    if a.stream.is_none() && selection.is_none() {
        if let Some(table) = stored_sim_table(&store_root_for(a.store.as_deref(), &a.target), &a.target) {
            let table = table.unwrap_or_else(|e| out.fail(format!("error: {e}")));
            let _ = out.data().write_all(&table);
            return;
        }
    }

    // A simulation cell on the run store: `traj.tsv`, or a named file of the
    // cell or of one of its by-products.
    if let Some(row) = resolve_stored_sim(&store_root_for(a.store.as_deref(), &a.target), &a.target) {
        let row = row.unwrap_or_else(|e| out.fail(format!("error: {e}")));
        if let Some(sel) = &selection {
            out.fail(format!("error: {} selects rows of a fit run's chain-keyed stream; `{}` names a simulation cell",
                sel.flag(), a.target));
        }
        let name = a.stream.as_deref().unwrap_or(crate::sim_store::TRAJ_FILE);
        let bytes = stored_sim_file(&store_root(a.store.as_deref()), &row, name).unwrap_or_else(|e| out.fail(format!("error: {e}")));
        let _ = out.data().write_all(&bytes);
        return;
    }

    // A filter run on the run store: `loglik.toml`, or a named result file.
    if let Some(row) = resolve_stored_pfilter(&store_root_for(a.store.as_deref(), &a.target), &a.target) {
        let row = row.unwrap_or_else(|e| out.fail(format!("error: {e}")));
        if let Some(sel) = &selection {
            out.fail(format!("error: {} selects rows of a fit run's chain-keyed stream; `{}` names a filter run",
                sel.flag(), a.target));
        }
        let name = a.stream.as_deref().unwrap_or(crate::pfilter_store::LOGLIK_FILE);
        let bytes = stored_pfilter_file(&store_root(a.store.as_deref()), &row, name)
            .unwrap_or_else(|e| out.fail(format!("error: {e}")));
        let _ = out.data().write_all(&bytes);
        return;
    }
    // A fit run, a fit problem or a prediction on the run store: a named
    // result file, `fit_state.toml` for a run, `problem.toml` for a problem.
    if let Some(found) = resolve_stored_fit(&store_root_for(a.store.as_deref(), &a.target), &a.target) {
        let found = found.unwrap_or_else(|e| out.fail(format!("error: {e}")));
        let bytes = found.cat(a.stream.as_deref(), selection.as_ref(), &out).unwrap_or_else(|e| out.fail(format!("error: {e}")));
        let _ = out.data().write_all(&bytes);
        return;
    }
    let resolved = resolve_named(old_root(a.root.as_deref(), a.store.as_deref()).as_deref(), &a.target, a.store.as_deref())
        .unwrap_or_else(|e| out.fail(format!("error: {}", e)));

    // A chain selection reads a fit run's own stream; every other kind is
    // refused rather than printed whole.
    if let Some(sel) = &selection {
        out.fail(format!(
            "error: {} selects rows of a fit run's chain-keyed stream (draws.tsv); \
             `{}` does not name a fit run",
            sel.flag(), a.target
        ));
    }

    // Every kind is new-format (`RunRecord`); each arm emits and returns.
    match resolved {
        Resolved::Sim { leaf, rel_path, .. } => {
            let bytes = if let Some(ref stream) = a.stream {
                let named_event_log = stream == "event_log.tsv" || stream == "event_log";
                let path = match find_obs_stream(&leaf.dir, stream) {
                    Some(p) => p,
                    None if named_event_log => leaf.dir.join("event_log").join("event_log.tsv"),
                    None => out.fail(format!(
                        "error: no observation stream '{}' in {} (the event log is `event_log.tsv`)",
                        stream, rel_path)),
                };
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

    }
}

/// The rows of a fit run's `draws.tsv` that a chain selection keeps
/// (`cat --chains` / `--exclude-chains`). The selection is resolved against
/// the file's own chains, warned about on stderr, and the kept rows are the
/// file's own bytes.
fn select_chain_rows(
    out: &crate::out::Out,
    name: &str,
    sel: &crate::chain_selection::ChainSelection,
    bytes: &[u8],
) -> Result<String, String> {
    let text = std::str::from_utf8(bytes).map_err(|_| format!("`{name}` is not UTF-8 text"))?;
    let scratch = tempfile::tempdir().map_err(|e| e.to_string())?;
    let copy = scratch.path().join(name);
    std::fs::write(&copy, bytes).map_err(|e| e.to_string())?;
    let present = crate::chain_selection::chains_in_draws(&copy)?;
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
// (data-driven depth through [`cas_read`]). The legacy fit/profile
// discovery below is M3-DELETION-BOUND (gh#147) — see [`load_run_common`].


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


/// A resolved run — always a new-format (`runid::RunRecord`) leaf, dispatched
/// by `ArtifactKind`.
#[derive(Debug)]
enum Resolved {
    Sim { leaf: cas_read::Leaf, rel_path: String, created: SystemTime },
    /// `ProfilePoint` leaf under `profiles/`.
    Profile { leaf: cas_read::Leaf, rel_path: String, created: SystemTime },
    /// `SimEnsemble` leaf under `ensembles/` (the combined multi-cell TSV).
    SimEnsemble { leaf: cas_read::Leaf, rel_path: String, created: SystemTime },
}

/// A run `show` and `cat` did not find in the run store: resolved under the
/// old results root `root` when there is one, else only by the path of its
/// folder.
fn resolve_named(root: Option<&Path>, key: &str, store: Option<&Path>) -> Result<Resolved, String> {
    match root {
        Some(root) => resolve_any(&root.to_string_lossy(), key),
        None => resolve_path(key)
            .unwrap_or_else(|| Err(format!("no run matches '{}' in the run store {}", key, store_root(store).display()))),
    }
}

/// A run of the old store named by the path of its folder, when `key` is one:
/// a folder holding a `run.json`.
fn resolve_path(key: &str) -> Option<Result<Resolved, String>> {
    let cwd = std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
    let as_path = Path::new(key);
    if as_path.is_dir() && as_path.join("run.json").exists() {
        let rec = match runid::read_leaf(as_path) {
            Ok(Some(leaf)) => leaf.record,
            Ok(None) => return Some(Err(format!("{} holds no run.json", as_path.display()))),
            Err(u) => {
                return Some(Err(format!("could not read run.json at {}: {}", as_path.display(), u.reason)))
            }
        };
        let kind = rec.kind;
        let leaf = cas_read::Leaf { dir: as_path.to_path_buf(), record: rec };
        let created = leaf_created(&leaf);
        let rel_path = pathdiff_str(as_path, &cwd);
        return Some(Ok(match kind {
            runid::ArtifactKind::ProfilePoint => Resolved::Profile { leaf, rel_path, created },
            runid::ArtifactKind::SimEnsemble => Resolved::SimEnsemble { leaf, rel_path, created },
            _ => Resolved::Sim { leaf, rel_path, created },
        }));
    }

    None
}

/// Resolve a user-supplied key to a single new-format leaf. Accepts either a
/// path to a `run.json`-containing leaf directory, or a hash prefix where
/// `{prefix}/{scenario}[/{seed_N}]` narrows sims further. An ambiguous prefix
/// errors, listing all candidates with their kinds.
fn resolve_any(root: &str, key: &str) -> Result<Resolved, String> {
    let cwd = std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."));

    // Path form: read the leaf `run.json` (`RunRecord`) directly.
    if let Some(resolved) = resolve_path(key) {
        return resolved;
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

    // New-format profile points (M3.3): match the run_id hex prefix under
    // profiles/. A profile-base prefix is the umbrella view — use `list
    // --parent <base>` for that; here a single leaf is addressed.
    let mut profile_matches: Vec<(cas_read::Leaf, String, SystemTime)> = Vec::new();
    for leaf in cas_read::resolve_profile_prefix(Path::new(root), hash_prefix) {
        let created = leaf_created(&leaf);
        let rel = pathdiff_str(&leaf.dir, &cwd);
        profile_matches.push((leaf, rel, created));
    }

    // New-format sim ensembles: match the run_id hex prefix under ensembles/.
    let mut ensemble_matches: Vec<(cas_read::Leaf, String, SystemTime)> = Vec::new();
    for leaf in cas_read::resolve_sim_ensemble_prefix(Path::new(root), hash_prefix) {
        let created = leaf_created(&leaf);
        let rel = pathdiff_str(&leaf.dir, &cwd);
        ensemble_matches.push((leaf, rel, created));
    }

    match sim_matches.len() + profile_matches.len() + ensemble_matches.len() {
        0 => Err(format!("no run matches '{}' in {}", key, root)),
        1 => {
            if let Some((leaf, rel_path, created)) = sim_matches.into_iter().next() {
                Ok(Resolved::Sim { leaf, rel_path, created })
            } else if let Some((leaf, rel_path, created)) = profile_matches.into_iter().next() {
                Ok(Resolved::Profile { leaf, rel_path, created })
            } else {
                let (leaf, rel_path, created) = ensemble_matches.into_iter().next().expect("one match");
                Ok(Resolved::SimEnsemble { leaf, rel_path, created })
            }
        }
        n => {
            let mut msg = format!("'{}' is ambiguous, matches {} entries:\n", key, n);
            for (_, rel, _) in &sim_matches {
                msg.push_str(&format!("  {:<14} {}\n", "sim", rel));
            }
            for (_, rel, _) in &profile_matches {
                msg.push_str(&format!("  {:<14} {}\n", "profile_point", rel));
            }
            for (_, rel, _) in &ensemble_matches {
                msg.push_str(&format!("  {:<14} {}\n", "sim_ensemble", rel));
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

/// The profiles on the run store, one row per invocation: its sequence
/// number, label, model, focal parameters, method and points.
fn print_stored_profile_table(out: &crate::out::Out, rows: &[crate::profile_store::ProfileInvocationRow], now: SystemTime) {
    use comfy_table::{Table, Cell, ContentArrangement, presets::NOTHING};
    let mut table = Table::new();
    table
        .load_preset(NOTHING)
        .set_content_arrangement(ContentArrangement::Dynamic)
        .set_header(["CREATED", "INVOCATION", "LABEL", "MODEL", "FOCAL", "METHOD", "POINTS"]
            .map(|h| Cell::new(h).add_attribute(comfy_table::Attribute::Bold)));
    for r in rows {
        table.add_row(vec![
            Cell::new(fmt_relative_time(r.created, now)).fg(comfy_table::Color::Yellow),
            Cell::new(r.seq),
            label_cell(&r.label),
            Cell::new(&r.model),
            Cell::new(r.focal.join(",")),
            Cell::new(&r.method).add_attribute(comfy_table::Attribute::Dim),
            Cell::new(format!("{}/{}", r.stored, r.points.len())),
        ]);
    }
    out.report_line(format_args!("{table}"));
}

fn print_pfilter_json(out: &crate::out::Out, rows: &[PfilterRow]) {
    for r in rows {
        out.report_line(format_args!("{}", serde_json::to_string(&r.json()).unwrap_or_default()));
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
        let id = r.stored.identity();
        let loglik = r.loglik().map(|l| format!("{:.2}", l.loglik.unwrap_or(f64::NEG_INFINITY))).unwrap_or_else(|| "—".into());
        table.add_row(vec![
            Cell::new(fmt_relative_time(r.created, now)).fg(comfy_table::Color::Yellow),
            short_hash_cell(&r.stored.id().key().to_hex()),
            label_cell(&r.label),
            Cell::new(&r.labels.model),
            Cell::new(crate::pfilter_store::config_label(id.config.particles, id.config.replicates, id.config.dt.get()))
                .add_attribute(comfy_table::Attribute::Dim),
            Cell::new(r.labels.seed),
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


// ── Shared show helpers ──────────────────────────────────────────────────────

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

// ── Tests ────────────────────────────────────────────────────────────────────

/// The `engine` line of a record: the readable version that ran it, the
/// engine's build fingerprint (`run.json`'s `engine_version`) and any engine
/// switch the run was keyed with (`engine_switches`), e.g.
/// `0.4.0+abc1234.dirty-3f9c21e0 (engine 9514b53cefba5cfc…)` or
/// `… (engine 9514b53cefba5cfc…, with CAMDL_EVAL_UNRESOLVED)`.
fn engine_line(rec: &runid::RunRecord) -> String {
    let id = &rec.engine_version;
    let switches: Vec<String> = rec.engine_switches.iter()
        .map(|s| match &s.value {
            Some(v) => format!("{}={v}", s.name),
            None => s.name.clone(),
        })
        .collect();
    let with = if switches.is_empty() { String::new() } else { format!(", with {}", switches.join(", ")) };
    match rec.provenance.camdl_version.as_deref() {
        Some(v) if v != id => format!("{v} (engine {id}{with})"),
        _ if !with.is_empty() => format!("{id} ({})", &with[2..]),
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
            store_layout: runid::STORE_LAYOUT,
            ir_version: "0.7".into(),
            engine_version: "test".into(),
            engine_switches: Vec::new(),
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
