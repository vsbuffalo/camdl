//! `camdl list`, `camdl show`, `camdl cat`: browse the run store's results.
//! Each reads one store root: `--store`, else `CAMDL_STORE`, else
//! `./camdl-store`; `show` and `cat` given a result's folder read the store
//! that folder lies in.

use std::path::{Path, PathBuf};
use std::time::SystemTime;

use crate::out::paint::{self as style, Style};

use crate::util::fmt_relative_time;

// ── Entry types ──────────────────────────────────────────────────────────────

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
    Score(store::Stored<crate::fit_store::Score>),
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
                let crate::fit_store::ParameterSourceRecord::Posterior { draws, .. } = &p.identity().source else {
                    continue;
                };
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
            StoredFitResult::Score(_) => "score",
        }
    }

    fn folder(&self) -> &Path {
        match self {
            StoredFitResult::Run(s, _) => s.folder(),
            StoredFitResult::Problem(s) => s.folder(),
            StoredFitResult::Prediction(s) => s.folder(),
            StoredFitResult::Score(s) => s.folder(),
        }
    }

    fn run_id(&self) -> String {
        match self {
            StoredFitResult::Run(s, _) => s.id().key().to_hex(),
            StoredFitResult::Problem(s) => s.id().key().to_hex(),
            StoredFitResult::Prediction(s) => s.id().key().to_hex(),
            StoredFitResult::Score(s) => s.id().key().to_hex(),
        }
    }

    fn result_files(&self) -> Vec<String> {
        let names = match self {
            StoredFitResult::Run(s, _) => s.result_files(),
            StoredFitResult::Problem(s) => s.result_files(),
            StoredFitResult::Prediction(s) => s.result_files(),
            StoredFitResult::Score(s) => s.result_files(),
        };
        names.into_iter().map(str::to_string).collect()
    }

    fn identity(&self) -> serde_json::Value {
        match self {
            StoredFitResult::Run(s, _) => serde_json::to_value(s.identity()),
            StoredFitResult::Problem(s) => serde_json::to_value(s.identity()),
            StoredFitResult::Prediction(s) => serde_json::to_value(s.identity()),
            StoredFitResult::Score(s) => serde_json::to_value(s.identity()),
        }
        .unwrap_or(serde_json::Value::Null)
    }

    fn read(&self, name: &str) -> Result<Vec<u8>, String> {
        match self {
            StoredFitResult::Run(s, _) => crate::camdl_store::read_stored(s, name),
            StoredFitResult::Problem(s) => crate::camdl_store::read_stored(s, name),
            StoredFitResult::Prediction(s) => crate::camdl_store::read_stored(s, name),
            StoredFitResult::Score(s) => crate::camdl_store::read_stored(s, name),
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
            match &s.identity().source {
                crate::fit_store::ParameterSourceRecord::Posterior { draws, .. } => {
                    field("draws of", &draws.id().key().to_hex());
                }
                crate::fit_store::ParameterSourceRecord::FixedPoint { model, trained_through, .. } => {
                    field("model", &model.id().key().to_hex());
                    field(
                        "parameters",
                        &"one fixed point (treatment plug_in): the bands carry process and observation \
                          noise and no parameter uncertainty",
                    );
                    if let Some(t) = trained_through {
                        field("trained through", &t.get());
                    }
                }
            }
        }
        if let StoredFitResult::Score(s) = self {
            field("prediction", &s.identity().prediction.id().key().to_hex());
            let outcomes: Vec<&String> = s.identity().outcomes.keys().collect();
            field("outcomes of", &outcomes.iter().map(|s| s.as_str()).collect::<Vec<_>>().join(", "));
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
            StoredFitResult::Score(_) => "summary.tsv",
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
    match store.resolve::<crate::fit_store::Prediction>(&store::RunRef::new(handle.clone())) {
        Ok(s) => return Some(Ok(StoredFitResult::Prediction(s))),
        Err(e) if not_here(&e) => {}
        Err(e) => return Some(Err(e.to_string())),
    }
    match store.resolve::<crate::fit_store::Score>(&store::RunRef::new(handle)) {
        Ok(s) => Some(Ok(StoredFitResult::Score(s))),
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

// ── cmd_list ─────────────────────────────────────────────────────────────────

/// `--kind` filter: which of sims / fits / profiles / pfilters to surface.
/// `All` is the default and includes every kind.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum KindFilter { Sim, Fit, Profile, Pfilter, All }

impl KindFilter {
    fn includes_sims(self)      -> bool { matches!(self, Self::Sim      | Self::All) }
    fn includes_fits(self)      -> bool { matches!(self, Self::Fit      | Self::All) }
    fn includes_profiles(self)  -> bool { matches!(self, Self::Profile  | Self::All) }
    fn includes_pfilters(self)  -> bool { matches!(self, Self::Pfilter  | Self::All) }
}

pub fn cmd_list(a: &crate::args::ListArgs) {
    let out = crate::out::Out::new(false);
    let filter_since: Option<std::time::Duration> = a.since.as_ref().map(|d| d.0);
    let filter_kind = match a.kind.as_str() {
        "sim" | "simulate"      => KindFilter::Sim,
        "fit"                   => KindFilter::Fit,
        "profile" | "profiles"  => KindFilter::Profile,
        "pfilter" | "pfilters"  => KindFilter::Pfilter,
        _                       => KindFilter::All,
    };
    let format_json = a.format.as_deref() == Some("json");
    let root = store_root(a.store.as_deref());

    let fits = if !filter_kind.includes_fits() { Vec::new() } else { discover_stored_fit_rows(&root) };
    let mut stored_profiles = if !filter_kind.includes_profiles() {
        Vec::new()
    } else {
        crate::profile_store::discover_invocations(&root)
    };
    let pfilters = if !filter_kind.includes_pfilters() { Vec::new() } else { discover_pfilter_rows(&root) };
    let stored_sims = if !filter_kind.includes_sims() { Vec::new() } else { discover_stored_sim_rows(&root) };

    let now = SystemTime::now();

    let mut fit_rows: Vec<StoredFitRow> = fits.into_iter()
        .filter(|r| a.model.as_deref().is_none_or(|m| r.model.contains(m)))
        .filter(|_| a.scenario.is_none())
        .filter(|r| match filter_since {
            Some(dur) => now.duration_since(r.created).is_ok_and(|d| d <= dur),
            None => true,
        })
        .collect();
    fit_rows.sort_by(|x, y| y.created.cmp(&x.created));

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

    if !a.all {
        fit_rows.truncate(a.limit);
        filtered_pfilters.truncate(a.limit);
        stored_profiles.truncate(a.limit);
        filtered_stored_sims.truncate(a.limit);
    }

    if format_json {
        for row in &fit_rows {
            out.report_line(format_args!("{}", serde_json::to_string(&row.json()).unwrap_or_default()));
        }
        print_pfilter_json(&out, &filtered_pfilters);
        for p in &stored_profiles {
            out.report_line(format_args!("{}", serde_json::to_string(&p.json()).unwrap_or_default()));
        }
        for r in &filtered_stored_sims {
            out.report_line(format_args!("{}", serde_json::to_string(&r.json()).unwrap_or_default()));
        }
    } else {
        let any = !fit_rows.is_empty()
            || !filtered_pfilters.is_empty()
            || !stored_profiles.is_empty()
            || !filtered_stored_sims.is_empty();
        if !fit_rows.is_empty() {
            out.say(format_args!("{}", style::stderr().paint("fits", Style::Bold)));
            print_stored_fit_table(&out, &fit_rows, now);
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
        if !any {
            out.say(format_args!("{}", style::stderr().paint("(no stored results)", Style::Dim)));
        }
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
    out.fail(format!("error: no run matches '{}' in the run store {}",
        a.target, store_root_for(a.store.as_deref(), &a.target).display()));
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
    out.fail(format!("error: no run matches '{}' in the run store {}",
        a.target, store_root_for(a.store.as_deref(), &a.target).display()));
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

// ── Output formatting ────────────────────────────────────────────────────────

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

    // ── Transitional reader: new-format (RunRecord) sim resolution ──────────

}
