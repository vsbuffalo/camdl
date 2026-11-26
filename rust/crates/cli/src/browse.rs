//! `camdl list`, `camdl show`, `camdl cat`, `camdl path`: browse the run
//! store's results. Each reads one store root: `--store`, else
//! `CAMDL_STORE`, else `./camdl-store`; `show`, `cat` and `path` given a
//! result's folder read the store that folder lies in. `show`, `cat` and
//! `path` find a result and its files through [`crate::resolve_file`].

use std::path::{Path, PathBuf};
use std::time::SystemTime;

use crate::out::paint::{self as style, Style};

use crate::util::fmt_relative_time;

// ── Entry types ──────────────────────────────────────────────────────────────

/// A filter run on the run store, prepared for `list` and `show`.
struct PfilterRow {
    stored: store::Stored<crate::pfilter_store::Pfilter>,
    attributes: crate::pfilter_store::PfilterAttributes,
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
            "attributes": self.attributes,
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
fn discover_pfilter_rows(store: &store::Store) -> Vec<PfilterRow> {
    let cwd = std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
    let mut rows: std::collections::BTreeMap<store::Key, PfilterRow> = std::collections::BTreeMap::new();
    for inv in store.invocations().iter() {
        let Ok(members) = inv.members::<crate::pfilter_store::Pfilter>() else { continue };
        for (id, attributes) in members {
            if let Some(row) = rows.get_mut(&id.key()) {
                if inv.label().is_some() {
                    row.label = inv.label().map(str::to_string);
                }
                continue;
            }
            let Ok(Some(stored)) = store.get(&id) else { continue };
            let created = std::fs::metadata(stored.folder()).and_then(|m| m.modified()).unwrap_or(SystemTime::UNIX_EPOCH);
            let rel_path = pathdiff_str(stored.folder(), &cwd);
            rows.insert(id.key(), PfilterRow { stored, attributes, label: inv.label().map(str::to_string), rel_path, created });
        }
    }
    rows.into_values().collect()
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
    if !members.iter().any(|(_, attributes)| attributes.point.is_some()) {
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
    field("model", &row.attributes.model);
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
    attributes: crate::fit_store::RunAttributes,
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
            "method": self.attributes.method,
            "seed": self.attributes.seed,
            "model": self.model,
            "path": self.rel_path,
            "best_loglik": self.best_loglik(),
        })
    }
}

/// The model a stored problem was posed on, by the name the invocation that
/// answers for it gave it ([`store::Store::listings`]).
fn problem_model_name(store: &store::Store, problem: &store::Id<crate::fit_store::FitProblem>) -> String {
    store
        .listings(&store::Handle::IdPrefix(problem.key().to_hex()))
        .ok()
        .and_then(|l| l.answering()?.attributes_of(problem))
        .map(|a| a.model)
        .unwrap_or_default()
}

/// Every completed fit run the run store's invocation records list, once,
/// with the label of the latest invocation that asked for it.
fn discover_stored_fit_rows(store: &store::Store) -> Vec<StoredFitRow> {
    let cwd = std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
    let mut rows: std::collections::BTreeMap<store::Key, StoredFitRow> = std::collections::BTreeMap::new();
    for inv in store.invocations().iter() {
        let Ok(members) = inv.members::<crate::fit_store::FitRun>() else { continue };
        for (id, attributes) in members {
            if let Some(row) = rows.get_mut(&id.key()) {
                if inv.label().is_some() {
                    row.label = inv.label().map(str::to_string);
                }
                continue;
            }
            let Ok(Some(stored)) = store.get(&id) else { continue };
            let created = std::fs::metadata(stored.folder()).and_then(|m| m.modified()).unwrap_or(SystemTime::UNIX_EPOCH);
            let rel_path = pathdiff_str(stored.folder(), &cwd);
            let model = problem_model_name(store, stored.identity().chain.problem.id());
            rows.insert(id.key(), StoredFitRow { stored, attributes, model, label: inv.label().map(str::to_string), rel_path, created });
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
            Cell::new(&r.attributes.method),
            Cell::new(r.attributes.seed),
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
        for inv in store.invocations().iter() {
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
            field("method", &id.chain.method.algorithm());
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
        // Each prediction's files, by the path `cat <run> NAME` takes.
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
}

/// A simulation cell on the run store, prepared for `list` and `show`.
struct StoredSimRow {
    stored: store::Stored<crate::sim_store::Sim>,
    attributes: crate::sim_store::CellAttributes,
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
            "attributes": self.attributes,
            "path": self.rel_path,
            "identity": self.stored.identity(),
            "files": self.stored.result_files(),
            "failure": crate::sim_store::Failure::of(&self.stored).ok().flatten(),
        })
    }
}

/// Every completed simulation cell the run store's invocation records list,
/// once, with the label of the latest invocation that asked for it.
fn discover_stored_sim_rows(store: &store::Store) -> Vec<StoredSimRow> {
    let cwd = std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
    let mut rows: std::collections::BTreeMap<store::Key, StoredSimRow> = std::collections::BTreeMap::new();
    for inv in store.invocations().iter() {
        let Ok(members) = inv.members::<crate::sim_store::Sim>() else { continue };
        for (id, attributes) in members {
            if let Some(row) = rows.get_mut(&id.key()) {
                if inv.label().is_some() {
                    row.label = inv.label().map(str::to_string);
                }
                continue;
            }
            let Ok(Some(stored)) = store.get(&id) else { continue };
            let created = std::fs::metadata(stored.folder()).and_then(|m| m.modified()).unwrap_or(SystemTime::UNIX_EPOCH);
            let rel_path = pathdiff_str(stored.folder(), &cwd);
            rows.insert(id.key(), StoredSimRow { stored, attributes, label: inv.label().map(str::to_string), rel_path, created });
        }
    }
    rows.into_values().collect()
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
    field("model", &row.attributes.model);
    field("config", &crate::sim_store::config_label(&id.config));
    field("scenario", &row.attributes.scenario);
    field("seed", &row.attributes.seed_label());
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
            Cell::new(&r.attributes.model),
            Cell::new(crate::sim_store::config_label(&r.stored.identity().config)).add_attribute(comfy_table::Attribute::Dim),
            Cell::new(&r.attributes.scenario),
            Cell::new(r.attributes.seed_label()),
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

/// The rows `camdl list` shows of the kinds `filter` keeps, from one opening
/// of the store, so the invocation records are read once.
#[derive(Default)]
struct Rows {
    fits: Vec<StoredFitRow>,
    profiles: Vec<crate::profile_store::ProfileInvocationRow>,
    pfilters: Vec<PfilterRow>,
    sims: Vec<StoredSimRow>,
}

fn rows(store: &store::Store, filter: KindFilter) -> Rows {
    Rows {
        fits: if filter.includes_fits() { discover_stored_fit_rows(store) } else { Vec::new() },
        profiles: if filter.includes_profiles() { crate::profile_store::discover_invocations(store) } else { Vec::new() },
        pfilters: if filter.includes_pfilters() { discover_pfilter_rows(store) } else { Vec::new() },
        sims: if filter.includes_sims() { discover_stored_sim_rows(store) } else { Vec::new() },
    }
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
    let Rows { fits, profiles: mut stored_profiles, pfilters, sims: stored_sims } =
        crate::camdl_store::open_existing(&root).map(|store| rows(&store, filter_kind)).unwrap_or_default();

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
        .filter(|p| a.model.as_deref().is_none_or(|m| p.attributes.model.contains(m)))
        .filter(|_| a.scenario.is_none())
        .filter(|p| match filter_since {
            Some(dur) => now.duration_since(p.created).is_ok_and(|d| d <= dur),
            None => true,
        })
        .collect();
    filtered_pfilters.sort_by(|x, y| y.created.cmp(&x.created));

    let mut filtered_stored_sims: Vec<StoredSimRow> = stored_sims.into_iter()
        .filter(|r| a.model.as_deref().is_none_or(|m| r.attributes.model.contains(m)))
        .filter(|r| a.scenario.as_deref().is_none_or(|s| r.attributes.scenario == s))
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

// ── The lookup `show`, `cat` and `path` share ────────────────────────────────

/// What a reading command names: a handle, or an output manifest.
enum Target {
    Handle(store::Handle),
    Manifest { path: PathBuf, manifest: crate::output_manifest::OutputManifest },
}

/// A reading command's store, the results it names and the file names it
/// asks for.
struct Lookup {
    store: store::Store,
    target: Target,
    narrow: crate::resolve_file::Narrow,
    names: Vec<String>,
}

impl Lookup {
    /// `args` as a handle then names, or with `manifest`, names alone; the
    /// store a handle's folder lies in, else `--store`'s.
    fn open(
        out: &crate::out::Out,
        store_flag: Option<&Path>,
        manifest: Option<&Path>,
        narrow: crate::resolve_file::Narrow,
        args: &[String],
    ) -> Lookup {
        if let Some(k) = &narrow.kind {
            if crate::result_files::contract(k).is_none() {
                out.fail(format!("error: --kind {k}: not a kind of result camdl stores"));
            }
        }
        let (target, names, root, shown) = match manifest {
            Some(m) => {
                let read = crate::output_manifest::OutputManifest::read(m).unwrap_or_else(|e| out.fail(format!("error: {e}")));
                let root = store_root(store_flag);
                (Target::Manifest { path: m.to_path_buf(), manifest: read }, args.to_vec(), root, m.display().to_string())
            }
            None => {
                let text = &args[0];
                let handle: store::Handle = text
                    .parse()
                    .unwrap_or_else(|_| out.fail(format!("error: `{text}` is not a handle: a run id or a prefix of one, @LABEL, @LABEL~N or a result's folder")));
                (Target::Handle(handle), args[1..].to_vec(), store_root_for(store_flag, text), text.clone())
            }
        };
        let store = crate::camdl_store::open_existing(&root).unwrap_or_else(|| match &target {
            Target::Handle(_) => out.fail(format!("error: no run matches '{shown}' in the run store {}", root.display())),
            Target::Manifest { .. } => out.fail(format!(
                "error: --manifest {shown}: no run store at {}, where the manifest's results are looked up; pass --store \
                 or set CAMDL_STORE",
                root.display()
            )),
        });
        Lookup { store, target, narrow, names }
    }

    fn source(&self) -> crate::resolve_file::Source<'_> {
        match &self.target {
            Target::Handle(h) => crate::resolve_file::Source::Handle(h),
            Target::Manifest { path, manifest } => crate::resolve_file::Source::Manifest { manifest, path },
        }
    }

    fn label(&self) -> Option<&store::Handle> {
        match &self.target {
            Target::Handle(h @ store::Handle::Label { .. }) => Some(h),
            _ => None,
        }
    }

    /// For a label, the line on stderr saying which result it named:
    /// `@pred → prediction 263942 (<folder>)`. A fit run is a `run`, as every
    /// reader of a fit run says it (`fit_store::say_resolution`).
    fn say_resolution(&self, located: &store::Located) {
        if let Some(h) = self.label() {
            let kind = match located.kind() {
                k if k == <crate::fit_store::FitRun as store::Artifact>::KIND => "run",
                k => k,
            };
            crate::out::detached().say(format_args!(
                "{h} → {kind} {} ({})",
                &located.run_id().to_hex()[..6],
                crate::fit::report::display_path(located.folder())
            ));
        }
    }

    /// A refusal, as the command's error, exiting with the refusal's status
    /// ([`crate::resolve_file::FileRefusal::exit_code`]): a handle that names
    /// nothing in the store is `no run matches`.
    fn fail(&self, out: &crate::out::Out, e: crate::resolve_file::FileRefusal) -> ! {
        if let (crate::resolve_file::FileRefusal::NoMatch, Target::Handle(h)) = (&e, &self.target) {
            out.fail(format!("error: no run matches '{h}' in the run store {}", self.store.root().display()));
        }
        let code = e.exit_code();
        out.error(crate::out::CommandError { code, message: format!("error: {e}"), hint: None }).exit()
    }
}

/// A usage error that argument parsing cannot express, with clap's exit
/// status 2.
fn usage(message: &str) -> ! {
    clap::Error::raw(clap::error::ErrorKind::ArgumentConflict, format!("{message}\n")).exit()
}

/// The label of the newest invocation with a `--label` that listed `run_id`
/// ([`store::Listings::labelled`]).
fn latest_label(store: &store::Store, run_id: store::Key) -> Option<String> {
    let listings = store.listings(&store::Handle::IdPrefix(run_id.to_hex())).ok()?;
    listings.labelled()?.label.as_ref().map(|l| l.as_str().to_string())
}

// ── cmd_show ─────────────────────────────────────────────────────────────────

pub fn cmd_show(a: &crate::args::ShowArgs) {
    let out = crate::out::Out::new(false);
    let json = show_json(&out, a.format.as_deref());
    let lookup = Lookup::open(&out, a.store.as_deref(), None, Default::default(), std::slice::from_ref(&a.target));
    let c = crate::resolve_file::resolve_result(&lookup.store, &lookup.source(), &lookup.narrow)
        .unwrap_or_else(|e| lookup.fail(&out, e));
    lookup.say_resolution(&c.located);
    show_result(&out, &lookup.store, &c, json);
}

/// `camdl show` of one located result: its kind's own fields, and its files.
fn show_result(out: &crate::out::Out, store: &store::Store, c: &crate::resolve_file::Candidate, json: bool) {
    use store::Artifact;
    use crate::{fit_store as fs, pfilter_store as ps, profile_store as pr, sim_store as ss};
    let located = &c.located;
    let fail = |e: store::Error| -> ! { out.fail(format!("error: {e}")) };
    let cwd = std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
    let rel_path = pathdiff_str(located.folder(), &cwd);
    let created = std::fs::metadata(located.folder()).and_then(|m| m.modified()).unwrap_or(SystemTime::UNIX_EPOCH);
    let label = || latest_label(store, located.run_id());
    let attributes_as = |what: &str| -> serde_json::Value {
        if c.attributes.is_null() {
            out.fail(format!("error: no invocation record gives the {what} {} its attributes", located.run_id().short12()));
        }
        c.attributes.clone()
    };
    match located.kind() {
        k if k == ps::Pfilter::KIND => {
            let stored = store.stored_as::<ps::Pfilter>(located).unwrap_or_else(|e| fail(e));
            let attributes = serde_json::from_value(attributes_as("filter run")).unwrap_or_else(|e| out.fail(format!("error: {e}")));
            let row = PfilterRow { stored, attributes, label: label(), rel_path, created };
            if json {
                out.report_line(format_args!("{}", serde_json::to_string_pretty(&row.json()).unwrap_or_default()));
            } else {
                show_stored_pfilter(out, &row);
            }
        }
        k if k == ss::Sim::KIND => {
            let stored = store.stored_as::<ss::Sim>(located).unwrap_or_else(|e| fail(e));
            let attributes = serde_json::from_value(attributes_as("cell")).unwrap_or_else(|e| out.fail(format!("error: {e}")));
            let row = StoredSimRow { stored, attributes, label: label(), rel_path, created };
            if json {
                out.report_line(format_args!("{}", serde_json::to_string_pretty(&row.json()).unwrap_or_default()));
            } else {
                show_stored_sim(out, &row);
            }
        }
        k if k == pr::ProfilePoint::KIND => {
            let point = store.stored_as::<pr::ProfilePoint>(located).unwrap_or_else(|e| fail(e));
            if json {
                let value = serde_json::json!({
                    "kind": pr::ProfilePoint::KIND,
                    "run_id": point.id().key().to_hex(),
                    "path": rel_path,
                    "identity": point.identity(),
                });
                out.report_line(format_args!("{}", serde_json::to_string_pretty(&value).unwrap_or_default()));
            } else {
                pr::show_point(out, &point, &rel_path);
            }
        }
        k if [fs::FitRun::KIND, fs::FitProblem::KIND, fs::Prediction::KIND, fs::Score::KIND].contains(&k) => {
            let found = match k {
                k if k == fs::FitRun::KIND => {
                    let s = store.stored_as::<fs::FitRun>(located).unwrap_or_else(|e| fail(e));
                    let extras = RunExtras::of(store, &s);
                    StoredFitResult::Run(s, extras)
                }
                k if k == fs::FitProblem::KIND => StoredFitResult::Problem(store.stored_as(located).unwrap_or_else(|e| fail(e))),
                k if k == fs::Prediction::KIND => StoredFitResult::Prediction(store.stored_as(located).unwrap_or_else(|e| fail(e))),
                _ => StoredFitResult::Score(store.stored_as(located).unwrap_or_else(|e| fail(e))),
            };
            if json {
                out.report_line(format_args!("{}", serde_json::to_string_pretty(&found.json()).unwrap_or_default()));
            } else {
                found.show(out);
            }
        }
        _ => show_any(out, c, &rel_path, json),
    }
}

/// `camdl show` of a result of a kind with no fields of its own to show,
/// such as a content artifact or a by-product: where it is, what it is a
/// by-product of, and its files by logical and stored name.
fn show_any(out: &crate::out::Out, c: &crate::resolve_file::Candidate, rel_path: &str, json: bool) {
    let located = &c.located;
    let files = c.file_names();
    if json {
        let value = serde_json::json!({
            "kind": located.kind(),
            "run_id": located.run_id().to_hex(),
            "upstream": located.upstream().map(|k| k.to_hex()),
            "path": rel_path,
            "attributes": c.attributes,
            "files": files.iter().map(|(l, s)| serde_json::json!({ "name": l, "file": s })).collect::<Vec<_>>(),
        });
        out.report_line(format_args!("{}", serde_json::to_string_pretty(&value).unwrap_or_default()));
        return;
    }
    let field = |name: &str, value: &dyn std::fmt::Display| {
        out.report_line(format_args!("{}", style::stdout().paint(name, Style::Dim)));
        out.report_line(format_args!("  {value}"));
    };
    field("path", &style::stdout().paint(rel_path, Style::Code));
    field("kind", &located.kind());
    field("run_id", &style::stdout().paint(&located.run_id().to_hex(), Style::Dim));
    if let Some(up) = located.upstream() {
        field("by-product of", &up.to_hex());
    }
    out.report_line(format_args!("{}", style::stdout().paint("files", Style::Dim)));
    for (logical, stored) in files {
        out.report_line(format_args!("  {logical:<24} {stored}"));
    }
}

// ── cmd_cat ──────────────────────────────────────────────────────────────────

pub fn cmd_cat(a: &crate::args::CatArgs) {
    use crate::resolve_file as rf;
    use std::io::Write as _;
    let out = crate::out::Out::new(false);
    let selection = crate::chain_selection::ChainSelection::from_flags(a.chains.as_deref(), a.exclude_chains.as_deref())
        .unwrap_or_else(|e| out.fail(format!("error: {e}")));
    let narrow = rf::Narrow { kind: a.kind.clone(), run: a.run.clone() };
    let lookup = Lookup::open(&out, a.store.as_deref(), a.manifest.as_deref(), narrow, &a.args);
    if lookup.names.len() > 1 {
        usage("`camdl cat --manifest FILE` takes one NAME");
    }
    let name = lookup.names.first();
    let source = lookup.source();
    if a.each {
        let Some(name) = name else { usage("--each takes the NAME of the file to print from each result") };
        if !source.lists_results() {
            out.fail(format!(
                "error: --each takes a manifest or a label, which list several results; `{}` names one result",
                a.args[0]
            ));
        }
        let table = rf::resolve_each(&lookup.store, &source, &lookup.narrow, name)
            .and_then(|files| rf::cat_table(&files))
            .unwrap_or_else(|e| lookup.fail(&out, e));
        let _ = out.data().write_all(&table);
        return;
    }
    if let Some(sel) = &selection {
        if name.is_none() {
            out.fail(format!("error: {} selects rows of a fit run's draws; name the file, `draws`", sel.flag()));
        }
    }
    // A label named alone: the table its invocation renders, for a
    // `pfilter --sweep-file` design, a profile or a `simulate`.
    if let (Some(label), None, None, true) = (lookup.label(), name, &selection, lookup.narrow.kind.is_none() && lookup.narrow.run.is_none()) {
        let root = lookup.store.root().to_path_buf();
        let text = label.to_string();
        let table = stored_design_table(&root, &text)
            .or_else(|| crate::profile_store::invocation_table(&root, &text))
            .or_else(|| stored_sim_table(&root, &text));
        if let Some(table) = table {
            let table = table.unwrap_or_else(|e| out.fail(format!("error: {e}")));
            let _ = out.data().write_all(&table);
            return;
        }
    }
    let file = match name {
        Some(n) => rf::resolve_file(&lookup.store, &source, &lookup.narrow, n),
        None => rf::default_file(&lookup.store, &source, &lookup.narrow),
    }
    .unwrap_or_else(|e| lookup.fail(&out, e));
    lookup.say_resolution(&file.located);
    let bytes = file.read().unwrap_or_else(|e| lookup.fail(&out, e));
    let bytes = match &selection {
        None => bytes,
        Some(sel) => {
            if file.located.kind() != <crate::fit_store::FitRun as store::Artifact>::KIND || file.stored != "draws.tsv" {
                out.fail(format!("error: {} selects rows by chain, and applies to a fit run's draws.tsv", sel.flag()));
            }
            select_chain_rows(&out, &file.stored, sel, &bytes).unwrap_or_else(|e| out.fail(format!("error: {e}"))).into_bytes()
        }
    };
    let _ = out.data().write_all(&bytes);
}

// ── cmd_path ─────────────────────────────────────────────────────────────────

pub fn cmd_path(a: &crate::args::PathArgs) {
    use crate::resolve_file as rf;
    let out = crate::out::Out::new(false);
    let narrow = rf::Narrow { kind: a.kind.clone(), run: a.run.clone() };
    let lookup = Lookup::open(&out, a.store.as_deref(), a.manifest.as_deref(), narrow, &a.args);
    let source = lookup.source();
    if a.all {
        if !lookup.names.is_empty() {
            usage("--all prints every file of one result, and takes no NAME");
        }
        let files = rf::every_file(&lookup.store, &source, &lookup.narrow).unwrap_or_else(|e| lookup.fail(&out, e));
        if let Some(f) = files.first() {
            lookup.say_resolution(&f.located);
        }
        for f in files {
            out.data_line(format_args!("{}\t{}", f.logical, f.path.display()));
        }
        return;
    }
    if lookup.names.is_empty() {
        usage("name the file to print the path of, or pass --all");
    }
    if a.each {
        let [name] = &lookup.names[..] else { usage("--each takes one NAME") };
        if !source.lists_results() {
            out.fail(format!(
                "error: --each takes a manifest or a label, which list several results; `{}` names one result",
                a.args[0]
            ));
        }
        let table = rf::resolve_each(&lookup.store, &source, &lookup.narrow, name)
            .and_then(|files| rf::path_table(&files))
            .unwrap_or_else(|e| lookup.fail(&out, e));
        out.data_raw(table);
        return;
    }
    let files: Vec<rf::ResolvedFile> = lookup
        .names
        .iter()
        .map(|n| rf::resolve_file(&lookup.store, &source, &lookup.narrow, n).unwrap_or_else(|e| lookup.fail(&out, e)))
        .collect();
    let mut said = std::collections::BTreeSet::new();
    for f in &files {
        if said.insert(f.located.run_id()) {
            lookup.say_resolution(&f.located);
        }
    }
    for f in files {
        out.data_line(f.path.display());
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
            Cell::new(&r.attributes.model),
            Cell::new(crate::pfilter_store::config_label(id.config.particles, id.config.replicates, id.config.dt.get()))
                .add_attribute(comfy_table::Attribute::Dim),
            Cell::new(r.attributes.seed),
            Cell::new(loglik).fg(comfy_table::Color::Magenta),
            Cell::new(&r.rel_path).fg(comfy_table::Color::Cyan),
        ]);
    }
    out.report_line(format_args!("{table}"));
}


/// 8-char hash prefix cell — what `camdl show <hash>` accepts.
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

    /// `camdl list` builds every kind's rows from one opening of the store,
    /// so it reads the invocation records once.
    #[test]
    fn list_reads_the_invocation_records_once() {
        let (tmp, _) = crate::camdl_store::model_source::store_of_four();
        let store = crate::camdl_store::open_existing(&tmp.path().join("st")).expect("the store");
        assert_eq!(store.directory_reads(), 0);
        let rows = rows(&store, KindFilter::All);
        assert!(!rows.fits.is_empty() && !rows.profiles.is_empty() && !rows.sims.is_empty());
        assert_eq!(store.directory_reads(), 1, "one read of the records answers every row");
    }

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
