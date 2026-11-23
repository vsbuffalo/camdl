//! `camdl simulate` on the run store: the identity of one simulation cell, the
//! stored result and its by-products, and the renderings its exports write
//! (`docs/dev/proposals/2026-11-20-run-store-api.md`, "Worked example:
//! `camdl simulate`" and "Layouts, kind by kind").
//!
//! A cell ([`Sim`]) is written atomically. It stores `traj.tsv`, and
//! `reactive_log.tsv` when a reactive policy was active. A cell whose engine
//! stopped with an error stores `failure.toml` in place of the trajectory: the
//! error is a function of the cell's identity, which holds its seed. The
//! observations drawn beside the trajectory ([`Obs`]), the event log
//! ([`EventLog`]) and the generated quantities ([`SimQuantities`]) are
//! by-products, each inside the cell under its kind. A by-product whose
//! computation is refused for its cell stores its own `failure.toml`, and the
//! cell keeps its trajectory.
//!
//! ```text
//! <build>/sim/<model>-…/<backend>-dt<dt>-…/<params>-…/<scenario>-…/seed_<seed>[-r<n>]-…/
//!   identity.json  run.json  traj.tsv
//!   obs/<streams>-…/          <stream>.tsv …
//!   event_log/event_log-…/    event_log.tsv
//!   sim_quantities/quantities-…/  quantities.json
//! ```
//!
//! The combined trajectory table (`-o`, `--stdout`, `camdl cat @L`), the
//! combined observation tables (`--obs`, `--obs-dir`) and the quantity tables
//! (`--quantities-out`) are renderings of the stored files and the invocation's
//! labels; none is stored.

// The store's error carries the paths and owners a person acts on, and is
// returned once per export or result.
#![allow(clippy::result_large_err)]

use std::collections::BTreeSet;
use std::io::Write;

use store::{Artifact, ByProduct, ByProductRendering, Dep, ExportSink, Finite, Key, Kinds, Placed, Primary, Rendering, Stored, WriteMode};

use crate::camdl_store::{camdl_groups, ActiveInterventions, CompiledModel, ProcessSeed, Quantities, ResolvedParams};

/// Compact `dt` rendering for a settings label (`1`, `0.5`, …).
pub(crate) fn fmt_dt(dt: f64) -> String {
    if (dt.round() - dt).abs() < 1e-9 {
        format!("{}", dt.round() as i64)
    } else {
        format!("{}", dt)
    }
}

/// The simulation backend.
#[derive(store::Identity, Clone, Copy, Debug, PartialEq, Eq)]
#[identity(tag = "sim_backend", version = 1)]
pub(crate) enum Backend {
    Gillespie,
    ChainBinomial,
    Ode,
}

impl Backend {
    pub fn of(b: crate::args::types::ForwardBackend) -> Self {
        use crate::args::types::ForwardBackend as F;
        match b {
            F::Gillespie => Backend::Gillespie,
            F::ChainBinomial => Backend::ChainBinomial,
            F::Ode => Backend::Ode,
        }
    }

    fn name(self) -> &'static str {
        match self {
            Backend::Gillespie => "gillespie",
            Backend::ChainBinomial => "chain_binomial",
            Backend::Ode => "ode",
        }
    }
}

/// The paired `(θ_i, X_i(T))` origin ensemble a `--init-state fit` run
/// forecasts from, as its identity holds it: the origin time and every
/// selected draw's restored state, in selection order. Content, not the fit's
/// run id, so two selections that restore different states never share a key.
/// θ is not here: each draw's parameter values are a cell's own parameters.
#[derive(store::Identity, Clone, Debug)]
#[identity(tag = "init_state_ensemble", version = 1)]
pub struct InitStateEnsemble {
    pub origin_t: store::Finite,
    pub rows: Vec<InitStateDraw>,
}

/// One selected draw of an [`InitStateEnsemble`]: its chain and draw, and
/// its integer and real compartment values at the origin, in the model's
/// compartment order.
#[derive(store::Identity, Clone, Debug)]
#[identity(tag = "init_state_draw", version = 1)]
pub struct InitStateDraw {
    pub chain: u64,
    pub draw: u64,
    pub counts: Vec<i64>,
    pub reals: Vec<store::Finite>,
}

/// The forecast-origin state a cell restores (`--init-state`): the content
/// digest of the state ensemble and the row the cell restores.
#[derive(store::Identity, Clone, Debug)]
#[identity(tag = "init_state_row", version = 1)]
pub(crate) struct InitStateRow {
    pub ensemble: Key,
    pub row: u64,
}

/// Everything besides the model, the parameters, the scenario and the seed
/// that can change a cell's stored files: the backend and step, the cell's
/// window and restored state, the evaluator policies and the column view.
#[derive(store::Identity, Clone, Debug)]
#[identity(tag = "sim_config", version = 1)]
pub(crate) struct SimConfig {
    pub backend: Backend,
    pub dt: Finite,
    /// The cell's effective start and end: a scenario's own horizon, `--to`
    /// and a restart's origin move them.
    pub t_start: Finite,
    pub t_end: Finite,
    pub init_state: Option<InitStateRow>,
    /// A collapsing rate is 0 instead of an error.
    pub allow_degenerate_rates: bool,
    /// A forcing evaluated outside its range holds its end value.
    pub allow_forcing_extrapolation: bool,
    /// `--no-flows` and `--columns`: which trajectory columns are stored.
    pub no_flows: bool,
    pub columns: BTreeSet<String>,
}

impl SimConfig {
    /// The settings group's label, `<backend>-dt<dt>`.
    pub fn label(&self) -> String {
        format!("{}-dt{}", self.backend.name(), fmt_dt(self.dt.get()))
    }
}

/// The settings label of a stored cell's identity record.
pub(crate) fn config_label(config: &SimConfigRecord) -> String {
    let backend = match config.backend {
        BackendRecord::Gillespie => Backend::Gillespie,
        BackendRecord::ChainBinomial => Backend::ChainBinomial,
        BackendRecord::Ode => Backend::Ode,
    };
    format!("{}-dt{}", backend.name(), fmt_dt(config.dt.get()))
}

/// One simulation cell.
#[derive(store::Identity, Clone, Debug)]
#[identity(tag = "sim", version = 1)]
pub(crate) struct SimIdentity {
    pub model: Dep<CompiledModel>,
    pub config: SimConfig,
    /// Every parameter's value after all sources: the model, `--params`, a
    /// draw or sweep point, the scenario, `--param` and `--param-vec`.
    pub params: ResolvedParams,
    /// The interventions the cell's scenario selection keeps.
    pub scenario: ActiveInterventions,
    pub seed: ProcessSeed,
}

/// Where a cell sits on the parameter-point axis, as the combined tables
/// show it.
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(tag = "axis", rename_all = "snake_case")]
pub(crate) enum PointLabel {
    /// One parameter point.
    None,
    /// `--draws`: the draw's 0-based index.
    Draw { index: usize },
    /// `--sweep`, `--sweep-file`: the point's id and its value of each swept
    /// parameter, by name.
    Sweep { point_id: usize, values: Vec<(String, f64)> },
}

/// A cell's labels: what a user recognises it by, never hashed.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub(crate) struct CellLabels {
    /// The model file's stem.
    pub model: String,
    /// `base`, or a sweep point's or draw row's values at those the cell ran.
    pub params: String,
    /// The scenario's name; `(none)` for the implicit arm.
    pub scenario: String,
    /// The cell's position in the invocation, from 0: the combined table's
    /// `replicate` column.
    pub replicate: usize,
    /// The seed the user typed (`--seed`, or the listed seed under `--seeds`).
    pub seed: u64,
    /// Which replicate of that seed this cell is, from 0.
    pub seed_replicate: usize,
    /// The seed the cell ran under: passed back to `--seeds`, it reproduces
    /// the cell.
    pub process_seed: u64,
    pub point: PointLabel,
}

impl CellLabels {
    /// `seed_<seed typed>`, with `-r<n>` for every replicate of that seed
    /// after the first.
    pub fn seed_label(&self) -> String {
        match self.seed_replicate {
            0 => format!("seed_{}", self.seed),
            n => format!("seed_{}-r{n}", self.seed),
        }
    }
}

/// One simulation cell, written atomically.
pub(crate) struct Sim;

impl Artifact for Sim {
    const KIND: &'static str = "sim";
    const WRITE: WriteMode = WriteMode::Atomic;
    type Identity = SimIdentity;
    type Labels = CellLabels;
}

impl Primary for Sim {
    fn layout(id: &SimIdentity, labels: &CellLabels) -> Placed {
        camdl_groups(&labels.model, &id.model, &id.config.label(), &id.config, Some((labels.params.as_str(), &id.params)))
            .group(&labels.scenario, &id.scenario)
            .result(&labels.seed_label())
    }
}

/// A cell's result files.
pub(crate) const TRAJ_FILE: &str = "traj.tsv";
pub(crate) const REACTIVE_LOG_FILE: &str = "reactive_log.tsv";
pub(crate) const FAILURE_FILE: &str = "failure.toml";
/// Where each parameter value came from, the resolution's notices and the
/// forcings held: a provenance file, never compared on a recompute.
pub(crate) const PROVENANCE_FILE: &str = "provenance.json";

/// Observations drawn while the trajectory is simulated.
#[derive(store::Identity, Clone, Debug)]
#[identity(tag = "obs", version = 1)]
pub(crate) struct ObsIdentity {
    pub sim: SimIdentityRecord,
    /// The digest of the cell's observation blocks, the `--emit-every`
    /// override and the file format (`crate::cas_sink::obs_subtree_hash`).
    pub observation: String,
    /// The streams, in declaration order: each is one file.
    pub streams: Vec<String>,
    /// The observation RNG's seed.
    pub seed: u64,
}

pub(crate) struct Obs;

impl Artifact for Obs {
    const KIND: &'static str = "obs";
    const WRITE: WriteMode = WriteMode::Atomic;
    type Identity = ObsIdentity;
    type Labels = ();
}

impl ByProduct for Obs {
    type Of = Sim;
    fn upstream(id: &ObsIdentity) -> &SimIdentityRecord {
        &id.sim
    }
    fn label(id: &ObsIdentity, _labels: &()) -> String {
        id.streams.join("+")
    }
}

/// The event log written while the trajectory is simulated. The recorder
/// draws nothing, so it has no settings of its own.
#[derive(store::Identity, Clone, Debug)]
#[identity(tag = "event_log", version = 1)]
pub(crate) struct EventLogIdentity {
    pub sim: SimIdentityRecord,
}

pub(crate) struct EventLog;

impl EventLog {
    pub const FILE: &'static str = "event_log.tsv";
}

impl Artifact for EventLog {
    const KIND: &'static str = "event_log";
    const WRITE: WriteMode = WriteMode::Atomic;
    type Identity = EventLogIdentity;
    type Labels = ();
}

impl ByProduct for EventLog {
    type Of = Sim;
    fn upstream(id: &EventLogIdentity) -> &SimIdentityRecord {
        &id.sim
    }
    fn label(_id: &EventLogIdentity, _labels: &()) -> String {
        "event_log".into()
    }
}

/// One cell's generated quantities, evaluated from the in-memory trajectory
/// and the cell's observation draws, at full precision.
#[derive(store::Identity, Clone, Debug)]
#[identity(tag = "sim_quantities", version = 1)]
pub(crate) struct SimQuantitiesIdentity {
    pub sim: SimIdentityRecord,
    /// The model's own block, or the `--quantities FILE` vocabulary.
    pub quantities: Dep<Quantities>,
    /// The `--emit-every` override an `observations.<stream>` quantity
    /// reduces under.
    pub emit_every: Option<String>,
    /// The observation RNG's seed.
    pub seed: u64,
}

pub(crate) struct SimQuantities;

impl SimQuantities {
    pub const FILE: &'static str = "quantities.json";
}

impl Artifact for SimQuantities {
    const KIND: &'static str = "sim_quantities";
    const WRITE: WriteMode = WriteMode::Atomic;
    type Identity = SimQuantitiesIdentity;
    type Labels = ();
}

impl ByProduct for SimQuantities {
    type Of = Sim;
    fn upstream(id: &SimQuantitiesIdentity) -> &SimIdentityRecord {
        &id.sim
    }
    fn label(_id: &SimQuantitiesIdentity, _labels: &()) -> String {
        "quantities".into()
    }
}

/// Adds every kind of this module to `kinds`.
pub(crate) fn register(kinds: Kinds) -> Kinds {
    kinds.register::<Sim>().register::<Obs>().register::<EventLog>().register::<SimQuantities>()
}

// ── Stored failures ──────────────────────────────────────────────────────────

/// What `failure.toml` holds: which computation stopped, and why.
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub(crate) struct Failure {
    /// `engine`, `observations`, `quantities` or `event log`.
    pub stage: String,
    pub reason: String,
}

impl Failure {
    pub fn new(stage: &str, reason: &str) -> Self {
        Failure { stage: stage.into(), reason: reason.strip_prefix("error: ").unwrap_or(reason).to_string() }
    }

    pub fn to_bytes(&self) -> Vec<u8> {
        format!("stage = {}\nreason = {}\n", toml::Value::String(self.stage.clone()), toml::Value::String(self.reason.clone()))
            .into_bytes()
    }

    pub fn from_bytes(bytes: &[u8]) -> Result<Self, String> {
        let text = std::str::from_utf8(bytes).map_err(|e| format!("{FAILURE_FILE}: {e}"))?;
        toml::from_str(text).map_err(|e| format!("{FAILURE_FILE}: {e}"))
    }

    /// The stored failure of a result, if it holds one.
    pub fn of<A: Artifact>(stored: &Stored<A>) -> Result<Option<Failure>, String> {
        if !stored.result_files().contains(&FAILURE_FILE) {
            return Ok(None);
        }
        crate::camdl_store::read_stored(stored, FAILURE_FILE).and_then(|b| Failure::from_bytes(&b)).map(Some)
    }
}

/// The per-cell quantity values, as `quantities.json` holds them: the
/// output times and each quantity's value, every number written in the
/// shortest form that reads back to the same `f64` (`NaN` and `inf`
/// included), so the banding over stored cells is the banding over the
/// cells as computed.
#[derive(serde::Serialize, serde::Deserialize)]
pub(crate) struct CellQuantities {
    pub times: Vec<String>,
    pub results: Vec<StoredQuantity>,
}

#[derive(serde::Serialize, serde::Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub(crate) enum StoredQuantity {
    Series { values: Vec<String> },
    Value { value: String },
    Censored,
}

impl CellQuantities {
    pub fn new(times: &[f64], results: &[sim::quantity::QuantityResult]) -> Self {
        use sim::quantity::{QuantityDrawValue as V, QuantityResult as R};
        let s = |x: &f64| format!("{x:?}");
        CellQuantities {
            times: times.iter().map(s).collect(),
            results: results
                .iter()
                .map(|r| match r {
                    R::Series(v) => StoredQuantity::Series { values: v.iter().map(s).collect() },
                    R::Scalar(V::Value(x)) => StoredQuantity::Value { value: s(x) },
                    R::Scalar(V::Censored) => StoredQuantity::Censored,
                })
                .collect(),
        }
    }

    pub fn parts(&self) -> Result<(Vec<f64>, Vec<sim::quantity::QuantityResult>), String> {
        use sim::quantity::{QuantityDrawValue as V, QuantityResult as R};
        let f = |x: &String| x.parse::<f64>().map_err(|_| format!("{}: `{x}` is not a number", SimQuantities::FILE));
        let times = self.times.iter().map(f).collect::<Result<Vec<_>, _>>()?;
        let results = self
            .results
            .iter()
            .map(|r| {
                Ok(match r {
                    StoredQuantity::Series { values } => R::Series(values.iter().map(f).collect::<Result<_, _>>()?),
                    StoredQuantity::Value { value } => R::Scalar(V::Value(f(value)?)),
                    StoredQuantity::Censored => R::Scalar(V::Censored),
                })
            })
            .collect::<Result<Vec<_>, String>>()?;
        Ok((times, results))
    }
}

// ── The combined tables ──────────────────────────────────────────────────────

fn parse(why: String) -> store::Error {
    store::Error::Parse { what: "a stored simulation".into(), why }
}

fn io(e: std::io::Error) -> store::Error {
    store::Error::Io { op: "write", path: "a simulation export".into(), source: e }
}

/// Refuses a combined table over members of which any stored a failure,
/// naming each such cell: leaving its rows out would write a table that looks
/// complete and is not.
fn refuse_failures<'a>(
    members: impl Iterator<Item = (&'a Stored<Sim>, &'a CellLabels, Option<(Failure, std::path::PathBuf)>)>,
) -> store::Result<()> {
    let mut failed = Vec::new();
    for (stored, labels, by_product) in members {
        let own = Failure::of(stored).map_err(parse)?.map(|f| (f, stored.folder().join(FAILURE_FILE)));
        if let Some((f, path)) = own.or(by_product) {
            failed.push(format!("{}: the {} stopped: {} ({})", cell_name(labels), f.stage, f.reason, path.display()));
        }
    }
    if failed.is_empty() {
        return Ok(());
    }
    Err(store::Error::RenderingRefused(format!(
        "the table is not written, because {} cell(s) stored a failure; each cell stays stored:\n  {}",
        failed.len(),
        failed.join("\n  ")
    )))
}

/// The stored failure of a by-product, with its `failure.toml`.
fn by_product_failure<B: Artifact>(stored: &Option<Stored<B>>) -> store::Result<Option<(Failure, std::path::PathBuf)>> {
    match stored {
        Some(s) => Ok(Failure::of(s).map_err(parse)?.map(|f| (f, s.folder().join(FAILURE_FILE)))),
        None => Ok(None),
    }
}

/// A cell as a person names it: its scenario and the seed typed.
pub(crate) fn cell_name(labels: &CellLabels) -> String {
    format!("scenario={} {}", labels.scenario, labels.seed_label())
}

/// The coordinate columns of a combined table: `replicate` and `scenario`,
/// then the point axis's columns, then `seed`. The same columns for one cell
/// and for many.
struct Coords {
    sweep: Option<Vec<String>>,
    draws: bool,
}

impl Coords {
    fn of(members: &[&CellLabels]) -> store::Result<Coords> {
        let first = members.first().map(|l| &l.point);
        let coords = match first {
            Some(PointLabel::Sweep { values, .. }) => Coords { sweep: Some(values.iter().map(|(k, _)| k.clone()).collect()), draws: false },
            Some(PointLabel::Draw { .. }) => Coords { sweep: None, draws: true },
            _ => Coords { sweep: None, draws: false },
        };
        Ok(coords)
    }

    fn header(&self, out: &mut dyn Write) -> std::io::Result<()> {
        write!(out, "replicate\tscenario\t")?;
        if let Some(params) = &self.sweep {
            write!(out, "point_id\t")?;
            for p in params {
                write!(out, "sweep:{p}\t")?;
            }
        }
        if self.draws {
            write!(out, "draw\t")?;
        }
        write!(out, "seed\t")
    }

    fn cells(&self, out: &mut dyn Write, l: &CellLabels) -> store::Result<()> {
        write!(out, "{}\t{}\t", l.replicate, l.scenario).map_err(io)?;
        match (&self.sweep, &l.point) {
            (Some(params), PointLabel::Sweep { point_id, values }) => {
                write!(out, "{point_id}\t").map_err(io)?;
                for p in params {
                    let v = values.iter().find(|(k, _)| k == p).map(|(_, v)| *v).ok_or_else(|| {
                        parse(format!("cell {} carries no value for swept parameter `{p}`", l.replicate))
                    })?;
                    write!(out, "{v}\t").map_err(io)?;
                }
            }
            (None, _) => {}
            (Some(_), _) => return Err(parse(format!("cell {} is not a point of the sweep", l.replicate))),
        }
        if self.draws {
            match &l.point {
                PointLabel::Draw { index } => write!(out, "{index}\t").map_err(io)?,
                _ => return Err(parse(format!("cell {} is not a draw", l.replicate))),
            }
        }
        write!(out, "{}\t", l.process_seed).map_err(io)
    }
}

/// The calendar a rendering writes date columns under: the model's origin
/// and time unit.
#[derive(Clone, Debug)]
pub(crate) struct Dates {
    pub origin: String,
    pub time_unit: String,
}

/// The combined trajectory table across an invocation's cells: `-o`,
/// `--stdout` and `camdl cat @L`. Each cell's stored `traj.tsv` rows,
/// prefixed with its labels as coordinate columns, in the order the cells
/// were asked for, after one version line and one header.
pub(crate) struct CombinedTable;

pub(crate) struct CombinedTableOptions {
    /// `--dates`: date columns after the three time columns.
    pub dates: Option<Dates>,
}

impl Rendering for CombinedTable {
    type Of = Sim;
    type Options = CombinedTableOptions;
    fn render(members: &[store::Member<Sim>], options: &CombinedTableOptions, out: &mut dyn Write) -> store::Result<()> {
        refuse_failures(members.iter().map(|(s, l)| (s, l, None)))?;
        let coords = Coords::of(&members.iter().map(|(_, l)| l).collect::<Vec<_>>())?;
        let mut header_written = false;
        for (stored, labels) in members {
            let traj = String::from_utf8(crate::camdl_store::read_stored(stored, TRAJ_FILE).map_err(parse)?)
                .map_err(|e| parse(e.to_string()))?;
            let mut lines = traj.lines();
            let header = lines.next().ok_or_else(|| parse("a stored traj.tsv is empty".into()))?;
            let rest_header = header
                .strip_prefix(crate::util::TRAJ_TIME_HEADER)
                .ok_or_else(|| parse(format!("a stored traj.tsv header `{header}` is not the trajectory format")))?;
            if !header_written {
                writeln!(out, "# {}", crate::version::VERSION_SHORT).map_err(io)?;
                coords.header(out).map_err(io)?;
                write!(out, "{}", crate::util::TRAJ_TIME_HEADER).map_err(io)?;
                if options.dates.is_some() {
                    write!(out, "{}", crate::util::TRAJ_DATE_HEADER).map_err(io)?;
                }
                writeln!(out, "{rest_header}").map_err(io)?;
                header_written = true;
            }
            for row in lines.filter(|l| !l.trim().is_empty()) {
                let mut it = row.splitn(4, '\t');
                let (t, start, stop) = match (it.next(), it.next(), it.next()) {
                    (Some(t), Some(a), Some(b)) => (t, a, b),
                    _ => return Err(parse(format!("a stored traj.tsv row `{row}` lacks the three time cells"))),
                };
                coords.cells(out, labels)?;
                write!(out, "{t}\t{start}\t{stop}").map_err(io)?;
                if let Some(d) = &options.dates {
                    let num = |x: &str| x.parse::<f64>().map_err(|_| parse(format!("a stored time cell `{x}` is not a number")));
                    let mut cells = Vec::new();
                    crate::util::write_traj_date_cells(&mut cells, &d.origin, &d.time_unit, num(t)?, (num(start)?, num(stop)?))
                        .map_err(parse)?;
                    out.write_all(&cells).map_err(io)?;
                }
                if let Some(rest) = it.next() {
                    write!(out, "\t{rest}").map_err(io)?;
                }
                writeln!(out).map_err(io)?;
            }
        }
        Ok(())
    }
}

/// One observation stream as the combined observation tables write it: its
/// name, the file it belongs in under `--obs-dir` and the cells that route
/// its rows back to it, and the rows its declaration assigns to the cells'
/// shared emission times.
#[derive(Clone, Debug)]
pub(crate) struct ObsStreamPlan {
    pub name: String,
    pub source: String,
    pub dims: Vec<(String, String)>,
    pub plan: crate::obs_emit::EmitPlan,
}

/// Which combined observation table to write.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum ObsLayout {
    /// `--obs FILE`: every stream in one wide table.
    OneFile,
    /// `--obs-dir DIR`: one table per observation source, under the columns
    /// the stream declared.
    PerSource,
}

pub(crate) struct ObsTableOptions {
    pub dates: Option<Dates>,
    pub layout: ObsLayout,
    /// The streams, in declaration order, planned once from the first cell's
    /// model: the cells share one observation axis.
    pub streams: Vec<ObsStreamPlan>,
}

/// The combined observation tables: each cell's stored stream files, with
/// the cell's labels as coordinate columns.
pub(crate) struct ObsTable;

impl ByProductRendering for ObsTable {
    type Of = Sim;
    type By = Obs;
    type Options = ObsTableOptions;
    fn render(members: &[(store::Member<Sim>, Option<Stored<Obs>>)], options: &ObsTableOptions, out: &mut dyn ExportSink) -> store::Result<()> {
        let mut obs_failures = Vec::with_capacity(members.len());
        for (_, obs) in members {
            obs_failures.push(by_product_failure(obs)?);
        }
        refuse_failures(members.iter().zip(obs_failures).map(|(((s, l), _), f)| (s, l, f)))?;
        let coords = Coords::of(&members.iter().map(|((_, l), _)| l).collect::<Vec<_>>())?;
        // Each member's values, per stream, as stored.
        let mut values: Vec<Vec<Vec<String>>> = Vec::with_capacity(members.len());
        for ((_, labels), obs) in members {
            let obs = obs.as_ref().ok_or_else(|| parse(format!("{} has no stored observations", cell_name(labels))))?;
            let mut per = Vec::with_capacity(options.streams.len());
            for s in &options.streams {
                let name = format!("{}.tsv", s.name);
                let text = String::from_utf8(crate::camdl_store::read_stored(obs, &name).map_err(parse)?)
                    .map_err(|e| parse(e.to_string()))?;
                let vals: Vec<String> = text
                    .lines()
                    .skip(1)
                    .filter(|l| !l.trim().is_empty())
                    .map(|l| l.rsplit('\t').next().unwrap_or("").to_string())
                    .collect();
                if vals.len() != s.plan.rows.len() {
                    return Err(parse(format!(
                        "the stored observations of stream '{}' have {} rows, and this run emits {}",
                        s.name,
                        vals.len(),
                        s.plan.rows.len()
                    )));
                }
                per.push(vals);
            }
            values.push(per);
        }
        let date = |t: f64| -> store::Result<String> {
            let d = options.dates.as_ref().expect("dated only under --dates");
            ir::caltime::internal_to_date_hires(&d.origin, t, &d.time_unit).map_err(|e| parse(e.to_string()))
        };
        match options.layout {
            ObsLayout::OneFile => {
                let w = out.file("")?;
                coords.header(w).map_err(io)?;
                write!(w, "time").map_err(io)?;
                if options.dates.is_some() {
                    write!(w, "\tdate").map_err(io)?;
                }
                for s in &options.streams {
                    write!(w, "\t{}", s.name).map_err(io)?;
                }
                writeln!(w).map_err(io)?;
                let n_times = options.streams.first().map(|s| s.plan.rows.len()).unwrap_or(0);
                for (mi, ((_, labels), _)) in members.iter().enumerate() {
                    for ti in 0..n_times {
                        coords.cells(w, labels)?;
                        let t = options.streams[0].plan.rows[ti].label;
                        write!(w, "{t}").map_err(io)?;
                        if options.dates.is_some() {
                            write!(w, "\t{}", date(t)?).map_err(io)?;
                        }
                        for si in 0..options.streams.len() {
                            write!(w, "\t{}", values[mi][si][ti]).map_err(io)?;
                        }
                        writeln!(w).map_err(io)?;
                    }
                }
            }
            ObsLayout::PerSource => {
                for group in source_groups(&options.streams) {
                    let first = &options.streams[group[0]];
                    let stem = if group.len() == 1 { &first.name } else { &first.source };
                    let w = out.file(&format!("{stem}.tsv"))?;
                    coords.header(w).map_err(io)?;
                    match &first.plan.columns {
                        crate::obs_emit::TemporalColumns::Time(t) => {
                            write!(w, "{t}").map_err(io)?;
                            if options.dates.is_some() {
                                write!(w, "\tdate").map_err(io)?;
                            }
                        }
                        crate::obs_emit::TemporalColumns::Window { start, stop } => {
                            write!(w, "{start}\t{stop}").map_err(io)?;
                            if options.dates.is_some() {
                                write!(w, "\tdate_start\tdate_stop").map_err(io)?;
                            }
                        }
                    }
                    for (header, _) in &first.dims {
                        write!(w, "\t{header}").map_err(io)?;
                    }
                    writeln!(w, "\t{}", first.plan.scored).map_err(io)?;
                    for (mi, ((_, labels), _)) in members.iter().enumerate() {
                        for ri in 0..first.plan.rows.len() {
                            for &si in &group {
                                let s = &options.streams[si];
                                let row = &s.plan.rows[ri];
                                coords.cells(w, labels)?;
                                match (&first.plan.columns, row.coverage) {
                                    (crate::obs_emit::TemporalColumns::Window { .. }, score::trace::Coverage::Interval { start, stop }) => {
                                        write!(w, "{start}\t{stop}").map_err(io)?;
                                        if options.dates.is_some() {
                                            write!(w, "\t{}\t{}", date(start)?, date(stop)?).map_err(io)?;
                                        }
                                    }
                                    _ => {
                                        write!(w, "{}", row.label).map_err(io)?;
                                        if options.dates.is_some() {
                                            write!(w, "\t{}", date(row.label)?).map_err(io)?;
                                        }
                                    }
                                }
                                for (_, level) in &s.dims {
                                    write!(w, "\t{level}").map_err(io)?;
                                }
                                writeln!(w, "\t{}", values[mi][si][ri]).map_err(io)?;
                            }
                        }
                    }
                }
            }
        }
        Ok(())
    }
}

/// The stream indices that share each `--obs-dir` file, in declaration
/// order: one group per observation source.
fn source_groups(streams: &[ObsStreamPlan]) -> Vec<Vec<usize>> {
    let mut groups: Vec<Vec<usize>> = Vec::new();
    let mut seen: Vec<&str> = Vec::new();
    for (si, s) in streams.iter().enumerate() {
        match seen.iter().position(|x| *x == s.source) {
            Some(g) => groups[g].push(si),
            None => {
                seen.push(&s.source);
                groups.push(vec![si]);
            }
        }
    }
    groups
}

/// The quantity tables `--quantities-out` writes: a value per cell for a
/// single parameter point with one replicate, else the band over each
/// (point, scenario) group's replicates, never across groups.
pub(crate) struct QuantityTable;

pub(crate) struct QuantityTableOptions {
    pub quantities: Vec<ir::quantity::Quantity>,
    pub mode: crate::quantity_output::Mode,
    /// Whether the run has a scenario axis: the `scenario` column is written
    /// only then.
    pub scenario_axis: bool,
    pub calendar: ir::caltime::CalendarAxis,
    /// The `--quantities FILE` vocabulary, which names the tables' directory.
    pub vocabulary: Option<crate::quantities_file::QuantitiesOverride>,
}

impl ByProductRendering for QuantityTable {
    type Of = Sim;
    type By = SimQuantities;
    type Options = QuantityTableOptions;
    fn render(
        members: &[(store::Member<Sim>, Option<Stored<SimQuantities>>)],
        options: &QuantityTableOptions,
        out: &mut dyn ExportSink,
    ) -> store::Result<()> {
        let mut q_failures = Vec::with_capacity(members.len());
        for (_, q) in members {
            q_failures.push(by_product_failure(q)?);
        }
        refuse_failures(members.iter().zip(q_failures).map(|(((s, l), _), f)| (s, l, f)))?;
        type Group = (Vec<Vec<sim::quantity::QuantityResult>>, Vec<f64>, Vec<(String, f64)>);
        let mut groups: indexmap::IndexMap<(usize, String), Group> = indexmap::IndexMap::new();
        for ((_, labels), q) in members {
            let q = q.as_ref().ok_or_else(|| parse(format!("{} has no stored quantities", cell_name(labels))))?;
            let stored: CellQuantities = serde_json::from_slice(&crate::camdl_store::read_stored(q, SimQuantities::FILE).map_err(parse)?)
                .map_err(|e| parse(e.to_string()))?;
            let (times, results) = stored.parts().map_err(parse)?;
            let (point, sweep) = match &labels.point {
                PointLabel::Sweep { point_id, values } => (*point_id, values.clone()),
                _ => (0, Vec::new()),
            };
            let g = groups.entry((point, labels.scenario.clone())).or_insert_with(|| (Vec::new(), times.clone(), sweep));
            if g.1.len() != times.len() {
                return Err(parse(format!(
                    "scenario '{}', point {point}: a cell has {} output times and this group's earlier cells have {}",
                    labels.scenario,
                    times.len(),
                    g.1.len()
                )));
            }
            g.0.push(results);
        }
        let mut stacked = crate::quantity_output::StackedQuantities::new(options.mode);
        for ((_, scenario), (draws, times, sweep)) in &groups {
            let coords = crate::quantity_output::DesignCoords { scenario: options.scenario_axis.then_some(scenario.as_str()), sweep };
            stacked.push_group(&options.quantities, coords, draws, times, None, None, &options.calendar).map_err(parse)?;
        }
        let rendered = stacked.finish(&options.calendar).map_err(parse)?;
        if !rendered.failures.is_empty() {
            let names: Vec<String> = rendered.failures.iter().map(|f| format!("quantity '{}': {}", f.name, f.reason())).collect();
            return Err(parse(names.join("; ")));
        }
        let sub = crate::quantities_file::quantities_dir_name(options.vocabulary.as_ref());
        for (name, content) in &rendered.files {
            out.file(&format!("{sub}/{name}.tsv"))?.write_all(content.as_bytes()).map_err(io)?;
        }
        let manifest = crate::quantities_file::stamp_provenance(&rendered.manifest, options.vocabulary.as_ref()).map_err(parse)?;
        out.file(&crate::quantities_file::quantities_manifest_name(options.vocabulary.as_ref()))?
            .write_all(manifest.as_bytes())
            .map_err(io)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests;
