//! The store half of `camdl simulate`: each cell's identity, the batch that
//! claims them, the engine run of each claimed cell under its claim, and the
//! exports rendered from what the store holds afterwards. Everything before
//! this, the resolution of the command line into a [`crate::sim_job::SimulateJob`]
//! and the run-wide [`crate::sim_config::ResolvedSimConfig`], is in
//! `main::run_simulate`.
//!
//! A cell's identity is built from its [`ResolvedCell`], the resolution the
//! engine itself runs, before anything runs; the engine runs that same value,
//! and checks that the model it runs is the stored compiled model with the
//! identity's parameters, interventions and window applied.

// The store's error carries the paths and owners a person acts on, and is
// returned once per export or result.
#![allow(clippy::result_large_err)]

use std::collections::BTreeSet;
use std::io::Write;

use store::{ByProductRun, ContentAddressed, ContentArtifact, Identity, Run};

use crate::camdl_store::{ActiveInterventions, CompiledModel, ProcessSeed, ResolvedParams};
use crate::sim_config::ResolvedCell;
use crate::sim_store::*;

/// Everything the store half needs from the resolution.
pub(crate) struct StoreRun<'a> {
    pub out: &'a crate::out::Out,
    pub job: &'a crate::sim_job::SimulateJob,
    pub sim: &'a crate::sim_config::ResolvedSimConfig,
    pub engine: &'a crate::sim_config::EngineSettings,
    /// The model file as named, for the compiled model's provenance.
    pub model_path: &'a str,
    /// The compiled IR the cells run, whose envelope is stored.
    pub ir_compiled: &'a str,
    /// The `--table` files and the replayed fit's tables, by name.
    pub table_files: &'a std::collections::HashMap<String, String>,
    pub request: store::StoreRequest,
    pub given: store::Given,
    pub execution: crate::args::SimulateInvocationExecution,
    pub lineage: Vec<store::Lineage>,
    pub wants: Wants,
    pub exports: Exports,
    pub dry_run: bool,
}

/// The by-products asked for of every cell.
pub(crate) struct Wants {
    pub obs: bool,
    pub event_log: bool,
    /// `--quantities-out`: the quantities the cells evaluate.
    pub quantities: Option<QuantitiesWant>,
    pub emit_every: Option<crate::emit_every::EmitEvery>,
}

pub(crate) struct QuantitiesWant {
    pub dir: std::path::PathBuf,
    pub mode: crate::quantity_output::Mode,
    pub scenario_axis: bool,
    pub vocabulary: Option<crate::quantities_file::QuantitiesOverride>,
}

/// The exports and their rendering options.
pub(crate) struct Exports {
    pub output: Option<std::path::PathBuf>,
    pub stdout: bool,
    pub obs_file: Option<std::path::PathBuf>,
    pub obs_dir: Option<std::path::PathBuf>,
    pub dates: bool,
    /// `--event-log PATH`; `None` for no event log or the bare flag.
    pub event_log: Option<std::path::PathBuf>,
    pub event_log_tsv: bool,
    pub event_log_format: Option<String>,
    pub reactive_log: Option<std::path::PathBuf>,
}

/// A cell planned for the batch: its spec, its resolution, its identity and
/// labels, and the by-products wanted of it.
struct Planned {
    spec: crate::engine::CellSpec,
    cell: ResolvedCell,
    identity: SimIdentity,
    labels: CellLabels,
    wanted: store::Wanted<Sim>,
}

/// What the store needs for each cell, built from the stored compiled model.
struct Shared {
    model: store::Dep<CompiledModel>,
    quantities: Option<store::Dep<crate::camdl_store::Quantities>>,
}

#[expect(clippy::disallowed_methods, reason = "reads the model file `model` names (declared `#[input(identity)]` on `SimulateArgs`), kept as provenance, and the IR camdlc compiled from it, the stored compiled model the cells' identities name")]
pub(crate) fn run(ctx: StoreRun<'_>) {
    let out = ctx.out;
    let fail = |e: String| -> ! { out.fail(format!("error: {e}")) };

    // The compiled model the cells' identities name: the model as compiled and
    // rewritten from data (its anchored times, the start override, the
    // integrator, its bound tables), before any cell's parameter values,
    // scenario selection and window are applied.
    let mut keyed = ctx.sim.model.ir.clone();
    let external = crate::external_inputs::ExternalBindings::from_cli(
        ctx.table_files.iter().map(|(k, v)| (k.clone(), std::path::PathBuf::from(v))),
    );
    crate::external_inputs::bind(&mut keyed, &external).unwrap_or_else(|e| fail(e.to_string()));
    let source = std::fs::read(ctx.model_path).unwrap_or_else(|e| fail(format!("cannot read {}: {e}", ctx.model_path)));
    let envelope = std::fs::read_to_string(ctx.ir_compiled)
        .map_err(|e| e.to_string())
        .and_then(|t| ir::envelope_from_str(&t).map_err(|e| e.to_string()))
        .unwrap_or_else(|e| fail(format!("IR load error from {}: {e}", ctx.ir_compiled)))
        .with_model(keyed);
    let stem = crate::hashing::path_stem_slug(ctx.model_path).unwrap_or_else(|| "model".to_string());
    let parts = crate::camdl_store::CompiledParts::split(
        envelope,
        stem.clone(),
        crate::camdl_store::ModelSource { bytes: &source, path: std::path::Path::new(ctx.model_path) },
    )
    .unwrap_or_else(|e| fail(e));

    let (specs, _grid) = crate::engine::plan_grid(ctx.job);
    crate::util::emit_run_notices(out, &specs.iter().map(|s| &s.sim_run).collect::<Vec<_>>());
    // A refusal of what was asked for, not of a cell, ends the command before
    // anything is stored: it is not a cell's failure.
    if ctx.wants.event_log {
        if let Some(refusal) = specs.first().and_then(|spec| crate::util::event_log_refusal(&spec.sim_run, &ctx.sim.model.ir)) {
            fail(refusal);
        }
    }

    if ctx.dry_run {
        dry_run_view(&ctx, parts, &specs, &stem);
        return;
    }

    let store = crate::camdl_store::open(&ctx.request)
        .unwrap_or_else(|e| fail(format!("the run store at {}: {e}", crate::camdl_store::root_of(&ctx.request).display())));
    let info = store::RunInfo::new(&ctx.given, &ctx.execution, &ctx.request).unwrap_or_else(|e| fail(e.to_string()));
    for item in &ctx.lineage {
        info.lineage(item.clone());
    }
    let crate::camdl_store::CompiledParts { model: model_part, quantities, labels: model_labels, provenance } = parts;
    let model = store
        .put::<CompiledModel>(model_part, &model_labels, &provenance, &info)
        .and_then(|s| s.dep(CompiledModel::FILE))
        .unwrap_or_else(|e| fail(format!("storing the compiled model: {e}")));
    let quantities = match &ctx.wants.quantities {
        Some(_) => Some(
            store
                .put::<crate::camdl_store::Quantities>(quantities, &model_labels, &store::Provenance::none(), &info)
                .and_then(|s| s.dep(crate::camdl_store::Quantities::FILE))
                .unwrap_or_else(|e| fail(format!("storing the quantities: {e}"))),
        ),
        None => {
            store::release(quantities, "no cell evaluates quantities without --quantities-out");
            None
        }
    };
    let shared = Shared { model, quantities };

    // ── Each cell's identity ────────────────────────────────────────────────
    let mut failures: Vec<String> = Vec::new();
    let n_specs = specs.len();
    let mut planned: Vec<Planned> = Vec::with_capacity(n_specs);
    let mut first_refusal: Option<String> = None;
    for spec in specs {
        match plan_cell(&ctx, &shared, spec) {
            Ok(p) => planned.push(p),
            Err((name, e)) => {
                first_refusal.get_or_insert_with(|| e.clone());
                failures.push(format!("{name}: {e}"));
            }
        }
    }
    // No cell resolved: the inputs are refused, as one error.
    if planned.is_empty() {
        if let Some(e) = first_refusal {
            drop(info);
            fail(e);
        }
    }
    let members: Vec<(SimIdentity, CellLabels, store::Wanted<Sim>)> =
        planned.iter().map(|p| (p.identity.clone(), p.labels.clone(), p.wanted.clone())).collect();
    // `begin_many` begins each distinct identity once, in the order first
    // asked for.
    let mut distinct: Vec<usize> = Vec::new();
    let mut seen: BTreeSet<store::Key> = BTreeSet::new();
    for (i, p) in planned.iter().enumerate() {
        if seen.insert(p.identity.key()) {
            distinct.push(i);
        }
    }

    let total = planned.len();
    let mut claimed: Vec<(Run<Sim>, usize)> = Vec::new();
    let mut n_reused = 0usize;
    if !planned.is_empty() {
        let mut batch = store.begin_many::<Sim>(&members, &info, ctx.request.reuse()).unwrap_or_else(|e| fail(e.to_string()));
        for (begun, &i) in batch.members().zip(&distinct) {
            match begun {
                store::Begin::Reused(_) => n_reused += 1,
                store::Begin::Busy(owner) => failures.push(format!(
                    "{}: {}",
                    cell_name(&planned[i].labels),
                    store::Error::InProgress(owner)
                )),
                store::Begin::Claimed(run) => claimed.push((run, i)),
            }
        }
    }

    if n_reused > 0 {
        note_model_stored_under_another_name(out, &store, &shared.model, &stem, ctx.model_path);
    }

    // ── The engine, under each claim ────────────────────────────────────────
    let bar = std::sync::Mutex::new((total > 1 && !claimed.is_empty()).then(|| crate::out::progress::Reporter::new().task(claimed.len() as u64, "simulate", "cells")));
    let lone = total == 1;
    let compute = |(run, i): (Run<Sim>, usize)| -> Option<String> {
        let p = &planned[i];
        let r = compute_cell(run, p, &ctx, lone);
        if let Some(t) = bar.lock().unwrap_or_else(|e| e.into_inner()).as_mut() {
            t.inc(1);
        }
        r.err().map(|e| format!("{}: {e}", cell_name(&p.labels)))
    };
    let computed = claimed.len();
    let errors: Vec<String> = if ctx.job.parallel > 1 {
        use rayon::prelude::*;
        log::debug!("simulate: computing cells on {} threads", rayon::current_num_threads());
        claimed.into_par_iter().filter_map(compute).collect()
    } else {
        log::debug!("simulate: computing cells on 1 threads");
        claimed.into_iter().filter_map(compute).collect()
    };
    if let Some(t) = bar.into_inner().unwrap_or_else(|e| e.into_inner()) {
        t.finish();
    }
    failures.extend(errors);
    let invocation = info.finish(&store).unwrap_or_else(|e| fail(e.to_string()));
    if let (Some(label), Some(before)) = (invocation.label(), invocation.took_label_from()) {
        out.say(format_args!("simulate: @{label} now names invocation {}, which held it from invocation {before}", invocation.seq()));
    }

    // ── Failures the cells stored ───────────────────────────────────────────
    failures.extend(stored_failures(&store, &invocation).unwrap_or_else(|e| fail(e)));
    let cells: Vec<(store::Stored<Sim>, CellLabels)> = invocation
        .members::<Sim>()
        .unwrap_or_else(|e| fail(e.to_string()))
        .into_iter()
        .filter_map(|(id, labels)| store.get(&id).ok().flatten().map(|s| (s, labels)))
        .collect();
    report(out, &store, &cells, total, computed, n_reused);
    if !failures.is_empty() {
        for f in &failures {
            out.say(format_args!("error: {f}"));
        }
        out.say(format_args!("simulate: {} of {} cell(s) did not produce what was asked; nothing was exported", failures.len(), n_specs));
        crate::out::ExitCode(1).exit();
    }

    // ── The exports ─────────────────────────────────────────────────────────
    let first = planned.first().map(|p| p.cell.run.model.clone());
    let dates = if ctx.exports.dates {
        let m = first.as_ref().unwrap_or(&ctx.sim.model.ir);
        match m.origin.clone() {
            Some(origin) => Some(Dates { origin, time_unit: m.time_unit.clone() }),
            None => fail("--dates requires the model to declare an `origin` (e.g. `origin = date(\"2020-01-01\")`).".into()),
        }
    } else {
        None
    };
    let traj_options = CombinedTableOptions { dates: dates.clone() };
    if let Some(path) = &ctx.exports.output {
        invocation
            .render::<CombinedTable>(&store, &traj_options, &store::ExportPath(path.clone()))
            .unwrap_or_else(|e| fail(format!("cannot write {}: {e}", path.display())));
        out.say(format_args!("trajectory written to {}", path.display()));
    }
    if ctx.exports.stdout {
        let mut bytes = Vec::new();
        invocation.render_to::<CombinedTable>(&store, &traj_options, &mut bytes).unwrap_or_else(|e| fail(e.to_string()));
        if let Err(e) = out.data().write_all(&bytes) {
            fail(format!("writing the trajectory to stdout: {e}"));
        }
    }
    if ctx.exports.obs_file.is_some() || ctx.exports.obs_dir.is_some() {
        let p = planned.first().unwrap_or_else(|| fail("no cell to export observations from".into()));
        let run_start = cells
            .first()
            .and_then(|(s, _)| crate::camdl_store::read_stored(s, TRAJ_FILE).ok())
            .and_then(|b| String::from_utf8(b).ok())
            .and_then(|t| t.lines().nth(1).and_then(|r| r.split('\t').next()).and_then(|t| t.parse::<f64>().ok()))
            .unwrap_or(p.cell.run.model.simulation.t_start);
        let streams =
            obs_plans(&p.cell, run_start, ctx.wants.emit_every.as_ref(), ctx.exports.obs_file.is_some()).unwrap_or_else(|e| fail(e));
        for (path, layout) in [(&ctx.exports.obs_file, ObsLayout::OneFile), (&ctx.exports.obs_dir, ObsLayout::PerSource)] {
            let Some(path) = path else { continue };
            let options = ObsTableOptions { dates: dates.clone(), layout, streams: streams.clone() };
            let mut files = store::ExportFiles::default();
            invocation.render_by_products_to::<ObsTable>(&store, &options, &mut files).unwrap_or_else(|e| fail(e.to_string()));
            write_export(out, path, &files, "observations written to");
        }
    }
    if let (Some(q), Some(p)) = (&ctx.wants.quantities, planned.first()) {
        let content = store
            .get(shared.quantities.as_ref().expect("stored with --quantities-out").id())
            .ok()
            .flatten()
            .unwrap_or_else(|| fail("the stored quantities are not found".into()));
        let quantities = crate::camdl_store::read_stored(&content, crate::camdl_store::Quantities::FILE)
            .and_then(|b| <crate::camdl_store::QuantitiesContent as store::Content>::from_bytes(&b).map_err(|e| e.to_string()))
            .and_then(|c| c.quantities())
            .unwrap_or_else(|e| fail(e));
        let options = QuantityTableOptions {
            quantities,
            mode: q.mode,
            scenario_axis: q.scenario_axis,
            calendar: ir::caltime::CalendarAxis::from_model(&p.cell.run.model).unwrap_or_else(|e| fail(e.to_string())),
            vocabulary: q.vocabulary.clone(),
        };
        let mut files = store::ExportFiles::default();
        invocation.render_by_products_to::<QuantityTable>(&store, &options, &mut files).unwrap_or_else(|e| fail(e.to_string()));
        write_export(out, &q.dir, &files, "quantities: wrote");
    }
    if let Some((stored, _)) = cells.first() {
        export_event_log(&ctx, &store, &invocation, stored);
        if let Some(dest) = &ctx.exports.reactive_log {
            if stored.result_files().contains(&REACTIVE_LOG_FILE) {
                if let Err(e) = stored.export(REACTIVE_LOG_FILE, &store::ExportPath(dest.clone())) {
                    out.say(format_args!("warning: --reactive-log mirror to {}: {}", dest.display(), e));
                }
            } else {
                out.say(
                    "warning: --reactive-log: this run has no active reactive policy \
                     (enable one with a scenario or --enable); nothing to mirror",
                );
            }
        }
    }
}

/// Says, once, that the cells reused were stored from a model file of another
/// name: the compiled model's first label, from the invocation that first
/// stored it, differs from this file's stem. The reuse is right, since a file
/// name is a label; the warning keeps it from being read as a fresh run.
fn note_model_stored_under_another_name(
    out: &crate::out::Out,
    store: &store::Store,
    model: &store::Dep<CompiledModel>,
    stem: &str,
    path: &str,
) {
    let first = store.invocations().into_iter().find_map(|inv| {
        inv.members::<CompiledModel>().ok()?.into_iter().find(|(id, _)| id == model.id()).map(|(_, l)| l.name)
    });
    let Some(first) = first.filter(|f| f != stem) else { return };
    let current = std::path::Path::new(path).file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
    let ext = std::path::Path::new(path).extension().map(|e| format!(".{}", e.to_string_lossy())).unwrap_or_default();
    out.say(format_args!(
        "warning: reusing stored cells computed from `{first}{ext}`: `{current}` is the same model under another \
         file name (the file name is a label, not part of a cell's identity). The stored cells keep `{first}`'s \
         directory names."
    ));
}

/// Writes an export the renderings collected in memory: one file named `""`
/// to the path itself, any other set under the path as a directory.
fn write_export(out: &crate::out::Out, to: &std::path::Path, files: &store::ExportFiles, said: &str) {
    files
        .write_to(&store::ExportPath(to.to_path_buf()))
        .unwrap_or_else(|e| out.fail(format!("error: cannot write {}: {e}", to.display())));
    for (name, _) in files.files() {
        let path = if name.is_empty() { to.to_path_buf() } else { to.join(name) };
        out.say(format_args!("{said} {}", path.display()));
    }
}

/// One cell: its resolution, identity and labels, and the by-products
/// wanted of it. Refused, with the cell's name, when the cell does not
/// resolve or runs on settings its identity does not hold.
fn plan_cell(ctx: &StoreRun<'_>, shared: &Shared, spec: crate::engine::CellSpec) -> Result<Planned, (String, String)> {
    // A named scenario is labelled by its name; every other arm, with or
    // without `--enable`/`--disable`/`--param`, is the implicit arm.
    let scenario = match &spec.scenario {
        crate::sim_job::ScenarioRef::Named(n) => n.clone(),
        crate::sim_job::ScenarioRef::Inline { .. } => crate::sim_job::IMPLICIT_ARM_LABEL.to_string(),
    };
    let rep = spec.run_idx % ctx.job.source.replicates().max(1);
    let (seed, seed_replicate) = match &ctx.job.seeds {
        crate::sim_job::Seeds::Explicit(_) => (spec.process_seed, 0),
        crate::sim_job::Seeds::Single(base) => (*base, rep),
    };
    let name = format!("scenario={scenario} seed={}", spec.process_seed);
    let err = |e: String| (name.clone(), e);
    let cell = ResolvedCell::resolve(ctx.sim.model.ir.clone(), &spec.sim_run).map_err(err)?;
    let e = ctx.engine;
    if cell.backend != e.backend || cell.dt.to_bits() != e.dt.to_bits() || cell.eval != e.eval {
        return Err(err(format!(
            "internal error: a cell runs on backend {:?} at dt = {} with {:?}, but the run is keyed on backend {:?} at dt = {} with {:?}",
            cell.backend, cell.dt, cell.eval, e.backend, e.dt, e.eval
        )));
    }
    if cell.binomial != sim::rng::BinomialAlgorithm::Btpe {
        return Err(err(format!("internal error: a cell runs the {:?} binomial sampler, which has no identity", cell.binomial)));
    }
    if !cell.required_output_times.is_empty() {
        return Err(err("internal error: a cell records snapshots beyond the model's schedule, which have no identity".into()));
    }
    let finite = |what: &str, x: f64| store::Finite::new(x).map_err(|e| err(format!("{what}: {e}")));
    let model = &cell.run.model;
    let values = crate::util::resolved_parameter_values(model);
    let run = &ctx.sim.run;
    let identity = SimIdentity {
        model: shared.model.clone(),
        config: SimConfig {
            backend: Backend::of(cell.backend),
            dt: finite("dt", cell.dt)?,
            t_start: finite("the start", model.simulation.t_start)?,
            t_end: finite("the end", model.simulation.t_end)?,
            init_state: cell.init_state.as_ref().map(|i| {
                Ok::<_, (String, String)>(InitStateRow {
                    ensemble: i.ensemble_digest,
                    row: i.row,
                })
            })
            .transpose()?,
            allow_degenerate_rates: run.allow_degenerate_rates,
            allow_forcing_extrapolation: run.allow_forcing_extrapolation,
            no_flows: run.view.no_flows,
            columns: run.view.allow.iter().cloned().collect(),
        },
        params: ResolvedParams::new(values.iter().map(|(k, v)| (k.clone(), *v))).map_err(err)?,
        scenario: ActiveInterventions::of(model),
        seed: ProcessSeed { seed: cell.seed },
    };
    // The label names the point's parameters at the values the cell ran.
    let ran_point: indexmap::IndexMap<String, f64> =
        spec.point_overrides.keys().filter_map(|k| values.get(k).map(|v| (k.clone(), *v))).collect();
    let point = match &ctx.job.source {
        crate::sim_job::ParamSource::Sweep { .. } => {
            let mut values: Vec<(String, f64)> = spec.point_overrides.iter().map(|(k, v)| (k.clone(), *v)).collect();
            values.sort_by(|a, b| a.0.cmp(&b.0));
            PointLabel::Sweep { point_id: spec.point_id, values }
        }
        crate::sim_job::ParamSource::Draws { .. } => PointLabel::Draw { index: spec.point_idx },
        crate::sim_job::ParamSource::Point { .. } => PointLabel::None,
    };
    let labels = CellLabels {
        model: ctx.sim.model.stem.clone(),
        params: params_label(&ran_point),
        scenario,
        replicate: spec.run_idx,
        seed,
        seed_replicate,
        process_seed: spec.process_seed,
        point,
    };
    let mut wanted = store::Wanted::<Sim>::none();
    if ctx.wants.obs && !model.observations.is_empty() {
        let id = ObsIdentity {
            sim: identity.record(),
            observation: crate::cas_sink::obs_subtree_hash(&model.observations, ctx.wants.emit_every.as_ref()).map_err(err)?,
            streams: model.observations.iter().map(|o| o.name.clone()).collect(),
            seed: spec.obs_seed,
        };
        wanted = wanted.with::<Obs>(id, ());
    }
    if ctx.wants.event_log {
        wanted = wanted.with::<EventLog>(EventLogIdentity { sim: identity.record() }, ());
    }
    if let Some(q) = &shared.quantities {
        let id = SimQuantitiesIdentity {
            sim: identity.record(),
            quantities: q.clone(),
            emit_every: ctx.wants.emit_every.as_ref().map(|e| e.identity_repr()),
            seed: spec.obs_seed,
        };
        wanted = wanted.with::<SimQuantities>(id, ());
    }
    Ok(Planned { spec, cell, identity, labels, wanted })
}

/// A readable label for a cell's parameters directory: `base`, or the
/// point's `name=value` pairs sorted by name, or `draws` when that text is
/// over 96 bytes.
fn params_label(point: &indexmap::IndexMap<String, f64>) -> String {
    if point.is_empty() {
        return "base".to_string();
    }
    let mut sorted: Vec<(&String, &f64)> = point.iter().collect();
    sorted.sort_by(|a, b| a.0.cmp(b.0));
    let joined = sorted.iter().map(|(k, v)| format!("{k}={v}")).collect::<Vec<_>>().join("_");
    if joined.len() <= 96 {
        joined
    } else {
        "draws".to_string()
    }
}

/// The model a run computes on, from the stored compiled model: the
/// identity's parameter values and interventions, and its window.
fn applied(model: &ir::Model, id: &SimIdentity) -> ir::Model {
    let mut m = crate::camdl_store::apply(model, &id.params, &id.scenario);
    m.simulation.t_start = id.config.t_start.get();
    m.simulation.t_end = id.config.t_end.get();
    m
}

/// Runs one claimed cell and commits it. An engine error, or a refused
/// by-product, is the cell's identity's and is stored; an error returned here
/// is the store's, and the cell is failed and leaves no folder.
fn compute_cell(mut run: Run<Sim>, p: &Planned, ctx: &StoreRun<'_>, lone: bool) -> Result<(), String> {
    let result = write_cell(&mut run, p, ctx, lone);
    match result {
        Ok(()) => run.finish().map(|_| ()).map_err(|e| e.to_string()),
        Err(e) => {
            let _ = run.fail("simulating", &e);
            Err(e)
        }
    }
}

fn write_cell(run: &mut Run<Sim>, p: &Planned, ctx: &StoreRun<'_>, lone: bool) -> Result<(), String> {
    let s = |e: store::Error| e.to_string();
    let identity = run.identity().clone();
    let stored = run.load(&identity.model).map_err(s)?;
    if crate::camdl_store::model_part_digest(&applied(stored.model(), &identity)) != crate::camdl_store::model_part_digest(&p.cell.run.model) {
        return Err("internal error: the model the cell runs is not the stored compiled model with the cell's \
                    parameters, interventions and window applied"
            .into());
    }
    let model = &p.cell.run.model;
    // The trajectory comes from the run a cell without by-products makes, so
    // a cell's stored files, its failure included, never depend on what else
    // was asked for. The event log is recorded by a second run under the
    // recorder, which draws nothing, and is the event log's alone.
    let engine = p.cell.compile().and_then(|compiled| simulate_with_progress(&compiled, &p.cell, lone));
    // Where each parameter value came from, the notices the resolution
    // reported, and under the forcing override each forcing the run held
    // outside its range: what the cell records besides its result.
    let mut record = serde_json::json!({
        "parameters_provenance": crate::run_meta::ParameterProvenance::of_all(&p.cell.run.params),
        "notices": crate::params_resolver::notices(&[&p.cell.run]),
    });
    ctx.engine.forcing.record_in(&mut record);
    let record = serde_json::to_vec_pretty(&record).map_err(|e| e.to_string())?;
    run.provenance_file(PROVENANCE_FILE).map_err(s)?.write_all(&record).map_err(s)?;
    let traj = match engine {
        Ok(t) => t,
        Err(e) => {
            let failure = Failure::new("engine", &e).to_bytes();
            run.result_file(FAILURE_FILE).map_err(s)?.write_all(&failure).map_err(s)?;
            if let Some(b) = run.also::<Obs>().map_err(s)? {
                b.result_file(FAILURE_FILE).map_err(s)?.write_all(&failure).map_err(s)?;
            }
            if let Some(b) = run.also::<EventLog>().map_err(s)? {
                b.result_file(FAILURE_FILE).map_err(s)?.write_all(&failure).map_err(s)?;
            }
            if let Some(b) = run.also::<SimQuantities>().map_err(s)? {
                b.result_file(FAILURE_FILE).map_err(s)?.write_all(&failure).map_err(s)?;
            }
            return Ok(());
        }
    };
    if p.spec.run_idx == 0 && !traj.transition_diagnostics.is_empty() {
        crate::write_transition_diagnostics(&traj);
    }
    let cols = ctx.sim.run.view.cols(model);
    run.result_file(TRAJ_FILE).map_err(s)?.write_all(&crate::util::traj_tsv_bytes(&traj, &cols)).map_err(s)?;
    if let Some(firings) = &traj.reactive_log {
        run.result_file(REACTIVE_LOG_FILE)
            .map_err(s)?
            .write_all(sim::reactive::format_reactive_log(firings).as_bytes())
            .map_err(s)?;
    }
    if let Some(b) = run.also::<Obs>().map_err(s)? {
        match obs_files(model, &traj, p, ctx.wants.emit_every.as_ref()) {
            Ok(files) => {
                for (name, bytes) in files {
                    b.result_file(&name).map_err(s)?.write_all(&bytes).map_err(s)?;
                }
            }
            Err(e) => b.result_file(FAILURE_FILE).map_err(s)?.write_all(&Failure::new("observations", &e).to_bytes()).map_err(s)?,
        }
    }
    if let Some(b) = run.also::<EventLog>().map_err(s)? {
        match crate::util::run_simulation_event_log(&p.spec.sim_run) {
            Ok((recorded, _, log, _exact)) => {
                if crate::util::traj_tsv_bytes(&recorded, &cols) != crate::util::traj_tsv_bytes(&traj, &cols) {
                    return Err("internal error: the run under the event-log recorder differs from the cell's run".into());
                }
                let bytes = sim::lineage::event_log_io::to_tsv_bytes(&log).map_err(|e| format!("event log: {e:?}"))?;
                b.result_file(EventLog::FILE).map_err(s)?.write_all(&bytes).map_err(s)?;
            }
            Err(e) => b.result_file(FAILURE_FILE).map_err(s)?.write_all(&Failure::new("event log", &e).to_bytes()).map_err(s)?,
        }
    }
    if let Some(b) = run.also::<SimQuantities>().map_err(s)? {
        let quantities = match &ctx.wants.quantities {
            Some(_) => stored_quantities(run, &b),
            None => Err("internal error: quantities wanted without --quantities-out".into()),
        };
        match quantities.and_then(|q| evaluate_quantities(&q, p, &traj, ctx.wants.emit_every.as_ref())) {
            Ok(bytes) => b.result_file(SimQuantities::FILE).map_err(s)?.write_all(&bytes).map_err(s)?,
            Err(e) => b.result_file(FAILURE_FILE).map_err(s)?.write_all(&Failure::new("quantities", &e).to_bytes()).map_err(s)?,
        }
    }
    Ok(())
}

/// The quantities a cell evaluates, read through its claim from the content
/// artifact its by-product's identity names.
fn stored_quantities(run: &mut Run<Sim>, b: &ByProductRun<SimQuantities>) -> Result<Vec<ir::quantity::Quantity>, String> {
    let dep = b.identity().quantities.clone();
    let content = run.load(&dep).map_err(|e| e.to_string())?;
    content.quantities()
}

/// Each stream's stored file for one cell, in declaration order. One
/// observation RNG, seeded with the cell's observation seed, is consumed in
/// declaration order across streams.
fn obs_files(
    model: &ir::Model,
    traj: &sim::Trajectory,
    p: &Planned,
    emit: Option<&crate::emit_every::EmitEvery>,
) -> Result<Vec<(String, Vec<u8>)>, String> {
    let restart = p.cell.restart;
    let eval = p.spec.sim_run.eval;
    let run_start = crate::run_start_of(traj, model);
    // Every stream is planned and checked before any is drawn.
    let mut plans = Vec::with_capacity(model.observations.len());
    for obs_ir in &model.observations {
        crate::obs_emit::check_data_columns_supplied(obs_ir, crate::obs_emit::ColumnSupply::Nothing)?;
        let times = crate::obs_emit_schedule_times(obs_ir, restart, model.simulation.t_end, emit)?;
        let plan = crate::obs_emit::plan_emission(
            obs_ir, &times, run_start, model.simulation.t_end, emit.and_then(|e| e.resolve_for(&obs_ir.source)),
        )?;
        crate::project_coverages(traj, obs_ir, model, &plan.scored(), eval)?;
        plans.push(plan);
    }
    let compiled = std::sync::Arc::new(
        sim::CompiledModel::new(model.clone()).map_err(|e| format!("model compile error for obs: {e:?}"))?.with_eval_settings(eval),
    );
    let params = compiled.default_params.clone();
    let mut obs_rng = sim::rng::StatefulRng::new(p.spec.obs_seed);
    let mut files = Vec::new();
    for (obs_ir, plan) in model.observations.iter().zip(&plans) {
        let sampler = sim::inference::obs_model::compile_obs_sample_pf(obs_ir, compiled.clone(), &params).map_err(|e| e.to_string())?;
        let rows = plan.scored();
        let projected = crate::project_coverages(traj, obs_ir, model, &rows, eval)?;
        let draws: Vec<f64> = rows
            .iter()
            .enumerate()
            .map(|(ti, &(at, _))| {
                let snap = crate::snap_at(traj, at);
                sampler(projected[ti], at, &snap.int_state.counts, &[], &mut obs_rng)
            })
            .collect();
        let name = format!("{}.tsv", obs_ir.name);
        let bytes = crate::obs_emit::stream_columns_bytes(
            std::path::Path::new(&name), plan, &plan.scored, &draws, &[], crate::obs_emit::RowValues::Observed,
        )?;
        files.push((name, bytes));
    }
    Ok(files)
}

/// Runs a compiled cell, with a per-timestep bar when it is the command's
/// only cell. The bar's tick draws no randomness, so the trajectory is the
/// one the bar-less run makes.
fn simulate_with_progress(compiled: &sim::CompiledModel, cell: &ResolvedCell, lone: bool) -> Result<sim::Trajectory, String> {
    if !lone || crate::out::progress::is_none() {
        return crate::util::simulate_compiled(compiled, cell, None);
    }
    let backend = cell.backend.as_str();
    let pb = indicatif::ProgressBar::with_draw_target(Some(1000), crate::out::progress::draw_target());
    pb.set_style(
        indicatif::ProgressStyle::with_template("simulate \u{b7} {prefix}  {bar:24.cyan/blue} ETA {eta}")
            .unwrap_or_else(|_| indicatif::ProgressStyle::default_bar())
            .progress_chars("\u{2588}\u{2591} "),
    );
    pb.set_prefix(backend.to_string());
    pb.enable_steady_tick(std::time::Duration::from_millis(100));
    if crate::out::progress::is_plain() {
        crate::out::live().progress(&format!("simulate \u{b7} {backend}: running \u{2026}"));
    }
    let r = crate::util::simulate_compiled(compiled, cell, Some(&pb));
    pb.finish_and_clear();
    r
}

/// One cell's quantity values, as `quantities.json` holds them.
fn evaluate_quantities(
    quantities: &[ir::quantity::Quantity],
    p: &Planned,
    traj: &sim::Trajectory,
    emit: Option<&crate::emit_every::EmitEvery>,
) -> Result<Vec<u8>, String> {
    let model = &p.cell.run.model;
    let compiled = std::sync::Arc::new(
        sim::CompiledModel::new(model.clone())
            .map_err(|e| format!("compiling model for quantities: {e:?}"))?
            .with_eval_settings(p.spec.sim_run.eval),
    );
    let eval = sim::quantity::QuantityEvaluator::new(quantities, compiled.as_ref()).map_err(|e| format!("building quantity evaluator: {e}"))?;
    let mut params = vec![f64::NAN; compiled.param_index.len()];
    for prm in &model.parameters {
        if let (Some(&idx), Some(v)) = (compiled.param_index.get(prm.name.as_str()), prm.value.resolved_value()) {
            params[idx] = v;
        }
    }
    if eval.references_obs_anchor() {
        return Err(format!(
            "quantity `{}` reads `value_at` at an observation anchor (`last_obs` / `first_obs`, with or \
             without an offset), but a forward simulation has no observed data to anchor to. Evaluate it \
             via `camdl predict` (where the fit's data supplies the observation times), or use a literal \
             time: `value_at(expr, date(\"...\"))`.",
            eval.obs_anchor_quantity_names().join("`, `"),
        ));
    }
    let streams = eval.obs_streams();
    let obs_set = if streams.is_empty() {
        None
    } else {
        Some(crate::materialize_obs_for_quantities(&streams, model, traj, &compiled, &params, p.spec.obs_seed, p.cell.restart, emit)?)
    };
    let results = eval.eval_draw(&params, traj, sim::quantity::ConditionedRead::Off, &compiled, obs_set.as_ref(), None);
    let times: Vec<f64> = traj.snapshots.iter().map(|s| s.t).collect();
    serde_json::to_vec_pretty(&CellQuantities::new(&times, &results)).map_err(|e| e.to_string())
}

/// Every stream as the combined observation tables write it, planned from
/// one cell's model: the cells share one observation axis (the command
/// refuses scenarios whose horizons differ before anything runs).
fn obs_plans(
    cell: &ResolvedCell,
    run_start: f64,
    emit: Option<&crate::emit_every::EmitEvery>,
    one_file: bool,
) -> Result<Vec<ObsStreamPlan>, String> {
    let model = &cell.run.model;
    let mut out = Vec::new();
    for obs in &model.observations {
        let times = crate::obs_emit_schedule_times(obs, cell.restart, model.simulation.t_end, emit)?;
        let plan = crate::obs_emit::plan_emission(obs, &times, run_start, model.simulation.t_end, emit.and_then(|e| e.resolve_for(&obs.source)))?;
        if one_file && matches!(plan.columns, crate::obs_emit::TemporalColumns::Window { .. }) {
            return Err(format!(
                "observation stream '{}' declares `window_start`/`window_stop` columns, which a single wide `--obs` \
                 file cannot carry.\n  Use --obs-dir (one file per stream, keeps trajectory) or --obs-only-dir \
                 (one file per stream, suppresses trajectory).",
                obs.name
            ));
        }
        out.push(ObsStreamPlan { name: obs.name.clone(), source: obs.source.clone(), dims: crate::obs_emit::dim_cells(obs)?, plan });
    }
    Ok(out)
}

/// Each failure a member of the invocation stored, with the cell's name.
fn stored_failures(store: &store::Store, invocation: &store::Invocation) -> Result<Vec<String>, String> {
    let mut out = Vec::new();
    let mut last: Option<String> = None;
    let members = invocation.members::<Sim>().map_err(|e| e.to_string())?;
    // A by-product's upstream is named by its identity's record of the cell.
    let mut by_record = std::collections::BTreeMap::new();
    for (id, labels) in &members {
        if let Some(s) = store.get(id).map_err(|e| e.to_string())? {
            by_record.entry(s.identity().content_hash()).or_insert_with(|| cell_name(labels));
            if let Some(f) = Failure::of(&s)? {
                let line = format!("{}: the {} stopped: {}", cell_name(labels), f.stage, f.reason);
                if last.as_deref() != Some(line.as_str()) {
                    out.push(line.clone());
                    last = Some(line);
                }
            }
        }
    }
    macro_rules! by_products {
        ($($t:ty),*) => {$(
            for (id, ()) in invocation.members::<$t>().map_err(|e| e.to_string())? {
                if let Some(s) = store.get(&id).map_err(|e| e.to_string())? {
                    if let Some(f) = Failure::of(&s)? {
                        if f.stage != "engine" {
                            let name = by_record.get(&s.identity().sim.content_hash()).cloned()
                                .unwrap_or_else(|| id.key().short12());
                            out.push(format!("{name}: the {} stopped: {}", f.stage, f.reason));
                        }
                    }
                }
            }
        )*};
    }
    by_products!(Obs, SimQuantities, EventLog);
    Ok(out)
}

/// The milestone a run prints: the one cell's folder, or the number of cells
/// and the store's root, with how many were read from the store.
fn report(out: &crate::out::Out, store: &store::Store, cells: &[(store::Stored<Sim>, CellLabels)], total: usize, computed: usize, reused: usize) {
    if reused > 0 && total > 1 {
        out.milestone("cached", format!("{reused} of {total} cells read from the store; simulated {computed}"));
    }
    let mut distinct: Vec<&store::Stored<Sim>> = Vec::new();
    for (stored, _) in cells {
        if !distinct.iter().any(|d| d.id() == stored.id()) {
            distinct.push(stored);
        }
    }
    match distinct.as_slice() {
        [] => {}
        [stored] => {
            let folder = crate::run_paths::display_absolute(stored.folder());
            out.milestone(if computed > 0 { "stored" } else { "cached" }, folder.display());
            out.hint(format_args!("camdl cat {}", stored.id().key().to_hex()));
        }
        many => {
            out.milestone("stored", format!("{} cells \u{b7} {}", many.len(), crate::run_paths::display_absolute(store.root()).display()));
            out.hint("camdl list");
        }
    }
}

/// `--event-log`: the stored log, exported to the path when one was given,
/// in the format `--tsv`, `--format` or the extension names.
fn export_event_log(ctx: &StoreRun<'_>, store: &store::Store, invocation: &store::Invocation, _cell: &store::Stored<Sim>) {
    if !ctx.wants.event_log {
        return;
    }
    let out = ctx.out;
    let fail = |e: String| -> ! { out.fail(format!("error: {e}")) };
    let (id, ()) = invocation
        .members::<EventLog>()
        .ok()
        .and_then(|m| m.into_iter().next())
        .unwrap_or_else(|| fail("internal error: --event-log stored no event log".into()));
    let stored = store.get(&id).ok().flatten().unwrap_or_else(|| fail("the event log is not stored".into()));
    let bytes = crate::camdl_store::read_stored(&stored, EventLog::FILE).unwrap_or_else(|e| fail(e));
    let file = stored.folder().join(EventLog::FILE);
    let log = sim::lineage::event_log_io::from_tsv_bytes(&bytes)
        .unwrap_or_else(|e| fail(format!("reading the stored event log: {e:?}")));
    let exact = matches!(ctx.engine.backend, crate::args::types::ForwardBackend::Gillespie);
    crate::lineage::finish_event_log(
        out,
        ctx.exports.event_log.as_deref(),
        ctx.exports.event_log_tsv,
        &ctx.exports.event_log_format,
        &log,
        &bytes,
        exact,
        &crate::run_paths::display_absolute(&file).display().to_string(),
    );
}

/// `--dry-run`'s store view: per cell, whether the store already holds it,
/// asked through the store's own lookup, and the hit and miss counts.
/// Nothing is written: the compiled model is looked up by its content and
/// never put.
fn dry_run_view(ctx: &StoreRun<'_>, parts: crate::camdl_store::CompiledParts, specs: &[crate::engine::CellSpec], _stem: &str) {
    let out = ctx.out;
    let root = crate::camdl_store::root_of(&ctx.request);
    out.say("");
    out.say(format_args!("{} {}", crate::out::paint::dim("store:"), root.display()));
    let store = crate::camdl_store::open_existing(&root);
    let model = store.as_ref().and_then(|s| s.get_content::<CompiledModel>(&parts.model).ok().flatten()).and_then(|m| m.dep(CompiledModel::FILE).ok());
    let quantities = store
        .as_ref()
        .and_then(|s| s.get_content::<crate::camdl_store::Quantities>(&parts.quantities).ok().flatten())
        .and_then(|q| q.dep(crate::camdl_store::Quantities::FILE).ok());
    // A cell whose quantities were asked for cannot be stored with them when
    // the quantities themselves are not stored.
    let quantities_missing = ctx.wants.quantities.is_some() && quantities.is_none();
    let (mut hits, mut misses, mut unresolved) = (0usize, 0usize, 0usize);
    for spec in specs {
        let (Some(store), Some(model)) = (&store, &model) else {
            misses += 1;
            continue;
        };
        let shared = Shared { model: model.clone(), quantities: if ctx.wants.quantities.is_some() { quantities.clone() } else { None } };
        match plan_cell(ctx, &shared, spec.clone()) {
            Ok(p) => match store.plan_with::<Sim>(&[(p.identity, p.labels.clone(), p.wanted)]) {
                Ok(v) if !quantities_missing && v.first().is_some_and(|x| x.state == store::PlanState::Completed) => {
                    hits += 1;
                    out.say(format_args!("  [hit ] {} {}", cell_name(&p.labels), v[0].id.key().short12()));
                }
                _ => {
                    misses += 1;
                    out.say(format_args!("  [miss] {}", cell_name(&p.labels)));
                }
            },
            Err((name, e)) => {
                unresolved += 1;
                out.say(format_args!("  [fail] {name} (identity unresolved: {e})"));
            }
        }
    }
    if ctx.request.reuse() == store::Reuse::Force {
        out.say(format_args!("  {hits} in the store; --force recomputes every cell ({} runs)", specs.len()));
    } else {
        out.say(format_args!("  {hits} cache hits \u{2192} read back, {misses} misses \u{2192} would simulate"));
    }
    if unresolved > 0 {
        out.say(format_args!("  {unresolved} cells whose identity does not resolve \u{2192} would fail"));
    }
    out.say("(dry run \u{2014} no simulation, no files written.)");
}
