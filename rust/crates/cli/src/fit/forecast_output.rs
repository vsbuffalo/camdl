//! `camdl fit predict --mode forecast`: the fixed-origin forecast of a fit,
//! from the end of its training rows out to `--horizon`
//! (`docs/dev/temporal-architecture.md` §2.5, §4.4; migration plan, Stage 5).
//!
//! Reads the fit's recorded training partition (verified against the data),
//! builds the [`ForecastPlan`], runs the [`fixed_origin_forecast`] driver over
//! the posterior draws, and writes the forecast artifact:
//!
//! - `forecast/<stream>.tsv`: one row per target, with its period, horizon,
//!   lead time, quantiles and mean;
//! - `forecast-samples.tsv`: joint samples, each a complete trajectory across
//!   every target (D54);
//! - `forecast.json`: the plan (origin, reach, targets and where each stream's
//!   came from), the parameter set and the state set, and the calendar.

use std::collections::BTreeMap;
use std::path::PathBuf;
use std::sync::Arc;

use crate::fit::failures::{DeterministicFailure, Site};
use crate::fit::forecast_driver::{extend_run, fixed_origin_forecast, DriverConfig, Forecast, StateConditioning};
use crate::fit::forecast_plan::{parse_reach, ForecastPlan, HeldOut, PlanStream, RowFile};
use crate::fit::observed_rows::{Measures, ObservedRow};
use crate::fit::predict::PosteriorDraws;
use crate::fit::training_partition::TrainingPartition;
use crate::quantile::{band, fmt_time, fmt_value, QUANTILE_LEVELS};

/// Joint samples written per posterior draw, evenly strided over its
/// particles. Every particle is a draw from the same predictive (equal weights
/// after the origin), so a stride is a sample of them; the bands use all.
pub(crate) const FORECAST_SAMPLES_PER_DRAW: usize = 5;

/// A posterior draw the forecast ran: its row of the fit's `draws.tsv` and
/// the `(chain, draw)` key written there, when the cloud carries one.
#[derive(Debug, Clone, Copy)]
pub(crate) struct DrawRef {
    pub cloud_row: usize,
    pub key: Option<(usize, usize)>,
}

impl DrawRef {
    /// How a message names the draw.
    pub(crate) fn describe(&self) -> String {
        match self.key {
            Some((c, d)) => format!(
                "posterior draw (chain {c}, draw {d}), row {} of draws.tsv", self.cloud_row),
            None => format!("posterior draw at row {} of draws.tsv", self.cloud_row),
        }
    }
}

/// The `schema` tag of `forecast.json`.
pub(crate) const FORECAST_SCHEMA: &str = "camdl.forecast/v1";

/// Refuse what forecast mode does not support, before any work: the backends
/// and options of the Stage 5 capability matrix that have no forecast yet.
pub(crate) fn refuse_unsupported(
    args: &crate::args::FitPredictArgs,
    posterior: &PosteriorDraws,
) -> Result<(), String> {
    use crate::args::types::ForwardBackend;
    let leaf = &posterior.name;
    match posterior.backend {
        ForwardBackend::ChainBinomial => {}
        ForwardBackend::Ode => return Err(format!(
            "the '{leaf}' method leaf ran on the ODE backend; a fixed-origin forecast of an \
             ODE posterior (the cumulative-flow difference on each draw's trajectory) is not \
             available yet. A chain-binomial posterior (PMMH, PGAS) can be forecast; for an \
             ODE fit use --mode free_forward.")),
        ForwardBackend::Gillespie => return Err(format!(
            "the '{leaf}' method leaf ran on the Gillespie backend, which is not a fitting \
             backend, so there is nothing to forecast.")),
    }
    let mut asked: Vec<&str> = Vec::new();
    if !args.scenarios.is_empty() { asked.push("--scenario"); }
    if !args.enable.is_empty() { asked.push("--enable"); }
    if !args.disable.is_empty() { asked.push("--disable"); }
    if !args.sweep.is_empty() { asked.push("--sweep"); }
    if args.by_chain { asked.push("--by-chain"); }
    if args.quantities.is_some() { asked.push("--quantities"); }
    if !asked.is_empty() {
        return Err(format!(
            "{} cannot be combined with --mode forecast yet: the forecast is the fitted \
             model's, conditioned on the training rows, and writes no quantities table. \
             Scenario arms that differ only after the forecast origin are the planned \
             extension; until then run them with --mode free_forward.",
            asked.join(", ")));
    }
    Ok(())
}

/// The binomial sampler the fit run in `leaf` drew with (gh#747), so the
/// forecast continues the fit's filter on the same sampler. Read from the
/// leaf's `run.json` (`inputs.binomial`, written by `fit run`). A leaf written
/// before that record has none: its sampler is `fit run --binomial`'s value
/// when its recorded command line carries one, else `configured`, the method
/// as `fit.toml` wrote it.
pub(crate) fn fit_binomial(
    leaf: &std::path::Path,
    configured: Option<sim::rng::BinomialAlgorithm>,
) -> Result<sim::rng::BinomialAlgorithm, String> {
    use sim::rng::BinomialAlgorithm;
    let path = leaf.join("run.json");
    let text = std::fs::read_to_string(&path)
        .map_err(|e| format!("cannot read {}: {e}", path.display()))?;
    let record: serde_json::Value = serde_json::from_str(&text)
        .map_err(|e| format!("{} is not JSON: {e}", path.display()))?;
    let parse = |v: &serde_json::Value| -> Result<BinomialAlgorithm, String> {
        serde_json::from_value(v.clone()).map_err(|e| format!(
            "{}: the recorded binomial sampler {v} is not one camdl knows: {e}", path.display()))
    };
    match record.pointer("/inputs/binomial") {
        Some(v) if !v.is_null() => return parse(v),
        _ => {}
    }
    let argv: Vec<&str> = record.pointer("/provenance/argv")
        .and_then(|a| a.as_array())
        .map(|a| a.iter().filter_map(|s| s.as_str()).collect())
        .unwrap_or_default();
    let from_argv = argv.iter().enumerate().find_map(|(i, a)| match a.strip_prefix("--binomial") {
        Some("") => argv.get(i + 1).copied(),
        Some(rest) => rest.strip_prefix('='),
        None => None,
    });
    match (from_argv, configured) {
        (Some(v), _) => parse(&serde_json::Value::String(v.to_string())),
        (None, Some(c)) => Ok(c),
        (None, None) => Err(format!(
            "{} records no binomial sampler and its fit.toml names no method; re-run the fit \
             so the forecast can continue its filter on the same sampler.", path.display())),
    }
}

/// What the forecast needs from `run_predict`.
pub(crate) struct ForecastJob<'a> {
    pub args: &'a crate::args::FitPredictArgs,
    /// The fitted model, observation anchors resolved.
    pub model: &'a ir::Model,
    /// The same model with every parameter seeded from a draw, so it compiles.
    pub seeded: &'a ir::Model,
    pub compiled: &'a Arc<sim::CompiledModel>,
    pub config: &'a crate::fit::config_v2::Problem,
    pub fit_base: &'a crate::util::FitBase,
    pub posterior: &'a PosteriorDraws,
    pub segment: &'a std::path::Path,
    pub out_dir: &'a std::path::Path,
    pub run_id: &'a str,
    pub options: &'a crate::fit::predict::PredictOptions,
    pub calendar: &'a ir::caltime::CalendarAxis,
    pub schema: Option<&'a crate::run_meta::ObsSchema>,
    pub dt: f64,
    pub seed: u64,
    /// The binomial sampler the fit's method ran (gh#747).
    pub binomial: sim::rng::BinomialAlgorithm,
}

/// Run the forecast and write its artifact. Returns the files written and the
/// streams refused a forecast (recorded in `report.json`).
pub(crate) fn run_forecast(
    out: &crate::out::Out,
    job: ForecastJob,
) -> Result<(Vec<PathBuf>, Vec<DeterministicFailure>), String> {
    let model = job.model;
    let raw = job.args.horizon.as_deref().expect("forecast mode carries --horizon");
    let reach = parse_reach(raw, &model.time_unit, model.origin.as_deref())?;
    let data = job.config.data_spec()
        .map_err(|e| format!("this fit has no [data] block to read its training rows from: {e}"))?;
    let bindings = crate::fit::stream_bindings::StreamBindings::resolve(model, data)
        .map_err(|e| format!("resolving [data]: {e}"))?;
    let time_opts = crate::caltime_load::TimeOpts {
        origin: model.origin.as_deref(),
        time_unit: &model.time_unit,
        dt: job.dt,
        t_start: model.simulation.t_start,
        format: crate::caltime_load::TimeFormat::Auto,
    };
    let mut streams = crate::fit::runner::resolve_and_load_obs_streams(
        model, job.compiled, bindings.training_by_source(), &time_opts)?;

    // The parameter set: the rows the fit recorded, verified against the data.
    let record = crate::run_meta::read_fit_sidecar(job.segment)?
        .and_then(|s| s.training_partition)
        .ok_or_else(|| format!(
            "the fit at {} records no training partition, so which rows its parameters \
             were fitted to cannot be read; re-run the fit with `camdl fit run`.",
            job.segment.display()))?;
    let rows: BTreeMap<String, Vec<ObservedRow>> =
        streams.iter().map(|s| (s.name.clone(), s.rows.clone())).collect();
    let partition = TrainingPartition::from_record(&record, &rows)?;

    // Rows after each stream's prefix, in its own file or a holdout file:
    // they only name a target's held-out row, and are never filtered.
    let mut held_out: BTreeMap<String, Vec<HeldOut>> = BTreeMap::new();
    for s in &streams {
        let mut tail: Vec<HeldOut> = s.rows[partition.rows(&s.name)..].iter()
            .map(|r| HeldOut { file: RowFile::Training, row: r.clone() })
            .collect();
        if let Some(path) = bindings.holdout_by_source().get(&s.obs_model_ir.source) {
            let siblings: Vec<&ir::observation::ObservationModel> =
                model.observations.iter().filter(|o| o.source == s.obs_model_ir.source).collect();
            tail.extend(crate::fit::runner::load_observations(path, &s.obs_model_ir, &siblings, &time_opts)?
                .into_iter()
                .map(|row| HeldOut { file: RowFile::Holdout, row }));
        }
        held_out.insert(s.name.clone(), tail);
    }
    for s in &mut streams {
        s.truncate_rows(partition.rows(&s.name));
    }

    // The plan: targets for the streams `--stream` selects; every stream
    // conditions the state.
    let selected = |o: &ir::observation::ObservationModel| match job.args.stream.as_deref() {
        None => true,
        Some(name) => o.source == name || o.name == name,
    };
    let plan_streams: Vec<PlanStream> = streams.iter()
        .filter(|s| selected(&s.obs_model_ir))
        .map(|s| PlanStream { obs: &s.obs_model_ir, training: &s.rows, held_out: &held_out[&s.name] })
        .collect();
    if plan_streams.is_empty() {
        return Err(format!("--stream {:?} names no bound observation stream",
            job.args.stream.as_deref().unwrap_or("")));
    }
    let plan = ForecastPlan::new(partition, reach, &plan_streams).map_err(|e| e.to_string())?;
    // A stream the fit bound no data to has no rows to continue and no stop
    // in the filter where its running total could start: refused, and named.
    let unbound: Vec<crate::fit::forecast_plan::StreamRefusal> = model.observations.iter()
        .filter(|o| selected(o) && !streams.iter().any(|s| s.name == o.name))
        .map(|o| crate::fit::forecast_plan::StreamRefusal::NoBoundRows { stream: o.name.clone() })
        .collect();
    let failures: Vec<DeterministicFailure> = plan.refusals().iter().chain(&unbound)
        .map(|r| {
            let stream = match r {
                crate::fit::forecast_plan::StreamRefusal::NoRhythm { stream, .. }
                | crate::fit::forecast_plan::StreamRefusal::NoBoundRows { stream }
                | crate::fit::forecast_plan::StreamRefusal::ReadsDataColumn { stream, .. }
                | crate::fit::forecast_plan::StreamRefusal::OverlapsObserved { stream, .. } => stream.clone(),
            };
            DeterministicFailure::EvaluationFailed { at: Site::ForecastStream(stream), reason: r.to_string() }
        })
        .collect();

    // The fitted model, its run extended to the latest target (D49).
    let mut fitted = job.seeded.clone();
    job.fit_base.filter(&mut fitted)?;
    extend_run(&mut fitted, &plan);
    if plan.run_end() > model.simulation.t_end {
        out.say(format_args!(
            "fit predict: the forecast runs to t = {}, past the model's `simulate {{ to }}` \
             (t = {}), with interventions planned to that end; nothing else is extended.",
            plan.run_end(), model.simulation.t_end));
    }
    let compiled = Arc::new(sim::CompiledModel::new(fitted)
        .map_err(|e| format!("compiling the fitted model for the forecast: {e:?}"))?);

    // The posterior draws, subsampled as the other modes do.
    let cap = job.args.n_draws.unwrap_or(crate::fit::predict::DEFAULT_PREDICT_DRAWS);
    let idx = crate::fit::predict::subsample_indices(job.posterior.n_draws(), cap);
    if idx.len() < job.posterior.n_draws() {
        out.say(format_args!(
            "fit predict: forecast — subsampling {} of {} posterior draws (raise with --n-draws)",
            idx.len(), job.posterior.n_draws()));
    }
    let draw_refs: Vec<DrawRef> = idx.iter()
        .map(|&i| DrawRef {
            cloud_row: i,
            key: job.posterior.keys().and_then(|k| k.per_draw[i]),
        })
        .collect();
    let param_sets: Vec<Vec<f64>> = idx.iter()
        .map(|&i| {
            let mut p = compiled.default_params.clone();
            for (name, &v) in &job.posterior.draws()[i] {
                if let Some(&k) = compiled.param_index.get(name.as_str()) {
                    p[k] = v;
                }
            }
            p
        })
        .collect();
    let fc = fixed_origin_forecast(plan, &streams, compiled, &param_sets, DriverConfig {
        n_particles: crate::fit::predict::PREDICT_N_PARTICLES,
        dt: job.dt,
        base_seed: job.seed,
        record_expected: false,
        binomial: job.binomial,
    }).map_err(|f| f.describe(|d| draw_refs[d].describe()))?;

    let mut written = write_tables(&fc, model, job.schema, job.out_dir, &draw_refs)?;
    let refused: Vec<crate::fit::forecast_plan::StreamRefusal> =
        fc.plan.refusals().iter().chain(&unbound).cloned().collect();
    written.push(write_manifest(&fc, &job, &record, raw, &refused, &draw_refs)?);
    out.say(format_args!(
        "fit predict: forecast from t = {} (the end of the training rows) to t = {}: {} \
         target(s) on {} stream(s), {} draws × {} particles",
        fc.plan.origin(), fc.plan.origin() + fc.plan.horizon(), fc.plan.targets().len(),
        fc.plan.sources().len(), fc.n_draws, fc.n_particles));
    Ok((written, failures))
}

fn leaf<'m>(model: &'m ir::Model, name: &str) -> &'m ir::observation::ObservationModel {
    model.observations.iter().find(|o| o.name == name)
        .expect("a target's stream is one of the model's observation blocks")
}

fn stratum_of(o: &ir::observation::ObservationModel) -> Vec<(String, String)> {
    o.stratum.iter().map(|k| (k.dim.clone(), k.level.clone())).collect()
}

/// `forecast/<source>.tsv` per logical stream, and `forecast-samples.tsv`.
fn write_tables(
    fc: &Forecast,
    model: &ir::Model,
    schema: Option<&crate::run_meta::ObsSchema>,
    out_dir: &std::path::Path,
    draw_refs: &[DrawRef],
) -> Result<Vec<PathBuf>, String> {
    let targets = fc.plan.targets();
    // Logical stream → its targets' indices, in the plan's order.
    let mut by_source: indexmap::IndexMap<String, Vec<usize>> = indexmap::IndexMap::new();
    for (k, t) in targets.iter().enumerate() {
        by_source.entry(leaf(model, &t.stream).source.clone()).or_default().push(k);
    }
    let mut written = Vec::new();
    for (source, ks) in &by_source {
        let first = leaf(model, &targets[ks[0]].stream);
        let dims = crate::fit::predict::index_dims_for(schema, source, &stratum_of(first).into_iter().map(|(d, _)| d).collect::<Vec<_>>());
        let mut header: Vec<String> = dims.clone();
        header.extend(["t_start", "t_stop", "horizon", "lead", "held_out_file", "held_out_row", "mean"]
            .map(String::from));
        header.extend(QUANTILE_LEVELS.iter().map(|(_, l)| l.to_string()));
        header.push("n_samples".into());
        let mut tsv = header.join("\t") + "\n";
        for &k in ks {
            let t = &targets[k];
            let stratum = stratum_of(leaf(model, &t.stream));
            let ys = &fc.draws[k];
            let qs = band(ys).map_err(|e| format!("forecast of '{}' ending at {}: {e}", t.stream, t.measures.end()))?;
            let mean = ys.iter().sum::<f64>() / ys.len() as f64;
            let mut cells: Vec<String> = dims.iter()
                .map(|d| stratum.iter().find(|(dd, _)| dd == d).map(|(_, l)| l.clone()).unwrap_or_default())
                .collect();
            cells.push(fmt_time(t.measures.start()));
            cells.push(fmt_time(t.measures.end()));
            cells.push(t.horizon.to_string());
            cells.push(fmt_time(t.lead));
            cells.push(t.observed.map(|h| h.file.as_str().to_string()).unwrap_or_default());
            cells.push(t.observed.map(|h| h.position.to_string()).unwrap_or_default());
            cells.push(fmt_value(mean));
            cells.extend(qs.iter().map(|q| fmt_value(*q)));
            cells.push(ys.len().to_string());
            tsv += &(cells.join("\t") + "\n");
        }
        written.push(crate::fit::predict::write_tsv(out_dir, "forecast", source, &tsv)?);
    }

    // Joint samples: per draw, `FORECAST_SAMPLES_PER_DRAW` particles evenly
    // strided; each sample's values across every target are one particle's.
    let n = fc.n_particles;
    let per = FORECAST_SAMPLES_PER_DRAW.min(n);
    let mut tsv = String::from(
        "sample\tchain\tdraw\tcloud_row\tparticle\tstream\tt_start\tt_stop\thorizon\tvalue\n");
    let mut sample = 0usize;
    for (d, r) in draw_refs.iter().enumerate().take(fc.n_draws) {
        let (chain, draw) = match r.key {
            Some((c, i)) => (c.to_string(), i.to_string()),
            None => (String::new(), String::new()),
        };
        for j in 0..per {
            let p = j * n / per;
            for (k, t) in targets.iter().enumerate() {
                tsv += &format!("{sample}\t{chain}\t{draw}\t{}\t{p}\t{}\t{}\t{}\t{}\t{}\n", r.cloud_row, t.stream,
                    fmt_time(t.measures.start()), fmt_time(t.measures.end()), t.horizon,
                    fmt_value(fc.draws[k][d * n + p]));
            }
            sample += 1;
        }
    }
    let path = out_dir.join("forecast-samples.tsv");
    std::fs::write(&path, tsv).map_err(|e| format!("cannot write {}: {e}", path.display()))?;
    written.push(path);
    Ok(written)
}

/// `forecast.json`: what the forecast predicted and what conditioned it.
fn write_manifest(
    fc: &Forecast,
    job: &ForecastJob,
    record: &crate::run_meta::TrainingPartitionRecord,
    reach_as_written: &str,
    refused: &[crate::fit::forecast_plan::StreamRefusal],
    draw_refs: &[DrawRef],
) -> Result<PathBuf, String> {
    let point = |t: f64| serde_json::json!({
        "t": t,
        "date": job.calendar.date_of(t).map(ir::caltime::RenderedInstant::into_string),
    });
    let StateConditioning::Filtered { through, rows } = &fc.state;
    let targets: Vec<serde_json::Value> = fc.plan.targets().iter().map(|t| {
        let (kind, start) = match t.measures {
            Measures::Period(p) => ("period", p.start()),
            Measures::Instant(x) => ("instant", x),
        };
        serde_json::json!({
            "stream": t.stream,
            "measures": kind,
            "start": point(start),
            "end": point(t.measures.end()),
            "horizon": t.horizon,
            "lead": t.lead,
            "held_out": t.observed.map(|h| serde_json::json!({
                "file": h.file.as_str(), "position": h.position,
            })),
        })
    }).collect();
    let doc = serde_json::json!({
        "schema": FORECAST_SCHEMA,
        "run_id": job.run_id,
        "options": job.options,
        "calendar": job.calendar.to_json(),
        "mode": "fixed_origin",
        "treatment": "posterior",
        "method": job.posterior.method.map(|m| m.as_str()),
        "origin": point(fc.plan.origin()),
        "horizon": { "as_written": reach_as_written, "duration": fc.plan.horizon() },
        "parameter_set": record,
        "state_set": {
            "conditioning": "filtered",
            "through": point(*through),
            "streams": rows.iter().map(|p| serde_json::json!({
                "stream": p.stream, "rows": p.rows, "digest": p.digest.to_hex(),
            })).collect::<Vec<_>>(),
        },
        "target_sources": fc.plan.sources().iter()
            .map(|(s, src)| serde_json::json!({ "stream": s, "source": src.describe() }))
            .collect::<Vec<_>>(),
        "refused": refused.iter().map(|r| r.to_string()).collect::<Vec<_>>(),
        "n_draws": fc.n_draws,
        // The posterior draws run, in the order of `forecast-samples.tsv`:
        // each one's row of draws.tsv and its (chain, draw) key there.
        "draws": draw_refs.iter().map(|r| serde_json::json!({
            "cloud_row": r.cloud_row,
            "chain": r.key.map(|k| k.0),
            "draw": r.key.map(|k| k.1),
        })).collect::<Vec<_>>(),
        "n_particles": fc.n_particles,
        "binomial": fc.binomial,
        "samples_per_draw": FORECAST_SAMPLES_PER_DRAW.min(fc.n_particles),
        "targets": targets,
    });
    let path = job.out_dir.join("forecast.json");
    let text = serde_json::to_string_pretty(&doc).map_err(|e| format!("serializing forecast.json: {e}"))?;
    std::fs::write(&path, text).map_err(|e| format!("cannot write {}: {e}", path.display()))?;
    Ok(path)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fit::forecast_driver::ForecastFailure;

    /// A failure names the posterior draw by its row of draws.tsv and its
    /// (chain, draw) key, not by its place among the parameter sets the
    /// driver was handed.
    #[test]
    fn a_failure_names_the_posterior_draw_by_its_key() {
        let refs = [DrawRef { cloud_row: 0, key: Some((0, 3)) }, DrawRef { cloud_row: 40, key: Some((1, 7)) }];
        let f = ForecastFailure { draw: Some(1), time: Some(26.0), cause: "particle 4 died".into() };
        let msg = f.describe(|d| refs[d].describe());
        assert!(msg.contains("posterior draw (chain 1, draw 7), row 40 of draws.tsv at t = 26"), "{msg}");
        let unkeyed = DrawRef { cloud_row: 12, key: None };
        assert_eq!(unkeyed.describe(), "posterior draw at row 12 of draws.tsv");
    }

    /// The fit's sampler is the leaf's record; a leaf written before the
    /// record takes `fit run --binomial` from its command line, else the
    /// method as fit.toml wrote it.
    #[test]
    fn the_fits_sampler_is_read_from_the_leaf() {
        use sim::rng::BinomialAlgorithm::{Btpe, Btrs};
        let leaf = |json: &str| {
            let dir = crate::test_support::unique_temp_dir("fit_binomial");
            std::fs::create_dir_all(&dir).unwrap();
            std::fs::write(dir.join("run.json"), json).unwrap();
            dir
        };
        let recorded = leaf(r#"{"inputs": {"binomial": "btrs"}, "provenance": {"argv": ["camdl"]}}"#);
        assert_eq!(fit_binomial(&recorded, Some(Btpe)).unwrap(), Btrs, "the record wins over fit.toml");
        let flag = leaf(r#"{"inputs": {}, "provenance": {"argv": ["camdl", "fit", "run", "f.toml", "--binomial", "btrs"]}}"#);
        assert_eq!(fit_binomial(&flag, Some(Btpe)).unwrap(), Btrs);
        let eq = leaf(r#"{"inputs": {}, "provenance": {"argv": ["camdl", "fit", "run", "--binomial=btrs"]}}"#);
        assert_eq!(fit_binomial(&eq, Some(Btpe)).unwrap(), Btrs);
        let plain = leaf(r#"{"inputs": {}, "provenance": {"argv": ["camdl", "fit", "run"]}}"#);
        assert_eq!(fit_binomial(&plain, Some(Btrs)).unwrap(), Btrs, "fit.toml's method");
        assert!(fit_binomial(&plain, None).is_err());
        let bad = leaf(r#"{"inputs": {"binomial": "fast"}}"#);
        assert!(fit_binomial(&bad, Some(Btpe)).is_err());
    }
}
