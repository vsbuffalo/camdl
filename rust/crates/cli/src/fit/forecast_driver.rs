//! The fixed-origin forecast driver (`docs/dev/temporal-architecture.md` §4.4;
//! D39, D53, D54, D57).
//!
//! For each posterior draw it runs the existing bootstrap particle filter over
//! the training rows plus the plan's targets, each appended to its stream as a
//! declared period with no value, and reads each stream's predictive draws
//! from the observation model at each target's end. Rows ending after the
//! origin are never handed to the filter, so nothing after the origin is
//! assimilated. A stop after the origin holds no observed value, so the filter
//! does not assimilate there and skips the resampling call altogether (D72):
//! no particle is copied into another's slot after the origin, which is what
//! makes particle `i`'s draws across targets one joint sample.
//!
//! No draw is dropped (D53): a filter that fails, collapses, or loses a
//! particle after the origin fails the forecast with the draw, the time and
//! the cause.

use std::sync::Arc;

use sim::inference::multi_stream_obs::{MultiStreamObsModel, StreamSpec};
use sim::inference::obs_attempt::ObsCellState;
use sim::inference::traits::{ObservationModel, SMCConfig};
use sim::inference::{bootstrap_filter_recording_at, BoundObs, ChainBinomialProcess, Period, PrequentialStops, StreamTimes};

use super::forecast_plan::ForecastPlan;
use super::observed_rows::Measures;
use super::runner::{ObsStream, TIME_TIE_EPS};
use super::training_partition::{row_digest, RowPrefix};

/// The rows the state at the origin was conditioned on (§2.4). Built only
/// here, from the rows handed to the filter: no caller can claim a
/// conditioning that did not happen.
#[derive(Debug, Clone, PartialEq)]
pub(crate) enum StateConditioning {
    /// A particle filter ran through these rows of each stream, ending at
    /// `through`.
    Filtered { through: f64, rows: Vec<RowPrefix> },
}

/// What the driver computed.
#[derive(Debug)]
pub(crate) struct Forecast {
    pub plan: ForecastPlan,
    pub state: StateConditioning,
    pub n_draws: usize,
    pub n_particles: usize,
    /// The binomial sampler the filters drew with.
    pub binomial: sim::rng::BinomialAlgorithm,
    /// `draws[k][s]`: the predictive draw of the observation at target `k`
    /// (the plan's order), sample `s = draw · n_particles + particle`. One `s`
    /// across targets is one particle's trajectory (D54).
    pub draws: Vec<Vec<f64>>,
    /// The same layout, holding each particle's expected observation
    /// `E[y | x, θ]` at the target, when the caller asked for it.
    #[cfg_attr(not(test), allow(dead_code))] // the command does not ask for it
    pub expected: Option<Vec<Vec<f64>>>,
}

/// Why a forecast failed (D53).
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct ForecastFailure {
    /// The posterior draw (its index in the cloud handed to the driver), or
    /// `None` for a failure before any draw ran.
    pub draw: Option<usize>,
    /// The model time of the failure, when one is known.
    pub time: Option<f64>,
    pub cause: String,
}

impl ForecastFailure {
    /// The failure in a sentence, naming the draw as `name_draw` says: the
    /// driver knows only a draw's index among the parameter sets it was
    /// handed, and the caller knows which posterior draw that is.
    pub(crate) fn describe(&self, name_draw: impl Fn(usize) -> String) -> String {
        let mut s = String::from("the forecast failed");
        if let Some(d) = self.draw {
            s += &format!(" on {}", name_draw(d));
        }
        if let Some(t) = self.time {
            s += &format!(" at t = {t}");
        }
        s + &format!(": {}. No draw is dropped from a forecast, so none was written.", self.cause)
    }
}

impl std::fmt::Display for ForecastFailure {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.describe(|d| format!("parameter set {d} of those handed to the driver")))
    }
}

fn setup(cause: impl Into<String>) -> ForecastFailure {
    ForecastFailure { draw: None, time: None, cause: cause.into() }
}

/// The driver's settings.
#[derive(Debug, Clone, Copy)]
pub(crate) struct DriverConfig {
    pub n_particles: usize,
    pub dt: f64,
    pub base_seed: u64,
    /// Also record each particle's expected observation at each target.
    /// Records the filter's ancestry, whose memory grows with particles ×
    /// compartments × steps; tests read it, the command does not.
    pub record_expected: bool,
    /// The binomial sampler the propagation draws use (gh#747): the one the
    /// fit's method ran, so a forecast continues the process the fit
    /// filtered with the same draws.
    pub binomial: sim::rng::BinomialAlgorithm,
}

/// The seed of draw `d`, for every filter `fit predict` runs per posterior
/// draw (the one-step band and the forecast): reproducible from the base seed
/// and the draw's index alone, and keyed on the index so a skipped draw does
/// not shift the seeds after it.
pub(crate) fn draw_seed(base_seed: u64, draw: usize) -> u64 {
    base_seed ^ (0x9E37_79B9_7F4A_7C15u64.wrapping_mul(draw as u64 + 1))
}

/// Run the forecast `plan` describes. `training` is every bound stream, each
/// cut to its training prefix; `compiled` is the fitted model with its run
/// extent reaching the plan's latest target end (D49); `param_sets` holds one
/// parameter vector per posterior draw.
pub(crate) fn fixed_origin_forecast(
    plan: ForecastPlan,
    training: &[ObsStream],
    compiled: Arc<sim::CompiledModel>,
    param_sets: &[Vec<f64>],
    config: DriverConfig,
) -> Result<Forecast, ForecastFailure> {
    let origin = plan.origin();
    let state = conditioning(&plan, training)?;
    let specs = specs_with_targets(&plan, training)?;
    let names: Vec<String> = training.iter().map(|s| s.name.clone()).collect();
    let t_start = compiled.model.simulation.t_start;
    let (bound, _report) = BoundObs::bind(t_start, specs)
        .map_err(|r| setup(format!("the training rows and targets do not bind:\n{}", r.render())))?;
    let obs_model = MultiStreamObsModel::new(bound, compiled.clone())
        .map_err(|e| setup(format!("observation model construction failed: {e:?}")))?;

    // Each target's (stream, union index): read by the stream's own schedule,
    // never by a sibling's value at the same time.
    let n_union = obs_model.n_observations();
    let union_times: Vec<f64> = (0..n_union).map(|i| obs_model.obs_time(i)).collect();
    let at: Vec<(usize, usize)> = plan.targets().iter()
        .map(|t| declared_read(&obs_model, &names, &t.stream, t.measures.end()))
        .collect::<Result<_, _>>()?;
    let after_origin: Vec<usize> = (0..n_union).filter(|&i| union_times[i] > origin + TIME_TIE_EPS).collect();

    crate::fit::methods::check_model_capabilities(
        crate::run_meta::InferenceBackend::ChainBinomial, &compiled).map_err(setup)?;
    let process = ChainBinomialProcess::new(compiled.clone());
    let smc = SMCConfig {
        n_particles: config.n_particles,
        dt: config.dt,
        t_start,
        skip_first_obs_from_loglik: false,
        record_ancestry: config.record_expected,
        record_prequential: true,
        record_predictions: false,
        max_substeps: sim::inference::degeneracy::ITER_BUDGET,
        binomial: config.binomial,
    };

    // D84: keep the prequential record only where it is read below, the stops
    // after the origin (the death check) and the targets.
    let record_at = PrequentialStops::only(after_origin.iter().copied().chain(at.iter().map(|&(_, ui)| ui)));

    let n = config.n_particles;
    let mut draws: Vec<Vec<f64>> = vec![Vec::with_capacity(param_sets.len() * n); at.len()];
    let mut expected: Option<Vec<Vec<f64>>> =
        config.record_expected.then(|| vec![Vec::with_capacity(param_sets.len() * n); at.len()]);
    for (d, params) in param_sets.iter().enumerate() {
        let fail = |time: Option<f64>, cause: String| ForecastFailure { draw: Some(d), time, cause };
        let result = bootstrap_filter_recording_at(&process, &obs_model, params, &smc, draw_seed(config.base_seed, d), &record_at)
            .map_err(|e| {
                let time = match &e {
                    sim::SimError::PFDegenerate { obs_window, .. } => union_times.get(*obs_window).copied(),
                    _ => None,
                };
                fail(time, format!("the particle filter stopped: {e}"))
            })?;
        if let Some(c) = &result.collapse {
            return Err(fail(union_times.get(c.obs_index).copied(), format!("the filter collapsed: {c}")));
        }
        let preq = result.prequential
            .ok_or_else(|| fail(None, "internal error: no prequential record".into()))?;
        // After the origin every cell is a row with no value, whose
        // log-likelihood is 0 for a live particle; a dead particle's
        // log-weight is −∞. So any −∞ after the origin is a death.
        for &i in &after_origin {
            // An unselected stop's record is empty, and an empty record would
            // pass this check without looking: refuse rather than go blind.
            if preq.log_liks[i].len() != n {
                return Err(fail(Some(union_times[i]), format!(
                    "internal error: the death check has {} of {n} particles' weights at this stop",
                    preq.log_liks[i].len())));
            }
            if let Some(p) = preq.log_liks[i].iter().position(|w| !w.is_finite()) {
                return Err(fail(Some(union_times[i]), format!(
                    "particle {p} died after the forecast origin t = {origin} (a numerical \
                     failure of the step, such as a negative count, not epidemic extinction)")));
            }
        }
        for (k, &(si, ui)) in at.iter().enumerate() {
            let ys = &preq.per_stream_samples[ui][si];
            if let Some(p) = ys.iter().position(|y| !y.is_finite()) {
                return Err(fail(Some(union_times[ui]), format!(
                    "particle {p}'s draw for stream '{}' is not finite ({})", names[si], ys[p])));
            }
            draws[k].extend_from_slice(ys);
            if let Some(e) = expected.as_mut() {
                let anc = result.ancestry.as_ref()
                    .ok_or_else(|| fail(None, "internal error: no ancestry record".into()))?;
                e[k].extend(anc.projections[ui].iter().map(|per_stream| per_stream[si]));
            }
        }
    }
    Ok(Forecast {
        plan, state, n_draws: param_sets.len(), n_particles: n, binomial: config.binomial, draws, expected,
    })
}

/// Where the driver may read `stream` at `end`: its index and the union index
/// of a declared period of that stream ending at `end` that carries no value.
/// Anything else is refused: a stream's running total resets only where a
/// declared period starts, so a reading at a time that ends no declared period
/// would include flow from before the period meant (spec §4.4), and a reading
/// at an observed row would predict what the filter already assimilated.
pub(crate) fn declared_read(
    obs_model: &MultiStreamObsModel,
    names: &[String],
    stream: &str,
    end: f64,
) -> Result<(usize, usize), ForecastFailure> {
    let si = names.iter().position(|n| n == stream)
        .ok_or_else(|| setup(format!("stream '{stream}' is not bound")))?;
    let ui = (0..obs_model.n_observations())
        .find(|&i| (obs_model.obs_time(i) - end).abs() <= TIME_TIE_EPS)
        .filter(|&i| obs_model.cell_state(si, i) == ObsCellState::Hole)
        .ok_or_else(|| setup(format!(
            "a prediction of stream '{stream}' at t = {end} was requested, but no declared \
             period of that stream with no value ends there; only the forecast plan \
             declares the periods a forecast reads")))?;
    Ok((si, ui))
}

/// The state set: the rows of each stream handed to the filter, which must be
/// the training prefix the parameter set records, and end at or before the
/// origin.
fn conditioning(plan: &ForecastPlan, training: &[ObsStream]) -> Result<StateConditioning, ForecastFailure> {
    let mut rows = Vec::with_capacity(training.len());
    let mut through = f64::NEG_INFINITY;
    for s in training {
        let digest = row_digest(&s.rows).map_err(setup)?;
        let recorded = plan.parameters().streams().iter().find(|p| p.stream == s.name);
        if recorded.map(|p| (p.rows, p.digest)) != Some((s.rows.len(), digest)) {
            return Err(setup(format!(
                "internal error: the rows of stream '{}' handed to the filter are not the \
                 training rows the fit recorded", s.name)));
        }
        for r in &s.rows {
            through = through.max(r.measures.end());
        }
        rows.push(RowPrefix { stream: s.name.clone(), rows: s.rows.len(), digest });
    }
    if through > plan.origin() + TIME_TIE_EPS {
        return Err(setup(format!(
            "internal error: a conditioning row ends at {through}, after the origin {}",
            plan.origin())));
    }
    Ok(StateConditioning::Filtered { through, rows })
}

/// Each training stream's spec with its targets appended as rows with no
/// value.
fn specs_with_targets(plan: &ForecastPlan, training: &[ObsStream]) -> Result<Vec<StreamSpec>, ForecastFailure> {
    let mut specs = crate::fit::runner::stream_specs_from_obs_streams(training);
    for spec in &mut specs {
        let name = spec.ir_model.name.clone();
        // An empty aux list means the stream reads no data column; `bind`
        // sizes it. A sized one gains an empty entry per target.
        let sized_aux = !spec.aux.is_empty();
        for t in plan.targets_of(&name) {
            match (&mut spec.times, t.measures) {
                (StreamTimes::Intervals(ps), Measures::Period(p)) => {
                    ps.push(Period::new(p.start(), p.stop()).map_err(setup)?);
                }
                (StreamTimes::Instants(ts), Measures::Instant(x)) => ts.push(x),
                _ => return Err(setup(format!(
                    "internal error: a target of '{name}' does not match what its rows measure"))),
            }
            spec.observations.push(None);
            if sized_aux {
                spec.aux.push(Vec::new());
            }
        }
    }
    Ok(specs)
}

/// Raise `model`'s run extent to the plan's latest target end (D49), so the
/// filter admits the target rows and the effect plan is built to that end.
/// Nothing else is extended: a schedule with its own end keeps it.
pub(crate) fn extend_run(model: &mut ir::Model, plan: &ForecastPlan) {
    let end = plan.run_end();
    if end > model.simulation.t_end {
        model.simulation.t_end = end;
    }
}

#[cfg(test)]
#[path = "forecast_driver_tests.rs"]
mod tests;
