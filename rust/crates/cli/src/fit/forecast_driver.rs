//! The fixed-origin forecast driver (`docs/engineering/forecasting.md` §3.4;
//! D39, D53, D54, D57).
//!
//! For each posterior draw it runs the existing bootstrap particle filter over
//! the training rows plus the plan's targets, each appended to its stream as a
//! declared period with no value the filter assimilates, and reads each
//! stream's predictive draws from the observation model at each target's end.
//! A target with a held-out value carries it as a held-out cell (D79), which
//! only the filter's scoring likelihood reads. Rows ending after the origin
//! are never handed to the filter as observations, so nothing after the origin
//! is assimilated. A stop after the origin holds no observed value, so the filter
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

/// The rows the latent state at the forecast origin was conditioned on
/// (`docs/engineering/forecasting.md` §1.3).
///
/// Its field is private to this module, and its one constructor is
/// [`conditioning`], which builds it from the rows handed to the particle
/// filter after checking that they are the training rows the fit recorded and
/// that none ends after the origin. A value of this type outside this module
/// therefore always describes a filter run that happened, on those rows.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct StateConditioning(Kind);

#[derive(Debug, Clone, PartialEq)]
enum Kind {
    /// A particle filter ran through these rows of each stream, ending at
    /// `through`.
    Filtered { through: f64, rows: Vec<RowPrefix> },
}

impl StateConditioning {
    /// The end of the last row the filter ran through, in model time.
    pub(crate) fn through(&self) -> f64 {
        let Kind::Filtered { through, .. } = &self.0;
        *through
    }

    /// The rows of each stream the filter ran through: per stream, the
    /// number of its first rows and their digest.
    pub(crate) fn rows(&self) -> &[RowPrefix] {
        let Kind::Filtered { rows, .. } = &self.0;
        rows
    }
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
    /// Per target, in the plan's order: each sample's log-likelihood of the
    /// target's held-out value (D79), `ℓ[s] = log p(y_held_out | x_s, θ_m)`,
    /// in the layout of `draws`; `None` for a target with no held-out value.
    /// Evaluated by the filter's own likelihood at the target's end, before
    /// anything could be assimilated there (D56); it enters no weight, so no
    /// other field of the forecast depends on the held-out values.
    pub held_out_log_liks: Vec<Option<Vec<f64>>>,
    /// Per target, why a held-out row it names is not scored
    /// ([`HeldOutValues`]): never a reason the prediction was refused.
    #[cfg_attr(not(test), allow(dead_code))] // read by Stage 6's score artifact
    pub not_scored: Vec<Option<String>>,
    /// The same layout, holding each particle's expected observation
    /// `E[y | x, θ]` at the target, when the caller asked for it.
    #[cfg_attr(not(test), allow(dead_code))] // the command does not ask for it
    pub expected: Option<Vec<Vec<f64>>>,
    /// Per draw, what its filter did at the stops at or before the origin:
    /// the record two scenario arms must share (D50). Empty unless
    /// [`DriverConfig::trace_pre_origin`] asked for it.
    pub pre_origin: Vec<PreOriginTrace>,
    /// Per draw, its filter's last assimilation (D85): `None` when no stop
    /// before the origin held an observed value.
    pub last_assimilation: Vec<Option<LastAssimilation>>,
    /// Per draw, the particles' states and log-weights at the last stop at
    /// or before the origin, before resampling, when the caller asked for
    /// the expected observations (they come from the same ancestry record).
    #[cfg_attr(not(test), allow(dead_code))] // read by the scenario-arm test
    pub origin_states: Option<Vec<(Vec<Vec<f64>>, Vec<f64>)>>,
}

/// A draw's filter at its last assimilation, the last stop at or before the
/// origin that held an observed value (D85): the effective sample size of
/// the particle weights there, before resampling, and its time. It
/// characterises the Monte Carlo reliability of the state the forecast
/// starts from; it says nothing of how many distinct ancestors the particles
/// have, and nothing of the forecast's accuracy.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct LastAssimilation {
    pub time: f64,
    pub ess: f64,
}

/// One draw's filter at the stops at or before the forecast origin: the
/// log-likelihood increment and the effective sample size at each, and a
/// SHA-256 over every particle's per-stream log-likelihood and predictive
/// draw there. Two arms whose models agree before the origin, run with the
/// same seed, have bit-identical traces; any difference means an arm changed
/// what was filtered.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct PreOriginTrace {
    pub times: Vec<f64>,
    pub ll_increments: Vec<f64>,
    pub ess: Vec<f64>,
    pub particles: [u8; 32],
}

impl PreOriginTrace {
    /// Where `other` first departs from this trace: the time of the first
    /// stop whose log-likelihood increment or ESS differs bitwise, and which;
    /// or, when those agree everywhere, the particle record as a whole.
    pub(crate) fn first_difference(&self, other: &PreOriginTrace) -> Option<(Option<f64>, &'static str)> {
        let same = |a: f64, b: f64| a.to_bits() == b.to_bits();
        if self.times.len() != other.times.len() {
            return Some((None, "the number of stops before the origin"));
        }
        for i in 0..self.times.len() {
            if !same(self.ll_increments[i], other.ll_increments[i]) {
                return Some((Some(self.times[i]), "the log-likelihood increment"));
            }
            if !same(self.ess[i], other.ess[i]) {
                return Some((Some(self.times[i]), "the effective sample size"));
            }
        }
        (self.particles != other.particles)
            .then_some((None, "the particles' log-likelihoods or predictive draws"))
    }
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
    /// Record every stop and build each draw's [`PreOriginTrace`], which
    /// scenario arms are compared by. Only a forecast with more than one arm
    /// needs it; without it the filter keeps its record only where the
    /// forecast reads it (D84).
    pub trace_pre_origin: bool,
    /// The binomial sampler the propagation draws use (gh#747): the one the
    /// fit's method ran, so a forecast continues the process the fit
    /// filtered with the same draws.
    pub binomial: sim::rng::BinomialAlgorithm,
}

/// The seed of draw `d`, for every filter `camdl predict` runs per posterior
/// draw (the one-step band and the forecast): reproducible from the base seed
/// and the draw's index alone, and keyed on the index so a skipped draw does
/// not shift the seeds after it.
pub(crate) fn draw_seed(base_seed: u64, draw: usize) -> u64 {
    base_seed ^ (0x9E37_79B9_7F4A_7C15u64.wrapping_mul(draw as u64 + 1))
}

/// Each target's held-out value, in the plan's order (D79): the value of the
/// held-out row the target names, bound as an
/// [`ObsCell::HeldOut`](sim::inference::ObsCell::HeldOut), which the filter
/// never assimilates and only its scoring likelihood reads. A target is not
/// scored, and records why, when its row is `NA`, when the binder's checks
/// refuse its value against the target's design (a count above its trials),
/// or when the forecast is prospective and the stream reads a design: a
/// prospective forecast is not scored. None of this can refuse a prediction.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct HeldOutValues {
    values: Vec<Option<f64>>,
    not_scored: Vec<Option<String>>,
}

impl HeldOutValues {
    /// No held-out value at any target: a forecast that is not scored.
    #[cfg(test)]
    pub(crate) fn none(plan: &ForecastPlan) -> Self {
        let n = plan.targets().len();
        HeldOutValues { values: vec![None; n], not_scored: vec![None; n] }
    }

    /// The values of the rows `plan`'s targets name, read from `held_out`
    /// (each stream's rows after its training prefix, by file), screened
    /// against each target's design by the binder's checks; `model` holds
    /// the streams' observation models.
    pub(crate) fn from_rows(
        plan: &ForecastPlan,
        held_out: &std::collections::BTreeMap<String, Vec<super::forecast_plan::HeldOut>>,
        model: &ir::Model,
    ) -> Result<Self, String> {
        let mut rows = Vec::with_capacity(plan.targets().len());
        for t in plan.targets() {
            let Some(at) = t.observed else {
                rows.push(None);
                continue;
            };
            let row = held_out.get(&t.stream).and_then(|rows| rows.iter()
                .find(|h| h.file == at.file && h.row.position == at.position))
                .ok_or_else(|| format!(
                    "internal error: target of '{}' ending at {} names its {}, which was not read",
                    t.stream, t.measures.end(), at.describe()))?;
            rows.push(Some((row.row.clone(), at.describe())));
        }
        Self::screened(plan, rows, model)
    }

    /// The values of the outcome rows that measure each target's period or
    /// instant, read from `outcomes` (each leaf's rows after the ones the
    /// prediction conditioned on), screened as [`Self::from_rows`] screens
    /// them. Two rows measuring one target are refused.
    pub(crate) fn from_outcomes(
        plan: &ForecastPlan,
        outcomes: &crate::fit::scoring::OutcomeBinding,
        model: &ir::Model,
    ) -> Result<Self, String> {
        let mut rows = Vec::with_capacity(plan.targets().len());
        for t in plan.targets() {
            let matching: Vec<&super::observed_rows::ObservedRow> = outcomes
                .after_conditioning(&t.stream)
                .iter()
                .filter(|r| super::forecast_plan::same_span(&r.measures, &t.measures))
                .collect();
            match matching[..] {
                [] => rows.push(None),
                [r] => rows.push(Some((r.clone(), format!("outcome row {} of '{}'", r.position + 1, t.stream)))),
                [a, b, ..] => return Err(format!(
                    "the outcome rows {} and {} of '{}' both measure the target ending at {}; each \
                     target is scored against one row",
                    a.position + 1, b.position + 1, t.stream, t.measures.end())),
            }
        }
        Self::screened(plan, rows, model)
    }

    /// Each target's row, as `(row, how a message names it)`, screened.
    fn screened(
        plan: &ForecastPlan,
        rows: Vec<Option<(super::observed_rows::ObservedRow, String)>>,
        model: &ir::Model,
    ) -> Result<Self, String> {
        use super::forecast_plan::DesignMode;
        use sim::inference::ObsCell;
        let mut values = Vec::with_capacity(plan.targets().len());
        let mut not_scored = Vec::with_capacity(plan.targets().len());
        for (t, row) in plan.targets().iter().zip(rows) {
            let Some((row, named)) = row else {
                values.push(None);
                not_scored.push(None);
                continue;
            };
            let obs = model.observations.iter().find(|o| o.name == t.stream)
                .ok_or_else(|| format!("internal error: no observation block '{}'", t.stream))?;
            let reads_design = !crate::pfilter::stream_aux_columns(obs).is_empty();
            let screened: Result<f64, String> = match row.cell {
                None => Err(format!("its {named} is NA")),
                Some(ObsCell::HeldOut(_)) => return Err(format!(
                    "internal error: the {named} of '{}' was read as a held-out cell; a row read \
                     from a file holds an observed value or NA", t.stream)),
                Some(ObsCell::Scalar(_)) if reads_design && plan.design_mode() == Some(DesignMode::Prospective) =>
                    Err("a prospective forecast is not scored: the held-out value was realised \
                         under its own design, not the design supplied at the origin; score this \
                         stream with --realised-design".to_string()),
                Some(ObsCell::Scalar(v)) => match sim::inference::multi_stream_obs::held_out_value_refusal(
                    obs, t.measures.end(), v, &t.design)
                {
                    None => Ok(v),
                    Some(why) => Err(format!("its {named} cannot be scored: {why}")),
                },
            };
            match screened {
                Ok(v) => { values.push(Some(v)); not_scored.push(None); }
                Err(why) => { values.push(None); not_scored.push(Some(why)); }
            }
        }
        Ok(HeldOutValues { values, not_scored })
    }

    /// The value bound at target `k`, when it is scored.
    pub(crate) fn at(&self, k: usize) -> Option<f64> {
        self.values[k]
    }

    /// Why target `k`'s row is not scored, when it names one that is not.
    pub(crate) fn not_scored(&self, k: usize) -> Option<&str> {
        self.not_scored[k].as_deref()
    }

    fn len(&self) -> usize {
        self.values.len()
    }
}

/// Run the forecast `plan` describes. `training` is every bound stream, each
/// cut to its training prefix; `held_out` holds each target's held-out value
/// ([`HeldOutValues::none`] for a forecast that is not scored); `compiled` is
/// the fitted model with its run extent reaching the plan's latest target end
/// (D49); `param_sets` holds one parameter vector per posterior draw.
pub(crate) fn fixed_origin_forecast(
    plan: ForecastPlan,
    training: &[ObsStream],
    held_out: &HeldOutValues,
    compiled: Arc<sim::CompiledModel>,
    param_sets: &[Vec<f64>],
    config: DriverConfig,
) -> Result<Forecast, ForecastFailure> {
    let origin = plan.origin();
    if held_out.len() != plan.targets().len() {
        return Err(setup(format!(
            "internal error: {} held-out values for {} targets", held_out.len(), plan.targets().len())));
    }
    let state = conditioning(&plan, training)?;
    let specs = specs_with_targets(&plan, training, held_out)?;
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
        record_ancestry: config.record_expected, path_recording: Default::default(),
        record_prequential: true,
        record_predictions: false,
        record_stream_means: false,
        max_substeps: sim::inference::degeneracy::ITER_BUDGET,
        binomial: config.binomial,
    };

    // D84: keep the prequential record only where it is read below: the
    // stops after the origin (the death check) and the targets, and, when
    // scenario arms are to be compared, every stop at or before the origin
    // too, which feeds each draw's [`PreOriginTrace`].
    let record_at = match config.trace_pre_origin {
        true => PrequentialStops::every(),
        false => PrequentialStops::only(after_origin.iter().copied().chain(at.iter().map(|&(_, ui)| ui))),
    };

    let n = config.n_particles;
    let mut draws: Vec<Vec<f64>> = vec![Vec::with_capacity(param_sets.len() * n); at.len()];
    let mut held_out_log_liks: Vec<Option<Vec<f64>>> = (0..at.len())
        .map(|k| held_out.at(k).map(|_| Vec::with_capacity(param_sets.len() * n)))
        .collect();
    let mut expected: Option<Vec<Vec<f64>>> =
        config.record_expected.then(|| vec![Vec::with_capacity(param_sets.len() * n); at.len()]);
    let before_origin: Vec<usize> = (0..n_union).filter(|i| !after_origin.contains(i)).collect();
    let mut pre_origin: Vec<PreOriginTrace> = Vec::with_capacity(param_sets.len());
    let mut last_assimilation: Vec<Option<LastAssimilation>> = Vec::with_capacity(param_sets.len());
    let mut origin_states: Option<Vec<(Vec<Vec<f64>>, Vec<f64>)>> =
        config.record_expected.then(|| Vec::with_capacity(param_sets.len()));
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
        // After the origin every cell is a row with no value the filter
        // assimilates (a hole or a held-out value), whose joint
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
            if let Some(ll) = held_out_log_liks[k].as_mut() {
                // The stop holds this target's held-out value, so the filter
                // kept its scoring likelihood there for every particle.
                let lls = preq.per_stream_held_out_log_liks.get(ui).and_then(|v| v.get(si))
                    .filter(|v| v.len() == n)
                    .ok_or_else(|| fail(Some(union_times[ui]), format!(
                        "internal error: the filter kept no held-out log-likelihood for stream '{}' \
                         at this target", names[si])))?;
                if let Some(p) = lls.iter().position(|l| l.is_nan()) {
                    return Err(fail(Some(union_times[ui]), format!(
                        "particle {p}'s log-likelihood of the held-out value of stream '{}' is NaN",
                        names[si])));
                }
                ll.extend_from_slice(lls);
            }
            if let Some(e) = expected.as_mut() {
                let anc = result.ancestry.as_ref()
                    .ok_or_else(|| fail(None, "internal error: no ancestry record".into()))?;
                e[k].extend(anc.projections[ui].iter().map(|per_stream| per_stream[si]));
            }
        }
        if config.trace_pre_origin {
            pre_origin.push(pre_origin_trace(&before_origin, &union_times, &result.ll_increments,
                &result.ess_trace, &preq, n).map_err(|e| fail(None, e))?);
        }
        // `ess_trace` holds each stop's ESS after the weights are set and
        // before the resampling call.
        last_assimilation.push(before_origin.iter().rev().find(|&&i| obs_model.assimilates_at(i))
            .map(|&i| LastAssimilation { time: union_times[i], ess: result.ess_trace[i] }));
        if let Some(o) = origin_states.as_mut() {
            let anc = result.ancestry.as_ref()
                .ok_or_else(|| fail(None, "internal error: no ancestry record".into()))?;
            let last = *before_origin.last()
                .ok_or_else(|| fail(None, "internal error: no stop at or before the origin".into()))?;
            o.push((anc.states[last].clone(), anc.log_weights[last].clone()));
        }
    }
    Ok(Forecast {
        plan, state, n_draws: param_sets.len(), n_particles: n, binomial: config.binomial, draws,
        held_out_log_liks, not_scored: held_out.not_scored.clone(), expected, pre_origin,
        last_assimilation, origin_states,
    })
}

/// The [`PreOriginTrace`] of one draw's filter over the stops `before`.
fn pre_origin_trace(
    before: &[usize],
    union_times: &[f64],
    ll_increments: &[f64],
    ess_trace: &[f64],
    preq: &sim::inference::particle_filter::PrequentialRecorded,
    n_particles: usize,
) -> Result<PreOriginTrace, String> {
    use sha2::{Digest, Sha256};
    let mut h = Sha256::new();
    for &i in before {
        // A stop the filter kept no record of would hash as nothing, and the
        // check on the arms would pass vacuously there.
        if preq.log_liks[i].len() != n_particles {
            return Err(format!(
                "internal error: the filter kept no per-particle record at stop {i} before the \
                 origin, which the check on scenario arms reads"));
        }
        for v in preq.log_liks[i].iter()
            .chain(preq.per_stream_log_liks[i].iter().flatten())
            .chain(preq.per_stream_samples[i].iter().flatten())
        {
            h.update(v.to_bits().to_le_bytes());
        }
    }
    Ok(PreOriginTrace {
        times: before.iter().map(|&i| union_times[i]).collect(),
        ll_increments: before.iter().map(|&i| ll_increments[i]).collect(),
        ess: before.iter().map(|&i| ess_trace[i]).collect(),
        particles: h.finalize().into(),
    })
}

/// Where the driver may read `stream` at `end`: its index and the union index
/// of a declared period of that stream ending at `end` that carries no value.
/// Anything else is refused: a stream's running total resets only where a
/// declared period starts, so a reading at a time that ends no declared period
/// would include flow from before the period meant (§1.3), and a reading
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
        // A declared period carrying no value the filter assimilates: a hole,
        // or a held-out value (D79), which the filter treats as one.
        .filter(|&i| match obs_model.cell_state(si, i) {
            ObsCellState::Hole | ObsCellState::HeldOut { .. } => true,
            ObsCellState::Scored { .. } | ObsCellState::NotScheduled => false,
        })
        .ok_or_else(|| setup(format!(
            "a prediction of stream '{stream}' at t = {end} was requested, but no declared \
             period of that stream with no value ends there; only the forecast plan \
             declares the periods a forecast reads")))?;
    Ok((si, ui))
}

/// The conditioning rows: the rows of each stream handed to the filter, which
/// must be the training prefix the training partition records, and end at or
/// before the origin.
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
    Ok(StateConditioning(Kind::Filtered { through, rows }))
}

/// Each training stream's spec with its targets appended as rows carrying no
/// value the filter assimilates: a hole, or the target's held-out value as a
/// held-out cell (D79).
fn specs_with_targets(
    plan: &ForecastPlan,
    training: &[ObsStream],
    held_out: &HeldOutValues,
) -> Result<Vec<StreamSpec>, ForecastFailure> {
    let mut specs = crate::fit::runner::stream_specs_from_obs_streams(training);
    for spec in &mut specs {
        let name = spec.ir_model.name.clone();
        // An empty aux list means the stream reads no data column; `bind`
        // sizes it. A stream that reads one gains, per target, the target's
        // design values (D81), which the observation model reads to draw the
        // prediction and to score a held-out value. They come from the plan,
        // never from a held-out outcome.
        if spec.aux.is_empty() && plan.targets_of(&name).any(|t| !t.design.is_empty()) {
            spec.aux = vec![Vec::new(); spec.observations.len()];
        }
        let sized_aux = !spec.aux.is_empty();
        for (k, t) in plan.targets().iter().enumerate().filter(|(_, t)| t.stream == name) {
            match (&mut spec.times, t.measures) {
                (StreamTimes::Intervals(ps), Measures::Period(p)) => {
                    ps.push(Period::new(p.start(), p.stop()).map_err(setup)?);
                }
                (StreamTimes::Instants(ts), Measures::Instant(x)) => ts.push(x),
                _ => return Err(setup(format!(
                    "internal error: a target of '{name}' does not match what its rows measure"))),
            }
            spec.observations.push(held_out.at(k).map(sim::inference::ObsCell::HeldOut));
            if sized_aux {
                spec.aux.push(t.design.clone());
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
