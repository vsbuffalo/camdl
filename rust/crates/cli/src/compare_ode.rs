//! The ODE half of `camdl compare`'s derive path (gh#312): the per-θ trace of
//! a fit that ran on the `ode` backend.
//!
//! A particle-filter fit's trace is derived by shelling out to `camdl
//! pfilter`. An ODE fit has no filter to shell out to, and needs none: given
//! θ the trajectory is a point, so the model's one-step-ahead predictive at
//! each observation is the observation density at that point, which the
//! fit's own likelihood already evaluates. This module therefore builds the
//! fit's likelihood exactly as `fit run` built it ([`FitRunConfig::build`],
//! the same model, data loader, observation binding and step), over the full
//! series rather than the training window, and records each observation's
//! term instead of summing them ([`sim::inference::ode_prequential_recorded`],
//! which shares its trajectory walk with `compute_ode_loglik`). The record goes
//! through the same trace builder the filter's does, so holes, per-stream
//! scores, CRPS, randomized PIT, intervals and the `--score-from` window behave
//! identically; only the basis stamp differs
//! ([`PredictiveBasis::OdeFreeForward`]).
//!
//! The posterior mixture over draws is formed by the caller with the same
//! combinator the filter path uses (`compare::combine_mixture`): per step,
//! `log (1/M) Σ_m p(y_t | x_t(θ_m))`.

use std::collections::BTreeMap;
use std::path::Path;

use score::trace::{Derivation, PredictiveBasis, PrequentialTrace, ThetaSource};

use crate::fit::handle::ResolvedFit;
use crate::fit::runner::FitRunConfig;

/// The fit's own ODE likelihood, bound to the full series, ready to score one
/// θ at a time.
pub(crate) struct OdeScorer {
    config: FitRunConfig,
    obs_model: sim::inference::MultiStreamObsModel,
    obs_times: Vec<f64>,
    dt: f64,
}

impl OdeScorer {
    /// Rebuild the fit's likelihood over `streams` (source → data file, the
    /// full series the trace scores: a `[data.holdout]` fit's merged files, a
    /// `holdout_after` fit's untruncated ones), on the model at `model_path`
    /// (the fit's archived IR when it has one).
    ///
    /// The holdout declaration is removed from the rebuilt problem — the fit
    /// applied it to train; scoring reads past it, and the caller stamps and
    /// verifies the held-out window as for a filter trace. Refused if the step
    /// or the ODE integrator the rebuilt problem resolves to is not the one
    /// the fit recorded (`fitted`): the score would come from a different
    /// trajectory than the one θ was estimated on.
    pub(crate) fn build(
        resolved: &ResolvedFit,
        model_path: &Path,
        streams: &indexmap::IndexMap<String, String>,
        fitted: &crate::fit::fitted_model::FittedSimulator,
        settings: crate::fit::fitted_model::FittedSettings,
    ) -> Result<Self, String> {
        let mut problem = resolved.config.problem.clone();
        // The settings the fit is scored under (the fit's, with compare's
        // flag folded in): the model is compiled under them.
        problem.config.allow_degenerate_rates = settings.allow_degenerate_rates;
        problem.config.allow_forcing_extrapolation = settings.allow_forcing_extrapolation;
        problem.compiled_ir = Some(model_path.to_string_lossy().into_owned());
        let data = problem.data.as_mut().ok_or_else(|| format!(
            "fit at {} has no [data] block — nothing to score",
            resolved.segment.display()))?;
        data.file = None;
        data.observations = streams.clone();
        data.holdout_after = None;
        data.holdout = None;

        let config = FitRunConfig::build(&problem, None, 1, 1, 1, 1.0, 1,
            crate::compare::DEFAULT_DERIVE_SEED, false)
            .map_err(|e| format!("rebuilding the ODE likelihood of the fit at {}: {e}",
                resolved.segment.display()))?;
        let dt = config.if2_config.dt;
        if dt != fitted.dt {
            return Err(format!(
                "the fit at {} ran at dt = {}, but its config now resolves dt = {dt}; \
                 the ODE scores would integrate a different trajectory than the one \
                 its θ was estimated on",
                resolved.segment.display(), fitted.dt));
        }
        if config.compiled.model.simulation.integrator != fitted.integrator {
            return Err(format!(
                "the fit at {} ran the ODE with integrator {:?}, but its model now \
                 declares {:?}; the ODE scores would integrate a different trajectory \
                 than the one its θ was estimated on",
                resolved.segment.display(), fitted.integrator,
                config.compiled.model.simulation.integrator));
        }
        let obs_model = config.build_obs_model();
        let obs_times: Vec<f64> = config.observations.iter().map(|o| o.time).collect();
        Ok(OdeScorer { config, obs_model, obs_times, dt })
    }

    /// The observation (union-axis) times the trace scores along.
    pub(crate) fn obs_times(&self) -> &[f64] {
        &self.obs_times
    }

    /// The trace of one θ: every observation scored by the exact observation
    /// density at the ODE state `x_t(θ)`, with `samples` observation-model
    /// draws per stream for CRPS, PIT and the interval, seeded by `seed`.
    /// Observations before index `t0` are not scored (`score_from` is the
    /// time that index came from, recorded on the trace).
    ///
    /// `theta` must name every estimated parameter; a name the model does not
    /// declare is refused rather than dropped. Parameters it does not name
    /// keep the fit's resolved `[fixed]` values.
    ///
    /// An `Err` is the trajectory failing at θ (the solver, or a missing
    /// output snapshot at an observation time).
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn score(
        &self,
        theta: &BTreeMap<String, f64>,
        samples: usize,
        seed: u64,
        t0: usize,
        score_from: Option<f64>,
        enabled: Vec<String>,
        disabled: Vec<String>,
    ) -> Result<PrequentialTrace, String> {
        let compiled = &self.config.compiled;
        let mut params = self.config.base_params.clone();
        for (name, &v) in theta {
            let idx = compiled.param_index.get(name.as_str()).ok_or_else(|| format!(
                "the parameter vector names '{name}', which the fitted model does not \
                 declare"))?;
            params[*idx] = v;
        }
        if let Some(p) = self.config.estimated_params.iter().find(|p| !theta.contains_key(&p.name)) {
            return Err(format!(
                "the parameter vector has no value for the estimated parameter '{}'",
                p.name));
        }
        let mut rng = sim::rng::StatefulRng::new_stream(
            seed, sim::inference::ODE_PREDICTIVE_RNG_STREAM);
        let recorded = sim::inference::ode_prequential_recorded(
            compiled, &self.obs_model, &self.obs_times, self.dt, &params,
            // No coarse burn-in: the scores integrate the warm-up at the fit's
            // step, the exact skeleton a `burnin_dt` approximates.
            self.dt, samples, &mut rng,
        ).map_err(|e| format!("the ODE trajectory failed at this θ: {e}"))?;
        // One deterministic state support point per step: its effective
        // sample size is 1 by construction.
        let ess = vec![1.0; recorded.obs_times.len()];
        let mut trace = sim::inference::prequential::build_trace(
            &recorded,
            &self.obs_model.per_stream_observed(),
            &self.obs_model.per_stream_coverage(),
            &ess, t0, seed,
            self.obs_model.first_observation_opens_after_run_start(
                compiled.model.simulation.t_start),
            score_from,
        );
        trace.basis = PredictiveBasis::OdeFreeForward { samples_per_draw: samples };
        // No filter ran: 0 particles (`Derivation::particles`).
        trace.derivation = Some(Derivation::single_pass(
            &trace, 0, self.dt, seed, ThetaSource::Supplied, enabled, disabled));
        Ok(trace)
    }
}
