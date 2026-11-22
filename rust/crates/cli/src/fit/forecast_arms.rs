//! The arms of a fixed-origin forecast (D50; migration plan, Stage 5).
//!
//! The fitted model is the baseline arm. Every `--scenario`, and every point
//! of a `--sweep`, is a further arm: the fitted model with the scenario's
//! interventions and parameter changes, or the swept values, applied on top,
//! filtered with the same seeds. An arm may differ from the baseline only at
//! or after the forecast origin: then every arm filters the training rows
//! through the same model with the same random numbers and starts its
//! forecast from the same particles, and the arms differ only by what the
//! scenario does after the origin.
//!
//! Before anything is filtered, [`build_arms`] refuses an arm whose first
//! difference from the baseline it cannot place at or after the origin:
//!
//! - an intervention the arm switches on or off that fires before the origin,
//!   or at a time that depends on a parameter or on the state;
//! - a parameter the arm changes (a scenario's `set` or `scale`, a swept
//!   value) that the model reads anywhere but inside an intervention (the
//!   dynamics, the initial state, a forcing, an observation model), since
//!   those read it from the start of the run, or inside an intervention that
//!   fires before the origin or at a time that is not fixed.
//!
//! After filtering, the forecast checks each arm's record before the origin
//! against the baseline's ([`PreOriginTrace`](super::forecast_driver::PreOriginTrace)).
//! That catches any pre-origin difference the static check missed that is
//! visible in the filter's observation record (log-likelihood increments,
//! ESS, and each particle's per-stream log-likelihoods and predictive
//! draws). It does not hash the latent states, so a difference confined to
//! compartments no observation reads before the origin is not seen there.

use std::collections::BTreeSet;
use std::sync::Arc;

use indexmap::IndexMap;

use super::forecast_driver::extend_run;
use super::forecast_plan::ForecastPlan;
use super::runner::TIME_TIE_EPS;
use crate::sim_job::ScenarioRef;

/// Which arm: a scenario name (`fitted` for the fitted model) and the swept
/// values, empty without `--sweep`.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct ArmLabel {
    pub scenario: String,
    pub sweep: Vec<(String, f64)>,
}

impl ArmLabel {
    pub(crate) fn baseline() -> Self {
        ArmLabel { scenario: crate::args::FITTED.to_string(), sweep: Vec::new() }
    }

    pub(crate) fn is_baseline(&self) -> bool {
        self.scenario == crate::args::FITTED && self.sweep.is_empty()
    }

    /// How a message names the arm: `'late'`, or `'late' at cull_frac = 0.5`.
    pub(crate) fn describe(&self) -> String {
        let mut s = format!("'{}'", self.scenario);
        if !self.sweep.is_empty() {
            let at: Vec<String> = self.sweep.iter().map(|(p, v)| format!("{p} = {v}")).collect();
            s += &format!(" at {}", at.join(", "));
        }
        s
    }
}

/// What an arm changes relative to the baseline, and when the first change
/// acts.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct Difference {
    /// The earliest grid instant at which the arm acts differently from the
    /// baseline, or `None` when nothing it changes acts within the run.
    pub first: Option<f64>,
    /// The interventions live in one arm and not the other.
    pub interventions: Vec<String>,
    /// The parameters the arm sets, scales or sweeps, or that take another
    /// value in it for some draw.
    pub parameters: Vec<String>,
}

/// One arm, ready for the driver.
pub(crate) struct Arm {
    pub label: ArmLabel,
    /// The arm's model, its run extended to the plan's latest target (D49).
    pub compiled: Arc<sim::CompiledModel>,
    /// One parameter vector per posterior draw, in the baseline's order.
    pub param_sets: Vec<Vec<f64>>,
    /// `None` for the baseline.
    pub difference: Option<Difference>,
}

/// What [`build_arms`] reads.
pub(crate) struct ArmInputs<'a> {
    /// The fitted model as stored, observation anchors resolved, before the
    /// fit's intervention selection is applied.
    pub model: &'a ir::Model,
    pub fit_base: &'a crate::util::FitBase,
    /// `camdl predict`'s scenario list: the fitted model first, then each
    /// `--scenario`.
    pub scenarios: &'a [ScenarioRef],
    /// The `--sweep` grid, one entry per point; empty without `--sweep`.
    pub sweep_points: &'a [Vec<(String, f64)>],
    /// The posterior draws the forecast runs, in the driver's order.
    pub draws: &'a [&'a IndexMap<String, f64>],
    /// The baseline: the fitted model, its run extended, and its parameter
    /// vectors.
    pub baseline: &'a Arc<sim::CompiledModel>,
    pub baseline_params: &'a [Vec<f64>],
    pub plan: &'a ForecastPlan,
    pub dt: f64,
}

/// The baseline and every scenario and sweep arm, each checked to differ
/// from the baseline only at or after the origin. An arm that cannot be
/// placed there is refused, before any filtering, with the reason.
pub(crate) fn build_arms(i: ArmInputs) -> Result<Vec<Arm>, String> {
    let mut arms = vec![Arm {
        label: ArmLabel::baseline(),
        compiled: i.baseline.clone(),
        param_sets: i.baseline_params.to_vec(),
        difference: None,
    }];
    let no_sweep = [Vec::new()];
    let points: &[Vec<(String, f64)>] = if i.sweep_points.is_empty() { &no_sweep } else { i.sweep_points };
    let read_outside = params_read_outside_interventions(i.model)?;
    for point in points {
        for sref in i.scenarios {
            let label = ArmLabel { scenario: sref.name().to_string(), sweep: point.clone() };
            if label.is_baseline() {
                continue;
            }
            let (compiled, param_sets) = resolve(&i, sref, point)?;
            let mut declared: BTreeSet<String> =
                crate::params_resolver::scenario_param_footprint(i.model, sref)?;
            declared.extend(point.iter().map(|(p, _)| p.clone()));
            let difference = first_difference(&i, &compiled, &param_sets, declared, &read_outside)
                .map_err(|why| refusal(&label, i.plan.origin(), &why))?;
            arms.push(Arm { label, compiled, param_sets, difference: Some(difference) });
        }
    }
    Ok(arms)
}

fn refusal(label: &ArmLabel, origin: f64, why: &str) -> String {
    format!(
        "the forecast arm {} {why}, before the forecast origin t = {origin}. A forecast arm \
         may differ from the fitted model only at or after the origin, so that every arm \
         starts from the state filtered through the same training rows by the same model. \
         Compare it over the whole run with --mode free_forward.",
        label.describe())
}

/// The arm's compiled model and its parameter vector per draw: the draw at
/// the draw tier, the swept values over it, the scenario on top, resolved as
/// every other `camdl predict` arm is ([`crate::fit::contrasts::resolve_arm`]).
fn resolve(
    i: &ArmInputs,
    sref: &ScenarioRef,
    point: &[(String, f64)],
) -> Result<(Arc<sim::CompiledModel>, Vec<Vec<f64>>), String> {
    let run = sref.name();
    let mut model: Option<ir::Model> = None;
    let mut sets = Vec::with_capacity(i.draws.len());
    for (d, row) in i.draws.iter().enumerate() {
        let mut overrides: IndexMap<String, f64> = (*row).clone();
        for (p, v) in point {
            overrides.insert(p.clone(), *v);
        }
        let overrides: Vec<(String, f64)> = overrides.into_iter().collect();
        let resolved = crate::fit::contrasts::resolve_arm(i.model, i.fit_base, "forecast arm", run, &overrides)?;
        let mut p = i.baseline_params[d].clone();
        for rp in &resolved.params {
            if let Some(&k) = i.baseline.param_index.get(rp.name.as_str()) {
                p[k] = rp.value;
            }
        }
        sets.push(p);
        if model.is_none() {
            model = Some(resolved.model);
        }
    }
    let mut model = model.ok_or_else(|| "internal error: a forecast arm with no posterior draw".to_string())?;
    extend_run(&mut model, i.plan);
    let compiled = Arc::new(sim::CompiledModel::new(model)
        .map_err(|e| format!("compiling forecast arm '{run}': {e:?}"))?
        .with_eval_settings(i.baseline.eval));
    if compiled.param_index != i.baseline.param_index {
        return Err(format!(
            "internal error: forecast arm '{run}' orders its parameters differently from the fitted model"));
    }
    Ok((compiled, sets))
}

/// When `arm` first acts differently from the baseline, or why that cannot
/// be placed at or after the origin.
fn first_difference(
    i: &ArmInputs,
    arm: &sim::CompiledModel,
    arm_params: &[Vec<f64>],
    mut parameters: BTreeSet<String>,
    read_outside: &BTreeSet<String>,
) -> Result<Difference, String> {
    let base = i.baseline.as_ref();
    let origin = i.plan.origin();
    let window = (base.model.simulation.t_start, i.dt, base.model.simulation.t_end);
    // A parameter that takes another value for some draw differs, declared
    // or not.
    for (name, &k) in &base.param_index {
        if arm_params.iter().zip(i.baseline_params)
            .any(|(a, b)| a[k].to_bits() != b[k].to_bits())
        {
            parameters.insert(name.clone());
        }
    }
    let live = |c: &sim::CompiledModel| -> BTreeSet<String> {
        c.model.interventions.iter().map(|iv| iv.name.clone()).collect()
    };
    let (base_live, arm_live) = (live(base), live(arm));
    let interventions: Vec<String> = base_live.symmetric_difference(&arm_live).cloned().collect();

    // Where an intervention live in either arm first fires, from whichever
    // compiled model runs it.
    let fire = |name: &str| -> Result<Option<f64>, String> {
        let c = if arm_live.contains(name) { arm } else { base };
        first_fire(c, name, window)
    };
    let mut first: Option<f64> = None;
    let mut acts_at = |t: Option<f64>| {
        if let Some(t) = t {
            first = Some(first.map_or(t, |f: f64| f.min(t)));
        }
    };
    for name in &interventions {
        let t = fire(name).map_err(|why| format!("switches intervention '{name}' on or off, which {why}"))?;
        if let Some(t) = t.filter(|&t| t < origin - TIME_TIE_EPS) {
            return Err(format!("switches intervention '{name}' on or off, which first fires at t = {t}"));
        }
        acts_at(t);
    }
    for p in &parameters {
        if read_outside.contains(p) {
            return Err(format!(
                "changes parameter '{p}', which the model reads outside any intervention (in its \
                 dynamics, initial state, forcings or observation models) from the start of the run"));
        }
        for name in base_live.union(&arm_live) {
            let c = if arm_live.contains(name) { arm } else { base };
            let iv = c.model.interventions.iter().find(|iv| &iv.name == name)
                .expect("a live intervention is in its model");
            if !mentions_param(&serde_json::to_value(iv).map_err(|e| e.to_string())?, p) {
                continue;
            }
            let t = fire(name).map_err(|why| format!(
                "changes parameter '{p}', which intervention '{name}' reads; that intervention {why}"))?;
            if let Some(t) = t.filter(|&t| t < origin - TIME_TIE_EPS) {
                return Err(format!(
                    "changes parameter '{p}', which intervention '{name}' reads when it fires at t = {t}"));
            }
            acts_at(t);
        }
    }
    Ok(Difference { first, interventions, parameters: parameters.into_iter().collect() })
}

/// The first grid instant at which `compiled` fires intervention `name` in
/// `window`, `None` when it does not fire there, or why no instant can be
/// given before the run.
fn first_fire(compiled: &sim::CompiledModel, name: &str, window: (f64, f64, f64)) -> Result<Option<f64>, String> {
    use crate::fit::contrasts::{earliest_grid_instant, intervention_earliest_fire, FireTime};
    let k = compiled.model.interventions.iter().position(|iv| iv.name == name)
        .ok_or_else(|| format!("internal error: intervention '{name}' is not in the arm's model"))?;
    match intervention_earliest_fire(&compiled.model.interventions[k]) {
        FireTime::Const(_) => Ok(earliest_grid_instant(
            &compiled.resolve_fire_times(&compiled.default_params)[k], window)),
        FireTime::Empty => Ok(None),
        FireTime::Parametric => Err("fires at a time that depends on a parameter".into()),
        FireTime::Reactive => Err("fires when the state meets a condition, at no time fixed before the run".into()),
    }
}

/// Every parameter the model reads outside its interventions: in its
/// dynamics, initial state, forcings, tables and observation models. The
/// scenario presets, quantities and contrasts are left out: a forecast runs
/// none of them.
fn params_read_outside_interventions(model: &ir::Model) -> Result<BTreeSet<String>, String> {
    let mut m = model.clone();
    m.interventions.clear();
    m.presets.clear();
    m.quantities.clear();
    m.contrasts.clear();
    let v = serde_json::to_value(&m).map_err(|e| format!("internal error: serializing the model: {e}"))?;
    let mut out = BTreeSet::new();
    collect_params(&v, &mut out);
    Ok(out)
}

/// Every name an expression in `v` reads as a parameter (`{"param": name}`,
/// the IR's only parameter reference).
fn collect_params(v: &serde_json::Value, out: &mut BTreeSet<String>) {
    match v {
        serde_json::Value::Object(o) => {
            if let (1, Some(serde_json::Value::String(p))) = (o.len(), o.get("param")) {
                out.insert(p.clone());
            }
            o.values().for_each(|x| collect_params(x, out));
        }
        serde_json::Value::Array(a) => a.iter().for_each(|x| collect_params(x, out)),
        _ => {}
    }
}

fn mentions_param(v: &serde_json::Value, p: &str) -> bool {
    let mut found = BTreeSet::new();
    collect_params(v, &mut found);
    found.contains(p)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The parameter walk finds a reference wherever an expression holds one,
    /// and nothing else: a parameter's own declaration names it under `name`.
    #[test]
    fn the_parameter_walk_reads_expression_references_only() {
        let v = serde_json::json!({
            "parameters": [{"name": "beta"}, {"name": "rho"}],
            "transitions": [{"rate": {"bin_op": {"op": "mul", "left": {"param": "beta"}, "right": {"pop": "S"}}}}],
            "notes": {"param": "rho", "other": 1},
        });
        let mut found = BTreeSet::new();
        collect_params(&v, &mut found);
        assert_eq!(found, BTreeSet::from(["beta".to_string()]));
    }
}
