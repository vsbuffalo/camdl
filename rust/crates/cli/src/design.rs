//! `camdl design`: tables of parameter points for `camdl simulate
//! --sweep-file`.
//!
//! ```text
//! MODEL + --fit's problem ──resolve_estimated──► Vec<EstimatedParam>
//!                                                      │  --n, --seed
//!                                                  lhs_design
//!                                                      ▼
//!                                   LhsDesign ──to_tsv──► stdout or -o FILE
//! ```
//!
//! `design lhs` is a function of the model, the fit config, `--n` and
//! `--seed`: the same four give the same bytes. It claims no store leaf and
//! evaluates no likelihood. The points are those
//! [`crate::fit::init::build_lhs_chain_starts`] draws, so a design and a fit's
//! `starts = "lhs"` share one sampler.

use std::sync::Arc;

use indexmap::{IndexMap, IndexSet};
use sim::compiled_model::CompiledModel;
use sim::inference::types::EstimatedParam;

use crate::fit::handle::FitProblem;

/// A Latin-hypercube design: `points[i][d]` is the natural-scale value of
/// parameter `params[d]` at point `i`. A point's id is its index.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct LhsDesign {
    /// The sampled parameters, in the order they were given.
    pub params: Vec<String>,
    pub points: Vec<Vec<f64>>,
}

impl LhsDesign {
    /// The design as the TSV [`crate::sweep::read_sweep_file`] reads: a
    /// `point_id` column, then one column per parameter. Each value is written
    /// in its shortest form that parses back to the same `f64`.
    pub(crate) fn to_tsv(&self) -> String {
        let mut text = String::from(crate::sweep::POINT_ID_COLUMN);
        for name in &self.params {
            text.push('\t');
            text.push_str(name);
        }
        text.push('\n');
        for (point_id, point) in self.points.iter().enumerate() {
            text.push_str(&point_id.to_string());
            for value in point {
                text.push('\t');
                text.push_str(&value.to_string());
            }
            text.push('\n');
        }
        text
    }
}

/// `n` Latin-hypercube points over the search boxes of `estimated`, from
/// `seed`.
///
/// Each parameter's box is cut into `n` bins of equal width, on the log scale
/// for a log-spaced parameter (`Transform::log_spaced`) and on the linear
/// scale otherwise, and each bin holds exactly one point.
///
/// Refused: `n` below 2, an empty `estimated`, and a parameter whose box has
/// an infinite end. For such a parameter the sampler returns the declared
/// start scaled by a factor in `[0.5, 1.5)`, which stratifies nothing.
pub(crate) fn lhs_design(
    estimated: &[EstimatedParam],
    n: usize,
    seed: u64,
) -> Result<LhsDesign, String> {
    refuse_fewer_than_two_points(n)?;
    if estimated.is_empty() {
        return Err("the fit config estimates no parameter, so there is nothing to sample. \
                    Name the parameters to vary in its [estimate] table.".into());
    }
    let open: Vec<String> = estimated
        .iter()
        .filter(|p| !(p.bounds().0.is_finite() && p.bounds().1.is_finite()))
        .map(|p| format!("`{}` [{}, {}]", p.name, p.bounds().0, p.bounds().1))
        .collect();
    if !open.is_empty() {
        return Err(format!(
            "design lhs samples a Latin hypercube within each estimated parameter's search \
             box, and these parameters have no finite one: {}. Give each `bounds = [lo, hi]` \
             in its [estimate] entry (or `in [lo, hi]` in the model).",
            open.join(", ")));
    }
    let drawn = crate::fit::init::build_lhs_chain_starts(estimated, n, seed);
    Ok(LhsDesign {
        params: estimated.iter().map(|p| p.name.clone()).collect(),
        points: drawn
            .iter()
            .map(|point| point.iter().map(|p| p.initial).collect())
            .collect(),
    })
}

/// The fit config's estimated parameters, resolved against the model at
/// `model_path`: one per `[estimate]` entry, in name order, each with
/// the search box (`bounds` in the entry, else the model's declared range) and
/// the transform its kind and box select.
///
/// The model's parameters are resolved as a fit of this config resolves them:
/// `[fixed]`, the config's scenario and its `[data.tables]` are applied, and a
/// scenario that sets an estimated parameter or moves a fixed one is refused.
/// Observation data is read only for a model anchored to its observed window
/// (`simulate { from = first_obs }`), which cannot be compiled until the
/// window is known.
fn resolve_estimated(
    out: &crate::out::Out,
    model_path: &str,
    fit: &FitProblem,
    compile: crate::util::CompileSettings,
    seed: u64,
) -> Result<Vec<EstimatedParam>, String> {
    use crate::fit::runner::{build_if2_params_from_specs, ParamSpec};

    let config = &fit.config.problem;
    let (mut model, _ir_json) = crate::util::load_model(model_path, compile)?;
    // The step the anchor reader checks observation times against is the
    // fit's, by the one rule.
    let dt = crate::dt_rule::resolved_dt(model.simulation.dt, config.config.dt);
    crate::obs_anchor::resolve_from_config(out, &mut model, config, dt)?;

    // Every parameter needs a value before the resolver runs; an `[estimate]`
    // entry without a `start` is given one here. The values do not reach the
    // design.
    crate::fit::runner::seed_estimate_starts(
        &mut model, &config.estimate, seed, &std::collections::HashSet::new());

    let mut fixed_cfg = config.fixed.clone();
    fixed_cfg.expand_from_scenario(&model, &config.estimate)?;
    let fixed: IndexMap<String, f64> = fixed_cfg.resolve_with_model(&model)?;

    let crate::util::ScenarioSelection { scenario, enable, disable } =
        crate::util::combine_fit_scenario(
            crate::util::ScenarioChoice { scenario: None, enable: &[], disable: &[] },
            crate::util::ScenarioChoice {
                scenario: config.scenario.as_deref(),
                enable: &config.enable,
                disable: &config.disable,
            },
        )?;
    if let Some(scenario) = scenario.as_deref() {
        let estimated: Vec<&str> = config.estimate.keys().map(|k| k.as_str()).collect();
        let refusals: Vec<String> = [
            crate::params_resolver::scenario_estimate_collision(
                &model, scenario, &estimated,
                crate::params_resolver::EstimatedBy::FitToml(&fit.handle),
            )?,
            crate::params_resolver::scenario_fixed_collision(
                &model, scenario, &fixed, &fit.handle,
            )?,
        ].into_iter().flatten().collect();
        if !refusals.is_empty() {
            return Err(refusals.join("\n\nerror: "));
        }
    }

    let fit_toml_estimate: IndexSet<String> = config.estimate.keys().cloned().collect();
    let tables = config.external_bindings();
    let resolved = crate::params_resolver::resolve_parameters(
        crate::params_resolver::ParameterInputs {
            model: &model,
            scenario: scenario.as_deref(),
            adhoc_enable: &enable,
            adhoc_disable: &disable,
            scenario_inline_name: None,
            scenario_inline_set: &[],
            scenario_inline_scale: &[],
            point_overrides: &[],
            fixed_cli: &[],
            fixed_files: &[],
            fit_toml_fixed: &fixed,
            fit_toml_estimate: &fit_toml_estimate,
            external: &tables,
        },
    ).map_err(|e| e.to_string())?;
    let model = resolved.model;
    crate::util::validate_parameter_values(&model)?;

    let compiled = Arc::new(CompiledModel::new(model.clone())
        .map_err(|e| format!("compile error: {:?}", e))?);
    let specs: Vec<ParamSpec> = config.estimate.iter()
        .map(|(name, spec)| ParamSpec {
            name: name.clone(),
            rw_sd: spec.rw_sd,
            perturb_only_at_t0: spec.perturb_only_at_t0,
            bounds: spec.bounds,
        })
        .collect();
    let mut estimated = build_if2_params_from_specs(&model, &compiled, &compiled.default_params, &specs)?;
    // The sampler draws parameter by parameter, so this order decides every
    // point. Name order, as `fit run` sorts, keeps the design independent of
    // the order the `[estimate]` table lists its entries in.
    estimated.sort_by(|a, b| a.name.cmp(&b.name));
    Ok(estimated)
}

/// `camdl design lhs`, under the compile settings `compile`.
pub fn cmd_design_lhs(a: &crate::args::DesignLhsArgs, compile: crate::util::CompileSettings) {
    let out = crate::out::Out::new(false);
    let crate::args::DesignLhsArgs { model, fit, n, seed, output } = a;
    if let Err(e) = refuse_fewer_than_two_points(*n) {
        out.fail(format!("error: {e}"));
    }
    let fit_problem = crate::fit::handle::resolve_fit_problem(&fit.to_string_lossy())
        .unwrap_or_else(|e| out.fail(format!("error: --fit {}: {e}", fit.display())));
    let design = resolve_estimated(&out, &model.to_string_lossy(), &fit_problem, compile, *seed)
        .and_then(|estimated| lhs_design(&estimated, *n, *seed))
        .unwrap_or_else(|e| out.fail(format!("error: {e}")));
    let text = design.to_tsv();
    match output {
        None => out.data_raw(text),
        Some(path) => {
            std::fs::write(path, text).unwrap_or_else(|e| {
                out.fail(format!("error: cannot write {}: {e}", path.display()))
            });
            out.milestone("wrote", format!(
                "{} \u{b7} {} points \u{d7} {} parameters",
                path.display(), design.points.len(), design.params.len()));
        }
    }
}

fn refuse_fewer_than_two_points(n: usize) -> Result<(), String> {
    if n >= 2 {
        return Ok(());
    }
    Err(format!(
        "--n must be at least 2 (got {n}): a Latin hypercube cuts each parameter's search \
         box into --n bins and puts one point in each."))
}

#[cfg(test)]
mod tests {
    use super::*;
    use sim::inference::types::Transform;

    fn ep(name: &str, index: usize, transform: Transform) -> EstimatedParam {
        EstimatedParam {
            name: name.into(),
            index,
            initial: 1.0,
            rw_sd: 0.1,
            transform,
            rw_sd_auto: false,
            perturb_only_at_t0: false,
        }
    }

    /// Three parameters, not in name order: a log-spaced rate, a linear
    /// interval that crosses zero, and a linear probability.
    fn estimated() -> Vec<EstimatedParam> {
        vec![
            ep("rho", 0, Transform::Interval { lo: 0.0, hi: 1.0 }),
            ep("beta", 1, Transform::LogInterval { lo: 1e-4, hi: 10.0 }),
            ep("shift", 2, Transform::Interval { lo: -2.0, hi: 6.0 }),
        ]
    }

    fn model_with(params: &[&str]) -> ir::Model {
        let mut m = crate::test_support::tiny_model();
        m.parameters = params
            .iter()
            .map(|n| ir::parameter::Parameter {
                name: n.to_string(),
                value: ir::parameter::ParamValue::Fixed { value: 0.5 },
                param_kind: None,
                param_dim: None,
            })
            .collect();
        m
    }

    /// The bin of `v` among `n` equal-width bins of `[lo, hi]`.
    fn bin(v: f64, lo: f64, hi: f64, n: usize) -> usize {
        (((v - lo) / (hi - lo)) * n as f64).floor() as usize
    }

    fn column(design: &LhsDesign, name: &str) -> Vec<f64> {
        let d = design.params.iter().position(|p| p == name).unwrap();
        design.points.iter().map(|p| p[d]).collect()
    }

    /// The written file is read back by `read_sweep_file` unchanged, and every
    /// value read equals, bit for bit, the one `build_lhs_chain_starts` drew
    /// for that point and parameter.
    #[test]
    fn the_tsv_reads_back_to_the_samplers_values_bit_for_bit() {
        let est = estimated();
        let (n, seed) = (25, 7);
        let design = lhs_design(&est, n, seed).unwrap();
        let text = design.to_tsv();

        let mut lines = text.lines();
        assert_eq!(lines.next(), Some("point_id\trho\tbeta\tshift"),
            "the header is point_id, then the parameters in the order given");
        assert_eq!(lines.count(), n);
        assert!(!text.contains('#'), "no comment lines");
        assert!(text.ends_with('\n'));

        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("design.tsv");
        std::fs::write(&path, &text).unwrap();
        let read = crate::sweep::read_sweep_file(&path, &model_with(&["beta", "rho", "shift"]))
            .expect("read_sweep_file accepts the design");

        let drawn = crate::fit::init::build_lhs_chain_starts(&est, n, seed);
        assert_eq!(read.points.len(), n);
        for (i, (point_id, point)) in read.points.iter().enumerate() {
            assert_eq!(*point_id, i, "point_id is the 0-based row");
            assert_eq!(point.len(), est.len());
            for spec in &drawn[i] {
                let (_, v) = point.iter().find(|(name, _)| *name == spec.name).unwrap();
                assert_eq!(v.to_bits(), spec.initial.to_bits(),
                    "point {i}, `{}`: read {v}, drawn {}", spec.name, spec.initial);
            }
        }
    }

    #[test]
    fn the_same_inputs_give_the_same_bytes_and_another_seed_gives_others() {
        let est = estimated();
        let a = lhs_design(&est, 40, 3).unwrap().to_tsv();
        let b = lhs_design(&est, 40, 3).unwrap().to_tsv();
        let c = lhs_design(&est, 40, 4).unwrap().to_tsv();
        assert_eq!(a, b);
        assert_ne!(a, c);
    }

    /// Each of the n equal-width bins holds exactly one point: bins in log
    /// space for the log-spaced parameter, on the linear scale for the
    /// others.
    #[test]
    fn each_parameter_is_stratified_on_its_own_scale() {
        let n = 50;
        let design = lhs_design(&estimated(), n, 11).unwrap();
        let one_per_bin = |bins: Vec<usize>| {
            let mut sorted = bins;
            sorted.sort();
            sorted == (0..n).collect::<Vec<_>>()
        };

        let beta = column(&design, "beta");
        assert!(one_per_bin(beta.iter().map(|v| bin(v.ln(), 1e-4f64.ln(), 10f64.ln(), n)).collect()),
            "beta holds one point per log-space bin: {beta:?}");
        assert!(!one_per_bin(beta.iter().map(|v| bin(*v, 1e-4, 10.0, n)).collect()),
            "beta is not stratified on the linear scale");

        let shift = column(&design, "shift");
        assert!(one_per_bin(shift.iter().map(|v| bin(*v, -2.0, 6.0, n)).collect()),
            "shift holds one point per linear bin: {shift:?}");
        let rho = column(&design, "rho");
        assert!(one_per_bin(rho.iter().map(|v| bin(*v, 0.0, 1.0, n)).collect()),
            "rho holds one point per linear bin: {rho:?}");
    }

    #[test]
    fn fewer_than_two_points_is_refused() {
        for n in [0, 1] {
            let err = lhs_design(&estimated(), n, 1).unwrap_err();
            assert!(err.contains("--n must be at least 2") && err.contains(&format!("got {n}")),
                "{err}");
        }
        assert!(lhs_design(&estimated(), 2, 1).is_ok());
    }

    #[test]
    fn a_parameter_without_a_finite_box_is_refused_by_name() {
        let mut est = estimated();
        est.push(ep("mu", 3, Transform::LowerBounded { lo: 0.0 }));
        est.push(ep("drift", 4, Transform::Identity));
        let err = lhs_design(&est, 10, 1).unwrap_err();
        assert!(err.contains("`mu` [0, inf]") && err.contains("`drift` [-inf, inf]"), "{err}");
        assert!(err.contains("bounds = [lo, hi]"), "{err}");
        assert!(!err.contains("`beta`") && !err.contains("`rho`"),
            "only the open boxes are named: {err}");
    }

    #[test]
    fn no_estimated_parameter_is_refused() {
        let err = lhs_design(&[], 10, 1).unwrap_err();
        assert!(err.contains("[estimate]"), "{err}");
    }
}
