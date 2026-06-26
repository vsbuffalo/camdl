//! Shared prior-resolution chain for `camdl profile --algorithm pmmh`
//! (gh#73) and `camdl fit run` Bayesian stages (gh#75).
//!
//! Background
//! ----------
//!
//! Before gh#73 the profile-PMMH path hardcoded `Prior::Flat` for every
//! estimated parameter at `profile.rs:1013-1021`, regardless of what
//! priors the `.camdl` model file declared via `~` syntax or what a
//! hypothetical fit toml would supply. The net effect was that
//! `--algorithm pmmh` silently behaved as per-cell MLE with flat
//! priors, which both contradicted the user's expectation
//! ("`pmmh` implies posterior semantics") and produced per-cell
//! parameter values outside the priors' plausibility support on real
//! models (see the camdl-book seed-timing chapter incident in gh#73's
//! body — `t_rep = −40` at a `Normal(4, 5)` prior; `n_seed = 1000`
//! pinned at the bound). gh#73 introduced this module to fix the
//! profile path; gh#75 extended the same resolver to `camdl fit run`
//! Bayesian stages, with one semantic difference (see below).
//!
//! Precedence chain
//! ----------------
//!
//! For each estimated parameter `p` (in profile, the focal swept
//! parameter is removed from the estimated set before this
//! resolution — it's not estimated, it's fixed at the cell value):
//!
//!   1. **fit-toml priors** (highest). When the user passed `--fit`
//!      (profile) or invoked `fit run` (which always loads a fit
//!      toml), look up `p` in the toml's `[estimate.<p>.prior]`
//!      block. Behavior matches `camdl fit run` byte-identically
//!      because this module routes through the same
//!      [`crate::fit::runner::resolve_prior`] helper.
//!
//!   2. **Model-IR `~` priors** (fallback). If the fit toml didn't
//!      supply a prior for `p`, fall through to the IR's
//!      `parameter.prior` (populated from `~` syntax during DSL
//!      compilation).
//!
//!   3. **`Prior::Flat`** (last resort). Used only when neither (1)
//!      nor (2) supplied a prior.
//!
//! Profile vs fit run semantics for tier 3
//! ---------------------------------------
//!
//! `camdl profile`: tier 3 is a *silent fallback* that emits a
//! warning naming the affected parameters (see
//! [`format_flat_fallback_warning`]). Per-cell IF2/PMMH with flat
//! priors is recoverable by spot-checking per-cell parameter values.
//!
//! `camdl fit run`: tier 3 is reachable ONLY via an *explicit
//! opt-in* — `prior = { flat = {} }` in the fit toml. Implicit
//! fall-through to flat priors is a hard validation error before
//! the fit starts. The downstream interpretation of a `fit run`
//! chain (canonical posterior in `fit_summary.json`) is too
//! authoritative to silently target the unconditioned likelihood,
//! so users who genuinely want flat priors declare the choice
//! accountably; the provenance records the source as
//! [`PriorSource::FlatExplicit`] for each such parameter.
//!
//! Why a separate module
//! ---------------------
//!
//! Keeping the resolution logic out of `profile.rs` and the fit
//! validator lets us unit-test the precedence chain against a
//! synthetic `ir::Model` + fit-toml `IndexMap` without standing up
//! the full CLI dispatch. The integration counterparts in
//! `rust/crates/cli/tests/profile_priors.rs` (profile) and
//! `rust/crates/cli/tests/fit_priors.rs` (fit run) drive the binary
//! end-to-end and check the wired-through behavior.

use indexmap::IndexMap;
use sim::inference::pmmh::Prior;

use crate::fit::config_v2::EstimateSpecV2;
use crate::fit::runner::resolve_prior;

/// Where a parameter's prior came from in the precedence chain.
///
/// Serialized verbatim into `run.json`'s `resolved_priors` array so
/// reviewers can audit which knob (model file vs fit toml) controlled
/// each parameter at run time. Wire format is the snake_case
/// discriminator below.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PriorSource {
    /// Resolved from the fit toml's `[estimate.<param>.prior]`
    /// block with a non-flat distribution (precedence tier 1).
    FitToml,
    /// Resolved from the model IR's `parameter.prior` field
    /// (precedence tier 2 — populated from DSL `~` syntax during
    /// compilation).
    ModelIr,
    /// Reached tier 3 (Prior::Flat) by silent fallback — neither
    /// the fit toml nor the model IR supplied a prior. Only fires
    /// in `camdl profile`'s per-cell PMMH path; `camdl fit run`
    /// treats this case as a hard validation error before the
    /// fit starts (gh#75). Warning-emitting case in profile.
    FlatFallback,
    /// User explicitly opted into a flat prior via
    /// `prior = { flat = {} }` in the fit toml (gh#75). Distinct
    /// from `FlatFallback`: the choice is accountable
    /// (serialized into the fit toml the user wrote) and no
    /// warning fires. Honoured by `camdl fit run` for users who
    /// genuinely want the chain to target the unconditioned
    /// likelihood (scaled-likelihood posterior).
    FlatExplicit,
}

impl PriorSource {
    /// Map the runner's source string (returned from
    /// [`crate::fit::runner::resolve_prior`]) into a typed
    /// `PriorSource`. The runner's strings are stable: see
    /// runner.rs:1999/2005/2011/2015 plus the gh#75 explicit-flat
    /// path.
    fn from_runner_source(s: &str) -> Self {
        match s {
            "fit.toml"             => PriorSource::FitToml,
            "model"                => PriorSource::ModelIr,
            "model (hierarchical)" => PriorSource::ModelIr,
            "flat (explicit)"      => PriorSource::FlatExplicit,
            // Includes "flat (default)" and any future fallback labels.
            _                      => PriorSource::FlatFallback,
        }
    }
}

/// One row of the resolution table — one per estimated parameter,
/// in the same order as the input `names` slice.
#[derive(Debug, Clone)]
pub struct ResolvedPrior {
    pub param:  String,
    pub prior:  Prior,
    pub source: PriorSource,
}

/// Resolve the prior for every estimated parameter via the three-tier
/// precedence chain (fit toml → model IR → flat). See module docs.
///
/// `names` is the ordered list of estimated parameter names — typically
/// `per_start_specs.iter().map(|p| p.name.clone()).collect()` at the
/// profile-PMMH call site. The returned vector is in the same order.
///
/// `estimate` is the fit toml's `[estimate]` map; pass an empty
/// `IndexMap` when `--fit` was not supplied (the resolver still walks
/// the model-IR tier for each parameter).
///
/// `model` is the compiled IR — the fallback source for parameters not
/// covered by `estimate`.
pub fn resolve_priors_with_precedence(
    names:    &[String],
    estimate: &IndexMap<String, EstimateSpecV2>,
    model:    &ir::Model,
) -> Vec<ResolvedPrior> {
    names.iter().map(|name| {
        let (prior, src) = resolve_prior(name, estimate, model);
        ResolvedPrior {
            param:  name.clone(),
            prior,
            source: PriorSource::from_runner_source(src),
        }
    }).collect()
}

/// Format the user-facing warning that fires when any resolved prior
/// ended up as `Prior::Flat`. Returns `None` if no parameters fell
/// through (no warning needed).
///
/// The text is intentionally structured (parameter table, two-line
/// remediation, suppression instructions) so a future retrofit to a
/// typed `DiagnosticKind::ProfileFlatPriorFallback { affected: ... }`
/// variant (gh#72) can splice the same content without reflowing the
/// surface.
///
/// `fit_toml_supplied` controls the per-param "reason" column and the
/// suppression hint at the end. When `true`, the user passed `--fit`
/// but the toml didn't declare priors for the affected params; when
/// `false`, no `--fit` was supplied at all.
pub fn format_flat_fallback_warning(
    resolved:        &[ResolvedPrior],
    fit_toml_supplied: bool,
) -> Option<String> {
    let affected: Vec<&ResolvedPrior> = resolved.iter()
        .filter(|r| r.source == PriorSource::FlatFallback)
        .collect();
    if affected.is_empty() {
        return None;
    }

    // Two-column reason table. Widths derived from the affected set so
    // the output stays compact for the typical case (1-3 params).
    let name_width = affected.iter()
        .map(|r| r.param.len()).max().unwrap_or(9)
        .max("parameter".len());

    let reason_for = |_r: &ResolvedPrior| -> &'static str {
        if fit_toml_supplied {
            "--fit toml supplied but no [estimate.<param>.prior] block"
        } else {
            "no prior declared in model file (no `~` syntax) and no --fit toml supplied"
        }
    };

    let mut s = String::new();
    s.push_str(
        "warning: profile is using flat priors for the following estimated\n\
         parameters in the per-cell PMMH:\n\n");
    s.push_str(&format!("  {:<width$}   reason\n", "parameter", width = name_width));
    s.push_str(&format!("  {:-<width$}   {:-<60}\n", "", "", width = name_width));
    for r in &affected {
        s.push_str(&format!(
            "  {:<width$}   {}\n",
            r.param, reason_for(r), width = name_width,
        ));
    }
    s.push_str("\nA flat prior gives the per-cell PMMH an MLE search, not posterior\n");
    s.push_str("sampling — even though `--algorithm pmmh` is set. To fix:\n\n");
    s.push_str("  (i)  Declare priors in the model file via `~` syntax for each\n");
    s.push_str("       estimated parameter, OR\n");
    s.push_str("  (ii) Supply a fit toml with `--fit <path.toml>` that has an\n");
    s.push_str("       [estimate.<param>.prior] block for each estimated parameter.\n\n");
    if fit_toml_supplied {
        s.push_str(
            "If flat priors are intentional, suppress this warning by setting\n\
             `[diagnostics] suppress = [\"profile_flat_prior_fallback\"]` in the\n\
             fit toml.\n");
    } else {
        s.push_str(
            "If flat priors are intentional, suppress this warning by passing\n\
             `--suppress-warnings` (or supplying a fit toml with the same\n\
             `[diagnostics] suppress = [\"profile_flat_prior_fallback\"]` block).\n");
    }
    Some(s)
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use indexmap::IndexMap;
    use ir::parameter::{
        BetaPrior, LogNormalPrior, NormalPrior, Parameter, PriorDist,
    };
    use std::collections::HashMap;

    /// Build a minimal `ir::Model` with the given parameters. The
    /// rest of the IR is filled in with empty-but-valid stubs so the
    /// resolver can run without compiling a real model.
    fn mk_model(parameters: Vec<Parameter>) -> ir::Model {
        ir::Model {
            name: "test".into(),
            version: "0.3".into(),
            time_unit: "days".into(),
            description: None,
            origin: None,
            origin_rata_die: None,
            compartments: vec![],
            transitions: vec![],
            ode_equations: vec![],
            time_functions: vec![],
            tables: vec![],
            interventions: vec![],
            observations: vec![],
            bindings: vec![],
            per_eval_bindings: vec![],
            parameters,
            initial_conditions: ir::model::InitialConditions::Explicit(HashMap::new()),
            output: ir::model::OutputConfig {
                times: ir::model::OutputSchedule::AtTimes(vec![]),
                format: "tsv".into(),
                trajectory: true,
                observations: false,
            },
            simulation: ir::model::SimulationConfig {
                t_start: 0.0,
                t_end: 1.0,
                time_semantics: "continuous".into(),
                dt: None,
                rng_seed: None,
                integrator: Default::default(),
            },
            presets: vec![],
            model_structure: None,
            balance: None,
            identity_tracked_compartments: vec![], quantities: vec![],
        }
    }

    fn mk_param(name: &str, prior: Option<PriorDist>) -> Parameter {
        Parameter {
            name: name.into(),
            value: ir::parameter::ParamValue::Estimated {
                init: None,
                bounds: Some((0.01, 2.0)),
                prior: match prior {
                    Some(pd) => ir::parameter::PriorSpec::Dist(pd),
                    None => ir::parameter::PriorSpec::Flat,
                },
                transform: ir::parameter::Transform::Identity,
            },
            param_kind: None,
            param_dim: None,
        }
    }

    fn mk_estimate_spec(prior: Option<PriorDist>) -> EstimateSpecV2 {
        EstimateSpecV2 {
            bounds: Some((0.01, 2.0)),
            transform: None,
            prior: prior.map(crate::fit::config_v2::EstimatePriorSpec::Dist),
            ivp: false,
            rw_sd: None,
            start: None,
        }
    }

    #[test]
    fn resolve_prior_from_fit_toml_takes_precedence_over_model_ir() {
        // beta has both an IR LogNormal prior and a fit-toml Normal
        // prior. The resolver must pick the fit-toml side.
        let model = mk_model(vec![
            mk_param("beta", Some(PriorDist::LogNormal(LogNormalPrior {
                mu: -1.0, sigma: 0.5,
            }))),
        ]);
        let mut estimate: IndexMap<String, EstimateSpecV2> = IndexMap::new();
        estimate.insert("beta".into(), mk_estimate_spec(Some(
            PriorDist::Normal(NormalPrior { mean: 0.3, sd: 0.1 })
        )));

        let resolved = resolve_priors_with_precedence(
            &["beta".to_string()], &estimate, &model,
        );
        assert_eq!(resolved.len(), 1);
        assert_eq!(resolved[0].param, "beta");
        assert_eq!(resolved[0].source, PriorSource::FitToml,
            "fit-toml prior must beat model-IR prior");
        match &resolved[0].prior {
            Prior::Fixed(sim::inference::prior::Density::Normal { mean, sd }) => {
                assert!((mean - 0.3).abs() < 1e-12);
                assert!((sd - 0.1).abs() < 1e-12);
            }
            other => panic!("expected Prior::Normal from fit-toml, got {:?}", other),
        }
    }

    #[test]
    fn resolve_prior_falls_back_to_model_ir_when_fit_toml_silent() {
        // beta has an IR LogNormal prior; the fit-toml estimate map
        // doesn't mention beta. The resolver must use the IR side.
        let model = mk_model(vec![
            mk_param("beta", Some(PriorDist::LogNormal(LogNormalPrior {
                mu: -2.0, sigma: 0.4,
            }))),
        ]);
        let estimate: IndexMap<String, EstimateSpecV2> = IndexMap::new();

        let resolved = resolve_priors_with_precedence(
            &["beta".to_string()], &estimate, &model,
        );
        assert_eq!(resolved[0].source, PriorSource::ModelIr,
            "model-IR prior must apply when fit-toml is silent");
        match &resolved[0].prior {
            // LogNormal in IR → TransformedNormal in runtime
            Prior::Fixed(sim::inference::prior::Density::TransformedNormal { mean, sd }) => {
                assert!((mean - (-2.0)).abs() < 1e-12);
                assert!((sd - 0.4).abs() < 1e-12);
            }
            other => panic!("expected TransformedNormal, got {:?}", other),
        }
    }

    #[test]
    fn resolve_prior_falls_back_to_flat_with_warning_when_both_silent() {
        // gamma has no IR prior and isn't in the fit-toml estimate
        // map. Resolution must fall through to Prior::Flat and the
        // warning text must name gamma.
        let model = mk_model(vec![mk_param("gamma", None)]);
        let estimate: IndexMap<String, EstimateSpecV2> = IndexMap::new();

        let resolved = resolve_priors_with_precedence(
            &["gamma".to_string()], &estimate, &model,
        );
        assert_eq!(resolved[0].source, PriorSource::FlatFallback);
        assert!(matches!(resolved[0].prior, Prior::Fixed(sim::inference::prior::Density::Flat)));

        // No fit toml supplied → "no --fit toml supplied" branch.
        let warning = format_flat_fallback_warning(&resolved, false)
            .expect("warning must fire when any param falls through to flat");
        assert!(warning.contains("gamma"),
            "warning must name affected parameter `gamma`. Got:\n{}", warning);
        assert!(warning.contains("flat priors"),
            "warning must explain the consequence (flat-prior semantics):\n{}", warning);
        assert!(warning.contains("--fit"),
            "warning must suggest --fit as a remedy:\n{}", warning);
        assert!(warning.contains("model file"),
            "warning must mention the model-file path as a remedy:\n{}", warning);
    }

    #[test]
    fn resolve_priors_mixed_sources_per_parameter() {
        // beta: fit-toml prior (tier 1)
        // gamma: model-IR prior (tier 2)
        // delta: no prior (tier 3, flat)
        let model = mk_model(vec![
            mk_param("beta",  Some(PriorDist::Normal(NormalPrior { mean: 0.0, sd: 1.0 }))),
            mk_param("gamma", Some(PriorDist::Beta(BetaPrior   { alpha: 2.0, beta: 5.0 }))),
            mk_param("delta", None),
        ]);
        let mut estimate: IndexMap<String, EstimateSpecV2> = IndexMap::new();
        estimate.insert("beta".into(), mk_estimate_spec(Some(
            PriorDist::Normal(NormalPrior { mean: 0.5, sd: 0.2 })
        )));

        let names = vec!["beta".to_string(), "gamma".to_string(), "delta".to_string()];
        let resolved = resolve_priors_with_precedence(&names, &estimate, &model);
        assert_eq!(resolved[0].source, PriorSource::FitToml);
        assert_eq!(resolved[1].source, PriorSource::ModelIr);
        assert_eq!(resolved[2].source, PriorSource::FlatFallback);

        // Warning only lists delta.
        let w = format_flat_fallback_warning(&resolved, true).unwrap();
        assert!(w.contains("delta"), "warning must list delta. Got:\n{}", w);
        assert!(!w.contains("\n  beta "),
            "warning must NOT list beta (resolved from fit toml). Got:\n{}", w);
        assert!(!w.contains("\n  gamma "),
            "warning must NOT list gamma (resolved from IR). Got:\n{}", w);
        // fit-toml-supplied wording differs from the no-toml case.
        assert!(w.contains("[estimate.<param>.prior]") || w.contains("[estimate."),
            "warning text for the fit-toml-supplied case should reference \
             the [estimate.<param>.prior] block. Got:\n{}", w);
    }

    #[test]
    fn no_warning_when_every_param_has_a_prior() {
        let model = mk_model(vec![
            mk_param("beta", Some(PriorDist::Normal(NormalPrior { mean: 0.0, sd: 1.0 }))),
        ]);
        let estimate: IndexMap<String, EstimateSpecV2> = IndexMap::new();
        let resolved = resolve_priors_with_precedence(
            &["beta".to_string()], &estimate, &model,
        );
        assert!(format_flat_fallback_warning(&resolved, false).is_none(),
            "no warning when every parameter resolves to a non-flat prior");
    }
}
