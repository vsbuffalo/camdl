//! Every estimated parameter's step for the method that runs: IF2's
//! random-walk perturbation from `[method.perturbation]`, or PMMH's and MH's
//! initial proposal SD from `[method.proposal]`, each in z, the coordinate the
//! method moves in (θ = g(z), [`Transform`]).
//!
//! `fit run` resolves the steps once, from the method and each estimated
//! parameter's effective domain ([`domains`]), before any value or start is
//! drawn. The one [`ResolvedSteps`] keys the run (`fit_store::MethodIdentity`)
//! and is what the method runs with (`FitRunConfig::build`). A name the table
//! leaves out takes its map's default, so a step written at its default and
//! an omitted one are the same value. Whether a step was written is
//! [`StepFrom`], shown in the launch table and never keyed.
//!
//! The rules are `docs/dev/proposals/2026-10-29-method-scoped-settings.md`
//! §4.4.

use indexmap::IndexMap;
use sim::inference::if2::EstimatedParam;
use sim::inference::types::{If2Step, PerturbAt, Perturbation, Transform, ZSd};

use super::config_v2::{self, Algorithm, EstimateSpecV2};

/// Whether a step was written in its table or is the map's default.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StepFrom {
    Written,
    Default,
}

/// IF2's step for one estimated parameter.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ResolvedPerturbation {
    pub step: If2Step,
    pub from: StepFrom,
}

/// One SD in z: PMMH's or MH's initial proposal, or a simplex member's IF2
/// SD on its log-ratio coordinate.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ResolvedProposal {
    pub sd: ZSd,
    pub from: StepFrom,
}

/// Every estimated parameter's step for one method, in `[estimate]` order.
#[derive(Clone, Debug, PartialEq)]
pub enum ResolvedSteps {
    If2 {
        params: IndexMap<String, ResolvedPerturbation>,
        /// One map per `[[simplex_groups]]` entry, members in declaration
        /// order: each member's SD on the group's log-ratio coordinate.
        simplex: Vec<IndexMap<String, ResolvedProposal>>,
    },
    /// `pmmh` and `mh`.
    Proposal(IndexMap<String, ResolvedProposal>),
    /// `pgas`, `pfilter` and the `nl-*` optimizers, which take no
    /// per-parameter step.
    None,
}

/// A step table that cannot be resolved.
#[derive(Debug, Clone, PartialEq)]
pub enum StepError {
    /// The table names a parameter that is not estimated.
    NotEstimated { table: &'static str, name: String },
    /// The parameter's map is the identity (an unbounded `real` or
    /// `instant`), which has no scale to take a default step from.
    NoDefault { table: &'static str, name: String, kind: &'static str },
    /// `initial_state = true` on a simplex member, whose group already
    /// perturbs it at t = 0 only.
    InitialStateOnSimplexMember { name: String, group: usize },
    /// `beta = {}`: an entry with no SD and no flag.
    EmptyEntry { name: String },
}

impl std::fmt::Display for StepError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            StepError::NotEstimated { table, name } => write!(
                f,
                "[method.{table}] names `{name}`, which is not in [estimate]. A step is a \
                 setting of an estimated parameter: estimate `{name}`, or delete its line."
            ),
            StepError::NoDefault { table, name, kind } => write!(
                f,
                "`{name}` is an unbounded `{kind}`, so its map is the identity and it has no \
                 scale to take a default step from. Write its step, `{name} = <sd>` in \
                 [method.{table}]: on this map z is `{name}` itself, so the value is a \
                 natural-scale SD. Or bound it (`in [lo, hi]` in the model, or \
                 `bounds = [lo, hi]` in [estimate])."
            ),
            StepError::InitialStateOnSimplexMember { name, group } => write!(
                f,
                "[method.perturbation] `{name}` has `initial_state = true`, and `{name}` is a \
                 member of simplex_groups[{group}], whose log-ratio perturbation already \
                 perturbs its members at t = 0 only. Delete `initial_state` on the member."
            ),
            StepError::EmptyEntry { name } => write!(
                f,
                "[method.perturbation] `{name} = {{}}` states no SD and no `initial_state`, \
                 which is what leaving the line out says. Delete it, or write `sd` or \
                 `initial_state = true`."
            ),
        }
    }
}

/// Each estimated parameter's effective domain (the model's support
/// intersected with the search box, `roles::effective_domain`), as the map
/// onto it, in `[estimate]` order. The search box is the one
/// `runner::resolve_search_box` gives the method, so these are the maps the
/// method's `EstimatedParam`s carry.
pub fn domains(
    estimate: &IndexMap<String, EstimateSpecV2>,
    model: &ir::Model,
) -> Result<IndexMap<String, Transform>, String> {
    use super::runner::{estimated_kind, resolve_search_box};
    estimate
        .iter()
        .map(|(name, spec)| {
            let p = model.parameters.iter().find(|p| p.name == *name).ok_or_else(|| {
                format!("parameter '{name}' not in model{}", crate::util::constant_hints(model, [name.as_str()]))
            })?;
            let kind = estimated_kind(p)?;
            let support = crate::roles::support(p, kind);
            let (lower, upper) = resolve_search_box(name, spec.bounds, p)?;
            let search = crate::roles::SearchBox { lower, upper, from: crate::roles::BoxFrom::FitConfig };
            let map = crate::roles::effective_domain(&support, &search, kind).map_err(|e| e.to_string())?;
            Ok((name.clone(), map))
        })
        .collect()
}

/// The step of every name in `domains` under `algorithm`, from its table.
///
/// A written SD is used as written. A name left out, or an entry with no
/// `sd`, takes [`Transform::default_if2_step`] under IF2 and
/// [`Transform::default_mh_step`] under PMMH and MH; a simplex member under
/// IF2 takes its group's default log-ratio step instead. Refused: a table
/// entry for a name not in `domains`, an empty entry, `initial_state` on a
/// simplex member, and a default asked of the identity map.
pub fn resolve_steps(
    algorithm: &Algorithm,
    domains: &IndexMap<String, Transform>,
    simplex_groups: &[config_v2::SimplexGroup],
) -> Result<ResolvedSteps, StepError> {
    match algorithm {
        Algorithm::IF2 { perturbation, .. } => {
            const TABLE: &str = "perturbation";
            not_estimated(TABLE, perturbation.keys(), domains)?;
            if let Some((name, _)) = perturbation.iter().find(|(_, s)| s.sd.is_none() && !s.initial_state) {
                return Err(StepError::EmptyEntry { name: name.clone() });
            }
            let group_of = |name: &str| simplex_groups.iter().position(|g| g.params.iter().any(|p| p == name));
            let mut params = IndexMap::new();
            for (name, map) in domains {
                let spec = perturbation.get(name);
                let from = if spec.and_then(|s| s.sd).is_some() { StepFrom::Written } else { StepFrom::Default };
                let step = match group_of(name) {
                    Some(group) => {
                        if spec.is_some_and(|s| s.initial_state) {
                            return Err(StepError::InitialStateOnSimplexMember { name: name.clone(), group });
                        }
                        If2Step::SimplexMember
                    }
                    None => {
                        let sd = match spec.and_then(|s| s.sd) {
                            Some(sd) => sd,
                            None => default_step(TABLE, name, map, map.default_if2_step())?,
                        };
                        let at = if spec.is_some_and(|s| s.initial_state) {
                            PerturbAt::TimeZeroOnly
                        } else {
                            PerturbAt::EveryObservation
                        };
                        If2Step::Single(Perturbation { sd, at })
                    }
                };
                params.insert(name.clone(), ResolvedPerturbation { step, from });
            }
            let simplex = simplex_groups
                .iter()
                .map(|g| {
                    g.params
                        .iter()
                        .map(|name| {
                            let map = domains.get(name).ok_or_else(|| StepError::NotEstimated {
                                table: "perturbation",
                                name: name.clone(),
                            })?;
                            let resolved = match perturbation.get(name).and_then(|s| s.sd) {
                                Some(sd) => ResolvedProposal { sd, from: StepFrom::Written },
                                None => ResolvedProposal {
                                    sd: default_step(TABLE, name, map, Some(simplex_member_default_step(map)))?,
                                    from: StepFrom::Default,
                                },
                            };
                            Ok((name.clone(), resolved))
                        })
                        .collect::<Result<IndexMap<_, _>, StepError>>()
                })
                .collect::<Result<Vec<_>, StepError>>()?;
            Ok(ResolvedSteps::If2 { params, simplex })
        }
        Algorithm::PMMH { proposal, .. } | Algorithm::Mh { proposal, .. } => {
            const TABLE: &str = "proposal";
            not_estimated(TABLE, proposal.keys(), domains)?;
            let resolved = domains
                .iter()
                .map(|(name, map)| {
                    let r = match proposal.get(name) {
                        Some(&sd) => ResolvedProposal { sd, from: StepFrom::Written },
                        None => ResolvedProposal {
                            sd: default_step(TABLE, name, map, map.default_mh_step())?,
                            from: StepFrom::Default,
                        },
                    };
                    Ok((name.clone(), r))
                })
                .collect::<Result<IndexMap<_, _>, StepError>>()?;
            Ok(ResolvedSteps::Proposal(resolved))
        }
        Algorithm::PGAS { .. }
        | Algorithm::PFilter { .. }
        | Algorithm::NlSbplx(_)
        | Algorithm::NlBobyqa(_)
        | Algorithm::NlLbfgs(_) => Ok(ResolvedSteps::None),
    }
}

fn not_estimated<'a>(
    table: &'static str,
    mut names: impl Iterator<Item = &'a String>,
    domains: &IndexMap<String, Transform>,
) -> Result<(), StepError> {
    match names.find(|n| !domains.contains_key(*n)) {
        Some(name) => Err(StepError::NotEstimated { table, name: name.clone() }),
        None => Ok(()),
    }
}

fn default_step(table: &'static str, name: &str, map: &Transform, default: Option<f64>) -> Result<ZSd, StepError> {
    default.and_then(|d| ZSd::new(d).ok()).ok_or_else(|| StepError::NoDefault {
        table,
        name: name.to_string(),
        kind: match map {
            Transform::Identity => "real or instant",
            _ => "parameter",
        },
    })
}

/// A simplex member's default log-ratio step under IF2: the natural-scale
/// step the member's box gave before support-keyed transforms, read as is on
/// the log-ratio coordinate, so the simplex perturbation is the one it was
/// (`(hi − lo)/6` on [0, 1]).
///
/// LogInterval: ln(hi/lo)/20 converted at the geometric midpoint.
/// LowerBounded: as LogInterval with hi = 1000·lo.
/// Interval: (hi − lo)/6.
/// Identity, UpperBounded: (hi − lo)/6, with ±1e6 for an infinite end.
pub(crate) fn simplex_member_default_step(transform: &Transform) -> f64 {
    let (lower, upper) = transform.bounds();
    let log_auto = |lo: f64, hi: f64| {
        let lo = lo.max(1e-300);
        let hi_val = if hi.is_finite() { hi } else { lo * 1000.0 };
        let log_range = (hi_val / lo).ln();
        let log_sd = log_range / 20.0;
        let midpoint = (lo * hi_val).sqrt();
        midpoint * log_sd
    };
    match *transform {
        Transform::LogInterval { lo, hi } => log_auto(lo, hi),
        Transform::LowerBounded { lo } => log_auto(lo, f64::INFINITY),
        Transform::Interval { lo, hi } => (hi - lo) / 6.0,
        Transform::Identity | Transform::UpperBounded { .. } => {
            let lo = if lower.is_finite() { lower } else { -1e6 };
            let hi = if upper.is_finite() { upper } else { 1e6 };
            (hi - lo) / 6.0
        }
    }
}

impl ResolvedSteps {
    /// IF2's step for each of `order`, by name; `None` for a method that is
    /// not IF2. A name the steps do not hold is an internal error.
    pub fn if2_steps(&self, order: &[EstimatedParam]) -> Option<Result<Vec<If2Step>, String>> {
        let ResolvedSteps::If2 { params, .. } = self else { return None };
        Some(
            order
                .iter()
                .map(|p| {
                    params.get(&p.name).map(|r| r.step).ok_or_else(|| {
                        format!("internal error: no resolved IF2 step for `{}`", p.name)
                    })
                })
                .collect(),
        )
    }

    /// Each simplex group's member SDs on its log-ratio coordinate, in
    /// declaration order; `None` for a method that is not IF2.
    pub fn simplex_sds(&self) -> Option<Vec<Vec<ZSd>>> {
        let ResolvedSteps::If2 { simplex, .. } = self else { return None };
        Some(simplex.iter().map(|g| g.values().map(|r| r.sd).collect()).collect())
    }

    /// PMMH's or MH's initial proposal SD in z for each of `order`, by name;
    /// `None` for a method that is not PMMH or MH.
    pub fn proposal_sd(&self, order: &[EstimatedParam]) -> Option<Result<Vec<f64>, String>> {
        let ResolvedSteps::Proposal(map) = self else { return None };
        Some(
            order
                .iter()
                .map(|p| {
                    map.get(&p.name).map(|r| r.sd.get()).ok_or_else(|| {
                        format!("internal error: no resolved proposal SD for `{}`", p.name)
                    })
                })
                .collect(),
        )
    }

    /// The SD in z each parameter starts at, whether it was written, and
    /// IF2's timing where there is one, by name; empty for a method with no
    /// step. A simplex member's SD is its group's.
    pub fn launch_rows(&self) -> IndexMap<String, (f64, StepFrom, Option<PerturbAt>)> {
        match self {
            ResolvedSteps::If2 { params, simplex } => params
                .iter()
                .map(|(name, r)| {
                    let row = match r.step {
                        If2Step::Single(p) => (p.sd.get(), r.from, Some(p.at)),
                        If2Step::SimplexMember => {
                            let m = simplex.iter().find_map(|g| g.get(name)).expect("a member is in a group");
                            (m.sd.get(), m.from, Some(PerturbAt::TimeZeroOnly))
                        }
                    };
                    (name.clone(), row)
                })
                .collect(),
            ResolvedSteps::Proposal(map) => {
                map.iter().map(|(name, r)| (name.clone(), (r.sd.get(), r.from, None))).collect()
            }
            ResolvedSteps::None => IndexMap::new(),
        }
    }

    /// The steps as the run's provenance records them (`run_inputs.json`,
    /// `steps`): each SD in z, IF2's timing, and whether it was written or
    /// is the default. `from` is recorded here and never keyed; the SDs and
    /// timings are the method identity's.
    pub fn record(&self) -> serde_json::Value {
        use serde_json::json;
        let from = |f: StepFrom| match f {
            StepFrom::Written => "written",
            StepFrom::Default => "default",
        };
        let at = |a: PerturbAt| match a {
            PerturbAt::EveryObservation => "every_observation",
            PerturbAt::TimeZeroOnly => "time_zero_only",
        };
        match self {
            ResolvedSteps::If2 { params, simplex } => json!({
                "kind": "if2",
                "perturbation": params.iter().filter_map(|(name, r)| match r.step {
                    If2Step::Single(p) => Some((name.clone(),
                        json!({ "sd_z": p.sd.get(), "at": at(p.at), "from": from(r.from) }))),
                    If2Step::SimplexMember => None,
                }).collect::<serde_json::Map<_, _>>(),
                "simplex": simplex.iter().map(|g| g.iter().map(|(name, r)|
                    (name.clone(), json!({ "sd_z": r.sd.get(), "from": from(r.from) })))
                    .collect::<serde_json::Map<_, _>>()).collect::<Vec<_>>(),
            }),
            ResolvedSteps::Proposal(map) => json!({
                "kind": "proposal",
                "proposal": map.iter().map(|(name, r)|
                    (name.clone(), json!({ "sd_z": r.sd.get(), "from": from(r.from) })))
                    .collect::<serde_json::Map<_, _>>(),
            }),
            ResolvedSteps::None => json!({ "kind": "none" }),
        }
    }

    /// The estimated parameters IF2 perturbs at t = 0 only, declared
    /// `initial_state`; empty for any other method.
    pub fn initial_state_params(&self) -> Vec<String> {
        match self {
            ResolvedSteps::If2 { params, .. } => params
                .iter()
                .filter(|(_, r)| matches!(r.step, If2Step::Single(Perturbation { at: PerturbAt::TimeZeroOnly, .. })))
                .map(|(n, _)| n.clone())
                .collect(),
            _ => Vec::new(),
        }
    }
}

/// The steps `algorithm` takes on `problem`, posed on `model`: [`domains`],
/// then [`resolve_steps`], each refusal as the message the command prints.
pub fn resolve_for(
    algorithm: &Algorithm,
    problem: &config_v2::Problem,
    model: &ir::Model,
) -> Result<ResolvedSteps, String> {
    let domains = domains(&problem.estimate, model)?;
    resolve_steps(algorithm, &domains, &problem.simplex_groups).map_err(|e| e.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use sim::inference::types::{IF2_LOG_INTERVAL_STEP, MH_DEFAULT_STEP};

    fn method(algorithm: &str, body: &str, table: &str) -> Algorithm {
        toml::from_str(&format!("algorithm = \"{algorithm}\"\nbackend = \"chain_binomial\"\n{body}\n{table}")).unwrap()
    }

    fn if2(table: &str) -> Algorithm {
        method("if2", "chains = 1\nparticles = 10\niterations = 1\ncooling = 0.5", table)
    }

    fn pmmh(table: &str) -> Algorithm {
        method("pmmh", "chains = 1\nparticles = 10\niterations = 1", table)
    }

    fn domains() -> IndexMap<String, Transform> {
        [
            ("beta".to_string(), Transform::LogInterval { lo: 0.05, hi: 1.0 }),
            ("x".to_string(), Transform::Identity),
        ]
        .into_iter()
        .collect()
    }

    fn beta_only() -> IndexMap<String, Transform> {
        domains().into_iter().filter(|(n, _)| n == "beta").collect()
    }

    /// `[method.proposal]` is the initial proposal SD as written, with no
    /// factor; a name left out takes the MH default.
    #[test]
    fn a_proposal_is_used_as_written_and_defaults_to_the_mh_step() {
        let steps = resolve_steps(&pmmh("[proposal]\nbeta = 0.3"), &beta_only(), &[]).unwrap();
        let ResolvedSteps::Proposal(map) = &steps else { panic!("{steps:?}") };
        assert_eq!((map["beta"].sd.get(), map["beta"].from), (0.3, StepFrom::Written));
        let steps = resolve_steps(&pmmh(""), &beta_only(), &[]).unwrap();
        let ResolvedSteps::Proposal(map) = &steps else { panic!("{steps:?}") };
        assert_eq!((map["beta"].sd.get(), map["beta"].from), (MH_DEFAULT_STEP, StepFrom::Default));
    }

    /// IF2: a left-out SD at its map's default, and `initial_state` sets the
    /// time-zero timing.
    #[test]
    fn an_if2_step_is_written_or_its_maps_default() {
        let steps = resolve_steps(&if2("[perturbation]\nbeta = { initial_state = true }"), &beta_only(), &[]).unwrap();
        let ResolvedSteps::If2 { params, .. } = &steps else { panic!("{steps:?}") };
        let If2Step::Single(p) = params["beta"].step else { panic!("beta is no simplex member") };
        assert_eq!((p.sd.get(), p.at, params["beta"].from),
            (IF2_LOG_INTERVAL_STEP, PerturbAt::TimeZeroOnly, StepFrom::Default));
    }

    #[test]
    fn each_unresolvable_table_is_refused() {
        let err = resolve_steps(&if2("[perturbation]\ngamma = 0.1"), &beta_only(), &[]).unwrap_err();
        assert_eq!(err, StepError::NotEstimated { table: "perturbation", name: "gamma".into() });
        let err = resolve_steps(&if2("[perturbation]\nbeta = {}"), &beta_only(), &[]).unwrap_err();
        assert_eq!(err, StepError::EmptyEntry { name: "beta".into() });
        let err = resolve_steps(&pmmh(""), &domains(), &[]).unwrap_err();
        assert!(matches!(&err, StepError::NoDefault { table: "proposal", name, .. } if name == "x"), "{err:?}");
        // A written step on the identity map needs no default.
        resolve_steps(&pmmh("[proposal]\nx = 0.5"), &domains(), &[]).unwrap();
        // A method with no step reads none.
        let pgas = method("pgas", "chains = 1\nparticles = 10\nsweeps = 1", "");
        assert_eq!(resolve_steps(&pgas, &domains(), &[]).unwrap(), ResolvedSteps::None);
    }

    /// The group perturbs its members at t = 0 only, so `initial_state` on a
    /// member is refused.
    #[test]
    fn initial_state_on_a_simplex_member_is_refused() {
        let domains: IndexMap<String, Transform> = ["a", "b"]
            .iter()
            .map(|n| (n.to_string(), Transform::Interval { lo: 0.0, hi: 1.0 }))
            .collect();
        let group = [config_v2::SimplexGroup { params: vec!["a".into(), "b".into()] }];
        let err = resolve_steps(&if2("[perturbation]\na = { initial_state = true }"), &domains, &group).unwrap_err();
        assert_eq!(err, StepError::InitialStateOnSimplexMember { name: "a".into(), group: 0 });
    }
}
