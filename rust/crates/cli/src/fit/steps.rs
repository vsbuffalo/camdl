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
use sim::inference::types::{If2Step, PerturbAt, Perturbation, Transform, ZSd, SIMPLEX_LOG_RATIO_STEP};

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
    estimate
        .iter()
        .map(|(name, spec)| Ok((name.clone(), domain_of(name, spec.bounds, model)?)))
        .collect()
}

/// The map onto one estimated parameter's effective domain, from its
/// `[estimate]` bounds and its declaration in `model` ([`domains`]).
pub fn domain_of(name: &str, bounds: Option<(f64, f64)>, model: &ir::Model) -> Result<Transform, String> {
    use super::runner::{estimated_kind, resolve_search_box};
    let p = model.parameters.iter().find(|p| p.name == name).ok_or_else(|| {
        format!("parameter '{name}' not in model{}", crate::util::constant_hints(model, [name]))
    })?;
    let kind = estimated_kind(p)?;
    let support = crate::roles::support(p, kind);
    let (lower, upper) = resolve_search_box(name, bounds, p)?;
    let search = crate::roles::SearchBox { lower, upper, from: crate::roles::BoxFrom::FitConfig };
    crate::roles::effective_domain(&support, &search, kind).map_err(|e| e.to_string())
}

/// Where θ₀, the point a natural-scale step and a step in z are converted
/// at, came from.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Theta0From {
    /// The entry's `[estimate]` `start`.
    Start,
    /// The model's value for the parameter, the base point the chains start
    /// around when no start is written.
    ModelValue,
    /// The middle of the box: arithmetic on the interval map, geometric on
    /// the log-interval map.
    BoxMidpoint,
}

/// The point one parameter's natural-scale step and its step in z are
/// converted at (`docs/dev/proposals/2026-10-29-method-scoped-settings.md`
/// §7.3), and where it came from.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Theta0 {
    pub value: f64,
    pub from: Theta0From,
}

impl Theta0 {
    /// "at start 0.25", "at the model's value 0.3", "at the box middle
    /// 0.2236".
    pub fn describe(&self) -> String {
        match self.from {
            Theta0From::Start => format!("at start {}", self.value),
            Theta0From::ModelValue => format!("at the model's value {}", self.value),
            Theta0From::BoxMidpoint => format!("at the box middle {}", sig(self.value, 4)),
        }
    }
}

/// θ₀ for a parameter on `map`: its written `start`, else the model's
/// value, else the middle of a box finite at both ends. `None` for a
/// half-bounded map with no start and no model value, which has no point to
/// take the slope at. The identity map's slope is 1 everywhere, so it
/// converts at any point, and takes 0 when nothing else is given.
pub fn conversion_point(start: Option<f64>, model_value: Option<f64>, map: &Transform) -> Option<Theta0> {
    if let Some(value) = start {
        return Some(Theta0 { value, from: Theta0From::Start });
    }
    if let Some(value) = model_value {
        return Some(Theta0 { value, from: Theta0From::ModelValue });
    }
    let value = match *map {
        Transform::Interval { lo, hi } => 0.5 * (lo + hi),
        Transform::LogInterval { lo, hi } => (lo * hi).sqrt(),
        Transform::Identity => 0.0,
        Transform::LowerBounded { .. } | Transform::UpperBounded { .. } => return None,
    };
    Some(Theta0 { value, from: Theta0From::BoxMidpoint })
}

/// Each estimated parameter's map and its conversion point
/// ([`conversion_point`]), in `[estimate]` order: what the launch table and
/// the run record convert each step at.
pub fn conversion_points(
    estimate: &IndexMap<String, EstimateSpecV2>,
    model: &ir::Model,
) -> Result<IndexMap<String, (Transform, Option<Theta0>)>, String> {
    estimate
        .iter()
        .map(|(name, spec)| {
            let map = domain_of(name, spec.bounds, model)?;
            let value = model.parameters.iter().find(|p| p.name == *name).and_then(|p| p.value.resolved_value());
            Ok((name.clone(), (map, conversion_point(spec.start, value, &map))))
        })
        .collect()
}

/// The launch table's natural-scale equivalent of a step of `z_sd` on
/// `map`, at `theta0`: "≈ 0.0200 natural at start 0.25", with "start near
/// bound" when θ₀ is within 1% of the box's width from a finite end, where
/// the slope, and so the natural step, is small. With no θ₀ there is no
/// equivalent to print.
pub fn natural_at_start(map: &Transform, z_sd: f64, theta0: Option<Theta0>) -> String {
    let Some(t0) = theta0.filter(|t| within(map, t.value)) else {
        return "(no start in its box to take its natural-scale equivalent at)".to_string();
    };
    let natural = sim::inference::diagnostic::format_start_value(map.natural_from_step(z_sd, t0.value));
    let (lo, hi) = map.bounds();
    let near = lo.is_finite() && hi.is_finite()
        && ((t0.value - lo) / (hi - lo) < 0.01 || (hi - t0.value) / (hi - lo) < 0.01);
    format!("≈ {natural} natural {}{}", t0.describe(), if near { ", start near bound" } else { "" })
}

/// Whether `theta` lies in `map`'s domain, ends included: where its slope
/// can be taken.
fn within(map: &Transform, theta: f64) -> bool {
    let (lo, hi) = map.bounds();
    lo <= theta && theta <= hi
}

/// `x` to `n` significant digits, in scientific notation below 10⁻³ and
/// from 10⁵.
pub(crate) fn sig(x: f64, n: usize) -> String {
    if x == 0.0 || !x.is_finite() {
        return x.to_string();
    }
    if !(1e-3..1e5).contains(&x.abs()) {
        return format!("{x:.prec$e}", prec = n.saturating_sub(1));
    }
    let magnitude = x.abs().log10().floor() as i32;
    let decimals = (n as i32 - 1 - magnitude).max(0) as usize;
    format!("{x:.decimals$}")
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
                            if !domains.contains_key(name) {
                                return Err(StepError::NotEstimated { table: TABLE, name: name.clone() });
                            }
                            let resolved = match perturbation.get(name).and_then(|s| s.sd) {
                                Some(sd) => ResolvedProposal { sd, from: StepFrom::Written },
                                None => ResolvedProposal { sd: simplex_member_default_step(), from: StepFrom::Default },
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

/// A simplex member's default step under IF2, on the group's log-ratio
/// coordinate: [`SIMPLEX_LOG_RATIO_STEP`], whatever the member's own box,
/// since the group perturbs the log-ratios and not the members' own maps.
pub(crate) fn simplex_member_default_step() -> ZSd {
    ZSd::new(SIMPLEX_LOG_RATIO_STEP).expect("a positive constant")
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
    /// `steps`): each parameter's map, its SD in z, IF2's timing, the point
    /// θ₀ it is converted at (from `points`, [`conversion_points`]) with the
    /// natural-scale step it is there, and whether it was written or is the
    /// default. `from` is recorded here and never keyed; the SDs and timings
    /// are the method identity's, and the map and θ₀ follow from the problem
    /// and the written starts. θ₀ and the natural step are null for a
    /// half-bounded map with no start and no model value, and for a θ₀
    /// outside the box. A simplex member's
    /// SD is on its group's log-ratio coordinate, so it has no natural-scale
    /// equivalent.
    pub fn record(&self, points: &IndexMap<String, (Transform, Option<Theta0>)>) -> serde_json::Value {
        use serde_json::json;
        let from = |f: StepFrom| match f {
            StepFrom::Written => "written",
            StepFrom::Default => "default",
        };
        let at = |a: PerturbAt| match a {
            PerturbAt::EveryObservation => "every_observation",
            PerturbAt::TimeZeroOnly => "time_zero_only",
        };
        let at_start = |name: &str, sd_z: f64| -> (serde_json::Value, serde_json::Value) {
            match points.get(name) {
                Some((map, Some(t0))) if within(map, t0.value) =>
                    (json!(t0.value), json!(map.natural_from_step(sd_z, t0.value))),
                _ => (serde_json::Value::Null, serde_json::Value::Null),
            }
        };
        let maps = |names: &mut dyn Iterator<Item = &String>| -> serde_json::Map<String, serde_json::Value> {
            names.filter_map(|n| points.get(n).map(|(m, _)| (n.clone(), json!(m.tag())))).collect()
        };
        match self {
            ResolvedSteps::If2 { params, simplex } => json!({
                "kind": "if2",
                "map": maps(&mut params.keys()),
                "perturbation": params.iter().filter_map(|(name, r)| match r.step {
                    If2Step::Single(p) => {
                        let (start, natural) = at_start(name, p.sd.get());
                        Some((name.clone(), json!({ "sd_z": p.sd.get(), "at": at(p.at),
                            "start": start, "natural_at_start": natural, "from": from(r.from) })))
                    }
                    If2Step::SimplexMember => None,
                }).collect::<serde_json::Map<_, _>>(),
                "simplex": simplex.iter().map(|g| g.iter().map(|(name, r)|
                    (name.clone(), json!({ "sd_z": r.sd.get(), "from": from(r.from) })))
                    .collect::<serde_json::Map<_, _>>()).collect::<Vec<_>>(),
            }),
            ResolvedSteps::Proposal(map) => json!({
                "kind": "proposal",
                "map": maps(&mut map.keys()),
                "proposal": map.iter().map(|(name, r)| {
                    let (start, natural) = at_start(name, r.sd.get());
                    (name.clone(), json!({ "sd_z": r.sd.get(), "start": start,
                        "natural_at_start": natural, "from": from(r.from) }))
                }).collect::<serde_json::Map<_, _>>(),
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

    /// D20: a simplex member's default IF2 step is one log-ratio constant,
    /// 1/6, whatever its box: a member on the log-interval map of [0.05, 1]
    /// and one on the interval map of [0, 0.5] both take it (their boxes gave
    /// 0.0335 and 0.0833 before). A written member SD is used as written.
    #[test]
    fn a_simplex_members_default_step_is_one_log_ratio_constant() {
        let domains: IndexMap<String, Transform> = [
            ("a".to_string(), Transform::LogInterval { lo: 0.05, hi: 1.0 }),
            ("b".to_string(), Transform::Interval { lo: 0.0, hi: 0.5 }),
            ("c".to_string(), Transform::Interval { lo: 0.0, hi: 1.0 }),
        ].into_iter().collect();
        let group = [config_v2::SimplexGroup { params: vec!["a".into(), "b".into(), "c".into()] }];
        let steps = resolve_steps(&if2("[perturbation]\nc = 0.3"), &domains, &group).unwrap();
        let sds: Vec<f64> = steps.simplex_sds().unwrap()[0].iter().map(|s| s.get()).collect();
        assert_eq!(sds, vec![1.0 / 6.0, 1.0 / 6.0, 0.3]);
    }

    /// θ₀ is the written start, else the model's value, else the box's
    /// middle (geometric on the log-interval map); a half-bounded map with
    /// neither has none.
    #[test]
    fn the_conversion_point_is_the_start_then_the_model_value_then_the_middle() {
        let log = Transform::LogInterval { lo: 0.05, hi: 1.0 };
        let at = |start, value, map: &Transform| conversion_point(start, value, map).map(|t| (t.value, t.from));
        assert_eq!(at(Some(0.25), Some(0.5), &log), Some((0.25, Theta0From::Start)));
        assert_eq!(at(None, Some(0.5), &log), Some((0.5, Theta0From::ModelValue)));
        assert_eq!(at(None, None, &log), Some((0.05f64.sqrt(), Theta0From::BoxMidpoint)));
        assert_eq!(at(None, None, &Transform::Interval { lo: 0.0, hi: 0.5 }), Some((0.25, Theta0From::BoxMidpoint)));
        assert_eq!(at(None, None, &Transform::LowerBounded { lo: 0.0 }), None);
        assert_eq!(at(Some(3.0), None, &Transform::LowerBounded { lo: 0.0 }), Some((3.0, Theta0From::Start)));
    }

    /// `run_inputs.json`'s `steps` gives each step's map, its start and the
    /// natural-scale step it is there: 0.107 in z on the log-interval map of
    /// [0.05, 1] at 0.25 (|dθ/dz| = 0.18619) is 0.01992. A map with no
    /// start records nulls.
    #[test]
    fn the_run_record_gives_each_step_at_its_start() {
        let steps = resolve_steps(&if2("[perturbation]\nbeta = 0.107"), &beta_only(), &[]).unwrap();
        let points: IndexMap<String, (Transform, Option<Theta0>)> = [(
            "beta".to_string(),
            (Transform::LogInterval { lo: 0.05, hi: 1.0 }, Some(Theta0 { value: 0.25, from: Theta0From::Start })),
        )].into_iter().collect();
        let r = steps.record(&points);
        assert_eq!(r["map"]["beta"], "log_interval");
        let beta = &r["perturbation"]["beta"];
        assert_eq!((beta["sd_z"].as_f64(), beta["start"].as_f64(), beta["from"].as_str()),
            (Some(0.107), Some(0.25), Some("written")));
        let natural = beta["natural_at_start"].as_f64().unwrap();
        assert!((natural - 0.107 * 0.186_19).abs() < 1e-5, "{natural}");
        let pmmh_steps = resolve_steps(&pmmh(""), &beta_only(), &[]).unwrap();
        let no_start: IndexMap<String, (Transform, Option<Theta0>)> =
            [("beta".to_string(), (Transform::LowerBounded { lo: 0.0 }, None))].into_iter().collect();
        let r = pmmh_steps.record(&no_start);
        assert!(r["proposal"]["beta"]["start"].is_null() && r["proposal"]["beta"]["natural_at_start"].is_null(), "{r}");
        assert_eq!(r["map"]["beta"], "lower_bounded");
    }

    /// The launch table's natural equivalent is taken at θ₀ and names it;
    /// a start within 1% of the box's width from an end is marked.
    #[test]
    fn the_launch_tables_natural_step_is_at_the_start() {
        let map = Transform::Interval { lo: 0.0, hi: 1.0 };
        let at = |value| Some(Theta0 { value, from: Theta0From::Start });
        // |dθ/dz| = 0.1 × 0.9 = 0.09 at 0.1, so 0.1 in z is 0.009.
        assert_eq!(natural_at_start(&map, 0.1, at(0.1)), "≈ 0.0090 natural at start 0.1");
        assert!(natural_at_start(&map, 0.1, at(0.005)).ends_with("at start 0.005, start near bound"));
        assert!(natural_at_start(&map, 0.1, None).contains("no start"));
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
