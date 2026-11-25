//! Where each parameter enters a compiled model, read from its IR.
//!
//! A parameter can reach three kinds of surface:
//!
//! - the **initial state**: an expression of `init { }`, a law's arguments
//!   included;
//! - the **dynamics**: a transition's rate, the σ² of an overdispersed draw, a
//!   lineage parent-pool weight, an ODE equation, an intervention's actions,
//!   its parametric `at [...]` times or a reactive trigger's threshold, and
//!   the balance expression;
//! - the **observations**: a stream's likelihood arguments and its
//!   `DerivedExpr` projection.
//!
//! Bindings, per-evaluation bindings, time functions and tables pass a
//! parameter on to whichever surface reads them, at any depth. Hierarchical
//! prior arguments, `quantities`, contrasts, presets and constants are no
//! surface: priors are read by [`use_classes`] instead, and the others are
//! outputs or supply values without referencing a parameter.
//!
//! [`param_surfaces`] destructures [`ir::Model`] and every IR type it walks
//! with each field named, and matches every [`ir::expr::Expr`] variant
//! without a wildcard, so a new field or variant does not compile here until
//! it is classified. A reference the walk cannot resolve (a binding, time
//! function or table absent from the model) gives
//! [`Classification::Uncertain`], under which every parameter is treated as
//! entering the dynamics. An uncertain classification therefore never makes
//! a parameter initial-state-only.
//!
//! [`refuse_for_method`] applies the two checks that depend on the method:
//! IF2's perturbation timing against the initial-state property, and the
//! use-class policy. The rules are
//! `docs/dev/proposals/2026-10-29-method-scoped-settings.md` §4.6.

use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};

use indexmap::IndexMap;
use ir::expr::Expr;
use ir::intervention::{Action, FireSource, InterventionSchedule, TriggerExpr, TriggerThreshold};
use ir::model::{InitCountLaw, InitRealLaw, InitSpec};
use ir::observation::{Likelihood, Projection};
use ir::table::TableSource;
use ir::time_func::{BSplineKnots, InterpSource, TimeFuncKind};
use sim::inference::pmmh::Prior;
use sim::inference::prior::Density;
use sim::inference::types::{If2Step, PerturbAt};

use super::config_v2::Algorithm;
use super::priors_precedence::ResolvedPrior;
use super::steps::ResolvedSteps;

/// One place in the model that reads a parameter, named for a message.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Surface {
    InitialCondition { compartment: String },
    TransitionRate { transition: String },
    Overdispersion { transition: String },
    LineageWeight { transition: String },
    OdeEquation { compartment: String },
    InterventionAction { intervention: String },
    InterventionTimes { intervention: String },
    TriggerThreshold { intervention: String },
    Balance { target: String },
    ObservationLikelihood { stream: String },
    ObservationProjection { stream: String },
}

impl std::fmt::Display for Surface {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Surface::InitialCondition { compartment } => write!(f, "the initial condition of `{compartment}`"),
            Surface::TransitionRate { transition } => write!(f, "the rate of transition `{transition}`"),
            Surface::Overdispersion { transition } => write!(f, "the overdispersion σ² of transition `{transition}`"),
            Surface::LineageWeight { transition } => write!(f, "a lineage parent-pool weight of transition `{transition}`"),
            Surface::OdeEquation { compartment } => write!(f, "the ODE equation of `{compartment}`"),
            Surface::InterventionAction { intervention } => write!(f, "an action of intervention `{intervention}`"),
            Surface::InterventionTimes { intervention } => write!(f, "the `at` times of intervention `{intervention}`"),
            Surface::TriggerThreshold { intervention } => write!(f, "the trigger threshold of intervention `{intervention}`"),
            Surface::Balance { target } => write!(f, "the balance expression for `{target}`"),
            Surface::ObservationLikelihood { stream } => write!(f, "the likelihood of observation stream `{stream}`"),
            Surface::ObservationProjection { stream } => write!(f, "the projection of observation stream `{stream}`"),
        }
    }
}

/// Every parameter each kind of surface reaches, each with the first surface
/// (in IR order) that reaches it.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct ParamSurfaces {
    pub initial_state: BTreeMap<String, Surface>,
    pub dynamics: BTreeMap<String, Surface>,
    pub observations: BTreeMap<String, Surface>,
}

impl ParamSurfaces {
    /// The first surface after t = 0 that reads `name`: a dynamics surface,
    /// else an observation surface.
    pub fn after_t0_use(&self, name: &str) -> Option<&Surface> {
        self.dynamics.get(name).or_else(|| self.observations.get(name))
    }

    /// Whether `name` reaches the initial state and nothing read after t = 0.
    pub fn initial_state_only(&self, name: &str) -> bool {
        self.initial_state.contains_key(name) && self.after_t0_use(name).is_none()
    }

    /// Whether the likelihood depends on `name` directly: through the
    /// initial state, the dynamics or the observations.
    pub fn reaches_likelihood(&self, name: &str) -> bool {
        self.initial_state.contains_key(name) || self.after_t0_use(name).is_some()
    }
}

/// The outcome of [`param_surfaces`].
#[derive(Clone, Debug, PartialEq)]
pub enum Classification {
    Resolved(ParamSurfaces),
    /// A reference the walk could not resolve, and the surface it was read
    /// from. Every parameter is then treated as entering the dynamics.
    Uncertain { reason: String, surface: Surface },
}

/// Classifies every parameter `model` references by the surfaces it reaches.
pub fn param_surfaces(model: &ir::Model) -> Classification {
    let ir::Model {
        // Identity, documentation and calendar metadata: no expressions.
        name: _,
        version: _,
        time_unit: _,
        description: _,
        origin: _,
        origin_rata_die: _,
        compartments: _,
        transitions,
        ode_equations,
        // Read through `TimeFunc`, `ForcingBasis` and `TableLookup` nodes.
        time_functions,
        tables,
        interventions,
        observations,
        // A parameter's own entry holds its value, box and prior. A
        // hierarchical prior's arguments are the prior graph of
        // `use_classes`, not a surface.
        parameters: _,
        // Constant values; `ConstRef` reads no parameter.
        constants: _,
        // Read through `BindingRef` and `PerEvalRef` nodes.
        bindings,
        per_eval_bindings,
        initial_conditions,
        // ∂(initial state)/∂θ: derivatives of the `initial_conditions`
        // expressions walked below, so they name no parameter those do not.
        ic_grad: _,
        output: _,
        simulation: _,
        // Values that replace parameters, not references to them.
        presets: _,
        model_structure: _,
        balance,
        identity_tracked_compartments: _,
        // Outputs: they change no simulated or scored quantity.
        quantities: _,
        contrasts: _,
    } = model;

    let mut walk = Walk {
        bindings: bindings.iter().map(|b| (b.name.as_str(), &b.expr)).collect(),
        per_eval: per_eval_bindings.iter().map(|b| (b.name.as_str(), &b.expr)).collect(),
        time_functions: time_functions.iter().map(|t| (t.name.as_str(), t)).collect(),
        tables: tables.iter().map(|t| (t.name.as_str(), t)).collect(),
        memo: HashMap::new(),
    };
    let mut out = ParamSurfaces::default();
    match classify(&mut walk, &mut out, transitions, ode_equations, interventions, observations,
        initial_conditions, balance.as_ref())
    {
        Ok(()) => Classification::Resolved(out),
        Err((reason, surface)) => Classification::Uncertain { reason, surface },
    }
}

type Uncertain = (String, Surface);

#[allow(clippy::too_many_arguments)]
fn classify<'m>(
    walk: &mut Walk<'m>,
    out: &mut ParamSurfaces,
    transitions: &'m [ir::transition::Transition],
    ode_equations: &'m [ir::ode_equation::OdeEquation],
    interventions: &'m [ir::intervention::Intervention],
    observations: &'m [ir::observation::ObservationModel],
    initial_conditions: &'m ir::model::InitialConditions,
    balance: Option<&'m ir::model::BalanceSpec>,
) -> Result<(), Uncertain> {
    for (compartment, spec) in initial_conditions {
        let surface = Surface::InitialCondition { compartment: compartment.clone() };
        let exprs: Vec<&Expr> = match spec {
            InitSpec::Deterministic(e) => vec![e],
            InitSpec::Count(InitCountLaw::Poisson(ir::observation::PoissonLikelihood { rate })) => vec![&rate.expr],
            InitSpec::Count(InitCountLaw::Binomial(ir::observation::BinomialLikelihood { n, p })) => vec![n, &p.expr],
            InitSpec::Count(InitCountLaw::NegBinomial(ir::observation::NegBinomialLikelihood { mean, dispersion })) => {
                vec![&mean.expr, &dispersion.expr]
            }
            InitSpec::Real(InitRealLaw::Normal(ir::observation::NormalLikelihood { mean, sd })) => {
                vec![&mean.expr, &sd.expr]
            }
        };
        walk.surface(&mut out.initial_state, &surface, exprs)?;
    }

    for t in transitions {
        let ir::transition::Transition {
            name,
            stoichiometry: _,
            rate,
            metadata: _,
            draw_method,
            // Derivatives of `rate`: they name no parameter `rate` does not.
            rate_grad: _,
            rate_state_grad: _,
            lineage,
        } = t;
        walk.surface(&mut out.dynamics, &Surface::TransitionRate { transition: name.clone() }, [rate])?;
        match draw_method {
            ir::transition::DrawMethod::Poisson | ir::transition::DrawMethod::Deterministic => {}
            ir::transition::DrawMethod::Overdispersed { sigma_sq, sigma_sq_grad: _ } => {
                walk.surface(&mut out.dynamics, &Surface::Overdispersion { transition: name.clone() }, [sigma_sq])?;
            }
        }
        if let Some(ir::transition::TransitionLineage { is_lineage_event: _, parent_pool_weights }) = lineage {
            walk.surface(&mut out.dynamics, &Surface::LineageWeight { transition: name.clone() },
                parent_pool_weights.iter().map(|(_, w)| w))?;
        }
    }

    for ir::ode_equation::OdeEquation { compartment, derivative } in ode_equations {
        walk.surface(&mut out.dynamics, &Surface::OdeEquation { compartment: compartment.clone() }, [derivative])?;
    }

    for iv in interventions {
        let ir::intervention::Intervention { name, base_name: _, fire, actions, kind: _ } = iv;
        match fire {
            FireSource::Scheduled(InterventionSchedule::AtTimes(_))
            | FireSource::Scheduled(InterventionSchedule::Recurring(_)) => {}
            FireSource::Scheduled(InterventionSchedule::AtTimesExpr(times)) => {
                walk.surface(&mut out.dynamics, &Surface::InterventionTimes { intervention: name.clone() }, times)?;
            }
            FireSource::Reactive(ir::intervention::ReactiveTrigger { when_, after: _, once: _, cooldown: _ }) => {
                let mut thresholds = Vec::new();
                trigger_thresholds(when_, &mut thresholds);
                for p in thresholds {
                    out.dynamics
                        .entry(p.to_string())
                        .or_insert_with(|| Surface::TriggerThreshold { intervention: name.clone() });
                }
            }
        }
        let surface = Surface::InterventionAction { intervention: name.clone() };
        for action in actions {
            let amount: &Expr = match action {
                Action::FractionTransfer(ir::intervention::FractionTransfer { src: _, dst: _, fraction, fraction_grad: _ }) => fraction,
                Action::AbsoluteTransfer(ir::intervention::AbsoluteTransfer { src: _, dst: _, count }) => count,
                Action::GroupedTransfer(ir::intervention::GroupedTransfer { pairs: _, amount }) => match amount {
                    ir::intervention::GroupedAmount::Fraction { fraction, fraction_grad: _ } => fraction,
                    ir::intervention::GroupedAmount::Count { count } => count,
                },
                Action::Set(ir::intervention::SetAction { compartment: _, value, value_grad: _ }) => value,
                Action::Add(ir::intervention::AddAction { compartment: _, count, count_grad: _ }) => count,
            };
            walk.surface(&mut out.dynamics, &surface, [amount])?;
        }
    }

    if let Some(ir::model::BalanceSpec { target, expr }) = balance {
        walk.surface(&mut out.dynamics, &Surface::Balance { target: target.clone() }, [expr])?;
    }

    for obs in observations {
        let ir::observation::ObservationModel {
            name,
            source: _,
            columns: _,
            scored: _,
            emit_schedule: _,
            stratum: _,
            covers: _,
            projection,
            // ∂projection/∂state: no parameter.
            projection_state_grad: _,
            likelihood,
        } = obs;
        match projection {
            Projection::DerivedExpr(e) => {
                walk.surface(&mut out.observations, &Surface::ObservationProjection { stream: name.clone() }, [e])?;
            }
            Projection::CumulativeFlow(_)
            | Projection::CurrentPop(_)
            | Projection::CurrentPopSum(_)
            | Projection::CumulativeFlowSum(_)
            | Projection::FlowRatio { .. } => {}
        }
        use ir::observation as o;
        let args: Vec<&Expr> = match likelihood {
            Likelihood::Poisson(o::PoissonLikelihood { rate }) => vec![&rate.expr],
            Likelihood::NegBinomial(o::NegBinomialLikelihood { mean, dispersion }) => vec![&mean.expr, &dispersion.expr],
            Likelihood::Normal(o::NormalLikelihood { mean, sd }) => vec![&mean.expr, &sd.expr],
            Likelihood::Binomial(o::BinomialLikelihood { n, p }) => vec![n, &p.expr],
            Likelihood::BetaBinomial(o::BetaBinomialLikelihood { n, alpha, beta }) => vec![n, &alpha.expr, &beta.expr],
            Likelihood::Beta(o::BetaLikelihood { mean, concentration }) => vec![&mean.expr, &concentration.expr],
            Likelihood::Bernoulli(o::BernoulliLikelihood { p }) => vec![&p.expr],
            Likelihood::ZeroInflatedNegBinomial(o::ZeroInflatedNegBinomialLikelihood { mean, dispersion, pi }) => {
                vec![&mean.expr, &dispersion.expr, &pi.expr]
            }
        };
        walk.surface(&mut out.observations, &Surface::ObservationLikelihood { stream: name.clone() }, args)?;
    }
    Ok(())
}

/// The parameters named as reactive-trigger thresholds in `when`.
fn trigger_thresholds<'m>(when: &'m TriggerExpr, out: &mut Vec<&'m str>) {
    match when {
        TriggerExpr::Cmp { lhs, op: _, rhs } => {
            // The trigger's left side is an observed data stream: data, not a
            // model quantity, so it reads no parameter.
            match lhs {
                ir::intervention::TriggerQuantity::Observed { stream: _, window: _, reducer: _ } => {}
            }
            match rhs {
                TriggerThreshold::Const(_) => {}
                TriggerThreshold::Param(p) => out.push(p),
            }
        }
        TriggerExpr::And(a, b) | TriggerExpr::Or(a, b) => {
            trigger_thresholds(a, out);
            trigger_thresholds(b, out);
        }
        TriggerExpr::Not(a) => trigger_thresholds(a, out),
    }
}

/// A named node an expression reads through.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
enum Node<'m> {
    Binding(&'m str),
    PerEval(&'m str),
    TimeFunc(&'m str),
    Table(&'m str),
}

/// The walk's lookups, and the parameters each finished node reaches.
struct Walk<'m> {
    bindings: HashMap<&'m str, &'m Expr>,
    per_eval: HashMap<&'m str, &'m Expr>,
    time_functions: HashMap<&'m str, &'m ir::time_func::TimeFunction>,
    tables: HashMap<&'m str, &'m ir::table::Table>,
    /// Only nodes whose reach is complete: one whose walk met a node still
    /// being walked (a cycle, possible only in hand-built IR) is not kept.
    memo: HashMap<Node<'m>, BTreeSet<String>>,
}

impl<'m> Walk<'m> {
    /// Records every parameter `exprs` reach under `surface` in `into`,
    /// keeping a surface already recorded for a parameter.
    fn surface<I>(&mut self, into: &mut BTreeMap<String, Surface>, surface: &Surface, exprs: I) -> Result<(), Uncertain>
    where
        I: IntoIterator<Item = &'m Expr>,
    {
        let mut params = BTreeSet::new();
        let mut stack = HashSet::new();
        let mut open = HashSet::new();
        for e in exprs {
            self.expr(e, &mut stack, &mut params, &mut open).map_err(|reason| (reason, surface.clone()))?;
        }
        for p in params {
            into.entry(p).or_insert_with(|| surface.clone());
        }
        Ok(())
    }

    /// Adds the parameters `e` reaches to `params`. `stack` holds the nodes
    /// being walked; `open` collects those of them this walk met again.
    fn expr(
        &mut self,
        e: &'m Expr,
        stack: &mut HashSet<Node<'m>>,
        params: &mut BTreeSet<String>,
        open: &mut HashSet<Node<'m>>,
    ) -> Result<(), String> {
        use ir::expr as x;
        match e {
            Expr::Param(x::ParamExpr { param }) => {
                params.insert(param.clone());
            }
            Expr::Const(_)
            | Expr::ConstRef(_)
            | Expr::Pop(_)
            | Expr::PopSum(_)
            | Expr::Time(_)
            | Expr::Dt(_)
            | Expr::Projected(_)
            | Expr::UnitLiteral(_)
            | Expr::ObsColumnRef(_)
            | Expr::ObsAnchor(_) => {}
            Expr::BinOp(x::BinOpWrap { bin_op: x::BinOpExpr { op: _, left, right } }) => {
                self.expr(left, stack, params, open)?;
                self.expr(right, stack, params, open)?;
            }
            Expr::UnOp(x::UnOpWrap { un_op: x::UnOpExpr { op: _, arg } }) => self.expr(arg, stack, params, open)?,
            Expr::Cond(x::CondWrap { cond: x::CondExpr { pred, then, else_ } }) => {
                self.expr(pred, stack, params, open)?;
                self.expr(then, stack, params, open)?;
                self.expr(else_, stack, params, open)?;
            }
            Expr::UncheckedDim(x::UncheckedDimWrap { unchecked_dim: x::UncheckedDimExpr { inner, dim: _, reason: _ } }) => {
                self.expr(inner, stack, params, open)?
            }
            Expr::Reduce(x::ReduceWrap { reduce }) => {
                for t in reduce {
                    self.expr(t, stack, params, open)?;
                }
            }
            Expr::TableLookup(x::TableLookupWrap { table_lookup: x::TableLookupExpr { table, indices } }) => {
                for i in indices {
                    self.expr(i, stack, params, open)?;
                }
                self.node(Node::Table(table), stack, params, open)?;
            }
            Expr::TimeFunc(x::TimeFuncWrap { time_func: x::TimeFuncRef { name } }) => {
                self.node(Node::TimeFunc(name), stack, params, open)?
            }
            Expr::ForcingBasis(x::ForcingBasisWrap { forcing_basis: x::ForcingBasisRef { name, index: _, space: _ } }) => {
                self.node(Node::TimeFunc(name), stack, params, open)?
            }
            Expr::BindingRef(x::BindingRefWrap { binding_ref }) => {
                self.node(Node::Binding(binding_ref), stack, params, open)?
            }
            Expr::PerEvalRef(x::PerEvalRefWrap { per_eval_ref }) => {
                self.node(Node::PerEval(per_eval_ref), stack, params, open)?
            }
        }
        Ok(())
    }

    /// Adds the parameters `node` reaches to `params`.
    fn node(
        &mut self,
        node: Node<'m>,
        stack: &mut HashSet<Node<'m>>,
        params: &mut BTreeSet<String>,
        open: &mut HashSet<Node<'m>>,
    ) -> Result<(), String> {
        if let Some(done) = self.memo.get(&node) {
            params.extend(done.iter().cloned());
            return Ok(());
        }
        if stack.contains(&node) {
            // Its parameters are added when its own walk, further up, ends.
            open.insert(node);
            return Ok(());
        }
        let body = self.body(node)?;
        stack.insert(node);
        let mut own = BTreeSet::new();
        let mut own_open = HashSet::new();
        let walked = body.into_iter().try_for_each(|e| self.expr(e, stack, &mut own, &mut own_open));
        stack.remove(&node);
        walked?;
        own_open.remove(&node);
        params.extend(own.iter().cloned());
        if own_open.is_empty() {
            self.memo.insert(node, own);
        } else {
            open.extend(own_open);
        }
        Ok(())
    }

    /// The expressions `node` evaluates, or why it cannot be found.
    fn body(&self, node: Node<'m>) -> Result<Vec<&'m Expr>, String> {
        match node {
            Node::Binding(n) => self.bindings.get(n).map(|e| vec![*e])
                .ok_or_else(|| format!("binding `{n}` is referenced and not defined")),
            Node::PerEval(n) => self.per_eval.get(n).map(|e| vec![*e])
                .ok_or_else(|| format!("per-evaluation binding `{n}` is referenced and not defined")),
            Node::TimeFunc(n) => self.time_functions.get(n).map(|t| time_function_exprs(t))
                .ok_or_else(|| format!("time function `{n}` is referenced and not defined")),
            Node::Table(n) => {
                let t = self.tables.get(n).ok_or_else(|| format!("table `{n}` is referenced and not defined"))?;
                let ir::table::Table { name: _, source, out_of_bounds: _, cell_kind: _ } = *t;
                Ok(match source {
                    TableSource::Inline { values } => values.iter().collect(),
                    // Bound from a file of numbers when the model is run.
                    TableSource::External { external: _, shape: _ } => Vec::new(),
                })
            }
        }
    }
}

/// Every expression a time function evaluates: its shape's and its lag.
fn time_function_exprs(t: &ir::time_func::TimeFunction) -> Vec<&Expr> {
    use ir::time_func as tf;
    let tf::TimeFunction { name: _, kind, dim: _, lag, data_source: _ } = t;
    let mut out: Vec<&Expr> = lag.iter().collect();
    match kind {
        TimeFuncKind::Sinusoidal(tf::Sinusoidal { amplitude, period, phase, baseline }) => {
            out.extend([amplitude, period, phase, baseline]);
        }
        TimeFuncKind::Piecewise(tf::Piecewise { breakpoints, values }) => {
            out.extend(breakpoints);
            out.extend(values);
        }
        TimeFuncKind::Interpolated(tf::Interpolated { source, method: _ }) => match source {
            InterpSource::Knots { times, values } => {
                out.extend(times);
                out.extend(values);
            }
            // A series read from a file; its scale is a number.
            InterpSource::External(_) => {}
        },
        TimeFuncKind::Periodic(tf::Periodic { period, values }) => {
            out.push(period);
            out.extend(values);
        }
        TimeFuncKind::Fourier(tf::Fourier { period, harmonics }) => {
            out.push(period);
            for (a, b) in harmonics {
                out.extend([a, b]);
            }
        }
        TimeFuncKind::PeriodicSpline(tf::PeriodicSpline { period, n_basis: _, degree: _, coefs, link: _ }) => {
            out.push(period);
            out.extend(coefs);
        }
        TimeFuncKind::BSpline(tf::BSpline { knots, degree: _, coefs, link: _ }) => {
            match knots {
                BSplineKnots::Uniform { lower, upper, n_intervals: _ } => out.extend([lower, upper]),
                BSplineKnots::Explicit { breakpoints } => out.extend(breakpoints),
            }
            match coefs {
                tf::SplineCoefs::Explicit { coefs } => out.extend(coefs),
                tf::SplineCoefs::Smooth { level, slope, scale, modes, per } => {
                    out.extend([level, slope, scale, per]);
                    out.extend(modes);
                }
            }
        }
    }
    out
}

// ─── Use classes ─────────────────────────────────────────────────────────────

/// How an estimated parameter can affect a statistical quantity.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum UseClass {
    /// Reaches the initial state, the dynamics or the observations.
    Likelihood,
    /// No direct connection, but a hyperparameter, at any depth, of an
    /// estimated `Likelihood` parameter.
    Hierarchical,
    /// No likelihood connection, direct or through the parameters whose
    /// priors it enters, and a proper prior of its own.
    PriorOnly,
    /// No likelihood connection and a flat prior.
    Unused,
}

/// The use class of each estimated parameter in `priors`, in their order.
///
/// `priors` are the resolved priors of the estimated parameters. An edge
/// h → p of the hierarchical-prior graph is an argument of `p`'s resolved
/// hierarchical prior that references `h`; only an estimated `p` has one.
/// The graph is followed to any depth. Under
/// [`Classification::Uncertain`], or when a prior argument holds a reference
/// that cannot be resolved, every parameter is `Likelihood`.
pub fn use_classes(
    classification: &Classification,
    priors: &[ResolvedPrior],
    model: &ir::Model,
) -> IndexMap<String, UseClass> {
    let all = |c| priors.iter().map(|p| (p.param.clone(), c)).collect();
    let Classification::Resolved(surfaces) = classification else { return all(UseClass::Likelihood) };

    // Parents of each estimated parameter: the names its hierarchical prior's
    // arguments reach.
    let mut walk = Walk {
        bindings: model.bindings.iter().map(|b| (b.name.as_str(), &b.expr)).collect(),
        per_eval: model.per_eval_bindings.iter().map(|b| (b.name.as_str(), &b.expr)).collect(),
        time_functions: model.time_functions.iter().map(|t| (t.name.as_str(), t)).collect(),
        tables: model.tables.iter().map(|t| (t.name.as_str(), t)).collect(),
        memo: HashMap::new(),
    };
    let mut children: HashMap<String, Vec<&str>> = HashMap::new();
    for rp in priors {
        if !rp.prior.is_hierarchical() {
            continue;
        }
        let Some(h) = model.parameters.iter().find(|p| p.name == rp.param).and_then(|p| p.hierarchical()) else {
            continue;
        };
        let mut parents = BTreeSet::new();
        for arg in h.args.values() {
            if walk.expr(arg, &mut HashSet::new(), &mut parents, &mut HashSet::new()).is_err() {
                return all(UseClass::Likelihood);
            }
        }
        for parent in parents {
            children.entry(parent).or_default().push(&rp.param);
        }
    }

    priors
        .iter()
        .map(|rp| {
            let name = rp.param.as_str();
            let class = if surfaces.reaches_likelihood(name) {
                UseClass::Likelihood
            } else if feeds_likelihood(name, &children, surfaces) {
                UseClass::Hierarchical
            } else if matches!(rp.prior, Prior::Fixed(Density::Flat)) {
                UseClass::Unused
            } else {
                UseClass::PriorOnly
            };
            (rp.param.clone(), class)
        })
        .collect()
}

/// Whether some estimated parameter whose prior `name` enters, at any depth,
/// reaches the likelihood.
fn feeds_likelihood(name: &str, children: &HashMap<String, Vec<&str>>, surfaces: &ParamSurfaces) -> bool {
    let mut seen = HashSet::from([name]);
    let mut todo = vec![name];
    while let Some(n) = todo.pop() {
        for &child in children.get(n).into_iter().flatten() {
            if surfaces.reaches_likelihood(child) {
                return true;
            }
            if seen.insert(child) {
                todo.push(child);
            }
        }
    }
    false
}

/// The estimated parameters whose priors `name`'s argument enters, for a
/// message.
fn prior_children<'a>(name: &str, priors: &'a [ResolvedPrior], model: &ir::Model) -> Vec<&'a str> {
    priors
        .iter()
        .filter(|rp| rp.prior.is_hierarchical())
        .filter(|rp| {
            model.parameters.iter().find(|p| p.name == rp.param).and_then(|p| p.hierarchical())
                .is_some_and(|h| h.args.values().any(|a| expr_names_param(a, name)))
        })
        .map(|rp| rp.param.as_str())
        .collect()
}

fn expr_names_param(e: &Expr, name: &str) -> bool {
    let mut found = BTreeSet::new();
    let mut walk = Walk {
        bindings: HashMap::new(),
        per_eval: HashMap::new(),
        time_functions: HashMap::new(),
        tables: HashMap::new(),
        memo: HashMap::new(),
    };
    let _ = walk.expr(e, &mut HashSet::new(), &mut found, &mut HashSet::new());
    found.contains(name)
}

// ─── The checks a method makes ───────────────────────────────────────────────

/// Refuses the estimated parameters `algorithm` cannot use as the run asks,
/// and returns the warnings for those it accepts with a caveat.
///
/// Under IF2, from `steps`: `initial_state` on a parameter that reaches the
/// dynamics or the observations is refused, naming the first such surface,
/// and an initial-state-only parameter without `initial_state` is refused.
/// Under [`Classification::Uncertain`] `initial_state` is refused on every
/// parameter and the second refusal is not raised. Simplex members are
/// outside both: their group perturbs them at t = 0. No other method has a
/// perturbation timing, so none is checked for one.
///
/// The use-class policy, from `classes`: IF2 and the `nl-*` optimizers,
/// which read no prior, accept only [`UseClass::Likelihood`]. PMMH, MH and
/// PGAS accept [`UseClass::Hierarchical`], warn on [`UseClass::PriorOnly`]
/// and refuse [`UseClass::Unused`]. `pfilter` estimates nothing and is not
/// checked.
///
/// Every refusal names its parameter; all are returned together.
pub fn refuse_for_method(
    algorithm: &Algorithm,
    steps: &ResolvedSteps,
    classification: &Classification,
    classes: &IndexMap<String, UseClass>,
    priors: &[ResolvedPrior],
    model: &ir::Model,
) -> Result<Vec<String>, String> {
    let method = algorithm.method_name();
    let mut refused = Vec::new();
    let mut warnings = Vec::new();

    if let ResolvedSteps::If2 { params, .. } = steps {
        for (name, r) in params {
            let If2Step::Single(p) = r.step else { continue };
            match (classification, p.at) {
                (Classification::Resolved(s), PerturbAt::TimeZeroOnly) => {
                    if let Some(surface) = s.after_t0_use(name) {
                        refused.push(format!(
                            "`{name}` has `initial_state = true` in [method.perturbation], and it \
                             enters the model after t = 0, in {surface}. IF2 would perturb it at \
                             t = 0 only, which leaves it almost unexplored. Delete \
                             `initial_state` on `{name}`."));
                    }
                }
                (Classification::Resolved(s), PerturbAt::EveryObservation) => {
                    if s.initial_state_only(name) {
                        let surface = &s.initial_state[name];
                        refused.push(format!(
                            "`{name}` enters only the initial state ({surface}). IF2 perturbing \
                             it before every observation moves it with nothing to select it \
                             after t = 0. Write `{name} = {{ initial_state = true }}` in \
                             [method.perturbation], with its `sd` if it has one."));
                    }
                }
                (Classification::Uncertain { reason, surface }, PerturbAt::TimeZeroOnly) => {
                    refused.push(format!(
                        "`{name}` has `initial_state = true` in [method.perturbation], and \
                         whether it enters only the initial state cannot be read from the \
                         compiled model: {reason}, in {surface}. `initial_state` is refused \
                         for every parameter of such a model. Delete it on `{name}`."));
                }
                (Classification::Uncertain { .. }, PerturbAt::EveryObservation) => {}
            }
        }
    }

    let reads_prior = match algorithm {
        Algorithm::PMMH { .. } | Algorithm::Mh { .. } | Algorithm::PGAS { .. } => true,
        Algorithm::IF2 { .. } | Algorithm::NlSbplx(_) | Algorithm::NlBobyqa(_) | Algorithm::NlLbfgs(_) => false,
        Algorithm::PFilter { .. } => return finish(refused, warnings),
    };
    const NOWHERE: &str = "it enters no initial condition, rate, intervention or observation";
    for (name, class) in classes {
        match (class, reads_prior) {
            (UseClass::Likelihood, _) | (UseClass::Hierarchical, true) => {}
            (UseClass::Hierarchical, false) => {
                let children = prior_children(name, priors, model)
                    .iter().map(|c| format!("`{c}`")).collect::<Vec<_>>().join(", ");
                refused.push(format!(
                    "`{name}` is estimated, and {NOWHERE}: it appears only in the prior of \
                     {children}. `{method}` reads no prior, so `{name}` has no effect on what it \
                     computes. Fix `{name}` in [fixed], or fit with a method that reads priors \
                     (pmmh, mh or pgas)."));
            }
            (UseClass::PriorOnly | UseClass::Unused, false) => refused.push(format!(
                "`{name}` is estimated, and {NOWHERE}, nor the prior of a parameter that does. \
                 `{method}` reads only the likelihood, so `{name}` has no effect on what it \
                 computes. Fix `{name}` in [fixed], or remove it from [estimate].")),
            (UseClass::PriorOnly, true) => warnings.push(format!(
                "`{name}`'s marginal posterior equals its prior: it does not connect to the \
                 likelihood")),
            (UseClass::Unused, true) => refused.push(format!(
                "`{name}` is estimated, and {NOWHERE}, nor the prior of a parameter that does, \
                 and its own prior is flat. Its posterior is then that flat density, which is \
                 not a distribution. Fix `{name}` in [fixed], remove it from [estimate], or \
                 give it a proper prior.")),
        }
    }
    finish(refused, warnings)
}

fn finish(refused: Vec<String>, warnings: Vec<String>) -> Result<Vec<String>, String> {
    if refused.is_empty() {
        Ok(warnings)
    } else {
        Err(refused.join("\n\n"))
    }
}

#[cfg(test)]
mod tests;
