//! Generated-quantities evaluator (proposal 2026-06-25): fold a finished
//! trajectory into per-draw quantity values. **Pure** — no RNG, no IO, no
//! inference coupling. Each quantity's state `Expr` is resolved ONCE against the
//! compiled model (the same `resolve_expr` path the obs scorer uses), then per
//! draw the series is evaluated over the trajectory snapshots and the reduction
//! folded. `Derived` reduction-arithmetic is evaluated over prior leaves' scalar
//! values (declaration order = topological).
//!
//! v1: latent-state source only. The `observations.<stream>` source (v1.1) and
//! the banding/IO (cli) are not here.

use std::collections::HashMap;

use ir::expr::{BinOp, UnOp};
use ir::quantity::{
    Quantity, QuantityBody, QuantitySource, ScalarExpr, TemporalReduce, TimeReduce, ValueReduce,
};
use ir::table::OobPolicy;

use crate::compiled_model::CompiledModel;
use crate::propensity::EvalCtx;
use crate::resolved_expr::{eval_resolved, resolve_expr, ResolveCtx, ResolvedExpr};
use crate::state::Trajectory;

/// One quantity's value for one draw: a series (one value per output snapshot)
/// or a scalar (possibly right-censored).
#[derive(Debug, Clone, PartialEq)]
pub enum QuantityResult {
    Series(Vec<f64>),
    Scalar(QuantityDrawValue),
}

/// A scalar quantity's per-draw value. A `Time` reduction whose crossing never
/// occurred within the trajectory window is `Censored` (right-censoring) — banding
/// excludes it and reports `n_censored`, rather than fabricating a time. A
/// non-finite `Value` reduction stays `Value(NaN)`; the banding layer rejects it
/// (a non-finite arithmetic result is a bug, not censoring).
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum QuantityDrawValue {
    Value(f64),
    Censored,
}

// ── Resolved programs (built once at `new`) ─────────────────────────────────────

/// What a `Reduced` quantity's series comes from.
enum QSource {
    /// A resolved state expression, evaluated against each trajectory snapshot.
    State(ResolvedExpr),
    /// The simulated observation series of a declared stream (`y_sim`), supplied
    /// per draw via [`ObsSeriesSet`]. v1.1.
    Observation(String),
}

enum QProgram {
    Reduced { source: QSource, reduce: Option<RReduce> },
    /// Reduction arithmetic over prior leaves' scalar values.
    Derived(RScalar),
}

/// The per-draw simulated observation series the run already drew, keyed by stream
/// name (v1.1). An `Observation`-source quantity reduces these — the SAME `y_sim`
/// the run published, never a fresh draw. `None` to `eval_draw` (or a missing
/// stream) yields an empty series for an `Observation` source.
pub struct ObsSeriesSet {
    /// Stream name → its `(obs times, per-time y_sim values)` (both same length).
    /// Per-stream times: streams may have different observation cadences.
    pub streams: std::collections::HashMap<String, (Vec<f64>, Vec<f64>)>,
}

enum RReduce {
    Final,
    Max,
    Min,
    Mean,
    CountAbove(ResolvedExpr),
    CountBelow(ResolvedExpr),
    ValueAt(RAnchor),
    TimeOfMax,
    TimeOfMin,
    FirstAbove(ResolvedExpr),
    FirstBelow(ResolvedExpr),
    LastAbove(ResolvedExpr),
    LastBelow(ResolvedExpr),
    Integral,
}

/// A resolved `value_at` anchor. `Expr` is evaluated once per draw (a constant
/// or param expression; params are fixed within a draw); `Obs` carries the
/// symbolic observation anchor plus its compile-folded offset, and needs the
/// caller-resolved observation times — a data-free caller must gate on
/// [`QuantityEvaluator::references_obs_anchor`] BEFORE `eval_draw`.
enum RAnchor {
    Expr(ResolvedExpr),
    Obs(ir::anchor::AnchoredTime),
}

/// The run's resolved observation anchors: the min and max observation time
/// over the run's bound streams. Both are carried because a model may read at
/// either end, and resolving only the one a given model happens to use would
/// re-introduce the "which anchor did the caller mean" question at every call
/// site.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ObsAnchorTimes {
    pub first: f64,
    pub last: f64,
}

impl ObsAnchorTimes {
    /// Fold a run's observation times into the pair. `None` when there are no
    /// observation times at all — the caller then has nothing to anchor to and
    /// must refuse rather than pass a fabricated value.
    pub fn of_times(times: impl IntoIterator<Item = f64>) -> Option<Self> {
        let mut it = times.into_iter();
        let t0 = it.next()?;
        let (first, last) = it.fold((t0, t0), |(lo, hi), t| (lo.min(t), hi.max(t)));
        Some(ObsAnchorTimes { first, last })
    }

    fn at(&self, a: ir::anchor::AnchoredTime) -> f64 {
        a.resolve(match a.anchor {
            ir::anchor::ObsAnchor::First => self.first,
            ir::anchor::ObsAnchor::Last => self.last,
        })
    }
}

enum RScalar {
    Const(f64),
    Param(usize),
    /// Index (into the quantities list) of a PRIOR scalar leaf.
    QRef(usize),
    UnOp { op: UnOp, arg: Box<RScalar> },
    BinOp { op: BinOp, left: Box<RScalar>, right: Box<RScalar> },
    Cond { pred: Box<RScalar>, then_: Box<RScalar>, else_: Box<RScalar> },
}

/// Pre-resolved evaluator over a model's `quantities`. Build once per compiled
/// model; `eval_draw` is pure and re-runs per draw.
pub struct QuantityEvaluator {
    programs: Vec<QProgram>,
    /// Leaf names, parallel to `programs` (for gate error messages).
    names: Vec<String>,
}

/// Canonical, order-independent key for a stratum cell.
fn stratum_key(stratum: &[ir::observation::StratumKey]) -> Vec<(String, String)> {
    let mut v: Vec<(String, String)> =
        stratum.iter().map(|s| (s.dim.clone(), s.level.clone())).collect();
    v.sort();
    v
}

impl QuantityEvaluator {
    pub fn new(quantities: &[Quantity], compiled: &CompiledModel) -> Result<Self, String> {
        let table_meta: Vec<(OobPolicy, usize)> = compiled
            .model
            .tables
            .iter()
            .zip(&compiled.table_values_cache)
            .map(|(t, cached)| (t.out_of_bounds.clone(), cached.len()))
            .collect();
        let ctx = ResolveCtx {
            comp_index: &compiled.comp_index,
            param_index: &compiled.param_index,
            time_func_index: &compiled.time_func_index,
            table_index: &compiled.table_index,
            global_to_int: &compiled.global_to_int,
            global_to_real: &compiled.global_to_real,
            table_meta: &table_meta,
            binding_index: &compiled.binding_index,
            per_eval_index: &compiled.per_eval_index,
        };
        let resolve = |e: &ir::expr::Expr| -> Result<ResolvedExpr, String> {
            resolve_expr(e, &ctx)
                .map_err(|err| format!("quantity state expression cannot resolve: {err:?}"))
        };

        // (name, stratum) → leaf index, for QRef resolution. Expansion guarantees
        // QRefs are backward (forward refs are an expander error), so a referenced
        // index is always < the referencing one.
        let mut leaf_index: HashMap<(String, Vec<(String, String)>), usize> = HashMap::new();
        for (i, q) in quantities.iter().enumerate() {
            leaf_index.insert((q.name.clone(), stratum_key(&q.stratum)), i);
        }

        let mut programs = Vec::with_capacity(quantities.len());
        for q in quantities {
            let prog = match &q.body {
                QuantityBody::Reduced { source, reduce } => {
                    let source = match source {
                        QuantitySource::State(expr) => QSource::State(resolve(expr)?),
                        QuantitySource::Observation { stream } => QSource::Observation(stream.clone()),
                    };
                    let reduce = match reduce {
                        None => None,
                        Some(r) => Some(resolve_reduce(r, &resolve)?),
                    };
                    QProgram::Reduced { source, reduce }
                }
                QuantityBody::Derived(se) => {
                    QProgram::Derived(resolve_scalar(se, &leaf_index, compiled, &q.name)?)
                }
            };
            programs.push(prog);
        }
        let names = quantities.iter().map(|q| q.name.clone()).collect();
        Ok(QuantityEvaluator { programs, names })
    }

    /// Whether any quantity reduces an `observations.<stream>` source — the cue
    /// for a caller to materialize the per-draw [`ObsSeriesSet`] before calling
    /// `eval_draw`. `false` ⇒ pass `None` and skip obs materialization entirely.
    pub fn references_observations(&self) -> bool {
        self.programs.iter().any(|p| {
            matches!(p, QProgram::Reduced { source: QSource::Observation(_), .. })
        })
    }

    /// Whether any quantity reads at an OBSERVATION ANCHOR — the cue for a
    /// caller to resolve the observed-data window before `eval_draw`. A
    /// data-free caller (forward simulate) must hard-error naming the
    /// quantities rather than pass `None` (proposal 2026-08-17).
    ///
    /// gh#616: matches EVERY anchor form, not just a bare `last_obs`. While
    /// this was keyed on one variant, a `first_obs` (or offset) quantity slipped
    /// past the gate and came back NaN — reported as *censored*, which reads as
    /// "the crossing never happened" rather than "we had no data".
    pub fn references_obs_anchor(&self) -> bool {
        self.programs.iter().any(|p| {
            matches!(p, QProgram::Reduced { reduce: Some(RReduce::ValueAt(RAnchor::Obs(_))), .. })
        })
    }

    /// Names of the quantities that read at an observation anchor (for the
    /// data-free caller's error message).
    pub fn obs_anchor_quantity_names(&self) -> Vec<&str> {
        self.programs
            .iter()
            .zip(&self.names)
            .filter_map(|(p, n)| match p {
                QProgram::Reduced { reduce: Some(RReduce::ValueAt(RAnchor::Obs(_))), .. } => {
                    Some(n.as_str())
                }
                _ => None,
            })
            .collect()
    }

    /// The distinct stream names reduced by `observations.<stream>` quantities
    /// (sorted, deduped). A caller materializes exactly these — a stream not in
    /// this list needs no `y_sim` for the run's quantities.
    pub fn obs_streams(&self) -> Vec<&str> {
        let mut v: Vec<&str> = self
            .programs
            .iter()
            .filter_map(|p| match p {
                QProgram::Reduced { source: QSource::Observation(s), .. } => Some(s.as_str()),
                _ => None,
            })
            .collect();
        v.sort_unstable();
        v.dedup();
        v
    }

    /// Fold every quantity over ONE draw: `params` is the draw's resolved param
    /// vector, `traj` the finished trajectory, `obs` the already-drawn `y_sim`
    /// series (v1.1; `None` for a state-only model). Pure; results are in the same
    /// order as the `quantities` list (so a `Derived` can read prior scalars).
    /// `obs_anchors`: the caller-resolved observed-data window (min and max
    /// observation time over the run's bound streams), REQUIRED iff
    /// [`references_obs_anchor`](Self::references_obs_anchor) — gate before
    /// calling from a data-free context. Passing `None` when a quantity needs
    /// it panics rather than censoring the value.
    pub fn eval_draw(
        &self,
        params: &[f64],
        traj: &Trajectory,
        compiled: &CompiledModel,
        obs: Option<&ObsSeriesSet>,
        obs_anchors: Option<ObsAnchorTimes>,
    ) -> Vec<QuantityResult> {
        let snap_times: Vec<f64> = traj.snapshots.iter().map(|s| s.t).collect();
        let mut results: Vec<QuantityResult> = Vec::with_capacity(self.programs.len());
        for prog in &self.programs {
            let r = match prog {
                QProgram::Reduced { source, reduce } => match source {
                    QSource::State(e) => {
                        let series = eval_series(e, traj, compiled, params);
                        match reduce {
                            None => QuantityResult::Series(series),
                            Some(red) => {
                                // State thresholds are evaluated at the snapshot times.
                                let thresh = |te: &ResolvedExpr| eval_series(te, traj, compiled, params);
                                QuantityResult::Scalar(fold_reduce(red, &series, &snap_times, &thresh, obs_anchors))
                            }
                        }
                    }
                    QSource::Observation(stream) => {
                        // The SAME y_sim the run published; an unmaterialized stream
                        // yields an empty series (a scalar then censors / is NaN).
                        let (otimes, series): (Vec<f64>, Vec<f64>) =
                            match obs.and_then(|o| o.streams.get(stream)) {
                                Some((t, v)) => (t.clone(), v.clone()),
                                None => (Vec::new(), Vec::new()),
                            };
                        match reduce {
                            None => QuantityResult::Series(series),
                            Some(red) => {
                                // Obs thresholds align to the obs times — evaluated
                                // against the snapshot at each obs time (a const/param
                                // threshold is state-independent; a state-dependent one
                                // reads the nearest snapshot).
                                let thresh = |te: &ResolvedExpr| -> Vec<f64> {
                                    otimes
                                        .iter()
                                        .map(|&t| eval_at(te, snap_at(traj, t), compiled, params, t))
                                        .collect()
                                };
                                QuantityResult::Scalar(fold_reduce(red, &series, &otimes, &thresh, obs_anchors))
                            }
                        }
                    }
                },
                QProgram::Derived(se) => QuantityResult::Scalar(eval_scalar(se, &results, params)),
            };
            results.push(r);
        }
        results
    }
}

fn resolve_reduce(
    r: &TemporalReduce,
    resolve: &impl Fn(&ir::expr::Expr) -> Result<ResolvedExpr, String>,
) -> Result<RReduce, String> {
    Ok(match r {
        TemporalReduce::Value(ValueReduce::Final) => RReduce::Final,
        TemporalReduce::Value(ValueReduce::Max) => RReduce::Max,
        TemporalReduce::Value(ValueReduce::Min) => RReduce::Min,
        TemporalReduce::Value(ValueReduce::Mean) => RReduce::Mean,
        TemporalReduce::Value(ValueReduce::CountAbove(t)) => RReduce::CountAbove(resolve(t)?),
        TemporalReduce::Value(ValueReduce::CountBelow(t)) => RReduce::CountBelow(resolve(t)?),
        TemporalReduce::Value(ValueReduce::ValueAt(ir::quantity::TimeAnchor::Time(t))) => {
            RReduce::ValueAt(RAnchor::Expr(resolve(t)?))
        }
        TemporalReduce::Value(ValueReduce::ValueAt(ir::quantity::TimeAnchor::Obs(a))) => {
            RReduce::ValueAt(RAnchor::Obs(*a))
        }
        TemporalReduce::Time(TimeReduce::TimeOfMax) => RReduce::TimeOfMax,
        TemporalReduce::Time(TimeReduce::TimeOfMin) => RReduce::TimeOfMin,
        TemporalReduce::Time(TimeReduce::FirstAbove(t)) => RReduce::FirstAbove(resolve(t)?),
        TemporalReduce::Time(TimeReduce::FirstBelow(t)) => RReduce::FirstBelow(resolve(t)?),
        TemporalReduce::Time(TimeReduce::LastAbove(t)) => RReduce::LastAbove(resolve(t)?),
        TemporalReduce::Time(TimeReduce::LastBelow(t)) => RReduce::LastBelow(resolve(t)?),
        TemporalReduce::Integral => RReduce::Integral,
    })
}

fn resolve_scalar(
    se: &ScalarExpr,
    leaf_index: &HashMap<(String, Vec<(String, String)>), usize>,
    compiled: &CompiledModel,
    qname: &str,
) -> Result<RScalar, String> {
    Ok(match se {
        ScalarExpr::Const(v) => RScalar::Const(*v),
        ScalarExpr::Param(name) => {
            let idx = *compiled.param_index.get(name.as_str()).ok_or_else(|| {
                format!("quantity '{qname}': reduction arithmetic references unknown parameter '{name}'")
            })?;
            RScalar::Param(idx)
        }
        ScalarExpr::QRef(q) => {
            let key = (q.name.clone(), stratum_key(&q.stratum));
            let idx = *leaf_index.get(&key).ok_or_else(|| {
                format!("quantity '{qname}': reduction arithmetic references unknown quantity '{}'", q.name)
            })?;
            RScalar::QRef(idx)
        }
        ScalarExpr::UnOp { op, arg } => RScalar::UnOp {
            op: op.clone(),
            arg: Box::new(resolve_scalar(arg, leaf_index, compiled, qname)?),
        },
        ScalarExpr::BinOp { op, left, right } => RScalar::BinOp {
            op: op.clone(),
            left: Box::new(resolve_scalar(left, leaf_index, compiled, qname)?),
            right: Box::new(resolve_scalar(right, leaf_index, compiled, qname)?),
        },
        ScalarExpr::Cond { pred, then, else_ } => RScalar::Cond {
            pred: Box::new(resolve_scalar(pred, leaf_index, compiled, qname)?),
            then_: Box::new(resolve_scalar(then, leaf_index, compiled, qname)?),
            else_: Box::new(resolve_scalar(else_, leaf_index, compiled, qname)?),
        },
    })
}

/// Evaluate a resolved state expr against one snapshot's state. No integrator
/// step (`dt = 0`), no projection/aux (those are likelihood-only and rejected in
/// a quantity by `ir::validate`).
fn eval_at(
    expr: &ResolvedExpr,
    snap: &crate::state::Snapshot,
    compiled: &CompiledModel,
    params: &[f64],
    t: f64,
) -> f64 {
    let ctx = EvalCtx {
        model: compiled,
        int_s: &snap.int_state,
        real_s: &snap.real_state,
        params,
        t,
        dt: 0.0,
        projected: None,
        aux: None,
        int_float_override: None,
        per_eval: None,
    };
    eval_resolved(expr, &ctx)
}

/// Evaluate a resolved state expr at every snapshot → a per-snapshot series.
fn eval_series(
    expr: &ResolvedExpr,
    traj: &Trajectory,
    compiled: &CompiledModel,
    params: &[f64],
) -> Vec<f64> {
    traj.snapshots
        .iter()
        .map(|snap| eval_at(expr, snap, compiled, params, snap.t))
        .collect()
}

/// The snapshot nearest a given time — for evaluating an `Observation`-source
/// threshold at the obs times (which may not coincide with the output grid).
fn snap_at(traj: &Trajectory, t: f64) -> &crate::state::Snapshot {
    traj.snapshots
        .iter()
        .min_by(|a, b| (a.t - t).abs().total_cmp(&(b.t - t).abs()))
        .expect("trajectory has at least one snapshot")
}

/// `thresh(expr)` produces the threshold series aligned to `series`/`times` (over
/// the snapshot times for a state source, the obs times for an observation source).
fn fold_reduce(
    r: &RReduce,
    series: &[f64],
    times: &[f64],
    thresh: &impl Fn(&ResolvedExpr) -> Vec<f64>,
    obs_anchors: Option<ObsAnchorTimes>,
) -> QuantityDrawValue {
    use QuantityDrawValue::*;
    match r {
        RReduce::Final => Value(series.last().copied().unwrap_or(f64::NAN)),
        RReduce::ValueAt(anchor) => {
            let t = match anchor {
                // Params are fixed within a draw, so evaluating the (constant
                // or param) time expression at the first snapshot is the
                // expression's value for the draw.
                RAnchor::Expr(e) => thresh(e).first().copied().unwrap_or(f64::NAN),
                // gh#616: an UNCONDITIONAL failure, not a `debug_assert!`.
                // Compiled out of release, the old assertion left the `None`
                // arm returning NaN, which `value_at_locf` reports as
                // *censored* — indistinguishable from "the anchor fell outside
                // the trajectory window", so a missing-data bug read as a
                // legitimate result. Callers gate on `references_obs_anchor`;
                // reaching here means a gate is missing, which is a defect to
                // surface, not to average over.
                RAnchor::Obs(a) => match obs_anchors {
                    Some(t) => t.at(*a),
                    None => panic!(
                        "value_at({a}) reached quantity evaluation with no observation \
                         times; every caller must gate on \
                         QuantityEvaluator::references_obs_anchor() and refuse, so this \
                         is a missing gate at the call site"
                    ),
                },
            };
            value_at_locf(series, times, t)
        }
        RReduce::Max => Value(reduce_finite(series, f64::NEG_INFINITY, |a, b| a.max(b))),
        RReduce::Min => Value(reduce_finite(series, f64::INFINITY, |a, b| a.min(b))),
        RReduce::Mean => {
            if series.is_empty() {
                Value(f64::NAN)
            } else {
                Value(series.iter().sum::<f64>() / series.len() as f64)
            }
        }
        RReduce::CountAbove(t) => Value(count_cross(series, &thresh(t), |s, th| s > th) as f64),
        RReduce::CountBelow(t) => Value(count_cross(series, &thresh(t), |s, th| s < th) as f64),
        RReduce::TimeOfMax => arg_to_time(argmax(series), times),
        RReduce::TimeOfMin => arg_to_time(argmin(series), times),
        RReduce::FirstAbove(t) => cross_time(series, &thresh(t), times, |s, th| s > th, true),
        RReduce::FirstBelow(t) => cross_time(series, &thresh(t), times, |s, th| s < th, true),
        RReduce::LastAbove(t) => cross_time(series, &thresh(t), times, |s, th| s > th, false),
        RReduce::LastBelow(t) => cross_time(series, &thresh(t), times, |s, th| s < th, false),
        RReduce::Integral => Value(trapezoid(series, times)),
    }
}

/// Evaluate reduction arithmetic over already-computed prior leaf values.
/// `Censored` propagates: a derived value that combines a censored scalar is
/// itself censored.
fn eval_scalar(se: &RScalar, results: &[QuantityResult], params: &[f64]) -> QuantityDrawValue {
    use QuantityDrawValue::*;
    match se {
        RScalar::Const(v) => Value(*v),
        RScalar::Param(idx) => Value(params.get(*idx).copied().unwrap_or(f64::NAN)),
        RScalar::QRef(idx) => match results.get(*idx) {
            Some(QuantityResult::Scalar(v)) => *v,
            // Series QRef is an expander error; defensively non-finite.
            _ => Value(f64::NAN),
        },
        RScalar::UnOp { op, arg } => match eval_scalar(arg, results, params) {
            Censored => Censored,
            Value(x) => Value(apply_un(op, x)),
        },
        RScalar::BinOp { op, left, right } => {
            match (eval_scalar(left, results, params), eval_scalar(right, results, params)) {
                (Censored, _) | (_, Censored) => Censored,
                (Value(a), Value(b)) => Value(apply_bin(op, a, b)),
            }
        }
        RScalar::Cond { pred, then_, else_ } => match eval_scalar(pred, results, params) {
            Censored => Censored,
            Value(p) => {
                if p != 0.0 {
                    eval_scalar(then_, results, params)
                } else {
                    eval_scalar(else_, results, params)
                }
            }
        },
    }
}

// ── Fold helpers ────────────────────────────────────────────────────────────────

/// The series value at the last time `<= t` (LOCF — "the state as of t").
/// Censored outside the window, never clamped: clamping would silently answer
/// a different question, which is the misreading `value_at` exists to prevent
/// (proposal 2026-08-17). `times` is ascending (the output grid or a stream's
/// observation times) and same-length as `series`.
fn value_at_locf(series: &[f64], times: &[f64], t: f64) -> QuantityDrawValue {
    use QuantityDrawValue::*;
    debug_assert_eq!(series.len(), times.len());
    if series.is_empty() || !t.is_finite() {
        return Censored;
    }
    if t < times[0] || t > *times.last().expect("nonempty") {
        return Censored;
    }
    // Number of times <= t; >= 1 here since t >= times[0].
    let idx = times.partition_point(|&x| x <= t);
    Value(series[idx - 1])
}

fn reduce_finite(series: &[f64], init: f64, f: impl Fn(f64, f64) -> f64) -> f64 {
    let mut acc = init;
    let mut any = false;
    for &v in series {
        if v.is_finite() {
            acc = f(acc, v);
            any = true;
        }
    }
    if any { acc } else { f64::NAN }
}

fn argmax(series: &[f64]) -> Option<usize> {
    arg_extreme(series, |v, best| v > best)
}
fn argmin(series: &[f64]) -> Option<usize> {
    arg_extreme(series, |v, best| v < best)
}
/// First index of the extreme finite value (`better(v, best)` strict, so ties
/// keep the first).
fn arg_extreme(series: &[f64], better: impl Fn(f64, f64) -> bool) -> Option<usize> {
    let mut best: Option<(usize, f64)> = None;
    for (i, &v) in series.iter().enumerate() {
        if !v.is_finite() {
            continue;
        }
        match best {
            None => best = Some((i, v)),
            Some((_, bv)) => {
                if better(v, bv) {
                    best = Some((i, v));
                }
            }
        }
    }
    best.map(|(i, _)| i)
}

fn arg_to_time(idx: Option<usize>, times: &[f64]) -> QuantityDrawValue {
    match idx {
        Some(i) => QuantityDrawValue::Value(times[i]),
        None => QuantityDrawValue::Value(f64::NAN),
    }
}

fn count_cross(series: &[f64], thresh: &[f64], pred: impl Fn(f64, f64) -> bool) -> usize {
    series
        .iter()
        .zip(thresh)
        .filter(|(&s, &th)| s.is_finite() && th.is_finite() && pred(s, th))
        .count()
}

/// First (or last) time the series crosses the threshold; `Censored` if it never
/// does within the window.
fn cross_time(
    series: &[f64],
    thresh: &[f64],
    times: &[f64],
    pred: impl Fn(f64, f64) -> bool,
    first: bool,
) -> QuantityDrawValue {
    let mut hit: Option<usize> = None;
    for (i, (&s, &th)) in series.iter().zip(thresh).enumerate() {
        if s.is_finite() && th.is_finite() && pred(s, th) {
            hit = Some(i);
            if first {
                break;
            }
        }
    }
    match hit {
        Some(i) => QuantityDrawValue::Value(times[i]),
        None => QuantityDrawValue::Censored,
    }
}

/// Trapezoidal ∫ series dt over the snapshot times.
fn trapezoid(series: &[f64], times: &[f64]) -> f64 {
    let n = series.len().min(times.len());
    if n < 2 {
        return 0.0;
    }
    let mut acc = 0.0;
    for i in 0..n - 1 {
        acc += 0.5 * (series[i] + series[i + 1]) * (times[i + 1] - times[i]);
    }
    acc
}

fn apply_bin(op: &BinOp, a: f64, b: f64) -> f64 {
    match op {
        BinOp::Add => a + b,
        BinOp::Sub => a - b,
        BinOp::Mul => a * b,
        BinOp::Div => a / b,
        BinOp::Pow => a.powf(b),
        BinOp::Mod => a.rem_euclid(b),
        BinOp::Min => a.min(b),
        BinOp::Max => a.max(b),
        BinOp::Eq => (a == b) as i32 as f64,
        BinOp::Neq => (a != b) as i32 as f64,
        BinOp::Lt => (a < b) as i32 as f64,
        BinOp::Gt => (a > b) as i32 as f64,
        BinOp::Le => (a <= b) as i32 as f64,
        BinOp::Ge => (a >= b) as i32 as f64,
    }
}

fn apply_un(op: &UnOp, x: f64) -> f64 {
    match op {
        UnOp::Neg => -x,
        UnOp::Exp => x.exp(),
        UnOp::Log => x.ln(),
        UnOp::Sqrt => x.sqrt(),
        UnOp::Abs => x.abs(),
        UnOp::Floor => x.floor(),
        UnOp::Ceil => x.ceil(),
        UnOp::Sin => x.sin(),
        UnOp::Cos => x.cos(),
        UnOp::Tanh => x.tanh(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use QuantityDrawValue::*;

    // ── Observation anchors (gh#616) ─────────────────────────────────────────

    /// The pair folds the run's observation times, and the offset is applied on
    /// top of whichever end the anchor names. An offset applied to the WRONG end
    /// would still produce a plausible in-window time, so pin both.
    #[test]
    fn obs_anchor_times_fold_and_resolve() {
        use ir::anchor::{AnchoredTime, ObsAnchor};
        let t = ObsAnchorTimes::of_times([21.0, 0.0, 14.0, 7.0]).expect("non-empty");
        assert_eq!(t, ObsAnchorTimes { first: 0.0, last: 21.0 });
        assert_eq!(t.at(AnchoredTime::bare(ObsAnchor::Last)), 21.0);
        assert_eq!(t.at(AnchoredTime::bare(ObsAnchor::First)), 0.0);
        assert_eq!(t.at(AnchoredTime { anchor: ObsAnchor::Last, offset: 28.0 }), 49.0);
        assert_eq!(t.at(AnchoredTime { anchor: ObsAnchor::First, offset: -7.0 }), -7.0);
    }

    /// No observation times at all → `None`, so a caller cannot mistake an empty
    /// stream set for an anchor at t = 0.
    #[test]
    fn obs_anchor_times_of_empty_is_none() {
        assert_eq!(ObsAnchorTimes::of_times([]), None);
    }

    /// The gate matches EVERY anchor form. Keyed on one variant (as it was
    /// before gh#616), a `first_obs` or offset quantity would slip past a
    /// data-free caller's refusal and come back censored.
    #[test]
    fn the_gate_matches_every_anchor_form() {
        use ir::anchor::{AnchoredTime, ObsAnchor};
        let anchored = |a: AnchoredTime| QuantityEvaluator {
            programs: vec![QProgram::Reduced {
                source: QSource::Observation("s".into()),
                reduce: Some(RReduce::ValueAt(RAnchor::Obs(a))),
            }],
            names: vec!["q".into()],
        };
        for a in [
            AnchoredTime::bare(ObsAnchor::Last),
            AnchoredTime::bare(ObsAnchor::First),
            AnchoredTime { anchor: ObsAnchor::Last, offset: 28.0 },
            AnchoredTime { anchor: ObsAnchor::First, offset: -7.0 },
        ] {
            let e = anchored(a);
            assert!(e.references_obs_anchor(), "gate must fire for {a}");
            assert_eq!(e.obs_anchor_quantity_names(), vec!["q"], "named for {a}");
        }
        // Negative control: a constant-time `value_at` needs no data and must
        // NOT be gated, or every literal-time quantity would start refusing.
        let literal = QuantityEvaluator {
            programs: vec![QProgram::Reduced {
                source: QSource::Observation("s".into()),
                reduce: Some(RReduce::ValueAt(RAnchor::Expr(ResolvedExpr::Const(20.0)))),
            }],
            names: vec!["q".into()],
        };
        assert!(!literal.references_obs_anchor());
        assert!(literal.obs_anchor_quantity_names().is_empty());
    }

    #[test]
    fn fold_reduce_resolves_an_anchor_through_the_run_window() {
        use ir::anchor::{AnchoredTime, ObsAnchor};
        let times = [0.0, 7.0, 14.0, 21.0];
        let series = [1.0, 5.0, 9.0, 13.0];
        let thresh = |_: &ResolvedExpr| Vec::new();
        let anchors = ObsAnchorTimes { first: 0.0, last: 14.0 };
        // last_obs → the t=14 value.
        let r = RReduce::ValueAt(RAnchor::Obs(AnchoredTime::bare(ObsAnchor::Last)));
        assert_eq!(fold_reduce(&r, &series, &times, &thresh, Some(anchors)), Value(9.0));
        // last_obs - 1 'weeks → t=7 (LOCF), not the t=14 value.
        let r = RReduce::ValueAt(RAnchor::Obs(AnchoredTime {
            anchor: ObsAnchor::Last,
            offset: -7.0,
        }));
        assert_eq!(fold_reduce(&r, &series, &times, &thresh, Some(anchors)), Value(5.0));
        // first_obs + 3 'days → t=3 → LOCF back to t=0.
        let r = RReduce::ValueAt(RAnchor::Obs(AnchoredTime {
            anchor: ObsAnchor::First,
            offset: 3.0,
        }));
        assert_eq!(fold_reduce(&r, &series, &times, &thresh, Some(anchors)), Value(1.0));
        // Past the trajectory window → censored, never clamped to the horizon.
        let r = RReduce::ValueAt(RAnchor::Obs(AnchoredTime {
            anchor: ObsAnchor::Last,
            offset: 28.0,
        }));
        assert_eq!(fold_reduce(&r, &series, &times, &thresh, Some(anchors)), Censored);
    }

    /// Reaching evaluation with no observation times is a MISSING GATE, and must
    /// fail loudly. Before gh#616 this was a `debug_assert!` — compiled out of
    /// release, where the arm returned NaN and the value was reported as
    /// *censored*, i.e. as a legitimate "outside the window" result.
    #[test]
    #[should_panic(expected = "no observation times")]
    fn a_missing_gate_panics_instead_of_censoring() {
        use ir::anchor::{AnchoredTime, ObsAnchor};
        let r = RReduce::ValueAt(RAnchor::Obs(AnchoredTime::bare(ObsAnchor::First)));
        let thresh = |_: &ResolvedExpr| Vec::new();
        let _ = fold_reduce(&r, &[1.0, 2.0], &[0.0, 1.0], &thresh, None);
    }

    #[test]
    fn value_at_locf_reads_last_time_at_or_before_anchor() {
        let times = [0.0, 7.0, 14.0, 21.0];
        let series = [1.0, 5.0, 9.0, 13.0];
        // Exactly on a snapshot → that snapshot.
        assert_eq!(value_at_locf(&series, &times, 14.0), Value(9.0));
        // Between snapshots → the LAST one at or before (LOCF), never
        // interpolated: 10.0 sits between t=7 and t=14 → the t=7 value.
        assert_eq!(value_at_locf(&series, &times, 10.0), Value(5.0));
        // At the window edges → the edge values, not censored.
        assert_eq!(value_at_locf(&series, &times, 0.0), Value(1.0));
        assert_eq!(value_at_locf(&series, &times, 21.0), Value(13.0));
    }

    #[test]
    fn value_at_locf_censors_outside_the_window_never_clamps() {
        let times = [0.0, 7.0, 14.0];
        let series = [1.0, 5.0, 9.0];
        // Past the end: censored — clamping to final(9.0) would silently
        // report the projection at the horizon, the misreading value_at
        // exists to prevent (proposal 2026-08-17).
        assert_eq!(value_at_locf(&series, &times, 14.0001), Censored);
        // Before the start, non-finite, empty: censored.
        assert_eq!(value_at_locf(&series, &times, -0.5), Censored);
        assert_eq!(value_at_locf(&series, &times, f64::NAN), Censored);
        assert_eq!(value_at_locf(&[], &[], 1.0), Censored);
    }

    #[test]
    fn argmax_argmin_first_on_ties() {
        // ties keep the FIRST index (strict `>`/`<`).
        let s = [1.0, 3.0, 3.0, 2.0, 3.0];
        assert_eq!(argmax(&s), Some(1));
        assert_eq!(argmin(&[2.0, 1.0, 1.0]), Some(1));
        // NaN is skipped, not chosen.
        assert_eq!(argmax(&[f64::NAN, 5.0, f64::NAN]), Some(1));
        assert_eq!(argmax(&[]), None);
    }

    #[test]
    fn cross_time_first_last_and_censoring() {
        let series = [0.0, 0.0, 5.0, 1.0, 9.0];
        let times = [0.0, 1.0, 2.0, 3.0, 4.0];
        let thr = [0.0, 0.0, 0.0, 0.0, 0.0]; // strictly > 0
        // first above 0 → t=2; last above 0 → t=4.
        assert_eq!(cross_time(&series, &thr, &times, |s, t| s > t, true), Value(2.0));
        assert_eq!(cross_time(&series, &thr, &times, |s, t| s > t, false), Value(4.0));
        // never crosses (threshold 100) → Censored, NOT a fabricated time.
        let high = [100.0; 5];
        assert_eq!(cross_time(&series, &high, &times, |s, t| s > t, true), Censored);
    }

    #[test]
    fn count_and_trapezoid_and_finite() {
        let series = [1.0, 2.0, 3.0];
        let thr = [1.5; 3];
        assert_eq!(count_cross(&series, &thr, |s, t| s > t), 2); // 2.0, 3.0
        // ∫ over t=[0,1,2] of [0,2,2] = trapezoid = 0.5*(0+2)*1 + 0.5*(2+2)*1 = 3.
        assert_eq!(trapezoid(&[0.0, 2.0, 2.0], &[0.0, 1.0, 2.0]), 3.0);
        assert_eq!(trapezoid(&[5.0], &[0.0]), 0.0); // <2 points
        // max/min skip NaN; all-NaN → NaN.
        assert_eq!(reduce_finite(&[1.0, f64::NAN, 4.0], f64::NEG_INFINITY, f64::max), 4.0);
        assert!(reduce_finite(&[f64::NAN], f64::NEG_INFINITY, f64::max).is_nan());
    }

    #[test]
    fn derived_arithmetic_propagates_censoring() {
        // results[0] = 10, results[1] = 3, results[2] = Censored.
        let results = vec![
            QuantityResult::Scalar(Value(10.0)),
            QuantityResult::Scalar(Value(3.0)),
            QuantityResult::Scalar(Censored),
        ];
        let params = [2.0];
        // QRef(0) - QRef(1) = 7.
        let sub = RScalar::BinOp {
            op: BinOp::Sub,
            left: Box::new(RScalar::QRef(0)),
            right: Box::new(RScalar::QRef(1)),
        };
        assert_eq!(eval_scalar(&sub, &results, &params), Value(7.0));
        // QRef(0) - QRef(2) = Censored (an endpoint never fired).
        let sub_c = RScalar::BinOp {
            op: BinOp::Sub,
            left: Box::new(RScalar::QRef(0)),
            right: Box::new(RScalar::QRef(2)),
        };
        assert_eq!(eval_scalar(&sub_c, &results, &params), Censored);
        // abs(QRef(1) - Param(0)) = |3 - 2| = 1.
        let abs = RScalar::UnOp {
            op: UnOp::Abs,
            arg: Box::new(RScalar::BinOp {
                op: BinOp::Sub,
                left: Box::new(RScalar::QRef(1)),
                right: Box::new(RScalar::Param(0)),
            }),
        };
        assert_eq!(eval_scalar(&abs, &results, &params), Value(1.0));
    }
}
