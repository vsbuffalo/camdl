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

/// The run's resolved observation window. Defined in `ir::anchor` beside
/// `AnchoredTime` (its unresolved counterpart) because the CLI's model-level
/// resolver and this evaluator must agree on it exactly.
pub use ir::anchor::ObsAnchorTimes;

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

/// Which trajectory a quantity's value must be read off (gh#722).
///
/// A `value_at` whose time falls **inside the observed record** is a
/// retrospective estimand: there are observations covering it, so the object
/// that answers it is the conditioned smoothing path `p(x | y, θ)`. Pushing θ
/// through a fresh unconditioned replay and reading the state there discards
/// every one of those observations — with a weakly identified initial
/// condition the replays span orders of magnitude and their median has no
/// relationship to the realised epidemic (`outbreak_size` came back BELOW the
/// confirmed-case count it is arithmetically bounded by).
///
/// Everything else keeps the replay: a reduction with no anchor (`final`,
/// `max`, `time_of_max`, …) is a property of a whole simulated path, and an
/// anchor PAST the last observation is a projection, which is what the replay
/// is for.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QuantityPath {
    /// Read on the conditioned smoothing path.
    Smoothed,
    /// Read on the free-forward replay — no anchor, or an anchor past the end
    /// of the observed record.
    Replay,
    /// Anchored inside the observed record, but its series is a SAMPLED
    /// observation (`observations.<stream>`), which no saved path carries: the
    /// smoothing file holds the conditioned projection (`inc_<stream>`, a mean),
    /// not a draw from it. Read on the replay — the caller must SAY SO, because
    /// this is the same defect, unfixed for this one source kind.
    ReplayUnconditioned,
}

impl QuantityPath {
    /// The manifest / diagnostic spelling.
    pub fn as_str(self) -> &'static str {
        match self {
            QuantityPath::Smoothed => "smoothed",
            QuantityPath::Replay => "replay",
            QuantityPath::ReplayUnconditioned => "replay_unconditioned",
        }
    }
}

/// What a caller has to offer a draw's [`QuantityPath::Smoothed`] quantities.
///
/// Three states, not `Option<&Trajectory>`, because "the caller is not doing a
/// conditioned read at all" and "the caller is, but THIS draw has no saved
/// path" must not collapse into one value: the first keeps today's replay
/// answer, the second must censor rather than quietly substitute the replay —
/// which is the whole of gh#722.
#[derive(Debug, Clone, Copy)]
pub enum ConditionedRead<'a> {
    /// No conditioned read is in play: every quantity reads the replay
    /// trajectory. `simulate` (which has no fit behind it) and the contrast
    /// arms (whose forked replay IS the object being differenced) pass this,
    /// and their output is unchanged.
    Off,
    /// This draw's saved smoothing path.
    Saved(&'a Trajectory),
    /// A conditioned read is in play, but this draw is outside the forkable
    /// subset — no path was saved for it. Its in-window `value_at` values are
    /// CENSORED, never taken off the replay.
    NotSaved,
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

    /// Per quantity, in `quantities` order: which trajectory its value must be
    /// read off, given the run's observed window (gh#722).
    ///
    /// Pure, and draw-independent by construction — the routing is a property
    /// of the QUANTITY, not of a draw, so a band cannot be a mixture of two
    /// objects. `eval_draw` folds through this same classifier, so what a
    /// caller reports and what it computes cannot disagree.
    ///
    /// `None` observation window ⇒ nothing is anchorable, so everything reads
    /// the replay (the data-free caller already refused any `Obs` anchor
    /// upstream via [`references_obs_anchor`](Self::references_obs_anchor)).
    pub fn eval_paths(&self, obs_anchors: Option<ObsAnchorTimes>) -> Vec<QuantityPath> {
        self.programs.iter().map(|p| program_path(p, obs_anchors)).collect()
    }

    /// Names of the quantities that route to `path` — for the caller's
    /// stderr note and manifest. Same classifier as
    /// [`eval_paths`](Self::eval_paths).
    pub fn quantity_names_on(
        &self,
        path: QuantityPath,
        obs_anchors: Option<ObsAnchorTimes>,
    ) -> Vec<&str> {
        self.programs
            .iter()
            .zip(&self.names)
            .filter(|(p, _)| program_path(p, obs_anchors) == path)
            .map(|(_, n)| n.as_str())
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
    /// vector, `traj` the finished FREE-FORWARD trajectory, `obs` the
    /// already-drawn `y_sim` series (v1.1; `None` for a state-only model).
    /// Pure; results are in the same order as the `quantities` list (so a
    /// `Derived` can read prior scalars).
    ///
    /// `conditioned` is what the caller can offer this draw's in-window
    /// `value_at` quantities ([`QuantityPath::Smoothed`], per
    /// [`eval_paths`](Self::eval_paths)): [`ConditionedRead::Off`] leaves every
    /// quantity on `traj` (unchanged behaviour), [`ConditionedRead::Saved`]
    /// reads them off the smoothing path instead, and
    /// [`ConditionedRead::NotSaved`] censors them rather than falling back to
    /// `traj` — a fallback is exactly the gh#722 defect.
    ///
    /// A `Derived` leaf reads the SPLICED results, so reduction arithmetic over
    /// a smoothed leaf carries the smoothed value (and its censoring)
    /// automatically.
    ///
    /// `obs_anchors`: the caller-resolved observed-data window (min and max
    /// observation time over the run's bound streams), REQUIRED iff
    /// [`references_obs_anchor`](Self::references_obs_anchor) — gate before
    /// calling from a data-free context. Passing `None` when a quantity needs
    /// it panics rather than censoring the value.
    pub fn eval_draw(
        &self,
        params: &[f64],
        traj: &Trajectory,
        conditioned: ConditionedRead<'_>,
        compiled: &CompiledModel,
        obs: Option<&ObsSeriesSet>,
        obs_anchors: Option<ObsAnchorTimes>,
    ) -> Vec<QuantityResult> {
        let snap_times: Vec<f64> = traj.snapshots.iter().map(|s| s.t).collect();
        // The smoothing path carries its own snapshot grid (PGAS writes at
        // substep resolution, finer than the output cadence), so a smoothed
        // read has to fold over THAT axis — reusing `snap_times` would pair a
        // conditioned series with the replay's times.
        let smoothed: Option<(&Trajectory, Vec<f64>)> = match conditioned {
            ConditionedRead::Saved(s) => {
                Some((s, s.snapshots.iter().map(|snap| snap.t).collect()))
            }
            ConditionedRead::Off | ConditionedRead::NotSaved => None,
        };
        let route = !matches!(conditioned, ConditionedRead::Off);
        let mut results: Vec<QuantityResult> = Vec::with_capacity(self.programs.len());
        for prog in &self.programs {
            // Which object THIS quantity is read off. `Off` short-circuits the
            // classifier so a caller with no conditioned path in play is
            // byte-identical to the pre-gh#722 evaluator.
            let on_smoothed =
                route && program_path(prog, obs_anchors) == QuantityPath::Smoothed;
            if on_smoothed {
                let QProgram::Reduced { source: QSource::State(e), reduce: Some(red) } = prog
                else {
                    // `program_path` returns `Smoothed` for exactly this shape.
                    unreachable!("only a state-source value_at routes to the smoothing path")
                };
                let r = match &smoothed {
                    Some((s, stimes)) => {
                        let series = eval_series(e, s, compiled, params);
                        let thresh = |te: &ResolvedExpr| eval_series(te, s, compiled, params);
                        QuantityResult::Scalar(fold_reduce(
                            red, &series, stimes, &thresh, obs_anchors,
                        ))
                    }
                    // No saved path for this draw: the band loses the draw, it
                    // does not gain a free-forward substitute.
                    None => QuantityResult::Scalar(QuantityDrawValue::Censored),
                };
                results.push(r);
                continue;
            }
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

/// Which trajectory one program's value must be read off (gh#722). The single
/// classifier — [`QuantityEvaluator::eval_paths`] reports it and
/// [`QuantityEvaluator::eval_draw`] acts on it, so the manifest cannot describe
/// one object while the number came from another.
///
/// `Smoothed` requires all three of:
///
///  - a `value_at` reduction — every other reduction (`final`, `max`, `mean`,
///    `time_of_max`, `integral`, a threshold crossing) is a property of a whole
///    path, not a reading at an instant, so the replay is its object;
///  - a time that resolves at or before the LAST observation, using the run's
///    own window. An anchor past it (`last_obs + 8 'weeks`) is a projection;
///  - a **state** series. `observations.<stream>` reduces a sampled `y_sim`,
///    and no saved path carries one — hence `ReplayUnconditioned`, which the
///    caller reports rather than passing off as a conditioned answer.
///
/// A `value_at` at a LITERAL time is classified too, from its folded constant:
/// `value_at(N0 - S, date("2026-08-10"))` inside the record has the same defect
/// as the `last_obs` spelling. A time expression that is not a constant (a
/// parameter carrying dim T) cannot be classified without a draw, and routing
/// per draw would make the band a mixture of two objects — so it stays on the
/// replay. The compiler documents this argument as "a constant time
/// expression" (`ocaml/lib/ir/ir.ml`, `time_anchor`).
fn program_path(prog: &QProgram, obs_anchors: Option<ObsAnchorTimes>) -> QuantityPath {
    let QProgram::Reduced { source, reduce: Some(RReduce::ValueAt(anchor)) } = prog else {
        return QuantityPath::Replay;
    };
    let Some(w) = obs_anchors else {
        return QuantityPath::Replay;
    };
    let t = match anchor {
        RAnchor::Obs(a) => w.at(*a),
        RAnchor::Expr(ResolvedExpr::Const(t)) => *t,
        RAnchor::Expr(_) => return QuantityPath::Replay,
    };
    // A NaN anchor time compares false here and falls to the replay, which is
    // the safe side: an unresolvable time must not claim to be conditioned.
    let inside_the_record = t <= w.last;
    if !inside_the_record {
        return QuantityPath::Replay;
    }
    match source {
        QSource::State(_) => QuantityPath::Smoothed,
        QSource::Observation(_) => QuantityPath::ReplayUnconditioned,
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

    // ── gh#722: routing an in-window `value_at` onto the smoothing path ──────

    /// A minimal compiled model: two integer compartments, one parameter.
    /// Enough for `eval_series` to read `IntPop(0)` off a snapshot.
    fn one_compartment_model() -> CompiledModel {
        use ir::{
            expr::Expr,
            model::{
                Compartment, CompartmentKind, InitialConditions, OutputConfig, OutputSchedule,
                SimulationConfig,
            },
            parameter::Parameter,
            transition::{DrawMethod, StoichiometryEntry, Transition},
            Model,
        };
        let m = Model {
            ic_grad: Default::default(),
            name: "q722".into(),
            version: "0.1".into(),
            time_unit: "days".into(),
            description: None,
            origin: None,
            origin_rata_die: None,
            compartments: vec![
                Compartment { name: "S".into(), kind: CompartmentKind::Integer },
                Compartment { name: "I".into(), kind: CompartmentKind::Integer },
            ],
            transitions: vec![Transition {
                rate_state_grad: Default::default(),
                name: "infection".into(),
                stoichiometry: vec![
                    StoichiometryEntry("S".into(), -1),
                    StoichiometryEntry("I".into(), 1),
                ],
                rate: Expr::const_(0.0),
                metadata: None,
                draw_method: DrawMethod::Poisson,
                rate_grad: Default::default(),
                lineage: None,
            }],
            ode_equations: vec![],
            time_functions: vec![],
            tables: vec![],
            interventions: vec![],
            observations: vec![],
            bindings: vec![],
            per_eval_bindings: vec![],
            parameters: vec![Parameter {
                name: "p".into(),
                value: ir::parameter::ParamValue::Fixed { value: 1.0 },
                param_kind: None,
                param_dim: None,
            }],
            initial_conditions: InitialConditions::Explicit({
                let mut h = HashMap::new();
                h.insert("S".into(), 1000.0);
                h.insert("I".into(), 0.0);
                h
            }),
            output: OutputConfig {
                times: OutputSchedule::AtTimes(vec![0.0, 1.0]),
                format: "tsv".into(),
                trajectory: true,
                observations: false,
            },
            simulation: SimulationConfig {
                t_start: 0.0,
                t_end: 30.0,
                time_semantics: "discrete".into(),
                dt: Some(1.0),
                rng_seed: Some(1),
                integrator: Default::default(),
                t_end_anchor: None,
            },
            presets: vec![],
            model_structure: None,
            balance: None,
            identity_tracked_compartments: vec![],
            quantities: vec![],
            contrasts: vec![],
        };
        CompiledModel::new(m).unwrap()
    }

    /// A path whose `S` takes the given value at each of `times`.
    fn path_of(times: &[f64], s: &[i64]) -> Trajectory {
        let mut t = Trajectory::new();
        for (&time, &sv) in times.iter().zip(s) {
            t.push(crate::state::Snapshot {
                t: time,
                int_state: crate::state::IntState::from_vec(vec![sv, 0]),
                real_state: crate::state::RealState::from_vec(vec![]),
                flows: crate::state::Flows::Int(vec![0]),
            });
        }
        t
    }

    /// `value_at(S, last_obs)` plus the three negative controls. Built directly
    /// (as the gh#616 tests do) so the classification is exercised without a
    /// compiler round-trip.
    fn evaluator_722() -> QuantityEvaluator {
        use ir::anchor::{AnchoredTime, ObsAnchor};
        QuantityEvaluator {
            programs: vec![
                // 0: value_at(S, last_obs) — in-window, state ⇒ Smoothed.
                QProgram::Reduced {
                    source: QSource::State(ResolvedExpr::IntPop(0)),
                    reduce: Some(RReduce::ValueAt(RAnchor::Obs(AnchoredTime::bare(
                        ObsAnchor::Last,
                    )))),
                },
                // 1: value_at(S, last_obs + 7) — past the record ⇒ Replay.
                QProgram::Reduced {
                    source: QSource::State(ResolvedExpr::IntPop(0)),
                    reduce: Some(RReduce::ValueAt(RAnchor::Obs(AnchoredTime {
                        anchor: ObsAnchor::Last,
                        offset: 7.0,
                    }))),
                },
                // 2: final(S) — not a value_at at all ⇒ Replay.
                QProgram::Reduced {
                    source: QSource::State(ResolvedExpr::IntPop(0)),
                    reduce: Some(RReduce::Final),
                },
                // 3: time_of_max(S) ⇒ Replay.
                QProgram::Reduced {
                    source: QSource::State(ResolvedExpr::IntPop(0)),
                    reduce: Some(RReduce::TimeOfMax),
                },
            ],
            names: vec![
                "at_last_obs".into(),
                "past_last_obs".into(),
                "final_s".into(),
                "peak_time".into(),
            ],
        }
    }

    /// The classification table, quantity by quantity. This is what makes the
    /// routing a property of the QUANTITY rather than of a draw.
    #[test]
    fn only_an_in_window_state_value_at_routes_to_the_smoothing_path() {
        use ir::anchor::{AnchoredTime, ObsAnchor};
        let w = ObsAnchorTimes { first: 0.0, last: 14.0 };
        assert_eq!(
            evaluator_722().eval_paths(Some(w)),
            vec![
                QuantityPath::Smoothed,
                QuantityPath::Replay,
                QuantityPath::Replay,
                QuantityPath::Replay,
            ],
        );
        assert_eq!(
            evaluator_722().quantity_names_on(QuantityPath::Smoothed, Some(w)),
            vec!["at_last_obs"],
        );
        // `first_obs` is inside the record too.
        let first = QuantityEvaluator {
            programs: vec![QProgram::Reduced {
                source: QSource::State(ResolvedExpr::IntPop(0)),
                reduce: Some(RReduce::ValueAt(RAnchor::Obs(AnchoredTime::bare(
                    ObsAnchor::First,
                )))),
            }],
            names: vec!["q".into()],
        };
        assert_eq!(first.eval_paths(Some(w)), vec![QuantityPath::Smoothed]);
        // A LITERAL time inside the record has the same defect as the
        // `last_obs` spelling; one past it is a projection.
        let literal = |t: f64| QuantityEvaluator {
            programs: vec![QProgram::Reduced {
                source: QSource::State(ResolvedExpr::IntPop(0)),
                reduce: Some(RReduce::ValueAt(RAnchor::Expr(ResolvedExpr::Const(t)))),
            }],
            names: vec!["q".into()],
        };
        assert_eq!(literal(10.0).eval_paths(Some(w)), vec![QuantityPath::Smoothed]);
        assert_eq!(literal(14.0).eval_paths(Some(w)), vec![QuantityPath::Smoothed]);
        assert_eq!(literal(14.001).eval_paths(Some(w)), vec![QuantityPath::Replay]);
        // A non-constant time cannot be classified without a draw — it stays on
        // the replay rather than making one band a mixture of two objects.
        let param_time = QuantityEvaluator {
            programs: vec![QProgram::Reduced {
                source: QSource::State(ResolvedExpr::IntPop(0)),
                reduce: Some(RReduce::ValueAt(RAnchor::Expr(ResolvedExpr::Param(0)))),
            }],
            names: vec!["q".into()],
        };
        assert_eq!(param_time.eval_paths(Some(w)), vec![QuantityPath::Replay]);
        // An `observations.<stream>` value_at inside the record is NAMED, not
        // silently passed off as conditioned: no saved path carries a y_sim.
        let obs_src = QuantityEvaluator {
            programs: vec![QProgram::Reduced {
                source: QSource::Observation("cases".into()),
                reduce: Some(RReduce::ValueAt(RAnchor::Obs(AnchoredTime::bare(
                    ObsAnchor::Last,
                )))),
            }],
            names: vec!["cases_at_last_obs".into()],
        };
        assert_eq!(obs_src.eval_paths(Some(w)), vec![QuantityPath::ReplayUnconditioned]);
        assert_eq!(
            obs_src.quantity_names_on(QuantityPath::ReplayUnconditioned, Some(w)),
            vec!["cases_at_last_obs"],
        );
        // No observation window ⇒ nothing is anchorable.
        assert_eq!(evaluator_722().eval_paths(None), vec![QuantityPath::Replay; 4]);
    }

    /// The gh#722 fix, with the two answers MEASURABLY apart: the replay says
    /// `S = 900` at `last_obs` while the smoothing path says `S = 100`. The
    /// in-window `value_at` must read 100; the three controls must read the
    /// replay's values, or the routing has leaked past `value_at`.
    #[test]
    fn an_in_window_value_at_reads_the_smoothing_path_not_the_replay() {
        let compiled = one_compartment_model();
        let w = ObsAnchorTimes { first: 0.0, last: 14.0 };
        // Replay: barely moves (the unconditioned epidemic never took off).
        let replay = path_of(&[0.0, 7.0, 14.0, 21.0], &[1000, 950, 900, 880]);
        // Smoothing path: the epidemic the data actually forced, ending with
        // the observed record.
        let smoothed = path_of(&[0.0, 7.0, 14.0], &[1000, 500, 100]);
        let eval = evaluator_722();

        let got = eval.eval_draw(
            &[1.0],
            &replay,
            ConditionedRead::Saved(&smoothed),
            &compiled,
            None,
            Some(w),
        );
        assert_eq!(got[0], QuantityResult::Scalar(Value(100.0)), "at_last_obs must be smoothed");
        assert_eq!(
            got[1],
            QuantityResult::Scalar(Value(880.0)),
            "past_last_obs stays on the replay"
        );
        assert_eq!(got[2], QuantityResult::Scalar(Value(880.0)), "final(S) stays on the replay");
        assert_eq!(got[3], QuantityResult::Scalar(Value(0.0)), "time_of_max stays on the replay");

        // The control that makes the first assertion non-vacuous: with no
        // conditioned read the SAME call returns the replay's 900.
        let off = eval.eval_draw(&[1.0], &replay, ConditionedRead::Off, &compiled, None, Some(w));
        assert_eq!(off[0], QuantityResult::Scalar(Value(900.0)), "Off is the pre-fix answer");
        assert_eq!(off[1..], got[1..], "Off changes nothing outside the routed quantity");
    }

    /// A draw outside the forkable subset loses the in-window value; it does
    /// NOT gain a free-forward substitute. Censoring is what keeps the band
    /// over the draws that have a conditioned answer.
    #[test]
    fn a_draw_with_no_saved_path_censors_rather_than_falling_back() {
        let compiled = one_compartment_model();
        let w = ObsAnchorTimes { first: 0.0, last: 14.0 };
        let replay = path_of(&[0.0, 7.0, 14.0, 21.0], &[1000, 950, 900, 880]);
        let got = evaluator_722().eval_draw(
            &[1.0],
            &replay,
            ConditionedRead::NotSaved,
            &compiled,
            None,
            Some(w),
        );
        assert_eq!(got[0], QuantityResult::Scalar(Censored));
        // Everything else is untouched — a missing path costs one quantity, not
        // the whole draw.
        assert_eq!(got[1], QuantityResult::Scalar(Value(880.0)));
        assert_eq!(got[2], QuantityResult::Scalar(Value(880.0)));
    }

    /// Reduction arithmetic over a smoothed leaf carries the smoothed value —
    /// and its censoring — because a `Derived` reads the SPLICED results, not a
    /// second replay-only pass.
    #[test]
    fn a_derived_leaf_reads_the_spliced_smoothed_value() {
        use ir::anchor::{AnchoredTime, ObsAnchor};
        let compiled = one_compartment_model();
        let w = ObsAnchorTimes { first: 0.0, last: 14.0 };
        let replay = path_of(&[0.0, 7.0, 14.0, 21.0], &[1000, 950, 900, 880]);
        let smoothed = path_of(&[0.0, 7.0, 14.0], &[1000, 500, 100]);
        // q0 = value_at(S, last_obs); q1 = 1000 - q0 (cumulative infections).
        let eval = QuantityEvaluator {
            programs: vec![
                QProgram::Reduced {
                    source: QSource::State(ResolvedExpr::IntPop(0)),
                    reduce: Some(RReduce::ValueAt(RAnchor::Obs(AnchoredTime::bare(
                        ObsAnchor::Last,
                    )))),
                },
                QProgram::Derived(RScalar::BinOp {
                    op: BinOp::Sub,
                    left: Box::new(RScalar::Const(1000.0)),
                    right: Box::new(RScalar::QRef(0)),
                }),
            ],
            names: vec!["s_at_last_obs".into(), "outbreak_size".into()],
        };
        let got = eval.eval_draw(
            &[1.0],
            &replay,
            ConditionedRead::Saved(&smoothed),
            &compiled,
            None,
            Some(w),
        );
        assert_eq!(got[1], QuantityResult::Scalar(Value(900.0)), "1000 - 100, not 1000 - 900");
        // Censoring propagates through the arithmetic the same way.
        let missing = eval.eval_draw(
            &[1.0],
            &replay,
            ConditionedRead::NotSaved,
            &compiled,
            None,
            Some(w),
        );
        assert_eq!(missing[1], QuantityResult::Scalar(Censored));
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
