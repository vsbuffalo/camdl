//! Reactive intervention runtime (gh#204, PR2) — forward chain-binomial.
//!
//! Slice 1: the **trigger-predicate evaluator**. Given a [`TriggerExpr`] and a
//! way to resolve its observed quantities and thresholds, decide whether a
//! policy fires. This is pure (no I/O, no RNG); the boolean structure and the
//! windowed reducers are unit-tested here. Later slices feed it the realized
//! observation draws (on a dedicated RNG stream) and enqueue the resulting
//! effects through the scheduled-intervention due-batch.

use crate::compiled_model::CompiledModel;
use crate::inference::multi_stream_obs::{eval_stream_projection, StreamProjection};
use crate::state::RealState;
use ir::intervention::{CmpOp, ObsReducer, TriggerExpr, TriggerQuantity, TriggerThreshold};
use ir::observation::{ObservationSchedule, TemporalKind};

/// Inclusive upper-bound tolerance for the trailing window: an emit landing
/// exactly at `now` is included. The window is `(now - window, now]`
/// (open-left, closed-right).
const WINDOW_EPS: f64 = 1e-9;

/// Fold one stream's realized-observation history into a single trigger value.
/// `history` is `(emit_time, realized_Y)` in ascending time; `now` is the
/// current trigger-evaluation time. `Latest` ignores the window (the most recent
/// draw); `Sum`/`Mean`/`Max` fold over the trailing window `(now - window, now]`.
/// An empty selection reduces to `0.0` ("nothing detected").
pub fn reduce_obs(
    history: &[(f64, f64)],
    window: Option<f64>,
    reducer: ObsReducer,
    now: f64,
) -> f64 {
    if let ObsReducer::Latest = reducer {
        return history.last().map(|&(_, y)| y).unwrap_or(0.0);
    }
    let lo = window.map(|w| now - w).unwrap_or(f64::NEG_INFINITY);
    let mut sum = 0.0;
    let mut count = 0usize;
    let mut max = f64::NEG_INFINITY;
    for &(t, y) in history {
        if t > lo && t <= now + WINDOW_EPS {
            sum += y;
            count += 1;
            if y > max {
                max = y;
            }
        }
    }
    match reducer {
        ObsReducer::Sum => sum,
        ObsReducer::Mean => {
            if count == 0 {
                0.0
            } else {
                sum / count as f64
            }
        }
        ObsReducer::Max => {
            if count == 0 {
                0.0
            } else {
                max
            }
        }
        ObsReducer::Latest => unreachable!("Latest handled above"),
    }
}

/// Evaluate a trigger predicate to a boolean. `quantity` resolves a
/// [`TriggerQuantity`] to its realized value (typically via [`reduce_obs`] over
/// the obs history); `threshold` resolves a [`TriggerThreshold`] (constant or
/// parameter). Both are injected so this stays pure and unit-testable; the
/// runtime supplies the obs-history- and params-backed closures.
pub fn eval_trigger(
    expr: &TriggerExpr,
    quantity: &dyn Fn(&TriggerQuantity) -> f64,
    threshold: &dyn Fn(&TriggerThreshold) -> f64,
) -> bool {
    match expr {
        TriggerExpr::Cmp { lhs, op, rhs } => apply_cmp(*op, quantity(lhs), threshold(rhs)),
        TriggerExpr::And(a, b) => {
            eval_trigger(a, quantity, threshold) && eval_trigger(b, quantity, threshold)
        }
        TriggerExpr::Or(a, b) => {
            eval_trigger(a, quantity, threshold) || eval_trigger(b, quantity, threshold)
        }
        TriggerExpr::Not(a) => !eval_trigger(a, quantity, threshold),
    }
}

fn apply_cmp(op: CmpOp, l: f64, r: f64) -> bool {
    match op {
        CmpOp::Lt => l < r,
        CmpOp::Le => l <= r,
        CmpOp::Gt => l > r,
        CmpOp::Ge => l >= r,
        CmpOp::Eq => l == r,
        CmpOp::Neq => l != r,
    }
}

// ── Slice 2: per-observation-stream interval evaluator ────────────────────────

/// Tolerance for matching an emit time to the current substep boundary.
const EMIT_EPS: f64 = 1e-9;

/// Resolve an observation `emit_schedule` to concrete emit times. Mirrors the
/// CLI's `obs_schedule_times` (kept here so the sim crate is self-contained).
fn schedule_emit_times(s: &ObservationSchedule) -> Vec<f64> {
    match s {
        ObservationSchedule::AtTimes(ts) => ts.clone(),
        ObservationSchedule::Regular(r) => {
            let mut out = Vec::new();
            let mut t = r.start;
            while t <= r.end + EMIT_EPS {
                out.push(t);
                t += r.step;
            }
            out
        }
    }
}

/// Forward-sim observation evaluator for the streams a reactive trigger reads.
/// Per stream it owns the resolved [`StreamProjection`] (reusing the inference
/// projection machinery) and a **per-stream interval flow accumulator reset at
/// that stream's own emit times** — NOT the output-tied `current_flows`, whose
/// cadence can differ (constraint: read flow over the *observation* interval).
///
/// Slice 2 is the deterministic projection over the obs interval; the realized
/// draw `Y` (the sampler on a dedicated obs RNG) is wired in slice 3.
pub struct ReactiveObs {
    streams: Vec<ReactiveStream>,
}

struct ReactiveStream {
    name: String,
    projection: StreamProjection,
    kind: TemporalKind,
    /// Per-transition flows accumulated since this stream's last emit (length =
    /// n_transitions). Meaningful for `Interval` (incidence) streams, where
    /// `eval_stream_projection`'s `FlowSum` sums the relevant indices; reset at
    /// each emit. `Instant` (prevalence) streams ignore it (read state at emit).
    interval_flows: Vec<u64>,
    emit_times: Vec<f64>,
}

impl ReactiveObs {
    /// Build the evaluator for the observation-stream names a reactive trigger
    /// references. Each must be a declared observation carrying an
    /// `emit_schedule`. (Stream existence is also enforced at compile time —
    /// E279 — so the not-found arms here are defensive.)
    pub fn from_model(compiled: &CompiledModel, stream_names: &[String]) -> Result<Self, String> {
        let n_tr = compiled.model.transitions.len();
        let mut streams = Vec::with_capacity(stream_names.len());
        for name in stream_names {
            let obs = compiled
                .model
                .observations
                .iter()
                .find(|o| o.name == *name)
                .ok_or_else(|| {
                    format!("reactive trigger references unknown observation stream '{name}'")
                })?;
            let projection = StreamProjection::from_ir(&obs.projection, compiled, name)?;
            let kind = obs.projection.temporal_kind();
            let emit_times = match &obs.emit_schedule {
                Some(s) => schedule_emit_times(s),
                None => {
                    return Err(format!(
                        "reactive trigger stream '{name}' has no emit_schedule — \
                         it cannot produce the observations the trigger reads"
                    ))
                }
            };
            streams.push(ReactiveStream {
                name: name.clone(),
                projection,
                kind,
                interval_flows: vec![0; n_tr],
                emit_times,
            });
        }
        Ok(ReactiveObs { streams })
    }

    /// Add one substep's per-transition `flows` to every `Interval` stream's
    /// interval accumulator.
    pub fn accumulate(&mut self, flows: &[u64]) {
        for s in &mut self.streams {
            if s.kind == TemporalKind::Interval {
                for (acc, &f) in s.interval_flows.iter_mut().zip(flows) {
                    *acc += f;
                }
            }
        }
    }

    /// Stream indices whose emit time matches `t` (within tolerance).
    pub fn due_at(&self, t: f64) -> Vec<usize> {
        self.streams
            .iter()
            .enumerate()
            .filter(|(_, s)| s.emit_times.iter().any(|&e| (e - t).abs() <= EMIT_EPS))
            .map(|(i, _)| i)
            .collect()
    }

    /// The projected value for stream `idx` at time `t`: interval incidence (for
    /// `Interval` streams, from the accumulator) or instantaneous prevalence
    /// (for `Instant` streams, from `counts`/state). Reuses the inference
    /// projection evaluator so forward and inference agree.
    pub fn projected(
        &self,
        idx: usize,
        counts: &[i64],
        real_s: &RealState,
        params: &[f64],
        compiled: &CompiledModel,
        t: f64,
    ) -> f64 {
        let s = &self.streams[idx];
        eval_stream_projection(&s.projection, &s.interval_flows, counts, params, compiled, real_s, t)
    }

    /// Reset stream `idx`'s interval accumulator (after its emit).
    pub fn reset(&mut self, idx: usize) {
        for a in &mut self.streams[idx].interval_flows {
            *a = 0;
        }
    }

    /// The stream name at `idx` (for the reactive log / diagnostics).
    pub fn stream_name(&self, idx: usize) -> &str {
        &self.streams[idx].name
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn obs(stream: &str, window: Option<f64>, reducer: ObsReducer) -> TriggerQuantity {
        TriggerQuantity::Observed { stream: stream.into(), window, reducer }
    }

    #[test]
    fn latest_ignores_window_and_empty_is_zero() {
        let h = [(7.0, 2.0), (14.0, 5.0), (21.0, 3.0)];
        assert_eq!(reduce_obs(&h, None, ObsReducer::Latest, 21.0), 3.0);
        assert_eq!(reduce_obs(&[], None, ObsReducer::Latest, 21.0), 0.0);
    }

    #[test]
    fn sum_over_trailing_window_is_open_left_closed_right() {
        // weekly emits; window 28 at now=28 covers (0, 28] = {7,14,21,28}.
        // The boundary emit at now-window=0 is excluded; the current emit at
        // now=28 is included.
        let h = [(0.0, 100.0), (7.0, 2.0), (14.0, 5.0), (21.0, 3.0), (28.0, 4.0)];
        assert_eq!(reduce_obs(&h, Some(28.0), ObsReducer::Sum, 28.0), 2.0 + 5.0 + 3.0 + 4.0);
        // a tighter window keeps only the most recent emits
        assert_eq!(reduce_obs(&h, Some(14.0), ObsReducer::Sum, 28.0), 3.0 + 4.0);
    }

    #[test]
    fn mean_and_max_over_window_with_empty_floor() {
        let h = [(7.0, 2.0), (14.0, 6.0), (21.0, 4.0)];
        assert_eq!(reduce_obs(&h, Some(21.0), ObsReducer::Mean, 21.0), (2.0 + 6.0 + 4.0) / 3.0);
        assert_eq!(reduce_obs(&h, Some(21.0), ObsReducer::Max, 21.0), 6.0);
        assert_eq!(reduce_obs(&[], Some(21.0), ObsReducer::Sum, 21.0), 0.0);
        assert_eq!(reduce_obs(&[], Some(21.0), ObsReducer::Mean, 21.0), 0.0);
        assert_eq!(reduce_obs(&[], Some(21.0), ObsReducer::Max, 21.0), 0.0);
    }

    #[test]
    fn every_comparison_operator() {
        let q = |_: &TriggerQuantity| 5.0;
        let t = |th: &TriggerThreshold| match th {
            TriggerThreshold::Const(c) => *c,
            TriggerThreshold::Param(_) => 0.0,
        };
        let cmp = |op| TriggerExpr::Cmp {
            lhs: obs("x", None, ObsReducer::Latest),
            op,
            rhs: TriggerThreshold::Const(5.0),
        };
        assert!(eval_trigger(&cmp(CmpOp::Ge), &q, &t));
        assert!(eval_trigger(&cmp(CmpOp::Le), &q, &t));
        assert!(!eval_trigger(&cmp(CmpOp::Gt), &q, &t));
        assert!(!eval_trigger(&cmp(CmpOp::Lt), &q, &t));
        assert!(eval_trigger(&cmp(CmpOp::Eq), &q, &t));
        assert!(!eval_trigger(&cmp(CmpOp::Neq), &q, &t));
    }

    #[test]
    fn and_or_not_and_param_threshold() {
        let q = |_: &TriggerQuantity| 5.0;
        let t = |th: &TriggerThreshold| match th {
            TriggerThreshold::Const(c) => *c,
            TriggerThreshold::Param(p) => {
                if p == "thr" {
                    3.0
                } else {
                    0.0
                }
            }
        };
        // 5 >= thr(3) → true
        let ge = TriggerExpr::Cmp {
            lhs: obs("x", None, ObsReducer::Latest),
            op: CmpOp::Ge,
            rhs: TriggerThreshold::Param("thr".into()),
        };
        // 5 < 4 → false
        let lt = TriggerExpr::Cmp {
            lhs: obs("x", None, ObsReducer::Latest),
            op: CmpOp::Lt,
            rhs: TriggerThreshold::Const(4.0),
        };
        assert!(eval_trigger(&ge, &q, &t));
        assert!(!eval_trigger(&lt, &q, &t));
        // true && !false
        assert!(eval_trigger(
            &TriggerExpr::And(Box::new(ge.clone()), Box::new(TriggerExpr::Not(Box::new(lt.clone())))),
            &q,
            &t
        ));
        // false || true
        assert!(eval_trigger(&TriggerExpr::Or(Box::new(lt.clone()), Box::new(ge.clone())), &q, &t));
        // !(false) → true
        assert!(eval_trigger(&TriggerExpr::Not(Box::new(lt)), &q, &t));
    }

    // ── slice 2: interval accumulator + emit schedule ──

    #[test]
    fn emit_schedule_regular_and_at_times() {
        use ir::observation::{ObservationSchedule, RegularSchedule};
        let reg = ObservationSchedule::Regular(RegularSchedule { start: 0.0, step: 7.0, end: 28.0 });
        assert_eq!(schedule_emit_times(&reg), vec![0.0, 7.0, 14.0, 21.0, 28.0]);
        let at = ObservationSchedule::AtTimes(vec![3.0, 9.0]);
        assert_eq!(schedule_emit_times(&at), vec![3.0, 9.0]);
    }

    #[test]
    fn interval_accumulator_accumulates_and_resets() {
        // One-stream ReactiveObs (FlowSum over transition 0, Interval), built
        // directly so the accumulator logic is tested without a CompiledModel.
        let mut ro = ReactiveObs {
            streams: vec![ReactiveStream {
                name: "weekly".into(),
                projection: StreamProjection::FlowSum(vec![0]),
                kind: TemporalKind::Interval,
                interval_flows: vec![0],
                emit_times: vec![7.0, 14.0],
            }],
        };
        ro.accumulate(&[3]);
        ro.accumulate(&[2]);
        assert_eq!(ro.streams[0].interval_flows, vec![5], "accumulates over the interval");
        assert_eq!(ro.due_at(7.0), vec![0]);
        assert!(ro.due_at(10.0).is_empty(), "no emit at 10");
        ro.reset(0);
        assert_eq!(ro.streams[0].interval_flows, vec![0], "reset zeros the interval");
        ro.accumulate(&[4]);
        assert_eq!(ro.streams[0].interval_flows, vec![4], "next interval starts fresh");
    }

    #[test]
    fn instant_stream_ignores_flow_accumulation() {
        let mut ro = ReactiveObs {
            streams: vec![ReactiveStream {
                name: "prev".into(),
                projection: StreamProjection::IntCompSum(vec![0]),
                kind: TemporalKind::Instant,
                interval_flows: vec![0],
                emit_times: vec![7.0],
            }],
        };
        ro.accumulate(&[9]);
        assert_eq!(
            ro.streams[0].interval_flows,
            vec![0],
            "Instant (prevalence) streams read state at the emit, not accumulated flow"
        );
    }
}
