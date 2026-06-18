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
use crate::inference::obs_model::{resolve_likelihood_from_model, sample_obs_resolved};
use crate::resolved_expr::ResolvedLikelihood;
use crate::rng::StatefulRng;
use crate::state::{IntState, RealState};
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
    /// The stream's resolved measurement model — used to draw the realized
    /// observation `Y` (slice 3) via [`sample_obs_resolved`].
    resolved: ResolvedLikelihood,
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
            let resolved = resolve_likelihood_from_model(&obs.likelihood, compiled)
                .map_err(|e| {
                    format!("reactive trigger stream '{name}': likelihood resolution failed: {e:?}")
                })?;
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
                resolved,
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

    /// Draw the realized observation `Y` for stream `idx` at time `t`: compute
    /// the interval projection (slice 2), then sample from the stream's
    /// measurement model on the supplied obs RNG. `rng` is a stream dedicated to
    /// observation draws (kept separate from the dynamics RNG by the caller, so
    /// paired-seed / CRN coupling holds). `aux` is empty — forward reactive
    /// triggers read projection-based likelihoods (poisson / normal /
    /// neg_binomial); aux-data-column likelihoods are not a PR2 trigger surface.
    pub fn draw(
        &self,
        idx: usize,
        int_s: &IntState,
        real_s: &RealState,
        params: &[f64],
        compiled: &CompiledModel,
        t: f64,
        rng: &mut StatefulRng,
    ) -> f64 {
        let projected = self.projected(idx, &int_s.counts, real_s, params, compiled, t);
        sample_obs_resolved(
            &self.streams[idx].resolved,
            t,
            projected,
            &[],
            params,
            compiled,
            int_s,
            real_s,
            rng,
        )
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

    /// The index of the stream named `name`, if present.
    pub fn stream_index(&self, name: &str) -> Option<usize> {
        self.streams.iter().position(|s| s.name == name)
    }

    /// Number of streams.
    pub fn len(&self) -> usize {
        self.streams.len()
    }

    /// Whether there are no streams.
    pub fn is_empty(&self) -> bool {
        self.streams.is_empty()
    }
}

/// Collect the observation-stream names a trigger predicate reads (deduplicated,
/// declaration order). Mirrors the expander's `trigger_stream_refs`.
pub fn trigger_stream_refs(expr: &TriggerExpr) -> Vec<String> {
    let mut out = Vec::new();
    collect_refs(expr, &mut out);
    out
}

fn collect_refs(expr: &TriggerExpr, out: &mut Vec<String>) {
    match expr {
        TriggerExpr::Cmp { lhs: TriggerQuantity::Observed { stream, .. }, .. } => {
            if !out.iter().any(|s| s == stream) {
                out.push(stream.clone());
            }
        }
        TriggerExpr::And(a, b) | TriggerExpr::Or(a, b) => {
            collect_refs(a, out);
            collect_refs(b, out);
        }
        TriggerExpr::Not(a) => collect_refs(a, out),
    }
}

/// The leftmost `Cmp` leaf of a trigger predicate — the comparison whose
/// realized observed value and threshold the reactive log reports. Exact for a
/// single-comparison trigger (the only kind any behavior golden uses); for a
/// compound predicate it is the leftmost leaf, with the full predicate in the
/// model.
fn primary_comparison(expr: &TriggerExpr) -> Option<(&TriggerQuantity, &TriggerThreshold)> {
    match expr {
        TriggerExpr::Cmp { lhs, rhs, .. } => Some((lhs, rhs)),
        TriggerExpr::And(a, b) | TriggerExpr::Or(a, b) => {
            primary_comparison(a).or_else(|| primary_comparison(b))
        }
        TriggerExpr::Not(a) => primary_comparison(a),
    }
}

/// The action verb(s) a policy fires, joined with `;` — the reactive log's
/// `action` column (`transfer`/`set`/`add`).
fn action_verbs(actions: &[ir::intervention::Action]) -> String {
    use ir::intervention::Action;
    actions
        .iter()
        .map(|a| match a {
            Action::FractionTransfer(_) | Action::AbsoluteTransfer(_) => "transfer",
            Action::Set(_) => "set",
            Action::Add(_) => "add",
        })
        .collect::<Vec<_>>()
        .join(";")
}

/// Render the reactive firings as the `reactive_log.tsv` body: a header row plus
/// one tab-separated row per firing. Numbers use the same `Display` formatting
/// as `traj.tsv`, so a whole-valued time prints without a trailing `.0`.
pub fn format_reactive_log(firings: &[ReactiveFiring]) -> String {
    let mut s = String::from("trigger_time\tpolicy\ttrigger_value\tthreshold\tfire_time\taction\n");
    for f in firings {
        s.push_str(&format!(
            "{}\t{}\t{}\t{}\t{}\t{}\n",
            f.trigger_time, f.policy, f.trigger_value, f.threshold, f.fire_time, f.action
        ));
    }
    s
}

// ── Slice 4: the reactive agenda ──────────────────────────────────────────────

/// Per-reactive-policy runtime state for one forward run.
struct PolicyRt {
    /// Index into `model.interventions` — the action this policy fires (applied
    /// through the SAME `apply_intervention_effects` due-batch as a scheduled
    /// intervention).
    iv_idx: usize,
    when: TriggerExpr,
    after: f64,
    once: bool,
    cooldown: Option<f64>,
    last_fired: Option<f64>,
    times_fired: u32,
}

/// A future effect discovered at an emit time, due at `fire_time = trigger + after`.
struct Pending {
    fire_time: f64,
    seq: u64,
    iv_idx: usize,
}

impl PartialEq for Pending {
    fn eq(&self, o: &Self) -> bool {
        self.fire_time == o.fire_time && self.seq == o.seq
    }
}
impl Eq for Pending {}
impl PartialOrd for Pending {
    fn partial_cmp(&self, o: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(o))
    }
}
impl Ord for Pending {
    /// Order by fire time, then enqueue sequence (stable tie-break). Used under
    /// `Reverse` so the `BinaryHeap` is a min-heap on `fire_time`.
    fn cmp(&self, o: &Self) -> std::cmp::Ordering {
        self.fire_time.total_cmp(&o.fire_time).then(self.seq.cmp(&o.seq))
    }
}

/// A reactive firing recorded for the `reactive_log.tsv` (slice 5).
///
/// `trigger_value`/`threshold` report the policy's *primary* (leftmost)
/// comparison — exact for a single-comparison trigger (what policies use in
/// practice and what every behavior golden exercises). For a compound `&&`/`||`
/// predicate they reflect the leftmost `Cmp` leaf; the full predicate stays in
/// the model. `action` is the action verb(s) the policy fires.
#[derive(Debug, Clone, PartialEq)]
pub struct ReactiveFiring {
    pub trigger_time: f64,
    pub policy: String,
    pub trigger_value: f64,
    pub threshold: f64,
    pub fire_time: f64,
    pub action: String,
}

/// The forward-run reactive agenda: the per-stream observation evaluator
/// ([`ReactiveObs`]), the per-stream realized-obs history (for the trigger
/// reducers), per-policy state, and a min-heap of pending effects. Owned by the
/// backend run state; never in the immutable shared `Schedule`.
pub struct ReactiveAgenda {
    obs: ReactiveObs,
    obs_history: Vec<Vec<(f64, f64)>>,
    policies: Vec<PolicyRt>,
    pending: std::collections::BinaryHeap<std::cmp::Reverse<Pending>>,
    seq: u64,
    /// Firings recorded this run (for the reactive log).
    log: Vec<ReactiveFiring>,
}

impl ReactiveAgenda {
    /// Build the agenda for a model's reactive policies, or `None` if it has
    /// none (so the forward backends pay nothing when no policy is present).
    pub fn from_model(compiled: &CompiledModel) -> Result<Option<Self>, String> {
        let mut policies = Vec::new();
        let mut stream_names: Vec<String> = Vec::new();
        for (iv_idx, iv) in compiled.model.interventions.iter().enumerate() {
            if let ir::intervention::FireSource::Reactive(t) = &iv.fire {
                for s in trigger_stream_refs(&t.when_) {
                    if !stream_names.iter().any(|n| *n == s) {
                        stream_names.push(s);
                    }
                }
                policies.push(PolicyRt {
                    iv_idx,
                    when: t.when_.clone(),
                    after: t.after,
                    once: t.once,
                    cooldown: t.cooldown,
                    last_fired: None,
                    times_fired: 0,
                });
            }
        }
        if policies.is_empty() {
            return Ok(None);
        }
        let obs = ReactiveObs::from_model(compiled, &stream_names)?;
        let n_streams = obs.len();
        Ok(Some(ReactiveAgenda {
            obs,
            obs_history: vec![Vec::new(); n_streams],
            policies,
            pending: std::collections::BinaryHeap::new(),
            seq: 0,
            log: Vec::new(),
        }))
    }

    /// Add one substep's per-transition flows to every Interval stream's
    /// interval accumulator.
    pub fn accumulate(&mut self, flows: &[u64]) {
        self.obs.accumulate(flows);
    }

    /// At an observation boundary `t`: draw the realized obs for every stream
    /// emitting at `t` (recording history + resetting that stream's interval),
    /// then evaluate each policy's trigger and enqueue its effect at
    /// `t + after`. Read-then-write split avoids aliasing the agenda's fields.
    pub fn on_boundary(
        &mut self,
        t: f64,
        int_s: &IntState,
        real_s: &RealState,
        params: &[f64],
        compiled: &CompiledModel,
        obs_rng: &mut StatefulRng,
    ) {
        // 1. realized draws for streams emitting at t.
        for si in self.obs.due_at(t) {
            let y = self.obs.draw(si, int_s, real_s, params, compiled, t, obs_rng);
            self.obs_history[si].push((t, y));
            self.obs.reset(si);
        }
        // 2. read phase: which policies fire now (gating: once + cooldown).
        let obs = &self.obs;
        let hist = &self.obs_history;
        let quantity = |q: &TriggerQuantity| -> f64 {
            let TriggerQuantity::Observed { stream, window, reducer } = q;
            obs.stream_index(stream)
                .map(|si| reduce_obs(&hist[si], *window, *reducer, t))
                .unwrap_or(0.0)
        };
        let threshold = |th: &TriggerThreshold| -> f64 {
            match th {
                TriggerThreshold::Const(c) => *c,
                TriggerThreshold::Param(p) => {
                    compiled.param_index.get(p).map(|&i| params[i]).unwrap_or(0.0)
                }
            }
        };
        let fired: Vec<usize> = (0..self.policies.len())
            .filter(|&pi| {
                let p = &self.policies[pi];
                if p.once && p.times_fired > 0 {
                    return false;
                }
                if let (Some(cd), Some(lf)) = (p.cooldown, p.last_fired) {
                    if t - lf < cd {
                        return false;
                    }
                }
                eval_trigger(&p.when, &quantity, &threshold)
            })
            .collect();
        // 3. write phase: enqueue + record + update policy state.
        for pi in fired {
            // The realized observed value + resolved threshold of the policy's
            // primary comparison — what the log reports for the crossing.
            let (trigger_value, threshold_value) = primary_comparison(&self.policies[pi].when)
                .map(|(q, th)| (quantity(q), threshold(th)))
                .unwrap_or((f64::NAN, f64::NAN));
            let p = &mut self.policies[pi];
            let fire_time = t + p.after;
            p.last_fired = Some(t);
            p.times_fired += 1;
            let iv_idx = p.iv_idx;
            self.pending
                .push(std::cmp::Reverse(Pending { fire_time, seq: self.seq, iv_idx }));
            self.seq += 1;
            let iv = &compiled.model.interventions[iv_idx];
            self.log.push(ReactiveFiring {
                trigger_time: t,
                policy: iv.name.clone(),
                trigger_value,
                threshold: threshold_value,
                fire_time,
                action: action_verbs(&iv.actions),
            });
        }
    }

    /// Pop every pending effect due at or before `boundary` and return the
    /// intervention indices to merge into the scheduled due-batch (so they fire
    /// through the same `apply_intervention_effects` + post-advance + balance +
    /// negative-count lifecycle). A `fire_time <= boundary` test (not step
    /// equality) makes `after = 0` fire at the first boundary after the trigger
    /// (post-observation, next interval).
    pub fn due_iv_idxs(&mut self, boundary: f64) -> Vec<usize> {
        let mut out = Vec::new();
        while let Some(std::cmp::Reverse(p)) = self.pending.peek() {
            if p.fire_time <= boundary + EMIT_EPS {
                let std::cmp::Reverse(p) = self.pending.pop().unwrap();
                out.push(p.iv_idx);
            } else {
                break;
            }
        }
        out
    }

    /// The firings recorded this run (for `reactive_log.tsv`, slice 5).
    pub fn firings(&self) -> &[ReactiveFiring] {
        &self.log
    }

    /// Consume the agenda, taking ownership of its recorded firings — used at
    /// the end of a run to move the log into the [`Trajectory`].
    pub fn into_firings(self) -> Vec<ReactiveFiring> {
        self.log
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
                resolved: ResolvedLikelihood::Poisson {
                    rate: crate::resolved_expr::ResolvedExpr::Const(1.0),
                },
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
                resolved: ResolvedLikelihood::Poisson {
                    rate: crate::resolved_expr::ResolvedExpr::Const(1.0),
                },
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

    #[test]
    fn format_reactive_log_matches_proposal_columns() {
        // Pins the exact `reactive_log.tsv` schema from the proposal's lag
        // fixture: `trigger_time policy trigger_value threshold fire_time
        // action`, whole-valued times printed without a trailing `.0`.
        let firings = vec![ReactiveFiring {
            trigger_time: 28.0,
            policy: "sia".into(),
            trigger_value: 2.0,
            threshold: 2.0,
            fire_time: 49.0,
            action: "transfer".into(),
        }];
        let tsv = format_reactive_log(&firings);
        assert_eq!(
            tsv,
            "trigger_time\tpolicy\ttrigger_value\tthreshold\tfire_time\taction\n\
             28\tsia\t2\t2\t49\ttransfer\n"
        );
        // No firings ⇒ header only (the declared-but-empty artifact a
        // reactive-active run with no crossing writes).
        assert_eq!(
            format_reactive_log(&[]),
            "trigger_time\tpolicy\ttrigger_value\tthreshold\tfire_time\taction\n"
        );
    }

    #[test]
    fn action_verbs_maps_each_variant() {
        use ir::intervention::{
            Action, AddAction, FractionTransfer, SetAction,
        };
        use ir::expr::Expr;
        let frac = Action::FractionTransfer(FractionTransfer {
            src: "S".into(), dst: "V".into(), fraction: Expr::const_(0.7),
        });
        let set = Action::Set(SetAction { compartment: "S".into(), value: Expr::const_(0.0) });
        let add = Action::Add(AddAction { compartment: "I".into(), count: Expr::const_(1.0) });
        assert_eq!(action_verbs(&[frac.clone()]), "transfer");
        assert_eq!(action_verbs(&[set.clone()]), "set");
        assert_eq!(action_verbs(&[add.clone()]), "add");
        assert_eq!(action_verbs(&[frac, set, add]), "transfer;set;add");
    }
}
