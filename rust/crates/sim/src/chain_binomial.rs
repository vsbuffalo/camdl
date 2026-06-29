use crate::{
    compiled_model::CompiledModel,
    config::{ChainBinomialConfig, SimConfig},
    rng::StatefulRng,
    error::SimError,
    boundary_times::{EffectTimes, OutputTimes},
    lineage::TransitionObserver,
    ode_integrator::rk4_step,
    propensity::{eval_propensities, EvalCtx},
    resolved_expr::eval_resolved,
    schedule::{Cursor, Schedule, MIN_STEP_EPS},
    simulate::Simulate,
    state::{Flows, FlowVec, IntState, RealState, Snapshot, Trajectory},
};

pub struct ChainBinomialSim;

/// Injected mid-run state for the start-from-state seam (gh#322). The resume
/// time is `cfg.t_start` (the caller sets it to T*); this carries the compartment
/// state to seed there.
pub struct StartState {
    pub int_s: IntState,
    pub real_s: RealState,
    /// `Some` → restore this RNG: the splice-invariant test feeds the head run's
    /// final RNG here, so the resumed tail is byte-identical to the continuous
    /// tail. `None` → seed fresh from `seed`: a contrast arm re-rolls its own
    /// forward noise from the fork (CRN desyncs by design — see the contrasts
    /// doc). The RNG-restore path is test-only.
    pub rng: Option<StatefulRng>,
}

/// Resume controls for [`run_chain_binomial_with_observer`]. `Resume::default()`
/// is "no resume" — every existing path passes it and is byte-identical to today.
#[derive(Default)]
pub struct Resume<'a> {
    /// The injected fork state, or `None` for a normal run from `initial_state`.
    pub start: Option<&'a StartState>,
    /// Splice-invariant test capture: if set, the run writes its FINAL RNG here
    /// (so a head run can hand its RNG to the resumed tail).
    pub capture_final_rng: Option<&'a mut StatefulRng>,
}

/// Minimum rate threshold: transitions with rate ≤ this are treated as
/// zero-rate (no draws, zero flow required). Used by both `step_one` and
/// `log_transition_density_substep` — must be identical to avoid simulation/
/// density mismatch.
pub const RATE_EPSILON: f64 = 1e-15;

/// Pre-allocated scratch buffers for `step_one`, eliminating per-call heap
/// allocations. Allocate one per particle (or per thread) and reuse across
/// all time steps.
pub struct StepScratch {
    int_s: IntState,
    real_s: RealState,
    propensities: Vec<f64>,
    draws: Vec<ResolvedDraw>,
    pending_deltas: Vec<(usize, i64)>,
    handled: Vec<bool>,
    probs: Vec<(usize, f64)>,
    /// When set, overrides the next `gamma_multiplier()` call in `step_one`.
    /// Used by correlated pseudo-marginal MCMC to inject pre-drawn Gamma
    /// noise for correlation across MCMC steps.
    pub gamma_override: Option<f64>,
    /// When non-empty, provides standard normal z-values for the total-exit
    /// binomial draw in each source group. `step_one` transforms z to a
    /// binomial count via normal approximation (large np) or inverse CDF
    /// (small np). Consumed in source-group order.
    /// Used by CPM-MCMC for correlated binomial draws.
    pub binomial_z_values: Vec<f64>,
    /// Current index into binomial_z_values. Incremented as z-values are consumed.
    pub binomial_z_idx: usize,
    /// Gamma multipliers actually used during step_one, in source-group order.
    /// Populated by step_one for each overdispersed source group encountered.
    /// Used by PGAS to record the gamma drawn at each substep for transition
    /// density evaluation. Cleared at the start of every `step_one` call —
    /// callers may read it after the call to retrieve the draws from the
    /// most recent step. Pre-clearing by the caller is redundant but harmless.
    pub gamma_used: Vec<f64>,
    /// Reusable buffer for resolved always-active event deltas (PROPOSE stage).
    /// `int` deltas are fused into the draw via `pending_deltas`; `real` deltas
    /// apply to the real reservoir. Cleared per `step_one`.
    event_deltas: crate::effects::EffectDeltas,
    /// Reusable due-batch for this substep — the interventions/events firing at
    /// `t + dt`. The CALLER populates it before each `step_one` (gh#216): the
    /// Snap-forward driver via the `round(t/dt)` key, the Exact-inference callers
    /// via cursor-keyed scheduled interventions plus `grid_dt`-keyed events.
    /// `step_one` consumes it at the PROPOSE (event_idx) and INTERVENE
    /// (intervention_idx) stages. Reused across substeps (one batch per particle)
    /// — no per-step allocation on the inference hot path.
    pub effect_batch: crate::schedule::EffectBatch,
}

/// How event counts are drawn — resolved from the IR at step start.
enum ResolvedDraw { Poisson, Deterministic, Overdispersed(f64) }

impl StepScratch {
    /// Create scratch buffers sized for `model`.
    pub fn new(model: &CompiledModel) -> Self {
        let n_int = model.int_local_to_global.len();
        let n_real = model.real_local_to_global.len();
        let n_tr = model.model.transitions.len();
        StepScratch {
            int_s: IntState::new(n_int),
            real_s: RealState::new(n_real),
            propensities: Vec::with_capacity(n_tr),
            draws: Vec::with_capacity(n_tr),
            pending_deltas: Vec::with_capacity(n_tr * 2),
            handled: vec![false; n_tr],
            gamma_override: None,
            binomial_z_values: Vec::new(),
            binomial_z_idx: 0,
            gamma_used: Vec::new(),
            event_deltas: crate::effects::EffectDeltas::default(),
            effect_batch: crate::schedule::EffectBatch::default(),
            probs: Vec::with_capacity(n_tr),
        }
    }
}

impl Simulate for ChainBinomialSim {
    fn run(
        &self,
        model: &CompiledModel,
        params: &[f64],
        seed: u64,
        config: &SimConfig,
    ) -> Result<Trajectory, SimError> {
        let cfg = match config {
            SimConfig::ChainBinomial(c) => c,
            _ => return Err(SimError::ConfigMismatch {
                expected: "ChainBinomial",
                got: config.variant_name(),
            }),
        };
        // gh#272 LICM: `run_chain_binomial_with_observer` stages the per-eval
        // prologue once for this θ-stable run and lends it into every `step_one`.
        run_chain_binomial(model, params, seed, cfg)
    }

    fn capabilities(&self) -> crate::Capabilities {
        crate::Capabilities::OVERDISPERSION
            | crate::Capabilities::REAL_COMPARTMENTS
            | crate::Capabilities::BALANCE  // gh#audit-C3
            | crate::Capabilities::LINEAGES
            // RUNTIME_DT: StepClock substeps feed the realized `dt` into
            // `EvalCtx.dt`, so a `dt`-referencing rate is meaningful here
            // (see `gate_dt_rate_exact_clip.rs`). gh#54.
            | crate::Capabilities::RUNTIME_DT
            // gh#204 PR2: forward chain-binomial runs the reactive agenda (the
            // realized-obs trigger + due-batch firing). FORWARD only — the
            // inference table in `fit/methods.rs::check_model_capabilities` still
            // withholds it (no reactive-aware filter yet), and gillespie/ode
            // forward still reject (PR3).
            | crate::Capabilities::REACTIVE_INTERVENTIONS
    }

    fn name(&self) -> &'static str { "chain_binomial" }
}

pub fn run_chain_binomial(
    model: &CompiledModel,
    params: &[f64],
    seed: u64,
    cfg: &ChainBinomialConfig,
) -> Result<Trajectory, SimError> {
    run_chain_binomial_with_observer(model, params, seed, cfg, None, None, Resume::default())
}

/// Chain-binomial run with an optional [`TransitionObserver`] (individual-
/// sampling layer, Phase 3). `observer = None` reproduces
/// [`run_chain_binomial`] byte-for-byte: the observer reads its own RNG stream
/// and is fed each transition's per-step flow count *after* `step_one` has
/// drawn it, against the **start-of-step** state captured before `step_one`
/// mutated the counts. This is the trajectory-invariance invariant (Tier 2a).
/// Chain-binomial fires `k` events per transition per step
/// against frozen start-of-step rates, so the observer samples parents from a
/// frozen pool snapshot — the `dt`-bias the diagnostic measures.
pub fn run_chain_binomial_with_observer(
    model: &CompiledModel,
    params: &[f64],
    seed: u64,
    cfg: &ChainBinomialConfig,
    mut observer: Option<&mut dyn TransitionObserver>,
    // Per-timestep progress tick: called once at the top of each step with the
    // current time `t`, BEFORE any RNG is drawn. Read-only and RNG-free, so
    // `None` and `Some(..)` produce byte-identical trajectories. See
    // tests/progress_tick_invariance.rs.
    mut tick: Option<&mut dyn FnMut(f64)>,
    // gh#322 start-from-state seam: when `resume.start` is `Some`, resume from an
    // injected `(state)` at `cfg.t_start` (= T*) instead of building the initial
    // state from the model. `Resume::default()` (every existing path) is
    // byte-identical to today.
    resume: Resume<'_>,
) -> Result<Trajectory, SimError> {
    // gh#126: reject a non-finite/non-positive dt or a non-finite fire
    // time at the entry point — a RELEASE-build check (the per-conversion
    // guards in `time.rs` are debug_assert only). A bad dt would otherwise
    // freeze the substep loop at the initial state (`dt <= 1e-15` break
    // fires immediately) or feed NaN/±∞ straight into the kernel.
    model.validate_schedule(cfg.dt, params)?;

    // gh#322 start-from-state seam: seed the compartment state from the injected
    // fork state when resuming; otherwise build it from the model. `None` (every
    // existing path) is byte-identical.
    let (mut int_s, mut real_s) = match resume.start {
        Some(ss) => (ss.int_s.clone(), ss.real_s.clone()),
        None => model.initial_state(params)?,
    };
    let n_transitions = model.model.transitions.len();
    let n_real = real_s.values.len();

    // gh#53: fire-step indices depend on the runtime integrator's dt
    // (not the compile-time `model.simulation.dt`). Resolve once per
    // sim run from the dt-invariant `fire_times` on CompiledModel.
    // gh#69: also threads `params` for parametric `at [...]` schedules.
    let fire_steps = model.resolve_fire_steps(cfg.dt, params);

    // gh#322 start-from-state seam: restore the injected RNG when one is supplied
    // (splice-invariant test → byte-identical tail); otherwise seed fresh from
    // `seed` (a normal run, or a contrast arm re-rolling its forward noise from
    // the fork). `resume.start = None` falls through to the fresh seed unchanged.
    let mut rng = match resume.start.and_then(|ss| ss.rng.as_ref()) {
        Some(r) => r.clone(),
        None => StatefulRng::new(seed),
    };
    let mut scratch = StepScratch::new(model);
    let mut flows = vec![0u64; n_transitions];

    // gh#272 LICM: stage the per-eval prologue ONCE for this θ-stable run. `params`
    // is fixed for the whole run, so the param/table-only `per_eval_bindings` are
    // evaluated here and lent into every `step_one` rate eval — not recomputed per
    // step. Owned here and passed as data (no shared cache to alias). `None` for
    // models without per-eval bindings (`PerEvalRef` falls through to on-demand).
    // `t`/`dt` are inert (a per-eval body reads no `Time`/`Dt`).
    let per_eval_scratch =
        crate::resolved_expr::stage_per_eval(model, params, cfg.t_start, cfg.dt);
    let per_eval = per_eval_scratch.as_deref();

    // Merged timeline spine. chain_binomial is the SNAP policy: it steps a full
    // dt every substep (never clipped to a boundary) and emits outputs at grid
    // times; interventions fire inside step_one (keyed on fire_steps), so the
    // schedule reports only output boundaries and the effect cursor advances with
    // chain_binomial's own cfg.dt*0.5 snap tolerance, not the schedule's.
    let schedule = Schedule::snap_forward(
        cfg.dt,
        cfg.t_end,
        OutputTimes::from_model(model)?,
        EffectTimes::from_model(model, params)?,
    );
    let mut cursor = Cursor::default();

    // gh#322 start-from-state seam: re-seat the OUTPUT cursor to the T* boundary
    // and validate that T* lands on the output grid.
    if resume.start.is_some() {
        // Advance the output cursor past every output time STRICTLY before T*
        // via the gh#233 boundary authority — no hand-rolled loop, no parallel
        // accessor. `until = T* - 2·OUTPUT_EPS` drains outputs `<= T* - OUTPUT_EPS`,
        // leaving the cursor pointing AT the T* output boundary, so the initial-row
        // emit below fires for T* with zeroed flows (post-fork incidence starts at
        // 0 — no extra flow-zeroing needed). The effect/obs cursors are NOT
        // re-seated: the Snap path reads neither (fire_steps are absolute and
        // already correct at T* = cfg.t_start; obs is inference-only).
        schedule.drain_outputs(
            &mut cursor,
            cfg.t_start - 2.0 * crate::schedule::OUTPUT_EPS,
            |_| {},
        );
        // T* MUST coincide with an output-emit time: flow accumulators reset only
        // at output emits, so the spliced tail matches the continuous tail only
        // when T* is itself an emit. Reject an off-grid T* with a located error —
        // never a silent snap to a neighbour.
        if !schedule.output_due_at(&cursor, cfg.t_start) {
            return Err(SimError::Validation(format!(
                "start-from-state resume: the resume time T* must coincide with an \
                 output-emit time (the saved cadence must contain T*); got {}",
                cfg.t_start
            )));
        }
    }

    let mut traj = Trajectory::new();
    let mut current_flows = FlowVec::new(n_transitions);
    let mut t = cfg.t_start;
    // Robust substep clock: rate/forcing is evaluated at t_start + s*dt (s*dt,
    // bit-exact) rather than the accumulated `t`, which drifts at fractional dt.
    // `t` still drives the loop/output bookkeeping unchanged, so time-homogeneous
    // and integer-dt models stay byte-identical; only time-inhomogeneous models
    // at fractional dt shift — and then agree with PGAS, which already uses
    // t_start + s*dt. See docs/dev/proposals/2026-06-05-substep-time-sdt-convention.md.
    let mut s: u64 = 0;

    // Initial-row convention (see `Trajectory` docs): emit the t_start
    // snapshot with zeroed flows before the loop, so `Σ flow == −Δstate`
    // reconciles over the whole path (gh#270).
    if schedule.output_due_at(&cursor, t) {
        traj.push(Snapshot {
            t, int_state: int_s.clone(), real_state: real_s.clone(), flows: Flows::Int(current_flows.counts.clone()),
        });
        current_flows.reset();
        cursor.pass_output();
    }

    // gh#204 PR2: reactive agenda. `None` unless the (post-scenario-filter) model
    // carries an active reactive policy, so non-reactive sims are byte-identical.
    // The realized-obs draws run on a DEDICATED RNG stream (distinct salt off the
    // run seed), so the surveillance trigger never perturbs the dynamics `rng`
    // (paired-seed / CRN preserved; the equivalence oracle stays byte-identical).
    const REACTIVE_OBS_SEED_SALT: u64 = 0x52454143_54564f42; // "REACTVOB"
    let mut agenda =
        crate::reactive::ReactiveAgenda::from_model(model).map_err(SimError::Validation)?;
    let mut obs_rng = StatefulRng::new(seed ^ REACTIVE_OBS_SEED_SALT);

    // gh#322 start-from-state seam: reactive interventions (and attached
    // observers) carry mid-run state — the `ReactiveAgenda`'s `obs_history`,
    // `once`/`cooldown` gating, the `pending` effect heap, partial interval
    // flows, and a second `obs_rng` stream — that an injected `(int_s, real_s,
    // rng)` cannot reconstruct. Reject such a resume at the seam with a located
    // error rather than forking it silently wrong (gh#187-class matrix gap). The
    // `agenda` value is reused here (built once above), not re-derived.
    if resume.start.is_some() && (agenda.is_some() || observer.is_some()) {
        return Err(SimError::Validation(
            "start-from-state resume does not support reactive interventions / \
             attached observers: their mid-run agenda state (observation history, \
             once/cooldown gating, the pending-effect queue, partial interval \
             flows, and the surveillance RNG stream) cannot be reconstructed from \
             an injected (state, rng). Remove the reactive policy / observer, or \
             run a continuous simulation from t_start."
                .to_string(),
        ));
    }

    while t < cfg.t_end {
        // Progress tick: report current time before drawing this step. RNG-free.
        if let Some(cb) = tick.as_deref_mut() { cb(t); }

        // Snap step: dt = cfg.dt.min(t_end - t), the original formula (bit-exact).
        let dt = schedule.substep(&cursor, t).expect("t < t_end inside loop");
        if dt <= MIN_STEP_EPS { break; }
        // Robust grid time for rate/forcing evaluation (drift-free vs `t`).
        let t_grid = schedule.substep_time(cfg.t_start, s);

        // Capture start-of-step state for the lineage observer (before RK4 and
        // step_one mutate it). Only when an observer is attached — zero cost
        // otherwise. The observer is fed this frozen state plus the per-step
        // transition flows that step_one draws, so it cannot perturb the count
        // trajectory (Tier 2a).
        let pre_step: Option<(IntState, RealState)> = observer
            .as_ref()
            .map(|_| (int_s.clone(), real_s.clone()));

        // Euler step for real compartments (before binomial draws)
        if n_real > 0 {
            rk4_step(model, &int_s, &mut real_s, params, t_grid, dt)?;
            real_s.clamp_nonneg();
        }

        // All Euler-multinomial draws, events, interventions, clamping,
        // and balance are done inside step_one. A prior version of this
        // function also called apply_interventions_at here after t += dt,
        // which caused interventions to fire TWICE per scheduled time
        // (once at t_end inside step_one, once at the new t here).
        // See docs/dev/incidents/2026-04-17-chain-binomial-double-fire.md.
        flows.fill(0);
        // Snap policy: decide the due batch on the `round(t/dt)` key at the
        // boundary t_grid + dt (the firing key step_one used internally before
        // gh#216 lifted the decision to the caller). Byte-identical.
        crate::effects::due_effects(model, &fire_steps, t_grid + dt, cfg.dt, &mut scratch.effect_batch);
        // gh#204 hook A: merge reactive effects due at this boundary into the
        // SAME due-batch as scheduled interventions, so they fire through the
        // identical apply_intervention_effects + post-advance + balance +
        // negative-count lifecycle (no fork).
        if let Some(a) = agenda.as_mut() {
            for iv_idx in a.due_iv_idxs(t_grid + dt) {
                scratch.effect_batch.intervention_idx.push(iv_idx);
            }
        }
        step_one(model, &mut int_s.counts, &mut flows, &mut real_s, params, t_grid, dt, per_eval, &mut rng, &mut scratch)?;

        // Lineage observer: feed each transition's per-step flow count against
        // the frozen start-of-step state. step_one has already drawn from the
        // simulation RNG; the observer uses its own stream.
        if let (Some(obs), Some((pre_int, pre_real))) = (observer.as_deref_mut(), pre_step.as_ref()) {
            obs.begin_batch_step();
            for (tr_idx, &count) in flows.iter().enumerate() {
                if count > 0 {
                    obs.on_fired(tr_idx, 0, count, t_grid, pre_int, pre_real, params)?;
                }
            }
            obs.end_batch_step();
        }

        // Accumulate flows into output FlowVec
        for (i, &f) in flows.iter().enumerate() {
            current_flows.add(i, f);
        }
        // gh#204 hook B: accumulate this substep's flows into each reactive obs
        // stream's interval (for incidence triggers).
        if let Some(a) = agenda.as_mut() {
            a.accumulate(&flows);
        }

        t += dt;
        s += 1;

        // No effect-cursor advance here (gh#233 task 4). chain is Snap: it fires
        // effects INSIDE step_one, keyed on round(t/dt) via the `due_effects`
        // batch above — and nothing on the Snap path reads the schedule's effect
        // cursor. The old `iv <= t + cfg.dt*0.5` half-step advance was dead
        // bookkeeping (it wrote `cursor.effect_idx`, which is never read here) on a
        // tolerance that did not even match the firing key; removed. Byte-identical.

        // Output
        schedule.drain_outputs(&mut cursor, t, |ot| {
            traj.push(Snapshot {
                t: ot,
                int_state: int_s.clone(),
                real_state: real_s.clone(),
                flows: Flows::Int(current_flows.counts.clone()),
            });
            current_flows.reset();
        });

        // gh#204 hook C: at an observation emit boundary (post-output, per the
        // lifecycle) draw the realized obs, evaluate the reactive triggers, and
        // enqueue their effects for a later boundary. No-op unless a stream emits
        // at `t`. The enqueued effect (lag ≥ 0) cannot affect this `t` — it is
        // merged in a subsequent iteration's hook A.
        if let Some(a) = agenda.as_mut() {
            a.on_boundary(t, &int_s, &real_s, params, model, &mut obs_rng);
        }
    }

    schedule.drain_outputs(&mut cursor, f64::INFINITY, |ot| {
        traj.push(Snapshot {
            t: ot,
            int_state: int_s.clone(),
            real_state: real_s.clone(),
            flows: Flows::Int(current_flows.counts.clone()),
        });
        current_flows.reset();
    });

    // gh#204: carry the reactive firings out for `reactive_log.tsv`. `Some`
    // (possibly empty) exactly when this run had an active reactive agenda, so
    // the log is a declared artifact whenever reactive is active.
    if let Some(a) = agenda {
        traj.reactive_log = Some(a.into_firings());
    }

    // gh#322 start-from-state seam: hand this run's FINAL RNG to the caller. The
    // splice-invariant test feeds the head run's final RNG into the resumed
    // tail, which is what makes the spliced continuation byte-identical.
    if let Some(out) = resume.capture_final_rng {
        *out = rng.clone();
    }

    Ok(traj)
}

/// Check if step tracing is enabled via CAMDL_TRACE_STEPS=1.
pub fn trace_enabled() -> bool {
    use std::sync::OnceLock;
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var("CAMDL_TRACE_STEPS").is_ok_and(|v| v == "1"))
}

/// Advance integer compartment state by one chain-binomial step.
///
/// This is the core Euler-multinomial step, extracted for use by the
/// particle filter and other inference algorithms. It operates on raw
/// slices to avoid coupling to IntState/FlowVec/ParticleState.
///
/// `dt` is `dt_actual` — the realized substep length the kernel advances:
/// rate evaluation, the transition probability `1 − exp(−rate·dt)`, the
/// overdispersion `shape = dt/σ²`, and the event-amount evaluation all use it.
///
/// `step_one` no longer DECIDES which effects fire — it APPLIES the batch the
/// CALLER has pre-populated in `scratch.effect_batch` (`event_idx` at PROPOSE,
/// `intervention_idx` at INTERVENE). This lifts the firing decision out of the
/// shared step (gh#216): the Snap-forward driver fills the batch via the
/// `round(t/dt)` key ([`crate::effects::due_effects`]); the Exact-inference
/// callers fire EVERY effect — events and scheduled interventions alike —
/// CURSOR-keyed from the timeline's effect boundaries, splitting the boundary's
/// batch by kind via [`crate::effects::split_due_batch`] (no `round(t/dt)` for
/// events; gh#216 events arm).
///
/// `scratch` holds pre-allocated buffers to avoid heap allocation per call.
/// Create one `StepScratch` per particle and reuse across all time steps.
#[allow(clippy::too_many_arguments)]
pub fn step_one(
    model: &CompiledModel,
    counts: &mut [i64],
    flows: &mut [u64],
    real: &mut RealState,
    params: &[f64],
    t: f64,
    dt: f64,
    // gh#272 LICM: the per-eval prologue for this θ-span (param/table-only
    // invariants), staged once by the caller and lent into the rate eval below.
    // Forward chain-binomial stages it once before its step loop; inference
    // producer steps pass `None` (on-demand, byte-identical; per-particle θ means
    // staging here is a Phase 2 wiring, not a correctness requirement).
    per_eval: Option<&[f64]>,
    rng: &mut StatefulRng,
    scratch: &mut StepScratch,
) -> Result<(), SimError> {
    // Copy current counts into scratch IntState for propensity evaluation.
    // This is a memcpy into pre-allocated memory, not a heap allocation.
    scratch.int_s.counts.copy_from_slice(counts);

    // Copy the caller's real compartment state into scratch RealState so the
    // propensity evaluator (and the pre-evaluated draw-method context below)
    // see the *current* real values, not scratch.real_s's zero init. Without
    // this, any integer transition whose rate couples to a real compartment
    // (e.g. cholera SIWR's water-borne infection term `beta_W*W/(W+kappa)`)
    // evaluates that real value as 0 — silently wrong. The caller advances
    // `real` (RK4) before this call; `step_one` reads it here and writes back
    // any real-compartment intervention mutations (apply_post_advance) below.
    // See docs/dev/incidents/2026-06-07-chain-binomial-stale-real-state.md.
    scratch.real_s.values.copy_from_slice(&real.values);

    // Reset per-step output buffer. Without this, overdispersed() models
    // accumulate one f64 per source-group per substep per particle for the
    // entire life of the scratch — an unbounded leak in IF2/PF/PMMH which
    // reuse a single scratch across iterations. See issue #10.
    scratch.gamma_used.clear();

    eval_propensities(model, &scratch.int_s, &scratch.real_s, params, t, dt,
                      per_eval, &mut scratch.propensities)?;

    // Pre-evaluate draw methods from start-of-step state
    scratch.draws.clear();
    {
        let ctx = EvalCtx { model, int_s: &scratch.int_s, real_s: &scratch.real_s, params, t, dt, projected: None, aux: None, int_float_override: None, per_eval };
        for (i, tr) in model.model.transitions.iter().enumerate() {
            scratch.draws.push(match &tr.draw_method {
                ir::transition::DrawMethod::Poisson => ResolvedDraw::Poisson,
                ir::transition::DrawMethod::Deterministic => ResolvedDraw::Deterministic,
                ir::transition::DrawMethod::Overdispersed(_) => {
                    let sigma_sq = eval_resolved(model.resolved.overdispersion[i].as_ref().unwrap(), &ctx);
                    ResolvedDraw::Overdispersed(sigma_sq)
                }
            });
        }
    }

    // ── Deferred state update (see run_chain_binomial for full explanation) ──
    scratch.pending_deltas.clear();
    scratch.handled.fill(false);

    // Euler-multinomial draws for transitions sharing a source compartment.
    //
    // Matches pomp's reulermultinom exactly:
    //   1. Compute effective per-capita rates (with gamma noise if overdispersed)
    //   2. Draw TOTAL exits from Binom(n_src, 1-exp(-sum_rates * dt))
    //   3. Split total exits across transitions proportional to their rates
    //
    // This is NOT equivalent to sequential conditional binomials with
    // individual probabilities p_i = 1-exp(-r_i*dt), because
    // Σ(1-exp(-r_i*dt)) > 1-exp(-Σr_i*dt) (subadditivity of 1-exp).
    // The old algorithm systematically over-counted total exits, causing
    // particle trajectories to drift and ESS to degrade over long runs.
    for &(src_local, ref group) in &model.source_groups {
        let n_src = counts[src_local].max(0);
        if n_src == 0 {
            for &tr_idx in group { scratch.handled[tr_idx] = true; }
            continue;
        }

        // Step 1: compute effective per-capita rates
        scratch.probs.clear(); // reuse as (tr_idx, effective_rate) pairs
        let mut total_rate = 0.0_f64;
        for &tr_idx in group {
            let rate = scratch.propensities[tr_idx];
            if rate <= RATE_EPSILON { scratch.handled[tr_idx] = true; continue; }
            let per_capita = rate / n_src as f64;
            if let ResolvedDraw::Deterministic = &scratch.draws[tr_idx] { scratch.handled[tr_idx] = true; continue; }
            let effective = match &scratch.draws[tr_idx] {
                ResolvedDraw::Overdispersed(sigma_sq) => {
                    let g = scratch.gamma_override.take()
                        .unwrap_or_else(|| rng.gamma_multiplier(*sigma_sq, dt));
                    scratch.gamma_used.push(g);
                    per_capita * g
                }
                _ => per_capita,
            };
            total_rate += effective;
            scratch.probs.push((tr_idx, effective));
        }

        if total_rate <= 0.0 || scratch.probs.is_empty() { continue; }

        // Step 2: draw total exits (pomp's first rbinom)
        // gh#audit-H3: stable (p, q) primitive (q discarded here).
        let (p_total, _q) = crate::inference::numerics::prob_q_from_rate_dt(total_rate, dt);
        let p_total = p_total.clamp(0.0, 1.0);
        let mut n_events = if scratch.binomial_z_idx < scratch.binomial_z_values.len() {
            // CPM: use pre-drawn z-value for correlated binomial
            let z = scratch.binomial_z_values[scratch.binomial_z_idx];
            scratch.binomial_z_idx += 1;
            let n = n_src as u64;
            let np = n as f64 * p_total;
            let nq = n as f64 * (1.0 - p_total);
            if np > 20.0 && nq > 20.0 {
                let sd = (np * (1.0 - p_total)).sqrt();
                (np + sd * z).round().clamp(0.0, n as f64) as u64
            } else if np > 0.0 {
                // Small np: inverse CDF via Phi(z)
                let u = crate::inference::correlated_pf::phi(z).clamp(1e-15, 1.0 - 1e-15);
                crate::inference::correlated_pf::binomial_quantile(n, p_total, u)
            } else {
                0
            }
        } else {
            rng.binomial(n_src as u64, p_total)
        };

        debug_assert!(n_events <= n_src as u64,
            "n_exit ({}) > n_src ({}) at source compartment {}",
            n_events, n_src, src_local);

        // Step 3: split total exits proportional to rates (pomp's inner loop)
        let n_competing = scratch.probs.len();
        let mut rate_remaining = total_rate;
        for (k, &(tr_idx, eff_rate)) in scratch.probs.iter().enumerate() {
            let count = if k == n_competing - 1 {
                // Last category gets the remainder (avoids rounding drift)
                n_events
            } else if n_events > 0 && rate_remaining > 0.0 {
                // pomp: if (rate[k] > p) p = rate[k]; trans[k] = rbinom(size, rate[k]/p)
                let p_split = (eff_rate / rate_remaining).clamp(0.0, 1.0);
                let c = rng.binomial(n_events, p_split);
                n_events -= c;
                rate_remaining -= eff_rate;
                c
            } else {
                0
            };
            for &(local, delta) in &model.transition_stoich[tr_idx] {
                scratch.pending_deltas.push((local, delta * count as i64));
            }
            flows[tr_idx] += count;
            scratch.handled[tr_idx] = true;
        }
    }

    // Inflows and ungrouped transitions
    for (i, &rate) in scratch.propensities.iter().enumerate() {
        if scratch.handled[i] || rate <= RATE_EPSILON { continue; }
        let mean = rate * dt;
        let count = match &scratch.draws[i] {
            ResolvedDraw::Poisson => rng.poisson(mean),
            ResolvedDraw::Deterministic => mean.round() as u64,
            ResolvedDraw::Overdispersed(sigma_sq) => rng.neg_binomial(mean, *sigma_sq, dt),
        };
        for &(local, delta) in &model.transition_stoich[i] {
            scratch.pending_deltas.push((local, delta * count as i64));
        }
        flows[i] += count;
    }

    // The due batch for this substep was pre-populated in `scratch.effect_batch`
    // by the caller (gh#216): the Snap-forward driver via the `round(t/dt)` key,
    // the Exact-inference callers via cursor-keyed scheduled interventions plus
    // `grid_dt`-keyed events. Consumed by PROPOSE (event_idx) here and INTERVENE
    // (intervention_idx) below — `step_one` never re-derives due-ness.

    // PROPOSE (stage 1): INFLOW event deltas (`Add`) from the start-of-step
    // snapshot (`scratch.int_s`/`scratch.real_s`, captured at the top of this
    // function before any draws). The integer deltas are fused into ADVANCE —
    // applied atomically with the transition deltas below; the real deltas apply
    // to the snapshot reservoir, which is written back to `real` at the end.
    //
    // gh#217: only the SNAPSHOT phase (inflow `Add`) fires here. Draining
    // transfers and `Set` (the RESIDUAL phase) are resolved AFTER the atomic
    // apply, against the post-transition residual — see below. This keeps the
    // inflow path byte-identical (cohort births fuse with the draw) while a
    // draining event on a transition's source reads what SURVIVED the interval
    // (matching ODE/Gillespie), instead of subtracting the full snapshot a second
    // time and overshooting to a negative count.
    scratch.event_deltas.clear();
    crate::effects::resolve_event_batch(
        model, &scratch.effect_batch.event_idx, &scratch.int_s, &scratch.real_s,
        params, t + dt, dt, crate::effects::EventPhase::Snapshot, &mut scratch.event_deltas,
    )?;
    for d in &scratch.event_deltas.int {
        scratch.pending_deltas.push((d.idx, d.delta));
    }
    for d in &scratch.event_deltas.real {
        scratch.real_s.values[d.idx] += d.delta;
    }

    // ADVANCE: apply all snapshot-phase deltas atomically (transitions + inflow
    // events). `counts` now holds the POST-TRANSITION residual state.
    for &(local, delta) in &scratch.pending_deltas {
        counts[local] += delta;
    }

    // RESIDUAL phase (gh#217): draining transfers (`from` side) and `Set` resolve
    // against the post-transition residual `counts` / `scratch.real_s`, then apply
    // to them. `fraction` × residual `from`; `count`.min(residual `from`); `Set`
    // overwrites the post-dynamics value. `scratch.int_s` is repurposed here as
    // the residual int read-state (it held the start-of-step snapshot, now stale);
    // `apply_post_advance` below re-syncs it to `counts` before INTERVENE.
    if !scratch.effect_batch.event_idx.is_empty() {
        scratch.int_s.counts.copy_from_slice(counts);
        scratch.event_deltas.clear();
        crate::effects::resolve_event_batch(
            model, &scratch.effect_batch.event_idx, &scratch.int_s, &scratch.real_s,
            params, t + dt, dt, crate::effects::EventPhase::Residual, &mut scratch.event_deltas,
        )?;
        for d in &scratch.event_deltas.int {
            counts[d.idx] += d.delta;
        }
        for d in &scratch.event_deltas.real {
            scratch.real_s.values[d.idx] += d.delta;
        }
    }

    // Per-substep trace (CAMDL_TRACE_STEPS=1)
    if trace_enabled() {
        // Header on first call
        use std::sync::OnceLock;
        static HEADER: OnceLock<bool> = OnceLock::new();
        HEADER.get_or_init(|| {
            eprint!("t");
            for c in &model.model.compartments { eprint!("\t{}", c.name); }
            for tr in &model.model.transitions { eprint!("\tflow_{}", tr.name); }
            eprint!("\ttotal_pop");
            for tr in model.model.transitions.iter() {
                eprint!("\trate_{}", tr.name);
            }
            eprintln!();
            true
        });
        eprint!("{:.1}", t + dt);
        for &c in counts.iter() { eprint!("\t{}", c); }
        for &f in flows.iter() { eprint!("\t{}", f); }
        let total: i64 = counts.iter().sum();
        eprint!("\t{}", total);
        for &p in scratch.propensities.iter() { eprint!("\t{:.4}", p); }
        eprintln!();
    }

    // gh#audit-C5 / S2. Negative compartment count after the binomial
    // split → BinomialOvershoot (rate·dt → 1, expected during inference
    // exploration). Previously silently clamped to 0; now returns
    // SimError::NegativeCount{BinomialOvershoot, ...}. Inference layers
    // catch this via SimError::is_per_particle_recoverable() and convert
    // to −Inf for the offending particle; forward sim halts. The balance
    // target is exempted because its negativity is a separate signal
    // (constraint-expression-yielded-negative) handled by the balance
    // block at lines 432-446.
    let bal_idx = model.balance.as_ref().map(|b| b.local_int_idx);
    for (i, c) in counts.iter_mut().enumerate() {
        if Some(i) == bal_idx { continue; }
        if *c < 0 {
            return Err(crate::error::SimError::NegativeCount {
                compartment: model.int_compartment_name(i),
                attempted_value: *c,
                t: t + dt,
                cause: crate::error::NegativeCountCause::BinomialOvershoot,
            });
        }
    }

    // INTERVENE (stage 3) then BALANCE (stage 4) on the current post-advance
    // state, in fixed canonical order. `scratch.int_s` starts == `counts`, so
    // the intervention reads the post-advance state and the balance reads the
    // post-intervention state — byte-identical to the prior inline blocks.
    scratch.int_s.counts.copy_from_slice(counts);
    crate::lifecycle::apply_post_advance(
        model, &scratch.effect_batch.intervention_idx, &mut scratch.int_s,
        &mut scratch.real_s, params, t, dt, model.balance.as_ref(),
    )?;
    counts.copy_from_slice(&scratch.int_s.counts);

    // Write back real-compartment mutations from apply_post_advance (e.g.
    // `set()`/`transfer()`/`add()` interventions targeting a real reservoir).
    // Previously these landed in scratch.real_s and were dropped — the run's
    // `real` never saw them, so real-compartment interventions were silently
    // ignored on the chain-binomial backend. Same incident as the rate-read
    // bug above: scratch.real_s was disconnected from the run's real state.
    real.values.copy_from_slice(&scratch.real_s.values);

    Ok(())
}
