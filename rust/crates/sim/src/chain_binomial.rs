use crate::{
    compiled_model::CompiledModel,
    config::{ChainBinomialConfig, SimConfig},
    rng::StatefulRng,
    error::{NegativeCountCause, SimError},
    boundary_times::OutputTimes,
    lineage::{DemeId, TransitionId, TransitionObserver},
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
    /// Gamma multipliers actually used during step_one, in source-group order.
    /// Populated by step_one for each overdispersed source group encountered.
    /// Used by PGAS to record the gamma drawn at each substep for transition
    /// density evaluation. Cleared at the start of every `step_one` call —
    /// callers may read it after the call to retrieve the draws from the
    /// most recent step. Pre-clearing by the caller is redundant but harmless.
    pub gamma_used: Vec<f64>,
    /// Reusable buffer for the resolved deltas of one event phase
    /// ([`crate::effects::apply_event_phases`]). Cleared per phase.
    event_deltas: crate::effects::EffectDeltas,
    /// Reusable due-batch for this substep — the interventions/events firing at
    /// `t + dt`. The CALLER populates it before each `step_one`, from the run's
    /// effect plan (`effect_plan::EffectPlan`): the forward driver and the PGAS
    /// producers by substep index, the inference filters at each landing.
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
    model.validate_schedule(cfg.dt, params, cfg.t_start, crate::effect_plan::FireClock::Grid { dt: cfg.dt })?;

    // gh#122: reject a source that mixes a deterministic exit with another exit
    // BEFORE any stepping — the chain-binomial competing-risk draw would
    // over-draw the source (deterministic + stochastic flows are capped
    // independently). Forward chain-binomial chokepoint; the inference producer
    // is gated separately (fit::methods::check_model_capabilities + pfilter).
    model.validate_deterministic_source_exits()?;

    // gh#121: reject a multi-source stochastic transition (`A + B --> C`) BEFORE
    // any stepping — chain-binomial bounds the drawn flow by only the FIRST
    // source, so a secondary source can be driven negative (silently in mild
    // regimes; as a cryptic NegativeCount otherwise). gillespie/ode apply the
    // multi-source firing correctly. Same chokepoint pattern as gh#122; the
    // inference producer is gated in check_model_capabilities + pfilter.
    model.validate_single_source_transitions()?;

    // Refuse a transition that removes more than one member of its source per
    // firing (`S + S --> D`), before any stepping: the Euler-multinomial draws
    // how many individuals leave, one at a time, so it has no multi-unit
    // meaning. Gillespie and the ODE run such a transition.
    model.validate_unit_source_consumption()?;

    // gh#125: chain_binomial is the SNAP policy — it steps a full `dt` per
    // substep and records outputs at grid times (`t_start + k*dt`); it never
    // lands on a sub-`dt` output time. An off-grid output time would therefore
    // be stamped with the POST-step state under an earlier label (silent-wrong:
    // the snapshot for `t=0.5` at `dt=1` would carry the state at `t=1.0`).
    // Reject a misaligned output time with a located error. ODE/Gillespie use
    // the EXACT policy — they clip exactly to each output time and record the
    // true state — so this guard is deliberately chain_binomial-only.
    //
    // Only the fresh forward path checks this: a resume (gh#322 start-from-state)
    // reuses a model that already passed this guard on its forward run — a model
    // with sub-dt output cannot forward-simulate on chain_binomial, so it cannot
    // be resumed either — and an off-grid resume time `T*` is rejected separately
    // by the output-cursor re-seat check below.
    if resume.start.is_none() {
        for &ot in &crate::output::output_times(
            &model.model.output.times, model.model.simulation.t_end)
        {
            let k = ((ot - cfg.t_start) / cfg.dt).round();
            let grid = cfg.t_start + k * cfg.dt;
            // gh#125 review: scale the tolerance by the time magnitude. `output_times`
            // enumerates by accumulation (`t += step`), so for a `dt` not exactly
            // representable in binary (0.1, 0.2, …) the accumulated `ot` drifts from
            // this freshly-computed `grid` by an amount that GROWS with `t` — an
            // absolute epsilon false-rejects a perfectly on-grid model at a long
            // horizon (output-every == dt: the drift crosses 1e-12 near t≈93 at
            // dt=0.1). The drift is O(t·ε); a genuine sub-dt misalignment is O(dt) ≫
            // that, so the scaled tolerance stays strict against real misalignment.
            if (grid - ot).abs() > crate::schedule::output_tol(ot) {
                return Err(SimError::Validation(format!(
                    "chain_binomial: output time {ot} is not on the dt grid \
                     (t_start={} + k·dt, dt={}); the nearest grid time is {grid}. \
                     The chain-binomial (Snap) backend records the post-step state \
                     at grid times only, so a sub-dt output time would be stamped \
                     with the wrong state. Make output times whole multiples of dt \
                     above t_start, or use the ode/gillespie backend (which clips \
                     exactly to each output time).",
                    cfg.t_start, cfg.dt)));
            }
        }
    }

    // gh#322 start-from-state seam: restore the injected RNG when one is supplied
    // (splice-invariant test → byte-identical tail); otherwise seed fresh from
    // `seed` (a normal run, or a contrast arm re-rolling its forward noise from
    // the fork). `resume.start = None` falls through to the fresh seed unchanged.
    // Constructed BEFORE the initial state because building the initial state is
    // itself a draw from this stream (`initial_state_draw`); construction
    // consumes no randomness, so the stream the loop below sees is unchanged.
    let mut rng = match resume.start.and_then(|ss| ss.rng.as_ref()) {
        Some(r) => r.clone(),
        None => StatefulRng::new(seed),
    };

    // gh#322 start-from-state seam: seed the compartment state from the injected
    // fork state when resuming; otherwise draw it from the model. `None` (every
    // existing path) is byte-identical.
    let (mut int_s, mut real_s) = match resume.start {
        Some(ss) => (ss.int_s.clone(), ss.real_s.clone()),
        None => model.initial_state_draw(params, &mut rng)?,
    };
    let n_transitions = model.model.transitions.len();
    let n_real = real_s.values.len();

    // The effect plan: chain-binomial is discrete-time, so every effect fires at
    // the end of a step of the run's own grid `t_start + k·dt` (`fire_step`),
    // built once per run at `params` (parametric `at [...]` schedules resolve
    // here, gh#69). A resumed run (gh#322) starts from a saved, already
    // post-effect state, so its origin batch is empty.
    let origin = if resume.start.is_some() {
        crate::effect_plan::Origin::Resumed
    } else {
        crate::effect_plan::Origin::Init
    };
    let plan = crate::effect_plan::EffectPlan::build(
        model, params, cfg.t_start, cfg.t_end,
        crate::effect_plan::FireClock::Grid { dt: cfg.dt }, origin,
    )?;

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
    // times; effects fire inside step_one from the plan's batch for the substep,
    // so nothing here reads the schedule's effect cursor.
    let schedule = Schedule::snap_forward(
        cfg.dt,
        cfg.t_end,
        OutputTimes::from_model(model)?,
        plan.effect_times(),
    );
    let mut cursor = Cursor::default();

    // gh#322 start-from-state seam: re-seat the OUTPUT cursor to the T* boundary
    // and validate that T* lands on the output grid.
    if resume.start.is_some() {
        // Advance the output cursor past every output time STRICTLY before T*
        // via the gh#233 boundary authority — no hand-rolled loop, no parallel
        // accessor. `until = T* - 2·output_tol(T*)` drains outputs `<= T* - output_tol(T*)`,
        // leaving the cursor pointing AT the T* output boundary, so the initial-row
        // emit below fires for T* with zeroed flows (post-fork incidence starts at
        // 0 — no extra flow-zeroing needed). The effect/obs cursors are NOT
        // re-seated: the Snap path reads neither (the effect plan is built on
        // this run's own window from T* = cfg.t_start; obs is inference-only).
        schedule.drain_outputs(
            &mut cursor,
            cfg.t_start - 2.0 * crate::schedule::output_tol(cfg.t_start),
            |_| {},
        );
        // T* MUST coincide with an output-emit time: flow accumulators reset only
        // at output emits, so the spliced tail matches the continuous tail only
        // when T* is itself an emit. Reject an off-grid T* with a located error —
        // never a silent snap to a neighbour.
        if !schedule.output_due_at(&cursor, cfg.t_start) {
            let t_star = cfg.t_start;
            let outs = crate::output::output_times(
                &model.model.output.times, model.model.simulation.t_end);
            let i = outs.partition_point(|&o| o < t_star);
            let near = |o: Option<&f64>| o.map_or_else(|| "none".to_string(), |v| v.to_string());
            return Err(SimError::Validation(format!(
                "start-from-state resume: the resume time T* = {t_star} is not one of \
                 this run's output times (the nearest are {} before and {} after). \
                 Flow accumulators reset only at an output, so a resumed run must \
                 start on one: restart from a state saved at an output time.",
                near(i.checked_sub(1).and_then(|j| outs.get(j))),
                near(outs.get(i)),
            )));
        }
    }

    // D1: effects at t_start apply to the starting state before the t_start row
    // is recorded (§13.10 without the transition stage: inflow events, drains
    // and `set`, interventions, balance, the non-negativity check). A no-op
    // when the plan has no origin batch, so a `balance {}` model's t_start row
    // is untouched.
    crate::effects::apply_origin_batch_discrete(
        model, &plan, &mut int_s, &mut real_s, params, cfg.t_start, cfg.dt, model.balance.as_ref(),
    )?;

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
        // The plan's batch for this substep: the effects whose grid key is
        // k = s + 1, the step ending at t_grid + dt.
        crate::effects::split_due_batch(model, plan.batch_ending_substep(s), &mut scratch.effect_batch);
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
                    obs.on_fired(TransitionId(tr_idx), DemeId(0), count, t_grid, pre_int, pre_real, params)?;
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

        // No effect-cursor advance here (gh#233 task 4): chain is Snap and fires
        // effects INSIDE step_one from the plan's per-substep batch above;
        // nothing on the Snap path reads the schedule's effect cursor.

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

/// The first integer compartment of `counts` below zero, other than the
/// balance target `bal_idx`, with its count.
fn negative_count(counts: &[i64], bal_idx: Option<usize>) -> Option<(usize, i64)> {
    counts
        .iter()
        .enumerate()
        .find(|&(i, &c)| c < 0 && Some(i) != bal_idx)
        .map(|(i, &c)| (i, c))
}

/// [`negative_count`] as a `NegativeCount` with `cause`; `Ok` when none.
fn first_negative(
    model: &CompiledModel,
    counts: &[i64],
    bal_idx: Option<usize>,
    t: f64,
    cause: NegativeCountCause,
) -> Result<(), SimError> {
    match negative_count(counts, bal_idx) {
        None => Ok(()),
        Some((i, c)) => Err(SimError::NegativeCount {
            compartment: model.int_compartment_name(i),
            attempted_value: c,
            t,
            cause,
        }),
    }
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
/// shared step (gh#216): every caller fills the batch from the run's
/// [`EffectPlan`](crate::effect_plan::EffectPlan), split by kind via
/// [`crate::effects::split_due_batch`] — the forward driver and the PGAS
/// producers by substep index, the inference filters at each landing.
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
                ir::transition::DrawMethod::Overdispersed { .. } => {
                    let mut sigma_sq =
                        eval_resolved(model.resolved.overdispersion[i].as_ref().unwrap(), &ctx);
                    // gh#517: the RATE is guaranteed finite — `eval_propensities`
                    // above guards every transition's propensity and errors by
                    // default. The overdispersion is not: it is resolved here by
                    // a bare `eval_resolved`, which has no error channel.
                    //
                    // A non-finite sigma^2 does not fail loudly downstream, it
                    // fails silently and in the wrong direction. Both consumers
                    // treat `sigma_sq <= 0.0` as a legitimate, COUNTED "no
                    // overdispersion" — but `NaN <= 0.0` is false, so a NaN
                    // slips past that arm into `Gamma::new`'s `Err(_) => 1.0`
                    // fallback (`rng.rs`), which is uncounted. The run then
                    // continues with the noise model switched off and reports a
                    // posterior for a model the user did not specify.
                    //
                    // Same policy as the rate, so the two cannot disagree about
                    // the same NaN: coerce under `--allow-degenerate-rates`,
                    // hard-error by default. Coercing to 0.0 lands on the
                    // documented "no overdispersion" path rather than inventing
                    // a value, and `neg_binomial` counts it.
                    if !sigma_sq.is_finite() {
                        if crate::eval_stats::allow_degenerate_rates() {
                            sigma_sq = 0.0;
                        } else {
                            return Err(SimError::NumericalCollapse {
                                kind: crate::error::CollapseKind::UnOpNan,
                                t,
                            });
                        }
                    }
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
            if let ResolvedDraw::Deterministic = &scratch.draws[tr_idx] {
                // gh#122: a sole-exit deterministic source transition FIRES
                // `round(rate*dt)`, capped by source availability for
                // conservation — the same `round(rate*dt)` convention as the
                // source-LESS deterministic path below. A source that MIXES a
                // deterministic exit with another exit is rejected upstream
                // (`validate_deterministic_source_exits`), so a deterministic
                // member here is always the group's only exit; it is therefore
                // NOT pushed to `probs`/`total_rate` and the group's
                // competing-risk draw (Step 2/3) does not run. This branch
                // consumes NO rng, so a source group with no deterministic
                // member is byte-identical to before it existed (CRN preserved:
                // every draw is untouched here).
                let count = ((rate * dt).round() as i64).clamp(0, n_src) as u64;
                for &(local, delta) in &model.transition_stoich[tr_idx] {
                    scratch.pending_deltas.push((local, delta * count as i64));
                }
                flows[tr_idx] += count;
                scratch.handled[tr_idx] = true;
                continue;
            }
            let effective = match &scratch.draws[tr_idx] {
                ResolvedDraw::Overdispersed(sigma_sq) => {
                    let g = rng.gamma_multiplier(*sigma_sq, dt);
                    scratch.gamma_used.push(g);
                    per_capita * g
                }
                _ => per_capita,
            };
            total_rate += effective;
            scratch.probs.push((tr_idx, effective));
        }

        if total_rate <= 0.0 || scratch.probs.is_empty() {
            // Every competing exit has zero effective hazard this step (each
            // overdispersed exit drew a gamma multiplier of exactly 0), so no
            // member leaves. Mark them handled: left unhandled they would fall
            // through to the source-less draw below, which is not bounded by
            // the source's count.
            for &(tr_idx, _) in &scratch.probs { scratch.handled[tr_idx] = true; }
            continue;
        }

        // Step 2: draw total exits (pomp's first rbinom)
        // gh#audit-H3: stable (p, q) primitive (q discarded here).
        let (p_total, _q) = crate::inference::numerics::prob_q_from_rate_dt(total_rate, dt);
        let p_total = p_total.clamp(0.0, 1.0);
        let mut n_events = rng.binomial(n_src as u64, p_total);

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
    // by the caller from the run's effect plan. Consumed by PROPOSE (event_idx)
    // here and INTERVENE (intervention_idx) below — `step_one` never re-derives
    // due-ness.

    // §13.10 stages 1-3 through the shared seam (gh#985): the INFLOW phase
    // (`add`) resolves against the start-of-step snapshot; the advance hook adds
    // the transition deltas drawn above from that same snapshot; the DRAIN-AND-
    // `set` phase resolves against the resulting post-transition state (gh#217:
    // a draining event on a transition's source moves a share of what survived
    // the interval, and cannot overdraw).
    //
    // The arena is `scratch.int_s` / `scratch.real_s`, which hold the
    // start-of-step snapshot on entry (copied at the top of this function;
    // `counts` is untouched since). The inflow and transition deltas are both
    // computed from the snapshot and summed into the state before the drain
    // phase reads it, as when the inflow deltas rode in `pending_deltas`;
    // integer addition makes the two orders of that sum identical.
    //
    // A compartment negative after the transition draws is reported here,
    // before the drain phase can touch it; a compartment the effect stages
    // leave negative is reported by the scan below, as `InterventionNegative`.
    //
    // The transition draws cannot make a compartment negative: each source's
    // exits are bounded by its count, each exit removes one member of one
    // source (`validate_single_source_transitions`,
    // `validate_unit_source_consumption`), and every other delta is an inflow.
    // So a negative here was already negative when the step began — an
    // `init {}` that evaluates below zero at these parameters (`InitialState`,
    // which depends on θ alone and kills the particle under inference) or that
    // a random initial law drew below zero (`InitialStateDrawn`, which stops
    // the run) — or it is a simulator bug (`SimulatorInvariant`, which stops
    // the run and is never a −∞ weight).
    let bal_idx = model.balance.as_ref().map(|b| b.local_int_idx);
    {
        let pending = &scratch.pending_deltas;
        let at_step_start: &[i64] = counts;
        let mut arena = crate::effects::DiscreteArena {
            int: &mut scratch.int_s,
            real: &mut scratch.real_s,
            deltas: &mut scratch.event_deltas,
        };
        let mut advance = |a: &mut crate::effects::DiscreteArena<'_>| {
            for &(local, delta) in pending {
                a.int.counts[local] += delta;
            }
            match negative_count(&a.int.counts, bal_idx) {
                None => Ok(()),
                Some((i, _)) if at_step_start[i] < 0 => Err(SimError::NegativeCount {
                    compartment: model.int_compartment_name(i),
                    attempted_value: at_step_start[i],
                    t: t + dt,
                    cause: model.negative_initial_state_cause(i),
                }),
                Some((i, c)) => Err(SimError::SimulatorInvariant {
                    what: format!(
                        "the transition stage left compartment '{}' at {c} (it held {} \
                         when the step began); no transition may remove more than its \
                         source holds",
                        model.int_compartment_name(i),
                        at_step_start[i],
                    ),
                    t: t + dt,
                }),
            }
        };
        crate::effects::apply_event_phases(
            model, &scratch.effect_batch.event_idx, &mut arena, params, t + dt, dt,
            Some(&mut advance),
        )?;
    }
    counts.copy_from_slice(&scratch.int_s.counts);

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

    // A compartment the inflow or drain-and-`set` event phase left negative —
    // a `set` to a negative value, say. Coincident drains that could overdraw
    // are refused before they apply (`effects::refuse_overdrawing_drains`), and
    // the transition draws were scanned in the advance hook above, so what
    // reaches here is a configuration error: `InterventionNegative`, as on the
    // ODE and Gillespie.
    // The balance target is exempt: its negativity is a separate signal
    // (constraint-expression-yielded-negative) handled by the balance stage.
    first_negative(model, counts, bal_idx, t + dt, NegativeCountCause::InterventionNegative)?;

    // INTERVENE (stage 3) then BALANCE (stage 4) on the current post-advance
    // state, in fixed canonical order. `scratch.int_s` starts == `counts`, so
    // the intervention reads the post-advance state and the balance reads the
    // post-intervention state — byte-identical to the prior inline blocks.
    scratch.int_s.counts.copy_from_slice(counts);
    crate::lifecycle::apply_post_advance(
        model, &scratch.effect_batch.intervention_idx, &mut scratch.int_s,
        &mut scratch.real_s, params, t + dt, dt, model.balance.as_ref(),
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
