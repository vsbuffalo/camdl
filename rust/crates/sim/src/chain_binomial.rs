use crate::{
    compiled_model::CompiledModel,
    config::{ChainBinomialConfig, SimConfig},
    rng::StatefulRng,
    error::SimError,
    intervention::{all_intervention_times, apply_interventions_at},
    lineage::TransitionObserver,
    ode_integrator::rk4_step,
    output::output_times as get_output_times,
    propensity::{eval_propensities, EvalCtx},
    resolved_expr::eval_resolved,
    simulate::Simulate,
    state::{FlowVec, IntState, RealState, Snapshot, Trajectory},
};

pub struct ChainBinomialSim;

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
        run_chain_binomial(model, params, seed, cfg)
    }

    fn capabilities(&self) -> crate::Capabilities {
        crate::Capabilities::OVERDISPERSION
            | crate::Capabilities::REAL_COMPARTMENTS
            | crate::Capabilities::BALANCE  // gh#audit-C3
            | crate::Capabilities::LINEAGES
    }

    fn name(&self) -> &'static str { "chain_binomial" }
}

pub fn run_chain_binomial(
    model: &CompiledModel,
    params: &[f64],
    seed: u64,
    cfg: &ChainBinomialConfig,
) -> Result<Trajectory, SimError> {
    run_chain_binomial_with_observer(model, params, seed, cfg, None)
}

/// Chain-binomial run with an optional [`TransitionObserver`] (individual-
/// sampling layer, Phase 3). `observer = None` reproduces
/// [`run_chain_binomial`] byte-for-byte: the observer reads its own RNG stream
/// and is fed each transition's per-step flow count *after* `step_one` has
/// drawn it, against the **start-of-step** state captured before `step_one`
/// mutated the counts. This is the trajectory-invariance invariant (Tier 2a).
/// Like tau-leap, chain-binomial fires `k` events per transition per step
/// against frozen start-of-step rates, so the observer samples parents from a
/// frozen pool snapshot — the `dt`-bias the diagnostic measures.
pub fn run_chain_binomial_with_observer(
    model: &CompiledModel,
    params: &[f64],
    seed: u64,
    cfg: &ChainBinomialConfig,
    mut observer: Option<&mut dyn TransitionObserver>,
) -> Result<Trajectory, SimError> {
    let (mut int_s, mut real_s) = model.initial_state(params)?;
    let n_transitions = model.model.transitions.len();
    let n_real = real_s.values.len();

    // gh#53: fire-step indices depend on the runtime integrator's dt
    // (not the compile-time `model.simulation.dt`). Resolve once per
    // sim run from the dt-invariant `fire_times` on CompiledModel.
    let fire_steps = model.resolve_fire_steps(cfg.dt);

    let mut rng = StatefulRng::new(seed);
    let mut scratch = StepScratch::new(model);
    let mut flows = vec![0u64; n_transitions];

    let output_times = get_output_times(&model.model.output.times);
    let mut output_idx = 0;
    let iv_times = all_intervention_times(model);
    let mut iv_idx = 0;

    let mut traj = Trajectory::new();
    let mut current_flows = FlowVec::new(n_transitions);
    let mut t = cfg.t_start;

    if output_idx < output_times.len() && output_times[output_idx] <= t + 1e-12 {
        traj.push(Snapshot {
            t, int_state: int_s.clone(), real_state: real_s.clone(), flows: current_flows.clone(),
        });
        current_flows.reset();
        output_idx += 1;
    }

    while t < cfg.t_end {
        let dt = cfg.dt.min(cfg.t_end - t);
        if dt <= 1e-15 { break; }

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
            rk4_step(model, &int_s, &mut real_s, params, t, dt)?;
            real_s.clamp_nonneg();
        }

        // All Euler-multinomial draws, events, interventions, clamping,
        // and balance are done inside step_one. A prior version of this
        // function also called apply_interventions_at here after t += dt,
        // which caused interventions to fire TWICE per scheduled time
        // (once at t_end inside step_one, once at the new t here).
        // See docs/dev/incidents/2026-04-17-chain-binomial-double-fire.md.
        flows.fill(0);
        step_one(model, &mut int_s.counts, &mut flows, params, t, dt, &mut rng, &mut scratch, &fire_steps)?;

        // Lineage observer: feed each transition's per-step flow count against
        // the frozen start-of-step state. step_one has already drawn from the
        // simulation RNG; the observer uses its own stream.
        if let (Some(obs), Some((pre_int, pre_real))) = (observer.as_deref_mut(), pre_step.as_ref()) {
            obs.begin_batch_step();
            for (tr_idx, &count) in flows.iter().enumerate() {
                if count > 0 {
                    obs.on_fired(tr_idx, 0, count, t, pre_int, pre_real, params)?;
                }
            }
            obs.end_batch_step();
        }

        // Accumulate flows into output FlowVec
        for (i, &f) in flows.iter().enumerate() {
            current_flows.add(i, f);
        }

        t += dt;

        // Bookkeeping: advance iv_idx past any intervention that fired
        // in step_one this step. The firing itself happens in step_one.
        while iv_idx < iv_times.len() && iv_times[iv_idx] <= t + cfg.dt * 0.5 {
            iv_idx += 1;
        }

        // Output
        while output_idx < output_times.len() && output_times[output_idx] <= t + 1e-12 {
            traj.push(Snapshot {
                t: output_times[output_idx],
                int_state: int_s.clone(),
                real_state: real_s.clone(),
                flows: current_flows.clone(),
            });
            current_flows.reset();
            output_idx += 1;
        }
    }

    while output_idx < output_times.len() {
        traj.push(Snapshot {
            t: output_times[output_idx],
            int_state: int_s.clone(),
            real_state: real_s.clone(),
            flows: current_flows.clone(),
        });
        current_flows.reset();
        output_idx += 1;
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
/// `scratch` holds pre-allocated buffers to avoid heap allocation per call.
/// Create one `StepScratch` per particle and reuse across all time steps.
pub fn step_one(
    model: &CompiledModel,
    counts: &mut [i64],
    flows: &mut [u64],
    params: &[f64],
    t: f64,
    dt: f64,
    rng: &mut StatefulRng,
    scratch: &mut StepScratch,
    fire_steps: &[std::collections::BTreeSet<i64>],
) -> Result<(), SimError> {
    // Copy current counts into scratch IntState for propensity evaluation.
    // This is a memcpy into pre-allocated memory, not a heap allocation.
    scratch.int_s.counts.copy_from_slice(counts);

    // Reset per-step output buffer. Without this, overdispersed() models
    // accumulate one f64 per source-group per substep per particle for the
    // entire life of the scratch — an unbounded leak in IF2/PF/PMMH which
    // reuse a single scratch across iterations. See issue #10.
    scratch.gamma_used.clear();

    eval_propensities(model, &scratch.int_s, &scratch.real_s, params, t, dt,
                      &mut scratch.propensities)?;

    // Pre-evaluate draw methods from start-of-step state
    scratch.draws.clear();
    {
        let ctx = EvalCtx { model, int_s: &scratch.int_s, real_s: &scratch.real_s, params, t, dt, projected: None, int_float_override: None };
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

    // Inject always_active event deltas (evaluated from snapshot, applied atomically)
    crate::intervention::inject_event_deltas(
        model, fire_steps, &scratch.int_s, &scratch.real_s, params, t, dt,
        &mut scratch.pending_deltas,
    )?;

    // Apply all deltas atomically (transitions + events)
    for &(local, delta) in &scratch.pending_deltas {
        counts[local] += delta;
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
            // Look up the compartment name by walking comp_index for
            // the matching local int slot. This is O(n) but only
            // executed in the error path.
            let name = model.comp_index.iter()
                .find(|(_, &g)| model.global_to_int.get(g).copied().flatten() == Some(i))
                .map(|(n, _)| n.clone())
                .unwrap_or_else(|| format!("(local-int-{i})"));
            return Err(crate::error::SimError::NegativeCount {
                compartment: name,
                attempted_value: *c,
                t: t + dt,
                cause: crate::error::NegativeCountCause::BinomialOvershoot,
            });
        }
    }

    // Apply interventions that fire at t + dt (within tolerance dt/2).
    if !model.model.interventions.is_empty() {
        let t_end = t + dt;
        scratch.int_s.counts.copy_from_slice(counts);
        let fired = apply_interventions_at(
            t_end, model, fire_steps, dt, &mut scratch.int_s, &mut scratch.real_s, params, dt * 0.5,
        )?;
        if fired {
            counts.copy_from_slice(&scratch.int_s.counts);
        }
    }

    // Apply balance constraint: overwrite target compartment so the population
    // budget holds. All other compartments are finalized at this point.
    if let Some(ref bal) = model.balance {
        scratch.int_s.counts.copy_from_slice(counts);
        let t_end = t + dt;
        let ctx = EvalCtx {
            model, int_s: &scratch.int_s, real_s: &scratch.real_s,
            params, t: t_end, dt, projected: None, int_float_override: None,
        };
        let val = eval_resolved(&bal.expr, &ctx);
        let bal_count = val.round() as i64;
        if bal_count < 0 {
            log::warn!("balance compartment went negative ({}) at t={:.1} — \
                        model may be inconsistent at these parameters", bal_count, t_end);
        }
        counts[bal.local_int_idx] = bal_count;
    }

    Ok(())
}
