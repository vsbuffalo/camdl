use crate::{
    compiled_model::CompiledModel,
    config::{SimConfig, TauLeapConfig},
    rng::StatefulRng,
    error::SimError,
    intervention::{all_intervention_times, apply_interventions_at},
    ode_integrator::rk4_step,
    output::output_times as get_output_times,
    propensity::{eval_propensities, EvalCtx},
    resolved_expr::eval_resolved,
    simulate::Simulate,
    state::{FlowVec, Snapshot, Trajectory},
};

pub struct TauLeapSim;

impl Simulate for TauLeapSim {
    fn run(
        &self,
        model: &CompiledModel,
        params: &[f64],
        seed: u64,
        config: &SimConfig,
    ) -> Result<Trajectory, SimError> {
        let cfg = match config {
            SimConfig::TauLeap(c) => c,
            _ => return Err(SimError::ConfigMismatch {
                expected: "TauLeap",
                got: config.variant_name(),
            }),
        };
        run_tau_leap(model, params, seed, cfg)
    }

    fn capabilities(&self) -> crate::Capabilities {
        crate::Capabilities::OVERDISPERSION
            | crate::Capabilities::REAL_COMPARTMENTS
            | crate::Capabilities::LINEAGES
    }

    fn name(&self) -> &'static str { "tau_leap" }
}

fn run_tau_leap(
    model: &CompiledModel,
    params: &[f64],
    seed: u64,
    cfg: &TauLeapConfig,
) -> Result<Trajectory, SimError> {
    let (mut int_s, mut real_s) = model.initial_state(params)?;
    let n_transitions = model.model.transitions.len();
    let n_real = real_s.values.len();

    // gh#53: resolve fire_steps using the runtime cfg.dt, not the
    // compile-time model.simulation.dt. See chain_binomial.rs for the
    // full explanation.
    let fire_steps = model.resolve_fire_steps(cfg.dt);

    let mut rng = StatefulRng::new(seed);
    let mut propensities = Vec::with_capacity(n_transitions);

    let output_times = get_output_times(&model.model.output.times);
    let mut output_idx = 0;
    let iv_times = all_intervention_times(model);
    let mut iv_idx = 0;

    let mut traj = Trajectory::new();
    let mut current_flows = FlowVec::new(n_transitions);
    let mut t = cfg.t_start;

    // Initial snapshot
    if output_idx < output_times.len() && output_times[output_idx] <= t + 1e-12 {
        traj.push(Snapshot {
            t,
            int_state: int_s.clone(),
            real_state: real_s.clone(),
            flows: current_flows.clone(),
        });
        current_flows.reset();
        output_idx += 1;
    }

    while t < cfg.t_end {
        // Determine actual step (might be truncated by boundary)
        let next_boundary = {
            let out_t = output_times.get(output_idx).copied().unwrap_or(f64::INFINITY);
            let iv_t = iv_times.get(iv_idx).copied().unwrap_or(f64::INFINITY);
            cfg.t_end.min(out_t).min(iv_t)
        };
        let dt = cfg.dt.min(next_boundary - t);
        if dt <= 0.0 {
            // At a boundary — handle it
            // Apply intervention if due
            if iv_times.get(iv_idx).copied().is_some_and(|iv| (iv - t).abs() < 1e-10) {
                apply_interventions_at(t, model, &fire_steps, cfg.dt, &mut int_s, &mut real_s, params, 1e-10)?;
                while iv_idx < iv_times.len() && iv_times[iv_idx] <= t + 1e-10 { iv_idx += 1; }
            }
            // Record output if due
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
            if t >= cfg.t_end { break; }
            continue;
        }

        // Evaluate propensities at current state
        eval_propensities(model, &int_s, &real_s, params, t, cfg.dt, &mut propensities)?;

        // Pre-evaluate draw method for each transition (resolves overdispersion
        // σ² expressions from start-of-step state before any mutations).
        enum ResolvedDraw { Poisson, Deterministic, Overdispersed(f64) }
        let draws: Vec<ResolvedDraw> = {
            let ctx = EvalCtx { model, int_s: &int_s, real_s: &real_s, params, t, dt: cfg.dt, projected: None, int_float_override: None };
            model.model.transitions.iter().enumerate()
                .map(|(i, tr)| match &tr.draw_method {
                    ir::transition::DrawMethod::Poisson => ResolvedDraw::Poisson,
                    ir::transition::DrawMethod::Deterministic => ResolvedDraw::Deterministic,
                    ir::transition::DrawMethod::Overdispersed(_) => {
                        let sigma_sq = eval_resolved(model.resolved.overdispersion[i].as_ref().unwrap(), &ctx);
                        ResolvedDraw::Overdispersed(sigma_sq)
                    }
                })
                .collect()
        };
        // RM1 in 2026-04-19 engine review: for transitions that share
        // a source compartment (competing exits), independent Poisson
        // draws can produce more total exits than the source has
        // individuals, silently violating population conservation via
        // clamp_nonneg. Match chain-binomial's Euler-multinomial:
        //  1. Draw total exits from Binomial(n_src, 1-exp(-Σr_k·dt)).
        //  2. Split total multinomially with weights r_k/Σr_k.
        // For ungrouped transitions (inflows, non-competing exits)
        // keep the standard tau-leap independent Poisson draw.
        let mut handled = vec![false; n_transitions];
        let mut pending_deltas: Vec<(usize, i64)> = Vec::new();
        for &(src_local, ref group) in &model.source_groups {
            let n_src = int_s.counts[src_local].max(0);
            if n_src == 0 {
                for &tr_idx in group { handled[tr_idx] = true; }
                continue;
            }
            // Compute effective per-capita rates (with overdispersion if any).
            let mut effective: Vec<(usize, f64)> = Vec::with_capacity(group.len());
            let mut total_rate = 0.0_f64;
            for &tr_idx in group {
                let rate = propensities[tr_idx];
                if rate <= 0.0 { handled[tr_idx] = true; continue; }
                let per_capita = rate / n_src as f64;
                let eff = match draws[tr_idx] {
                    ResolvedDraw::Deterministic => {
                        // Handle deterministic separately below; don't compete.
                        handled[tr_idx] = true;
                        continue;
                    }
                    ResolvedDraw::Overdispersed(sigma_sq) => {
                        per_capita * rng.gamma_multiplier(sigma_sq, dt)
                    }
                    ResolvedDraw::Poisson => per_capita,
                };
                total_rate += eff;
                effective.push((tr_idx, eff));
            }
            if total_rate <= 0.0 || effective.is_empty() { continue; }
            // gh#audit-H3: stable (p, q) primitive (q discarded here).
            let (p_total, _q) = crate::inference::numerics::prob_q_from_rate_dt(total_rate, dt);
            let p_total = p_total.clamp(0.0, 1.0);
            let mut n_events = rng.binomial(n_src as u64, p_total);
            let n_competing = effective.len();
            let mut rate_remaining = total_rate;
            for (k, &(tr_idx, eff_rate)) in effective.iter().enumerate() {
                let count = if k == n_competing - 1 {
                    n_events
                } else if n_events > 0 && rate_remaining > 0.0 {
                    let p_split = (eff_rate / rate_remaining).clamp(0.0, 1.0);
                    let c = rng.binomial(n_events, p_split);
                    n_events -= c;
                    rate_remaining -= eff_rate;
                    c
                } else {
                    0
                };
                for &(local, delta) in &model.transition_stoich[tr_idx] {
                    pending_deltas.push((local, delta * count as i64));
                }
                current_flows.add(tr_idx, count);
                handled[tr_idx] = true;
            }
        }

        // Inflows and ungrouped transitions: independent draws per the
        // standard tau-leap approximation.
        for (i, &lambda) in propensities.iter().enumerate() {
            if handled[i] { continue; }
            let mean = lambda * dt;
            let count = match draws[i] {
                ResolvedDraw::Poisson => rng.poisson(mean),
                ResolvedDraw::Deterministic => mean.round() as u64,
                ResolvedDraw::Overdispersed(sigma_sq) => rng.neg_binomial(mean, sigma_sq, dt),
            };
            for &(local, delta) in &model.transition_stoich[i] {
                pending_deltas.push((local, delta * count as i64));
            }
            current_flows.add(i, count);
        }

        for (local, delta) in pending_deltas.drain(..) {
            int_s.counts[local] += delta;
        }

        // gh#audit-C5 / S2. Negative count after stoichiometry → hard
        // error (BinomialOvershoot cause). The multinomial invariant
        // (RM10 / 2026-04-19 review) says this shouldn't happen on
        // tau-leap; if it does, the user wants to know. Inference
        // layers catch and recover per-particle.
        if let Some((local, val)) = int_s.first_negative() {
            return Err(crate::error::SimError::NegativeCount {
                compartment: model.comp_index.iter()
                    .find(|(_, &g)| model.global_to_int.get(g).copied().flatten() == Some(local))
                    .map(|(n, _)| n.clone())
                    .unwrap_or_else(|| format!("(local-int-{local})")),
                attempted_value: val,
                t,
                cause: crate::error::NegativeCountCause::BinomialOvershoot,
            });
        }

        // RK4 for real compartments (integer state now at end-of-step)
        if n_real > 0 {
            rk4_step(model, &int_s, &mut real_s, params, t, dt)?;
            real_s.clamp_nonneg();
        }

        t += dt;

        // Apply intervention if now at that time
        if iv_times.get(iv_idx).copied().is_some_and(|iv| (iv - t).abs() < 1e-10) {
            apply_interventions_at(t, model, &fire_steps, cfg.dt, &mut int_s, &mut real_s, params, 1e-10)?;
            while iv_idx < iv_times.len() && iv_times[iv_idx] <= t + 1e-10 { iv_idx += 1; }
        }

        // Record outputs
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

    // Flush remaining outputs
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
