//! Post-fit Richardson dt-convergence check at θ̂ (gh#52).
//!
//! Detects when an MLE is discretization-dependent — synthetic-recovery
//! fits at the same dt share the bias and pass spuriously, but PPC
//! against real data exposes the trap. This module evaluates
//! `loglik(θ̂; dt)` on a halving ladder and warns when the loglik is
//! still drifting past `τ` nats.
//!
//! See `docs/dev/proposals/2026-05-07-richardson-dt-check.md` for the
//! full design rationale and the boarding-school SIR reproducer.
//!
//! This module ships in three layers:
//!   1. **Verdict logic** (this commit) — pure functions over a ladder
//!      vector; fully unit-testable without I/O or particle-filter
//!      machinery.
//!   2. **Richardson runner** (next commit) — runs `pfilter` at each
//!      ladder rung by overriding `SMCConfig.dt`.
//!   3. **fit-pipeline wiring** (subsequent commits) — auto-run after
//!      compound gate; serialise to `fit_state.toml.dt_check`; `camdl
//!      fit dt-check <fit_dir>` standalone.

use crate::fit::config_v2::{CombineMode, DtCheckConfig};
use crate::fit::loglik_eval::combine_with_se;
use crate::fit::runner::{
    run_quick_pfilter_with_dt, ruled_out_or_surface, FitRunConfig,
};
use sim::inference::compute_ode_loglik;
use crate::run_meta::InferenceBackend;
use serde::{Deserialize, Serialize};

/// One rung of the Richardson halving ladder: `loglik(θ̂; dt)` with its
/// per-evaluation standard error from the `n_replicates`-replicate
/// combiner.
#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
pub struct LadderEntry {
    pub dt: f64,
    pub loglik: f64,
    pub se: f64,
}

/// Two-leg verdict states. See proposal §"Two-leg verdict".
#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum DtCheckVerdict {
    /// Both legs pass (`|Δ_leg1| < τ` AND `|Δ_leg2| < τ`).
    Pass,
    /// Halving stability passes but plateau width is in `[τ, 2τ]` —
    /// fit is borderline; soft warning, not error.
    Marginal,
    /// At least one leg exceeds `τ`. Hard warning; user should re-fit
    /// at finer dt before interpreting θ̂.
    Fail,
    /// Check disabled (via `[stages.X.dt_check]` `enabled = false`,
    /// or `--no-dt-check` on the CLI). No ladder is run.
    Skipped,
}

impl std::fmt::Display for DtCheckVerdict {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            DtCheckVerdict::Pass     => "pass",
            DtCheckVerdict::Marginal => "marginal",
            DtCheckVerdict::Fail     => "fail",
            DtCheckVerdict::Skipped  => "skipped",
        })
    }
}

/// Full result of the dt-check, serialisable into `fit_state.toml`'s
/// `[dt_check]` block. Fields parallel the proposal's TOML schema.
#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
pub struct DtCheckResult {
    pub verdict: DtCheckVerdict,
    /// `n_halvings + 1` rungs (the fit-dt rung plus halvings). Empty
    /// for `Skipped`.
    pub ladder: Vec<LadderEntry>,
    /// `|ll(dt_fit) − ll(dt_fit/2)|`, the halving-stability leg. NaN
    /// for skipped or single-rung ladders.
    pub leg1_delta_nats: f64,
    /// `|ll(dt_fit) − ll(dt_min)|`, the plateau-width leg. Equals
    /// `leg1_delta_nats` when `n_halvings == 1`.
    pub leg2_delta_nats: f64,
    /// User-set `threshold_nats` floor.
    pub threshold_nats: f64,
    /// `max(threshold_nats, 4 · σ_max)` — the actual gate.
    pub threshold_se_aware_nats: f64,
    /// Auxiliary signal: did `σ` inflate non-monotonically as dt
    /// halved? See `pf_se_inflation_fires`.
    pub pf_se_inflation: bool,
    /// One-line human-readable summary stitched together from the
    /// numbers above. Stable enough for `fit summary` to display
    /// verbatim.
    pub notes: String,
}

/// InferenceBackend-specific default thresholds. See proposal §"InferenceBackend-specific
/// τ defaults" for the convergence-order rationale.
///
/// All fit-stage backends have a dt parameter — the fit-stage `InferenceBackend`
/// enum only carries `ChainBinomial` and `Ode` (Gillespie is a
/// forward-sim backend, not a fit-stage backend). The threshold differs
/// by convergence order:
///
/// - **`ChainBinomial`** is O(dt) weak-convergent (Euler-multinomial /
///   `reulermultinom` family); the boarding-school SIR reproducer's
///   20.8-nat drift at dt=1.0 vs 1.5-nat at dt=0.1 calibrates the
///   2.0-nat default.
/// - **`Ode`** is O(dt⁴) when using RK4 (camdl's default); the finer
///   convergence order earns a tighter 0.5-nat default. Calibration
///   against He2010 measles at the published dt = 2/365 weeks is
///   tracked in the L9 external-validation TODO.
///
/// `--strict` halves both defaults (chain-binomial → 0.5; ode → 0.1)
/// for research-quality fits where sub-nat differences matter.
pub fn default_threshold_for_backend(backend: InferenceBackend, strict: bool) -> f64 {
    match backend {
        InferenceBackend::ChainBinomial => if strict { 0.5 } else { 2.0 },
        InferenceBackend::Ode           => if strict { 0.1 } else { 0.5 },
    }
}

/// Compute the SE-aware threshold from a user-set floor and the
/// largest observed standard error in the ladder.
///
/// The 4× multiplier is half the compound gate's `8·σ_max` because
/// this is a per-evaluation comparison rather than a chain-level
/// spread (degree-of-freedom argument).
pub fn se_aware_threshold(threshold_nats: f64, sigma_max: f64) -> f64 {
    threshold_nats.max(4.0 * sigma_max)
}

/// True iff the ladder shows `σ(dt_fine) > 2 · σ(dt_fit)` at any
/// point as dt halves. Auxiliary "PF-SE inflation" signal — a
/// dt-misconfigured fit's particle filter is structurally degenerate
/// at finer dt because the trajectories the coarse-dt MLE was tuned
/// to explain become low-probability under finer-grained dynamics
/// (see proposal §"Auxiliary: PF-SE inflation signal").
///
/// Non-monotonic dips back below 2× don't cancel an earlier rise —
/// once SE has more than doubled at *some* finer dt, the signal
/// fires.
pub fn pf_se_inflation_fires(ladder: &[LadderEntry]) -> bool {
    if ladder.is_empty() { return false; }
    let sigma_fit = ladder[0].se;
    if sigma_fit <= 0.0 || !sigma_fit.is_finite() { return false; }
    ladder.iter().skip(1).any(|rung| rung.se > 2.0 * sigma_fit)
}

/// Compute the verdict from a ladder + threshold inputs. Pure
/// function; no I/O. `ladder[0]` is the fit-dt rung; `ladder[1..]`
/// are halvings in order.
pub fn compute_verdict(
    ladder: &[LadderEntry],
    threshold_nats: f64,
) -> DtCheckResult {
    if ladder.is_empty() {
        return DtCheckResult {
            verdict: DtCheckVerdict::Skipped,
            ladder: Vec::new(),
            leg1_delta_nats: f64::NAN,
            leg2_delta_nats: f64::NAN,
            threshold_nats,
            threshold_se_aware_nats: threshold_nats,
            pf_se_inflation: false,
            notes: "skipped: empty ladder (backend has no dt parameter \
                or check disabled).".into(),
        };
    }

    let sigma_max = ladder.iter().map(|r| r.se).fold(0.0_f64, f64::max);
    let tau = se_aware_threshold(threshold_nats, sigma_max);

    let ll_fit = ladder[0].loglik;
    let leg1 = if ladder.len() >= 2 {
        ll_fit - ladder[1].loglik
    } else {
        f64::NAN
    };
    let ll_min = ladder.last().unwrap().loglik;
    let leg2 = ll_fit - ll_min;

    let leg1_passes = leg1.abs() < tau;
    let leg2_passes = leg2.abs() < tau;
    let leg2_marginal = leg2.abs() >= tau && leg2.abs() < 2.0 * tau;

    let verdict = match (leg1_passes, leg2_passes, leg2_marginal) {
        (true, true, _) => DtCheckVerdict::Pass,
        // halving-stability OK, plateau width borderline (within 2τ)
        (true, false, true) => DtCheckVerdict::Marginal,
        _ => DtCheckVerdict::Fail,
    };

    let pf_se_inflation = pf_se_inflation_fires(ladder);
    let notes = build_notes(verdict, leg1, leg2, tau, pf_se_inflation);

    DtCheckResult {
        verdict,
        ladder: ladder.to_vec(),
        leg1_delta_nats: leg1,
        leg2_delta_nats: leg2,
        threshold_nats,
        threshold_se_aware_nats: tau,
        pf_se_inflation,
        notes,
    }
}

/// Defaults inherited from the stage's `loglik_eval` block when the
/// dt-check user didn't override n_particles/n_replicates/combine.
/// Caller passes these as the inherit-from values.
pub struct DtCheckInherits {
    pub n_particles: usize,
    pub n_replicates: usize,
    pub combine: CombineMode,
}

/// A `Skipped` verdict with an empty ladder, shared by the disabled-path
/// early returns of every wrapper. `threshold_nats` carries the floor that
/// *would* have applied so `fit summary` can still report it.
fn skipped_result(threshold_nats: f64, notes: &str) -> DtCheckResult {
    DtCheckResult {
        verdict: DtCheckVerdict::Skipped,
        ladder: Vec::new(),
        leg1_delta_nats: f64::NAN,
        leg2_delta_nats: f64::NAN,
        threshold_nats,
        threshold_se_aware_nats: f64::NAN,
        pf_se_inflation: false,
        notes: notes.into(),
    }
}

/// Build the dt ladder (`dt_fit`, then `n_halvings` successive halvings) and
/// reduce it to a verdict, given a per-rung evaluator. This is the
/// backend-agnostic seam: the stochastic PF path passes an evaluator that runs
/// `n_replicates` particle filters and combines them to `(loglik, se)`; the
/// deterministic ODE path passes one that calls `compute_ode_loglik` once
/// (`se = 0`). Everything downstream of the ladder — the two-leg verdict,
/// threshold, SE-aware floor, rendering, serialization — is shared and lives in
/// [`compute_verdict`] (gh#52, gh#227).
///
/// `eval_rung(dt, rung_i) -> Result<(loglik, se), String>`: an `Err` aborts the
/// whole ladder (a structural error must not be hidden as a quietly poisoned
/// rung); `Ok((-inf, _))` is a ruled-out θ excursion at that dt and is recorded
/// faithfully.
pub fn run_ladder(
    dt_fit: f64,
    n_halvings: usize,
    threshold_floor: f64,
    mut eval_rung: impl FnMut(f64, usize) -> Result<(f64, f64), String>,
) -> Result<DtCheckResult, String> {
    let mut dts: Vec<f64> = Vec::with_capacity(n_halvings + 1);
    dts.push(dt_fit);
    let mut next = dt_fit;
    for _ in 0..n_halvings {
        next *= 0.5;
        dts.push(next);
    }

    let mut ladder: Vec<LadderEntry> = Vec::with_capacity(dts.len());
    for (rung_i, &dt) in dts.iter().enumerate() {
        let (loglik, se) = eval_rung(dt, rung_i)?;
        ladder.push(LadderEntry { dt, loglik, se });
    }

    Ok(compute_verdict(&ladder, threshold_floor))
}

/// Run the Richardson halving ladder + verdict end-to-end on the **stochastic
/// particle-filter** likelihood. Returns a fully-populated `DtCheckResult`
/// ready to write to `fit_state.toml.dt_check`.
///
/// `theta_hat` is the MLE θ̂ (e.g., the IF2 winner's clean-eval θ);
/// the ladder evaluates `pfilter(loglik | θ̂; dt)` at
/// `dt ∈ {dt_fit, dt_fit/2, ..., dt_fit/2^n_halvings}`. Each rung
/// runs `n_replicates` independent PF replicates and combines them
/// via `combine` to a single `(loglik, se)` pair (matches the
/// loglik_eval/clean-eval shape used by the compound gate).
///
/// `seed` is mixed with the rung index and replicate index to derive
/// per-(rung, replicate) PF seeds; the same `seed` reproduces the
/// same ladder.
///
/// Disabled (`config.enabled = false`) → returns a `Skipped` verdict
/// with empty ladder. The deterministic sibling is
/// [`run_richardson_ladder_ode`].
pub fn run_richardson_ladder(
    run_config: &FitRunConfig,
    theta_hat: &[f64],
    config: &DtCheckConfig,
    backend: InferenceBackend,
    strict: bool,
    inherits: &DtCheckInherits,
    seed: u64,
) -> Result<DtCheckResult, String> {
    let threshold_floor = config.threshold_nats
        .unwrap_or_else(|| default_threshold_for_backend(backend, strict));
    if !config.enabled {
        return Ok(skipped_result(threshold_floor, "skipped: dt_check.enabled = false."));
    }

    let n_particles  = config.n_particles.unwrap_or(inherits.n_particles);
    let n_replicates = config.n_replicates.unwrap_or(inherits.n_replicates);
    let combine      = config.combine.unwrap_or(inherits.combine);

    run_ladder(run_config.if2_config.dt, config.n_halvings, threshold_floor, |dt, rung_i| {
        let mut per_rep: Vec<f64> = Vec::with_capacity(n_replicates);
        for k in 0..n_replicates {
            // Seed scheme: (seed, rung_i, k) → distinct PF seed.
            // 1_000_003 is a small prime spacing the rung block —
            // collisions need (rung × 1_000_003 + replicate) to alias
            // into another rung's range, which doesn't happen at
            // realistic n_replicates (< 10⁶).
            let pf_seed = seed
                .wrapping_add((rung_i as u64).wrapping_mul(1_000_003))
                .wrapping_add(k as u64);
            // gh#224: a ruled-out θ scores −∞; a structural error aborts the
            // dt-check rather than poisoning the ladder with a hidden failure.
            let (ll, _stats) = ruled_out_or_surface(run_quick_pfilter_with_dt(
                run_config, theta_hat, n_particles, Some(dt), pf_seed))
                .map_err(|e| format!(
                    "dt-check: structural error at dt={} (rung {}, rep {}): {}",
                    dt, rung_i, k, e))?;
            per_rep.push(ll);
        }
        Ok(combine_with_se(&per_rep, combine))
    })
}

/// Deterministic ODE dt-convergence check at θ̂ — the deterministic sibling of
/// [`run_richardson_ladder`], for the `nl-sbplx` / `nl-bobyqa` (ODE-MLE) and
/// `mh` (ODE marginal-likelihood MH) stages (gh#52, gh#227).
///
/// Re-evaluates the SAME `compute_ode_loglik(θ̂; dt)` the stage scored with, on
/// a halving ladder `dt ∈ {dt_fit, dt_fit/2, …}`, and warns when the fitted
/// θ̂'s likelihood is still drifting past τ nats — i.e. when the MLE/MAP is
/// discretization-dependent. This is exactly the trap a coarse-dt
/// synthetic-recovery fit passes spuriously: data-generation and fit share the
/// same dt bias, so only an independent dt ladder exposes it.
///
/// The ODE likelihood is deterministic, so each rung is a single eval with
/// `se = 0` — there are no replicates and no particles, and the PF-only
/// [`DtCheckInherits`] knobs do not apply. Because ODE snapshots land on the
/// model's fixed output grid (`StepPolicy::Exact`), halving `dt` only refines
/// the RK4 substep *between* outputs — the obs-time alignment inside
/// `compute_ode_loglik` is preserved at every rung.
///
/// Convergence-order note: since the augmented-flow unification (gh#166 Q1B),
/// BOTH prevalence (RK4 state, O(dt⁴)) and incidence (augmented flow, `dc/dt =
/// rate` through the same RK4 stages, also O(dt⁴)) converge at the same high
/// order, so the `InferenceBackend::Ode` τ calibrated to the prevalence case applies to
/// incidence too. The one exception is a model that references `dt` in a rate
/// (`Expr::Dt` / RUNTIME_DT), which keeps the first-order Euler flow (O(dt), the
/// augmented derivative is undefined when the rate depends on the step size) —
/// for those, an incidence-heavy fit can still register Marginal/Fail at a dt
/// where prevalence passes, correctly flagging the dt-sensitive incidence.
///
/// Disabled (`config.enabled = false`) → returns a `Skipped` verdict.
pub fn run_richardson_ladder_ode(
    compiled: &sim::compiled_model::CompiledModel,
    obs_model: &sim::inference::MultiStreamObsModel,
    obs_times: &[f64],
    theta_hat: &[f64],
    dt_fit: f64,
    config: &DtCheckConfig,
    strict: bool,
) -> Result<DtCheckResult, String> {
    let threshold_floor = config.threshold_nats
        .unwrap_or_else(|| default_threshold_for_backend(InferenceBackend::Ode, strict));
    if !config.enabled {
        return Ok(skipped_result(threshold_floor, "skipped: dt_check.enabled = false."));
    }

    run_ladder(dt_fit, config.n_halvings, threshold_floor, |dt, _rung_i| {
        match compute_ode_loglik(compiled, obs_model, obs_times, dt, theta_hat, dt) {
            Ok(ll) => Ok((ll, 0.0)),
            // gh#224: a structural error (config bug, unknown compartment)
            // aborts the dt-check rather than poisoning the ladder; a per-θ
            // excursion at a finer dt scores −∞ (recorded faithfully), mirroring
            // the PF path's `ruled_out_or_surface` contract.
            Err(e) if e.is_structural() => Err(format!(
                "dt-check: structural error at dt={}: {}", dt, e)),
            Err(_) => Ok((f64::NEG_INFINITY, 0.0)),
        }
    })
}

/// Render the dt-check verdict to stderr in the proposal's
/// terminal-output format. Pass case is one line; fail/marginal is
/// the ladder + verdict + the synth-recovery warning text.
pub fn print_terminal_report(result: &DtCheckResult) {
    use std::io::Write as _;
    let stderr = std::io::stderr();
    let mut w = stderr.lock();

    if matches!(result.verdict, DtCheckVerdict::Skipped) {
        // Don't shout when the user explicitly opted out.
        let _ = writeln!(w, "\ndt-convergence at θ̂: skipped");
        return;
    }
    if matches!(result.verdict, DtCheckVerdict::Pass) {
        let _ = writeln!(w,
            "\ndt-convergence at θ̂: \x1b[32mPASS\x1b[0m  ({})",
            result.notes);
        return;
    }

    let _ = writeln!(w);
    let _ = writeln!(w,
        "dt-convergence at θ̂ (Richardson check, {} ladder rungs):",
        result.ladder.len());
    for (i, rung) in result.ladder.iter().enumerate() {
        if i == 0 {
            let _ = writeln!(w,
                "  dt = {:.4}   ll = {:.2} ± {:.2}   (fit)",
                rung.dt, rung.loglik, rung.se);
        } else {
            let delta = rung.loglik - result.ladder[0].loglik;
            let _ = writeln!(w,
                "  dt = {:.4}   ll = {:.2} ± {:.2}   Δ = {:+.2} nats",
                rung.dt, rung.loglik, rung.se, delta);
        }
    }
    let label = match result.verdict {
        DtCheckVerdict::Fail     => "\x1b[31mFAIL\x1b[0m",
        DtCheckVerdict::Marginal => "\x1b[33mMARGINAL\x1b[0m",
        DtCheckVerdict::Pass | DtCheckVerdict::Skipped => unreachable!(),
    };
    let _ = writeln!(w,
        "  ⚠ {}: {} (threshold τ = {:.2} nats; SE-aware floor 4·σ_max = {:.2} nats).",
        label, result.notes, result.threshold_nats, result.threshold_se_aware_nats);
    if matches!(result.verdict, DtCheckVerdict::Fail) {
        let _ = writeln!(w,
            "    MLE is discretization-dependent. Re-fit at dt ≤ {:.3} \
             before interpreting θ̂.",
            result.ladder.last().map(|r| r.dt).unwrap_or(f64::NAN));
        // gh#53: high-magnitude failures (≫ 100·τ) often indicate an
        // integrator-side bug rather than a too-coarse fit. Surface
        // this caveat so users don't immediately re-fit at finer dt
        // and trust the result without external cross-checking.
        let leg2_abs = result.leg2_delta_nats.abs();
        if leg2_abs > 100.0 * result.threshold_se_aware_nats {
            let _ = writeln!(w);
            let _ = writeln!(w,
                "    \x1b[33m⚠ For failures of this magnitude (≥ 100·τ; here {:.0}×),\x1b[0m",
                leg2_abs / result.threshold_se_aware_nats);
            let _ = writeln!(w,
                "    the discretization itself may be the issue — i.e. the");
            let _ = writeln!(w,
                "    integrator may be exhibiting a sub-step numerical bug,");
            let _ = writeln!(w,
                "    not just a too-coarse fit dt. Cross-check against an");
            let _ = writeln!(w,
                "    independent reference (e.g. pomp's pfilter at the same θ̂");
            let _ = writeln!(w,
                "    and finer delta.t) before re-fitting. See gh#53.");
        }
        let _ = writeln!(w);
        let _ = writeln!(w,
            "    Note: synthetic recovery at the same dt cannot detect this");
        let _ = writeln!(w,
            "    bias — the simulator and inference loop share the same dt and");
        let _ = writeln!(w,
            "    the discretization error cancels in the recovery metric. This");
        let _ = writeln!(w,
            "    Richardson check is the supplementary validator for that");
        let _ = writeln!(w,
            "    failure mode.");
    }
    if result.pf_se_inflation {
        let _ = writeln!(w);
        let ses: Vec<String> = result.ladder.iter()
            .map(|r| format!("{:.2}", r.se)).collect();
        let _ = writeln!(w,
            "  ⚠ PF-SE inflation: σ went {} nats as dt halved.",
            ses.join(" → "));
        let _ = writeln!(w,
            "    Often co-occurs with dt-bias; the misconfigured MLE's");
        let _ = writeln!(w,
            "    trajectories are improbable under finer dynamics.");
    }
}

fn build_notes(
    verdict: DtCheckVerdict,
    leg1: f64,
    leg2: f64,
    tau: f64,
    pf_se_inflation: bool,
) -> String {
    match verdict {
        DtCheckVerdict::Pass => format!(
            "|Δ_leg1| = {:.2}, |Δ_leg2| = {:.2} nats (vs τ = {:.2}); converged.",
            leg1.abs(), leg2.abs(), tau),
        DtCheckVerdict::Marginal => format!(
            "leg-1 |Δ| = {:.2} nats < τ = {:.2}; leg-2 |Δ| = {:.2} nats in \
             [τ, 2τ] — borderline. Plateau is wider than ideal but the \
             halving step is small; either re-fit at one extra halving or \
             accept with a flag.",
            leg1.abs(), tau, leg2.abs()),
        DtCheckVerdict::Fail => {
            let leg1_str = if leg1.is_nan() {
                "leg-1 unavailable (n_halvings < 1)".to_string()
            } else if leg1.abs() >= tau {
                format!("leg-1 |Δ| = {:.2} nats > τ = {:.2} ({:.1}×)",
                    leg1.abs(), tau, leg1.abs() / tau)
            } else {
                format!("leg-1 |Δ| = {:.2} nats ≤ τ", leg1.abs())
            };
            let leg2_str = if leg2.abs() >= tau {
                format!("leg-2 |Δ| = {:.2} nats > τ = {:.2} ({:.1}×)",
                    leg2.abs(), tau, leg2.abs() / tau)
            } else {
                format!("leg-2 |Δ| = {:.2} nats ≤ τ", leg2.abs())
            };
            let inflation_str = if pf_se_inflation {
                "; PF-SE inflated as dt halved (auxiliary signal)"
            } else { "" };
            format!("{}; {}{}", leg1_str, leg2_str, inflation_str)
        }
        DtCheckVerdict::Skipped => "skipped: backend has no dt parameter \
            or check disabled.".into(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rung(dt: f64, loglik: f64, se: f64) -> LadderEntry {
        LadderEntry { dt, loglik, se }
    }

    // ── default_threshold_for_backend ───────────────────────────────

    #[test]
    fn default_threshold_chain_binomial_is_2_nats() {
        assert_eq!(
            default_threshold_for_backend(InferenceBackend::ChainBinomial, false),
            2.0);
    }

    #[test]
    fn default_threshold_chain_binomial_strict_is_half_nat() {
        assert_eq!(
            default_threshold_for_backend(InferenceBackend::ChainBinomial, true),
            0.5);
    }

    #[test]
    fn default_threshold_ode_rk4_is_half_nat() {
        // ODE backend: O(dt⁴) convergence earns a tighter default.
        assert_eq!(
            default_threshold_for_backend(InferenceBackend::Ode, false), 0.5);
    }

    #[test]
    fn default_threshold_ode_strict_is_tenth_nat() {
        assert_eq!(
            default_threshold_for_backend(InferenceBackend::Ode, true), 0.1);
    }

    // ── se_aware_threshold ──────────────────────────────────────────

    #[test]
    fn se_aware_threshold_floor_wins_at_low_sigma() {
        // σ_max = 0.1 → 4·σ_max = 0.4 < 2.0 floor → τ = 2.0.
        assert_eq!(se_aware_threshold(2.0, 0.1), 2.0);
    }

    #[test]
    fn se_aware_threshold_se_wins_at_high_sigma() {
        // σ_max = 1.0 → 4·σ_max = 4.0 > 2.0 floor → τ = 4.0.
        assert_eq!(se_aware_threshold(2.0, 1.0), 4.0);
    }

    // ── pf_se_inflation_fires ───────────────────────────────────────

    #[test]
    fn pf_se_inflation_fires_on_3x_increase() {
        let ladder = vec![
            rung(1.0,  -50.0, 0.05),
            rung(0.5,  -50.5, 0.10),
            rung(0.25, -51.0, 0.30),  // 6× σ_fit
        ];
        assert!(pf_se_inflation_fires(&ladder));
    }

    #[test]
    fn pf_se_inflation_quiet_on_stable_se() {
        let ladder = vec![
            rung(1.0,  -50.0, 0.20),
            rung(0.5,  -50.5, 0.21),
            rung(0.25, -51.0, 0.19),
        ];
        assert!(!pf_se_inflation_fires(&ladder));
    }

    #[test]
    fn pf_se_inflation_fires_even_if_later_rung_drops_back() {
        // Vince's reproducer-shape: σ inflates at one mid-rung but
        // drops back at the finest. Once any rung exceeds 2× σ_fit,
        // the signal fires (the drop-back is consistent with PF
        // re-stabilising under enough particles, but the inflation
        // still indicates dt-bias).
        let ladder = vec![
            rung(1.0,  -50.0, 0.10),
            rung(0.5,  -55.0, 0.30),  // 3× — fires
            rung(0.25, -60.0, 0.10),
        ];
        assert!(pf_se_inflation_fires(&ladder));
    }

    #[test]
    fn pf_se_inflation_quiet_on_zero_or_nan_sigma_fit() {
        // Defensive: division-by-zero / NaN-comparison protection.
        let ladder = vec![
            rung(1.0, -50.0, 0.0),
            rung(0.5, -55.0, 0.5),
        ];
        assert!(!pf_se_inflation_fires(&ladder));
    }

    // ── run_ladder (the backend-agnostic driver / seam) ─────────────

    #[test]
    fn run_ladder_builds_halving_grid() {
        // dt_fit = 1.0, n_halvings = 2 → rungs at {1.0, 0.5, 0.25}. The
        // closure records the dts it was asked for, in order.
        let mut seen: Vec<f64> = Vec::new();
        let r = run_ladder(1.0, 2, 2.0, |dt, _i| {
            seen.push(dt);
            Ok((-50.0, 0.0)) // flat loglik
        })
        .expect("flat ladder should not error");
        assert_eq!(seen, vec![1.0, 0.5, 0.25]);
        assert_eq!(r.ladder.iter().map(|e| e.dt).collect::<Vec<_>>(), vec![1.0, 0.5, 0.25]);
    }

    #[test]
    fn run_ladder_passes_on_flat_loglik() {
        // A converged θ̂: loglik identical at every dt → both legs zero → Pass.
        let r = run_ladder(1.0, 2, 0.5, |_dt, _i| Ok((-123.4, 0.0))).unwrap();
        assert_eq!(r.verdict, DtCheckVerdict::Pass);
        assert_eq!(r.leg1_delta_nats, 0.0);
        assert_eq!(r.leg2_delta_nats, 0.0);
    }

    #[test]
    fn run_ladder_fails_on_drifting_loglik() {
        // Discretization-dependent θ̂: loglik climbs 5 nats per halving, far
        // past the 0.5-nat ODE floor → Fail. This is the artifact the ODE
        // dt-check exists to catch (coarse dt creates a fake basin).
        let r = run_ladder(1.0, 2, 0.5, |dt, _i| {
            // dt = 1.0 → -60, 0.5 → -55, 0.25 → -50: monotone, big drift.
            let ll = -60.0 + (1.0 - dt) * 10.0;
            Ok((ll, 0.0))
        })
        .unwrap();
        assert_eq!(r.verdict, DtCheckVerdict::Fail);
        assert!(r.leg2_delta_nats.abs() > 0.5);
    }

    #[test]
    fn run_ladder_propagates_eval_error() {
        // A structural error on any rung aborts the whole ladder rather than
        // being hidden as a quietly poisoned rung (gh#224 contract).
        let r = run_ladder(1.0, 2, 0.5, |dt, rung_i| {
            if rung_i == 1 {
                Err(format!("boom at dt={dt}"))
            } else {
                Ok((-50.0, 0.0))
            }
        });
        assert!(r.is_err());
        assert!(r.unwrap_err().contains("boom"));
    }

    // ── compute_verdict ─────────────────────────────────────────────

    #[test]
    fn verdict_pass_when_both_legs_below_tau() {
        // Vince's converged dt=0.1 fit: ll flat to within 1.5 nats.
        let ladder = vec![
            rung(0.1,  -58.7, 0.07),
            rung(0.05, -59.3, 0.07),
            rung(0.025, -59.0, 0.08),
        ];
        let r = compute_verdict(&ladder, 2.0);
        assert_eq!(r.verdict, DtCheckVerdict::Pass);
        assert!(r.leg1_delta_nats.abs() < 2.0);
        assert!(r.leg2_delta_nats.abs() < 2.0);
    }

    #[test]
    fn verdict_fail_on_leg1_drift() {
        // Vince's misconfigured dt=1.0 fit, halving leg-1: -3.17 nats.
        let ladder = vec![
            rung(1.0,  -62.6, 0.07),
            rung(0.5,  -65.7, 0.24),
            rung(0.25, -73.9, 0.63),
        ];
        let r = compute_verdict(&ladder, 2.0);
        assert_eq!(r.verdict, DtCheckVerdict::Fail);
        assert!(r.leg1_delta_nats.abs() > 2.0);
        assert!(r.leg2_delta_nats.abs() > 2.0);
    }

    #[test]
    fn verdict_marginal_when_plateau_in_tau_to_2tau() {
        // leg-1 small (1.5 < τ = 2.0), leg-2 in [2, 4) → Marginal.
        let ladder = vec![
            rung(1.0,  -50.0, 0.10),
            rung(0.5,  -51.5, 0.10),
            rung(0.25, -53.0, 0.10),
        ];
        let r = compute_verdict(&ladder, 2.0);
        assert_eq!(r.verdict, DtCheckVerdict::Marginal,
            "leg1=1.5 < τ=2 (passes), leg2=3.0 in [τ, 2τ] = [2, 4) (marginal)");
    }

    #[test]
    fn verdict_fail_when_plateau_exceeds_2tau() {
        // leg-1 small, leg-2 > 2τ → Fail (cumulative drift big).
        let ladder = vec![
            rung(1.0,  -50.0, 0.10),
            rung(0.5,  -51.5, 0.10),
            rung(0.25, -55.0, 0.10),  // leg-2 = 5.0 > 2τ = 4.0
        ];
        let r = compute_verdict(&ladder, 2.0);
        assert_eq!(r.verdict, DtCheckVerdict::Fail);
    }

    #[test]
    fn verdict_skipped_on_empty_ladder() {
        let r = compute_verdict(&[], 2.0);
        assert_eq!(r.verdict, DtCheckVerdict::Skipped);
        assert!(r.ladder.is_empty());
    }

    #[test]
    fn verdict_se_aware_threshold_floor_kicks_in() {
        // Same loglik drift, but σ_max = 1.0 → 4·σ = 4.0 → τ = 4.0
        // (was 2.0). Drift of 3.0 nats now passes leg-1 (3 < 4).
        let ladder = vec![
            rung(1.0, -50.0, 0.10),
            rung(0.5, -53.0, 1.00),   // σ inflated
            rung(0.25, -53.5, 1.00),
        ];
        let r = compute_verdict(&ladder, 2.0);
        assert_eq!(r.threshold_se_aware_nats, 4.0);
        assert_eq!(r.verdict, DtCheckVerdict::Pass,
            "high σ should bump τ to 4·σ_max=4.0; drift of 3 nats then passes");
    }

    #[test]
    fn verdict_round_trips_through_toml() {
        // fit_state.toml schema check: serialise + deserialise the
        // verdict struct as a top-level TOML table.
        let r = compute_verdict(&[
            rung(1.0,  -62.6, 0.07),
            rung(0.5,  -65.7, 0.24),
            rung(0.25, -73.9, 0.63),
        ], 2.0);
        let toml_str = toml::to_string_pretty(&r).expect("serialise");
        let r2: DtCheckResult = toml::from_str(&toml_str).expect("deserialise");
        assert_eq!(r, r2);
    }
}
