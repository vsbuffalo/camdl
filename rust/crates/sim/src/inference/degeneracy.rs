//! gh#110. Shared particle-filter degeneracy watchdog.
//!
//! `bootstrap_filter` and IF2's inner per-iteration PF loop both
//! advance particles, reweight, and resample at each observation
//! window. Both have the same implicit contract: *return a finite
//! log-likelihood in bounded time for any θ within the parameter
//! bounds.* That contract silently breaks at bound-box extremes
//! (σ at the upper bound, R₀ ≈ 50, …) where ESS collapses to ~1
//! and the filter runs effectively forever producing no usable
//! information.
//!
//! This module collects the three detectable failure modes into one
//! helper so the two filter loops agree on what counts as "degenerate"
//! and what the bail looks like. Call from each loop after the
//! per-window weight-normalisation step:
//!
//! ```ignore
//! ess_history.push(swarm.ess());
//! if let Some(kind) = check_pf_degeneracy(
//!     &ess_history, t0.elapsed(), obs_idx, dead_count, n_particles,
//! ) {
//!     return Err(SimError::PFDegenerate {
//!         kind, obs_window: obs_idx, elapsed_s: t0.elapsed().as_secs_f64(),
//!     });
//! }
//! ```
//!
//! The ESS thresholds are hardcoded; the wall-clock budget is
//! workload-scaled and overridable (gh#133 — a user demanded it after a
//! healthy-but-slow 1500-particle fit was killed as "degenerate"). The
//! budget is `max(WALLCLOCK_FLOOR_S = 120, n_particles · per-particle)`,
//! overridable via `CAMDL_PF_WALLCLOCK_TIMEOUT_S` (`0` disables it); the
//! CLI `--pf-wallclock-timeout` / fit.toml field layer on top (same
//! precedence chain). `ESS_FLOOR = 2.0` over
//! `ESS_COLLAPSE_WINDOWS = 3` consecutive obs windows on a 200-obs
//! SEIR fit means we bail only if the filter is producing one usable
//! particle across 1.5% of the observation series — generous
//! against legitimate peak-dynamics dips, tight against sustained
//! collapse.

use std::time::Duration;

use crate::error::{PFDegenerateKind, SimError};

/// gh#110. Effective sample size below this value at an observation
/// window counts as "collapsed" for that window. Floor of 1.0 is
/// "literally one particle dominates"; 2.0 gives a small margin and
/// matches conventional resampling triggers.
pub const ESS_FLOOR: f64 = 2.0;

/// gh#110. Number of *consecutive* obs windows below `ESS_FLOOR`
/// required to bail. Single-window dips are normal at epidemic
/// peaks; sustained collapse is the pathology this watchdog catches.
pub const ESS_COLLAPSE_WINDOWS: usize = 3;

/// gh#110/gh#133. Floor for the per-call wall-clock budget. Small filters
/// still bail fast if genuinely stuck in `step()`; large filters get a
/// workload-scaled budget on top (see [`resolve_wallclock_budget_s`]) so a
/// slow-but-healthy big swarm is not false-flagged as degenerate.
pub const WALLCLOCK_FLOOR_S: u64 = 120;

/// gh#133. Per-particle contribution to the wall-clock budget. A healthy
/// 1500-particle filter on a moderate model legitimately needs
/// minutes/eval, so the budget scales with swarm size above the floor.
/// Deliberately generous — false-killing a real fit is the worse failure;
/// the env override / CLI flag handle the cases this heuristic misses.
const WALLCLOCK_PER_PARTICLE_S: f64 = 0.5;

/// gh#133. Env override for the wall-clock budget, in whole seconds.
/// `0` disables the wall-clock check entirely; unset or unparseable →
/// the workload-scaled default.
pub const WALLCLOCK_ENV: &str = "CAMDL_PF_WALLCLOCK_TIMEOUT_S";

/// gh#147 (M3.1). Fixed engine budget on the cumulative number of
/// *particle-substeps* a single particle-filter evaluation may execute.
/// This is the content-addressing-safe replacement for the wall-clock
/// watchdog's compute-blowup role.
///
/// The wall-clock watchdog (`WallClockExceeded`) makes a fit's
/// log-likelihood depend on machine speed and thread count — two runs
/// with identical inputs can diverge — which is fatal to content
/// addressing (a pure `f : Inputs → Artifacts` must be deterministic).
/// The wall-clock check only ever caught *compute blow-up*: a particle
/// filter never θ-wedges (it requires the chain-binomial backend, whose
/// per-window substep count `ceil(Δt/dt)` is deterministic and whose
/// per-substep work is θ-bounded), so "runs effectively forever" means
/// "executes pathologically many substeps". The deterministic analog of
/// "too many substeps" is a cap on the substep *count* itself —
/// independent of how fast each one runs.
///
/// Sizing: the national worst case is ~2000 particles × a 2-year horizon
/// at dt = 0.1 day ≈ 2000 × 7300 ≈ 1.5e7 substeps for one filter pass.
/// A larger-than-national pass (5000 particles × 4 years × dt = 0.05 ≈
/// 1.5e8) is still two orders of magnitude under this budget, while a
/// genuine wedge (a sub-microsecond dt, or a non-positive dt) overshoots
/// it by many orders of magnitude and is caught immediately. 1e10 is the
/// headroom point: generous enough never to false-trip a legitimate fit,
/// tight enough that a misconfiguration fails fast instead of hanging.
///
/// It is a *fixed* constant, never derived from core count, wall-clock,
/// or any machine state — deriving it would re-introduce exactly the
/// impurity the wall-clock watchdog has. The budget bounds a single PF
/// evaluation (one `bootstrap_filter` call; one IF2 iteration's inner
/// filter), so it is independent of how many IF2 iterations or chains a
/// fit runs.
pub const ITER_BUDGET: u64 = 10_000_000_000; // 1e10 particle-substeps

/// Resolve the per-call wall-clock budget in seconds, or `None` to disable
/// the check. `override_secs` is the raw override string (the env var
/// today; the CLI flag / fit.toml field will pass through this same path)
/// — taken as a parameter so the precedence logic is unit-testable without
/// touching process env. Precedence: a parseable override wins (`0` ⇒
/// disabled); otherwise scale with particle count above the floor.
fn resolve_wallclock_budget_s(override_secs: Option<&str>, n_particles: usize) -> Option<u64> {
    if let Some(raw) = override_secs {
        if let Ok(secs) = raw.trim().parse::<u64>() {
            return if secs == 0 { None } else { Some(secs) };
        }
        // Unparseable override → fall through to the scaled default rather
        // than silently disabling the guard.
    }
    let scaled = (n_particles as f64 * WALLCLOCK_PER_PARTICLE_S) as u64;
    Some(WALLCLOCK_FLOOR_S.max(scaled))
}

/// The effective wall-clock budget for this call, reading the env
/// override. `None` ⇒ the wall-clock check is disabled.
fn pf_wallclock_budget_s(n_particles: usize) -> Option<u64> {
    resolve_wallclock_budget_s(std::env::var(WALLCLOCK_ENV).ok().as_deref(), n_particles)
}

/// gh#110. Return the degeneracy mode if the filter has bailed at
/// this observation window, otherwise `None`.
///
/// Inputs:
/// - `ess_history` — every ESS recorded so far, length = obs_windows
///   processed. Only the last `ESS_COLLAPSE_WINDOWS` are inspected.
/// - `elapsed` — wall-clock since the filter call started.
/// - `_obs_window` — the just-processed window index. Reserved for
///   future diagnostics that want to localise the bail (e.g.
///   logging "ESS dropped at obs 47–50"). The current implementation
///   doesn't read it but plumbing it through the helper keeps the
///   call sites honest about what window they're reporting.
/// - `dead_count` — number of particles currently marked dead. Used
///   only for the `AllParticlesDead` check; pass 0 when the caller
///   doesn't track per-particle death (e.g. IF2's inner loop does
///   not, since its `process.step` errors propagate immediately).
/// - `n_particles` — total particles in the swarm.
///
/// Discrimination order (deterministic on tie cases):
///   1. `AllParticlesDead` — the limit case of ESS collapse,
///      cheap and diagnostically distinct; check first.
///   2. `WallClockExceeded` — independent of swarm state, fires
///      even if ESS looks healthy (e.g. step() is just slow).
///   3. `EssCollapsed` — requires `ESS_COLLAPSE_WINDOWS` history.
pub fn check_pf_degeneracy(
    ess_history: &[f64],
    elapsed: Duration,
    _obs_window: usize,
    dead_count: usize,
    n_particles: usize,
) -> Option<PFDegenerateKind> {
    // AllParticlesDead: every particle hit a per-particle-recoverable
    // error. Resampling on the next step would have zero weight
    // everywhere; bail before the divide-by-zero.
    if n_particles > 0 && dead_count >= n_particles {
        return Some(PFDegenerateKind::AllParticlesDead);
    }

    // WallClockExceeded: independent of swarm state. A filter still
    // running past its budget is stuck in `step()` — *or* over-particled
    // for the budget (gh#133). The budget scales with swarm size above the
    // floor and is overridable (env / CLI), so a slow-but-healthy big
    // filter is not false-flagged. `None` budget ⇒ the check is disabled.
    if let Some(budget_s) = pf_wallclock_budget_s(n_particles) {
        if elapsed.as_secs() >= budget_s {
            return Some(PFDegenerateKind::WallClockExceeded);
        }
    }

    // EssCollapsed: K consecutive obs windows at or below the floor.
    // Single-window dips during epidemic peaks are not pathology;
    // sustained collapse is. We need at least K windows of history
    // before this can fire — if the filter bails sooner it's via
    // WallClockExceeded (or AllParticlesDead).
    if ess_history.len() >= ESS_COLLAPSE_WINDOWS {
        let tail = &ess_history[ess_history.len() - ESS_COLLAPSE_WINDOWS..];
        if tail.iter().all(|&ess| ess <= ESS_FLOOR) {
            return Some(PFDegenerateKind::EssCollapsed {
                last_ess: tail.to_vec(),
            });
        }
    }

    None
}

/// gh#147 (M3.1). The deterministic substep cost of propagating one
/// observation window: `n_particles · ceil((obs_time − t)/dt)`.
///
/// This is a closed-form scalar over values already in scope at the top
/// of the per-window loop — NOT a per-particle reduction — so it is
/// identical regardless of thread count (`--parallel`-invariant by
/// construction). The `ceil` matches the per-particle substep loop in
/// `bootstrap_filter`/`run_if2` (`while t_local < obs_time − ε`, last
/// step clamped), so accumulating this per window equals the true total
/// substep count up to the ε boundary (a ≤1-step-per-window slack that is
/// irrelevant against a 1e10 budget).
///
/// Degenerate `dt`: a non-positive or non-finite `dt` is itself a wedge
/// (the substep loop would never advance), so it reports `u64::MAX` to
/// trip the budget rather than silently returning 0. A window with
/// `obs_time <= t` does no substeps and costs 0.
pub fn window_substep_cost(n_particles: usize, t: f64, obs_time: f64, dt: f64) -> u64 {
    if !dt.is_finite() || dt <= 0.0 {
        return u64::MAX;
    }
    let span = obs_time - t;
    // No work when the window has already passed (`span <= 0`). A NaN span
    // (a broken obs schedule) also costs 0 — the substep loop's
    // `t_local < obs_time` guard runs zero iterations on a NaN bound, so
    // matching that here keeps the budget honest rather than misreporting a
    // schedule bug as a compute blow-up.
    if span <= 0.0 || span.is_nan() {
        return 0;
    }
    let substeps = (span / dt).ceil();
    if !substeps.is_finite() || substeps < 0.0 {
        return u64::MAX;
    }
    // Saturating throughout: an astronomically small dt can overflow u64,
    // which is itself a wedge → saturate to MAX and let the budget trip.
    let substeps = if substeps >= u64::MAX as f64 {
        u64::MAX
    } else {
        substeps as u64
    };
    (n_particles as u64).saturating_mul(substeps)
}

/// gh#147 (M3.1). Deterministic cumulative-substep budget check. Given the
/// substeps already executed (`iters`) and the projected `cost` of the
/// next window (from [`window_substep_cost`]), return
/// `Some(IterationBudgetExceeded)` iff propagating that window would push
/// the cumulative count *past* `budget` (strictly greater — exactly
/// hitting the budget is allowed). `None` means "proceed". Pure: no clock,
/// no env, no thread state — the same inputs always give the same answer.
pub fn check_iteration_budget(iters: u64, cost: u64, budget: u64) -> Option<PFDegenerateKind> {
    let attempted = iters.saturating_add(cost);
    if attempted > budget {
        Some(PFDegenerateKind::IterationBudgetExceeded {
            attempted_substeps: attempted,
            budget_substeps: budget,
        })
    } else {
        None
    }
}

/// gh#133/gh#147. Map a watchdog bail (the `kind` from
/// [`check_pf_degeneracy`] or [`check_iteration_budget`]) to the right
/// `SimError`. Both `WallClockExceeded` and `IterationBudgetExceeded` are
/// *resource* limits and get their own error types
/// ([`SimError::PFWallclockTimeout`] / [`SimError::PFIterationBudget`]),
/// distinct from the statistical `PFDegenerate` pathologies
/// (EssCollapsed/AllParticlesDead). All are whole-call bails, so call-site
/// behaviour is preserved — only the type and message differ. The single
/// branch point keeps the two filter loops (`if2`, `bootstrap_filter`)
/// agreeing. `elapsed_s` is the wall-clock-relevant field; the
/// deterministic iteration budget ignores it (its diagnostic data — the
/// substep counts — rides on the `kind`).
pub fn pf_bail_error(kind: PFDegenerateKind, obs_window: usize, elapsed_s: f64) -> SimError {
    match kind {
        PFDegenerateKind::WallClockExceeded => {
            SimError::PFWallclockTimeout { obs_window, elapsed_s }
        }
        PFDegenerateKind::IterationBudgetExceeded { attempted_substeps, budget_substeps } => {
            SimError::PFIterationBudget { obs_window, attempted_substeps, budget_substeps }
        }
        other => SimError::PFDegenerate { kind: other, obs_window, elapsed_s },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Healthy run: ESS comfortably above the floor, fast wall-clock,
    /// no dead particles. Must NOT bail.
    #[test]
    fn healthy_run_returns_none() {
        let ess = vec![800.0, 750.0, 820.0, 790.0, 810.0];
        let elapsed = Duration::from_secs(5);
        assert!(check_pf_degeneracy(&ess, elapsed, 4, 0, 1000).is_none());
    }

    /// A single-window dip below the floor is normal during epidemic
    /// peaks. Must NOT trigger EssCollapsed.
    #[test]
    fn single_window_dip_does_not_trigger() {
        let ess = vec![800.0, 1.5, 750.0]; // mid-series dip
        assert!(check_pf_degeneracy(&ess, Duration::from_secs(1), 2, 0, 1000).is_none());
    }

    /// Two consecutive low windows still under K=3 threshold. Must NOT trigger.
    #[test]
    fn two_consecutive_low_windows_do_not_trigger() {
        let ess = vec![800.0, 1.5, 1.5];
        assert!(check_pf_degeneracy(&ess, Duration::from_secs(1), 2, 0, 1000).is_none());
    }

    /// K=3 consecutive windows at or below the floor → EssCollapsed
    /// with the K-window history attached.
    #[test]
    fn k_consecutive_low_windows_trigger_ess_collapsed() {
        let ess = vec![800.0, 1.8, 1.2, 1.5];
        let kind = check_pf_degeneracy(&ess, Duration::from_secs(1), 3, 0, 1000)
            .expect("should bail with ESS collapse");
        match kind {
            PFDegenerateKind::EssCollapsed { last_ess } => {
                assert_eq!(last_ess, vec![1.8, 1.2, 1.5]);
            }
            other => panic!("expected EssCollapsed, got {:?}", other),
        }
    }

    /// Boundary case: ESS exactly == ESS_FLOOR (= 2.0) counts as
    /// collapsed (the comparison is `<=`). One usable particle plus a
    /// thin margin is exactly the pathology we want to catch.
    #[test]
    fn ess_equal_to_floor_counts_as_collapsed() {
        let ess = vec![ESS_FLOOR, ESS_FLOOR, ESS_FLOOR];
        let kind = check_pf_degeneracy(&ess, Duration::from_secs(0), 2, 0, 1000)
            .expect("should bail with ESS at the floor");
        assert!(matches!(kind, PFDegenerateKind::EssCollapsed { .. }));
    }

    /// Wall-clock at or above the budget → WallClockExceeded, even with
    /// healthy ESS (the filter might be stuck in step()). Small swarm, so
    /// the floor (120 s) is the budget.
    #[test]
    fn wall_clock_timeout_triggers() {
        let ess = vec![800.0, 750.0]; // healthy
        let elapsed = Duration::from_secs(WALLCLOCK_FLOOR_S);
        let kind = check_pf_degeneracy(&ess, elapsed, 1, 0, 100)
            .expect("should bail on wall-clock");
        assert!(matches!(kind, PFDegenerateKind::WallClockExceeded));
    }

    /// Wall-clock just under the budget must NOT trigger (small swarm →
    /// floor budget).
    #[test]
    fn wall_clock_just_under_does_not_trigger() {
        let ess = vec![800.0];
        let elapsed = Duration::from_secs(WALLCLOCK_FLOOR_S - 1);
        assert!(check_pf_degeneracy(&ess, elapsed, 0, 0, 100).is_none());
    }

    /// gh#133. A large-but-healthy swarm is slow, not stuck — the budget
    /// must scale with particle count so it does not false-positive as
    /// WallClockExceeded just past the old fixed 120 s. (Repro: a 1500-
    /// particle IF2 fit killed at 120 s with uniform cross-chain progress.)
    #[test]
    fn large_swarm_slow_but_healthy_does_not_false_trigger() {
        let ess = vec![800.0, 750.0, 820.0]; // healthy ESS
        assert!(
            check_pf_degeneracy(&ess, Duration::from_secs(200), 50, 0, 1500).is_none(),
            "a slow-but-healthy 1500-particle filter must not be killed as degenerate"
        );
    }

    /// All particles dead → AllParticlesDead, even with no ESS
    /// history and zero wall-clock.
    #[test]
    fn all_particles_dead_triggers() {
        let ess: Vec<f64> = vec![];
        let kind = check_pf_degeneracy(&ess, Duration::from_secs(0), 0, 1000, 1000)
            .expect("should bail with AllParticlesDead");
        assert!(matches!(kind, PFDegenerateKind::AllParticlesDead));
    }

    /// AllParticlesDead has priority over ESS collapse: when every
    /// particle is dead, the K-window history is irrelevant — the
    /// more specific diagnostic wins.
    #[test]
    fn all_particles_dead_wins_over_ess_collapse() {
        let ess = vec![0.0, 0.0, 0.0];
        let kind = check_pf_degeneracy(&ess, Duration::from_secs(0), 2, 500, 500)
            .expect("should bail");
        assert!(matches!(kind, PFDegenerateKind::AllParticlesDead),
            "AllParticlesDead must take priority over EssCollapsed");
    }

    /// `dead_count == 0` with `n_particles == 0` must NOT trigger
    /// AllParticlesDead (vacuous case — guards against the trivial
    /// `0 >= 0` true that would always fire on an empty swarm).
    #[test]
    fn empty_swarm_does_not_trigger_all_dead() {
        // No way an empty swarm should hit a watchdog — this guards
        // against an off-by-one that returns AllParticlesDead on
        // every call with n_particles=0.
        assert!(check_pf_degeneracy(&[], Duration::from_secs(0), 0, 0, 0).is_none());
    }

    /// Exactly ESS_COLLAPSE_WINDOWS-1 windows of history with all at
    /// floor: not enough history yet, must NOT trigger.
    #[test]
    fn insufficient_history_does_not_trigger() {
        assert!(ESS_COLLAPSE_WINDOWS >= 2,
            "test assumes K-window threshold >= 2");
        let short: Vec<f64> = (0..ESS_COLLAPSE_WINDOWS - 1).map(|_| 0.5).collect();
        assert!(check_pf_degeneracy(&short, Duration::from_secs(0), 0, 0, 1000).is_none());
    }

    /// gh#133. Budget resolution: scaled default, env disable, positive
    /// override, and unparseable-falls-through — tested without touching
    /// process env (the precedence is a pure function of its arg).
    #[test]
    fn wallclock_budget_resolution() {
        // Scaled default: small swarm floored at 120; large swarm scaled.
        assert_eq!(resolve_wallclock_budget_s(None, 100), Some(WALLCLOCK_FLOOR_S));
        assert_eq!(resolve_wallclock_budget_s(None, 1500), Some(750)); // 1500 * 0.5
        // Override `0` disables the wall-clock check.
        assert_eq!(resolve_wallclock_budget_s(Some("0"), 1500), None);
        // A positive override wins over the scaled default (trimmed).
        assert_eq!(resolve_wallclock_budget_s(Some(" 45 "), 1500), Some(45));
        // Unparseable override falls through to the scaled default — it
        // must NOT silently disable the guard.
        assert_eq!(resolve_wallclock_budget_s(Some("nonsense"), 1500), Some(750));
    }

    /// gh#133. The wall-clock bail maps to the resource-limit error type
    /// (`PFWallclockTimeout`), while the statistical pathologies stay under
    /// `PFDegenerate`.
    #[test]
    fn wallclock_bail_is_a_timeout_not_degeneracy() {
        assert!(matches!(
            pf_bail_error(PFDegenerateKind::WallClockExceeded, 5, 130.0),
            SimError::PFWallclockTimeout { .. }
        ));
        assert!(matches!(
            pf_bail_error(PFDegenerateKind::AllParticlesDead, 5, 1.0),
            SimError::PFDegenerate { .. }
        ));
        assert!(matches!(
            pf_bail_error(PFDegenerateKind::EssCollapsed { last_ess: vec![1.0] }, 5, 1.0),
            SimError::PFDegenerate { .. }
        ));
    }

    /// gh#147 (M3.1). The iteration-budget bail maps to its own
    /// resource-limit error (`PFIterationBudget`), carrying the substep
    /// counts forward from the kind — distinct from both the statistical
    /// `PFDegenerate` pathologies and the wall-clock timeout.
    #[test]
    fn iteration_budget_bail_is_its_own_resource_error() {
        let kind = PFDegenerateKind::IterationBudgetExceeded {
            attempted_substeps: 12_000_000_000,
            budget_substeps: ITER_BUDGET,
        };
        match pf_bail_error(kind, 7, 0.0) {
            SimError::PFIterationBudget { obs_window, attempted_substeps, budget_substeps } => {
                assert_eq!(obs_window, 7);
                assert_eq!(attempted_substeps, 12_000_000_000);
                assert_eq!(budget_substeps, ITER_BUDGET);
            }
            other => panic!("expected PFIterationBudget, got {:?}", other),
        }
    }

    /// gh#147 (M3.1). The per-window substep cost is exactly
    /// `n_particles · ceil((obs_time − t)/dt)`, matching the substep loop's
    /// iteration count (last step clamped → ceil, not floor).
    #[test]
    fn window_substep_cost_matches_ceil_of_span_over_dt() {
        // Exact division: span/dt = 2 → 2 substeps.
        assert_eq!(window_substep_cost(10, 0.0, 1.0, 0.5), 20);
        // Non-exact: span/dt = 3.33 → ceil = 4 (the final clamped step counts).
        assert_eq!(window_substep_cost(10, 0.0, 1.0, 0.3), 40);
        // dt == span → exactly 1 substep.
        assert_eq!(window_substep_cost(7, 0.0, 1.0, 1.0), 7);
        // Mid-trajectory window: span = obs_time − t.
        assert_eq!(window_substep_cost(3, 5.0, 6.0, 0.25), 12); // ceil(1/0.25)=4, ×3
    }

    /// gh#147 (M3.1). Degenerate inputs: a window that does no work costs
    /// 0; a non-positive / non-finite dt is itself a wedge and reports
    /// `u64::MAX` so the budget trips rather than silently passing.
    #[test]
    fn window_substep_cost_handles_degenerate_inputs() {
        // obs_time == t (and obs_time < t): no substeps.
        assert_eq!(window_substep_cost(100, 1.0, 1.0, 0.1), 0);
        assert_eq!(window_substep_cost(100, 2.0, 1.0, 0.1), 0);
        // Non-positive / non-finite dt → MAX (a wedge: the loop never advances).
        assert_eq!(window_substep_cost(100, 0.0, 1.0, 0.0), u64::MAX);
        assert_eq!(window_substep_cost(100, 0.0, 1.0, -0.1), u64::MAX);
        assert_eq!(window_substep_cost(100, 0.0, 1.0, f64::NAN), u64::MAX);
        // A sub-microsecond dt is a large but representable cost (no
        // overflow): 2000 × ceil(1/1e-9) = 2e12, well over ITER_BUDGET but
        // far under u64::MAX — it trips the budget without saturating.
        assert_eq!(window_substep_cost(2000, 0.0, 1.0, 1e-9), 2_000_000_000_000);
        // A dt so tiny the substep count × swarm overflows u64 → saturates
        // to MAX rather than wrapping to a small value that slips the budget.
        assert_eq!(window_substep_cost(2000, 0.0, 1.0, 1e-18), u64::MAX);
    }

    /// gh#147 (M3.1). The budget check fires strictly past the budget;
    /// exactly hitting it is allowed. `None` means "proceed".
    #[test]
    fn check_iteration_budget_fires_strictly_past_budget() {
        // Under budget → proceed.
        assert!(check_iteration_budget(0, 100, 1000).is_none());
        // Exactly at budget → proceed (boundary is inclusive).
        assert!(check_iteration_budget(900, 100, 1000).is_none());
        // One past budget → bail, carrying the projected total + budget.
        match check_iteration_budget(901, 100, 1000) {
            Some(PFDegenerateKind::IterationBudgetExceeded {
                attempted_substeps, budget_substeps,
            }) => {
                assert_eq!(attempted_substeps, 1001);
                assert_eq!(budget_substeps, 1000);
            }
            other => panic!("expected IterationBudgetExceeded, got {:?}", other),
        }
        // Saturating add: a MAX cost cannot wrap around the budget.
        assert!(check_iteration_budget(1, u64::MAX, ITER_BUDGET).is_some());
    }
}
