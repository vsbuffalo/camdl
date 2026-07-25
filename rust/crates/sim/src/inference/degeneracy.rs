//! gh#110/gh#147. Shared particle-filter watchdogs — both **deterministic**.
//!
//! `bootstrap_filter` and IF2's inner per-iteration PF loop both advance
//! particles, reweight, and resample at each observation window. Both have
//! the same implicit contract: *return a finite log-likelihood for any θ
//! within the parameter bounds, in bounded compute.* That contract can break
//! at bound-box extremes (σ at the upper bound, R₀ ≈ 50, …) where ESS
//! collapses to ~1, and under a misconfiguration (e.g. a sub-microsecond `dt`)
//! where the substep count explodes.
//!
//! Two watchdogs, each a **pure function of inputs** — no clock, no thread
//! state — so a content-addressed fit's log-likelihood is reproducible across
//! machines:
//!
//!   1. [`check_pf_degeneracy`] — the statistical pathologies (ESS collapse,
//!      all-particles-dead), read from the swarm state.
//!   2. [`window_substep_cost`] + [`check_iteration_budget`] — a cap on the
//!      cumulative particle-substep count (default [`ITER_BUDGET`]), the
//!      compute-blowup safety.
//!
//! The module also owns [`DeathMask`] — the per-particle recovery policy the
//! `dead_count` those watchdogs read is derived from. It lives here rather than
//! in either filter because both filters must apply the *same* policy; see its
//! own docs (gh#367).
//!
//! gh#241: this module previously also carried a **wall-clock** timeout
//! (`CAMDL_PF_WALLCLOCK_TIMEOUT_S` + a `--pf-wallclock-timeout` flag that set
//! that env var). A wall-clock budget made the log-likelihood depend on
//! machine speed — two runs with identical inputs could diverge (one
//! completes, one aborts) — which is fatal to content addressing and was an
//! un-typed input channel outside the CLI/TOML surface. It is **removed**: the
//! deterministic substep budget (2) already covered its only real job
//! (compute blow-up), aborting at the same point on every machine. The budget
//! is now a typed input (`--pf-max-substeps` / `SMCConfig::max_substeps`),
//! not an env var.
//!
//! `ESS_FLOOR = 2.0` over `ESS_COLLAPSE_WINDOWS = 3` consecutive obs windows
//! on a 200-obs SEIR fit means we bail only if the filter is producing one
//! usable particle across 1.5% of the observation series — generous against
//! legitimate peak-dynamics dips, tight against sustained collapse.

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

/// gh#147 (M3.1). Default budget on the cumulative number of
/// *particle-substeps* a single particle-filter evaluation may execute — the
/// content-addressing-safe compute-blowup guard, and the deterministic
/// replacement for the removed wall-clock watchdog (gh#241).
///
/// A particle filter never θ-wedges (it requires the chain-binomial backend,
/// whose per-window substep count `ceil(Δt/dt)` is deterministic and whose
/// per-substep work is θ-bounded), so "runs effectively forever" means
/// "executes pathologically many substeps". The deterministic analog of "too
/// many substeps" is a cap on the substep *count* itself — independent of how
/// fast each one runs.
///
/// Sizing: the national worst case is ~2000 particles × a 2-year horizon at
/// dt = 0.1 day ≈ 1.5e7 substeps for one filter pass. A larger-than-national
/// pass (5000 particles × 4 years × dt = 0.05 ≈ 1.5e8) is still two orders of
/// magnitude under this budget, while a genuine wedge (a sub-microsecond dt,
/// or a non-positive dt) overshoots it by many orders and is caught
/// immediately. 1e10 is the headroom point: generous enough never to
/// false-trip a legitimate fit, tight enough that a misconfiguration fails
/// fast instead of hanging.
///
/// This is the *default*; callers may override it via `SMCConfig::max_substeps`
/// / `IF2Config::max_substeps` (a typed input). It bounds a single PF
/// evaluation (one `bootstrap_filter` call; one IF2 iteration's inner filter),
/// so it is independent of how many IF2 iterations or chains a fit runs. Never
/// derived from core count, wall-clock, or any machine state.
pub const ITER_BUDGET: u64 = 10_000_000_000; // 1e10 particle-substeps

/// gh#audit-C5/C6, gh#367. The per-particle **death mask** — the single
/// definition of "one bad particle must not kill the whole evaluation" shared
/// by the two particle filters
/// ([`super::particle_filter::bootstrap_filter`] and
/// [`super::correlated_pf::bootstrap_filter_correlated`]).
///
/// The policy has three coupled parts, and they must not drift apart:
///
///   1. **Classify** ([`DeathMask::classify`]) — a particle that hits a
///      *per-particle-recoverable* `SimError`
///      ([`SimError::is_per_particle_recoverable`]: `NumericalCollapse`,
///      `NegativeCount{BinomialOvershoot}`, `NonFiniteParameter`,
///      `TableLookup`) is marked dead and stops propagating. **Every other**
///      error propagates and tears the evaluation down.
///   2. **Score** — a dead particle's log-weight is `−∞`, so systematic
///      resampling discards it (call sites read [`DeathMask::is_dead`]).
///   3. **Clear** ([`DeathMask::clear`]) — after resampling, every surviving
///      particle had finite weight, and resampling shuffles by index anyway,
///      so the mask is reset.
///
/// [`DeathMask::count`] feeds [`check_pf_degeneracy`]'s `AllParticlesDead`
/// branch: the limit case where the mask has nothing left to absorb.
///
/// gh#367: the correlated PF previously had **no** mask — it propagated any
/// particle's error out of the whole evaluation, which the PMMH driver maps to
/// a `−∞` log-likelihood for that θ (`fit/pmmh.rs`: `Err(e) if
/// e.is_structural() => Err(e), Err(_) => Ok(f64::NEG_INFINITY)`). One
/// particle's recoverable excursion therefore rejected the entire proposal,
/// silently biasing correlated PMMH against boundary regions where occasional
/// particle failure is expected. Routing both filters through this type is
/// what keeps the policy from being live in one filter and absent in the other.
pub struct DeathMask {
    dead: Vec<bool>,
}

impl DeathMask {
    /// A mask with every particle alive.
    pub fn new(n_particles: usize) -> Self {
        Self { dead: vec![false; n_particles] }
    }

    /// Classify ONE particle's propagation outcome, inside the per-particle
    /// (parallel) closure. `Ok(false)` = survived, keep stepping; `Ok(true)` =
    /// died recoverably, stop stepping this particle; `Err` = not recoverable,
    /// propagate out of the filter.
    ///
    /// Call sites read as `if DeathMask::classify(step(...))? { return
    /// Ok(true); }` — the `?` is the "tear the run down" branch, the `true` is
    /// the "kill this particle only" branch.
    pub fn classify(outcome: Result<(), SimError>) -> Result<bool, SimError> {
        match outcome {
            Ok(()) => Ok(false),
            Err(e) if e.is_per_particle_recoverable() => Ok(true),
            Err(e) => Err(e),
        }
    }

    /// Fold the swarm's per-particle outcomes (index-aligned with the swarm,
    /// each the return of a closure that used [`DeathMask::classify`]) into the
    /// mask. The first non-recoverable error propagates.
    pub fn absorb(&mut self, outcomes: Vec<Result<bool, SimError>>) -> Result<(), SimError> {
        assert_eq!(
            outcomes.len(), self.dead.len(),
            "death-mask outcomes must be index-aligned with the swarm",
        );
        for (i, r) in outcomes.into_iter().enumerate() {
            if r? {
                self.dead[i] = true;
            }
        }
        Ok(())
    }

    /// Did particle `i` die this observation window? Its log-weight must then
    /// be `−∞` rather than the observation model's score.
    pub fn is_dead(&self, i: usize) -> bool {
        self.dead[i]
    }

    /// Per-particle view, for zipping into a parallel iterator.
    pub fn as_slice(&self) -> &[bool] {
        &self.dead
    }

    /// Number of particles currently dead — the `dead_count` argument of
    /// [`check_pf_degeneracy`].
    pub fn count(&self) -> usize {
        self.dead.iter().filter(|&&d| d).count()
    }

    /// Every particle is dead: the limit case the mask cannot absorb, because
    /// the whole weight vector is `−∞` and resampling has nothing to select.
    /// Non-vacuous only for a non-empty swarm.
    pub fn all_dead(&self) -> bool {
        !self.dead.is_empty() && self.dead.iter().all(|&d| d)
    }

    /// Reset after resampling.
    pub fn clear(&mut self) {
        self.dead.fill(false);
    }
}

/// gh#110. Return the statistical degeneracy mode if the filter has bailed at
/// this observation window, otherwise `None`. Pure: a function of the swarm
/// state only (no clock, no env, no thread state).
///
/// Inputs:
/// - `ess_history` — every ESS recorded so far, length = obs_windows
///   processed. Only the last `ESS_COLLAPSE_WINDOWS` are inspected.
/// - `dead_count` — number of particles currently marked dead. Used only for
///   the `AllParticlesDead` check; pass 0 when the caller doesn't track
///   per-particle death (e.g. IF2's inner loop, since its `process.step`
///   errors propagate immediately).
/// - `n_particles` — total particles in the swarm.
///
/// Discrimination order (deterministic on tie cases):
///   1. `AllParticlesDead` — the limit case of ESS collapse, cheap and
///      diagnostically distinct; check first.
///   2. `EssCollapsed` — requires `ESS_COLLAPSE_WINDOWS` history.
pub fn check_pf_degeneracy(
    ess_history: &[f64],
    dead_count: usize,
    n_particles: usize,
) -> Option<PFDegenerateKind> {
    // AllParticlesDead: every particle hit a per-particle-recoverable
    // error. Resampling on the next step would have zero weight
    // everywhere; bail before the divide-by-zero.
    if n_particles > 0 && dead_count >= n_particles {
        return Some(PFDegenerateKind::AllParticlesDead);
    }

    // EssCollapsed: K consecutive obs windows at or below the floor.
    // Single-window dips during epidemic peaks are not pathology;
    // sustained collapse is. We need at least K windows of history
    // before this can fire.
    if ess_history.len() >= ESS_COLLAPSE_WINDOWS {
        let tail = &ess_history[ess_history.len() - ESS_COLLAPSE_WINDOWS..];
        if tail.iter().all(|&ess| ess <= ESS_FLOOR) {
            return Some(PFDegenerateKind::EssCollapsed { last_ess: tail.to_vec() });
        }
    }

    None
}

/// gh#147 (M3.1). The deterministic substep cost of propagating one
/// observation window: `n_particles · ceil((obs_time − t)/dt)`.
///
/// This is a closed-form scalar over values already in scope at the top of the
/// per-window loop — NOT a per-particle reduction — so it is identical
/// regardless of thread count (`--parallel`-invariant by construction). The
/// `ceil` matches the per-particle substep loop in `bootstrap_filter`/`run_if2`
/// (`while t_local < obs_time − ε`, last step clamped), so accumulating this
/// per window equals the true total substep count up to the ε boundary (a
/// ≤1-step-per-window slack that is irrelevant against a 1e10 budget).
///
/// Degenerate `dt`: a non-positive or non-finite `dt` is itself a wedge (the
/// substep loop would never advance), so it reports `u64::MAX` to trip the
/// budget rather than silently returning 0. A window with `obs_time <= t` does
/// no substeps and costs 0.
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
/// substeps already executed (`iters`) and the projected `cost` of the next
/// window (from [`window_substep_cost`]), return
/// `Some(IterationBudgetExceeded)` iff propagating that window would push the
/// cumulative count *past* `budget` (strictly greater — exactly hitting the
/// budget is allowed). `None` means "proceed". Pure: no clock, no env, no
/// thread state — the same inputs always give the same answer.
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

/// gh#147. Map a watchdog bail (the `kind` from [`check_pf_degeneracy`] or
/// [`check_iteration_budget`]) to the right `SimError`.
/// `IterationBudgetExceeded` is a *resource* limit and gets its own error type
/// ([`SimError::PFIterationBudget`]), distinct from the statistical
/// `PFDegenerate` pathologies (EssCollapsed/AllParticlesDead). All are
/// whole-call bails, so call-site behaviour is preserved — only the type and
/// message differ. `elapsed_s` is a display-only diagnostic on the statistical
/// bail (how long the doomed call ran); the deterministic iteration budget
/// ignores it (its diagnostic — the substep counts — rides on the `kind`).
pub fn pf_bail_error(kind: PFDegenerateKind, obs_window: usize, elapsed_s: f64) -> SimError {
    match kind {
        PFDegenerateKind::IterationBudgetExceeded { attempted_substeps, budget_substeps } => {
            SimError::PFIterationBudget { obs_window, attempted_substeps, budget_substeps }
        }
        other => SimError::PFDegenerate { kind: other, obs_window, elapsed_s },
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::{CollapseKind, NegativeCountCause};

    /// gh#367. `classify` is the single definition of the recovery policy: a
    /// per-particle-recoverable error kills the particle (`Ok(true)`); a
    /// structural one propagates so the caller tears the evaluation down.
    #[test]
    fn classify_kills_recoverable_and_propagates_structural() {
        assert!(!DeathMask::classify(Ok(())).unwrap(), "a clean step keeps the particle alive");

        let recoverable = SimError::NumericalCollapse { kind: CollapseKind::DivByZero, t: 1.0 };
        assert!(recoverable.is_per_particle_recoverable());
        assert!(DeathMask::classify(Err(recoverable)).unwrap());

        let overshoot = SimError::NegativeCount {
            compartment: "S".into(), attempted_value: -1, t: 1.0,
            cause: NegativeCountCause::BinomialOvershoot,
        };
        assert!(DeathMask::classify(Err(overshoot)).unwrap());

        // Structural: must NOT be absorbed by the mask.
        let structural = SimError::Validation("model cannot run".into());
        assert!(structural.is_structural());
        assert!(matches!(
            DeathMask::classify(Err(structural)),
            Err(SimError::Validation(_)),
        ));

        // Neither recoverable nor structural (a per-θ excursion): also NOT
        // absorbed — only `is_per_particle_recoverable` opens the mask.
        let neg_rate = SimError::NegativePropensity {
            transition: "infection".into(), value: -1.0, t: 2.0,
        };
        assert!(!neg_rate.is_per_particle_recoverable());
        assert!(matches!(
            DeathMask::classify(Err(neg_rate)),
            Err(SimError::NegativePropensity { .. }),
        ));
    }

    /// gh#367. `absorb` folds per-particle outcomes index-aligned, and the
    /// first non-recoverable error propagates instead of being folded.
    #[test]
    fn absorb_folds_index_aligned_and_propagates_errors() {
        let mut mask = DeathMask::new(4);
        mask.absorb(vec![Ok(false), Ok(true), Ok(false), Ok(true)]).unwrap();
        assert_eq!(mask.as_slice(), &[false, true, false, true]);
        assert_eq!(mask.count(), 2);
        assert!(!mask.all_dead());
        assert!(mask.is_dead(1) && !mask.is_dead(0));

        // Deaths accumulate across windows until cleared.
        mask.absorb(vec![Ok(true), Ok(false), Ok(false), Ok(false)]).unwrap();
        assert_eq!(mask.as_slice(), &[true, true, false, true]);

        mask.clear();
        assert_eq!(mask.count(), 0);
        assert!(!mask.all_dead(), "a cleared mask has no dead particles");

        let err = mask
            .absorb(vec![Ok(false), Err(SimError::Validation("boom".into())), Ok(true), Ok(false)])
            .expect_err("a structural error must propagate out of absorb");
        assert!(matches!(err, SimError::Validation(_)));
    }

    /// `all_dead` is the limit case the mask cannot absorb. Vacuously false on
    /// an empty swarm (matching `check_pf_degeneracy`'s `n_particles > 0` guard).
    #[test]
    fn all_dead_only_when_every_particle_is_dead() {
        let mut mask = DeathMask::new(3);
        mask.absorb(vec![Ok(true), Ok(true), Ok(false)]).unwrap();
        assert!(!mask.all_dead());
        mask.absorb(vec![Ok(false), Ok(false), Ok(true)]).unwrap();
        assert!(mask.all_dead());
        assert_eq!(mask.count(), 3);

        assert!(!DeathMask::new(0).all_dead(), "empty swarm is not 'all dead'");
    }

    /// Healthy run: ESS comfortably above the floor, no dead particles.
    /// Must NOT bail.
    #[test]
    fn healthy_run_returns_none() {
        let ess = vec![800.0, 750.0, 820.0, 790.0, 810.0];
        assert!(check_pf_degeneracy(&ess, 0, 1000).is_none());
    }

    /// A single-window dip below the floor is normal during epidemic
    /// peaks. Must NOT trigger EssCollapsed.
    #[test]
    fn single_window_dip_does_not_trigger() {
        let ess = vec![800.0, 1.5, 750.0]; // mid-series dip
        assert!(check_pf_degeneracy(&ess, 0, 1000).is_none());
    }

    /// Two consecutive low windows still under K=3 threshold. Must NOT trigger.
    #[test]
    fn two_consecutive_low_windows_do_not_trigger() {
        let ess = vec![800.0, 1.5, 1.5];
        assert!(check_pf_degeneracy(&ess, 0, 1000).is_none());
    }

    /// K=3 consecutive windows at or below the floor → EssCollapsed
    /// with the K-window history attached.
    #[test]
    fn k_consecutive_low_windows_trigger_ess_collapsed() {
        let ess = vec![800.0, 1.8, 1.2, 1.5];
        let kind = check_pf_degeneracy(&ess, 0, 1000).expect("should bail with ESS collapse");
        match kind {
            PFDegenerateKind::EssCollapsed { last_ess } => {
                assert_eq!(last_ess, vec![1.8, 1.2, 1.5]);
            }
            other => panic!("expected EssCollapsed, got {:?}", other),
        }
    }

    /// Boundary case: ESS exactly == ESS_FLOOR (= 2.0) counts as
    /// collapsed (the comparison is `<=`).
    #[test]
    fn ess_equal_to_floor_counts_as_collapsed() {
        let ess = vec![ESS_FLOOR, ESS_FLOOR, ESS_FLOOR];
        let kind = check_pf_degeneracy(&ess, 0, 1000).expect("should bail with ESS at the floor");
        assert!(matches!(kind, PFDegenerateKind::EssCollapsed { .. }));
    }

    /// All particles dead → AllParticlesDead, even with no ESS history.
    #[test]
    fn all_particles_dead_triggers() {
        let ess: Vec<f64> = vec![];
        let kind = check_pf_degeneracy(&ess, 1000, 1000).expect("should bail with AllParticlesDead");
        assert!(matches!(kind, PFDegenerateKind::AllParticlesDead));
    }

    /// AllParticlesDead has priority over ESS collapse: when every
    /// particle is dead, the K-window history is irrelevant.
    #[test]
    fn all_particles_dead_wins_over_ess_collapse() {
        let ess = vec![0.0, 0.0, 0.0];
        let kind = check_pf_degeneracy(&ess, 500, 500).expect("should bail");
        assert!(
            matches!(kind, PFDegenerateKind::AllParticlesDead),
            "AllParticlesDead must take priority over EssCollapsed"
        );
    }

    /// `dead_count == 0` with `n_particles == 0` must NOT trigger
    /// AllParticlesDead (vacuous case — guards the trivial `0 >= 0`).
    #[test]
    fn empty_swarm_does_not_trigger_all_dead() {
        assert!(check_pf_degeneracy(&[], 0, 0).is_none());
    }

    /// Exactly ESS_COLLAPSE_WINDOWS-1 windows of history at floor: not
    /// enough history yet, must NOT trigger.
    #[test]
    fn insufficient_history_does_not_trigger() {
        assert!(ESS_COLLAPSE_WINDOWS >= 2, "test assumes K-window threshold >= 2");
        let short: Vec<f64> = (0..ESS_COLLAPSE_WINDOWS - 1).map(|_| 0.5).collect();
        assert!(check_pf_degeneracy(&short, 0, 1000).is_none());
    }

    /// gh#147 (M3.1). The iteration-budget bail maps to its own
    /// resource-limit error (`PFIterationBudget`), carrying the substep counts
    /// forward — distinct from the statistical `PFDegenerate` pathologies.
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

    /// The statistical bail maps to `PFDegenerate` (not a resource error).
    #[test]
    fn statistical_bail_maps_to_pf_degenerate() {
        assert!(matches!(
            pf_bail_error(PFDegenerateKind::AllParticlesDead, 5, 1.0),
            SimError::PFDegenerate { .. }
        ));
        assert!(matches!(
            pf_bail_error(PFDegenerateKind::EssCollapsed { last_ess: vec![1.0] }, 5, 1.0),
            SimError::PFDegenerate { .. }
        ));
    }

    /// gh#147 (M3.1). The per-window substep cost is exactly
    /// `n_particles · ceil((obs_time − t)/dt)`, matching the substep loop's
    /// iteration count (last step clamped → ceil, not floor).
    #[test]
    fn window_substep_cost_matches_ceil_of_span_over_dt() {
        assert_eq!(window_substep_cost(10, 0.0, 1.0, 0.5), 20);
        assert_eq!(window_substep_cost(10, 0.0, 1.0, 0.3), 40);
        assert_eq!(window_substep_cost(7, 0.0, 1.0, 1.0), 7);
        assert_eq!(window_substep_cost(3, 5.0, 6.0, 0.25), 12);
    }

    /// gh#147 (M3.1). Degenerate inputs: a window that does no work costs 0; a
    /// non-positive / non-finite dt reports `u64::MAX` so the budget trips.
    #[test]
    fn window_substep_cost_handles_degenerate_inputs() {
        assert_eq!(window_substep_cost(100, 1.0, 1.0, 0.1), 0);
        assert_eq!(window_substep_cost(100, 2.0, 1.0, 0.1), 0);
        assert_eq!(window_substep_cost(100, 0.0, 1.0, 0.0), u64::MAX);
        assert_eq!(window_substep_cost(100, 0.0, 1.0, -0.1), u64::MAX);
        assert_eq!(window_substep_cost(100, 0.0, 1.0, f64::NAN), u64::MAX);
        assert_eq!(window_substep_cost(2000, 0.0, 1.0, 1e-9), 2_000_000_000_000);
        assert_eq!(window_substep_cost(2000, 0.0, 1.0, 1e-18), u64::MAX);
    }

    /// gh#147 (M3.1). The budget check fires strictly past the budget;
    /// exactly hitting it is allowed. `None` means "proceed".
    #[test]
    fn check_iteration_budget_fires_strictly_past_budget() {
        assert!(check_iteration_budget(0, 100, 1000).is_none());
        assert!(check_iteration_budget(900, 100, 1000).is_none());
        match check_iteration_budget(901, 100, 1000) {
            Some(PFDegenerateKind::IterationBudgetExceeded { attempted_substeps, budget_substeps }) => {
                assert_eq!(attempted_substeps, 1001);
                assert_eq!(budget_substeps, 1000);
            }
            other => panic!("expected IterationBudgetExceeded, got {:?}", other),
        }
        assert!(check_iteration_budget(1, u64::MAX, ITER_BUDGET).is_some());
    }
}
