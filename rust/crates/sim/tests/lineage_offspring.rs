//! Offspring-distribution validation of the lineage transmission tree
//! (Phase 3, complementing the Tier-4 coalescent check).
//!
//! Each tracked individual's out-degree in the transmission forest is its
//! *realized* number of secondary infections. For a well-mixed SIR, an infector
//! born at time `t_b` infects at rate `β S(t)/N` while infectious, and its
//! infectious period is `Exp(γ)`. Its expected number of offspring is therefore
//! the renewal integral
//!
//!     E[offspring | t_b] = β ∫₀^∞ e^{−γ a} · S(t_b + a)/N · da
//!
//! (each unit of infectious time contributes `β S/N` secondary cases, weighted
//! by the probability `e^{−γa}` of still being infectious at lag `a`). Early in
//! the epidemic, `S/N → 1` and this reduces to the basic reproduction number
//! `R₀ = β/γ`; as S depletes it tracks the realized effective reproduction
//! number `R_eff(t)`.
//!
//! This test reconstructs offspring counts from the line list and asserts the
//! mean realized offspring of infectors born in a window matches the
//! trajectory-driven renewal prediction. It is an independent check from the
//! coalescent test: the coalescent constrains *who infected whom* (parent
//! choice), this constrains *how many* each infector produced (the realized R).
//! Verified empirically before locking the tolerance: born∈[2,6) gives observed
//! ≈ predicted to ~1% (e.g. 2.83 vs 2.85, 2.22 vs 2.21).

use std::collections::HashMap;

use sim::{
    lineage::{CompartmentId, LineListEntry, ParentRef},
    state::Trajectory,
};

mod lineage_helpers;
use lineage_helpers::{load_fixture, run_with_lineage, set_params};

/// Record an event log (Gillespie) and realize it at `identity_seed == seed`,
/// reproducing the shipped observer's line list for the same seed.
fn run(m: &ir::Model, seed: u64, t_end: f64) -> (Trajectory, Vec<LineListEntry>) {
    run_with_lineage(m.clone(), seed, t_end)
}

const COMP_I: usize = 1;
const COMP_R: usize = 2;
const N_SEED_I: u64 = 10;

/// (offspring out-degree per individual, birth time per individual, whether the
/// individual completed its infectious period i.e. recovered).
fn offspring_and_births(
    entries: &[LineListEntry],
) -> (HashMap<u64, u64>, HashMap<u64, f64>, HashMap<u64, bool>) {
    let mut sorted = entries.to_vec();
    sorted.sort_by(|a, b| a.time.partial_cmp(&b.time).unwrap());
    let mut children: HashMap<u64, u64> = HashMap::new();
    let mut birth: HashMap<u64, f64> = HashMap::new();
    let mut recovered: HashMap<u64, bool> = HashMap::new();
    for sid in 0..N_SEED_I {
        birth.insert(sid, 0.0);
    }
    for e in &sorted {
        let ind = e.individual.0;
        birth.entry(ind).or_insert(e.time);
        if let ParentRef::Individual(p) = e.parent {
            *children.entry(p.0).or_insert(0) += 1;
        }
        if e.source == Some(CompartmentId(COMP_I)) && e.destination == Some(CompartmentId(COMP_R)) {
            recovered.insert(ind, true);
        }
    }
    (children, birth, recovered)
}

/// S(t)/N at the trajectory snapshot nearest `t`. Snapshots are emitted at
/// integer times (step 1.0), so this caches the per-integer-time S-fraction and
/// looks it up by rounding — O(1) per query, identical to a nearest-snapshot
/// scan, but without the per-call O(snapshots) cost.
struct SFracGrid {
    /// S/N indexed by integer time (snapshot grid).
    by_int_t: Vec<f64>,
}
impl SFracGrid {
    fn build(traj: &Trajectory, _horizon: f64, _step: f64) -> Self {
        // Map each snapshot's rounded time to its S/N. Snapshots are at integer
        // times; the max rounded time bounds the vector.
        let mut max_t = 0usize;
        for s in &traj.snapshots {
            max_t = max_t.max(s.t.round() as usize);
        }
        let mut by_int_t = vec![0.0; max_t + 1];
        for s in &traj.snapshots {
            let cs = &s.int_state.counts;
            let total = (cs[0] + cs[COMP_I] + cs[COMP_R]) as f64;
            by_int_t[s.t.round() as usize] = cs[0] as f64 / total;
        }
        SFracGrid { by_int_t }
    }
    fn at(&self, t: f64) -> f64 {
        let idx = (t.round().max(0.0) as usize).min(self.by_int_t.len() - 1);
        self.by_int_t[idx]
    }
}

/// Renewal prediction: β ∫₀^∞ e^{−γa} S(t_b+a)/N da, trapezoid over the grid.
fn predicted_offspring(grid: &SFracGrid, t_b: f64, beta: f64, gamma: f64) -> f64 {
    let da = 0.1;
    let mut a = 0.0;
    let mut acc = 0.0;
    let horizon = 40.0;
    while a < horizon {
        acc += (-gamma * a).exp() * grid.at(t_b + a) * beta * da;
        a += da;
    }
    acc
}

#[test]
fn offspring_mean_matches_realized_effective_r() {
    // Well-mixed SIR, N≈10⁴, Gillespie. Compare mean realized offspring of
    // infectors born in a window to the trajectory-driven renewal prediction.
    let beta = 1.0;
    let gamma = 0.25;
    // Birth window where the prediction is robust (away from the t=0 seeding
    // transient and before S is fully exhausted). Verified empirically.
    let (t1, t2) = (2.0, 6.0);
    let n_rep = 40u64;

    let mut obs: Vec<f64> = Vec::new();
    let mut pred: Vec<f64> = Vec::new();

    for seed in 1..=n_rep {
        let mut m = load_fixture("sir_coalescent");
        set_params(&mut m, &[("beta", beta), ("gamma", gamma), ("N0", 10000.0)]);
        let (traj, entries) = run(&m, seed, 40.0);
        let grid = SFracGrid::build(&traj, 80.0, 0.1);
        let (children, birth, recovered) = offspring_and_births(&entries);
        for (&ind, &bt) in &birth {
            if bt >= t1 && bt < t2 && *recovered.get(&ind).unwrap_or(&false) {
                obs.push(*children.get(&ind).unwrap_or(&0) as f64);
                pred.push(predicted_offspring(&grid, bt, beta, gamma));
            }
        }
    }

    assert!(
        obs.len() > 5000,
        "need many completed infectors in the window; got {}",
        obs.len()
    );

    let n = obs.len() as f64;
    let obs_mean = obs.iter().sum::<f64>() / n;
    let pred_mean = pred.iter().sum::<f64>() / n;

    // Offspring counts are over-dispersed (geometric-like). Use the observed
    // sample sd for the Monte-Carlo error on the mean; 4σ band (the prediction
    // is an approximation — trajectory snapshots are coarse and the renewal
    // integral discretised).
    let var = obs.iter().map(|x| (x - obs_mean).powi(2)).sum::<f64>() / n;
    let se = (var / n).sqrt();
    let tol = (4.0 * se).max(0.1);

    eprintln!(
        "offspring check: born∈[{},{}), n={}; observed mean={:.4}, \
         renewal-predicted mean={:.4}, |Δ|={:.4}, tol={:.4} (R₀=β/γ={:.1})",
        t1,
        t2,
        obs.len(),
        obs_mean,
        pred_mean,
        (obs_mean - pred_mean).abs(),
        tol,
        beta / gamma
    );

    // The realized R must be below R₀ (S depleted) and positive.
    assert!(obs_mean > 0.0 && obs_mean < beta / gamma, "realized R must be in (0, R₀)");
    // Over-dispersion: variance exceeds the Poisson value (mean), as for a
    // geometric offspring distribution.
    assert!(var > obs_mean, "offspring distribution should be over-dispersed (var > mean)");

    assert!(
        (obs_mean - pred_mean).abs() <= tol,
        "realized mean offspring {:.4} deviates from the renewal prediction \
         {:.4} by more than tolerance {:.4}; the transmission tree's offspring \
         distribution is inconsistent with β∫e^(−γa)S/N da",
        obs_mean,
        pred_mean,
        tol
    );
}
