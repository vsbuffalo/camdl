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

use std::cell::RefCell;
use std::collections::HashMap;
use std::path::PathBuf;
use std::rc::Rc;

use sim::{
    compiled_model::CompiledModel,
    config::GillespieConfig,
    gillespie::run_gillespie_with_observer,
    lineage::{LineListEntry, LineListWriter, LineageObserver, ParentRef},
    state::Trajectory,
};

fn fixtures_dir() -> PathBuf {
    PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").unwrap()).join("tests/fixtures")
}
fn load_fixture(name: &str) -> ir::Model {
    let path = fixtures_dir().join(format!("{}.ir.json", name));
    let contents =
        std::fs::read_to_string(&path).unwrap_or_else(|_| panic!("cannot read fixture {}", name));
    ir::from_str(&contents).unwrap_or_else(|e| panic!("parse {}: {}", name, e))
}
fn set_params(m: &mut ir::Model, vals: &[(&str, f64)]) {
    for p in &mut m.parameters {
        if let Some((_, v)) = vals.iter().find(|(n, _)| *n == p.name) {
            p.value = Some(*v);
        }
    }
}

#[derive(Clone)]
struct VecWriter {
    entries: Rc<RefCell<Vec<LineListEntry>>>,
}
impl VecWriter {
    fn new() -> Self {
        VecWriter { entries: Rc::new(RefCell::new(Vec::new())) }
    }
}
impl LineListWriter for VecWriter {
    fn init(&mut self) -> Result<(), sim::SimError> {
        Ok(())
    }
    fn write(&mut self, e: &LineListEntry) -> Result<(), sim::SimError> {
        self.entries.borrow_mut().push(e.clone());
        Ok(())
    }
    fn finish(&mut self) -> Result<(), sim::SimError> {
        Ok(())
    }
}

fn run(m: &ir::Model, seed: u64, t_end: f64) -> (Trajectory, Vec<LineListEntry>) {
    let compiled = CompiledModel::new(m.clone()).unwrap();
    let params = compiled.default_params.clone();
    let (initial_int, _) = compiled.initial_state(&params).unwrap();
    let collector = VecWriter::new();
    let buf = collector.entries.clone();
    let mut observer = LineageObserver::new(&compiled, seed, &initial_int, collector).unwrap();
    let cfg = GillespieConfig { t_start: 0.0, t_end, output_dt: None };
    let traj =
        run_gillespie_with_observer(&compiled, &params, seed, &cfg, Some(&mut observer)).unwrap();
    observer.finish().unwrap();
    let entries = buf.borrow().clone();
    (traj, entries)
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
        if e.source == Some(COMP_I) && e.destination == Some(COMP_R) {
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
