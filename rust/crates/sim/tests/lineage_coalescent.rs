//! Tier 4 — large-N coalescent-limit validation of the lineage transmission
//! tree (2026-05-19 individual-sampling-layer proposal, §"Validation").
//!
//! Under homogeneous (well-mixed) mixing, the SIR transmission tree converges
//! to the structured-coalescent prediction (Volz 2009, "Phylodynamics of
//! infectious disease epidemics," Genetics 183:1421–1430; see also Volz 2012,
//! Genetics 190:187 for the general well-mixed rate). The pairwise coalescent
//! rate for two lineages both in the infected class, with frequency-dependent
//! transmission `f(t) = β S(t) I(t) / N(t)` and uniform-within-pool infector
//! choice, is
//!
//!     λ_pair(t) = 2 · f(t) / I(t)²
//!
//! and for `k` extant lineages the total coalescent rate is
//! `λ_k = C(k,2) · 2 f / I²`. The waiting time (going back from a reference
//! time `t*` where S, I, N are ~constant over the short interval) until the
//! first coalescence among `k` sampled lineages is therefore
//! `Exp(C(k,2) · 2 f / I²)`, with mean `1 / (C(k,2) · 2 f / I²)`.
//!
//! NOTE ON THE PROPOSAL'S WRITTEN FORMULA. The proposal's prose states the
//! rate as `Exp(C(k,2) · 2 β S I / N²)`. That is dimensionally the *transmission*
//! intensity, not the per-pair coalescent rate: dividing the transmission rate
//! `f = βSI/N` by `I²` (the number of ordered pairs, for the 2/I² parent×child
//! probability) gives `2βS/(N I)`, which differs from `2βSI/N²` by a factor
//! `I²/N`. The correct per-pair rate is `2f/I²`. This was verified empirically
//! before writing the test: at the epidemic peak (S≈3.7k, I≈3.7k, N=10010,
//! β=1) the observed mean first-coalescence interval for k=200 lineages is
//! 0.249, matching `1/(C(200,2)·2f/I²) = 0.252` to ~1.5%, whereas the
//! proposal's `2βSI/N²` would predict a per-pair rate ~1400× too large. The
//! discrepancy is flagged here and in the Phase-3 report; the implementation
//! and this test use the verified `2f/I²` form.
//!
//! Population N ≥ 10⁴ (the diffusion approximation has O(1/N) bias that makes
//! the test flaky at smaller N — proposal's stated threshold).

use std::cell::RefCell;
use std::collections::{HashMap, HashSet};
use std::path::PathBuf;
use std::rc::Rc;

use sim::{
    compiled_model::CompiledModel,
    config::GillespieConfig,
    gillespie::run_gillespie_with_observer,
    lineage::{LineListEntry, LineListWriter, LineageObserver, ParentRef},
    rng::StatefulRng,
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

/// Compartment ids in sir_coalescent: S=0, I=1, R=2.
const COMP_S: usize = 0;
const COMP_I: usize = 1;
const COMP_R: usize = 2;
/// Initial infecteds seeded at t=0 (sir_coalescent ICs: I=10).
const N_SEED_I: u64 = 10;

/// Build the parent map and birth times from a line list, plus the set of
/// individuals live in the I pool at `t_star`.
fn build(entries: &[LineListEntry], t_star: f64) -> (HashMap<u64, u64>, HashMap<u64, f64>, Vec<u64>) {
    let mut sorted = entries.to_vec();
    sorted.sort_by(|a, b| a.time.partial_cmp(&b.time).unwrap());

    let mut parent: HashMap<u64, u64> = HashMap::new();
    let mut birth: HashMap<u64, f64> = HashMap::new();
    let mut in_i: HashSet<u64> = HashSet::new();
    // Seed infecteds (ids 0..N_SEED_I) are born at t=0, in I.
    for sid in 0..N_SEED_I {
        birth.insert(sid, 0.0);
        in_i.insert(sid);
    }

    for e in &sorted {
        let ind = e.individual.0;
        birth.entry(ind).or_insert(e.time);
        if let ParentRef::Individual(p) = e.parent {
            parent.entry(ind).or_insert(p.0);
        }
        if e.time <= t_star {
            if e.source == Some(COMP_I) {
                in_i.remove(&ind);
            }
            if e.destination == Some(COMP_I) {
                in_i.insert(ind);
            }
        }
    }
    let live: Vec<u64> = in_i.into_iter().collect();
    (parent, birth, live)
}

/// The first coalescent interval (time back from `t_star`) for `k` lineages
/// sampled from `live`. Returns `None` if fewer than `k` live or no coalescence
/// occurs before all lineages reach a root.
fn first_coalescent_interval(
    parent: &HashMap<u64, u64>,
    birth: &HashMap<u64, f64>,
    live: &[u64],
    k: usize,
    t_star: f64,
    rng: &mut StatefulRng,
) -> Option<f64> {
    if live.len() < k {
        return None;
    }
    // Sample k distinct lineages (Fisher-Yates partial shuffle).
    let mut pool = live.to_vec();
    let n = pool.len();
    let mut chosen: Vec<u64> = Vec::with_capacity(k);
    for i in 0..k {
        let j = i + (rng.uniform() * (n - i) as f64) as usize;
        let j = j.min(n - 1);
        pool.swap(i, j);
        chosen.push(pool[i]);
    }

    // Walk ancestry backward from t_star, collapsing the lineage with the most
    // recent birth into its parent, until two lineages share an ancestor
    // (coalescence) or all reach roots.
    let mut cnt: HashMap<u64, usize> = HashMap::new();
    for &c in &chosen {
        *cnt.entry(c).or_insert(0) += 1;
    }
    // If sampling already produced a duplicate (cannot, distinct) skip.
    let mut tau = t_star;
    loop {
        // Find the current ancestor with the largest birth strictly below tau.
        let mut cand: Option<u64> = None;
        let mut cb = f64::NEG_INFINITY;
        for (&a, _) in cnt.iter() {
            let b = *birth.get(&a).unwrap_or(&0.0);
            if b < tau && b > cb {
                cb = b;
                cand = Some(a);
            }
        }
        let cand = match cand {
            Some(c) => c,
            None => return None, // all at roots / t=0
        };
        tau = cb;
        let c_count = cnt.remove(&cand).unwrap();
        match parent.get(&cand) {
            None => {
                // Root reached for this lineage; drop it.
                if cnt.len() <= 1 {
                    return None;
                }
            }
            Some(&par) => {
                if let Some(slot) = cnt.get_mut(&par) {
                    // Coalescence: cand merges into a parent already tracked.
                    *slot += c_count;
                    return Some(t_star - tau);
                } else {
                    cnt.insert(par, c_count);
                }
            }
        }
    }
}

/// (S, I, N) at the snapshot time `t_star` from the trajectory.
fn sin_at(traj: &Trajectory, t_star: f64) -> Option<(f64, f64, f64)> {
    traj.snapshots
        .iter()
        .find(|s| (s.t - t_star).abs() < 1e-6)
        .map(|s| {
            let cs = &s.int_state.counts;
            let s_ = cs[COMP_S] as f64;
            let i_ = cs[COMP_I] as f64;
            let r_ = cs[COMP_R] as f64;
            (s_, i_, s_ + i_ + r_)
        })
}

#[test]
fn tier4_coalescent_interval_matches_structured_coalescent() {
    // Well-mixed SIR, N ≈ 10⁴, Gillespie. Sample k lineages alive at the
    // epidemic peak (t* where S, I, N are quasi-stationary over the short first
    // coalescent interval) and compare the mean first-coalescence interval to
    // the structured-coalescent prediction 1/(C(k,2)·2f/I²), f = βSI/N.
    let beta = 1.0;
    let gamma = 0.25;
    let t_star = 10.0; // near the epidemic peak for these parameters
    let k = 200usize;
    let n_rep = 60u64;

    let mut intervals: Vec<f64> = Vec::new();
    let mut s_sum = 0.0;
    let mut i_sum = 0.0;
    let mut n_sum = 0.0;
    let mut n_sin = 0usize;

    for seed in 1..=n_rep {
        let mut m = load_fixture("sir_coalescent");
        set_params(&mut m, &[("beta", beta), ("gamma", gamma), ("N0", 10000.0)]);
        let (traj, entries) = run(&m, seed, 40.0);

        let (s_, i_, n_) = match sin_at(&traj, t_star) {
            Some(v) => v,
            None => continue,
        };
        // Require a real epidemic at t* with enough live lineages.
        if i_ < k as f64 {
            continue;
        }
        s_sum += s_;
        i_sum += i_;
        n_sum += n_;
        n_sin += 1;

        let (parent, birth, live) = build(&entries, t_star);
        let mut rng = StatefulRng::new(seed.wrapping_mul(2654435761));
        if let Some(iv) = first_coalescent_interval(&parent, &birth, &live, k, t_star, &mut rng) {
            intervals.push(iv);
        }
    }

    assert!(
        intervals.len() >= 30,
        "need many replicates with a coalescence; got {}",
        intervals.len()
    );

    let s = s_sum / n_sin as f64;
    let i = i_sum / n_sin as f64;
    let n = n_sum / n_sin as f64;
    let f = beta * s * i / n;
    let pair_rate = 2.0 * f / (i * i);
    let lam = (k * (k - 1) / 2) as f64 * pair_rate;
    let pred_mean = 1.0 / lam;

    let m = intervals.len() as f64;
    let obs_mean = intervals.iter().sum::<f64>() / m;
    // For Exp(λ) the mean estimator has sd = mean/sqrt(n_rep). Use 3σ.
    let mc_sigma = pred_mean / m.sqrt();
    let tol = 3.0 * mc_sigma;

    eprintln!(
        "Tier4 coalescent: S={:.0} I={:.0} N={:.0} f={:.1}; \
         pair_rate(2f/I²)={:.3e} λ(k={})={:.4}; \
         predicted mean interval={:.4}, observed={:.4} (n={}), tol(3σ)={:.4}",
        s, i, n, f, pair_rate, k, lam, pred_mean, obs_mean, intervals.len(), tol
    );

    assert!(
        (obs_mean - pred_mean).abs() <= tol,
        "coalescent interval mean {:.4} deviates from structured-coalescent \
         prediction {:.4} by more than 3σ ({:.4}); the transmission tree does \
         not match Volz 2009 / 2f·I⁻² at the epidemic peak",
        obs_mean,
        pred_mean,
        tol
    );
}
