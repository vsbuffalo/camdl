//! gh#732 — the bootstrap particle filter draws x₀ PER PARTICLE.
//!
//! The filter used to evaluate one initial state and copy it into every
//! particle. For the whole existing corpus — every model whose `init { }`
//! computes its compartments from expressions — that was exact, because such a
//! draw is deterministic. For a model whose `init { }` DECLARES a law
//! (`I ~ poisson(rate = I0)`, ir/VERSION 0.35) it was not: the swarm would have
//! conditioned on ONE realization of x₀ instead of integrating over
//! p(x₀ | θ), which is a wrong likelihood rather than a noisy one. Such a model
//! was therefore refused outright.
//!
//! Both halves are asserted here, against numbers rather than against "it ran":
//!
//! 1. **A law-bearing model is accepted, and its swarm has real spread at
//!    t=0** — non-zero variance in every law-seeded compartment, measured
//!    across particles at the first observation.
//! 2. **A deterministic model's swarm has EXACTLY zero spread** — the negative
//!    control. Without it, (1) could be satisfied by any change that made
//!    particles differ for some unrelated reason, and the byte-identity claim
//!    for the existing corpus would rest on nothing.
//!
//! The measurement point is an observation placed AT `t_start`, so the
//! recorded pre-resample states are x₀ itself: the filter walks zero substeps
//! into that window, so no transition noise can contribute to the variance.
//!
//! Proposal: `docs/dev/proposals/2026-08-23-initial-state-parameters.md`
//! (staging step 5).

use std::sync::Arc;

use sim::{
    compiled_model::CompiledModel,
    inference::{
        particle_filter::bootstrap_filter,
        traits::{ObservationModel, SMCConfig},
        ChainBinomialProcess, ParticleState,
    },
};

const SEED: u64 = 20260824;
const N_PARTICLES: usize = 400;

/// A flat observation model: two observation times, the first AT `t_start`,
/// and a log-likelihood that ignores the state.
///
/// Flat on purpose. A state-dependent likelihood would reweight the swarm and
/// the degeneracy watchdog could bail before the assertion is reached; here
/// every particle keeps weight 1, ESS stays at N, and the only thing the
/// recorded states can reflect is the initial draw.
struct FlatObs {
    times: Vec<f64>,
}

impl ObservationModel<ParticleState> for FlatObs {
    fn log_likelihood(&self, _s: &ParticleState, _i: usize, _p: &[f64]) -> f64 { 0.0 }
    fn n_observations(&self) -> usize { self.times.len() }
    fn obs_time(&self, i: usize) -> f64 { self.times[i] }
    fn n_streams(&self) -> usize { 0 }
}

/// The same, but one that PROJECTS: it declares a stream and returns a
/// non-empty `mean()`.
///
/// That combination is what drives the filter's `has_predictions` probe, and
/// the probe is the one place left that must reach for a state when the swarm
/// is empty. A non-projecting obs model short-circuits it (`n_streams() > 0`
/// is checked first), so the degenerate-swarm test below would assert nothing
/// about that path without this.
struct ProjectingFlatObs {
    times: Vec<f64>,
}

impl ObservationModel<ParticleState> for ProjectingFlatObs {
    fn log_likelihood(&self, _s: &ParticleState, _i: usize, _p: &[f64]) -> f64 { 0.0 }
    fn n_observations(&self) -> usize { self.times.len() }
    fn obs_time(&self, i: usize) -> f64 { self.times[i] }
    fn n_streams(&self) -> usize { 1 }
    fn mean(&self, s: &ParticleState, _i: usize, _p: &[f64]) -> Vec<f64> {
        vec![s.counts[0] as f64]
    }
}

fn load(rel: &str) -> ir::Model {
    let path = format!("{}/../../../{}", env!("CARGO_MANIFEST_DIR"), rel);
    let json = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("read {path}: {e}"));
    ir::from_str(&json).unwrap_or_else(|e| panic!("parse {path}: {e}"))
}

/// `ocaml/golden/init_laws.camdl` under its `baseline` scenario: an SEIR whose
/// `init { }` draws `I`, `E` and `R` from laws and computes `S` from what they
/// drew.
fn law_bearing_fixture() -> (Arc<CompiledModel>, Vec<f64>) {
    let mut model = load("ocaml/golden/init_laws.ir.json");
    let preset = model.presets.first().cloned().expect("the fixture declares a scenario");
    for p in &mut model.parameters {
        if let Some(&v) = preset.params.get(&p.name) {
            p.value = p.value.with_value(v);
        }
    }
    let compiled = Arc::new(CompiledModel::new(model).expect("fixture must compile"));
    let params = compiled.default_params.clone();
    (compiled, params)
}

/// The two deterministic-`init { }` shapes, matching the pair
/// `initial_state_seam.rs` covers: one whose entries are expressions over
/// parameters and one whose entries are bare literals. They take different
/// paths through the initial-state producer (`eval_expr` against the partially
/// built state vs constant placement), so both are controlled.
fn deterministic_fixtures() -> Vec<(&'static str, Arc<CompiledModel>, Vec<f64>)> {
    let mut out = Vec::new();

    let mut parameterized = load("tests/fixtures/gradient/ir/seir_seasonal_lagged.ir.json");
    for p in &mut parameterized.parameters {
        if p.value.resolved_value().is_none() {
            p.value = p.value.with_value(match p.name.as_str() {
                "beta" => 0.3,
                "sigma" => 0.2,
                "gamma" => 0.1,
                "alpha" => 0.15,
                "phi_season" => 90.0,
                "N0" => 1_000_000.0,
                "I0" => 10.0,
                _ => 0.5,
            });
        }
    }
    let compiled = Arc::new(CompiledModel::new(parameterized).expect("compile parameterized"));
    let params = compiled.default_params.clone();
    out.push(("parameterized", compiled, params));

    let explicit = load("tests/fixtures/corner_cases/ir/dt_rate.ir.json");
    let compiled = Arc::new(CompiledModel::new(explicit).expect("compile explicit"));
    let params = compiled.default_params.clone();
    out.push(("explicit", compiled, params));

    out
}

fn int_idx(c: &CompiledModel, name: &str) -> usize {
    let global = c.comp_index[name];
    c.global_to_int[global].expect("an integer compartment")
}

/// Run the filter over `[t_start, 5]` with observations at `t_start` and 5,
/// and return the PRE-RESAMPLE particle states recorded at the FIRST
/// observation — which, because that observation sits at `t_start`, are the
/// per-particle initial states with no propagation in between.
fn x0_per_particle(compiled: &Arc<CompiledModel>, params: &[f64]) -> Vec<Vec<f64>> {
    let t_start = compiled.model.simulation.t_start;
    let process = ChainBinomialProcess::new(Arc::clone(compiled));
    let obs = FlatObs { times: vec![t_start, t_start + 5.0] };
    let config = SMCConfig {
        n_particles: N_PARTICLES,
        dt: 1.0,
        t_start,
        skip_first_obs_from_loglik: false,
        record_ancestry: true,
        record_prequential: false,
        max_substeps: sim::inference::degeneracy::ITER_BUDGET,
    };
    let result = bootstrap_filter(&process, &obs, params, &config, SEED)
        .expect("a bootstrap filter run must not be refused (gh#732)");
    let trace = result.ancestry.expect("record_ancestry was requested");
    trace.states.first().expect("one recorded step per observation").clone()
}

/// Population variance and mean of one compartment column across particles.
fn var_mean(states: &[Vec<f64>], col: usize) -> (f64, f64) {
    let xs: Vec<f64> = states.iter().map(|s| s[col]).collect();
    let m = xs.iter().sum::<f64>() / xs.len() as f64;
    let v = xs.iter().map(|x| (x - m).powi(2)).sum::<f64>() / xs.len() as f64;
    (v, m)
}

/// A model whose `init { }` declares a law is ACCEPTED by the bootstrap filter,
/// and its swarm carries genuine initial-state spread.
///
/// `init_laws.camdl` seeds `I ~ poisson(rate = I0)`, `E ~ neg_binomial(...)`,
/// `R ~ binomial(n = N0, p = frac_immune)` and computes `S = N0 - I - E - R`.
/// Every one of the four must vary across particles — `S` included, because it
/// reads what the three laws DREW.
#[test]
fn a_declared_init_law_gives_the_swarm_spread_at_t0() {
    let (compiled, params) = law_bearing_fixture();
    assert!(compiled.has_init_law, "the fixture must declare at least one law");

    let x0 = x0_per_particle(&compiled, &params);
    assert_eq!(x0.len(), N_PARTICLES, "one recorded state per particle");

    // Poisson(rate = I0 = 50): Var = mean = 50. Loose bounds — this asserts a
    // sampler is running per particle, not a distributional fit.
    let (v_i, m_i) = var_mean(&x0, int_idx(&compiled, "I"));
    assert!(v_i > 5.0, "I has no per-particle initial-state variance (var = {v_i}, mean = {m_i})");
    assert!((m_i - 50.0).abs() < 15.0, "I's across-particle mean should sit near I0 = 50, got {m_i}");

    // NegBinomial(mean = 1.5 I, r = 5): overdispersed, so var > mean.
    let (v_e, m_e) = var_mean(&x0, int_idx(&compiled, "E"));
    assert!(v_e > m_e, "E must be overdispersed across particles (var {v_e} <= mean {m_e})");

    // Binomial(N0 = 100000, p = 0.2): Var = N p (1-p) = 16000.
    let (v_r, m_r) = var_mean(&x0, int_idx(&compiled, "R"));
    assert!(v_r > 1000.0, "R has no per-particle initial-state variance (var = {v_r})");
    assert!((m_r - 20000.0).abs() < 700.0, "R's mean should sit near N0 p = 20000, got {m_r}");

    // The deterministic entry inherits the spread, because it reads the DRAWN
    // values — the dependency-ordered evaluation working inside the filter.
    let (v_s, _) = var_mean(&x0, int_idx(&compiled, "S"));
    assert!(v_s > 0.0, "S must inherit the draws' spread; got zero variance");

    // Distinct states, not one value repeated: a swarm of 400 copies would
    // satisfy "variance > 0" for none of the above, but say it plainly.
    let distinct: std::collections::HashSet<i64> =
        x0.iter().map(|s| s[int_idx(&compiled, "I")] as i64).collect();
    assert!(distinct.len() > 10, "only {} distinct x₀ values for I across {N_PARTICLES} particles",
        distinct.len());

    // Surfaced under `--nocapture` so the measured spread can be read off a
    // run rather than inferred from the thresholds above.
    eprintln!(
        "gh#732 across-particle x0 spread (N = {N_PARTICLES}): \
         I var {v_i:.1} mean {m_i:.1} | E var {v_e:.1} mean {m_e:.1} | \
         R var {v_r:.1} mean {m_r:.1} | S var {v_s:.1}"
    );

    // The population budget the dependency order exists to guarantee holds on
    // EVERY particle, not just on average: S + E + I + R == N0.
    let n0 = params[compiled.param_index["N0"]];
    for (j, s) in x0.iter().enumerate() {
        let total = s[int_idx(&compiled, "S")] + s[int_idx(&compiled, "E")]
            + s[int_idx(&compiled, "I")] + s[int_idx(&compiled, "R")];
        assert_eq!(total, n0, "particle {j}: S+E+I+R = {total}, expected N0 = {n0}");
    }
}

/// The negative control, and the load-bearing half of the byte-identity claim
/// for every model in the corpus: with a deterministic `init { }` the
/// per-particle draw returns the SAME state for every particle, so the swarm's
/// spread at t=0 is exactly zero — not small, zero.
#[test]
fn a_deterministic_init_gives_the_swarm_no_spread_at_t0() {
    for (who, compiled, params) in deterministic_fixtures() {
        assert!(!compiled.has_init_law, "{who}: this control fixture must declare no law");

        let x0 = x0_per_particle(&compiled, &params);
        assert_eq!(x0.len(), N_PARTICLES, "{who}: one recorded state per particle");

        let first = &x0[0];
        assert!(first.iter().any(|&c| c > 0.0),
            "{who}: the control's initial state is all zeros — the assertion below \
             would be vacuous");
        for (j, s) in x0.iter().enumerate() {
            assert_eq!(s, first,
                "{who}: particle {j} differs from particle 0 at t=0 under a deterministic \
                 init {{ }}; the existing corpus's trajectories would have shifted");
        }
    }
}

/// The degenerate swarm sizes. `n_particles == 0` is deliberately tolerated by
/// the degeneracy layer (`check_pf_degeneracy`'s `n_particles > 0` guard,
/// pinned by `degeneracy::tests::empty_swarm_does_not_trigger_all_dead`), so
/// neither the per-particle draw nor the `has_predictions` probe may index a
/// per-particle RNG stream that is not there.
///
/// Run against a PROJECTING obs model on purpose: with `n_streams() == 0` the
/// probe short-circuits and this test would exercise only the draw loop.
#[test]
fn a_swarm_of_one_or_zero_particles_does_not_panic() {
    let (compiled, params) = law_bearing_fixture();
    let t_start = compiled.model.simulation.t_start;
    let process = ChainBinomialProcess::new(Arc::clone(&compiled));
    let obs = ProjectingFlatObs { times: vec![t_start, t_start + 5.0] };
    for n_particles in [0usize, 1] {
        let config = SMCConfig {
            n_particles,
            dt: 1.0,
            t_start,
            skip_first_obs_from_loglik: false,
            record_ancestry: false,
            record_prequential: false,
            max_substeps: sim::inference::degeneracy::ITER_BUDGET,
        };
        // The outcome for an empty swarm is not meaningful (there is no
        // estimate of anything); what is asserted is that asking for one is a
        // return, not a panic.
        let _ = bootstrap_filter(&process, &obs, &params, &config, SEED);
    }
}
