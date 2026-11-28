//! A/B gate for the runtime binding cache (the byte-identical soundness proof
//! the cache's claim rests on).
//!
//! The cache (`resolved_expr.rs`, `BindingCache`/`CacheScope`) memoizes each
//! model binding's value for the lifetime of one propensity-vector evaluation,
//! so a binding referenced N times in the rate trees (e.g. `N_p`/`I_agg_p` in a
//! spatial FOI, ~945× on a dense P=44 model) is computed once per state instead
//! of N times. It claims to be *trajectory-preserving*: a memoized value within
//! one state snapshot is identical to recomputing it. This gate makes that claim
//! a test.
//!
//! Unlike the constant-fold gate, the A side and B side are the SAME model — the
//! only difference is the cache flag, flipped in-process via
//! `set_binding_cache_disabled`. The fixture `sparse_coupling_ab_folded.ir.json`
//! has 18 hoisted bindings (`beta`, `seas`, and per-patch `I_agg_p*`/`N_p*`),
//! referenced repeatedly across the spatial transition rates.
//!
//! Two assertions (counting hits of both the flat evaluator's cache and the
//! tree walk's):
//!   1. NON-VACUITY — with the cache ON, a run must register binding-cache hits
//!      (`take_binding_cache_hits() > 0`). A green test where the cache never
//!      served a hit would prove nothing; this guards it.
//!   2. SOUNDNESS — for every supported backend at a fixed seed, the cache-on and
//!      cache-off runs simulate to a byte-identical trajectory (same FNV-1a
//!      hash). This is what "trajectory-preserving" means.

use std::path::PathBuf;
use sim::{
    compiled_model::CompiledModel,
    config::{ChainBinomialConfig, GillespieConfig, OdeConfig, SimConfig},
    resolved_expr::{set_binding_cache_disabled, take_binding_cache_hits},
    simulate::Simulate,
    ChainBinomialSim, GillespieSim, OdeSim,
};

const SEED: u64 = 42;

fn fixtures_dir() -> PathBuf {
    let manifest = std::env::var("CARGO_MANIFEST_DIR").unwrap();
    PathBuf::from(&manifest).join("tests/fixtures")
}

fn load(name: &str) -> ir::Model {
    let path = fixtures_dir().join(format!("{}.ir.json", name));
    let contents = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("could not read {:?}: {}", path, e));
    ir::from_str(&contents).unwrap_or_else(|e| panic!("failed to parse {}: {}", name, e))
}

/// `model` with every forcing reference in an expression replaced by the
/// constant 1. The fixture's rates read the forcing `school(t)`, which changes
/// between events, so the Gillespie backend refuses the model as written
/// (`TIME_VARYING_RATES`); the Gillespie side of the gate runs this copy, whose
/// bindings and their references are unchanged.
fn forcings_held_at_one(model: &ir::Model) -> ir::Model {
    fn replace(v: &mut serde_json::Value, one: &serde_json::Value) {
        match v {
            serde_json::Value::Object(m) if m.len() == 1 && m.contains_key("time_func") => {
                *v = one.clone();
            }
            serde_json::Value::Object(m) => m.values_mut().for_each(|c| replace(c, one)),
            serde_json::Value::Array(a) => a.iter_mut().for_each(|c| replace(c, one)),
            _ => {}
        }
    }
    let one = serde_json::to_value(ir::expr::Expr::const_(1.0)).unwrap();
    let mut v = serde_json::to_value(model).unwrap();
    replace(&mut v, &one);
    serde_json::from_value(v).expect("the forcing-free copy must still parse")
}

/// FNV-1a/64 over the full trajectory numeric content — the same hash the
/// constant-fold A/B gate uses. Deterministic and platform-independent.
fn trajectory_hash(traj: &sim::state::Trajectory) -> u64 {
    let mut h: u64 = 0xcbf29ce484222325;
    let mut mix = |bytes: &[u8]| {
        for &b in bytes {
            h ^= b as u64;
            h = h.wrapping_mul(0x100000001b3);
        }
    };
    for snap in &traj.snapshots {
        mix(&snap.t.to_bits().to_le_bytes());
        for &c in &snap.int_state.counts {
            mix(&c.to_le_bytes());
        }
        for &v in &snap.real_state.values {
            mix(&v.to_bits().to_le_bytes());
        }
        match &snap.flows {
            sim::state::Flows::Int(fs) => {
                for &f in fs {
                    mix(&f.to_le_bytes());
                }
            }
            sim::state::Flows::Real(fs) => {
                for &f in fs {
                    mix(&f.to_bits().to_le_bytes());
                }
            }
        }
    }
    h
}

#[test]
fn gate_binding_cache_is_byte_identical() {

    let model = load("sparse_coupling_ab_folded");

    // The cache only proves something if the model actually has bindings to
    // cache. (The hit-count assertion below is the load-bearing non-vacuity
    // check; this is a fast sanity guard on the fixture.)
    assert!(
        !model.bindings.is_empty(),
        "fixture has no bindings — the binding cache A/B gate is vacuous; pick a \
         model with hoisted bindings (per-patch N_p/I_agg_p)"
    );

    let compile = |m: ir::Model| {
        CompiledModel::new(m).expect("model failed to compile")
            .with_eval_settings(sim::EvalSettings { allow_degenerate_rates: true, ..Default::default() })
    };
    let compiled_forced = compile(model.clone());
    let compiled_gillespie = compile(forcings_held_at_one(&model));
    assert!(
        compiled_forced.required_capabilities().contains(sim::Capabilities::TIME_VARYING_RATES)
            && !compiled_gillespie.required_capabilities().contains(sim::Capabilities::TIME_VARYING_RATES),
        "the fixture's rates must read its forcing, and the Gillespie copy's must not"
    );

    let t_start = model.simulation.t_start;
    let t_end = model.simulation.t_end;

    let backends: &[(&str, SimConfig)] = &[
        ("gillespie", SimConfig::Gillespie(GillespieConfig { t_start, t_end, output_dt: None })),
        ("chain_binomial", SimConfig::ChainBinomial(ChainBinomialConfig { t_start, t_end, dt: 1.0, binomial: sim::rng::BinomialAlgorithm::Btpe })),
        ("ode", SimConfig::Ode(OdeConfig { t_start, t_end, dt: 1.0 })),
    ];

    let mut checked = 0usize;
    for (backend, config) in backends {
        let sim: &dyn Simulate = match *backend {
            "gillespie" => &GillespieSim,
            "ode" => &OdeSim,
            _ => &ChainBinomialSim,
        };
        let compiled = if *backend == "gillespie" { &compiled_gillespie } else { &compiled_forced };
        let params = compiled.default_params.clone();
        if !(compiled.required_capabilities() - sim.capabilities()).is_empty() {
            continue;
        }

        // B side: cache OFF (the pre-cache on-demand evaluator).
        set_binding_cache_disabled(true);
        let _ = take_binding_cache_hits();
        let traj_off = sim
            .run(compiled, &params, SEED, config)
            .unwrap_or_else(|e| panic!("cache-off {backend} sim failed: {e:?}"));
        let h_off = trajectory_hash(&traj_off);
        // The B side must really be off: a cache that ignored the switch would
        // still hash identically (it is sound), and the A/B would compare the
        // cache with itself.
        let off_hits = take_binding_cache_hits();
        assert_eq!(
            off_hits, 0,
            "{backend}: the cache-off run served {off_hits} binding-cache hits — \
             the off switch is not honoured, so the A/B compares the cache with itself"
        );

        // A side: cache ON. Clear the hit counter, run, then read it back.
        set_binding_cache_disabled(false);
        let _ = take_binding_cache_hits();
        let traj_on = sim
            .run(compiled, &params, SEED, config)
            .unwrap_or_else(|e| panic!("cache-on {backend} sim failed: {e:?}"));
        let hits = take_binding_cache_hits();
        let h_on = trajectory_hash(&traj_on);

        // ── 1. NON-VACUITY ──────────────────────────────────────────────────
        // Every backend here evaluates propensities through `eval_propensities`
        // (the flat evaluator's cache; the ODE backend also builds derivatives
        // through the tree walk's cache), so each must register hits —
        // verified: gillespie 16554, chain_binomial 22630, ode 90520 on this
        // fixture. Zero hits means the
        // cache was never exercised and byte-identity would prove nothing.
        assert!(
            hits > 0,
            "{backend}: binding cache served 0 hits — the A/B gate is vacuous \
             (cache never exercised). Expected the spatial FOI to re-reference \
             N_p/I_agg_p within a propensity step."
        );

        // ── 2. SOUNDNESS ────────────────────────────────────────────────────
        assert_eq!(
            h_off, h_on,
            "TRAJECTORY DIVERGED on {backend}: the binding cache is NOT \
             byte-identical (cache-off 0x{h_off:016x} != cache-on 0x{h_on:016x}). \
             This is a soundness bug in the cache, not a golden update."
        );
        eprintln!("{backend}: byte-identical (hash 0x{h_off:016x}), cache hits = {hits}");
        checked += 1;
    }
    assert_eq!(checked, 3, "every backend must run the gate, got {checked}");

    // Leave the override cleared so other tests on this thread are unaffected.
    set_binding_cache_disabled(false);
}
