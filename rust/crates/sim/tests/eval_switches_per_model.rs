//! The evaluator switches are a property of the compiled model, not of the
//! process.
//!
//! `EvalSettings::eval_unresolved` (`CAMDL_EVAL_UNRESOLVED`) routes a model's
//! propensities through the string-keyed `eval_expr`, and
//! `EvalSettings::no_binding_cache` (`CAMDL_NO_BINDING_CACHE`) turns its
//! binding cache off. Each is set on the model a command compiles, so two
//! models in one process, on one thread, evaluate on the paths their own
//! settings name: the switched one on the switched path, the other on the
//! default path, in either order. Which path ran is read from the engine's
//! per-thread counters: the propensity vectors evaluated through
//! `eval_expr` (`take_unresolved_propensity_evals`) and the binding-cache hits
//! (`take_binding_cache_hits`).
//!
//! Each pair of runs also simulates to the same trajectory, since the
//! switched paths are required to agree with the default one.

use std::path::PathBuf;
use sim::{
    compiled_model::CompiledModel,
    config::{ChainBinomialConfig, SimConfig},
    propensity::take_unresolved_propensity_evals,
    resolved_expr::take_binding_cache_hits,
    simulate::Simulate,
    ChainBinomialSim,
};

/// The spatial fixture the binding-cache A/B gate uses: 18 hoisted bindings,
/// each referenced many times across the rates, so its cache serves hits.
fn model() -> ir::Model {
    let path = PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").unwrap())
        .join("tests/fixtures/sparse_coupling_ab_folded.ir.json");
    let text = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("reading {path:?}: {e}"));
    ir::from_str(&text).expect("the fixture parses")
}

fn compiled(eval: sim::EvalSettings) -> CompiledModel {
    CompiledModel::new(model()).expect("the fixture compiles").with_eval_settings(eval)
}

/// Run `m` on chain-binomial at a fixed seed, and return its trajectory with
/// what the two counters recorded during the run.
fn run(m: &CompiledModel) -> (sim::state::Trajectory, u64, u64) {
    let _ = (take_unresolved_propensity_evals(), take_binding_cache_hits());
    let t = &m.model.simulation;
    let config = SimConfig::ChainBinomial(ChainBinomialConfig {
        t_start: t.t_start,
        t_end: t.t_end,
        dt: 1.0,
        binomial: sim::rng::BinomialAlgorithm::Btpe,
    });
    let traj = ChainBinomialSim.run(m, &m.default_params, 42, &config).expect("the run succeeds");
    (traj, take_unresolved_propensity_evals(), take_binding_cache_hits())
}

fn snapshots(t: &sim::state::Trajectory) -> Vec<(u64, Vec<i64>)> {
    t.snapshots.iter().map(|s| (s.t.to_bits(), s.int_state.counts.clone())).collect()
}

#[test]
fn a_model_compiled_with_eval_unresolved_takes_the_string_keyed_path_and_another_does_not() {
    let default = compiled(sim::EvalSettings { allow_degenerate_rates: true, ..Default::default() });
    let switched = compiled(sim::EvalSettings {
        allow_degenerate_rates: true,
        eval_unresolved: true,
        ..Default::default()
    });

    let (a, a_unresolved, _) = run(&default);
    let (b, b_unresolved, _) = run(&switched);
    let (c, c_unresolved, _) = run(&default);

    assert_eq!(a_unresolved, 0, "the default model evaluates no rate through eval_expr");
    assert!(b_unresolved > 0, "the switched model evaluates its rates through eval_expr");
    assert_eq!(c_unresolved, 0, "and the default model still does not, after the switched one ran");
    assert_eq!(snapshots(&a), snapshots(&b), "the two evaluators agree");
    assert_eq!(snapshots(&a), snapshots(&c));
}

#[test]
fn a_model_compiled_with_no_binding_cache_serves_no_hit_and_another_does() {
    let default = compiled(sim::EvalSettings { allow_degenerate_rates: true, ..Default::default() });
    let switched = compiled(sim::EvalSettings {
        allow_degenerate_rates: true,
        no_binding_cache: true,
        ..Default::default()
    });

    let (a, _, a_hits) = run(&default);
    let (b, _, b_hits) = run(&switched);
    let (c, _, c_hits) = run(&default);

    assert!(a_hits > 0, "the default model's binding cache serves hits");
    assert_eq!(b_hits, 0, "the switched model's binding cache is off");
    assert!(c_hits > 0, "and the default model's is on again, after the switched one ran");
    assert_eq!(snapshots(&a), snapshots(&b), "the cache is value-preserving");
    assert_eq!(snapshots(&a), snapshots(&c));
}

