//! A/B gate for the sparse-coupling constant-fold pass (the byte-identical
//! soundness proof the fold's claim rests on).
//!
//! The fold (`ocaml/lib/ir/constant_fold.ml`) resolves constant-indexed
//! inline-table lookups and drops zero-`W` terms from the force-of-infection
//! `Reduce`, collapsing the dense P-term spatial sum to its k nonzero terms
//! (O(P^2) -> O(P*k)). It claims to be *trajectory-preserving*. This gate makes
//! that claim a test, on a model where the fold actually fires.
//!
//! Two committed fixtures compiled from the SAME source
//! (`sparse_coupling_ab.camdl`, a sparse ring W with K=2 neighbours per patch):
//!   - `sparse_coupling_ab_unfolded.ir.json` — `camdlc` with the fold OFF
//!   - `sparse_coupling_ab_folded.ir.json`   — `camdlc CAMDL_CONSTANT_FOLD=1`
//!
//! See the source header for the exact regeneration commands. The fixtures are
//! static IR (the test does not recompile), so the default-flag flip is
//! decoupled from this gate.
//!
//! Two assertions:
//!   1. NON-VACUITY — the folded IR is strictly smaller, with strictly fewer
//!      FOI `Reduce` terms (the dense 8-term sum collapses to 2). A test that is
//!      green only because the fold was a no-op proves nothing; this guards it.
//!   2. SOUNDNESS — for every supported backend at a fixed seed, the folded and
//!      unfolded models simulate to a byte-identical trajectory (same FNV-1a
//!      hash). This is what "trajectory-preserving" means.

use std::path::PathBuf;
use ir::expr::Expr;
use sim::{
    compiled_model::CompiledModel,
    config::{ChainBinomialConfig, GillespieConfig, OdeConfig, SimConfig},
    simulate::Simulate,
    ChainBinomialSim, GillespieSim, OdeSim,
};

const SEED: u64 = 42;

fn fixtures_dir() -> PathBuf {
    let manifest = std::env::var("CARGO_MANIFEST_DIR").unwrap();
    PathBuf::from(&manifest).join("tests/fixtures")
}

fn load(name: &str) -> (ir::Model, usize) {
    let path = fixtures_dir().join(format!("{}.ir.json", name));
    let contents = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("could not read {:?}: {}", path, e));
    let model = ir::from_str(&contents)
        .unwrap_or_else(|e| panic!("failed to parse {}: {}", name, e));
    (model, contents.len())
}

/// Largest `Reduce` term-count anywhere in an expression tree. For this model
/// the only `Reduce` is the spatial FOI sum, so this is its term count: P (=8)
/// unfolded, k (=2) folded.
fn max_reduce_terms(e: &Expr) -> usize {
    match e {
        Expr::Reduce(w) => {
            let here = w.reduce.len();
            w.reduce.iter().map(max_reduce_terms).max().unwrap_or(0).max(here)
        }
        Expr::BinOp(w) => max_reduce_terms(&w.bin_op.left).max(max_reduce_terms(&w.bin_op.right)),
        Expr::UnOp(w) => max_reduce_terms(&w.un_op.arg),
        Expr::Cond(w) => max_reduce_terms(&w.cond.pred)
            .max(max_reduce_terms(&w.cond.then))
            .max(max_reduce_terms(&w.cond.else_)),
        Expr::TableLookup(w) => w.table_lookup.indices.iter().map(max_reduce_terms).max().unwrap_or(0),
        Expr::UncheckedDim(w) => max_reduce_terms(&w.unchecked_dim.inner),
        _ => 0,
    }
}

/// Largest FOI `Reduce` term count across all transition rates in the model.
fn max_foi_reduce_terms(m: &ir::Model) -> usize {
    m.transitions.iter().map(|t| max_reduce_terms(&t.rate)).max().unwrap_or(0)
}

/// FNV-1a/64 over the full trajectory numeric content — the same hash the
/// trajectory-baseline gate uses. Deterministic and platform-independent.
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
fn gate_constant_fold_is_byte_identical() {
    sim::eval_stats::set_allow_degenerate_rates(true);

    let (unfolded, unfolded_bytes) = load("sparse_coupling_ab_unfolded");
    let (folded, folded_bytes) = load("sparse_coupling_ab_folded");

    // ── 1. NON-VACUITY ──────────────────────────────────────────────────────
    // The fold must actually have fired on this fixture, or the soundness check
    // below is testing nothing.
    let unfolded_terms = max_foi_reduce_terms(&unfolded);
    let folded_terms = max_foi_reduce_terms(&folded);
    assert!(
        unfolded_terms > 0,
        "unfolded fixture has no FOI Reduce — fixture is wrong (expected a dense \
         spatial sum to collapse); got {unfolded_terms} terms"
    );
    assert!(
        folded_terms < unfolded_terms,
        "fold did not fire: FOI Reduce term count did not shrink \
         (unfolded={unfolded_terms}, folded={folded_terms}). This A/B gate is \
         vacuous unless the fold collapses the dense sum — regenerate the \
         fixtures from a sparse-coupling model (see the .camdl header)."
    );
    assert!(
        folded_bytes < unfolded_bytes,
        "fold did not shrink the IR (unfolded={unfolded_bytes} bytes, \
         folded={folded_bytes} bytes)"
    );
    eprintln!(
        "non-vacuity: FOI Reduce terms {unfolded_terms} -> {folded_terms}; \
         IR {unfolded_bytes} -> {folded_bytes} bytes"
    );

    // ── 2. SOUNDNESS ────────────────────────────────────────────────────────
    // Same source, same seed, same backend: the folded and unfolded models must
    // produce a byte-identical trajectory.
    let compiled_unfolded = CompiledModel::new(unfolded.clone())
        .expect("unfolded model failed to compile");
    let compiled_folded = CompiledModel::new(folded.clone())
        .expect("folded model failed to compile");

    let params_unfolded = compiled_unfolded.default_params.clone();
    let params_folded = compiled_folded.default_params.clone();

    let t_start = unfolded.simulation.t_start;
    let t_end = unfolded.simulation.t_end;
    assert_eq!(
        t_end, folded.simulation.t_end,
        "fixtures disagree on t_end — regenerated from different sources?"
    );

    let backends: &[(&str, SimConfig)] = &[
        ("gillespie", SimConfig::Gillespie(GillespieConfig { t_start, t_end, output_dt: None })),
        ("chain_binomial", SimConfig::ChainBinomial(ChainBinomialConfig { t_start, t_end, dt: 1.0 })),
        ("ode", SimConfig::Ode(OdeConfig { t_start, t_end, dt: 1.0 })),
    ];

    let required = compiled_unfolded.required_capabilities();
    let mut checked = 0usize;
    for (backend, config) in backends {
        let sim: &dyn Simulate = match *backend {
            "gillespie" => &GillespieSim,
            "ode" => &OdeSim,
            _ => &ChainBinomialSim,
        };
        if !(required - sim.capabilities()).is_empty() {
            continue;
        }
        let traj_unfolded = sim
            .run(&compiled_unfolded, &params_unfolded, SEED, config)
            .unwrap_or_else(|e| panic!("unfolded {backend} sim failed: {e:?}"));
        let traj_folded = sim
            .run(&compiled_folded, &params_folded, SEED, config)
            .unwrap_or_else(|e| panic!("folded {backend} sim failed: {e:?}"));

        let h_unfolded = trajectory_hash(&traj_unfolded);
        let h_folded = trajectory_hash(&traj_folded);
        assert_eq!(
            h_unfolded, h_folded,
            "TRAJECTORY DIVERGED on {backend}: the constant-fold is NOT \
             byte-identical (unfolded 0x{h_unfolded:016x} != folded \
             0x{h_folded:016x}). This is a soundness bug in the fold, not a \
             golden update."
        );
        eprintln!("{backend}: byte-identical (hash 0x{h_unfolded:016x})");
        checked += 1;
    }
    assert!(checked >= 3, "expected at least 3 backends checked, got {checked}");
}
