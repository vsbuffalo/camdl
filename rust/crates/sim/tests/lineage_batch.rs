//! Acceptance tests for the lineage path on the **batch** backends — tau-leap
//! and chain-binomial (2026-05-20 three-layer proposal). These now go through
//! the Layer-1 event recorder → Layer-2 `realize` path; the assertions are
//! unchanged.
//!
//! The critical invariant is the same Tier 2a guarantee Gillespie has:
//! attaching the event recorder must NOT change the count trajectory by a
//! single byte at a fixed seed — now trivially true, the recorder draws no
//! randomness. The batch backends record `multiplicity` and a `batched` flag
//! per event; `realize` samples all `k` attributions against the start-of-step
//! pools (so a child minted this step cannot be its own same-step parent) and
//! accumulates the sub-`dt` bias diagnostic. We test:
//!   - Tier 2a: byte-identity with/without recorder, both backends, many seeds.
//!   - Tier 1: structural invariants on the realized line list.
//!   - sub-`dt` diagnostic (now computed by `realize`): 0 ≤ fraction ≤ 1, grows
//!     with dt, positive for a real epidemic on chain-binomial.

use std::collections::HashMap;

use sim::{
    compiled_model::CompiledModel,
    config::{ChainBinomialConfig, SimConfig, TauLeapConfig},
    lineage::{LineListEntry, ParentRef},
    state::Trajectory,
    ChainBinomialSim, Simulate, TauLeapSim,
};

mod lineage_helpers;
use lineage_helpers::{load_fixture, record_event_log, realize_log, set_params};

/// Which batch backend a test runs under.
#[derive(Clone, Copy)]
enum Backend {
    TauLeap,
    ChainBinomial,
}

fn traj_signature(t: &Trajectory) -> Vec<(String, Vec<i64>, Vec<u64>)> {
    t.snapshots
        .iter()
        .map(|s| (format!("{:?}", s.t), s.int_state.counts.clone(), s.flows.counts.clone()))
        .collect()
}

/// Baseline run (no recorder) via the public `Simulate` dispatch.
fn run_baseline(m: &ir::Model, backend: Backend, seed: u64, dt: f64) -> Trajectory {
    let compiled = CompiledModel::new(m.clone()).unwrap();
    let params = compiled.default_params.clone();
    let t_start = m.simulation.t_start;
    let t_end = m.simulation.t_end;
    match backend {
        Backend::TauLeap => TauLeapSim
            .run(&compiled, &params, seed, &SimConfig::TauLeap(TauLeapConfig { t_start, t_end, dt }))
            .unwrap(),
        Backend::ChainBinomial => ChainBinomialSim
            .run(
                &compiled,
                &params,
                seed,
                &SimConfig::ChainBinomial(ChainBinomialConfig { t_start, t_end, dt }),
            )
            .unwrap(),
    }
}

/// Record an event log under the given batch backend, then realize it at
/// `identity_seed == seed`. Returns (trajectory, line list, sub-dt fraction,
/// edges) — the sub-dt diagnostic now comes from `realize`, not the recorder.
fn run_with_lineage(
    m: &ir::Model,
    backend: Backend,
    seed: u64,
    dt: f64,
) -> (Trajectory, Vec<LineListEntry>, f64, u64) {
    let t_end = m.simulation.t_end;
    let helper_backend = match backend {
        Backend::TauLeap => lineage_helpers::Backend::TauLeap { dt },
        Backend::ChainBinomial => lineage_helpers::Backend::ChainBinomial { dt },
    };
    let (traj, log) = record_event_log(m, helper_backend, seed, t_end);
    let (entries, summary) = realize_log(&log, seed);
    (traj, entries, summary.sub_dt_fraction, summary.edges)
}

// ── Tier 2a — trajectory byte-identity, BOTH batch backends ─────────────────────

#[test]
fn tier2a_tau_leap_byte_identical_with_and_without_lineages() {
    for seed in [1u64, 2, 7, 13, 42, 99, 1000] {
        let mut m = load_fixture("sir_lineage");
        set_params(&mut m, &[("beta", 0.6), ("gamma", 0.2), ("N0", 500.0)]);
        let baseline = run_baseline(&m, Backend::TauLeap, seed, 0.25);
        let (with_lineage, entries, _, _) = run_with_lineage(&m, Backend::TauLeap, seed, 0.25);
        assert_eq!(
            traj_signature(&baseline),
            traj_signature(&with_lineage),
            "tau-leap trajectory diverged with --lineages at seed {}",
            seed
        );
        let _ = entries;
    }
}

#[test]
fn tier2a_chain_binomial_byte_identical_with_and_without_lineages() {
    for seed in [1u64, 2, 7, 13, 42, 99, 1000] {
        let mut m = load_fixture("sir_lineage");
        set_params(&mut m, &[("beta", 0.6), ("gamma", 0.2), ("N0", 500.0)]);
        let baseline = run_baseline(&m, Backend::ChainBinomial, seed, 0.5);
        let (with_lineage, _, _, _) = run_with_lineage(&m, Backend::ChainBinomial, seed, 0.5);
        assert_eq!(
            traj_signature(&baseline),
            traj_signature(&with_lineage),
            "chain-binomial trajectory diverged with --lineages at seed {}",
            seed
        );
    }
}

#[test]
fn batch_backends_produce_line_lists() {
    let mut m = load_fixture("sir_lineage");
    set_params(&mut m, &[("beta", 0.8), ("gamma", 0.2), ("N0", 500.0)]);
    for backend in [Backend::TauLeap, Backend::ChainBinomial] {
        let (_, entries, _, _) = run_with_lineage(&m, backend, 7, 0.25);
        assert!(!entries.is_empty(), "batch backend should emit line-list entries");
        let n_lineage = entries
            .iter()
            .filter(|e| matches!(e.parent, ParentRef::Individual(_)))
            .count();
        assert!(n_lineage > 0, "expected some transmission (lineage) edges");
    }
}

// ── Tier 1 — structural invariants on the batch line list ───────────────────────

#[test]
fn tier1_batch_every_lineage_child_has_one_parent_not_itself() {
    let mut m = load_fixture("sir_lineage");
    set_params(&mut m, &[("beta", 0.9), ("gamma", 0.2), ("N0", 500.0)]);
    for backend in [Backend::TauLeap, Backend::ChainBinomial] {
        let mut total = 0usize;
        for seed in [1u64, 2, 3, 4, 5] {
            let (_, entries, _, _) = run_with_lineage(&m, backend, seed, 0.25);
            let mut parents: HashMap<u64, u64> = HashMap::new();
            for e in &entries {
                if let ParentRef::Individual(p) = e.parent {
                    total += 1;
                    let prev = parents.insert(e.individual.0, p.0);
                    assert!(prev.is_none(), "individual recorded as child twice");
                    assert_ne!(p.0, e.individual.0, "individual cannot be its own parent");
                }
            }
        }
        assert!(total > 0, "expected lineage events on a batch backend");
    }
}

#[test]
fn tier1_batch_parent_never_a_same_step_child() {
    // The frozen-snapshot guarantee: a parent recorded for a lineage edge must
    // have been born strictly before the step in which it acts as a parent (it
    // was already in the snapshot pool). Operationally: replay in time order
    // and assert each recorded parent was first seen as a focal individual at a
    // strictly earlier time than the child's event (or is the t=0 seed).
    let mut m = load_fixture("sir_lineage");
    set_params(&mut m, &[("beta", 0.9), ("gamma", 0.2), ("N0", 500.0)]);
    for backend in [Backend::TauLeap, Backend::ChainBinomial] {
        let (_, mut entries, _, _) = run_with_lineage(&m, backend, 3, 0.5);
        entries.sort_by(|a, b| a.time.partial_cmp(&b.time).unwrap());
        // Birth time of each individual (earliest time it appears as focal).
        let mut born: HashMap<u64, f64> = HashMap::new();
        born.insert(0, 0.0); // initial infective seeded at t=0
        for e in &entries {
            if let ParentRef::Individual(p) = e.parent {
                let parent_born = born.get(&p.0).copied();
                assert!(
                    parent_born.is_some(),
                    "{:?}: parent {} of child {} was never born before the edge at t={}",
                    backend_name(backend),
                    p.0,
                    e.individual.0,
                    e.time
                );
                // The parent must have been born at a STRICTLY earlier step.
                // With the frozen start-of-step snapshot, the parent was in the
                // pool at step start, so a same-step child (born at this same
                // step time) is invisible as a parent. Because all events in a
                // batched step share the step time, "earlier step" ⇒ strictly
                // smaller birth time. A non-strict check would silently admit a
                // same-step-parent regression.
                assert!(
                    parent_born.unwrap() < e.time - 1e-12,
                    "{:?}: parent {} born at {} is not from a strictly earlier \
                     step than the child edge at {} — same-step parenthood means \
                     the frozen-pool snapshot leaked",
                    backend_name(backend),
                    p.0,
                    parent_born.unwrap(),
                    e.time
                );
            }
            born.entry(e.individual.0).or_insert(e.time);
        }
    }
}

fn backend_name(b: Backend) -> &'static str {
    match b {
        Backend::TauLeap => "tau_leap",
        Backend::ChainBinomial => "chain_binomial",
    }
}

// ── Sub-dt bias diagnostic ──────────────────────────────────────────────────────

#[test]
fn sub_dt_fraction_in_unit_interval_and_positive_on_real_epidemic() {
    let mut m = load_fixture("sir_lineage");
    set_params(&mut m, &[("beta", 1.2), ("gamma", 0.2), ("N0", 2000.0)]);
    // A coarse dt on a fast epidemic should produce a measurable sub-dt fraction
    // on chain-binomial (many infections crowd into single steps).
    let (_, _, frac, edges) = run_with_lineage(&m, Backend::ChainBinomial, 11, 1.0);
    assert!(edges > 0, "need transmission edges to measure bias");
    assert!(
        (0.0..=1.0).contains(&frac),
        "sub-dt fraction must be a probability, got {}",
        frac
    );
    assert!(
        frac > 0.0,
        "a coarse dt on a fast epidemic should lose some sub-dt edges; frac={}",
        frac
    );
}

#[test]
fn sub_dt_fraction_grows_with_dt() {
    // Edge-weighted sub-dt fraction must be monotone-ish in dt: a coarser step
    // crowds more children into a step, raising m/(p+m). We assert the coarse-dt
    // fraction strictly exceeds the fine-dt fraction (averaged over seeds to tame
    // Monte-Carlo noise).
    let mut m = load_fixture("sir_lineage");
    set_params(&mut m, &[("beta", 1.2), ("gamma", 0.2), ("N0", 2000.0)]);
    let mut fine_sum = 0.0;
    let mut coarse_sum = 0.0;
    let seeds = [1u64, 2, 3, 4, 5, 6, 7, 8];
    for &seed in &seeds {
        let (_, _, fine, _) = run_with_lineage(&m, Backend::ChainBinomial, seed, 0.1);
        let (_, _, coarse, _) = run_with_lineage(&m, Backend::ChainBinomial, seed, 2.0);
        fine_sum += fine;
        coarse_sum += coarse;
    }
    let fine_avg = fine_sum / seeds.len() as f64;
    let coarse_avg = coarse_sum / seeds.len() as f64;
    assert!(
        coarse_avg > fine_avg,
        "sub-dt fraction should grow with dt: fine(dt=0.1)={:.4} vs coarse(dt=2.0)={:.4}",
        fine_avg,
        coarse_avg
    );
}

#[test]
fn batch_line_list_reproducible_given_seed() {
    let mut m = load_fixture("sir_lineage");
    set_params(&mut m, &[("beta", 0.7), ("gamma", 0.2), ("N0", 500.0)]);
    for backend in [Backend::TauLeap, Backend::ChainBinomial] {
        let (_, e1, f1, _) = run_with_lineage(&m, backend, 55, 0.5);
        let (_, e2, f2, _) = run_with_lineage(&m, backend, 55, 0.5);
        assert_eq!(e1, e2, "same seed must yield identical batch line lists");
        assert_eq!(f1, f2, "sub-dt fraction must be reproducible");
    }
}
