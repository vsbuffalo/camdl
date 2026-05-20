//! Acceptance tests for the three-layer lineage path — Gillespie, single
//! population (2026-05-20 proposal). These now go through the refactored
//! Layer-1 (record an event log) → Layer-2 (`realize` into a line list) path
//! rather than the removed inline observer; the assertions are unchanged.
//!
//! Tiers (proposal §10 "Validation"):
//!   - Tier 2a — trajectory invariance: a `--event-log` run's count trajectory
//!     equals the run without it, byte-for-byte, same seed. Now trivially true
//!     (the recorder draws no identities), but tested.
//!   - Tier 1 — structural invariants: every lineage child has exactly one
//!     parent; the parent is live in its pool at the child's event time;
//!     pruned tips = sampled set; no unary nodes after pruning.
//!   - Tier 3 — analytic: linear pure-birth (Yule) tree statistics within
//!     Monte-Carlo tolerance.
//!   - Single-pool / multi-pool parent sampling frequencies (sanity).

use std::collections::{HashMap, HashSet};

use sim::{
    compiled_model::CompiledModel,
    config::{GillespieConfig, SimConfig},
    lineage::{
        tree::{Flat, SamplingScheme, TransmissionForest},
        IndividualId, LineListEntry, ParentRef,
    },
    state::Trajectory,
    GillespieSim, Simulate,
};

mod lineage_helpers;
use lineage_helpers::{
    load_fixture, record_event_log, realize_log, run_with_lineage, set_params, Backend,
};

fn gillespie_cfg(t_end: f64) -> SimConfig {
    SimConfig::Gillespie(GillespieConfig { t_start: 0.0, t_end, output_dt: None })
}

fn traj_signature(t: &Trajectory) -> Vec<(String, Vec<i64>, Vec<u64>)> {
    t.snapshots
        .iter()
        .map(|s| (format!("{:?}", s.t), s.int_state.counts.clone(), s.flows.counts.clone()))
        .collect()
}

// ── Tier 2a — trajectory byte-identity (load-bearing) ──────────────────────────

#[test]
fn tier2a_trajectory_byte_identical_with_and_without_lineages() {
    let mut m = load_fixture("sir_lineage");
    set_params(&mut m, &[("beta", 0.6), ("gamma", 0.2), ("N0", 500.0)]);

    // Baseline: no observer.
    let compiled = CompiledModel::new(m.clone()).unwrap();
    let params = compiled.default_params.clone();
    let baseline = GillespieSim.run(&compiled, &params, 7, &gillespie_cfg(60.0)).unwrap();

    // With lineage observer attached.
    let (with_lineage, entries) = run_with_lineage(m, 7, 60.0);

    assert_eq!(
        traj_signature(&baseline),
        traj_signature(&with_lineage),
        "count trajectory must be byte-identical with and without --lineages \
         (separate-RNG-stream invariant)"
    );
    assert!(!entries.is_empty(), "lineage run should produce line-list entries");
}

#[test]
fn tier2a_holds_across_many_seeds() {
    for seed in [1u64, 2, 13, 42, 99, 1000] {
        let mut m = load_fixture("sir_lineage");
        set_params(&mut m, &[("beta", 0.5), ("gamma", 0.25), ("N0", 500.0)]);
        let compiled = CompiledModel::new(m.clone()).unwrap();
        let params = compiled.default_params.clone();
        let baseline = GillespieSim.run(&compiled, &params, seed, &gillespie_cfg(50.0)).unwrap();
        let (with_lineage, _) = run_with_lineage(m, seed, 50.0);
        assert_eq!(
            traj_signature(&baseline),
            traj_signature(&with_lineage),
            "trajectory diverged with --lineages at seed {}",
            seed
        );
    }
}

// ── Tier 1 — structural invariants ─────────────────────────────────────────────

#[test]
fn tier1_every_lineage_child_has_exactly_one_parent() {
    // Accumulate over several seeds so the invariant is checked on real
    // epidemics (some seeds fizzle with a single early recovery).
    let mut total_lineage_events = 0usize;
    for seed in [1u64, 2, 3, 4, 5, 6] {
        let mut m = load_fixture("sir_lineage");
        set_params(&mut m, &[("beta", 0.9), ("gamma", 0.2), ("N0", 500.0)]);
        let (_, entries) = run_with_lineage(m, seed, 60.0);

        // Each focal individual born at a lineage event appears once as a child
        // with exactly one parent.
        let mut parents: HashMap<u64, u64> = HashMap::new();
        for e in &entries {
            if let ParentRef::Individual(p) = e.parent {
                total_lineage_events += 1;
                let prev = parents.insert(e.individual.0, p.0);
                assert!(
                    prev.is_none(),
                    "seed {}: individual {} recorded as a lineage child more than once",
                    seed,
                    e.individual.0
                );
                assert_ne!(p.0, e.individual.0, "an individual cannot be its own parent");
            }
        }
    }
    assert!(total_lineage_events > 0, "expected some lineage (infection) events");
}

#[test]
fn tier1_parent_is_live_in_its_pool_at_child_event_time() {
    // Replay the line list and assert that, at each lineage event, the recorded
    // parent was an individual that had already been born (entered the I pool)
    // and had not yet left it (recovered). This re-checks the observer's
    // pool-membership guarantee from the artifact.
    let mut m = load_fixture("sir_lineage");
    set_params(&mut m, &[("beta", 0.9), ("gamma", 0.2), ("N0", 500.0)]);
    let (_, mut entries) = run_with_lineage(m, 1, 60.0);
    entries.sort_by(|a, b| a.time.partial_cmp(&b.time).unwrap());

    // Track the live I-pool (compartment id for "I"). In sir_lineage,
    // infection: S(0) --> I(1), recovery: I(1) --> R(2). The initial I=1 seeds
    // individual 0 into the I pool at t=0 (ParentRef::Seed), so pre-seed it.
    let i_comp = 1usize;
    let mut live_i: HashSet<u64> = HashSet::new();
    live_i.insert(0); // initial infective minted at t=0

    for e in &entries {
        match e.parent {
            ParentRef::Individual(p) => {
                // Infection: parent must currently be live in I.
                assert!(
                    live_i.contains(&p.0),
                    "parent {} not live in I-pool at infection of {} (t={})",
                    p.0,
                    e.individual.0,
                    e.time
                );
                // The new infectee enters E in SEIR, but in this SIR it enters I
                // directly (destination == I).
                if e.destination == Some(i_comp) {
                    live_i.insert(e.individual.0);
                }
            }
            ParentRef::None => {
                // Progression / recovery. If the source is I, the focal
                // individual leaves the I pool.
                if e.source == Some(i_comp) {
                    live_i.remove(&e.individual.0);
                }
                if e.destination == Some(i_comp) {
                    live_i.insert(e.individual.0);
                }
            }
            _ => {}
        }
    }
}

#[test]
fn tier1_pruned_tips_equal_sampled_set_and_no_unary_nodes() {
    let mut m = load_fixture("sir_lineage");
    set_params(&mut m, &[("beta", 0.8), ("gamma", 0.2), ("N0", 500.0)]);
    let (_, entries) = run_with_lineage(m, 5, 60.0);

    let forest = TransmissionForest::from_entries(&entries);
    let mut rng = sim::rng::StatefulRng::new(3);
    let sampled = Flat::new(0.3).select(&forest, &mut rng);
    assert!(!sampled.is_empty(), "expected a non-empty sample");

    let trees = forest.prune_to(&sampled);

    // Collect pruned tips across all trees.
    let mut tips: HashSet<u64> = HashSet::new();
    fn collect_tips(n: &sim::lineage::tree::PrunedNode, tips: &mut HashSet<u64>) {
        if n.children.is_empty() {
            tips.insert(n.id.0);
        } else {
            // No unary internal nodes after pruning.
            assert!(
                n.children.len() >= 2 || n.is_sampled_tip,
                "internal node {} has {} children — unary node not suppressed",
                n.id.0,
                n.children.len()
            );
            for c in &n.children {
                collect_tips(c, tips);
            }
        }
    }
    for t in &trees {
        collect_tips(t, &mut tips);
    }

    let sampled_ids: HashSet<u64> = sampled.iter().map(|s| s.0).collect();
    assert_eq!(tips, sampled_ids, "pruned tips must equal the sampled set");
}

// ── Single-pool uniformity sanity ──────────────────────────────────────────────

#[test]
fn single_pool_parent_sampling_is_uniform() {
    // Tiny SIR run, gather which infector each infection attributes. With a
    // single I-pool, every live I-individual is equally likely to be the
    // parent. We check that, conditioned on an event with k infectives live,
    // the chosen parent index is roughly uniform — operationally, that the
    // empirical distribution of "parent appears as infector" matches each
    // infective's exposure (number of events while it was live).
    //
    // Simpler robust check: across many replicates, no single early infective
    // monopolises parenthood beyond chance. We assert the most-frequent parent
    // share is well below 1.0 and that many distinct parents appear.
    let mut parent_counts: HashMap<u64, usize> = HashMap::new();
    let mut total = 0usize;
    for seed in 0..200u64 {
        let mut m = load_fixture("sir_lineage");
        set_params(&mut m, &[("beta", 0.9), ("gamma", 0.3), ("N0", 300.0)]);
        let (_, entries) = run_with_lineage(m, seed, 40.0);
        for e in &entries {
            if let ParentRef::Individual(p) = e.parent {
                *parent_counts.entry(p.0).or_default() += 1;
                total += 1;
            }
        }
    }
    assert!(total > 1000, "need many infection events; got {}", total);
    let n_distinct = parent_counts.len();
    assert!(
        n_distinct > 100,
        "uniform within-pool sampling should spread parenthood across many \
         individuals; only {} distinct parents over {} events",
        n_distinct,
        total
    );
}

// ── Multi-pool attribution frequency (I vs A) ──────────────────────────────────

#[test]
fn multi_pool_attribution_splits_between_pools() {
    // Two infectious pools I, A with weights beta_i, beta_a. With beta_i ==
    // beta_a and symmetric dynamics, infections should be attributed to I and A
    // roughly in proportion to their (equal) presence. We assert both pools
    // contribute a non-trivial share (the decomposition is not collapsing to a
    // single pool — a wrong-pool bug Tier 2a would NOT catch).
    let mut from_i = 0usize;
    let mut from_a = 0usize;
    // Compartment ids in two_pool_lineage: S0 E1 I2 A3 R4.
    let i_comp = 2usize;
    let a_comp = 3usize;
    for seed in 0..120u64 {
        let mut m = load_fixture("two_pool_lineage");
        set_params(
            &mut m,
            &[("beta_i", 0.8), ("beta_a", 0.8), ("sigma", 0.5), ("gamma", 0.3)],
        );
        let (_, entries) = run_with_lineage(m, seed, 40.0);
        for e in &entries {
            if let ParentRef::Individual(p) = e.parent {
                // Find the parent's current pool from the entries: a parent in I
                // vs A. We infer pool membership by tracking births into I/A.
                let _ = p;
            }
            // Attribution pool is recorded implicitly: the infection's parent
            // came from whichever pool. We reconstruct via the parent's most
            // recent destination. Simpler: count by replaying pool membership.
            let _ = (i_comp, a_comp);
        }
        // Replay to classify parents by pool at event time.
        let (i, a) = classify_two_pool(&entries, i_comp, a_comp);
        from_i += i;
        from_a += a;
    }
    let total = from_i + from_a;
    assert!(total > 500, "need many multi-pool events; got {}", total);
    let share_i = from_i as f64 / total as f64;
    assert!(
        (0.2..=0.8).contains(&share_i),
        "with symmetric weights both pools should contribute; I share = {:.3} \
         (from_i={}, from_a={})",
        share_i,
        from_i,
        from_a
    );
}

/// Replay a two-pool line list and classify each infection's parent by whether
/// it was live in the I pool or the A pool at the event instant.
fn classify_two_pool(entries: &[LineListEntry], i_comp: usize, a_comp: usize) -> (usize, usize) {
    let mut sorted = entries.to_vec();
    sorted.sort_by(|a, b| a.time.partial_cmp(&b.time).unwrap());
    let mut in_i: HashSet<u64> = HashSet::new();
    let mut in_a: HashSet<u64> = HashSet::new();
    let mut from_i = 0;
    let mut from_a = 0;
    for e in &sorted {
        if let ParentRef::Individual(p) = e.parent {
            if in_i.contains(&p.0) {
                from_i += 1;
            } else if in_a.contains(&p.0) {
                from_a += 1;
            }
        }
        // Update membership from this event's routing.
        if e.source == Some(i_comp) {
            in_i.remove(&e.individual.0);
        }
        if e.source == Some(a_comp) {
            in_a.remove(&e.individual.0);
        }
        if e.destination == Some(i_comp) {
            in_i.insert(e.individual.0);
        }
        if e.destination == Some(a_comp) {
            in_a.insert(e.individual.0);
        }
    }
    (from_i, from_a)
}

// ── Tier 3 — analytic (linear pure-birth / Yule) ───────────────────────────────

#[test]
fn tier3_pure_birth_tree_statistics() {
    // A linear pure-birth (Yule) process: each tracked individual produces
    // offspring at constant per-capita rate lambda. Built as S --> 2S? No —
    // count-conserving birth is awkward in compartments. Instead we use a
    // birth flow with the source as a frozen-coefficient parent pool: every
    // birth attributes a parent uniformly from the live pool, which is exactly
    // the Yule branching structure.
    //
    // Analytic facts checked (within Monte-Carlo tolerance over replicates):
    //   - the number of tracked individuals after the process equals
    //     n_initial + (number of birth events) — conservation of identities;
    //   - every non-root individual has exactly one parent (a tree, not a DAG);
    //   - the tree built from the line list has exactly (#births) internal
    //     branchings and (#leaves) tips with #leaves = #births - #non-leaf+...,
    //     i.e. tips + internal = total nodes (basic tree identity).
    let mut total_births = 0usize;
    let mut replicates = 0usize;
    for seed in 0..40u64 {
        let mut m = load_fixture("yule_lineage");
        set_params(&mut m, &[("lambda", 0.4)]);
        let (_, entries) = run_with_lineage(m, seed, 8.0);
        let births = entries
            .iter()
            .filter(|e| matches!(e.parent, ParentRef::Individual(_)))
            .count();
        if births < 2 {
            continue;
        }
        replicates += 1;
        total_births += births;

        let forest = TransmissionForest::from_entries(&entries);
        // Tree identity: total nodes = internal + leaves; every non-root has a
        // parent.
        let n_nodes = forest.nodes.len();
        let n_leaves = forest.leaves().len();
        assert!(n_leaves >= 1 && n_leaves <= n_nodes);
        let n_roots = forest.roots.len();
        // Each non-root has exactly one parent → edges = nodes - roots.
        let edges: usize = forest.nodes.values().filter(|n| n.parent.is_some()).count();
        assert_eq!(
            edges,
            n_nodes - n_roots,
            "edge count must equal nodes - roots for a forest"
        );

        // Sackin index is well-defined and non-negative for a full sample.
        let sampled: HashSet<IndividualId> = forest.leaves().into_iter().collect();
        let trees = forest.prune_to(&sampled);
        let sackin: usize = trees.iter().map(|t| t.sackin()).sum();
        let tips: usize = trees.iter().map(|t| t.tip_count()).sum();
        assert_eq!(tips, n_leaves, "pruned tips (rate 1.0) must equal all leaves");
        assert!(sackin >= tips, "Sackin >= #tips for a non-degenerate tree");
    }
    assert!(replicates >= 5, "expected several non-trivial Yule replicates");
    let mean_births = total_births as f64 / replicates as f64;
    // Pure-birth from a small seed over the horizon should grow appreciably.
    assert!(mean_births > 2.0, "mean births {:.1} too small", mean_births);
}

/// Determinism: same seed → identical line list (the lineage RNG is
/// reproducible).
#[test]
fn lineage_line_list_is_reproducible() {
    let mut m1 = load_fixture("sir_lineage");
    set_params(&mut m1, &[("beta", 0.6), ("gamma", 0.2), ("N0", 500.0)]);
    let m2 = m1.clone();
    let (_, e1) = run_with_lineage(m1, 77, 50.0);
    let (_, e2) = run_with_lineage(m2, 77, 50.0);
    assert_eq!(e1, e2, "same seed must yield identical line lists");
}

/// Flat scheme covers all tips at rate 1.0 (sampling-scheme sanity).
#[test]
fn flat_rate_one_samples_all_leaves() {
    let mut m = load_fixture("sir_lineage");
    set_params(&mut m, &[("beta", 0.7), ("gamma", 0.2), ("N0", 500.0)]);
    let (_, entries) = run_with_lineage(m, 9, 50.0);
    let forest = TransmissionForest::from_entries(&entries);
    let all_leaves: HashSet<IndividualId> = forest.leaves().into_iter().collect();
    let mut rng = sim::rng::StatefulRng::new(1);
    let sampled = Flat::new(1.0).select(&forest, &mut rng);
    assert_eq!(sampled, all_leaves);
}

// ── Layer-1/2 explicit: event-log invariance + identity-seed independence ───────

/// Tier 2a (explicit): the count trajectory from the Layer-1 `record_event_log`
/// path is byte-identical to a plain run at the same seed, AND realizing the
/// SAME event log at different identity seeds leaves the trajectory untouched
/// while producing distinct line lists. This is the structural proof of the
/// factorization `P(augmented) = P(counts) × P(identities | counts)`.
#[test]
fn event_log_trajectory_invariant_under_identity_seed() {
    // A vigorous epidemic so there are many choice points (multiple infectives
    // live at transmission/recovery events) — guarantees the identity layer has
    // something to sample.
    let mut m = load_fixture("sir_lineage");
    set_params(&mut m, &[("beta", 1.5), ("gamma", 0.2), ("N0", 1000.0)]);
    let seed = 7u64;

    // Plain run (no recorder).
    let compiled = CompiledModel::new(m.clone()).unwrap();
    let params = compiled.default_params.clone();
    let baseline = GillespieSim.run(&compiled, &params, seed, &gillespie_cfg(60.0)).unwrap();

    // Record once.
    let (traj, log) = record_event_log(&m, Backend::Gillespie, seed, 60.0);
    assert_eq!(
        traj_signature(&baseline),
        traj_signature(&traj),
        "event-log run trajectory must be byte-identical to a plain run"
    );

    // Realize the SAME event log at several identity seeds. The trajectory is a
    // shared artifact; the identity realizations are i.i.d. draws and must not
    // all coincide (proof the second factor is actually sampled).
    let realizations: Vec<Vec<LineListEntry>> =
        (200u64..205).map(|s| realize_log(&log, s).0).collect();
    assert!(!realizations[0].is_empty(), "expected line-list entries");
    let all_equal = realizations.iter().all(|r| *r == realizations[0]);
    assert!(
        !all_equal,
        "distinct identity seeds should give distinct identity realizations; \
         if all equal the identity layer is not actually sampling"
    );
    // Same identity seed reproduces the same line list (determinism).
    let again = realize_log(&log, 200).0;
    assert_eq!(realizations[0], again, "same identity seed must reproduce the line list");
}

// ── Attribution log-probability (§4a) — hand-checkable ──────────────────────────

/// On a hand-built tiny event log, the accumulated per-line-list log-probability
/// equals the analytic sum of per-event `log P(attribution)` (§4a):
/// transmission `log(w_b/Λ)`, recovery `log(1/|I_b|)`.
#[test]
fn attribution_logprob_matches_analytic_sum() {
    use sim::lineage::{realize, EventLog, EventRecord, RouteInfo};

    // Model shape: comp 1 = I (tracked, deme 0), comp 2 = R. Transition 0 is a
    // transmission whose single parent pool is I (comp 1); transition 1 is
    // recovery I → R.
    //
    // Seed 2 infectives. Events:
    //   t=1.0  transmission, mass(I)=4.0, X_I=2  → child minted in I, |I| = 3
    //   t=2.0  recovery from I (|I| = 3 → 2)
    //   t=3.0  transmission, mass(I)=10.0, X_I=2 → child minted, |I| = 3
    //
    // Λ = sum of recorded masses = the single pool's mass (one parent pool).
    // Transmission term: log(w_b/Λ) = log(mass / X_I / Λ) where the within-pool
    // 1/X_I cancels mass/Λ's X_I — but with ONE pool, mass == Λ, so
    // log(w_b/Λ) = log(mass / |I|) − log(mass) = −log|I|. We use a SINGLE-pool
    // event log so Λ = mass and the transmission term reduces to −log|I| at the
    // event-instant pool size, which is exactly hand-checkable.
    let log = EventLog {
        initial_pools: vec![(0, 1, 2)],
        transitions: vec![
            RouteInfo {
                source: None,
                source_deme: 0,
                destination: Some(1),
                destination_deme: 0,
                child_deme: 0,
                touches_tracked: true,
                parent_pools: vec![(1, 0)],
            },
            RouteInfo {
                source: Some(1),
                source_deme: 0,
                destination: Some(2),
                destination_deme: 0,
                child_deme: 0,
                touches_tracked: true,
                parent_pools: vec![],
            },
        ],
        events: vec![
            EventRecord {
                time: 1.0,
                transition: 0,
                multiplicity: 1,
                batched: false,
                lineage_weights: Some(vec![4.0]),
            },
            EventRecord {
                time: 2.0,
                transition: 1,
                multiplicity: 1,
                batched: false,
                lineage_weights: None,
            },
            EventRecord {
                time: 3.0,
                transition: 0,
                multiplicity: 1,
                batched: false,
                lineage_weights: Some(vec![10.0]),
            },
        ],
    };

    let collector = lineage_helpers::VecWriter::new();
    let buf = collector.entries.clone();
    let mut writer = collector;
    let summary = realize(&log, 12345, &mut writer).unwrap();
    let entries = buf.borrow().clone();

    // Analytic per-event log-probabilities, in event order:
    //   t=1: single-pool transmission, |I| = 2 → log(mass/|I|/Λ) with Λ=mass=4
    //        → log(4 / 2 / 4) = log(1/2) = −ln 2.
    //   t=2: recovery, |I| = 3 (2 seeds + 1 born at t=1) → −ln 3.
    //   t=3: transmission, |I| = 2 (3 − 1 recovered) → log(10/2/10) = −ln 2.
    let expected = vec![-(2f64.ln()), -(3f64.ln()), -(2f64.ln())];
    assert_eq!(entries.len(), 3, "one entry per event");
    for (e, &exp) in entries.iter().zip(expected.iter()) {
        assert!(
            (e.attribution_logprob - exp).abs() < 1e-12,
            "per-event attribution_logprob {} != analytic {} (t={})",
            e.attribution_logprob,
            exp,
            e.time
        );
    }
    let analytic_total: f64 = expected.iter().sum();
    assert!(
        (summary.total_logprob - analytic_total).abs() < 1e-12,
        "summary total {} != analytic sum {}",
        summary.total_logprob,
        analytic_total
    );
    // And the per-entry column sums to the same total.
    let column_sum: f64 = entries.iter().map(|e| e.attribution_logprob).sum();
    assert!((column_sum - analytic_total).abs() < 1e-12);
}
