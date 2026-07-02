//! Phase-2 acceptance tests for stratified / spatial parent attribution
//! (2026-05-19 individual-sampling-layer proposal, §"Mathematical structure"
//! — stratified case, and §"Validation" Tier 2b).
//!
//! The load-bearing test is **Tier 2b**: with an asymmetric contact matrix and
//! unequal per-stratum infectious counts, every lineage event's parent must be
//! attributed to stratum `b` with probability
//!
//!     P(b) = w_b · count_b / Σ_b' w_b' · count_b'
//!
//! evaluated at the *event-instant* state, where `w_b` is the per-stratum
//! weight the OCaml compiler emitted (for frequency-dependent stratified
//! transmission `w_b = β·C[a,b]·S[a]/N[b]`; the common `β·S[a]` factor cancels
//! in the ratio, leaving `C[a,b]·I[b]/N[b]`). We reconstruct the exact
//! event-instant per-stratum state by replaying the line list (which records
//! each event's source/destination compartment) on top of the known initial
//! conditions, accumulate the Poisson-binomial expectation of "parent from
//! stratum b" across all events and replicates, and assert the observed count
//! lies within 3σ.
//!
//! The test is constructed to **fail** if attribution were uniform over all
//! infectious (P_unif(b) = I[b] / Σ I[b']): we assert the observed from-`b`
//! count is many σ away from that null. The asymmetric matrix C[a,b]=0.3 vs
//! C[b,a]=1.5 and the unequal seeds I_a=5, I_b=40 make the contact-weighted
//! and uniform predictions diverge sharply.

use std::collections::HashMap;

use sim::lineage::{CompartmentId, DemeId, LineListEntry, ParentRef};

mod lineage_helpers;
use lineage_helpers::{load_fixture, record_event_log, realize_log, set_params, Backend};

/// Record an event log (Gillespie) and realize it at `identity_seed == seed`,
/// returning the realized line list (the Tier-2b attribution test now exercises
/// the realize path, per the 2026-05-20 proposal §10).
fn run_lineage(m: ir::Model, seed: u64, t_end: f64) -> Vec<LineListEntry> {
    let (_, log) = record_event_log(&m, Backend::Gillespie, seed, t_end);
    realize_log(&log, seed).0
}

// Compartment ids in spatial_lineage (S_a, S_b, I_a, I_b, R_a, R_b).
const S_A: usize = 0;
const S_B: usize = 1;
const I_A: usize = 2;
const I_B: usize = 3;
const R_A: usize = 4;
const R_B: usize = 5;
// Transition ids: infection_a, infection_b, recovery_a, recovery_b.
const INFECTION_A: usize = 0;
const INFECTION_B: usize = 1;

// Contact matrix C[row=focal patch, col=infector patch] from the fixture.
//   C = [[1.0, 0.3],   row a: C[a,a]=1.0, C[a,b]=0.3
//        [1.5, 0.2]]   row b: C[b,a]=1.5, C[b,b]=0.2
const C_AA: f64 = 1.0;
const C_AB: f64 = 0.3;
const C_BA: f64 = 1.5;
const C_BB: f64 = 0.2;

/// Exact per-stratum counts, reconstructed by replaying the line list on top
/// of the initial conditions.
#[derive(Clone, Copy)]
struct State {
    s_a: f64,
    i_a: f64,
    r_a: f64,
    s_b: f64,
    i_b: f64,
    r_b: f64,
}
impl State {
    fn initial() -> Self {
        State { s_a: 4000.0, i_a: 5.0, r_a: 0.0, s_b: 2000.0, i_b: 40.0, r_b: 0.0 }
    }
    fn n_a(&self) -> f64 {
        self.s_a + self.i_a + self.r_a
    }
    fn n_b(&self) -> f64 {
        self.s_b + self.i_b + self.r_b
    }
    /// Apply one event's stoichiometry (source -1, destination +1) using the
    /// recorded compartment ids.
    fn apply(&mut self, e: &LineListEntry) {
        if let Some(src) = e.source {
            *self.comp_mut(src.0) -= 1.0;
        }
        if let Some(dst) = e.destination {
            *self.comp_mut(dst.0) += 1.0;
        }
    }
    fn comp_mut(&mut self, c: usize) -> &mut f64 {
        match c {
            S_A => &mut self.s_a,
            S_B => &mut self.s_b,
            I_A => &mut self.i_a,
            I_B => &mut self.i_b,
            R_A => &mut self.r_a,
            R_B => &mut self.r_b,
            _ => unreachable!("unexpected compartment id {c}"),
        }
    }
}

/// Accumulated comparison: observed vs predicted "parent from stratum b".
#[derive(Default)]
struct Tally {
    n_events: usize,
    observed_from_b: f64, // # events whose parent was in patch b
    expected_from_b: f64, // Σ P(parent in b)  (contact-weighted)
    var_from_b: f64,      // Σ P(1-P)          (Poisson-binomial variance)
    expected_from_b_uniform: f64, // Σ P_unif(parent in b)  (the null we must reject)
}

#[test]
fn tier2b_stratified_attribution_matches_contact_weighted_prediction() {
    // Two transitions to test independently: infection_a (focal patch a) and
    // infection_b (focal patch b). For infection into patch `a`, the parent's
    // patch is drawn ∝ (C[a,a]·I_a/N_a, C[a,b]·I_b/N_b); for patch `b`,
    // ∝ (C[b,a]·I_a/N_a, C[b,b]·I_b/N_b).
    let mut tally_a = Tally::default();
    let mut tally_b = Tally::default();

    // Replicates: ≥10⁴ as the proposal requires. Each replicate is a fresh
    // seed; the count dynamics differ but the per-event prediction uses the
    // exact reconstructed event-instant state, so all events pool correctly.
    let n_replicates = 10_000u64;
    // A short horizon keeps the per-stratum populations from collapsing so the
    // weight·count masses stay well-separated, keeps runtime modest, and still
    // yields > 10⁴ lineage events per transition (far more than needed for a
    // tight test).
    let t_end = 3.0;

    for seed in 0..n_replicates {
        let mut m = load_fixture("spatial_lineage");
        set_params(&mut m, &[("beta", 0.6), ("gamma", 0.2)]);
        let mut entries = run_lineage(m, seed, t_end);
        // Events are appended in time order by the Gillespie loop; sort to be
        // safe (ties at identical times are rare and order-independent here).
        entries.sort_by(|x, y| x.time.partial_cmp(&y.time).unwrap());

        let mut st = State::initial();
        for e in &entries {
            // Predict using the state *before* this event's stoichiometry.
            if let ParentRef::Individual(_) = e.parent {
                let parent_deme = e.parent_deme.expect("lineage event must record parent_deme");
                // parent_deme: 0 = patch a, 1 = patch b.
                let from_b = if parent_deme.0 == 1 { 1.0 } else { 0.0 };

                let (c_to_a, c_to_b) = match e.transition.0 {
                    INFECTION_A => (C_AA, C_AB),
                    INFECTION_B => (C_BA, C_BB),
                    other => panic!("unexpected lineage transition {other}"),
                };
                // Contact-weighted masses (β·S[focal] cancels in the ratio).
                let mass_a = c_to_a * st.i_a / st.n_a();
                let mass_b = c_to_b * st.i_b / st.n_b();
                let total = mass_a + mass_b;
                assert!(total > 0.0, "both parent pools empty at a lineage event");
                let p_b = mass_b / total;

                // Uniform-over-all-infectious null: P_unif(b) = I_b/(I_a+I_b),
                // ignoring contact structure and per-stratum normalisers.
                let p_b_uniform = st.i_b / (st.i_a + st.i_b);

                let tally = if e.transition.0 == INFECTION_A { &mut tally_a } else { &mut tally_b };
                tally.n_events += 1;
                tally.observed_from_b += from_b;
                tally.expected_from_b += p_b;
                tally.var_from_b += p_b * (1.0 - p_b);
                tally.expected_from_b_uniform += p_b_uniform;
            }
            st.apply(e);
        }
    }

    for (label, t) in [("infection_a", &tally_a), ("infection_b", &tally_b)] {
        assert!(
            t.n_events > 10_000,
            "{label}: need many lineage events for a tight test; got {}",
            t.n_events
        );
        let sd = t.var_from_b.sqrt();
        let z = (t.observed_from_b - t.expected_from_b) / sd;
        eprintln!(
            "{label}: events={} observed_from_b={:.0} predicted(contact)={:.1} \
             predicted(uniform)={:.1} sd={:.1} z={:.2}",
            t.n_events, t.observed_from_b, t.expected_from_b, t.expected_from_b_uniform, sd, z
        );
        // (1) Observed matches the contact-weighted prediction within 3σ.
        assert!(
            z.abs() < 3.0,
            "{label}: observed parent-from-b count {:.0} deviates {:.2}σ from the \
             contact-weighted prediction {:.1} (sd {:.1}) — stratified attribution \
             is WRONG",
            t.observed_from_b,
            z,
            t.expected_from_b,
            sd
        );
        // (2) The uniform-over-all-infectious null is rejected: the observed
        // count is many σ from it. If attribution were uniform, (1) would fail
        // and (2) would pass — this is the discriminating clause.
        let z_uniform = (t.observed_from_b - t.expected_from_b_uniform) / sd;
        assert!(
            z_uniform.abs() > 10.0,
            "{label}: contact-weighted and uniform predictions are not separated \
             enough to discriminate (uniform pred {:.1}, observed {:.0}, {:.2}σ); \
             the test would not catch a uniform-attribution bug",
            t.expected_from_b_uniform,
            t.observed_from_b,
            z_uniform
        );
    }
}

/// Structural multi-deme invariants on the stratified model: every lineage
/// child's recorded `deme` is its focal patch (infection_a → patch a / deme 0,
/// infection_b → patch b / deme 1), and the recorded `parent_deme` is a valid
/// patch index. Catches a deme-mislabelling bug Tier 2b's frequency test does
/// not directly target.
#[test]
fn multi_deme_structural_invariants() {
    let mut m = load_fixture("spatial_lineage");
    set_params(&mut m, &[("beta", 0.6), ("gamma", 0.2)]);
    let entries = run_lineage(m, 1, 20.0);

    let mut n_lineage = 0;
    for e in &entries {
        match e.parent {
            ParentRef::Individual(_) => {
                n_lineage += 1;
                // Child deme = focal patch of the infection transition.
                match e.transition.0 {
                    INFECTION_A => {
                        assert_eq!(e.deme, DemeId(0), "infection_a child must be in patch a (deme 0)");
                        assert_eq!(e.destination, Some(CompartmentId(I_A)));
                        assert_eq!(e.source, Some(CompartmentId(S_A)));
                    }
                    INFECTION_B => {
                        assert_eq!(e.deme, DemeId(1), "infection_b child must be in patch b (deme 1)");
                        assert_eq!(e.destination, Some(CompartmentId(I_B)));
                        assert_eq!(e.source, Some(CompartmentId(S_B)));
                    }
                    other => panic!("unexpected lineage transition {other}"),
                }
                let pd = e.parent_deme.expect("lineage event records parent_deme");
                assert!(pd.0 <= 1, "parent_deme must be a valid patch index, got {}", pd.0);
            }
            ParentRef::None => {
                // Recovery: I[p] -> R[p], child & no parent_deme.
                assert!(e.parent_deme.is_none(), "non-lineage event must not record parent_deme");
            }
            _ => {}
        }
    }
    assert!(n_lineage > 0, "expected lineage events");
}

/// Parent compartment is correctly stratified: a parent attributed to patch b
/// must have been live in the I_b pool (not I_a) at the child's event time.
/// Reconstructs per-pool membership from the line list and cross-checks
/// `parent_deme` against where the parent id actually lives.
#[test]
fn parent_in_correct_stratum_pool_at_event_time() {
    use std::collections::HashSet;
    let mut m = load_fixture("spatial_lineage");
    set_params(&mut m, &[("beta", 0.6), ("gamma", 0.2)]);
    let mut entries = run_lineage(m, 7, 20.0);
    entries.sort_by(|x, y| x.time.partial_cmp(&y.time).unwrap());

    // Seed the initial I pools with their t=0 individuals. The observer mints
    // ids 0.. by seeding tracked compartments in declaration order:
    // I_a (5), I_b (40), R_a (0), R_b (0). So ids 0..5 are I_a, 5..45 are I_b.
    let mut in_i_a: HashSet<u64> = (0..5).collect();
    let mut in_i_b: HashSet<u64> = (5..45).collect();

    for e in &entries {
        if let ParentRef::Individual(p) = e.parent {
            let pd = e.parent_deme.unwrap();
            if pd.0 == 0 {
                assert!(
                    in_i_a.contains(&p.0),
                    "parent {} labelled patch a (deme 0) but not live in I_a at t={}",
                    p.0,
                    e.time
                );
            } else {
                assert!(
                    in_i_b.contains(&p.0),
                    "parent {} labelled patch b (deme 1) but not live in I_b at t={}",
                    p.0,
                    e.time
                );
            }
        }
        // Update pool membership from this event's routing.
        if e.source == Some(CompartmentId(I_A)) {
            in_i_a.remove(&e.individual.0);
        }
        if e.source == Some(CompartmentId(I_B)) {
            in_i_b.remove(&e.individual.0);
        }
        if e.destination == Some(CompartmentId(I_A)) {
            in_i_a.insert(e.individual.0);
        }
        if e.destination == Some(CompartmentId(I_B)) {
            in_i_b.insert(e.individual.0);
        }
    }
}

/// Sanity: the matrix asymmetry shows up in the *conditional cross-stratum
/// shares*. Among infections whose focal patch is `b`, the share whose parent
/// is in patch `a` (the C[b,a]=1.5 route) is large; among infections whose
/// focal patch is `a`, the share whose parent is in patch `b` (the C[a,b]=0.3
/// route) is small. This is a direct, model-grounded directional reading of
/// the asymmetry — it compares shares (rates), not raw event counts, so it is
/// not confounded by the differing infection volumes per patch.
///
/// Concretely, conditioning on roughly comparable per-stratum normalisers, the
/// b-focal share-from-a tracks C[b,a]·I_a/N_a relative to C[b,b]·I_b/N_b, and
/// the a-focal share-from-b tracks C[a,b]·I_b/N_b relative to C[a,a]·I_a/N_a.
/// With C[b,a]=1.5, C[a,b]=0.3 the former is the larger pull.
#[test]
fn cross_stratum_flow_is_asymmetric() {
    let mut m = load_fixture("spatial_lineage");
    set_params(&mut m, &[("beta", 0.6), ("gamma", 0.2)]);
    let mut a_into_b = 0usize; // infection_b with parent in a
    let mut b_focal = 0usize; // all infection_b events
    let mut b_into_a = 0usize; // infection_a with parent in b
    let mut a_focal = 0usize; // all infection_a events
    let mut counts: HashMap<(usize, u32), usize> = HashMap::new();
    for seed in 0..200u64 {
        let entries = run_lineage(m.clone(), seed, 8.0);
        for e in &entries {
            if let ParentRef::Individual(_) = e.parent {
                let pd = e.parent_deme.unwrap();
                *counts.entry((e.transition.0, pd.0)).or_default() += 1;
                match e.transition.0 {
                    INFECTION_B => {
                        b_focal += 1;
                        if pd.0 == 0 {
                            a_into_b += 1;
                        }
                    }
                    INFECTION_A => {
                        a_focal += 1;
                        if pd.0 == 1 {
                            b_into_a += 1;
                        }
                    }
                    _ => {}
                }
            }
        }
    }
    let share_b_from_a = a_into_b as f64 / b_focal as f64; // C[b,a] route share
    let share_a_from_b = b_into_a as f64 / a_focal as f64; // C[a,b] route share
    eprintln!("cross-stratum counts by (transition, parent_deme): {:?}", counts);
    eprintln!(
        "share(b-focal parent in a) = {:.3} (C[b,a]=1.5 route); \
         share(a-focal parent in b) = {:.3} (C[a,b]=0.3 route)",
        share_b_from_a, share_a_from_b
    );
    // The C[b,a]=1.5 route share must clearly dominate the C[a,b]=0.3 route
    // share — the qualitative signature of the asymmetric contact matrix. A
    // 2.5× margin is comfortably satisfied (observed ≈ 3×) and would be
    // violated by a symmetric or transposed matrix.
    assert!(
        share_b_from_a > 2.5 * share_a_from_b,
        "expected the C[b,a]=1.5 route share ({:.3}) to dominate the C[a,b]=0.3 \
         route share ({:.3}); a symmetric or transposed matrix would not show this",
        share_b_from_a,
        share_a_from_b
    );
}
