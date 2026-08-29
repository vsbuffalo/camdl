//! gh#772 — the correlated PF draws x₀ from the pre-drawn correlated vector,
//! and draws the SAME laws the ordinary path draws.
//!
//! Correlated PMMH stores every random as a standard normal and transforms it
//! at consumption time, so an `init { }` law has to be inverted from a normal
//! rather than sampled from a stream. That is a second sampler for the same
//! four laws, and the thing to be afraid of is that it silently samples
//! something slightly different: a CPM arm would then be a different target
//! from the PGAS arm it is being compared against, which is the failure gh#372
//! shipped for the transition-kernel Gamma (a Wilson–Hilferty approximation
//! that biased the multiplier low and clamped a growing fraction of draws to
//! zero).
//!
//! What is asserted here, on `ocaml/golden/init_laws.camdl` — the fixture that
//! exercises all four admissible laws in one block:
//!
//! 1. **The two samplers agree distributionally**, law by law, in mean and in
//!    variance, to within Monte Carlo error over 20,000 draws.
//! 2. **Every slot of a particle's block changes the state it draws** — the
//!    offset table is a bijection onto the block. Asserted at the producer
//!    rather than through the filter, because the real-valued `W` consumes a
//!    slot that no particle filter can show in a log-likelihood: none of them
//!    advance a real compartment
//!    (`docs/dev/incidents/2026-06-07-chain-binomial-stale-real-state.md`).
//!    The filter-level twin, over the integer laws, is
//!    `cpm_reads_every_slot_of_every_init_block` in `tests/pmmh.rs`.
//! 3. **The same normals give the same state**, which is the property the whole
//!    method rests on: the same random reused at the same (particle,
//!    compartment) across MCMC iterations.
//! 4. **A resume state written before this change is refused, not misread.**

use serde::Serialize;

use sim::{
    compiled_model::CompiledModel,
    inference::pmmh::PMMHResumeState,
    rng::StatefulRng,
};

const N_DRAWS: usize = 20_000;
const SEED: u64 = 20260829;

fn load(rel: &str) -> ir::Model {
    let path = format!("{}/../../../{}", env!("CARGO_MANIFEST_DIR"), rel);
    let json = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("read {path}: {e}"));
    ir::from_str(&json).unwrap_or_else(|e| panic!("parse {path}: {e}"))
}

/// `ocaml/golden/init_laws.camdl` under its `baseline` scenario: an SEIR whose
/// `init { }` draws `I ~ poisson`, `E ~ neg_binomial`, `R ~ binomial` and
/// `W ~ normal`, and computes `S = N0 - I - E - R` from what they drew.
fn law_fixture() -> (CompiledModel, Vec<f64>) {
    let mut model = load("ocaml/golden/init_laws.ir.json");
    let preset = model.presets.first().cloned().expect("the fixture declares a scenario");
    for p in &mut model.parameters {
        if let Some(&v) = preset.params.get(&p.name) {
            p.value = p.value.with_value(v);
        }
    }
    let compiled = CompiledModel::new(model).expect("fixture must compile");
    let params = compiled.default_params.clone();
    (compiled, params)
}

fn int_idx(c: &CompiledModel, name: &str) -> usize {
    c.global_to_int[c.comp_index[name]].expect("an integer compartment")
}

fn real_idx(c: &CompiledModel, name: &str) -> usize {
    c.global_to_real[c.comp_index[name]].expect("a real compartment")
}

fn mean_var(xs: &[f64]) -> (f64, f64) {
    let m = xs.iter().sum::<f64>() / xs.len() as f64;
    let v = xs.iter().map(|x| (x - m).powi(2)).sum::<f64>() / xs.len() as f64;
    (m, v)
}

/// The block's shape, read off the model rather than assumed: one normal per
/// law, two for the NegBinomial (a Gamma multiplier mixed into a Poisson), none
/// for the deterministic entry — and the offsets partition the block.
#[test]
fn the_init_noise_block_is_one_slot_per_draw_and_two_for_neg_binomial() {
    let (compiled, _) = law_fixture();
    assert!(compiled.has_init_law);
    assert_eq!(
        compiled.init_noise_width, 5,
        "poisson 1 + neg_binomial 2 + binomial 1 + normal 1"
    );

    let ic = &compiled.model.initial_conditions;
    for (i, (name, spec)) in ic.iter().enumerate() {
        let off = compiled.init_noise_offsets[i];
        if spec.is_law() {
            assert!(off.is_some(), "{name} is drawn but has no slot");
        } else {
            assert!(off.is_none(), "{name} is deterministic but was given slot {off:?}");
        }
    }

    // Every slot in [0, width) is the start of exactly one entry's normals, and
    // the entries do not overlap.
    let mut covered = vec![0usize; compiled.init_noise_width];
    for (i, (_, spec)) in ic.iter().enumerate() {
        if let Some(off) = compiled.init_noise_offsets[i] {
            let w = if matches!(
                spec,
                ir::model::InitSpec::Count(ir::model::InitCountLaw::NegBinomial(_))
            ) { 2 } else { 1 };
            for c in covered.iter_mut().skip(off).take(w) {
                *c += 1;
            }
        }
    }
    assert!(
        covered.iter().all(|&c| c == 1),
        "the offsets must tile the block exactly once each, got {covered:?}"
    );
}

/// The two samplers draw the same laws.
///
/// Not "both produce numbers" — mean AND variance, law by law, against each
/// other, with the tolerance set by the Monte Carlo error of the comparison
/// rather than by taste. `S` is in the table too: it is deterministic, but it
/// reads what the three count laws DREW, so a disagreement in any of them shows
/// up in its spread as well.
#[test]
fn the_correlated_draw_samples_the_same_laws_as_the_stream_draw() {
    let (compiled, params) = law_fixture();
    let width = compiled.init_noise_width;

    let cols = ["I", "E", "R", "S"];
    let mut stream: Vec<Vec<f64>> = vec![Vec::with_capacity(N_DRAWS); cols.len()];
    let mut stream_w: Vec<f64> = Vec::with_capacity(N_DRAWS);
    let mut corr: Vec<Vec<f64>> = vec![Vec::with_capacity(N_DRAWS); cols.len()];
    let mut corr_w: Vec<f64> = Vec::with_capacity(N_DRAWS);

    let mut rng = StatefulRng::new(SEED);
    for _ in 0..N_DRAWS {
        let (int_s, real_s) = compiled.initial_state_draw(&params, &mut rng)
            .expect("stream draw");
        for (k, name) in cols.iter().enumerate() {
            stream[k].push(int_s.counts[int_idx(&compiled, name)] as f64);
        }
        stream_w.push(real_s.values[real_idx(&compiled, "W")]);
    }

    let mut rng = StatefulRng::new(SEED ^ 0xF0F0);
    for _ in 0..N_DRAWS {
        let z: Vec<f64> = (0..width).map(|_| rng.normal()).collect();
        let (int_s, real_s) = compiled.initial_state_draw_correlated(&params, &z)
            .expect("correlated draw");
        for (k, name) in cols.iter().enumerate() {
            corr[k].push(int_s.counts[int_idx(&compiled, name)] as f64);
        }
        corr_w.push(real_s.values[real_idx(&compiled, "W")]);
    }

    let check = |name: &str, a: &[f64], b: &[f64]| {
        let (ma, va) = mean_var(a);
        let (mb, vb) = mean_var(b);
        eprintln!(
            "{name}: stream mean {ma:.4} var {va:.4} | correlated mean {mb:.4} var {vb:.4}"
        );
        // Four standard errors of the difference of two independent sample
        // means, each of variance ~va/N.
        let se = (2.0 * va.max(vb) / N_DRAWS as f64).sqrt();
        assert!(
            (ma - mb).abs() < 4.0 * se,
            "{name}: means disagree — stream {ma}, correlated {mb} (4 SE = {})",
            4.0 * se
        );
        // The sample variance of a sample variance is ~2 sigma^4 / N for a
        // normal; 10% is many standard errors of that at N = 20000 and is
        // loose enough not to depend on the laws' fourth moments.
        assert!(
            (va - vb).abs() < 0.10 * va.max(vb),
            "{name}: variances disagree by more than 10% — stream {va}, \
             correlated {vb}"
        );
        assert!(va > 0.0, "{name}: the stream sampler produced no spread at all");
    };
    for (k, name) in cols.iter().enumerate() {
        check(name, &stream[k], &corr[k]);
    }
    check("W", &stream_w, &corr_w);
}

/// Every slot of a particle's block changes the state it draws — the offset
/// table is onto the block, so no two entries share a normal and none goes
/// unread. Covers the real-valued `W`, which the filter cannot show.
#[test]
fn every_slot_of_a_particles_block_changes_the_drawn_state() {
    let (compiled, params) = law_fixture();
    let width = compiled.init_noise_width;

    let zeros = vec![0.0; width];
    let (base_int, base_real) = compiled.initial_state_draw_correlated(&params, &zeros)
        .expect("baseline draw");

    // Far into the tail on whichever law the slot belongs to.
    const LARGE_Z: f64 = 6.0;
    for slot in 0..width {
        let mut z = zeros.clone();
        z[slot] = LARGE_Z;
        let (int_s, real_s) = compiled.initial_state_draw_correlated(&params, &z)
            .expect("perturbed draw");
        assert!(
            int_s.counts != base_int.counts || real_s.values != base_real.values,
            "slot {slot} of {width} is never read — the offset table is not onto \
             the block, so two entries share a normal"
        );
    }
}

/// The same normals give the same state, every time. This is the property the
/// whole method rests on: the same random reused at the same (particle,
/// compartment) across MCMC iterations, so the two likelihood estimates in the
/// acceptance ratio share their noise.
#[test]
fn the_same_normals_give_the_same_initial_state() {
    let (compiled, params) = law_fixture();
    let mut rng = StatefulRng::new(SEED);
    let z: Vec<f64> = (0..compiled.init_noise_width).map(|_| rng.normal()).collect();

    let (a_int, a_real) = compiled.initial_state_draw_correlated(&params, &z).expect("draw");
    let (b_int, b_real) = compiled.initial_state_draw_correlated(&params, &z).expect("draw");
    assert_eq!(a_int.counts, b_int.counts);
    assert_eq!(a_real.values, b_real.values);

    // And it is a draw, not the mean: the fixture's `I` averages 50 with
    // variance 50, so a draw landing exactly on every mean would be a
    // one-in-a-lifetime coincidence — but say it plainly rather than trusting
    // that, by moving one normal and requiring the state to move.
    let mut z2 = z.clone();
    z2[0] += 3.0;
    let (c_int, _) = compiled.initial_state_draw_correlated(&params, &z2).expect("draw");
    assert_ne!(a_int.counts, c_int.counts, "the draw does not depend on the normals");
}

/// A resume state written before the init block existed is REFUSED, not
/// misread.
///
/// `PFRandomState` is bincode-serialised inside `PMMHResumeState`
/// (`chain_*/resume_state.bin`), so adding `init_noise` and `init_width`
/// changes the byte layout. bincode carries no field names and no version, so
/// the question is not whether old files stop working — they must — but
/// whether they stop working LOUDLY. `fit/pmmh.rs` reports "cannot deserialize
/// resume state for chain N … re-run with --force" and exits on a
/// deserialization error; a silent misread would instead hand the sampler
/// another field's bytes as its correlated vector.
///
/// Both halves are asserted, because only one of them changed: a correlated
/// chain's state (`current_randoms = Some`) must fail, and a vanilla PMMH
/// chain's (`None`) must still load, since the `Option` tag is all that was
/// written for it.
#[test]
fn a_resume_state_predating_the_init_block_is_refused_not_misread() {
    /// The pre-gh#772 `PFRandomState`, field for field.
    #[derive(Serialize)]
    struct LegacyPFRandomState {
        gamma_noise: Vec<Vec<f64>>,
        resample_noise: Vec<f64>,
        binomial_noise: Vec<Vec<f64>>,
        n_source_groups: usize,
    }

    /// `PMMHResumeState` with the legacy randoms in place of the current ones.
    /// `adaptive` is `None` here, which bincode writes as the same one-byte tag
    /// whatever the payload type would have been.
    #[derive(Serialize)]
    struct LegacyResumeState {
        config_hash: String,
        completed_steps: usize,
        params: Vec<f64>,
        transformed: Vec<f64>,
        param_names: Vec<String>,
        current_ll: f64,
        current_log_prior: f64,
        n_accepted: usize,
        adaptive: Option<()>,
        current_randoms: Option<LegacyPFRandomState>,
        map_params: Vec<f64>,
        map_loglik: f64,
        map_log_posterior: f64,
    }

    let legacy = |randoms: Option<LegacyPFRandomState>| LegacyResumeState {
        config_hash: "deadbeef".into(),
        completed_steps: 5000,
        params: vec![0.3, 0.1],
        transformed: vec![-1.2, -2.3],
        param_names: vec!["beta".into(), "gamma".into()],
        current_ll: -4755.9,
        current_log_prior: -3.2,
        n_accepted: 1234,
        adaptive: None,
        current_randoms: randoms,
        map_params: vec![0.31, 0.11],
        map_loglik: -4750.1,
        map_log_posterior: -4753.3,
    };

    let correlated = legacy(Some(LegacyPFRandomState {
        gamma_noise: vec![vec![0.1, 0.2], vec![0.3, 0.4]],
        resample_noise: vec![0.5, 0.6],
        binomial_noise: vec![vec![0.7, 0.8], vec![0.9, 1.0]],
        n_source_groups: 2,
    }));
    let bytes = bincode::serialize(&correlated).expect("legacy state serialises");
    let decoded = bincode::deserialize::<PMMHResumeState>(&bytes);
    assert!(
        decoded.is_err(),
        "a correlated-PMMH resume state written before the init block must fail \
         to deserialize — the CLI turns that into 'cannot deserialize resume \
         state … re-run with --force'. It decoded instead, which means the \
         sampler would resume on another field's bytes."
    );

    let vanilla = legacy(None);
    let bytes = bincode::serialize(&vanilla).expect("legacy state serialises");
    let decoded = bincode::deserialize::<PMMHResumeState>(&bytes)
        .expect("a vanilla PMMH resume state carries no randoms and still loads");
    assert_eq!(decoded.completed_steps, 5000);
    assert_eq!(decoded.map_params, vec![0.31, 0.11]);
    assert!(decoded.current_randoms.is_none());
}
