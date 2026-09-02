//! **Investigational probe, not suite-ready** (`zz_` prefix). gh#718, the one
//! question the incident report left open.
//!
//! # The question
//!
//! `docs/dev/incidents/2026-08-23-pgas-not-invariant.md`, Part 7, records one
//! claim from the upstream review as explicitly unresolved: that conditional
//! **systematic** resampling combined with ancestor sampling is invalid *even on
//! substeps that do resample*, because the ancestor-sampling accept/reject ratio
//! is derived assuming the resampling picks are independent across slots — true
//! for multinomial, false for systematic.
//!
//! The incident could not settle it: TRAP cannot test it (both its substeps have
//! equal incoming weights, so ancestor sampling is gated off there and the
//! "suppress AS" arm is the control in disguise), and the only evidence either
//! way was DENSE at `z = 1.16` and SPARSE at `z = −0.36`.
//!
//! # Why those two runs could not have settled it
//!
//! The goodness-of-fit statistic the invariance test reports is
//! `z ≈ M·D / sqrt(2·df)`, where `D = Σᵢ (π̃ᵢ − πᵢ)²/πᵢ` is the chi-square
//! divergence between the kernel's stationary law `π̃` and `π` over the scored
//! bins. At fixed divergence `z` falls as `1/sqrt(df)`, and the fixtures'
//! `df` differ by 15×:
//!
//! | fixture | support | scored bins `df` at `M = 400k` | `z` | implied `D` |
//! | ------- | ------: | -----------------------------: | --: | ----------: |
//! | TRAP    |     131 |                            ~126 | 2.36 (a REAL defect) | 9.4e−5 |
//! | DENSE   |    3538 |                           1956 | 1.16 |      1.8e−4 |
//!
//! Read that the right way round. DENSE's `z = 1.16` does **not** bound the
//! divergence below TRAP's — it corresponds to a divergence about **twice**
//! TRAP's, measured at 1.2 standard errors. Conversely, a defect of exactly
//! TRAP's size would have shown up on DENSE at
//! `z = 400000 × 9.4e−5 / sqrt(2×1956) = 0.60`. **DENSE had 0.6σ of power
//! against the defect in question.** SPARSE, at `M = 150k` on the same support,
//! had less. Absence of evidence, not evidence of absence.
//!
//! (`df` is measured, not assumed: `bins=1956` from
//! `CSMC_INVARIANCE_M=400000 cargo test -p sim --test csmc_exact_invariance --
//! one_sweep_leaves --nocapture`. It grows with `M` because the `E ≥ 25` cut
//! admits more of the 3538 paths, so quoting the `df` from a smaller run
//! understates the dilution.)
//!
//! So this file measures the thing directly instead.
//!
//! # What this probe does
//!
//! [`conditional_systematic_ancestor_sampling_is_not_a_valid_move`] compares two
//! laws over the *whole* ancestor vector, both computed exactly:
//!
//! - what the extended target requires — `P(a) ∝ ρ_W(a)·L(aʳ)`, where `ρ_W` is
//!   the cycle-randomised systematic resampling law (Chopin & Singh 2015 §5.2)
//!   and `L` is the ancestor-sampling likelihood factor;
//! - what `csmc_as` realises — draw the free block from `ρ_W(· | Aʳ = r)`
//!   (their Algorithm 4), then **overwrite** `Aʳ` with an independent ancestor-
//!   sampling draw `∝ Wʲ·L(j)`.
//!
//! No model, no simulation, no Monte Carlo, and — this is the point — **no
//! repair mechanism, no state degeneracy, and no projection down to a single
//! lineage**. Those are the three ways an end-to-end fixture can come back
//! falsely clean, and this probe has none of them. It is the analogue of the
//! resampling unit test that would have caught defect 1 in twenty lines.
//!
//! The control, [`multinomial_ancestor_sampling_needs_no_correction`], runs the
//! identical comparison with `ρ_W(a) = Π Wᵃⁿ` and must give total variation
//! **exactly** zero — that is what makes the systematic result a finding about
//! systematic resampling rather than a bug in this file's algebra.
//!
//! # The end-to-end fixture, and what it needs
//!
//! [`snare_geometry_is_a_live_unrepaired_ancestor_move`] is the invariance
//! fixture the incident lacked: two substeps, observations at **both**, so the
//! single ancestor-sampling opportunity (substep 1) runs on genuinely unequal
//! weights *and* has nothing after it. TRAP has the isolation but no live AS
//! move; DENSE has live AS moves but 2 of its 3 are followed by further
//! resampling. This geometry has both properties and neither existing fixture
//! does.
//!
//! It runs here against the **shipped** (multinomial) resampler, where it must
//! pass — that establishes the geometry is sound and non-vacuous. Pointing it at
//! conditional systematic requires a one-line change to production source, which
//! this file deliberately does not make; see
//! [`snare_under_conditional_systematic`] for the exact patch.

use std::collections::HashMap;
use std::sync::Arc;

use sim::compiled_model::CompiledModel;
use sim::inference::dense_cells;
use sim::inference::multi_stream_obs::{BoundObs, MultiStreamObsModel, StreamProjection, StreamSpec};
use sim::inference::particle_filter::Observation;
use sim::inference::pgas::{
    build_obs_at_substep, complete_data_loglik, csmc_as, EffectFiring, ObsAtSubstep,
    PGASTrajectory, SubstepRecord,
};
use sim::rng::StatefulRng;

// ═══════════════════════════════════════════════════════════════════════════
// Part 1 — the ancestry law, exactly.
// ═══════════════════════════════════════════════════════════════════════════

/// Local copy of `systematic_resample_core` (it is `pub(crate)`, and an
/// integration test is a separate crate). Kept byte-equivalent to
/// `rust/crates/sim/src/inference/resampling.rs:32` so the enumeration below is
/// of the selection loop camdl actually runs.
fn systematic_core(weights: &[f64], u0: f64) -> Vec<usize> {
    let n = weights.len();
    let u = u0 / n as f64;
    let mut indices = Vec::with_capacity(n);
    let mut cumsum = 0.0;
    let mut j = 0;
    for i in 0..n {
        let threshold = u + i as f64 / n as f64;
        while j < n - 1 && cumsum + weights[j] < threshold {
            cumsum += weights[j];
            j += 1;
        }
        indices.push(j);
    }
    indices
}

/// `conditional_systematic_resample`, recovered from `55178dd1` (superseded on
/// main by `conditional_multinomial_resample`). Chopin & Singh (2015)
/// Algorithm 4: condition `U` so the reference receives an offspring with the
/// right law for how many, run plain systematic selection at that `U`, then
/// cycle the output uniformly over the reference's own copies.
///
/// Present so this probe can exercise the scheme without reinstating it in
/// production source.
#[allow(dead_code)]
fn conditional_systematic_resample(w: &[f64], reference: usize, rng: &mut StatefulRng) -> Vec<usize> {
    let n = w.len();
    if n == 1 {
        return vec![0];
    }
    let nf = n as f64;
    let rot = |i: usize| (i + reference) % n;
    let wr: Vec<f64> = (0..n).map(|i| w[rot(i)]).collect();

    let nw1 = nf * wr[0];
    let u0 = if nw1 <= 1.0 {
        rng.uniform() * nw1
    } else {
        let floor = nw1.floor();
        let frac = nw1 - floor;
        if rng.uniform() < frac * (floor + 1.0) / nw1 {
            rng.uniform() * frac
        } else {
            frac + rng.uniform() * (1.0 - frac)
        }
    };
    let abar = systematic_core(&wr, u0);
    let copies: Vec<usize> = (0..n).filter(|&i| abar[i] == 0).collect();
    let c0 = if copies.is_empty() {
        0
    } else {
        copies[((rng.uniform() * copies.len() as f64) as usize).min(copies.len() - 1)]
    };
    let mut out = vec![0usize; n];
    for slot in 0..n {
        out[rot(slot)] = rot(abar[(c0 + slot) % n]);
    }
    out
}

/// The exact law `ρ_W` of the **unconditional** cycle-randomised systematic
/// ancestor vector — the scheme Chopin & Singh (2015) §5.2 make marginally
/// unbiased by randomly cycling the output.
///
/// `systematic_core` is piecewise constant in `u0`: it can only change where a
/// threshold `u0/n + i/n` crosses a cumulative weight `V_m`, i.e. at
/// `u0 = n·V_m − i`. Integrating exactly over those pieces (and averaging over
/// the `n` cycles) gives the law with **no Monte-Carlo error**, which is what
/// lets the comparison below be a statement about the algorithm rather than
/// about a sample size.
fn rho_systematic(w: &[f64]) -> HashMap<Vec<usize>, f64> {
    let n = w.len();
    let nf = n as f64;
    let mut cum = Vec::with_capacity(n);
    let mut acc = 0.0;
    for &x in w {
        acc += x;
        cum.push(acc);
    }
    let mut bps: Vec<f64> = vec![0.0, 1.0];
    for m in 0..n {
        for i in 0..n {
            let u = nf * cum[m] - i as f64;
            if u > 0.0 && u < 1.0 {
                bps.push(u);
            }
        }
    }
    bps.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let mut law: HashMap<Vec<usize>, f64> = HashMap::new();
    for pair in bps.windows(2) {
        let (lo, hi) = (pair[0], pair[1]);
        if hi - lo <= 1e-15 {
            continue;
        }
        let abar = systematic_core(w, 0.5 * (lo + hi));
        for c0 in 0..n {
            let cyc: Vec<usize> = (0..n).map(|s| abar[(c0 + s) % n]).collect();
            *law.entry(cyc).or_insert(0.0) += (hi - lo) / nf;
        }
    }
    law
}

/// The same object for multinomial resampling: `ρ_W(a) = Π_n W^{aⁿ}`. The
/// control — this is the scheme the accept/reject ratio was derived under, so
/// the correction must vanish identically here.
fn rho_multinomial(w: &[f64]) -> HashMap<Vec<usize>, f64> {
    let n = w.len();
    let mut law: HashMap<Vec<usize>, f64> = HashMap::new();
    let mut a = vec![0usize; n];
    loop {
        let p: f64 = a.iter().map(|&x| w[x]).product();
        *law.entry(a.clone()).or_insert(0.0) += p;
        let mut i = 0;
        loop {
            if i == n {
                return law;
            }
            a[i] += 1;
            if a[i] < n {
                break;
            }
            a[i] = 0;
            i += 1;
        }
    }
}

struct Verdict {
    tv: f64,
    off_support: f64,
    erased_camdl: f64,
    erased_correct: f64,
}

/// Compare the ancestry law `csmc_as` realises against the one the extended
/// target requires, for a given resampling scheme.
///
/// - **target**: `P(a) ∝ ρ_W(a)·L(aʳ)`. The ancestry block enters the extended
///   target of Chopin & Singh's eq. (19) through `ρ` and through the frozen
///   path's attachment; nothing else in the block's full conditional depends on
///   `a`, so this is it.
/// - **realised**: `P̃(a⁻ʳ, j) = ρ_W(a⁻ʳ, r)/Z · Wʲ·L(j)/ΣWᵐL(m)`. The free
///   block is drawn conditional on `Aʳ = r` and `Aʳ` is then overwritten by an
///   ancestor-sampling draw that never looks at `a⁻ʳ` —
///   `pgas.rs:2152` then `pgas.rs:2372`.
///
/// `L` stands for the ancestor-sampling likelihood factor
/// `f(x⋆ₛ | xʲₛ₋₁) · S(j)`, i.e. the Eq.-(17) transition density times the
/// Metropolis suffix ratio. Whatever it is, it is a function of `j` alone, and
/// that is all this comparison needs. Using the *converged* AS conditional
/// `∝ Wʲ·L(j)` rather than camdl's single independence-proposal MH step is
/// generous to the implementation: one MH step lies between `δ_r` and this, and
/// neither endpoint is supported on `ρ_W`.
fn compare(w: &[f64], l: &[f64], r: usize, rho: &HashMap<Vec<usize>, f64>) -> Verdict {
    let n = w.len();

    // Guard the enumeration itself: `ρ_W` must be marginally unbiased, which is
    // the precondition Chopin & Singh §5 place on the resampling distribution.
    // If this fails, nothing below means anything.
    for slot in 0..n {
        for m in 0..n {
            let got: f64 = rho.iter().filter(|(a, _)| a[slot] == m).map(|(_, p)| p).sum();
            assert!(
                (got - w[m]).abs() < 1e-9,
                "the enumerated ρ_W is not marginally unbiased at slot {slot}, ancestor {m}: \
                 {got:.6} vs W = {:.6}. The comparison below would be meaningless.",
                w[m]
            );
        }
    }

    let z_as: f64 = (0..n).map(|j| w[j] * l[j]).sum();
    let as_p: Vec<f64> = (0..n).map(|j| w[j] * l[j] / z_as).collect();

    // Algorithm 4's law = ρ_W conditioned on the reference keeping itself.
    let z_c: f64 = rho.iter().filter(|(a, _)| a[r] == r).map(|(_, p)| p).sum();

    let mut realised: HashMap<Vec<usize>, f64> = HashMap::new();
    for (a, p) in rho.iter().filter(|(a, _)| a[r] == r) {
        for (j, &pj) in as_p.iter().enumerate() {
            let mut b = a.clone();
            b[r] = j;
            *realised.entry(b).or_insert(0.0) += (p / z_c) * pj;
        }
    }

    let z_t: f64 = rho.iter().map(|(a, p)| p * l[a[r]]).sum();
    let target: HashMap<Vec<usize>, f64> =
        rho.iter().map(|(a, p)| (a.clone(), p * l[a[r]] / z_t)).collect();

    let mut keys: Vec<&Vec<usize>> = realised.keys().collect();
    for k in target.keys() {
        if !realised.contains_key(k) {
            keys.push(k);
        }
    }
    let tv = 0.5
        * keys
            .iter()
            .map(|k| (realised.get(*k).unwrap_or(&0.0) - target.get(*k).unwrap_or(&0.0)).abs())
            .sum::<f64>();
    let off_support: f64 = realised
        .iter()
        .filter(|(k, _)| *target.get(*k).unwrap_or(&0.0) <= 1e-15)
        .map(|(_, p)| p)
        .sum();
    let erased = |d: &HashMap<Vec<usize>, f64>| -> f64 {
        d.iter().filter(|(a, _)| !a.contains(&r)).map(|(_, p)| p).sum()
    };
    Verdict { tv, off_support, erased_camdl: erased(&realised), erased_correct: erased(&target) }
}

/// `W` and the ancestor-sampling factor `L`. The reference is the last slot,
/// as in `csmc_as` (`j_ref = n − 1`).
///
/// Three regimes, because Algorithm 4 branches on `N·W_ref` and because the
/// severity of the defect depends on which side of `1/N` the reference sits:
/// below it the reference receives exactly one offspring and it goes in the
/// reference slot, so a splice erases its history outright; above it the
/// reference has free-slot copies and the error weakens to a ratio distortion.
const CASES: &[(&str, &[f64], &[f64])] = &[
    ("ref light (W_ref = 0.10 < 1/N)", &[0.30, 0.25, 0.20, 0.15, 0.10], &[1.0, 1.0, 1.0, 1.0, 1.0]),
    ("ref light, AS favours slot 0", &[0.30, 0.25, 0.20, 0.15, 0.10], &[4.0, 1.0, 1.0, 1.0, 1.0]),
    ("ref heavy (W_ref = 0.40 > 1/N)", &[0.20, 0.15, 0.15, 0.10, 0.40], &[1.0, 1.0, 1.0, 1.0, 1.0]),
    ("near-uniform weights", &[0.21, 0.20, 0.20, 0.20, 0.19], &[1.0, 1.0, 1.0, 1.0, 1.0]),
    ("N = 3", &[0.5, 0.3, 0.2], &[1.0, 1.0, 1.0]),
    ("N = 2 (the review's own smallest case)", &[0.7, 0.3], &[1.0, 1.0]),
];

/// **The control.** Under multinomial resampling the ancestry factorises across
/// slots, so `ρ_W(a⁻ʳ, j) = Wʲ · Π_{n≠r} W^{aⁿ}`: the `Wʲ` is exactly what the
/// Eq.-(17) ancestor weight already carries, and the rest is free of `j` and
/// cancels out of the Metropolis ratio. The correction the review's §8 asks for
/// is therefore identically 1 here, and total variation must be **exactly** zero
/// — not small, zero.
///
/// This test is what makes the systematic result below a finding rather than an
/// artefact of this file's algebra: the two tests share every line of the
/// comparison and differ only in `ρ_W`.
#[test]
fn multinomial_ancestor_sampling_needs_no_correction() {
    for (label, w, l) in CASES {
        let r = w.len() - 1;
        let v = compare(w, l, r, &rho_multinomial(w));
        eprintln!(
            "multinomial  {label:<40} TV={:.3e}  off-support={:.3e}",
            v.tv, v.off_support
        );
        assert!(
            v.tv < 1e-12,
            "{label}: multinomial should need no correction, got TV = {:.3e}. \
             Either the comparison in `compare` is wrong or the extended target \
             has been misidentified — fix that before reading the systematic case.",
            v.tv
        );
    }
}

/// **The finding.** Under conditional systematic resampling the ancestry does
/// not factorise, so `ρ_W(a⁻ʳ, j) ≠ Wʲ × (const in j)` and the residual
/// `ρ_W(a⁻ʳ, j)/Wʲ ÷ ρ_W(a⁻ʳ, r)/Wʳ` survives in the Metropolis ratio. The
/// review's §8 is correct.
///
/// It is not a small correction. Systematic resampling pins each particle's
/// offspring count to `⌊N·Wⁿ⌋` or `⌊N·Wⁿ⌋+1` and returns them in cyclic order,
/// so given `a⁻ʳ` the admissible values of `Aʳ` are usually one or two indices.
/// Ancestor sampling proposes over all `N`, so most of what it accepts lands on
/// an ancestry the extended target gives **zero** density — a support violation,
/// which no reweighting of the accept/reject ratio can repair. The two honest
/// fixes are to redraw `a⁻ʳ ~ ρ_W(· | Aʳ = j)` after the ancestor move (which is
/// free if the move is made *before* the free particles are propagated, since
/// marginal unbiasedness makes `∝ Wʲ·L(j)` the correct marginal for `Aʳ`), or to
/// use multinomial resampling wherever ancestor sampling runs — the fix that
/// landed in `2ba0329d`.
///
/// Expected output (exact, no sample size involved):
///
/// ```text
/// cond-systematic  ref light (W_ref = 0.10 < 1/N)   TV=0.800  off-support=0.750
/// cond-systematic  ref light, AS favours slot 0      TV=0.737  off-support=0.632
/// cond-systematic  ref heavy (W_ref = 0.40 > 1/N)    TV=0.600  off-support=0.600
/// cond-systematic  near-uniform weights              TV=0.800  off-support=0.600
/// cond-systematic  N = 3                             TV=0.617  off-support=0.333
/// cond-systematic  N = 2                             TV=0.300  off-support=0.000
/// ```
///
/// Note the `near-uniform` row in particular: the reference's history is erased
/// from the ensemble entirely on 81% of steps where the target says 5%. That is
/// defect 1's pathology — history flows one way — reappearing on substeps that
/// *do* resample, which is precisely what the incident report could not rule out.
#[test]
fn conditional_systematic_ancestor_sampling_is_not_a_valid_move() {
    let mut worst: f64 = 0.0;
    for (label, w, l) in CASES {
        let r = w.len() - 1;
        let v = compare(w, l, r, &rho_systematic(w));
        eprintln!(
            "cond-systematic  {label:<40} TV={:.3}  off-support={:.3}  \
             P(reference history erased) realised={:.3} target={:.3}",
            v.tv, v.off_support, v.erased_camdl, v.erased_correct
        );
        worst = worst.max(v.tv);
    }
    assert!(
        worst > 0.05,
        "conditional systematic + ancestor sampling came out consistent with the extended \
         target (worst TV = {worst:.3e}). That contradicts the algebra — the ancestry does not \
         factorise, so the resampling law cannot cancel out of the accept/reject ratio. \
         Check `rho_systematic` before concluding the review's §8 is wrong."
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// Part 2 — SNARE: one live ancestor move, nothing after it.
// ═══════════════════════════════════════════════════════════════════════════

const DT: f64 = 1.0;

/// Two substeps, an observation at **each**.
///
/// - substep 0: incoming weights are uniform (nothing scored yet), so
///   `did_resample` is false and ancestor sampling is gated off. Every particle
///   starts from the same deterministic state, so nothing is lost.
/// - substep 0's observation then scores the diverged particles, so substep 1's
///   incoming weights are **genuinely unequal**: `did_resample` is true and the
///   ancestor move is live.
/// - substep 1's observation drives no resampling (weights are consumed by the
///   *following* substep, and there is none). So the single live ancestor move
///   has **nothing after it**.
///
/// TRAP has the isolation but gates AS off; DENSE has live AS moves but dilutes
/// them across 433 scored bins and repairs 2 of 3 with later resampling. This
/// geometry is the missing cell.
const SNARE: [(usize, f64); 2] = [(0, 3.0), (1, 2.0)];
const SNARE_SUBSTEPS: usize = 2;

const S_IDX: usize = 0;
const I_IDX: usize = 1;
const R_IDX: usize = 2;
const TR_INFECTION: usize = 0;
const TR_RECOVERY: usize = 1;

fn prevalence_obs_block() -> ir::observation::ObservationModel {
    use ir::expr::*;
    use ir::observation::*;
    let rate = Expr::Projected(ProjectedExpr { projected: () });
    ObservationModel {
        name: "prevalence".into(),
        source: "prevalence".into(),
        columns: vec![
            ObsColumn { name: "time".into(), role: ColumnRole::Time },
            ObsColumn {
                name: "prevalence".into(),
                role: ColumnRole::Value(ir::parameter::ParamKind::Count),
            },
        ],
        scored: "prevalence".into(),
        emit_schedule: Some(ObservationSchedule::AtTimes(vec![])),
        stratum: vec![],
        projection: Projection::CurrentPop("I".into()),
        projection_state_grad: Default::default(),
        likelihood: Likelihood::Poisson(PoissonLikelihood { rate: ir::Diffable::new(rate) }),
    }
}

fn model(n_substeps: usize) -> Arc<CompiledModel> {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../../ocaml/golden/sir_basic.ir.json");
    let json = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("read sir_basic golden {path:?}: {e}"));
    let mut m = ir::from_str(&json).expect("parse sir_basic");
    m.observations = vec![prevalence_obs_block()];
    m.simulation.t_start = 0.0;
    m.simulation.t_end = n_substeps as f64 * DT;
    for p in &mut m.parameters {
        let v = match p.name.as_str() {
            "beta" => 1.2,
            "gamma" => 0.5,
            "N0" => 6.0,
            "I0" => 2.0,
            other => panic!("unexpected parameter {other} in sir_basic"),
        };
        p.value = ir::parameter::ParamValue::Fixed { value: v };
    }
    Arc::new(CompiledModel::new(m).expect("compile sir_basic"))
}

struct Fixture {
    compiled: Arc<CompiledModel>,
    params: Vec<f64>,
    obs: Vec<Observation>,
    obs_model: MultiStreamObsModel,
    obs_at_substep: ObsAtSubstep,
    initial_counts: Vec<i64>,
    n_substeps: usize,
}

fn fixture_with(schedule: &[(usize, f64)], n_substeps: usize) -> Fixture {
    let compiled = model(n_substeps);
    let params = compiled.default_params.clone();
    let (init, _) = compiled.initial_state_mean(&params).expect("initial state");
    let initial_counts = init.counts.clone();

    let obs: Vec<Observation> = schedule
        .iter()
        .map(|&(s, v)| Observation { time: ((s + 1) as f64) * DT, value: v })
        .collect();

    let obs_model = MultiStreamObsModel::new(
        BoundObs::bind(vec![StreamSpec::dense(
            StreamProjection::IntCompSum(vec![I_IDX]),
            compiled.model.observations[0].clone(),
            dense_cells(obs.iter().map(|o| o.value).collect()),
            obs.iter().map(|o| o.time).collect(),
        )])
        .unwrap()
        .0,
        compiled.clone(),
    )
    .unwrap();

    let obs_at_substep: ObsAtSubstep =
        build_obs_at_substep(&obs, compiled.model.simulation.t_start, DT).expect("obs_at_substep");
    assert_eq!(obs_at_substep.len(), schedule.len());

    Fixture { compiled, params, obs, obs_model, obs_at_substep, initial_counts, n_substeps }
}

fn key(traj: &PGASTrajectory) -> Vec<u64> {
    traj.substeps.iter().flat_map(|r| r.flows.iter().copied()).collect()
}

fn enumerate_paths(initial: &[i64], n_substeps: usize) -> Vec<PGASTrajectory> {
    let mut out: Vec<PGASTrajectory> = Vec::new();
    let mut stack: Vec<(Vec<i64>, Vec<SubstepRecord>)> = vec![(initial.to_vec(), Vec::new())];
    while let Some((state, recs)) = stack.pop() {
        let s = recs.len();
        if s == n_substeps {
            out.push(PGASTrajectory { initial_counts: initial.to_vec(), substeps: recs });
            continue;
        }
        for k_inf in 0..=(state[S_IDX] as u64) {
            for k_rec in 0..=(state[I_IDX] as u64) {
                let mut flows = vec![0u64; 2];
                flows[TR_INFECTION] = k_inf;
                flows[TR_RECOVERY] = k_rec;
                let mut after = state.clone();
                after[S_IDX] -= k_inf as i64;
                after[I_IDX] += k_inf as i64 - k_rec as i64;
                after[R_IDX] += k_rec as i64;
                let mut next = recs.clone();
                next.push(SubstepRecord {
                    counts_before: state.clone(),
                    counts_after: after.clone(),
                    flows,
                    gammas: Vec::new(),
                    t0: s as f64 * DT,
                    dt_substep: DT,
                });
                stack.push((after, next));
            }
        }
    }
    out
}

fn exact_target(f: &Fixture) -> (Vec<PGASTrajectory>, Vec<f64>, HashMap<Vec<u64>, usize>) {
    let mut paths = Vec::new();
    let mut logp = Vec::new();
    for traj in enumerate_paths(&f.initial_counts, f.n_substeps) {
        let ll = complete_data_loglik(
            &f.compiled, &traj, &f.params, &f.obs, DT, &f.obs_model, &f.obs_at_substep,
        )
        .expect("complete_data_loglik")
        .total;
        if ll.is_finite() {
            paths.push(traj);
            logp.push(ll);
        }
    }
    let max = logp.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let mut w: Vec<f64> = logp.iter().map(|l| (l - max).exp()).collect();
    let z: f64 = w.iter().sum();
    for x in w.iter_mut() {
        *x /= z;
    }
    let index: HashMap<Vec<u64>, usize> =
        paths.iter().enumerate().map(|(i, t)| (key(t), i)).collect();
    (paths, w, index)
}

fn draw_categorical(p: &[f64], u: f64) -> usize {
    let mut c = 0.0;
    for (i, &q) in p.iter().enumerate() {
        c += q;
        if u < c {
            return i;
        }
    }
    p.len() - 1
}

/// The SNARE invariance run, against whatever resampler `csmc_as` is compiled
/// with.
///
/// Reports `D = (χ²/df − 1)·df/M`, the divergence between the kernel's
/// stationary law and `π` **normalised for the number of scored bins**. That is
/// the number to compare across fixtures; raw `z` is not, and comparing raw `z`
/// across DENSE (df = 1956) and TRAP (df ≈ 126) is the mistake that left this
/// question open. A `D` at or above TRAP's confirmed `9.4e−5` is a defect of the
/// same size as the one the incident proved real.
///
/// Measured against the shipped multinomial resampler at `M = 400k`:
/// `bins=126 chi2/df=1.040 z=0.35 D=1.25e-5 ± 3.97e-5`, with 0.993 resampling
/// substeps and 0.712 ancestor-sampling proposals per sweep at 85% acceptance.
/// Clean, and non-vacuously so.
#[test]
fn snare_geometry_is_a_live_unrepaired_ancestor_move() {
    let f = fixture_with(&SNARE, SNARE_SUBSTEPS);
    let (paths, pi, index) = exact_target(&f);

    let ess = 1.0 / pi.iter().map(|p| p * p).sum::<f64>();
    eprintln!("SNARE support: {} paths, effective support size {ess:.1}", paths.len());
    assert!(ess > 8.0, "target too concentrated to test anything (ESS {ess:.1})");

    let m: usize = std::env::var("CSMC_INVARIANCE_M")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(400_000);
    let n_particles: usize = std::env::var("CSMC_INVARIANCE_NP")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(5);

    let mut rng = StatefulRng::new(20260823);
    let mut tally = vec![0u64; paths.len()];
    let (mut n_proposed, mut n_accepted, mut n_resampled) = (0usize, 0usize, 0usize);

    for i in 0..m {
        let x0 = draw_categorical(&pi, rng.uniform());
        let (x1, diag) = csmc_as(
            &f.compiled,
            &f.params,
            &f.obs,
            &paths[x0],
            n_particles,
            DT,
            &f.obs_model,
            0x5eed_a5a5_0000_0000u64.wrapping_add(i as u64),
            &f.obs_at_substep,
            EffectFiring::default(),
            sim::rng::BinomialAlgorithm::Btpe,
        )
        .expect("csmc_as");
        n_proposed += diag.n_as_proposed;
        n_accepted += diag.n_as_accepted;
        n_resampled += diag.n_resampled;

        let k = key(&x1);
        let idx = *index
            .get(&k)
            .unwrap_or_else(|| panic!("csmc_as returned a path outside the support: {k:?}"));
        tally[idx] += 1;
    }

    // Non-vacuity, and the two properties this geometry exists to guarantee.
    eprintln!(
        "SNARE: resampling substeps {:.3}/sweep, AS proposals {:.3}/sweep, accepted {} ({:.1}%)",
        n_resampled as f64 / m as f64,
        n_proposed as f64 / m as f64,
        n_accepted,
        100.0 * n_accepted as f64 / n_proposed.max(1) as f64
    );
    assert!(
        (n_resampled as f64 / m as f64) > 0.95,
        "SNARE should draw an ancestry on exactly one substep per sweep, got {:.3} — \
         substep 0's observation is not producing unequal weights, so the ancestor move \
         at substep 1 is gated off and this fixture is TRAP in disguise",
        n_resampled as f64 / m as f64
    );
    assert!(
        n_accepted > m / 10,
        "SNARE accepted only {n_accepted} splices over {m} sweeps — the move under test \
         is barely exercised"
    );

    let mf = m as f64;
    let mut chi2 = 0.0;
    let mut df = 0usize;
    let mut worst = (0.0f64, 0usize);
    for i in 0..paths.len() {
        let e = mf * pi[i];
        if e < 25.0 {
            continue;
        }
        df += 1;
        let z = (tally[i] as f64 - e) / (e * (1.0 - pi[i])).sqrt();
        chi2 += z * z;
        if z.abs() > worst.0 {
            worst = (z.abs(), i);
        }
    }
    assert!(df > 5, "too few well-populated bins ({df}) — raise CSMC_INVARIANCE_M");
    let dff = df as f64;
    let z_agg =
        ((chi2 / dff).powf(1.0 / 3.0) - (1.0 - 2.0 / (9.0 * dff))) / (2.0 / (9.0 * dff)).sqrt();
    let divergence = (chi2 / dff - 1.0) * dff / mf;

    // `D` is an estimate, and its own noise floor is `sqrt(2·df)/M` — print it,
    // so a `D` inside the floor is never read as a defect. SNARE enumerates the
    // same 131 paths as TRAP and scores nearly the same number of bins, so its
    // `D` is directly comparable to TRAP's confirmed 9.8e−5 with no correction.
    let d_noise = (2.0 * dff).sqrt() / mf;
    eprintln!(
        "SNARE M={m} np={n_particles} bins={df} chi2/df={:.3} z={z_agg:.2} \
         D={divergence:.2e} ± {d_noise:.2e}  (TRAP’s confirmed defect: D = 9.4e-5)",
        chi2 / dff
    );
    assert!(
        d_noise < 9.4e-5,
        "at M={m} the noise floor on D is {d_noise:.2e}, above TRAP’s confirmed 9.4e-5 — \
         this run cannot resolve a defect of the size the incident proved real. \
         Raise CSMC_INVARIANCE_M to at least 200000."
    );
    assert!(
        z_agg < 6.0,
        "SNARE: csmc_as is not invariant on a single unrepaired ancestor move — \
         χ²/df = {:.3} on {df} bins (z = {z_agg:.2}), D = {divergence:.2e}, \
         worst bin |z| = {:.2}",
        chi2 / dff,
        worst.0
    );
}

/// The end-to-end arm for the review's §8, which **cannot be run from this file
/// alone**: `csmc_as` calls `conditional_multinomial_resample` directly, and
/// wiring the systematic scheme in means editing production source.
///
/// To run it, apply this one-line change and re-run
/// [`snare_geometry_is_a_live_unrepaired_ancestor_move`]:
///
/// ```text
/// rust/crates/sim/src/inference/pgas.rs:2153
///   -   conditional_multinomial_resample(&log_weights, j_ref, &mut resample_rng);
///   +   conditional_systematic_resample(&log_weights, j_ref, &mut resample_rng);
/// ```
///
/// restoring `conditional_systematic_resample` from `55178dd1` (a copy is in
/// this file). Then compare `D` against `9.8e−5`.
///
/// **Read Part 1 first.** The ancestry-law probe above already settles the
/// question exactly, with no sample size, no repair path, and no projection down
/// to a single lineage. This arm only measures how much of that ancestry-level
/// error survives the projection to a trajectory tally — useful for sizing the
/// damage to real fits, not for deciding whether the defect exists.
#[test]
#[ignore = "requires a one-line patch to pgas.rs; see the doc comment"]
fn snare_under_conditional_systematic() {
    panic!("see the doc comment: this arm needs conditional_systematic_resample wired into csmc_as");
}
