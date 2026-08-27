//! A/B: `rand_distr` BTPE vs the in-house BTRS, at the `(n, p)` regimes a real
//! PGAS fit actually visits (gh#747).
//!
//!   cargo bench -p sim --bench binomial_ab
//!
//! Custom main, no criterion harness — mirrors `flat_eval.rs` / `eval_ab.rs` and
//! the median-of-9 convention the flat-evaluator note established. Both arms run
//! in ONE process, INTERLEAVED per rep with the order alternating by rep parity,
//! so background load, thermal drift and any within-rep warming hit them
//! equally; that is the whole reason `set_binomial_algorithm` is a thread-local
//! override rather than an env var.
//!
//! Three things this bench deliberately does NOT do:
//!
//!  - It does not report a whole-fit speedup as a single number. It prints the
//!    implied factor across a RANGE of assumed binomial shares, because that
//!    share is a profile measurement with a percentage point or two of
//!    discretion in it (whether `step_one`'s unattributed self-time contains
//!    inlined sampler code, whether `UniformFloat::new` counts), and quoting one
//!    figure would hide that. Nothing downstream should quote the implied factor
//!    as an OUTCOME either: it is Amdahl applied to a measured sampler ratio,
//!    not an end-to-end observation. BTRS is not reachable from production, so
//!    no end-to-end figure exists yet.
//!  - It does not claim its regimes are the posterior's. They were read off a
//!    chain at an infeasible start with 0% NUTS acceptance, so the real `(n, p)`
//!    mix at stationarity may differ — `np` scales with prevalence, and `kappa`
//!    is log-uniform over seven decades. Two split cells sit exactly ON the
//!    `n·min(p,1−p) = 10` routing boundary (`40 × 0.25`, `20 × 0.5`, both exact
//!    in binary), so a hair less `p` at stationarity silently moves them to
//!    BINV-in-both-arms at 1.00×. Re-harvest from a healthy chain before
//!    treating any of this as final.
//!  - It does not weight its cells equally. See `weight` below.
//!
//! The split-draw regimes matter as much as the total-exit ones: the province
//! model draws 12 total-exit binomials and 12 competing-risk splits per
//! particle-substep, and the splits live at `n ≈ 20..200` where BTRS's squeeze
//! fires least often (`v_r` 0.30 at `n=20, p=0.5` against 0.76 at `n=400,
//! p=0.476`).
//!
//! Note which way that cuts, because an earlier version of this comment had it
//! backwards. BTRS is slower at the splits in ABSOLUTE terms, but BTPE degrades
//! more there — its immediate-accept triangle region collapses from 63% to 21%
//! of the `u` range — so the RATIO is larger at the splits, not smaller. A bench
//! at `np ≈ 190` alone would have measured ~1.54× and UNDERSTATED the win, not
//! overstated it. The trap the flat-evaluator note recorded (synthetic 2.5×,
//! real 1.27×) is still the right trap to fear; it just is not this cell set.

use std::hint::black_box;
use std::time::Instant;

use sim::rng::{set_binomial_algorithm, BinomialAlgorithm, StatefulRng};

const DRAWS: usize = 400_000;
const REPS: usize = 9;

/// One benchmarked regime.
///
/// `weight` is how many of the 24 draws per particle-substep this cell stands
/// for, so the blend is a weighted mean rather than a mean over cells. The
/// earlier equal-weight version gave the `E` regime 20% where the model gives it
/// 50%, and — worse — implicitly asserted that all 24 draws reach the arm under
/// test, when several do not (see `BINV_ROUTED`).
struct Cell {
    label: &'static str,
    n: u64,
    p: f64,
    weight: f64,
}

const fn cell(label: &'static str, n: u64, p: f64, weight: f64) -> Cell {
    Cell { label, n, p, weight }
}

/// Total-exit draws: `Binom(n_src, 1 − exp(−Σr·dt))`, one per source group.
///
/// 12 draws over 4 distinct regimes: `S` once per province, `E` over 2 provinces
/// × 3 stages, `I` and `C` once per province.
const TOTAL_EXIT: &[Cell] = &[
    cell("S  ituri     n=6.3e6 np=192", 6_300_000, 3.05e-5, 1.0),
    cell("S  nord_kivu n=8.8e6 np=192", 8_750_000, 2.2e-5, 1.0),
    cell("E  stage     n=400   np=190", 400, 0.476, 6.0),
    cell("I            n=520   np=153", 520, 0.295, 2.0),
    cell("C            n=780   np=87 ", 780, 0.111, 2.0),
];

/// Competing-risk splits: `Binom(n_exit, rate_k / rate_remaining)`.
///
/// 9 of the 12 splits; the other 3 are in `BINV_ROUTED`.
const SPLIT: &[Cell] = &[
    cell("split n=190 p=0.70", 190, 0.70, 2.0),
    cell("split n=150 p=0.50", 150, 0.50, 2.0),
    cell("split n=100 p=0.30", 100, 0.30, 2.0),
    cell("split n=40  p=0.25", 40, 0.25, 2.0),
    cell("split n=20  p=0.50", 20, 0.50, 1.0),
];

/// The splits BTRS never sees, and the reason the blend is not a mean over the
/// two sections above.
///
/// The province model's competing exits include hazards three orders of
/// magnitude below the group's dominant rate — `export_e` on each `E` stage is
/// `travel_rate / (3σ + travel_rate) ≈ 6e-4`, and `travel_rate`'s prior median
/// is 2.96e-4/d. On `n_exit ≈ 190` that is `np ≈ 0.1`, two orders of magnitude
/// below `BINV_THRESHOLD = 10`, so these draws route to BINV in BOTH arms and
/// BTRS does exactly nothing for them. Including them at 1.00× is what stops the
/// blend from asserting 24/24 arm-routed draws — which the note's own figure of
/// BINV at ~17% of binomial time already contradicted.
///
/// The count is the 3 `export_e` splits on Ituri's `E` stages, which is a
/// CONSERVATIVE floor: the two `S` importation splits are
/// `kappa`-dependent and `kappa` is log-uniform over `[1e-9, 1e-2]`, so they may
/// sit on either side of the threshold depending on θ. Understating the
/// BINV-routed count understates the correction, which is the safe direction.
const BINV_ROUTED: &[Cell] = &[cell("export_e  n=190  np≈0.1", 190, 6e-4, 3.0)];

/// One timed measurement: `DRAWS` draws at `(n, p)` under `algo`, in ns/draw.
fn time_one(algo: BinomialAlgorithm, n: u64, p: f64, seed: u64) -> f64 {
    set_binomial_algorithm(algo);
    let mut rng = StatefulRng::new(seed);
    // Warm the branch predictor and the ChaCha8 block buffer. Outside the timed
    // region, as is `StatefulRng::new` and the selection itself.
    for _ in 0..10_000 {
        black_box(rng.binomial(n, p));
    }
    let t = Instant::now();
    let mut acc = 0u64;
    for _ in 0..DRAWS {
        acc = acc.wrapping_add(rng.binomial(n, p));
    }
    let ns = t.elapsed().as_secs_f64() * 1e9 / DRAWS as f64;
    // After `elapsed()`, so the accumulator is kept alive without being timed.
    black_box(acc);
    ns
}

/// Median and full range. The range is printed: without it a reader cannot tell
/// whether a gap between two cells' factors (1.54 vs 1.61, say) clears
/// rep-to-rep noise, and this bench's per-section factors are quoted elsewhere
/// as if they do.
fn stats(mut v: Vec<f64>) -> (f64, f64, f64) {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    (v[v.len() / 2], v[0], v[v.len() - 1])
}

/// Median-of-REPS for both arms, interleaved rep by rep.
///
/// The arm order ALTERNATES by rep parity. Interleaving alone cancels drift
/// between reps but not a systematic within-rep order effect — whichever arm
/// runs second always runs on a core just warmed by ~26 ms of the other.
fn ab(n: u64, p: f64) -> ((f64, f64, f64), (f64, f64, f64)) {
    let (mut btpe, mut btrs) = (Vec::new(), Vec::new());
    for rep in 0..REPS {
        let seed = 1000 + rep as u64;
        if rep % 2 == 0 {
            btpe.push(time_one(BinomialAlgorithm::Btpe, n, p, seed));
            btrs.push(time_one(BinomialAlgorithm::Btrs, n, p, seed));
        } else {
            btrs.push(time_one(BinomialAlgorithm::Btrs, n, p, seed));
            btpe.push(time_one(BinomialAlgorithm::Btpe, n, p, seed));
        }
    }
    (stats(btpe), stats(btrs))
}

/// A section's totals. `w_*` are weighted sums over draws (divide by `weight`
/// for a per-draw cost); `u_*` are plain means over CELLS, kept only so the
/// header line can show what the old equal-weight accounting would have said.
struct Totals {
    w_a: f64,
    w_b: f64,
    weight: f64,
    u_a: f64,
    u_b: f64,
}

/// Weighted per-draw cost for a section.
fn section(title: &str, cells: &[Cell]) -> Totals {
    println!("\n{title}");
    println!(
        "  {:<28} {:>3} {:>9} {:>13} {:>9} {:>13} {:>8}",
        "regime", "n/24", "BTPE ns", "[min,max]", "BTRS ns", "[min,max]", "speedup"
    );
    println!("  {:-<28} {:->3} {:->9} {:->13} {:->9} {:->13} {:->8}", "", "", "", "", "", "", "");
    let (mut wa, mut wb, mut wsum) = (0.0, 0.0, 0.0);
    let (mut ua, mut ub) = (0.0, 0.0);
    for c in cells {
        let ((a, a_lo, a_hi), (b, b_lo, b_hi)) = ab(c.n, c.p);
        wa += a * c.weight;
        wb += b * c.weight;
        wsum += c.weight;
        ua += a;
        ub += b;
        println!(
            "  {:<28} {:>3.0} {a:>9.2} {:>13} {b:>9.2} {:>13} {:>7.2}x",
            c.label,
            c.weight,
            format!("[{a_lo:.1},{a_hi:.1}]"),
            format!("[{b_lo:.1},{b_hi:.1}]"),
            a / b
        );
    }
    let (ma, mb) = (wa / wsum, wb / wsum);
    println!(
        "  {:<28} {wsum:>3.0} {ma:>9.2} {:>13} {mb:>9.2} {:>13} {:>7.2}x",
        "weighted mean", "", "", ma / mb
    );
    let n = cells.len() as f64;
    Totals { w_a: wa, w_b: wb, weight: wsum, u_a: ua / n, u_b: ub / n }
}

fn main() {
    println!(
        "binomial A/B — {DRAWS} draws x {REPS} reps, median, interleaved arms\n\
         (arm order alternates by rep parity; [min,max] is the full rep range)\n\
         Cells are weighted by how many of the 24 draws per particle-substep\n\
         they stand for, INCLUDING the splits that route to BINV in both arms."
    );

    let te = section("TOTAL-EXIT draws (12 per particle-substep)", TOTAL_EXIT);
    let sp = section("SPLIT draws, arm-routed (9 per particle-substep)", SPLIT);
    let bv = section("SPLIT draws, BINV in BOTH arms (3 per particle-substep)", BINV_ROUTED);

    let total_w = te.weight + sp.weight + bv.weight;
    let blend_a = (te.w_a + sp.w_a + bv.w_a) / total_w;
    let blend_b = (te.w_b + sp.w_b + bv.w_b) / total_w;
    let sampler_speedup = blend_a / blend_b;

    // Exactly the superseded accounting — mean over cells within each section,
    // equal weight between the two sections, BINV-routed splits absent — so the
    // size of the correction is visible rather than silently absorbed.
    let naive = ((te.u_a + sp.u_a) / 2.0) / ((te.u_b + sp.u_b) / 2.0);

    println!(
        "\nBLENDED over all {total_w:.0} draws\n  \
         BTPE {blend_a:.2} ns/draw -> BTRS {blend_b:.2} ns/draw = {sampler_speedup:.2}x sampler\n  \
         (superseded accounting — equal weight per cell, BINV-routed splits\n   \
         excluded — would report {naive:.2}x; the gap is what correct draw\n   \
         weighting and the BINV-routed draws are worth)"
    );

    // Whole-fit factor: if the sampler is `share` of the fit and gets `s` times
    // faster, the fit gets 1 / (1 - share*(1 - 1/s)) times faster.
    println!(
        "\nIMPLIED WHOLE-FIT FACTOR (sampler {sampler_speedup:.2}x), by assumed binomial share.\n\
         IMPLIED, not measured — Amdahl on the ratio above, at a share this bench\n\
         does not measure. Do not quote one of these as an outcome:"
    );
    for share in [0.30, 0.35, 0.389, 0.45, 0.50] {
        let fit = 1.0 / (1.0 - share * (1.0 - 1.0 / sampler_speedup));
        println!("  share {:.1}%  ->  {fit:.3}x", share * 100.0);
    }
    println!(
        "\nThe profiled share was 38.9% for `binomial` alone (SE ~0.4pp, i.e. ±0.8pp at\n\
         95%, and a LOWER bound: rand_distr is only partially inlined). BTRS keeps two\n\
         uniforms per attempt, so the 8.6% ChaCha8 slice is NOT removed by this change.\n\
         That share also includes BINV, which BTRS does not touch — the two scope\n\
         errors have opposite signs and largely cancel, which is why the implied\n\
         figure survives; a scope-consistent reconstruction gives 1.13-1.21x."
    );
}
