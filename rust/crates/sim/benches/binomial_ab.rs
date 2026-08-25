//! A/B: `rand_distr` BTPE vs the in-house BTRS, at the `(n, p)` regimes a real
//! PGAS fit actually visits (gh#747).
//!
//!   cargo bench -p sim --bench binomial_ab
//!
//! Custom main, no criterion harness — mirrors `flat_eval.rs` / `eval_ab.rs` and
//! the median-of-9 convention the flat-evaluator note established. Both arms run
//! in ONE process, INTERLEAVED per rep, so background load and thermal drift hit
//! them equally; that is the whole reason `set_binomial_algorithm` is a
//! thread-local override rather than an env var.
//!
//! Two things this bench deliberately does NOT do:
//!
//!  - It does not report a whole-fit speedup as a single number. It prints the
//!    implied factor across a RANGE of assumed binomial shares, because that
//!    share is a profile measurement with a percentage point or two of
//!    discretion in it (whether `step_one`'s unattributed self-time contains
//!    inlined sampler code, whether `UniformFloat::new` counts), and quoting one
//!    figure would hide that.
//!  - It does not claim its regimes are the posterior's. They were read off a
//!    chain at an infeasible start with 0% NUTS acceptance, so the real `(n, p)`
//!    mix at stationarity may differ — `np` scales with prevalence. Re-harvest
//!    from a healthy chain before treating any of this as final.
//!
//! The split-draw regimes matter as much as the total-exit ones: the province
//! model draws 12 total-exit binomials and 12 competing-risk splits per
//! particle-substep, and the splits live at `n ≈ 20..200` where BTRS's squeeze
//! fires least often. A bench at `np ≈ 190` alone would overstate the win — the
//! exact trap the flat-evaluator note recorded (synthetic 2.5×, real 1.27×).

use std::hint::black_box;
use std::time::Instant;

use sim::rng::{set_binomial_algorithm, BinomialAlgorithm, StatefulRng};

const DRAWS: usize = 400_000;
const REPS: usize = 9;

/// `(label, n, p)`. `p` is as the caller passes it; `StatefulRng::binomial` does
/// its own flipping.
const TOTAL_EXIT: &[(&str, u64, f64)] = &[
    ("S  ituri     n=6.3e6 np=192", 6_300_000, 3.05e-5),
    ("S  nord_kivu n=8.8e6 np=192", 8_750_000, 2.2e-5),
    ("E  stage     n=400   np=190", 400, 0.476),
    ("I            n=520   np=153", 520, 0.295),
    ("C            n=780   np=87 ", 780, 0.111),
];

/// The competing-risk splits: `Binom(n_exit, rate_k / rate_remaining)`.
const SPLIT: &[(&str, u64, f64)] = &[
    ("split n=190 p=0.70", 190, 0.70),
    ("split n=150 p=0.50", 150, 0.50),
    ("split n=100 p=0.30", 100, 0.30),
    ("split n=40  p=0.25", 40, 0.25),
    ("split n=20  p=0.50", 20, 0.50),
];

fn time_one(algo: BinomialAlgorithm, n: u64, p: f64, seed: u64) -> f64 {
    set_binomial_algorithm(algo);
    let mut rng = StatefulRng::new(seed);
    // Warm the branch predictor and the ChaCha8 block buffer.
    for _ in 0..10_000 {
        black_box(rng.binomial(n, p));
    }
    let t = Instant::now();
    let mut acc = 0u64;
    for _ in 0..DRAWS {
        acc = acc.wrapping_add(rng.binomial(n, p));
    }
    let ns = t.elapsed().as_secs_f64() * 1e9 / DRAWS as f64;
    black_box(acc);
    ns
}

fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[v.len() / 2]
}

/// Median-of-REPS for both arms, interleaved rep by rep.
fn ab(n: u64, p: f64) -> (f64, f64) {
    let (mut btpe, mut btrs) = (Vec::new(), Vec::new());
    for rep in 0..REPS {
        let seed = 1000 + rep as u64;
        btpe.push(time_one(BinomialAlgorithm::Btpe, n, p, seed));
        btrs.push(time_one(BinomialAlgorithm::Btrs, n, p, seed));
    }
    (median(btpe), median(btrs))
}

fn section(title: &str, cells: &[(&str, u64, f64)]) -> (f64, f64) {
    println!("\n{title}");
    println!("  {:<30} {:>10} {:>10} {:>9}", "regime", "BTPE ns", "BTRS ns", "speedup");
    println!("  {:-<30} {:->10} {:->10} {:->9}", "", "", "", "");
    let (mut sum_a, mut sum_b) = (0.0, 0.0);
    for &(label, n, p) in cells {
        let (a, b) = ab(n, p);
        sum_a += a;
        sum_b += b;
        println!("  {label:<30} {a:>10.2} {b:>10.2} {:>8.2}x", a / b);
    }
    let (ma, mb) = (sum_a / cells.len() as f64, sum_b / cells.len() as f64);
    println!("  {:<30} {ma:>10.2} {mb:>10.2} {:>8.2}x", "mean", ma / mb);
    (ma, mb)
}

fn main() {
    println!(
        "binomial A/B — {DRAWS} draws x {REPS} reps, median, interleaved arms\n\
         BINV (n*min(p,1-p) < 10) is shared by both arms and is NOT under test."
    );

    let (te_a, te_b) = section("TOTAL-EXIT draws (12 per particle-substep)", TOTAL_EXIT);
    let (sp_a, sp_b) = section("SPLIT draws (12 per particle-substep)", SPLIT);

    // Equal counts, so the blended per-draw cost is the plain mean of the two.
    let (blend_a, blend_b) = ((te_a + sp_a) / 2.0, (te_b + sp_b) / 2.0);
    let sampler_speedup = blend_a / blend_b;
    println!(
        "\nBLENDED (12 total-exit + 12 split, equal weight)\n  \
         BTPE {blend_a:.2} ns/draw -> BTRS {blend_b:.2} ns/draw = {sampler_speedup:.2}x sampler"
    );

    // Whole-fit factor: if the sampler is `share` of the fit and gets `s` times
    // faster, the fit gets 1 / (1 - share*(1 - 1/s)) times faster.
    println!(
        "\nIMPLIED WHOLE-FIT FACTOR (sampler {sampler_speedup:.2}x), by assumed binomial share:"
    );
    for share in [0.30, 0.35, 0.389, 0.45, 0.50] {
        let fit = 1.0 / (1.0 - share * (1.0 - 1.0 / sampler_speedup));
        println!("  share {:.1}%  ->  {fit:.3}x", share * 100.0);
    }
    println!(
        "\nThe profiled share was 38.9% for `binomial` alone (SE ~0.4pp, and a LOWER\n\
         bound: rand_distr is only partially inlined). BTRS keeps two uniforms per\n\
         attempt, so the 8.6% ChaCha8 slice is NOT removed by this change."
    );
}
