//! gh#201: the bootstrap particle filter's marginal likelihood against an
//! EXACT oracle.
//!
//! # What is being tested
//!
//! `bootstrap_filter` returns `ll = log p̂(y | θ)`. For the bootstrap filter with
//! multinomial-type (here systematic) resampling, `p̂ = exp(ll)` is an
//! **unbiased** estimator of the marginal likelihood `p(y | θ)` (Del Moral 2004,
//! *Feynman–Kac Formulae*, Thm 7.4.2; Andrieu, Doucet & Holenstein 2010, JRSS-B
//! 72:269, §4.1 — the property PMMH's exactness rests on). A filter that
//! miscounts a substep, double-counts or drops an increment, scores the wrong
//! window, or mis-resets an accumulator is not unbiased for the model it claims
//! to fit, and that bias survives every existing PF test (determinism,
//! finiteness, variance-shrinks-with-N), because it is deterministic.
//!
//! # The oracle is independent of camdl's likelihood code
//!
//! The model is a plain-binomial SIR (`ocaml/golden/sir_basic`) with `N₀ ≈ 6`.
//! Its state space is tiny, so `p(y | θ)` is computed EXACTLY by the forward
//! algorithm over the discrete state `(S, I, R, open incidence bins)`:
//!
//! ```text
//!   α₀(x)       = p(x₀ = x)
//!   substep:      α'(x') = Σ_x α(x) · K_h(x → x')         (hand-written kernel)
//!   observation:  c_k    = Σ_x α(x) · Π_streams f(y | proj(x))
//!                 α      ← α · Π f / c_k
//!   log p(y|θ)  = Σ_k log c_k
//! ```
//!
//! Nothing here calls camdl's density or likelihood code: the transition kernel
//! is written from the model's definition (two independent binomials per
//! substep, `p = 1 − exp(−rate·h)`), and every observation family is written
//! from the language spec §12.2 using `libm` special functions. The one place
//! the oracle and the engine must agree on a *convention* — that the kernel is
//! the law `step_one` actually draws — is measured, not assumed:
//! [`hand_written_kernel_is_the_law_step_one_draws`].
//!
//! # The test statistic
//!
//! Run the filter `R` times with independent seeds. Let
//! `w_r = exp(ll_r − log p(y|θ))`. Under H₀ (unbiased) `E[w] = 1`, so
//!
//! ```text
//!   z = (mean(w) − 1) / (sd(w) / √R)
//! ```
//!
//! is asymptotically N(0, 1). The test fails at `|z| > Z_CRIT = 4.5`, a
//! two-sided false-alarm rate of ≈ 7·10⁻⁶ per configuration under the normal
//! approximation. (The seeds are fixed, so a given build is deterministic; the
//! false-alarm rate is the chance that a correct filter lands on an unlucky
//! seed set.) `w` is right-skewed, which makes `sd(w)` an under-estimate in
//! small samples and the test anticonservative for a *positive* deviation;
//! `R` is chosen so the skew is negligible at these particle counts, and the
//! printed `sd(w)` is small (≈ 0.1–0.3).

use std::collections::HashMap;
use std::sync::Arc;

use serde_json::{json, Value};
use sim::compiled_model::CompiledModel;
use sim::inference::multi_stream_obs::{
    BoundObs, MultiStreamObsModel, ObsCell, Period, StreamProjection, StreamSpec,
};
use sim::inference::particle_filter::bootstrap_filter;
use sim::inference::{ChainBinomialProcess, SMCConfig};

const Z_CRIT: f64 = 4.5;

// ─── The model ──────────────────────────────────────────────────────────────

/// Compartment order in `sir_basic` (asserted at load).
const S: usize = 0;
const I: usize = 1;
/// Transition order in `sir_basic` (asserted at load).
const INF: usize = 0;
const REC: usize = 1;

/// Everything that defines one configuration: the dynamics, the initial law,
/// the scheduled effects and the observation streams. Both the camdl objects
/// and the oracle are built from this one description.
#[derive(Clone)]
struct Config {
    label: &'static str,
    n0: i64,
    i0: i64,
    beta: f64,
    gamma: f64,
    /// recovery rate is `gamma · (1 + gamma_slope · t)` — exercises the time
    /// passed to rate evaluation. 0 = the plain golden rate.
    gamma_slope: f64,
    dt: f64,
    t_start: f64,
    /// The model's declared end, `simulate { to }`: the horizon of the effect
    /// plan (the data never move an effect). `sir_basic` declares 80.
    t_end: f64,
    /// `Some(p)`: `I ~ binomial(n = N0, p)` and `S = N0 − I` (an initial-state
    /// law, gh#732). `None`: deterministic `S = N0 − I0`, `I = I0`.
    init_binomial_p: Option<f64>,
    /// Scheduled interventions and events (deterministic actions).
    effects: Vec<Effect>,
    streams: Vec<TStream>,
    skip_first: bool,
}

/// A deterministic state change.
#[derive(Clone, Copy, Debug)]
enum Act {
    /// `absolute_transfer(S -> R, count)`, capped at `S`.
    SToR(f64),
    /// `add(I, count)`.
    AddI(f64),
}

/// A scheduled effect at the declared times `ts` (an `at [...]` list): an
/// `interventions {}` entry, or an `events {}` entry when `event` is true.
#[derive(Clone, Copy, Debug)]
struct Effect {
    ts: &'static [f64],
    act: Act,
    event: bool,
}

impl Effect {
    /// The instants the model DEFINES this effect to act at, on the
    /// chain-binomial filter's grid (proposal 2026-10-04 §4–§5, the rule the
    /// language spec §13.7 states): a declared time `t` inside the window
    /// `[t_start, t_end]` acts at the end of step
    /// `k = ⌊(t − t_start)/dt + ½⌋` of the run's grid `t_start + k·dt`, the
    /// nearest step, a tie to the later one; inside a shortened last step
    /// (`t_end` off the grid) at the nearer of that step's two ends, a tie to
    /// `t_end`. One effect acts once per step however many of its times share
    /// it. Events and interventions alike. Written out here from that
    /// statement, not from camdl's `effect_plan`.
    fn acts_at(&self, cfg: &Config, t_end: f64) -> Vec<f64> {
        const TOL: f64 = 1e-9;
        let span = (t_end - cfg.t_start) / cfg.dt;
        let n = if span <= TOL { 0 } else { (span - TOL).ceil() as i64 };
        let mut out: Vec<f64> = Vec::new();
        for &t in self.ts {
            let u = (t - cfg.t_start) / cfg.dt;
            if u < -TOL || u > span + TOL {
                continue;
            }
            let mut k = ((u + 0.5 + TOL).floor() as i64).clamp(0, n);
            let last = (n - 1) as f64;
            if n > 0 && span - last < 1.0 - TOL && u > last {
                k = if span - u <= u - last + TOL { n } else { n - 1 };
            }
            let instant = if k == n && n > 0 { t_end } else { cfg.t_start + k as f64 * cfg.dt };
            if !out.iter().any(|&o| same(o, instant)) {
                out.push(instant);
            }
        }
        out
    }

    fn apply(&self, s: &mut i64, i: &mut i64, r: &mut i64) {
        match self.act {
            Act::SToR(c) => {
                let x = (c.round() as i64).min(*s);
                *s -= x;
                *r += x;
            }
            Act::AddI(c) => *i += c.round() as i64,
        }
    }
}

#[derive(Clone)]
enum Proj {
    /// Count in the named compartments at the observation instant.
    Prev(Vec<usize>),
    /// Flow of the named transitions over the row's period.
    Inc(Vec<usize>),
    /// `Σ numerator flow / Σ denominator flow` over the row's period (NaN
    /// when the denominator flow is 0).
    Ratio(Vec<usize>, Vec<usize>),
}

impl Proj {
    /// The transition sets this projection accumulates, one per bin.
    fn bins(&self) -> Vec<Vec<usize>> {
        match self {
            Proj::Prev(_) => vec![],
            Proj::Inc(t) => vec![t.clone()],
            Proj::Ratio(n, d) => vec![n.clone(), d.clone()],
        }
    }
}

/// Observation families, written from spec §12.2. `x` is the projected value.
#[derive(Clone, Copy, Debug)]
enum Fam {
    /// `poisson(rate = a·x + b)`
    Poisson { a: f64, b: f64 },
    /// `neg_binomial(mean = a·x + b, r)` — NB2, variance `μ + μ²/r`.
    NegBin { a: f64, b: f64, r: f64 },
    /// `binomial(n, p = a·x + b)`
    Binomial { n: u64, a: f64, b: f64 },
    /// `binomial(n = tested, p = a·x + b)`, `tested` a per-row data column.
    BinomialAux { a: f64, b: f64 },
    /// `beta_binomial(n, mean = a·x + b, concentration = c)` —
    /// `α = mean·c`, `β = (1 − mean)·c`.
    BetaBinomial { n: u64, a: f64, b: f64, c: f64 },
    /// `normal(mean = a·x + b, sd)` — camdl scores this as the pomp /
    /// He et al. (2010) discretized-Normal count likelihood (see the
    /// `normal_is_scored_as_a_discretized_count` test), NOT a continuous pdf.
    DNormal { a: f64, b: f64, sd: f64 },
}

#[derive(Clone)]
enum TTimes {
    /// A prevalence stream: the instants. An incidence stream: contiguous
    /// periods closing at each label, the first opening at `t_start`.
    Labels(Vec<f64>),
    /// Declared `[start, stop)` periods (incidence only).
    Periods(Vec<(f64, f64)>),
}

#[derive(Clone)]
struct TStream {
    name: &'static str,
    proj: Proj,
    fam: Fam,
    times: TTimes,
    /// One per row; `None` is a hole (NA).
    cells: Vec<Option<f64>>,
    /// The per-row `tested` column, for `Fam::BinomialAux`.
    tested: Option<Vec<u64>>,
}

impl TStream {
    /// The `[start, stop)` each row covers (incidence), or `(t, t)` (instant).
    fn rows(&self, t_start: f64) -> Vec<(f64, f64)> {
        match (&self.times, &self.proj) {
            (TTimes::Labels(ts), Proj::Prev(_)) => ts.iter().map(|&t| (t, t)).collect(),
            (TTimes::Labels(ts), Proj::Inc(_) | Proj::Ratio(..)) => {
                let mut prev = t_start;
                ts.iter()
                    .map(|&t| {
                        let r = (prev, t);
                        prev = t;
                        r
                    })
                    .collect()
            }
            (TTimes::Periods(ps), _) => ps.clone(),
        }
    }
}

// ─── Oracle: hand-written pmfs ──────────────────────────────────────────────

fn ln_choose(n: u64, k: u64) -> f64 {
    libm::lgamma(n as f64 + 1.0) - libm::lgamma(k as f64 + 1.0) - libm::lgamma((n - k) as f64 + 1.0)
}

fn binom_pmf(n: i64, k: i64, p: f64) -> f64 {
    if k < 0 || k > n {
        return 0.0;
    }
    if p <= 0.0 {
        return if k == 0 { 1.0 } else { 0.0 };
    }
    if p >= 1.0 {
        return if k == n { 1.0 } else { 0.0 };
    }
    let (n, k) = (n as u64, k as u64);
    (ln_choose(n, k) + k as f64 * p.ln() + (n - k) as f64 * (1.0 - p).ln()).exp()
}

fn phi(z: f64) -> f64 {
    0.5 * libm::erfc(-z / std::f64::consts::SQRT_2)
}

/// `f(y | x)` for one family, as a probability (not a log).
fn obs_pmf(fam: Fam, y: f64, x: f64, tested: Option<u64>) -> f64 {
    let yk = y.round() as u64;
    // A ratio over an empty window is not a number (spec §12.1): a binomial
    // row with n = 0 has one outcome, with n > 0 it cannot be scored.
    if x.is_nan() {
        let n = match fam {
            Fam::Binomial { n, .. } => n,
            Fam::BinomialAux { .. } => tested.unwrap(),
            other => panic!("{other:?} fed a NaN projection"),
        };
        return if n == 0 && yk == 0 { 1.0 } else { 0.0 };
    }
    match fam {
        Fam::BinomialAux { a, b } => binom_pmf(tested.unwrap() as i64, yk as i64, a * x + b),
        Fam::Poisson { a, b } => {
            let lam = a * x + b;
            if lam <= 0.0 {
                return if yk == 0 { 1.0 } else { 0.0 };
            }
            (yk as f64 * lam.ln() - lam - libm::lgamma(yk as f64 + 1.0)).exp()
        }
        Fam::NegBin { a, b, r } => {
            let mu = a * x + b;
            // P(y) = Γ(y+r)/(Γ(r) y!) · (r/(r+μ))^r · (μ/(r+μ))^y
            let lp = libm::lgamma(yk as f64 + r) - libm::lgamma(r) - libm::lgamma(yk as f64 + 1.0)
                + r * (r / (r + mu)).ln()
                + if yk == 0 { 0.0 } else { yk as f64 * (mu / (r + mu)).ln() };
            lp.exp()
        }
        Fam::Binomial { n, a, b } => binom_pmf(n as i64, yk as i64, a * x + b),
        Fam::BetaBinomial { n, a, b, c } => {
            if yk > n {
                return 0.0;
            }
            let m = a * x + b;
            let (al, be) = (m * c, (1.0 - m) * c);
            let lbeta = |p: f64, q: f64| libm::lgamma(p) + libm::lgamma(q) - libm::lgamma(p + q);
            (ln_choose(n, yk) + lbeta(yk as f64 + al, (n - yk) as f64 + be) - lbeta(al, be)).exp()
        }
        Fam::DNormal { a, b, sd } => {
            // He, Ionides & King (2010) J R Soc Interface 7:271, the pomp
            // measles dmeasure: P(y) = Φ((y+½−μ)/σ) − Φ((y−½−μ)/σ) for y > 0,
            // and Φ((½−μ)/σ) at y = 0 (the lower tail lumped into zero).
            let m = a * x + b;
            if yk == 0 {
                phi((0.5 - m) / sd)
            } else {
                phi((yk as f64 + 0.5 - m) / sd) - phi((yk as f64 - 0.5 - m) / sd)
            }
        }
    }
}

// ─── Oracle: exact forward algorithm ────────────────────────────────────────

/// `(S, I, R, open incidence bins)` — one bin per incidence stream, in stream
/// order.
type Key = (i64, i64, i64, Vec<u64>);

/// What happens at one instant on the oracle's timeline, in lifecycle order.
#[derive(Default, Debug)]
struct Instant {
    /// Effects acting here (index into `Config::effects`).
    effects: Vec<usize>,
    /// `(stream, row)` scored here.
    scores: Vec<(usize, usize)>,
    /// Incidence bins that OPEN here (period starts) — zeroed after scoring.
    opens: Vec<usize>,
}

fn same(a: f64, b: f64) -> bool {
    (a - b).abs() <= 1e-9
}

/// The exact `log p(y | θ)` and the per-scoring-instant increments
/// `log p(y_k | y_{1:k−1})` (one per instant carrying at least one scored
/// cell, in time order).
fn oracle(cfg: &Config) -> (f64, Vec<f64>) {
    let (bin_of, bin_trs) = bin_layout(cfg);
    let n_bins = bin_trs.len();

    // The timeline: every instant anything happens.
    let mut times: Vec<f64> = Vec::new();
    for (i, s) in cfg.streams.iter().enumerate() {
        for (start, stop) in s.rows(cfg.t_start) {
            times.push(stop);
            if !bin_of[i].is_empty() && start > cfg.t_start + 1e-9 {
                times.push(start);
            }
        }
    }
    let last_obs = times.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    for e in &cfg.effects {
        for t in e.acts_at(cfg, cfg.t_end) {
            if t > cfg.t_start + 1e-9 && t <= last_obs + 1e-9 {
                times.push(t);
            }
        }
    }
    times.sort_by(|a, b| a.partial_cmp(b).unwrap());
    times.dedup_by(|a, b| same(*a, *b));
    let mut instants: Vec<Instant> = times.iter().map(|_| Instant::default()).collect();
    let at = |t: f64| times.iter().position(|&u| same(u, t)).unwrap();
    for (k, e) in cfg.effects.iter().enumerate() {
        for t in e.acts_at(cfg, cfg.t_end) {
            if t > cfg.t_start + 1e-9 && t <= last_obs + 1e-9 {
                instants[at(t)].effects.push(k);
            }
        }
    }
    for (i, s) in cfg.streams.iter().enumerate() {
        for (row, (start, stop)) in s.rows(cfg.t_start).into_iter().enumerate() {
            instants[at(stop)].scores.push((i, row));
            if start > cfg.t_start + 1e-9 {
                instants[at(start)].opens.extend(bin_of[i].iter().copied());
            }
        }
    }

    // Initial law.
    let mut alpha: HashMap<Key, f64> = HashMap::new();
    match cfg.init_binomial_p {
        None => {
            alpha.insert((cfg.n0 - cfg.i0, cfg.i0, 0, vec![0; n_bins]), 1.0);
        }
        Some(p) => {
            for i in 0..=cfg.n0 {
                let w = binom_pmf(cfg.n0, i, p);
                if w > 0.0 {
                    alpha.insert((cfg.n0 - i, i, 0, vec![0; n_bins]), w);
                }
            }
        }
    }

    // An effect AT `t_start` acts on the initial state.
    let at_start: Vec<usize> = (0..cfg.effects.len())
        .filter(|&k| cfg.effects[k].acts_at(cfg, cfg.t_end).iter().any(|&t| same(t, cfg.t_start)))
        .collect();
    alpha = apply_effects(cfg, alpha, &at_start);

    let mut total = 0.0;
    let mut increments = Vec::new();
    let mut first_scored = true;
    let mut t = cfg.t_start;
    for (tau, inst) in times.iter().zip(&instants) {
        // Substeps from `t` to `tau`, re-anchored at `t`: t0 = t + s·dt,
        // h = min(dt, tau − t0), the last one clipped onto `tau`.
        let mut s = 0u64;
        loop {
            let t0 = t + s as f64 * cfg.dt;
            let h = cfg.dt.min(tau - t0);
            if h <= 1e-12 {
                break;
            }
            alpha = substep(cfg, &alpha, t0, h);
            s += 1;
        }
        t = *tau;

        // Effects (post-advance), in declaration order.
        alpha = apply_effects(cfg, alpha, &inst.effects);

        // Score every observed cell closing here.
        let observed: Vec<(usize, f64, usize)> = inst
            .scores
            .iter()
            .filter_map(|&(si, row)| cfg.streams[si].cells[row].map(|y| (si, y, row)))
            .collect();
        if !observed.is_empty() {
            let mut c = 0.0;
            for ((s_, i_, r_, bins), w) in alpha.iter_mut() {
                let counts = [*s_, *i_, *r_];
                let mut f = 1.0;
                for &(si, y, row) in &observed {
                    let st = &cfg.streams[si];
                    let x = match &st.proj {
                        Proj::Prev(cs) => cs.iter().map(|&c| counts[c] as f64).sum(),
                        Proj::Inc(_) => bins[bin_of[si][0]] as f64,
                        Proj::Ratio(..) => {
                            let (n, d) = (bins[bin_of[si][0]] as f64, bins[bin_of[si][1]] as f64);
                            if d == 0.0 { f64::NAN } else { n / d }
                        }
                    };
                    f *= obs_pmf(st.fam, y, x, st.tested.as_ref().map(|t| t[row]));
                }
                *w *= f;
                c += *w;
            }
            assert!(c > 0.0, "{}: the data have zero probability under the model", cfg.label);
            for w in alpha.values_mut() {
                *w /= c;
            }
            increments.push(c.ln());
            if !(cfg.skip_first && first_scored) {
                total += c.ln();
            }
            first_scored = false;
        }

        // Reset: an incidence stream on contiguous labels restarts at every
        // close; a declared period opens at its own start.
        let mut to_zero: Vec<usize> = inst.opens.clone();
        for &(si, _) in &inst.scores {
            if let TTimes::Labels(_) = &cfg.streams[si].times {
                to_zero.extend(bin_of[si].iter().copied());
            }
        }
        if !to_zero.is_empty() {
            let mut next: HashMap<Key, f64> = HashMap::new();
            for ((s_, i_, r_, mut bins), w) in alpha {
                for &b in &to_zero {
                    bins[b] = 0;
                }
                *next.entry((s_, i_, r_, bins)).or_insert(0.0) += w;
            }
            alpha = next;
        }
    }
    (total, increments)
}

/// Which bins each stream owns, and which transitions each bin sums — in
/// stream order, a ratio stream's numerator bin before its denominator.
fn bin_layout(cfg: &Config) -> (Vec<Vec<usize>>, Vec<Vec<usize>>) {
    let mut bin_of = Vec::new();
    let mut bin_trs = Vec::new();
    for s in &cfg.streams {
        let mut mine = Vec::new();
        for trs in s.proj.bins() {
            mine.push(bin_trs.len());
            bin_trs.push(trs);
        }
        bin_of.push(mine);
    }
    (bin_of, bin_trs)
}

/// The effects acting at one instant, in the language spec §13.10 order: the
/// events first, then the interventions, each in declaration order. (The
/// events here are `add`, inflow only, so the events' own two phases do not
/// arise.)
fn apply_effects(cfg: &Config, alpha: HashMap<Key, f64>, which: &[usize]) -> HashMap<Key, f64> {
    if which.is_empty() {
        return alpha;
    }
    let mut order: Vec<usize> = which.to_vec();
    order.sort_by_key(|&k| (!cfg.effects[k].event, k));
    let mut next: HashMap<Key, f64> = HashMap::new();
    for ((mut s, mut i, mut r, bins), w) in alpha {
        for &k in &order {
            cfg.effects[k].apply(&mut s, &mut i, &mut r);
        }
        *next.entry((s, i, r, bins)).or_insert(0.0) += w;
    }
    next
}

/// One substep of length `h` starting at `t0`: `k_inf ~ Bin(S, 1 − e^{−β·I/N·h})`
/// and `k_rec ~ Bin(I, 1 − e^{−γ(t0)·h})`, independent — the two transitions
/// have different source compartments, so there is no competing-exit split.
fn substep(cfg: &Config, alpha: &HashMap<Key, f64>, t0: f64, h: f64) -> HashMap<Key, f64> {
    let (_, bin_trs) = bin_layout(cfg);
    let mut next: HashMap<Key, f64> = HashMap::new();
    for ((s, i, r, bins), &w) in alpha {
        let n = (s + i + r) as f64;
        let p_inf = if *s > 0 && *i > 0 { 1.0 - (-cfg.beta * *i as f64 / n * h).exp() } else { 0.0 };
        let g = cfg.gamma * (1.0 + cfg.gamma_slope * t0);
        let p_rec = if *i > 0 { 1.0 - (-g * h).exp() } else { 0.0 };
        for ki in 0..=*s {
            let pi = binom_pmf(*s, ki, p_inf);
            if pi == 0.0 {
                continue;
            }
            for kr in 0..=*i {
                let pr = binom_pmf(*i, kr, p_rec);
                if pr == 0.0 {
                    continue;
                }
                let mut b2 = bins.clone();
                let flows = [ki as u64, kr as u64];
                for (bk, trs) in bin_trs.iter().enumerate() {
                    b2[bk] += trs.iter().map(|&tr| flows[tr]).sum::<u64>();
                }
                *next.entry((s - ki, i + ki - kr, r + kr, b2)).or_insert(0.0) += w * pi * pr;
            }
        }
    }
    next
}

// ─── camdl side: build the model and the observation model ──────────────────

fn affine(a: f64, b: f64) -> Value {
    json!({"bin_op": {"op": "add",
        "left": {"bin_op": {"op": "mul", "left": {"const": a}, "right": {"projected": null}}},
        "right": {"const": b}}})
}

fn likelihood_json(f: Fam) -> Value {
    match f {
        Fam::Poisson { a, b } => json!({"poisson": {"rate": {"expr": affine(a, b)}}}),
        Fam::NegBin { a, b, r } => json!({"neg_binomial": {
            "mean": {"expr": affine(a, b)}, "dispersion": {"expr": {"const": r}}}}),
        Fam::Binomial { n, a, b } => json!({"binomial": {
            "n": {"const": n as f64}, "p": {"expr": affine(a, b)}}}),
        Fam::BinomialAux { a, b } => json!({"binomial": {
            "n": {"obs_column_ref": "tested"}, "p": {"expr": affine(a, b)}}}),
        Fam::BetaBinomial { n, a, b, c } => json!({"beta_binomial": {
            "n": {"const": n as f64},
            "alpha": {"expr": {"bin_op": {"op": "mul", "left": affine(a, b), "right": {"const": c}}}},
            "beta": {"expr": {"bin_op": {"op": "mul",
                "left": {"bin_op": {"op": "sub", "left": {"const": 1.0}, "right": affine(a, b)}},
                "right": {"const": c}}}}}}),
        Fam::DNormal { a, b, sd } => json!({"normal": {
            "mean": {"expr": affine(a, b)}, "sd": {"expr": {"const": sd}}}}),
    }
}

const COMP: [&str; 3] = ["S", "I", "R"];
const TRANS: [&str; 2] = ["infection", "recovery"];

fn projection_json(p: &Proj) -> Value {
    match p {
        Proj::Prev(cs) if cs.len() == 1 => json!({"current_pop": COMP[cs[0]]}),
        Proj::Prev(cs) => json!({"current_pop_sum": cs.iter().map(|&c| COMP[c]).collect::<Vec<_>>()}),
        Proj::Inc(ts) if ts.len() == 1 => json!({"cumulative_flow": TRANS[ts[0]]}),
        Proj::Inc(ts) => {
            json!({"cumulative_flow_sum": ts.iter().map(|&t| TRANS[t]).collect::<Vec<_>>()})
        }
        Proj::Ratio(n, d) => json!({"flow_ratio": {
            "numerator": n.iter().map(|&t| TRANS[t]).collect::<Vec<_>>(),
            "denominator": d.iter().map(|&t| TRANS[t]).collect::<Vec<_>>()}}),
    }
}

fn obs_block(s: &TStream) -> ir::observation::ObservationModel {
    let v = json!({
        "name": s.name, "source": s.name,
        "columns": if s.tested.is_some() {
            json!([{"name": "time", "role": "time"}, {"name": s.name, "role": {"value": "count"}},
                   {"name": "tested", "role": {"value": "count"}}])
        } else {
            json!([{"name": "time", "role": "time"}, {"name": s.name, "role": {"value": "count"}}])
        },
        "scored": s.name,
        "projection": projection_json(&s.proj),
        "likelihood": likelihood_json(s.fam),
    });
    serde_json::from_value(v).expect("observation block JSON")
}

fn build(cfg: &Config) -> (Arc<CompiledModel>, Vec<f64>, MultiStreamObsModel) {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../../ocaml/golden/sir_basic.ir.json");
    let mut doc: Value =
        serde_json::from_str(&std::fs::read_to_string(&path).expect("read sir_basic")).unwrap();
    let m = &mut doc["model"];
    if cfg.gamma_slope != 0.0 {
        // recovery : I --> R @ gamma * (1 + slope * t) * I
        m["transitions"][REC]["rate"] = json!({"bin_op": {"op": "mul",
            "left": {"bin_op": {"op": "mul", "left": {"param": "gamma"},
                "right": {"bin_op": {"op": "add", "left": {"const": 1.0},
                    "right": {"bin_op": {"op": "mul", "left": {"const": cfg.gamma_slope},
                        "right": {"time": null}}}}}}},
            "right": {"pop": "I"}}});
    }
    if let Some(p) = cfg.init_binomial_p {
        m["initial_conditions"] = json!({
            "S": {"deterministic": {"bin_op": {"op": "sub", "left": {"param": "N0"}, "right": {"pop": "I"}}}},
            "I": {"count": {"binomial": {"n": {"param": "N0"}, "p": {"expr": {"const": p}}}}},
        });
    }
    if !cfg.effects.is_empty() {
        m["interventions"] = Value::Array(
            cfg.effects
                .iter()
                .enumerate()
                .map(|(k, e)| {
                    let action = match e.act {
                        Act::SToR(c) => json!({"absolute_transfer": {"src": "S", "dst": "R", "count": {"const": c}}}),
                        Act::AddI(c) => json!({"add": {"compartment": "I", "count": {"const": c}}}),
                    };
                    let mut v = json!({"name": format!("effect_{k}"),
                        "fire": {"scheduled": {"at_times": e.ts}},
                        "actions": [action]});
                    if e.event {
                        v["kind"] = json!("event");
                    }
                    v
                })
                .collect(),
        );
    }
    let mut model = ir::from_str(&serde_json::to_string(&doc).unwrap()).expect("parse patched sir_basic");
    model.observations = cfg.streams.iter().map(obs_block).collect();
    model.simulation.t_start = cfg.t_start;
    model.simulation.t_end = cfg.t_end;
    for p in &mut model.parameters {
        let v = match p.name.as_str() {
            "beta" => cfg.beta,
            "gamma" => cfg.gamma,
            "N0" => cfg.n0 as f64,
            "I0" => cfg.i0 as f64,
            other => panic!("unexpected parameter {other}"),
        };
        p.value = ir::parameter::ParamValue::Fixed { value: v };
    }
    let compiled = Arc::new(CompiledModel::new(model).expect("compile"));
    assert_eq!(
        compiled.model.compartments.iter().map(|c| c.name.as_str()).collect::<Vec<_>>(),
        COMP.to_vec()
    );
    assert_eq!(
        compiled.model.transitions.iter().map(|t| t.name.as_str()).collect::<Vec<_>>(),
        TRANS.to_vec()
    );
    let params = compiled.default_params.clone();

    let specs: Vec<StreamSpec> = cfg
        .streams
        .iter()
        .zip(&compiled.model.observations)
        .map(|(s, ir_obs)| {
            let proj = StreamProjection::from_ir(&ir_obs.projection, &compiled, s.name).unwrap();
            let cells: Vec<Option<ObsCell>> = s.cells.iter().map(|c| c.map(ObsCell::Scalar)).collect();
            let mut spec = match &s.times {
                TTimes::Labels(ts) => StreamSpec::dense(cfg.t_start, proj, ir_obs.clone(), cells, ts.clone()),
                TTimes::Periods(ps) => StreamSpec::dense_covering(
                    proj,
                    ir_obs.clone(),
                    cells,
                    ps.iter().map(|&(a, b)| Period::new(a, b).unwrap()).collect(),
                ),
            };
            if let Some(tested) = &s.tested {
                spec.aux = tested.iter().map(|&n| vec![("tested".to_string(), n as f64)]).collect();
            }
            spec
        })
        .collect();
    let bound = BoundObs::bind(cfg.t_start, specs)
        .unwrap_or_else(|r| panic!("{}: bind failed: {}", cfg.label, r.render()))
        .0;
    let obs_model = MultiStreamObsModel::new(bound, compiled.clone()).expect("obs model");
    (compiled, params, obs_model)
}

fn run_pf(
    cfg: &Config,
    built: &(Arc<CompiledModel>, Vec<f64>, MultiStreamObsModel),
    n_particles: usize,
    seed: u64,
) -> (f64, Vec<f64>) {
    let (compiled, params, obs_model) = built;
    let process = ChainBinomialProcess::new(compiled.clone());
    let config = SMCConfig {
        n_particles,
        dt: cfg.dt,
        t_start: cfg.t_start,
        skip_first_obs_from_loglik: cfg.skip_first,
        record_ancestry: false,
        record_prequential: false,
        record_predictions: false,
        max_substeps: sim::inference::degeneracy::ITER_BUDGET,
    };
    let r = bootstrap_filter(&process, obs_model, params, &config, seed)
        .unwrap_or_else(|e| panic!("{}: bootstrap_filter failed: {e}", cfg.label));
    (r.log_likelihood, r.ll_increments)
}

/// The unbiasedness check. Returns `(z, exact, log-mean-exp)` for reporting.
fn check_unbiased(cfg: &Config, n_particles: usize, reps: usize) -> (f64, f64, f64) {
    let (exact, _) = oracle(cfg);
    let built = build(cfg);
    let w: Vec<f64> = (0..reps)
        .map(|r| {
            let (ll, _) = run_pf(cfg, &built, n_particles, 0x0bad_5eed_0000u64 + r as u64 * 7919);
            (ll - exact).exp()
        })
        .collect();
    let n = w.len() as f64;
    let mean = w.iter().sum::<f64>() / n;
    let sd = (w.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (n - 1.0)).sqrt();
    let z = (mean - 1.0) / (sd / n.sqrt());
    let lme = exact + mean.ln();
    eprintln!(
        "[{}] exact log p(y|θ) = {exact:.6}; PF log-mean-exp = {lme:.6} (gap {:+.4} nats); \
         mean(w) = {mean:.4}, sd(w) = {sd:.3}, R = {reps}, N = {n_particles}, z = {z:+.2}",
        cfg.label,
        lme - exact
    );
    (z, exact, lme)
}

fn assert_unbiased(cfg: &Config, n_particles: usize, reps: usize) {
    let (z, exact, lme) = check_unbiased(cfg, n_particles, reps);
    assert!(
        z.abs() < Z_CRIT,
        "{}: the particle-filter likelihood is biased. exact log p(y|θ) = {exact:.5}, \
         log of the mean PF likelihood over {reps} seeds = {lme:.5} (gap {:+.4} nats), \
         z = {z:+.2} (|z| < {Z_CRIT} required)",
        cfg.label,
        lme - exact
    );
    assert_consistent(cfg, exact);
}

/// Unbiasedness across seeds is not enough on its own: a filter that
/// conditions every particle on ONE draw of a random quantity (x₀, say) is
/// still unbiased over seeds — it is a valid but noisier estimator — yet a
/// single run no longer converges to `p(y | θ)` as N grows. So also require
/// every one of a few large-N runs to sit within `CONSISTENT_NATS` of the
/// exact value. At N = 5000 the per-run sd of `ll` here is ≈ 0.01–0.02 nats,
/// so the bound is ≳ 5 sd.
fn assert_consistent(cfg: &Config, exact: f64) {
    const N_BIG: usize = 5000;
    const RUNS: usize = 20;
    const CONSISTENT_NATS: f64 = 0.1;
    let built = build(cfg);
    let lls: Vec<f64> =
        (0..RUNS).map(|r| run_pf(cfg, &built, N_BIG, 0xc0_115e_0000u64 + r as u64 * 104729).0).collect();
    let worst = lls.iter().map(|ll| (ll - exact).abs()).fold(0.0, f64::max);
    let mean = lls.iter().sum::<f64>() / RUNS as f64;
    eprintln!(
        "[{}] consistency at N = {N_BIG}: mean ll − exact = {:+.4}, worst |ll − exact| = {worst:.4}",
        cfg.label,
        mean - exact
    );
    assert!(
        worst < CONSISTENT_NATS,
        "{}: a single large-N run does not converge to the exact likelihood: worst |ll − exact| = \
         {worst:.4} nats over {RUNS} runs at N = {N_BIG} (bound {CONSISTENT_NATS})",
        cfg.label
    );
}

// ─── Configurations ─────────────────────────────────────────────────────────

fn base(label: &'static str, streams: Vec<TStream>) -> Config {
    Config {
        label,
        n0: 6,
        i0: 2,
        beta: 1.2,
        gamma: 0.5,
        gamma_slope: 0.0,
        dt: 1.0,
        t_start: 0.0,
        t_end: 80.0,
        init_binomial_p: None,
        effects: vec![],
        streams,
        skip_first: false,
    }
}

fn prev(name: &'static str, fam: Fam, times: Vec<f64>, cells: Vec<Option<f64>>) -> TStream {
    TStream { name, proj: Proj::Prev(vec![I]), fam, times: TTimes::Labels(times), cells, tested: None }
}

fn inc(name: &'static str, trs: Vec<usize>, fam: Fam, times: TTimes, cells: Vec<Option<f64>>) -> TStream {
    TStream { name, proj: Proj::Inc(trs), fam, times, cells, tested: None }
}

fn some(v: &[f64]) -> Vec<Option<f64>> {
    v.iter().map(|&x| Some(x)).collect()
}

const POIS: Fam = Fam::Poisson { a: 0.8, b: 0.3 };

// ─── The oracle checks itself ───────────────────────────────────────────────

/// The oracle's building blocks are right before anything is compared to it:
/// every family sums to 1 over its support, and the one-substep marginal of a
/// hand-computed case matches a closed form.
#[test]
fn oracle_building_blocks_are_probability_laws() {
    for fam in [
        Fam::Poisson { a: 0.8, b: 0.3 },
        Fam::NegBin { a: 0.8, b: 0.3, r: 2.5 },
        Fam::Binomial { n: 7, a: 0.1, b: 0.1 },
        Fam::BetaBinomial { n: 7, a: 0.1, b: 0.1, c: 4.0 },
        Fam::DNormal { a: 1.0, b: 0.2, sd: 1.3 },
    ] {
        for x in [0.0, 1.0, 3.0, 6.0] {
            let tot: f64 = (0..400).map(|y| obs_pmf(fam, y as f64, x, None)).sum();
            assert!((tot - 1.0).abs() < 1e-9, "{fam:?} at x={x} sums to {tot}");
        }
    }
    // One substep from (S,I,R)=(4,2,0), one prevalence reading of I with
    // Poisson(rate = I): p(y) = Σ_{ki,kr} Bin(ki;4,pi)·Bin(kr;2,pr)·Pois(y; 2+ki−kr).
    let cfg = Config {
        n0: 6,
        i0: 2,
        ..base("closed form", vec![prev("p", Fam::Poisson { a: 1.0, b: 0.0 }, vec![1.0], some(&[3.0]))])
    };
    let pi = 1.0 - (-1.2f64 * 2.0 / 6.0).exp();
    let pr = 1.0 - (-0.5f64).exp();
    let mut want = 0.0;
    for ki in 0..=4i64 {
        for kr in 0..=2i64 {
            let lam = (2 + ki - kr) as f64;
            let pois = if lam == 0.0 { 0.0 } else { (-lam).exp() * lam.powi(3) / 6.0 };
            want += binom_pmf(4, ki, pi) * binom_pmf(2, kr, pr) * pois;
        }
    }
    let (got, _) = oracle(&cfg);
    assert!((got - want.ln()).abs() < 1e-12, "oracle {got} vs closed form {}", want.ln());
}

/// The oracle's transition kernel must be the law `step_one` DRAWS, or the
/// oracle describes a different model than the engine runs. Measured cell by
/// cell against 400k draws, at a full step at t=0, a clipped half step, and —
/// with the time-dependent recovery rate — a step starting at t=3.
#[test]
fn hand_written_kernel_is_the_law_step_one_draws() {
    use sim::chain_binomial::{step_one, StepScratch};
    use sim::rng::StatefulRng;
    use sim::state::RealState;

    for (label, slope, t0, h) in
        [("full step", 0.0, 0.0, 1.0), ("clipped step", 0.0, 0.0, 0.5), ("t-dependent at t=3", 0.4, 3.0, 1.0)]
    {
        let mut cfg = base("kernel", vec![prev("p", POIS, vec![1.0], some(&[1.0]))]);
        cfg.gamma_slope = slope;
        let (compiled, params, _) = build(&cfg);
        let start = vec![4i64, 2, 0];
        let m = 400_000usize;
        let mut tally: HashMap<(u64, u64), u64> = HashMap::new();
        let mut scratch = StepScratch::new(&compiled);
        for k in 0..m {
            let mut rng = StatefulRng::new_stream(0x7e57_0000u64 + k as u64, 0);
            let mut counts = start.clone();
            let mut flows = vec![0u64; 2];
            let mut real = RealState::new(compiled.real_local_to_global.len());
            step_one(&compiled, &mut counts, &mut flows, &mut real, &params, t0, h, None, &mut rng, &mut scratch)
                .unwrap();
            *tally.entry((flows[INF], flows[REC])).or_insert(0) += 1;
        }
        let mut alpha = HashMap::new();
        alpha.insert((4i64, 2i64, 0i64, vec![]), 1.0);
        let cfg0 = Config { streams: vec![], ..cfg.clone() };
        let next = substep(&cfg0, &alpha, t0, h);
        let mut worst = 0.0f64;
        for ((s2, _i2, r2, _), p) in next {
            let key = ((4 - s2) as u64, r2 as u64);
            let emp = *tally.get(&key).unwrap_or(&0) as f64 / m as f64;
            if p * m as f64 >= 25.0 {
                worst = worst.max(((emp - p) / (p * (1.0 - p) / m as f64).sqrt()).abs());
            }
        }
        eprintln!("kernel check [{label}]: worst cell |z| = {worst:.2}");
        assert!(worst < 5.0, "[{label}] the hand-written kernel is not step_one's law (|z| = {worst:.2})");
    }
}

// ─── The sweep ──────────────────────────────────────────────────────────────

const NP: usize = 100;
const REPS: usize = 2000;

fn daily(n: usize) -> Vec<f64> {
    (1..=n).map(|k| k as f64).collect()
}

#[test]
fn prevalence_poisson_every_substep() {
    let cfg = base("prevalence, poisson, every substep", vec![prev("p", POIS, daily(5), some(&[2.0, 3.0, 2.0, 1.0, 1.0]))]);
    assert_unbiased(&cfg, NP, REPS);
}

#[test]
fn incidence_poisson_every_substep() {
    let cfg = base(
        "incidence, poisson, every substep",
        vec![inc("c", vec![INF], POIS, TTimes::Labels(daily(5)), some(&[2.0, 1.0, 1.0, 0.0, 0.0]))],
    );
    assert_unbiased(&cfg, NP, REPS);
}

/// A window spanning several substeps: the incidence bin must sum every
/// substep's flow in the window, and the prevalence read must be at the end.
#[test]
fn multi_substep_windows() {
    let mut cfg = base(
        "3-substep windows, incidence + prevalence",
        vec![
            inc("c", vec![INF], POIS, TTimes::Labels(vec![3.0, 6.0, 9.0]), some(&[3.0, 1.0, 0.0])),
            prev("p", POIS, vec![3.0, 6.0, 9.0], some(&[2.0, 1.0, 0.0])),
        ],
    );
    cfg.beta = 0.9;
    cfg.gamma = 0.3;
    assert_unbiased(&cfg, NP, REPS);
}

#[test]
fn each_family_on_prevalence_and_incidence() {
    for fam in [
        Fam::NegBin { a: 0.8, b: 0.3, r: 2.5 },
        Fam::Binomial { n: 7, a: 0.1, b: 0.1 },
        Fam::BetaBinomial { n: 7, a: 0.1, b: 0.1, c: 4.0 },
        Fam::DNormal { a: 1.0, b: 0.2, sd: 1.3 },
    ] {
        let label: &'static str = Box::leak(format!("family {fam:?}").into_boxed_str());
        let cfg = base(
            label,
            vec![
                prev("p", fam, vec![1.0, 2.0, 4.0], some(&[2.0, 3.0, 1.0])),
                inc("c", vec![INF], fam, TTimes::Labels(vec![2.0, 4.0]), some(&[2.0, 1.0])),
            ],
        );
        assert_unbiased(&cfg, NP, REPS);
    }
}

/// Holes (NA cells): leading, interior and trailing, on incidence and
/// prevalence. A hole scores nothing but still closes its incidence period.
#[test]
fn holes_leading_interior_trailing() {
    let cfg = base(
        "holes",
        vec![
            inc("c", vec![INF], POIS, TTimes::Labels(daily(6)), vec![None, Some(2.0), None, Some(1.0), Some(0.0), None]),
            prev("p", POIS, daily(6), vec![Some(3.0), None, None, Some(2.0), Some(1.0), None]),
        ],
    );
    assert_unbiased(&cfg, NP, REPS);
}

/// Streams on different cadences: a daily incidence stream, a 2-day
/// recovery-incidence stream (its bin must keep accumulating across the union
/// point where only its sibling scores), and a 3-day prevalence stream.
#[test]
fn multi_cadence_streams() {
    let cfg = base(
        "multi-cadence",
        vec![
            inc("inf_daily", vec![INF], POIS, TTimes::Labels(daily(6)), some(&[2.0, 1.0, 1.0, 0.0, 0.0, 0.0])),
            inc("rec_2day", vec![REC], POIS, TTimes::Labels(vec![2.0, 4.0, 6.0]), some(&[2.0, 2.0, 1.0])),
            prev("p_3day", POIS, vec![3.0, 6.0], some(&[2.0, 0.0])),
        ],
    );
    assert_unbiased(&cfg, NP, REPS);
}

/// An incidence projection summing two transitions.
#[test]
fn incidence_sum_of_two_flows() {
    let cfg = base(
        "incidence sum inf+rec",
        vec![inc("c", vec![INF, REC], POIS, TTimes::Labels(vec![2.0, 4.0, 5.0]), some(&[4.0, 2.0, 1.0]))],
    );
    assert_unbiased(&cfg, NP, REPS);
}

/// Declared periods: the first opening after `t_start` (a leading reset-only
/// boundary, gh#833) and a gap between two periods whose flow belongs to no
/// row.
#[test]
fn declared_periods_leading_edge_and_gap() {
    let cfg = base(
        "declared periods: leading edge + gap",
        vec![inc(
            "c",
            vec![INF],
            POIS,
            TTimes::Periods(vec![(1.0, 2.0), (2.0, 3.0), (4.0, 6.0)]),
            some(&[2.0, 1.0, 0.0]),
        )],
    );
    assert_unbiased(&cfg, NP, REPS);
}

/// Observation times off the dt grid: the substep is clipped onto each one.
#[test]
fn off_grid_observation_times() {
    let cfg = base(
        "off-grid obs (dt=1, obs at 1.5, 2.5, 4, 4.25)",
        vec![
            inc("c", vec![INF], POIS, TTimes::Labels(vec![1.5, 2.5, 4.0, 4.25]), some(&[2.0, 1.0, 1.0, 0.0])),
            prev("p", POIS, vec![1.5, 4.0], some(&[3.0, 2.0])),
        ],
    );
    assert_unbiased(&cfg, NP, REPS);
}

/// A time-dependent rate: catches a substep-start time that is off by a
/// substep, not re-anchored at a window start, or measured from 0 rather than
/// `t_start`. Run with windows of several substeps, off-grid obs, and a
/// non-zero `t_start`.
#[test]
fn time_dependent_rate() {
    let mut cfg = base(
        "t-dependent recovery, t_start=10, off-grid",
        vec![
            inc("rec", vec![REC], POIS, TTimes::Labels(vec![12.0, 13.5, 16.0]), some(&[2.0, 2.0, 1.0])),
            prev("p", POIS, vec![12.0, 13.5, 16.0], some(&[2.0, 1.0, 0.0])),
        ],
    );
    cfg.t_start = 10.0;
    cfg.gamma = 0.05;
    cfg.gamma_slope = 0.4;
    assert_unbiased(&cfg, NP, REPS);
}

/// A prevalence reading AT `t_start`: the first window has zero substeps and
/// scores the initial state.
#[test]
fn observation_at_t_start() {
    let mut cfg = base(
        "prevalence at t_start (init law)",
        vec![prev("p", POIS, vec![0.0, 1.0, 2.0, 3.0], some(&[3.0, 3.0, 2.0, 2.0]))],
    );
    cfg.init_binomial_p = Some(0.35);
    assert_unbiased(&cfg, NP, REPS);
}

/// An initial-state law (`I ~ binomial(N0, p)`, `S = N0 − I`), scored from
/// the first observation on.
#[test]
fn initial_state_law() {
    let mut cfg = base(
        "init law I ~ binomial(6, 0.35)",
        vec![
            prev("p", POIS, daily(4), some(&[3.0, 2.0, 2.0, 1.0])),
            inc("c", vec![INF], POIS, TTimes::Labels(daily(4)), some(&[1.0, 1.0, 0.0, 0.0])),
        ],
    );
    cfg.init_binomial_p = Some(0.35);
    assert_unbiased(&cfg, NP, REPS);
}

/// IC-free inference: the first observation conditions but is not scored, so
/// the estimate is of `log p(y_{2:T} | y_1)`.
#[test]
fn ic_free_first_observation_skipped() {
    let mut cfg = base(
        "ic-free (skip first), init law",
        vec![prev("p", POIS, daily(4), some(&[3.0, 2.0, 2.0, 1.0]))],
    );
    cfg.init_binomial_p = Some(0.35);
    cfg.skip_first = true;
    // The skipped term makes `exp(ll)` a RATIO of two unbiased estimates, which
    // is not itself unbiased. Its bias is O(1/N), so the check runs at a
    // particle count where that is well below the test's resolution.
    assert_unbiased(&cfg, 2000, 400);

    // A leading hole: the first OBSERVED row is the second union index, and
    // that is the one left out (gh#833), not index 0.
    let mut cfg = base(
        "ic-free (skip first) behind a leading hole",
        vec![prev("p", POIS, daily(5), vec![None, Some(3.0), Some(2.0), Some(2.0), Some(1.0)])],
    );
    cfg.init_binomial_p = Some(0.35);
    cfg.skip_first = true;
    assert_unbiased(&cfg, 2000, 400);
}

/// A scheduled intervention inside a window and one coincident with an
/// observation: it must fire post-advance at its time, and the reading at that
/// time must see the post-intervention state.
#[test]
fn scheduled_intervention_inside_and_at_observation() {
    let mut cfg = base(
        "intervention at t=2 (inside a window) and t=4 (at an obs)",
        vec![
            TStream {
                name: "s",
                proj: Proj::Prev(vec![S]),
                fam: POIS,
                times: TTimes::Labels(vec![1.0, 4.0, 5.0]),
                cells: some(&[3.0, 0.0, 0.0]),
                tested: None,
            },
            inc("c", vec![INF], POIS, TTimes::Labels(vec![1.0, 4.0, 5.0]), some(&[1.0, 1.0, 0.0])),
        ],
    );
    cfg.effects = vec![
        Effect { ts: &[2.0], act: Act::SToR(1.0), event: false },
        Effect { ts: &[4.0], act: Act::SToR(2.0), event: false },
    ];
    assert_unbiased(&cfg, NP, REPS);
}

/// An `events {}` entry whose time is OFF the dt grid fires on the step
/// nearest it (gh#957): `add(I, 2)` at 2.4 with dt = 1 acts at the end of the
/// step ending at 2, in the filter as in forward chain-binomial simulation
/// (`forward_chain_binomial_snaps_an_off_grid_event`). Before the effect plan
/// the filter landed on 2.4 itself, scoring a different model (−0.355 nats).
#[test]
fn off_grid_event_fires_where_the_model_says() {
    let cfg = off_grid_event_config();
    assert_unbiased(&cfg, NP, REPS);
    assert_if2_unbiased(&cfg);
}

fn off_grid_event_config() -> Config {
    let mut cfg = base(
        "event add(I, 2) at t=2.4, dt=1",
        vec![
            prev("p", POIS, daily(4), some(&[2.0, 4.0, 3.0, 2.0])),
            inc("c", vec![INF], POIS, TTimes::Labels(daily(4)), some(&[1.0, 1.0, 1.0, 0.0])),
        ],
    );
    cfg.n0 = 8;
    cfg.effects = vec![Effect { ts: &[2.4], act: Act::AddI(2.0), event: true }];
    cfg
}

/// The forward half of the evidence: chain-binomial simulation applies an
/// off-grid event at the nearest grid step. Deterministic (beta = gamma = 0).
#[test]
fn forward_chain_binomial_snaps_an_off_grid_event() {
    use sim::config::{ChainBinomialConfig, SimConfig};
    use sim::simulate::Simulate;
    let mut cfg = base("forward probe", vec![prev("p", POIS, vec![4.0], some(&[1.0]))]);
    cfg.beta = 0.0;
    cfg.gamma = 0.0;
    cfg.effects = vec![Effect { ts: &[2.3], act: Act::AddI(1.0), event: true }];
    let (compiled, params, _) = build(&cfg);
    let traj = sim::ChainBinomialSim
        .run(&compiled, &params, 1, &SimConfig::ChainBinomial(ChainBinomialConfig { t_start: 0.0, t_end: 4.0, dt: 1.0 }))
        .unwrap();
    let i_at = |t: f64| traj.snapshots.iter().find(|s| same(s.t, t)).unwrap().int_state.counts[I];
    assert_eq!((i_at(1.0), i_at(2.0), i_at(3.0)), (2, 3, 3), "the event is applied at the t = 2 step");
}

/// A flow-ratio projection (`recovery / (infection + recovery)` over each
/// 2-day window) scored by `binomial(n = tested, …)` with `tested` a per-row
/// data column, beside a daily prevalence stream — so the ratio stream's own
/// row index differs from the union index, and a row with `tested = 0` scores
/// exactly 0 even where the window had no events.
#[test]
fn flow_ratio_with_a_per_row_denominator_column() {
    let cfg = base(
        "flow ratio, binomial(n = tested), multi-cadence",
        vec![
            TStream {
                name: "frac_rec",
                proj: Proj::Ratio(vec![REC], vec![INF, REC]),
                fam: Fam::BinomialAux { a: 0.8, b: 0.1 },
                times: TTimes::Labels(vec![2.0, 4.0, 6.0]),
                cells: some(&[1.0, 0.0, 2.0]),
                tested: Some(vec![3, 0, 4]),
            },
            prev("p", POIS, daily(6), some(&[3.0, 3.0, 2.0, 2.0, 1.0, 1.0])),
        ],
    );
    assert_unbiased(&cfg, NP, REPS);
}

/// A scheduled intervention at exactly `t_start` acts on the initial state,
/// before the first substep (D1, gh#958), which is what `at [0]` says. Before
/// the effect plan neither the filter nor forward chain-binomial simulation
/// applied it: the filter counted an effect at `t <= t_start` as already fired
/// (+0.85 nats against this oracle).
#[test]
fn intervention_at_t_start_fires() {
    let mut cfg = base(
        "intervention S->R x3 at t_start",
        vec![prev("p", POIS, daily(4), some(&[3.0, 3.0, 2.0, 1.0]))],
    );
    cfg.effects = vec![Effect { ts: &[0.0], act: Act::SToR(3.0), event: false }];
    assert_unbiased(&cfg, NP, REPS);
    assert_if2_unbiased(&cfg);
}

/// A fine step (dt = 0.1) with decimal observation times, where
/// `t_start + s·dt` lands a few ulps either side of each boundary: a sliver
/// substep, a lost substep, or a boundary scored twice would show here.
#[test]
fn fine_dt_with_decimal_observation_times() {
    let mut cfg = base(
        "dt = 0.1, obs at 0.3, 0.7, 1.0, 1.3",
        vec![
            inc("c", vec![INF], POIS, TTimes::Labels(vec![0.3, 0.7, 1.0, 1.3]), some(&[1.0, 1.0, 0.0, 1.0])),
            prev("p", POIS, vec![0.3, 0.7, 1.0, 1.3], some(&[2.0, 3.0, 3.0, 3.0])),
        ],
    );
    cfg.dt = 0.1;
    cfg.beta = 2.0;
    cfg.gamma = 0.8;
    assert_unbiased(&cfg, NP, REPS);
}

/// A non-integer `t_start` with a time-dependent rate, observations off the
/// dt grid, and an on-grid intervention (on-grid relative to `t_start`), plus
/// a second intervention coinciding with a declared period's leading edge —
/// a boundary where nothing is scored.
#[test]
fn shifted_origin_with_interventions_and_a_leading_edge() {
    let mut cfg = base(
        "t_start = 10.5, interventions at 12.5 and 13.5 (a period start)",
        vec![
            inc("c", vec![INF], POIS, TTimes::Periods(vec![(13.5, 14.0), (14.0, 16.5)]), some(&[1.0, 1.0])),
            prev("p", POIS, vec![11.25, 14.0, 16.5], some(&[3.0, 2.0, 1.0])),
        ],
    );
    cfg.t_start = 10.5;
    cfg.gamma = 0.05;
    cfg.gamma_slope = 0.3;
    cfg.effects = vec![
        Effect { ts: &[12.5], act: Act::SToR(1.0), event: false },
        Effect { ts: &[13.5], act: Act::AddI(1.0), event: false },
    ];
    assert_unbiased(&cfg, NP, REPS);
}

/// Asking the filter to record ancestry, prequential scores or predictions
/// must not change the likelihood it returns (those paths draw from separate
/// diagnostic RNG streams).
#[test]
fn recording_flags_do_not_move_the_likelihood() {
    let cfg = base(
        "recording flags",
        vec![
            inc("c", vec![INF], POIS, TTimes::Labels(daily(4)), some(&[2.0, 1.0, 1.0, 0.0])),
            prev("p", POIS, daily(4), some(&[3.0, 2.0, 2.0, 1.0])),
        ],
    );
    let (compiled, params, obs_model) = build(&cfg);
    let process = ChainBinomialProcess::new(compiled.clone());
    let run = |rec: bool| {
        let config = SMCConfig {
            n_particles: 50,
            dt: 1.0,
            t_start: 0.0,
            skip_first_obs_from_loglik: false,
            record_ancestry: rec,
            record_prequential: rec,
            record_predictions: rec,
            max_substeps: sim::inference::degeneracy::ITER_BUDGET,
        };
        bootstrap_filter(&process, &obs_model, &params, &config, 99).unwrap().log_likelihood
    };
    assert_eq!(run(false).to_bits(), run(true).to_bits(), "a recording flag changed the log-likelihood");
}

// ─── IF2 against the same exact marginal ────────────────────────────────────

/// IF2 drives its own filter loop (`run_if2`), not `bootstrap_filter`, so it
/// gets its own harness against the same oracle. With every perturbation
/// zero (one estimated parameter, `rw_sd = 0`, which no rate reads through a
/// perturbation since it never moves) each IF2 iteration is a bootstrap
/// filter at θ on its own RNG streams, and its in-run log-likelihood
/// `if2_perturbed_loglik` is an unbiased estimate of `p(y | θ)` like the PF's.
/// The iterations are the replicates.
fn if2_lls(cfg: &Config, n_particles: usize, reps: usize, seed: u64) -> Vec<f64> {
    use sim::inference::if2::{run_if2, EstimatedParam, IF2Config, Transform};
    let (compiled, params, obs_model) = build(cfg);
    let process = ChainBinomialProcess::new(compiled.clone());
    let beta = compiled.param_index["beta"];
    let if2_params = vec![EstimatedParam {
        name: "beta".into(),
        index: beta,
        initial: params[beta],
        rw_sd: 0.0,
        transform: Transform::None,
        lower: 0.0,
        upper: 100.0,
        rw_sd_auto: false,
        perturb_only_at_t0: false,
    }];
    let config = IF2Config {
        n_particles,
        n_iterations: reps,
        cooling_fraction: 0.5,
        cooling_target_iters: 50,
        dt: cfg.dt,
        t_start: cfg.t_start,
        simplex_groups: vec![],
        skip_first_obs_from_loglik: cfg.skip_first,
        max_substeps: sim::inference::degeneracy::ITER_BUDGET,
    };
    let r = run_if2(&process, &obs_model, &params, &if2_params, &config, seed)
        .unwrap_or_else(|e| panic!("{}: run_if2 failed: {e}", cfg.label));
    r.iterations.iter().map(|it| it.if2_perturbed_loglik).collect()
}

fn assert_if2_unbiased(cfg: &Config) {
    let (exact, _) = oracle(cfg);
    let w: Vec<f64> = if2_lls(cfg, NP, REPS, 0x1f2_5eed).iter().map(|ll| (ll - exact).exp()).collect();
    let n = w.len() as f64;
    let mean = w.iter().sum::<f64>() / n;
    let sd = (w.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (n - 1.0)).sqrt();
    let z = (mean - 1.0) / (sd / n.sqrt());
    eprintln!(
        "[{} / IF2] exact log p(y|θ) = {exact:.6}; log-mean-exp = {:.6}; mean(w) = {mean:.4}, \
         sd(w) = {sd:.3}, z = {z:+.2}",
        cfg.label,
        exact + mean.ln()
    );
    assert!(
        z.abs() < Z_CRIT,
        "{}: the IF2 filter likelihood is biased against the exact marginal: z = {z:+.2}",
        cfg.label
    );
}

// ─── Effect timing on the filter's grid (proposal 2026-10-04 §6.1) ──────────

/// An off-grid scheduled INTERVENTION is snapped like an event: `S -> R` of 2
/// at 2.6 with dt = 1 acts at the end of the step ending at 3. (Before the
/// plan the filter refused an off-grid intervention outright.)
#[test]
fn off_grid_intervention_is_snapped() {
    let mut cfg = base(
        "intervention S->R x2 at t=2.6, dt=1",
        vec![
            prev("p", POIS, daily(4), some(&[2.0, 3.0, 3.0, 2.0])),
            inc("c", vec![INF], POIS, TTimes::Labels(daily(4)), some(&[1.0, 1.0, 0.0, 0.0])),
        ],
    );
    cfg.effects = vec![Effect { ts: &[2.6], act: Act::SToR(2.0), event: false }];
    assert_unbiased(&cfg, NP, REPS);
    assert_if2_unbiased(&cfg);
}

/// An event at `t_start` with an observation AT `t_start`: the observation
/// scores the post-effect state (D1).
#[test]
fn event_at_t_start_with_an_observation_at_t_start() {
    let mut cfg = base(
        "event add(I, 2) at t_start, observed at t_start",
        vec![prev("p", POIS, vec![0.0, 1.0, 2.0, 3.0], some(&[4.0, 3.0, 3.0, 2.0]))],
    );
    cfg.effects = vec![Effect { ts: &[0.0], act: Act::AddI(2.0), event: true }];
    assert_unbiased(&cfg, NP, REPS);
    assert_if2_unbiased(&cfg);
}

/// `t_start = 0.5`, `dt = 1`, effects at 1 and 2.4: they act at 1.5 and 2.5,
/// on the run's own grid. Times on that grid (1.5, 2.5) would not test the
/// key: there the old absolute key `round(t/dt)` and the relative key agree.
#[test]
fn offset_origin_effects_fire_on_the_runs_grid() {
    let mut cfg = base(
        "t_start = 0.5, event at 1, intervention at 2.4",
        vec![
            prev("p", POIS, vec![1.5, 2.5, 3.5], some(&[3.0, 4.0, 2.0])),
            inc("c", vec![REC], POIS, TTimes::Labels(vec![1.5, 2.5, 3.5]), some(&[1.0, 1.0, 1.0])),
        ],
    );
    cfg.t_start = 0.5;
    cfg.effects = vec![
        Effect { ts: &[1.0], act: Act::AddI(1.0), event: true },
        Effect { ts: &[2.4], act: Act::SToR(2.0), event: false },
    ];
    assert_unbiased(&cfg, NP, REPS);
    assert_if2_unbiased(&cfg);
}

/// `at [2.3, 2.4]` on one intervention keys both times to the step ending at
/// 2, so it acts once there (the filter used to split them into two fires).
#[test]
fn two_times_in_one_step_fire_once() {
    let mut cfg = base(
        "intervention S->R x1 at [2.3, 2.4], dt=1",
        vec![prev("p", POIS, daily(4), some(&[2.0, 3.0, 3.0, 2.0]))],
    );
    cfg.effects = vec![Effect { ts: &[2.3, 2.4], act: Act::SToR(1.0), event: false }];
    assert_unbiased(&cfg, NP, REPS);
    assert_if2_unbiased(&cfg);
}

/// An effect in the model's shortened last step: `to = 4.5`, dt = 1, an
/// intervention at 4.3 acts at 4.5 (nearer than 4), the step's end, where the
/// last reading is scored. A reading at 4.4 before it does not see it.
#[test]
fn effect_in_the_models_shortened_last_step() {
    let mut cfg = base(
        "intervention S->R x2 at 4.3, to = 4.5",
        vec![prev("p", POIS, vec![1.0, 2.0, 3.0, 4.0, 4.4, 4.5], some(&[2.0, 3.0, 3.0, 2.0, 2.0, 1.0]))],
    );
    cfg.t_end = 4.5;
    cfg.effects = vec![Effect { ts: &[4.3], act: Act::SToR(2.0), event: false }];
    assert_unbiased(&cfg, NP, REPS);
    assert_if2_unbiased(&cfg);
}
