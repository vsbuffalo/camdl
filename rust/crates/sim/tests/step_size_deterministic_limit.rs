//! At a very large population the chain-binomial run follows its own
//! deterministic map, and that map is not the ODE unless `dt` is small against
//! the per-capita exit rates.
//!
//! In one chain-binomial step a member of a compartment leaves with
//! probability `1 − exp(−h·dt)` for a per-capita exit rate `h`, and anyone who
//! enters during the step cannot leave before the next one. The stay in a
//! stage is therefore a whole number of steps with mean `dt/(1 − e^{−h·dt})`,
//! longer than `1/h` by about `h·dt/2`. For an SEIR with short stages (latent 5 days,
//! infectious 4 days, exit rates 0.2 and 0.25 per day) at `dt = 1` each stage
//! is about half a day too long, the generation interval about a day too long,
//! and a seasonally forced system is sensitive to exactly that.
//!
//! Fixture: `tests/fixtures/step_size/seir_seasonal_births.camdl` (R0 = 17,
//! sinusoidal forcing of amplitude 0.04, births and deaths at 1/50 years,
//! N = 3.3 × 10¹⁰, so demographic noise is
//! negligible), simulated for 15 years from near the unforced equilibrium. The
//! ODE (RK4 at dt = 0.1; dt = 1 and dt = 0.01 give the same weekly incidence
//! to four digits) settles to annual epidemics. The comparison is on weekly
//! incidence (the `infection` flow per 7-day output row) from year 4 on.
//!
//! Measured on seeds 1-5 (`camdl simulate … --backend chain_binomial --dt …`),
//! as relative L2 error of weekly incidence against the ODE / largest relative
//! error of a yearly peak / alternation of successive yearly peaks from year 8
//! (mean |log(peak_{y+1}/peak_y)|; the ODE's is 0.030):
//!
//! | dt   | rel. L2       | max peak err  | alternation   |
//! | ---- | ------------- | ------------- | ------------- |
//! | 1    | 1.409 – 1.416 | 2.888 – 2.906 | 1.720 – 1.806 |
//! | 0.5  | 0.349 – 0.354 | 0.391 – 0.396 | 0.260 – 0.265 |
//! | 0.25 | 0.150 – 0.155 | 0.091 – 0.096 | 0.096 – 0.108 |
//! | 0.1  | 0.056 – 0.061 | 0.034 – 0.040 | 0.039 – 0.044 |
//! | 0.05 | 0.027 – 0.032 | 0.014 – 0.020 | 0.028 – 0.033 |
//!
//! So at dt = 1 the run alternates large and small years (a biennial pattern)
//! where the ODE has annual epidemics, and the error then falls roughly in
//! proportion to dt. The seed-to-seed spread is a few thousandths, so the
//! bounds below sit far outside the noise; the small-dt bounds are about 1.5
//! times the measured bias (a change that doubles the O(dt) bias fails), and
//! the dt = 1 bounds sit well under the measured gap (they fail only if the
//! phenomenon disappears, which would itself need explaining).
//!
//! The same file checks the coarse-step warning the forward run records
//! (`sim::chain_binomial::CoarseStep`): at dt = 1 it lists both `I` (dt × rate
//! 0.25) and `E` (0.20), at dt = 0.5 both again (0.125 and 0.100), and it is
//! silent at dt = 0.25 and below. Note that dt = 0.25 does not warn yet is off by 15%: the threshold
//! is a guard against coarse steps, not a certificate of convergence.

use sim::{
    compiled_model::CompiledModel,
    config::{ChainBinomialConfig, OdeConfig, SimConfig},
    run_notes::RunNote,
    simulate::Simulate,
    ChainBinomialSim, OdeSim, Trajectory,
};
use std::path::PathBuf;

const YEAR: f64 = 365.25;

fn load_seir() -> ir::Model {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../../tests/fixtures/step_size/ir/seir_seasonal_births.ir.json");
    let text = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("could not read {path:?}: {e}"));
    ir::from_str(&text).unwrap_or_else(|e| panic!("failed to parse {path:?}: {e}"))
}

/// `(t, weekly incidence)` for every output row: the `infection` flow
/// accumulated since the previous row.
fn weekly_incidence(model: &CompiledModel, traj: &Trajectory) -> Vec<(f64, f64)> {
    let k = model.model.transitions.iter().position(|t| t.name == "infection").expect("infection");
    traj.snapshots.iter().skip(1).map(|s| (s.t, s.flows.value(k))).collect()
}

/// The largest weekly incidence in each model year `y` in `from..to`.
fn yearly_peaks(inc: &[(f64, f64)], from: usize, to: usize) -> Vec<f64> {
    (from..to)
        .map(|y| {
            let (lo, hi) = (y as f64 * YEAR, (y + 1) as f64 * YEAR);
            inc.iter().filter(|(t, _)| *t >= lo && *t < hi).map(|(_, v)| *v).fold(f64::MIN, f64::max)
        })
        .collect()
}

struct Gap {
    rel_l2: f64,
    max_peak_err: f64,
    alternation: f64,
}

fn alternation(peaks: &[f64]) -> f64 {
    peaks.windows(2).map(|w| (w[1] / w[0]).ln().abs()).sum::<f64>() / (peaks.len() - 1) as f64
}

fn gap(run: &[(f64, f64)], ode: &[(f64, f64)]) -> Gap {
    assert_eq!(run.len(), ode.len(), "runs must share the output grid");
    let (mut num, mut den) = (0.0, 0.0);
    for ((t, a), (t_ode, b)) in run.iter().zip(ode) {
        assert!((t - t_ode).abs() < 1e-6, "output times differ: {t} vs {t_ode}");
        if *t >= 4.0 * YEAR {
            num += (a - b) * (a - b);
            den += b * b;
        }
    }
    let (pr, po) = (yearly_peaks(run, 4, 15), yearly_peaks(ode, 4, 15));
    let max_peak_err = pr.iter().zip(&po).map(|(a, b)| (a - b).abs() / b).fold(0.0, f64::max);
    Gap { rel_l2: (num / den).sqrt(), max_peak_err, alternation: alternation(&yearly_peaks(run, 8, 15)) }
}

#[test]
fn chain_binomial_converges_to_the_ode_as_dt_shrinks_and_differs_at_dt_1() {
    let model = load_seir();
    let (t_start, t_end) = (model.simulation.t_start, model.simulation.t_end);
    assert!(t_end >= 15.0 * YEAR, "the fixture must run 15 years");
    let compiled = CompiledModel::new(model).unwrap();
    let params = compiled.default_params.clone();

    let ode_traj = OdeSim
        .run(&compiled, &params, 0, &SimConfig::Ode(OdeConfig { t_start, t_end, dt: 0.1 }))
        .unwrap();
    let ode = weekly_incidence(&compiled, &ode_traj);
    let ode_alt = alternation(&yearly_peaks(&ode, 8, 15));
    println!("ode: alternation {ode_alt:.3}");
    assert!(ode_alt < 0.1, "the ODE reference should have annual epidemics; alternation {ode_alt}");

    // Drain notes other runs in this process may have queued, so the checks
    // below see only this test's. This is the only test in the file that
    // reads the queue.
    let _ = sim::run_notes::take();

    let dts = [1.0, 0.5, 0.25, 0.1, 0.05];
    let mut gaps = Vec::new();
    let mut warned = Vec::new();
    for &dt in &dts {
        let cfg = SimConfig::ChainBinomial(ChainBinomialConfig { t_start, t_end, dt, binomial: sim::rng::BinomialAlgorithm::Btpe });
        let traj = ChainBinomialSim.run(&compiled, &params, 1, &cfg).unwrap();
        let g = gap(&weekly_incidence(&compiled, &traj), &ode);
        println!(
            "dt = {dt}: rel L2 {:.4}, max peak err {:.4}, alternation {:.3}",
            g.rel_l2, g.max_peak_err, g.alternation
        );
        gaps.push(g);
        let notes: Vec<_> = sim::run_notes::take()
            .into_iter()
            .filter_map(|n| match n {
                RunNote::CoarseStep(c) => Some(c),
                _ => None,
            })
            .collect();
        warned.push(notes);
    }

    // A detectable gap at dt = 1: a biennial pattern the ODE does not have.
    let g1 = &gaps[0];
    assert!(g1.rel_l2 > 1.0, "dt = 1 should differ from the ODE: rel L2 {}", g1.rel_l2);
    assert!(g1.max_peak_err > 2.0, "dt = 1 yearly peaks should differ from the ODE: {}", g1.max_peak_err);
    assert!(g1.alternation > 1.0, "dt = 1 should alternate large and small years: {}", g1.alternation);

    // Convergence: the error falls at every halving of the step...
    for (w, d) in gaps.windows(2).zip(dts.windows(2)) {
        assert!(w[1].rel_l2 < w[0].rel_l2,
            "rel L2 should fall from dt = {} ({}) to dt = {} ({})", d[0], w[0].rel_l2, d[1], w[1].rel_l2);
    }
    // ...and at dt = 0.05 the run is close to the ODE, with annual epidemics.
    let g = gaps.last().unwrap();
    assert!(g.rel_l2 < 0.05, "dt = 0.05 should be close to the ODE: rel L2 {}", g.rel_l2);
    assert!(g.max_peak_err < 0.04, "dt = 0.05 yearly peaks should be close to the ODE: {}", g.max_peak_err);
    assert!(g.alternation < 0.1, "dt = 0.05 should have annual epidemics: alternation {}", g.alternation);

    // The coarse-step warning: one note at dt = 1 and at dt = 0.5, each listing
    // the infectious compartment (exit rate gamma + mu = 0.2500548) and the
    // exposed one (sigma + mu = 0.2000548; 0.10003 at dt = 0.5, just over),
    // and none at dt <= 0.25.
    for (&dt, notes) in dts.iter().zip(&warned) {
        let expected: &[(&str, f64)] = if dt >= 0.5 {
            &[("I", 0.2500548), ("E", 0.2000548)]
        } else {
            &[]
        };
        if expected.is_empty() {
            assert!(notes.is_empty(), "dt = {dt}: no coarse-step note expected, got {notes:?}");
            continue;
        }
        assert_eq!(notes.len(), 1, "dt = {dt}: expected one coarse-step note, got {notes:?}");
        let c = &notes[0];
        assert_eq!(c.dt, dt);
        let got: Vec<&str> = c.compartments.iter().map(|x| x.compartment.as_str()).collect();
        let want: Vec<&str> = expected.iter().map(|(n, _)| *n).collect();
        assert_eq!(got, want, "dt = {dt}: compartments over the threshold, largest first");
        for (x, (_, rate)) in c.compartments.iter().zip(expected) {
            assert!((x.exit_rate - rate).abs() < 1e-9, "{}: exit rate {}", x.compartment, x.exit_rate);
            assert!((x.rate_times_dt - rate * dt).abs() < 1e-9);
        }
        let msg = RunNote::CoarseStep(c.clone()).to_string();
        println!("{msg}");
        assert_eq!(msg.lines().count(), 2 + expected.len(), "one line per compartment: {msg}");
        assert!(msg.contains("may not reproduce the intended continuous-time dynamics"), "{msg}");
        assert!(msg.contains(&format!("rerun at dt = {}", dt / 2.0)), "{msg}");
        assert!(msg.contains("post-fit Richardson dt-convergence check"), "{msg}");
    }
}
