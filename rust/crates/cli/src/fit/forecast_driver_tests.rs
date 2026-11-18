//! The driver's red tests (migration plan, Stage 5). The model is the
//! committed fixture `cli/tests/fixtures/forecast/two_cadence_sir.camdl`, a
//! chain-binomial SIR with a daily and a weekly incidence stream.

use super::*;
use crate::fit::forecast_plan::{DesignInput, PlanStream, Target};
use crate::fit::observed_rows::ObservedRow;
use crate::fit::training_partition::{TrainingPartition, WholeFiles};
use sim::inference::bootstrap_filter;
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

fn fixture_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/forecast")
}

/// Compile `src` with the camdlc this checkout built.
fn compile(src: &str, tag: &str) -> ir::Model {
    let camdlc = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../../ocaml/_build/default/bin/camdlc.exe");
    assert!(camdlc.exists(), "camdlc missing at {}: run `make build-ocaml`", camdlc.display());
    let dir = crate::test_support::unique_temp_dir(&format!("forecast_{tag}"));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("model.camdl");
    std::fs::write(&path, src).unwrap();
    let out = std::process::Command::new(&camdlc).arg(&path).output().unwrap();
    assert!(out.status.success(), "camdlc: {}", String::from_utf8_lossy(&out.stderr));
    let ir_path = dir.join("model.ir.json");
    std::fs::write(&ir_path, &out.stdout).unwrap();
    crate::util::load_model(ir_path.to_str().unwrap()).unwrap().0
}

pub(super) fn fixture_source() -> String {
    std::fs::read_to_string(fixture_dir().join("two_cadence_sir.camdl")).unwrap()
}

const TRUTH: [(&str, f64); 6] =
    [("beta", 0.5), ("gamma", 0.25), ("rho", 0.5), ("k", 10.0), ("N0", 1000.0), ("I0", 5.0)];

fn with_truth(mut model: ir::Model) -> ir::Model {
    for p in &mut model.parameters {
        if let Some((_, v)) = TRUTH.iter().find(|(n, _)| *n == p.name) {
            p.value = p.value.with_value(*v);
        }
    }
    model
}

/// The first day the daily stream reports in these tests. Fewer conditioning
/// rows keep the importance-sampling reference's effective sample size
/// usable; the week under way at the origin still begins before them.
const FIRST_DAY: usize = 20;

/// The fixture's daily counts for days `FIRST_DAY`–`through` (row `d` covers
/// `[d − 1, d)`), and a weekly stream with `NA` at 7, 14 and 21: the state is
/// conditioned on the daily data alone, as in the importance-sampling
/// reference.
fn daily_through(through: usize) -> (String, String) {
    let all = std::fs::read_to_string(fixture_dir().join("daily_cases.tsv")).unwrap();
    let mut lines = all.lines();
    let header = lines.next().unwrap();
    let daily: String = std::iter::once(format!("{header}\n"))
        .chain(lines.skip(FIRST_DAY - 1).take(through + 1 - FIRST_DAY).map(|l| format!("{l}\n")))
        .collect();
    let weekly = "time\tweekly_cases\n7\tNA\n14\tNA\n21\tNA\n".to_string();
    (daily, weekly)
}

/// The daily counts by step: `Some(y)` at step `d − 1`, the step `[d − 1, d)`
/// that row `d` covers.
fn daily_values(through: usize) -> Vec<Option<f64>> {
    let (d, _) = daily_through(through);
    let mut out = vec![None; through];
    for l in d.lines().skip(1) {
        let mut c = l.split('\t');
        let day: usize = c.next().unwrap().parse().unwrap();
        out[day - 1] = Some(c.next().unwrap().parse().unwrap());
    }
    out
}

struct Setup {
    model: ir::Model,
    compiled: Arc<sim::CompiledModel>,
    streams: Vec<ObsStream>,
    rows: BTreeMap<String, Vec<ObservedRow>>,
}

fn setup(model: ir::Model, daily: &str, weekly: &str, tag: &str) -> Result<Setup, String> {
    let compiled = Arc::new(sim::CompiledModel::new(model.clone()).map_err(|e| format!("{e:?}"))?);
    let dir = crate::test_support::unique_temp_dir(&format!("forecast_data_{tag}"));
    std::fs::create_dir_all(&dir).unwrap();
    let (d, w) = (dir.join("daily.tsv"), dir.join("weekly.tsv"));
    std::fs::write(&d, daily).unwrap();
    std::fs::write(&w, weekly).unwrap();
    let effective: indexmap::IndexMap<String, String> = [
        ("daily_cases".to_string(), d.to_string_lossy().into_owned()),
        ("weekly_cases".to_string(), w.to_string_lossy().into_owned()),
    ].into_iter().collect();
    let time_opts = crate::caltime_load::TimeOpts {
        origin: None, time_unit: "days", dt: 1.0, t_start: 0.0,
        format: crate::caltime_load::TimeFormat::Auto,
    };
    let streams = crate::fit::runner::resolve_and_load_obs_streams(
        &model, &compiled, &effective, &time_opts)?;
    let rows = streams.iter().map(|s| (s.name.clone(), s.rows.clone())).collect();
    Ok(Setup { model, compiled, streams, rows })
}

impl Setup {
    /// A plan whose targets are on the streams named in `with_targets`; every
    /// stream conditions the state either way.
    fn plan(&self, horizon: f64, with_targets: &[&str]) -> ForecastPlan {
        let partition = TrainingPartition::whole(&self.rows, WholeFiles::NoHoldout).unwrap();
        let streams: Vec<PlanStream> = self.model.observations.iter()
            .filter(|o| with_targets.contains(&o.name.as_str()))
            .map(|o| PlanStream { obs: o, training: &self.rows[&o.name], held_out: &[], design: DesignInput::NotSupplied })
            .collect();
        ForecastPlan::new(partition, horizon, &streams).unwrap()
    }

    fn params(&self) -> Vec<f64> {
        let mut p = self.compiled.default_params.clone();
        for (n, v) in TRUTH {
            p[self.compiled.param_index[n]] = v;
        }
        p
    }

    /// The compiled model with its run extent raised to the plan's (D49).
    fn extended(&self, plan: &ForecastPlan) -> Arc<sim::CompiledModel> {
        let mut m = self.model.clone();
        extend_run(&mut m, plan);
        Arc::new(sim::CompiledModel::new(m).unwrap())
    }

    fn run(&self, plan: ForecastPlan, n_draws: usize, n_particles: usize, seed: u64)
        -> Result<Forecast, ForecastFailure>
    {
        let compiled = self.extended(&plan);
        self.run_on(compiled, plan, n_draws, n_particles, seed)
    }

    fn run_on(&self, compiled: Arc<sim::CompiledModel>, plan: ForecastPlan, n_draws: usize,
              n_particles: usize, seed: u64) -> Result<Forecast, ForecastFailure>
    {
        let params = vec![self.params(); n_draws];
        fixed_origin_forecast(plan.clone(), &self.streams, &HeldOutValues::none(&plan), compiled, &params, DriverConfig {
            n_particles, dt: 1.0, base_seed: seed, record_expected: true, trace_pre_origin: true,
            binomial: sim::rng::BinomialAlgorithm::Btpe,
        })
    }
}

fn index_of(plan: &ForecastPlan, stream: &str, end: f64) -> usize {
    plan.targets().iter()
        .position(|t: &Target| t.stream == stream && (t.measures.end() - end).abs() < 1e-9)
        .unwrap_or_else(|| panic!("no target of {stream} ending at {end}"))
}

fn mean_var(xs: &[f64]) -> (f64, f64) {
    let n = xs.len() as f64;
    let m = xs.iter().sum::<f64>() / n;
    (m, xs.iter().map(|x| (x - m).powi(2)).sum::<f64>() / n)
}

/// A pooled statistic and its Monte Carlo error, taking each draw's particles
/// (correlated through resampling) as one unit.
fn pooled_with_se(xs: &[f64], n_draws: usize, f: impl Fn(&[f64]) -> f64) -> (f64, f64) {
    let per = xs.len() / n_draws;
    let stats: Vec<f64> = xs.chunks(per).map(&f).collect();
    let (m, v) = mean_var(&stats);
    (m, (v / (n_draws as f64 - 1.0)).sqrt())
}

/// The importance-sampling reference of §4.4: prior trajectories of the
/// chain-binomial process at `dt = 1`, weighted by the Poisson likelihood of
/// the daily counts (computed here by hand), and the weighted mean and
/// variance of the infections over days `[a, b)`, each with its Monte Carlo
/// error, and the effective sample size.
pub(super) fn is_reference(compiled: &Arc<sim::CompiledModel>, params: &[f64], daily: &[Option<f64>],
                           a: usize, b: usize, m: usize) -> ((f64, f64), (f64, f64), f64)
{
    use sim::inference::traits::ProcessModel;
    use sim::inference::ParticleState;
    let process = ChainBinomialProcess::new(compiled.clone());
    let rho = params[compiled.param_index["rho"]];
    let inf = compiled.model.transitions.iter().position(|t| t.name == "infection").unwrap();
    let (n_int, n_tr) = (process.n_compartments(), process.n_transitions());
    let mut scratch = process.new_scratch();
    let ln_fact = |y: f64| (1..=(y as u64)).map(|k| (k as f64).ln()).sum::<f64>();
    let mut logw = Vec::with_capacity(m);
    let mut xs = Vec::with_capacity(m);
    for i in 0..m {
        let mut rng = sim::rng::StatefulRng::new_stream(0xC0FFEE, i as u64);
        let mut st = ParticleState::new(n_int, n_tr, 0);
        let x0 = process.initial_state_draw(params, &mut rng).unwrap();
        st.counts.copy_from_slice(&x0.counts);
        let (mut lw, mut w) = (0.0, 0.0);
        for day in 0..b {
            st.reset_flows();
            process.step(&mut st, params, day as f64, 1.0, None, &mut rng, &mut scratch, &[]).unwrap();
            let f = st.flow_accumulators[inf] as f64;
            if let Some(&Some(y)) = daily.get(day) {
                let lam = rho * f;
                lw += if lam > 0.0 {
                    y * lam.ln() - lam - ln_fact(y)
                } else if y == 0.0 {
                    0.0
                } else {
                    f64::NEG_INFINITY
                };
            }
            if day >= a {
                w += f;
            }
        }
        logw.push(lw);
        xs.push(w);
    }
    let max = logw.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let ws: Vec<f64> = logw.iter().map(|l| (l - max).exp()).collect();
    let sw: f64 = ws.iter().sum();
    let mu = ws.iter().zip(&xs).map(|(w, x)| w * x).sum::<f64>() / sw;
    let var = ws.iter().zip(&xs).map(|(w, x)| w * (x - mu).powi(2)).sum::<f64>() / sw;
    let se_mu = ws.iter().zip(&xs).map(|(w, x)| (w * (x - mu)).powi(2)).sum::<f64>().sqrt() / sw;
    let se_var = ws.iter().zip(&xs)
        .map(|(w, x)| (w * ((x - mu).powi(2) - var)).powi(2)).sum::<f64>().sqrt() / sw;
    let ess = sw * sw / ws.iter().map(|w| w * w).sum::<f64>();
    ((mu, se_mu), (var, se_var), ess)
}

/// The pass threshold for a driver-against-reference comparison, stated
/// before measuring: |z| below it passes, z being the difference over the
/// combined Monte Carlo error.
const Z_PASS: f64 = 3.0;

fn z_of(a: (f64, f64), b: (f64, f64)) -> f64 {
    (a.0 - b.0) / (a.1.powi(2) + b.1.powi(2)).sqrt()
}

fn agree(what: &str, a: (f64, f64), b: (f64, f64), summary: &str) {
    let z = z_of(a, b);
    assert!(z.abs() < Z_PASS, "{what}: driver {:.3} ± {:.3} vs reference {:.3} ± {:.3} (z = {z:.2})\n{summary}",
        a.0, a.1, b.0, b.1);
}

/// The reference comparison at a given precision: `n_draws` independent
/// filters of `n_particles`, against `m` importance-sampling simulations.
/// Returns a one-paragraph summary of every number compared.
fn reference_check(n_draws: usize, n_particles: usize, m: usize) -> String {
    let (daily, weekly) = daily_through(24);
    let s = setup(with_truth(compile(&fixture_source(), "reference")), &daily, &weekly, "reference").unwrap();
    let plan = s.plan(7.0, &["weekly_cases"]);
    let k = index_of(&plan, "weekly_cases", 28.0);
    assert_eq!((plan.targets()[k].measures.start(), plan.targets()[k].horizon), (21.0, 0));
    let fc = s.run(plan, n_draws, n_particles, 11).unwrap();
    let (rho, kd) = (0.5, 10.0);
    let w: Vec<f64> = fc.expected.as_ref().unwrap()[k].iter().map(|e| e / rho).collect();
    let drv_mean = pooled_with_se(&w, n_draws, |c| mean_var(c).0);
    let drv_var = pooled_with_se(&w, n_draws, |c| mean_var(c).1);
    let ((mu, se_mu), (var, se_var), ess) =
        is_reference(&s.compiled, &s.params(), &daily_values(24), 21, 28, m);

    // The observation noise: the reported count's moments.
    let y = &fc.draws[k];
    let y_mean = pooled_with_se(y, n_draws, |c| mean_var(c).0);
    let y_var = pooled_with_se(y, n_draws, |c| mean_var(c).1);
    let want_mean = (rho * mu, rho * se_mu);
    let want_var = (rho * rho * var + rho * mu + rho * rho * (var + mu * mu) / kd,
                    (rho * rho * se_var).hypot(rho * se_mu) * (1.0 + 2.0 * rho / kd));
    let summary = format!(
        "W mean: driver {:.3} ± {:.3}, reference {mu:.3} ± {se_mu:.3}, z = {:.2}; \
         W variance: driver {:.2} ± {:.2}, reference {var:.2} ± {se_var:.2}, z = {:.2}; \
         reported mean: driver {:.3} ± {:.3}, implied {:.3} ± {:.3}, z = {:.2}; \
         reported variance: driver {:.2} ± {:.2}, implied {:.2} ± {:.2}, z = {:.2}; \
         ESS {ess:.0}; {n_draws} draws x {n_particles} particles, {m} simulations",
        drv_mean.0, drv_mean.1, z_of(drv_mean, (mu, se_mu)),
        drv_var.0, drv_var.1, z_of(drv_var, (var, se_var)),
        y_mean.0, y_mean.1, want_mean.0, want_mean.1, z_of(y_mean, want_mean),
        y_var.0, y_var.1, want_var.0, want_var.1, z_of(y_var, want_var));
    assert!(ess > 1000.0, "the reference needs an effective sample size; got {ess}\n{summary}");
    agree("mean of the week's infections", drv_mean, (mu, se_mu), &summary);
    agree("variance of the week's infections", drv_var, (var, se_var), &summary);
    agree("mean of the reported weekly count", y_mean, want_mean, &summary);
    agree("variance of the reported weekly count", y_var, want_var, &summary);
    summary
}

/// The reference distribution (spec §4.4): origin day 24, the weekly target
/// [21, 28) under way. The driver's predictive mean and variance of the
/// week's infections agree with importance sampling from the prior weighted
/// by the daily data; the reported count's draws carry the declared
/// negative-binomial noise on top: mean ρ·E[W], variance
/// ρ²·Var[W] + E[ρW + (ρW)²/k]. Each comparison passes at |z| < `Z_PASS`.
#[test]
fn the_week_under_way_matches_the_importance_sampling_reference() {
    reference_check(40, 1000, 100_000);
}

/// Where [`the_measurement_table`] writes the figures the Stage 5 note
/// quotes. Tests may not print (the print lint), so the measurement is a file.
const MEASUREMENT_FILE: &str = "camdl-stage5-forecast-measurement.txt";

/// The measurement the Stage 5 note records
/// (`docs/dev/notes/2026-11-16-stage5-forecast-plan.md`): the reference
/// comparison at high precision (a combined Monte Carlo error of about 0.15 on
/// the mean) and at the default test's, and what the unconditioned prior and
/// an undeclared week give. Every comparison is asserted as in the default
/// tests; the figures are written to `$TMPDIR/` + [`MEASUREMENT_FILE`].
#[test]
#[ignore = "costly: ~1.5 minutes in release; run with `cargo test --release -p cli --bin camdl \
            the_measurement_table -- --ignored`, then read $TMPDIR/camdl-stage5-forecast-measurement.txt"]
fn the_measurement_table() {
    let report = [
        format!("high precision: {}", reference_check(800, 1000, 1_000_000)),
        format!("default test precision: {}", reference_check(40, 1000, 100_000)),
        format!("what the reference rules out: {}", conditioning_check(50_000)),
    ].join("\n\n") + "\n";
    let path = std::env::temp_dir().join(MEASUREMENT_FILE);
    std::fs::write(&path, report).unwrap();
}

fn fixture_setup(through: usize, tag: &str) -> Setup {
    let (daily, weekly) = daily_through(through);
    setup(with_truth(compile(&fixture_source(), tag)), &daily, &weekly, tag).unwrap()
}

/// Undeclared periods are refused: a reading at a time that ends no declared
/// period of the stream, or at an observed row, is an error.
#[test]
fn a_reading_that_ends_no_declared_period_is_refused() {
    let s = fixture_setup(24, "undeclared");
    let names: Vec<String> = s.streams.iter().map(|x| x.name.clone()).collect();
    let model_of = |specs| {
        let (bound, _) = BoundObs::bind(0.0, specs).unwrap();
        MultiStreamObsModel::new(bound, s.compiled.clone()).unwrap()
    };
    let bare = model_of(crate::fit::runner::stream_specs_from_obs_streams(&s.streams));
    // The week [21, 28) was not declared: nothing of that stream ends at 28.
    let err = declared_read(&bare, &names, "weekly_cases", 28.0).unwrap_err();
    assert!(err.cause.contains("no declared"), "{err}");
    // Day 24 is an observed row, not a period with no value.
    assert!(declared_read(&bare, &names, "daily_cases", 24.0).is_err());
    // Declared through the plan, the same reading is allowed.
    let plan = s.plan(7.0, &["weekly_cases"]);
    let declared = model_of(specs_with_targets(&plan, &s.streams, &HeldOutValues::none(&plan)).unwrap());
    assert!(declared_read(&declared, &names, "weekly_cases", 28.0).is_ok());
}

/// Layout invariance of the latent flow: with matched random streams, the
/// week's flow is bit-identical whether only the week is declared, every day
/// after the origin is declared too, or the reach is longer.
#[test]
fn the_latent_weekly_flow_does_not_depend_on_which_other_targets_are_declared() {
    let s = fixture_setup(24, "layout");
    let weekly_of = |plan: ForecastPlan| {
        let k = index_of(&plan, "weekly_cases", 28.0);
        s.run(plan, 3, 200, 5).unwrap().expected.unwrap()[k].clone()
    };
    let alone = weekly_of(s.plan(7.0, &["weekly_cases"]));
    let with_days = weekly_of(s.plan(7.0, &["weekly_cases", "daily_cases"]));
    let longer = weekly_of(s.plan(14.0, &["weekly_cases", "daily_cases"]));
    assert_eq!(alone, with_days);
    assert_eq!(alone, longer);
    assert!(alone.iter().any(|&w| w > 0.0), "a vacuous comparison of zeros");
}

/// Joint samples (D54): with the origin on a week boundary, the week
/// [21, 28) is exactly the days [21, 22) … [27, 28). Sample `s`'s weekly flow
/// equals the sum of sample `s`'s daily flows, which holds only if each
/// sample's draws across the eight targets come from one particle.
#[test]
fn each_sample_across_targets_is_one_particles_trajectory() {
    let s = fixture_setup(21, "joint");
    let plan = s.plan(7.0, &["weekly_cases", "daily_cases"]);
    assert_eq!(plan.origin(), 21.0);
    let wk = index_of(&plan, "weekly_cases", 28.0);
    let days: Vec<usize> = (22..=28).map(|e| index_of(&plan, "daily_cases", e as f64)).collect();
    let fc = s.run(plan, 2, 300, 9).unwrap();
    let e = fc.expected.unwrap();
    let sum_days = |i: usize| days.iter().map(|&k| e[k][i]).sum::<f64>();
    for i in 0..e[wk].len() {
        assert_eq!(e[wk][i], sum_days(i), "sample {i}");
    }
    // Not vacuous: pairing sample s with sample s + 1 breaks it.
    let n = e[wk].len();
    assert!((0..n).any(|i| e[wk][i] != sum_days((i + 1) % n)));
}

/// Failure policy (D53): a draw whose filter collapses fails the forecast
/// with the draw and the time; nothing is dropped.
#[test]
fn a_draw_whose_filter_collapses_fails_the_forecast() {
    let (daily, weekly) = daily_through(24);
    let daily: String = daily.lines()
        .map(|l| match l.split('\t').next() {
            Some(d @ ("20" | "21" | "22")) => format!("{d}\t90000"),
            _ => l.to_string(),
        })
        .map(|l| l + "\n")
        .collect();
    let s = setup(with_truth(compile(&fixture_source(), "collapse")), &daily, &weekly, "collapse").unwrap();
    let err = s.run(s.plan(7.0, &["weekly_cases"]), 2, 100, 3).unwrap_err();
    assert_eq!(err.draw, Some(0), "{err}");
    assert_eq!(err.time, Some(22.0), "{err}");
    assert!(err.to_string().contains("No draw is dropped"), "{err}");
}

/// Run extent (D49): a model whose run ends at the origin is extended to the
/// latest target, and an intervention scheduled after the original end fires
/// there. Without the extension the filter refuses the target rows.
#[test]
fn the_run_extends_past_its_end_and_a_later_intervention_fires() {
    let src = fixture_source()
        .replace("to = 63 'days", "to = 24 'days")
        .replace("observations {",
            "interventions {\n  cull : transfer(fraction = 1.0, from = I, to = R) at [26]\n}\n\nobservations {");
    let (daily, weekly) = daily_through(24);
    let s = setup(with_truth(compile(&src, "extent")), &daily, &weekly, "extent").unwrap();
    let plan = s.plan(7.0, &["daily_cases"]);
    assert!(s.run_on(s.compiled.clone(), plan.clone(), 1, 100, 1).is_err(),
        "the unextended run must refuse targets past its end");
    let fc = s.run(plan.clone(), 2, 200, 1).unwrap();
    let e = fc.expected.unwrap();
    let total = |end: f64| e[index_of(&plan, "daily_cases", end)].iter().sum::<f64>();
    assert!(total(26.0) > 0.0, "infections before the cull");
    for end in [27.0, 28.0, 31.0] {
        assert_eq!(total(end), 0.0, "the cull at 26 removed every infective; day ending {end}");
    }
}

/// What the reference test rules out, measured on the same model
/// (`docs/dev/notes/2026-11-16-stage5-forecast-plan.md`): the week's flow
/// without the daily data (the prior, where a forecast that ignored the
/// conditioning would centre), and the flow a reading at day 28 with no reset
/// at day 21 would hold (the previous week's flow included). Both differ from
/// the conditioned week by many Monte Carlo errors.
#[test]
fn the_conditioning_and_the_declaration_each_move_the_week_by_many_errors() {
    conditioning_check(50_000);
}

/// The comparison of [`the_conditioning_and_the_declaration_each_move_the_week_by_many_errors`]
/// over `m` reference simulations each; returns its figures.
fn conditioning_check(m: usize) -> String {
    let s = fixture_setup(24, "measure");
    let daily = daily_values(24);
    let ((week, se_week), _, _) = is_reference(&s.compiled, &s.params(), &daily, 21, 28, m);
    let ((prior, se_prior), (prior_var, _), _) =
        is_reference(&s.compiled, &s.params(), &[], 21, 28, m);
    let ((undeclared, se_und), _, _) = is_reference(&s.compiled, &s.params(), &daily, 14, 28, m);
    let measured = format!("week [21,28): conditioned {week:.2} ± {se_week:.2}; prior {prior:.2} ± {se_prior:.2} \
               (sd {:.2}); no reset at 21, [14,28): {undeclared:.2} ± {se_und:.2}", prior_var.sqrt());
    assert!((prior - week).abs() > 10.0 * se_week.hypot(se_prior), "{measured}");
    assert!((undeclared - week).abs() > 10.0 * se_week.hypot(se_und), "{measured}");
    format!("{measured}; {m} simulations each")
}

/// What declaring a target does to the filter over the observed rows: nothing.
/// The driver puts each target's start and end on the filter's axis. Two cases:
///
/// - the target's start is already a stop of the bound data (the week
///   [21, 28) when a daily row covers [20, 21)): the stops before the origin
///   are unchanged;
/// - the start falls between two data stops (daily rows only on days 23 and
///   24, so the stops are 22, 23 and 24, and none at 21): the target adds a
///   stop at 21 before the origin. It holds no observed value, so the filter
///   does not assimilate or resample there (D72), and at `dt = 1` the step
///   grid is the same.
///
/// Either way the observed rows' log-likelihood increments are bit-identical
/// with and without the target.
#[test]
fn a_targets_start_leaves_the_observed_likelihood_unchanged() {
    let filter = |s: &Setup, plan: Option<&ForecastPlan>| {
        let specs = match plan {
            Some(p) => specs_with_targets(p, &s.streams, &HeldOutValues::none(p)).unwrap(),
            None => crate::fit::runner::stream_specs_from_obs_streams(&s.streams),
        };
        let compiled = match plan { Some(p) => s.extended(p), None => s.compiled.clone() };
        let (bound, _) = BoundObs::bind(0.0, specs).unwrap();
        let obs = MultiStreamObsModel::new(bound, compiled.clone()).unwrap();
        let times: Vec<f64> = (0..obs.n_observations()).map(|i| obs.obs_time(i)).collect();
        let process = ChainBinomialProcess::new(compiled);
        let smc = SMCConfig {
            n_particles: 200, dt: 1.0, t_start: 0.0, skip_first_obs_from_loglik: false,
            record_ancestry: false, record_prequential: false, record_predictions: false,
            max_substeps: sim::inference::degeneracy::ITER_BUDGET,
            binomial: sim::rng::BinomialAlgorithm::Btpe,
        };
        let r = bootstrap_filter(&process, &obs, &s.params(), &smc, 17).unwrap();
        // The increments at the observed rows (times at or before 24).
        times.iter().zip(&r.ll_increments)
            .filter(|(t, _)| **t <= 24.0 + 1e-9)
            .map(|(t, ll)| (*t, *ll))
            .collect::<Vec<_>>()
    };
    let observed = |v: &[(f64, f64)]| -> Vec<(f64, f64)> {
        v.iter().copied().filter(|(_, ll)| *ll != 0.0).collect()
    };

    // A start on an existing stop: identical.
    let s = fixture_setup(24, "stop_on");
    let plan = s.plan(7.0, &["weekly_cases"]);
    assert_eq!(filter(&s, None), filter(&s, Some(&plan)));

    // A start between stops: a new stop at 21 before the origin.
    let all = std::fs::read_to_string(fixture_dir().join("daily_cases.tsv")).unwrap();
    let daily: String = all.lines()
        .filter(|l| l.starts_with("time") || l.starts_with("23\t") || l.starts_with("24\t"))
        .map(|l| format!("{l}\n")).collect();
    let weekly = "time\tweekly_cases\n7\tNA\n14\tNA\n".to_string();
    let s = setup(with_truth(compile(&fixture_source(), "stop_between")), &daily, &weekly, "stop_between").unwrap();
    let plan = s.plan(7.0, &["weekly_cases"]);
    let (without, with) = (filter(&s, None), filter(&s, Some(&plan)));
    let has_21 = |v: &[(f64, f64)]| v.iter().any(|(t, _)| (*t - 21.0).abs() < 1e-9);
    assert!(!has_21(&without) && has_21(&with), "the target adds a stop at 21");
    let (a, b) = (observed(&without), observed(&with));
    assert_eq!(a.len(), b.len());
    // The new stop holds no observed value: no resampling draw is spent
    // there (D72), so every later increment is the same too.
    assert_eq!(a, b, "the stop at 21 moved the observed increments");
}

/// gh#747: the driver's filters draw with the sampler `DriverConfig` names,
/// and the forecast records it. BTRS and BTPE draw the same Binomial(n, p)
/// but not the same values from one stream, so the same seed gives different
/// draws under each.
#[test]
fn the_forecast_draws_with_the_sampler_it_is_given() {
    use sim::rng::BinomialAlgorithm::{Btpe, Btrs};
    let s = fixture_setup(24, "sampler");
    let run = |binomial| {
        let plan = s.plan(7.0, &["weekly_cases"]);
        let compiled = s.extended(&plan);
        fixed_origin_forecast(plan.clone(), &s.streams, &HeldOutValues::none(&plan), compiled, &[s.params(), s.params()], DriverConfig {
            n_particles: 100, dt: 1.0, base_seed: 3, record_expected: false, trace_pre_origin: false, binomial,
        }).unwrap()
    };
    let (btpe, btrs, btrs_again) = (run(Btpe), run(Btrs), run(Btrs));
    assert_eq!((btpe.binomial, btrs.binomial), (Btpe, Btrs));
    assert_ne!(btpe.draws, btrs.draws, "the sampler must reach the particles' draws");
    assert_eq!(btrs.draws, btrs_again.draws, "and the draws are reproducible under it");
}

/// Scenario arms (D50): two arms of the same seed whose models differ only
/// by an intervention at day 30, after the origin (day 24), share the filter
/// before the origin bit for bit (every stop's log-likelihood increment and
/// ESS, every particle's log-likelihood and predictive draw) and the
/// particles' states and log-weights at the origin, so the running totals of
/// the week under way agree too: every target ending by day 29 is identical,
/// the week [21, 28) included. After the cull the arms differ. An arm whose
/// intervention fires at day 20, before the origin, departs from the
/// baseline's record before the origin, which is what the forecast's check
/// on its arms reads.
#[test]
fn arms_that_differ_only_after_the_origin_share_the_filter_before_it() {
    let s = fixture_setup(24, "arms");
    let cull_at = |day: u32, fraction: f64| {
        let src = fixture_source().replace("observations {", &format!(
            "interventions {{\n  cull : transfer(fraction = {fraction}, from = I, to = R) at [{day}]\n}}\n\nobservations {{"));
        let (daily, weekly) = daily_through(24);
        setup(with_truth(compile(&src, &format!("arms_{day}"))), &daily, &weekly, &format!("arms_{day}")).unwrap()
    };
    let plan = s.plan(14.0, &["weekly_cases", "daily_cases"]);
    let base = s.run(plan.clone(), 3, 200, 11).unwrap();
    let late = cull_at(30, 1.0).run(plan.clone(), 3, 200, 11).unwrap();
    let early = cull_at(20, 0.2).run(plan.clone(), 3, 200, 11).unwrap();

    for d in 0..3 {
        assert_eq!(base.pre_origin[d].first_difference(&late.pre_origin[d]), None, "draw {d}");
        assert!(!base.pre_origin[d].times.is_empty() && base.pre_origin[d].times.iter().all(|&t| t <= 24.0));
    }
    assert_eq!(base.origin_states, late.origin_states, "states and log-weights at the origin");
    let (be, le) = (base.expected.as_ref().unwrap(), late.expected.as_ref().unwrap());
    let mut compared = 0;
    for (k, t) in plan.targets().iter().enumerate() {
        if t.measures.end() <= 29.0 {
            assert_eq!(base.draws[k], late.draws[k], "{} ending {}", t.stream, t.measures.end());
            assert_eq!(be[k], le[k], "{} ending {}", t.stream, t.measures.end());
            compared += 1;
        }
    }
    assert_eq!(compared, 6, "days 25–29 and the week [21, 28)");
    let after = index_of(&plan, "daily_cases", 31.0);
    assert_ne!(be[after], le[after], "the cull at day 30 changes the arm after it");
    assert!(le[after].iter().all(|&x| x == 0.0), "no infective is left after the cull");

    let departs = (0..3).find_map(|d| base.pre_origin[d].first_difference(&early.pre_origin[d]));
    let (time, what) = departs.expect("a cull at day 20 changes the filter before the origin");
    assert!(time.is_some_and(|t| t <= 24.0), "{what} at {time:?}");

    // The forecast's check on its arms passes the late arm and fails the early one.
    use crate::fit::forecast_arms::ArmLabel;
    use crate::fit::forecast_output::{check_shared_past, DrawRef};
    let refs: Vec<DrawRef> = (0..3).map(|i| DrawRef { cloud_row: i, key: None }).collect();
    let label = |s: &str| ArmLabel { scenario: s.into(), sweep: Vec::new() };
    assert_eq!(check_shared_past(&base, &late, &label("late"), &refs), Ok(()));
    let err = check_shared_past(&base, &early, &label("early"), &refs).unwrap_err();
    assert!(err.contains("the forecast arm 'early' departs from the fitted model before the forecast origin t = 24"),
        "{err}");
}

/// D84 for a single-arm forecast: recording only the stops after the origin
/// and the targets, instead of every stop, leaves the forecast bit-identical
/// (its draws, expected observations, origin states and last-assimilation
/// ESS) and builds no pre-origin trace; with no trace, the check on arms
/// refuses rather than passing.
#[test]
fn a_single_arm_forecast_is_the_same_under_the_narrowed_stop_list() {
    let s = fixture_setup(24, "narrowed");
    let plan = s.plan(14.0, &["weekly_cases", "daily_cases"]);
    let run = |trace_pre_origin| {
        fixed_origin_forecast(plan.clone(), &s.streams, &HeldOutValues::none(&plan), s.extended(&plan), &vec![s.params(); 3], DriverConfig {
            n_particles: 200, dt: 1.0, base_seed: 7, record_expected: true, trace_pre_origin,
            binomial: sim::rng::BinomialAlgorithm::Btpe,
        }).unwrap()
    };
    let (narrow, every) = (run(false), run(true));
    assert_eq!(narrow.draws, every.draws);
    assert_eq!(narrow.expected, every.expected);
    assert_eq!(narrow.origin_states, every.origin_states);
    assert_eq!(narrow.last_assimilation, every.last_assimilation);
    assert!(narrow.draws.iter().flatten().any(|&y| y > 0.0), "a vacuous comparison of zeros");
    assert_eq!((narrow.pre_origin.len(), every.pre_origin.len()), (0, 3));

    use crate::fit::forecast_arms::ArmLabel;
    use crate::fit::forecast_output::{check_shared_past, DrawRef};
    let refs: Vec<DrawRef> = (0..3).map(|i| DrawRef { cloud_row: i, key: None }).collect();
    let label = ArmLabel { scenario: "late".into(), sweep: Vec::new() };
    let err = check_shared_past(&every, &narrow, &label, &refs).unwrap_err();
    assert!(err.contains("recorded no trace before the origin"), "{err}");
}

// ── D79: held-out values and the log scores ──────────────────────────────

use crate::fit::forecast_plan::{HeldOut, RowFile};

/// A held-out row of the holdout file covering `[start, end)` with value `v`.
fn held(position: usize, start: f64, end: f64, v: Option<f64>) -> HeldOut {
    HeldOut {
        file: RowFile::Holdout,
        row: ObservedRow {
            position,
            label: end,
            measures: Measures::Period(Period::new(start, end).unwrap()),
            temporal: vec![crate::caltime_load::TemporalCell::Number(end)],
            cell: v.map(sim::inference::ObsCell::Scalar),
            aux: vec![],
            design: vec![],
        },
    }
}

/// The weekly targets [21, 28) and [28, 35) and the daily days 25 to 28,
/// held out, each value scaled by `f`; day 26 is `NA`.
fn held_out_rows(f: f64) -> BTreeMap<String, Vec<HeldOut>> {
    BTreeMap::from([
        ("weekly_cases".to_string(), vec![held(0, 21.0, 28.0, Some((60.0 * f).round())),
                                          held(1, 28.0, 35.0, Some((45.0 * f).round()))]),
        ("daily_cases".to_string(), vec![held(0, 24.0, 25.0, Some((9.0 * f).round())),
                                         held(1, 25.0, 26.0, None),
                                         held(2, 26.0, 27.0, Some((8.0 * f).round())),
                                         held(3, 27.0, 28.0, Some((7.0 * f).round()))]),
    ])
}

impl Setup {
    /// A plan over both streams with `held` as their held-out rows.
    fn plan_held(&self, horizon: f64, held: &BTreeMap<String, Vec<HeldOut>>) -> ForecastPlan {
        let partition = TrainingPartition::whole(&self.rows, WholeFiles::NoHoldout).unwrap();
        let streams: Vec<PlanStream> = self.model.observations.iter()
            .map(|o| PlanStream { obs: o, training: &self.rows[&o.name], held_out: &held[&o.name], design: DesignInput::NotSupplied })
            .collect();
        ForecastPlan::new(partition, horizon, &streams).unwrap()
    }

    fn run_held(&self, plan: &ForecastPlan, values: &HeldOutValues, seed: u64) -> Forecast {
        fixed_origin_forecast(plan.clone(), &self.streams, values, self.extended(plan), &vec![self.params(); 3],
            DriverConfig {
                n_particles: 200, dt: 1.0, base_seed: seed, record_expected: true, trace_pre_origin: true,
                binomial: sim::rng::BinomialAlgorithm::Btpe,
            }).unwrap()
    }
}

/// D79 at the driver: two sets of held-out values, and none, leave the
/// forecast bit-identical (its predictive draws, the particles' expected
/// observations, the states at the origin, the last assimilation and the
/// pre-origin trace of every draw's filter) and change only the held-out
/// log-likelihoods, which exist exactly at the targets with a value.
#[test]
fn held_out_values_change_only_the_forecasts_held_out_log_likelihoods() {
    let s = fixture_setup(24, "held_out");
    let (rows_a, rows_b) = (held_out_rows(1.0), held_out_rows(1.6));
    let plan = s.plan_held(14.0, &rows_a);
    assert_eq!(plan.targets(), s.plan_held(14.0, &rows_b).targets(), "the targets do not depend on the values");
    let (va, vb) = (HeldOutValues::from_rows(&plan, &rows_a, &s.model).unwrap(),
                    HeldOutValues::from_rows(&plan, &rows_b, &s.model).unwrap());
    for seed in [3, 11] {
        let (a, b, none) = (s.run_held(&plan, &va, seed), s.run_held(&plan, &vb, seed),
                            s.run_held(&plan, &HeldOutValues::none(&plan), seed));
        for other in [&b, &none] {
            assert_eq!(a.draws, other.draws, "seed {seed}: predictive draws");
            assert_eq!(a.expected, other.expected, "seed {seed}: expected observations");
            assert_eq!(a.origin_states, other.origin_states, "seed {seed}: states at the origin");
            assert_eq!(a.last_assimilation, other.last_assimilation);
            assert_eq!(a.pre_origin, other.pre_origin, "seed {seed}: the filter before the origin");
        }
        assert!(a.draws.iter().flatten().any(|&y| y > 0.0), "a vacuous comparison of zeros");
        assert!(none.held_out_log_liks.iter().all(Option::is_none));
        let scored: Vec<bool> = a.held_out_log_liks.iter().map(Option::is_some).collect();
        let want: Vec<bool> = (0..plan.targets().len()).map(|k| va.at(k).is_some()).collect();
        assert_eq!(scored, want, "a held-out log-likelihood exactly where a target has a value");
        assert_eq!(scored.iter().filter(|x| **x).count(), 5, "two weeks and three days (day 26 is NA)");
        for (k, (la, lb)) in a.held_out_log_liks.iter().zip(&b.held_out_log_liks).enumerate() {
            if let (Some(la), Some(lb)) = (la, lb) {
                assert_ne!(la, lb, "seed {seed}: target {k}'s values move its log-likelihoods");
            }
        }
    }
}

/// The held-out log-likelihood is the observation model's density of the
/// held-out value at each particle's state, recomputed here from the
/// particle's expected observation `E[y | x, θ] = ρ·projected` with the
/// families written out: negative binomial (dispersion k) for the weekly
/// stream, Poisson for the daily. The marginal log score is
/// `logsumexp − log(MN)` over all draws' particles, and the joint score of a
/// week and a day sums each sample's two log-likelihoods first; both agree
/// with a direct computation, and the joint is not the sum of the marginals.
#[test]
fn the_held_out_log_likelihoods_and_scores_agree_with_an_independent_computation() {
    use crate::fit::forecast_score::{joint_log_score, marginal_log_scores, LogScoreKind};
    let s = fixture_setup(24, "held_out_oracle");
    let rows = held_out_rows(1.0);
    let plan = s.plan_held(14.0, &rows);
    let values = HeldOutValues::from_rows(&plan, &rows, &s.model).unwrap();
    let fc = s.run_held(&plan, &values, 5);
    let expected = fc.expected.as_ref().unwrap();
    let k_disp = TRUTH.iter().find(|(n, _)| *n == "k").unwrap().1;
    for (k, t) in plan.targets().iter().enumerate() {
        let Some(ll) = &fc.held_out_log_liks[k] else { continue };
        let y = values.at(k).unwrap();
        for (i, (&l, &mean)) in ll.iter().zip(&expected[k]).enumerate() {
            let want = match t.stream.as_str() {
                "weekly_cases" => sim::inference::negbin_logpmf(y, mean, k_disp),
                "daily_cases" => sim::inference::obs_loglik::poisson_logpmf(y, mean),
                other => panic!("{other}"),
            };
            assert!((l - want).abs() <= 1e-12 * want.abs().max(1.0),
                "target {k} ({}, end {}), sample {i}: {l} vs {want}", t.stream, t.measures.end());
        }
    }
    // The scores, against a direct max-shifted computation in f64.
    let direct = |xs: &[f64]| {
        let m = xs.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        m + (xs.iter().map(|x| (x - m).exp()).sum::<f64>() / xs.len() as f64).ln()
    };
    let marginals = marginal_log_scores(&fc).unwrap();
    assert_eq!(marginals.len(), 5);
    for m in &marginals {
        assert_eq!((m.kind, m.n_samples), (LogScoreKind::Marginal, 3 * 200));
        let ll = fc.held_out_log_liks[m.targets[0]].as_ref().unwrap();
        assert!((m.log_score - direct(ll)).abs() < 1e-10, "{} vs {}", m.log_score, direct(ll));
        // Never below the mean of the draws' own log scores (Jensen).
        let per_draw: f64 = ll.chunks(200).map(direct).sum::<f64>() / 3.0;
        assert!(m.log_score >= per_draw - 1e-12);
    }
    let (wk, day) = (index_of(&plan, "weekly_cases", 28.0), index_of(&plan, "daily_cases", 28.0));
    let joint = joint_log_score(&fc, &[day, wk]).unwrap();
    let mut both = vec![wk, day];
    both.sort_unstable();
    assert_eq!((joint.kind.as_str(), joint.targets.clone(), joint.n_samples), ("joint_log_score", both, 600));
    let (lw, ld) = (fc.held_out_log_liks[wk].as_ref().unwrap(), fc.held_out_log_liks[day].as_ref().unwrap());
    let summed: Vec<f64> = lw.iter().zip(ld).map(|(a, b)| a + b).collect();
    assert!((joint.log_score - direct(&summed)).abs() < 1e-10);
    let sum_of_marginals = direct(lw) + direct(ld);
    assert!((joint.log_score - sum_of_marginals).abs() > 1e-6,
        "the joint score is not the sum of the marginals ({} vs {sum_of_marginals})", joint.log_score);
    // A target with no held-out value cannot enter a joint score.
    let na = index_of(&plan, "daily_cases", 26.0);
    assert!(joint_log_score(&fc, &[wk, na]).unwrap_err().contains("no held-out value"));
}

// ── D81: a design-reading stream, and the acceptance test of D79/D81 ─────

/// The committed fixture `survey_sir.camdl`: the two-cadence SIR with a
/// survey, `positives ~ binomial(n = tested, p = I/N)` every 7 days, whose
/// likelihood reads the design column `tested`.
fn survey_source() -> String {
    std::fs::read_to_string(fixture_dir().join("survey_sir.camdl")).unwrap()
}

/// The survey's held-out rows at days 28 and 35, `tested` fixed and the
/// positives given.
fn survey_held(pos: [f64; 2], tested: [f64; 2]) -> Vec<HeldOut> {
    [(28.0, 0), (35.0, 1)].iter().map(|&(t, i)| HeldOut {
        file: RowFile::Holdout,
        row: ObservedRow {
            position: i,
            label: t,
            measures: Measures::Instant(t),
            temporal: vec![crate::caltime_load::TemporalCell::Number(t)],
            cell: Some(sim::inference::ObsCell::Scalar(pos[i])),
            aux: vec![("tested".to_string(), tested[i])],
            design: vec![("tested".to_string(), tested[i])],
        },
    }).collect()
}

/// The survey fixture conditioned on daily counts through day 24 and survey
/// rows on days 7, 14 and 21.
fn survey_setup(tag: &str) -> Setup {
    let model = with_truth(compile(&survey_source(), tag));
    let (daily, weekly) = daily_through(24);
    let compiled = Arc::new(sim::CompiledModel::new(model.clone()).unwrap());
    let dir = crate::test_support::unique_temp_dir(&format!("forecast_{tag}_data"));
    std::fs::create_dir_all(&dir).unwrap();
    let path = |n: &str, body: &str| { let p = dir.join(n); std::fs::write(&p, body).unwrap(); p.to_string_lossy().into_owned() };
    let effective: indexmap::IndexMap<String, String> = [
        ("daily_cases".to_string(), path("daily.tsv", &daily)),
        ("weekly_cases".to_string(), path("weekly.tsv", &weekly)),
        ("survey".to_string(), path("survey.tsv", "time\tpositives\ttested\n7\t1\t40\n14\t5\t40\n21\t8\t40\n")),
    ].into_iter().collect();
    let time_opts = crate::caltime_load::TimeOpts {
        origin: None, time_unit: "days", dt: 1.0, t_start: 0.0, format: crate::caltime_load::TimeFormat::Auto,
    };
    let streams = crate::fit::runner::resolve_and_load_obs_streams(&model, &compiled, &effective, &time_opts).unwrap();
    Setup { rows: streams.iter().map(|x| (x.name.clone(), x.rows.clone())).collect(), model, compiled, streams }
}

impl Setup {
    /// A 14-day plan over every stream, the survey's held-out rows `held`
    /// and its design from `design`.
    fn survey_plan(&self, held: &[HeldOut], design: DesignInput) -> ForecastPlan {
        let partition = TrainingPartition::whole(&self.rows, WholeFiles::NoHoldout).unwrap();
        let ps: Vec<PlanStream> = self.model.observations.iter().map(|o| PlanStream {
            obs: o, training: &self.rows[&o.name],
            held_out: if o.name == "survey" { held } else { &[] },
            design,
        }).collect();
        ForecastPlan::new(partition, 14.0, &ps).unwrap()
    }

    fn survey_values(&self, plan: &ForecastPlan, held: &[HeldOut]) -> HeldOutValues {
        let rows = BTreeMap::from([("survey".to_string(), held.to_vec()),
            ("daily_cases".to_string(), vec![]), ("weekly_cases".to_string(), vec![])]);
        HeldOutValues::from_rows(plan, &rows, &self.model).unwrap()
    }
}

/// A prospective design row of the survey at day `t`.
fn design_row(position: usize, t: f64, tested: f64) -> ObservedRow {
    ObservedRow {
        position, label: t, measures: Measures::Instant(t),
        temporal: vec![crate::caltime_load::TemporalCell::Number(t)],
        cell: None, aux: vec![], design: vec![("tested".to_string(), tested)],
    }
}

/// A prospective forecast is not scored: with `--design` supplying
/// tested = 5 and held-out rows whose positives (6) exceed it, the forecast
/// is written, its predictions are identical whatever the held-out values
/// are, and no held-out value of the design-reading stream is bound; each
/// survey target records why it is not scored.
#[test]
fn a_prospective_forecast_binds_no_held_out_value_of_a_design_reading_stream() {
    let s = survey_setup("survey_prospective");
    let design = vec![design_row(0, 28.0, 5.0), design_row(1, 35.0, 5.0)];
    let run = |pos: [f64; 2]| {
        let held = survey_held(pos, [40.0, 50.0]);
        let plan = s.survey_plan(&held, DesignInput::Prospective(&design));
        let values = s.survey_values(&plan, &held);
        s.run_held(&plan, &values, 17)
    };
    let (a, b) = (run([6.0, 6.0]), run([0.0, 1.0]));
    assert_eq!(a.draws, b.draws, "the prediction does not depend on the held-out values");
    let k = index_of(&a.plan, "survey", 28.0);
    assert!(a.draws[k].iter().all(|&y| y <= 5.0), "the draws read the supplied tested = 5");
    assert!(a.held_out_log_liks[k].is_none(), "no held-out value bound");
    let why = a.not_scored[k].as_deref().unwrap();
    assert!(why.contains("prospective"), "{why}");
}

/// A held-out row the binder would refuse (positives 6 out of tested 5,
/// read ex post from the row itself) refuses that target's score, never
/// the prediction: the forecast is written, identical to one whose held-out
/// value is admissible, and the target records the binder's reason.
#[test]
fn a_held_out_row_the_binder_refuses_refuses_its_score_not_the_prediction() {
    let s = survey_setup("survey_bad_row");
    let run = |pos: [f64; 2]| {
        let held = survey_held(pos, [5.0, 50.0]);
        let plan = s.survey_plan(&held, DesignInput::ExPost);
        let values = s.survey_values(&plan, &held);
        s.run_held(&plan, &values, 19)
    };
    let (bad, good) = (run([6.0, 3.0]), run([2.0, 3.0]));
    assert_eq!(bad.draws, good.draws, "the prediction is written and unchanged");
    let (k28, k35) = (index_of(&bad.plan, "survey", 28.0), index_of(&bad.plan, "survey", 35.0));
    assert!(bad.held_out_log_liks[k28].is_none() && good.held_out_log_liks[k28].is_some());
    let why = bad.not_scored[k28].as_deref().unwrap();
    assert!(why.contains("exceeds"), "the binder's reason: {why}");
    assert!(bad.held_out_log_liks[k35].is_some() && bad.not_scored[k35].is_none(), "the other target is scored");
}

/// The acceptance test (D79, D81): changing the held-out outcomes, with the
/// training data, the design inputs and the targets fixed, leaves the
/// predictive samples and the prediction's address identical and changes
/// only the held-out log-likelihoods and the scores. The design-reading
/// survey is forecast ex post, from the held-out rows' realised `tested`
/// (D81); a change of design, by contrast, moves both the samples and the
/// address.
#[test]
fn held_out_outcomes_change_only_the_scores_and_never_the_samples_or_the_address() {
    use crate::fit::forecast_plan::DesignMode;
    use crate::fit::forecast_score::marginal_log_scores;
    use crate::fit::predict::PredictOptions;
    let model = with_truth(compile(&survey_source(), "survey"));
    let (daily, weekly) = daily_through(24);
    let compiled = Arc::new(sim::CompiledModel::new(model.clone()).unwrap());
    let dir = crate::test_support::unique_temp_dir("forecast_survey_data");
    std::fs::create_dir_all(&dir).unwrap();
    let path = |n: &str, body: &str| { let p = dir.join(n); std::fs::write(&p, body).unwrap(); p.to_string_lossy().into_owned() };
    let effective: indexmap::IndexMap<String, String> = [
        ("daily_cases".to_string(), path("daily.tsv", &daily)),
        ("weekly_cases".to_string(), path("weekly.tsv", &weekly)),
        ("survey".to_string(), path("survey.tsv", "time\tpositives\ttested\n7\t1\t40\n14\t5\t40\n21\t8\t40\n")),
    ].into_iter().collect();
    let time_opts = crate::caltime_load::TimeOpts {
        origin: None, time_unit: "days", dt: 1.0, t_start: 0.0, format: crate::caltime_load::TimeFormat::Auto,
    };
    let streams = crate::fit::runner::resolve_and_load_obs_streams(&model, &compiled, &effective, &time_opts).unwrap();
    let s = Setup { rows: streams.iter().map(|x| (x.name.clone(), x.rows.clone())).collect(), model, compiled, streams };

    let none: Vec<HeldOut> = Vec::new();
    let plan_of = |held: &[HeldOut]| {
        let partition = TrainingPartition::whole(&s.rows, WholeFiles::NoHoldout).unwrap();
        let ps: Vec<PlanStream> = s.model.observations.iter().map(|o| PlanStream {
            obs: o, training: &s.rows[&o.name],
            held_out: if o.name == "survey" { held } else { &none },
            design: DesignInput::ExPost,
        }).collect();
        ForecastPlan::new(partition, 14.0, &ps).unwrap()
    };
    let address = |plan: &ForecastPlan| crate::fit::forecast_output::forecast_options(
        &PredictOptions::defaults(), plan).address_key("run-under-test");
    let forecast = |held: &[HeldOut]| {
        let plan = plan_of(held);
        let rows = BTreeMap::from([("survey".to_string(), held.to_vec()),
            ("daily_cases".to_string(), vec![]), ("weekly_cases".to_string(), vec![])]);
        let values = HeldOutValues::from_rows(&plan, &rows, &s.model).unwrap();
        let fc = s.run_held(&plan, &values, 13);
        (address(&plan), fc)
    };

    let (key_a, a) = forecast(&survey_held([2.0, 1.0], [40.0, 50.0]));
    let (key_b, b) = forecast(&survey_held([9.0, 6.0], [40.0, 50.0]));
    assert_eq!(a.plan.design_mode(), Some(DesignMode::ExPost));
    let k28 = index_of(&a.plan, "survey", 28.0);
    assert_eq!(a.plan.targets()[k28].design, vec![("tested".to_string(), 40.0)]);
    assert_eq!(a.plan.targets(), b.plan.targets(), "the targets, and their design values, are fixed");
    assert_eq!(a.draws, b.draws, "identical predictive samples");
    assert_eq!(a.expected, b.expected);
    assert_eq!(a.origin_states, b.origin_states, "the state conditioning is unchanged");
    assert_eq!(key_a, key_b, "identical predict address");
    assert!(a.draws[k28].iter().any(|&y| y > 0.0) && a.draws[k28].iter().all(|&y| y <= 40.0),
        "the survey's draws read tested = 40");
    assert_ne!(a.held_out_log_liks[k28], b.held_out_log_liks[k28], "the held-out log-likelihoods change");
    let (sa, sb) = (marginal_log_scores(&a).unwrap(), marginal_log_scores(&b).unwrap());
    assert_eq!(sa.len(), 2);
    assert!(sa.iter().zip(&sb).all(|(x, y)| x.targets == y.targets && x.log_score != y.log_score),
        "the scores change: {sa:?} vs {sb:?}");

    // The control: a design value is an input to the prediction.
    let (key_c, c) = forecast(&survey_held([2.0, 1.0], [80.0, 50.0]));
    assert_ne!(a.draws[k28], c.draws[k28], "a design change moves the samples");
    assert_ne!(key_a, key_c, "and the address");
}
