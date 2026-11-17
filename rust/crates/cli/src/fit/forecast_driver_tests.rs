//! The driver's red tests (migration plan, Stage 5). The model is the
//! committed fixture `cli/tests/fixtures/forecast/two_cadence_sir.camdl`, a
//! chain-binomial SIR with a daily and a weekly incidence stream.

use super::*;
use crate::fit::forecast_plan::{PlanStream, Target};
use crate::fit::observed_rows::ObservedRow;
use crate::fit::training_partition::{TrainingPartition, WholeFiles};
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
            .map(|o| PlanStream { obs: o, training: &self.rows[&o.name], held_out: &[] })
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
        fixed_origin_forecast(plan, &self.streams, compiled, &params, DriverConfig {
            n_particles, dt: 1.0, base_seed: seed, record_expected: true,
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

fn agree(what: &str, a: (f64, f64), b: (f64, f64)) {
    let z = (a.0 - b.0).abs() / (a.1.powi(2) + b.1.powi(2)).sqrt();
    assert!(z < 4.0, "{what}: driver {:.3} ± {:.3} vs reference {:.3} ± {:.3} (z = {z:.2})",
        a.0, a.1, b.0, b.1);
}

/// The reference distribution (spec §4.4): origin day 24, the weekly target
/// [21, 28) under way. The driver's predictive mean and variance of the
/// week's infections agree with importance sampling from the prior weighted
/// by the daily data; the reported count's draws carry the declared
/// negative-binomial noise on top: mean ρ·E[W], variance
/// ρ²·Var[W] + E[ρW + (ρW)²/k].
#[test]
fn the_week_under_way_matches_the_importance_sampling_reference() {
    let (daily, weekly) = daily_through(24);
    let s = setup(with_truth(compile(&fixture_source(), "reference")), &daily, &weekly, "reference").unwrap();
    let plan = s.plan(7.0, &["weekly_cases"]);
    let k = index_of(&plan, "weekly_cases", 28.0);
    assert_eq!((plan.targets()[k].measures.start(), plan.targets()[k].horizon), (21.0, 0));
    let (n_draws, n_particles) = (40, 1000);
    let fc = s.run(plan, n_draws, n_particles, 11).unwrap();
    let (rho, kd) = (0.5, 10.0);
    let w: Vec<f64> = fc.expected.as_ref().unwrap()[k].iter().map(|e| e / rho).collect();
    let drv_mean = pooled_with_se(&w, n_draws, |c| mean_var(c).0);
    let drv_var = pooled_with_se(&w, n_draws, |c| mean_var(c).1);

    let ((mu, se_mu), (var, se_var), ess) =
        is_reference(&s.compiled, &s.params(), &daily_values(24), 21, 28, 100_000);
    eprintln!("IS reference: mean {mu:.3} ± {se_mu:.3}, var {var:.2} ± {se_var:.2}, ESS {ess:.0}");
    eprintln!("driver: mean {:.3} ± {:.3}, var {:.2} ± {:.2}", drv_mean.0, drv_mean.1, drv_var.0, drv_var.1);
    assert!(ess > 1000.0, "the reference needs an effective sample size; got {ess}");
    agree("mean of the week's infections", drv_mean, (mu, se_mu));
    agree("variance of the week's infections", drv_var, (var, se_var));

    // The observation noise: the reported count's moments.
    let y = &fc.draws[k];
    let y_mean = pooled_with_se(y, n_draws, |c| mean_var(c).0);
    let y_var = pooled_with_se(y, n_draws, |c| mean_var(c).1);
    let want_mean = rho * mu;
    let want_var = rho * rho * var + rho * mu + rho * rho * (var + mu * mu) / kd;
    eprintln!("reported: mean {:.3} ± {:.3} (want {want_mean:.3}), var {:.2} ± {:.2} (want {want_var:.2})",
        y_mean.0, y_mean.1, y_var.0, y_var.1);
    agree("mean of the reported weekly count", y_mean, (want_mean, rho * se_mu));
    let se_want = (rho * rho * se_var).hypot(rho * se_mu) * (1.0 + 2.0 * rho / kd);
    agree("variance of the reported weekly count", y_var, (want_var, se_want));
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
    let declared = model_of(specs_with_targets(&plan, &s.streams).unwrap());
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
    let s = fixture_setup(24, "measure");
    let daily = daily_values(24);
    let ((week, se_week), _, _) = is_reference(&s.compiled, &s.params(), &daily, 21, 28, 50_000);
    let ((prior, se_prior), (prior_var, _), _) =
        is_reference(&s.compiled, &s.params(), &[], 21, 28, 50_000);
    let ((undeclared, se_und), _, _) = is_reference(&s.compiled, &s.params(), &daily, 14, 28, 50_000);
    eprintln!("week [21,28): conditioned {week:.2} ± {se_week:.2}; prior {prior:.2} ± {se_prior:.2} \
               (sd {:.2}); no reset at 21, [14,28): {undeclared:.2} ± {se_und:.2}", prior_var.sqrt());
    assert!((prior - week).abs() > 10.0 * se_week.hypot(se_prior));
    assert!((undeclared - week).abs() > 10.0 * se_week.hypot(se_und));
}
