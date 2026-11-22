//! `camdl predict --horizon DURATION` (`--mode forecast`): the
//! fixed-origin forecast from the end of a fit's training rows (migration
//! plan, Stage 5; `docs/dev/temporal-architecture.md` §2.5, §4.3, §4.4).
//!
//! The fixture is `tests/fixtures/forecast/two_cadence_sir.camdl`, a
//! chain-binomial SIR reported daily and weekly, fitted by PMMH with
//! `holdout_after = 24`: the training rows end at day 24, inside the week
//! [21, 28).

mod fit_store_support;

use std::path::{Path, PathBuf};
use std::process::Command;

fn binary() -> PathBuf {
    let bin = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../target/release/camdl");
    assert!(bin.exists(), "release camdl binary missing: {} — run `make build-rust`", bin.display());
    bin
}

fn fixture(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/forecast").join(name)
}

fn run(dir: &Path, args: &[&str]) -> std::process::Output {
    Command::new(binary()).env_remove("CLICOLOR_FORCE")
        .args(args)
        .current_dir(dir)
        .env("CAMDL_SKIP_VERSION_CHECK", "1")
        .env("CAMDL_STORE", dir.join("results"))
        .output()
        .expect("spawn camdl")
}

fn stderr(o: &std::process::Output) -> String {
    String::from_utf8_lossy(&o.stderr).into_owned()
}

/// A PMMH fit of the fixture in `dir`, training rows through day 24. The
/// weekly file also carries rows past the cutoff, which are held out.
fn fit(dir: &Path, method: &str, extra_data: &str) {
    fit_bound(dir, method, extra_data, "daily_cases = \"daily_cases.tsv\"\nweekly_cases = \"weekly_cases.tsv\"");
}

fn fit_bound(dir: &Path, method: &str, extra_data: &str, bound: &str) {
    copy_fixture(dir);
    fit_config(dir, method, &format!("holdout_after = 24.0\n{extra_data}"), bound, "");
}

fn copy_fixture(dir: &Path) {
    std::fs::copy(fixture("two_cadence_sir.camdl"), dir.join("model.camdl")).unwrap();
    std::fs::copy(fixture("daily_cases.tsv"), dir.join("daily_cases.tsv")).unwrap();
    std::fs::copy(fixture("weekly_cases.tsv"), dir.join("weekly_cases.tsv")).unwrap();
}

/// Write `fit.toml` (`[data]` body `data`, bindings `bound`, then `tail`,
/// e.g. a `[data.holdout]` block) and run the fit.
fn fit_config(dir: &Path, method: &str, data: &str, bound: &str, tail: &str) {
    std::fs::write(dir.join("fit.toml"), format!(r#"store = "results"

[model]
camdl = "model.camdl"

[data]
{data}
[data.observations]
{bound}
{tail}
[estimate]
beta = {{ bounds = [0.2, 1.0], start = 0.5 }}

[fixed]
gamma = 0.25
rho   = 0.5
k     = 10.0
N0    = 1000
I0    = 5

{method}
"#)).unwrap();
    let out = run(dir, &["fit", "run", "fit.toml", "--seed", "1", "--label", "fit"]);
    assert!(out.status.success(), "fit run: {}", stderr(&out));
}

const PMMH: &str = r#"[method]
algorithm = "pmmh"
backend = "chain_binomial"
chains = 2
particles = 100
iterations = 60
burn_in = 20
thin = 1"#;

/// The one forecast stored in `dir`'s store: the prediction folder
/// (`<run>/predict/forecast-<id12>/`) holding a `forecast.json`.
fn forecast_dir(dir: &Path) -> PathBuf {
    let mut found: Vec<PathBuf> = fit_store_support::predictions(&dir.join("results")).into_iter()
        .filter(|d| d.join("forecast.json").is_file())
        .collect();
    assert_eq!(found.len(), 1, "one forecast directory: {found:?}");
    found.remove(0)
}

/// Rows of a TSV as header-keyed maps.
fn rows(path: &Path) -> Vec<std::collections::HashMap<String, String>> {
    let text = std::fs::read_to_string(path).unwrap();
    let mut lines = text.lines();
    let header: Vec<String> = lines.next().unwrap().split('\t').map(String::from).collect();
    lines.map(|l| header.iter().cloned().zip(l.split('\t').map(String::from)).collect()).collect()
}

/// The headline: from the training extent (day 24, inside a week),
/// `--horizon "4 weeks"` forecasts the week under way as weekly horizon 0
/// and daily horizons 0–27; the targets are the declared periods, not the
/// most common gap; nothing after the origin is filtered, and the held-out
/// weekly rows name their targets.
#[test]
fn a_forecast_starts_at_the_training_extent_and_counts_each_streams_own_periods() {
    let tmp = tempfile::tempdir().unwrap();
    let dir = tmp.path();
    fit(dir, PMMH, "");
    let out = run(dir, &["predict", "@fit", "--horizon", "4 weeks", "--n-draws", "8"]);
    assert!(out.status.success(), "predict: {}", stderr(&out));
    let fdir = forecast_dir(dir);

    let manifest: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(fdir.join("forecast.json")).unwrap()).unwrap();
    assert_eq!(manifest["schema"], "camdl.forecast/v4");
    let fit_run = fit_store_support::the_run(&dir.join("results"));
    let identity = fit_store_support::json(&fit_run.join("identity.json"));
    // The engine stamp is spelled as the run's `identity.json` spells it: the
    // build fingerprint the fit's own record names (one binary made both),
    // and no `engine_switches` for a prediction made with none.
    assert_eq!(manifest["engine_version"], identity["ambient"]["build"]);
    assert!(manifest.get("engine_switches").is_none(), "{manifest}");
    assert_eq!(manifest["run_id"], identity["run_id"]);
    assert_eq!((&manifest["samples"]["per_draw"], &manifest["samples"]["of_particles"]),
        (&serde_json::json!(5), &serde_json::json!(100)));
    assert!(manifest["samples"]["description"].as_str().unwrap().contains("use all 100 particles"));
    // The model runs to day 63, past the last target (day 52): not extended.
    assert_eq!((&manifest["run_extent"]["model_end"]["t"], &manifest["run_extent"]["run_end"]["t"],
                &manifest["run_extent"]["extended"]),
        (&serde_json::json!(63.0), &serde_json::json!(63.0), &serde_json::json!(false)));
    // The held-out rows (days 25–63 and the weeks ending 28–63, all in the
    // training files under `holdout_after`) are not described here: forecast
    // artifacts record nothing about held-out rows.
    assert!(manifest.get("held_out").is_none(), "no held-out record in forecast.json");
    assert!(manifest["targets"].as_array().unwrap().iter()
        .all(|t| t.get("held_out").is_none() && t.get("has_held_out_row").is_none()));
    assert_eq!(manifest["mode"], "fixed_origin");
    assert_eq!(manifest["origin"]["t"], 24.0, "the origin is the training extent");
    assert_eq!(manifest["horizon"]["duration"], 28.0);
    assert_eq!(manifest["state_set"]["through"]["t"], 24.0);
    let state_rows: Vec<(String, u64)> = manifest["state_set"]["streams"].as_array().unwrap().iter()
        .map(|s| (s["stream"].as_str().unwrap().to_string(), s["rows"].as_u64().unwrap()))
        .collect();
    assert_eq!(state_rows, vec![("daily_cases".into(), 24), ("weekly_cases".into(), 3)],
        "the state set is the training rows: no row ending after the origin is filtered");

    let weekly = rows(&fdir.join("forecast/weekly_cases.tsv"));
    let spans: Vec<(&str, &str, &str, &str)> = weekly.iter()
        .map(|r| (r["t_start"].as_str(), r["t_stop"].as_str(), r["horizon"].as_str(),
                  r["lead"].as_str()))
        .collect();
    assert_eq!(spans, vec![
        ("21", "28", "0", "4"),
        ("28", "35", "1", "11"),
        ("35", "42", "2", "18"),
        ("42", "49", "3", "25"),
    ], "the week under way is horizon 0; horizons count weeks");
    assert!(weekly.iter().all(|r| !r.keys().any(|k| k.contains("held_out"))), "no held-out column");
    let daily = rows(&fdir.join("forecast/daily_cases.tsv"));
    assert_eq!(daily.len(), 28, "daily horizons 0–27");
    assert_eq!((daily[0]["t_start"].as_str(), daily[0]["t_stop"].as_str()), ("24", "25"));
    assert_eq!(daily[27]["horizon"], "27");
    for r in daily.iter().chain(&weekly) {
        assert_eq!(r["n_samples"], (8 * 100).to_string());
    }

    // Joint samples: one value per target per sample.
    let samples = rows(&fdir.join("forecast-samples.tsv"));
    assert_eq!(samples.len(), 8 * 5 * 32);

    // Each draw is named by its row of the fit's draws.tsv and the
    // (chain, draw) key written there, never by its place in the subsample.
    let draws_tsv = walk(&dir.join("results")).into_iter()
        .find(|p| p.file_name().is_some_and(|n| n == "draws.tsv"))
        .expect("the fit wrote draws.tsv");
    let cloud = rows(&draws_tsv);
    let named: Vec<(u64, u64, u64)> = manifest["draws"].as_array().unwrap().iter()
        .map(|d| (d["cloud_row"].as_u64().unwrap(), d["chain"].as_u64().unwrap(), d["draw"].as_u64().unwrap()))
        .collect();
    assert_eq!(named.iter().map(|d| d.0).collect::<Vec<_>>(), (0..8).map(|i| i * cloud.len() as u64 / 8).collect::<Vec<_>>(),
        "the strided subsample of the cloud");
    for &(row, chain, draw) in &named {
        assert_eq!((cloud[row as usize]["chain"].parse::<u64>().unwrap(), cloud[row as usize]["draw"].parse::<u64>().unwrap()),
            (chain, draw), "draws.tsv row {row}");
    }
    let s0 = &samples[0];
    assert_eq!((s0["cloud_row"].parse::<u64>().unwrap(), s0["chain"].parse::<u64>().unwrap(), s0["draw"].parse::<u64>().unwrap()),
        named[0]);
    let last = &samples[samples.len() - 1];
    assert_eq!(last["cloud_row"].parse::<u64>().unwrap(), named[7].0);
}

/// Flag rules (D41): `--horizon` alone is the forecast, `--mode forecast`
/// needs a reach, and a reach means nothing to another mode.
#[test]
fn the_mode_and_horizon_flags_are_checked() {
    let tmp = tempfile::tempdir().unwrap();
    let out = run(tmp.path(), &["predict", "x", "--mode", "forecast"]);
    assert_eq!(out.status.code(), Some(1), "{}", stderr(&out));
    assert!(stderr(&out).contains("--mode forecast needs --horizon"), "{}", stderr(&out));
    let out = run(tmp.path(), &["predict", "x", "--mode", "one_step", "--horizon", "4 weeks"]);
    assert!(stderr(&out).contains("sets the reach of a forecast"), "{}", stderr(&out));
    // The old spelling of the mode flag is answered with the new one.
    let out = run(tmp.path(), &["predict", "x", "--horizon", "one_step"]);
    assert!(stderr(&out).contains("--mode one_step"), "{}", stderr(&out));
}

/// The capability matrix's refused cells for a fit that exists here: an
/// optimizer (IF2) has no posterior, and a per-chain band has no forecast yet.
#[test]
fn single_estimate_fits_and_per_chain_bands_are_refused_a_forecast() {
    let tmp = tempfile::tempdir().unwrap();
    let dir = tmp.path();
    fit(dir, r#"[method]
algorithm = "if2"
backend = "chain_binomial"
chains = 2
particles = 100
iterations = 3
cooling = 0.7"#, "");
    let out = run(dir, &["predict", "@fit", "--horizon", "2 weeks"]);
    assert!(!out.status.success());
    assert!(stderr(&out).contains("single"), "IF2 is a single estimate: {}", stderr(&out));

    let tmp = tempfile::tempdir().unwrap();
    let dir = tmp.path();
    fit(dir, PMMH, "");
    let out = run(dir, &["predict", "@fit", "--horizon", "2 weeks",
                         "--by-chain"]);
    assert!(!out.status.success());
    assert!(stderr(&out).contains("cannot be combined with --mode forecast"), "{}", stderr(&out));
}

/// Held-out rows in the training file under `holdout_after = 24`, or in a
/// `[data.holdout]` file, leave no trace in the forecast's tables: which
/// targets they measure, which rows they are and their values belong to the
/// score artifact (Stage 6). Either way the week under way is the first
/// target.
#[test]
fn held_out_rows_in_either_file_leave_no_trace_in_the_forecast_tables() {
    let tmp = tempfile::tempdir().unwrap();
    let dir = tmp.path();
    fit(dir, PMMH, "");
    let out = run(dir, &["predict", "@fit", "--horizon", "1 weeks", "--n-draws", "2"]);
    assert!(out.status.success(), "{}", stderr(&out));
    let week = rows(&forecast_dir(dir).join("forecast/weekly_cases.tsv")).remove(0);
    assert_eq!(week["t_start"].as_str(), "21");
    assert!(!week.keys().any(|k| k.contains("held_out")), "{:?}", week.keys());

    let tmp = tempfile::tempdir().unwrap();
    let dir = tmp.path();
    copy_fixture(dir);
    let split = |file: &str, keep: &dyn Fn(f64) -> bool| -> String {
        let text = std::fs::read_to_string(dir.join(file)).unwrap();
        let mut lines = text.lines();
        let header = lines.next().unwrap();
        std::iter::once(header.to_string())
            .chain(lines.filter(|l| keep(l.split('\t').next().unwrap().parse().unwrap())).map(String::from))
            .map(|l| l + "\n").collect()
    };
    let (daily, weekly_train, weekly_hold) = (
        split("daily_cases.tsv", &|t| t <= 24.0),
        split("weekly_cases.tsv", &|t| t <= 24.0),
        split("weekly_cases.tsv", &|t| t > 24.0),
    );
    std::fs::write(dir.join("daily_cases.tsv"), daily).unwrap();
    std::fs::write(dir.join("weekly_cases.tsv"), weekly_train).unwrap();
    std::fs::write(dir.join("weekly_hold.tsv"), weekly_hold).unwrap();
    fit_config(dir, PMMH, "",
        "daily_cases = \"daily_cases.tsv\"\nweekly_cases = \"weekly_cases.tsv\"",
        "\n[data.holdout]\nweekly_cases = \"weekly_hold.tsv\"\n");
    let out = run(dir, &["predict", "@fit", "--horizon", "1 weeks", "--n-draws", "2"]);
    assert!(out.status.success(), "{}", stderr(&out));
    let week = rows(&forecast_dir(dir).join("forecast/weekly_cases.tsv")).remove(0);
    assert_eq!(week["t_start"].as_str(), "21");
    assert!(!week.keys().any(|k| k.contains("held_out")), "{:?}", week.keys());
}

/// Every file under `root`.
fn walk(root: &Path) -> Vec<PathBuf> {
    let mut out = Vec::new();
    for e in std::fs::read_dir(root).into_iter().flatten().flatten() {
        let p = e.path();
        if p.is_dir() {
            out.extend(walk(&p));
        } else {
            out.push(p);
        }
    }
    out
}

/// gh#747: a forecast continues the fit's filter on the binomial sampler the
/// fit ran. One problem fitted twice, the second time with `--binomial btrs`:
/// each run records its sampler, each forecast names its run's, and the two
/// forecasts of the same seed differ.
#[test]
fn a_forecast_uses_the_binomial_sampler_its_fit_ran() {
    let tmp = tempfile::tempdir().unwrap();
    let dir = tmp.path();
    fit(dir, PMMH, "");
    let out = run(dir, &["fit", "run", "fit.toml", "--seed", "1", "--binomial", "btrs"]);
    assert!(out.status.success(), "fit run --binomial btrs: {}", stderr(&out));
    let leaves = fit_store_support::runs(&dir.join("results"));
    assert_eq!(leaves.len(), 2, "{leaves:?}");
    let sampler_of = |leaf: &Path| -> String {
        fit_store_support::run_inputs(leaf)["binomial"].as_str().unwrap().to_string()
    };
    let mut forecasts = Vec::new();
    for leaf in &leaves {
        let out = run(dir, &["predict", leaf.to_str().unwrap(), "--horizon", "1 weeks", "--n-draws", "2"]);
        assert!(out.status.success(), "{}", stderr(&out));
        let fdir = fit_store_support::predictions(leaf).into_iter()
            .find(|d| {
                d.join("forecast.json").is_file()
                    && fit_store_support::json(&d.join("forecast.json"))["run_id"] == fit_store_support::run_id(leaf).as_str()
            })
            .unwrap_or_else(|| panic!("no forecast of {}", leaf.display()));
        let m: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(fdir.join("forecast.json")).unwrap()).unwrap();
        assert_eq!(m["binomial"].as_str().unwrap(), sampler_of(leaf), "{}", leaf.display());
        forecasts.push((sampler_of(leaf), std::fs::read_to_string(fdir.join("forecast-samples.tsv")).unwrap()));
    }
    forecasts.sort();
    assert_eq!((forecasts[0].0.as_str(), forecasts[1].0.as_str()), ("btpe", "btrs"));
    assert_ne!(forecasts[0].1, forecasts[1].1, "the two samplers give different draws");
}

type Sample = std::collections::HashMap<(String, String), f64>;

/// Joint samples on the written draws (D54): each sample in
/// `forecast-samples.tsv` is one particle's trajectory. The fixture gains
/// three streams reported without noise (`binomial(p = 1)`, so a draw is the
/// compartment itself), `S`, `I` and `R`, each observed once, at `t = 0`,
/// where the state is known exactly, so they condition nothing. Per sample,
/// `S + I + R` is the population at every target (one particle across
/// streams), `S` never increases and `R` never decreases across targets (one
/// particle across time). Pairing sample `s`'s `S` with sample `s + 1`'s `I`
/// and `R` breaks the sum, so the check is not vacuous.
#[test]
fn each_written_joint_sample_is_one_particles_trajectory() {
    let tmp = tempfile::tempdir().unwrap();
    let dir = tmp.path();
    copy_fixture(dir);
    let model = std::fs::read_to_string(dir.join("model.camdl")).unwrap();
    let exact: String = ["S", "I", "R"].iter().map(|c| format!(r#"
  {c}_count {{
    columns       {{ time : time, {c}_count : count }}
    projected     = {c}
    emit_schedule = every 7 'days
    {c}_count ~ binomial(n = projected, p = 1.0)
  }}"#)).collect();
    let at = model.rfind('}').unwrap();
    std::fs::write(dir.join("model.camdl"), format!("{}{exact}\n}}\n", &model[..at])).unwrap();
    for (c, v) in [("S", 995), ("I", 5), ("R", 0)] {
        std::fs::write(dir.join(format!("{c}.tsv")), format!("time\t{c}_count\n0\t{v}\n")).unwrap();
    }
    fit_config(dir, PMMH, "holdout_after = 24.0\n",
        "daily_cases = \"daily_cases.tsv\"\nweekly_cases = \"weekly_cases.tsv\"\n\
         S_count = \"S.tsv\"\nI_count = \"I.tsv\"\nR_count = \"R.tsv\"",
        "");
    let out = run(dir, &["predict", "@fit", "--horizon", "4 weeks", "--n-draws", "4"]);
    assert!(out.status.success(), "predict: {}", stderr(&out));

    let mut by_sample: std::collections::BTreeMap<u64, Sample> = std::collections::BTreeMap::new();
    for r in rows(&forecast_dir(dir).join("forecast-samples.tsv")) {
        by_sample.entry(r["sample"].parse().unwrap()).or_default()
            .insert((r["stream"].clone(), r["t_stop"].clone()), r["value"].parse().unwrap());
    }
    assert_eq!(by_sample.len(), 4 * 5, "five joint samples per draw");
    let samples: Vec<&Sample> = by_sample.values().collect();
    let v = |s: &Sample, stream: &str, t: u32| -> f64 {
        *s.get(&(stream.to_string(), t.to_string()))
            .unwrap_or_else(|| panic!("no {stream} target at {t}"))
    };
    let weeks = [28, 35, 42, 49];
    for (i, s) in samples.iter().enumerate() {
        for &t in &weeks {
            assert_eq!(v(s, "S_count", t) + v(s, "I_count", t) + v(s, "R_count", t), 1000.0,
                "sample {i} at t = {t}: S + I + R is the population");
        }
        let series = |c: &str| weeks.iter().map(|&t| v(s, c, t)).collect::<Vec<f64>>();
        assert!(series("S_count").windows(2).all(|w| w[0] >= w[1]), "sample {i}: S increased");
        assert!(series("R_count").windows(2).all(|w| w[0] <= w[1]), "sample {i}: R decreased");
    }
    let n = samples.len();
    let mixed = |i: usize, t: u32| {
        let o = samples[(i + 1) % n];
        v(samples[i], "S_count", t) + v(o, "I_count", t) + v(o, "R_count", t)
    };
    assert!((0..n).any(|i| weeks.iter().any(|&t| mixed(i, t) != 1000.0)),
        "pairing one sample's S with another's I and R must break the sum");
}

/// D86: a forecast from a chain selection carries the same `chain_selection`
/// block as `predictive.json`, with its sentence; the full cloud's carries
/// none.
#[test]
fn a_forecast_from_a_chain_selection_records_it() {
    let manifest = |extra: &[&str]| -> serde_json::Value {
        let tmp = tempfile::tempdir().unwrap();
        let dir = tmp.path();
        fit(dir, PMMH, "");
        let mut args = vec!["predict", "@fit", "--horizon", "1 weeks", "--n-draws", "2"];
        args.extend_from_slice(extra);
        let out = run(dir, &args);
        assert!(out.status.success(), "predict: {}", stderr(&out));
        serde_json::from_str(&std::fs::read_to_string(forecast_dir(dir).join("forecast.json")).unwrap()).unwrap()
    };
    assert!(manifest(&[]).get("chain_selection").is_none(), "the full cloud records no selection");
    let m = manifest(&["--exclude-chains", "1"]);
    let cs = &m["chain_selection"];
    assert_eq!((&cs["excluded"], &cs["kept"], &cs["n_total"]),
        (&serde_json::json!([1]), &serde_json::json!([0]), &serde_json::json!(2)));
    assert_eq!(cs["summary"].as_str().unwrap(),
        "chain(s) 1 left out by the chain selection: the draws are from chain(s) 0 only (1 of 2 \
         chains). Leaving chains out biases the posterior toward the retained chains' mode.");
    assert!(m["draws"].as_array().unwrap().iter().all(|d| d["chain"] == 0), "the draws are chain 0's");
}

/// D49 in the manifest: a model whose run ends at the origin (day 24) is run
/// to the last target (day 38) and `forecast.json` says so.
#[test]
fn a_forecast_past_the_models_end_records_the_extension() {
    let tmp = tempfile::tempdir().unwrap();
    let dir = tmp.path();
    copy_fixture(dir);
    let model = std::fs::read_to_string(dir.join("model.camdl")).unwrap().replace("to = 63 'days", "to = 24 'days");
    std::fs::write(dir.join("model.camdl"), model).unwrap();
    fit_config(dir, PMMH, "holdout_after = 24.0\n",
        "daily_cases = \"daily_cases.tsv\"\nweekly_cases = \"weekly_cases.tsv\"", "");
    let out = run(dir, &["predict", "@fit", "--horizon", "2 weeks", "--n-draws", "2"]);
    assert!(out.status.success(), "predict: {}", stderr(&out));
    let m: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(forecast_dir(dir).join("forecast.json")).unwrap()).unwrap();
    assert_eq!((&m["run_extent"]["model_end"]["t"], &m["run_extent"]["run_end"]["t"], &m["run_extent"]["extended"]),
        (&serde_json::json!(24.0), &serde_json::json!(38.0), &serde_json::json!(true)));
}

/// The fixture with a cull of the infectives at day 30 (`late`) or day 20
/// (`early`), each a scenario, the cull's fraction a fixed parameter read only
/// by the culls, and a scenario `boost` that raises `beta`. Fitted with
/// `holdout_after = 24`.
fn fit_with_scenarios(dir: &Path) {
    copy_fixture(dir);
    let model = std::fs::read_to_string(dir.join("model.camdl")).unwrap()
        .replace("  I0    : count\n}", "  I0    : count\n  cull_frac : probability\n}")
        .replace("simulate {", "interventions {
  late_cull  : transfer(fraction = cull_frac, from = I, to = R) at [30]
  early_cull : transfer(fraction = cull_frac, from = I, to = R) at [20]
}

scenarios {
  late  { enable = [late_cull] }
  early { enable = [early_cull] }
  boost { set = { beta = 0.9 } }
}

simulate {");
    std::fs::write(dir.join("model.camdl"), model).unwrap();
    // `[fixed]` is the table open where the method block begins.
    fit_config(dir, &format!("cull_frac = 1.0\n\n{PMMH}"), "holdout_after = 24.0\n",
        "daily_cases = \"daily_cases.tsv\"\nweekly_cases = \"weekly_cases.tsv\"", "");
}

/// Scenario arms (D50). A scenario whose cull fires at day 30, after the
/// origin (day 24), is forecast as a second arm: its rows match the fitted
/// model's for every target ending by day 29 and hold no infections after the
/// cull, and `forecast.json` names both arms and when `late` first acts. A
/// sweep over the cull's fraction, read only by the culls, is a set of arms
/// too. A scenario whose cull fires at day 20, one that changes `beta`, and a
/// sweep over `rho` (read by the observation models from the start) are
/// refused before anything is filtered.
#[test]
fn scenario_arms_that_act_after_the_origin_are_forecast_and_earlier_ones_refused() {
    let tmp = tempfile::tempdir().unwrap();
    let dir = tmp.path();
    fit_with_scenarios(dir);
    let predict = |extra: &[&str]| {
        let mut args = vec!["predict", "@fit", "--horizon", "2 weeks", "--n-draws", "4"];
        args.extend_from_slice(extra);
        run(dir, &args)
    };

    let out = predict(&["--scenario", "late"]);
    assert!(out.status.success(), "predict: {}", stderr(&out));
    let fdir = forecast_dir(dir);
    let daily = rows(&fdir.join("forecast/daily_cases.tsv"));
    let arm = |name: &str, t: &str| daily.iter()
        .find(|r| r["scenario"] == name && r["t_stop"] == t)
        .unwrap_or_else(|| panic!("no {name} row ending {t}"));
    for t in 25..=29 {
        let (f, l) = (arm("fitted", &t.to_string()), arm("late", &t.to_string()));
        for c in ["mean", "q05", "q50", "q95"] {
            assert_eq!(f[c], l[c], "{c} of the day ending {t}");
        }
    }
    for t in 31..=38 {
        assert_eq!(arm("late", &t.to_string())["mean"], "0", "no infective after the cull, day ending {t}");
    }
    assert_ne!(arm("fitted", "31")["mean"], "0");
    let samples = rows(&fdir.join("forecast-samples.tsv"));
    assert_eq!(samples.len(), 2 * 4 * 5 * (14 + 2), "two arms × 4 draws × 5 samples × 16 targets");
    let m: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(fdir.join("forecast.json")).unwrap()).unwrap();
    let arms = m["arms"].as_array().unwrap();
    assert_eq!(arms.len(), 2);
    assert_eq!((&arms[0]["scenario"], &arms[0]["baseline"]), (&serde_json::json!("fitted"), &serde_json::json!(true)));
    assert_eq!((&arms[1]["scenario"], &arms[1]["first_difference"]["t"], &arms[1]["interventions_changed"]),
        (&serde_json::json!("late"), &serde_json::json!(30.0), &serde_json::json!(["late_cull"])));

    let out = predict(&["--scenario", "early"]);
    assert!(!out.status.success());
    assert!(stderr(&out).contains("switches intervention 'early_cull' on or off, which first fires at t = 20, \
        before the forecast origin t = 24"), "{}", stderr(&out));
    let out = predict(&["--scenario", "boost"]);
    assert!(!out.status.success());
    assert!(stderr(&out).contains("changes parameter 'beta', which the model reads outside any intervention"),
        "{}", stderr(&out));
    let out = predict(&["--sweep", "rho=0.4,0.6"]);
    assert!(!out.status.success());
    assert!(stderr(&out).contains("changes parameter 'rho'"), "{}", stderr(&out));

    let tmp = tempfile::tempdir().unwrap();
    let dir = tmp.path();
    fit_with_scenarios(dir);
    let out = run(dir, &["predict", "@fit", "--horizon", "2 weeks", "--n-draws", "4",
                         "--scenario", "late", "--sweep", "cull_frac=0.5,1.0"]);
    assert!(out.status.success(), "predict: {}", stderr(&out));
    let fdir = forecast_dir(dir);
    let daily = rows(&fdir.join("forecast/daily_cases.tsv"));
    let arms: Vec<(String, String)> = daily.iter().filter(|r| r["t_stop"] == "31")
        .map(|r| (r["scenario"].clone(), r["sweep:cull_frac"].clone())).collect();
    assert_eq!(arms, vec![
        ("fitted".into(), "".into()), ("fitted".into(), "0.5".into()), ("late".into(), "0.5".into()),
        ("fitted".into(), "1".into()), ("late".into(), "1".into()),
    ]);
    let mean = |s: &str, x: &str| daily.iter().find(|r| r["t_stop"] == "31" && r["scenario"] == s
        && r["sweep:cull_frac"] == x).unwrap()["mean"].parse::<f64>().unwrap();
    assert_eq!(mean("fitted", "0.5"), mean("fitted", ""), "the fraction is read only by a cull the fitted model does not run");
    assert_eq!(mean("late", "1"), 0.0);
    assert!(mean("late", "0.5") > 0.0 && mean("late", "0.5") < mean("fitted", ""));
}

/// D83: a scenario forecast writes, per arm and target, the difference of the
/// predictive means and its Monte Carlo standard error, and records how the
/// arms are coupled. Before the cull the arms are identical, so the
/// difference and its error are exactly 0; after it, `late` has no infections
/// and the difference is minus the fitted model's mean.
#[test]
fn a_scenario_forecast_writes_the_difference_of_means_and_its_monte_carlo_error() {
    let tmp = tempfile::tempdir().unwrap();
    let dir = tmp.path();
    fit_with_scenarios(dir);
    let out = run(dir, &["predict", "@fit", "--horizon", "2 weeks", "--n-draws", "80",
                         "--scenario", "late"]);
    assert!(out.status.success(), "predict: {}", stderr(&out));
    let fdir = forecast_dir(dir);
    let contrasts = rows(&fdir.join("forecast-contrasts.tsv"));
    assert_eq!(contrasts.len(), 16, "one row per target of the one scenario arm");
    let daily = rows(&fdir.join("forecast/daily_cases.tsv"));
    for r in &contrasts {
        assert_eq!(r["scenario"], "late");
        assert_eq!(r["n_draws"], "80");
        let means = |s: &str| daily.iter().chain(&rows(&fdir.join("forecast/weekly_cases.tsv")))
            .find(|x| x["scenario"] == s && x["t_stop"] == r["t_stop"] && x["t_start"] == r["t_start"])
            .map(|x| x["mean"].parse::<f64>().unwrap());
        if let (Some(a), Some(b)) = (means("late"), means("fitted")) {
            let delta: f64 = r["delta"].parse().unwrap();
            assert!((delta - (a - b)).abs() <= 1e-6 * (1.0 + b.abs()), "{r:?}");
        }
        let end: f64 = r["t_stop"].parse().unwrap();
        if end <= 29.0 {
            assert_eq!((r["delta"].as_str(), r["delta_mcse"].as_str()), ("0", "0"), "{r:?}");
        } else if r["stream"] == "daily_cases" && end >= 31.0 {
            assert_eq!(r["mean"], "0");
            let mcse_given = r["delta_mcse"].parse::<f64>().is_ok_and(|s| s > 0.0);
            assert!(mcse_given || !r["mcse_note"].is_empty(), "an error, or the reason there is none: {r:?}");
        }
    }
    let m: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(fdir.join("forecast.json")).unwrap()).unwrap();
    assert_eq!(m["coupling"]["method"], "shared_seed_per_particle_streams");
    assert_eq!(m["contrasts"]["delta_mcse"]["kind"], "monte_carlo_standard_error");
    assert_eq!(m["contrasts"]["file"], "forecast-contrasts.tsv");
}

/// D85: the filters' particle count defaults to the PMMH fit's own (100
/// here) and `--particles` sets it. The resolved count, not its source, is
/// in the output address whenever a filter runs: `--particles 100` writes
/// the same bytes to the same directory as no flag, and another count keys
/// another directory. `--particles` where no filter runs is refused. Each
/// draw records its filter's ESS at its last assimilation (day 24), before
/// resampling, with ESS/N. The one-step band uses the same count and
/// `predictive.json` records it.
#[test]
fn the_particle_count_defaults_to_the_fits_and_is_recorded_and_addressed() {
    let tmp = tempfile::tempdir().unwrap();
    let dir = tmp.path();
    fit(dir, PMMH, "");
    let predict_dirs = || -> Vec<PathBuf> {
        let mut v: Vec<PathBuf> = walk(&dir.join("results")).into_iter()
            .filter(|p| p.file_name().is_some_and(|n| n == "forecast.json"))
            .map(|p| p.parent().unwrap().to_path_buf()).collect();
        v.sort();
        v
    };
    let manifest = |d: &Path| -> serde_json::Value {
        serde_json::from_str(&std::fs::read_to_string(d.join("forecast.json")).unwrap()).unwrap()
    };
    let forecast = |extra: &[&str]| {
        let mut args = vec!["predict", "@fit", "--horizon", "1 weeks", "--n-draws", "3"];
        args.extend_from_slice(extra);
        let out = run(dir, &args);
        assert!(out.status.success(), "predict: {}", stderr(&out));
    };
    forecast(&[]);
    let default_dir = predict_dirs().remove(0);
    let m = manifest(&default_dir);
    assert_eq!(m["particles"], serde_json::json!({"n": 100, "source": "fit_pmmh"}));
    assert_eq!(m["options"]["particles"], 100, "the resolved count is in the address");
    let samples = || std::fs::read(default_dir.join("forecast-samples.tsv")).unwrap();
    let before = samples();
    let diag = &m["filter_diagnostics"];
    assert_eq!(diag["n_particles"], 100);
    let last = diag["last_assimilation"].as_array().unwrap();
    assert_eq!(last.len(), 3);
    for (a, d) in last.iter().zip(m["draws"].as_array().unwrap()) {
        assert_eq!(a["cloud_row"], d["cloud_row"]);
        assert_eq!(a["t"], 24.0, "the last training row ends at day 24");
        let ess = a["ess"].as_f64().unwrap();
        assert!(ess > 0.0 && ess <= 100.0 + 1e-9, "{ess}");
        assert!((a["ess_over_n"].as_f64().unwrap() - ess / 100.0).abs() < 1e-12);
    }
    // Before resampling the weights are unequal; after it they are equal and
    // the ESS is N. An ESS read after resampling would equal 100 everywhere.
    assert!(last.iter().any(|a| a["ess"].as_f64().unwrap() < 100.0 - 1e-6),
        "some draw's ESS before resampling is below N: {last:?}");

    // The default count given as a flag is the same computation: the stored
    // forecast is reused, its manifest as the first predict wrote it, so
    // where the count came from is not part of the address.
    let manifest_before = std::fs::read(default_dir.join("forecast.json")).unwrap();
    forecast(&["--particles", "100"]);
    assert_eq!(predict_dirs(), vec![default_dir.clone()], "the default count keeps the default address");
    assert_eq!(samples(), before, "and the same output");
    assert_eq!(std::fs::read(default_dir.join("forecast.json")).unwrap(), manifest_before,
        "the source is not addressed: the stored forecast is the one reused");
    forecast(&["--particles", "40"]);
    let dirs = predict_dirs();
    assert_eq!(dirs.len(), 2, "another count, another directory: {dirs:?}");
    let other = dirs.iter().find(|d| **d != default_dir).unwrap();
    let m = manifest(other);
    assert_eq!((&m["particles"]["n"], &m["options"]["particles"]), (&serde_json::json!(40), &serde_json::json!(40)));
    let weekly = rows(&other.join("forecast/weekly_cases.tsv"));
    assert_eq!(weekly[0]["n_samples"], (3 * 40).to_string());

    let out = run(dir, &["predict", "@fit", "--mode", "one_step", "--n-draws", "3",
                         "--particles", "40"]);
    assert!(out.status.success(), "predict: {}", stderr(&out));
    let p = walk(&dir.join("results")).into_iter().find(|p| p.file_name().is_some_and(|n| n == "predictive.json"))
        .expect("the one-step predictive.json");
    let pj: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(p).unwrap()).unwrap();
    assert_eq!(pj["particles"], serde_json::json!({"n": 40, "source": "flag"}));
    let out = run(dir, &["predict", "@fit", "--mode", "free_forward", "--particles", "40"]);
    assert!(!out.status.success(), "{}", stderr(&out));
    assert!(stderr(&out).contains("this predict runs neither (--mode free_forward)"), "{}", stderr(&out));
}

/// Forecast artifacts record nothing about held-out rows: removing the
/// weekly stream's trailing held-out rows (the weeks ending 42 to 63) after
/// the fit writes every forecast file byte-identical to the same address.
#[test]
fn removing_held_out_rows_changes_nothing_the_forecast_writes() {
    let tmp = tempfile::tempdir().unwrap();
    let dir = tmp.path();
    fit(dir, PMMH, "");
    let snapshot = || {
        let out = run(dir, &["predict", "@fit", "--horizon", "4 weeks", "--n-draws", "2"]);
        assert!(out.status.success(), "predict: {}", stderr(&out));
        let d = forecast_dir(dir);
        ["forecast.json", "forecast-samples.tsv", "forecast/weekly_cases.tsv", "forecast/daily_cases.tsv"]
            .map(|f| (f, std::fs::read(d.join(f)).unwrap()))
    };
    let before = snapshot();
    let weekly = std::fs::read_to_string(dir.join("weekly_cases.tsv")).unwrap();
    let kept: Vec<&str> = weekly.lines()
        .filter(|l| l.split('\t').next().unwrap().parse::<f64>().map_or(true, |t| t <= 35.0))
        .collect();
    assert!(kept.len() < weekly.lines().count(), "rows were removed");
    std::fs::write(dir.join("weekly_cases.tsv"), kept.join("\n") + "\n").unwrap();
    for ((name, a), (_, b)) in snapshot().iter().zip(&before) {
        assert!(a == b, "{name} is byte-identical when trailing held-out rows are removed");
    }
}

/// A `[data.holdout]` file's values are outcomes, not inputs to the
/// predictions: editing a held-out value (the week ending 28, 127 → 128)
/// after the fit writes the forecast to the same directory with every file
/// byte-identical, `forecast.json` included.
#[test]
fn a_holdout_files_values_change_nothing_the_forecast_writes() {
    let tmp = tempfile::tempdir().unwrap();
    let dir = tmp.path();
    copy_fixture(dir);
    let text = std::fs::read_to_string(dir.join("weekly_cases.tsv")).unwrap();
    let mut lines = text.lines();
    let header = lines.next().unwrap().to_string();
    let (mut train, mut hold) = (vec![header.clone()], vec![header]);
    for l in lines {
        let t: f64 = l.split('\t').next().unwrap().parse().unwrap();
        if t <= 24.0 { train.push(l.to_string()) } else { hold.push(l.to_string()) }
    }
    std::fs::write(dir.join("weekly_cases.tsv"), train.join("\n") + "\n").unwrap();
    std::fs::write(dir.join("weekly_hold.tsv"), hold.join("\n") + "\n").unwrap();
    let daily = std::fs::read_to_string(dir.join("daily_cases.tsv")).unwrap();
    let daily: Vec<&str> = daily.lines().enumerate()
        .filter(|(i, l)| *i == 0 || l.split('\t').next().unwrap().parse::<f64>().unwrap() <= 24.0)
        .map(|(_, l)| l).collect();
    std::fs::write(dir.join("daily_cases.tsv"), daily.join("\n") + "\n").unwrap();
    fit_config(dir, PMMH, "",
        "daily_cases = \"daily_cases.tsv\"\nweekly_cases = \"weekly_cases.tsv\"",
        "\n[data.holdout]\nweekly_cases = \"weekly_hold.tsv\"\n");
    let snapshot = || {
        let out = run(dir, &["predict", "@fit", "--horizon", "2 weeks", "--n-draws", "2"]);
        assert!(out.status.success(), "predict: {}", stderr(&out));
        let d = forecast_dir(dir);
        ["forecast.json", "forecast-samples.tsv", "forecast/weekly_cases.tsv"]
            .map(|f| (f, std::fs::read(d.join(f)).unwrap()))
    };
    let before = snapshot();
    let hold = std::fs::read_to_string(dir.join("weekly_hold.tsv")).unwrap();
    assert!(hold.contains("28\t127"), "{hold}");
    std::fs::write(dir.join("weekly_hold.tsv"), hold.replace("28\t127", "28\t128")).unwrap();
    for ((name, a), (_, b)) in snapshot().iter().zip(&before) {
        assert!(a == b, "{name} is byte-identical when a held-out value changes");
    }
}

fn read_json(p: &Path) -> serde_json::Value {
    serde_json::from_str(&std::fs::read_to_string(p).unwrap()).unwrap()
}

/// Under a chain selection a forecast names each draw by its row of the fit's
/// `draws.tsv`, not by its place in the retained subset. Chain 0's rows come
/// first in `draws.tsv`, so dropping it shifts every retained draw's place:
/// a subset index cannot pass for a `draws.tsv` row.
#[test]
fn a_chain_subset_forecast_names_each_draw_by_its_draws_tsv_row() {
    let tmp = tempfile::tempdir().unwrap();
    let dir = tmp.path();
    fit(dir, PMMH, "");
    let out = run(dir, &["predict", "@fit", "--horizon", "1 weeks", "--n-draws", "4",
                         "--exclude-chains", "0"]);
    assert!(out.status.success(), "predict: {}", stderr(&out));
    let fdir = forecast_dir(dir);
    let manifest = read_json(&fdir.join("forecast.json"));
    let draws_tsv = walk(&dir.join("results")).into_iter()
        .find(|p| p.file_name().is_some_and(|n| n == "draws.tsv"))
        .expect("the fit wrote draws.tsv");
    let cloud = rows(&draws_tsv);
    let draws = manifest["draws"].as_array().unwrap();
    assert_eq!(draws.len(), 4);
    for d in draws {
        let (row, chain, draw) = (d["cloud_row"].as_u64().unwrap() as usize,
            d["chain"].as_u64().unwrap(), d["draw"].as_u64().unwrap());
        assert_eq!(chain, 1, "chain 0 was dropped: {d}");
        assert_eq!((cloud[row]["chain"].parse::<u64>().unwrap(), cloud[row]["draw"].parse::<u64>().unwrap()),
            (chain, draw), "cloud_row {row} is not this draw's row of draws.tsv");
    }
    let samples = rows(&fdir.join("forecast-samples.tsv"));
    assert_eq!(samples[0]["cloud_row"].parse::<u64>().unwrap(), draws[0]["cloud_row"].as_u64().unwrap());
    assert_eq!(samples[0]["chain"], "1");
}

/// D86: a forecast's identity is its keyed directory. The same forecast with
/// and without `--exclude-chains 1` writes to two directories, each with the
/// same unsuffixed file names (`forecast.json`, `report.json`), and
/// `forecast.json` states `options.excluded_chains` either way.
#[test]
fn a_chain_selection_keys_the_forecasts_directory_and_not_its_file_names() {
    let tmp = tempfile::tempdir().unwrap();
    let dir = tmp.path();
    fit(dir, PMMH, "");
    for extra in [&[][..], &["--exclude-chains", "1"][..]] {
        let mut args = vec!["predict", "@fit", "--horizon", "1 weeks", "--n-draws", "2"];
        args.extend_from_slice(extra);
        let out = run(dir, &args);
        assert!(out.status.success(), "predict: {}", stderr(&out));
    }
    let mut dirs: Vec<PathBuf> = walk(&dir.join("results")).into_iter()
        .filter(|p| p.file_name().is_some_and(|n| n == "forecast.json"))
        .map(|p| p.parent().unwrap().to_path_buf())
        .collect();
    dirs.sort();
    assert_eq!(dirs.len(), 2, "two selections, two directories: {dirs:?}");
    let mut excluded: Vec<serde_json::Value> = Vec::new();
    let mut keys: Vec<String> = Vec::new();
    for d in &dirs {
        let names: Vec<String> = std::fs::read_dir(d).unwrap().flatten()
            .map(|e| e.file_name().to_string_lossy().into_owned()).collect();
        assert!(names.iter().all(|n| !n.contains("-excl")), "no selection suffix in {d:?}: {names:?}");
        assert!(d.join("report.json").is_file(), "report.json in {d:?}: {names:?}");
        let m = read_json(&d.join("forecast.json"));
        // The folder is named by the prediction's id, which the selection keys.
        let id = fit_store_support::json(&d.join("identity.json"))["run_id"].as_str().unwrap().to_string();
        assert!(id.starts_with(d.file_name().unwrap().to_str().unwrap().rsplit('-').next().unwrap()), "{d:?}");
        keys.push(m["address_key"].as_str().unwrap().to_string());
        excluded.push(m["options"]["excluded_chains"].clone());
    }
    assert_ne!(keys[0], keys[1], "the selection keys the forecast's address");
    excluded.sort_by_key(|v| v.as_array().unwrap().len());
    assert_eq!(excluded, vec![serde_json::json!([]), serde_json::json!([1])]);
}

// ── D81: a stream whose likelihood reads a design column ─────────────────

/// A PMMH fit of `survey_sir.camdl` in `dir`, `holdout_after = 24`: the
/// survey's rows on days 28, 35 and 42 are held out in its own file.
fn fit_survey(dir: &Path) {
    copy_fixture(dir);
    std::fs::copy(fixture("survey_sir.camdl"), dir.join("model.camdl")).unwrap();
    std::fs::copy(fixture("survey.tsv"), dir.join("survey.tsv")).unwrap();
    fit_config(dir, PMMH, "holdout_after = 24.0\n",
        "daily_cases = \"daily_cases.tsv\"\nweekly_cases = \"weekly_cases.tsv\"\nsurvey = \"survey.tsv\"", "");
}

/// Every predict directory under `dir` holding a `forecast.json`.
fn forecast_dirs(dir: &Path) -> Vec<PathBuf> {
    let mut d: Vec<PathBuf> = walk(&dir.join("results")).into_iter()
        .filter(|p| p.file_name().is_some_and(|n| n == "forecast.json"))
        .map(|p| p.parent().unwrap().to_path_buf())
        .collect();
    d.sort();
    d
}

/// D81, ex post: a forecast from the held-out rows' realised design is
/// refused. A stored fit holds the dates of its held-out rows, not their
/// values, so there is no realised design to read; the refusal says so and
/// names `--design`, and nothing is stored.
#[test]
fn a_realised_design_forecast_is_refused() {
    let tmp = tempfile::tempdir().unwrap();
    let dir = tmp.path();
    fit_survey(dir);
    let out = run(dir, &["predict", "@fit", "--horizon", "2 weeks", "--n-draws", "2",
                         "--realised-design"]);
    assert!(!out.status.success(), "predict --realised-design ran: {}", stderr(&out));
    assert!(stderr(&out).contains("--realised-design") && stderr(&out).contains("held-out")
        && stderr(&out).contains("--design SOURCE=FILE"), "{}", stderr(&out));
    assert!(forecast_dirs(dir).is_empty(), "a refused forecast stores nothing");
}

/// D81: without a design the survey is refused, naming `--design`, and the
/// other streams are still forecast (exit status 1, D76). With
/// `--design survey=FILE` the forecast is prospective, reads the file's
/// `tested` (the file has no outcome column), and is a different address
/// from the forecast without one.
#[test]
fn a_design_reading_stream_needs_a_design_and_a_prospective_one_is_its_own_address() {
    let tmp = tempfile::tempdir().unwrap();
    let dir = tmp.path();
    fit_survey(dir);
    let out = run(dir, &["predict", "@fit", "--horizon", "2 weeks", "--n-draws", "2"]);
    assert_eq!(out.status.code(), Some(1), "predict: {}", stderr(&out));
    let m = read_json(&forecast_dirs(dir)[0].join("forecast.json"));
    let failed = m["failed"].to_string();
    assert!(failed.contains("survey"), "{failed}");
    assert!(stderr(&out).contains("--design"), "{}", stderr(&out));
    assert!(m["design"].is_null());
    assert!(m["produced"].as_array().unwrap().contains(&serde_json::json!("weekly_cases")));

    std::fs::write(dir.join("design.tsv"), "time\ttested\n28\t60\n35\t70\n").unwrap();
    let out = run(dir, &["predict", "@fit", "--horizon", "2 weeks", "--n-draws", "2",
                         "--design", "survey=design.tsv"]);
    assert!(out.status.success(), "predict: {}", stderr(&out));
    let manifests: Vec<serde_json::Value> = forecast_dirs(dir).iter().map(|d| read_json(&d.join("forecast.json"))).collect();
    assert_eq!(manifests.len(), 2, "two addresses: no design, prospective");
    let prospective = manifests.iter().find(|m| m["design"]["mode"] == "prospective").expect("a prospective forecast");
    assert!(prospective["design"]["label"].as_str().unwrap().starts_with("a prospective forecast"));
    // The design file is recorded by its content, not by the path typed.
    use sha2::Digest;
    let want = hex::encode(sha2::Sha256::digest(b"time\ttested\n28\t60\n35\t70\n"));
    assert_eq!(prospective["design"]["files"], serde_json::json!([{ "stream": "survey", "sha256": want }]),
        "{}", prospective["design"]);
    let tested: Vec<f64> = prospective["targets"].as_array().unwrap().iter()
        .filter(|t| t["stream"] == "survey").map(|t| t["design"]["tested"].as_f64().unwrap()).collect();
    assert_eq!(tested, vec![60.0, 70.0]);

    // Another design file is another prediction, with its own folder: the
    // file is part of the prediction's identity, by its bytes. This one gives
    // every target the same design values as `design.tsv` and differs in a
    // column no target reads, so only the file tells the two apart. The same
    // file again is the stored one.
    let before = forecast_dirs(dir).len();
    std::fs::write(dir.join("design2.tsv"), "time\ttested\tnote\n28\t60\ta\n35\t70\tb\n").unwrap();
    let out = run(dir, &["predict", "@fit", "--horizon", "2 weeks", "--n-draws", "2",
                         "--design", "survey=design2.tsv"]);
    assert!(out.status.success(), "predict: {}", stderr(&out));
    assert_eq!(forecast_dirs(dir).len(), before + 1, "a different design file is a different prediction");
    let out = run(dir, &["predict", "@fit", "--horizon", "2 weeks", "--n-draws", "2",
                         "--design", "survey=design2.tsv"]);
    assert!(out.status.success(), "predict: {}", stderr(&out));
    assert_eq!(forecast_dirs(dir).len(), before + 1, "the same design file is the stored prediction");

    // A design row missing for a target is refused, not filled in.
    std::fs::write(dir.join("design.tsv"), "time\ttested\n28\t60\n").unwrap();
    let out = run(dir, &["predict", "@fit", "--horizon", "2 weeks", "--n-draws", "2",
                         "--design", "survey=design.tsv"]);
    assert_eq!(out.status.code(), Some(1));
    assert!(stderr(&out).contains("has no value of the design column 'tested'"), "{}", stderr(&out));
    // Outside forecast mode the flags are refused.
    let out = run(dir, &["predict", "@fit", "--mode", "free_forward", "--realised-design"]);
    assert!(!out.status.success() && stderr(&out).contains("not a forecast"), "{}", stderr(&out));
}
