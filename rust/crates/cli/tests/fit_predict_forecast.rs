//! `camdl fit predict --horizon DURATION` (`--mode forecast`): the
//! fixed-origin forecast from the end of a fit's training rows (migration
//! plan, Stage 5; `docs/dev/temporal-architecture.md` §2.5, §4.3, §4.4).
//!
//! The fixture is `tests/fixtures/forecast/two_cadence_sir.camdl`, a
//! chain-binomial SIR reported daily and weekly, fitted by PMMH with
//! `holdout_after = 24`: the training rows end at day 24, inside the week
//! [21, 28).

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
    std::fs::write(dir.join("fit.toml"), format!(r#"output_dir = "results"

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
    let out = run(dir, &["fit", "run", "fit.toml", "--seed", "1"]);
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

/// The directory the latest predict wrote: `<segment>/predict/<run id>-<key>/`.
fn forecast_dir(dir: &Path) -> PathBuf {
    let mut found = Vec::new();
    for seg in std::fs::read_dir(dir.join("results/fits")).unwrap().flatten() {
        for d in std::fs::read_dir(seg.path().join("predict")).into_iter().flatten().flatten() {
            if d.path().join("forecast.json").is_file() {
                found.push(d.path());
            }
        }
    }
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
    let out = run(dir, &["fit", "predict", "--fit", "fit.toml", "--horizon", "4 weeks", "--n-draws", "8"]);
    assert!(out.status.success(), "fit predict: {}", stderr(&out));
    let fdir = forecast_dir(dir);

    let manifest: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(fdir.join("forecast.json")).unwrap()).unwrap();
    assert_eq!(manifest["schema"], "camdl.forecast/v2");
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
    let spans: Vec<(&str, &str, &str, &str, &str)> = weekly.iter()
        .map(|r| (r["t_start"].as_str(), r["t_stop"].as_str(), r["horizon"].as_str(),
                  r["lead"].as_str(), r["held_out_row"].as_str()))
        .collect();
    assert_eq!(spans, vec![
        ("21", "28", "0", "4", "3"),
        ("28", "35", "1", "11", "4"),
        ("35", "42", "2", "18", "5"),
        ("42", "49", "3", "25", "6"),
    ], "the week under way is horizon 0; horizons count weeks; held-out rows name their targets");
    let daily = rows(&fdir.join("forecast/daily_cases.tsv"));
    assert_eq!(daily.len(), 28, "daily horizons 0–27");
    assert_eq!((daily[0]["t_start"].as_str(), daily[0]["t_stop"].as_str()), ("24", "25"));
    assert_eq!(daily[27]["horizon"], "27");
    for r in daily.iter().chain(&weekly) {
        assert_eq!(r["n_samples"], (8 * 500).to_string());
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
    let out = run(tmp.path(), &["fit", "predict", "x", "--mode", "forecast"]);
    assert_eq!(out.status.code(), Some(1), "{}", stderr(&out));
    assert!(stderr(&out).contains("--mode forecast needs --horizon"), "{}", stderr(&out));
    let out = run(tmp.path(), &["fit", "predict", "x", "--mode", "one_step", "--horizon", "4 weeks"]);
    assert!(stderr(&out).contains("sets the reach of a forecast"), "{}", stderr(&out));
    // The old spelling of the mode flag is answered with the new one.
    let out = run(tmp.path(), &["fit", "predict", "x", "--horizon", "one_step"]);
    assert!(stderr(&out).contains("--mode one_step"), "{}", stderr(&out));
}

/// The capability matrix's refused cells for a fit that exists here: an
/// optimizer (IF2) has no posterior, and scenarios are not arms yet.
#[test]
fn single_estimate_fits_and_scenarios_are_refused_a_forecast() {
    let tmp = tempfile::tempdir().unwrap();
    let dir = tmp.path();
    fit(dir, r#"[method]
algorithm = "if2"
backend = "chain_binomial"
chains = 2
particles = 100
iterations = 3
cooling = 0.7"#, "");
    let out = run(dir, &["fit", "predict", "--fit", "fit.toml", "--horizon", "2 weeks"]);
    assert!(!out.status.success());
    assert!(stderr(&out).contains("single"), "IF2 is a single estimate: {}", stderr(&out));

    let tmp = tempfile::tempdir().unwrap();
    let dir = tmp.path();
    fit(dir, PMMH, "");
    let out = run(dir, &["fit", "predict", "--fit", "fit.toml", "--horizon", "2 weeks",
                         "--sweep", "rho=0.4,0.6"]);
    assert!(!out.status.success());
    assert!(stderr(&out).contains("cannot be combined with --mode forecast"), "{}", stderr(&out));
}

/// A held-out row is named by its file and its position within that file
/// (design call B): the week [21, 28) is row 3 of the training file under
/// `holdout_after = 24`, and row 0 of the `[data.holdout]` file when the
/// weekly rows after day 24 are in one. Without the file, row 0 of the
/// holdout file reads as training row [0, 7).
#[test]
fn a_held_out_row_is_named_by_its_file_and_its_position_in_it() {
    let tmp = tempfile::tempdir().unwrap();
    let dir = tmp.path();
    fit(dir, PMMH, "");
    let out = run(dir, &["fit", "predict", "--fit", "fit.toml", "--horizon", "1 weeks", "--n-draws", "2"]);
    assert!(out.status.success(), "{}", stderr(&out));
    let week = rows(&forecast_dir(dir).join("forecast/weekly_cases.tsv")).remove(0);
    assert_eq!((week["t_start"].as_str(), week["held_out_file"].as_str(), week["held_out_row"].as_str()),
        ("21", "training", "3"));

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
    let out = run(dir, &["fit", "predict", "--fit", "fit.toml", "--horizon", "1 weeks", "--n-draws", "2"]);
    assert!(out.status.success(), "{}", stderr(&out));
    let week = rows(&forecast_dir(dir).join("forecast/weekly_cases.tsv")).remove(0);
    assert_eq!((week["t_start"].as_str(), week["held_out_file"].as_str(), week["held_out_row"].as_str()),
        ("21", "holdout", "0"));
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
/// each leaf records its sampler, each forecast names the leaf's, and the two
/// forecasts of the same seed differ.
#[test]
fn a_forecast_uses_the_binomial_sampler_its_fit_ran() {
    let tmp = tempfile::tempdir().unwrap();
    let dir = tmp.path();
    fit(dir, PMMH, "");
    let out = run(dir, &["fit", "run", "fit.toml", "--seed", "1", "--binomial", "btrs"]);
    assert!(out.status.success(), "fit run --binomial btrs: {}", stderr(&out));
    let leaves: Vec<PathBuf> = walk(&dir.join("results")).into_iter()
        .filter(|p| p.file_name().is_some_and(|n| n == "run.json"))
        .map(|p| p.parent().unwrap().to_path_buf())
        .collect();
    assert_eq!(leaves.len(), 2, "{leaves:?}");
    let sampler_of = |leaf: &Path| -> String {
        let r: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(leaf.join("run.json")).unwrap()).unwrap();
        r["inputs"]["binomial"].as_str().unwrap().to_string()
    };
    let mut forecasts = Vec::new();
    for leaf in &leaves {
        let out = run(dir, &["fit", "predict", leaf.to_str().unwrap(), "--horizon", "1 weeks", "--n-draws", "2"]);
        assert!(out.status.success(), "{}", stderr(&out));
        let fdir = walk(&leaf.parent().unwrap().parent().unwrap().join("predict")).into_iter()
            .filter(|p| p.file_name().is_some_and(|n| n == "forecast.json"))
            .find(|p| {
                let m: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(p).unwrap()).unwrap();
                let r: serde_json::Value =
                    serde_json::from_str(&std::fs::read_to_string(leaf.join("run.json")).unwrap()).unwrap();
                m["run_id"] == r["run_id"]
            })
            .unwrap_or_else(|| panic!("no forecast of {}", leaf.display()))
            .parent().unwrap().to_path_buf();
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
    let out = run(dir, &["fit", "predict", "--fit", "fit.toml", "--horizon", "4 weeks", "--n-draws", "4"]);
    assert!(out.status.success(), "fit predict: {}", stderr(&out));

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
