//! gh#312: `camdl compare` on fits that ran `mh` on the `ode` backend.
//!
//! Two ODE + MH fits of a small SIR (one estimates β and γ, the other fixes
//! γ), both sealed at `holdout_after = 42` and compared on the two held-out
//! weeks. The derived trace is the posterior predictive of the ODE skeleton,
//! so its score at a held-out week is, exactly,
//!
//!   log (1/M) Σ_m NB(y_t | mean = ρ_m · incidence_t(θ_m), size = k)
//!
//! over the M thinned posterior draws — the log of the mixture density, not
//! the mean of the per-draw logs. The test recomputes that number without
//! `compare`: it thins `draws.tsv` by the documented rule, has `camdl simulate`
//! replay each draw on the fit's simulator and write the projected incidence
//! on the full design (`--design-from --projected-out`, full precision), and
//! evaluates the negative-binomial mass by hand.
//!
//! Shells out to the release `camdl`; the `.camdl` model is compiled by the
//! `camdlc` on PATH (`make test-rust` puts this checkout's first).

use std::path::{Path, PathBuf};
use std::process::{Command, Output};

fn binary() -> PathBuf {
    let bin = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../target/release/camdl");
    assert!(bin.exists(),
        "release camdl binary missing: {} - run `make build-rust` or `make test`",
        bin.display());
    bin
}

const MODEL: &str = r#"
time_unit = 'days
compartments { S, I, R }
parameters {
  beta  : rate         in [0.05, 1.0]  ~ log_normal(mu = -1.0, sigma = 0.5)
  gamma : rate         in [0.01, 0.5]  ~ log_normal(mu = -2.0, sigma = 0.5)
  N0    : count
  I0    : count
  rho   : probability  in [0.1, 0.9]
  k     : positive     in [1.0, 100.0]
}
let N = S + I + R
transitions {
  infection : S --> I  @ beta * S * I / N
  recovery  : I --> R  @ gamma * I
}
init { S = N0 - I0  I = I0 }
observations {
  weekly_cases {
    columns       { time : time, weekly_cases : count }
    covers        = closing_at(time, 7 'days)
    projected     = incidence(infection)
    emit_schedule = every 7 'days
    weekly_cases  ~ neg_binomial(mean = rho * projected, r = k)
  }
}
simulate { from = 0 'days  to = 80 'days }
"#;

const DATA: &str = "time\tweekly_cases\n7\t16\n14\t166\n21\t626\n28\t1303\n35\t1260\n42\t1023\n49\t327\n56\t91\n";

/// The held-out weeks (`t > 42`) and their counts.
const HELD_OUT: [(f64, u64); 2] = [(49.0, 327), (56.0, 91)];
const RHO: f64 = 0.6;
const K: f64 = 10.0;
const DRAWS: usize = 8;

/// The fit config; `estimate_gamma` is the one structural difference between
/// the two compared models. `holdout` false gives the design config the
/// replay reads (the same data, nothing withheld).
fn fit_toml(estimate_gamma: bool, holdout: bool) -> String {
    let gamma = if estimate_gamma {
        "gamma = { bounds = [0.01, 0.5], start = 0.15 }\n"
    } else { "" };
    let fixed_gamma = if estimate_gamma { "" } else { "gamma = 0.2\n" };
    let holdout = if holdout { "holdout_after = 42\n" } else { "" };
    format!(r#"output_dir = "results"
[model]
camdl = "model.camdl"
[data]
{holdout}[data.observations]
weekly_cases = "cases.tsv"
[estimate]
beta  = {{ bounds = [0.05, 1.0], start = 0.4 }}
{gamma}[fixed]
{fixed_gamma}N0  = 10000
I0  = 10
rho = {RHO}
k   = {K}
[method]
algorithm = "mh"
backend = "ode"
chains = 1
iterations = 300
burn_in = 100
starts = "single"
"#)
}

fn camdl(dir: &Path, args: &[&str]) -> Output {
    Command::new(binary()).env_remove("CLICOLOR_FORCE")
        .env_remove("CAMDL_OUTPUT_DIR")
        .env_remove("CAMDL_STORE")
        .env("CAMDL_SKIP_VERSION_CHECK", "1")
        .args(args)
        .current_dir(dir)
        .output().expect("spawn camdl")
}

fn ok(out: &Output, what: &str) -> String {
    assert!(out.status.success(), "{what} failed:\nstdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&out.stdout), String::from_utf8_lossy(&out.stderr));
    String::from_utf8_lossy(&out.stdout).into_owned()
}

/// Fit one model in `root/name` and return its directory.
fn fit(root: &Path, name: &str, estimate_gamma: bool) -> PathBuf {
    let dir = root.join(name);
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(dir.join("model.camdl"), MODEL).unwrap();
    std::fs::write(dir.join("cases.tsv"), DATA).unwrap();
    std::fs::write(dir.join("fit.toml"), fit_toml(estimate_gamma, true)).unwrap();
    std::fs::write(dir.join("design.toml"), fit_toml(estimate_gamma, false)).unwrap();
    ok(&camdl(&dir, &["fit", "run", "fit.toml", "--seed", "1"]), "fit run");
    dir
}

fn find_named(dir: &Path, name: &str) -> Vec<PathBuf> {
    let mut found = Vec::new();
    let mut stack = vec![dir.to_path_buf()];
    while let Some(d) = stack.pop() {
        for e in std::fs::read_dir(&d).into_iter().flatten().flatten() {
            let p = e.path();
            if p.is_dir() { stack.push(p); }
            else if p.file_name().and_then(|n| n.to_str()) == Some(name) { found.push(p); }
        }
    }
    found
}

/// A TSV with a header, `#` comment lines skipped, as (header, rows).
fn read_tsv(path: &Path) -> (Vec<String>, Vec<Vec<String>>) {
    let text = std::fs::read_to_string(path).unwrap();
    let mut lines = text.lines().filter(|l| !l.starts_with('#') && !l.is_empty());
    let header: Vec<String> = lines.next().unwrap().split('\t').map(String::from).collect();
    let rows = lines.map(|l| l.split('\t').map(String::from).collect()).collect();
    (header, rows)
}

/// The negative-binomial log mass with mean `mu` and size `r`, by hand:
/// `Γ(y+r)/(Γ(r)·y!) = Π_{j<y} (r+j)/(j+1)` for integer `y`.
fn nb_logpmf(y: u64, mu: f64, r: f64) -> f64 {
    let log_coef: f64 = (0..y).map(|j| ((r + j as f64) / (j as f64 + 1.0)).ln()).sum();
    log_coef + r * (r / (r + mu)).ln() + y as f64 * (mu / (r + mu)).ln()
}

/// Each held-out week's per-draw log density `log NB(y_t | ρ·incidence_t(θ), k)`
/// for every θ in `thetas` (rows of a column-per-parameter TSV), from
/// `camdl simulate`'s replay of those θ on the fit's simulator.
fn per_draw_log_densities(dir: &Path, thetas: &Path) -> Vec<Vec<f64>> {
    let proj = dir.join("proj");
    let _ = std::fs::remove_dir_all(&proj);
    ok(&camdl(dir, &["simulate", "model.camdl", "--draws", &thetas.to_string_lossy(),
        "--fit", "fit.toml", "--design-from", "design.toml",
        "--projected-out", &proj.to_string_lossy()]), "simulate --projected-out");
    let n = read_tsv(thetas).1.len();
    (0..n).map(|i| {
        let file = if n == 1 { proj.join("weekly_cases.tsv") }
                   else { proj.join(format!("ds_{:02}", i + 1)).join("weekly_cases.tsv") };
        let (h, rows) = read_tsv(&file);
        let (ti, pi) = (h.iter().position(|c| c == "time").unwrap(),
                        h.iter().position(|c| c == "projected").unwrap());
        HELD_OUT.iter().map(|&(t, y)| {
            let row = rows.iter().find(|r| r[ti].parse::<f64>().unwrap() == t)
                .unwrap_or_else(|| panic!("no projection at t = {t} in {}", file.display()));
            nb_logpmf(y, RHO * row[pi].parse::<f64>().unwrap(), K)
        }).collect()
    }).collect()
}

/// The fit's posterior draws (estimated + fixed columns, the `chain`/`draw`
/// keys dropped) in file order.
fn posterior(dir: &Path) -> (Vec<String>, Vec<Vec<String>>) {
    let draws = find_named(&dir.join("results"), "draws.tsv");
    assert_eq!(draws.len(), 1, "one draws.tsv under {}", dir.display());
    let (h, rows) = read_tsv(&draws[0]);
    (h[2..].to_vec(), rows.into_iter().map(|r| r[2..].to_vec()).collect())
}

fn write_thetas(path: &Path, header: &[String], rows: &[Vec<String>]) {
    let mut s = header.join("\t") + "\n";
    for r in rows { s += &(r.join("\t") + "\n"); }
    std::fs::write(path, s).unwrap();
}

fn log_mean_exp(v: &[f64]) -> f64 {
    let m = v.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    m + (v.iter().map(|x| (x - m).exp()).sum::<f64>() / v.len() as f64).ln()
}

fn compare_json(root: &Path, extra: &[&str]) -> serde_json::Value {
    let mut args = vec!["compare", "m0/fit.toml", "m1/fit.toml", "--format", "json",
        "--particles", "200"];
    args.extend_from_slice(extra);
    serde_json::from_str(&ok(&camdl(root, &args), "compare")).unwrap()
}

fn row<'a>(v: &'a serde_json::Value, name: &str) -> &'a serde_json::Value {
    v["rows"].as_array().unwrap().iter().find(|r| r["name"] == name)
        .unwrap_or_else(|| panic!("no row {name}: {v}"))
}

#[test]
fn ode_fits_compare_held_out_on_the_exact_posterior_mixture() {
    let tmp = tempfile::tempdir().unwrap();
    let root = tmp.path();
    let m0 = fit(root, "m0", true);
    let m1 = fit(root, "m1", false);

    let v = compare_json(root, &["--draws", &DRAWS.to_string()]);
    for (name, dir) in [("m0/fit.toml", &m0), ("m1/fit.toml", &m1)] {
        let r = row(&v, name);
        assert_eq!(r["basis"], serde_json::json!({"ode_free_forward": {"samples_per_draw": 200}}),
            "{name}: stamped as the ODE skeleton's predictive: {r}");
        assert_eq!(r["provenance"], serde_json::json!({"posterior": {"n_draws": DRAWS}}), "{r}");
        assert_eq!(r["conditioning"]["hold_out_tail"]["train_end"], 42.0, "{r}");
        assert_eq!(r["t_score"], 2, "the two weeks past the boundary: {r}");
        let d = &r["derivation"];
        assert_eq!((d["particles"].as_u64(), d["replicates"].as_u64()), (Some(0), Some(1)),
            "no filter ran, and a deterministic score has no replicates: {d}");
        assert!(d.get("mc_se_elpd").is_none(), "and so no filter-noise SE: {d}");

        // The same draws `compare` thinned (an even stride over the retained
        // rows in file order), replayed and scored by hand.
        let (header, rows) = posterior(dir);
        assert!(rows.len() > DRAWS, "the fixture must thin, not use every draw");
        let thinned: Vec<Vec<String>> =
            (0..DRAWS).map(|i| rows[i * rows.len() / DRAWS].clone()).collect();
        let thetas = dir.join("thinned.tsv");
        write_thetas(&thetas, &header, &thinned);
        let per_draw = per_draw_log_densities(dir, &thetas);
        let per_week: Vec<Vec<f64>> = (0..HELD_OUT.len())
            .map(|w| per_draw.iter().map(|d| d[w]).collect()).collect();
        let hand: f64 = per_week.iter().map(|w| log_mean_exp(w)).sum();
        let mean_of_logs: f64 = per_week.iter()
            .map(|w| w.iter().sum::<f64>() / w.len() as f64).sum();
        let elpd = r["elpd"].as_f64().unwrap();
        assert!((elpd - hand).abs() < 1e-9,
            "{name}: elpd {elpd} must be Σ_t log mean_s p(y_t | θ_s) = {hand}");
        assert!((hand - mean_of_logs).abs() > 1e-6,
            "non-vacuous: the mixture and the mean of logs differ ({hand} vs {mean_of_logs})");
    }
    // Two ODE fits on one basis and one window: a Δ is formed.
    assert!(row(&v, "m1/fit.toml")["delta_elpd"].is_f64()
        || row(&v, "m0/fit.toml")["delta_elpd"].is_f64(), "{v}");

    // `--draws 1`: the plug-in predictive at the posterior mean, also exact.
    let v = compare_json(root, &["--draws", "1"]);
    let r = row(&v, "m0/fit.toml");
    assert_eq!(r["provenance"], "plug_in", "{r}");
    let (header, rows) = posterior(&m0);
    let mean: Vec<String> = (0..header.len()).map(|c| {
        let s: f64 = rows.iter().map(|r| r[c].parse::<f64>().unwrap()).sum();
        format!("{:e}", s / rows.len() as f64)
    }).collect();
    let thetas = m0.join("mean.tsv");
    write_thetas(&thetas, &header, &[mean]);
    let hand: f64 = per_draw_log_densities(&m0, &thetas)[0].iter().sum();
    let elpd = r["elpd"].as_f64().unwrap();
    assert!((elpd - hand).abs() < 1e-9,
        "plug-in elpd {elpd} must be Σ_t log p(y_t | θ̄) = {hand}");
}

/// A filtered trace (a particle filter's estimate) and an ODE trace (exact) are
/// refused as a pair: the Δ would carry the filter's log bias as a model
/// difference. No flag waives it.
#[test]
fn an_ode_trace_is_not_differenced_against_a_filtered_one() {
    let tmp = tempfile::tempdir().unwrap();
    let root = tmp.path();
    let m0 = fit(root, "m0", true);
    std::fs::write(m0.join("theta.toml"),
        "beta = 0.4\ngamma = 0.15\nN0 = 10000\nI0 = 10\nrho = 0.6\nk = 10.0\n").unwrap();
    ok(&camdl(&m0, &["pfilter", "model.camdl", "--params", "theta.toml",
        "--data", "weekly_cases=cases.tsv", "--particles", "100",
        "--save-prequential", "pf", "--no-store"]), "pfilter --save-prequential");

    // Both in-sample, so the conditioning check passes and the basis check is
    // the one that refuses.
    let out = camdl(root, &["compare", "m0/fit.toml", "m0/pf.json", "--in-sample",
        "--draws", "2", "--particles", "100", "--allow-mixed-provenance",
        "--allow-mismatched-filter-settings", "--allow-data-mismatch"]);
    assert_eq!(out.status.code(), Some(2), "refused as incommensurable:\n{}",
        String::from_utf8_lossy(&out.stderr));
    let err = String::from_utf8_lossy(&out.stderr);
    assert!(err.contains("mix predictive bases"), "{err}");
    assert!(err.contains("ode_free_forward") && err.contains("filtered"), "{err}");
}
