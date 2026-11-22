//! The `bspline` forcing kind end to end (proposal
//! 2026-10-27-estimable-bsplines §4.3, §4.4, §9 Stage 2): through `camdlc` for
//! the unit handling, and through the `camdl` binary for observation anchors in
//! the window and a run past the window (refused, or held under
//! `--allow-forcing-extrapolation`).

use std::path::{Path, PathBuf};
use std::process::Command;

use sim::compiled_model::CompiledModel;
use sim::resolved_expr::{eval_resolved, ResolvedExpr};

fn repo() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../..")
}

fn bin() -> PathBuf {
    let p = repo().join("rust/target/release/camdl");
    assert!(p.exists(), "release camdl binary missing: {} — run `make build`", p.display());
    p
}

fn camdlc() -> PathBuf {
    let p = repo().join("ocaml/_build/default/bin/camdlc.exe");
    assert!(p.exists(), "camdlc missing: {} — run `make build-ocaml`", p.display());
    p
}

/// Compile `src` with `camdlc` into a model whose parameters are all fixed.
fn compile(dir: &Path, name: &str, src: &str, values: &[(&str, f64)]) -> ir::Model {
    let camdl = dir.join(format!("{name}.camdl"));
    let out = dir.join(format!("{name}.ir.json"));
    std::fs::write(&camdl, src).unwrap();
    let o = Command::new(camdlc()).env_remove("CLICOLOR_FORCE").arg(&camdl).arg("-o").arg(&out).output().unwrap();
    assert!(o.status.success(), "camdlc {name}:\n{}", String::from_utf8_lossy(&o.stderr));
    let mut m = ir::from_str(&std::fs::read_to_string(&out).unwrap()).unwrap();
    for p in &mut m.parameters {
        let v = values.iter().find(|(n, _)| *n == p.name).map(|(_, v)| *v).unwrap_or(0.1);
        p.value = p.value.with_value(v);
    }
    m
}

fn forcing_at(model: &CompiledModel, t: f64) -> f64 {
    let (int_s, real_s) = model.initial_state_mean(&model.default_params).unwrap();
    let ctx = sim::propensity::EvalCtx {
        model, int_s: &int_s, real_s: &real_s, params: &model.default_params,
        t, dt: 1.0, projected: None, aux: None, int_float_override: None, per_eval: None,
    };
    eval_resolved(&ResolvedExpr::TimeFunc(0), &ctx)
}

fn sir_with(forcing: &str, header: &str) -> String {
    format!(
        r#"time_unit = 'days
{header}
compartments {{ S, I, R }}
parameters {{
  gamma : rate in [0.01, 1.0]
}}
let N = S + I + R
forcing {{
{forcing}
}}
transitions {{
  infection : S --> I @ beta_t * S * (I / N)
  recovery  : I --> R @ gamma * I
}}
init {{ S = 990  I = 10 }}
simulate {{ from = 0 'days  to = 100 'days }}
"#
    )
}

/// §4.4: with `link = log` the declared unit scales the value, not the
/// coefficients' meaning. A `'per_week` spline is the `'per_day` spline with
/// the same coefficients times 1/7, to 1e-12 relative, inside the window and
/// in both held regions.
#[test]
fn log_link_per_week_is_per_day_over_seven() {
    let tmp = tempfile::tempdir().unwrap();
    let spline = |unit: &str| {
        format!(
            "  beta_t : bspline {unit} {{\n    window = [0 'days, 12 'weeks]\n    n_basis = 6\n    \
             link = log\n    coefs = [-1.2, -0.7, -0.4, -0.9, -1.6, -1.1]\n  }}"
        )
    };
    let week = CompiledModel::new(compile(tmp.path(), "week", &sir_with(&spline("'per_week"), ""), &[])).unwrap();
    let day = CompiledModel::new(compile(tmp.path(), "day", &sir_with(&spline("'per_day"), ""), &[])).unwrap();
    for i in 0..=300 {
        let t = -10.0 + 110.0 * i as f64 / 300.0;
        let (w, d) = (forcing_at(&week, t), forcing_at(&day, t) / 7.0);
        assert!(((w - d) / d).abs() <= 1e-12, "t={t}: per_week {w} vs per_day/7 {d}");
    }
}

/// Breakpoints written as calendar dates under an `origin` compile to model
/// days, and the forcing interpolates its end coefficients there.
#[test]
fn date_breakpoints_compile_to_model_days() {
    let tmp = tempfile::tempdir().unwrap();
    let src = sir_with(
        "  beta_t : bspline 'per_day {\n    \
         breakpoints = [date(\"2025-01-01\"), date(\"2025-01-15\"), date(\"2025-03-01\")]\n    \
         coefs = [0.3, 0.5, 0.4, 0.2, 0.25]\n  }",
        "origin = date(\"2025-01-01\")",
    );
    let m = compile(tmp.path(), "dated", &src, &[]);
    let ir::time_func::TimeFuncKind::BSpline(b) = &m.time_functions[0].kind else { panic!("bspline") };
    let ir::time_func::BSplineKnots::Explicit { breakpoints } = &b.knots else { panic!("explicit") };
    assert_eq!(breakpoints.len(), 3);
    let model = CompiledModel::new(m).unwrap();
    // 2025-01-01 is t = 0 and 2025-03-01 is t = 59.
    assert!((forcing_at(&model, 0.0) - 0.3).abs() <= 1e-15);
    assert!((forcing_at(&model, 59.0) - 0.25).abs() <= 1e-15);
    assert_eq!(forcing_at(&model, 80.0), forcing_at(&model, 59.0), "held past the last breakpoint");
}

// ── Through the binary: anchored windows and a run past the window ──────────

/// SIR scaled by a bspline whose window is `window`, observed weekly.
fn anchored_model(dir: &Path, window: &str) -> PathBuf {
    let p = dir.join("model.camdl");
    std::fs::write(
        &p,
        format!(
            r#"time_unit = 'days

compartments {{ S, I, R }}

parameters {{
  beta  : rate in [0.05, 2.0]
  gamma : rate in [0.01, 1.0]
  rho   : probability in [0.05, 0.95]
}}

let N = S + I + R

forcing {{
  mult : bspline 'ratio {{
    window  = {window}
    n_basis = 4
    coefs   = [1.0, 0.8, 0.6, 0.5]
  }}
}}

transitions {{
  infection : S --> I @ mult * beta * S * (I / N)
  recovery  : I --> R @ gamma * I
}}

init {{ S = 990  I = 10 }}

observations {{
  cases {{
    columns       {{ time : time, cases : count }}
    covers        = closing_at(time, 7 'days)
    projected     = incidence(infection)
    emit_schedule = every 7 'days
    cases         ~ poisson(rate = rho * projected)
  }}
}}

simulate {{ from = 0 'days  to = 120 'days }}
"#
        ),
    )
    .unwrap();
    p
}

/// Weekly observations at t = 7, 14, …, 35, so `last_obs` = 35.
fn fit_toml(dir: &Path, model: &Path) -> PathBuf {
    let data = dir.join("cases.tsv");
    let mut body = String::from("time\tcases\n");
    for i in 1..=5 {
        body.push_str(&format!("{}\t{}\n", i * 7, 5 + i));
    }
    std::fs::write(&data, body).unwrap();
    let toml = dir.join("fit.toml");
    std::fs::write(
        &toml,
        format!(
            r#"
output_dir = "{out}"
[model]
camdl = "{model}"
[data.observations]
cases = "{data}"
[config]
dt = 1.0
[estimate]
gamma = {{ bounds = [0.01, 1.0], start = 0.1, prior = {{ uniform = {{}} }} }}
[fixed]
beta = 0.12
rho = 0.6
[method]
algorithm  = "pmmh"
backend    = "chain_binomial"
chains     = 1
particles  = 20
iterations = 20
burn_in    = 5
thin       = 1
"#,
            out = dir.join("results").display(),
            model = model.display(),
            data = data.display(),
        ),
    )
    .unwrap();
    toml
}

/// Run `camdl simulate` with `extra` arguments; returns (success, stderr, CAS
/// root).
fn simulate(dir: &Path, model: &Path, fit: Option<&Path>, extra: &[&str]) -> (bool, String, PathBuf) {
    let params = dir.join("params.toml");
    std::fs::write(&params, "beta = 0.12\ngamma = 0.1\nrho = 0.6\n").unwrap();
    let cas = dir.join("cas");
    let mut cmd = Command::new(bin());
    cmd.env_remove("CLICOLOR_FORCE");
    cmd.arg("simulate")
        .arg(model)
        .args(["--backend", "chain_binomial", "--seed", "1", "--dt", "1"])
        .args(["--params", params.to_str().unwrap()])
        .args(extra)
        .env("CAMDL_SKIP_VERSION_CHECK", "1")
        .env("CAMDL_IR_CACHE_DIR", dir.join("irc"))
        .env("CAMDL_OUTPUT_DIR", &cas).env("CAMDL_STORE", std::path::Path::new(&&cas).join("camdl-store"));
    if let Some(f) = fit {
        cmd.args(["--fit", f.to_str().unwrap()]);
    }
    let o = cmd.output().expect("spawn simulate");
    (o.status.success(), String::from_utf8_lossy(&o.stderr).into_owned(), cas)
}

/// The forcings every committed run.json under `cas` records as held.
fn recorded_holds(cas: &Path) -> Vec<String> {
    let mut out = Vec::new();
    let mut stack = vec![cas.to_path_buf()];
    while let Some(d) = stack.pop() {
        let Ok(rd) = std::fs::read_dir(&d) else { continue };
        for e in rd.flatten() {
            let p = e.path();
            if p.is_dir() {
                stack.push(p);
            } else if p.file_name().is_some_and(|n| n == "provenance.json") {
                // A cell's provenance file records each forcing it held.
                let v: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(&p).unwrap()).unwrap();
                let held = v.pointer("/forcing_extrapolation/held").and_then(|n| n.as_array()).cloned().unwrap_or_default();
                out.extend(held.iter().filter_map(|n| n["forcing"].as_str().map(str::to_string)));
            }
        }
    }
    out
}

const OVERRIDE: &str = "--allow-forcing-extrapolation";

/// The warning a run held past the window `[0, end]` prints.
fn held_warning(end: u32) -> String {
    format!(
        "warning: forcing `mult` is defined on t = 0 to t = {end} (its window); the run evaluated it after the \
         end of its window, and its value at t = {end} is held from t = {end} to the end of the run"
    )
}

/// A window ending at `last_obs - 2 'weeks` resolves against the bound data
/// (last_obs = 35, so b = 21). A run past it is refused; under the override
/// it holds, says so once, and the run's record names the forcing held.
#[test]
fn anchored_window_resolves_and_the_hold_is_refused_or_warned() {
    let tmp = tempfile::tempdir().unwrap();
    let model = anchored_model(tmp.path(), "[0 'days, last_obs - 2 'weeks]");
    let fit = fit_toml(tmp.path(), &model);
    let (ok, stderr, _) = simulate(tmp.path(), &model, Some(&fit), &[]);
    assert!(!ok, "a run past the window must be refused:\n{stderr}");
    assert!(stderr.contains("forcing 'mult'") && stderr.contains("t = 21"), "resolved window end announced:\n{stderr}");
    assert!(stderr.contains("error: forcing `mult` was evaluated after the end of its window"), "{stderr}");

    let (ok, stderr, cas) = simulate(tmp.path(), &model, Some(&fit), &[OVERRIDE]);
    assert!(ok, "simulate under the override must succeed:\n{stderr}");
    assert_eq!(stderr.matches(&held_warning(21)).count(), 1, "the warning, once:\n{stderr}");
    assert_eq!(recorded_holds(&cas), vec!["mult".to_string()], "the hold is recorded with the run");
}

/// An anchored window that inverts under this data vintage is refused, naming
/// the forcing and the anchors: [30, last_obs − 14] = [30, 21].
#[test]
fn an_inverted_anchored_window_is_refused() {
    let tmp = tempfile::tempdir().unwrap();
    let model = anchored_model(tmp.path(), "[30 'days, last_obs - 2 'weeks]");
    let fit = fit_toml(tmp.path(), &model);
    let (ok, stderr, _) = simulate(tmp.path(), &model, Some(&fit), &[OVERRIDE]);
    assert!(!ok, "an inverted window must be refused:\n{stderr}");
    assert!(
        stderr.contains("forcing 'mult' has its window or breakpoints out of order after resolving its \
                         observation anchors: [30.0, 21.0]"),
        "{stderr}"
    );
}

/// A literal window: a run whose horizon (120) passes the window end is
/// refused without the override and warned under it; a window covering the
/// run gets neither.
#[test]
fn only_a_run_past_the_window_is_refused() {
    let tmp = tempfile::tempdir().unwrap();
    let model = anchored_model(tmp.path(), "[0 'days, 50 'days]");
    let (ok, stderr, _) = simulate(tmp.path(), &model, None, &[]);
    assert!(!ok, "{stderr}");
    let (ok, stderr, cas) = simulate(tmp.path(), &model, None, &[OVERRIDE]);
    assert!(ok, "{stderr}");
    assert_eq!(stderr.matches(&held_warning(50)).count(), 1, "{stderr}");
    assert_eq!(recorded_holds(&cas), vec!["mult".to_string()]);

    let tmp = tempfile::tempdir().unwrap();
    let model = anchored_model(tmp.path(), "[0 'days, 150 'days]");
    let (ok, stderr, cas) = simulate(tmp.path(), &model, None, &[]);
    assert!(ok, "{stderr}");
    assert!(!stderr.contains("forcing `mult`"), "a window covering the run says nothing:\n{stderr}");
    assert!(recorded_holds(&cas).is_empty());
}

/// `pfilter` and `profile` resolve the same anchored window and,
/// with the data running past it, refuse the run without the override and
/// warn once each under it.
#[test]
fn pfilter_and_profile_refuse_or_warn_the_hold() {
    let tmp = tempfile::tempdir().unwrap();
    let model = anchored_model(tmp.path(), "[0 'days, last_obs - 2 'weeks]");
    let fit = fit_toml(tmp.path(), &model);
    let params = tmp.path().join("params.toml");
    std::fs::write(&params, "beta = 0.12\ngamma = 0.1\nrho = 0.6\n").unwrap();
    let line = held_warning(21);
    let run = |label: &str, args: Vec<String>| {
        let go = |extra: &[&str]| {
            let o = Command::new(bin()).env_remove("CLICOLOR_FORCE")
                .args(&args)
                .args(extra)
                .env("CAMDL_SKIP_VERSION_CHECK", "1")
                .env("CAMDL_IR_CACHE_DIR", tmp.path().join("irc"))
                .env("CAMDL_OUTPUT_DIR", tmp.path().join("cas")).env("CAMDL_STORE", std::path::Path::new(&tmp.path().join("cas")).join("camdl-store"))
                .output()
                .expect("spawn");
            (o.status.success(), String::from_utf8_lossy(&o.stderr).into_owned())
        };
        let (ok, stderr) = go(&[]);
        assert!(!ok, "{label} must be refused without the override:\n{stderr}");
        assert!(stderr.contains("error: forcing `mult` was evaluated after the end of its window (it is defined \
                                 on t = 0 to t = 21"), "{label}:\n{stderr}");
        let (ok, stderr) = go(&[OVERRIDE]);
        assert!(ok, "{label} must run under the override:\n{stderr}");
        assert_eq!(stderr.matches(&line).count(), 1, "{label}: the warning, once:\n{stderr}");
    };
    let s = |x: &Path| x.to_str().unwrap().to_string();
    let m = s(&model);
    let f = s(&fit);
    run("pfilter", ["pfilter", &m, "--params", &s(&params), "--fit", &f, "--particles", "50", "--seed", "1",
                    "--no-save-samples"].map(String::from).to_vec());
    run("profile", ["profile", &m, "--fit", &f, "--sweep", "gamma=lin(0.08,0.12,2)", "--algorithm", "if2",
                    "--particles", "30", "--iterations", "3", "--n-starts", "1", "--rw-sd", "auto", "--seed", "1"]
        .map(String::from).to_vec());
}

/// A tiny pgas `fit run` of `model` on weekly synthetic counts, with the given
/// `[estimate]` and extra `[fixed]` lines; returns (success, stderr).
fn pgas_fit(tmp: &Path, tag: &str, model: &Path, estimate: &str, fixed: &str) -> (bool, String) {
    let data = tmp.join("cases.tsv");
    let mut body = String::from("time\tcases\n");
    for w in 1..=9 {
        body.push_str(&format!("{}\t{}\n", 7 * w, 20 + 7 * w));
    }
    std::fs::write(&data, body).unwrap();
    let toml = tmp.join(format!("fit_{tag}.toml"));
    std::fs::write(&toml, format!(
        "output_dir = \"{out}\"\n[model]\ncamdl = \"{model}\"\n[data.observations]\ncases = \"{data}\"\n\
         [config]\ndt = 1.0\n[estimate]\n{estimate}[fixed]\n{fixed}\
         [method]\nalgorithm = \"pgas\"\nbackend = \"chain_binomial\"\nchains = 1\nparticles = 8\nsweeps = 4\nburn_in = 2\n",
        out = tmp.join(format!("results_{tag}")).display(),
        model = model.display(), data = data.display(),
    )).unwrap();
    let o = Command::new(bin()).env_remove("CLICOLOR_FORCE")
        .args(["fit", "run", toml.to_str().unwrap()])
        .env("CAMDL_SKIP_VERSION_CHECK", "1")
        .env("CAMDL_IR_CACHE_DIR", tmp.join("irc"))
        .env("CAMDL_OUTPUT_DIR", tmp.join("cas")).env("CAMDL_STORE", std::path::Path::new(&tmp.join("cas")).join("camdl-store"))
        .output().unwrap();
    (o.status.success(), String::from_utf8_lossy(&o.stderr).into_owned())
}

/// An unbounded `real` with a proper prior and no `start` runs under the
/// default `starts = from_prior`: the placeholder start is a prior draw (it
/// was refused as "has no value" before). Explicit log-link coefficients.
#[test]
fn pgas_runs_unbounded_coefficients_without_starts() {
    let tmp = tempfile::tempdir().unwrap();
    let model = tmp.path().join("sir_unbounded.camdl");
    std::fs::write(&model, r#"time_unit = 'days
compartments { S, I, R }
let N = S + I + R
parameters {
  gamma : rate in [0.01, 1.0]
  c1 : real dim 1 ~ normal(mu = -1.5, sigma = 1.0)
  c2 : real dim 1 ~ normal(mu = -1.5, sigma = 1.0)
  c3 : real dim 1 ~ normal(mu = -1.5, sigma = 1.0)
  c4 : real dim 1 ~ normal(mu = -1.5, sigma = 1.0)
  rho : probability in [0.01, 1.0]
}
forcing {
  beta_t : bspline 'per_day {
    window  = [0 'days, 63 'days]
    n_basis = 4
    link    = log
    coefs   = [c1, c2, c3, c4]
  }
}
transitions {
  infection : S --> I @ beta_t * S * (I / N)
  recovery  : I --> R @ gamma * I
}
init { S = 9900  I = 100 }
observations {
  cases {
    columns       { time : time, cases : count }
    covers        = closing_at(time, 7 'days)
    projected     = incidence(infection)
    emit_schedule = every 7 'days
    cases         ~ poisson(rate = rho * projected)
  }
}
simulate {
  from = 0 'days
  to   = 63 'days
}
"#).unwrap();
    let (ok, stderr) = pgas_fit(tmp.path(), "unbounded", &model, "c1 = {}\nc2 = {}\nc3 = {}\nc4 = {}\n",
                                "gamma = 0.15\nrho = 0.5\n");
    assert!(ok && !stderr.contains("has no value"), "the unbounded fit must run:\n{stderr}");
}

/// The smooth coefficient form is implemented but not yet released: every
/// command refuses a model that uses it, with the explicit-coefficient remedy.
#[test]
fn smooth_coefficients_are_not_yet_available() {
    let tmp = tempfile::tempdir().unwrap();
    let model = repo().join("tests/fixtures/gradient/sir_smooth_fit.camdl");
    let (ok, stderr) = pgas_fit(tmp.path(), "smooth", &model, "a = {}\n",
        "gamma = 0.15\nb = 0.0\nrho = 0.5\ntau = 0.3\nz_m1 = 0.0\nz_m2 = 0.0\nz_m3 = 0.0\nz_m4 = 0.0\n");
    assert!(!ok && stderr.contains("`coefs = smooth(...)` is not yet available in this release")
        && stderr.contains("keep `n_basis` small"), "fit run must refuse smooth:\n{stderr}");
    let o = Command::new(bin()).env_remove("CLICOLOR_FORCE")
        .args(["simulate", model.to_str().unwrap(), "--param", "a=-1.5", "--param", "b=0", "--param", "tau=0.3",
               "--param", "z_m1=0", "--param", "z_m2=0", "--param", "z_m3=0", "--param", "z_m4=0",
               "--param", "gamma=0.15", "--param", "rho=0.5", "--seed", "1", "--stdout"])
        .env("CAMDL_SKIP_VERSION_CHECK", "1")
        .env("CAMDL_IR_CACHE_DIR", tmp.path().join("irc"))
        .env("CAMDL_OUTPUT_DIR", tmp.path().join("cas")).env("CAMDL_STORE", std::path::Path::new(&tmp.path().join("cas")).join("camdl-store"))
        .output().unwrap();
    let stderr = String::from_utf8_lossy(&o.stderr);
    assert!(!o.status.success() && stderr.contains("is not yet available in this release"), "simulate must refuse smooth:\n{stderr}");
}
