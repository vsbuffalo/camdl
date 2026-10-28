//! The `bspline` forcing kind end to end (proposal
//! 2026-10-27-estimable-bsplines §4.3, §4.4, §9 Stage 2): through `camdlc` for
//! the unit handling, and through the `camdl` binary for observation anchors in
//! the window and the `spline-held-past-window` notice.

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
    let o = Command::new(camdlc()).arg(&camdl).arg("-o").arg(&out).output().unwrap();
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

// ── Through the binary: anchored windows and the notice ─────────────────────

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

/// Run `camdl simulate`; returns (success, stderr, CAS root).
fn simulate(dir: &Path, model: &Path, fit: Option<&Path>) -> (bool, String, PathBuf) {
    let params = dir.join("params.toml");
    std::fs::write(&params, "beta = 0.12\ngamma = 0.1\nrho = 0.6\n").unwrap();
    let cas = dir.join("cas");
    let mut cmd = Command::new(bin());
    cmd.arg("simulate")
        .arg(model)
        .args(["--backend", "chain_binomial", "--seed", "1", "--dt", "1"])
        .args(["--params", params.to_str().unwrap()])
        .env("CAMDL_SKIP_VERSION_CHECK", "1")
        .env("CAMDL_IR_CACHE_DIR", dir.join("irc"))
        .env("CAMDL_OUTPUT_DIR", &cas);
    if let Some(f) = fit {
        cmd.args(["--fit", f.to_str().unwrap()]);
    }
    let o = cmd.output().expect("spawn simulate");
    (o.status.success(), String::from_utf8_lossy(&o.stderr).into_owned(), cas)
}

/// Every `notices` record in the committed run.json files under `cas`.
fn recorded_notice_codes(cas: &Path) -> Vec<String> {
    let mut out = Vec::new();
    let mut stack = vec![cas.to_path_buf()];
    while let Some(d) = stack.pop() {
        let Ok(rd) = std::fs::read_dir(&d) else { continue };
        for e in rd.flatten() {
            let p = e.path();
            if p.is_dir() {
                stack.push(p);
            } else if p.file_name().is_some_and(|n| n == "run.json") {
                let v: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(&p).unwrap()).unwrap();
                let notices = v.pointer("/inputs/notices").and_then(|n| n.as_array()).cloned().unwrap_or_default();
                out.extend(notices.iter().filter_map(|n| n["code"].as_str().map(str::to_string)));
            }
        }
    }
    out
}

const NOTICE_CODE: &str = "spline-held-past-window";

/// A window ending at `last_obs - 2 'weeks` resolves against the bound data
/// (last_obs = 35, so b = 21), and the run past it carries the pinned notice,
/// on stderr and in the run's record.
#[test]
fn anchored_window_resolves_and_the_hold_is_noticed() {
    let tmp = tempfile::tempdir().unwrap();
    let model = anchored_model(tmp.path(), "[0 'days, last_obs - 2 'weeks]");
    let fit = fit_toml(tmp.path(), &model);
    let (ok, stderr, cas) = simulate(tmp.path(), &model, Some(&fit));
    assert!(ok, "simulate must succeed:\n{stderr}");
    assert!(stderr.contains("forcing 'mult'") && stderr.contains("t = 21"), "resolved window end announced:\n{stderr}");
    let line = "info: forcing `mult` is held at its value at t = 21 from t = 21 to the horizon t = 120; \
                its window ends at last_obs - 14";
    assert_eq!(stderr.matches(line).count(), 1, "the notice, once:\n{stderr}");
    assert_eq!(recorded_notice_codes(&cas), vec![NOTICE_CODE.to_string()], "the notice is recorded with the run");
}

/// An anchored window that inverts under this data vintage is refused, naming
/// the forcing and the anchors: [30, last_obs − 14] = [30, 21].
#[test]
fn an_inverted_anchored_window_is_refused() {
    let tmp = tempfile::tempdir().unwrap();
    let model = anchored_model(tmp.path(), "[30 'days, last_obs - 2 'weeks]");
    let fit = fit_toml(tmp.path(), &model);
    let (ok, stderr, _) = simulate(tmp.path(), &model, Some(&fit));
    assert!(!ok, "an inverted window must be refused:\n{stderr}");
    assert!(
        stderr.contains("forcing 'mult' has its window or breakpoints out of order after resolving its \
                         observation anchors: [30.0, 21.0]"),
        "{stderr}"
    );
}

/// A literal window: the notice fires when the horizon (120) passes the window
/// end, without the anchor clause, and not when the window covers the run.
#[test]
fn the_notice_fires_only_past_the_window() {
    let tmp = tempfile::tempdir().unwrap();
    let model = anchored_model(tmp.path(), "[0 'days, 50 'days]");
    let (ok, stderr, cas) = simulate(tmp.path(), &model, None);
    assert!(ok, "{stderr}");
    let line = "info: forcing `mult` is held at its value at t = 50 from t = 50 to the horizon t = 120\n";
    assert_eq!(stderr.matches(line).count(), 1, "{stderr}");
    assert_eq!(recorded_notice_codes(&cas), vec![NOTICE_CODE.to_string()]);

    let tmp = tempfile::tempdir().unwrap();
    let model = anchored_model(tmp.path(), "[0 'days, 150 'days]");
    let (ok, stderr, cas) = simulate(tmp.path(), &model, None);
    assert!(ok, "{stderr}");
    assert!(!stderr.contains("is held at its value"), "a window covering the run gets no notice:\n{stderr}");
    assert!(recorded_notice_codes(&cas).is_empty());
}
