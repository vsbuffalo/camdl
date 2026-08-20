//! gh#616 — a model's own observation anchors, end to end through the binary.
//!
//! Two things are proven here, and the first is the load-bearing one:
//!
//!  1. **A resolved anchor re-keys the run.** A model whose forcing fork is
//!     `breakpoints = [last_obs]` runs a *different* forcing under two data
//!     vintages, so the two runs must not share a `run_id` — otherwise the CAS
//!     serves the first vintage's trajectory for the second. The model's
//!     `simulate { to }` is a LITERAL here on purpose: the config level
//!     (`ResolvedEntry.t_end`) is then byte-identical between the two runs, and
//!     the only thing that can distinguish them is the substituted model. That
//!     makes this test the empirical proof that resolving into the hashed
//!     `base_model` is sufficient, with no extra hashed identity level.
//!
//!  2. The anchored horizon actually moves the run, is announced on stderr, and
//!     is refused (by name) where no data is bound.

use std::path::{Path, PathBuf};
use std::process::Command;

fn bin() -> PathBuf {
    let p = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../target/release/camdl");
    assert!(
        p.exists(),
        "release camdl binary missing: {} — run `make build-rust` or `make test` (gh#105)",
        p.display()
    );
    p
}

/// As [`write_model`], plus a scenario carrying its OWN horizon. Used for the
/// model-vs-scenario horizon conflict.
fn write_model_with_scenario(dir: &Path, sim_to: &str, scenario_to: &str) -> PathBuf {
    let base = std::fs::read_to_string(write_model(dir, sim_to)).unwrap();
    let body = format!(
        "{base}\nscenarios {{\n  forecast {{\n    label = \"forecast\"\n    \
         simulate {{ to = {scenario_to} }}\n    set = {{ beta = 0.35 }}\n  }}\n}}\n"
    );
    let p = dir.join("anchored.camdl");
    std::fs::write(&p, body).unwrap();
    p
}

/// SIR whose transmission is scaled by a piecewise forcing forked at the end of
/// the observed record. `simulate { to }` is a LITERAL — see the module doc.
fn write_model(dir: &Path, sim_to: &str) -> PathBuf {
    let body = format!(
        r#"time_unit = 'days

compartments {{ S, I, R }}

parameters {{
  beta  : rate in [0.05, 2.0]
  gamma : rate in [0.01, 1.0]
  rho   : probability in [0.05, 0.95]
}}

let N = S + I + R

forcing {{
  ramp : piecewise 'ratio {{
    breakpoints = [last_obs]
    values      = [1.0, 0.4]
  }}
}}

transitions {{
  infection : S --> I @ ramp * beta * S * (I / N)
  recovery  : I --> R @ gamma * I
}}

init {{ S = 990  I = 10 }}

observations {{
  cases {{
    columns       {{ time : time, cases : count }}
    projected     = incidence(infection)
    emit_schedule = every 7 'days
    cases         ~ poisson(rate = rho * projected)
  }}
}}

simulate {{ from = 0 'days  to = {sim_to} }}
"#
    );
    let p = dir.join("anchored.camdl");
    std::fs::write(&p, body).unwrap();
    p
}

/// `n_rows` weekly observations starting at t = 7. Two calls with different
/// `n_rows` are the two data vintages.
fn write_data(path: &Path, n_rows: usize) {
    let mut body = String::from("time\tcases\n");
    for i in 1..=n_rows {
        body.push_str(&format!("{}\t{}\n", i * 7, 5 + i));
    }
    std::fs::write(path, body).unwrap();
}

/// A minimal fit toml. Only `[data.observations]` is consulted when resolving
/// anchors; the rest is the schema-required minimum. A POSTERIOR stage (pmmh),
/// deliberately tiny, because `fit predict` draws bands and refuses an
/// optimizer-only fit.
fn write_fit_toml(dir: &Path, model: &Path, data: &Path) -> PathBuf {
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
[stages.posterior]
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

fn params_file(dir: &Path) -> PathBuf {
    let p = dir.join("params.toml");
    std::fs::write(&p, "beta = 0.3\ngamma = 0.1\nrho = 0.6\n").unwrap();
    p
}

/// Every `run_id` committed under a CAS root.
fn run_ids(cas_root: &Path) -> Vec<String> {
    let mut out = Vec::new();
    let mut stack = vec![cas_root.to_path_buf()];
    while let Some(d) = stack.pop() {
        let Ok(rd) = std::fs::read_dir(&d) else { continue };
        for e in rd.flatten() {
            let p = e.path();
            if p.is_dir() {
                let rj = p.join("run.json");
                if rj.is_file() {
                    if let Ok(txt) = std::fs::read_to_string(&rj) {
                        if let Ok(v) = serde_json::from_str::<serde_json::Value>(&txt) {
                            if let Some(id) = v.get("run_id").and_then(|x| x.as_str()) {
                                out.push(id.to_string());
                            }
                        }
                    }
                }
                stack.push(p);
            }
        }
    }
    out.sort();
    out.dedup();
    out
}

/// THE no-collision test. Same model, same literal horizon, same params, same
/// seed, same fit toml, same data PATH — only the file's contents move
/// `last_obs`, which moves the forcing fork. The two runs must key differently.
///
/// Mutation-checked: dropping the `base_model` substitution in
/// `build_simulate_cas_sink` (leaving only `resolve_run_model`'s, i.e. resolving
/// on the model that runs but not the one that is hashed) collapses this to one
/// `run_id` and the test goes red.
#[test]
fn two_data_vintages_do_not_share_a_run_id() {
    let bin = bin();
    let tmp = tempfile::tempdir().unwrap();
    let model = write_model(tmp.path(), "120 'days");
    let data = tmp.path().join("cases.tsv");
    let toml = write_fit_toml(tmp.path(), &model, &data);
    let params = params_file(tmp.path());
    let cas = tmp.path().join("cas");

    let run = |label: &str| -> String {
        let o = Command::new(&bin)
            .arg("simulate")
            .arg(&model)
            .args(["--backend", "chain_binomial", "--seed", "1", "--dt", "1"])
            .args(["--params", params.to_str().unwrap()])
            .args(["--fit", toml.to_str().unwrap()])
            .args(["-o", tmp.path().join(format!("{label}.tsv")).to_str().unwrap()])
            .env("CAMDL_SKIP_VERSION_CHECK", "1")
            .env("CAMDL_IR_CACHE_DIR", tmp.path().join("irc"))
            .env("CAMDL_OUTPUT_DIR", &cas)
            .output()
            .expect("spawn simulate");
        let stderr = String::from_utf8_lossy(&o.stderr).into_owned();
        assert!(o.status.success(), "simulate ({label}) must succeed:\n{stderr}");
        stderr
    };

    // Vintage 1: data ends at t = 28 → the fork sits at 28.
    write_data(&data, 4);
    let s1 = run("v1");
    assert!(
        s1.contains("forcing 'ramp'") && s1.contains("t = 28"),
        "the resolved knot must be announced on stderr:\n{s1}"
    );
    let after_first = run_ids(&cas);
    assert_eq!(after_first.len(), 1, "one run committed: {after_first:?}");

    // Vintage 2: data ends at t = 56 → the fork sits at 56. Nothing else moved.
    write_data(&data, 8);
    let s2 = run("v2");
    assert!(
        s2.contains("forcing 'ramp'") && s2.contains("t = 56"),
        "the second vintage must resolve to a different knot:\n{s2}"
    );
    let after_second = run_ids(&cas);
    assert_eq!(
        after_second.len(),
        2,
        "two data vintages fork the forcing differently, so they must commit two \
         DISTINCT run_ids — one id here means the CAS would serve the first \
         vintage's trajectory for the second: {after_second:?}"
    );

    // And the two runs really did produce different trajectories, so the
    // distinct keys are not merely bookkeeping.
    let t1 = std::fs::read_to_string(tmp.path().join("v1.tsv")).unwrap();
    let t2 = std::fs::read_to_string(tmp.path().join("v2.tsv")).unwrap();
    assert_ne!(t1, t2, "a fork at t=28 and a fork at t=56 must differ");
}

/// Negative control for the test above: with the SAME data twice, the second
/// run must hit the cache — one `run_id`, not two. Without this, a resolver that
/// simply randomised the digest would pass the no-collision test.
#[test]
fn the_same_data_twice_keeps_one_run_id() {
    let bin = bin();
    let tmp = tempfile::tempdir().unwrap();
    let model = write_model(tmp.path(), "120 'days");
    let data = tmp.path().join("cases.tsv");
    let toml = write_fit_toml(tmp.path(), &model, &data);
    let params = params_file(tmp.path());
    let cas = tmp.path().join("cas");
    write_data(&data, 4);

    for label in ["a", "b"] {
        let o = Command::new(&bin)
            .arg("simulate")
            .arg(&model)
            .args(["--backend", "chain_binomial", "--seed", "1", "--dt", "1"])
            .args(["--params", params.to_str().unwrap()])
            .args(["--fit", toml.to_str().unwrap()])
            .args(["-o", tmp.path().join(format!("{label}.tsv")).to_str().unwrap()])
            .env("CAMDL_SKIP_VERSION_CHECK", "1")
            .env("CAMDL_IR_CACHE_DIR", tmp.path().join("irc"))
            .env("CAMDL_OUTPUT_DIR", &cas)
            .output()
            .expect("spawn simulate");
        assert!(
            o.status.success(),
            "simulate ({label}) must succeed:\n{}",
            String::from_utf8_lossy(&o.stderr)
        );
    }
    let ids = run_ids(&cas);
    assert_eq!(ids.len(), 1, "unchanged data must re-key to the SAME run: {ids:?}");
}

/// An anchored MODEL with no `--fit` is refused by name — the same posture an
/// anchored `--to` already has, and the reason a data-free command can never
/// run one by accident.
#[test]
fn an_anchored_model_without_fit_is_refused() {
    let bin = bin();
    let tmp = tempfile::tempdir().unwrap();
    let model = write_model(tmp.path(), "120 'days");
    let params = params_file(tmp.path());

    let o = Command::new(&bin)
        .arg("simulate")
        .arg(&model)
        .args(["--backend", "chain_binomial", "--seed", "1", "--dt", "1"])
        .args(["--params", params.to_str().unwrap()])
        .arg("--stdout")
        .env("CAMDL_SKIP_VERSION_CHECK", "1")
        .env("CAMDL_IR_CACHE_DIR", tmp.path().join("irc"))
        .env("CAMDL_OUTPUT_DIR", tmp.path().join("cas"))
        .output()
        .expect("spawn simulate");
    let stderr = String::from_utf8_lossy(&o.stderr);
    assert!(!o.status.success(), "an anchored model without --fit must refuse");
    assert!(
        stderr.contains("anchored to observed data")
            && stderr.contains("ramp")
            && stderr.contains("--fit"),
        "the refusal must name the anchored construct and the fix:\n{stderr}"
    );
}

/// An anchored HORIZON resolves and actually extends the run: data ends at
/// t = 28, so `last_obs + 4 'weeks` runs to exactly t = 56.
#[test]
fn an_anchored_horizon_resolves_and_sets_the_grid() {
    let bin = bin();
    let tmp = tempfile::tempdir().unwrap();
    let model = write_model(tmp.path(), "last_obs + 4 'weeks");
    let data = tmp.path().join("cases.tsv");
    write_data(&data, 4); // last_obs = 28
    let toml = write_fit_toml(tmp.path(), &model, &data);
    let params = params_file(tmp.path());

    let out = tmp.path().join("traj.tsv");
    let o = Command::new(&bin)
        .arg("simulate")
        .arg(&model)
        .args(["--backend", "chain_binomial", "--seed", "1", "--dt", "1"])
        .args(["--params", params.to_str().unwrap()])
        .args(["--fit", toml.to_str().unwrap()])
        .args(["-o", out.to_str().unwrap()])
        .env("CAMDL_SKIP_VERSION_CHECK", "1")
        .env("CAMDL_IR_CACHE_DIR", tmp.path().join("irc"))
        .env("CAMDL_OUTPUT_DIR", tmp.path().join("cas"))
        .output()
        .expect("spawn simulate");
    let stderr = String::from_utf8_lossy(&o.stderr);
    assert!(o.status.success(), "an anchored horizon must run:\n{stderr}");
    assert!(
        stderr.contains("simulate { to }") && stderr.contains("t = 56"),
        "the resolved horizon must be announced:\n{stderr}"
    );

    let tsv = std::fs::read_to_string(&out).unwrap();
    let mut lines = tsv.lines().filter(|l| !l.trim_start().starts_with('#'));
    let header = lines.next().expect("header");
    let ti = header
        .split('\t')
        .position(|c| c == "time" || c == "t")
        .unwrap_or_else(|| panic!("time column in {header:?}"));
    let max_t = lines
        .filter(|l| !l.trim().is_empty())
        .map(|l| l.split('\t').nth(ti).unwrap().parse::<f64>().unwrap())
        .fold(f64::NEG_INFINITY, f64::max);
    assert_eq!(max_t, 56.0, "last_obs(28) + 4 'weeks = 56 must be the output horizon");
}

/// The proposal's red test for the sentinel decision, end to end.
///
/// Model horizon `last_obs + 4 'weeks`, scenario horizon `last_obs + 8 'weeks`.
/// `fit predict` emits at the OBSERVED times, so it cannot honour a scenario's
/// own window — and must SAY so rather than silently drop it. That refusal is
/// `refuse_scenario_horizon`, which works by comparing the two horizons for
/// equality. With the first draft's placeholder `t_end` both sides would have
/// carried the SAME placeholder number, compared equal, and the scenario's
/// eight-week window would have been discarded without a word. Resolving before
/// the guard — and NaN, not a placeholder, until then — is what makes this
/// refuse.
#[test]
fn predict_refuses_a_scenario_horizon_that_differs_after_resolution() {
    let bin = bin();
    let tmp = tempfile::tempdir().unwrap();
    let model =
        write_model_with_scenario(tmp.path(), "last_obs + 4 'weeks", "last_obs + 8 'weeks");
    let data = tmp.path().join("cases.tsv");
    write_data(&data, 4); // last_obs = 28
    let toml = write_fit_toml(tmp.path(), &model, &data);

    let fit = Command::new(&bin)
        .args(["fit", "run"])
        .arg(&toml)
        .args(["--seed", "1"])
        .env("CAMDL_SKIP_VERSION_CHECK", "1")
        .env("CAMDL_IR_CACHE_DIR", tmp.path().join("irc"))
        .output()
        .expect("spawn fit run");
    assert!(
        fit.status.success(),
        "the fit must run against an anchored model:\n{}",
        String::from_utf8_lossy(&fit.stderr)
    );

    // The stored fit segment (`fits/<stem>-<h8>/`), which is what `fit predict`
    // takes — the output_dir root holds many.
    let fits = tmp.path().join("results").join("fits");
    let segment = std::fs::read_dir(&fits)
        .unwrap_or_else(|e| panic!("no fits dir {}: {e}", fits.display()))
        .filter_map(|d| d.ok().map(|d| d.path()))
        .find(|p| p.is_dir())
        .expect("one fit segment");

    let predict = || -> (bool, String) {
        let o = Command::new(&bin)
            .args(["fit", "predict"])
            .arg(&segment)
            .args(["--scenario", "forecast"])
            .args(["--n-draws", "5"])
            .env("CAMDL_SKIP_VERSION_CHECK", "1")
            .env("CAMDL_IR_CACHE_DIR", tmp.path().join("irc"))
            .output()
            .expect("spawn fit predict");
        (o.status.success(), String::from_utf8_lossy(&o.stderr).into_owned())
    };

    let check = |ok: bool, stderr: String, via: &str| {
        assert!(
            !ok,
            "[{via}] a scenario horizon predict cannot honour must be REFUSED, \
             not dropped:\n{stderr}"
        );
        assert!(
            stderr.contains("forecast") && stderr.contains("horizon"),
            "[{via}] the refusal must name the scenario and the horizon:\n{stderr}"
        );
        // Both horizons resolved to real, DIFFERENT numbers before the
        // comparison: t = 56 (model, +4 weeks) and t = 84 (scenario, +8 weeks).
        // If either were still NaN, or both a shared placeholder, this number
        // could not appear.
        assert!(
            stderr.contains("84"),
            "[{via}] the message must quote the RESOLVED scenario horizon \
             (28 + 56 = 84):\n{stderr}"
        );
    };

    // (a) The normal path: `fit run` archived an already-resolved IR, so
    //     predict reads resolved numbers straight off the segment.
    let (ok, stderr) = predict();
    check(ok, stderr, "archived IR");

    // (b) The fallback path: an older run archived no IR, so predict recompiles
    //     `config.model.camdl` — which is STILL ANCHORED — and must resolve it
    //     itself before the guard. Without predict's own resolution the guard
    //     would compare NaN to NaN: it would still refuse (fail-closed), but the
    //     message would name no horizon a reader could check, and any command
    //     that proceeded would inherit an unresolved window.
    std::fs::remove_file(segment.join("model.ir.json")).expect("archived IR present to remove");
    let (ok, stderr) = predict();
    check(ok, stderr, "recompiled from model.camdl");
}
