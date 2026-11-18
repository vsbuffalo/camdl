//! A case for every input of `camdl pfilter`, checked as its class predicts
//! (`rust/crates/cli/src/input_class.rs`, `PFILTER`; proposal
//! `docs/dev/proposals/2026-11-17-honest-input-addressed-store.md` phase 3
//! "Tests"). The runner is `input_class_runner`.
//!
//! Each case is one test, named `<class>_<input>` (`keyed_particles`,
//! `execution_parallel`, `presentation_trace`); an environment variable's
//! case is `<class>_env_<field>`; `input_class::tests::every_pfilter_input_has_a_case`
//! fails if an input has none. What each class asserts:
//!
//! - `keyed(level)`: changed alone, the pfilter leaf's run id changes, and so
//!   does its hash at the level the class table names. An environment
//!   variable the table keys at a simulation's `model` level enters a pfilter
//!   at its `model` level too, which holds the model and the engine.
//! - `execution`: the run id and the stored bytes stay the same.
//! - `presentation`: the leaf is unaffected (or, for `--no-store`, none is
//!   written).
//!
//! These are regression evidence on small fixtures, not proof (the proposal,
//! "What tests can and cannot show"). Each filter runs twenty particles over
//! twenty days.
//!
//! The fixtures: `sir.camdl`, an SIR model with a vaccination intervention,
//! two scenarios that set every parameter, two observed streams and a
//! parameter-only rate factor that loop-invariant code motion hoists;
//! `patch.camdl`, a two-patch model with an `external` table and a coupling
//! matrix whose structural zeros the constant fold drops. The filters run on
//! the compiled IR except where the compile itself is the input under test.

mod input_class_runner;

use std::path::PathBuf;
use std::sync::{Arc, Mutex, Weak};

use input_class_runner::*;

const KIND: &str = "pfilter";

const SIR: &str = r#"
time_unit = 'days

compartments { S, I, R, V }

let N = S + I + R + V

parameters {
  beta_a : rate        in [0.01, 5.0]
  beta_b : rate        in [0.01, 5.0]
  gamma  : rate        in [0.01, 5.0]
  rho    : probability in [0.001, 1.0]
}

transitions {
  infection_a : S --> I  @ beta_a * exp(-rho) * S * I / N
  infection_b : S --> I  @ beta_b * S * I / N
  recovery    : I --> R  @ gamma * I
}

init { S = 990  I = 10  R = 0  V = 0 }

interventions {
  sia : transfer(fraction = 0.05, from = S, to = V) at [5 'days]
}

scenarios {
  set_all {
    set = {
      beta_a = 0.4
      beta_b = 0.2
      gamma = 0.2
      rho = 0.5
    }
  }
  set_fast {
    set = {
      beta_a = 0.4
      beta_b = 0.3
      gamma = 0.2
      rho = 0.5
    }
  }
}

simulate { from = 0 'days  to = 20 'days }

observations {
  cases {
    columns       { time : time, cases : count }
    covers        = closing_at(time, 1 'days)
    projected     = incidence(infection_a)
    emit_schedule = every 1 'days
    cases ~ poisson(rate = rho * projected + 0.5)
  }
  prevalent {
    columns       { time : time, prevalent : count }
    projected     = I
    emit_schedule = every 1 'days
    prevalent ~ poisson(rate = projected + 0.5)
  }
}
"#;

const PATCH: &str = r#"
time_unit = 'days
dimensions { patch = [p0, p1] }
compartments { S, I, R }
stratify(by = patch)
parameters { beta : rate in [0.01, 2.0]  gamma : rate in [0.01, 1.0]  rho : probability in [0.05, 0.95] }
tables {
  w : patch = external("w")
  W : patch × patch = [[0.0, 0.5],
                       [0.5, 0.0]]
}
let N[l in patch] = S[l] + I[l] + R[l]
transitions {
  infection[l in patch] : S[l] --> I[l] @ beta * w[l] * S[l] * sum(q in patch, W[l, q] * I[q] / N[q])
  recovery[l in patch]  : I[l] --> R[l] @ gamma * I[l]
}
init {
  S[p0] = 990  I[p0] = 10
  S[p1] = 1000  I[p1] = 5
}
observations {
  cases {
    columns       { time : time, cases : count }
    covers        = closing_at(time, 1 'days)
    projected     = sum(l in patch, incidence(infection[l]))
    emit_schedule = every 1 'days
    cases         ~ poisson(rate = rho * projected + 0.5)
  }
}
simulate { from = 0 'days  to = 10 'days }
"#;

/// A fit config over `sir.ir.json`'s `cases` stream; `extra` lands in it
/// before `[data.observations]`.
fn fit_toml(dir: &std::path::Path, extra: &str) -> String {
    format!(
        "[model]\ncamdl = \"{}\"\n{extra}[data.observations]\ncases = \"{}\"\n\
         [estimate]\nbeta_a = {{ bounds = [0.05, 2.0], start = 0.4 }}\n\
         [fixed]\nbeta_b = 0.2\ngamma = 0.2\nrho = 0.5\n\
         [method]\nalgorithm = \"if2\"\nbackend = \"chain_binomial\"\nchains = 1\nparticles = 20\niterations = 2\ncooling = 0.9\n",
        dir.join("sir.ir.json").display(),
        dir.join("cases.tsv").display(),
    )
}

/// The fixture files for the directory `d`.
fn files(d: &std::path::Path) -> Vec<(&'static str, String)> {
    let series = |f: &dyn Fn(u32) -> u32, from: u32, col: &str| {
        (from..=20).fold(format!("time\t{col}\n"), |s, t| s + &format!("{t}\t{}\n", f(t)))
    };
    vec![
        ("sir.camdl", SIR.to_string()),
        ("sir_other.camdl", SIR.replace("S = 990  I = 10", "S = 980  I = 20")),
        ("patch.camdl", PATCH.to_string()),
        ("p.toml", "beta_a = 0.4\nbeta_b = 0.2\ngamma = 0.2\nrho = 0.5\n".into()),
        ("p2.toml", "beta_a = 0.5\nbeta_b = 0.2\ngamma = 0.2\nrho = 0.5\n".into()),
        ("patch.toml", "beta = 0.4\ngamma = 0.2\nrho = 0.5\n".into()),
        ("w1.tsv", "1.0\n0.5\n".into()),
        ("w2.tsv", "1.0\n0.7\n".into()),
        ("cases.tsv", series(&|t| 3 + t, 1, "cases")),
        ("cases2.tsv", series(&|t| 4 + t, 1, "cases")),
        ("prevalent.tsv", series(&|t| 10 + 2 * t, 0, "prevalent")),
        ("both.tsv", (1..=20).fold("time\tcases\tprevalent\n".to_string(), |s, t| s + &format!("{t}\t{}\t{}\n", 3 + t, 10 + 2 * t))),
        ("patch_cases.tsv", (1..=10).fold("time\tcases\n".to_string(), |s, t| s + &format!("{t}\t{}\n", 2 + t))),
        ("fit_plain.toml", fit_toml(d, "")),
        ("fit_adr.toml", fit_toml(d, "[config]\nallow_degenerate_rates = true\n")),
        ("fit_holdout.toml", fit_toml(d, "[data]\nholdout_after = 15.0\n")),
    ]
}

/// The fixture, shared by the tests that hold it at once and created again
/// when none does.
fn fx() -> Arc<Fixture> {
    static LIVE: Mutex<Weak<Fixture>> = Mutex::new(Weak::new());
    let mut live = LIVE.lock().unwrap_or_else(|e| e.into_inner());
    if let Some(fx) = live.upgrade() {
        return fx;
    }
    let fx = Arc::new(Fixture::create("pfilter_input_classes", files, &["sir.camdl", "sir_other.camdl", "patch.camdl"]));
    *live = Arc::downgrade(&fx);
    fx
}

/// `camdl pfilter sir.ir.json --params p.toml --data cases=cases.tsv` at
/// twenty particles, seed 1, with `extra`.
fn pf(fx: &Arc<Fixture>, extra: &[&str]) -> Invocation {
    pf_on(fx, "sir.ir.json", extra)
}

fn pf_on(fx: &Arc<Fixture>, model: &str, extra: &[&str]) -> Invocation {
    let (m, p, d) = (fx.path(model), fx.path("p.toml"), format!("cases={}", fx.path("cases.tsv")));
    Invocation::new(fx, &["pfilter", &m, "--params", &p, "--data", &d, "--particles", "20", "--seed", "1"]).with(extra)
}

/// The two-patch model from source, its `w` table from `w`.
fn patch(fx: &Arc<Fixture>, w: &str, extra: &[&str]) -> Invocation {
    let (m, p, t, d) = (fx.path("patch.camdl"), fx.path("patch.toml"), format!("w={}", fx.path(w)), fx.path("patch_cases.tsv"));
    Invocation::new(fx, &["pfilter", &m, "--params", &p, "--table", &t, "--data", &d, "--particles", "20", "--seed", "1"]).with(extra)
}

/// A pfilter of `sir.ir.json` from the fit config `config`.
fn pf_fit(fx: &Arc<Fixture>, config: &str, extra: &[&str]) -> Invocation {
    let (m, p, f) = (fx.path("sir.ir.json"), fx.path("p.toml"), fx.path(config));
    Invocation::new(fx, &["pfilter", &m, "--params", &p, "--fit", &f, "--particles", "20", "--seed", "1"]).with(extra)
}

fn level(input: &str) -> String {
    declared_level("pub const PFILTER", input, "model")
}

fn assert_rekeys(input: &str, a: &Invocation, b: &Invocation) {
    assert_rekeyed(KIND, input, &level(input), &run(a), &run(b));
}

fn assert_same(a: &Invocation, b: &Invocation) {
    assert_same_result(KIND, &run(a), &run(b));
}

fn assert_unaffected(input: &str, a: &Invocation, b: &Invocation) {
    assert_leaf_unaffected(KIND, input, &run(a), &run(b));
}

// ─── keyed ───────────────────────────────────────────────────────────────────

#[test]
fn keyed_model() {
    let fx = fx();
    assert_rekeys("model", &pf(&fx, &[]), &pf_on(&fx, "sir_other.ir.json", &[]));
}

#[test]
fn keyed_particles() {
    let fx = fx();
    assert_rekeys("particles", &pf(&fx, &[]), &pf(&fx, &[]).set("--particles", "30"));
}

#[test]
fn keyed_data() {
    let fx = fx();
    let other = format!("cases={}", fx.path("cases2.tsv"));
    let mut b = pf(&fx, &[]);
    b.args[5] = other;
    assert_rekeys("data", &pf(&fx, &[]), &b);
}

/// `--fit` supplies the problem: here its data and its `[config]
/// allow_degenerate_rates`, which the flags do not set.
#[test]
fn keyed_fit() {
    let fx = fx();
    assert_rekeys("fit", &pf(&fx, &[]), &pf_fit(&fx, "fit_adr.toml", &[]));
}

#[test]
fn keyed_replicates() {
    let fx = fx();
    assert_rekeys("replicates", &pf(&fx, &[]), &pf(&fx, &["--replicates", "2"]));
}

/// Under the fit's `holdout_after`, `--save-final-state` binds only the
/// training rows (gh#1138).
#[test]
fn keyed_save_final_state() {
    let fx = fx();
    assert_rekeys(
        "save_final_state",
        &pf_fit(&fx, "fit_holdout.toml", &[]),
        &pf_fit(&fx, "fit_holdout.toml", &["--save-final-state", "state.tsv"]),
    );
}

#[test]
fn keyed_params() {
    let fx = fx();
    let mut b = pf(&fx, &[]);
    b.args[3] = fx.path("p2.toml");
    assert_rekeys("params", &pf(&fx, &[]), &b);
}

#[test]
fn keyed_param() {
    let fx = fx();
    assert_rekeys("param", &pf(&fx, &[]), &pf(&fx, &["--param", "beta_a=0.45"]));
}

#[test]
fn keyed_table() {
    let fx = fx();
    assert_rekeys("table", &patch(&fx, "w1.tsv", &[]), &patch(&fx, "w2.tsv", &[]));
}

/// `--scenario` refuses `--params` beside it (gh#194), so both sides take
/// their values from a scenario that sets every parameter.
#[test]
fn keyed_scenario() {
    let fx = fx();
    let (m, d) = (fx.path("sir.ir.json"), format!("cases={}", fx.path("cases.tsv")));
    let base = Invocation::new(&fx, &["pfilter", &m, "--data", &d, "--particles", "20", "--seed", "1"]);
    assert_rekeys("scenario", &base.with(&["--scenario", "set_all"]), &base.with(&["--scenario", "set_fast"]));
}

#[test]
fn keyed_enable() {
    let fx = fx();
    assert_rekeys("enable", &pf(&fx, &[]), &pf(&fx, &["--enable", "sia"]));
}

#[test]
fn keyed_disable() {
    let fx = fx();
    assert_rekeys("disable", &pf(&fx, &["--enable", "sia"]), &pf(&fx, &["--disable", "sia"]));
}

#[test]
fn keyed_dt() {
    let fx = fx();
    assert_rekeys("dt", &pf(&fx, &[]), &pf(&fx, &["--dt", "0.5"]));
}

#[test]
fn keyed_seed() {
    let fx = fx();
    assert_rekeys("seed", &pf(&fx, &[]), &pf(&fx, &[]).set("--seed", "2"));
}

#[test]
fn keyed_allow_degenerate_rates() {
    let fx = fx();
    assert_rekeys("allow_degenerate_rates", &pf(&fx, &[]), &pf(&fx, &["--allow-degenerate-rates"]));
}

#[test]
fn keyed_allow_forcing_extrapolation() {
    let fx = fx();
    assert_rekeys("allow_forcing_extrapolation", &pf(&fx, &[]), &pf(&fx, &["--allow-forcing-extrapolation"]));
}

#[test]
fn keyed_binomial() {
    let fx = fx();
    assert_rekeys("binomial", &pf(&fx, &[]), &pf(&fx, &["--binomial", "btrs"]));
}

/// `--obs` names the stream a single `--data PATH` binds.
#[test]
fn keyed_obs() {
    let fx = fx();
    let (m, p, d) = (fx.path("sir.ir.json"), fx.path("p.toml"), fx.path("both.tsv"));
    let base = Invocation::new(&fx, &["pfilter", &m, "--params", &p, "--data", &d, "--particles", "20", "--seed", "1"]);
    assert_rekeys("obs", &base.with(&["--obs", "cases"]), &base.with(&["--obs", "prevalent"]));
}

/// `--no-licm` changes the IR the compiler emits for a model with a
/// parameter-only rate factor; the compile is given it, so the model level
/// moves.
#[test]
fn keyed_no_licm() {
    let fx = fx();
    assert_rekeys("no_licm", &pf_on(&fx, "sir.camdl", &[]), &pf_on(&fx, "sir.camdl", &["--no-licm"]));
}

#[test]
fn keyed_env_no_licm() {
    let fx = fx();
    assert_rekeys("env:no_licm", &pf_on(&fx, "sir.camdl", &[]), &pf_on(&fx, "sir.camdl", &[]).env("CAMDL_NO_LICM", "1"));
}

#[test]
fn keyed_env_no_constant_fold() {
    let fx = fx();
    assert_rekeys(
        "env:no_constant_fold",
        &patch(&fx, "w1.tsv", &[]),
        &patch(&fx, "w1.tsv", &[]).env("CAMDL_NO_CONSTANT_FOLD", "1"),
    );
}

/// An engine switch keys the pfilter, and the leaf's record names it beside
/// the build; the log-likelihood is the default path's.
#[test]
fn keyed_env_eval_unresolved() {
    let fx = fx();
    let (a, b) = (run(&pf(&fx, &[])), run(&pf(&fx, &[]).env("CAMDL_EVAL_UNRESOLVED", "1")));
    assert_rekeyed(KIND, "env:eval_unresolved", &level("env:eval_unresolved"), &a, &b);
    let (plain, switched) = (a.of(KIND).into_values().next().unwrap().clone(), b.of(KIND).into_values().next().unwrap().clone());
    assert_eq!(switched.record["engine_version"], plain.record["engine_version"], "engine_version is the build");
    assert_eq!(switched.record["engine_switches"], serde_json::json!([{ "name": "CAMDL_EVAL_UNRESOLVED" }]));
    assert!(plain.record.get("engine_switches").is_none(), "a default run's record has no switch list");
    assert_eq!(switched.files, plain.files, "the same stored bytes under the alternative path");
}

#[test]
fn keyed_env_no_binding_cache() {
    let fx = fx();
    assert_rekeys("env:no_binding_cache", &pf(&fx, &[]), &pf(&fx, &[]).env("CAMDL_NO_BINDING_CACHE", "1"));
}

#[test]
fn keyed_env_experimental() {
    let fx = fx();
    let (a, b) = (run(&pf(&fx, &[])), run(&pf(&fx, &[]).env("CAMDL_EXPERIMENTAL_DIFFERENTIAL_TEST", "1")));
    assert_rekeyed(KIND, "env:experimental", &level("env:experimental"), &a, &b);
    let leaf = b.of(KIND).into_values().next().unwrap().clone();
    assert_eq!(
        leaf.record["engine_switches"],
        serde_json::json!([{ "name": "CAMDL_EXPERIMENTAL_DIFFERENTIAL_TEST", "value": "1" }])
    );
}

// ─── execution ───────────────────────────────────────────────────────────────
//
// Every execution and presentation case runs on both of the filter's paths,
// one run and `--replicates 3` ([`bases_on`]), except the outputs that
// describe one run, which the replicate path refuses (gh#686).

/// Both paths of the filter on `model`: one run, and the replicate path.
fn bases_on(fx: &Arc<Fixture>, model: &str) -> [Invocation; 2] {
    [pf_on(fx, model, &[]), pf_on(fx, model, &["--replicates", "3"])]
}

fn bases(fx: &Arc<Fixture>) -> [Invocation; 2] {
    bases_on(fx, "sir.ir.json")
}

/// The filter runs its particles, and the replicate path its replicates, on
/// the scoped pool `--parallel` sizes.
#[test]
fn execution_parallel() {
    let fx = fx();
    for b in bases(&fx) {
        assert_same(&b.with(&["--parallel", "1"]), &b.with(&["--parallel", "4"]));
    }
}

#[test]
fn execution_time_format() {
    let fx = fx();
    for b in bases(&fx) {
        assert_same(&b, &b.with(&["--time-format", "numeric"]));
    }
}

#[test]
fn execution_pf_max_substeps() {
    let fx = fx();
    for b in bases(&fx) {
        assert_same(&b, &b.with(&["--pf-max-substeps", "100000000"]));
    }
}

#[test]
fn execution_no_ir_cache() {
    let fx = fx();
    for b in bases_on(&fx, "sir.camdl") {
        assert_same(&b, &b.with(&["--no-ir-cache"]));
    }
}

#[test]
fn execution_env_no_ir_cache() {
    let fx = fx();
    for b in bases_on(&fx, "sir.camdl") {
        assert_same(&b, &b.clone().env("CAMDL_NO_IR_CACHE", "1"));
    }
}

#[test]
fn execution_env_ir_cache_dir() {
    let fx = fx();
    let elsewhere = tempfile::tempdir().unwrap();
    for b in bases_on(&fx, "sir.camdl") {
        assert_same(&b, &b.clone().env("CAMDL_IR_CACHE_DIR", &elsewhere.path().to_string_lossy()));
    }
}

#[test]
fn execution_env_xdg_cache_home() {
    let fx = fx();
    let (x, y) = (tempfile::tempdir().unwrap(), tempfile::tempdir().unwrap());
    for b in bases_on(&fx, "sir.camdl") {
        let under = |d: &std::path::Path| b.clone().without("CAMDL_IR_CACHE_DIR").env("XDG_CACHE_HOME", &d.to_string_lossy());
        assert_same(&under(x.path()), &under(y.path()));
    }
}

#[test]
fn execution_env_home() {
    let fx = fx();
    let (x, y) = (tempfile::tempdir().unwrap(), tempfile::tempdir().unwrap());
    for b in bases_on(&fx, "sir.camdl") {
        let under = |d: &std::path::Path| b.clone().without("CAMDL_IR_CACHE_DIR").env("HOME", &d.to_string_lossy());
        assert_same(&under(x.path()), &under(y.path()));
    }
}

#[test]
fn execution_env_camdlc_path() {
    let fx = fx();
    let camdlc = fx.camdlc.to_string_lossy().into_owned();
    for b in bases_on(&fx, "sir.camdl") {
        assert_same(&b, &b.clone().env("CAMDLC_PATH", &camdlc));
    }
}

#[test]
fn execution_env_camdlc() {
    let fx = fx();
    let bin = tempfile::tempdir().unwrap();
    std::os::unix::fs::symlink(&fx.camdlc, bin.path().join("camdlc")).unwrap();
    let path = format!("{}:{}", bin.path().display(), std::env::var("PATH").unwrap_or_default());
    for b in bases_on(&fx, "sir.camdl") {
        assert_same(&b, &b.clone().without("CAMDLC").env("PATH", &path));
    }
}

#[test]
fn execution_env_output_dir() {
    let fx = fx();
    for b in bases(&fx) {
        let (x, y) = (tempfile::tempdir().unwrap(), tempfile::tempdir().unwrap());
        let elsewhere = y.path().join("elsewhere");
        let moved = b.clone().env("CAMDL_OUTPUT_DIR", &elsewhere.to_string_lossy());
        let (sx, sy) = (run_into(x.path(), &b), run_into(y.path(), &moved));
        assert!(sy.leaves.iter().all(|l| l.dir.starts_with(&elsewhere)), "the store root moved");
        assert_same_result(KIND, &sx, &sy);
    }
}

// ─── presentation ────────────────────────────────────────────────────────────

#[test]
fn presentation_trace() {
    let fx = fx();
    assert_unaffected("--trace", &pf(&fx, &[]), &pf(&fx, &["--trace", "trace.tsv"]));
}

#[test]
fn presentation_pf_health() {
    let fx = fx();
    assert_unaffected("--pf-health", &pf(&fx, &[]), &pf(&fx, &["--pf-health", "health.tsv"]));
}

/// On one run the log-likelihood, on the replicate path each replicate's.
#[test]
fn presentation_output() {
    let fx = fx();
    for b in bases(&fx) {
        assert_unaffected("--output", &b, &b.with(&["--output", "ll.txt"]));
    }
}

#[test]
fn presentation_save_paths() {
    let fx = fx();
    assert_unaffected("--save-paths", &pf(&fx, &[]), &pf(&fx, &["--save-paths", "paths.tsv"]));
}

#[test]
fn presentation_n_paths() {
    let fx = fx();
    assert_unaffected(
        "--n-paths",
        &pf(&fx, &["--save-paths", "paths.tsv"]),
        &pf(&fx, &["--save-paths", "paths.tsv", "--n-paths", "3"]),
    );
}

#[test]
fn presentation_save_filtering() {
    let fx = fx();
    assert_unaffected("--save-filtering", &pf(&fx, &[]), &pf(&fx, &["--save-filtering", "filtering.tsv"]));
}

#[test]
fn presentation_save_prequential() {
    let fx = fx();
    assert_unaffected("--save-prequential", &pf(&fx, &[]), &pf(&fx, &["--save-prequential", "preq"]));
}

#[test]
fn presentation_score_from() {
    let fx = fx();
    assert_unaffected(
        "--score-from",
        &pf(&fx, &["--save-prequential", "preq"]),
        &pf(&fx, &["--save-prequential", "preq", "--score-from", "5"]),
    );
}

#[test]
fn presentation_no_save_samples() {
    let fx = fx();
    assert_unaffected(
        "--no-save-samples",
        &pf(&fx, &["--save-prequential", "preq"]),
        &pf(&fx, &["--save-prequential", "preq", "--no-save-samples"]),
    );
}

/// `--no-store` asks for the evaluation without a stored leaf: none is
/// written, and what is printed is the stored run's.
#[test]
fn presentation_no_store() {
    let fx = fx();
    for b in bases(&fx) {
        let store = tempfile::tempdir().unwrap();
        let out = run_in(store.path(), &b.with(&["--no-store"]));
        assert!(out.status.success(), "{}", String::from_utf8_lossy(&out.stderr));
        assert!(leaves(store.path()).is_empty(), "--no-store writes no leaf");
        let stored = run_in(tempfile::tempdir().unwrap().path(), &b);
        assert_eq!(out.stdout, stored.stdout, "the same log-likelihood");
    }
}

#[test]
fn presentation_verbosity() {
    let fx = fx();
    for b in bases(&fx) {
        assert_unaffected("--verbosity", &b, &b.with(&["--verbosity", "debug"]));
    }
}

#[test]
fn presentation_progress() {
    let fx = fx();
    for b in bases(&fx) {
        assert_unaffected("--progress", &b.with(&["--progress", "none"]), &b.with(&["--progress", "plain"]));
    }
}

#[test]
fn presentation_no_progress() {
    let fx = fx();
    for b in bases(&fx) {
        assert_unaffected("--no-progress", &b.with(&["--progress", "plain"]), &b.with(&["--no-progress"]));
    }
}

#[test]
fn presentation_env_trace_steps() {
    let fx = fx();
    for b in bases(&fx) {
        assert_unaffected("CAMDL_TRACE_STEPS", &b, &b.clone().env("CAMDL_TRACE_STEPS", "1"));
    }
}

#[test]
fn presentation_env_no_color() {
    let fx = fx();
    for b in bases(&fx) {
        assert_unaffected("NO_COLOR", &b, &b.clone().env("NO_COLOR", "1"));
    }
}

#[test]
fn presentation_env_clicolor_force() {
    let fx = fx();
    for b in bases(&fx) {
        assert_unaffected("CLICOLOR_FORCE", &b, &b.clone().env("CLICOLOR_FORCE", "1"));
    }
}

// ─── the --fit handle ────────────────────────────────────────────────────────

/// `--fit` takes the handle forms `fit predict` takes. A stored fit named by
/// its leaf directory, its problem folder, its run-id prefix or its label
/// names the problem its archived config states, which here is the file's
/// (no command-line option changed it): every form scores the one
/// computation, and the `fit.toml` names the same problem without consulting
/// the store.
#[test]
fn every_fit_handle_names_the_fits_problem() {
    let fx = fx();
    let store = tempfile::tempdir().unwrap();
    let fitted = run_into(store.path(), &Invocation::new(&fx, &["fit", "run", &fx.path("fit_plain.toml"), "--label", "plain"]));
    let leaf = fitted.of("fit_method").into_values().next().expect("the fit's leaf").clone();
    let segment: PathBuf = leaf.dir.parent().unwrap().parent().unwrap().to_path_buf();
    let handles = [
        fx.path("fit_plain.toml"),
        leaf.dir.to_string_lossy().into_owned(),
        segment.to_string_lossy().into_owned(),
        leaf.run_id[..12].to_string(),
        "@plain".to_string(),
    ];
    let mut ids = Vec::new();
    for h in &handles {
        let mut inv = pf_fit(&fx, "fit_plain.toml", &[]);
        let at = inv.args.iter().position(|a| a == "--fit").unwrap() + 1;
        inv.args[at] = h.clone();
        let s = run_into(store.path(), &inv);
        assert!(!s.stderr.contains("Is a directory"), "--fit {h}:\n{}", s.stderr);
        ids.push(s.ids(KIND));
    }
    assert_eq!(ids.iter().collect::<std::collections::BTreeSet<_>>().len(), 1, "every handle names one problem: {ids:?}");
}

/// A fit config not named `*.toml` is still a config: an existing regular
/// file names the problem it states, as it did before `--fit` took handles.
#[test]
fn a_fit_config_of_any_name_is_a_config() {
    let fx = fx();
    let store = tempfile::tempdir().unwrap();
    let cfg = store.path().join("fit_plain.cfg");
    std::fs::copy(fx.path("fit_plain.toml"), &cfg).unwrap();
    let by_toml = run_into(store.path(), &pf_fit(&fx, "fit_plain.toml", &[])).ids(KIND);
    let by_cfg = run_into(store.path(), &pf_fit(&fx, "fit_plain.toml", &[]).set("--fit", &cfg.to_string_lossy())).ids(KIND);
    assert!(!by_toml.is_empty());
    assert_eq!(by_toml, by_cfg, "the config names the same problem whatever its file name");
}

/// A point of `fit run --sweep` named by its leaf directory, its problem
/// folder or its run-id prefix reads the point's problem from its fit
/// folder's `fit.resolved.toml`, and stores the bytes a `pfilter` of
/// a fit file that writes the point's value stores. `pfilter`
/// reads none of the fit's `[fixed]` (the point it scores is `--params`), so
/// the file's own value stores the same bytes too: the swept value cannot
/// change a `pfilter` result, and this case shows every form is accepted and
/// names the point's problem.
#[test]
fn a_sweep_point_handle_names_the_points_problem() {
    let fx = fx();
    let store = tempfile::tempdir().unwrap();
    let fitted = run_into(store.path(), &Invocation::new(&fx, &["fit", "run", &fx.path("fit_plain.toml"), "--sweep", "rho=0.5,0.6"]));
    let (_, other) = sweep_points(&fitted, "rho", 0.5);
    assert_eq!(segment_fixed(&other.dir, "rho"), Some(0.6));
    let point_file = store.path().join("fit_rho.toml");
    let plain = std::fs::read_to_string(fx.path("fit_plain.toml")).unwrap();
    assert!(plain.contains("rho = 0.5\n"));
    std::fs::write(&point_file, plain.replace("rho = 0.5\n", "rho = 0.6\n")).unwrap();
    let written = run(&pf_fit(&fx, "fit_plain.toml", &[]).set("--fit", &point_file.to_string_lossy())).bytes(KIND);
    assert!(!written.is_empty());
    let segment = other.dir.parent().unwrap().parent().unwrap().to_string_lossy().into_owned();
    for h in [other.dir.to_string_lossy().into_owned(), segment, other.run_id[..12].to_string()] {
        let s = run_into(store.path(), &pf_fit(&fx, "fit_plain.toml", &[]).set("--fit", &h));
        assert_eq!(s.bytes(KIND), written, "--fit {h}: the pfilter of the point's problem");
    }
}
