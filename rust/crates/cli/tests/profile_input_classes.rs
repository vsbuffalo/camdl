//! A case for every input of `camdl profile`, checked as its class predicts
//! (`rust/crates/cli/src/input_class.rs`, `PROFILE`; proposal
//! `docs/dev/proposals/2026-11-17-honest-input-addressed-store.md` phase 3
//! "Tests"). The runner is `input_class_runner`.
//!
//! Each case is one test, named `<class>_<input>`; an environment variable's
//! case is `<class>_env_<field>`; `input_class::tests::every_profile_input_has_a_case`
//! fails if an input has none. What each class asserts, over the profile point
//! leaves:
//!
//! - `keyed(level)`: changed alone, every point's run id changes, and so does
//!   its hash at the level the class table names (`profile`, `method` or
//!   `seed`). An environment variable the table keys at a simulation's `model`
//!   level enters a profile at its `profile` level, which holds the model and
//!   the engine.
//! - `execution`: the run ids and the stored bytes stay the same.
//! - `expands`: the points common to both values keep their run ids.
//! - `presentation`, `provenance`: the points are unaffected.
//! - `removed`: the old spelling is refused, naming it, and nothing is stored.
//!
//! These are regression evidence on small fixtures, not proof. Each profile
//! is two cells of one start, each an IF2 run of ten particles over two
//! iterations (or a one-parameter ODE optimum, for nl-*).
//!
//! The fixtures: `sir.camdl`, an SIR model with a vaccination intervention, a
//! scenario that enables it, two observed streams and a parameter-only rate
//! factor that loop-invariant code motion hoists; `patch.camdl`, a two-patch
//! model with an `external` table and a coupling matrix whose structural zeros
//! the constant fold drops; two stored fits of the fit config, a one-chain
//! IF2 (a `from_mle` source with no verdict) and a two-chain PMMH (a
//! `from_posterior` source).

mod input_class_runner;

use std::path::PathBuf;
use std::sync::{Arc, Mutex, Weak};

use input_class_runner::*;

const KIND: &str = "profile_point";

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
  with_sia { enable = [sia] }
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

/// A fit config over `sir.ir.json`'s `cases` stream, `beta_a` and `gamma`
/// estimated, with the `[method]` table `method`.
fn fit_toml(d: &std::path::Path, rho: f64, method: &str) -> String {
    format!(
        "[model]\ncamdl = \"{}\"\n[data.observations]\ncases = \"{}\"\n\
         [estimate]\nbeta_a = {{ bounds = [0.05, 2.0], start = 0.4, prior = {{ uniform = {{}} }} }}\ngamma = {{ bounds = [0.01, 1.0], start = 0.2, prior = {{ uniform = {{}} }} }}\n\
         [fixed]\nbeta_b = 0.2\nrho = {rho}\n[method]\n{method}",
        d.join("sir.ir.json").display(),
        d.join("cases.tsv").display(),
    )
}

const IF2: &str = "algorithm = \"if2\"\nbackend = \"chain_binomial\"\nchains = 1\nparticles = 20\niterations = 2\ncooling = 0.9\n";
const PMMH: &str = "algorithm = \"pmmh\"\nbackend = \"chain_binomial\"\nchains = 2\nparticles = 20\niterations = 20\nburn_in = 5\n";

fn files(d: &std::path::Path) -> Vec<(&'static str, String)> {
    let series = |f: &dyn Fn(u32) -> u32, from: u32, col: &str| {
        (from..=20).fold(format!("time\t{col}\n"), |s, t| s + &format!("{t}\t{}\n", f(t)))
    };
    vec![
        ("sir.camdl", SIR.to_string()),
        ("sir_other.camdl", SIR.replace("S = 990  I = 10", "S = 980  I = 20")),
        ("patch.camdl", PATCH.to_string()),
        ("start.toml", "beta_a = 0.4\n".into()),
        ("start_patch.toml", "beta = 0.4\n".into()),
        ("ff1.toml", "beta_b = 0.2\n".into()),
        ("ff2.toml", "beta_b = 0.25\n".into()),
        ("w1.tsv", "1.0\n0.5\n".into()),
        ("w2.tsv", "1.0\n0.7\n".into()),
        ("cases.tsv", series(&|t| 3 + t, 1, "cases")),
        ("cases2.tsv", series(&|t| 4 + t, 1, "cases")),
        ("both.tsv", (1..=20).fold("time\tcases\tprevalent\n".to_string(), |s, t| s + &format!("{t}\t{}\t{}\n", 3 + t, 10 + 2 * t))),
        ("patch_cases.tsv", (1..=10).fold("time\tcases\n".to_string(), |s, t| s + &format!("{t}\t{}\n", 2 + t))),
        ("fit_plain.toml", fit_toml(d, 0.5, IF2)),
        ("fit_rho.toml", fit_toml(d, 0.6, IF2)),
        ("fit_pmmh.toml", fit_toml(d, 0.5, PMMH)),
    ]
}

/// The fixture and the two stored fits the `--starts` cases read.
struct Fx {
    fx: Arc<Fixture>,
    /// A completed one-chain IF2 leaf: a `from_mle` source with no verdict.
    mle_source: String,
    /// A completed two-chain PMMH leaf: a `from_posterior` source.
    posterior_source: String,
}

fn fx() -> Arc<Fx> {
    static LIVE: Mutex<Weak<Fx>> = Mutex::new(Weak::new());
    let mut live = LIVE.lock().unwrap_or_else(|e| e.into_inner());
    if let Some(fx) = live.upgrade() {
        return fx;
    }
    let fx = Arc::new(Fixture::create("profile_input_classes", files, &["sir.camdl", "sir_other.camdl", "patch.camdl"]));
    let sources = fx.dir.join("sources");
    std::fs::create_dir_all(&sources).unwrap();
    let source = |config: &str| -> String {
        let stored = run_into(&sources, &Invocation::new(&fx, &["fit", "run", &fx.path(config)]));
        let leaf = stored.of("fit_method").into_values().find(|l| {
            l.record["inputs"]["method"].as_str() == Some(if config.contains("pmmh") { "pmmh" } else { "if2" })
        });
        leaf.expect("the source fit's leaf").dir.to_string_lossy().into_owned()
    };
    let mle_source = source("fit_plain.toml");
    let posterior_source = source("fit_pmmh.toml");
    let fx = Arc::new(Fx { fx, mle_source, posterior_source });
    *live = Arc::downgrade(&fx);
    fx
}

const CELLS: &[&str] = &["--sweep", "gamma=0.15,0.2", "--n-starts", "1", "--seed", "1"];
const IF2_CELL: &[&str] = &["--particles", "10", "--iterations", "2"];

/// `camdl profile <model> --fit fit_plain.toml` over two cells, IF2.
fn prof_on(fx: &Fx, model: &str, extra: &[&str]) -> Invocation {
    let (m, f) = (fx.fx.path(model), fx.fx.path("fit_plain.toml"));
    Invocation::new(&fx.fx, &["profile", &m, "--fit", &f]).with(CELLS).with(IF2_CELL).with(extra)
}

fn prof(fx: &Fx, extra: &[&str]) -> Invocation {
    prof_on(fx, "sir.ir.json", extra)
}

/// The same with nl-sbplx (or `algorithm`) on the ODE backend.
fn prof_nl(fx: &Fx, algorithm: &str, extra: &[&str]) -> Invocation {
    prof_nl_on(fx, "sir.ir.json", algorithm, extra)
}

fn prof_nl_on(fx: &Fx, model: &str, algorithm: &str, extra: &[&str]) -> Invocation {
    let (m, f) = (fx.fx.path(model), fx.fx.path("fit_plain.toml"));
    Invocation::new(&fx.fx, &["profile", &m, "--fit", &f, "--algorithm", algorithm, "--backend", "ode"]).with(CELLS).with(extra)
}

/// A profile from the command line alone: `rho` pinned by `--fixed`,
/// `beta_b` by `--fixed-file ff1.toml`, `beta_a` estimated from
/// `start.toml`, the `cases` stream bound by `--data`.
fn prof_cli(fx: &Fx, extra: &[&str]) -> Invocation {
    let p = |n: &str| fx.fx.path(n);
    let (m, ff, st, d) = (p("sir.ir.json"), p("ff1.toml"), format!("from_params={}", p("start.toml")), format!("cases={}", p("cases.tsv")));
    Invocation::new(&fx.fx, &["profile", &m, "--fixed", "rho=0.5", "--fixed-file", &ff, "--starts", &st, "--rw-sd", "beta_a=0.05", "--data", &d])
        .with(CELLS)
        .with(IF2_CELL)
        .with(extra)
}

/// The two-patch model from source, its `w` table from `w`.
fn patch(fx: &Fx, w: &str, extra: &[&str]) -> Invocation {
    let p = |n: &str| fx.fx.path(n);
    let (m, st, t, d) = (p("patch.camdl"), format!("from_params={}", p("start_patch.toml")), format!("w={}", p(w)), p("patch_cases.tsv"));
    Invocation::new(&fx.fx, &["profile", &m, "--fixed", "rho=0.5", "--starts", &st, "--table", &t, "--data", &d, "--rw-sd", "beta=0.05"])
        .with(&["--sweep", "gamma=0.15,0.2", "--n-starts", "1", "--seed", "1"])
        .with(IF2_CELL)
        .with(extra)
}

fn level(input: &str) -> String {
    declared_level("pub const PROFILE", input, "profile")
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
    assert_rekeys("model", &prof(&fx, &[]), &prof_on(&fx, "sir_other.ir.json", &[]));
}

#[test]
fn keyed_data() {
    let fx = fx();
    let other = format!("cases={}", fx.fx.path("cases2.tsv"));
    assert_rekeys("data", &prof_cli(&fx, &[]), &prof_cli(&fx, &[]).set("--data", &other));
}

#[test]
fn keyed_fit() {
    let fx = fx();
    assert_rekeys("fit", &prof(&fx, &[]), &prof(&fx, &[]).set("--fit", &fx.fx.path("fit_rho.toml")));
}

/// The grid is folded into every point's base: a different grid is a
/// different profile.
#[test]
fn keyed_sweep() {
    let fx = fx();
    assert_rekeys("sweep", &prof(&fx, &[]), &prof(&fx, &[]).set("--sweep", "gamma=0.15,0.25"));
}

#[test]
fn keyed_particles() {
    let fx = fx();
    assert_rekeys("particles", &prof(&fx, &[]), &prof(&fx, &[]).set("--particles", "12"));
}

#[test]
fn keyed_iterations() {
    let fx = fx();
    assert_rekeys("iterations", &prof(&fx, &[]), &prof(&fx, &[]).set("--iterations", "3"));
}

#[test]
fn keyed_n_starts() {
    let fx = fx();
    assert_rekeys("n_starts", &prof(&fx, &[]), &prof(&fx, &[]).set("--n-starts", "2"));
}

#[test]
fn keyed_starts() {
    let fx = fx();
    assert_rekeys("starts", &prof(&fx, &[]), &prof(&fx, &["--starts", "lhs"]));
}

#[test]
fn keyed_starts_chains() {
    let fx = fx();
    let source = format!("from_posterior={}", fx.posterior_source);
    // A subset is judged by its own R̂, which a fixture this short need not
    // pass: the verdict is lifted on both sides.
    let base = prof(&fx, &["--starts", &source, "--allow-nonconverged-source"]);
    assert_rekeys("starts_chains", &base, &base.with(&["--starts-chains", "0"]));
}

#[test]
fn keyed_cooling() {
    let fx = fx();
    assert_rekeys("cooling", &prof(&fx, &[]), &prof(&fx, &["--cooling", "0.8"]));
}

#[test]
fn keyed_cooling_target_iters() {
    let fx = fx();
    assert_rekeys("cooling_target_iters", &prof(&fx, &[]), &prof(&fx, &["--cooling-target-iters", "2"]));
}

#[test]
fn keyed_rw_sd() {
    let fx = fx();
    assert_rekeys("rw_sd", &prof(&fx, &[]), &prof(&fx, &["--rw-sd", "beta_a=0.07"]));
}

#[test]
fn keyed_algorithm() {
    let fx = fx();
    assert_rekeys("algorithm", &prof_nl(&fx, "nl-sbplx", &[]), &prof_nl(&fx, "nl-bobyqa", &[]));
}

#[test]
fn keyed_fixed_cli() {
    let fx = fx();
    assert_rekeys("fixed_cli", &prof_cli(&fx, &[]), &prof_cli(&fx, &[]).set("--fixed", "rho=0.6"));
}

#[test]
fn keyed_fixed_files() {
    let fx = fx();
    assert_rekeys("fixed_files", &prof_cli(&fx, &[]), &prof_cli(&fx, &[]).set("--fixed-file", &fx.fx.path("ff2.toml")));
}

#[test]
fn keyed_table() {
    let fx = fx();
    assert_rekeys("table", &patch(&fx, "w1.tsv", &[]), &patch(&fx, "w2.tsv", &[]));
}

#[test]
fn keyed_scenario() {
    let fx = fx();
    assert_rekeys("scenario", &prof(&fx, &[]), &prof(&fx, &["--scenario", "with_sia"]));
}

#[test]
fn keyed_enable() {
    let fx = fx();
    assert_rekeys("enable", &prof(&fx, &[]), &prof(&fx, &["--enable", "sia"]));
}

#[test]
fn keyed_disable() {
    let fx = fx();
    assert_rekeys("disable", &prof(&fx, &["--enable", "sia"]), &prof(&fx, &["--disable", "sia"]));
}

#[test]
fn keyed_dt() {
    let fx = fx();
    assert_rekeys("dt", &prof(&fx, &[]), &prof(&fx, &["--dt", "0.5"]));
}

#[test]
fn keyed_seed() {
    let fx = fx();
    assert_rekeys("seed", &prof(&fx, &[]), &prof(&fx, &[]).set("--seed", "2"));
}

#[test]
fn keyed_allow_degenerate_rates() {
    let fx = fx();
    assert_rekeys("allow_degenerate_rates", &prof(&fx, &[]), &prof(&fx, &["--allow-degenerate-rates"]));
}

#[test]
fn keyed_allow_forcing_extrapolation() {
    let fx = fx();
    assert_rekeys("allow_forcing_extrapolation", &prof(&fx, &[]), &prof(&fx, &["--allow-forcing-extrapolation"]));
}

#[test]
fn keyed_pf_max_substeps() {
    let fx = fx();
    assert_rekeys("pf_max_substeps", &prof(&fx, &[]), &prof(&fx, &["--pf-max-substeps", "100000000"]));
}

#[test]
fn keyed_binomial() {
    let fx = fx();
    assert_rekeys("binomial", &prof(&fx, &[]), &prof(&fx, &["--binomial", "btrs"]));
}

#[test]
fn keyed_obs() {
    let fx = fx();
    let both = fx.fx.path("both.tsv");
    let base = prof_cli(&fx, &[]).set("--data", &both);
    assert_rekeys("obs", &base.with(&["--obs", "cases"]), &base.with(&["--obs", "prevalent"]));
}

/// `--no-licm` changes the IR the compiler emits for a model with a
/// parameter-only rate factor; the compile is given it, so the base moves.
#[test]
fn keyed_no_licm() {
    let fx = fx();
    assert_rekeys("no_licm", &prof_on(&fx, "sir.camdl", &[]), &prof_on(&fx, "sir.camdl", &["--no-licm"]));
}

#[test]
fn keyed_env_no_licm() {
    let fx = fx();
    assert_rekeys("env:no_licm", &prof_on(&fx, "sir.camdl", &[]), &prof_on(&fx, "sir.camdl", &[]).env("CAMDL_NO_LICM", "1"));
}

#[test]
fn keyed_env_no_constant_fold() {
    let fx = fx();
    assert_rekeys("env:no_constant_fold", &patch(&fx, "w1.tsv", &[]), &patch(&fx, "w1.tsv", &[]).env("CAMDL_NO_CONSTANT_FOLD", "1"));
}

/// An engine switch keys every point, and each point's record names it
/// beside the build; the stored cells are the default path's.
#[test]
fn keyed_env_eval_unresolved() {
    let fx = fx();
    let (a, b) = (run(&prof(&fx, &[])), run(&prof(&fx, &[]).env("CAMDL_EVAL_UNRESOLVED", "1")));
    assert_rekeyed(KIND, "env:eval_unresolved", &level("env:eval_unresolved"), &a, &b);
    for (plain, switched) in a.of(KIND).values().zip(b.of(KIND).values()) {
        assert_eq!(switched.record["engine_version"], plain.record["engine_version"], "engine_version is the build");
        assert_eq!(switched.record["engine_switches"], serde_json::json!([{ "name": "CAMDL_EVAL_UNRESOLVED" }]));
        assert!(plain.record.get("engine_switches").is_none(), "a default run's record has no switch list");
    }
    let cells = |s: &Stored| -> std::collections::BTreeSet<_> { s.of(KIND).values().map(|l| l.files.clone()).collect() };
    assert_eq!(cells(&a), cells(&b), "the same stored cells under the alternative path");
}

#[test]
fn keyed_env_no_binding_cache() {
    let fx = fx();
    assert_rekeys("env:no_binding_cache", &prof(&fx, &[]), &prof(&fx, &[]).env("CAMDL_NO_BINDING_CACHE", "1"));
}

#[test]
fn keyed_env_experimental() {
    let fx = fx();
    let (a, b) = (run(&prof(&fx, &[])), run(&prof(&fx, &[]).env("CAMDL_EXPERIMENTAL_DIFFERENTIAL_TEST", "1")));
    assert_rekeyed(KIND, "env:experimental", &level("env:experimental"), &a, &b);
    let leaf = b.of(KIND).into_values().next().unwrap().clone();
    assert_eq!(
        leaf.record["engine_switches"],
        serde_json::json!([{ "name": "CAMDL_EXPERIMENTAL_DIFFERENTIAL_TEST", "value": "1" }])
    );
}

// ─── execution ───────────────────────────────────────────────────────────────
//
// Every execution, expands, presentation and provenance case runs on both of
// a profile's cell methods, IF2 and an nl-* optimizer ([`bases_on`]), except
// `--backend`, which an nl-* profile must name.

/// Both cell methods on `model`: IF2, and nl-sbplx on the ODE backend.
fn bases_on(fx: &Fx, model: &str) -> [Invocation; 2] {
    [prof_on(fx, model, &[]), prof_nl_on(fx, model, "nl-sbplx", &[])]
}

fn bases(fx: &Fx) -> [Invocation; 2] {
    bases_on(fx, "sir.ir.json")
}

/// The cells run on the scoped pool `--parallel` sizes, IF2's particles
/// inside each; on IF2 and on an nl-* optimizer.
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

/// Each algorithm runs on one backend: naming it changes nothing, and
/// naming another is refused. (An nl-* profile cannot omit it, since the
/// default backend is IF2's.)
#[test]
fn execution_backend() {
    let fx = fx();
    assert_same(&prof(&fx, &[]), &prof(&fx, &["--backend", "chain_binomial"]));
}

#[test]
fn execution_allow_nonconverged_source() {
    let fx = fx();
    let source = format!("from_mle={}", fx.mle_source);
    for b in bases(&fx) {
        assert_same(&b.with(&["--starts", &source]), &b.with(&["--starts", &source, "--allow-nonconverged-source"]));
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
    let camdlc = fx.fx.camdlc.to_string_lossy().into_owned();
    for b in bases_on(&fx, "sir.camdl") {
        assert_same(&b, &b.clone().env("CAMDLC_PATH", &camdlc));
    }
}

#[test]
fn execution_env_camdlc() {
    let fx = fx();
    let bin = tempfile::tempdir().unwrap();
    std::os::unix::fs::symlink(&fx.fx.camdlc, bin.path().join("camdlc")).unwrap();
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
        assert!(sy.of(KIND).values().all(|l| l.dir.starts_with(&elsewhere)), "the store root moved");
        assert_same_result(KIND, &sx, &sy);
    }
}

// ─── expands ─────────────────────────────────────────────────────────────────

#[test]
fn expands_seeds() {
    let fx = fx();
    for b in bases(&fx) {
        assert_shared_leaves_keep_ids(KIND, &run(&b.with(&["--seeds", "1"])), &run(&b.with(&["--seeds", "1,2"])));
    }
}

// ─── presentation and provenance ────────────────────────────────────────────

#[test]
fn provenance_label() {
    let fx = fx();
    for b in bases(&fx) {
        assert_unaffected("--label", &b, &b.with(&["--label", "a-label"]));
    }
}

#[test]
fn presentation_output() {
    let fx = fx();
    for b in bases(&fx) {
        assert_unaffected("--output", &b, &b.with(&["--output", "table.tsv"]));
    }
}

#[test]
fn presentation_format() {
    let fx = fx();
    for b in bases(&fx) {
        assert_unaffected("--format", &b, &b.with(&["--format", "json"]));
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

// ─── removed ─────────────────────────────────────────────────────────────────

#[test]
fn removed_init() {
    let fx = fx();
    assert_removed("--init", &prof(&fx, &["--init", "lhs"]));
}

#[test]
fn removed_posterior() {
    let fx = fx();
    assert_removed("--posterior", &prof(&fx, &["--posterior", "draws.tsv"]));
}

#[test]
fn removed_mle() {
    let fx = fx();
    assert_removed("--mle", &prof(&fx, &["--mle", "mle.toml"]));
}

#[test]
fn removed_params() {
    let fx = fx();
    assert_removed("--params", &prof(&fx, &["--params", "p.toml"]));
}

#[test]
fn removed_pmmh_steps() {
    let fx = fx();
    assert_removed("--pmmh-steps", &prof(&fx, &["--pmmh-steps", "10"]));
}

#[test]
fn removed_pmmh_particles() {
    let fx = fx();
    assert_removed("--pmmh-particles", &prof(&fx, &["--pmmh-particles", "10"]));
}

#[test]
fn removed_pmmh_rho() {
    let fx = fx();
    assert_removed("--pmmh-rho", &prof(&fx, &["--pmmh-rho", "0.9"]));
}

#[test]
fn removed_param() {
    let fx = fx();
    assert_removed("--param", &prof(&fx, &["--param", "beta_a=0.4"]));
}

// ─── the --fit handle ────────────────────────────────────────────────────────

/// `--fit` takes the handle forms `fit predict` takes. The stored IF2 fit of
/// `fit_plain.toml`, named by its leaf directory, its problem folder or its
/// run-id prefix, names the problem its archived config states, the file's
/// (no command-line option changed it), as the `fit.toml` does: every form
/// profiles the one computation, so after the first no form adds a point to
/// the store that holds the fits (where the run-id prefix resolves).
#[test]
fn every_fit_handle_names_the_fits_problem() {
    let fx = fx();
    let leaf = PathBuf::from(&fx.mle_source);
    let segment = leaf.parent().unwrap().parent().unwrap().to_path_buf();
    let record: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(leaf.join("run.json")).unwrap()).unwrap();
    let run_id = record["run_id"].as_str().unwrap()[..12].to_string();
    // The store that holds the source fits, where the run-id prefix
    // resolves; no other case stores a profile point in it.
    let store = fx.fx.dir.join("sources");
    let handles = [fx.fx.path("fit_plain.toml"), fx.mle_source.clone(), segment.to_string_lossy().into_owned(), run_id];
    let mut counts = Vec::new();
    for h in &handles {
        let s = run_into(&store, &prof(&fx, &[]).set("--fit", h));
        assert!(!s.stderr.contains("Is a directory"), "--fit {h}:\n{}", s.stderr);
        counts.push(s.ids(KIND));
    }
    assert!(!counts[0].is_empty(), "the profile stored its points");
    assert!(counts.iter().all(|c| *c == counts[0]), "every handle names one problem: {counts:?}");
}

/// The folders under `root` that hold a `fit.meta.json` and lie beneath a
/// directory named `kind_dir` (`fits` or `profiles`).
fn sidecar_folders(root: &std::path::Path, kind_dir: &str) -> Vec<PathBuf> {
    let mut out = Vec::new();
    let mut stack = vec![root.to_path_buf()];
    while let Some(d) = stack.pop() {
        for e in std::fs::read_dir(&d).into_iter().flatten().flatten() {
            let p = e.path();
            if p.is_dir() {
                stack.push(p);
            } else if p.file_name().is_some_and(|n| n == "fit.meta.json")
                && p.strip_prefix(root).unwrap().components().any(|c| c.as_os_str() == kind_dir)
            {
                out.push(p.parent().unwrap().to_path_buf());
            }
        }
    }
    out
}

/// A profile's folder stores the config as a fit folder does: the user's
/// text, byte for byte, as `fit.toml.original` and the resolved
/// configuration as `fit.resolved.toml`. Named by a sweep point's leaf, the
/// pair is the fit folder's own pair (the file's text, and the point's
/// resolved config); named by the file, the file's text and its resolution.
#[test]
fn a_profile_stores_the_original_and_the_resolved_config() {
    let fx = fx();
    let store = tempfile::tempdir().unwrap();
    let fitted = run_into(store.path(), &Invocation::new(&fx.fx, &["fit", "run", &fx.fx.path("fit_plain.toml"), "--sweep", "rho=0.5,0.6"]));
    let (_, other) = sweep_points(&fitted, "rho", 0.5);
    let fit_folder = other.dir.parent().unwrap().parent().unwrap().to_path_buf();
    let profiled = tempfile::tempdir().unwrap();
    run_into(profiled.path(), &prof(&fx, &[]).set("--fit", &other.dir.to_string_lossy()));
    let folders = sidecar_folders(profiled.path(), "profiles");
    assert_eq!(folders.len(), 1, "one profile folder: {folders:?}");
    let read = |d: &std::path::Path, f: &str| std::fs::read_to_string(d.join(f)).unwrap_or_else(|e| panic!("{}: {e}", d.join(f).display()));
    assert_eq!(read(&folders[0], "fit.toml.original"), std::fs::read_to_string(fx.fx.path("fit_plain.toml")).unwrap(),
        "the profile's original is the user's file, byte for byte");
    assert_eq!(read(&folders[0], "fit.toml.original"), read(&fit_folder, "fit.toml.original"));
    assert_eq!(read(&folders[0], "fit.resolved.toml"), read(&fit_folder, "fit.resolved.toml"),
        "the profile's resolved config is the sweep point's");
    let by_file = tempfile::tempdir().unwrap();
    run_into(by_file.path(), &prof(&fx, &[]));
    let folders = sidecar_folders(by_file.path(), "profiles");
    assert_eq!(folders.len(), 1, "one profile folder: {folders:?}");
    assert_eq!(read(&folders[0], "fit.toml.original"), std::fs::read_to_string(fx.fx.path("fit_plain.toml")).unwrap());
    let resolved = read(&folders[0], "fit.resolved.toml");
    assert!(resolved.starts_with("# fit.resolved.toml:"), "{resolved}");
}

/// A point of `fit run --sweep` named by its leaf directory, its problem
/// folder or its run-id prefix is profiled at the point's own value, which
/// its fit folder's `fit.resolved.toml` holds: every form stores the bytes a
/// profile of `fit_rho.toml`, whose file writes the point's value, stores, and
/// not those of the file's own value. The forms name one computation.
#[test]
fn a_sweep_point_handle_profiles_at_the_points_value() {
    let fx = fx();
    let store = tempfile::tempdir().unwrap();
    let fitted = run_into(store.path(), &Invocation::new(&fx.fx, &["fit", "run", &fx.fx.path("fit_plain.toml"), "--sweep", "rho=0.5,0.6"]));
    let (_, other) = sweep_points(&fitted, "rho", 0.5);
    assert_eq!(segment_fixed(&other.dir, "rho"), Some(0.6));
    let written = run(&prof(&fx, &[]).set("--fit", &fx.fx.path("fit_rho.toml"))).bytes(KIND);
    let unswept = run(&prof(&fx, &[]).set("--fit", &fx.fx.path("fit_plain.toml"))).bytes(KIND);
    assert!(!written.is_empty());
    assert_ne!(written, unswept, "the swept value changes what a profile stores");
    let segment = other.dir.parent().unwrap().parent().unwrap().to_string_lossy().into_owned();
    let mut ids = Vec::new();
    for h in [other.dir.to_string_lossy().into_owned(), segment, other.run_id[..12].to_string()] {
        let s = run_into(store.path(), &prof(&fx, &[]).set("--fit", &h));
        assert_eq!(s.bytes(KIND), written, "--fit {h}: the profile of the point's value");
        ids.push(s.ids(KIND));
    }
    assert!(ids.iter().all(|i| *i == ids[0]), "every handle names one computation: {ids:?}");
}
