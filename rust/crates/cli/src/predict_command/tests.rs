//! `camdl predict` on the run store, checked by running the release binary:
//! every declared input against its class (`store::check_input_classes`),
//! the bytes a prediction computes against those `fit predict` computed
//! before it was replaced, and the refusals of the removed spellings.
//!
//! The binary is `rust/target/release/camdl` and the compiler
//! `ocaml/_build/default/bin/camdlc.exe`, both built by `make build` from the
//! source this test was built from. A store holding two PMMH runs of one
//! problem, `@fit` (seed 1) and `@other` (seed 2), is made once; every run
//! starts from a copy of it, in a fresh copy of the fixture files, so the
//! relative paths `crate::input_fixtures` returns name them.

// The harness's `run` returns the store's own error, as `InputHarness::run`
// requires.
#![allow(clippy::result_large_err)]

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};
use std::sync::OnceLock;

use store::{Artifact, Class, ClassFailure, FieldInfo, InputHarness, Inputs, Key, Observed, ObservedResult};

use crate::args::PredictInvocation;
use crate::test_support::Pins;
use crate::input_fixtures::{predict_base, predict_from};

const FORECAST: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/forecast");

/// The fixture model: the forecast tests' survey model, with an
/// intervention `vax` that moves half of S to R on day 30, a scenario
/// enabling it, and one quantity.
fn model() -> String {
    let survey = std::fs::read_to_string(Path::new(FORECAST).join("survey_sir.camdl")).unwrap();
    format!(
        "{survey}\ninterventions {{\n  vax : transfer(fraction = 0.5, from = S, to = R) at [30]\n}}\n\n\
         scenarios {{\n  vax {{ enable = [vax] }}\n}}\n\nquantities {{\n  peak_I = max(I)\n}}\n"
    )
}

/// A PMMH fit of the three streams, the rows after day 24 held out.
const FIT: &str = r#"[model]
camdl = "model.camdl"

[data]
holdout_after = 24.0

[data.observations]
daily_cases = "daily_cases.tsv"
weekly_cases = "weekly_cases.tsv"
survey = "survey.tsv"

[estimate]
beta = { bounds = [0.2, 1.0], start = 0.5 }

[fixed]
gamma = 0.25
rho   = 0.5
k     = 10.0
N0    = 1000
I0    = 5

[method]
algorithm = "pmmh"
backend = "chain_binomial"
chains = 2
particles = 50
iterations = 30
burn_in = 10
thin = 1
"#;

/// The fixture model with an infection rate whose parameter-only factor,
/// `beta * exp(-0.01 * k)`, loop-invariant code motion hoists, so a compile
/// with it and one without it are different IR.
fn hoisted_model() -> String {
    let m = model();
    let hoisted = m.replace("beta * S * I / N", "beta * exp(-0.01 * k) * S * I / N");
    assert_ne!(m, hoisted, "the fixture's infection rate moved");
    hoisted
}

/// The binaries under test, the fixture directory, and the store holding
/// `@fit` and `@other`, `@hoisted`, a fit of [`hoisted_model`], and `@ode`,
/// an MH fit on the ODE backend, made once.
struct Fixture {
    camdl: PathBuf,
    camdlc: PathBuf,
    dir: PathBuf,
    store: PathBuf,
}

fn fixture() -> &'static Fixture {
    static FX: OnceLock<Fixture> = OnceLock::new();
    FX.get_or_init(|| {
        let manifest = Path::new(env!("CARGO_MANIFEST_DIR"));
        let camdl = manifest.join("../../target/release/camdl");
        let camdlc = manifest.join("../../../ocaml/_build/default/bin/camdlc.exe");
        assert!(camdl.exists(), "release camdl missing: {} — run `make build`", camdl.display());
        assert!(camdlc.exists(), "camdlc missing: {} — run `make build`", camdlc.display());
        let version = Command::new(&camdl).arg("--version").output().unwrap();
        let version = String::from_utf8_lossy(&version.stdout);
        assert!(
            version.contains(&format!("+{} ", env!("CAMDL_SOURCE_LABEL"))),
            "{} is not built from this source ({}): {version} — run `make build`",
            camdl.display(),
            env!("CAMDL_SOURCE_LABEL")
        );
        let dir = crate::test_support::unique_temp_dir("predict_command_fixture");
        std::fs::create_dir_all(&dir).unwrap();
        for f in ["daily_cases.tsv", "weekly_cases.tsv", "survey.tsv"] {
            std::fs::copy(Path::new(FORECAST).join(f), dir.join(f)).unwrap();
        }
        std::fs::write(dir.join("model.camdl"), model()).unwrap();
        std::fs::write(dir.join("fit.toml"), FIT).unwrap();
        std::fs::write(dir.join("hoisted.camdl"), hoisted_model()).unwrap();
        std::fs::write(dir.join("fit_hoisted.toml"), FIT.replace("model.camdl", "hoisted.camdl")).unwrap();
        std::fs::write(
            dir.join("fit_ode.toml"),
            FIT.split("[method]").next().unwrap().to_string()
                + "[method]\nalgorithm = \"mh\"\nbackend = \"ode\"\nchains = 2\niterations = 40\nburn_in = 10\n",
        )
        .unwrap();
        std::fs::write(dir.join("vocab.camdl"), "quantities {\n  peak_R = max(R)\n}\n").unwrap();
        std::fs::write(dir.join("design.tsv"), "time\ttested\n28\t60\n35\t70\n").unwrap();
        std::fs::write(dir.join("design2.tsv"), "time\ttested\n28\t61\n35\t70\n").unwrap();
        let store = dir.join("template-store");
        let fx = Fixture { camdl, camdlc, dir, store };
        for (seed, label) in [("1", "fit"), ("2", "other")] {
            let args: Vec<String> = ["fit", "run", "fit.toml", "--seed", seed, "--label", label, "--progress", "none"].map(String::from).into();
            let out = camdl_with(&fx, &fx.dir, &fx.store, &args, &[]);
            assert!(out.status.success(), "the fixture's fit @{label}: {}", String::from_utf8_lossy(&out.stderr));
        }
        let args: Vec<String> =
            ["fit", "run", "fit_hoisted.toml", "--label", "hoisted", "--progress", "none"].map(String::from).into();
        let out = camdl_with(&fx, &fx.dir, &fx.store, &args, &[]);
        assert!(out.status.success(), "the fixture's fit @hoisted: {}", String::from_utf8_lossy(&out.stderr));
        let args: Vec<String> = ["fit", "run", "fit_ode.toml", "--label", "ode", "--progress", "none"].map(String::from).into();
        let out = camdl_with(&fx, &fx.dir, &fx.store, &args, &[]);
        assert!(out.status.success(), "the fixture's fit @ode: {}", String::from_utf8_lossy(&out.stderr));
        fx
    })
}

/// A fresh working directory holding a copy of the fixture files, and a copy
/// of the fixture's store at `root`. The copy keeps each file's times, which
/// the store checks a completed result's files against.
fn workdir(root: &Path) -> tempfile::TempDir {
    let fx = fixture();
    let d = tempfile::tempdir().unwrap();
    for e in std::fs::read_dir(&fx.dir).unwrap().flatten() {
        if e.path().is_file() {
            std::fs::copy(e.path(), d.path().join(e.file_name())).unwrap();
        }
    }
    let copied = Command::new("cp").arg("-Rp").arg(&fx.store).arg(root).output().unwrap();
    assert!(copied.status.success(), "copying the fixture's store: {}", String::from_utf8_lossy(&copied.stderr));
    d
}

/// `camdl` in `cwd`, with a cleared environment holding `PATH`, `TMPDIR`, a
/// private `HOME` and IR cache, the compiler under test, `CAMDL_STORE` and
/// `env`.
fn camdl_with(fx: &Fixture, cwd: &Path, store: &Path, args: &[String], env: &[(&str, &str)]) -> Output {
    let mut cmd = Command::new(&fx.camdl);
    cmd.current_dir(cwd).env_clear();
    for k in ["PATH", "TMPDIR"] {
        if let Some(v) = std::env::var_os(k) {
            cmd.env(k, v);
        }
    }
    cmd.env("HOME", cwd.join("home"))
        .env("CAMDL_SKIP_VERSION_CHECK", "1")
        .env("CAMDLC", &fx.camdlc)
        .env("CAMDL_IR_CACHE_DIR", cwd.join("ir-cache"))
        .env("CAMDL_STORE", store);
    for (k, v) in env {
        cmd.env(k, v);
    }
    cmd.args(args).output().expect("spawn camdl")
}

fn camdl(cwd: &Path, store: &Path, args: &[String], env: &[(&str, &str)]) -> Output {
    camdl_with(fixture(), cwd, store, args, env)
}

/// `inv` as a command line.
fn argv(inv: &PredictInvocation) -> Vec<String> {
    let PredictInvocation { args, no_licm, no_ir_cache, verbosity: _, progress: _, no_progress: _ } = inv;
    let crate::args::PredictArgs {
        source,
        flags:
            crate::args::PredictFlags {
                forcing_extrapolation: crate::args::ForcingExtrapolationArg { allow_forcing_extrapolation },
                stream,
                scenarios,
                enable,
                disable,
                sweep,
                mode,
                horizon,
                design,
                quantities,
                n_draws,
                particles,
                seed,
            },
        posterior: crate::args::PosteriorFlags { by_chain, exclude_chains, chains, realised_design },
        _removed_fit: _,
    } = args;
    let mut v: Vec<String> = vec!["predict".into(), source.clone().expect("a run")];
    let mut opt = |flag: &str, value: Option<String>| {
        if let Some(x) = value {
            v.push(flag.into());
            v.push(x);
        }
    };
    for s in stream {
        opt("--stream", Some(s.clone()));
    }
    for s in scenarios {
        opt("--scenario", Some(s.clone()));
    }
    for s in enable {
        opt("--enable", Some(s.clone()));
    }
    for s in disable {
        opt("--disable", Some(s.clone()));
    }
    for s in sweep {
        let values: Vec<String> = s.grid.expand().iter().map(f64::to_string).collect();
        opt("--sweep", Some(format!("{}={}", s.name, values.join(","))));
    }
    opt("--mode", mode.map(|m| serde_json::to_value(m).unwrap().as_str().unwrap().to_string()));
    opt("--horizon", horizon.clone());
    for d in design {
        opt("--design", Some(d.clone()));
    }
    opt("--quantities", quantities.as_ref().map(|p| p.to_string_lossy().into_owned()));
    opt("--n-draws", n_draws.map(|x| x.to_string()));
    opt("--particles", particles.map(|x| x.to_string()));
    opt("--seed", seed.map(|x| x.to_string()));
    opt("--exclude-chains", exclude_chains.clone());
    opt("--chains", chains.clone());
    let mut flag = |set: bool, name: &str| {
        if set {
            v.push(name.into());
        }
    };
    flag(*allow_forcing_extrapolation, "--allow-forcing-extrapolation");
    flag(*by_chain, "--by-chain");
    flag(*realised_design, "--realised-design");
    flag(*no_licm, "--no-licm");
    flag(*no_ir_cache, "--no-ir-cache");
    v.extend(["--progress".to_string(), "none".to_string()]);
    v
}

/// Every result folder under `root` with its `identity.json` and `run.json`,
/// in path order.
fn folders(root: &Path) -> Vec<(PathBuf, serde_json::Value, serde_json::Value)> {
    fn walk(dir: &Path, out: &mut Vec<(PathBuf, serde_json::Value, serde_json::Value)>) {
        let Ok(entries) = std::fs::read_dir(dir) else { return };
        let mut entries: Vec<_> = entries.flatten().collect();
        entries.sort_by_key(|e| e.file_name());
        for e in entries {
            let p = e.path();
            if !p.is_dir() || e.file_name().to_string_lossy().starts_with('.') {
                continue;
            }
            if p.join("identity.json").is_file() {
                let read = |f: &str| serde_json::from_slice(&std::fs::read(p.join(f)).unwrap()).unwrap();
                out.push((p.clone(), read("identity.json"), read("run.json")));
            }
            walk(&p, out);
        }
    }
    let mut out = Vec::new();
    walk(root, &mut out);
    out
}

/// Every prediction under `root`, and every data table a prediction
/// depends on: its kind, run id and result files with their digests. The
/// fit runs and their problems, which every run starts from, are left out.
fn observe(root: &Path) -> Vec<ObservedResult> {
    let mut out = Vec::new();
    for (_, id, run) in folders(root) {
        let kind = id["kind"].as_str().unwrap().to_string();
        if kind != crate::fit_store::Prediction::KIND && kind != crate::camdl_store::DataTable::KIND {
            continue;
        }
        let files = run["files"]
            .as_object()
            .unwrap()
            .iter()
            .filter(|(_, f)| f["role"] == "result")
            .map(|(n, f)| (n.clone(), Key::from_hex(f["sha256"].as_str().unwrap()).unwrap()))
            .collect();
        out.push(ObservedResult {
            content: kind == crate::camdl_store::DataTable::KIND,
            kind,
            run_id: Key::from_hex(id["run_id"].as_str().unwrap()).unwrap(),
            files,
            upstream: None,
        });
    }
    out
}

/// What a run's execution inputs did, each under `used.<field>`: from the
/// debug log of the compile layer (`camdl::util`), whether camdlc ran with
/// loop-invariant code motion off; from the IR cache directory, whether the
/// compile bypassed the cache. A compile the log reports twice with
/// different settings is recorded as both, so it matches no single value.
fn used(stderr: &str, ir_cache: &Path) -> BTreeMap<String, String> {
    let mut seen: BTreeMap<String, std::collections::BTreeSet<String>> = BTreeMap::new();
    for line in stderr.lines() {
        if let Some((_, rest)) = line.split_once("camdlc runs with loop-invariant code motion ") {
            let off = rest.split_whitespace().next() == Some("off");
            seen.entry("used.no_licm".into()).or_default().insert(off.to_string());
        }
    }
    seen.entry("used.no_ir_cache".into())
        .or_default()
        .insert((!crate::test_support::ir_cache_filled(ir_cache)).to_string());
    seen.into_iter().map(|(k, v)| (k, v.into_iter().collect::<Vec<_>>().join(","))).collect()
}

/// The harness of `predict`'s class checks, from one base invocation.
struct Harness {
    base: PredictInvocation,
}

impl Harness {
    /// The base each field is checked from: a prediction on which the field
    /// is live. A filter's particle count is checked on the one-step band, a
    /// forecast's reach and design on a forecast.
    fn for_field(name: &str) -> Harness {
        let field = name.rsplit('.').next().unwrap();
        let base = match field {
            "particles" => predict_from(&["@fit", "--mode", "one_step", "--n-draws", "4"]),
            "horizon" => predict_from(&["@fit", "--horizon", "2 weeks", "--stream", "daily_cases", "--n-draws", "4", "--particles", "20"]),
            "design" => predict_from(&[
                "@fit", "--horizon", "2 weeks", "--stream", "survey", "--design", "survey=design.tsv", "--n-draws", "4",
                "--particles", "20",
            ]),
            _ => predict_base(),
        };
        Harness { base }
    }
}

impl InputHarness for Harness {
    type Invocation = PredictInvocation;

    fn base(&self) -> PredictInvocation {
        self.base.clone()
    }

    fn varied(&self, field: &FieldInfo) -> PredictInvocation {
        match field.class {
            Class::Identity => self.base.varied(field.name),
            class => panic!("no second value for {} ({class:?})", field.name),
        }
    }

    fn exercised(&self, field: &FieldInfo) -> (PredictInvocation, PredictInvocation) {
        <PredictInvocation as Inputs>::exercised(field.name)
    }

    /// An execution input is used when what the run did with it ([`used`])
    /// is the value it recorded
    /// ([`crate::test_support::recorded_execution`]).
    fn shows_use(&self, field: &FieldInfo, seen: &Observed) -> bool {
        let effect = |prefix: &str| seen.effects.get(&format!("{prefix}.{}", field.name)).cloned();
        matches!((effect("recorded"), effect("used")), (Some(r), Some(u)) if r == u)
    }

    fn run(&self, invocation: &PredictInvocation, store: Option<&Path>) -> store::Result<Observed> {
        let fresh = tempfile::tempdir().unwrap();
        let root = store.map(Path::to_path_buf).unwrap_or_else(|| fresh.path().join("camdl-store"));
        let cwd = workdir(&root);
        let out = camdl(cwd.path(), &root, &argv(invocation), &[("RUST_LOG", "camdl::util=debug")]);
        if !out.status.success() {
            return Err(store::Error::Parse {
                what: format!("camdl {:?}", argv(invocation)),
                why: String::from_utf8_lossy(&out.stderr).into_owned(),
            });
        }
        let mut effects = crate::test_support::recorded_execution(&root, crate::fit_store::Prediction::KIND);
        effects.extend(used(&String::from_utf8_lossy(&out.stderr), &cwd.path().join("ir-cache")));
        Ok(Observed { results: observe(&root), effects, ..Observed::default() })
    }
}

/// The declared inputs the class check does not run, each with the reason.
const NOT_RUN: &[(&str, &str)] = &[(
    "args.posterior.realised_design",
    "a forecast from a stored fit refuses it until held-out scores exist (`a_realised_design_forecast_is_refused`)",
)];

/// Every declared input of `camdl predict` keeps the promise its class
/// makes.
#[test]
fn every_predict_input_keeps_its_class() {
    let fields: Vec<FieldInfo> =
        PredictInvocation::fields().iter().filter(|f| NOT_RUN.iter().all(|(n, _)| *n != f.name)).cloned().collect();
    assert_eq!(fields.len() + NOT_RUN.len(), PredictInvocation::fields().len(), "every input the check skips is declared");
    let failures: Vec<ClassFailure> = std::thread::scope(|s| {
        let chunks: Vec<Vec<FieldInfo>> = fields.chunks(fields.len().div_ceil(4)).map(<[_]>::to_vec).collect();
        let handles: Vec<_> = chunks
            .into_iter()
            .map(|chunk| {
                s.spawn(move || {
                    let mut failures = Vec::new();
                    for f in chunk {
                        let h = Harness::for_field(f.name);
                        if let Err(mut e) = store::check_input_classes(std::slice::from_ref(&f), &h) {
                            failures.append(&mut e);
                        }
                    }
                    failures
                })
            })
            .collect();
        handles.into_iter().flat_map(|h| h.join().unwrap()).collect()
    });
    assert!(failures.is_empty(), "inputs that break their class:\n{failures:#?}");
}

/// Every argument clap accepts for `predict`, and every global option, is a
/// declared input; and every declared input is one clap accepts.
#[test]
fn every_predict_argument_is_declared() {
    use clap::CommandFactory;
    let cmd = crate::Cli::command();
    let sub = cmd.find_subcommand("predict").expect("the predict subcommand");
    let mut accepted: std::collections::BTreeSet<String> =
        sub.get_arguments().map(|a| a.get_id().to_string()).filter(|id| id != "help").collect();
    accepted.extend(cmd.get_arguments().map(|a| a.get_id().to_string()).filter(|id| id != "help" && id != "version"));
    let declared: std::collections::BTreeSet<String> =
        PredictInvocation::fields().iter().map(|f| f.name.rsplit('.').next().unwrap().to_string()).collect();
    assert_eq!(accepted, declared);
}

/// A forecast's design values from a stored fit's held-out rows are refused:
/// the stored problem keeps their dates and not their values.
#[test]
fn a_realised_design_forecast_is_refused() {
    let fresh = tempfile::tempdir().unwrap();
    let root = fresh.path().join("camdl-store");
    let cwd = workdir(&root);
    let inv = predict_from(&["@fit", "--horizon", "2 weeks", "--stream", "survey", "--realised-design", "--n-draws", "4"]);
    let out = camdl(cwd.path(), &root, &argv(&inv), &[]);
    let err = String::from_utf8_lossy(&out.stderr);
    assert!(!out.status.success(), "--realised-design ran:\n{err}");
    assert!(err.contains("--realised-design") && err.contains("held-out"), "{err}");
    assert!(folders(&root).iter().all(|(_, id, _)| id["kind"] != crate::fit_store::Prediction::KIND), "nothing is stored");
}

/// `camdl fit predict` and `camdl predict --fit` are refused, each naming
/// `camdl predict @run`; nothing is stored.
#[test]
fn the_removed_spellings_are_refused_with_their_replacement() {
    let fresh = tempfile::tempdir().unwrap();
    let root = fresh.path().join("camdl-store");
    let cwd = workdir(&root);
    for line in [&["fit", "predict", "@fit", "--mode", "free_forward"][..], &["fit", "predict", "--fit", "@fit"], &["predict", "--fit", "@fit"]] {
        let args: Vec<String> = line.iter().map(|s| s.to_string()).collect();
        let out = camdl(cwd.path(), &root, &args, &[]);
        let err = String::from_utf8_lossy(&out.stderr);
        assert!(!out.status.success(), "{line:?} ran:\n{err}");
        assert!(err.contains("camdl predict @"), "{line:?}: the refusal names the replacement:\n{err}");
    }
    assert!(folders(&root).iter().all(|(_, id, _)| id["kind"] != crate::fit_store::Prediction::KIND), "nothing is stored");
}

/// A reporting vocabulary is checked against the fit's compile whatever
/// either was compiled with: loop-invariant code motion moves a hoistable
/// rate's factor into a binding without changing the model. A fit compiled
/// with it is predicted with `--quantities` under `--no-licm`, and a fit
/// compiled under `--no-licm` is predicted without it; both run, and give
/// the bytes the matching setting gives.
#[test]
fn a_vocabulary_is_checked_against_the_fit_whatever_its_licm_setting() {
    let fresh = tempfile::tempdir().unwrap();
    let root = fresh.path().join("camdl-store");
    let cwd = workdir(&root);
    let run = |args: &[&str]| camdl(cwd.path(), &root, &args.iter().map(|s| s.to_string()).collect::<Vec<_>>(), &[]);
    let ok = |o: Output, what: &str| assert!(o.status.success(), "{what}: {}", String::from_utf8_lossy(&o.stderr));
    ok(run(&["--no-licm", "fit", "run", "fit_hoisted.toml", "--label", "inlined", "--progress", "none"]), "the --no-licm fit");
    let vocab = ["--mode", "free_forward", "--n-draws", "4", "--quantities", "vocab.camdl", "--progress", "none"];
    let predict = |licm_off: bool, handle: &str| {
        let mut a: Vec<&str> = if licm_off { vec!["--no-licm"] } else { vec![] };
        a.extend(["predict", handle]);
        a.extend(vocab);
        run(&a)
    };
    for (off, handle) in [(true, "@hoisted"), (false, "@inlined"), (false, "@hoisted"), (true, "@inlined")] {
        let out = predict(off, handle);
        let err = String::from_utf8_lossy(&out.stderr);
        assert!(out.status.success(), "predict {handle} (--no-licm: {off}):\n{err}");
    }
    let predictions: Vec<PathBuf> = folders(&root)
        .into_iter()
        .filter(|(_, id, _)| id["kind"] == crate::fit_store::Prediction::KIND)
        .map(|(p, _, _)| p)
        .collect();
    assert_eq!(predictions.len(), 2, "one prediction of each run, whatever --no-licm: {predictions:?}");
}

/// With no `--mode` the prediction is every mode that applies to the fit's
/// backend, resolved before its identity: on an ODE fit that is the
/// free-forward band alone, so the default and `--mode free_forward` are one
/// prediction. On a chain-binomial fit the default is both bands, which no
/// single `--mode` asks for.
#[test]
fn the_default_mode_is_the_mode_it_resolves_to() {
    let fresh = tempfile::tempdir().unwrap();
    let root = fresh.path().join("camdl-store");
    let cwd = workdir(&root);
    let predictions = || -> Vec<PathBuf> {
        folders(&root)
            .into_iter()
            .filter(|(_, id, _)| id["kind"] == crate::fit_store::Prediction::KIND)
            .map(|(p, _, _)| p)
            .collect()
    };
    for line in [&["@ode", "--n-draws", "4"][..], &["@ode", "--mode", "free_forward", "--n-draws", "4"]] {
        let out = camdl(cwd.path(), &root, &argv(&predict_from(line)), &[]);
        assert!(out.status.success(), "{line:?}: {}", String::from_utf8_lossy(&out.stderr));
    }
    let ode = predictions();
    assert_eq!(ode.len(), 1, "the default and --mode free_forward are one prediction on an ODE fit: {ode:?}");
    assert!(ode[0].file_name().unwrap().to_string_lossy().starts_with("free_forward-"), "{ode:?}");
    for line in [&["@fit", "--n-draws", "4"][..], &["@fit", "--mode", "free_forward", "--n-draws", "4"]] {
        let out = camdl(cwd.path(), &root, &argv(&predict_from(line)), &[]);
        assert!(out.status.success(), "{line:?}: {}", String::from_utf8_lossy(&out.stderr));
    }
    assert_eq!(predictions().len(), 3, "on a chain-binomial fit the default is both bands, its own prediction");
}

/// A run named by its folder is predicted in the store that holds the
/// folder, whatever `CAMDL_STORE` names: the prediction is stored inside
/// the run, and the other store is left untouched.
#[test]
fn a_folder_handle_predicts_into_the_store_holding_it() {
    let fresh = tempfile::tempdir().unwrap();
    let root = fresh.path().join("camdl-store");
    let cwd = workdir(&root);
    let run = folders(&root)
        .into_iter()
        .find(|(_, id, _)| id["kind"] == crate::fit_store::FitRun::KIND)
        .map(|(p, _, _)| p)
        .expect("a fit run");
    let elsewhere = fresh.path().join("elsewhere");
    let args: Vec<String> = ["predict", &run.to_string_lossy(), "--mode", "free_forward", "--n-draws", "2", "--progress", "none"]
        .map(String::from)
        .into();
    let out = camdl(cwd.path(), &elsewhere, &args, &[]);
    assert!(out.status.success(), "{}", String::from_utf8_lossy(&out.stderr));
    assert!(
        folders(&run).iter().any(|(_, id, _)| id["kind"] == crate::fit_store::Prediction::KIND),
        "the prediction is inside the run"
    );
    assert!(!elsewhere.exists(), "the store CAMDL_STORE names is not made");
}

/// A prediction from an optimizer's run is refused, naming the estimate
/// `fit summary --params-only` prints and the `--params` that takes it.
#[test]
fn an_optimizer_run_is_refused_with_its_point_estimate() {
    let fresh = tempfile::tempdir().unwrap();
    let root = fresh.path().join("camdl-store");
    let cwd = workdir(&root);
    let if2 = FIT.split("[method]").next().unwrap().to_string()
        + "[method]\nalgorithm = \"if2\"\nbackend = \"chain_binomial\"\nchains = 2\nparticles = 30\niterations = 3\ncooling = 0.9\n";
    std::fs::write(cwd.path().join("fit_if2.toml"), if2).unwrap();
    let run = |args: &[&str]| camdl(cwd.path(), &root, &args.iter().map(|s| s.to_string()).collect::<Vec<_>>(), &[]);
    let fitted = run(&["fit", "run", "fit_if2.toml", "--label", "scout", "--progress", "none"]);
    assert!(fitted.status.success(), "{}", String::from_utf8_lossy(&fitted.stderr));
    let out = run(&["predict", "@scout", "--mode", "free_forward", "--n-draws", "4"]);
    let err = String::from_utf8_lossy(&out.stderr);
    assert!(!out.status.success(), "an IF2 run was predicted from:\n{err}");
    assert!(err.contains("camdl fit summary @scout --params-only") && err.contains("--params"), "{err}");
}

/// The bytes each prediction computes, as `camdl fit predict` computed them
/// for the same runs before `camdl predict` replaced it: the digest of each
/// result file, with the fit run's id written `<run>`, and the lines that
/// name the build dropped (`engine_version`, and `address_key`, which hashes
/// it). The run id holds the build too, so it is checked to be the run's own
/// rather than pinned.
///
/// The pins are a regression pin per platform, keyed by `target_os` and
/// `target_arch`, because the digests depend on the platform's libm. That
/// `camdl predict` computes what `camdl fit predict` computed was checked on
/// macOS arm64 against the binary of `78f32ece0`; linux arm64 computes the
/// same table. On a platform with no pins the test fails and prints the
/// table to add.
#[test]
fn each_prediction_computes_what_fit_predict_computed() {
    use sha2::{Digest, Sha256};
    let digests = |dir: &Path, run_id: &str| -> BTreeMap<String, String> {
        let mut out = BTreeMap::new();
        let mut stack = vec![dir.to_path_buf()];
        while let Some(d) = stack.pop() {
            for e in std::fs::read_dir(&d).unwrap().flatten() {
                let p = e.path();
                if p.is_dir() {
                    stack.push(p);
                    continue;
                }
                let name = e.file_name().to_string_lossy().into_owned();
                if ["run.json", "identity.json"].contains(&name.as_str()) {
                    continue;
                }
                let text = std::fs::read_to_string(&p).unwrap().replace(run_id, "<run>");
                let kept: String = text
                    .split_inclusive('\n')
                    .filter(|l| !l.contains("engine_version") && !l.contains("address_key"))
                    .collect();
                let rel = p.strip_prefix(dir).unwrap().to_string_lossy().into_owned();
                out.insert(rel, format!("{:x}", Sha256::digest(kept.as_bytes()))[..16].to_string());
            }
        }
        out
    };
    let cases: &[(&str, &[&str])] = &[
        ("all", &["@fit", "--n-draws", "8"]),
        ("forecast", &[
            "@fit", "--horizon", "2 weeks", "--design", "survey=design.tsv", "--n-draws", "4", "--particles", "20",
            "--stream", "survey", "--stream", "daily_cases",
        ]),
        ("by_chain", &["@fit", "--mode", "free_forward", "--by-chain", "--exclude-chains", "1", "--n-draws", "8"]),
        ("sweep", &["@fit", "--mode", "free_forward", "--sweep", "gamma=0.2,0.3", "--n-draws", "4"]),
        ("scenario", &["@fit", "--mode", "free_forward", "--scenario", "vax", "--n-draws", "8", "--seed", "3"]),
        ("quantities", &["@fit", "--mode", "free_forward", "--quantities", "vocab.camdl", "--n-draws", "4"]),
        ("one_step", &["@fit", "--mode", "one_step", "--n-draws", "4", "--particles", "20"]),
    ];
    let mut got = BTreeMap::new();
    for (name, line) in cases {
        let fresh = tempfile::tempdir().unwrap();
        let root = fresh.path().join("camdl-store");
        let cwd = workdir(&root);
        let out = camdl(cwd.path(), &root, &argv(&predict_from(line)), &[]);
        assert!(out.status.success(), "{name}: {}", String::from_utf8_lossy(&out.stderr));
        let predictions: Vec<PathBuf> = folders(&root)
            .into_iter()
            .filter(|(_, id, _)| id["kind"] == crate::fit_store::Prediction::KIND)
            .map(|(p, _, _)| p)
            .collect();
        assert_eq!(predictions.len(), 1, "{name}: one prediction");
        // The prediction sits at `<run>/predict/<mode>-<id>/`.
        let run = predictions[0].parent().and_then(Path::parent).expect("inside its run");
        let run_id: serde_json::Value = serde_json::from_slice(&std::fs::read(run.join("identity.json")).unwrap()).unwrap();
        got.insert(name.to_string(), digests(&predictions[0], run_id["run_id"].as_str().unwrap()));
    }
    crate::test_support::assert_pinned(PINS, &got, "the files a prediction computes moved");
}

/// The pins of [`each_prediction_computes_what_fit_predict_computed`], per
/// `(target_os, target_arch)`.
/// Linux arm64 computes the macOS arm64 table, digest for digest, so the two
/// share it. Linux x86-64 is not measured, and has no row.
const PINS: &[((&str, &str), Pins)] = &[(("macos", "aarch64"), ARM64), (("linux", "aarch64"), ARM64)];

const ARM64: Pins = &[
    ("all", &[
        ("observed.json", "3e32bfb6e055c935"),
        ("observed/daily_cases.tsv", "ee9d3cb76b7457fc"),
        ("observed/survey.tsv", "739e02d5718619d2"),
        ("observed/weekly_cases.tsv", "884f26939b41d3c2"),
        ("predictive.json", "da72e7198647b896"),
        ("predictive/daily_cases.tsv", "f4b3973ac06fa000"),
        ("predictive/survey.tsv", "94eb2ecbb1179831"),
        ("predictive/weekly_cases.tsv", "9c57d6c2ccb76817"),
        ("quantities-model/peak_I.tsv", "1e14411cf72c645f"),
        ("quantities.json", "daf5039b8d4a73b3"),
        ("report.json", "d47528701a6bf746"),
    ]),
    ("by_chain", &[
        ("observed.json", "6aa2e125b201fcc3"),
        ("observed/daily_cases.tsv", "ee9d3cb76b7457fc"),
        ("observed/survey.tsv", "739e02d5718619d2"),
        ("observed/weekly_cases.tsv", "884f26939b41d3c2"),
        ("predictive-excl1.json", "7030036c794e1eef"),
        ("predictive-excl1/daily_cases.tsv", "b150b7f0190d7f96"),
        ("predictive-excl1/survey.tsv", "774013e49290c9a3"),
        ("predictive-excl1/weekly_cases.tsv", "49e626bca412254a"),
        ("quantities-excl1.json", "daf5039b8d4a73b3"),
        ("quantities-model-excl1/peak_I.tsv", "291336833bcaf220"),
        ("report-excl1.json", "d47528701a6bf746"),
    ]),
    ("forecast", &[
        ("forecast-samples.tsv", "e477eaa254909a73"),
        ("forecast.json", "d84f5c750c68d59f"),
        ("forecast/daily_cases.tsv", "ddeb925925f18839"),
        ("forecast/survey.tsv", "694c705a6c201294"),
        ("report.json", "d47528701a6bf746"),
    ]),
    ("one_step", &[
        ("observed.json", "5d45d37e3a179a2b"),
        ("observed/daily_cases.tsv", "ee9d3cb76b7457fc"),
        ("observed/survey.tsv", "739e02d5718619d2"),
        ("observed/weekly_cases.tsv", "884f26939b41d3c2"),
        ("predictive.json", "0ff4348a961f8a9a"),
        ("predictive/daily_cases.tsv", "000af757269a046b"),
        ("predictive/survey.tsv", "59e3e87d969c0d3c"),
        ("predictive/weekly_cases.tsv", "3b870200d27a5302"),
        ("report.json", "d47528701a6bf746"),
    ]),
    ("quantities", &[
        ("observed.json", "d5f50fb8a31bdf75"),
        ("observed/daily_cases.tsv", "ee9d3cb76b7457fc"),
        ("observed/survey.tsv", "739e02d5718619d2"),
        ("observed/weekly_cases.tsv", "884f26939b41d3c2"),
        ("predictive.json", "21c7dcdfcb802175"),
        ("predictive/daily_cases.tsv", "9602066714780373"),
        ("predictive/survey.tsv", "7e10f0ff10ea6c71"),
        ("predictive/weekly_cases.tsv", "950f5ee11b86ef0a"),
        ("quantities-38c32924.json", "760ad761ec3f118d"),
        ("quantities-38c32924/peak_R.tsv", "35c08c2b2cfab668"),
        ("report.json", "d47528701a6bf746"),
    ]),
    ("scenario", &[
        ("observed.json", "8c43589857002a12"),
        ("observed/daily_cases.tsv", "ee9d3cb76b7457fc"),
        ("observed/survey.tsv", "739e02d5718619d2"),
        ("observed/weekly_cases.tsv", "884f26939b41d3c2"),
        ("predictive.json", "a33a48ab9c5b59c7"),
        ("predictive/daily_cases.tsv", "caf63fa92179de78"),
        ("predictive/survey.tsv", "6c84b9ac3579308e"),
        ("predictive/weekly_cases.tsv", "a7704544263eade8"),
        ("quantities-model/peak_I.tsv", "052077fa3035b289"),
        ("quantities.json", "bf916eed6f943d43"),
        ("report.json", "d47528701a6bf746"),
    ]),
    ("sweep", &[
        ("observed.json", "d3764863a400a0da"),
        ("observed/daily_cases.tsv", "ee9d3cb76b7457fc"),
        ("observed/survey.tsv", "739e02d5718619d2"),
        ("observed/weekly_cases.tsv", "884f26939b41d3c2"),
        ("predictive.json", "2ea0bbb2c40c8bd5"),
        ("predictive/daily_cases.tsv", "6fc04da9b451e4b1"),
        ("predictive/survey.tsv", "6c1f1df987847fd0"),
        ("predictive/weekly_cases.tsv", "da7d2ce85347b11c"),
        ("quantities-model/peak_I.tsv", "7b0623e902b6352c"),
        ("quantities.json", "b22fead586ddd990"),
        ("report.json", "d47528701a6bf746"),
    ]),
];
