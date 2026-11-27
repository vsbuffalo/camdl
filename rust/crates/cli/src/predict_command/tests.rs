//! `camdl predict` on the run store, checked by running the release binary:
//! every declared input against its class (`store::check_input_classes`),
//! and the bytes each prediction computes against pins.
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
use crate::input_fixtures::{predict_base, predict_fixed_base, predict_from};

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

/// A two-patch SIR model whose infection rate reads an external weight
/// table `w`, for the check of `--table`.
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

/// The fixture model with its infection rate scaled by `pop`, a forcing read
/// linearly from a table with rows to day 63, and by `week`, a periodic
/// forcing.
fn forced_model() -> String {
    let m = model();
    let forced = m.replace("beta * S * I / N", "beta * pop * week * S * I / N").replace(
        "transitions {",
        "forcing {\n  pop : interpolated 'ratio {\n    times  = [0 'days, 20 'days, 30 'days, 63 'days]\n    \
         values = [1.0, 1.2, 0.7, 1.0]\n    method = linear\n  }\n  week : periodic 'ratio {\n    \
         period = 7 'days\n    values = [1.0, 0.9]\n  }\n}\n\ntransitions {",
    );
    assert!(forced.contains("beta * pop * week") && forced.contains("forcing {"), "the fixture's infection rate moved");
    forced
}

/// The forced fixture model with `pop` a step table through rows at days −10,
/// 20, 40 and 63, read at `t − tau` for a parameter `tau : duration`, which
/// the point `point_lagged.toml` sets to 5 days. The first row is before day
/// 0 so that the run's start, which reads the table at −5, is inside it.
fn lagged_model() -> String {
    let m = forced_model();
    let lagged = m
        .replace("[0 'days, 20 'days, 30 'days, 63 'days]", "[-10 'days, 20 'days, 40 'days, 63 'days]")
        .replace("    method = linear\n  }", "    method = constant\n    lag    = tau\n  }")
        .replace("  I0    : count\n", "  I0    : count\n  tau   : duration\n");
    assert!(lagged.contains("lag    = tau") && lagged.contains("tau   : duration") && lagged.contains("40 'days"),
        "the forced fixture moved");
    lagged
}

/// The point the fixture model is predicted at, `@fit`'s fixed values with
/// `beta` at 0.5.
const POINT: &str = "beta = 0.5\ngamma = 0.25\nrho = 0.5\nk = 10.0\nN0 = 1000\nI0 = 5\n";

/// The files of the fixed-point predictions: two points of the fixture
/// model, its daily cases with other counts, one file holding the daily
/// cases and the survey (for `--obs`), and the patch model with its data,
/// point and two weight tables.
fn fixed_point_files() -> Vec<(&'static str, String)> {
    let daily = std::fs::read_to_string(Path::new(FORECAST).join("daily_cases.tsv")).unwrap();
    let daily2: String = daily
        .lines()
        .enumerate()
        .map(|(i, l)| match (i, l.split_once('\t')) {
            (0, _) | (_, None) => format!("{l}\n"),
            (_, Some((t, y))) => format!("{t}\t{}\n", y.parse::<u64>().unwrap() + 1),
        })
        .collect();
    let mixed = (1..=28).fold("time\tdaily_cases\tpositives\ttested\n".to_string(), |s, t| {
        s + &format!("{t}\t{}\t{}\t40\n", 2 + t % 5, t % 7)
    });
    vec![
        ("point.toml", POINT.to_string()),
        ("point2.toml", POINT.replace("beta = 0.5", "beta = 0.55")),
        ("daily_cases2.tsv", daily2),
        ("mixed.tsv", mixed),
        ("patch.camdl", PATCH.to_string()),
        ("forced.camdl", forced_model()),
        ("lagged.camdl", lagged_model()),
        ("fit_forced.toml", FIT.replace("model.camdl", "forced.camdl")),
        ("point_lagged.toml", format!("{POINT}tau = 5.0\n")),
        ("patch.toml", "beta = 0.4\ngamma = 0.2\nrho = 0.5\n".into()),
        ("patch_cases.tsv", (1..=10).fold("time\tcases\n".to_string(), |s, t| s + &format!("{t}\t{}\n", 2 + t))),
        ("w1.tsv", "1.0\n0.5\n".into()),
        ("w2.tsv", "1.0\n0.7\n".into()),
    ]
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
        std::fs::write(dir.join("reported.camdl"), model().replace("peak_I = max(I)", "peak_R = max(R)")).unwrap();
        std::fs::write(dir.join("design.tsv"), "time\ttested\n28\t60\n35\t70\n").unwrap();
        std::fs::write(dir.join("design2.tsv"), "time\ttested\n28\t61\n35\t70\n").unwrap();
        for (name, text) in fixed_point_files() {
            std::fs::write(dir.join(name), text).unwrap();
        }
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
                covariate,
                n_draws,
                particles,
                seed,
                output,
                manifest,
                link_root,
                force,
                label,
                store,
            },
        posterior: crate::args::PosteriorFlags { by_chain, exclude_chains, chains, realised_design, model },
        fixed_point:
            crate::args::FixedPointFlags {
                params,
                param,
                table,
                data,
                obs,
                holdout_after,
                backend,
                dt,
                integrator,
                binomial,
                allow_degenerate_rates,
                time_format,
            },
    } = args;
    let mut v: Vec<String> = vec!["predict".into(), source.clone().expect("a run or a model")];
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
    for c in covariate {
        opt("--covariate", Some(format!("{}={}", c.forcing, c.statement.as_str())));
    }
    opt("--n-draws", n_draws.map(|x| x.to_string()));
    opt("--particles", particles.map(|x| x.to_string()));
    opt("--seed", seed.map(|x| x.to_string()));
    opt("--exclude-chains", exclude_chains.clone());
    opt("--chains", chains.clone());
    opt("--model", model.as_ref().map(|p| p.to_string_lossy().into_owned()));
    for p in params {
        opt("--params", Some(p.to_string_lossy().into_owned()));
    }
    for p in param {
        opt("--param", Some(format!("{}={}", p.name, p.value)));
    }
    for t in table {
        opt("--table", Some(format!("{}={}", t.name, t.path.display())));
    }
    for d in data {
        opt("--data", Some(match d {
            crate::args::types::DataSpec::Single(p) => p.to_string_lossy().into_owned(),
            crate::args::types::DataSpec::Named { name, path } => format!("{name}={}", path.display()),
        }));
    }
    opt("--obs", obs.clone());
    opt("-o", output.as_ref().map(|p| p.to_string_lossy().into_owned()));
    opt("--label", label.clone());
    opt("--store", store.as_ref().map(|p| p.to_string_lossy().into_owned()));
    opt("--manifest", manifest.as_ref().map(|p| p.to_string_lossy().into_owned()));
    opt("--link-root", link_root.as_ref().map(|p| p.to_string_lossy().into_owned()));
    opt("--holdout-after", holdout_after.clone());
    opt("--backend", backend.map(|b| b.as_str().to_string()));
    opt("--dt", dt.map(|x| x.to_string()));
    opt("--integrator", integrator.map(|i| serde_json::to_value(i).unwrap().as_str().unwrap().to_string()));
    opt("--binomial", binomial.map(|b| b.to_string()));
    if *time_format != crate::caltime_load::TimeFormat::Auto {
        opt("--time-format", Some(serde_json::to_value(time_format).unwrap().as_str().unwrap().to_string()));
    }
    let mut flag = |set: bool, name: &str| {
        if set {
            v.push(name.into());
        }
    };
    flag(*allow_forcing_extrapolation, "--allow-forcing-extrapolation");
    flag(*by_chain, "--by-chain");
    flag(*allow_degenerate_rates, "--allow-degenerate-rates");
    flag(*force, "--force");
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
        if let Some((_, rest)) = line.split_once("predict: --data checked under --time-format ") {
            let f = rest.split_whitespace().next().unwrap_or_default().to_lowercase();
            seen.entry("used.args.fixed_point.time_format".into()).or_default().insert(f);
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
            "covariate" => predict_from(&forced_forecast(&["--covariate", "pop=prospective"])),
            _ if name.starts_with("args.fixed_point.") => match field {
                "table" => predict_from(&[
                    "patch.camdl", "--params", "patch.toml", "--table", "w=w1.tsv", "--data", "cases=patch_cases.tsv",
                    "--mode", "free_forward", "--n-draws", "4",
                ]),
                // `--integrator` runs only on the ODE backend.
                "integrator" => predict_from(&[
                    "model.camdl", "--params", "point.toml", "--data", "daily_cases=daily_cases.tsv", "--mode",
                    "free_forward", "--n-draws", "4", "--backend", "ode",
                ]),
                "obs" => predict_from(&[
                    "model.camdl", "--params", "point.toml", "--data", "mixed.tsv", "--obs", "daily_cases", "--mode",
                    "free_forward", "--n-draws", "4",
                ]),
                _ => predict_fixed_base(),
            },
            // A model file's compile is the fixed point's model.
            "no_licm" => predict_from(&[
                "hoisted.camdl", "--params", "point.toml", "--data", "daily_cases=daily_cases.tsv", "--mode", "free_forward",
                "--n-draws", "4",
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
            Class::Export if field.name == "args.flags.output" => {
                let mut inv = self.base.clone();
                inv.args.flags.output = Some("exported".into());
                inv
            }
            Class::Export if field.name == "args.flags.manifest" => {
                let mut inv = self.base.clone();
                inv.args.flags.manifest = Some("m.json".into());
                inv
            }
            Class::Export if field.name == "args.flags.link_root" => {
                let mut inv = self.base.clone();
                inv.args.flags.link_root = Some("links".into());
                inv
            }
            class => panic!("no second value for {} ({class:?})", field.name),
        }
    }

    fn exercised(&self, field: &FieldInfo) -> (PredictInvocation, PredictInvocation) {
        match field.name.strip_prefix("args.fixed_point.") {
            Some(f) => {
                let (x, y) = <crate::args::FixedPointFlags as Inputs>::exercised(f);
                let (mut a, mut b) = (predict_fixed_base(), predict_fixed_base());
                a.args.fixed_point = x;
                b.args.fixed_point = y;
                (a, b)
            }
            None => <PredictInvocation as Inputs>::exercised(field.name),
        }
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
        let out = camdl(cwd.path(), &root, &argv(invocation), &[("RUST_LOG", "camdl::util=debug,camdl::fit::fixed_point=debug")]);
        if !out.status.success() {
            return Err(store::Error::Parse {
                what: format!("camdl {:?}", argv(invocation)),
                why: String::from_utf8_lossy(&out.stderr).into_owned(),
            });
        }
        let mut effects = crate::test_support::recorded_execution(&root, crate::fit_store::Prediction::KIND);
        effects.extend(used(&String::from_utf8_lossy(&out.stderr), &cwd.path().join("ir-cache")));
        // `-o DIR` copies the prediction's result files: each export names
        // the stored file it copies.
        let results = observe(&root);
        let mut exports = BTreeMap::new();
        let mut copies = BTreeMap::new();
        if let Some(dir) = &invocation.args.flags.output {
            let prediction = results
                .iter()
                .find(|r| r.kind == crate::fit_store::Prediction::KIND)
                .map(|r| r.run_id)
                .expect("a stored prediction");
            let mut stack = vec![cwd.path().join(dir)];
            while let Some(d) = stack.pop() {
                for e in std::fs::read_dir(&d).into_iter().flatten().flatten() {
                    let p = e.path();
                    if p.is_dir() {
                        stack.push(p);
                        continue;
                    }
                    let rel = p.strip_prefix(cwd.path()).unwrap().to_path_buf();
                    let file = p.strip_prefix(cwd.path().join(dir)).unwrap().to_string_lossy().into_owned();
                    exports.insert(rel.clone(), Key::digest_bytes(&std::fs::read(&p).unwrap()));
                    copies.insert(rel, store::StoredFile { run_id: prediction, file });
                }
            }
        }
        if let Some(path) = &invocation.args.flags.manifest {
            let bytes = std::fs::read(cwd.path().join(path)).unwrap_or_default();
            exports.insert(path.clone(), Key::digest_bytes(&bytes));
        }
        if let Some(links) = &invocation.args.flags.link_root {
            crate::test_support::link_exports(cwd.path(), links, &mut exports, &mut copies);
        }
        Ok(Observed { results, exports, copies, effects })
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

/// The forecast of the forced fixture model from a fixed point: the daily
/// cases to day 24 assimilated and two weeks of reach, with `extra`.
fn forced_forecast(extra: &[&'static str]) -> Vec<&'static str> {
    let mut line = vec![
        "forced.camdl", "--params", "point.toml", "--data", "daily_cases=daily_cases.tsv", "--holdout-after", "24",
        "--horizon", "2 weeks", "--n-draws", "4", "--particles", "20",
    ];
    line.extend_from_slice(extra);
    line
}

/// A forecast that reads the table forcing `pop` after its origin is refused
/// with no statement, naming the forcing, the origin, the reach and the
/// table's end; a statement for the periodic forcing `week`, which needs
/// none, is refused, as is `--covariate` outside a forecast. Each of the
/// three statements then runs, as a prediction of its own; `realised` labels
/// the forecast's manifest, and `hold` records the value it carries.
#[test]
fn a_forecast_reading_a_table_forcing_after_its_origin_needs_a_statement() {
    let fresh = tempfile::tempdir().unwrap();
    let root = fresh.path().join("camdl-store");
    let cwd = workdir(&root);
    let run = |line: &[&str]| {
        let mut v = vec!["predict"];
        v.extend_from_slice(line);
        predict_in(cwd.path(), &root, &v)
    };
    let refused = |line: &[&str], want: &[&str]| {
        let out = run(line);
        let err = String::from_utf8_lossy(&out.stderr).into_owned();
        assert!(!out.status.success(), "{line:?} ran:\n{err}");
        for w in want {
            assert!(err.contains(w), "{line:?}: missing {w:?} in:\n{err}");
        }
        assert!(predictions(&root).is_empty(), "{line:?}: nothing is stored");
    };
    refused(&forced_forecast(&[]), &[
        "forcing `pop` is read after the forecast origin and nothing says how.",
        "origin t = 24, reach 2 weeks (to t = 38); its table has rows to t = 63.",
        "--covariate pop=prospective",
        "--covariate pop=realised",
        "--covariate pop=hold",
    ]);
    refused(&forced_forecast(&["--covariate", "pop=prospective", "--covariate", "week=hold"]), &[
        "forcing `week` needs no statement",
    ]);
    refused(
        &["forced.camdl", "--params", "point.toml", "--data", "daily_cases=daily_cases.tsv", "--mode", "free_forward",
          "--n-draws", "4", "--covariate", "pop=hold"],
        &["--covariate pop=hold", "not a forecast"],
    );

    let mut manifests = BTreeMap::new();
    let mut tables = BTreeMap::new();
    for statement in ["pop=prospective", "pop=realised", "pop=hold"] {
        let before: std::collections::BTreeSet<PathBuf> = predictions(&root).into_iter().map(|(p, _)| p).collect();
        let out = run(&forced_forecast(&["--covariate", statement]));
        let err = String::from_utf8_lossy(&out.stderr);
        assert!(out.status.success(), "--covariate {statement}:\n{err}");
        let new: Vec<PathBuf> = predictions(&root).into_iter().map(|(p, _)| p).filter(|p| !before.contains(p)).collect();
        assert_eq!(new.len(), 1, "--covariate {statement} is a prediction of its own: {new:?}");
        let doc: serde_json::Value = serde_json::from_slice(&std::fs::read(new[0].join("forecast.json")).unwrap()).unwrap();
        manifests.insert(statement, doc);
        tables.insert(statement, std::fs::read_to_string(new[0].join("forecast/daily_cases.tsv")).unwrap());
    }
    assert_eq!(tables["pop=prospective"], tables["pop=realised"], "both read the table as it is");
    assert_ne!(tables["pop=prospective"], tables["pop=hold"], "hold forecasts with pop held at its value at the origin");
    for (statement, doc) in &manifests {
        let block = &doc["covariates"];
        assert_eq!(block["forcings"][0]["forcing"], "pop", "{statement}: {block:#}");
        assert_eq!(&format!("pop={}", block["forcings"][0]["statement"].as_str().unwrap()), statement);
        assert_eq!(doc["options"]["forecast"]["covariates"]["pop"], statement.trim_start_matches("pop="));
    }
    assert_eq!(
        manifests["pop=realised"]["covariates"]["label"],
        crate::fit::forecast_covariates::REALISED_LABEL,
        "a realised statement labels the forecast"
    );
    assert!(manifests["pop=prospective"]["covariates"].get("label").is_none());
    // pop(24) on the segment from (20, 1.2) to (30, 0.7).
    let held = manifests["pop=hold"]["covariates"]["forcings"][0]["held"]["value"].as_f64().unwrap();
    assert!((held - 1.0).abs() < 1e-12, "hold carries pop's value at the origin: {held}");
    // `pop`'s rows are at 0, 20, 30 and 63: the origin, day 24, falls between
    // two rows, so its values before the origin depend on the row at 30, and
    // every forecast records it beside its statement.
    for (statement, doc) in &manifests {
        let interp = &doc["covariates"]["interpolation"];
        assert_eq!(interp[0]["forcing"], "pop", "{statement}: {:#}", doc["covariates"]);
        assert_eq!(interp[0]["method"], "linear");
        assert_eq!(interp[0]["rows_after_cut"], 1);
        assert_eq!(interp[0]["cut"]["t"], 24.0);
        assert_eq!(interp.as_array().unwrap().len(), 1, "{statement}: `week` is periodic and has no table");
    }
    // The filter before the origin is the same under every statement, so
    // each draw's last assimilation has the same effective sample size.
    for statement in ["pop=realised", "pop=hold"] {
        assert_eq!(
            manifests[statement]["filter_diagnostics"], manifests["pop=prospective"]["filter_diagnostics"],
            "{statement}: the filtering before the origin moved"
        );
    }
}

/// A fit of the forced fixture model, its rows after day 24 held out: the
/// training extent, day 24, falls between `pop`'s rows at 20 and 30, so the
/// fit says once that `pop`'s values before the cut depend on 1 row after it,
/// and records it in the run's `fit.meta.json`; `week`, periodic, is not
/// named.
#[test]
fn a_fit_records_a_linear_table_whose_training_extent_falls_between_rows() {
    let fresh = tempfile::tempdir().unwrap();
    let root = fresh.path().join("camdl-store");
    let cwd = workdir(&root);
    let args: Vec<String> =
        ["fit", "run", "fit_forced.toml", "--label", "forced", "--progress", "none"].map(String::from).into();
    let out = camdl(cwd.path(), &root, &args, &[]);
    let err = String::from_utf8_lossy(&out.stderr);
    assert!(out.status.success(), "{err}");
    let note = "note: forcing `pop` (linear table): values before the cut (t = 24, the training extent) depend on 1 row after it";
    assert_eq!(err.matches(note).count(), 1, "said once:\n{err}");
    assert!(!err.contains("forcing `week`"), "{err}");
    let mut metas = Vec::new();
    let mut stack = vec![root.clone()];
    while let Some(d) = stack.pop() {
        for e in std::fs::read_dir(&d).into_iter().flatten().flatten() {
            let p = e.path();
            if p.is_dir() {
                stack.push(p);
            } else if p.file_name().is_some_and(|n| n == "fit.meta.json") {
                let doc: serde_json::Value = serde_json::from_slice(&std::fs::read(&p).unwrap()).unwrap();
                if doc["forcing_interpolation"].is_array() {
                    metas.push(doc);
                }
            }
        }
    }
    assert_eq!(metas.len(), 1, "the forced fit's run records it, and no fixture fit does");
    assert_eq!(
        metas[0]["forcing_interpolation"],
        serde_json::json!([{ "forcing": "pop", "method": "linear", "cut": { "t": 24.0 }, "rows_after_cut": 1 }])
    );
}

/// A step table whose next row after the origin (day 40) is past a two-week
/// reach from day 24, read at `t − tau` with `tau` a `duration` parameter at
/// 5 days. The forecast classifier takes every row after the origin as read
/// under a parameter lag, which may be shorter than the reach, so the
/// forecast is refused with no statement. A parameter lag is at least 0, so
/// `hold` is offered and runs, and the filtering before the origin is the
/// same under `hold` as under `prospective`.
#[test]
fn a_parameter_lag_reads_a_step_tables_rows_after_the_origin() {
    let fresh = tempfile::tempdir().unwrap();
    let root = fresh.path().join("camdl-store");
    let cwd = workdir(&root);
    let line = |extra: &[&'static str]| {
        let mut v = vec![
            "predict", "lagged.camdl", "--params", "point_lagged.toml", "--data", "daily_cases=daily_cases.tsv",
            "--holdout-after", "24", "--horizon", "2 weeks", "--n-draws", "4", "--particles", "20",
        ];
        v.extend_from_slice(extra);
        v
    };
    let refused = |extra: &[&'static str], want: &[&str], not: &[&str]| {
        let out = predict_in(cwd.path(), &root, &line(extra));
        let err = String::from_utf8_lossy(&out.stderr).into_owned();
        assert!(!out.status.success(), "{extra:?} ran:\n{err}");
        for w in want {
            assert!(err.contains(w), "{extra:?}: missing {w:?} in:\n{err}");
        }
        for n in not {
            assert!(!err.contains(n), "{extra:?}: {n:?} in:\n{err}");
        }
    };
    refused(&[], &["forcing `pop` is read after the forecast origin", "Its lag is a parameter", "--covariate pop=hold "], &[
        "`hold` is not offered",
        "negative",
    ]);
    assert!(predictions(&root).is_empty(), "nothing is stored");
    let mut manifests = BTreeMap::new();
    for statement in ["pop=prospective", "pop=hold"] {
        let before: std::collections::BTreeSet<PathBuf> = predictions(&root).into_iter().map(|(p, _)| p).collect();
        let out = predict_in(cwd.path(), &root, &line(&["--covariate", statement]));
        assert!(out.status.success(), "--covariate {statement}:\n{}", String::from_utf8_lossy(&out.stderr));
        let new: Vec<PathBuf> = predictions(&root).into_iter().map(|(p, _)| p).filter(|p| !before.contains(p)).collect();
        assert_eq!(new.len(), 1, "--covariate {statement} is a prediction of its own: {new:?}");
        let doc: serde_json::Value = serde_json::from_slice(&std::fs::read(new[0].join("forecast.json")).unwrap()).unwrap();
        manifests.insert(statement, doc);
    }
    assert_eq!(manifests["pop=hold"]["covariates"]["forcings"][0]["lag"], "parameter");
    assert_eq!(
        manifests["pop=hold"]["filter_diagnostics"], manifests["pop=prospective"]["filter_diagnostics"],
        "a hold under a parameter lag moved the filtering before the origin"
    );
}

/// A lag parameter whose declared range admits negative values, which would
/// read the forcing's rows after the origin while filtering before it, is
/// refused when the model compiles (E425), before any forecast runs.
#[test]
fn a_lag_parameter_that_can_be_negative_is_refused_at_compile_time() {
    let fx = fixture();
    let dir = tempfile::tempdir().unwrap();
    let model = dir.path().join("lagged_real.camdl");
    let src = lagged_model().replace("  tau   : duration\n", "  tau   : real dim T\n");
    assert!(src.contains("tau   : real dim T"), "the lagged fixture moved");
    std::fs::write(&model, src).unwrap();
    let out = Command::new(&fx.camdlc).arg("check").arg(&model).output().unwrap();
    let err = format!("{}{}", String::from_utf8_lossy(&out.stderr), String::from_utf8_lossy(&out.stdout));
    assert!(!out.status.success(), "a `real` lag parameter compiled:\n{err}");
    assert!(err.contains("E425") && err.contains("reads parameter 'tau', declared `real` with no lower bound"),
        "the refusal is E425 naming `tau`:\n{err}");
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

/// A prediction from an optimizer's run (IF2) is a prediction of its winning
/// point, once, labelled `plug_in` as a params file's is; its score's
/// conditioning is verified from the run's stored training rows
/// (`hold_out_tail` at the training extent, day 24), as a sampler's is, not
/// declared or unknown. `--n-draws` above one is refused: there is one point.
#[test]
fn an_optimizer_run_is_predicted_at_its_winner_with_verified_conditioning() {
    let fresh = tempfile::tempdir().unwrap();
    let root = fresh.path().join("camdl-store");
    let cwd = workdir(&root);
    let if2 = FIT.split("[method]").next().unwrap().to_string()
        + "[method]\nalgorithm = \"if2\"\nbackend = \"chain_binomial\"\nchains = 2\nparticles = 30\niterations = 3\ncooling = 0.9\n";
    std::fs::write(cwd.path().join("fit_if2.toml"), if2).unwrap();
    let run = |args: &[&str]| camdl(cwd.path(), &root, &args.iter().map(|s| s.to_string()).collect::<Vec<_>>(), &[]);
    let fitted = run(&["fit", "run", "fit_if2.toml", "--label", "scout", "--progress", "none"]);
    assert!(fitted.status.success(), "{}", String::from_utf8_lossy(&fitted.stderr));
    let out = run(&["predict", "@scout", "--mode", "one_step", "--n-draws", "2"]);
    let err = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(1), "{err}");
    assert!(err.contains("--n-draws 2") && err.contains("optimizer fit"), "{err}");

    let out = run(&["predict", "@scout", "--mode", "one_step", "--particles", "20", "--label", "point"]);
    assert!(out.status.success(), "{}", String::from_utf8_lossy(&out.stderr));
    let (folder, _) = predictions(&root).into_iter().next().expect("a prediction");
    let band = tsv_rows(&std::fs::read_to_string(folder.join("predictive/daily_cases.tsv")).unwrap());
    assert!(!band.is_empty() && band.iter().all(|r| r["treatment"] == "plug_in"), "{band:?}");
    let out = run(&[
        "score", "@point", "--data", "daily_cases=daily_cases.tsv", "--data", "weekly_cases=weekly_cases.tsv",
        "--data", "survey=survey.tsv",
    ]);
    assert!(out.status.success(), "{}", String::from_utf8_lossy(&out.stderr));
    let score = folders(&root).into_iter().find(|(_, id, _)| id["kind"] == crate::fit_store::Score::KIND).unwrap().0;
    let manifest: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(score.join("score.json")).unwrap()).unwrap();
    assert_eq!(manifest["parameters"], "winner", "{manifest:#}");
    assert_eq!(manifest["modes"]["one_step"]["provenance"], "plug_in", "{manifest:#}");
    assert_eq!(manifest["conditioning"]["hold_out_tail"]["train_end"], serde_json::json!(24.0), "{manifest:#}");
}

/// The bytes each prediction computes, pinned: the digest of each
/// result file, with the fit run's id written `<run>`, and the lines that
/// name the build dropped (`engine_version`, and `address_key`, which hashes
/// it). The run id holds the build too, so it is checked to be the run's own
/// rather than pinned.
///
/// The pins are a regression pin per platform, keyed by `target_os` and
/// `target_arch`, because the digests depend on the platform's libm; linux
/// arm64 computes the macOS arm64 table. On a platform with no pins the test
/// fails and prints the table to add.
#[test]
fn each_prediction_computes_its_pinned_bytes() {
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

/// The pins of [`each_prediction_computes_its_pinned_bytes`], per
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
        ("predictive.json", "905583d6696d580b"),
        ("predictive/daily_cases.tsv", "5a36ac41383fcb03"),
        ("predictive/survey.tsv", "94018928058468fd"),
        ("predictive/weekly_cases.tsv", "f0103657206f0947"),
        ("quantities-model/peak_I.tsv", "1e14411cf72c645f"),
        ("quantities.json", "daf5039b8d4a73b3"),
        ("report.json", "d47528701a6bf746"),
    ]),
    ("by_chain", &[
        ("observed.json", "6aa2e125b201fcc3"),
        ("observed/daily_cases.tsv", "ee9d3cb76b7457fc"),
        ("observed/survey.tsv", "739e02d5718619d2"),
        ("observed/weekly_cases.tsv", "884f26939b41d3c2"),
        ("predictive-excl1.json", "1b4d8be81e620517"),
        ("predictive-excl1/daily_cases.tsv", "8bb55a7dec5076d5"),
        ("predictive-excl1/survey.tsv", "cf8a7ef2889d33ca"),
        ("predictive-excl1/weekly_cases.tsv", "138201d8e42e0e8b"),
        ("quantities-excl1.json", "daf5039b8d4a73b3"),
        ("quantities-model-excl1/peak_I.tsv", "291336833bcaf220"),
        ("report-excl1.json", "d47528701a6bf746"),
    ]),
    ("forecast", &[
        ("forecast-samples.tsv", "6a967b3d7a0da464"),
        ("forecast.json", "c29fb0ae029dface"),
        ("forecast/daily_cases.tsv", "5341c4b2397f0319"),
        ("forecast/survey.tsv", "ec4ee1c0b309eac2"),
        ("report.json", "d47528701a6bf746"),
    ]),
    ("one_step", &[
        ("observed.json", "5d45d37e3a179a2b"),
        ("observed/daily_cases.tsv", "ee9d3cb76b7457fc"),
        ("observed/survey.tsv", "739e02d5718619d2"),
        ("observed/weekly_cases.tsv", "884f26939b41d3c2"),
        ("predictive.json", "4226ef39718fdcfc"),
        ("predictive/daily_cases.tsv", "f96fd17cceb68f09"),
        ("predictive/survey.tsv", "c432e7aa21f31fc0"),
        ("predictive/weekly_cases.tsv", "23ebe0adc9bbd05e"),
        ("report.json", "d47528701a6bf746"),
    ]),
    ("scenario", &[
        ("observed.json", "8c43589857002a12"),
        ("observed/daily_cases.tsv", "ee9d3cb76b7457fc"),
        ("observed/survey.tsv", "739e02d5718619d2"),
        ("observed/weekly_cases.tsv", "884f26939b41d3c2"),
        ("predictive.json", "ef5142560d1bdda8"),
        ("predictive/daily_cases.tsv", "80f1133f9a90e505"),
        ("predictive/survey.tsv", "250926969b132326"),
        ("predictive/weekly_cases.tsv", "aa6055e15604bde6"),
        ("quantities-model/peak_I.tsv", "052077fa3035b289"),
        ("quantities.json", "bf916eed6f943d43"),
        ("report.json", "d47528701a6bf746"),
    ]),
    ("sweep", &[
        ("observed.json", "d3764863a400a0da"),
        ("observed/daily_cases.tsv", "ee9d3cb76b7457fc"),
        ("observed/survey.tsv", "739e02d5718619d2"),
        ("observed/weekly_cases.tsv", "884f26939b41d3c2"),
        ("predictive.json", "cbc880dd870f205b"),
        ("predictive/daily_cases.tsv", "02aa9d7af9cba827"),
        ("predictive/survey.tsv", "cc0a39caf5ed39db"),
        ("predictive/weekly_cases.tsv", "e6f6e8145ee4455c"),
        ("quantities-model/peak_I.tsv", "7b0623e902b6352c"),
        ("quantities.json", "b22fead586ddd990"),
        ("report.json", "d47528701a6bf746"),
    ]),
];

// ── Fixed-point predictions ──────────────────────────────────────────────────

/// `camdl predict` in a fresh copy of the fixture, as a line of words.
fn predict_in(cwd: &Path, root: &Path, line: &[&str]) -> Output {
    let mut args: Vec<String> = line.iter().map(|s| s.to_string()).collect();
    args.extend(["--progress".to_string(), "none".to_string()]);
    camdl(cwd, root, &args, &[])
}

/// Every stored prediction under `root`: its folder and identity record.
fn predictions(root: &Path) -> Vec<(PathBuf, serde_json::Value)> {
    folders(root)
        .into_iter()
        .filter(|(_, id, _)| id["kind"] == crate::fit_store::Prediction::KIND)
        .map(|(p, id, _)| (p, id))
        .collect()
}

/// The fixed-point base line: four copies of `point.toml`, free-forward,
/// conditioned on the daily cases.
const FIXED: &[&str] = &[
    "predict", "model.camdl", "--params", "point.toml", "--data", "daily_cases=daily_cases.tsv", "--mode", "free_forward",
    "--n-draws", "4",
];

/// The rows of a TSV, as maps from its header.
fn tsv_rows(text: &str) -> Vec<BTreeMap<String, String>> {
    let mut lines = text.lines().filter(|l| !l.starts_with('#'));
    let header: Vec<&str> = lines.next().unwrap().split('\t').collect();
    lines
        .filter(|l| !l.is_empty())
        .map(|l| header.iter().zip(l.split('\t')).map(|(h, v)| (h.to_string(), v.to_string())).collect())
        .collect()
}

/// A fixed-point prediction is stored under `prediction/`, grouped by its
/// model and named by its mode; its identity record is the `FixedPoint`
/// variant; every band row is `plug_in` under the `(none)` arm, with the
/// columns that compare chains or count effective draws empty.
#[test]
fn a_fixed_point_prediction_is_stored_under_prediction_and_labelled_plug_in() {
    let fresh = tempfile::tempdir().unwrap();
    let root = fresh.path().join("camdl-store");
    let cwd = workdir(&root);
    let out = predict_in(cwd.path(), &root, &[&FIXED[..2], &["--params", "point.toml", "--data", "daily_cases=daily_cases.tsv", "--n-draws", "4"]].concat());
    let err = String::from_utf8_lossy(&out.stderr);
    assert!(out.status.success(), "{err}");
    let found = predictions(&root);
    let fixed: Vec<&(PathBuf, serde_json::Value)> =
        found.iter().filter(|(_, id)| id["identity"]["source"].get("FixedPoint").is_some()).collect();
    assert_eq!(fixed.len(), 1, "one fixed-point prediction: {found:#?}");
    let (folder, id) = fixed[0];
    let parts: Vec<String> = folder.components().rev().take(3).map(|c| c.as_os_str().to_string_lossy().into_owned()).collect();
    assert!(parts[0].starts_with("all-"), "named by the mode the default resolves to on chain_binomial: {parts:?}");
    assert!(parts[1].starts_with("model-"), "grouped by the model: {parts:?}");
    assert_eq!(parts[2], "prediction", "{parts:?}");
    let source = &id["identity"]["source"]["FixedPoint"];
    assert_eq!(source["params"]["values"]["beta"], serde_json::json!(0.5), "{source:#}");
    assert!(source["trained_through"].is_null(), "no --holdout-after: every row is conditioned on");
    let rows = tsv_rows(&std::fs::read_to_string(folder.join("predictive/daily_cases.tsv")).unwrap());
    assert!(!rows.is_empty());
    for r in &rows {
        assert_eq!(r["treatment"], "plug_in", "{r:?}");
        assert_eq!(r["scenario"], "(none)", "{r:?}");
        for c in ["fit_rhat_max", "fit_ess_min", "rhat_mean", "ess_mean", "rhat_pred", "ess_pred"] {
            assert_eq!(r[c], "", "{c} is empty for a fixed point: {r:?}");
        }
    }
    assert!(rows.iter().any(|r| r["mode"] == "one_step") && rows.iter().any(|r| r["mode"] == "free_forward"));
    let ff: Vec<&BTreeMap<String, String>> = rows.iter().filter(|r| r["mode"] == "free_forward").collect();
    assert!(ff.iter().all(|r| r["n_draws"] == "4"), "the band pools the four copies");
    // Outside a forecast a row has no periods_ahead, and every band is of
    // the observation.
    assert!(rows.iter().all(|r| r["periods_ahead"].is_empty()), "no periods_ahead outside a forecast");
    assert!(rows.iter().all(|r| r["quantity"] == "observation"), "every band is of the observation");
    // No --holdout-after: which rows the point was fitted to is not known, so
    // every data row is `unknown`, and every row past them is `future`.
    assert!(rows.iter().any(|r| r["role"] == "unknown"), "{rows:?}");
    assert!(rows.iter().all(|r| r["role"] == "unknown" || r["role"] == "future"), "no row is trained or held out");
    // `camdl show` says what the bands carry.
    let show = camdl(cwd.path(), &root, &["show".to_string(), folder.to_string_lossy().into_owned()], &[]);
    let text = String::from_utf8_lossy(&show.stdout);
    assert!(show.status.success(), "{}", String::from_utf8_lossy(&show.stderr));
    assert!(text.contains("no parameter uncertainty"), "{text}");
}

/// A fixed-point prediction already stored is reused on a repeat, without
/// being computed again.
#[test]
fn a_fixed_point_prediction_is_reused_on_repeat() {
    let fresh = tempfile::tempdir().unwrap();
    let root = fresh.path().join("camdl-store");
    let cwd = workdir(&root);
    let first = predict_in(cwd.path(), &root, FIXED);
    assert!(first.status.success(), "{}", String::from_utf8_lossy(&first.stderr));
    let before = predictions(&root);
    let second = predict_in(cwd.path(), &root, FIXED);
    let err = String::from_utf8_lossy(&second.stderr);
    assert!(second.status.success(), "{err}");
    assert!(err.contains("reused the stored prediction"), "{err}");
    assert!(!err.contains("predict: mode="), "nothing is computed on a repeat:\n{err}");
    assert_eq!(predictions(&root), before, "no prediction is added or changed");
}

/// The run id of the one fixed-point prediction the line makes in a fresh
/// copy of the fixture, and its identity record.
fn fixed_point_id(line: &[&str], files: &[(&str, String)]) -> (String, serde_json::Value) {
    let fresh = tempfile::tempdir().unwrap();
    let root = fresh.path().join("camdl-store");
    let cwd = workdir(&root);
    for (name, text) in files {
        std::fs::write(cwd.path().join(name), text).unwrap();
    }
    let out = predict_in(cwd.path(), &root, line);
    assert!(out.status.success(), "{line:?}: {}", String::from_utf8_lossy(&out.stderr));
    let fixed: Vec<(PathBuf, serde_json::Value)> = predictions(&root)
        .into_iter()
        .filter(|(_, id)| id["identity"]["source"].get("FixedPoint").is_some())
        .collect();
    assert_eq!(fixed.len(), 1, "{line:?}");
    (fixed[0].1["run_id"].as_str().unwrap().to_string(), fixed[0].1.clone())
}

/// The identity follows the point, the conditioned data and the numerics,
/// and not the held-out rows: changing a row after `--holdout-after`, or
/// appending one, leaves the run id and the stored data table as they were,
/// while changing a conditioned row changes them.
#[test]
fn the_identity_follows_point_data_and_numerics_and_not_held_out_rows() {
    let base = fixed_point_id(FIXED, &[]).0;
    let with = |extra: &[&'static str]| -> Vec<&'static str> { [FIXED, extra].concat() };
    assert_ne!(fixed_point_id(&with(&["--param", "beta=0.45"]), &[]).0, base, "the point");
    assert_ne!(fixed_point_id(&with(&["--dt", "0.5"]), &[]).0, base, "the numerics: dt");
    assert_ne!(fixed_point_id(&with(&["--allow-degenerate-rates"]), &[]).0, base, "the numerics: degenerate rates");
    assert_ne!(fixed_point_id(&with(&["--binomial", "btrs"]), &[]).0, base, "the numerics: the sampler");

    let daily = std::fs::read_to_string(Path::new(FORECAST).join("daily_cases.tsv")).unwrap();
    let edit = |at: &str, to: &str| -> String {
        daily
            .lines()
            .map(|l| match l.split_once('\t') {
                Some((t, _)) if t == at => format!("{t}\t{to}\n"),
                _ => format!("{l}\n"),
            })
            .collect()
    };
    let cut = [
        "predict", "model.camdl", "--params", "point.toml", "--data", "daily_cases=edited.tsv", "--mode", "free_forward",
        "--n-draws", "4", "--holdout-after", "20",
    ];
    let (held, held_id) = fixed_point_id(&cut, &[("edited.tsv", daily.clone())]);
    assert_ne!(held, base, "--holdout-after cuts the conditioned rows");
    assert_eq!(held_id["identity"]["source"]["FixedPoint"]["trained_through"], serde_json::json!(20.0), "{held_id:#}");
    let (moved_late, _) = fixed_point_id(&cut, &[("edited.tsv", edit("30", "999"))]);
    assert_eq!(moved_late, held, "a held-out row's value is not in the identity");
    let (appended, _) = fixed_point_id(&cut, &[("edited.tsv", daily.clone() + "64\t1\n")]);
    assert_eq!(appended, held, "an appended held-out row is not in the identity");
    let (moved_early, _) = fixed_point_id(&cut, &[("edited.tsv", edit("10", "999"))]);
    assert_ne!(moved_early, held, "a conditioned row's value is");
}

/// A held-out row is not stored: the data table a prediction with
/// `--holdout-after` depends on holds the conditioned lines only.
#[test]
fn a_held_out_row_is_not_stored() {
    let fresh = tempfile::tempdir().unwrap();
    let root = fresh.path().join("camdl-store");
    let cwd = workdir(&root);
    let out = predict_in(cwd.path(), &root, &[FIXED, &["--holdout-after", "20"]].concat());
    assert!(out.status.success(), "{}", String::from_utf8_lossy(&out.stderr));
    let tables: Vec<String> = folders(&root)
        .into_iter()
        .filter(|(_, id, _)| id["kind"] == crate::camdl_store::DataTable::KIND)
        .map(|(p, _, _)| std::fs::read_to_string(p.join(<crate::camdl_store::DataTable as store::ContentArtifact>::FILE)).unwrap())
        .collect();
    assert_eq!(tables.len(), 1, "{tables:?}");
    let times: Vec<f64> = tsv_rows(&tables[0]).iter().map(|r| r["time"].parse().unwrap()).collect();
    assert_eq!(times.last(), Some(&20.0), "the stored table ends at the cutoff: {times:?}");
}

/// A fixed point given `--holdout-after` labels the rows it conditions on
/// `trained`, as declared; without it the same rows are `unknown`.
#[test]
fn a_declared_holdout_labels_the_conditioned_rows_trained() {
    let fresh = tempfile::tempdir().unwrap();
    let root = fresh.path().join("camdl-store");
    let cwd = workdir(&root);
    let out = predict_in(cwd.path(), &root, &[FIXED, &["--holdout-after", "20"]].concat());
    assert!(out.status.success(), "{}", String::from_utf8_lossy(&out.stderr));
    let found = predictions(&root);
    assert_eq!(found.len(), 1, "{found:#?}");
    let rows = tsv_rows(&std::fs::read_to_string(found[0].0.join("predictive/daily_cases.tsv")).unwrap());
    let role_at = |t: f64| -> Vec<&str> {
        rows.iter().filter(|r| r["time"].parse::<f64>().unwrap() == t).map(|r| r["role"].as_str()).collect()
    };
    assert_eq!(role_at(10.0), ["trained"], "a conditioned row: {rows:?}");
    assert!(rows.iter().all(|r| r["role"] != "unknown"), "a declared holdout leaves no row unknown: {rows:?}");
}

/// A free-forward prediction at a fixed point replays what `simulate`
/// simulates at that point under the same seed. The point is the posterior
/// mean of `@fit`'s draws; it is repeated once per draw, and each copy runs
/// under the process seed `simulate --draws` gives the draw at its place. So
/// the band of the latent quantity `peak_I = max(I)` over the four copies
/// has, quantile for quantile and digit for digit, the band of the four
/// simulated trajectories' maxima of `I`, and the copies differ.
#[test]
fn a_fixed_point_free_forward_band_replays_what_simulate_simulates() {
    let fresh = tempfile::tempdir().unwrap();
    let root = fresh.path().join("camdl-store");
    let cwd = workdir(&root);
    let run = |line: &[&str]| camdl(cwd.path(), &root, &line.iter().map(|s| s.to_string()).collect::<Vec<_>>(), &[]);
    // `@fit`'s run folder, as `camdl show` names it, and its posterior draws.
    let show = run(&["show", "@fit"]);
    let text = String::from_utf8_lossy(&show.stdout).into_owned();
    assert!(show.status.success(), "{}", String::from_utf8_lossy(&show.stderr));
    let folder = text.lines().skip_while(|l| l.trim() != "path").nth(1).expect("a path").trim().to_string();
    let folder = cwd.path().join(folder);
    let mut stack = vec![folder.clone()];
    let mut found = None;
    while let Some(d) = stack.pop() {
        for e in std::fs::read_dir(&d).unwrap().flatten() {
            if e.path().is_dir() {
                stack.push(e.path());
            } else if e.file_name() == "draws.tsv" {
                found = Some(e.path());
            }
        }
    }
    let draws = tsv_rows(&std::fs::read_to_string(found.expect("@fit's draws.tsv")).unwrap());
    let names = ["beta", "gamma", "rho", "k", "N0", "I0"];
    let mean = |n: &str| draws.iter().map(|r| r[n].parse::<f64>().unwrap()).sum::<f64>() / draws.len() as f64;
    let point: Vec<(&str, f64)> = names.iter().map(|n| (*n, mean(n))).collect();
    let toml_text: String = point.iter().map(|(n, v)| format!("{n} = {v:?}\n")).collect();
    std::fs::write(cwd.path().join("mean.toml"), &toml_text).unwrap();
    let mut rows = names.join("\t") + "\n";
    for _ in 0..4 {
        rows += &(point.iter().map(|(_, v)| format!("{v:?}")).collect::<Vec<_>>().join("\t") + "\n");
    }
    std::fs::write(cwd.path().join("draws4.tsv"), rows).unwrap();

    let out = predict_in(cwd.path(), &root, &[
        "predict", "model.camdl", "--params", "mean.toml", "--data", "daily_cases=daily_cases.tsv", "--mode", "free_forward",
        "--n-draws", "4", "--seed", "7",
    ]);
    assert!(out.status.success(), "{}", String::from_utf8_lossy(&out.stderr));
    let (folder, _) = predictions(&root)
        .into_iter()
        .find(|(_, id)| id["identity"]["source"].get("FixedPoint").is_some())
        .expect("the fixed-point prediction");
    let peak = tsv_rows(&std::fs::read_to_string(folder.join("quantities-model/peak_I.tsv")).unwrap());
    assert_eq!(peak.len(), 1, "{peak:?}");
    assert_eq!(peak[0]["n_draws"], "4", "{peak:?}");
    for c in ["rhat", "ess"] {
        assert!(peak[0].get(c).is_none_or(|v| v.is_empty()), "{c} is empty for a fixed point: {peak:?}");
    }

    let sim = run(&["simulate", "model.camdl", "--draws", "draws4.tsv", "--seed", "7", "-o", "traj.tsv", "--progress", "none"]);
    assert!(sim.status.success(), "{}", String::from_utf8_lossy(&sim.stderr));
    let traj = tsv_rows(&std::fs::read_to_string(cwd.path().join("traj.tsv")).unwrap());
    let mut peaks: BTreeMap<String, f64> = BTreeMap::new();
    for r in &traj {
        let i: f64 = r.get("I").unwrap_or_else(|| panic!("traj.tsv columns: {:?}", r.keys())).parse().unwrap();
        let e = peaks.entry(r["replicate"].clone()).or_insert(f64::NEG_INFINITY);
        *e = e.max(i);
    }
    assert_eq!(peaks.len(), 4, "four simulated copies: {:?}", peaks.keys());
    let simulated: Vec<f64> = peaks.values().copied().collect();
    assert!(simulated.iter().any(|p| *p != simulated[0]), "the copies differ in their process noise: {simulated:?}");
    let qs = crate::quantile::band(&simulated).unwrap();
    for ((_, label), q) in crate::quantile::QUANTILE_LEVELS.iter().zip(&qs) {
        assert_eq!(
            peak[0][*label],
            crate::quantile::fmt_value(*q),
            "the band's {label} of peak_I against simulate's four trajectories at the point ({simulated:?})"
        );
    }
}

/// Each refusal of a fixed-point prediction, and of the flags of one source
/// given with the other, before anything is stored.
#[test]
fn the_fixed_point_refusals() {
    let cases: &[(&str, Vec<&str>, &[&str])] = &[
        ("a posterior flag with a model file", [FIXED, &["--by-chain"]].concat(), &["--by-chain", "model file"]),
        ("--exclude-chains with a model file", [FIXED, &["--exclude-chains", "1"]].concat(), &["--exclude-chains"]),
        ("a fixed-point flag with a handle", vec!["predict", "@fit", "--params", "point.toml", "--mode", "free_forward"], &["--params", "stored fit run"]),
        ("no --data", vec!["predict", "model.camdl", "--params", "point.toml"], &["--data"]),
        ("one-step on ODE", [FIXED, &["--backend", "ode"]].concat().into_iter().map(|s| if s == "free_forward" { "one_step" } else { s }).collect(), &["ODE", "free_forward"]),
        ("one-step on Gillespie", [FIXED, &["--backend", "gillespie"]].concat().into_iter().map(|s| if s == "free_forward" { "one_step" } else { s }).collect(), &["Gillespie"]),
        ("a forecast on Gillespie", vec!["predict", "model.camdl", "--params", "point.toml", "--data", "daily_cases=daily_cases.tsv", "--horizon", "2 weeks", "--backend", "gillespie"], &["Gillespie", "forecast"]),
        ("--particles without a filter", [FIXED, &["--particles", "20"]].concat(), &["--particles"]),
        ("--design without a forecast", [FIXED, &["--design", "survey=design.tsv"]].concat(), &["--design"]),
        ("--binomial off chain-binomial", [FIXED, &["--backend", "ode", "--binomial", "btrs"]].concat(), &["--binomial"]),
        (
            "--integrator off ode",
            [FIXED, &["--integrator", "rk45"]].concat(),
            &["--integrator selects the ODE integrator, but this prediction runs on chain_binomial, which has none.\n  \
               Fix: drop --integrator, or pass --backend ode."],
        ),
        ("a parameter with no value", vec!["predict", "model.camdl", "--data", "daily_cases=daily_cases.tsv"], &["no value"]),
        ("an undeclared --param", [FIXED, &["--param", "nosuch=1"]].concat(), &["nosuch"]),
        ("--chains with a model file", [FIXED, &["--chains", "0"]].concat(), &["--chains", "model file"]),
        (
            "--realised-design with a model file",
            vec![
                "predict", "model.camdl", "--params", "point.toml", "--data", "survey=survey.tsv", "--horizon", "2 weeks",
                "--realised-design",
            ],
            &["--realised-design", "model file"],
        ),
    ];
    for (what, line, says) in cases {
        let fresh = tempfile::tempdir().unwrap();
        let root = fresh.path().join("camdl-store");
        let cwd = workdir(&root);
        let before: Vec<String> = files_under(&root).into_keys().collect();
        let out = predict_in(cwd.path(), &root, line);
        let err = String::from_utf8_lossy(&out.stderr);
        assert!(!out.status.success(), "{what}: ran\n{err}");
        for s in *says {
            assert!(err.contains(s), "{what}: the refusal names `{s}`:\n{err}");
        }
        let after: Vec<String> = files_under(&root).into_keys().collect();
        assert_eq!(after, before, "{what}: a refused command writes nothing to the store");
    }
}

/// A window-column stream with no emit schedule has nothing to forecast at
/// from a fixed point, which keeps no held-out rows: it is refused with the
/// reason a fit with no held-out rows gives.
#[test]
fn a_windowed_stream_with_no_rhythm_is_refused_for_a_forecast() {
    let fresh = tempfile::tempdir().unwrap();
    let root = fresh.path().join("camdl-store");
    let cwd = workdir(&root);
    let model = std::fs::read_to_string(cwd.path().join("model.camdl")).unwrap();
    let windowed = model.replace(
        "  daily_cases {\n    columns       { time : time, daily_cases : count }\n    covers        = closing_at(time, 1 'days)\n    projected     = incidence(infection)\n    emit_schedule = every 1 'days\n",
        "  daily_cases {\n    columns       { start : window_start, stop : window_stop, daily_cases : count }\n    projected     = incidence(infection)\n",
    );
    assert_ne!(windowed, model, "the fixture's daily stream moved");
    std::fs::write(cwd.path().join("windowed.camdl"), windowed).unwrap();
    let rows = (0..24).fold("start\tstop\tdaily_cases\n".to_string(), |s, t| s + &format!("{t}\t{}\t{}\n", t + 1, 2 + t % 4));
    std::fs::write(cwd.path().join("windows.tsv"), rows).unwrap();
    let out = predict_in(cwd.path(), &root, &[
        "predict", "windowed.camdl", "--params", "point.toml", "--data", "daily_cases=windows.tsv", "--horizon", "1 weeks",
        "--n-draws", "2", "--particles", "10",
    ]);
    let err = String::from_utf8_lossy(&out.stderr);
    assert!(!out.status.success(), "the windowed stream was forecast:\n{err}");
    assert!(err.contains("daily_cases") && err.contains("which period comes next"), "{err}");
}

// ── The store and export flags, for both sources ─────────────────────────────

/// A posterior prediction and a fixed-point one, each as a command line.
fn both_sources() -> [(&'static str, Vec<&'static str>); 2] {
    [("posterior", vec!["predict", "@fit", "--mode", "free_forward", "--n-draws", "4"]), ("fixed point", FIXED.to_vec())]
}

/// Every file under `dir`, by its path there, with its bytes.
fn files_under(dir: &Path) -> BTreeMap<String, Vec<u8>> {
    let mut out = BTreeMap::new();
    let mut stack = vec![dir.to_path_buf()];
    while let Some(d) = stack.pop() {
        for e in std::fs::read_dir(&d).into_iter().flatten().flatten() {
            let p = e.path();
            if p.is_dir() {
                stack.push(p);
            } else {
                out.insert(p.strip_prefix(dir).unwrap().to_string_lossy().into_owned(), std::fs::read(&p).unwrap());
            }
        }
    }
    out
}

/// `--store DIR` names the store over `CAMDL_STORE`: a run handle is looked
/// up there and the prediction stored inside the run, and a fixed point's
/// prediction is stored there; the store `CAMDL_STORE` names is not made.
#[test]
fn store_names_the_store_for_both_sources() {
    for (what, line) in both_sources() {
        let fresh = tempfile::tempdir().unwrap();
        let root = fresh.path().join("camdl-store");
        let cwd = workdir(&root);
        let elsewhere = fresh.path().join("elsewhere");
        let root_text = root.to_string_lossy().into_owned();
        let out = predict_in(cwd.path(), &elsewhere, &[line.as_slice(), &["--store", &root_text]].concat());
        assert!(out.status.success(), "{what}: {}", String::from_utf8_lossy(&out.stderr));
        assert_eq!(predictions(&root).len(), 1, "{what}: the prediction is in the --store store");
        assert!(!elsewhere.exists(), "{what}: the CAMDL_STORE store is not made");
    }
}

/// `--label NAME` names the invocation: `camdl show @NAME` then shows the
/// prediction it made.
#[test]
fn label_names_the_invocation_for_both_sources() {
    for (what, line) in both_sources() {
        let fresh = tempfile::tempdir().unwrap();
        let root = fresh.path().join("camdl-store");
        let cwd = workdir(&root);
        let out = predict_in(cwd.path(), &root, &[line.as_slice(), &["--label", "mine"]].concat());
        assert!(out.status.success(), "{what}: {}", String::from_utf8_lossy(&out.stderr));
        let show = camdl(cwd.path(), &root, &["show".to_string(), "@mine".to_string()], &[]);
        let text = String::from_utf8_lossy(&show.stdout);
        assert!(show.status.success(), "{what}: {}", String::from_utf8_lossy(&show.stderr));
        let (folder, _) = &predictions(&root)[0];
        let name = folder.file_name().unwrap().to_string_lossy().into_owned();
        assert!(text.contains(&name), "{what}: @mine shows the prediction {name}:\n{text}");
    }
}

/// `--force` computes a stored prediction again, moving the stored one to
/// the store's quarantine; without it a repeat is reused.
#[test]
fn force_computes_again_for_both_sources() {
    for (what, line) in both_sources() {
        let fresh = tempfile::tempdir().unwrap();
        let root = fresh.path().join("camdl-store");
        let cwd = workdir(&root);
        let first = predict_in(cwd.path(), &root, &line);
        assert!(first.status.success(), "{what}: {}", String::from_utf8_lossy(&first.stderr));
        let forced = predict_in(cwd.path(), &root, &[line.as_slice(), &["--force"]].concat());
        let err = String::from_utf8_lossy(&forced.stderr);
        assert!(forced.status.success(), "{what}: {err}");
        assert!(!err.contains("reused the stored prediction") && err.contains("predict: mode="), "{what}: computed again:\n{err}");
        let quarantined = files_under(&root.join(".quarantine"));
        assert!(quarantined.keys().any(|k| k.ends_with("predictive.json")), "{what}: the stored prediction is quarantined: {:?}", quarantined.keys());
        assert_eq!(predictions(&root).len(), 1, "{what}: one prediction stays stored");
    }
}

/// `-o DIR` exports a copy of the prediction's result files, byte for byte,
/// both when it is computed and when it is reused.
#[test]
fn output_exports_the_result_files_for_both_sources() {
    for (what, line) in both_sources() {
        let fresh = tempfile::tempdir().unwrap();
        let root = fresh.path().join("camdl-store");
        let cwd = workdir(&root);
        for dir in ["computed", "reused"] {
            let out = predict_in(cwd.path(), &root, &[line.as_slice(), &["-o", dir]].concat());
            assert!(out.status.success(), "{what} ({dir}): {}", String::from_utf8_lossy(&out.stderr));
            let (folder, _) = &predictions(&root)[0];
            let mut stored = files_under(folder);
            stored.retain(|k, _| k != "identity.json" && k != "run.json");
            let exported = files_under(&cwd.path().join(dir));
            assert!(!exported.is_empty(), "{what} ({dir}): nothing exported");
            assert_eq!(exported, stored, "{what} ({dir}): the export is the stored result files");
        }
    }
}

// ── Review fixes ─────────────────────────────────────────────────────────────

/// A stored prediction whose report records a deterministic failure fails
/// a rerun that reuses it as it failed the run that computed it: exit 1,
/// with the failure named, for both sources.
#[test]
fn a_reused_prediction_with_failures_exits_1_for_both_sources() {
    let fresh = tempfile::tempdir().unwrap();
    let root = fresh.path().join("camdl-store");
    let cwd = workdir(&root);
    // A posterior fit of the daily cases alone, so a stream the fit did not
    // bind can be asked for, as the fixed point's data binds the daily cases
    // alone.
    let daily_only = FIT
        .replace("weekly_cases = \"weekly_cases.tsv\"\n", "")
        .replace("survey = \"survey.tsv\"\n", "");
    std::fs::write(cwd.path().join("fit_daily.toml"), daily_only).unwrap();
    let fitted = camdl(
        cwd.path(),
        &root,
        &["fit", "run", "fit_daily.toml", "--label", "daily", "--progress", "none"].map(String::from),
        &[],
    );
    assert!(fitted.status.success(), "{}", String::from_utf8_lossy(&fitted.stderr));
    let posterior: Vec<&str> = vec!["predict", "@daily", "--mode", "free_forward", "--n-draws", "4"];
    for (what, line) in [("posterior", posterior), ("fixed point", FIXED.to_vec())] {
        // `--stream weekly_cases` asks for a stream with no bound data: a
        // deterministic failure, recorded, with the daily cases still written.
        let line = [line.as_slice(), &["--stream", "daily_cases", "--stream", "weekly_cases"]].concat();
        let first = predict_in(cwd.path(), &root, &line);
        let err = String::from_utf8_lossy(&first.stderr);
        assert_eq!(first.status.code(), Some(1), "{what}: the computing run fails:\n{err}");
        assert!(err.contains("weekly_cases"), "{what}: {err}");
        let second = predict_in(cwd.path(), &root, &line);
        let err = String::from_utf8_lossy(&second.stderr);
        assert!(err.contains("reused the stored prediction"), "{what}: {err}");
        assert_eq!(second.status.code(), Some(1), "{what}: the reusing run fails too:\n{err}");
        assert!(err.contains("✗") && err.contains("weekly_cases"), "{what}: the failure is named:\n{err}");
    }
}

/// A posterior prediction given no `--model` compiles nothing, so
/// `--no-licm` changes nothing: with and without it, one prediction with one
/// set of bytes.
#[test]
fn no_licm_changes_no_posterior_prediction() {
    let (off, on) = crate::input_fixtures::predict_no_licm();
    let h = Harness { base: off.clone() };
    let a = h.run(&off, None).unwrap();
    let b = h.run(&on, None).unwrap();
    let ids = |o: &Observed| o.results.iter().map(|r| (r.kind.clone(), r.run_id, r.files.clone())).collect::<Vec<_>>();
    assert!(!a.results.is_empty());
    assert_eq!(ids(&a), ids(&b), "--no-licm moved a posterior prediction");
}

#[path = "score_tests.rs"]
mod score;
#[path = "compare_tests.rs"]
mod compare;
