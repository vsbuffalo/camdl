//! `camdl fit run` on the run store, checked by running the release binary:
//! every declared input against its class (`store::check_input_classes`), the
//! directory tree a run makes, and the files each method computes against the
//! bytes the old store held.
//!
//! The binary is `rust/target/release/camdl` and the compiler
//! `ocaml/_build/default/bin/camdlc.exe`, both built by `make build` from the
//! source this test was built from. Every run's working directory is a fresh
//! copy of the fixture files, so the relative paths `crate::input_fixtures`
//! returns name them, and its environment is cleared but for an allowlist.

// The harness's `run` returns the store's own error, as `InputHarness::run`
// requires.
#![allow(clippy::result_large_err)]

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};
use std::sync::OnceLock;

use store::{Artifact, Class, ClassFailure, FieldInfo, InputHarness, Inputs, Key, Observed, ObservedResult};

use crate::args::FitRunInvocation;
use crate::test_support::Pins;
use crate::input_fixtures::{fit_base, fit_from};

const SIR: &str = r#"
time_unit = 'days

compartments { S, I, R }

parameters {
  beta  : rate        in [0.05, 2.0]  ~ log_normal(mu = -1.0, sigma = 0.3)
  gamma : rate        in [0.01, 1.0]  ~ log_normal(mu = -1.5, sigma = 0.3)
  rho   : probability in [0.05, 0.95]
  k     : probability in [0.01, 0.99]
}

let N = S + I + R

transitions {
  infection : S --> I @ beta * exp(-k) * S * I / N
  recovery  : I --> R @ gamma * I
}

init { S = 990  I = 10  R = 0 }

observations {
  cases {
    columns       { time : time, cases : count }
    covers        = closing_at(time, 1 'days)
    projected     = incidence(infection)
    emit_schedule = every 1 'days
    cases         ~ poisson(rate = rho * projected)
  }
}

simulate { from = 0 'days  to = 20 'days }
"#;

/// The `[method]` tables the fixtures run, by name: each a few iterations of
/// a few particles.
const METHODS: &[(&str, &str)] = &[
    ("if2", "algorithm = \"if2\"\nbackend = \"chain_binomial\"\nchains = 2\nparticles = 30\niterations = 3\ncooling = 0.9\n"),
    ("pgas", "algorithm = \"pgas\"\nbackend = \"chain_binomial\"\nchains = 2\nparticles = 20\nsweeps = 6\nburn_in = 2\n"),
    ("pmmh", "algorithm = \"pmmh\"\nbackend = \"chain_binomial\"\nchains = 2\nparticles = 20\niterations = 20\nburn_in = 5\n"),
    ("pmmh_long", "algorithm = \"pmmh\"\nbackend = \"chain_binomial\"\nchains = 2\nparticles = 20\niterations = 40\nburn_in = 5\n"),
    ("pfilter", "algorithm = \"pfilter\"\nbackend = \"chain_binomial\"\nparticles = 30\nreplicates = 2\n"),
    // `record_prequential` defaults on; off here so `--record-prequential`
    // turns it on.
    ("pfilter_quiet", "algorithm = \"pfilter\"\nbackend = \"chain_binomial\"\nparticles = 30\nreplicates = 1\nrecord_prequential = false\n"),
    ("mh", "algorithm = \"mh\"\nbackend = \"ode\"\nchains = 2\niterations = 60\nburn_in = 20\n"),
    ("sbplx", "algorithm = \"nl-sbplx\"\nbackend = \"ode\"\nchains = 2\n"),
    ("bobyqa", "algorithm = \"nl-bobyqa\"\nbackend = \"ode\"\nchains = 2\n"),
    ("lbfgs", "algorithm = \"nl-lbfgs\"\nbackend = \"ode\"\nchains = 2\n"),
];

/// A fit of `model` to `cases.tsv` with `rho` fixed at `rho`.
fn problem(model: &str, rho: f64) -> String {
    format!(
        "[model]\ncamdl = \"{model}\"\n[data.observations]\ncases = \"cases.tsv\"\n\
         [estimate]\nbeta = {{ bounds = [0.05, 2.0], start = 0.4 }}\ngamma = {{ bounds = [0.01, 1.0], start = 0.2 }}\n\
         [fixed]\nrho = {rho}\nk = 0.3\n[method]\n"
    )
}

fn method(name: &str) -> &'static str {
    METHODS.iter().find(|(n, _)| *n == name).unwrap().1
}

fn fixture_files() -> Vec<(String, String)> {
    let mut v: Vec<(String, String)> = vec![
        ("sir.camdl".into(), SIR.into()),
        // An instant stream beside the counts, which `--emit-every` can thin.
        (
            "sir_prevalent.camdl".into(),
            SIR.replace(
                "    cases         ~ poisson(rate = rho * projected)\n  }\n",
                "    cases         ~ poisson(rate = rho * projected)\n  }\n  prevalent {\n    columns       { time : time, prevalent : count }\n    \
                 projected     = I\n    emit_schedule = every 1 'days\n    prevalent     ~ poisson(rate = projected)\n  }\n",
            ),
        ),
        ("cases.tsv".into(), (1..=20).fold("time\tcases\n".to_string(), |s, t| s + &format!("{t}\t{}\n", 3 + t))),
        ("truth.toml".into(), "beta = 0.4\ngamma = 0.2\nrho = 0.5\nk = 0.3\n".into()),
        ("fit.toml".into(), problem("sir.ir.json", 0.5) + method("if2")),
        ("fit_other.toml".into(), problem("sir.ir.json", 0.6) + method("if2")),
        ("fit_source.toml".into(), problem("sir.camdl", 0.5) + method("if2")),
        (
            "fit_synthetic.toml".into(),
            format!(
                "[model]\ncamdl = \"sir_prevalent.ir.json\"\n[synthetic]\ntrue_params = \"truth.toml\"\nsim_seeds = [1]\n\
                 [estimate]\nbeta = {{ bounds = [0.05, 2.0], start = 0.4 }}\n[fixed]\ngamma = 0.2\nrho = 0.5\nk = 0.3\n\
                 [method]\n{}",
                method("if2")
            ),
        ),
    ];
    for (name, m) in METHODS {
        v.push((format!("fit_{name}.toml"), problem("sir.ir.json", 0.5) + m));
    }
    v
}

/// The binaries under test, and the fixture directory, made once.
struct Fixture {
    camdl: PathBuf,
    camdlc: PathBuf,
    dir: PathBuf,
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
        let dir = crate::test_support::unique_temp_dir("fit_command_fixture");
        std::fs::create_dir_all(&dir).unwrap();
        for (name, text) in fixture_files() {
            std::fs::write(dir.join(name), text).unwrap();
        }
        for stem in ["sir", "sir_prevalent"] {
            let out = Command::new(&camdlc).env_clear().arg(dir.join(format!("{stem}.camdl"))).output().unwrap();
            assert!(out.status.success(), "camdlc {stem}.camdl: {}", String::from_utf8_lossy(&out.stderr));
            std::fs::write(dir.join(format!("{stem}.ir.json")), out.stdout).unwrap();
        }
        let fx = Fixture { camdl, camdlc, dir };
        // `draws.tsv`: the posterior of a stored PMMH run, a `from_posterior`
        // source named as a file, which carries no verdict.
        let store = tempfile::tempdir().unwrap();
        let out = camdl_with(&fx, &fx.dir, store.path(), &["fit".into(), "run".into(), "fit_pmmh.toml".into()], &[]);
        assert!(out.status.success(), "the fixture's PMMH fit: {}", String::from_utf8_lossy(&out.stderr));
        let draws = run_folders(store.path()).into_iter().next().expect("the fixture's PMMH run").join("draws.tsv");
        std::fs::copy(draws, fx.dir.join("draws.tsv")).unwrap();
        fx
    })
}

/// A fresh working directory holding a copy of the fixture files.
fn workdir() -> tempfile::TempDir {
    let d = tempfile::tempdir().unwrap();
    for e in std::fs::read_dir(&fixture().dir).unwrap().flatten() {
        if e.path().is_file() {
            std::fs::copy(e.path(), d.path().join(e.file_name())).unwrap();
        }
    }
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
fn argv(inv: &FitRunInvocation) -> Vec<String> {
    let FitRunInvocation { args, no_licm, no_ir_cache, verbosity: _, progress: _, no_progress: _ } = inv;
    let crate::args::FitRunArgs {
        config,
        problem: crate::args::FitProblemFlags {
            forcing_extrapolation: crate::args::ForcingExtrapolationArg { allow_forcing_extrapolation },
            emit_every,
            dt,
            sweep,
        },
        run: crate::args::FitRunFlags { seed, starts, starts_chains, resume },
        method:
            crate::args::FitMethodFlags {
                decibans_thresh,
                no_dt_check,
                dt_check_strict,
                binomial,
                dt_check_halvings,
                cooling_target_iters,
                max_tree_depth,
                trajectory_warmup,
                csmc_sweeps_per_nuts,
                n_trajectories,
                diagonal_mass,
                no_nuts,
                no_ancestor_sampling,
                no_adapt,
                adapt_start,
                record_ancestry,
                record_prequential,
            },
        allow_nonconverged_source,
        parallel,
        force,
        label,
        store,
        manifest,
        link_root,
        ..
    } = args;
    let mut v: Vec<String> = vec!["fit".into(), "run".into(), config.to_string_lossy().into_owned()];
    let mut opt = |flag: &str, value: Option<String>| {
        if let Some(x) = value {
            v.push(flag.into());
            v.push(x);
        }
    };
    for e in emit_every {
        opt("--emit-every", Some(e.clone()));
    }
    for s in sweep {
        let values: Vec<String> = s.grid.expand().iter().map(f64::to_string).collect();
        opt("--sweep", Some(format!("{}={}", s.name, values.join(","))));
    }
    opt("--dt", dt.map(|x| x.to_string()));
    opt("--seed", seed.map(|x| x.to_string()));
    opt(
        "--starts",
        starts.as_ref().map(|s| match serde_json::to_value(s).unwrap() {
            serde_json::Value::String(tag) => tag,
            serde_json::Value::Object(m) => {
                let (tag, src) = m.into_iter().next().unwrap();
                format!("{tag}={}", src.as_str().unwrap())
            }
            other => panic!("a starts rule serializes as {other}"),
        }),
    );
    opt("--starts-chains", starts_chains.as_ref().map(|c| serde_json::to_value(c).unwrap()["ids"].as_str().unwrap().to_string()));
    opt("--resume", resume.clone());
    opt("--decibans-thresh", decibans_thresh.map(|x| x.to_string()));
    opt("--binomial", binomial.map(|b| serde_json::to_value(b).unwrap().as_str().unwrap().to_string()));
    opt("--dt-check-halvings", dt_check_halvings.map(|x| x.to_string()));
    opt("--cooling-target-iters", cooling_target_iters.map(|x| x.to_string()));
    opt("--max-tree-depth", max_tree_depth.map(|x| x.to_string()));
    opt("--trajectory-warmup", trajectory_warmup.map(|x| x.to_string()));
    opt("--csmc-sweeps-per-nuts", csmc_sweeps_per_nuts.map(|x| x.to_string()));
    opt("--n-trajectories", n_trajectories.map(|x| x.to_string()));
    opt("--adapt-start", adapt_start.map(|x| x.to_string()));
    opt("--parallel", Some(parallel.to_string()));
    opt("--label", label.clone());
    opt("--store", store.as_ref().map(|p| p.to_string_lossy().into_owned()));
    opt("--manifest", manifest.as_ref().map(|p| p.to_string_lossy().into_owned()));
    opt("--link-root", link_root.as_ref().map(|p| p.to_string_lossy().into_owned()));
    let mut flag = |set: bool, name: &str| {
        if set {
            v.push(name.into());
        }
    };
    flag(*allow_forcing_extrapolation, "--allow-forcing-extrapolation");
    flag(*no_dt_check, "--no-dt-check");
    flag(*dt_check_strict, "--dt-check-strict");
    flag(*diagonal_mass, "--diagonal-mass");
    flag(*no_nuts, "--no-nuts");
    flag(*no_ancestor_sampling, "--no-ancestor-sampling");
    flag(*no_adapt, "--no-adapt");
    flag(*record_ancestry, "--record-ancestry");
    flag(*record_prequential, "--record-prequential");
    flag(*allow_nonconverged_source, "--allow-nonconverged-source");
    flag(*force, "--force");
    flag(*no_licm, "--no-licm");
    flag(*no_ir_cache, "--no-ir-cache");
    v.extend(["--progress".to_string(), "none".to_string()]);
    v
}

/// Every result folder under `root`: its kind, run id, result files with
/// their digests, and upstream.
fn observe(root: &Path) -> Vec<ObservedResult> {
    let mut out = Vec::new();
    for (_, id, run) in folders(root) {
        let files = run["files"]
            .as_object()
            .unwrap()
            .iter()
            .filter(|(_, f)| f["role"] == "result")
            .map(|(n, f)| (n.clone(), Key::from_hex(f["sha256"].as_str().unwrap()).unwrap()))
            .collect();
        let upstream = match &id["parent"] {
            serde_json::Value::Object(o) if o["relation"] == "upstream" => Some(Key::from_hex(o["run_id"].as_str().unwrap()).unwrap()),
            _ => None,
        };
        let kind = id["kind"].as_str().unwrap().to_string();
        out.push(ObservedResult {
            content: [crate::camdl_store::CompiledModel::KIND, crate::camdl_store::DataTable::KIND].contains(&kind.as_str()),
            kind,
            run_id: Key::from_hex(id["run_id"].as_str().unwrap()).unwrap(),
            files,
            upstream,
        });
    }
    out
}

/// Every result folder under `root` with its `identity.json` and `run.json`,
/// in path order.
fn folders(root: &Path) -> Vec<(PathBuf, serde_json::Value, serde_json::Value)> {
    complete_folders(root).unwrap_or_else(|e| panic!("{e}"))
}

/// [`folders`], or the first record that cannot be read. A claim writes
/// `identity.json` and then `run.json`, so a folder a running process is
/// claiming can hold the first without the second.
fn complete_folders(root: &Path) -> Result<Vec<(PathBuf, serde_json::Value, serde_json::Value)>, String> {
    fn walk(dir: &Path, out: &mut Vec<(PathBuf, serde_json::Value, serde_json::Value)>) -> Result<(), String> {
        let Ok(entries) = std::fs::read_dir(dir) else { return Ok(()) };
        let mut entries: Vec<_> = entries.flatten().collect();
        entries.sort_by_key(|e| e.file_name());
        for e in entries {
            let p = e.path();
            if !p.is_dir() || e.file_name().to_string_lossy().starts_with('.') {
                continue;
            }
            if p.join("identity.json").is_file() {
                let read = |f: &str| -> Result<serde_json::Value, String> {
                    let bytes = std::fs::read(p.join(f)).map_err(|e| format!("{}: {e}", p.join(f).display()))?;
                    serde_json::from_slice(&bytes).map_err(|e| format!("{}: {e}", p.join(f).display()))
                };
                out.push((p.clone(), read("identity.json")?, read("run.json")?));
            }
            walk(&p, out)?;
        }
        Ok(())
    }
    let mut out = Vec::new();
    walk(root, &mut out)?;
    Ok(out)
}

/// The folders of every fit run under `root`.
fn run_folders(root: &Path) -> Vec<PathBuf> {
    folders(root).into_iter().filter(|(_, id, _)| id["kind"] == "fit_run").map(|(p, _, _)| p).collect()
}

const USE_LOG: &str = "camdl::fit_command=debug";

/// What a run's execution inputs did, each under `used.<field>`: from its
/// debug log (`USE_LOG`), the threads the run's pool holds and whether a
/// starts source's verdict was lifted; and from the IR cache directory,
/// whether the compile bypassed the cache (`--no-ir-cache`). An effect the
/// log reports twice with different values is recorded as both, so it
/// matches no single value.
fn used(stderr: &str, ir_cache: &Path) -> BTreeMap<String, String> {
    let mut seen: BTreeMap<String, std::collections::BTreeSet<String>> = BTreeMap::new();
    let mut note = |field: &str, value: &str| {
        seen.entry(format!("used.{field}")).or_default().insert(value.to_string());
    };
    for line in stderr.lines() {
        let after = |marker: &str| line.split_once(marker).and_then(|(_, rest)| rest.split_whitespace().next());
        if let Some(n) = after("fit run: running on ") {
            note("args.parallel", n);
        }
        if let Some(b) = after("fit run: a starts source's verdict is lifted: ") {
            note("args.allow_nonconverged_source", b);
        }
    }
    note("no_ir_cache", if crate::test_support::ir_cache_filled(ir_cache) { "false" } else { "true" });
    seen.into_iter().map(|(k, v)| (k, v.into_iter().collect::<Vec<_>>().join(","))).collect()
}

/// The harness of `fit run`'s class checks, from one base invocation.
struct Harness {
    base: FitRunInvocation,
    /// Store `fit_pmmh.toml` as `@base` before each run, for `--resume`.
    stores_base: bool,
}

impl Harness {
    /// The base each field is checked from: a fixture on which the field is
    /// live. A method's own flag is checked on a fit of that method.
    fn for_field(name: &str) -> Harness {
        let field = name.rsplit('.').next().unwrap();
        let mut stores_base = false;
        let base = match field {
            "max_tree_depth" | "trajectory_warmup" | "csmc_sweeps_per_nuts" | "n_trajectories" | "diagonal_mass"
            | "no_nuts" | "no_ancestor_sampling" => fit_from(&["fit_pgas.toml"]),
            "no_adapt" | "adapt_start" => fit_from(&["fit_pmmh.toml"]),
            "record_ancestry" | "record_prequential" => fit_from(&["fit_pfilter_quiet.toml"]),
            "emit_every" => fit_from(&["fit_synthetic.toml"]),
            // The subset is judged by its own R̂, which a fixture this short
            // need not pass: the verdict is lifted on both sides.
            "starts_chains" => fit_from(&["fit.toml", "--starts", "from_posterior=draws.tsv", "--allow-nonconverged-source"]),
            "resume" => {
                stores_base = true;
                fit_from(&["fit_pmmh_long.toml"])
            }
            "sweep" => fit_from(&["fit.toml", "--sweep", "rho=0.5"]),
            "no_licm" => fit_from(&["fit_source.toml"]),
            _ => fit_base(),
        };
        Harness { base, stores_base }
    }
}

impl InputHarness for Harness {
    type Invocation = FitRunInvocation;

    fn base(&self) -> FitRunInvocation {
        self.base.clone()
    }

    fn varied(&self, field: &FieldInfo) -> FitRunInvocation {
        let mut inv = self.base.clone();
        let m = &mut inv.args.method;
        match (field.name, field.class) {
            ("args.problem.sweep", Class::Expands) => {
                inv.args.problem.sweep = vec!["rho=0.5,0.6".parse().unwrap()];
            }
            // A second value each method accepts, for a flag whose default
            // under `Vary` (zero) is refused or is the method's own default.
            ("args.method.decibans_thresh", _) => m.decibans_thresh = Some(5.0),
            ("args.method.dt_check_halvings", _) => m.dt_check_halvings = Some(3),
            ("args.method.cooling_target_iters", _) => m.cooling_target_iters = Some(2),
            ("args.method.max_tree_depth", _) => m.max_tree_depth = Some(3),
            ("args.method.trajectory_warmup", _) => m.trajectory_warmup = Some(1),
            ("args.method.csmc_sweeps_per_nuts", _) => m.csmc_sweeps_per_nuts = Some(2),
            ("args.method.n_trajectories", _) => m.n_trajectories = Some(2),
            ("args.method.adapt_start", _) => m.adapt_start = Some(3),
            ("args.manifest", Class::Export) => inv.args.manifest = Some("m.json".into()),
            ("args.link_root", Class::Export) => inv.args.link_root = Some("links".into()),
            (name, Class::Identity) => return self.base.varied(name),
            (name, class) => panic!("no second value for {name} ({class:?})"),
        }
        inv
    }

    fn exercised(&self, field: &FieldInfo) -> (FitRunInvocation, FitRunInvocation) {
        match field.name.strip_prefix("args.") {
            Some(f) => {
                let (x, y) = <crate::args::FitRunArgs as Inputs>::exercised(f);
                let (mut a, mut b) = (self.base.clone(), self.base.clone());
                a.args = x;
                b.args = y;
                (a, b)
            }
            None => <FitRunInvocation as Inputs>::exercised(field.name),
        }
    }

    /// An execution input is used when what the run did with it ([`used`])
    /// is the value it recorded
    /// ([`crate::test_support::recorded_execution`]).
    fn shows_use(&self, field: &FieldInfo, seen: &Observed) -> bool {
        let effect = |prefix: &str| seen.effects.get(&format!("{prefix}.{}", field.name)).cloned();
        matches!((effect("recorded"), effect("used")), (Some(r), Some(u)) if r == u)
    }

    fn run(&self, invocation: &FitRunInvocation, store: Option<&Path>) -> store::Result<Observed> {
        let cwd = workdir();
        let fresh = tempfile::tempdir().unwrap();
        let root = store.map(Path::to_path_buf).unwrap_or_else(|| fresh.path().join("camdl-store"));
        if self.stores_base {
            let base: Vec<String> = ["fit", "run", "fit_pmmh.toml", "--label", "base", "--progress", "none"].map(String::from).into();
            let out = camdl(cwd.path(), &root, &base, &[]);
            assert!(out.status.success(), "the stored base: {}", String::from_utf8_lossy(&out.stderr));
        }
        let out = camdl(cwd.path(), &root, &argv(invocation), &[("RUST_LOG", USE_LOG)]);
        if !out.status.success() {
            return Err(store::Error::Parse {
                what: format!("camdl {:?}", argv(invocation)),
                why: String::from_utf8_lossy(&out.stderr).into_owned(),
            });
        }
        let mut effects = crate::test_support::recorded_execution(&root, crate::fit_store::FitRun::KIND);
        effects.extend(used(&String::from_utf8_lossy(&out.stderr), &cwd.path().join("ir-cache")));
        let mut exports = BTreeMap::new();
        if let Some(path) = &invocation.args.manifest {
            let bytes = std::fs::read(cwd.path().join(path)).unwrap_or_default();
            exports.insert(path.clone(), Key::digest_bytes(&bytes));
        }
        let mut copies = BTreeMap::new();
        if let Some(links) = &invocation.args.link_root {
            crate::test_support::link_exports(cwd.path(), links, &mut exports, &mut copies);
        }
        Ok(Observed { results: observe(&root), exports, copies, effects })
    }
}

/// Every declared input of `camdl fit run` keeps the promise its class makes.
#[test]
fn every_fit_run_input_keeps_its_class() {
    let fields: Vec<FieldInfo> = FitRunInvocation::fields().to_vec();
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

/// Every argument clap accepts for `fit run`, and every global option, is a
/// declared input; and every declared input is one clap accepts.
#[test]
fn every_fit_run_argument_is_declared() {
    use clap::CommandFactory;
    let cmd = crate::Cli::command();
    let sub = cmd.find_subcommand("fit").and_then(|f| f.find_subcommand("run")).expect("the fit run subcommand");
    let mut accepted: std::collections::BTreeSet<String> = sub
        .get_arguments()
        .map(|a| a.get_id().to_string())
        .filter(|id| id != "help")
        .collect();
    accepted.extend(cmd.get_arguments().map(|a| a.get_id().to_string()).filter(|id| id != "help" && id != "version"));
    let declared: std::collections::BTreeSet<String> =
        FitRunInvocation::fields().iter().map(|f| f.name.rsplit('.').next().unwrap().to_string()).collect();
    assert_eq!(accepted, declared);
}

/// Each directory's entries under `dir`, indented by depth, with the 12 hex
/// digits of each suffix replaced by `*`.
fn shape(dir: &Path, depth: usize, out: &mut Vec<String>) {
    let mut names: Vec<String> = std::fs::read_dir(dir)
        .unwrap()
        .flatten()
        .map(|e| e.file_name().to_string_lossy().into_owned())
        .filter(|n| !n.starts_with('.'))
        .collect();
    names.sort();
    for n in names {
        let starred = match n.rsplit_once('-') {
            Some((head, tail)) if tail.len() == 12 && tail.chars().all(|c| c.is_ascii_hexdigit()) => format!("{head}-*"),
            _ => n.clone(),
        };
        out.push(format!("{}{}", "  ".repeat(depth), starred));
        let p = dir.join(&n);
        if p.is_dir() {
            shape(&p, depth + 1, out);
        }
    }
}

/// The tree one IF2 fit with a held-out tail makes, as the fit example's
/// "What lands in the store" draws it: the compiled model, the problem named
/// by its model with its stored inputs, held-out rows and the model file as
/// given (here the compiled IR, so `model.ir.json`), the compile's quantities
/// part, which the problem names as lineage, and the run inside
/// it under `runs/`, grouped by the method's settings and named by the seed
/// typed. A second run of the same line is served the stored run.
#[test]
fn a_fit_makes_the_tree_of_the_proposal() {
    let cwd = workdir();
    let store = cwd.path().join("camdl-store");
    std::fs::write(
        cwd.path().join("fit_holdout.toml"),
        std::fs::read_to_string(cwd.path().join("fit.toml")).unwrap().replace("[estimate]", "[data]\nholdout_after = 15.0\n[estimate]"),
    )
    .unwrap();
    let args: Vec<String> = ["fit", "run", "fit_holdout.toml", "--seed", "3", "--progress", "none"].map(String::from).into();
    let out = camdl(cwd.path(), &store, &args, &[]);
    assert!(out.status.success(), "{}", String::from_utf8_lossy(&out.stderr));
    let mut got = Vec::new();
    shape(&store, 0, &mut got);
    let build = format!("{}-*", store::path_label(crate::version::VERSION_SHORT));
    let want = [
        build.as_str(),
        "  compiled_model",
        "    sir-*",
        "      declared.json",
        "      docs.json",
        "      identity.json",
        "      model.ir.json",
        "      run.json",
        "      source.ir.json",
        "  fit",
        "    sir-*",
        "      held_out",
        "        cases.tsv",
        "      identity.json",
        "      inputs",
        "        cases.tsv",
        "      model.ir.json",
        "      partition.json",
        "      problem.toml",
        "      run.json",
        "      runs",
        "        if2-*",
        "          seed_3-*",
        "            chain_0",
        "              final_params.toml",
        "              parameter_traces.tsv",
        "            chain_1",
        "              final_params.toml",
        "              parameter_traces.tsv",
        "            chain_evaluations.tsv",
        "            chain_starts.tsv",
        "            diagnostics.json",
        "            diagnostics.tsv",
        "            events.jsonl",
        "            execution.json",
        "            final_params.toml",
        "            fit.meta.json",
        "            fit.toml",
        "            fit_state.toml",
        "            identity.json",
        "            method.toml",
        "            mle_params.toml",
        "            progress.json",
        "            run.json",
        "            run_inputs.json",
        "            summary.json",
        "  quantities",
        "    sir-*",
        "      identity.json",
        "      quantities.json",
        "      run.json",
        "store.json",
    ];
    assert_eq!(got, want, "the store's tree:\n{}", got.join("\n"));
    let again = camdl(cwd.path(), &store, &args, &[]);
    assert!(again.status.success(), "{}", String::from_utf8_lossy(&again.stderr));
    assert!(String::from_utf8_lossy(&again.stderr).contains("cache hit"), "{}", String::from_utf8_lossy(&again.stderr));
    let mut after = Vec::new();
    shape(&store, 0, &mut after);
    assert_eq!(after, got, "the second run stores nothing new");
}

/// A resume is a new run beside its base, under the same method directory,
/// and a config file is refused where a stored run is asked for, with the
/// runs made from that file listed.
#[test]
fn a_resume_sits_beside_its_base_and_a_config_file_names_no_run() {
    let cwd = workdir();
    let store = cwd.path().join("camdl-store");
    let run = |args: &[&str]| camdl(cwd.path(), &store, &args.iter().map(|s| s.to_string()).collect::<Vec<_>>(), &[]);
    let ok = |o: Output| assert!(o.status.success(), "{}", String::from_utf8_lossy(&o.stderr));
    ok(run(&["fit", "run", "fit_pmmh.toml", "--label", "scout", "--progress", "none"]));
    ok(run(&["fit", "run", "fit_pmmh_long.toml", "--resume", "@scout", "--progress", "none"]));
    let runs = run_folders(&store);
    assert_eq!(runs.len(), 2, "{runs:?}");
    assert_eq!(runs[0].parent(), runs[1].parent(), "the resumed run sits beside its base: {runs:?}");
    let summary = run(&["fit", "summary", "fit_pmmh.toml"]);
    assert!(!summary.status.success(), "a config file names no stored run");
    let err = String::from_utf8_lossy(&summary.stderr);
    assert!(err.contains("never names a run"), "{err}");
    // The runs made from that file: the base, and not the resume, which
    // `fit_pmmh_long.toml` made.
    let listed: Vec<&PathBuf> = runs
        .iter()
        .filter(|r| err.contains(r.file_name().unwrap().to_string_lossy().rsplit_once('-').unwrap().1))
        .collect();
    assert_eq!(listed.len(), 1, "the refusal lists the one run the file made:\n{err}");
}

/// Two runs of one identity, in two stores at different times, store the same
/// `summary.json`, a result file, and it holds no time, path, label, build or
/// baseline field: the two runs differ in their store, their time, their
/// label and the path their configuration is named by.
#[test]
fn fit_run_summary_is_a_function_of_the_run() {
    let cwd = workdir();
    // The second run names its configuration by an absolute path, which the
    // run records as given.
    let summary = |name: &str, config: &str| -> Vec<u8> {
        let store = cwd.path().join(name);
        let args: Vec<String> = ["fit", "run", config, "--seed", "7", "--label", name, "--progress", "none"].map(String::from).into();
        let out = camdl(cwd.path(), &store, &args, &[]);
        assert!(out.status.success(), "{}", String::from_utf8_lossy(&out.stderr));
        let dir = run_folders(&store).into_iter().next().unwrap();
        let run: serde_json::Value = serde_json::from_slice(&std::fs::read(dir.join("run.json")).unwrap()).unwrap();
        assert_eq!(run["files"]["summary.json"]["role"], "result", "{run}");
        std::fs::read(dir.join("summary.json")).unwrap()
    };
    let a = summary("store-a", "fit_mh.toml");
    std::thread::sleep(std::time::Duration::from_millis(1100));
    let b = summary("store-b-at-another-path", &cwd.path().join("fit_mh.toml").to_string_lossy());
    assert_eq!(String::from_utf8_lossy(&a), String::from_utf8_lossy(&b));
    let text = String::from_utf8_lossy(&a);
    let v: serde_json::Value = serde_json::from_slice(&a).unwrap();
    assert_eq!(v["schema"], "camdl.fit-run-summary/v1");
    assert_eq!(v["report"]["method"], "mh");
    for field in ["created_at", "age_seconds", "fit_dir", "\"leaf\"", "config_diff_from_baseline", "camdl_version",
                  "wall_time_secs", "sampling_seconds", "\"label\"", "\"stem\"", "store-a"] {
        assert!(!text.contains(field), "summary.json holds {field}:\n{text}");
    }
    assert!(!text.contains(&cwd.path().to_string_lossy().into_owned()), "summary.json holds a path:\n{text}");
}

/// The bytes each method computes for the same fit at seed 7: the digest of
/// each result file of the run, with every line naming the build
/// (`0.1.0+<commit>`) dropped, and every line naming a run id: the problem's
/// (`fit_hash` in `mle_params.toml`, `fit_hash` and `fit_id` in
/// `summary.json`) and the run's own (`run_id` in `summary.json`), each of
/// which hashes the build's fingerprint. A chain's
/// `resume_state.bin` is left out: it holds the chain's identity.
///
/// The pins are a regression pin per platform, keyed by `target_os` and
/// `target_arch`, because the digests depend on the platform's libm (`ln`,
/// `exp` and `lgamma` differ in their last bits between macOS and glibc). That
/// the move onto the run store left these bytes unchanged was checked on macOS
/// arm64 and on linux arm64, each against the code from before the move; linux
/// x86-64 computes the linux arm64 bytes. On a platform with no pins the test
/// fails and prints the table to add.
#[test]
fn each_method_computes_what_it_computed_before_the_move() {
    use sha2::{Digest, Sha256};
    let skip = [
        "events.jsonl", "execution.json", "progress.json", "run.json", "opt_trace.tsv", "identity.json",
        "fit.meta.json", "fit.toml", "method.toml", "run_inputs.json",
    ];
    let build = format!("{}+", env!("CARGO_PKG_VERSION"));
    let digests = |dir: &Path| -> BTreeMap<String, String> {
        let mut out = BTreeMap::new();
        let mut stack = vec![dir.to_path_buf()];
        while let Some(d) = stack.pop() {
            for e in std::fs::read_dir(&d).unwrap().flatten() {
                let p = e.path();
                if p.is_dir() {
                    stack.push(p);
                    continue;
                }
                let rel = p.strip_prefix(dir).unwrap().to_string_lossy().into_owned();
                let name = e.file_name().to_string_lossy().into_owned();
                if skip.contains(&name.as_str()) || name.ends_with(".bin") {
                    continue;
                }
                let text = std::fs::read_to_string(&p).unwrap();
                let names_a_run_id = |l: &str| {
                    l.starts_with("fit_hash = ")
                        || ["\"fit_hash\": ", "\"fit_id\": ", "\"run_id\": "].iter().any(|k| l.trim_start().starts_with(k))
                };
                let kept: Vec<&str> = text.split('\n').filter(|l| !l.contains(&build) && !names_a_run_id(l)).collect();
                out.insert(rel, format!("{:x}", Sha256::digest(kept.join("\n").as_bytes()))[..16].to_string());
            }
        }
        out
    };
    let cwd = workdir();
    let mut got = BTreeMap::new();
    for m in ["if2", "pgas", "pmmh", "pfilter", "mh", "sbplx", "bobyqa", "lbfgs"] {
        let store = cwd.path().join(format!("store-{m}"));
        let args: Vec<String> = ["fit", "run", &format!("fit_{m}.toml"), "--seed", "7", "--progress", "none"].map(String::from).into();
        let out = camdl(cwd.path(), &store, &args, &[]);
        assert!(out.status.success(), "{m}: {}", String::from_utf8_lossy(&out.stderr));
        let dir = run_folders(&store).into_iter().next().unwrap();
        got.insert(m.to_string(), digests(&dir));
    }
    crate::test_support::assert_pinned(PINS, &got, "the files each method computes moved");
}

/// Each method's fixture run holds exactly the files its method declares:
/// those that do not depend on the outcome (`MethodFiles::files`) and those
/// its recorded outcome states (`MethodFiles::outcome_files`, from the run's
/// own `chain_starts.tsv` and `pgas_summary.json`), each in its declared
/// role. The method is planned from the fixture's own `[method]` table, so
/// PGAS's `n_trajectories` is the fixture's. Each stored table's header row
/// is the one its variant states in the result-file contract
/// (`ResultFile::columns`). Every chain of these fixtures runs to the end.
#[test]
fn every_method_declares_its_files() {
    use crate::result_files::fit_run::{ChainEnd, FitOutcome, PlannedMethod};
    use crate::result_files::{header_matches, ResultFile, Role};
    let cwd = workdir();
    for m in ["if2", "pgas", "pmmh", "pfilter", "pfilter_quiet", "mh", "sbplx", "bobyqa", "lbfgs"] {
        let path = cwd.path().join(format!("fit_{m}.toml"));
        let text = std::fs::read_to_string(&path).unwrap();
        let config = crate::fit::config_v2::FitConfig::parse_anchored_at(&text, &path, &path).unwrap();
        let algorithm = &config.inference.method.as_ref().unwrap().algorithm;
        let planned = PlannedMethod::of(algorithm);
        let n_chains = u32::try_from(algorithm.chains()).unwrap();

        let store = cwd.path().join(format!("store-{m}"));
        let args: Vec<String> = ["fit", "run", &format!("fit_{m}.toml"), "--seed", "7", "--progress", "none"].map(String::from).into();
        let out = camdl(cwd.path(), &store, &args, &[]);
        assert!(out.status.success(), "{m}: {}", String::from_utf8_lossy(&out.stderr));
        let dir = run_folders(&store).into_iter().next().unwrap();
        let outcome = FitOutcome::read(&dir);
        for c in 0..n_chains {
            assert_eq!(outcome.end(c), ChainEnd::Accepted, "{m}: chain {c} of the fixture runs to the end");
        }
        let declared: BTreeMap<String, crate::result_files::fit_run::FitRunFile> = planned
            .files(n_chains, None)
            .into_iter()
            .chain(planned.outcome_files(n_chains, &outcome))
            .map(|f| (f.stored(), f))
            .collect();
        let run: serde_json::Value = serde_json::from_slice(&std::fs::read(dir.join("run.json")).unwrap()).unwrap();
        let got: BTreeMap<String, String> = run["files"]
            .as_object()
            .unwrap()
            .iter()
            .filter(|(_, e)| e["role"] != "attempt")
            .map(|(n, e)| (n.clone(), e["role"].as_str().unwrap().to_string()))
            .collect();
        let want: BTreeMap<String, String> = declared
            .iter()
            .map(|(n, f)| (n.clone(), if f.role() == Role::Result { "result" } else { "provenance" }.to_string()))
            .collect();
        assert_eq!(got, want, "{m}: the files its run holds, against the files its method and outcome declare");
        for (name, file) in &declared {
            if file.columns().is_empty() {
                continue;
            }
            let text = std::fs::read_to_string(dir.join(name)).unwrap();
            let mut lines = text.lines();
            let header = if file.commented() {
                lines.find(|l| !l.starts_with('#')).unwrap_or("")
            } else {
                let first = lines.next().unwrap_or("");
                assert!(!first.starts_with('#'), "{m}: `{name}` opens with a comment line the contract does not state");
                first
            };
            let header: Vec<&str> = header.split('\t').collect();
            assert!(header_matches(file.columns(), &header), "{m}: `{name}`'s header {header:?} is not the contract's {:?}", file.columns());
        }
        if m == "pgas" {
            for f in ["filter_ess.tsv", "cross_chain_compat.json", "chain_0/trajectories.tsv", "chain_1/trajectories.json"] {
                assert!(want.contains_key(f), "pgas: the fixture's outcome states `{f}`: {want:?}");
            }
        }
    }
}

/// The pins of [`each_method_computes_what_it_computed_before_the_move`], per
/// `(target_os, target_arch)`.
const PINS: &[((&str, &str), Pins)] = &[
    (("macos", "aarch64"), MACOS_AARCH64),
    (("linux", "aarch64"), LINUX),
    (("linux", "x86_64"), LINUX),
];

/// macOS arm64: taken from the old store's runs of the same fixtures.
const MACOS_AARCH64: Pins = &[
    ("if2", &[
        ("chain_0/final_params.toml", "b97b9ff5f353bf92"),
        ("chain_0/parameter_traces.tsv", "cb84c128fdcbc9a3"),
        ("chain_1/final_params.toml", "35f27452398c553f"),
        ("chain_1/parameter_traces.tsv", "5e73e9296f17cebb"),
        ("chain_evaluations.tsv", "2923ff37b508ffb3"),
        ("chain_starts.tsv", "292f106ae1b81126"),
        ("diagnostics.json", "5cd12af2ed706bbe"),
        ("diagnostics.tsv", "f71d947e76c79190"),
        ("final_params.toml", "a0dab3f9637a3c9a"),
        ("fit_state.toml", "6e9941f83e6d8e7d"),
        ("mle_params.toml", "73e526922980c918"),
        ("summary.json", "9080721789552981"),
    ]),
    ("pgas", &[
        ("chain_0/trace.tsv", "3521d6750b3cc250"),
        ("chain_0/trajectories.json", "eeaadccdfd0124cf"),
        ("chain_0/trajectories.tsv", "4f929cad9a282ccc"),
        ("chain_1/trace.tsv", "c17427750ba594be"),
        ("chain_1/trajectories.json", "eeaadccdfd0124cf"),
        ("chain_1/trajectories.tsv", "5ec1beba0b8ee3b0"),
        ("chain_starts.tsv", "292f106ae1b81126"),
        ("cross_chain_compat.json", "3e24690854c0a743"),
        ("diagnostics.json", "04729676dd7c5f4f"),
        ("draws.tsv", "79b665a9f1d748eb"),
        ("filter_ess.tsv", "554688ea3f8badab"),
        ("fit_state.toml", "e67fddbcdf771833"),
        ("pgas_summary.json", "bbdcb6daca32e217"),
        ("summary.json", "885c5627560de3b1"),
    ]),
    ("pmmh", &[
        ("chain_0/trace.tsv", "92443daae545beb8"),
        ("chain_1/trace.tsv", "db2aa4819183236f"),
        ("chain_starts.tsv", "292f106ae1b81126"),
        ("diagnostics.json", "4f53cda18c2baa0c"),
        ("draws.tsv", "5994b706bd37af2b"),
        ("fit_state.toml", "6c55b85981879ba8"),
        ("pmmh_summary.json", "2be2a76600467f11"),
        ("summary.json", "20008983b19f8f99"),
    ]),
    ("pfilter", &[
        ("logliks.tsv", "a6ad8f50f8dc90c0"),
        ("prequential.json", "5e41e81e467a7023"),
        ("prequential.streams.tsv", "ef3604651959eab8"),
        ("prequential.tsv", "c7945ecb9f5cc355"),
        ("summary.json", "b50e036ada4e5c04"),
    ]),
    ("mh", &[
        ("chain_0/trace.tsv", "eca3a76177c94127"),
        ("chain_1/trace.tsv", "4fd257df73bd1dfc"),
        ("chain_starts.tsv", "292f106ae1b81126"),
        ("diagnostics.json", "e06e1a6aedf33167"),
        ("draws.tsv", "fa9f7eec64e9b83d"),
        ("fit_state.toml", "378f1260aac7388e"),
        ("mh_summary.json", "73bd43f302f43a55"),
        ("summary.json", "bd4d73f76b1147c6"),
    ]),
    ("sbplx", &[
        ("chain_results.tsv", "dbdead979318df7d"),
        ("diagnostics.json", "4f53cda18c2baa0c"),
        ("fit_state.toml", "e7280a7cdba437eb"),
        ("mle_params.toml", "7f5a3480f4b3eb8b"),
        ("summary.json", "3721196ec2e37eff"),
    ]),
    ("bobyqa", &[
        ("chain_results.tsv", "fffd90043409d23a"),
        ("diagnostics.json", "4f53cda18c2baa0c"),
        ("fit_state.toml", "b3f101419aae03fc"),
        ("mle_params.toml", "0899337e6d6f7949"),
        ("summary.json", "45c327f014a7ee26"),
    ]),
    ("lbfgs", &[
        ("chain_results.tsv", "4db4a4b639626dc1"),
        ("diagnostics.json", "4f53cda18c2baa0c"),
        ("fit_state.toml", "9fbc652a646888de"),
        ("mle_params.toml", "697bcdcd8b6fa78c"),
        ("summary.json", "f09adf3ef96ac626"),
    ]),
];

/// linux (glibc), arm64 and x86-64 alike: computed on camdl-cloud (arm64),
/// identical before and after the move, and on GitHub CI (x86-64) after it.
const LINUX: Pins = &[
    ("if2", &[
        ("chain_0/final_params.toml", "b97b9ff5f353bf92"),
        ("chain_0/parameter_traces.tsv", "323f55f46a4e7f40"),
        ("chain_1/final_params.toml", "35f27452398c553f"),
        ("chain_1/parameter_traces.tsv", "1c9eca97a3bf59f4"),
        ("chain_evaluations.tsv", "3db3458c992b13bf"),
        ("chain_starts.tsv", "292f106ae1b81126"),
        ("diagnostics.json", "5cd12af2ed706bbe"),
        ("diagnostics.tsv", "f71d947e76c79190"),
        ("final_params.toml", "a0dab3f9637a3c9a"),
        ("fit_state.toml", "7dbe99255bc21ff9"),
        ("mle_params.toml", "73e526922980c918"),
        ("summary.json", "df6c7834aab3c696"),
    ]),
    ("pgas", &[
        ("chain_0/trace.tsv", "3521d6750b3cc250"),
        ("chain_0/trajectories.json", "eeaadccdfd0124cf"),
        ("chain_0/trajectories.tsv", "4f929cad9a282ccc"),
        ("chain_1/trace.tsv", "c17427750ba594be"),
        ("chain_1/trajectories.json", "eeaadccdfd0124cf"),
        ("chain_1/trajectories.tsv", "5ec1beba0b8ee3b0"),
        ("chain_starts.tsv", "292f106ae1b81126"),
        ("cross_chain_compat.json", "26d6b9c7cd8be49b"),
        ("diagnostics.json", "04729676dd7c5f4f"),
        ("draws.tsv", "79b665a9f1d748eb"),
        ("filter_ess.tsv", "554688ea3f8badab"),
        ("fit_state.toml", "e67fddbcdf771833"),
        ("pgas_summary.json", "bbdcb6daca32e217"),
        ("summary.json", "885c5627560de3b1"),
    ]),
    ("pmmh", &[
        ("chain_0/trace.tsv", "92443daae545beb8"),
        ("chain_1/trace.tsv", "db2aa4819183236f"),
        ("chain_starts.tsv", "292f106ae1b81126"),
        ("diagnostics.json", "4f53cda18c2baa0c"),
        ("draws.tsv", "5994b706bd37af2b"),
        ("fit_state.toml", "6c55b85981879ba8"),
        ("pmmh_summary.json", "2be2a76600467f11"),
        ("summary.json", "20008983b19f8f99"),
    ]),
    ("pfilter", &[
        ("logliks.tsv", "a6ad8f50f8dc90c0"),
        ("prequential.json", "5e41e81e467a7023"),
        ("prequential.streams.tsv", "ef3604651959eab8"),
        ("prequential.tsv", "c7945ecb9f5cc355"),
        ("summary.json", "b50e036ada4e5c04"),
    ]),
    ("mh", &[
        ("chain_0/trace.tsv", "eca3a76177c94127"),
        ("chain_1/trace.tsv", "5644bd42bb0d5684"),
        ("chain_starts.tsv", "292f106ae1b81126"),
        ("diagnostics.json", "e06e1a6aedf33167"),
        ("draws.tsv", "fa9f7eec64e9b83d"),
        ("fit_state.toml", "378f1260aac7388e"),
        ("mh_summary.json", "73bd43f302f43a55"),
        ("summary.json", "bd4d73f76b1147c6"),
    ]),
    ("sbplx", &[
        ("chain_results.tsv", "dbdead979318df7d"),
        ("diagnostics.json", "4f53cda18c2baa0c"),
        ("fit_state.toml", "e7280a7cdba437eb"),
        ("mle_params.toml", "7f5a3480f4b3eb8b"),
        ("summary.json", "3721196ec2e37eff"),
    ]),
    ("bobyqa", &[
        ("chain_results.tsv", "13807c14836c3048"),
        ("diagnostics.json", "4f53cda18c2baa0c"),
        ("fit_state.toml", "597ac41b3e837bd4"),
        ("mle_params.toml", "0899337e6d6f7949"),
        ("summary.json", "10b522dfa1526183"),
    ]),
    ("lbfgs", &[
        ("chain_results.tsv", "4db4a4b639626dc1"),
        ("diagnostics.json", "4f53cda18c2baa0c"),
        ("fit_state.toml", "9fbc652a646888de"),
        ("mle_params.toml", "697bcdcd8b6fa78c"),
        ("summary.json", "f09adf3ef96ac626"),
    ]),
];

/// A stored fit keeps its held-out rows' dates and not their values, so
/// `compare` of two IF2 runs that held rows out refuses with no outcome file,
/// scores the held-out rows from the files the fit was given (a
/// `holdout_after` fit's whole file by `--data`, a `[data.holdout]` file by
/// `--data-held-out`), and scores the training rows with `--in-sample`, with
/// any `--draws`.
#[test]
fn compare_scores_held_out_rows_of_a_stored_fit_from_their_outcome_files() {
    let cwd = workdir();
    let store = cwd.path().join("camdl-store");
    let fit = std::fs::read_to_string(cwd.path().join("fit.toml")).unwrap();
    std::fs::write(cwd.path().join("fit_after.toml"), fit.replace("[estimate]", "[data]\nholdout_after = 15.0\n[estimate]")).unwrap();
    let prevalent = (1..=20).fold("time\tprevalent\n".to_string(), |s, t| s + &format!("{t}\t{}\n", 10 + 2 * t));
    let (train, held): (Vec<&str>, Vec<&str>) = {
        let lines: Vec<&str> = prevalent.lines().skip(1).collect();
        (lines[..15].to_vec(), lines[15..].to_vec())
    };
    // Every held-out row ends after every stream's training rows.
    std::fs::write(cwd.path().join("cases15.tsv"), (1..=15).fold("time\tcases\n".to_string(), |s, t| s + &format!("{t}\t{}\n", 3 + t))).unwrap();
    std::fs::write(cwd.path().join("prev_train.tsv"), format!("time\tprevalent\n{}\n", train.join("\n"))).unwrap();
    std::fs::write(cwd.path().join("prev_held.tsv"), format!("time\tprevalent\n{}\n", held.join("\n"))).unwrap();
    std::fs::write(
        cwd.path().join("fit_partial.toml"),
        problem("sir_prevalent.ir.json", 0.5)
            .replace("cases = \"cases.tsv\"\n", "cases = \"cases15.tsv\"\nprevalent = \"prev_train.tsv\"\n[data.holdout]\nprevalent = \"prev_held.tsv\"\n")
            + method("if2"),
    )
    .unwrap();
    let run = |args: &[&str]| camdl(cwd.path(), &store, &args.iter().map(|s| s.to_string()).collect::<Vec<_>>(), &[]);
    let ok = |o: &Output| assert!(o.status.success(), "{}", String::from_utf8_lossy(&o.stderr));
    for (config, label) in [("fit_after.toml", "after"), ("fit_partial.toml", "partial")] {
        ok(&run(&["fit", "run", config, "--label", label, "--progress", "none"]));
        ok(&run(&["fit", "run", config, "--seed", "2", "--label", &format!("{label}2"), "--progress", "none"]));
        let (a, b) = (format!("@{label}"), format!("@{label}2"));
        let base = ["compare", a.as_str(), b.as_str(), "--replicates", "1", "--format", "json"];
        let refused = run(&base);
        let err = String::from_utf8_lossy(&refused.stderr);
        assert_eq!(refused.status.code(), Some(1), "{label}:\n{err}");
        assert!(err.contains("the held-out rows lie past") && err.contains("--in-sample"), "{label}: {err}");
        let outcomes: &[&str] = match label {
            "after" => &["--data", "cases=cases.tsv"],
            _ => &["--data", "cases=cases15.tsv", "--data-held-out", "prevalent=prev_held.tsv"],
        };
        for (extra, role) in [(outcomes, "held_out"), (&["--in-sample"][..], "trained")] {
            let out = run(&[&base[..], extra].concat());
            ok(&out);
            let v: serde_json::Value = serde_json::from_slice(&out.stdout).unwrap();
            for row in v["rows"].as_array().unwrap() {
                assert_eq!(row["role"], role, "{label}: {row}");
                assert_eq!(row["derivation"]["theta"], "optimizer_winner", "{label}: {row}");
            }
        }
        // `--draws` reaches only a posterior: an optimizer run is predicted
        // at its winner whatever it is.
        let out = run(&[&base[..], &["--in-sample", "--draws", "20"]].concat());
        ok(&out);
        let v: serde_json::Value = serde_json::from_slice(&out.stdout).unwrap();
        for row in v["rows"].as_array().unwrap() {
            assert_eq!(row["derivation"]["theta"], "optimizer_winner", "{label}: {row}");
        }
    }
}

/// A prediction evaluates the quantities the command that names the run
/// compiled. A user fits a model, edits only its quantities block and runs
/// `fit run` again: the problem and the run are reused, since the
/// quantities do not change the fit, and that invocation records the edited
/// quantities. `camdl predict @second` evaluates the edited block and `@first`
/// the original; the run's id, which both invocations named with different
/// quantities, is refused with them listed.
#[test]
fn a_prediction_evaluates_the_quantities_of_the_invocation_it_names() {
    let cwd = workdir();
    let store = cwd.path().join("camdl-store");
    let with_quantities = |q: &str| format!("{SIR}\nquantities {{\n  {q}\n}}\n");
    std::fs::write(cwd.path().join("q.camdl"), with_quantities("peak_I = max(I)")).unwrap();
    std::fs::write(cwd.path().join("fit_q.toml"), problem("q.camdl", 0.5) + method("pmmh")).unwrap();
    let run = |args: &[&str]| camdl(cwd.path(), &store, &args.iter().map(|s| s.to_string()).collect::<Vec<_>>(), &[]);
    let ok = |o: Output| -> String {
        assert!(o.status.success(), "{}", String::from_utf8_lossy(&o.stderr));
        String::from_utf8_lossy(&o.stderr).into_owned()
    };
    ok(run(&["fit", "run", "fit_q.toml", "--label", "first", "--progress", "none"]));
    std::fs::write(cwd.path().join("q.camdl"), with_quantities("peak_R = max(R)")).unwrap();
    let again = ok(run(&["fit", "run", "fit_q.toml", "--label", "second", "--progress", "none"]));
    assert!(again.contains("cache hit"), "the edited quantities reuse the run:\n{again}");
    assert_eq!(run_folders(&store).len(), 1, "one run");

    let quantities_of = |handle: &str| -> Vec<String> {
        ok(run(&["predict", handle, "--mode", "free_forward", "--n-draws", "2"]));
        let mut names: Vec<String> = folders(&store)
            .into_iter()
            .filter(|(_, id, _)| id["kind"] == "prediction")
            .flat_map(|(p, _, _)| std::fs::read_dir(p.join("quantities-model")).into_iter().flatten().flatten())
            .map(|e| e.file_name().to_string_lossy().into_owned())
            .collect();
        names.sort();
        names.dedup();
        names
    };
    assert_eq!(quantities_of("@second"), ["peak_R.tsv"], "the edited block, as the second invocation compiled it");
    assert_eq!(quantities_of("@first"), ["peak_I.tsv", "peak_R.tsv"], "the original block, beside it");

    let id = run_folders(&store)[0].file_name().unwrap().to_string_lossy().rsplit_once('-').unwrap().1.to_string();
    let refused = run(&["predict", &id, "--mode", "free_forward", "--n-draws", "2"]);
    let err = String::from_utf8_lossy(&refused.stderr);
    assert!(!refused.status.success(), "an id two invocations compiled differently is refused:\n{err}");
    assert!(err.contains("different quantities blocks") && err.contains("invocation 1") && err.contains("invocation 2")
        && err.contains("@label"), "{err}");
}

/// A filter run given a stored fit that held rows out is refused before it
/// runs: the stored problem keeps its training lines only, so filtering it
/// would score fewer rows than the same problem named by its fit file, and
/// two names for one problem would give two numbers. The fit file scores
/// every row; a stored fit with no held-out rows is read as before.
#[test]
fn pfilter_refuses_a_stored_fit_that_held_rows_out() {
    let cwd = workdir();
    let store = cwd.path().join("camdl-store");
    let fit = std::fs::read_to_string(cwd.path().join("fit.toml")).unwrap();
    std::fs::write(cwd.path().join("fit_h.toml"), fit.replace("[estimate]", "[data]\nholdout_after = 10.0\n[estimate]")).unwrap();
    let run = |args: &[&str]| camdl(cwd.path(), &store, &args.iter().map(|s| s.to_string()).collect::<Vec<_>>(), &[]);
    let ok = |o: &Output| assert!(o.status.success(), "{}", String::from_utf8_lossy(&o.stderr));
    ok(&run(&["fit", "run", "fit_h.toml", "--label", "h", "--progress", "none"]));
    ok(&run(&["fit", "run", "fit.toml", "--label", "whole", "--progress", "none"]));
    let filter = |fit: &str| {
        run(&["pfilter", "sir.ir.json", "--fit", fit, "--param", "beta=0.4", "--param", "gamma=0.2",
            "--particles", "50", "--seed", "1", "--progress", "none"])
    };
    let refused = filter("@h");
    let err = String::from_utf8_lossy(&refused.stderr);
    assert!(!refused.status.success(), "a stored fit that held rows out is refused:\n{err}");
    assert!(err.contains("keeps no held-out values") && err.contains("--fit <the fit config>") && err.contains("step 7"),
        "{err}");
    ok(&filter("fit_h.toml"));
    ok(&filter("@whole"));
}

/// A fit run is written in place: while a long PMMH fit runs, its folder in
/// the store reads as running and its chain's `trace.tsv` already holds rows,
/// beside the run's progress snapshot and event log. A process killed while
/// it runs leaves the folder, which the store reads as interrupted, and the
/// next run of the same fit claims the folder again.
#[test]
fn a_running_fit_shows_its_traces_growing_in_its_folder() {
    let cwd = workdir();
    let store_root = cwd.path().join("camdl-store");
    let long = problem("sir.ir.json", 0.5) + &method("pmmh").replace("iterations = 20", "iterations = 200000");
    std::fs::write(cwd.path().join("fit_long.toml"), long).unwrap();
    let fx = fixture();
    // Killed when dropped, so a failing assertion leaves no fit running.
    struct Running(std::process::Child);
    impl Drop for Running {
        fn drop(&mut self) {
            let _ = self.0.kill();
            let _ = self.0.wait();
        }
    }
    let spawn = || {
        Running(Command::new(&fx.camdl)
            .current_dir(cwd.path())
            .env_clear()
            .env("PATH", std::env::var_os("PATH").unwrap_or_default())
            .env("HOME", cwd.path().join("home"))
            .env("CAMDL_SKIP_VERSION_CHECK", "1")
            .env("CAMDLC", &fx.camdlc)
            .env("CAMDL_IR_CACHE_DIR", cwd.path().join("ir-cache"))
            .env("CAMDL_STORE", &store_root)
            .args(["fit", "run", "fit_long.toml", "--seed", "2", "--progress", "none"])
            .stderr(std::process::Stdio::null())
            .spawn()
            .unwrap())
    };
    // The run's folder once its record says running and its first chain's
    // trace holds at least two rows under its header.
    let growing = || -> Option<PathBuf> {
        let runs = complete_folders(&store_root).ok()?;
        let (dir, _, record) = runs.into_iter().find(|(_, id, _)| id["kind"] == "fit_run")?;
        let rows = std::fs::read_to_string(dir.join("chain_0/trace.tsv")).ok()?.lines().count().saturating_sub(1);
        (record["status"] == "running" && rows >= 2).then_some(dir)
    };
    let wait_for = |what: &str, f: &dyn Fn() -> Option<PathBuf>| -> PathBuf {
        let started = std::time::Instant::now();
        loop {
            if let Some(d) = f() {
                return d;
            }
            assert!(started.elapsed() < std::time::Duration::from_secs(120), "{what}");
            std::thread::sleep(std::time::Duration::from_millis(50));
        }
    };

    let child = spawn();
    let dir = wait_for("the running fit's trace grows in its folder", &growing);
    assert!(dir.join("progress.json").is_file(), "the progress snapshot is beside the trace");
    let events = std::fs::read_to_string(dir.join("events.jsonl")).unwrap();
    let first: serde_json::Value = serde_json::from_str(events.lines().next().unwrap()).unwrap();
    assert_eq!(first["event"]["event"], "begin", "{events}");
    let rows_then = std::fs::read_to_string(dir.join("chain_0/trace.tsv")).unwrap().lines().count();
    let more = || -> Option<PathBuf> {
        let rows = std::fs::read_to_string(dir.join("chain_0/trace.tsv")).ok()?.lines().count();
        (rows > rows_then).then(|| dir.clone())
    };
    wait_for("the trace keeps growing", &more);
    drop(child);

    let store = store::Store::open(
        &store_root,
        crate::camdl_store::ambient(),
        crate::camdl_store::camdl_top_level(),
        crate::camdl_store::camdl_kinds(),
    )
    .unwrap();
    let entry = store
        .resolve_entry::<crate::fit_store::FitRun>(&store::RunRef::new(store::Handle::Path(dir.clone())))
        .unwrap();
    assert!(matches!(entry.state(), store::EntryState::Interrupted { .. }), "{:?}", entry.state());

    let again = spawn();
    let pid = serde_json::Value::from(again.0.id());
    let reclaimed = || -> Option<PathBuf> {
        let record: serde_json::Value = serde_json::from_slice(&std::fs::read(dir.join("run.json")).ok()?).ok()?;
        (record["status"] == "running" && record["owner"]["pid"] == pid && dir.join(".attempts").is_dir())
            .then(|| dir.clone())
    };
    wait_for("the next run claims the interrupted folder", &reclaimed);
}

/// A label's history is the store's: an invocation whose results did not
/// all complete (here a run killed while it ran) never held the label, so
/// `@lab~1` names the run before it, and the history a refusal lists counts
/// only the holders.
#[test]
fn a_killed_run_holds_no_place_in_a_labels_history() {
    let cwd = workdir();
    let store = cwd.path().join("camdl-store");
    let run = |args: &[&str]| camdl(cwd.path(), &store, &args.iter().map(|s| s.to_string()).collect::<Vec<_>>(), &[]);
    let ok = |o: Output| -> String {
        assert!(o.status.success(), "{}", String::from_utf8_lossy(&o.stderr));
        String::from_utf8_lossy(&o.stderr).into_owned()
    };
    ok(run(&["fit", "run", "fit.toml", "--seed", "1", "--label", "lab", "--progress", "none"]));
    let first = run_folders(&store);
    assert_eq!(first.len(), 1);

    // A long run under the label, killed while it runs.
    let long = problem("sir.ir.json", 0.5) + &method("pmmh").replace("iterations = 20", "iterations = 200000");
    std::fs::write(cwd.path().join("fit_long.toml"), long).unwrap();
    let fx = fixture();
    let mut child = Command::new(&fx.camdl)
        .current_dir(cwd.path())
        .env_clear()
        .env("PATH", std::env::var_os("PATH").unwrap_or_default())
        .env("HOME", cwd.path().join("home"))
        .env("CAMDL_SKIP_VERSION_CHECK", "1")
        .env("CAMDLC", &fx.camdlc)
        .env("CAMDL_IR_CACHE_DIR", cwd.path().join("ir-cache"))
        .env("CAMDL_STORE", &store)
        .args(["fit", "run", "fit_long.toml", "--seed", "2", "--label", "lab", "--progress", "none"])
        .stderr(std::process::Stdio::null())
        .spawn()
        .unwrap();
    let started = std::time::Instant::now();
    // The long run is claimed once both of its records can be read.
    let claimed = || complete_folders(&store).map_or(0, |f| f.iter().filter(|(_, id, _)| id["kind"] == "fit_run").count());
    while claimed() < 2 {
        assert!(started.elapsed() < std::time::Duration::from_secs(60), "the long run started");
        std::thread::sleep(std::time::Duration::from_millis(100));
    }
    child.kill().unwrap();
    let _ = child.wait();

    ok(run(&["fit", "run", "fit.toml", "--seed", "3", "--label", "lab", "--progress", "none"]));
    let third: Vec<PathBuf> = run_folders(&store).into_iter().filter(|r| r.to_string_lossy().contains("seed_3-")).collect();
    assert_eq!(third.len(), 1);
    let name = |p: &PathBuf| p.file_name().unwrap().to_string_lossy().into_owned();

    let now = ok(run(&["fit", "summary", "@lab"]));
    assert!(now.contains(&name(&third[0])) && now.contains("(1 earlier holder)"), "{now}");
    let before = ok(run(&["fit", "summary", "@lab~1"]));
    assert!(before.contains(&name(&first[0])), "@lab~1 is the first run, not the killed one:\n{before}");
    let past = run(&["fit", "summary", "@lab~2"]);
    let err = String::from_utf8_lossy(&past.stderr);
    assert!(!past.status.success());
    assert!(err.contains("has 2 holder(s)"), "the history counts the holders:\n{err}");
}

/// With `--save-final-state`, a filter run binds only the fit's training rows
/// and saves the state at their end, so a stored fit that held rows out and
/// its fit file are one computation: the same log-likelihood and the same
/// final state, byte for byte.
#[test]
fn a_saved_final_state_reads_a_stored_holdout_fit_as_its_fit_file() {
    let cwd = workdir();
    let store = cwd.path().join("camdl-store");
    let fit = std::fs::read_to_string(cwd.path().join("fit.toml")).unwrap();
    std::fs::write(cwd.path().join("fit_h.toml"), fit.replace("[estimate]", "[data]\nholdout_after = 10.0\n[estimate]")).unwrap();
    let run = |args: &[&str]| camdl(cwd.path(), &store, &args.iter().map(|s| s.to_string()).collect::<Vec<_>>(), &[]);
    let ok = |o: &Output| assert!(o.status.success(), "{}", String::from_utf8_lossy(&o.stderr));
    ok(&run(&["fit", "run", "fit_h.toml", "--label", "h", "--progress", "none"]));
    let filter = |fit: &str, tag: &str| -> (Vec<u8>, Vec<u8>) {
        let (ll, state) = (format!("ll_{tag}.txt"), format!("state_{tag}.tsv"));
        ok(&run(&["pfilter", "sir.ir.json", "--fit", fit, "--param", "beta=0.4", "--param", "gamma=0.2",
            "--particles", "50", "--seed", "1", "--output", &ll, "--save-final-state", &state, "--progress", "none"]));
        (std::fs::read(cwd.path().join(ll)).unwrap(), std::fs::read(cwd.path().join(state)).unwrap())
    };
    let (ll_stored, state_stored) = filter("@h", "stored");
    let (ll_file, state_file) = filter("fit_h.toml", "file");
    assert_eq!(String::from_utf8_lossy(&ll_stored), String::from_utf8_lossy(&ll_file), "one log-likelihood");
    assert_eq!(state_stored, state_file, "one final state");
}
