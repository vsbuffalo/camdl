//! `camdl simulate` on the run store, checked by running the release binary:
//! every declared input against its class (`store::check_input_classes`), the
//! directory tree a run makes, reuse, by-products added later, a stored
//! failure, the combined tables, labels, and the trajectories of three
//! fixtures against the bytes the old store held.
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

use store::{Class, ClassFailure, FieldInfo, InputHarness, Inputs, Key, Observed, ObservedResult};

use crate::args::SimulateInvocation;
use crate::input_fixtures::{sim_base, sim_from};

const SIR: &str = r#"
time_unit = 'days

compartments { S, I, R, V }

let N = S + I + R + V

parameters {
  beta_a : rate        in [0.01, 5.0]  ~ log_normal(mu = -1.0, sigma = 0.3)
  beta_b : rate        in [0.01, 5.0]
  gamma  : rate        in [0.01, 5.0]
  rho    : probability in [0.001, 1.0]
}

transitions {
  #[lineage]
  infection_a : S --> I  @ beta_a * exp(-rho) * S * I / N
  infection_b : S --> I  @ beta_b * S * I / N
  recovery    : I --> R  @ gamma * I
}

init { S = 990  I = 10  R = 0  V = 0 }

interventions {
  sia  : transfer(fraction = 0.5, from = S, to = V) at [5 'days]
  sia2 : transfer(fraction = 0.2, from = S, to = V) at [8 'days]
}

scenarios {
  fast     { set = { beta_b = 0.3 } }
  with_sia { enable = [sia] }
  with_sia_too { enable = [sia] }
  slow     { disable = [sia2] }
}

simulate { from = 0 'days  to = 20 'days }

observations {
  cases {
    columns       { time : time, cases : count }
    covers        = closing_at(time, 1 'days)
    projected     = incidence(infection_a)
    emit_schedule = every 1 'days
    cases ~ poisson(rate = rho * projected)
  }
}

quantities {
  peak = max(I / N)
}
"#;

const PATCH: &str = r#"
time_unit = 'days
dimensions { patch = [p0, p1] }
compartments { S, I, R }
stratify(by = patch)
parameters { beta : rate in [0.01, 2.0]  gamma : rate in [0.01, 1.0] }
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
simulate { from = 0 'days  to = 10 'days }
"#;

/// A rate that is negative below `beta = 0.3`: the engine stops with an
/// error at those parameters, whatever the seed.
const NEGATIVE: &str = r#"
time_unit = 'days
compartments { S, I }
parameters { beta : rate in [0.01, 2.0] }
let c : rate = 0.3
transitions { infection : S --> I @ (beta - c) * S * I / (S + I) }
init { S = 990  I = 10 }
simulate { from = 0 'days  to = 10 'days }
"#;

fn fit_toml(fixed_gamma: f64) -> String {
    format!(
        "[model]\ncamdl = \"sir.ir.json\"\n[data.observations]\ncases = \"cases.tsv\"\n\
         [estimate]\nbeta_a = {{ bounds = [0.01, 1.0], start = 0.3, prior = {{ uniform = {{}} }} }}\n\
         [fixed]\nbeta_b = 0.2\ngamma = {fixed_gamma}\nrho = 0.5\n\
         [method]\nalgorithm = \"if2\"\nbackend = \"chain_binomial\"\nchains = 1\nparticles = 50\niterations = 3\ncooling = 0.7\n"
    )
}

fn fixture_files() -> Vec<(&'static str, String)> {
    let mut post = String::from("chain\tdraw\tbeta_a\tbeta_b\tgamma\trho\n");
    for chain in 0..3 {
        for draw in 0..2 {
            post.push_str(&format!("{chain}\t{draw}\t{}\t0.2\t0.2\t0.5\n", 0.3 + 0.05 * (2 * chain + draw) as f64));
        }
    }
    vec![
        ("sir.camdl", SIR.to_string()),
        ("sir_other.camdl", SIR.replace("S = 990  I = 10", "S = 980  I = 20")),
        ("sir_dated.camdl", SIR.replace("time_unit = 'days\n", "time_unit = 'days\norigin = date(\"2025-01-01\")\n")),
        ("patch.camdl", PATCH.to_string()),
        ("sir_anchor.camdl", SIR.replace("peak = max(I / N)", "late = value_at(I, last_obs)")),
        ("negative.camdl", NEGATIVE.to_string()),
        ("vocabulary.camdl", "quantities {\n  peak_i = max(I)\n}\n".into()),
        ("p.toml", "beta_a = 0.4\nbeta_b = 0.2\ngamma = 0.2\nrho = 0.5\n".into()),
        ("p2.toml", "beta_a = 0.5\nbeta_b = 0.2\ngamma = 0.2\nrho = 0.5\n".into()),
        ("patch.toml", "beta = 0.4\ngamma = 0.2\n".into()),
        ("w1.tsv", "1.0\n0.5\n".into()),
        ("w2.tsv", "1.0\n0.7\n".into()),
        ("beta_vec.tsv", "a\t0.45\nb\t0.25\n".into()),
        ("cases.tsv", "time\tcases\n1\t3\n2\t4\n3\t5\n".into()),
        ("fit.toml", fit_toml(0.2)),
        ("fit2.toml", fit_toml(0.25)),
        ("state.tsv", "# camdl-final-state v1\tt=10\nparticle\tS\tI\tR\tV\n0\t800\t50\t150\t0\n1\t810\t40\t150\t0\n".into()),
        ("draws2.tsv", "beta_a\tbeta_b\n0.3\t0.2\n0.4\t0.2\n".into()),
        ("draws3.tsv", "beta_a\tbeta_b\n0.3\t0.2\n0.4\t0.2\n0.5\t0.2\n".into()),
        ("draws_a.tsv", "beta_a\tbeta_b\n0.3\t0.2\n0.4\t0.2\n".into()),
        ("sweep2.tsv", "beta_a\n0.3\n0.4\n".into()),
        ("sweep3.tsv", "beta_a\n0.3\n0.4\n0.5\n".into()),
        ("post/draws.tsv", post),
    ]
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
        let dir = crate::test_support::unique_temp_dir("sim_store_fixture");
        std::fs::create_dir_all(dir.join("post")).unwrap();
        for (name, text) in fixture_files() {
            std::fs::write(dir.join(name), text).unwrap();
        }
        for name in ["sir", "sir_other", "sir_dated", "patch", "negative", "sir_anchor"] {
            let out = Command::new(&camdlc).env_clear().arg(dir.join(format!("{name}.camdl"))).output().unwrap();
            assert!(out.status.success(), "camdlc {name}: {}", String::from_utf8_lossy(&out.stderr));
            std::fs::write(dir.join(format!("{name}.ir.json")), out.stdout).unwrap();
        }
        Fixture { camdl, camdlc, dir }
    })
}

/// A fresh working directory holding a copy of the fixture files.
fn workdir() -> tempfile::TempDir {
    fn copy(from: &Path, to: &Path) {
        std::fs::create_dir_all(to).unwrap();
        for e in std::fs::read_dir(from).unwrap().flatten() {
            if e.path().is_dir() {
                copy(&e.path(), &to.join(e.file_name()));
            } else {
                std::fs::copy(e.path(), to.join(e.file_name())).unwrap();
            }
        }
    }
    let d = tempfile::tempdir().unwrap();
    copy(&fixture().dir, d.path());
    d
}

/// The reactive-policy fixture of the repository and its parameters.
fn reactive_model() -> String {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../../tests/fixtures/reactive/ir/reactive_sir_observed_threshold.ir.json")
        .to_string_lossy()
        .into_owned()
}

const REACTIVE_PARAMS: &[&str] = &[
    "--param", "beta=0.3", "--param", "gamma=0.1", "--param", "rho=0.2", "--param", "trigger_threshold=2", "--param", "sia_cov=0.7",
    "--param", "N0=1000", "--param", "I0=10",
];

/// `camdl` in `cwd`, with a cleared environment holding `PATH`, `TMPDIR`, a
/// private `HOME` and IR cache, the compiler under test, `CAMDL_STORE` and
/// `env`.
fn camdl(cwd: &Path, store: &Path, args: &[String], env: &[(&str, &str)]) -> Output {
    let fx = fixture();
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

/// `inv` as a command line.
fn argv(inv: &SimulateInvocation) -> Vec<String> {
    let SimulateInvocation { args, no_licm, no_ir_cache, verbosity: _, progress: _, no_progress: _ } = inv;
    let crate::args::SimulateArgs {
        model,
        to,
        from,
        init_state,
        allow_degenerate_rates,
        forcing_extrapolation: crate::args::ForcingExtrapolationArg { allow_forcing_extrapolation },
        model_overrides: crate::args::ModelOverrides { params, param, table },
        param_vec,
        scenarios,
        enable,
        disable,
        backend: crate::args::SimBackend { backend, dt, integrator },
        output_view: crate::args::OutputView { every, no_flows, columns },
        seed,
        seeds,
        replicates,
        draws,
        sweep,
        sweep_file,
        fit,
        n_draws,
        chains,
        exclude_chains,
        draws_out,
        output,
        stdout,
        emit_every,
        obs,
        obs_dir,
        obs_only,
        obs_only_dir,
        design_from,
        projected_out,
        dates,
        quantities_out,
        quantities,
        dry_run,
        store,
        parallel,
        force,
        label,
        event_log,
        format,
        tsv,
        reactive_log,
    } = args;
    let p = |p: &Path| p.to_string_lossy().into_owned();
    let mut v: Vec<String> = vec!["simulate".into(), p(model)];
    let mut opt = |flag: &str, value: Option<String>| {
        if let Some(x) = value {
            v.push(flag.into());
            v.push(x);
        }
    };
    opt("--to", to.clone());
    opt("--from", from.clone());
    opt("--init-state", init_state.clone());
    for f in params {
        opt("--params", Some(p(f)));
    }
    for o in param {
        opt("--param", Some(format!("{}={}", o.name, o.value)));
    }
    for t in table {
        opt("--table", Some(format!("{}={}", t.name, p(&t.path))));
    }
    for pv in param_vec {
        opt("--param-vec", Some(format!("{}={}", pv.prefix, pv.file)));
    }
    for s in scenarios {
        opt("--scenario", Some(s.clone()));
    }
    for e in enable {
        opt("--enable", Some(e.clone()));
    }
    for d in disable {
        opt("--disable", Some(d.clone()));
    }
    opt("--backend", backend.map(|b| b.as_str().to_string()));
    opt("--dt", dt.map(|x| x.to_string()));
    opt("--integrator", integrator.map(|i| serde_json::to_value(i).unwrap().as_str().unwrap().to_string()));
    opt("--output-every", every.map(|x| x.to_string()));
    if !columns.is_empty() {
        opt("--columns", Some(columns.join(",")));
    }
    match seeds {
        Some(crate::args::types::SeedSpec::List(l)) => {
            opt("--seeds", Some(l.iter().map(|s| s.to_string()).collect::<Vec<_>>().join(",")));
        }
        Some(crate::args::types::SeedSpec::Range { from, to }) => opt("--seeds", Some(format!("{from}:{to}"))),
        None => opt("--seed", Some(seed.to_string())),
    }
    opt("--replicates", replicates.map(|r| r.to_string()));
    opt("--draws", draws.clone());
    for s in sweep {
        let grid = match &s.grid {
            crate::args::types::Grid::List(v) => v.iter().map(|x| x.to_string()).collect::<Vec<_>>().join(","),
            crate::args::types::Grid::Linear { min, max, n } => format!("lin({min},{max},{n})"),
            crate::args::types::Grid::Log10 { min, max, n } => format!("log10({min},{max},{n})"),
        };
        opt("--sweep", Some(format!("{}={grid}", s.name)));
    }
    opt("--sweep-file", sweep_file.as_deref().map(p));
    opt("--fit", fit.as_deref().map(p));
    opt("-n", n_draws.map(|n| n.to_string()));
    opt("--chains", chains.clone());
    opt("--exclude-chains", exclude_chains.clone());
    opt("--draws-out", draws_out.as_deref().map(p));
    opt("--output", output.as_deref().map(p));
    for e in emit_every {
        opt("--emit-every", Some(e.clone()));
    }
    opt("--obs", obs.as_deref().map(p));
    opt("--obs-dir", obs_dir.as_deref().map(p));
    opt("--obs-only", obs_only.as_deref().map(p));
    opt("--obs-only-dir", obs_only_dir.as_deref().map(p));
    opt("--design-from", design_from.as_deref().map(p));
    opt("--projected-out", projected_out.as_deref().map(p));
    opt("--quantities-out", quantities_out.as_deref().map(p));
    opt("--quantities", quantities.as_deref().map(p));
    opt("--store", store.as_deref().map(p));
    opt("--parallel", parallel.map(|n| n.to_string()));
    opt("--label", label.clone());
    opt("--format", format.clone());
    opt("--reactive-log", reactive_log.as_deref().map(p));
    if let Some(e) = event_log {
        if e.as_os_str() == "auto" {
            v.push("--event-log".into());
        } else {
            v.push(format!("--event-log={}", p(e)));
        }
    }
    let mut flag = |set: bool, name: &str| {
        if set {
            v.push(name.into());
        }
    };
    flag(*allow_degenerate_rates, "--allow-degenerate-rates");
    flag(*allow_forcing_extrapolation, "--allow-forcing-extrapolation");
    flag(*no_flows, "--no-flows");
    flag(*stdout, "--stdout");
    flag(*dates, "--dates");
    flag(*dry_run, "--dry-run");
    flag(*force, "--force");
    flag(*tsv, "--tsv");
    flag(*no_licm, "--no-licm");
    flag(*no_ir_cache, "--no-ir-cache");
    v.extend(["--progress".to_string(), "none".to_string()]);
    v
}

/// Every result folder under `root`: its kind, run id, result files with
/// their digests, and upstream.
fn observe(root: &Path) -> Vec<ObservedResult> {
    fn walk(dir: &Path, out: &mut Vec<ObservedResult>) {
        let Ok(entries) = std::fs::read_dir(dir) else { return };
        let mut entries: Vec<_> = entries.flatten().collect();
        entries.sort_by_key(|e| e.file_name());
        for e in entries {
            let p = e.path();
            let name = e.file_name().to_string_lossy().into_owned();
            if !p.is_dir() || name.starts_with('.') {
                continue;
            }
            let ident = p.join("identity.json");
            if ident.is_file() {
                let id: serde_json::Value = serde_json::from_slice(&std::fs::read(&ident).unwrap()).unwrap();
                let run: serde_json::Value = serde_json::from_slice(&std::fs::read(p.join("run.json")).unwrap()).unwrap();
                let files = run["files"]
                    .as_object()
                    .unwrap()
                    .iter()
                    .filter(|(_, f)| f["role"] == "result")
                    .map(|(n, f)| (n.clone(), Key::from_hex(f["sha256"].as_str().unwrap()).unwrap()))
                    .collect();
                let upstream = match &id["parent"] {
                    serde_json::Value::Object(o) if o["relation"] == "upstream" => {
                        Some(Key::from_hex(o["run_id"].as_str().unwrap()).unwrap())
                    }
                    _ => None,
                };
                let kind = id["kind"].as_str().unwrap().to_string();
                out.push(ObservedResult {
                    content: ["compiled_model", "quantities", "data_table"].contains(&kind.as_str()),
                    kind,
                    run_id: Key::from_hex(id["run_id"].as_str().unwrap()).unwrap(),
                    files,
                    upstream,
                });
            }
            walk(&p, out);
        }
    }
    let mut out = Vec::new();
    walk(root, &mut out);
    out
}

/// Each file at `path`, or under it when it is a directory, by its path, with
/// the digest of its bytes.
fn digests(cwd: &Path, path: &Path, out: &mut BTreeMap<PathBuf, Key>) {
    let full = cwd.join(path);
    if full.is_dir() {
        let mut entries: Vec<_> = std::fs::read_dir(&full).unwrap().flatten().collect();
        entries.sort_by_key(|e| e.file_name());
        for e in entries {
            digests(cwd, &path.join(e.file_name()), out);
        }
    } else if let Ok(bytes) = std::fs::read(&full) {
        out.insert(path.to_path_buf(), Key::digest_bytes(&bytes));
    }
}

/// Runs `inv` in a fresh copy of the fixtures, storing into `store` or a
/// fresh store, and reads back what it stored and exported.
fn run(inv: &SimulateInvocation, store: Option<&Path>) -> store::Result<Observed> {
    let cwd = workdir();
    let fresh = tempfile::tempdir().unwrap();
    let root = store.map(Path::to_path_buf).unwrap_or_else(|| fresh.path().join("camdl-store"));
    let out = camdl(cwd.path(), &root, &argv(inv), &[]);
    if !out.status.success() {
        return Err(store::Error::Parse { what: format!("camdl {:?}", argv(inv)), why: String::from_utf8_lossy(&out.stderr).into_owned() });
    }
    let mut exports = BTreeMap::new();
    let a = &inv.args;
    for path in [&a.output, &a.draws_out, &a.obs_only_dir, &a.projected_out, &a.reactive_log, &a.quantities_out, &a.obs, &a.obs_dir]
        .into_iter()
        .flatten()
    {
        digests(cwd.path(), path, &mut exports);
    }
    if let Some(e) = a.event_log.as_ref().filter(|e| e.as_os_str() != "auto") {
        digests(cwd.path(), e, &mut exports);
    }
    if a.stdout {
        exports.insert(PathBuf::from("<stdout>"), Key::digest_bytes(&out.stdout));
    }
    Ok(Observed { results: observe(&root), exports, ..Observed::default() })
}

/// The harness of `simulate`'s class checks, from one base invocation.
struct Harness {
    base: SimulateInvocation,
}

/// The single cell: `simulate sir.ir.json --params p.toml`.
fn sir(extra: &[&str]) -> SimulateInvocation {
    let mut v = vec!["sir.ir.json", "--params", "p.toml"];
    v.extend_from_slice(extra);
    sim_from(&v)
}

impl Harness {
    /// The base each field is checked from: a fixture on which the field is
    /// live.
    fn for_field(name: &str) -> Harness {
        let base = match name.strip_prefix("args.").unwrap_or(name) {
            "init_state" => sir(&["--replicates", "2"]),
            "model_overrides.table" => sim_from(&["patch.ir.json", "--params", "patch.toml", "--table", "w=w1.tsv"]),
            "disable" => sir(&["--enable", "sia"]),
            "backend.integrator" => sir(&["--backend", "ode"]),
            "seed" | "to" | "from" | "event_log" | "stdout" | "reactive_log" | "output" | "model" | "param_vec" | "enable" => sir(&[]),
            "fit" => sim_from(&["sir.ir.json", "--draws", "draws_a.tsv", "--fit", "fit.toml"]),
            "emit_every" => sir(&["--obs", "obs.tsv"]),
            "quantities" => sim_from(&["sir.camdl", "--params", "p.toml", "--quantities-out", "q"]),
            "scenarios" => sir(&["--scenario", "with_sia"]),
            "replicates" => sir(&["--replicates", "2"]),
            "draws" => sir(&["--draws", "draws2.tsv"]),
            "n_draws" => sir(&["--draws", "prior", "-n", "2"]),
            "chains" => sir(&["--draws", "posterior", "--fit", "post", "--chains", "0"]),
            "exclude_chains" => sir(&["--draws", "posterior", "--fit", "post", "--exclude-chains", "1,2"]),
            "sweep" => sir(&["--sweep", "beta_a=0.3,0.4"]),
            "sweep_file" => sir(&["--sweep-file", "sweep2.tsv"]),
            "draws_out" => sir(&["--draws", "draws2.tsv"]),
            "design_from" => sir(&["--obs-only-dir", "ds"]),
            "projected_out" => sir(&["--design-from", "fit.toml"]),
            "dates" => sim_from(&["sir_dated.ir.json", "--params", "p.toml", "-o", "traj.tsv"]),
            "format" | "tsv" => sir(&["--event-log=el.tsv"]),
            "no_licm" => sim_from(&["sir.camdl", "--params", "p.toml"]),
            _ => sim_base(),
        };
        let base = if name == "args.reactive_log" {
            let mut v = vec![reactive_model()];
            v.extend(REACTIVE_PARAMS.iter().map(|s| s.to_string()));
            v.extend(["--enable".to_string(), "sia".to_string()]);
            sim_from(&v.iter().map(String::as_str).collect::<Vec<_>>())
        } else {
            base
        };
        Harness { base }
    }
}

impl InputHarness for Harness {
    type Invocation = SimulateInvocation;

    fn base(&self) -> SimulateInvocation {
        self.base.clone()
    }

    fn varied(&self, field: &FieldInfo) -> SimulateInvocation {
        let mut inv = self.base.clone();
        let a = &mut inv.args;
        let parse = |s: &str| s.parse().unwrap_or_else(|_| panic!("{s} does not parse"));
        match (field.name, field.class) {
            ("args.disable", _) => {
                a.enable.clear();
                a.disable = vec!["sia".into()];
            }
            ("args.scenarios", _) => a.scenarios.push("fast".into()),
            ("args.seeds", _) => a.seeds = Some(parse("1,2,3")),
            ("args.replicates", _) => a.replicates = Some(3),
            ("args.draws", _) => a.draws = Some("draws3.tsv".into()),
            ("args.n_draws", _) => a.n_draws = Some(3),
            ("args.chains", _) => a.chains = Some("0,1".into()),
            ("args.exclude_chains", _) => a.exclude_chains = Some("2".into()),
            ("args.sweep", _) => a.sweep = vec!["beta_a=0.3,0.4,0.5".parse().unwrap()],
            ("args.sweep_file", _) => a.sweep_file = Some("sweep3.tsv".into()),
            ("args.obs", _) => a.obs = Some("obs.tsv".into()),
            ("args.obs_dir", _) => a.obs_dir = Some("od".into()),
            ("args.obs_only", _) => a.obs_only = Some("obs.tsv".into()),
            ("args.obs_only_dir", _) => a.obs_only_dir = Some("od".into()),
            ("args.event_log", _) => a.event_log = Some("auto".into()),
            ("args.quantities_out", _) => a.quantities_out = Some("q".into()),
            ("args.output", _) => a.output = Some("traj.tsv".into()),
            ("args.stdout", _) => a.stdout = true,
            ("args.draws_out", _) => a.draws_out = Some("d.tsv".into()),
            ("args.design_from", _) => a.design_from = Some("fit.toml".into()),
            ("args.projected_out", _) => a.projected_out = Some("proj".into()),
            ("args.reactive_log", _) => a.reactive_log = Some("rl.tsv".into()),
            ("args.dates", _) => a.dates = true,
            ("args.format", _) => a.format = Some("tsv".into()),
            ("args.tsv", _) => a.tsv = true,
            ("args.init_state", _) => a.init_state = Some("state.tsv".into()),
            (name, Class::Identity | Class::Render) => return self.base.varied(name),
            (name, class) => panic!("no second value for {name} ({class:?})"),
        }
        inv
    }

    fn exercised(&self, field: &FieldInfo) -> (SimulateInvocation, SimulateInvocation) {
        match field.name.strip_prefix("args.") {
            Some(f) => {
                let (x, y) = <crate::args::SimulateArgs as Inputs>::exercised(f);
                let (mut a, mut b) = (self.base.clone(), self.base.clone());
                a.args = x;
                b.args = y;
                (a, b)
            }
            None => <SimulateInvocation as Inputs>::exercised(field.name),
        }
    }

    /// Every execution fixture uses its field: two cells run on a pool of
    /// four threads, and `--no-ir-cache` is read on a model given as source.
    fn shows_use(&self, _field: &FieldInfo, _seen: &Observed) -> bool {
        true
    }

    fn run(&self, invocation: &SimulateInvocation, store: Option<&Path>) -> store::Result<Observed> {
        run(invocation, store)
    }
}

/// Every declared input of `camdl simulate` keeps the promise its class makes.
#[test]
fn every_simulate_input_keeps_its_class() {
    let fields: Vec<FieldInfo> = SimulateInvocation::fields().to_vec();
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

/// Every argument clap accepts for `simulate`, and every global option, is a
/// declared input; and every declared input is one clap accepts.
#[test]
fn every_simulate_argument_is_declared() {
    use clap::CommandFactory;
    let cmd = crate::Cli::command();
    let sub = cmd.find_subcommand("simulate").expect("the simulate subcommand");
    let mut accepted: std::collections::BTreeSet<String> =
        sub.get_arguments().map(|a| a.get_id().to_string()).filter(|id| id != "help").collect();
    accepted.extend(cmd.get_arguments().map(|a| a.get_id().to_string()).filter(|id| id != "help" && id != "version"));
    let declared: std::collections::BTreeSet<String> =
        SimulateInvocation::fields().iter().map(|f| f.name.rsplit('.').next().unwrap().to_string()).collect();
    assert_eq!(accepted, declared);
}

/// Each directory's entries, with the 12 hex digits of each suffix replaced
/// by `*`, so the shape is compared and not the keys.
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

/// The trees of "Layouts, kind by kind" and the worked example: one cell
/// with its three by-products inside it, under its kind; three replicates of
/// the seed typed, `seed_7`, `seed_7-r1`, `seed_7-r2`; and two scenarios
/// that differ only in their interventions under one parameters directory,
/// each seed a cell below its scenario.
#[test]
fn a_run_makes_the_tree_of_the_proposal() {
    let build = format!("{}-*", store::path_label(crate::version::VERSION_SHORT));
    let tree = |line: &[&str]| -> Vec<String> {
        let cwd = workdir();
        let store = cwd.path().join("camdl-store");
        let out = camdl(cwd.path(), &store, &argv(&sim_from(line)), &[]);
        assert!(out.status.success(), "{}", String::from_utf8_lossy(&out.stderr));
        let mut got = Vec::new();
        shape(&store, 0, &mut got);
        got
    };
    let one = tree(&["sir.ir.json", "--params", "p.toml", "--obs", "obs.tsv", "--event-log", "--quantities-out", "q"]);
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
        "  quantities",
        "    sir-*",
        "      identity.json",
        "      quantities.json",
        "      run.json",
        "  sim",
        "    sir-*",
        "      chain_binomial-dt1-*",
        "        base-*",
        "          _none_-*",
        "            seed_1-*",
        "              event_log",
        "                event_log-*",
        "                  event_log.tsv",
        "                  identity.json",
        "                  run.json",
        "              identity.json",
        "              obs",
        "                cases-*",
        "                  cases.tsv",
        "                  identity.json",
        "                  run.json",
        "              provenance.json",
        "              run.json",
        "              sim_quantities",
        "                quantities-*",
        "                  identity.json",
        "                  quantities.json",
        "                  run.json",
        "              traj.tsv",
        "store.json",
    ];
    assert_eq!(one, want, "the store's tree:\n{}", one.join("\n"));

    let replicates = tree(&["sir.ir.json", "--params", "p.toml", "--seed", "7", "--replicates", "3"]);
    let cells: Vec<&str> = replicates.iter().map(String::as_str).filter(|l| l.trim_start().starts_with("seed_")).collect();
    assert_eq!(cells, ["            seed_7-*", "            seed_7-r1-*", "            seed_7-r2-*"]);

    let grid = tree(&["sir.ir.json", "--params", "p.toml", "--scenario", "with_sia", "--scenario", "slow", "--seeds", "1,2"]);
    let i = grid.iter().position(|l| l.trim_start().starts_with("base-")).expect("a parameters directory");
    assert_eq!(grid.iter().filter(|l| l.trim_start().starts_with("base-")).count(), 1, "one parameters directory:\n{}", grid.join("\n"));
    assert_eq!(
        &grid[i + 1..i + 8],
        [
            "          slow-*",
            "            seed_1-*",
            "              identity.json",
            "              provenance.json",
            "              run.json",
            "              traj.tsv",
            "            seed_2-*",
        ],
        "the store's tree:\n{}",
        grid.join("\n")
    );
    assert!(grid.iter().any(|l| l == "          with_sia-*"), "{}", grid.join("\n"));
}

/// The trajectories (and the observations of one fixture) three guide models
/// simulate are byte for byte those the old store held before `simulate`
/// moved: one cell, three replicates with observations, and a grid of two
/// scenarios by two seeds. The digests are the old store's `traj.tsv` and
/// `obs/<…>/in_bed.tsv` files at the commit before the move.
#[test]
fn the_guide_models_simulate_what_they_simulated_before_the_move() {
    let guide = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../../docs/guide");
    let g = |p: &str| guide.join(p).to_string_lossy().into_owned();
    let cases: [(&str, Vec<String>, &[(&str, &str)]); 3] = [
        (
            "beta = 1.6\ngamma = 0.4\nN0 = 763\nI0 = 3\n",
            vec![g("first-fit/sir-beta.camdl"), "--seed".into(), "7".into()],
            &[("seed_7/traj.tsv", "8b9bf9fd038674a7")],
        ),
        (
            "R0 = 3.75\ngamma = 0.4\nk = 20\nN0 = 763\nI0 = 3\n",
            vec![g("first-fit/sir-negbin.camdl"), "--seed".into(), "7".into(), "--replicates".into(), "3".into(), "--obs-dir".into(), "od".into()],
            &[
                ("seed_7/traj.tsv", "b4f500bdb05bc1dd"),
                ("seed_7-r1/traj.tsv", "9a85cc62c8c39fab"),
                ("seed_7-r2/traj.tsv", "32b323df4960a829"),
                ("seed_7/obs/in_bed/in_bed.tsv", "54d806e5657c4c1f"),
                ("seed_7-r1/obs/in_bed/in_bed.tsv", "6cd26cc784dbf985"),
                ("seed_7-r2/obs/in_bed/in_bed.tsv", "b99a367ff57be36d"),
            ],
        ),
        (
            "",
            vec![
                g("dimensions/sis_sex_gonorrhea.camdl"),
                "--scenario".into(),
                "baseline".into(),
                "--scenario".into(),
                "asymmetric".into(),
                "--seeds".into(),
                "1,2".into(),
            ],
            &[
                ("baseline/seed_1/traj.tsv", "4e7ca82f3621e39c"),
                ("baseline/seed_2/traj.tsv", "04e9cab1d8c4db04"),
                ("asymmetric/seed_1/traj.tsv", "4c5d552fa6be6bef"),
                ("asymmetric/seed_2/traj.tsv", "ac185532a92622d9"),
            ],
        ),
    ];
    for (params, line, pins) in cases {
        let cwd = workdir();
        let store = cwd.path().join("camdl-store");
        let mut args: Vec<String> = vec!["simulate".into()];
        args.extend(line);
        if !params.is_empty() {
            std::fs::write(cwd.path().join("params.toml"), params).unwrap();
            args.extend(["--params".into(), "params.toml".into()]);
        }
        args.extend(["--progress".into(), "none".into()]);
        let out = camdl(cwd.path(), &store, &args, &[]);
        assert!(out.status.success(), "{}", String::from_utf8_lossy(&out.stderr));
        // Every stored file, by its path below the scenario directory with
        // the suffixes removed, and the first 16 hex digits of its digest.
        let mut found: BTreeMap<String, String> = BTreeMap::new();
        let mut stack = vec![store.clone()];
        while let Some(d) = stack.pop() {
            for e in std::fs::read_dir(&d).unwrap().flatten() {
                let p = e.path();
                if p.is_dir() {
                    stack.push(p);
                    continue;
                }
                let name = e.file_name().to_string_lossy().into_owned();
                if !name.ends_with(".tsv") || !p.to_string_lossy().contains("/sim/") {
                    continue;
                }
                let rel: Vec<String> = p
                    .strip_prefix(&store)
                    .unwrap()
                    .components()
                    .map(|c| {
                        let c = c.as_os_str().to_string_lossy().into_owned();
                        match c.rsplit_once('-') {
                            Some((h, t)) if t.len() == 12 && t.chars().all(|x| x.is_ascii_hexdigit()) => h.to_string(),
                            _ => c,
                        }
                    })
                    .collect();
                found.insert(rel.join("/"), Key::digest_bytes(&std::fs::read(&p).unwrap()).to_hex()[..16].to_string());
            }
        }
        for (path, digest) in pins {
            let got = found
                .iter()
                .find(|(k, _)| k.ends_with(&format!("/{path}")))
                .map(|(_, v)| v.clone())
                .unwrap_or_else(|| panic!("{path} is not stored: {found:?}"));
            assert_eq!(got, *digest, "{path}");
        }
    }
}

/// The ids and files of every result `extra` stores in `store`, and stderr.
fn run_in(store: &Path, cwd: &Path, line: &[&str]) -> (Vec<ObservedResult>, String, Output) {
    let out = camdl(cwd, store, &argv(&sim_from(line)), &[]);
    let stderr = String::from_utf8_lossy(&out.stderr).into_owned();
    (observe(store), stderr, out)
}

/// Asking for observations after a cell is stored simulates it again, adds
/// the by-product, and leaves the cell's id and files unchanged; a rerun is a
/// cache hit that exports the stored observations.
#[test]
fn observations_are_added_later_and_then_reused() {
    let cwd = workdir();
    let store = cwd.path().join("camdl-store");
    let (first, err, out) = run_in(&store, cwd.path(), &["sir.ir.json", "--params", "p.toml"]);
    assert!(out.status.success(), "{err}");
    let (second, err, out) = run_in(&store, cwd.path(), &["sir.ir.json", "--params", "p.toml", "--obs", "obs.tsv"]);
    assert!(out.status.success(), "{err}");
    let sims = |r: &[ObservedResult]| r.iter().filter(|x| x.kind == "sim").map(|x| (x.run_id, x.files.clone())).collect::<Vec<_>>();
    assert_eq!(sims(&first), sims(&second), "the cell's id and files are unchanged");
    assert!(second.iter().any(|r| r.kind == "obs"), "the observations are stored");
    let exported = std::fs::read(cwd.path().join("obs.tsv")).unwrap();
    std::fs::remove_file(cwd.path().join("obs.tsv")).unwrap();
    let (third, err, out) = run_in(&store, cwd.path(), &["sir.ir.json", "--params", "p.toml", "--obs", "obs.tsv"]);
    assert!(out.status.success(), "{err}");
    assert!(err.contains("cached"), "the rerun is a cache hit: {err}");
    assert_eq!(second.len(), third.len());
    assert_eq!(std::fs::read(cwd.path().join("obs.tsv")).unwrap(), exported, "the export is rendered from the stored files");
}

/// `--label` names the invocation; `camdl cat @L` renders the combined table
/// `-o` writes, with the coordinate columns for one cell as for many; a
/// rerun under another label reuses every cell; `--stdout` writes the same
/// table.
#[test]
fn a_label_names_the_invocation_and_cat_renders_its_table() {
    let cwd = workdir();
    let store = cwd.path().join("camdl-store");
    for (label, line) in [
        ("one", vec!["sir.ir.json", "--params", "p.toml", "-o", "one.tsv"]),
        ("grid", vec!["sir.ir.json", "--params", "p.toml", "--scenario", "fast", "--scenario", "with_sia", "--seeds", "1,2", "-o", "grid.tsv"]),
    ] {
        let mut l = line.clone();
        l.extend(["--label", label]);
        let (_, err, out) = run_in(&store, cwd.path(), &l);
        assert!(out.status.success(), "{err}");
        let cat = camdl(cwd.path(), &store, &["cat".into(), format!("@{label}")], &[]);
        assert!(cat.status.success(), "{}", String::from_utf8_lossy(&cat.stderr));
        let file = std::fs::read(cwd.path().join(format!("{label}.tsv"))).unwrap();
        assert_eq!(cat.stdout, file, "cat @{label} renders the table -o wrote");
        let header = String::from_utf8_lossy(&file).lines().nth(1).unwrap().to_string();
        assert!(header.starts_with("replicate\tscenario\tseed\tt\t"), "{header}");
    }
    let (_, err, out) = run_in(
        &store,
        cwd.path(),
        &["sir.ir.json", "--params", "p.toml", "--scenario", "fast", "--scenario", "with_sia", "--seeds", "1,2", "--label", "again"],
    );
    assert!(out.status.success(), "{err}");
    assert!(err.contains("4 of 4 cells read from the store"), "{err}");
    let (_, err, out) = run_in(&store, cwd.path(), &["sir.ir.json", "--params", "p.toml", "--stdout"]);
    assert!(out.status.success(), "{err}");
    assert_eq!(out.stdout, std::fs::read(cwd.path().join("one.tsv")).unwrap(), "--stdout writes the table -o writes");
}

/// A cell whose engine stops with an error is stored with the failure; the
/// other cells of the batch are computed; the command lists the failure,
/// exports nothing and exits non-zero; a rerun reuses the stored failure.
#[test]
fn a_cell_the_engine_refuses_is_stored_with_its_failure_and_the_batch_carries_on() {
    let cwd = workdir();
    let store = cwd.path().join("camdl-store");
    let line = ["negative.ir.json", "--sweep", "beta=0.2,0.5", "-o", "t.tsv"];
    let (results, err, out) = run_in(&store, cwd.path(), &line);
    assert_eq!(out.status.code(), Some(1), "{err}");
    assert!(err.contains("the engine stopped"), "{err}");
    let sims: Vec<&ObservedResult> = results.iter().filter(|r| r.kind == "sim").collect();
    assert_eq!(sims.len(), 2, "both cells are stored: {results:#?}");
    assert_eq!(sims.iter().filter(|r| r.files.contains_key(super::FAILURE_FILE)).count(), 1, "one holds its failure");
    assert_eq!(sims.iter().filter(|r| r.files.contains_key(super::TRAJ_FILE)).count(), 1, "the other its trajectory");
    assert!(!cwd.path().join("t.tsv").exists(), "a table over a stored failure is not written");
    let (again, err, out) = run_in(&store, cwd.path(), &line);
    assert_eq!(out.status.code(), Some(1), "{err}");
    assert!(err.contains("2 of 2 cells read from the store"), "the rerun reuses both cells: {err}");
    assert_eq!(again, results);
}

/// `--dry-run` asks the store which cells it holds and writes nothing.
#[test]
fn a_dry_run_reports_what_is_stored_and_writes_nothing() {
    let cwd = workdir();
    let store = cwd.path().join("camdl-store");
    let (_, err, out) = run_in(&store, cwd.path(), &["sir.ir.json", "--params", "p.toml", "--seeds", "1,2", "--dry-run"]);
    assert!(out.status.success(), "{err}");
    assert!(!store.exists(), "a dry run writes no store");
    let (_, err, out) = run_in(&store, cwd.path(), &["sir.ir.json", "--params", "p.toml", "--seeds", "1"]);
    assert!(out.status.success(), "{err}");
    let before = observe(&store);
    let (_, err, out) = run_in(&store, cwd.path(), &["sir.ir.json", "--params", "p.toml", "--seeds", "1,2", "--dry-run"]);
    assert!(out.status.success(), "{err}");
    assert!(err.contains("1 cache hits"), "{err}");
    assert_eq!(observe(&store), before, "nothing is written");
}

/// `--no-licm` and `CAMDL_NO_LICM` change the compiled model a cell stores,
/// and so the cell's id; an engine switch changes every id through the
/// ambient identity and leaves the stored trajectory alone.
#[test]
fn the_compile_settings_and_the_engine_switches_key_the_cell() {
    let ids = |no_licm: bool, env: &[(&str, &str)]| -> BTreeMap<String, Vec<(Key, BTreeMap<String, Key>)>> {
        let cwd = workdir();
        let store = cwd.path().join("camdl-store");
        let mut inv = sim_from(&["sir.camdl", "--params", "p.toml"]);
        inv.no_licm = no_licm;
        let out = camdl(cwd.path(), &store, &argv(&inv), env);
        assert!(out.status.success(), "{}", String::from_utf8_lossy(&out.stderr));
        let mut m: BTreeMap<String, Vec<(Key, BTreeMap<String, Key>)>> = BTreeMap::new();
        for r in observe(&store) {
            m.entry(r.kind).or_default().push((r.run_id, r.files));
        }
        m
    };
    let base = ids(false, &[]);
    for (what, other) in [("--no-licm", ids(true, &[])), ("CAMDL_NO_LICM", ids(false, &[("CAMDL_NO_LICM", "1")]))] {
        assert_ne!(other["compiled_model"][0].1, base["compiled_model"][0].1, "{what} changes the stored model's bytes");
        assert_ne!(other["sim"][0].0, base["sim"][0].0, "{what} changes the cell's id");
    }
    for var in ["CAMDL_EVAL_UNRESOLVED", "CAMDL_NO_BINDING_CACHE", "CAMDL_EXPERIMENTAL_DIFFERENTIAL_TEST"] {
        let other = ids(false, &[(var, "1")]);
        assert_ne!(other["sim"][0].0, base["sim"][0].0, "{var} changes the cell's id");
        assert_eq!(other["sim"][0].1, base["sim"][0].1, "{var} stores the same trajectory");
    }
}

/// `--event-log` on a model with no `#[lineage]` transition is refused before
/// any cell runs and stores nothing: the refusal is about what was asked for,
/// not about the cell, so the cell is not stored as a failure, and a run
/// without the flag computes it.
#[test]
fn an_event_log_the_model_cannot_record_is_refused_and_stores_nothing() {
    let cwd = workdir();
    let store = cwd.path().join("camdl-store");
    let line = ["patch.ir.json", "--params", "patch.toml", "--table", "w=w1.tsv"];
    let mut with = line.to_vec();
    with.push("--event-log");
    let (results, err, out) = run_in(&store, cwd.path(), &with);
    assert!(!out.status.success(), "{err}");
    assert!(err.contains("#[lineage]"), "{err}");
    assert!(!results.iter().any(|r| r.kind == "sim"), "no cell is stored: {results:#?}");
    let (results, err, out) = run_in(&store, cwd.path(), &line);
    assert!(out.status.success(), "{err}");
    let sims: Vec<&ObservedResult> = results.iter().filter(|r| r.kind == "sim").collect();
    assert_eq!(sims.len(), 1);
    assert!(sims[0].files.contains_key(super::TRAJ_FILE), "the cell holds its trajectory: {:?}", sims[0].files);
}

/// A by-product that fails is stored with its own failure and the cell keeps
/// its trajectory: a quantity read at an observation anchor has no observed
/// data to anchor to in a forward simulation.
#[test]
fn a_by_product_that_fails_stores_its_failure_and_the_cell_keeps_its_trajectory() {
    let cwd = workdir();
    let store = cwd.path().join("camdl-store");
    let (results, err, out) = run_in(&store, cwd.path(), &["sir_anchor.ir.json", "--params", "p.toml", "--quantities-out", "q"]);
    assert_eq!(out.status.code(), Some(1), "{err}");
    assert!(err.contains("the quantities stopped"), "{err}");
    let sim = results.iter().find(|r| r.kind == "sim").expect("the cell is stored");
    assert!(sim.files.contains_key(super::TRAJ_FILE) && !sim.files.contains_key(super::FAILURE_FILE), "{:?}", sim.files);
    let q = results.iter().find(|r| r.kind == "sim_quantities").expect("the quantities are stored");
    assert!(q.files.contains_key(super::FAILURE_FILE), "{:?}", q.files);
    assert!(!cwd.path().join("q").exists(), "nothing is exported");
}

/// A combined table over a cell that stored a failure is refused, naming the
/// cell and its `failure.toml`.
#[test]
fn a_table_over_a_stored_failure_names_its_failure_file() {
    let cwd = workdir();
    let store = cwd.path().join("camdl-store");
    let (_, err, out) = run_in(&store, cwd.path(), &["negative.ir.json", "--sweep", "beta=0.2,0.5", "--label", "neg"]);
    assert_eq!(out.status.code(), Some(1), "{err}");
    let cat = camdl(cwd.path(), &store, &["cat".into(), "@neg".into()], &[]);
    let err = String::from_utf8_lossy(&cat.stderr);
    assert!(!cat.status.success(), "{err}");
    assert!(err.contains("stored a failure") && err.contains("failure.toml"), "{err}");
}

/// Two scenarios with the same effect are one stored cell, and the run says
/// one cell; a run with no scenario is labelled `(none)` whatever `--param`
/// it carries.
#[test]
fn the_banner_counts_stored_cells_and_the_implicit_arm_is_labelled_none() {
    let cwd = workdir();
    let store = cwd.path().join("camdl-store");
    let (results, err, out) = run_in(&store, cwd.path(), &["sir.ir.json", "--params", "p.toml", "--scenario", "with_sia", "--scenario", "with_sia_too"]);
    assert!(out.status.success(), "{err}");
    assert_eq!(results.iter().filter(|r| r.kind == "sim").count(), 1);
    assert!(!err.contains("2 cells"), "{err}");
    let cwd = workdir();
    let store = cwd.path().join("camdl-store");
    let out = camdl(cwd.path(), &store, &argv(&sir(&["--param", "beta_a=0.45"])), &[]);
    assert!(out.status.success(), "{}", String::from_utf8_lossy(&out.stderr));
    let mut tree = Vec::new();
    shape(&store, 0, &mut tree);
    assert!(tree.iter().any(|l| l.trim() == "_none_-*"), "{}", tree.join("\n"));
}

/// An export path in a directory that does not exist is refused before any
/// cell runs, and nothing is stored.
#[test]
fn an_export_into_a_missing_directory_is_refused_before_anything_runs() {
    let cwd = workdir();
    let store = cwd.path().join("camdl-store");
    for flag in ["-o", "--obs", "--reactive-log"] {
        let (results, err, out) = run_in(&store, cwd.path(), &["sir.ir.json", "--params", "p.toml", flag, "missing/out.tsv"]);
        assert!(!out.status.success(), "{flag}: {err}");
        assert!(err.contains("missing") && err.contains("does not exist"), "{flag}: {err}");
        assert!(!results.iter().any(|r| r.kind == "sim"), "{flag}: nothing is stored");
    }
}
