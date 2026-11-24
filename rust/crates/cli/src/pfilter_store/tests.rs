//! `camdl pfilter` on the run store, checked by running the release binary:
//! every declared input against its class (`store::check_input_classes`), the
//! directory tree a run makes, reuse and export, labels, and the
//! log-likelihoods of the committed guide models against the values the old
//! store held.
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

use crate::args::types::DataSpec;
use crate::args::PfilterInvocation;
use crate::input_fixtures::{pfilter_base, pfilter_from};

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

/// A fit config over `sir.ir.json`'s `cases` stream; `extra` lands before
/// `[data.observations]`.
fn fit_toml(extra: &str) -> String {
    format!(
        "[model]\ncamdl = \"sir.ir.json\"\n{extra}[data.observations]\ncases = \"cases.tsv\"\n\
         [estimate]\nbeta_a = {{ bounds = [0.05, 2.0], start = 0.4 }}\n\
         [fixed]\nbeta_b = 0.2\ngamma = 0.2\nrho = 0.5\n\
         [method]\nalgorithm = \"if2\"\nbackend = \"chain_binomial\"\nchains = 1\nparticles = 20\niterations = 2\ncooling = 0.9\n",
    )
}

fn fixture_files() -> Vec<(&'static str, String)> {
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
        ("both.tsv", (1..=20).fold("time\tcases\tprevalent\n".to_string(), |s, t| s + &format!("{t}\t{}\t{}\n", 3 + t, 10 + 2 * t))),
        ("patch_cases.tsv", (1..=10).fold("time\tcases\n".to_string(), |s, t| s + &format!("{t}\t{}\n", 2 + t))),
        ("fit_plain.toml", fit_toml("")),
        ("fit_adr.toml", fit_toml("[config]\nallow_degenerate_rates = true\n")),
        ("fit_holdout.toml", fit_toml("[data]\nholdout_after = 15.0\n")),
        ("design2.tsv", "beta_a\n0.4\n0.5\n".into()),
        ("design3.tsv", "beta_a\n0.4\n0.5\n0.6\n".into()),
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
        let dir = crate::test_support::unique_temp_dir("pfilter_store_fixture");
        std::fs::create_dir_all(&dir).unwrap();
        for (name, text) in fixture_files() {
            std::fs::write(dir.join(name), text).unwrap();
        }
        for name in ["sir.camdl", "sir_other.camdl", "patch.camdl"] {
            let out = Command::new(&camdlc).env_clear().arg(dir.join(name)).output().unwrap();
            assert!(out.status.success(), "camdlc {name}: {}", String::from_utf8_lossy(&out.stderr));
            std::fs::write(dir.join(name).with_extension("ir.json"), out.stdout).unwrap();
        }
        Fixture { camdl, camdlc, dir }
    })
}

/// A fresh working directory holding a copy of the fixture files.
fn workdir() -> tempfile::TempDir {
    let d = tempfile::tempdir().unwrap();
    for e in std::fs::read_dir(&fixture().dir).unwrap().flatten() {
        std::fs::copy(e.path(), d.path().join(e.file_name())).unwrap();
    }
    d
}

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
fn argv(inv: &PfilterInvocation) -> Vec<String> {
    let PfilterInvocation { args, no_licm, no_ir_cache, verbosity: _, progress: _, no_progress: _ } = inv;
    let crate::args::PfilterArgs {
        model,
        model_overrides: crate::args::ModelOverrides { params, param, table },
        scenario: crate::args::ScenarioArgs { scenario, enable, disable },
        particles,
        inference:
            crate::args::PfilterCore {
                dt,
                seed,
                parallel,
                allow_degenerate_rates,
                forcing_extrapolation: crate::args::ForcingExtrapolationArg { allow_forcing_extrapolation },
                time_format,
                pf_max_substeps,
                binomial,
            },
        stream: crate::args::StreamSelection { obs },
        data,
        fit,
        replicates,
        sweep_file,
        trace,
        pf_health,
        output,
        save_final_state,
        save_paths,
        n_paths,
        save_filtering,
        save_filtered_streams,
        save_prequential,
        score_from,
        no_save_samples,
        no_store,
        force,
        label,
        store,
        manifest,
        link_root,
    } = args;
    let p = |p: &Path| p.to_string_lossy().into_owned();
    let mut v: Vec<String> = vec!["pfilter".into(), p(model)];
    let mut opt = |flag: &str, value: Option<String>| {
        if let Some(x) = value {
            v.push(flag.into());
            v.push(x);
        }
    };
    for f in params {
        opt("--params", Some(p(f)));
    }
    for o in param {
        opt("--param", Some(format!("{}={}", o.name, o.value)));
    }
    for t in table {
        opt("--table", Some(format!("{}={}", t.name, p(&t.path))));
    }
    opt("--scenario", scenario.clone());
    for e in enable {
        opt("--enable", Some(e.clone()));
    }
    for d in disable {
        opt("--disable", Some(d.clone()));
    }
    opt("--particles", Some(particles.to_string()));
    opt("--dt", dt.map(|x| x.to_string()));
    opt("--seed", Some(seed.to_string()));
    opt("--parallel", Some(parallel.to_string()));
    opt("--time-format", Some(serde_json::to_value(time_format).unwrap().as_str().unwrap().to_string()));
    opt("--pf-max-substeps", pf_max_substeps.map(|x| x.to_string()));
    opt("--binomial", binomial.map(|b| serde_json::to_value(b).unwrap().as_str().unwrap().to_string()));
    opt("--obs", obs.clone());
    for d in data {
        opt("--data", Some(match d {
            DataSpec::Single(path) => p(path),
            DataSpec::Named { name, path } => format!("{name}={}", p(path)),
        }));
    }
    opt("--fit", fit.as_deref().map(p));
    opt("--replicates", Some(replicates.to_string()));
    opt("--sweep-file", sweep_file.as_deref().map(p));
    opt("--trace", trace.clone());
    opt("--pf-health", pf_health.clone());
    opt("--output", output.as_deref().map(p));
    opt("--save-final-state", save_final_state.as_deref().map(p));
    opt("--save-paths", save_paths.as_deref().map(p));
    opt("--n-paths", Some(n_paths.to_string()));
    opt("--save-filtering", save_filtering.as_deref().map(p));
    opt("--save-filtered-streams", save_filtered_streams.as_deref().map(p));
    opt("--save-prequential", save_prequential.clone());
    opt("--score-from", score_from.clone());
    opt("--manifest", manifest.as_deref().map(p));
    opt("--link-root", link_root.as_deref().map(p));
    opt("--label", label.clone());
    opt("--store", store.as_deref().map(p));
    let mut flag = |set: bool, name: &str| {
        if set {
            v.push(name.into());
        }
    };
    flag(*allow_degenerate_rates, "--allow-degenerate-rates");
    flag(*allow_forcing_extrapolation, "--allow-forcing-extrapolation");
    flag(*no_save_samples, "--no-save-samples");
    flag(*no_store, "--no-store");
    flag(*force, "--force");
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
                    content: [crate::camdl_store::CompiledModel::KIND, crate::camdl_store::DataTable::KIND].contains(&kind.as_str()),
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

/// Every result folder's kind and recorded status.
fn walk_records(root: &Path) -> Vec<(String, String)> {
    let mut out = Vec::new();
    let mut stack = vec![root.to_path_buf()];
    while let Some(d) = stack.pop() {
        let Ok(entries) = std::fs::read_dir(&d) else { continue };
        for e in entries.flatten() {
            let p = e.path();
            if !p.is_dir() || e.file_name().to_string_lossy().starts_with('.') {
                continue;
            }
            if p.join("identity.json").is_file() {
                let id: serde_json::Value = serde_json::from_slice(&std::fs::read(p.join("identity.json")).unwrap()).unwrap();
                let run: serde_json::Value = serde_json::from_slice(&std::fs::read(p.join("run.json")).unwrap()).unwrap();
                out.push((id["kind"].as_str().unwrap().to_string(), run["status"].as_str().unwrap().to_string()));
            }
            stack.push(p);
        }
    }
    out
}

/// Runs `inv` in a fresh copy of the fixtures, storing into `store` or a
/// fresh store, and reads back what it stored and exported.
fn run(inv: &PfilterInvocation, store: Option<&Path>) -> store::Result<Observed> {
    let cwd = workdir();
    let fresh = tempfile::tempdir().unwrap();
    let root = store.map(Path::to_path_buf).unwrap_or_else(|| fresh.path().join("camdl-store"));
    let out = camdl(cwd.path(), &root, &argv(inv), &[("RUST_LOG", USE_LOG)]);
    if !out.status.success() {
        return Err(store::Error::Parse {
            what: format!("camdl {:?}", argv(inv)),
            why: String::from_utf8_lossy(&out.stderr).into_owned(),
        });
    }
    // `-o` is rendered from the stored result, not copied from a stored
    // file, so no export is named in `copies`.
    let mut exports = BTreeMap::new();
    let a = &inv.args;
    if let Some(path) = a.output.as_ref().map(|p| p.to_string_lossy().into_owned()) {
        let bytes = std::fs::read(cwd.path().join(&path)).unwrap_or_default();
        exports.insert(PathBuf::from(path), Key::digest_bytes(&bytes));
    }
    if let Some(path) = &a.manifest {
        let bytes = std::fs::read(cwd.path().join(path)).unwrap_or_default();
        exports.insert(path.clone(), Key::digest_bytes(&bytes));
    }
    let mut copies = BTreeMap::new();
    if let Some(links) = &a.link_root {
        crate::test_support::link_exports(cwd.path(), links, &mut exports, &mut copies);
    }
    let mut effects = crate::test_support::recorded_execution(&root, crate::pfilter_store::Pfilter::KIND);
    effects.extend(used(&String::from_utf8_lossy(&out.stderr), &cwd.path().join("ir-cache")));
    Ok(Observed { results: observe(&root), exports, copies, effects })
}

/// The log filter under which a run reports what its execution inputs did:
/// the threads it filtered on, the budget it handed the engine, and the
/// `--time-format` the data loader read.
const USE_LOG: &str = "camdl::pfilter=debug,camdl::caltime_load=debug";

/// What a run's execution inputs did, each under `used.<field>`: from its
/// debug log (`USE_LOG`), the threads it filtered on, the budget it handed the
/// engine and the `--time-format` its data loader read; and from the IR cache
/// directory, whether the compile bypassed the cache (`--no-ir-cache`). An
/// effect the log reports twice with different values is recorded as both,
/// so it matches no single value.
fn used(stderr: &str, ir_cache: &Path) -> BTreeMap<String, String> {
    let mut seen: BTreeMap<String, std::collections::BTreeSet<String>> = BTreeMap::new();
    let mut note = |field: &str, value: &str| {
        seen.entry(format!("used.{field}")).or_default().insert(value.to_string());
    };
    for line in stderr.lines() {
        let after = |marker: &str| line.split_once(marker).and_then(|(_, rest)| rest.split_whitespace().next());
        if let Some(n) = after("pfilter: filtering on ") {
            note("args.inference.parallel", n);
        }
        if let Some(n) = after("pfilter: a budget of ") {
            note("args.inference.pf_max_substeps", n);
        }
        if let Some(f) = after("time column read under --time-format ") {
            note("args.inference.time_format", &f.to_lowercase());
        }
    }
    note("no_ir_cache", if crate::test_support::ir_cache_filled(ir_cache) { "false" } else { "true" });
    seen.into_iter().map(|(k, v)| (k, v.into_iter().collect::<Vec<_>>().join(","))).collect()
}

/// The harness of `pfilter`'s class checks, from one base invocation.
struct Harness {
    base: PfilterInvocation,
}

impl Harness {
    /// The base each field is checked from: a fixture on which the field is
    /// live.
    fn for_field(name: &str) -> Harness {
        let mut base = pfilter_base();
        match name {
            "args.model_overrides.table" => {
                base = pfilter_from(&["patch.camdl", "--params", "patch.toml", "--table", "w=w1.tsv",
                    "--data", "patch_cases.tsv", "--particles", "20", "--seed", "1"]);
            }
            "args.scenario.scenario" => {
                base = pfilter_from(&["sir.ir.json", "--scenario", "set_all", "--data", "cases=cases.tsv",
                    "--particles", "20", "--seed", "1"]);
            }
            "args.scenario.disable" => base.args.scenario.enable = vec!["sia".into()],
            "args.stream.obs" => {
                base = pfilter_from(&["sir.ir.json", "--params", "p.toml", "--data", "both.tsv", "--obs", "cases",
                    "--particles", "20", "--seed", "1"]);
            }
            "args.save_final_state" => {
                base = pfilter_from(&["sir.ir.json", "--params", "p.toml", "--fit", "fit_holdout.toml",
                    "--particles", "20", "--seed", "1"]);
            }
            "args.n_paths" => base.args.save_paths = Some("paths.tsv".into()),
            "args.score_from" | "args.no_save_samples" => base.args.save_prequential = Some("preq".into()),
            "no_licm" => base.args.model = "sir.camdl".into(),
            // A design point is stored as the single call at its values, so
            // a larger design keeps the run id of every point it shares
            // with a smaller one.
            "args.sweep_file" => base.args.sweep_file = Some("design2.tsv".into()),
            _ => {}
        }
        Harness { base }
    }
}

impl InputHarness for Harness {
    type Invocation = PfilterInvocation;

    fn base(&self) -> PfilterInvocation {
        self.base.clone()
    }

    fn varied(&self, field: &FieldInfo) -> PfilterInvocation {
        let mut inv = self.base.clone();
        let a = &mut inv.args;
        match (field.name, field.class) {
            ("args.scenario.disable", _) => {
                a.scenario.enable.clear();
                a.scenario.disable = vec!["sia".into()];
            }
            ("args.save_paths", _) => a.save_paths = Some("paths.tsv".into()),
            ("args.save_filtering", _) => a.save_filtering = Some("filtering.tsv".into()),
            ("args.save_filtered_streams", _) => a.save_filtered_streams = Some("streams.tsv".into()),
            ("args.save_prequential", _) => a.save_prequential = Some("preq".into()),
            ("args.trace", _) => a.trace = Some("trace.tsv".into()),
            ("args.pf_health", _) => a.pf_health = Some("health.tsv".into()),
            ("args.output", _) => a.output = Some("ll.txt".into()),
            ("args.manifest", _) => a.manifest = Some("m.json".into()),
            ("args.link_root", _) => a.link_root = Some("links".into()),
            ("args.sweep_file", Class::Expands) => a.sweep_file = Some("design3.tsv".into()),
            (name, Class::Identity | Class::Render) => return self.base.varied(name),
            (name, class) => panic!("no second value for {name} ({class:?})"),
        }
        inv
    }

    fn exercised(&self, field: &FieldInfo) -> (PfilterInvocation, PfilterInvocation) {
        match field.name.strip_prefix("args.inference.") {
            Some(f) => {
                let (x, y) = <crate::args::PfilterCore as Inputs>::exercised(f);
                let (mut a, mut b) = (self.base.clone(), self.base.clone());
                a.args.inference = x;
                b.args.inference = y;
                (a, b)
            }
            None => <PfilterInvocation as Inputs>::exercised(field.name),
        }
    }

    /// A run shows an execution input was used when the effect it reports
    /// ([`used`]) matches the value it recorded ([`crate::test_support::recorded_execution`]):
    /// `--parallel N` filters on N threads, the budget handed to the engine is
    /// `--pf-max-substeps` or, unset, the engine's default, the data loader
    /// reads the `--time-format` given, and `--no-ir-cache` leaves the IR
    /// cache empty while its absence fills it. An input with no effect named
    /// here does not show use.
    fn shows_use(&self, field: &FieldInfo, seen: &Observed) -> bool {
        let effect = |prefix: &str| seen.effects.get(&format!("{prefix}.{}", field.name)).cloned();
        let (Some(recorded), Some(used)) = (effect("recorded"), effect("used")) else { return false };
        let expected = match (field.name, recorded.as_str()) {
            ("args.inference.pf_max_substeps", "null") => sim::inference::degeneracy::ITER_BUDGET.to_string(),
            (_, r) => r.to_string(),
        };
        used == expected
    }

    fn run(&self, invocation: &PfilterInvocation, store: Option<&Path>) -> store::Result<Observed> {
        run(invocation, store)
    }
}

/// Every declared input of `camdl pfilter` keeps the promise its class makes.
#[test]
fn every_pfilter_input_keeps_its_class() {
    let fields: Vec<FieldInfo> = PfilterInvocation::fields().to_vec();
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

/// Every argument clap accepts for `pfilter`, and every global option, is a
/// declared input; and every declared input is one clap accepts.
#[test]
fn every_pfilter_argument_is_declared() {
    use clap::CommandFactory;
    let cmd = crate::Cli::command();
    let sub = cmd.find_subcommand("pfilter").expect("the pfilter subcommand");
    let mut accepted: std::collections::BTreeSet<String> =
        sub.get_arguments().map(|a| a.get_id().to_string()).filter(|id| id != "help").collect();
    accepted.extend(cmd.get_arguments().map(|a| a.get_id().to_string()).filter(|id| id != "help" && id != "version"));
    let declared: std::collections::BTreeSet<String> =
        PfilterInvocation::fields().iter().map(|f| f.name.rsplit('.').next().unwrap().to_string()).collect();
    assert_eq!(accepted, declared);
}

/// The tree one filter run with a saved by-product makes, as "Layouts, kind
/// by kind" draws it: the build directory, the compiled model and the data
/// table as content artifacts, the run under its model, settings and
/// parameters groups, named by the seed typed, and the by-product inside it
/// under its kind.
#[test]
fn a_run_makes_the_tree_of_the_proposal() {
    let cwd = workdir();
    let store = cwd.path().join("camdl-store");
    let args = argv(&pfilter_from(&["sir.ir.json", "--params", "p.toml", "--data", "cases=cases.tsv",
        "--particles", "1000", "--seed", "1", "--save-paths", "paths.tsv"]));
    let out = camdl(cwd.path(), &store, &args, &[]);
    assert!(out.status.success(), "{}", String::from_utf8_lossy(&out.stderr));

    // Each directory's entries, with the 12 hex digits of each suffix
    // replaced by `*`, so the shape is compared and not the keys.
    fn shape(dir: &Path, depth: usize, out: &mut Vec<String>) {
        let mut names: Vec<String> = std::fs::read_dir(dir).unwrap().flatten()
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
        "  data_table",
        "    cases-*",
        "      identity.json",
        "      run.json",
        "      table.tsv",
        "  pfilter",
        "    sir-*",
        "      pf-n1000-r1-dt1-*",
        "        params-*",
        "          seed_1-*",
        "            identity.json",
        "            loglik.toml",
        "            provenance.json",
        "            run.json",
        "            saved_paths",
        "              paths-*",
        "                identity.json",
        "                paths.tsv",
        "                run.json",
        "store.json",
    ];
    assert_eq!(got, want, "the store's tree:\n{}", got.join("\n"));
    assert!(!std::fs::read(cwd.path().join("paths.tsv")).unwrap().is_empty(), "the export is written");
}

/// The run's id and folder for the base fixture with `extra`, and the
/// command's stderr.
fn run_in(store: &Path, cwd: &Path, extra: &[&str]) -> (Vec<ObservedResult>, String) {
    let mut line = vec!["sir.ir.json", "--params", "p.toml", "--data", "cases=cases.tsv", "--particles", "20", "--seed", "1"];
    line.extend_from_slice(extra);
    let out = camdl(cwd, store, &argv(&pfilter_from(&line)), &[]);
    let stderr = String::from_utf8_lossy(&out.stderr).into_owned();
    assert!(out.status.success(), "{stderr}");
    (observe(store), stderr)
}

/// A rerun with `--trace` after the trace is stored computes nothing and
/// exports the stored copy; asking for the trace of a stored run adds the
/// by-product and leaves the run's id unchanged.
#[test]
fn a_by_product_is_added_later_and_then_reused() {
    let cwd = workdir();
    let store = cwd.path().join("camdl-store");
    let (first, err) = run_in(&store, cwd.path(), &[]);
    assert!(err.contains("stored"), "{err}");
    let (second, err) = run_in(&store, cwd.path(), &["--trace", "trace.tsv"]);
    assert!(err.contains("stored"), "the run is reproduced to add its trace: {err}");
    let pf = |r: &[ObservedResult]| r.iter().filter(|x| x.kind == "pfilter").map(|x| (x.run_id, x.files.clone())).collect::<Vec<_>>();
    assert_eq!(pf(&first), pf(&second), "the run's id and files are unchanged");
    assert!(second.iter().any(|r| r.kind == "trace"), "the trace is stored");
    let exported = std::fs::read(cwd.path().join("trace.tsv")).unwrap();
    std::fs::remove_file(cwd.path().join("trace.tsv")).unwrap();
    let (third, err) = run_in(&store, cwd.path(), &["--trace", "trace.tsv"]);
    assert!(err.contains("cached"), "the rerun is a cache hit: {err}");
    assert_eq!(observe(&store).len(), third.len());
    assert_eq!(std::fs::read(cwd.path().join("trace.tsv")).unwrap(), exported, "the export is the stored copy");
}

/// `--label` names the invocation; `camdl cat @L` and `camdl cat <id>` print
/// the stored log-likelihood; a rerun under another label reuses the result.
#[test]
fn a_label_names_the_invocation_and_cat_finds_it() {
    let cwd = workdir();
    let store = cwd.path().join("camdl-store");
    let (results, _) = run_in(&store, cwd.path(), &["--label", "first"]);
    let (again, err) = run_in(&store, cwd.path(), &["--label", "second"]);
    assert!(err.contains("cached"), "{err}");
    assert_eq!(results, again);
    let id = results.iter().find(|r| r.kind == "pfilter").unwrap().run_id;
    let cat = |handle: &str| {
        let out = camdl(cwd.path(), &store, &["cat".into(), handle.into()], &[]);
        assert!(out.status.success(), "cat {handle}: {}", String::from_utf8_lossy(&out.stderr));
        String::from_utf8(out.stdout).unwrap()
    };
    let by_id = cat(&id.short12());
    assert!(by_id.starts_with("loglik = "), "{by_id}");
    assert_eq!(cat("@first"), by_id);
    assert_eq!(cat("@second"), by_id);
}

/// `--no-store` writes no store, and prints what the stored run prints.
#[test]
fn no_store_leaves_nothing_behind() {
    let cwd = workdir();
    let store = cwd.path().join("camdl-store");
    let args = argv(&pfilter_from(&["sir.ir.json", "--params", "p.toml", "--data", "cases=cases.tsv",
        "--particles", "20", "--seed", "1", "--no-store"]));
    let out = camdl(cwd.path(), &store, &args, &[]);
    assert!(out.status.success(), "{}", String::from_utf8_lossy(&out.stderr));
    assert!(!store.exists(), "--no-store writes no store");
    let stored = camdl(cwd.path(), &store, &args[..args.len() - 3].iter().cloned()
        .chain(["--progress".into(), "none".into()]).collect::<Vec<_>>(), &[]);
    assert_eq!(out.stdout, stored.stdout, "the same log-likelihood");
}

/// One pass of the base fixture costs 400 particle-substeps: a budget of 399
/// is refused with exit status 3 and stores no run.
#[test]
fn a_budget_below_one_pass_is_refused_and_stores_nothing() {
    let cwd = workdir();
    let store = cwd.path().join("camdl-store");
    let args = argv(&pfilter_from(&["sir.ir.json", "--params", "p.toml", "--data", "cases=cases.tsv",
        "--particles", "20", "--seed", "1", "--pf-max-substeps", "399"]));
    let out = camdl(cwd.path(), &store, &args, &[]);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(3), "{stderr}");
    assert!(stderr.contains("PFIterationBudget"), "{stderr}");
    // An in-place run that fails keeps its folder, marked failed, and is
    // never served; a rerun reclaims it.
    for r in observe(&store).iter().filter(|r| r.kind == "pfilter") {
        assert!(r.files.is_empty(), "a refused run records no result file: {:?}", r.files);
    }
    let failed = walk_records(&store).into_iter().filter(|(kind, _)| kind == "pfilter").map(|(_, status)| status).collect::<Vec<_>>();
    assert_eq!(failed, ["failed"], "the refused run is recorded as failed");
}

/// The base fixture's command line with `extra`.
fn held_line(extra: &[&str]) -> Vec<String> {
    let mut v: Vec<String> = ["pfilter", "sir.ir.json", "--params", "p.toml", "--data", "cases=cases.tsv",
        "--particles", "20", "--seed", "1", "--progress", "none"].iter().map(|s| s.to_string()).collect();
    v.extend(extra.iter().map(|s| s.to_string()));
    v
}

/// An owner on another host, as the store's records name one.
const REMOTE_HOST: &str = "another-host.invalid";
const REMOTE_PID: u32 = 4242;
const REMOTE_STARTED: &str = "2026-01-01T00:00:00.000Z";

/// Stores the base fixture's run in `store`, then rewrites its record as a
/// claim held by a process on another host: running, and owned there.
/// Returns the first run's output, the stored run, and its folder.
fn hold_from_another_host(cwd: &Path, store: &Path) -> (Output, ObservedResult, PathBuf) {
    let first = camdl(cwd, store, &held_line(&[]), &[]);
    assert!(first.status.success(), "{}", String::from_utf8_lossy(&first.stderr));
    let stored = observe(store).into_iter().find(|r| r.kind == "pfilter").expect("the stored run");
    let mut dirs = vec![store.to_path_buf()];
    let mut run_dir = None;
    while let Some(d) = dirs.pop() {
        for e in std::fs::read_dir(&d).unwrap().flatten() {
            let p = e.path();
            if p.is_dir() && !e.file_name().to_string_lossy().starts_with('.') {
                let id = p.join("identity.json");
                if id.is_file() && std::fs::read_to_string(&id).unwrap().contains(&stored.run_id.to_hex()) {
                    run_dir = Some(p.clone());
                }
                dirs.push(p);
            }
        }
    }
    let run_dir = run_dir.expect("the run's folder");
    let remote = serde_json::json!({ "host": REMOTE_HOST, "pid": REMOTE_PID, "started": REMOTE_STARTED });
    let mut record: serde_json::Value = serde_json::from_slice(&std::fs::read(run_dir.join("run.json")).unwrap()).unwrap();
    record["status"] = "running".into();
    record["owner"] = remote.clone();
    std::fs::write(run_dir.join("run.json"), serde_json::to_vec_pretty(&record).unwrap()).unwrap();
    std::fs::write(run_dir.join(".lock"), serde_json::to_vec_pretty(&serde_json::json!({ "owner": remote })).unwrap()).unwrap();
    (first, stored, run_dir)
}

/// A run whose record names an owner on another host that is no longer
/// running (here one with no beat file) is refused, with `--force` named, and
/// `--force` moves the folder to the quarantine and computes the run again.
#[test]
fn a_run_held_from_another_host_is_refused_and_force_computes_it_again() {
    let cwd = workdir();
    let store = cwd.path().join("camdl-store");
    let (first, stored, _) = hold_from_another_host(cwd.path(), &store);

    let refused = camdl(cwd.path(), &store, &held_line(&[]), &[]);
    let stderr = String::from_utf8_lossy(&refused.stderr);
    assert!(!refused.status.success(), "a run held from another host is not reclaimed without --force: {stderr}");
    assert!(stderr.contains(REMOTE_HOST) && stderr.contains("--force"), "the refusal names the host and --force: {stderr}");

    let forced = camdl(cwd.path(), &store, &held_line(&["--force"]), &[]);
    let stderr = String::from_utf8_lossy(&forced.stderr);
    assert!(forced.status.success(), "--force computes the run again: {stderr}");
    assert_eq!(forced.stdout, first.stdout, "the same log-likelihood");
    let again = observe(&store).into_iter().find(|r| r.kind == "pfilter").expect("the run, stored again");
    assert_eq!((again.run_id, again.files), (stored.run_id, stored.files), "the same run id and files");
    assert_eq!(walk_records(&store).into_iter().filter(|(k, _)| k == "pfilter").map(|(_, s)| s).collect::<Vec<_>>(),
        ["completed"], "the run is recorded as completed");
    assert_eq!(std::fs::read_dir(store.join(".quarantine")).map(|d| d.count()).unwrap_or(0), 1,
        "the held folder is in the quarantine");
}

/// A run held by an owner on another host whose beat counter is still
/// rising is refused even under `--force`, and its folder is left in place.
#[test]
fn a_run_held_from_another_host_with_a_live_beat_is_refused_under_force() {
    let cwd = workdir();
    let store = cwd.path().join("camdl-store");
    let (_, _, run_dir) = hold_from_another_host(cwd.path(), &store);
    let before = std::fs::read(run_dir.join("run.json")).unwrap();

    // The owner's beat file, raised every half second while the command runs.
    let beat = store.join(".owners").join(format!("{}.json",
        store::path_label(&format!("{REMOTE_HOST}-{REMOTE_PID}-{REMOTE_STARTED}"))));
    std::fs::create_dir_all(beat.parent().unwrap()).unwrap();
    let write_beat = |n: u64| {
        let body = serde_json::json!({ "format": 1, "host": REMOTE_HOST, "pid": REMOTE_PID,
            "started": REMOTE_STARTED, "beat": n, "invocation": null });
        let tmp = beat.with_extension("json.tmp");
        std::fs::write(&tmp, serde_json::to_vec(&body).unwrap()).unwrap();
        std::fs::rename(&tmp, &beat).unwrap();
    };
    write_beat(0);
    let stop = std::sync::atomic::AtomicBool::new(false);
    let refused = std::thread::scope(|s| {
        s.spawn(|| {
            let mut n = 1;
            while !stop.load(std::sync::atomic::Ordering::Relaxed) {
                std::thread::sleep(std::time::Duration::from_millis(500));
                write_beat(n);
                n += 1;
            }
        });
        let out = camdl(cwd.path(), &store, &held_line(&["--force"]), &[]);
        stop.store(true, std::sync::atomic::Ordering::Relaxed);
        out
    });
    let stderr = String::from_utf8_lossy(&refused.stderr);
    assert!(!refused.status.success(), "--force does not take a run whose owner's beat is rising: {stderr}");
    assert!(stderr.contains(REMOTE_HOST), "the refusal names the owner: {stderr}");
    assert_eq!(std::fs::read(run_dir.join("run.json")).unwrap(), before, "the held run is left as it was");
    assert!(!store.join(".quarantine").exists() || std::fs::read_dir(store.join(".quarantine")).unwrap().count() == 0,
        "nothing is moved to the quarantine");
}

/// `camdl` in `cwd` as [`camdl`] runs it, but with `CAMDL_STORE` set only
/// when `store` is given.
fn camdl_bare(cwd: &Path, store: Option<&Path>, args: &[&str]) -> Output {
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
        .env("CAMDL_IR_CACHE_DIR", cwd.join("ir-cache"));
    if let Some(s) = store {
        cmd.env("CAMDL_STORE", s);
    }
    cmd.args(args).output().expect("spawn camdl")
}

/// A reading command pointed at a run store, by `--store` or by
/// `CAMDL_STORE`, reads that store and nothing beside it: not a
/// `./camdl-store` in the working directory, here holding a second filter run
/// and a fit. With nothing named, `./camdl-store` is read.
#[test]
fn a_named_store_is_read_alone() {
    let cwd = workdir();
    let named = cwd.path().join("named");
    let base = ["pfilter", "sir.ir.json", "--params", "p.toml", "--data", "cases=cases.tsv", "--particles", "20",
        "--progress", "none"];
    let run = |store: &str, seed: &str| {
        let mut line: Vec<&str> = base.to_vec();
        line.extend(["--seed", seed, "--store", store]);
        let out = camdl_bare(cwd.path(), None, &line);
        assert!(out.status.success(), "{}", String::from_utf8_lossy(&out.stderr));
    };
    run("named", "1");
    run("camdl-store", "2");
    let fit = camdl_bare(cwd.path(), None, &["fit", "run", "fit_plain.toml", "--progress", "none"]);
    assert!(fit.status.success(), "{}", String::from_utf8_lossy(&fit.stderr));
    assert!(cwd.path().join("camdl-store").is_dir(), "the stray store is there");

    let kinds = |out: Output| -> Vec<String> {
        assert!(out.status.success(), "{}", String::from_utf8_lossy(&out.stderr));
        let mut k: Vec<String> = String::from_utf8_lossy(&out.stdout).lines()
            .filter_map(|l| serde_json::from_str::<serde_json::Value>(l).ok())
            .map(|v| v["kind"].as_str().unwrap_or("?").to_string())
            .collect();
        k.sort();
        k
    };
    let list = ["list", "--format", "json", "--all"];
    let by_flag = kinds(camdl_bare(cwd.path(), None, &[&list[..], &["--store", "named"]].concat()));
    let by_env = kinds(camdl_bare(cwd.path(), Some(&named), &list));
    assert_eq!(by_flag, ["pfilter"], "--store lists the one run in that store");
    assert_eq!(by_env, ["pfilter"], "CAMDL_STORE lists the one run in that store");
    let default = kinds(camdl_bare(cwd.path(), None, &list));
    assert!(default.iter().any(|k| k != "pfilter") && default.iter().filter(|k| *k == "pfilter").count() == 1,
        "with nothing named, ./camdl-store is read, its filter run and its fit: {default:?}");
}

/// `--no-licm` and `CAMDL_NO_LICM` change the compiled model a run stores,
/// and so the run's id; an engine switch changes every id through the
/// ambient identity and leaves the stored log-likelihood alone.
#[test]
fn the_compile_settings_and_the_engine_switches_key_the_run() {
    let ids = |no_licm: bool, env: &[(&str, &str)]| -> BTreeMap<String, Vec<(Key, BTreeMap<String, Key>)>> {
        let cwd = workdir();
        let store = cwd.path().join("camdl-store");
        let mut inv = pfilter_from(&["sir.camdl", "--params", "p.toml", "--data", "cases=cases.tsv", "--particles", "20", "--seed", "1"]);
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
        assert_ne!(other["pfilter"][0].0, base["pfilter"][0].0, "{what} changes the run's id");
        assert_eq!(other["data_table"], base["data_table"], "{what} leaves the data alone");
    }
    for var in ["CAMDL_EVAL_UNRESOLVED", "CAMDL_NO_BINDING_CACHE", "CAMDL_EXPERIMENTAL_DIFFERENTIAL_TEST"] {
        let other = ids(false, &[(var, "1")]);
        assert_ne!(other["pfilter"][0].0, base["pfilter"][0].0, "{var} changes the run's id");
        assert_eq!(other["pfilter"][0].1, base["pfilter"][0].1, "{var} stores the same log-likelihood");
    }
}

/// The log-likelihoods of the committed guide models
/// (`docs/guide/first-fit/`), 200 particles and seed 7, as the old store
/// held them in `loglik.toml` before `pfilter` moved: one run and the mean of
/// three replicates, bit for bit; and the digests of the saved paths and the
/// prequential trace of the negative-binomial model.
#[test]
fn the_guide_models_score_what_they_scored_before_the_move() {
    let guide = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../../docs/guide/first-fit");
    let pins: [(&str, &str, &str, &str); 3] = [
        ("sir-r0", "R0 = 3.75\ngamma = 0.4\nN0 = 763\nI0 = 3\n", "-75.90308777220476", "-76.78530058456637"),
        ("sir-beta", "beta = 1.6\ngamma = 0.4\nN0 = 763\nI0 = 3\n", "-75.07473219243356", "-74.81864958962596"),
        ("sir-negbin", "R0 = 3.75\ngamma = 0.4\nk = 20\nN0 = 763\nI0 = 3\n", "-72.820360863145", "-72.1762014916757"),
    ];
    for (model, params, single, mean) in pins {
        let cwd = workdir();
        let store = cwd.path().join("camdl-store");
        std::fs::write(cwd.path().join("params.toml"), params).unwrap();
        let line = |extra: &[&str]| -> Vec<String> {
            let mut v: Vec<String> = ["pfilter", &guide.join(format!("{model}.camdl")).to_string_lossy(), "--params", "params.toml",
                "--data", &guide.join("data/in_bed.tsv").to_string_lossy(), "--particles", "200", "--seed", "7",
                "--progress", "none"].iter().map(|s| s.to_string()).collect();
            v.extend(extra.iter().map(|s| s.to_string()));
            v
        };
        for (extra, want) in [(&[][..], single), (&["--replicates", "3"][..], mean)] {
            let out = camdl(cwd.path(), &store, &line(extra), &[]);
            assert!(out.status.success(), "{}", String::from_utf8_lossy(&out.stderr));
            let mut found = None;
            for r in observe(&store).iter().filter(|r| r.kind == "pfilter") {
                let replicates = r.files.contains_key(super::REPLICATES_FILE);
                if replicates != extra.is_empty() {
                    found = Some(r.run_id);
                }
            }
            let id = found.expect("the run is stored");
            let out = camdl(cwd.path(), &store, &["cat".into(), id.to_hex()], &[]);
            let ll = super::Loglik::from_bytes(&out.stdout).unwrap();
            let ll = ll.loglik.expect("a scored run");
            assert_eq!(ll.to_bits(), want.parse::<f64>().unwrap().to_bits(), "{model} {extra:?}: {ll}");
        }
        if model == "sir-negbin" {
            let out = camdl(cwd.path(), &store, &line(&["--save-paths", "p.tsv", "--save-prequential", "q"]), &[]);
            assert!(out.status.success(), "{}", String::from_utf8_lossy(&out.stderr));
            for (file, digest) in [
                ("p.tsv", "84c4e24d7b569426e7849a81746b5f8dd7bc35842c2f62493c7d5834ca03ed0e"),
                ("q.tsv", "8e96b4c7f8b8a0f611d7518632b69af731caf63ae7f6c06ee6d4f7722d8b664f"),
                ("q.json", "e70b86853d1df0bcd9b1d7edb3a6919642895a19409d8d67402c210c05f050dc"),
            ] {
                assert_eq!(Key::digest_bytes(&std::fs::read(cwd.path().join(file)).unwrap()).to_hex(), digest, "{file}");
            }
        }
    }
}

/// A `--no-store` run that fails after its temporary store is made removes
/// that store: here the filter is refused by its budget.
#[test]
fn a_failed_no_store_run_leaves_no_temporary_store() {
    let cwd = workdir();
    let tmp = tempfile::tempdir().unwrap();
    let tmpdir = tmp.path().to_string_lossy().into_owned();
    let line = ["sir.ir.json", "--params", "p.toml", "--data", "cases=cases.tsv", "--particles", "20",
        "--seed", "1", "--no-store", "--pf-max-substeps", "399"];
    let out = camdl(cwd.path(), &cwd.path().join("camdl-store"), &argv(&pfilter_from(&line)), &[("TMPDIR", &tmpdir)]);
    assert_eq!(out.status.code(), Some(3), "the budget refuses the run");
    let left: Vec<_> = std::fs::read_dir(tmp.path()).unwrap().flatten().map(|e| e.file_name()).collect();
    assert!(left.is_empty(), "the run left {left:?} in TMPDIR");
}

/// Undeclared reads (proposal, "Guarding against undeclared reads"): every
/// stored file a filter run does not declare as a dependency is corrupted
/// in place, keeping its size and modification time so the store still
/// serves it: the compiled model's provenance files, and every file of
/// another filter run. The run is then computed again, to add its trace,
/// and must reproduce its stored log-likelihood and write the trace a fresh
/// store writes.
#[test]
fn corrupting_undeclared_files_changes_nothing_a_run_computes() {
    let cwd = workdir();
    let store = cwd.path().join("camdl-store");
    let (results, _) = run_in(&store, cwd.path(), &[]);
    run_in(&store, cwd.path(), &["--param", "beta_a=0.45"]);
    let declared: Vec<String> = ["model.ir.json", "table.tsv"].map(String::from).to_vec();
    let mut poisoned = 0;
    let mut stack = vec![store.clone()];
    let ours = results.iter().find(|r| r.kind == "pfilter").unwrap().run_id;
    while let Some(d) = stack.pop() {
        for e in std::fs::read_dir(&d).unwrap().flatten() {
            let p = e.path();
            let name = e.file_name().to_string_lossy().into_owned();
            if p.is_dir() {
                // This run's own folder is the result under test, not an
                // upstream of it.
                let own = std::fs::read(p.join("identity.json")).ok()
                    .and_then(|b| serde_json::from_slice::<serde_json::Value>(&b).ok())
                    .is_some_and(|v| v["run_id"] == ours.to_hex());
                if !name.starts_with('.') && !own {
                    stack.push(p);
                }
                continue;
            }
            if ["identity.json", "run.json", "store.json"].contains(&name.as_str()) || declared.contains(&name) {
                continue;
            }
            let meta = std::fs::metadata(&p).unwrap();
            std::fs::write(&p, vec![b'#'; meta.len() as usize]).unwrap();
            std::fs::File::options().write(true).open(&p).unwrap().set_modified(meta.modified().unwrap()).unwrap();
            poisoned += 1;
        }
    }
    assert!(poisoned >= 5, "the fixture corrupts the model's provenance and the other run: {poisoned}");
    let (after, err) = run_in(&store, cwd.path(), &["--trace", "trace.tsv"]);
    assert!(err.contains("stored"), "the run is computed again for its trace: {err}");
    let pf = |r: &[ObservedResult]| r.iter().filter(|x| x.run_id == ours).map(|x| x.files.clone()).collect::<Vec<_>>();
    assert_eq!(pf(&results), pf(&after), "the stored log-likelihood is reproduced");
    let fresh = workdir();
    run_in(&fresh.path().join("camdl-store"), fresh.path(), &["--trace", "trace.tsv"]);
    assert_eq!(std::fs::read(cwd.path().join("trace.tsv")).unwrap(), std::fs::read(fresh.path().join("trace.tsv")).unwrap());
}

/// The compiled model is keyed before the parameters and the scenario
/// selection are applied: runs at two parameter files, and under two
/// intervention selections, share one compiled model and one model
/// directory, and differ in their parameters and scenario identity fields.
#[test]
fn the_parameters_and_the_scenario_do_not_key_the_compiled_model() {
    let cwd = workdir();
    let store = cwd.path().join("camdl-store");
    run_in(&store, cwd.path(), &[]);
    let mut line = vec!["sir.ir.json", "--params", "p2.toml", "--data", "cases=cases.tsv", "--particles", "20", "--seed", "1"];
    let out = camdl(cwd.path(), &store, &argv(&pfilter_from(&line)), &[]);
    assert!(out.status.success(), "{}", String::from_utf8_lossy(&out.stderr));
    line.extend_from_slice(&["--enable", "sia"]);
    let out = camdl(cwd.path(), &store, &argv(&pfilter_from(&line)), &[]);
    assert!(out.status.success(), "{}", String::from_utf8_lossy(&out.stderr));
    let results = observe(&store);
    let count = |kind: &str| results.iter().filter(|r| r.kind == kind).count();
    assert_eq!(count("compiled_model"), 1, "one compiled model for every point and selection");
    assert_eq!(count("pfilter"), 3);
    let build = std::fs::read_dir(&store).unwrap().flatten().find(|e| e.path().join("pfilter").is_dir()).unwrap().path();
    let models: Vec<_> = std::fs::read_dir(build.join("pfilter")).unwrap().flatten().collect();
    assert_eq!(models.len(), 1, "one model directory");
}

/// An export path in a directory that does not exist is refused before the
/// filter runs, and nothing is stored.
#[test]
fn an_export_into_a_missing_directory_is_refused_before_the_filter_runs() {
    let cwd = workdir();
    let store = cwd.path().join("camdl-store");
    for flag in ["--save-final-state", "--trace", "--output", "--save-paths"] {
        let out = camdl(cwd.path(), &store, &argv(&pfilter_from(&["sir.ir.json", "--params", "p.toml", "--data", "cases=cases.tsv",
            "--particles", "20", "--seed", "1", flag, "sub/c.tsv"])), &[]);
        let err = String::from_utf8_lossy(&out.stderr);
        assert!(!out.status.success(), "{flag}: {err}");
        assert!(err.contains("sub") && err.contains("does not exist"), "{flag}: {err}");
        assert!(!observe(&store).iter().any(|r| r.kind == "pfilter"), "{flag}: nothing is stored");
    }
}
