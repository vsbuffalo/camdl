//! `camdl profile` on the run store, checked by running the release binary:
//! every declared input against its class (`store::check_input_classes`), the
//! tree a profile makes, a grid that grows, and the stored points of two
//! deterministic profiles against the values the old store held.
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
use crate::args::ProfileInvocation;
use crate::input_fixtures::{profile_base, profile_from};

/// The class checks' model: two infection routes, a recovery, one
/// intervention and two scenarios that each set `gamma`, observed as daily
/// cases and daily prevalence.
const SIR: &str = r#"
time_unit = 'days

compartments { S, I, R, V }

let N = S + I + R + V

parameters {
  beta_a : rate        in [0.01, 5.0]
  beta_b : rate        in [0.01, 5.0]
  gamma  : rate        in [0.01, 5.0]
  rho    : probability in [0.1, 0.9]
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
  slow { set = { gamma = 0.25 } }
  fast { set = { gamma = 0.15 } }
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

/// A model with an external table, for `--table`.
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

/// The pinned profiles' model: an SIR observed as weekly cases with
/// negative-binomial noise.
const WEEKLY: &str = r#"time_unit = 'days

compartments { S, I, R }

parameters {
  beta  : rate         in [0.05, 1.0]
  gamma : rate         in [0.01, 0.5]
  N0    : count        in [100, 100000]
  I0    : count        in [1, 1000]
  rho   : probability  in [0.1, 0.9]
  k     : positive     in [1.0, 100.0]
}

let N = S + I + R

transitions {
  infection : S --> I  @ beta * S * I / N
  recovery  : I --> R  @ gamma * I
}

init {
  I = I0
  S = N0 - I0
}

observations {
  weekly_cases {
    columns       { time : time, weekly_cases : count }
    covers        = closing_at(time, 7 'days)
    projected     = incidence(infection)
    emit_schedule = every 7 'days
    weekly_cases  ~ neg_binomial(mean = rho * projected, r = k)
  }
}

simulate {
  from = 0 'days
  to   = 56 'days
}
"#;

/// A fit config over `sir.ir.json`'s `cases` stream estimating `beta_a`, the
/// parameter the profiles sweep, and `rho`, with `fixed` as its `[fixed]`.
fn fit_toml(fixed: &str) -> String {
    format!(
        "[model]\ncamdl = \"sir.ir.json\"\n[data.observations]\ncases = \"cases.tsv\"\n\
         [estimate]\nbeta_a = {{ bounds = [0.05, 2.0] }}\nrho = {{ bounds = [0.1, 0.9] }}\n[fixed]\n{fixed}\
         [method]\nalgorithm = \"if2\"\nbackend = \"chain_binomial\"\nchains = 1\nparticles = 20\niterations = 2\ncooling = 0.9\n",
    )
}

fn fixture_files() -> Vec<(&'static str, String)> {
    let series = |f: &dyn Fn(u32) -> u32, to: u32, col: &str| {
        (1..=to).fold(format!("time\t{col}\n"), |s, t| s + &format!("{t}\t{}\n", f(t)))
    };
    vec![
        ("sir.camdl", SIR.to_string()),
        ("sir_other.camdl", SIR.replace("S = 990  I = 10", "S = 980  I = 20")),
        ("patch.camdl", PATCH.to_string()),
        ("weekly.camdl", WEEKLY.to_string()),
        ("cases.tsv", series(&|t| 3 + t, 20, "cases")),
        ("cases2.tsv", series(&|t| 4 + t, 20, "cases")),
        ("both.tsv", (1..=20).fold("time\tcases\tprevalent\n".to_string(), |s, t| s + &format!("{t}\t{}\t{}\n", 3 + t, 10 + 2 * t))),
        ("patch_cases.tsv", series(&|t| 2 + t, 10, "cases")),
        ("w1.tsv", "1.0\n0.5\n".into()),
        ("w2.tsv", "1.0\n0.7\n".into()),
        ("fit.toml", fit_toml("beta_b = 0.2\ngamma = 0.2\n")),
        ("fit_btrs.toml", fit_toml("beta_b = 0.2\ngamma = 0.2\n").replace("cooling = 0.9\n", "cooling = 0.9\nbinomial = \"btrs\"\n")),
        ("fit2.toml", fit_toml("beta_b = 0.25\ngamma = 0.2\n")),
        ("fit_scenario.toml", fit_toml("beta_b = 0.2\n")),
        ("fit_other_model.toml", fit_toml("beta_b = 0.2\ngamma = 0.2\n").replace("sir.ir.json", "sir_other.ir.json")),
        ("fit_source.toml", fit_toml("beta_b = 0.2\ngamma = 0.2\n").replace("sir.ir.json", "sir.camdl")),
        (
            "patch_fit.toml",
            "[model]\ncamdl = \"patch.ir.json\"\n[data.observations]\ncases = \"patch_cases.tsv\"\n\
             [data.tables]\nw = \"w1.tsv\"\n\
             [estimate]\nbeta = { bounds = [0.01, 2.0] }\nrho = { bounds = [0.05, 0.95], start = 0.5 }\n\
             [fixed]\ngamma = 0.2\n"
                .into(),
        ),
        ("fixed2.toml", "gamma = 0.25\n".into()),
        ("pstart.toml", "rho = 0.5\n".into()),
        ("draws.tsv", "chain\trho\n0\t0.4\n0\t0.5\n1\t0.6\n".into()),
        (
            "weekly_cases.tsv",
            "time\tweekly_cases\n7\t47\n14\t111\n21\t474\n28\t1066\n35\t1471\n42\t533\n49\t300\n56\t40\n".into(),
        ),
        (
            "weekly_fit.toml",
            "[model]\ncamdl = \"weekly.camdl\"\n\n[data.observations]\nweekly_cases = \"weekly_cases.tsv\"\n\n\
             [estimate]\nbeta = { bounds = [0.05, 1.0], start = 0.4 }\ngamma = { bounds = [0.01, 0.5], start = 0.15 }\n\n\
             [fixed]\nN0 = 10000\nI0 = 10\nrho = 0.6\nk = 10.0\n\n\
             [method]\nalgorithm = \"nl-sbplx\"\nbackend = \"ode\"\nchains = 1\n"
                .into(),
        ),
        ("weekly_start.toml", "rho = 0.5\n".into()),
        ("weekly_if2.toml", weekly_if2("")),
        ("weekly_if2_perturbation.toml", weekly_if2("[method.perturbation]\nbeta = 0.1\ngamma = 0.2\n")),
        (
            "weekly_rho.toml",
            "[model]\ncamdl = \"weekly.camdl\"\n\n[data.observations]\nweekly_cases = \"weekly_cases.tsv\"\n\n\
             [estimate]\nbeta = { bounds = [0.05, 1.0] }\ngamma = { bounds = [0.01, 0.5] }\nrho = { bounds = [0.1, 0.9] }\n\n\
             [fixed]\nN0 = 10000\nI0 = 10\nk = 10.0\n\n\
             [method]\nalgorithm = \"nl-bobyqa\"\nbackend = \"ode\"\nchains = 1\n"
                .into(),
        ),
    ]
}

/// `weekly_fit.toml`'s problem under an IF2 method, with `table` after it.
fn weekly_if2(table: &str) -> String {
    format!(
        "[model]\ncamdl = \"weekly.camdl\"\n\n[data.observations]\nweekly_cases = \"weekly_cases.tsv\"\n\n\
         [estimate]\nbeta = {{ bounds = [0.05, 1.0], start = 0.4 }}\ngamma = {{ bounds = [0.01, 0.5], start = 0.15 }}\n\n\
         [fixed]\nN0 = 10000\nI0 = 10\nrho = 0.6\nk = 10.0\n\n\
         [method]\nalgorithm = \"if2\"\nbackend = \"chain_binomial\"\nchains = 1\nparticles = 20\niterations = 2\ncooling = 0.9\n{table}"
    )
}

/// An nl-* profile perturbs nothing, so its points do not key the `--fit`
/// config's `[method.perturbation]`: a profile of a file with the table and
/// of the same file without it stores the same points.
#[test]
fn an_nl_profile_point_does_not_key_the_fits_perturbation() {
    let ids = |fit: &str| {
        let cwd = workdir();
        let store = cwd.path().join("camdl-store");
        let args: Vec<String> = ["profile", "weekly.camdl", "--fit", fit, "--sweep", "gamma=0.12,0.15",
            "--algorithm", "nl-sbplx", "--backend", "ode", "--n-starts", "1", "--seed", "3", "--progress", "none"]
            .iter().map(|s| s.to_string()).collect();
        let out = camdl(cwd.path(), &store, &args, &[]);
        assert!(out.status.success(), "{fit}: {}", String::from_utf8_lossy(&out.stderr));
        points(&store).into_keys().collect::<Vec<_>>()
    };
    let (bare, with) = (ids("weekly_if2.toml"), ids("weekly_if2_perturbation.toml"));
    assert_eq!(bare.len(), 2);
    assert_eq!(bare, with);
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
        let dir = crate::test_support::unique_temp_dir("profile_store_fixture");
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
fn argv(inv: &ProfileInvocation) -> Vec<String> {
    let ProfileInvocation { args, no_licm, no_ir_cache, verbosity: _, progress: _, no_progress: _ } = inv;
    let crate::args::ProfileArgs {
        model,
        model_overrides: crate::args::InferenceModelOverrides { fixed_cli, fixed_files, table },
        scenario: crate::args::ScenarioArgs { scenario, enable, disable },
        inference:
            crate::args::ProfileCore {
                dt,
                seed,
                parallel,
                allow_degenerate_rates,
                forcing_extrapolation: crate::args::ForcingExtrapolationArg { allow_forcing_extrapolation },
                time_format,
                pf_max_substeps,
                binomial,
            },
        obs,
        data,
        fit,
        from_manifest,
        sweep,
        particles,
        iterations,
        n_starts,
        starts,
        starts_chains,
        allow_nonconverged_source,
        cooling,
        cooling_target_iters,
        rw_sd,
        output,
        format,
        seeds,
        label,
        force,
        store,
        manifest,
        link_root,
        algorithm,
        backend,
    } = args;
    let p = |p: &Path| p.to_string_lossy().into_owned();
    let mut v: Vec<String> = vec!["profile".into(), p(model)];
    let mut opt = |flag: &str, value: Option<String>| {
        if let Some(x) = value {
            v.push(flag.into());
            v.push(x);
        }
    };
    for f in fixed_cli {
        opt("--fixed", Some(format!("{}={}", f.name, f.value)));
    }
    for f in fixed_files {
        opt("--fixed-file", Some(p(f)));
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
    opt("--from-manifest", from_manifest.as_deref().map(p));
    for s in sweep {
        opt("--sweep", Some(format!("{}={}", s.name, s.grid.expand().iter().map(|x| x.to_string()).collect::<Vec<_>>().join(","))));
    }
    opt("--particles", particles.map(|x| x.to_string()));
    opt("--iterations", iterations.map(|x| x.to_string()));
    opt("--n-starts", Some(n_starts.to_string()));
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
    opt("--cooling", cooling.map(|x| x.to_string()));
    opt("--cooling-target-iters", cooling_target_iters.map(|x| x.to_string()));
    opt("--rw-sd", rw_sd.as_ref().map(|r| match r {
        crate::args::types::RwSd::Auto => "auto".to_string(),
        crate::args::types::RwSd::Map(m) => {
            let mut parts: Vec<String> = m.iter()
                .map(|(k, v)| format!("{k}={}", v.map(|x| x.to_string()).unwrap_or_else(|| "auto".into())))
                .collect();
            parts.sort();
            parts.join(",")
        }
    }));
    opt("--output", output.as_deref().map(p));
    opt("--format", Some(serde_json::to_value(format).unwrap().as_str().unwrap().to_string()));
    opt("--seeds", seeds.as_ref().map(|s| s.expand().iter().map(|x| x.to_string()).collect::<Vec<_>>().join(",")));
    opt("--label", label.clone());
    opt("--store", store.as_deref().map(p));
    opt("--manifest", manifest.as_deref().map(p));
    opt("--link-root", link_root.as_deref().map(p));
    opt("--algorithm", algorithm.clone());
    opt("--backend", backend.clone());
    let mut flag = |set: bool, name: &str| {
        if set {
            v.push(name.into());
        }
    };
    flag(*allow_degenerate_rates, "--allow-degenerate-rates");
    flag(*allow_forcing_extrapolation, "--allow-forcing-extrapolation");
    flag(*allow_nonconverged_source, "--allow-nonconverged-source");
    flag(*force, "--force");
    flag(*no_licm, "--no-licm");
    flag(*no_ir_cache, "--no-ir-cache");
    v.extend(["--progress".to_string(), "none".to_string()]);
    v
}

/// Every result folder under `root`: its kind, run id and result files with
/// their digests.
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
                    .map(|o| {
                        o.iter()
                            .filter(|(_, f)| f["role"] == "result")
                            .map(|(n, f)| (n.clone(), Key::from_hex(f["sha256"].as_str().unwrap()).unwrap()))
                            .collect()
                    })
                    .unwrap_or_default();
                let kind = id["kind"].as_str().unwrap().to_string();
                out.push(ObservedResult {
                    content: [crate::camdl_store::CompiledModel::KIND, crate::camdl_store::DataTable::KIND].contains(&kind.as_str()),
                    kind,
                    run_id: Key::from_hex(id["run_id"].as_str().unwrap()).unwrap(),
                    files,
                    upstream: None,
                });
            }
            walk(&p, out);
        }
    }
    let mut out = Vec::new();
    walk(root, &mut out);
    out
}

/// The last invocation record under `root`, as JSON.
fn last_invocation(root: &Path) -> serde_json::Value {
    let dir = root.join(".invocations");
    let seq = std::fs::read_dir(&dir).unwrap().flatten()
        .filter_map(|e| e.file_name().to_string_lossy().strip_suffix(".json")?.parse::<u64>().ok())
        .max()
        .expect("an invocation record");
    serde_json::from_slice(&std::fs::read(dir.join(format!("{seq}.json"))).unwrap()).unwrap()
}

/// Runs `inv` in a fresh copy of the fixtures, storing into `store` or a
/// fresh store, and reads back what it stored and exported.
fn run(inv: &ProfileInvocation, store: Option<&Path>) -> store::Result<Observed> {
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
    // The table is rendered from the stored points, never copied from a
    // stored file, so no export is named in `copies`.
    let mut exports = BTreeMap::new();
    if let Some(path) = inv.args.output.as_ref() {
        let bytes = std::fs::read(cwd.path().join(path)).unwrap_or_default();
        exports.insert(path.clone(), Key::digest_bytes(&bytes));
    }
    if let Some(path) = &inv.args.manifest {
        let bytes = std::fs::read(cwd.path().join(path)).unwrap_or_default();
        exports.insert(path.clone(), Key::digest_bytes(&bytes));
    }
    let mut copies = BTreeMap::new();
    if let Some(links) = &inv.args.link_root {
        crate::test_support::link_exports(cwd.path(), links, &mut exports, &mut copies);
    }
    let mut effects = crate::test_support::recorded_execution(&root, super::ProfilePoint::KIND);
    effects.extend(used(&String::from_utf8_lossy(&out.stderr), &cwd.path().join("ir-cache")));
    Ok(Observed { results: observe(&root), exports, copies, effects })
}

/// The log filter under which a run reports what its execution inputs did.
const USE_LOG: &str = "camdl::profile=debug,camdl::caltime_load=debug";

/// What a run's execution inputs did, each under `used.<field>`: from its
/// debug log (`USE_LOG`), the threads it computed on, the backend its cells
/// ran on, whether a starts source's verdict was lifted and the
/// `--time-format` its data loader read; and from the IR cache directory,
/// whether the compile bypassed the cache (`--no-ir-cache`). An effect the
/// log reports twice with different values is recorded as both, so it
/// matches no single value.
fn used(stderr: &str, ir_cache: &Path) -> BTreeMap<String, String> {
    let mut seen: BTreeMap<String, std::collections::BTreeSet<String>> = BTreeMap::new();
    let mut note = |field: &str, value: &str| {
        seen.entry(format!("used.{field}")).or_default().insert(value.to_string());
    };
    // The problem is posed from the fit file with the time column read as
    // `fit run` reads it; the cells' reads follow this line.
    let mut cells_bind = false;
    for line in stderr.lines() {
        cells_bind |= line.contains("profile: the cells bind the stored problem's data");
        let after = |marker: &str| line.split_once(marker).and_then(|(_, rest)| rest.split_whitespace().next());
        if let Some(n) = after("profile: computing on ") {
            note("args.inference.parallel", n);
        }
        if let Some(b) = after("profile: cells run on backend ") {
            note("args.backend", b);
        }
        if let Some(b) = after("profile: a starts source's verdict is lifted: ") {
            note("args.allow_nonconverged_source", b);
        }
        if let Some(f) = after("time column read under --time-format ").filter(|_| cells_bind) {
            note("args.inference.time_format", &f.to_lowercase());
        }
    }
    note("no_ir_cache", if crate::test_support::ir_cache_filled(ir_cache) { "false" } else { "true" });
    seen.into_iter().map(|(k, v)| (k, v.into_iter().collect::<Vec<_>>().join(","))).collect()
}

/// The harness of `profile`'s class checks, from one base invocation.
struct Harness {
    base: ProfileInvocation,
}

impl Harness {
    /// The base each field is checked from: a fixture on which the field is
    /// live.
    fn for_field(name: &str) -> Harness {
        let mut base = profile_base();
        match name {
            "args.model_overrides.table" => {
                base = profile_from(&["patch.ir.json", "--fit", "patch_fit.toml",
                    "--sweep", "beta=0.3,0.4", "--particles", "20", "--iterations", "2", "--n-starts", "1"]);
            }
            "args.scenario.scenario" => {
                base.args.fit = Some("fit_scenario.toml".into());
                base.args.scenario.scenario = Some("slow".into());
            }
            "args.scenario.disable" => base.args.scenario.enable = vec!["sia".into()],
            // One start is the same point under every spread rule that draws
            // it at the same values; two are not.
            "args.starts" => base.args.n_starts = 2,
            "args.starts_chains" => {
                base.args.starts = Some("from_posterior=draws.tsv".parse().unwrap());
                base.args.allow_nonconverged_source = true;
            }
            "args.algorithm" => {
                base = profile_from(&["sir.ir.json", "--fit", "fit.toml", "--sweep", "beta_a=0.3,0.4",
                    "--n-starts", "1", "--algorithm", "nl-sbplx", "--backend", "ode"]);
            }
            "no_licm" => {
                base.args.model = "sir.camdl".into();
                base.args.fit = Some("fit_source.toml".into());
            }
            _ => {}
        }
        Harness { base }
    }
}

impl InputHarness for Harness {
    type Invocation = ProfileInvocation;

    fn base(&self) -> ProfileInvocation {
        self.base.clone()
    }

    fn varied(&self, field: &FieldInfo) -> ProfileInvocation {
        let mut inv = self.base.clone();
        let a = &mut inv.args;
        match (field.name, field.class) {
            ("args.scenario.scenario", _) => a.scenario.scenario = Some("fast".into()),
            ("args.scenario.enable", _) => a.scenario.enable = vec!["sia".into()],
            ("args.scenario.disable", _) => {
                a.scenario.enable.clear();
                a.scenario.disable = vec!["sia".into()];
            }
            // The model must be the fit file's, so a second model comes with
            // a fit file naming it.
            ("args.model", _) => {
                a.model = "sir_other.ir.json".into();
                a.fit = Some("fit_other_model.toml".into());
            }
            ("args.sweep", Class::Expands) => a.sweep = vec!["beta_a=0.3,0.4,0.5".parse().unwrap()],
            ("args.seeds", Class::Expands) => a.seeds = Some("1,2".parse().unwrap()),
            ("args.output", _) => a.output = Some("table.tsv".into()),
            ("args.manifest", _) => a.manifest = Some("m.json".into()),
            ("args.link_root", _) => a.link_root = Some("links".into()),
            (name, Class::Identity | Class::Render) => return self.base.varied(name),
            (name, class) => panic!("no second value for {name} ({class:?})"),
        }
        inv
    }

    fn exercised(&self, field: &FieldInfo) -> (ProfileInvocation, ProfileInvocation) {
        match field.name.strip_prefix("args.inference.") {
            Some(f) => {
                let (x, y) = <crate::args::ProfileCore as Inputs>::exercised(f);
                let (mut a, mut b) = (self.base.clone(), self.base.clone());
                a.args.inference = x;
                b.args.inference = y;
                (a, b)
            }
            None => match field.name.strip_prefix("args.") {
                Some(f) if f != "args" && !f.contains('.') => {
                    let (x, y) = <crate::args::ProfileArgs as Inputs>::exercised(f);
                    let (mut a, mut b) = (self.base.clone(), self.base.clone());
                    a.args = x;
                    b.args = y;
                    (a, b)
                }
                _ => <ProfileInvocation as Inputs>::exercised(field.name),
            },
        }
    }

    /// A run shows an execution input was used when the effect it reports
    /// ([`used`]) matches the value it recorded
    /// ([`crate::test_support::recorded_execution`]): `--parallel N` computes
    /// on N threads, the cells run on `--backend` or, unset, `chain_binomial`,
    /// the verdict-lifting flag is reported as given, the data loader reads
    /// the `--time-format` given, and `--no-ir-cache` leaves the IR cache
    /// empty while its absence fills it.
    fn shows_use(&self, field: &FieldInfo, seen: &Observed) -> bool {
        let effect = |prefix: &str| seen.effects.get(&format!("{prefix}.{}", field.name)).cloned();
        let (Some(recorded), Some(used)) = (effect("recorded"), effect("used")) else { return false };
        let expected = match (field.name, recorded.as_str()) {
            ("args.backend", "null") => "chain_binomial".to_string(),
            (_, r) => r.to_string(),
        };
        used == expected
    }

    fn run(&self, invocation: &ProfileInvocation, store: Option<&Path>) -> store::Result<Observed> {
        run(invocation, store)
    }
}

/// The declared inputs the class check does not run, each with the reason.
const NOT_RUN: &[(&str, &str)] = &[(
    "args.from_manifest",
    "names a stored fit run, which this harness's fresh stores do not hold; `tests/from_manifest.rs` \
     (`from_manifest_names_the_result_its_handle_names`) checks that it names the result `--fit <run id>` names",
)];

/// Every declared input of `camdl profile` keeps the promise its class makes.
#[test]
fn every_profile_input_keeps_its_class() {
    let fields: Vec<FieldInfo> =
        ProfileInvocation::fields().iter().filter(|f| NOT_RUN.iter().all(|(n, _)| *n != f.name)).cloned().collect();
    assert_eq!(fields.len() + NOT_RUN.len(), ProfileInvocation::fields().len(), "every input the check skips is declared");
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

/// Every argument clap accepts for `profile`, and every global option, is a
/// declared input; and every declared input is one clap accepts.
#[test]
fn every_profile_argument_is_declared() {
    use clap::CommandFactory;
    let cmd = crate::Cli::command();
    let sub = cmd.find_subcommand("profile").expect("the profile subcommand");
    let mut accepted: std::collections::BTreeSet<String> =
        sub.get_arguments().map(|a| a.get_id().to_string()).filter(|id| id != "help").collect();
    accepted.extend(cmd.get_arguments().map(|a| a.get_id().to_string()).filter(|id| id != "help" && id != "version"));
    let declared: std::collections::BTreeSet<String> =
        ProfileInvocation::fields().iter().map(|f| f.name.rsplit('.').next().unwrap().to_string()).collect();
    assert_eq!(accepted, declared);
}

/// Each directory's entries under `dir`, with the 12 hex digits of each
/// suffix replaced by `*`, so the shape is compared and not the keys.
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

/// The tree a profile makes, as "Layouts, kind by kind" draws it: the build
/// directory, the compiled model and its quantities as content artifacts, the
/// fit problem the profile is of, stored as `fit run` stores it, and each
/// point under its problem, its focal values and its method, named by the
/// seed typed and the start's index.
#[test]
fn a_profile_makes_the_tree_of_the_proposal() {
    let cwd = workdir();
    let store = cwd.path().join("camdl-store");
    let args = argv(&profile_from(&["sir.ir.json", "--fit", "fit.toml", "--sweep", "beta_a=0.3",
        "--particles", "20", "--iterations", "2", "--n-starts", "2", "--seeds", "1,2"]));
    let out = camdl(cwd.path(), &store, &args, &[]);
    assert!(out.status.success(), "{}", String::from_utf8_lossy(&out.stderr));
    let mut got = Vec::new();
    shape(&store, 0, &mut got);
    let build = format!("{}-*", store::path_label(crate::version::VERSION_SHORT));
    let point = |name: &str| -> [String; 5] {
        [
            format!("          {name}-*"),
            "            identity.json".into(),
            "            mle.toml".into(),
            "            provenance.json".into(),
            "            run.json".into(),
        ]
    };
    let mut want: Vec<String> = [
        build.as_str(),
        "  compiled_model",
        "    sir-*",
        "      identity.json",
        "      model.ir.json",
        "      run.json",
        "  fit",
        "    sir-*",
        "      identity.json",
        "      inputs",
        "        cases.tsv",
        "      model.ir.json",
        "      partition.json",
        "      problem.toml",
        "      run.json",
        "  model_source",
        "    sir-*",
        "      declared.json",
        "      docs.json",
        "      identity.json",
        "      run.json",
        "      source.ir.json",
        "  profile_point",
        "    sir-*",
        "      beta_a_0.3000-*",
        "        if2-*",
    ].map(String::from).to_vec();
    for name in ["seed_1-start_0", "seed_1-start_1", "seed_2-start_0", "seed_2-start_1"] {
        want.extend(point(name));
    }
    want.extend(["  quantities", "    sir-*", "      identity.json", "      quantities.json", "      run.json", "store.json"].map(String::from));
    assert_eq!(got, want, "the store's tree:\n{}", got.join("\n"));

    // `fit run` of the same file finds the problem the profile stored.
    let problems = |store: &Path| -> Vec<Key> { observe(store).into_iter().filter(|r| r.kind == "fit").map(|r| r.run_id).collect() };
    let before = problems(&store);
    let fit = camdl(cwd.path(), &store, &["fit".into(), "run".into(), "fit.toml".into(), "--progress".into(), "none".into()], &[]);
    assert!(fit.status.success(), "{}", String::from_utf8_lossy(&fit.stderr));
    assert_eq!(problems(&store), before, "a profile and a fit of one file share one stored problem");
}

/// The points of `store`'s profile results, by run id, with their files.
fn points(store: &Path) -> BTreeMap<Key, BTreeMap<String, Key>> {
    observe(store).into_iter().filter(|r| r.kind == super::ProfilePoint::KIND).map(|r| (r.run_id, r.files)).collect()
}

/// A grid and a larger grid that holds it, into one store: the second
/// command finds every point of the first stored and computes only the
/// points the first did not ask for, and the first's points are untouched.
#[test]
fn a_larger_grid_computes_only_its_new_points() {
    let cwd = workdir();
    let store = cwd.path().join("camdl-store");
    let line = |grid: &str| argv(&profile_from(&["sir.ir.json", "--fit", "fit.toml", "--sweep", grid,
        "--particles", "20", "--iterations", "2", "--n-starts", "2", "--seeds", "1,2"]));
    let first = camdl(cwd.path(), &store, &line("beta_a=0.3,0.4"), &[]);
    assert!(first.status.success(), "{}", String::from_utf8_lossy(&first.stderr));
    let before = points(&store);
    assert_eq!(before.len(), 8, "two values by two seeds by two starts");
    let records: BTreeMap<PathBuf, Vec<u8>> = walk_files(&store, "run.json");
    assert!(records.len() >= 8, "every point's record is read: {}", records.len());

    let second = camdl(cwd.path(), &store, &line("beta_a=0.3,0.4,0.5"), &[]);
    let stderr = String::from_utf8_lossy(&second.stderr);
    assert!(second.status.success(), "{stderr}");
    assert!(stderr.contains("profile: 8 of 12 points already stored"), "{stderr}");
    let after = points(&store);
    assert_eq!(after.len(), 12);
    for (id, files) in &before {
        assert_eq!(after.get(id), Some(files), "a point of the smaller grid keeps its id and its files");
    }
    let outcomes: Vec<(String, String)> = last_invocation(&store)["results"].as_array().unwrap().iter()
        .filter(|r| r["kind"] == super::ProfilePoint::KIND)
        .map(|r| (r["attributes"]["focal"].as_str().unwrap().to_string(), r["outcome"].as_str().unwrap().to_string()))
        .collect();
    assert_eq!(outcomes.len(), 12);
    for (focal, outcome) in &outcomes {
        let want = if focal == "beta_a=0.5000" { "computed" } else { "reused" };
        assert_eq!(outcome, want, "{focal}: {outcomes:?}");
    }
    for (path, bytes) in &records {
        assert_eq!(&std::fs::read(path).unwrap(), bytes, "{} is not written again", path.display());
    }
    // Each command's table holds its own grid's rows.
    let rows = |out: &Output| String::from_utf8_lossy(&out.stdout).lines().count() - 1;
    assert_eq!((rows(&first), rows(&second)), (2, 3));
}

/// Every file named `name` under `root`, outside the store's own `.`
/// directories, with its bytes.
fn walk_files(root: &Path, name: &str) -> BTreeMap<PathBuf, Vec<u8>> {
    let mut out = BTreeMap::new();
    let mut stack = vec![root.to_path_buf()];
    while let Some(d) = stack.pop() {
        for e in std::fs::read_dir(&d).unwrap().flatten() {
            let p = e.path();
            if e.file_name().to_string_lossy().starts_with('.') {
                continue;
            }
            if p.is_dir() {
                stack.push(p);
            } else if e.file_name() == name {
                out.insert(p.clone(), std::fs::read(&p).unwrap());
            }
        }
    }
    out
}

/// A rerun is served from the store and prints the same table; `--force`
/// computes every point again, moving the stored ones to the quarantine, and
/// stores the same bytes.
#[test]
fn a_rerun_prints_the_same_table_and_force_computes_it_again() {
    let cwd = workdir();
    let store = cwd.path().join("camdl-store");
    let line = |extra: &[&str]| {
        let mut l = vec!["sir.ir.json", "--fit", "fit.toml", "--sweep", "beta_a=0.3,0.4", "--particles", "20",
            "--iterations", "2", "--n-starts", "1"];
        l.extend_from_slice(extra);
        argv(&profile_from(&l))
    };
    let first = camdl(cwd.path(), &store, &line(&[]), &[]);
    assert!(first.status.success(), "{}", String::from_utf8_lossy(&first.stderr));
    let again = camdl(cwd.path(), &store, &line(&[]), &[]);
    assert!(String::from_utf8_lossy(&again.stderr).contains("2 of 2 points already stored"));
    assert_eq!(again.stdout, first.stdout, "the same table");
    let before = points(&store);
    let forced = camdl(cwd.path(), &store, &line(&["--force"]), &[]);
    assert!(forced.status.success(), "{}", String::from_utf8_lossy(&forced.stderr));
    assert!(!String::from_utf8_lossy(&forced.stderr).contains("already stored"));
    assert_eq!(points(&store), before, "the same ids and bytes");
    assert_eq!(std::fs::read_dir(store.join(".quarantine")).map(|d| d.count()).unwrap_or(0), 2);
    assert_eq!(forced.stdout, first.stdout);
}

/// The run store is the one `fit run` would store the fit in: `--store`, else
/// the fit file's `store` key (relative to the file), else `CAMDL_STORE`.
#[test]
fn the_store_is_the_flag_then_the_fit_files_key_then_the_environment() {
    let cwd = workdir();
    let sub = cwd.path().join("sub");
    std::fs::create_dir(&sub).unwrap();
    let text = format!("store = \"st\"\n{}", fit_toml("beta_b = 0.2\ngamma = 0.2\n"))
        .replace("\"sir.ir.json\"", "\"../sir.ir.json\"")
        .replace("\"cases.tsv\"", "\"../cases.tsv\"");
    std::fs::write(sub.join("fit.toml"), text).unwrap();
    let env_store = cwd.path().join("env-store");
    let line = |extra: &[&str]| {
        let mut l = vec!["sir.ir.json", "--fit", "sub/fit.toml", "--sweep", "beta_a=0.3", "--particles", "20",
            "--iterations", "2", "--n-starts", "1"];
        l.extend_from_slice(extra);
        argv(&profile_from(&l))
    };
    let out = camdl(cwd.path(), &env_store, &line(&[]), &[]);
    assert!(out.status.success(), "{}", String::from_utf8_lossy(&out.stderr));
    assert_eq!(walk_files(&sub.join("st"), "mle.toml").len(), 1, "the file's store, beside the file");
    assert!(!env_store.exists(), "not CAMDL_STORE");
    let flag_store = cwd.path().join("flag-store");
    let out = camdl(cwd.path(), &env_store, &line(&["--store", "flag-store"]), &[]);
    assert!(out.status.success(), "{}", String::from_utf8_lossy(&out.stderr));
    assert_eq!(walk_files(&flag_store, "mle.toml").len(), 1, "--store");
    assert!(!env_store.exists(), "not CAMDL_STORE");
}

/// A fit file is read once: its absolute-path warning is printed once.
#[test]
fn a_fit_file_is_read_once() {
    let cwd = workdir();
    let store = cwd.path().join("camdl-store");
    let model = cwd.path().join("sir.ir.json");
    let text = fit_toml("beta_b = 0.2\ngamma = 0.2\n").replace("\"sir.ir.json\"", &format!("{:?}", model.display().to_string()));
    std::fs::write(cwd.path().join("abs.toml"), text).unwrap();
    let out = camdl(cwd.path(), &store, &argv(&profile_from(&["sir.ir.json", "--fit", "abs.toml", "--sweep", "beta_a=0.3",
        "--particles", "20", "--iterations", "2", "--n-starts", "1"])), &[]);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(out.status.success(), "{stderr}");
    assert_eq!(stderr.matches("[model] camdl is an absolute path").count(), 1, "{stderr}");
}

/// A value `--fixed` or `--fixed-file` writes over one the fit config gives is
/// reported on stderr and recorded in every point's `provenance.json`, in the
/// lines the other commands print (run spec §1.3), and so is a `--fixed` that
/// takes a parameter out of `[estimate]`.
#[test]
fn a_replaced_value_is_reported_and_recorded() {
    let cases: [(&[&str], &str); 4] = [
        (&["--fixed", "gamma=0.3"], "info: --param/--fixed replaced gamma = 0.2 (from fit.toml [fixed]) with 0.3"),
        (&["--fixed-file", "fixed2.toml"], "info: params file fixed2.toml replaced gamma = 0.2 (from fit.toml [fixed]) with 0.25"),
        (
            &["--fixed", "rho=0.4"],
            "info: rho: `--fixed rho=0.4` pins it, so it is not estimated (the fit config lists it in [estimate])",
        ),
        (
            &["--fixed-file", "pstart.toml"],
            "info: rho: `--fixed-file pstart.toml` pins it at 0.5, so it is not estimated (the fit config lists it in [estimate])",
        ),
    ];
    for (flags, line) in cases {
        let cwd = workdir();
        let store = cwd.path().join("camdl-store");
        let mut l = vec!["sir.ir.json", "--fit", "fit.toml", "--sweep", "beta_a=0.3", "--particles", "20",
            "--iterations", "2", "--n-starts", "1"];
        l.extend_from_slice(flags);
        let out = camdl(cwd.path(), &store, &argv(&profile_from(&l)), &[]);
        let stderr = String::from_utf8_lossy(&out.stderr);
        assert!(out.status.success(), "{stderr}");
        assert!(stderr.lines().any(|s| s == line), "{flags:?}: no `{line}` in\n{stderr}");
        let message = line.split_once(": ").unwrap().1;
        let provenance = walk_files(&store, crate::profile_store::PROVENANCE_FILE);
        assert_eq!(provenance.len(), 1, "{flags:?}");
        let record: serde_json::Value = serde_json::from_slice(provenance.values().next().unwrap()).unwrap();
        let recorded = record["provenance"]["notices"].as_array().into_iter().flatten().any(|n| n["message"] == message);
        assert!(recorded, "{flags:?}: `{message}` not in {}", record["provenance"]["notices"]);
    }
}

/// A fit's `mle_params.toml`, its `[provenance]` block included, is a
/// `--fixed-file`: each value it gives is held, out of `[estimate]`, and the
/// profile is of that conditional problem. The swept `beta_a` it also gives
/// is not held: the grid sets it, so neither stderr nor any point records it
/// as pinned at the file's value.
#[test]
fn a_fits_estimate_is_a_fixed_file() {
    let cwd = workdir();
    let store = cwd.path().join("camdl-store");
    let fit = camdl(cwd.path(), &store, &["fit", "run", "fit.toml", "--progress", "none"].map(String::from), &[]);
    assert!(fit.status.success(), "{}", String::from_utf8_lossy(&fit.stderr));
    let mle = walk_files(&store, "mle_params.toml");
    assert_eq!(mle.len(), 1, "{mle:?}");
    let text = String::from_utf8_lossy(mle.values().next().unwrap()).into_owned();
    assert!(text.contains("[provenance]"), "{text}");
    std::fs::write(cwd.path().join("mle.toml"), &text).unwrap();
    let out = camdl(cwd.path(), &store, &argv(&profile_from(&["sir.ir.json", "--fit", "fit.toml", "--sweep",
        "beta_a=0.3", "--particles", "20", "--iterations", "2", "--n-starts", "1", "--fixed-file", "mle.toml"])), &[]);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(out.status.success(), "{stderr}");
    let pinned = |name: &str| format!("info: {name}: `--fixed-file mle.toml` pins it at");
    assert!(stderr.lines().any(|l| l.starts_with(&pinned("rho"))
        && l.ends_with("so it is not estimated (the fit config lists it in [estimate])")), "no rho line in\n{stderr}");
    assert!(!stderr.lines().any(|l| l.contains("beta_a") && l.contains("pins it")), "beta_a is swept:\n{stderr}");
    let provenance = walk_files(&store, crate::profile_store::PROVENANCE_FILE);
    assert_eq!(provenance.len(), 1, "one point");
    let record: serde_json::Value = serde_json::from_slice(provenance.values().next().unwrap()).unwrap();
    let notices = record["provenance"]["notices"].to_string();
    assert!(notices.contains("rho: `--fixed-file mle.toml` pins it") && !notices.contains("beta_a: `--fixed-file"), "{notices}");
    let beta_a = &record["provenance"]["parameters_provenance"]["beta_a"];
    assert_ne!(beta_a["source"], "fixed_file", "{beta_a}");
}

/// A profile point that scores no finite log-likelihood is a completed
/// point holding `final_loglik = -inf`: here every filter evaluation trips a
/// budget of one particle-substep. A rerun finds it stored.
#[test]
fn a_point_with_no_finite_loglik_is_stored_completed() {
    let cwd = workdir();
    let store = cwd.path().join("camdl-store");
    let args = argv(&profile_from(&["sir.ir.json", "--fit", "fit.toml", "--sweep", "beta_a=0.3",
        "--particles", "20", "--iterations", "2", "--n-starts", "1", "--pf-max-substeps", "1"]));
    let out = camdl(cwd.path(), &store, &args, &[]);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(out.status.success(), "{stderr}");
    assert!(stderr.contains("non-finite"), "{stderr}");
    let mle = walk_files(&store, "mle.toml");
    assert_eq!(mle.len(), 1, "one point: {mle:?}");
    let text = String::from_utf8_lossy(mle.values().next().unwrap()).into_owned();
    assert!(text.contains("final_loglik = -inf"), "{text}");
    // The table's row names the stored point.
    let identity: serde_json::Value =
        serde_json::from_slice(&std::fs::read(mle.keys().next().unwrap().with_file_name("identity.json")).unwrap()).unwrap();
    let table = String::from_utf8_lossy(&out.stdout).into_owned();
    let row = table.lines().nth(1).unwrap_or_else(|| panic!("a row: {table}"));
    assert_eq!(row.rsplit('\t').next(), identity["run_id"].as_str(), "{table}");
    let again = camdl(cwd.path(), &store, &args, &[]);
    assert!(String::from_utf8_lossy(&again.stderr).contains("1 of 1 points already stored"));
}

/// The `mle.toml` digest of every point the last invocation in `store`
/// asked for, by its labels (`<focal> seed_<n> start_<i>`).
fn pinned_view(store: &Path) -> BTreeMap<String, String> {
    let files = points(store);
    last_invocation(store)["results"].as_array().unwrap().iter()
        .filter(|r| r["kind"] == super::ProfilePoint::KIND)
        .map(|r| {
            let l = &r["attributes"];
            let key = format!("{} seed_{} start_{}", l["focal"].as_str().unwrap(), l["seed"], l["start"]);
            let id = Key::from_hex(r["run_id"].as_str().unwrap()).unwrap();
            (key, files[&id][super::MLE_FILE].to_hex())
        })
        .collect()
}

/// Two deterministic (nl-*) profiles, whose cells draw no random number, and
/// the `mle.toml` each point stored before `profile` moved to the run store,
/// by platform: a point's seed changed with the move, and these cells read
/// none, so every point's bytes are as they were. One profiles a fit file
/// with `nl-sbplx`; the other profiles a fit file on a two-axis grid at two
/// seeds, with `nl-bobyqa` from a starts file, so its starts coincide and each
/// cell's four starts are one point per seed.
#[test]
fn the_deterministic_profiles_store_what_they_stored_before_the_move() {
    let fixtures: [(&str, &[&str]); 2] = [
        (
            "sbplx",
            &["weekly.camdl", "--fit", "weekly_fit.toml", "--sweep", "gamma=0.12,0.15,0.18", "--algorithm", "nl-sbplx",
                "--backend", "ode", "--n-starts", "2", "--seed", "3"],
        ),
        (
            "bobyqa",
            &["weekly.camdl", "--fit", "weekly_rho.toml", "--sweep", "beta=0.35,0.45", "--sweep", "gamma=0.12,0.15",
                "--starts", "from_params=weekly_start.toml", "--algorithm", "nl-bobyqa", "--backend", "ode",
                "--n-starts", "2", "--seeds", "1,2"],
        ),
    ];
    let mut got = BTreeMap::new();
    for (name, line) in fixtures {
        let cwd = workdir();
        let store = cwd.path().join("camdl-store");
        let mut args: Vec<String> = vec!["profile".into()];
        args.extend(line.iter().map(|s| s.to_string()));
        args.extend(["--progress".into(), "none".into()]);
        let out = camdl(cwd.path(), &store, &args, &[]);
        assert!(out.status.success(), "{name}: {}", String::from_utf8_lossy(&out.stderr));
        got.insert(name.to_string(), pinned_view(&store));
    }
    crate::test_support::assert_pinned(PINS, &got, "the points the deterministic profiles store moved");
}

use crate::test_support::Pins;

/// The pins of [`the_deterministic_profiles_store_what_they_stored_before_the_move`],
/// per `(target_os, target_arch)`.
const PINS: &[((&str, &str), Pins)] =
    &[(("macos", "aarch64"), MACOS_AARCH64), (("linux", "aarch64"), LINUX_AARCH64)];

/// macOS arm64: taken from the old store's points of the same fixtures, at
/// `78f32ece0`.
const MACOS_AARCH64: Pins = &[
    ("bobyqa", &[
        ("beta=0.3500__gamma=0.1200 seed_1 start_0", "691bb4d7d932044d553d467eb89f5151b556a4ec959074aa0649f876ba93a320"),
        ("beta=0.3500__gamma=0.1200 seed_1 start_1", "691bb4d7d932044d553d467eb89f5151b556a4ec959074aa0649f876ba93a320"),
        ("beta=0.3500__gamma=0.1200 seed_2 start_0", "691bb4d7d932044d553d467eb89f5151b556a4ec959074aa0649f876ba93a320"),
        ("beta=0.3500__gamma=0.1200 seed_2 start_1", "691bb4d7d932044d553d467eb89f5151b556a4ec959074aa0649f876ba93a320"),
        ("beta=0.3500__gamma=0.1500 seed_1 start_0", "00e7d2868c226e27e88eed2bf647dc3eb7976483d792daa09d87a86e432dfb41"),
        ("beta=0.3500__gamma=0.1500 seed_1 start_1", "00e7d2868c226e27e88eed2bf647dc3eb7976483d792daa09d87a86e432dfb41"),
        ("beta=0.3500__gamma=0.1500 seed_2 start_0", "00e7d2868c226e27e88eed2bf647dc3eb7976483d792daa09d87a86e432dfb41"),
        ("beta=0.3500__gamma=0.1500 seed_2 start_1", "00e7d2868c226e27e88eed2bf647dc3eb7976483d792daa09d87a86e432dfb41"),
        ("beta=0.4500__gamma=0.1200 seed_1 start_0", "256698f47d7430579060a2950f70588185f3f9f9bdcd325e39a15346d8721ecf"),
        ("beta=0.4500__gamma=0.1200 seed_1 start_1", "256698f47d7430579060a2950f70588185f3f9f9bdcd325e39a15346d8721ecf"),
        ("beta=0.4500__gamma=0.1200 seed_2 start_0", "256698f47d7430579060a2950f70588185f3f9f9bdcd325e39a15346d8721ecf"),
        ("beta=0.4500__gamma=0.1200 seed_2 start_1", "256698f47d7430579060a2950f70588185f3f9f9bdcd325e39a15346d8721ecf"),
        ("beta=0.4500__gamma=0.1500 seed_1 start_0", "a7976e9e659725f5fd441668c07d9dfa16a53a8b8d399d957843cd7945ce881c"),
        ("beta=0.4500__gamma=0.1500 seed_1 start_1", "a7976e9e659725f5fd441668c07d9dfa16a53a8b8d399d957843cd7945ce881c"),
        ("beta=0.4500__gamma=0.1500 seed_2 start_0", "a7976e9e659725f5fd441668c07d9dfa16a53a8b8d399d957843cd7945ce881c"),
        ("beta=0.4500__gamma=0.1500 seed_2 start_1", "a7976e9e659725f5fd441668c07d9dfa16a53a8b8d399d957843cd7945ce881c"),
    ]),
    ("sbplx", &[
        ("gamma=0.1200 seed_3 start_0", "3c340f8e2b3509c292cad86dc52c44162b6f233e4d2ae042b6f15fb36e4622d6"),
        ("gamma=0.1200 seed_3 start_1", "00ff697b29fa02b31a7fb2f3657eba724c5b1b82f40a59e3b88e2b520b611dae"),
        ("gamma=0.1500 seed_3 start_0", "d7933595634e1f26b743bdbf26cfc0161ee2bd35102abfdb8a863e9037f36131"),
        ("gamma=0.1500 seed_3 start_1", "6c37c7171f927e494d1af463300371101da044481c49f79d75f99d22bd7d07ba"),
        ("gamma=0.1800 seed_3 start_0", "8e0836d7dd6eaf10cb515761c05c973fe13a35c261a8d8bb1d371a54fbde17e8"),
        ("gamma=0.1800 seed_3 start_1", "383a24337d84a947ad1767db0a77dbc59f3f937a4875528a0ae924d74d42d31c"),
    ]),
];

/// linux/aarch64: taken from the old store's points of the same fixtures, at
/// `78f32ece0` on linux/aarch64, through the capture commit `eaa82d730`.
const LINUX_AARCH64: Pins = &[
    ("bobyqa", &[
        ("beta=0.3500__gamma=0.1200 seed_1 start_0", "691bb4d7d932044d553d467eb89f5151b556a4ec959074aa0649f876ba93a320"),
        ("beta=0.3500__gamma=0.1200 seed_1 start_1", "691bb4d7d932044d553d467eb89f5151b556a4ec959074aa0649f876ba93a320"),
        ("beta=0.3500__gamma=0.1200 seed_2 start_0", "691bb4d7d932044d553d467eb89f5151b556a4ec959074aa0649f876ba93a320"),
        ("beta=0.3500__gamma=0.1200 seed_2 start_1", "691bb4d7d932044d553d467eb89f5151b556a4ec959074aa0649f876ba93a320"),
        ("beta=0.3500__gamma=0.1500 seed_1 start_0", "00e7d2868c226e27e88eed2bf647dc3eb7976483d792daa09d87a86e432dfb41"),
        ("beta=0.3500__gamma=0.1500 seed_1 start_1", "00e7d2868c226e27e88eed2bf647dc3eb7976483d792daa09d87a86e432dfb41"),
        ("beta=0.3500__gamma=0.1500 seed_2 start_0", "00e7d2868c226e27e88eed2bf647dc3eb7976483d792daa09d87a86e432dfb41"),
        ("beta=0.3500__gamma=0.1500 seed_2 start_1", "00e7d2868c226e27e88eed2bf647dc3eb7976483d792daa09d87a86e432dfb41"),
        ("beta=0.4500__gamma=0.1200 seed_1 start_0", "256698f47d7430579060a2950f70588185f3f9f9bdcd325e39a15346d8721ecf"),
        ("beta=0.4500__gamma=0.1200 seed_1 start_1", "256698f47d7430579060a2950f70588185f3f9f9bdcd325e39a15346d8721ecf"),
        ("beta=0.4500__gamma=0.1200 seed_2 start_0", "256698f47d7430579060a2950f70588185f3f9f9bdcd325e39a15346d8721ecf"),
        ("beta=0.4500__gamma=0.1200 seed_2 start_1", "256698f47d7430579060a2950f70588185f3f9f9bdcd325e39a15346d8721ecf"),
        ("beta=0.4500__gamma=0.1500 seed_1 start_0", "a7976e9e659725f5fd441668c07d9dfa16a53a8b8d399d957843cd7945ce881c"),
        ("beta=0.4500__gamma=0.1500 seed_1 start_1", "a7976e9e659725f5fd441668c07d9dfa16a53a8b8d399d957843cd7945ce881c"),
        ("beta=0.4500__gamma=0.1500 seed_2 start_0", "a7976e9e659725f5fd441668c07d9dfa16a53a8b8d399d957843cd7945ce881c"),
        ("beta=0.4500__gamma=0.1500 seed_2 start_1", "a7976e9e659725f5fd441668c07d9dfa16a53a8b8d399d957843cd7945ce881c"),
    ]),
    ("sbplx", &[
        ("gamma=0.1200 seed_3 start_0", "3c340f8e2b3509c292cad86dc52c44162b6f233e4d2ae042b6f15fb36e4622d6"),
        ("gamma=0.1200 seed_3 start_1", "00ff697b29fa02b31a7fb2f3657eba724c5b1b82f40a59e3b88e2b520b611dae"),
        ("gamma=0.1500 seed_3 start_0", "d7933595634e1f26b743bdbf26cfc0161ee2bd35102abfdb8a863e9037f36131"),
        ("gamma=0.1500 seed_3 start_1", "6c37c7171f927e494d1af463300371101da044481c49f79d75f99d22bd7d07ba"),
        ("gamma=0.1800 seed_3 start_0", "8e0836d7dd6eaf10cb515761c05c973fe13a35c261a8d8bb1d371a54fbde17e8"),
        ("gamma=0.1800 seed_3 start_1", "383a24337d84a947ad1767db0a77dbc59f3f937a4875528a0ae924d74d42d31c"),
    ]),
];

/// `camdl list` shows a profile as its invocation, `camdl cat @L` renders the
/// profile table the command printed, and `camdl show` and `camdl cat` of a
/// point's id read that point.
#[test]
fn list_show_and_cat_read_the_stored_profile() {
    let cwd = workdir();
    let store = cwd.path().join("camdl-store");
    let args = argv(&profile_from(&["sir.ir.json", "--fit", "fit.toml", "--sweep", "beta_a=0.3,0.4",
        "--particles", "20", "--iterations", "2", "--n-starts", "1", "--label", "prof"]));
    let run = camdl(cwd.path(), &store, &args, &[]);
    assert!(run.status.success(), "{}", String::from_utf8_lossy(&run.stderr));
    let read = |line: &[&str]| {
        let out = camdl(cwd.path(), &store, &line.iter().map(|s| s.to_string()).collect::<Vec<_>>(), &[]);
        assert!(out.status.success(), "{line:?}: {}", String::from_utf8_lossy(&out.stderr));
        String::from_utf8(out.stdout).unwrap()
    };
    let listed: Vec<serde_json::Value> = read(&["list", "--format", "json", "--all"]).lines()
        .filter_map(|l| serde_json::from_str(l).ok())
        .filter(|v: &serde_json::Value| v["kind"] == "profile")
        .collect();
    assert_eq!(listed.len(), 1, "{listed:?}");
    assert_eq!(listed[0]["label"], "prof");
    assert_eq!(listed[0]["points"].as_array().unwrap().len(), 2);
    assert_eq!(read(&["cat", "@prof"]), String::from_utf8(run.stdout).unwrap(), "the same table");
    let id = listed[0]["points"][0].as_str().unwrap()[..12].to_string();
    assert!(read(&["show", &id]).contains("final_loglik"));
    assert!(read(&["cat", &id]).starts_with("# Per-start MLE"));
}

/// `--fit` names a fit whose method drew its chain-binomial binomials with
/// `btrs`. Each IF2 cell runs on the sampler the fit ran, so the points are
/// the ones `--binomial btrs` names, not the default `btpe` ones.
#[test]
fn a_fit_files_binomial_sampler_is_the_profiles() {
    let ids = |fit: &str, extra: &[&str]| -> Vec<Key> {
        let cwd = workdir();
        let store = cwd.path().join("camdl-store");
        let mut line = vec!["sir.ir.json", "--fit", fit, "--sweep", "beta_a=0.3", "--particles", "20",
            "--iterations", "2", "--n-starts", "1", "--seeds", "1"];
        line.extend_from_slice(extra);
        let out = camdl(cwd.path(), &store, &argv(&profile_from(&line)), &[]);
        assert!(out.status.success(), "{}", String::from_utf8_lossy(&out.stderr));
        points(&store).into_keys().collect()
    };
    let from_fit = ids("fit_btrs.toml", &[]);
    assert_eq!(from_fit.len(), 1, "{from_fit:?}");
    assert_eq!(from_fit, ids("fit_btrs.toml", &["--binomial", "btrs"]), "the fit's sampler, btrs");
    assert_ne!(from_fit, ids("fit.toml", &[]), "not the default sampler");
}

/// An `nl-lbfgs` profile on the ODE backend compiles the state Jacobian of
/// the model file the profile is given, so a model that reads a table by a
/// path relative to its own file profiles: the table is beside that file,
/// and not beside the copy the stored problem keeps.
#[test]
fn an_nl_lbfgs_profile_compiles_the_model_file_it_is_given() {
    let cwd = workdir();
    let store = cwd.path().join("camdl-store");
    std::fs::create_dir_all(cwd.path().join("data")).unwrap();
    std::fs::write(cwd.path().join("data/w.tsv"), "g\tw\na\t1.0\nb\t2.0\n").unwrap();
    std::fs::write(
        cwd.path().join("rel.camdl"),
        "time_unit = 'days\ndimensions { g = [a, b] }\ncompartments { S, I, R }\n\
         tables { w : g = read(\"data/w.tsv\") }\n\
         parameters {\n  beta  : rate  in [0.05, 1.0]\n  gamma : rate  in [0.01, 0.5]\n  N0    : count\n  I0    : count\n}\n\
         let N = S + I + R\n\
         transitions {\n  infection : S --> I  @ beta * w[a] * S * I / N\n  recovery  : I --> R  @ gamma * I\n}\n\
         init { S = N0 - I0  I = I0 }\n\
         observations {\n  cases {\n    columns       { time : time, cases : count }\n    covers        = closing_at(time, 1 'days)\n    \
         projected     = incidence(infection)\n    emit_schedule = every 1 'days\n    cases ~ poisson(rate = projected)\n  }\n}\n\
         simulate { from = 0 'days  to = 20 'days }\n",
    )
    .unwrap();
    std::fs::write(
        cwd.path().join("ode.toml"),
        "[model]\ncamdl = \"rel.camdl\"\n[data.observations]\ncases = \"cases.tsv\"\n\
         [estimate]\nbeta = { bounds = [0.1, 0.9], start = 0.4 }\ngamma = { bounds = [0.05, 0.4], start = 0.2 }\n\
         [fixed]\nN0 = 1000\nI0 = 5\n[method]\nalgorithm = \"nl-lbfgs\"\nbackend = \"ode\"\nchains = 1\n",
    )
    .unwrap();
    let run = |args: &[&str]| camdl(cwd.path(), &store, &args.iter().map(|s| s.to_string()).collect::<Vec<_>>(), &[]);
    let fit = run(&["fit", "run", "ode.toml", "--seed", "1", "--label", "o", "--progress", "none"]);
    assert!(fit.status.success(), "the fit: {}", String::from_utf8_lossy(&fit.stderr));
    let profile = run(&[
        "profile", "rel.camdl", "--fit", "@o", "--sweep", "beta=lin(0.3,0.6,2)", "--algorithm", "nl-lbfgs",
        "--backend", "ode", "--n-starts", "1", "--progress", "none",
    ]);
    assert!(profile.status.success(), "the profile: {}", String::from_utf8_lossy(&profile.stderr));
    assert_eq!(points(&store).len(), 2, "one point per grid value");

    // A file that differs from the fitted one only in its description is the
    // same model, so its Jacobian compile is the same compile.
    let source = std::fs::read_to_string(cwd.path().join("rel.camdl")).unwrap();
    std::fs::write(cwd.path().join("described.camdl"), format!("description = \"An SIR model.\"\n{source}")).unwrap();
    let described = run(&[
        "profile", "described.camdl", "--fit", "@o", "--sweep", "beta=lin(0.3,0.6,2)", "--algorithm", "nl-lbfgs",
        "--backend", "ode", "--n-starts", "1", "--progress", "none",
    ]);
    assert!(described.status.success(), "the described profile: {}", String::from_utf8_lossy(&described.stderr));
    assert_eq!(points(&store).len(), 2, "the same points");
}
