//! `--no-licm` (compile without loop-invariant code motion) changes the IR the
//! compiler emits, so every compile a command makes must run under it when it
//! is given, and none when it is not. Each command takes its compile settings
//! as a value from `main` (`util::CompileSettings`) and passes them to every
//! compile it makes; nothing reads a process-wide setting (proposal
//! 2026-11-17, phase 3).
//!
//! Each case runs a command with a logging `camdlc` shim first on `PATH`: the
//! shim records, for every compiler invocation but the version probe, whether
//! it was told to disable LICM (`CAMDL_NO_LICM=1`, how camdl tells camdlc),
//! then runs the real compiler. A case passes when the command compiled at
//! least once, every compile under `--no-licm` was told, and no compile
//! without it was. The IR cache is off, so every compile reaches the shim.
//!
//! `simulate`, `fit run`, `pfilter` and `profile` are covered by
//! their input-class tests (`keyed_no_licm`), which show the run identity
//! move with the flag.

mod fit_store_support;

use std::path::{Path, PathBuf};
use std::process::{Command, Output};
use std::sync::OnceLock;

fn binary() -> PathBuf {
    let bin = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../target/release/camdl");
    assert!(bin.exists(), "release camdl binary missing: {} — run `make build`", bin.display());
    bin
}

fn real_camdlc() -> PathBuf {
    let cc = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../../ocaml/_build/default/bin/camdlc.exe");
    assert!(cc.exists(), "camdlc not built at {} — run `make build-ocaml`", cc.display());
    cc
}

const MODEL: &str = r#"time_unit = 'days

compartments { S, I, R }

parameters {
  beta  : rate         in [0.05, 1.0]  ~ log_normal(mu = -1.0, sigma = 0.5)
  gamma : rate         in [0.01, 0.5]  ~ log_normal(mu = -2.0, sigma = 0.5)
  N0    : count
  I0    : count
  rho   : probability  in [0.1, 0.9]
  k     : positive     in [1.0, 100.0]
}

let N = S + I + R

transitions {
  infection : S --> I  @ beta * S * I / N
  recovery  : I --> R  @ gamma * I
}

init {
  S = N0 - I0
  I = I0
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

const DATA: &str = "time\tweekly_cases\n7\t16\n14\t166\n21\t626\n28\t1303\n35\t1260\n42\t1023\n49\t327\n56\t91\n";

fn fit_toml() -> String {
    r#"[model]
camdl = "model.camdl"

[data.observations]
weekly_cases = "weekly_cases.tsv"

[estimate]
beta  = { bounds = [0.05, 1.0], start = 0.4 }
gamma = { bounds = [0.01, 0.5], start = 0.15 }

[fixed]
N0  = 10000
I0  = 10
rho = 0.6
k   = 10.0

[method]
algorithm = "mh"
backend = "ode"
chains = 1
iterations = 40
burn_in = 10
thin = 1
"#.to_string()
}

/// The project every case runs in: the model, its data, two fit configs, a
/// reporting vocabulary, parameters, and one fit of `fit.toml` (made once per
/// test binary, before any shim is in place). Each case gets its own copy of
/// the directory, so cases do not share a store or a log.
struct Project {
    _tmp: tempfile::TempDir,
    root: PathBuf,
}

fn project() -> &'static Project {
    static P: OnceLock<Project> = OnceLock::new();
    P.get_or_init(|| {
        let tmp = tempfile::tempdir().unwrap();
        let root = tmp.path().to_path_buf();
        std::fs::write(root.join("model.camdl"), MODEL).unwrap();
        std::fs::write(root.join("other.camdl"), MODEL.replace("to   = 56 'days", "to   = 49 'days")).unwrap();
        std::fs::write(root.join("weekly_cases.tsv"), DATA).unwrap();
        std::fs::write(root.join("fit.toml"), fit_toml()).unwrap();
        std::fs::write(root.join("other.toml"), fit_toml().replace("model.camdl", "other.camdl"))
            .unwrap();
        std::fs::write(root.join("vocab.camdl"), "quantities {\n  attack_rate = final((N0 - S) / N0)\n}\n").unwrap();
        std::fs::write(root.join("p.toml"), "beta = 0.3\ngamma = 0.1\nN0 = 10000\nI0 = 10\nrho = 0.6\nk = 10\n")
            .unwrap();
        let o = Command::new(binary())
            .env_remove("CLICOLOR_FORCE")
            .env_remove("CAMDL_NO_LICM")
            .env("CAMDL_SKIP_VERSION_CHECK", "1")
            .env("CAMDL_STORE", std::path::Path::new(&root.join("store")).join("camdl-store"))
            .current_dir(&root)
            .args(["fit", "run", "fit.toml", "--seed", "1"])
            .output()
            .expect("spawn camdl");
        assert!(o.status.success(), "fit run:\n{}", String::from_utf8_lossy(&o.stderr));
        Project { _tmp: tmp, root }
    })
}

/// The fit's run folder and its problem folder, in the store under `root`.
fn fit_leaf(root: &Path) -> (PathBuf, PathBuf) {
    let run = fit_store_support::the_run(&root.join("store/camdl-store"));
    let problem = fit_store_support::problem_of(&run);
    (run, problem)
}

/// A copy of the project with a logging shim; returns the work dir.
struct Case {
    _tmp: tempfile::TempDir,
    dir: PathBuf,
    log: PathBuf,
    shim_dir: PathBuf,
}

fn case() -> Case {
    let tmp = tempfile::tempdir().unwrap();
    let dir = tmp.path().join("work");
    // A copied fit keeps relative paths only within the project; the stored
    // config names the model by an absolute path into the original project,
    // which is unchanged, so the copy is read as the original is.
    fit_store_support::copy_tree(&project().root, &dir);
    let shim_dir = tmp.path().join("shim");
    std::fs::create_dir_all(&shim_dir).unwrap();
    let log = tmp.path().join("compiles.log");
    let shim = shim_dir.join("camdlc");
    std::fs::write(
        &shim,
        format!(
            "#!/bin/sh\ncase \"$1\" in\n  --camdl-version) ;;\n  *) echo \"no_licm=${{CAMDL_NO_LICM:-0}} $*\" >> '{log}' ;;\nesac\nexec '{real}' \"$@\"\n",
            log = log.display(),
            real = real_camdlc().display(),
        ),
    )
    .unwrap();
    use std::os::unix::fs::PermissionsExt;
    std::fs::set_permissions(&shim, std::fs::Permissions::from_mode(0o755)).unwrap();
    Case { _tmp: tmp, dir, log, shim_dir }
}

impl Case {
    fn run(&self, args: &[&str]) -> (Output, Vec<String>) {
        let _ = std::fs::remove_file(&self.log);
        let path = format!("{}:{}", self.shim_dir.display(), std::env::var("PATH").unwrap_or_default());
        let o = Command::new(binary())
            .env_remove("CLICOLOR_FORCE")
            .env_remove("CAMDL_NO_LICM")
            .env_remove("CAMDLC")
            .env("CAMDL_SKIP_VERSION_CHECK", "1")
            .env("CAMDL_NO_IR_CACHE", "1")
            .env("CAMDL_STORE", std::path::Path::new(&self.dir.join("store")).join("camdl-store"))
            .env("PATH", path)
            .current_dir(&self.dir)
            .args(args)
            .output()
            .expect("spawn camdl");
        let log = std::fs::read_to_string(&self.log).unwrap_or_default();
        (o, log.lines().map(str::to_string).collect())
    }
}

fn text(o: &Output) -> String {
    format!(
        "status={:?}\nstdout:\n{}\nstderr:\n{}",
        o.status,
        String::from_utf8_lossy(&o.stdout),
        String::from_utf8_lossy(&o.stderr)
    )
}

/// Run `args` in `c` without and then with `--no-licm` (given before the
/// subcommand, as a global flag) and check what each compile was told.
fn assert_every_compile_follows_the_flag_in(c: &Case, name: &str, args: &[&str]) {
    let (o, compiles) = c.run(args);
    assert!(o.status.success(), "{name}: {}", text(&o));
    assert!(!compiles.is_empty(), "{name}: compiled nothing, so the case shows nothing\n{}", text(&o));
    assert!(
        compiles.iter().all(|l| l.starts_with("no_licm=0")),
        "{name}: without --no-licm no compile is told to disable LICM: {compiles:#?}"
    );
    let mut flagged = vec!["--no-licm"];
    flagged.extend_from_slice(args);
    let (o, compiles) = c.run(&flagged);
    assert!(o.status.success(), "{name} --no-licm: {}", text(&o));
    assert!(!compiles.is_empty(), "{name} --no-licm: compiled nothing\n{}", text(&o));
    assert!(
        compiles.iter().all(|l| l.starts_with("no_licm=1")),
        "{name}: under --no-licm every compile is told to disable LICM: {compiles:#?}"
    );
}

fn assert_every_compile_follows_the_flag(name: &str, args: &[&str]) {
    assert_every_compile_follows_the_flag_in(&case(), name, args);
}

/// The fit's run folder, relative to the project.
fn leaf_arg() -> String {
    let (leaf, _) = fit_leaf(&project().root);
    leaf.strip_prefix(&project().root).unwrap().to_string_lossy().into_owned()
}

#[test]
fn dev_eval() {
    assert_every_compile_follows_the_flag(
        "dev eval",
        &["dev", "eval", "model.camdl", "--params", "p.toml", "--expr", "beta", "--at", "0"],
    );
}

#[test]
fn diff() {
    assert_every_compile_follows_the_flag("diff", &["diff", "model.camdl", "model.camdl"]);
}

/// The compiler subcommands camdl passes through to camdlc.
#[test]
fn check_and_dev_compile() {
    assert_every_compile_follows_the_flag("check", &["check", "model.camdl"]);
    assert_every_compile_follows_the_flag("dev compile", &["dev", "compile", "model.camdl"]);
}

#[test]
fn fit_diff() {
    assert_every_compile_follows_the_flag("fit diff", &["fit", "diff", "fit.toml", "other.toml"]);
}

/// `camdl predict` replays the archived model; a reporting vocabulary is
/// compiled against the model source, under the flag.
#[test]
fn fit_predict_with_a_vocabulary() {
    assert_every_compile_follows_the_flag(
        "predict --quantities",
        &["predict", &leaf_arg(), "--mode", "free_forward", "--quantities", "vocab.camdl", "--n-draws", "2"],
    );
}

/// `simulate --draws prior --fit` compiles the model, and the fit config's
/// model to check it is the same one: both under the flag, so the two agree.
#[test]
fn simulate_replaying_a_fit_config() {
    assert_every_compile_follows_the_flag(
        "simulate --draws prior --fit",
        &["simulate", "model.camdl", "--draws", "prior", "--fit", "fit.toml", "--n-draws", "2", "--seed", "1"],
    );
}

#[test]
fn mre_simulate_and_fit() {
    assert_every_compile_follows_the_flag(
        "mre simulate",
        &["mre", "simulate", "model.camdl", "--params", "p.toml", "-b", "s.mre.tar.gz"],
    );
    assert_every_compile_follows_the_flag("mre fit", &["mre", "fit", "fit.toml", "-b", "f.mre.tar.gz"]);
}

/// The reproduce command an `mre` bundle carries runs the bundled command
/// under the compile settings the bundle was made under: `--no-licm` once, as
/// a global flag, wherever on the command line it was given.
#[test]
fn mre_reproduce_carries_no_licm() {
    let c = case();
    let reproduce = |args: &[&str]| -> String {
        let (o, _) = c.run(args);
        assert!(o.status.success(), "{}", text(&o));
        let all = format!("{}{}", String::from_utf8_lossy(&o.stdout), String::from_utf8_lossy(&o.stderr));
        all.lines()
            .find_map(|l| l.trim().strip_prefix("reproduce: ").map(str::to_string))
            .unwrap_or_else(|| panic!("no reproduce line:\n{}", text(&o)))
    };
    assert_eq!(reproduce(&["mre", "fit", "fit.toml", "-b", "f1.tgz"]), "camdl fit run fit.toml");
    assert_eq!(reproduce(&["--no-licm", "mre", "fit", "fit.toml", "-b", "f2.tgz"]), "camdl --no-licm fit run fit.toml");
    assert_eq!(
        reproduce(&["mre", "simulate", "model.camdl", "--params", "p.toml", "-b", "s1.tgz"]),
        "camdl simulate model.camdl --params p.toml"
    );
    for args in [
        &["--no-licm", "mre", "simulate", "model.camdl", "--params", "p.toml", "-b", "s2.tgz"][..],
        &["mre", "simulate", "model.camdl", "--no-licm", "--params", "p.toml", "-b", "s3.tgz"][..],
    ] {
        assert_eq!(reproduce(args), "camdl --no-licm simulate model.camdl --params p.toml", "{args:?}");
    }
}
