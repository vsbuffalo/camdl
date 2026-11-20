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
//! `simulate`, `fit run`, `pfilter`, `profile` and `survey` are covered by
//! their input-class tests (`keyed_no_licm`), which show the run identity
//! move with the flag.

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

fn fit_toml(output_dir: &str) -> String {
    format!(
        r#"output_dir = "{output_dir}"

[model]
camdl = "model.camdl"

[data.observations]
weekly_cases = "weekly_cases.tsv"

[estimate]
beta  = {{ bounds = [0.05, 1.0], start = 0.4 }}
gamma = {{ bounds = [0.01, 0.5], start = 0.15 }}

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
"#
    )
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
        std::fs::write(root.join("fit.toml"), fit_toml("results")).unwrap();
        std::fs::write(root.join("other.toml"), fit_toml("other_results").replace("model.camdl", "other.camdl"))
            .unwrap();
        std::fs::write(root.join("vocab.camdl"), "quantities {\n  attack_rate = final((N0 - S) / N0)\n}\n").unwrap();
        std::fs::write(root.join("p.toml"), "beta = 0.3\ngamma = 0.1\nN0 = 10000\nI0 = 10\nrho = 0.6\nk = 10\n")
            .unwrap();
        let o = Command::new(binary())
            .env_remove("CLICOLOR_FORCE")
            .env_remove("CAMDL_NO_LICM")
            .env("CAMDL_SKIP_VERSION_CHECK", "1")
            .env("CAMDL_OUTPUT_DIR", root.join("store"))
            .current_dir(&root)
            .args(["fit", "run", "fit.toml", "--seed", "1"])
            .output()
            .expect("spawn camdl");
        assert!(o.status.success(), "fit run:\n{}", String::from_utf8_lossy(&o.stderr));
        Project { _tmp: tmp, root }
    })
}

/// The fit's method leaf and its problem folder.
fn fit_leaf(root: &Path) -> (PathBuf, PathBuf) {
    let fits = root.join("results/fits");
    let segment = std::fs::read_dir(&fits).unwrap().flatten().map(|e| e.path()).find(|p| p.is_dir()).unwrap();
    let method = std::fs::read_dir(&segment)
        .unwrap()
        .flatten()
        .map(|e| e.path())
        .find(|p| p.file_name().unwrap().to_string_lossy().starts_with("mh-"))
        .unwrap();
    let leaf = std::fs::read_dir(&method).unwrap().flatten().map(|e| e.path()).find(|p| p.join("run.json").is_file()).unwrap();
    (leaf, segment)
}

fn copy_dir(from: &Path, to: &Path) {
    std::fs::create_dir_all(to).unwrap();
    for e in std::fs::read_dir(from).unwrap().flatten() {
        let (src, dst) = (e.path(), to.join(e.file_name()));
        if src.is_dir() {
            copy_dir(&src, &dst);
        } else {
            std::fs::copy(&src, &dst).unwrap();
        }
    }
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
    copy_dir(&project().root, &dir);
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
            .env("CAMDL_OUTPUT_DIR", self.dir.join("store"))
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

/// The fit's method leaf, relative to the project.
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

#[test]
fn fit_new() {
    let c = case();
    for (flag, dest, want) in [(None, "a.toml", "no_licm=0"), (Some("--no-licm"), "b.toml", "no_licm=1")] {
        let mut args: Vec<&str> = flag.into_iter().collect();
        args.extend(["fit", "new", "--from", "fit.toml", dest]);
        let (o, compiles) = c.run(&args);
        assert!(o.status.success(), "fit new: {}", text(&o));
        assert!(!compiles.is_empty(), "fit new compiled nothing:\n{}", text(&o));
        assert!(compiles.iter().all(|l| l.starts_with(want)), "fit new {flag:?}: {compiles:#?}");
    }
}

#[test]
fn fit_summary() {
    assert_every_compile_follows_the_flag("fit summary", &["fit", "summary", &leaf_arg()]);
}

/// `fit predict` replays the archived model; a reporting vocabulary is
/// compiled against the model source, under the flag.
#[test]
fn fit_predict_with_a_vocabulary() {
    assert_every_compile_follows_the_flag(
        "fit predict --quantities",
        &["fit", "predict", &leaf_arg(), "--mode", "free_forward", "--quantities", "vocab.camdl", "--n-draws", "2"],
    );
}

/// `compare` scores a fit on the model it archived, which needs no compile.
/// A fit whose archive is missing is scored on its config's model source,
/// compiled once under the flag.
#[test]
fn compare_of_a_fit_without_its_archive() {
    let c = case();
    let (leaf, segment) = fit_leaf(&c.dir);
    std::fs::remove_file(segment.join("model.ir.json")).unwrap();
    let leaf = leaf.to_string_lossy().into_owned();
    assert_every_compile_follows_the_flag_in(&c, "compare", &["compare", &leaf, &leaf]);
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
