//! gh#896 — a fit handle names one run, and a label never moves.
//!
//! A problem folder (`results/fits/<stem>-<h8>/`) holds every run of one
//! problem: rerunning a file with a changed `[method]` setting writes a second
//! leaf beside the first. These tests pin what each handle names in that
//! store: a leaf directory, a label, a `fit.toml`, and a problem folder that
//! holds several runs.
//!
//! End to end because the defects are process exit codes, what the store
//! records, and which run a verb reads — none of which a unit test over one
//! resolver observes. Deliberately cheap: the SEIR golden IR, one estimated
//! parameter, IF2 at 2 chains × 10 particles × 2 iterations.

use std::path::{Path, PathBuf};
use std::process::{Command, Output};

fn binary() -> PathBuf {
    let manifest = std::env::var("CARGO_MANIFEST_DIR").unwrap();
    Path::new(&manifest).join("../../target/release/camdl")
}

fn camdlc_bin() -> PathBuf {
    let manifest = std::env::var("CARGO_MANIFEST_DIR").unwrap();
    Path::new(&manifest).join("../../../ocaml/_build/default/bin/camdlc.exe")
}

fn golden_ir() -> PathBuf {
    let manifest = std::env::var("CARGO_MANIFEST_DIR").unwrap();
    Path::new(&manifest).join("../../../ocaml/golden/seir_observations.ir.json")
}

/// A scratch project: one data file, configs written on demand, every command
/// run from the project directory with the store at `results/`.
struct Project {
    tmp: tempfile::TempDir,
}

impl Project {
    fn new() -> Project {
        assert!(binary().exists(), "release camdl missing: {} — run `make build-rust`",
            binary().display());
        assert!(camdlc_bin().exists(), "camdlc.exe missing: {} — run `make build-ocaml`",
            camdlc_bin().display());
        let tmp = tempfile::tempdir().unwrap();
        std::fs::write(
            tmp.path().join("obs.tsv"),
            "time\tweekly_cases\n7\t1\n14\t2\n21\t3\n28\t4\n35\t5\n",
        )
        .unwrap();
        Project { tmp }
    }

    fn dir(&self) -> &Path {
        self.tmp.path()
    }

    fn fits(&self) -> PathBuf {
        self.dir().join("results").join("fits")
    }

    /// Write `name` holding the shared problem and `method` as its `[method]`.
    fn config(&self, name: &str, method: &str) {
        let text = format!(
            r#"
[model]
camdl = "{ir}"

[data.observations]
weekly_cases = "obs.tsv"

[estimate.beta]
bounds = [0.01, 0.5]
start  = 0.123
prior  = {{ log_normal = {{ mu = -2.0, sigma = 0.5 }} }}

[fixed]
sigma    = 0.25
gamma    = 0.3
rho      = 0.5
k        = 10.0
p_detect = 0.5
N0       = 1000
I0       = 1

[method]
{method}

[config]
dt = 1.0
"#,
            ir = golden_ir().display(),
        );
        std::fs::write(self.dir().join(name), text).unwrap();
    }

    fn run(&self, args: &[&str]) -> Output {
        Command::new(binary())
            .current_dir(self.dir())
            .env("CAMDL_OUTPUT_DIR", self.dir().join("results"))
            .env("CAMDL_SKIP_VERSION_CHECK", "1")
            .env("CAMDLC", camdlc_bin())
            .args(args)
            .output()
            .expect("spawn camdl")
    }

    /// Run and require success, returning stderr.
    fn ok(&self, args: &[&str]) -> String {
        let out = self.run(args);
        let stderr = String::from_utf8_lossy(&out.stderr).into_owned();
        assert!(out.status.success(), "`camdl {}` failed:\n{stderr}", args.join(" "));
        format!("{}{}", String::from_utf8_lossy(&out.stdout), stderr)
    }

    /// Every fit leaf directory in the store (the directories holding a
    /// `run.json`), sorted.
    fn leaves(&self) -> Vec<PathBuf> {
        let mut out = Vec::new();
        let mut stack = vec![self.fits()];
        while let Some(d) = stack.pop() {
            let Ok(rd) = std::fs::read_dir(&d) else { continue };
            for e in rd.flatten() {
                let p = e.path();
                if p.is_dir() {
                    if p.join("run.json").is_file() {
                        out.push(p);
                    } else {
                        stack.push(p);
                    }
                }
            }
        }
        out.sort();
        out
    }
}

fn if2(chains: u32) -> String {
    format!(
        "algorithm = \"if2\"\nbackend = \"chain_binomial\"\nchains = {chains}\n\
         particles = 10\niterations = 2\ncooling = 0.7"
    )
}

fn text(out: &Output) -> String {
    format!("{}{}", String::from_utf8_lossy(&out.stdout), String::from_utf8_lossy(&out.stderr))
}

/// `fit summary <leaf directory>` summarises that run. It used to read the
/// leaf as a problem folder, find no sidecar beside it, print "(no completed
/// methods found …)" and exit 0.
#[test]
fn summary_of_a_leaf_directory_summarises_that_run() {
    let p = Project::new();
    p.config("scout.toml", &if2(2));
    p.ok(&["fit", "run", "scout.toml", "--seed", "1"]);
    let leaves = p.leaves();
    assert_eq!(leaves.len(), 1, "premise: one run");

    let out = p.run(&["fit", "summary", &leaves[0].to_string_lossy()]);
    let all = text(&out);
    assert!(out.status.success(), "a leaf directory summarises:\n{all}");
    assert!(!all.contains("no completed methods"), "the leaf is a completed run:\n{all}");
    assert!(all.contains("if2"), "the run's method is reported:\n{all}");
}

/// A directory that holds no completed run has nothing to summarise, and the
/// exit status says so.
#[test]
fn summary_of_a_directory_with_no_run_fails() {
    let p = Project::new();
    let empty = p.dir().join("empty");
    std::fs::create_dir_all(&empty).unwrap();
    let out = p.run(&["fit", "summary", &empty.to_string_lossy()]);
    assert!(!out.status.success(), "nothing to summarise must not exit 0:\n{}", text(&out));
}

fn pgas_from(starts: &str) -> String {
    format!(
        "algorithm = \"pgas\"\nbackend = \"chain_binomial\"\nchains = 2\nparticles = 20\n\
         sweeps = 4\nburn_in = 1\nthin = 1\nstarts = {starts}"
    )
}

/// The problem folder (`results/fits/<stem>-<h8>`) a leaf sits in.
fn segment_of(leaf: &Path) -> PathBuf {
    leaf.parent().and_then(Path::parent).unwrap().to_path_buf()
}

/// One file run twice with a changed `[method]` setting: two runs in one
/// problem folder, differing in `[method]` alone.
fn two_runs_of_one_file(p: &Project) -> (PathBuf, Vec<PathBuf>) {
    p.config("scout.toml", &if2(2));
    p.ok(&["fit", "run", "scout.toml", "--seed", "1"]);
    p.config("scout.toml", &if2(3));
    p.ok(&["fit", "run", "scout.toml", "--seed", "1"]);
    let leaves = p.leaves();
    assert_eq!(leaves.len(), 2, "premise: two runs");
    assert_eq!(segment_of(&leaves[0]), segment_of(&leaves[1]), "premise: one problem folder");
    (segment_of(&leaves[0]), leaves)
}

/// A handle naming a folder that holds two runs is refused, and the refusal
/// says what actually sets them apart. It used to say "several seeds or
/// cells" for two runs with one seed that differ in `[method]`.
#[test]
fn an_ambiguous_starts_source_names_what_sets_the_runs_apart() {
    let p = Project::new();
    let (segment, leaves) = two_runs_of_one_file(&p);
    p.config("post.toml", &pgas_from(&format!("{{ from_mle = \"{}\" }}", segment.display())));

    let out = p.run(&["fit", "run", "post.toml", "--seed", "1"]);
    let all = text(&out);
    assert!(!out.status.success(), "two candidate runs must be refused:\n{all}");
    assert!(!all.contains("several seeds or cells"), "the runs share a seed:\n{all}");
    assert!(all.contains("names 2 runs, which differ in `[method]` settings or `--starts`:"),
        "the refusal names the axis the runs differ on:\n{all}");
    assert!(!all.contains("fit seed"), "the seeds are equal:\n{all}");
    for leaf in &leaves {
        assert!(all.contains(&leaf.display().to_string()), "lists {}:\n{all}", leaf.display());
    }
}

/// `fit summary` of a problem folder holding two runs of one method lists
/// them and fails instead of reporting the one that sorts first.
#[test]
fn summary_of_a_folder_with_two_runs_of_one_method_lists_them() {
    let p = Project::new();
    let (segment, leaves) = two_runs_of_one_file(&p);

    let out = p.run(&["fit", "summary", &segment.to_string_lossy()]);
    let all = text(&out);
    assert!(!out.status.success(), "a choice between two runs is the user's:\n{all}");
    for leaf in &leaves {
        assert!(all.contains(&leaf.display().to_string()), "lists {}:\n{all}", leaf.display());
    }
    // Each run is still summarised when named.
    for leaf in &leaves {
        p.ok(&["fit", "summary", &leaf.to_string_lossy()]);
    }
}

fn pgas(sweeps: u32) -> String {
    format!(
        "algorithm = \"pgas\"\nbackend = \"chain_binomial\"\nchains = 2\nparticles = 20\n\
         sweeps = {sweeps}\nburn_in = 1\nthin = 1\nstarts = \"from_prior\""
    )
}

/// `fit predict <leaf>` replays that run's draws. Two PGAS runs of one file
/// differ in `sweeps`, so their draw counts tell which was read; a leaf
/// directory used to be refused for having no `fit.meta.json` beside it.
#[test]
fn predict_of_a_leaf_reads_that_run() {
    let p = Project::new();
    p.config("post.toml", &pgas(4));
    p.ok(&["fit", "run", "post.toml", "--seed", "1"]);
    let short = p.leaves();
    p.config("post.toml", &pgas(7));
    p.ok(&["fit", "run", "post.toml", "--seed", "1"]);
    let long: Vec<PathBuf> = p.leaves().into_iter().filter(|l| !short.contains(l)).collect();
    assert_eq!((short.len(), long.len()), (1, 1), "premise: two runs");

    // 2 chains × (sweeps − burn_in) retained draws.
    for (leaf, draws) in [(&short[0], 6), (&long[0], 12)] {
        let all = p.ok(&["fit", "predict", &leaf.to_string_lossy(), "--horizon", "free_forward"]);
        assert!(all.contains(&format!("{draws} draws from")),
            "{} holds {draws} draws:\n{all}", leaf.display());
    }
}
