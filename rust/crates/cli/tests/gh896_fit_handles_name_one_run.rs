//! gh#896 — a fit handle names one run.
//!
//! A problem folder (`<build>/fit/<model>-<id12>/`) holds every run of one
//! problem under `runs/`: rerunning a file with a changed `[method]` setting
//! writes a second run beside the first. These tests pin what each handle
//! names in that store: a run's folder, a label, a run-id prefix, a `fit.toml`
//! (never a run) and a problem folder (a problem, not a run).
//!
//! End to end because the defects are process exit codes, what the store
//! records, and which run a verb reads — none of which a unit test over one
//! resolver observes. Deliberately cheap: the SEIR golden IR, one estimated
//! parameter, IF2 at 2 chains × 10 particles × 2 iterations.

mod fit_store_support;

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

    /// The run store every command of the project reads and writes.
    fn store(&self) -> PathBuf {
        self.dir().join("results")
    }

    /// Write `name` holding the shared problem and `method` as its `[method]`.
    fn config(&self, name: &str, method: &str) {
        self.config_with(name, "", method)
    }

    /// [`Project::config`] with `top` written before the first table.
    fn config_with(&self, name: &str, top: &str, method: &str) {
        let text = format!(
            r#"{top}
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
        Command::new(binary()).env_remove("CLICOLOR_FORCE")
            .current_dir(self.dir())
            .env("CAMDL_STORE", self.store())
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

    /// Every stored fit run's folder in the store, sorted.
    fn leaves(&self) -> Vec<PathBuf> {
        let mut out = fit_store_support::runs(&self.store());
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

/// `fit summary <run folder>` summarises that run.
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

/// The problem folder (`<build>/fit/<model>-<id12>/`) a run sits in.
fn segment_of(leaf: &Path) -> PathBuf {
    fit_store_support::problem_of(leaf)
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

/// A problem folder offered as a starts source names a problem, not a run, so
/// it is refused, and the refusal lists the runs the folder holds so the user
/// can name one.
#[test]
fn a_problem_folder_as_a_starts_source_lists_its_runs() {
    let p = Project::new();
    let (segment, leaves) = two_runs_of_one_file(&p);
    p.config("post.toml", &pgas_from(&format!("{{ from_mle = \"{}\" }}", segment.display())));

    let out = p.run(&["fit", "run", "post.toml", "--seed", "1"]);
    let all = text(&out);
    assert!(!out.status.success(), "a problem is not a run:\n{all}");
    assert!(all.contains("names a fit problem") && all.contains("Name one by its id or folder"),
        "the refusal says what the folder names:\n{all}");
    for leaf in &leaves {
        assert!(all.contains(&folder_name(leaf)), "lists {}:\n{all}", leaf.display());
    }
}

/// `fit summary` of a problem folder holding two runs refuses, listing them,
/// instead of reporting one; each run is summarised when named.
#[test]
fn summary_of_a_folder_with_two_runs_of_one_method_lists_them() {
    let p = Project::new();
    let (segment, leaves) = two_runs_of_one_file(&p);

    let out = p.run(&["fit", "summary", &segment.to_string_lossy()]);
    let all = text(&out);
    assert!(!out.status.success(), "a choice between two runs is the user's:\n{all}");
    for leaf in &leaves {
        assert!(all.contains(&folder_name(leaf)), "lists {}:\n{all}", leaf.display());
    }
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

/// `camdl predict <run folder>` replays that run's draws. Two PGAS runs of one
/// file differ in `sweeps`, so their draw counts tell which was read.
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
        let all = p.ok(&["predict", &leaf.to_string_lossy(), "--mode", "free_forward"]);
        assert!(all.contains(&format!("{draws} draws from")),
            "{} holds {draws} draws:\n{all}", leaf.display());
    }
}

/// A run's id, the label of the invocation that stored or reused it (the
/// newest such row `camdl list` shows), and its `run_inputs.json`.
fn record(leaf: &Path) -> (String, Option<String>, serde_json::Value) {
    let inputs = fit_store_support::run_inputs(leaf);
    let id = inputs["run_id"].as_str().unwrap().to_string();
    (id, None, inputs)
}

/// The leaves in the store that `before` did not hold.
fn new_leaves(p: &Project, before: &[PathBuf]) -> Vec<PathBuf> {
    p.leaves().into_iter().filter(|l| !before.contains(l)).collect()
}

/// A fit label names the run most recently given it. A changed file under the
/// same label runs and takes it; the earlier holder is `@L~1`; a cache hit
/// given the label makes that run `@L` again; a cache hit given another label
/// is named by that label too; and the handle forms that name no run are
/// refused by name.
#[test]
fn a_label_moves_to_the_run_last_given_it() {
    let p = Project::new();
    p.config("scout.toml", &if2(2));
    p.ok(&["fit", "run", "scout.toml", "--seed", "1", "--label", "scout"]);
    let two = p.leaves();
    let (two_id, ..) = record(&two[0]);

    // A new identity under the held label runs, and the label moves to it.
    p.config("scout.toml", &if2(3));
    p.ok(&["fit", "run", "scout.toml", "--seed", "1", "--label", "scout"]);
    let three = new_leaves(&p, &two);
    assert_eq!(three.len(), 1);
    let (three_id, ..) = record(&three[0]);

    // `@scout` is the 3-chain run, `@scout~0` the same, `@scout~1` the 2-chain run.
    assert!(p.ok(&["fit", "summary", "@scout"]).contains(&folder_name(&three[0])));
    assert!(p.ok(&["fit", "summary", "@scout~0"]).contains(&folder_name(&three[0])));
    assert!(p.ok(&["fit", "summary", "@scout~1"]).contains(&folder_name(&two[0])));
    assert_ne!(two_id, three_id);

    // Reverting the change: a cache hit given `scout` makes that run `@scout`.
    p.config("scout.toml", &if2(2));
    let all = p.ok(&["fit", "run", "scout.toml", "--seed", "1", "--label", "scout"]);
    assert!(all.contains("cache hit"), "{all}");
    assert!(p.ok(&["fit", "summary", "@scout"]).contains(&folder_name(&two[0])));
    assert!(p.ok(&["fit", "summary", "@scout~1"]).contains(&folder_name(&three[0])));

    // A cache hit given another label: that label names the run.
    p.config("scout.toml", &if2(3));
    let all = p.ok(&["fit", "run", "scout.toml", "--seed", "1", "--label", "wide"]);
    assert!(all.contains("cache hit"), "{all}");
    assert!(p.ok(&["fit", "summary", "@wide"]).contains(&folder_name(&three[0])));

    // A cache hit on an unlabelled run: it takes the label.
    p.config("scout.toml", &if2(4));
    p.ok(&["fit", "run", "scout.toml", "--seed", "1"]);
    let four = new_leaves(&p, &[two.clone(), three.clone()].concat());
    assert_eq!(four.len(), 1);
    p.ok(&["fit", "run", "scout.toml", "--seed", "1", "--label", "four"]);
    assert!(p.ok(&["fit", "summary", "@four"]).contains(&folder_name(&four[0])));
}

/// A malformed label offset is refused naming the form `@<label>~<N>`, and an
/// offset past a label's last earlier holder is refused listing the runs that
/// held it.
#[test]
fn a_label_offset_that_names_no_holder_is_refused_with_the_form_and_the_history() {
    let p = Project::new();
    p.config("scout.toml", &if2(2));
    p.ok(&["fit", "run", "scout.toml", "--seed", "1", "--label", "scout"]);
    let (two_id, ..) = record(&p.leaves()[0]);
    p.config("scout.toml", &if2(3));
    let before = p.leaves();
    p.ok(&["fit", "run", "scout.toml", "--seed", "1", "--label", "scout"]);
    let (three_id, ..) = record(&new_leaves(&p, &before)[0]);
    for bad in ["@scout~", "@scout~x", "@scout~+1"] {
        let out = p.run(&["fit", "summary", bad]);
        assert!(!out.status.success(), "{bad} is refused");
        assert!(text(&out).contains("@<label>~<N>"), "{bad}: {}", text(&out));
    }
    let out = p.run(&["fit", "summary", "@scout~2"]);
    assert!(!out.status.success());
    let past = text(&out);
    assert!(past.contains("@scout~2") && past.contains(&two_id[..12])
        && past.contains(&three_id[..12]), "past the end lists the history:\n{past}");
}

/// A run started from `@scout` records the run `@scout` named, so moving the
/// label does not change what an existing run started from; rerunning it
/// after the move resolves the new source, a new dependency, so it re-keys.
#[test]
fn a_downstream_run_rekeys_after_its_source_label_moves() {
    let p = Project::new();
    p.config("scout.toml", &if2(2));
    p.ok(&["fit", "run", "scout.toml", "--seed", "1", "--label", "scout"]);
    let (scout_a, ..) = record(&p.leaves()[0]);
    p.config("post.toml", &pgas_from("{ from_mle = \"@scout\" }"));
    // The 10-particle scout can lose a chain to the filter's degeneracy
    // watchdog, so its verdict may not pass; this test is about identity.
    p.ok(&["fit", "run", "post.toml", "--seed", "1", "--allow-nonconverged-source"]);
    let before = p.leaves();
    let post_a = before.iter().find(|l| record(l).2["starts_source"]["handle"] == "@scout")
        .unwrap().clone();
    assert_eq!(record(&post_a).2["starts_source"]["run_id"], scout_a.as_str());

    p.config("scout.toml", &if2(3));
    p.ok(&["fit", "run", "scout.toml", "--seed", "1", "--label", "scout"]);
    assert_eq!(record(&post_a).2["starts_source"]["run_id"], scout_a.as_str(),
        "the existing run's source is unchanged");

    let all = p.ok(&["fit", "run", "post.toml", "--seed", "1", "--allow-nonconverged-source"]);
    assert!(!all.contains("cache hit"), "the moved source is a new dependency:\n{all}");
    let post_b: Vec<PathBuf> = new_leaves(&p, &before).into_iter()
        .filter(|l| record(l).2["starts_source"]["handle"] == "@scout").collect();
    assert_eq!(post_b.len(), 1, "the downstream run re-keyed to a new run");
    assert_ne!(record(&post_b[0]).2["starts_source"]["run_id"], scout_a.as_str());
}

/// Every verb that resolves `@L` says which run it resolved to, on stderr.
#[test]
fn every_resolving_verb_prints_the_resolution_line() {
    let p = Project::new();
    p.config("post.toml", &pgas(4));
    p.ok(&["fit", "run", "post.toml", "--seed", "1", "--label", "post"]);
    let (id, ..) = record(&p.leaves()[0]);
    let root = p.store().to_string_lossy().into_owned();
    let want = format!("@post → run {}", &id[..6]);
    for args in [
        vec!["fit", "summary", "@post"],
        vec!["predict", "@post", "--mode", "free_forward", "--n-draws", "2"],
        vec!["show", "@post", "--store", &root],
        vec!["cat", "@post", "draws.tsv", "--store", &root],
    ] {
        let out = p.run(&args);
        assert!(out.status.success(), "{args:?}: {}", text(&out));
        assert!(String::from_utf8_lossy(&out.stderr).contains(&want),
            "{args:?} prints `{want}` on stderr:\n{}", text(&out));
        assert!(!String::from_utf8_lossy(&out.stdout).contains(&want), "never on stdout");
    }
}

/// A label given to an invocation that writes several runs names no one run:
/// `@L` is refused with the runs listed.
#[test]
fn a_label_on_an_invocation_that_writes_several_runs_names_no_one_run() {
    let p = Project::new();
    p.config_with("seeds.toml", "fit_seeds = [1, 2]", &if2(2));
    p.ok(&["fit", "run", "seeds.toml", "--label", "both"]);
    let leaves = p.leaves();
    assert_eq!(leaves.len(), 2, "premise: two seeds, two runs");
    let out = p.run(&["fit", "summary", "@both"]);
    let all = text(&out);
    assert!(!out.status.success(), "{all}");
    assert!(all.contains("names 2 results, where one is needed"), "{all}");
    for leaf in &leaves {
        assert!(all.contains(&record(leaf).0[..12]), "lists {}:\n{all}", leaf.display());
    }
}

/// A `fit.toml` is not a run handle: `fit summary scout.toml` is refused,
/// listing the run the file as it now reads produced, so the user can name it.
#[test]
fn a_config_is_refused_as_a_handle_with_its_runs_listed() {
    let p = Project::new();
    let (_segment, leaves) = two_runs_of_one_file(&p); // 2 chains, then 3
    let three = leaves.iter().find(|l| record(l).2["n_chains"] == 3).unwrap();
    let out = p.run(&["fit", "summary", "scout.toml"]);
    let all = text(&out);
    assert!(!out.status.success(), "{all}");
    assert!(all.contains("a configuration never names a run"), "{all}");
    assert!(all.contains(&record(three).0[..12]) && all.contains(&folder_name(three)),
        "lists the run the file now makes:\n{all}");
}

/// A second config that reaches a stored run by a cache hit lists that run
/// when offered as a handle: `fit_seeds = [1]` and `--seed 1` are one run
/// from two config texts.
#[test]
fn a_config_that_reused_a_stored_run_lists_it() {
    let p = Project::new();
    p.config("scout.toml", &if2(2));
    p.ok(&["fit", "run", "scout.toml", "--seed", "1"]);
    p.config_with("seeded.toml", "fit_seeds = [1]", &if2(2));
    let run = p.ok(&["fit", "run", "seeded.toml"]);
    assert!(run.contains("cache hit"), "premise: one run, reused:\n{run}");
    assert_eq!(p.leaves().len(), 1);
    let out = p.run(&["fit", "summary", "seeded.toml"]);
    let all = text(&out);
    assert!(!out.status.success() && all.contains("a configuration never names a run"), "{all}");
    assert!(all.contains(&record(&p.leaves()[0]).0[..12]), "lists the reused run:\n{all}");
}

/// Two PGAS runs of one file, sweeps 4 then 7: (folder, short run, long run).
fn two_pgas_runs(p: &Project) -> (PathBuf, PathBuf, PathBuf) {
    p.config("post.toml", &pgas(4));
    p.ok(&["fit", "run", "post.toml", "--seed", "1"]);
    let short = p.leaves();
    p.config("post.toml", &pgas(7));
    p.ok(&["fit", "run", "post.toml", "--seed", "1"]);
    let long: Vec<PathBuf> = p.leaves().into_iter().filter(|l| !short.contains(l)).collect();
    assert_eq!((short.len(), long.len()), (1, 1), "premise: two runs");
    (segment_of(&short[0]), short[0].clone(), long[0].clone())
}

/// `camdl predict <problem folder>` holding two runs is refused, listing them.
#[test]
fn predict_of_a_folder_with_two_runs_lists_them() {
    let p = Project::new();
    let (segment, short, long) = two_pgas_runs(&p);
    let out = p.run(&["predict", &segment.to_string_lossy(), "--mode", "free_forward"]);
    let all = text(&out);
    assert!(!out.status.success(), "a choice between two runs is the user's:\n{all}");
    assert!(all.contains("names a fit problem"), "{all}");
    for leaf in [&short, &long] {
        assert!(all.contains(&folder_name(leaf)), "lists {}:\n{all}", leaf.display());
    }
}

/// `camdl predict` writes inside the run it read (`<run>/predict/<mode>-<id12>/`),
/// and its manifests record that run's id; nothing is written into the
/// problem folder the two runs share.
#[test]
fn predict_writes_under_the_run_it_read() {
    let p = Project::new();
    let (segment, short, long) = two_pgas_runs(&p);
    for leaf in [&short, &long] {
        p.ok(&["predict", &leaf.to_string_lossy(), "--mode", "free_forward"]);
    }
    assert!(!segment.join("predictive").exists() && !segment.join("predictive.json").exists()
        && !segment.join("predict").exists(),
        "nothing is written into the shared folder");
    for (leaf, draws) in [(&short, 6), (&long, 12)] {
        let (id, _, _) = record(leaf);
        let outs = fit_store_support::predictions(leaf);
        assert_eq!(outs.len(), 1, "one prediction inside {}: {outs:?}", leaf.display());
        let out = &outs[0];
        let manifest: serde_json::Value = serde_json::from_slice(
            &std::fs::read(out.join("predictive.json")).unwrap()).unwrap();
        assert_eq!(manifest["run_id"], id.as_str(), "predictive.json names its run");
        let observed: serde_json::Value = serde_json::from_slice(
            &std::fs::read(out.join("observed.json")).unwrap()).unwrap();
        assert_eq!(observed["run_id"], id.as_str());
        let tsv = std::fs::read_to_string(out.join("predictive").join("weekly_cases.tsv")).unwrap();
        assert!(tsv.lines().skip(1).any(|l| l.split('\t').any(|c| c == draws.to_string())),
            "{} holds this run's {draws}-draw bands", out.display());
        assert!(out.join("report.json").is_file());
    }
}

/// `compare` scores the run each handle names. Two IF2 runs of one problem
/// end at different θ̂, so at one seed and particle count their scores
/// differ; scored from the folder's choice instead, both rows would be one
/// run and agree exactly.
#[test]
fn compare_scores_the_named_runs_estimate() {
    let p = Project::new();
    let if2_iters = |n: u32| format!(
        "algorithm = \"if2\"\nbackend = \"chain_binomial\"\nchains = 2\n\
         particles = 30\niterations = {n}\ncooling = 0.7");
    p.config("scout.toml", &if2_iters(2));
    p.ok(&["fit", "run", "scout.toml", "--seed", "1"]);
    p.config("scout.toml", &if2_iters(6));
    p.ok(&["fit", "run", "scout.toml", "--seed", "1"]);
    let leaves = p.leaves();
    assert_eq!(leaves.len(), 2, "premise: two runs in one folder");
    let theta = |leaf: &Path| std::fs::read_to_string(leaf.join("final_params.toml")).unwrap();
    assert_ne!(theta(&leaves[0]), theta(&leaves[1]), "premise: two different estimates");

    let (a, b) = (leaves[0].to_string_lossy(), leaves[1].to_string_lossy());
    let out = p.run(&["compare", &a, &b, "--particles", "100", "--seed", "1",
        "--replicates", "1", "--draws", "1", "--format", "json"]);
    assert!(out.status.success(), "compare failed:\n{}", text(&out));
    let v: serde_json::Value = serde_json::from_slice(&out.stdout).expect("compare JSON");
    let elpd = |h: &str| -> f64 {
        v["rows"].as_array().unwrap().iter()
            .find(|r| r["path"] == h).unwrap_or_else(|| panic!("no row {h}: {v}"))["elpd"]
            .as_f64().unwrap()
    };
    assert_ne!(elpd(&a), elpd(&b), "each row scores its own run's θ̂: {v}");
}

/// A sourced run's identity is the source run, not how its handle was typed.
/// `@scout`, the scout's leaf directory and a prefix of its run id name one
/// run, so a run started from each is one run: the second and third are cache
/// hits. A different source is a different run.
#[test]
fn the_starts_handle_spelling_is_not_identity() {
    let p = Project::new();
    p.config("scout.toml", &if2(2));
    p.ok(&["fit", "run", "scout.toml", "--seed", "1", "--label", "scout"]);
    let scout = p.leaves();
    assert_eq!(scout.len(), 1);
    let (scout_id, _, _) = record(&scout[0]);

    p.config("post.toml", &pgas_from("\"from_prior\""));
    // The scouts are two-iteration IF2 runs, whose convergence verdict can be
    // refused (an Â that is not finite reads incomplete); this test is about
    // identity, not convergence, so every sourced run lifts that refusal.
    // `--allow-nonconverged-source` is a gate, not part of the run identity.
    let mut post_ids: Vec<String> = Vec::new();
    for (i, handle) in ["@scout".to_string(), scout[0].display().to_string(),
                        scout_id[..12].to_string()].iter().enumerate() {
        let out = p.ok(&["fit", "run", "post.toml", "--seed", "1",
            "--allow-nonconverged-source",
            "--starts", &format!("from_mle={handle}")]);
        if i > 0 {
            assert!(out.contains("cache hit"), "{handle} names the same source run:\n{out}");
        }
        let post: Vec<PathBuf> = p.leaves().into_iter().filter(|l| !scout.contains(l)).collect();
        assert_eq!(post.len(), 1, "one sourced run, whatever the spelling ({handle})");
        post_ids.push(record(&post[0]).0);
    }
    assert!(post_ids.windows(2).all(|w| w[0] == w[1]), "{post_ids:?}");

    // Control: a different source run is a different run.
    p.config("scout.toml", &if2(3));
    p.ok(&["fit", "run", "scout.toml", "--seed", "1", "--label", "scout3"]);
    p.ok(&["fit", "run", "post.toml", "--seed", "1", "--allow-nonconverged-source",
        "--starts", "from_mle=@scout3"]);
    let ids: std::collections::BTreeSet<String> = p.leaves().iter()
        .map(|l| record(l).0).collect();
    assert_eq!(ids.len(), 4, "two scouts and two sourced runs: {ids:?}");
}

/// `fit table` gives each run its own row, with that run's label, time and
/// numbers.
#[test]
fn fit_table_gives_each_run_its_own_row() {
    let p = Project::new();
    p.config("post.toml", &pgas(4));
    p.ok(&["fit", "run", "post.toml", "--seed", "1", "--label", "short"]);
    let short = p.leaves();
    p.config("post.toml", &pgas(7));
    p.ok(&["fit", "run", "post.toml", "--seed", "1", "--label", "long"]);
    let leaves = p.leaves();
    assert_eq!(leaves.len(), 2);

    let out = p.run(&["fit", "table", &p.store().to_string_lossy(), "--format", "json"]);
    assert!(out.status.success(), "{}", text(&out));
    let v: serde_json::Value = serde_json::from_slice(&out.stdout).unwrap();
    let rows = v["rows"].as_array().unwrap();
    assert_eq!(rows.len(), 2, "one row per run: {v}");
    for leaf in &leaves {
        let (id, ..) = record(leaf);
        let label = if short.contains(leaf) { "short" } else { "long" };
        let row = rows.iter().find(|r| r["run_id"] == id.as_str())
            .unwrap_or_else(|| panic!("no row for run {id}: {v}"));
        assert_eq!(row["label"].as_str(), Some(label), "the row's label is its run's: {row}");
        let run_json = fit_store_support::json(&leaf.join("run.json"));
        assert_eq!(row["created_at"], run_json["started"], "and its time: {row}");
    }

    let csv = p.ok(&["fit", "table", &p.store().to_string_lossy(), "--format", "csv"]);
    assert!(csv.lines().next().unwrap().split(',').any(|c| c == "run_id"), "{csv}");
    let txt = p.ok(&["fit", "table", &p.store().to_string_lossy()]);
    for leaf in &leaves {
        assert!(txt.contains(&record(leaf).0[..12]), "the text table names each run:\n{txt}");
    }
}

/// `simulate --draws posterior --fit` reads the run its handle names: a
/// problem folder holding two runs is refused with them listed, and `@label`
/// is accepted.
#[test]
fn simulate_posterior_draws_come_from_the_named_run() {
    let p = Project::new();
    p.config("post.toml", &pgas(4));
    p.ok(&["fit", "run", "post.toml", "--seed", "1", "--label", "short"]);
    let short = p.leaves().remove(0);
    p.config("post.toml", &pgas(7));
    p.ok(&["fit", "run", "post.toml", "--seed", "1"]);
    let leaves = p.leaves();
    assert_eq!(leaves.len(), 2);
    let segment = segment_of(&short);
    let ir = golden_ir().to_string_lossy().into_owned();
    let sim = |fit: &str| p.run(&["simulate", &ir, "--draws", "posterior", "--fit", fit,
        "--seed", "1"]);

    let out = sim(&segment.to_string_lossy());
    let all = text(&out);
    assert!(!out.status.success(), "a folder holding two runs is refused:\n{all}");
    for leaf in &leaves {
        assert!(all.contains(&folder_name(leaf)), "lists {}:\n{all}", leaf.display());
    }

    let out = sim("@short");
    let all = text(&out);
    assert!(out.status.success(), "@label names one run:\n{all}");
    assert!(all.contains("6 draws from") && all.contains(&folder_name(&short)),
        "the draws are the labelled run's:\n{all}");
}

/// `camdl list --kind fit --format json` is one row per run, each with its
/// own label.
#[test]
fn camdl_list_json_labels_each_run() {
    let p = Project::new();
    p.config("post.toml", &pgas(4));
    p.ok(&["fit", "run", "post.toml", "--seed", "1", "--label", "short"]);
    let short = p.leaves();
    p.config("post.toml", &pgas(7));
    p.ok(&["fit", "run", "post.toml", "--seed", "1", "--label", "long"]);
    let out = p.run(&["list", "--kind", "fit", "--format", "json", "--store",
        &p.store().to_string_lossy()]);
    assert!(out.status.success(), "{}", text(&out));
    let rows: Vec<serde_json::Value> = String::from_utf8_lossy(&out.stdout).lines()
        .filter(|l| l.contains("\"kind\":\"fit_run\""))
        .map(|l| serde_json::from_str(l).unwrap()).collect();
    assert_eq!(rows.len(), 2, "{rows:?}");
    for leaf in p.leaves() {
        let (id, ..) = record(&leaf);
        let label = if short.contains(&leaf) { "short" } else { "long" };
        let row = rows.iter().find(|r| r["run_id"] == id.as_str()).unwrap_or_else(|| panic!("{rows:?}"));
        assert_eq!(row["label"].as_str(), Some(label), "{row}");
    }
}

/// A predict output's address is keyed by the options that shape it, as well
/// as by its run: the default predict and `--n-draws 3` each have their own
/// folder, whose manifest records the options.
#[test]
fn predict_options_key_their_own_output() {
    let p = Project::new();
    p.config("post.toml", &pgas(7));
    p.ok(&["fit", "run", "post.toml", "--seed", "1"]);
    let leaf = p.leaves().remove(0);
    let l = leaf.to_string_lossy().into_owned();
    p.ok(&["predict", &l, "--mode", "free_forward"]);
    p.ok(&["predict", &l, "--mode", "free_forward", "--n-draws", "3"]);
    let dirs = fit_store_support::predictions(&leaf);
    assert_eq!(dirs.len(), 2, "two option sets, two addresses: {dirs:?}");
    let mut n_draws: Vec<u64> = dirs.iter().map(|d| {
        let m: serde_json::Value = serde_json::from_slice(
            &std::fs::read(d.join("predictive.json")).unwrap()).unwrap();
        m["options"]["n_draws"].as_u64().unwrap_or_else(|| panic!("options recorded: {m}"))
    }).collect();
    n_draws.sort();
    assert_eq!(n_draws[0], 3, "{n_draws:?}");
}

/// `fit table --label-pattern` selects runs, the unit a row is: of two runs
/// in one problem folder, only the matching one has a row.
#[test]
fn fit_table_filters_select_runs() {
    let p = Project::new();
    p.config("post.toml", &pgas(4));
    p.ok(&["fit", "run", "post.toml", "--seed", "1", "--label", "short"]);
    p.config("post.toml", &pgas(7));
    p.ok(&["fit", "run", "post.toml", "--seed", "1", "--label", "long"]);
    let rows = |extra: &[&str]| -> Vec<String> {
        let mut args = vec!["fit", "table"];
        let fits = p.store().to_string_lossy().into_owned();
        args.push(&fits);
        args.extend_from_slice(&["--format", "json"]);
        args.extend_from_slice(extra);
        let out = p.run(&args);
        assert!(out.status.success(), "{}", text(&out));
        let v: serde_json::Value = serde_json::from_slice(&out.stdout).unwrap();
        v["rows"].as_array().unwrap().iter()
            .map(|r| r["label"].as_str().unwrap_or("").to_string()).collect()
    };
    assert_eq!(rows(&["--label-pattern", "short"]), vec!["short"]);
}

/// A prediction is a result of its own inside its run
/// (`<run>/predict/<mode>-<id12>/`): `cat <run> predictive/…` names
/// nothing in the run, and the prediction's own folder shows and reads its
/// files.
#[test]
fn a_prediction_is_read_through_its_own_folder() {
    let p = Project::new();
    p.config("post.toml", &pgas(7));
    p.ok(&["fit", "run", "post.toml", "--seed", "1", "--label", "post"]);
    p.ok(&["predict", "@post", "--mode", "free_forward", "--n-draws", "3"]);
    let root = p.store().to_string_lossy().into_owned();

    let out = p.run(&["cat", "@post", "predictive/weekly_cases.tsv", "--store", &root]);
    assert!(!out.status.success(),
        "the run holds no predictive file of its own:\n{}", text(&out));
    let leaf = p.leaves().remove(0);
    let preds = fit_store_support::predictions(&leaf);
    assert_eq!(preds.len(), 1, "{preds:?}");
    let pred = preds[0].to_string_lossy().into_owned();
    let show = p.ok(&["show", &pred]);
    assert!(show.contains("prediction") && show.contains("predictive/weekly_cases.tsv"),
        "the prediction lists its files:\n{show}");
    let out = p.run(&["cat", &pred, "predictive/weekly_cases.tsv"]);
    assert!(out.status.success(), "{}", text(&out));
    assert_eq!(out.stdout, std::fs::read(preds[0].join("predictive/weekly_cases.tsv")).unwrap());
}

/// Two labelled PGAS runs of one problem folder, `@short` (4 sweeps) and
/// `@long` (7): the same folder, so only the run each label names tells them
/// apart. Returns `(short leaf, long leaf)`.
fn two_labelled_posteriors(p: &Project) -> (PathBuf, PathBuf) {
    p.config("post.toml", &pgas(4));
    p.ok(&["fit", "run", "post.toml", "--seed", "1", "--label", "short"]);
    let short = p.leaves();
    p.config("post.toml", &pgas(7));
    p.ok(&["fit", "run", "post.toml", "--seed", "1", "--label", "long"]);
    let long: Vec<PathBuf> = p.leaves().into_iter().filter(|l| !short.contains(l)).collect();
    assert_eq!((short.len(), long.len()), (1, 1), "premise: two runs");
    assert_eq!(segment_of(&short[0]), segment_of(&long[0]), "premise: one problem folder");
    (short[0].clone(), long[0].clone())
}

fn stdout(out: &Output) -> String {
    String::from_utf8_lossy(&out.stdout).into_owned()
}

/// `show @L` describes the run `@L` names — its id, its draws — not the
/// folder it shares with another run.
#[test]
fn show_of_a_label_describes_that_run() {
    let p = Project::new();
    let (short, long) = two_labelled_posteriors(&p);
    let (short_id, ..) = record(&short);
    let (long_id, ..) = record(&long);
    let root = p.store().to_string_lossy().into_owned();

    let out = p.run(&["show", "@short", "--store", &root]);
    assert!(out.status.success(), "{}", text(&out));
    let shown = stdout(&out);
    assert!(shown.contains(&short_id[..12]), "{shown}");
    assert!(!shown.contains(&long_id[..12]), "the other run is not described:\n{shown}");
    assert!(shown.lines().any(|l| l.trim_start().starts_with("draws.tsv")),
        "the run's draws.tsv is among its outputs:\n{shown}");
    let field = |name: &str| shown.lines().skip_while(|l| l.trim() != name).nth(1).map(str::trim);
    assert_eq!((field("method"), field("seed")), (Some("pgas"), Some("1")), "{shown}");
}

/// `show --format json` on a run is the run's record: its kind, id, folder,
/// files and identity; on a problem folder it is the problem's.
#[test]
fn show_format_json_on_a_run_and_on_a_folder() {
    let p = Project::new();
    let (short, _long) = two_labelled_posteriors(&p);
    let (short_id, ..) = record(&short);
    let root = p.store().to_string_lossy().into_owned();

    let out = p.run(&["show", "@short", "--store", &root, "--format", "json"]);
    assert!(out.status.success(), "{}", text(&out));
    let v: serde_json::Value = serde_json::from_slice(&out.stdout)
        .unwrap_or_else(|e| panic!("show --format json is JSON ({e}):\n{}", text(&out)));
    assert_eq!(v["kind"], "fit_run", "{v}");
    assert_eq!(v["run_id"], short_id.as_str());
    assert!(v["path"].as_str().unwrap().ends_with(
        short.file_name().unwrap().to_str().unwrap()), "{v}");
    assert!(v["files"].as_array().unwrap().iter().any(|f| f == "draws.tsv"), "draws.tsv listed: {v}");
    assert!(v["identity"].is_object(), "{v}");

    let seg = segment_of(&short).to_string_lossy().into_owned();
    let out = p.run(&["show", &seg, "--format", "json"]);
    assert!(out.status.success(), "{}", text(&out));
    let v: serde_json::Value = serde_json::from_slice(&out.stdout).unwrap();
    assert!(v["files"].as_array().unwrap().iter().any(|f| f == "problem.toml"), "{v}");

    let out = p.run(&["show", "@short", "--store", &root, "--format", "yaml"]);
    assert!(!out.status.success(), "an unknown format is refused:\n{}", text(&out));
}

/// `cat @L draws.tsv` reads the run's own draws; a problem folder
/// holds the problem's files, not a run's, so its `draws.tsv` is refused with
/// the files it does hold.
#[test]
fn cat_of_a_label_reads_the_runs_draws_and_a_folder_reads_its_problem() {
    let p = Project::new();
    let (short, long) = two_labelled_posteriors(&p);
    let root = p.store().to_string_lossy().into_owned();

    let out = p.run(&["cat", "@long", "draws.tsv", "--store", &root]);
    assert!(out.status.success(), "{}", text(&out));
    assert_eq!(out.stdout, std::fs::read(long.join("draws.tsv")).unwrap(),
        "the bytes of @long's draws.tsv");

    let seg = segment_of(&short).to_string_lossy().into_owned();
    let out = p.run(&["cat", &seg, "draws.tsv"]);
    let all = text(&out);
    assert!(!out.status.success(), "a problem folder holds no draws:\n{all}");
    assert!(all.contains("has no `draws.tsv`") && all.contains("problem"), "the refusal lists the problem's files:\n{all}");

    // A path the folder itself holds still reads.
    p.ok(&["cat", &seg, "problem.toml"]);
}

/// The data rows of `camdl list --kind fit` (stdout lines under the table
/// header, not the header or the section title).
fn list_rows(stdout: &str) -> Vec<String> {
    stdout.lines()
        .filter(|l| !l.trim().is_empty() && l.trim() != "fits" && !l.contains("RUN_ID"))
        .map(str::to_string)
        .collect()
}

/// Two runs list two rows, each naming its run, its method and its label; a
/// run started from another records that run's id with the handle that named
/// it.
#[test]
fn list_shows_one_row_per_run_with_its_source() {
    let p = Project::new();
    p.config("scout.toml", &if2(2));
    p.ok(&["fit", "run", "scout.toml", "--seed", "1", "--label", "scout"]);
    let (scout_id, ..) = record(&p.leaves()[0]);
    p.config("post.toml", &pgas_from("{ from_mle = \"@scout\" }"));
    p.ok(&["fit", "run", "post.toml", "--seed", "1", "--allow-nonconverged-source"]);
    let root = p.store().to_string_lossy().into_owned();
    let out = p.run(&["list", "--kind", "fit", "--store", &root]);
    assert!(out.status.success(), "{}", text(&out));
    let stdout = stdout(&out);
    let rows = list_rows(&stdout);
    let scout = rows.iter().find(|r| r.contains("if2")).unwrap_or_else(|| panic!("{stdout}"));
    assert!(scout.contains("scout") && scout.contains(&scout_id[..8]), "{stdout}");
    let post_leaf = p.leaves().into_iter().find(|l| l.join("draws.tsv").is_file()).unwrap();
    let (post_id, _, post) = record(&post_leaf);
    assert!(rows.iter().any(|r| r.contains("pgas") && r.contains(&post_id[..8])), "{stdout}");
    assert_eq!(post["starts_source"]["run_id"], scout_id.as_str(), "{post}");
    assert_eq!(post["starts_source"]["handle"], "@scout", "{post}");
}

/// Two seeds of one method are two runs and two rows; `--limit` counts runs.
#[test]
fn list_shows_each_seed_as_a_run() {
    let p = Project::new();
    p.config_with("seeds.toml", "fit_seeds = [1, 2]", &if2(2));
    p.ok(&["fit", "run", "seeds.toml"]);
    let root = p.store().to_string_lossy().into_owned();
    let listed = stdout(&p.run(&["list", "--kind", "fit", "--store", &root]));
    for leaf in p.leaves() {
        assert!(listed.contains(&record(&leaf).0[..8]), "each run has a row:\n{listed}");
    }
    let json = stdout(&p.run(&["list", "--kind", "fit", "--format", "json", "--limit", "1", "--store", &root]));
    assert_eq!(json.lines().filter(|l| l.contains("\"kind\":\"fit_run\"")).count(), 1,
        "--limit counts runs:\n{json}");
}

/// A label given to a second run moves to it, and the first is its earlier
/// holder, `@L~1`. The fits table lists every run, each with the label of an
/// invocation that holds it, so both rows show `scout`; `--history` lists the
/// same rows.
#[test]
fn list_labels_each_run_with_the_label_that_held_it() {
    let p = Project::new();
    p.config("scout.toml", &if2(2));
    p.ok(&["fit", "run", "scout.toml", "--seed", "1", "--label", "scout"]);
    let first = p.leaves();
    p.config("scout.toml", &if2(3));
    p.ok(&["fit", "run", "scout.toml", "--seed", "1", "--label", "scout"]);
    assert_eq!(p.leaves().len(), 2, "premise: two runs");
    let root = p.store().to_string_lossy().into_owned();

    let plain = stdout(&p.run(&["list", "--kind", "fit", "--store", &root]));
    let rows = list_rows(&plain);
    assert_eq!(rows.len(), 2, "one row per run:\n{plain}");
    for leaf in p.leaves() {
        let id = record(&leaf).0;
        assert!(rows.iter().any(|r| r.contains(&id[..8]) && r.contains("scout")),
            "run {id} is listed labelled scout:\n{plain}");
    }
    assert!(p.ok(&["fit", "summary", "@scout~1"]).contains(&folder_name(&first[0])),
        "the earlier holder is @scout~1");
    let hist = stdout(&p.run(&["list", "--kind", "fit", "--history", "--store", &root]));
    assert_eq!(list_rows(&hist), rows, "--history changes nothing in the fits table");
}

/// A run folder's name (`seed_<n>-<id12>`), unique among runs, which every
/// message naming the run carries whether it spells the path relative or
/// absolute.
fn folder_name(run: &std::path::Path) -> String {
    run.file_name().unwrap().to_string_lossy().into_owned()
}
