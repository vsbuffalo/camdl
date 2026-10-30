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

/// The `run_id` a leaf's `run.json` records, and its label.
fn record(leaf: &Path) -> (String, Option<String>, serde_json::Value) {
    let v: serde_json::Value =
        serde_json::from_slice(&std::fs::read(leaf.join("run.json")).unwrap()).unwrap();
    let id = v["run_id"].as_str().unwrap().to_string();
    let label = v["provenance"]["label"].as_str().map(String::from);
    (id, label, v)
}

/// A label names the one run it was given to and never moves. Rerunning a
/// changed file under the same label is refused before any sampling; a
/// second label on a rerun names the new run without touching the first;
/// and a run started from `@scout` records the scout's run id.
#[test]
fn a_label_names_one_run_and_never_moves() {
    let p = Project::new();
    p.config("scout.toml", &if2(2));
    p.ok(&["fit", "run", "scout.toml", "--seed", "1", "--label", "scout"]);
    let first = p.leaves();
    assert_eq!(first.len(), 1);
    let (scout_id, scout_label, _) = record(&first[0]);
    assert_eq!(scout_label.as_deref(), Some("scout"), "the label is on the run");

    // Same file, changed `[method]`, same label: refused, nothing written.
    p.config("scout.toml", &if2(3));
    let out = p.run(&["fit", "run", "scout.toml", "--seed", "1", "--label", "scout"]);
    let all = text(&out);
    assert!(!out.status.success(), "a held label is refused:\n{all}");
    assert!(all.contains(&format!("already names run {}", &scout_id[..12])), "{all}");
    assert_eq!(p.leaves(), first, "the refused run wrote no leaf");

    // Under a new label it runs, and the old label still names the old run.
    p.ok(&["fit", "run", "scout.toml", "--seed", "1", "--label", "scout3"]);
    let second: Vec<PathBuf> = p.leaves().into_iter().filter(|l| !first.contains(l)).collect();
    assert_eq!(second.len(), 1);
    assert_eq!(record(&second[0]).1.as_deref(), Some("scout3"));
    assert_eq!(record(&first[0]).1.as_deref(), Some("scout"), "the first label did not move");

    // A cache hit under its own label is fine; under another it is refused.
    p.config("scout.toml", &if2(2));
    p.ok(&["fit", "run", "scout.toml", "--seed", "1", "--label", "scout"]);
    let out = p.run(&["fit", "run", "scout.toml", "--seed", "1", "--label", "renamed"]);
    assert!(!out.status.success(), "a run keeps its label:\n{}", text(&out));
    assert!(text(&out).contains("already stored as @scout"), "{}", text(&out));

    // `@scout` names the first run; the dependent run records its run id.
    p.config("post.toml", &pgas_from("{ from_mle = \"@scout\" }"));
    p.ok(&["fit", "run", "post.toml", "--seed", "1"]);
    let post: Vec<PathBuf> = p
        .leaves()
        .into_iter()
        .filter(|l| !first.contains(l) && !second.contains(l))
        .collect();
    assert_eq!(post.len(), 1);
    let (_, _, rec) = record(&post[0]);
    let deps: Vec<&str> =
        rec["deps"].as_array().unwrap().iter().map(|d| d["run_id"].as_str().unwrap()).collect();
    assert_eq!(deps, vec![scout_id.as_str()], "the source's run id is in deps");
    assert_eq!(rec["inputs"]["starts_source"]["handle"], "@scout");
    assert_eq!(rec["inputs"]["starts_source"]["run_id"], scout_id.as_str(),
        "the record links the handle to the run it read");
    let summary = p.ok(&["fit", "summary", &post[0].to_string_lossy()]);
    assert!(summary.contains(&format!("from_mle @scout, run {}", &scout_id[..12])),
        "the summary names the source run beside its handle:\n{summary}");
}

/// One invocation writing several runs has no one run to give a label to.
#[test]
fn a_label_on_an_invocation_that_writes_several_runs_is_refused() {
    let p = Project::new();
    p.config_with("seeds.toml", "fit_seeds = [1, 2]", &if2(2));
    let out = p.run(&["fit", "run", "seeds.toml", "--label", "both"]);
    let all = text(&out);
    assert!(!out.status.success(), "{all}");
    assert!(all.contains("names one run, but this invocation writes 2 method leaves"), "{all}");
    assert!(p.leaves().is_empty(), "nothing ran");
}

/// `camdl label` gives a fit run a label once: a held label or a relabel is
/// refused, and a fit id (a problem folder, not a run) is refused with its
/// runs listed.
#[test]
fn camdl_label_labels_a_fit_run_once() {
    let p = Project::new();
    p.config("scout.toml", &if2(2));
    p.ok(&["fit", "run", "scout.toml", "--seed", "1", "--label", "scout"]);
    p.config("scout.toml", &if2(3));
    p.ok(&["fit", "run", "scout.toml", "--seed", "1"]);
    let leaves = p.leaves();
    let ids: Vec<(String, Option<String>)> =
        leaves.iter().map(|l| { let (i, lab, _) = record(l); (i, lab) }).collect();
    let (labelled, unlabelled) = if ids[0].1.is_some() { (&ids[0], &ids[1]) } else { (&ids[1], &ids[0]) };

    let held = p.run(&["label", &unlabelled.0[..12], "scout"]);
    assert!(!held.status.success(), "a held label is refused:\n{}", text(&held));
    let relabel = p.run(&["label", &labelled.0[..12], "other"]);
    assert!(!relabel.status.success(), "a label never moves:\n{}", text(&relabel));

    p.ok(&["label", &unlabelled.0[..12], "wide"]);
    let out = p.ok(&["fit", "summary", "@wide"]);
    assert!(out.contains("3 chains"), "@wide is the 3-chain run:\n{out}");

    let fit_id = record(&leaves[0]).2["levels"][0]["hash"].as_str().unwrap()[..8].to_string();
    let folder = p.run(&["label", &fit_id, "folder"]);
    assert!(!folder.status.success(), "{}", text(&folder));
    assert!(text(&folder).contains("a problem folder, not a run"), "{}", text(&folder));
}

/// A `fit.toml` names the run made from the file as it now reads. One file
/// run with 2 chains then 3 leaves both runs in one folder whose archive is
/// the 3-chain config; `fit summary scout.toml` used to report whichever run
/// sorted first.
#[test]
fn a_config_names_the_run_made_from_it() {
    let p = Project::new();
    let (_segment, _leaves) = two_runs_of_one_file(&p); // 2 chains, then 3
    let now = p.ok(&["fit", "summary", "scout.toml"]);
    assert!(now.contains("3 chains"), "the file says 3 chains:\n{now}");
    p.config("scout.toml", &if2(2));
    let back = p.ok(&["fit", "summary", "scout.toml"]);
    assert!(back.contains("2 chains"), "the file says 2 chains again:\n{back}");
}

/// A second config that reaches a stored run by a cache hit names it too:
/// `fit_seeds = [1]` and `--seed 1` are one run from two config texts.
#[test]
fn a_config_that_reused_a_stored_run_names_it() {
    let p = Project::new();
    p.config("scout.toml", &if2(2));
    p.ok(&["fit", "run", "scout.toml", "--seed", "1"]);
    p.config_with("seeded.toml", "fit_seeds = [1]", &if2(2));
    let run = p.ok(&["fit", "run", "seeded.toml"]);
    assert!(run.contains("cache hit"), "premise: one run, reused:\n{run}");
    assert_eq!(p.leaves().len(), 1);
    p.ok(&["fit", "summary", "seeded.toml"]);
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

/// `fit predict <folder>` holding two runs lists them and fails. It used to
/// replay the last leaf with draws in directory order.
#[test]
fn predict_of_a_folder_with_two_runs_lists_them() {
    let p = Project::new();
    let (segment, short, long) = two_pgas_runs(&p);
    let out = p.run(&["fit", "predict", &segment.to_string_lossy(), "--horizon", "free_forward"]);
    let all = text(&out);
    assert!(!out.status.success(), "a choice between two runs is the user's:\n{all}");
    assert!(all.contains("names 2 runs"), "{all}");
    for leaf in [&short, &long] {
        assert!(all.contains(&leaf.display().to_string()), "lists {}:\n{all}", leaf.display());
    }
}

/// `fit predict` writes under a directory keyed by the run it read, and its
/// manifests record that run's id. Two runs of one problem used to write into
/// the shared folder, the second overwriting the first, and `predictive.json`
/// named no run.
#[test]
fn predict_writes_under_the_run_it_read() {
    let p = Project::new();
    let (segment, short, long) = two_pgas_runs(&p);
    for leaf in [&short, &long] {
        p.ok(&["fit", "predict", &leaf.to_string_lossy(), "--horizon", "free_forward"]);
    }
    assert!(!segment.join("predictive").exists() && !segment.join("predictive.json").exists(),
        "nothing is written into the shared folder");
    for (leaf, draws) in [(&short, 6), (&long, 12)] {
        let (id, _, _) = record(leaf);
        let out = segment.join("predict").join(&id[..12]);
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
    let mut post_ids: Vec<String> = Vec::new();
    for (i, handle) in ["@scout".to_string(), scout[0].display().to_string(),
                        scout_id[..12].to_string()].iter().enumerate() {
        let out = p.ok(&["fit", "run", "post.toml", "--seed", "1",
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
    p.ok(&["fit", "run", "post.toml", "--seed", "1", "--starts", "from_mle=@scout3"]);
    let ids: std::collections::BTreeSet<String> = p.leaves().iter()
        .map(|l| record(l).0).collect();
    assert_eq!(ids.len(), 4, "two scouts and two sourced runs: {ids:?}");
}

/// `fit table` gives each run its own row, with that run's label, time and
/// numbers. With two runs in one folder it used to print one row whose label
/// and time were the newest run's and whose run id and estimates were the
/// other's.
#[test]
fn fit_table_gives_each_run_its_own_row() {
    let p = Project::new();
    p.config("post.toml", &pgas(4));
    p.ok(&["fit", "run", "post.toml", "--seed", "1", "--label", "short"]);
    p.config("post.toml", &pgas(7));
    p.ok(&["fit", "run", "post.toml", "--seed", "1", "--label", "long"]);
    let leaves = p.leaves();
    assert_eq!(leaves.len(), 2);

    let out = p.run(&["fit", "table", &p.fits().to_string_lossy(), "--format", "json"]);
    assert!(out.status.success(), "{}", text(&out));
    let v: serde_json::Value = serde_json::from_slice(&out.stdout).unwrap();
    let rows = v["rows"].as_array().unwrap();
    assert_eq!(rows.len(), 2, "one row per run: {v}");
    for leaf in &leaves {
        let (id, label, rec) = record(leaf);
        let row = rows.iter().find(|r| r["run_id"] == id.as_str())
            .unwrap_or_else(|| panic!("no row for run {id}: {v}"));
        assert_eq!(row["label"].as_str(), label.as_deref(), "the row's label is its run's");
        assert_eq!(row["created_at"], rec["provenance"]["created_at"], "and its time");
    }

    let csv = p.ok(&["fit", "table", &p.fits().to_string_lossy(), "--format", "csv"]);
    assert!(csv.lines().next().unwrap().split(',').any(|c| c == "run_id"), "{csv}");
    let txt = p.ok(&["fit", "table", &p.fits().to_string_lossy()]);
    for leaf in &leaves {
        assert!(txt.contains(&record(leaf).0[..12]), "the text table names each run:\n{txt}");
    }
}

/// `simulate --draws posterior --fit` reads the run its handle names: a
/// folder holding two runs is refused with them listed (it used to replay the
/// last in directory order), and `@label` is accepted.
#[test]
fn simulate_posterior_draws_come_from_the_named_run() {
    let p = Project::new();
    p.config("post.toml", &pgas(4));
    p.ok(&["fit", "run", "post.toml", "--seed", "1", "--label", "short"]);
    p.config("post.toml", &pgas(7));
    p.ok(&["fit", "run", "post.toml", "--seed", "1"]);
    let leaves = p.leaves();
    let short = leaves.iter().find(|l| record(l).1.as_deref() == Some("short")).unwrap().clone();
    let segment = segment_of(&short);
    let ir = golden_ir().to_string_lossy().into_owned();
    let sim = |fit: &str| p.run(&["simulate", &ir, "--draws", "posterior", "--fit", fit,
        "--seed", "1", "--output-dir", "sims"]);

    let out = sim(&segment.to_string_lossy());
    let all = text(&out);
    assert!(!out.status.success(), "a folder holding two runs is refused:\n{all}");
    assert!(all.contains("names 2 runs"), "{all}");

    let out = sim("@short");
    let all = text(&out);
    assert!(out.status.success(), "@label names one run:\n{all}");
    assert!(all.contains("6 draws from") && all.contains(&short.display().to_string()),
        "the draws are the labelled run's:\n{all}");
}

/// A `fit.toml` handle names only a run the file alone makes. A run made with
/// an identity-bearing override — `--seed`, `--starts`, `--sweep`,
/// `--decibans-thresh` — is not the file's run, so the file does not name it:
/// `fit summary seedy.toml` used to report the `--seed 7` run although
/// `fit run seedy.toml` computes seed 1.
#[test]
fn a_config_does_not_name_a_run_made_with_overrides() {
    let p = Project::new();
    p.config("seedy.toml", &if2(2));
    let overridden: [&[&str]; 4] = [
        &["--seed", "7"],
        &["--seed", "1", "--starts", "lhs"],
        &["--seed", "1", "--sweep", "gamma=0.25"],
        &["--seed", "1", "--decibans-thresh", "3"],
    ];
    for extra in overridden {
        let mut args = vec!["fit", "run", "seedy.toml"];
        args.extend_from_slice(extra);
        p.ok(&args);
        let out = p.run(&["fit", "summary", "seedy.toml"]);
        assert!(!out.status.success(),
            "the file does not name a run made with {extra:?}:\n{}", text(&out));
    }
    // An override that restates the file is the file's run.
    p.ok(&["fit", "run", "seedy.toml", "--seed", "1"]);
    let made = p.leaves();
    let out = p.ok(&["fit", "summary", "seedy.toml"]);
    assert!(out.contains("2 chains"), "{out}");
    assert_eq!(made.len(), 5, "four overridden runs and the file's own: {made:?}");
}

/// Concurrent `camdl label` of different runs with one label: exactly one
/// takes it. The uniqueness check and the write are one step under a
/// store-wide lock; without it several could find the label free.
#[test]
fn concurrent_labels_give_one_label_to_one_run() {
    let p = Project::new();
    for k in 0..6 {
        // Six runs of one cheap method, told apart by their cooling.
        p.config("scout.toml", &if2(2).replace("cooling = 0.7", &format!("cooling = 0.7{k}")));
        p.ok(&["fit", "run", "scout.toml", "--seed", "1"]);
    }
    let ids: Vec<String> = p.leaves().iter().map(|l| record(l).0).collect();
    assert_eq!(ids.len(), 6);
    for _round in 0..3 {
        let label = format!("race{_round}");
        let children: Vec<std::process::Child> = ids
            .iter()
            .map(|id| {
                Command::new(binary())
                    .current_dir(p.dir())
                    .env("CAMDL_OUTPUT_DIR", p.dir().join("results"))
                    .args(["label", &id[..12], &label])
                    .stdout(std::process::Stdio::null())
                    .stderr(std::process::Stdio::null())
                    .spawn()
                    .unwrap()
            })
            .collect();
        let ok = children.into_iter()
            .map(|c| c.wait_with_output().unwrap().status.success())
            .filter(|s| *s)
            .count();
        let holders = p.leaves().iter().filter(|l| record(l).1.as_deref() == Some(label.as_str())).count();
        assert_eq!((ok, holders), (1, 1), "one run takes {label}");
    }
}

/// A label stored by an older camdl on the problem folder's fit.meta.json is
/// no longer read; `@label` says where it is and prints the command that
/// gives it to a run.
#[test]
fn a_folder_label_from_an_older_camdl_gets_the_relabel_command() {
    let p = Project::new();
    p.config("scout.toml", &if2(2));
    p.ok(&["fit", "run", "scout.toml", "--seed", "1"]);
    let leaf = p.leaves().remove(0);
    let meta = segment_of(&leaf).join("fit.meta.json");
    let mut side: serde_json::Value = serde_json::from_slice(&std::fs::read(&meta).unwrap()).unwrap();
    side["label"] = serde_json::json!("old");
    std::fs::write(&meta, side.to_string()).unwrap();

    let out = p.run(&["fit", "summary", "@old"]);
    let all = text(&out);
    assert!(!out.status.success(), "{all}");
    let id = record(&leaf).0;
    assert!(all.contains(&format!("camdl label {} \"old\"", &id[..12])),
        "names the command that labels the folder's run:\n{all}");
}
