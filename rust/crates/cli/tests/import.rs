//! `camdl import`: a fit run, its fit problem and a prediction inside the
//! run, made in one run store under another build and copied as `pull`
//! copies them, placed in a second store and recorded there.
//!
//! End to end because what is tested is what the second store's commands
//! then see (`list`, `@LABEL`, `verify`, a prediction from the imported run)
//! and the exit status of a refused copy. The fit is the SEIR golden IR with
//! one estimated parameter under PGAS at 2 chains x 20 particles x 4 sweeps.

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

/// The environment variable that puts the other store's results under
/// another build: every set `CAMDL_EXPERIMENTAL_*` is part of the ambient
/// identity.
const OTHER_BUILD: (&str, &str) = ("CAMDL_EXPERIMENTAL_RECEIVE_TEST", "remote");

/// A scratch directory holding the data, a fit config and two stores:
/// `remote/`, where the results are made, and `local/`, which imports them.
struct Project {
    tmp: tempfile::TempDir,
}

impl Project {
    fn new() -> Project {
        assert!(binary().exists(), "release camdl missing: {} — run `make build-rust`", binary().display());
        assert!(camdlc_bin().exists(), "camdlc.exe missing: {} — run `make build-ocaml`", camdlc_bin().display());
        let tmp = tempfile::tempdir().unwrap();
        std::fs::write(tmp.path().join("obs.tsv"), "time\tweekly_cases\n7\t1\n14\t2\n21\t3\n28\t4\n35\t5\n").unwrap();
        let config = format!(
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
algorithm = "pgas"
backend = "chain_binomial"
chains = 2
particles = 20
sweeps = 4
burn_in = 1
thin = 1

[config]
dt = 1.0
"#,
            ir = golden_ir().display(),
        );
        std::fs::write(tmp.path().join("fit.toml"), config).unwrap();
        Project { tmp }
    }

    fn dir(&self) -> &Path {
        self.tmp.path()
    }

    fn remote(&self) -> PathBuf {
        self.dir().join("remote")
    }

    fn local(&self) -> PathBuf {
        self.dir().join("local")
    }

    fn copy(&self) -> PathBuf {
        self.dir().join("pull-staging")
    }

    fn run(&self, store: &Path, env: &[(&str, &str)], args: &[&str]) -> Output {
        let mut c = Command::new(binary());
        c.env_remove("CLICOLOR_FORCE")
            .current_dir(self.dir())
            .env("CAMDL_STORE", store)
            .env("CAMDL_SKIP_VERSION_CHECK", "1")
            .env("CAMDLC", camdlc_bin());
        for (k, v) in env {
            c.env(k, v);
        }
        c.args(args).output().expect("spawn camdl")
    }

    fn ok(&self, store: &Path, env: &[(&str, &str)], args: &[&str]) -> String {
        let out = self.run(store, env, args);
        let all = text(&out);
        assert!(out.status.success(), "`camdl {}` failed:\n{all}", args.join(" "));
        all
    }

    /// A fit run labelled `m4w`, and a prediction from it, in `remote/`.
    fn make_remote(&self, env: &[(&str, &str)]) {
        self.ok(&self.remote(), env, &["fit", "run", "fit.toml", "--seed", "1", "--label", "m4w"]);
        self.ok(&self.remote(), env, &["predict", "@m4w", "--mode", "free_forward", "--label", "m4w_pred"]);
    }

    /// What `pull` hands over: the remote store's `store.json`, invocation
    /// records and result trees, without its index or owners' beats.
    fn pull(&self) {
        copy_tree(&self.remote(), &self.copy(), &|name| !name.starts_with('.') || name == ".invocations");
    }
}

fn text(out: &Output) -> String {
    format!("{}{}", String::from_utf8_lossy(&out.stdout), String::from_utf8_lossy(&out.stderr))
}

/// Copies `from` to `to`; at the top level only the entries `keep` accepts.
fn copy_tree(from: &Path, to: &Path, keep: &dyn Fn(&str) -> bool) {
    std::fs::create_dir_all(to).unwrap();
    for e in std::fs::read_dir(from).unwrap().flatten() {
        let name = e.file_name().to_string_lossy().into_owned();
        if !keep(&name) {
            continue;
        }
        if e.path().is_dir() {
            copy_tree(&e.path(), &to.join(&name), &|_| true);
        } else {
            std::fs::copy(e.path(), to.join(&name)).unwrap();
        }
    }
}

/// The result folders of each fit kind under `root`, relative to it.
fn fit_folders(root: &Path) -> Vec<PathBuf> {
    let mut out = Vec::new();
    for kind in ["fit", "fit_run", "prediction"] {
        for f in fit_store_support::folders_of(root, kind) {
            out.push(f.strip_prefix(root).unwrap().to_path_buf());
        }
    }
    out
}

#[test]
fn import_places_and_records_a_pulled_fit() {
    let p = Project::new();
    p.make_remote(&[OTHER_BUILD]);
    p.pull();
    let report = p.ok(&p.local(), &[], &["import", "pull-staging", "--label", "m4w_cloud"]);
    assert!(report.contains("placed"), "{report}");

    // The fit problem, the run inside it and the prediction inside the run
    // are at the paths they had, under the other build's directory.
    let remote = fit_folders(&p.remote());
    assert_eq!(remote.len(), 3, "premise: a problem, a run and a prediction: {remote:?}");
    assert_eq!(fit_folders(&p.local()), remote);
    let run = fit_store_support::the_run(&p.local());
    assert!(fit_store_support::predictions(&p.local())[0].starts_with(&run));
    assert!(run.starts_with(&fit_store_support::problems(&p.local())[0]));
    assert!(!p.copy().exists(), "the copy is consumed");

    // `list` shows the run under the import's label, `@LABEL` names it, and
    // every imported result verifies.
    let list = p.ok(&p.local(), &[], &["list", "--kind", "fit"]);
    assert!(list.contains("m4w_cloud"), "{list}");
    let summary = p.ok(&p.local(), &[], &["fit", "summary", "@m4w_cloud"]);
    assert!(summary.contains("pgas"), "{summary}");
    p.ok(&p.local(), &[], &["verify", "@m4w_cloud"]);

    // This build predicts from the imported run: its prediction sits beside
    // the imported one, inside the run.
    p.ok(&p.local(), &[], &["predict", "@m4w_cloud", "--mode", "free_forward", "--n-draws", "2"]);
    let preds = fit_store_support::predictions(&p.local());
    assert_eq!(preds.len(), 2, "{preds:?}");
    assert!(preds.iter().all(|d| d.starts_with(&run)));
    // The same fit run here goes under this build's directory, which is not
    // the imported one.
    p.ok(&p.local(), &[], &["fit", "run", "fit.toml", "--seed", "1"]);
    let top = |d: &Path| d.strip_prefix(p.local()).unwrap().components().next().unwrap().as_os_str().to_owned();
    let tops: std::collections::BTreeSet<_> = fit_store_support::runs(&p.local()).iter().map(|d| top(d)).collect();
    assert_eq!(tops.len(), 2, "one fit run per build directory: {tops:?}");
    assert!(tops.contains(&top(&run)));
}

/// A copy without the quantities the fit run's lineage names is refused, so
/// no imported run is left whose prediction here could not find them.
#[test]
fn import_refuses_a_copy_without_the_runs_quantities() {
    let p = Project::new();
    p.make_remote(&[OTHER_BUILD]);
    p.pull();
    for kind in ["prediction", "quantities"] {
        for d in fit_store_support::folders_of(&p.copy(), kind) {
            std::fs::remove_dir_all(d).unwrap();
        }
    }
    let out = p.run(&p.local(), &[], &["import", "pull-staging", "--label", "m4w_cloud"]);
    let all = text(&out);
    assert_eq!(out.status.code(), Some(1), "{all}");
    assert!(all.contains("dependency") && all.contains("quantities"), "{all}");
    assert!(fit_folders(&p.local()).is_empty(), "nothing is placed");
}

#[test]
fn import_refuses_a_damaged_copy_and_places_nothing() {
    let p = Project::new();
    p.make_remote(&[OTHER_BUILD]);
    p.pull();
    let run = fit_store_support::the_run(&p.copy());
    let file = std::fs::read_dir(&run)
        .unwrap()
        .flatten()
        .map(|e| e.path())
        .find(|f| f.extension().is_some_and(|x| x == "tsv"))
        .expect("a fit run holds a .tsv");
    let mut bytes = std::fs::read(&file).unwrap();
    let last = bytes.len() - 2;
    bytes[last] = if bytes[last] == b'0' { b'1' } else { b'0' };
    std::fs::write(&file, &bytes).unwrap();

    let out = p.run(&p.local(), &[], &["import", "pull-staging", "--label", "m4w_cloud"]);
    let all = text(&out);
    assert_eq!(out.status.code(), Some(1), "{all}");
    assert!(all.contains("cannot be imported") && all.contains("recorded digest"), "{all}");
    assert!(fit_folders(&p.local()).is_empty(), "nothing is placed");
    assert!(p.copy().join("store.json").exists(), "the copy is back where it was given");
    let list = p.ok(&p.local(), &[], &["list"]);
    assert!(!list.contains("m4w_cloud"), "{list}");
}

/// A copy without the compiled model the fit depends on is refused, naming
/// the model, so no imported run is left that a prediction here could not
/// read.
#[test]
fn import_refuses_a_copy_without_its_compiled_model() {
    let p = Project::new();
    p.make_remote(&[OTHER_BUILD]);
    p.pull();
    let models = fit_store_support::folders_of(&p.copy(), "compiled_model");
    assert_eq!(models.len(), 1, "premise: one compiled model: {models:?}");
    std::fs::remove_dir_all(&models[0]).unwrap();

    let out = p.run(&p.local(), &[], &["import", "pull-staging", "--label", "m4w_cloud"]);
    let all = text(&out);
    assert_eq!(out.status.code(), Some(1), "{all}");
    assert!(all.contains("dependency") && all.contains("compiled_model"), "{all}");
    assert!(fit_folders(&p.local()).is_empty(), "nothing is placed");
}

/// `import --manifest` lists every result it placed, each as the remote
/// command's own manifest lists it; importing the results again, now
/// present, writes the same entries.
#[test]
fn import_writes_the_manifest_of_what_it_placed_or_found() {
    let p = Project::new();
    p.ok(&p.remote(), &[OTHER_BUILD], &["fit", "run", "fit.toml", "--seed", "1", "--label", "m4w", "--manifest", "remote.json"]);
    p.pull();
    p.ok(&p.local(), &[], &["import", "pull-staging", "--label", "m4w_cloud", "--manifest", "local.json"]);
    let read = |name: &str| -> serde_json::Value { serde_json::from_slice(&std::fs::read(p.dir().join(name)).unwrap()).unwrap() };
    let (remote, local) = (read("remote.json"), read("local.json"));
    assert_eq!(local["label"], "m4w_cloud");
    assert_ne!(local["camdl"]["ambient_key"], remote["camdl"]["ambient_key"], "premise: the copy was made under another build");
    let entries = |m: &serde_json::Value| -> std::collections::BTreeMap<String, serde_json::Value> {
        m["results"].as_array().unwrap().iter().map(|r| (r["run_id"].as_str().unwrap().to_string(), r.clone())).collect()
    };
    let (remote_entries, local_entries) = (entries(&remote), entries(&local));
    assert_eq!(local_entries, remote_entries, "the imported results, each as the remote manifest lists it");
    let kinds: Vec<&str> = local["results"].as_array().unwrap().iter().map(|r| r["kind"].as_str().unwrap()).collect();
    assert!(kinds.contains(&"fit_run") && kinds.contains(&"fit"), "{kinds:?}");
    // Parents first: the fit problem before the run that sits inside it.
    let at = |k: &str| kinds.iter().position(|x| *x == k).unwrap();
    assert!(at("fit") < at("fit_run"), "the problem is listed before its run: {kinds:?}");

    // The results are now here: a second import of the same copy finds
    // every one present and lists the same entries.
    p.pull();
    p.ok(&p.local(), &[], &["import", "pull-staging", "--label", "m4w_cloud", "--manifest", "again.json"]);
    assert_eq!(entries(&read("again.json")), local_entries);
}

#[test]
fn import_of_results_already_here_leaves_them_in_place() {
    let p = Project::new();
    // One build for both stores, so the fit computed here has the run ids of
    // the fit in the copy.
    p.make_remote(&[]);
    p.pull();
    p.ok(&p.local(), &[], &["fit", "run", "fit.toml", "--seed", "1"]);
    let run = fit_store_support::the_run(&p.local());
    let report = p.ok(&p.local(), &[], &["import", "pull-staging", "--label", "m4w_cloud"]);
    let present: Vec<&str> = report.lines().filter(|l| l.starts_with("present")).collect();
    assert!(present.iter().any(|l| l.contains("fit_run")), "{report}");
    assert!(report.lines().any(|l| l.starts_with("placed") && l.contains("prediction")), "{report}");
    assert_eq!(fit_store_support::the_run(&p.local()), run);
    assert!(fit_store_support::predictions(&p.local())[0].starts_with(&run));
    p.ok(&p.local(), &[], &["verify"]);
}

/// `import --link-root` links the imported results by the import's label,
/// as every other storing command does: the fit run under `fits/`, the
/// prediction inside it under `predictions/`, each link resolving to the
/// file `camdl path` names through the import's manifest.
#[test]
fn import_links_what_it_received() {
    let p = Project::new();
    p.make_remote(&[OTHER_BUILD]);
    p.pull();
    p.ok(&p.local(), &[], &["import", "pull-staging", "--label", "m4w_cloud", "--manifest", "local.json", "--link-root", "links"]);
    let path = |name: &str| -> PathBuf {
        let out = p.run(&p.local(), &[], &["path", "--manifest", "local.json", name]);
        assert!(out.status.success(), "{}", text(&out));
        PathBuf::from(String::from_utf8(out.stdout).unwrap().trim())
    };
    let link = p.dir().join("links/fits/m4w_cloud/draws.tsv");
    assert!(std::fs::symlink_metadata(&link).unwrap().file_type().is_symlink());
    assert_eq!(link.canonicalize().unwrap(), path("draws").canonicalize().unwrap());
    let report = p.dir().join("links/predictions/m4w_cloud/report.json");
    assert_eq!(report.canonicalize().unwrap(), path("report").canonicalize().unwrap());
}

/// A model in camdl source, with a documented parameter, for the tests of
/// the model sources an import follows: each `fit run` of a file stores its
/// model source and records it on the compiled model's listing.
const SIR: &str = r#"
time_unit = 'days
compartments { S, I, R }
parameters {
  #' The transmission rate, per day.
  beta  : rate         in [0.05, 1.0]  ~ log_normal(mu = -1.0, sigma = 0.5)
  gamma : rate         in [0.01, 0.5]  ~ log_normal(mu = -2.0, sigma = 0.5)
  N0    : count
  I0    : count
}
let N = S + I + R
transitions {
  infection : S --> I  @ beta * S * I / N
  recovery  : I --> R  @ gamma * I
}
init { S = N0 - I0  I = I0 }
observations {
  cases {
    columns       { time : time, cases : count }
    covers        = closing_at(time, 1 'days)
    projected     = incidence(infection)
    emit_schedule = every 1 'days
    cases ~ poisson(rate = projected)
  }
}
simulate { from = 0 'days  to = 20 'days }
"#;

const SIR_FIT: &str = r#"
[model]
camdl = "model.camdl"
[data]
file = "cases.tsv"
[estimate]
beta = { bounds = [0.1, 0.9], start = 0.4 }
[fixed]
gamma = 0.2
N0 = 1000
I0 = 5
[method]
algorithm = "if2"
backend = "chain_binomial"
chains = 1
particles = 50
iterations = 2
cooling = 0.7
starts = "from_prior"
"#;

/// A project whose `remote/` store holds one fit run, fitted from
/// `model.camdl` with each of `models` in turn as its text, under the
/// matching label.
fn sir_remote(models: &[(&str, String)]) -> Project {
    let p = Project::new();
    let cases = [3, 6, 9, 14, 20, 27, 33, 38, 40, 39, 36, 31, 26, 21, 16, 12, 9, 7, 5, 4];
    let mut tsv = String::from("time\tcases\n");
    for (t, c) in cases.iter().enumerate() {
        tsv.push_str(&format!("{}\t{c}\n", t + 1));
    }
    std::fs::write(p.dir().join("cases.tsv"), tsv).unwrap();
    std::fs::write(p.dir().join("sir.toml"), SIR_FIT).unwrap();
    for (label, model) in models {
        std::fs::write(p.dir().join("model.camdl"), model).unwrap();
        p.ok(&p.remote(), &[], &["fit", "run", "sir.toml", "--seed", "1", "--label", label, "--progress", "none"]);
    }
    p
}

/// The listings of `kind` in the store `root`'s invocation record `seq`.
fn listings(root: &Path, seq: u64, kind: &str) -> Vec<serde_json::Value> {
    let record = fit_store_support::json(&root.join(format!(".invocations/{seq}.json")));
    record["results"].as_array().unwrap().iter().filter(|r| r["kind"] == kind).cloned().collect()
}

/// A copy without the model sources its records name as the sources of its
/// compiled model is refused, naming the missing source, and nothing is
/// placed: a fit imported without its model file would show no
/// documentation and predict with no known quantities.
#[test]
fn import_refuses_a_copy_without_the_model_sources_its_records_name() {
    let edited = SIR.replace("The transmission rate, per day.", "Transmission rate (edited doc).");
    let p = sir_remote(&[("f", SIR.to_string()), ("g", edited)]);
    p.pull();
    let sources = fit_store_support::folders_of(&p.copy(), "model_source");
    assert_eq!(sources.len(), 2, "premise: one model source per edit of the file: {sources:?}");
    for d in &sources {
        std::fs::remove_dir_all(d).unwrap();
    }
    let out = p.run(&p.local(), &[], &["import", "pull-staging", "--label", "got"]);
    let all = text(&out);
    assert_eq!(out.status.code(), Some(1), "{all}");
    assert!(all.contains("model_source") && all.contains("source"), "{all}");
    assert!(fit_folders(&p.local()).is_empty(), "nothing is placed");
}

/// A fit run that two invocations fitted from files with different
/// quantities blocks is ambiguous by its run id in the store it was made in.
/// The import keeps one listing per source the copy's records gave the
/// compiled model, so the run id is refused here too, saying the other
/// store's labels did not travel; those labels name nothing here.
#[test]
fn import_keeps_each_source_so_an_ambiguous_run_stays_ambiguous() {
    let with_q = format!("{SIR}\nquantities {{\n  total_inf = final(R)\n}}\n");
    let p = sir_remote(&[("plain", SIR.to_string()), ("withq", with_q)]);
    let id = fit_store_support::run_id(&fit_store_support::the_run(&p.remote()));
    let predict = |store: &Path, handle: &str| p.run(store, &[], &["predict", handle, "--mode", "free_forward", "--progress", "none"]);
    let out = predict(&p.remote(), &id);
    assert!(!out.status.success(), "premise: ambiguous in the store it was made in: {}", text(&out));

    // Without the second quantities, the second source's dependency is
    // missing, and the copy is refused.
    p.pull();
    let quantities = fit_store_support::folders_of(&p.copy(), "quantities");
    assert_eq!(quantities.len(), 2, "premise: one quantities result per block: {quantities:?}");
    let named = |d: &Path| fit_store_support::json(&d.join("identity.json"))["run_id"].as_str().unwrap().to_string();
    let second_source = listings(&p.remote(), 2, "compiled_model")[0]["source"]["run_id"].as_str().unwrap().to_string();
    let source_dir = fit_store_support::folders_of(&p.copy(), "model_source").into_iter().find(|d| named(d) == second_source).unwrap();
    let source_run = fit_store_support::json(&source_dir.join("run.json"));
    let second_q = source_run["deps"].as_array().unwrap().iter().find(|d| d["kind"] == "quantities").unwrap()["run_id"].clone();
    std::fs::remove_dir_all(quantities.iter().find(|d| named(d) == second_q.as_str().unwrap()).unwrap()).unwrap();
    let out = p.run(&p.local(), &[], &["import", "pull-staging", "--label", "got"]);
    assert_eq!(out.status.code(), Some(1), "{}", text(&out));
    assert!(fit_folders(&p.local()).is_empty(), "nothing is placed");
    std::fs::remove_dir_all(p.copy()).unwrap();

    // With both, the import lists the compiled model once per source.
    p.pull();
    p.ok(&p.local(), &[], &["import", "pull-staging", "--label", "got"]);
    let models = listings(&p.local(), 1, "compiled_model");
    let sources: std::collections::BTreeSet<String> = models.iter().map(|m| m["source"]["run_id"].as_str().unwrap().to_string()).collect();
    assert_eq!(models.len(), 2, "one listing per source: {models:?}");
    assert_eq!(sources.len(), 2, "{models:?}");
    for handle in [id.as_str(), "@got"] {
        let out = predict(&p.local(), handle);
        let all = text(&out);
        assert!(!out.status.success(), "{handle}: {all}");
        assert!(all.contains("different quantities") && all.contains("did not travel"), "{handle}: {all}");
    }
    for label in ["@plain", "@withq"] {
        let out = predict(&p.local(), label);
        assert!(!out.status.success(), "the other store's label {label} names nothing here: {}", text(&out));
    }
}

/// A result folder of a kind this build does not register, such as a model
/// presentation an older build stored, is not placed and is reported; the
/// rest of the copy is imported.
#[test]
fn import_reports_and_leaves_a_folder_of_an_unregistered_kind() {
    let p = sir_remote(&[("f", SIR.to_string())]);
    p.pull();
    let source = fit_store_support::folders_of(&p.copy(), "model_source").pop().unwrap();
    let kind_dir = source.parent().unwrap().parent().unwrap().join("model_presentation");
    let old = kind_dir.join(source.file_name().unwrap());
    fit_store_support::copy_tree(&source, &old);
    let ident = old.join("identity.json");
    let rewritten = std::fs::read_to_string(&ident).unwrap().replace("\"model_source\"", "\"model_presentation\"");
    std::fs::write(&ident, rewritten).unwrap();

    let report = p.ok(&p.local(), &[], &["import", "pull-staging", "--label", "got"]);
    assert!(report.contains("not placed") && report.contains("model_presentation"), "{report}");
    assert!(fit_folders(&p.local()).len() == 2, "the problem and the run are placed");
    let kinds: Vec<String> = std::fs::read_dir(p.local()).unwrap().flatten().flat_map(|b| std::fs::read_dir(b.path()).into_iter().flatten().flatten().map(|k| k.file_name().to_string_lossy().into_owned()).collect::<Vec<_>>()).collect();
    assert!(!kinds.iter().any(|k| k == "model_presentation"), "not placed: {kinds:?}");
}
