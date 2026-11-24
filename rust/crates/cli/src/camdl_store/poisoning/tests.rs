//! The poisoning test of the run store (run-store proposal, "Guarding against
//! undeclared reads"): a stored result is a function of its identity, so a
//! computation may read a stored file only through a dependency its identity
//! holds.
//!
//! A base store holds one result of every kind ([`BASE`]). Each case runs one
//! command twice, each time on a copy of the base store. The first run records
//! the results the command's invocation names and the dependencies each
//! records (`deps` in its `run.json`). Before the second run, every file of
//! every result in the copy is corrupted except the files those dependencies
//! reach ([`readable`]). A corrupted file keeps its size and modification
//! time, so the store's own check of a result's folder does not notice it, and
//! only a read of its bytes does. The second run must succeed, compute every
//! result the base store did not hold, name the same results, and write each
//! one's files with the digests of the first run. A read of a corrupted file
//! shows up as a failure (a read through the store checks the file's digest)
//! or as other bytes (a plain read of a path built by hand).
//!
//! Every kind camdl registers ([`crate::camdl_store::camdl_kinds`]) must be computed by
//! some case; a kind no case computes fails the test.
//!
//! The binary is `rust/target/release/camdl` and the compiler
//! `ocaml/_build/default/bin/camdlc.exe`, both built by `make build` from the
//! source this test was built from.

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};
use std::process::{Command, Output};

const FORECAST: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/forecast");

/// The forecast tests' two-cadence SIR, with a lineage transition (for the
/// event log) and one quantity.
fn model() -> String {
    let m = std::fs::read_to_string(Path::new(FORECAST).join("two_cadence_sir.camdl")).unwrap();
    let lineage = m.replace("  infection : S --> I", "  #[lineage]\n  infection : S --> I");
    assert_ne!(m, lineage, "the fixture's infection transition moved");
    format!("{lineage}\nquantities {{\n  peak_I = max(I)\n}}\n")
}

/// [`model`] simulated a week longer and with another quantity: another
/// compiled model and other quantities, so a fit of it stores both anew.
fn other_model() -> String {
    let m = model();
    let other = m.replace("to = 63 'days", "to = 70 'days").replace("peak_I = max(I)", "peak_R = max(R)");
    assert_ne!(m.replace("peak_I = max(I)", "peak_R = max(R)"), other, "the fixture's simulate block moved");
    other
}

/// A PMMH fit of both streams.
const FIT: &str = r#"[model]
camdl = "model.camdl"

[data.observations]
daily_cases = "daily_cases.tsv"
weekly_cases = "weekly_cases.tsv"

[estimate]
beta  = { bounds = [0.2, 1.0], start = 0.5 }
gamma = { bounds = [0.1, 0.5], start = 0.25 }

[fixed]
rho = 0.5
k   = 10.0
N0  = 1000
I0  = 5

[method]
algorithm = "pmmh"
backend = "chain_binomial"
chains = 2
particles = 30
iterations = 20
burn_in = 5
thin = 1
"#;

/// The commands that make the base store: a fit labelled `fit`, a second run
/// of its problem with other `[method]` settings labelled `other_method`, and
/// one result of every other kind, each at a seed or setting no case uses.
const BASE: &[&[&str]] = &[
    &["fit", "run", "fit.toml", "--seed", "1", "--label", "fit", "--progress", "none"],
    &["fit", "run", "other_method.toml", "--seed", "1", "--label", "other_method", "--progress", "none"],
    &[
        "simulate", "model.camdl", "--params", "truth.params.toml", "--seed", "99", "--obs-dir", "base-obs",
        "--event-log", "base-events.tsv", "--quantities-out", "base-quantities", "-o", "base-traj.tsv",
    ],
    &[
        "pfilter", "model.camdl", "--params", "truth.params.toml", "--data", "weekly_cases=weekly_cases.tsv",
        "--particles", "40", "--seed", "99", "--trace", "base-trace.tsv", "--pf-health", "base-health.tsv",
        "--save-filtering", "base-filtering.tsv", "--save-filtered-streams", "base-streams.tsv",
        "--save-final-state", "base-final.tsv", "--save-paths", "base-paths.tsv", "--save-prequential",
        "base-prequential",
    ],
    &["predict", "@fit", "--n-draws", "10", "--particles", "30", "--seed", "99", "--label", "pred", "--progress", "none"],
    &[
        "predict", "model.camdl", "--params", "truth.params.toml", "--data", "daily_cases=daily_cases.tsv",
        "--data", "weekly_cases=weekly_cases.tsv", "--n-draws", "10", "--particles", "30", "--seed", "99",
        "--holdout-after", "28", "--label", "point_pred", "--progress", "none",
    ],
    &[
        "profile", "model.camdl", "--fit", "@fit", "--sweep", "gamma=0.25", "--particles", "20", "--iterations",
        "2", "--n-starts", "1",
    ],
];

/// One command whose upstream files are poisoned.
struct Case {
    name: &'static str,
    command: &'static [&'static str],
    /// The label of the stored fit the command names as `@label`.
    named: Option<&'static str>,
}

const CASES: &[Case] = &[
    Case {
        name: "simulate",
        command: &[
            "simulate", "model.camdl", "--params", "truth.params.toml", "--seed", "3", "--obs-dir", "obs",
            "--event-log", "events.tsv", "--quantities-out", "quantities", "-o", "traj.tsv",
        ],
        named: None,
    },
    Case {
        name: "pfilter",
        command: &[
            "pfilter", "model.camdl", "--params", "truth.params.toml", "--data", "daily_cases=other_daily.tsv",
            "--data", "weekly_cases=weekly_cases.tsv", "--particles", "40", "--seed", "3", "--trace", "trace.tsv",
            "--pf-health", "health.tsv", "--save-filtering", "filtering.tsv", "--save-filtered-streams",
            "streams.tsv", "--save-final-state", "final.tsv", "--save-paths", "paths.tsv", "--save-prequential",
            "prequential",
        ],
        named: None,
    },
    Case { name: "fit run", command: &["fit", "run", "other_fit.toml", "--seed", "3", "--progress", "none"], named: None },
    Case {
        name: "pfilter of a stored fit",
        command: &["pfilter", "model.camdl", "--fit", "@fit", "--params", "truth.params.toml", "--particles", "40", "--seed", "3"],
        named: Some("fit"),
    },
    Case {
        name: "simulate from a stored fit's draws",
        command: &["simulate", "model.camdl", "--fit", "@fit", "--draws", "posterior", "--seed", "3", "-o", "draws.tsv"],
        named: Some("fit"),
    },
    Case {
        name: "predict from a fit",
        command: &["predict", "@fit", "--n-draws", "10", "--particles", "30", "--seed", "3", "--progress", "none"],
        named: Some("fit"),
    },
    Case {
        name: "predict from a fixed point",
        command: &[
            "predict", "model.camdl", "--params", "truth.params.toml", "--data", "daily_cases=daily_cases.tsv",
            "--data", "weekly_cases=weekly_cases.tsv", "--n-draws", "10", "--particles", "30", "--seed", "3",
            "--progress", "none",
        ],
        named: None,
    },
    // The score opens the fit run the prediction's draws came from, as
    // `predict @fit` does.
    Case {
        name: "score of a stored prediction",
        command: &["score", "@pred", "--data", "daily_cases=daily_cases.tsv", "--data", "weekly_cases=weekly_cases.tsv"],
        named: Some("fit"),
    },
    Case {
        name: "score of a stored fixed-point prediction",
        command: &["score", "@point_pred", "--data", "daily_cases=daily_cases.tsv", "--data", "weekly_cases=weekly_cases.tsv"],
        named: None,
    },
    Case {
        name: "profile of a stored fit",
        command: &[
            "profile", "model.camdl", "--fit", "@fit", "--sweep", "gamma=0.2,0.3", "--particles", "20",
            "--iterations", "2", "--n-starts", "1",
        ],
        named: Some("fit"),
    },
];

/// The binaries under test, and a directory holding the fixture files and
/// the base store, removed when the fixture is dropped. Each test makes its
/// own: a fixture in a `static` would never be dropped, so its directory would
/// outlive the test run.
struct Fixture {
    camdl: PathBuf,
    camdlc: PathBuf,
    dir: PathBuf,
    _holder: tempfile::TempDir,
}

fn fixture() -> Fixture {
    {
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
        let holder = tempfile::Builder::new().prefix("store_poisoning_fixture").tempdir().unwrap();
        let dir = holder.path().to_path_buf();
        for f in ["daily_cases.tsv", "weekly_cases.tsv", "truth.params.toml"] {
            std::fs::copy(Path::new(FORECAST).join(f), dir.join(f)).unwrap();
        }
        std::fs::write(dir.join("model.camdl"), model()).unwrap();
        // Daily counts no base result reads, so the pfilter case stores a data
        // table anew.
        let daily = std::fs::read_to_string(dir.join("daily_cases.tsv")).unwrap();
        let other: String = daily
            .lines()
            .enumerate()
            .map(|(i, l)| match (i, l.split_once('\t')) {
                (0, _) | (_, None) => format!("{l}\n"),
                (_, Some((t, y))) => format!("{t}\t{}\n", y.parse::<u64>().unwrap() + 1),
            })
            .collect();
        std::fs::write(dir.join("other_daily.tsv"), other).unwrap();
        std::fs::write(dir.join("other.camdl"), other_model()).unwrap();
        std::fs::write(dir.join("fit.toml"), FIT).unwrap();
        std::fs::write(dir.join("other_fit.toml"), FIT.replace("model.camdl", "other.camdl")).unwrap();
        let other_method = FIT.replace("particles = 30", "particles = 45").replace("iterations = 20", "iterations = 24");
        assert_ne!(other_method, FIT, "the fixture's [method] moved");
        std::fs::write(dir.join("other_method.toml"), other_method).unwrap();
        let fx = Fixture { camdl, camdlc, dir, _holder: holder };
        let base = fx.dir.join("base-store");
        for args in BASE {
            let out = self::camdl(&fx, &fx.dir, &base, args);
            assert!(out.status.success(), "the base store: camdl {}\n{}", args.join(" "), String::from_utf8_lossy(&out.stderr));
        }
        fx
    }
}

/// `camdl args` in `cwd` on the store at `store`, with a cleared environment
/// holding `PATH`, `TMPDIR`, a private `HOME` and IR cache, and the compiler
/// under test.
fn camdl(fx: &Fixture, cwd: &Path, store: &Path, args: &[&str]) -> Output {
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
        .env("CAMDL_STORE", store)
        .args(args)
        .output()
        .expect("spawn camdl")
}

/// A fresh working directory holding a copy of the fixture files, and a copy
/// of the base store at each of `stores`. `cp -p` keeps each file's
/// modification time, which the store checks a completed result's files
/// against.
fn workdir(fx: &Fixture, stores: &[&str]) -> tempfile::TempDir {
    let d = tempfile::tempdir().unwrap();
    for e in std::fs::read_dir(&fx.dir).unwrap().flatten() {
        if e.path().is_file() {
            std::fs::copy(e.path(), d.path().join(e.file_name())).unwrap();
        }
    }
    for s in stores {
        let out = Command::new("cp").arg("-Rp").arg(fx.dir.join("base-store")).arg(d.path().join(s)).output().unwrap();
        assert!(out.status.success(), "copying the base store: {}", String::from_utf8_lossy(&out.stderr));
    }
    d
}

/// A completed result as its folder records it.
struct StoredResult {
    dir: PathBuf,
    /// Each file of the manifest, with its role and digest.
    files: BTreeMap<String, (String, Option<String>)>,
    /// `run.json`'s `deps`.
    deps: Vec<serde_json::Value>,
    /// The run ids of the results `run.json`'s `lineage` names.
    lineage: Vec<String>,
}

impl StoredResult {
    /// The result files and their digests.
    fn result_digests(&self) -> BTreeMap<&str, &str> {
        self.files
            .iter()
            .filter(|(_, (role, _))| role == "result")
            .map(|(n, (_, d))| (n.as_str(), d.as_deref().unwrap_or("")))
            .collect()
    }
}

fn read_json(path: &Path) -> serde_json::Value {
    serde_json::from_slice(&std::fs::read(path).unwrap_or_else(|e| panic!("{}: {e}", path.display())))
        .unwrap_or_else(|e| panic!("{}: {e}", path.display()))
}

/// Every completed result under `root`, by run id. The store's own
/// directories (the index, the invocation records, the quarantine) are not
/// results and are skipped.
fn completed_results(root: &Path) -> BTreeMap<String, StoredResult> {
    let mut out = BTreeMap::new();
    let mut stack = vec![root.to_path_buf()];
    while let Some(dir) = stack.pop() {
        for e in std::fs::read_dir(&dir).unwrap().flatten() {
            if e.path().is_dir() && !e.file_name().to_string_lossy().starts_with('.') {
                stack.push(e.path());
            }
        }
        let (ident, run) = (dir.join("identity.json"), dir.join("run.json"));
        if !(ident.is_file() && run.is_file()) {
            continue;
        }
        let ident = read_json(&ident);
        let run = read_json(&run);
        if run["status"] != "completed" {
            continue;
        }
        let files = run["files"]
            .as_object()
            .unwrap()
            .iter()
            .map(|(n, e)| (n.clone(), (e["role"].as_str().unwrap().to_string(), e["sha256"].as_str().map(str::to_string))))
            .collect();
        let lineage = run["lineage"]
            .as_array()
            .into_iter()
            .flatten()
            .filter(|l| l["from"] == "result")
            .map(|l| l["run_id"].as_str().unwrap().to_string())
            .collect();
        let deps = run["deps"].as_array().cloned().unwrap_or_default();
        out.insert(ident["run_id"].as_str().unwrap().to_string(), StoredResult { dir, files, deps, lineage });
    }
    out
}

/// The results the latest invocation on `root` names, as `(kind, run id)`,
/// with the outcome it records for each (`computed`, `reused`, ...).
fn last_invocation(root: &Path) -> BTreeMap<(String, String), String> {
    let dir = root.join(".invocations");
    let latest = std::fs::read_dir(&dir)
        .unwrap()
        .flatten()
        .filter_map(|e| e.file_name().to_string_lossy().strip_suffix(".json")?.parse::<u64>().ok())
        .max()
        .expect("an invocation record");
    read_json(&dir.join(format!("{latest}.json")))["results"]
        .as_array()
        .unwrap()
        .iter()
        .map(|r| {
            let at = (r["kind"].as_str().unwrap().to_string(), r["run_id"].as_str().unwrap().to_string());
            (at, r["outcome"].as_str().unwrap().to_string())
        })
        .collect()
}

/// The fit problem and run the invocation labelled `label` stored on `root`,
/// which `@label` names.
fn labelled_fit(root: &Path, label: &str) -> BTreeSet<String> {
    let mut out = BTreeSet::new();
    for e in std::fs::read_dir(root.join(".invocations")).unwrap().flatten() {
        let record = read_json(&e.path());
        if record["label"] != label {
            continue;
        }
        for r in record["results"].as_array().unwrap() {
            if r["kind"] == "fit" || r["kind"] == "fit_run" {
                out.insert(r["run_id"].as_str().unwrap().to_string());
            }
        }
    }
    assert!(!out.is_empty(), "no invocation on {} is labelled {label}", root.display());
    out
}

/// The stored files a command may read, as `(run id, file)`.
///
/// - Every file of a stored fit the command names on its command line
///   (`named`), and of the results the fit's lineage names. A command opens a
///   named fit whole, each file checked against its digest, while it resolves
///   its inputs; what it takes from the fit enters its identity by value,
///   which the input-class tests check, not this one.
/// - The files `results` and the named fit declare as dependencies, with the
///   files those dependencies declare in turn. A dependency's identity holds
///   the digests of its own dependencies, so a file reached through a chain of
///   dependencies is fixed by the identity at its head.
///
/// A dependency's file list is `*` for every result file, names joined by `,`,
/// or `<file>:<part>` for a part.
fn readable(
    stored: &BTreeMap<String, StoredResult>,
    results: &BTreeSet<(String, String)>,
    named: &BTreeSet<String>,
) -> BTreeSet<(String, String)> {
    let mut keep = BTreeSet::new();
    let opened = named.iter().chain(named.iter().flat_map(|id| &stored[id].lineage));
    for id in opened {
        keep.extend(stored[id].files.keys().map(|n| (id.clone(), n.clone())));
    }
    let mut todo: Vec<String> = results.iter().map(|(_, id)| id.clone()).chain(named.iter().cloned()).collect();
    let mut seen = BTreeSet::new();
    while let Some(id) = todo.pop() {
        if !seen.insert(id.clone()) {
            continue;
        }
        let Some(r) = stored.get(&id) else { continue };
        for dep in &r.deps {
            let run_id = dep["run_id"].as_str().unwrap().to_string();
            let file = dep["file"].as_str().unwrap();
            let names: Vec<String> = if file == "*" {
                stored[&run_id].result_digests().keys().map(|s| s.to_string()).collect()
            } else {
                file.split(',').map(|f| f.split(':').next().unwrap().to_string()).collect()
            };
            keep.extend(names.into_iter().map(|n| (run_id.clone(), n)));
            todo.push(run_id);
        }
    }
    keep
}

/// Corrupts every manifest file of every completed result under `root` that
/// is not in `keep`, keeping its size and modification time. Returns the
/// number of files corrupted.
fn poison(root: &Path, keep: &BTreeSet<(String, String)>) -> usize {
    let mut poisoned = 0;
    for (id, r) in completed_results(root) {
        for name in r.files.keys() {
            if keep.contains(&(id.clone(), name.clone())) {
                continue;
            }
            let path = r.dir.join(name);
            let bytes = std::fs::read(&path).unwrap();
            if bytes.is_empty() {
                continue;
            }
            let mut bad: Vec<u8> = bytes.iter().rev().copied().collect();
            if bad == bytes {
                bad = bytes.iter().map(|b| b ^ 0x01).collect();
            }
            let mtime = std::fs::metadata(&path).unwrap().modified().unwrap();
            std::fs::write(&path, &bad).unwrap();
            std::fs::File::options().write(true).open(&path).unwrap().set_modified(mtime).unwrap();
            poisoned += 1;
        }
    }
    poisoned
}

/// Runs `case` on a copy of the base store, and again on a poisoned copy;
/// returns the kinds the poisoned run computed, or what went wrong.
fn check_case(fx: &Fixture, case: &Case) -> Result<BTreeSet<String>, String> {
    let work = workdir(fx, &["first-store", "poisoned-store"]);
    let (first, poisoned) = (work.path().join("first-store"), work.path().join("poisoned-store"));
    let out = camdl(fx, work.path(), &first, case.command);
    if !out.status.success() {
        return Err(format!("{}: camdl {}\n{}", case.name, case.command.join(" "), String::from_utf8_lossy(&out.stderr)));
    }
    let produced: BTreeSet<(String, String)> = last_invocation(&first).into_keys().collect();
    let stored = completed_results(&first);
    let named = case.named.map(|label| labelled_fit(&first, label)).unwrap_or_default();
    if poison(&poisoned, &readable(&stored, &produced, &named)) == 0 {
        return Err(format!("{}: the base store left nothing to corrupt", case.name));
    }

    let out = camdl(fx, work.path(), &poisoned, case.command);
    if !out.status.success() {
        return Err(format!(
            "{}: with every undeclared upstream file corrupted the command failed, so it read one of them:\n{}",
            case.name,
            String::from_utf8_lossy(&out.stderr)
        ));
    }
    let again = last_invocation(&poisoned);
    if again.keys().ne(produced.iter()) {
        return Err(format!(
            "{}: with every undeclared upstream file corrupted the command named other results\n  first: {produced:?}\n  then:  {:?}",
            case.name,
            again.keys().collect::<Vec<_>>()
        ));
    }
    let reused: Vec<&(String, String)> = again
        .iter()
        .filter(|((kind, _), outcome)| *outcome != "computed" && !crate::result_files::is_content(kind))
        .map(|(at, _)| at)
        .collect();
    if !reused.is_empty() {
        return Err(format!("{}: the base store already holds {reused:?}, so the case computes nothing new there", case.name));
    }
    let after = completed_results(&poisoned);
    let mut differs = Vec::new();
    for (kind, id) in &produced {
        let (Some(a), Some(b)) = (stored.get(id), after.get(id)) else {
            differs.push(format!("{kind} {id}: not completed in both stores"));
            continue;
        };
        if a.result_digests() != b.result_digests() {
            differs.push(format!("{kind} {id}:\n    first: {:?}\n    then:  {:?}", a.result_digests(), b.result_digests()));
        }
    }
    if !differs.is_empty() {
        return Err(format!(
            "{}: results computed with every undeclared upstream file corrupted differ, so they read one of them:\n  {}",
            case.name,
            differs.join("\n  ")
        ));
    }
    Ok(again.into_iter().filter(|(_, outcome)| outcome == "computed").map(|((kind, _), _)| kind).collect())
}

/// Every kind camdl stores is a function of its declared dependencies: with
/// every other stored file corrupted, each command computes the same bytes
/// under the same run ids, and together the cases compute every registered
/// kind.
#[test]
fn every_stored_kind_reads_only_its_declared_dependencies() {
    let fx = &fixture();
    let results: Vec<Result<BTreeSet<String>, String>> = std::thread::scope(|s| {
        let handles: Vec<_> = CASES.iter().map(|case| s.spawn(move || check_case(fx, case))).collect();
        handles.into_iter().map(|h| h.join().expect("a case panicked")).collect()
    });
    let mut failures = Vec::new();
    let mut computed = BTreeSet::new();
    for r in results {
        match r {
            Ok(kinds) => computed.extend(kinds),
            Err(e) => failures.push(e),
        }
    }
    let registered: BTreeSet<String> = crate::camdl_store::camdl_kinds().names().map(str::to_string).collect();
    let uncovered: Vec<&String> = registered.difference(&computed).collect();
    if failures.is_empty() && !uncovered.is_empty() {
        failures.push(format!("no case computes the kinds {uncovered:?}; add a case that does"));
    }
    assert!(failures.is_empty(), "{}", failures.join("\n\n"));
}

/// What two runs of one problem that differ only in their `[method]` give a
/// command reading them.
#[derive(Clone, Copy, PartialEq, Debug)]
enum Across {
    /// The same run ids: the command takes nothing from the method, so the
    /// results are compared byte for byte. If the method were keyed into the
    /// command's identity the ids would differ, and the case would compare
    /// nothing.
    SameIds,
    /// Other run ids for every result: the command reads values the two runs
    /// computed differently (`--draws posterior` reads each run's draws).
    OtherIds,
}

/// The commands that read a stored fit, with `@FIT` for the handle, and what
/// the two runs give each.
const FIT_READERS: &[(&str, &[&str], Across)] = &[
    (
        "pfilter",
        &["pfilter", "model.camdl", "--fit", "@FIT", "--params", "truth.params.toml", "--particles", "40", "--seed", "5"],
        Across::SameIds,
    ),
    (
        "simulate --draws posterior",
        &["simulate", "model.camdl", "--fit", "@FIT", "--draws", "posterior", "--seed", "5", "-o", "draws.tsv"],
        Across::OtherIds,
    ),
    (
        "profile",
        &[
            "profile", "model.camdl", "--fit", "@FIT", "--sweep", "gamma=0.22", "--particles", "20", "--iterations",
            "2", "--n-starts", "1",
        ],
        Across::SameIds,
    ),
];

/// The results `args` computes on `store`, by kind, each with its result
/// files' digests; contents, which hold no run's settings, are left out.
fn results_of(fx: &Fixture, cwd: &Path, store: &Path, args: &[&str]) -> Result<BTreeMap<String, BTreeMap<String, BTreeMap<String, String>>>, String> {
    let out = camdl(fx, cwd, store, args);
    if !out.status.success() {
        return Err(format!("camdl {}\n{}", args.join(" "), String::from_utf8_lossy(&out.stderr)));
    }
    let stored = completed_results(store);
    let mut by_kind: BTreeMap<String, BTreeMap<String, BTreeMap<String, String>>> = BTreeMap::new();
    for (kind, id) in last_invocation(store).into_keys() {
        if crate::result_files::is_content(&kind) {
            continue;
        }
        let digests = stored[&id].result_digests().into_iter().map(|(n, d)| (n.to_string(), d.to_string())).collect();
        by_kind.entry(kind).or_default().insert(id, digests);
    }
    Ok(by_kind)
}

/// A run's `[method]` reaches a result computed from the run only through
/// the result's identity. Each command is given two runs of one problem that
/// differ only in their `[method]` settings. `pfilter` and `profile` take
/// nothing from the method: they must make the same run ids, and each result
/// must then have the same bytes, since one run id with two byte sets would be
/// the method read and not keyed. `simulate --draws posterior` reads each
/// run's draws, and must make other run ids for every cell.
#[test]
fn a_fit_run_s_method_reaches_what_is_computed_from_it_only_through_its_id() {
    let fx = &fixture();
    let mut failures = Vec::new();
    for (name, template, across) in FIT_READERS {
        // Each run on its own copy of the base store, so the second cannot
        // reuse what the first stored under a shared run id.
        let work = workdir(fx, &["store-a", "store-b"]);
        let (store_a, store_b) = (work.path().join("store-a"), work.path().join("store-b"));
        let with = |label: &str| -> Vec<String> { template.iter().map(|a| a.replace("@FIT", &format!("@{label}"))).collect() };
        let (a, b) = (with("fit"), with("other_method"));
        let a: Vec<&str> = a.iter().map(String::as_str).collect();
        let b: Vec<&str> = b.iter().map(String::as_str).collect();
        let (first, second) = match (results_of(fx, work.path(), &store_a, &a), results_of(fx, work.path(), &store_b, &b)) {
            (Ok(x), Ok(y)) => (x, y),
            (Err(e), _) | (_, Err(e)) => {
                failures.push(format!("{name}: {e}"));
                continue;
            }
        };
        let ids = |m: &BTreeMap<String, BTreeMap<String, BTreeMap<String, String>>>| -> BTreeSet<(String, String)> {
            m.iter().flat_map(|(k, ids)| ids.keys().map(move |id| (k.clone(), id.clone()))).collect()
        };
        let (ids_a, ids_b) = (ids(&first), ids(&second));
        if ids_a.is_empty() {
            failures.push(format!("{name}: the command computed no result"));
            continue;
        }
        match across {
            Across::SameIds if ids_a != ids_b => failures.push(format!(
                "{name}: the two runs gave other run ids, so the method is in the identity and nothing was compared\n  \
                 @fit:          {ids_a:?}\n  @other_method: {ids_b:?}"
            )),
            Across::SameIds => {
                for (kind, id) in &ids_a {
                    let (x, y) = (&first[kind][id], &second[kind][id]);
                    if x != y {
                        failures.push(format!(
                            "{name}: the {kind} {id} of @fit and of @other_method share a run id and differ:\n    @fit:          {x:?}\n    @other_method: {y:?}"
                        ));
                    }
                }
            }
            Across::OtherIds => {
                let shared: Vec<&(String, String)> = ids_a.intersection(&ids_b).collect();
                if !shared.is_empty() {
                    failures.push(format!("{name}: the two runs' draws gave the run ids {shared:?} to both"));
                }
            }
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n\n"));
}
