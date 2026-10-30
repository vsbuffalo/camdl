//! gh#990: every user-facing chain index counts from 0.
//!
//! camdl numbers every index a user sees from 0 — `draws.tsv`'s `chain`
//! column always did, `simulate`'s `replicate` / `draw` / `point_id` do, and
//! with gh#990 the fit side does too: the `chain_N/` directories, the
//! per-chain tables of `fit summary` (text and json), `chain_starts.tsv`'s
//! `chain_id`, `fit_state.toml`'s `best_chain` and `chain_eval_ids`,
//! `--exclude-chains`, and the `chain` column `fit predict --by-chain` adds.
//! One base means there is nothing to convert, so the class of off-by-one that
//! gh#781, gh#823, gh#910 and gh#972 each fixed at one site cannot recur.
//!
//! Pinned here on real two-chain leaves, one IF2 and one PGAS, from the
//! committed SEIR golden IR. Cheap on purpose: nothing depends on convergence,
//! only on what the leaf says about its own chains.
//!
//! The third test is the migration guard: a leaf written by a camdl that
//! numbered its directories from 1 (`chain_1/`, `chain_2/`, no `chain_0/`)
//! is refused by every reader that addresses a chain by name or reads its
//! posterior cloud — `fit summary` (text, json, md), `fit predict`,
//! `fit table --quantity`, `compare`, `fit run --resume` — rather than read
//! with every chain shifted by one.

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};
use std::process::Command;

/// Where the latest `fit predict` of a run in `seg` wrote:
/// `<seg>/predict/<run id>[-<options key>]/` (gh#896), the directory whose
/// `report*.json` (one written by every predict) is newest. A path that does not
/// exist while nothing has been predicted.
fn predict_out(seg: &std::path::Path) -> std::path::PathBuf {
    std::fs::read_dir(seg.join("predict"))
        .into_iter().flatten().flatten().map(|e| e.path())
        .filter_map(|d| {
            // Newest `report*.json` (keyed by chain selection) in the directory.
            let t = std::fs::read_dir(&d).ok()?.flatten()
                .filter(|e| {
                    let n = e.file_name().to_string_lossy().into_owned();
                    n.starts_with("report") && n.ends_with(".json")
                })
                .filter_map(|e| e.metadata().and_then(|m| m.modified()).ok())
                .max()?;
            Some((t, d))
        })
        .max_by_key(|(t, _)| *t)
        .map(|(_, d)| d)
        .unwrap_or_else(|| seg.join("predict").join("none"))
}


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

fn require_tools() {
    assert!(
        binary().exists(),
        "release camdl binary missing: {} — run `make build-rust` or `make test`",
        binary().display()
    );
    assert!(
        camdlc_bin().exists(),
        "camdlc.exe missing: {} — run `make build-ocaml`",
        camdlc_bin().display()
    );
}

/// The problem both fits share: one estimated parameter, the rest fixed.
fn problem(dir: &Path) -> String {
    let data = dir.join("obs.tsv");
    std::fs::write(&data, "time\tweekly_cases\n7\t1\n14\t2\n21\t3\n28\t4\n35\t5\n").unwrap();
    format!(
        r#"
[model]
camdl = "{ir}"

[data.observations]
weekly_cases = "{data}"

[estimate.beta]
bounds = [0.01, 0.5]
prior  = {{ log_normal = {{ mu = -1.2, sigma = 0.4 }} }}

[fixed]
sigma    = 0.25
gamma    = 0.3
rho      = 0.5
k        = 10.0
p_detect = 0.5
N0       = 1000
I0       = 1

[config]
dt = 1.0
"#,
        ir = golden_ir().display(),
        data = data.display()
    )
}

const IF2_METHOD: &str = r#"
[method]
algorithm   = "if2"
backend     = "chain_binomial"
chains      = 2
particles   = 20
iterations  = 2
cooling     = 0.5
starts      = "uniform_unconstrained"
loglik_eval = { n_particles = 50, n_replicates = 2 }
dt_check    = { enabled = false }
"#;

fn pgas_method(sweeps: usize) -> String {
    format!(
        r#"
[method]
algorithm = "pgas"
backend   = "chain_binomial"
chains    = 2
particles = 20
sweeps    = {sweeps}
burn_in   = 2
thin      = 1
starts    = "uniform_unconstrained"
"#
    )
}

fn camdl(dir: &Path, args: &[&str]) -> std::process::Output {
    Command::new(binary())
        .current_dir(dir)
        .env("CAMDL_OUTPUT_DIR", dir.join("results"))
        .env("CAMDL_SKIP_VERSION_CHECK", "1")
        .env("CAMDLC", camdlc_bin())
        .args(args)
        .output()
        .expect("spawn camdl")
}

fn ok(out: &std::process::Output, what: &str) -> String {
    let stderr = String::from_utf8_lossy(&out.stderr).to_string();
    assert!(out.status.success(), "{what} failed:\nstderr={stderr}");
    stderr
}

/// The single fit segment directory under `<dir>/results/fits/`.
fn segment(dir: &Path) -> PathBuf {
    let fits = dir.join("results").join("fits");
    let mut segs: Vec<PathBuf> = std::fs::read_dir(&fits)
        .unwrap_or_else(|e| panic!("{}: {e}", fits.display()))
        .flatten()
        .map(|e| e.path())
        .filter(|p| p.is_dir())
        .collect();
    assert_eq!(segs.len(), 1, "one fit segment under {}: {segs:?}", fits.display());
    segs.pop().unwrap()
}

/// The leaf holding `fit_state.toml` under a segment.
fn leaf(segment: &Path) -> PathBuf {
    let mut stack = vec![segment.to_path_buf()];
    while let Some(d) = stack.pop() {
        if d.join("fit_state.toml").is_file() {
            return d;
        }
        if let Ok(es) = std::fs::read_dir(&d) {
            for e in es.flatten() {
                if e.path().is_dir() {
                    stack.push(e.path());
                }
            }
        }
    }
    panic!("no fit_state.toml under {}", segment.display());
}

/// The trailing integer of every `chain_<N>/` directory on the leaf.
fn chain_dirs(leaf: &Path) -> BTreeSet<usize> {
    std::fs::read_dir(leaf)
        .unwrap()
        .flatten()
        .filter(|e| e.path().is_dir())
        .filter_map(|e| e.file_name().to_str()?.strip_prefix("chain_")?.parse().ok())
        .collect()
}

/// The `chain_id` column of `chain_starts.tsv`, deduplicated.
fn chain_start_ids(leaf: &Path) -> BTreeSet<usize> {
    let text = std::fs::read_to_string(leaf.join("chain_starts.tsv")).unwrap();
    let mut body = text.lines().filter(|l| !l.starts_with('#') && !l.trim().is_empty());
    let cols: Vec<&str> = body.next().expect("chain_starts.tsv header").split('\t').collect();
    let id = cols.iter().position(|c| *c == "chain_id").expect("chain_id column");
    body.map(|l| l.split('\t').nth(id).unwrap().parse().unwrap()).collect()
}

/// The first column of `draws.tsv`, deduplicated — the 0-based join key.
fn draws_chains(leaf: &Path) -> BTreeSet<usize> {
    let text = std::fs::read_to_string(leaf.join("draws.tsv")).unwrap();
    let mut lines = text.lines().filter(|l| !l.trim().is_empty());
    let header: Vec<&str> = lines.next().unwrap().split('\t').collect();
    assert_eq!(header[0], "chain");
    lines.map(|l| l.split('\t').next().unwrap().parse().unwrap()).collect()
}

/// The chain numbers in a `fit summary` text table: the lines after the one
/// containing `section_marker` whose first token is an integer, up to the
/// first blank line.
fn table_chain_ids(summary: &str, section_marker: &str) -> BTreeSet<usize> {
    let mut lines = summary.lines().skip_while(|l| !l.contains(section_marker));
    lines
        .next()
        .unwrap_or_else(|| panic!("no `{section_marker}` section in:\n{summary}"));
    lines
        .take_while(|l| !l.trim().is_empty())
        .filter_map(|l| l.split_whitespace().next()?.parse().ok())
        .collect()
}

/// `(chain N)` on the stderr line containing `marker`.
fn stderr_chain(stderr: &str, marker: &str) -> usize {
    let line = stderr
        .lines()
        .find(|l| l.contains(marker))
        .unwrap_or_else(|| panic!("no stderr line containing {marker:?}:\n{stderr}"));
    let after = line.split_once("(chain ").unwrap_or_else(|| panic!("no `(chain N)` on {line:?}")).1;
    after.chars().take_while(char::is_ascii_digit).collect::<String>().parse().unwrap()
}

fn zero_one() -> BTreeSet<usize> {
    [0usize, 1].into_iter().collect()
}

/// IF2: directories, `chain_starts.tsv`, `fit_state.toml`'s `best_chain` and
/// `chain_eval_ids`, the stderr winner line, and `fit summary`'s per-chain
/// loglik-eval table in text and json all name the two chains 0 and 1.
#[test]
fn an_if2_leaf_numbers_its_chains_from_zero_on_every_surface() {
    require_tools();
    let tmp = tempfile::tempdir().unwrap();
    let dir = tmp.path();
    let toml = dir.join("optimum.toml");
    std::fs::write(&toml, format!("{}{IF2_METHOD}", problem(dir))).unwrap();

    let stderr = ok(&camdl(dir, &["fit", "run", "optimum.toml", "--seed", "1", "--no-progress"]),
        "if2 fit run");
    let seg = segment(dir);
    let leaf = leaf(&seg);

    assert_eq!(chain_dirs(&leaf), zero_one(),
        "a two-chain leaf holds chain_0/ and chain_1/ (leaf {})", leaf.display());
    assert_eq!(chain_start_ids(&leaf), zero_one(),
        "chain_starts.tsv names the chains by the same numbers as their directories");

    let state: toml::Value =
        toml::from_str(&std::fs::read_to_string(leaf.join("fit_state.toml")).unwrap()).unwrap();
    let best = state["best_chain"].as_integer().unwrap() as usize;
    assert!(best < 2, "best_chain = {best} names a chain of a two-chain fit (0 or 1)");
    assert!(leaf.join(format!("chain_{best}")).is_dir(),
        "chain_{best}/ is the winning chain's directory");
    let eval_ids: Vec<usize> = state["chain_eval_ids"].as_array().unwrap()
        .iter().map(|v| v.as_integer().unwrap() as usize).collect();
    assert_eq!(eval_ids, vec![0, 1], "chain_eval_ids holds the chain numbers 0 and 1");
    assert_eq!(stderr_chain(&stderr, "best ll="), best,
        "the stderr winner line and fit_state.toml name the same chain");

    let seg_s = seg.to_string_lossy().into_owned();
    let text = String::from_utf8(camdl(dir, &["fit", "summary", &seg_s]).stdout).unwrap();
    assert_eq!(table_chain_ids(&text, "per-chain loglik-eval"), zero_one(),
        "fit summary's loglik-eval table names chains 0 and 1:\n{text}");

    let json: serde_json::Value = serde_json::from_slice(
        &camdl(dir, &["fit", "summary", &seg_s, "--format", "json"]).stdout).unwrap();
    let ids: Vec<u64> = json["methods"][0]["chains"].as_array().expect("chains[]")
        .iter().map(|c| c["chain_id"].as_u64().unwrap()).collect();
    assert_eq!(ids, vec![0, 1], "fit summary --format json chains[].chain_id:\n{json:#}");
}

/// PGAS: directories, the summary's per-chain table, `--exclude-chains`
/// (its parse, its validation message and its provenance record) and the
/// `chain` column `fit predict --by-chain` writes all use the numbers of
/// `draws.tsv`'s `chain` column, with no arithmetic between any two of them.
#[test]
fn a_pgas_leaf_numbers_its_chains_from_zero_on_every_surface() {
    require_tools();
    let tmp = tempfile::tempdir().unwrap();
    let dir = tmp.path();
    let toml = dir.join("posterior.toml");
    std::fs::write(&toml, format!("{}{}", problem(dir), pgas_method(8))).unwrap();

    let stderr = ok(&camdl(dir, &["fit", "run", "posterior.toml", "--seed", "1", "--no-progress"]),
        "pgas fit run");
    let seg = segment(dir);
    let leaf = leaf(&seg);
    let seg_s = seg.to_string_lossy().into_owned();

    assert_eq!(chain_dirs(&leaf), zero_one(), "chain_0/ and chain_1/ (leaf {})", leaf.display());
    assert_eq!(chain_start_ids(&leaf), zero_one());
    assert_eq!(draws_chains(&leaf), zero_one(), "draws.tsv's chain key is unchanged");
    let best = stderr_chain(&stderr, "best complete-data ll:");
    assert!(leaf.join(format!("chain_{best}")).is_dir(), "chain_{best}/ exists");

    // The per-chain table's header carries `mod-z` whatever the sampler's
    // loglik class (`obs ll` for PGAS, `mean loglik` for the marginal samplers).
    let text = String::from_utf8(camdl(dir, &["fit", "summary", &seg_s]).stdout).unwrap();
    assert_eq!(table_chain_ids(&text, "mod-z"), zero_one(),
        "the per-chain table names chains 0 and 1:\n{text}");

    // `--exclude-chains 1` drops the second chain: the one whose draws carry
    // `chain = 1` in draws.tsv, and whose trace is chain_1/.
    ok(&camdl(dir, &["fit", "predict", &seg_s, "--horizon", "free_forward",
        "--exclude-chains", "1"]), "predict --exclude-chains 1");
    let prov: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(predict_out(&seg).join("predictive-excl1.json")).unwrap()).unwrap();
    assert_eq!(prov["chain_selection"]["excluded"], serde_json::json!([1]));
    assert_eq!(prov["chain_selection"]["kept"], serde_json::json!([0]));
    assert_eq!(prov["chain_selection"]["n_total"], serde_json::json!(2));

    // A chain this fit does not have is refused, and the message states the
    // range in the same base.
    let bad = camdl(dir, &["fit", "predict", &seg_s, "--horizon", "free_forward",
        "--exclude-chains", "2"]);
    assert!(!bad.status.success(), "chain 2 is not a chain of a two-chain fit");
    let e = String::from_utf8_lossy(&bad.stderr);
    assert!(e.contains("chain 2 not in this fit (chains 0..1)"),
        "the refusal names the valid range 0..1:\n{e}");

    // `--by-chain`: the `chain` column carries 0, 1 and `all`.
    ok(&camdl(dir, &["fit", "predict", &seg_s, "--horizon", "free_forward", "--by-chain"]),
        "predict --by-chain");
    let pred = std::fs::read_to_string(predict_out(&seg).join("predictive").join("weekly_cases.tsv")).unwrap();
    let mut lines = pred.lines();
    let header: Vec<&str> = lines.next().unwrap().split('\t').collect();
    assert_eq!(header[0], "chain", "{header:?}");
    let cells: BTreeSet<&str> = lines.map(|l| l.split('\t').next().unwrap()).collect();
    assert_eq!(cells, ["0", "1", "all"].into_iter().collect(),
        "the per-chain bands are labelled 0 and 1 beside the pooled `all`");

    // `fit summary --exclude-chains 1` recomputes over chain 0 alone and says so.
    let sub = camdl(dir, &["fit", "summary", &seg_s, "--exclude-chains", "1"]);
    ok(&sub, "summary --exclude-chains 1");
    let sub_err = String::from_utf8_lossy(&sub.stderr);
    assert!(sub_err.contains("dropped chain(s) 1"), "the warning names chain 1:\n{sub_err}");
}

/// A leaf written by a camdl that numbered directories from 1 is refused by
/// name, not read with every chain shifted. Built by renaming the directories
/// of a fresh leaf: `chain_1/` → `chain_2/`, `chain_0/` → `chain_1/`, so the
/// leaf has the old shape exactly (its `draws.tsv` still keys chains 0 and 1).
#[test]
fn a_leaf_with_one_based_chain_directories_is_refused_not_misread() {
    require_tools();
    let tmp = tempfile::tempdir().unwrap();
    let dir = tmp.path();
    std::fs::write(dir.join("posterior.toml"), format!("{}{}", problem(dir), pgas_method(8)))
        .unwrap();
    ok(&camdl(dir, &["fit", "run", "posterior.toml", "--seed", "1", "--no-progress"]), "pgas run");
    let seg = segment(dir);
    let leaf = leaf(&seg);
    let seg_s = seg.to_string_lossy().into_owned();

    std::fs::rename(leaf.join("chain_1"), leaf.join("chain_2")).unwrap();
    std::fs::rename(leaf.join("chain_0"), leaf.join("chain_1")).unwrap();
    assert_eq!(chain_dirs(&leaf), [1usize, 2].into_iter().collect::<BTreeSet<_>>());

    let refused = |out: std::process::Output, what: &str| {
        let e = String::from_utf8_lossy(&out.stderr).to_string();
        assert!(!out.status.success(), "{what} must refuse the old layout, but succeeded:\n{e}");
        assert!(e.contains("counted chains from 1") && e.contains("gh#990"),
            "{what} must say why the leaf is refused and cite the change:\n{e}");
    };
    refused(camdl(dir, &["fit", "summary", &seg_s]), "fit summary");
    // The json/md/latex document is built by a second loop; it refuses too,
    // including under `--exclude-chains`, whose advisory would otherwise label
    // chains by directory number.
    refused(camdl(dir, &["fit", "summary", &seg_s, "--format", "json"]), "fit summary --format json");
    refused(camdl(dir, &["fit", "summary", &seg_s, "--format", "md"]), "fit summary --format md");
    refused(camdl(dir, &["fit", "summary", &seg_s, "--format", "json", "--exclude-chains", "1"]),
        "fit summary --format json --exclude-chains 1");
    refused(camdl(dir, &["fit", "predict", &seg_s, "--horizon", "free_forward"]), "fit predict");
    // `fit table --quantity` and `compare` fall back to a point estimate when a
    // fit has no posterior cloud; the refusal must not be taken for that.
    let fits = dir.join("results").join("fits").to_string_lossy().into_owned();
    refused(camdl(dir, &["fit", "table", &fits, "--quantity", "anything"]),
        "fit table --quantity");
    refused(camdl(dir, &["compare", &seg_s, &seg_s]), "compare");
    // `--draws 1` scores the plug-in point, the path with the winner fallback.
    refused(camdl(dir, &["compare", &seg_s, &seg_s, "--draws", "1"]), "compare --draws 1");

    // `--resume` addresses `chain_N/resume_state.bin` by name: refused too.
    let run_json: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(leaf.join("run.json")).unwrap()).unwrap();
    let run_id = run_json["run_id"].as_str().expect("run_id").to_string();
    std::fs::write(dir.join("longer.toml"), format!("{}{}", problem(dir), pgas_method(12)))
        .unwrap();
    let out = camdl(dir, &["fit", "run", "longer.toml", "--seed", "1", "--no-progress",
        "--resume", &run_id]);
    let e = String::from_utf8_lossy(&out.stderr).to_string();
    refused(out, "fit run --resume");
    // The message names the leaf the user resumed from, and the refusal comes
    // before any chain state is copied: no `resume_state.bin` exists outside
    // the base leaf.
    assert!(e.contains(&leaf.display().to_string()),
        "the refusal names the base leaf {}:\n{e}", leaf.display());
    let mut stray = Vec::new();
    let mut stack = vec![dir.join("results")];
    while let Some(d) = stack.pop() {
        for ent in std::fs::read_dir(&d).unwrap().flatten() {
            let p = ent.path();
            if p.is_dir() {
                stack.push(p);
            } else if p.file_name().is_some_and(|n| n == "resume_state.bin")
                && !p.starts_with(&leaf)
            {
                stray.push(p);
            }
        }
    }
    assert!(stray.is_empty(), "the refused resume copied chain state into a new leaf: {stray:?}");
}
