//! What a PMMH leaf whose every start is refused at chain init records about
//! itself, as values a reader keys on.
//!
//! The fixture is a PMMH fit whose every start is refused at chain init (see
//! [`FIT`]): the filter degenerates at every start, each chain is refused
//! with a `bad_init` finding, and the method fails with every chain refused.

use std::path::{Path, PathBuf};
use std::process::Command;

fn manifest() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

fn camdl() -> PathBuf {
    let camdl = manifest().join("../../target/release/camdl");
    assert!(camdl.exists(), "release camdl binary missing: {} — run `make build-rust`", camdl.display());
    camdl
}

fn camdlc() -> PathBuf {
    let camdlc = manifest().join("../../../ocaml/_build/default/bin/camdlc.exe");
    assert!(camdlc.exists(), "camdlc missing: {} — run `make build-ocaml`", camdlc.display());
    camdlc
}

/// The fit, over `tests/fixtures/mre`'s two-patch SIR and its weekly cases.
/// The bounds hold transmission far below recovery, so the epidemic dies out
/// within a week and a later week's positive count has a projected mean of
/// zero: no start can be scored.
const FIT: &str = r#"
[model]
camdl = "model/sir_patches.camdl"
[data]
file = "cases.tsv"
[estimate.beta]
bounds = [0.0011, 0.0012]
start  = 0.00115
prior  = { flat = {} }
[estimate.gamma]
bounds = [0.9, 1.0]
start  = 0.95
prior  = { flat = {} }
[fixed]
rho = 0.6
k   = 20
I0  = 1
[method]
algorithm  = "pmmh"
backend    = "chain_binomial"
chains     = 2
particles  = 50
iterations = 20
burn_in    = 10
starts     = "from_prior"
"#;

/// A camdl command in `dir`, against the compiler under test.
fn camdl_in(dir: &Path) -> Command {
    let mut c = Command::new(camdl());
    c.env_remove("CLICOLOR_FORCE")
        .env("CAMDL_SKIP_VERSION_CHECK", "1")
        .env("CAMDLC", camdlc())
        .env("CAMDL_IR_CACHE_DIR", dir.join(".ir-cache"))
        .current_dir(dir);
    c
}

/// Run the failing fit; return its stderr, the one leaf it claimed, and the
/// directory holding both.
fn failed_run() -> (String, PathBuf, tempfile::TempDir) {
    let tmp = tempfile::tempdir().unwrap();
    let dir = tmp.path();
    let fixture = manifest().join("../../../tests/fixtures/mre");
    std::fs::create_dir(dir.join("model")).unwrap();
    for f in ["model/sir_patches.camdl", "model/pop.tsv", "cases.tsv"] {
        std::fs::copy(fixture.join(f), dir.join(f)).unwrap();
    }
    std::fs::write(dir.join("fit.toml"), FIT).unwrap();
    let out = camdl_in(dir)
        .args(["fit", "run", "fit.toml", "--seed", "1", "--progress", "none"])
        .output()
        .unwrap();
    let stderr = String::from_utf8_lossy(&out.stderr).into_owned();
    assert_eq!(out.status.code(), Some(1), "the run fails with status 1:\n{stderr}");
    assert!(stderr.contains("all 2 chains failed init-eval"),
        "the run fails at chain init, every chain refused:\n{stderr}");
    let subdirs = |d: &Path| -> Vec<PathBuf> {
        std::fs::read_dir(d).unwrap().flatten().map(|e| e.path()).filter(|p| p.is_dir()).collect()
    };
    let leaves: Vec<PathBuf> = subdirs(&dir.join("results/fits")).iter()
        .flat_map(|fit| subdirs(fit))
        .flat_map(|m| subdirs(&m))
        .filter(|p| p.file_name().unwrap().to_string_lossy().starts_with("seed_"))
        .collect();
    assert_eq!(leaves.len(), 1, "one claimed leaf: {leaves:?}");
    (stderr, leaves[0].clone(), tmp)
}

fn json(path: &Path) -> serde_json::Value {
    let text = std::fs::read_to_string(path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    serde_json::from_str(&text).unwrap_or_else(|e| panic!("{}: {e}", path.display()))
}

fn events(leaf: &Path) -> Vec<serde_json::Value> {
    std::fs::read_to_string(leaf.join("events.jsonl")).unwrap()
        .lines().map(|l| serde_json::from_str(l).unwrap()).collect()
}

/// The band the run's acceptance diagnostics apply is published as values on
/// the event log's `begin` line, so a reader has it although no acceptance
/// finding fired — none can, on a run whose chains never started. It was
/// only ever in a finding's message ("outside healthy range [15%, 50%]").
#[test]
fn the_begin_event_publishes_the_acceptance_band() {
    let (_stderr, leaf, _tmp) = failed_run();
    let begin = events(&leaf).into_iter().next().unwrap();
    assert_eq!(begin["event"], "begin");
    assert_eq!(begin["schema"], "camdl.events/v1", "an added field is not a new version");
    assert_eq!(begin["acceptance"],
        serde_json::json!({"kernel": "random_walk", "band_lo": 0.15, "band_hi": 0.5}),
        "PMMH's θ-move is random-walk MH: {begin}");
}

/// A refused chain is named by one number everywhere: a `bad_init` finding's
/// sentence ("Chain N starting parameters …"), its `chain_id`, the stderr
/// refusal line, `chain_starts.tsv`'s `chain_id` column and the `chain_N/`
/// directory all carry the runner's own index, counted from 0 (gh#990,
/// `chain_index.rs`), in `diagnostics.json` and in the event log alike.
#[test]
fn a_refused_chains_number_agrees_across_its_records() {
    let (stderr, leaf, _tmp) = failed_run();
    let findings: Vec<serde_json::Value> =
        serde_json::from_value(json(&leaf.join("diagnostics.json"))).unwrap();
    let logged: Vec<serde_json::Value> = events(&leaf).into_iter()
        .filter(|e| e["event"] == "finding").map(|e| e["finding"].clone()).collect();
    assert_eq!(findings, logged, "the log records the findings diagnostics.json holds");
    let starts = std::fs::read_to_string(leaf.join("chain_starts.tsv")).unwrap();
    let mut header: Option<Vec<&str>> = None;
    let mut start_ids = std::collections::BTreeSet::new();
    for line in starts.lines().filter(|l| !l.starts_with('#')) {
        let cells: Vec<&str> = line.split('\t').collect();
        match &header {
            None => header = Some(cells),
            Some(h) => {
                let col = h.iter().position(|c| *c == "chain_id").expect("a chain_id column");
                start_ids.insert(cells[col].parse::<u64>().unwrap());
            }
        }
    }

    let mut ids = Vec::new();
    for f in findings.iter().filter(|f| f["kind"]["type"] == "bad_init") {
        let id = f["kind"]["chain_id"].as_u64().expect("an integer chain_id");
        let message = f["message"].as_str().unwrap();
        assert!(message.starts_with(&format!("Chain {id} starting parameters")),
            "the sentence names the chain the record names: chain_id {id}, {message:?}");
        assert!(leaf.join(format!("chain_{id}")).is_dir(), "chain_{id}/ is that chain's directory");
        assert!(stderr.contains(&format!("chain {id}: ✗ BadInit")), "and stderr names it so:\n{stderr}");
        assert!(start_ids.contains(&id), "and chain_starts.tsv: {start_ids:?}");
        ids.push(id);
    }
    ids.sort();
    assert_eq!(ids, [0, 1], "both chains are refused, numbered from 0");
}
