//! gh#905 — `camdl fit summary` exited 0 when the only method leaf it was
//! asked about could not be read back.
//!
//! The text formatter printed `warning: cannot load <leaf>: …`, skipped the
//! leaf, rendered a summary with no stage in it, and returned success. A
//! script that asks "summarise this fit" and checks the status code was told
//! the fit had been summarised. `--format json` was worse: the load error was
//! swallowed entirely (`.ok()`), so the reader got `"methods": []` with nothing
//! on stderr saying why.
//!
//! End to end because the defect is the process exit code and the emitted
//! document, neither of which a unit test over the formatter observes. The
//! mixed case — a segment holding several leaves of which only some fail,
//! which still exits 0 — is pinned as a unit test beside the rule it
//! exercises (`fit::fit_summary::tests`), because a second readable leaf in
//! one segment is not something a single `fit run` produces.
//!
//! Deliberately cheap: the SEIR golden IR, one estimated parameter, 2 chains,
//! 6 sweeps, 40 particles, 5 observations. Nothing here depends on the fit
//! converging or being a good fit — only on the leaf existing and then being
//! unreadable.

use std::path::{Path, PathBuf};
use std::process::Command;

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

/// A store with exactly one PGAS leaf, plus the paths a caller needs: the
/// segment directory (`results/fits/<stem>-<h8>`, what the handle names) and
/// the leaf's `pgas_summary.json` (what the test corrupts).
struct Store {
    _tmp: tempfile::TempDir,
    segment: PathBuf,
    summary_json: PathBuf,
}

fn build_store() -> Store {
    let bin = binary();
    assert!(
        bin.exists(),
        "release camdl binary missing: {} — run `make build-rust` or `make test`",
        bin.display()
    );
    let camdlc = camdlc_bin();
    assert!(
        camdlc.exists(),
        "camdlc.exe missing: {} — run `make build-ocaml`",
        camdlc.display()
    );

    let tmp = tempfile::tempdir().unwrap();
    let dir = tmp.path();

    let data = dir.join("obs.tsv");
    std::fs::write(&data, "time\tweekly_cases\n7\t1\n14\t2\n21\t3\n28\t4\n35\t5\n").unwrap();

    let fit_toml = dir.join("onefit.toml");
    std::fs::write(
        &fit_toml,
        format!(
            r#"
[model]
camdl = "{ir}"

[data.observations]
weekly_cases = "{data}"

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
backend   = "chain_binomial"
chains    = 2
particles = 40
sweeps    = 6
burn_in   = 2
thin      = 1

[config]
dt = 1.0
"#,
            ir = golden_ir().display(),
            data = data.display()
        ),
    )
    .unwrap();

    let results = dir.join("results");
    let out = Command::new(&bin)
        .env("CAMDL_OUTPUT_DIR", &results)
        .env("CAMDL_SKIP_VERSION_CHECK", "1")
        .env("CAMDLC", &camdlc)
        .args(["fit", "run", &fit_toml.to_string_lossy(), "--seed", "1"])
        .output()
        .expect("spawn fit run");
    assert!(
        out.status.success(),
        "fixture premise: the fit must complete.\nstderr:\n{}",
        String::from_utf8_lossy(&out.stderr)
    );

    // `results/fits/<stem>-<h8>/<algorithm>-<h8>/seed_N-<h8>/pgas_summary.json`
    let summary_json = find_one(&results.join("fits"), "pgas_summary.json");
    // The segment is three levels up: leaf → algorithm → segment.
    let segment = summary_json
        .parent()
        .and_then(Path::parent)
        .and_then(Path::parent)
        .expect("leaf sits three levels under the segment")
        .to_path_buf();

    Store { _tmp: tmp, segment, summary_json }
}

/// The single file named `name` anywhere under `root`. Asserts uniqueness, so
/// a store that grew a second leaf fails here rather than silently picking one.
fn find_one(root: &Path, name: &str) -> PathBuf {
    let mut hits: Vec<PathBuf> = Vec::new();
    let mut stack = vec![root.to_path_buf()];
    while let Some(d) = stack.pop() {
        for entry in std::fs::read_dir(&d).unwrap() {
            let p = entry.unwrap().path();
            if p.is_dir() {
                stack.push(p);
            } else if p.file_name().and_then(|f| f.to_str()) == Some(name) {
                hits.push(p);
            }
        }
    }
    assert_eq!(hits.len(), 1, "expected exactly one {name} under {}: {hits:?}", root.display());
    hits.pop().unwrap()
}

/// Truncate the leaf's typed payload mid-value. The file still exists and is
/// still JSON-shaped, so the failure is a parse error inside the loader rather
/// than a missing-file branch somewhere earlier.
fn corrupt(summary_json: &Path) {
    std::fs::write(summary_json, "{ \"truncated\": ").unwrap();
}

fn summary(segment: &Path, extra: &[&str]) -> std::process::Output {
    let mut args: Vec<String> =
        vec!["fit".into(), "summary".into(), segment.to_string_lossy().into_owned()];
    args.extend(extra.iter().map(|s| s.to_string()));
    Command::new(binary())
        .env("CAMDL_SKIP_VERSION_CHECK", "1")
        .env("CAMDLC", camdlc_bin())
        .args(&args)
        .output()
        .expect("spawn fit summary")
}

/// The whole finding: the one leaf the handle names cannot be read, so there
/// is no summary, and the status code has to say so.
#[test]
fn an_unreadable_only_leaf_fails_the_summary() {
    let store = build_store();

    // Premise: the store summarises cleanly before it is damaged.
    let before = summary(&store.segment, &[]);
    assert!(
        before.status.success(),
        "fixture premise: an intact store must summarise.\nstderr:\n{}",
        String::from_utf8_lossy(&before.stderr)
    );

    corrupt(&store.summary_json);

    let after = summary(&store.segment, &[]);
    let stderr = String::from_utf8_lossy(&after.stderr).into_owned();
    assert!(
        !after.status.success(),
        "the only leaf could not be read, so the summary is empty — \
         `fit summary` must not report success.\nstderr:\n{stderr}"
    );
    assert!(
        stderr.contains(&store.summary_json.parent().unwrap().to_string_lossy().into_owned()),
        "the failure must name the leaf that could not be read:\n{stderr}"
    );
    assert!(
        stderr.contains("pgas_summary.json"),
        "the failure must carry the loader's reason:\n{stderr}"
    );
}

/// `--format json` has the same duty, and one more: a reader parsing the
/// document must find the cause in it, not only on a stderr stream it may not
/// have captured.
#[test]
fn the_json_document_carries_the_failure_it_exits_on() {
    let store = build_store();
    corrupt(&store.summary_json);

    let out = summary(&store.segment, &["--format", "json"]);
    let stdout = String::from_utf8_lossy(&out.stdout).into_owned();
    assert!(
        !out.status.success(),
        "`--format json` must fail on an unreadable only leaf too.\nstdout:\n{stdout}"
    );

    let doc: serde_json::Value =
        serde_json::from_str(&stdout).expect("the document must still be valid JSON");
    assert_eq!(
        doc["methods"].as_array().map(Vec::len),
        Some(0),
        "premise: the unreadable leaf contributes no method"
    );
    let failures = doc["failures"].as_array().expect("the document must carry a `failures` list");
    assert_eq!(failures.len(), 1, "one leaf failed, so one entry: {failures:?}");
    let leaf = failures[0]["leaf"].as_str().expect("each failure names its leaf");
    assert_eq!(
        Path::new(leaf),
        store.summary_json.parent().unwrap(),
        "the `leaf` field is the method-leaf directory"
    );
    let reason = failures[0]["reason"].as_str().expect("each failure carries a reason");
    assert!(
        reason.contains("pgas_summary.json"),
        "the reason must name the file that could not be read: {reason}"
    );
}
