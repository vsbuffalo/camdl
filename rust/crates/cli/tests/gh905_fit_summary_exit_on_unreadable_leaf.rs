//! gh#905 — `camdl fit summary` must fail when the only run it was asked
//! about cannot be read back.
//!
//! A formatter that warned `cannot load <leaf>: …`, skipped the run and
//! rendered a summary with no method in it would tell a script that checks
//! the status code that the fit had been summarised. A stored file that no
//! longer matches the bytes the store recorded makes its run damaged, and the
//! run is refused by its folder and the file, as text and under
//! `--format json` alike.
//!
//! End to end because the defect is the process exit code and the emitted
//! document, neither of which a unit test over the formatter observes. The
//! mixed case — several leaves of which only some fail, which still exits 0
//! — is pinned as a unit test beside the rule it exercises
//! (`fit::fit_summary::tests`), because a single `fit run` does not produce
//! it.
//!
//! Deliberately cheap: the SEIR golden IR, one estimated parameter, 2 chains,
//! 6 sweeps, 40 particles, 5 observations. Nothing here depends on the fit
//! converging or being a good fit — only on the leaf existing and then being
//! unreadable.

mod fit_store_support;

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

/// A store with exactly one PGAS run, plus the paths a caller needs: the run's
/// folder (what the handle names) and its `pgas_summary.json` (what the test
/// corrupts).
struct Store {
    _tmp: tempfile::TempDir,
    results: PathBuf,
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
    let out = Command::new(&bin).env_remove("CLICOLOR_FORCE")
        .env("CAMDL_STORE", &results)
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

    let segment = fit_store_support::the_run(&results);
    let summary_json = find_one(&segment, "pgas_summary.json");
    assert_eq!(summary_json.parent(), Some(segment.as_path()), "the summary sits in the run's folder");

    Store { _tmp: tmp, results, segment, summary_json }
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
/// still JSON-shaped; its bytes no longer match the digest the store recorded
/// for it.
fn corrupt(summary_json: &Path) {
    std::fs::write(summary_json, "{ \"truncated\": ").unwrap();
}

fn summary(store: &Store, extra: &[&str]) -> std::process::Output {
    let segment = &store.segment;
    let mut args: Vec<String> =
        vec!["fit".into(), "summary".into(), segment.to_string_lossy().into_owned()];
    args.extend(extra.iter().map(|s| s.to_string()));
    Command::new(binary()).env_remove("CLICOLOR_FORCE")
        .env("CAMDL_STORE", &store.results)
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
    let before = summary(&store, &[]);
    assert!(
        before.status.success(),
        "fixture premise: an intact store must summarise.\nstderr:\n{}",
        String::from_utf8_lossy(&before.stderr)
    );

    corrupt(&store.summary_json);

    let after = summary(&store, &[]);
    let stderr = String::from_utf8_lossy(&after.stderr).into_owned();
    assert!(
        !after.status.success(),
        "the only leaf could not be read, so the summary is empty — \
         `fit summary` must not report success.\nstderr:\n{stderr}"
    );
    assert_damaged_refusal(&store, &stderr);
}

/// `--format json` fails the same way: the damaged run is refused before a
/// document is written, so no document reports a summary with no method in
/// it.
#[test]
fn the_json_form_fails_on_a_damaged_run_and_writes_no_document() {
    let store = build_store();
    corrupt(&store.summary_json);

    let out = summary(&store, &["--format", "json"]);
    let stdout = String::from_utf8_lossy(&out.stdout).into_owned();
    let stderr = String::from_utf8_lossy(&out.stderr).into_owned();
    assert!(
        !out.status.success(),
        "`--format json` must fail on an unreadable only leaf too.\nstdout:\n{stdout}\nstderr:\n{stderr}"
    );
    assert!(stdout.trim().is_empty(), "no document for a refused run:\n{stdout}");
    assert_damaged_refusal(&store, &stderr);
}

/// The refusal of the one run, damaged because its `pgas_summary.json` no
/// longer matches the bytes the store recorded: it names the run's folder and
/// the file.
fn assert_damaged_refusal(store: &Store, stderr: &str) {
    assert!(
        stderr.contains("names a fit run that did not complete and is not read")
            && stderr.contains("is damaged"),
        "the run is refused as damaged:\n{stderr}"
    );
    assert!(
        stderr.contains(&store.segment.to_string_lossy().into_owned()),
        "the refusal names the run's folder:\n{stderr}"
    );
    assert!(stderr.contains("pgas_summary.json"), "and the file that changed:\n{stderr}");
}
