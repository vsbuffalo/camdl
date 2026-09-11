//! gh#904 — a chain refused on its only attempt was recorded `accepted`.
//!
//! `chain_starts.tsv` is written before the sampler runs, when every row can
//! only say `accepted`. It was rewritten at the end of the run from the final
//! `DrawnStarts` only when a retry had rejected something, and a refusal only
//! reached that record when a retry had happened (`if attempt > 0`). Under a
//! point rule — `single`, `from_mle`, `from_params` — there is no redraw, so
//! neither condition held: the file kept its pre-run row, and the one artifact
//! that is supposed to say which starts the chains ran from said `accepted`
//! for a chain that never ran.
//!
//! The reading is silently wrong rather than merely absent, which is why it is
//! worth an end-to-end test: the stderr refusal, `diagnostics.json`'s
//! `bad_init` entry and the sampler's own error all say the chain was refused,
//! and the file a reader is pointed at ("Inspect chain_starts.tsv to see which
//! init was used") contradicted them.
//!
//! The fixture makes the refusal deterministic rather than probabilistic: the
//! SEIR golden IR at `beta = 0.001` in a population of 1000 with one infective
//! produces no infections over a week, so the negative-binomial mean
//! `rho * projected` is exactly 0 while the data say 5000 — a value outside
//! the family's support, so every particle scores `-inf` and the start is
//! refused on its first and only attempt.

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

/// One data row of `chain_starts.tsv`, split on tabs. Comment lines (`#`) and
/// the column header are dropped.
struct StartRow {
    chain_id: String,
    attempt: String,
    status: String,
    source: String,
}

/// A completed refused run: the `chain_starts.tsv` rows, the method leaf, and
/// the temp dir that must outlive both.
struct RefusedRun {
    _tmp: tempfile::TempDir,
    rows: Vec<StartRow>,
    leaf: PathBuf,
}

/// Run a one-chain fit whose only start the sampler refuses.
///
/// `method_body` is spliced under `[method]`, so the two callers vary the
/// algorithm and nothing else.
fn refused_run(stem: &str, method_body: &str) -> RefusedRun {
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

    // Counts the model cannot produce at the declared start: the projected
    // mean is 0, and 5000 is outside a negative binomial's support there.
    let data = dir.join("obs.tsv");
    std::fs::write(
        &data,
        "time\tweekly_cases\n7\t5000\n14\t5200\n21\t5100\n28\t5300\n35\t5400\n",
    )
    .unwrap();

    let fit_toml = dir.join(format!("{stem}.toml"));
    std::fs::write(
        &fit_toml,
        format!(
            r#"
[model]
camdl = "{ir}"

[data.observations]
weekly_cases = "{data}"

[estimate.beta]
bounds = [0.001, 0.002]
start  = 0.001
prior  = {{ log_normal = {{ mu = -7.0, sigma = 0.5 }} }}

[fixed]
sigma    = 0.25
gamma    = 0.3
rho      = 0.5
k        = 10.0
p_detect = 0.5
N0       = 1000
I0       = 1

[method]
backend   = "chain_binomial"
chains    = 1
particles = 20
burn_in   = 1
thin      = 1
# A point rule: every chain starts at the declared value and there is no
# redraw, which is the case gh#904 records wrongly.
starts    = "single"
{method_body}

[config]
dt = 1.0
"#,
            ir = golden_ir().display(),
            data = data.display(),
            method_body = method_body
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
    let stderr = String::from_utf8_lossy(&out.stderr).into_owned();
    assert!(
        !out.status.success() && stderr.contains("BadInit"),
        "fixture premise: the only chain must be refused at its start.\nstderr:\n{stderr}"
    );

    let tsv = find_one(&results.join("fits"), "chain_starts.tsv");
    let leaf = tsv.parent().unwrap().to_path_buf();
    let text = std::fs::read_to_string(&tsv).unwrap();
    let rows: Vec<StartRow> = text
        .lines()
        .filter(|l| !l.starts_with('#') && !l.starts_with("chain_id\t"))
        .filter(|l| !l.trim().is_empty())
        .map(|l| {
            let f: Vec<&str> = l.split('\t').collect();
            StartRow {
                chain_id: f[0].to_string(),
                attempt: f[1].to_string(),
                status: f[2].to_string(),
                source: f[3].to_string(),
            }
        })
        .collect();

    RefusedRun { _tmp: tmp, rows, leaf }
}

/// The single file named `name` anywhere under `root`.
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

/// Every `chain_N/trace.tsv` under `leaf`, with its data-row count (the header
/// line does not count). A refused chain writes no sweep.
fn trace_data_rows(leaf: &Path) -> Vec<(PathBuf, usize)> {
    let mut out = Vec::new();
    let mut stack = vec![leaf.to_path_buf()];
    while let Some(d) = stack.pop() {
        for entry in std::fs::read_dir(&d).unwrap() {
            let p = entry.unwrap().path();
            if p.is_dir() {
                stack.push(p);
            } else if p.file_name().and_then(|f| f.to_str()) == Some("trace.tsv") {
                let text = std::fs::read_to_string(&p).unwrap();
                let n = text.lines().filter(|l| !l.trim().is_empty()).count();
                out.push((p, n.saturating_sub(1)));
            }
        }
    }
    out
}

/// The finding, asserted on the file a reader is told to inspect: the row must
/// say the chain did not run, and nothing in the leaf may claim it sampled.
fn assert_refused_and_never_ran(rows: &[StartRow], leaf: &Path, rule: &str) {
    assert_eq!(rows.len(), 1, "one chain, no redraw under a point rule — one row");
    let r = &rows[0];
    assert_eq!(r.chain_id, "1", "the column is 1-based (gh#781)");
    assert_eq!(r.attempt, "0", "a point rule redraws nothing, so the only attempt is 0");
    assert_eq!(r.source, rule, "the source is the rule that produced the point");
    assert_eq!(
        r.status, "refused",
        "the chain was refused at its only start and never ran; \
         `accepted` here contradicts the stderr refusal, the `bad_init` \
         diagnostic and the sampler's own error"
    );

    for (path, n) in trace_data_rows(leaf) {
        assert_eq!(
            n, 0,
            "the chain never ran, so {} must carry no sweep rows",
            path.display()
        );
    }
}

#[test]
fn pgas_records_a_refused_point_start_as_refused() {
    let run = refused_run("pgasfit", "algorithm = \"pgas\"\nsweeps    = 4\n");
    assert_refused_and_never_ran(&run.rows, &run.leaf, "single");
}

#[test]
fn pmmh_records_a_refused_point_start_as_refused() {
    let run = refused_run("pmmhfit", "algorithm = \"pmmh\"\niterations = 4\n");
    assert_refused_and_never_ran(&run.rows, &run.leaf, "single");
}
