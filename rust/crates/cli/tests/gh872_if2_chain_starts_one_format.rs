//! gh#872: an IF2 stage writes `chain_starts.tsv` once, in the format every
//! other stage kind writes.
//!
//! Two functions used to write that file into the same stage directory: the
//! provenance-carrying writer the PGAS and PMMH stages also use, and then a
//! second, IF2-only one that overwrote it. The survivor had no `source`
//! column, so an IF2 stage's `chain_starts.tsv` could not say whether the
//! chains were independently initialised — the audit question the file exists
//! to answer (gh#871) — and it numbered chains from 1 while the sampler stages
//! numbered them from 0, so one parser could not read both and an off-by-one
//! was available to anyone who tried.
//!
//! The pin is end-to-end and comparative: run one fit with an IF2 stage and a
//! PGAS stage and require the two files to have the same columns and the same
//! chain numbering. Asserting the IF2 column names literally would restate the
//! writer; asserting them against another stage kind states the invariant that
//! was broken, and would catch a future divergence from either side.
//!
//! The per-chain source labels are checked against the values they describe,
//! so the format assertion cannot pass over a file whose provenance is wrong:
//! `uniform_unconstrained` draws each chain its own point, and the two betas
//! must therefore differ.
//!
//! Cheap on purpose — 2 chains, 20 particles, 1 IF2 iteration, 4 PGAS sweeps.
//! Starting points are fixed before either sampler updates anything, so
//! nothing here depends on either stage converging.

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

/// The `chain_starts.tsv` written by the method whose CAS leaf sits under a
/// `<method>-<h8>` directory.
fn chain_starts_for_stage(root: &Path, method: &str) -> PathBuf {
    let marker = format!("{method}-");
    let mut found: Vec<PathBuf> = Vec::new();
    let mut stack = vec![root.to_path_buf()];
    while let Some(dir) = stack.pop() {
        let Ok(entries) = std::fs::read_dir(&dir) else { continue };
        for e in entries.flatten() {
            let p = e.path();
            if p.is_dir() {
                stack.push(p);
            } else if p.file_name().is_some_and(|n| n == "chain_starts.tsv")
                && p.components().any(|c| {
                    c.as_os_str().to_string_lossy().starts_with(&marker)
                })
            {
                found.push(p);
            }
        }
    }
    assert_eq!(found.len(), 1,
        "expected exactly one chain_starts.tsv under a `{method}` method \
         leaf below {}, found {found:?}", root.display());
    found.pop().unwrap()
}

/// Parse `chain_starts.tsv` into (comment lines, column names, data rows).
fn parse(path: &Path) -> (Vec<String>, Vec<String>, Vec<Vec<String>>) {
    let txt = std::fs::read_to_string(path)
        .unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
    let comments: Vec<String> = txt.lines()
        .filter(|l| l.starts_with('#')).map(|s| s.to_string()).collect();
    let mut body = txt.lines()
        .filter(|l| !l.starts_with('#') && !l.trim().is_empty());
    let cols: Vec<String> = body.next().expect("header row")
        .split('\t').map(|s| s.to_string()).collect();
    let rows: Vec<Vec<String>> = body
        .map(|l| l.split('\t').map(|s| s.to_string()).collect())
        .collect();
    (comments, cols, rows)
}

/// The `col` field of every row, in file order.
fn column<'a>(cols: &[String], rows: &'a [Vec<String>], col: &str) -> Vec<&'a str> {
    let i = cols.iter().position(|c| c == col)
        .unwrap_or_else(|| panic!("no `{col}` column in {cols:?}"));
    rows.iter().map(|r| r[i].as_str()).collect()
}

#[test]
fn if2_chain_starts_matches_the_format_every_other_stage_writes() {
    let bin = binary();
    assert!(bin.exists(),
        "release camdl binary missing: {} — run `make build-rust` or `make test`",
        bin.display());
    let camdlc = camdlc_bin();
    assert!(camdlc.exists(),
        "camdlc.exe missing: {} — run `make build-ocaml`", camdlc.display());

    let tmp = tempfile::tempdir().unwrap();
    let dir = tmp.path();

    let data = dir.join("obs.tsv");
    std::fs::write(&data,
        "time\tweekly_cases\n7\t1\n14\t2\n21\t3\n28\t4\n35\t5\n").unwrap();

    // Both methods draw their own starts with the same rule, so the two files
    // are directly comparable: neither is chained off the other, and a
    // difference between them can only come from the writers. PGAS needs a
    // proper prior on every estimated parameter, hence the log-normal. One
    // problem half, two `[method]` files.
    let problem = format!(r#"
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
"#, ir = golden_ir().display(), data = data.display());
    let if2_toml = dir.join("optimum.toml");
    std::fs::write(&if2_toml, format!("{problem}
[method]
algorithm  = \"if2\"
backend    = \"chain_binomial\"
chains     = 2
particles  = 20
iterations = 1
cooling    = 0.5
starts     = \"uniform_unconstrained\"
")).unwrap();
    let pgas_toml = dir.join("posterior.toml");
    std::fs::write(&pgas_toml, format!("{problem}
[method]
algorithm = \"pgas\"
backend   = \"chain_binomial\"
chains    = 2
particles = 20
sweeps    = 4
burn_in   = 1
starts    = \"uniform_unconstrained\"
")).unwrap();

    for toml in [&if2_toml, &pgas_toml] {
        let out = Command::new(&bin)
            .env("CAMDL_OUTPUT_DIR", dir.join("results"))
            .env("CAMDL_SKIP_VERSION_CHECK", "1")
            .env("CAMDLC", &camdlc)
            .args(["fit", "run", &toml.to_string_lossy(), "--seed", "1", "--no-progress"])
            .output()
            .expect("spawn fit run");
        let stderr = String::from_utf8_lossy(&out.stderr);
        assert!(out.status.success(), "fit run {} failed:\nstderr={stderr}", toml.display());
    }

    let results = dir.join("results");
    let (if2_comments, if2_cols, if2_rows) =
        parse(&chain_starts_for_stage(&results, "if2"));
    let (_, pgas_cols, pgas_rows) =
        parse(&chain_starts_for_stage(&results, "pgas"));
    // The rows the chains ran from. A start the filter refused is on the
    // record ahead of its redraw as `rejected` (gh#887) — under twenty
    // particles that happens — and is not a chain's start.
    let accepted = |cols: &[String], rows: &[Vec<String>]| -> Vec<Vec<String>> {
        let i = cols.iter().position(|c| c == "status")
            .unwrap_or_else(|| panic!("no `status` column in {cols:?}"));
        rows.iter().filter(|r| r[i] == "accepted").cloned().collect()
    };
    let if2_rows = accepted(&if2_cols, &if2_rows);
    let pgas_rows = accepted(&pgas_cols, &pgas_rows);

    // The audit question: can a reader tell from this file alone whether the
    // chains were started apart?
    assert!(if2_cols.iter().any(|c| c == "source"),
        "the IF2 stage's chain_starts.tsv has columns {if2_cols:?} and no \
         `source`. That column is how the file answers whether the chains were \
         independently initialised, which is the whole reason it exists — an R̂ \
         computed over chains that all began at one point says nothing, and \
         this is the artifact that would have shown it.");

    // One shape across stage kinds, so one parser reads both.
    assert_eq!(if2_cols, pgas_cols,
        "the IF2 stage writes chain_starts.tsv columns {if2_cols:?} while the \
         PGAS stage in the same fit writes {pgas_cols:?}. A consumer cannot \
         read both with one parser.");

    let if2_ids = column(&if2_cols, &if2_rows, "chain_id");
    let pgas_ids = column(&pgas_cols, &pgas_rows, "chain_id");
    assert_eq!(if2_ids, ["0", "1"],
        "chain ids in the IF2 stage's chain_starts.tsv are {if2_ids:?}; the \
         column is 0-based on every stage that writes this file, and one that \
         numbers from 1 hands an off-by-one to anyone joining two stages.");
    assert_eq!(if2_ids, pgas_ids,
        "the IF2 stage numbers its chains {if2_ids:?} and the PGAS stage \
         numbers them {pgas_ids:?}.");

    // Provenance reaching disk, checked against the values it describes:
    // `uniform_unconstrained` draws each chain its own point, so each row
    // names its chain and no two rows may share a value.
    let if2_sources = column(&if2_cols, &if2_rows, "source");
    assert_eq!(if2_sources,
        ["uniform_unconstrained:chain-0", "uniform_unconstrained:chain-1"],
        "the IF2 stage's per-chain sources are {if2_sources:?}; the stage drew \
         each chain its own point under `starts = \"uniform_unconstrained\"`, and \
         the file has to say so.");

    let betas: Vec<f64> = column(&if2_cols, &if2_rows, "beta").iter()
        .map(|v| v.parse().unwrap_or_else(|_| panic!("non-numeric beta {v:?}")))
        .collect();
    assert!((betas[0] - betas[1]).abs() > 1e-12,
        "both IF2 chains started at beta={:?} under `uniform_unconstrained`, \
         which is supposed to draw each chain its own point. The per-chain \
         `source` labels asserted above would then be a claim the values \
         contradict.", betas);

    let header = if2_comments.iter().find(|c| c.contains("chain_starts"))
        .unwrap_or_else(|| panic!(
            "no `# camdl chain_starts` header in the IF2 stage's file: \
             {if2_comments:?}. The header names the init method, so a reader \
             who skims it gets the same answer as the rows."));
    assert!(header.contains("starts=uniform_unconstrained") && header.contains("kind=spread"),
        "the IF2 stage's header is {header:?} and does not name the rule that \
         supplied the starts, or its kind.");

    // The control for the point-start demotion (proposal 2026-09-08 §3.4):
    // both leaves record a spread start, and the PGAS leaf's R̂ is assessed.
    for method in ["if2", "pgas"] {
        let leaf = chain_starts_for_stage(&results, method);
        let state = std::fs::read_to_string(leaf.parent().unwrap().join("fit_state.toml")).unwrap();
        assert!(state.contains("chain_starts_kind = \"spread\""), "{method}: {state}");
    }
    let pgas_leaf = chain_starts_for_stage(&results, "pgas");
    let segment = pgas_leaf.parent().unwrap().parent().unwrap().parent().unwrap();
    let out = Command::new(&bin)
        .env("CAMDL_SKIP_VERSION_CHECK", "1")
        .args(["fit", "summary", &segment.to_string_lossy()])
        .output()
        .expect("spawn fit summary");
    let summary = String::from_utf8_lossy(&out.stdout).into_owned();
    assert!(out.status.success(), "fit summary failed:\n{}", String::from_utf8_lossy(&out.stderr));
    assert!(summary.contains("seeded from:  uniform_unconstrained (one independent draw per chain)"),
        "{summary}");
    // Whatever this tiny fit's R̂ turns out to be (a chain of a 4-sweep run
    // can be refused at its start), it is never withheld for the reason a
    // point start is: the chains began apart.
    assert!(!summary.contains("started at one point"),
        "a spread start's R̂ is not demoted for its starts:\n{summary}");
}
