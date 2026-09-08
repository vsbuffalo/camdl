//! gh#871: `chain_starts.tsv` must name the source that actually supplied the
//! values, not the `init` the stage happened to declare.
//!
//! A stage chained off another with `init_mle = "<upstream stage>"` starts
//! every chain at the upstream point estimate; the declared `init` never runs.
//! The writer was handed that declared `init` regardless, so the file recorded
//! N independent per-chain draws (`uniform_unconstrained:chain-0` …) beside N
//! identical values. `chain_starts.tsv` is the artifact an auditor reads to
//! answer "were these chains started apart?" — the question behind any R̂ that
//! looks too good — so a wrong answer there is durable and looks authoritative.
//!
//! The defect was at a call site, as in gh#506 and gh#513:
//! `write_chain_starts_tsv` was correct given its arguments, and the stage
//! dispatcher passed the wrong one. So the pin is end-to-end — run a real
//! two-stage fit and read the file the fit itself wrote.
//!
//! The premise is asserted alongside the claim, because either alone can pass
//! vacuously: the recorded source must not name the declared `init` and must
//! carry no `:chain-<id>` suffix (a per-chain claim over one shared point),
//! and the recorded values must really be one point repeated, so that the
//! source assertion is about this defect rather than an unrelated run.
//!
//! Cheap on purpose — 2 chains, 20 particles, 1 IF2 iteration, 4 PGAS sweeps.
//! Starting points are fixed before any sweep runs, so nothing here depends on
//! either stage converging.

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

/// The `chain_starts.tsv` written by the stage whose CAS leaf sits under a
/// `NN-<stage_name>-<h8>` directory. The two-stage fit writes one per stage;
/// only the chained one exercises gh#871.
fn chain_starts_for_stage(root: &Path, stage_name: &str) -> PathBuf {
    let marker = format!("-{stage_name}-");
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
                    c.as_os_str().to_string_lossy().contains(&marker)
                })
            {
                found.push(p);
            }
        }
    }
    assert_eq!(found.len(), 1,
        "expected exactly one chain_starts.tsv under a `{stage_name}` stage \
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

#[test]
fn chained_stage_records_the_point_it_started_from_not_the_declared_init() {
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

    // `prime` draws its own two chains apart; `posterior` is chained off it
    // with `init_mle`, so both its chains take `prime`'s single point estimate
    // and the `init` it would otherwise have used never runs. PGAS needs a
    // proper prior on every estimated parameter, hence the log-normal on beta.
    let fit_toml = dir.join("fit.toml");
    std::fs::write(&fit_toml, format!(r#"
[model]
camdl = "{ir}"

[data.observations]
weekly_cases = "{data}"

[estimate.beta]
bounds = [0.01, 0.5]
start  = 0.123
prior  = {{ log_normal = {{ mu = -1.2, sigma = 0.4 }} }}

[fixed]
sigma    = 0.25
gamma    = 0.3
rho      = 0.5
k        = 10.0
p_detect = 0.5
N0       = 1000
I0       = 1

[stages.prime]
algorithm  = "if2"
backend    = "chain_binomial"
chains     = 2
particles  = 20
iterations = 1
cooling    = 0.5
init       = "uniform_unconstrained"

[stages.posterior]
algorithm = "pgas"
backend   = "chain_binomial"
chains    = 2
particles = 20
sweeps    = 4
burn_in   = 1
init      = "uniform_unconstrained"
init_mle  = "prime"

[config]
dt = 1.0
"#, ir = golden_ir().display(), data = data.display())).unwrap();

    let out = Command::new(&bin)
        .env("CAMDL_OUTPUT_DIR", dir.join("results"))
        .env("CAMDL_SKIP_VERSION_CHECK", "1")
        .env("CAMDLC", &camdlc)
        .args(["fit", "run", &fit_toml.to_string_lossy(),
               "--seed", "1", "--allow-nonconverged-scout", "--no-progress"])
        .output()
        .expect("spawn fit run");
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(out.status.success(), "fit run failed:\nstderr={stderr}");

    let path = chain_starts_for_stage(&dir.join("results"), "posterior");
    let (comments, cols, rows) = parse(&path);

    let source_idx = cols.iter().position(|c| c == "source")
        .unwrap_or_else(|| panic!("no source column in {cols:?}"));
    let beta_idx = cols.iter().position(|c| c == "beta")
        .unwrap_or_else(|| panic!("no beta column in {cols:?}"));
    assert_eq!(rows.len(), 2, "expected 2 chain rows, got {rows:?}");

    // Premise: `init_mle` really did put both chains on one point. If this
    // ever stops holding the source assertion below stops being about gh#871,
    // so state it rather than assume it.
    let betas: Vec<f64> = rows.iter()
        .map(|r| r[beta_idx].parse::<f64>().expect("numeric beta"))
        .collect();
    assert!((betas[0] - betas[1]).abs() < 1e-12,
        "`init_mle = \"prime\"` is supposed to seed every chain from one \
         upstream point, but the chains started at {betas:?}. The rest of \
         this test assumes the single-point premise.");

    for (i, row) in rows.iter().enumerate() {
        let source = &row[source_idx];
        assert!(!source.contains("uniform_unconstrained"),
            "chain {i} started at the `prime` stage's point estimate, but \
             chain_starts.tsv records its source as {source:?} — the stage's \
             declared `init`, which never ran. gh#871: the file claims \
             independent per-chain draws over {} identical values, and it is \
             the artifact an auditor reads to check whether the chains were \
             started apart.", rows.len());
        assert!(!source.contains(":chain-"),
            "chain {i}'s source is {source:?}. A `:chain-<id>` suffix says \
             this chain got its own draw; every chain here took the same \
             upstream point, so the suffix is a per-chain claim the values \
             contradict.");
        assert_eq!(source, "from_mle",
            "chain {i}'s source is {source:?}; the values came from the \
             upstream stage's point estimate, so the record should name that.");
    }

    let header = comments.iter().find(|c| c.contains("chain_starts"))
        .unwrap_or_else(|| panic!("no `# camdl chain_starts` header in {comments:?}"));
    assert!(!header.contains("uniform_unconstrained"),
        "the file header still names the declared `init`: {header:?}. A reader \
         who only skims the header gets the same wrong answer as gh#871's row \
         labels.");
}
