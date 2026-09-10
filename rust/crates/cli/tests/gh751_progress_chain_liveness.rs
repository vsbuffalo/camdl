//! gh#751: `progress.json` says how many chains are actually sampling.
//!
//! While a PGAS stage runs, `progress.json` is the only thing written at the
//! seed level, and it carried a global sweep counter and nothing about chains:
//!
//! ```json
//! {"updated_at": 1787670244, "pid": 2882,
//!  "state": {"running": {"phase": "burn_in", "step": 276, "total": 2000}}}
//! ```
//!
//! A fit that refused 23 of its 24 chains at their starts wrote exactly that,
//! byte-for-byte what a healthy 24-chain fit writes, for hours. Refusal is not
//! rare in the downstream national models — eight consecutive arms got between
//! 1 and 16 of the 24 chains they paid for — and a chain is refused at its
//! START, so the count is knowable in the first minutes. It was reported only
//! afterwards, in `fit_state.toml` and `diagnostics.json`.
//!
//! The fixture: `iota = 1e-4` at `S = 1000` gives an expected importation count
//! of 0.6 over the window, so the unconditional reference draw is all-zero
//! about half the time, and with 2 particles the conditional SMC move often
//! cannot rescue it. `starts = "single"` puts every chain at that same point,
//! and a point rule permits no redraw (gh#887), so the chains that draw badly
//! are refused outright — a real fit in which some chains run and some do not.

use std::path::{Path, PathBuf};
use std::process::Command;

fn binary() -> PathBuf {
    let manifest = std::env::var("CARGO_MANIFEST_DIR").unwrap();
    Path::new(&manifest).join("../../target/release/camdl")
}

fn skip_if_missing_binary() -> PathBuf {
    let bin = binary();
    assert!(
        bin.exists(),
        "release camdl binary missing: {} — run `make build-rust` or `make test`",
        bin.display()
    );
    bin
}

const MODEL: &str = r#"
time_unit = 'days
compartments { S, I, R }
parameters {
  beta  : rate  in [0.001, 5.0]
  gamma : rate  in [0.001, 1.0]
  iota  : rate  in [0.0, 1.0]
  N0    : count in [100, 10000]
}
transitions {
  infection : S --> I @ beta * S * I / N0 + iota * S
  recovery  : I --> R @ gamma * I
}
observations {
  cases {
    columns       { time : time, cases : count }
    projected  = prevalence(I)
    emit_schedule = every 1 'days
    cases ~ poisson(rate = projected)
  }
}
init { S = 1000  I = 0 }
simulate { from = 0 'days  to = 6 'days }
"#;

const DATA: &str = "time\tcases\n1\t150\n2\t300\n3\t450\n4\t550\n5\t600\n6\t620\n";

const FIT_TOML: &str = r#"output_dir = "results"
[model]
camdl = "sir.camdl"
[data.observations]
cases = "cases.tsv"
[config]
dt = 1.0
[estimate]
beta  = { bounds = [0.01, 5.0], prior = { log_normal = { mu = -0.3, sigma = 0.5 } }, start = 0.3 }
gamma = { bounds = [0.01, 1.0], prior = { log_normal = { mu = -1.2, sigma = 0.5 } }, start = 0.1 }
iota  = { bounds = [0.0, 1.0],  prior = { uniform = { lower = 0.0, upper = 1.0 } }, start = 0.0001 }
[fixed]
N0 = 1000
[method]
algorithm      = "pgas"
backend        = "chain_binomial"
chains         = 6
particles      = 2
sweeps         = 6
burn_in        = 1
thin           = 1
starts         = "single"
"#;

/// The seed leaf: the directory holding `progress.json` and `diagnostics.json`.
fn seed_leaf(root: &Path) -> PathBuf {
    let mut stack = vec![root.to_path_buf()];
    while let Some(d) = stack.pop() {
        if d.join("progress.json").is_file() {
            return d;
        }
        if let Ok(entries) = std::fs::read_dir(&d) {
            for e in entries.flatten() {
                if e.path().is_dir() {
                    stack.push(e.path());
                }
            }
        }
    }
    panic!("no progress.json under {}", root.display());
}

fn read_json(p: &Path) -> serde_json::Value {
    serde_json::from_str(&std::fs::read_to_string(p).unwrap_or_else(|e| {
        panic!("cannot read {}: {e}", p.display())
    }))
    .unwrap_or_else(|e| panic!("cannot parse {}: {e}", p.display()))
}

#[test]
fn progress_json_names_the_refused_chains_and_counts_them() {
    let bin = skip_if_missing_binary();
    let tmp = std::env::temp_dir().join(format!("camdl_gh751_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&tmp);
    std::fs::create_dir_all(&tmp).unwrap();
    std::fs::write(tmp.join("sir.camdl"), MODEL).unwrap();
    std::fs::write(tmp.join("cases.tsv"), DATA).unwrap();
    std::fs::write(tmp.join("fit.toml"), FIT_TOML).unwrap();

    let out = Command::new(&bin)
        .args(["fit", "run", "fit.toml", "--seed", "1", "--progress", "none"])
        .current_dir(&tmp)
        .env("CAMDL_SKIP_VERSION_CHECK", "1")
        .output()
        .expect("spawn camdl");
    let stderr = String::from_utf8_lossy(&out.stderr).into_owned();
    assert!(
        out.status.success(),
        "the fit must complete on its surviving chains:\n{stderr}"
    );

    let leaf = seed_leaf(&tmp.join("results"));
    let progress = read_json(&leaf.join("progress.json"));

    // The three fields every existing reader has are untouched.
    assert!(progress["updated_at"].is_u64(), "{progress:#}");
    assert!(progress["pid"].is_u64(), "{progress:#}");
    assert_eq!(progress["state"], "done", "{progress:#}");

    // The new block.
    let chains = &progress["chains"];
    assert!(
        !chains.is_null(),
        "progress.json must carry per-chain liveness:\n{progress:#}"
    );
    assert_eq!(chains["total"], 6, "the roster the stage was configured to run");
    let refused = chains["refused"].as_u64().unwrap();
    let completed = chains["completed"].as_u64().unwrap();
    assert!(
        refused > 0,
        "this fixture must leave at least one chain refused, or it exercises \
         nothing. If the sampler's RNG changed, re-tune `iota`/`particles` or \
         the seed rather than dropping the assertion.\n{progress:#}\n{stderr}"
    );
    assert!(completed > 0, "and at least one must survive:\n{progress:#}");
    assert_eq!(
        refused + completed,
        6,
        "every chain reported, so none is left as `not_started`:\n{progress:#}"
    );

    // Each row is flat, 1-based, and a refusal carries its short tag.
    let rows = chains["chains"].as_array().expect("per-chain rows");
    assert_eq!(rows.len(), 6);
    let mut refused_ids: Vec<u64> = Vec::new();
    for r in rows {
        let id = r["chain"].as_u64().unwrap();
        assert!((1..=6).contains(&id), "1-based chain id: {r:#}");
        match r["status"].as_str().unwrap() {
            "refused" => {
                assert_eq!(r["reason"], "non_finite_start", "{r:#}");
                refused_ids.push(id);
            }
            "completed" => assert!(r["reason"].is_null(), "{r:#}"),
            other => panic!("unexpected status {other}: {r:#}"),
        }
    }
    refused_ids.sort_unstable();

    // The claim that matters: the ids agree with the artifact written after
    // the stage, which is what a reader would otherwise have had to wait for.
    // `diagnostics.json` numbers chains 0-based, `progress.json` 1-based (and
    // says so in its own `numbering` field), so they differ by exactly one.
    assert_eq!(
        chains["numbering"], "1-based, matching the chain_N/ directories",
        "the file explains its own chain column:\n{progress:#}"
    );
    let diagnostics = read_json(&leaf.join("diagnostics.json"));
    let mut bad_init: Vec<u64> = diagnostics
        .as_array()
        .expect("diagnostics.json is a list")
        .iter()
        .filter(|d| d["kind"]["type"] == "bad_init")
        .map(|d| d["kind"]["chain_id"].as_u64().unwrap() + 1)
        .collect();
    bad_init.sort_unstable();
    assert_eq!(
        refused_ids, bad_init,
        "the chains progress.json calls refused are the ones diagnostics.json \
         records a bad_init for:\n{progress:#}"
    );

    // …and the directory each refused chain names holds no draws to read.
    for id in &refused_ids {
        let dir = leaf.join(format!("chain_{id}"));
        assert!(dir.is_dir(), "chain_{id}/ exists but contributed nothing");
    }

    let _ = std::fs::remove_dir_all(&tmp);
}
