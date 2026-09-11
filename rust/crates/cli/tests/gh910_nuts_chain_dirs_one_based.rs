//! gh#910: a NUTS leaf's `chain_N/` directories carry the numbers its own
//! artifacts hand the reader.
//!
//! gh#781 made every artifact that names a chain name it the same way —
//! `chain_starts.tsv`'s `chain_id`, the `fit summary` chain table, the
//! `bad_init` records, every stderr refusal — all 1-based, "matching the
//! `chain_N/` directories". NUTS wrote its directories from the 0-based loop
//! index, so a two-chain NUTS leaf held `chain_0/` and `chain_1/` while
//! `chain_starts.tsv` named chains 1 and 2. A reader who took chain 1 from any
//! of those artifacts and opened `chain_1/` got chain 2's trace; the last
//! chain's directory did not exist. Nothing errored and the trace was
//! plausible.
//!
//! The two claims pinned here, on a real two-chain NUTS leaf:
//!
//! 1. The set of `chain_N/` directory names is exactly the set of `chain_id`
//!    values in `chain_starts.tsv` — so the lookup a refusal sends a reader on
//!    is a match on the number they were given, with no arithmetic.
//! 2. Following `draws.tsv`'s `chain` join key into the directory it names
//!    lands on that chain's own trace. `draws.tsv`'s `chain` column is the
//!    0-based join key to `trajectories.tsv` (gh#666) and stays 0-based, so
//!    chain `c`'s draws must be the draws in `chain_{c+1}/trace.tsv` — this is
//!    the claim that actually fails silently when the directories are misnamed.

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};
use std::process::Command;

fn binary() -> PathBuf {
    let manifest = std::env::var("CARGO_MANIFEST_DIR").unwrap();
    Path::new(&manifest).join("../../target/release/camdl")
}

fn camdlc() -> PathBuf {
    let manifest = std::env::var("CARGO_MANIFEST_DIR").unwrap();
    Path::new(&manifest).join("../../../ocaml/_build/default/bin/camdlc.exe")
}

fn require_tools() {
    assert!(
        binary().exists(),
        "release camdl binary missing: {} — run `make build-rust` or `make test`",
        binary().display()
    );
    assert!(
        camdlc().exists(),
        "camdlc.exe missing: {} — run `make build-ocaml`",
        camdlc().display()
    );
}

struct TempDir(PathBuf);
impl TempDir {
    fn path(&self) -> &Path {
        &self.0
    }
}
impl Drop for TempDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}
fn tempdir() -> TempDir {
    let ns = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let base =
        std::env::temp_dir().join(format!("camdl_gh910_{}_{}", std::process::id(), ns));
    std::fs::create_dir_all(&base).unwrap();
    TempDir(base)
}

/// An SIR whose `beta` carries a proper prior — NUTS refuses an implicit
/// improper-uniform — scored on ODE incidence. Small and short: the claim is
/// about file names, not about recovery, so the chains only have to run.
const MODEL: &str = r#"
time_unit = 'days
compartments { S, I, R }
parameters {
  beta  : rate  in [0.05, 5.0] ~ log_normal(mu = 0.0, sigma = 1.0)
  gamma : rate  in [0.01, 1.0]
  N0    : count in [100, 100000]
}
transitions {
  infection : S --> I @ beta * S * I / N0
  recovery  : I --> R @ gamma * I
}
observations {
  cases {
    columns       { time : time, cases : count }
    covers        = closing_at(time, 3 'days)
    projected     = incidence(infection)
    emit_schedule = every 3 'days
    cases ~ poisson(rate = projected)
  }
}
init { S = 9990  I = 10 }
simulate { from = 0 'days  to = 30 'days }
"#;

const N_CHAINS: usize = 2;

/// The seed leaf: the directory holding the stage's `nuts_summary.json`.
fn seed_leaf(root: &Path) -> PathBuf {
    let mut stack = vec![root.to_path_buf()];
    while let Some(d) = stack.pop() {
        if d.join("nuts_summary.json").is_file() {
            return d;
        }
        if let Ok(entries) = std::fs::read_dir(&d) {
            for e in entries.flatten() {
                let p = e.path();
                if p.is_dir() {
                    stack.push(p);
                }
            }
        }
    }
    panic!("no nuts stage leaf under {}", root.display());
}

/// The trailing integer of every `chain_<N>/` directory on the leaf.
fn chain_dir_numbers(leaf: &Path) -> BTreeSet<usize> {
    std::fs::read_dir(leaf)
        .unwrap()
        .flatten()
        .filter(|e| e.path().is_dir())
        .filter_map(|e| {
            e.file_name().to_str()?.strip_prefix("chain_")?.parse::<usize>().ok()
        })
        .collect()
}

/// The `chain_id` column of `chain_starts.tsv`, deduplicated.
fn chain_start_ids(leaf: &Path) -> BTreeSet<usize> {
    let text = std::fs::read_to_string(leaf.join("chain_starts.tsv")).unwrap();
    let mut body = text.lines().filter(|l| !l.starts_with('#') && !l.trim().is_empty());
    let cols: Vec<&str> = body.next().expect("chain_starts.tsv header").split('\t').collect();
    let id_col = cols
        .iter()
        .position(|c| *c == "chain_id")
        .unwrap_or_else(|| panic!("no chain_id column in {cols:?}"));
    body.map(|l| {
        l.split('\t').nth(id_col).expect("chain_id field").parse::<usize>().unwrap()
    })
    .collect()
}

/// One named column of a TSV, parsed as f64, skipping `#` comment lines.
fn column(path: &Path, name: &str) -> Vec<f64> {
    let text = std::fs::read_to_string(path)
        .unwrap_or_else(|e| panic!("cannot read {}: {e}", path.display()));
    let mut lines = text.lines().filter(|l| !l.starts_with('#') && !l.trim().is_empty());
    let header: Vec<&str> = lines.next().expect("header").split('\t').collect();
    let col = header
        .iter()
        .position(|c| *c == name)
        .unwrap_or_else(|| panic!("no `{name}` column in {}: {header:?}", path.display()));
    lines
        .map(|l| l.split('\t').nth(col).expect("field").parse::<f64>().unwrap())
        .collect()
}

/// `draws.tsv` rows for one 0-based `chain` key, as that chain's `beta` values.
fn draws_for_chain(leaf: &Path, chain: usize) -> Vec<f64> {
    let text = std::fs::read_to_string(leaf.join("draws.tsv")).unwrap();
    let mut lines = text.lines().filter(|l| !l.trim().is_empty());
    let header: Vec<&str> = lines.next().expect("draws.tsv header").split('\t').collect();
    assert_eq!(header[0], "chain", "draws.tsv leads with its chain key");
    let beta = header.iter().position(|c| *c == "beta").expect("beta column");
    lines
        .filter_map(|l| {
            let f: Vec<&str> = l.split('\t').collect();
            (f[0].parse::<usize>().ok()? == chain).then(|| f[beta].parse::<f64>().unwrap())
        })
        .collect()
}

#[test]
fn a_nuts_leaf_numbers_its_chain_directories_from_one() {
    require_tools();
    let tmp = tempdir();

    let model = tmp.path().join("sir.camdl");
    std::fs::write(&model, MODEL).unwrap();
    let out = Command::new(camdlc()).arg(&model).output().unwrap();
    assert!(out.status.success(), "camdlc failed: {}", String::from_utf8_lossy(&out.stderr));
    let ir = tmp.path().join("sir.ir.json");
    std::fs::write(&ir, &out.stdout).unwrap();

    // Synthetic incidence at a known beta — the data only has to be scoreable.
    let truth = tmp.path().join("truth.toml");
    std::fs::write(&truth, "beta = 0.9\ngamma = 0.3\nN0 = 10000\n").unwrap();
    let data = tmp.path().join("cases.tsv");
    let sim = Command::new(binary())
        .args(["simulate"])
        .arg(&ir)
        .args(["--params"])
        .arg(&truth)
        .args(["--backend", "ode", "--dt", "1", "--seed", "1", "--obs-only"])
        .arg(&data)
        .env("CAMDL_SKIP_VERSION_CHECK", "1")
        .output()
        .unwrap();
    assert!(sim.status.success(), "simulate failed: {}", String::from_utf8_lossy(&sim.stderr));

    // `starts = "single"` puts both chains at one point, so the run is short and
    // deterministic; the chains still differ, because each carries its own seed.
    let out_dir = tmp.path().join("out");
    let fit_toml = tmp.path().join("fit.toml");
    std::fs::write(
        &fit_toml,
        format!(
            r#"output_dir = "{out}"

[model]
camdl = "{ir}"

[data.observations]
cases = "{data}"

[estimate]
beta = {{ bounds = [0.05, 5.0], start = 0.4 }}

[fixed]
gamma = 0.3
N0 = 10000

[method]
algorithm = "nuts"
backend = "ode"
chains = {N_CHAINS}
warmup = 10
samples = 10
starts = "single"
"#,
            out = out_dir.display(),
            ir = ir.display(),
            data = data.display()
        ),
    )
    .unwrap();

    let run = Command::new(binary())
        .args(["fit", "run"])
        .arg(&fit_toml)
        .env("CAMDL_SKIP_VERSION_CHECK", "1")
        .output()
        .unwrap();
    assert!(
        run.status.success(),
        "nuts+ode `fit run` must succeed: {}",
        String::from_utf8_lossy(&run.stderr)
    );

    let leaf = seed_leaf(&out_dir);
    let dirs = chain_dir_numbers(&leaf);
    let starts = chain_start_ids(&leaf);

    // Claim 1: the directories and the starts file name the same chains.
    assert_eq!(
        dirs, starts,
        "the leaf holds chain directories {dirs:?} but chain_starts.tsv names \
         chains {starts:?}; a reader joining the two would need arithmetic that \
         neither states (leaf {})",
        leaf.display()
    );
    // The proposition `io::progress::CHAIN_NUMBERING` states — "1-based,
    // matching the chain_N/ directories" — checked directly against the leaf,
    // since a NUTS stage writes no `chains` block to carry the sentence (below).
    let one_based: BTreeSet<usize> = (1..=N_CHAINS).collect();
    assert_eq!(
        dirs, one_based,
        "the chain directories of a {N_CHAINS}-chain leaf are chain_1..chain_{N_CHAINS} \
         — \"1-based, matching the chain_N/ directories\", which is what every \
         artifact on the leaf says of its chain column — but they are {dirs:?} \
         (leaf {})",
        leaf.display()
    );

    // Claim 2: following the join key lands on that chain's own trace. Chain
    // `c` of `draws.tsv` (0-based, gh#666) is `chain_{c+1}/`.
    for c in 0..N_CHAINS {
        let dir = leaf.join(format!("chain_{}", c + 1));
        assert!(
            dir.is_dir(),
            "draws.tsv carries chain {c}, whose trace is chain_{}/ — no such directory \
             (leaf holds {dirs:?})",
            c + 1
        );
        let drawn = draws_for_chain(&leaf, c);
        assert!(!drawn.is_empty(), "chain {c} has no rows in draws.tsv");
        let traced = column(&dir.join("trace.tsv"), "beta");
        assert!(
            traced.len() >= drawn.len(),
            "chain_{}/trace.tsv has {} rows, fewer than the {} draws.tsv keeps for \
             chain {c}",
            c + 1,
            traced.len(),
            drawn.len()
        );
        let tail = &traced[traced.len() - drawn.len()..];
        for (i, (d, t)) in drawn.iter().zip(tail).enumerate() {
            assert!(
                (d - t).abs() <= 1e-12 * d.abs().max(t.abs()).max(1.0),
                "draw {i} of chain {c} is beta={d}, but chain_{}/trace.tsv holds \
                 beta={t} there — the join key names a directory that is not that \
                 chain's (leaf {})",
                c + 1,
                leaf.display()
            );
        }
    }

    // `progress.json`'s numbering sentence, where the file states one. A NUTS
    // stage's heartbeat is built with a chain roster of 0 (`fit/mod.rs`,
    // `stage_heartbeat`), so it writes no `chains` block today and there is no
    // sentence to check; when one is wired, the ids in it must name the
    // directories the sentence claims.
    let progress: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(leaf.join("progress.json")).unwrap())
            .unwrap();
    if let Some(chains) = progress.get("chains") {
        assert_eq!(
            chains["numbering"], "1-based, matching the chain_N/ directories",
            "the file explains its own chain column:\n{progress:#}"
        );
        for row in chains["chains"].as_array().expect("chain rows") {
            let id = row["chain"].as_u64().unwrap() as usize;
            assert!(
                dirs.contains(&id),
                "progress.json names chain {id}, but the leaf holds {dirs:?}"
            );
        }
    }
}
