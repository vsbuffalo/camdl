//! gh#652 — a fit whose `fit.toml` lives OUTSIDE the results tree, with data
//! paths written relative to that toml (`../data/streams/x.tsv`), must stay
//! reachable through its run handle.
//!
//! This is the layout of any project that keeps its configs portable (and camdl
//! itself warns about absolute paths): configs in `fits/`, data in `data/`,
//! results in `results/`. Recovering the config from a run handle reads the
//! archived `fit.toml.original` inside the segment, so anchoring its relative
//! paths at the archive's own location resolved `../data/streams/x.tsv` to
//! `results/fits/data/streams/x.tsv` — a file that never existed — and
//! `camdl compare` could not derive a prequential for ANY such fit.
//!
//! The fit.toml is edited after the fit completes (a comment appended), which is
//! how the downstream project hit this: the live config no longer matches the
//! archive byte-for-byte, so the archive is what gets read.

use std::path::{Path, PathBuf};
use std::process::Command;

fn binary() -> PathBuf {
    let manifest = std::env::var("CARGO_MANIFEST_DIR").unwrap();
    Path::new(&manifest).join("../../target/release/camdl")
}

/// A closed SIR with a weekly NegBinomial observation — small and well-behaved.
const MODEL: &str = r#"
time_unit = 'days

compartments { S, I, R }

parameters {
  beta  : rate         in [0.05, 1.0]  ~ log_normal(mu = -1.0, sigma = 0.5)
  gamma : rate         in [0.01, 0.5]  ~ log_normal(mu = -2.0, sigma = 0.5)
  N0    : count
  I0    : count
  rho   : probability  in [0.05, 0.95] ~ beta(alpha = 2.0, beta = 5.0)
  k     : positive     in [1.0, 100.0] ~ half_normal(sigma = 10.0)
}

let N = S + I + R

transitions {
  infection : S --> I  @ beta * S * I / N
  recovery  : I --> R  @ gamma * I
}

init {
  S = N0 - I0
  I = I0
}

observations {
  weekly_cases {
    columns       { time : time, weekly_cases : count }
    projected     = incidence(infection)
    emit_schedule = every 7 'days
    weekly_cases  ~ neg_binomial(mean = rho * projected, r = k)
  }
}

simulate {
  from = 0 'days
  to   = 80 'days
}
"#;

const DATA: &str = "time\tweekly_cases\n7\t16\n14\t166\n21\t626\n28\t1303\n35\t1260\n42\t1023\n49\t327\n56\t91\n";

/// Every path relative and reaching UP out of the config's own directory: the
/// model, the data stream, and the results root all sit beside `fits/`, not
/// inside it. `rho` is fixed per-fit so the two configs are genuinely distinct.
fn portable_fit_toml(rho: f64) -> String {
    format!(
        r#"output_dir = "../results"

[model]
camdl = "../models/sir.camdl"

[data.observations]
weekly_cases = "../data/streams/weekly_cases.tsv"

[estimate]
beta  = {{ bounds = [0.05, 1.0], start = 0.4 }}
gamma = {{ bounds = [0.01, 0.5], start = 0.15 }}

[fixed]
N0  = 10000
I0  = 10
rho = {rho}
k   = 10.0

[stages.scout]
algorithm = "if2"
backend = "chain_binomial"
chains = 2
particles = 150
iterations = 15
cooling = 0.7
"#
    )
}

fn run(bin: &Path, dir: &Path, args: &[&str]) -> std::process::Output {
    Command::new(bin)
        .args(args)
        .current_dir(dir)
        .env("CAMDL_SKIP_VERSION_CHECK", "1")
        .output()
        .expect("spawn camdl")
}

/// The fit segment whose sidecar carries `label`.
fn segment_with_label(results: &Path, label: &str) -> PathBuf {
    let fits = results.join("fits");
    for e in std::fs::read_dir(&fits).unwrap().flatten() {
        let meta = e.path().join("fit.meta.json");
        if let Ok(txt) = std::fs::read_to_string(&meta) {
            if let Ok(j) = serde_json::from_str::<serde_json::Value>(&txt) {
                if j["label"] == label {
                    return e.path();
                }
            }
        }
    }
    panic!("no fit segment labeled {label} under {}", fits.display());
}

#[test]
fn compare_derives_a_prequential_for_a_fit_with_relative_data_paths() {
    let bin = binary();
    assert!(
        bin.exists(),
        "release camdl binary missing: {} — run `make build-rust` or `make test`",
        bin.display()
    );

    let tmp = std::env::temp_dir().join(format!("camdl_gh652_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&tmp);
    std::fs::create_dir_all(tmp.join("models")).unwrap();
    std::fs::create_dir_all(tmp.join("data/streams")).unwrap();
    std::fs::create_dir_all(tmp.join("fits")).unwrap();
    std::fs::write(tmp.join("models/sir.camdl"), MODEL).unwrap();
    std::fs::write(tmp.join("data/streams/weekly_cases.tsv"), DATA).unwrap();
    std::fs::write(tmp.join("fits/a.toml"), portable_fit_toml(0.5)).unwrap();
    std::fs::write(tmp.join("fits/b.toml"), portable_fit_toml(0.6)).unwrap();

    for (cfg, label) in [("fits/a.toml", "a"), ("fits/b.toml", "b")] {
        let out = run(&bin, &tmp, &["fit", "run", cfg, "--label", label, "--seed", "1"]);
        assert!(
            out.status.success(),
            "fit run {cfg} failed:\nstdout={}\nstderr={}",
            String::from_utf8_lossy(&out.stdout),
            String::from_utf8_lossy(&out.stderr)
        );
    }

    let results = tmp.join("results");
    let seg_a = segment_with_label(&results, "a");
    let seg_b = segment_with_label(&results, "b");
    // The data really does live outside the results tree — otherwise the
    // segment-anchored resolution would accidentally be right.
    assert!(
        !results.join("fits/data/streams/weekly_cases.tsv").exists(),
        "the data must NOT be co-located with the segments for this test to bite"
    );

    // Edit the live configs so the archive is what the run handle must read.
    for cfg in ["fits/a.toml", "fits/b.toml"] {
        let p = tmp.join(cfg);
        let text = std::fs::read_to_string(&p).unwrap();
        std::fs::write(&p, format!("# a comment added after the fit ran\n{text}")).unwrap();
    }

    let a = seg_a.to_string_lossy().into_owned();
    let b = seg_b.to_string_lossy().into_owned();
    let out = run(
        &bin,
        &tmp,
        &["compare", &a, &b, "--particles", "300", "--seed", "1"],
    );
    let stdout = String::from_utf8_lossy(&out.stdout);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        out.status.success(),
        "compare on two run-dir handles whose configs use `../`-relative data \
         paths must derive prequentials (gh#652):\nstdout={stdout}\nstderr={stderr}"
    );
    assert!(
        stdout.contains("elpd"),
        "a real comparison table, not an empty one:\n{stdout}"
    );
    // A positive scored horizon proves the observation data was actually found
    // and scored, not silently skipped. The row name is the segment's dir name.
    for seg in [&seg_a, &seg_b] {
        let name = seg.file_name().unwrap().to_string_lossy().into_owned();
        let row = stdout
            .lines()
            .find(|l| l.trim_start().starts_with(&name))
            .unwrap_or_else(|| panic!("no row for {name}:\n{stdout}"));
        let t: usize = row
            .split_whitespace()
            .nth(1)
            .and_then(|c| c.parse().ok())
            .unwrap_or_else(|| panic!("no T_score cell in row {row:?}"));
        assert!(t > 0, "{name} scored a positive number of observations; row: {row}");
    }

    let _ = std::fs::remove_dir_all(&tmp);
}
