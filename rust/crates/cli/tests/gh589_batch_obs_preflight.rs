//! gh#589: a misaligned stream must not leave a partial obs child behind.
//!
//! The batch obs loop validates and writes in the same iteration, and writes
//! `obs.json` only after every stream. Adding the alignment guard made that
//! loop body fallible partway through a multi-file write, so an aligned stream
//! followed by a misaligned one would leave the first stream's `.tsv` on disk
//! with no provenance file. Validation is hoisted; this pins that it stays
//! hoisted.

use std::path::{Path, PathBuf};
use std::process::Command;

fn binary() -> PathBuf {
    let manifest = std::env::var("CARGO_MANIFEST_DIR").unwrap();
    Path::new(&manifest).join("../../target/release/camdl")
}

/// Two streams: `weekly` aligns with a 7-day recording cadence, `daily` does
/// not. Declaration order matters — the aligned one is first, so a
/// validate-as-you-write loop would emit it before failing on the second.
const MODEL: &str = r#"
time_unit = 'days

compartments { S, I, R }

parameters {
  beta  : rate in [0.001, 2.0]
  gamma : rate in [0.001, 1.0]
}

transitions {
  infection : S --> I @ beta * S * (I / (S + I + R))
  recovery  : I --> R @ gamma * I
}

init { S = 990  I = 10  R = 0 }

output { trajectories { every = 7 } }

observations {
  weekly {
    columns       { time : time, weekly : count }
    projected     = incidence(infection)
    emit_schedule = every 7 'days
    weekly        ~ poisson(rate = projected)
  }
  daily {
    columns       { time : time, daily : count }
    projected     = incidence(recovery)
    emit_schedule = every 1 'days
    daily         ~ poisson(rate = projected)
  }
}

simulate { from = 0 'days  to = 21 'days }
"#;

#[test]
fn a_misaligned_stream_leaves_no_partial_obs_child() {
    let tmp = tempfile::tempdir().unwrap();
    let model = tmp.path().join("m.camdl");
    std::fs::write(&model, MODEL).unwrap();
    std::fs::write(tmp.path().join("p.toml"), "beta = 0.5\ngamma = 0.1\n").unwrap();
    std::fs::write(
        tmp.path().join("exp.toml"),
        "[config]\nmodel = \"m.camdl\"\nparams = \"p.toml\"\n\
         backend = \"chain_binomial\"\ndt = 1.0\noutput_dir = \"out\"\n\n\
         [obs]\nenabled = true\n",
    )
    .unwrap();

    let out = Command::new(binary())
        .args(["batch", "run", "exp.toml"])
        .current_dir(tmp.path())
        .env("CAMDL_SKIP_VERSION_CHECK", "1")
        .output()
        .expect("spawn camdl");

    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        stderr.contains("not a recorded output time"),
        "the misaligned `daily` stream must be reported, got:\n{stderr}"
    );

    // The load-bearing assertion: NO stream file may survive. A
    // validate-as-you-write loop would have emitted `weekly.tsv` before
    // failing on `daily`, leaving a child with data and no obs.json.
    let mut leftovers = Vec::new();
    for e in walk(&tmp.path().join("out")) {
        let name = e.file_name().unwrap().to_string_lossy().to_string();
        if name.ends_with(".tsv") && (name.starts_with("weekly") || name.starts_with("daily")) {
            leftovers.push(e.display().to_string());
        }
        if name == "obs.json" {
            leftovers.push(e.display().to_string());
        }
    }
    assert!(
        leftovers.is_empty(),
        "a refused obs emission must write no stream files and no obs.json; found: {leftovers:?}"
    );
}

fn walk(root: &Path) -> Vec<PathBuf> {
    let mut out = Vec::new();
    let mut stack = vec![root.to_path_buf()];
    while let Some(d) = stack.pop() {
        let Ok(rd) = std::fs::read_dir(&d) else { continue };
        for e in rd.flatten() {
            let p = e.path();
            if p.is_dir() { stack.push(p); } else { out.push(p); }
        }
    }
    out
}
