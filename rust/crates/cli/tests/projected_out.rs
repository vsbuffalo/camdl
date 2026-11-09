//! `simulate --design-from FIT --projected-out DIR`: each observation stream's
//! projected value on the fit's bound design. Checked end to end through
//! `camdlc` and the `camdl` binary.
//!
//! The stream here reads its negative-binomial dispersion from a data column
//! (`r_disp`). A sampled dataset cannot be written for it (gh#829: the
//! simulator has no file to read the column from), but a projection needs no
//! likelihood argument at all, so `--projected-out` must write it.

use std::path::{Path, PathBuf};
use std::process::{Command, Output};

fn camdl_bin() -> PathBuf {
    let manifest = std::env::var("CARGO_MANIFEST_DIR").unwrap();
    Path::new(&manifest).join("../../target/release/camdl")
}

fn real_camdlc() -> PathBuf {
    let manifest = std::env::var("CARGO_MANIFEST_DIR").unwrap();
    Path::new(&manifest).join("../../../ocaml/_build/default/bin/camdlc.exe")
}

fn built() -> bool {
    let ok = camdl_bin().exists() && real_camdlc().exists();
    if !ok {
        eprintln!("skipping: camdl/camdlc not built");
    }
    ok
}

const MODEL: &str = r#"time_unit = 'days
compartments { S, I, R }
parameters { beta : rate  gamma : rate }
let N = S + I + R
transitions {
  infection : S --> I @ beta * S * I / N
  recovery  : I --> R @ gamma * I
}
observations {
  cases {
    columns   { day : time, cases : count, r_disp : real }
    covers    = starting_on(day, 7 'days)
    projected = incidence(infection)
    cases ~ neg_binomial(mean = 0.5 * projected, r = r_disp)
  }
}
init { S = 9990  I = 10 }
simulate { from = 0 'days  to = 42 'days }
"#;

/// Weekly windows, one of them a hole.
const DATA: &str = "day\tcases\tr_disp\n0\t12\t5\n7\t40\t5\n14\tNA\t5\n21\t90\t5\n";

const FIT: &str = r#"[model]
camdl = "m.camdl"

[data.observations]
cases = "cases.tsv"

[estimate]
beta = { bounds = [0.1, 1.0], prior = { uniform = {} } }

[fixed]
gamma = 0.2

[method]
algorithm  = "mh"
backend    = "ode"
chains     = 1
iterations = 10
burn_in    = 5
"#;

fn setup() -> tempfile::TempDir {
    let dir = tempfile::tempdir().unwrap();
    std::fs::write(dir.path().join("m.camdl"), MODEL).unwrap();
    std::fs::write(dir.path().join("cases.tsv"), DATA).unwrap();
    std::fs::write(dir.path().join("fit.toml"), FIT).unwrap();
    dir
}

fn camdl(dir: &Path, extra: &[&str]) -> Output {
    let mut args = vec![
        "simulate", "m.camdl", "--backend", "ode", "--param", "beta=0.5",
        "--param", "gamma=0.2", "--progress", "none",
    ];
    args.extend_from_slice(extra);
    Command::new(camdl_bin()).env_remove("CLICOLOR_FORCE")
        .current_dir(dir)
        .args(&args)
        .env("CAMDLC", real_camdlc())
        .env("CAMDL_SKIP_VERSION_CHECK", "1")
        .env("CAMDL_IR_CACHE_DIR", dir.join("ircache"))
        .output()
        .expect("spawn camdl")
}

#[test]
fn a_stream_reading_a_data_column_is_projected_but_not_sampled() {
    if !built() {
        return;
    }
    let dir = setup();

    let sampled = camdl(dir.path(), &["--design-from", "fit.toml", "--obs-only-dir", "synth"]);
    assert!(!sampled.status.success(), "a sampled dataset for this stream must stay refused");
    let err = String::from_utf8_lossy(&sampled.stderr);
    assert!(err.contains("gh#829"), "the refusal names gh#829: {err}");

    let projected = camdl(dir.path(), &["--design-from", "fit.toml", "--projected-out", "proj"]);
    assert!(projected.status.success(),
        "projections need no data column: {}", String::from_utf8_lossy(&projected.stderr));
    let text = std::fs::read_to_string(dir.path().join("proj/cases.tsv")).unwrap();
    let mut lines = text.lines();
    assert_eq!(lines.next().unwrap(), "day\tprojected",
        "the stream's time column and `projected`, never the observed column name");
    let rows: Vec<(f64, f64)> = lines
        .map(|l| {
            let mut c = l.split('\t');
            (c.next().unwrap().parse().unwrap(), c.next().unwrap().parse().unwrap())
        })
        .collect();
    assert_eq!(rows.iter().map(|r| r.0).collect::<Vec<_>>(), vec![0.0, 7.0, 14.0, 21.0],
        "one row per bound row, the hole at day 14 included");
    assert!(rows.iter().all(|r| r.1.is_finite() && r.1 > 0.0),
        "every row, the hole included, carries a positive projection: {rows:?}");
    assert!(rows[1].1.fract() != 0.0,
        "an ODE projection is written at full precision, not rounded to a count: {rows:?}");
}

#[test]
fn projected_out_requires_a_design() {
    if !built() {
        return;
    }
    let dir = setup();
    let out = camdl(dir.path(), &["--projected-out", "proj"]);
    assert!(!out.status.success(), "--projected-out without --design-from has no design to project on");
    let err = String::from_utf8_lossy(&out.stderr);
    assert!(err.contains("--design-from"), "the error names the missing flag: {err}");
}
