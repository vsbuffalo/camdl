//! A simulated or predicted observation reads the state where the fit scores
//! the row: the instant it reads, or its period's stop (`StreamTimes::closes`),
//! never the row's label (D59, `docs/dev/temporal-architecture.md`).
//!
//! The witness is a `starting_on(time, 7 'days)` stream, whose label is its
//! period's start, with a likelihood whose mean reads both the state and the
//! time, `I * t / 1 day`, under negligible noise. Integrated with the ODE
//! backend the trajectory is deterministic, so each written or predicted value
//! is `I(s)·s` at the time `s` it was evaluated: the row labelled 0 must carry
//! `7·I(7)`. Reading the state at the label would give `0·I(0)`; reading the
//! state at the stop but passing the label as `t` would give `0·I(7)`.

use std::path::{Path, PathBuf};
use std::process::Command;

fn camdl_bin() -> PathBuf {
    let manifest = std::env::var("CARGO_MANIFEST_DIR").unwrap();
    let p = Path::new(&manifest).join("../../target/release/camdl");
    assert!(p.exists(), "release camdl binary missing: {} - run `make build-rust` or `make test`", p.display());
    p
}

fn camdlc_bin() -> PathBuf {
    let manifest = std::env::var("CARGO_MANIFEST_DIR").unwrap();
    let p = Path::new(&manifest).join("../../../ocaml/_build/default/bin/camdlc.exe");
    assert!(p.exists(), "camdlc missing: {} - run `make build-ocaml` or `make test`", p.display());
    p
}

struct Tmp(PathBuf);
impl Drop for Tmp { fn drop(&mut self) { let _ = std::fs::remove_dir_all(&self.0); } }
fn tempdir(tag: &str) -> Tmp {
    let ns = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos();
    let p = std::env::temp_dir().join(format!("camdl_state_where_scored_{tag}_{}_{ns}", std::process::id()));
    std::fs::create_dir_all(&p).unwrap();
    Tmp(p)
}

fn camdl(dir: &Path, args: &[&str]) -> std::process::Output {
    Command::new(camdl_bin()).env_remove("CLICOLOR_FORCE")
        .current_dir(dir)
        .args(args)
        .env("CAMDL_SKIP_VERSION_CHECK", "1")
        .output()
        .expect("camdl must invoke")
}

fn ok(o: std::process::Output, what: &str) -> std::process::Output {
    assert!(o.status.success(), "{what} failed:\n{}", String::from_utf8_lossy(&o.stderr));
    o
}

const MODEL: &str = r#"
time_unit = 'days
compartments { S, I, R }
parameters {
  beta  : rate     in [0.001, 5.0]
  gamma : rate     in [0.01, 1.0]
  sd    : positive in [0.0001, 0.001]
}
transitions {
  infection : S --> I @ beta * S * I / (S + I + R)
  recovery  : I --> R @ gamma * I
}
observations {
  cases {
    columns   { time : time, cases : real }
    covers    = starting_on(time, 7 'days)
    projected = incidence(infection)
    cases     ~ normal(mean = 0 * projected + I * t / (1 'days), sd = sd)
  }
}
init { S = 990  I = 10  R = 0 }
simulate { from = 0 'days  to = 42 'days  dt = 1 'days }
"#;

/// The row labels of the bound data: weekly, one hole.
const LABELS: [f64; 5] = [0.0, 7.0, 14.0, 21.0, 28.0];
const HOLE: f64 = 21.0;

/// Compile the model, integrate it at the truth, and return the IR path and
/// `I` at each whole day.
fn setup(dir: &Path) -> (PathBuf, Vec<f64>) {
    let src = dir.join("m.camdl");
    std::fs::write(&src, MODEL).unwrap();
    let out = Command::new(camdlc_bin()).env_remove("CLICOLOR_FORCE").arg(&src).output().unwrap();
    assert!(out.status.success(), "camdlc failed: {}", String::from_utf8_lossy(&out.stderr));
    let ir = dir.join("m.ir.json");
    std::fs::write(&ir, &out.stdout).unwrap();
    std::fs::write(dir.join("truth.toml"), "beta = 0.6\ngamma = 0.2\nsd = 0.0001\n").unwrap();
    ok(camdl(dir, &["simulate", "m.ir.json", "--params", "truth.toml", "--backend", "ode",
        "-o", "traj.tsv"]), "simulate -o");
    let text = std::fs::read_to_string(dir.join("traj.tsv")).unwrap();
    let mut lines = text.lines().filter(|l| !l.starts_with('#'));
    let header: Vec<&str> = lines.next().unwrap().split('\t').collect();
    let (ti, ii) = (header.iter().position(|c| *c == "t").unwrap(),
        header.iter().position(|c| *c == "I").unwrap());
    let mut i_at: Vec<f64> = Vec::new();
    for l in lines {
        let f: Vec<&str> = l.split('\t').collect();
        let t: f64 = f[ti].parse().unwrap();
        assert_eq!(t, i_at.len() as f64, "the trajectory is daily from 0");
        i_at.push(f[ii].parse().unwrap());
    }
    // The witness only discriminates if `I` moves across a week.
    assert!((i_at[0] - i_at[7]).abs() > 10.0, "I must change over a week: {i_at:?}");
    (ir, i_at)
}

/// The mean the likelihood gives a row scored at day `s`.
fn mean_at(i_at: &[f64], s: usize) -> f64 {
    i_at[s] * s as f64
}

/// Write the bound data, each observed value the mean at the row's period stop.
fn write_data(dir: &Path, i_at: &[f64]) {
    let mut s = String::from("time\tcases\n");
    for t in LABELS {
        match t == HOLE {
            true => s.push_str(&format!("{t}\tNA\n")),
            false => s.push_str(&format!("{t}\t{}\n", mean_at(i_at, t as usize + 7))),
        }
    }
    std::fs::write(dir.join("cases.tsv"), s).unwrap();
}

/// `simulate --design-from` draws each row from the state at its period's
/// stop, where the fit scores it, and writes the row under its own label.
#[test]
fn design_from_draws_each_row_from_the_state_where_it_is_scored() {
    let tmp = tempdir("design");
    let dir = &tmp.0;
    let (_, i_at) = setup(dir);
    write_data(dir, &i_at);
    std::fs::write(dir.join("fit.toml"),
        "[model]\ncamdl = \"m.ir.json\"\n[data.observations]\ncases = \"cases.tsv\"\n\
         [estimate]\nsd = { bounds = [0.0001, 0.001], start = 0.0005 }\n\
         [fixed]\nbeta = 0.6\ngamma = 0.2\n\
         [method]\nalgorithm = \"nl-sbplx\"\nbackend = \"ode\"\nchains = 1\n").unwrap();
    ok(camdl(dir, &["simulate", "m.ir.json", "--params", "truth.toml", "--design-from", "fit.toml",
        "--obs-only-dir", "synth", "--backend", "ode", "--seed", "3"]), "simulate --design-from");

    let text = std::fs::read_to_string(dir.join("synth/cases.tsv")).unwrap();
    let rows: Vec<(String, String)> = text.lines().skip(1)
        .map(|l| { let mut f = l.split('\t'); (f.next().unwrap().into(), f.next().unwrap().into()) })
        .collect();
    assert_eq!(rows.iter().map(|(t, _)| t.as_str()).collect::<Vec<_>>(),
        vec!["0", "7", "14", "21", "28"], "the time column is the bound file's, as written");
    for (t, v) in &rows {
        let label: f64 = t.parse().unwrap();
        if label == HOLE {
            assert_eq!(v, "NA", "the hole stays a hole");
            continue;
        }
        let v: f64 = v.parse().unwrap();
        let (at_stop, at_label) =
            (mean_at(&i_at, label as usize + 7), mean_at(&i_at, label as usize));
        assert!((v - at_stop).abs() < 0.01,
            "row labelled {label} measures [{label}, {}) and is scored at its stop: \
             expected I·t there = {at_stop}, got {v} (at the label it is {at_label})",
            label + 7.0);
    }
}

/// `fit predict`'s free-forward band draws each row from the state, and with
/// the time, where the row is scored: its period's stop. The rows themselves,
/// and the table's time column, are unchanged.
#[test]
fn fit_predict_evaluates_each_row_where_it_is_scored() {
    let tmp = tempdir("predict");
    let dir = &tmp.0;
    let (_, i_at) = setup(dir);
    write_data(dir, &i_at);
    std::fs::write(dir.join("fit.toml"),
        "[model]\ncamdl = \"m.ir.json\"\n[data.observations]\ncases = \"cases.tsv\"\n\
         [estimate]\nsd = { bounds = [0.0001, 0.001], start = 0.0005, \
         prior = { uniform = { lower = 0.0001, upper = 0.001 } } }\n\
         [fixed]\nbeta = 0.6\ngamma = 0.2\n\
         [method]\nalgorithm = \"mh\"\nbackend = \"ode\"\nchains = 2\niterations = 40\nburn_in = 10\n")
        .unwrap();
    ok(camdl(dir, &["fit", "run", "fit.toml"]), "fit run");
    ok(camdl(dir, &["fit", "predict", "--fit", "fit.toml", "--horizon", "free_forward",
        "--n-draws", "4"]), "fit predict");

    let find = |root: &Path, tail: &str| -> PathBuf {
        let mut stack = vec![root.to_path_buf()];
        while let Some(d) = stack.pop() {
            for e in std::fs::read_dir(&d).unwrap() {
                let p = e.unwrap().path();
                if p.is_dir() { stack.push(p) } else if p.to_string_lossy().ends_with(tail) { return p }
            }
        }
        panic!("no {tail} under {}", root.display())
    };
    let text = std::fs::read_to_string(find(&dir.join("results"), "predictive/cases.tsv")).unwrap();
    let mut lines = text.lines();
    let header: Vec<&str> = lines.next().unwrap().split('\t').collect();
    let col = |name: &str| header.iter().position(|c| *c == name).unwrap();
    let (tc, qc) = (col("time"), col("q50"));
    let rows: Vec<(f64, f64)> = lines.map(|l| {
        let f: Vec<&str> = l.split('\t').collect();
        (f[tc].parse().unwrap(), f[qc].parse().unwrap())
    }).collect();
    let times: Vec<f64> = rows.iter().map(|r| r.0).collect();
    assert_eq!(times, vec![0.0, 7.0, 14.0, 21.0, 28.0, 35.0, 42.0],
        "the observed rows under their labels, then the weekly forecast");
    // Every row whose period closes inside the trajectory read above.
    for (t, q50) in rows.into_iter().filter(|(t, _)| *t + 7.0 <= 42.0) {
        let stop = t as usize + 7;
        let expected = mean_at(&i_at, stop);
        assert!((q50 - expected).abs() < 0.01,
            "the row labelled {t} measures [{t}, {stop}) and is drawn from the state \
             and time at its stop: expected I·t = {expected}, got {q50} (at the label \
             it is {})", mean_at(&i_at, t as usize));
    }
}
